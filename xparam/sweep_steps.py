"""
sweep_steps.py
--------------
Runs parameter sweep experiments over different denoising step counts
(and optionally precision settings) to find the best speed/quality trade-off.

The model is loaded ONCE and reused across all step configurations,
avoiding the one-time model-loading overhead from polluting timing results.

Sweep dimensions:
  - n_denoise_step : [5, 10, 20, 30, 50, 65, 100]   (most important axis)
  - precision      : fp32 only, or fp32 + fp16        (via --test_fp16 flag)
  - batch size     : one or more candidate inference batch sizes
  - repeats        : repeated runs for stable averages

For each configuration, the script records:
  - Average inference time per image and images/hour
  - Peak GPU memory
  - Average PSNR and SSIM (quality)
  - Average BPP

Results are written to sweep_results.csv and sweep_summary.csv,
which can be consumed directly by plot_results.py for visualization.

Usage example:
  python sweep_steps.py \
    --ckpt /path/to/b0.2048.pt \
    --img_dir /path/to/drone_imgs \
    --out_dir /path/to/sweep_out \
    --lpips_weight 0.9 \
    --n_images 5 \
    --batch_sizes 1 2 \
    --repeats 3 \
    --test_fp16

Tip: use --n_images 5 for a quick sanity check, 20+ for reliable averages.
"""

import argparse
import csv
import os
import pathlib
import time

import numpy as np
import torch
import torch.cuda.amp as amp
import torchvision
from ema_pytorch import EMA
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

from modules.compress_modules import ResnetCompressor
from modules.denoising_diffusion import GaussianDiffusion
from modules.unet import Unet

# ── Argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Sweep denoising steps for speed/quality trade-off")
parser.add_argument("--ckpt",         type=str,   required=True,  help="Path to model checkpoint (.pt)")
parser.add_argument("--img_dir",      type=str,   required=True,  help="Directory with input images")
parser.add_argument("--out_dir",      type=str,   required=True,  help="Output directory for CSV and per-run images")
parser.add_argument("--lpips_weight", type=float, required=True,  help="LPIPS weight matching the checkpoint")
parser.add_argument("--gamma",        type=float, default=0.8,    help="Noise init scale")
parser.add_argument("--n_images",     type=int,   default=5,      help="Images per configuration (5 for quick test, 20+ for reliable stats)")
parser.add_argument("--start_index",  type=int,   default=0,      help="Start index within sorted image list")
parser.add_argument("--device",       type=int,   default=0,      help="CUDA device index")
parser.add_argument("--batch_size",   type=int,   default=None,   help="Single inference batch size (legacy alias)")
parser.add_argument("--batch_sizes",  type=int,   nargs="+",      default=None, help="One or more inference batch sizes to test")
parser.add_argument("--repeats",      type=int,   default=1,      help="Repeat each configuration this many times")
parser.add_argument("--test_fp16",    action="store_true",        help="Also benchmark fp16 for each step count")
parser.add_argument(
    "--steps",
    type=int,
    nargs="+",
    default=[5, 10, 20, 30, 50, 65, 100],
    help="List of denoising step counts to sweep (default: 5 10 20 30 50 65 100)",
)
config = parser.parse_args()

UNCOMPRESSED_BPP = 24.0

if config.batch_sizes is not None:
    batch_size_values = config.batch_sizes
elif config.batch_size is not None:
    batch_size_values = [config.batch_size]
else:
    batch_size_values = [1]
config.batch_sizes = sorted(dict.fromkeys(batch_size_values))

if any(bs < 1 for bs in config.batch_sizes):
    raise ValueError("--batch_size/--batch_sizes must be positive")
if config.repeats < 1:
    raise ValueError("--repeats must be >= 1")

RESULT_FIELDS = [
    "status",
    "repeat",
    "batch_index",
    "batch_size",
    "effective_batch_size",
    "n_denoise_step",
    "precision",
    "image",
    "inference_sec",
    "batch_inference_sec",
    "postproc_sec",
    "batch_postproc_sec",
    "total_sec",
    "images_per_hour",
    "peak_gpu_mem_mb",
    "bpp",
    "compression_ratio",
    "psnr_db",
    "ssim",
    "orig_size_bytes",
    "recon_size_bytes",
    "error",
]

SUMMARY_FIELDS = [
    "n_denoise_step",
    "precision",
    "batch_size",
    "avg_inference_sec",
    "avg_images_per_hour",
    "avg_peak_gpu_mem_mb",
    "avg_psnr_db",
    "avg_ssim",
    "avg_bpp",
    "n_images",
]


# ── CUDA timer ────────────────────────────────────────────────────────────────

class CudaTimer:
    """
    GPU-accurate timer using CUDA Events instead of time.time().
    CUDA ops are async -- time.time() returns before GPU kernels finish,
    causing systematic under-reporting of inference latency.
    """

    def __init__(self, device_index):
        self.device_index = device_index
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event   = torch.cuda.Event(enable_timing=True)

    def start(self):
        self.start_event.record()

    def stop(self) -> float:
        """Sync GPU and return elapsed seconds."""
        self.end_event.record()
        torch.cuda.synchronize(self.device_index)
        return self.start_event.elapsed_time(self.end_event) / 1000.0  # ms -> s


def get_cuda_device(device_index):
    """Select a CUDA device and return both torch.device and the integer index."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this script inside a GPU allocation.")
    visible_devices = torch.cuda.device_count()
    if device_index < 0 or device_index >= visible_devices:
        raise RuntimeError(
            f"Requested CUDA device {device_index}, but PyTorch sees {visible_devices} visible device(s)."
        )
    torch.cuda.set_device(device_index)
    return torch.device("cuda", device_index), device_index


# ── Model builder ─────────────────────────────────────────────────────────────

def build_and_load_model(ckpt_path, device, lpips_weight):
    """Build model architecture and load EMA weights from checkpoint."""
    denoise_model = Unet(
        dim=64,
        channels=3,
        context_channels=64,
        dim_mults=[1, 2, 3, 4, 5, 6],
        context_dim_mults=[1, 2, 3, 4],
        embd_type="01",
    )
    context_model = ResnetCompressor(
        dim=64,
        dim_mults=[1, 2, 3, 4],
        reverse_dim_mults=[4, 3, 2, 1],
        hyper_dims_mults=[4, 4, 4],
        channels=3,
        out_channels=64,
    )
    diffusion = GaussianDiffusion(
        denoise_fn=denoise_model,
        context_fn=context_model,
        ae_fn=None,
        num_timesteps=8193,
        loss_type="l2",
        lagrangian=0.0032,
        pred_mode="x",
        aux_loss_weight=lpips_weight,
        aux_loss_type="lpips",
        var_schedule="cosine",
        use_loss_weight=True,
        loss_weight_min=5,
        use_aux_loss_weight_schedule=False,
    )
    loaded_param = torch.load(ckpt_path, map_location=lambda s, _: s)
    ema = EMA(diffusion, beta=0.999, update_every=10, power=0.75, update_after_step=100)
    ema.load_state_dict(loaded_param["ema"])
    diffusion = ema.ema_model
    diffusion.to(device)
    diffusion.eval()
    return diffusion


# ── Quality metrics ───────────────────────────────────────────────────────────

def compute_psnr_ssim(original, reconstructed):
    """PSNR (dB) and SSIM for two [1,C,H,W] float tensors in [0,1]."""
    orig_np  = (original.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    recon_np = (reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    psnr_val = psnr_fn(orig_np, recon_np, data_range=255)
    ssim_val = ssim_fn(orig_np, recon_np, data_range=255, channel_axis=2)
    return float(psnr_val), float(ssim_val)


# ── Single configuration run ──────────────────────────────────────────────────

def is_cuda_oom(exc):
    """Return True when a RuntimeError looks like a CUDA out-of-memory failure."""
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def failure_row(n_steps, precision_label, batch_size, error_text):
    """Create a CSV-compatible row for a failed batch-size configuration."""
    row = {field: "" for field in RESULT_FIELDS}
    row.update({
        "status": "failed",
        "batch_size": batch_size,
        "n_denoise_step": n_steps,
        "precision": precision_label,
        "error": error_text.replace("\n", " ")[:500],
    })
    return row


def run_config(diffusion, images, device, device_index, n_steps, use_fp16, gamma, out_subdir, batch_size, repeats):
    """
    Run inference on pre-loaded CPU images for one configuration.

    Args:
        diffusion   : already-loaded GaussianDiffusion model (shared across configs)
        images      : list of (img_name, tensor, orig_tensor, orig_bytes)
        n_steps     : number of denoising steps
        use_fp16    : whether to enable autocast fp16
        out_subdir  : directory to save reconstructed PNGs for this config
        batch_size  : number of images per inference batch
        repeats     : repeated runs per configuration

    Returns list of per-image result dicts.
    """
    precision_label = "fp16" if use_fp16 else "fp32"
    out_subdir.mkdir(parents=True, exist_ok=True)
    timer = CudaTimer(device_index)
    results = []

    for repeat_idx in range(1, repeats + 1):
        for batch_start in range(0, len(images), batch_size):
            batch_items = images[batch_start : batch_start + batch_size]
            effective_batch_size = len(batch_items)
            batch_index = batch_start // batch_size

            batch_tensor = torch.cat([item[1] for item in batch_items], dim=0).to(device)
            orig_batch = torch.cat([item[2] for item in batch_items], dim=0).to(device)
            batch_names = [item[0] for item in batch_items]
            batch_orig_bytes = [item[3] for item in batch_items]

            torch.cuda.reset_peak_memory_stats(device_index)

            # Diffusion inference -- the bottleneck we are characterising.
            timer.start()
            with torch.no_grad():
                with amp.autocast(enabled=use_fp16):
                    compressed, bpp = diffusion.compress(
                        batch_tensor * 2.0 - 1.0,
                        sample_steps=n_steps,
                        bpp_return_mean=False,
                        init=torch.randn_like(batch_tensor) * gamma,
                    )
            batch_inference_sec = timer.stop()
            peak_mem_mb = torch.cuda.max_memory_allocated(device_index) / 1024 ** 2

            # Wall-clock timing is used for post-processing because PNG writing
            # and quality metric preparation are CPU/filesystem work.
            torch.cuda.synchronize(device_index)
            t_post_start = time.perf_counter()
            compressed_01 = compressed.clamp(-1, 1) / 2.0 + 0.5
            save_dir = out_subdir / f"repeat{repeat_idx:02d}" if repeats > 1 else out_subdir
            save_dir.mkdir(parents=True, exist_ok=True)
            out_paths = []
            for img_name, recon in zip(batch_names, compressed_01):
                out_path = save_dir / (pathlib.Path(img_name).stem + "_recon.png")
                torchvision.utils.save_image(recon.unsqueeze(0).cpu(), str(out_path))
                out_paths.append(out_path)
            batch_postproc_sec = time.perf_counter() - t_post_start

            bpp_values = bpp.detach().float().cpu().reshape(-1).tolist()
            psnr_values = []
            ssim_values = []
            for item_idx in range(effective_batch_size):
                psnr_val, ssim_val = compute_psnr_ssim(
                    orig_batch[item_idx:item_idx + 1],
                    compressed_01[item_idx:item_idx + 1],
                )
                psnr_values.append(psnr_val)
                ssim_values.append(ssim_val)

            inference_per_image = batch_inference_sec / effective_batch_size
            postproc_per_image = batch_postproc_sec / effective_batch_size
            images_per_hour = 3600.0 * effective_batch_size / batch_inference_sec

            for item_idx, img_name in enumerate(batch_names):
                bpp_val = float(bpp_values[item_idx])
                results.append({
                    "status":              "success",
                    "repeat":              repeat_idx,
                    "batch_index":         batch_index,
                    "batch_size":          batch_size,
                    "effective_batch_size": effective_batch_size,
                    "n_denoise_step":      n_steps,
                    "precision":           precision_label,
                    "image":               img_name,
                    "inference_sec":       round(inference_per_image, 3),
                    "batch_inference_sec": round(batch_inference_sec, 3),
                    "postproc_sec":        round(postproc_per_image, 3),
                    "batch_postproc_sec":  round(batch_postproc_sec, 3),
                    "total_sec":           round(inference_per_image + postproc_per_image, 3),
                    "images_per_hour":     round(images_per_hour, 2),
                    "peak_gpu_mem_mb":     round(peak_mem_mb, 1),
                    "bpp":                 round(bpp_val, 4),
                    "compression_ratio":   round(UNCOMPRESSED_BPP / bpp_val, 2),
                    "psnr_db":             round(psnr_values[item_idx], 2),
                    "ssim":                round(ssim_values[item_idx], 4),
                    "orig_size_bytes":     batch_orig_bytes[item_idx],
                    "recon_size_bytes":    os.path.getsize(str(out_paths[item_idx])),
                    "error":               "",
                })

            print(
                f"  repeat={repeat_idx:02d} | steps={n_steps:3d} | {precision_label} | "
                f"batch={batch_size} ({effective_batch_size} imgs) | "
                f"infer {batch_inference_sec:7.2f}s "
                f"({inference_per_image:6.2f}s/img, {images_per_hour:5.1f} img/hr) | "
                f"PSNR {np.mean(psnr_values):5.2f} dB | "
                f"SSIM {np.mean(ssim_values):.4f} | "
                f"mem {peak_mem_mb:.0f} MB"
            )

            del batch_tensor, orig_batch, compressed, compressed_01

    return results


# ── Main sweep ────────────────────────────────────────────────────────────────

def main():
    device, device_index = get_cuda_device(config.device)
    out_dir = pathlib.Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    precision_variants = [False]           # fp32 always included
    if config.test_fp16:
        precision_variants.append(True)    # additionally test fp16

    print("=" * 68)
    print("  CDC RECONSTRUCTION -- PARAMETER SWEEP")
    print("=" * 68)
    print(f"  Checkpoint   : {config.ckpt}")
    print(f"  Steps to test: {config.steps}")
    print(f"  Precisions   : {'fp32 + fp16' if config.test_fp16 else 'fp32 only'}")
    print(f"  Batch sizes  : {config.batch_sizes}")
    print(f"  Repeats      : {config.repeats}")
    print(f"  Images/config: {config.n_images}")
    print("=" * 68 + "\n")

    # Load model ONCE -- the key optimisation: avoid reloading for each config
    print("Loading model (shared across all sweep configurations)...")
    t0 = time.time()
    diffusion = build_and_load_model(config.ckpt, device, config.lpips_weight)
    torch.cuda.synchronize(device_index)
    print(f"Model loaded in {time.time() - t0:.2f}s\n")

    # Pre-load images into GPU memory to eliminate disk I/O from timing
    all_imgs = sorted([
        f for f in os.listdir(config.img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    selected = all_imgs[config.start_index : config.start_index + config.n_images]
    if not selected:
        raise ValueError("No input images selected. Check --img_dir, --start_index, and --n_images.")
    print(f"Pre-loading {len(selected)} images...\n")

    raw_images = []
    min_h64 = None
    min_w64 = None
    for img_name in selected:
        img_path = os.path.join(config.img_dir, img_name)
        tensor = torchvision.io.read_image(img_path).unsqueeze(0).float() / 255.0
        H, W = tensor.shape[-2], tensor.shape[-1]
        H64, W64 = (H // 64) * 64, (W // 64) * 64
        min_h64 = H64 if min_h64 is None else min(min_h64, H64)
        min_w64 = W64 if min_w64 is None else min(min_w64, W64)
        raw_images.append((img_name, tensor, os.path.getsize(img_path)))

    images = []
    for img_name, tensor, orig_bytes in raw_images:
        tensor = tensor[:, :, :min_h64, :min_w64]
        images.append((img_name, tensor, tensor.clone(), orig_bytes))

    print(f"Using common cropped size: {min_w64} x {min_h64} pixels")

    # Run the sweep
    all_results = []
    for use_fp16 in precision_variants:
        for n_steps in config.steps:
            for batch_size in config.batch_sizes:
                prec_label = "fp16" if use_fp16 else "fp32"
                print(f"\n--- steps={n_steps} | {prec_label} | batch={batch_size} ---")
                subdir = out_dir / f"steps{n_steps}_{prec_label}_batch{batch_size}"
                try:
                    run_results = run_config(
                        diffusion,
                        images,
                        device,
                        device_index,
                        n_steps,
                        use_fp16,
                        config.gamma,
                        subdir,
                        batch_size,
                        config.repeats,
                    )
                    all_results.extend(run_results)
                except RuntimeError as exc:
                    if not is_cuda_oom(exc):
                        raise
                    torch.cuda.empty_cache()
                    error_text = str(exc)
                    print(f"  CUDA OOM for steps={n_steps}, precision={prec_label}, batch={batch_size}. Continuing.")
                    all_results.append(failure_row(n_steps, prec_label, batch_size, error_text))

    # Build summary table (one row per configuration)
    print("\n\n" + "=" * 68)
    print("  SWEEP SUMMARY  (averages per configuration)")
    print("=" * 68)
    print(f"  {'Steps':>5}  {'Prec':>5}  {'Batch':>5}  {'Infer(s)':>8}  {'Img/hr':>7}  {'Mem(MB)':>7}  {'PSNR(dB)':>8}  {'SSIM':>6}  {'BPP':>6}  {'N':>3}")
    print("-" * 68)

    success_results = [r for r in all_results if r.get("status") == "success"]
    configs_seen = []
    for r in success_results:
        key = (r["n_denoise_step"], r["precision"], r["batch_size"])
        if key not in configs_seen:
            configs_seen.append(key)

    summary_rows = []
    for n_steps, prec, batch_size in configs_seen:
        group = [
            r for r in success_results
            if r["n_denoise_step"] == n_steps
            and r["precision"] == prec
            and r["batch_size"] == batch_size
        ]
        row = {
            "n_denoise_step":      n_steps,
            "precision":           prec,
            "batch_size":          batch_size,
            "avg_inference_sec":   round(np.mean([r["inference_sec"]   for r in group]), 3),
            "avg_images_per_hour": round(np.mean([r["images_per_hour"] for r in group]), 2),
            "avg_peak_gpu_mem_mb": round(np.mean([r["peak_gpu_mem_mb"] for r in group]), 1),
            "avg_psnr_db":         round(np.mean([r["psnr_db"]         for r in group]), 2),
            "avg_ssim":            round(np.mean([r["ssim"]            for r in group]), 4),
            "avg_bpp":             round(np.mean([r["bpp"]             for r in group]), 4),
            "n_images":            len(group),
        }
        summary_rows.append(row)
        print(
            f"  {n_steps:>5}  {prec:>5}  {batch_size:>5}  "
            f"{row['avg_inference_sec']:>8.2f}  "
            f"{row['avg_images_per_hour']:>7.1f}  "
            f"{row['avg_peak_gpu_mem_mb']:>7.1f}  "
            f"{row['avg_psnr_db']:>8.2f}  "
            f"{row['avg_ssim']:>6.4f}  "
            f"{row['avg_bpp']:>6.4f}  "
            f"{row['n_images']:>3}"
        )
    print("=" * 68)

    # Save CSVs
    csv_path = out_dir / "sweep_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nPer-image CSV  -> {csv_path}")

    summary_csv = out_dir / "sweep_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary CSV    -> {summary_csv}")
    print(f"\nRun: python plot_results.py --sweep_csv {csv_path}")


if __name__ == "__main__":
    main()
