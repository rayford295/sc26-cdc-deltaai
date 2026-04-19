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

For each configuration, the script records:
  - Average inference time per image
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
parser.add_argument("--device",       type=int,   default=0,      help="CUDA device index")
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


# ── CUDA timer ────────────────────────────────────────────────────────────────

class CudaTimer:
    """
    GPU-accurate timer using CUDA Events instead of time.time().
    CUDA ops are async -- time.time() returns before GPU kernels finish,
    causing systematic under-reporting of inference latency.
    """

    def __init__(self, device):
        self.device = device
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event   = torch.cuda.Event(enable_timing=True)

    def start(self):
        self.start_event.record()

    def stop(self) -> float:
        """Sync GPU and return elapsed seconds."""
        self.end_event.record()
        torch.cuda.synchronize(self.device)
        return self.start_event.elapsed_time(self.end_event) / 1000.0  # ms -> s


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

def run_config(diffusion, images, device, n_steps, use_fp16, gamma, out_subdir):
    """
    Run inference on pre-loaded images for one (n_steps, precision) configuration.

    Args:
        diffusion   : already-loaded GaussianDiffusion model (shared across configs)
        images      : list of (img_name, tensor, orig_tensor, orig_bytes)
        n_steps     : number of denoising steps
        use_fp16    : whether to enable autocast fp16
        out_subdir  : directory to save reconstructed PNGs for this config

    Returns list of per-image result dicts.
    """
    precision_label = "fp16" if use_fp16 else "fp32"
    out_subdir.mkdir(parents=True, exist_ok=True)
    timer = CudaTimer(device)
    results = []

    for img_name, tensor, orig_tensor, orig_bytes in images:
        torch.cuda.reset_peak_memory_stats(device)

        # Diffusion inference -- the bottleneck we are characterising
        timer.start()
        with torch.no_grad():
            with amp.autocast(enabled=use_fp16):
                compressed, bpp = diffusion.compress(
                    tensor * 2.0 - 1.0,
                    sample_steps=n_steps,
                    bpp_return_mean=True,
                    init=torch.randn_like(tensor) * gamma,
                )
        inference_sec = timer.stop()
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2

        compressed_01 = compressed.clamp(-1, 1) / 2.0 + 0.5
        out_path = out_subdir / (pathlib.Path(img_name).stem + "_recon.png")
        torchvision.utils.save_image(compressed_01.cpu(), str(out_path))

        psnr_val, ssim_val = compute_psnr_ssim(orig_tensor, compressed_01)
        bpp_val = float(bpp)

        results.append({
            "n_denoise_step":    n_steps,
            "precision":         precision_label,
            "image":             img_name,
            "inference_sec":     round(inference_sec, 3),
            "peak_gpu_mem_mb":   round(peak_mem_mb, 1),
            "bpp":               round(bpp_val, 4),
            "compression_ratio": round(UNCOMPRESSED_BPP / bpp_val, 2),
            "psnr_db":           round(psnr_val, 2),
            "ssim":              round(ssim_val, 4),
            "orig_size_bytes":   orig_bytes,
            "recon_size_bytes":  os.path.getsize(str(out_path)),
        })

        print(
            f"  steps={n_steps:3d} | {precision_label} | {img_name:25s} | "
            f"infer {inference_sec:6.2f}s | "
            f"PSNR {psnr_val:5.2f} dB | SSIM {ssim_val:.4f} | "
            f"mem {peak_mem_mb:.0f} MB"
        )

    return results


# ── Main sweep ────────────────────────────────────────────────────────────────

def main():
    device = torch.device(f"cuda:{config.device}")
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
    print(f"  Images/config: {config.n_images}")
    print("=" * 68 + "\n")

    # Load model ONCE -- the key optimisation: avoid reloading for each config
    print("Loading model (shared across all sweep configurations)...")
    t0 = time.time()
    diffusion = build_and_load_model(config.ckpt, device, config.lpips_weight)
    torch.cuda.synchronize(device)
    print(f"Model loaded in {time.time() - t0:.2f}s\n")

    # Pre-load images into GPU memory to eliminate disk I/O from timing
    all_imgs = sorted([
        f for f in os.listdir(config.img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    selected = all_imgs[: config.n_images]
    print(f"Pre-loading {len(selected)} images...\n")

    images = []
    for img_name in selected:
        img_path = os.path.join(config.img_dir, img_name)
        tensor = torchvision.io.read_image(img_path).unsqueeze(0).float().to(device) / 255.0
        H, W = tensor.shape[-2], tensor.shape[-1]
        H64, W64 = (H // 64) * 64, (W // 64) * 64
        tensor = tensor[:, :, :H64, :W64]
        images.append((img_name, tensor, tensor.clone(), os.path.getsize(img_path)))

    # Run the sweep
    all_results = []
    for use_fp16 in precision_variants:
        for n_steps in config.steps:
            prec_label = "fp16" if use_fp16 else "fp32"
            print(f"\n--- steps={n_steps} | {prec_label} ---")
            subdir = out_dir / f"steps{n_steps}_{prec_label}"
            run_results = run_config(diffusion, images, device, n_steps, use_fp16, config.gamma, subdir)
            all_results.extend(run_results)

    # Build summary table (one row per configuration)
    print("\n\n" + "=" * 68)
    print("  SWEEP SUMMARY  (averages per configuration)")
    print("=" * 68)
    print(f"  {'Steps':>5}  {'Prec':>5}  {'Infer(s)':>8}  {'Mem(MB)':>7}  {'PSNR(dB)':>8}  {'SSIM':>6}  {'BPP':>6}")
    print("-" * 68)

    configs_seen = []
    for r in all_results:
        key = (r["n_denoise_step"], r["precision"])
        if key not in configs_seen:
            configs_seen.append(key)

    summary_rows = []
    for n_steps, prec in configs_seen:
        group = [r for r in all_results if r["n_denoise_step"] == n_steps and r["precision"] == prec]
        row = {
            "n_denoise_step":      n_steps,
            "precision":           prec,
            "avg_inference_sec":   round(np.mean([r["inference_sec"]   for r in group]), 3),
            "avg_peak_gpu_mem_mb": round(np.mean([r["peak_gpu_mem_mb"] for r in group]), 1),
            "avg_psnr_db":         round(np.mean([r["psnr_db"]         for r in group]), 2),
            "avg_ssim":            round(np.mean([r["ssim"]            for r in group]), 4),
            "avg_bpp":             round(np.mean([r["bpp"]             for r in group]), 4),
            "n_images":            len(group),
        }
        summary_rows.append(row)
        print(
            f"  {n_steps:>5}  {prec:>5}  "
            f"{row['avg_inference_sec']:>8.2f}  "
            f"{row['avg_peak_gpu_mem_mb']:>7.1f}  "
            f"{row['avg_psnr_db']:>8.2f}  "
            f"{row['avg_ssim']:>6.4f}  "
            f"{row['avg_bpp']:>6.4f}"
        )
    print("=" * 68)

    # Save CSVs
    csv_path = out_dir / "sweep_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nPer-image CSV  -> {csv_path}")

    summary_csv = out_dir / "sweep_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary CSV    -> {summary_csv}")
    print(f"\nRun: python plot_results.py --sweep_csv {csv_path}")


if __name__ == "__main__":
    main()
