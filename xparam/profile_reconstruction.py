"""
profile_reconstruction.py
--------------------------
Profiles the CDC reconstruction (decoding / diffusion) pipeline with detailed
timing breakdown, GPU memory tracking, quality metrics, and optional fp16 support.

Three timing phases are measured for each image:
  1. model_load_sec   -- one-time cost: building + loading weights onto GPU
  2. inference_sec    -- diffusion sampling (the dominant bottleneck)
  3. postproc_sec     -- clamp, rescale, save PNG

Quality metrics computed against the original image:
  - PSNR  (Peak Signal-to-Noise Ratio, dB)
  - SSIM  (Structural Similarity Index)

GPU memory is sampled after inference with torch.cuda.max_memory_allocated().

Usage example:
  python profile_reconstruction.py \
    --ckpt /path/to/b0.2048.pt \
    --img_dir /path/to/drone_imgs \
    --out_dir /path/to/profile_out \
    --n_denoise_step 65 \
    --lpips_weight 0.9 \
    --n_images 10 \
    --fp16

Outputs:
  profile_results.csv   -- per-image timing + quality + memory
  profile_report.txt    -- human-readable summary
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

parser = argparse.ArgumentParser(description="Profile CDC reconstruction pipeline")
parser.add_argument("--ckpt",           type=str,   required=True,  help="Path to model checkpoint (.pt)")
parser.add_argument("--img_dir",        type=str,   required=True,  help="Directory containing input images")
parser.add_argument("--out_dir",        type=str,   required=True,  help="Directory for reconstructed images and reports")
parser.add_argument("--n_denoise_step", type=int,   default=65,     help="Number of diffusion denoising steps")
parser.add_argument("--gamma",          type=float, default=0.8,    help="Noise init scale (gamma)")
parser.add_argument("--lpips_weight",   type=float, required=True,  help="LPIPS auxiliary loss weight used during training")
parser.add_argument("--n_images",       type=int,   default=10,     help="Number of images to profile")
parser.add_argument("--start_index",    type=int,   default=0,      help="Start index within sorted image list")
parser.add_argument("--device",         type=int,   default=0,      help="CUDA device index")
parser.add_argument("--fp16",           action="store_true",        help="Enable fp16 inference via torch.cuda.amp.autocast")
config = parser.parse_args()

UNCOMPRESSED_BPP = 24.0  # 8-bit RGB = 24 bits per pixel


# ── GPU-accurate timer helper ─────────────────────────────────────────────────

class CudaTimer:
    """
    Wraps torch.cuda.Event for accurate GPU-side timing.

    Why not time.time()?
    CUDA operations are asynchronous -- time.time() on the CPU returns before
    GPU kernels actually finish, so it consistently under-reports inference time.
    CUDA Events record timestamps directly on the GPU stream, giving true latency.
    """

    def __init__(self, device_index):
        self.device_index = device_index
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event   = torch.cuda.Event(enable_timing=True)

    def start(self):
        self.start_event.record()

    def stop(self) -> float:
        """Stop timer, sync GPU, and return elapsed time in seconds."""
        self.end_event.record()
        torch.cuda.synchronize(self.device_index)   # block until all GPU work is done
        return self.start_event.elapsed_time(self.end_event) / 1000.0  # ms -> s


def get_cuda_device(device_index: int):
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

def build_and_load_model(ckpt_path: str, device: torch.device, lpips_weight: float) -> GaussianDiffusion:
    """
    Instantiate UNet + ResnetCompressor + GaussianDiffusion, then load EMA weights.
    Architecture is kept identical to evaluate_compression.py so results are comparable.
    """
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

    # Load EMA (exponential moving average) weights -- only the EMA model is saved
    loaded_param = torch.load(ckpt_path, map_location=lambda s, _: s)
    ema = EMA(diffusion, beta=0.999, update_every=10, power=0.75, update_after_step=100)
    ema.load_state_dict(loaded_param["ema"])
    diffusion = ema.ema_model
    diffusion.to(device)
    diffusion.eval()
    return diffusion


# ── Quality metrics ───────────────────────────────────────────────────────────

def compute_psnr_ssim(original: torch.Tensor, reconstructed: torch.Tensor):
    """
    Compute PSNR and SSIM between two [1, C, H, W] float tensors in [0, 1] range.
    Converts to uint8 numpy arrays as required by skimage.

    Returns:
        psnr_val (float): PSNR in dB -- higher is better (>30 dB = good quality)
        ssim_val (float): SSIM in [0, 1] -- higher is better (>0.9 = visually similar)
    """
    orig_np  = (original.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    recon_np = (reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    psnr_val = psnr_fn(orig_np, recon_np, data_range=255)
    # channel_axis=2 tells skimage to treat dim 2 as colour channels (multi-channel SSIM)
    ssim_val = ssim_fn(orig_np, recon_np, data_range=255, channel_axis=2)
    return float(psnr_val), float(ssim_val)


# ── Main profiling loop ───────────────────────────────────────────────────────

def main():
    device, device_index = get_cuda_device(config.device)
    out_dir = pathlib.Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    precision_label = "fp16" if config.fp16 else "fp32"
    print(f"Profiling CDC reconstruction -- {config.n_denoise_step} steps | {precision_label}")
    print(f"Checkpoint : {config.ckpt}")
    print(f"Output dir : {out_dir}\n")

    # ── Phase 1: Model loading (timed once, not per-image) ───────────────────
    torch.cuda.reset_peak_memory_stats(device_index)
    t_load_start = time.time()
    diffusion = build_and_load_model(config.ckpt, device, config.lpips_weight)
    torch.cuda.synchronize(device_index)        # ensure all weight transfers finish
    model_load_sec = time.time() - t_load_start
    mem_after_load_mb = torch.cuda.memory_allocated(device_index) / 1024 ** 2

    print(f"Model loaded in {model_load_sec:.2f}s  |  GPU memory after load: {mem_after_load_mb:.1f} MB\n")

    # ── Collect images ────────────────────────────────────────────────────────
    all_imgs = sorted([
        f for f in os.listdir(config.img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    selected = all_imgs[config.start_index : config.start_index + config.n_images]
    print(f"Processing {len(selected)} images "
          f"(index {config.start_index} to {config.start_index + len(selected) - 1})\n")

    results = []
    timer = CudaTimer(device_index)

    for i, img_name in enumerate(selected):
        img_path = os.path.join(config.img_dir, img_name)

        # Crop to multiples of 64 -- the compressor / hyperprior require this
        t_data_start = time.perf_counter()
        tensor = torchvision.io.read_image(img_path).unsqueeze(0).float().to(device) / 255.0
        H, W = tensor.shape[-2], tensor.shape[-1]
        H64, W64 = (H // 64) * 64, (W // 64) * 64
        tensor = tensor[:, :, :H64, :W64]
        orig_tensor = tensor.clone()            # save unmodified copy for PSNR/SSIM
        torch.cuda.synchronize(device_index)
        data_load_sec = time.perf_counter() - t_data_start

        torch.cuda.reset_peak_memory_stats(device_index)  # reset before each inference

        # ── Phase 2: Diffusion inference (the bottleneck) ────────────────────
        timer.start()
        with torch.no_grad():
            # autocast enables fp16 mixed precision on Tensor Core GPUs.
            # When fp16=False, autocast is a no-op (no overhead).
            with amp.autocast(enabled=config.fp16):
                compressed, bpp = diffusion.compress(
                    tensor * 2.0 - 1.0,             # scale [0,1] -> [-1,1]
                    sample_steps=config.n_denoise_step,
                    bpp_return_mean=True,
                    init=torch.randn_like(tensor) * config.gamma,
                )
        inference_sec = timer.stop()
        peak_mem_mb = torch.cuda.max_memory_allocated(device_index) / 1024 ** 2

        # ── Phase 3: Post-processing (clamp + rescale + PNG save) ────────────
        # Wall-clock timing is used here because image saving is CPU/filesystem work,
        # which CUDA Events do not measure correctly.
        torch.cuda.synchronize(device_index)
        t_post_start = time.perf_counter()
        compressed_01 = compressed.clamp(-1, 1) / 2.0 + 0.5   # back to [0, 1]
        out_path = out_dir / (pathlib.Path(img_name).stem + "_recon.png")
        torchvision.utils.save_image(compressed_01.cpu(), str(out_path))
        postproc_sec = time.perf_counter() - t_post_start

        # ── Quality metrics ───────────────────────────────────────────────────
        psnr_val, ssim_val = compute_psnr_ssim(orig_tensor, compressed_01)

        bpp_val           = float(bpp)
        compression_ratio = UNCOMPRESSED_BPP / bpp_val
        orig_bytes        = os.path.getsize(img_path)
        recon_bytes       = os.path.getsize(str(out_path))
        total_sec         = data_load_sec + inference_sec + postproc_sec   # model load excluded

        results.append({
            "image":             img_name,
            "width":             W64,
            "height":            H64,
            "n_denoise_step":    config.n_denoise_step,
            "precision":         precision_label,
            "model_load_sec":    round(model_load_sec, 3),  # same value in every row
            "data_load_sec":     round(data_load_sec, 3),
            "inference_sec":     round(inference_sec, 3),
            "postproc_sec":      round(postproc_sec, 3),
            "total_sec":         round(total_sec, 3),
            "peak_gpu_mem_mb":   round(peak_mem_mb, 1),
            "bpp":               round(bpp_val, 4),
            "compression_ratio": round(compression_ratio, 2),
            "psnr_db":           round(psnr_val, 2),
            "ssim":              round(ssim_val, 4),
            "orig_size_bytes":   orig_bytes,
            "recon_size_bytes":  recon_bytes,
        })

        print(
            f"[{i+1:3d}/{len(selected)}] {img_name:30s} | "
            f"load {data_load_sec:5.2f}s | "
            f"infer {inference_sec:6.2f}s | "
            f"post {postproc_sec:5.2f}s | "
            f"mem {peak_mem_mb:7.1f} MB | "
            f"PSNR {psnr_val:5.2f} dB | "
            f"SSIM {ssim_val:.4f} | "
            f"bpp {bpp_val:.4f}"
        )

    # ── Aggregate and print summary ───────────────────────────────────────────
    avg_inference = np.mean([r["inference_sec"]    for r in results])
    avg_data_load = np.mean([r["data_load_sec"]    for r in results])
    avg_postproc  = np.mean([r["postproc_sec"]     for r in results])
    avg_total     = np.mean([r["total_sec"]        for r in results])
    avg_mem       = np.mean([r["peak_gpu_mem_mb"]  for r in results])
    avg_psnr      = np.mean([r["psnr_db"]          for r in results])
    avg_ssim      = np.mean([r["ssim"]             for r in results])
    avg_bpp       = np.mean([r["bpp"]              for r in results])

    report_lines = [
        "=" * 68,
        "  CDC RECONSTRUCTION PROFILING REPORT",
        "=" * 68,
        f"  Checkpoint        : {config.ckpt}",
        f"  Images profiled   : {len(results)}",
        f"  Denoising steps   : {config.n_denoise_step}",
        f"  Precision         : {precision_label}",
        f"  Gamma             : {config.gamma}",
        "-" * 68,
        "  TIMING BREAKDOWN (per-image averages)",
        f"    Model load time  : {model_load_sec:.2f}s  (one-time cost, not per-image)",
        f"    Data load/preproc : {avg_data_load:.2f}s",
        f"    Inference time   : {avg_inference:.2f}s  <-- dominant bottleneck",
        f"    Post-processing  : {avg_postproc:.2f}s",
        f"    Total per image  : {avg_total:.2f}s  (data load + inference + post)",
        f"    Inference share  : {avg_inference / avg_total * 100:.1f}% of total runtime",
        "-" * 68,
        "  GPU MEMORY",
        f"    After model load : {mem_after_load_mb:.1f} MB",
        f"    Peak during infer: {avg_mem:.1f} MB (average across images)",
        "-" * 68,
        "  QUALITY METRICS",
        f"    Average PSNR     : {avg_psnr:.2f} dB",
        f"    Average SSIM     : {avg_ssim:.4f}",
        f"    Average BPP      : {avg_bpp:.4f}",
        f"    Compression ratio: {UNCOMPRESSED_BPP / avg_bpp:.2f}x vs uncompressed RGB",
        "=" * 68,
    ]
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    report_path = out_dir / "profile_report.txt"
    report_path.write_text(report_text + "\n")
    print(f"\nReport saved  -> {report_path}")

    csv_path = out_dir / "profile_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV saved     -> {csv_path}")


if __name__ == "__main__":
    main()
