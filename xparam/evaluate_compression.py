"""
Compression Evaluation Script for CDC Model on Drone Images
------------------------------------------------------------
Tasks:
  1. Apply the compression model on 100 drone images, report overall compression rate.
  2. Compare size of original vs reconstructed images.

Usage:
  python evaluate_compression.py \
    --ckpt /path/to/checkpoint.pt \
    --img_dir /path/to/drone_imgs \
    --out_dir /path/to/output \
    --lpips_weight 0.9

Output:
  - Reconstructed images saved to --out_dir
  - compression_report.txt  (human-readable summary)
  - compression_results.csv (per-image data)
"""

import argparse
import os
import csv
import time
import pathlib
import torch
import torchvision
import numpy as np
from PIL import Image
from modules.denoising_diffusion import GaussianDiffusion
from modules.unet import Unet
from modules.compress_modules import ResnetCompressor
from ema_pytorch import EMA

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",          type=str,   required=True)
parser.add_argument("--img_dir",       type=str,   default="../imgs")
parser.add_argument("--out_dir",       type=str,   default="../compressed_imgs")
parser.add_argument("--gamma",         type=float, default=0.8)
parser.add_argument("--n_denoise_step",type=int,   default=65)
parser.add_argument("--device",        type=int,   default=0)
parser.add_argument("--lpips_weight",  type=float, required=True)
parser.add_argument("--n_images",      type=int,   default=100,
                    help="Number of images to process (default: 100)")
config = parser.parse_args()

UNCOMPRESSED_BPP = 24.0  # RGB, 8 bits per channel


def load_model(rank):
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
        aux_loss_weight=config.lpips_weight,
        aux_loss_type="lpips",
        var_schedule="cosine",
        use_loss_weight=True,
        loss_weight_min=5,
        use_aux_loss_weight_schedule=False,
    )
    loaded_param = torch.load(config.ckpt, map_location=lambda s, l: s)
    ema = EMA(diffusion, beta=0.999, update_every=10, power=0.75, update_after_step=100)
    ema.load_state_dict(loaded_param["ema"])
    diffusion = ema.ema_model
    diffusion.to(rank)
    diffusion.eval()
    return diffusion


def format_size(bytes_val):
    """Convert bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} GB"


def main():
    rank = config.device
    out_dir = pathlib.Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect images ─────────────────────────────────────────────────────────
    all_imgs = sorted([
        f for f in os.listdir(config.img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    selected = all_imgs[:config.n_images]
    print(f"Found {len(all_imgs)} images, processing first {len(selected)}.")
    print(f"Output directory: {out_dir}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading model...")
    diffusion = load_model(rank)
    print("Model loaded.\n")

    # ── Per-image results ─────────────────────────────────────────────────────
    results = []
    total_orig_bytes = 0
    total_recon_bytes = 0
    total_bpp = 0.0

    for i, img_name in enumerate(selected):
        img_path = os.path.join(config.img_dir, img_name)
        orig_bytes = os.path.getsize(img_path)

        # Load image and crop to multiples of 64 so the compressor and
        # hyperprior stay aligned through the downsample / upsample stack.
        tensor = torchvision.io.read_image(img_path).unsqueeze(0).float().to(rank) / 255.0
        H, W = tensor.shape[-2], tensor.shape[-1]
        H64 = (H // 64) * 64
        W64 = (W // 64) * 64
        tensor = tensor[:, :, :H64, :W64]
        H, W = tensor.shape[-2], tensor.shape[-1]

        t0 = time.time()
        with torch.no_grad():
            compressed, bpp = diffusion.compress(
                tensor * 2.0 - 1.0,
                sample_steps=config.n_denoise_step,
                bpp_return_mean=True,
                init=torch.randn_like(tensor) * config.gamma,
            )
        elapsed = time.time() - t0

        compressed = compressed.clamp(-1, 1) / 2.0 + 0.5
        bpp_val = float(bpp)

        # Save reconstructed image as PNG
        out_path = out_dir / (pathlib.Path(img_name).stem + "_recon.png")
        torchvision.utils.save_image(compressed.cpu(), str(out_path))
        recon_bytes = os.path.getsize(str(out_path))

        # Compression ratio based on BPP (vs uncompressed RGB 24bpp)
        compression_ratio = UNCOMPRESSED_BPP / bpp_val

        total_orig_bytes  += orig_bytes
        total_recon_bytes += recon_bytes
        total_bpp         += bpp_val

        results.append({
            "image":            img_name,
            "width":            W,
            "height":           H,
            "orig_size_bytes":  orig_bytes,
            "recon_size_bytes": recon_bytes,
            "bpp":              round(bpp_val, 4),
            "compression_ratio": round(compression_ratio, 2),
            "time_sec":         round(elapsed, 2),
        })

        print(
            f"[{i+1:3d}/{len(selected)}] {img_name:30s} | "
            f"orig {format_size(orig_bytes):>9} | "
            f"recon {format_size(recon_bytes):>9} | "
            f"bpp {bpp_val:.4f} | "
            f"ratio {compression_ratio:.1f}x | "
            f"{elapsed:.1f}s"
        )

    # ── Aggregate stats ───────────────────────────────────────────────────────
    avg_bpp            = total_bpp / len(results)
    overall_ratio      = UNCOMPRESSED_BPP / avg_bpp
    file_size_ratio    = total_orig_bytes / total_recon_bytes  # orig .jpg vs recon .png

    report_lines = [
        "=" * 65,
        "  CDC COMPRESSION EVALUATION REPORT",
        "=" * 65,
        f"  Images processed     : {len(results)}",
        f"  Model checkpoint     : {config.ckpt}",
        f"  LPIPS weight         : {config.lpips_weight}",
        f"  Denoise steps        : {config.n_denoise_step}",
        f"  Gamma                : {config.gamma}",
        "-" * 65,
        "  TASK 1 — Overall Compression Rate (vs uncompressed RGB)",
        f"    Average BPP        : {avg_bpp:.4f} bits/pixel",
        f"    Uncompressed BPP   : {UNCOMPRESSED_BPP:.1f} bits/pixel (RGB 8-bit)",
        f"    Compression ratio  : {overall_ratio:.2f}x  ({(1 - 1/overall_ratio)*100:.1f}% size reduction)",
        "-" * 65,
        "  TASK 2 — Original vs Reconstructed File Sizes",
        f"    Total original     : {format_size(total_orig_bytes)} ({len(results)} JPEG files)",
        f"    Total reconstructed: {format_size(total_recon_bytes)} ({len(results)} PNG files)",
        f"    File size ratio    : {file_size_ratio:.2f}x  (orig JPEG / recon PNG)",
        "=" * 65,
    ]

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # ── Save report ───────────────────────────────────────────────────────────
    report_path = out_dir / "compression_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text + "\n")
    print(f"\nReport saved to: {report_path}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = out_dir / "compression_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Per-image CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
