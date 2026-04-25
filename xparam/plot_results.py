"""
plot_results.py
---------------
Reads sweep_results.csv (produced by sweep_steps.py) and generates five plots:

  1. Inference time vs denoising steps
  2. PSNR vs denoising steps
  3. SSIM vs denoising steps
  4. Quality vs speed trade-off (PSNR on y-axis, inference time on x-axis)
  5. GPU memory vs denoising steps

Each plot is saved as a PNG and the script also prints a summary table.

Usage:
  python plot_results.py \
    --sweep_csv /path/to/sweep_out/sweep_results.csv \
    --out_dir   /path/to/plots

Dependencies:
  pip install matplotlib pandas
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── Arguments ─────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Plot CDC reconstruction sweep results")
parser.add_argument("--sweep_csv", type=str, required=True, help="Path to sweep_results.csv from sweep_steps.py")
parser.add_argument("--out_dir",   type=str, default=None,  help="Directory to save plot PNGs (default: same folder as CSV)")
args = parser.parse_args()

csv_path = pathlib.Path(args.sweep_csv)
out_dir  = pathlib.Path(args.out_dir) if args.out_dir else csv_path.parent
out_dir.mkdir(parents=True, exist_ok=True)

# ── Load and aggregate data ───────────────────────────────────────────────────

df = pd.read_csv(csv_path)

if "status" in df.columns:
    df = df[df["status"].fillna("success") == "success"].copy()
if "batch_size" not in df.columns:
    df["batch_size"] = 1
if "images_per_hour" not in df.columns:
    df["images_per_hour"] = 3600.0 / df["inference_sec"]
if df.empty:
    raise ValueError("No successful sweep rows found in the input CSV.")

# Compute per-configuration averages (group by steps + precision + batch size)
summary = (
    df.groupby(["n_denoise_step", "precision", "batch_size"])
    .agg(
        avg_inference_sec  = ("inference_sec",   "mean"),
        std_inference_sec  = ("inference_sec",   "std"),
        avg_images_per_hour = ("images_per_hour", "mean"),
        avg_psnr_db        = ("psnr_db",         "mean"),
        std_psnr_db        = ("psnr_db",         "std"),
        avg_ssim           = ("ssim",            "mean"),
        std_ssim           = ("ssim",            "std"),
        avg_peak_mem_mb    = ("peak_gpu_mem_mb", "mean"),
        avg_bpp            = ("bpp",             "mean"),
        n_images           = ("image",           "count"),
    )
    .reset_index()
    .sort_values(["precision", "batch_size", "n_denoise_step"])
)

# Print summary table to terminal
print("\n" + "=" * 80)
print("  SWEEP SUMMARY")
print("=" * 80)
print(f"  {'Steps':>5}  {'Prec':>5}  {'Batch':>5}  {'Infer(s)':>9}  {'Img/hr':>7}  {'Mem(MB)':>7}  "
      f"{'PSNR(dB)':>8}  {'SSIM':>6}  {'BPP':>6}  {'N':>3}")
print("-" * 80)
for _, row in summary.iterrows():
    print(
        f"  {int(row.n_denoise_step):>5}  {row.precision:>5}  "
        f"{int(row.batch_size):>5}  "
        f"{row.avg_inference_sec:>9.2f}  "
        f"{row.avg_images_per_hour:>7.1f}  "
        f"{row.avg_peak_mem_mb:>7.1f}  "
        f"{row.avg_psnr_db:>8.2f}  "
        f"{row.avg_ssim:>6.4f}  "
        f"{row.avg_bpp:>6.4f}  "
        f"{int(row.n_images):>3}"
    )
print("=" * 80 + "\n")

# Colour scheme: one colour per (precision, batch size) variant
variant_keys = list(summary[["precision", "batch_size"]].drop_duplicates().itertuples(index=False, name=None))
cmap = plt.get_cmap("tab10")
palette = {key: cmap(i % 10) for i, key in enumerate(variant_keys)}


def variant_label(prec, batch_size):
    return f"{prec}, batch={int(batch_size)}"


# ── Plot 1: Inference time vs steps ───────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
for (prec, batch_size), color in palette.items():
    sub = summary[
        (summary["precision"] == prec) &
        (summary["batch_size"] == batch_size)
    ].sort_values("n_denoise_step")
    steps = sub["n_denoise_step"].values
    ax.plot(
        steps,
        sub["avg_inference_sec"],
        marker="o",
        color=color,
        label=variant_label(prec, batch_size),
        linewidth=2,
    )
    # Shaded band shows +/- 1 std deviation across images
    ax.fill_between(
        steps,
        sub["avg_inference_sec"] - sub["std_inference_sec"].fillna(0),
        sub["avg_inference_sec"] + sub["std_inference_sec"].fillna(0),
        alpha=0.15, color=color,
    )

ax.set_xlabel("Denoising Steps", fontsize=12)
ax.set_ylabel("Avg Inference Time per Image (s)", fontsize=12)
ax.set_title("Reconstruction Time vs Denoising Steps", fontsize=13, fontweight="bold")
ax.legend(title="Precision / batch")
ax.grid(True, linestyle="--", alpha=0.5)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tight_layout()
p1 = out_dir / "plot_time_vs_steps.png"
fig.savefig(p1, dpi=150)
print(f"Saved: {p1}")
plt.close()

# ── Plot 2: PSNR vs steps ─────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
for (prec, batch_size), color in palette.items():
    sub = summary[
        (summary["precision"] == prec) &
        (summary["batch_size"] == batch_size)
    ].sort_values("n_denoise_step")
    steps = sub["n_denoise_step"].values
    ax.plot(
        steps,
        sub["avg_psnr_db"],
        marker="s",
        color=color,
        label=variant_label(prec, batch_size),
        linewidth=2,
    )
    ax.fill_between(
        steps,
        sub["avg_psnr_db"] - sub["std_psnr_db"].fillna(0),
        sub["avg_psnr_db"] + sub["std_psnr_db"].fillna(0),
        alpha=0.15, color=color,
    )

ax.set_xlabel("Denoising Steps", fontsize=12)
ax.set_ylabel("Average PSNR (dB)", fontsize=12)
ax.set_title("Reconstruction Quality (PSNR) vs Denoising Steps", fontsize=13, fontweight="bold")
ax.legend(title="Precision / batch")
ax.grid(True, linestyle="--", alpha=0.5)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tight_layout()
p2 = out_dir / "plot_psnr_vs_steps.png"
fig.savefig(p2, dpi=150)
print(f"Saved: {p2}")
plt.close()

# ── Plot 3: SSIM vs steps ─────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
for (prec, batch_size), color in palette.items():
    sub = summary[
        (summary["precision"] == prec) &
        (summary["batch_size"] == batch_size)
    ].sort_values("n_denoise_step")
    steps = sub["n_denoise_step"].values
    ax.plot(
        steps,
        sub["avg_ssim"],
        marker="^",
        color=color,
        label=variant_label(prec, batch_size),
        linewidth=2,
    )
    ax.fill_between(
        steps,
        sub["avg_ssim"] - sub["std_ssim"].fillna(0),
        sub["avg_ssim"] + sub["std_ssim"].fillna(0),
        alpha=0.15, color=color,
    )

ax.set_xlabel("Denoising Steps", fontsize=12)
ax.set_ylabel("Average SSIM", fontsize=12)
ax.set_title("Reconstruction Quality (SSIM) vs Denoising Steps", fontsize=13, fontweight="bold")
ax.legend(title="Precision / batch")
ax.grid(True, linestyle="--", alpha=0.5)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tight_layout()
p3 = out_dir / "plot_ssim_vs_steps.png"
fig.savefig(p3, dpi=150)
print(f"Saved: {p3}")
plt.close()

# ── Plot 4: Quality vs Speed trade-off (PSNR vs inference time) ───────────────
# This is the key chart for finding the elbow point:
# x-axis = speed (lower = faster), y-axis = quality (higher = better)

fig, ax = plt.subplots(figsize=(8, 5))
for (prec, batch_size), color in palette.items():
    sub = summary[
        (summary["precision"] == prec) &
        (summary["batch_size"] == batch_size)
    ].sort_values("n_denoise_step")
    sc = ax.scatter(
        sub["avg_inference_sec"],
        sub["avg_psnr_db"],
        color=color, s=80, label=variant_label(prec, batch_size), zorder=3,
    )
    # Annotate each point with its step count
    for _, row in sub.iterrows():
        ax.annotate(
            f"{int(row.n_denoise_step)}",
            xy=(row.avg_inference_sec, row.avg_psnr_db),
            xytext=(4, 4), textcoords="offset points",
            fontsize=8, color=color,
        )

ax.set_xlabel("Avg Inference Time per Image (s)  [lower = faster]", fontsize=12)
ax.set_ylabel("Average PSNR (dB)  [higher = better]", fontsize=12)
ax.set_title("Quality vs Speed Trade-off\n(annotated with step count)", fontsize=13, fontweight="bold")
ax.legend(title="Precision / batch")
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
p4 = out_dir / "plot_quality_vs_speed.png"
fig.savefig(p4, dpi=150)
print(f"Saved: {p4}")
plt.close()

# ── Plot 5: GPU memory vs steps ───────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
for (prec, batch_size), color in palette.items():
    sub = summary[
        (summary["precision"] == prec) &
        (summary["batch_size"] == batch_size)
    ].sort_values("n_denoise_step")
    steps = sub["n_denoise_step"].values
    ax.plot(
        steps,
        sub["avg_peak_mem_mb"],
        marker="D",
        color=color,
        label=variant_label(prec, batch_size),
        linewidth=2,
    )

ax.set_xlabel("Denoising Steps", fontsize=12)
ax.set_ylabel("Peak GPU Memory (MB)", fontsize=12)
ax.set_title("GPU Memory Usage vs Denoising Steps", fontsize=13, fontweight="bold")
ax.legend(title="Precision / batch")
ax.grid(True, linestyle="--", alpha=0.5)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tight_layout()
p5 = out_dir / "plot_memory_vs_steps.png"
fig.savefig(p5, dpi=150)
print(f"Saved: {p5}")
plt.close()

print("\nAll plots saved to:", out_dir)
