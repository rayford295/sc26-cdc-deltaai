#!/usr/bin/env python3
"""
Run one SC26 compression experiment configuration.

The script supports the experiment axes requested for the next meeting:

- image resolution: native, 4K, 2K, 1K, or any max-edge value
- checkpoint-based compression sweep
- full-image batch sizes
- tiled inference with stitching-quality metrics
- shared/local storage labels for later comparison

It writes one per-image CSV, one one-row summary CSV, a JSON manifest, and a
short Markdown report. The model is loaded once per invocation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import sys
import time
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
import torchvision
from ema_pytorch import EMA
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
XPARAM_ROOT = REPO_ROOT / "xparam"
sys.path.insert(0, str(XPARAM_ROOT))

from modules.compress_modules import ResnetCompressor  # noqa: E402
from modules.denoising_diffusion import GaussianDiffusion  # noqa: E402
from modules.unet import Unet  # noqa: E402


UNCOMPRESSED_BPP = 24.0
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


@dataclass
class ExperimentConfig:
    ckpt: str
    img_dir: str
    out_dir: str
    lpips_weight: float
    experiment_name: str = "sc26_compression"
    compression_setting: str = "baseline"
    resolution_label: str = "native"
    max_edge: int = 0
    n_images: int = 4
    start_index: int = 0
    batch_size: int = 1
    tile_size: int = 0
    tile_batch_size: int = 0
    n_denoise_step: int = 65
    gamma: float = 0.8
    device: int = 0
    fp16: bool = False
    save_visual_limit: int = 2
    save_all_images: bool = False
    storage_label: str = "shared"
    notes: str = ""


class CudaTimer:
    def __init__(self, device_index: int):
        self.device_index = device_index
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        self.start_event.record()

    def stop(self) -> float:
        self.end_event.record()
        torch.cuda.synchronize(self.device_index)
        return self.start_event.elapsed_time(self.end_event) / 1000.0


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Run one SC26 CDC compression experiment")
    parser.add_argument("--ckpt", required=True, help="Path to pretrained x-param checkpoint")
    parser.add_argument("--img_dir", required=True, help="Directory containing input images")
    parser.add_argument("--out_dir", required=True, help="Output directory for this experiment")
    parser.add_argument("--lpips_weight", type=float, required=True, help="LPIPS weight matching checkpoint")
    parser.add_argument("--experiment_name", default="sc26_compression")
    parser.add_argument("--compression_setting", default="baseline")
    parser.add_argument("--resolution_label", default="native")
    parser.add_argument(
        "--max_edge",
        type=int,
        default=0,
        help="Resize so the longest edge equals this value. Use 0 for native size.",
    )
    parser.add_argument("--n_images", type=int, default=4)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--tile_size", type=int, default=0, help="0 disables tiling")
    parser.add_argument(
        "--tile_batch_size",
        type=int,
        default=0,
        help="Tiles per inference batch. Defaults to --batch_size when tiling is enabled.",
    )
    parser.add_argument("--n_denoise_step", type=int, default=65)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save_visual_limit", type=int, default=2)
    parser.add_argument("--save_all_images", action="store_true")
    parser.add_argument("--storage_label", default="shared")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.tile_size and args.tile_size % 64 != 0:
        raise ValueError("--tile_size must be a multiple of 64")
    if args.tile_batch_size < 0:
        raise ValueError("--tile_batch_size must be >= 0")
    return ExperimentConfig(**vars(args))


def get_cuda_device(device_index: int) -> tuple[torch.device, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run inside a GPU allocation.")
    visible_devices = torch.cuda.device_count()
    if device_index < 0 or device_index >= visible_devices:
        raise RuntimeError(
            f"Requested CUDA device {device_index}, but PyTorch sees {visible_devices} visible device(s)."
        )
    torch.cuda.set_device(device_index)
    return torch.device("cuda", device_index), device_index


def build_and_load_model(ckpt_path: str, device: torch.device, lpips_weight: float) -> GaussianDiffusion:
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
    loaded_param = torch.load(ckpt_path, map_location=lambda storage, _: storage)
    ema = EMA(diffusion, beta=0.999, update_every=10, power=0.75, update_after_step=100)
    ema.load_state_dict(loaded_param["ema"])
    diffusion = ema.ema_model
    diffusion.to(device)
    diffusion.eval()
    return diffusion


def list_images(img_dir: pathlib.Path, start_index: int, n_images: int) -> list[pathlib.Path]:
    images = sorted(path for path in img_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    selected = images[start_index : start_index + n_images]
    if not selected:
        raise ValueError(f"No images selected from {img_dir}. Check --start_index and --n_images.")
    return selected


def read_image_tensor(path: pathlib.Path) -> torch.Tensor:
    return torchvision.io.read_image(str(path)).unsqueeze(0).float() / 255.0


def resize_to_max_edge(tensor: torch.Tensor, max_edge: int) -> torch.Tensor:
    if max_edge <= 0:
        return tensor
    _, _, height, width = tensor.shape
    current_max = max(height, width)
    if current_max == max_edge:
        return tensor
    scale = max_edge / float(current_max)
    new_height = max(64, int(round(height * scale)))
    new_width = max(64, int(round(width * scale)))
    return F.interpolate(tensor, size=(new_height, new_width), mode="bilinear", align_corners=False)


def crop_to_multiple(tensor: torch.Tensor, multiple: int = 64) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    height_m = (height // multiple) * multiple
    width_m = (width // multiple) * multiple
    if height_m < multiple or width_m < multiple:
        raise ValueError(f"Image became too small after resize/crop: {width} x {height}")
    return tensor[:, :, :height_m, :width_m]


def prepare_image(path: pathlib.Path, max_edge: int) -> tuple[torch.Tensor, dict[str, int]]:
    tensor = read_image_tensor(path)
    native_height, native_width = tensor.shape[-2:]
    tensor = resize_to_max_edge(tensor, max_edge)
    resized_height, resized_width = tensor.shape[-2:]
    tensor = crop_to_multiple(tensor, 64)
    crop_height, crop_width = tensor.shape[-2:]
    meta = {
        "native_width": native_width,
        "native_height": native_height,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "width": crop_width,
        "height": crop_height,
    }
    return tensor.contiguous(), meta


def common_crop(images: list[tuple[pathlib.Path, torch.Tensor, dict[str, int]]]) -> list[tuple[pathlib.Path, torch.Tensor, dict[str, int]]]:
    min_height = min(item[1].shape[-2] for item in images)
    min_width = min(item[1].shape[-1] for item in images)
    min_height = (min_height // 64) * 64
    min_width = (min_width // 64) * 64
    cropped = []
    for path, tensor, meta in images:
        tensor = tensor[:, :, :min_height, :min_width].contiguous()
        updated = dict(meta)
        updated["width"] = min_width
        updated["height"] = min_height
        cropped.append((path, tensor, updated))
    return cropped


def compute_psnr_ssim(original: torch.Tensor, reconstructed: torch.Tensor) -> tuple[float, float]:
    orig_np = (
        original.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
    ).astype(np.uint8)
    recon_np = (
        reconstructed.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
    ).astype(np.uint8)
    return (
        float(psnr_fn(orig_np, recon_np, data_range=255)),
        float(ssim_fn(orig_np, recon_np, data_range=255, channel_axis=2)),
    )


def safe_ratio(bpp: float) -> float:
    if not math.isfinite(bpp) or bpp <= 0:
        return float("nan")
    return UNCOMPRESSED_BPP / bpp


def estimated_bytes_from_bpp(width: int, height: int, bpp: float) -> int | None:
    if not math.isfinite(bpp) or bpp < 0:
        return None
    return int(round(width * height * bpp / 8.0))


def should_save_image(index: int, config: ExperimentConfig) -> bool:
    return config.save_all_images or index < config.save_visual_limit


def run_full_image(
    diffusion: GaussianDiffusion,
    selected: list[pathlib.Path],
    config: ExperimentConfig,
    device: torch.device,
    device_index: int,
    visuals_dir: pathlib.Path,
) -> list[dict[str, object]]:
    timer = CudaTimer(device_index)
    prepared = []
    for path in selected:
        tensor, meta = prepare_image(path, config.max_edge)
        prepared.append((path, tensor, meta))
    prepared = common_crop(prepared)

    rows: list[dict[str, object]] = []
    global_index = 0
    precision_label = "fp16" if config.fp16 else "fp32"

    for batch_start in range(0, len(prepared), config.batch_size):
        batch_items = prepared[batch_start : batch_start + config.batch_size]
        batch_tensor_cpu = torch.cat([item[1] for item in batch_items], dim=0)
        batch_paths = [item[0] for item in batch_items]
        batch_metas = [item[2] for item in batch_items]
        effective_batch_size = len(batch_items)

        torch.cuda.reset_peak_memory_stats(device_index)
        wall_start = time.perf_counter()
        batch_tensor = batch_tensor_cpu.to(device)
        timer.start()
        with torch.no_grad():
            with amp.autocast(enabled=config.fp16):
                compressed, bpp = diffusion.compress(
                    batch_tensor * 2.0 - 1.0,
                    sample_steps=config.n_denoise_step,
                    bpp_return_mean=False,
                    init=torch.randn_like(batch_tensor) * config.gamma,
                )
        batch_inference_sec = timer.stop()
        peak_mem_mb = torch.cuda.max_memory_allocated(device_index) / 1024**2

        post_start = time.perf_counter()
        reconstructed = compressed.clamp(-1, 1) / 2.0 + 0.5
        bpp_values = bpp.detach().float().cpu().reshape(-1).tolist()
        postproc_sec = time.perf_counter() - post_start
        wall_sec = time.perf_counter() - wall_start

        for item_index, path in enumerate(batch_paths):
            recon_single = reconstructed[item_index : item_index + 1]
            orig_single = batch_tensor_cpu[item_index : item_index + 1]
            psnr_val, ssim_val = compute_psnr_ssim(orig_single, recon_single)
            bpp_val = float(bpp_values[item_index])
            width = int(batch_metas[item_index]["width"])
            height = int(batch_metas[item_index]["height"])
            estimated_bytes = estimated_bytes_from_bpp(width, height, bpp_val)
            out_path = ""
            recon_png_bytes = ""
            if should_save_image(global_index, config):
                out_file = visuals_dir / f"{path.stem}_recon.png"
                torchvision.utils.save_image(recon_single.cpu(), str(out_file))
                out_path = str(out_file)
                recon_png_bytes = os.path.getsize(out_file)

            rows.append(
                {
                    "status": "success",
                    "experiment_name": config.experiment_name,
                    "compression_setting": config.compression_setting,
                    "resolution_label": config.resolution_label,
                    "max_edge": config.max_edge,
                    "storage_label": config.storage_label,
                    "mode": "full_image",
                    "tile_size": 0,
                    "tile_batch_size": "",
                    "batch_size": config.batch_size,
                    "effective_batch_size": effective_batch_size,
                    "precision": precision_label,
                    "n_denoise_step": config.n_denoise_step,
                    "image": path.name,
                    "native_width": batch_metas[item_index]["native_width"],
                    "native_height": batch_metas[item_index]["native_height"],
                    "width": width,
                    "height": height,
                    "num_tiles": 0,
                    "batch_inference_sec": round(batch_inference_sec, 4),
                    "inference_sec": round(batch_inference_sec / effective_batch_size, 4),
                    "wall_sec": round(wall_sec / effective_batch_size, 4),
                    "postproc_sec": round(postproc_sec / effective_batch_size, 4),
                    "peak_gpu_mem_mb": round(peak_mem_mb, 1),
                    "bpp": round(bpp_val, 6),
                    "estimated_compressed_bytes": estimated_bytes if estimated_bytes is not None else "",
                    "compression_ratio": round(safe_ratio(bpp_val), 4),
                    "psnr_db": round(psnr_val, 4),
                    "ssim": round(ssim_val, 6),
                    "seam_error_mean": "",
                    "seam_error_max": "",
                    "orig_size_bytes": os.path.getsize(path),
                    "recon_png_bytes": recon_png_bytes,
                    "recon_path": out_path,
                    "error": "",
                }
            )
            global_index += 1

        del batch_tensor, compressed, reconstructed
        torch.cuda.empty_cache()
        print(
            f"full | batch={config.batch_size} | {batch_start + effective_batch_size}/{len(prepared)} "
            f"| infer={batch_inference_sec:.2f}s | mem={peak_mem_mb:.0f} MB"
        )

    return rows


def pad_to_tile_multiple(tensor: torch.Tensor, tile_size: int) -> tuple[torch.Tensor, int, int]:
    height, width = tensor.shape[-2:]
    pad_h = (tile_size - height % tile_size) % tile_size
    pad_w = (tile_size - width % tile_size) % tile_size
    if pad_h == 0 and pad_w == 0:
        return tensor, 0, 0
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate"), pad_h, pad_w


def iter_tiles(tensor: torch.Tensor, tile_size: int) -> Iterable[tuple[int, int, torch.Tensor]]:
    height, width = tensor.shape[-2:]
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            yield y, x, tensor[:, :, y : y + tile_size, x : x + tile_size]


def seam_artifact_metrics(original: torch.Tensor, reconstructed: torch.Tensor, tile_size: int) -> tuple[float, float]:
    _, _, height, width = original.shape
    seam_errors = []
    for x in range(tile_size, width, tile_size):
        orig_grad = (original[:, :, :, x] - original[:, :, :, x - 1]).abs()
        recon_grad = (reconstructed[:, :, :, x] - reconstructed[:, :, :, x - 1]).abs()
        seam_errors.append((recon_grad - orig_grad).abs().mean().item())
    for y in range(tile_size, height, tile_size):
        orig_grad = (original[:, :, y, :] - original[:, :, y - 1, :]).abs()
        recon_grad = (reconstructed[:, :, y, :] - reconstructed[:, :, y - 1, :]).abs()
        seam_errors.append((recon_grad - orig_grad).abs().mean().item())
    if not seam_errors:
        return 0.0, 0.0
    return float(np.mean(seam_errors)), float(np.max(seam_errors))


def run_tiled(
    diffusion: GaussianDiffusion,
    selected: list[pathlib.Path],
    config: ExperimentConfig,
    device: torch.device,
    device_index: int,
    visuals_dir: pathlib.Path,
) -> list[dict[str, object]]:
    tile_batch_size = config.tile_batch_size or config.batch_size
    timer = CudaTimer(device_index)
    rows: list[dict[str, object]] = []
    precision_label = "fp16" if config.fp16 else "fp32"

    for image_index, path in enumerate(selected):
        wall_start = time.perf_counter()
        tensor, meta = prepare_image(path, config.max_edge)
        original = tensor.clone()
        padded, pad_h, pad_w = pad_to_tile_multiple(tensor, config.tile_size)
        padded_h, padded_w = padded.shape[-2:]
        tile_specs = list(iter_tiles(padded, config.tile_size))
        stitched = torch.zeros_like(padded)
        bpp_values: list[float] = []
        total_inference_sec = 0.0
        peak_mem_mb = 0.0

        for tile_start in range(0, len(tile_specs), tile_batch_size):
            batch_specs = tile_specs[tile_start : tile_start + tile_batch_size]
            batch_tensor_cpu = torch.cat([spec[2] for spec in batch_specs], dim=0)
            torch.cuda.reset_peak_memory_stats(device_index)
            batch_tensor = batch_tensor_cpu.to(device)

            timer.start()
            with torch.no_grad():
                with amp.autocast(enabled=config.fp16):
                    compressed, bpp = diffusion.compress(
                        batch_tensor * 2.0 - 1.0,
                        sample_steps=config.n_denoise_step,
                        bpp_return_mean=False,
                        init=torch.randn_like(batch_tensor) * config.gamma,
                    )
            batch_inference_sec = timer.stop()
            total_inference_sec += batch_inference_sec
            peak_mem_mb = max(peak_mem_mb, torch.cuda.max_memory_allocated(device_index) / 1024**2)

            reconstructed_tiles = (compressed.clamp(-1, 1) / 2.0 + 0.5).detach().cpu()
            for local_index, (y, x, _) in enumerate(batch_specs):
                stitched[:, :, y : y + config.tile_size, x : x + config.tile_size] = reconstructed_tiles[
                    local_index : local_index + 1
                ]
            bpp_values.extend(bpp.detach().float().cpu().reshape(-1).tolist())

            del batch_tensor, compressed, reconstructed_tiles
            torch.cuda.empty_cache()

        stitched = stitched[:, :, : meta["height"], : meta["width"]]
        psnr_val, ssim_val = compute_psnr_ssim(original, stitched)
        seam_mean, seam_max = seam_artifact_metrics(original, stitched, config.tile_size)

        total_bits = 0.0
        for bpp_val in bpp_values:
            total_bits += float(bpp_val) * config.tile_size * config.tile_size
        effective_bpp = total_bits / float(meta["width"] * meta["height"])
        estimated_bytes = estimated_bytes_from_bpp(meta["width"], meta["height"], effective_bpp)
        wall_sec = time.perf_counter() - wall_start

        out_path = ""
        recon_png_bytes = ""
        if should_save_image(image_index, config):
            out_file = visuals_dir / f"{path.stem}_tile{config.tile_size}_stitched.png"
            torchvision.utils.save_image(stitched.cpu(), str(out_file))
            out_path = str(out_file)
            recon_png_bytes = os.path.getsize(out_file)

        rows.append(
            {
                "status": "success",
                "experiment_name": config.experiment_name,
                "compression_setting": config.compression_setting,
                "resolution_label": config.resolution_label,
                "max_edge": config.max_edge,
                "storage_label": config.storage_label,
                "mode": "tiled",
                "tile_size": config.tile_size,
                "tile_batch_size": tile_batch_size,
                "batch_size": config.batch_size,
                "effective_batch_size": tile_batch_size,
                "precision": precision_label,
                "n_denoise_step": config.n_denoise_step,
                "image": path.name,
                "native_width": meta["native_width"],
                "native_height": meta["native_height"],
                "width": meta["width"],
                "height": meta["height"],
                "num_tiles": len(tile_specs),
                "batch_inference_sec": round(total_inference_sec, 4),
                "inference_sec": round(total_inference_sec, 4),
                "wall_sec": round(wall_sec, 4),
                "postproc_sec": "",
                "peak_gpu_mem_mb": round(peak_mem_mb, 1),
                "bpp": round(float(effective_bpp), 6),
                "estimated_compressed_bytes": estimated_bytes if estimated_bytes is not None else "",
                "compression_ratio": round(safe_ratio(float(effective_bpp)), 4),
                "psnr_db": round(psnr_val, 4),
                "ssim": round(ssim_val, 6),
                "seam_error_mean": round(seam_mean, 8),
                "seam_error_max": round(seam_max, 8),
                "orig_size_bytes": os.path.getsize(path),
                "recon_png_bytes": recon_png_bytes,
                "recon_path": out_path,
                "error": "",
                "pad_h": pad_h,
                "pad_w": pad_w,
                "padded_width": padded_w,
                "padded_height": padded_h,
            }
        )
        print(
            f"tile={config.tile_size} | {image_index + 1}/{len(selected)} | "
            f"tiles={len(tile_specs)} | infer={total_inference_sec:.2f}s | "
            f"mem={peak_mem_mb:.0f} MB | seam={seam_mean:.6f}"
        )

    return rows


def write_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def numeric_values(rows: list[dict[str, object]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field, "")
        if value == "" or value is None:
            continue
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value_float):
            values.append(value_float)
    return values


def summarize_rows(rows: list[dict[str, object]], config: ExperimentConfig, model_load_sec: float, total_wall_sec: float) -> dict[str, object]:
    success_rows = [row for row in rows if row.get("status") == "success"]
    first = success_rows[0] if success_rows else {}

    def avg(field: str) -> object:
        values = numeric_values(success_rows, field)
        return round(float(np.mean(values)), 6) if values else ""

    def max_value(field: str) -> object:
        values = numeric_values(success_rows, field)
        return round(float(np.max(values)), 6) if values else ""

    total_pixels = sum(float(row.get("width", 0)) * float(row.get("height", 0)) for row in success_rows)
    total_estimated_bytes = sum(float(row.get("estimated_compressed_bytes", 0) or 0) for row in success_rows)
    effective_bpp = (total_estimated_bytes * 8.0 / total_pixels) if total_pixels else float("nan")

    return {
        "experiment_name": config.experiment_name,
        "compression_setting": config.compression_setting,
        "resolution_label": config.resolution_label,
        "max_edge": config.max_edge,
        "storage_label": config.storage_label,
        "mode": first.get("mode", ""),
        "tile_size": config.tile_size,
        "tile_batch_size": config.tile_batch_size or config.batch_size if config.tile_size else "",
        "batch_size": config.batch_size,
        "precision": "fp16" if config.fp16 else "fp32",
        "n_denoise_step": config.n_denoise_step,
        "n_images": len(success_rows),
        "width": first.get("width", ""),
        "height": first.get("height", ""),
        "model_load_sec": round(model_load_sec, 4),
        "total_wall_sec": round(total_wall_sec, 4),
        "avg_wall_sec": avg("wall_sec"),
        "avg_inference_sec": avg("inference_sec"),
        "avg_peak_gpu_mem_mb": avg("peak_gpu_mem_mb"),
        "max_peak_gpu_mem_mb": max_value("peak_gpu_mem_mb"),
        "avg_bpp": round(float(effective_bpp), 6) if math.isfinite(effective_bpp) else avg("bpp"),
        "avg_compression_ratio": round(safe_ratio(float(effective_bpp)), 4)
        if math.isfinite(effective_bpp)
        else avg("compression_ratio"),
        "avg_psnr_db": avg("psnr_db"),
        "avg_ssim": avg("ssim"),
        "avg_seam_error_mean": avg("seam_error_mean"),
        "max_seam_error_max": max_value("seam_error_max"),
        "total_estimated_compressed_bytes": int(total_estimated_bytes) if total_estimated_bytes else "",
        "checkpoint": config.ckpt,
        "notes": config.notes,
    }


def write_report(path: pathlib.Path, summary: dict[str, object], config: ExperimentConfig) -> None:
    lines = [
        f"# {config.experiment_name}",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for key in [
        "compression_setting",
        "resolution_label",
        "mode",
        "tile_size",
        "batch_size",
        "n_images",
        "avg_wall_sec",
        "avg_inference_sec",
        "avg_peak_gpu_mem_mb",
        "avg_bpp",
        "avg_compression_ratio",
        "avg_psnr_db",
        "avg_ssim",
        "avg_seam_error_mean",
        "max_seam_error_max",
    ]:
        lines.append(f"| `{key}` | {summary.get(key, '')} |")
    lines.extend(["", "## Files", "", "- `results.csv`: per-image metrics", "- `summary.csv`: one-row aggregate"])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    config = parse_args()
    out_dir = pathlib.Path(config.out_dir)
    visuals_dir = out_dir / "visuals"
    out_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("SC26 CDC compression experiment")
    print("=" * 72)
    print(json.dumps(asdict(config), indent=2))

    selected = list_images(pathlib.Path(config.img_dir), config.start_index, config.n_images)
    device, device_index = get_cuda_device(config.device)

    total_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device_index)
    load_start = time.perf_counter()
    diffusion = build_and_load_model(config.ckpt, device, config.lpips_weight)
    torch.cuda.synchronize(device_index)
    model_load_sec = time.perf_counter() - load_start
    print(f"Model loaded in {model_load_sec:.2f}s")

    if config.tile_size:
        rows = run_tiled(diffusion, selected, config, device, device_index, visuals_dir)
    else:
        rows = run_full_image(diffusion, selected, config, device, device_index, visuals_dir)

    total_wall_sec = time.perf_counter() - total_start
    summary = summarize_rows(rows, config, model_load_sec, total_wall_sec)

    write_csv(out_dir / "results.csv", rows)
    write_csv(out_dir / "summary.csv", [summary])
    (out_dir / "manifest.json").write_text(json.dumps({"config": asdict(config), "summary": summary}, indent=2) + "\n")
    write_report(out_dir / "report.md", summary, config)

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {out_dir}")


if __name__ == "__main__":
    main()
