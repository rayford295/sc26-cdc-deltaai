#!/usr/bin/env python3
"""Combine SC26 compression experiment summaries into CSV and Markdown tables."""

from __future__ import annotations

import argparse
import csv
import pathlib
from typing import Iterable


DEFAULT_COLUMNS = [
    "run_start_utc",
    "run_end_utc",
    "run_stamp",
    "slurm_job_id",
    "experiment_name",
    "compression_setting",
    "resolution_label",
    "mode",
    "tile_size",
    "tile_batch_size",
    "batch_size",
    "n_images",
    "width",
    "height",
    "avg_wall_sec",
    "avg_inference_sec",
    "avg_peak_gpu_mem_mb",
    "avg_bpp",
    "avg_compression_ratio",
    "avg_psnr_db",
    "avg_ssim",
    "avg_mse",
    "avg_rmse",
    "avg_mae",
    "avg_error_p95",
    "avg_error_p99",
    "avg_max_abs_error",
    "avg_bias_mean",
    "avg_seam_error_mean",
    "storage_label",
]


def read_summary(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def find_summary_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    for path in sorted(root.rglob("summary.csv")):
        if path.name.startswith("combined_"):
            continue
        yield path


def write_csv(path: pathlib.Path, rows: list[dict[str, str]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return fieldnames


def write_markdown(path: pathlib.Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    lines = ["# SC26 Compression Experiment Summary", ""]
    if not rows:
        lines.append("No summary rows found.")
        path.write_text("\n".join(lines) + "\n")
        return
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine SC26 compression summary.csv files")
    parser.add_argument("--root", required=True, help="Root output directory to scan")
    parser.add_argument("--out_csv", default=None, help="Combined CSV path")
    parser.add_argument("--out_md", default=None, help="Markdown table path")
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    rows: list[dict[str, str]] = []
    for summary_file in find_summary_files(root):
        for row in read_summary(summary_file):
            row["summary_path"] = str(summary_file)
            rows.append(row)

    out_csv = pathlib.Path(args.out_csv) if args.out_csv else root / "combined_summary.csv"
    out_md = pathlib.Path(args.out_md) if args.out_md else root / "combined_summary.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    write_csv(out_csv, rows)
    columns = [column for column in DEFAULT_COLUMNS if any(column in row for row in rows)]
    write_markdown(out_md, rows, columns)
    print(f"Found {len(rows)} summary rows")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
