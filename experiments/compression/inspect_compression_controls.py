#!/usr/bin/env python3
"""Write a short report on compression-control knobs available in this repo."""

from __future__ import annotations

import pathlib
import re


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "experiments" / "compression" / "compression_controls_report.md"


def extract_signature(path: pathlib.Path) -> str:
    text = path.read_text()
    match = re.search(r"def compress\((.*?)\n\s*\):", text, flags=re.S)
    if not match:
        return "not found"
    body = "def compress(" + match.group(1) + "\n):"
    return "\n".join(line.rstrip() for line in body.splitlines())


def main() -> None:
    xparam_sig = extract_signature(REPO_ROOT / "xparam" / "modules" / "denoising_diffusion.py")
    epsilon_sig = extract_signature(REPO_ROOT / "epsilonparam" / "modules" / "denoising_diffusion.py")

    report = f"""# Compression Control Inspection

## x-param path used by the SC26 compression scripts

`xparam/modules/denoising_diffusion.py` exposes this `compress()` signature:

```python
{xparam_sig}
```

There is no direct runtime argument named `compression_level`, `quality`, `bitrate`, `lambda`, or `quantization`.
For x-param experiments, treat compression level as checkpoint-controlled. Sweep the pretrained checkpoint files and
use the measured BPP and compression ratio to label the practical low, medium, and high compression settings.

## epsilon-param path

`epsilonparam/modules/denoising_diffusion.py` exposes this `compress()` signature:

```python
{epsilon_sig}
```

This path includes `bitrate_scale`, but the current SC26 scripts and checkpoint list use the x-param workflow.
Do not mix epsilon-param variable-bitrate behavior into x-param results unless the experiment is explicitly changed.

## Recommended SC26 reporting choice

For the next meeting, run the x-param checkpoint sweep and report compression setting by checkpoint plus measured BPP:

- checkpoint name
- average BPP
- compression ratio, `24 / BPP`
- runtime
- PSNR and SSIM

Then choose the best setup from the measured speed, size, and quality table.
"""
    REPORT_PATH.write_text(report)
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
