# Compression Control Inspection

## x-param path used by the SC26 compression scripts

`xparam/modules/denoising_diffusion.py` exposes this `compress()` signature:

```python
def compress(
        self,
        images,
        sample_steps=None,
        bpp_return_mean=True,
        init=None,
        eta=0,
):
```

There is no direct runtime argument named `compression_level`, `quality`, `bitrate`, `lambda`, or `quantization`.
For x-param experiments, treat compression level as checkpoint-controlled. Sweep the pretrained checkpoint files and
use the measured BPP and compression ratio to label the practical low, medium, and high compression settings.

## epsilon-param path

`epsilonparam/modules/denoising_diffusion.py` exposes this `compress()` signature:

```python
def compress(
        self,
        images,
        sample_steps=None,
        bitrate_scale=None,
        sample_mode="ddpm",
        bpp_return_mean=True,
        init=None,
        eta=0,
):
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
