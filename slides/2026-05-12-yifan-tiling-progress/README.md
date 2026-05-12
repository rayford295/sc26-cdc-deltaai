# 2026-05-12 Yifan Tiling Progress Slides

This folder contains the editable slide deck for the SC26 CDC tiling update.

## Deck

| File | Use |
|------|-----|
| `SC26_CDC_Yifan_Tiling_Progress_2026-05-12.pptx` | Standalone editable PowerPoint deck for sharing |
| `output/SC26_CDC_Yifan_Tiling_Progress_2026-05-12.pptx` | Build output copy of the same deck |
| `src/create_yifan_tiling_progress_deck.mjs` | Rebuild script for the deck |
| `scratch/previews/` | Rendered PNG previews used for visual QA |

## Coverage

The deck summarizes the current Yifan tiling progress:

- DeltaAI GH200 smoke test and `N_IMAGES=8` pilot are complete.
- `512 x 512` tiling is the current recommended setup for speed and memory.
- Pilot result: runtime drops from `143.55` to `86.01` seconds per image, and peak GPU memory drops from `52.0` GB to `3.0` GB.
- Visual artifact check found no obvious grid-like stitching seams in the checked sample.
- Next step: run a larger selected-setup experiment for poster-ready numbers.

## Rebuild

Run from this folder after the Codex presentation runtime has linked `@oai/artifact-tool`:

```bash
node src/create_yifan_tiling_progress_deck.mjs
```
