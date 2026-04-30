# 2026-04-30 Weekly Progress Slides

This folder contains the editable slide deck for the CDC reconstruction weekly update.

## Deck

| File | Use |
|------|-----|
| `SC26_CDC_Weekly_Progress_2026-04-30.pptx` | Standalone editable PowerPoint deck for sharing |
| `output/output.pptx` | Build output copy of the same deck |
| `src/create_weekly_progress_deck.mjs` | Rebuild script for the deck |
| `scratch/previews/` | Rendered PNG previews used for quality checks |

## Coverage

The deck summarizes the current SC26 reconstruction progress:

- DeltaAI GH200 full reconstruction profiling and step sweep.
- Delta H200 quick reconstruction sweep and batch-size pilot.
- Current batch-size decision: use `batch_size=1`.
- Matched GH200-vs-H200 timing comparison for steps `5`, `20`, and `65`.
- Key conclusion: denoising step count is the main speed lever; fp16 helps memory and speed but currently has invalid BPP output.

## Rebuild

Run from this folder after the Codex presentation runtime has linked `@oai/artifact-tool`:

```bash
node src/create_weekly_progress_deck.mjs
```
