# Results Archive

This folder stores lightweight, report-ready experiment outputs that can be committed to GitHub.

Large raw outputs, full-resolution reconstructed PNGs, logs, model weights, and full datasets should stay on DeltaAI under `/projects/bfod/$USER/cdc-deltaai/`.

## Index

| Folder | Cycle | Owner | Purpose |
|--------|-------|-------|---------|
| `2026-04-26-reconstruction/` | 2026-04-25 to 2026-05-01 meeting cycle | Yifan | Reconstruction profiling, step sweep, batch-size decision, plots, reports, and visual examples |
| `2026-04-28-h200-reconstruction/` | 2026-04-28 hardware comparison extension | Yifan | Delta H200 batch pilot and quick reconstruction step sweep |

## Hardware Scope

The first committed reconstruction result set is DeltaAI GH200. Delta is a separate NCSA system and has visible H200 partitions under the `bfod-delta-gpu` account:

- `gpuH200x8`
- `gpuH200x8-interactive`

Delta H200 results are stored separately under `2026-04-28-h200-reconstruction/` so they are not mixed with the 2026-04-26 DeltaAI GH200 result set.

## Convention for Future Results

Use one dated folder per experiment cycle:

```text
results/YYYY-MM-DD-short-description/
├── plots/
├── reports/
├── tables/
└── visual_examples_small/
```

Commit small summaries, CSVs, and slide-ready compressed images. Keep full-resolution generated outputs on DeltaAI unless they are specifically needed in the repository.
