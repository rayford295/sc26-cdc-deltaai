# Compression Results Template

Use this folder layout when copying lightweight outputs back into the repository.

```text
results/YYYY-MM-DD-compression-suite/
├── tables/
│   ├── baseline_resolution_batch_summary.csv
│   ├── checkpoint_level_summary.csv
│   ├── tiling_summary.csv
│   ├── hpc_scaling_summary.csv
│   └── storage_compare_summary.csv
├── reports/
│   ├── baseline_resolution_batch.md
│   ├── checkpoint_level.md
│   ├── tiling.md
│   ├── hpc_scaling.md
│   └── storage_compare.md
└── visual_examples_small/
    ├── native_reference.png
    ├── tile_256_stitched.png
    ├── tile_256_error_heatmap.png
    ├── tile_256_comparison.png
    ├── tile_512_stitched.png
    ├── tile_512_error_heatmap.png
    ├── tile_512_comparison.png
    └── tile_1024_stitched.png
```

Commit small tables, Markdown reports, and reduced-size visual examples. Keep full-resolution outputs, raw logs, checkpoints, and full datasets on DeltaAI.
