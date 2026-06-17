# Data Directory

Dataset files are intentionally not included in this upload folder.

Expected local layout after downloading public datasets:

```text
data/
  kvasir_seg/
    images/
    masks/
  isic_2018_task1/
    images/
    masks/
  busi/
    images/
    masks/
```

The repository includes split manifests under `results/splits/` and dataset-preparation scripts under `scripts/`. Keep raw image and mask files outside Git unless you intentionally share them through an approved dataset archive.
