# Repository Contents

This folder is a clean public copy of the Deployable MedSAM study code and lightweight artifacts.

## Included

- `src/`: reusable package code.
- `scripts/`: command-line workflows and smoke tests.
- `configs/`: project configuration.
- `notebooks/deployable_medsam_study.ipynb`: executable study notebook.
- `docs/`: implementation notes.
- `results/tables/`: lightweight CSV summaries.
- `results/figures/`: lightweight PNG figures.
- `results/splits/`: dataset split manifests.
- `README.md`: setup and workflow guide.
- `requirements.txt` and `requirements-experiments.txt`: dependency lists.

## Excluded

- Dataset images and masks.
- Downloaded dataset archives.
- Model checkpoints.
- ONNX model files.
- Raw predictions.
- Logs and intermediate training outputs.
- Python virtual environments and caches.

## Upload Check

Before uploading this folder to GitHub, confirm:

- No local dataset files are present under `data/`.
- No model files are present under `results/models/`.
- No raw prediction files are present under `results/raw/`.
- The `.gitignore` file is included.
- A license file is added if code reuse permissions should be explicit.
