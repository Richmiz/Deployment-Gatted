# Deployable MedSAM Study Code

This repository contains code and lightweight artifacts for a deployable medical image segmentation study. The workflow adapts a MedSAM-style teacher with LoRA, distills the adapted teacher into a compact U-Net student, exports the student to ONNX, and evaluates INT8 deployment paths with an explicit Dice-drop gate.

The repository is meant to let other researchers inspect, reuse, and adapt the implementation. Dataset images, model checkpoints, ONNX files, and raw prediction outputs are not included.

## What Is Included

- `src/`: Python package code for dataset handling, prompt generation, LoRA adapters, student models, training losses, distillation, quantization, QAT, thresholding, and metrics.
- `scripts/`: command-line entry points for dataset preparation, teacher evaluation, LoRA training, student training, ONNX export, INT8 quantization, Stage 8/9 quantization workflows, external dataset evaluation, CPU timing, and smoke tests.
- `configs/`: project configuration files.
- `notebooks/`: executable study notebook.
- `docs/`: implementation notes and architecture decisions.
- `results/tables/`: lightweight CSV result summaries.
- `results/figures/`: lightweight PNG result figures.
- `results/splits/`: dataset split manifests.

## Repository Layout

```text
.
|-- configs/
|-- data/
|-- docs/
|-- notebooks/
|-- results/
|   |-- figures/
|   |-- splits/
|   `-- tables/
|-- scripts/
|-- src/
|-- requirements.txt
`-- requirements-experiments.txt
```

## Environment Setup

Create and activate a Python environment, then install the base dependencies:

```bash
python -m pip install -r requirements.txt
```

For the full experiment workflow, install the experiment dependencies:

```bash
python -m pip install -r requirements-experiments.txt
```

Optional notebook kernel:

```bash
python -m ipykernel install --user --name deployable_medsam --display-name "Deployable MedSAM"
```

## Data Layout

Dataset files are not included. Place downloaded datasets in this layout:

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

Prepare or validate dataset manifests with:

```bash
python scripts/prepare_segmentation_dataset.py --dataset kvasir_seg
python scripts/prepare_segmentation_dataset.py --dataset isic_2018_task1
python scripts/prepare_segmentation_dataset.py --dataset busi
```

## Main Workflows

Run smoke tests:

```bash
python scripts/smoke_test_dataset_discovery.py
python scripts/smoke_test_segmentation_metrics.py
python scripts/smoke_test_lora_injection.py
python scripts/smoke_test_student_onnx_export.py
python scripts/smoke_test_student_onnx_quantization.py
python scripts/smoke_test_stage8_ablation_contract.py
python scripts/smoke_test_stage9_int8_rescue_contract.py
python -m compileall src scripts
```

Open the study notebook:

```bash
jupyter notebook notebooks/deployable_medsam_study.ipynb
```

Run the LoRA teacher pipeline:

```bash
python scripts/run_lora_teacher_pipeline.py --epochs 10 --rank 8 --device auto
```

Run a fast LoRA pilot:

```bash
python scripts/run_lora_teacher_pipeline.py \
  --epochs 1 \
  --train-sample-limit 8 \
  --validation-sample-limit 4 \
  --eval-sample-limit 4 \
  --device auto
```

Train or evaluate student models:

```bash
python scripts/train_student_baseline.py
python scripts/train_student_distilled.py
python scripts/evaluate_student_model.py --split test --device auto
```

Export and quantize the student:

```bash
python scripts/export_student_onnx.py
python scripts/quantize_student_onnx.py
```

Run the INT8 deployment gate:

```bash
python scripts/run_int8_deployment_gate.py
```

Run the Stage 9 INT8 rescue workflow:

```bash
python scripts/run_stage9_int8_rescue.py --no-run-qat
```

Run CPU deployment timing:

```bash
python scripts/benchmark_onnx_cpu_latency.py \
  --sample-limit 150 \
  --batch-size 1 \
  --warmup-runs 2 \
  --repeat-runs 5 \
  --intra-op-num-threads 1 \
  --inter-op-num-threads 1
```

## Study Outputs

The included `results/tables/` and `results/figures/` files are lightweight summaries from the completed study runs. They are included so users can inspect the evidence without downloading large raw outputs.

Heavy artifacts are excluded by design:

- local dataset files
- model checkpoints
- ONNX model exports
- raw prediction CSVs
- logs and intermediate training outputs

Regenerate those artifacts locally from the scripts or notebook when needed.
