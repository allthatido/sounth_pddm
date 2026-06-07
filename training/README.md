# MiniCPM-V 4.6 Plant Diagnosis Fine-Tuning

This folder contains a cost-conscious Modal + SWIFT pipeline for fine-tuning
`openbmb/MiniCPM-V-4.6` on `plant_disease_training.jsonl`.

The dataset content is preserved. The converter only repackages each record into
the chat/image format expected by SWIFT.

## 1. Install local tools

```powershell
pip install modal
modal setup
```

If you want authenticated Hugging Face downloads or W&B logging:

```powershell
modal secret create huggingface-secret HF_TOKEN=your_hf_token
modal secret create wandb-secret WANDB_API_KEY=your_wandb_key
```

## 2. Convert and split locally

```powershell
python training/convert_dataset.py `
  --input plant_disease_training.jsonl `
  --output-dir training/prepared `
  --image-root . `
  --image-prefix /data `
  --val-size 5000 `
  --test-size 2000 `
  --seed 42
```

This writes:

```text
training/prepared/train.jsonl
training/prepared/val.jsonl
training/prepared/test.jsonl
training/prepared/manifest.json
```

## 3. Upload data to Modal volumes

```powershell
modal volume create --version=2 minicpmv46-plant-data
modal volume create --version=2 minicpmv46-plant-checkpoints
modal volume create --version=2 minicpmv46-hf-cache
modal volume put minicpmv46-plant-data training/prepared/train.jsonl /train.jsonl
modal volume put minicpmv46-plant-data training/prepared/val.jsonl /val.jsonl
modal volume put minicpmv46-plant-data training/prepared/test.jsonl /test.jsonl
modal volume put minicpmv46-plant-data training/prepared/manifest.json /manifest.json
modal volume put minicpmv46-plant-data datasets /datasets
```

Image upload can take a while the first time. Future runs reuse the Modal volume.
Volume v2 is used because this dataset has hundreds of thousands of image files.

## 4. Smoke test

```powershell
modal run training/modal_app.py::train --run-name smoke-4x --max-steps 50 --train-file /data/train.jsonl --val-file /data/val.jsonl
```

## 5. Main 4x vision-inclusive LoRA run

```powershell
modal run training/modal_app.py::train --run-name plant-v46-4x-lora
```

The training command uses:

```text
GPU preference: L40S, fallback A100-40GB
DOWNSAMPLE_MODE=4x
freeze_vit=false
LoRA
1 epoch
learning_rate=1e-5
checkpoint/eval every 1000 steps
resume from latest checkpoint when available
```

## 6. Evaluate saved checkpoints

```powershell
modal run training/modal_app.py::list_checkpoints --run-name plant-v46-4x-lora
modal run training/modal_app.py::sample_generate --run-name plant-v46-4x-lora --num-samples 8
```

## 7. Compare base vs trained with BLEU/ROUGE/MoverScore

Generate predictions for the same held-out examples from both the base model and
your trained adapter:

```powershell
modal run training/modal_app.py::generate_for_eval --run-name plant-v46-4x-lora --max-samples 300
```

Then score both outputs:

```powershell
modal run training/modal_app.py::score_eval --run-name plant-v46-4x-lora --max-samples 300
```

Metrics are written to:

```text
/checkpoints/plant-v46-4x-lora/eval/metrics.json
```

You can download them with:

```powershell
modal volume get minicpmv46-plant-checkpoints /plant-v46-4x-lora/eval/metrics.json training/metrics.json
```

## Notes

- Checkpoints live in Modal volume `minicpmv46-plant-checkpoints`.
- Data lives in Modal volume `minicpmv46-plant-data`.
- Hugging Face cache lives in Modal volume `minicpmv46-hf-cache`.
- To use W&B, set a Modal secret named `wandb-secret` with `WANDB_API_KEY`.
- If W&B is not configured, training runs with `--report_to none`.
