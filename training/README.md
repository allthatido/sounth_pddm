# MiniCPM-V 4.6 Plant Diagnosis Fine-Tuning

This folder contains a cost-conscious Modal + SWIFT pipeline for fine-tuning
`openbmb/MiniCPM-V-4.6` on `plant_disease_training.jsonl`.

The dataset content is preserved. The converter only repackages each record into
the chat/image format expected by SWIFT.

Because the full `datasets/` tree may contain more images than this training
file uses, the pipeline copies only images referenced by the prepared
train/val/test JSONL files before uploading to Modal.

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

Run this from the repository root.

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

Then copy only the images referenced by those prepared files:

```powershell
python training/copy_prepared_images.py `
  --prepared-dir training/prepared `
  --source-root . `
  --output-dir training/prepared
```

This writes referenced images under:

```text
training/prepared/datasets/...
training/prepared/image_manifest.json
```

The converted JSONL points images at Modal paths such as:

```text
/data/datasets/plantclinic/images/frvnxc.jpg
```

So `training/prepared/datasets/...` must be uploaded to `/datasets` in the
Modal data volume.

## 3. Upload data to Modal volumes

```powershell
modal volume create --version=2 minicpmv46-plant-data
modal volume create --version=2 minicpmv46-plant-checkpoints
modal volume create --version=2 minicpmv46-hf-cache

$env:PYTHONIOENCODING="utf-8"
python training/upload_prepared_to_modal.py --volume minicpmv46-plant-data
```

Image upload can take a while the first time. Future runs reuse the Modal volume.
Volume v2 is used because this dataset has hundreds of thousands of image files.
The upload helper uses `modal volume put --force` for metadata and Modal's
batch upload API for images. Images are uploaded in file batches with retries,
existing remote files are detected before uploading, and progress is saved to
`training/prepared/modal_upload_progress.json`.

If you rerun the split or image-copy step, rerun the relevant `modal volume put`
commands so Modal sees the updated files.

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
GPU_PREFERENCE="L40S"
FALLBACK_GPU="A100-40GB"
PRECISION="bf16"
DOWNSAMPLE_MODE=4x
attn_impl="sdpa"
freeze_vit=false
num_train_epochs=1
learning_rate=2e-6
lr_scheduler_type="cosine"
warmup_ratio=0.03
per_device_train_batch_size=8
gradient_accumulation_steps=16
logging_steps=10
checkpoint_every_steps=350
evaluation_strategy="steps"
eval_every_steps=350
resume_from_checkpoint="latest"
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

## Plant-ID Data Preparation

Use this flow for `plant_id_training.jsonl`. It writes to
`training/prepared_plant_id/` and uploads to `minicpmv46-plant-id-data`, so it
does not touch the active plant disease training data or checkpoint volumes.

Convert and split plant-id examples:

```powershell
python training/convert_dataset.py `
  --input plant_id_training.jsonl `
  --output-dir training/prepared_plant_id `
  --image-root . `
  --image-prefix /data `
  --val-size 3000 `
  --test-size 1000 `
  --seed 42
```

Copy only the referenced plant-id images:

```powershell
python training/copy_prepared_images.py `
  --prepared-dir training/prepared_plant_id `
  --source-root . `
  --output-dir training/prepared_plant_id
```

Resize copied images in-place to `448x448`:

```powershell
python training/resize_prepared_images.py `
  --image-root training/prepared_plant_id/datasets `
  --size 448 `
  --manifest training/prepared_plant_id/resize_manifest.json
```

Create the dedicated Modal data volume and upload:

```powershell
modal volume create --version=2 minicpmv46-plant-id-data

$env:PYTHONIOENCODING="utf-8"
python training/upload_prepared_to_modal.py `
  --volume minicpmv46-plant-id-data `
  --prepared-dir training/prepared_plant_id `
  --progress-file training/prepared_plant_id/modal_upload_progress.json
```

Validate before upload:

```powershell
Get-Content training/prepared_plant_id/manifest.json
Get-Content training/prepared_plant_id/image_manifest.json
Get-Content training/prepared_plant_id/resize_manifest.json
```

Expected values:

```text
bad_json_records: 0
image_manifest.missing: 0
resize_manifest.failed: 0
resize_manifest.target_size: 448x448
```

Start the full plant-id training run:

```powershell
$env:PYTHONIOENCODING="utf-8"
modal run training/modal_app.py::launch_plant_id_training --run-name plant-id-v46-4x-lora --num-train-epochs 2
```

The plant-id training function mounts `minicpmv46-plant-id-data`, writes
checkpoints under `/checkpoints/plant-id-v46-4x-lora`, and defaults to
`num_train_epochs=2` with checkpoint/eval every 100 steps. Modal caps this
function at 24 hours; rerun the same command to resume from the latest
checkpoint if it times out.

## Full command sequence

```powershell
python training/convert_dataset.py `
  --input plant_disease_training.jsonl `
  --output-dir training/prepared `
  --image-root . `
  --image-prefix /data `
  --val-size 5000 `
  --test-size 2000 `
  --seed 42

python training/copy_prepared_images.py `
  --prepared-dir training/prepared `
  --source-root . `
  --output-dir training/prepared

modal volume create --version=2 minicpmv46-plant-data
modal volume create --version=2 minicpmv46-plant-checkpoints
modal volume create --version=2 minicpmv46-hf-cache

$env:PYTHONIOENCODING="utf-8"
python training/upload_prepared_to_modal.py --volume minicpmv46-plant-data

modal run training/modal_app.py::train --run-name smoke-4x --max-steps 50
modal run training/modal_app.py::train --run-name plant-v46-4x-lora
modal run training/modal_app.py::generate_for_eval --run-name plant-v46-4x-lora --max-samples 300
modal run training/modal_app.py::score_eval --run-name plant-v46-4x-lora --max-samples 300
```
