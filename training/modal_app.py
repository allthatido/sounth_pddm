import os
import shlex
import subprocess
from pathlib import Path

import modal


APP_NAME = "minicpmv46-plant-training"

DATA_VOL = modal.Volume.from_name("minicpmv46-plant-data", create_if_missing=True)
CKPT_VOL = modal.Volume.from_name("minicpmv46-plant-checkpoints", create_if_missing=True)
HF_VOL = modal.Volume.from_name("minicpmv46-hf-cache", create_if_missing=True)

DATA_DIR = Path("/data")
CKPT_DIR = Path("/checkpoints")
HF_DIR = Path("/hf-cache")
EVAL_SCRIPT = Path(__file__).parent / "evaluate_metrics.py"
MODEL_NAME = "openbmb/MiniCPM-V-4.6"


image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "git-lfs", "libgl1", "libglib2.0-0", "ninja-build")
    .pip_install("wheel")
    .pip_install(
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "datasets",
        "peft",
        "pillow",
        "wandb",
        "modelscope",
        "ms-swift[llm]",
        "qwen-vl-utils",
        "decord",
        "sacrebleu",
        "rouge-score",
        "pyemd",
    )
    .pip_install("git+https://github.com/AIPHES/emnlp19-moverscore.git")
)

app = modal.App(APP_NAME, image=image)


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(printable, flush=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(cmd, check=True, env=merged_env)


def latest_checkpoint(run_name: str) -> str | None:
    run_dir = CKPT_DIR / run_name
    if not run_dir.exists():
        return None
    checkpoints = sorted(
        [p for p in run_dir.rglob("checkpoint-*") if p.is_dir()],
        key=lambda p: (
            p.stat().st_mtime,
            int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
        ),
    )
    return str(checkpoints[-1]) if checkpoints else None


def model_cache_env() -> dict[str, str]:
    return {
        "HF_HOME": str(HF_DIR),
        "TRANSFORMERS_CACHE": str(HF_DIR / "transformers"),
        "HF_HUB_CACHE": str(HF_DIR / "hub"),
        "MODELSCOPE_CACHE": str(HF_DIR / "modelscope"),
        "MODELSCOPE_HOME": str(HF_DIR / "modelscope"),
        "TORCH_HOME": str(HF_DIR / "torch"),
        "XDG_CACHE_HOME": str(HF_DIR / "xdg"),
    }


@app.function(
    timeout=24 * 60 * 60,
    volumes={HF_DIR: HF_VOL},
)
def prefetch_model(model_name: str = MODEL_NAME) -> None:
    for path in [
        HF_DIR,
        HF_DIR / "transformers",
        HF_DIR / "hub",
        HF_DIR / "modelscope",
        HF_DIR / "torch",
        HF_DIR / "xdg",
    ]:
        path.mkdir(parents=True, exist_ok=True)

    env = model_cache_env()
    run(
        [
            "python",
            "-c",
            (
                "from huggingface_hub import snapshot_download; "
                "import os; "
                "model=os.environ['MODEL_NAME']; "
                "print(f'Prefetching {model}'); "
                "print(snapshot_download(model))"
            ),
        ],
        env={**env, "MODEL_NAME": model_name},
    )
    HF_VOL.commit()


@app.function(
    gpu=["L40S", "A100-40GB"],
    timeout=24 * 60 * 60,
    volumes={DATA_DIR: DATA_VOL, CKPT_DIR: CKPT_VOL, HF_DIR: HF_VOL},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def train(
    run_name: str = "plant-v46-4x-lora",
    train_file: str = "/data/train.jsonl",
    val_file: str = "/data/val.jsonl",
    max_steps: int = -1,
    learning_rate: str = "2e-6",
    lr_scheduler_type: str = "cosine",
    lora_rank: int = 16,
    lora_alpha: int = 32,
    per_device_train_batch_size: int = 8,
    gradient_accumulation_steps: int = 16,
    logging_steps: int = 10,
    checkpoint_every_steps: int = 350,
    eval_steps: int = 350,
    evaluation_strategy: str = "steps",
    attn_impl: str = "sdpa",
) -> None:
    output_dir = CKPT_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    report_to = "wandb" if os.environ.get("WANDB_API_KEY") else "none"
    resume = latest_checkpoint(run_name)

    env = {
        **model_cache_env(),
        "DOWNSAMPLE_MODE": "4x",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }

    cmd = [
        "swift",
        "sft",
        "--model",
        MODEL_NAME,
        "--model_type",
        "minicpmv4_6",
        "--template",
        "minicpmv4_6",
        "--dataset",
        train_file,
        "--val_dataset",
        val_file,
        "--tuner_type",
        "lora",
        "--torch_dtype",
        "bfloat16",
        "--freeze_vit",
        "false",
        "--packing",
        "false",
        "--max_length",
        "4096",
        "--per_device_train_batch_size",
        str(per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(gradient_accumulation_steps),
        "--learning_rate",
        learning_rate,
        "--lr_scheduler_type",
        lr_scheduler_type,
        "--num_train_epochs",
        "1",
        "--warmup_ratio",
        "0.03",
        "--logging_steps",
        str(logging_steps),
        "--eval_strategy",
        evaluation_strategy,
        "--save_steps",
        str(checkpoint_every_steps),
        "--eval_steps",
        str(eval_steps),
        "--save_total_limit",
        "10",
        "--attn_impl",
        attn_impl,
        "--lora_rank",
        str(lora_rank),
        "--lora_alpha",
        str(lora_alpha),
        "--output_dir",
        str(output_dir),
        "--report_to",
        report_to,
    ]
    if max_steps > 0:
        cmd.extend(["--max_steps", str(max_steps)])
    if resume:
        cmd.extend(["--resume_from_checkpoint", resume])

    try:
        run(cmd, env=env)
    finally:
        CKPT_VOL.commit()
        HF_VOL.commit()


@app.function(volumes={CKPT_DIR: CKPT_VOL}, timeout=10 * 60)
def list_checkpoints(run_name: str = "plant-v46-4x-lora") -> None:
    run_dir = CKPT_DIR / run_name
    if not run_dir.exists():
        print(f"no run directory: {run_dir}")
        return
    for path in sorted(run_dir.glob("checkpoint-*")):
        print(path)


@app.function(
    gpu=["L40S", "A100-40GB"],
    timeout=2 * 60 * 60,
    volumes={DATA_DIR: DATA_VOL, CKPT_DIR: CKPT_VOL, HF_DIR: HF_VOL},
)
def sample_generate(
    run_name: str = "plant-v46-4x-lora",
    test_file: str = "/data/test.jsonl",
    num_samples: int = 8,
) -> None:
    checkpoint = latest_checkpoint(run_name)
    if not checkpoint:
        raise RuntimeError(f"No checkpoint found for {run_name}")
    env = {
        **model_cache_env(),
        "DOWNSAMPLE_MODE": "4x",
    }
    sample_file = CKPT_DIR / run_name / "eval_sample.jsonl"
    with open(test_file, "r", encoding="utf-8") as src, sample_file.open(
        "w", encoding="utf-8"
    ) as dst:
        for idx, line in enumerate(src):
            if num_samples > 0 and idx >= num_samples:
                break
            dst.write(line)

    out_file = CKPT_DIR / run_name / "sample_generations.jsonl"
    cmd = [
        "swift",
        "infer",
        "--model",
        MODEL_NAME,
        "--adapter",
        checkpoint,
        "--model_type",
        "minicpmv4_6",
        "--template",
        "minicpmv4_6",
        "--val_dataset",
        str(sample_file),
        "--max_new_tokens",
        "512",
        "--write_batch_size",
        "1",
        "--result_path",
        str(out_file),
        "--max_batch_size",
        "1",
        "--infer_backend",
        "pt",
    ]
    run(cmd, env=env)
    CKPT_VOL.commit()
    print(f"wrote {out_file}")


@app.function(
    gpu=["L40S", "A100-40GB"],
    timeout=8 * 60 * 60,
    volumes={DATA_DIR: DATA_VOL, CKPT_DIR: CKPT_VOL, HF_DIR: HF_VOL},
)
def generate_for_eval(
    run_name: str = "plant-v46-4x-lora",
    eval_file: str = "/data/test.jsonl",
    max_samples: int = 300,
) -> None:
    checkpoint = latest_checkpoint(run_name)
    if not checkpoint:
        raise RuntimeError(f"No checkpoint found for {run_name}")

    eval_dir = CKPT_DIR / run_name / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_subset = eval_dir / f"eval_{max_samples}.jsonl"
    with open(eval_file, "r", encoding="utf-8") as src, eval_subset.open(
        "w", encoding="utf-8"
    ) as dst:
        for idx, line in enumerate(src):
            if max_samples > 0 and idx >= max_samples:
                break
            dst.write(line)

    env = {
        **model_cache_env(),
        "DOWNSAMPLE_MODE": "4x",
    }

    baseline_out = eval_dir / "baseline_predictions.jsonl"
    trained_out = eval_dir / "trained_predictions.jsonl"

    base_cmd = [
        "swift",
        "infer",
        "--model",
        MODEL_NAME,
        "--model_type",
        "minicpmv4_6",
        "--template",
        "minicpmv4_6",
        "--val_dataset",
        str(eval_subset),
        "--max_new_tokens",
        "512",
        "--write_batch_size",
        "1",
        "--result_path",
        str(baseline_out),
        "--max_batch_size",
        "1",
        "--infer_backend",
        "pt",
    ]
    trained_cmd = base_cmd.copy()
    trained_cmd.extend(["--adapter", checkpoint])
    trained_cmd[trained_cmd.index(str(baseline_out))] = str(trained_out)

    run(base_cmd, env=env)
    run(trained_cmd, env=env)
    CKPT_VOL.commit()
    print(f"baseline: {baseline_out}")
    print(f"trained: {trained_out}")


@app.function(
    timeout=4 * 60 * 60,
    volumes={DATA_DIR: DATA_VOL, CKPT_DIR: CKPT_VOL},
)
def score_eval(
    run_name: str = "plant-v46-4x-lora",
    max_samples: int = 300,
) -> None:
    eval_dir = CKPT_DIR / run_name / "eval"
    refs = eval_dir / f"eval_{max_samples}.jsonl"
    baseline = eval_dir / "baseline_predictions.jsonl"
    trained = eval_dir / "trained_predictions.jsonl"
    output = eval_dir / "metrics.json"
    for path in [refs, baseline, trained]:
        if not path.exists():
            raise FileNotFoundError(path)
    run(
        [
            "python",
            str(EVAL_SCRIPT),
            "--references",
            str(refs),
            "--baseline",
            str(baseline),
            "--trained",
            str(trained),
            "--output",
            str(output),
            "--limit",
            str(max_samples),
        ]
    )
    CKPT_VOL.commit()
    print(f"metrics: {output}")
