import argparse
import csv
import json
from pathlib import Path


def parse_step(value: object, fallback: object = None) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and "/" in value:
        return int(value.split("/", 1)[0])
    if isinstance(fallback, int):
        return fallback
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logging-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging_jsonl = Path(args.logging_jsonl)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with logging_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            step = parse_step(obj.get("global_step/max_steps"), obj.get("step"))
            if "loss" in obj:
                rows.append(
                    {
                        "step": step,
                        "split": "train",
                        "loss": obj.get("loss"),
                        "token_acc": obj.get("token_acc"),
                        "learning_rate": obj.get("learning_rate"),
                        "epoch": obj.get("epoch"),
                    }
                )
            if "eval_loss" in obj:
                rows.append(
                    {
                        "step": step,
                        "split": "val",
                        "loss": obj.get("eval_loss"),
                        "token_acc": obj.get("eval_token_acc"),
                        "learning_rate": "",
                        "epoch": obj.get("epoch"),
                    }
                )

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "split", "loss", "token_acc", "learning_rate", "epoch"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
