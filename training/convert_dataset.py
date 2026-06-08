import argparse
import hashlib
import json
import posixpath
import random
import re
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_record(
    obj: dict,
    image_root: Path,
    image_prefix: str,
    local_images: bool,
) -> dict | None:
    image = obj.get("image")
    conversations = obj.get("conversations") or []
    if not image or not conversations:
        return None

    image_path = Path(image)
    if not image_path.is_absolute():
        image_path = image_root / image_path
    if not image_path.exists():
        return None

    user_parts = []
    assistant_parts = []
    for message in conversations:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            if content != "<image>":
                user_parts.append(content)
        elif role == "assistant":
            assistant_parts.append(content)

    if not user_parts or not assistant_parts:
        return None

    user_content = "<image>\n" + "\n".join(user_parts).strip()
    assistant_content = assistant_parts[-1].strip()
    if local_images:
        out_image = str(image_path.resolve())
    else:
        out_image = posixpath.join(image_prefix.rstrip("/"), image.replace("\\", "/"))

    return {
        "id": obj.get("id"),
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "images": [out_image],
    }


def image_key(row: dict) -> str:
    image = str((row.get("images") or [""])[0]).strip().lower()
    parts = [part for part in re.split(r"[\\/]+", image) if part]
    if "datasets" in parts:
        parts = parts[parts.index("datasets") :]
    return "/".join(parts)


def dedupe_by_image(rows: list[dict]) -> tuple[list[dict], int]:
    seen = set()
    deduped = []
    removed = 0
    for row in rows:
        key = image_key(row)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        deduped.append(row)
    return deduped, removed


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="plant_disease_training.jsonl")
    parser.add_argument("--output-dir", default="training/prepared")
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-prefix", default="/data")
    parser.add_argument("--local-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    image_root = Path(args.image_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = 0
    bad_json = 0
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue
            row = normalize_record(
                obj=obj,
                image_root=image_root,
                image_prefix=args.image_prefix,
                local_images=args.local_images,
            )
            if row is None:
                skipped += 1
                continue
            rows.append(row)

    rows, duplicate_image_records = dedupe_by_image(rows)

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    val_size = min(args.val_size, len(rows))
    test_size = min(args.test_size, max(0, len(rows) - val_size))
    val_rows = rows[:val_size]
    test_rows = rows[val_size : val_size + test_size]
    train_rows = rows[val_size + test_size :]

    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "val.jsonl", val_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)

    manifest = {
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "seed": args.seed,
        "records": len(rows),
        "train_records": len(train_rows),
        "val_records": len(val_rows),
        "test_records": len(test_rows),
        "skipped_records": skipped,
        "bad_json_records": bad_json,
        "duplicate_image_records_removed": duplicate_image_records,
        "image_root": str(image_root),
        "image_prefix": args.image_prefix,
        "local_images": args.local_images,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
