import argparse
import json
import shutil
import re
from pathlib import Path


def normalize_image_path(image: object) -> Path:
    image_text = str(image).strip()
    parts = [part for part in re.split(r"[\\/]+", image_text) if part]
    lowered_parts = [part.lower() for part in parts]
    if "datasets" in lowered_parts:
        parts = parts[lowered_parts.index("datasets") :]
    elif len(parts) > 1 and (Path(image_text).is_absolute() or image_text.startswith(("/", "\\"))):
        parts = parts[-1:]
    return Path(*parts)


def iter_image_paths(jsonl_paths: list[Path]) -> set[Path]:
    images: set[Path] = set()
    for jsonl_path in jsonl_paths:
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                obj = json.loads(line)
                for image in obj.get("images") or []:
                    images.add(normalize_image_path(image))
    return images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", default="training/prepared")
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--output-dir", default="training/prepared")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared_dir = Path(args.prepared_dir)
    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir)

    jsonl_paths = [
        prepared_dir / "train.jsonl",
        prepared_dir / "val.jsonl",
        prepared_dir / "test.jsonl",
    ]
    missing_jsonl = [path for path in jsonl_paths if not path.exists()]
    if missing_jsonl:
        raise FileNotFoundError(f"Missing prepared files: {missing_jsonl}")

    image_paths = iter_image_paths(jsonl_paths)
    copied = 0
    skipped_existing = 0
    missing = []

    for relative_path in sorted(image_paths):
        src = source_root / relative_path
        dst = output_dir / relative_path
        if not src.exists():
            missing.append(str(relative_path))
            continue
        if dst.exists() and not args.overwrite:
            skipped_existing += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    manifest = {
        "referenced_images": len(image_paths),
        "copied": copied,
        "skipped_existing": skipped_existing,
        "missing": len(missing),
        "missing_examples": missing[:20],
        "source_root": str(source_root),
        "output_dir": str(output_dir),
    }
    manifest_path = output_dir / "image_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
