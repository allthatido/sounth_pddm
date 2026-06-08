import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def save_image(image: Image.Image, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image = image.convert("RGB")
        image.save(path, quality=95)
    else:
        image.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", default="training/prepared/datasets")
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument("--manifest", default="training/prepared/resize_manifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_root = Path(args.image_root)
    target_size = (args.size, args.size)
    manifest_path = Path(args.manifest)

    if not image_root.exists():
        raise FileNotFoundError(image_root)

    resized = 0
    already_target = 0
    failed = []
    original_sizes: dict[str, int] = {}

    for path in iter_images(image_root):
        try:
            with Image.open(path) as image:
                original_size = image.size
                original_sizes[f"{original_size[0]}x{original_size[1]}"] = (
                    original_sizes.get(f"{original_size[0]}x{original_size[1]}", 0) + 1
                )
                if original_size == target_size:
                    already_target += 1
                    continue
                resized_image = ImageOps.fit(
                    image,
                    target_size,
                    method=Image.Resampling.LANCZOS,
                    centering=(0.5, 0.5),
                )
                save_image(resized_image, path)
                resized += 1
        except Exception as exc:
            failed.append({"path": str(path), "error": str(exc)})

    manifest = {
        "image_root": str(image_root),
        "target_size": f"{target_size[0]}x{target_size[1]}",
        "resized": resized,
        "already_target": already_target,
        "failed": len(failed),
        "failed_examples": failed[:20],
        "original_sizes": original_sizes,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
