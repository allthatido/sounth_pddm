import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Iterable

import modal


DEFAULT_VOLUME = "minicpmv46-plant-data"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def run_modal_put(
    volume: str,
    local_path: Path,
    remote_path: str,
    retries: int,
    retry_delay: int,
) -> None:
    cmd = [
        "modal",
        "volume",
        "put",
        "--force",
        volume,
        str(local_path),
        remote_path,
    ]
    for attempt in range(1, retries + 1):
        print(f"[{attempt}/{retries}] {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd)
        if result.returncode == 0:
            return
        if attempt == retries:
            raise SystemExit(result.returncode)
        sleep_for = retry_delay * attempt
        print(f"Upload failed; retrying in {sleep_for}s...", flush=True)
        time.sleep(sleep_for)


def upload_file_if_exists(
    volume: str,
    prepared_dir: Path,
    filename: str,
    retries: int,
    retry_delay: int,
) -> None:
    path = prepared_dir / filename
    if path.exists():
        run_modal_put(volume, path, f"/{filename}", retries, retry_delay)
    else:
        print(f"Skipping missing file: {path}", flush=True)


def remote_image_path(prepared_dir: Path, path: Path) -> str:
    relative = path.relative_to(prepared_dir).as_posix()
    return f"/{relative}"


def chunks(items: list[Path], batch_size: int) -> Iterable[list[Path]]:
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


def load_progress(path: Path, ignore_progress: bool) -> set[str]:
    if ignore_progress or not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data.get("uploaded", []))


def remote_uploaded_images(volume_name: str) -> set[str]:
    volume = modal.Volume.from_name(volume_name)
    try:
        entries = volume.listdir("/datasets", recursive=True)
    except Exception as exc:
        print(f"Could not list remote /datasets: {exc}", flush=True)
        return set()
    uploaded = {
        f"/{entry.path.lstrip('/')}"
        for entry in entries
        if Path(entry.path).suffix.lower() in IMAGE_EXTENSIONS
    }
    print(f"remote images already present: {len(uploaded)}", flush=True)
    return uploaded


def save_progress(path: Path, uploaded: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"uploaded": sorted(uploaded)}, indent=2),
        encoding="utf-8",
    )


def upload_file_batch(
    volume: modal.Volume,
    prepared_dir: Path,
    files: list[Path],
    retries: int,
    retry_delay: int,
) -> None:
    for attempt in range(1, retries + 1):
        print(
            f"[{attempt}/{retries}] uploading {len(files)} files "
            f"from {files[0].parent if files else prepared_dir}",
            flush=True,
        )
        try:
            with volume.batch_upload(force=True) as batch:
                for path in files:
                    batch.put_file(path, remote_image_path(prepared_dir, path))
            return
        except Exception as exc:
            if attempt == retries:
                raise
            sleep_for = retry_delay * attempt
            print(f"Upload failed: {exc}. Retrying in {sleep_for}s...", flush=True)
            time.sleep(sleep_for)


def upload_images_in_batches(
    volume_name: str,
    prepared_dir: Path,
    datasets_dir: Path,
    batch_size: int,
    retries: int,
    retry_delay: int,
    progress_path: Path,
    ignore_progress: bool,
    sync_remote: bool,
) -> None:
    volume = modal.Volume.from_name(volume_name)
    uploaded = load_progress(progress_path, ignore_progress)
    if sync_remote:
        uploaded.update(remote_uploaded_images(volume_name))
        save_progress(progress_path, uploaded)
    files = sorted(
        path
        for path in datasets_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    remaining = [
        path for path in files if remote_image_path(prepared_dir, path) not in uploaded
    ]
    print(
        f"image files: total={len(files)} uploaded_marker={len(uploaded)} "
        f"remaining={len(remaining)}",
        flush=True,
    )

    for batch_files in chunks(remaining, batch_size):
        upload_file_batch(volume, prepared_dir, batch_files, retries, retry_delay)
        uploaded.update(remote_image_path(prepared_dir, path) for path in batch_files)
        save_progress(progress_path, uploaded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", default=DEFAULT_VOLUME)
    parser.add_argument("--prepared-dir", default="training/prepared")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--progress-file",
        default="training/prepared/modal_upload_progress.json",
    )
    parser.add_argument("--ignore-progress", action="store_true")
    parser.add_argument(
        "--no-sync-remote",
        action="store_true",
        help="Do not list Modal /datasets before uploading.",
    )
    parser.add_argument("--skip-metadata", action="store_true")
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument(
        "--directory-mode",
        action="store_true",
        help="Use modal volume put on each top-level dataset directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared_dir = Path(args.prepared_dir)
    datasets_dir = prepared_dir / "datasets"

    if not prepared_dir.exists():
        raise FileNotFoundError(prepared_dir)

    if not args.skip_metadata:
        for filename in [
            "train.jsonl",
            "val.jsonl",
            "test.jsonl",
            "manifest.json",
            "image_manifest.json",
            "resize_manifest.json",
        ]:
            upload_file_if_exists(
                args.volume,
                prepared_dir,
                filename,
                args.retries,
                args.retry_delay,
            )

    if args.skip_images:
        return

    if not datasets_dir.exists():
        raise FileNotFoundError(datasets_dir)

    if args.directory_mode:
        for child in sorted(path for path in datasets_dir.iterdir() if path.is_dir()):
            run_modal_put(
                args.volume,
                child,
                f"/datasets/{child.name}",
                args.retries,
                args.retry_delay,
            )
        return

    upload_images_in_batches(
        volume_name=args.volume,
        prepared_dir=prepared_dir,
        datasets_dir=datasets_dir,
        batch_size=args.batch_size,
        retries=args.retries,
        retry_delay=args.retry_delay,
        progress_path=Path(args.progress_file),
        ignore_progress=args.ignore_progress,
        sync_remote=not args.no_sync_remote,
    )


if __name__ == "__main__":
    main()
