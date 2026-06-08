import asyncio
import base64
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
import uuid
import wave
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Iterable

import gradio as gr
from fastapi import File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse


APP_ROOT = Path(__file__).parent
FRONTEND_DIR = APP_ROOT / "frontend"


def default_data_root() -> Path:
    if os.environ.get("ARANYA_DATA_ROOT"):
        return Path(os.environ["ARANYA_DATA_ROOT"])
    hf_data = Path("/data")
    if hf_data.exists() and os.access(hf_data, os.W_OK):
        return hf_data
    return APP_ROOT / "data"


DATA_ROOT = default_data_root()
DB_PATH = Path(os.environ.get("ARANYA_DB_PATH", str(DATA_ROOT / "aranya.sqlite3")))
UPLOAD_ROOT = DATA_ROOT / "uploads"
AUDIO_ROOT = DATA_ROOT / "audio"

PROMPT_VERSION = "aranya-v1"
MODE_CONFIG = {
    "identify": {
        "title": "Embark on Discovery",
        "model_path_env": "IDENTIFY_MODEL_PATH",
        "mmproj_path_env": "IDENTIFY_MMPROJ_PATH",
        "prompt": (
            "You are Aranya, a careful wildkeeper botanist. Identify the plant in the image. "
            "Give a concise field-journal answer with: common name, scientific name when likely, "
            "confidence, visible traits used, habitat/care notes, and one nature-safe action."
        ),
    },
    "health": {
        "title": "Plant Rescue",
        "model_path_env": "HEALTH_MODEL_PATH",
        "mmproj_path_env": "HEALTH_MMPROJ_PATH",
        "prompt": (
            "You are Aranya, a careful plant health rescuer. Analyze visible plant health issues "
            "from the image. Give a concise field-journal answer with: observed symptoms, likely "
            "causes, severity, rescue steps, prevention tips, and a reminder that image-only "
            "diagnosis can be uncertain."
        ),
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    for path in [DATA_ROOT, UPLOAD_ROOT, AUDIO_ROOT, DB_PATH.parent]:
        path.mkdir(parents=True, exist_ok=True)


def connect_db() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with connect_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                hf_username TEXT,
                mode TEXT NOT NULL,
                image_sha256 TEXT NOT NULL,
                image_path TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                model_path TEXT,
                mmproj_path TEXT,
                status TEXT NOT NULL,
                latency_ms INTEGER,
                error TEXT
            );

            CREATE TABLE IF NOT EXISTS outputs (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                final_text TEXT,
                parsed_json TEXT,
                audio_path TEXT
            );

            CREATE TABLE IF NOT EXISTS discoveries (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                species_common TEXT,
                species_scientific TEXT,
                health_status TEXT,
                confidence TEXT,
                thumbnail_path TEXT
            );
            """
        )


def save_upload(upload: UploadFile, mode: str) -> tuple[Path, str]:
    stamp = datetime.now(timezone.utc)
    target_dir = UPLOAD_ROOT / stamp.strftime("%Y") / stamp.strftime("%m") / stamp.strftime("%d")
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(upload.filename or "plant.jpg").suffix.lower() or ".jpg"
    target = target_dir / f"{mode}-{uuid.uuid4().hex}{suffix}"
    digest = hashlib.sha256()

    with target.open("wb") as dst:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            dst.write(chunk)

    upload.file.seek(0)
    return target, digest.hexdigest()


def detect_hf_username(
    request: Request,
    x_hf_username: str | None,
    x_forwarded_user: str | None,
) -> str | None:
    if x_hf_username:
        return x_hf_username
    if x_forwarded_user:
        return x_forwarded_user
    for key in ["x-hf-username", "x-forwarded-user", "x-huggingface-user"]:
        value = request.headers.get(key)
        if value:
            return value
    return None


def ndjson(event: dict) -> bytes:
    return (json.dumps(event, ensure_ascii=True) + "\n").encode("utf-8")


@dataclass
class NativeWorkerConfig:
    mode: str
    model_path: str | None
    mmproj_path: str | None
    executable: str | None
    ctx_size: int
    threads: int
    n_gpu_layers: int


class NativeInferenceWorker:
    """Persistent inference facade.

    The real native executable contract is line-delimited JSON:
    request: {"id", "image_path", "prompt"}
    response events: {"id", "type": "text_delta"|"done"|"error", ...}

    Until the custom libmtmd worker binary and GGUFs are present, this class
    streams a deterministic demo response so the Space remains usable.
    """

    def __init__(self, config: NativeWorkerConfig):
        self.config = config
        self.process: subprocess.Popen | None = None
        self.lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return bool(
            self.config.executable
            and self.config.model_path
            and self.config.mmproj_path
            and Path(self.config.executable).exists()
            and Path(self.config.model_path).exists()
            and Path(self.config.mmproj_path).exists()
        )

    async def start(self) -> None:
        if not self.ready or self.process:
            return
        args = [
            self.config.executable,
            "--model",
            self.config.model_path,
            "--mmproj",
            self.config.mmproj_path,
            "--ctx-size",
            str(self.config.ctx_size),
            "--threads",
            str(self.config.threads),
            "--n-gpu-layers",
            str(self.config.n_gpu_layers),
            "--jsonl",
        ]
        self.process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    async def stream(self, image_path: Path, prompt: str) -> AsyncIterator[str]:
        async with self.lock:
            await self.start()
            if self.process and self.process.stdin and self.process.stdout:
                request_id = uuid.uuid4().hex
                payload = {
                    "id": request_id,
                    "image_path": str(image_path),
                    "prompt": prompt,
                }
                self.process.stdin.write(json.dumps(payload) + "\n")
                self.process.stdin.flush()
                while True:
                    line = await asyncio.to_thread(self.process.stdout.readline)
                    if not line:
                        raise RuntimeError("Native worker stopped before completing inference.")
                    event = json.loads(line)
                    if event.get("id") != request_id:
                        continue
                    if event.get("type") == "text_delta":
                        yield str(event.get("delta", ""))
                    elif event.get("type") == "done":
                        return
                    elif event.get("type") == "error":
                        raise RuntimeError(str(event.get("message", "Native worker error")))
            else:
                for token in self.demo_response(image_path):
                    await asyncio.sleep(0.035)
                    yield token

    def demo_response(self, image_path: Path) -> Iterable[str]:
        if self.config.mode == "identify":
            text = (
                "Field journal entry: I can see enough structure to begin a discovery, "
                "but the trained MiniCPM-V GGUF is not configured yet. Once "
                "IDENTIFY_MODEL_PATH and IDENTIFY_MMPROJ_PATH point to the quantized model, "
                "this stream will report the likely species, confidence, visible traits, "
                "and care notes for the uploaded plant."
            )
        else:
            text = (
                "Rescue notes: the health-analysis GGUF is not configured yet, so this is "
                "a readiness check rather than a diagnosis. Once HEALTH_MODEL_PATH and "
                "HEALTH_MMPROJ_PATH are set, Aranya will stream symptoms, likely causes, "
                "severity, rescue steps, and prevention tips."
            )
        for word in text.split(" "):
            yield word + " "

    def close(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
        self.process = None


def build_worker(mode: str) -> NativeInferenceWorker:
    config = MODE_CONFIG[mode]
    return NativeInferenceWorker(
        NativeWorkerConfig(
            mode=mode,
            model_path=os.environ.get(config["model_path_env"]),
            mmproj_path=os.environ.get(config["mmproj_path_env"]),
            executable=os.environ.get("ARANYA_NATIVE_WORKER"),
            ctx_size=int(os.environ.get("LLAMA_CTX_SIZE", "4096")),
            threads=int(os.environ.get("LLAMA_THREADS", str(os.cpu_count() or 4))),
            n_gpu_layers=int(os.environ.get("LLAMA_N_GPU_LAYERS", "99")),
        )
    )


workers: dict[str, NativeInferenceWorker] = {}


def make_silent_wav(duration_seconds: float = 0.18, sample_rate: int = 16000) -> bytes:
    frames = int(duration_seconds * sample_rate)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        with wave.open(str(temp_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(b"\x00\x00" * frames)
        return temp_path.read_bytes()
    finally:
        temp_path.unlink(missing_ok=True)


async def synthesize_voice_chunk(text: str, voice_sample: Path | None) -> bytes:
    """VoxCPM hook.

    A real deployment should install VoxCPM and replace this fallback with the
    model call. The API has moved quickly, so this keeps the route contract
    stable while the model package is pinned during Space deployment.
    """

    _ = (text, voice_sample)
    return await asyncio.to_thread(make_silent_wav)


def sentence_ready(buffer: str) -> bool:
    return any(buffer.rstrip().endswith(mark) for mark in [".", "!", "?", "\n"])


async def run_analysis_stream(
    run_id: str,
    mode: str,
    image_path: Path,
    voice_sample_path: Path | None,
) -> AsyncIterator[bytes]:
    started = time.perf_counter()
    worker = workers[mode]
    final_text = ""
    voice_buffer = ""
    audio_file = AUDIO_ROOT / f"{run_id}.wav"
    AUDIO_ROOT.mkdir(parents=True, exist_ok=True)

    yield ndjson({"type": "status", "message": "Opening the Wildkeeper journal..."})
    try:
        prompt = MODE_CONFIG[mode]["prompt"]
        async for delta in worker.stream(image_path, prompt):
            final_text += delta
            voice_buffer += delta
            yield ndjson({"type": "text_delta", "delta": delta})
            if sentence_ready(voice_buffer) and len(voice_buffer.strip()) > 24:
                chunk = await synthesize_voice_chunk(voice_buffer, voice_sample_path)
                with audio_file.open("ab") as dst:
                    dst.write(chunk)
                yield ndjson(
                    {
                        "type": "audio_chunk",
                        "mime_type": "audio/wav",
                        "data": base64.b64encode(chunk).decode("ascii"),
                    }
                )
                voice_buffer = ""

        if voice_buffer.strip():
            chunk = await synthesize_voice_chunk(voice_buffer, voice_sample_path)
            with audio_file.open("ab") as dst:
                dst.write(chunk)
            yield ndjson(
                {
                    "type": "audio_chunk",
                    "mime_type": "audio/wav",
                    "data": base64.b64encode(chunk).decode("ascii"),
                }
            )

        latency_ms = int((time.perf_counter() - started) * 1000)
        with connect_db() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, latency_ms = ? WHERE id = ?",
                ("complete", latency_ms, run_id),
            )
            conn.execute(
                "INSERT INTO outputs (id, run_id, final_text, parsed_json, audio_path) VALUES (?, ?, ?, ?, ?)",
                (uuid.uuid4().hex, run_id, final_text, None, str(audio_file)),
            )
            conn.execute(
                """
                INSERT INTO discoveries (
                    id, run_id, species_common, species_scientific, health_status, confidence, thumbnail_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    run_id,
                    "Pending model result" if mode == "identify" else None,
                    None,
                    "Pending model result" if mode == "health" else None,
                    None,
                    str(image_path),
                ),
            )
        yield ndjson({"type": "record_saved", "run_id": run_id})
        yield ndjson({"type": "done", "run_id": run_id})
    except Exception as exc:
        with connect_db() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, error = ? WHERE id = ?",
                ("error", str(exc), run_id),
            )
        yield ndjson({"type": "error", "message": str(exc)})


@asynccontextmanager
async def lifespan(_: gr.Server):
    ensure_dirs()
    init_db()
    workers.update({mode: build_worker(mode) for mode in MODE_CONFIG})
    for worker in workers.values():
        await worker.start()
    try:
        yield
    finally:
        for worker in workers.values():
            worker.close()


app = gr.Server(lifespan=lifespan)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse((FRONTEND_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/frontend/{path:path}")
async def frontend_asset(path: str) -> FileResponse:
    target = FRONTEND_DIR / path
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(target)


@app.get("/api/status")
async def api_status() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ready",
            "db_path": str(DB_PATH),
            "workers": {
                mode: {
                    "native_ready": worker.ready,
                    "model_path": worker.config.model_path,
                    "mmproj_path": worker.config.mmproj_path,
                }
                for mode, worker in workers.items()
            },
        }
    )


@app.get("/api/journal")
async def api_journal() -> JSONResponse:
    with connect_db() as conn:
        stats = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN mode = 'identify' AND status = 'complete' THEN 1 ELSE 0 END) AS species,
                SUM(CASE WHEN mode = 'health' AND status = 'complete' THEN 1 ELSE 0 END) AS rescues
            FROM runs
            """
        ).fetchone()
        rows = conn.execute(
            """
            SELECT r.id, r.created_at, r.mode, r.status, d.species_common, d.health_status, d.thumbnail_path
            FROM runs r
            LEFT JOIN discoveries d ON d.run_id = r.id
            ORDER BY r.created_at DESC
            LIMIT 6
            """
        ).fetchall()
    return JSONResponse(
        {
            "stats": {
                "total": stats["total"] or 0,
                "species": stats["species"] or 0,
                "rescues": stats["rescues"] or 0,
            },
            "recent": [dict(row) for row in rows],
        }
    )


@app.post("/api/run")
async def api_run(
    request: Request,
    mode: str = Form(...),
    image: UploadFile = File(...),
    voice_sample: UploadFile | None = File(default=None),
    x_hf_username: str | None = Header(default=None),
    x_forwarded_user: str | None = Header(default=None),
) -> StreamingResponse:
    if mode not in MODE_CONFIG:
        raise HTTPException(status_code=400, detail="mode must be identify or health")
    run_id = uuid.uuid4().hex
    image_path, image_sha256 = save_upload(image, mode)
    voice_sample_path = None
    if voice_sample and voice_sample.filename:
        voice_sample_path, _ = save_upload(voice_sample, f"{mode}-voice")
    username = detect_hf_username(request, x_hf_username, x_forwarded_user)
    worker = workers.get(mode) or build_worker(mode)
    workers[mode] = worker

    with connect_db() as conn:
        conn.execute(
            """
            INSERT INTO runs (
                id, created_at, hf_username, mode, image_sha256, image_path,
                prompt_version, model_path, mmproj_path, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                utc_now(),
                username,
                mode,
                image_sha256,
                str(image_path),
                PROMPT_VERSION,
                worker.config.model_path,
                worker.config.mmproj_path,
                "running",
            ),
        )

    return StreamingResponse(
        run_analysis_stream(run_id, mode, image_path, voice_sample_path),
        media_type="application/x-ndjson",
    )


@app.api(name="identify")
async def identify_api(image_path: str) -> str:
    path = Path(image_path)
    text = ""
    async for delta in workers["identify"].stream(path, MODE_CONFIG["identify"]["prompt"]):
        text += delta
    return text


@app.api(name="health")
async def health_api(image_path: str) -> str:
    path = Path(image_path)
    text = ""
    async for delta in workers["health"].stream(path, MODE_CONFIG["health"]["prompt"]):
        text += delta
    return text


if __name__ == "__main__":
    app.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
    )
