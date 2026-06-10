import asyncio
import base64
import hashlib
import io
import json
import logging
import mimetypes
import os
import site
import socket
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Iterable

import gradio as gr
from fastapi import File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from PIL import Image, ImageOps

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # Keeps local/demo runs working without the optional downloader.
    hf_hub_download = None

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

if os.name == "nt":
    dll_candidate_roots = []
    try:
        dll_candidate_roots.extend(site.getsitepackages())
    except Exception:
        pass
    dll_candidate_roots.extend(sys.path)
    dll_dirs: list[str] = []
    for root in dll_candidate_roots:
        if not root:
            continue
        root_path = Path(root)
        dll_dirs.extend(
            [
                str(root_path / "llama_cpp" / "lib"),
                str(root_path / "bin"),
            ]
        )
        nvidia_root = root_path / "nvidia"
        if nvidia_root.exists():
            dll_dirs.extend(str(path) for path in nvidia_root.glob("**/bin") if path.is_dir())
    for env_name in ["CUDA_PATH", "CUDA_PATH_V12_4", "CUDA_PATH_V13_2", "CUDA_PATH_V13_3"]:
        cuda_root = os.environ.get(env_name)
        if cuda_root:
            dll_dirs.append(str(Path(cuda_root) / "bin"))
    for dll_dir in dict.fromkeys(dll_dirs):
        if Path(dll_dir).exists():
            try:
                os.add_dll_directory(dll_dir)
            except Exception:
                pass

try:
    from llama_cpp import Llama
    import llama_cpp.llama_chat_format as llama_chat_format
    from llama_cpp import llama_supports_gpu_offload
    LLAMA_CPP_IMPORT_ERROR = None
except Exception as exc:
    Llama = None
    llama_chat_format = None
    llama_supports_gpu_offload = None
    LLAMA_CPP_IMPORT_ERROR = f"llama-cpp-python is not installed or could not be imported: {exc}"


APP_ROOT = Path(__file__).parent
FRONTEND_DIR = APP_ROOT / "frontend"
logger = logging.getLogger("aranya")


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
MODEL_CACHE_ROOT = Path(os.environ.get("ARANYA_MODEL_CACHE_DIR", str(DATA_ROOT / "models")))
DEFAULT_VOICE_SAMPLE_MP3 = FRONTEND_DIR / "assets" / "voice_sample.mp3"
DEFAULT_VOICE_SAMPLE_WAV = FRONTEND_DIR / "assets" / "voice_prompt.wav"
VOICE_PROMPT_WAV = AUDIO_ROOT / "voice_prompt.wav"

DEFAULT_VLM_REPO = "openbmb/MiniCPM-V-4.6-gguf"
DEFAULT_VLM_FILE = "MiniCPM-V-4_6-Q8_0.gguf"
DEFAULT_VLM_MMPROJ_FILE = "mmproj-model-f16.gguf"
DEFAULT_TTS_REPO = "bluryar/VoxCPM-GGUF"
DEFAULT_TTS_FILE = "voxcpm-0.5b-q4_k-audiovae-f16.gguf"
DEFAULT_TTS_PROMPT_TEXT = "Nature once determined how we survive, now we determine how nature survives"
DEFAULT_VOXCPM_CPP_REPO = "https://github.com/bluryar/VoxCPM.cpp.git"
BLOCKING_MODEL_STARTUP = os.environ.get("ARANYA_BLOCKING_MODEL_STARTUP", "1") == "1"
MODEL_IMAGE_SIZE = 448
DB_INITIALIZED = False
REQUIRE_TTS = os.environ.get("ARANYA_REQUIRE_TTS", "1") != "0"
REQUIRE_LLAMA_GPU = os.environ.get("ARANYA_REQUIRE_LLAMA_GPU", "0") == "1"
REQUIRE_TTS_GPU = os.environ.get("ARANYA_REQUIRE_TTS_GPU", "0") == "1"
VOXCPM_BUILD_ATTEMPTED: set[str] = set()

PROMPT_VERSION = "aranya-v1"
MODE_CONFIG = {
    "identify": {
        "title": "Embark on Discovery",
        "model_path_env": "IDENTIFY_MODEL_PATH",
        "mmproj_path_env": "IDENTIFY_MMPROJ_PATH",
        "model_repo_env": "IDENTIFY_MODEL_REPO",
        "model_file_env": "IDENTIFY_MODEL_FILE",
        "mmproj_file_env": "IDENTIFY_MMPROJ_FILE",
        "prompt": (
            "You are Aranya, a careful wildkeeper botanist. Identify the plant in this image. "
            "Write a short field-journal answer in natural spoken prose."
        ),
    },
    "health": {
        "title": "Plant Rescue",
        "model_path_env": "HEALTH_MODEL_PATH",
        "mmproj_path_env": "HEALTH_MMPROJ_PATH",
        "model_repo_env": "HEALTH_MODEL_REPO",
        "model_file_env": "HEALTH_MODEL_FILE",
        "mmproj_file_env": "HEALTH_MMPROJ_FILE",
        "prompt": (
            "You are Aranya, a careful plant health rescuer. Analyze visible plant health issues "
            "from this image. Write a short field-journal answer in natural spoken prose. Include observed symptoms, likely causes, severity, immediate rescue steps, "
            "prevention tips"
        ),
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    for path in [DATA_ROOT, UPLOAD_ROOT, AUDIO_ROOT, MODEL_CACHE_ROOT, DB_PATH.parent]:
        path.mkdir(parents=True, exist_ok=True)


def connect_db() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    global DB_INITIALIZED
    if DB_INITIALIZED:
        return
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
        """
    )
    DB_INITIALIZED = True


def init_db() -> None:
    with connect_db():
        pass


def normalized_upload_bytes(upload: UploadFile) -> bytes:
    try:
        upload.file.seek(0)
        with Image.open(upload.file) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size
            side = min(width, height)
            left = (width - side) // 2
            top = (height - side) // 2
            image = image.crop((left, top, left + side, top + side))
            image = image.resize((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), Image.Resampling.LANCZOS)

            if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image.convert("RGBA"), mask=image.convert("RGBA").getchannel("A"))
                image = background
            else:
                image = image.convert("RGB")

            output = io.BytesIO()
            image.save(output, format="JPEG", quality=92, optimize=True)
            return output.getvalue()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image upload: {exc}") from exc
    finally:
        upload.file.seek(0)


def save_upload(upload: UploadFile, mode: str) -> tuple[Path, str]:
    stamp = datetime.now(timezone.utc)
    target_dir = UPLOAD_ROOT / stamp.strftime("%Y") / stamp.strftime("%m") / stamp.strftime("%d")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{mode}-{uuid.uuid4().hex}.jpg"
    image_bytes = normalized_upload_bytes(upload)
    digest = hashlib.sha256(image_bytes).hexdigest()
    target.write_bytes(image_bytes)
    return target, digest


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


_STREAM_DONE = object()


def next_stream_chunk(iterator: Iterable) -> object:
    try:
        return next(iterator)
    except StopIteration:
        return _STREAM_DONE


def image_data_uri(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def host_gpu_available() -> bool:
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {"", "-1"}:
        return False
    if os.environ.get("NVIDIA_VISIBLE_DEVICES") not in {None, "", "void", "none"}:
        return True
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    try:
        result = subprocess.run(
            [nvidia_smi, "-L"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def nvidia_smi_summary() -> str | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        return f"nvidia-smi failed: {exc}"
    if result.returncode != 0:
        return (result.stderr or result.stdout).strip() or "nvidia-smi returned a non-zero status."
    return result.stdout.strip()


def cuda_toolkit_summary() -> dict:
    summary = {
        "nvcc": shutil.which("nvcc"),
        "cl": shutil.which("cl") if os.name == "nt" else shutil.which("cc"),
        "cuda_path": os.environ.get("CUDA_PATH"),
        "cuda_path_v12_4": os.environ.get("CUDA_PATH_V12_4"),
        "cuda_path_v13_2": os.environ.get("CUDA_PATH_V13_2"),
        "cuda_path_v13_3": os.environ.get("CUDA_PATH_V13_3"),
        "cudart64_12": shutil.which("cudart64_12.dll") if os.name == "nt" else None,
        "cudart64_13": shutil.which("cudart64_13.dll") if os.name == "nt" else None,
        "cublas64_12": shutil.which("cublas64_12.dll") if os.name == "nt" else None,
        "cublas64_13": shutil.which("cublas64_13.dll") if os.name == "nt" else None,
    }
    if summary["nvcc"]:
        try:
            result = subprocess.run(
                [summary["nvcc"], "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
                check=False,
            )
            summary["nvcc_version"] = (result.stdout or result.stderr).strip()
        except Exception as exc:
            summary["nvcc_version"] = f"nvcc failed: {exc}"
    return summary


def llama_gpu_offload_supported() -> bool:
    if llama_supports_gpu_offload is None:
        return False
    try:
        return bool(llama_supports_gpu_offload())
    except Exception:
        return False


def default_llama_gpu_layers() -> int:
    if os.environ.get("LLAMA_N_GPU_LAYERS") is not None:
        return int(os.environ["LLAMA_N_GPU_LAYERS"])
    return 99 if host_gpu_available() and llama_gpu_offload_supported() else 0


def voxcpm_cuda_build_enabled() -> bool:
    configured = os.environ.get("VOXCPM_BUILD_CUDA")
    if configured is not None:
        return configured == "1"
    return host_gpu_available()


def default_tts_backend(executable: str | None = None) -> str:
    if os.environ.get("TTS_BACKEND"):
        return os.environ["TTS_BACKEND"]
    if executable and "build-cuda" in str(executable).lower():
        return "cuda"
    return "cpu"


def voxcpm_executable_candidates(binary_name: str) -> list[Path]:
    executable_name = f"{binary_name}.exe" if os.name == "nt" else binary_name
    return [
        APP_ROOT / "VoxCPM.cpp" / "build-cuda" / "examples" / "Release" / executable_name,
        APP_ROOT / "VoxCPM.cpp" / "build-cuda" / "examples" / executable_name,
        APP_ROOT / "VoxCPM.cpp" / "build" / "examples" / "Release" / executable_name,
        APP_ROOT / "VoxCPM.cpp" / "build" / "examples" / executable_name,
        APP_ROOT / "bin" / executable_name,
    ]


def voxcpm_cuda_executable_candidates(binary_name: str) -> list[Path]:
    executable_name = f"{binary_name}.exe" if os.name == "nt" else binary_name
    return [
        APP_ROOT / "VoxCPM.cpp" / "build-cuda" / "examples" / "Release" / executable_name,
        APP_ROOT / "VoxCPM.cpp" / "build-cuda" / "examples" / executable_name,
    ]


def existing_voxcpm_executable(binary_name: str) -> str | None:
    path_match = shutil.which(binary_name)
    if path_match:
        return path_match
    for candidate in voxcpm_executable_candidates(binary_name):
        if candidate.exists():
            return str(candidate)
    return None


def existing_voxcpm_cuda_executable(binary_name: str) -> str | None:
    for candidate in voxcpm_cuda_executable_candidates(binary_name):
        if candidate.exists():
            return str(candidate)
    return None


def ensure_voxcpm_source() -> Path | None:
    source_dir = APP_ROOT / "VoxCPM.cpp"
    if (source_dir / "CMakeLists.txt").exists():
        return source_dir
    git = shutil.which("git")
    if not git:
        logger.warning("git is unavailable; cannot clone VoxCPM.cpp.")
        return None
    repo_url = os.environ.get("VOXCPM_CPP_REPO", DEFAULT_VOXCPM_CPP_REPO)
    result = subprocess.run(
        [git, "clone", "--depth", "1", repo_url, str(source_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        logger.warning("Could not clone VoxCPM.cpp: %s", result.stderr.strip() or result.stdout.strip())
        return None
    return source_dir if (source_dir / "CMakeLists.txt").exists() else None


def build_voxcpm_executable(binary_name: str, target_name: str, build_cuda: bool | None = None) -> str | None:
    global VOXCPM_BUILD_ATTEMPTED
    if build_cuda is None:
        build_cuda = voxcpm_cuda_build_enabled()
    build_key = "cuda" if build_cuda else "cpu"
    if build_key in VOXCPM_BUILD_ATTEMPTED or os.environ.get("ARANYA_BUILD_TTS_WORKER", "1") == "0":
        return None
    VOXCPM_BUILD_ATTEMPTED.add(build_key)

    cmake = shutil.which("cmake")
    if not cmake:
        logger.warning("cmake is unavailable; cannot build VoxCPM.cpp.")
        return None
    source_dir = ensure_voxcpm_source()
    if not source_dir:
        return None

    build_dir = source_dir / ("build-cuda" if build_cuda else "build")
    build_dir.mkdir(parents=True, exist_ok=True)
    cuda_enabled = "ON" if build_cuda else "OFF"
    configure_args = [
        cmake,
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-DVOXCPM_BUILD_TESTS=OFF",
        "-DVOXCPM_BUILD_BENCHMARK=OFF",
        f"-DVOXCPM_CUDA={cuda_enabled}",
    ]
    if build_cuda:
        configure_args.append(f"-DCMAKE_CUDA_ARCHITECTURES={os.environ.get('VOXCPM_CUDA_ARCHITECTURES', '89')}")
    configure = subprocess.run(
        configure_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if configure.returncode != 0:
        logger.warning("VoxCPM.cpp configure failed: %s", configure.stderr.strip() or configure.stdout.strip())
        return None

    jobs = os.environ.get("VOXCPM_BUILD_JOBS", str(max(1, min(os.cpu_count() or 2, 8))))
    build = subprocess.run(
        [cmake, "--build", str(build_dir), "--config", "Release", "--target", target_name, "-j", jobs],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if build.returncode != 0:
        logger.warning("VoxCPM.cpp build failed: %s", build.stderr.strip() or build.stdout.strip())
        return None
    return existing_voxcpm_executable(binary_name)


def find_voxcpm_server_executable() -> str | None:
    configured = os.environ.get("ARANYA_TTS_WORKER")
    if configured:
        return configured
    if voxcpm_cuda_build_enabled():
        return (
            existing_voxcpm_cuda_executable("voxcpm-server")
            or build_voxcpm_executable("voxcpm-server", "voxcpm-server", build_cuda=True)
            or existing_voxcpm_executable("voxcpm-server")
            or build_voxcpm_executable("voxcpm-server", "voxcpm-server", build_cuda=False)
        )
    return existing_voxcpm_executable("voxcpm-server") or build_voxcpm_executable(
        "voxcpm-server", "voxcpm-server", build_cuda=False
    )



def resolve_model_path(
    *,
    path_env_value: str | None,
    repo_id: str,
    filename: str,
    cache_dir: Path,
) -> str | None:
    if path_env_value:
        return path_env_value
    if not filename:
        return None

    candidate = cache_dir / repo_id.replace("/", "--") / filename
    if candidate.exists():
        return str(candidate)

    if hf_hub_download is None:
        return None

    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(cache_dir),
            local_dir=str(cache_dir / repo_id.replace("/", "--")),
            local_dir_use_symlinks=False,
        )
    except Exception as exc:
        logger.warning("Could not resolve %s/%s from Hugging Face: %s", repo_id, filename, exc)
        return None


def source_voice_sample_path() -> Path | None:
    if DEFAULT_VOICE_SAMPLE_MP3.exists():
        return DEFAULT_VOICE_SAMPLE_MP3
    if DEFAULT_VOICE_SAMPLE_WAV.exists():
        return DEFAULT_VOICE_SAMPLE_WAV
    return None


def convert_voice_sample_to_wav() -> Path | None:
    if VOICE_PROMPT_WAV.exists():
        return VOICE_PROMPT_WAV
    source = source_voice_sample_path()
    if not source:
        return None
    if source.suffix.lower() == ".wav":
        return source

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg and imageio_ffmpeg is not None:
        try:
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            logger.warning("Could not load bundled ffmpeg: %s", exc)
    if not ffmpeg:
        logger.warning("ffmpeg is unavailable; cannot convert %s to WAV.", source)
        return None

    VOICE_PROMPT_WAV.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(source),
            "-ar",
            "24000",
            "-ac",
            "1",
            str(VOICE_PROMPT_WAV),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode == 0 and VOICE_PROMPT_WAV.exists():
        return VOICE_PROMPT_WAV
    logger.warning("ffmpeg failed to convert %s to %s.", source, VOICE_PROMPT_WAV)
    return None


@dataclass
class NativeWorkerConfig:
    mode: str
    model_repo: str
    model_file: str
    model_path: str | None
    mmproj_path: str | None
    ctx_size: int
    threads: int
    n_gpu_layers: int


class NativeInferenceWorker:
    """Persistent in-process llama-cpp-python inference facade."""

    _shared_models: dict[tuple[str, str | None], tuple[object, asyncio.Lock]] = {}
    _shared_load_locks: dict[tuple[str, str | None], asyncio.Lock] = {}

    def __init__(self, config: NativeWorkerConfig):
        self.config = config
        self.llm: object | None = None
        self.llm_lock: asyncio.Lock | None = None
        self.lock = asyncio.Lock()
        self.start_lock = asyncio.Lock()
        self.last_error: str | None = None

    @property
    def ready(self) -> bool:
        if not self.config.model_path or not Path(self.config.model_path).exists():
            return False
        if self.config.mmproj_path and not Path(self.config.mmproj_path).exists():
            return False
        return bool(Llama is not None)

    @property
    def loaded(self) -> bool:
        return bool(self.llm)

    async def start(self) -> None:
        async with self.start_lock:
            if not self.ready or self.loaded:
                return
            if REQUIRE_LLAMA_GPU and self.config.n_gpu_layers <= 0:
                self.last_error = (
                    "ARANYA_REQUIRE_LLAMA_GPU=1 but llama-cpp-python does not report GPU offload support. "
                    "Install a CUDA-enabled llama-cpp-python wheel or build it with GGML_CUDA=on."
                )
                raise RuntimeError(self.last_error)
            try:
                await self.start_llama_cpp()
                self.last_error = None
            except Exception as exc:
                self.last_error = str(exc)
                raise

    async def start_llama_cpp(self) -> None:
        if not self.config.model_path:
            return
        cache_key = (self.config.model_path, self.config.mmproj_path)
        shared = self._shared_models.get(cache_key)
        if shared:
            self.llm, self.llm_lock = shared
            return
        load_lock = self._shared_load_locks.setdefault(cache_key, asyncio.Lock())
        async with load_lock:
            shared = self._shared_models.get(cache_key)
            if shared:
                self.llm, self.llm_lock = shared
                return
            llm, llm_lock = await asyncio.to_thread(self.load_llama_cpp)
            self._shared_models[cache_key] = (llm, llm_lock)
            self.llm = llm
            self.llm_lock = llm_lock

    def load_llama_cpp(self) -> tuple[object, asyncio.Lock]:
        if Llama is None:
            raise RuntimeError("llama-cpp-python is not installed.")
        kwargs = {
            "model_path": self.config.model_path,
            "n_ctx": self.config.ctx_size,
            "n_threads": self.config.threads,
            "n_gpu_layers": self.config.n_gpu_layers,
            "verbose": os.environ.get("LLAMA_VERBOSE", "0") == "1",
        }
        chat_handler = self.build_chat_handler()
        if chat_handler:
            kwargs["chat_handler"] = chat_handler
        chat_format = os.environ.get("LLAMA_CHAT_FORMAT")
        if chat_format:
            kwargs["chat_format"] = chat_format
        llm = Llama(**kwargs)
        return llm, asyncio.Lock()

    def build_chat_handler(self) -> object | None:
        if not self.config.mmproj_path or llama_chat_format is None:
            return None
        for name in [
            os.environ.get("LLAMA_CHAT_HANDLER", ""),
            "MiniCPMv26ChatHandler",
            "MiniCPMv25ChatHandler",
            "Llava15ChatHandler",
        ]:
            if not name:
                continue
            handler_class = getattr(llama_chat_format, name, None)
            if handler_class:
                return handler_class(clip_model_path=self.config.mmproj_path)
        return None

    async def stream(self, image_path: Path, prompt: str) -> AsyncIterator[str]:
        async with self.lock:
            await self.start()
            if self.llm and self.llm_lock:
                async for delta in self.stream_llama_cpp(image_path, prompt):
                    yield delta
                return
            detail = self.last_error or LLAMA_CPP_IMPORT_ERROR or "Unknown runtime load failure."
            raise RuntimeError(f"MiniCPM-V model is resolved but no GGUF runtime could be loaded: {detail}")

    async def stream_llama_cpp(self, image_path: Path, prompt: str) -> AsyncIterator[str]:
        image_uri = await asyncio.to_thread(image_data_uri, image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        max_tokens = int(os.environ.get("LLAMA_MAX_TOKENS", "384"))
        temperature = float(os.environ.get("LLAMA_TEMPERATURE", "0.2"))

        async with self.llm_lock:
            stream = await asyncio.to_thread(
                self.llm.create_chat_completion,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            iterator = iter(stream)
            while True:
                chunk = await asyncio.to_thread(next_stream_chunk, iterator)
                if chunk is _STREAM_DONE:
                    return
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                text = delta.get("content")
                if text:
                    yield str(text)

    def close(self) -> None:
        self.llm = None
        self.llm_lock = None


def build_worker(mode: str) -> NativeInferenceWorker:
    config = MODE_CONFIG[mode]
    model_repo = os.environ.get(config["model_repo_env"], DEFAULT_VLM_REPO)
    model_file = os.environ.get(config["model_file_env"], DEFAULT_VLM_FILE)
    model_path = resolve_model_path(
        path_env_value=os.environ.get(config["model_path_env"]),
        repo_id=model_repo,
        filename=model_file,
        cache_dir=MODEL_CACHE_ROOT,
    )
    mmproj_file = os.environ.get(config["mmproj_file_env"], DEFAULT_VLM_MMPROJ_FILE)
    mmproj_path = resolve_model_path(
        path_env_value=os.environ.get(config["mmproj_path_env"]),
        repo_id=model_repo,
        filename=mmproj_file,
        cache_dir=MODEL_CACHE_ROOT,
    )
    return NativeInferenceWorker(
        NativeWorkerConfig(
            mode=mode,
            model_repo=model_repo,
            model_file=model_file,
            model_path=model_path,
            mmproj_path=mmproj_path,
            ctx_size=int(os.environ.get("LLAMA_CTX_SIZE", "4096")),
            threads=int(os.environ.get("LLAMA_THREADS", str(os.cpu_count() or 4))),
            n_gpu_layers=default_llama_gpu_layers(),
        )
    )


workers: dict[str, NativeInferenceWorker] = {}
startup_state = {
    "phase": "not_started",
    "message": "Startup has not begun.",
    "started_at": None,
    "finished_at": None,
}
warmup_task: asyncio.Task | None = None
app_started = False


def set_startup_state(phase: str, message: str, finished: bool = False) -> None:
    startup_state["phase"] = phase
    startup_state["message"] = message
    if startup_state["started_at"] is None:
        startup_state["started_at"] = utc_now()
    if finished:
        startup_state["finished_at"] = utc_now()


async def warm_models() -> None:
    set_startup_state("loading", "Loading MiniCPM-V and TTS models.")
    try:
        for mode, worker in workers.items():
            set_startup_state("loading", f"Loading {mode} MiniCPM-V model.")
            await worker.start()
            if not worker.loaded:
                raise RuntimeError(
                    f"{mode} model did not load. model={worker.config.model_path}, "
                    f"mmproj={worker.config.mmproj_path}, error={worker.last_error}"
                )

        set_startup_state("loading", "Preparing VoxCPM TTS model.")
        await ensure_tts_worker()
        if REQUIRE_TTS and (not tts_worker or not tts_worker.ready):
            raise RuntimeError((tts_worker.last_error if tts_worker else None) or "VoxCPM TTS runtime is not ready.")

        message = "MiniCPM-V is loaded. TTS is ready." if tts_worker.ready else "MiniCPM-V is loaded. TTS runtime is unavailable."
        set_startup_state("ready", message, finished=True)
    except Exception as exc:
        logger.exception("Model warmup failed")
        set_startup_state("error", str(exc), finished=True)


@dataclass
class TtsWorkerConfig:
    model_repo: str
    model_file: str
    model_path: str | None
    executable: str | None
    prompt_audio_path: Path | None
    prompt_text: str
    voice_id: str
    model_name: str
    host: str
    port: int
    threads: int
    backend: str
    response_format: str
    output_sample_rate: int
    max_queue: int
    inference_timesteps: int
    cfg_value: float
    retry_badcase: bool
    retry_badcase_max_times: int
    retry_badcase_ratio_threshold: float


@dataclass
class TtsSynthesisResult:
    data: bytes
    mime_type: str
    audio_format: str
    sample_rate: int


class NativeTtsWorker:
    """Persistent VoxCPM.cpp server TTS facade."""

    def __init__(self, config: TtsWorkerConfig):
        self.config = config
        self.process: subprocess.Popen | None = None
        self.lock = asyncio.Lock()
        self.start_lock = asyncio.Lock()
        self.last_error: str | None = None
        self.voice_registered = False
        self.log_path = AUDIO_ROOT / "voxcpm-server.log"
        self.log_handle = None

    @property
    def ready(self) -> bool:
        return bool(
            self.config.executable
            and self.config.model_path
            and self.config.prompt_audio_path
            and self.config.prompt_text.strip()
            and Path(self.config.executable).exists()
            and Path(self.config.model_path).exists()
            and Path(self.config.prompt_audio_path).exists()
            and self.process
            and self.process.poll() is None
            and self.voice_registered
        )

    async def start(self) -> None:
        async with self.start_lock:
            if self.ready:
                return
            if not self.config.executable or not Path(self.config.executable).exists():
                self.last_error = self.missing_reason()
                return
            if not self.config.model_path or not Path(self.config.model_path).exists():
                self.last_error = self.missing_reason()
                return
            if not self.config.prompt_audio_path or not Path(self.config.prompt_audio_path).exists():
                self.last_error = self.missing_reason()
                return
            if not self.config.prompt_text.strip():
                self.last_error = self.missing_reason()
                return
            if REQUIRE_TTS_GPU and self.config.backend != "cuda":
                self.last_error = (
                    "ARANYA_REQUIRE_TTS_GPU=1 but the selected VoxCPM backend is not cuda. "
                    "Build hf_space/VoxCPM.cpp/build-cuda or set ARANYA_TTS_WORKER to a CUDA voxcpm-server."
                )
                return
            await asyncio.to_thread(self.start_server_and_register_voice)

    def missing_reason(self) -> str:
        if not self.config.executable:
            return "VoxCPM.cpp executable `voxcpm-server` was not found. Build VoxCPM.cpp or set ARANYA_TTS_WORKER."
        if not Path(self.config.executable).exists():
            return f"VoxCPM.cpp executable does not exist: {self.config.executable}"
        if not self.config.model_path or not Path(self.config.model_path).exists():
            return f"VoxCPM GGUF model is missing: {self.config.model_path}"
        if not self.config.prompt_audio_path or not Path(self.config.prompt_audio_path).exists():
            return f"Voice sample prompt audio is missing: {self.config.prompt_audio_path}"
        if not self.config.prompt_text.strip():
            return "TTS_PROMPT_TEXT is required for VoxCPM voice cloning."
        if REQUIRE_TTS_GPU and self.config.backend != "cuda":
            return (
                "ARANYA_REQUIRE_TTS_GPU=1 but the selected VoxCPM backend is not cuda. "
                "Build hf_space/VoxCPM.cpp/build-cuda or set ARANYA_TTS_WORKER to a CUDA voxcpm-server."
            )
        return "VoxCPM TTS runtime is not ready."

    def start_server_and_register_voice(self) -> None:
        if not self.process or self.process.poll() is not None:
            self.process = self.launch_server()
        self.wait_for_health()
        self.register_voice()
        self.last_error = None
        self.voice_registered = True

    def launch_server(self) -> subprocess.Popen:
        voice_dir = DATA_ROOT / "voices"
        voice_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.log_handle:
            self.log_handle.close()
        self.log_handle = self.log_path.open("ab")
        args = [
            self.config.executable,
            "--model-path",
            self.config.model_path,
            "--model-name",
            self.config.model_name,
            "--voice-dir",
            str(voice_dir),
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
            "--backend",
            self.config.backend,
            "--threads",
            str(self.config.threads),
            "--max-queue",
            str(self.config.max_queue),
            "--output-sample-rate",
            str(self.config.output_sample_rate),
            "--disable-auth",
        ]
        env = os.environ.copy()
        env.setdefault("VOXCPM_CPU_SHORT_DECODE_CAP", os.environ.get("TTS_CPU_SHORT_DECODE_CAP", "160"))
        env.setdefault("VOXCPM_CPU_MEDIUM_DECODE_CAP", os.environ.get("TTS_CPU_MEDIUM_DECODE_CAP", "128"))
        env.setdefault("VOXCPM_CPU_LONG_DECODE_CAP", os.environ.get("TTS_CPU_LONG_DECODE_CAP", "96"))
        env.setdefault("VOXCPM_ACCEL_SHORT_DECODE_CAP", os.environ.get("TTS_ACCEL_SHORT_DECODE_CAP", "192"))
        env.setdefault("VOXCPM_ACCEL_MEDIUM_DECODE_CAP", os.environ.get("TTS_ACCEL_MEDIUM_DECODE_CAP", "160"))
        env.setdefault("VOXCPM_ACCEL_LONG_DECODE_CAP", os.environ.get("TTS_ACCEL_LONG_DECODE_CAP", "128"))
        if os.name == "nt":
            dll_dirs = [
                str(APP_ROOT / "VoxCPM.cpp" / "build-cuda" / "bin" / "Release"),
                str(APP_ROOT / "VoxCPM.cpp" / "build" / "bin" / "Release"),
            ]
            env["PATH"] = os.pathsep.join(dll_dirs + [env.get("PATH", "")])
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        return subprocess.Popen(
            args,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=creationflags,
        )

    def wait_for_health(self) -> None:
        deadline = time.monotonic() + float(os.environ.get("TTS_SERVER_START_TIMEOUT", "90"))
        last_error = ""
        while time.monotonic() < deadline:
            if self.process and self.process.poll() is not None:
                raise RuntimeError("VoxCPM server exited during startup.")
            try:
                data, _ = self.http_request("GET", "/healthz")
                if data:
                    return
            except Exception as exc:
                last_error = str(exc)
                time.sleep(0.25)
        raise RuntimeError(f"VoxCPM server did not become healthy: {last_error}")

    def register_voice(self) -> None:
        try:
            data, _ = self.http_request("GET", f"/v1/voices/{self.config.voice_id}")
            metadata = json.loads(data.decode("utf-8"))
            if metadata.get("prompt_text") == self.config.prompt_text:
                return
            self.http_request("DELETE", f"/v1/voices/{self.config.voice_id}")
        except Exception:
            pass
        boundary = f"----aranya-{uuid.uuid4().hex}"
        fields = [
            ("id", self.config.voice_id.encode("utf-8")),
            ("text", self.config.prompt_text.encode("utf-8")),
        ]
        audio_bytes = Path(self.config.prompt_audio_path).read_bytes()
        body = bytearray()
        for name, value in fields:
            body.extend(f"--{boundary}\r\n".encode("ascii"))
            body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("ascii"))
            body.extend(value)
            body.extend(b"\r\n")
        body.extend(f"--{boundary}\r\n".encode("ascii"))
        body.extend(
            (
                'Content-Disposition: form-data; name="audio"; filename="voice_prompt.wav"\r\n'
                "Content-Type: audio/wav\r\n\r\n"
            ).encode("ascii")
        )
        body.extend(audio_bytes)
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode("ascii"))
        self.http_request("POST", "/v1/voices", data=bytes(body), content_type=f"multipart/form-data; boundary={boundary}")

    def http_request(
        self,
        method: str,
        path: str,
        data: bytes | None = None,
        content_type: str | None = None,
    ) -> tuple[bytes, str]:
        url = f"http://{self.config.host}:{self.config.port}{path}"
        request = urllib.request.Request(url, data=data, method=method)
        if content_type:
            request.add_header("Content-Type", content_type)
        if data is not None:
            request.add_header("Content-Length", str(len(data)))
        with urllib.request.urlopen(request, timeout=float(os.environ.get("TTS_HTTP_TIMEOUT", "120"))) as response:
            return response.read(), response.headers.get("Content-Type", "")

    async def synthesize(self, text: str) -> TtsSynthesisResult | None:
        cleaned = " ".join(text.split())
        if not cleaned:
            return None
        async with self.lock:
            await self.start()
            if not self.ready:
                raise RuntimeError(self.last_error or self.missing_reason())
            try:
                return await asyncio.to_thread(self.synthesize_http, cleaned)
            except Exception as exc:
                self.last_error = self.describe_server_failure(exc)
                if self.process and self.process.poll() is not None:
                    self.voice_registered = False
                    self.process = None
                    await self.start()
                    return await asyncio.to_thread(self.synthesize_http, cleaned)
                raise RuntimeError(self.last_error) from exc

    def describe_server_failure(self, exc: Exception) -> str:
        details = str(exc)
        if isinstance(exc, urllib.error.HTTPError):
            try:
                body = exc.read().decode("utf-8", errors="replace")
                if body:
                    details = f"{details}: {body}"
            except Exception:
                pass
        if self.process and self.process.poll() is not None:
            details = f"VoxCPM server exited with code {self.process.returncode}. {details}"
        tail = self.server_log_tail()
        return f"{details}\nVoxCPM log tail:\n{tail}" if tail else details

    def server_log_tail(self, max_bytes: int = 4096) -> str:
        try:
            if not self.log_path.exists():
                return ""
            with self.log_path.open("rb") as log:
                if self.log_path.stat().st_size > max_bytes:
                    log.seek(-max_bytes, os.SEEK_END)
                return log.read().decode("utf-8", errors="replace")
        except Exception:
            return ""

    def synthesize_http(self, text: str) -> TtsSynthesisResult:
        payload = {
            "model": self.config.model_name,
            "input": text,
            "voice": self.config.voice_id,
            "response_format": self.config.response_format,
            "stream_format": "audio",
            "inference_timesteps": self.config.inference_timesteps,
            "cfg_value": self.config.cfg_value,
            "retry_badcase": self.config.retry_badcase,
            "retry_badcase_max_times": self.config.retry_badcase_max_times,
            "retry_badcase_ratio_threshold": self.config.retry_badcase_ratio_threshold,
        }
        data, mime_type = self.http_request(
            "POST",
            "/v1/audio/speech",
            data=json.dumps(payload).encode("utf-8"),
            content_type="application/json",
        )
        if not data:
            raise RuntimeError("VoxCPM server returned an empty audio payload.")
        return TtsSynthesisResult(
            data=data,
            mime_type=mime_type or ("application/octet-stream" if self.config.response_format == "pcm" else "audio/wav"),
            audio_format=self.config.response_format,
            sample_rate=self.config.output_sample_rate,
        )

    def close(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
        self.process = None
        self.voice_registered = False
        if self.log_handle:
            self.log_handle.close()
            self.log_handle = None


def free_tcp_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def build_tts_worker(prompt_audio_path: Path | None) -> NativeTtsWorker:
    model_repo = os.environ.get("TTS_MODEL_REPO", DEFAULT_TTS_REPO)
    model_file = os.environ.get("TTS_MODEL_FILE", DEFAULT_TTS_FILE)
    model_path = resolve_model_path(
        path_env_value=os.environ.get("TTS_MODEL_PATH"),
        repo_id=model_repo,
        filename=model_file,
        cache_dir=MODEL_CACHE_ROOT,
    )
    executable = find_voxcpm_server_executable()
    configured_port = os.environ.get("TTS_SERVER_PORT")
    host = os.environ.get("TTS_SERVER_HOST", "127.0.0.1")
    return NativeTtsWorker(
        TtsWorkerConfig(
            model_repo=model_repo,
            model_file=model_file,
            model_path=model_path,
            executable=executable,
            prompt_audio_path=prompt_audio_path,
            prompt_text=os.environ.get("TTS_PROMPT_TEXT", DEFAULT_TTS_PROMPT_TEXT),
            voice_id=os.environ.get("TTS_VOICE_ID", "aranya"),
            model_name=os.environ.get("TTS_MODEL_NAME", "aranya-voxcpm"),
            host=host,
            port=int(configured_port) if configured_port else free_tcp_port(host),
            threads=int(os.environ.get("TTS_THREADS", str(max(1, (os.cpu_count() or 4) // 2)))),
            backend=default_tts_backend(executable),
            response_format=os.environ.get("TTS_RESPONSE_FORMAT", "pcm"),
            output_sample_rate=int(os.environ.get("TTS_OUTPUT_SAMPLE_RATE", "24000")),
            max_queue=int(os.environ.get("TTS_SERVER_MAX_QUEUE", "8")),
            inference_timesteps=int(os.environ.get("TTS_INFERENCE_TIMESTEPS", "14")),
            cfg_value=float(os.environ.get("TTS_CFG_VALUE", "2.0")),
            retry_badcase=os.environ.get("TTS_RETRY_BADCASE", "0") == "1",
            retry_badcase_max_times=int(os.environ.get("TTS_RETRY_BADCASE_MAX_TIMES", "2")),
            retry_badcase_ratio_threshold=float(os.environ.get("TTS_RETRY_BADCASE_RATIO_THRESHOLD", "12.0")),
        )
    )


tts_worker: NativeTtsWorker | None = None
tts_init_lock = asyncio.Lock()


async def ensure_tts_worker() -> NativeTtsWorker:
    global tts_worker
    async with tts_init_lock:
        if tts_worker and tts_worker.ready:
            return tts_worker
        tts_worker = build_tts_worker(convert_voice_sample_to_wav())
        await tts_worker.start()
        if REQUIRE_TTS and not tts_worker.ready:
            raise RuntimeError(tts_worker.last_error or tts_worker.missing_reason())
        return tts_worker


async def synthesize_voice_chunk(text: str) -> TtsSynthesisResult | None:
    worker = await ensure_tts_worker()
    if not worker.ready:
        return None
    return await worker.synthesize(text)


_QUEUE_DONE = object()


def pop_speakable_segment(buffer: str) -> tuple[str | None, str]:
    text = buffer.strip()
    if not text:
        return None, ""

    sentence_min_chars = int(os.environ.get("TTS_MIN_SENTENCE_CHARS", "32"))
    phrase_min_chars = int(os.environ.get("TTS_MIN_PHRASE_CHARS", "45"))
    long_sentence_chars = int(os.environ.get("TTS_LONG_SENTENCE_PHRASE_CHARS", "95"))
    max_chars = int(os.environ.get("TTS_MAX_SEGMENT_CHARS", "150"))
    sentence_marks = ".!?\n"
    phrase_marks = ",;:"

    if len(text) >= sentence_min_chars:
        for index, char in enumerate(text):
            if char in sentence_marks and index + 1 >= sentence_min_chars:
                sentence = text[: index + 1].strip()
                if len(sentence) >= long_sentence_chars:
                    phrase_split = phrase_split_index(sentence, phrase_min_chars, len(sentence))
                    if phrase_split >= 0:
                        return ensure_sentence_end(sentence[: phrase_split + 1].strip()), sentence[phrase_split + 1 :] + text[index + 1 :]
                return text[: index + 1].strip(), text[index + 1 :]

    if len(text) >= phrase_min_chars:
        split_at = phrase_split_index(text, phrase_min_chars, max_chars)
        if split_at >= 0:
            return ensure_sentence_end(text[: split_at + 1].strip()), text[split_at + 1 :]

    if len(text) < max_chars:
        return None, buffer

    split_at = -1
    for mark in phrase_marks:
        split_at = max(split_at, text.rfind(mark, sentence_min_chars, max_chars))
    if split_at < 0:
        split_at = text.rfind(" ", sentence_min_chars, max_chars)
    if split_at < 0:
        split_at = max_chars
    return ensure_sentence_end(text[: split_at + 1].strip()), text[split_at + 1 :]


def phrase_split_index(text: str, min_chars: int, max_chars: int) -> int:
    phrase_marks = ",;:"
    split_at = -1
    for mark in phrase_marks:
        position = text.find(mark, min_chars, max_chars)
        if position >= 0:
            split_at = position if split_at < 0 else min(split_at, position)
    return split_at


def ensure_sentence_end(text: str) -> str:
    text = text.strip().rstrip(",;:")
    if not text:
        return text
    return text if text.endswith((".", "!", "?")) else f"{text}."


def audio_file_suffix() -> str:
    fmt = os.environ.get("TTS_RESPONSE_FORMAT", "pcm").lower()
    return "pcm" if fmt == "pcm" else fmt


async def run_analysis_stream(
    run_id: str,
    mode: str,
    image_path: Path,
) -> AsyncIterator[bytes]:
    started = time.perf_counter()
    worker = workers[mode]
    final_text = ""
    audio_file = AUDIO_ROOT / f"{run_id}.{audio_file_suffix()}"
    AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    event_queue: asyncio.Queue[dict | object] = asyncio.Queue()
    tts_queue: asyncio.Queue[tuple[int, str] | object] = asyncio.Queue()
    first_text_ms: int | None = None
    first_tts_request_ms: int | None = None
    first_audio_ms: int | None = None

    async def put_event(event: dict) -> None:
        await event_queue.put(event)

    async def llm_producer() -> None:
        nonlocal final_text, first_text_ms
        prompt = MODE_CONFIG[mode]["prompt"]
        segment_id = 0
        segment_text = ""
        await put_event(
            {
                "type": "status",
                "message": "Opening the Wildkeeper journal...",
                "text_delay_ms": int(os.environ.get("TTS_BALANCED_TEXT_DELAY_MS", "600")),
                "audio_playback_rate": float(os.environ.get("TTS_AUDIO_PLAYBACK_RATE", "0.92")),
                "audio_prebuffer_chunks": int(os.environ.get("TTS_AUDIO_PREBUFFER_CHUNKS", "3")),
                "audio_prebuffer_max_ms": int(os.environ.get("TTS_AUDIO_PREBUFFER_MAX_MS", "2000")),
            }
        )
        await put_event({"type": "segment_start", "segment_id": segment_id})
        async for delta in worker.stream(image_path, prompt):
            if first_text_ms is None:
                first_text_ms = int((time.perf_counter() - started) * 1000)
            final_text += delta
            segment_text += delta
            await put_event({"type": "text_delta", "delta": delta, "segment_id": segment_id})
            while True:
                text, remainder = pop_speakable_segment(segment_text)
                if not text:
                    segment_text = remainder
                    break
                await tts_queue.put((segment_id, text))
                await put_event({"type": "segment_done", "segment_id": segment_id})
                segment_id += 1
                segment_text = remainder
                await put_event({"type": "segment_start", "segment_id": segment_id})
        if segment_text.strip():
            await tts_queue.put((segment_id, segment_text.strip()))
            await put_event({"type": "segment_done", "segment_id": segment_id})
        await tts_queue.put(_QUEUE_DONE)

    async def tts_producer() -> None:
        nonlocal first_tts_request_ms, first_audio_ms
        await put_event({"type": "status", "message": "Starting voice engine..."})
        await ensure_tts_worker()
        await put_event({"type": "status", "message": "Voice ready."})
        while True:
            item = await tts_queue.get()
            if item is _QUEUE_DONE:
                return
            segment_id, text = item
            if first_tts_request_ms is None:
                first_tts_request_ms = int((time.perf_counter() - started) * 1000)
            await put_event({"type": "status", "message": "Speaking segment...", "segment_id": segment_id})
            audio = await synthesize_voice_chunk(text)
            if not audio:
                if REQUIRE_TTS:
                    raise RuntimeError("VoxCPM TTS did not produce audio for the streamed text.")
                continue
            if first_audio_ms is None:
                first_audio_ms = int((time.perf_counter() - started) * 1000)
            with audio_file.open("ab") as dst:
                dst.write(audio.data)
            await put_event(
                {
                    "type": "audio_chunk",
                    "mime_type": audio.mime_type,
                    "audio_format": audio.audio_format,
                    "sample_rate": audio.sample_rate,
                    "segment_id": segment_id,
                    "tts_text": text if os.environ.get("TTS_DEBUG_TEXT", "0") == "1" else None,
                    "data": base64.b64encode(audio.data).decode("ascii"),
                }
            )

    llm_task = asyncio.create_task(llm_producer())
    tts_task = asyncio.create_task(tts_producer())
    tasks = {llm_task, tts_task}
    try:
        while tasks:
            if event_queue.empty():
                done, tasks = await asyncio.wait(tasks, timeout=0.05, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    exc = task.exception()
                    if exc:
                        for pending in tasks:
                            pending.cancel()
                        raise exc
            while not event_queue.empty():
                event = await event_queue.get()
                yield ndjson(event)
        while not event_queue.empty():
            event = await event_queue.get()
            yield ndjson(event)
        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "run=%s first_text_ms=%s first_tts_request_ms=%s first_audio_ms=%s total_ms=%s",
            run_id,
            first_text_ms,
            first_tts_request_ms,
            first_audio_ms,
            latency_ms,
        )
        with connect_db() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, latency_ms = ? WHERE id = ?",
                ("complete", latency_ms, run_id),
            )
            conn.execute(
                "INSERT INTO outputs (id, run_id, final_text, parsed_json, audio_path) VALUES (?, ?, ?, ?, ?)",
                (uuid.uuid4().hex, run_id, final_text, None, str(audio_file)),
            )
        yield ndjson({"type": "record_saved", "run_id": run_id})
        yield ndjson({"type": "done", "run_id": run_id})
    except Exception as exc:
        for task in tasks:
            task.cancel()
        with connect_db() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, error = ? WHERE id = ?",
                ("error", str(exc), run_id),
            )
        yield ndjson({"type": "error", "message": str(exc)})


async def start_application() -> None:
    global warmup_task, app_started
    if app_started:
        return
    app_started = True
    ensure_dirs()
    init_db()
    workers.update({mode: build_worker(mode) for mode in MODE_CONFIG})
    if BLOCKING_MODEL_STARTUP:
        await warm_models()
        if startup_state["phase"] == "error":
            raise RuntimeError(str(startup_state["message"]))
    else:
        warmup_task = asyncio.create_task(warm_models())


def start_application_sync() -> None:
    asyncio.run(start_application())


async def stop_application() -> None:
    global app_started
    if warmup_task and not warmup_task.done():
        warmup_task.cancel()
    for worker in workers.values():
        worker.close()
    if tts_worker:
        tts_worker.close()
    app_started = False


@asynccontextmanager
async def lifespan(_: gr.Server):
    await start_application()
    try:
        yield
    finally:
        await stop_application()


app = gr.Server(lifespan=lifespan)


@app.on_event("startup")
async def on_startup() -> None:
    await start_application()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await stop_application()


@app.get("/")
async def index() -> HTMLResponse:
    if not app_started:
        if BLOCKING_MODEL_STARTUP:
            await start_application()
        else:
            asyncio.create_task(start_application())
    return HTMLResponse(
        (FRONTEND_DIR / "index.html").read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/frontend/{path:path}")
async def frontend_asset(path: str) -> FileResponse:
    target = FRONTEND_DIR / path
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(target, headers={"Cache-Control": "no-store"})


@app.get("/api/status")
async def api_status() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ready",
            "startup": startup_state,
            "db_path": str(DB_PATH),
            "runtime": {
                "host_gpu_available": host_gpu_available(),
                "nvidia_smi": nvidia_smi_summary(),
                "cuda_toolkit": cuda_toolkit_summary(),
                "llama_gpu_offload_supported": llama_gpu_offload_supported(),
                "llama_cpp_import_error": LLAMA_CPP_IMPORT_ERROR,
                "llama_cuda_wheel_index": os.environ.get("LLAMA_CUDA_WHEEL_INDEX", "cu132"),
                "llama_gpu_required": REQUIRE_LLAMA_GPU,
                "tts_gpu_required": REQUIRE_TTS_GPU,
                "voxcpm_build_cuda": voxcpm_cuda_build_enabled(),
                "voxcpm_cuda_architectures": os.environ.get("VOXCPM_CUDA_ARCHITECTURES", "89"),
            },
            "workers": {
                mode: {
                    "runtime_ready": worker.ready,
                    "model_loaded": worker.loaded,
                    "last_error": worker.last_error,
                    "runtime_import_error": LLAMA_CPP_IMPORT_ERROR,
                    "n_gpu_layers": worker.config.n_gpu_layers,
                    "model_repo": worker.config.model_repo,
                    "model_file": worker.config.model_file,
                    "model_path": worker.config.model_path,
                    "mmproj_path": worker.config.mmproj_path,
                }
                for mode, worker in workers.items()
            },
            "tts": {
                "runtime_ready": bool(tts_worker and tts_worker.ready),
                "model_repo": tts_worker.config.model_repo if tts_worker else None,
                "model_file": tts_worker.config.model_file if tts_worker else None,
                "model_path": tts_worker.config.model_path if tts_worker else None,
                "prompt_audio_path": str(tts_worker.config.prompt_audio_path) if tts_worker and tts_worker.config.prompt_audio_path else None,
                "prompt_text_set": bool(tts_worker and tts_worker.config.prompt_text.strip()),
                "voice_id": tts_worker.config.voice_id if tts_worker else None,
                "server_pid": tts_worker.process.pid if tts_worker and tts_worker.process else None,
                "server_url": f"http://{tts_worker.config.host}:{tts_worker.config.port}" if tts_worker else None,
                "executable": tts_worker.config.executable if tts_worker else None,
                "backend": tts_worker.config.backend if tts_worker else None,
                "cuda_binary_present": bool(existing_voxcpm_cuda_executable("voxcpm-server")),
                "response_format": tts_worker.config.response_format if tts_worker else None,
                "sample_rate": tts_worker.config.output_sample_rate if tts_worker else None,
                "last_error": tts_worker.last_error if tts_worker else None,
                "required": REQUIRE_TTS,
                "gpu_required": REQUIRE_TTS_GPU,
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
    return JSONResponse(
        {
            "stats": {
                "total": stats["total"] or 0,
                "species": stats["species"] or 0,
                "rescues": stats["rescues"] or 0,
            },
        }
    )


@app.post("/api/run")
async def api_run(
    request: Request,
    mode: str = Form(...),
    image: UploadFile = File(...),
    x_hf_username: str | None = Header(default=None),
    x_forwarded_user: str | None = Header(default=None),
) -> StreamingResponse:
    if mode not in MODE_CONFIG:
        raise HTTPException(status_code=400, detail="mode must be identify or health")
    run_id = uuid.uuid4().hex
    image_path, image_sha256 = save_upload(image, mode)
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
        run_analysis_stream(run_id, mode, image_path),
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
    start_application_sync()
    app.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
    )
