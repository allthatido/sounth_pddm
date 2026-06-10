---
title: Aranya A Wildkeepers Adventure
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.35.0
app_file: app.py
pinned: false
hf_oauth: true
preload_from_hub:
  - openbmb/MiniCPM-V-4.6-gguf MiniCPM-V-4_6-Q8_0.gguf
  - openbmb/MiniCPM-V-4.6-gguf mmproj-model-f16.gguf
  - bluryar/VoxCPM-GGUF voxcpm-0.5b-q8_0-audiovae-f16.gguf
---

# Aranya: A Wildkeeper's Adventure

A custom Gradio Server app for plant identification and plant-health analysis.
The UI is a forest treasure hunt, while MiniCPM-V GGUF inference is loaded at
startup with `llama-cpp-python` and kept warm for button clicks.

## Runtime configuration

The default Space startup downloads/resolves these GGUF files and begins loading
the vision model automatically in the background. Set variables only to override
defaults:

- `IDENTIFY_MODEL_REPO`: default `openbmb/MiniCPM-V-4.6-gguf`.
- `IDENTIFY_MODEL_FILE`: default `MiniCPM-V-4_6-Q8_0.gguf`.
- `IDENTIFY_MODEL_PATH`: path to the identification GGUF.
- `IDENTIFY_MMPROJ_PATH`: path to its multimodal projector GGUF.
- `IDENTIFY_MMPROJ_FILE`: default `mmproj-model-f16.gguf`.
- `HEALTH_MODEL_REPO`: default `openbmb/MiniCPM-V-4.6-gguf`.
- `HEALTH_MODEL_FILE`: default `MiniCPM-V-4_6-Q8_0.gguf`.
- `HEALTH_MODEL_PATH`: path to the health-analysis GGUF.
- `HEALTH_MMPROJ_PATH`: path to its multimodal projector GGUF.
- `HEALTH_MMPROJ_FILE`: default `mmproj-model-f16.gguf`.
- `ARANYA_MODEL_CACHE_DIR`: default `/data/models` when `/data` is writable.
- `LLAMA_CTX_SIZE`: default `4096`.
- `LLAMA_THREADS`: default CPU count.
- `LLAMA_N_GPU_LAYERS`: defaults to `99` only when a host GPU and llama-cpp GPU offload support are detected; otherwise `0`.
- `LLAMA_CUDA_WHEEL_INDEX`: default `cu132`; documentation/status marker for the CUDA wheel family used by `requirements.txt`.
- `LLAMA_CHAT_HANDLER`: optional llama-cpp-python chat handler override.
- `LLAMA_CHAT_FORMAT`: optional chat format override.
- `LLAMA_MAX_TOKENS`: default `384`.
- `LLAMA_TEMPERATURE`: default `0.2`.
- `ARANYA_BLOCKING_MODEL_STARTUP`: default `0`; set `1` only if the Space should fail before serving UI when model warmup fails.
- `ARANYA_TTS_WORKER`: optional path to the `voxcpm-server` executable.
- `TTS_MODEL_REPO`: default `bluryar/VoxCPM-GGUF`.
- `TTS_MODEL_FILE`: default `voxcpm-0.5b-q8_0-audiovae-f16.gguf`.
- `TTS_MODEL_PATH`: optional direct path to the VoxCPM GGUF.
- `TTS_PROMPT_TEXT`: transcript for `frontend/assets/voice_sample.mp3` or `frontend/assets/voice_prompt.wav`; this is used only for voice conditioning, not spoken output.
- `TTS_VOICE_ID`: default `aranya`.
- `TTS_SERVER_HOST`: default `127.0.0.1`.
- `TTS_SERVER_PORT`: optional fixed port; by default the app picks a free local port.
- `TTS_RESPONSE_FORMAT`: default `pcm`.
- `TTS_OUTPUT_SAMPLE_RATE`: default `24000`.
- `TTS_BALANCED_TEXT_DELAY_MS`: default `600`.
- `TTS_AUDIO_PREBUFFER_CHUNKS`: default `3`.
- `TTS_AUDIO_PREBUFFER_MAX_MS`: default `2200`.
- `TTS_AUDIO_PLAYBACK_RATE`: default `0.95`; slightly slows streamed playback so the next TTS chunk has more time to synthesize.
- `TTS_MIN_SENTENCE_CHARS`: default `32`.
- `TTS_MIN_PHRASE_CHARS`: default `45`.
- `TTS_LONG_SENTENCE_PHRASE_CHARS`: default `95`.
- `TTS_MIN_SEGMENT_CHARS`: default `70`.
- `TTS_MAX_SEGMENT_CHARS`: default `150`.
- `TTS_INFERENCE_TIMESTEPS`: default `18`.
- `TTS_CFG_VALUE`: default `2.0`.
- `TTS_RETRY_BADCASE`: default `0`.
- `TTS_CPU_SHORT_DECODE_CAP`: default `160`.
- `TTS_CPU_MEDIUM_DECODE_CAP`: default `128`.
- `TTS_CPU_LONG_DECODE_CAP`: default `96`.
- `TTS_BACKEND`: defaults to `cuda` for a CUDA VoxCPM.cpp build path, otherwise `cpu`.
- `VOXCPM_CPP_REPO`: default `https://github.com/bluryar/VoxCPM.cpp.git`.
- `ARANYA_BUILD_TTS_WORKER`: default `1`; automatically builds `voxcpm-server` at startup when no executable is found.
- `VOXCPM_BUILD_CUDA`: defaults to `1` when a GPU is visible; set `0` to force a CPU build.
- `VOXCPM_CUDA_ARCHITECTURES`: default `89` for RTX 4060 and RTX 6000 Ada; use `86` for RTX A6000/Ampere.
- `VOXCPM_BUILD_JOBS`: defaults to up to 8 parallel build jobs.
- `ARANYA_REQUIRE_TTS`: default `1`; voice cloning is required and runs will error if VoxCPM.cpp is unavailable.
- `ARANYA_REQUIRE_LLAMA_GPU`: default `0`; set `1` to fail startup/runs unless llama-cpp-python GPU offload is available.
- `ARANYA_REQUIRE_TTS_GPU`: default `0`; set `1` to fail TTS unless a CUDA VoxCPM server is selected.
- `ARANYA_DB_PATH`: default `/data/aranya.sqlite3`.

`requirements.txt` uses the CUDA 13.2 `llama-cpp-python` wheel by default.
On GPU machines, `/api/status` should report `host_gpu_available: true`,
`llama_gpu_offload_supported: true`, and each MiniCPM worker should use
`n_gpu_layers: 99`. If the CUDA wheel is not compatible with a target runtime,
install the matching CUDA wheel family or build `llama-cpp-python` with
`GGML_CUDA=on`.

On Windows, `nvidia-smi` only confirms the NVIDIA driver and visible GPU. The
local CUDA wheel still needs the CUDA runtime DLLs it was built against, such as
`cudart64_13.dll` and `cublas64_13.dll`, discoverable through `CUDA_PATH`/PATH or
the app's DLL search bootstrap. A local VoxCPM CUDA build also needs the Visual
Studio C++ build tools available through `VsDevCmd.bat`; `/api/status` reports
`runtime.cuda_toolkit` so missing `nvcc`, `cl`, or DLL paths are obvious.

Real voice cloning requires the VoxCPM.cpp `voxcpm-server` executable. Build
VoxCPM.cpp and place the binary at `hf_space/VoxCPM.cpp/build-cuda/examples/`,
`hf_space/VoxCPM.cpp/build/examples/`, or set `ARANYA_TTS_WORKER`. If no binary
is found, the app clones/builds VoxCPM.cpp automatically at startup when
`ARANYA_BUILD_TTS_WORKER=1`. When a GPU is visible, it tries the CUDA build
directory first and launches VoxCPM with `--backend cuda`; if that fails it
falls back to CPU unless `ARANYA_REQUIRE_TTS_GPU=1`. Local Windows CUDA builds
can be started with `hf_space\scripts\build_voxcpm_cuda_windows.bat`; HF/Linux
CUDA builds can use `bash hf_space/scripts/build_voxcpm_cuda_linux.sh`.
The app uses `frontend/assets/voice_prompt.wav`
directly when present, or converts `frontend/assets/voice_sample.mp3` to a
cached 24 kHz mono WAV prompt at startup using system `ffmpeg` or the bundled
`imageio-ffmpeg` binary. It registers the prompt once as voice id `aranya` and
sends only LLM response text to speech synthesis.

Model warmup state is visible at `/api/status`. If the specialized VoxCPM TTS
executable is not available, `/api/status` reports the missing runtime and
voice-cloned audio will not be silently replaced by browser speech.
