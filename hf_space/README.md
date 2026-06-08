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
---

# Aranya: A Wildkeeper's Adventure

A custom Gradio Server app for plant identification and plant-health analysis.
The UI is a forest treasure hunt, while inference is routed through persistent
native llama.cpp/libmtmd workers when the quantized MiniCPM-V GGUFs are present.

## Runtime configuration

Set these Space variables after publishing the final GGUF files:

- `ARANYA_NATIVE_WORKER`: path to the persistent JSONL native worker executable.
- `IDENTIFY_MODEL_PATH`: path to the identification GGUF.
- `IDENTIFY_MMPROJ_PATH`: path to its multimodal projector GGUF.
- `HEALTH_MODEL_PATH`: path to the health-analysis GGUF.
- `HEALTH_MMPROJ_PATH`: path to its multimodal projector GGUF.
- `LLAMA_CTX_SIZE`: default `4096`.
- `LLAMA_THREADS`: default CPU count.
- `LLAMA_N_GPU_LAYERS`: default `99`.
- `ARANYA_DB_PATH`: default `/data/aranya.sqlite3`.

If the worker executable or model files are missing, the Space starts in demo
scout mode so the UI, streaming protocol, SQLite persistence, and voice pipeline
can still be tested.
