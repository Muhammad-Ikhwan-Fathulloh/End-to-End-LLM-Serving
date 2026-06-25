import os
import time
import platform
import subprocess
import asyncio
from contextlib import asynccontextmanager

import httpx
import saka
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Konfigurasi — bisa di-override lewat environment variable tanpa ubah kode.
# Contoh (PowerShell): $env:LLAMA_NGL="20"  (offload 20 layer ke GPU)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, "models")
BIN_DIR = os.path.join(BASE_DIR, "bin")

LLM_MODEL_GGUF = os.environ.get("LLM_MODEL_GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf")
MODEL_PATH = os.path.join(MODELS_DIR, LLM_MODEL_GGUF)

_EXE_NAME = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
LLAMA_SERVER_EXE = os.path.join(BIN_DIR, _EXE_NAME)

LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8080"))
LLAMA_CTX = os.environ.get("LLAMA_CTX", "2048")
# 0 = full CPU, aman & ringan untuk laptop tanpa GPU dedicated.
# Naikkan nilainya kalau ada GPU NVIDIA/Metal supaya lebih cepat.
LLAMA_NGL = os.environ.get("LLAMA_NGL", "0")
# Otomatis pakai (jumlah core - 1) supaya OS tetap responsif.
LLAMA_THREADS = os.environ.get("LLAMA_THREADS", str(max(1, (os.cpu_count() or 4) - 1)))
LLAMA_READY_TIMEOUT = int(os.environ.get("LLAMA_READY_TIMEOUT", "60"))  # detik

# PENTING: pakai 127.0.0.1 (bukan "localhost"). Di Windows, "localhost" kadang
# resolve ke ::1 (IPv6) dulu, sementara llama-server cuma listen di IPv4 —
# ini bikin health-check lambat/gagal padahal servernya sudah hidup.
LLAMA_BASE_URL = f"http://127.0.0.1:{LLAMA_PORT}"
LOG_PATH = os.path.join(BASE_DIR, "llama_server.log")

state = {"process": None, "ready": False, "client": None}


def _read_log_tail(n_chars: int = 2000) -> str:
    """Baca beberapa ratus baris terakhir log llama-server untuk debugging."""
    try:
        with open(LOG_PATH, "r", errors="ignore") as f:
            return f.read()[-n_chars:]
    except OSError:
        return "(log tidak ditemukan)"


async def start_llama_server() -> None:
    if not os.path.isfile(LLAMA_SERVER_EXE):
        raise RuntimeError(f"llama-server tidak ditemukan di: {LLAMA_SERVER_EXE}")
    if not os.path.isfile(MODEL_PATH):
        raise RuntimeError(f"Model GGUF tidak ditemukan di: {MODEL_PATH}")

    cmd = [
        LLAMA_SERVER_EXE,
        "-m", MODEL_PATH,
        "--host", "127.0.0.1",
        "--port", str(LLAMA_PORT),
        "-c", LLAMA_CTX,
        "--n-gpu-layers", LLAMA_NGL,
        "--threads", LLAMA_THREADS,
        "--threads-batch", LLAMA_THREADS,
    ]

    print(f"[P1] Menjalankan llama-server: {' '.join(cmd)}")
    log_file = open(LOG_PATH, "w")
    process = subprocess.Popen(
        cmd,
        cwd=BIN_DIR,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    state["process"] = process

    async with httpx.AsyncClient(timeout=5.0) as probe:
        deadline = time.monotonic() + LLAMA_READY_TIMEOUT
        while time.monotonic() < deadline:
            if process.poll() is not None:
                log_file.flush()
                raise RuntimeError(
                    f"llama-server berhenti sendiri (exit code {process.returncode}).\n"
                    f"--- isi llama_server.log ---\n{_read_log_tail()}"
                )
            try:
                resp = await probe.get(f"{LLAMA_BASE_URL}/health")
                if resp.status_code == 200:
                    print("[P1] llama-server siap.")
                    state["ready"] = True
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)

    raise RuntimeError(
        f"llama-server tidak siap setelah {LLAMA_READY_TIMEOUT} detik.\n"
        f"--- isi llama_server.log ---\n{_read_log_tail()}"
    )


def stop_llama_server() -> None:
    process = state.get("process")
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Satu client yang dipakai ulang untuk semua request /chat — jauh lebih
    # ringan daripada buka-tutup koneksi TCP baru setiap kali ada chat masuk.
    state["client"] = httpx.AsyncClient(timeout=120.0)
    try:
        await start_llama_server()
    except Exception as e:
        # App tetap dijalankan supaya /health bisa melaporkan masalahnya
        # dengan jelas, daripada FastAPI gagal start total tanpa pesan.
        print(f"[P1] ERROR saat startup llama-server: {e}")
    yield
    stop_llama_server()
    await state["client"].aclose()


app = FastAPI(title="P1: Basic LLM & Indonesia Optimization", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful assistant."
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(256, ge=1, le=2048)  # batasi panjang jawaban -> lebih ringan & cepat


@app.get("/health")
async def health():
    process = state.get("process")
    alive = process is not None and process.poll() is None
    return {
        "app": "ok",
        "llama_server_alive": alive,
        "llama_server_ready": state.get("ready", False),
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    if not state.get("ready"):
        raise HTTPException(
            status_code=503,
            detail="Model backend belum siap. Cek endpoint /health atau file llama_server.log.",
        )

    normalized_message = saka.normalize(request.message)

    payload = {
        "messages": [
            {
                "role": "system",
                "content": f"{request.system_prompt}. Jawablah dalam Bahasa Indonesia yang baik.",
            },
            {"role": "user", "content": normalized_message},
        ],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }

    client: httpx.AsyncClient = state["client"]
    try:
        resp = await client.post(f"{LLAMA_BASE_URL}/v1/chat/completions", json=payload)
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Tidak bisa terhubung ke llama-server. Proses backend mungkin sudah mati, cek llama_server.log.",
        )

    try:
        result = resp.json()
    except ValueError:
        raise HTTPException(status_code=502, detail=f"Respons backend bukan JSON valid: {resp.text[:300]}")

    if "choices" not in result:
        print(f"[P1] ERROR: Respons backend tidak terduga: {result}")
        raise HTTPException(status_code=502, detail=f"Backend Error: {result}")

    return {
        "original_input": request.message,
        "normalized_input": normalized_message,
        "response": result["choices"][0]["message"]["content"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)