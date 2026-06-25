import os
import subprocess
import time
import httpx
import asyncio
import saka
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="P2: Dedicated Saka-NLP Optimization")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
LLM_MODEL_GGUF = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
BIN_DIR = os.path.join(ROOT_DIR, "backend", "bin")
LLAMA_SERVER_EXE = os.path.join(BIN_DIR, "llama-server.exe")

llm_server_process = None

class IDChatRequest(BaseModel):
    message: str

@app.on_event("startup")
async def startup_event():
    global llm_server_process
    model_path = os.path.join(MODELS_DIR, LLM_MODEL_GGUF)
    cmd = [LLAMA_SERVER_EXE, "-m", model_path, "--port", "8002", "-c", "2048", "--n-gpu-layers", "0"]
    llm_server_process = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    await asyncio.sleep(5)

@app.on_event("shutdown")
def shutdown_event():
    if llm_server_process:
        llm_server_process.terminate()

@app.post("/chat")
async def chat(request: IDChatRequest):
    # Normalize input
    normalized = saka.normalize(request.message)
    
    url = "http://localhost:8002/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "system", "content": "Jawablah dalam Bahasa Indonesia yang formal dan santun."},
            {"role": "user", "content": normalized}
        ],
        "temperature": 0.5
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=payload)
        response_text = resp.json()["choices"][0]["message"]["content"]
    
    return {
        "original": request.message,
        "normalized": normalized,
        "response": response_text
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
