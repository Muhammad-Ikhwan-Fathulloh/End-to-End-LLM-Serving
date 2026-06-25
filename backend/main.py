import os
import subprocess
import time
import httpx
import asyncio
import saka
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="P1: Basic LLM & Indonesia Optimization")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, "models")
BIN_DIR = os.path.join(BASE_DIR, "bin")

LLM_MODEL_GGUF = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
LLAMA_SERVER_EXE = os.path.join(BIN_DIR, "llama-server.exe")

llm_server_process = None

class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful assistant."

@app.on_event("startup")
async def startup_event():
    global llm_server_process
    
    model_path = os.path.join(MODELS_DIR, LLM_MODEL_GGUF)
    # Use relative path for model if possible to avoid space issues in some binaries
    rel_model_path = os.path.relpath(model_path, BIN_DIR)
    
    cmd = [
        LLAMA_SERVER_EXE,
        "-m", model_path,
        "--port", "8080",
        "-c", "2048",
        "--n-gpu-layers", "0"
    ]
    
    print(f"[P1] Starting LLM Server: {' '.join(cmd)} in {BIN_DIR}")
    
    # Write logs to a file for better debugging
    log_file = open(os.path.join(BASE_DIR, "llama_server.log"), "w")
    try:
        llm_server_process = subprocess.Popen(
            cmd, 
            cwd=BIN_DIR,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True
        )
    except Exception as e:
        print(f"[P1] ERROR: Failed to Popen: {e}")
        return
    
    # Wait for ready
    max_retries = 30
    for i in range(max_retries):
        if llm_server_process.poll() is not None:
            stdout, stderr = llm_server_process.communicate()
            print(f"[P1] LLM Server died immediately! Stderr: {stderr}")
            break
            
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get("http://localhost:8080/health")
                if resp.status_code == 200:
                    print(f"[P1] LLM Server is ready.")
                    break
        except:
            pass
        await asyncio.sleep(1)

@app.on_event("shutdown")
def shutdown_event():
    if llm_server_process:
        llm_server_process.terminate()

@app.post("/chat")
async def chat(request: ChatRequest):
    # Normalize input for Indonesian
    normalized_message = saka.normalize(request.message)
    
    url = "http://localhost:8080/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "system", "content": f"{request.system_prompt}. Jawablah dalam Bahasa Indonesia yang baik."},
            {"role": "user", "content": normalized_message}
        ],
        "temperature": 0.7
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload)
            print(f"[P1] Backend status: {resp.status_code}")
            result = resp.json()
            if "choices" not in result:
                print(f"[P1] ERROR: Unexpected backend response from {url}: {result}")
                raise HTTPException(status_code=500, detail=f"Backend Error: {result}")
            return {
                "original_input": request.message,
                "normalized_input": normalized_message,
                "response": result["choices"][0]["message"]["content"]
            }
    except Exception as e:
        print(f"[P1] Exception in /chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
