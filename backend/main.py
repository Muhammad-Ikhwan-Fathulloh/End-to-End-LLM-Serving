import os
import subprocess
import time
import httpx
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Smart Manufacturing AI (Ultra-High Performance GGUF Mode)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")
BIN_DIR = os.path.join(BASE_DIR, "bin")

# Config
LLM_MODEL_GGUF = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
LLAMA_SERVER_EXE = os.path.join(BIN_DIR, "llama-server.exe")

# Global Instance
llm_server = None

class PersistentGGUFServer:
    def __init__(self, name: str, model_path: str, port: int, mmproj_path: str = None):
        self.name = name
        self.model_path = model_path
        self.port = port
        self.mmproj_path = mmproj_path
        self.process = None

    def start(self):
        if not os.path.exists(self.model_path):
            print(f"[{self.name}] ERROR: Model not found: {self.model_path}")
            return False
            
        cmd = [
            LLAMA_SERVER_EXE,
            "-m", self.model_path,
            "--port", str(self.port),
            "-c", "2048",
            "--n-gpu-layers", "0"  # Force CPU
        ]
        
        if self.mmproj_path:
            if os.path.exists(self.mmproj_path):
                cmd.extend(["--mmproj", self.mmproj_path])
                print(f"[{self.name}] Using mmproj: {self.mmproj_path}")
            else:
                print(f"[{self.name}] WARNING: MMProj not found: {self.mmproj_path}")

        print(f"[{self.name}] Starting server on port {self.port} with command: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd, 
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Wait for ready
        print(f"[{self.name}] Waiting for server to become ready (max 20s)...")
        for i in range(20):
            try:
                with httpx.Client(timeout=2.0) as client:
                    resp = client.get(f"http://localhost:{self.port}/health")
                    if resp.status_code == 200:
                        print(f"[{self.name}] Server is ready and listening on http://localhost:{self.port}")
                        return True
            except Exception as e:
                # Check if process died
                if self.process.poll() is not None:
                    output, _ = self.process.communicate()
                    print(f"[{self.name}] Server failed to start! Output:\n{output}")
                    return False
                pass
            time.sleep(1)
        
        # If we got here, timeout
        print(f"[{self.name}] ERROR: Server failed to start within 20 seconds!")
        return False

    def stop(self):
        if self.process:
            self.process.terminate()

    async def generate_chat(self, messages: list):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        payload = {
            "messages": messages,
            "temperature": 0.2
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload)
            return resp.json()["choices"][0]["message"]["content"]

@app.on_event("startup")
async def startup_event():
    global llm_server
    
    # Start LLM Engine (Qwen 2.5)
    llm_model = os.path.join(MODELS_DIR, LLM_MODEL_GGUF)
    llm_server = PersistentGGUFServer("LLM", llm_model, 8080)
    llm_server.start()

@app.on_event("shutdown")
def shutdown_event():
    if llm_server: llm_server.stop()

class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a professional manufacturing engineer."

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.message}
        ]
        response = await llm_server.generate_chat(messages)

        return {
            "response": response,
            "status": "success"
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
