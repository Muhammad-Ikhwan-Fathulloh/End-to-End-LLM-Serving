import os
import subprocess
import time
import httpx
import asyncio
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

app = FastAPI(title="P4: Feedback Loop with pgvector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config (Updated for Docker)
DB_CONFIG = "dbname=mydatabase user=user password=password host=localhost port=5432"
LLM_MODEL_GGUF = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
BIN_DIR = os.path.join(ROOT_DIR, "backend", "bin")
LLAMA_SERVER_EXE = os.path.join(BIN_DIR, "llama-server.exe")

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
llm_server_process = None

class ChatRequest(BaseModel):
    message: str

class FeedbackRequest(BaseModel):
    interaction_id: str
    is_like: bool

def get_db_connection():
    try:
        conn = psycopg2.connect(DB_CONFIG)
        register_vector(conn)
        return conn
    except:
        return None

@app.on_event("startup")
async def startup_event():
    global llm_server_process
    conn = get_db_connection()
    if conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("CREATE TABLE IF NOT EXISTS interactions (interaction_id TEXT PRIMARY KEY, prompt TEXT, response TEXT, feedback BOOLEAN, embedding vector(384))")
        conn.commit()
        conn.close()

    model_path = os.path.join(MODELS_DIR, LLM_MODEL_GGUF)
    cmd = [LLAMA_SERVER_EXE, "-m", model_path, "--port", "8004", "-c", "2048", "--n-gpu-layers", "0"]
    llm_server_process = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    await asyncio.sleep(5)

@app.on_event("shutdown")
def shutdown_event():
    if llm_server_process:
        llm_server_process.terminate()

@app.post("/chat")
async def chat(request: ChatRequest):
    url = "http://localhost:8004/v1/chat/completions"
    payload = {"messages": [{"role": "user", "content": request.message}]}
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, json=payload)
        response_text = resp.json()["choices"][0]["message"]["content"]
    
    interaction_id = str(uuid.uuid4())
    embedding = embed_model.encode(request.message)
    conn = get_db_connection()
    if conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO interactions (interaction_id, prompt, response, embedding) VALUES (%s, %s, %s, %s)", (interaction_id, request.message, response_text, embedding))
        conn.commit()
        conn.close()
    
    return {"interaction_id": interaction_id, "response": response_text}

@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    conn = get_db_connection()
    if conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE interactions SET feedback = %s WHERE interaction_id = %s", (request.is_like, request.interaction_id))
        conn.commit()
        conn.close()
        return {"status": "success"}
    raise HTTPException(status_code=500, detail="Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
