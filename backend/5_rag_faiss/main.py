import os
import subprocess
import time
import httpx
import asyncio
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

app = FastAPI(title="P5: RAG with FAISS")

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

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
llm_server_process = None

DOCUMENTS = ["The capital of Indonesia is Jakarta.", "Saka-NLP is for Indonesian optimization."]
index = None
doc_embeddings = None

class ChatRequest(BaseModel):
    message: str

@app.on_event("startup")
async def startup_event():
    global llm_server_process, index, doc_embeddings
    doc_embeddings = embed_model.encode(DOCUMENTS)
    d = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(doc_embeddings).astype('float32'))

    model_path = os.path.join(MODELS_DIR, LLM_MODEL_GGUF)
    cmd = [LLAMA_SERVER_EXE, "-m", model_path, "--port", "8005", "-c", "2048", "--n-gpu-layers", "0"]
    llm_server_process = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    await asyncio.sleep(5)

@app.on_event("shutdown")
def shutdown_event():
    if llm_server_process:
        llm_server_process.terminate()

@app.post("/chat")
async def chat(request: ChatRequest):
    query_embedding = embed_model.encode([request.message])
    D, I = index.search(np.array(query_embedding).astype('float32'), k=1)
    context = DOCUMENTS[I[0][0]]
    
    url = "http://localhost:8005/v1/chat/completions"
    payload = {"messages": [{"role": "user", "content": f"Context: {context}\n\nQuestion: {request.message}"}]}
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, json=payload)
        response_text = resp.json()["choices"][0]["message"]["content"]
    
    return {"response": response_text, "context": context}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
