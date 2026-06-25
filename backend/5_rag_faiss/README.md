# Project 5: RAG with FAISS

## Installation
```powershell
cd backend\5_rag_faiss
python -m venv venv
.\venv\Scripts\activate
pip install fastapi uvicorn httpx pydantic sentence-transformers faiss-cpu
uvicorn main:app --port 8005
```
