# Project 6: RAG with pgvector

## Installation
```powershell
cd backend\6_rag_pgvector
python -m venv venv
.\venv\Scripts\activate
pip install fastapi uvicorn httpx pydantic sentence-transformers psycopg2-binary pgvector
uvicorn main:app --port 8006
```
