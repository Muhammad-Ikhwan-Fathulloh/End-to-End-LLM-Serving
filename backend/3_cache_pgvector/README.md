# Project 3: Semantic Cache

## Installation
```powershell
cd backend\3_cache_pgvector
python -m venv venv
.\venv\Scripts\activate
pip install fastapi uvicorn httpx pydantic sentence-transformers psycopg2-binary pgvector
uvicorn main:app --port 8003
```
