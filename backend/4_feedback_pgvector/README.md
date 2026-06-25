# Project 4: Feedback Loop

## Installation
```powershell
cd backend\4_feedback_pgvector
python -m venv venv
.\venv\Scripts\activate
pip install fastapi uvicorn httpx pydantic sentence-transformers psycopg2-binary pgvector
uvicorn main:app --port 8004
```
