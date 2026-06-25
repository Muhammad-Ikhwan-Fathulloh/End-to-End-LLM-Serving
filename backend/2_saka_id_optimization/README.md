# Project 2: Dedicated Saka-NLP Optimization

Indonesian language optimization focus.

## Installation
```powershell
cd backend\2_saka_id_optimization
python -m venv venv
.\venv\Scripts\activate
pip install fastapi uvicorn httpx pydantic saka-nlp
uvicorn main:app --port 8002
```
