# Project 1: Basic LLM & Indonesia Optimization

Fundamental implementation of local LLM with Indonesian slang normalization.

## Installation
```powershell
cd backend\1_basic_llm
python -m venv venv
.\venv\Scripts\activate
pip install fastapi uvicorn httpx pydantic saka-nlp
uvicorn main:app --port 8001
```
