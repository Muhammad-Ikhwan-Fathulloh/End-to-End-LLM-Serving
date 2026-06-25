# End-to-End LLM Serving Platform

A modular LLM serving platform demonstrating 6 distinct patterns for LLM integration, optimized for Indonesian language and high-performance local inference.

[![Docker](https://img.shields.io/badge/Docker-enabled-blue.svg)](https://www.docker.com/)
[![pgvector](https://img.shields.io/badge/pgvector-enabled-green.svg)](https://github.com/pgvector/pgvector)

## Project Modules

1. **Project 1: Basic LLM & Indonesia Optimization** (Port 8001)
   - Core LLM interaction with automatic Indonesian slang normalization using Saka-NLP.
2. **Project 2: Dedicated Saka-NLP Optimization** (Port 8002)
   - Specialized Indonesian language processing for professional contexts.
3. **Project 3: Semantic Cache (pgvector)** (Port 8003)
   - Cost and latency optimization using vector similarity search in PostgreSQL.
4. **Project 4: Feedback Loop (pgvector)** (Port 8004)
   - Interactive feedback collection (Like/Dislike) with persistent storage.
5. **Project 5: RAG with FAISS** (Port 8005)
   - Fast, local Retrieval-Augmented Generation using the Facebook AI Similarity Search library.
6. **Project 6: RAG with pgvector** (Port 8006)
   - Production-grade persistent RAG using a relational database with vector capabilities.

---

## Getting Started

### 1. Install Docker Desktop
To run the vector database and UI, you need Docker Desktop.
- Download and install: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
- Ensure Docker is running before proceeding.

### 2. Start the Database
Run the following command in the root directory to spin up the PostgreSQL (pgvector) and PgWeb UI:
```bash
docker-compose up -d
```
- **PostgreSQL**: `localhost:5432` (user: `user`, pass: `password`, db: `mydatabase`)
- **PgWeb UI**: `http://localhost:8081`

### 3. Local Model Setup
- Place your Qwen 2.5 GGUF model in the `models/` folder.
- Ensure the filename in `main.py` matches your model file.

### 4. Running a Module
Navigate to any module folder and follow the instructions in its `README.md`. Example:
```bash
cd backend/1_basic_llm
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --port 8001
```

### 5. Frontend Dashboard
Simply open `frontend/index.html` in your browser to access the central management interface.

---

## Deployment to GitHub
To push your changes to your repository:
```bash
git push -u origin main
```

---

## Resources
- **Saka-NLP**: [Muhammad-Ikhwan-Fathulloh/Saka-NLP](https://github.com/Muhammad-Ikhwan-Fathulloh/Saka-NLP)
- **Llama.cpp**: [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
- **pgvector**: [pgvector/pgvector](https://github.com/pgvector/pgvector)
