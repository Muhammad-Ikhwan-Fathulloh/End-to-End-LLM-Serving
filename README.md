# End-to-End LLM Serving Platform

Platform serving LLM modular yang mendemonstrasikan 6 pola integrasi LLM berbeda, dioptimalkan untuk bahasa Indonesia dan inferensi lokal berperforma tinggi.

[![Docker](https://img.shields.io/badge/Docker-enabled-blue.svg)](https://www.docker.com/)
[![pgvector](https://img.shields.io/badge/pgvector-enabled-green.svg)](https://github.com/pgvector/pgvector)

## Struktur Proyek

```
.
├── backend/
│   ├── p1_basic_llm.py          (Port 8001) - Basic LLM
│   ├── p2_saka_id_optimization.py (Port 8002) - Saka-NLP Optimization
│   ├── p3_cache_pgvector.py     (Port 8003) - Semantic Cache (pgvector)
│   ├── p4_feedback_pgvector.py  (Port 8004) - Feedback Loop (pgvector)
│   ├── p5_rag_faiss.py          (Port 8005) - RAG with FAISS
│   ├── p6_rag_pgvector.py       (Port 8006) - RAG with pgvector
│   ├── main.py
│   ├── requirements.txt
│   ├── .env.example             (Contoh konfigurasi environment)
│   └── bin/                     (llama-server executable)
├── frontend/
│   └── index.html
├── models/                      (tempat model GGUF)
└── docker-compose.yml
```

## Modul Proyek

1. **P1: Basic LLM** (Port 8001)
   - Interaksi LLM dasar
2. **P2: Saka-NLP Optimization** (Port 8002)
   - Optimisasi bahasa Indonesia khusus
3. **P3: Semantic Cache (pgvector)** (Port 8003)
   - Optimisasi biaya dan latency menggunakan pencarian kesamaan vektor di PostgreSQL
4. **P4: Feedback Loop (pgvector)** (Port 8004)
   - Koleksi feedback interaktif (Like/Dislike) dengan penyimpanan persisten
5. **P5: RAG with FAISS** (Port 8005)
   - Retrieval-Augmented Generation cepat dan lokal menggunakan FAISS
6. **P6: RAG with pgvector** (Port 8006)
   - RAG persisten grade produksi menggunakan database relasional dengan kemampuan vektor

---

## Memulai

### 1. Instal Docker Desktop
Untuk menjalankan database vektor dan UI, Anda memerlukan Docker Desktop.
- Unduh dan instal: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
- Pastikan Docker berjalan sebelum melanjutkan.

### 2. Konfigurasi Environment
Salin file contoh environment di folder `backend` dan sesuaikan sesuai kebutuhan:
```bash
cd backend
cp .env.example .env
```

Edit file `backend/.env` untuk mengkonfigurasi:
- PostgreSQL (pgvector)
- Model LLM
- Port dan parameter lainnya

### 3. Mulai Database
Jalankan perintah berikut di direktori root untuk menyalakan PostgreSQL (pgvector) dan PgWeb UI:
```bash
docker-compose up -d
```
- **PostgreSQL**: `localhost:5432` (sesuai `.env`)
- **PgWeb UI**: `http://localhost:8081`

Jalankan perintah berikut di direktori root untuk memberhentikan PostgreSQL (pgvector) dan PgWeb UI:
```bash
docker-compose down
```

### 4. Setup Model Lokal
- Tempatkan model GGUF Anda di folder `models/`.
- Pastikan nama file di `.env` sesuai dengan file model Anda.

Contoh model yang didukung:
- `qwen2.5-0.5b-instruct-q4_k_m.gguf`

### 5. Instal Dependensi Backend
```bash
cd backend
pip install -r requirements.txt
```

### 6. Menjalankan Modul
Jalankan modul yang diinginkan dengan uvicorn. Contoh untuk P3:
```bash
cd backend
uvicorn p3_cache_pgvector:app --port 8003
```

### 7. Frontend Dashboard
Cukup buka `frontend/index.html` di browser Anda untuk mengakses antarmuka manajemen pusat.

---

## Deploy ke GitHub
Untuk mendorong perubahan ke repositori Anda:
```bash
git push -u origin main
```

---

## Sumber Daya
- **Saka-NLP**: [Muhammad-Ikhwan-Fathulloh/Saka-NLP](https://github.com/Muhammad-Ikhwan-Fathulloh/Saka-NLP)
- **Llama.cpp**: [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
- **pgvector**: [pgvector/pgvector](https://github.com/pgvector/pgvector)
