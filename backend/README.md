# Panduan Menjalankan Backend P1-P6

Berikut adalah panduan untuk menjalankan masing-masing backend P1-P6. Semua file sudah diletakkan di luar folder masing-masing dan dapat dijalankan secara langsung.

## Prasyarat

Pastikan semua dependencies sudah terinstal:
```bash
pip install -r requirements.txt
```

## Cara Menjalankan Setiap Backend

### P1: Basic LLM & Indonesia Optimization
```bash
uvicorn p1_basic_llm:app --port 8001
```
- **Port API**: 8001
- **Port llama-server**: 8081
- **Fitur**: Normalisasi Bahasa Indonesia dengan Saka-NLP

---

### P2: Dedicated Saka-NLP Optimization
```bash
uvicorn p2_saka_id_optimization:app --port 8002
```
- **Port API**: 8002
- **Port llama-server**: 8082
- **Fitur**: Fokus pada normalisasi Bahasa Indonesia

---

### P3: Semantic Cache with pgvector
```bash
uvicorn p3_cache_pgvector:app --port 8003
```
- **Port API**: 8003
- **Port llama-server**: 8083
- **Fitur**: Cache respons LLM dengan pgvector
- **Peringatan**: Pastikan Docker Compose (pgvector) sudah berjalan

---

### P4: Feedback Loop with pgvector
```bash
uvicorn p4_feedback_pgvector:app --port 8004
```
- **Port API**: 8004
- **Port llama-server**: 8084
- **Fitur**: Simpan interaksi dan feedback pengguna
- **Peringatan**: Pastikan Docker Compose (pgvector) sudah berjalan

---

### P5: RAG with FAISS
```bash
uvicorn p5_rag_faiss:app --port 8005
```
- **Port API**: 8005
- **Port llama-server**: 8085
- **Fitur**: RAG dengan pencarian vektor FAISS (lokal)

---

### P6: RAG with pgvector
```bash
uvicorn p6_rag_pgvector:app --port 8006
```
- **Port API**: 8006
- **Port llama-server**: 8086
- **Fitur**: RAG dengan pencarian vektor pgvector
- **Peringatan**: Pastikan Docker Compose (pgvector) sudah berjalan

---

## Environment Variables (Opsional)

Anda dapat mengonfigurasi variabel lingkungan berikut sebelum menjalankan backend:

| Variabel | Deskripsi | Default |
|----------|-----------|---------|
| `LLM_MODEL_GGUF` | Nama file model GGUF di folder `../models` | `qwen2.5-0.5b-instruct-q4_k_m.gguf` |
| `LLAMA_NGL` | Jumlah layer yang di-offload ke GPU (0 = CPU saja) | `0` |
| `LLAMA_CTX` | Ukuran konteks model | `2048` |
| `LLAMA_PORT` | Port llama-server (untuk P1 gunakan 8081, P2 8082, dst.) | Sesuai backend |

Contoh penggunaan (PowerShell):
```powershell
$env:LLAMA_NGL="20"
uvicorn p1_basic_llm:app --port 8001
```

Contoh penggunaan (CMD):
```cmd
set LLAMA_NGL=20
uvicorn p1_basic_llm:app --port 8001
```

Contoh penggunaan (Bash):
```bash
export LLAMA_NGL=20
uvicorn p1_basic_llm:app --port 8001
```

---

## Endpoint

Setiap backend memiliki endpoint berikut:
- `GET /health`: Menunjukkan status app dan llama-server
- `POST /chat`: Mengirim permintaan chat ke LLM

Untuk backend P4, ada tambahan endpoint:
- `POST /feedback`: Menyimpan feedback pengguna
