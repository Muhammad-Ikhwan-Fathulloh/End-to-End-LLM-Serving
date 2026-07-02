"""
P6: Mempelajari RAG dengan pgvector Tanpa LLM (FastAPI Version)
==================================================================
FastAPI App pembelajaran untuk menunjukkan *Retrieval* (pengambilan dokumen) 
pada sistem RAG menggunakan pgvector di PostgreSQL tanpa melibatkan LLM (Generation).

Jalankan dengan:
    uvicorn p6_rag_without_llm:app --port 8007

Pastikan Docker Compose (pgvector) sudah jalan sebelum menjalankan aplikasi ini.
"""

import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

# Muat variabel environment dari file .env
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

DB_CONFIG = os.environ.get("DB_CONFIG", "dbname=mydatabase user=user password=password host=localhost port=5432")

# Global state untuk menyimpan model & status
state = {"embed_model": None}

def get_db_connection(register: bool = True):
    """Membuka koneksi ke PostgreSQL database."""
    try:
        conn = psycopg2.connect(DB_CONFIG)
        if register:
            register_vector(conn)
        return conn
    except Exception as e:
        print(f"[Learning App] Error koneksi DB: {e}")
        return None

def init_database() -> bool:
    """Membuat ekstensi pgvector dan tabel documents."""
    print("[Learning App] Inisialisasi Database PostgreSQL...")
    conn = get_db_connection(register=False)
    if not conn:
        print("[Learning App] ✗ Koneksi gagal. Pastikan Docker pgvector sudah aktif.")
        return False
    
    try:
        with conn.cursor() as cur:
            # Pastikan ekstensi vector aktif di database
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()

        # Daftarkan tipe data vector pada koneksi
        register_vector(conn)
        with conn.cursor() as cur:
            # Buat tabel untuk menyimpan potongan teks (chunks) dan embedding-nya
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector(384)
                )
            """)
        conn.commit()
        print("[Learning App] ✓ Database & tabel 'documents' siap.")
        return True
    except Exception as e:
        print(f"[Learning App] ✗ Gagal inisialisasi DB: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Inisialisasi model embedding SentenceTransformer
    print("[Learning App] Memuat model embedding (all-MiniLM-L6-v2)...")
    state["embed_model"] = SentenceTransformer("all-MiniLM-L6-v2")
    print("[Learning App] ✓ Model embedding siap.")

    # 2. Setup skema database
    init_database()
    yield

app = FastAPI(
    title="P6 RAG Pembelajaran (Tanpa LLM)", 
    description="Aplikasi pembelajaran konsep retrieval menggunakan pgvector",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- PyDantic Request/Response Models ---

class AddTextRequest(BaseModel):
    texts: list[str] = Field(..., description="Daftar teks mentah untuk ditambahkan ke database")

class SearchRequest(BaseModel):
    message: str = Field(..., description="Query pencarian semantik")
    top_k: int = Field(3, ge=1, le=10, description="Jumlah dokumen teratas yang ingin diambil")

# --- Endpoints ---

@app.get("/health")
async def health():
    """Mengecek status aplikasi & koneksi database."""
    conn = get_db_connection(register=False)
    db_ok = conn is not None
    if conn:
        conn.close()
    
    return {
        "status": "ok",
        "database_connected": db_ok,
        "embedding_model_loaded": state["embed_model"] is not None
    }

@app.post("/clear")
async def clear_database():
    """Mengosongkan semua data dokumen dari database (untuk keperluan reset testing)."""
    conn = get_db_connection(register=False)
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE documents RESTART IDENTITY")
        conn.commit()
        return {"status": "success", "message": "Database documents cleared"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        conn.close()

@app.post("/add-text")
async def add_text(request: AddTextRequest):
    """
    Menghasilkan embedding untuk setiap teks mentah di dalam list,
    lalu menyimpannya di database pgvector.
    """
    if not state["embed_model"]:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")

    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    inserted_count = 0
    try:
        with conn.cursor() as cur:
            for text in request.texts:
                if not text.strip():
                    continue
                # 1. Kalkulasikan embedding dari teks mentah
                embedding = state["embed_model"].encode(text)
                
                # 2. Insert konten beserta embedding-nya ke database
                cur.execute(
                    "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
                    (text, embedding)
                )
                inserted_count += 1
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        conn.close()

    return {"status": "success", "inserted": inserted_count}

@app.post("/search")
async def search(request: SearchRequest):
    """
    Mengambil potongan dokumen terdekat secara semantis menggunakan
    operator cosine distance (<=>) di pgvector.
    """
    if not state["embed_model"]:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")

    # 1. Konversikan teks query pencarian menjadi embedding vektor
    query_vector = state["embed_model"].encode(request.message)

    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    results = []
    try:
        with conn.cursor() as cur:
            # Menggunakan operator <=> untuk menghitung cosine distance antara vector disimpan dengan query_vector.
            # Jarak yang lebih kecil berarti kemiripan semantik yang lebih tinggi.
            cur.execute("""
                SELECT id, content, embedding <=> %s AS distance 
                FROM documents 
                ORDER BY embedding <=> %s 
                LIMIT %s
            """, (query_vector, query_vector, request.top_k))
            
            rows = cur.fetchall()
            for row in rows:
                doc_id, content, distance = row
                results.append({
                    "id": doc_id,
                    "content": content,
                    "distance": float(distance),
                    "similarity": round(1.0 - float(distance), 4) # cosine_similarity = 1 - cosine_distance
                })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")
    finally:
        conn.close()

    return {
        "query": request.message,
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)
