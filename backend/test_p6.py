from fastapi.testclient import TestClient
from p6_rag_pgvector import app, get_db_connection

def test_endpoints():
    # Set up FastApi TestClient
    with TestClient(app) as client:
        # Clear existing test documents from database to have clean verification
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM documents")
            conn.commit()
            conn.close()
            print("[TEST] Cleaned up documents table.")

        # 1. Test /add-text endpoint
        print("[TEST] Calling /add-text...")
        payload = {
            "texts": [
                "Bunga mawar merah tumbuh subur di halaman depan rumah.",
                "Kucing oren suka tidur siang di atas genteng tetangga."
            ]
        }
        res_text = client.post("/add-text", json=payload)
        print(f"[TEST] /add-text Response: {res_text.status_code} - {res_text.json()}")
        assert res_text.status_code == 200, "add-text endpoint failed"
        assert res_text.json()["inserted"] == 2

        # 2. Verify database records
        conn = get_db_connection()
        assert conn is not None, "Failed to connect to database for verification"
        with conn.cursor() as cur:
            cur.execute("SELECT id, content FROM documents ORDER BY id")
            rows = cur.fetchall()
            print(f"[TEST] Database contents ({len(rows)} records):")
            for r in rows:
                print(f" - ID: {r[0]}, Content: {r[1][:60]}...")
            assert len(rows) == 2, f"Expected 2 records, but got {len(rows)}"
        conn.close()
        
        print("[TEST] ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    import sys
    import traceback
    
    # Redirect all stdout/stderr to a log file
    sys.stdout = open("test_output.log", "w", encoding="utf-8")
    sys.stderr = sys.stdout
    
    try:
        test_endpoints()
    except Exception as e:
        print("[TEST] EXCEPTION OCCURRED:")
        traceback.print_exc()
        sys.exit(1)
    finally:
        sys.stdout.close()

