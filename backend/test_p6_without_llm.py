import sys
import traceback
from fastapi.testclient import TestClient
from p6_rag_without_llm import app, get_db_connection

def test_learning_endpoints():
    # Set up FastApi TestClient
    with TestClient(app) as client:
        # 1. Test /health endpoint
        print("[TEST] Calling /health...")
        res_health = client.get("/health")
        print(f"[TEST] /health Response: {res_health.status_code} - {res_health.json()}")
        assert res_health.status_code == 200
        assert res_health.json()["embedding_model_loaded"] is True

        # 2. Reset database using /clear endpoint
        print("[TEST] Resetting database via /clear...")
        res_clear = client.post("/clear")
        print(f"[TEST] /clear Response: {res_clear.status_code} - {res_clear.json()}")
        assert res_clear.status_code == 200

        # 3. Add text items using /add-text endpoint
        print("[TEST] Calling /add-text...")
        payload = {
            "texts": [
                "Kambing gunung berkacamata hitam tinggal di puncak tertinggi Jayawijaya.",
                "Kucing oren suka tidur siang di atas genteng tetangga dari jam 12 hingga jam 2 siang.",
                "Bunga mawar merah tumbuh subur di halaman depan rumah Ikhwan.",
                "Ikhwan sangat suka meminum secangkir kopi luwak setiap sore pukul 3."
            ]
        }
        res_add = client.post("/add-text", json=payload)
        print(f"[TEST] /add-text Response: {res_add.status_code} - {res_add.json()}")
        assert res_add.status_code == 200
        assert res_add.json()["inserted"] == 4

        # 4. Search for similar texts using /search
        print("[TEST] Querying /search for 'hewan berdarah oren'...")
        search_payload = {
            "message": "kebiasaan kucing oren di siang hari",
            "top_k": 2
        }
        res_search = client.post("/search", json=search_payload)
        print(f"[TEST] /search Response: {res_search.status_code} - {res_search.json()}")
        assert res_search.status_code == 200
        
        results = res_search.json()["results"]
        assert len(results) == 2
        # First result should be the kucing oren text
        assert "Kucing oren" in results[0]["content"]
        # Distance should be small, similarity high
        assert results[0]["similarity"] > 0.4

        print("[TEST] ALL NO-LLM RAG ENDPOINTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    # Redirect all stdout/stderr to a log file
    sys.stdout = open("test_p6_without_llm_output.log", "w", encoding="utf-8")
    sys.stderr = sys.stdout
    
    try:
        test_learning_endpoints()
    except Exception as e:
        print("[TEST] EXCEPTION OCCURRED:")
        traceback.print_exc()
        sys.exit(1)
    finally:
        sys.stdout.close()
