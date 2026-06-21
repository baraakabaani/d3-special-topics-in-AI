"""
Run this after `docker-compose up -d` to verify all three services are ready.
Usage: python scripts/check_services.py
"""

import sys
import os
from dotenv import load_dotenv

# host scripts use .env.local — override=True so shell vars don't shadow it
load_dotenv(".env.local", override=True)


def check_mongo():
    try:
        from pymongo import MongoClient
        uri = os.getenv("MONGO_URI", "mongodb://admin:changeme@localhost:27017")
        client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        print("  MongoDB   OK")
        return True
    except Exception as e:
        print(f"  MongoDB   FAIL — {e}")
        return False


def check_qdrant():
    try:
        import requests
        r = requests.get("http://localhost:6333/healthz", timeout=3)
        if r.status_code == 200:
            print("  Qdrant    OK")
            return True
        print(f"  Qdrant    FAIL — status {r.status_code}")
        return False
    except Exception as e:
        print(f"  Qdrant    FAIL — {e}")
        return False


def check_neo4j():
    try:
        import requests
        r = requests.get("http://localhost:7474", timeout=3)
        if r.status_code == 200:
            print("  Neo4j     OK")
            return True
        print(f"  Neo4j     FAIL — status {r.status_code}")
        return False
    except Exception as e:
        print(f"  Neo4j     FAIL — {e}")
        return False


if __name__ == "__main__":
    print("Checking services...")
    results = [check_mongo(), check_qdrant(), check_neo4j()]
    if all(results):
        print("\nAll services up. Team can start.")
        sys.exit(0)
    else:
        print("\nSome services failed. Run: docker-compose up -d")
        sys.exit(1)
