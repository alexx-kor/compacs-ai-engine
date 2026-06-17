"""Demo: upload document via HTTP and verify."""
import json
import sys
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:3080"
FILE = Path("data/demo_upload/operator_note.txt")
COLLECTION = "demo-upload"


def main() -> int:
    if not FILE.is_file():
        print("missing", FILE, file=sys.stderr)
        return 1

    with httpx.Client(timeout=180.0) as client:
        r = client.get(f"{BASE}/health")
        print("health", r.status_code, r.json())

        r = client.post(
            f"{BASE}/v1/collections",
            json={"id": COLLECTION, "name": "Demo Upload", "description": "HTTP upload test"},
        )
        print("create_collection", r.status_code, r.text[:300])
        if r.status_code not in (200, 400):
            return 1

        with FILE.open("rb") as handle:
            r = client.post(
                f"{BASE}/load",
                params={"collection_id": COLLECTION},
                files={"file": (FILE.name, handle, "text/plain")},
            )
        print("upload", r.status_code)
        print(json.dumps(r.json(), ensure_ascii=False, indent=2))

        r = client.get(f"{BASE}/sources", params={"format": "json"})
        sources = r.json().get("sources", [])
        uploaded = [s for s in sources if COLLECTION in (s.get("collection_id") or "")]
        print("sources_in_collection", len(uploaded))

        r = client.post(
            f"{BASE}/v1/query",
            json={
                "question": "Какой SFTP-адрес для загрузки ZIP на AI-сервер?",
                "collection_ids": [COLLECTION],
            },
        )
        print("query", r.status_code)
        q = r.json()
        print("answer_preview:", (q.get("answer") or "")[:400])
        print("sources:", q.get("sources"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
