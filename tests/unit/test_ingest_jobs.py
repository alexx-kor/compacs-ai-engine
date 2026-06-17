from __future__ import annotations

from unittest.mock import patch

from core.ingest_jobs import IngestJobService, JobStatus


def test_ingest_job_completes() -> None:
    service = IngestJobService(max_workers=1)

    class FakeDoc:
        filename = "note.txt"
        source = "collections/test/note.txt"
        uploaded_at = "2026-01-01T00:00:00+00:00"
        chunk_count = 2
        size_bytes = 100

    with patch("core.collections.collection_service") as mock_cs:
        mock_cs.ingest_document.return_value = FakeDoc()
        job = service.submit("test", "note.txt", b"hello world content for chunking test")

        import time

        for _ in range(50):
            current = service.get(job.id)
            assert current is not None
            if current.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                break
            time.sleep(0.05)

    final = service.get(job.id)
    assert final is not None
    assert final.status == JobStatus.COMPLETED
    assert final.result is not None
    assert final.result["document"]["chunk_count"] == 2
