"""Print local vector store diagnostics (chunk counts and embedding dimensions)."""

from __future__ import annotations

import argparse
import json
import logging

from core.database import db
from core.logger import setup_logging

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local vector store stats")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    stats = db.store_stats()
    dims = db.embedding_dim_summary()

    log.info("%s", "\n1. LOCAL VECTOR STORE")
    log.info("   store_dir: %s", stats["store_dir"])
    log.info("   main_chunks: %s", stats["main_chunks"])
    log.info("   hypothesis_chunks: %s", stats["hypothesis_chunks"])
    log.info("   graph_chunks: %s", stats["graph_chunks"])

    log.info("%s", "\n2. Active table: %s", db.load_active_table())

    log.info("%s", "\n3. Embedding dimensions (hypothesis):")
    for dim, count in sorted(dims["hypothesis"].items()):
        log.info("   Dim %s: %s chunks", dim, count)

    log.info("%s", "\n4. Embedding dimensions (main):")
    for dim, count in sorted(dims["main"].items()):
        log.info("   Dim %s: %s chunks", dim, count)

    log.info("%s", "\n5. Search uses active store: %s", db.load_active_table())
    log.info("%s", "\nFull stats JSON:\n%s", json.dumps({**stats, "dims": dims}, indent=2))


if __name__ == "__main__":
    main()
