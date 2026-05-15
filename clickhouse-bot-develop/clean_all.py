#!/usr/bin/env python3
"""Reset local vector store files (main and hypothesis datasets)."""

from __future__ import annotations

import argparse
import logging

from core.database import db
from core.logger import setup_logging

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for cleanup workflow."""
    parser = argparse.ArgumentParser(description="Reset local RAG vector store files")
    parser.add_argument(
        "--active-table",
        choices=["main", "hypothesis"],
        default="main",
        help="Table mode to keep as active after cleanup",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging")
    return parser.parse_args()


def main() -> None:
    """Recreate main and hypothesis tables and set active table."""
    args = parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    log.info("%s", "=" * 60)
    log.info("CLEANING SUPPORTED TABLES")
    log.info("%s", "=" * 60)

    log.info("%s", "\n1. Recreating hypothesis_chunks...")
    db.init_hypothesis_database(force_recreate=True)
    log.info("   [OK]")

    log.info("%s", "\n2. Recreating rag_chunks...")
    db.init_main_database(force_recreate=True)
    log.info("   [OK]")

    log.info("\n3. Switching active table to %s...", args.active_table)
    db.set_active_table(args.active_table)
    log.info("   Active table: %s", db.load_active_table())

    log.info("%s", "\n" + "=" * 60)
    log.info("TABLE CLEANUP COMPLETE")
    log.info("%s", "=" * 60)


if __name__ == "__main__":
    main()
