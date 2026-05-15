#!/usr/bin/env python3
"""Legacy CLI wrapper around unified ``app`` commands."""

from __future__ import annotations

import typer

from app import app as root_app

cli = typer.Typer(help="Legacy RAG CLI")


@cli.command("query")
def legacy_query(question: str) -> None:
    """Ask one question via the unified RAG service."""
    from app import query

    query(question)


@cli.command("ingest")
def legacy_ingest(
    input_dir: str = typer.Option("instructions/raw", "--input-dir"),
    force_reload: bool = typer.Option(False, "--force-reload"),
) -> None:
    """Ingest documents from a directory."""
    from app import ingest

    ingest(source=input_dir, force_reload=force_reload)


def main() -> None:
    if len(typer.main.get_command(cli).commands) > 0:
        cli()
    else:
        root_app()


if __name__ == "__main__":
    cli()
