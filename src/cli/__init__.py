from __future__ import annotations

"""Sentio CLI entrypoint.

Simplified to expose sub-commands directly (no nested *ingest ingest*).
"""

import typer

from src.cli.ingest import ingest_app, ingest_command  # noqa: WPS433 – runtime import OK

app = typer.Typer(help="Sentio vNext CLI", add_completion=False)

# Register commands ----------------------------------------------------
app.command(name="ingest")(ingest_command)

__all__ = ["app"]
