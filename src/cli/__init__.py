"""
Command-line interface modules for Sentio vNext.
"""

import typer

from src.cli.ingest import ingest_app

# Create root CLI app
app = typer.Typer(help="Sentio vNext CLI", add_completion=False)

# Add subcommands
app.add_typer(ingest_app, name="ingest")

__all__ = ["app", "ingest_app"]
