"""
LangGraph Studio launcher for Sentio RAG system.

This module provides a command to launch LangGraph Studio for
visualization and debugging of the RAG pipeline.
"""

import logging
import os
import subprocess
import sys
import typer
from typing_extensions import Annotated
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

# Create Typer app
studio_app = typer.Typer(help="LangGraph Studio commands")


@studio_app.callback(invoke_without_command=True)
def callback(ctx: typer.Context):
    """
    Launch LangGraph Studio for visualization.
    
    If no subcommand is provided, the start command will be invoked.
    """
    if ctx.invoked_subcommand is None:
        # No subcommand was provided, so run the start command
        ctx.invoke(start_studio)


@studio_app.command("start")
def start_studio(
    port: Annotated[int, typer.Option("--port", "-p", help="Port for LangGraph Studio")] = 8001,
):
    """
    Start LangGraph Studio for visualization.
    
    This command launches LangGraph Studio interface for visualizing
    and debugging the RAG pipeline graph.
    """
    try:
        # Проверяем, установлен ли langgraph-cli
        try:
            subprocess.run(["langgraph", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            typer.echo("LangGraph CLI not found. Installing required packages...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "langgraph-cli[inmem]>=0.0.15"
            ], check=True)
        
        # Запускаем LangGraph Studio
        typer.echo(f"Starting LangGraph Studio on port {port}...")
        
        # Используем subprocess для запуска команды langgraph dev
        cmd = ["langgraph", "dev", "--port", str(port), "--allow-blocking"]
        typer.echo(f"Running: {' '.join(cmd)}")
        
        # Запускаем процесс
        process = subprocess.run(cmd)
        
        if process.returncode != 0:
            typer.echo(f"LangGraph Studio exited with code {process.returncode}", err=True)
            raise typer.Exit(code=process.returncode)
            
    except KeyboardInterrupt:
        typer.echo("LangGraph Studio stopped by user")
    except Exception as e:
        typer.echo(f"Error starting LangGraph Studio: {e}", err=True)
        raise typer.Exit(code=1)


@studio_app.command("install-deps")
def install_studio_dependencies():
    """
    Install LangGraph Studio dependencies.
    
    This command installs the necessary dependencies for running LangGraph Studio.
    """
    import subprocess
    import sys
    
    try:
        typer.echo("Installing LangGraph Studio dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "langgraph-cli[inmem]>=0.0.15"
        ], check=True)
        typer.echo("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        typer.echo(f"Error installing dependencies: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    studio_app() 