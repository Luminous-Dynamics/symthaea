#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix CLI - Main Application

A beautiful command-line interface for the Mycelix Federated Learning system.
Built with Typer and Rich for an excellent user experience.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from . import __version__

# Initialize Typer app
app = typer.Typer(
    name="mycelix",
    help="Mycelix FL - Byzantine-Resilient Federated Learning CLI",
    add_completion=True,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# Rich console for beautiful output
console = Console()


# ASCII Art Banner
BANNER = r"""
[bold cyan]
    __  __                 _ _
   |  \/  |_   _  ___ ___| (_)_  __
   | |\/| | | | |/ __/ _ \ | \ \/ /
   | |  | | |_| | (_|  __/ | |>  <
   |_|  |_|\__, |\___\___|_|_/_/\_\
          |___/
[/bold cyan]
[dim]Byzantine-Resilient Federated Learning[/dim]
"""

TAGLINE = "[bold green]Secure. Decentralized. Resilient.[/bold green]"


def show_banner():
    """Display the Mycelix banner."""
    console.print(BANNER)
    console.print(f"  {TAGLINE}", justify="center")
    console.print(f"  [dim]Version {__version__}[/dim]\n", justify="center")


def version_callback(value: bool):
    """Show version and exit."""
    if value:
        console.print(f"[bold cyan]Mycelix CLI[/bold cyan] version [green]{__version__}[/green]")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
):
    """
    [bold cyan]Mycelix CLI[/bold cyan] - Byzantine-Resilient Federated Learning

    A unified command-line interface for running demos, benchmarks,
    and managing your Mycelix FL deployment.

    [bold]Examples:[/bold]

        [dim]# Run a healthcare scenario demo[/dim]
        $ mycelix demo --scenario healthcare --nodes 10

        [dim]# Initialize a new project[/dim]
        $ mycelix init my-fl-project

        [dim]# Run all benchmarks[/dim]
        $ mycelix benchmark --all

        [dim]# Validate system health[/dim]
        $ mycelix validate --quick

        [dim]# Check system status[/dim]
        $ mycelix status
    """
    pass


# Import and register commands
from .commands import demo, init, benchmark, validate, status

app.add_typer(demo.app, name="demo", help="Run interactive FL demos")
app.add_typer(init.app, name="init", help="Initialize a new Mycelix project")
app.add_typer(benchmark.app, name="benchmark", help="Run performance benchmarks")
app.add_typer(validate.app, name="validate", help="Validate system and run tests")
app.add_typer(status.app, name="status", help="Show system status")


@app.command()
def info():
    """
    Show detailed information about Mycelix.
    """
    show_banner()

    from rich.table import Table

    # System capabilities table
    table = Table(title="System Capabilities", show_header=True, header_style="bold magenta")
    table.add_column("Feature", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Description")

    # Check capabilities
    capabilities = [
        ("Byzantine Detection", True, "Multi-layer defense with POGQ, Shapley, HyperVectors"),
        ("HyperFeel Compression", True, "1000x+ gradient compression with semantic preservation"),
        ("P2P Gossip Protocol", True, "Fully decentralized communication"),
        ("Rust Backend", _check_rust_backend(), "High-performance aggregation"),
        ("GPU Acceleration", _check_gpu(), "CUDA/ROCm support for training"),
        ("Zero-Knowledge Proofs", _check_zk(), "ZK-STARK verification"),
    ]

    for feature, available, description in capabilities:
        status = "[green]Available[/green]" if available else "[yellow]Not Installed[/yellow]"
        table.add_row(feature, status, description)

    console.print(table)
    console.print()

    # Quick start panel
    quick_start = """
[bold]Quick Start:[/bold]

1. [cyan]mycelix demo --scenario healthcare[/cyan]
   Run a healthcare federated learning demo

2. [cyan]mycelix benchmark --all[/cyan]
   Run comprehensive performance benchmarks

3. [cyan]mycelix init my-project[/cyan]
   Create a new Mycelix project

4. [cyan]mycelix validate --quick[/cyan]
   Validate your installation

[dim]For more help: mycelix --help[/dim]
"""
    console.print(Panel(quick_start, title="Getting Started", border_style="green"))


def _check_rust_backend() -> bool:
    """Check if Rust backend is available."""
    try:
        import fl_aggregator
        return True
    except ImportError:
        return False


def _check_gpu() -> bool:
    """Check if GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _check_zk() -> bool:
    """Check if ZK-STARK is available."""
    try:
        from gen7_zkstark import ZKStarkProver
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    app()
