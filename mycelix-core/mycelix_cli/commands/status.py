#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix Status Command

Show system status, configuration, and running services.
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

import numpy as np

app = typer.Typer(
    name="status",
    help="Show system status",
    no_args_is_help=False,
)

console = Console()


def get_system_info() -> Dict[str, Any]:
    """Collect system information."""
    return {
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "platform_version": platform.version(),
        "platform_release": platform.release(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "python_path": sys.executable,
        "cpu_count": os.cpu_count(),
    }


def get_dependency_versions() -> Dict[str, Optional[str]]:
    """Get versions of installed dependencies."""
    dependencies = {}

    # NumPy (always available if CLI runs)
    dependencies["numpy"] = np.__version__

    # PyTorch
    try:
        import torch
        dependencies["torch"] = torch.__version__
        dependencies["cuda"] = torch.version.cuda if torch.cuda.is_available() else None
    except ImportError:
        dependencies["torch"] = None

    # Rich/Typer (always available)
    try:
        import rich
        dependencies["rich"] = rich.__version__
    except (ImportError, AttributeError):
        dependencies["rich"] = "installed"

    try:
        import typer
        dependencies["typer"] = typer.__version__
    except (ImportError, AttributeError):
        dependencies["typer"] = "installed"

    # Optional dependencies
    optional = [
        ("pyyaml", "yaml"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("websockets", "websockets"),
        ("aiohttp", "aiohttp"),
    ]

    for pkg_name, import_name in optional:
        try:
            mod = __import__(import_name)
            dependencies[pkg_name] = getattr(mod, "__version__", "installed")
        except ImportError:
            dependencies[pkg_name] = None

    return dependencies


def check_gpu_status() -> Dict[str, Any]:
    """Check GPU availability and status."""
    gpu_info = {
        "available": False,
        "devices": [],
        "memory_total": None,
        "memory_used": None,
    }

    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["device_count"] = torch.cuda.device_count()

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info["devices"].append({
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": props.total_memory / 1e9,
                    "compute_capability": f"{props.major}.{props.minor}",
                })

            # Memory info for device 0
            if torch.cuda.device_count() > 0:
                mem_total = torch.cuda.get_device_properties(0).total_memory
                mem_allocated = torch.cuda.memory_allocated(0)
                gpu_info["memory_total_gb"] = mem_total / 1e9
                gpu_info["memory_used_gb"] = mem_allocated / 1e9
    except ImportError:
        pass

    return gpu_info


def check_docker_status() -> Dict[str, Any]:
    """Check Docker status."""
    docker_info = {
        "installed": False,
        "running": False,
        "containers": [],
    }

    try:
        # Check if docker is installed
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            docker_info["installed"] = True
            docker_info["version"] = result.stdout.strip()

            # Check if Docker is running
            result = subprocess.run(
                ["docker", "ps", "-q"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                docker_info["running"] = True

                # List Mycelix containers
                result = subprocess.run(
                    ["docker", "ps", "--filter", "name=mycelix", "--format", "{{.Names}}:{{.Status}}"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.stdout.strip():
                    for line in result.stdout.strip().split("\n"):
                        if ":" in line:
                            name, status = line.split(":", 1)
                            docker_info["containers"].append({
                                "name": name,
                                "status": status,
                            })
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return docker_info


def find_config_file() -> Optional[Path]:
    """Find Mycelix configuration file."""
    search_paths = [
        Path.cwd() / "config" / "mycelix.yaml",
        Path.cwd() / "mycelix.yaml",
        Path.home() / ".config" / "mycelix" / "config.yaml",
        Path("/etc/mycelix/config.yaml"),
    ]

    for path in search_paths:
        if path.exists():
            return path

    return None


def parse_config(config_path: Path) -> Dict[str, Any]:
    """Parse configuration file."""
    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


@app.callback(invoke_without_command=True)
def status_main(
    ctx: typer.Context,
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed information",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON",
    ),
):
    """
    Show Mycelix system status.

    Displays system information, dependency versions,
    GPU status, and running services.

    [bold]Examples:[/bold]

        mycelix status
        mycelix status --verbose
        mycelix status --json
    """
    if ctx.invoked_subcommand is not None:
        return

    # Collect information
    sys_info = get_system_info()
    deps = get_dependency_versions()
    gpu_info = check_gpu_status()
    docker_info = check_docker_status()
    config_path = find_config_file()

    if json_output:
        import json
        output = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "system": sys_info,
            "dependencies": deps,
            "gpu": gpu_info,
            "docker": docker_info,
            "config_path": str(config_path) if config_path else None,
        }
        console.print(json.dumps(output, indent=2))
        return

    # Header
    console.print("\n[bold cyan]Mycelix System Status[/bold cyan]")
    console.print(f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n")

    # System Info Table
    sys_table = Table(title="System Information", show_header=False, box=None)
    sys_table.add_column("Key", style="cyan")
    sys_table.add_column("Value")

    sys_table.add_row("Hostname", sys_info["hostname"])
    sys_table.add_row("Platform", f"{sys_info['platform']} ({sys_info['architecture']})")
    sys_table.add_row("Python", f"{sys_info['python_version']}")
    sys_table.add_row("CPUs", str(sys_info["cpu_count"]))

    console.print(sys_table)
    console.print()

    # Dependencies Table
    dep_table = Table(title="Dependencies", show_header=True, header_style="bold")
    dep_table.add_column("Package", style="cyan")
    dep_table.add_column("Version", justify="right")
    dep_table.add_column("Status", justify="center")

    core_deps = ["numpy", "torch", "rich", "typer"]
    optional_deps = ["pyyaml", "scipy", "scikit-learn", "websockets", "aiohttp"]

    for pkg in core_deps:
        version = deps.get(pkg)
        if version:
            dep_table.add_row(pkg, version, "[green]OK[/green]")
        else:
            dep_table.add_row(pkg, "-", "[yellow]Not installed[/yellow]")

    if verbose:
        for pkg in optional_deps:
            version = deps.get(pkg)
            if version:
                dep_table.add_row(pkg, version, "[green]OK[/green]")
            else:
                dep_table.add_row(pkg, "-", "[dim]Optional[/dim]")

    console.print(dep_table)
    console.print()

    # GPU Status
    if gpu_info["available"]:
        gpu_table = Table(title="GPU Status", show_header=True)
        gpu_table.add_column("Device", style="cyan")
        gpu_table.add_column("Name")
        gpu_table.add_column("Memory", justify="right")
        gpu_table.add_column("Compute", justify="center")

        for device in gpu_info["devices"]:
            gpu_table.add_row(
                f"GPU {device['index']}",
                device["name"],
                f"{device['total_memory_gb']:.1f} GB",
                device["compute_capability"],
            )

        console.print(gpu_table)
        console.print()
    else:
        console.print("[yellow]GPU: Not available[/yellow]\n")

    # Docker Status
    if verbose:
        docker_status = "[green]Running[/green]" if docker_info["running"] else (
            "[yellow]Installed but not running[/yellow]" if docker_info["installed"] else "[dim]Not installed[/dim]"
        )
        console.print(f"Docker: {docker_status}")

        if docker_info["containers"]:
            console.print("\nMycelix Containers:")
            for container in docker_info["containers"]:
                console.print(f"  [cyan]{container['name']}[/cyan]: {container['status']}")
        console.print()

    # Configuration
    if config_path:
        console.print(f"Config: [green]{config_path}[/green]")
        if verbose:
            config = parse_config(config_path)
            if config:
                console.print(f"  Nodes: {config.get('network', {}).get('num_nodes', 'N/A')}")
                console.print(f"  Byzantine Detection: {config.get('byzantine', {}).get('enabled', 'N/A')}")
    else:
        console.print("Config: [dim]No configuration file found[/dim]")

    console.print()


@app.command()
def deps():
    """Show detailed dependency information."""
    console.print("\n[bold cyan]Dependency Status[/bold cyan]\n")

    deps = get_dependency_versions()

    # Core dependencies
    table = Table(title="Core Dependencies", show_header=True)
    table.add_column("Package", style="cyan")
    table.add_column("Version", justify="right")
    table.add_column("Status", justify="center")

    core = ["numpy", "torch", "rich", "typer"]
    for pkg in core:
        version = deps.get(pkg)
        status = "[green]Installed[/green]" if version else "[red]Missing[/red]"
        table.add_row(pkg, version or "-", status)

    console.print(table)
    console.print()

    # Optional dependencies
    table = Table(title="Optional Dependencies", show_header=True)
    table.add_column("Package", style="cyan")
    table.add_column("Version", justify="right")
    table.add_column("Status", justify="center")

    optional = ["pyyaml", "scipy", "scikit-learn", "pandas", "matplotlib", "websockets", "aiohttp"]
    for pkg in optional:
        version = deps.get(pkg)
        status = "[green]Installed[/green]" if version else "[dim]Not installed[/dim]"
        table.add_row(pkg, version or "-", status)

    console.print(table)

    # CUDA status
    if deps.get("cuda"):
        console.print(f"\n[green]CUDA[/green]: {deps['cuda']}")
    elif deps.get("torch"):
        console.print("\n[yellow]CUDA[/yellow]: Not available (CPU mode)")


@app.command()
def gpu():
    """Show detailed GPU information."""
    console.print("\n[bold cyan]GPU Status[/bold cyan]\n")

    gpu_info = check_gpu_status()

    if not gpu_info["available"]:
        console.print("[yellow]No GPU available[/yellow]")
        console.print("\nPossible reasons:")
        console.print("  - PyTorch not installed with CUDA support")
        console.print("  - No NVIDIA GPU detected")
        console.print("  - CUDA drivers not installed")
        return

    for device in gpu_info["devices"]:
        console.print(f"[bold]GPU {device['index']}: {device['name']}[/bold]")
        console.print(f"  Total Memory: {device['total_memory_gb']:.1f} GB")
        console.print(f"  Compute Capability: {device['compute_capability']}")
        console.print()

    if gpu_info.get("memory_total_gb"):
        used_pct = (gpu_info["memory_used_gb"] / gpu_info["memory_total_gb"]) * 100
        console.print(f"Memory Usage: {gpu_info['memory_used_gb']:.2f} / {gpu_info['memory_total_gb']:.1f} GB ({used_pct:.1f}%)")


@app.command()
def config():
    """Show configuration status."""
    console.print("\n[bold cyan]Configuration Status[/bold cyan]\n")

    config_path = find_config_file()

    if not config_path:
        console.print("[yellow]No configuration file found[/yellow]")
        console.print("\nSearched locations:")
        console.print("  - ./config/mycelix.yaml")
        console.print("  - ./mycelix.yaml")
        console.print("  - ~/.config/mycelix/config.yaml")
        console.print("  - /etc/mycelix/config.yaml")
        console.print("\n[dim]Run 'mycelix init' to create a new project with configuration.[/dim]")
        return

    console.print(f"Config File: [green]{config_path}[/green]\n")

    config = parse_config(config_path)
    if not config:
        console.print("[red]Error parsing configuration file[/red]")
        return

    # Display configuration
    tree = Tree(f"[cyan]{config_path.name}[/cyan]")

    def add_to_tree(node, data, prefix=""):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    child = node.add(f"[bold]{key}[/bold]")
                    add_to_tree(child, value, prefix + "  ")
                elif isinstance(value, list):
                    child = node.add(f"[bold]{key}[/bold]: [{len(value)} items]")
                else:
                    node.add(f"[cyan]{key}[/cyan]: {value}")

    add_to_tree(tree, config)
    console.print(tree)


@app.command()
def env():
    """Show environment variables related to Mycelix."""
    console.print("\n[bold cyan]Environment Variables[/bold cyan]\n")

    mycelix_vars = []
    python_vars = []
    cuda_vars = []

    for key, value in os.environ.items():
        key_lower = key.lower()
        if "mycelix" in key_lower:
            mycelix_vars.append((key, value))
        elif "python" in key_lower or key in ["VIRTUAL_ENV", "CONDA_DEFAULT_ENV"]:
            python_vars.append((key, value))
        elif "cuda" in key_lower or "nvidia" in key_lower:
            cuda_vars.append((key, value))

    if mycelix_vars:
        console.print("[bold]Mycelix Variables:[/bold]")
        for key, value in mycelix_vars:
            console.print(f"  [cyan]{key}[/cyan]={value}")
        console.print()

    console.print("[bold]Python Environment:[/bold]")
    for key, value in python_vars[:5]:  # Limit to 5
        console.print(f"  [cyan]{key}[/cyan]={value[:50]}{'...' if len(value) > 50 else ''}")
    console.print()

    if cuda_vars:
        console.print("[bold]CUDA Variables:[/bold]")
        for key, value in cuda_vars[:5]:
            console.print(f"  [cyan]{key}[/cyan]={value[:50]}{'...' if len(value) > 50 else ''}")


if __name__ == "__main__":
    app()
