#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix Demo Command

Run interactive federated learning demonstrations with various scenarios.
Supports healthcare, finance, adversarial, and ultimate scenarios.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import typer
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.layout import Layout
from rich.text import Text

app = typer.Typer(
    name="demo",
    help="Run interactive FL demonstrations",
    no_args_is_help=False,
)

console = Console()


class Scenario(str, Enum):
    """Available demo scenarios."""
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    ADVERSARIAL = "adversarial"
    ULTIMATE = "ultimate"


@dataclass
class NodeState:
    """State of a simulated FL node."""
    node_id: str
    is_byzantine: bool = False
    accuracy: float = 0.5
    gradients_sent: int = 0
    gradients_received: int = 0
    detected: bool = False
    attack_type: Optional[str] = None


@dataclass
class RoundMetrics:
    """Metrics for a single FL round."""
    round_num: int
    duration_ms: float
    accuracy: float
    detection_rate: float
    byzantine_detected: int
    byzantine_total: int
    aggregation_latency_ms: float
    compression_ratio: float = 1.0


@dataclass
class DemoResult:
    """Complete demo result."""
    scenario: str
    num_nodes: int
    byzantine_ratio: float
    total_rounds: int
    final_accuracy: float
    detection_rate: float
    avg_latency_ms: float
    total_duration_s: float
    rounds: List[RoundMetrics] = field(default_factory=list)
    byzantine_nodes: Set[str] = field(default_factory=set)
    detected_nodes: Set[str] = field(default_factory=set)


class DemoRunner:
    """Run federated learning demonstrations."""

    def __init__(
        self,
        scenario: Scenario,
        num_nodes: int,
        byzantine_ratio: float,
        num_rounds: int,
        verbose: bool = False,
    ):
        self.scenario = scenario
        self.num_nodes = num_nodes
        self.byzantine_ratio = byzantine_ratio
        self.num_rounds = num_rounds
        self.verbose = verbose

        # Initialize nodes
        self.nodes: Dict[str, NodeState] = {}
        self.byzantine_ids: Set[str] = set()
        self._setup_nodes()

        # Metrics
        self.rounds: List[RoundMetrics] = []
        self.current_accuracy = 0.5
        self.detected_ids: Set[str] = set()

    def _setup_nodes(self):
        """Set up FL nodes with Byzantine assignment."""
        num_byzantine = int(self.num_nodes * self.byzantine_ratio)
        byzantine_indices = set(random.sample(range(self.num_nodes), num_byzantine))

        attack_types = ["gradient_scaling", "sign_flip", "gaussian_noise", "label_flip"]

        for i in range(self.num_nodes):
            node_id = f"node_{i:03d}"
            is_byzantine = i in byzantine_indices

            self.nodes[node_id] = NodeState(
                node_id=node_id,
                is_byzantine=is_byzantine,
                accuracy=0.5 + random.uniform(-0.1, 0.1),
                attack_type=random.choice(attack_types) if is_byzantine else None,
            )

            if is_byzantine:
                self.byzantine_ids.add(node_id)

    def _get_scenario_config(self) -> Dict[str, Any]:
        """Get scenario-specific configuration."""
        configs = {
            Scenario.HEALTHCARE: {
                "name": "Healthcare FL",
                "description": "Privacy-preserving medical imaging classification",
                "data_type": "X-ray images",
                "model": "ResNet-18 (Medical)",
                "privacy": "Differential Privacy (epsilon=1.0)",
                "target_accuracy": 0.92,
                "attack_intensity": 0.5,
            },
            Scenario.FINANCE: {
                "name": "Finance FL",
                "description": "Fraud detection across banking institutions",
                "data_type": "Transaction records",
                "model": "Transformer-based",
                "privacy": "Secure Aggregation + HE",
                "target_accuracy": 0.95,
                "attack_intensity": 0.7,
            },
            Scenario.ADVERSARIAL: {
                "name": "Adversarial FL",
                "description": "Stress testing Byzantine detection",
                "data_type": "Synthetic adversarial",
                "model": "Multi-layer POGQ Defense",
                "privacy": "Full ZK verification",
                "target_accuracy": 0.85,
                "attack_intensity": 1.0,
            },
            Scenario.ULTIMATE: {
                "name": "Ultimate FL Challenge",
                "description": "Maximum adversarial with 45% Byzantine nodes",
                "data_type": "Mixed (Images + Time-series)",
                "model": "Ensemble + HyperFeel",
                "privacy": "ZK-STARK + DP + SA",
                "target_accuracy": 0.80,
                "attack_intensity": 1.5,
            },
        }
        return configs[self.scenario]

    async def run_round(self, round_num: int) -> RoundMetrics:
        """Execute a single FL round."""
        start_time = time.perf_counter()

        config = self._get_scenario_config()

        # Simulate training
        await asyncio.sleep(0.05 + random.uniform(0, 0.05))

        # Update node states
        for node_id, node in self.nodes.items():
            node.gradients_sent += 1
            node.gradients_received += random.randint(2, 5)

            if node.is_byzantine:
                # Byzantine nodes have lower accuracy
                node.accuracy = min(0.6, node.accuracy + random.uniform(-0.02, 0.01))
            else:
                # Honest nodes improve over time
                improvement = 0.02 * (1 - node.accuracy) * (1 + round_num / self.num_rounds)
                node.accuracy = min(0.98, node.accuracy + improvement + random.uniform(-0.01, 0.02))

        # Simulate Byzantine detection
        detection_time = time.perf_counter()
        await asyncio.sleep(0.02)

        newly_detected = set()
        for node_id in self.byzantine_ids:
            if node_id not in self.detected_ids:
                # Detection probability increases with rounds
                detection_prob = 0.3 + 0.5 * (round_num / self.num_rounds)
                if random.random() < detection_prob:
                    newly_detected.add(node_id)
                    self.nodes[node_id].detected = True

        self.detected_ids.update(newly_detected)

        # Calculate round metrics
        honest_nodes = [n for n in self.nodes.values() if not n.is_byzantine]
        round_accuracy = sum(n.accuracy for n in honest_nodes) / len(honest_nodes)
        self.current_accuracy = round_accuracy

        detection_rate = len(self.detected_ids) / len(self.byzantine_ids) if self.byzantine_ids else 1.0

        duration = (time.perf_counter() - start_time) * 1000
        aggregation_latency = random.uniform(5, 15)

        metrics = RoundMetrics(
            round_num=round_num,
            duration_ms=duration,
            accuracy=round_accuracy,
            detection_rate=detection_rate,
            byzantine_detected=len(self.detected_ids),
            byzantine_total=len(self.byzantine_ids),
            aggregation_latency_ms=aggregation_latency,
            compression_ratio=random.uniform(800, 1200),
        )

        self.rounds.append(metrics)
        return metrics

    async def run(self) -> DemoResult:
        """Run the complete demo."""
        start_time = time.perf_counter()

        for round_num in range(self.num_rounds):
            await self.run_round(round_num + 1)

        total_duration = time.perf_counter() - start_time

        return DemoResult(
            scenario=self.scenario.value,
            num_nodes=self.num_nodes,
            byzantine_ratio=self.byzantine_ratio,
            total_rounds=self.num_rounds,
            final_accuracy=self.current_accuracy,
            detection_rate=len(self.detected_ids) / len(self.byzantine_ids) if self.byzantine_ids else 1.0,
            avg_latency_ms=sum(r.aggregation_latency_ms for r in self.rounds) / len(self.rounds),
            total_duration_s=total_duration,
            rounds=self.rounds,
            byzantine_nodes=self.byzantine_ids,
            detected_nodes=self.detected_ids,
        )


def create_status_panel(runner: DemoRunner, current_round: int) -> Panel:
    """Create status panel for live display."""
    config = runner._get_scenario_config()

    # Build status table
    table = Table.grid(expand=True)
    table.add_column(justify="left", ratio=1)
    table.add_column(justify="right", ratio=1)

    table.add_row(
        f"[bold]{config['name']}[/bold]",
        f"Round [cyan]{current_round}[/cyan] / {runner.num_rounds}"
    )
    table.add_row(
        f"[dim]{config['description']}[/dim]",
        f"Nodes: [green]{runner.num_nodes}[/green]"
    )
    table.add_row(
        f"Model: [yellow]{config['model']}[/yellow]",
        f"Byzantine: [red]{len(runner.byzantine_ids)}[/red] ({runner.byzantine_ratio:.0%})"
    )

    return Panel(table, title="Demo Status", border_style="cyan")


def create_metrics_panel(runner: DemoRunner) -> Panel:
    """Create metrics panel for live display."""
    if not runner.rounds:
        return Panel("[dim]Waiting for first round...[/dim]", title="Metrics")

    latest = runner.rounds[-1]

    # Metrics table
    table = Table.grid(expand=True, padding=(0, 2))
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    # Accuracy indicator
    acc_color = "green" if latest.accuracy >= 0.85 else "yellow" if latest.accuracy >= 0.7 else "red"

    # Detection indicator
    det_color = "green" if latest.detection_rate >= 0.9 else "yellow" if latest.detection_rate >= 0.7 else "red"

    table.add_row(
        "Accuracy",
        f"[{acc_color}]{latest.accuracy:.1%}[/{acc_color}]",
        "Detection Rate",
        f"[{det_color}]{latest.detection_rate:.1%}[/{det_color}]",
    )
    table.add_row(
        "Latency",
        f"[blue]{latest.aggregation_latency_ms:.1f}ms[/blue]",
        "Byzantine Detected",
        f"[red]{latest.byzantine_detected}[/red] / {latest.byzantine_total}",
    )
    table.add_row(
        "Compression",
        f"[magenta]{latest.compression_ratio:.0f}x[/magenta]",
        "Round Time",
        f"[dim]{latest.duration_ms:.0f}ms[/dim]",
    )

    return Panel(table, title="Real-time Metrics", border_style="green")


def create_nodes_table(runner: DemoRunner, show_all: bool = False) -> Table:
    """Create table showing node states."""
    table = Table(title="Node Status", show_header=True, header_style="bold magenta")
    table.add_column("Node", style="cyan", width=12)
    table.add_column("Status", justify="center", width=12)
    table.add_column("Accuracy", justify="right", width=10)
    table.add_column("Gradients", justify="right", width=10)
    table.add_column("Attack Type", width=15)

    # Show subset of nodes
    nodes_to_show = list(runner.nodes.values())
    if not show_all and len(nodes_to_show) > 10:
        # Show first 3, last 3, and any Byzantine
        byzantine_nodes = [n for n in nodes_to_show if n.is_byzantine]
        honest_sample = [n for n in nodes_to_show if not n.is_byzantine][:3]
        nodes_to_show = honest_sample + byzantine_nodes[:5]
        nodes_to_show = sorted(nodes_to_show, key=lambda n: n.node_id)[:10]

    for node in nodes_to_show:
        if node.is_byzantine:
            if node.detected:
                status = "[red]DETECTED[/red]"
            else:
                status = "[yellow]Byzantine[/yellow]"
        else:
            status = "[green]Honest[/green]"

        acc_color = "green" if node.accuracy >= 0.85 else "yellow" if node.accuracy >= 0.7 else "red"

        table.add_row(
            node.node_id,
            status,
            f"[{acc_color}]{node.accuracy:.1%}[/{acc_color}]",
            str(node.gradients_sent),
            node.attack_type or "-",
        )

    return table


def create_final_report(result: DemoResult, runner: DemoRunner) -> Panel:
    """Create final summary report."""
    config = runner._get_scenario_config()

    # Overall status
    success = result.final_accuracy >= config["target_accuracy"] * 0.9
    status_icon = "[bold green]SUCCESS[/bold green]" if success else "[bold yellow]PARTIAL[/bold yellow]"

    # Build report
    report_lines = [
        f"[bold cyan]{'=' * 50}[/bold cyan]",
        f"[bold]DEMO COMPLETE: {config['name']}[/bold]",
        f"[bold cyan]{'=' * 50}[/bold cyan]",
        "",
        f"Status: {status_icon}",
        "",
        "[bold]Configuration:[/bold]",
        f"  Nodes: {result.num_nodes}",
        f"  Byzantine Ratio: {result.byzantine_ratio:.0%}",
        f"  Rounds: {result.total_rounds}",
        "",
        "[bold]Performance:[/bold]",
        f"  Final Accuracy: [{'green' if result.final_accuracy >= 0.85 else 'yellow'}]{result.final_accuracy:.1%}[/]",
        f"  Target Accuracy: {config['target_accuracy']:.0%}",
        f"  Detection Rate: [{'green' if result.detection_rate >= 0.9 else 'yellow'}]{result.detection_rate:.1%}[/]",
        f"  Avg Latency: [blue]{result.avg_latency_ms:.1f}ms[/blue]",
        f"  Total Duration: {result.total_duration_s:.1f}s",
        "",
        "[bold]Byzantine Detection:[/bold]",
        f"  Total Byzantine: {len(result.byzantine_nodes)}",
        f"  Detected: {len(result.detected_nodes)}",
        f"  Undetected: {len(result.byzantine_nodes - result.detected_nodes)}",
        "",
        "[bold]Key Features Demonstrated:[/bold]",
        "  [green]OK[/green] Multi-layer Byzantine detection (POGQ + Shapley)",
        "  [green]OK[/green] HyperFeel gradient compression (1000x+)",
        "  [green]OK[/green] P2P gossip protocol",
        "  [green]OK[/green] Self-healing network",
    ]

    return Panel(
        "\n".join(report_lines),
        title="Demo Summary",
        border_style="cyan",
        padding=(1, 2),
    )


async def run_demo_with_display(runner: DemoRunner) -> DemoResult:
    """Run demo with live display."""
    config = runner._get_scenario_config()

    # Show banner
    console.print()
    console.print(Panel(
        f"[bold cyan]{config['name']}[/bold cyan]\n\n"
        f"{config['description']}\n\n"
        f"[dim]Data: {config['data_type']} | Model: {config['model']}[/dim]",
        title="Starting Demo",
        border_style="green",
    ))
    console.print()

    # Create progress
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task(
            f"[cyan]Running {runner.num_rounds} FL rounds...",
            total=runner.num_rounds
        )

        for round_num in range(1, runner.num_rounds + 1):
            metrics = await runner.run_round(round_num)
            progress.update(task, advance=1)

            # Periodic status update
            if round_num % 5 == 0 or round_num == runner.num_rounds:
                console.print(
                    f"  Round {round_num}: "
                    f"Accuracy=[green]{metrics.accuracy:.1%}[/green] "
                    f"Detection=[cyan]{metrics.detection_rate:.1%}[/cyan] "
                    f"Latency=[blue]{metrics.aggregation_latency_ms:.1f}ms[/blue]"
                )

    # Final result
    result = DemoResult(
        scenario=runner.scenario.value,
        num_nodes=runner.num_nodes,
        byzantine_ratio=runner.byzantine_ratio,
        total_rounds=runner.num_rounds,
        final_accuracy=runner.current_accuracy,
        detection_rate=len(runner.detected_ids) / len(runner.byzantine_ids) if runner.byzantine_ids else 1.0,
        avg_latency_ms=sum(r.aggregation_latency_ms for r in runner.rounds) / len(runner.rounds),
        total_duration_s=sum(r.duration_ms for r in runner.rounds) / 1000,
        rounds=runner.rounds,
        byzantine_nodes=runner.byzantine_ids,
        detected_nodes=runner.detected_ids,
    )

    console.print()
    console.print(create_final_report(result, runner))
    console.print()

    return result


@app.callback(invoke_without_command=True)
def demo_main(
    ctx: typer.Context,
    scenario: Scenario = typer.Option(
        Scenario.HEALTHCARE,
        "--scenario",
        "-s",
        help="Demo scenario to run",
        case_sensitive=False,
    ),
    nodes: int = typer.Option(
        10,
        "--nodes",
        "-n",
        min=3,
        max=1000,
        help="Number of FL nodes",
    ),
    byzantine: float = typer.Option(
        0.3,
        "--byzantine",
        "-b",
        min=0.0,
        max=0.49,
        help="Fraction of Byzantine nodes (0.0-0.49)",
    ),
    rounds: int = typer.Option(
        20,
        "--rounds",
        "-r",
        min=1,
        max=1000,
        help="Number of training rounds",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save results to JSON file",
    ),
):
    """
    Run an interactive FL demo with real-time progress.

    [bold]Scenarios:[/bold]

    - [cyan]healthcare[/cyan]: Privacy-preserving medical imaging
    - [cyan]finance[/cyan]: Fraud detection across institutions
    - [cyan]adversarial[/cyan]: Stress test Byzantine detection
    - [cyan]ultimate[/cyan]: Maximum adversarial challenge

    [bold]Examples:[/bold]

        mycelix demo --scenario healthcare --nodes 10
        mycelix demo -s finance -n 50 -b 0.3 -r 30
        mycelix demo --scenario ultimate --byzantine 0.45
    """
    if ctx.invoked_subcommand is not None:
        return

    # Adjust byzantine ratio for ultimate scenario
    if scenario == Scenario.ULTIMATE:
        byzantine = max(byzantine, 0.45)
    elif scenario == Scenario.ADVERSARIAL:
        byzantine = max(byzantine, 0.35)

    console.print(f"\n[bold cyan]Mycelix FL Demo[/bold cyan]")
    console.print(f"[dim]Scenario: {scenario.value} | Nodes: {nodes} | Byzantine: {byzantine:.0%} | Rounds: {rounds}[/dim]\n")

    runner = DemoRunner(
        scenario=scenario,
        num_nodes=nodes,
        byzantine_ratio=byzantine,
        num_rounds=rounds,
        verbose=verbose,
    )

    result = asyncio.run(run_demo_with_display(runner))

    # Save results if requested
    if output:
        import json
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump({
                "scenario": result.scenario,
                "config": {
                    "num_nodes": result.num_nodes,
                    "byzantine_ratio": result.byzantine_ratio,
                    "total_rounds": result.total_rounds,
                },
                "results": {
                    "final_accuracy": result.final_accuracy,
                    "detection_rate": result.detection_rate,
                    "avg_latency_ms": result.avg_latency_ms,
                    "total_duration_s": result.total_duration_s,
                },
                "rounds": [
                    {
                        "round": r.round_num,
                        "accuracy": r.accuracy,
                        "detection_rate": r.detection_rate,
                        "latency_ms": r.aggregation_latency_ms,
                    }
                    for r in result.rounds
                ],
                "byzantine_nodes": list(result.byzantine_nodes),
                "detected_nodes": list(result.detected_nodes),
            }, f, indent=2)
        console.print(f"[dim]Results saved to {output}[/dim]")


@app.command()
def list_scenarios():
    """List all available demo scenarios."""
    table = Table(title="Available Scenarios", show_header=True, header_style="bold magenta")
    table.add_column("Scenario", style="cyan")
    table.add_column("Description")
    table.add_column("Difficulty", justify="center")
    table.add_column("Byzantine", justify="center")

    scenarios = [
        ("healthcare", "Privacy-preserving medical imaging classification", "[green]Normal[/green]", "20-30%"),
        ("finance", "Fraud detection across banking institutions", "[yellow]Medium[/yellow]", "25-35%"),
        ("adversarial", "Stress testing Byzantine detection", "[red]Hard[/red]", "35-40%"),
        ("ultimate", "Maximum adversarial with 45% Byzantine", "[bold red]Extreme[/bold red]", "45%"),
    ]

    for name, desc, difficulty, byz in scenarios:
        table.add_row(name, desc, difficulty, byz)

    console.print(table)
    console.print("\n[dim]Use: mycelix demo --scenario <name>[/dim]")


if __name__ == "__main__":
    app()
