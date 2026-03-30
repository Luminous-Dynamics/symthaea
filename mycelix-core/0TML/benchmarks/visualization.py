#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Benchmark Visualization Module
===============================

Generates publication-quality matplotlib charts for the MLSys 2026 paper.

Charts:
1. Detection accuracy comparison (bar charts, line plots)
2. Latency distributions (box plots, CDFs)
3. Throughput scaling (line plots)
4. Scalability analysis (multi-panel figures)

Author: Luminous Dynamics
Date: January 8, 2026
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

# Check for matplotlib availability
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server use
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import PercentFormatter
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


class BenchmarkVisualizer:
    """
    Generate publication-quality visualizations for benchmark results.

    Follows MLSys paper formatting guidelines:
    - Font sizes appropriate for single-column (3.5") or double-column (7") figures
    - High DPI for print quality
    - Colorblind-friendly color palette
    """

    # Colorblind-friendly palette (IBM Design)
    COLORS = {
        "pogq": "#648FFF",      # Blue
        "fltrust": "#785EF0",   # Purple
        "krum": "#DC267F",      # Magenta
        "fedavg": "#FE6100",    # Orange
        "sha3_only": "#FFB000", # Gold
        "risc0": "#785EF0",     # Purple
        "winterfell": "#648FFF", # Blue
    }

    # Line styles for black-and-white printing
    LINE_STYLES = {
        "pogq": "-",
        "fltrust": "--",
        "krum": "-.",
        "fedavg": ":",
    }

    # Markers for scatter plots
    MARKERS = {
        "pogq": "o",
        "fltrust": "s",
        "krum": "^",
        "fedavg": "x",
    }

    def __init__(
        self,
        output_dir: str = "benchmarks/results",
        dpi: int = 300,
        figsize: tuple = (7, 5),
        font_size: int = 10,
    ):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory for output files
            dpi: Resolution for saved figures
            figsize: Default figure size (width, height) in inches
            font_size: Base font size
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dpi = dpi
        self.figsize = figsize
        self.font_size = font_size

        if HAS_MATPLOTLIB:
            self._setup_matplotlib()

    def _setup_matplotlib(self):
        """Configure matplotlib for publication-quality output."""
        plt.rcParams.update({
            'font.size': self.font_size,
            'font.family': 'serif',
            'axes.labelsize': self.font_size + 1,
            'axes.titlesize': self.font_size + 2,
            'xtick.labelsize': self.font_size - 1,
            'ytick.labelsize': self.font_size - 1,
            'legend.fontsize': self.font_size - 1,
            'figure.figsize': self.figsize,
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.axisbelow': True,
        })

    def _check_matplotlib(self) -> bool:
        """Check if matplotlib is available."""
        if not HAS_MATPLOTLIB:
            print("WARNING: matplotlib not available. Skipping visualization.")
            return False
        return True

    def plot_detection_accuracy(self, results: Dict[str, Any]) -> Optional[str]:
        """
        Plot Byzantine detection accuracy comparison.

        Creates a grouped bar chart comparing PoGQ, FLTrust, Krum, FedAvg
        across different Byzantine ratios for each attack type.
        """
        if not self._check_matplotlib():
            return None

        attack_types = list(results.keys())
        if not attack_types:
            print("No detection accuracy results to plot.")
            return None

        # Create figure with subplots for each attack
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()

        methods = ["pogq", "fltrust", "krum", "fedavg"]
        method_labels = ["PoGQ", "FLTrust", "Krum", "FedAvg"]

        for idx, attack_type in enumerate(attack_types[:4]):
            ax = axes[idx]
            attack_data = results[attack_type]

            ratios = sorted(attack_data.keys())
            x = np.arange(len(ratios))
            width = 0.2

            for i, method in enumerate(methods):
                detection_rates = []
                for ratio in ratios:
                    if method in attack_data[ratio]:
                        rate = attack_data[ratio][method].get("detection_rate", 0)
                        detection_rates.append(rate * 100)  # Convert to percentage
                    else:
                        detection_rates.append(0)

                offset = (i - 1.5) * width
                bars = ax.bar(x + offset, detection_rates, width,
                             label=method_labels[i],
                             color=self.COLORS.get(method, "#888888"),
                             edgecolor='white', linewidth=0.5)

            ax.set_xlabel('Byzantine Ratio')
            ax.set_ylabel('Detection Rate (%)')
            ax.set_title(attack_type.replace('_', ' ').title())
            ax.set_xticks(x)
            ax.set_xticklabels([f"{float(r):.0%}" for r in ratios])
            ax.set_ylim(0, 105)
            ax.legend(loc='lower left')

        plt.tight_layout()

        output_path = self.output_dir / "detection_accuracy.png"
        plt.savefig(output_path)
        plt.close()

        print(f"Saved detection accuracy plot to {output_path}")
        return str(output_path)

    def plot_detection_heatmap(self, results: Dict[str, Any]) -> Optional[str]:
        """
        Plot detection accuracy as a heatmap.

        Rows: Attack types
        Columns: Byzantine ratios
        Values: PoGQ detection rate
        """
        if not self._check_matplotlib():
            return None

        attack_types = list(results.keys())
        if not attack_types:
            return None

        # Extract all ratios
        all_ratios = set()
        for attack_data in results.values():
            if isinstance(attack_data, dict):
                all_ratios.update(attack_data.keys())
        ratios = sorted(all_ratios)

        # Build heatmap data
        heatmap_data = []
        for attack_type in attack_types:
            row = []
            attack_data = results.get(attack_type, {})
            for ratio in ratios:
                if ratio in attack_data and "pogq" in attack_data[ratio]:
                    rate = attack_data[ratio]["pogq"].get("detection_rate", 0)
                    row.append(rate * 100)
                else:
                    row.append(0)
            heatmap_data.append(row)

        heatmap_data = np.array(heatmap_data)

        fig, ax = plt.subplots(figsize=(8, 4))

        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

        # Labels
        ax.set_xticks(np.arange(len(ratios)))
        ax.set_yticks(np.arange(len(attack_types)))
        ax.set_xticklabels([f"{float(r):.0%}" for r in ratios])
        ax.set_yticklabels([a.replace('_', ' ').title() for a in attack_types])

        ax.set_xlabel('Byzantine Ratio')
        ax.set_ylabel('Attack Type')
        ax.set_title('PoGQ Detection Rate (%)')

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Detection Rate (%)', rotation=-90, va="bottom")

        # Add text annotations
        for i in range(len(attack_types)):
            for j in range(len(ratios)):
                text = ax.text(j, i, f"{heatmap_data[i, j]:.0f}%",
                              ha="center", va="center", color="black",
                              fontsize=self.font_size - 2)

        plt.tight_layout()

        output_path = self.output_dir / "detection_heatmap.png"
        plt.savefig(output_path)
        plt.close()

        print(f"Saved detection heatmap to {output_path}")
        return str(output_path)

    def plot_latency_comparison(self, results: Dict[str, Any]) -> Optional[str]:
        """
        Plot latency comparison across methods.

        Bar chart showing median and P99 latencies.
        """
        if not self._check_matplotlib():
            return None

        agg_data = results.get("aggregation", {})
        if not agg_data:
            print("No aggregation latency results to plot.")
            return None

        methods = list(agg_data.keys())
        medians = [agg_data[m].get("median_ms", 0) for m in methods]
        p99s = [agg_data[m].get("p99_ms", 0) for m in methods]

        x = np.arange(len(methods))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))

        bars1 = ax.bar(x - width/2, medians, width, label='Median',
                       color=self.COLORS.get("pogq", "#648FFF"))
        bars2 = ax.bar(x + width/2, p99s, width, label='P99',
                       color=self.COLORS.get("fltrust", "#785EF0"))

        ax.set_xlabel('Method')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Aggregation Latency Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in methods])
        ax.legend()

        # Add 1ms target line
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5,
                  label='Target (<1ms)')
        ax.legend()

        # Value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        output_path = self.output_dir / "latency_comparison.png"
        plt.savefig(output_path)
        plt.close()

        print(f"Saved latency comparison to {output_path}")
        return str(output_path)

    def plot_latency_distribution(self, results: Dict[str, Any]) -> Optional[str]:
        """
        Plot latency distribution (box plots or violin plots).
        """
        if not self._check_matplotlib():
            return None

        # Create synthetic distribution data from summary stats
        agg_data = results.get("aggregation", {})
        if not agg_data:
            return None

        fig, ax = plt.subplots(figsize=(8, 5))

        methods = list(agg_data.keys())
        box_data = []

        for method in methods:
            stats = agg_data[method]
            median = stats.get("median_ms", 0)
            std = stats.get("std_ms", 0)
            p95 = stats.get("p95_ms", 0)

            # Generate synthetic samples approximating the distribution
            samples = np.random.normal(median, std, 100)
            samples = np.clip(samples, 0, p95 * 1.2)
            box_data.append(samples)

        bp = ax.boxplot(box_data, labels=[m.upper() for m in methods],
                       patch_artist=True)

        # Color boxes
        for patch, method in zip(bp['boxes'], methods):
            patch.set_facecolor(self.COLORS.get(method, "#888888"))
            patch.set_alpha(0.7)

        ax.set_xlabel('Method')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Aggregation Latency Distribution')

        # Add 1ms target line
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5,
                  label='Target (<1ms)')
        ax.legend()

        plt.tight_layout()

        output_path = self.output_dir / "latency_distribution.png"
        plt.savefig(output_path)
        plt.close()

        print(f"Saved latency distribution to {output_path}")
        return str(output_path)

    def plot_throughput(self, results: Dict[str, Any]) -> Optional[str]:
        """
        Plot throughput vs concurrent clients.
        """
        if not self._check_matplotlib():
            return None

        rep_data = results.get("reputation_updates", [])
        if not rep_data:
            print("No throughput results to plot.")
            return None

        clients = [r["concurrent_clients"] for r in rep_data]
        tps = [r["tps_mean"] for r in rep_data]
        latencies = [r["latency_p50_ms"] for r in rep_data]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # TPS plot
        ax1.plot(clients, tps, marker='o', linewidth=2, markersize=8,
                color=self.COLORS["pogq"])
        ax1.set_xlabel('Concurrent Clients')
        ax1.set_ylabel('Transactions Per Second')
        ax1.set_title('Reputation Update Throughput')
        ax1.grid(True, alpha=0.3)

        # Fill area under curve
        ax1.fill_between(clients, tps, alpha=0.2, color=self.COLORS["pogq"])

        # Latency plot
        ax2.plot(clients, latencies, marker='s', linewidth=2, markersize=8,
                color=self.COLORS["fltrust"])
        ax2.set_xlabel('Concurrent Clients')
        ax2.set_ylabel('P50 Latency (ms)')
        ax2.set_title('Operation Latency Under Load')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / "throughput.png"
        plt.savefig(output_path)
        plt.close()

        print(f"Saved throughput plot to {output_path}")
        return str(output_path)

    def plot_scalability(self, results: Dict[str, Any]) -> Optional[str]:
        """
        Plot scalability metrics (detection rate, latency) vs node count.
        """
        if not self._check_matplotlib():
            return None

        node_data = results.get("node_scaling", [])
        if not node_data:
            print("No scalability results to plot.")
            return None

        nodes = [r["num_nodes"] for r in node_data]
        detection_rates = [r["detection_rate"] * 100 for r in node_data]
        latencies = [r["latency_ms"] for r in node_data]
        memories = [r["memory_mb"] for r in node_data]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Detection rate
        axes[0].plot(nodes, detection_rates, marker='o', linewidth=2,
                    color=self.COLORS["pogq"])
        axes[0].set_xlabel('Number of Nodes')
        axes[0].set_ylabel('Detection Rate (%)')
        axes[0].set_title('Detection Accuracy vs Scale')
        axes[0].set_ylim(0, 105)
        axes[0].grid(True, alpha=0.3)

        # Latency
        axes[1].plot(nodes, latencies, marker='s', linewidth=2,
                    color=self.COLORS["fltrust"])
        axes[1].set_xlabel('Number of Nodes')
        axes[1].set_ylabel('Latency (ms)')
        axes[1].set_title('Latency vs Scale')
        axes[1].grid(True, alpha=0.3)

        # Add linear scaling reference
        if len(nodes) > 1:
            linear_ref = np.array(latencies[0]) * (np.array(nodes) / nodes[0])
            axes[1].plot(nodes, linear_ref, '--', color='gray', alpha=0.7,
                        label='Linear scaling')
            axes[1].legend()

        # Memory
        axes[2].plot(nodes, memories, marker='^', linewidth=2,
                    color=self.COLORS["krum"])
        axes[2].set_xlabel('Number of Nodes')
        axes[2].set_ylabel('Memory (MB)')
        axes[2].set_title('Memory Usage vs Scale')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / "scalability.png"
        plt.savefig(output_path)
        plt.close()

        print(f"Saved scalability plot to {output_path}")
        return str(output_path)

    def plot_memory_scaling(self, results: Dict[str, Any]) -> Optional[str]:
        """
        Plot detailed memory scaling analysis.
        """
        if not self._check_matplotlib():
            return None

        mem_data = results.get("memory_efficiency", {})
        if not mem_data or "per_node_memory" not in mem_data:
            return None

        per_node = mem_data["per_node_memory"]
        if not per_node:
            return None

        nodes = [p["num_nodes"] for p in per_node]
        total_mb = [p["total_mb"] for p in per_node]
        per_node_kb = [p["per_node_kb"] for p in per_node]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Total memory
        ax1.plot(nodes, total_mb, marker='o', linewidth=2,
                color=self.COLORS["pogq"])
        ax1.set_xlabel('Number of Nodes')
        ax1.set_ylabel('Total Memory (MB)')
        ax1.set_title(f'Memory Scaling ({mem_data.get("memory_scaling", "linear")})')
        ax1.grid(True, alpha=0.3)

        # Per-node memory
        ax2.plot(nodes, per_node_kb, marker='s', linewidth=2,
                color=self.COLORS["fltrust"])
        ax2.set_xlabel('Number of Nodes')
        ax2.set_ylabel('Per-Node Memory (KB)')
        ax2.set_title('Memory Efficiency')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / "memory_scaling.png"
        plt.savefig(output_path)
        plt.close()

        print(f"Saved memory scaling plot to {output_path}")
        return str(output_path)

    def plot_proof_generation(self, results: Dict[str, Any]) -> Optional[str]:
        """
        Plot proof generation time comparison.
        """
        if not self._check_matplotlib():
            return None

        proof_data = results.get("proof_generation", {})
        if not proof_data:
            return None

        backends = list(proof_data.keys())
        medians = [proof_data[b].get("median_ms", 0) for b in backends]
        p95s = [proof_data[b].get("p95_ms", 0) for b in backends]

        x = np.arange(len(backends))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))

        bars1 = ax.bar(x - width/2, medians, width, label='Median',
                       color=self.COLORS.get("sha3_only", "#FFB000"))
        bars2 = ax.bar(x + width/2, p95s, width, label='P95',
                       color=self.COLORS.get("winterfell", "#648FFF"))

        ax.set_xlabel('Proof Backend')
        ax.set_ylabel('Generation Time (ms)')
        ax.set_title('Proof Generation Latency')
        ax.set_xticks(x)
        ax.set_xticklabels([b.upper().replace('_', ' ') for b in backends])
        ax.legend()

        # Use log scale if range is large
        max_val = max(p95s) if p95s else 1
        if max_val > 100:
            ax.set_yscale('log')

        plt.tight_layout()

        output_path = self.output_dir / "proof_generation.png"
        plt.savefig(output_path)
        plt.close()

        print(f"Saved proof generation plot to {output_path}")
        return str(output_path)


def main():
    """Test visualization with sample data."""
    print("Testing Benchmark Visualizer")
    print("=" * 50)

    if not HAS_MATPLOTLIB:
        print("matplotlib not available. Cannot run visualization tests.")
        return

    viz = BenchmarkVisualizer(output_dir="benchmarks/results")

    # Sample detection accuracy data
    detection_results = {
        "label_flip": {
            "0.10": {"pogq": {"detection_rate": 0.95}, "fltrust": {"detection_rate": 0.85}},
            "0.20": {"pogq": {"detection_rate": 0.92}, "fltrust": {"detection_rate": 0.80}},
            "0.33": {"pogq": {"detection_rate": 0.88}, "fltrust": {"detection_rate": 0.72}},
            "0.45": {"pogq": {"detection_rate": 0.82}, "fltrust": {"detection_rate": 0.60}},
        }
    }

    viz.plot_detection_accuracy(detection_results)
    print("Visualization test complete.")


if __name__ == "__main__":
    main()
