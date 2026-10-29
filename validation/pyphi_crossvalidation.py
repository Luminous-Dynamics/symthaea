#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Exploratory PyPhi comparison scaffold — NOT a qualified IIT validation.

This script currently builds a topology-derived HDC representation metric and a
separate OR-neighbour transition-probability matrix for PyPhi. Those are not
demonstrably the same causal model, and this script does not pin an IIT
formalism/version. Therefore any paired numbers are descriptive only and must
never be reported as validation, calibration, or evidence that the HDC metric is
IIT Phi.

Measurement failures fail closed: there are no hard-coded fallback scores.
Qualification remains NOT_QUALIFIED until a shared causal-model contract,
explicit PyPhi formalism/version, and independent validation criteria exist.

Requirements:
    pip install pyphi numpy

Author: Luminous Dynamics
"""

import subprocess
import json
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import PyPhi (optional for systems without it)
try:
    import pyphi
    PYPHI_AVAILABLE = True
except ImportError:
    PYPHI_AVAILABLE = False
    logger.warning("PyPhi not installed. Exact Φ validation unavailable.")
    logger.warning("Install with: pip install pyphi")


@dataclass
class TopologySpec:
    """Specification for a network topology."""
    name: str
    n_nodes: int
    edges: List[Tuple[int, int]]
    description: str = ""


@dataclass
class ValidationResult:
    """Outcome for a single topology; absent measurements remain absent."""
    topology_name: str
    n_nodes: int
    hdc_phi: Optional[float] = None
    pyphi_phi: Optional[float] = None
    hdc_std: Optional[float] = None
    pyphi_computation_time: Optional[float] = None
    hdc_computation_time: Optional[float] = None
    error_message: Optional[str] = None
    status: str = "inconclusive"


@dataclass
class ValidationReport:
    """Descriptive result ledger with an explicit qualification ceiling."""
    results: List[ValidationResult] = field(default_factory=list)
    correlation: Optional[float] = None
    mean_absolute_error: Optional[float] = None
    max_absolute_error: Optional[float] = None
    agreement_within_10_percent: Optional[float] = None
    n_paired: int = 0
    n_failed: int = 0
    n_inconclusive: int = 0
    qualification_status: str = "NOT_QUALIFIED"
    qualification_reasons: List[str] = field(default_factory=lambda: [
        "The HDC path and PyPhi path do not currently establish a shared causal transition model and state.",
        "The PyPhi IIT formalism and package version are not explicitly pinned by this script.",
        "The Rust adapter contract does not require a matching model digest and state identifier.",
    ])


class PyPhiValidator:
    """Collect descriptive measurements; this scaffold is not qualified to validate IIT Φ."""

    def __init__(self, symthaea_binary: str = "cargo"):
        """
        Initialize validator.

        Args:
            symthaea_binary: Path to cargo or compiled Symthaea binary
        """
        self.symthaea_binary = symthaea_binary
        self.results: List[ValidationResult] = []

    def create_test_topologies(self) -> List[TopologySpec]:
        """
        Create standard test topologies for validation.

        Returns topologies with n ≤ 8 nodes where exact Φ is tractable.
        """
        topologies = [
            # 2-node topologies
            TopologySpec(
                name="line_2",
                n_nodes=2,
                edges=[(0, 1)],
                description="Simple 2-node line (K₂)"
            ),

            # 3-node topologies
            TopologySpec(
                name="line_3",
                n_nodes=3,
                edges=[(0, 1), (1, 2)],
                description="3-node line"
            ),
            TopologySpec(
                name="ring_3",
                n_nodes=3,
                edges=[(0, 1), (1, 2), (2, 0)],
                description="3-node ring (K₃)"
            ),

            # 4-node topologies
            TopologySpec(
                name="line_4",
                n_nodes=4,
                edges=[(0, 1), (1, 2), (2, 3)],
                description="4-node line"
            ),
            TopologySpec(
                name="ring_4",
                n_nodes=4,
                edges=[(0, 1), (1, 2), (2, 3), (3, 0)],
                description="4-node ring"
            ),
            TopologySpec(
                name="star_4",
                n_nodes=4,
                edges=[(0, 1), (0, 2), (0, 3)],
                description="4-node star"
            ),
            TopologySpec(
                name="square_hypercube",
                n_nodes=4,
                edges=[(0, 1), (1, 2), (2, 3), (3, 0)],
                description="2D hypercube (square)"
            ),

            # 5-node topologies
            TopologySpec(
                name="ring_5",
                n_nodes=5,
                edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
                description="5-node ring"
            ),
            TopologySpec(
                name="star_5",
                n_nodes=5,
                edges=[(0, 1), (0, 2), (0, 3), (0, 4)],
                description="5-node star"
            ),

            # 6-node topologies
            TopologySpec(
                name="ring_6",
                n_nodes=6,
                edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
                description="6-node ring"
            ),
            TopologySpec(
                name="torus_2x3",
                n_nodes=6,
                edges=[
                    (0, 1), (1, 2), (2, 0),  # Row 1 ring
                    (3, 4), (4, 5), (5, 3),  # Row 2 ring
                    (0, 3), (1, 4), (2, 5),  # Vertical connections
                ],
                description="2×3 torus"
            ),

            # 7-node topologies
            TopologySpec(
                name="binary_tree_7",
                n_nodes=7,
                edges=[
                    (0, 1), (0, 2),          # Root to level 1
                    (1, 3), (1, 4),          # Left subtree
                    (2, 5), (2, 6),          # Right subtree
                ],
                description="Complete binary tree (3 levels)"
            ),

            # 8-node topologies
            TopologySpec(
                name="ring_8",
                n_nodes=8,
                edges=[
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (4, 5), (5, 6), (6, 7), (7, 0)
                ],
                description="8-node ring"
            ),
            TopologySpec(
                name="star_8",
                n_nodes=8,
                edges=[
                    (0, 1), (0, 2), (0, 3), (0, 4),
                    (0, 5), (0, 6), (0, 7)
                ],
                description="8-node star"
            ),
            TopologySpec(
                name="cube_3d",
                n_nodes=8,
                edges=[
                    # Bottom face
                    (0, 1), (1, 2), (2, 3), (3, 0),
                    # Top face
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    # Vertical edges
                    (0, 4), (1, 5), (2, 6), (3, 7)
                ],
                description="3D hypercube (cube)"
            ),
            TopologySpec(
                name="torus_3x3",
                n_nodes=9,
                edges=[
                    # Row rings
                    (0, 1), (1, 2), (2, 0),
                    (3, 4), (4, 5), (5, 3),
                    (6, 7), (7, 8), (8, 6),
                    # Column rings
                    (0, 3), (3, 6), (6, 0),
                    (1, 4), (4, 7), (7, 1),
                    (2, 5), (5, 8), (8, 2),
                ],
                description="3×3 torus (if tractable)"
            ),
        ]

        return topologies

    def edges_to_adjacency(self, n_nodes: int, edges: List[Tuple[int, int]]) -> np.ndarray:
        """Convert edge list to adjacency matrix."""
        adj = np.zeros((n_nodes, n_nodes), dtype=int)
        for i, j in edges:
            adj[i, j] = 1
            adj[j, i] = 1  # Undirected
        return adj

    def compute_pyphi_phi(self, topology: TopologySpec) -> Tuple[float, float]:
        """
        Compute exact Φ using PyPhi.

        Args:
            topology: Topology specification

        Returns:
            (phi_value, computation_time_seconds)
        """
        if not PYPHI_AVAILABLE:
            raise RuntimeError("PyPhi not installed")

        import time

        # Create adjacency matrix
        adj = self.edges_to_adjacency(topology.n_nodes, topology.edges)

        # PyPhi requires a TPM (transition probability matrix)
        # For a simple deterministic network, we use identity-like TPM
        # This is a simplification - real validation would use actual network dynamics

        # Create simple network: each node XORs its inputs
        # TPM shape: (2^n_nodes, n_nodes) - probability of each node being ON
        # given each possible past state

        n = topology.n_nodes
        n_states = 2 ** n

        # Simple TPM: each node takes OR of neighbors
        tpm = np.zeros((n_states, n))
        for state in range(n_states):
            state_bits = [(state >> i) & 1 for i in range(n)]
            for node in range(n):
                # Node is ON if any neighbor was ON
                neighbors = [j for j in range(n) if adj[node, j] == 1]
                if neighbors:
                    neighbor_or = any(state_bits[j] for j in neighbors)
                    tpm[state, node] = 1.0 if neighbor_or else 0.0
                else:
                    tpm[state, node] = state_bits[node]  # Isolated nodes stay same

        # Add small noise to avoid deterministic issues
        tpm = np.clip(tpm * 0.98 + 0.01, 0.01, 0.99)

        start = time.time()

        try:
            # Create PyPhi network
            network = pyphi.Network(tpm, connectivity_matrix=adj)

            # Get subsystem for all nodes in all-on state
            state = tuple([1] * n)
            subsystem = pyphi.Subsystem(network, state)

            # Compute Φ (this is the expensive operation)
            sia = pyphi.compute.sia(subsystem)
            phi = sia.phi

            elapsed = time.time() - start
            return phi, elapsed

        except Exception as e:
            logger.error(f"PyPhi computation failed for {topology.name}: {e}")
            raise

    def compute_hdc_phi(self, topology: TopologySpec, n_samples: int = 10) -> Tuple[float, float, float]:
        """Call the real Rust adapter and reject every invalid or failed result.

        The adapter must emit one JSON object with finite numeric mean and std
        fields on stdout. Logs belong on stderr. There is intentionally no
        estimated-value fallback: a failed calculation is a failed case.
        """
        import time

        start = time.monotonic()
        request = json.dumps({
            "n_nodes": topology.n_nodes,
            "edges": topology.edges,
            "n_samples": n_samples,
        }, separators=(",", ":"), sort_keys=True)
        try:
            completed = subprocess.run(
                [
                    self.symthaea_binary, "run", "--example",
                    "phi_for_topology", "--release", "--", request,
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=Path(__file__).resolve().parent.parent,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Rust HDC adapter timed out for {topology.name} after 60 seconds"
            ) from exc
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Rust HDC adapter executable not found: {self.symthaea_binary!r}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"Could not execute Rust HDC adapter: {exc}") from exc

        elapsed = time.monotonic() - start
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip().replace("\n", " ")[:400]
            raise RuntimeError(
                f"Rust HDC adapter exited {completed.returncode} for {topology.name}; "
                f"stderr={stderr!r}"
            )

        try:
            output = json.loads(completed.stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError(
                f"Rust HDC adapter returned invalid JSON for {topology.name}"
            ) from exc

        if not isinstance(output, dict):
            raise RuntimeError("Rust HDC adapter JSON must be an object")

        values: Dict[str, float] = {}
        for field_name in ("mean", "std"):
            value = output.get(field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RuntimeError(
                    f"Rust HDC adapter field {field_name!r} must be numeric"
                )
            numeric = float(value)
            if not np.isfinite(numeric):
                raise RuntimeError(
                    f"Rust HDC adapter field {field_name!r} must be finite"
                )
            values[field_name] = numeric

        if values["std"] < 0.0:
            raise RuntimeError("Rust HDC adapter field 'std' must be non-negative")

        return values["mean"], values["std"], elapsed

    def validate_topology(self, topology: TopologySpec) -> ValidationResult:
        """Collect both measurements without converting failures into scores."""
        logger.info("Measuring %s (n=%s)...", topology.name, topology.n_nodes)
        result = ValidationResult(topology_name=topology.name, n_nodes=topology.n_nodes)

        try:
            hdc_phi, hdc_std, hdc_time = self.compute_hdc_phi(topology)
            result.hdc_phi = hdc_phi
            result.hdc_std = hdc_std
            result.hdc_computation_time = hdc_time
        except Exception as exc:
            result.status = "failed"
            result.error_message = f"HDC adapter error: {type(exc).__name__}: {exc}"
            logger.warning("%s", result.error_message)
            return result

        if not PYPHI_AVAILABLE:
            result.status = "inconclusive"
            result.error_message = "PyPhi is unavailable; no reference measurement was produced."
            return result

        if topology.n_nodes > 8:
            result.status = "inconclusive"
            result.error_message = "PyPhi tractability gate: topology exceeds 8 nodes."
            return result

        try:
            pyphi_phi, pyphi_time = self.compute_pyphi_phi(topology)
            if isinstance(pyphi_phi, bool) or not np.isfinite(float(pyphi_phi)):
                raise ValueError("PyPhi returned a non-finite or non-numeric Φ value")
            result.pyphi_phi = float(pyphi_phi)
            result.pyphi_computation_time = float(pyphi_time)
            result.status = "paired_not_qualified"
        except Exception as exc:
            result.status = "failed"
            result.error_message = f"PyPhi reference error: {type(exc).__name__}: {exc}"
            logger.warning("%s", result.error_message)

        return result

    def run_validation(self, max_nodes: int = 8) -> ValidationReport:
        """Run bounded measurements and return a descriptive, non-qualified report."""
        if max_nodes < 1:
            raise ValueError("max_nodes must be at least 1")

        topologies = [
            topology for topology in self.create_test_topologies()
            if topology.n_nodes <= max_nodes
        ]
        report = ValidationReport()

        for topology in topologies:
            result = self.validate_topology(topology)
            report.results.append(result)

        report.n_paired = sum(
            result.hdc_phi is not None
            and result.pyphi_phi is not None
            and np.isfinite(result.hdc_phi)
            and np.isfinite(result.pyphi_phi)
            for result in report.results
        )
        report.n_failed = sum(result.status == "failed" for result in report.results)
        report.n_inconclusive = sum(
            result.status == "inconclusive" for result in report.results
        )
        self._compute_report_metrics(report)
        return report

    def _compute_report_metrics(self, report: ValidationReport) -> None:
        """Compute descriptive summaries only; never promote them to validation."""
        paired = [
            (float(result.hdc_phi), float(result.pyphi_phi))
            for result in report.results
            if result.hdc_phi is not None
            and result.pyphi_phi is not None
            and np.isfinite(result.hdc_phi)
            and np.isfinite(result.pyphi_phi)
        ]
        report.n_paired = len(paired)
        if not paired:
            report.qualification_reasons.append(
                "No finite paired measurements were available."
            )
            return

        hdc_values = np.asarray([pair[0] for pair in paired], dtype=float)
        pyphi_values = np.asarray([pair[1] for pair in paired], dtype=float)
        errors = np.abs(hdc_values - pyphi_values)
        report.mean_absolute_error = float(np.mean(errors))
        report.max_absolute_error = float(np.max(errors))
        relative_errors = errors / np.maximum(np.abs(pyphi_values), 1e-12)
        report.agreement_within_10_percent = float(
            np.mean(relative_errors < 0.10) * 100.0
        )

        if len(paired) < 3:
            report.qualification_reasons.append(
                "Fewer than three paired cases: correlation is not reported."
            )
            return
        if np.ptp(hdc_values) == 0.0 or np.ptp(pyphi_values) == 0.0:
            report.qualification_reasons.append(
                "Correlation is undefined because at least one metric is constant."
            )
            return

        correlation = float(np.corrcoef(hdc_values, pyphi_values)[0, 1])
        if not np.isfinite(correlation):
            report.qualification_reasons.append(
                "Correlation was non-finite and has been suppressed."
            )
            return
        report.correlation = correlation

    def generate_report(self, report: ValidationReport, output_path: Path) -> None:
        """Write a report that cannot label an unqualified comparison as PASS."""
        from datetime import datetime

        def fmt(value: Optional[float], precision: int = 4) -> str:
            return "N/A" if value is None else f"{value:.{precision}f}"

        lines = [
            "# PyPhi Comparison Report",
            "",
            f"**Generated:** {datetime.now().astimezone().isoformat()}",
            f"**Qualification status:** {report.qualification_status}",
            "",
            "> This is a descriptive scaffold, not a qualified IIT validation. "
            "A high correlation does not repair a mismatch in causal model, state, "
            "or formalism.",
            "",
            "## Descriptive statistics",
            "",
            f"- Finite paired measurements: {report.n_paired}",
            f"- Failed cases: {report.n_failed}",
            f"- Inconclusive cases: {report.n_inconclusive}",
            f"- Pearson correlation (descriptive only): {fmt(report.correlation)}",
            f"- Mean absolute difference (descriptive only): {fmt(report.mean_absolute_error)}",
            f"- Maximum absolute difference (descriptive only): {fmt(report.max_absolute_error)}",
            f"- Agreement within 10% (descriptive only): {fmt(report.agreement_within_10_percent, 1)}%",
            "",
            "## Qualification blockers",
            "",
        ]
        lines.extend(f"- {reason}" for reason in dict.fromkeys(report.qualification_reasons))
        lines.extend([
            "",
            "## Case ledger",
            "",
            "| Topology | Nodes | HDC metric | PyPhi value | Absolute difference | Case status | Error |",
            "|---|---:|---:|---:|---:|---|---|",
        ])

        for result in report.results:
            delta = (
                abs(result.hdc_phi - result.pyphi_phi)
                if result.hdc_phi is not None and result.pyphi_phi is not None
                else None
            )
            error = (result.error_message or "").replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {result.topology_name} | {result.n_nodes} | "
                f"{fmt(result.hdc_phi)} | {fmt(result.pyphi_phi)} | "
                f"{fmt(delta)} | {result.status} | {error} |"
            )

        lines.extend([
            "",
            "## Method limits",
            "",
            "- The HDC-side quantity is computed from node representations; the PyPhi-side TPM is constructed from an OR-of-neighbours rule. This script does not prove these encode the same causal model and state.",
            "- The PyPhi formalism and package version are not explicitly pinned here.",
            "- The result is therefore **NOT QUALIFIED**, regardless of correlation or agreement.",
            "- A failed/missing reference computation stays missing; it is never replaced by a topology-based estimate.",
            "",
        ])
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Descriptive report saved to %s", output_path)


def main():
    """Run cross-validation and generate report."""
    import argparse

    parser = argparse.ArgumentParser(description="PyPhi Cross-Validation")
    parser.add_argument("--max-nodes", type=int, default=8,
                       help="Maximum nodes to test (PyPhi limit)")
    parser.add_argument("--output", type=str, default="PYPHI_VALIDATION_REPORT.md",
                       help="Output report path")
    args = parser.parse_args()

    validator = PyPhiValidator()

    print("\n" + "=" * 60)
    print("🧠 Symthaea PyPhi Cross-Validation Framework")
    print("=" * 60)
    print(f"\nPyPhi available: {PYPHI_AVAILABLE}")
    print(f"Max nodes: {args.max_nodes}")
    print("")

    report = validator.run_validation(max_nodes=args.max_nodes)

    print("\n📊 Results Summary")
    print("-" * 40)
    correlation_text = "N/A" if report.correlation is None else f"{report.correlation:.4f}"
    mae_text = "N/A" if report.mean_absolute_error is None else f"{report.mean_absolute_error:.4f}"
    print(f"Descriptive correlation (not validation): {correlation_text}")
    print(f"Mean Absolute Difference: {mae_text}")
    print(f"Paired: {report.n_paired}, Failed: {report.n_failed}, Inconclusive: {report.n_inconclusive}")
    print("\n⛔ NOT QUALIFIED: shared causal-model and formalism gates are not satisfied.")

    validator.generate_report(report, Path(args.output))
    print(f"\n📄 Report saved to: {args.output}")


if __name__ == "__main__":
    main()
