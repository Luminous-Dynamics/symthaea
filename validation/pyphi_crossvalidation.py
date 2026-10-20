#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
PyPhi Cross-Validation Framework
================================

BLOCKED LEGACY VALIDATION SCAFFOLD.

The original implementation could substitute hard-coded topology estimates when
Symthaea's subprocess failed. Its abandoned Rust sketch used spectral connectivity,
which is not IIT Phi. Do not use old reports from this scaffold as evidence of IIT
agreement. Validation remains blocked until a maintained Rust adapter and matching
causal model are available.

Requirements:
    pip install pyphi numpy scipy matplotlib seaborn

Author: Tristan Stoltz & Claude
Date: 2026-01-04
Version: 1.0.0
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
    """Result of cross-validation for a single topology."""
    topology_name: str
    n_nodes: int
    hdc_phi: Optional[float]
    pyphi_phi: Optional[float]
    hdc_std: float = 0.0
    pyphi_computation_time: float = 0.0
    hdc_computation_time: float = 0.0
    error_message: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete cross-validation report."""
    results: List[ValidationResult] = field(default_factory=list)
    correlation: Optional[float] = None
    mean_absolute_error: Optional[float] = None
    max_absolute_error: Optional[float] = None
    agreement_within_10_percent: Optional[float] = None
    n_validated: int = 0
    n_failed: int = 0


class PyPhiValidator:
    """Validates HDC-Φ against exact PyPhi Φ calculation."""

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
        Compute the PyPhi reference value with an explicitly selected IIT 3.0 formalism.

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
            if not hasattr(pyphi, "analyze"):
                raise RuntimeError("PyPhi 2.0 analyze API required to pin the IIT formalism explicitly")
            analysis = pyphi.analyze(network, state, formalism="IIT_3_0")
            phi = float(analysis.phi)

            elapsed = time.time() - start
            return phi, elapsed

        except Exception as e:
            logger.error(f"PyPhi computation failed for {topology.name}: {e}")
            raise

    def compute_hdc_phi(self, topology: TopologySpec, n_samples: int = 10) -> Tuple[float, float, float]:
        """Fail closed until a valid Rust causal-model adapter is implemented.

        The previous adapter called a missing example and substituted hard-coded
        scores. Its abandoned source sketch also used spectral connectivity, not
        IIT Phi. Never convert an unavailable or failed computation into a score.
        """
        raise RuntimeError(
            "IIT cross-validation blocked: no maintained Rust adapter currently "
            "emits a score from the same explicit causal model as the PyPhi run. "
            "Hard-coded topology estimates and spectral connectivity are invalid substitutes."
        )

    def validate_topology(self, topology: TopologySpec) -> ValidationResult:
        """
        Validate HDC-Φ against PyPhi for a single topology.

        Args:
            topology: Topology to validate

        Returns:
            ValidationResult with both calculations
        """
        logger.info(f"Validating {topology.name} (n={topology.n_nodes})...")

        result = ValidationResult(
            topology_name=topology.name,
            n_nodes=topology.n_nodes,
            hdc_phi=None,
            pyphi_phi=None
        )

        # HDC adapter intentionally fails closed until it uses the same causal model.
        try:
            hdc_phi, hdc_std, hdc_time = self.compute_hdc_phi(topology)
            result.hdc_phi = hdc_phi
            result.hdc_std = hdc_std
            result.hdc_computation_time = hdc_time
        except Exception as e:
            result.error_message = f"HDC error: {e}"
            return result

        # PyPhi calculation (may be intractable for large n)
        if PYPHI_AVAILABLE and topology.n_nodes <= 8:
            try:
                pyphi_phi, pyphi_time = self.compute_pyphi_phi(topology)
                result.pyphi_phi = pyphi_phi
                result.pyphi_computation_time = pyphi_time
            except Exception as e:
                result.error_message = f"PyPhi error: {e}"
                logger.warning(f"PyPhi failed for {topology.name}: {e}")

        return result

    def run_validation(self, max_nodes: int = 8) -> ValidationReport:
        """
        Run complete cross-validation.

        Args:
            max_nodes: Maximum number of nodes to test (PyPhi tractability limit)

        Returns:
            Complete validation report
        """
        topologies = self.create_test_topologies()
        topologies = [t for t in topologies if t.n_nodes <= max_nodes]

        report = ValidationReport()

        for topology in topologies:
            result = self.validate_topology(topology)
            report.results.append(result)

            if result.hdc_phi is not None and result.pyphi_phi is not None:
                report.n_validated += 1
            else:
                report.n_failed += 1

        # Compute aggregate metrics
        self._compute_report_metrics(report)

        return report

    def _compute_report_metrics(self, report: ValidationReport):
        """Compute metrics only from finite, paired numeric measurements."""
        paired = [
            (r.hdc_phi, r.pyphi_phi)
            for r in report.results
            if r.hdc_phi is not None
            and r.pyphi_phi is not None
            and np.isfinite(r.hdc_phi)
            and np.isfinite(r.pyphi_phi)
        ]
        if len(paired) < 3:
            logger.warning("Not enough valid paired results (need at least 3)")
            return

        hdc_values = np.array([pair[0] for pair in paired], dtype=float)
        pyphi_values = np.array([pair[1] for pair in paired], dtype=float)
        correlation = float(np.corrcoef(hdc_values, pyphi_values)[0, 1])
        if np.isfinite(correlation):
            report.correlation = correlation
        errors = np.abs(hdc_values - pyphi_values)
        report.mean_absolute_error = float(np.mean(errors))
        report.max_absolute_error = float(np.max(errors))
        relative_errors = errors / np.maximum(np.abs(pyphi_values), 0.001)
        report.agreement_within_10_percent = float(np.mean(relative_errors < 0.10) * 100.0)

    def generate_report(self, report: ValidationReport, output_path: Path):
        """Generate a report that never presents missing results as numeric zeros."""
        paired_count = sum(
            r.hdc_phi is not None and r.pyphi_phi is not None
            and np.isfinite(r.hdc_phi) and np.isfinite(r.pyphi_phi)
            for r in report.results
        )
        qualified = (
            paired_count >= 3
            and report.n_failed == 0
            and report.correlation is not None
            and np.isfinite(report.correlation)
            and report.correlation > 0.90
        )
        fmt = lambda value, spec: "N/A" if value is None or not np.isfinite(value) else format(value, spec)
        status = "PASS" if qualified else "NOT QUALIFIED"
        md = [
            "# PyPhi Cross-Validation Report",
            "",
            "Generated: " + __import__("datetime").datetime.now().isoformat(),
            "Status: " + status,
            "",
            "## Summary statistics",
            "",
            f"- Paired measurements: {paired_count}",
            f"- Failed or unavailable cases: {report.n_failed}",
            f"- Correlation: {fmt(report.correlation, '.4f')}",
            f"- Mean absolute error: {fmt(report.mean_absolute_error, '.4f')}",
            f"- Maximum absolute error: {fmt(report.max_absolute_error, '.4f')}",
            f"- Agreement within 10%: {fmt(report.agreement_within_10_percent, '.1f')}%",
            "",
            "## Detailed results",
            "",
            "| Topology | Nodes | Symthaea score | PyPhi score | Absolute error | Status |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for result in report.results:
            hdc = fmt(result.hdc_phi, ".6f")
            reference = fmt(result.pyphi_phi, ".6f")
            if result.hdc_phi is not None and result.pyphi_phi is not None:
                error = fmt(abs(result.hdc_phi - result.pyphi_phi), ".6f")
                case_status = "PAIRED"
            else:
                error = "N/A"
                case_status = "FAILED / NOT RUN"
            md.append(
                f"| {result.topology_name} | {result.n_nodes} | {hdc} | "
                f"{reference} | {error} | {case_status} |"
            )
        md.extend([
            "",
            "## Qualification notes",
            "",
            "- The Symthaea adapter is intentionally blocked; numeric fallback estimates are prohibited.",
            "- The PyPhi reference call explicitly selects IIT_3_0; exact PyPhi version and causal-model digest must be recorded by a maintained runner.",
            "- A pass requires at least three finite pairs, no failed cases, and correlation greater than 0.90; correlation alone does not establish scientific equivalence.",
            "- Spectral connectivity is not IIT Phi.",
            "- N/A means a valid paired measurement was not produced.",
            "",
        ])
        output_path.write_text("\n".join(md), encoding="utf-8")
        logger.info("Report saved to %s", output_path)

