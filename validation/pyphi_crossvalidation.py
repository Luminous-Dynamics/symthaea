#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Legacy PyPhi cross-validation scaffold — NOT QUALIFIED.

The previous implementation mixed topology-generated inputs with an HDC adapter
that was not present, and could substitute hard-coded scores when it failed.
That fallback has been removed. This module now fails closed until both sides
compute named quantities over the same explicit causal model/state with a
pinned formalism and provenance record.

Use validation/generate_pyphi_reference_fixtures.py to run the independent
upstream PyPhi literature-reference smoke test. It does not validate Symthaea's
HDC proxy and must not be described as consciousness validation.
"""

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
    hdc_std: Optional[float] = None
    pyphi_computation_time: float = 0.0
    hdc_computation_time: float = 0.0
    error_message: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete cross-validation report."""
    results: List[ValidationResult] = field(default_factory=list)
    correlation: float = 0.0
    mean_absolute_error: float = 0.0
    max_absolute_error: float = 0.0
    agreement_within_10_percent: float = 0.0
    n_validated: int = 0
    n_failed: int = 0
    qualification_status: str = "NOT_QUALIFIED"


class PyPhiValidator:
    """Legacy scaffold; it cannot qualify an HDC-to-IIT comparison."""

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
        """Fail closed: the legacy topology-to-TPM path is not a qualified reference adapter."""
        raise RuntimeError(
            "NOT QUALIFIED: this legacy PyPhi 1.x-style adapter does not bind an "
            "explicit shared causal-model digest and formalism; use the pinned "
            "literature reference runner instead"
        )
    def compute_hdc_phi(self, topology: TopologySpec, n_samples: int = 10) -> Tuple[float, float, float]:
        """Fail closed until a maintained same-model Rust adapter exists."""
        raise RuntimeError("NOT QUALIFIED: no maintained Rust adapter computes this estimator from the same explicit causal model and state")
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
            pyphi_phi=None,
            hdc_std=None
        )

        # HDC calculation (always possible)
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

            if result.hdc_phi is not None and result.pyphi_phi is not None and result.error_message is None:
                report.n_validated += 1
            else:
                report.n_failed += 1

        # Compute aggregate metrics
        self._compute_report_metrics(report)

        return report

    def _compute_report_metrics(self, report: ValidationReport):
        """Compute descriptive metrics only; correlation cannot qualify this scaffold."""
        paired = [(r.hdc_phi, r.pyphi_phi) for r in report.results if r.hdc_phi is not None and r.pyphi_phi is not None and r.error_message is None]
        report.qualification_status = "NOT_QUALIFIED"
        if len(paired) < 5:
            logger.warning("NOT QUALIFIED: fewer than five valid paired cases")
            return
        hdc_values = np.asarray([p[0] for p in paired], dtype=float)
        pyphi_values = np.asarray([p[1] for p in paired], dtype=float)
        if not (np.isfinite(hdc_values).all() and np.isfinite(pyphi_values).all()):
            logger.warning("NOT QUALIFIED: non-finite metric")
            return
        if np.ptp(hdc_values) == 0 or np.ptp(pyphi_values) == 0:
            logger.warning("NOT QUALIFIED: constant-score correlation is undefined")
            return
        correlation = float(np.corrcoef(hdc_values, pyphi_values)[0, 1])
        if not np.isfinite(correlation):
            logger.warning("NOT QUALIFIED: non-finite correlation")
            return
        report.correlation = correlation
        errors = np.abs(hdc_values - pyphi_values)
        report.mean_absolute_error = float(np.mean(errors))
        report.max_absolute_error = float(np.max(errors))
        relative_errors = errors / np.maximum(np.abs(pyphi_values), 0.001)
        report.agreement_within_10_percent = float(np.mean(relative_errors < 0.10) * 100)
    def generate_report(self, report: ValidationReport, output_path: Path):
        """
        Generate markdown validation report.

        Args:
            report: Validation report
            output_path: Path to save markdown file
        """
        md = []
        md.append("# PyPhi Cross-Validation Report")
        md.append(f"\n**Generated**: {__import__('datetime').datetime.now().isoformat()}")
        md.append(f"**Qualification status**: {report.qualification_status}")
        md.append("")

        md.append("## Summary Statistics")
        md.append("")
        md.append(f"- **Correlation (r)**: {report.correlation:.4f}")
        md.append(f"- **Mean Absolute Error**: {report.mean_absolute_error:.4f}")
        md.append(f"- **Max Absolute Error**: {report.max_absolute_error:.4f}")
        md.append(f"- **Agreement within 10%**: {report.agreement_within_10_percent:.1f}%")
        md.append(f"- **Validated topologies**: {report.n_validated}")
        md.append(f"- **Failed topologies**: {report.n_failed}")
        md.append("")

        # Validation target
        md.append("## Validation Target")
        md.append("")
        md.append("NOT QUALIFIED: matching causal-model/state digests, pinned formalism/version, and independent sample gates are not implemented.")
        md.append("")

        md.append("## Detailed Results")
        md.append("")
        md.append("| Topology | Nodes | HDC-Φ | PyPhi-Φ | Error | Status |")
        md.append("|----------|-------|-------|---------|-------|--------|")

        for r in report.results:
            hdc_str = f"{r.hdc_phi:.4f}" if r.hdc_phi is not None else "N/A"
            pyphi_str = f"{r.pyphi_phi:.4f}" if r.pyphi_phi is not None else "N/A"
            if r.hdc_phi is not None and r.pyphi_phi is not None:
                error_str = f"{abs(r.hdc_phi - r.pyphi_phi):.4f}"
                status = "DESCRIPTIVE ONLY"
            else:
                error_str = "N/A"
                status = "NOT MEASURED"

            md.append(f"| {r.topology_name} | {r.n_nodes} | {hdc_str} | "
                     f"{pyphi_str} | {error_str} | {status} |")

        md.append("")
        md.append("## Notes")
        md.append("")
        md.append("- N/A means no measurement was produced; it is never replaced with a numeric estimate.")
        md.append("- The legacy HDC adapter and legacy PyPhi adapter are deliberately disabled pending a shared causal-model contract.")
        md.append("- This report is NOT QUALIFIED and makes no IIT-validation or consciousness claim.")
        md.append("")

        md.append("## Methodology")
        md.append("")
        md.append("1. This legacy scaffold does not currently perform a paired reference comparison.")
        md.append("2. The independent pinned PyPhi literature-reference smoke test is in validation/generate_pyphi_reference_fixtures.py.")
        md.append("3. Implement a maintained Rust adapter and shared TPM/state digests before re-enabling HDC-to-reference metrics.")
        md.append("4. Select the formalism explicitly and record the PyPhi artifact/version in every generated result.")
        md.append("")

        output_path.write_text("\n".join(md))
        logger.info(f"Report saved to {output_path}")


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
    print(f"Correlation (r): {report.correlation:.4f}")
    print(f"Mean Absolute Error: {report.mean_absolute_error:.4f}")
    print(f"Validated: {report.n_validated}, Failed: {report.n_failed}")

    print(f"\nQualification status: {report.qualification_status}")
    print("No IIT validation claim is supported by this scaffold.")

    validator.generate_report(report, Path(args.output))
    print(f"\n📄 Report saved to: {args.output}")


if __name__ == "__main__":
    main()
