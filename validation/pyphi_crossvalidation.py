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
    python -m pip install -r validation/requirements-pyphi.txt

Author: Tristan Stoltz & Claude
Date: 2026-01-04
Version: 1.0.0
"""

import argparse
import hashlib
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
    logger.warning("Install with: python -m pip install -r validation/requirements-pyphi.txt")


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
    reference_model_digest: Optional[str] = None
    symthaea_model_digest: Optional[str] = None
    reference_formalism: str = "IIT_3_0"


@dataclass
class ValidationReport:
    """Complete cross-validation report."""
    results: List[ValidationResult] = field(default_factory=list)
    correlation: Optional[float] = None
    spearman_correlation: Optional[float] = None
    mean_absolute_error: Optional[float] = None
    root_mean_square_error: Optional[float] = None
    max_absolute_error: Optional[float] = None
    agreement_within_10_percent: Optional[float] = None
    pyphi_version: Optional[str] = None
    reference_formalism: str = "IIT_3_0"
    shared_model_verified: bool = False
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

    def build_reference_model(
        self, topology: TopologySpec
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Build a canonical graph-derived TPM and digest the exact reference model.

        This model is a deliberately simple noisy OR-neighbor network. It is a
        reproducible PyPhi reference fixture, not yet the model used by Symthaea's
        causal-cut Rust benchmark; comparisons must remain blocked until both share
        the same canonical model.
        """
        if topology.n_nodes < 1:
            raise ValueError("topology must have at least one node")
        if topology.n_nodes > 8:
            raise ValueError("reference TPM construction is capped at 8 nodes")
        for source, target in topology.edges:
            if not (0 <= source < topology.n_nodes and 0 <= target < topology.n_nodes):
                raise ValueError(
                    f"edge ({source}, {target}) is outside the {topology.n_nodes}-node topology"
                )

        adjacency = self.edges_to_adjacency(topology.n_nodes, topology.edges)
        # The fallback rule for an isolated node is self-memory. Reflect that
        # dependency in the connectivity matrix so the graph and TPM agree.
        for node in range(topology.n_nodes):
            if not np.any(adjacency[node]):
                adjacency[node, node] = 1
        n_states = 2 ** topology.n_nodes
        tpm = np.zeros((n_states, topology.n_nodes), dtype=np.float64)
        for state_index in range(n_states):
            state_bits = [(state_index >> i) & 1 for i in range(topology.n_nodes)]
            for node in range(topology.n_nodes):
                neighbors = [j for j in range(topology.n_nodes) if adjacency[node, j] == 1]
                output = any(state_bits[j] for j in neighbors)
                probability = 1.0 if output else 0.0
                tpm[state_index, node] = np.clip(probability * 0.98 + 0.01, 0.01, 0.99)

        # PyPhi's integer state rows are little-endian by node index: node 0 varies
        # fastest. Include both adjacency and TPM in the digest to prevent topology
        # names alone from serving as provenance.
        digest = hashlib.sha256()
        digest.update(b"symthaea-pyphi-reference-model/v1\0")
        digest.update(repr(adjacency.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(adjacency, dtype=np.uint8).tobytes())
        digest.update(repr(tpm.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(tpm, dtype="<f8").tobytes())
        digest.update(b"state=all-ones\0formalism=IIT_3_0\0")
        return adjacency, tpm, digest.hexdigest()

    def compute_pyphi_phi(self, topology: TopologySpec) -> Tuple[float, float]:
        """Compute the graph-derived PyPhi reference under IIT 3.0.

        The Symthaea/PyPhi comparison remains scientifically blocked until the
        Rust side consumes the same explicit transition model.
        """
        if not PYPHI_AVAILABLE:
            raise RuntimeError("PyPhi not installed")

        from importlib.metadata import version as package_version
        import time

        installed_version = package_version("pyphi")
        if installed_version != "2.0.0rc4":
            raise RuntimeError(
                f"PyPhi version {installed_version} is not qualified for this harness; "
                "install the pinned validation dependency pyphi==2.0.0rc4."
            )

        adjacency, tpm, _model_digest = self.build_reference_model(topology)
        start = time.perf_counter()
        substrate = pyphi.Substrate(
            tpm,
            cm=adjacency,
            node_labels=tuple(f"node_{index}" for index in range(topology.n_nodes)),
        )
        state = tuple([1] * topology.n_nodes)
        analysis = pyphi.analyze(substrate, state, formalism="IIT_3_0")
        phi = float(analysis.phi)
        if not np.isfinite(phi):
            raise RuntimeError(f"PyPhi returned a non-finite value for {topology.name}")
        return phi, time.perf_counter() - start

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
            pyphi_phi=None,
        )
        try:
            _adjacency, _tpm, digest = self.build_reference_model(topology)
            result.reference_model_digest = digest
        except Exception as error:
            result.error_message = f"Reference model error: {error}"
            return result

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
        if PYPHI_AVAILABLE:
            try:
                from importlib.metadata import version as package_version
                report.pyphi_version = package_version("pyphi")
            except Exception as error:
                logger.warning("Could not record PyPhi package version: %s", error)

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
        pearson = float(np.corrcoef(hdc_values, pyphi_values)[0, 1])
        if np.isfinite(pearson):
            report.correlation = pearson

        def average_ranks(values: np.ndarray) -> np.ndarray:
            order = np.argsort(values, kind="mergesort")
            ranks = np.empty(len(values), dtype=float)
            start = 0
            while start < len(values):
                end = start + 1
                while end < len(values) and values[order[end]] == values[order[start]]:
                    end += 1
                # Average of the 1-based rank interval [start + 1, end].
                ranks[order[start:end]] = (start + 1 + end) / 2.0
                start = end
            return ranks

        spearman = float(np.corrcoef(average_ranks(hdc_values), average_ranks(pyphi_values))[0, 1])
        if np.isfinite(spearman):
            report.spearman_correlation = spearman

        errors = np.abs(hdc_values - pyphi_values)
        report.mean_absolute_error = float(np.mean(errors))
        report.root_mean_square_error = float(np.sqrt(np.mean(np.square(errors))))
        report.max_absolute_error = float(np.max(errors))
        relative_errors = errors / np.maximum(np.abs(pyphi_values), 0.001)
        report.agreement_within_10_percent = float(np.mean(relative_errors < 0.10) * 100.0)

    def is_qualified(self, report: ValidationReport) -> bool:
        """Single source of truth for both report status and CLI exit status."""
        paired_count = sum(
            r.hdc_phi is not None and r.pyphi_phi is not None
            and np.isfinite(r.hdc_phi) and np.isfinite(r.pyphi_phi)
            for r in report.results
        )
        return (
            paired_count >= 3
            and paired_count == len(report.results)
            and report.n_validated == paired_count
            and report.n_failed == 0
            and report.correlation is not None
            and np.isfinite(report.correlation)
            and report.correlation > 0.90
        )

    def generate_report(self, report: ValidationReport, output_path: Path):
        """Generate a report that never presents missing results as numeric zeros."""
        paired_count = sum(
            r.hdc_phi is not None and r.pyphi_phi is not None
            and np.isfinite(r.hdc_phi) and np.isfinite(r.pyphi_phi)
            for r in report.results
        )
        qualified = self.is_qualified(report)
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
            "- Pinned PyPhi reference version: 2.0.0rc4 (pre-release)",
            "- Declared reference formalism: IIT_3_0",
            "- Reference state: all nodes ON; PyPhi bit order is least-significant node first.",
            "- Reference TPM: noisy OR-neighbor update with per-node next-state probability 0.01 or 0.99.",
            f"- Correlation: {fmt(report.correlation, '.4f')}",
            f"- Mean absolute error: {fmt(report.mean_absolute_error, '.4f')}",
            f"- Maximum absolute error: {fmt(report.max_absolute_error, '.4f')}",
            f"- Agreement within 10%: {fmt(report.agreement_within_10_percent, '.1f')}%",
            "",
            "## Detailed results",
            "",
            "| Topology | Nodes | Reference model SHA-256 | Symthaea score | PyPhi score | Absolute error | Status |",
            "|---|---:|---|---:|---:|---:|---|",
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
            model_digest = result.reference_model_digest or "N/A"
            md.append(
                f"| {result.topology_name} | {result.n_nodes} | {model_digest} | "
                f"{hdc} | {reference} | {error} | {case_status} |"
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
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(md), encoding="utf-8")
        logger.info("Report saved to %s", output_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the fail-closed Symthaea/PyPhi validation scaffold."
    )
    parser.add_argument("--max-nodes", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pyphi_validation_results.md"),
        help="Markdown report destination.",
    )
    args = parser.parse_args()

    validator = PyPhiValidator()
    report = validator.run_validation(max_nodes=args.max_nodes)
    validator.generate_report(report, args.output)
    paired_count = sum(
        result.hdc_phi is not None and result.pyphi_phi is not None
        for result in report.results
    )
    qualified = validator.is_qualified(report)
    status = "PASS" if qualified else "NOT QUALIFIED"
    print(f"Status: {status} (paired measurements={paired_count}, failures={report.n_failed})")
    print(f"Report: {args.output}")
    return 0 if qualified else 2


if __name__ == "__main__":
    raise SystemExit(main())

