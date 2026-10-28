#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Fail-closed paired-model comparison for Symthaea's IIT 3.0 candidate score.

This script sends one explicit state-by-node TPM and state to both:
- Symthaea's Rust candidate MIP score adapter; and
- the pinned PyPhi 2.0.0rc4 IIT_3_0 reference profile.

The Rust value is deliberately named a candidate score, not assumed to be
formal IIT Phi. The report may qualify the candidate only for a declared
comparison profile and acceptance gates. It never establishes that IIT is
correct or that Symthaea is conscious.

Failure policy:
- no hard-coded score fallbacks;
- nonzero exits, timeouts, malformed output, wrong model identity, missing
  PyPhi, unexpected PyPhi versions, and non-finite values are failures;
- unpaired / insufficient / constant-score data cannot yield a correlation;
- acceptance thresholds must be supplied explicitly before a run can qualify.

PyPhi is an offline research dependency; it is not a production dependency.
Expected reference artifact: PyPhi 2.0.0rc4, formalism IIT_3_0.
"""


import hashlib
import json
import logging
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

EXPECTED_PYPHI_VERSION = "2.0.0rc4"
FORMALISM_PROFILE = "IIT_3_0"
RUST_CANDIDATE_METRIC_ID = "symthaea_iit3_full_system_mechanism_mip_candidate_v1"
MODEL_SCHEMA_VERSION = "symthaea-iit3-tpm-input-v1"
NOISE_LEVEL = 0.02
MAX_EXACT_NODES = 8

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import pyphi
    PYPHI_AVAILABLE = True
    PYPHI_VERSION = getattr(pyphi, "__version__", None)
except ImportError:
    pyphi = None
    PYPHI_AVAILABLE = False
    PYPHI_VERSION = None
    logger.warning("PyPhi is not installed; comparison will remain NOT_QUALIFIED.")


@dataclass
class TopologySpec:
    """Specification for a network topology."""
    name: str
    n_nodes: int
    edges: List[Tuple[int, int]]
    description: str = ""


@dataclass
class ValidationResult:
    """Paired measurement result for one explicit TPM and current state."""
    topology_name: str
    n_nodes: int
    model_sha256: str
    candidate_phi: Optional[float] = None
    pyphi_phi_s: Optional[float] = None
    pyphi_computation_time: float = 0.0
    candidate_computation_time: float = 0.0
    candidate_metric_id: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class ValidationReport:
    """Fail-closed comparison report; status is NOT_QUALIFIED by default."""
    results: List[ValidationResult] = field(default_factory=list)
    correlation: Optional[float] = None
    mean_absolute_error: Optional[float] = None
    max_absolute_error: Optional[float] = None
    agreement_within_10_percent: Optional[float] = None
    n_validated: int = 0
    n_failed: int = 0
    qualification_status: str = "NOT_QUALIFIED"
    qualification_reasons: List[str] = field(default_factory=list)
    pyphi_version: Optional[str] = None
    formalism_profile: str = FORMALISM_PROFILE
    min_valid_pairs: Optional[int] = None
    min_pearson: Optional[float] = None


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

    def build_model(self, topology: TopologySpec) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Build one explicit state-by-node TPM shared by both implementations."""
        if topology.n_nodes < 2 or topology.n_nodes > MAX_EXACT_NODES:
            raise ValueError(
                f"Exact comparison supports 2..={MAX_EXACT_NODES} nodes; got {topology.n_nodes}"
            )

        adjacency = self.edges_to_adjacency(topology.n_nodes, topology.edges)
        n_states = 1 << topology.n_nodes
        tpm = np.zeros((n_states, topology.n_nodes), dtype=np.float64)

        # Fixed, explicit dynamics: each node is ON when any neighbor is ON,
        # with independent per-node flip noise. Little-endian state indexing
        # matches the Rust TransitionProbabilityMatrix contract and PyPhi's TPM.
        for state_index in range(n_states):
            bits = [(state_index >> i) & 1 for i in range(topology.n_nodes)]
            for node in range(topology.n_nodes):
                neighbors = [j for j in range(topology.n_nodes) if adjacency[node, j] == 1]
                base = float(any(bits[j] for j in neighbors)) if neighbors else float(bits[node])
                tpm[state_index, node] = (
                    base * (1.0 - NOISE_LEVEL) + (1.0 - base) * NOISE_LEVEL
                )

        state = np.ones(topology.n_nodes, dtype=np.uint8)
        model = {
            "formalism_profile": FORMALISM_PROFILE,
            "n_nodes": topology.n_nodes,
            "noise_level": NOISE_LEVEL,
            "state": state.astype(int).tolist(),
            "topology_name": topology.name,
            "tpm": tpm.tolist(),
        }
        canonical_model = json.dumps(
            model, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        model_sha256 = hashlib.sha256(canonical_model).hexdigest()
        payload = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "model_sha256": model_sha256,
            **model,
        }
        return tpm, state, payload

    def compute_pyphi_phi(
        self, topology: TopologySpec, tpm: np.ndarray, state: np.ndarray
    ) -> Tuple[float, float]:
        """Compute PyPhi system integrated information under explicit IIT_3_0."""
        if not PYPHI_AVAILABLE or pyphi is None:
            raise RuntimeError("PyPhi is not installed; reference result is unavailable")
        if PYPHI_VERSION != EXPECTED_PYPHI_VERSION:
            raise RuntimeError(
                f"PyPhi version mismatch: expected {EXPECTED_PYPHI_VERSION}, got {PYPHI_VERSION!r}"
            )

        start = __import__("time").monotonic()
        adjacency = self.edges_to_adjacency(topology.n_nodes, topology.edges)
        substrate = pyphi.Substrate(
            tpm,
            cm=adjacency,
            node_labels=tuple(f"n{i}" for i in range(topology.n_nodes)),
        )
        analysis = pyphi.analyze(
            substrate,
            tuple(int(bit) for bit in state.tolist()),
            formalism=FORMALISM_PROFILE,
        )
        value = float(analysis.phi)  # PyPhi's system integrated information φ_s.
        elapsed = __import__("time").monotonic() - start
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError(f"PyPhi returned an invalid φ_s value: {value!r}")
        return value, elapsed

    def compute_candidate_phi(
        self, payload: dict
    ) -> Tuple[float, float, str]:
        """Call the maintained Rust TPM adapter; failures never create scores."""
        start = __import__("time").monotonic()
        command = [
            self.symthaea_binary,
            "run",
            "--quiet",
            "--release",
            "--example",
            "iit3_tpm_phi_adapter",
            "--",
        ]
        try:
            result = subprocess.run(
                command,
                input=json.dumps(payload, separators=(",", ":"), allow_nan=False),
                capture_output=True,
                text=True,
                timeout=120,
                cwd=Path(__file__).resolve().parent.parent,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Rust TPM adapter timed out; no score produced") from exc
        except FileNotFoundError as exc:
            raise RuntimeError(f"Rust adapter executable not found: {self.symthaea_binary}") from exc
        elapsed = __import__("time").monotonic() - start

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()[-1500:]
            raise RuntimeError(
                f"Rust TPM adapter exited {result.returncode}; no score produced. stderr={stderr!r}"
            )

        try:
            output = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError("Rust TPM adapter returned malformed JSON; no score produced") from exc
        if not isinstance(output, dict):
            raise RuntimeError("Rust TPM adapter output must be a JSON object")
        required = {
            "schema_version",
            "metric_id",
            "formalism_profile",
            "model_sha256",
            "n_nodes",
            "state",
            "value",
        }
        missing = sorted(required - output.keys())
        if missing:
            raise RuntimeError(f"Rust TPM adapter output missing fields {missing}; no score produced")
        if output["schema_version"] != "symthaea-iit3-tpm-output-v1":
            raise RuntimeError(f"Unsupported Rust output schema: {output['schema_version']!r}")
        if output["metric_id"] != RUST_CANDIDATE_METRIC_ID:
            raise RuntimeError(f"Unexpected Rust metric identity: {output['metric_id']!r}")
        for key in ("formalism_profile", "model_sha256", "n_nodes", "state"):
            if output[key] != payload[key]:
                raise RuntimeError(
                    f"Rust output {key} does not match the submitted model; no score produced"
                )
        value = output["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError("Rust candidate score must be numeric")
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError(f"Rust adapter returned invalid candidate score {value!r}")
        return value, elapsed, output["metric_id"]

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
            hdc_phi=0.0,
            pyphi_phi=None
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

            if result.pyphi_phi is not None:
                report.n_validated += 1
            elif result.error_message:
                report.n_failed += 1

        # Compute aggregate metrics
        self._compute_report_metrics(report)

        return report

    def _compute_report_metrics(self, report: ValidationReport):
        """Compute correlation and error metrics for the report."""
        # Get paired results where both methods succeeded
        paired = [
            (r.hdc_phi, r.pyphi_phi)
            for r in report.results
            if r.pyphi_phi is not None
        ]

        if len(paired) < 2:
            logger.warning("Not enough paired results for correlation")
            return

        hdc_values = np.array([p[0] for p in paired])
        pyphi_values = np.array([p[1] for p in paired])

        # Correlation
        correlation = np.corrcoef(hdc_values, pyphi_values)[0, 1]
        report.correlation = correlation

        # Errors
        errors = np.abs(hdc_values - pyphi_values)
        report.mean_absolute_error = float(np.mean(errors))
        report.max_absolute_error = float(np.max(errors))

        # Agreement within 10%
        relative_errors = errors / np.maximum(pyphi_values, 0.001)
        report.agreement_within_10_percent = float(
            np.mean(relative_errors < 0.10) * 100
        )

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
        md.append(f"**Status**: {'PASS' if report.correlation > 0.90 else 'NEEDS REVIEW'}")
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
        if report.correlation > 0.90:
            md.append("✅ **PASS**: Correlation r > 0.90 achieved!")
        elif report.correlation > 0.80:
            md.append("⚠️ **MARGINAL**: Correlation 0.80 < r < 0.90")
        else:
            md.append("❌ **FAIL**: Correlation r < 0.80")
        md.append("")

        md.append("## Detailed Results")
        md.append("")
        md.append("| Topology | Nodes | HDC-Φ | PyPhi-Φ | Error | Status |")
        md.append("|----------|-------|-------|---------|-------|--------|")

        for r in report.results:
            pyphi_str = f"{r.pyphi_phi:.4f}" if r.pyphi_phi else "N/A"
            if r.pyphi_phi:
                error = abs(r.hdc_phi - r.pyphi_phi)
                error_str = f"{error:.4f}"
                status = "✅" if error < 0.05 else "⚠️"
            else:
                error_str = "N/A"
                status = "⏳"

            md.append(f"| {r.topology_name} | {r.n_nodes} | {r.hdc_phi:.4f} | "
                     f"{pyphi_str} | {error_str} | {status} |")

        md.append("")
        md.append("## Notes")
        md.append("")
        md.append("- HDC-Φ: Approximation using algebraic connectivity of HDC similarity graph")
        md.append("- PyPhi-Φ: Exact IIT 3.0 calculation (intractable for n > 10)")
        md.append("- Error: Absolute difference between methods")
        md.append("- N/A: PyPhi computation failed or exceeded timeout")
        md.append("")

        md.append("## Methodology")
        md.append("")
        md.append("1. Generate identical topologies for both methods")
        md.append("2. Compute HDC-Φ using Symthaea's RealPhiCalculator")
        md.append("3. Compute exact Φ using PyPhi (IIT 3.0)")
        md.append("4. Calculate Pearson correlation coefficient")
        md.append("5. Report absolute and relative errors")
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

    if report.correlation > 0.90:
        print("\n✅ VALIDATION PASSED: r > 0.90")
    else:
        print("\n⚠️ VALIDATION NEEDS REVIEW")

    validator.generate_report(report, Path(args.output))
    print(f"\n📄 Report saved to: {args.output}")


if __name__ == "__main__":
    main()
