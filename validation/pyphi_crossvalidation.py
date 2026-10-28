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
        """Compare both methods on one exact TPM, state, profile, and model digest."""
        logger.info("Validating %s (n=%d)...", topology.name, topology.n_nodes)
        result = ValidationResult(
            topology_name=topology.name,
            n_nodes=topology.n_nodes,
            model_sha256="",
        )
        try:
            tpm, state, payload = self.build_model(topology)
            result.model_sha256 = payload["model_sha256"]
        except Exception as exc:
            result.error_message = f"model construction: {type(exc).__name__}: {exc}"
            return result

        errors: List[str] = []
        try:
            value, elapsed, metric_id = self.compute_candidate_phi(payload)
            result.candidate_phi = value
            result.candidate_computation_time = elapsed
            result.candidate_metric_id = metric_id
        except Exception as exc:
            errors.append(f"candidate: {type(exc).__name__}: {exc}")

        try:
            value, elapsed = self.compute_pyphi_phi(topology, tpm, state)
            result.pyphi_phi_s = value
            result.pyphi_computation_time = elapsed
        except Exception as exc:
            errors.append(f"PyPhi: {type(exc).__name__}: {exc}")

        result.error_message = "; ".join(errors) if errors else None
        return result

    def run_validation(
        self,
        max_nodes: int = 8,
        min_valid_pairs: Optional[int] = None,
        min_pearson: Optional[float] = None,
    ) -> ValidationReport:
        """Run the paired-model comparison; qualification is disabled without explicit gates."""
        report = ValidationReport(
            pyphi_version=PYPHI_VERSION,
            formalism_profile=FORMALISM_PROFILE,
            min_valid_pairs=min_valid_pairs,
            min_pearson=min_pearson,
        )
        if max_nodes < 2 or max_nodes > MAX_EXACT_NODES:
            report.qualification_reasons.append(
                f"max_nodes must be in [2, {MAX_EXACT_NODES}], got {max_nodes}"
            )
            return report
        if not PYPHI_AVAILABLE:
            report.qualification_reasons.append("PyPhi is unavailable")
        if PYPHI_VERSION != EXPECTED_PYPHI_VERSION:
            report.qualification_reasons.append(
                f"PyPhi version must be exactly {EXPECTED_PYPHI_VERSION}; got {PYPHI_VERSION!r}"
            )
        if min_valid_pairs is None or min_valid_pairs < 2:
            report.qualification_reasons.append(
                "A preregistered --min-valid-pairs >= 2 gate was not supplied"
            )
        if min_pearson is None or not math.isfinite(min_pearson) or not -1.0 <= min_pearson <= 1.0:
            report.qualification_reasons.append(
                "A preregistered finite --min-pearson gate in [-1, 1] was not supplied"
            )

        topologies = [
            topology for topology in self.create_test_topologies()
            if topology.n_nodes <= max_nodes
        ]
        if not topologies:
            report.qualification_reasons.append("No topologies were selected")
            return report

        report.results = [self.validate_topology(topology) for topology in topologies]
        paired = [
            item for item in report.results
            if item.candidate_phi is not None and item.pyphi_phi_s is not None
            and item.error_message is None and len(item.model_sha256) == 64
        ]
        report.n_validated = len(paired)
        report.n_failed = len(report.results) - report.n_validated

        self._compute_report_metrics(report)

        if report.n_failed:
            report.qualification_reasons.append(
                f"{report.n_failed} of {len(report.results)} selected cases did not produce a valid paired result"
            )
        if min_valid_pairs is not None and report.n_validated < min_valid_pairs:
            report.qualification_reasons.append(
                f"Only {report.n_validated} valid pairs; minimum required is {min_valid_pairs}"
            )
        if report.correlation is None or not math.isfinite(report.correlation):
            report.qualification_reasons.append("Pearson correlation is unavailable or non-finite")
        elif min_pearson is not None and math.isfinite(min_pearson) and report.correlation < min_pearson:
            report.qualification_reasons.append(
                f"Pearson correlation {report.correlation:.6f} is below the preregistered threshold {min_pearson:.6f}"
            )

        # This can qualify only the candidate/reference comparison on this declared
        # finite fixture set. It does not certify formal equivalence to IIT.
        if not report.qualification_reasons:
            report.qualification_status = "COMPARISON_CRITERIA_MET_NOT_FORMAL_EQUIVALENCE"
        else:
            report.qualification_status = "NOT_QUALIFIED"
        return report

    def _compute_report_metrics(self, report: ValidationReport) -> None:
        """Calculate metrics only from finite, explicitly paired results."""
        paired = [
            (r.candidate_phi, r.pyphi_phi_s)
            for r in report.results
            if r.candidate_phi is not None and r.pyphi_phi_s is not None
            and r.error_message is None and len(r.model_sha256) == 64
            and math.isfinite(r.candidate_phi) and math.isfinite(r.pyphi_phi_s)
        ]
        report.correlation = None
        report.mean_absolute_error = None
        report.max_absolute_error = None
        report.agreement_within_10_percent = None
        if len(paired) < 2:
            report.qualification_reasons.append(
                f"Need at least 2 finite paired measurements to calculate metrics; got {len(paired)}"
            )
            return

        candidate_values = np.asarray([p[0] for p in paired], dtype=np.float64)
        reference_values = np.asarray([p[1] for p in paired], dtype=np.float64)
        if (
            not np.all(np.isfinite(candidate_values))
            or not np.all(np.isfinite(reference_values))
            or float(np.ptp(candidate_values)) <= 1e-12
            or float(np.ptp(reference_values)) <= 1e-12
        ):
            report.qualification_reasons.append(
                "Correlation is undefined: data are non-finite or one score vector is constant"
            )
            return

        correlation = float(np.corrcoef(candidate_values, reference_values)[0, 1])
        if not math.isfinite(correlation):
            report.qualification_reasons.append("Pearson correlation is non-finite")
            return
        errors = np.abs(candidate_values - reference_values)
        relative_errors = errors / np.maximum(np.abs(reference_values), 1e-12)
        report.correlation = correlation
        report.mean_absolute_error = float(np.mean(errors))
        report.max_absolute_error = float(np.max(errors))
        report.agreement_within_10_percent = float(np.mean(relative_errors <= 0.10) * 100.0)

    def generate_report(self, report: ValidationReport, output_path: Path) -> None:
        """Write a claim-bounded report; unqualified reports never print a PASS."""
        md = [
            "# Symthaea / PyPhi Paired-Model Comparison",
            "",
            f"**Generated:** {__import__('datetime').datetime.now().isoformat()}",
            f"**Status:** {report.qualification_status}",
            f"**PyPhi version:** {report.pyphi_version or 'unavailable'}",
            f"**Formalism profile:** {report.formalism_profile}",
            f"**Candidate metric:** {RUST_CANDIDATE_METRIC_ID}",
            f"**Minimum valid pairs:** {report.min_valid_pairs if report.min_valid_pairs is not None else 'NOT SET'}",
            f"**Minimum Pearson r:** {report.min_pearson if report.min_pearson is not None else 'NOT SET'}",
            "",
            "## Claim ceiling",
            "",
            "The Rust value is a candidate full-system-mechanism MIP score computed from an explicit TPM. "
            "It is compared with PyPhi's IIT_3_0 system integrated information (φ_s) on the same TPM "
            "and state. This comparison does not establish formal equivalence to IIT, agreement outside "
            "the selected finite cases, or any claim about consciousness.",
            "",
            "A candidate score is never substituted when the Rust adapter fails. Missing and invalid "
            "results remain unpaired and count against qualification.",
            "",
            "## Summary statistics",
            "",
            f"- **Pearson r:** {report.correlation:.6f}" if report.correlation is not None else "- **Pearson r:** unavailable",
            f"- **Mean absolute error:** {report.mean_absolute_error:.6f}" if report.mean_absolute_error is not None else "- **Mean absolute error:** unavailable",
            f"- **Maximum absolute error:** {report.max_absolute_error:.6f}" if report.max_absolute_error is not None else "- **Maximum absolute error:** unavailable",
            f"- **Agreement within 10% (relative to |φ_s|):** {report.agreement_within_10_percent:.2f}%" if report.agreement_within_10_percent is not None else "- **Agreement within 10%:** unavailable",
            f"- **Valid paired cases:** {report.n_validated}",
            f"- **Unpaired/failed cases:** {report.n_failed}",
            "",
            "## Qualification reasons",
            "",
        ]
        if report.qualification_reasons:
            md.extend(f"- {reason}" for reason in report.qualification_reasons)
        else:
            md.append("- All declared comparison gates passed for this finite fixture set.")
        md.extend([
            "",
            "## Per-case results",
            "",
            "| Case | Nodes | Model SHA-256 | Candidate score | PyPhi φ_s | Abs. error | Status |",
            "|---|---:|---|---:|---:|---:|---|",
        ])
        for result in report.results:
            paired = result.candidate_phi is not None and result.pyphi_phi_s is not None and result.error_message is None
            error = abs(result.candidate_phi - result.pyphi_phi_s) if paired else None
            candidate = f"{result.candidate_phi:.8f}" if result.candidate_phi is not None else "N/A"
            reference = f"{result.pyphi_phi_s:.8f}" if result.pyphi_phi_s is not None else "N/A"
            error_text = f"{error:.8f}" if error is not None else "N/A"
            status = "PAIRED" if paired else "FAILED / UNPAIRED"
            md.append(
                f"| {result.topology_name} | {result.n_nodes} | {result.model_sha256 or 'N/A'} | "
                f"{candidate} | {reference} | {error_text} | {status} |"
            )
            if result.error_message:
                md.append("")
                md.append(f"**{result.topology_name} error:** {result.error_message}")
        md.extend([
            "",
            "## Reproducibility contract",
            "",
            "- TPM rows are state-by-node; state ordering is little-endian (unit 0 is the least significant bit).",
            "- Both implementations receive the same TPM, candidate state, topology-derived connectivity, and IIT_3_0 profile.",
            "- The model SHA-256 is computed over the canonical JSON model description and returned by the Rust adapter; a mismatched digest fails the pair.",
            "- The current acceptance status is for this finite fixture set only; held-out generator families and independent known-answer fixtures remain separate gates.",
            "",
        ])
        output_path.write_text("\n".join(md) + "\n", encoding="utf-8")
        logger.info("Report saved to %s", output_path)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the comparison; exit nonzero unless explicitly gated criteria are met."""
    import argparse

    parser = argparse.ArgumentParser(description="Fail-closed Symthaea / PyPhi comparison")
    parser.add_argument("--max-nodes", type=int, default=4,
                        help="Maximum nodes to test (2..8; default 4 keeps the first run small)")
    parser.add_argument("--min-valid-pairs", type=int, default=None,
                        help="Predeclared minimum number of valid paired cases; required to qualify")
    parser.add_argument("--min-pearson", type=float, default=None,
                        help="Predeclared minimum Pearson r in [-1,1]; required to qualify")
    parser.add_argument("--cargo", type=str, default="cargo",
                        help="Cargo executable used to run the maintained Rust TPM adapter")
    parser.add_argument("--output", type=str, default="PYPHI_VALIDATION_REPORT.md",
                        help="Output report path")
    args = parser.parse_args(argv)

    if args.max_nodes < 2 or args.max_nodes > MAX_EXACT_NODES:
        parser.error(f"--max-nodes must be in [2, {MAX_EXACT_NODES}]")
    if args.min_valid_pairs is not None and args.min_valid_pairs < 2:
        parser.error("--min-valid-pairs must be >= 2")
    if args.min_pearson is not None and (not math.isfinite(args.min_pearson) or not -1.0 <= args.min_pearson <= 1.0):
        parser.error("--min-pearson must be finite and in [-1,1]")

    validator = PyPhiValidator(symthaea_binary=args.cargo)
    report = validator.run_validation(
        max_nodes=args.max_nodes,
        min_valid_pairs=args.min_valid_pairs,
        min_pearson=args.min_pearson,
    )
    validator.generate_report(report, Path(args.output))
    print(f"Status: {report.qualification_status}")
    print(f"Valid paired cases: {report.n_validated}; unpaired/failed: {report.n_failed}")
    print(f"Report saved to: {args.output}")
    if report.qualification_status != "COMPARISON_CRITERIA_MET_NOT_FORMAL_EQUIVALENCE":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
