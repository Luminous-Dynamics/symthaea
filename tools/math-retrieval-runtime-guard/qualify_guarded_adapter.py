#!/usr/bin/env python3
"""MATH-RET-RUNTIME-001D guarded S-arm end-to-end qualification.

Reuses the predecessor #4324 graph builder, but replaces its opaque candidate-set
placeholder with the SHA-256 of an actual MATH-RET-CANDIDATE-001A artifact.
The runtime backend is then executed through MembershipGuardBackend and the
resulting evidence must satisfy the unchanged graph/trace/payload validators plus
the candidate-membership validator.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SEAM = ROOT / "tools" / "math-retrieval-runtime-seam"
SCRIPTS = ROOT / ".github" / "scripts"
WORK = HERE / "target" / "guarded-adapter-fixture"
GRAPH = WORK / "graph"
RUNTIME = WORK / "runtime-positive"
NEGATIVE_RUNTIME = WORK / "runtime-negative"
AUTH = "MeasurementOnly"


def load_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_digest(n: int) -> str:
    return f"sha256:{n:064x}"


def build_real_candidate_graph(base):
    # Redirect the predecessor builder into this tranche's isolated evidence tree.
    base.WORK = WORK
    base.GRAPH = GRAPH
    base.RUNTIME = RUNTIME

    exp_v21 = base.load_script(
        "validate-math-search-experiment-v2.1.py", "guarded_adapter_exp_v21_preview"
    )
    preview = exp_v21.fixture()
    shared = preview["shared_contract"]

    # 29 ordinary fixture identities + the exact three identities the positive
    # S-arm backend will return. Fixed-width lowercase hex makes lexical order
    # identical to integer order here.
    candidates = sorted(
        [source_digest(n) for n in range(1, 30)]
        + [source_digest(0xA1), source_digest(0xA2), source_digest(0xA3)]
    )
    assert len(candidates) == 32 and len(set(candidates)) == 32
    assert source_digest(0xEE) not in candidates

    candidate_doc = {
        "version": "math-retrieval-candidate-set-v1",
        "candidate_set_id": "math-ret-runtime-001d-fixture-candidates",
        "authority": AUTH,
        "corpus_snapshot_sha256": shared["corpus_snapshot_sha256"],
        "knowledge_boundary_sha256": shared["knowledge_boundary_sha256"],
        "candidate_eligibility_policy_sha256": base.digest_label(
            "runtime-001b-candidate-eligibility"
        ),
        "source_identity_kind": "SourceObjectDigest",
        "canonical_order": "SourceObjectDigestAscending",
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    candidate_path = GRAPH / "candidate-set.json"
    candidate_digest = base.write_json(candidate_path, candidate_doc)

    # The predecessor graph builder already centralizes the candidate-set
    # placeholder in one label. Replace only that label so every generated
    # index/binding/experiment digest cascades from the actual candidate file.
    original_digest_label = base.digest_label

    def guarded_digest_label(label: str) -> str:
        if label == "runtime-001b-candidate-set":
            return candidate_digest
        return original_digest_label(label)

    base.digest_label = guarded_digest_label
    try:
        graph = base.build_graph_fixture()
    finally:
        base.digest_label = original_digest_label

    universe = graph["candidate_universe"]
    assert universe["candidate_set_sha256"] == candidate_digest
    assert universe["candidate_count"] == len(candidates)
    assert universe["corpus_snapshot_sha256"] == candidate_doc["corpus_snapshot_sha256"]
    assert universe["knowledge_boundary_sha256"] == candidate_doc["knowledge_boundary_sha256"]
    assert (
        universe["candidate_eligibility_policy_sha256"]
        == candidate_doc["candidate_eligibility_policy_sha256"]
    )

    # Give this successor evidence event its own experiment identity. This only
    # changes the experiment artifact and bundle reference; index/binding
    # identities remain exactly those produced by the predecessor builder.
    graph["experiment"]["experiment_id"] = "math-ret-runtime-001d-guarded-fixture"
    graph["experiment_digest"] = base.write_json(
        graph["experiment_path"], graph["experiment"]
    )
    bundle = json.loads(graph["bundle_path"].read_text(encoding="utf-8"))
    rows = [x for x in bundle["artifacts"] if x["kind"] == "ExperimentManifest"]
    if len(rows) != 1:
        raise RuntimeError("bundle must contain exactly one ExperimentManifest")
    rows[0]["sha256"] = graph["experiment_digest"]
    base.write_json(graph["bundle_path"], bundle)

    graph["candidate_path"] = candidate_path
    graph["candidate_digest"] = candidate_digest
    graph["candidate_sources"] = candidates
    return graph


def append_candidate_config(base, config_path: Path, graph: dict) -> None:
    with config_path.open("a", encoding="utf-8") as handle:
        handle.write("candidate_source_sha256s=" + ",".join(graph["candidate_sources"]) + "\n")


def main() -> int:
    if WORK.exists():
        shutil.rmtree(WORK)
    GRAPH.mkdir(parents=True)

    base = load_path(SEAM / "qualify_contract_adapter.py", "guarded_adapter_parent")
    graph = build_real_candidate_graph(base)

    # First prove the materialized candidate artifact is internally canonical.
    base.run(
        sys.executable,
        SCRIPTS / "validate-math-retrieval-candidate-set.py",
        graph["candidate_path"],
    )

    graph_report_path = WORK / "graph-report.json"
    base.run(
        sys.executable,
        SCRIPTS / "validate-math-retrieval-graph-v1.1.py",
        graph["bundle_path"],
        "--repo-root",
        ROOT,
        "--report",
        graph_report_path,
    )
    graph_report = json.loads(graph_report_path.read_text(encoding="utf-8"))
    if not graph_report.get("all_checks_passed"):
        raise RuntimeError("graph qualification did not pass")
    if graph_report["shared_candidate_set_sha256"] != graph["candidate_digest"]:
        raise RuntimeError("qualified graph does not bind materialized candidate-set digest")
    if graph_report["shared_candidate_count"] != len(graph["candidate_sources"]):
        raise RuntimeError("qualified graph candidate count differs from materialized universe")

    config_path = base.write_runtime_config(graph, graph_report, graph_report_path)
    append_candidate_config(base, config_path, graph)

    # Negative canary first: an injected out-of-universe source must fail in the
    # membership wrapper before materialization or evidence and create no output.
    base.run(
        "cargo",
        "run",
        "--quiet",
        "--manifest-path",
        HERE / "Cargo.toml",
        "--locked",
        "--bin",
        "emit_guarded_fixture",
        "--",
        config_path,
        NEGATIVE_RUNTIME,
        "--inject-illegal",
    )
    if NEGATIVE_RUNTIME.exists():
        raise RuntimeError("illegal-candidate execution created runtime evidence")

    # Positive guarded execution.
    base.run(
        "cargo",
        "run",
        "--quiet",
        "--manifest-path",
        HERE / "Cargo.toml",
        "--locked",
        "--bin",
        "emit_guarded_fixture",
        "--",
        config_path,
        RUNTIME,
    )

    trace_path = RUNTIME / "trace.json"
    trace_report_path = WORK / "trace-report.json"
    base.run(
        sys.executable,
        SCRIPTS / "validate-math-retrieval-trace.py",
        trace_path,
        graph["bundle_path"],
        "--repo-root",
        ROOT,
        "--report",
        trace_report_path,
    )
    trace_report = json.loads(trace_report_path.read_text(encoding="utf-8"))
    if not trace_report.get("all_checks_passed"):
        raise RuntimeError("trace qualification did not pass")

    membership_report_path = WORK / "candidate-membership-report.json"
    base.run(
        sys.executable,
        SCRIPTS / "validate-math-retrieval-candidate-membership.py",
        graph["candidate_path"],
        trace_path,
        graph["bundle_path"],
        "--repo-root",
        ROOT,
        "--report",
        membership_report_path,
    )
    membership_report = json.loads(membership_report_path.read_text(encoding="utf-8"))
    if not membership_report.get("all_checks_passed"):
        raise RuntimeError("candidate-membership qualification did not pass")

    audit_path = base.build_payload_audit(trace_path, graph, graph_report_path)
    payload_report_path = WORK / "payload-report.json"
    base.run(
        sys.executable,
        SCRIPTS / "validate-math-retrieval-payload-audit.py",
        audit_path,
        trace_path,
        graph["bundle_path"],
        "--repo-root",
        ROOT,
        "--payload-root",
        RUNTIME,
        "--report",
        payload_report_path,
    )
    payload_report = json.loads(payload_report_path.read_text(encoding="utf-8"))
    if not payload_report.get("all_checks_passed"):
        raise RuntimeError("payload qualification did not pass")

    summary = {
        "authority": AUTH,
        "experiment_sha256": graph["experiment_digest"],
        "candidate_set_sha256": graph["candidate_digest"],
        "candidate_count": len(graph["candidate_sources"]),
        "bundle_sha256": base.digest_bytes(graph["bundle_path"].read_bytes()),
        "graph_report_sha256": base.digest_bytes(graph_report_path.read_bytes()),
        "trace_sha256": base.digest_bytes(trace_path.read_bytes()),
        "candidate_membership_report_sha256": base.digest_bytes(
            membership_report_path.read_bytes()
        ),
        "payload_audit_sha256": base.digest_bytes(audit_path.read_bytes()),
        "illegal_candidate_rejected_before_evidence": True,
        "graph_all_checks_passed": True,
        "trace_all_checks_passed": True,
        "candidate_membership_all_checks_passed": True,
        "payload_all_checks_passed": True,
    }
    (WORK / "qualification-summary.json").write_bytes(base.canonical_bytes(summary))
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
