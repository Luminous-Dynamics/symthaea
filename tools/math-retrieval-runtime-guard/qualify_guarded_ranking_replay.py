#!/usr/bin/env python3
"""MATH-RET-E2E-001A: guarded S fixture through exact ranking replay.

This is a mechanical interoperability capsule. It deliberately uses synthetic
canonical-sparse representation wires whose geometry is frozen to reproduce the
existing guarded S fixture ranking [a1, a2, a3]. It makes no claim about the
real canonical-AST encoder or retrieval quality.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SEAM = ROOT / "tools" / "math-retrieval-runtime-seam"
SCRIPTS = ROOT / ".github" / "scripts"
WORK = HERE / "target" / "guarded-ranking-replay-fixture"
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


def sparse_wire(features: list[tuple[str, int]]) -> dict:
    return {
        "version": "math-canonical-sparse-wire-v1",
        "feature_order": "FeatureIdUtf8Ascending",
        "features": [
            {"feature_id": feature_id, "count": count}
            for feature_id, count in sorted(features)
        ],
    }


def build_synthetic_inputs(base, guarded, replay_core):
    exp_v21 = base.load_script(
        "validate-math-search-experiment-v2.1.py", "ranking_capsule_exp_preview"
    )
    preview = exp_v21.fixture()
    shared = preview["shared_contract"]
    budget = preview["budget"]

    candidates = sorted(
        [guarded.source_digest(n) for n in range(1, 30)]
        + [guarded.source_digest(0xA1), guarded.source_digest(0xA2), guarded.source_digest(0xA3)]
    )
    candidate_doc = {
        "version": "math-retrieval-candidate-set-v1",
        "candidate_set_id": "math-ret-e2e-001a-fixture-candidates",
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
    candidate_raw = base.canonical_bytes(candidate_doc)
    candidate_digest = base.digest_bytes(candidate_raw)

    representation_sha256 = base.digest_label("runtime-001b-representation-S")
    serialization_sha256 = base.digest_label("runtime-001b-item-serialization-S")
    policy = replay_core.policy_fixture("CanonicalSparseV1")

    wires: dict[str, tuple[dict, bytes]] = {}
    a1, a2, a3 = (guarded.source_digest(x) for x in (0xA1, 0xA2, 0xA3))
    for source in candidates:
        if source == a1:
            doc = sparse_wire([("a", 2), ("b", 2)])
        elif source == a2:
            doc = sparse_wire([("a", 3), ("b", 1)])
        elif source == a3:
            doc = sparse_wire([("a", 3)])
        else:
            doc = sparse_wire([(f"other:{source[-8:]}", 1)])
        wires[source] = (doc, base.canonical_bytes(doc))

    items = []
    for source in candidates:
        _, raw = wires[source]
        items.append(
            {
                "source_object_sha256": source,
                "representation_object_sha256": base.digest_bytes(raw),
                "serialized_bytes": len(raw),
                "payload_base64": base64.b64encode(raw).decode("ascii"),
            }
        )
    artifact = {
        "version": "math-retrieval-exact-index-artifact-v1",
        "index_id": "math-ret-runtime-001b-s",
        "authority": AUTH,
        "candidate_set_sha256": candidate_digest,
        "target_id": "S",
        "representation_sha256": representation_sha256,
        "item_serialization_sha256": serialization_sha256,
        "payload_encoding": "Base64",
        "item_order": "SourceObjectDigestAscending",
        "item_count": len(items),
        "items": items,
    }
    artifact_raw = base.canonical_bytes(artifact)
    artifact_digest = base.digest_bytes(artifact_raw)

    overrides = {
        "runtime-001b-score-S": replay_core.component_digest(policy["scoring"]),
        "runtime-001b-precision-S": replay_core.component_digest(policy["precision"]),
        "runtime-001b-query-S": replay_core.component_digest(policy["query_normalization"]),
        "runtime-001b-index-artifact-S": artifact_digest,
    }
    return {
        "shared": shared,
        "budget": budget,
        "candidate_doc": candidate_doc,
        "candidate_raw": candidate_raw,
        "candidate_digest": candidate_digest,
        "candidates": candidates,
        "representation_sha256": representation_sha256,
        "serialization_sha256": serialization_sha256,
        "policy": policy,
        "wires": wires,
        "artifact": artifact,
        "artifact_raw": artifact_raw,
        "artifact_digest": artifact_digest,
        "overrides": overrides,
        "expected_top3": [a1, a2, a3],
    }


def write_coverage(base, graph: dict, prepared: dict) -> tuple[Path, Path]:
    shared = graph["experiment"]["shared_contract"]
    target = {
        "target_id": "S",
        "channel": "Syntax",
        "representation_family": "CanonicalSparse",
        "representation_sha256": prepared["representation_sha256"],
        "item_serialization_sha256": prepared["serialization_sha256"],
        "input_stage": "ParsedFolFormulaExt",
        "max_serialized_item_bytes": graph["experiment"]["budget"]["retrieved_item_bytes_max"],
    }
    coverage = {
        "version": "math-retrieval-source-coverage-v1",
        "coverage_id": "math-ret-e2e-001a-synthetic-s-coverage",
        "authority": AUTH,
        "candidate_universe": {
            "corpus_snapshot_sha256": prepared["candidate_doc"]["corpus_snapshot_sha256"],
            "knowledge_boundary_sha256": prepared["candidate_doc"]["knowledge_boundary_sha256"],
            "candidate_eligibility_policy_sha256": prepared["candidate_doc"]["candidate_eligibility_policy_sha256"],
            "candidate_set_sha256": prepared["candidate_digest"],
            "candidate_count": len(prepared["candidates"]),
            "source_identity_kind": "SourceObjectDigest",
            "canonical_candidate_order": "SourceObjectDigestAscending",
        },
        "parser": {
            "source_object_contract_sha256": shared["source_object_contract_sha256"],
            "parser_contract_sha256": base.digest_label("e2e-001a-parser-contract"),
            "parser_implementation_sha256": base.digest_label("e2e-001a-parser-implementation"),
            "parsed_object_kind": "FolFormulaExt",
            "parsed_object_serialization_sha256": base.digest_label("e2e-001a-parsed-serialization"),
        },
        "normalizer": {
            "normalization_contract_sha256": shared["normalization_contract_sha256"],
            "normalization_implementation_sha256": shared["normalization_implementation_sha256"],
            "normalized_object_kind": "ExactNormalForm",
            "normalized_object_serialization_sha256": base.digest_label("e2e-001a-normalized-serialization"),
        },
        "coverage_policy": "CommonIntersectionRequired",
        "targets": [target],
        "rows": [],
    }
    for source in prepared["candidates"]:
        _, raw = prepared["wires"][source]
        coverage["rows"].append(
            {
                "source_object_sha256": source,
                "parse_status": "Parsed",
                "parsed_object_sha256": base.digest_label(f"e2e-001a-parsed:{source}"),
                "normalization_status": "Normalized",
                "normalized_object_sha256": base.digest_label(f"e2e-001a-normalized:{source}"),
                "representations": [
                    {
                        "target_id": "S",
                        "status": "Ready",
                        "representation_object_sha256": base.digest_bytes(raw),
                        "serialized_bytes": len(raw),
                    }
                ],
            }
        )
    coverage_path = WORK / "source-coverage.json"
    base.write_json(coverage_path, coverage)
    report_path = WORK / "source-coverage-report.json"
    base.run(
        sys.executable,
        SCRIPTS / "validate-math-retrieval-source-coverage.py",
        coverage_path,
        graph["candidate_path"],
        graph["experiment_path"],
        "--report",
        report_path,
    )
    return coverage_path, report_path


def write_index_build(base, graph: dict, prepared: dict, coverage_path: Path) -> tuple[Path, Path, Path]:
    artifact_path = WORK / "exact-index-S.json"
    artifact_path.write_bytes(prepared["artifact_raw"])
    index_path = GRAPH / "indices" / "S.json"
    index_doc = json.loads(index_path.read_text(encoding="utf-8"))
    if index_doc["index"]["index_artifact_sha256"] != prepared["artifact_digest"]:
        raise RuntimeError("S index manifest did not bind precomputed exact artifact")
    build_mod = base.load_script("validate-math-retrieval-index-build.py", "e2e_index_build")
    input_set_sha256 = build_mod.input_set_sha(prepared["artifact"]["items"])
    receipt = {
        "version": "math-retrieval-index-build-receipt-v1",
        "receipt_id": "math-ret-e2e-001a-S-build",
        "authority": AUTH,
        "coverage_sha256": base.digest_bytes(coverage_path.read_bytes()),
        "candidate_set_sha256": prepared["candidate_digest"],
        "index_manifest_sha256": base.digest_bytes(index_path.read_bytes()),
        "index_artifact_sha256": prepared["artifact_digest"],
        "target_id": "S",
        "index_build_policy_sha256": index_doc["index"]["index_build_policy_sha256"],
        "builder_implementation_sha256": base.digest_label("e2e-001a-transparent-builder"),
        "toolchain_manifest_sha256": graph["experiment"]["shared_contract"]["toolchain_manifest_sha256"],
        "index_seed": index_doc["index"]["index_seed"],
        "input_set_sha256": input_set_sha256,
        "build_mode": "ExactDeterministicMaterializedScan",
    }
    receipt_path = WORK / "index-build-receipt.json"
    base.write_json(receipt_path, receipt)
    report_path = WORK / "index-build-report.json"
    base.run(
        sys.executable,
        SCRIPTS / "validate-math-retrieval-index-build.py",
        receipt_path,
        coverage_path,
        graph["candidate_path"],
        graph["experiment_path"],
        index_path,
        artifact_path,
        "--report",
        report_path,
    )
    return index_path, artifact_path, report_path


def write_query_representation(base, graph: dict, prepared: dict) -> tuple[Path, Path, Path]:
    query_doc = sparse_wire([("a", 1), ("b", 1)])
    query_path = WORK / "query-wire-S.json"
    query_digest = base.write_json(query_path, query_doc)
    query_raw = query_path.read_bytes()
    receipt = {
        "version": "math-retrieval-query-representation-v1",
        "receipt_id": "math-ret-e2e-001a-query-S",
        "authority": AUTH,
        "query_id": "fixture-query-001",
        "query_source_object_sha256": guarded_source_digest(0xF0),
        "source_object_contract_sha256": graph["experiment"]["shared_contract"]["source_object_contract_sha256"],
        "input_stage": "ParsedFolFormulaExt",
        "input_object_sha256": base.digest_label("e2e-001a-query-parsed-object"),
        "representation_sha256": prepared["representation_sha256"],
        "item_serialization_sha256": prepared["serialization_sha256"],
        "wire_kind": "CanonicalSparseV1",
        "producer_implementation_sha256": base.digest_label("e2e-001a-query-producer"),
        "toolchain_manifest_sha256": graph["experiment"]["shared_contract"]["toolchain_manifest_sha256"],
        "query_wire_sha256": query_digest,
        "serialized_bytes": len(query_raw),
    }
    receipt_path = WORK / "query-representation-receipt.json"
    base.write_json(receipt_path, receipt)
    report_path = WORK / "query-representation-report.json"
    base.run(
        sys.executable,
        SCRIPTS / "validate-math-retrieval-query-representation.py",
        receipt_path,
        query_path,
        "--report",
        report_path,
    )
    return query_path, receipt_path, report_path


def guarded_source_digest(n: int) -> str:
    return f"sha256:{n:064x}"


def main() -> int:
    if WORK.exists():
        shutil.rmtree(WORK)
    GRAPH.mkdir(parents=True)

    guarded = load_path(HERE / "qualify_guarded_adapter.py", "e2e_guarded_parent")
    base = load_path(SEAM / "qualify_contract_adapter.py", "e2e_contract_parent")
    replay_core = load_path(SCRIPTS / "math-retrieval-ranking-replay-core.py", "e2e_rank_core")
    prepared = build_synthetic_inputs(base, guarded, replay_core)

    # Redirect all predecessor builders into this capsule's isolated tree.
    guarded.WORK = WORK
    guarded.GRAPH = GRAPH
    guarded.RUNTIME = RUNTIME
    guarded.NEGATIVE_RUNTIME = NEGATIVE_RUNTIME

    original_digest_label = base.digest_label
    def replay_digest_label(label: str) -> str:
        return prepared["overrides"].get(label, original_digest_label(label))
    base.digest_label = replay_digest_label
    try:
        graph = guarded.build_real_candidate_graph(base)
    finally:
        base.digest_label = original_digest_label

    if graph["candidate_path"].read_bytes() != prepared["candidate_raw"]:
        raise RuntimeError("materialized candidate artifact differs from precomputed bytes")
    if graph["candidate_digest"] != prepared["candidate_digest"]:
        raise RuntimeError("candidate digest drift")
    if graph["s_index_artifact_digest"] != prepared["artifact_digest"]:
        raise RuntimeError("S index artifact commitment drift")

    # Existing candidate + graph qualification remains unchanged.
    base.run(sys.executable, SCRIPTS / "validate-math-retrieval-candidate-set.py", graph["candidate_path"])
    graph_report_path = WORK / "graph-report.json"
    base.run(
        sys.executable,
        SCRIPTS / "validate-math-retrieval-graph-v1.1.py",
        graph["bundle_path"],
        "--repo-root", ROOT,
        "--report", graph_report_path,
    )
    graph_report = json.loads(graph_report_path.read_text(encoding="utf-8"))
    if not graph_report.get("all_checks_passed"):
        raise RuntimeError("graph qualification did not pass")

    coverage_path, coverage_report_path = write_coverage(base, graph, prepared)
    index_path, artifact_path, build_report_path = write_index_build(base, graph, prepared, coverage_path)
    query_path, query_receipt_path, query_report_path = write_query_representation(base, graph, prepared)
    policy_path = WORK / "scoring-replay-policy-S.json"
    base.write_json(policy_path, prepared["policy"])

    config_path = base.write_runtime_config(graph, graph_report, graph_report_path)
    guarded.append_candidate_config(base, config_path, graph)

    # Negative membership canary first.
    base.run(
        "cargo", "run", "--quiet",
        "--manifest-path", HERE / "Cargo.toml", "--locked",
        "--bin", "emit_guarded_fixture", "--",
        config_path, NEGATIVE_RUNTIME, "--inject-illegal",
    )
    if NEGATIVE_RUNTIME.exists():
        raise RuntimeError("illegal-candidate execution created runtime evidence")

    base.run(
        "cargo", "run", "--quiet",
        "--manifest-path", HERE / "Cargo.toml", "--locked",
        "--bin", "emit_guarded_fixture", "--",
        config_path, RUNTIME,
    )
    trace_path = RUNTIME / "trace.json"
    trace_report_path = WORK / "trace-report.json"
    base.run(
        sys.executable, SCRIPTS / "validate-math-retrieval-trace.py",
        trace_path, graph["bundle_path"], "--repo-root", ROOT,
        "--report", trace_report_path,
    )
    membership_report_path = WORK / "candidate-membership-report.json"
    base.run(
        sys.executable, SCRIPTS / "validate-math-retrieval-candidate-membership.py",
        graph["candidate_path"], trace_path, graph["bundle_path"],
        "--repo-root", ROOT, "--report", membership_report_path,
    )
    audit_path = base.build_payload_audit(trace_path, graph, graph_report_path)
    payload_report_path = WORK / "payload-report.json"
    base.run(
        sys.executable, SCRIPTS / "validate-math-retrieval-payload-audit.py",
        audit_path, trace_path, graph["bundle_path"],
        "--repo-root", ROOT, "--payload-root", RUNTIME,
        "--report", payload_report_path,
    )

    # Bind the already-qualified artifacts into independent ranking replay.
    replay_receipt = {
        "version": "math-retrieval-ranking-replay-receipt-v1",
        "receipt_id": "math-ret-e2e-001a-S-replay",
        "authority": AUTH,
        "trace_selector": "SingleIndex",
        "scoring_policy_bundle_sha256": base.digest_bytes(policy_path.read_bytes()),
        "query_representation_receipt_sha256": base.digest_bytes(query_receipt_path.read_bytes()),
        "query_representation_report_sha256": base.digest_bytes(query_report_path.read_bytes()),
        "index_build_report_sha256": base.digest_bytes(build_report_path.read_bytes()),
        "index_manifest_sha256": base.digest_bytes(index_path.read_bytes()),
        "index_artifact_sha256": base.digest_bytes(artifact_path.read_bytes()),
        "trace_sha256": base.digest_bytes(trace_path.read_bytes()),
        "trace_validation_report_sha256": base.digest_bytes(trace_report_path.read_bytes()),
    }
    replay_receipt_path = WORK / "ranking-replay-receipt.json"
    base.write_json(replay_receipt_path, replay_receipt)
    replay_report_path = WORK / "ranking-replay-report.json"
    witness_path = WORK / "ranking-replay-witness.json"
    base.run(
        sys.executable, SCRIPTS / "validate-math-retrieval-ranking-replay.py",
        replay_receipt_path, policy_path, query_receipt_path, query_report_path,
        query_path, build_report_path, index_path, artifact_path,
        trace_path, trace_report_path,
        "--report", replay_report_path, "--witness", witness_path,
    )
    replay_report = json.loads(replay_report_path.read_text(encoding="utf-8"))
    if replay_report.get("expected_top_k") != prepared["expected_top3"]:
        raise RuntimeError("independent replay did not recover frozen synthetic S top-3")
    if replay_report.get("runtime_top_k") != prepared["expected_top3"]:
        raise RuntimeError("Rust guarded runtime ranking differs from frozen synthetic S top-3")
    if not replay_report.get("all_checks_passed"):
        raise RuntimeError("ranking replay qualification did not pass")

    summary = {
        "authority": AUTH,
        "fixture_representation_kind": "SyntheticCanonicalSparseForMechanicalQualificationOnly",
        "candidate_set_sha256": prepared["candidate_digest"],
        "experiment_sha256": graph["experiment_digest"],
        "bundle_sha256": base.digest_bytes(graph["bundle_path"].read_bytes()),
        "source_coverage_sha256": base.digest_bytes(coverage_path.read_bytes()),
        "source_coverage_report_sha256": base.digest_bytes(coverage_report_path.read_bytes()),
        "index_artifact_sha256": prepared["artifact_digest"],
        "index_build_report_sha256": base.digest_bytes(build_report_path.read_bytes()),
        "query_representation_report_sha256": base.digest_bytes(query_report_path.read_bytes()),
        "trace_sha256": base.digest_bytes(trace_path.read_bytes()),
        "candidate_membership_report_sha256": base.digest_bytes(membership_report_path.read_bytes()),
        "payload_report_sha256": base.digest_bytes(payload_report_path.read_bytes()),
        "ranking_replay_report_sha256": base.digest_bytes(replay_report_path.read_bytes()),
        "ranking_replay_witness_sha256": base.digest_bytes(witness_path.read_bytes()),
        "illegal_candidate_rejected_before_evidence": True,
        "expected_and_runtime_top3": prepared["expected_top3"],
        "all_checks_passed": True,
    }
    (WORK / "qualification-summary.json").write_bytes(base.canonical_bytes(summary))
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
