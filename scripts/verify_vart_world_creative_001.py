#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SHA256_HEX = set("0123456789abcdef")
FORBIDDEN_AGGREGATES = {
    "world_quality",
    "creative_score",
    "beauty_score",
    "cinematic_quality",
    "intelligence_score",
}
PAIR_POLICIES = {"full_symthaea", "random_valid", "heuristic"}


class Reject(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise Reject("INCOMPLETE_EVIDENCE_CLOSURE", str(path))
    except json.JSONDecodeError as exc:
        raise Reject("INVALID_JSON", f"{path}: {exc}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    except FileNotFoundError:
        raise Reject("INCOMPLETE_EVIDENCE_CLOSURE", str(path))
    return h.hexdigest()


def require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in SHA256_HEX for c in value)
    ):
        raise Reject("MANIFEST_SCHEMA_INVALID", f"{name} is not lowercase sha256 hex")
    return value


def require(cond: bool, code: str, detail: str) -> None:
    if not cond:
        raise Reject(code, detail)


def walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)


def sha256_counter_draw(
    seed: int,
    paired_block_id: str,
    candidate_set_sha256: str,
    count: int,
) -> tuple[int, int, str]:
    """Return (selected_index, accepted_counter, accepted_digest_hex)."""
    if count <= 0:
        raise ValueError("count must be positive")
    domain = b"SYMTHAEA-VART-RANDOM-VALID-v1\x00"
    seed_bytes = seed.to_bytes(8, "big", signed=False)
    pair = paired_block_id.encode("utf-8")
    cset = bytes.fromhex(candidate_set_sha256)
    limit = (1 << 256) - ((1 << 256) % count)
    counter = 0
    while True:
        digest = hashlib.sha256(
            domain
            + seed_bytes
            + len(pair).to_bytes(4, "big")
            + pair
            + cset
            + counter.to_bytes(8, "big")
        ).digest()
        x = int.from_bytes(digest, "big")
        if x < limit:
            return x % count, counter, digest.hex()
        counter += 1


def validate_manifest_shape(m: dict[str, Any]) -> None:
    required = [
        "schema",
        "experiment_id",
        "campaign",
        "trial_id",
        "paired_block_id",
        "policy",
        "policy_sha256",
        "world_fixture_sha256",
        "seed",
        "revision_index",
        "world_version_before",
        "experience_episode_sha256",
        "revision_hypothesis_sha256",
        "candidate_set_sha256",
        "admissible_candidate_count",
        "selected_proposal_sha256",
        "selection_index",
        "included_in_confirmatory_analysis",
        "trial_state",
        "evidence_bundle_sha256",
        "metric_definition_set_sha256",
        "analysis_contract_sha256",
    ]
    for key in required:
        require(key in m, "MANIFEST_SCHEMA_INVALID", f"missing {key}")
    require(
        m["schema"] == "symthaea.vart-world-creative-001.trial-manifest.v1",
        "MANIFEST_SCHEMA_INVALID",
        "schema",
    )
    require(
        m["experiment_id"] == "VART-WORLD-CREATIVE-001",
        "MANIFEST_SCHEMA_INVALID",
        "experiment_id",
    )
    for key in [
        "policy_sha256",
        "world_fixture_sha256",
        "experience_episode_sha256",
        "revision_hypothesis_sha256",
        "candidate_set_sha256",
        "selected_proposal_sha256",
        "evidence_bundle_sha256",
        "metric_definition_set_sha256",
        "analysis_contract_sha256",
    ]:
        require_sha256(m[key], key)
    require(
        isinstance(m["seed"], int) and m["seed"] >= 0,
        "MANIFEST_SCHEMA_INVALID",
        "seed",
    )
    require(
        isinstance(m["revision_index"], int) and m["revision_index"] >= 0,
        "MANIFEST_SCHEMA_INVALID",
        "revision_index",
    )
    require(
        isinstance(m["admissible_candidate_count"], int)
        and m["admissible_candidate_count"] > 0,
        "MANIFEST_SCHEMA_INVALID",
        "admissible_candidate_count",
    )
    require(
        isinstance(m["selection_index"], int)
        and 0 <= m["selection_index"] < m["admissible_candidate_count"],
        "MANIFEST_SCHEMA_INVALID",
        "selection_index",
    )
    if m["campaign"] == "pilot":
        require(
            m["included_in_confirmatory_analysis"] is False,
            "PILOT_CONFIRMATORY_CONTAMINATION",
            m["trial_id"],
        )
    if m["policy"] == "random_valid":
        require_sha256(m.get("random_draw_receipt_sha256"), "random_draw_receipt_sha256")
    if m["trial_state"] == "complete":
        for key in [
            "applied_receipt_sha256",
            "revisit_observation_sha256",
            "revision_outcome_sha256",
        ]:
            require_sha256(m.get(key), key)
        require(
            isinstance(m.get("world_version_after"), str)
            and bool(m["world_version_after"]),
            "MANIFEST_SCHEMA_INVALID",
            "world_version_after",
        )


@dataclass
class Trial:
    manifest_path: Path
    manifest: dict[str, Any]
    evidence_index: dict[str, Any]
    candidate_set: dict[str, Any]


def file_from_index(trial_dir: Path, idx: dict[str, Any], logical_name: str) -> Path:
    files = idx.get("files")
    require(
        isinstance(files, dict),
        "INCOMPLETE_EVIDENCE_CLOSURE",
        f"{trial_dir}: no files map",
    )
    rel = files.get(logical_name)
    require(
        isinstance(rel, str)
        and rel
        and not rel.startswith("/")
        and ".." not in Path(rel).parts,
        "INCOMPLETE_EVIDENCE_CLOSURE",
        f"{trial_dir}: missing/unsafe {logical_name}",
    )
    return trial_dir / rel


def verify_file_digest(
    trial_dir: Path, idx: dict[str, Any], logical_name: str, expected: str
) -> Path:
    path = file_from_index(trial_dir, idx, logical_name)
    actual = sha256_file(path)
    require(
        actual == expected,
        "EVIDENCE_DIGEST_MISMATCH",
        f"{logical_name}: {actual} != {expected}",
    )
    return path


def load_trial(manifest_path: Path, analysis_sha: str, metric_sha: str) -> Trial:
    m = read_json(manifest_path)
    require(isinstance(m, dict), "MANIFEST_SCHEMA_INVALID", str(manifest_path))
    validate_manifest_shape(m)
    require(
        m["analysis_contract_sha256"] == analysis_sha,
        "ANALYSIS_CONTRACT_MISMATCH",
        m["trial_id"],
    )
    require(
        m["metric_definition_set_sha256"] == metric_sha,
        "ANALYSIS_CONTRACT_MISMATCH",
        f"metric set {m['trial_id']}",
    )

    trial_dir = manifest_path.parent
    idx_path = trial_dir / "evidence_index.json"
    idx = read_json(idx_path)
    require(
        isinstance(idx, dict),
        "INCOMPLETE_EVIDENCE_CLOSURE",
        f"{m['trial_id']}: evidence_index",
    )
    require(
        idx.get("trial_id") == m["trial_id"],
        "INCOMPLETE_EVIDENCE_CLOSURE",
        f"{m['trial_id']}: index trial id",
    )
    require(
        sha256_file(idx_path) == m["evidence_bundle_sha256"],
        "EVIDENCE_DIGEST_MISMATCH",
        f"{m['trial_id']}: evidence index",
    )

    verify_file_digest(
        trial_dir, idx, "experience_episode", m["experience_episode_sha256"]
    )
    hyp_path = verify_file_digest(
        trial_dir, idx, "revision_hypothesis", m["revision_hypothesis_sha256"]
    )
    cset_path = verify_file_digest(
        trial_dir, idx, "candidate_set", m["candidate_set_sha256"]
    )
    verify_file_digest(
        trial_dir, idx, "selected_proposal", m["selected_proposal_sha256"]
    )

    candidate_set = read_json(cset_path)
    require(
        isinstance(candidate_set, dict),
        "INCOMPLETE_EVIDENCE_CLOSURE",
        f"{m['trial_id']}: candidate_set",
    )
    candidates = candidate_set.get("candidates")
    require(
        isinstance(candidates, list),
        "INCOMPLETE_EVIDENCE_CLOSURE",
        f"{m['trial_id']}: candidates",
    )
    admitted = [
        c
        for c in candidates
        if isinstance(c, dict) and c.get("physically_admitted") is True
    ]
    require(
        len(admitted) == m["admissible_candidate_count"],
        "INCOMPLETE_EVIDENCE_CLOSURE",
        f"{m['trial_id']}: admitted count",
    )
    require(
        0 <= m["selection_index"] < len(admitted),
        "SELECTION_NOT_PHYSICALLY_ADMITTED",
        m["trial_id"],
    )
    selected = admitted[m["selection_index"]]
    require(
        selected.get("proposal_sha256") == m["selected_proposal_sha256"],
        "SELECTION_NOT_PHYSICALLY_ADMITTED",
        m["trial_id"],
    )

    hyp = read_json(hyp_path)
    require(
        hyp.get("prospective") is True,
        "POST_HOC_HYPOTHESIS_MUTATION",
        m["trial_id"],
    )

    ts = idx.get("timestamps_ns", {})
    if m["trial_state"] == "complete":
        for key in [
            "hypothesis_closed",
            "selection_closed",
            "applied_receipt",
            "revisit_closed",
            "outcome_closed",
        ]:
            require(
                isinstance(ts.get(key), int),
                "INCOMPLETE_EVIDENCE_CLOSURE",
                f"{m['trial_id']}: timestamp {key}",
            )
        require(
            ts["hypothesis_closed"]
            <= ts["selection_closed"]
            < ts["applied_receipt"]
            <= ts["revisit_closed"]
            <= ts["outcome_closed"],
            "POST_HOC_HYPOTHESIS_MUTATION",
            m["trial_id"],
        )

        receipt_path = verify_file_digest(
            trial_dir, idx, "applied_receipt", m["applied_receipt_sha256"]
        )
        revisit_path = verify_file_digest(
            trial_dir,
            idx,
            "revisit_observation",
            m["revisit_observation_sha256"],
        )
        verify_file_digest(
            trial_dir, idx, "revision_outcome", m["revision_outcome_sha256"]
        )
        receipt = read_json(receipt_path)
        require(
            receipt.get("selected_proposal_sha256") == m["selected_proposal_sha256"],
            "SELECTED_PROPOSAL_RECEIPT_MISMATCH",
            m["trial_id"],
        )
        require(
            receipt.get("world_version_before") == m["world_version_before"]
            and receipt.get("world_version_after") == m["world_version_after"],
            "WORLD_VERSION_CHAIN_MISMATCH",
            m["trial_id"],
        )
        revisit = read_json(revisit_path)
        require(
            revisit.get("world_version") == m["world_version_after"],
            "WORLD_VERSION_CHAIN_MISMATCH",
            m["trial_id"],
        )
        require(
            revisit.get("provenance_domain")
            in {"digital_committed", "physical_grounded"},
            "PROVENANCE_DOMAIN_SUBSTITUTION",
            m["trial_id"],
        )

    if m["policy"] == "random_valid":
        draw_path = verify_file_digest(
            trial_dir,
            idx,
            "random_draw_receipt",
            m["random_draw_receipt_sha256"],
        )
        draw = read_json(draw_path)
        require(
            draw.get("algorithm") == "sha256-counter-v1",
            "RANDOM_VALID_DRAW_MISMATCH",
            m["trial_id"],
        )
        require(
            draw.get("candidate_set_sha256") == m["candidate_set_sha256"],
            "RANDOM_VALID_DRAW_MISMATCH",
            m["trial_id"],
        )
        expected_index, expected_counter, expected_digest = sha256_counter_draw(
            m["seed"],
            m["paired_block_id"],
            m["candidate_set_sha256"],
            m["admissible_candidate_count"],
        )
        require(
            draw.get("seed") == m["seed"]
            and draw.get("paired_block_id") == m["paired_block_id"]
            and draw.get("admissible_candidate_count")
            == m["admissible_candidate_count"]
            and draw.get("selected_index") == expected_index
            and draw.get("counter") == expected_counter
            and draw.get("accepted_digest_sha256") == expected_digest
            and m["selection_index"] == expected_index,
            "RANDOM_VALID_DRAW_MISMATCH",
            f"{m['trial_id']}: expected index {expected_index}",
        )

    violations = m.get("integrity_violations", []) or []
    if violations:
        require(
            m["trial_state"] == "invalid_integrity",
            "INTEGRITY_FAILURE_RECLASSIFIED",
            m["trial_id"],
        )
    if (
        m["trial_state"] == "complete"
        and not violations
        and not m["included_in_confirmatory_analysis"]
        and m["campaign"] != "pilot"
    ):
        allowed = set(idx.get("prospective_exclusion_reason_classes", []))
        reason = m.get("exclusion_reason")
        require(
            reason in allowed,
            "INVALID_EXCLUSION_RECLASSIFICATION",
            m["trial_id"],
        )

    return Trial(manifest_path, m, idx, candidate_set)


def verify_bundle(root: Path) -> dict[str, Any]:
    freeze_path = root / "confirmatory_freeze.json"
    analysis_path = root / "analysis_contract.json"
    metrics_path = root / "metric_definitions.json"
    inventory_path = root / "trial_inventory.json"

    freeze = read_json(freeze_path)
    read_json(analysis_path)
    read_json(metrics_path)
    inventory = read_json(inventory_path)

    analysis_sha = sha256_file(analysis_path)
    metric_sha = sha256_file(metrics_path)
    require(
        freeze.get("analysis_contract_sha256") == analysis_sha,
        "POST_FREEZE_CONTRACT_MUTATION",
        "analysis contract",
    )
    require(
        freeze.get("metric_definition_set_sha256") == metric_sha,
        "POST_FREEZE_CONTRACT_MUTATION",
        "metric definitions",
    )
    require(
        freeze.get("trial_inventory_sha256") == sha256_file(inventory_path),
        "POST_FREEZE_CONTRACT_MUTATION",
        "trial inventory",
    )

    forbidden = set(freeze.get("forbidden_primary_aggregates", [])) | FORBIDDEN_AGGREGATES
    for key in walk_keys(read_json(root / "primary_results.json")):
        require(key not in forbidden, "FORBIDDEN_AGGREGATE", key)

    expected_ids = inventory.get("trial_ids")
    require(
        isinstance(expected_ids, list) and all(isinstance(x, str) for x in expected_ids),
        "PREREGISTRATION_INVALID",
        "trial_ids",
    )
    require(
        len(set(expected_ids)) == len(expected_ids),
        "DUPLICATE_TRIAL_IDENTITY",
        "inventory duplicates",
    )

    manifests = sorted((root / "trials").glob("*/manifest.json"))
    trials = [load_trial(path, analysis_sha, metric_sha) for path in manifests]
    actual_ids = [t.manifest["trial_id"] for t in trials]
    require(
        len(set(actual_ids)) == len(actual_ids),
        "DUPLICATE_TRIAL_IDENTITY",
        "manifest duplicates",
    )
    missing = sorted(set(expected_ids) - set(actual_ids))
    extra = sorted(set(actual_ids) - set(expected_ids))
    require(not missing, "PREREGISTERED_TRIAL_MISSING", ",".join(missing))
    require(not extra, "UNPREREGISTERED_TRIAL_PRESENT", ",".join(extra))

    by_block: dict[str, list[Trial]] = {}
    for trial in trials:
        by_block.setdefault(trial.manifest["paired_block_id"], []).append(trial)
    for block, block_trials in by_block.items():
        primary = [
            trial
            for trial in block_trials
            if trial.manifest["policy"] in PAIR_POLICIES
        ]
        if len(primary) > 1:
            require(
                len({t.manifest["candidate_set_sha256"] for t in primary}) == 1,
                "PAIRED_CANDIDATE_SET_MISMATCH",
                block,
            )
            require(
                len({t.manifest["world_fixture_sha256"] for t in primary}) == 1
                and len({t.manifest["seed"] for t in primary}) == 1
                and len({t.manifest["revision_index"] for t in primary}) == 1,
                "PAIRED_BLOCK_IDENTITY_MISMATCH",
                block,
            )

    chains: dict[tuple[str, str, int], list[Trial]] = {}
    for trial in trials:
        key = (
            trial.manifest["policy"],
            trial.manifest["world_fixture_sha256"],
            trial.manifest["seed"],
        )
        chains.setdefault(key, []).append(trial)
    for key, chain in chains.items():
        complete = sorted(
            (t for t in chain if t.manifest["trial_state"] == "complete"),
            key=lambda t: t.manifest["revision_index"],
        )
        for previous, current in zip(complete, complete[1:]):
            if (
                current.manifest["revision_index"]
                == previous.manifest["revision_index"] + 1
            ):
                require(
                    previous.manifest["world_version_after"]
                    == current.manifest["world_version_before"],
                    "WORLD_VERSION_CHAIN_MISMATCH",
                    str(key),
                )

    for block_trials in by_block.values():
        for trial in block_trials:
            require(
                trial.evidence_index.get(
                    "cross_policy_outcome_observed_before_selection", False
                )
                is False,
                "POLICY_ORDER_INFORMATION_LEAK",
                trial.manifest["trial_id"],
            )

    for trial in trials:
        if trial.manifest["campaign"] == "confirmatory_generalization":
            disclosed = trial.evidence_index.get("fixture_disclosed_to_policy_at_ns")
            reveal = trial.evidence_index.get("generalization_fixture_reveal_at_ns")
            require(
                isinstance(disclosed, int)
                and isinstance(reveal, int)
                and disclosed >= reveal,
                "GENERALIZATION_FIXTURE_LEAK",
                trial.manifest["trial_id"],
            )

    expected_count = inventory.get("expected_trial_count", len(expected_ids))
    require(
        expected_count == len(expected_ids),
        "PREREGISTRATION_INVALID",
        "expected_trial_count",
    )
    require(
        len(trials) == expected_count,
        "UNAUTHORIZED_EARLY_STOP",
        f"{len(trials)} != {expected_count}",
    )

    return {
        "verdict": "ACCEPT",
        "experiment_id": "VART-WORLD-CREATIVE-001",
        "trial_count": len(trials),
        "paired_block_count": len(by_block),
        "analysis_contract_sha256": analysis_sha,
        "metric_definition_set_sha256": metric_sha,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Independent VART-WORLD-CREATIVE-001 evidence verifier"
    )
    parser.add_argument("root", type=Path, help="evidence package root")
    parser.add_argument("--json", action="store_true", help="emit machine-readable verdict")
    args = parser.parse_args()
    try:
        result = verify_bundle(args.root)
    except Reject as exc:
        result = {"verdict": "REJECT", "reason_class": exc.code, "detail": exc.detail}
        if args.json:
            print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        else:
            print(f"REJECT {exc.code}: {exc.detail}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    else:
        print(
            f"ACCEPT: {result['trial_count']} trials, "
            f"{result['paired_block_count']} paired blocks"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
