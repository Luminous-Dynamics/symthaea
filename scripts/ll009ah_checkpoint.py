#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from capture_ll009ag_environment import capture
from ll009ag_common import CE, E, M, P, PRE, cb, closed, file, hb, hf, lj, receipt, req, vr, vp, wj
from ll009ag_campaign import _verify_dependency_bindings, abind, prepare, verify_pre, vm

LOCK = "ll009ah.pre-evidence-lineage-lock.v1"
GUARD = "ll009ah.stage-guard.v1"
CHECKPOINT = "ll009ah.stage-checkpoint.v1"
STAGE_MAP = "ll009ah.artifact-stage-map.v1"


def expect(label: str, fn) -> None:
    try:
        fn()
    except CE:
        return
    raise CE(f"expected failure did not occur: {label}")


def native_receipt(v: dict[str, Any]) -> dict[str, Any]:
    if "receipt_sha256" in v:
        raise CE("native receipt payload already hashed")
    out = dict(v)
    out["receipt_sha256"] = hb((json.dumps(v, sort_keys=True, indent=2, separators=(",", ": ")) + "\n").encode())
    return out


def _git(repo: Path, *args: str) -> str:
    exe = shutil.which("git")
    if not exe:
        raise CE("git missing from AH qualification environment")
    try:
        cp = subprocess.run(
            [exe, "-C", str(repo), *args],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise CE(f"synthetic git command failed {args!r}: {e.stderr.strip()}") from e
    return cp.stdout.strip()


def _map(v: Any, policy: dict[str, Any], artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(v, dict) or v.get("schema_version") != STAGE_MAP or v.get("study_id") != policy["study_id"]:
        raise CE("invalid AH stage map identity")
    stages = v.get("stages")
    if not isinstance(stages, list) or len(stages) != len(policy["stage_plan"]):
        raise CE("AH stage map must reproduce the exact AG stage count")
    assigned: list[str] = []
    out: list[dict[str, Any]] = []
    for index, (got, want) in enumerate(zip(stages, policy["stage_plan"])):
        if not isinstance(got, dict):
            raise CE(f"AH stage {index} must be object")
        sid = req(got.get("id"), f"AH stage {index} id")
        profile = req(got.get("environment_profile"), f"AH stage {sid} profile")
        network = got.get("may_access_network")
        ids = got.get("artifact_ids")
        if sid != want["id"] or profile != want["environment_profile"] or network is not want["may_access_network"]:
            raise CE(f"AH stage {index} does not exactly reproduce AG stage policy")
        if not isinstance(ids, list) or len(ids) != len(set(ids)):
            raise CE(f"AH stage {sid} artifact_ids must be unique list")
        ids = [req(x, f"AH stage {sid} artifact id") for x in ids]
        assigned.extend(ids)
        out.append(
            {
                "index": index,
                "id": sid,
                "environment_profile": profile,
                "may_access_network": network,
                "artifact_ids": ids,
            }
        )
    if len(assigned) != len(set(assigned)):
        raise CE("AH artifact assigned to more than one stage")
    if set(assigned) != set(artifacts):
        raise CE(
            f"AH stage map must assign every manifest artifact exactly once; "
            f"missing={sorted(set(artifacts)-set(assigned))} extra={sorted(set(assigned)-set(artifacts))}"
        )
    return out


def _runtime_witness(policy: dict[str, Any], pre: dict[str, Any], runtime_root: Path, profile: str) -> dict[str, Any]:
    entries = [x for x in pre.get("environment_capsules", []) if isinstance(x, dict) and x.get("profile") == profile]
    if len(entries) != 1:
        raise CE(f"PREPARED must bind exactly one {profile} environment capsule")
    entry = entries[0]
    capsule = lj(file(runtime_root, req(entry.get("path"), f"{profile} capsule path"), f"{profile} capsule"))
    actual = capture(policy, profile)
    if cb(actual) != cb(capsule):
        raise CE(f"actual runtime does not equal PREPARED {profile} environment capsule")
    return {
        "profile": profile,
        "prepared_capsule_file_sha256": req(entry.get("sha256"), f"{profile} prepared capsule SHA"),
        "actual_canonical_payload_sha256": hb(cb(actual)),
    }


def _manifest_binding(path: Path, value: Any) -> dict[str, Any]:
    return {"sha256": hf(path), "byte_count": path.stat().st_size, "canonical_payload_sha256": hb(cb(value))}


def create_lock(
    pre: dict[str, Any],
    policy_path: Path,
    repo: Path,
    head: str,
    runtime_root: Path,
    manifest_path: Path,
    stage_map_path: Path,
    evidence_root: Path,
) -> dict[str, Any]:
    policy = verify_pre(pre, policy_path, repo, head, runtime_root)
    mv = lj(manifest_path)
    artifacts = vm(mv, policy["study_id"])
    sv = lj(stage_map_path)
    stages = _map(sv, policy, artifacts)
    if closed(evidence_root):
        raise CE("AH pre-evidence lock requires an empty evidence root")
    return receipt(
        {
            "schema_version": LOCK,
            "state": "FROZEN_PRE_EVIDENCE",
            "study_id": policy["study_id"],
            "ag_preparation_receipt_sha256": pre["receipt_sha256"],
            "manifest": _manifest_binding(manifest_path, mv),
            "stage_map": _manifest_binding(stage_map_path, sv),
            "stage_plan_sha256": hb(cb(stages)),
            "stage_count": len(stages),
            "initial_evidence_file_set": [],
            "next_stage_index": 0,
            "continuous_runtime_mediation_performed": False,
            "theorem": "lineage_and_artifact_plan_frozen_before_first_scientific_output",
        }
    )


def verify_lock(
    lock: dict[str, Any],
    pre: dict[str, Any],
    policy_path: Path,
    repo: Path,
    head: str,
    runtime_root: Path,
    manifest_path: Path,
    stage_map_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    vr(lock, LOCK)
    vr(pre, PRE)
    policy = verify_pre(pre, policy_path, repo, head, runtime_root)
    if lock.get("state") != "FROZEN_PRE_EVIDENCE" or lock.get("study_id") != policy["study_id"]:
        raise CE("AH lock state/study mismatch")
    if lock.get("ag_preparation_receipt_sha256") != pre["receipt_sha256"]:
        raise CE("AH lock does not bind exact AG PREPARED receipt")
    mv = lj(manifest_path)
    artifacts = vm(mv, policy["study_id"])
    sv = lj(stage_map_path)
    stages = _map(sv, policy, artifacts)
    if lock.get("manifest") != _manifest_binding(manifest_path, mv):
        raise CE("AH manifest drift after pre-evidence lock")
    if lock.get("stage_map") != _manifest_binding(stage_map_path, sv):
        raise CE("AH stage-map drift after pre-evidence lock")
    if lock.get("stage_plan_sha256") != hb(cb(stages)) or lock.get("stage_count") != len(stages):
        raise CE("AH stage-plan drift after pre-evidence lock")
    if lock.get("initial_evidence_file_set") != [] or lock.get("next_stage_index") != 0:
        raise CE("AH lock initial state drift")
    return policy, artifacts, stages


def _partial_bind(artifacts: dict[str, dict[str, Any]], evidence_root: Path) -> tuple[dict[str, dict[str, Any]], set[str]]:
    actual = closed(evidence_root)
    declared = {x["path"]: i for i, x in artifacts.items() if x["path"] is not None}
    extra = actual - set(declared)
    if extra:
        raise CE(f"AH evidence root contains undeclared files: {sorted(extra)}")
    bound: dict[str, dict[str, Any]] = {}
    for i, x in artifacts.items():
        if x["requirement"] == "not_yet_available":
            bound[i] = {
                "id": i,
                "requirement": "not_yet_available",
                "availability": "not_yet_available",
                "dependencies": list(x["dependencies"]),
            }
        elif x["path"] in actual:
            bound[i] = abind(x, evidence_root)
        elif x["requirement"] == "required":
            bound[i] = {
                "id": i,
                "path": x["path"],
                "requirement": "required",
                "availability": "pending_required",
                "dependencies": list(x["dependencies"]),
                "self_hash_mode": x["self_hash_mode"],
            }
        else:
            bound[i] = {
                "id": i,
                "path": x["path"],
                "requirement": "optional_diagnostic",
                "availability": "absent_optional_diagnostic",
                "dependencies": list(x["dependencies"]),
                "self_hash_mode": x["self_hash_mode"],
            }
    _verify_dependency_bindings(artifacts, bound, evidence_root)
    return bound, actual


def _present(bound: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {i: x for i, x in bound.items() if x.get("availability") == "present"}


def _previous(lock: dict[str, Any], previous: dict[str, Any] | None) -> tuple[int, dict[str, dict[str, Any]], str | None]:
    if previous is None:
        return -1, {}, None
    vr(previous, CHECKPOINT)
    if previous.get("lock_receipt_sha256") != lock["receipt_sha256"]:
        raise CE("AH predecessor checkpoint belongs to different lock")
    index = previous.get("stage_index")
    if not isinstance(index, int) or index < 0:
        raise CE("AH predecessor stage index invalid")
    items = previous.get("artifact_bindings")
    if not isinstance(items, list):
        raise CE("AH predecessor artifact bindings missing")
    by_id = {x.get("id"): x for x in items if isinstance(x, dict)}
    if len(by_id) != len(items) or None in by_id:
        raise CE("AH predecessor artifact bindings malformed")
    return index, _present(by_id), previous["receipt_sha256"]


def create_guard(
    lock: dict[str, Any],
    previous: dict[str, Any] | None,
    pre: dict[str, Any],
    policy_path: Path,
    repo: Path,
    head: str,
    runtime_root: Path,
    manifest_path: Path,
    stage_map_path: Path,
    evidence_root: Path,
    stage_id: str,
) -> dict[str, Any]:
    policy, artifacts, stages = verify_lock(lock, pre, policy_path, repo, head, runtime_root, manifest_path, stage_map_path)
    prev_index, prev_present, prev_sha = _previous(lock, previous)
    next_index = prev_index + 1
    if next_index >= len(stages):
        raise CE("AH checkpoint chain already completed all stages")
    stage = stages[next_index]
    if stage_id != stage["id"]:
        raise CE(f"AH stage order violation: expected {stage['id']}, got {stage_id}")
    bound, actual = _partial_bind(artifacts, evidence_root)
    current_present = _present(bound)
    if current_present != prev_present:
        raise CE("AH guard observed uncheckpointed evidence growth/removal/mutation before stage")
    witness = _runtime_witness(policy, pre, runtime_root, stage["environment_profile"])
    return receipt(
        {
            "schema_version": GUARD,
            "state": "STAGE_GUARDED",
            "study_id": policy["study_id"],
            "lock_receipt_sha256": lock["receipt_sha256"],
            "predecessor_checkpoint_receipt_sha256": prev_sha,
            "stage_index": next_index,
            "stage_id": stage["id"],
            "environment_profile": stage["environment_profile"],
            "may_access_network": stage["may_access_network"],
            "runtime_witness": witness,
            "pre_stage_exact_file_set": sorted(actual),
            "pre_stage_present_artifact_ids": sorted(current_present),
            "continuous_runtime_mediation_performed": False,
            "boundary_semantics": "guard_proves_pre_stage_boundary_only",
        }
    )


def create_checkpoint(
    guard: dict[str, Any],
    lock: dict[str, Any],
    previous: dict[str, Any] | None,
    pre: dict[str, Any],
    policy_path: Path,
    repo: Path,
    head: str,
    runtime_root: Path,
    manifest_path: Path,
    stage_map_path: Path,
    evidence_root: Path,
) -> dict[str, Any]:
    vr(guard, GUARD)
    policy, artifacts, stages = verify_lock(lock, pre, policy_path, repo, head, runtime_root, manifest_path, stage_map_path)
    prev_index, prev_present, prev_sha = _previous(lock, previous)
    stage_index = prev_index + 1
    if stage_index >= len(stages):
        raise CE("AH checkpoint would exceed declared stage count")
    stage = stages[stage_index]
    if (
        guard.get("lock_receipt_sha256") != lock["receipt_sha256"]
        or guard.get("predecessor_checkpoint_receipt_sha256") != prev_sha
        or guard.get("stage_index") != stage_index
        or guard.get("stage_id") != stage["id"]
    ):
        raise CE("AH guard/predecessor/stage mismatch")
    witness = _runtime_witness(policy, pre, runtime_root, stage["environment_profile"])
    if guard.get("runtime_witness") != witness:
        raise CE("AH runtime profile changed between guard and checkpoint")
    bound, actual = _partial_bind(artifacts, evidence_root)
    current_present = _present(bound)
    for i, old in prev_present.items():
        if current_present.get(i) != old:
            raise CE(f"AH previously checkpointed artifact changed or disappeared: {i}")
    newly_present = set(current_present) - set(prev_present)
    assigned = set(stage["artifact_ids"])
    if newly_present - assigned:
        raise CE(f"AH stage produced artifact assigned to another stage: {sorted(newly_present-assigned)}")
    if set(prev_present) & assigned:
        raise CE(f"AH stage artifact was already present before its stage: {sorted(set(prev_present)&assigned)}")
    required_here = {i for i in assigned if artifacts[i]["requirement"] == "required"}
    missing_required = required_here - set(current_present)
    if missing_required:
        raise CE(f"AH stage checkpoint missing required outputs: {sorted(missing_required)}")
    not_yet_here = {i for i in assigned if artifacts[i]["requirement"] == "not_yet_available"}
    if not_yet_here & set(current_present):
        raise CE(f"AH not-yet-available artifact materialized: {sorted(not_yet_here & set(current_present))}")
    optional_absent = sorted(
        i for i in assigned if artifacts[i]["requirement"] == "optional_diagnostic" and i not in current_present
    )
    all_done = stage_index + 1 == len(stages)
    return receipt(
        {
            "schema_version": CHECKPOINT,
            "state": "FROZEN_CHECKPOINT",
            "study_id": policy["study_id"],
            "lock_receipt_sha256": lock["receipt_sha256"],
            "predecessor_checkpoint_receipt_sha256": prev_sha,
            "guard_receipt_sha256": guard["receipt_sha256"],
            "stage_index": stage_index,
            "stage_id": stage["id"],
            "environment_profile": stage["environment_profile"],
            "may_access_network": stage["may_access_network"],
            "runtime_witness": witness,
            "newly_present_artifact_ids": sorted(newly_present),
            "optional_diagnostic_absent_for_stage": optional_absent,
            "artifact_bindings": [bound[i] for i in sorted(bound)],
            "exact_file_set": sorted(actual),
            "all_declared_stages_checkpointed": all_done,
            "next_stage_id": None if all_done else stages[stage_index + 1]["id"],
            "continuous_runtime_mediation_performed": False,
            "boundary_semantics": "checkpoint_proves_post_stage_boundary_and_append_only_evidence_not_continuous_process_mediation",
        }
    )


def selftest() -> None:
    old_env = os.environ.get("LL009AH_TEST")
    os.environ["LL009AH_TEST"] = "baseline"
    try:
        with tempfile.TemporaryDirectory(prefix="ll009ah-") as td:
            root = Path(td)
            repo = root / "repo"
            rt = root / "runtime"
            er = root / "evidence"
            repo.mkdir(); rt.mkdir(); er.mkdir()
            policy = {
                "schema_version": P,
                "study_id": "synthetic-ah",
                "required_environment_profiles": ["acquisition", "gis", "analysis"],
                "environment_contracts": {
                    k: {"required_packages": [], "required_libraries": [], "required_environment_variables": ["LL009AH_TEST"]}
                    for k in ["acquisition", "gis", "analysis"]
                },
                "protected_files": ["tool.py"],
                "stage_plan": [
                    {"id": "acquire", "environment_profile": "acquisition", "may_access_network": True},
                    {"id": "analyze", "environment_profile": "analysis", "may_access_network": False},
                ],
                "network_authorized_stage_ids": ["acquire"],
                "classification_order": ["descriptive_geometry"],
                "classification_ceiling": "descriptive_geometry",
                "semantic_rules": {"descriptive_geometry": {"enabled": True, "requires_all_artifact_ids": ["z"]}},
            }
            pp = repo / "policy.json"; wj(pp, policy)
            (repo / "tool.py").write_text("fixed\n")
            _git(repo, "init", "-q"); _git(repo, "config", "user.email", "ll009ah@example.invalid"); _git(repo, "config", "user.name", "LL-009AH Synthetic")
            _git(repo, "add", "policy.json", "tool.py"); _git(repo, "commit", "-qm", "synthetic AH subject")
            head = _git(repo, "rev-parse", "HEAD")
            env_map: dict[str, str] = {}
            for profile in ["acquisition", "gis", "analysis"]:
                env_map[profile] = profile + ".json"
                wj(rt / env_map[profile], capture(vp(policy), profile))
            pre = prepare(pp, repo, head, rt, env_map)

            q = native_receipt({"schema_version": "q.v1", "value": 1})
            z_template = {"schema_version": "z.v1", "upstream": q["receipt_sha256"]}
            manifest = {
                "schema_version": M,
                "study_id": "synthetic-ah",
                "artifacts": [
                    {"id": "q", "path": "q.json", "requirement": "required", "dependencies": [], "dependency_bindings": [], "self_hash_mode": "ll009_indent2_receipt_sha256"},
                    {"id": "z", "path": "z.json", "requirement": "required", "dependencies": ["q"], "dependency_bindings": [{"dependency_id": "q", "field_path": ["upstream"], "identity": "receipt_sha256"}], "self_hash_mode": "ll009_indent2_receipt_sha256"},
                ],
            }
            mp = root / "manifest.json"; wj(mp, manifest)
            stage_map = {
                "schema_version": STAGE_MAP,
                "study_id": "synthetic-ah",
                "stages": [
                    {"id": "acquire", "environment_profile": "acquisition", "may_access_network": True, "artifact_ids": ["q"]},
                    {"id": "analyze", "environment_profile": "analysis", "may_access_network": False, "artifact_ids": ["z"]},
                ],
            }
            smp = root / "stage-map.json"; wj(smp, stage_map)

            lock = create_lock(pre, pp, repo, head, rt, mp, smp, er)
            if lock != create_lock(pre, pp, repo, head, rt, mp, smp, er):
                raise CE("AH lock not deterministic")
            (er / "q.json").write_text("{}\n")
            expect("nonempty pre-evidence root", lambda: create_lock(pre, pp, repo, head, rt, mp, smp, er))
            (er / "q.json").unlink()

            expect("stage skip", lambda: create_guard(lock, None, pre, pp, repo, head, rt, mp, smp, er, "analyze"))
            os.environ["LL009AH_TEST"] = "drift"
            expect("actual runtime drift", lambda: create_guard(lock, None, pre, pp, repo, head, rt, mp, smp, er, "acquire"))
            os.environ["LL009AH_TEST"] = "baseline"
            g0 = create_guard(lock, None, pre, pp, repo, head, rt, mp, smp, er, "acquire")
            expect("required output missing", lambda: create_checkpoint(g0, lock, None, pre, pp, repo, head, rt, mp, smp, er))
            wj(er / "q.json", q)
            c0 = create_checkpoint(g0, lock, None, pre, pp, repo, head, rt, mp, smp, er)
            if c0["newly_present_artifact_ids"] != ["q"]:
                raise CE("AH stage-0 checkpoint population drift")

            g1 = create_guard(lock, c0, pre, pp, repo, head, rt, mp, smp, er, "analyze")
            old_q = (er / "q.json").read_bytes(); wj(er / "q.json", native_receipt({"schema_version": "q.v1", "value": 2}))
            expect("prior artifact mutation", lambda: create_checkpoint(g1, lock, c0, pre, pp, repo, head, rt, mp, smp, er))
            (er / "q.json").write_bytes(old_q)
            bad_z = native_receipt({"schema_version": "z.v1", "upstream": "2" * 64}); wj(er / "z.json", bad_z)
            expect("wrong parent digest", lambda: create_checkpoint(g1, lock, c0, pre, pp, repo, head, rt, mp, smp, er))
            wj(er / "z.json", native_receipt(z_template))
            c1 = create_checkpoint(g1, lock, c0, pre, pp, repo, head, rt, mp, smp, er)
            if not c1["all_declared_stages_checkpointed"] or c1["newly_present_artifact_ids"] != ["z"]:
                raise CE("AH final checkpoint theorem failed")
            expect("stage replay", lambda: create_guard(lock, c1, pre, pp, repo, head, rt, mp, smp, er, "analyze"))

            (er / "extra.json").write_text("{}\n")
            expect("undeclared evidence", lambda: _partial_bind(vm(manifest, "synthetic-ah"), er))
            (er / "extra.json").unlink()

            old_tool = (repo / "tool.py").read_bytes(); (repo / "tool.py").write_text("drift\n")
            expect("protected code drift after lock", lambda: create_guard(lock, c0, pre, pp, repo, head, rt, mp, smp, er, "analyze"))
            (repo / "tool.py").write_bytes(old_tool)

            print("LL-009AH self-test PASS: pre-evidence lock, runtime witness, stage order, append-only checkpoints, parent bindings, and drift failures")
    finally:
        if old_env is None:
            os.environ.pop("LL009AH_TEST", None)
        else:
            os.environ["LL009AH_TEST"] = old_env


def _rr(path: Path, schema: str) -> dict[str, Any]:
    v = lj(path); vr(v, schema); return v


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--repo-head", required=True)
    p.add_argument("--runtime-root", type=Path, required=True)
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--stage-map", type=Path, required=True)
    p.add_argument("--evidence-root", type=Path, required=True)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LL-009AH pre-evidence lineage lock and monotonic stage checkpoints")
    sp = p.add_subparsers(dest="cmd", required=True)
    x = sp.add_parser("lock"); _common(x); x.add_argument("--output", type=Path, required=True)
    x = sp.add_parser("guard"); _common(x); x.add_argument("--lock", type=Path, required=True); x.add_argument("--previous-checkpoint", type=Path); x.add_argument("--stage-id", required=True); x.add_argument("--output", type=Path, required=True)
    x = sp.add_parser("checkpoint"); _common(x); x.add_argument("--lock", type=Path, required=True); x.add_argument("--guard", type=Path, required=True); x.add_argument("--previous-checkpoint", type=Path); x.add_argument("--output", type=Path, required=True)
    sp.add_parser("self-test")
    return p


def main() -> int:
    a = parser().parse_args()
    try:
        if a.cmd == "self-test":
            selftest(); return 0
        pre = _rr(a.preparation, PRE)
        if a.cmd == "lock":
            out = create_lock(pre, a.policy, a.repo_root, a.repo_head, a.runtime_root, a.manifest, a.stage_map, a.evidence_root)
            wj(a.output, out); print("AH LOCKED", out["receipt_sha256"]); return 0
        lock = _rr(a.lock, LOCK)
        previous = _rr(a.previous_checkpoint, CHECKPOINT) if a.previous_checkpoint else None
        if a.cmd == "guard":
            out = create_guard(lock, previous, pre, a.policy, a.repo_root, a.repo_head, a.runtime_root, a.manifest, a.stage_map, a.evidence_root, a.stage_id)
            wj(a.output, out); print("AH GUARDED", out["stage_id"], out["receipt_sha256"]); return 0
        guard = _rr(a.guard, GUARD)
        out = create_checkpoint(guard, lock, previous, pre, a.policy, a.repo_root, a.repo_head, a.runtime_root, a.manifest, a.stage_map, a.evidence_root)
        wj(a.output, out); print("AH CHECKPOINT", out["stage_id"], out["receipt_sha256"]); return 0
    except CE as e:
        print("LL-009AH FAIL:", e, file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
