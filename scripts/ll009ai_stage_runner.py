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
from ll009ag_common import CE, M, P, PRE, cb, closed, file, hb, hf, lj, receipt, rel, req, vr, vp, wj
from ll009ag_campaign import prepare
from ll009ah_checkpoint import (
    CHECKPOINT,
    GUARD,
    LOCK as AH_LOCK,
    STAGE_MAP,
    _git,
    create_checkpoint,
    create_guard,
    create_lock as create_ah_lock,
    native_receipt,
    verify_lock as verify_ah_lock,
)

PLAN = "ll009ai.command-plan.v1"
AI_LOCK = "ll009ai.execution-plan-lock.v1"
EXECUTION = "ll009ai.mediated-stage-execution.v1"


def _binding(path: Path, value: Any) -> dict[str, Any]:
    return {"sha256": hf(path), "byte_count": path.stat().st_size, "canonical_payload_sha256": hb(cb(value))}


def write_once(path: Path, value: dict[str, Any]) -> None:
    data = cb(value)
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise CE(f"AI output is not an ordinary file: {path}")
        if path.read_bytes() != data:
            raise CE(f"AI refuses to overwrite differing receipt: {path}")
        return
    wj(path, value)


def _safe_dir(repo: Path, raw: Any, label: str) -> Path:
    r = rel(raw, label)
    root = repo.resolve(strict=True)
    p = root
    for part in r.split("/"):
        p = p / part
        if p.is_symlink():
            raise CE(f"{label} traverses symlink: {p}")
    try:
        q = p.resolve(strict=True)
    except FileNotFoundError as e:
        raise CE(f"{label} missing: {p}") from e
    try:
        q.relative_to(root)
    except ValueError as e:
        raise CE(f"{label} escapes repository root") from e
    if not q.is_dir():
        raise CE(f"{label} is not directory: {q}")
    return q


def _plan(v: Any, policy: dict[str, Any], stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(v, dict) or v.get("schema_version") != PLAN or v.get("study_id") != policy["study_id"]:
        raise CE("invalid AI command-plan identity")
    got_stages = v.get("stages")
    if not isinstance(got_stages, list) or len(got_stages) != len(stages):
        raise CE("AI command plan must reproduce exact AH stage count")
    out: list[dict[str, Any]] = []
    for index, (got, want) in enumerate(zip(got_stages, stages)):
        if not isinstance(got, dict):
            raise CE(f"AI command stage {index} must be object")
        sid = req(got.get("id"), f"AI stage {index} id")
        expected = got.get("expected_artifact_ids")
        mode = got.get("network_mode")
        commands = got.get("commands")
        if sid != want["id"]:
            raise CE(f"AI command stage order mismatch at {index}")
        if not isinstance(expected, list) or len(expected) != len(set(expected)):
            raise CE(f"AI stage {sid} expected_artifact_ids must be unique list")
        expected = [req(x, f"AI stage {sid} expected artifact") for x in expected]
        if set(expected) != set(want["artifact_ids"]):
            raise CE(f"AI stage {sid} artifact population differs from AH stage map")
        required_mode = "authorized_network" if want["may_access_network"] else "declared_offline_only"
        if mode != required_mode:
            raise CE(f"AI stage {sid} network mode must be {required_mode}")
        if not isinstance(commands, list) or not commands:
            raise CE(f"AI stage {sid} must declare at least one command")
        normalized = []
        for command_index, command in enumerate(commands):
            if not isinstance(command, dict):
                raise CE(f"AI stage {sid} command {command_index} must be object")
            if command.get("runner") != "prepared_python":
                raise CE(f"AI stage {sid} command {command_index} runner must be prepared_python")
            argv = command.get("argv")
            if not isinstance(argv, list) or not argv:
                raise CE(f"AI stage {sid} command {command_index} argv must be non-empty list")
            argv = [req(x, f"AI stage {sid} command argv token") for x in argv]
            cwd = rel(command.get("cwd", "."), f"AI stage {sid} command cwd")
            normalized.append({"runner": "prepared_python", "argv": argv, "cwd": cwd})
        out.append(
            {
                "index": index,
                "id": sid,
                "expected_artifact_ids": sorted(expected),
                "network_mode": mode,
                "commands": normalized,
            }
        )
    return out


def create_execution_lock(
    ah_lock: dict[str, Any],
    pre: dict[str, Any],
    policy_path: Path,
    repo: Path,
    head: str,
    runtime_root: Path,
    manifest_path: Path,
    stage_map_path: Path,
    command_plan_path: Path,
    evidence_root: Path,
) -> dict[str, Any]:
    policy, artifacts, stages = verify_ah_lock(
        ah_lock, pre, policy_path, repo, head, runtime_root, manifest_path, stage_map_path
    )
    if closed(evidence_root):
        raise CE("AI execution-plan lock requires evidence root to remain empty")
    pv = lj(command_plan_path)
    commands = _plan(pv, policy, stages)
    return receipt(
        {
            "schema_version": AI_LOCK,
            "state": "EXECUTION_PLAN_LOCKED",
            "study_id": policy["study_id"],
            "ah_lock_receipt_sha256": ah_lock["receipt_sha256"],
            "ag_preparation_receipt_sha256": pre["receipt_sha256"],
            "command_plan": _binding(command_plan_path, pv),
            "command_plan_sha256": hb(cb(commands)),
            "stage_count": len(commands),
            "initial_evidence_file_set": [],
            "shell_execution_permitted": False,
            "network_enforcement_level": "declared_only",
            "filesystem_write_boundary_enforced": False,
        }
    )


def verify_execution_lock(
    ai_lock: dict[str, Any],
    ah_lock: dict[str, Any],
    pre: dict[str, Any],
    policy_path: Path,
    repo: Path,
    head: str,
    runtime_root: Path,
    manifest_path: Path,
    stage_map_path: Path,
    command_plan_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    vr(ai_lock, AI_LOCK)
    policy, artifacts, stages = verify_ah_lock(
        ah_lock, pre, policy_path, repo, head, runtime_root, manifest_path, stage_map_path
    )
    if ai_lock.get("ah_lock_receipt_sha256") != ah_lock["receipt_sha256"]:
        raise CE("AI lock does not bind exact AH pre-evidence lock")
    if ai_lock.get("ag_preparation_receipt_sha256") != pre["receipt_sha256"]:
        raise CE("AI lock does not bind exact AG PREPARED receipt")
    pv = lj(command_plan_path)
    commands = _plan(pv, policy, stages)
    if ai_lock.get("command_plan") != _binding(command_plan_path, pv):
        raise CE("AI command-plan drift after lock")
    if ai_lock.get("command_plan_sha256") != hb(cb(commands)) or ai_lock.get("stage_count") != len(commands):
        raise CE("AI normalized command-plan drift")
    if ai_lock.get("shell_execution_permitted") is not False:
        raise CE("AI shell-execution policy drift")
    return policy, commands


def _resolve_python() -> Path:
    p = Path(sys.executable)
    try:
        q = p.resolve(strict=True)
    except FileNotFoundError as e:
        raise CE(f"AI prepared Python executable disappeared: {p}") from e
    if not q.is_file():
        raise CE(f"AI prepared Python executable is not regular file: {q}")
    return q


def _verify_receipt_file(path: Path, value: dict[str, Any], schema: str) -> None:
    got = lj(path)
    vr(got, schema)
    if got != value:
        raise CE(f"AI receipt changed after persistence: {path}")


def _run_commands(
    stage: dict[str, Any],
    repo: Path,
    evidence_root: Path,
    guard_output: Path,
) -> list[dict[str, Any]]:
    python = _resolve_python()
    python_sha = hf(python)
    witnesses: list[dict[str, Any]] = []
    for index, command in enumerate(stage["commands"]):
        cwd = _safe_dir(repo, command["cwd"], f"AI stage {stage['id']} command cwd")
        actual_argv = [str(python), *command["argv"]]
        env = os.environ.copy()
        env["LL009AI_GUARD_PATH"] = str(guard_output.resolve())
        env["LL009AI_STAGE_ID"] = stage["id"]
        env["LL009AI_EVIDENCE_ROOT"] = str(evidence_root.resolve())
        try:
            cp = subprocess.run(
                actual_argv,
                cwd=str(cwd),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=None,
                stderr=None,
                shell=False,
                check=False,
            )
        except OSError as e:
            raise CE(f"AI could not launch stage command {index}: {e}") from e
        witness = {
            "command_index": index,
            "runner": "prepared_python",
            "python_executable_path": str(python),
            "python_executable_sha256": python_sha,
            "argv": actual_argv,
            "cwd": str(cwd),
            "returncode": cp.returncode,
            "shell_used": False,
            "network_mode": stage["network_mode"],
            "network_namespace_enforced": False,
            "filesystem_write_boundary_enforced": False,
            "control_environment": {
                "LL009AI_GUARD_PATH": str(guard_output.resolve()),
                "LL009AI_STAGE_ID": stage["id"],
                "LL009AI_EVIDENCE_ROOT": str(evidence_root.resolve()),
            },
        }
        witnesses.append(witness)
        if cp.returncode != 0:
            raise CE(f"AI stage command {index} exited non-zero: {cp.returncode}")
    return witnesses


def run_stage(
    ai_lock: dict[str, Any],
    ah_lock: dict[str, Any],
    previous: dict[str, Any] | None,
    pre: dict[str, Any],
    policy_path: Path,
    repo: Path,
    head: str,
    runtime_root: Path,
    manifest_path: Path,
    stage_map_path: Path,
    command_plan_path: Path,
    evidence_root: Path,
    stage_id: str,
    guard_output: Path,
    checkpoint_output: Path,
    execution_output: Path,
) -> dict[str, Any]:
    policy, commands = verify_execution_lock(
        ai_lock, ah_lock, pre, policy_path, repo, head, runtime_root, manifest_path, stage_map_path, command_plan_path
    )
    stage_index = 0 if previous is None else previous["stage_index"] + 1
    if stage_index >= len(commands) or commands[stage_index]["id"] != stage_id:
        expected = None if stage_index >= len(commands) else commands[stage_index]["id"]
        raise CE(f"AI stage order violation: expected {expected}, got {stage_id}")
    stage = commands[stage_index]
    guard = create_guard(
        ah_lock, previous, pre, policy_path, repo, head, runtime_root, manifest_path, stage_map_path, evidence_root, stage_id
    )
    write_once(guard_output, guard)
    _verify_receipt_file(guard_output, guard, GUARD)
    command_witnesses = _run_commands(stage, repo, evidence_root, guard_output)
    _verify_receipt_file(guard_output, guard, GUARD)
    checkpoint = create_checkpoint(
        guard, ah_lock, previous, pre, policy_path, repo, head, runtime_root, manifest_path, stage_map_path, evidence_root
    )
    write_once(checkpoint_output, checkpoint)
    _verify_receipt_file(checkpoint_output, checkpoint, CHECKPOINT)
    out = receipt(
        {
            "schema_version": EXECUTION,
            "state": "MEDIATED_STAGE_EXECUTED",
            "study_id": policy["study_id"],
            "ai_lock_receipt_sha256": ai_lock["receipt_sha256"],
            "ah_lock_receipt_sha256": ah_lock["receipt_sha256"],
            "predecessor_checkpoint_receipt_sha256": None if previous is None else previous["receipt_sha256"],
            "guard_receipt_sha256": guard["receipt_sha256"],
            "checkpoint_receipt_sha256": checkpoint["receipt_sha256"],
            "stage_index": stage_index,
            "stage_id": stage_id,
            "command_plan_stage_sha256": hb(cb(stage)),
            "commands": command_witnesses,
            "guard_persisted_before_process_launch": True,
            "guard_reverified_after_process_exit": True,
            "checkpoint_persisted_after_successful_process_exit": True,
            "shell_used": False,
            "network_enforcement_level": "declared_only",
            "filesystem_write_boundary_enforced": False,
            "continuous_filesystem_immutability_proven": False,
            "theorem": "controller_enforced_guard_then_exact_argv_then_checkpoint_sequence",
        }
    )
    write_once(execution_output, out)
    return out


def selftest() -> None:
    old_env = os.environ.get("LL009AI_TEST")
    os.environ["LL009AI_TEST"] = "baseline"
    try:
        with tempfile.TemporaryDirectory(prefix="ll009ai-") as td:
            root = Path(td)
            repo = root / "repo"; rt = root / "runtime"; er = root / "evidence"; ctl = root / "control"
            repo.mkdir(); rt.mkdir(); er.mkdir(); ctl.mkdir()
            policy = {
                "schema_version": P,
                "study_id": "synthetic-ai",
                "required_environment_profiles": ["acquisition", "gis", "analysis"],
                "environment_contracts": {
                    k: {"required_packages": [], "required_libraries": [], "required_environment_variables": ["LL009AI_TEST"]}
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
            pp = repo / "policy.json"; wj(pp, policy); (repo / "tool.py").write_text("fixed\n")
            _git(repo, "init", "-q"); _git(repo, "config", "user.email", "ll009ai@example.invalid"); _git(repo, "config", "user.name", "LL-009AI Synthetic")
            _git(repo, "add", "policy.json", "tool.py"); _git(repo, "commit", "-qm", "synthetic AI subject"); head = _git(repo, "rev-parse", "HEAD")
            env_map: dict[str, str] = {}
            for profile in ["acquisition", "gis", "analysis"]:
                env_map[profile] = profile + ".json"; wj(rt / env_map[profile], capture(vp(policy), profile))
            pre = prepare(pp, repo, head, rt, env_map)
            q = native_receipt({"schema_version": "q.v1", "value": 1})
            z = native_receipt({"schema_version": "z.v1", "upstream": q["receipt_sha256"]})
            manifest = {
                "schema_version": M,
                "study_id": "synthetic-ai",
                "artifacts": [
                    {"id": "q", "path": "q.json", "requirement": "required", "dependencies": [], "dependency_bindings": [], "self_hash_mode": "ll009_indent2_receipt_sha256"},
                    {"id": "z", "path": "z.json", "requirement": "required", "dependencies": ["q"], "dependency_bindings": [{"dependency_id": "q", "field_path": ["upstream"], "identity": "receipt_sha256"}], "self_hash_mode": "ll009_indent2_receipt_sha256"},
                ],
            }
            mp = root / "manifest.json"; wj(mp, manifest)
            stage_map = {
                "schema_version": STAGE_MAP,
                "study_id": "synthetic-ai",
                "stages": [
                    {"id": "acquire", "environment_profile": "acquisition", "may_access_network": True, "artifact_ids": ["q"]},
                    {"id": "analyze", "environment_profile": "analysis", "may_access_network": False, "artifact_ids": ["z"]},
                ],
            }
            smp = root / "stage-map.json"; wj(smp, stage_map)
            ah_lock = create_ah_lock(pre, pp, repo, head, rt, mp, smp, er)

            q_text = json.dumps(q, sort_keys=True)
            z_text = json.dumps(z, sort_keys=True)
            literal = "$(touch SHOULD_NOT_EXIST)"
            code0 = (
                "import os,sys; from pathlib import Path; "
                "g=Path(os.environ['LL009AI_GUARD_PATH']); assert g.exists(); "
                f"assert sys.argv[1]=={literal!r}; "
                f"(Path(os.environ['LL009AI_EVIDENCE_ROOT'])/'q.json').write_text({q_text!r})"
            )
            code1 = (
                "import os; from pathlib import Path; "
                "assert Path(os.environ['LL009AI_GUARD_PATH']).exists(); "
                f"(Path(os.environ['LL009AI_EVIDENCE_ROOT'])/'z.json').write_text({z_text!r})"
            )
            plan = {
                "schema_version": PLAN,
                "study_id": "synthetic-ai",
                "stages": [
                    {"id": "acquire", "expected_artifact_ids": ["q"], "network_mode": "authorized_network", "commands": [{"runner": "prepared_python", "argv": ["-c", code0, literal], "cwd": "."}]},
                    {"id": "analyze", "expected_artifact_ids": ["z"], "network_mode": "declared_offline_only", "commands": [{"runner": "prepared_python", "argv": ["-c", code1], "cwd": "."}]},
                ],
            }
            cpp = root / "command-plan.json"; wj(cpp, plan)
            ai_lock = create_execution_lock(ah_lock, pre, pp, repo, head, rt, mp, smp, cpp, er)

            e0 = run_stage(ai_lock, ah_lock, None, pre, pp, repo, head, rt, mp, smp, cpp, er, "acquire", ctl / "g0.json", ctl / "c0.json", ctl / "e0.json")
            c0 = lj(ctl / "c0.json"); vr(c0, CHECKPOINT)
            if not e0["guard_persisted_before_process_launch"] or e0["shell_used"]:
                raise CE("AI sequencing/shell theorem failed")
            if (repo / "SHOULD_NOT_EXIST").exists():
                raise CE("AI shell metacharacter escaped argv semantics")
            e1 = run_stage(ai_lock, ah_lock, c0, pre, pp, repo, head, rt, mp, smp, cpp, er, "analyze", ctl / "g1.json", ctl / "c1.json", ctl / "e1.json")
            c1 = lj(ctl / "c1.json"); vr(c1, CHECKPOINT)
            if not c1["all_declared_stages_checkpointed"] or e1["network_enforcement_level"] != "declared_only":
                raise CE("AI final stage theorem drift")
            print("LL-009AI self-test PASS: persisted guard precedes shell-free exact argv; child observes guard; checkpoints follow successful exits; shell metacharacters remain literal")
    finally:
        if old_env is None:
            os.environ.pop("LL009AI_TEST", None)
        else:
            os.environ["LL009AI_TEST"] = old_env


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
    p.add_argument("--command-plan", type=Path, required=True)
    p.add_argument("--evidence-root", type=Path, required=True)
    p.add_argument("--ah-lock", type=Path, required=True)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LL-009AI mediated guard→process→checkpoint runner")
    sp = p.add_subparsers(dest="cmd", required=True)
    x = sp.add_parser("lock"); _common(x); x.add_argument("--output", type=Path, required=True)
    x = sp.add_parser("run-stage"); _common(x); x.add_argument("--ai-lock", type=Path, required=True); x.add_argument("--previous-checkpoint", type=Path); x.add_argument("--stage-id", required=True); x.add_argument("--guard-output", type=Path, required=True); x.add_argument("--checkpoint-output", type=Path, required=True); x.add_argument("--execution-output", type=Path, required=True)
    sp.add_parser("self-test")
    return p


def main() -> int:
    a = parser().parse_args()
    try:
        if a.cmd == "self-test":
            selftest(); return 0
        pre = _rr(a.preparation, PRE); ah_lock = _rr(a.ah_lock, AH_LOCK)
        if a.cmd == "lock":
            out = create_execution_lock(ah_lock, pre, a.policy, a.repo_root, a.repo_head, a.runtime_root, a.manifest, a.stage_map, a.command_plan, a.evidence_root)
            write_once(a.output, out); print("AI LOCKED", out["receipt_sha256"]); return 0
        ai_lock = _rr(a.ai_lock, AI_LOCK); previous = _rr(a.previous_checkpoint, CHECKPOINT) if a.previous_checkpoint else None
        out = run_stage(ai_lock, ah_lock, previous, pre, a.policy, a.repo_root, a.repo_head, a.runtime_root, a.manifest, a.stage_map, a.command_plan, a.evidence_root, a.stage_id, a.guard_output, a.checkpoint_output, a.execution_output)
        print("AI EXECUTED", out["stage_id"], out["receipt_sha256"]); return 0
    except CE as e:
        print("LL-009AI FAIL:", e, file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
