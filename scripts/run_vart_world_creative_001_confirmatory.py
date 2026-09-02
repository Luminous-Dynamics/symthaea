#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import verify_vart_world_creative_001_confirmatory_launch as launch_gate

EXPERIMENT_ID = "VART-WORLD-CREATIVE-001"
SCHEMA = "symthaea.vart-world-creative-001.confirmatory-run.v1"


class RunError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RunError(f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RunError(f"invalid JSON at {path}: {exc}") from exc


def write_new_json(path: Path, value: Any) -> None:
    if path.exists():
        raise RunError(f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def run(argv: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def git(repo: Path, *args: str) -> str:
    proc = run(["git", "-C", str(repo), *args], cwd=repo)
    if proc.returncode != 0:
        raise RunError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def meaningful_status(repo: Path) -> list[str]:
    lines = git(repo, "status", "--porcelain=v1", "--untracked-files=all").splitlines()
    return [line for line in lines if "/__pycache__/" not in f"/{line[3:]}" and not line[3:].endswith(".pyc")]


def require_clean_identity(repo: Path, expected: Any, label: str) -> tuple[str, str]:
    if not isinstance(expected, dict):
        raise RunError(f"{label} source identity must be object")
    head, tree = expected.get("head"), expected.get("tree")
    if not isinstance(head, str) or len(head) != 40 or not isinstance(tree, str) or len(tree) != 40:
        raise RunError(f"{label} HEAD/TREE must be full 40-hex identities")
    if meaningful_status(repo):
        raise RunError(f"{label} checkout is dirty")
    actual_head = git(repo, "rev-parse", "HEAD")
    actual_tree = git(repo, "rev-parse", "HEAD^{tree}")
    if (actual_head, actual_tree) != (head, tree):
        raise RunError(f"{label} identity mismatch: {actual_head}/{actual_tree} != {head}/{tree}")
    return actual_head, actual_tree


def resolve(value: str, base: Path) -> Path:
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def expand_argv(template: Any, values: dict[str, Any]) -> list[str]:
    if not isinstance(template, list) or not template:
        raise RunError("runtime_argv must be a non-empty array")
    out: list[str] = []
    for raw in template:
        if not isinstance(raw, str):
            raise RunError("runtime_argv entries must be strings")
        if raw.startswith("__REPLACE_"):
            raise RunError("runtime_argv contains unresolved placeholder")
        try:
            out.append(raw.format_map(values))
        except KeyError as exc:
            raise RunError(f"unknown runtime_argv placeholder: {exc}") from exc
    return out


def tree_closure(root: Path, excluded: set[str]) -> tuple[str, list[dict[str, str]]]:
    entries: list[dict[str, str]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel not in excluded:
            entries.append({"path": rel, "sha256": sha256_file(path)})
    return sha256_bytes(canonical_bytes(entries)), entries


def copy_contracts(cfg: dict[str, Any], base: Path, root: Path) -> dict[str, str]:
    raw = cfg.get("contract_inputs")
    if not isinstance(raw, dict):
        raise RunError("contract_inputs must be an object")
    required = {"analysis_contract": "analysis_contract.json", "metric_definitions": "metric_definitions.json"}
    copied: dict[str, str] = {}
    for key, dest in required.items():
        value = raw.get(key)
        if not isinstance(value, str) or not value:
            raise RunError(f"missing contract_inputs.{key}")
        src = resolve(value, base)
        if not src.is_file():
            raise RunError(f"missing contract file: {src}")
        shutil.copyfile(src, root / dest)
        copied[dest] = sha256_file(root / dest)
    return copied


def preserve_logs(log_dir: Path, trial_id: str, stdout: str, stderr: str) -> tuple[str, str]:
    out = log_dir / f"{trial_id}.stdout.txt"
    err = log_dir / f"{trial_id}.stderr.txt"
    out.write_text(stdout, encoding="utf-8")
    err.write_text(stderr, encoding="utf-8")
    out.chmod(0o600)
    err.chmod(0o600)
    return sha256_file(out), sha256_file(err)


def require_stage(stage: Path, trial_id: str) -> Path:
    trials = stage / "trials"
    if not trials.is_dir():
        raise RunError(f"{trial_id}: runtime did not emit trials/")
    children = sorted(p.name for p in trials.iterdir())
    if children != [trial_id]:
        raise RunError(f"{trial_id}: unexpected staged trials {children}")
    return trials / trial_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Zero-peeking VART-WORLD-CREATIVE-001 confirmatory runner")
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    cfg = read_json(config_path)
    if not isinstance(cfg, dict) or cfg.get("schema") != SCHEMA or cfg.get("experiment_id") != EXPERIMENT_ID:
        raise RunError("unexpected confirmatory run config")
    if cfg.get("claim_authorized") is not False:
        raise RunError("confirmatory runner requires claim_authorized=false")

    base = config_path.parent
    freeze_path = resolve(cfg["freeze_path"], base)
    inventory_path = resolve(cfg["trial_inventory_path"], base)
    expected_freeze = cfg.get("expected_freeze_sha256")
    if not isinstance(expected_freeze, str):
        raise RunError("expected_freeze_sha256 missing")
    try:
        launch = launch_gate.verify(freeze_path, expected_freeze, inventory_path)
    except launch_gate.Reject as exc:
        raise RunError(f"launch gate rejected: {exc.code}: {exc.detail}") from exc
    if launch.get("verdict") != "CONFIRMATORY_LAUNCH_READY":
        raise RunError("launch gate did not return CONFIRMATORY_LAUNCH_READY")

    runtime_root = resolve(cfg["runtime_root"], base)
    instrument_root = Path(__file__).resolve().parent.parent
    subject_head, subject_tree = require_clean_identity(runtime_root, cfg.get("expected_subject_source"), "subject")
    instrument_head, instrument_tree = require_clean_identity(instrument_root, cfg.get("expected_instrument_source"), "instrument")

    evidence_root = resolve(cfg["evidence_root"], base)
    if evidence_root.exists() and any(evidence_root.iterdir()):
        raise RunError(f"evidence_root must be fresh and empty: {evidence_root}")

    inventory = read_json(inventory_path)
    rows = launch_gate.rows_from_inventory(inventory)
    rows = sorted(rows, key=lambda row: row["run_order"])
    argv_template = cfg.get("runtime_argv")
    preview: list[dict[str, Any]] = []
    for row in rows:
        values = {**row, "runtime_root": str(runtime_root), "output_root": "<private-stage>",
                  "experiment_id": EXPERIMENT_ID, "campaign": "confirmatory"}
        preview.append({"run_order": row["run_order"], "trial_id": row["trial_id"],
                        "argv": expand_argv(argv_template, values)})

    verifier = resolve(cfg["qualified_verifier_path"], base)
    if not verifier.is_file():
        raise RunError(f"qualified verifier not found: {verifier}")

    if args.dry_run:
        print(json.dumps({
            "verdict": "CONFIRMATORY_DRY_RUN_READY",
            "trial_count": len(rows),
            "schedule_sha256": launch["schedule_sha256"],
            "freeze_sha256": expected_freeze,
            "subject_head": subject_head,
            "subject_tree": subject_tree,
            "instrument_head": instrument_head,
            "instrument_tree": instrument_tree,
            "zero_peeking": True,
            "automatic_retry": False,
            "claim_authorized": False,
        }, sort_keys=True))
        return 0

    evidence_root.mkdir(parents=True, exist_ok=True)
    orch = evidence_root / "_orchestrator"
    trials_dest = evidence_root / "trials"
    logs = orch / "private_logs"
    orch.mkdir()
    trials_dest.mkdir()
    logs.mkdir()
    logs.chmod(0o700)

    shutil.copyfile(freeze_path, evidence_root / "confirmatory_freeze.json")
    shutil.copyfile(inventory_path, evidence_root / "trial_inventory.json")
    contract_shas = copy_contracts(cfg, base, evidence_root)

    resolved = {
        "schema": "symthaea.vart-world-creative-001.confirmatory-resolved-schedule.v1",
        "experiment_id": EXPERIMENT_ID,
        "freeze_sha256": expected_freeze,
        "trial_inventory_sha256": sha256_file(evidence_root / "trial_inventory.json"),
        "schedule_sha256": launch["schedule_sha256"],
        "subject_source": {"head": subject_head, "tree": subject_tree},
        "instrument_source": {"head": instrument_head, "tree": instrument_tree},
        "zero_peeking": True,
        "automatic_retry": False,
        "trials": preview,
        "claim_authorized": False,
    }
    write_new_json(orch / "resolved_schedule.json", resolved)

    receipts: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows, start=1):
        trial_id = row["trial_id"]
        print(f"[{ordinal}/{len(rows)}] {trial_id}: START", flush=True)
        with tempfile.TemporaryDirectory(prefix=f"vart-confirm-{row['run_order']:03d}-") as td:
            stage = Path(td)
            values = {**row, "runtime_root": str(runtime_root), "output_root": str(stage),
                      "experiment_id": EXPERIMENT_ID, "campaign": "confirmatory"}
            argv = expand_argv(argv_template, values)
            env = os.environ.copy()
            env.update({
                "VART_EXPERIMENT_ID": EXPERIMENT_ID,
                "VART_CAMPAIGN": "confirmatory",
                "VART_ZERO_PEEKING": "1",
                "VART_CLAIM_AUTHORIZED": "0",
                "VART_TRIAL_ID": trial_id,
                "VART_RUN_ORDER": str(row["run_order"]),
                "VART_OUTPUT_ROOT": str(stage),
                "VART_FREEZE_SHA256": expected_freeze,
            })
            proc = run(argv, cwd=runtime_root, env=env)
            stdout_sha, stderr_sha = preserve_logs(logs, trial_id, proc.stdout, proc.stderr)
            receipt = {
                "trial_id": trial_id,
                "run_order": row["run_order"],
                "argv_sha256": sha256_bytes(canonical_bytes(argv)),
                "returncode": proc.returncode,
                "stdout_sha256": stdout_sha,
                "stderr_sha256": stderr_sha,
                "automatic_retry": False,
            }
            if proc.returncode != 0:
                abort_dir = evidence_root / "aborted" / trial_id
                abort_dir.parent.mkdir(exist_ok=True)
                shutil.copytree(stage, abort_dir)
                receipts.append(receipt)
                write_new_json(orch / "CAMPAIGN_ABORT_RECEIPT.json", {
                    "schema": "symthaea.vart-world-creative-001.confirmatory-abort-receipt.v1",
                    "experiment_id": EXPERIMENT_ID,
                    "freeze_sha256": expected_freeze,
                    "failed_trial_id": trial_id,
                    "completed_trial_count": len(receipts) - 1,
                    "attempted_trial_count": len(receipts),
                    "trial_receipts": receipts,
                    "reason": "runtime_process_nonzero_exit",
                    "automatic_retry": False,
                    "claim_authorized": False,
                })
                print(f"[{ordinal}/{len(rows)}] {trial_id}: ABORT returncode={proc.returncode} stdout={stdout_sha} stderr={stderr_sha}", flush=True)
                return 2

            trial_dir = require_stage(stage, trial_id)
            destination = trials_dest / trial_id
            if destination.exists():
                raise RunError(f"duplicate trial evidence destination: {destination}")
            shutil.copytree(trial_dir, destination)
            receipt["trial_evidence_sha256"] = tree_closure(destination, set())[0]
            receipts.append(receipt)
            print(f"[{ordinal}/{len(rows)}] {trial_id}: SEALED stdout={stdout_sha} stderr={stderr_sha}", flush=True)

    expected_ids = [row["trial_id"] for row in rows]
    actual_ids = sorted(p.name for p in trials_dest.iterdir() if p.is_dir())
    if sorted(expected_ids) != actual_ids:
        raise RunError("complete trial accounting mismatch before campaign seal")

    closure, entries = tree_closure(evidence_root, {"_orchestrator/CONFIRMATORY_CAMPAIGN_RECEIPT.json"})
    campaign_receipt = {
        "schema": "symthaea.vart-world-creative-001.confirmatory-campaign-receipt.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "sealed",
        "freeze_sha256": expected_freeze,
        "trial_inventory_sha256": sha256_file(evidence_root / "trial_inventory.json"),
        "schedule_sha256": launch["schedule_sha256"],
        "subject_source": {"head": subject_head, "tree": subject_tree},
        "instrument_source": {"head": instrument_head, "tree": instrument_tree},
        "contract_shas": contract_shas,
        "trial_count": len(rows),
        "trial_receipts": receipts,
        "evidence_closure_sha256": closure,
        "evidence_file_count": len(entries),
        "zero_peeking": True,
        "automatic_retry": False,
        "claim_authorized": False,
    }
    write_new_json(orch / "CONFIRMATORY_CAMPAIGN_RECEIPT.json", campaign_receipt)

    verify_proc = run([sys.executable, str(verifier), str(evidence_root),
                       "--expected-freeze-sha256", expected_freeze, "--json"], cwd=verifier.parent)
    verifier_stdout_sha = sha256_bytes(verify_proc.stdout.encode("utf-8"))
    verifier_stderr_sha = sha256_bytes(verify_proc.stderr.encode("utf-8"))
    if verify_proc.returncode != 0:
        print(f"CONFIRMATORY_CAMPAIGN_SEALED_VERIFIER_REJECT stdout={verifier_stdout_sha} stderr={verifier_stderr_sha}")
        return 2

    try:
        verifier_result = json.loads(verify_proc.stdout)
    except json.JSONDecodeError as exc:
        raise RunError("qualified verifier did not emit JSON") from exc
    if verifier_result.get("verdict") not in {"ACCEPT", None}:
        raise RunError(f"unexpected verifier verdict: {verifier_result.get('verdict')}")

    print(json.dumps({
        "verdict": "CONFIRMATORY_CAMPAIGN_SEALED_AND_VERIFIED",
        "trial_count": len(rows),
        "freeze_sha256": expected_freeze,
        "evidence_closure_sha256": closure,
        "campaign_receipt_sha256": sha256_file(orch / "CONFIRMATORY_CAMPAIGN_RECEIPT.json"),
        "verifier_stdout_sha256": verifier_stdout_sha,
        "zero_peeking": True,
        "claim_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RunError, KeyError, TypeError, ValueError, OSError) as exc:
        print(json.dumps({"verdict": "CONFIRMATORY_RUN_REJECT", "error": str(exc),
                          "claim_authorized": False}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
