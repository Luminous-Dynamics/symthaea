#!/usr/bin/env python3
"""Dependency-free WCARE-39 synthetic Git campaign."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "wcare39_execution_capsule.py"
QUALIFIER = HERE / "wcare39-qualify.sh"
PROTOCOL = "wcare39-execution-capsule-v1"


def run(argv: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check, timeout=120)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def init_repo(root: Path) -> str:
    run(["git", "init", "-q"], root)
    run(["git", "config", "user.email", "wcare39@example.invalid"], root)
    run(["git", "config", "user.name", "WCARE39 Selftest"], root)
    write(root / ".gitignore", "target/\n")
    write(root / "Cargo.lock", "# synthetic cargo lock\n")
    write(root / "flake.lock", "{}\n")
    write(root / "rust-toolchain.toml", "[toolchain]\nchannel = \"stable\"\n")
    write(root / "pass_stage.py", "from pathlib import Path\nPath('target/pass-receipt.json').write_text('{\"ok\":true}\\n')\n")
    write(root / "fail_stage.py", "raise SystemExit(7)\n")
    write(root / "drift_stage.py", "from pathlib import Path\np=Path('Cargo.lock')\np.write_text(p.read_text()+'# drift\\n')\n")
    write(root / "delete_stage.py", "from pathlib import Path\nPath(__file__).unlink()\n")
    write(root / "wcare38_dummy.py", "raise SystemExit(0)\n")
    run(["git", "add", "."], root)
    run(["git", "commit", "-q", "-m", "synthetic subject"], root)
    return run(["git", "rev-parse", "HEAD"], root).stdout.decode().strip()


def plan(root: Path, head: str, script: str, output: str | None) -> dict:
    return {
        "protocol_version": PROTOCOL,
        "subject_git_head": head,
        "subject_digests": {script: sha(root / script)},
        "materials": [],
        "safe_environment": [],
        "network_policy_declared": "Unspecified",
        "sandbox_policy_declared": "Unspecified",
        "deterministic_seed_commitments": {},
        "stages": [{
            "stage_id": script.removesuffix(".py"),
            "argv": [sys.executable, script],
            "cwd": "",
            "timeout_seconds": 30,
            "output_receipt_path": output,
        }],
    }


def write_plan(root: Path, plan_value: dict, name: str) -> Path:
    target = root / "target"
    target.mkdir(exist_ok=True)
    plan_path = target / f"{name}-plan.json"
    plan_path.write_text(json.dumps(plan_value, sort_keys=True, separators=(",", ":")) + "\n")
    return plan_path


def execute(root: Path, plan_value: dict, name: str) -> tuple[int, dict]:
    plan_path = write_plan(root, plan_value, name)
    proc = run([sys.executable, str(RUNNER), "run", str(plan_path), f"target/{name}-capsule"], root, check=False)
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"non-json runner output for {name}: {proc.stdout!r} {proc.stderr!r}") from exc
    return proc.returncode, result


def execute_qualifier(root: Path, plan_value: dict, name: str) -> tuple[int, dict]:
    plan_path = write_plan(root, plan_value, f"qualifier-{name}")
    proc = run(["bash", str(QUALIFIER), str(plan_path), f"target/qualifier-{name}-capsule"], root, check=False)
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"non-json qualifier output for {name}: {proc.stdout!r} {proc.stderr!r}") from exc
    return proc.returncode, result


def assert_prepared_binding(root: Path, name: str, final: dict) -> None:
    prepared = root / "target" / f"{name}-capsule" / "prepared.json"
    final_path = root / "target" / f"{name}-capsule" / "final.json"
    assert prepared.is_file(), f"{name}: prepared not persisted"
    assert final_path.is_file(), f"{name}: final not persisted"
    assert final["prepared_capsule_sha256"] == sha(prepared), f"{name}: final not bound to prepared"
    compare = run([sys.executable, str(RUNNER), "compare", str(prepared), str(final_path)], root, check=False)
    compared = json.loads(compare.stdout)
    assert compared["classification"] == final["classification"], (name, compared, final)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="wcare39-") as temp:
        root = Path(temp)
        head = init_repo(root)

        pass_plan = plan(root, head, "pass_stage.py", "target/pass-receipt.json")
        code, passed = execute(root, pass_plan, "pass")
        assert code == 0, passed
        assert passed["classification"] == "QUALIFIED_EXECUTION" and passed["environment_integrity"] == "QUALIFIED", passed
        assert passed["subject_outcome"] == "PASS", passed
        assert passed["commands"][0]["output_receipt_sha256"] == sha(root / "target/pass-receipt.json")
        assert_prepared_binding(root, "pass", passed)

        # Exact PREPARED binding is mandatory: a copied FINAL cannot point at another seal.
        prepared_path = root / "target/pass-capsule/prepared.json"
        tampered_final_path = root / "target/tampered-final.json"
        tampered_final = json.loads((root / "target/pass-capsule/final.json").read_text())
        tampered_final["prepared_capsule_sha256"] = "0" * 64
        tampered_final_path.write_text(json.dumps(tampered_final, sort_keys=True, separators=(",", ":")) + "\n")
        tampered = run([sys.executable, str(RUNNER), "compare", str(prepared_path), str(tampered_final_path)], root, check=False)
        assert tampered.returncode == 4, tampered.stdout
        assert json.loads(tampered.stdout)["classification"] == "INVALID_CAPSULE"

        # Existing output must never be reused as if this stage produced it.
        code, stale = execute(root, pass_plan, "stale-output")
        assert code == 0, stale
        assert stale["classification"] == "QUALIFIED_EXECUTION", stale
        assert stale["subject_outcome"] == "INVALID", stale
        assert stale["commands"][0]["output_receipt_sha256"] is None, stale

        code, failed = execute(root, plan(root, head, "fail_stage.py", None), "fail")
        assert code == 0, failed
        assert failed["classification"] == "QUALIFIED_EXECUTION" and failed["environment_integrity"] == "QUALIFIED", failed
        assert failed["subject_outcome"] == "FAIL", failed
        assert_prepared_binding(root, "fail", failed)

        code, drifted = execute(root, plan(root, head, "drift_stage.py", None), "drift")
        assert code == 2, drifted
        assert drifted["classification"] == "ENVIRONMENT_DRIFT" and drifted["environment_integrity"] == "DRIFTED", drifted
        assert "materials" in drifted["drift_fields"] and "worktree_clean" in drifted["drift_fields"], drifted
        assert_prepared_binding(root, "drift", drifted)
        run(["git", "reset", "--hard", "-q", "HEAD"], root)

        code, deleted = execute(root, plan(root, head, "delete_stage.py", None), "delete-subject")
        assert code == 2, deleted
        assert deleted["classification"] == "ENVIRONMENT_DRIFT", deleted
        assert "subject_digests" in deleted["drift_fields"], deleted
        assert deleted["subject_digests"]["delete_stage.py"] is None, deleted
        assert_prepared_binding(root, "delete-subject", deleted)
        run(["git", "reset", "--hard", "-q", "HEAD"], root)

        code, blocked = execute(root, plan(root, head, "wcare38_dummy.py", None), "wcare38-lock")
        assert code == 3, blocked
        assert blocked["classification"] == "INFRASTRUCTURE_INDETERMINATE", blocked
        material = {item["path"]: item for item in blocked["materials"]}
        lock = material["tools/wcare37_attestation_verifier/Cargo.lock"]
        assert lock["required"] is True and lock["present"] is False, blocked
        assert not (root / "target/wcare38-lock-capsule/prepared.json").exists()

        secret_plan = plan(root, head, "fail_stage.py", None)
        secret_plan["safe_environment"] = [{"key": "API_TOKEN", "mode": "Literal"}]
        code, secret = execute(root, secret_plan, "secret-literal")
        assert code == 4, secret
        assert secret["classification"] == "INVALID_CAPSULE", secret
        assert "sensitive_environment_literal_forbidden" in secret["detail"], secret

        # The CI-facing qualifier must preserve subject outcome in its exit status.
        (root / "target/pass-receipt.json").unlink(missing_ok=True)
        qcode, qpass = execute_qualifier(root, pass_plan, "pass")
        assert qcode == 0 and qpass["classification"] == "QUALIFIED_EXECUTION" and qpass["subject_outcome"] == "PASS", qpass

        qcode, qfail = execute_qualifier(root, plan(root, head, "fail_stage.py", None), "fail")
        assert qcode == 1 and qfail["classification"] == "QUALIFIED_EXECUTION" and qfail["subject_outcome"] == "FAIL", qfail

        qcode, qdrift = execute_qualifier(root, plan(root, head, "drift_stage.py", None), "drift")
        assert qcode == 2 and qdrift["classification"] == "ENVIRONMENT_DRIFT", qdrift
        run(["git", "reset", "--hard", "-q", "HEAD"], root)

        qcode, qindeterminate = execute_qualifier(root, plan(root, head, "wcare38_dummy.py", None), "indeterminate")
        assert qcode == 3 and qindeterminate["classification"] == "INFRASTRUCTURE_INDETERMINATE", qindeterminate

        qsecret_plan = plan(root, head, "fail_stage.py", None)
        qsecret_plan["safe_environment"] = [{"key": "API_TOKEN", "mode": "Literal"}]
        qcode, qinvalid = execute_qualifier(root, qsecret_plan, "invalid")
        assert qcode == 4 and qinvalid["classification"] == "INVALID_CAPSULE", qinvalid

    print(json.dumps({
        "authority": "MeasurementOnly",
        "classification": "PASS_WCARE39_SELFTEST",
        "qualified_pass_observed": True,
        "qualified_subject_failure_observed": True,
        "stale_output_rejected": True,
        "prepared_binding_tamper_rejected": True,
        "environment_drift_observed": True,
        "missing_subject_drift_observed": True,
        "wcare37_lock_blocker_preserved": True,
        "sensitive_literal_rejected": True,
        "qualifier_exit_contract_verified": True,
        "runtime_authority_granted": False,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
