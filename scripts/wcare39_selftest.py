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
    write(root / "pass_stage.py", "from pathlib import Path\nPath('target/pass-receipt.json').write_text('{"ok":true}\\n')\n")
    write(root / "fail_stage.py", "raise SystemExit(7)\n")
    write(root / "drift_stage.py", "from pathlib import Path\np=Path('Cargo.lock')\np.write_text(p.read_text()+'# drift\\n')\n")
    write(root / "wcare38_dummy.py", "raise SystemExit(0)\n")
    run(["git", "add", "."], root)
    run(["git", "commit", "-q", "-m", "synthetic subject"], root)
    return run(["git", "rev-parse", "HEAD"], root).stdout.decode().strip()


def plan(root: Path, head: str, script: str, output: str | None) -> dict:
    script_path = root / script
    return {
        "protocol_version": PROTOCOL,
        "subject_git_head": head,
        "subject_digests": {script: sha(script_path)},
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


def execute(root: Path, plan_value: dict, name: str) -> tuple[int, dict]:
    target = root / "target"
    target.mkdir(exist_ok=True)
    plan_path = target / f"{name}-plan.json"
    plan_path.write_text(json.dumps(plan_value, sort_keys=True, separators=(",", ":")) + "\n")
    proc = run(
        [sys.executable, str(RUNNER), "run", str(plan_path), f"target/{name}-capsule"],
        root,
        check=False,
    )
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"non-json runner output for {name}: {proc.stdout!r} {proc.stderr!r}") from exc
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

        code, passed = execute(root, plan(root, head, "pass_stage.py", "target/pass-receipt.json"), "pass")
        assert code == 0, passed
        assert passed["classification"] == "QUALIFIED_EXECUTION", passed
        assert passed["environment_integrity"] == "QUALIFIED", passed
        assert passed["subject_outcome"] == "PASS", passed
        assert passed["commands"][0]["output_receipt_sha256"] == sha(root / "target/pass-receipt.json")
        assert_prepared_binding(root, "pass", passed)

        code, failed = execute(root, plan(root, head, "fail_stage.py", None), "fail")
        assert code == 0, failed
        assert failed["classification"] == "QUALIFIED_EXECUTION", failed
        assert failed["environment_integrity"] == "QUALIFIED", failed
        assert failed["subject_outcome"] == "FAIL", failed
        assert_prepared_binding(root, "fail", failed)

        code, drifted = execute(root, plan(root, head, "drift_stage.py", None), "drift")
        assert code == 2, drifted
        assert drifted["classification"] == "ENVIRONMENT_DRIFT", drifted
        assert drifted["environment_integrity"] == "DRIFTED", drifted
        assert "materials" in drifted["drift_fields"], drifted
        assert "worktree_clean" in drifted["drift_fields"], drifted
        assert_prepared_binding(root, "drift", drifted)
        run(["git", "reset", "--hard", "-q", "HEAD"], root)

        code, blocked = execute(root, plan(root, head, "wcare38_dummy.py", None), "wcare38-lock")
        assert code == 3, blocked
        assert blocked["classification"] == "INFRASTRUCTURE_INDETERMINATE", blocked
        material = {item["path"]: item for item in blocked["materials"]}
        lock = material["tools/wcare37_attestation_verifier/Cargo.lock"]
        assert lock["required"] is True and lock["present"] is False, blocked
        assert not (root / "target" / "wcare38-lock-capsule" / "prepared.json").exists()

    print(json.dumps({
        "authority": "MeasurementOnly",
        "classification": "PASS_WCARE39_SELFTEST",
        "qualified_pass_observed": True,
        "qualified_subject_failure_observed": True,
        "environment_drift_observed": True,
        "wcare37_lock_blocker_preserved": True,
        "runtime_authority_granted": False,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
