#!/usr/bin/env python3
"""Hermetic helper-level self-tests for WCARE-50."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


materializer = load("wcare50_materializer", ROOT / "scripts/wcare50_materialize_fixtures.py")
precondition = load("wcare50_precondition", ROOT / "scripts/wcare50_verify_w49_precondition.py")
executor = load("wcare50_executor", ROOT / "scripts/wcare50_execute_qualification.py")


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def expect_exception(exc_type, fn, marker: str) -> None:
    try:
        fn()
    except exc_type as exc:
        assert marker in str(exc), (marker, str(exc))
    else:
        raise AssertionError(f"expected {exc_type.__name__}: {marker}")


def test_fixture_package(tmp: Path) -> None:
    archive = ROOT / "docs/release/evidence/WCARE50_FIXTURE_PACKAGE_V1.zlib.b64"
    package, package_bytes = materializer.load_package(archive)
    materializer.validate(package)
    assert materializer.sha256(package_bytes) == materializer.PACKAGE_SHA256
    assert [case["name"] for case in package["cases"]] == list(materializer.EXPECTED_CASES)

    output = tmp / "fixtures"
    materializer.materialize(package, output)
    for case in package["cases"]:
        case_dir = output / case["name"]
        executor.verify_materialized_case(case, case_dir)

    altered = json.loads(json.dumps(package))
    altered["cases"][0]["file_sha256"]["plan.json"] = "00" * 32
    expect_exception(SystemExit, lambda: materializer.validate(altered), "fixture_file_digest_mismatch")


def exact_w49_run(status: str = "queued", conclusion=None) -> dict:
    return {
        "id": precondition.RUN_ID,
        "name": precondition.RUN_NAME,
        "path": precondition.RUN_PATH,
        "head_branch": precondition.RUN_BRANCH,
        "head_sha": precondition.RUN_HEAD,
        "run_number": precondition.RUN_NUMBER,
        "run_attempt": precondition.RUN_ATTEMPT,
        "event": "push",
        "status": status,
        "conclusion": conclusion,
    }


def test_w49_pending_and_identity_rejection(tmp: Path) -> None:
    run_path = tmp / "w49-run.json"
    write_json(run_path, exact_w49_run())
    value, code = precondition.verify(run_path, None, None)
    assert code == 3
    assert value["classification"] == "WCARE49_PRECONDITION_PENDING"
    assert value["execution_inputs_closed"] is False
    assert value["wcare42_executable_qualification_established"] is False

    wrong = exact_w49_run()
    wrong["run_attempt"] = 2
    write_json(run_path, wrong)
    expect_exception(
        precondition.EvidenceFailure,
        lambda: precondition.verify(run_path, None, None),
        "run_identity_mismatch:run_attempt",
    )


def test_w49_precondition_input_binding(tmp: Path) -> None:
    path = tmp / "precondition.json"
    value = {
        "authority": executor.AUTHORITY,
        "classification": "WCARE49_PRECONDITION_SATISFIED",
        "detail": "exact_hosted_wcare49_execution_input_closure_verified",
        "wcare49_run_id": "34877774593",
        "wcare49_run_number": "7",
        "wcare49_run_attempt": "1",
        "wcare49_head": executor.W49_HEAD,
        "final_subject": executor.FINAL,
        "lock_sha256": executor.LOCK_SHA256,
        "lock_git_blob": executor.LOCK_BLOB,
        "run_json_sha256": "11" * 32,
        "jobs_json_sha256": "22" * 32,
        "job_log_sha256": "33" * 32,
        "wcare49_result_sha256": "44" * 32,
        "wcare47_result_sha256": "55" * 32,
        "execution_inputs_closed": True,
        "wcare42_executable_qualification_established": False,
        "runtime_authority_granted": False,
    }
    write_json(path, value)
    parsed, digest = executor.load_w49_precondition(path)
    assert parsed["execution_inputs_closed"] is True
    assert len(digest) == 64

    value["wcare42_executable_qualification_established"] = True
    write_json(path, value)
    expect_exception(
        executor.QualificationFailure,
        lambda: executor.load_w49_precondition(path),
        "wcare49_precondition_mismatch:wcare42_executable_qualification_established",
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="wcare50-selftest-") as raw:
        tmp = Path(raw)
        test_fixture_package(tmp)
        test_w49_pending_and_identity_rejection(tmp)
        test_w49_precondition_input_binding(tmp)
    print("PASS_WCARE50_FAIL_CLOSED_SELFTESTS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
