#!/usr/bin/env python3
"""Hermetic helper-level self-tests for WCARE-49 fail-closed states."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pre = load("wcare49_precondition", ROOT / "scripts/wcare49_verify_admission_precondition.py")
close = load("wcare49_close", ROOT / "scripts/wcare49_close_inputs.py")


def exact_run(status: str = "queued", conclusion=None) -> dict:
    return {
        "id": pre.RUN_ID,
        "name": pre.RUN_NAME,
        "path": pre.RUN_PATH,
        "head_branch": pre.RUN_BRANCH,
        "head_sha": pre.RUN_HEAD,
        "run_number": pre.RUN_NUMBER,
        "run_attempt": pre.RUN_ATTEMPT,
        "event": "push",
        "status": status,
        "conclusion": conclusion,
    }


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")


def expect_exception(exc_type, fn, contains: str):
    try:
        fn()
    except exc_type as exc:
        assert contains in str(exc), (contains, str(exc))
        return exc
    raise AssertionError(f"expected {exc_type.__name__}: {contains}")


def init_repo(repo: Path) -> None:
    subprocess.run(["git", "init", "-q", str(repo)], check=True)


def commit_fixture(repo: Path) -> None:
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=WCARE49 Selftest",
            "-c",
            "user.email=wcare49-selftest@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repo,
        check=True,
    )


def test_precondition_pending_and_identity_rejection(tmp: Path) -> None:
    run_path = tmp / "run.json"
    write_json(run_path, exact_run())
    result, code = pre.verify(run_path, None, None)
    assert code == 3
    assert result["classification"] == "ADMISSION_PRECONDITION_PENDING"
    assert result["lock_admitted"] is False
    assert result["execution_inputs_closed"] is False

    wrong = exact_run()
    wrong["run_attempt"] = 2
    write_json(run_path, wrong)
    expect_exception(pre.EvidenceFailure, lambda: pre.verify(run_path, None, None), "run_identity_mismatch:run_attempt")


def test_duplicate_identical_hosted_result_rows_are_unambiguous() -> None:
    result = dict(pre.EXPECTED_RESULT)
    result["protocol_version"] = pre.PROTOCOL
    result["cargo_identity"] = "cargo 1.96.0 (fixture)"
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256((canonical + "\n").encode("utf-8")).hexdigest()
    log = "\n".join(
        [
            f"2026-09-14T00:00:00Z {canonical}",
            f"2026-09-14T00:00:01Z {canonical}",
            "2026-09-14T00:00:02Z PASS_PROTOCOL_INTEGRITY",
            "2026-09-14T00:00:03Z PASS_WCARE47_CURRENT_STATE:LOCK_ADMITTED",
            f"2026-09-14T00:00:04Z WCARE47Q_RESULT_SHA256={digest}",
            "2026-09-14T00:00:05Z PASS_WCARE47Q_EXACT_FINAL_LOCK_ADMISSION",
        ]
    )
    parsed, marker = pre.extract_result(log)
    assert parsed == result
    assert marker == digest

    contradictory = dict(result)
    contradictory["runtime_authority_granted"] = True
    contradictory_canonical = json.dumps(contradictory, sort_keys=True, separators=(",", ":"))
    log_with_conflict = log + "\n2026-09-14T00:00:06Z " + contradictory_canonical
    parsed_again, marker_again = pre.extract_result(log_with_conflict)
    assert parsed_again == result
    assert marker_again == digest


def test_config_census_repository_and_ambient_rejection(tmp: Path) -> None:
    repo = tmp / "repo"
    verifier = repo / close.VERIFIER_DIR
    verifier.mkdir(parents=True)
    init_repo(repo)

    checked, found = close.assert_empty_config_census(repo, test_ceiling=tmp)
    assert found == []
    repo_paths = [row["path"] for row in checked if row["scope"] == "repository"]
    assert repo_paths == [
        "tools/wcare42_builder_attestation_verifier/.cargo/config.toml",
        "tools/wcare42_builder_attestation_verifier/.cargo/config",
        "tools/.cargo/config.toml",
        "tools/.cargo/config",
        ".cargo/config.toml",
        ".cargo/config",
    ]
    ambient = [row for row in checked if row["scope"] == "ambient_parent"]
    assert len(ambient) == 2
    assert {row["name"] for row in ambient} == {".cargo/config.toml", ".cargo/config"}
    assert all(row["ancestor_depth"] == 1 for row in ambient)
    assert all(len(row["path_locator_sha256"]) == 64 for row in ambient)

    repo_config = repo / ".cargo" / "config.toml"
    repo_config.parent.mkdir()
    repo_config.write_text("[build]\nrustflags=['--cfg','injected']\n", encoding="utf-8")
    expect_exception(
        close.ClosureFailure,
        lambda: close.assert_empty_config_census(repo, test_ceiling=tmp),
        "cargo_config_census_not_empty",
    )
    repo_config.unlink()
    repo_config.parent.rmdir()

    ambient_config = tmp / ".cargo" / "config"
    ambient_config.parent.mkdir()
    ambient_config.write_text("[build]\ntarget='ambient-target'\n", encoding="utf-8")
    exc = expect_exception(
        close.ClosureFailure,
        lambda: close.assert_empty_config_census(repo, test_ceiling=tmp),
        "cargo_config_census_not_empty",
    )
    assert str(tmp) not in str(exc)
    assert "ambient_parent" in str(exc)
    assert close.locator_sha256(ambient_config) in str(exc)


def test_checkout_cleanliness_rejects_ordinary_and_ignored_untracked(tmp: Path) -> None:
    repo = tmp / "cleanliness-repo"
    repo.mkdir()
    init_repo(repo)
    (repo / ".gitignore").write_text("ignored-input\n", encoding="utf-8")
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    commit_fixture(repo)
    close.assert_checkout_clean(repo, "fixture")

    ordinary = repo / "ordinary-input"
    ordinary.write_text("input\n", encoding="utf-8")
    expect_exception(close.ClosureFailure, lambda: close.assert_checkout_clean(repo, "fixture"), "status_not_clean")
    ordinary.unlink()

    ignored = repo / "ignored-input"
    ignored.write_text("hidden input\n", encoding="utf-8")
    expect_exception(close.ClosureFailure, lambda: close.assert_checkout_clean(repo, "fixture"), "ignored_untracked_files")


def test_environment_override_rejection() -> None:
    original = dict(os.environ)
    try:
        for key in list(os.environ):
            if key in close.FIXED_OVERRIDE_KEYS or any(p.match(key) for p in close.DYNAMIC_OVERRIDE_PATTERNS):
                os.environ.pop(key, None)
        fixed, dynamic = close.environment_census()
        assert dynamic == []
        assert all(value in (None, "") for value in fixed.values())

        for key, value, marker in [
            ("RUSTFLAGS", "--cfg injected", "semantic_environment_override"),
            ("RUSTC_BOOTSTRAP", "1", "semantic_environment_override"),
            ("CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS", "-C target-cpu=native", "dynamic_cargo_environment_override"),
            ("CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER", "/tmp/injected-linker", "dynamic_cargo_environment_override"),
            ("CARGO_PROFILE_TEST_OPT_LEVEL", "3", "dynamic_cargo_environment_override"),
            ("CARGO_SOURCE_CRATES_IO_REPLACE_WITH", "mirror", "dynamic_cargo_environment_override"),
        ]:
            os.environ[key] = value
            expect_exception(close.ClosureFailure, close.environment_census, marker)
            os.environ.pop(key)
    finally:
        os.environ.clear()
        os.environ.update(original)


def test_admission_input_binding(tmp: Path) -> None:
    path = tmp / "admission.json"
    value = {
        "authority": close.AUTHORITY,
        "classification": "ADMISSION_PRECONDITION_SATISFIED",
        "detail": "exact_hosted_wcare47q_lock_admission_verified",
        "wcare47q_run_id": close.WCARE47Q_RUN_ID,
        "wcare47q_run_number": "4",
        "wcare47q_run_attempt": "1",
        "wcare47q_head": close.WCARE47Q_HEAD,
        "final_subject": close.FINAL,
        "lock_sha256": close.LOCK_SHA256,
        "lock_git_blob": close.LOCK_BLOB,
        "wcare47_result_sha256": "ab" * 32,
        "lock_admitted": True,
        "execution_inputs_closed": False,
        "wcare42_executable_qualification_established": False,
        "runtime_authority_granted": False,
    }
    write_json(path, value)
    parsed, digest = close.validate_admission(path)
    assert parsed["lock_admitted"] is True
    assert len(digest) == 64

    value["wcare47q_run_id"] = "wrong"
    write_json(path, value)
    expect_exception(close.ClosureFailure, lambda: close.validate_admission(path), "admission_precondition_mismatch:wcare47q_run_id")


def test_external_directory_and_cargo_home_config_rejection(tmp: Path) -> None:
    repo = tmp / "repo-external"
    repo.mkdir()
    original = os.environ.get("CARGO_HOME")
    try:
        os.environ["CARGO_HOME"] = str(repo / "inside")
        expect_exception(
            close.ClosureFailure,
            lambda: close.assert_external_empty_dir(repo.resolve(), "CARGO_HOME"),
            "cargo_home_inside_repository",
        )
    finally:
        if original is None:
            os.environ.pop("CARGO_HOME", None)
        else:
            os.environ["CARGO_HOME"] = original

    cargo_home = tmp / "external-cargo-home"
    cargo_home.mkdir()
    assert close.assert_cargo_home_configs_absent(cargo_home, "fixture") == ["config.toml", "config"]
    (cargo_home / "config.toml").write_text("[net]\noffline=true\n", encoding="utf-8")
    expect_exception(
        close.ClosureFailure,
        lambda: close.assert_cargo_home_configs_absent(cargo_home, "fixture"),
        "cargo_home_config_present",
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="wcare49-selftest-") as raw:
        tmp = Path(raw)
        test_precondition_pending_and_identity_rejection(tmp)
        test_config_census_repository_and_ambient_rejection(tmp)
        test_checkout_cleanliness_rejects_ordinary_and_ignored_untracked(tmp)
        test_admission_input_binding(tmp)
        test_external_directory_and_cargo_home_config_rejection(tmp)
    test_duplicate_identical_hosted_result_rows_are_unambiguous()
    test_environment_override_rejection()
    print("PASS_WCARE49_FAIL_CLOSED_SELFTESTS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
