import hashlib
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "verify_cargo_lock_additive_local_package.py"
spec = importlib.util.spec_from_file_location("verify_cargo_lock_additive_local_package", SCRIPT)
assert spec and spec.loader
lock = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lock)

BASE = {
    "version": 4,
    "package": [
        {
            "name": "blake3",
            "version": "1.8.2",
            "source": "registry+https://github.com/rust-lang/crates.io-index",
            "checksum": "a" * 64,
            "dependencies": ["arrayref"],
        },
        {
            "name": "symthaea-existing",
            "version": "0.1.0",
            "dependencies": ["blake3"],
        },
    ],
}


def candidate_with(package):
    return {"version": 4, "package": [*BASE["package"], package]}


def expected_package():
    return {
        "name": "symthaea-qualification-receipt-core",
        "version": "0.1.0",
        "dependencies": ["blake3"],
    }


def verify(candidate):
    return lock.verify_additive_local_package(
        BASE,
        candidate,
        expected_name="symthaea-qualification-receipt-core",
        expected_version="0.1.0",
        expected_dependencies=["blake3"],
    )


def test_exact_one_local_package_addition_passes():
    result = verify(candidate_with(expected_package()))
    assert result["disposition"] == "ExactAdditiveLocalPackageTransition"
    assert result["candidate_package_records"] == result["base_package_records"] + 1


def test_existing_registry_record_churn_fails():
    candidate = candidate_with(expected_package())
    candidate["package"][0] = dict(candidate["package"][0])
    candidate["package"][0]["checksum"] = "b" * 64
    with pytest.raises(lock.LockTransitionError, match="removed or changed"):
        verify(candidate)


def test_extra_package_fails():
    candidate = candidate_with(expected_package())
    candidate["package"].append({"name": "unexpected", "version": "0.1.0"})
    with pytest.raises(lock.LockTransitionError, match="exactly one"):
        verify(candidate)


def test_dependency_mismatch_fails():
    package = expected_package()
    package["dependencies"] = ["blake3", "serde"]
    with pytest.raises(lock.LockTransitionError, match="dependency set mismatch"):
        verify(candidate_with(package))


@pytest.mark.parametrize("field,value", [("source", "registry+example"), ("checksum", "c" * 64)])
def test_new_local_package_source_or_checksum_fails(field, value):
    package = expected_package()
    package[field] = value
    with pytest.raises(lock.LockTransitionError, match="source/checksum"):
        verify(candidate_with(package))


def test_top_level_lock_metadata_change_fails():
    candidate = candidate_with(expected_package())
    candidate["metadata"] = {"unexpected": "value"}
    with pytest.raises(lock.LockTransitionError, match="metadata changed"):
        verify(candidate)


def test_unsorted_or_duplicate_expected_dependencies_fail():
    candidate = candidate_with(expected_package())
    with pytest.raises(lock.LockTransitionError, match="sorted unique"):
        lock.verify_additive_local_package(
            BASE,
            candidate,
            expected_name="symthaea-qualification-receipt-core",
            expected_version="0.1.0",
            expected_dependencies=["serde", "blake3"],
        )


def test_file_receipt_binds_exact_base_and_candidate_bytes(tmp_path):
    base_bytes = (
        'version = 4\n\n'
        '[[package]]\nname = "blake3"\nversion = "1.8.2"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{"a" * 64}"\ndependencies = ["arrayref"]\n\n'
        '[[package]]\nname = "symthaea-existing"\nversion = "0.1.0"\n'
        'dependencies = ["blake3"]\n'
    ).encode()
    candidate_bytes = base_bytes + (
        '\n[[package]]\nname = "symthaea-qualification-receipt-core"\n'
        'version = "0.1.0"\ndependencies = ["blake3"]\n'
    ).encode()
    base_path = tmp_path / "base.lock"
    candidate_path = tmp_path / "candidate.lock"
    base_path.write_bytes(base_bytes)
    candidate_path.write_bytes(candidate_bytes)

    result = lock.verify_lock_files(
        base_path,
        candidate_path,
        expected_name="symthaea-qualification-receipt-core",
        expected_version="0.1.0",
        expected_dependencies=["blake3"],
    )

    assert result["schema"] == "symthaea.cargo-lock-additive-local-package-transition.v1"
    assert result["base_lock_sha256"] == "sha256:" + hashlib.sha256(base_bytes).hexdigest()
    assert result["candidate_lock_sha256"] == "sha256:" + hashlib.sha256(candidate_bytes).hexdigest()


def test_duplicate_of_preexisting_expected_local_package_fails():
    existing = expected_package()
    base = {"version": 4, "package": [*BASE["package"], existing]}
    candidate = {"version": 4, "package": [*base["package"], dict(existing)]}
    with pytest.raises(lock.LockTransitionError, match="already contains expected local package"):
        lock.verify_additive_local_package(
            base,
            candidate,
            expected_name="symthaea-qualification-receipt-core",
            expected_version="0.1.0",
            expected_dependencies=["blake3"],
        )
