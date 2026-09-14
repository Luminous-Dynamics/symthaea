#!/usr/bin/env python3
"""Adversarial helper campaign for WCARE-48V."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "scripts/wcare48v_verify_final.py"

spec = importlib.util.spec_from_file_location("wcare48v", VERIFIER)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def expect_invalid(fn, marker: str) -> None:
    try:
        fn()
    except mod.InvalidFinal as exc:
        assert marker in str(exc), (marker, str(exc))
    else:
        raise AssertionError(f"expected InvalidFinal containing {marker}")


def receipt(lock_format: int = 4, package_count: int = 2) -> dict:
    out = {key: None for key in mod.EXPECTED_RECEIPT_KEYS}
    out["lock_format"] = lock_format
    out["package_count"] = package_count
    return out


def main() -> int:
    expect_invalid(
        lambda: mod.strict_json(b'{"a":1,"a":2}'),
        "duplicate_receipt_key:a",
    )
    expect_invalid(
        lambda: mod.strict_json(b'{"authority":"MeasurementOnly"}'),
        "receipt_keyset_mismatch",
    )

    good = b'''version = 4\n\n[[package]]\nname = "wcare42-builder-attestation-verifier"\nversion = "0.1.0"\n\n[[package]]\nname = "dep"\nversion = "1.0.0"\nsource = "registry+https://github.com/rust-lang/crates.io-index"\nchecksum = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"\n'''
    mod.validate_lock(good, receipt())

    expect_invalid(
        lambda: mod.validate_lock(good, receipt(lock_format=3)),
        "receipt_lock_format_not_v4",
    )
    expect_invalid(
        lambda: mod.validate_lock(good, receipt(package_count=3)),
        "package_count_mismatch",
    )

    bad_checksum = good.replace(b"a" * 64, b"short")
    expect_invalid(
        lambda: mod.validate_lock(bad_checksum, receipt()),
        "invalid_registry_checksum:dep",
    )

    bad_source = good.replace(
        b"registry+https://github.com/rust-lang/crates.io-index",
        b"git+https://example.invalid/repo",
    )
    expect_invalid(
        lambda: mod.validate_lock(bad_source, receipt()),
        "non_registry_dependency:dep",
    )

    no_root = good.replace(
        b'wcare42-builder-attestation-verifier',
        b'not-the-root-package',
    )
    expect_invalid(
        lambda: mod.validate_lock(no_root, receipt()),
        "non_registry_dependency:not-the-root-package",
    )

    print("PASS_WCARE48V_HELPER_ADVERSARIALS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
