#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import tempfile

SCRIPT = pathlib.Path(__file__).with_name("vart_hidden_benchmark_commit_reveal.py")
spec = importlib.util.spec_from_file_location("vart_commit_reveal", SCRIPT)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def source() -> dict:
    return {
        "schema": mod.SCHEMA_SOURCE,
        "campaign_id": "VART-002-HIDDEN-R1",
        "custodian_id": "custodian-independent-a",
        "fixtures": [
            {"fixture_id": "hidden-fixture-alpha", "nonce_hex": "11" * 32},
            {"fixture_id": "hidden-fixture-beta", "nonce_hex": "22" * 32},
        ],
        "seeds": [
            {"seed": 910001, "nonce_hex": "33" * 32},
            {"seed": 910002, "nonce_hex": "44" * 32},
        ],
    }


def raw(value: dict) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def manifest(value: dict) -> dict:
    return mod.public_manifest_from_source(value, raw(value))


def expect_reject(fn, name: str) -> None:
    try:
        fn()
    except mod.CommitRevealError:
        return
    raise AssertionError(f"{name} should reject")


def main() -> None:
    clean = source()
    public = manifest(clean)
    mod.validate_public_manifest(public)
    mod.verify_reveal(public, clean, raw(clean))

    changed_fixture = copy.deepcopy(clean)
    changed_fixture["fixtures"][0]["fixture_id"] = "hidden-fixture-rewritten"
    expect_reject(
        lambda: mod.verify_reveal(public, changed_fixture, raw(changed_fixture)),
        "C1 fixture mutation",
    )

    changed_seed = copy.deepcopy(clean)
    changed_seed["seeds"][1]["seed"] += 1
    expect_reject(
        lambda: mod.verify_reveal(public, changed_seed, raw(changed_seed)),
        "C2 seed mutation",
    )

    changed_nonce = copy.deepcopy(clean)
    changed_nonce["fixtures"][1]["nonce_hex"] = "55" * 32
    expect_reject(
        lambda: mod.verify_reveal(public, changed_nonce, raw(changed_nonce)),
        "C3 nonce substitution",
    )

    reused_nonce = copy.deepcopy(clean)
    reused_nonce["seeds"][0]["nonce_hex"] = reused_nonce["fixtures"][0]["nonce_hex"]
    expect_reject(lambda: mod.normalize_source(reused_nonce), "C4 nonce reuse")

    cross_campaign = copy.deepcopy(clean)
    cross_campaign["campaign_id"] = "VART-002-HIDDEN-R2"
    expect_reject(
        lambda: mod.verify_reveal(public, cross_campaign, raw(cross_campaign)),
        "C5 cross-campaign replay",
    )

    leaked_public = copy.deepcopy(public)
    leaked_public["seeds"] = [910001]
    expect_reject(lambda: mod.validate_public_manifest(leaked_public), "C6 public seed leak")

    authority_escalation = copy.deepcopy(public)
    authority_escalation["confirmatory_execution_authorized"] = True
    expect_reject(
        lambda: mod.validate_public_manifest(authority_escalation),
        "C7 authority escalation",
    )

    reordered = copy.deepcopy(public)
    reordered["fixture_commitments_sha256"] = list(
        reversed(reordered["fixture_commitments_sha256"])
    )
    expect_reject(lambda: mod.validate_public_manifest(reordered), "C8 ordering drift")

    # Exercise the actual file-based CLI helpers enough to ensure raw-byte source
    # hashing and canonical public output compose correctly.
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        source_path = root / "hidden.json"
        public_path = root / "public.json"
        source_path.write_bytes(raw(clean))
        args = type("Args", (), {"source": str(source_path), "public_out": str(public_path)})()
        assert mod.cmd_commit(args) == 0
        generated, _ = mod.read_json(public_path)
        verify_args = type(
            "Args",
            (),
            {"public": str(public_path), "reveal": str(source_path)},
        )()
        assert mod.cmd_verify(verify_args) == 0
        assert generated["private_source_sha256"] == mod.sha256_bytes(source_path.read_bytes())

    print("PASS: hidden benchmark commit/reveal acceptance + C1-C8 deterministic rejection")


if __name__ == "__main__":
    main()
