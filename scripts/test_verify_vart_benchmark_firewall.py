#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import verify_vart_benchmark_firewall as fw

A = "a" * 64
B = "b" * 64
C = "c" * 64
D = "d" * 64
E = "e" * 64
F = "f" * 64


def base(hidden_root: str) -> dict:
    return {
        "schema": fw.SCHEMA,
        "firewall_id": "test-firewall",
        "status": "prospective_committed",
        "development_domain_tag": "SYMTHAEA-DEVART-v1",
        "evaluation_domain_tag": "SYMTHAEA-VART-HIDDEN-v1",
        "devart": {
            "benchmark_ids": ["dev-1"],
            "fixture_commitments_sha256": [A],
            "seed_commitments_sha256": [B],
            "used_as_development_feedback": True,
        },
        "vart": {
            "campaign_id": "vart-hidden-1",
            "benchmark_custodian_id": "custodian-test",
            "fixture_count": 1,
            "seed_count": 1,
            "fixture_commitments_sha256": [C],
            "seed_commitments_sha256": [D],
            "generator_policy_sha256": E,
            "scoring_contract_sha256": F,
            "custodian_receipt_sha256": "1" * 64,
            "hidden_material_root": hidden_root,
            "revealed": False,
            "plaintext_fixture_ids": [],
            "plaintext_seeds": [],
            "expected_solutions": [],
            "target_defects": [],
            "trap_labels": [],
        },
        "reuse_policy": {
            "prior_vart_fixture_commitments_sha256": ["2" * 64],
            "prior_vart_seed_commitments_sha256": ["3" * 64],
            "spent_commitments_may_not_return_to_hidden_confirmatory_use": True,
        },
        "claim_authorized": False,
    }


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def expect_reject(repo: Path, manifest: Path, mutation, needle: str) -> None:
    value = base(str(repo.parent / "hidden"))
    mutation(value)
    write(manifest, value)
    try:
        fw.verify(manifest, repo)
    except fw.Reject as exc:
        assert needle in str(exc), (needle, str(exc))
        return
    raise AssertionError(f"expected rejection containing {needle}")


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        repo = root / "repo"
        repo.mkdir()
        manifest = root / "firewall.json"

        value = base(str(root / "hidden"))
        write(manifest, value)
        assert fw.verify(manifest, repo)["verdict"] == "VART_BENCHMARK_FIREWALL_PASS"

        expect_reject(repo, manifest, lambda x: x["vart"].update(fixture_commitments_sha256=[A]), "DEVART_VART_FIXTURE_OVERLAP")
        expect_reject(repo, manifest, lambda x: x["vart"].update(seed_commitments_sha256=[B]), "DEVART_VART_SEED_OVERLAP")
        expect_reject(repo, manifest, lambda x: x["vart"].update(hidden_material_root=str(repo / "secret")), "HIDDEN_MATERIAL_INSIDE_REPOSITORY")
        expect_reject(repo, manifest, lambda x: x["vart"].update(plaintext_seeds=[42]), "HIDDEN_BENCHMARK_SECRET_EXPOSED")
        expect_reject(repo, manifest, lambda x: x["reuse_policy"].update(prior_vart_fixture_commitments_sha256=[C]), "SPENT_VART_FIXTURE_REUSE")
        expect_reject(repo, manifest, lambda x: x.update(evaluation_domain_tag=x["development_domain_tag"]), "DOMAIN_SEPARATION_FAILURE")

    print("PASS: DEVART/VART firewall acceptance + adversarial rejection")


if __name__ == "__main__":
    main()
