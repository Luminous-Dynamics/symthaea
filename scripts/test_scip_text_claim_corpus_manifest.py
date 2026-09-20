#!/usr/bin/env python3
"""Independent hostile-input harness for the V19 corpus manifest contract."""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "scip_text_claim_corpus_manifest.py"
POLICY = ROOT / "scripts" / "qualification" / "scip_text_claim_corpus_manifest_policy_v1.json"
POLICY_SHA = "9a30ba36b3e872aaec349647f71433579bc0932f24c96335bb32afe77498ffc0"
V18_SHA = "96e2ec5e1fad213f4261405c22d1f80cb6f1d106fd5621481eb0e95a6cca4530"
CASE_DOMAIN = b"symthaea-scip-text-claim-case-v1\0"
SEED_DOMAIN = b"symthaea-scip-text-claim-seed-commitment-v1\0"
RANK_DOMAIN = b"symthaea-scip-text-claim-split-rank-v1\0"
DIMENSIONS = (
    "entity-reference", "relation-direction", "numeric-value-and-unit", "polarity-and-negation",
    "quantifier-and-cardinality", "temporal-scope", "epistemic-modality", "attribution-and-source",
    "causal-strength", "unsupported-additions", "required-detail-coverage",
)
FAMILIES = (
    "long-distance-coreference", "cross-sentence-negation", "nested-attribution", "temporal-contrast",
    "multiple-entities-same-type", "multiple-numbers-same-unit", "causal-versus-correlational-contrast",
    "mixed-certain-and-uncertain-claims",
)
CASE_ID_FIELDS = (
    "kind", "dimension", "polarity", "discourse_family", "surface_sha256",
    "source_inventory_sha256", "expected_inventory_sha256", "annotation_receipt_sha256",
    "template_sha256", "named_entity_tuple_sha256", "numeric_tuple_sha256", "exact_sentence_sha256",
)
SEED = bytes(range(32))
SEED_HEX = SEED.hex()


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def case_id(case: dict) -> str:
    return hashlib.sha256(CASE_DOMAIN + canonical({k: case[k] for k in CASE_ID_FIELDS})).hexdigest()


def rank(key_hex: str) -> bytes:
    return hmac.new(SEED, RANK_DOMAIN + bytes.fromhex(key_hex), hashlib.sha256).digest()


def stratum(case: dict) -> tuple[str, ...]:
    return ("dimension", case["dimension"], case["polarity"]) if case["kind"] == "dimension" else ("discourse", case["discourse_family"])


def build_manifest() -> dict:
    cases = []
    counter = 0
    def add(kind, dimension, polarity, family):
        nonlocal counter
        tag = f"hostile-{counter:04d}"
        counter += 1
        case = {
            "assignment_key_sha256": digest("assign:" + tag), "case_id": "", "kind": kind,
            "dimension": dimension, "polarity": polarity, "discourse_family": family,
            "surface_sha256": digest("surface:" + tag), "source_inventory_sha256": digest("source:" + tag),
            "expected_inventory_sha256": digest("expected:" + tag), "annotation_receipt_sha256": digest("annotation:" + tag),
            "template_sha256": digest("template:" + tag), "named_entity_tuple_sha256": digest("entities:" + tag),
            "numeric_tuple_sha256": digest("numbers:" + tag), "exact_sentence_sha256": digest("sentence:" + tag),
            "split": "calibration",
        }
        case["case_id"] = case_id(case)
        cases.append(case)
    for dimension in DIMENSIONS:
        for polarity in ("positive", "negative"):
            for _ in range(52): add("dimension", dimension, polarity, None)
    for family in FAMILIES:
        for _ in range(27): add("discourse", None, None, family)

    strata = defaultdict(list)
    for case in cases: strata[stratum(case)].append(case)
    for key, rows in strata.items():
        rows.sort(key=lambda c: (rank(c["assignment_key_sha256"]), c["assignment_key_sha256"]))
        quota = 20 if key[0] == "dimension" else 11
        for i, case in enumerate(rows): case["split"] = "calibration" if i < quota else "confirmatory"
    return {
        "schema": "symthaea.scip-text-claim-corpus-manifest/v1",
        "authority": "manifest-contract-only",
        "policy_semantic_sha256": POLICY_SHA,
        "v18_preregistration_sha256": V18_SHA,
        "seed_commitment_sha256": hashlib.sha256(SEED_DOMAIN + SEED).hexdigest(),
        "cases": cases,
    }


def execute(manifest: dict, seed_hex: str | None = SEED_HEX) -> tuple[int, str, str]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "manifest.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False), "utf-8")
        cmd = [sys.executable, "-B", str(VALIDATOR), str(path), "--policy", str(POLICY)]
        if seed_hex is not None:
            cmd.extend(["--seed-hex", seed_hex])
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return proc.returncode, proc.stdout, proc.stderr


def reject(manifest: dict, name: str, seed_hex: str | None = SEED_HEX) -> None:
    code, out, err = execute(manifest, seed_hex)
    if code == 0:
        raise AssertionError(f"{name}: hostile manifest accepted: {out[:300]}")
    if "ERROR:" not in err:
        raise AssertionError(f"{name}: rejection lacked explicit error: {err}")


def main() -> int:
    manifest = build_manifest()
    code, out, err = execute(manifest)
    assert code == 0, err
    result = json.loads(out)
    assert result["case_count"] == 1360
    assert result["calibration_count"] == 528
    assert result["confirmatory_count"] == 832
    assert result["split_reconstruction_verified"] is True
    assert result["surface_fidelity_established"] is False
    assert result["confirmatory_execution_authorized"] is False

    code, out, err = execute(manifest, None)
    assert code == 0, err
    assert json.loads(out)["split_reconstruction_verified"] is False

    reject(manifest, "wrong-seed", (bytes(reversed(range(32)))).hex())

    changed = copy.deepcopy(manifest)
    changed["cases"][0]["case_id"] = digest("forged-case-id")
    reject(changed, "case-id-tamper")

    changed = copy.deepcopy(manifest)
    changed["cases"][0]["surface_sha256"] = digest("tampered-surface")
    reject(changed, "content-tamper-without-rehash")

    changed = copy.deepcopy(manifest)
    changed["cases"][1]["assignment_key_sha256"] = changed["cases"][0]["assignment_key_sha256"]
    reject(changed, "duplicate-assignment-key", None)

    # Keep all visible counts identical while swapping two seed-selected assignments
    # inside the same stratum. Only rank reconstruction should catch this.
    changed = copy.deepcopy(manifest)
    same = [c for c in changed["cases"] if c["kind"] == "dimension" and c["dimension"] == DIMENSIONS[0] and c["polarity"] == "positive"]
    cal = next(c for c in same if c["split"] == "calibration")
    conf = next(c for c in same if c["split"] == "confirmatory")
    cal["split"], conf["split"] = conf["split"], cal["split"]
    code, _, err = execute(changed, None)
    assert code == 0, err
    reject(changed, "same-stratum-split-swap", SEED_HEX)

    # Manufacture cross-split content collisions while recomputing case identity,
    # proving leakage gates are independent of case-id integrity.
    changed = copy.deepcopy(manifest)
    cal = next(c for c in changed["cases"] if c["split"] == "calibration")
    conf = next(c for c in changed["cases"] if c["split"] == "confirmatory")
    conf["template_sha256"] = cal["template_sha256"]
    conf["case_id"] = case_id(conf)
    reject(changed, "cross-split-template-leak", None)

    for field in ("named_entity_tuple_sha256", "numeric_tuple_sha256", "exact_sentence_sha256"):
        changed = copy.deepcopy(manifest)
        cal = next(c for c in changed["cases"] if c["split"] == "calibration")
        conf = next(c for c in changed["cases"] if c["split"] == "confirmatory")
        conf[field] = cal[field]
        conf["case_id"] = case_id(conf)
        reject(changed, f"cross-split-{field}", None)

    changed = copy.deepcopy(manifest)
    changed["policy_semantic_sha256"] = digest("wrong-policy")
    reject(changed, "policy-substitution", None)

    changed = copy.deepcopy(manifest)
    changed["v18_preregistration_sha256"] = digest("wrong-v18")
    reject(changed, "v18-substitution", None)

    changed = copy.deepcopy(manifest)
    changed["surface_fidelity_established"] = True
    reject(changed, "unknown-authority-field", None)

    raw = json.dumps(manifest, separators=(",", ":"))
    raw = raw.replace('"authority":"manifest-contract-only"', '"authority":"manifest-contract-only","authority":"qualified"', 1)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "manifest.json"
        path.write_text(raw, "utf-8")
        proc = subprocess.run([sys.executable, "-B", str(VALIDATOR), str(path), "--policy", str(POLICY)], capture_output=True, text=True, check=False)
        assert proc.returncode != 0

    print(f"PASS_CORPUS_MANIFEST_ADVERSARIAL policy_sha256={POLICY_SHA} cases=1360")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
