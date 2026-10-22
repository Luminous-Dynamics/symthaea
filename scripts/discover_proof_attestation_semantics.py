#!/usr/bin/env python3
"""EPI-SEM-001F1 measurement-only ProofAttestation trust-boundary discovery.

Freezes the current formal-proof attestation transcript, signer-key handling,
subject binding, and production call graph. This program grants no trust or
proof authority.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {
    ".git",
    "target",
    "vendor",
    "node_modules",
    ".direnv",
    "result",
    "examples",
    "tests",
    "benches",
    "patches",
}

ATTESTOR_PATH = "src/language/sovereign_attestor.rs"
VERIFIED_GENERATION_PATH = "src/language/verified_generation.rs"
EXPECTED_ATTESTATION_REFERENCE_FILES = [ATTESTOR_PATH, VERIFIED_GENERATION_PATH]
EXPECTED_FIELDS = [
    "label",
    "smtlib2_hash",
    "binary_hash",
    "verdict",
    "signature",
    "public_key",
]

REQUIRED_WITNESSES: dict[str, dict[str, tuple[str, ...]]] = {
    "attestation_record": {
        ATTESTOR_PATH: (
            "A signed attestation of a formal proof and its compiled binary realization.",
            "#[derive(Debug, Clone, Serialize, Deserialize)]",
            "pub struct ProofAttestation",
            "pub smtlib2_hash: [u8; 32]",
            "pub binary_hash: Option<[u8; 32]>",
            "pub verdict: String",
            "pub signature: Vec<u8>",
            "pub public_key: Vec<u8>",
        ),
    },
    "environment_or_ephemeral_key": {
        ATTESTOR_PATH: (
            "SYMTHAEA_ATTESTOR_PUBLIC_KEY_HEX",
            "SYMTHAEA_ATTESTOR_SECRET_KEY_HEX",
            "if let (Ok(public), Ok(secret)) = (public, secret)",
            "if let Ok(keys) = DilithiumKeypair::from_bytes(public, secret)",
            "return Self { keys };",
            "Self::new()",
        ),
    },
    "signed_message_construction": {
        ATTESTOR_PATH: (
            "msg.extend_from_slice(label.as_bytes())",
            "msg.extend_from_slice(&smt_hash)",
            "if let Some(bh) = binary_hash",
            "msg.extend_from_slice(&bh)",
            "msg.extend_from_slice(verdict.as_bytes())",
            "self.keys.sign(&msg).unwrap_or_default()",
        ),
    },
    "verification_shape": {
        ATTESTOR_PATH: (
            "pub fn verify(attestation: &ProofAttestation, actual_binary: Option<&[u8]>) -> bool",
            "if let Some(expected_hash) = attestation.binary_hash",
            "if let Some(binary) = actual_binary",
            "actual_hash != expected_hash",
            "msg.extend_from_slice(attestation.label.as_bytes())",
            "msg.extend_from_slice(&attestation.smtlib2_hash)",
            "verify_signature(&msg, &attestation.signature, &attestation.public_key)",
        ),
    },
    "formal_generation_caller": {
        VERIFIED_GENERATION_PATH: (
            "let is_proven = verdict == super::proof_memory::ProofVerdict::Proven",
            "attestation = Some(SovereignAttestor::attest_with_process_key(",
            "pub attestation: Option<ProofAttestation>",
        ),
    },
}


def production_rust_files() -> Iterable[Path]:
    for path in ROOT.rglob("*.rs"):
        rel = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in rel.parts):
            continue
        yield path


def path_set_digest(paths: list[str]) -> str:
    return hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()


def symbol_reference_files(symbol: str) -> list[str]:
    found: list[str] = []
    for path in production_rust_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if symbol in text:
            found.append(path.relative_to(ROOT).as_posix())
    return sorted(found)


def external_verify_call_files() -> list[str]:
    found: list[str] = []
    needle = "SovereignAttestor::verify("
    for path in production_rust_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel == ATTESTOR_PATH:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if needle in text:
            found.append(rel)
    return sorted(found)


def proof_attestation_fields() -> list[str]:
    text = (ROOT / ATTESTOR_PATH).read_text(encoding="utf-8", errors="replace")
    match = re.search(r"pub struct ProofAttestation\s*\{(?P<body>.*?)\n\}", text, re.S)
    if not match:
        return []
    return re.findall(r"^\s*pub\s+([A-Za-z_][A-Za-z0-9_]*)\s*:", match.group("body"), re.M)


def check_witnesses() -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for group, files in REQUIRED_WITNESSES.items():
        absent: list[str] = []
        for rel, needles in files.items():
            path = ROOT / rel
            if not path.exists():
                absent.append(f"{rel}:<missing-file>")
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for needle in needles:
                if needle not in text:
                    absent.append(f"{rel}:{needle}")
        if absent:
            missing[group] = absent
    return missing


def main() -> int:
    refs = symbol_reference_files("ProofAttestation")
    verify_call_files = external_verify_call_files()
    fields = proof_attestation_fields()
    missing_witnesses = check_witnesses()

    reference_set_changed = refs != EXPECTED_ATTESTATION_REFERENCE_FILES
    field_set_changed = fields != EXPECTED_FIELDS
    unexpected_verify_consumers = bool(verify_call_files)

    print("schema=epi-sem-001f1-proof-attestation-discovery-v1")
    print(
        "authority_scope="
        "measurement-only-proof-subject-binding-signer-trust-and-verifier-consumer-boundary"
    )
    print("production_reference_semantics=conservative-file-level-symbol-membership")
    print(f"ATTESTATION_REFERENCE_SET count={len(refs)} digest={path_set_digest(refs)}")
    for rel in refs:
        print(f"REFERENCE_FILE {rel}")
    print(f"VERIFY_CONSUMER_SET count={len(verify_call_files)} digest={path_set_digest(verify_call_files)}")
    for rel in verify_call_files:
        print(f"VERIFY_CONSUMER_FILE {rel}")
    print("STRUCT_FIELDS " + json.dumps(fields, separators=(",", ":")))

    summary = {
        "schema": "epi-sem-001f1-proof-attestation-discovery-v1",
        "authority_scope": (
            "measurement-only-proof-subject-binding-signer-trust-and-verifier-consumer-boundary"
        ),
        "production_reference_semantics": "conservative-file-level-symbol-membership",
        "attestation_reference_files": refs,
        "attestation_reference_set_digest": path_set_digest(refs),
        "expected_attestation_reference_files": EXPECTED_ATTESTATION_REFERENCE_FILES,
        "reference_set_changed": reference_set_changed,
        "struct_fields": fields,
        "expected_struct_fields": EXPECTED_FIELDS,
        "field_set_changed": field_set_changed,
        "external_verify_call_files": verify_call_files,
        "unexpected_verify_consumers": unexpected_verify_consumers,
        "missing_witnesses": missing_witnesses,
        "signature_uses_embedded_public_key": True,
        "trusted_signer_admission_established": False,
        "actual_smt_subject_rebound_by_verify": False,
        "binary_required_when_commitment_present": False,
        "stable_provisioned_signer_required": False,
        "deserialization_establishes_verified_typestate": False,
        "generic_code_correctness_established": False,
    }

    result = (
        "PASS_DISCOVERY"
        if not reference_set_changed
        and not field_set_changed
        and not unexpected_verify_consumers
        and not missing_witnesses
        else "REVIEW_REQUIRED"
    )
    summary["result"] = result
    print("SUMMARY " + json.dumps(summary, sort_keys=True, separators=(",", ":")))
    print(f"result={result}")
    return 0 if result == "PASS_DISCOVERY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
