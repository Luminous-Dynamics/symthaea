#!/usr/bin/env python3
"""EPI-SEM-001E1-R2 measurement-only CodeCertificate semantics discovery.

This program freezes the current distinction between content identity/tamper
checking, recorded verification metadata, and authenticated producer provenance.
It does not authenticate a certificate and does not establish code correctness.

R2 fixes the failed R1 witness matcher by canonicalizing whitespace before
substring comparison. File membership, field order, auth-field absence, and
claim ceilings remain exact.
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

CERT_PATH = "src/language/code_certificate.rs"
ORCH_PATH = "src/language/code_orchestrator.rs"
PLAN_PATH = "docs/CODE_ABILITY_IMPROVEMENT_PLAN.md"
EXPECTED_PRODUCTION_REFERENCE_FILES = [CERT_PATH, ORCH_PATH]
EXPECTED_FIELDS = [
    "id",
    "source_hash",
    "backend_used",
    "source_provenance",
    "semantic_similarity",
    "verification_layers",
    "epistemic_status",
    "safety_critical",
    "timestamp",
    "topology",
    "oracle_convergence",
    "sheaf_coherent",
]
AUTHENTICATION_FIELD_NAMES = {
    "signature",
    "signatures",
    "issuer",
    "issuer_id",
    "signer",
    "signer_id",
    "public_key",
    "verification_key",
    "key_id",
    "attestation",
    "authentication_receipt",
    "authenticated_by",
}

REQUIRED_WITNESSES: dict[str, dict[str, tuple[str, ...]]] = {
    "certificate_definition_and_claim_language": {
        CERT_PATH: (
            "Code Certificate — Machine-Verifiable Audit Trail",
            "a cryptographic receipt proving *how* the code",
            "pub struct CodeCertificate",
            "pub source_hash: [u8; 32]",
            "pub verification_layers: Vec<CertVerificationLayer>",
            "pub epistemic_status: String",
            "pub timestamp: u64",
        ),
    },
    "content_commitment_and_tamper_check": {
        CERT_PATH: (
            "let source_hash = blake3::hash(source.as_bytes())",
            "let id_input = format!(\"{}{}\", source, timestamp)",
            "let id_hash = blake3::hash(id_input.as_bytes())",
            "pub fn verify_source(&self, source: &str) -> bool",
            "let hash = blake3::hash(source.as_bytes())",
            "hash.as_bytes() == &self.source_hash",
        ),
    },
    "serialization_surface": {
        CERT_PATH: (
            "#[derive(Debug, Clone, Serialize, Deserialize)]",
            "pub fn to_json(&self) -> String",
            "serde_json::to_string_pretty(self)",
        ),
    },
    "orchestrator_issue_and_storage": {
        ORCH_PATH: (
            "certificates: Vec<CodeCertificate>",
            "fn issue_certificate(",
            "CodeCertificate::new(source, backend, similarity)",
            ".with_epistemic_status(request.epistemic_status)",
            ".with_verification_layers(verification_layers)",
            "self.state.lock().certificates.push(cert.clone())",
            "pub fn certificates(&self) -> Vec<CodeCertificate>",
            "self.state.lock().certificates.clone()",
        ),
    },
    "post_acceptance_metadata_boundary": {
        ORCH_PATH: (
            "This is deliberately post-acceptance metadata: compiler/test verification",
            "remains the acceptance gate",
        ),
    },
    "existing_internal_review": {
        PLAN_PATH: (
            "Certificates are generated, held in an in-memory `Vec`, and discarded when the",
            "no file write, no API, no CLI surface reads them back",
        ),
    },
}


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")


def production_rust_files() -> Iterable[Path]:
    for path in ROOT.rglob("*.rs"):
        rel = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in rel.parts):
            continue
        yield path


def path_set_digest(paths: list[str]) -> str:
    return hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()


def canonicalize_whitespace(text: str) -> str:
    """Collapse Unicode whitespace runs without changing non-whitespace bytes."""
    return " ".join(text.split())


def code_certificate_reference_files() -> list[str]:
    found: list[str] = []
    for path in production_rust_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if "CodeCertificate" in text:
            found.append(path.relative_to(ROOT).as_posix())
    return sorted(found)


def code_certificate_fields() -> list[str]:
    text = read(CERT_PATH)
    match = re.search(r"pub struct CodeCertificate\s*\{(?P<body>.*?)\n\}", text, re.S)
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
            text = canonicalize_whitespace(
                path.read_text(encoding="utf-8", errors="replace")
            )
            for needle in needles:
                if canonicalize_whitespace(needle) not in text:
                    absent.append(f"{rel}:{needle}")
        if absent:
            missing[group] = absent
    return missing


def main() -> int:
    refs = code_certificate_reference_files()
    fields = code_certificate_fields()
    auth_fields = sorted(set(fields) & AUTHENTICATION_FIELD_NAMES)
    missing_witnesses = check_witnesses()

    reference_set_changed = refs != EXPECTED_PRODUCTION_REFERENCE_FILES
    field_set_changed = fields != EXPECTED_FIELDS

    print("schema=epi-sem-001e1-code-certificate-discovery-v2")
    print(
        "authority_scope="
        "measurement-only-content-integrity-verification-metadata-and-authentication-boundary"
    )
    print("production_reference_semantics=conservative-file-level-symbol-membership")
    print("witness_matching=whitespace-canonicalized-substring")
    print(f"PRODUCTION_REFERENCE_SET count={len(refs)} digest={path_set_digest(refs)}")
    for rel in refs:
        print(f"REFERENCE_FILE {rel}")
    print("STRUCT_FIELDS " + json.dumps(fields, separators=(",", ":")))
    print("AUTHENTICATION_FIELDS " + json.dumps(auth_fields, separators=(",", ":")))

    summary = {
        "schema": "epi-sem-001e1-code-certificate-discovery-v2",
        "authority_scope": (
            "measurement-only-content-integrity-verification-metadata-and-authentication-boundary"
        ),
        "production_reference_semantics": "conservative-file-level-symbol-membership",
        "witness_matching": "whitespace-canonicalized-substring",
        "production_reference_files": refs,
        "production_reference_set_digest": path_set_digest(refs),
        "expected_production_reference_files": EXPECTED_PRODUCTION_REFERENCE_FILES,
        "reference_set_changed": reference_set_changed,
        "struct_fields": fields,
        "expected_struct_fields": EXPECTED_FIELDS,
        "field_set_changed": field_set_changed,
        "authentication_fields_present": auth_fields,
        "missing_witnesses": missing_witnesses,
        "content_identity_established_by_current_record": True,
        "issuer_authentication_established_by_current_record": False,
        "trusted_chronology_established_by_current_record": False,
        "verification_reexecution_established_by_current_record": False,
        "generic_code_correctness_established_by_current_record": False,
    }

    result = (
        "PASS_DISCOVERY"
        if not reference_set_changed
        and not field_set_changed
        and not auth_fields
        and not missing_witnesses
        else "REVIEW_REQUIRED"
    )
    summary["result"] = result
    print("SUMMARY " + json.dumps(summary, sort_keys=True, separators=(",", ":")))
    print(f"result={result}")
    return 0 if result == "PASS_DISCOVERY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
