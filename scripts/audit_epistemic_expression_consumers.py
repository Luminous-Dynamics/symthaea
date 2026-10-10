#!/usr/bin/env python3
"""R5R5-R2 exact production-consumer witness audit.

This complements the broad file-level lexical discovery with exact production
witnesses. It does not attempt to classify every lexical hit as runtime code;
embedded #[cfg(test)] modules may make an otherwise production file appear in a
lexical surface. Runtime/authority conclusions require the witnesses below.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WITNESSES: dict[str, dict[str, tuple[str, ...]]] = {
    "coding_confidence_to_generic_llm_context": {
        "src/coding_agent/generation.rs": (
            "let confidence = self.cognitive_loop.prediction_confidence()",
            "Self::confidence_to_epistemic(confidence)",
            "ConsciousnessContext {",
            "epistemic_status: format!(\"{:?}\", epistemic)",
            "phi: current_phi",
            "consciousness_context: Some(consciousness_ctx)",
        ),
    },
    "generic_llm_context_to_system_prompt": {
        "src/language/llm_backend.rs": (
            "pub struct ConsciousnessContext",
            "pub fn to_system_supplement(&self) -> String",
            "EPISTEMIC_STATUS: {}",
            "CONSCIOUSNESS_LEVEL: {:.2}",
            "ctx.to_system_supplement()",
        ),
    },
    "generic_llm_context_is_internal_not_admission": {
        "src/language/llm_backend.rs": (
            "High type_confidence",
            "High error_likelihood",
            "Low epistemic confidence",
            "temperature_adjustment",
        ),
    },
    "verified_generation_property_split": {
        "src/language/verified_generation.rs": (
            "pub struct VerifiedCode",
            "pub compiled: bool",
            "pub tests_passed: bool",
            "pub formally_verified: Option<bool>",
            "pub attestation: Option<ProofAttestation>",
            "pub struct VerificationConfidence",
        ),
    },
    "verified_generation_overbroad_guarantee_vocabulary": {
        "src/language/verified_generation.rs": (
            "Whether this code meets the \"guaranteed correct\" bar",
            "pub fn is_guaranteed(&self) -> bool",
            "self.compiled && self.tests_passed",
            "generated code is verified before returning",
            "proven to compile and pass its own tests",
        ),
    },
    "verified_generation_real_execution_fail_closed": {
        "src/language/verified_generation.rs": (
            "if !executor.supports_real_execution()",
            "simulation mode cannot claim compilation or test verification",
            "if result.simulated",
        ),
    },
    "formal_verification_is_separate_optional_property": {
        "src/language/verified_generation.rs": (
            "formally_verified = Some(is_proven)",
            "Formal SMT contract was refuted",
            "SMT proof required",
            "SovereignAttestor::attest_with_process_key",
        ),
    },
}


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")


def main() -> int:
    missing: dict[str, list[str]] = {}
    for group, files in WITNESSES.items():
        absent: list[str] = []
        for rel, needles in files.items():
            path = ROOT / rel
            if not path.exists():
                absent.append(f"{rel}:<missing-file>")
                continue
            text = read(rel)
            for needle in needles:
                if needle not in text:
                    absent.append(f"{rel}:{needle}")
        if absent:
            missing[group] = absent

    summary = {
        "schema": "epi-auth-flow-001-r5r5-r2-production-witness-v1",
        "authority_scope": "measurement-only-exact-production-consumer-witnesses",
        "lexical_surface_semantics": "conservative-file-level-membership-may-include-embedded-cfg-test-code",
        "witness_groups": sorted(WITNESSES),
        "missing_witnesses": missing,
    }
    result = "PASS_CONSUMER_WITNESS" if not missing else "REVIEW_REQUIRED"
    summary["result"] = result

    print("schema=epi-auth-flow-001-r5r5-r2-production-witness-v1")
    print("authority_scope=measurement-only-exact-production-consumer-witnesses")
    print("lexical_surface_semantics=conservative-file-level-membership-may-include-embedded-cfg-test-code")
    for group in sorted(WITNESSES):
        print(f"WITNESS {group}")
    print("SUMMARY " + json.dumps(summary, sort_keys=True, separators=(",", ":")))
    print(f"result={result}")
    return 0 if result == "PASS_CONSUMER_WITNESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
