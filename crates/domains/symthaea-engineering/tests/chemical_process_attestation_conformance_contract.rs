// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Structural contract tests for the design-only chemical-process attestation
// challenge inventory. These tests validate the challenge corpus, not a crypto
// verifier; CPAE cases are not considered executed attestation tests.

use serde_json::Value;
use std::collections::BTreeSet;

const CORPUS_JSON: &str = include_str!(
    "../../../../docs/engineering/fixtures/CHEMICAL_PROCESS_ATTESTATION_CONFORMANCE_V1.json"
);

fn corpus() -> Value {
    serde_json::from_str(CORPUS_JSON)
        .expect("chemical-process attestation conformance inventory must be valid JSON")
}

fn case<'a>(root: &'a Value, id: &str) -> &'a Value {
    root["cases"]
        .as_array()
        .expect("cases must be an array")
        .iter()
        .find(|entry| entry["id"].as_str() == Some(id))
        .unwrap_or_else(|| panic!("required conformance case {id} must exist"))
}

#[test]
fn attestation_inventory_is_versioned_and_explicitly_not_executable() {
    let root = corpus();
    assert_eq!(
        root["schema_version"].as_str(),
        Some("symthaea.chemical-process-attestation-conformance.v1")
    );
    assert_eq!(root["corpus_version"].as_u64(), Some(1));
    assert_eq!(
        root["status"].as_str(),
        Some("DESIGN_ONLY_NOT_EXECUTABLE"),
        "the inventory must not be mislabeled as tests already executed"
    );
    assert!(
        root["intent"]
            .as_str()
            .is_some_and(|text| text.contains("not evidence that tests exist, run, or pass")),
        "inventory must state its evidentiary limitation"
    );
}

#[test]
fn challenge_ids_are_unique_and_all_cases_are_hard_gated() {
    let root = corpus();
    let cases = root["cases"].as_array().expect("cases must be an array");
    assert!(
        cases.len() >= 28,
        "the initial adversarial inventory must not silently shrink"
    );

    let mut ids = BTreeSet::new();
    for entry in cases {
        let id = entry["id"]
            .as_str()
            .filter(|text| !text.trim().is_empty())
            .expect("every case needs a nonempty stable ID");
        assert!(ids.insert(id.to_owned()), "duplicate case ID: {id}");
        assert!(
            entry["hard_gate"].as_bool() == Some(true),
            "{id}: every listed challenge is a hard gate"
        );
        for field in ["class", "setup", "mutation", "expected_outcome", "rationale"] {
            assert!(
                entry[field].as_str().is_some_and(|text| !text.trim().is_empty()),
                "{id}: {field} must be specified"
            );
        }
        assert!(
            entry["expected_stages"].is_object(),
            "{id}: expected_stages must be an object"
        );
    }
}

#[test]
fn expected_outcomes_and_stage_statuses_are_from_declared_enums() {
    let root = corpus();
    let outcomes = root["accepted_outcomes"]
        .as_array()
        .expect("accepted_outcomes must be an array")
        .iter()
        .map(|value| value.as_str().expect("outcome must be a string").to_owned())
        .collect::<BTreeSet<_>>();
    let statuses = root["stage_statuses"]
        .as_array()
        .expect("stage_statuses must be an array")
        .iter()
        .map(|value| value.as_str().expect("stage status must be a string").to_owned())
        .collect::<BTreeSet<_>>();

    for entry in root["cases"].as_array().unwrap() {
        let id = entry["id"].as_str().unwrap();
        let outcome = entry["expected_outcome"]
            .as_str()
            .unwrap_or_else(|| panic!("{id}: expected_outcome must be a string"));
        assert!(outcomes.contains(outcome), "{id}: undeclared outcome {outcome}");

        for (stage, status) in entry["expected_stages"]
            .as_object()
            .unwrap()
        {
            let status = status
                .as_str()
                .unwrap_or_else(|| panic!("{id}.{stage}: stage status must be a string"));
            assert!(
                statuses.contains(status),
                "{id}.{stage}: undeclared stage status {status}"
            );
        }
    }
}

#[test]
fn minimum_adversarial_families_remain_present() {
    let root = corpus();
    for id in [
        "CPAE-001", // valid positive control
        "CPAE-002", // stale envelope digest
        "CPAE-003", // attacker recomputes digest but cannot reuse old signature
        "CPAE-004", // subject transplant
        "CPAE-006", // purpose confusion
        "CPAE-007", // signature corruption
        "CPAE-010", // unknown signer
        "CPAE-014", // revoked signer
        "CPAE-016", // stale trust snapshot
        "CPAE-017", // trust sequence rollback
        "CPAE-018", // same-sequence collision
        "CPAE-021", // historical cryptographic validity vs current authorization
        "CPAE-022", // NOT_RUN status must not be promoted
        "CPAE-025", // no rollback checkpoint
        "CPAE-028", // resource-bound checks
    ] {
        let _ = case(&root, id);
    }
}

#[test]
fn rehashing_an_envelope_does_not_authenticate_the_rewritten_subject() {
    let root = corpus();
    let challenge = case(&root, "CPAE-003");
    assert_eq!(challenge["class"].as_str(), Some("digest_rewrite"));
    assert_eq!(
        challenge["expected_outcome"].as_str(),
        Some("REJECT")
    );
    assert_eq!(
        challenge["expected_stages"]["envelope_integrity"].as_str(),
        Some("PASS"),
        "an attacker may recompute an internally consistent envelope digest"
    );
    assert_eq!(
        challenge["expected_stages"]["subject_binding"].as_str(),
        Some("FAIL"),
        "the old signature/statement must not authenticate a rewritten subject"
    );
}

#[test]
fn valid_signature_cannot_promote_an_unrun_scientific_case() {
    let root = corpus();
    let challenge = case(&root, "CPAE-022");
    assert_eq!(
        challenge["expected_outcome"].as_str(),
        Some("AUTHENTIC_ENDORSEMENT")
    );
    assert_eq!(challenge["scientific_status"].as_str(), Some("NOT_RUN"));
    assert_eq!(
        challenge["expected_stages"]["signature"].as_str(),
        Some("PASS")
    );
    assert_eq!(
        challenge["expected_stages"]["signer_lifecycle"].as_str(),
        Some("PASS")
    );
}

#[test]
fn stateless_verifier_must_not_claim_rollback_protection_without_checkpoint() {
    let root = corpus();
    let challenge = case(&root, "CPAE-025");
    assert_eq!(
        challenge["expected_outcome"].as_str(),
        Some("UNRESOLVED_NO_ROLLBACK_CHECKPOINT")
    );
    assert_eq!(
        challenge["expected_stages"]["trust_freshness"].as_str(),
        Some("UNRESOLVED")
    );
}

#[test]
fn cryptographic_validity_and_current_authorization_are_separate_stages() {
    let root = corpus();
    let challenge = case(&root, "CPAE-021");
    assert_eq!(
        challenge["expected_outcome"].as_str(),
        Some("CRYPTOGRAPHICALLY_VALID_CURRENT_AUTHORIZATION_DENIED")
    );
    assert_eq!(
        challenge["expected_stages"]["signature"].as_str(),
        Some("PASS")
    );
    assert_eq!(
        challenge["expected_stages"]["signer_lifecycle"].as_str(),
        Some("FAIL")
    );
}

#[test]
fn all_hard_gate_targets_are_conservative() {
    let root = corpus();
    let targets = &root["hard_gate_targets"];
    assert_eq!(
        targets["false_accepts_in_seeded_invalid_attestations"].as_u64(),
        Some(0)
    );
    assert_eq!(
        targets["unexpected_not_run_or_invalid_status_promotions"].as_u64(),
        Some(0)
    );
    assert_eq!(
        targets["missing_or_unaccounted_for_corpus_cases"].as_u64(),
        Some(0)
    );
    assert_eq!(
        targets["detection_rate_each_required_negative_control_family"].as_f64(),
        Some(1.0)
    );
}
