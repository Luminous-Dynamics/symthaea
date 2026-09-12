// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Source-level ETK-3C authority-surface non-regression ratchet.

#[test]
fn analytical_trust_crate_cannot_mutate_legacy_obligation_authority() {
    let source = include_str!("../src/lib.rs");

    assert!(source.contains("EvidenceKind::Analysis"));
    assert!(!source.contains("EvidenceKind::Simulation"));
    assert!(!source.contains("ObligationStatus"));
    assert!(!source.contains(".status ="));
    assert!(!source.contains("evidence_refs.push"));
    assert!(!source.contains(".passes"));
    assert!(!source.contains("passes:"));
}

#[test]
fn authority_ids_remain_one_way_capabilities() {
    let source = include_str!("../src/lib.rs");
    let authority_start = source
        .find("macro_rules! authority_id")
        .expect("authority-id macro must remain explicit");
    let authority_tail = &source[authority_start..];
    let authority_end = authority_tail
        .find("premise_id!(SubjectRevisionIdV1)")
        .expect("premise declarations must follow authority macro");
    let authority_macro = &authority_tail[..authority_end];

    assert!(authority_macro.contains("fn from_digest"));
    assert!(!authority_macro.contains("pub fn parse"));
    assert!(!source.contains("Deserialize"));
}

#[test]
fn authority_ladder_remains_explicit() {
    let source = include_str!("../src/lib.rs");

    for boundary in [
        "AcceptedAnalysisRequirementV1",
        "NativeAnalyticalPlanV1",
        "AdmittedAnalyticalEvidenceV1",
        "NativeAnalyticalDischargeReceiptV1",
        "CurrentNativeAnalyticalDischargeFactV1",
    ] {
        assert!(source.contains(boundary), "missing authority boundary: {boundary}");
    }
}
