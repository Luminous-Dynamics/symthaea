// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Source-level ETK-3C authority-surface non-regression ratchet.

fn production_source() -> String {
    [
        include_str!("../src/lib.rs"),
        include_str!("../src/canonical.rs"),
        include_str!("../src/context.rs"),
        include_str!("../src/analysis.rs"),
        include_str!("../src/authority.rs"),
    ]
    .join("\n")
}

#[test]
fn analytical_trust_crate_cannot_mutate_legacy_obligation_authority() {
    let source = production_source();

    assert!(source.contains("EvidenceKind::Analysis"));
    assert!(!source.contains("EvidenceKind::Simulation"));
    assert!(!source.contains("ObligationStatus"));
    assert!(!source.contains(".status ="));
    assert!(!source.contains("evidence_refs.push"));
    assert!(!source.contains(".passes"));
    assert!(!source.contains("passes:"));
}

#[test]
fn authority_and_semantic_ids_remain_one_way() {
    let canonical = include_str!("../src/canonical.rs");
    let semantic_start = canonical
        .find("macro_rules! semantic_id")
        .expect("semantic-id macro must remain explicit");
    let semantic_tail = &canonical[semantic_start..];
    let semantic_end = semantic_tail
        .find("premise_id!(AcceptanceRecordDigestV1)")
        .expect("premise declarations must follow semantic-id macro");
    let semantic_macro = &semantic_tail[..semantic_end];

    assert!(semantic_macro.contains("pub(crate) fn from_digest"));
    assert!(!semantic_macro.contains("pub fn parse"));
    assert!(!canonical.contains("premise_id!(SubjectRevisionIdV1)"));
    assert!(!canonical.contains("premise_id!(TwinRevisionIdV1)"));
    assert!(!canonical.contains("premise_id!(ValidityDomainRevisionIdV1)"));
    assert!(!canonical.contains("premise_id!(CurrentnessAssertionIdV1)"));
    assert!(!production_source().contains("Deserialize"));
}

#[test]
fn authority_critical_external_premises_remain_role_safe() {
    let canonical = include_str!("../src/canonical.rs");
    for role in [
        "AcceptanceRecordDigestV1",
        "SubjectStateDigestV1",
        "TwinStateDigestV1",
        "TwinSchemaDigestV1",
        "ModelRevisionDigestV1",
        "AnalysisConfigurationDigestV1",
        "ValidityDimensionDigestV1",
        "CurrentnessAttestationDigestV1",
        "ImplementationArtifactDigestV1",
        "AlgorithmRevisionDigestV1",
        "ModelQualificationRecordDigestV1",
        "ExecutionArtifactDigestV1",
    ] {
        assert!(
            canonical.contains(&format!("premise_id!({role})")),
            "missing role-safe premise type: {role}"
        );
    }
}

#[test]
fn context_must_be_derived_from_explicit_semantic_records() {
    let context = include_str!("../src/context.rs");
    let authority = include_str!("../src/authority.rs");

    for schema in [
        "symthaea.etk-engineering-subject.v1",
        "symthaea.etk-twin-revision.v1",
        "symthaea.etk-validity-domain.v1",
        "symthaea.etk-currentness-assertion.v1",
    ] {
        assert!(context.contains(schema), "missing semantic context schema: {schema}");
    }

    assert!(authority.contains("subject: &SubjectRevisionV1"));
    assert!(authority.contains("twin: &TwinRevisionV1"));
    assert!(authority.contains("validity_domain: &ValidityDomainRevisionV1"));
    assert!(authority.contains("currentness: &CurrentnessAssertionV1"));
    assert!(authority.contains("ValidityContextMismatch"));
    assert!(authority.contains("CurrentnessContextMismatch"));
}

#[test]
fn authority_critical_evidence_kind_names_come_from_formal_safety() {
    let context = include_str!("../src/context.rs");

    assert!(context.contains("EvidenceKind::Analysis.canonical_name()"));
    assert!(!context.contains("\"expected_evidence_kind\": \"Analysis\""));
}

#[test]
fn exact_input_equations_and_requirement_policy_link_cannot_silently_disappear() {
    let source = production_source();

    for theorem in [
        "PolicyDoesNotDischargeRequirement",
        "requirement_max_bending_stress_pa",
        "result.method_revision_id",
        "result.input_revision_id",
        "expected_max_moment_nm",
        "expected_max_bending_stress_pa",
        "expected_max_deflection_m",
        "AnalyticalEquationMismatch",
        "conservative_stress_pa",
        "conservative_factor_of_safety",
    ] {
        assert!(source.contains(theorem), "missing ETK-3C theorem: {theorem}");
    }
}

#[test]
fn authority_ladder_remains_explicit() {
    let source = production_source();

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
