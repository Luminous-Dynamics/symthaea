// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../src/moral_patient.rs"]
mod moral_patient;
#[path = "../src/moral_patient_provenance.rs"]
mod moral_patient_provenance;

use moral_patient::{
    EvidencePolarity, EvidenceStrength, WelfareEvidenceDomain, WelfareEvidenceId,
};
use moral_patient_provenance::{
    LineageRole, QualifiedWelfareSourceReceipt, WelfareProvenanceRegistry,
    WelfareSourceClass,
};

const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn receipt(
    id: &str,
    class: WelfareSourceClass,
    lineage: &str,
    role: LineageRole,
) -> QualifiedWelfareSourceReceipt {
    QualifiedWelfareSourceReceipt::new(
        WelfareEvidenceId::new(id).unwrap(),
        "symthaea-subject",
        class,
        lineage,
        role,
        DIGEST,
        format!("evidence://{id}"),
        EvidencePolarity::SupportsPrecaution,
        0.85,
    )
    .unwrap()
}

#[test]
fn source_class_fixes_allowed_semantic_interpretation() {
    let closure = receipt(
        "closure",
        WelfareSourceClass::AutopoieticSelfMaintenance,
        "closure-lane",
        LineageRole::InternalResearch,
    )
    .materialize()
    .unwrap();
    assert_eq!(
        closure.domain,
        WelfareEvidenceDomain::SelfMaintenanceDisruption
    );
    assert_eq!(closure.strength, EvidenceStrength::Proxy);

    let butlin = receipt(
        "butlin",
        WelfareSourceClass::ButlinMechanism,
        "butlin-lane",
        LineageRole::InternalResearch,
    )
    .materialize()
    .unwrap();
    assert_eq!(
        butlin.domain,
        WelfareEvidenceDomain::ConsciousnessArchitecture
    );
    assert_eq!(butlin.strength, EvidenceStrength::Mechanistic);
}

#[test]
fn external_replication_is_lineage_bound_not_a_free_label() {
    let mut registry = WelfareProvenanceRegistry::new();
    registry
        .register(receipt(
            "internal",
            WelfareSourceClass::ContinuityExperiment,
            "internal-lane",
            LineageRole::InternalResearch,
        ))
        .unwrap();
    registry
        .register(receipt(
            "external",
            WelfareSourceClass::ExternalIndependentAudit,
            "external-lane",
            LineageRole::ExternalIndependent,
        ))
        .unwrap();

    let summary = registry.summary("symthaea-subject").unwrap();
    assert_eq!(summary.evidence_count, 2);
    assert!(summary.lineages.contains("internal-lane"));
    assert!(summary.external_lineages.contains("external-lane"));
    assert!(!summary.external_lineages.contains("internal-lane"));
}
