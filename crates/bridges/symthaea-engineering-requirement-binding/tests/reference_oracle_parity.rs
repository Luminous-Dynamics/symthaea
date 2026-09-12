use symthaea_engineering_evidence_plan::{
    AcceptedRequirementRevisionV1, RequirementCriticalityV1, Sha256DigestV1,
};
use symthaea_engineering_requirement_binding::{
    BindingAcceptanceRecordDigestV1, DerivationPolicyRevisionDigestV1,
    DerivationRecordDigestV1, RequirementObligationBindingV1,
};
use symthaea_formal_safety::{EvidenceKind, ObligationStatus, ProofObligation};
use symthaea_sim_bridge::EngineeringDomain;
use uuid::Uuid;

fn digest(ch: char) -> Sha256DigestV1 {
    Sha256DigestV1::parse(format!("sha256:{}", ch.to_string().repeat(64))).unwrap()
}

#[test]
fn production_relation_matches_independent_oracle_vector() {
    let requirement = AcceptedRequirementRevisionV1::new(
        "REQ-STRESS",
        EngineeringDomain::Civil,
        "stress remains below allowable",
        RequirementCriticalityV1::Blocking,
        EvidenceKind::Simulation,
        ["stress <= 250 MPa"],
        digest('a'),
    )
    .unwrap();
    assert_eq!(
        requirement.revision_id().as_str(),
        "sha256:e340c5030eebc978c41443ffd64f340dc5febad31e376080340bacfceda60faa"
    );

    let obligation = ProofObligation {
        id: Uuid::parse_str("00000000-0000-4000-8000-000000000042").unwrap(),
        claim: "stress remains below allowable under service load".into(),
        expected_evidence: EvidenceKind::Simulation,
        status: ObligationStatus::Open,
        evidence_refs: Vec::new(),
    };

    let binding = RequirementObligationBindingV1::derived_safety_obligation(
        &requirement,
        &obligation,
        DerivationRecordDigestV1::from_digest(digest('7')),
        DerivationPolicyRevisionDigestV1::from_digest(digest('8')),
        BindingAcceptanceRecordDigestV1::from_digest(digest('9')),
    )
    .unwrap();

    assert_eq!(
        binding.obligation_revision_id().as_str(),
        "sha256:743d2c13cfcc52bdfb4cfd9a4a836ed0806ace7b2a68b4c7284b1910d8863c29"
    );
    assert_eq!(
        binding.binding_id().as_str(),
        "sha256:1cc54ee11ff9b7e99279d4c9a9d43f751809f7f30a43469cff8e18fe70eee408"
    );
    assert_eq!(binding.audit_record_v1()["authority"], "relationship-binding-only");
    assert_eq!(obligation.status, ObligationStatus::Open);
}

#[test]
fn derivation_record_change_changes_only_relationship_identity_for_fixed_endpoints() {
    let requirement = AcceptedRequirementRevisionV1::new(
        "REQ-STRESS",
        EngineeringDomain::Civil,
        "stress remains below allowable",
        RequirementCriticalityV1::Blocking,
        EvidenceKind::Simulation,
        ["stress <= 250 MPa"],
        digest('a'),
    )
    .unwrap();
    let obligation = ProofObligation {
        id: Uuid::parse_str("00000000-0000-4000-8000-000000000042").unwrap(),
        claim: "stress remains below allowable under service load".into(),
        expected_evidence: EvidenceKind::Simulation,
        status: ObligationStatus::Open,
        evidence_refs: Vec::new(),
    };

    let a = RequirementObligationBindingV1::derived_safety_obligation(
        &requirement,
        &obligation,
        DerivationRecordDigestV1::from_digest(digest('7')),
        DerivationPolicyRevisionDigestV1::from_digest(digest('8')),
        BindingAcceptanceRecordDigestV1::from_digest(digest('9')),
    )
    .unwrap();
    let b = RequirementObligationBindingV1::derived_safety_obligation(
        &requirement,
        &obligation,
        DerivationRecordDigestV1::from_digest(digest('6')),
        DerivationPolicyRevisionDigestV1::from_digest(digest('8')),
        BindingAcceptanceRecordDigestV1::from_digest(digest('9')),
    )
    .unwrap();

    assert_eq!(a.requirement_revision_id(), b.requirement_revision_id());
    assert_eq!(a.obligation_revision_id(), b.obligation_revision_id());
    assert_ne!(a.binding_id(), b.binding_id());
}
