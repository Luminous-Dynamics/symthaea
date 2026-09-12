use symthaea_engineering_evidence_plan::{
    AcceptedRequirementRevisionV1, CurrentnessAssertionV1, MetricOperatorV1,
    MetricPredicateV1, ObligationRevisionIdV1, RequirementCriticalityV1,
    Sha256DigestV1, SimulationEvidencePlanV1, SimulationEvidencePolicyRevisionV1,
    SimulationRequestRevisionV1, SubjectRevisionV1, TwinKindV1, TwinRevisionV1,
    ValidityDomainRevisionV1, WarningPolicyV1,
};
use symthaea_formal_safety::{EvidenceKind, ObligationStatus, ProofObligation};
use symthaea_sim_bridge::{EngineeringDomain, SimulationRequest, SolverKind};
use uuid::Uuid;

fn digest(ch: char) -> Sha256DigestV1 {
    Sha256DigestV1::parse(format!("sha256:{}", ch.to_string().repeat(64))).unwrap()
}

struct Fixture {
    requirement: AcceptedRequirementRevisionV1,
    subject: SubjectRevisionV1,
    twin: TwinRevisionV1,
    request: SimulationRequestRevisionV1,
    policy: SimulationEvidencePolicyRevisionV1,
    validity: ValidityDomainRevisionV1,
    currentness: CurrentnessAssertionV1,
    obligation: ProofObligation,
}

fn fixture() -> Fixture {
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

    let subject = SubjectRevisionV1::new("design", "bracket-alpha", digest('b')).unwrap();
    let twin = TwinRevisionV1::new(
        &subject,
        TwinKindV1::Design,
        digest('c'),
        digest('d'),
        None,
    )
    .unwrap();

    let mut raw_request = SimulationRequest::new(
        "sim-static-G17-LC9",
        EngineeringDomain::Civil,
        SolverKind::FiniteElement,
        "check bracket service stress",
    )
    .with_parameter("load_n", 10_000.0, "N", "load-case:LC9")
    .with_parameter("thickness_mm", 8.0, "mm", "design:G17");
    raw_request.requested_metrics = vec![
        "max_stress_mpa".into(),
        "max_displacement_mm".into(),
    ];
    let request = SimulationRequestRevisionV1::from_request(&raw_request).unwrap();

    let policy = SimulationEvidencePolicyRevisionV1::new(
        "service-stress-policy",
        MetricPredicateV1::new(
            "max_stress_mpa",
            "MPa",
            MetricOperatorV1::Le,
            250.0,
            0.2,
            0.1,
        )
        .unwrap(),
        WarningPolicyV1::DenyAny,
    )
    .unwrap();

    let validity = ValidityDomainRevisionV1::new(
        &subject,
        &twin,
        digest('e'),
        digest('f'),
        vec![
            ("load_case".into(), digest('1')),
            ("material_state".into(), digest('2')),
            ("boundary_conditions".into(), digest('3')),
        ],
    )
    .unwrap();

    let currentness =
        CurrentnessAssertionV1::new(&twin, &validity, digest('4'), 1_789_123_456_000).unwrap();

    let obligation = ProofObligation {
        id: Uuid::parse_str("00000000-0000-4000-8000-000000000042").unwrap(),
        claim: "stress remains below allowable under service load".into(),
        expected_evidence: EvidenceKind::Simulation,
        status: ObligationStatus::Open,
        evidence_refs: Vec::new(),
    };

    Fixture {
        requirement,
        subject,
        twin,
        request,
        policy,
        validity,
        currentness,
        obligation,
    }
}

#[test]
fn production_semantic_ids_match_independent_oracle_vectors() {
    let f = fixture();
    assert_eq!(
        f.requirement.revision_id().as_str(),
        "sha256:e340c5030eebc978c41443ffd64f340dc5febad31e376080340bacfceda60faa"
    );
    assert_eq!(
        f.subject.revision_id().as_str(),
        "sha256:38a9505d423fa3464020107b4e8abc8acc6ac6af33bcc98a2418480e1d33390e"
    );
    assert_eq!(
        f.twin.revision_id().as_str(),
        "sha256:19d558d6e7579f0f44c71398c7ddac659677aca0fa0e65ed46227670e4876cd9"
    );
    assert_eq!(
        f.request.revision_id().as_str(),
        "sha256:aec313ebf10fd7b611f7ff8d6239cf0b4006edde6ae580e7c800ecca8398decc"
    );
    assert_eq!(
        f.policy.revision_id().as_str(),
        "sha256:ec15bcd4e0adbb135e14287b0791f07b5200faa18a702689b7b131a94b9543e3"
    );
    assert_eq!(
        f.validity.revision_id().as_str(),
        "sha256:90bff4adf8917f2ae071f405d5c46afb310b12a98dd2e4eec845de1261a24890"
    );
    assert_eq!(
        f.currentness.assertion_id().as_str(),
        "sha256:f85bd7d5129b08a3256cfdd5b3506f3cae1dbc625b097a8bf95bf1f6fe709dd9"
    );
    assert_eq!(
        ObligationRevisionIdV1::for_obligation(&f.obligation)
            .unwrap()
            .as_str(),
        "sha256:743d2c13cfcc52bdfb4cfd9a4a836ed0806ace7b2a68b4c7284b1910d8863c29"
    );

    let plan = SimulationEvidencePlanV1::new(
        &f.subject,
        &f.twin,
        &f.requirement,
        &f.obligation,
        &f.request,
        &f.policy,
        &f.validity,
        &f.currentness,
        digest('6'),
    )
    .unwrap();
    assert_eq!(
        plan.plan_id().as_str(),
        "sha256:17b14595f14fcf06c5f8f7c38f3953702d5cf93b5e9b40a106e638ed85337f62"
    );
}

#[test]
fn currentness_refresh_invalidates_only_currentness_and_composed_plan_identity() {
    let f = fixture();
    let baseline = SimulationEvidencePlanV1::new(
        &f.subject,
        &f.twin,
        &f.requirement,
        &f.obligation,
        &f.request,
        &f.policy,
        &f.validity,
        &f.currentness,
        digest('6'),
    )
    .unwrap();

    let refreshed =
        CurrentnessAssertionV1::new(&f.twin, &f.validity, digest('5'), 1_789_123_457_000).unwrap();
    let refreshed_plan = SimulationEvidencePlanV1::new(
        &f.subject,
        &f.twin,
        &f.requirement,
        &f.obligation,
        &f.request,
        &f.policy,
        &f.validity,
        &refreshed,
        digest('6'),
    )
    .unwrap();

    assert_eq!(f.twin.revision_id(), refreshed.twin_revision_id());
    assert_eq!(f.validity.revision_id(), refreshed.validity_domain_revision_id());
    assert_ne!(f.currentness.assertion_id(), refreshed.assertion_id());
    assert_ne!(baseline.plan_id(), refreshed_plan.plan_id());
}
