// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_continuity::{
    ApprovalBasis, ContinuityContractV1, ContinuityRequirementV1, DependencyBasis,
    DependencyClaimV1, EvidenceBasis, EvidenceClass, EquivalencePredicate, ObservationCoverage,
    ObservationEnvelopeV1, RequirementCriticality, ValidatedContinuityContractV1,
    VerifierAdoptionAuthorityGrantV1, VerifierAdoptionScopeV1,
    VerifierProfileAdoptionAdmissionPolicyV1, VerifierProfileAdoptionAuthorityRootSnapshotV1,
    VerifierProfileAdoptionClockObservationV1, VerifierProfileAdoptionHeadV1,
    VerifierProfileAdoptionSubjectV1, VerifierProfileAdoptionTransitionV1,
    VerifierProfileRuntimeAuthorityBaselineV1, VerifierProfileRuntimeAuthorityError,
    VerifierProfileV1, bind_policy_checked_adoption_to_authority_grant,
    check_current_verifier_runtime_policy,
};

fn contract() -> ValidatedContinuityContractV1 {
    let mut requirements = Vec::new();
    for seed in [1u8, 2u8] {
        let observation = ObservationEnvelopeV1::new(
            format!("machine-{seed}"),
            "workflow.dependency",
            "fixture",
            "1",
            1_700_000_000_000 + seed as u64,
            ObservationCoverage::Complete,
            EvidenceBasis::Tested,
            [seed; 32],
            vec![],
        )
        .unwrap();
        let dependency = DependencyClaimV1::new(
            "role:research",
            "requires",
            format!("capability-{seed}"),
            DependencyBasis::Observed,
            vec![observation.id()],
            vec![],
        )
        .unwrap();
        requirements.push(
            ContinuityRequirementV1::new(
                dependency.id(),
                format!("workflow-{seed}"),
                RequirementCriticality::Must,
                EquivalencePredicate::BehavioralScenario {
                    scenario_id: format!("scenario-{seed}"),
                },
                ApprovalBasis::ExplicitPolicy,
                [seed + 10; 32],
            )
            .unwrap(),
        );
    }
    ContinuityContractV1::new("research-fleet", [33; 32], requirements)
        .unwrap()
        .validate()
        .unwrap()
}

fn profile() -> VerifierProfileV1 {
    VerifierProfileV1::new(
        "hardware-verifier-v1",
        [9; 32],
        7,
        EvidenceClass::HardwareVerified,
    )
    .unwrap()
}

fn root() -> VerifierProfileAdoptionAuthorityRootSnapshotV1 {
    VerifierProfileAdoptionAuthorityRootSnapshotV1::new(
        "organization:test",
        "adoption-root-1",
        [0x55; 32],
        8,
    )
    .unwrap()
}

fn grant(
    epoch: u64,
    class: EvidenceClass,
    scope: VerifierAdoptionScopeV1,
) -> VerifierAdoptionAuthorityGrantV1 {
    VerifierAdoptionAuthorityGrantV1::new(
        "grant-1",
        "organization:test",
        "adoption-root-1",
        [0x55; 32],
        "hardware-verifier-v1",
        epoch,
        class,
        scope,
    )
    .unwrap()
}

fn clock() -> VerifierProfileAdoptionClockObservationV1 {
    VerifierProfileAdoptionClockObservationV1::new("trusted-clock", 4, 1_400, 1_600).unwrap()
}

fn fixture() -> (
    ValidatedContinuityContractV1,
    VerifierProfileV1,
    VerifierProfileAdoptionTransitionV1,
    VerifierProfileRuntimeAuthorityBaselineV1,
    VerifierProfileAdoptionHeadV1,
    VerifierProfileAdoptionAuthorityRootSnapshotV1,
) {
    let contract = contract();
    let profile = profile();
    let adopted_scope = VerifierAdoptionScopeV1::Contract {
        contract_id: contract.id(),
    };
    let transition = VerifierProfileAdoptionTransitionV1::bootstrap(
        VerifierProfileAdoptionSubjectV1::new(
            "adopt-1",
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            &profile,
            1,
            1_000,
            2_000,
            EvidenceClass::HardwareVerified,
            adopted_scope.clone(),
        )
        .unwrap(),
    )
    .unwrap();
    let checked = VerifierProfileAdoptionAdmissionPolicyV1::new(
        "organization:test",
        "adoption-root-1",
        [0x55; 32],
        "hardware-verifier-v1",
        VerifierProfileAdoptionHeadV1::Uninitialized,
    )
    .unwrap()
    .check(1_500, &transition, &profile, Some(&contract))
    .unwrap();
    let historical_grant = grant(3, EvidenceClass::HardwareVerified, adopted_scope);
    let granted = bind_policy_checked_adoption_to_authority_grant(
        checked,
        &historical_grant,
        Some(&contract),
    )
    .unwrap();
    let root = root();
    let baseline = VerifierProfileRuntimeAuthorityBaselineV1::from_granted_adoption(
        &granted,
        &root,
    )
    .unwrap();
    let head = VerifierProfileAdoptionHeadV1::from_transition(&transition).unwrap();
    (contract, profile, transition, baseline, head, root)
}

#[test]
fn newer_grant_can_narrow_to_exact_requirement_scope() {
    let (contract, profile, transition, baseline, head, root) = fixture();
    let requirement = contract.requirements()[0].id();
    let narrowed_scope =
        VerifierAdoptionScopeV1::requirements(contract.id(), vec![requirement]).unwrap();
    let narrowed = grant(
        4,
        EvidenceClass::DifferentiallyVerified,
        narrowed_scope.clone(),
    );

    let current = check_current_verifier_runtime_policy(
        &baseline,
        &transition,
        &head,
        &root,
        Some(&narrowed),
        &profile,
        &clock(),
        250,
        Some(&contract),
    )
    .unwrap();

    assert_eq!(
        current.effective_evidence_class(),
        EvidenceClass::DifferentiallyVerified
    );
    assert_eq!(current.effective_scope(), &narrowed_scope);
}

#[test]
fn same_epoch_semantic_change_is_equivocation() {
    let (contract, profile, transition, baseline, head, root) = fixture();
    let conflicting = grant(
        3,
        EvidenceClass::DifferentiallyVerified,
        VerifierAdoptionScopeV1::Contract {
            contract_id: contract.id(),
        },
    );

    assert_eq!(
        check_current_verifier_runtime_policy(
            &baseline,
            &transition,
            &head,
            &root,
            Some(&conflicting),
            &profile,
            &clock(),
            250,
            Some(&contract),
        )
        .unwrap_err(),
        VerifierProfileRuntimeAuthorityError::SameEpochGrantEquivocation { epoch: 3 },
    );
}
