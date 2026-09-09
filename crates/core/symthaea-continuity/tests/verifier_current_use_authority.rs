// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_continuity::{
    ApprovalBasis, ContinuityContractV1, ContinuityRequirementV1, DependencyBasis,
    DependencyClaimV1, EvidenceBasis, EvidenceClass, EquivalencePredicate,
    GrantBoundVerifierProfileAdoptionCommitPreconditionsV1, ObservationCoverage,
    ObservationEnvelopeV1, RequirementCriticality, RootBoundPolicyCheckedVerifierProfileAdoptionV1,
    TimeBoundVerifierProfileAdoptionCommitPreconditionsV1, ValidatedContinuityContractV1,
    VerifierAdoptionAuthorityGrantV1, VerifierAdoptionScopeV1,
    VerifierProfileAdoptionAdmissionPolicyV1, VerifierProfileAdoptionAuthorityRootSnapshotV1,
    VerifierProfileAdoptionClockObservationV1, VerifierProfileAdoptionCommitPreconditionsV1,
    VerifierProfileAdoptionHeadV1, VerifierProfileAdoptionSubjectV1,
    VerifierProfileAdoptionTransitionV1, VerifierProfileRuntimeAuthorityBaselineV1,
    VerifierProfileRuntimeAuthorityError, VerifierProfileV1,
    bind_root_bound_adoption_to_authority_grant, check_current_verifier_runtime_policy,
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

fn root(epoch: u64) -> VerifierProfileAdoptionAuthorityRootSnapshotV1 {
    VerifierProfileAdoptionAuthorityRootSnapshotV1::new(
        "organization:test",
        "adoption-root-1",
        [0x55; 32],
        epoch,
    )
    .unwrap()
}

fn grant(
    root: VerifierProfileAdoptionAuthorityRootSnapshotV1,
    epoch: u64,
    class: EvidenceClass,
    scope: VerifierAdoptionScopeV1,
) -> VerifierAdoptionAuthorityGrantV1 {
    VerifierAdoptionAuthorityGrantV1::new(
        "grant-1",
        root,
        "hardware-verifier-v1",
        epoch,
        class,
        scope,
    )
    .unwrap()
}

fn clock(source: &str, epoch: u64, earliest: u64, latest: u64) -> VerifierProfileAdoptionClockObservationV1 {
    VerifierProfileAdoptionClockObservationV1::new(source, epoch, earliest, latest).unwrap()
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
    let root = root(8);
    let adopted_scope = VerifierAdoptionScopeV1::Contract {
        contract_id: contract.id(),
    };
    let transition = VerifierProfileAdoptionTransitionV1::bootstrap(
        VerifierProfileAdoptionSubjectV1::new(
            "adopt-1",
            root.authority_subject(),
            root.authority_root_id(),
            root.authority_root_digest(),
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
        root.authority_subject(),
        root.authority_root_id(),
        root.authority_root_digest(),
        profile.profile_name(),
        VerifierProfileAdoptionHeadV1::Uninitialized,
    )
    .unwrap()
    .check(1_500, &transition, &profile, Some(&contract))
    .unwrap();
    let root_bound = RootBoundPolicyCheckedVerifierProfileAdoptionV1::bind(
        checked,
        root.clone(),
    )
    .unwrap();
    let historical_grant = grant(
        root.clone(),
        3,
        EvidenceClass::HardwareVerified,
        adopted_scope,
    );
    let grant_bound = bind_root_bound_adoption_to_authority_grant(
        root_bound.clone(),
        &historical_grant,
        Some(&contract),
    )
    .unwrap();
    let commit = VerifierProfileAdoptionCommitPreconditionsV1::from_root_bound(root_bound).unwrap();
    let checked_clock = clock("trusted-clock", 4, 1_490, 1_510);
    let time_bound = TimeBoundVerifierProfileAdoptionCommitPreconditionsV1::new(
        commit,
        &checked_clock,
        250,
    )
    .unwrap();
    let grant_commit = GrantBoundVerifierProfileAdoptionCommitPreconditionsV1::new(
        grant_bound,
        time_bound,
    )
    .unwrap();
    let baseline = VerifierProfileRuntimeAuthorityBaselineV1::from_commit_preconditions(&grant_commit);
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
        root.clone(),
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
        &clock("trusted-clock", 4, 1_400, 1_600),
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
        root.clone(),
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
            &clock("trusted-clock", 4, 1_400, 1_600),
            Some(&contract),
        )
        .unwrap_err(),
        VerifierProfileRuntimeAuthorityError::SameEpochGrantEquivocation { epoch: 3 },
    );
}

#[test]
fn root_reprovisioning_and_clock_lineage_change_fail_closed() {
    let (contract, profile, transition, baseline, head, root) = fixture();
    let current_grant = baseline.historical_grant().clone();

    assert_eq!(
        check_current_verifier_runtime_policy(
            &baseline,
            &transition,
            &head,
            &root(9),
            Some(&current_grant),
            &profile,
            &clock("trusted-clock", 4, 1_400, 1_500),
            Some(&contract),
        )
        .unwrap_err(),
        VerifierProfileRuntimeAuthorityError::AuthorityRootSnapshotChanged,
    );

    assert!(matches!(
        check_current_verifier_runtime_policy(
            &baseline,
            &transition,
            &head,
            &root,
            Some(&current_grant),
            &profile,
            &clock("trusted-clock", 5, 1_400, 1_500),
            Some(&contract),
        ),
        Err(VerifierProfileRuntimeAuthorityError::ClockLineageChanged { .. })
    ));
}
