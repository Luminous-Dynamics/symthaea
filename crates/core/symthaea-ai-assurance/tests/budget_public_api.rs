use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, SystemTime};

use symthaea_ai_assurance::{
    BudgetAuthorityDomain, BudgetDimension, BudgetEnforcement, BudgetError, BudgetProfile,
    BudgetQuantities, EnforcementClass, PrincipalId, Scope,
};

fn scope(parts: &[&str]) -> Scope {
    Scope::new("agent", parts.iter().copied()).unwrap()
}

fn profile(compute: u64, subprocesses: u64) -> BudgetProfile {
    let limits = BudgetQuantities::zero()
        .with(BudgetDimension::ComputeUnits, compute)
        .with(BudgetDimension::Subprocesses, subprocesses);
    let enforcement = BudgetEnforcement::soft()
        .with(BudgetDimension::ComputeUnits, EnforcementClass::CoreMetered)
        .with(BudgetDimension::Subprocesses, EnforcementClass::CoreMetered);
    BudgetProfile::new(limits, enforcement, None).unwrap()
}

#[test]
fn public_pool_is_conserved_under_concurrent_reservation() {
    let domain = Arc::new(BudgetAuthorityDomain::new(
        PrincipalId::new(),
        profile(1, 0),
    ));
    let barrier = Arc::new(Barrier::new(3));
    let mut workers = Vec::new();

    for tag in [1_u8, 2_u8] {
        let domain = Arc::clone(&domain);
        let barrier = Arc::clone(&barrier);
        workers.push(thread::spawn(move || {
            barrier.wait();
            domain.reserve(
                PrincipalId::new(),
                scope(&["root"]),
                [tag; 32],
                BudgetQuantities::zero().with(BudgetDimension::ComputeUnits, 1),
                None,
            )
        }));
    }

    barrier.wait();
    let successes = workers
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .filter(Result::is_ok)
        .count();
    assert_eq!(successes, 1);
    assert_eq!(domain.remaining().get(BudgetDimension::ComputeUnits), 0);
}

#[test]
fn public_lease_is_exact_action_subject_scope_and_domain_bound() {
    let domain = BudgetAuthorityDomain::new(PrincipalId::new(), profile(5, 1));
    let other = BudgetAuthorityDomain::new(PrincipalId::new(), profile(5, 1));
    let actor = PrincipalId::new();
    let lease = domain
        .reserve(
            actor,
            scope(&["root"]),
            [3; 32],
            BudgetQuantities::zero().with(BudgetDimension::ComputeUnits, 2),
            None,
        )
        .unwrap();

    assert!(lease
        .validate_for(
            &domain.verifier(),
            actor,
            &scope(&["root", "child"]),
            [3; 32]
        )
        .is_ok());
    assert!(matches!(
        lease.validate_for(&domain.verifier(), actor, &scope(&["root"]), [4; 32]),
        Err(BudgetError::ActionBindingMismatch)
    ));
    assert!(lease
        .validate_for(&other.verifier(), actor, &scope(&["root"]), [3; 32])
        .is_err());
}

#[test]
fn public_affine_split_and_release_conserve_root_capacity() {
    let domain = BudgetAuthorityDomain::new(PrincipalId::new(), profile(10, 2));
    let parent_actor = PrincipalId::new();
    let child_actor = PrincipalId::new();
    let parent = domain
        .reserve(
            parent_actor,
            scope(&["root"]),
            [5; 32],
            BudgetQuantities::zero()
                .with(BudgetDimension::ComputeUnits, 8)
                .with(BudgetDimension::Subprocesses, 2),
            Some(SystemTime::now() + Duration::from_secs(60)),
        )
        .unwrap();
    let source = parent.lease_id();
    assert_eq!(domain.remaining().get(BudgetDimension::ComputeUnits), 2);

    let (remainder, child) = domain
        .split(
            parent,
            child_actor,
            scope(&["root", "child"]),
            [6; 32],
            BudgetQuantities::zero()
                .with(BudgetDimension::ComputeUnits, 3)
                .with(BudgetDimension::Subprocesses, 1),
            Some(SystemTime::now() + Duration::from_secs(30)),
        )
        .unwrap();

    assert_eq!(remainder.parent_lease_id(), Some(source));
    assert_eq!(child.parent_lease_id(), Some(source));
    assert_eq!(
        remainder.allocation().get(BudgetDimension::ComputeUnits)
            + child.allocation().get(BudgetDimension::ComputeUnits),
        8
    );
    assert_eq!(
        remainder.allocation().get(BudgetDimension::Subprocesses)
            + child.allocation().get(BudgetDimension::Subprocesses),
        2
    );

    remainder.release().unwrap();
    child.release().unwrap();
    assert_eq!(domain.remaining().get(BudgetDimension::ComputeUnits), 10);
    assert_eq!(domain.remaining().get(BudgetDimension::Subprocesses), 2);
}

#[test]
fn public_recoverable_split_preserves_exact_parent_on_scope_rejection() {
    let domain = BudgetAuthorityDomain::new(PrincipalId::new(), profile(10, 2));
    let actor = PrincipalId::new();
    let allocation = BudgetQuantities::zero()
        .with(BudgetDimension::ComputeUnits, 7)
        .with(BudgetDimension::Subprocesses, 1);
    let parent = domain
        .reserve(
            actor,
            scope(&["root"]),
            [20; 32],
            allocation,
            Some(SystemTime::now() + Duration::from_secs(60)),
        )
        .unwrap();
    let id = parent.lease_id();
    let budget_domain = parent.domain_id();
    let epoch = parent.epoch();
    let action_binding = parent.action_binding();
    let profile_digest = parent.profile().digest();
    let remaining_before = domain.remaining();

    let failure = domain
        .split_recoverable(
            parent,
            PrincipalId::new(),
            scope(&["other"]),
            [21; 32],
            BudgetQuantities::zero().with(BudgetDimension::ComputeUnits, 2),
            Some(SystemTime::now() + Duration::from_secs(30)),
        )
        .unwrap_err();

    assert!(matches!(failure.error(), BudgetError::ScopeWidening { .. }));
    assert_eq!(failure.parent().lease_id(), id);
    assert_eq!(failure.parent().allocation(), allocation);
    assert_eq!(failure.parent().domain_id(), budget_domain);
    assert_eq!(failure.parent().epoch(), epoch);
    assert_eq!(failure.parent().action_binding(), action_binding);
    assert_eq!(failure.parent().profile().digest(), profile_digest);
    assert_eq!(domain.remaining(), remaining_before);

    failure.into_parent().release().unwrap();
    assert_eq!(domain.remaining(), domain.profile().limits());
}

#[test]
fn public_recoverable_split_rejects_stale_child_without_capacity_loss() {
    let domain = BudgetAuthorityDomain::new(PrincipalId::new(), profile(10, 2));
    let actor = PrincipalId::new();
    let parent = domain
        .reserve(
            actor,
            scope(&["root"]),
            [22; 32],
            BudgetQuantities::zero().with(BudgetDimension::ComputeUnits, 6),
            Some(SystemTime::now() + Duration::from_secs(60)),
        )
        .unwrap();
    let id = parent.lease_id();
    let remaining_before = domain.remaining();

    let failure = domain
        .split_recoverable(
            parent,
            PrincipalId::new(),
            scope(&["root", "child"]),
            [23; 32],
            BudgetQuantities::zero().with(BudgetDimension::ComputeUnits, 2),
            Some(SystemTime::UNIX_EPOCH),
        )
        .unwrap_err();

    assert!(matches!(
        failure.error(),
        BudgetError::ExpiredDelegationRequest { .. }
    ));
    assert_eq!(failure.parent().lease_id(), id);
    assert_eq!(domain.remaining(), remaining_before);
    failure.into_parent().release().unwrap();
    assert_eq!(domain.remaining(), domain.profile().limits());
}

#[test]
fn public_recoverable_split_rejects_overallocation_without_capacity_loss() {
    let domain = BudgetAuthorityDomain::new(PrincipalId::new(), profile(10, 2));
    let actor = PrincipalId::new();
    let parent = domain
        .reserve(
            actor,
            scope(&["root"]),
            [24; 32],
            BudgetQuantities::zero().with(BudgetDimension::ComputeUnits, 4),
            Some(SystemTime::now() + Duration::from_secs(60)),
        )
        .unwrap();
    let id = parent.lease_id();
    let remaining_before = domain.remaining();

    let failure = domain
        .split_recoverable(
            parent,
            PrincipalId::new(),
            scope(&["root", "child"]),
            [25; 32],
            BudgetQuantities::zero().with(BudgetDimension::ComputeUnits, 5),
            Some(SystemTime::now() + Duration::from_secs(30)),
        )
        .unwrap_err();

    assert!(matches!(
        failure.error(),
        BudgetError::InsufficientBudget { .. }
    ));
    assert_eq!(failure.parent().lease_id(), id);
    assert_eq!(domain.remaining(), remaining_before);
    failure.into_parent().release().unwrap();
    assert_eq!(domain.remaining(), domain.profile().limits());
}

#[test]
fn public_budget_epoch_rotation_revokes_outstanding_lease() {
    let domain = BudgetAuthorityDomain::new(PrincipalId::new(), profile(5, 1));
    let actor = PrincipalId::new();
    let lease = domain
        .reserve(
            actor,
            scope(&["root"]),
            [7; 32],
            BudgetQuantities::zero().with(BudgetDimension::ComputeUnits, 1),
            None,
        )
        .unwrap();
    domain.revoke_all().unwrap();

    assert!(lease
        .validate_for(&domain.verifier(), actor, &scope(&["root"]), [7; 32])
        .is_err());
}

#[test]
fn public_external_hard_label_requires_external_evidence_commitment() {
    let limits = BudgetQuantities::zero().with(BudgetDimension::MemoryBytes, 1024);
    let enforcement = BudgetEnforcement::soft()
        .with(BudgetDimension::MemoryBytes, EnforcementClass::ExternalHard);

    assert!(matches!(
        BudgetProfile::new(limits, enforcement, None),
        Err(BudgetError::MissingExternalEnforcementEvidence)
    ));
    assert!(BudgetProfile::new(limits, enforcement, Some([9; 32])).is_ok());
}
