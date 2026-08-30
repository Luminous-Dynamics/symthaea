use std::sync::atomic::{AtomicBool, Ordering};

use symthaea_ai_assurance::{
    ActionRisk, AdapterSchema, ApprovalEvidence, AuthorityDomain, BudgetAuthorityDomain,
    BudgetDimension, BudgetEnforcement, BudgetGuardedAuthorizeError, BudgetGuardedRuntime,
    BudgetProfile, BudgetQuantities, EnforcementClass, Observation, Observe, ObservedOutcome,
    PolicyDescriptor, PolicyEvaluatorDomain, PolicyExecutionDomain, PolicyGuardedRuntime,
    PolicyMode, PolicyResourceRuntime, PrincipalId, ResolutionAuthorityDomain, ResolutionDecision,
    ResourceIdentity, ResourceResolverDomain, ResourceRuntime, Scope, TrustedRuntime, Write,
};

fn scope() -> Scope {
    Scope::new("workspace", ["symthaea", "src"]).unwrap()
}

fn resource_identity() -> ResourceIdentity {
    ResourceIdentity::new(
        scope(),
        "worktree-file",
        [1; 32],
        [2; 32],
        AdapterSchema::new("budget-guard-test", 1).unwrap(),
    )
    .unwrap()
}

fn budget_profile() -> BudgetProfile {
    let limits = BudgetQuantities::zero()
        .with(BudgetDimension::ComputeUnits, 100)
        .with(BudgetDimension::BytesWritten, 4096)
        .with(BudgetDimension::Subprocesses, 2);
    let enforcement = BudgetEnforcement::soft()
        .with(BudgetDimension::ComputeUnits, EnforcementClass::CoreMetered)
        .with(BudgetDimension::BytesWritten, EnforcementClass::Measured)
        .with(BudgetDimension::Subprocesses, EnforcementClass::CoreMetered);
    BudgetProfile::new(limits, enforcement, None).unwrap()
}

struct Fixture {
    evaluator: PolicyEvaluatorDomain,
    execution: PolicyExecutionDomain,
    observation: AuthorityDomain,
    resolution: ResolutionAuthorityDomain,
    resources: ResourceResolverDomain,
    budgets: BudgetAuthorityDomain,
    runtime: BudgetGuardedRuntime,
}

fn fixture() -> Fixture {
    let descriptor = PolicyDescriptor::new("magi-gate", 1, [3; 32], 1).unwrap();
    let evaluator = PolicyEvaluatorDomain::new(PrincipalId::new(), descriptor);
    let policy_verifier = evaluator.verifier();
    let execution = PolicyExecutionDomain::new(PrincipalId::new(), policy_verifier.clone());
    let observation = AuthorityDomain::new(PrincipalId::new());
    let resolution = ResolutionAuthorityDomain::new(PrincipalId::new());
    let resources = ResourceResolverDomain::new(PrincipalId::new());
    let budgets = BudgetAuthorityDomain::new(PrincipalId::new(), budget_profile());

    let trusted = TrustedRuntime::new(
        execution.verifier(),
        observation.verifier(),
        resolution.verifier(),
    );
    let resource_runtime = ResourceRuntime::new(trusted, resources.verifier());
    let policy_runtime = PolicyResourceRuntime::new(resource_runtime);
    let policy_guard = PolicyGuardedRuntime::new(policy_runtime, policy_verifier);
    let runtime = BudgetGuardedRuntime::new(policy_guard, budgets.verifier());

    Fixture {
        evaluator,
        execution,
        observation,
        resolution,
        resources,
        budgets,
        runtime,
    }
}

fn policy_grant(
    fixture: &Fixture,
    actor: PrincipalId,
    binding: [u8; 32],
    risk: ActionRisk,
) -> symthaea_ai_assurance::PolicyGrant<Write> {
    let admission = fixture.evaluator.admit(
        binding,
        scope(),
        risk,
        PolicyMode::Autonomous,
        ApprovalEvidence::new([4; 32], [5; 32], true),
        [6; 32],
        [7; 32],
        [8; 32],
        None,
    );
    fixture
        .execution
        .issue::<Write>(actor, scope(), None, binding, admission)
        .unwrap()
}

fn budget_allocation() -> BudgetQuantities {
    BudgetQuantities::zero()
        .with(BudgetDimension::ComputeUnits, 10)
        .with(BudgetDimension::BytesWritten, 512)
        .with(BudgetDimension::Subprocesses, 1)
}

#[test]
fn budget_revocation_after_authorization_blocks_adapter_entry() {
    let fixture = fixture();
    let actor = PrincipalId::new();
    let action = fixture
        .runtime
        .admit_resolved::<Write, _>(
            actor,
            "edit-source",
            fixture
                .resources
                .resolve(0_u64, resource_identity(), None),
            b"patch",
        )
        .unwrap()
        .assess(ActionRisk::Reversible);
    let binding = action.authorization_binding();
    let policy_grant = policy_grant(&fixture, actor, binding, action.risk());
    let budget_lease = fixture
        .budgets
        .reserve(actor, scope(), binding, budget_allocation(), None)
        .unwrap();
    let action = action.authorize(policy_grant, budget_lease).unwrap();

    fixture.budgets.revoke_all().unwrap();
    let entered = AtomicBool::new(false);
    let result = action.execute_with(|_| -> Result<[u8; 32], &'static str> {
        entered.store(true, Ordering::SeqCst);
        Ok([9; 32])
    });

    assert!(result.is_err());
    assert!(!entered.load(Ordering::SeqCst));
}

#[test]
fn budget_revocation_after_effect_preserves_evidence_collection() {
    let fixture = fixture();
    let actor = PrincipalId::new();
    let observer = PrincipalId::new();
    let resolver = PrincipalId::new();
    let allocation = budget_allocation();
    let action = fixture
        .runtime
        .admit_resolved::<Write, _>(
            actor,
            "edit-source",
            fixture
                .resources
                .resolve(0_u64, resource_identity(), None),
            b"patch",
        )
        .unwrap()
        .assess(ActionRisk::Reversible);
    let binding = action.authorization_binding();
    let policy_grant = policy_grant(&fixture, actor, binding, action.risk());
    let budget_lease = fixture
        .budgets
        .reserve(actor, scope(), binding, allocation, None)
        .unwrap();
    let expected_lease = budget_lease.lease_id();
    let expected_profile = budget_lease.profile().digest();

    let action = action
        .authorize(policy_grant, budget_lease)
        .unwrap()
        .execute_with(|handle| -> Result<[u8; 32], &'static str> {
            *handle += 1;
            Ok([10; 32])
        })
        .unwrap();

    fixture.budgets.revoke_all().unwrap();

    let observer_grant = fixture.observation.issue_bound_one_shot::<Observe>(
        observer,
        scope(),
        None,
        action.observation_binding(),
    );
    let action = action
        .observe(
            observer_grant,
            Observation::new(ObservedOutcome::Success, [11; 32]),
        )
        .unwrap();
    let decision = ResolutionDecision::Confirmed;
    let resolution_grant = fixture.resolution.issue_bound_one_shot(
        resolver,
        scope(),
        None,
        action.resolution_binding(decision),
    );
    let (resolved, receipt) = action.resolve(resolution_grant, decision).unwrap();

    assert_eq!(receipt.budget().lease_id(), expected_lease);
    assert_eq!(receipt.budget().profile_digest(), expected_profile);
    assert_eq!(receipt.budget().allocation(), allocation);
    assert_eq!(receipt.budget().action_binding(), binding);
    assert_eq!(receipt.budget().domain(), fixture.budgets.domain_id());

    let release = resolved.release_budget().unwrap();
    assert_eq!(release.lease_id(), expected_lease);
    assert_eq!(release.released(), allocation);
}

#[test]
fn budget_lease_for_another_action_is_rejected_before_authorization() {
    let fixture = fixture();
    let actor = PrincipalId::new();
    let action = fixture
        .runtime
        .admit_resolved::<Write, _>(
            actor,
            "edit-source",
            fixture
                .resources
                .resolve(0_u64, resource_identity(), None),
            b"patch-a",
        )
        .unwrap()
        .assess(ActionRisk::Reversible);
    let binding = action.authorization_binding();
    let policy_grant = policy_grant(&fixture, actor, binding, action.risk());
    let wrong_budget = fixture
        .budgets
        .reserve(actor, scope(), [99; 32], budget_allocation(), None)
        .unwrap();

    assert!(action.authorize(policy_grant, wrong_budget).is_err());
}

#[test]
fn recoverable_wrong_action_budget_returns_exact_original_lease() {
    let fixture = fixture();
    let actor = PrincipalId::new();
    let action = fixture
        .runtime
        .admit_resolved::<Write, _>(
            actor,
            "edit-source",
            fixture
                .resources
                .resolve(0_u64, resource_identity(), None),
            b"patch-recover",
        )
        .unwrap()
        .assess(ActionRisk::Reversible);
    let binding = action.authorization_binding();
    let policy_grant = policy_grant(&fixture, actor, binding, action.risk());
    let allocation = budget_allocation();
    let lease = fixture
        .budgets
        .reserve(actor, scope(), [98; 32], allocation, None)
        .unwrap();
    let lease_id = lease.lease_id();
    let lease_domain = lease.domain_id();
    let lease_epoch = lease.epoch();
    let remaining_before = fixture.budgets.remaining();

    let failure = action
        .authorize_recoverable(policy_grant, lease)
        .unwrap_err();
    assert!(matches!(
        failure.error(),
        BudgetGuardedAuthorizeError::Budget(_)
    ));
    assert_eq!(failure.budget_lease().lease_id(), lease_id);
    assert_eq!(failure.budget_lease().allocation(), allocation);
    assert_eq!(failure.budget_lease().domain_id(), lease_domain);
    assert_eq!(failure.budget_lease().epoch(), lease_epoch);
    assert_eq!(fixture.budgets.remaining(), remaining_before);

    failure.into_budget_lease().release().unwrap();
    assert_eq!(fixture.budgets.remaining(), fixture.budgets.profile().limits());
}

#[test]
fn recoverable_policy_rejection_returns_exact_original_lease() {
    let fixture = fixture();
    let actor = PrincipalId::new();
    let action = fixture
        .runtime
        .admit_resolved::<Write, _>(
            actor,
            "edit-source",
            fixture
                .resources
                .resolve(0_u64, resource_identity(), None),
            b"patch-policy-recover",
        )
        .unwrap()
        .assess(ActionRisk::Reversible);
    let binding = action.authorization_binding();

    let rogue_evaluator = PolicyEvaluatorDomain::new(
        PrincipalId::new(),
        PolicyDescriptor::new("rogue-policy", 1, [44; 32], 1).unwrap(),
    );
    let rogue_execution =
        PolicyExecutionDomain::new(PrincipalId::new(), rogue_evaluator.verifier());
    let rogue_admission = rogue_evaluator.admit(
        binding,
        scope(),
        action.risk(),
        PolicyMode::Autonomous,
        ApprovalEvidence::new([45; 32], [46; 32], true),
        [47; 32],
        [48; 32],
        [49; 32],
        None,
    );
    let rogue_grant = rogue_execution
        .issue::<Write>(actor, scope(), None, binding, rogue_admission)
        .unwrap();

    let allocation = budget_allocation();
    let lease = fixture
        .budgets
        .reserve(actor, scope(), binding, allocation, None)
        .unwrap();
    let lease_id = lease.lease_id();
    let lease_domain = lease.domain_id();
    let lease_epoch = lease.epoch();
    let remaining_before = fixture.budgets.remaining();

    let failure = action
        .authorize_recoverable(rogue_grant, lease)
        .unwrap_err();
    assert!(matches!(
        failure.error(),
        BudgetGuardedAuthorizeError::Policy(_)
    ));
    assert_eq!(failure.budget_lease().lease_id(), lease_id);
    assert_eq!(failure.budget_lease().allocation(), allocation);
    assert_eq!(failure.budget_lease().domain_id(), lease_domain);
    assert_eq!(failure.budget_lease().epoch(), lease_epoch);
    assert_eq!(fixture.budgets.remaining(), remaining_before);

    failure.into_budget_lease().release().unwrap();
    assert_eq!(fixture.budgets.remaining(), fixture.budgets.profile().limits());
}
