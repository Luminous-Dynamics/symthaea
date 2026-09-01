use std::time::{Duration, SystemTime};

use symthaea_ai_assurance::{
    ActionRisk, AdapterSchema, ApprovalEvidence, AuthorityDomain, BudgetAuthorityDomain,
    BudgetDimension, BudgetEnforcement, BudgetGuardedRuntime, BudgetProfile,
    BudgetPurposeAuthorityDomain, BudgetPurposeDescriptor, BudgetPurposeError, BudgetPurposeRules,
    BudgetQuantities, EffectAttemptOutcome, EffectGuardedRuntime, EnforcementClass,
    IndependenceGuardedRuntime, IndependencePolicy, Observation, Observe, ObservedOutcome,
    PolicyDescriptor, PolicyGuardedRuntime, PolicyMode, PolicyResourceRuntime, PrincipalId,
    PurposeGuardedRuntime, ResolutionAuthorityDomain, ResolutionDecision, ResourceIdentity,
    ResourceResolverDomain, ResourceRuntime, Scope, TemporalPolicyEvaluatorDomain,
    TemporalPolicyExecutionDomain, TemporalPolicyRules, TrustedRuntime, Write,
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
        AdapterSchema::new("purpose-public-test", 1).unwrap(),
    )
    .unwrap()
}

fn budget_profile() -> BudgetProfile {
    let limits = BudgetQuantities::zero().with(BudgetDimension::ComputeUnits, 10);
    let enforcement = BudgetEnforcement::soft()
        .with(BudgetDimension::ComputeUnits, EnforcementClass::CoreMetered);
    BudgetProfile::new(limits, enforcement, None).unwrap()
}

struct Harness {
    evaluator: TemporalPolicyEvaluatorDomain,
    execution: TemporalPolicyExecutionDomain,
    observation: AuthorityDomain,
    resolution: ResolutionAuthorityDomain,
    resources: ResourceResolverDomain,
    budgets: BudgetAuthorityDomain,
    purpose: BudgetPurposeAuthorityDomain,
    runtime: PurposeGuardedRuntime,
}

fn harness() -> Harness {
    let rules = TemporalPolicyRules::strict();
    let evaluator = TemporalPolicyEvaluatorDomain::new(
        PrincipalId::new(),
        PolicyDescriptor::new("purpose-general", 1, [3; 32], 1).unwrap(),
        rules,
    );
    let execution =
        TemporalPolicyExecutionDomain::new(PrincipalId::new(), evaluator.verifier(), rules);
    let observation = AuthorityDomain::new(PrincipalId::new());
    let resolution = ResolutionAuthorityDomain::new(PrincipalId::new());
    let resources = ResourceResolverDomain::new(PrincipalId::new());
    let budgets = BudgetAuthorityDomain::new(PrincipalId::new(), budget_profile());

    let strict = TrustedRuntime::new(
        execution.verifier(),
        observation.verifier(),
        resolution.verifier(),
    );
    let resource_runtime = ResourceRuntime::new(strict, resources.verifier());
    let policy_runtime = PolicyResourceRuntime::new(resource_runtime);
    let policy_guard = PolicyGuardedRuntime::new(policy_runtime, evaluator.verifier());
    let budget_runtime = BudgetGuardedRuntime::new(policy_guard, budgets.verifier());
    let effect = EffectGuardedRuntime::new(budget_runtime, execution.verifier());
    let independence = IndependenceGuardedRuntime::new(
        effect,
        observation.verifier(),
        resolution.verifier(),
        IndependencePolicy::SeparationOfDuties,
    )
    .unwrap();

    let purpose = BudgetPurposeAuthorityDomain::new(
        PrincipalId::new(),
        BudgetPurposeDescriptor::new("magi-budget-purpose", 1, [4; 32], 1).unwrap(),
        BudgetPurposeRules::strict(),
        budgets.verifier(),
        evaluator.verifier(),
        execution.verifier(),
    );
    let runtime = PurposeGuardedRuntime::new(independence, purpose.verifier(), budgets.verifier());

    Harness {
        evaluator,
        execution,
        observation,
        resolution,
        resources,
        budgets,
        purpose,
        runtime,
    }
}

#[test]
fn public_end_to_end_evidence_binds_policy_quantity_and_purpose() {
    let h = harness();
    let expiry = SystemTime::now() + Duration::from_secs(60);
    let actor = PrincipalId::new();
    let action = h
        .runtime
        .admit_resolved::<Write, _>(
            actor,
            "edit-source",
            h.resources
                .resolve(0_u64, resource_identity(), Some(expiry)),
            b"patch-v1",
        )
        .unwrap()
        .assess(ActionRisk::Reversible);
    let binding = action.authorization_binding();

    let admission = h
        .evaluator
        .admit(
            binding,
            scope(),
            action.risk(),
            PolicyMode::Autonomous,
            ApprovalEvidence::new([5; 32], [6; 32], true),
            [7; 32],
            [8; 32],
            [9; 32],
            Some(expiry),
        )
        .unwrap();
    let grant = h
        .execution
        .issue::<Write>(actor, scope(), Some(expiry), binding, admission)
        .unwrap();
    let lease = h
        .budgets
        .reserve(
            actor,
            scope(),
            binding,
            BudgetQuantities::zero().with(BudgetDimension::ComputeUnits, 3),
            Some(expiry),
        )
        .unwrap();
    let purpose_digest = [10; 32];
    let purpose_lease = h
        .purpose
        .approve(&grant, lease, actor, purpose_digest, Some(expiry))
        .unwrap();

    let action = action
        .authorize(grant, purpose_lease)
        .unwrap()
        .execute_attempt_with(|handle| {
            *handle += 1;
            EffectAttemptOutcome::Succeeded {
                evidence_digest: [11; 32],
            }
        })
        .unwrap();

    let observer = PrincipalId::new();
    let observer_grant = h.observation.issue_bound_one_shot::<Observe>(
        observer,
        scope(),
        Some(expiry),
        action.observation_binding(),
    );
    let action = action
        .observe(
            observer_grant,
            Observation::new(ObservedOutcome::Success, [12; 32]),
        )
        .unwrap();

    let resolver = PrincipalId::new();
    let decision = ResolutionDecision::Confirmed;
    let resolution_grant = h.resolution.issue_bound_one_shot(
        resolver,
        scope(),
        Some(expiry),
        action.resolution_binding(decision),
    );
    let (resolved, receipt) = action.resolve(resolution_grant, decision).unwrap();

    let purpose = receipt.purpose_evidence();
    assert_eq!(purpose.receipt().action_binding(), binding);
    assert_eq!(purpose.receipt().subject(), actor);
    assert_eq!(purpose.receipt().scope(), &scope());
    assert_eq!(purpose.receipt().purpose_digest(), purpose_digest);
    assert_eq!(
        purpose
            .receipt()
            .allocation()
            .get(BudgetDimension::ComputeUnits),
        3
    );
    assert_eq!(purpose.receipt().budget_domain(), h.budgets.domain_id());
    assert_eq!(purpose.receipt().policy_domain(), h.evaluator.domain_id());
    assert_eq!(
        purpose.receipt().execution_domain(),
        h.execution.domain_id()
    );
    assert_eq!(purpose.purpose_domain(), h.purpose.domain_id());
    assert_eq!(resolved.purpose_receipt(), &receipt);
}

#[test]
fn public_purpose_approval_rejects_lifetime_wider_than_budget() {
    let h = harness();
    let parent_expiry = SystemTime::now() + Duration::from_secs(30);
    let actor = PrincipalId::new();
    let action = h
        .runtime
        .admit_resolved::<Write, _>(
            actor,
            "edit-source",
            h.resources
                .resolve(0_u64, resource_identity(), Some(parent_expiry)),
            b"patch-v2",
        )
        .unwrap()
        .assess(ActionRisk::Reversible);
    let binding = action.authorization_binding();
    let admission = h
        .evaluator
        .admit(
            binding,
            scope(),
            action.risk(),
            PolicyMode::Autonomous,
            ApprovalEvidence::new([13; 32], [14; 32], true),
            [15; 32],
            [16; 32],
            [17; 32],
            Some(parent_expiry),
        )
        .unwrap();
    let grant = h
        .execution
        .issue::<Write>(actor, scope(), Some(parent_expiry), binding, admission)
        .unwrap();
    let lease = h
        .budgets
        .reserve(
            actor,
            scope(),
            binding,
            BudgetQuantities::zero().with(BudgetDimension::ComputeUnits, 1),
            Some(parent_expiry),
        )
        .unwrap();

    let result = h.purpose.approve(
        &grant,
        lease,
        actor,
        [18; 32],
        Some(parent_expiry + Duration::from_secs(1)),
    );
    assert!(matches!(
        result,
        Err(BudgetPurposeError::PurposeExpiryWidening { .. })
    ));
}

#[test]
fn public_unrelated_purpose_root_cannot_validate_same_lease() {
    let h = harness();
    let expiry = SystemTime::now() + Duration::from_secs(60);
    let actor = PrincipalId::new();
    let action = h
        .runtime
        .admit_resolved::<Write, _>(
            actor,
            "edit-source",
            h.resources
                .resolve(0_u64, resource_identity(), Some(expiry)),
            b"patch-v3",
        )
        .unwrap()
        .assess(ActionRisk::Reversible);
    let binding = action.authorization_binding();
    let admission = h
        .evaluator
        .admit(
            binding,
            scope(),
            action.risk(),
            PolicyMode::Autonomous,
            ApprovalEvidence::new([19; 32], [20; 32], true),
            [21; 32],
            [22; 32],
            [23; 32],
            Some(expiry),
        )
        .unwrap();
    let grant = h
        .execution
        .issue::<Write>(actor, scope(), Some(expiry), binding, admission)
        .unwrap();
    let lease = h
        .budgets
        .reserve(
            actor,
            scope(),
            binding,
            BudgetQuantities::zero().with(BudgetDimension::ComputeUnits, 2),
            Some(expiry),
        )
        .unwrap();
    let purpose_lease = h
        .purpose
        .approve(&grant, lease, actor, [24; 32], Some(expiry))
        .unwrap();

    let wrong = BudgetPurposeAuthorityDomain::new(
        PrincipalId::new(),
        BudgetPurposeDescriptor::new("magi-budget-purpose", 1, [4; 32], 1).unwrap(),
        BudgetPurposeRules::strict(),
        h.budgets.verifier(),
        h.evaluator.verifier(),
        h.execution.verifier(),
    );
    assert!(
        purpose_lease
            .validate_for(
                &wrong.verifier(),
                &h.budgets.verifier(),
                &grant,
                actor,
                &scope(),
                binding,
            )
            .is_err()
    );
}
