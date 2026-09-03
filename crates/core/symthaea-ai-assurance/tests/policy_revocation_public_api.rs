use std::sync::atomic::{AtomicBool, Ordering};

use symthaea_ai_assurance::{
    ActionRisk, AdapterSchema, ApprovalEvidence, AuthorityDomain, Observation, Observe,
    ObservedOutcome, PolicyDescriptor, PolicyEvaluatorDomain, PolicyExecutionDomain,
    PolicyGuardError, PolicyGuardedAuthorizeError, PolicyGuardedExecutionError,
    PolicyGuardedRuntime, PolicyMode, PolicyResourceRuntime, PrincipalId,
    ResolutionAuthorityDomain, ResolutionDecision, ResourceIdentity, ResourceResolverDomain,
    ResourceRuntime, Scope, TrustedRuntime, Write,
};

fn scope() -> Scope {
    Scope::new("workspace", ["symthaea", "src"]).unwrap()
}

fn identity() -> ResourceIdentity {
    ResourceIdentity::new(
        scope(),
        "worktree-file",
        [1; 32],
        [2; 32],
        AdapterSchema::new("policy-revocation-public", 1).unwrap(),
    )
    .unwrap()
}

struct Fixture {
    evaluator: PolicyEvaluatorDomain,
    execution: PolicyExecutionDomain,
    observation: AuthorityDomain,
    resolution: ResolutionAuthorityDomain,
    resources: ResourceResolverDomain,
    runtime: PolicyGuardedRuntime,
}

fn fixture() -> Fixture {
    let descriptor = PolicyDescriptor::new("magi-gate", 1, [3; 32], 1).unwrap();
    let evaluator = PolicyEvaluatorDomain::new(PrincipalId::new(), descriptor);
    let policy_verifier = evaluator.verifier();
    let execution = PolicyExecutionDomain::new(PrincipalId::new(), policy_verifier.clone());
    let observation = AuthorityDomain::new(PrincipalId::new());
    let resolution = ResolutionAuthorityDomain::new(PrincipalId::new());
    let resources = ResourceResolverDomain::new(PrincipalId::new());
    let strict = TrustedRuntime::new(
        execution.verifier(),
        observation.verifier(),
        resolution.verifier(),
    );
    let resource_runtime = ResourceRuntime::new(strict, resources.verifier());
    let policy_runtime = PolicyResourceRuntime::new(resource_runtime);
    let runtime = PolicyGuardedRuntime::new(policy_runtime, policy_verifier);
    Fixture {
        evaluator,
        execution,
        observation,
        resolution,
        resources,
        runtime,
    }
}

fn policy_grant(
    fixture: &Fixture,
    actor: PrincipalId,
    action_binding: [u8; 32],
    risk: ActionRisk,
) -> symthaea_ai_assurance::PolicyGrant<Write> {
    let admission = fixture.evaluator.admit(
        action_binding,
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
        .issue::<Write>(actor, scope(), None, action_binding, admission)
        .unwrap()
}

#[test]
fn public_guard_rejects_policy_revocation_after_grant_minting() {
    let fixture = fixture();
    let actor = PrincipalId::new();
    let action = fixture
        .runtime
        .admit_resolved::<Write, _>(
            actor,
            "edit-source",
            fixture.resources.resolve((), identity(), None),
            b"patch",
        )
        .unwrap()
        .assess(ActionRisk::Reversible);
    let grant = policy_grant(
        &fixture,
        actor,
        action.authorization_binding(),
        action.risk(),
    );
    fixture.evaluator.revoke_all().unwrap();

    assert!(matches!(
        action.authorize(grant),
        Err(PolicyGuardedAuthorizeError::Guard(
            PolicyGuardError::RevokedPolicyEpoch { .. }
        ))
    ));
}

#[test]
fn public_guard_rejects_policy_revocation_after_authorization_before_effect() {
    let fixture = fixture();
    let actor = PrincipalId::new();
    let action = fixture
        .runtime
        .admit_resolved::<Write, _>(
            actor,
            "edit-source",
            fixture.resources.resolve(0_u64, identity(), None),
            b"patch",
        )
        .unwrap()
        .assess(ActionRisk::StateModifying);
    let grant = policy_grant(
        &fixture,
        actor,
        action.authorization_binding(),
        action.risk(),
    );
    let action = action.authorize(grant).unwrap();
    fixture.evaluator.revoke_all().unwrap();

    let entered = AtomicBool::new(false);
    let result = action.execute_with(|handle| -> Result<[u8; 32], &'static str> {
        entered.store(true, Ordering::SeqCst);
        *handle += 1;
        Ok([9; 32])
    });
    assert!(matches!(
        result,
        Err(PolicyGuardedExecutionError::Guard(
            PolicyGuardError::RevokedPolicyEpoch { .. }
        ))
    ));
    assert!(!entered.load(Ordering::SeqCst));
}

#[test]
fn policy_rotation_after_effect_does_not_block_evidence_collection() {
    let fixture = fixture();
    let actor = PrincipalId::new();
    let observer = PrincipalId::new();
    let resolver = PrincipalId::new();
    let action = fixture
        .runtime
        .admit_resolved::<Write, _>(
            actor,
            "edit-source",
            fixture.resources.resolve(0_u64, identity(), None),
            b"patch",
        )
        .unwrap()
        .assess(ActionRisk::Reversible);
    let grant = policy_grant(
        &fixture,
        actor,
        action.authorization_binding(),
        action.risk(),
    );
    let action = action
        .authorize(grant)
        .unwrap()
        .execute_with(|handle| -> Result<[u8; 32], &'static str> {
            *handle += 1;
            Ok([10; 32])
        })
        .unwrap();

    fixture.evaluator.revoke_all().unwrap();

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
    let resolver_grant = fixture.resolution.issue_bound_one_shot(
        resolver,
        scope(),
        None,
        action.resolution_binding(decision),
    );
    let (_, receipt) = action.resolve(resolver_grant, decision).unwrap();

    assert_eq!(
        receipt.policy_evidence().policy_domain(),
        fixture.evaluator.domain_id()
    );
}
