use std::sync::atomic::{AtomicBool, Ordering};

use symthaea_ai_assurance::{
    ActionRisk, AdapterSchema, AuthorityDomain, Observation, Observe, ObservedOutcome, PrincipalId,
    ResolutionAuthorityDomain, ResolutionDecision, ResourceIdentity, ResourceResolverDomain,
    ResourceRuntime, Scope, TrustedRuntime, Write,
};

fn scope() -> Scope {
    Scope::new("workspace", ["symthaea", "src"]).unwrap()
}

fn identity(stable: u8, environment: u8) -> ResourceIdentity {
    ResourceIdentity::new(
        scope(),
        "worktree-file",
        [stable; 32],
        [environment; 32],
        AdapterSchema::new("public-test-adapter", 1).unwrap(),
    )
    .unwrap()
}

fn runtime(
    resources: &ResourceResolverDomain,
) -> (
    AuthorityDomain,
    AuthorityDomain,
    ResolutionAuthorityDomain,
    ResourceRuntime,
) {
    let execution = AuthorityDomain::new(PrincipalId::new());
    let observation = AuthorityDomain::new(PrincipalId::new());
    let resolution = ResolutionAuthorityDomain::new(PrincipalId::new());
    let strict = TrustedRuntime::new(
        execution.verifier(),
        observation.verifier(),
        resolution.verifier(),
    );
    (
        execution,
        observation,
        resolution,
        ResourceRuntime::new(strict, resources.verifier()),
    )
}

#[test]
fn public_api_rejects_cross_resource_authority_replay() {
    let resources = ResourceResolverDomain::new(PrincipalId::new());
    let (execution, _, _, runtime) = runtime(&resources);
    let actor = PrincipalId::new();

    let first = runtime
        .admit_resolved::<Write, _>(
            actor,
            "edit-source",
            resources.resolve((), identity(1, 1), None),
            b"same-operation",
        )
        .unwrap()
        .assess(ActionRisk::Reversible);

    let second = runtime
        .admit_resolved::<Write, _>(
            actor,
            "edit-source",
            resources.resolve((), identity(2, 1), None),
            b"same-operation",
        )
        .unwrap()
        .assess(ActionRisk::Reversible);

    let grant = execution.issue_bound_one_shot::<Write>(
        actor,
        scope(),
        None,
        first.authorization_binding(),
    );

    assert!(second.authorize(grant).is_err());
}

#[test]
fn public_api_rechecks_resource_revocation_before_adapter_entry() {
    let resources = ResourceResolverDomain::new(PrincipalId::new());
    let (execution, _, _, runtime) = runtime(&resources);
    let actor = PrincipalId::new();

    let action = runtime
        .admit_resolved::<Write, _>(
            actor,
            "edit-source",
            resources.resolve(0_u64, identity(3, 1), None),
            b"operation",
        )
        .unwrap()
        .assess(ActionRisk::Reversible);

    let grant = execution.issue_bound_one_shot::<Write>(
        actor,
        scope(),
        None,
        action.authorization_binding(),
    );
    let action = action.authorize(grant).unwrap();

    resources.revoke_all().unwrap();
    let entered = AtomicBool::new(false);
    let result = action.execute_with(|handle| -> Result<[u8; 32], &'static str> {
        entered.store(true, Ordering::SeqCst);
        *handle += 1;
        Ok([4; 32])
    });

    assert!(result.is_err());
    assert!(!entered.load(Ordering::SeqCst));
}

#[test]
fn public_api_preserves_resource_identity_in_final_evidence() {
    let resources = ResourceResolverDomain::new(PrincipalId::new());
    let (execution, observation, resolution, runtime) = runtime(&resources);
    let actor = PrincipalId::new();
    let observer = PrincipalId::new();
    let resolver = PrincipalId::new();
    let expected_identity = identity(7, 8);

    let action = runtime
        .admit_resolved::<Write, _>(
            actor,
            "edit-source",
            resources.resolve(5_u64, expected_identity.clone(), None),
            b"operation",
        )
        .unwrap()
        .assess(ActionRisk::Reversible);

    let execution_grant = execution.issue_bound_one_shot::<Write>(
        actor,
        scope(),
        None,
        action.authorization_binding(),
    );
    let action = action
        .authorize(execution_grant)
        .unwrap()
        .execute_with(|handle| -> Result<[u8; 32], &'static str> {
            *handle += 1;
            Ok([9; 32])
        })
        .unwrap();

    let observer_grant = observation.issue_bound_one_shot::<Observe>(
        observer,
        scope(),
        None,
        action.observation_binding(),
    );
    let action = action
        .observe(
            observer_grant,
            Observation::new(ObservedOutcome::Success, [10; 32]),
        )
        .unwrap();

    let decision = ResolutionDecision::Confirmed;
    let resolution_grant = resolution.issue_bound_one_shot(
        resolver,
        scope(),
        None,
        action.resolution_binding(decision),
    );
    let (_, receipt) = action.resolve(resolution_grant, decision).unwrap();

    assert_eq!(receipt.resource_identity(), &expected_identity);
    assert_eq!(receipt.resource_resolver_domain(), resources.domain_id());
}
