use symthaea_ai_assurance::{
    ActionRisk, AuthorityDomain, Observation, Observe, Observed, ObservedOutcome, PrincipalId,
    ResolutionAuthorityDomain, ResolutionDecision, ResolutionError, RuntimeAction, Scope,
    TrustedRuntime, Write,
};

fn scope() -> Scope {
    Scope::new("workspace", ["symthaea", "src"]).unwrap()
}

fn observed_action(
    execution: &AuthorityDomain,
    observation: &AuthorityDomain,
    runtime: &TrustedRuntime,
    actor: PrincipalId,
    observer: PrincipalId,
    payload: &[u8],
    output_digest: [u8; 32],
    evidence_digest: [u8; 32],
) -> RuntimeAction<Write, Observed> {
    let action_scope = scope();
    let action = runtime
        .admit::<Write>(actor, "edit-source", action_scope.clone(), payload)
        .assess(ActionRisk::Reversible);
    let execution_grant = execution.issue_bound_one_shot::<Write>(
        actor,
        action_scope.clone(),
        None,
        action.authorization_binding(),
    );
    let action = action
        .authorize(execution_grant)
        .unwrap()
        .record_execution(output_digest)
        .unwrap();
    let observer_grant = observation.issue_bound_one_shot::<Observe>(
        observer,
        action_scope,
        None,
        action.observation_binding(),
    );
    action
        .observe(
            observer_grant,
            Observation::new(ObservedOutcome::Success, evidence_digest),
        )
        .unwrap()
}

#[test]
fn resolution_grant_cannot_replay_across_actions() {
    let execution = AuthorityDomain::new(PrincipalId::new());
    let observation = AuthorityDomain::new(PrincipalId::new());
    let resolution = ResolutionAuthorityDomain::new(PrincipalId::new());
    let runtime = TrustedRuntime::new(
        execution.verifier(),
        observation.verifier(),
        resolution.verifier(),
    );
    let actor = PrincipalId::new();
    let observer = PrincipalId::new();

    let action_a = observed_action(
        &execution,
        &observation,
        &runtime,
        actor,
        observer,
        b"patch-a",
        [1; 32],
        [2; 32],
    );
    let action_b = observed_action(
        &execution,
        &observation,
        &runtime,
        actor,
        observer,
        b"patch-b",
        [1; 32],
        [2; 32],
    );
    let grant = resolution.issue_bound_one_shot(
        PrincipalId::new(),
        scope(),
        None,
        action_a.resolution_binding(ResolutionDecision::Confirmed),
    );

    assert!(matches!(
        action_b.resolve(grant, ResolutionDecision::Confirmed),
        Err(ResolutionError::BindingMismatch { .. })
    ));
}

#[test]
fn observation_evidence_changes_resolution_binding() {
    let execution = AuthorityDomain::new(PrincipalId::new());
    let observation = AuthorityDomain::new(PrincipalId::new());
    let resolution = ResolutionAuthorityDomain::new(PrincipalId::new());
    let runtime = TrustedRuntime::new(
        execution.verifier(),
        observation.verifier(),
        resolution.verifier(),
    );
    let actor = PrincipalId::new();
    let observer = PrincipalId::new();

    let first = observed_action(
        &execution,
        &observation,
        &runtime,
        actor,
        observer,
        b"same-patch",
        [3; 32],
        [4; 32],
    );
    let second = observed_action(
        &execution,
        &observation,
        &runtime,
        actor,
        observer,
        b"same-patch",
        [3; 32],
        [5; 32],
    );

    assert_ne!(
        first.resolution_binding(ResolutionDecision::Confirmed),
        second.resolution_binding(ResolutionDecision::Confirmed)
    );
}

#[test]
fn each_final_decision_has_a_distinct_binding() {
    let execution = AuthorityDomain::new(PrincipalId::new());
    let observation = AuthorityDomain::new(PrincipalId::new());
    let resolution = ResolutionAuthorityDomain::new(PrincipalId::new());
    let runtime = TrustedRuntime::new(
        execution.verifier(),
        observation.verifier(),
        resolution.verifier(),
    );
    let observed = observed_action(
        &execution,
        &observation,
        &runtime,
        PrincipalId::new(),
        PrincipalId::new(),
        b"patch",
        [6; 32],
        [7; 32],
    );

    let confirmed = observed.resolution_binding(ResolutionDecision::Confirmed);
    let contradicted = observed.resolution_binding(ResolutionDecision::Contradicted);
    let inconclusive = observed.resolution_binding(ResolutionDecision::Inconclusive);

    assert_ne!(confirmed, contradicted);
    assert_ne!(confirmed, inconclusive);
    assert_ne!(contradicted, inconclusive);
}
