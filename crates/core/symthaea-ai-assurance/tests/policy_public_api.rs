use symthaea_ai_assurance::{
    ActionRisk, AdapterSchema, ApprovalEvidence, AuthorityDomain, Observation, Observe,
    ObservedOutcome, PolicyDescriptor, PolicyError, PolicyEvaluatorDomain, PolicyExecutionDomain,
    PolicyMode, PolicyResourceRuntime, PrincipalId, ResolutionAuthorityDomain, ResolutionDecision,
    ResourceIdentity, ResourceResolverDomain, ResourceRuntime, Scope, TrustedRuntime, Write,
};

fn scope() -> Scope {
    Scope::new("workspace", ["symthaea", "src"]).unwrap()
}

fn descriptor(tag: u8) -> PolicyDescriptor {
    PolicyDescriptor::new("magi-gate", 1, [tag; 32], 1).unwrap()
}

fn approvals(satisfied: bool, tag: u8) -> ApprovalEvidence {
    ApprovalEvidence::new([tag; 32], [tag.wrapping_add(1); 32], satisfied)
}

fn identity(tag: u8) -> ResourceIdentity {
    ResourceIdentity::new(
        scope(),
        "worktree-file",
        [tag; 32],
        [42; 32],
        AdapterSchema::new("policy-public-test", 1).unwrap(),
    )
    .unwrap()
}

fn runtime(
    execution: &PolicyExecutionDomain,
    observation: &AuthorityDomain,
    resolution: &ResolutionAuthorityDomain,
    resources: &ResourceResolverDomain,
) -> PolicyResourceRuntime {
    let strict = TrustedRuntime::new(
        execution.verifier(),
        observation.verifier(),
        resolution.verifier(),
    );
    PolicyResourceRuntime::new(ResourceRuntime::new(strict, resources.verifier()))
}

#[test]
fn unrelated_policy_evaluator_cannot_justify_execution() {
    let trusted = PolicyEvaluatorDomain::new(PrincipalId::new(), descriptor(1));
    let attacker = PolicyEvaluatorDomain::new(PrincipalId::new(), descriptor(1));
    let execution = PolicyExecutionDomain::new(PrincipalId::new(), trusted.verifier());
    let binding = [7; 32];
    let admission = attacker.admit(
        binding,
        scope(),
        ActionRisk::Reversible,
        PolicyMode::Autonomous,
        approvals(true, 2),
        [3; 32],
        [4; 32],
        [5; 32],
        None,
    );

    assert!(execution
        .issue::<Write>(PrincipalId::new(), scope(), None, binding, admission)
        .is_err());
}

#[test]
fn revoking_policy_epoch_invalidates_unconsumed_admission() {
    let evaluator = PolicyEvaluatorDomain::new(PrincipalId::new(), descriptor(1));
    let execution = PolicyExecutionDomain::new(PrincipalId::new(), evaluator.verifier());
    let binding = [7; 32];
    let admission = evaluator.admit(
        binding,
        scope(),
        ActionRisk::Reversible,
        PolicyMode::Autonomous,
        approvals(true, 2),
        [3; 32],
        [4; 32],
        [5; 32],
        None,
    );
    evaluator.revoke_all().unwrap();

    assert!(execution
        .issue::<Write>(PrincipalId::new(), scope(), None, binding, admission)
        .is_err());
}

#[test]
fn supervised_execution_requires_satisfied_approval_evidence() {
    let evaluator = PolicyEvaluatorDomain::new(PrincipalId::new(), descriptor(1));
    let execution = PolicyExecutionDomain::new(PrincipalId::new(), evaluator.verifier());
    let binding = [8; 32];
    let admission = evaluator.admit(
        binding,
        scope(),
        ActionRisk::StateModifying,
        PolicyMode::Supervised,
        approvals(false, 9),
        [3; 32],
        [4; 32],
        [5; 32],
        None,
    );

    assert!(matches!(
        execution.issue::<Write>(PrincipalId::new(), scope(), None, binding, admission),
        Err(PolicyError::SupervisionUnsatisfied)
    ));
}

#[test]
fn policy_grant_for_action_a_cannot_authorize_action_b() {
    let evaluator = PolicyEvaluatorDomain::new(PrincipalId::new(), descriptor(1));
    let execution = PolicyExecutionDomain::new(PrincipalId::new(), evaluator.verifier());
    let observation = AuthorityDomain::new(PrincipalId::new());
    let resolution = ResolutionAuthorityDomain::new(PrincipalId::new());
    let resources = ResourceResolverDomain::new(PrincipalId::new());
    let runtime = runtime(&execution, &observation, &resolution, &resources);
    let actor = PrincipalId::new();

    let action_a = runtime
        .admit_resolved::<Write, _>(
            actor,
            "edit-source",
            resources.resolve((), identity(1), None),
            b"same-operation",
        )
        .unwrap()
        .assess(ActionRisk::Reversible);
    let action_b = runtime
        .admit_resolved::<Write, _>(
            actor,
            "edit-source",
            resources.resolve((), identity(2), None),
            b"same-operation",
        )
        .unwrap()
        .assess(ActionRisk::Reversible);

    let admission = evaluator.admit(
        action_a.authorization_binding(),
        scope(),
        action_a.risk(),
        PolicyMode::Autonomous,
        approvals(true, 3),
        [4; 32],
        [5; 32],
        [6; 32],
        None,
    );
    let grant = execution
        .issue::<Write>(
            actor,
            scope(),
            None,
            action_a.authorization_binding(),
            admission,
        )
        .unwrap();

    assert!(matches!(
        action_b.authorize(grant),
        Err(PolicyError::ActionBindingMismatch)
    ));
}

#[test]
fn final_evidence_preserves_policy_and_resource_lineage() {
    let evaluator = PolicyEvaluatorDomain::new(PrincipalId::new(), descriptor(11));
    let execution = PolicyExecutionDomain::new(PrincipalId::new(), evaluator.verifier());
    let observation = AuthorityDomain::new(PrincipalId::new());
    let resolution = ResolutionAuthorityDomain::new(PrincipalId::new());
    let resources = ResourceResolverDomain::new(PrincipalId::new());
    let runtime = runtime(&execution, &observation, &resolution, &resources);
    let actor = PrincipalId::new();
    let observer = PrincipalId::new();
    let resolver = PrincipalId::new();
    let resource_identity = identity(12);

    let action = runtime
        .admit_resolved::<Write, _>(
            actor,
            "edit-source",
            resources.resolve(0_u64, resource_identity.clone(), None),
            b"operation",
        )
        .unwrap()
        .assess(ActionRisk::StateModifying);

    let admission = evaluator.admit(
        action.authorization_binding(),
        scope(),
        action.risk(),
        PolicyMode::Supervised,
        approvals(true, 13),
        [14; 32],
        [15; 32],
        [16; 32],
        None,
    );
    let expected_admission_digest = admission.receipt().digest();
    let grant = execution
        .issue::<Write>(
            actor,
            scope(),
            None,
            action.authorization_binding(),
            admission,
        )
        .unwrap();
    let expected_policy_binding = grant.policy_binding();

    let action = action
        .authorize(grant)
        .unwrap()
        .execute_with(|handle| -> Result<[u8; 32], &'static str> {
            *handle += 1;
            Ok([17; 32])
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
            Observation::new(ObservedOutcome::Success, [18; 32]),
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

    assert_eq!(
        receipt.policy_evidence().receipt().digest(),
        expected_admission_digest
    );
    assert_eq!(
        receipt.policy_evidence().policy_binding(),
        expected_policy_binding
    );
    assert_eq!(
        receipt.resource_receipt().resource_identity(),
        &resource_identity
    );
    assert_eq!(receipt.policy_evidence().policy_domain(), evaluator.domain_id());
    assert_eq!(receipt.policy_evidence().execution_domain(), execution.domain_id());
}
