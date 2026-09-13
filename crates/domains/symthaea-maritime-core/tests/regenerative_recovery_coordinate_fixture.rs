include!("manta_forge_support_topology_continuity_v1.rs");

const RECOVERY_FIXTURE: &str =
    include_str!("../fixtures/regenerative-recovery-coordinate-v1.txt");

fn recovery_scalar(key: &str) -> u64 {
    RECOVERY_FIXTURE
        .lines()
        .find_map(|line| line.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("missing recovery-coordinate fixture scalar {key}"))
        .parse()
        .unwrap()
}

fn recovery_text(key: &str) -> &str {
    RECOVERY_FIXTURE
        .lines()
        .find_map(|line| line.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("missing recovery-coordinate fixture text {key}"))
}

fn recovery_policy() -> RegenerativeRecoveryCoordinatePolicyV1 {
    RegenerativeRecoveryCoordinatePolicyV1 {
        schema_version: REGENERATIVE_RECOVERY_COORDINATE_SCHEMA_V1,
        policy_id: "recovery-policy:metrology-v4".into(),
        evidence_binding: "recovery-policy:metrology-v4:v1".into(),
        target_dependency_id: recovery_text("target_dependency").into(),
        flow_kind: RegenerativeFlowKindV1::Production,
        qualified_units_per_period: recovery_scalar("healthy_units_per_period"),
        external_recovery_reserve_id: recovery_text("external_recovery_reserve_id").into(),
        external_recovery_reserve_binding: recovery_text(
            "external_recovery_reserve_binding",
        )
        .into(),
        external_recovery_reserve_units: recovery_scalar("external_recovery_reserve_units"),
        reserve_units_per_recovery: recovery_scalar("reserve_units_per_recovery"),
        recovery_qualification_binding: "recovery-qualification:metrology-v4:v1".into(),
    }
}

#[test]
fn recovery_qualifies_against_exact_safe_topology_successor_without_mutating_nominal_subject() {
    let successor_model = model(4, true);
    let successor_support = support(4, false);
    let successor_profile = profile(4);

    assert_eq!(successor_model.model_id, "closure-v4-topology");
    assert_eq!(successor_model.evidence_binding, "model:closure-v4-topology");
    assert_eq!(successor_support.support_id, "support:v4:direct");
    assert_eq!(successor_support.evidence_binding, "flow-support:v4:direct");
    assert_eq!(successor_profile.profile_id, "profile-v4-topology");
    assert!(!successor_model.dependencies.iter().any(|dependency| {
        dependency.dependency_id == recovery_text("external_recovery_reserve_id")
    }));

    let report = qualify_regenerative_recovery_coordinate(
        &recovery_policy(),
        &successor_profile,
        &successor_model,
        &successor_support,
    )
    .unwrap();
    assert_eq!(report.closure_model_id, "closure-v4-topology");
    assert_eq!(report.flow_support_id, "support:v4:direct");
    assert_eq!(report.profile_id, "profile-v4-topology");
    assert_eq!(
        report.recovery_qualification_binding,
        "recovery-qualification:metrology-v4:v1"
    );
    assert!(report.disturbance_recovery_coordinate_qualified);
    assert!(report.reserve_external_to_nominal_closure);
    assert!(report
        .role_support_dependency_ids
        .contains(&"metrology-v4".to_string()));
    assert!(!report
        .role_support_dependency_ids
        .contains(&recovery_text("external_recovery_reserve_id").to_string()));

    let disturbance = RegenerativeDisturbanceContextV1 {
        disturbance_id: "disturbance:metrology-production-loss-v4".into(),
        evidence_binding: "disturbance:metrology-production-loss-v4:v1".into(),
        target_dependency_id: recovery_text("target_dependency").into(),
        flow_kind: RegenerativeFlowKindV1::Production,
        degraded_units_per_period: recovery_scalar("degraded_units_per_period"),
    };
    let authorization = authorize_regenerative_disturbance_recovery(&disturbance, &report).unwrap();
    assert!(authorization.disturbance_conditioned_recovery_authorized);
    assert_eq!(
        authorization.qualified_restore_units_per_period,
        recovery_scalar("healthy_units_per_period")
    );
    assert_eq!(
        authorization.external_recovery_reserve_id,
        recovery_text("external_recovery_reserve_id")
    );
}

#[test]
fn external_recovery_reserve_cannot_alias_a_nominal_dependency() {
    let mut policy = recovery_policy();
    policy.external_recovery_reserve_id = "forge-tooling-v4".into();
    assert!(matches!(
        qualify_regenerative_recovery_coordinate(
            &policy,
            &profile(4),
            &model(4, true),
            &support(4, false),
        ),
        Err(
            RegenerativeRecoveryCoordinateError::ExternalRecoveryReserveCollidesWithModelDependency {
                ..
            }
        )
    ));
}

#[test]
fn recovery_rate_reserve_and_disturbance_subject_are_fail_closed() {
    let mut wrong_rate = recovery_policy();
    wrong_rate.qualified_units_per_period += 1;
    assert!(matches!(
        qualify_regenerative_recovery_coordinate(
            &wrong_rate,
            &profile(4),
            &model(4, true),
            &support(4, false),
        ),
        Err(RegenerativeRecoveryCoordinateError::TargetFlowRateMismatch { .. })
    ));

    let mut insufficient = recovery_policy();
    insufficient.external_recovery_reserve_units = 0;
    assert!(matches!(
        qualify_regenerative_recovery_coordinate(
            &insufficient,
            &profile(4),
            &model(4, true),
            &support(4, false),
        ),
        Err(RegenerativeRecoveryCoordinateError::ExternalRecoveryReserveInsufficient { .. })
    ));

    let report = qualify_regenerative_recovery_coordinate(
        &recovery_policy(),
        &profile(4),
        &model(4, true),
        &support(4, false),
    )
    .unwrap();
    let wrong_target = RegenerativeDisturbanceContextV1 {
        disturbance_id: "disturbance:wrong-target".into(),
        evidence_binding: "disturbance:wrong-target:v1".into(),
        target_dependency_id: "local-controller-v4".into(),
        flow_kind: RegenerativeFlowKindV1::Production,
        degraded_units_per_period: 0,
    };
    assert!(matches!(
        authorize_regenerative_disturbance_recovery(&wrong_target, &report),
        Err(RegenerativeRecoveryCoordinateError::DisturbanceRecoverySubjectMismatch)
    ));

    let no_degradation = RegenerativeDisturbanceContextV1 {
        disturbance_id: "disturbance:no-degradation".into(),
        evidence_binding: "disturbance:no-degradation:v1".into(),
        target_dependency_id: recovery_text("target_dependency").into(),
        flow_kind: RegenerativeFlowKindV1::Production,
        degraded_units_per_period: recovery_scalar("healthy_units_per_period"),
    };
    assert!(matches!(
        authorize_regenerative_disturbance_recovery(&no_degradation, &report),
        Err(RegenerativeRecoveryCoordinateError::DisturbanceDoesNotDegradeQualifiedFlow)
    ));
}
