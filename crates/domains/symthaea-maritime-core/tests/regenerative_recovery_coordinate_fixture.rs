use std::collections::BTreeSet;
use symthaea_maritime_core::*;

const FIXTURE: &str = include_str!("../fixtures/regenerative-recovery-coordinate-v1.txt");

fn scalar(key: &str) -> u64 {
    FIXTURE
        .lines()
        .find_map(|line| line.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("missing recovery-coordinate fixture scalar {key}"))
        .parse()
        .unwrap()
}

fn dependency(
    id: &str,
    kind: RegenerativeDependencyKind,
    governance: DependencyGovernance,
    demand: u64,
    production: u64,
    recycling: u64,
    stockpile: u64,
) -> RegenerativeDependency {
    RegenerativeDependency {
        dependency_id: id.into(),
        kind,
        governance,
        demand_units_per_period: demand,
        local_production_units_per_period: production,
        recycling_units_per_period: recycling,
        stockpile_units: stockpile,
        unit_mass_grams: None,
        evidence_binding: format!("dependency:{id}:v1"),
    }
}

fn capability(id: &str, essential: bool, dependencies: &[&str]) -> RegenerativeCapability {
    RegenerativeCapability {
        capability_id: id.into(),
        essential,
        dependency_ids: dependencies.iter().map(|value| (*value).into()).collect(),
        evidence_binding: format!("capability:{id}:v1"),
    }
}

fn model() -> RegenerativeClosureModel {
    RegenerativeClosureModel {
        model_id: "manta-v4-recovery-coordinate".into(),
        period_duration_ms: scalar("period_duration_ms"),
        dependencies: vec![
            dependency(
                "controller-support-v4",
                RegenerativeDependencyKind::Component,
                DependencyGovernance::Ordinary,
                1,
                0,
                0,
                0,
            ),
            dependency(
                "forge-tooling-v4",
                RegenerativeDependencyKind::Tooling,
                DependencyGovernance::Ordinary,
                1,
                0,
                0,
                2,
            ),
            dependency(
                "local-controller-v4",
                RegenerativeDependencyKind::Component,
                DependencyGovernance::Ordinary,
                1,
                1,
                0,
                0,
            ),
            dependency(
                "metrology-v4",
                RegenerativeDependencyKind::Metrology,
                DependencyGovernance::Ordinary,
                1,
                scalar("healthy_units_per_period"),
                0,
                0,
            ),
            dependency(
                "protected-service-v4",
                RegenerativeDependencyKind::ExternalService,
                DependencyGovernance::SafeguardedExternal,
                1,
                0,
                0,
                98,
            ),
            dependency(
                "repair-reserve-v4",
                RegenerativeDependencyKind::Component,
                DependencyGovernance::Ordinary,
                1,
                0,
                0,
                scalar("reserve_stockpile_units"),
            ),
            dependency(
                "structural-stock-v4",
                RegenerativeDependencyKind::Material,
                DependencyGovernance::Ordinary,
                1,
                0,
                1,
                18,
            ),
        ],
        capabilities: vec![
            capability(
                "lineage-reproduction-v4",
                true,
                &[
                    "forge-tooling-v4",
                    "local-controller-v4",
                    "metrology-v4",
                    "protected-service-v4",
                    "structural-stock-v4",
                ],
            ),
            capability(
                "operation-v4",
                true,
                &["protected-service-v4", "structural-stock-v4"],
            ),
        ],
        evidence_binding: "model:manta-v4-recovery-coordinate:v1".into(),
    }
}

fn claim(
    dependency_id: &str,
    flow_kind: RegenerativeFlowKindV1,
    prerequisite: &str,
) -> RegenerativeFlowSupportClaimV1 {
    RegenerativeFlowSupportClaimV1 {
        dependency_id: dependency_id.into(),
        flow_kind,
        capability_binding: format!("flow-capability:{dependency_id}:{flow_kind:?}"),
        prerequisite_dependency_ids: vec![prerequisite.into()],
        external_input_binding: None,
        metrology_binding: format!("flow-metrology:{dependency_id}:{flow_kind:?}"),
        qualification_binding: format!("flow-qualification:{dependency_id}:{flow_kind:?}"),
        bootstrap_binding: None,
    }
}

fn support() -> RegenerativeFlowSupportV1 {
    RegenerativeFlowSupportV1 {
        schema_version: REGENERATIVE_FLOW_SUPPORT_SCHEMA_V1,
        support_id: "support:manta-v4-recovery-coordinate".into(),
        closure_model_id: "manta-v4-recovery-coordinate".into(),
        closure_model_evidence_binding: "model:manta-v4-recovery-coordinate:v1".into(),
        claims: vec![
            claim(
                "local-controller-v4",
                RegenerativeFlowKindV1::Production,
                "protected-service-v4",
            ),
            claim(
                "metrology-v4",
                RegenerativeFlowKindV1::Production,
                "protected-service-v4",
            ),
            claim(
                "structural-stock-v4",
                RegenerativeFlowKindV1::Recycling,
                "forge-tooling-v4",
            ),
        ],
        evidence_binding: "flow-support:manta-v4-recovery-coordinate:v1".into(),
    }
}

fn profile() -> RegenerativeLineageViabilityProfileV1 {
    RegenerativeLineageViabilityProfileV1 {
        schema_version: REGENERATIVE_LINEAGE_VIABILITY_SCHEMA_V1,
        profile_id: "profile:manta-v4-recovery-coordinate".into(),
        closure_model_id: "manta-v4-recovery-coordinate".into(),
        closure_model_evidence_binding: "model:manta-v4-recovery-coordinate:v1".into(),
        operational_capability_ids: vec!["operation-v4".into()],
        successor_construction_capability_ids: vec!["lineage-reproduction-v4".into()],
        successor_qualification_capability_ids: vec!["lineage-reproduction-v4".into()],
        evidence_binding: "profile:manta-v4-recovery-coordinate:v1".into(),
    }
}

fn recovery_policy() -> RegenerativeRecoveryCoordinatePolicyV1 {
    RegenerativeRecoveryCoordinatePolicyV1 {
        schema_version: REGENERATIVE_RECOVERY_COORDINATE_SCHEMA_V1,
        policy_id: "recovery-policy:metrology-v4".into(),
        evidence_binding: "recovery-policy:metrology-v4:v1".into(),
        target_dependency_id: "metrology-v4".into(),
        flow_kind: RegenerativeFlowKindV1::Production,
        qualified_units_per_period: scalar("healthy_units_per_period"),
        reserve_dependency_id: "repair-reserve-v4".into(),
        reserve_units_per_recovery: scalar("reserve_units_per_recovery"),
        recovery_qualification_binding: "recovery-qualification:metrology-v4:v1".into(),
    }
}

#[test]
fn recovery_is_qualified_as_a_coordinate_outside_nominal_role_support_closure() {
    let report = qualify_regenerative_recovery_coordinate(
        &recovery_policy(),
        &profile(),
        &model(),
        &support(),
    )
    .unwrap();
    assert!(report.disturbance_recovery_coordinate_qualified);
    assert!(report.reserve_outside_nominal_role_support_closure);
    assert!(report
        .role_support_dependency_ids
        .contains(&"metrology-v4".to_string()));
    assert!(!report
        .role_support_dependency_ids
        .contains(&"repair-reserve-v4".to_string()));

    let disturbance = RegenerativeDisturbanceContextV1 {
        disturbance_id: "disturbance:metrology-production-loss-v4".into(),
        evidence_binding: "disturbance:metrology-production-loss-v4:v1".into(),
        target_dependency_id: "metrology-v4".into(),
        flow_kind: RegenerativeFlowKindV1::Production,
        degraded_units_per_period: scalar("degraded_units_per_period"),
    };
    let authorization = authorize_regenerative_disturbance_recovery(&disturbance, &report).unwrap();
    assert!(authorization.disturbance_conditioned_recovery_authorized);
    assert_eq!(
        authorization.qualified_restore_units_per_period,
        scalar("healthy_units_per_period")
    );
}

#[test]
fn recovery_reserve_inside_nominal_reproduction_closure_is_rejected_in_v1() {
    let mut model = model();
    let reproduction = model
        .capabilities
        .iter_mut()
        .find(|capability| capability.capability_id == "lineage-reproduction-v4")
        .unwrap();
    reproduction
        .dependency_ids
        .insert("repair-reserve-v4".into());

    assert!(matches!(
        qualify_regenerative_recovery_coordinate(
            &recovery_policy(),
            &profile(),
            &model,
            &support(),
        ),
        Err(RegenerativeRecoveryCoordinateError::RecoveryReserveInsideRoleSupportClosure { .. })
    ));
}

#[test]
fn recovery_rate_and_disturbance_subject_are_fail_closed() {
    let mut wrong_rate = recovery_policy();
    wrong_rate.qualified_units_per_period += 1;
    assert!(matches!(
        qualify_regenerative_recovery_coordinate(
            &wrong_rate,
            &profile(),
            &model(),
            &support(),
        ),
        Err(RegenerativeRecoveryCoordinateError::TargetFlowRateMismatch { .. })
    ));

    let report = qualify_regenerative_recovery_coordinate(
        &recovery_policy(),
        &profile(),
        &model(),
        &support(),
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
        target_dependency_id: "metrology-v4".into(),
        flow_kind: RegenerativeFlowKindV1::Production,
        degraded_units_per_period: scalar("healthy_units_per_period"),
    };
    assert!(matches!(
        authorize_regenerative_disturbance_recovery(&no_degradation, &report),
        Err(RegenerativeRecoveryCoordinateError::DisturbanceDoesNotDegradeQualifiedFlow)
    ));
}
