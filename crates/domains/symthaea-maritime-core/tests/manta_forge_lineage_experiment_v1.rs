use std::collections::BTreeSet;
use symthaea_maritime_core::{
    assess_regenerative_role_assumption_changes, calibrate_regenerative_viability_role,
    calibrate_regenerative_viability_roles, derive_regenerative_role_assumption_scopes,
    evaluate_regenerative_lineage_viability, evaluate_supported_closure, DependencyGovernance,
    RegenerativeAssumptionChangeSetV1, RegenerativeCapability, RegenerativeClosureModel,
    RegenerativeDependency, RegenerativeDependencyKind, RegenerativeFlowKindV1,
    RegenerativeFlowSupportClaimV1, RegenerativeFlowSupportV1, RegenerativeGenomeRequirementV1,
    RegenerativeGenomeV1, RegenerativeHorizon, RegenerativeLineageRoleV1,
    RegenerativeLineageViabilityProfileV1, RegenerativeViabilityCalibrationClassV1,
    RegenerativeViabilityDynamicObservationV1, REGENERATIVE_FLOW_SUPPORT_SCHEMA_V1,
    REGENERATIVE_GENOME_SCHEMA_V1, REGENERATIVE_LINEAGE_VIABILITY_SCHEMA_V1,
};

const SYMTROPY_EXPERIMENT_HEAD: &str = "10e2bb188d7cb043b0ae74d11b8718db103268dd";

fn dependency(
    id: &str,
    kind: RegenerativeDependencyKind,
    governance: DependencyGovernance,
    production: u64,
    recycling: u64,
    stockpile: u64,
) -> RegenerativeDependency {
    RegenerativeDependency {
        dependency_id: id.into(),
        kind,
        governance,
        demand_units_per_period: 1,
        local_production_units_per_period: production,
        recycling_units_per_period: recycling,
        stockpile_units: stockpile,
        unit_mass_grams: None,
        evidence_binding: format!("dependency:{id}:lineage-experiment"),
    }
}

fn capability(id: &str, essential: bool, dependencies: &[&str]) -> RegenerativeCapability {
    RegenerativeCapability {
        capability_id: id.into(),
        essential,
        dependency_ids: dependencies.iter().map(|id| (*id).to_owned()).collect(),
        evidence_binding: format!("capability:{id}:lineage-experiment"),
    }
}

fn model() -> RegenerativeClosureModel {
    RegenerativeClosureModel {
        model_id: "manta-v2-lineage-experiment".into(),
        period_duration_ms: 1,
        dependencies: vec![
            dependency(
                "forge-tooling-v2",
                RegenerativeDependencyKind::Tooling,
                DependencyGovernance::Ordinary,
                0,
                0,
                6,
            ),
            dependency(
                "local-controller-v2",
                RegenerativeDependencyKind::Component,
                DependencyGovernance::Ordinary,
                1,
                0,
                0,
            ),
            dependency(
                "metrology-v2",
                RegenerativeDependencyKind::Metrology,
                DependencyGovernance::Ordinary,
                1,
                0,
                4,
            ),
            dependency(
                "reactor-service-v2",
                RegenerativeDependencyKind::ExternalService,
                DependencyGovernance::SafeguardedExternal,
                0,
                0,
                8,
            ),
            dependency(
                "structural-stock-v2",
                RegenerativeDependencyKind::Material,
                DependencyGovernance::Ordinary,
                0,
                1,
                18,
            ),
        ],
        capabilities: vec![
            capability(
                "operation-v2",
                true,
                &["reactor-service-v2", "structural-stock-v2"],
            ),
            capability(
                "construction-controller-v2",
                false,
                &["local-controller-v2"],
            ),
            capability(
                "construction-tooling-v2",
                false,
                &["forge-tooling-v2", "structural-stock-v2"],
            ),
            capability(
                "successor-qualification-v2",
                false,
                &["metrology-v2", "reactor-service-v2"],
            ),
        ],
        evidence_binding: "model:manta-v2-lineage-experiment".into(),
    }
}

fn support_claim(
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
    let mut claims = vec![
        support_claim(
            "local-controller-v2",
            RegenerativeFlowKindV1::Production,
            "metrology-v2",
        ),
        support_claim(
            "metrology-v2",
            RegenerativeFlowKindV1::Production,
            "reactor-service-v2",
        ),
        support_claim(
            "structural-stock-v2",
            RegenerativeFlowKindV1::Recycling,
            "forge-tooling-v2",
        ),
    ];
    claims.sort_by(|left, right| {
        (left.dependency_id.as_str(), left.flow_kind)
            .cmp(&(right.dependency_id.as_str(), right.flow_kind))
    });
    RegenerativeFlowSupportV1 {
        schema_version: REGENERATIVE_FLOW_SUPPORT_SCHEMA_V1,
        support_id: "support:manta-v2-lineage-experiment".into(),
        closure_model_id: "manta-v2-lineage-experiment".into(),
        closure_model_evidence_binding: "model:manta-v2-lineage-experiment".into(),
        claims,
        evidence_binding: "flow-support:manta-v2-lineage-experiment".into(),
    }
}

fn requirement(
    requirement_id: &str,
    capability_id: &str,
    dependency_id: &str,
) -> RegenerativeGenomeRequirementV1 {
    RegenerativeGenomeRequirementV1 {
        requirement_id: requirement_id.into(),
        capability_id: capability_id.into(),
        baseline_dependency_id: dependency_id.into(),
        design_binding: format!("design:{requirement_id}"),
        metrology_profile_binding: format!("metrology-profile:{requirement_id}"),
        requalification_profile_binding: format!("requalification:{requirement_id}"),
        disassembly_profile_binding: format!("disassembly:{requirement_id}"),
        recovery_profile_binding: format!("recovery:{requirement_id}"),
        qualified_substitution_bindings: Vec::new(),
    }
}

fn genome() -> RegenerativeGenomeV1 {
    let mut requirements = vec![
        requirement(
            "req-construction-controller-local-controller",
            "construction-controller-v2",
            "local-controller-v2",
        ),
        requirement(
            "req-construction-tooling-forge",
            "construction-tooling-v2",
            "forge-tooling-v2",
        ),
        requirement(
            "req-construction-tooling-structural",
            "construction-tooling-v2",
            "structural-stock-v2",
        ),
        requirement(
            "req-operation-reactor",
            "operation-v2",
            "reactor-service-v2",
        ),
        requirement(
            "req-operation-structural",
            "operation-v2",
            "structural-stock-v2",
        ),
        requirement(
            "req-qualification-metrology",
            "successor-qualification-v2",
            "metrology-v2",
        ),
        requirement(
            "req-qualification-reactor",
            "successor-qualification-v2",
            "reactor-service-v2",
        ),
    ];
    requirements.sort_by(|left, right| left.requirement_id.cmp(&right.requirement_id));
    RegenerativeGenomeV1 {
        schema_version: REGENERATIVE_GENOME_SCHEMA_V1,
        genome_id: "manta-genome-v2-lineage-experiment".into(),
        lineage_parent_binding: Some("genome:manta-v1-lineage-experiment".into()),
        closure_model_id: "manta-v2-lineage-experiment".into(),
        closure_model_evidence_binding: "model:manta-v2-lineage-experiment".into(),
        requirements,
        evidence_binding: "genome:manta-v2-lineage-experiment".into(),
    }
}

fn profile() -> RegenerativeLineageViabilityProfileV1 {
    RegenerativeLineageViabilityProfileV1 {
        schema_version: REGENERATIVE_LINEAGE_VIABILITY_SCHEMA_V1,
        profile_id: "profile:manta-v2-lineage-experiment".into(),
        closure_model_id: "manta-v2-lineage-experiment".into(),
        closure_model_evidence_binding: "model:manta-v2-lineage-experiment".into(),
        operational_capability_ids: vec!["operation-v2".into()],
        successor_construction_capability_ids: vec![
            "construction-controller-v2".into(),
            "construction-tooling-v2".into(),
        ],
        successor_qualification_capability_ids: vec!["successor-qualification-v2".into()],
        evidence_binding: "viability-profile:manta-v2-lineage-experiment".into(),
    }
}

fn observation(
    role_id: &str,
    static_horizon: RegenerativeHorizon,
    first_unavailable_tick: Option<u64>,
    static_assumptions_held: bool,
) -> RegenerativeViabilityDynamicObservationV1 {
    RegenerativeViabilityDynamicObservationV1 {
        role_id: role_id.into(),
        static_horizon,
        first_unavailable_tick,
        observed_through_tick: 9,
        static_assumptions_held,
        observation_binding: format!("symtropy-run:{SYMTROPY_EXPERIMENT_HEAD}:{role_id}"),
    }
}

#[test]
fn static_v2_lineage_horizons_are_support_qualified_not_failure_forecasts() {
    let model = model();
    let support = support();
    let report = evaluate_regenerative_lineage_viability(&profile(), &genome(), &model, &support)
        .unwrap();

    assert_eq!(
        report.operation.conservative_horizon,
        RegenerativeHorizon::FinitePeriods(6)
    );
    assert_eq!(
        report.operation.root_limiting_dependency_ids,
        vec!["forge-tooling-v2"]
    );
    assert_eq!(
        report.successor_construction.conservative_horizon,
        RegenerativeHorizon::FinitePeriods(6)
    );
    assert_eq!(
        report.successor_construction.root_limiting_dependency_ids,
        vec!["forge-tooling-v2"]
    );
    assert_eq!(
        report.successor_qualification.conservative_horizon,
        RegenerativeHorizon::FinitePeriods(8)
    );
    assert_eq!(
        report.successor_qualification.root_limiting_dependency_ids,
        vec!["reactor-service-v2"]
    );
    assert_eq!(
        report.regenerative_viability_horizon,
        RegenerativeHorizon::FinitePeriods(6)
    );
    assert_eq!(
        report.limiting_roles,
        vec![
            RegenerativeLineageRoleV1::Operation,
            RegenerativeLineageRoleV1::SuccessorConstruction,
        ]
    );
    assert!(report.fully_modeled_regenerative_viability);
}

#[test]
fn metrology_shock_scope_is_derived_from_the_exact_support_graph() {
    let scopes =
        derive_regenerative_role_assumption_scopes(&profile(), &genome(), &model(), &support())
            .unwrap();
    let changes = RegenerativeAssumptionChangeSetV1 {
        change_set_id: "change:metrology-production-shock".into(),
        changed_dependency_ids: vec!["metrology-v2".into()],
        changed_external_input_bindings: Vec::new(),
        evidence_binding: format!("symtropy-run:{SYMTROPY_EXPERIMENT_HEAD}:metrology-shock"),
    };
    let assessments = assess_regenerative_role_assumption_changes(&scopes, &changes).unwrap();

    let operation = assessments
        .iter()
        .find(|assessment| assessment.role == RegenerativeLineageRoleV1::Operation)
        .unwrap();
    let construction = assessments
        .iter()
        .find(|assessment| {
            assessment.role == RegenerativeLineageRoleV1::SuccessorConstruction
        })
        .unwrap();
    let qualification = assessments
        .iter()
        .find(|assessment| {
            assessment.role == RegenerativeLineageRoleV1::SuccessorQualification
        })
        .unwrap();

    assert!(operation.static_assumptions_held);
    assert!(operation.intersecting_changed_dependency_ids.is_empty());
    assert!(!construction.static_assumptions_held);
    assert_eq!(
        construction.intersecting_changed_dependency_ids,
        vec!["metrology-v2"]
    );
    assert!(!qualification.static_assumptions_held);
    assert_eq!(
        qualification.intersecting_changed_dependency_ids,
        vec!["metrology-v2"]
    );

    let operation_scope = scopes
        .iter()
        .find(|scope| scope.role == RegenerativeLineageRoleV1::Operation)
        .unwrap();
    assert_eq!(
        operation_scope.dependency_ids,
        vec![
            "forge-tooling-v2",
            "reactor-service-v2",
            "structural-stock-v2",
        ]
    );
}

#[test]
fn dynamic_v2_history_calibrates_without_manual_assumption_assertions() {
    let model = model();
    let support = support();
    let static_report =
        evaluate_regenerative_lineage_viability(&profile(), &genome(), &model, &support).unwrap();
    let scopes = derive_regenerative_role_assumption_scopes(&profile(), &genome(), &model, &support)
        .unwrap();
    let changes = RegenerativeAssumptionChangeSetV1 {
        change_set_id: "change:metrology-production-shock".into(),
        changed_dependency_ids: vec!["metrology-v2".into()],
        changed_external_input_bindings: Vec::new(),
        evidence_binding: format!("symtropy-run:{SYMTROPY_EXPERIMENT_HEAD}:metrology-shock"),
    };
    let assumptions = assess_regenerative_role_assumption_changes(&scopes, &changes).unwrap();
    let held = |role| {
        assumptions
            .iter()
            .find(|assessment| assessment.role == role)
            .unwrap()
            .static_assumptions_held
    };

    let observations = vec![
        observation(
            "operation",
            static_report.operation.conservative_horizon,
            Some(9),
            held(RegenerativeLineageRoleV1::Operation),
        ),
        observation(
            "successor_construction",
            static_report.successor_construction.conservative_horizon,
            Some(6),
            held(RegenerativeLineageRoleV1::SuccessorConstruction),
        ),
        observation(
            "successor_qualification",
            static_report.successor_qualification.conservative_horizon,
            Some(5),
            held(RegenerativeLineageRoleV1::SuccessorQualification),
        ),
    ];
    let calibration = calibrate_regenerative_viability_roles(&observations).unwrap();

    assert!(!calibration.any_static_model_conflict);
    assert!(calibration.conflict_role_ids.is_empty());
    assert_eq!(calibration.refinement_role_ids, vec!["operation"]);
    assert_eq!(
        calibration.roles[0].classification,
        RegenerativeViabilityCalibrationClassV1::SurvivedBeyondConservativeBound
    );
    assert_eq!(
        calibration.roles[1].classification,
        RegenerativeViabilityCalibrationClassV1::AssumptionsChanged
    );
    assert_eq!(
        calibration.roles[2].classification,
        RegenerativeViabilityCalibrationClassV1::AssumptionsChanged
    );

    let supported = evaluate_supported_closure(&model, &support).unwrap();
    let forge = supported
        .dependencies
        .iter()
        .find(|dependency| dependency.dependency_id == "forge-tooling-v2")
        .unwrap();
    let reactor = supported
        .dependencies
        .iter()
        .find(|dependency| dependency.dependency_id == "reactor-service-v2")
        .unwrap();

    let forge_calibration = calibrate_regenerative_viability_role(&observation(
        "construction_tooling_support",
        forge.conservative_horizon,
        Some(7),
        true,
    ))
    .unwrap();
    let reactor_calibration = calibrate_regenerative_viability_role(&observation(
        "safeguarded_reactor_service",
        reactor.conservative_horizon,
        Some(9),
        true,
    ))
    .unwrap();
    assert_eq!(
        forge_calibration.classification,
        RegenerativeViabilityCalibrationClassV1::CorroboratedFiniteBoundary
    );
    assert_eq!(
        reactor_calibration.classification,
        RegenerativeViabilityCalibrationClassV1::CorroboratedFiniteBoundary
    );
}
