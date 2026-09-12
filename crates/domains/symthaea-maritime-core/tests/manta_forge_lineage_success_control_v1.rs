use std::collections::BTreeSet;
use symthaea_maritime_core::{
    assess_regenerative_role_assumption_changes, calibrate_regenerative_viability_roles,
    derive_regenerative_role_assumption_scopes, evaluate_regenerative_lineage_viability,
    qualify_regenerative_epoch_handoff, DependencyGovernance, RegenerativeAssumptionChangeSetV1,
    RegenerativeCapability, RegenerativeClosureModel, RegenerativeDependency,
    RegenerativeDependencyKind, RegenerativeEpochHandoffEvidenceV1,
    RegenerativeEpochTransferQualificationV1, RegenerativeFlowKindV1,
    RegenerativeFlowSupportClaimV1, RegenerativeFlowSupportV1, RegenerativeGenomeRequirementV1,
    RegenerativeGenomeV1, RegenerativeHorizon, RegenerativeLineageRoleV1,
    RegenerativeLineageViabilityProfileV1, RegenerativeViabilityCalibrationClassV1,
    RegenerativeViabilityDynamicObservationV1, REGENERATIVE_EPOCH_HANDOFF_SCHEMA_V1,
    REGENERATIVE_FLOW_SUPPORT_SCHEMA_V1, REGENERATIVE_GENOME_SCHEMA_V1,
    REGENERATIVE_LINEAGE_VIABILITY_SCHEMA_V1,
};

const SYMTROPY_SUCCESS_CONTROL_HEAD: &str = "3444c33977e7822385ebcbbc833037c5d35b7259";

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
        evidence_binding: format!("dependency:{id}:success-control"),
    }
}

fn capability(id: &str, essential: bool, dependencies: &[&str]) -> RegenerativeCapability {
    RegenerativeCapability {
        capability_id: id.into(),
        essential,
        dependency_ids: dependencies.iter().map(|id| (*id).to_owned()).collect(),
        evidence_binding: format!("capability:{id}:success-control"),
    }
}

fn model(generation: &str, forge: u64, metrology: u64, reactor: u64, structure: u64) -> RegenerativeClosureModel {
    RegenerativeClosureModel {
        model_id: format!("manta-{generation}-success-control"),
        period_duration_ms: 1,
        dependencies: vec![
            dependency(
                &format!("forge-tooling-{generation}"),
                RegenerativeDependencyKind::Tooling,
                DependencyGovernance::Ordinary,
                0,
                0,
                forge,
            ),
            dependency(
                &format!("local-controller-{generation}"),
                RegenerativeDependencyKind::Component,
                DependencyGovernance::Ordinary,
                1,
                0,
                0,
            ),
            dependency(
                &format!("metrology-{generation}"),
                RegenerativeDependencyKind::Metrology,
                DependencyGovernance::Ordinary,
                1,
                0,
                metrology,
            ),
            dependency(
                &format!("reactor-service-{generation}"),
                RegenerativeDependencyKind::ExternalService,
                DependencyGovernance::SafeguardedExternal,
                0,
                0,
                reactor,
            ),
            dependency(
                &format!("structural-stock-{generation}"),
                RegenerativeDependencyKind::Material,
                DependencyGovernance::Ordinary,
                0,
                1,
                structure,
            ),
        ],
        capabilities: vec![
            capability(
                &format!("operation-{generation}"),
                true,
                &[
                    &format!("reactor-service-{generation}"),
                    &format!("structural-stock-{generation}"),
                ],
            ),
            capability(
                &format!("construction-controller-{generation}"),
                false,
                &[&format!("local-controller-{generation}")],
            ),
            capability(
                &format!("construction-tooling-{generation}"),
                false,
                &[
                    &format!("forge-tooling-{generation}"),
                    &format!("structural-stock-{generation}"),
                ],
            ),
            capability(
                &format!("successor-qualification-{generation}"),
                false,
                &[
                    &format!("metrology-{generation}"),
                    &format!("reactor-service-{generation}"),
                ],
            ),
        ],
        evidence_binding: format!("model:manta-{generation}-success-control"),
    }
}

fn support_claim(
    dependency_id: String,
    flow_kind: RegenerativeFlowKindV1,
    prerequisite_dependency_id: String,
) -> RegenerativeFlowSupportClaimV1 {
    RegenerativeFlowSupportClaimV1 {
        capability_binding: format!("flow-capability:{dependency_id}:{flow_kind:?}"),
        metrology_binding: format!("flow-metrology:{dependency_id}:{flow_kind:?}"),
        qualification_binding: format!("flow-qualification:{dependency_id}:{flow_kind:?}"),
        dependency_id,
        flow_kind,
        prerequisite_dependency_ids: vec![prerequisite_dependency_id],
        external_input_binding: None,
        bootstrap_binding: None,
    }
}

fn support(generation: &str) -> RegenerativeFlowSupportV1 {
    let mut claims = vec![
        support_claim(
            format!("local-controller-{generation}"),
            RegenerativeFlowKindV1::Production,
            format!("metrology-{generation}"),
        ),
        support_claim(
            format!("metrology-{generation}"),
            RegenerativeFlowKindV1::Production,
            format!("reactor-service-{generation}"),
        ),
        support_claim(
            format!("structural-stock-{generation}"),
            RegenerativeFlowKindV1::Recycling,
            format!("forge-tooling-{generation}"),
        ),
    ];
    claims.sort_by(|left, right| {
        (left.dependency_id.as_str(), left.flow_kind)
            .cmp(&(right.dependency_id.as_str(), right.flow_kind))
    });
    RegenerativeFlowSupportV1 {
        schema_version: REGENERATIVE_FLOW_SUPPORT_SCHEMA_V1,
        support_id: format!("support:manta-{generation}-success-control"),
        closure_model_id: format!("manta-{generation}-success-control"),
        closure_model_evidence_binding: format!("model:manta-{generation}-success-control"),
        claims,
        evidence_binding: format!("flow-support:manta-{generation}-success-control"),
    }
}

fn requirement(
    requirement_id: String,
    capability_id: String,
    dependency_id: String,
) -> RegenerativeGenomeRequirementV1 {
    RegenerativeGenomeRequirementV1 {
        design_binding: format!("design:{requirement_id}"),
        metrology_profile_binding: format!("metrology-profile:{requirement_id}"),
        requalification_profile_binding: format!("requalification:{requirement_id}"),
        disassembly_profile_binding: format!("disassembly:{requirement_id}"),
        recovery_profile_binding: format!("recovery:{requirement_id}"),
        requirement_id,
        capability_id,
        baseline_dependency_id: dependency_id,
        qualified_substitution_bindings: Vec::new(),
    }
}

fn genome(generation: &str, parent_binding: Option<&str>) -> RegenerativeGenomeV1 {
    let mut requirements = vec![
        requirement(
            format!("req-{generation}-construction-controller"),
            format!("construction-controller-{generation}"),
            format!("local-controller-{generation}"),
        ),
        requirement(
            format!("req-{generation}-construction-forge"),
            format!("construction-tooling-{generation}"),
            format!("forge-tooling-{generation}"),
        ),
        requirement(
            format!("req-{generation}-construction-structure"),
            format!("construction-tooling-{generation}"),
            format!("structural-stock-{generation}"),
        ),
        requirement(
            format!("req-{generation}-operation-reactor"),
            format!("operation-{generation}"),
            format!("reactor-service-{generation}"),
        ),
        requirement(
            format!("req-{generation}-operation-structure"),
            format!("operation-{generation}"),
            format!("structural-stock-{generation}"),
        ),
        requirement(
            format!("req-{generation}-qualification-metrology"),
            format!("successor-qualification-{generation}"),
            format!("metrology-{generation}"),
        ),
        requirement(
            format!("req-{generation}-qualification-reactor"),
            format!("successor-qualification-{generation}"),
            format!("reactor-service-{generation}"),
        ),
    ];
    requirements.sort_by(|left, right| left.requirement_id.cmp(&right.requirement_id));
    RegenerativeGenomeV1 {
        schema_version: REGENERATIVE_GENOME_SCHEMA_V1,
        genome_id: format!("manta-genome-{generation}-success-control"),
        lineage_parent_binding: parent_binding.map(str::to_owned),
        closure_model_id: format!("manta-{generation}-success-control"),
        closure_model_evidence_binding: format!("model:manta-{generation}-success-control"),
        requirements,
        evidence_binding: format!("genome:manta-{generation}-success-control"),
    }
}

fn profile(generation: &str) -> RegenerativeLineageViabilityProfileV1 {
    RegenerativeLineageViabilityProfileV1 {
        schema_version: REGENERATIVE_LINEAGE_VIABILITY_SCHEMA_V1,
        profile_id: format!("profile:manta-{generation}-success-control"),
        closure_model_id: format!("manta-{generation}-success-control"),
        closure_model_evidence_binding: format!("model:manta-{generation}-success-control"),
        operational_capability_ids: vec![format!("operation-{generation}")],
        successor_construction_capability_ids: vec![
            format!("construction-controller-{generation}"),
            format!("construction-tooling-{generation}"),
        ],
        successor_qualification_capability_ids: vec![
            format!("successor-qualification-{generation}"),
        ],
        evidence_binding: format!("viability-profile:manta-{generation}-success-control"),
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
        observed_through_tick: 4,
        static_assumptions_held,
        observation_binding: format!(
            "symtropy-run:{SYMTROPY_SUCCESS_CONTROL_HEAD}:{role_id}"
        ),
    }
}

#[test]
fn successful_control_has_a_qualified_v2_to_v3_reproduction_window() {
    let v2_model = model("v2", 8, 1, 8, 18);
    let v2_support = support("v2");
    let v2_genome = genome("v2", Some("genome:manta-v1-success-control"));
    let v2_profile = profile("v2");
    let v2_static = evaluate_regenerative_lineage_viability(
        &v2_profile,
        &v2_genome,
        &v2_model,
        &v2_support,
    )
    .unwrap();

    assert_eq!(v2_static.operation.conservative_horizon, RegenerativeHorizon::FinitePeriods(8));
    assert_eq!(
        v2_static.successor_construction.conservative_horizon,
        RegenerativeHorizon::FinitePeriods(8)
    );
    assert_eq!(
        v2_static.successor_qualification.conservative_horizon,
        RegenerativeHorizon::FinitePeriods(8)
    );
    assert_eq!(
        v2_static.regenerative_viability_horizon,
        RegenerativeHorizon::FinitePeriods(8)
    );

    let scopes = derive_regenerative_role_assumption_scopes(
        &v2_profile,
        &v2_genome,
        &v2_model,
        &v2_support,
    )
    .unwrap();
    let changes = RegenerativeAssumptionChangeSetV1 {
        change_set_id: "change:success-control-metrology-shock".into(),
        changed_dependency_ids: vec!["metrology-v2".into()],
        changed_external_input_bindings: Vec::new(),
        evidence_binding: format!(
            "symtropy-run:{SYMTROPY_SUCCESS_CONTROL_HEAD}:metrology-shock"
        ),
    };
    let assumptions = assess_regenerative_role_assumption_changes(&scopes, &changes).unwrap();
    let held = |role| {
        assumptions
            .iter()
            .find(|assessment| assessment.role == role)
            .unwrap()
            .static_assumptions_held
    };

    assert!(held(RegenerativeLineageRoleV1::Operation));
    assert!(!held(RegenerativeLineageRoleV1::SuccessorConstruction));
    assert!(!held(RegenerativeLineageRoleV1::SuccessorQualification));

    let calibration = calibrate_regenerative_viability_roles(&[
        observation(
            "operation",
            v2_static.operation.conservative_horizon,
            None,
            held(RegenerativeLineageRoleV1::Operation),
        ),
        observation(
            "successor_construction",
            v2_static.successor_construction.conservative_horizon,
            Some(3),
            held(RegenerativeLineageRoleV1::SuccessorConstruction),
        ),
        observation(
            "successor_qualification",
            v2_static.successor_qualification.conservative_horizon,
            Some(2),
            held(RegenerativeLineageRoleV1::SuccessorQualification),
        ),
    ])
    .unwrap();

    assert!(!calibration.any_static_model_conflict);
    assert!(calibration.conflict_role_ids.is_empty());
    assert!(calibration.refinement_role_ids.is_empty());
    assert_eq!(
        calibration.roles[0].classification,
        RegenerativeViabilityCalibrationClassV1::InconclusiveBeforeFiniteBoundary
    );
    assert_eq!(
        calibration.roles[1].classification,
        RegenerativeViabilityCalibrationClassV1::AssumptionsChanged
    );
    assert_eq!(
        calibration.roles[2].classification,
        RegenerativeViabilityCalibrationClassV1::AssumptionsChanged
    );

    let v3_model = model("v3", 4, 0, 4, 18);
    let v3_support = support("v3");
    let v3_genome = genome("v3", Some(v2_genome.evidence_binding.as_str()));
    let v3_profile = profile("v3");

    let handoff = RegenerativeEpochHandoffEvidenceV1 {
        schema_version: REGENERATIVE_EPOCH_HANDOFF_SCHEMA_V1,
        handoff_id: "handoff-manta-v2-v3-success-control".into(),
        source_epoch_id: "manta-v2-control".into(),
        source_epoch_evidence_binding: "epoch:manta-v2-success-control".into(),
        successor_epoch_id: "manta-v3-control".into(),
        successor_epoch_evidence_binding: "epoch:manta-v3-success-control".into(),
        source_genome_id: v2_genome.genome_id.clone(),
        source_genome_evidence_binding: v2_genome.evidence_binding.clone(),
        successor_genome_id: v3_genome.genome_id.clone(),
        successor_genome_evidence_binding: v3_genome.evidence_binding.clone(),
        dynamic_handoff_receipt_binding: format!(
            "symtropy-handoff:{SYMTROPY_SUCCESS_CONTROL_HEAD}:v2-v3"
        ),
        transfer_qualifications: vec![
            RegenerativeEpochTransferQualificationV1 {
                source_dependency_id: "forge-tooling-v2".into(),
                successor_dependency_id: "forge-tooling-v3".into(),
                transfer_qualification_binding: "qualification:forge-v2-v3-control".into(),
                safeguarded_continuity_binding: None,
            },
            RegenerativeEpochTransferQualificationV1 {
                source_dependency_id: "reactor-service-v2".into(),
                successor_dependency_id: "reactor-service-v3".into(),
                transfer_qualification_binding: "qualification:reactor-v2-v3-control".into(),
                safeguarded_continuity_binding: Some(
                    "safeguarded:reactor-v2-v3-control".into(),
                ),
            },
            RegenerativeEpochTransferQualificationV1 {
                source_dependency_id: "structural-stock-v2".into(),
                successor_dependency_id: "structural-stock-v3".into(),
                transfer_qualification_binding: "qualification:structure-v2-v3-control".into(),
                safeguarded_continuity_binding: None,
            },
        ],
        external_admission_qualifications: Vec::new(),
        evidence_binding: "handoff-qualification:manta-v2-v3-success-control".into(),
    };
    let qualified = qualify_regenerative_epoch_handoff(
        &handoff,
        &v2_genome,
        &v2_model,
        &v3_genome,
        &v3_model,
    )
    .unwrap();
    assert_eq!(qualified.qualified_transfer_count, 3);
    assert_eq!(qualified.cross_id_transfer_count, 3);
    assert_eq!(qualified.safeguarded_transfer_count, 1);
    assert_eq!(qualified.external_admission_count, 0);

    let v3_static = evaluate_regenerative_lineage_viability(
        &v3_profile,
        &v3_genome,
        &v3_model,
        &v3_support,
    )
    .unwrap();
    assert_eq!(v3_static.operation.conservative_horizon, RegenerativeHorizon::FinitePeriods(4));
    assert_eq!(
        v3_static.successor_construction.conservative_horizon,
        RegenerativeHorizon::FinitePeriods(4)
    );
    assert_eq!(
        v3_static.successor_qualification.conservative_horizon,
        RegenerativeHorizon::FinitePeriods(4)
    );
    assert_eq!(
        v3_static.regenerative_viability_horizon,
        RegenerativeHorizon::FinitePeriods(4)
    );
}
