use std::collections::BTreeSet;
use symthaea_maritime_core::*;

const FIXTURE: &str =
    include_str!("../fixtures/manta-forge-intergenerational-support-basis-v1.txt");

fn scalar(key: &str) -> u64 {
    FIXTURE
        .lines()
        .find_map(|line| line.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("missing fixture scalar {key}"))
        .parse()
        .unwrap()
}

fn bool_scalar(key: &str) -> bool {
    FIXTURE
        .lines()
        .find_map(|line| line.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("missing fixture bool {key}"))
        .parse()
        .unwrap()
}

fn dependency(
    id: &str,
    kind: RegenerativeDependencyKind,
    governance: DependencyGovernance,
    demand: u64,
    stockpile: u64,
) -> RegenerativeDependency {
    RegenerativeDependency {
        dependency_id: id.into(),
        kind,
        governance,
        demand_units_per_period: demand,
        local_production_units_per_period: 0,
        recycling_units_per_period: 0,
        stockpile_units: stockpile,
        unit_mass_grams: None,
        evidence_binding: format!("dependency:{id}:evidence"),
    }
}

fn model(
    id: &str,
    period_ms: u64,
    tooling_id: &str,
    tooling_demand: u64,
    tooling_stockpile: u64,
    reactor_id: &str,
    reactor_stockpile: u64,
) -> RegenerativeClosureModel {
    RegenerativeClosureModel {
        model_id: id.into(),
        period_duration_ms: period_ms,
        dependencies: vec![
            dependency(
                tooling_id,
                RegenerativeDependencyKind::Tooling,
                DependencyGovernance::Ordinary,
                tooling_demand,
                tooling_stockpile,
            ),
            dependency(
                reactor_id,
                RegenerativeDependencyKind::ExternalService,
                DependencyGovernance::SafeguardedExternal,
                1,
                reactor_stockpile,
            ),
        ],
        capabilities: vec![RegenerativeCapability {
            capability_id: format!("lineage-reproduction:{id}"),
            essential: true,
            dependency_ids: BTreeSet::from([tooling_id.into(), reactor_id.into()]),
            evidence_binding: format!("capability:{id}:evidence"),
        }],
        evidence_binding: format!("model:{id}:evidence"),
    }
}

fn handoff() -> (
    RegenerativeEpochHandoffEvidenceV1,
    RegenerativeEpochHandoffQualificationReportV1,
) {
    let evidence = RegenerativeEpochHandoffEvidenceV1 {
        schema_version: 1,
        handoff_id: "handoff-basis-v3-v4".into(),
        source_epoch_id: "epoch-v3".into(),
        source_epoch_evidence_binding: "epoch:v3:basis-evidence".into(),
        successor_epoch_id: "epoch-v4".into(),
        successor_epoch_evidence_binding: "epoch:v4:basis-evidence".into(),
        source_genome_id: "genome-v3".into(),
        source_genome_evidence_binding: "genome:v3:basis-evidence".into(),
        successor_genome_id: "genome-v4".into(),
        successor_genome_evidence_binding: "genome:v4:basis-evidence".into(),
        dynamic_handoff_receipt_binding:
            "symtropy:pr-821:intergenerational-support-basis".into(),
        transfer_qualifications: vec![
            RegenerativeEpochTransferQualificationV1 {
                source_dependency_id: "forge-tooling-v3".into(),
                successor_dependency_id: "forge-tooling-v4".into(),
                transfer_qualification_binding: "transfer:forge-tooling:v3-v4".into(),
                safeguarded_continuity_binding: None,
            },
            RegenerativeEpochTransferQualificationV1 {
                source_dependency_id: "reactor-service-v3".into(),
                successor_dependency_id: "reactor-service-v4".into(),
                transfer_qualification_binding: "transfer:reactor-service:v3-v4".into(),
                safeguarded_continuity_binding: Some(
                    "safeguard:reactor-service:v3-v4".into(),
                ),
            },
        ],
        external_admission_qualifications: Vec::new(),
        evidence_binding: "symthaea:intergenerational-support-basis:v1".into(),
    };
    let report = RegenerativeEpochHandoffQualificationReportV1 {
        handoff_id: evidence.handoff_id.clone(),
        source_genome_id: evidence.source_genome_id.clone(),
        successor_genome_id: evidence.successor_genome_id.clone(),
        dynamic_handoff_receipt_binding: evidence.dynamic_handoff_receipt_binding.clone(),
        qualified_transfer_count: 2,
        cross_id_transfer_count: 2,
        safeguarded_transfer_count: 1,
        external_admission_count: 0,
        safeguarded_external_admission_count: 0,
    };
    (evidence, report)
}

fn basis_policy() -> RegenerativeIntergenerationalSupportBasisPolicyV1 {
    RegenerativeIntergenerationalSupportBasisPolicyV1 {
        policy_id: "manta-forge-bootstrap-support-basis-v1".into(),
        evidence_binding: "basis-policy:manta-forge:v1".into(),
        requirements: vec![
            RegenerativeSupportBasisRequirementV1 {
                source_dependency_id: "forge-tooling-v3".into(),
                successor_dependency_id: "forge-tooling-v4".into(),
                quantity_basis_equivalence_binding: "quantity-basis:forge-tooling:v3-v4".into(),
            },
            RegenerativeSupportBasisRequirementV1 {
                source_dependency_id: "reactor-service-v3".into(),
                successor_dependency_id: "reactor-service-v4".into(),
                quantity_basis_equivalence_binding:
                    "quantity-basis:reactor-service:v3-v4".into(),
            },
        ],
    }
}

fn role(
    role: RegenerativeLineageRoleV1,
    horizon: RegenerativeHorizon,
) -> RegenerativeLineageRoleReportV1 {
    RegenerativeLineageRoleReportV1 {
        role,
        conservative_horizon: horizon,
        limiting_requirement_ids: vec![format!("requirement:{role:?}")],
        root_limiting_dependency_ids: vec![format!("dependency:{role:?}")],
        fully_modeled_support: true,
        externally_conditioned_requirement_ids: Vec::new(),
    }
}

fn lineage_report() -> RegenerativeLineageViabilityReportV1 {
    let h4 = RegenerativeHorizon::FinitePeriods(scalar("source_horizon_periods"));
    RegenerativeLineageViabilityReportV1 {
        profile_id: "profile:manta-v3-basis".into(),
        genome_id: "genome-v3".into(),
        closure_model_id: "closure-v3-basis".into(),
        flow_support_id: "flow-support:manta-v3-basis".into(),
        operation: role(
            RegenerativeLineageRoleV1::Operation,
            RegenerativeHorizon::FinitePeriods(100),
        ),
        successor_construction: role(RegenerativeLineageRoleV1::SuccessorConstruction, h4),
        successor_qualification: role(RegenerativeLineageRoleV1::SuccessorQualification, h4),
        successor_reproduction_horizon: h4,
        regenerative_viability_horizon: h4,
        limiting_roles: vec![
            RegenerativeLineageRoleV1::SuccessorConstruction,
            RegenerativeLineageRoleV1::SuccessorQualification,
        ],
        fully_modeled_regenerative_viability: true,
    }
}

fn reproduction_policies() -> Vec<RegenerativeReproductionPolicySpecV1> {
    (1..=4)
        .map(|maturity_periods| RegenerativeReproductionPolicySpecV1 {
            reproduction_policy_id: format!("maturity-{maturity_periods}"),
            reproduction_policy_evidence_binding: format!(
                "reproduction-policy:maturity-{maturity_periods}:v1"
            ),
            maturity_periods,
        })
        .collect()
}

fn expected_preserved_counts() -> Vec<(u64, u64, u64, u64)> {
    FIXTURE
        .lines()
        .filter_map(|line| line.strip_prefix("preserved_basis_policy="))
        .map(|line| {
            let mut maturity = 0;
            let mut founded = 0;
            let mut matured = 0;
            let mut transitions = 0;
            let mut residual = 0;
            for field in line.split('|') {
                let (key, value) = field.split_once(':').unwrap();
                let value: u64 = value.parse().unwrap();
                match key {
                    "maturity_periods" => maturity = value,
                    "founded_descendants" => founded = value,
                    "maturity_completed_descendants" => matured = value,
                    "descendant_reproduction_transitions" => transitions = value,
                    "terminal_residual_periods" => residual = value,
                    other => panic!("unknown fixture field {other}"),
                }
            }
            (maturity, founded, matured, transitions.max(residual))
        })
        .collect()
}

#[test]
fn changed_support_basis_blocks_h4_scalar_policy_projection() {
    assert!(!bool_scalar("changed_basis_scalar_projection_safe"));
    let source = model(
        "closure-v3-basis",
        scalar("source_period_duration_ms"),
        "forge-tooling-v3",
        scalar("source_tooling_demand_units_per_period"),
        scalar("source_tooling_inventory_units"),
        "reactor-service-v3",
        scalar("nonlimiting_reactor_inventory_units"),
    );
    let successor = model(
        "closure-v4-basis",
        scalar("changed_successor_period_duration_ms"),
        "forge-tooling-v4",
        scalar("changed_successor_tooling_demand_units_per_period"),
        0,
        "reactor-service-v4",
        0,
    );
    let (evidence, handoff_report) = handoff();
    let basis = assess_intergenerational_support_basis(
        &basis_policy(),
        &evidence,
        &handoff_report,
        &source,
        &successor,
    )
    .unwrap();
    assert!(!basis.scalar_runway_projection_safe);
    assert_eq!(basis.assessments[0].source_stockpile_draw_units_per_period, 2);
    assert_eq!(basis.assessments[0].successor_stockpile_draw_units_per_period, 1);
    assert!(matches!(
        evaluate_basis_qualified_regenerative_policy_sensitivity_surface(
            &lineage_report(),
            &reproduction_policies(),
            &basis,
        ),
        Err(RegenerativeBasisQualifiedPolicySensitivityError::SupportBasis(
            RegenerativeSupportBasisError::ScalarRunwayProjectionUnsafe
        ))
    ));
}

#[test]
fn preserved_support_basis_allows_the_canonical_h4_surface() {
    assert!(bool_scalar("preserved_basis_scalar_projection_safe"));
    let source = model(
        "closure-v3-basis",
        scalar("source_period_duration_ms"),
        "forge-tooling-v3",
        scalar("source_tooling_demand_units_per_period"),
        scalar("source_tooling_inventory_units"),
        "reactor-service-v3",
        scalar("nonlimiting_reactor_inventory_units"),
    );
    let successor = model(
        "closure-v4-basis",
        scalar("preserved_successor_period_duration_ms"),
        "forge-tooling-v4",
        scalar("preserved_successor_tooling_demand_units_per_period"),
        0,
        "reactor-service-v4",
        0,
    );
    let (evidence, handoff_report) = handoff();
    let basis = assess_intergenerational_support_basis(
        &basis_policy(),
        &evidence,
        &handoff_report,
        &source,
        &successor,
    )
    .unwrap();
    assert!(basis.scalar_runway_projection_safe);

    let surface = evaluate_basis_qualified_regenerative_policy_sensitivity_surface(
        &lineage_report(),
        &reproduction_policies(),
        &basis,
    )
    .unwrap();
    let observed: Vec<_> = surface
        .policy_points
        .iter()
        .map(|point| {
            let projection = &point.generation_policy_projection;
            let finite = |value| match value {
                RegenerativeGenerationCountV1::Finite(value) => value,
                RegenerativeGenerationCountV1::IndefiniteUnderStaticModel => {
                    panic!("expected finite H=4 projection")
                }
            };
            (
                projection.maturity_periods,
                finite(projection.founded_descendant_generations),
                finite(projection.maturity_completed_descendant_generations),
                finite(projection.descendant_reproduction_transitions),
            )
        })
        .collect();
    assert_eq!(
        observed,
        vec![(1, 4, 4, 3), (2, 2, 2, 1), (3, 2, 1, 1), (4, 1, 1, 0)]
    );
    assert_eq!(expected_preserved_counts().len(), 4);
}
