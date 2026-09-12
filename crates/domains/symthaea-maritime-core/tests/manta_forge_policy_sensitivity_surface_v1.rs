use symthaea_maritime_core::{
    evaluate_regenerative_policy_sensitivity_surface, RegenerativeGenerationCountV1,
    RegenerativeHorizon, RegenerativeLineageRoleReportV1, RegenerativeLineageRoleV1,
    RegenerativeLineageViabilityReportV1, RegenerativeReproductionPolicySpecV1,
};

const FIXTURE: &str = include_str!("../fixtures/manta-forge-policy-sensitivity-surface-v1.txt");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExpectedPoint {
    maturity_periods: u64,
    founded: u64,
    matured: u64,
    transitions: u64,
    residual: u64,
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

fn manta_v3_report() -> RegenerativeLineageViabilityReportV1 {
    RegenerativeLineageViabilityReportV1 {
        profile_id: "profile:manta-v3-policy-surface".into(),
        genome_id: "genome:manta-v3".into(),
        closure_model_id: "closure:manta-v3".into(),
        flow_support_id: "support:manta-v3".into(),
        operation: role(
            RegenerativeLineageRoleV1::Operation,
            RegenerativeHorizon::FinitePeriods(8),
        ),
        successor_construction: role(
            RegenerativeLineageRoleV1::SuccessorConstruction,
            RegenerativeHorizon::FinitePeriods(4),
        ),
        successor_qualification: role(
            RegenerativeLineageRoleV1::SuccessorQualification,
            RegenerativeHorizon::FinitePeriods(5),
        ),
        successor_reproduction_horizon: RegenerativeHorizon::FinitePeriods(4),
        regenerative_viability_horizon: RegenerativeHorizon::FinitePeriods(4),
        limiting_roles: vec![RegenerativeLineageRoleV1::SuccessorConstruction],
        fully_modeled_regenerative_viability: true,
    }
}

fn expected_points() -> Vec<ExpectedPoint> {
    FIXTURE
        .lines()
        .filter_map(|line| line.strip_prefix("policy="))
        .map(|line| {
            let mut point = ExpectedPoint {
                maturity_periods: 0,
                founded: 0,
                matured: 0,
                transitions: 0,
                residual: 0,
            };
            for field in line.split('|') {
                let (key, value) = field.split_once(':').unwrap();
                let value: u64 = value.parse().unwrap();
                match key {
                    "maturity_periods" => point.maturity_periods = value,
                    "founded_descendants" => point.founded = value,
                    "maturity_completed_descendants" => point.matured = value,
                    "descendant_reproduction_transitions" => point.transitions = value,
                    "terminal_residual_periods" => point.residual = value,
                    other => panic!("unknown policy-surface field {other}"),
                }
            }
            point
        })
        .collect()
}

fn finite(value: RegenerativeGenerationCountV1) -> u64 {
    match value {
        RegenerativeGenerationCountV1::Finite(value) => value,
        RegenerativeGenerationCountV1::IndefiniteUnderStaticModel => {
            panic!("expected finite MANTA policy surface")
        }
    }
}

#[test]
fn exact_manta_v3_physics_projects_to_the_shared_policy_surface() {
    let expected = expected_points();
    let policies: Vec<_> = expected
        .iter()
        .map(|point| RegenerativeReproductionPolicySpecV1 {
            reproduction_policy_id: format!("manta-v3-maturity-{}", point.maturity_periods),
            reproduction_policy_evidence_binding: format!(
                "policy:manta-v3:maturity:{}:v1",
                point.maturity_periods
            ),
            maturity_periods: point.maturity_periods,
        })
        .collect();

    let surface =
        evaluate_regenerative_policy_sensitivity_surface(&manta_v3_report(), &policies).unwrap();
    assert_eq!(
        surface.physical_successor_reproduction_horizon,
        RegenerativeHorizon::FinitePeriods(4)
    );
    assert_eq!(
        surface.physical_regenerative_viability_horizon,
        RegenerativeHorizon::FinitePeriods(4)
    );
    assert!(surface.fully_modeled_successor_reproduction);
    assert!(surface.fully_modeled_regenerative_viability);
    assert_eq!(surface.policy_points.len(), expected.len());

    for (point, expected) in surface.policy_points.iter().zip(expected) {
        let projection = &point.generation_policy_projection;
        assert_eq!(projection.maturity_periods, expected.maturity_periods);
        assert_eq!(finite(projection.founded_descendant_generations), expected.founded);
        assert_eq!(
            finite(projection.maturity_completed_descendant_generations),
            expected.matured
        );
        assert_eq!(
            finite(projection.descendant_reproduction_transitions),
            expected.transitions
        );
        assert_eq!(projection.terminal_generation_residual_periods, Some(expected.residual));
    }
}
