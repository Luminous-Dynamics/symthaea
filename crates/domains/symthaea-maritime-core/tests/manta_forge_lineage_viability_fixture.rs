use std::collections::{BTreeMap, BTreeSet};
use symthaea_maritime_core::{
    DependencyGovernance, RegenerativeCapability, RegenerativeClosureModel,
    RegenerativeDependency, RegenerativeDependencyKind, RegenerativeFlowSupportV1,
    RegenerativeGenomeRequirementV1, RegenerativeGenomeV1, RegenerativeHorizon,
    RegenerativeLineageRoleV1, RegenerativeLineageViabilityProfileV1,
    REGENERATIVE_FLOW_SUPPORT_SCHEMA_V1, REGENERATIVE_GENOME_SCHEMA_V1,
    REGENERATIVE_LINEAGE_VIABILITY_SCHEMA_V1, evaluate_regenerative_lineage_viability,
};

const FIXTURE: &str = include_str!("../fixtures/manta-forge-lineage-viability-v1.txt");

fn scalar(key: &str) -> &str {
    FIXTURE
        .lines()
        .find_map(|line| line.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("missing fixture key {key}"))
}

fn role_capabilities() -> BTreeMap<&'static str, Vec<String>> {
    FIXTURE
        .lines()
        .filter_map(|line| line.strip_prefix("role="))
        .map(|record| {
            let fields: Vec<_> = record.split('|').collect();
            assert_eq!(fields.len(), 2);
            let role = match fields[0] {
                "operation" => "operation",
                "successor_construction" => "successor_construction",
                "successor_qualification" => "successor_qualification",
                other => panic!("unexpected role {other}"),
            };
            (
                role,
                fields[1]
                    .split(',')
                    .map(str::to_owned)
                    .collect::<Vec<_>>(),
            )
        })
        .collect()
}

#[test]
fn manta_forge_fixture_pins_role_separated_lineage_viability() {
    assert_eq!(scalar("schema"), "manta-forge-lineage-viability-v1");

    let dependencies = FIXTURE
        .lines()
        .filter_map(|line| line.strip_prefix("dependency="))
        .map(|record| {
            let fields: Vec<_> = record.split('|').collect();
            assert_eq!(fields.len(), 6);
            RegenerativeDependency {
                dependency_id: fields[0].into(),
                kind: RegenerativeDependencyKind::Component,
                governance: match fields[1] {
                    "ordinary" => DependencyGovernance::Ordinary,
                    "safeguarded_external" => DependencyGovernance::SafeguardedExternal,
                    other => panic!("unexpected governance {other}"),
                },
                demand_units_per_period: fields[2].parse().unwrap(),
                local_production_units_per_period: fields[3].parse().unwrap(),
                recycling_units_per_period: fields[4].parse().unwrap(),
                stockpile_units: fields[5].parse().unwrap(),
                unit_mass_grams: Some(1),
                evidence_binding: format!("fixture:{}", fields[0]),
            }
        })
        .collect::<Vec<_>>();

    let capabilities = FIXTURE
        .lines()
        .filter_map(|line| line.strip_prefix("capability="))
        .map(|record| {
            let fields: Vec<_> = record.split('|').collect();
            assert_eq!(fields.len(), 3);
            RegenerativeCapability {
                capability_id: fields[0].into(),
                essential: match fields[1] {
                    "essential" => true,
                    "optional" => false,
                    other => panic!("unexpected capability class {other}"),
                },
                dependency_ids: fields[2]
                    .split(',')
                    .map(str::to_owned)
                    .collect::<BTreeSet<_>>(),
                evidence_binding: format!("fixture:capability:{}", fields[0]),
            }
        })
        .collect::<Vec<_>>();

    let model = RegenerativeClosureModel {
        model_id: scalar("schema").into(),
        period_duration_ms: scalar("period_duration_ms").parse().unwrap(),
        dependencies,
        capabilities: capabilities.clone(),
        evidence_binding: "fixture:lineage-model".into(),
    };

    let requirements = capabilities
        .iter()
        .map(|capability| {
            let dependency_id = capability.dependency_ids.iter().next().unwrap().clone();
            RegenerativeGenomeRequirementV1 {
                requirement_id: format!("req-{}", capability.capability_id),
                capability_id: capability.capability_id.clone(),
                baseline_dependency_id: dependency_id,
                design_binding: format!("fixture:design:{}", capability.capability_id),
                metrology_profile_binding: format!(
                    "fixture:metrology:{}",
                    capability.capability_id
                ),
                requalification_profile_binding: format!(
                    "fixture:requalification:{}",
                    capability.capability_id
                ),
                disassembly_profile_binding: format!(
                    "fixture:disassembly:{}",
                    capability.capability_id
                ),
                recovery_profile_binding: format!(
                    "fixture:recovery:{}",
                    capability.capability_id
                ),
                qualified_substitution_bindings: Vec::new(),
            }
        })
        .collect::<Vec<_>>();

    let genome = RegenerativeGenomeV1 {
        schema_version: REGENERATIVE_GENOME_SCHEMA_V1,
        genome_id: "fixture:manta-v1".into(),
        lineage_parent_binding: None,
        closure_model_id: model.model_id.clone(),
        closure_model_evidence_binding: model.evidence_binding.clone(),
        requirements,
        evidence_binding: "fixture:genome:manta-v1".into(),
    };
    let support = RegenerativeFlowSupportV1 {
        schema_version: REGENERATIVE_FLOW_SUPPORT_SCHEMA_V1,
        support_id: "fixture:lineage-support".into(),
        closure_model_id: model.model_id.clone(),
        closure_model_evidence_binding: model.evidence_binding.clone(),
        claims: Vec::new(),
        evidence_binding: "fixture:lineage-support-evidence".into(),
    };

    let roles = role_capabilities();
    let profile = RegenerativeLineageViabilityProfileV1 {
        schema_version: REGENERATIVE_LINEAGE_VIABILITY_SCHEMA_V1,
        profile_id: "fixture:lineage-profile".into(),
        closure_model_id: model.model_id.clone(),
        closure_model_evidence_binding: model.evidence_binding.clone(),
        operational_capability_ids: roles["operation"].clone(),
        successor_construction_capability_ids: roles["successor_construction"].clone(),
        successor_qualification_capability_ids: roles["successor_qualification"].clone(),
        evidence_binding: "fixture:lineage-profile-evidence".into(),
    };

    let report = evaluate_regenerative_lineage_viability(&profile, &genome, &model, &support)
        .expect("fixture should produce lineage viability evidence");

    assert_eq!(
        report.operation.conservative_horizon,
        RegenerativeHorizon::FinitePeriods(
            scalar("expected_operation_horizon_periods")
                .parse()
                .unwrap()
        )
    );
    assert_eq!(
        report.successor_construction.conservative_horizon,
        RegenerativeHorizon::FinitePeriods(
            scalar("expected_construction_horizon_periods")
                .parse()
                .unwrap()
        )
    );
    assert_eq!(
        report.successor_qualification.conservative_horizon,
        RegenerativeHorizon::FinitePeriods(
            scalar("expected_qualification_horizon_periods")
                .parse()
                .unwrap()
        )
    );
    assert_eq!(
        report.regenerative_viability_horizon,
        RegenerativeHorizon::FinitePeriods(
            scalar("expected_regenerative_viability_horizon_periods")
                .parse()
                .unwrap()
        )
    );
    assert_eq!(
        report.limiting_roles,
        vec![RegenerativeLineageRoleV1::SuccessorQualification]
    );
    assert_eq!(scalar("expected_limiting_role"), "successor_qualification");
    assert_eq!(
        report.successor_qualification.root_limiting_dependency_ids,
        vec![scalar("expected_limiting_dependency").to_string()]
    );
    assert!(report.fully_modeled_regenerative_viability);

    // Symtropy consumes the same fixture bytes and independently proves that
    // each role becomes unavailable on the next tick after its static horizon.
    assert_eq!(scalar("expected_operation_failure_tick"), "101");
    assert_eq!(scalar("expected_construction_failure_tick"), "41");
    assert_eq!(scalar("expected_qualification_failure_tick"), "31");
}
