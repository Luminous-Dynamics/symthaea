use std::collections::BTreeSet;
use symthaea_maritime_core::{
    DependencyGovernance, RegenerativeCapability, RegenerativeClosureModel,
    RegenerativeDependency, RegenerativeDependencyKind, RegenerativeHorizon,
};

const FIXTURE: &str = include_str!("../fixtures/manta-forge-regenerative-v1.txt");

fn scalar(key: &str) -> &str {
    FIXTURE
        .lines()
        .find_map(|line| line.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("missing fixture key {key}"))
}

#[test]
fn manta_forge_fixture_pins_static_closure_theorem() {
    assert_eq!(scalar("schema"), "manta-forge-regenerative-v1");

    let dependencies = FIXTURE
        .lines()
        .filter_map(|line| line.strip_prefix("dependency="))
        .map(|record| {
            let fields: Vec<_> = record.split('|').collect();
            assert_eq!(fields.len(), 7);
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
                unit_mass_grams: Some(fields[6].parse().unwrap()),
                evidence_binding: format!("fixture:{}", fields[0]),
            }
        })
        .collect();

    let capability_record = scalar("capability");
    let capability_fields: Vec<_> = capability_record.split('|').collect();
    assert_eq!(capability_fields.len(), 3);
    let capability = RegenerativeCapability {
        capability_id: capability_fields[0].into(),
        essential: match capability_fields[1] {
            "essential" => true,
            "optional" => false,
            other => panic!("unexpected capability class {other}"),
        },
        dependency_ids: capability_fields[2].split(',').map(str::to_owned).collect::<BTreeSet<_>>(),
        evidence_binding: "fixture:capability".into(),
    };

    let model = RegenerativeClosureModel {
        model_id: scalar("schema").into(),
        period_duration_ms: scalar("period_duration_ms").parse().unwrap(),
        dependencies,
        capabilities: vec![capability],
        evidence_binding: "fixture:model".into(),
    };
    let report = model.evaluate().unwrap();

    assert_eq!(
        report.physical_mass_flow_closure_basis_points,
        Some(scalar("expected_mass_closure_basis_points").parse().unwrap())
    );
    assert_eq!(
        report.essential_horizon,
        RegenerativeHorizon::FinitePeriods(
            scalar("expected_static_horizon_periods").parse().unwrap()
        )
    );
    assert_eq!(
        report.limiting_dependency_ids,
        vec![scalar("expected_limiting_dependency").to_string()]
    );

    // The dynamic expectations are carried in the same byte-level fixture and
    // consumed by Symtropy's independent simulator test.
    assert_eq!(scalar("expected_dynamic_survived_ticks"), "30");
    assert_eq!(scalar("expected_dynamic_failure_tick"), "31");
}
