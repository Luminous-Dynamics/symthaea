include!("manta_forge_intergenerational_support_basis_v1.rs");

fn exact_preserved_fixture_points() -> Vec<(u64, u64, u64, u64, u64)> {
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
            (maturity, founded, matured, transitions, residual)
        })
        .collect()
}

#[test]
fn preserved_basis_surface_matches_all_five_fixture_coordinates_exactly() {
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
                projection.terminal_generation_residual_periods.unwrap(),
            )
        })
        .collect();

    assert_eq!(observed, exact_preserved_fixture_points());
}
