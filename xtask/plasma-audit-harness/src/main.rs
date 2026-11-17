use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let repository_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let report = xtask::plasma_energy_audit::run_audit(
        &repository_root.join("data/research/plasma_energy_reference_cases_v0_1.json"),
        &repository_root.join("data/research/hts_fusion_magnet_reference_cases_v0_1.json"),
        &repository_root.join("data/research/thermal_store_accounting_fixtures_v0_1.json"),
    )?;
    xtask::plasma_energy_audit::print_report(&report, true)?;

    let source_register = std::fs::read_to_string(
        repository_root.join("data/research/plasma_energy_reference_cases_v0_1.json"),
    )?;
    let (registered_disposition, registered_groups) =
        xtask::plasma_pathway_decision::evaluate_registered_target_json(&source_register)
            .map_err(anyhow::Error::msg)?;
    println!("PATHWAY TARGET: {registered_disposition}");
    for (group_id, disposition) in &registered_groups {
        println!("  {group_id}: {disposition}");
    }
    anyhow::ensure!(
        registered_disposition == "ABSTAIN_ALL_GROUPS_NOT_COMPARABLE",
        "source pathway groups must abstain until service contracts are matched"
    );

    let fixture = std::fs::read_to_string(
        repository_root
            .join("xtask/plasma-audit-harness/fixtures/pathway-selection-synthetic-v0.json"),
    )?;
    let demo = xtask::plasma_pathway_decision::evaluate_decision_manifest_json(&fixture)
        .map_err(anyhow::Error::msg)?;
    println!(
        "SYNTHETIC PATHWAY DEMO: {} groups / {} paired draws / no physical result",
        demo.groups.len(),
        demo.draws
    );
    let sensitivity_settings = std::fs::read_to_string(
        repository_root
            .join("xtask/plasma-audit-harness/fixtures/pathway-sensitivity-v0.json"),
    )?;
    let sensitivity =
        xtask::plasma_pathway_decision::evaluate_decision_sensitivity_json(
            &fixture,
            &sensitivity_settings,
        )
        .map_err(anyhow::Error::msg)?;
    println!(
        "SYNTHETIC SENSITIVITY: {} runs / {} seeds / {} draw budgets / no physical calibration",
        sensitivity.run_count,
        sensitivity.seed_count,
        sensitivity.draw_counts.len()
    );
    for group in &sensitivity.groups {
        println!(
            "  {}: {}",
            group.group_id,
            group.disposition_counts
                .iter()
                .map(|(disposition, count)| format!("{disposition}={count}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        for route in &group.routes {
            println!(
                "    {}: feasibility range {:.5}; robust classification stable={}; Pareto range={:?}",
                route.route_id,
                route.feasibility_probability_range,
                route.robust_classification_stable,
                route.pareto_membership_probability_range
            );
        }
    }
    anyhow::ensure!(
        sensitivity.run_count == 6
            && sensitivity.claim_ceiling
                == "MONTE_CARLO_STABILITY_ONLY_NOT_EMPIRICAL_CALIBRATION_OR_PHYSICAL_VALIDATION",
        "pathway sensitivity must remain a six-run synthetic stability analysis"
    );
    anyhow::ensure!(
        demo.groups
            .iter()
            .all(|group| group.disposition == "SYNTHETIC_PAIRED_PARETO_ANALYSIS"),
        "synthetic paired pathway demonstration did not complete"
    );
    Ok(())
}
