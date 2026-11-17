use subterra_inspection_campaign::{
    AssuranceBindings, CampaignBudget, CampaignProblem, DecisionRequirement, DeploymentBundle,
    GreedyMetric, InspectionOption, InventoryDisposition, ResourceVector, greedy_baseline,
    optimize_campaign,
};

fn option(option_id: &str, decision_id: &str, value: u64, energy: u64) -> InspectionOption {
    InspectionOption {
        option_id: option_id.into(),
        decision_id: decision_id.into(),
        deployment_bundle_id: "west_corridor_mobilization".into(),
        gross_evsi_micro_units: value,
        unknown_state_debt_reduction_units: value / 2,
        variable_resources: ResourceVector {
            energy_millijoules: energy,
            duration_milliseconds: 2_000,
            data_bytes: 5_000,
            exposure_milliunits: 10,
        },
        mission_interruption_upper_ppm: 1_000,
        robot_loss_upper_ppm: 500,
        external_gate_passed: true,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let problem = CampaignProblem {
        campaign_id: "synthetic-west-corridor".into(),
        decisions: vec![
            DecisionRequirement {
                decision_id: "pipe-segment-14".into(),
                mandatory: false,
            },
            DecisionRequirement {
                decision_id: "cable-support-08".into(),
                mandatory: false,
            },
            DecisionRequirement {
                decision_id: "drawpoint-03".into(),
                mandatory: false,
            },
        ],
        deployment_bundles: vec![DeploymentBundle {
            bundle_id: "west_corridor_mobilization".into(),
            fixed_resources: ResourceVector {
                energy_millijoules: 1_000,
                duration_milliseconds: 10_000,
                data_bytes: 1_000,
                exposure_milliunits: 100,
            },
        }],
        options: vec![
            option("pipe-high-value-long-scan", "pipe-segment-14", 19, 10_000),
            option("cable-targeted-followup", "cable-support-08", 10, 6_000),
            option("drawpoint-metrology", "drawpoint-03", 10, 6_000),
        ],
        budget: CampaignBudget {
            resources: ResourceVector {
                energy_millijoules: 13_000,
                duration_milliseconds: 20_000,
                data_bytes: 20_000,
                exposure_milliunits: 500,
            },
            mission_interruption_upper_ppm: 10_000,
            robot_loss_upper_ppm: 5_000,
        },
        assurance: AssuranceBindings {
            common_cause_graph_digest: "synthetic-only:graph".into(),
            fault_domain_profile_digest: "synthetic-only:profile".into(),
            common_cause_assessment_digest: "synthetic-only:assessment".into(),
            portfolio_value_assessment_digest: "synthetic-only:factorization".into(),
            inventory_disposition: InventoryDisposition::Unknown,
            additive_value_factorization_complete: false,
        },
        maximum_search_nodes: 100_000,
    };

    let exact = optimize_campaign(&problem)?;
    let greedy = greedy_baseline(&problem, GreedyMetric::GrossEvsi)?;
    let energy_greedy = greedy_baseline(&problem, GreedyMetric::EvsiPerEnergy)?;
    let time_greedy = greedy_baseline(&problem, GreedyMetric::EvsiPerDuration)?;

    println!("campaign={}", exact.campaign_id);
    println!("exact_options={:?}", exact.selection.selected_option_ids);
    println!(
        "exact_additive_evsi={} optimality={:?} nodes={}",
        exact.selection.additive_evsi_estimate_micro_units,
        exact.search_disposition,
        exact.evaluated_nodes
    );
    println!(
        "exact_resources energy_mj={} duration_ms={} data_bytes={} exposure_milliunits={}",
        exact.selection.resources.energy_millijoules,
        exact.selection.resources.duration_milliseconds,
        exact.selection.resources.data_bytes,
        exact.selection.resources.exposure_milliunits
    );
    println!("assurance={:?}", exact.assurance_disposition);
    println!(
        "greedy_value={} energy_greedy_value={} time_greedy_value={}",
        greedy.selection.additive_evsi_estimate_micro_units,
        energy_greedy.selection.additive_evsi_estimate_micro_units,
        time_greedy.selection.additive_evsi_estimate_micro_units
    );
    Ok(())
}
