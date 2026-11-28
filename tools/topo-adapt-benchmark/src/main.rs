use std::collections::BTreeSet;

use topo_adapt_benchmark::{evaluate_episode, select_best_single_edge, Fixture, FIXTURE_JSON};

fn main() {
    if let Err(error) = run() {
        eprintln!("topology benchmark failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let fixture = Fixture::parse(FIXTURE_JSON)?;
    fixture.validate()?;

    let mut reports = Vec::new();
    for case in &fixture.cases {
        let mut times: BTreeSet<u64> = case.demand_schedule.iter().map(|s| s.time).collect();
        times.extend(case.interventions.iter().map(|i| i.time));

        for time in times {
            let mut variants: Vec<(String, Vec<&str>)> =
                vec![("no_edit".to_string(), Vec::new())];
            for edge in &case.candidate_edges {
                let id = edge
                    .id
                    .as_deref()
                    .ok_or_else(|| format!("{}: candidate edge has no ID", case.id))?;
                variants.push((format!("edit:{id}"), vec![id]));
            }

            for (variant, edits) in variants {
                let result = evaluate_episode(case, &fixture.graph_semantics, time, &edits)?;
                reports.push(serde_json::json!({
                    "case_id": case.id,
                    "time": time,
                    "variant": variant,
                    "accepted_candidate_edges": edits,
                    "max_flow": result.max_flow,
                    "min_cut": result.min_cut,
                    "requested_flow": result.requested_flow,
                    "accepted_edge_additions": result.accepted_edge_additions,
                    "edit_cost_units": result.edit_cost_units,
                    "edge_maintenance_cost_units": result.edge_maintenance_cost_units
                }));
            }

            let decision = select_best_single_edge(case, &fixture.graph_semantics, time)?;
            let selected_edges: Vec<&str> = decision
                .selected_candidate_id
                .as_deref()
                .into_iter()
                .collect();
            reports.push(serde_json::json!({
                "case_id": case.id,
                "time": time,
                "variant": "one_step_candidate_sweep_oracle_upper_bound",
                "selected_candidate_edge": decision.selected_candidate_id,
                "accepted_candidate_edges": selected_edges,
                "max_flow": decision.result.max_flow,
                "min_cut": decision.result.min_cut,
                "requested_flow": decision.result.requested_flow,
                "accepted_edge_additions": decision.result.accepted_edge_additions,
                "edit_cost_units": decision.result.edit_cost_units,
                "edge_maintenance_cost_units": decision.result.edge_maintenance_cost_units
            }));
        }
    }

    let source_commit = std::env::var("GITHUB_SHA").ok();
    let output = serde_json::json!({
        "status": "synthetic_reference_only",
        "fixture_schema": fixture.schema_version,
        "fixture_blake3": blake3::hash(FIXTURE_JSON.as_bytes()).to_hex().to_string(),
        "source_commit": source_commit,
        "evaluations": reports,
        "claim_limit": "No trained controller, biological mechanism, physical repair or SOTA claim."
    });
    let rendered = serde_json::to_string_pretty(&output).map_err(|e| e.to_string())?;
    println!("{rendered}");
    Ok(())
}
