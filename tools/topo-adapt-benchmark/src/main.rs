use std::collections::BTreeSet;

use topo_adapt_benchmark::{evaluate_episode, Fixture, FIXTURE_JSON};

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
                let id = edge.id.as_deref().ok_or("candidate edge has no ID")?;
                variants.push((format!("edit:{id}"), vec![id]));
            }

            for (variant, edits) in variants {
                let result = evaluate_episode(
                    case,
                    &fixture.graph_semantics,
                    time,
                    &edits,
                )?;
                reports.push(serde_json::json!({
                    "case_id": case.id,
                    "time": time,
                    "variant": variant,
                    "accepted_candidate_edges": edits,
                    "max_flow": result.max_flow,
                    "min_cut": result.min_cut,
                    "requested_flow": result.requested_flow
                }));
            }
        }
    }

    println!("{}", serde_json::to_string_pretty(&reports)?);
    Ok(())
}
