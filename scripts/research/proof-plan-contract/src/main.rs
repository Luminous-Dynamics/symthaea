use std::env;
use std::fs;
use std::process::ExitCode;

use proof_plan_contract::global_obligations::analyze_global_obligations;
use proof_plan_contract::validate_json;

fn main() -> ExitCode {
    let Some(path) = env::args().nth(1) else {
        eprintln!("usage: proof-plan-contract <ProofPlanV1.json>");
        return ExitCode::from(2);
    };

    let input = match fs::read_to_string(&path) {
        Ok(input) => input,
        Err(error) => {
            eprintln!("could not read {path}: {error}");
            return ExitCode::from(2);
        }
    };

    match validate_json(&input) {
        Ok(plan) => {
            let report = analyze_global_obligations(&plan);
            let produced = report
                .alternatives
                .iter()
                .map(|alternative| alternative.produced_subgoals.len())
                .sum::<usize>();
            let transformed = report
                .alternatives
                .iter()
                .map(|alternative| alternative.transformed_goals.len())
                .sum::<usize>();
            let discharged = report
                .alternatives
                .iter()
                .map(|alternative| alternative.discharged_goals.len())
                .sum::<usize>();
            let unresolved = report
                .alternatives
                .iter()
                .map(|alternative| alternative.unresolved_goals.len())
                .sum::<usize>();
            let cycles = report
                .alternatives
                .iter()
                .map(|alternative| alternative.cycle_nodes.len())
                .sum::<usize>();
            let primary_id = plan
                .alternatives
                .iter()
                .find(|alternative| matches!(&alternative.status, proof_plan_contract::AlternativeStatus::Primary))
                .map(|alternative| alternative.id.as_str());
            let primary = report
                .alternatives
                .iter()
                .find(|alternative| Some(alternative.alternative_id.as_str()) == primary_id);
            let primary_unresolved = primary.map_or(0, |alternative| alternative.unresolved_goals.len());
            let primary_cycles = primary.map_or(0, |alternative| alternative.cycle_nodes.len());
            let primary_terminal = primary.map_or(0, |alternative| alternative.terminal_goals.len());
            let accounting_complete = report.errors.is_empty()
                && primary_unresolved == 0
                && primary_cycles == 0
                && primary_terminal > 0
                && report.primary_alternative_count == 1;

            println!("PROOF_PLAN_STRUCTURALLY_VALID plan_id={}", plan.plan_id);
            println!(
                "PRIMARY_ROUTE_ACCOUNTING={}",
                if accounting_complete { "complete" } else { "incomplete" }
            );
            println!("primary_route_unresolved_goals={primary_unresolved}");
            println!("primary_route_terminal_goals={primary_terminal}");
            println!("primary_route_cycle_nodes={primary_cycles}");
            println!("primary_alternatives={}", report.primary_alternative_count);
            println!("produced_subgoals={produced}");
            println!("transformed_goals={transformed}");
            println!("discharged_goals={discharged}");
            println!("unresolved_goals={unresolved}");
            println!("goal_transition_cycle_nodes={cycles}");
            for alternative in &report.alternatives {
                println!(
                    "ALTERNATIVE={} produced={} transformed={} discharged={} unresolved={} cycles={}",
                    alternative.alternative_id,
                    alternative.produced_subgoals.len(),
                    alternative.transformed_goals.len(),
                    alternative.discharged_goals.len(),
                    alternative.unresolved_goals.len(),
                    alternative.cycle_nodes.len()
                );
                for goal in &alternative.unresolved_goals {
                    println!("UNRESOLVED_GOAL alternative={} goal={goal}", alternative.alternative_id);
                }
            }
            println!("claim_ceiling=graph_and_obligation_accounting_only; no_mathematical_or_kernel_verdict");
            ExitCode::SUCCESS
        }
        Err(errors) => {
            eprintln!("PROOF_PLAN_STRUCTURALLY_INVALID");
            for error in errors {
                eprintln!("- {error}");
            }
            eprintln!("claim_ceiling=no_mathematical_or_kernel_verdict");
            ExitCode::from(1)
        }
    }
}
