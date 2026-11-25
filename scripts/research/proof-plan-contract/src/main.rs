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
            let accounting_complete = report.errors.is_empty()
                && unresolved == 0
                && cycles == 0
                && report.primary_alternative_count == 1
                && report
                    .alternatives
                    .iter()
                    .any(|alternative| !alternative.terminal_goals.is_empty());

            println!("PROOF_PLAN_STRUCTURALLY_VALID plan_id={}", plan.plan_id);
            println!(
                "GLOBAL_OBLIGATION_ACCOUNTING={}",
                if accounting_complete { "complete" } else { "incomplete" }
            );
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
