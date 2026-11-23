use std::env;
use std::fs;
use std::process::ExitCode;

use proof_plan_contract::local_global::{compute_cirr, validate_pair_json};

fn main() -> ExitCode {
    let Some(path) = env::args().nth(1) else {
        eprintln!("usage: local-global <LocalGlobalPairV1.json>");
        return ExitCode::from(2);
    };

    let input = match fs::read_to_string(&path) {
        Ok(input) => input,
        Err(error) => {
            eprintln!("could not read {path}: {error}");
            return ExitCode::from(2);
        }
    };

    let pair = match validate_pair_json(&input) {
        Ok(pair) => pair,
        Err(errors) => {
            eprintln!("LOCAL_GLOBAL_PAIR_STRUCTURALLY_INVALID");
            for error in errors {
                eprintln!("- {error}");
            }
            return ExitCode::from(1);
        }
    };

    let summary = match compute_cirr(std::slice::from_ref(&pair)) {
        Ok(summary) => summary,
        Err(errors) => {
            eprintln!("CIRR_COMPUTATION_REJECTED");
            for error in errors {
                eprintln!("- {error}");
            }
            return ExitCode::from(1);
        }
    };

    println!(
        "LOCAL_GLOBAL_PAIR_STRUCTURALLY_VALID pair_id={}",
        pair.candidate.pair_id
    );
    println!("scope_match={}", pair.scope_matches());
    if let Some(evaluator) = &pair.evaluator_only {
        if let Some(countermodel) = &evaluator.finite_boolean_countermodel {
            println!(
                "finite_boolean_countermodel=verified premises={} conclusion=false",
                countermodel.premises.len()
            );
        }
    }
    println!(
        "cirr_numerator={} cirr_denominator={} unresolved={}",
        summary.numerator_definite_reuse,
        summary.denominator_eligible_local_rejections,
        summary.unresolved_global_outcomes
    );
    match (
        summary.lower_bound,
        summary.upper_bound,
        summary.point_estimate,
    ) {
        (Some(lower), Some(upper), Some(point)) => {
            println!(
                "cirr_lower_bound={lower:.6} cirr_upper_bound={upper:.6} cirr_point_estimate={point:.6}"
            );
        }
        (Some(lower), Some(upper), None) => {
            println!(
                "cirr_lower_bound={lower:.6} cirr_upper_bound={upper:.6} cirr_point_estimate=undefined"
            );
        }
        _ => println!("cirr_point_estimate=undefined (zero eligible denominator)"),
    }
    println!("claim_ceiling=synthetic_or_receipt_structure_only_not_lean_kernel_proof");
    ExitCode::SUCCESS
}
