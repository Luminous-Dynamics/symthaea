use std::env;
use std::fs;
use std::process::ExitCode;

use proof_plan_contract::local_global::{
    check_finite_propositional_entailment, compute_cirr, validate_pair_json,
    FiniteTruthTableResult, MAX_FINITE_BOOLEAN_ATOMS,
};

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
            match check_finite_propositional_entailment(
                &countermodel.premises,
                &countermodel.conclusion,
                MAX_FINITE_BOOLEAN_ATOMS,
            ) {
                Ok(FiniteTruthTableResult::Countermodel {
                    valuation,
                    assignments_checked,
                }) => {
                    println!(
                        "finite_truth_table=countermodel assignments_checked={} witness={}",
                        assignments_checked,
                        valuation
                            .iter()
                            .map(|assignment| format!("{}={}", assignment.atom, assignment.value))
                            .collect::<Vec<_>>()
                            .join(",")
                    );
                }
                Ok(FiniteTruthTableResult::NoCountermodelFound {
                    assignments_checked,
                }) => println!(
                    "finite_truth_table=no_countermodel assignments_checked={}",
                    assignments_checked
                ),
                Ok(FiniteTruthTableResult::AtomBudgetExceeded {
                    atom_count,
                    max_atoms,
                }) => println!(
                    "finite_truth_table=unknown atom_count={} max_atoms={}",
                    atom_count, max_atoms
                ),
                Err(error) => {
                    eprintln!("finite truth-table check rejected: {error}");
                    return ExitCode::from(1);
                }
            }
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
