use std::env;
use std::fs;
use std::process::ExitCode;

use proof_plan_contract::hard_mode::{FiniteSemanticStatus, evaluate_receipt_json};

fn main() -> ExitCode {
    let Some(path) = env::args().nth(1) else {
        eprintln!("usage: hard-mode <HardModeReceiptV1.json>");
        return ExitCode::from(2);
    };
    let input = match fs::read_to_string(&path) {
        Ok(input) => input,
        Err(error) => {
            eprintln!("could not read {path}: {error}");
            return ExitCode::from(2);
        }
    };
    let report = match evaluate_receipt_json(&input) {
        Ok(report) => report,
        Err(errors) => {
            eprintln!("HARD_MODE_RECEIPT_INVALID");
            for error in errors {
                eprintln!("- {error}");
            }
            return ExitCode::from(1);
        }
    };

    println!("HARD_MODE_RECEIPT_VALID task_id={}", report.task_id);
    println!("task_variant={:?}", report.task_variant);
    println!(
        "answer_correct={}",
        report
            .answer_correct
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unknown".to_owned())
    );
    println!(
        "formalization_matches_target={}",
        report
            .formalization_matches_target
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unknown".to_owned())
    );
    match &report.finite_semantics {
        FiniteSemanticStatus::CountermodelFound {
            assignments_checked,
            witness,
        } => {
            let text = witness
                .iter()
                .map(|item| format!("{}={}", item.atom, item.value))
                .collect::<Vec<_>>()
                .join(",");
            println!(
                "finite_semantics=countermodel assignments_checked={} witness={}",
                assignments_checked, text
            );
        }
        FiniteSemanticStatus::NoCountermodelFound {
            assignments_checked,
        } => {
            println!(
                "finite_semantics=no_countermodel_found assignments_checked={}",
                assignments_checked
            );
        }
        FiniteSemanticStatus::UnknownAtomBudgetExceeded {
            atom_count,
            max_atoms,
        } => {
            println!(
                "finite_semantics=unknown atom_count={} max_atoms={}",
                atom_count, max_atoms
            );
        }
        FiniteSemanticStatus::NotApplicable => println!("finite_semantics=not_applicable"),
    }
    println!("candidate_witness={:?}", report.candidate_witness);
    println!("proof_status={:?}", report.proof_status);
    println!("claim_ceiling={}", report.claim_ceiling);
    ExitCode::SUCCESS
}
