use std::env;
use std::fs;
use std::process::ExitCode;

use proof_plan_contract::semantic_contract::{evaluate_receipt_json, ObligationStatus};

fn main() -> ExitCode {
    let Some(path) = env::args().nth(1) else {
        eprintln!("usage: theorem-sandwich <TheoremSandwichReceiptV1.json>");
        return ExitCode::from(2);
    };
    let input = match fs::read_to_string(&path) {
        Ok(input) => input,
        Err(error) => {
            eprintln!("could not read {path}: {error}");
            return ExitCode::from(2);
        }
    };
    let (receipt, report) = match evaluate_receipt_json(&input) {
        Ok(value) => value,
        Err(errors) => {
            eprintln!("THEOREM_SANDWICH_RECEIPT_INVALID");
            for error in errors { eprintln!("- {error}"); }
            return ExitCode::from(1);
        }
    };
    let count_pass = |items: &[proof_plan_contract::semantic_contract::ObligationResult]| {
        items.iter().filter(|item| matches!(
            &item.status,
            ObligationStatus::PassedFiniteTruthTable { .. }
        )).count()
    };
    println!("THEOREM_SANDWICH_RECEIPT_VALID task_id={}", receipt.candidate.task_id);
    println!("classification={}", report.classification.as_str());
    println!(
        "upper_passed={} upper_total={} lower_passed={} lower_total={}",
        count_pass(&report.upper_obligations),
        report.upper_obligations.len(),
        count_pass(&report.lower_obligations),
        report.lower_obligations.len()
    );
    println!("claim_ceiling={}", report.claim_ceiling);
    ExitCode::SUCCESS
}
