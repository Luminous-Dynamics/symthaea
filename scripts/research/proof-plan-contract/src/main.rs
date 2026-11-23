use std::env;
use std::fs;
use std::process::ExitCode;

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
            println!("PROOF_PLAN_STRUCTURALLY_VALID plan_id={}", plan.plan_id);
            println!("claim_ceiling=structure_and_reference_integrity_only");
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
