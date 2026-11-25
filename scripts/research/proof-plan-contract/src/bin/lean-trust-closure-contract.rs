use proof_plan_contract::trust_closure::validate_json;
use std::env;
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: lean-trust-closure-contract <capsule.json>");
        return ExitCode::from(2);
    };
    if args.next().is_some() {
        eprintln!("usage: lean-trust-closure-contract <capsule.json>");
        return ExitCode::from(2);
    }

    let input = match fs::read_to_string(&path) {
        Ok(input) => input,
        Err(error) => {
            eprintln!("could not read {path}: {error}");
            return ExitCode::from(2);
        }
    };

    match validate_json(&input) {
        Ok(capsule) => {
            let disposition = capsule
                .get("disposition")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown");
            println!(
                "LEAN_TRUST_CLOSURE_CAPSULE_SEMANTICALLY_CONSISTENT disposition={disposition}"
            );
            println!("claim_ceiling=assurance_record_internal_consistency_only");
            ExitCode::SUCCESS
        }
        Err(errors) => {
            eprintln!("LEAN_TRUST_CLOSURE_CAPSULE_SEMANTICALLY_INCONSISTENT");
            for error in errors {
                eprintln!("- {error}");
            }
            eprintln!("claim_ceiling=no_kernel_or_independent_checker_verdict");
            ExitCode::from(1)
        }
    }
}
