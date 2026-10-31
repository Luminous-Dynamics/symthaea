use std::env;
use std::process::ExitCode;

use connectome_perturbation_oracle::reference_suite;

fn main() -> ExitCode {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let mode = args.first().map(String::as_str).unwrap_or("--json");

    if args.len() > 1 || !matches!(mode, "--json" | "--text" | "--help") {
        eprintln!("usage: connectome-perturbation-oracle [--json|--text|--help]");
        return ExitCode::from(2);
    }

    if mode == "--help" {
        println!(
            "connectome-perturbation-oracle\n\
             Deterministic synthetic known-answer suite; not a biological simulator.\n\n\
             USAGE:\n    connectome-perturbation-oracle [--json|--text|--help]\n\n\
             DEFAULT:\n    --json"
        );
        return ExitCode::SUCCESS;
    }

    let cases = match reference_suite() {
        Ok(cases) => cases,
        Err(error) => {
            eprintln!("oracle failed closed: {error}");
            return ExitCode::FAILURE;
        }
    };

    if mode == "--json" {
        match connectome_perturbation_oracle::reference_suite_json() {
            Ok(json) => print!("{json}"),
            Err(error) => {
                eprintln!("oracle report failed closed: {error}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        println!("Connectome perturbation synthetic oracle");
        println!("Evidence class: SYNTHETIC_KNOWN_ANSWER");
        println!("Claim ceiling: SYNTHETIC_ONLY");
        for case in &cases {
            let status = if case.passed() { "PASS" } else { "FAIL" };
            println!(
                "{:42} target={} rmse={:.12} status={}",
                case.id, case.target_node, case.target_rmse, status
            );
        }
    }

    if cases.iter().all(|case| case.passed()) {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
