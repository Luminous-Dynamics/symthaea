use proof_plan_contract::checker_diversity::validate_json_inputs;
use std::env;
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(registry_path) = args.next() else {
        eprintln!("usage: lean-checker-diversity-contract <registry.json> <run-profile.json>");
        return ExitCode::from(2);
    };
    let Some(profile_path) = args.next() else {
        eprintln!("usage: lean-checker-diversity-contract <registry.json> <run-profile.json>");
        return ExitCode::from(2);
    };
    if args.next().is_some() {
        eprintln!("usage: lean-checker-diversity-contract <registry.json> <run-profile.json>");
        return ExitCode::from(2);
    }

    let registry = match fs::read_to_string(&registry_path) {
        Ok(input) => input,
        Err(error) => {
            eprintln!("could not read {registry_path}: {error}");
            return ExitCode::from(2);
        }
    };
    let profile = match fs::read_to_string(&profile_path) {
        Ok(input) => input,
        Err(error) => {
            eprintln!("could not read {profile_path}: {error}");
            return ExitCode::from(2);
        }
    };

    match validate_json_inputs(&registry, &profile) {
        Ok(summary) => {
            println!("LEAN_CHECKER_DIVERSITY_PROFILE_STRUCTURALLY_VALID");
            println!("requested_checkers={}", summary.requested_count);
            println!("executed_checkers={}", summary.executed_count);
            println!("accepted_checkers={}", summary.accepted_count);
            println!("distinct_implementation_families={}", summary.distinct_implementation_families);
            println!("distinct_algorithm_lineages={}", summary.distinct_algorithm_lineages);
            println!("independent_external_algorithm_lineages={}", summary.independent_external_algorithm_lineages);
            println!("distinct_languages={}", summary.distinct_languages);
            println!("known_term_representations={}", summary.known_term_representation_families);
            println!("known_parser_families={}", summary.known_parser_families);
            println!("unknown_pairwise_relationships={}", summary.unknown_relationship_count);
            println!("claim_ceiling=registry_and_run_record_consistency_only; no_kernel_verdict");
            ExitCode::SUCCESS
        }
        Err(errors) => {
            eprintln!("LEAN_CHECKER_DIVERSITY_PROFILE_INVALID");
            for error in errors {
                eprintln!("- {error}");
            }
            eprintln!("claim_ceiling=no_checker_execution_or_theorem_validity_verdict");
            ExitCode::from(1)
        }
    }
}
