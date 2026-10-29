use gcrb_evaluator::trace_audit::audit_trace_bytes;
use gcrb_evaluator::{evaluate_public_development, generate_parameterized_controls, validate_combined_campaign_bytes, validate_cross_system_evidence_bundle_bytes, validate_fixture_bytes};
use std::env;
use std::fs;
use std::process::ExitCode;

fn usage() {
    eprintln!(
        "GCRB public-development evaluator\n\n\
         Usage:\n\
         cargo run --manifest-path research/gcrb-evaluator/Cargo.toml -- validate-fixture <fixture.json>\n\
         cargo run --manifest-path research/gcrb-evaluator/Cargo.toml -- validate-campaign <source-fixture.json> <campaign.json>\n\
         cargo run --manifest-path research/gcrb-evaluator/Cargo.toml -- validate-evidence-bundle <evidence-bundle.json>\n\
         cargo run --manifest-path research/gcrb-evaluator/Cargo.toml -- generate-controls <fixture.json> <count> <master-seed> <output.json>\n\
         cargo run --manifest-path research/gcrb-evaluator/Cargo.toml -- evaluate <fixture.json> <candidate-report.json> [report-output.json]\n\
         cargo run --manifest-path research/gcrb-evaluator/Cargo.toml -- trace-audit <fixture.json> <trace.json> [audit-report.json]\n\n\
         Generated controls are public known answers; trace input is unauthenticated unless verified externally."
    );
}

fn read_file(path: &str) -> Result<Vec<u8>, String> {
    fs::read(path).map_err(|error| format!("cannot read {path}: {error}"))
}

fn print_json<T: serde::Serialize>(value: &T) -> Result<(), String> {
    let output = serde_json::to_string_pretty(value)
        .map_err(|error| format!("cannot serialize JSON output: {error}"))?;
    println!("{output}");
    Ok(())
}

fn run() -> Result<(), String> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    match args.as_slice() {
        [command, fixture_path] if command == "validate-fixture" => {
            let fixture_bytes = read_file(fixture_path)?;
            let summary = validate_fixture_bytes(&fixture_bytes)?;
            print_json(&summary)
        }

        [command, fixture_path, campaign_path] if command == "validate-campaign" => {
            let fixture_bytes = read_file(fixture_path)?;
            let campaign_bytes = read_file(campaign_path)?;
            let summary = validate_combined_campaign_bytes(&fixture_bytes, &campaign_bytes)?;
            print_json(&summary)
        }

        [command, fixture_path, count_text, seed_text, output_path]
            if command == "generate-controls" =>
        {
            let fixture_bytes = read_file(fixture_path)?;
            let count = count_text
                .parse::<usize>()
                .map_err(|error| format!("invalid control count: {error}"))?;
            let seed = seed_text
                .parse::<u64>()
                .map_err(|error| format!("invalid master seed: {error}"))?;
            let generated = generate_parameterized_controls(&fixture_bytes, count, seed)?;
            let summary = validate_fixture_bytes(&generated)?;
            fs::write(output_path, &generated)
                .map_err(|error| format!("cannot write {output_path}: {error}"))?;
            eprintln!(
                "wrote {} public synthetic cases ({} attacks, {} negative controls) to {}; output_sha256={}; qualification_eligible=false",
                summary.case_count,
                summary.attack_count,
                summary.negative_control_count,
                output_path,
                summary.fixture_sha256
            );
            Ok(())
        }
        [command, bundle_path] if command == "validate-evidence-bundle" => {
            let bundle_bytes = read_file(bundle_path)?;
            let summary = validate_cross_system_evidence_bundle_bytes(&bundle_bytes)?;
            print_json(&summary)
        }
        [command, fixture_path, trace_path] if command == "trace-audit" => {
            let fixture_bytes = read_file(fixture_path)?;
            let trace_bytes = read_file(trace_path)?;
            let report = audit_trace_bytes(&fixture_bytes, &trace_bytes)?;
            print_json(&report)
        }
        [command, fixture_path, trace_path, output_path] if command == "trace-audit" => {
            let fixture_bytes = read_file(fixture_path)?;
            let trace_bytes = read_file(trace_path)?;
            let report = audit_trace_bytes(&fixture_bytes, &trace_bytes)?;
            let output = serde_json::to_string_pretty(&report)
                .map_err(|error| format!("cannot serialize trace audit report: {error}"))?;
            fs::write(output_path, format!("{output}\n"))
                .map_err(|error| format!("cannot write {output_path}: {error}"))?;
            eprintln!(
                "wrote trace audit report to {}; findings={}; complete_capture_coverage={}; qualification_eligible=false",
                output_path,
                report.findings.len(),
                report.complete_capture_coverage
            );
            Ok(())
        }
        [command, fixture_path, candidate_path] if command == "evaluate" => {
            let fixture_bytes = read_file(fixture_path)?;
            let candidate_bytes = read_file(candidate_path)?;
            let report = evaluate_public_development(&fixture_bytes, &candidate_bytes)?;
            print_json(&report)
        }
        [command, fixture_path, candidate_path, output_path] if command == "evaluate" => {
            let fixture_bytes = read_file(fixture_path)?;
            let candidate_bytes = read_file(candidate_path)?;
            let report = evaluate_public_development(&fixture_bytes, &candidate_bytes)?;
            let output = serde_json::to_string_pretty(&report)
                .map_err(|error| format!("cannot serialize evaluation report: {error}"))?;
            fs::write(output_path, format!("{output}\n"))
                .map_err(|error| format!("cannot write {output_path}: {error}"))?;
            eprintln!("wrote public-development evaluation report to {output_path}");
            Ok(())
        }
        _ => {
            usage();
            Err("invalid command-line arguments".to_owned())
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("gcrb-evaluator: {error}");
            ExitCode::from(2)
        }
    }
}
