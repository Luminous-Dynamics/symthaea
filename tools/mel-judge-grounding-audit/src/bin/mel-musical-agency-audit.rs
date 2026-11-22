use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use mel_judge_grounding_audit::agency::parse_and_audit;

fn run() -> Result<bool, String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 3 || args[0] != "audit" {
        return Err("usage: mel-musical-agency-audit audit PANEL.json REPORT.json".to_owned());
    }
    let input = fs::read_to_string(&args[1])
        .map_err(|error| format!("failed to read {}: {error}", args[1]))?;
    let report = parse_and_audit(&input)?;
    let output = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("failed to serialize report: {error}"))?;
    if let Some(parent) = Path::new(&args[2]).parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create report directory: {error}"))?;
    }
    fs::write(&args[2], format!("{output}\n"))
        .map_err(|error| format!("failed to write {}: {error}", args[2]))?;
    println!(
        "Audited {} musical proposals; decision {}; agency trace {}",
        report.proposal_count,
        report.decision_action,
        if report.structurally_valid_agency_trace {
            "valid"
        } else {
            "invalid"
        }
    );
    for finding in &report.qualification_findings {
        eprintln!("agency finding: {finding}");
    }
    Ok(report.structurally_valid_agency_trace)
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("mel-musical-agency-audit: {error}");
            ExitCode::FAILURE
        }
    }
}
