use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use mel_judge_grounding_audit::collab_trace::parse_and_audit;

fn run() -> Result<bool, String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 3 || args[0] != "audit" {
        return Err("usage: mel-collab-trace-audit audit TRACE.json REPORT.json".to_owned());
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
        "Audited {} events for session {}; trace contract {}",
        report.event_count,
        report.session_id,
        if report.qualified_for_declared_trace_contract {
            "qualified"
        } else {
            "not qualified"
        }
    );
    for finding in &report.qualification_findings {
        eprintln!("qualification finding: {finding}");
    }
    Ok(report.qualified_for_declared_trace_contract)
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("mel-collab-trace-audit: {error}");
            ExitCode::FAILURE
        }
    }
}
