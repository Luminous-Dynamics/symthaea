use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use mel_judge_grounding_audit::musical_timing::parse_and_score;

fn run() -> Result<bool, String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 3 || args[0] != "score" {
        return Err("usage: mel-musical-timing-audit score PANEL.json REPORT.json".to_owned());
    }
    let input = fs::read_to_string(&args[1])
        .map_err(|error| format!("failed to read {}: {error}", args[1]))?;
    let report = parse_and_score(&input)?;
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
        "Scored {} observed of {} target events; {} extra; timing contract {}",
        report.observed_count,
        report.target_count,
        report.extra_onset_count,
        if report.qualified_for_declared_timing_contract {
            "qualified"
        } else {
            "not qualified"
        }
    );
    for finding in &report.qualification_findings {
        eprintln!("qualification finding: {finding}");
    }
    Ok(report.qualified_for_declared_timing_contract)
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("mel-musical-timing-audit: {error}");
            ExitCode::FAILURE
        }
    }
}
