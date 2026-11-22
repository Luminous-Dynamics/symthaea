use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use mel_judge_grounding_audit::event_universe::parse_and_score;

fn run() -> Result<bool, String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 3 || args[0] != "score" {
        return Err("usage: mel-event-universe-audit score PANEL.json REPORT.json".to_owned());
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
        "Scored {} of {} independently frozen opportunity slots; {} candidate predictions omitted; panel structure {}",
        report.scored_slot_count,
        report.opportunity_slot_count,
        report.omitted_slot_count,
        if report.structurally_valid_panel { "valid" } else { "invalid" }
    );
    println!(
        "Presence Brier: {:?}; categorical Brier: {:?}; categorical log loss: {:?}; ECE: {:?}",
        report.presence_brier_score,
        report.categorical_brier_score,
        report.categorical_log_loss,
        report.presence_expected_calibration_error
    );
    Ok(report.structurally_valid_panel)
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("mel-event-universe-audit: {error}");
            ExitCode::FAILURE
        }
    }
}
