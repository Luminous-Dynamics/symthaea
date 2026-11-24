use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use mel_judge_grounding_audit::closed_loop_episode::parse_and_reduce;

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 3 || args[0] != "audit" {
        return Err("usage: mel-closed-loop-episode-audit audit EPISODE.json REPORT.json".into());
    }
    let input = fs::read_to_string(&args[1])
        .map_err(|error| format!("failed to read {}: {error}", args[1]))?;
    let report = parse_and_reduce(&input)?;
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
        "Episode {} reduced: action={:?}, observation={:?}, structural_capture={}",
        report.episode_id,
        report.action_disposition,
        report.observation_outcome,
        report.capture_accounting_complete
    );
    for finding in &report.findings {
        eprintln!("finding: {finding}");
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("mel-closed-loop-episode-audit: {error}");
            ExitCode::FAILURE
        }
    }
}
