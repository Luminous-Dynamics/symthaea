//! Offline REA agent scorecard CLI.
//!
//! Usage: rea-agent-scorecard <hidden-oracle.json> <agent-submissions.json>

use rea_agent_scorecard::{MAX_ORACLE_BYTES, MAX_SUBMISSIONS_BYTES, evaluate_scorecard, parse_oracle_json, parse_submissions_json};
use std::fs::File;
use std::io::Read;
use std::path::Path;

fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>, String> {
    let file = File::open(path).map_err(|error| format!("cannot open '{}': {error}", path.display()))?;
    let size = file.metadata().map_err(|error| format!("cannot stat '{}': {error}", path.display()))?.len();
    if size > limit as u64 {
        return Err(format!("'{}' exceeds {limit} bytes", path.display()));
    }
    let mut bytes = Vec::new();
    file.take((limit as u64).saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| format!("cannot read '{}': {error}", path.display()))?;
    if bytes.len() > limit {
        return Err(format!("'{}' exceeds {limit} bytes", path.display()));
    }
    Ok(bytes)
}

fn run() -> Result<(), String> {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if arguments.len() != 2 {
        return Err("usage: rea-agent-scorecard <hidden-oracle.json> <agent-submissions.json>".to_owned());
    }
    let oracle_bytes = read_bounded(Path::new(&arguments[0]), MAX_ORACLE_BYTES)?;
    let submission_bytes = read_bounded(Path::new(&arguments[1]), MAX_SUBMISSIONS_BYTES)?;
    let oracle = parse_oracle_json(&oracle_bytes).map_err(|error| format!("oracle: {error}"))?;
    let submissions = parse_submissions_json(&submission_bytes).map_err(|error| format!("submissions: {error}"))?;
    let scorecard = evaluate_scorecard(&oracle, &submissions).map_err(|error| format!("scoring: {error}"))?;
    println!("{}", serde_json::to_string_pretty(&scorecard).map_err(|error| error.to_string())?);
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}
