use std::fs;
use std::path::PathBuf;

use symthaea_rea_evidence_verifier::{
    differential::{score_case, Observation, OracleCase},
    verify_bundle,
};

fn usage() -> &'static str {
    "Usage:
  symthaea-rea-evidence-verifier --artifact-dir DIR --source-root DIR --expected-head 40_HEX_SHA [--expected-receipt-sha256 SHA256]
  symthaea-rea-evidence-verifier score-case --oracle ORACLE.json --observation OBSERVATION.json

The default mode verifies a downloaded Unblob provider-evidence artifact.
score-case compares one provider-neutral observation against an independent oracle.
Both modes emit JSON; score-case exits non-zero unless the observation earns a full pass."
}

fn main() {
    if let Err(message) = run() {
        eprintln!("verification failed: {message}");
        std::process::exit(2);
    }
}

fn run_score_case(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let mut oracle_path = None;
    let mut observation_path = None;
    while let Some(arg) = args.next() {
        let value = args.next().ok_or_else(|| usage().to_owned())?;
        match arg.as_str() {
            "--oracle" if oracle_path.is_none() => oracle_path = Some(PathBuf::from(value)),
            "--observation" if observation_path.is_none() => observation_path = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown or duplicate score-case argument: {arg}\n{}", usage())),
        }
    }
    let oracle_path = oracle_path.ok_or_else(|| format!("missing --oracle\n{}", usage()))?;
    let observation_path = observation_path.ok_or_else(|| format!("missing --observation\n{}", usage()))?;
    let oracle: OracleCase = serde_json::from_slice(
        &fs::read(&oracle_path).map_err(|e| format!("cannot read oracle {}: {e}", oracle_path.display()))?,
    )
    .map_err(|e| format!("oracle JSON is invalid: {e}"))?;
    let observation: Observation = serde_json::from_slice(
        &fs::read(&observation_path).map_err(|e| format!("cannot read observation {}: {e}", observation_path.display()))?,
    )
    .map_err(|e| format!("observation JSON is invalid: {e}"))?;
    let score = score_case(&oracle, &observation).map_err(|e| e.to_string())?;
    let status = score.status.clone();
    println!("{}", serde_json::to_string_pretty(&score).map_err(|e| e.to_string())?);
    if status != "pass" {
        return Err(format!("score status is {status}; only an exact known-answer pass returns exit status 0"));
    }
    Ok(())
}

fn run_verify(args: impl Iterator<Item = String>) -> Result<(), String> {
    let mut artifact = None;
    let mut source = None;
    let mut head = None;
    let mut receipt_sha = None;
    let mut args = args;
    while let Some(arg) = args.next() {
        let value = args.next().ok_or_else(|| usage().to_owned())?;
        match arg.as_str() {
            "--artifact-dir" if artifact.is_none() => artifact = Some(PathBuf::from(value)),
            "--source-root" if source.is_none() => source = Some(PathBuf::from(value)),
            "--expected-head" if head.is_none() => head = Some(value),
            "--expected-receipt-sha256" if receipt_sha.is_none() => receipt_sha = Some(value),
            _ => return Err(format!("unknown or duplicate argument: {arg}\n{}", usage())),
        }
    }
    let artifact = artifact.ok_or_else(|| format!("missing --artifact-dir\n{}", usage()))?;
    let source = source.ok_or_else(|| format!("missing --source-root\n{}", usage()))?;
    let head = head.ok_or_else(|| format!("missing --expected-head\n{}", usage()))?;
    let verified = verify_bundle(&artifact, &source, &head, receipt_sha.as_deref())
        .map_err(|e| e.to_string())?;
    println!("{}", serde_json::to_string_pretty(&verified).map_err(|e| e.to_string())?);
    Ok(())
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let first = args.next().ok_or_else(|| usage().to_owned())?;
    if first == "score-case" {
        run_score_case(args)
    } else {
        run_verify(std::iter::once(first).chain(args))
    }
}
