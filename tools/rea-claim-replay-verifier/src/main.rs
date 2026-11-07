//! Read-only CLI for independently replaying a serialized REA claim result.
//!
//! Usage:
//!   rea-claim-replay <claim.json> <result.json> [evidence-1.json ...]
//!
//! Exit codes: 0 = replay match, 1 = well-formed but semantically mismatched
//! result, 2 = invalid input, resource limit, IO or canonicalization failure.

use rea_claim_replay_verifier::{MAX_EVIDENCE_RECORDS, ReplayOutcome, replay_claim_evaluation};
use std::env;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use symthaea_rea_compat::MAX_EVIDENCE_JSON_BYTES;

const MAX_CLAIM_BYTES: usize = 1024 * 1024;
const MAX_RESULT_BYTES: usize = 4 * 1024 * 1024;
const MAX_TOTAL_EVIDENCE_BYTES: usize = 128 * 1024 * 1024;

fn read_bounded(path: &Path, max_bytes: usize, label: &str) -> Result<Vec<u8>, String> {
    let file = File::open(path)
        .map_err(|error| format!("cannot open {label} '{}': {error}", path.display()))?;
    let mut bytes = Vec::new();
    file.take((max_bytes as u64).saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| format!("cannot read {label} '{}': {error}", path.display()))?;
    if bytes.len() > max_bytes {
        return Err(format!(
            "{label} '{}' exceeds the {}-byte input limit",
            path.display(),
            max_bytes
        ));
    }
    Ok(bytes)
}

fn run() -> Result<i32, String> {
    let arguments: Vec<_> = env::args_os().skip(1).collect();
    if arguments.len() < 2 {
        return Err(
            "usage: rea-claim-replay <claim.json> <result.json> [evidence-1.json ...]".to_owned(),
        );
    }

    let evidence_count = arguments.len().saturating_sub(2);
    if evidence_count > MAX_EVIDENCE_RECORDS {
        return Err(format!(
            "received {evidence_count} Evidence paths; limit is {MAX_EVIDENCE_RECORDS}"
        ));
    }

    let claim_path = PathBuf::from(&arguments[0]);
    let result_path = PathBuf::from(&arguments[1]);
    let claim = read_bounded(&claim_path, MAX_CLAIM_BYTES, "claim request")?;
    let result = read_bounded(&result_path, MAX_RESULT_BYTES, "claim result")?;

    let mut total_evidence_bytes = 0usize;
    let mut evidence = Vec::with_capacity(arguments.len().saturating_sub(2));
    for (index, argument) in arguments.iter().skip(2).enumerate() {
        let path = PathBuf::from(argument);
        let remaining = MAX_TOTAL_EVIDENCE_BYTES
            .checked_sub(total_evidence_bytes)
            .ok_or_else(|| "total Evidence input limit exceeded".to_owned())?;
        let per_file_limit = MAX_EVIDENCE_JSON_BYTES.min(remaining);
        let bytes = read_bounded(
            &path,
            per_file_limit,
            &format!("Evidence record {}", index + 1),
        )?;
        total_evidence_bytes = total_evidence_bytes
            .checked_add(bytes.len())
            .ok_or_else(|| "total Evidence input size overflow".to_owned())?;
        evidence.push(bytes);
    }

    let evidence_refs: Vec<&[u8]> = evidence.iter().map(Vec::as_slice).collect();
    let receipt = replay_claim_evaluation(&claim, &evidence_refs, &result)
        .map_err(|error| format!("claim replay failed ({:?}): {error}", error.kind))?;
    let output = serde_json::to_string_pretty(&receipt)
        .map_err(|error| format!("cannot serialize replay receipt: {error}"))?;
    println!("{output}");

    Ok(match receipt.outcome {
        ReplayOutcome::ReplayMatch => 0,
        ReplayOutcome::ReplayMismatch => 1,
    })
}

fn main() {
    let exit_code = match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("{error}");
            2
        }
    };
    if exit_code != 0 {
        std::process::exit(exit_code);
    }
}
