//! Independent inventory audit for the frozen validation-domain corpus.
//! This binary checks fixture structure only; it does not call the validator.

use std::collections::BTreeSet;
use std::process::ExitCode;

const CORPUS: &str = include_str!("../../fixtures/validation-domain-cases-v1.tsv");
const REQUIRED: [&str; 16] = [
    "same-facility-new-operating-point",
    "different-case-same-facility",
    "different-facility",
    "same-domain-scope",
    "calibration-episode-overlap",
    "calibration-dataset-reuse",
    "calibration-artifact-reuse",
    "holdout-used-for-training",
    "holdout-used-for-selection",
    "solver-output-posed-as-measurement",
    "missing-measurement-uncertainty",
    "geometry-outside-model-domain",
    "unsupported-intended-use",
    "stale-model-domain-revision",
    "holdout-not-withheld",
    "test-holdout-episode-overlap",
];

fn main() -> ExitCode {
    match audit() {
        Ok((accept, reject)) => {
            println!(
                "PASS: {} validation-domain fixtures; ACCEPT={accept}, REJECT={reject}",
                REQUIRED.len()
            );
            println!(
                "SCOPE: inventory integrity only; split validator and physical validity are not evaluated"
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("FAIL: {error}");
            ExitCode::FAILURE
        }
    }
}

fn audit() -> Result<(usize, usize), String> {
    let mut lines = CORPUS
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'));

    if lines.next() != Some("case_id|expected|mutation") {
        return Err("missing or invalid validation-domain fixture header".into());
    }

    let mut ids = BTreeSet::new();
    let mut accept = 0usize;
    let mut reject = 0usize;

    for line in lines {
        let fields: Vec<&str> = line.split('|').collect();
        if fields.len() != 3 {
            return Err(format!("expected exactly three fields: {line}"));
        }
        let [case_id, expected, mutation] = [fields[0], fields[1], fields[2]];
        if case_id.is_empty() || mutation.is_empty() || !ids.insert(case_id) {
            return Err(format!("empty or duplicate fixture identity: {line}"));
        }
        match expected {
            "ACCEPT" => accept += 1,
            "REJECT" => reject += 1,
            other => return Err(format!("unknown expected disposition {other}: {case_id}")),
        }
    }

    if ids.len() != REQUIRED.len() {
        return Err(format!(
            "expected {} fixtures, found {}",
            REQUIRED.len(),
            ids.len()
        ));
    }
    for required in REQUIRED {
        if !ids.contains(required) {
            return Err(format!("required fixture missing: {required}"));
        }
    }
    if accept != 3 || reject != 13 {
        return Err(format!(
            "unexpected disposition counts: ACCEPT={accept}, REJECT={reject}"
        ));
    }
    Ok((accept, reject))
}
