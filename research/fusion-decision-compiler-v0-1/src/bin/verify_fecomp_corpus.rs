//! Independent structural check for the frozen FECOMP-0 corpus.
//!
//! This executable does not call compile_candidate. It checks only corpus
//! inventory, syntax, unique identities and required negative-control labels.

use std::collections::BTreeSet;
use std::process::ExitCode;

const CORPUS: &str = include_str!("../../fixtures/fecomp-cases-v1.tsv");

const REQUIRED_CASES: [&str; 26] = [
    "complete-synthetic-candidate",
    "missing-service-boundary",
    "missing-model-capability",
    "out-of-domain-model",
    "unknown-applicability",
    "zero-exit-nonconverged",
    "nonzero-exit",
    "missing-observable",
    "missing-energy-gate",
    "failed-replacement-gate",
    "unknown-material-gate",
    "synthetic-promoted-to-measured",
    "stale-claim-revision",
    "correlated-solvers",
    "unknown-solver-dependence",
    "missing-model-run",
    "malformed-geometry-digest",
    "dangling-gate-evidence",
    "changed-material-invalidates-design",
    "dependency-cycle",
    "solver-stale-candidate-revision",
    "solver-stale-geometry",
    "unsupported-observable",
    "boundary-evidence-missing",
    "boundary-evidence-wrong-scope",
    "unsupported-model-use",
];

fn main() -> ExitCode {
    match audit() {
        Ok((accept, blocked, indeterminate, reject)) => {
            println!(
                "PASS: {} frozen rows; eligible={accept}, blocked={blocked}, indeterminate={indeterminate}, schema-reject={reject}",
                REQUIRED_CASES.len()
            );
            println!(
                "SCOPE: structural corpus audit only; compiler behavior and physical correctness are not evaluated"
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("FAIL: {error}");
            ExitCode::FAILURE
        }
    }
}

fn audit() -> Result<(usize, usize, usize, usize), String> {
    let mut lines = CORPUS
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'));

    if lines.next() != Some("case_id|expected|mutation") {
        return Err("unexpected or missing header".to_string());
    }

    let mut ids = BTreeSet::new();
    let mut counts = (0usize, 0usize, 0usize, 0usize);
    let mut rows = 0usize;

    for line in lines {
        let fields: Vec<&str> = line.split('|').collect();
        if fields.len() != 3 {
            return Err(format!("row must have exactly three fields: {line}"));
        }
        let [case_id, expected, mutation] = [fields[0], fields[1], fields[2]];
        if case_id.is_empty() || !ids.insert(case_id) {
            return Err(format!("empty or duplicate case identity: {case_id}"));
        }
        if mutation.is_empty() {
            return Err(format!("missing mutation label: {case_id}"));
        }

        match expected {
            "REVIEW" => counts.0 += 1,
            "BLOCKED" => counts.1 += 1,
            "INDETERMINATE" => counts.2 += 1,
            "REJECT" => counts.3 += 1,
            other => {
                return Err(format!(
                    "unknown expected disposition {other} for {case_id}"
                ));
            }
        }
        rows += 1;
    }

    if rows != REQUIRED_CASES.len() {
        return Err(format!(
            "expected {} rows, found {rows}",
            REQUIRED_CASES.len()
        ));
    }
    for required in REQUIRED_CASES {
        if !ids.contains(required) {
            return Err(format!("required case absent: {required}"));
        }
    }

    if counts != (5, 7, 6, 8) {
        return Err(format!(
            "unexpected expected-disposition counts: {counts:?}"
        ));
    }

    Ok(counts)
}
