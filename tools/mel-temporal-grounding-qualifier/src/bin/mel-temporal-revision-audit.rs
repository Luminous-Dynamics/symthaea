//! Command-line runner for deterministic temporal revision audits.
//!
//! Single-case input: one RevisionCaseV2 JSON object.
//! Matched experiment input: an array containing exactly one no-evidence,
//! one local-evidence and one yoked-evidence case per experiment_unit_id.

use std::env;
use std::fs;
use std::io::{self, Read};
use std::process::ExitCode;

use mel_temporal_grounding_qualifier::temporal_revision::{
    audit_revision_v2, compare_matched_evidence_arms, MatchedEvidenceContrastV1,
    RevisionCaseV2, RevisionReportV1,
};
use serde::Serialize;
use serde_json::Value;

#[derive(Serialize)]
struct BatchReportV1 {
    schema: &'static str,
    cases: usize,
    reports: Vec<RevisionReportV1>,
    matched_evidence_contrasts: Vec<MatchedEvidenceContrastV1>,
}

fn read_input() -> Result<String, String> {
    let mut args = env::args_os().skip(1);
    let Some(path) = args.next() else {
        return Err("usage: mel-temporal-revision-audit <case.json|->".into());
    };
    if args.next().is_some() {
        return Err("expected exactly one input path".into());
    }
    if path == "-" {
        let mut input = String::new();
        io::stdin()
            .read_to_string(&mut input)
            .map_err(|error| format!("failed reading stdin: {error}"))?;
        Ok(input)
    } else {
        fs::read_to_string(&path)
            .map_err(|error| format!("failed reading {}: {error}", path.to_string_lossy()))
    }
}

fn run() -> Result<String, String> {
    let input = read_input()?;
    let value: Value = serde_json::from_str(&input)
        .map_err(|error| format!("invalid JSON input: {error}"))?;
    if value.is_array() {
        let cases: Vec<RevisionCaseV2> = serde_json::from_value(value)
            .map_err(|error| format!("invalid revision case batch: {error}"))?;
        let reports = cases
            .iter()
            .map(audit_revision_v2)
            .collect::<Result<Vec<_>, _>>()?;
        let matched_evidence_contrasts = compare_matched_evidence_arms(&cases)?;
        let report = BatchReportV1 {
            schema: "mel-temporal-revision-batch-report-v1",
            cases: reports.len(),
            reports,
            matched_evidence_contrasts,
        };
        serde_json::to_string_pretty(&report)
            .map_err(|error| format!("failed serializing batch report: {error}"))
    } else {
        let case: RevisionCaseV2 = serde_json::from_value(value)
            .map_err(|error| format!("invalid revision case: {error}"))?;
        let report = audit_revision_v2(&case)?;
        serde_json::to_string_pretty(&report)
            .map_err(|error| format!("failed serializing revision report: {error}"))
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(output) => {
            println!("{output}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("mel-temporal-revision-audit: {error}");
            ExitCode::from(2)
        }
    }
}
