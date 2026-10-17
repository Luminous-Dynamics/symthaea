// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Independent checker for candidate dispositions against the frozen synthetic
// chemical-process qualification corpus. This checks benchmark contracts only;
// it is not a chemistry solver, process simulator, safety assessor, or authority
// to qualify or operate a physical process.

use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    error::Error,
    fs,
    process::ExitCode,
};

const ALLOWED_DISPOSITIONS: &[&str] = &[
    "ACCEPTED_WITHIN_SCOPE",
    "BLOCKED",
    "UNRESOLVED",
    "OUT_OF_DOMAIN",
];

const USAGE: &str = "Usage: chemical_process_benchmark_evaluate --corpus <path> --results <path> --expected-head <git-commit-sha>";

struct Args {
    corpus_path: String,
    results_path: String,
    expected_head: String,
}

fn parse_args() -> Result<Args, String> {
    let mut corpus_path = None;
    let mut results_path = None;
    let mut expected_head = None;
    let mut args = env::args().skip(1);

    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}\n{USAGE}"))?;
        match flag.as_str() {
            "--corpus" => corpus_path = Some(value),
            "--results" => results_path = Some(value),
            "--expected-head" => expected_head = Some(value),
            _ => return Err(format!("unknown argument {flag:?}\n{USAGE}")),
        }
    }

    let expected_head = expected_head.ok_or_else(|| format!("--expected-head is required\n{USAGE}"))?;
    let is_hex_sha = matches!(expected_head.len(), 40 | 64)
        && expected_head.chars().all(|character| character.is_ascii_hexdigit());
    if !is_hex_sha {
        return Err(format!("expected head must be a 40- or 64-character hexadecimal commit ID\n{USAGE}"));
    }

    Ok(Args {
        corpus_path: corpus_path.ok_or_else(|| format!("--corpus is required\n{USAGE}"))?,
        results_path: results_path.ok_or_else(|| format!("--results is required\n{USAGE}"))?,
        expected_head,
    })
}

fn read_json(path: &str) -> Result<Value, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let parsed = serde_json::from_slice(&bytes)?;
    Ok(parsed)
}

fn text<'a>(value: &'a Value, field: &str) -> Option<&'a str> {
    value.get(field).and_then(Value::as_str)
}

fn strings(value: &Value, field: &str) -> Option<BTreeSet<String>> {
    value
        .get(field)?
        .as_array()?
        .iter()
        .map(|item| item.as_str().map(ToOwned::to_owned))
        .collect()
}

fn evaluate(corpus: &Value, results: &Value, expected_head: &str) -> Value {
    let mut global_errors = Vec::<String>::new();

    let benchmark_id = text(corpus, "benchmark_id").unwrap_or_default();
    let corpus_version = text(corpus, "corpus_version").unwrap_or_default();

    if benchmark_id.is_empty() || corpus_version.is_empty() {
        global_errors.push("corpus is missing benchmark_id or corpus_version".to_owned());
    }
    if text(results, "benchmark_id") != Some(benchmark_id) {
        global_errors.push("results benchmark_id does not match corpus".to_owned());
    }
    if text(results, "corpus_version") != Some(corpus_version) {
        global_errors.push("results corpus_version does not match corpus".to_owned());
    }

    let reported_head = text(results, "evaluated_head").unwrap_or_default();
    if reported_head != expected_head {
        global_errors.push("results evaluated_head does not match externally supplied --expected-head".to_owned());
    }

    let expected_cases = match corpus.get("cases").and_then(Value::as_array) {
        Some(cases) => cases,
        None => {
            global_errors.push("corpus cases field is missing or not an array".to_owned());
            return report(corpus, expected_head, reported_head, global_errors, Vec::new(), 0);
        }
    };
    let observed_cases = match results.get("cases").and_then(Value::as_array) {
        Some(cases) => cases,
        None => {
            global_errors.push("results cases field is missing or not an array".to_owned());
            return report(
                corpus,
                expected_head,
                reported_head,
                global_errors,
                expected_cases
                    .iter()
                    .map(|case| {
                        let id = text(case, "id").unwrap_or("<missing-id>");
                        json!({
                            "id": id,
                            "passed": false,
                            "errors": ["no case results were provided"]
                        })
                    })
                    .collect(),
                0,
            );
        }
    };

    let mut observed_by_id = BTreeMap::<String, Vec<&Value>>::new();
    for observed in observed_cases {
        match text(observed, "id") {
            Some(id) if !id.trim().is_empty() => {
                observed_by_id.entry(id.to_owned()).or_default().push(observed);
            }
            _ => global_errors.push("result case has a missing or empty id".to_owned()),
        }
    }

    let expected_ids = expected_cases
        .iter()
        .filter_map(|case| text(case, "id").map(ToOwned::to_owned))
        .collect::<BTreeSet<_>>();
    for id in observed_by_id.keys() {
        if !expected_ids.contains(id) {
            global_errors.push(format!("unexpected result case id {id:?}"));
        }
        if observed_by_id.get(id).is_some_and(|matches| matches.len() != 1) {
            global_errors.push(format!("duplicate result case id {id:?}"));
        }
    }

    let mut case_reports = Vec::with_capacity(expected_cases.len());
    for expected in expected_cases {
        let id = text(expected, "id").unwrap_or("<missing-id>");
        let expected_disposition = text(expected, "expected_disposition").unwrap_or_default();
        let mut errors = Vec::<String>::new();

        if !ALLOWED_DISPOSITIONS.contains(&expected_disposition) {
            errors.push(format!("corpus contains unknown expected disposition {expected_disposition:?}"));
        }

        let actual = observed_by_id.get(id).filter(|matches| matches.len() == 1);
        let Some(actual) = actual.and_then(|matches| matches.first()).copied() else {
            errors.push("result missing or duplicated".to_owned());
            case_reports.push(json!({
                "id": id,
                "expected_disposition": expected_disposition,
                "observed_disposition": Value::Null,
                "passed": false,
                "errors": errors
            }));
            continue;
        };

        let observed_disposition = text(actual, "observed_disposition").unwrap_or_default();
        if !ALLOWED_DISPOSITIONS.contains(&observed_disposition) {
            errors.push(format!("unknown observed disposition {observed_disposition:?}"));
        }
        if observed_disposition != expected_disposition {
            errors.push("observed disposition does not match frozen expectation".to_owned());
        }

        let required = strings(expected, "required_findings").unwrap_or_default();
        let observed_findings = match strings(actual, "observed_findings") {
            Some(findings) => findings,
            None => {
                errors.push("observed_findings missing or malformed".to_owned());
                BTreeSet::new()
            }
        };
        let missing_findings = required
            .difference(&observed_findings)
            .cloned()
            .collect::<Vec<_>>();
        if !missing_findings.is_empty() {
            errors.push("one or more required findings are missing".to_owned());
        }

        let prohibited = strings(expected, "prohibited_promotions").unwrap_or_default();
        let observed_promotions = match strings(actual, "observed_promotions") {
            Some(promotions) => promotions,
            None => {
                errors.push("observed_promotions missing or malformed".to_owned());
                BTreeSet::new()
            }
        };
        let violations = prohibited
            .intersection(&observed_promotions)
            .cloned()
            .collect::<Vec<_>>();
        if !violations.is_empty() {
            errors.push("prohibited evidence promotion detected".to_owned());
        }

        let passed = errors.is_empty();
        case_reports.push(json!({
            "id": id,
            "expected_disposition": expected_disposition,
            "observed_disposition": observed_disposition,
            "missing_required_findings": missing_findings,
            "prohibited_promotions_observed": violations,
            "passed": passed,
            "errors": errors
        }));
    }

    report(
        corpus,
        expected_head,
        reported_head,
        global_errors,
        case_reports,
        observed_cases.len(),
    )
}

fn report(
    corpus: &Value,
    expected_head: &str,
    reported_head: &str,
    global_errors: Vec<String>,
    cases: Vec<Value>,
    observed_count: usize,
) -> Value {
    let case_count = cases.len();
    let passed_count = cases
        .iter()
        .filter(|case| case.get("passed").and_then(Value::as_bool) == Some(true))
        .count();
    let failed_count = case_count.saturating_sub(passed_count);
    let no_global_errors = global_errors.is_empty();
    let passed = no_global_errors && failed_count == 0 && observed_count == case_count;

    json!({
        "report_schema_version": "1.0.0",
        "report_kind": "synthetic_chemical_process_contract_evaluation",
        "benchmark_id": text(corpus, "benchmark_id").unwrap_or("UNKNOWN"),
        "corpus_version": text(corpus, "corpus_version").unwrap_or("UNKNOWN"),
        "expected_head": expected_head,
        "reported_head": reported_head,
        "expected_case_count": case_count,
        "observed_case_count": observed_count,
        "passed_cases": passed_count,
        "failed_cases": failed_count,
        "global_errors": global_errors,
        "passed": passed,
        "claim_ceiling": "This result checks only agreement with the frozen synthetic benchmark contract. It does not establish chemistry, process-model accuracy, product grade, safety, economics, plant qualification, or physical operating authority.",
        "cases": cases
    })
}

fn run() -> Result<bool, Box<dyn Error>> {
    let args = parse_args().map_err(|message| std::io::Error::new(std::io::ErrorKind::InvalidInput, message))?;
    let corpus = read_json(&args.corpus_path)?;
    let results = read_json(&args.results_path)?;
    let report = evaluate(&corpus, &results, &args.expected_head);
    let passed = report.get("passed").and_then(Value::as_bool) == Some(true);
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(passed)
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(error) => {
            eprintln!("chemical-process benchmark evaluator: {error}");
            ExitCode::from(2)
        }
    }
}
