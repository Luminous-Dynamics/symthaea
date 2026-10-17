// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Independent checker for candidate dispositions against the frozen synthetic
// chemical-process qualification corpus. This checks benchmark contracts only;
// it is not a chemistry solver, process simulator, safety assessor, or authority
// to qualify or operate a physical process.

use serde_json::{json, Value};
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

    let expected_head = expected_head
        .ok_or_else(|| format!("--expected-head is required\n{USAGE}"))?;
    let is_hex_sha = matches!(expected_head.len(), 40 | 64)
        && expected_head.chars().all(|character| character.is_ascii_hexdigit());
    if !is_hex_sha {
        return Err(format!(
            "expected head must be a 40- or 64-character hexadecimal commit ID\n{USAGE}"
        ));
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
        global_errors.push(
            "results evaluated_head does not match externally supplied --expected-head".to_owned(),
        );
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
            errors.push(format!(
                "corpus contains unknown expected disposition {expected_disposition:?}"
            ));
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

        let mut computed_numeric_checks = Value::Null;
        if text(expected, "class") == Some("analytic-reference") {
            let fixture = expected.get("fixture");
            let equation = fixture.and_then(|value| text(value, "equation"));
            let fresh = fixture.and_then(|value| value.get("fresh")).and_then(Value::as_f64);
            let recycle_fraction = fixture
                .and_then(|value| value.get("recycle_fraction"))
                .and_then(Value::as_f64);
            let analytic_solution = fixture
                .and_then(|value| value.get("analytic_solution"))
                .and_then(Value::as_f64);
            let observed_solution = actual
                .get("observed_numeric_values")
                .and_then(|value| value.get("solution"))
                .and_then(Value::as_f64);
            let tolerance = corpus
                .get("residual_profile")
                .and_then(|value| value.get("target_max"))
                .and_then(Value::as_f64);

            if equation != Some("x = fresh + recycle_fraction*x") {
                errors.push(
                    "analytic-reference fixture has no supported oracle equation".to_owned(),
                );
            } else if let (Some(fresh), Some(recycle_fraction), Some(analytic_solution),
                Some(observed_solution), Some(tolerance)) =
                (fresh, recycle_fraction, analytic_solution, observed_solution, tolerance)
            {
                if ![fresh, recycle_fraction, analytic_solution, observed_solution, tolerance]
                    .iter()
                    .all(|number| number.is_finite())
                    || tolerance <= 0.0
                {
                    errors.push(
                        "analytic-reference numeric values must be finite and tolerance positive"
                            .to_owned(),
                    );
                } else {
                    let rhs = fresh + recycle_fraction * observed_solution;
                    let residual_scale = 1.0_f64.max(
                        observed_solution.abs()
                            + fresh.abs()
                            + (recycle_fraction * observed_solution).abs(),
                    );
                    let normalized_residual = (observed_solution - rhs).abs() / residual_scale;
                    let solution_relative_error = (observed_solution - analytic_solution).abs()
                        / 1.0_f64.max(analytic_solution.abs());
                    let numeric_passed = normalized_residual <= tolerance
                        && solution_relative_error <= tolerance;
                    if normalized_residual > tolerance {
                        errors.push(
                            "independently calculated recycle-equation residual exceeds frozen tolerance"
                                .to_owned(),
                        );
                    }
                    if solution_relative_error > tolerance {
                        errors.push(
                            "observed solution differs from the frozen analytic solution".to_owned(),
                        );
                    }
                    computed_numeric_checks = json!({
                        "oracle": "linear_recycle_v1",
                        "observed_solution": observed_solution,
                        "analytic_solution": analytic_solution,
                        "normalized_equation_residual": normalized_residual,
                        "solution_relative_error": solution_relative_error,
                        "tolerance": tolerance,
                        "passed": numeric_passed
                    });
                }
            } else {
                errors.push(
                    "analytic-reference result is missing numeric values or tolerance".to_owned(),
                );
            }
        }

        let passed = errors.is_empty();
        case_reports.push(json!({
            "id": id,
            "expected_disposition": expected_disposition,
            "observed_disposition": observed_disposition,
            "missing_required_findings": missing_findings,
            "prohibited_promotions_observed": violations,
            "computed_numeric_checks": computed_numeric_checks,
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
        "claim_ceiling": concat!(
            "This result checks only agreement with the frozen synthetic benchmark contract. ",
            "It does not establish chemistry, process-model accuracy, product grade, safety, ",
            "economics, plant qualification, or physical operating authority."
        ),
        "cases": cases
    })
}

fn run() -> Result<bool, Box<dyn Error>> {
    let args = parse_args().map_err(|message| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, message)
    })?;
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


#[cfg(test)]
mod tests {
    use super::*;

    const HEAD: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn mini_corpus() -> Value {
        json!({
            "benchmark_id": "TEST-BENCH",
            "corpus_version": "1.0.0",
            "cases": [
                {
                    "id": "CASE-1",
                    "expected_disposition": "ACCEPTED_WITHIN_SCOPE",
                    "required_findings": ["balance_checked"],
                    "prohibited_promotions": ["plant_qualified"]
                },
                {
                    "id": "CASE-2",
                    "expected_disposition": "BLOCKED",
                    "required_findings": ["missing_utility_found"],
                    "prohibited_promotions": ["production_capability_complete"]
                }
            ]
        })
    }

    fn result_case(
        id: &str,
        disposition: &str,
        findings: &[&str],
        promotions: &[&str],
    ) -> Value {
        json!({
            "id": id,
            "observed_disposition": disposition,
            "observed_findings": findings,
            "observed_promotions": promotions
        })
    }

    fn good_results() -> Value {
        json!({
            "benchmark_id": "TEST-BENCH",
            "corpus_version": "1.0.0",
            "evaluated_head": HEAD,
            "cases": [
                result_case("CASE-1", "ACCEPTED_WITHIN_SCOPE", &["balance_checked"], &[]),
                result_case("CASE-2", "BLOCKED", &["missing_utility_found"], &[])
            ]
        })
    }

    #[test]
    fn matching_dispositions_and_findings_pass_contract() {
        let output = evaluate(&mini_corpus(), &good_results(), HEAD);
        assert_eq!(output["passed"], true);
        assert_eq!(output["passed_cases"], 2);
        assert_eq!(output["failed_cases"], 0);
    }

    #[test]
    fn disposition_mismatch_and_missing_findings_fail_closed() {
        let mut results = good_results();
        results["cases"][0]["observed_disposition"] = json!("BLOCKED");
        results["cases"][0]["observed_findings"] = json!([]);
        let output = evaluate(&mini_corpus(), &results, HEAD);
        assert_eq!(output["passed"], false);
        assert_eq!(output["cases"][0]["passed"], false);
        assert!(!output["cases"][0]["errors"].as_array().unwrap().is_empty());
    }

    #[test]
    fn prohibited_promotions_fail_even_when_disposition_matches() {
        let mut results = good_results();
        results["cases"][0]["observed_promotions"] = json!(["plant_qualified"]);
        let output = evaluate(&mini_corpus(), &results, HEAD);
        assert_eq!(output["passed"], false);
        assert_eq!(
            output["cases"][0]["prohibited_promotions_observed"],
            json!(["plant_qualified"])
        );
    }

    #[test]
    fn report_head_must_match_independently_supplied_head() {
        let mut results = good_results();
        results["evaluated_head"] = json!("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
        let output = evaluate(&mini_corpus(), &results, HEAD);
        assert_eq!(output["passed"], false);
        assert!(
            output["global_errors"]
                .as_array()
                .unwrap()
                .iter()
                .any(|error| error.as_str().unwrap().contains("expected-head"))
        );
    }

    #[test]
    fn duplicate_or_unknown_case_ids_fail_closed() {
        let mut results = good_results();
        results["cases"] = json!([
            result_case("CASE-1", "ACCEPTED_WITHIN_SCOPE", &["balance_checked"], &[]),
            result_case("CASE-1", "ACCEPTED_WITHIN_SCOPE", &["balance_checked"], &[]),
            result_case("UNEXPECTED", "BLOCKED", &["some_finding"], &[])
        ]);
        let output = evaluate(&mini_corpus(), &results, HEAD);
        assert_eq!(output["passed"], false);
        assert!(
            output["global_errors"]
                .as_array()
                .unwrap()
                .iter()
                .any(|error| error.as_str().unwrap().contains("duplicate result case id"))
        );
        assert!(
            output["global_errors"]
                .as_array()
                .unwrap()
                .iter()
                .any(|error| error.as_str().unwrap().contains("unexpected result case id"))
        );
    }

    #[test]
    fn skipped_required_case_is_not_treated_as_pass() {
        let mut results = good_results();
        results["cases"] = json!([
            result_case("CASE-1", "ACCEPTED_WITHIN_SCOPE", &["balance_checked"], &[])
        ]);
        let output = evaluate(&mini_corpus(), &results, HEAD);
        assert_eq!(output["passed"], false);
        assert_eq!(output["cases"][1]["passed"], false);
    }
}


    fn analytic_corpus() -> Value {
        json!({
            "benchmark_id": "ANALYTIC-TEST",
            "corpus_version": "1.0.0",
            "residual_profile": { "target_max": 1e-8 },
            "cases": [{
                "id": "RECYCLE-1",
                "class": "analytic-reference",
                "expected_disposition": "ACCEPTED_WITHIN_SCOPE",
                "required_findings": [
                    "solution_matches_analytic_reference",
                    "residual_within_frozen_tolerance"
                ],
                "prohibited_promotions": [],
                "fixture": {
                    "equation": "x = fresh + recycle_fraction*x",
                    "fresh": 1.0,
                    "recycle_fraction": 0.2,
                    "analytic_solution": 1.25
                }
            }]
        })
    }

    fn analytic_results(solution: f64) -> Value {
        json!({
            "benchmark_id": "ANALYTIC-TEST",
            "corpus_version": "1.0.0",
            "evaluated_head": HEAD,
            "cases": [{
                "id": "RECYCLE-1",
                "observed_disposition": "ACCEPTED_WITHIN_SCOPE",
                "observed_findings": [
                    "solution_matches_analytic_reference",
                    "residual_within_frozen_tolerance"
                ],
                "observed_promotions": [],
                "observed_numeric_values": { "solution": solution }
            }]
        })
    }

    #[test]
    fn analytical_recycle_oracle_recomputes_residual() {
        let output = evaluate(&analytic_corpus(), &analytic_results(1.25), HEAD);
        assert_eq!(output["passed"], true);
        assert_eq!(
            output["cases"][0]["computed_numeric_checks"]["normalized_equation_residual"],
            0.0
        );
        assert_eq!(output["cases"][0]["computed_numeric_checks"]["passed"], true);
    }

    #[test]
    fn analytical_recycle_oracle_rejects_a_plausible_but_wrong_solution() {
        let output = evaluate(&analytic_corpus(), &analytic_results(2.0), HEAD);
        assert_eq!(output["passed"], false);
        assert_eq!(output["cases"][0]["computed_numeric_checks"]["passed"], false);
    }
