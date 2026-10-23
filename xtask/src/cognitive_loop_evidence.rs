//! Structural and semantic validation for the cognitive-loop evidence benchmark manifest.
//!
//! This checks the benchmark contract, not model performance. Passing means that scenarios
//! are well-formed and evidence/run provenance requirements are present; it does not mean
//! the candidate system has passed any benchmark.

use anyhow::{Context, bail, ensure};
use serde_json::Value;
use std::{
    collections::BTreeSet,
    fs,
    path::Path,
};

#[derive(Debug, PartialEq, Eq)]
pub struct ManifestSummary {
    pub baseline_count: usize,
    pub suite_count: usize,
    pub case_count: usize,
}

const REQUIRED_BASELINES: [&str; 8] = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7"];
const REQUIRED_DRIFT_DETECTORS: [&str; 14] = [
    "fixed-threshold",
    "ewma-control-chart",
    "page-hinkley",
    "adwin",
    "kswin",
    "discriminative-drift-detector",
    "image-based-drift-detector",
    "semi-parametric-log-likelihood",
    "seed",
    "stepd",
    "abcd-feature-space",
    "abcd-error-stream",
    "studd-adwin-compression",
    "dynamic-threshold-dtd",
];
const REQUIRED_RESULT_FIELDS: [&str; 12] = [
    "repository_head_sha",
    "configuration_digest",
    "toolchain",
    "feature_flags",
    "hardware",
    "scenario_id",
    "seed",
    "baseline_id",
    "metric_version",
    "cycle_records_digest",
    "result_digest",
    "hard_gate_violations",
];

pub fn validate_manifest(path: &Path) -> anyhow::Result<ManifestSummary> {
    let bytes = fs::read(path).with_context(|| format!("reading manifest {}", path.display()))?;
    let value: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing JSON manifest {}", path.display()))?;
    validate_value(&value)
}

fn validate_value(root: &Value) -> anyhow::Result<ManifestSummary> {
    ensure!(root.is_object(), "manifest root must be a JSON object");
    ensure!(
        root.get("schema_version").and_then(Value::as_u64) == Some(3),
        "schema_version must be the integer 3"
    );
    ensure!(
        string_field(root, "benchmark", "manifest")? == "symthaea-cognitive-loop-evidence",
        "unexpected benchmark identifier"
    );
    ensure!(
        string_field(root, "status", "manifest")? == "proposal-not-measured",
        "manifest status must remain proposal-not-measured until measured results are recorded"
    );
    nonempty_string_array(root, "notes", "manifest")?;

    let baselines = array_field(root, "baselines", "manifest")?;
    ensure!(!baselines.is_empty(), "baselines must not be empty");
    let mut baseline_ids = BTreeSet::new();
    for (index, baseline) in baselines.iter().enumerate() {
        let context = format!("baselines[{index}]");
        let id = string_field(baseline, "id", &context)?;
        ensure!(baseline_ids.insert(id.to_owned()), "duplicate baseline id: {id}");
        nonempty_string_field(baseline, "name", &context)?;
        nonempty_string_field(baseline, "purpose", &context)?;
    }
    for expected in REQUIRED_BASELINES {
        ensure!(
            baseline_ids.contains(expected),
            "baseline matrix is missing required baseline {expected}"
        );
    }

    let drift_detectors = array_field(root, "external_drift_detectors", "manifest")?;
    ensure!(
        !drift_detectors.is_empty(),
        "external_drift_detectors must not be empty"
    );
    let mut detector_ids = BTreeSet::new();
    for (index, detector) in drift_detectors.iter().enumerate() {
        let context = format!("external_drift_detectors[{index}]");
        let id = string_field(detector, "id", &context)?;
        ensure!(
            detector_ids.insert(id.to_owned()),
            "duplicate external drift detector id: {id}"
        );
        nonempty_string_field(detector, "name", &context)?;
        nonempty_string_field(detector, "family", &context)?;
        nonempty_string_field(detector, "input_contract", &context)?;
        nonempty_string_field(detector, "implementation_status", &context)?;
        let source = nonempty_string_field(detector, "source", &context)?;
        ensure!(
            source.starts_with("https://"),
            "{context}.source must be an HTTPS reference"
        );
        let labels_required = detector
            .get("labels_required")
            .and_then(Value::as_bool)
            .with_context(|| format!("{context}.labels_required must be boolean"))?;
        let track = nonempty_string_field(detector, "comparison_track", &context)?;
        ensure!(
            ["valid-pe-only", "feature-only-unsupervised", "auxiliary-loss-unsupervised", "observed-outcome"].contains(&track),
            "{context}.comparison_track uses an unknown evidence-access track"
        );
        ensure!(
            labels_required == (track == "observed-outcome"),
            "{context}.labels_required conflicts with comparison_track"
        );
    }
    for expected in REQUIRED_DRIFT_DETECTORS {
        ensure!(
            detector_ids.contains(expected),
            "external drift detector track is missing required comparator {expected}"
        );
    }
    let detector_protocol = nonempty_string_field(
        root,
        "drift_detector_comparison_protocol",
        "manifest",
    )?;
    for required_phrase in [
        "disjoint calibration stream",
        "invalid/missing PE",
        "false alarms",
        "detection-delay",
        "not yet implemented or measured",
    ] {
        ensure!(
            detector_protocol.contains(required_phrase),
            "drift_detector_comparison_protocol must state: {required_phrase}"
        );
    }

    let groups = array_field(root, "scenario_groups", "manifest")?;
    ensure!(!groups.is_empty(), "scenario_groups must not be empty");
    let mut group_ids = BTreeSet::new();
    let mut case_ids = BTreeSet::new();
    let mut case_count = 0usize;
    let mut group_hard_gate = std::collections::BTreeMap::new();

    for (group_index, group) in groups.iter().enumerate() {
        let context = format!("scenario_groups[{group_index}]");
        let id = string_field(group, "id", &context)?;
        ensure!(group_ids.insert(id.to_owned()), "duplicate scenario group id: {id}");
        let hard_gate = group
            .get("hard_gate")
            .and_then(Value::as_bool)
            .with_context(|| format!("{context}.hard_gate must be boolean"))?;
        group_hard_gate.insert(id.to_owned(), hard_gate);

        let cases = array_field(group, "cases", &context)?;
        ensure!(!cases.is_empty(), "{context}.cases must not be empty");
        for (case_index, case) in cases.iter().enumerate() {
            let case_context = format!("{context}.cases[{case_index}]");
            let case_id = string_field(case, "id", &case_context)?;
            ensure!(
                case_ids.insert(case_id.to_owned()),
                "duplicate scenario case id across manifest: {case_id}"
            );
            nonempty_string_field(case, "inject", &case_context)?;

            if let Some(metrics) = case.get("metrics") {
                validate_nonempty_string_array(metrics, &format!("{case_context}.metrics"))?;
            }
            if let Some(assertions) = case.get("assertions") {
                validate_nonempty_string_array(assertions, &format!("{case_context}.assertions"))?;
            }
            if hard_gate {
                let assertions = array_field(case, "assertions", &case_context)?;
                ensure!(
                    !assertions.is_empty(),
                    "hard-gate scenario {case_id} must have at least one assertion"
                );
            }

            if id == "S0-validity" {
                validate_validity_case(case, &case_context)?;
            }
            case_count += 1;
        }
    }

    for required_gate in ["S0-validity", "S3-phase-causality"] {
        ensure!(
            group_hard_gate.get(required_gate) == Some(&true),
            "{required_gate} must exist and be marked as a hard gate"
        );
    }

    let result_contract = root
        .get("result_contract")
        .with_context(|| "manifest.result_contract is required")?;
    let required_fields = nonempty_string_array(result_contract, "required_fields", "result_contract")?;
    let mut field_set = BTreeSet::new();
    for field in &required_fields {
        ensure!(
            field_set.insert(field.as_str()),
            "duplicate result_contract.required_fields entry: {field}"
        );
    }
    for expected in REQUIRED_RESULT_FIELDS {
        ensure!(
            field_set.contains(expected),
            "result contract is missing required provenance field {expected}"
        );
    }
    ensure!(
        field_set.contains("per_seed_metrics"),
        "result contract is missing required provenance field per_seed_metrics"
    );
    let ci_policy = nonempty_string_field(result_contract, "ci_policy", "result_contract")?;
    ensure!(
        ci_policy.contains("skipped") && ci_policy.contains("not a pass"),
        "result_contract.ci_policy must explicitly state that skipped checks are not passes"
    );
    nonempty_string_field(result_contract, "claim_policy", "result_contract")?;

    let expected_shift_cases = [
        "class-prior-abrupt", "class-prior-gradual",
        "label-swap-abrupt", "label-swap-gradual",
        "feature-permutation-abrupt", "feature-permutation-gradual",
        "feature-filtering-abrupt", "feature-filtering-gradual",
    ];
    for expected in expected_shift_cases {
        ensure!(case_ids.contains(expected), "S1 drift matrix is missing required case {expected}");
    }

    let drift_protocol = root.get("drift_protocol").context("manifest.drift_protocol is required")?;
    ensure!(
        nonempty_string_field(drift_protocol, "protocol_id", "drift_protocol")? == "leave-one-dataset-out-monte-carlo-v1",
        "unexpected drift protocol id"
    );
    ensure!(
        drift_protocol.get("monte_carlo_seeds").and_then(Value::as_u64).is_some_and(|n| n >= 30),
        "drift_protocol.monte_carlo_seeds must be at least 30"
    );
    for field in ["drift_types", "transition_modes", "metrics", "research_extensions_not_yet_in_core_matrix"] {
        nonempty_string_array(drift_protocol, field, "drift_protocol")?;
    }
    let access_tracks = array_field(drift_protocol, "detector_access_tracks", "drift_protocol")?;
    let access_ids: BTreeSet<String> = access_tracks.iter().enumerate().map(|(i, track)| {
        string_field(track, "id", &format!("drift_protocol.detector_access_tracks[{i}]")).map(str::to_owned)
    }).collect::<anyhow::Result<_>>()?;
    for required_track in ["valid-pe-only", "feature-only-unsupervised", "auxiliary-loss-unsupervised", "observed-outcome"] {
        ensure!(access_ids.contains(required_track), "drift protocol is missing evidence access track {required_track}");
    }
    let matching_rule = nonempty_string_field(drift_protocol, "matching_rule", "drift_protocol")?;
    for phrase in ["earliest unused alarm", "false alarm", "missed", "invalid/missing PE"] {
        ensure!(matching_rule.contains(phrase), "drift_protocol.matching_rule must state: {phrase}");
    }
    nonempty_string_field(drift_protocol, "tuning_policy", "drift_protocol")?;
    nonempty_string_field(drift_protocol, "source_paper", "drift_protocol")?;
    nonempty_string_field(drift_protocol, "claims", "drift_protocol")?;

    Ok(ManifestSummary {
        baseline_count: baselines.len(),
        suite_count: groups.len(),
        case_count,
    })
}

fn validate_validity_case(case: &Value, context: &str) -> anyhow::Result<()> {
    let expected_valid = case
        .get("expected_valid")
        .with_context(|| format!("{context}.expected_valid is required for S0 validity cases"))?;
    let pe_update = case
        .get("expected_pe_training_update")
        .with_context(|| format!("{context}.expected_pe_training_update is required"))?;
    let replay = case
        .get("expected_replay_admission")
        .with_context(|| format!("{context}.expected_replay_admission is required"))?;

    match expected_valid {
        Value::Bool(false) => {
            ensure!(
                pe_update == &Value::Bool(false),
                "{context}: invalid evidence must not update PE-driven training"
            );
            ensure!(
                replay == &Value::Bool(false),
                "{context}: invalid evidence must not enter replay"
            );
        }
        Value::Bool(true) => {
            ensure!(
                pe_update == &Value::Bool(true),
                "{context}: valid evidence must explicitly exercise the PE training path"
            );
            ensure!(
                replay == &Value::Bool(true),
                "{context}: valid evidence must explicitly exercise replay admission"
            );
        }
        Value::String(policy) if policy == "per-comparison-contract" => {
            ensure!(
                pe_update.as_str() == Some("if-valid"),
                "{context}: conditional validity requires expected_pe_training_update='if-valid'"
            );
            ensure!(
                replay.as_str() == Some("if-valid"),
                "{context}: conditional validity requires expected_replay_admission='if-valid'"
            );
        }
        _ => bail!(
            "{context}.expected_valid must be boolean or 'per-comparison-contract'"
        ),
    }
    Ok(())
}

fn string_field<'a>(value: &'a Value, field: &str, context: &str) -> anyhow::Result<&'a str> {
    value
        .get(field)
        .and_then(Value::as_str)
        .with_context(|| format!("{context}.{field} must be a string"))
}

fn nonempty_string_field<'a>(
    value: &'a Value,
    field: &str,
    context: &str,
) -> anyhow::Result<&'a str> {
    let result = string_field(value, field, context)?;
    ensure!(
        !result.trim().is_empty(),
        "{context}.{field} must not be empty"
    );
    Ok(result)
}

fn array_field<'a>(value: &'a Value, field: &str, context: &str) -> anyhow::Result<&'a Vec<Value>> {
    value
        .get(field)
        .and_then(Value::as_array)
        .with_context(|| format!("{context}.{field} must be an array"))
}

fn nonempty_string_array(value: &Value, field: &str, context: &str) -> anyhow::Result<Vec<String>> {
    let items = array_field(value, field, context)?;
    validate_string_items(items, &format!("{context}.{field}"))?;
    Ok(items.iter().filter_map(Value::as_str).map(str::to_owned).collect())
}

fn validate_nonempty_string_array(value: &Value, context: &str) -> anyhow::Result<()> {
    let items = value
        .as_array()
        .with_context(|| format!("{context} must be an array"))?;
    ensure!(!items.is_empty(), "{context} must not be empty");
    validate_string_items(items, context)
}

fn validate_string_items(items: &[Value], context: &str) -> anyhow::Result<()> {
    ensure!(!items.is_empty(), "{context} must not be empty");
    for (index, item) in items.iter().enumerate() {
        ensure!(
            item.as_str().is_some_and(|s| !s.trim().is_empty()),
            "{context}[{index}] must be a non-empty string"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn valid_manifest() -> Value {
        let baselines = REQUIRED_BASELINES
            .iter()
            .map(|id| {
                json!({
                    "id": id,
                    "name": format!("baseline-{id}"),
                    "purpose": "test-only baseline"
                })
            })
            .collect::<Vec<_>>();
        json!({
            "schema_version": 3,
            "benchmark": "symthaea-cognitive-loop-evidence",
            "status": "proposal-not-measured",
            "notes": ["A hard invariant cannot be averaged away."],
            "baselines": baselines,
            "external_drift_detectors": REQUIRED_DRIFT_DETECTORS.iter().map(|id| json!({
                "id": id,
                "name": format!("detector-{id}"),
                "family": "fixture-family",
                "comparison_track": if ["seed", "stepd", "abcd-error-stream", "dynamic-threshold-dtd"].contains(id) { "observed-outcome" } else if *id == "studd-adwin-compression" { "auxiliary-loss-unsupervised" } else if ["discriminative-drift-detector", "image-based-drift-detector", "semi-parametric-log-likelihood", "abcd-feature-space"].contains(id) { "feature-only-unsupervised" } else { "valid-pe-only" },
                "labels_required": ["seed", "stepd", "abcd-error-stream", "dynamic-threshold-dtd"].contains(id),
                "input_contract": "Valid observations only; declared access contract applies.",
                "implementation_status": "planned-baseline",
                "source": "https://example.com/reference"
            })).collect::<Vec<_>>(),
            "drift_detector_comparison_protocol": "Pin versions; tune on a disjoint calibration stream; never remap invalid/missing PE; report false alarms and detection-delay; candidates are not yet implemented or measured.",
            "drift_protocol": {
                "protocol_id": "leave-one-dataset-out-monte-carlo-v1",
                "monte_carlo_seeds": 50,
                "drift_types": ["class-prior", "label-swap", "feature-permutation", "feature-filtering"],
                "transition_modes": ["abrupt", "gradual"],
                "metrics": ["event-f1", "false-alarms-per-1000-valid-observations"],
                "research_extensions_not_yet_in_core_matrix": ["incremental drift", "recurring drift"],
                "detector_access_tracks": [
                    {"id":"valid-pe-only","description":"PE only."},
                    {"id":"feature-only-unsupervised","description":"Features only."},
                    {"id":"auxiliary-loss-unsupervised","description":"Aux loss only."},
                    {"id":"observed-outcome","description":"Observed outcomes."}
                ],
                "matching_rule": "Match the earliest unused alarm in each event interval. Unmatched alarm is a false alarm; unmatched event is missed; invalid/missing PE cannot be observations.",
                "tuning_policy": "Tune on all but one dataset and hold that dataset out.",
                "source_paper": "https://doi.org/10.1145/3770855.3819070",
                "claims": "Comparator definitions are not measured results."
            },
            "scenario_groups": [
                {
                    "id": "S0-validity",
                    "hard_gate": true,
                    "cases": [
                        {
                            "id": "invalid-first-cycle",
                            "inject": "No prior prediction exists.",
                            "expected_valid": false,
                            "expected_pe_training_update": false,
                            "expected_replay_admission": false,
                            "assertions": ["PE history is unchanged."]
                        },
                        {
                            "id": "valid-identical",
                            "inject": "The prediction equals the observation.",
                            "expected_valid": true,
                            "expected_pe_training_update": true,
                            "expected_replay_admission": true,
                            "assertions": ["PE is near the metric minimum."]
                        },
                        {
                            "id": "conditional-norm",
                            "inject": "Magnitude depends on comparison metric.",
                            "expected_valid": "per-comparison-contract",
                            "expected_pe_training_update": "if-valid",
                            "expected_replay_admission": "if-valid",
                            "assertions": ["Norm tolerance is explicit."]
                        }
                    ]
                },
                {
                    "id": "S3-phase-causality",
                    "hard_gate": true,
                    "cases": [{
                        "id": "one-urgency-update",
                        "inject": "Run one perception cycle.",
                        "assertions": ["History advances at most once."]
                    }]
                }
            ],
            "result_contract": {
                "required_fields": [
                    "repository_head_sha", "configuration_digest", "toolchain",
                    "feature_flags", "hardware", "scenario_id", "seed", "baseline_id",
                    "metric_version", "cycle_records_digest", "result_digest",
                    "hard_gate_violations", "per_seed_metrics"
                ],
                "ci_policy": "queued/skipped/missing is not a pass",
                "claim_policy": "No SOTA claim without pinned comparators and raw records."
            }
        })
    }

    #[test]
    fn accepts_valid_manifest_and_counts_scenarios() {
        let summary = validate_value(&valid_manifest()).expect("fixture is valid");
        assert_eq!(summary.baseline_count, 8);
        assert_eq!(summary.suite_count, 2);
        assert_eq!(summary.case_count, 4);
    }

    #[test]
    fn rejects_detector_track_label_mismatch() {
        let mut manifest = valid_manifest();
        manifest["external_drift_detectors"][0]["comparison_track"] = json!("observed-outcome");
        let error = validate_value(&manifest).unwrap_err().to_string();
        assert!(error.contains("labels_required conflicts with comparison_track"));
    }

    #[test]
    fn rejects_low_monte_carlo_seed_count() {
        let mut manifest = valid_manifest();
        manifest["drift_protocol"]["monte_carlo_seeds"] = json!(2);
        let error = validate_value(&manifest).unwrap_err().to_string();
        assert!(error.contains("must be at least 30"));
    }

    #[test]
    fn rejects_duplicate_external_detector_ids() {
        let mut manifest = valid_manifest();
        manifest["external_drift_detectors"][1]["id"] = json!("fixed-threshold");
        let error = validate_value(&manifest).unwrap_err().to_string();
        assert!(error.contains("duplicate external drift detector id"));
    }

    #[test]
    fn rejects_missing_external_drift_comparator() {
        let mut manifest = valid_manifest();
        manifest["external_drift_detectors"].as_array_mut().unwrap().pop();
        let error = validate_value(&manifest).unwrap_err().to_string();
        assert!(error.contains("missing required comparator"));
    }

    #[test]
    fn rejects_duplicate_case_ids_across_groups() {
        let mut manifest = valid_manifest();
        manifest["scenario_groups"][1]["cases"][0]["id"] = json!("valid-identical");
        let error = validate_value(&manifest).unwrap_err().to_string();
        assert!(error.contains("duplicate scenario case id"));
    }

    #[test]
    fn rejects_invalid_evidence_admitted_to_replay() {
        let mut manifest = valid_manifest();
        manifest["scenario_groups"][0]["cases"][0]["expected_replay_admission"] = json!(true);
        let error = validate_value(&manifest).unwrap_err().to_string();
        assert!(error.contains("invalid evidence must not enter replay"));
    }

    #[test]
    fn rejects_missing_provenance_fields() {
        let mut manifest = valid_manifest();
        manifest["result_contract"]["required_fields"] = json!(["scenario_id", "seed"]);
        let error = validate_value(&manifest).unwrap_err().to_string();
        assert!(error.contains("missing required provenance field"));
    }

    #[test]
    fn rejects_hard_gate_cases_without_assertions() {
        let mut manifest = valid_manifest();
        manifest["scenario_groups"][0]["cases"][0]["assertions"] = json!([]);
        let error = validate_value(&manifest).unwrap_err().to_string();
        assert!(error.contains("must not be empty"));
    }
}
