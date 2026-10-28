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
const REQUIRED_KDD_FAMILIES: [&str; 14] = [
    "ADWIN",
    "CUSUM",
    "DDM",
    "EDDM",
    "EWMA",
    "GMA",
    "HDDMA",
    "HDDMW",
    "Page-Hinkley (PH)",
    "RDDM",
    "SEED",
    "STEPD",
    "ABCD (error stream and ABCD(X) feature stream)",
    "STUDD+ADWIN",
];
const REQUIRED_DRIFT_DETECTORS: [&str; 26] = [
    "fixed-threshold",
    "ewma-control-chart",
    "page-hinkley",
    "predictive-rank-martingale-portfolio",
    "predictive-rank-martingale-ons-portfolio",
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
    "kdd26-adwin-error-stream",
    "kdd26-cusum-error-stream",
    "kdd26-ddm-error-stream",
    "kdd26-eddm-error-stream",
    "kdd26-ewma-error-stream",
    "kdd26-gma-error-stream",
    "kdd26-hddma-error-stream",
    "kdd26-hddmw-error-stream",
    "kdd26-page-hinkley-error-stream",
    "kdd26-rddm-error-stream",
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
    let kdd_protocol = root.get("drift_protocol")
        .and_then(|value| value.get("kdd26_reference_protocol"))
        .context("drift_protocol.kdd26_reference_protocol is required")?;
    ensure!(
        nonempty_string_field(kdd_protocol, "paper_doi", "kdd26_reference_protocol")?
            == "10.1145/3770855.3819070",
        "KDD drift benchmark reference DOI is missing or unexpected"
    );
    ensure!(
        kdd_protocol.get("detector_family_count").and_then(Value::as_u64) == Some(14)
            && kdd_protocol.get("datasets_count").and_then(Value::as_u64) == Some(7),
        "KDD reference protocol must state 14 method families and seven datasets"
    );
    let dataset_names = nonempty_string_array(kdd_protocol, "dataset_names_to_pin", "kdd26_reference_protocol")?;
    ensure!(
        dataset_names.len() == 7 && dataset_names.iter().collect::<BTreeSet<_>>().len() == 7,
        "KDD reference protocol must list seven unique datasets to pin"
    );
    let method_families = nonempty_string_array(kdd_protocol, "method_families", "kdd26_reference_protocol")?;
    ensure!(
        method_families.len() == 14 && method_families.iter().collect::<BTreeSet<_>>().len() == 14,
        "KDD reference protocol must list 14 unique detector families"
    );
    for expected in REQUIRED_KDD_FAMILIES {
        ensure!(
            method_families.iter().any(|family| family == expected),
            "KDD reference protocol is missing detector family {expected}"
        );
    }
    let drift_types = nonempty_string_array(kdd_protocol, "drift_types", "kdd26_reference_protocol")?;
    ensure!(drift_types.len() == 4, "KDD reference protocol must cover four drift types");
    let transitions = nonempty_string_array(kdd_protocol, "transitions", "kdd26_reference_protocol")?;
    ensure!(transitions.len() == 2, "KDD reference protocol must cover abrupt and gradual transitions");
    for field in ["monitored_classifier", "stream_protocol", "hyperparameter_protocol", "important_limit", "status"] {
        nonempty_string_field(kdd_protocol, field, "kdd26_reference_protocol")?;
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
    for phrase in ["matching-window deadline", "earliest unused valid-evidence alarm", "false alarm", "missed", "invalid/missing PE"] {
        ensure!(matching_rule.contains(phrase), "drift_protocol.matching_rule must state: {phrase}");
    }
    let trace_contract = nonempty_string_field(drift_protocol, "trace_contract", "drift_protocol")?;
    for phrase in ["detector_version", "comparison_track", "label_delay_cycles", "evidence_valid", "hard gate"] {
        ensure!(trace_contract.contains(phrase), "drift_protocol.trace_contract must state: {phrase}");
    }
    nonempty_string_field(drift_protocol, "tuning_policy", "drift_protocol")?;
    nonempty_string_field(drift_protocol, "source_paper", "drift_protocol")?;
    nonempty_string_field(drift_protocol, "claims", "drift_protocol")?;

    let utility_contract = drift_protocol
        .get("adaptation_counterfactual_protocol")
        .context("drift_protocol.adaptation_counterfactual_protocol is required")?;
    ensure!(
        nonempty_string_field(utility_contract, "protocol_id", "adaptation_counterfactual_protocol")?
            == "same-stream-three-arm-counterfactual-v1",
        "unexpected adaptation counterfactual protocol id"
    );
    let arms = nonempty_string_array(utility_contract, "arms", "adaptation_counterfactual_protocol")?;
    ensure!(
        arms.len() == 3
            && ["no-adapt", "detector-triggered", "oracle-timed"]
                .iter().all(|expected| arms.iter().any(|arm| arm == expected)),
        "adaptation counterfactual protocol must declare no-adapt, detector-triggered, and oracle-timed arms"
    );
    let replay_identity = nonempty_string_array(
        utility_contract,
        "replay_identity_requirements",
        "adaptation_counterfactual_protocol",
    )?;
    for phrase in [
        "same pinned exogenous stream digest",
        "same contiguous cycle panel",
        "repository head SHA",
        "not a deployable detector",
        "Each arm carries its own observed exogenous-stream digest",
    ] {
        ensure!(
            replay_identity.iter().any(|requirement| requirement.contains(phrase)),
            "adaptation counterfactual replay contract must state: {phrase}"
        );
    }
    let utility_metrics = nonempty_string_array(
        utility_contract,
        "required_metrics",
        "adaptation_counterfactual_protocol",
    )?;
    for expected in [
        "no_adapt.mean_task_loss",
        "detector_triggered.mean_task_loss",
        "oracle_timed.mean_task_loss",
        "no_adapt.exogenous_stream_sha256",
        "detector_triggered.exogenous_stream_sha256",
        "oracle_timed.exogenous_stream_sha256",
        "detector_task_loss_improvement_vs_no_adapt",
        "detector_task_loss_regret_vs_oracle_timed",
        "detector_adaptation_cost_delta_vs_no_adapt",
        "canonical_input_sha256",
        "result_sha256",
    ] {
        ensure!(
            utility_metrics.iter().any(|metric| metric == expected),
            "adaptation counterfactual protocol is missing metric {expected}"
        );
    }
    let unit_policy = nonempty_string_field(
        utility_contract,
        "unit_policy",
        "adaptation_counterfactual_protocol",
    )?;
    ensure!(
        unit_policy.contains("separate axes")
            && unit_policy.contains("preregistered"),
        "adaptation utility contract must keep loss and cost separate unless weights are preregistered"
    );
    let utility_claim_policy = nonempty_string_field(
        utility_contract,
        "claim_policy",
        "adaptation_counterfactual_protocol",
    )?;
    ensure!(
        utility_claim_policy.contains("do not prove hidden-state identity")
            && utility_claim_policy.contains("not a deployable detector"),
        "adaptation utility claim policy must limit counterfactual/oracle claims"
    );

    let comparison_contract = drift_protocol
        .get("paired_comparison_contract")
        .context("drift_protocol.paired_comparison_contract is required")?;
    ensure!(
        nonempty_string_field(comparison_contract, "protocol_id", "paired_comparison_contract")?
            == "paired-dataset-bootstrap-v1",
        "unexpected paired comparison protocol id"
    );
    ensure!(
        comparison_contract.get("minimum_independent_datasets").and_then(Value::as_u64) == Some(3),
        "paired comparison must require at least three independent datasets"
    );
    ensure!(
        comparison_contract.get("minimum_unique_seeds").and_then(Value::as_u64) == Some(2),
        "paired comparison must require at least two distinct seeds"
    );
    ensure!(
        comparison_contract.get("bootstrap_replicates_min").and_then(Value::as_u64) == Some(1000)
            && comparison_contract.get("bootstrap_replicates_max").and_then(Value::as_u64) == Some(100000),
        "paired bootstrap replicate bounds must be [1000, 100000]"
    );
    for field in ["matching_key", "required_outputs"] {
        nonempty_string_array(comparison_contract, field, "paired_comparison_contract")?;
    }
    let resampling = nonempty_string_field(comparison_contract, "resampling", "paired_comparison_contract")?;
    for phrase in ["dataset clusters", "seeds", "complete paired scenario block", "identical scenario panel"] {
        ensure!(resampling.contains(phrase), "paired comparison resampling contract must state: {phrase}");
    }
    let required_outputs = nonempty_string_array(comparison_contract, "required_outputs", "paired_comparison_contract")?;
    for expected in [
        "candidate_dataset_macro_mean",
        "baseline_dataset_macro_mean",
        "dataset_macro_paired_effect",
        "median_oriented_paired_delta",
        "paired_win_fraction",
        "percentile_95_ci_oriented_effect",
        "bootstrap_fraction_effect_above_zero",
        "per_dataset",
        "canonical_input_sha256",
        "result_sha256",
        "minimum_seeds_per_dataset",
        "sota_coverage_gate_pass",
        "sota_coverage_gate_reason",
    ] {
        ensure!(required_outputs.iter().any(|field| field == expected), "paired comparison contract is missing output {expected}");
    }
    let claim_policy = nonempty_string_field(comparison_contract, "claim_policy", "paired_comparison_contract")?;
    ensure!(claim_policy.contains("not a universal SOTA claim"), "paired comparison claim policy must forbid universal SOTA conclusions");
    let coverage_gate = comparison_contract.get("sota_coverage_gate")
        .context("paired_comparison_contract.sota_coverage_gate is required")?;
    ensure!(
        coverage_gate.get("minimum_datasets").and_then(Value::as_u64) == Some(7)
            && coverage_gate.get("minimum_seeds_per_dataset").and_then(Value::as_u64) == Some(50),
        "SOTA coverage gate must require seven datasets and 50 seeds per dataset"
    );
    let coverage_meaning = nonempty_string_field(coverage_gate, "meaning", "sota_coverage_gate")?;
    ensure!(
        coverage_meaning.contains("not a performance result")
            && coverage_meaning.contains("not evidence of superiority"),
        "SOTA coverage gate must explicitly distinguish coverage from performance"
    );

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
                "comparison_track": if ["seed", "stepd", "abcd-error-stream", "dynamic-threshold-dtd", "kdd26-adwin-error-stream", "kdd26-cusum-error-stream", "kdd26-ddm-error-stream", "kdd26-eddm-error-stream", "kdd26-ewma-error-stream", "kdd26-gma-error-stream", "kdd26-hddma-error-stream", "kdd26-hddmw-error-stream", "kdd26-page-hinkley-error-stream", "kdd26-rddm-error-stream"].contains(id) { "observed-outcome" } else if *id == "studd-adwin-compression" { "auxiliary-loss-unsupervised" } else if ["discriminative-drift-detector", "image-based-drift-detector", "semi-parametric-log-likelihood", "abcd-feature-space"].contains(id) { "feature-only-unsupervised" } else { "valid-pe-only" },
                "labels_required": (["seed", "stepd", "abcd-error-stream", "dynamic-threshold-dtd", "kdd26-adwin-error-stream", "kdd26-cusum-error-stream", "kdd26-ddm-error-stream", "kdd26-eddm-error-stream", "kdd26-ewma-error-stream", "kdd26-gma-error-stream", "kdd26-hddma-error-stream", "kdd26-hddmw-error-stream", "kdd26-page-hinkley-error-stream", "kdd26-rddm-error-stream"].contains(id)),
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
                "matching_rule": "Process events by earliest matching-window deadline. Match the earliest unused valid-evidence alarm in the event window. Unmatched alarm is a false alarm; unmatched event is missed; invalid/missing PE cannot be observations.",
                "tuning_policy": "Tune on all but one dataset and hold that dataset out.",
                "trace_contract": "Each trace pins detector_version and comparison_track; label_delay_cycles is recorded; alarms mark evidence_valid. Invalid-evidence alarms cannot match, count as false alarms, and fail the hard gate.",
                "source_paper": "https://doi.org/10.1145/3770855.3819070",
                "claims": "Comparator definitions are not measured results.",
                "adaptation_counterfactual_protocol": {
                    "protocol_id": "same-stream-three-arm-counterfactual-v1",
                    "arms": ["no-adapt", "detector-triggered", "oracle-timed"],
                    "replay_identity_requirements": [
                        "All arms consume the same pinned exogenous stream digest.",
                        "All arms emit the same contiguous cycle panel beginning at zero.",
                        "Pin arm version, repository head SHA, and configuration digest separately.",
                        "Oracle-timed policy uses ground-truth change timing only as a reference arm; it is not a deployable detector.",
                        "Each arm carries its own observed exogenous-stream digest and must match the shared stream digest."
                    ],
                    "required_metrics": [
                        "no_adapt.mean_task_loss",
                        "detector_triggered.mean_task_loss",
                        "oracle_timed.mean_task_loss",
                        "no_adapt.exogenous_stream_sha256",
                        "detector_triggered.exogenous_stream_sha256",
                        "oracle_timed.exogenous_stream_sha256",
                        "detector_task_loss_improvement_vs_no_adapt",
                        "detector_task_loss_regret_vs_oracle_timed",
                        "detector_adaptation_cost_delta_vs_no_adapt",
                        "detector_dominated_by_no_adapt",
                        "adaptation_cost_units_per_task_loss_unit_reduced",
                        "canonical_input_sha256",
                        "result_sha256"
                    ],
                    "unit_policy": "Report task loss and adaptation resource cost as separate axes. Do not collapse them into a composite utility unless the cost conversion was preregistered.",
                    "claim_policy": "A shared stream digest and paired arm records do not prove hidden-state identity or implementation correctness. Oracle-timed is not a deployable detector; report all arm-level values."
                },
                "kdd26_reference_protocol": {
                    "paper_doi": "10.1145/3770855.3819070",
                    "detector_family_count": 14,
                    "datasets_count": 7,
                    "dataset_names_to_pin": ["Asfault", "Electricity", "Covertype", "GasSensorArray", "NOAA", "Posture", "Rialto"],
                    "method_families": ["ADWIN", "CUSUM", "DDM", "EDDM", "EWMA", "GMA", "HDDMA", "HDDMW", "Page-Hinkley (PH)", "RDDM", "SEED", "STEPD", "ABCD (error stream and ABCD(X) feature stream)", "STUDD+ADWIN"],
                    "monitored_classifier": "Hoeffding Tree",
                    "stream_protocol": "Prequential test-then-train, per-instance observed correctness/error.",
                    "hyperparameter_protocol": "Leave one dataset out, tune on remaining datasets.",
                    "drift_types": ["class-prior", "label-swap", "feature-permutation", "feature-filtering"],
                    "transitions": ["abrupt", "gradual"],
                    "important_limit": "Feature-only detectors cannot identify pure target changes with unchanged feature distribution.",
                    "status": "Reference protocol declared; implementations and datasets are not yet pinned."
                },
                "paired_comparison_contract": {
                    "protocol_id": "paired-dataset-bootstrap-v1",
                    "minimum_independent_datasets": 3,
                    "minimum_unique_seeds": 2,
                    "bootstrap_replicates_min": 1000,
                    "bootstrap_replicates_max": 100000,
                    "resampling": "sample dataset clusters with replacement, then seeds with each complete paired scenario block; require identical scenario panel across datasets and give each dataset equal final weight",
                    "matching_key": ["dataset_id", "seed", "scenario_id"],
                    "required_outputs": ["candidate_dataset_macro_mean", "baseline_dataset_macro_mean", "dataset_macro_paired_effect", "median_oriented_paired_delta", "paired_win_fraction", "percentile_95_ci_oriented_effect", "bootstrap_fraction_effect_above_zero", "per_dataset", "canonical_input_sha256", "result_sha256", "minimum_seeds_per_dataset", "sota_coverage_gate_pass", "sota_coverage_gate_reason"],
                    "claim_policy": "A positive interval is not a universal SOTA claim.",
                    "sota_coverage_gate": {
                        "minimum_datasets": 7,
                        "minimum_seeds_per_dataset": 50,
                        "meaning": "Coverage eligibility only, not a performance result and not evidence of superiority. Actual comparator runs, uncertainty, and costs are still required."
                    }
                }
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
                },
                {
                    "id": "S1-calibration-and-shift",
                    "hard_gate": false,
                    "cases": [
                        {"id":"class-prior-abrupt","inject":"Abrupt class-prior drift.","metrics":["event-f1"]},
                        {"id":"class-prior-gradual","inject":"Gradual class-prior drift.","metrics":["event-f1"]},
                        {"id":"label-swap-abrupt","inject":"Abrupt label-swap drift.","metrics":["event-f1"]},
                        {"id":"label-swap-gradual","inject":"Gradual label-swap drift.","metrics":["event-f1"]},
                        {"id":"feature-permutation-abrupt","inject":"Abrupt feature-permutation drift.","metrics":["event-f1"]},
                        {"id":"feature-permutation-gradual","inject":"Gradual feature-permutation drift.","metrics":["event-f1"]},
                        {"id":"feature-filtering-abrupt","inject":"Abrupt feature-filtering drift.","metrics":["event-f1"]},
                        {"id":"feature-filtering-gradual","inject":"Gradual feature-filtering drift.","metrics":["event-f1"]}
                    ]
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
        assert_eq!(summary.suite_count, 3);
        assert_eq!(summary.case_count, 12);
    }

    #[test]
    fn rejects_detector_track_label_mismatch() {
        let mut manifest = valid_manifest();
        manifest["external_drift_detectors"][0]["comparison_track"] = json!("observed-outcome");
        let error = validate_value(&manifest).unwrap_err().to_string();
        assert!(error.contains("labels_required conflicts with comparison_track"));
    }

    #[test]
    fn rejects_counterfactual_utility_protocol_that_hides_loss_cost_tradeoff() {
        let mut manifest = valid_manifest();
        manifest["drift_protocol"]["adaptation_counterfactual_protocol"]["unit_policy"] =
            json!("Only report one composite utility score.");
        let error = validate_value(&manifest).unwrap_err().to_string();
        assert!(error.contains("keep loss and cost separate"));
    }

    #[test]
    fn rejects_counterfactual_protocol_that_omits_per_arm_stream_attestation() {
        let mut manifest = valid_manifest();
        manifest["drift_protocol"]["adaptation_counterfactual_protocol"]["replay_identity_requirements"]
            .as_array_mut()
            .unwrap()
            .retain(|item| !item.as_str().unwrap_or("").contains("Each arm carries its own"));
        let error = validate_value(&manifest).unwrap_err().to_string();
        assert!(error.contains("Each arm carries its own observed exogenous-stream digest"));
    }

    #[test]
    fn rejects_counterfactual_protocol_that_omits_oracle_limitation() {
        let mut manifest = valid_manifest();
        manifest["drift_protocol"]["adaptation_counterfactual_protocol"]["replay_identity_requirements"][3] =
            json!("Oracle timing is the production baseline.");
        let error = validate_value(&manifest).unwrap_err().to_string();
        assert!(error.contains("not a deployable detector"));
    }

    #[test]
    fn rejects_coverage_gate_that_can_be_misread_as_performance() {
        let mut manifest = valid_manifest();
        manifest["drift_protocol"]["paired_comparison_contract"]["sota_coverage_gate"]["meaning"] =
            json!("Coverage is enough to claim superiority.");
        let error = validate_value(&manifest).unwrap_err().to_string();
        assert!(error.contains("distinguish coverage from performance"));
    }

    #[test]
    fn rejects_paired_comparison_contract_with_too_few_datasets() {
        let mut manifest = valid_manifest();
        manifest["drift_protocol"]["paired_comparison_contract"]["minimum_independent_datasets"] = json!(1);
        let error = validate_value(&manifest).unwrap_err().to_string();
        assert!(error.contains("at least three independent datasets"));
    }

    #[test]
    fn rejects_incomplete_trace_contract() {
        let mut manifest = valid_manifest();
        manifest["drift_protocol"]["trace_contract"] = json!("Detector metadata is not recorded.");
        let error = validate_value(&manifest).unwrap_err().to_string();
        assert!(error.contains("trace_contract must state: detector_version"));
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
    fn rejects_missing_required_kdd_family() {
        let mut manifest = valid_manifest();
        manifest["drift_protocol"]["kdd26_reference_protocol"]["method_families"][0] = json!("made-up-detector");
        let error = validate_value(&manifest).unwrap_err().to_string();
        assert!(error.contains("KDD reference protocol is missing detector family ADWIN"));
    }

    #[test]
    fn rejects_duplicate_kdd_method_families() {
        let mut manifest = valid_manifest();
        manifest["drift_protocol"]["kdd26_reference_protocol"]["method_families"][1] = json!("ADWIN");
        let error = validate_value(&manifest).unwrap_err().to_string();
        assert!(error.contains("14 unique detector families"));
    }

    #[test]
    fn rejects_incomplete_kdd_reference_coverage() {
        let mut manifest = valid_manifest();
        manifest["drift_protocol"]["kdd26_reference_protocol"]["datasets_count"] = json!(3);
        let error = validate_value(&manifest).unwrap_err().to_string();
        assert!(error.contains("14 method families and seven datasets"));
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
