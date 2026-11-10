use jsonschema::{Validator, draft202012};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const SCHEMA_ID: &str =
    "https://luminousdynamics.io/schemas/transport/ev-soh-benchmark-protocol/0.1";
const CORPUS_VERSION: &str = "ev-soh-benchmark-corpus/0.1";
const PROTOCOL_VERSION: &str = "ev-soh-benchmark-protocol/0.1";

#[derive(Debug)]
struct CaseResult {
    case_id: String,
    expectation: String,
    accepted: bool,
    errors: Vec<String>,
    summary: Option<BenchmarkSummary>,
}

#[derive(Debug, serde::Serialize)]
pub struct BenchmarkSummary {
    pub benchmark_id: String,
    pub relation: String,
    pub disjoint_key: String,
    pub observation_count: usize,
    pub train_count: usize,
    pub validation_count: usize,
    pub test_count: usize,
    pub dataset_count: usize,
    pub preprocessing_step_count: usize,
    pub metric_count: usize,
    pub uncertainty_status: String,
    pub physical_split_audit: String,
    pub leakage_audit: String,
    pub target_audit: String,
}

#[derive(Debug, serde::Serialize)]
struct ReportPayload {
    corpus_version: String,
    schema_id: String,
    schema_sha256: String,
    canonical_corpus_sha256: String,
    case_count: usize,
    accepted_count: usize,
    rejected_count: usize,
    cases: Vec<CaseResultOutput>,
}

#[derive(Debug, serde::Serialize)]
struct CaseResultOutput {
    case_id: String,
    expectation: String,
    accepted: bool,
    errors: Vec<String>,
    summary: Option<BenchmarkSummary>,
}

#[derive(Debug, serde::Serialize)]
pub struct CorpusReport {
    #[serde(flatten)]
    payload: ReportPayload,
    report_sha256: String,
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn schema_validator(bytes: &[u8]) -> Result<Validator, String> {
    let schema: Value =
        serde_json::from_slice(bytes).map_err(|error| format!("schema JSON: {error}"))?;
    if schema.get("$schema").and_then(Value::as_str)
        != Some("https://json-schema.org/draft/2020-12/schema")
    {
        return Err("schema must declare JSON Schema Draft 2020-12".to_owned());
    }
    if schema.get("$id").and_then(Value::as_str) != Some(SCHEMA_ID) {
        return Err("schema identity does not match SOH benchmark protocol v0.1".to_owned());
    }
    if !draft202012::meta::is_valid(&schema) {
        return Err("schema fails Draft 2020-12 meta-schema validation".to_owned());
    }
    draft202012::options()
        .should_validate_formats(true)
        .build(&schema)
        .map_err(|error| format!("schema compilation: {error}"))
}

fn string_at<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing or invalid string field: {key}"))
}

fn array_at<'a>(value: &'a Value, key: &str) -> Result<&'a Vec<Value>, String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing or invalid array field: {key}"))
}

fn required_digest(value: &Value, key: &str) -> Result<(), String> {
    let digest = string_at(value, key)?;
    if is_digest(digest) {
        Ok(())
    } else {
        Err(format!("field {key} is not a lowercase SHA-256 digest"))
    }
}

fn relation_key(relation: &str) -> Option<&'static str> {
    match relation {
        "unseen_cell" => Some("cell_identity_digest"),
        "unseen_pack" => Some("pack_identity_digest"),
        "unseen_vehicle" => Some("vehicle_identity_digest"),
        "unseen_lab" => Some("lab_identity_digest"),
        "unseen_domain" | "cross_fidelity" => Some("domain_identity_digest"),
        "same_cell_temporal" => Some("none_temporal"),
        _ => None,
    }
}

fn check_dataset_sources(benchmark: &Value) -> Result<BTreeSet<String>, String> {
    let sources = array_at(benchmark, "dataset_sources")?;
    if sources.is_empty() {
        return Err("at least one dataset source is required".to_owned());
    }
    let mut digests = BTreeSet::new();
    for source in sources {
        required_digest(source, "source_digest")?;
        let digest = string_at(source, "source_digest")?.to_owned();
        if !digests.insert(digest) {
            return Err("duplicate dataset source digest".to_owned());
        }
        for field in ["dataset_name", "version", "source_ref", "license_ref", "retrieved_on"] {
            if !nonempty(string_at(source, field)?) {
                return Err(format!("dataset source field {field} is empty"));
            }
        }
    }
    Ok(digests)
}

fn check_target(benchmark: &Value) -> Result<(), String> {
    let target = benchmark
        .get("target")
        .ok_or_else(|| "target specification is missing".to_owned())?;
    let quantity = string_at(target, "quantity")?;
    let basis = string_at(target, "denominator_basis")?;
    let denominator = target.get("denominator_ref");
    match quantity {
        "soh_fraction" => {
            if !matches!(basis, "nominal_capacity" | "initial_measured_capacity") {
                return Err("SOH target requires an explicit capacity denominator".to_owned());
            }
            if denominator.and_then(Value::as_str).is_none_or(|value| !nonempty(value)) {
                return Err("SOH target denominator reference is missing".to_owned());
            }
            if string_at(target, "unit")? != "fraction" {
                return Err("SOH fraction target must use fraction units".to_owned());
            }
        }
        "capacity_ah" => {
            if basis != "none_direct_capacity" || !denominator.is_some_and(Value::is_null) {
                return Err("direct capacity target must not claim an SOH denominator".to_owned());
            }
            if string_at(target, "unit")? != "Ah" {
                return Err("direct capacity target must use Ah units".to_owned());
            }
        }
        _ => return Err("unsupported health target quantity".to_owned()),
    }
    for field in [
        "numerator_ref",
        "reference_protocol_ref",
        "reference_protocol_digest",
        "target_definition_digest",
    ] {
        if field.ends_with("_digest") {
            required_digest(target, field)?;
        } else if !nonempty(string_at(target, field)?) {
            return Err(format!("target field {field} is empty"));
        }
    }
    Ok(())
}

fn check_evaluation_and_splits(
    benchmark: &Value,
    datasets: &BTreeSet<String>,
) -> Result<(String, String, usize, usize, usize, usize), String> {
    let evaluation = benchmark
        .get("evaluation")
        .ok_or_else(|| "evaluation protocol is missing".to_owned())?;
    let relation = string_at(evaluation, "relation")?.to_owned();
    let declared_key = string_at(evaluation, "disjoint_key")?.to_owned();
    let expected_key = relation_key(&relation)
        .ok_or_else(|| format!("unsupported evaluation relation: {relation}"))?;
    if declared_key != expected_key {
        return Err(format!(
            "evaluation relation {relation} requires disjoint key {expected_key}"
        ));
    }
    required_digest(evaluation, "split_manifest_digest")?;
    if string_at(evaluation, "train_partition")? != "train"
        || string_at(evaluation, "validation_partition")? != "validation"
        || string_at(evaluation, "test_partition")? != "test"
    {
        return Err("partition names do not match the frozen protocol".to_owned());
    }

    let observations = array_at(benchmark, "observations")?;
    let mut observation_ids = BTreeSet::new();
    let mut partitions = BTreeMap::<String, usize>::new();
    let mut group_membership = BTreeMap::<String, BTreeSet<String>>::new();
    let mut temporal = BTreeMap::<String, BTreeMap<String, Vec<u64>>>::new();

    for observation in observations {
        required_digest(observation, "observation_id")?;
        required_digest(observation, "dataset_digest")?;
        required_digest(observation, "target_digest")?;
        let observation_id = string_at(observation, "observation_id")?.to_owned();
        if !observation_ids.insert(observation_id) {
            return Err("duplicate observation identity".to_owned());
        }
        let dataset_digest = string_at(observation, "dataset_digest")?;
        if !datasets.contains(dataset_digest) {
            return Err("observation references a dataset digest outside the declared bundle".to_owned());
        }
        let partition = string_at(observation, "partition")?.to_owned();
        *partitions.entry(partition.clone()).or_default() += 1;
        let time_index = observation
            .get("time_index")
            .and_then(Value::as_u64)
            .ok_or_else(|| "observation time_index must be a non-negative integer".to_owned())?;
        let identities = observation
            .get("physical_identities")
            .ok_or_else(|| "physical identity map is missing".to_owned())?;
        for key in [
            "cell_identity_digest",
            "pack_identity_digest",
            "vehicle_identity_digest",
            "lab_identity_digest",
            "domain_identity_digest",
        ] {
            required_digest(identities, key)?;
        }
        if expected_key == "none_temporal" {
            let cell = string_at(identities, "cell_identity_digest")?.to_owned();
            temporal
                .entry(cell)
                .or_default()
                .entry(partition)
                .or_default()
                .push(time_index);
        } else {
            let identity = string_at(identities, expected_key)?.to_owned();
            group_membership
                .entry(identity)
                .or_default()
                .insert(partition);
        }
    }

    let train = *partitions.get("train").unwrap_or(&0);
    let validation = *partitions.get("validation").unwrap_or(&0);
    let test = *partitions.get("test").unwrap_or(&0);
    if train == 0 || validation == 0 || test == 0 {
        return Err("train, validation and test partitions must all be non-empty".to_owned());
    }
    if expected_key == "none_temporal" {
        if temporal.is_empty() {
            return Err("same-cell temporal evaluation has no cell identities".to_owned());
        }
        for (cell, times) in temporal {
            let train_times = times.get("train").ok_or_else(|| {
                format!("same-cell temporal evaluation cell {cell} has no train observations")
            })?;
            let validation_times = times.get("validation").ok_or_else(|| {
                format!("same-cell temporal evaluation cell {cell} has no validation observations")
            })?;
            let test_times = times.get("test").ok_or_else(|| {
                format!("same-cell temporal evaluation cell {cell} has no test observations")
            })?;
            let train_max = *train_times.iter().max().expect("non-empty train times");
            let validation_min = *validation_times.iter().min().expect("non-empty validation times");
            let validation_max = *validation_times.iter().max().expect("non-empty validation times");
            let test_min = *test_times.iter().min().expect("non-empty test times");
            if train_max >= validation_min || validation_max >= test_min {
                return Err(format!(
                    "same-cell temporal boundaries overlap or reverse for cell {cell}"
                ));
            }
        }
    } else {
        if group_membership.values().any(|partitions| partitions.len() > 1) {
            return Err(format!(
                "physical identity leakage across partitions for {expected_key}"
            ));
        }
        for partition in ["train", "validation", "test"] {
            if !group_membership.values().any(|partitions| partitions.contains(partition)) {
                return Err(format!("no physical groups assigned to {partition}"));
            }
        }
    }

    Ok((relation, declared_key, observations.len(), train, validation, test))
}

fn check_preprocessing(benchmark: &Value) -> Result<usize, String> {
    let steps = array_at(benchmark, "preprocessing")?;
    if steps.is_empty() {
        return Err("preprocessing pipeline must be declared, including explicit no-op".to_owned());
    }
    for step in steps {
        let scope = string_at(step, "fit_scope")?;
        if !matches!(scope, "train_only" | "fixed_predeclared") {
            return Err(format!(
                "preprocessing step {} fits outside permitted scope",
                string_at(step, "step_id")?
            ));
        }
        required_digest(step, "transform_digest")?;
        if !nonempty(string_at(step, "transform_ref")?) {
            return Err("preprocessing transform reference is empty".to_owned());
        }
    }
    let selection = benchmark
        .get("model_selection")
        .ok_or_else(|| "model-selection contract is missing".to_owned())?;
    if selection.get("test_labels_used").and_then(Value::as_bool) != Some(false) {
        return Err("test labels must not influence model selection".to_owned());
    }
    if !matches!(
        string_at(selection, "feature_selection_fit_scope")?,
        "train_only" | "fixed_predeclared"
    ) {
        return Err("feature selection was fitted outside the permitted scope".to_owned());
    }
    if !matches!(
        string_at(selection, "tuning_partition")?,
        "validation" | "nested_group_validation" | "none"
    ) {
        return Err("invalid model tuning partition".to_owned());
    }
    required_digest(selection, "configuration_digest")?;
    Ok(steps.len())
}

fn check_metrics(benchmark: &Value) -> Result<usize, String> {
    let metrics = array_at(benchmark, "metrics")?;
    let mut aggregations = BTreeSet::new();
    let mut names = BTreeSet::new();
    for metric in metrics {
        let name = string_at(metric, "metric_name")?;
        let aggregation = string_at(metric, "aggregation")?;
        names.insert(name.to_owned());
        aggregations.insert(aggregation.to_owned());
        if !matches!(name, "MAE" | "RMSE" | "R2") {
            return Err(format!("unsupported metric name: {name}"));
        }
        if !matches!(
            aggregation,
            "sample_pooled" | "macro_per_cell" | "macro_per_domain" | "worst_domain"
        ) {
            return Err(format!("unsupported metric aggregation: {aggregation}"));
        }
        let value = metric
            .get("value")
            .and_then(Value::as_f64)
            .ok_or_else(|| "metric value must be finite numeric".to_owned())?;
        if !value.is_finite() {
            return Err("metric value is not finite".to_owned());
        }
        if metric.get("sample_count").and_then(Value::as_u64).unwrap_or(0) == 0 {
            return Err("metric sample count must be positive".to_owned());
        }
        required_digest(metric, "result_digest")?;
    }
    if !names.contains("MAE") || !names.contains("RMSE") {
        return Err("benchmark must report both MAE and RMSE".to_owned());
    }
    if !aggregations.contains("macro_per_cell") || !aggregations.contains("worst_domain") {
        return Err("benchmark must report macro-per-cell and worst-domain aggregation".to_owned());
    }
    Ok(metrics.len())
}

fn check_uncertainty(benchmark: &Value) -> Result<String, String> {
    let uncertainty = benchmark
        .get("uncertainty")
        .ok_or_else(|| "uncertainty reporting status is missing".to_owned())?;
    match string_at(uncertainty, "status")? {
        "not_reported" => Ok("not_reported".to_owned()),
        "reported" => {
            if !nonempty(string_at(uncertainty, "method_ref")?) {
                return Err("uncertainty method reference is missing".to_owned());
            }
            let nominal = uncertainty
                .get("nominal_interval_level")
                .and_then(Value::as_f64)
                .ok_or_else(|| "nominal interval level is missing".to_owned())?;
            let observed = uncertainty
                .get("observed_coverage")
                .and_then(Value::as_f64)
                .ok_or_else(|| "observed interval coverage is missing".to_owned())?;
            if !(0.0..1.0).contains(&nominal) || !(0.0..=1.0).contains(&observed) {
                return Err("uncertainty coverage values are outside [0,1]".to_owned());
            }
            if !matches!(
                string_at(uncertainty, "calibration_partition")?,
                "validation" | "nested_group_validation"
            ) {
                return Err("uncertainty calibration must be tied to validation data".to_owned());
            }
            if uncertainty
                .get("calibration_group_disjointness")
                .and_then(Value::as_bool)
                != Some(true)
            {
                return Err("uncertainty calibration group disjointness is unproven".to_owned());
            }
            Ok("reported_with_calibration_receipt".to_owned())
        }
        _ => Err("unknown uncertainty status".to_owned()),
    }
}

fn audit_benchmark(benchmark: &Value) -> Result<BenchmarkSummary, Vec<String>> {
    let mut errors = Vec::new();
    if string_at(benchmark, "protocol_version").ok() != Some(PROTOCOL_VERSION) {
        errors.push("protocol version mismatch".to_owned());
    }
    let benchmark_id = string_at(benchmark, "benchmark_id")
        .map(str::to_owned)
        .unwrap_or_default();
    if !nonempty(&benchmark_id) {
        errors.push("benchmark identity is empty".to_owned());
    }

    let datasets = match check_dataset_sources(benchmark) {
        Ok(value) => Some(value),
        Err(error) => {
            errors.push(error);
            None
        }
    };
    if let Err(error) = check_target(benchmark) {
        errors.push(error);
    }
    let split = datasets
        .as_ref()
        .and_then(|datasets| match check_evaluation_and_splits(benchmark, datasets) {
            Ok(value) => Some(value),
            Err(error) => {
                errors.push(error);
                None
            }
        });
    let preprocessing_count = match check_preprocessing(benchmark) {
        Ok(value) => Some(value),
        Err(error) => {
            errors.push(error);
            None
        }
    };
    let metric_count = match check_metrics(benchmark) {
        Ok(value) => Some(value),
        Err(error) => {
            errors.push(error);
            None
        }
    };
    let uncertainty_status = match check_uncertainty(benchmark) {
        Ok(value) => Some(value),
        Err(error) => {
            errors.push(error);
            None
        }
    };
    let execution = benchmark.get("execution");
    if let Some(execution) = execution {
        for key in ["source_commit_digest", "dependencies_digest"] {
            if let Err(error) = required_digest(execution, key) {
                errors.push(error);
            }
        }
        if !nonempty(execution.get("toolchain").and_then(Value::as_str).unwrap_or(""))
            || !nonempty(execution.get("hardware").and_then(Value::as_str).unwrap_or(""))
        {
            errors.push("execution toolchain or hardware is not identified".to_owned());
        }
    } else {
        errors.push("execution environment is missing".to_owned());
    }

    if !errors.is_empty() {
        return Err(errors);
    }
    let (relation, disjoint_key, observation_count, train_count, validation_count, test_count) =
        split.expect("successful split audit");
    Ok(BenchmarkSummary {
        benchmark_id,
        relation,
        disjoint_key,
        observation_count,
        train_count,
        validation_count,
        test_count,
        dataset_count: datasets.expect("successful source audit").len(),
        preprocessing_step_count: preprocessing_count.expect("successful preprocessing audit"),
        metric_count: metric_count.expect("successful metric audit"),
        uncertainty_status: uncertainty_status.expect("successful uncertainty audit"),
        physical_split_audit: "PASS_WITH_DECLARED_RELATION".to_owned(),
        leakage_audit: "NO_PROHIBITED_PREPROCESSING_OR_TEST_LABEL_FIT".to_owned(),
        target_audit: "OPERATIONAL_TARGET_IDENTIFIED".to_owned(),
    })
}

fn canonicalize(value: &mut Value) {
    if let Some(cases) = value.get_mut("cases").and_then(Value::as_array_mut) {
        cases.sort_by(|left, right| {
            left.get("case_id")
                .and_then(Value::as_str)
                .cmp(&right.get("case_id").and_then(Value::as_str))
        });
        for case in cases {
            if let Some(benchmark) = case.get_mut("benchmark") {
                for key in ["dataset_sources", "observations", "preprocessing", "metrics"] {
                    if let Some(rows) = benchmark.get_mut(key).and_then(Value::as_array_mut) {
                        let identity_key = match key {
                            "dataset_sources" => "source_digest",
                            "observations" => "observation_id",
                            "preprocessing" => "step_id",
                            "metrics" => "metric_name",
                            _ => unreachable!(),
                        };
                        rows.sort_by(|left, right| {
                            left.get(identity_key)
                                .and_then(Value::as_str)
                                .cmp(&right.get(identity_key).and_then(Value::as_str))
                        });
                    }
                }
            }
        }
    }
}

pub fn audit_corpus(schema_bytes: &[u8], corpus_bytes: &[u8]) -> Result<CorpusReport, String> {
    let validator = schema_validator(schema_bytes)?;
    let mut value: Value =
        serde_json::from_slice(corpus_bytes).map_err(|error| format!("corpus JSON: {error}"))?;
    canonicalize(&mut value);
    let canonical = serde_json::to_vec(&value)
        .map_err(|error| format!("canonical corpus serialization: {error}"))?;
    let corpus: Value =
        serde_json::from_slice(&canonical).map_err(|error| format!("canonical corpus: {error}"))?;
    if corpus.get("corpus_version").and_then(Value::as_str) != Some(CORPUS_VERSION)
        || corpus.get("schema_id").and_then(Value::as_str) != Some(SCHEMA_ID)
    {
        return Err("corpus version or schema identity mismatch".to_owned());
    }
    let cases = corpus
        .get("cases")
        .and_then(Value::as_array)
        .ok_or_else(|| "corpus cases array is missing".to_owned())?;
    let mut seen = BTreeSet::new();
    let mut outputs = Vec::new();
    let mut accepted_count = 0usize;
    let mut rejected_count = 0usize;

    for case in cases {
        let case_id = case
            .get("case_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        if !nonempty(&case_id) || !seen.insert(case_id.clone()) {
            return Err(format!("case identity is empty or duplicated: {case_id}"));
        }
        let expectation = case
            .get("expectation")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        if !matches!(expectation.as_str(), "accept" | "reject") {
            return Err(format!("case {case_id} has an invalid expectation"));
        }
        let benchmark = case
            .get("benchmark")
            .ok_or_else(|| format!("case {case_id} has no benchmark"))?;
        let schema_errors: Vec<String> = validator
            .iter_errors(benchmark)
            .map(|error| error.to_string())
            .collect();
        let outcome = if schema_errors.is_empty() {
            audit_benchmark(benchmark)
        } else {
            Err(schema_errors)
        };
        let accepted = outcome.is_ok();
        if accepted != (expectation == "accept") {
            let details = outcome.err().unwrap_or_else(|| vec!["unexpected acceptance".to_owned()]);
            return Err(format!(
                "case {case_id} expected {expectation} but accepted={accepted}: {}",
                details.join("; ")
            ));
        }
        let (summary, errors) = match outcome {
            Ok(summary) => {
                accepted_count += 1;
                (Some(summary), Vec::new())
            }
            Err(errors) => {
                rejected_count += 1;
                (None, errors)
            }
        };
        outputs.push(CaseResultOutput {
            case_id,
            expectation,
            accepted,
            errors,
            summary,
        });
    }

    let payload = ReportPayload {
        corpus_version: CORPUS_VERSION.to_owned(),
        schema_id: SCHEMA_ID.to_owned(),
        schema_sha256: sha256(schema_bytes),
        canonical_corpus_sha256: sha256(&canonical),
        case_count: outputs.len(),
        accepted_count,
        rejected_count,
        cases: outputs,
    };
    let bytes = serde_json::to_vec(&payload)
        .map_err(|error| format!("report serialization: {error}"))?;
    Ok(CorpusReport {
        payload,
        report_sha256: sha256(&bytes),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCHEMA: &[u8] =
        include_bytes!("../../../schemas/transport/ev_soh_benchmark_protocol_v0_1.schema.json");
    const CORPUS: &[u8] =
        include_bytes!("../../../fixtures/ev_battery/soh_benchmark_protocol_scenarios_v0_1.json");

    fn report() -> CorpusReport {
        audit_corpus(SCHEMA, CORPUS).expect("frozen SOH benchmark protocol corpus")
    }

    #[test]
    fn frozen_protocol_corpus_accepts_two_valid_designs_and_rejects_nine_controls() {
        let report = report();
        assert_eq!(report.payload.case_count, 11);
        assert_eq!(report.payload.accepted_count, 2);
        assert_eq!(report.payload.rejected_count, 9);
    }

    #[test]
    fn unseen_cell_and_same_cell_temporal_claims_are_distinguished() {
        let report = report();
        let unseen = report.payload.cases.iter()
            .find(|case| case.case_id == "valid-unseen-cell")
            .and_then(|case| case.summary.as_ref())
            .unwrap();
        let temporal = report.payload.cases.iter()
            .find(|case| case.case_id == "valid-same-cell-temporal")
            .and_then(|case| case.summary.as_ref())
            .unwrap();
        assert_eq!(unseen.relation, "unseen_cell");
        assert_eq!(unseen.disjoint_key, "cell_identity_digest");
        assert_eq!(temporal.relation, "same_cell_temporal");
        assert_eq!(temporal.disjoint_key, "none_temporal");
    }

    #[test]
    fn held_out_cell_overlap_and_preprocessing_leakage_are_rejected() {
        let report = report();
        for case_id in ["cell-identity-overlap", "preprocessing-all-data", "test-labels-used-in-model-selection"] {
            assert!(!report.payload.cases.iter().find(|case| case.case_id == case_id).unwrap().accepted);
        }
    }

    #[test]
    fn benchmark_target_and_metric_aggregation_must_be_operational() {
        let report = report();
        for case_id in ["soh-target-without-denominator", "pooled-only-metrics"] {
            assert!(!report.payload.cases.iter().find(|case| case.case_id == case_id).unwrap().accepted);
        }
    }

    #[test]
    fn protocol_report_digest_is_invariant_to_observation_order() {
        let original = report();
        let mut value: Value = serde_json::from_slice(CORPUS).expect("corpus JSON");
        value["cases"].as_array_mut().unwrap().reverse();
        for case in value["cases"].as_array_mut().unwrap() {
            case["benchmark"]["observations"].as_array_mut().unwrap().reverse();
            case["benchmark"]["metrics"].as_array_mut().unwrap().reverse();
        }
        let bytes = serde_json::to_vec(&value).expect("permuted corpus");
        let replay = audit_corpus(SCHEMA, &bytes).expect("permuted benchmark audit");
        assert_eq!(original.report_sha256, replay.report_sha256);
    }
}
