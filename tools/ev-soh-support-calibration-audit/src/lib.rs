use jsonschema::{Validator, draft202012};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const SCHEMA_ID: &str =
    "https://luminousdynamics.io/schemas/transport/ev-soh-support-calibration/0.1";
const CORPUS_VERSION: &str = "ev-soh-support-calibration-corpus/0.1";
const PROTOCOL_VERSION: &str = "ev-soh-support-calibration/0.1";
const EPSILON: f64 = 1e-12;
const UNDER_COVERAGE_90: f64 = 0.87;
const UNDER_COVERAGE_95: f64 = 0.92;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Corpus {
    corpus_version: String,
    schema_id: String,
    cases: Vec<CorpusCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CorpusCase {
    case_id: String,
    expectation: String,
    expected_extrapolation_undercovered: Option<bool>,
    manifest: Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    protocol_version: String,
    manifest_id: String,
    benchmark_manifest_digest: String,
    dataset_digest: String,
    split_manifest_digest: String,
    model_digest: String,
    evaluation_mode: String,
    split_frozen_at_ms: u64,
    train_group_identity_digests: Vec<String>,
    validation_group_identity_digests: Vec<String>,
    support_scoring: SupportScoring,
    calibration: Calibration,
    test_entries: Vec<TestEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SupportScoring {
    method_ref: String,
    method_digest: String,
    feature_schema_digest: String,
    fit_scope: String,
    reference_partition: String,
    uses_validation_or_test_labels: bool,
    score_definition: String,
    score_direction: String,
    in_support_max_score: f64,
    extrapolation_min_score: f64,
    thresholds_frozen_at_ms: u64,
    risk_coverage_cutoffs: Vec<f64>,
    threshold_evidence_digest: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Calibration {
    method_ref: String,
    method_digest: String,
    partition: String,
    calibration_group_identity_digests: Vec<String>,
    calibration_split_digest: String,
    external_source_digest: Option<String>,
    fitted_at_ms: u64,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct TestEntry {
    sample_digest: String,
    group_identity_digest: String,
    support_score: f64,
    support_score_evidence_digest: String,
    support_score_evidence_class: String,
    target_value: f64,
    target_evidence_digest: String,
    target_evidence_class: String,
    predicted_mean: f64,
    lower_90: f64,
    upper_90: f64,
    lower_95: f64,
    upper_95: f64,
    prediction_digest: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct StratumSummary {
    pub stratum_id: String,
    pub sample_count: usize,
    pub distinct_group_count: usize,
    pub picp_90: f64,
    pub picp_95: f64,
    pub picp_90_wilson_lower_95: f64,
    pub picp_95_wilson_lower_95: f64,
    pub mean_interval_width_90: f64,
    pub mean_interval_width_95: f64,
    pub mae: f64,
    pub rmse: f64,
    pub mean_interval_score_90: f64,
    pub mean_interval_score_95: f64,
    pub undercovered_90: bool,
    pub undercovered_95: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RiskCoveragePoint {
    pub support_score_cutoff: f64,
    pub retained_sample_count: usize,
    pub retained_group_count: usize,
    pub retained_fraction: f64,
    pub mae: f64,
    pub rmse: f64,
    pub worst_group_mae: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SupportCalibrationResult {
    pub manifest_id: String,
    pub test_sample_count: usize,
    pub test_group_count: usize,
    pub calibration_partition: String,
    pub support_method_digest: String,
    pub support_threshold_evidence_digest: String,
    pub calibration_method_digest: String,
    pub strata: Vec<StratumSummary>,
    pub risk_coverage: Vec<RiskCoveragePoint>,
    pub audit_status: String,
    pub extrapolation_undercovered_90: bool,
    pub extrapolation_undercovered_95: bool,
}

#[derive(Debug, Serialize)]
struct CaseAudit {
    case_id: String,
    expectation: String,
    accepted: bool,
    errors: Vec<String>,
    result: Option<SupportCalibrationResult>,
}

#[derive(Debug, Serialize)]
struct ReportPayload {
    corpus_version: String,
    schema_id: String,
    schema_sha256: String,
    canonical_corpus_sha256: String,
    case_count: usize,
    accepted_count: usize,
    rejected_count: usize,
    extrapolation_undercovered_count: usize,
    cases: Vec<CaseAudit>,
}

#[derive(Debug, Serialize)]
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

fn valid_evidence_class(value: &str) -> bool {
    matches!(
        value,
        "physical_observation"
            | "derived_from_physical"
            | "simulation"
            | "synthetic_fixture"
            | "vendor_claim"
    )
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
        return Err("schema identity does not match support calibration v0.1".to_owned());
    }
    if !draft202012::meta::is_valid(&schema) {
        return Err("schema fails Draft 2020-12 meta-schema validation".to_owned());
    }
    draft202012::options()
        .should_validate_formats(true)
        .build(&schema)
        .map_err(|error| format!("schema compilation: {error}"))
}

fn unique_digest_set(values: &[String], label: &str, errors: &mut Vec<String>) -> BTreeSet<String> {
    let mut output = BTreeSet::new();
    for value in values {
        if !is_digest(value) || !output.insert(value.clone()) {
            errors.push(format!("{label} contains an invalid or duplicate identity digest"));
        }
    }
    output
}

fn stratum_for(score: f64, scoring: &SupportScoring) -> &'static str {
    if score <= scoring.in_support_max_score {
        "in_support"
    } else if score < scoring.extrapolation_min_score {
        "boundary"
    } else {
        "extrapolation"
    }
}

fn validate_manifest(manifest: &Manifest) -> Vec<String> {
    let mut errors = Vec::new();
    if manifest.protocol_version != PROTOCOL_VERSION || !nonempty(&manifest.manifest_id) {
        errors.push("protocol version or manifest identity is invalid".to_owned());
    }
    if !is_digest(&manifest.benchmark_manifest_digest)
        || !is_digest(&manifest.dataset_digest)
        || !is_digest(&manifest.split_manifest_digest)
        || !is_digest(&manifest.model_digest)
    {
        errors.push("parent benchmark, dataset, split or model digest is invalid".to_owned());
    }
    if !matches!(
        manifest.evaluation_mode.as_str(),
        "unseen_cell" | "unseen_module" | "unseen_pack" | "unseen_vehicle" | "unseen_lab" | "unseen_domain"
    ) {
        errors.push("support calibration v0.1 requires a group-disjoint unseen-asset evaluation mode".to_owned());
    }
    if manifest.split_frozen_at_ms == 0 {
        errors.push("split freeze timestamp is missing".to_owned());
    }

    let train = unique_digest_set(
        &manifest.train_group_identity_digests,
        "train group identities",
        &mut errors,
    );
    let validation = unique_digest_set(
        &manifest.validation_group_identity_digests,
        "validation group identities",
        &mut errors,
    );
    if train.is_empty() || validation.is_empty() {
        errors.push("train and validation group identity sets must be non-empty".to_owned());
    }
    if !train.is_disjoint(&validation) {
        errors.push("train and validation physical group identities overlap".to_owned());
    }

    let support = &manifest.support_scoring;
    if !nonempty(&support.method_ref)
        || !is_digest(&support.method_digest)
        || !is_digest(&support.feature_schema_digest)
        || support.fit_scope != "train_only"
        || support.reference_partition != "train"
        || support.uses_validation_or_test_labels
        || !nonempty(&support.score_definition)
        || support.score_direction != "higher_is_less_supported"
        || !support.in_support_max_score.is_finite()
        || !support.extrapolation_min_score.is_finite()
        || support.in_support_max_score < 0.0
        || support.extrapolation_min_score <= support.in_support_max_score
        || support.thresholds_frozen_at_ms == 0
        || support.thresholds_frozen_at_ms >= manifest.split_frozen_at_ms
        || !is_digest(&support.threshold_evidence_digest)
    {
        errors.push("support score, fitting scope, label access or frozen thresholds are invalid".to_owned());
    }
    if support.risk_coverage_cutoffs.len() < 3
        || support
            .risk_coverage_cutoffs
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || support
            .risk_coverage_cutoffs
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || support
            .risk_coverage_cutoffs
            .last()
            .is_none_or(|last| *last < support.extrapolation_min_score)
    {
        errors.push("risk-coverage cutoffs must be finite, strictly increasing and reach the extrapolation range".to_owned());
    }

    let calibration = &manifest.calibration;
    let calibration_groups = unique_digest_set(
        &calibration.calibration_group_identity_digests,
        "calibration group identities",
        &mut errors,
    );
    if !nonempty(&calibration.method_ref)
        || !is_digest(&calibration.method_digest)
        || !is_digest(&calibration.calibration_split_digest)
        || calibration.fitted_at_ms == 0
        || calibration.fitted_at_ms > manifest.split_frozen_at_ms
    {
        errors.push("calibration method, split identity or fit timestamp is invalid".to_owned());
    }
    match calibration.partition.as_str() {
        "validation_only" => {
            if calibration.external_source_digest.is_some()
                || !calibration_groups.is_subset(&validation)
            {
                errors.push("validation-only calibration groups must be a subset of validation groups and have no external source digest".to_owned());
            }
        }
        "external_calibration" => {
            if calibration
                .external_source_digest
                .as_deref()
                .is_none_or(|digest| !is_digest(digest))
                || !calibration_groups.is_disjoint(&train)
                || !calibration_groups.is_disjoint(&validation)
            {
                errors.push("external calibration must bind a source digest and remain disjoint from development groups".to_owned());
            }
        }
        _ => errors.push("unknown calibration partition".to_owned()),
    }

    let mut sample_ids = BTreeSet::new();
    let mut test_groups = BTreeSet::new();
    let mut group_strata: BTreeMap<String, String> = BTreeMap::new();
    let mut rows_by_stratum: BTreeMap<String, Vec<&TestEntry>> = BTreeMap::new();
    for entry in &manifest.test_entries {
        if !is_digest(&entry.sample_digest) || !sample_ids.insert(entry.sample_digest.clone()) {
            errors.push("test sample identities contain an invalid or duplicate digest".to_owned());
        }
        if !is_digest(&entry.group_identity_digest) {
            errors.push("test entry has an invalid physical group identity digest".to_owned());
        }
        if !is_digest(&entry.support_score_evidence_digest)
            || !valid_evidence_class(&entry.support_score_evidence_class)
            || !is_digest(&entry.target_evidence_digest)
            || !valid_evidence_class(&entry.target_evidence_class)
            || !is_digest(&entry.prediction_digest)
        {
            errors.push("test entry evidence digest or evidence class is invalid".to_owned());
        }
        if !entry.support_score.is_finite() || entry.support_score < 0.0 {
            errors.push("support score must be finite and non-negative".to_owned());
        }
        if !entry.target_value.is_finite()
            || !(0.0..=1.0).contains(&entry.target_value)
            || !entry.predicted_mean.is_finite()
            || !(0.0..=1.0).contains(&entry.predicted_mean)
            || !entry.lower_90.is_finite()
            || !entry.upper_90.is_finite()
            || !entry.lower_95.is_finite()
            || !entry.upper_95.is_finite()
            || !(0.0..=1.0).contains(&entry.lower_90)
            || !(0.0..=1.0).contains(&entry.upper_90)
            || !(0.0..=1.0).contains(&entry.lower_95)
            || !(0.0..=1.0).contains(&entry.upper_95)
            || !(entry.lower_95 <= entry.lower_90
                && entry.lower_90 <= entry.predicted_mean
                && entry.predicted_mean <= entry.upper_90
                && entry.upper_90 <= entry.upper_95)
        {
            errors.push("target or nested 90/95 prediction intervals are invalid".to_owned());
        }

        let stratum = stratum_for(entry.support_score, support).to_owned();
        if let Some(existing) = group_strata.insert(entry.group_identity_digest.clone(), stratum.clone()) {
            if existing != stratum {
                errors.push("samples from one physical group were assigned to different support strata".to_owned());
            }
        }
        test_groups.insert(entry.group_identity_digest.clone());
        rows_by_stratum.entry(stratum).or_default().push(entry);
    }

    if test_groups.is_empty() {
        errors.push("test set contains no physical group identities".to_owned());
    }
    if !test_groups.is_disjoint(&train) || !test_groups.is_disjoint(&validation) {
        errors.push("test physical group identities overlap train or validation groups".to_owned());
    }
    if calibration.partition == "validation_only" && !calibration_groups.is_disjoint(&test_groups) {
        errors.push("calibration physical group identities overlap test groups".to_owned());
    }
    if calibration.partition == "external_calibration"
        && !calibration_groups.is_disjoint(&test_groups)
    {
        errors.push("external calibration physical groups overlap test groups".to_owned());
    }

    for stratum in ["in_support", "boundary", "extrapolation"] {
        let rows = rows_by_stratum.get(stratum);
        let count = rows.map_or(0, Vec::len);
        let groups = rows.map_or(0, |entries| {
            entries
                .iter()
                .map(|entry| entry.group_identity_digest.as_str())
                .collect::<BTreeSet<_>>()
                .len()
        });
        if count < 2 || groups < 2 {
            errors.push(format!(
                "support stratum {stratum} requires at least two independent test groups"
            ));
        }
    }
    errors
}

fn wilson_lower(successes: usize, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let n = total as f64;
    let z = 1.959963984540054;
    let p = successes as f64 / n;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let centre = p + z2 / (2.0 * n);
    let adjustment = z * ((p * (1.0 - p) / n) + (z2 / (4.0 * n * n))).sqrt();
    ((centre - adjustment) / denominator).max(0.0)
}

fn interval_score(target: f64, lower: f64, upper: f64, nominal: f64) -> f64 {
    let alpha = 1.0 - nominal;
    let width = upper - lower;
    if target < lower {
        width + (2.0 / alpha) * (lower - target)
    } else if target > upper {
        width + (2.0 / alpha) * (target - upper)
    } else {
        width
    }
}

fn summarize_stratum(stratum_id: &str, rows: &[&TestEntry]) -> StratumSummary {
    let count = rows.len();
    let covered_90 = rows
        .iter()
        .filter(|row| row.lower_90 <= row.target_value && row.target_value <= row.upper_90)
        .count();
    let covered_95 = rows
        .iter()
        .filter(|row| row.lower_95 <= row.target_value && row.target_value <= row.upper_95)
        .count();
    let mae = rows
        .iter()
        .map(|row| (row.predicted_mean - row.target_value).abs())
        .sum::<f64>()
        / count as f64;
    let mse = rows
        .iter()
        .map(|row| (row.predicted_mean - row.target_value).powi(2))
        .sum::<f64>()
        / count as f64;
    let groups = rows
        .iter()
        .map(|row| row.group_identity_digest.clone())
        .collect::<BTreeSet<_>>();
    StratumSummary {
        stratum_id: stratum_id.to_owned(),
        sample_count: count,
        distinct_group_count: groups.len(),
        picp_90: covered_90 as f64 / count as f64,
        picp_95: covered_95 as f64 / count as f64,
        picp_90_wilson_lower_95: wilson_lower(covered_90, count),
        picp_95_wilson_lower_95: wilson_lower(covered_95, count),
        mean_interval_width_90: rows
            .iter()
            .map(|row| row.upper_90 - row.lower_90)
            .sum::<f64>()
            / count as f64,
        mean_interval_width_95: rows
            .iter()
            .map(|row| row.upper_95 - row.lower_95)
            .sum::<f64>()
            / count as f64,
        mae,
        rmse: mse.sqrt(),
        mean_interval_score_90: rows
            .iter()
            .map(|row| interval_score(row.target_value, row.lower_90, row.upper_90, 0.90))
            .sum::<f64>()
            / count as f64,
        mean_interval_score_95: rows
            .iter()
            .map(|row| interval_score(row.target_value, row.lower_95, row.upper_95, 0.95))
            .sum::<f64>()
            / count as f64,
        undercovered_90: covered_90 as f64 / count as f64 < UNDER_COVERAGE_90,
        undercovered_95: covered_95 as f64 / count as f64 < UNDER_COVERAGE_95,
    }
}

fn summarize_risk_coverage(manifest: &Manifest) -> Vec<RiskCoveragePoint> {
    let total = manifest.test_entries.len();
    manifest
        .support_scoring
        .risk_coverage_cutoffs
        .iter()
        .map(|cutoff| {
            let retained: Vec<&TestEntry> = manifest
                .test_entries
                .iter()
                .filter(|entry| entry.support_score <= *cutoff)
                .collect();
            let groups = retained
                .iter()
                .map(|entry| entry.group_identity_digest.clone())
                .collect::<BTreeSet<_>>();
            let mae = retained
                .iter()
                .map(|entry| (entry.predicted_mean - entry.target_value).abs())
                .sum::<f64>()
                / retained.len() as f64;
            let mse = retained
                .iter()
                .map(|entry| (entry.predicted_mean - entry.target_value).powi(2))
                .sum::<f64>()
                / retained.len() as f64;
            let mut per_group: BTreeMap<String, Vec<f64>> = BTreeMap::new();
            for entry in &retained {
                per_group
                    .entry(entry.group_identity_digest.clone())
                    .or_default()
                    .push((entry.predicted_mean - entry.target_value).abs());
            }
            let worst_group_mae = per_group
                .values()
                .map(|values| values.iter().sum::<f64>() / values.len() as f64)
                .fold(0.0, f64::max);
            RiskCoveragePoint {
                support_score_cutoff: *cutoff,
                retained_sample_count: retained.len(),
                retained_group_count: groups.len(),
                retained_fraction: retained.len() as f64 / total as f64,
                mae,
                rmse: mse.sqrt(),
                worst_group_mae,
            }
        })
        .collect()
}

fn compute_result(manifest: &Manifest) -> SupportCalibrationResult {
    let mut rows_by_stratum: BTreeMap<String, Vec<&TestEntry>> = BTreeMap::new();
    for entry in &manifest.test_entries {
        rows_by_stratum
            .entry(stratum_for(entry.support_score, &manifest.support_scoring).to_owned())
            .or_default()
            .push(entry);
    }
    let strata: Vec<StratumSummary> = ["in_support", "boundary", "extrapolation"]
        .iter()
        .map(|stratum| summarize_stratum(stratum, &rows_by_stratum[*stratum]))
        .collect();
    let extrapolation = strata
        .iter()
        .find(|summary| summary.stratum_id == "extrapolation")
        .expect("validation requires extrapolation stratum");
    let extrapolation_undercovered_90 = extrapolation.undercovered_90;
    let extrapolation_undercovered_95 = extrapolation.undercovered_95;
    let any_undercovered = strata
        .iter()
        .any(|summary| summary.undercovered_90 || summary.undercovered_95);
    SupportCalibrationResult {
        manifest_id: manifest.manifest_id.clone(),
        test_sample_count: manifest.test_entries.len(),
        test_group_count: manifest
            .test_entries
            .iter()
            .map(|entry| entry.group_identity_digest.clone())
            .collect::<BTreeSet<_>>()
            .len(),
        calibration_partition: manifest.calibration.partition.clone(),
        support_method_digest: manifest.support_scoring.method_digest.clone(),
        support_threshold_evidence_digest: manifest.support_scoring.threshold_evidence_digest.clone(),
        calibration_method_digest: manifest.calibration.method_digest.clone(),
        strata,
        risk_coverage: summarize_risk_coverage(manifest),
        audit_status: if any_undercovered {
            "SUPPORT_STRATUM_UNDERCOVERAGE_DETECTED".to_owned()
        } else {
            "SUPPORT_CONDITIONAL_COVERAGE_WITHIN_TOLERANCE".to_owned()
        },
        extrapolation_undercovered_90,
        extrapolation_undercovered_95,
    }
}

fn canonicalize_corpus(value: &mut Value) {
    let Some(cases) = value.get_mut("cases").and_then(Value::as_array_mut) else {
        return;
    };
    cases.sort_by(|left, right| {
        left.get("case_id")
            .and_then(Value::as_str)
            .cmp(&right.get("case_id").and_then(Value::as_str))
    });
    for case in cases {
        if let Some(manifest) = case.get_mut("manifest") {
            for field in ["train_group_identity_digests", "validation_group_identity_digests"] {
                if let Some(rows) = manifest.get_mut(field).and_then(Value::as_array_mut) {
                    rows.sort_by(|left, right| left.as_str().cmp(&right.as_str()));
                }
            }
            if let Some(rows) = manifest
                .get_mut("calibration")
                .and_then(|calibration| calibration.get_mut("calibration_group_identity_digests"))
                .and_then(Value::as_array_mut)
            {
                rows.sort_by(|left, right| left.as_str().cmp(&right.as_str()));
            }
            if let Some(rows) = manifest.get_mut("test_entries").and_then(Value::as_array_mut) {
                rows.sort_by(|left, right| {
                    left.get("sample_digest")
                        .and_then(Value::as_str)
                        .cmp(&right.get("sample_digest").and_then(Value::as_str))
                });
            }
        }
    }
}

pub fn audit_corpus(schema_bytes: &[u8], corpus_bytes: &[u8]) -> Result<CorpusReport, String> {
    let validator = schema_validator(schema_bytes)?;
    let mut corpus_value: Value =
        serde_json::from_slice(corpus_bytes).map_err(|error| format!("corpus JSON: {error}"))?;
    canonicalize_corpus(&mut corpus_value);
    let canonical = serde_json::to_vec(&corpus_value)
        .map_err(|error| format!("canonical corpus serialization: {error}"))?;
    let corpus: Corpus =
        serde_json::from_slice(&canonical).map_err(|error| format!("typed corpus: {error}"))?;
    if corpus.corpus_version != CORPUS_VERSION || corpus.schema_id != SCHEMA_ID {
        return Err("corpus version or schema identity mismatch".to_owned());
    }

    let mut seen = BTreeSet::new();
    let mut cases = Vec::new();
    let mut accepted_count = 0usize;
    let mut rejected_count = 0usize;
    let mut extrapolation_undercovered_count = 0usize;

    for case in corpus.cases {
        if !nonempty(&case.case_id) || !seen.insert(case.case_id.clone()) {
            return Err(format!("case identity is empty or duplicated: {}", case.case_id));
        }
        if !matches!(case.expectation.as_str(), "accept" | "reject") {
            return Err(format!("case {} has unknown expectation", case.case_id));
        }
        let schema_errors: Vec<String> = validator
            .iter_errors(&case.manifest)
            .map(|error| error.to_string())
            .collect();
        let outcome = if schema_errors.is_empty() {
            match serde_json::from_value::<Manifest>(case.manifest) {
                Ok(manifest) => {
                    let errors = validate_manifest(&manifest);
                    if errors.is_empty() {
                        Ok(compute_result(&manifest))
                    } else {
                        Err(errors)
                    }
                }
                Err(error) => Err(vec![format!("typed manifest parse: {error}")]),
            }
        } else {
            Err(schema_errors)
        };
        let accepted = outcome.is_ok();
        if accepted != (case.expectation == "accept") {
            let detail = match &outcome {
                Ok(_) => "unexpected acceptance".to_owned(),
                Err(errors) => errors.join("; "),
            };
            return Err(format!("case {} expectation mismatch: {}", case.case_id, detail));
        }
        let (result, errors) = match outcome {
            Ok(result) => {
                if case.expected_extrapolation_undercovered
                    != Some(result.extrapolation_undercovered_95)
                {
                    return Err(format!(
                        "case {} extrapolation coverage status differs from its known-truth expectation",
                        case.case_id
                    ));
                }
                if result.extrapolation_undercovered_90 || result.extrapolation_undercovered_95 {
                    extrapolation_undercovered_count += 1;
                }
                accepted_count += 1;
                (Some(result), Vec::new())
            }
            Err(errors) => {
                if case.expected_extrapolation_undercovered.is_some() {
                    return Err(format!(
                        "rejected case {} must not declare an extrapolation result",
                        case.case_id
                    ));
                }
                rejected_count += 1;
                (None, errors)
            }
        };
        cases.push(CaseAudit {
            case_id: case.case_id,
            expectation: case.expectation,
            accepted,
            errors,
            result,
        });
    }

    let payload = ReportPayload {
        corpus_version: CORPUS_VERSION.to_owned(),
        schema_id: SCHEMA_ID.to_owned(),
        schema_sha256: sha256(schema_bytes),
        canonical_corpus_sha256: sha256(&canonical),
        case_count: cases.len(),
        accepted_count,
        rejected_count,
        extrapolation_undercovered_count,
        cases,
    };
    let bytes =
        serde_json::to_vec(&payload).map_err(|error| format!("report serialization: {error}"))?;
    Ok(CorpusReport {
        payload,
        report_sha256: sha256(&bytes),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const SCHEMA: &[u8] =
        include_bytes!("../../../schemas/transport/ev_soh_support_calibration_v0_1.schema.json");
    const CORPUS: &[u8] =
        include_bytes!("../../../fixtures/ev_battery/soh_support_calibration_scenarios_v0_1.json");

    fn report() -> CorpusReport {
        audit_corpus(SCHEMA, CORPUS).expect("frozen support calibration corpus")
    }

    #[test]
    fn frozen_corpus_accepts_two_and_rejects_eight_controls() {
        let report = report();
        assert_eq!(report.payload.case_count, 10);
        assert_eq!(report.payload.accepted_count, 2);
        assert_eq!(report.payload.rejected_count, 8);
        assert_eq!(report.payload.extrapolation_undercovered_count, 1);
    }

    #[test]
    fn marginally_good_but_ood_undercovered_model_is_flagged_not_hidden() {
        let report = report();
        let case = report.payload.cases.iter()
            .find(|case| case.case_id == "marginal-good-but-extrapolation-undercovered")
            .and_then(|case| case.result.as_ref())
            .expect("OOD undercoverage result");
        let extrapolation = case.strata.iter()
            .find(|summary| summary.stratum_id == "extrapolation")
            .expect("extrapolation stratum");
        assert!(extrapolation.undercovered_95);
        assert!(case.extrapolation_undercovered_95);
        assert_eq!(case.audit_status, "SUPPORT_STRATUM_UNDERCOVERAGE_DETECTED");
    }

    #[test]
    fn risk_coverage_retention_is_monotonic_and_replayable() {
        let report = report();
        let case = report.payload.cases.iter()
            .find(|case| case.case_id == "valid-support-conditioned-calibration")
            .and_then(|case| case.result.as_ref())
            .expect("valid support calibration");
        assert!(case.risk_coverage.windows(2).all(|pair| {
            pair[0].retained_sample_count <= pair[1].retained_sample_count
                && pair[0].retained_fraction <= pair[1].retained_fraction
        }));
        assert_eq!(case.risk_coverage.last().unwrap().retained_sample_count, 6);
    }

    #[test]
    fn calibration_and_test_assets_must_be_physically_disjoint() {
        let report = report();
        let case = report.payload.cases.iter()
            .find(|case| case.case_id == "calibration-group-overlaps-test")
            .expect("calibration leakage control");
        assert!(!case.accepted);
    }

    #[test]
    fn support_thresholds_must_be_frozen_before_test_split() {
        let report = report();
        let case = report.payload.cases.iter()
            .find(|case| case.case_id == "support-threshold-frozen-after-split")
            .expect("late threshold control");
        assert!(!case.accepted);
    }

    #[test]
    fn support_score_must_be_label_free_and_train_fitted() {
        let report = report();
        let case = report.payload.cases.iter()
            .find(|case| case.case_id == "support-score-uses-test-labels")
            .expect("test label leakage control");
        assert!(!case.accepted);
    }

    #[test]
    fn group_and_sample_identity_leakage_is_rejected() {
        let report = report();
        for case_id in [
            "duplicate-sample-identity",
            "test-group-overlaps-validation",
        ] {
            let case = report.payload.cases.iter()
                .find(|case| case.case_id == case_id)
                .expect("identity leakage control");
            assert!(!case.accepted);
        }
    }

    #[test]
    fn incomplete_support_strata_and_invalid_intervals_are_rejected() {
        let report = report();
        for case_id in [
            "no-extrapolation-test-stratum",
            "non-nested-prediction-intervals",
            "nonmonotonic-risk-coverage-cutoffs",
        ] {
            let case = report.payload.cases.iter()
                .find(|case| case.case_id == case_id)
                .expect("support/calibration control");
            assert!(!case.accepted);
        }
    }

    #[test]
    fn canonical_report_is_invariant_to_case_group_and_sample_order() {
        let original = report();
        let mut value: Value = serde_json::from_slice(CORPUS).expect("corpus JSON");
        value["cases"].as_array_mut().unwrap().reverse();
        for case in value["cases"].as_array_mut().unwrap() {
            case["manifest"]["train_group_identity_digests"].as_array_mut().unwrap().reverse();
            case["manifest"]["validation_group_identity_digests"].as_array_mut().unwrap().reverse();
            case["manifest"]["calibration"]["calibration_group_identity_digests"]
                .as_array_mut().unwrap().reverse();
            case["manifest"]["test_entries"].as_array_mut().unwrap().reverse();
        }
        let bytes = serde_json::to_vec(&value).expect("permuted support corpus");
        let replay = audit_corpus(SCHEMA, &bytes).expect("permuted support audit");
        assert_eq!(original.report_sha256, replay.report_sha256);
        assert_eq!(
            original.payload.canonical_corpus_sha256,
            replay.payload.canonical_corpus_sha256
        );
    }
}
