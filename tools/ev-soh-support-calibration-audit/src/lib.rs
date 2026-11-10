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
const MIN_INDEPENDENT_GROUPS_PER_STRATUM_V0_1: usize = 30;
const MIN_GROUP_EXTRAPOLATION_RECALL_V0_1: f64 = 0.90;
const MAX_GROUP_FALSE_EXTRAPOLATION_RATE_V0_1: f64 = 0.10;

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
    expected_audit_status: Option<String>,
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
    test_accessed_at_ms: u64,
    minimum_independent_groups_per_stratum: usize,
    reference_support: ReferenceSupport,
    train_group_identity_digests: Vec<String>,
    validation_group_identity_digests: Vec<String>,
    support_scoring: SupportScoring,
    calibration: Calibration,
    test_entries: Vec<TestEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceSupport {
    method_ref: String,
    method_digest: String,
    feature_schema_digest: String,
    fit_scope: String,
    reference_partition: String,
    uses_validation_or_test_labels: bool,
    uses_test_targets: bool,
    score_definition: String,
    score_direction: String,
    in_support_max_score: f64,
    extrapolation_min_score: f64,
    thresholds_frozen_at_ms: u64,
    threshold_evidence_digest: String,
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
    observation_time_ms: u64,
    reference_support_score: f64,
    reference_support_state: String,
    reference_support_evidence_digest: String,
    reference_support_evidence_class: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct StratumSummary {
    pub stratum_id: String,
    pub sample_count: usize,
    pub distinct_group_count: usize,
    pub picp_90: f64,
    pub picp_95: f64,
    pub group_coverage_pass_rate_90: f64,
    pub group_coverage_pass_rate_95: f64,
    pub group_coverage_pass_rate_90_wilson_lower_95: f64,
    pub group_coverage_pass_rate_95_wilson_lower_95: f64,
    pub evidence_status: String,
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
pub struct SupportDetectionSummary {
    pub reference_extrapolation_group_count: usize,
    pub predicted_extrapolation_group_count: usize,
    pub true_positive_group_count: usize,
    pub false_negative_group_count: usize,
    pub false_positive_group_count: usize,
    pub true_negative_group_count: usize,
    pub groups_with_both_support_states: usize,
    pub extrapolation_recall: f64,
    pub false_extrapolation_rate: f64,
    pub extrapolation_auroc: Option<f64>,
    pub within_group_auc_group_count: usize,
    pub sample_extrapolation_recall: f64,
    pub sample_false_extrapolation_rate: f64,
    pub sample_extrapolation_auroc: f64,
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
    pub support_detection: SupportDetectionSummary,
    pub strata: Vec<StratumSummary>,
    pub risk_coverage: Vec<RiskCoveragePoint>,
    pub audit_status: String,
    pub extrapolation_undercovered_90: bool,
    pub extrapolation_undercovered_95: bool,
    pub support_detector_failed: bool,
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
    support_detector_failure_count: usize,
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
    if manifest.split_frozen_at_ms == 0
        || manifest.test_accessed_at_ms <= manifest.split_frozen_at_ms
        || manifest.minimum_independent_groups_per_stratum < MIN_INDEPENDENT_GROUPS_PER_STRATUM_V0_1
    {
        errors.push("split/test-access timing or independent-group evidence minimum is invalid".to_owned());
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

    let reference = &manifest.reference_support;
    if !nonempty(&reference.method_ref)
        || !is_digest(&reference.method_digest)
        || !is_digest(&reference.feature_schema_digest)
        || reference.fit_scope != "train_only"
        || reference.reference_partition != "train"
        || reference.uses_validation_or_test_labels
        || reference.uses_test_targets
        || !nonempty(&reference.score_definition)
        || reference.score_direction != "higher_is_less_supported"
        || !reference.in_support_max_score.is_finite()
        || !reference.extrapolation_min_score.is_finite()
        || reference.in_support_max_score < 0.0
        || reference.extrapolation_min_score <= reference.in_support_max_score
        || reference.thresholds_frozen_at_ms == 0
        || reference.thresholds_frozen_at_ms >= manifest.split_frozen_at_ms
        || !is_digest(&reference.threshold_evidence_digest)
    {
        errors.push("offline reference-support method must be train-fitted, target-blind, and frozen before the split".to_owned());
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
        || calibration.fitted_at_ms <= manifest.split_frozen_at_ms
        || calibration.fitted_at_ms >= manifest.test_accessed_at_ms
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
    let mut observation_times: BTreeMap<String, Vec<u64>> = BTreeMap::new();
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
            || !matches!(entry.reference_support_state.as_str(), "in_support" | "boundary" | "extrapolation")
            || !is_digest(&entry.reference_support_evidence_digest)
            || !valid_evidence_class(&entry.reference_support_evidence_class)
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

        // Reference support state belongs to each observation, not permanently
        // to a physical cell. A trajectory may cross support boundaries as it ages.
        if entry.observation_time_ms == 0
            || !entry.reference_support_score.is_finite()
            || entry.reference_support_score < 0.0
        {
            errors.push("reference-support observation time or numeric score is invalid".to_owned());
        }
        let expected_reference_state = if entry.reference_support_score <= reference.in_support_max_score {
            "in_support"
        } else if entry.reference_support_score < reference.extrapolation_min_score {
            "boundary"
        } else {
            "extrapolation"
        };
        if entry.reference_support_state != expected_reference_state {
            errors.push("reference-support categorical state does not replay from its independent numeric score".to_owned());
        }
        observation_times
            .entry(entry.group_identity_digest.clone())
            .or_default()
            .push(entry.observation_time_ms);
        let stratum = entry.reference_support_state.clone();
        test_groups.insert(entry.group_identity_digest.clone());
        rows_by_stratum.entry(stratum).or_default().push(entry);
    }

    for (group, times) in &mut observation_times {
        times.sort_unstable();
        if times.windows(2).any(|pair| pair[0] == pair[1]) {
            errors.push(format!("physical group {group} has duplicate observation timestamps"));
        }
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
                "reference support stratum {stratum} requires at least two independent test groups"
            ));
        }
    }
    if let (Some(first), Some(last)) = (
        support.risk_coverage_cutoffs.first(),
        support.risk_coverage_cutoffs.last(),
    ) {
        if !manifest.test_entries.iter().any(|entry| entry.support_score <= *first)
            || manifest.test_entries.iter().any(|entry| entry.support_score > *last)
        {
            errors.push("risk-coverage cutoffs must retain at least one sample initially and all samples at the final point".to_owned());
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

fn summarize_stratum(
    stratum_id: &str,
    rows: &[&TestEntry],
    minimum_independent_groups: usize,
) -> StratumSummary {
    let count = rows.len();
    let covered_90 = rows.iter().filter(|row| {
        row.lower_90 <= row.target_value && row.target_value <= row.upper_90
    }).count();
    let covered_95 = rows.iter().filter(|row| {
        row.lower_95 <= row.target_value && row.target_value <= row.upper_95
    }).count();
    let mae = rows.iter().map(|row| (row.predicted_mean - row.target_value).abs())
        .sum::<f64>() / count as f64;
    let mse = rows.iter().map(|row| (row.predicted_mean - row.target_value).powi(2))
        .sum::<f64>() / count as f64;
    let mut per_group: BTreeMap<String, Vec<&TestEntry>> = BTreeMap::new();
    for row in rows {
        per_group.entry(row.group_identity_digest.clone()).or_default().push(row);
    }
    let groups = per_group.len();
    let mut group_pass_90 = 0usize;
    let mut group_pass_95 = 0usize;
    for entries in per_group.values() {
        let c90 = entries.iter().filter(|row| {
            row.lower_90 <= row.target_value && row.target_value <= row.upper_90
        }).count() as f64 / entries.len() as f64;
        let c95 = entries.iter().filter(|row| {
            row.lower_95 <= row.target_value && row.target_value <= row.upper_95
        }).count() as f64 / entries.len() as f64;
        if c90 >= 0.90 { group_pass_90 += 1; }
        if c95 >= 0.95 { group_pass_95 += 1; }
    }
    let picp_90 = covered_90 as f64 / count as f64;
    let picp_95 = covered_95 as f64 / count as f64;
    let undercovered_90 = picp_90 < UNDER_COVERAGE_90;
    let undercovered_95 = picp_95 < UNDER_COVERAGE_95;
    let evidence_status = if undercovered_90 || undercovered_95 {
        "OBSERVED_UNDERCOVERAGE"
    } else if groups < minimum_independent_groups {
        "INSUFFICIENT_PHYSICAL_GROUP_EVIDENCE"
    } else {
        "OBSERVED_COVERAGE_WITHIN_TOLERANCE"
    };
    StratumSummary {
        stratum_id: stratum_id.to_owned(),
        sample_count: count,
        distinct_group_count: groups,
        picp_90,
        picp_95,
        group_coverage_pass_rate_90: group_pass_90 as f64 / groups as f64,
        group_coverage_pass_rate_95: group_pass_95 as f64 / groups as f64,
        group_coverage_pass_rate_90_wilson_lower_95: wilson_lower(group_pass_90, groups),
        group_coverage_pass_rate_95_wilson_lower_95: wilson_lower(group_pass_95, groups),
        evidence_status: evidence_status.to_owned(),
        mean_interval_width_90: rows.iter().map(|row| row.upper_90 - row.lower_90)
            .sum::<f64>() / count as f64,
        mean_interval_width_95: rows.iter().map(|row| row.upper_95 - row.lower_95)
            .sum::<f64>() / count as f64,
        mae,
        rmse: mse.sqrt(),
        mean_interval_score_90: rows.iter().map(|row| {
            interval_score(row.target_value, row.lower_90, row.upper_90, 0.90)
        }).sum::<f64>() / count as f64,
        mean_interval_score_95: rows.iter().map(|row| {
            interval_score(row.target_value, row.lower_95, row.upper_95, 0.95)
        }).sum::<f64>() / count as f64,
        undercovered_90,
        undercovered_95,
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

fn auc(positive_scores: &[f64], negative_scores: &[f64]) -> f64 {
    let mut wins = 0.0;
    for positive in positive_scores {
        for negative in negative_scores {
            if positive > negative { wins += 1.0; }
            else if (positive - negative).abs() <= EPSILON { wins += 0.5; }
        }
    }
    wins / (positive_scores.len() * negative_scores.len()) as f64
}

fn compute_support_detection(manifest: &Manifest) -> SupportDetectionSummary {
    // Sample-level metrics are retained as descriptive diagnostics. Their
    // windows are correlated; the pass/fail summary below macro-averages
    // within each independent physical group instead of treating windows as
    // independent batteries.
    let mut sample_tp = 0usize;
    let mut sample_fn = 0usize;
    let mut sample_fp = 0usize;
    let mut sample_tn = 0usize;
    let mut sample_pos = Vec::new();
    let mut sample_neg = Vec::new();

    // Keep reference-positive and reference-negative observations separate
    // inside each physical group. A cell can cross from in-support to
    // extrapolation as it ages; collapsing the entire cell to one binary
    // label (e.g. "ever OOD") would make its earlier in-support observations
    // disappear from the false-positive denominator.
    let mut by_group: BTreeMap<String, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for entry in &manifest.test_entries {
        let actual_ood = entry.reference_support_state == "extrapolation";
        let predicted_ood =
            entry.support_score >= manifest.support_scoring.extrapolation_min_score;
        match (actual_ood, predicted_ood) {
            (true, true) => sample_tp += 1,
            (true, false) => sample_fn += 1,
            (false, true) => sample_fp += 1,
            (false, false) => sample_tn += 1,
        }
        if actual_ood {
            sample_pos.push(entry.support_score);
        } else {
            sample_neg.push(entry.support_score);
        }
        let group = by_group
            .entry(entry.group_identity_digest.clone())
            .or_insert_with(|| (Vec::new(), Vec::new()));
        if actual_ood {
            group.0.push(entry.support_score);
        } else {
            group.1.push(entry.support_score);
        }
    }

    let threshold = manifest.support_scoring.extrapolation_min_score;
    let mut positive_groups = 0usize;
    let mut negative_groups = 0usize;
    let mut tp_groups = 0usize;
    let mut fn_groups = 0usize;
    let mut fp_negative_only_groups = 0usize;
    let mut tn_negative_only_groups = 0usize;
    let mut recall_sum = 0.0;
    let mut false_rate_sum = 0.0;
    let mut within_group_auc_sum = 0.0;
    let mut within_group_auc_count = 0usize;

    for (_group_id, (positive_scores, negative_scores)) in &by_group {
        let group_has_positive = !positive_scores.is_empty();
        let group_has_negative = !negative_scores.is_empty();

        if group_has_positive {
            positive_groups += 1;
            let detected = positive_scores
                .iter()
                .filter(|score| **score >= threshold)
                .count();
            let group_recall = detected as f64 / positive_scores.len() as f64;
            recall_sum += group_recall;
            if detected > 0 {
                tp_groups += 1;
            } else {
                fn_groups += 1;
            }
        }

        if group_has_negative {
            negative_groups += 1;
            let false_alarms = negative_scores
                .iter()
                .filter(|score| **score >= threshold)
                .count();
            let group_false_rate = false_alarms as f64 / negative_scores.len() as f64;
            false_rate_sum += group_false_rate;
            // The confusion counts below describe negative-only groups. Mixed
            // temporal groups are represented by their within-group rates.
            if !group_has_positive {
                if false_alarms > 0 {
                    fp_negative_only_groups += 1;
                } else {
                    tn_negative_only_groups += 1;
                }
            }
        }

        if group_has_positive && group_has_negative {
            within_group_auc_sum += auc(positive_scores, negative_scores);
            within_group_auc_count += 1;
        }
    }

    let sample_positives = sample_pos.len();
    let sample_negatives = sample_neg.len();
    let within_group_auroc = (within_group_auc_count > 0)
        .then_some(within_group_auc_sum / within_group_auc_count as f64);

    SupportDetectionSummary {
        reference_extrapolation_group_count: positive_groups,
        predicted_extrapolation_group_count: by_group
            .values()
            .filter(|(positive, negative)| {
                positive.iter().chain(negative.iter()).any(|score| *score >= threshold)
            })
            .count(),
        true_positive_group_count: tp_groups,
        false_negative_group_count: fn_groups,
        false_positive_group_count: fp_negative_only_groups,
        true_negative_group_count: tn_negative_only_groups,
        groups_with_both_support_states: within_group_auc_count,
        extrapolation_recall: recall_sum / positive_groups as f64,
        false_extrapolation_rate: false_rate_sum / negative_groups as f64,
        // Average the per-cell AUC only across cells with both reference
        // support states. If no cell crosses the support boundary during the
        // observed trajectory, report null rather than inventing an AUC.
        extrapolation_auroc: within_group_auroc,
        within_group_auc_group_count: within_group_auc_count,
        sample_extrapolation_recall: sample_tp as f64 / sample_positives as f64,
        sample_false_extrapolation_rate: sample_fp as f64 / sample_negatives as f64,
        sample_extrapolation_auroc: auc(&sample_pos, &sample_neg),
    }
}

fn compute_result(manifest: &Manifest) -> SupportCalibrationResult {
    let mut rows_by_stratum: BTreeMap<String, Vec<&TestEntry>> = BTreeMap::new();
    for entry in &manifest.test_entries {
        rows_by_stratum.entry(entry.reference_support_state.clone()).or_default().push(entry);
    }
    let strata: Vec<StratumSummary> = ["in_support", "boundary", "extrapolation"]
        .iter()
        .map(|stratum| summarize_stratum(
            stratum,
            &rows_by_stratum[*stratum],
            manifest.minimum_independent_groups_per_stratum,
        ))
        .collect();
    let extrapolation = strata.iter()
        .find(|summary| summary.stratum_id == "extrapolation")
        .expect("validation requires reference extrapolation stratum");
    let extrapolation_undercovered_90 = extrapolation.undercovered_90;
    let extrapolation_undercovered_95 = extrapolation.undercovered_95;
    let any_undercovered = strata.iter()
        .any(|summary| summary.undercovered_90 || summary.undercovered_95);
    let support_detection = compute_support_detection(manifest);
    let support_detector_failed = support_detection.extrapolation_recall < MIN_GROUP_EXTRAPOLATION_RECALL_V0_1
        || support_detection.false_extrapolation_rate > MAX_GROUP_FALSE_EXTRAPOLATION_RATE_V0_1
        || support_detection.sample_extrapolation_recall < MIN_GROUP_EXTRAPOLATION_RECALL_V0_1
        || support_detection.sample_false_extrapolation_rate > MAX_GROUP_FALSE_EXTRAPOLATION_RATE_V0_1;
    let insufficient_groups = strata.iter().any(|summary| {
        summary.distinct_group_count < manifest.minimum_independent_groups_per_stratum
    });
    let audit_status = if any_undercovered && support_detector_failed {
        "OOD_MISDETECTION_AND_UNDERCOVERAGE"
    } else if any_undercovered {
        "SUPPORT_STRATUM_UNDERCOVERAGE_DETECTED"
    } else if support_detector_failed {
        "SUPPORT_DETECTOR_FAILED"
    } else if insufficient_groups {
        "INSUFFICIENT_PHYSICAL_GROUP_EVIDENCE"
    } else {
        "SUPPORT_CONDITIONAL_COVERAGE_WITHIN_TOLERANCE"
    };
    SupportCalibrationResult {
        manifest_id: manifest.manifest_id.clone(),
        test_sample_count: manifest.test_entries.len(),
        test_group_count: manifest.test_entries.iter()
            .map(|entry| entry.group_identity_digest.clone())
            .collect::<BTreeSet<_>>().len(),
        calibration_partition: manifest.calibration.partition.clone(),
        support_method_digest: manifest.support_scoring.method_digest.clone(),
        support_threshold_evidence_digest: manifest.support_scoring.threshold_evidence_digest.clone(),
        calibration_method_digest: manifest.calibration.method_digest.clone(),
        support_detection,
        strata,
        risk_coverage: summarize_risk_coverage(manifest),
        audit_status: audit_status.to_owned(),
        extrapolation_undercovered_90,
        extrapolation_undercovered_95,
        support_detector_failed,
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
    let mut support_detector_failure_count = 0usize;

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
                if result.support_detector_failed {
                    support_detector_failure_count += 1;
                }
                if case.expected_audit_status.as_deref() != Some(result.audit_status.as_str()) {
                    return Err(format!(
                        "case {} audit status differs from expected status: {}",
                        case.case_id, result.audit_status
                    ));
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
        support_detector_failure_count,
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
    fn frozen_corpus_accepts_four_and_rejects_twelve_controls() {
        let report = report();
        assert_eq!(report.payload.case_count, 16);
        assert_eq!(report.payload.accepted_count, 4);
        assert_eq!(report.payload.rejected_count, 12);
        assert_eq!(report.payload.extrapolation_undercovered_count, 2);
        assert_eq!(report.payload.support_detector_failure_count, 1);
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
    fn one_physical_cell_may_cross_support_strata_over_its_aging_trajectory() {
        let report = report();
        let case = report.payload.cases.iter()
            .find(|case| case.case_id == "valid-support-conditioned-calibration")
            .and_then(|case| case.result.as_ref())
            .expect("temporal support crossing fixture");
        assert_eq!(case.audit_status, "INSUFFICIENT_PHYSICAL_GROUP_EVIDENCE");
        let manifest: Manifest = serde_json::from_slice(
            &serde_json::to_vec(
                &serde_json::from_slice::<Value>(CORPUS).unwrap()["cases"]
                    .as_array().unwrap().iter()
                    .find(|value| value["case_id"] == "valid-support-conditioned-calibration")
                    .unwrap()["manifest"]
            ).unwrap()
        ).unwrap();
        let first_group = &manifest.test_entries[0].group_identity_digest;
        let states: BTreeSet<&str> = manifest.test_entries.iter()
            .filter(|entry| &entry.group_identity_digest == first_group)
            .map(|entry| entry.reference_support_state.as_str())
            .collect();
        assert!(states.contains("in_support"));
        assert!(states.contains("extrapolation"));
    }

    #[test]
    fn assets_that_each_cross_support_states_have_estimable_within_group_metrics() {
        let report = report();
        let result = report.payload.cases.iter()
            .find(|case| case.case_id == "all-physical-groups-cross-support-states")
            .and_then(|case| case.result.as_ref())
            .expect("all-groups-crossing fixture");
        assert_eq!(result.support_detection.groups_with_both_support_states, 6);
        assert_eq!(result.support_detection.within_group_auc_group_count, 6);
        assert_eq!(result.support_detection.extrapolation_auroc, Some(1.0));
        assert_eq!(result.support_detection.extrapolation_recall, 1.0);
        assert_eq!(result.support_detection.false_extrapolation_rate, 0.0);
        assert_eq!(result.audit_status, "INSUFFICIENT_PHYSICAL_GROUP_EVIDENCE");
    }

    #[test]
    fn within_group_auc_is_optional_when_no_asset_has_both_reference_states() {
        let report = report();
        let result = report.payload.cases.iter()
            .find(|case| case.case_id == "marginal-good-but-extrapolation-undercovered")
            .and_then(|case| case.result.as_ref())
            .expect("separate-assets support fixture");
        assert_eq!(result.support_detection.within_group_auc_group_count, 0);
        assert_eq!(result.support_detection.extrapolation_auroc, None);
    }

    #[test]
    fn undercoverage_with_a_support_detector_false_negative_is_distinguished() {
        let report = report();
        let case = report.payload.cases.iter()
            .find(|case| case.case_id == "ood-undercoverage-with-support-detector-false-negative")
            .and_then(|case| case.result.as_ref())
            .expect("detector miss case");
        assert!(case.extrapolation_undercovered_95);
        assert!(case.support_detector_failed);
        assert!(case.support_detection.false_negative_group_count > 0);
        assert_eq!(case.audit_status, "OOD_MISDETECTION_AND_UNDERCOVERAGE");
    }

    #[test]
    fn tiny_synthetic_group_count_is_never_described_as_calibrated_evidence() {
        let report = report();
        let case = report.payload.cases.iter()
            .find(|case| case.case_id == "valid-support-conditioned-calibration")
            .and_then(|case| case.result.as_ref())
            .expect("valid calibration fixture");
        assert_eq!(case.audit_status, "INSUFFICIENT_PHYSICAL_GROUP_EVIDENCE");
        assert!(case.strata.iter().all(|stratum| {
            stratum.distinct_group_count < 30
                && stratum.evidence_status == "INSUFFICIENT_PHYSICAL_GROUP_EVIDENCE"
        }));
    }

    #[test]
    fn reference_support_scores_replay_categories_and_cannot_use_targets() {
        let report = report();
        let case = report.payload.cases.iter()
            .find(|case| case.case_id == "reference-support-state-score-mismatch-rejected")
            .expect("reference score/state mismatch");
        assert!(!case.accepted);
        let leakage = report.payload.cases.iter()
            .find(|case| case.case_id == "reference-support-method-uses-target-labels-rejected")
            .expect("reference support target leakage");
        assert!(!leakage.accepted);
    }

    #[test]
    fn physical_observation_times_must_be_unique_within_each_group() {
        let report = report();
        let case = report.payload.cases.iter()
            .find(|case| case.case_id == "duplicate-observation-time-within-physical-cell-rejected")
            .expect("duplicate observation time control");
        assert!(!case.accepted);
        assert!(case.errors.iter().any(|error| error.contains("duplicate observation timestamps")));
    }

    #[test]
    fn independent_group_evidence_floor_cannot_be_lowered_by_manifest_author() {
        let report = report();
        let case = report.payload.cases.iter()
            .find(|case| case.case_id == "minimum-independent-groups-below-protocol-floor")
            .expect("minimum-group floor control");
        assert!(!case.accepted);
        assert!(case.errors.iter().any(|error| error.contains("independent-group evidence minimum")));
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
