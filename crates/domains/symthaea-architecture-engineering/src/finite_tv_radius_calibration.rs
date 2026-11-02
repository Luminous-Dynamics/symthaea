//! Finite-sample total-variation radius calibration for finite path supports.
//!
//! For every policy/model pair, calibration observations are independent,
//! identically distributed complete-path draws on a known finite categorical
//! support. A Weissman-style L1 deviation bound gives a confidence radius
//! around the empirical law. The triangle inequality adds its distance from
//! the frozen nominal path law. A Bonferroni allocation over every row yields
//! simultaneous coverage for the complete policy/model family.
//!
//! These assumptions are checked as declared metadata, not proven by this
//! module. Digests bind content identity but do not authenticate provenance,
//! independence, IID sampling, or support semantics.

use super::policy_frontier_lp::PolicyFrontierPlan;
use super::tv_robust_policy_frontier::{
    TvAlternativePathEnvelope, TvAmbiguityEnvelope, TvNullPathEnvelope, TvRobustnessError,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, error::Error, fmt};

const REQUEST_SCHEMA: &str = "symthaea.architecture-engineering.finite-tv-calibration-request";
const REQUEST_VERSION: &str = "0.1.0";
const REPORT_SCHEMA: &str = "symthaea.architecture-engineering.finite-tv-calibration-report";
const REPORT_VERSION: &str = "0.1.0";
const MAX_PATH_SUPPORT: usize = 100_000;
const MAX_EXACT_F64_INTEGER: u64 = 9_007_199_254_740_992;
const PROBABILITY_SUM_TOLERANCE: f64 = 1e-10;
const LN_2: f64 = std::f64::consts::LN_2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FiniteTvCalibrationError {
    pub path: String,
    pub message: String,
}

impl FiniteTvCalibrationError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self { path: path.into(), message: message.into() }
    }

    fn from_robustness(error: TvRobustnessError) -> Self {
        Self { path: error.path, message: error.message }
    }

    fn from_plan(path: impl Into<String>, error: impl fmt::Display) -> Self {
        Self { path: path.into(), message: error.to_string() }
    }
}

impl fmt::Display for FiniteTvCalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for FiniteTvCalibrationError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FiniteTvPathSample {
    pub policy_id: String,
    /// Digest of the canonical finite support and its category ordering.
    pub path_support_sha256: String,
    /// Digest of the calibration data artifact from which these counts came.
    pub calibration_data_sha256: String,
    /// Counts over the frozen ordered path support.
    pub observed_path_counts: Vec<u64>,
    /// Frozen nominal probabilities over exactly the same ordered support.
    pub nominal_path_probabilities: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FiniteTvNullModelCalibration {
    pub model_id: String,
    pub policy_samples: Vec<FiniteTvPathSample>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FiniteTvAlternativeModelCalibration {
    pub model_id: String,
    pub policy_samples: Vec<FiniteTvPathSample>,
    pub utility_lower_bound: f64,
    pub utility_upper_bound: f64,
    pub evaluator_cost_upper_bound: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FiniteTvCalibrationRequest {
    pub schema: String,
    pub version: String,
    pub calibration_id: String,
    /// SHA-256 of the canonical nominal PolicyFrontierPlan.
    pub source_plan_sha256: String,
    /// Simultaneous failure budget across all policy × null-model rows.
    pub null_family_coverage_failure_probability: f64,
    /// Simultaneous failure budget across all policy × alternative-model rows.
    pub alternative_family_coverage_failure_probability: f64,
    /// Calibration observations must be independent across complete paths.
    pub iid_complete_path_draws: bool,
    /// Calibration data must be independent of the confirmatory stream.
    pub independent_of_confirmatory_stream: bool,
    pub null_models: Vec<FiniteTvNullModelCalibration>,
    pub alternatives: Vec<FiniteTvAlternativeModelCalibration>,
}

impl FiniteTvCalibrationRequest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        calibration_id: impl Into<String>,
        source_plan_sha256: impl Into<String>,
        null_family_coverage_failure_probability: f64,
        alternative_family_coverage_failure_probability: f64,
        iid_complete_path_draws: bool,
        independent_of_confirmatory_stream: bool,
        null_models: Vec<FiniteTvNullModelCalibration>,
        alternatives: Vec<FiniteTvAlternativeModelCalibration>,
    ) -> Result<Self, FiniteTvCalibrationError> {
        let mut request = Self {
            schema: REQUEST_SCHEMA.into(),
            version: REQUEST_VERSION.into(),
            calibration_id: calibration_id.into(),
            source_plan_sha256: source_plan_sha256.into(),
            null_family_coverage_failure_probability,
            alternative_family_coverage_failure_probability,
            iid_complete_path_draws,
            independent_of_confirmatory_stream,
            null_models,
            alternatives,
        };
        request.canonicalize()?;
        Ok(request)
    }

    pub fn canonicalize(&mut self) -> Result<(), FiniteTvCalibrationError> {
        self.null_models.sort_by(|a, b| a.model_id.cmp(&b.model_id));
        self.alternatives.sort_by(|a, b| a.model_id.cmp(&b.model_id));
        for model in &mut self.null_models {
            model.policy_samples.sort_by(|a, b| a.policy_id.cmp(&b.policy_id));
        }
        for model in &mut self.alternatives {
            model.policy_samples.sort_by(|a, b| a.policy_id.cmp(&b.policy_id));
        }
        self.validate_structure()
    }

    fn validate_structure(&self) -> Result<(), FiniteTvCalibrationError> {
        if self.schema != REQUEST_SCHEMA || self.version != REQUEST_VERSION {
            return Err(FiniteTvCalibrationError::new(
                "request.schema",
                "unsupported finite-TV calibration request schema/version",
            ));
        }
        if self.calibration_id.trim().is_empty() {
            return Err(FiniteTvCalibrationError::new(
                "request.calibration_id",
                "calibration ID must be non-empty",
            ));
        }
        validate_hash(&self.source_plan_sha256, "request.source_plan_sha256")?;
        validate_open_probability(
            self.null_family_coverage_failure_probability,
            "request.null_family_coverage_failure_probability",
        )?;
        validate_open_probability(
            self.alternative_family_coverage_failure_probability,
            "request.alternative_family_coverage_failure_probability",
        )?;
        if !self.iid_complete_path_draws {
            return Err(FiniteTvCalibrationError::new(
                "request.iid_complete_path_draws",
                "finite-support concentration requires IID complete-path calibration draws",
            ));
        }
        if !self.independent_of_confirmatory_stream {
            return Err(FiniteTvCalibrationError::new(
                "request.independent_of_confirmatory_stream",
                "calibration data must be independent of the confirmatory stream",
            ));
        }
        if self.null_models.is_empty() || self.alternatives.len() < 2 {
            return Err(FiniteTvCalibrationError::new(
                "request.models",
                "null family must be non-empty and alternative family must contain at least two models",
            ));
        }
        validate_unique_model_ids(
            self.null_models.iter().map(|model| model.model_id.as_str()),
            "request.null_models",
        )?;
        validate_unique_model_ids(
            self.alternatives.iter().map(|model| model.model_id.as_str()),
            "request.alternatives",
        )?;

        for (model_index, model) in self.null_models.iter().enumerate() {
            let path = format!("request.null_models[{model_index}]");
            validate_policy_samples(&model.policy_samples, &format!("{path}.policy_samples"))?;
        }
        for (model_index, model) in self.alternatives.iter().enumerate() {
            let path = format!("request.alternatives[{model_index}]");
            validate_policy_samples(&model.policy_samples, &format!("{path}.policy_samples"))?;
            if !model.utility_lower_bound.is_finite()
                || !model.utility_upper_bound.is_finite()
                || model.utility_lower_bound > model.utility_upper_bound
            {
                return Err(FiniteTvCalibrationError::new(
                    format!("{path}.utility_bounds"),
                    "utility support bounds must be finite and ordered",
                ));
            }
            if !model.evaluator_cost_upper_bound.is_finite()
                || model.evaluator_cost_upper_bound < 0.0
            {
                return Err(FiniteTvCalibrationError::new(
                    format!("{path}.evaluator_cost_upper_bound"),
                    "evaluator-cost upper bound must be finite and nonnegative",
                ));
            }
        }
        Ok(())
    }

    pub fn request_sha256(&self) -> Result<String, FiniteTvCalibrationError> {
        let mut canonical = self.clone();
        canonical.canonicalize()?;
        digest(&canonical)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FiniteTvRadiusComponent {
    pub model_id: String,
    pub policy_id: String,
    pub path_support_sha256: String,
    pub calibration_data_sha256: String,
    pub support_size: usize,
    pub sample_size: u64,
    pub empirical_tv_distance_to_nominal: f64,
    pub allocated_failure_probability: f64,
    pub concentration_slack: f64,
    pub total_variation_radius: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FiniteTvCalibrationReport {
    pub schema: String,
    pub version: String,
    pub calibration_id: String,
    pub source_plan_sha256: String,
    pub request_sha256: String,
    pub null_components: Vec<FiniteTvRadiusComponent>,
    pub alternative_components: Vec<FiniteTvRadiusComponent>,
    /// Aggregated over policies using the maximum component radius per model.
    pub envelope: TvAmbiguityEnvelope,
    pub report_sha256: String,
}

#[derive(Serialize)]
struct ReportPayload<'a> {
    schema: &'a str,
    version: &'a str,
    calibration_id: &'a str,
    source_plan_sha256: &'a str,
    request_sha256: &'a str,
    null_components: &'a [FiniteTvRadiusComponent],
    alternative_components: &'a [FiniteTvRadiusComponent],
    envelope: &'a TvAmbiguityEnvelope,
}

#[derive(Serialize)]
struct NullCoveragePayload<'a> {
    calibration_id: &'a str,
    source_plan_sha256: &'a str,
    family_failure_probability: f64,
    iid_complete_path_draws: bool,
    independent_of_confirmatory_stream: bool,
    models: &'a [FiniteTvNullModelCalibration],
}

#[derive(Serialize)]
struct AlternativeCoveragePayload<'a> {
    calibration_id: &'a str,
    source_plan_sha256: &'a str,
    family_failure_probability: f64,
    iid_complete_path_draws: bool,
    independent_of_confirmatory_stream: bool,
    models: &'a [FiniteTvAlternativeModelCalibration],
}

fn validate_open_probability(value: f64, path: &str) -> Result<(), FiniteTvCalibrationError> {
    if !value.is_finite() || !(0.0..1.0).contains(&value) {
        return Err(FiniteTvCalibrationError::new(
            path,
            "failure probability must be finite and strictly between zero and one",
        ));
    }
    Ok(())
}

fn validate_hash(value: &str, path: &str) -> Result<(), FiniteTvCalibrationError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(FiniteTvCalibrationError::new(
            path,
            "expected a 64-character hexadecimal SHA-256 digest",
        ));
    }
    Ok(())
}

fn validate_unique_model_ids<'a>(
    ids: impl Iterator<Item = &'a str>,
    path: &str,
) -> Result<(), FiniteTvCalibrationError> {
    let mut seen = BTreeSet::new();
    for (index, id) in ids.enumerate() {
        if id.trim().is_empty() || !seen.insert(id) {
            return Err(FiniteTvCalibrationError::new(
                format!("{path}[{index}].model_id"),
                "model IDs must be non-empty and unique",
            ));
        }
    }
    Ok(())
}

fn validate_policy_samples(
    samples: &[FiniteTvPathSample],
    path: &str,
) -> Result<(), FiniteTvCalibrationError> {
    if samples.is_empty() {
        return Err(FiniteTvCalibrationError::new(path, "policy calibration samples must be non-empty"));
    }
    let mut policy_ids = BTreeSet::new();
    let mut reference_support: Option<(&str, usize)> = None;
    for (index, sample) in samples.iter().enumerate() {
        let row_path = format!("{path}[{index}]");
        if sample.policy_id.trim().is_empty() || !policy_ids.insert(sample.policy_id.as_str()) {
            return Err(FiniteTvCalibrationError::new(
                format!("{row_path}.policy_id"),
                "policy IDs must be non-empty and unique within a model",
            ));
        }
        validate_hash(&sample.path_support_sha256, &format!("{row_path}.path_support_sha256"))?;
        validate_hash(&sample.calibration_data_sha256, &format!("{row_path}.calibration_data_sha256"))?;
        let support_size = sample.observed_path_counts.len();
        if support_size == 0
            || support_size > MAX_PATH_SUPPORT
            || sample.nominal_path_probabilities.len() != support_size
        {
            return Err(FiniteTvCalibrationError::new(
                format!("{row_path}.path_support"),
                format!("path support must contain 1..={MAX_PATH_SUPPORT} aligned categories"),
            ));
        }
        if let Some((support_digest, expected_size)) = reference_support {
            if support_digest != sample.path_support_sha256.as_str() || expected_size != support_size {
                return Err(FiniteTvCalibrationError::new(
                    format!("{row_path}.path_support"),
                    "support digest, size and category ordering must match across policies for one model",
                ));
            }
        } else {
            reference_support = Some((&sample.path_support_sha256, support_size));
        }

        let sample_size = checked_sample_size(&sample.observed_path_counts, &row_path)?;
        if sample_size == 0 {
            return Err(FiniteTvCalibrationError::new(
                format!("{row_path}.observed_path_counts"),
                "calibration sample must contain at least one complete path",
            ));
        }
        let mut probability_sum = 0.0;
        for (category, probability) in sample.nominal_path_probabilities.iter().enumerate() {
            if !probability.is_finite() || !(0.0..=1.0).contains(probability) {
                return Err(FiniteTvCalibrationError::new(
                    format!("{row_path}.nominal_path_probabilities[{category}]"),
                    "nominal category probability must be finite and lie in [0, 1]",
                ));
            }
            probability_sum += probability;
        }
        if !probability_sum.is_finite()
            || (probability_sum - 1.0).abs() > PROBABILITY_SUM_TOLERANCE
        {
            return Err(FiniteTvCalibrationError::new(
                format!("{row_path}.nominal_path_probabilities"),
                "nominal path probabilities must sum to one within tolerance",
            ));
        }
    }
    Ok(())
}

fn checked_sample_size(counts: &[u64], path: &str) -> Result<u64, FiniteTvCalibrationError> {
    let total = counts.iter().try_fold(0_u64, |sum, count| {
        sum.checked_add(*count).ok_or_else(|| FiniteTvCalibrationError::new(
            format!("{path}.observed_path_counts"),
            "calibration sample count overflow",
        ))
    })?;
    if total > MAX_EXACT_F64_INTEGER {
        return Err(FiniteTvCalibrationError::new(
            format!("{path}.observed_path_counts"),
            "sample size exceeds exact f64 integer conversion supported by this reference implementation",
        ));
    }
    Ok(total)
}

fn validate_request_against_plan(
    nominal: &PolicyFrontierPlan,
    request: &FiniteTvCalibrationRequest,
) -> Result<(), FiniteTvCalibrationError> {
    let mut plan = nominal.clone();
    plan.canonicalize().map_err(|error| FiniteTvCalibrationError::from_plan("nominal_plan", error))?;
    let plan_digest = plan.plan_sha256()
        .map_err(|error| FiniteTvCalibrationError::from_plan("nominal_plan", error))?;
    if request.source_plan_sha256 != plan_digest {
        return Err(FiniteTvCalibrationError::new(
            "request.source_plan_sha256",
            "calibration request is not bound to this canonical nominal plan",
        ));
    }
    if request.null_family_coverage_failure_probability >= plan.significance_level {
        return Err(FiniteTvCalibrationError::new(
            "request.null_family_coverage_failure_probability",
            "null-family calibration failure must be strictly smaller than nominal alpha",
        ));
    }

    let expected_policy_ids = plan.policies.iter()
        .map(|policy| policy.policy_id.clone()).collect::<BTreeSet<_>>();
    let expected_null_ids = plan.policies[0].null_models.iter()
        .map(|row| row.model_id.clone()).collect::<BTreeSet<_>>();
    let expected_alternative_ids = plan.policies[0].alternatives.iter()
        .map(|row| row.model_id.clone()).collect::<BTreeSet<_>>();
    let supplied_null_ids = request.null_models.iter()
        .map(|model| model.model_id.clone()).collect::<BTreeSet<_>>();
    let supplied_alternative_ids = request.alternatives.iter()
        .map(|model| model.model_id.clone()).collect::<BTreeSet<_>>();
    if expected_null_ids != supplied_null_ids {
        return Err(FiniteTvCalibrationError::new(
            "request.null_models",
            "calibration must contain exactly every declared null model",
        ));
    }
    if expected_alternative_ids != supplied_alternative_ids {
        return Err(FiniteTvCalibrationError::new(
            "request.alternatives",
            "calibration must contain exactly every declared alternative model",
        ));
    }

    for (model_index, model) in request.null_models.iter().enumerate() {
        validate_policy_coverage(
            &model.policy_samples,
            &expected_policy_ids,
            &format!("request.null_models[{model_index}].policy_samples"),
        )?;
    }
    for (model_index, model) in request.alternatives.iter().enumerate() {
        validate_policy_coverage(
            &model.policy_samples,
            &expected_policy_ids,
            &format!("request.alternatives[{model_index}].policy_samples"),
        )?;
        for policy in &plan.policies {
            let receipt = policy.alternatives.iter()
                .find(|row| row.model_id == model.model_id)
                .ok_or_else(|| FiniteTvCalibrationError::new(
                    "nominal_plan.alternatives",
                    "nominal alternative receipt disappeared after validation",
                ))?;
            if receipt.decision_utility < model.utility_lower_bound - PROBABILITY_SUM_TOLERANCE
                || receipt.decision_utility > model.utility_upper_bound + PROBABILITY_SUM_TOLERANCE
            {
                return Err(FiniteTvCalibrationError::new(
                    format!("request.alternatives[{model_index}].utility_bounds"),
                    "utility support bounds do not contain every nominal policy receipt",
                ));
            }
            if receipt.expected_evaluator_cost
                > model.evaluator_cost_upper_bound + PROBABILITY_SUM_TOLERANCE
            {
                return Err(FiniteTvCalibrationError::new(
                    format!("request.alternatives[{model_index}].evaluator_cost_upper_bound"),
                    "cost support bound is below a nominal policy receipt",
                ));
            }
        }
    }
    Ok(())
}

fn validate_policy_coverage(
    samples: &[FiniteTvPathSample],
    expected: &BTreeSet<String>,
    path: &str,
) -> Result<(), FiniteTvCalibrationError> {
    let supplied = samples.iter().map(|sample| sample.policy_id.clone()).collect::<BTreeSet<_>>();
    if &supplied != expected {
        return Err(FiniteTvCalibrationError::new(
            path,
            "every declared policy must have exactly one calibration row for every model",
        ));
    }
    Ok(())
}

fn calibrate_component(
    model_id: &str,
    sample: &FiniteTvPathSample,
    allocated_failure_probability: f64,
) -> Result<FiniteTvRadiusComponent, FiniteTvCalibrationError> {
    let path = format!("calibration.{model_id}.{}", sample.policy_id);
    let sample_size = checked_sample_size(&sample.observed_path_counts, &path)?;
    if sample_size == 0 {
        return Err(FiniteTvCalibrationError::new(&path, "sample size must be positive"));
    }
    let support_size = sample.observed_path_counts.len();
    let mut empirical_tv = 0.0;
    for (count, nominal_probability) in sample.observed_path_counts
        .iter()
        .zip(&sample.nominal_path_probabilities)
    {
        let empirical_probability = (*count as f64) / (sample_size as f64);
        empirical_tv += (empirical_probability - nominal_probability).abs();
    }
    empirical_tv = (0.5 * empirical_tv).min(1.0);
    let concentration_slack = if support_size == 1 {
        0.0
    } else {
        let numerator = (support_size as f64) * LN_2 - allocated_failure_probability.ln();
        (numerator / (2.0 * sample_size as f64)).sqrt().min(1.0)
    };
    if !empirical_tv.is_finite() || !concentration_slack.is_finite() {
        return Err(FiniteTvCalibrationError::new(
            &path,
            "non-finite empirical TV or concentration slack",
        ));
    }
    let radius = (empirical_tv + concentration_slack).min(1.0);
    Ok(FiniteTvRadiusComponent {
        model_id: model_id.to_owned(),
        policy_id: sample.policy_id.clone(),
        path_support_sha256: sample.path_support_sha256.clone(),
        calibration_data_sha256: sample.calibration_data_sha256.clone(),
        support_size,
        sample_size,
        empirical_tv_distance_to_nominal: empirical_tv,
        allocated_failure_probability,
        concentration_slack,
        total_variation_radius: radius,
    })
}

fn aggregate_null_radii(
    models: &[FiniteTvNullModelCalibration],
    components: &[FiniteTvRadiusComponent],
) -> Vec<TvNullPathEnvelope> {
    models.iter().map(|model| TvNullPathEnvelope {
        model_id: model.model_id.clone(),
        total_variation_radius: components.iter()
            .filter(|component| component.model_id == model.model_id)
            .map(|component| component.total_variation_radius)
            .fold(0.0_f64, f64::max),
    }).collect()
}

fn aggregate_alternative_radii(
    models: &[FiniteTvAlternativeModelCalibration],
    components: &[FiniteTvRadiusComponent],
) -> Vec<TvAlternativePathEnvelope> {
    models.iter().map(|model| TvAlternativePathEnvelope {
        model_id: model.model_id.clone(),
        total_variation_radius: components.iter()
            .filter(|component| component.model_id == model.model_id)
            .map(|component| component.total_variation_radius)
            .fold(0.0_f64, f64::max),
        utility_lower_bound: model.utility_lower_bound,
        utility_upper_bound: model.utility_upper_bound,
        evaluator_cost_upper_bound: model.evaluator_cost_upper_bound,
    }).collect()
}

/// Derive a finite-support TV envelope and its row-level audit report.
///
/// The Weissman-style bound used here is
/// P(||p_hat-p||_1 >= x) <= 2^K exp(-n*x^2/2).
/// Since TV is half the L1 distance, the row slack is
/// sqrt((K*ln(2)+ln(1/delta_row))/(2*n)). The empirical-to-nominal distance
/// is added by the triangle inequality. A Bonferroni allocation across all
/// policy/model rows gives the requested simultaneous family coverage.
pub fn calibrate_finite_tv_ambiguity_envelope(
    nominal: &PolicyFrontierPlan,
    input_request: &FiniteTvCalibrationRequest,
) -> Result<FiniteTvCalibrationReport, FiniteTvCalibrationError> {
    let mut plan = nominal.clone();
    plan.canonicalize().map_err(|error| FiniteTvCalibrationError::from_plan("nominal_plan", error))?;
    let mut request = input_request.clone();
    request.canonicalize()?;
    validate_request_against_plan(&plan, &request)?;

    let null_row_count = request.null_models.iter().map(|model| model.policy_samples.len()).sum::<usize>();
    let alternative_row_count = request.alternatives.iter()
        .map(|model| model.policy_samples.len()).sum::<usize>();
    if null_row_count == 0 || alternative_row_count == 0 {
        return Err(FiniteTvCalibrationError::new("request.policy_samples", "calibration row family is empty"));
    }
    let null_delta_per_row =
        request.null_family_coverage_failure_probability / (null_row_count as f64);
    let alternative_delta_per_row =
        request.alternative_family_coverage_failure_probability / (alternative_row_count as f64);
    if null_delta_per_row <= 0.0 || alternative_delta_per_row <= 0.0 {
        return Err(FiniteTvCalibrationError::new(
            "request.coverage_failure_probabilities",
            "per-row family failure allocation underflowed to zero",
        ));
    }

    let mut null_components = Vec::with_capacity(null_row_count);
    for model in &request.null_models {
        for sample in &model.policy_samples {
            null_components.push(calibrate_component(
                &model.model_id,
                sample,
                null_delta_per_row,
            )?);
        }
    }
    let mut alternative_components = Vec::with_capacity(alternative_row_count);
    for model in &request.alternatives {
        for sample in &model.policy_samples {
            alternative_components.push(calibrate_component(
                &model.model_id,
                sample,
                alternative_delta_per_row,
            )?);
        }
    }

    let request_sha256 = request.request_sha256()?;
    let null_payload = NullCoveragePayload {
        calibration_id: &request.calibration_id,
        source_plan_sha256: &request.source_plan_sha256,
        family_failure_probability: request.null_family_coverage_failure_probability,
        iid_complete_path_draws: request.iid_complete_path_draws,
        independent_of_confirmatory_stream: request.independent_of_confirmatory_stream,
        models: &request.null_models,
    };
    let alternative_payload = AlternativeCoveragePayload {
        calibration_id: &request.calibration_id,
        source_plan_sha256: &request.source_plan_sha256,
        family_failure_probability: request.alternative_family_coverage_failure_probability,
        iid_complete_path_draws: request.iid_complete_path_draws,
        independent_of_confirmatory_stream: request.independent_of_confirmatory_stream,
        models: &request.alternatives,
    };
    let null_receipt_sha256 = digest(&null_payload)?;
    let alternative_receipt_sha256 = digest(&alternative_payload)?;
    let envelope = TvAmbiguityEnvelope::new(
        request.calibration_id.clone(),
        request.source_plan_sha256.clone(),
        request.null_family_coverage_failure_probability,
        request.alternative_family_coverage_failure_probability,
        null_receipt_sha256,
        alternative_receipt_sha256,
        request.independent_of_confirmatory_stream,
        aggregate_null_radii(&request.null_models, &null_components),
        aggregate_alternative_radii(&request.alternatives, &alternative_components),
    ).map_err(FiniteTvCalibrationError::from_robustness)?;

    let payload = ReportPayload {
        schema: REPORT_SCHEMA,
        version: REPORT_VERSION,
        calibration_id: &request.calibration_id,
        source_plan_sha256: &request.source_plan_sha256,
        request_sha256: &request_sha256,
        null_components: &null_components,
        alternative_components: &alternative_components,
        envelope: &envelope,
    };
    let report_sha256 = digest(&payload)?;
    Ok(FiniteTvCalibrationReport {
        schema: REPORT_SCHEMA.into(),
        version: REPORT_VERSION.into(),
        calibration_id: request.calibration_id,
        source_plan_sha256: request.source_plan_sha256,
        request_sha256,
        null_components,
        alternative_components,
        envelope,
        report_sha256,
    })
}

pub fn verify_finite_tv_calibration_report(
    nominal: &PolicyFrontierPlan,
    request: &FiniteTvCalibrationRequest,
    report: &FiniteTvCalibrationReport,
) -> Result<(), FiniteTvCalibrationError> {
    if calibrate_finite_tv_ambiguity_envelope(nominal, request)? != *report {
        return Err(FiniteTvCalibrationError::new(
            "report",
            "stored finite-TV calibration report differs from complete deterministic reconstruction",
        ));
    }
    Ok(())
}

fn digest<T: Serialize>(value: &T) -> Result<String, FiniteTvCalibrationError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| FiniteTvCalibrationError::new("serialization", error.to_string()))?;
    Ok(Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::policy_frontier_lp::{
        FrontierAlternativeReceipt, FrontierNullReceipt, FrontierPolicyReceipt,
    };
    use super::super::tv_robust_policy_frontier::analyze_tv_robust_policy_frontier;

    const UTILITY_HASH: &str =
        "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
    const SUPPORT_HASH: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const DATA_HASH_A: &str =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const DATA_HASH_B: &str =
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn policy(id: &str, null_a: f64, null_b: f64) -> FrontierPolicyReceipt {
        FrontierPolicyReceipt {
            policy_id: id.into(),
            utility_contract_sha256: UTILITY_HASH.into(),
            null_models: vec![
                FrontierNullReceipt { model_id: "null-a".into(), null_size: null_a },
                FrontierNullReceipt { model_id: "null-b".into(), null_size: null_b },
            ],
            alternatives: vec![
                FrontierAlternativeReceipt {
                    model_id: "alt-a".into(),
                    rejection_power: if id == "policy-a" { 0.8 } else { 0.6 },
                    expected_evaluator_cost: if id == "policy-a" { 4.0 } else { 2.0 },
                    decision_utility: if id == "policy-a" { 8.0 } else { 6.0 },
                },
                FrontierAlternativeReceipt {
                    model_id: "alt-b".into(),
                    rejection_power: if id == "policy-a" { 0.7 } else { 0.9 },
                    expected_evaluator_cost: if id == "policy-a" { 6.0 } else { 8.0 },
                    decision_utility: if id == "policy-a" { 6.0 } else { 9.0 },
                },
            ],
        }
    }

    fn nominal_plan() -> PolicyFrontierPlan {
        PolicyFrontierPlan::new(
            "finite-tv-calibration-fixture",
            0.05,
            Some(9.0),
            vec![
                policy("policy-a", 0.005, 0.008),
                policy("policy-b", 0.010, 0.015),
            ],
        ).unwrap()
    }

    fn sample(policy_id: &str, data_hash: &str) -> FiniteTvPathSample {
        FiniteTvPathSample {
            policy_id: policy_id.into(),
            path_support_sha256: SUPPORT_HASH.into(),
            calibration_data_sha256: data_hash.into(),
            observed_path_counts: vec![5_000, 5_000],
            nominal_path_probabilities: vec![0.5, 0.5],
        }
    }

    fn request(plan: &PolicyFrontierPlan) -> FiniteTvCalibrationRequest {
        FiniteTvCalibrationRequest::new(
            "finite-tv-calibration-001",
            plan.plan_sha256().unwrap(),
            0.01,
            0.02,
            true,
            true,
            vec![
                FiniteTvNullModelCalibration {
                    model_id: "null-a".into(),
                    policy_samples: vec![sample("policy-a", DATA_HASH_A), sample("policy-b", DATA_HASH_B)],
                },
                FiniteTvNullModelCalibration {
                    model_id: "null-b".into(),
                    policy_samples: vec![sample("policy-a", DATA_HASH_A), sample("policy-b", DATA_HASH_B)],
                },
            ],
            vec![
                FiniteTvAlternativeModelCalibration {
                    model_id: "alt-a".into(),
                    policy_samples: vec![sample("policy-a", DATA_HASH_A), sample("policy-b", DATA_HASH_B)],
                    utility_lower_bound: 0.0,
                    utility_upper_bound: 10.0,
                    evaluator_cost_upper_bound: 10.0,
                },
                FiniteTvAlternativeModelCalibration {
                    model_id: "alt-b".into(),
                    policy_samples: vec![sample("policy-a", DATA_HASH_A), sample("policy-b", DATA_HASH_B)],
                    utility_lower_bound: 0.0,
                    utility_upper_bound: 10.0,
                    evaluator_cost_upper_bound: 10.0,
                },
            ],
        ).unwrap()
    }

    #[test]
    fn empirical_tv_and_weissman_slack_match_hand_calculation() {
        let row = FiniteTvPathSample {
            policy_id: "policy".into(),
            path_support_sha256: SUPPORT_HASH.into(),
            calibration_data_sha256: DATA_HASH_A.into(),
            observed_path_counts: vec![75, 25],
            nominal_path_probabilities: vec![0.5, 0.5],
        };
        let component = calibrate_component("null", &row, 0.01).unwrap();
        let expected_slack = ((2.0 * LN_2 + 100.0_f64.ln()) / 200.0).sqrt();
        assert!((component.empirical_tv_distance_to_nominal - 0.25).abs() < 1e-12);
        assert!((component.concentration_slack - expected_slack).abs() < 1e-12);
        assert!((component.total_variation_radius - (0.25 + expected_slack).min(1.0)).abs() < 1e-12);
    }

    #[test]
    fn singleton_support_has_zero_statistical_radius() {
        let row = FiniteTvPathSample {
            policy_id: "policy".into(),
            path_support_sha256: SUPPORT_HASH.into(),
            calibration_data_sha256: DATA_HASH_A.into(),
            observed_path_counts: vec![100],
            nominal_path_probabilities: vec![1.0],
        };
        let component = calibrate_component("null", &row, 0.01).unwrap();
        assert_eq!(component.empirical_tv_distance_to_nominal, 0.0);
        assert_eq!(component.concentration_slack, 0.0);
        assert_eq!(component.total_variation_radius, 0.0);
    }

    #[test]
    fn simultaneous_row_allocation_and_max_policy_radius_are_reported() {
        let plan = nominal_plan();
        let req = request(&plan);
        let report = calibrate_finite_tv_ambiguity_envelope(&plan, &req).unwrap();
        assert_eq!(report.null_components.len(), 4);
        assert_eq!(report.alternative_components.len(), 4);
        assert!(report.null_components.iter().all(|row| {
            (row.allocated_failure_probability - 0.0025).abs() < 1e-12
        }));
        assert!(report.alternative_components.iter().all(|row| {
            (row.allocated_failure_probability - 0.005).abs() < 1e-12
        }));
        for envelope_row in &report.envelope.null_models {
            let expected = report.null_components.iter()
                .filter(|row| row.model_id == envelope_row.model_id)
                .map(|row| row.total_variation_radius)
                .fold(0.0_f64, f64::max);
            assert_eq!(envelope_row.total_variation_radius, expected);
        }
    }

    #[test]
    fn calibrated_envelope_flows_into_tv_robust_frontier() {
        let plan = nominal_plan();
        let req = request(&plan);
        let report = calibrate_finite_tv_ambiguity_envelope(&plan, &req).unwrap();
        let frontier = analyze_tv_robust_policy_frontier(&plan, &report.envelope).unwrap();
        assert!(frontier.worst_case_unconditional_type_i_upper_bound <= plan.significance_level + 1e-9);
        assert!(frontier.frontier.worst_case_decision_utility.is_finite());
        verify_finite_tv_calibration_report(&plan, &req, &report).unwrap();
    }

    #[test]
    fn rejects_stale_plan_digest_and_missing_policy_rows() {
        let plan = nominal_plan();
        let mut req = request(&plan);
        let changed_plan = PolicyFrontierPlan::new(
            "finite-tv-calibration-fixture",
            0.04,
            Some(9.0),
            vec![policy("policy-a", 0.005, 0.008), policy("policy-b", 0.010, 0.015)],
        ).unwrap();
        assert!(calibrate_finite_tv_ambiguity_envelope(&changed_plan, &req).is_err());
        req.null_models[0].policy_samples.pop();
        assert!(calibrate_finite_tv_ambiguity_envelope(&plan, &req).is_err());
    }

    #[test]
    fn rejects_non_iid_dependent_or_misaligned_support_inputs() {
        let plan = nominal_plan();
        let mut non_iid = request(&plan);
        non_iid.iid_complete_path_draws = false;
        assert!(calibrate_finite_tv_ambiguity_envelope(&plan, &non_iid).is_err());
        let mut dependent = request(&plan);
        dependent.independent_of_confirmatory_stream = false;
        assert!(calibrate_finite_tv_ambiguity_envelope(&plan, &dependent).is_err());
        let mut support = request(&plan);
        support.null_models[0].policy_samples[0].path_support_sha256 = DATA_HASH_B.into();
        assert!(calibrate_finite_tv_ambiguity_envelope(&plan, &support).is_err());
    }

    #[test]
    fn rejects_invalid_counts_probabilities_and_support_bounds() {
        let plan = nominal_plan();
        let mut bad_counts = request(&plan);
        bad_counts.null_models[0].policy_samples[0].observed_path_counts = vec![0, 0];
        assert!(calibrate_finite_tv_ambiguity_envelope(&plan, &bad_counts).is_err());
        let mut bad_probability = request(&plan);
        bad_probability.null_models[0].policy_samples[0].nominal_path_probabilities = vec![0.4, 0.4];
        assert!(calibrate_finite_tv_ambiguity_envelope(&plan, &bad_probability).is_err());
        let mut bad_bound = request(&plan);
        bad_bound.alternatives[0].evaluator_cost_upper_bound = 1.0;
        assert!(calibrate_finite_tv_ambiguity_envelope(&plan, &bad_bound).is_err());
    }

    #[test]
    fn report_tampering_and_input_reordering_are_detected_or_canonicalized() {
        let plan = nominal_plan();
        let req = request(&plan);
        let report = calibrate_finite_tv_ambiguity_envelope(&plan, &req).unwrap();
        let mut reordered = req.clone();
        reordered.null_models.reverse();
        reordered.alternatives.reverse();
        for model in &mut reordered.null_models {
            model.policy_samples.reverse();
        }
        for model in &mut reordered.alternatives {
            model.policy_samples.reverse();
        }
        assert_eq!(report, calibrate_finite_tv_ambiguity_envelope(&plan, &reordered).unwrap());
        let mut altered = report.clone();
        altered.null_components[0].total_variation_radius += 0.01;
        assert!(verify_finite_tv_calibration_report(&plan, &req, &altered).is_err());
    }
}
