//! Held-out calibration and sharpness evaluator for central predictive intervals.
//!
//! This module is intentionally separate from conservative-bound comparison. Nominal
//! coverage is meaningful only for reviewed central prediction intervals with an explicit
//! nominal level; physical absolute bounds and standard deviations are not silently
//! converted into probabilities. Confidence intervals here describe observed coverage
//! across declared independent evaluation groups, not the truth of the independence claim.

use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const COVERAGE_BASIS_POINTS: u16 = 10_000;
const WILSON_95_Z: f64 = 1.959_963_984_540_054;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ApplicabilityStatus {
    WithinDeclaredEnvelope,
    Borderline,
    OutOfDomain,
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationSliceRequirement {
    pub operating_case_id: String,
    pub observable_id: String,
    pub unit_id: String,
    /// Nominal central interval coverage in basis points; 9_500 means 95%.
    pub nominal_coverage_bp: u16,
    pub minimum_observations: usize,
    /// The Wilson 95% lower endpoint must be at least this value, in basis points.
    pub minimum_wilson_95_lower_bp: u16,
    /// The Wilson 95% upper endpoint must be no greater than this value, in basis points.
    pub maximum_wilson_95_upper_bp: u16,
    /// Positive, predeclared scale in the observable unit.
    pub normalization_scale: f64,
    /// Maximum mean central-interval score divided by normalization_scale.
    pub max_normalized_mean_interval_score: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationProfile {
    pub profile_id: String,
    pub calibration_manifest_sha256: String,
    pub evaluation_manifest_sha256: String,
    pub prediction_manifest_sha256: String,
    pub model_configuration_sha256: String,
    pub split_manifest_sha256: String,
    /// Attestation only; it does not prove that no one inspected the hold-out early.
    pub model_frozen_before_holdout: bool,
    pub interval_method_id: String,
    pub interval_method_revision: String,
    /// True only after a reviewer has checked the interval method's stated central-coverage semantics.
    pub interval_method_reviewed: bool,
    /// Canonical digest of all acceptance-relevant profile content except this field.
    pub profile_manifest_sha256: String,
    /// Predeclared independent group identities; repeated rows within a group are not
    /// counted as independent calibration trials.
    pub calibration_group_ids: Vec<String>,
    pub evaluation_group_ids: Vec<String>,
    pub required_slices: Vec<CalibrationSliceRequirement>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationObservation {
    pub observation_id: String,
    pub evaluation_manifest_sha256: String,
    pub prediction_manifest_sha256: String,
    pub model_configuration_sha256: String,
    pub evaluation_group_id: String,
    pub operating_case_id: String,
    pub observable_id: String,
    pub unit_id: String,
    pub nominal_coverage_bp: u16,
    /// Inclusive limits of a declared central predictive interval, not physical bounds.
    pub lower: f64,
    pub upper: f64,
    pub observed_value: f64,
    pub applicability: ApplicabilityStatus,
    pub interval_method_id: String,
    pub interval_method_revision: String,
    /// Must be true: a central interval is not interchangeable with a one-sided bound.
    pub is_central_interval: bool,
    /// Must be true: this is a declared ordering check, not proof of blindness.
    pub prediction_issued_before_outcome: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationDisposition {
    PassDeclaredCalibrationThresholds,
    FailDeclaredCalibrationThresholds,
    IncompleteCoverage,
    DomainNotEstablished,
    InvalidProfile,
    InvalidObservations,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum CalibrationIssue {
    EmptyProfileField { field: &'static str },
    InvalidProfileDigest { field: &'static str },
    ProfileManifestDigestMismatch,
    CalibrationAndEvaluationManifestMatch,
    ModelNotFrozenBeforeHoldout,
    IntervalMethodNotReviewed,
    EmptyGroupIdentifier { role: &'static str },
    DuplicateGroupIdentifier { role: &'static str, group_id: String },
    CalibrationEvaluationGroupOverlap { group_id: String },
    NoRequiredSlices,
    DuplicateRequiredSlice {
        operating_case_id: String,
        observable_id: String,
        unit_id: String,
        nominal_coverage_bp: u16,
    },
    InvalidRequirementField {
        operating_case_id: String,
        observable_id: String,
        field: &'static str,
    },
    EmptyObservationField {
        observation_id: String,
        field: &'static str,
    },
    InvalidObservationDigest {
        observation_id: String,
        field: &'static str,
    },
    DuplicateObservationId { observation_id: String, count: usize },
    DuplicateEvaluationGroupObservation {
        evaluation_group_id: String,
        operating_case_id: String,
        observable_id: String,
        unit_id: String,
        nominal_coverage_bp: u16,
        count: usize,
    },
    DatasetManifestMismatch { observation_id: String },
    PredictionManifestMismatch { observation_id: String },
    ModelConfigurationMismatch { observation_id: String },
    UndeclaredEvaluationGroup { observation_id: String },
    MissingEvaluationGroup { evaluation_group_id: String },
    UndeclaredSlice {
        observation_id: String,
        operating_case_id: String,
        observable_id: String,
        unit_id: String,
        nominal_coverage_bp: u16,
    },
    IntervalMethodMismatch { observation_id: String },
    IntervalNotCentral { observation_id: String },
    PredictionIssuedAfterOutcome { observation_id: String },
    InvalidNominalCoverage { observation_id: String },
    InvalidInterval { observation_id: String, field: &'static str },
    NonFiniteIntervalScore { observation_id: String },
    ApplicabilityNotEstablished {
        observation_id: String,
        status: ApplicabilityStatus,
    },
    InsufficientCoverage {
        operating_case_id: String,
        observable_id: String,
        unit_id: String,
        nominal_coverage_bp: u16,
        minimum: usize,
        observed: usize,
    },
    NonFiniteMetric {
        operating_case_id: String,
        observable_id: String,
        unit_id: String,
        nominal_coverage_bp: u16,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationSliceScore {
    pub operating_case_id: String,
    pub observable_id: String,
    pub unit_id: String,
    pub nominal_coverage_bp: u16,
    pub observation_count: usize,
    pub required_observation_count: usize,
    pub covered_count: usize,
    /// Empirical coverage fraction in [0, 1].
    pub empirical_coverage: Option<f64>,
    /// Wilson score interval at nominal 95%; assumes declared evaluation groups are independent.
    pub wilson_95_lower: Option<f64>,
    pub wilson_95_upper: Option<f64>,
    pub mean_interval_width: Option<f64>,
    pub normalized_mean_interval_width: Option<f64>,
    pub mean_interval_score: Option<f64>,
    pub normalized_mean_interval_score: Option<f64>,
    pub within_declared_envelope_count: usize,
    pub borderline_count: usize,
    pub out_of_domain_count: usize,
    pub unknown_applicability_count: usize,
    /// False when any evaluated record is borderline, out of domain or unknown.
    pub applicability_gate_met: bool,
    /// None means incomplete groups/counts, unestablished applicability or invalid metric.
    pub coverage_gate_met: Option<bool>,
    pub interval_score_gate_met: Option<bool>,
    pub thresholds_met: Option<bool>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationReport {
    pub profile_id: String,
    pub profile_manifest_sha256: String,
    pub disposition: CalibrationDisposition,
    pub input_observation_count: usize,
    pub unique_observation_count: usize,
    pub duplicated_observation_record_count: usize,
    pub duplicated_evaluation_group_record_count: usize,
    pub slices: Vec<CalibrationSliceScore>,
    pub issues: Vec<CalibrationIssue>,
}

#[derive(Default)]
struct CanonicalEncoder(Vec<u8>);

impl CanonicalEncoder {
    fn new(domain: &str) -> Self {
        let mut encoder = Self::default();
        encoder.text("domain", domain);
        encoder
    }

    fn field(&mut self, name: &str, value: &[u8]) {
        self.0.extend_from_slice(&(name.len() as u64).to_be_bytes());
        self.0.extend_from_slice(name.as_bytes());
        self.0.extend_from_slice(&(value.len() as u64).to_be_bytes());
        self.0.extend_from_slice(value);
    }

    fn text(&mut self, name: &str, value: &str) {
        self.field(name, value.as_bytes());
    }

    fn count(&mut self, name: &str, value: usize) {
        self.field(name, &(value as u64).to_be_bytes());
    }

    fn number(&mut self, name: &str, value: u64) {
        self.field(name, &value.to_be_bytes());
    }

    fn float(&mut self, name: &str, value: f64) {
        self.field(name, &value.to_bits().to_be_bytes());
    }

    fn boolean(&mut self, name: &str, value: bool) {
        self.field(name, &[u8::from(value)]);
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut result = String::with_capacity(64);
    use std::fmt::Write as _;
    for byte in digest {
        write!(&mut result, "{byte:02x}").expect("writing to String cannot fail");
    }
    result
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.as_bytes().iter().all(u8::is_ascii_hexdigit)
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn slice_key(
    requirement: &CalibrationSliceRequirement,
) -> (String, String, String, u16) {
    (
        requirement.operating_case_id.trim().to_owned(),
        requirement.observable_id.trim().to_owned(),
        requirement.unit_id.trim().to_owned(),
        requirement.nominal_coverage_bp,
    )
}

/// Canonical digest of every acceptance-relevant field, including group splits and thresholds.
/// List order does not affect the digest, but changing any value does.
pub fn canonical_calibration_profile_sha256(profile: &CalibrationProfile) -> String {
    let mut encoder = CanonicalEncoder::new("marine-prediction-interval-calibration-v1");
    encoder.text("profile_id", profile.profile_id.trim());
    encoder.text(
        "calibration_manifest_sha256",
        &profile.calibration_manifest_sha256.to_ascii_lowercase(),
    );
    encoder.text(
        "evaluation_manifest_sha256",
        &profile.evaluation_manifest_sha256.to_ascii_lowercase(),
    );
    encoder.text(
        "prediction_manifest_sha256",
        &profile.prediction_manifest_sha256.to_ascii_lowercase(),
    );
    encoder.text(
        "model_configuration_sha256",
        &profile.model_configuration_sha256.to_ascii_lowercase(),
    );
    encoder.text(
        "split_manifest_sha256",
        &profile.split_manifest_sha256.to_ascii_lowercase(),
    );
    encoder.boolean("model_frozen_before_holdout", profile.model_frozen_before_holdout);
    encoder.text("interval_method_id", profile.interval_method_id.trim());
    encoder.text("interval_method_revision", profile.interval_method_revision.trim());
    encoder.boolean("interval_method_reviewed", profile.interval_method_reviewed);

    let mut calibration_groups = profile.calibration_group_ids.clone();
    calibration_groups.sort();
    encoder.count("calibration_group_count", calibration_groups.len());
    for group in &calibration_groups {
        encoder.text("calibration_group", group.trim());
    }

    let mut evaluation_groups = profile.evaluation_group_ids.clone();
    evaluation_groups.sort();
    encoder.count("evaluation_group_count", evaluation_groups.len());
    for group in &evaluation_groups {
        encoder.text("evaluation_group", group.trim());
    }

    let mut requirements = profile.required_slices.clone();
    requirements.sort_by(|a, b| {
        slice_key(a)
            .cmp(&slice_key(b))
            .then_with(|| a.minimum_observations.cmp(&b.minimum_observations))
            .then_with(|| a.minimum_wilson_95_lower_bp.cmp(&b.minimum_wilson_95_lower_bp))
            .then_with(|| a.maximum_wilson_95_upper_bp.cmp(&b.maximum_wilson_95_upper_bp))
            .then_with(|| a.normalization_scale.total_cmp(&b.normalization_scale))
            .then_with(|| {
                a.max_normalized_mean_interval_score
                    .total_cmp(&b.max_normalized_mean_interval_score)
            })
    });
    encoder.count("required_slice_count", requirements.len());
    for requirement in &requirements {
        let mut item = CanonicalEncoder::new("marine-prediction-interval-slice-v1");
        item.text("operating_case_id", requirement.operating_case_id.trim());
        item.text("observable_id", requirement.observable_id.trim());
        item.text("unit_id", requirement.unit_id.trim());
        item.number("nominal_coverage_bp", requirement.nominal_coverage_bp as u64);
        item.count("minimum_observations", requirement.minimum_observations);
        item.number(
            "minimum_wilson_95_lower_bp",
            requirement.minimum_wilson_95_lower_bp as u64,
        );
        item.number(
            "maximum_wilson_95_upper_bp",
            requirement.maximum_wilson_95_upper_bp as u64,
        );
        item.float("normalization_scale", requirement.normalization_scale);
        item.float(
            "max_normalized_mean_interval_score",
            requirement.max_normalized_mean_interval_score,
        );
        encoder.field("required_slice", &item.0);
    }
    sha256_hex(&encoder.0)
}

fn validate_profile(
    profile: &CalibrationProfile,
    issues: &mut Vec<CalibrationIssue>,
) -> bool {
    let mut invalid = false;
    for (value, field) in [
        (profile.profile_id.as_str(), "profile_id"),
        (profile.calibration_manifest_sha256.as_str(), "calibration_manifest_sha256"),
        (profile.evaluation_manifest_sha256.as_str(), "evaluation_manifest_sha256"),
        (profile.prediction_manifest_sha256.as_str(), "prediction_manifest_sha256"),
        (profile.model_configuration_sha256.as_str(), "model_configuration_sha256"),
        (profile.split_manifest_sha256.as_str(), "split_manifest_sha256"),
        (profile.profile_manifest_sha256.as_str(), "profile_manifest_sha256"),
        (profile.interval_method_id.as_str(), "interval_method_id"),
        (profile.interval_method_revision.as_str(), "interval_method_revision"),
    ] {
        if !nonempty(value) {
            issues.push(CalibrationIssue::EmptyProfileField { field });
            invalid = true;
        }
    }
    for (value, field) in [
        (profile.calibration_manifest_sha256.as_str(), "calibration_manifest_sha256"),
        (profile.evaluation_manifest_sha256.as_str(), "evaluation_manifest_sha256"),
        (profile.prediction_manifest_sha256.as_str(), "prediction_manifest_sha256"),
        (profile.model_configuration_sha256.as_str(), "model_configuration_sha256"),
        (profile.split_manifest_sha256.as_str(), "split_manifest_sha256"),
        (profile.profile_manifest_sha256.as_str(), "profile_manifest_sha256"),
    ] {
        if !valid_sha256(value) {
            issues.push(CalibrationIssue::InvalidProfileDigest { field });
            invalid = true;
        }
    }
    if valid_sha256(&profile.profile_manifest_sha256) {
        let expected = canonical_calibration_profile_sha256(profile);
        if !profile.profile_manifest_sha256.eq_ignore_ascii_case(&expected) {
            issues.push(CalibrationIssue::ProfileManifestDigestMismatch);
            invalid = true;
        }
    }
    if profile.calibration_manifest_sha256.eq_ignore_ascii_case(&profile.evaluation_manifest_sha256) {
        issues.push(CalibrationIssue::CalibrationAndEvaluationManifestMatch);
        invalid = true;
    }
    if !profile.model_frozen_before_holdout {
        issues.push(CalibrationIssue::ModelNotFrozenBeforeHoldout);
        invalid = true;
    }
    if !profile.interval_method_reviewed {
        issues.push(CalibrationIssue::IntervalMethodNotReviewed);
        invalid = true;
    }
    if profile.required_slices.is_empty() {
        issues.push(CalibrationIssue::NoRequiredSlices);
        invalid = true;
    }

    let mut calibration_groups = BTreeSet::new();
    for group in &profile.calibration_group_ids {
        let normalized = group.trim().to_owned();
        if normalized.is_empty() {
            issues.push(CalibrationIssue::EmptyGroupIdentifier { role: "calibration" });
            invalid = true;
        } else if !calibration_groups.insert(normalized.clone()) {
            issues.push(CalibrationIssue::DuplicateGroupIdentifier {
                role: "calibration",
                group_id: normalized,
            });
            invalid = true;
        }
    }
    let mut evaluation_groups = BTreeSet::new();
    for group in &profile.evaluation_group_ids {
        let normalized = group.trim().to_owned();
        if normalized.is_empty() {
            issues.push(CalibrationIssue::EmptyGroupIdentifier { role: "evaluation" });
            invalid = true;
        } else if !evaluation_groups.insert(normalized.clone()) {
            issues.push(CalibrationIssue::DuplicateGroupIdentifier {
                role: "evaluation",
                group_id: normalized,
            });
            invalid = true;
        }
    }
    if profile.calibration_group_ids.is_empty() {
        issues.push(CalibrationIssue::EmptyGroupIdentifier { role: "calibration" });
        invalid = true;
    }
    if profile.evaluation_group_ids.is_empty() {
        issues.push(CalibrationIssue::EmptyGroupIdentifier { role: "evaluation" });
        invalid = true;
    }
    for group_id in calibration_groups.intersection(&evaluation_groups) {
        issues.push(CalibrationIssue::CalibrationEvaluationGroupOverlap {
            group_id: group_id.clone(),
        });
        invalid = true;
    }

    let mut seen = BTreeSet::new();
    for requirement in &profile.required_slices {
        let key = slice_key(requirement);
        for (value, field) in [
            (key.0.as_str(), "operating_case_id"),
            (key.1.as_str(), "observable_id"),
            (key.2.as_str(), "unit_id"),
        ] {
            if !nonempty(value) {
                issues.push(CalibrationIssue::InvalidRequirementField {
                    operating_case_id: key.0.clone(),
                    observable_id: key.1.clone(),
                    field,
                });
                invalid = true;
            }
        }
        if !seen.insert(key.clone()) {
            issues.push(CalibrationIssue::DuplicateRequiredSlice {
                operating_case_id: key.0.clone(),
                observable_id: key.1.clone(),
                unit_id: key.2.clone(),
                nominal_coverage_bp: key.3,
            });
            invalid = true;
        }
        if requirement.minimum_observations == 0 {
            issues.push(CalibrationIssue::InvalidRequirementField {
                operating_case_id: key.0.clone(),
                observable_id: key.1.clone(),
                field: "minimum_observations",
            });
            invalid = true;
        }
        if requirement.nominal_coverage_bp == 0
            || requirement.nominal_coverage_bp >= COVERAGE_BASIS_POINTS
        {
            issues.push(CalibrationIssue::InvalidRequirementField {
                operating_case_id: key.0.clone(),
                observable_id: key.1.clone(),
                field: "nominal_coverage_bp",
            });
            invalid = true;
        }
        if requirement.minimum_wilson_95_lower_bp > COVERAGE_BASIS_POINTS
            || requirement.maximum_wilson_95_upper_bp > COVERAGE_BASIS_POINTS
            || requirement.minimum_wilson_95_lower_bp > requirement.maximum_wilson_95_upper_bp
        {
            issues.push(CalibrationIssue::InvalidRequirementField {
                operating_case_id: key.0.clone(),
                observable_id: key.1.clone(),
                field: "empirical_coverage_band",
            });
            invalid = true;
        }
        if !requirement.normalization_scale.is_finite() || requirement.normalization_scale <= 0.0 {
            issues.push(CalibrationIssue::InvalidRequirementField {
                operating_case_id: key.0.clone(),
                observable_id: key.1.clone(),
                field: "normalization_scale",
            });
            invalid = true;
        }
        if !requirement.max_normalized_mean_interval_score.is_finite()
            || requirement.max_normalized_mean_interval_score < 0.0
        {
            issues.push(CalibrationIssue::InvalidRequirementField {
                operating_case_id: key.0.clone(),
                observable_id: key.1.clone(),
                field: "max_normalized_mean_interval_score",
            });
            invalid = true;
        }
    }
    invalid
}

#[derive(Debug, Clone, Copy, Default)]
struct Accumulator {
    count: usize,
    covered_count: usize,
    width_sum: f64,
    width_compensation: f64,
    score_sum: f64,
    score_compensation: f64,
    within_count: usize,
    borderline_count: usize,
    out_of_domain_count: usize,
    unknown_applicability_count: usize,
}

impl Accumulator {
    fn add_sum(sum: &mut f64, compensation: &mut f64, value: f64) {
        let adjusted = value - *compensation;
        let next = *sum + adjusted;
        *compensation = (next - *sum) - adjusted;
        *sum = next;
    }

    fn add(&mut self, observation: &CalibrationObservation) -> bool {
        let width = observation.upper - observation.lower;
        let alpha = (COVERAGE_BASIS_POINTS - observation.nominal_coverage_bp) as f64
            / COVERAGE_BASIS_POINTS as f64;
        if !width.is_finite() || alpha <= 0.0 {
            return false;
        }
        let mut interval_score = width;
        if observation.observed_value < observation.lower {
            interval_score += (2.0 / alpha) * (observation.lower - observation.observed_value);
        } else if observation.observed_value > observation.upper {
            interval_score += (2.0 / alpha) * (observation.observed_value - observation.upper);
        }
        if !interval_score.is_finite() || interval_score < 0.0 {
            return false;
        }
        let next_width_sum = self.width_sum;
        let next_score_sum = self.score_sum;
        Self::add_sum(&mut self.width_sum, &mut self.width_compensation, width);
        Self::add_sum(&mut self.score_sum, &mut self.score_compensation, interval_score);
        if !self.width_sum.is_finite() || !self.score_sum.is_finite() {
            self.width_sum = next_width_sum;
            self.score_sum = next_score_sum;
            return false;
        }

        self.count += 1;
        if observation.observed_value >= observation.lower
            && observation.observed_value <= observation.upper
        {
            self.covered_count += 1;
        }
        match observation.applicability {
            ApplicabilityStatus::WithinDeclaredEnvelope => self.within_count += 1,
            ApplicabilityStatus::Borderline => self.borderline_count += 1,
            ApplicabilityStatus::OutOfDomain => self.out_of_domain_count += 1,
            ApplicabilityStatus::Unknown => self.unknown_applicability_count += 1,
        }
        true
    }

    fn metrics(
        &self,
        scale: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64, f64)> {
        if self.count == 0 {
            return None;
        }
        let n = self.count as f64;
        let coverage = self.covered_count as f64 / n;
        let mean_width = self.width_sum / n;
        let normalized_mean_width = mean_width / scale;
        let mean_score = self.score_sum / n;
        let normalized_mean_score = mean_score / scale;
        let (wilson_lower, wilson_upper) = wilson_95(self.covered_count, self.count)?;
        let values = [
            coverage,
            wilson_lower,
            wilson_upper,
            mean_width,
            normalized_mean_width,
            mean_score,
            normalized_mean_score,
        ];
        values.iter().all(|value| value.is_finite()).then_some((
            coverage,
            wilson_lower,
            wilson_upper,
            mean_width,
            normalized_mean_width,
            mean_score,
            normalized_mean_score,
        ))
    }
}

/// Wilson score interval at 95% nominal confidence for a binomial count.
/// Independence is conditional on the externally declared evaluation-group design.
fn wilson_95(successes: usize, trials: usize) -> Option<(f64, f64)> {
    if trials == 0 || successes > trials {
        return None;
    }
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = WILSON_95_Z * WILSON_95_Z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half_width = WILSON_95_Z
        * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt())
        / denominator;
    let lower = (center - half_width).max(0.0);
    let upper = (center + half_width).min(1.0);
    (lower.is_finite() && upper.is_finite() && lower <= upper).then_some((lower, upper))
}

/// Score frozen central predictive intervals against outcomes from the declared evaluation
/// groups. Duplicate IDs and repeated rows for one group/slice are excluded rather than
/// counted as independent evidence. No point from an undeclared model, prediction artifact,
/// data split, interval method or operating slice can contribute to a pass.
pub fn evaluate_calibration(
    profile: &CalibrationProfile,
    observations: &[CalibrationObservation],
) -> CalibrationReport {
    let mut issues = Vec::new();
    let invalid_profile = validate_profile(profile, &mut issues);

    let mut id_counts: BTreeMap<String, usize> = BTreeMap::new();
    for observation in observations {
        *id_counts.entry(observation.observation_id.trim().to_owned()).or_default() += 1;
    }
    let duplicate_ids: BTreeSet<String> = id_counts
        .iter()
        .filter(|(_, count)| **count > 1)
        .map(|(id, _)| id.clone())
        .collect();
    for (observation_id, count) in &id_counts {
        if *count > 1 {
            issues.push(CalibrationIssue::DuplicateObservationId {
                observation_id: observation_id.clone(),
                count: *count,
            });
        }
    }
    let duplicated_observation_record_count: usize = duplicate_ids
        .iter()
        .filter_map(|id| id_counts.get(id))
        .sum();
    let unique_observation_count = id_counts.len() - duplicate_ids.len();

    type SliceKey = (String, String, String, u16);
    type GroupSliceKey = (String, String, String, String, u16);
    let mut requirements: BTreeMap<SliceKey, &CalibrationSliceRequirement> = BTreeMap::new();
    for requirement in &profile.required_slices {
        requirements.entry(slice_key(requirement)).or_insert(requirement);
    }

    let mut group_counts: BTreeMap<GroupSliceKey, usize> = BTreeMap::new();
    for observation in observations
        .iter()
        .filter(|o| !duplicate_ids.contains(o.observation_id.trim()))
    {
        let key = (
            observation.evaluation_group_id.trim().to_owned(),
            observation.operating_case_id.trim().to_owned(),
            observation.observable_id.trim().to_owned(),
            observation.unit_id.trim().to_owned(),
            observation.nominal_coverage_bp,
        );
        *group_counts.entry(key).or_default() += 1;
    }
    let duplicate_group_keys: BTreeSet<GroupSliceKey> = group_counts
        .iter()
        .filter(|(_, count)| **count > 1)
        .map(|(key, _)| key.clone())
        .collect();
    let duplicated_evaluation_group_record_count: usize = duplicate_group_keys
        .iter()
        .filter_map(|key| group_counts.get(key))
        .sum();
    for (key, count) in &group_counts {
        if *count > 1 {
            issues.push(CalibrationIssue::DuplicateEvaluationGroupObservation {
                evaluation_group_id: key.0.clone(),
                operating_case_id: key.1.clone(),
                observable_id: key.2.clone(),
                unit_id: key.3.clone(),
                nominal_coverage_bp: key.4,
                count: *count,
            });
        }
    }

    let mut accumulators: BTreeMap<SliceKey, Accumulator> = BTreeMap::new();
    for key in requirements.keys() {
        accumulators.insert(key.clone(), Accumulator::default());
    }
    let eval_groups: BTreeSet<String> = profile
        .evaluation_group_ids
        .iter()
        .map(|s| s.trim().to_owned())
        .collect();
    let mut seen_evaluation_groups = BTreeSet::new();
    let mut sorted_observations: Vec<&CalibrationObservation> = observations
        .iter()
        .filter(|o| {
            !duplicate_ids.contains(o.observation_id.trim())
                && !duplicate_group_keys.contains(&(
                    o.evaluation_group_id.trim().to_owned(),
                    o.operating_case_id.trim().to_owned(),
                    o.observable_id.trim().to_owned(),
                    o.unit_id.trim().to_owned(),
                    o.nominal_coverage_bp,
                ))
        })
        .collect();
    sorted_observations.sort_by(|a, b| {
        a.operating_case_id
            .trim()
            .cmp(b.operating_case_id.trim())
            .then_with(|| a.observable_id.trim().cmp(b.observable_id.trim()))
            .then_with(|| a.unit_id.trim().cmp(b.unit_id.trim()))
            .then_with(|| a.nominal_coverage_bp.cmp(&b.nominal_coverage_bp))
            .then_with(|| a.evaluation_group_id.trim().cmp(b.evaluation_group_id.trim()))
            .then_with(|| a.observation_id.trim().cmp(b.observation_id.trim()))
    });

    let mut invalid_observations = !duplicate_ids.is_empty() || !duplicate_group_keys.is_empty();
    let mut invalid_metric_slices = BTreeSet::new();
    let mut domain_not_established = false;
    let mut missing_required_coverage = false;

    for observation in sorted_observations {
        let id = observation.observation_id.trim().to_owned();
        let mut valid = true;
        for (value, field) in [
            (observation.observation_id.as_str(), "observation_id"),
            (observation.evaluation_group_id.as_str(), "evaluation_group_id"),
            (observation.operating_case_id.as_str(), "operating_case_id"),
            (observation.observable_id.as_str(), "observable_id"),
            (observation.unit_id.as_str(), "unit_id"),
            (observation.interval_method_id.as_str(), "interval_method_id"),
            (observation.interval_method_revision.as_str(), "interval_method_revision"),
        ] {
            if !nonempty(value) {
                issues.push(CalibrationIssue::EmptyObservationField {
                    observation_id: id.clone(),
                    field,
                });
                valid = false;
            }
        }
        for (value, field) in [
            (observation.evaluation_manifest_sha256.as_str(), "evaluation_manifest_sha256"),
            (observation.prediction_manifest_sha256.as_str(), "prediction_manifest_sha256"),
            (observation.model_configuration_sha256.as_str(), "model_configuration_sha256"),
        ] {
            if !valid_sha256(value) {
                issues.push(CalibrationIssue::InvalidObservationDigest {
                    observation_id: id.clone(),
                    field,
                });
                valid = false;
            }
        }
        if !observation.evaluation_manifest_sha256.eq_ignore_ascii_case(&profile.evaluation_manifest_sha256) {
            issues.push(CalibrationIssue::DatasetManifestMismatch { observation_id: id.clone() });
            valid = false;
        }
        if !observation.prediction_manifest_sha256.eq_ignore_ascii_case(&profile.prediction_manifest_sha256) {
            issues.push(CalibrationIssue::PredictionManifestMismatch { observation_id: id.clone() });
            valid = false;
        }
        if !observation.model_configuration_sha256.eq_ignore_ascii_case(&profile.model_configuration_sha256) {
            issues.push(CalibrationIssue::ModelConfigurationMismatch { observation_id: id.clone() });
            valid = false;
        }
        if !eval_groups.contains(observation.evaluation_group_id.trim()) {
            issues.push(CalibrationIssue::UndeclaredEvaluationGroup { observation_id: id.clone() });
            valid = false;
        }
        if observation.interval_method_id.trim() != profile.interval_method_id.trim()
            || observation.interval_method_revision.trim() != profile.interval_method_revision.trim()
        {
            issues.push(CalibrationIssue::IntervalMethodMismatch { observation_id: id.clone() });
            valid = false;
        }
        if !observation.is_central_interval {
            issues.push(CalibrationIssue::IntervalNotCentral { observation_id: id.clone() });
            valid = false;
        }
        if !observation.prediction_issued_before_outcome {
            issues.push(CalibrationIssue::PredictionIssuedAfterOutcome { observation_id: id.clone() });
            valid = false;
        }
        if observation.nominal_coverage_bp == 0 || observation.nominal_coverage_bp >= COVERAGE_BASIS_POINTS {
            issues.push(CalibrationIssue::InvalidNominalCoverage { observation_id: id.clone() });
            valid = false;
        }
        for (value, field) in [
            (observation.lower, "lower"),
            (observation.upper, "upper"),
            (observation.observed_value, "observed_value"),
        ] {
            if !value.is_finite() {
                issues.push(CalibrationIssue::InvalidInterval {
                    observation_id: id.clone(),
                    field,
                });
                valid = false;
            }
        }
        if observation.lower.is_finite()
            && observation.upper.is_finite()
            && observation.lower > observation.upper
        {
            issues.push(CalibrationIssue::InvalidInterval {
                observation_id: id.clone(),
                field: "ordering",
            });
            valid = false;
        }

        let key = (
            observation.operating_case_id.trim().to_owned(),
            observation.observable_id.trim().to_owned(),
            observation.unit_id.trim().to_owned(),
            observation.nominal_coverage_bp,
        );
        if !requirements.contains_key(&key) {
            issues.push(CalibrationIssue::UndeclaredSlice {
                observation_id: id.clone(),
                operating_case_id: key.0.clone(),
                observable_id: key.1.clone(),
                unit_id: key.2.clone(),
                nominal_coverage_bp: key.3,
            });
            valid = false;
        }
        if !valid {
            invalid_observations = true;
            continue;
        }

        let group_id = observation.evaluation_group_id.trim().to_owned();
        seen_evaluation_groups.insert(group_id.clone());
        if observation.applicability != ApplicabilityStatus::WithinDeclaredEnvelope {
            issues.push(CalibrationIssue::ApplicabilityNotEstablished {
                observation_id: id.clone(),
                status: observation.applicability,
            });
            domain_not_established = true;
        }

        let accumulator = accumulators.entry(key.clone()).or_default();
        if !accumulator.add(observation) {
            issues.push(CalibrationIssue::NonFiniteIntervalScore { observation_id: id.clone() });
            invalid_observations = true;
            invalid_metric_slices.insert(key);
        }
    }

    for group_id in &eval_groups {
        if !seen_evaluation_groups.contains(group_id) {
            issues.push(CalibrationIssue::MissingEvaluationGroup {
                evaluation_group_id: group_id.clone(),
            });
            missing_required_coverage = true;
        }
    }

    let mut slices = Vec::new();
    let mut threshold_failure = false;
    for (key, requirement) in &requirements {
        let accumulator = accumulators.get(key).copied().unwrap_or_default();
        let enough = accumulator.count >= requirement.minimum_observations;
        if !enough {
            issues.push(CalibrationIssue::InsufficientCoverage {
                operating_case_id: key.0.clone(),
                observable_id: key.1.clone(),
                unit_id: key.2.clone(),
                nominal_coverage_bp: key.3,
                minimum: requirement.minimum_observations,
                observed: accumulator.count,
            });
            missing_required_coverage = true;
        }
        let metrics = if invalid_metric_slices.contains(key) {
            None
        } else {
            accumulator.metrics(requirement.normalization_scale)
        };
        if accumulator.count > 0 && metrics.is_none() {
            issues.push(CalibrationIssue::NonFiniteMetric {
                operating_case_id: key.0.clone(),
                observable_id: key.1.clone(),
                unit_id: key.2.clone(),
                nominal_coverage_bp: key.3,
            });
            invalid_observations = true;
        }
        let (
            empirical_coverage,
            wilson_95_lower,
            wilson_95_upper,
            mean_interval_width,
            normalized_mean_interval_width,
            mean_interval_score,
            normalized_mean_interval_score,
        ) = match metrics {
            Some((coverage, lower_ci, upper_ci, width, normalized_width, score, normalized_score)) => (
                Some(coverage),
                Some(lower_ci),
                Some(upper_ci),
                Some(width),
                Some(normalized_width),
                Some(score),
                Some(normalized_score),
            ),
            None => (None, None, None, None, None, None, None),
        };

        let coverage_gate_met = if !enough || empirical_coverage.is_none() {
            None
        } else {
            let lower_ok = wilson_95_lower.is_some_and(|value| {
                value * COVERAGE_BASIS_POINTS as f64
                    >= requirement.minimum_wilson_95_lower_bp as f64
            });
            let upper_ok = wilson_95_upper.is_some_and(|value| {
                value * COVERAGE_BASIS_POINTS as f64
                    <= requirement.maximum_wilson_95_upper_bp as f64
            });
            Some(lower_ok && upper_ok)
        };
        let interval_score_gate_met = if !enough || normalized_mean_interval_score.is_none() {
            None
        } else {
            Some(
                normalized_mean_interval_score
                    .is_some_and(|score| score <= requirement.max_normalized_mean_interval_score),
            )
        };
        let applicability_gate_met = accumulator.count > 0
            && accumulator.within_count == accumulator.count;
        let thresholds_met = if !applicability_gate_met {
            None
        } else {
            match (coverage_gate_met, interval_score_gate_met) {
                (Some(coverage_ok), Some(score_ok)) => Some(coverage_ok && score_ok),
                _ => None,
            }
        };
        if thresholds_met == Some(false) {
            threshold_failure = true;
        }

        slices.push(CalibrationSliceScore {
            operating_case_id: key.0.clone(),
            observable_id: key.1.clone(),
            unit_id: key.2.clone(),
            nominal_coverage_bp: key.3,
            observation_count: accumulator.count,
            required_observation_count: requirement.minimum_observations,
            covered_count: accumulator.covered_count,
            empirical_coverage,
            wilson_95_lower,
            wilson_95_upper,
            mean_interval_width,
            normalized_mean_interval_width,
            mean_interval_score,
            normalized_mean_interval_score,
            within_declared_envelope_count: accumulator.within_count,
            borderline_count: accumulator.borderline_count,
            out_of_domain_count: accumulator.out_of_domain_count,
            unknown_applicability_count: accumulator.unknown_applicability_count,
            applicability_gate_met,
            coverage_gate_met,
            interval_score_gate_met,
            thresholds_met,
        });
    }
    slices.sort_by(|a, b| {
        a.operating_case_id
            .cmp(&b.operating_case_id)
            .then_with(|| a.observable_id.cmp(&b.observable_id))
            .then_with(|| a.unit_id.cmp(&b.unit_id))
            .then_with(|| a.nominal_coverage_bp.cmp(&b.nominal_coverage_bp))
    });
    issues.sort();
    issues.dedup();

    let disposition = if invalid_profile {
        CalibrationDisposition::InvalidProfile
    } else if invalid_observations {
        CalibrationDisposition::InvalidObservations
    } else if threshold_failure {
        CalibrationDisposition::FailDeclaredCalibrationThresholds
    } else if missing_required_coverage {
        CalibrationDisposition::IncompleteCoverage
    } else if domain_not_established {
        CalibrationDisposition::DomainNotEstablished
    } else {
        CalibrationDisposition::PassDeclaredCalibrationThresholds
    };

    CalibrationReport {
        profile_id: profile.profile_id.clone(),
        profile_manifest_sha256: profile.profile_manifest_sha256.clone(),
        disposition,
        input_observation_count: observations.len(),
        unique_observation_count,
        duplicated_observation_record_count,
        duplicated_evaluation_group_record_count,
        slices,
        issues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CAL_MANIFEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const EVAL_MANIFEST: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const PRED_MANIFEST: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const MODEL_MANIFEST: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const SPLIT_MANIFEST: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

    fn requirement() -> CalibrationSliceRequirement {
        CalibrationSliceRequirement {
            operating_case_id: "depth-ratio-1.2".to_owned(),
            observable_id: "resistance".to_owned(),
            unit_id: "N".to_owned(),
            nominal_coverage_bp: 9_000,
            minimum_observations: 2,
            // Broad bounds keep this tiny fixture focused on evaluator arithmetic.
            // Real profiles must preregister evidence-appropriate Wilson bounds.
            minimum_wilson_95_lower_bp: 0,
            maximum_wilson_95_upper_bp: 10_000,
            normalization_scale: 100.0,
            max_normalized_mean_interval_score: 10.0,
        }
    }

    fn profile() -> CalibrationProfile {
        let mut profile = CalibrationProfile {
            profile_id: "dtc-shallow-water-calibration-v1".to_owned(),
            calibration_manifest_sha256: CAL_MANIFEST.to_owned(),
            evaluation_manifest_sha256: EVAL_MANIFEST.to_owned(),
            prediction_manifest_sha256: PRED_MANIFEST.to_owned(),
            model_configuration_sha256: MODEL_MANIFEST.to_owned(),
            split_manifest_sha256: SPLIT_MANIFEST.to_owned(),
            model_frozen_before_holdout: true,
            interval_method_id: "central-residual-quantile".to_owned(),
            interval_method_revision: "r1".to_owned(),
            interval_method_reviewed: true,
            profile_manifest_sha256: String::new(),
            calibration_group_ids: vec!["cal-A".to_owned(), "cal-B".to_owned()],
            evaluation_group_ids: vec!["eval-A".to_owned(), "eval-B".to_owned()],
            required_slices: vec![requirement()],
        };
        profile.profile_manifest_sha256 = canonical_calibration_profile_sha256(&profile);
        profile
    }

    fn observation(id: &str, group: &str, lower: f64, upper: f64, value: f64) -> CalibrationObservation {
        CalibrationObservation {
            observation_id: id.to_owned(),
            evaluation_manifest_sha256: EVAL_MANIFEST.to_owned(),
            prediction_manifest_sha256: PRED_MANIFEST.to_owned(),
            model_configuration_sha256: MODEL_MANIFEST.to_owned(),
            evaluation_group_id: group.to_owned(),
            operating_case_id: "depth-ratio-1.2".to_owned(),
            observable_id: "resistance".to_owned(),
            unit_id: "N".to_owned(),
            nominal_coverage_bp: 9_000,
            lower,
            upper,
            observed_value: value,
            applicability: ApplicabilityStatus::WithinDeclaredEnvelope,
            interval_method_id: "central-residual-quantile".to_owned(),
            interval_method_revision: "r1".to_owned(),
            is_central_interval: true,
            prediction_issued_before_outcome: true,
        }
    }

    fn two_covered() -> Vec<CalibrationObservation> {
        vec![
            observation("obs-1", "eval-A", 8.0, 12.0, 10.0),
            observation("obs-2", "eval-B", 8.0, 12.0, 11.0),
        ]
    }

    fn has_issue(report: &CalibrationReport, expected: &CalibrationIssue) -> bool {
        report.issues.iter().any(|issue| issue == expected)
    }

    #[test]
    fn covered_intervals_report_width_score_and_wilson_bounds() {
        let report = evaluate_calibration(&profile(), &two_covered());
        assert_eq!(report.disposition, CalibrationDisposition::PassDeclaredCalibrationThresholds);
        let slice = &report.slices[0];
        assert_eq!(slice.observation_count, 2);
        assert_eq!(slice.covered_count, 2);
        assert_eq!(slice.empirical_coverage, Some(1.0));
        assert_eq!(slice.mean_interval_width, Some(4.0));
        assert_eq!(slice.mean_interval_score, Some(4.0));
        assert!(slice.wilson_95_lower.unwrap() < 1.0);
        assert_eq!(slice.wilson_95_upper, Some(1.0));
        assert_eq!(slice.thresholds_met, Some(true));
    }

    #[test]
    fn central_interval_score_penalizes_both_width_and_misses() {
        let obs = vec![
            observation("obs-1", "eval-A", 0.0, 10.0, 5.0),
            observation("obs-2", "eval-B", 0.0, 10.0, 11.0),
        ];
        // The permissive coverage gate lets the known-answer score be inspected; the
        // deliberately tight score threshold makes this fixture fail rather than pass.
        let mut p = profile();
        p.required_slices[0].minimum_wilson_95_lower_bp = 0;
        p.required_slices[0].max_normalized_mean_interval_score = 0.15;
        p.profile_manifest_sha256 = canonical_calibration_profile_sha256(&p);
        let report = evaluate_calibration(&p, &obs);
        assert_eq!(report.slices[0].covered_count, 1);
        // Nominal 90% means alpha=0.1 and a miss incurs 20 times the miss distance.
        assert_eq!(report.slices[0].mean_interval_score, Some(20.0));
        assert_eq!(report.slices[0].normalized_mean_interval_score, Some(0.2));
    }

    #[test]
    fn duplicate_observation_ids_are_excluded_and_fail_closed() {
        let mut obs = two_covered();
        obs.push(obs[0].clone());
        let report = evaluate_calibration(&profile(), &obs);
        assert_eq!(report.disposition, CalibrationDisposition::InvalidObservations);
        assert_eq!(report.duplicated_observation_record_count, 2);
        assert_eq!(report.slices[0].observation_count, 1);
    }

    #[test]
    fn distinct_operating_slices_may_have_distinct_independent_group_membership() {
        let mut p = profile();
        p.required_slices[0].minimum_observations = 1;
        let mut second = requirement();
        second.operating_case_id = "depth-ratio-4.0".to_owned();
        second.minimum_observations = 1;
        p.required_slices.push(second);
        p.profile_manifest_sha256 = canonical_calibration_profile_sha256(&p);

        let observations = vec![
            observation("deep-1", "eval-A", 8.0, 12.0, 10.0),
            {
                let mut o = observation("shallow-1", "eval-B", 8.0, 12.0, 10.0);
                o.operating_case_id = "depth-ratio-1.2".to_owned();
                o
            },
        ];
        let report = evaluate_calibration(&p, &observations);
        assert_eq!(report.disposition, CalibrationDisposition::PassDeclaredCalibrationThresholds);
        assert_eq!(report.slices.len(), 2);
        assert!(report.slices.iter().all(|slice| slice.observation_count == 1));
        assert!(report.slices.iter().all(|slice| slice.thresholds_met == Some(true)));
    }

    #[test]
    fn duplicate_rows_from_one_independent_group_cannot_inflate_sample_size() {
        let mut obs = two_covered();
        obs[1].evaluation_group_id = "eval-A".to_owned();
        let report = evaluate_calibration(&profile(), &obs);
        assert_eq!(report.disposition, CalibrationDisposition::InvalidObservations);
        assert_eq!(report.duplicated_evaluation_group_record_count, 2);
        assert_eq!(report.slices[0].observation_count, 0);
        assert!(has_issue(&report, &CalibrationIssue::DuplicateEvaluationGroupObservation {
            evaluation_group_id: "eval-A".to_owned(),
            operating_case_id: "depth-ratio-1.2".to_owned(),
            observable_id: "resistance".to_owned(),
            unit_id: "N".to_owned(),
            nominal_coverage_bp: 9_000,
            count: 2,
        }));
    }

    #[test]
    fn calibration_and_evaluation_groups_must_be_disjoint() {
        let mut p = profile();
        p.evaluation_group_ids[0] = "cal-A".to_owned();
        p.profile_manifest_sha256 = canonical_calibration_profile_sha256(&p);
        let report = evaluate_calibration(&p, &two_covered());
        assert_eq!(report.disposition, CalibrationDisposition::InvalidProfile);
        assert!(has_issue(&report, &CalibrationIssue::CalibrationEvaluationGroupOverlap {
            group_id: "cal-A".to_owned(),
        }));
    }

    #[test]
    fn model_must_be_frozen_before_holdout() {
        let mut p = profile();
        p.model_frozen_before_holdout = false;
        p.profile_manifest_sha256 = canonical_calibration_profile_sha256(&p);
        let report = evaluate_calibration(&p, &two_covered());
        assert_eq!(report.disposition, CalibrationDisposition::InvalidProfile);
        assert!(has_issue(&report, &CalibrationIssue::ModelNotFrozenBeforeHoldout));
    }

    #[test]
    fn wrong_dataset_model_or_prediction_manifest_cannot_pass() {
        for mutate in 0..3 {
            let mut obs = two_covered();
            match mutate {
                0 => obs[0].evaluation_manifest_sha256 = CAL_MANIFEST.to_owned(),
                1 => obs[0].model_configuration_sha256 = CAL_MANIFEST.to_owned(),
                _ => obs[0].prediction_manifest_sha256 = CAL_MANIFEST.to_owned(),
            }
            let report = evaluate_calibration(&profile(), &obs);
            assert_eq!(report.disposition, CalibrationDisposition::InvalidObservations);
            assert_eq!(report.slices[0].observation_count, 1);
        }
    }

    #[test]
    fn unreviewed_interval_method_cannot_pass() {
        let mut p = profile();
        p.interval_method_reviewed = false;
        p.profile_manifest_sha256 = canonical_calibration_profile_sha256(&p);
        let report = evaluate_calibration(&p, &two_covered());
        assert_eq!(report.disposition, CalibrationDisposition::InvalidProfile);
        assert!(has_issue(&report, &CalibrationIssue::IntervalMethodNotReviewed));
    }

    #[test]
    fn noncentral_or_post_outcome_intervals_are_rejected() {
        for mutate in 0..2 {
            let mut obs = two_covered();
            if mutate == 0 {
                obs[0].is_central_interval = false;
            } else {
                obs[0].prediction_issued_before_outcome = false;
            }
            let report = evaluate_calibration(&profile(), &obs);
            assert_eq!(report.disposition, CalibrationDisposition::InvalidObservations);
            assert_eq!(report.slices[0].observation_count, 1);
        }
    }

    #[test]
    fn invalid_order_and_nonfinite_limits_are_rejected() {
        for mutate in 0..2 {
            let mut obs = two_covered();
            if mutate == 0 {
                obs[0].lower = 20.0;
                obs[0].upper = 10.0;
            } else {
                obs[0].lower = f64::NAN;
            }
            let report = evaluate_calibration(&profile(), &obs);
            assert_eq!(report.disposition, CalibrationDisposition::InvalidObservations);
            assert_eq!(report.slices[0].observation_count, 1);
        }
    }

    #[test]
    fn unknown_and_out_of_domain_records_report_metrics_but_never_pass() {
        for status in [ApplicabilityStatus::Unknown, ApplicabilityStatus::OutOfDomain] {
            let mut obs = two_covered();
            obs[0].applicability = status;
            let report = evaluate_calibration(&profile(), &obs);
            assert_eq!(report.disposition, CalibrationDisposition::DomainNotEstablished);
            assert_eq!(report.slices[0].observation_count, 2);
            assert_eq!(report.slices[0].covered_count, 2);
            assert!(!report.slices[0].applicability_gate_met);
            assert_eq!(report.slices[0].thresholds_met, None);
            assert!(has_issue(&report, &CalibrationIssue::ApplicabilityNotEstablished {
                observation_id: "obs-1".to_owned(),
                status,
            }));
        }
    }

    #[test]
    fn incomplete_required_coverage_is_not_a_pass() {
        let report = evaluate_calibration(&profile(), &two_covered()[..1]);
        assert_eq!(report.disposition, CalibrationDisposition::IncompleteCoverage);
        assert_eq!(report.slices[0].thresholds_met, None);
    }

    #[test]
    fn coverage_gate_requires_wilson_interval_inside_frozen_band() {
        let mut p = profile();
        p.required_slices[0].minimum_wilson_95_lower_bp = 8_500;
        p.required_slices[0].maximum_wilson_95_upper_bp = 10_000;
        p.profile_manifest_sha256 = canonical_calibration_profile_sha256(&p);
        let report = evaluate_calibration(&p, &two_covered());
        assert_eq!(report.slices[0].empirical_coverage, Some(1.0));
        assert!(report.slices[0].wilson_95_lower.unwrap() < 0.85);
        assert_eq!(report.slices[0].coverage_gate_met, Some(false));
        assert_eq!(
            report.disposition,
            CalibrationDisposition::FailDeclaredCalibrationThresholds
        );
    }

    #[test]
    fn outlier_misses_cannot_be_hidden_by_width_averages() {
        let mut obs = two_covered();
        obs[1].observed_value = 1_000.0;
        let report = evaluate_calibration(&profile(), &obs);
        assert_eq!(report.disposition, CalibrationDisposition::FailDeclaredCalibrationThresholds);
        assert_eq!(report.slices[0].covered_count, 1);
        assert_eq!(report.slices[0].coverage_gate_met, Some(false));
        assert!(report.slices[0].mean_interval_score.unwrap() > 4.0);
    }

    #[test]
    fn changing_any_threshold_invalidates_the_profile_digest() {
        let mut p = profile();
        p.required_slices[0].max_normalized_mean_interval_score = 0.1;
        let report = evaluate_calibration(&p, &two_covered());
        assert_eq!(report.disposition, CalibrationDisposition::InvalidProfile);
        assert!(has_issue(&report, &CalibrationIssue::ProfileManifestDigestMismatch));
    }

    #[test]
    fn requirement_order_and_input_permutation_are_deterministic() {
        let mut p1 = profile();
        let mut p2 = p1.clone();
        let mut second = requirement();
        second.operating_case_id = "depth-ratio-4.0".to_owned();
        p1.required_slices.push(second.clone());
        p1.profile_manifest_sha256 = canonical_calibration_profile_sha256(&p1);
        p2.required_slices.push(second);
        p2.required_slices.reverse();
        p2.profile_manifest_sha256 = canonical_calibration_profile_sha256(&p2);
        assert_eq!(p1.profile_manifest_sha256, p2.profile_manifest_sha256);

        let obs_a = two_covered();
        let mut obs_b = obs_a.clone();
        obs_b.reverse();
        assert_eq!(
            evaluate_calibration(&p1, &obs_a),
            evaluate_calibration(&p1, &obs_b)
        );
    }

    #[test]
    fn invalid_zero_or_full_nominal_coverage_is_rejected() {
        for nominal in [0, 10_000] {
            let mut p = profile();
            p.required_slices[0].nominal_coverage_bp = nominal;
            p.profile_manifest_sha256 = canonical_calibration_profile_sha256(&p);
            let report = evaluate_calibration(&p, &two_covered());
            assert_eq!(report.disposition, CalibrationDisposition::InvalidProfile);
        }
    }

    #[test]
    fn wilson_boundary_case_is_finite_and_bounded() {
        let (lower, upper) = wilson_95(10, 10).unwrap();
        assert!((0.72..0.73).contains(&lower));
        assert_eq!(upper, 1.0);
        assert_eq!(wilson_95(0, 0), None);
        assert_eq!(wilson_95(11, 10), None);
    }

    #[test]
    fn sha256_matches_standard_known_answer() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
