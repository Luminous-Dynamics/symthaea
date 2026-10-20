//! Paired, same-group comparison of baseline and candidate prediction intervals.
//!
//! This module is deliberately separate from interval calibration and shift-claim admission:
//! it reuses the central interval scoring primitive, computes paired deltas on identical outcomes,
//! and tests declared non-degradation thresholds. It does not implement a shift correction, infer
//! causal effects from observational trials, validate CFD, or authorize operational use.

use crate::prediction_interval_calibration::{
    central_interval_covers, central_interval_score, ApplicabilityStatus,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const COVERAGE_BASIS_POINTS: u16 = 10_000;
const MIN_GROUPS_FOR_NORMAL_CI: usize = 30;
const NORMAL_95_Z: f64 = 1.959_963_984_540_054;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConstraintDirection {
    /// The physical outcome is considered safe when it does not exceed the limit.
    MaximumAllowed,
    /// The physical outcome is considered safe when it is not below the limit.
    MinimumRequired,
}

impl ConstraintDirection {
    fn as_str(self) -> &'static str {
        match self {
            Self::MaximumAllowed => "maximum_allowed",
            Self::MinimumRequired => "minimum_required",
        }
    }
}

/// Optional physical decision contract. The limit and direction must come from the engineering
/// requirement; they are not inferred from prediction outcomes.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineeringConstraint {
    pub direction: ConstraintDirection,
    pub limit: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairedSliceRequirement {
    pub operating_case_id: String,
    pub observable_id: String,
    pub unit_id: String,
    pub nominal_coverage_bp: u16,
    pub minimum_pairs: usize,
    /// Positive, predeclared scale in the observable unit.
    pub normalization_scale: f64,
    /// Maximum tolerated coverage drop: baseline coverage - candidate coverage.
    pub maximum_coverage_drop: f64,
    /// Maximum tolerated increase in normalized central interval score.
    pub maximum_normalized_score_increase: f64,
    /// Maximum tolerated increase in normalized mean interval width.
    pub maximum_normalized_width_increase: f64,
    /// Maximum candidate false-safe rate; denominator is all valid pairs in this slice.
    pub maximum_false_safe_rate: f64,
    pub maximum_missed_feasible_rate: f64,
    pub engineering_constraint: Option<EngineeringConstraint>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairedReplayProfile {
    pub profile_id: String,
    pub evaluation_manifest_sha256: String,
    pub split_manifest_sha256: String,
    pub baseline_prediction_manifest_sha256: String,
    pub candidate_prediction_manifest_sha256: String,
    pub baseline_model_configuration_sha256: String,
    pub candidate_model_configuration_sha256: String,
    pub baseline_method_id: String,
    pub baseline_method_revision: String,
    pub candidate_method_id: String,
    pub candidate_method_revision: String,
    /// Attestation that methods, splits, scoring contract and thresholds were frozen before
    /// the final confirmation outcomes were inspected.
    pub baseline_target_label_group_ids: Vec<String>,
    pub candidate_target_label_group_ids: Vec<String>,
    pub final_confirmation_frozen_before_outcomes: bool,
    /// Maximum allowed difference between highest and lowest candidate coverage across
    /// operating cases for the same observable, unit and nominal coverage level.
    pub maximum_conditional_coverage_gap: f64,
    /// Expected independent evaluation groups. A group can be present in only the operating
    /// slices it actually measured, but every declared group must occur somewhere.
    pub evaluation_group_ids: Vec<String>,
    pub required_slices: Vec<PairedSliceRequirement>,
    /// Canonical digest of all acceptance-relevant profile content except this field.
    pub profile_manifest_sha256: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairedPolicyObservation {
    pub observation_id: String,
    pub evaluation_manifest_sha256: String,
    pub split_manifest_sha256: String,
    pub baseline_prediction_manifest_sha256: String,
    pub candidate_prediction_manifest_sha256: String,
    pub baseline_model_configuration_sha256: String,
    pub candidate_model_configuration_sha256: String,
    pub baseline_method_id: String,
    pub baseline_method_revision: String,
    pub candidate_method_id: String,
    pub candidate_method_revision: String,
    pub evaluation_group_id: String,
    pub operating_case_id: String,
    pub observable_id: String,
    pub unit_id: String,
    pub nominal_coverage_bp: u16,
    /// The same observed outcome is scored against both arms.
    pub observed_value: f64,
    pub baseline_lower: f64,
    pub baseline_upper: f64,
    pub candidate_lower: f64,
    pub candidate_upper: f64,
    pub applicability: ApplicabilityStatus,
    /// An attestation, not cryptographic proof that the final outcome was hidden.
    pub both_predictions_issued_before_outcome: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairedReplayDisposition {
    PassDeclaredNonDegradation,
    FailDeclaredRegression,
    IncompleteCoverage,
    DomainNotEstablished,
    InvalidProfile,
    InvalidObservations,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PairedReplayIssue {
    EmptyProfileField { field: &'static str },
    InvalidProfileDigest { field: &'static str },
    ProfileManifestDigestMismatch,
    ModelNotFrozenBeforeFinalConfirmation,
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
    EmptyGroupIdentifier { field: &'static str },
    DuplicateGroupIdentifier { field: &'static str, group_id: String },
    TargetLabelBudgetRosterMismatch,
    ConditionalGapRequiresMultipleOperatingCases {
        observable_id: String,
        unit_id: String,
        nominal_coverage_bp: u16,
    },
    MissedFeasibleRateExceeded {
        operating_case_id: String,
        observable_id: String,
        unit_id: String,
        nominal_coverage_bp: u16,
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
    DuplicateGroupSlicePair {
        evaluation_group_id: String,
        operating_case_id: String,
        observable_id: String,
        unit_id: String,
        nominal_coverage_bp: u16,
        count: usize,
    },
    EvaluationManifestMismatch { observation_id: String },
    SplitManifestMismatch { observation_id: String },
    BaselinePredictionManifestMismatch { observation_id: String },
    CandidatePredictionManifestMismatch { observation_id: String },
    BaselineModelConfigurationMismatch { observation_id: String },
    CandidateModelConfigurationMismatch { observation_id: String },
    BaselineMethodMismatch { observation_id: String },
    CandidateMethodMismatch { observation_id: String },
    UndeclaredEvaluationGroup { observation_id: String },
    MissingEvaluationGroup { evaluation_group_id: String },
    UndeclaredSlice {
        observation_id: String,
        operating_case_id: String,
        observable_id: String,
        unit_id: String,
        nominal_coverage_bp: u16,
    },
    PredictionIssuedAfterOutcome { observation_id: String },
    InvalidNominalCoverage { observation_id: String },
    InvalidInterval { observation_id: String, arm: &'static str },
    NonFiniteIntervalScore { observation_id: String, arm: &'static str },
    ApplicabilityNotEstablished {
        observation_id: String,
        status: ApplicabilityStatus,
    },
    InsufficientPairs {
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
    FalseSafeRateExceeded {
        operating_case_id: String,
        observable_id: String,
        unit_id: String,
        nominal_coverage_bp: u16,
    },
    CoverageDropExceeded {
        operating_case_id: String,
        observable_id: String,
        unit_id: String,
        nominal_coverage_bp: u16,
    },
    ScoreRegressionExceeded {
        operating_case_id: String,
        observable_id: String,
        unit_id: String,
        nominal_coverage_bp: u16,
    },
    WidthRegressionExceeded {
        operating_case_id: String,
        observable_id: String,
        unit_id: String,
        nominal_coverage_bp: u16,
    },
    ConditionalCoverageGapExceeded {
        observable_id: String,
        unit_id: String,
        nominal_coverage_bp: u16,
        maximum: String,
        observed: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairedSliceScore {
    pub operating_case_id: String,
    pub observable_id: String,
    pub unit_id: String,
    pub nominal_coverage_bp: u16,
    pub pair_count: usize,
    pub required_pair_count: usize,
    pub baseline_coverage: Option<f64>,
    pub candidate_coverage: Option<f64>,
    /// Candidate minus baseline; negative means lower candidate coverage.
    pub coverage_delta: Option<f64>,
    pub coverage_delta_ci95_lower: Option<f64>,
    pub coverage_delta_ci95_upper: Option<f64>,
    pub baseline_mean_interval_width: Option<f64>,
    pub candidate_mean_interval_width: Option<f64>,
    /// Candidate minus baseline; negative means narrower candidate intervals.
    pub mean_interval_width_delta: Option<f64>,
    pub normalized_mean_interval_width_delta: Option<f64>,
    pub width_delta_ci95_lower: Option<f64>,
    pub width_delta_ci95_upper: Option<f64>,
    pub baseline_mean_interval_score: Option<f64>,
    pub candidate_mean_interval_score: Option<f64>,
    /// Candidate minus baseline; negative is better.
    pub mean_interval_score_delta: Option<f64>,
    pub normalized_mean_interval_score_delta: Option<f64>,
    pub score_delta_ci95_lower: Option<f64>,
    pub score_delta_ci95_upper: Option<f64>,
    /// Available only when >= 30 independent paired groups are present. Normal approximation
    /// is descriptive; it is not a substitute for design-specific small-sample inference.
    pub paired_ci_method: Option<&'static str>,
    pub baseline_false_safe_count: usize,
    pub candidate_false_safe_count: usize,
    pub baseline_missed_feasible_count: usize,
    pub candidate_missed_feasible_count: usize,
    /// Counts divided by all valid pairs, making denominators comparable across arms.
    pub baseline_false_safe_rate: Option<f64>,
    pub candidate_false_safe_rate: Option<f64>,
    pub baseline_missed_feasible_rate: Option<f64>,
    pub candidate_missed_feasible_rate: Option<f64>,
    pub within_declared_envelope_count: usize,
    pub borderline_count: usize,
    pub out_of_domain_count: usize,
    pub unknown_applicability_count: usize,
    pub applicability_gate_met: bool,
    pub coverage_non_degradation_gate_met: Option<bool>,
    pub score_non_degradation_gate_met: Option<bool>,
    pub width_non_degradation_gate_met: Option<bool>,
    pub false_safe_gate_met: Option<bool>,
    pub missed_feasible_gate_met: Option<bool>,
    pub thresholds_met: Option<bool>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConditionalCoverageGap {
    pub observable_id: String,
    pub unit_id: String,
    pub nominal_coverage_bp: u16,
    pub baseline_min_coverage: Option<f64>,
    pub baseline_max_coverage: Option<f64>,
    pub baseline_gap: Option<f64>,
    pub candidate_min_coverage: Option<f64>,
    pub candidate_max_coverage: Option<f64>,
    pub candidate_gap: Option<f64>,
    pub candidate_worst_operating_case_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairedReplayReport {
    pub profile_id: String,
    pub profile_manifest_sha256: String,
    pub disposition: PairedReplayDisposition,
    pub input_observation_count: usize,
    pub scored_pair_count: usize,
    pub duplicated_observation_record_count: usize,
    pub duplicated_group_slice_record_count: usize,
    pub slices: Vec<PairedSliceScore>,
    pub conditional_coverage_gaps: Vec<ConditionalCoverageGap>,
    pub issues: Vec<PairedReplayIssue>,
}

type SliceKey = (String, String, String, u16);
type PairKey = (String, SliceKey);

#[derive(Default)]
struct CanonicalEncoder(Vec<u8>);

impl CanonicalEncoder {
    fn field(&mut self, name: &str, value: &[u8]) {
        self.0.extend_from_slice(&(name.len() as u64).to_be_bytes());
        self.0.extend_from_slice(name.as_bytes());
        self.0.extend_from_slice(&(value.len() as u64).to_be_bytes());
        self.0.extend_from_slice(value);
    }

    fn text(&mut self, name: &str, value: &str) {
        self.field(name, value.as_bytes());
    }

    fn boolean(&mut self, name: &str, value: bool) {
        self.field(name, &[u8::from(value)]);
    }

    fn usize(&mut self, name: &str, value: usize) {
        self.field(name, &(value as u64).to_be_bytes());
    }

    fn number(&mut self, name: &str, value: f64) {
        self.field(name, &value.to_bits().to_be_bytes());
    }

    fn strings(&mut self, name: &str, values: &[String]) {
        let mut sorted: Vec<&str> = values.iter().map(String::as_str).collect();
        sorted.sort_unstable();
        self.usize(&format!("{name}.count"), sorted.len());
        for value in sorted {
            self.text(name, value);
        }
    }
}

fn hex_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.as_bytes().iter().all(u8::is_ascii_hexdigit)
}

fn slice_key(
    operating_case_id: &str,
    observable_id: &str,
    unit_id: &str,
    nominal_coverage_bp: u16,
) -> SliceKey {
    (
        operating_case_id.to_owned(),
        observable_id.to_owned(),
        unit_id.to_owned(),
        nominal_coverage_bp,
    )
}

fn push_requirement(encoder: &mut CanonicalEncoder, requirement: &PairedSliceRequirement) {
    encoder.text("slice.operating_case_id", &requirement.operating_case_id);
    encoder.text("slice.observable_id", &requirement.observable_id);
    encoder.text("slice.unit_id", &requirement.unit_id);
    encoder.usize("slice.nominal_coverage_bp", requirement.nominal_coverage_bp as usize);
    encoder.usize("slice.minimum_pairs", requirement.minimum_pairs);
    encoder.number("slice.normalization_scale", requirement.normalization_scale);
    encoder.number("slice.maximum_coverage_drop", requirement.maximum_coverage_drop);
    encoder.number(
        "slice.maximum_normalized_score_increase",
        requirement.maximum_normalized_score_increase,
    );
    encoder.number(
        "slice.maximum_normalized_width_increase",
        requirement.maximum_normalized_width_increase,
    );
    encoder.number("slice.maximum_false_safe_rate", requirement.maximum_false_safe_rate);
    match &requirement.engineering_constraint {
        Some(constraint) => {
            encoder.boolean("slice.constraint.present", true);
            encoder.text("slice.constraint.direction", constraint.direction.as_str());
            encoder.number("slice.constraint.limit", constraint.limit);
        }
        None => encoder.boolean("slice.constraint.present", false),
    }
}

/// Compute the canonical SHA-256 for a paired replay profile. Slice and group roster ordering
/// does not affect the digest; duplicate roster entries remain represented and are rejected by
/// assessment. The digest field itself is excluded from the canonical preimage.
pub fn canonical_paired_replay_profile_sha256(profile: &PairedReplayProfile) -> String {
    let mut encoder = CanonicalEncoder::default();
    encoder.text("domain", "marine-paired-policy-replay-v0");
    encoder.text("profile_id", &profile.profile_id);
    encoder.text("evaluation_manifest_sha256", &profile.evaluation_manifest_sha256);
    encoder.text("split_manifest_sha256", &profile.split_manifest_sha256);
    encoder.text(
        "baseline_prediction_manifest_sha256",
        &profile.baseline_prediction_manifest_sha256,
    );
    encoder.text(
        "candidate_prediction_manifest_sha256",
        &profile.candidate_prediction_manifest_sha256,
    );
    encoder.text(
        "baseline_model_configuration_sha256",
        &profile.baseline_model_configuration_sha256,
    );
    encoder.text(
        "candidate_model_configuration_sha256",
        &profile.candidate_model_configuration_sha256,
    );
    encoder.text("baseline_method_id", &profile.baseline_method_id);
    encoder.text("baseline_method_revision", &profile.baseline_method_revision);
    encoder.text("candidate_method_id", &profile.candidate_method_id);
    encoder.text("candidate_method_revision", &profile.candidate_method_revision);
    encoder.strings("baseline_target_label_group_ids", &profile.baseline_target_label_group_ids);
    encoder.strings("candidate_target_label_group_ids", &profile.candidate_target_label_group_ids);
    encoder.boolean(
        "final_confirmation_frozen_before_outcomes",
        profile.final_confirmation_frozen_before_outcomes,
    );
    encoder.number(
        "maximum_conditional_coverage_gap",
        profile.maximum_conditional_coverage_gap,
    );
    encoder.strings("evaluation_group_ids", &profile.evaluation_group_ids);
    let mut requirements = profile.required_slices.clone();
    requirements.sort_by(|a, b| {
        (
            &a.operating_case_id,
            &a.observable_id,
            &a.unit_id,
            a.nominal_coverage_bp,
        )
            .cmp(&(
                &b.operating_case_id,
                &b.observable_id,
                &b.unit_id,
                b.nominal_coverage_bp,
            ))
    });
    encoder.usize("required_slices.count", requirements.len());
    for requirement in &requirements {
        push_requirement(&mut encoder, requirement);
    }
    hex_sha256(&encoder.0)
}

fn validate_profile(profile: &PairedReplayProfile) -> (bool, Vec<PairedReplayIssue>) {
    let mut issues = Vec::new();
    for (value, field) in [
        (profile.profile_id.as_str(), "profile_id"),
        (profile.baseline_method_id.as_str(), "baseline_method_id"),
        (profile.baseline_method_revision.as_str(), "baseline_method_revision"),
        (profile.candidate_method_id.as_str(), "candidate_method_id"),
        (profile.candidate_method_revision.as_str(), "candidate_method_revision"),
    ] {
        if value.trim().is_empty() {
            issues.push(PairedReplayIssue::EmptyProfileField { field });
        }
    }
    for (value, field) in [
        (profile.evaluation_manifest_sha256.as_str(), "evaluation_manifest_sha256"),
        (profile.split_manifest_sha256.as_str(), "split_manifest_sha256"),
        (
            profile.baseline_prediction_manifest_sha256.as_str(),
            "baseline_prediction_manifest_sha256",
        ),
        (
            profile.candidate_prediction_manifest_sha256.as_str(),
            "candidate_prediction_manifest_sha256",
        ),
        (
            profile.baseline_model_configuration_sha256.as_str(),
            "baseline_model_configuration_sha256",
        ),
        (
            profile.candidate_model_configuration_sha256.as_str(),
            "candidate_model_configuration_sha256",
        ),
    ] {
        if !valid_sha256(value) {
            issues.push(PairedReplayIssue::InvalidProfileDigest { field });
        }
    }
    if profile.profile_manifest_sha256 != canonical_paired_replay_profile_sha256(profile)
        || !valid_sha256(&profile.profile_manifest_sha256)
    {
        issues.push(PairedReplayIssue::ProfileManifestDigestMismatch);
    }
    if !profile.final_confirmation_frozen_before_outcomes {
        issues.push(PairedReplayIssue::ModelNotFrozenBeforeFinalConfirmation);
    }
    if !profile.maximum_conditional_coverage_gap.is_finite()
        || !(0.0..=1.0).contains(&profile.maximum_conditional_coverage_gap)
    {
        issues.push(PairedReplayIssue::InvalidRequirementField {
            operating_case_id: String::new(),
            observable_id: String::new(),
            field: "maximum_conditional_coverage_gap",
        });
    }
    if profile.required_slices.is_empty() {
        issues.push(PairedReplayIssue::NoRequiredSlices);
    }

    let mut group_ids = BTreeSet::new();
    for group_id in &profile.evaluation_group_ids {
        if group_id.trim().is_empty() {
            issues.push(PairedReplayIssue::EmptyGroupIdentifier {
                field: "evaluation_group_ids",
            });
        } else if !group_ids.insert(group_id.as_str()) {
            issues.push(PairedReplayIssue::DuplicateGroupIdentifier {
                field: "evaluation_group_ids",
                group_id: group_id.clone(),
            });
        }
    }

    let mut seen_slices = BTreeSet::new();
    for requirement in &profile.required_slices {
        let key = slice_key(
            &requirement.operating_case_id,
            &requirement.observable_id,
            &requirement.unit_id,
            requirement.nominal_coverage_bp,
        );
        if !seen_slices.insert(key.clone()) {
            issues.push(PairedReplayIssue::DuplicateRequiredSlice {
                operating_case_id: key.0,
                observable_id: key.1,
                unit_id: key.2,
                nominal_coverage_bp: key.3,
            });
        }
        for (value, field) in [
            (requirement.operating_case_id.as_str(), "operating_case_id"),
            (requirement.observable_id.as_str(), "observable_id"),
            (requirement.unit_id.as_str(), "unit_id"),
        ] {
            if value.trim().is_empty() {
                issues.push(PairedReplayIssue::InvalidRequirementField {
                    operating_case_id: requirement.operating_case_id.clone(),
                    observable_id: requirement.observable_id.clone(),
                    field,
                });
            }
        }
        if requirement.nominal_coverage_bp == 0
            || requirement.nominal_coverage_bp >= COVERAGE_BASIS_POINTS
        {
            issues.push(PairedReplayIssue::InvalidRequirementField {
                operating_case_id: requirement.operating_case_id.clone(),
                observable_id: requirement.observable_id.clone(),
                field: "nominal_coverage_bp",
            });
        }
        if requirement.minimum_pairs == 0 {
            issues.push(PairedReplayIssue::InvalidRequirementField {
                operating_case_id: requirement.operating_case_id.clone(),
                observable_id: requirement.observable_id.clone(),
                field: "minimum_pairs",
            });
        }
        if !requirement.normalization_scale.is_finite() || requirement.normalization_scale <= 0.0 {
            issues.push(PairedReplayIssue::InvalidRequirementField {
                operating_case_id: requirement.operating_case_id.clone(),
                observable_id: requirement.observable_id.clone(),
                field: "normalization_scale",
            });
        }
        for (value, field) in [
            (requirement.maximum_coverage_drop, "maximum_coverage_drop"),
            (requirement.maximum_false_safe_rate, "maximum_false_safe_rate"),
            (requirement.maximum_missed_feasible_rate, "maximum_missed_feasible_rate"),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                issues.push(PairedReplayIssue::InvalidRequirementField {
                    operating_case_id: requirement.operating_case_id.clone(),
                    observable_id: requirement.observable_id.clone(),
                    field,
                });
            }
        }
        for (value, field) in [
            (
                requirement.maximum_normalized_score_increase,
                "maximum_normalized_score_increase",
            ),
            (
                requirement.maximum_normalized_width_increase,
                "maximum_normalized_width_increase",
            ),
        ] {
            if !value.is_finite() || value < 0.0 {
                issues.push(PairedReplayIssue::InvalidRequirementField {
                    operating_case_id: requirement.operating_case_id.clone(),
                    observable_id: requirement.observable_id.clone(),
                    field,
                });
            }
        }
        if requirement
            .engineering_constraint
            .as_ref()
            .is_some_and(|constraint| !constraint.limit.is_finite())
        {
            issues.push(PairedReplayIssue::InvalidRequirementField {
                operating_case_id: requirement.operating_case_id.clone(),
                observable_id: requirement.observable_id.clone(),
                field: "engineering_constraint.limit",
            });
        }
    }

    issues.sort();
    issues.dedup();
    (issues.is_empty(), issues)
}

fn pair_is_safe(lower: f64, upper: f64, constraint: &EngineeringConstraint) -> bool {
    match constraint.direction {
        ConstraintDirection::MaximumAllowed => upper <= constraint.limit,
        ConstraintDirection::MinimumRequired => lower >= constraint.limit,
    }
}

fn outcome_is_safe(value: f64, constraint: &EngineeringConstraint) -> bool {
    match constraint.direction {
        ConstraintDirection::MaximumAllowed => value <= constraint.limit,
        ConstraintDirection::MinimumRequired => value >= constraint.limit,
    }
}

#[derive(Default)]
struct Moments {
    count: usize,
    mean: f64,
    m2: f64,
}

impl Moments {
    fn add(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta_after = value - self.mean;
        self.m2 += delta * delta_after;
    }

    fn mean(&self) -> Option<f64> {
        (self.count > 0 && self.mean.is_finite()).then_some(self.mean)
    }

    fn normal_ci95(&self) -> Option<(f64, f64)> {
        if self.count < MIN_GROUPS_FOR_NORMAL_CI || !self.mean.is_finite() || !self.m2.is_finite() {
            return None;
        }
        let variance = self.m2 / (self.count - 1) as f64;
        if !variance.is_finite() || variance < 0.0 {
            return None;
        }
        let half = NORMAL_95_Z * (variance / self.count as f64).sqrt();
        let lower = self.mean - half;
        let upper = self.mean + half;
        (lower.is_finite() && upper.is_finite()).then_some((lower, upper))
    }
}

#[derive(Default)]
struct SliceAccumulator {
    baseline_cover_count: usize,
    candidate_cover_count: usize,
    coverage_delta: Moments,
    baseline_width: Moments,
    candidate_width: Moments,
    width_delta: Moments,
    baseline_score: Moments,
    candidate_score: Moments,
    score_delta: Moments,
    baseline_false_safe_count: usize,
    candidate_false_safe_count: usize,
    baseline_missed_feasible_count: usize,
    candidate_missed_feasible_count: usize,
    within_count: usize,
    borderline_count: usize,
    out_of_domain_count: usize,
    unknown_count: usize,
}

impl SliceAccumulator {
    fn add(&mut self, observation: &PairedPolicyObservation, requirement: &PairedSliceRequirement) {
        let baseline_covers = central_interval_covers(
            observation.baseline_lower,
            observation.baseline_upper,
            observation.observed_value,
        ).unwrap_or(false);
        let candidate_covers = central_interval_covers(
            observation.candidate_lower,
            observation.candidate_upper,
            observation.observed_value,
        ).unwrap_or(false);
        self.baseline_cover_count += usize::from(baseline_covers);
        self.candidate_cover_count += usize::from(candidate_covers);
        self.coverage_delta.add(
            f64::from(candidate_covers) - f64::from(baseline_covers),
        );

        let baseline_width = observation.baseline_upper - observation.baseline_lower;
        let candidate_width = observation.candidate_upper - observation.candidate_lower;
        self.baseline_width.add(baseline_width);
        self.candidate_width.add(candidate_width);
        self.width_delta.add(candidate_width - baseline_width);

        let baseline_score = central_interval_score(
            observation.baseline_lower,
            observation.baseline_upper,
            observation.observed_value,
            observation.nominal_coverage_bp,
        ).unwrap_or(f64::NAN);
        let candidate_score = central_interval_score(
            observation.candidate_lower,
            observation.candidate_upper,
            observation.observed_value,
            observation.nominal_coverage_bp,
        ).unwrap_or(f64::NAN);
        self.baseline_score.add(baseline_score);
        self.candidate_score.add(candidate_score);
        self.score_delta.add(candidate_score - baseline_score);

        if let Some(constraint) = &requirement.engineering_constraint {
            let actual_safe = outcome_is_safe(observation.observed_value, constraint);
            let baseline_predicted_safe = pair_is_safe(
                observation.baseline_lower,
                observation.baseline_upper,
                constraint,
            );
            let candidate_predicted_safe = pair_is_safe(
                observation.candidate_lower,
                observation.candidate_upper,
                constraint,
            );
            self.baseline_false_safe_count +=
                usize::from(baseline_predicted_safe && !actual_safe);
            self.candidate_false_safe_count +=
                usize::from(candidate_predicted_safe && !actual_safe);
            self.baseline_missed_feasible_count +=
                usize::from(!baseline_predicted_safe && actual_safe);
            self.candidate_missed_feasible_count +=
                usize::from(!candidate_predicted_safe && actual_safe);
        }

        match observation.applicability {
            ApplicabilityStatus::WithinDeclaredEnvelope => self.within_count += 1,
            ApplicabilityStatus::Borderline => self.borderline_count += 1,
            ApplicabilityStatus::OutOfDomain => self.out_of_domain_count += 1,
            ApplicabilityStatus::Unknown => self.unknown_count += 1,
        }
    }
}

/// Compare baseline and candidate on identical held-out outcomes, pair roster and operating
/// slices. The output separates non-degradation from statistical evidence of improvement.
pub fn evaluate_paired_replay(
    profile: &PairedReplayProfile,
    observations: &[PairedPolicyObservation],
) -> PairedReplayReport {
    let (profile_valid, mut issues) = validate_profile(profile);
    let expected_manifest = &profile.evaluation_manifest_sha256;
    let expected_split = &profile.split_manifest_sha256;
    let required: BTreeMap<SliceKey, &PairedSliceRequirement> = profile
        .required_slices
        .iter()
        .map(|requirement| {
            (
                slice_key(
                    &requirement.operating_case_id,
                    &requirement.observable_id,
                    &requirement.unit_id,
                    requirement.nominal_coverage_bp,
                ),
                requirement,
            )
        })
        .collect();
    let declared_groups: BTreeSet<&str> = profile
        .evaluation_group_ids
        .iter()
        .map(String::as_str)
        .collect();

    let mut valid_rows = Vec::<(usize, &PairedPolicyObservation, SliceKey)>::new();
    let mut observation_id_counts = BTreeMap::<&str, usize>::new();
    let mut pair_counts = BTreeMap::<PairKey, usize>::new();

    for (index, observation) in observations.iter().enumerate() {
        *observation_id_counts
            .entry(observation.observation_id.as_str())
            .or_default() += 1;

        let key = slice_key(
            &observation.operating_case_id,
            &observation.observable_id,
            &observation.unit_id,
            observation.nominal_coverage_bp,
        );
        let pair_key = (observation.evaluation_group_id.clone(), key.clone());
        *pair_counts.entry(pair_key).or_default() += 1;

        let mut row_valid = true;
        if observation.observation_id.trim().is_empty() {
            issues.push(PairedReplayIssue::EmptyObservationField {
                observation_id: observation.observation_id.clone(),
                field: "observation_id",
            });
            row_valid = false;
        }
        for (value, field) in [
            (observation.evaluation_manifest_sha256.as_str(), "evaluation_manifest_sha256"),
            (observation.split_manifest_sha256.as_str(), "split_manifest_sha256"),
            (
                observation.baseline_prediction_manifest_sha256.as_str(),
                "baseline_prediction_manifest_sha256",
            ),
            (
                observation.candidate_prediction_manifest_sha256.as_str(),
                "candidate_prediction_manifest_sha256",
            ),
            (
                observation.baseline_model_configuration_sha256.as_str(),
                "baseline_model_configuration_sha256",
            ),
            (
                observation.candidate_model_configuration_sha256.as_str(),
                "candidate_model_configuration_sha256",
            ),
        ] {
            if !valid_sha256(value) {
                issues.push(PairedReplayIssue::InvalidObservationDigest {
                    observation_id: observation.observation_id.clone(),
                    field,
                });
                row_valid = false;
            }
        }
        if observation.evaluation_manifest_sha256 != *expected_manifest {
            issues.push(PairedReplayIssue::EvaluationManifestMismatch {
                observation_id: observation.observation_id.clone(),
            });
            row_valid = false;
        }
        if observation.split_manifest_sha256 != *expected_split {
            issues.push(PairedReplayIssue::SplitManifestMismatch {
                observation_id: observation.observation_id.clone(),
            });
            row_valid = false;
        }
        if observation.baseline_prediction_manifest_sha256
            != profile.baseline_prediction_manifest_sha256
        {
            issues.push(PairedReplayIssue::BaselinePredictionManifestMismatch {
                observation_id: observation.observation_id.clone(),
            });
            row_valid = false;
        }
        if observation.candidate_prediction_manifest_sha256
            != profile.candidate_prediction_manifest_sha256
        {
            issues.push(PairedReplayIssue::CandidatePredictionManifestMismatch {
                observation_id: observation.observation_id.clone(),
            });
            row_valid = false;
        }
        if observation.baseline_model_configuration_sha256
            != profile.baseline_model_configuration_sha256
        {
            issues.push(PairedReplayIssue::BaselineModelConfigurationMismatch {
                observation_id: observation.observation_id.clone(),
            });
            row_valid = false;
        }
        if observation.candidate_model_configuration_sha256
            != profile.candidate_model_configuration_sha256
        {
            issues.push(PairedReplayIssue::CandidateModelConfigurationMismatch {
                observation_id: observation.observation_id.clone(),
            });
            row_valid = false;
        }
        if observation.baseline_method_id != profile.baseline_method_id
            || observation.baseline_method_revision != profile.baseline_method_revision
        {
            issues.push(PairedReplayIssue::BaselineMethodMismatch {
                observation_id: observation.observation_id.clone(),
            });
            row_valid = false;
        }
        if observation.candidate_method_id != profile.candidate_method_id
            || observation.candidate_method_revision != profile.candidate_method_revision
        {
            issues.push(PairedReplayIssue::CandidateMethodMismatch {
                observation_id: observation.observation_id.clone(),
            });
            row_valid = false;
        }
        if !declared_groups.contains(observation.evaluation_group_id.as_str()) {
            issues.push(PairedReplayIssue::UndeclaredEvaluationGroup {
                observation_id: observation.observation_id.clone(),
            });
            row_valid = false;
        }
        if !required.contains_key(&key) {
            issues.push(PairedReplayIssue::UndeclaredSlice {
                observation_id: observation.observation_id.clone(),
                operating_case_id: key.0.clone(),
                observable_id: key.1.clone(),
                unit_id: key.2.clone(),
                nominal_coverage_bp: key.3,
            });
            row_valid = false;
        }
        for (value, field) in [
            (observation.operating_case_id.as_str(), "operating_case_id"),
            (observation.observable_id.as_str(), "observable_id"),
            (observation.unit_id.as_str(), "unit_id"),
            (observation.evaluation_group_id.as_str(), "evaluation_group_id"),
        ] {
            if value.trim().is_empty() {
                issues.push(PairedReplayIssue::EmptyObservationField {
                    observation_id: observation.observation_id.clone(),
                    field,
                });
                row_valid = false;
            }
        }
        if observation.nominal_coverage_bp == 0
            || observation.nominal_coverage_bp >= COVERAGE_BASIS_POINTS
        {
            issues.push(PairedReplayIssue::InvalidNominalCoverage {
                observation_id: observation.observation_id.clone(),
            });
            row_valid = false;
        }
        if !observation.both_predictions_issued_before_outcome {
            issues.push(PairedReplayIssue::PredictionIssuedAfterOutcome {
                observation_id: observation.observation_id.clone(),
            });
            row_valid = false;
        }
        if !observation.observed_value.is_finite() {
            issues.push(PairedReplayIssue::InvalidInterval {
                observation_id: observation.observation_id.clone(),
                arm: "observed_value",
            });
            row_valid = false;
        }
        for (arm, lower, upper) in [
            ("baseline", observation.baseline_lower, observation.baseline_upper),
            ("candidate", observation.candidate_lower, observation.candidate_upper),
        ] {
            if !lower.is_finite() || !upper.is_finite() || lower > upper {
                issues.push(PairedReplayIssue::InvalidInterval {
                    observation_id: observation.observation_id.clone(),
                    arm,
                });
                row_valid = false;
            } else if central_interval_score(
                lower,
                upper,
                observation.observed_value,
                observation.nominal_coverage_bp,
            ).is_none() {
                issues.push(PairedReplayIssue::NonFiniteIntervalScore {
                    observation_id: observation.observation_id.clone(),
                    arm,
                });
                row_valid = false;
            }
        }
        if observation.applicability != ApplicabilityStatus::WithinDeclaredEnvelope {
            issues.push(PairedReplayIssue::ApplicabilityNotEstablished {
                observation_id: observation.observation_id.clone(),
                status: observation.applicability,
            });
        }
        if row_valid {
            valid_rows.push((index, observation, key));
        }
    }

    let mut duplicate_ids = BTreeSet::new();
    for (observation_id, count) in observation_id_counts {
        if count > 1 {
            issues.push(PairedReplayIssue::DuplicateObservationId {
                observation_id: observation_id.to_string(),
                count,
            });
            duplicate_ids.insert(observation_id);
        }
    }

    let mut duplicate_pairs = BTreeSet::new();
    for ((group_id, key), count) in pair_counts {
        if count > 1 {
            issues.push(PairedReplayIssue::DuplicateGroupSlicePair {
                evaluation_group_id: group_id.clone(),
                operating_case_id: key.0.clone(),
                observable_id: key.1.clone(),
                unit_id: key.2.clone(),
                nominal_coverage_bp: key.3,
                count,
            });
            duplicate_pairs.insert((group_id, key));
        }
    }

    let mut accounted_groups = BTreeSet::new();
    let mut by_slice = BTreeMap::<SliceKey, Vec<&PairedPolicyObservation>>::new();
    let mut scored_pairs = 0_usize;
    for (index, observation, key) in valid_rows {
        let _ = index;
        if duplicate_ids.contains(observation.observation_id.as_str())
            || duplicate_pairs.contains(&(observation.evaluation_group_id.clone(), key.clone()))
        {
            continue;
        }
        accounted_groups.insert(observation.evaluation_group_id.as_str());
        by_slice.entry(key).or_default().push(observation);
        scored_pairs += 1;
    }

    for group_id in &profile.evaluation_group_ids {
        if !accounted_groups.contains(group_id.as_str()) {
            issues.push(PairedReplayIssue::MissingEvaluationGroup {
                evaluation_group_id: group_id.clone(),
            });
        }
    }

    let mut slice_scores = Vec::new();
    let mut domain_not_established = false;
    let mut regression_found = false;

    for (key, requirement) in &required {
        let rows = by_slice.get(key).cloned().unwrap_or_default();
        let pair_count = rows.len();
        if pair_count < requirement.minimum_pairs {
            issues.push(PairedReplayIssue::InsufficientPairs {
                operating_case_id: key.0.clone(),
                observable_id: key.1.clone(),
                unit_id: key.2.clone(),
                nominal_coverage_bp: key.3,
                minimum: requirement.minimum_pairs,
                observed: pair_count,
            });
        }

        let mut acc = SliceAccumulator::default();
        for observation in &rows {
            acc.add(observation, requirement);
        }
        let n = pair_count as f64;
        let baseline_coverage = (pair_count > 0).then_some(acc.baseline_cover_count as f64 / n);
        let candidate_coverage = (pair_count > 0).then_some(acc.candidate_cover_count as f64 / n);
        let coverage_delta = candidate_coverage.zip(baseline_coverage).map(|(candidate, baseline)| {
            candidate - baseline
        });

        let baseline_mean_width = acc.baseline_width.mean();
        let candidate_mean_width = acc.candidate_width.mean();
        let width_delta = acc.width_delta.mean();
        let normalized_width_delta = width_delta.map(|value| value / requirement.normalization_scale);
        let width_ci = acc.width_delta.normal_ci95();

        let baseline_mean_score = acc.baseline_score.mean();
        let candidate_mean_score = acc.candidate_score.mean();
        let score_delta = acc.score_delta.mean();
        let normalized_score_delta = score_delta.map(|value| value / requirement.normalization_scale);
        let score_ci = acc.score_delta.normal_ci95();

        let baseline_false_safe_rate = requirement.engineering_constraint.as_ref().and_then(|_| {
            (pair_count > 0).then_some(acc.baseline_false_safe_count as f64 / n)
        });
        let candidate_false_safe_rate = requirement.engineering_constraint.as_ref().and_then(|_| {
            (pair_count > 0).then_some(acc.candidate_false_safe_count as f64 / n)
        });

        let applicability_gate_met = acc.borderline_count == 0
            && acc.out_of_domain_count == 0
            && acc.unknown_count == 0;
        domain_not_established |= !applicability_gate_met;

        let enough = pair_count >= requirement.minimum_pairs && pair_count > 0;
        let coverage_gate_met = coverage_delta.map(|delta| delta >= -requirement.maximum_coverage_drop);
        let score_gate_met = normalized_score_delta
            .map(|delta| delta <= requirement.maximum_normalized_score_increase);
        let width_gate_met = normalized_width_delta
            .map(|delta| delta <= requirement.maximum_normalized_width_increase);
        let false_safe_gate_met = if requirement.engineering_constraint.is_some() {
            candidate_false_safe_rate.map(|rate| rate <= requirement.maximum_false_safe_rate)
        } else {
            Some(true)
        };
        let thresholds_met = if !enough || !applicability_gate_met {
            None
        } else {
            match (coverage_gate_met, score_gate_met, width_gate_met, false_safe_gate_met) {
                (Some(coverage), Some(score), Some(width), Some(false_safe)) => {
                    Some(coverage && score && width && false_safe)
                }
                _ => None,
            }
        };

        if enough && applicability_gate_met {
            if coverage_gate_met == Some(false) {
                issues.push(PairedReplayIssue::CoverageDropExceeded {
                    operating_case_id: key.0.clone(),
                    observable_id: key.1.clone(),
                    unit_id: key.2.clone(),
                    nominal_coverage_bp: key.3,
                });
                regression_found = true;
            }
            if score_gate_met == Some(false) {
                issues.push(PairedReplayIssue::ScoreRegressionExceeded {
                    operating_case_id: key.0.clone(),
                    observable_id: key.1.clone(),
                    unit_id: key.2.clone(),
                    nominal_coverage_bp: key.3,
                });
                regression_found = true;
            }
            if width_gate_met == Some(false) {
                issues.push(PairedReplayIssue::WidthRegressionExceeded {
                    operating_case_id: key.0.clone(),
                    observable_id: key.1.clone(),
                    unit_id: key.2.clone(),
                    nominal_coverage_bp: key.3,
                });
                regression_found = true;
            }
            if false_safe_gate_met == Some(false) {
                issues.push(PairedReplayIssue::FalseSafeRateExceeded {
                    operating_case_id: key.0.clone(),
                    observable_id: key.1.clone(),
                    unit_id: key.2.clone(),
                    nominal_coverage_bp: key.3,
                });
                regression_found = true;
            }
        }

        if [
            baseline_coverage,
            candidate_coverage,
            baseline_mean_width,
            candidate_mean_width,
            baseline_mean_score,
            candidate_mean_score,
        ].iter().flatten().any(|metric| !metric.is_finite()) {
            issues.push(PairedReplayIssue::NonFiniteMetric {
                operating_case_id: key.0.clone(),
                observable_id: key.1.clone(),
                unit_id: key.2.clone(),
                nominal_coverage_bp: key.3,
            });
            regression_found = true;
        }

        slice_scores.push(PairedSliceScore {
            operating_case_id: key.0.clone(),
            observable_id: key.1.clone(),
            unit_id: key.2.clone(),
            nominal_coverage_bp: key.3,
            pair_count,
            required_pair_count: requirement.minimum_pairs,
            baseline_coverage,
            candidate_coverage,
            coverage_delta,
            baseline_mean_interval_width: baseline_mean_width,
            candidate_mean_interval_width: candidate_mean_width,
            mean_interval_width_delta: width_delta,
            normalized_mean_interval_width_delta: normalized_width_delta,
            width_delta_ci95_lower: width_ci.map(|ci| ci.0),
            width_delta_ci95_upper: width_ci.map(|ci| ci.1),
            baseline_mean_interval_score: baseline_mean_score,
            candidate_mean_interval_score: candidate_mean_score,
            mean_interval_score_delta: score_delta,
            normalized_mean_interval_score_delta: normalized_score_delta,
            score_delta_ci95_lower: score_ci.map(|ci| ci.0),
            score_delta_ci95_upper: score_ci.map(|ci| ci.1),
            paired_ci_method: (pair_count >= MIN_GROUPS_FOR_NORMAL_CI)
                .then_some("paired-group normal approximation (95%)"),
            baseline_false_safe_count: acc.baseline_false_safe_count,
            candidate_false_safe_count: acc.candidate_false_safe_count,
            baseline_false_safe_rate,
            candidate_false_safe_rate,
            within_declared_envelope_count: acc.within_count,
            borderline_count: acc.borderline_count,
            out_of_domain_count: acc.out_of_domain_count,
            unknown_applicability_count: acc.unknown_count,
            applicability_gate_met,
            coverage_non_degradation_gate_met: enough.then_some(coverage_gate_met).flatten(),
            score_non_degradation_gate_met: enough.then_some(score_gate_met).flatten(),
            width_non_degradation_gate_met: enough.then_some(width_gate_met).flatten(),
            false_safe_gate_met: enough.then_some(false_safe_gate_met).flatten(),
            thresholds_met,
        });
    }

    let mut gap_groups = BTreeMap::<(String, String, u16), Vec<&PairedSliceScore>>::new();
    for score in &slice_scores {
        gap_groups
            .entry((
                score.observable_id.clone(),
                score.unit_id.clone(),
                score.nominal_coverage_bp,
            ))
            .or_default()
            .push(score);
    }

    let mut conditional_coverage_gaps = Vec::new();
    for ((observable_id, unit_id, nominal_coverage_bp), scores) in gap_groups {
        let baseline_extremes = coverage_extremes(&scores, false);
        let candidate_extremes = coverage_extremes(&scores, true);
        let candidate_worst_case = scores
            .iter()
            .filter_map(|score| {
                score
                    .candidate_coverage
                    .map(|coverage| (coverage, score.operating_case_id.as_str()))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, case_id)| case_id.to_owned());
        conditional_coverage_gaps.push(ConditionalCoverageGap {
            observable_id,
            unit_id,
            nominal_coverage_bp,
            baseline_min_coverage: baseline_extremes.map(|v| v.0),
            baseline_max_coverage: baseline_extremes.map(|v| v.1),
            baseline_gap: baseline_extremes.map(|v| v.1 - v.0),
            candidate_min_coverage: candidate_extremes.map(|v| v.0),
            candidate_max_coverage: candidate_extremes.map(|v| v.1),
            candidate_gap: candidate_extremes.map(|v| v.1 - v.0),
            candidate_worst_operating_case_id: candidate_worst_case,
        });
    }

    for gap in &conditional_coverage_gaps {
        if let Some(observed) = gap.candidate_gap {
            if observed > profile.maximum_conditional_coverage_gap {
                issues.push(PairedReplayIssue::ConditionalCoverageGapExceeded {
                    observable_id: gap.observable_id.clone(),
                    unit_id: gap.unit_id.clone(),
                    nominal_coverage_bp: gap.nominal_coverage_bp,
                    maximum: format!("{:.6}", profile.maximum_conditional_coverage_gap),
                    observed: format!("{observed:.6}"),
                });
                regression_found = true;
            }
        }
    }

    issues.sort();
    issues.dedup();

    let invalid_observation = issues.iter().any(|issue| matches!(
        issue,
        PairedReplayIssue::NonFiniteMetric { .. }
            | PairedReplayIssue::InvalidObservationDigest { .. }
            | PairedReplayIssue::EmptyObservationField { .. }
            | PairedReplayIssue::DuplicateObservationId { .. }
            | PairedReplayIssue::DuplicateGroupSlicePair { .. }
            | PairedReplayIssue::EvaluationManifestMismatch { .. }
            | PairedReplayIssue::SplitManifestMismatch { .. }
            | PairedReplayIssue::BaselinePredictionManifestMismatch { .. }
            | PairedReplayIssue::CandidatePredictionManifestMismatch { .. }
            | PairedReplayIssue::BaselineModelConfigurationMismatch { .. }
            | PairedReplayIssue::CandidateModelConfigurationMismatch { .. }
            | PairedReplayIssue::BaselineMethodMismatch { .. }
            | PairedReplayIssue::CandidateMethodMismatch { .. }
            | PairedReplayIssue::UndeclaredEvaluationGroup { .. }
            | PairedReplayIssue::UndeclaredSlice { .. }
            | PairedReplayIssue::PredictionIssuedAfterOutcome { .. }
            | PairedReplayIssue::InvalidNominalCoverage { .. }
            | PairedReplayIssue::InvalidInterval { .. }
            | PairedReplayIssue::NonFiniteIntervalScore { .. }
    ));
    let incomplete = issues.iter().any(|issue| matches!(
        issue,
        PairedReplayIssue::InsufficientPairs { .. } | PairedReplayIssue::MissingEvaluationGroup { .. }
    ));

    let disposition = if !profile_valid {
        PairedReplayDisposition::InvalidProfile
    } else if invalid_observation {
        PairedReplayDisposition::InvalidObservations
    } else if domain_not_established {
        PairedReplayDisposition::DomainNotEstablished
    } else if incomplete {
        PairedReplayDisposition::IncompleteCoverage
    } else if regression_found {
        PairedReplayDisposition::FailDeclaredRegression
    } else if slice_scores.iter().all(|score| score.thresholds_met == Some(true)) {
        PairedReplayDisposition::PassDeclaredNonDegradation
    } else {
        PairedReplayDisposition::IncompleteCoverage
    };

    PairedReplayReport {
        profile_id: profile.profile_id.clone(),
        profile_manifest_sha256: profile.profile_manifest_sha256.clone(),
        disposition,
        input_observation_count: observations.len(),
        scored_pair_count: scored_pairs,
        duplicated_observation_record_count: issues
            .iter()
            .filter(|issue| matches!(issue, PairedReplayIssue::DuplicateObservationId { .. }))
            .map(|issue| match issue {
                PairedReplayIssue::DuplicateObservationId { count, .. } => *count,
                _ => 0,
            })
            .sum(),
        duplicated_group_slice_record_count: issues
            .iter()
            .filter(|issue| matches!(issue, PairedReplayIssue::DuplicateGroupSlicePair { .. }))
            .map(|issue| match issue {
                PairedReplayIssue::DuplicateGroupSlicePair { count, .. } => *count,
                _ => 0,
            })
            .sum(),
        slices: slice_scores,
        conditional_coverage_gaps,
        issues,
    }
}

fn coverage_extremes(scores: &[&PairedSliceScore], candidate: bool) -> Option<(f64, f64)> {
    let mut values = scores.iter().filter_map(|score| {
        if candidate {
            score.candidate_coverage
        } else {
            score.baseline_coverage
        }
    });
    let first = values.next()?;
    let (minimum, maximum, count) = values.fold(
        (first, first, 1_usize),
        |(minimum, maximum, count), value| {
            (minimum.min(value), maximum.max(value), count + 1)
        },
    );
    (count >= 2).then_some((minimum, maximum))
}

#[cfg(test)]
mod tests {
    use super::*;

    const DATASET: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SPLIT: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const BASELINE_PREDICTIONS: &str =
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const CANDIDATE_PREDICTIONS: &str =
        "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const BASELINE_MODEL: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const CANDIDATE_MODEL: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn requirement(case_id: &str) -> PairedSliceRequirement {
        PairedSliceRequirement {
            operating_case_id: case_id.to_string(),
            observable_id: "resistance".to_string(),
            unit_id: "N".to_string(),
            nominal_coverage_bp: 9_000,
            minimum_pairs: 3,
            normalization_scale: 10.0,
            maximum_coverage_drop: 0.05,
            maximum_normalized_score_increase: 0.2,
            maximum_normalized_width_increase: 0.2,
            maximum_false_safe_rate: 0.0,
            maximum_missed_feasible_rate: 1.0,
            engineering_constraint: Some(EngineeringConstraint {
                direction: ConstraintDirection::MaximumAllowed,
                limit: 100.0,
            }),
        }
    }

    fn profile() -> PairedReplayProfile {
        let mut value = PairedReplayProfile {
            profile_id: "paired-profile".to_string(),
            evaluation_manifest_sha256: DATASET.to_string(),
            split_manifest_sha256: SPLIT.to_string(),
            baseline_prediction_manifest_sha256: BASELINE_PREDICTIONS.to_string(),
            candidate_prediction_manifest_sha256: CANDIDATE_PREDICTIONS.to_string(),
            baseline_model_configuration_sha256: BASELINE_MODEL.to_string(),
            candidate_model_configuration_sha256: CANDIDATE_MODEL.to_string(),
            baseline_method_id: "baseline".to_string(),
            baseline_method_revision: "r1".to_string(),
            candidate_method_id: "candidate".to_string(),
            candidate_method_revision: "r1".to_string(),
            baseline_target_label_group_ids: vec!["label-1".to_string(), "label-2".to_string()],
            candidate_target_label_group_ids: vec!["label-1".to_string(), "label-2".to_string()],
            final_confirmation_frozen_before_outcomes: true,
            maximum_conditional_coverage_gap: 0.50,
            evaluation_group_ids: (0..6).map(|i| format!("group-{i}")).collect(),
            required_slices: vec![requirement("deep"), requirement("shallow")],
            profile_manifest_sha256: String::new(),
        };
        value.profile_manifest_sha256 = canonical_paired_replay_profile_sha256(&value);
        value
    }

    fn observation(group: usize, case_id: &str, value: f64, shift_candidate: f64) -> PairedPolicyObservation {
        PairedPolicyObservation {
            observation_id: format!("{case_id}-group-{group}"),
            evaluation_manifest_sha256: DATASET.to_string(),
            split_manifest_sha256: SPLIT.to_string(),
            baseline_prediction_manifest_sha256: BASELINE_PREDICTIONS.to_string(),
            candidate_prediction_manifest_sha256: CANDIDATE_PREDICTIONS.to_string(),
            baseline_model_configuration_sha256: BASELINE_MODEL.to_string(),
            candidate_model_configuration_sha256: CANDIDATE_MODEL.to_string(),
            baseline_method_id: "baseline".to_string(),
            baseline_method_revision: "r1".to_string(),
            candidate_method_id: "candidate".to_string(),
            candidate_method_revision: "r1".to_string(),
            evaluation_group_id: format!("group-{group}"),
            operating_case_id: case_id.to_string(),
            observable_id: "resistance".to_string(),
            unit_id: "N".to_string(),
            nominal_coverage_bp: 9_000,
            observed_value: value,
            baseline_lower: value - 1.0,
            baseline_upper: value + 1.0,
            candidate_lower: value - shift_candidate - 0.75,
            candidate_upper: value - shift_candidate + 0.75,
            applicability: ApplicabilityStatus::WithinDeclaredEnvelope,
            both_predictions_issued_before_outcome: true,
        }
    }

    fn panel(candidate_shift: f64) -> Vec<PairedPolicyObservation> {
        let mut rows = Vec::new();
        for i in 0..6 {
            rows.push(observation(i, "deep", 50.0 + i as f64, candidate_shift));
            rows.push(observation(i, "shallow", 60.0 + i as f64, candidate_shift));
        }
        rows
    }

    fn refresh(profile: &mut PairedReplayProfile) {
        profile.profile_manifest_sha256 = canonical_paired_replay_profile_sha256(profile);
    }

    #[test]
    fn identical_outcomes_and_narrower_candidate_pass_declared_non_degradation() {
        let profile = profile();
        let rows = panel(0.0);
        let report = evaluate_paired_replay(&profile, &rows);
        assert_eq!(report.disposition, PairedReplayDisposition::PassDeclaredNonDegradation);
        assert_eq!(report.scored_pair_count, 12);
        assert!(report.slices.iter().all(|s| s.normalized_mean_interval_score_delta == Some(-0.05)));
    }

    #[test]
    fn same_case_reports_conditional_coverage_gap_instead_of_pooling() {
        let profile = profile();
        let mut rows = panel(0.0);
        for row in rows.iter_mut().filter(|r| r.operating_case_id == "shallow").take(4) {
            row.candidate_lower = row.observed_value + 10.0;
            row.candidate_upper = row.observed_value + 11.0;
        }
        let report = evaluate_paired_replay(&profile, &rows);
        assert!(report.conditional_coverage_gaps.iter().any(|gap| {
            gap.candidate_gap.is_some_and(|value| value > 0.0)
        }));
        assert_eq!(report.disposition, PairedReplayDisposition::FailDeclaredRegression);
    }

    #[test]
    fn conditional_coverage_gap_has_its_own_frozen_acceptance_gate() {
        let mut profile = profile();
        profile.maximum_conditional_coverage_gap = 0.10;
        refresh(&mut profile);
        let mut rows = panel(0.0);
        for row in rows.iter_mut().filter(|r| r.operating_case_id == "shallow").take(4) {
            row.candidate_lower = row.observed_value + 10.0;
            row.candidate_upper = row.observed_value + 11.0;
        }
        let report = evaluate_paired_replay(&profile, &rows);
        assert_eq!(report.disposition, PairedReplayDisposition::FailDeclaredRegression);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            PairedReplayIssue::ConditionalCoverageGapExceeded { .. }
        )));
    }

    #[test]
    fn candidate_that_creates_false_safe_outcomes_fails_closed_on_constraint_gate() {
        let profile = profile();
        let mut rows = panel(0.0);
        for row in rows.iter_mut().filter(|r| r.operating_case_id == "deep").take(2) {
            row.observed_value = 120.0;
            row.baseline_lower = 90.0;
            row.baseline_upper = 95.0;
            row.candidate_lower = 90.0;
            row.candidate_upper = 95.0;
        }
        let report = evaluate_paired_replay(&profile, &rows);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            PairedReplayIssue::FalseSafeRateExceeded { .. }
        )));
    }

    #[test]
    fn duplicate_group_slice_pair_is_never_counted_as_extra_evidence() {
        let profile = profile();
        let mut rows = panel(0.0);
        rows.push(rows[0].clone());
        let report = evaluate_paired_replay(&profile, &rows);
        assert_eq!(report.disposition, PairedReplayDisposition::InvalidObservations);
        assert_eq!(report.scored_pair_count, 10);
    }

    #[test]
    fn duplicate_observation_ids_are_invalid() {
        let profile = profile();
        let mut rows = panel(0.0);
        rows[1].observation_id = rows[0].observation_id.clone();
        let report = evaluate_paired_replay(&profile, &rows);
        assert_eq!(report.disposition, PairedReplayDisposition::InvalidObservations);
    }

    #[test]
    fn mismatched_candidate_prediction_artifact_is_invalid() {
        let profile = profile();
        let mut rows = panel(0.0);
        rows[0].candidate_prediction_manifest_sha256 = BASELINE_PREDICTIONS.to_string();
        let report = evaluate_paired_replay(&profile, &rows);
        assert_eq!(report.disposition, PairedReplayDisposition::InvalidObservations);
    }

    #[test]
    fn post_outcome_candidate_prediction_is_invalid() {
        let profile = profile();
        let mut rows = panel(0.0);
        rows[0].both_predictions_issued_before_outcome = false;
        let report = evaluate_paired_replay(&profile, &rows);
        assert_eq!(report.disposition, PairedReplayDisposition::InvalidObservations);
    }

    #[test]
    fn missing_independent_group_makes_coverage_incomplete() {
        let mut profile = profile();
        let mut rows = panel(0.0);
        profile.evaluation_group_ids.push("not-observed".to_string());
        refresh(&mut profile);
        let report = evaluate_paired_replay(&profile, &rows);
        assert_eq!(report.disposition, PairedReplayDisposition::IncompleteCoverage);
    }

    #[test]
    fn stale_threshold_digest_invalidates_profile() {
        let mut profile = profile();
        profile.required_slices[0].maximum_coverage_drop = 0.2;
        let rows = panel(0.0);
        let report = evaluate_paired_replay(&profile, &rows);
        assert_eq!(report.disposition, PairedReplayDisposition::InvalidProfile);
    }

    #[test]
    fn unknown_or_out_of_domain_application_never_passes() {
        let profile = profile();
        let mut rows = panel(0.0);
        rows[0].applicability = ApplicabilityStatus::Unknown;
        let report = evaluate_paired_replay(&profile, &rows);
        assert_eq!(report.disposition, PairedReplayDisposition::DomainNotEstablished);
    }

    #[test]
    fn target_predictions_must_share_the_same_group_outcome_pair() {
        let profile = profile();
        let rows = panel(0.0);
        let report = evaluate_paired_replay(&profile, &rows);
        assert_eq!(report.scored_pair_count, rows.len());
        assert!(report.slices.iter().all(|slice| slice.pair_count == 6));
    }

    #[test]
    fn undercoverage_and_overcoverage_remain_visible_by_operating_case() {
        let profile = profile();
        let mut rows = panel(0.0);
        for row in rows.iter_mut().filter(|r| r.operating_case_id == "shallow").take(3) {
            row.candidate_lower = row.observed_value + 2.0;
            row.candidate_upper = row.observed_value + 3.0;
        }
        let report = evaluate_paired_replay(&profile, &rows);
        let shallow = report.slices.iter().find(|s| s.operating_case_id == "shallow").unwrap();
        assert!(shallow.candidate_coverage.unwrap() < shallow.baseline_coverage.unwrap());
    }
}
