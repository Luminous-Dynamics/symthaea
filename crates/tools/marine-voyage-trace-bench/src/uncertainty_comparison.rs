//! Pointwise comparison of numerical-solution and reference-measurement intervals.
//!
//! This module intentionally keeps reference measurement bounds, numerical-solution
//! bounds, nominal residuals, and applicability status distinct. It does not infer
//! probabilities, combine uncertainties, estimate model discrepancy, or validate CFD
//! outside the declared observation points.

use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IntervalSemantics {
    ConservativeAbsoluteBound,
    StandardDeviation,
    ExpandedUncertainty,
    ConfidenceInterval,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ApplicabilityStatus {
    WithinDeclaredEnvelope,
    Borderline,
    OutOfDomain,
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundedInterval {
    pub lower: f64,
    pub nominal: f64,
    pub upper: f64,
    pub semantics: IntervalSemantics,
    pub semantics_reviewed: bool,
    pub source_id: String,
    pub source_revision: String,
    pub source_reference_sha256: String,
    pub method_id: String,
    pub method_revision: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IntervalSliceRequirement {
    pub operating_case_id: String,
    pub condition_sha256: String,
    pub observable_id: String,
    pub unit_id: String,
    pub minimum_observations: usize,
    /// Frozen, strictly positive normalizer in the declared observable unit.
    pub normalization_scale: f64,
    pub max_normalized_interval_gap: f64,
    pub require_interval_overlap: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IntervalComparisonProfile {
    pub profile_id: String,
    pub dataset_manifest_sha256: String,
    pub model_configuration_sha256: String,
    /// Canonical SHA-256 of all acceptance-relevant profile content.
    pub profile_manifest_sha256: String,
    pub required_slices: Vec<IntervalSliceRequirement>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IntervalComparisonObservation {
    pub observation_id: String,
    pub dataset_manifest_sha256: String,
    pub model_configuration_sha256: String,
    pub operating_case_id: String,
    pub condition_sha256: String,
    pub observable_id: String,
    pub unit_id: String,
    pub applicability: ApplicabilityStatus,
    /// Represents measurement uncertainty only when its semantics have been reviewed.
    pub reference_measurement: BoundedInterval,
    /// Represents numerical-solution/discretization uncertainty only when reviewed.
    pub simulation_solution: BoundedInterval,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntervalComparisonDisposition {
    CompatibleAtDeclaredPoints,
    FailDeclaredIntervalGate,
    IncompleteCoverage,
    DomainNotEstablished,
    InvalidProfile,
    InvalidObservations,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum IntervalComparisonIssue {
    EmptyProfileField { field: &'static str },
    InvalidProfileDigest { field: &'static str },
    ProfileManifestDigestMismatch,
    NoRequiredSlices,
    DuplicateRequiredSlice {
        operating_case_id: String,
        observable_id: String,
    },
    InvalidRequirementField {
        operating_case_id: String,
        observable_id: String,
        field: &'static str,
    },
    InsufficientCoverage {
        operating_case_id: String,
        observable_id: String,
        minimum: usize,
        observed: usize,
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
    DatasetManifestMismatch { observation_id: String },
    ModelConfigurationMismatch { observation_id: String },
    UndeclaredSlice {
        observation_id: String,
        operating_case_id: String,
        observable_id: String,
    },
    ConditionMismatch { observation_id: String },
    UnitMismatch { observation_id: String },
    InvalidInterval {
        observation_id: String,
        side: &'static str,
        field: &'static str,
    },
    InvalidIntervalSemantics {
        observation_id: String,
        side: &'static str,
    },
    IntervalSemanticsNotReviewed {
        observation_id: String,
        side: &'static str,
    },
    OutOfDomain { observation_id: String },
    ApplicabilityUnknown { observation_id: String },
    ApplicabilityBorderline { observation_id: String },
    NonFiniteComparison { observation_id: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct IntervalComparisonResult {
    pub observation_id: String,
    pub operating_case_id: String,
    pub observable_id: String,
    pub unit_id: String,
    pub applicability: ApplicabilityStatus,
    pub reference_lower: f64,
    pub reference_nominal: f64,
    pub reference_upper: f64,
    pub simulation_lower: f64,
    pub simulation_nominal: f64,
    pub simulation_upper: f64,
    /// Simulation nominal minus reference nominal. This is descriptive residual only.
    pub nominal_residual: f64,
    pub intervals_overlap: bool,
    /// Zero if the intervals overlap, else the positive distance between them.
    pub interval_gap: f64,
    pub normalized_interval_gap: f64,
    pub interval_gate_met: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IntervalSliceScore {
    pub operating_case_id: String,
    pub observable_id: String,
    pub unit_id: String,
    pub observation_count: usize,
    pub required_observation_count: usize,
    pub overlap_count: usize,
    pub nonoverlap_count: usize,
    pub maximum_normalized_interval_gap: Option<f64>,
    pub maximum_absolute_nominal_residual: Option<f64>,
    pub gate_met: Option<bool>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IntervalComparisonAudit {
    pub profile_id: String,
    pub profile_manifest_sha256: String,
    pub disposition: IntervalComparisonDisposition,
    pub input_observation_count: usize,
    pub unique_observation_count: usize,
    pub duplicated_observation_record_count: usize,
    pub results: Vec<IntervalComparisonResult>,
    pub slices: Vec<IntervalSliceScore>,
    pub issues: Vec<IntervalComparisonIssue>,
}

#[derive(Default)]
struct CanonicalEncoder(Vec<u8>);

impl CanonicalEncoder {
    fn new(domain: &str) -> Self {
        let mut encoder = Self::default();
        encoder.text("domain", domain);
        encoder
    }

    fn field(&mut self, key: &str, value: &[u8]) {
        self.0.extend_from_slice(&(key.len() as u64).to_be_bytes());
        self.0.extend_from_slice(key.as_bytes());
        self.0.extend_from_slice(&(value.len() as u64).to_be_bytes());
        self.0.extend_from_slice(value);
    }

    fn text(&mut self, key: &str, value: &str) {
        self.field(key, value.as_bytes());
    }

    fn count(&mut self, key: &str, value: usize) {
        self.field(key, &(value as u64).to_be_bytes());
    }

    fn float(&mut self, key: &str, value: f64) {
        self.field(key, &value.to_bits().to_be_bytes());
    }

    fn boolean(&mut self, key: &str, value: bool) {
        self.field(key, &[u8::from(value)]);
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
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

fn semantics_name(semantics: IntervalSemantics) -> &'static str {
    match semantics {
        IntervalSemantics::ConservativeAbsoluteBound => "conservative-absolute-bound",
        IntervalSemantics::StandardDeviation => "standard-deviation",
        IntervalSemantics::ExpandedUncertainty => "expanded-uncertainty",
        IntervalSemantics::ConfidenceInterval => "confidence-interval",
        IntervalSemantics::Unknown => "unknown",
    }
}

/// Canonical digest of every required case, condition, observable, unit, count,
/// normalization scale, gap threshold and overlap requirement. Equivalent requirement
/// ordering yields the same digest; changing any acceptance-relevant field changes it.
pub fn canonical_interval_profile_sha256(profile: &IntervalComparisonProfile) -> String {
    let mut encoder = CanonicalEncoder::new("marine-interval-comparison-profile-v1");
    encoder.text("profile_id", profile.profile_id.trim());
    encoder.text(
        "dataset_manifest_sha256",
        &profile.dataset_manifest_sha256.to_ascii_lowercase(),
    );
    encoder.text(
        "model_configuration_sha256",
        &profile.model_configuration_sha256.to_ascii_lowercase(),
    );

    let mut requirements = profile.required_slices.clone();
    requirements.sort_by(|a, b| {
        a.operating_case_id
            .trim()
            .cmp(b.operating_case_id.trim())
            .then_with(|| a.condition_sha256.to_ascii_lowercase().cmp(&b.condition_sha256.to_ascii_lowercase()))
            .then_with(|| a.observable_id.trim().cmp(b.observable_id.trim()))
            .then_with(|| a.unit_id.trim().cmp(b.unit_id.trim()))
            .then_with(|| a.minimum_observations.cmp(&b.minimum_observations))
            .then_with(|| a.normalization_scale.total_cmp(&b.normalization_scale))
            .then_with(|| a.max_normalized_interval_gap.total_cmp(&b.max_normalized_interval_gap))
            .then_with(|| a.require_interval_overlap.cmp(&b.require_interval_overlap))
    });
    encoder.count("required_slices_count", requirements.len());
    for requirement in &requirements {
        let mut item = CanonicalEncoder::new("marine-interval-slice-v1");
        item.text("operating_case_id", requirement.operating_case_id.trim());
        item.text("condition_sha256", &requirement.condition_sha256.to_ascii_lowercase());
        item.text("observable_id", requirement.observable_id.trim());
        item.text("unit_id", requirement.unit_id.trim());
        item.count("minimum_observations", requirement.minimum_observations);
        item.float("normalization_scale", requirement.normalization_scale);
        item.float("max_normalized_interval_gap", requirement.max_normalized_interval_gap);
        item.boolean("require_interval_overlap", requirement.require_interval_overlap);
        encoder.field("required_slice", &item.0);
    }
    sha256_hex(&encoder.0)
}

fn validate_profile(
    profile: &IntervalComparisonProfile,
    issues: &mut Vec<IntervalComparisonIssue>,
) -> bool {
    let mut invalid = false;
    for (value, field) in [
        (profile.profile_id.as_str(), "profile_id"),
        (profile.dataset_manifest_sha256.as_str(), "dataset_manifest_sha256"),
        (profile.model_configuration_sha256.as_str(), "model_configuration_sha256"),
        (profile.profile_manifest_sha256.as_str(), "profile_manifest_sha256"),
    ] {
        if !nonempty(value) {
            issues.push(IntervalComparisonIssue::EmptyProfileField { field });
            invalid = true;
        }
    }
    for (value, field) in [
        (profile.dataset_manifest_sha256.as_str(), "dataset_manifest_sha256"),
        (profile.model_configuration_sha256.as_str(), "model_configuration_sha256"),
        (profile.profile_manifest_sha256.as_str(), "profile_manifest_sha256"),
    ] {
        if !valid_sha256(value) {
            issues.push(IntervalComparisonIssue::InvalidProfileDigest { field });
            invalid = true;
        }
    }
    if valid_sha256(&profile.profile_manifest_sha256) {
        let expected = canonical_interval_profile_sha256(profile);
        if !profile.profile_manifest_sha256.eq_ignore_ascii_case(&expected) {
            issues.push(IntervalComparisonIssue::ProfileManifestDigestMismatch);
            invalid = true;
        }
    }
    if profile.required_slices.is_empty() {
        issues.push(IntervalComparisonIssue::NoRequiredSlices);
        invalid = true;
    }

    let mut seen = BTreeSet::new();
    for requirement in &profile.required_slices {
        let key = (
            requirement.operating_case_id.trim().to_owned(),
            requirement.observable_id.trim().to_owned(),
        );
        if !nonempty(&key.0) {
            issues.push(IntervalComparisonIssue::InvalidRequirementField {
                operating_case_id: key.0.clone(),
                observable_id: key.1.clone(),
                field: "operating_case_id",
            });
            invalid = true;
        }
        if !nonempty(&key.1) {
            issues.push(IntervalComparisonIssue::InvalidRequirementField {
                operating_case_id: key.0.clone(),
                observable_id: key.1.clone(),
                field: "observable_id",
            });
            invalid = true;
        }
        if !seen.insert(key.clone()) {
            issues.push(IntervalComparisonIssue::DuplicateRequiredSlice {
                operating_case_id: key.0.clone(),
                observable_id: key.1.clone(),
            });
            invalid = true;
        }
        if !valid_sha256(&requirement.condition_sha256) {
            issues.push(IntervalComparisonIssue::InvalidRequirementField {
                operating_case_id: key.0.clone(),
                observable_id: key.1.clone(),
                field: "condition_sha256",
            });
            invalid = true;
        }
        if !nonempty(&requirement.unit_id) {
            issues.push(IntervalComparisonIssue::InvalidRequirementField {
                operating_case_id: key.0.clone(),
                observable_id: key.1.clone(),
                field: "unit_id",
            });
            invalid = true;
        }
        if requirement.minimum_observations == 0 {
            issues.push(IntervalComparisonIssue::InvalidRequirementField {
                operating_case_id: key.0.clone(),
                observable_id: key.1.clone(),
                field: "minimum_observations",
            });
            invalid = true;
        }
        if !requirement.normalization_scale.is_finite() || requirement.normalization_scale <= 0.0 {
            issues.push(IntervalComparisonIssue::InvalidRequirementField {
                operating_case_id: key.0.clone(),
                observable_id: key.1.clone(),
                field: "normalization_scale",
            });
            invalid = true;
        }
        if !requirement.max_normalized_interval_gap.is_finite()
            || requirement.max_normalized_interval_gap < 0.0
        {
            issues.push(IntervalComparisonIssue::InvalidRequirementField {
                operating_case_id: key.0.clone(),
                observable_id: key.1.clone(),
                field: "max_normalized_interval_gap",
            });
            invalid = true;
        }
    }
    invalid
}

fn validate_interval(
    interval: &BoundedInterval,
    observation_id: &str,
    side: &'static str,
    issues: &mut Vec<IntervalComparisonIssue>,
) -> bool {
    let mut invalid = false;
    for (value, field) in [
        (interval.source_id.as_str(), "source_id"),
        (interval.source_revision.as_str(), "source_revision"),
        (interval.method_id.as_str(), "method_id"),
        (interval.method_revision.as_str(), "method_revision"),
    ] {
        if !nonempty(value) {
            issues.push(IntervalComparisonIssue::InvalidInterval {
                observation_id: observation_id.to_owned(),
                side,
                field,
            });
            invalid = true;
        }
    }
    if !valid_sha256(&interval.source_reference_sha256) {
        issues.push(IntervalComparisonIssue::InvalidInterval {
            observation_id: observation_id.to_owned(),
            side,
            field: "source_reference_sha256",
        });
        invalid = true;
    }
    for (value, field) in [
        (interval.lower, "lower"),
        (interval.nominal, "nominal"),
        (interval.upper, "upper"),
    ] {
        if !value.is_finite() {
            issues.push(IntervalComparisonIssue::InvalidInterval {
                observation_id: observation_id.to_owned(),
                side,
                field,
            });
            invalid = true;
        }
    }
    if interval.lower.is_finite()
        && interval.nominal.is_finite()
        && interval.upper.is_finite()
        && !(interval.lower <= interval.nominal && interval.nominal <= interval.upper)
    {
        issues.push(IntervalComparisonIssue::InvalidInterval {
            observation_id: observation_id.to_owned(),
            side,
            field: "ordering",
        });
        invalid = true;
    }
    if interval.semantics != IntervalSemantics::ConservativeAbsoluteBound {
        issues.push(IntervalComparisonIssue::InvalidIntervalSemantics {
            observation_id: observation_id.to_owned(),
            side,
        });
        invalid = true;
    }
    if !interval.semantics_reviewed {
        issues.push(IntervalComparisonIssue::IntervalSemanticsNotReviewed {
            observation_id: observation_id.to_owned(),
            side,
        });
        invalid = true;
    }
    invalid
}

fn interval_gap(a_lower: f64, a_upper: f64, b_lower: f64, b_upper: f64) -> Option<(bool, f64)> {
    if ![a_lower, a_upper, b_lower, b_upper].iter().all(|value| value.is_finite())
        || a_lower > a_upper
        || b_lower > b_upper
    {
        return None;
    }
    if a_upper < b_lower {
        Some((false, b_lower - a_upper))
    } else if b_upper < a_lower {
        Some((false, a_lower - b_upper))
    } else {
        Some((true, 0.0))
    }
}

/// Audit pointwise interval compatibility. This never combines the two intervals into
/// a synthetic total uncertainty and never promotes overlap to model validation.
pub fn compare_uncertainty_intervals(
    profile: &IntervalComparisonProfile,
    observations: &[IntervalComparisonObservation],
) -> IntervalComparisonAudit {
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
    for (id, count) in &id_counts {
        if *count > 1 {
            issues.push(IntervalComparisonIssue::DuplicateObservationId {
                observation_id: id.clone(),
                count: *count,
            });
        }
    }
    let duplicated_observation_record_count: usize = duplicate_ids.iter().filter_map(|id| id_counts.get(id)).sum();
    let unique_observation_count = id_counts.len() - duplicate_ids.len();

    let mut requirements: BTreeMap<(String, String), &IntervalSliceRequirement> = BTreeMap::new();
    for requirement in &profile.required_slices {
        requirements.entry((
            requirement.operating_case_id.trim().to_owned(),
            requirement.observable_id.trim().to_owned(),
        )).or_insert(requirement);
    }
    let mut counts: BTreeMap<(String, String), usize> = BTreeMap::new();
    let mut overlap_counts: BTreeMap<(String, String), usize> = BTreeMap::new();
    let mut max_gap: BTreeMap<(String, String), f64> = BTreeMap::new();
    let mut max_nominal_residual: BTreeMap<(String, String), f64> = BTreeMap::new();
    for key in requirements.keys() {
        counts.insert(key.clone(), 0);
        overlap_counts.insert(key.clone(), 0);
        max_gap.insert(key.clone(), 0.0);
        max_nominal_residual.insert(key.clone(), 0.0);
    }

    let mut sorted: Vec<&IntervalComparisonObservation> = observations.iter()
        .filter(|observation| !duplicate_ids.contains(observation.observation_id.trim()))
        .collect();
    sorted.sort_by(|a,b| {
        a.operating_case_id.trim().cmp(b.operating_case_id.trim())
            .then_with(|| a.observable_id.trim().cmp(b.observable_id.trim()))
            .then_with(|| a.observation_id.trim().cmp(b.observation_id.trim()))
    });

    let mut invalid_observations = !duplicate_ids.is_empty();
    let mut domain_not_established = false;
    let mut interval_gate_failed = false;
    let mut results = Vec::new();
    for observation in sorted {
        let id = observation.observation_id.trim().to_owned();
        let mut invalid = false;
        for (value, field) in [
            (id.as_str(), "observation_id"),
            (observation.operating_case_id.as_str(), "operating_case_id"),
            (observation.observable_id.as_str(), "observable_id"),
            (observation.unit_id.as_str(), "unit_id"),
        ] {
            if !nonempty(value) {
                issues.push(IntervalComparisonIssue::EmptyObservationField {
                    observation_id: id.clone(),
                    field,
                });
                invalid = true;
            }
        }
        for (value, field) in [
            (observation.dataset_manifest_sha256.as_str(), "dataset_manifest_sha256"),
            (observation.model_configuration_sha256.as_str(), "model_configuration_sha256"),
            (observation.condition_sha256.as_str(), "condition_sha256"),
        ] {
            if !valid_sha256(value) {
                issues.push(IntervalComparisonIssue::InvalidObservationDigest {
                    observation_id: id.clone(),
                    field,
                });
                invalid = true;
            }
        }
        if !observation.dataset_manifest_sha256.eq_ignore_ascii_case(&profile.dataset_manifest_sha256) {
            issues.push(IntervalComparisonIssue::DatasetManifestMismatch { observation_id: id.clone() });
            invalid = true;
        }
        if !observation.model_configuration_sha256.eq_ignore_ascii_case(&profile.model_configuration_sha256) {
            issues.push(IntervalComparisonIssue::ModelConfigurationMismatch { observation_id: id.clone() });
            invalid = true;
        }
        if validate_interval(&observation.reference_measurement, &id, "reference_measurement", &mut issues) {
            invalid = true;
        }
        if validate_interval(&observation.simulation_solution, &id, "simulation_solution", &mut issues) {
            invalid = true;
        }
        if invalid {
            invalid_observations = true;
            continue;
        }

        let key = (
            observation.operating_case_id.trim().to_owned(),
            observation.observable_id.trim().to_owned(),
        );
        let Some(requirement) = requirements.get(&key) else {
            issues.push(IntervalComparisonIssue::UndeclaredSlice {
                observation_id: id,
                operating_case_id: key.0,
                observable_id: key.1,
            });
            invalid_observations = true;
            continue;
        };
        if !observation.condition_sha256.eq_ignore_ascii_case(&requirement.condition_sha256) {
            issues.push(IntervalComparisonIssue::ConditionMismatch { observation_id: id });
            invalid_observations = true;
            continue;
        }
        if observation.unit_id.trim() != requirement.unit_id.trim() {
            issues.push(IntervalComparisonIssue::UnitMismatch { observation_id: id });
            invalid_observations = true;
            continue;
        }

        match observation.applicability {
            ApplicabilityStatus::WithinDeclaredEnvelope => {}
            ApplicabilityStatus::Borderline => {
                issues.push(IntervalComparisonIssue::ApplicabilityBorderline { observation_id: id.clone() });
                domain_not_established = true;
            }
            ApplicabilityStatus::OutOfDomain => {
                issues.push(IntervalComparisonIssue::OutOfDomain { observation_id: id.clone() });
                domain_not_established = true;
            }
            ApplicabilityStatus::Unknown => {
                issues.push(IntervalComparisonIssue::ApplicabilityUnknown { observation_id: id.clone() });
                domain_not_established = true;
            }
        }

        let reference = &observation.reference_measurement;
        let simulation = &observation.simulation_solution;
        let nominal_residual = simulation.nominal - reference.nominal;
        let Some((overlap, gap)) = interval_gap(
            reference.lower, reference.upper, simulation.lower, simulation.upper
        ) else {
            issues.push(IntervalComparisonIssue::NonFiniteComparison { observation_id: id.clone() });
            invalid_observations = true;
            continue;
        };
        let normalized_gap = gap / requirement.normalization_scale;
        if !nominal_residual.is_finite() || !normalized_gap.is_finite() {
            issues.push(IntervalComparisonIssue::NonFiniteComparison { observation_id: id.clone() });
            invalid_observations = true;
            continue;
        }
        let gate_met = (!requirement.require_interval_overlap || overlap)
            && normalized_gap <= requirement.max_normalized_interval_gap;
        if !gate_met {
            interval_gate_failed = true;
        }
        *counts.entry(key.clone()).or_default() += 1;
        if overlap {
            *overlap_counts.entry(key.clone()).or_default() += 1;
        }
        let old_gap = *max_gap.get(&key).unwrap_or(&0.0);
        max_gap.insert(key.clone(), old_gap.max(normalized_gap));
        let old_residual = *max_nominal_residual.get(&key).unwrap_or(&0.0);
        max_nominal_residual.insert(key.clone(), old_residual.max(nominal_residual.abs()));
        results.push(IntervalComparisonResult {
            observation_id: id,
            operating_case_id: key.0,
            observable_id: key.1,
            unit_id: observation.unit_id.trim().to_owned(),
            applicability: observation.applicability,
            reference_lower: reference.lower,
            reference_nominal: reference.nominal,
            reference_upper: reference.upper,
            simulation_lower: simulation.lower,
            simulation_nominal: simulation.nominal,
            simulation_upper: simulation.upper,
            nominal_residual,
            intervals_overlap: overlap,
            interval_gap: gap,
            normalized_interval_gap: normalized_gap,
            interval_gate_met: gate_met,
        });
    }

    let mut incomplete = false;
    let mut slices = Vec::new();
    for (key, requirement) in &requirements {
        let count = *counts.get(key).unwrap_or(&0);
        let overlaps = *overlap_counts.get(key).unwrap_or(&0);
        let observed_max_gap = *max_gap.get(key).unwrap_or(&0.0);
        let observed_max_residual = *max_nominal_residual.get(key).unwrap_or(&0.0);
        if count < requirement.minimum_observations {
            issues.push(IntervalComparisonIssue::InsufficientCoverage {
                operating_case_id: key.0.clone(),
                observable_id: key.1.clone(),
                minimum: requirement.minimum_observations,
                observed: count,
            });
            incomplete = true;
        }
        let gate_met = if count < requirement.minimum_observations {
            None
        } else {
            let meets_gap = observed_max_gap <= requirement.max_normalized_interval_gap;
            let meets_overlap = !requirement.require_interval_overlap || overlaps == count;
            Some(meets_gap && meets_overlap)
        };
        slices.push(IntervalSliceScore {
            operating_case_id: key.0.clone(),
            observable_id: key.1.clone(),
            unit_id: requirement.unit_id.clone(),
            observation_count: count,
            required_observation_count: requirement.minimum_observations,
            overlap_count: overlaps,
            nonoverlap_count: count.saturating_sub(overlaps),
            maximum_normalized_interval_gap: (count > 0).then_some(observed_max_gap),
            maximum_absolute_nominal_residual: (count > 0).then_some(observed_max_residual),
            gate_met,
        });
    }
    results.sort_by(|a,b| {
        a.operating_case_id.cmp(&b.operating_case_id)
            .then_with(|| a.observable_id.cmp(&b.observable_id))
            .then_with(|| a.observation_id.cmp(&b.observation_id))
    });
    slices.sort_by(|a,b| a.operating_case_id.cmp(&b.operating_case_id).then_with(|| a.observable_id.cmp(&b.observable_id)));
    issues.sort();
    issues.dedup();

    let disposition = if invalid_profile {
        IntervalComparisonDisposition::InvalidProfile
    } else if invalid_observations {
        IntervalComparisonDisposition::InvalidObservations
    } else if interval_gate_failed {
        IntervalComparisonDisposition::FailDeclaredIntervalGate
    } else if domain_not_established {
        IntervalComparisonDisposition::DomainNotEstablished
    } else if incomplete {
        IntervalComparisonDisposition::IncompleteCoverage
    } else {
        IntervalComparisonDisposition::CompatibleAtDeclaredPoints
    };

    IntervalComparisonAudit {
        profile_id: profile.profile_id.clone(),
        profile_manifest_sha256: profile.profile_manifest_sha256.clone(),
        disposition,
        input_observation_count: observations.len(),
        unique_observation_count,
        duplicated_observation_record_count,
        results,
        slices,
        issues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DATASET: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const MODEL: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const CONDITION: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const SOURCE: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

    fn interval(lower: f64, nominal: f64, upper: f64, source: &str, method: &str) -> BoundedInterval {
        BoundedInterval {
            lower, nominal, upper,
            semantics: IntervalSemantics::ConservativeAbsoluteBound,
            semantics_reviewed: true,
            source_id: source.to_owned(),
            source_revision: "rev-1".to_owned(),
            source_reference_sha256: SOURCE.to_owned(),
            method_id: method.to_owned(),
            method_revision: "method-rev-1".to_owned(),
        }
    }

    fn requirement() -> IntervalSliceRequirement {
        IntervalSliceRequirement {
            operating_case_id: "depth-ratio-1.2".to_owned(),
            condition_sha256: CONDITION.to_owned(),
            observable_id: "resistance".to_owned(),
            unit_id: "N".to_owned(),
            minimum_observations: 2,
            normalization_scale: 100.0,
            max_normalized_interval_gap: 0.01,
            require_interval_overlap: true,
        }
    }

    fn profile() -> IntervalComparisonProfile {
        let mut profile = IntervalComparisonProfile {
            profile_id: "synthetic-interval-profile-r1".to_owned(),
            dataset_manifest_sha256: DATASET.to_owned(),
            model_configuration_sha256: MODEL.to_owned(),
            profile_manifest_sha256: String::new(),
            required_slices: vec![requirement()],
        };
        profile.profile_manifest_sha256 = canonical_interval_profile_sha256(&profile);
        profile
    }

    fn observation(id: &str, reference: (f64, f64, f64), simulation: (f64, f64, f64)) -> IntervalComparisonObservation {
        IntervalComparisonObservation {
            observation_id: id.to_owned(),
            dataset_manifest_sha256: DATASET.to_owned(),
            model_configuration_sha256: MODEL.to_owned(),
            operating_case_id: "depth-ratio-1.2".to_owned(),
            condition_sha256: CONDITION.to_owned(),
            observable_id: "resistance".to_owned(),
            unit_id: "N".to_owned(),
            applicability: ApplicabilityStatus::WithinDeclaredEnvelope,
            reference_measurement: interval(reference.0, reference.1, reference.2, "reference-data", "measurement-bound"),
            simulation_solution: interval(simulation.0, simulation.1, simulation.2, "solver-output", "mesh-study"),
        }
    }

    fn has_issue(audit: &IntervalComparisonAudit, expected: &IntervalComparisonIssue) -> bool {
        audit.issues.contains(expected)
    }

    #[test]
    fn overlapping_intervals_have_zero_gap_but_do_not_assert_validation() {
        let profile = profile();
        let observations = vec![
            observation("sample-1", (90.0, 100.0, 110.0), (105.0, 112.0, 120.0)),
            observation("sample-2", (190.0, 200.0, 210.0), (205.0, 212.0, 220.0)),
        ];
        let audit = compare_uncertainty_intervals(&profile, &observations);
        assert_eq!(audit.disposition, IntervalComparisonDisposition::CompatibleAtDeclaredPoints);
        assert!(audit.results.iter().all(|r| r.intervals_overlap && r.interval_gap == 0.0));
        assert!(audit.results.iter().all(|r| (r.nominal_residual - 12.0).abs() < 1e-12));
    }

    #[test]
    fn disjoint_intervals_report_positive_gap_and_fail_gate() {
        let profile = profile();
        let observations = vec![
            observation("sample-1", (90.0, 100.0, 110.0), (130.0, 140.0, 150.0)),
            observation("sample-2", (190.0, 200.0, 210.0), (230.0, 240.0, 250.0)),
        ];
        let audit = compare_uncertainty_intervals(&profile, &observations);
        assert_eq!(audit.disposition, IntervalComparisonDisposition::FailDeclaredIntervalGate);
        assert!(audit.results.iter().all(|r| !r.intervals_overlap && (r.interval_gap - 20.0).abs() < 1e-12));
    }

    #[test]
    fn out_of_domain_never_passes_even_when_intervals_overlap() {
        let profile = profile();
        let mut obs = vec![
            observation("sample-1", (90.0, 100.0, 110.0), (100.0, 105.0, 110.0)),
            observation("sample-2", (190.0, 200.0, 210.0), (200.0, 205.0, 210.0)),
        ];
        obs[0].applicability = ApplicabilityStatus::OutOfDomain;
        let audit = compare_uncertainty_intervals(&profile, &obs);
        assert_eq!(audit.disposition, IntervalComparisonDisposition::DomainNotEstablished);
        assert!(has_issue(&audit, &IntervalComparisonIssue::OutOfDomain { observation_id: "sample-1".to_owned() }));
    }

    #[test]
    fn standard_deviation_is_not_silently_reinterpreted_as_a_bound() {
        let profile = profile();
        let mut obs = vec![
            observation("sample-1", (90.0, 100.0, 110.0), (100.0, 105.0, 110.0)),
            observation("sample-2", (190.0, 200.0, 210.0), (200.0, 205.0, 210.0)),
        ];
        obs[0].reference_measurement.semantics = IntervalSemantics::StandardDeviation;
        let audit = compare_uncertainty_intervals(&profile, &obs);
        assert_eq!(audit.disposition, IntervalComparisonDisposition::InvalidObservations);
        assert!(has_issue(&audit, &IntervalComparisonIssue::InvalidIntervalSemantics {
            observation_id: "sample-1".to_owned(),
            side: "reference_measurement",
        }));
    }

    #[test]
    fn reversed_and_nonfinite_intervals_are_invalid() {
        let profile = profile();
        let mut obs = vec![
            observation("sample-1", (110.0, 100.0, 120.0), (100.0, 105.0, 110.0)),
            observation("sample-2", (190.0, 200.0, f64::INFINITY), (200.0, 205.0, 210.0)),
        ];
        let audit = compare_uncertainty_intervals(&profile, &obs);
        assert_eq!(audit.disposition, IntervalComparisonDisposition::InvalidObservations);
        assert!(audit.issues.iter().any(|issue| matches!(issue, IntervalComparisonIssue::InvalidInterval { .. })));
    }

    #[test]
    fn stale_profile_digest_after_threshold_change_is_rejected() {
        let mut profile = profile();
        profile.required_slices[0].max_normalized_interval_gap = 0.001;
        let observations = vec![
            observation("sample-1", (90.0, 100.0, 110.0), (100.0, 105.0, 110.0)),
            observation("sample-2", (190.0, 200.0, 210.0), (200.0, 205.0, 210.0)),
        ];
        let audit = compare_uncertainty_intervals(&profile, &observations);
        assert_eq!(audit.disposition, IntervalComparisonDisposition::InvalidProfile);
        assert!(has_issue(&audit, &IntervalComparisonIssue::ProfileManifestDigestMismatch));
    }

    #[test]
    fn duplicate_observation_ids_are_excluded_in_full() {
        let profile = profile();
        let mut obs = vec![
            observation("sample-1", (90.0, 100.0, 110.0), (100.0, 105.0, 110.0)),
            observation("sample-2", (190.0, 200.0, 210.0), (200.0, 205.0, 210.0)),
        ];
        let mut alias = obs[0].clone();
        alias.observation_id = " sample-1 ".to_owned();
        obs.push(alias);
        let audit = compare_uncertainty_intervals(&profile, &obs);
        assert_eq!(audit.disposition, IntervalComparisonDisposition::InvalidObservations);
        assert!(has_issue(&audit, &IntervalComparisonIssue::DuplicateObservationId {
            observation_id: "sample-1".to_owned(),
            count: 2,
        }));
        assert_eq!(audit.slices[0].observation_count, 1);
    }

    #[test]
    fn condition_and_unit_mismatches_are_rejected() {
        let profile = profile();
        let mut obs = vec![
            observation("sample-1", (90.0, 100.0, 110.0), (100.0, 105.0, 110.0)),
            observation("sample-2", (190.0, 200.0, 210.0), (200.0, 205.0, 210.0)),
        ];
        obs[0].condition_sha256 = "e".repeat(64);
        obs[1].unit_id = "kN".to_owned();
        let audit = compare_uncertainty_intervals(&profile, &obs);
        assert_eq!(audit.disposition, IntervalComparisonDisposition::InvalidObservations);
        assert!(has_issue(&audit, &IntervalComparisonIssue::ConditionMismatch { observation_id: "sample-1".to_owned() }));
        assert!(has_issue(&audit, &IntervalComparisonIssue::UnitMismatch { observation_id: "sample-2".to_owned() }));
    }

    #[test]
    fn deterministic_under_observation_and_requirement_permutation() {
        let mut profile = profile();
        let mut obs = vec![
            observation("sample-1", (90.0, 100.0, 110.0), (100.0, 105.0, 110.0)),
            observation("sample-2", (190.0, 200.0, 210.0), (200.0, 205.0, 210.0)),
        ];
        let first = compare_uncertainty_intervals(&profile, &obs);
        profile.required_slices.reverse();
        obs.reverse();
        let second = compare_uncertainty_intervals(&profile, &obs);
        assert_eq!(first, second);
    }

    #[test]
    fn canonical_interval_gap_known_answer() {
        assert_eq!(interval_gap(0.0, 10.0, 15.0, 20.0), Some((false, 5.0)));
        assert_eq!(interval_gap(0.0, 10.0, 10.0, 20.0), Some((true, 0.0)));
        assert_eq!(interval_gap(15.0, 20.0, 0.0, 10.0), Some((false, 5.0)));
        assert_eq!(interval_gap(3.0, 2.0, 0.0, 1.0), None);
    }

    #[test]
    fn borderline_and_unknown_applicability_are_not_a_pass() {
        let profile = profile();
        for applicability in [ApplicabilityStatus::Borderline, ApplicabilityStatus::Unknown] {
            let mut obs = vec![
                observation("sample-1", (90.0, 100.0, 110.0), (100.0, 105.0, 110.0)),
                observation("sample-2", (190.0, 200.0, 210.0), (200.0, 205.0, 210.0)),
            ];
            obs[0].applicability = applicability;
            let audit = compare_uncertainty_intervals(&profile, &obs);
            assert_eq!(audit.disposition, IntervalComparisonDisposition::DomainNotEstablished);
        }
    }

    #[test]
    fn missing_interval_semantics_review_blocks_comparison() {
        let profile = profile();
        let mut obs = vec![
            observation("sample-1", (90.0, 100.0, 110.0), (100.0, 105.0, 110.0)),
            observation("sample-2", (190.0, 200.0, 210.0), (200.0, 205.0, 210.0)),
        ];
        obs[1].simulation_solution.semantics_reviewed = false;
        let audit = compare_uncertainty_intervals(&profile, &obs);
        assert_eq!(audit.disposition, IntervalComparisonDisposition::InvalidObservations);
        assert!(has_issue(&audit, &IntervalComparisonIssue::IntervalSemanticsNotReviewed {
            observation_id: "sample-2".to_owned(),
            side: "simulation_solution",
        }));
    }
}
