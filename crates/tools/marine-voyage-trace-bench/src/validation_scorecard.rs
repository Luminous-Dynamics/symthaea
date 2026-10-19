//! Per-operating-condition, per-observable validation scorecard.
//!
//! The scorecard reports prediction residuals against a frozen profile and preserves
//! each required slice. It intentionally has no aggregate score that can hide a failed
//! operating condition. It does not estimate model discrepancy, measurement uncertainty,
//! confidence intervals, or physical-vessel qualification.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationSliceRequirement {
    pub operating_case_id: String,
    pub observable_id: String,
    pub minimum_observations: usize,
    /// Frozen positive scale with the same unit as the observable; never inferred from
    /// the candidate's observed residuals.
    pub normalization_scale: f64,
    /// Maximum RMSE / scale allowed for this exact slice.
    pub max_normalized_rmse: f64,
    /// Maximum absolute pointwise error / scale allowed for this exact slice.
    pub max_normalized_absolute_error: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationProfile {
    pub profile_id: String,
    pub dataset_manifest_sha256: String,
    pub model_configuration_sha256: String,
    pub required_slices: Vec<ValidationSliceRequirement>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationObservation {
    pub observation_id: String,
    pub dataset_manifest_sha256: String,
    pub model_configuration_sha256: String,
    pub operating_case_id: String,
    pub observable_id: String,
    /// Reference value in the profile-declared canonical unit.
    pub reference_value: f64,
    /// Frozen model's prediction in the same canonical unit.
    pub predicted_value: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationDisposition {
    PassDeclaredThresholds,
    FailDeclaredThresholds,
    IncompleteCoverage,
    InvalidProfile,
    InvalidObservations,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValidationIssue {
    EmptyProfileField { field: &'static str },
    InvalidProfileDigest { field: &'static str },
    NoRequiredSlices,
    DuplicateRequiredSlice {
        operating_case_id: String,
        observable_id: String,
    },
    InvalidSliceField {
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
    DuplicateObservationId {
        observation_id: String,
        count: usize,
    },
    DatasetManifestMismatch { observation_id: String },
    ModelConfigurationMismatch { observation_id: String },
    UndeclaredSlice {
        observation_id: String,
        operating_case_id: String,
        observable_id: String,
    },
    NonFiniteValue {
        observation_id: String,
        field: &'static str,
    },
    NonFiniteResidual { observation_id: String },
    NonFiniteMetric {
        operating_case_id: String,
        observable_id: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct SliceScore {
    pub operating_case_id: String,
    pub observable_id: String,
    pub observation_count: usize,
    pub required_observation_count: usize,
    pub mean_signed_error: Option<f64>,
    pub mean_absolute_error: Option<f64>,
    pub root_mean_squared_error: Option<f64>,
    pub maximum_absolute_error: Option<f64>,
    pub normalized_rmse: Option<f64>,
    pub maximum_normalized_absolute_error: Option<f64>,
    /// None means coverage was insufficient; Some(false) is a real threshold failure.
    pub thresholds_met: Option<bool>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationScorecard {
    pub profile_id: String,
    pub disposition: ValidationDisposition,
    pub input_observation_count: usize,
    pub unique_observation_count: usize,
    pub duplicated_observation_record_count: usize,
    /// Stable order: operating case, then observable.
    pub slices: Vec<SliceScore>,
    /// Sorted and deduplicated to make audit output order-independent.
    pub issues: Vec<ValidationIssue>,
}

#[derive(Debug, Clone, Copy, Default)]
struct ResidualAccumulator {
    count: usize,
    signed_sum: f64,
    signed_compensation: f64,
    absolute_sum: f64,
    absolute_compensation: f64,
    normalized_squared_sum: f64,
    normalized_squared_compensation: f64,
    max_absolute_error: f64,
    max_normalized_absolute_error: f64,
}

impl ResidualAccumulator {
    fn kahan_add(sum: &mut f64, compensation: &mut f64, value: f64) {
        let adjusted = value - *compensation;
        let next = *sum + adjusted;
        *compensation = (next - *sum) - adjusted;
        *sum = next;
    }

    fn add(&mut self, residual: f64, scale: f64) -> bool {
        let absolute = residual.abs();
        let normalized = absolute / scale;
        if !absolute.is_finite() || !normalized.is_finite() {
            return false;
        }
        let normalized_squared = normalized * normalized;
        if !normalized_squared.is_finite() {
            return false;
        }
        Self::kahan_add(
            &mut self.signed_sum,
            &mut self.signed_compensation,
            residual,
        );
        Self::kahan_add(
            &mut self.absolute_sum,
            &mut self.absolute_compensation,
            absolute,
        );
        Self::kahan_add(
            &mut self.normalized_squared_sum,
            &mut self.normalized_squared_compensation,
            normalized_squared,
        );
        if !self.signed_sum.is_finite()
            || !self.absolute_sum.is_finite()
            || !self.normalized_squared_sum.is_finite()
        {
            return false;
        }
        self.max_absolute_error = self.max_absolute_error.max(absolute);
        self.max_normalized_absolute_error = self.max_normalized_absolute_error.max(normalized);
        self.count += 1;
        true
    }

    fn metrics(
        &self,
        requirement: &ValidationSliceRequirement,
    ) -> Option<(f64, f64, f64, f64, f64, f64)> {
        if self.count == 0 {
            return None;
        }
        let n = self.count as f64;
        let mean_signed_error = self.signed_sum / n;
        let mean_absolute_error = self.absolute_sum / n;
        let normalized_rmse = (self.normalized_squared_sum / n).sqrt();
        let root_mean_squared_error = normalized_rmse * requirement.normalization_scale;
        let values = (
            mean_signed_error,
            mean_absolute_error,
            root_mean_squared_error,
            self.max_absolute_error,
            normalized_rmse,
            self.max_normalized_absolute_error,
        );
        [
            values.0,
            values.1,
            values.2,
            values.3,
            values.4,
            values.5,
        ]
        .iter()
        .all(|value| value.is_finite())
        .then_some(values)
    }
}

fn is_nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.as_bytes().iter().all(u8::is_ascii_hexdigit)
}

fn validate_profile(profile: &ValidationProfile, issues: &mut Vec<ValidationIssue>) -> bool {
    let mut invalid = false;
    for (value, field) in [
        (profile.profile_id.as_str(), "profile_id"),
        (
            profile.dataset_manifest_sha256.as_str(),
            "dataset_manifest_sha256",
        ),
        (
            profile.model_configuration_sha256.as_str(),
            "model_configuration_sha256",
        ),
    ] {
        if !is_nonempty(value) {
            issues.push(ValidationIssue::EmptyProfileField { field });
            invalid = true;
        }
    }
    for (value, field) in [
        (
            profile.dataset_manifest_sha256.as_str(),
            "dataset_manifest_sha256",
        ),
        (
            profile.model_configuration_sha256.as_str(),
            "model_configuration_sha256",
        ),
    ] {
        if !valid_sha256_hex(value) {
            issues.push(ValidationIssue::InvalidProfileDigest { field });
            invalid = true;
        }
    }
    if profile.required_slices.is_empty() {
        issues.push(ValidationIssue::NoRequiredSlices);
        invalid = true;
    }

    let mut seen = BTreeSet::new();
    for requirement in &profile.required_slices {
        let key = (
            requirement.operating_case_id.trim().to_owned(),
            requirement.observable_id.trim().to_owned(),
        );
        if !is_nonempty(&key.0) {
            issues.push(ValidationIssue::InvalidSliceField {
                operating_case_id: requirement.operating_case_id.clone(),
                observable_id: requirement.observable_id.clone(),
                field: "operating_case_id",
            });
            invalid = true;
        }
        if !is_nonempty(&key.1) {
            issues.push(ValidationIssue::InvalidSliceField {
                operating_case_id: requirement.operating_case_id.clone(),
                observable_id: requirement.observable_id.clone(),
                field: "observable_id",
            });
            invalid = true;
        }
        if !seen.insert(key.clone()) {
            issues.push(ValidationIssue::DuplicateRequiredSlice {
                operating_case_id: key.0.clone(),
                observable_id: key.1.clone(),
            });
            invalid = true;
        }
        if requirement.minimum_observations == 0 {
            issues.push(ValidationIssue::InvalidSliceField {
                operating_case_id: key.0.clone(),
                observable_id: key.1.clone(),
                field: "minimum_observations",
            });
            invalid = true;
        }
        if !requirement.normalization_scale.is_finite()
            || requirement.normalization_scale <= 0.0
        {
            issues.push(ValidationIssue::InvalidSliceField {
                operating_case_id: key.0.clone(),
                observable_id: key.1.clone(),
                field: "normalization_scale",
            });
            invalid = true;
        }
        for (value, field) in [
            (requirement.max_normalized_rmse, "max_normalized_rmse"),
            (
                requirement.max_normalized_absolute_error,
                "max_normalized_absolute_error",
            ),
        ] {
            if !value.is_finite() || value < 0.0 {
                issues.push(ValidationIssue::InvalidSliceField {
                    operating_case_id: key.0.clone(),
                    observable_id: key.1.clone(),
                    field,
                });
                invalid = true;
            }
        }
    }
    invalid
}

/// Score reference predictions against a frozen operating-case × observable matrix.
///
/// Invalid/duplicated observations cannot be used to obtain a pass. Valid observations
/// are sorted by stable identity before accumulation, making floating-point summation
/// deterministic under input permutation. Every required slice gets its own thresholds;
/// no aggregate metric can hide a failed or missing operating condition.
pub fn score_validation(
    profile: &ValidationProfile,
    observations: &[ValidationObservation],
) -> ValidationScorecard {
    let mut issues = Vec::new();
    let invalid_profile = validate_profile(profile, &mut issues);

    let mut id_counts: BTreeMap<String, usize> = BTreeMap::new();
    for observation in observations {
        *id_counts
            .entry(observation.observation_id.trim().to_owned())
            .or_default() += 1;
    }
    let duplicate_ids: BTreeSet<String> = id_counts
        .iter()
        .filter(|(_, count)| **count > 1)
        .map(|(id, _)| id.clone())
        .collect();
    for (observation_id, count) in &id_counts {
        if *count > 1 {
            issues.push(ValidationIssue::DuplicateObservationId {
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

    let mut requirements: BTreeMap<(String, String), &ValidationSliceRequirement> =
        BTreeMap::new();
    for requirement in &profile.required_slices {
        requirements
            .entry((
                requirement.operating_case_id.trim().to_owned(),
                requirement.observable_id.trim().to_owned(),
            ))
            .or_insert(requirement);
    }
    let mut accumulators: BTreeMap<(String, String), ResidualAccumulator> = BTreeMap::new();
    for key in requirements.keys() {
        accumulators.insert(key.clone(), ResidualAccumulator::default());
    }

    let mut sorted_observations: Vec<&ValidationObservation> = observations
        .iter()
        .filter(|observation| !duplicate_ids.contains(observation.observation_id.trim()))
        .collect();
    sorted_observations.sort_by(|a, b| {
        a.operating_case_id
            .trim()
            .cmp(b.operating_case_id.trim())
            .then_with(|| a.observable_id.trim().cmp(b.observable_id.trim()))
            .then_with(|| a.observation_id.cmp(&b.observation_id))
    });

    let mut invalid_observations = !duplicate_ids.is_empty();
    let mut invalid_metric_slices = BTreeSet::new();
    for observation in sorted_observations {
        let mut valid = true;
        for (value, field) in [
            (observation.observation_id.as_str(), "observation_id"),
            (observation.operating_case_id.as_str(), "operating_case_id"),
            (observation.observable_id.as_str(), "observable_id"),
        ] {
            if !is_nonempty(value) {
                issues.push(ValidationIssue::EmptyObservationField {
                    observation_id: observation.observation_id.clone(),
                    field,
                });
                valid = false;
            }
        }
        for (value, field) in [
            (
                observation.dataset_manifest_sha256.as_str(),
                "dataset_manifest_sha256",
            ),
            (
                observation.model_configuration_sha256.as_str(),
                "model_configuration_sha256",
            ),
        ] {
            if !valid_sha256_hex(value) {
                issues.push(ValidationIssue::InvalidObservationDigest {
                    observation_id: observation.observation_id.clone(),
                    field,
                });
                valid = false;
            }
        }
        if !observation.dataset_manifest_sha256.eq_ignore_ascii_case(&profile.dataset_manifest_sha256) {
            issues.push(ValidationIssue::DatasetManifestMismatch {
                observation_id: observation.observation_id.clone(),
            });
            valid = false;
        }
        if !observation.model_configuration_sha256.eq_ignore_ascii_case(&profile.model_configuration_sha256) {
            issues.push(ValidationIssue::ModelConfigurationMismatch {
                observation_id: observation.observation_id.clone(),
            });
            valid = false;
        }
        for (value, field) in [
            (observation.reference_value, "reference_value"),
            (observation.predicted_value, "predicted_value"),
        ] {
            if !value.is_finite() {
                issues.push(ValidationIssue::NonFiniteValue {
                    observation_id: observation.observation_id.clone(),
                    field,
                });
                valid = false;
            }
        }

        let key = (
            observation.operating_case_id.trim().to_owned(),
            observation.observable_id.trim().to_owned(),
        );
        if !requirements.contains_key(&key) {
            issues.push(ValidationIssue::UndeclaredSlice {
                observation_id: observation.observation_id.clone(),
                operating_case_id: key.0,
                observable_id: key.1,
            });
            invalid_observations = true;
            continue;
        }
        if !valid {
            invalid_observations = true;
            continue;
        }

        let residual = observation.predicted_value - observation.reference_value;
        if !residual.is_finite() {
            issues.push(ValidationIssue::NonFiniteResidual {
                observation_id: observation.observation_id.clone(),
            });
            invalid_observations = true;
            continue;
        }
        let requirement = requirements[&key];
        let accumulator = accumulators.get_mut(&key).expect("required slice initialized");
        if !accumulator.add(residual, requirement.normalization_scale) {
            issues.push(ValidationIssue::NonFiniteMetric {
                operating_case_id: key.0.clone(),
                observable_id: key.1.clone(),
            });
            invalid_observations = true;
            invalid_metric_slices.insert(key);
        }
    }

    let mut slices = Vec::new();
    let mut insufficient_coverage = false;
    let mut threshold_failure = false;
    for (key, requirement) in &requirements {
        let accumulator = accumulators.get(key).copied().unwrap_or_default();
        if accumulator.count < requirement.minimum_observations {
            issues.push(ValidationIssue::InsufficientCoverage {
                operating_case_id: key.0.clone(),
                observable_id: key.1.clone(),
                minimum: requirement.minimum_observations,
                observed: accumulator.count,
            });
            insufficient_coverage = true;
        }
        let metrics = if invalid_metric_slices.contains(key) {
            None
        } else {
            accumulator.metrics(requirement)
        };
        if accumulator.count > 0 && metrics.is_none() {
            issues.push(ValidationIssue::NonFiniteMetric {
                operating_case_id: key.0.clone(),
                observable_id: key.1.clone(),
            });
            invalid_observations = true;
        }
        let (mean_signed_error, mean_absolute_error, rmse, max_abs, normalized_rmse, max_normalized_abs) =
            match metrics {
                Some(values) => (
                    Some(values.0),
                    Some(values.1),
                    Some(values.2),
                    Some(values.3),
                    Some(values.4),
                    Some(values.5),
                ),
                None => (None, None, None, None, None, None),
            };
        let thresholds_met = if accumulator.count < requirement.minimum_observations
            || metrics.is_none()
        {
            None
        } else {
            let meets = normalized_rmse
                .is_some_and(|value| value <= requirement.max_normalized_rmse)
                && max_normalized_abs.is_some_and(|value| {
                    value <= requirement.max_normalized_absolute_error
                });
            if !meets {
                threshold_failure = true;
            }
            Some(meets)
        };
        slices.push(SliceScore {
            operating_case_id: key.0.clone(),
            observable_id: key.1.clone(),
            observation_count: accumulator.count,
            required_observation_count: requirement.minimum_observations,
            mean_signed_error,
            mean_absolute_error,
            root_mean_squared_error: rmse,
            maximum_absolute_error: max_abs,
            normalized_rmse,
            maximum_normalized_absolute_error: max_normalized_abs,
            thresholds_met,
        });
    }
    slices.sort_by(|a, b| {
        a.operating_case_id
            .cmp(&b.operating_case_id)
            .then_with(|| a.observable_id.cmp(&b.observable_id))
    });
    issues.sort();
    issues.dedup();

    let disposition = if invalid_profile {
        ValidationDisposition::InvalidProfile
    } else if invalid_observations {
        ValidationDisposition::InvalidObservations
    } else if insufficient_coverage {
        ValidationDisposition::IncompleteCoverage
    } else if threshold_failure {
        ValidationDisposition::FailDeclaredThresholds
    } else {
        ValidationDisposition::PassDeclaredThresholds
    };

    ValidationScorecard {
        profile_id: profile.profile_id.clone(),
        disposition,
        input_observation_count: observations.len(),
        unique_observation_count,
        duplicated_observation_record_count,
        slices,
        issues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DATASET_DIGEST: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const MODEL_DIGEST: &str =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn slice(case: &str, observable: &str, min: usize, scale: f64, rmse: f64, max_abs: f64) -> ValidationSliceRequirement {
        ValidationSliceRequirement {
            operating_case_id: case.to_owned(),
            observable_id: observable.to_owned(),
            minimum_observations: min,
            normalization_scale: scale,
            max_normalized_rmse: rmse,
            max_normalized_absolute_error: max_abs,
        }
    }

    fn profile() -> ValidationProfile {
        ValidationProfile {
            profile_id: "dtc-shallow-water-r1".to_owned(),
            dataset_manifest_sha256: DATASET_DIGEST.to_owned(),
            model_configuration_sha256: MODEL_DIGEST.to_owned(),
            required_slices: vec![
                slice("depth-ratio-4.0", "resistance", 2, 100.0, 0.05, 0.10),
                slice("depth-ratio-1.2", "resistance", 2, 100.0, 0.05, 0.10),
            ],
        }
    }

    fn observation(id: &str, case: &str, reference: f64, prediction: f64) -> ValidationObservation {
        ValidationObservation {
            observation_id: id.to_owned(),
            dataset_manifest_sha256: DATASET_DIGEST.to_owned(),
            model_configuration_sha256: MODEL_DIGEST.to_owned(),
            operating_case_id: case.to_owned(),
            observable_id: "resistance".to_owned(),
            reference_value: reference,
            predicted_value: prediction,
        }
    }

    fn valid_observations() -> Vec<ValidationObservation> {
        vec![
            observation("deep-1", "depth-ratio-4.0", 100.0, 101.0),
            observation("deep-2", "depth-ratio-4.0", 200.0, 202.0),
            observation("shallow-1", "depth-ratio-1.2", 100.0, 101.0),
            observation("shallow-2", "depth-ratio-1.2", 200.0, 202.0),
        ]
    }

    fn has_issue(scorecard: &ValidationScorecard, target: &ValidationIssue) -> bool {
        scorecard.issues.contains(target)
    }

    #[test]
    fn known_answer_metrics_are_correct_for_each_slice() {
        let scorecard = score_validation(&profile(), &valid_observations());
        assert_eq!(scorecard.disposition, ValidationDisposition::PassDeclaredThresholds);
        assert_eq!(scorecard.slices.len(), 2);
        let deep = scorecard.slices.iter().find(|s| s.operating_case_id == "depth-ratio-4.0").unwrap();
        assert_eq!(deep.observation_count, 2);
        assert!((deep.mean_signed_error.unwrap() - 1.5).abs() < 1e-12);
        assert!((deep.mean_absolute_error.unwrap() - 1.5).abs() < 1e-12);
        assert!((deep.root_mean_squared_error.unwrap() - (2.5_f64).sqrt()).abs() < 1e-12);
        assert!((deep.maximum_absolute_error.unwrap() - 2.0).abs() < 1e-12);
        assert!((deep.normalized_rmse.unwrap() - (2.5_f64).sqrt() / 100.0).abs() < 1e-12);
    }

    #[test]
    fn a_bad_shallow_water_slice_cannot_be_hidden_by_many_good_deep_points() {
        let mut observations = valid_observations();
        observations.extend((0..200).map(|i| observation(
            &format!("deep-extra-{i:03}"),
            "depth-ratio-4.0",
            100.0,
            100.0,
        )));
        observations.push(observation("shallow-outlier", "depth-ratio-1.2", 100.0, 140.0));
        let scorecard = score_validation(&profile(), &observations);
        assert_eq!(scorecard.disposition, ValidationDisposition::FailDeclaredThresholds);
        let shallow = scorecard.slices.iter().find(|s| s.operating_case_id == "depth-ratio-1.2").unwrap();
        assert_eq!(shallow.thresholds_met, Some(false));
        assert!(shallow.maximum_normalized_absolute_error.unwrap() > 0.10);
    }

    #[test]
    fn missing_required_operating_case_is_incomplete_not_silently_dropped() {
        let observations = vec![
            observation("deep-1", "depth-ratio-4.0", 100.0, 101.0),
            observation("deep-2", "depth-ratio-4.0", 200.0, 202.0),
        ];
        let scorecard = score_validation(&profile(), &observations);
        assert_eq!(scorecard.disposition, ValidationDisposition::IncompleteCoverage);
        assert!(has_issue(&scorecard, &ValidationIssue::InsufficientCoverage {
            operating_case_id: "depth-ratio-1.2".to_owned(),
            observable_id: "resistance".to_owned(),
            minimum: 2,
            observed: 0,
        }));
    }

    #[test]
    fn dataset_or_model_digest_mismatch_invalidates_observation() {
        let mut observations = valid_observations();
        observations[0].dataset_manifest_sha256 = "c".repeat(64);
        observations[1].model_configuration_sha256 = "d".repeat(64);
        let scorecard = score_validation(&profile(), &observations);
        assert_eq!(scorecard.disposition, ValidationDisposition::InvalidObservations);
        assert!(has_issue(&scorecard, &ValidationIssue::DatasetManifestMismatch { observation_id: "deep-1".to_owned() }));
        assert!(has_issue(&scorecard, &ValidationIssue::ModelConfigurationMismatch { observation_id: "deep-2".to_owned() }));
    }

    #[test]
    fn duplicate_observation_ids_are_excluded_in_full() {
        let mut observations = valid_observations();
        let mut whitespace_alias = observations[0].clone();
        whitespace_alias.observation_id = " deep-1 ".to_owned();
        observations.push(whitespace_alias);
        let scorecard = score_validation(&profile(), &observations);
        assert_eq!(scorecard.disposition, ValidationDisposition::InvalidObservations);
        assert!(has_issue(&scorecard, &ValidationIssue::DuplicateObservationId {
            observation_id: "deep-1".to_owned(),
            count: 2,
        }));
        assert_eq!(scorecard.slices.iter().find(|s| s.operating_case_id == "depth-ratio-4.0").unwrap().observation_count, 1);
    }

    #[test]
    fn undeclared_slice_is_invalid() {
        let mut observations = valid_observations();
        observations.push(observation("unknown-case", "depth-ratio-2.0", 100.0, 100.0));
        let scorecard = score_validation(&profile(), &observations);
        assert_eq!(scorecard.disposition, ValidationDisposition::InvalidObservations);
        assert!(scorecard.issues.iter().any(|issue| matches!(issue, ValidationIssue::UndeclaredSlice { .. })));
    }

    #[test]
    fn nonfinite_values_and_invalid_scales_are_rejected() {
        let mut observations = valid_observations();
        observations[0].predicted_value = f64::NAN;
        let scorecard = score_validation(&profile(), &observations);
        assert_eq!(scorecard.disposition, ValidationDisposition::InvalidObservations);

        let mut p = profile();
        p.required_slices[0].normalization_scale = 0.0;
        assert_eq!(score_validation(&p, &valid_observations()).disposition, ValidationDisposition::InvalidProfile);
    }

    #[test]
    fn deterministic_under_permutation_of_observations_and_requirements() {
        let mut p = profile();
        let mut observations = valid_observations();
        let original = score_validation(&p, &observations);
        p.required_slices.reverse();
        observations.reverse();
        let permuted = score_validation(&p, &observations);
        assert_eq!(original, permuted);
    }

    #[test]
    fn zero_error_has_zero_error_metrics() {
        let observations = vec![
            observation("deep-1", "depth-ratio-4.0", 10.0, 10.0),
            observation("deep-2", "depth-ratio-4.0", 20.0, 20.0),
            observation("shallow-1", "depth-ratio-1.2", 30.0, 30.0),
            observation("shallow-2", "depth-ratio-1.2", 40.0, 40.0),
        ];
        let scorecard = score_validation(&profile(), &observations);
        assert_eq!(scorecard.disposition, ValidationDisposition::PassDeclaredThresholds);
        assert!(scorecard.slices.iter().all(|s| s.root_mean_squared_error == Some(0.0)));
    }

    #[test]
    fn invalid_threshold_and_duplicate_required_slices_reject_profile() {
        let mut p = profile();
        p.required_slices[0].max_normalized_rmse = f64::INFINITY;
        p.required_slices.push(p.required_slices[0].clone());
        let scorecard = score_validation(&p, &valid_observations());
        assert_eq!(scorecard.disposition, ValidationDisposition::InvalidProfile);
        assert!(scorecard.issues.iter().any(|issue| matches!(issue, ValidationIssue::DuplicateRequiredSlice { .. })));
    }
}
