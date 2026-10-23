// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Explicit prediction contracts for safety-oriented, evidence-grounded research.
//!
//! This module reports predictions for already-characterized compounds and
//! safety-relevant endpoints. It deliberately has no representation for
//! dosage, route, psychoactive intensity, euphoria, or potency optimization.
//! Numerical confidence is not asserted to be calibrated merely because a
//! value exists; calibration requires explicit validation provenance.

use serde::{Deserialize, Serialize};

/// Version of the serialized assessment contract.
pub const PREDICTION_SCHEMA_VERSION: &str = "1.0";

/// Nature of the endpoint being assessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SafetyEndpointKind {
    ToxicityClassification,
    AdverseEventRiskClassification,
    SafetyRelevantContinuousEndpoint,
}

/// Whether the model considers an input inside its declared applicability domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApplicabilityStatus {
    InDomain,
    Borderline,
    OutOfDomain,
    NotAssessed,
}

/// State of empirical calibration evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationStatus {
    NotAssessed,
    AssessedButNotCalibrated,
    CalibratedOnHeldOutValidationSet,
}

/// Structured reason that no numeric prediction is being returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AbstentionReason {
    InsufficientEvidence,
    AmbiguousCompoundIdentity,
    ConflictingEvidence,
    OutOfApplicabilityDomain,
    UnsupportedEndpoint,
    CalibrationUnavailable,
    HumanReviewRequired,
}

/// Bounded numeric result for a safety-relevant task. This is a model output,
/// not a clinical conclusion or evidence that a compound is safe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SafetyPredictionValue {
    /// Estimated probability that the predeclared positive/hazard label applies.
    /// The label definition belongs in `endpoint_definition` and in the
    /// versioned benchmark manifest.
    HazardProbability { probability: f64 },
    /// Continuous estimate, including an optional prediction interval.
    ContinuousEstimate {
        estimate: f64,
        unit: String,
        lower_bound: Option<f64>,
        upper_bound: Option<f64>,
    },
}

/// Minimal version/provenance record for the model and data used to produce an
/// assessment. Digests should bind to immutable manifests, not mutable labels.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredictionProvenance {
    pub model_id: String,
    pub model_version: String,
    pub model_artifact_digest: String,
    pub training_data_manifest_digest: String,
    /// ISO-8601 date YYYY-MM-DD; the latest evidence allowed during training.
    pub training_data_cutoff: String,
    pub benchmark_split_id: String,
    pub inference_config_digest: String,
}

/// Calibration and uncertainty information. Values remain descriptive until
/// the calibration method and validation dataset can be independently checked.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredictionUncertainty {
    pub calibration_status: CalibrationStatus,
    pub calibration_method: Option<String>,
    pub calibration_validation_manifest_digest: Option<String>,
    /// Optional entropy/variance/ensemble disagreement or other named signal.
    pub uncertainty_method: Option<String>,
    pub uncertainty_value: Option<f64>,
    pub limitations: Vec<String>,
}

/// One prediction-or-abstention record. The invariant is exactly one of
/// `prediction` or `abstention_reason`; callers cannot publish both or neither.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredictionAssessment {
    pub schema_version: String,
    pub assessment_id: String,
    pub compound_id: String,
    pub endpoint: SafetyEndpointKind,
    pub endpoint_definition: String,
    pub prediction: Option<SafetyPredictionValue>,
    pub abstention_reason: Option<AbstentionReason>,
    pub applicability: ApplicabilityStatus,
    pub uncertainty: PredictionUncertainty,
    pub provenance: PredictionProvenance,
    pub supporting_evidence_record_ids: Vec<String>,
    pub contradictory_evidence_record_ids: Vec<String>,
    pub limitations: Vec<String>,
}

impl PredictionAssessment {
    /// Validate publication invariants. This validates record consistency;
    /// it does not certify pharmacological accuracy, calibration, or safety.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        if self.schema_version != PREDICTION_SCHEMA_VERSION {
            errors.push(format!(
                "unsupported schema_version {:?}; expected {:?}",
                self.schema_version, PREDICTION_SCHEMA_VERSION
            ));
        }
        require_nonempty(&self.assessment_id, "assessment_id", &mut errors);
        require_nonempty(&self.compound_id, "compound_id", &mut errors);
        require_nonempty(&self.endpoint_definition, "endpoint_definition", &mut errors);
        require_nonempty(&self.provenance.model_id, "provenance.model_id", &mut errors);
        require_nonempty(&self.provenance.model_version, "provenance.model_version", &mut errors);
        require_digest(
            &self.provenance.model_artifact_digest,
            "provenance.model_artifact_digest",
            &mut errors,
        );
        require_digest(
            &self.provenance.training_data_manifest_digest,
            "provenance.training_data_manifest_digest",
            &mut errors,
        );
        require_digest(
            &self.provenance.inference_config_digest,
            "provenance.inference_config_digest",
            &mut errors,
        );
        require_nonempty(
            &self.provenance.benchmark_split_id,
            "provenance.benchmark_split_id",
            &mut errors,
        );
        if chrono::NaiveDate::parse_from_str(&self.provenance.training_data_cutoff, "%Y-%m-%d")
            .is_err()
        {
            errors.push("provenance.training_data_cutoff must be a valid YYYY-MM-DD date".into());
        }

        match (&self.prediction, &self.abstention_reason) {
            (Some(_), Some(_)) => errors.push(
                "prediction and abstention_reason cannot both be present".into(),
            ),
            (None, None) => errors.push(
                "exactly one of prediction or abstention_reason must be present".into(),
            ),
            _ => {}
        }

        if self.applicability == ApplicabilityStatus::OutOfDomain
            && self.abstention_reason != Some(AbstentionReason::OutOfApplicabilityDomain)
        {
            errors.push(
                "out-of-domain assessment must abstain with OutOfApplicabilityDomain".into(),
            );
        }

        match &self.prediction {
            Some(SafetyPredictionValue::HazardProbability { probability }) => {
                if !probability.is_finite() || !(0.0..=1.0).contains(probability) {
                    errors.push("hazard probability must be finite and within [0, 1]".into());
                }
                if self.uncertainty.calibration_status == CalibrationStatus::CalibratedOnHeldOutValidationSet
                    && (self.uncertainty.calibration_method.is_none()
                        || self.uncertainty.calibration_validation_manifest_digest.is_none())
                {
                    errors.push(
                        "calibrated status requires method and validation manifest digest".into(),
                    );
                }
            }
            Some(SafetyPredictionValue::ContinuousEstimate {
                estimate,
                unit,
                lower_bound,
                upper_bound,
            }) => {
                if !estimate.is_finite() {
                    errors.push("continuous estimate must be finite".into());
                }
                require_nonempty(unit, "prediction.unit", &mut errors);
                for (label, value) in [("lower_bound", lower_bound), ("upper_bound", upper_bound)] {
                    if value.is_some_and(|v| !v.is_finite()) {
                        errors.push(format!("prediction.{label} must be finite when present"));
                    }
                }
                if let (Some(lower), Some(upper)) = (lower_bound, upper_bound) {
                    if lower > upper {
                        errors.push("prediction lower_bound must be <= upper_bound".into());
                    }
                    if estimate < lower || estimate > upper {
                        errors.push("continuous estimate must lie within its reported interval".into());
                    }
                }
                if self.uncertainty.calibration_status == CalibrationStatus::CalibratedOnHeldOutValidationSet
                    && (self.uncertainty.calibration_method.is_none()
                        || self.uncertainty.calibration_validation_manifest_digest.is_none())
                {
                    errors.push(
                        "calibrated status requires method and validation manifest digest".into(),
                    );
                }
            }
            None => {}
        }

        if let Some(value) = self.uncertainty.uncertainty_value {
            if !value.is_finite() || value < 0.0 {
                errors.push("uncertainty_value must be finite and non-negative".into());
            }
        }
        if self.uncertainty.calibration_status == CalibrationStatus::CalibratedOnHeldOutValidationSet {
            if self.uncertainty.calibration_method.as_deref().is_none_or(str::is_empty) {
                errors.push("calibrated status requires a non-empty calibration_method".into());
            }
            if self
                .uncertainty
                .calibration_validation_manifest_digest
                .as_deref()
                .is_none_or(|digest| !is_digest(digest))
            {
                errors.push(
                    "calibrated status requires a 64-character validation manifest digest".into(),
                );
            }
        }
        if let Some(digest) = &self.uncertainty.calibration_validation_manifest_digest {
            if !is_digest(digest) {
                errors.push(
                    "uncertainty.calibration_validation_manifest_digest must be a 64-character hexadecimal digest".into(),
                );
            }
        }
        if let Some(method) = &self.uncertainty.calibration_method {
            if method.trim().is_empty() {
                errors.push("calibration_method must not be empty when present".into());
            }
        }

        if self.supporting_evidence_record_ids.iter().any(|id| id.trim().is_empty()) {
            errors.push("supporting_evidence_record_ids must not contain empty IDs".into());
        }
        if self.contradictory_evidence_record_ids.iter().any(|id| id.trim().is_empty()) {
            errors.push("contradictory_evidence_record_ids must not contain empty IDs".into());
        }
        if self.supporting_evidence_record_ids.iter().any(|id| {
            self.contradictory_evidence_record_ids.iter().any(|other| other == id)
        }) {
            errors.push("same evidence record cannot be both supporting and contradictory".into());
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

fn require_nonempty(value: &str, field: &str, errors: &mut Vec<String>) {
    if value.trim().is_empty() {
        errors.push(format!("{field} must not be empty"));
    }
}

fn is_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn require_digest(value: &str, field: &str, errors: &mut Vec<String>) {
    if !is_digest(value) {
        errors.push(format!("{field} must be a 64-character hexadecimal digest"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assessment() -> PredictionAssessment {
        PredictionAssessment {
            schema_version: PREDICTION_SCHEMA_VERSION.to_string(),
            assessment_id: "assessment-001".to_string(),
            compound_id: "known-compound-id-001".to_string(),
            endpoint: SafetyEndpointKind::ToxicityClassification,
            endpoint_definition: "positive means the pre-registered hazard label is present".to_string(),
            prediction: Some(SafetyPredictionValue::HazardProbability { probability: 0.25 }),
            abstention_reason: None,
            applicability: ApplicabilityStatus::InDomain,
            uncertainty: PredictionUncertainty {
                calibration_status: CalibrationStatus::NotAssessed,
                calibration_method: None,
                calibration_validation_manifest_digest: None,
                uncertainty_method: Some("ensemble disagreement".to_string()),
                uncertainty_value: Some(0.12),
                limitations: vec!["illustrative fixture only".to_string()],
            },
            provenance: PredictionProvenance {
                model_id: "example-model".to_string(),
                model_version: "0.1.0".to_string(),
                model_artifact_digest: "a".repeat(64),
                training_data_manifest_digest: "b".repeat(64),
                training_data_cutoff: "2025-12-31".to_string(),
                benchmark_split_id: "frozen-split-v1".to_string(),
                inference_config_digest: "c".repeat(64),
            },
            supporting_evidence_record_ids: vec!["ev-1".to_string()],
            contradictory_evidence_record_ids: vec!["ev-2".to_string()],
            limitations: vec!["No clinical conclusion follows from this prediction.".to_string()],
        }
    }

    #[test]
    fn basic_safety_prediction_is_structurally_valid() {
        assert!(assessment().validate().is_ok());
    }

    #[test]
    fn exactly_one_prediction_or_abstention_is_required() {
        let mut both = assessment();
        both.abstention_reason = Some(AbstentionReason::HumanReviewRequired);
        assert!(both.validate().expect_err("both must fail").iter().any(|e| e.contains("both be present")));

        let mut neither = assessment();
        neither.prediction = None;
        assert!(neither.validate().expect_err("neither must fail").iter().any(|e| e.contains("exactly one")));
    }

    #[test]
    fn out_of_domain_prediction_must_fail_closed() {
        let mut record = assessment();
        record.applicability = ApplicabilityStatus::OutOfDomain;
        assert!(record.validate().expect_err("must abstain").iter().any(|e| e.contains("out-of-domain")));

        record.prediction = None;
        record.abstention_reason = Some(AbstentionReason::OutOfApplicabilityDomain);
        assert!(record.validate().is_ok());
    }

    #[test]
    fn probability_must_be_finite_and_bounded() {
        for probability in [f64::NAN, f64::INFINITY, -0.01, 1.01] {
            let mut record = assessment();
            record.prediction = Some(SafetyPredictionValue::HazardProbability { probability });
            assert!(record.validate().expect_err("bad probability must fail").iter().any(|e| e.contains("within [0, 1]")));
        }
    }

    #[test]
    fn calibrated_claim_requires_held_out_validation_provenance() {
        let mut record = assessment();
        record.uncertainty.calibration_status = CalibrationStatus::CalibratedOnHeldOutValidationSet;
        assert!(record.validate().expect_err("missing calibration provenance must fail").iter().any(|e| e.contains("calibrated status")));

        record.uncertainty.calibration_method = Some("isotonic regression".to_string());
        record.uncertainty.calibration_validation_manifest_digest = Some("d".repeat(64));
        assert!(record.validate().is_ok());
    }

    #[test]
    fn continuous_prediction_interval_must_be_finite_ordered_and_contain_estimate() {
        let mut record = assessment();
        record.endpoint = SafetyEndpointKind::SafetyRelevantContinuousEndpoint;
        record.prediction = Some(SafetyPredictionValue::ContinuousEstimate {
            estimate: 5.0,
            unit: "normalized_unit".to_string(),
            lower_bound: Some(6.0),
            upper_bound: Some(8.0),
        });
        assert!(record.validate().expect_err("estimate outside interval must fail").iter().any(|e| e.contains("within its reported interval")));
    }

    #[test]
    fn supporting_and_contradictory_evidence_sets_must_not_overlap() {
        let mut record = assessment();
        record.contradictory_evidence_record_ids = vec!["ev-1".to_string()];
        assert!(record.validate().expect_err("overlap must fail").iter().any(|e| e.contains("both supporting and contradictory")));
    }

    #[test]
    fn rejects_bad_provenance_digest_and_training_cutoff() {
        let mut record = assessment();
        record.provenance.model_artifact_digest = "bad".to_string();
        record.provenance.training_data_cutoff = "not-a-date".to_string();
        let errors = record.validate().expect_err("bad provenance must fail");
        assert!(errors.iter().any(|e| e.contains("model_artifact_digest")));
        assert!(errors.iter().any(|e| e.contains("training_data_cutoff")));
    }

    #[test]
    fn json_round_trip_is_lossless() {
        let original = assessment();
        let encoded = serde_json::to_string(&original).expect("serialize");
        let decoded: PredictionAssessment = serde_json::from_str(&encoded).expect("deserialize");
        assert_eq!(decoded, original);
        assert!(decoded.validate().is_ok());
    }
}
