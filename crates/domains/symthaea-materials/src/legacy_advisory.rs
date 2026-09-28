//! Explicit epistemic boundary for legacy materials heuristics.
//!
//! These wrappers preserve exploratory/search functionality while making the
//! advisory ceiling machine-visible. They intentionally contain no conversion
//! into source-bearing scientific evidence.

use crate::aging::AgingPrediction;
use crate::compound_stability::StabilityPrediction;
use crate::database::MaterialSearchResult;
use crate::properties::{MaterialCategory, MaterialProperty};
use serde::{Deserialize, Serialize};

/// Explicit ceiling for legacy materials outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// Stable ordered case identifiers for the independent MAT-LEGACY-001B1 qualifier.
///
/// This is a deterministic software corpus manifest, not scientific evidence.
pub const LEGACY_ADVISORY_QUALIFIER_CASES_V1: &[&str] = &[
    "authority-explicit",
    "model-generation-distinct",
    "input-generation-distinct",
    "numeric-match-no-promotion",
    "fep-action-not-authorization",
    "malformed-stability-fails-closed",
    "nonfinite-feature-fails-closed",
    "serialization-preserves-ceiling",
    "domain-predictions-advisory",
    "no-scientific-evidence-conversion",
];

pub enum AdvisoryAuthorityV1 {
    /// Candidate generation, ranking, or hypothesis support only.
    Advisory,
}

/// Stable identity for an advisory computation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdvisoryIdentityV1 {
    /// Domain producing the advisory result.
    pub domain: String,
    /// Algorithm/model generation identifier.
    pub model_generation: String,
    /// Deterministic input generation identifier.
    pub input_generation: String,
}

impl AdvisoryIdentityV1 {
    /// Construct an advisory identity.
    pub fn new(domain: impl Into<String>, model_generation: impl Into<String>, input_generation: impl Into<String>) -> Self {
        Self { domain: domain.into(), model_generation: model_generation.into(), input_generation: input_generation.into() }
    }
}

/// A legacy material preset explicitly marked as non-evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvisoryMaterialPropertyV1 {
    /// Stable advisory identity.
    pub identity: AdvisoryIdentityV1,
    /// Legacy material value.
    pub material: MaterialProperty,
    /// Explicit epistemic ceiling.
    pub authority: AdvisoryAuthorityV1,
}

impl AdvisoryMaterialPropertyV1 {
    /// Wrap a legacy material without changing its value.
    pub fn from_legacy(material: MaterialProperty) -> Self {
        Self {
            identity: AdvisoryIdentityV1::new("materials.property_preset", "legacy-material-property-v1", "preset"),
            material,
            authority: AdvisoryAuthorityV1::Advisory,
        }
    }

    /// Expose the value for legacy/search workloads.
    pub fn material(&self) -> &MaterialProperty { &self.material }

    /// Return the legacy category without promoting authority.
    pub fn category(&self) -> MaterialCategory { self.material.category }
}

/// Validation failure for the safe advisory stability entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdvisoryInputError {
    /// Temperature is not finite or is negative.
    InvalidTemperature,
    /// An element fraction is not finite or is negative.
    InvalidFraction,
    /// Fractions do not form a normalized composition within tolerance.
    UnnormalizedComposition,
    /// An acquisition feature is not finite.
    InvalidFeatureValue,
}


/// Typed advisory boundary for the critical-minerals mining predictor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvisoryMiningPredictionV1 {
    /// Stable advisory identity.
    pub identity: AdvisoryIdentityV1,
    /// Explicit epistemic ceiling.
    pub authority: AdvisoryAuthorityV1,
    /// Prediction horizon in seconds.
    pub horizon_seconds: f32,
    /// Model-predicted HDC state; not an observation.
    pub predicted_state: Vec<f32>,
}

impl AdvisoryMiningPredictionV1 {
    /// Wrap a mining prediction as advisory-only model output.
    pub fn from_prediction(identity: AdvisoryIdentityV1, horizon_seconds: f32, predicted_state: Vec<f32>) -> Self {
        Self { identity, authority: AdvisoryAuthorityV1::Advisory, horizon_seconds, predicted_state }
    }
}

/// Typed advisory boundary for the strategic-materials predictor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvisoryStrategicPredictionV1 {
    /// Stable advisory identity.
    pub identity: AdvisoryIdentityV1,
    /// Explicit epistemic ceiling.
    pub authority: AdvisoryAuthorityV1,
    /// Prediction horizon in seconds.
    pub horizon_seconds: f32,
    /// Model-predicted HDC state; not an observation.
    pub predicted_state: Vec<f32>,
}

impl AdvisoryStrategicPredictionV1 {
    /// Wrap a strategic-materials prediction as advisory-only model output.
    pub fn from_prediction(identity: AdvisoryIdentityV1, horizon_seconds: f32, predicted_state: Vec<f32>) -> Self {
        Self { identity, authority: AdvisoryAuthorityV1::Advisory, horizon_seconds, predicted_state }
    }
}

/// Typed advisory boundary for a mining FEP action recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvisoryMiningActionV1 {
    /// Stable advisory identity.
    pub identity: AdvisoryIdentityV1,
    /// Explicit epistemic ceiling.
    pub authority: AdvisoryAuthorityV1,
    /// Model-recommended action; not an authorization.
    pub action: crate::mining::MiningFepAction,
}

/// Typed advisory boundary for a strategic-materials FEP action recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvisoryStrategicActionV1 {
    /// Stable advisory identity.
    pub identity: AdvisoryIdentityV1,
    /// Explicit epistemic ceiling.
    pub authority: AdvisoryAuthorityV1,
    /// Model-recommended action; not an authorization.
    pub action: crate::strategic::StrategicFepAction,
}

/// A legacy heuristic stability prediction, never a thermodynamic evidence record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvisoryStabilityPredictionV1 {
    /// Stable advisory identity.
    pub identity: AdvisoryIdentityV1,
    /// Legacy prediction payload.
    pub prediction: StabilityPrediction,
    /// Explicit epistemic ceiling.
    pub authority: AdvisoryAuthorityV1,
}

impl AdvisoryStabilityPredictionV1 {
    /// Safely run and wrap the existing heuristic stability predictor.
    ///
    /// This checked facade rejects malformed inputs before invoking the legacy
    /// predictor. It does not make the underlying heuristic scientifically authoritative.
    pub fn try_predict(elements: &[(u16, f64)], temperature_k: f64) -> Result<Self, AdvisoryInputError> {
        if !temperature_k.is_finite() || temperature_k < 0.0 {
            return Err(AdvisoryInputError::InvalidTemperature);
        }
        if elements.iter().any(|(_, x)| !x.is_finite() || *x < 0.0) {
            return Err(AdvisoryInputError::InvalidFraction);
        }
        let total: f64 = elements.iter().map(|(_, x)| *x).sum();
        if !elements.is_empty() && (total - 1.0).abs() > 1e-9 {
            return Err(AdvisoryInputError::UnnormalizedComposition);
        }
        Ok(Self::predict(elements, temperature_k))
    }

    /// Run and wrap the existing heuristic stability predictor without changing legacy behavior.
    pub fn predict(elements: &[(u16, f64)], temperature_k: f64) -> Self {
        Self {
            identity: AdvisoryIdentityV1::new(
                "materials.compound_stability",
                "legacy-miedema-inspired-v1",
                format!("elements-temp:{elements:?}:{temperature_k:?}"),
            ),
            prediction: crate::compound_stability::predict_stability(elements, temperature_k),
            authority: AdvisoryAuthorityV1::Advisory,
        }
    }

    /// Whether the heuristic predicts stability; advisory only.
    pub fn predicted_stable(&self) -> bool { self.prediction.is_stable }

    /// The heuristic's confidence; not calibrated uncertainty.
    pub fn advisory_confidence(&self) -> f64 { self.prediction.confidence }
}

/// An HDC search result explicitly classified as retrieval/ranking output.
#[derive(Debug, Clone)]
pub struct AdvisorySimilarityResultV1 {
    /// Stable advisory identity.
    pub identity: AdvisoryIdentityV1,
    /// Legacy search result.
    pub result: MaterialSearchResult,
    /// Explicit epistemic ceiling.
    pub authority: AdvisoryAuthorityV1,
}

impl AdvisorySimilarityResultV1 {
    /// Wrap a similarity result without treating similarity as evidence.
    pub fn from_legacy(result: MaterialSearchResult) -> Self {
        Self {
            identity: AdvisoryIdentityV1::new("materials.hdc_similarity", "legacy-material-hdc-v1", result.material.name.clone()),
            result,
            authority: AdvisoryAuthorityV1::Advisory,
        }
    }

    /// Similarity score for ranking only.
    pub fn similarity(&self) -> f32 { self.result.similarity }
}

/// A legacy aging prediction explicitly bounded by its model horizon.
#[derive(Debug, Clone)]
pub struct AdvisoryAgingPredictionV1 {
    /// Stable advisory identity.
    pub identity: AdvisoryIdentityV1,
    /// Legacy aging prediction.
    pub prediction: AgingPrediction,
    /// Explicit epistemic ceiling.
    pub authority: AdvisoryAuthorityV1,
}

impl AdvisoryAgingPredictionV1 {
    /// Wrap a legacy aging result and record its requested horizon.
    pub fn from_legacy(prediction: AgingPrediction) -> Self {
        Self {
            identity: AdvisoryIdentityV1::new("materials.aging", "legacy-cfc-hdc-v1", format!("horizon-seconds:{:?}", prediction.horizon_seconds)),
            prediction,
            authority: AdvisoryAuthorityV1::Advisory,
        }
    }

    /// Requested horizon represented by this advisory prediction.
    pub fn horizon_seconds(&self) -> f32 { self.prediction.horizon_seconds }
}

/// A search/acquisition feature with no evidence authority.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcquisitionFeatureV1 {
    /// Stable advisory identity.
    pub identity: AdvisoryIdentityV1,
    /// Feature name.
    pub name: String,
    /// Numeric feature value.
    pub value: f64,
    /// Explicit epistemic ceiling.
    pub authority: AdvisoryAuthorityV1,
}

impl AcquisitionFeatureV1 {
    /// Construct a search/acquisition feature.
    pub fn new(identity: AdvisoryIdentityV1, name: impl Into<String>, value: f64) -> Self {
        Self { identity, name: name.into(), value, authority: AdvisoryAuthorityV1::Advisory }
    }

    /// Validate a feature before it enters an acquisition/ranking path.
    pub fn try_new(identity: AdvisoryIdentityV1, name: impl Into<String>, value: f64) -> Result<Self, AdvisoryInputError> {
        if !value.is_finite() { return Err(AdvisoryInputError::InvalidFeatureValue); }
        Ok(Self::new(identity, name, value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualifier_case_manifest_is_ordered_and_unique() {
        let mut sorted = LEGACY_ADVISORY_QUALIFIER_CASES_V1.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), LEGACY_ADVISORY_QUALIFIER_CASES_V1.len());
        assert_eq!(LEGACY_ADVISORY_QUALIFIER_CASES_V1.first(), Some(&"authority-explicit"));
        assert_eq!(LEGACY_ADVISORY_QUALIFIER_CASES_V1.last(), Some(&"no-scientific-evidence-conversion"));
    }

    #[test]
    fn preset_is_explicitly_advisory() {
        let wrapped = AdvisoryMaterialPropertyV1::from_legacy(MaterialProperty::steel_a36());
        assert_eq!(wrapped.authority, AdvisoryAuthorityV1::Advisory);
        assert_eq!(wrapped.category(), MaterialCategory::Metal);
    }

    #[test]
    fn stability_true_does_not_change_authority() {
        let wrapped = AdvisoryStabilityPredictionV1::predict(&[(11, 0.5), (17, 0.5)], 300.0);
        assert!(wrapped.predicted_stable());
        assert_eq!(wrapped.authority, AdvisoryAuthorityV1::Advisory);
    }

    #[test]
    fn confidence_one_remains_advisory() {
        let mut prediction = crate::compound_stability::predict_stability(&[(26, 1.0)], 300.0);
        prediction.confidence = 1.0;
        let wrapped = AdvisoryStabilityPredictionV1 {
            identity: AdvisoryIdentityV1::new("test", "fixture", "confidence-one"),
            prediction,
            authority: AdvisoryAuthorityV1::Advisory,
        };
        assert_eq!(wrapped.advisory_confidence(), 1.0);
        assert_eq!(wrapped.authority, AdvisoryAuthorityV1::Advisory);
    }

    #[test]
    fn numeric_equality_does_not_merge_identity() {
        assert_ne!(
            AdvisoryIdentityV1::new("model-a", "generation-1", "input-1"),
            AdvisoryIdentityV1::new("model-b", "generation-1", "input-1")
        );
    }

    #[test]
    fn similarity_is_explicitly_advisory() {
        let result = MaterialSearchResult { material: MaterialProperty::steel_a36(), similarity: 1.0 };
        let wrapped = AdvisorySimilarityResultV1::from_legacy(result);
        assert_eq!(wrapped.similarity(), 1.0);
        assert_eq!(wrapped.authority, AdvisoryAuthorityV1::Advisory);
    }

    #[test]
    fn aging_horizon_is_part_of_identity() {
        let prediction = crate::aging::MaterialAgingModel::new().predict_at_horizon(&MaterialProperty::steel_a36(), 86_400.0);
        let wrapped = AdvisoryAgingPredictionV1::from_legacy(prediction);
        assert_eq!(wrapped.horizon_seconds(), 86_400.0);
        assert!(wrapped.identity.input_generation.contains("86400"));
    }

    #[test]
    fn malformed_stability_inputs_are_rejected_by_safe_facade() {
        assert_eq!(AdvisoryStabilityPredictionV1::try_predict(&[(26, f64::NAN)], 300.0), Err(AdvisoryInputError::InvalidFraction));
        assert_eq!(AdvisoryStabilityPredictionV1::try_predict(&[(26, 1.0)], f64::NAN), Err(AdvisoryInputError::InvalidTemperature));
        assert_eq!(AdvisoryStabilityPredictionV1::try_predict(&[(26, 0.2), (28, 0.2)], 300.0), Err(AdvisoryInputError::UnnormalizedComposition));
        assert!(AdvisoryStabilityPredictionV1::try_predict(&[(26, 0.5), (28, 0.5)], 300.0).is_ok());
    }


    #[test]
    fn domain_predictions_remain_advisory() {
        use crate::mining::{MiningHdcEncoder, MiningPredictor, MiningReading};
        use crate::strategic::{StrategicHdcEncoder, StrategicPredictor, StrategicReading};
        use symthaea_core::hdc::unified_hv::HDC_DIMENSION;

        let mining = MiningHdcEncoder::new().encode(&MiningReading { ore_grade: 0.5, extraction_rate: 0.7, environmental_impact: 0.1, cost: 0.4 });
        let strategic = StrategicHdcEncoder::new().encode(&StrategicReading { extreme_temp_resilience: 0.9, radiation_dose: 0.1, time_at_condition: 86_400.0, failure_probability: 0.001 });
        let mp = MiningPredictor::new().predict_at_horizon(&mining, 86_400.0);
        let sp = StrategicPredictor::new().predict_at_horizon(&strategic, 86_400.0);
        assert_eq!(mp.dim(), HDC_DIMENSION);
        assert_eq!(sp.dim(), HDC_DIMENSION);
        let mi = AdvisoryIdentityV1 { domain: "critical_minerals".into(), model_generation: "mining-predictor-v1".into(), input_generation: "synthetic-fixture-v1".into() };
        let si = AdvisoryIdentityV1 { domain: "strategic_materials".into(), model_generation: "strategic-predictor-v1".into(), input_generation: "synthetic-fixture-v1".into() };
        let ma = AdvisoryMiningPredictionV1 { identity: mi.clone(), authority: AdvisoryAuthorityV1::Advisory, horizon_seconds: 86_400.0, predicted_state: mp.values.clone() };
        let sa = AdvisoryStrategicPredictionV1 { identity: si.clone(), authority: AdvisoryAuthorityV1::Advisory, horizon_seconds: 86_400.0, predicted_state: sp.values.clone() };
        assert_eq!(ma.authority, AdvisoryAuthorityV1::Advisory);
        assert_eq!(sa.authority, AdvisoryAuthorityV1::Advisory);
    }

    #[test]
    fn acquisition_features_reject_non_finite_values() {
        let id = AdvisoryIdentityV1::new("acquisition", "fixture-v1", "input-v1");
        assert_eq!(AcquisitionFeatureV1::try_new(id.clone(), "score", f64::NAN), Err(AdvisoryInputError::InvalidFeatureValue));
        assert!(AcquisitionFeatureV1::try_new(id, "score", 0.5).is_ok());
    }

    #[test]
    fn domain_advisory_constructors_force_advisory_authority() {
        let id = AdvisoryIdentityV1::new("fixture", "model-v1", "input-v1");
        let mining = AdvisoryMiningPredictionV1::from_prediction(id.clone(), 10.0, vec![0.1, 0.2]);
        let strategic = AdvisoryStrategicPredictionV1::from_prediction(id, 10.0, vec![0.3, 0.4]);
        assert_eq!(mining.authority, AdvisoryAuthorityV1::Advisory);
        assert_eq!(strategic.authority, AdvisoryAuthorityV1::Advisory);
    }

    #[test]
    fn model_generation_changes_identity_even_when_payload_matches() {
        let a = AdvisoryIdentityV1::new("materials", "model-v1", "input-v1");
        let b = AdvisoryIdentityV1::new("materials", "model-v2", "input-v1");
        let pa = AdvisoryMiningPredictionV1::from_prediction(a.clone(), 10.0, vec![0.25, 0.75]);
        let pb = AdvisoryMiningPredictionV1::from_prediction(b.clone(), 10.0, vec![0.25, 0.75]);
        assert_eq!(pa.predicted_state, pb.predicted_state);
        assert_ne!(pa.identity, pb.identity);
        assert_eq!(pa.authority, AdvisoryAuthorityV1::Advisory);
        assert_eq!(pb.authority, AdvisoryAuthorityV1::Advisory);
    }

    #[test]
    fn input_generation_changes_identity_even_when_payload_matches() {
        let a = AdvisoryIdentityV1::new("materials", "model-v1", "input-v1");
        let b = AdvisoryIdentityV1::new("materials", "model-v1", "input-v2");
        let fa = AcquisitionFeatureV1::new(a.clone(), "acquisition-score", 0.5);
        let fb = AcquisitionFeatureV1::new(b.clone(), "acquisition-score", 0.5);
        assert_eq!(fa.value, fb.value);
        assert_ne!(fa.identity, fb.identity);
        assert_eq!(fa.authority, AdvisoryAuthorityV1::Advisory);
        assert_eq!(fb.authority, AdvisoryAuthorityV1::Advisory);
    }

    #[test]
    fn advisory_action_is_not_an_authorization() {
        use crate::mining::MiningFepAction;
        let action = AdvisoryMiningActionV1 {
            identity: AdvisoryIdentityV1::new("critical_minerals", "mining-fep-v1", "fixture-v1"),
            authority: AdvisoryAuthorityV1::Advisory,
            action: MiningFepAction::ContinueExtraction,
        };
        assert_eq!(action.authority, AdvisoryAuthorityV1::Advisory);
        assert_eq!(action.action, MiningFepAction::ContinueExtraction);
    }

    #[test]
    fn advisory_payload_cannot_become_scientific_by_numeric_match() {
        let advisory = AdvisoryStabilityPredictionV1::predict(&[(26, 0.5), (28, 0.5)], 300.0);
        let numeric_stability = advisory.prediction.is_stable;
        let numeric_confidence = advisory.prediction.confidence;

        // Equal numbers are intentionally insufficient evidence of common identity,
        // provenance, calibration, or authority.
        let hypothetical_measurement = (numeric_stability, numeric_confidence);
        assert_eq!(hypothetical_measurement.0, advisory.prediction.is_stable);
        assert_eq!(hypothetical_measurement.1, advisory.prediction.confidence);
        assert_eq!(advisory.authority, AdvisoryAuthorityV1::Advisory);
        assert_ne!(advisory.identity.domain, "measured.material.stability");
    }

    #[test]
    fn serialized_advisory_round_trip_preserves_ceiling() {
        let wrapped = AdvisoryStabilityPredictionV1::predict(&[(26, 0.5), (28, 0.5)], 300.0);
        let json = serde_json::to_string(&wrapped).unwrap();
        let decoded: AdvisoryStabilityPredictionV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.authority, AdvisoryAuthorityV1::Advisory);
        assert_eq!(decoded.identity, wrapped.identity);
    }
}
