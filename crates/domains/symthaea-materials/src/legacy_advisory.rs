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
    /// Run and wrap the existing heuristic stability predictor.
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn serialized_advisory_round_trip_preserves_ceiling() {
        let wrapped = AdvisoryStabilityPredictionV1::predict(&[(26, 0.5), (28, 0.5)], 300.0);
        let json = serde_json::to_string(&wrapped).unwrap();
        let decoded: AdvisoryStabilityPredictionV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.authority, AdvisoryAuthorityV1::Advisory);
        assert_eq!(decoded.identity, wrapped.identity);
    }
}
