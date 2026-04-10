// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Scoring Model Registry — Pluggable Governance Scoring
//!
//! Defines the [`ScoringModel`] trait and provides built-in implementations
//! that communities can choose from.  All models are validated against the
//! [`ConstitutionalEnvelope`](crate::constitutional_envelope) before activation.
//!
//! ## Built-in Models
//!
//! | Model | Dims | Default Use Case |
//! |-------|------|-----------------|
//! | [`Canonical4D`] | 4 | The original I/R/C/E profile |
//! | [`Sovereign8D`] | 8 | Anti-tyranny civic identity |
//! | [`MinimalCivic`] | 3 | Simple communities (Identity/Reputation/Engagement) |
//!
//! ## Research Workflow
//!
//! 1. Implement [`ScoringModel`] for your model
//! 2. Call [`validate_model`](crate::constitutional_envelope::validate_model) to check invariants
//! 3. Register in the [`ModelRegistry`]
//! 4. Run in shadow mode (Phase 3) to gather divergence data
//! 5. Community votes to adopt via Steward-tier governance

use serde::{Deserialize, Serialize};

use crate::constitutional_envelope::{
    self, sanitize_score, ConstitutionalViolation, LAMBDA_MAX, LAMBDA_MIN,
};

// ============================================================================
// ScoringModel Trait
// ============================================================================

/// A pluggable governance scoring model.
///
/// Each model defines dimensions, weights, and decay — all validated
/// against constitutional invariants before activation.
pub trait ScoringModel {
    /// Unique identifier (e.g., "canonical-4d-v1", "sovereign-8d-v1").
    fn model_id(&self) -> &str;

    /// Human-readable name.
    fn display_name(&self) -> &str;

    /// Number of dimensions.
    fn dimension_count(&self) -> usize;

    /// Names of each dimension (for audit trails and display).
    fn dimension_names(&self) -> &[&str];

    /// Per-dimension weights.  Must sum to 1.0, each ≤ 0.50.
    fn weights(&self) -> &[f64];

    /// Compute combined score from dimension values.
    ///
    /// The default implementation is a weighted linear sum.  Override
    /// for non-linear scoring (e.g., quadratic mean, minimum-threshold).
    fn compute_score(&self, dimensions: &[f64]) -> f64 {
        let w = self.weights();
        if dimensions.len() != w.len() {
            return 0.0;
        }
        let raw: f64 = dimensions
            .iter()
            .zip(w.iter())
            .map(|(&d, &w)| sanitize_score(d) * w)
            .sum();
        sanitize_score(raw)
    }

    /// Decay rate (lambda per day).  Must be in [LAMBDA_MIN, LAMBDA_MAX].
    fn decay_lambda(&self) -> f64;

    /// Validate this model against constitutional invariants.
    fn validate(&self) -> Result<(), ConstitutionalViolation> {
        constitutional_envelope::validate_model(self.weights(), self.decay_lambda(), 72)
    }
}

// ============================================================================
// Built-in Models
// ============================================================================

/// The canonical 4-dimensional governance profile.
///
/// Dimensions: Identity (I), Reputation (R), Community (C), Engagement (E).
/// Weights: 0.25 / 0.25 / 0.30 / 0.20.
///
/// This is the battle-tested model from the original Mycelix architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Canonical4D {
    /// Per-dimension weights (default: [0.25, 0.25, 0.30, 0.20]).
    pub weights: [f64; 4],
    /// Decay rate per day (default: 0.002, half-life ≈ 347 days).
    pub lambda: f64,
}

impl Default for Canonical4D {
    fn default() -> Self {
        Self {
            weights: [0.25, 0.25, 0.30, 0.20],
            lambda: 0.002,
        }
    }
}

const CANONICAL_4D_NAMES: [&str; 4] = ["Identity", "Reputation", "Community", "Engagement"];

impl ScoringModel for Canonical4D {
    fn model_id(&self) -> &str {
        "canonical-4d-v1"
    }
    fn display_name(&self) -> &str {
        "Canonical 4D (I/R/C/E)"
    }
    fn dimension_count(&self) -> usize {
        4
    }
    fn dimension_names(&self) -> &[&str] {
        &CANONICAL_4D_NAMES
    }
    fn weights(&self) -> &[f64] {
        &self.weights
    }
    fn decay_lambda(&self) -> f64 {
        self.lambda
    }
}

/// The 8-dimensional anti-tyranny civic profile.
///
/// Dimensions: Epistemic Integrity, Thermodynamic Yield, Network Resilience,
/// Economic Velocity, Civic Participation, Stewardship Care, Semantic
/// Resonance, Domain Competence.
///
/// Community-configurable weight presets: governance (default), energy
/// cooperative, knowledge commons, care community.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sovereign8D {
    /// Per-dimension weights (must sum to 1.0, each ≤ 0.50).
    pub weights: [f64; 8],
    /// Decay rate per day.
    pub lambda: f64,
    /// Name of the active weight preset (informational).
    pub preset_name: String,
}

impl Default for Sovereign8D {
    fn default() -> Self {
        Self::governance()
    }
}

const SOVEREIGN_8D_NAMES: [&str; 8] = [
    "Epistemic Integrity",
    "Thermodynamic Yield",
    "Network Resilience",
    "Economic Velocity",
    "Civic Participation",
    "Stewardship Care",
    "Semantic Resonance",
    "Domain Competence",
];

impl Sovereign8D {
    /// Governance preset: balanced civic participation.
    pub fn governance() -> Self {
        Self {
            weights: [0.15, 0.10, 0.10, 0.12, 0.18, 0.13, 0.12, 0.10],
            lambda: 0.002,
            preset_name: "governance".into(),
        }
    }

    /// Energy cooperative: thermodynamic yield and network resilience elevated.
    pub fn energy_cooperative() -> Self {
        Self {
            weights: [0.08, 0.22, 0.18, 0.10, 0.12, 0.12, 0.08, 0.10],
            lambda: 0.005,
            preset_name: "energy_cooperative".into(),
        }
    }

    /// Knowledge commons: epistemic integrity and domain competence elevated.
    pub fn knowledge_commons() -> Self {
        Self {
            weights: [0.22, 0.06, 0.06, 0.08, 0.12, 0.10, 0.16, 0.20],
            lambda: 0.002,
            preset_name: "knowledge_commons".into(),
        }
    }

    /// Care community: stewardship and semantic resonance elevated.
    pub fn care_community() -> Self {
        Self {
            weights: [0.08, 0.06, 0.06, 0.08, 0.15, 0.22, 0.20, 0.15],
            lambda: 0.003,
            preset_name: "care_community".into(),
        }
    }
}

impl ScoringModel for Sovereign8D {
    fn model_id(&self) -> &str {
        "sovereign-8d-v1"
    }
    fn display_name(&self) -> &str {
        "Sovereign 8D (Anti-Tyranny)"
    }
    fn dimension_count(&self) -> usize {
        8
    }
    fn dimension_names(&self) -> &[&str] {
        &SOVEREIGN_8D_NAMES
    }
    fn weights(&self) -> &[f64] {
        &self.weights
    }
    fn decay_lambda(&self) -> f64 {
        self.lambda
    }
}

/// A minimal 3-dimensional model for simple communities.
///
/// Dimensions: Identity, Reputation, Engagement.
/// Drops Community (peer attestation) for bootstrapping scenarios where
/// the community is too small for meaningful peer review.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimalCivic {
    /// Per-dimension weights (default: [0.35, 0.35, 0.30]).
    pub weights: [f64; 3],
    /// Decay rate per day.
    pub lambda: f64,
}

impl Default for MinimalCivic {
    fn default() -> Self {
        Self {
            weights: [0.35, 0.35, 0.30],
            lambda: 0.003,
        }
    }
}

const MINIMAL_CIVIC_NAMES: [&str; 3] = ["Identity", "Reputation", "Engagement"];

impl ScoringModel for MinimalCivic {
    fn model_id(&self) -> &str {
        "minimal-civic-v1"
    }
    fn display_name(&self) -> &str {
        "Minimal Civic (I/R/E)"
    }
    fn dimension_count(&self) -> usize {
        3
    }
    fn dimension_names(&self) -> &[&str] {
        &MINIMAL_CIVIC_NAMES
    }
    fn weights(&self) -> &[f64] {
        &self.weights
    }
    fn decay_lambda(&self) -> f64 {
        self.lambda
    }
}

// ============================================================================
// Model Registry
// ============================================================================

/// A serializable model descriptor for DHT storage and cross-cluster transfer.
///
/// This is the data representation of a scoring model — the trait impl
/// lives in code, but the parameters can be stored on-chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDescriptor {
    /// Unique model identifier.
    pub model_id: String,
    /// Human-readable name.
    pub display_name: String,
    /// Dimension names.
    pub dimension_names: Vec<String>,
    /// Per-dimension weights.
    pub weights: Vec<f64>,
    /// Decay rate (lambda per day).
    pub lambda: f64,
    /// When this model was registered (microseconds since epoch).
    pub registered_at: u64,
    /// Who registered this model (agent DID).
    pub registered_by: String,
}

impl ModelDescriptor {
    /// Create a descriptor from any ScoringModel implementation.
    pub fn from_model(model: &dyn ScoringModel, registered_at: u64, registered_by: String) -> Self {
        Self {
            model_id: model.model_id().to_string(),
            display_name: model.display_name().to_string(),
            dimension_names: model.dimension_names().iter().map(|s| s.to_string()).collect(),
            weights: model.weights().to_vec(),
            lambda: model.decay_lambda(),
            registered_at,
            registered_by,
        }
    }

    /// Validate against constitutional invariants.
    pub fn validate(&self) -> Result<(), ConstitutionalViolation> {
        constitutional_envelope::validate_model(&self.weights, self.lambda, 72)
    }

    /// Compute a score using this descriptor's weights (linear weighted sum).
    pub fn compute_score(&self, dimensions: &[f64]) -> f64 {
        if dimensions.len() != self.weights.len() {
            return 0.0;
        }
        let raw: f64 = dimensions
            .iter()
            .zip(self.weights.iter())
            .map(|(&d, &w)| sanitize_score(d) * w)
            .sum();
        sanitize_score(raw)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constitutional_envelope::score_to_tier;
    use crate::consciousness_profile::ConsciousnessTier;

    // ---- Canonical 4D ----

    #[test]
    fn canonical_4d_passes_constitutional_check() {
        let model = Canonical4D::default();
        assert!(model.validate().is_ok());
    }

    #[test]
    fn canonical_4d_weights_sum_to_one() {
        let model = Canonical4D::default();
        let sum: f64 = model.weights().iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn canonical_4d_score_computation() {
        let model = Canonical4D::default();
        // I=0.8, R=0.6, C=0.7, E=0.5
        let score = model.compute_score(&[0.8, 0.6, 0.7, 0.5]);
        // 0.8*0.25 + 0.6*0.25 + 0.7*0.30 + 0.5*0.20 = 0.20+0.15+0.21+0.10 = 0.66
        assert!(
            (score - 0.66).abs() < 1e-6,
            "Expected 0.66, got {}",
            score
        );
        assert_eq!(score_to_tier(score), ConsciousnessTier::Steward);
    }

    #[test]
    fn canonical_4d_zero_dimensions_gives_observer() {
        let model = Canonical4D::default();
        let score = model.compute_score(&[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(score, 0.0);
        assert_eq!(score_to_tier(score), ConsciousnessTier::Observer);
    }

    #[test]
    fn canonical_4d_max_dimensions_gives_guardian() {
        let model = Canonical4D::default();
        let score = model.compute_score(&[1.0, 1.0, 1.0, 1.0]);
        assert!((score - 1.0).abs() < 1e-6);
        assert_eq!(score_to_tier(score), ConsciousnessTier::Guardian);
    }

    #[test]
    fn canonical_4d_wrong_dimension_count_returns_zero() {
        let model = Canonical4D::default();
        assert_eq!(model.compute_score(&[0.5, 0.5]), 0.0); // too few
        assert_eq!(model.compute_score(&[0.5; 8]), 0.0); // too many
    }

    // ---- Sovereign 8D ----

    #[test]
    fn sovereign_8d_governance_passes_constitutional_check() {
        assert!(Sovereign8D::governance().validate().is_ok());
    }

    #[test]
    fn sovereign_8d_energy_passes_constitutional_check() {
        assert!(Sovereign8D::energy_cooperative().validate().is_ok());
    }

    #[test]
    fn sovereign_8d_knowledge_passes_constitutional_check() {
        assert!(Sovereign8D::knowledge_commons().validate().is_ok());
    }

    #[test]
    fn sovereign_8d_care_passes_constitutional_check() {
        assert!(Sovereign8D::care_community().validate().is_ok());
    }

    #[test]
    fn sovereign_8d_all_presets_have_8_dimensions() {
        for model in [
            Sovereign8D::governance(),
            Sovereign8D::energy_cooperative(),
            Sovereign8D::knowledge_commons(),
            Sovereign8D::care_community(),
        ] {
            assert_eq!(model.dimension_count(), 8);
            assert_eq!(model.dimension_names().len(), 8);
            assert_eq!(model.weights().len(), 8);
        }
    }

    #[test]
    fn sovereign_8d_no_weight_exceeds_50_percent() {
        for model in [
            Sovereign8D::governance(),
            Sovereign8D::energy_cooperative(),
            Sovereign8D::knowledge_commons(),
            Sovereign8D::care_community(),
        ] {
            for (i, &w) in model.weights().iter().enumerate() {
                assert!(
                    w <= 0.50,
                    "Preset '{}' dimension {} weight {} > 0.50",
                    model.preset_name,
                    i,
                    w
                );
            }
        }
    }

    #[test]
    fn sovereign_8d_score_computation() {
        let model = Sovereign8D::governance();
        let dims = [0.8, 0.5, 0.6, 0.7, 0.9, 0.6, 0.7, 0.5];
        let score = model.compute_score(&dims);
        // Manual: 0.8*0.15 + 0.5*0.10 + 0.6*0.10 + 0.7*0.12 + 0.9*0.18 + 0.6*0.13 + 0.7*0.12 + 0.5*0.10
        //       = 0.120 + 0.050 + 0.060 + 0.084 + 0.162 + 0.078 + 0.084 + 0.050 = 0.688
        assert!(
            (score - 0.688).abs() < 1e-6,
            "Expected 0.688, got {}",
            score
        );
        assert_eq!(score_to_tier(score), ConsciousnessTier::Steward);
    }

    #[test]
    fn sovereign_8d_energy_preset_weights_thermodynamic() {
        let model = Sovereign8D::energy_cooperative();
        // Thermodynamic Yield (index 1) should be highest
        let max_idx = model
            .weights()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 1, "Energy preset should weight Thermodynamic Yield highest");
    }

    // ---- Minimal Civic ----

    #[test]
    fn minimal_civic_passes_constitutional_check() {
        assert!(MinimalCivic::default().validate().is_ok());
    }

    #[test]
    fn minimal_civic_has_3_dimensions() {
        let model = MinimalCivic::default();
        assert_eq!(model.dimension_count(), 3);
    }

    #[test]
    fn minimal_civic_score_computation() {
        let model = MinimalCivic::default();
        let score = model.compute_score(&[0.8, 0.6, 0.5]);
        // 0.8*0.35 + 0.6*0.35 + 0.5*0.30 = 0.28+0.21+0.15 = 0.64
        assert!(
            (score - 0.64).abs() < 1e-6,
            "Expected 0.64, got {}",
            score
        );
    }

    // ---- Model Descriptor ----

    #[test]
    fn descriptor_from_model_preserves_data() {
        let model = Canonical4D::default();
        let desc = ModelDescriptor::from_model(&model, 1000, "did:mycelix:test".into());
        assert_eq!(desc.model_id, "canonical-4d-v1");
        assert_eq!(desc.weights.len(), 4);
        assert_eq!(desc.lambda, 0.002);
    }

    #[test]
    fn descriptor_validates_against_constitution() {
        let model = Canonical4D::default();
        let desc = ModelDescriptor::from_model(&model, 1000, "did:mycelix:test".into());
        assert!(desc.validate().is_ok());
    }

    #[test]
    fn descriptor_computes_same_score_as_model() {
        let model = Canonical4D::default();
        let desc = ModelDescriptor::from_model(&model, 1000, "did:mycelix:test".into());
        let dims = [0.8, 0.6, 0.7, 0.5];
        let model_score = model.compute_score(&dims);
        let desc_score = desc.compute_score(&dims);
        assert!((model_score - desc_score).abs() < 1e-10);
    }

    #[test]
    fn descriptor_serde_roundtrip() {
        let model = Sovereign8D::governance();
        let desc = ModelDescriptor::from_model(&model, 1000, "did:mycelix:test".into());
        let json = serde_json::to_string(&desc).unwrap();
        let back: ModelDescriptor = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model_id, desc.model_id);
        assert_eq!(back.weights, desc.weights);
        assert_eq!(back.lambda, desc.lambda);
    }

    #[test]
    fn invalid_descriptor_rejected() {
        let desc = ModelDescriptor {
            model_id: "bad-model".into(),
            display_name: "Bad".into(),
            dimension_names: vec!["A".into(), "B".into()],
            weights: vec![0.60, 0.40], // 0.60 > 0.50 max
            lambda: 0.002,
            registered_at: 0,
            registered_by: "did:mycelix:bad".into(),
        };
        assert!(desc.validate().is_err());
    }

    // ---- Cross-model consistency ----

    #[test]
    fn all_builtin_models_have_unique_ids() {
        let m1 = Canonical4D::default();
        let m2 = Sovereign8D::default();
        let m3 = MinimalCivic::default();
        let ids: Vec<&str> = vec![m1.model_id(), m2.model_id(), m3.model_id()];
        let unique: std::collections::HashSet<&str> = ids.iter().copied().collect();
        assert_eq!(ids.len(), unique.len(), "Model IDs must be unique");
    }

    #[test]
    fn all_builtin_models_pass_constitution() {
        let models: Vec<Box<dyn ScoringModel>> = vec![
            Box::new(Canonical4D::default()),
            Box::new(Sovereign8D::governance()),
            Box::new(Sovereign8D::energy_cooperative()),
            Box::new(Sovereign8D::knowledge_commons()),
            Box::new(Sovereign8D::care_community()),
            Box::new(MinimalCivic::default()),
        ];
        for model in &models {
            assert!(
                model.validate().is_ok(),
                "Model '{}' failed constitutional check: {:?}",
                model.model_id(),
                model.validate().err()
            );
        }
    }

    #[test]
    fn all_builtin_models_produce_observer_for_zero_input() {
        let models: Vec<Box<dyn ScoringModel>> = vec![
            Box::new(Canonical4D::default()),
            Box::new(Sovereign8D::governance()),
            Box::new(MinimalCivic::default()),
        ];
        for model in &models {
            let zeros = vec![0.0; model.dimension_count()];
            let score = model.compute_score(&zeros);
            assert_eq!(
                score, 0.0,
                "Model '{}' should return 0.0 for zero input",
                model.model_id()
            );
        }
    }

    #[test]
    fn all_builtin_models_produce_max_for_unit_input() {
        let models: Vec<Box<dyn ScoringModel>> = vec![
            Box::new(Canonical4D::default()),
            Box::new(Sovereign8D::governance()),
            Box::new(MinimalCivic::default()),
        ];
        for model in &models {
            let ones = vec![1.0; model.dimension_count()];
            let score = model.compute_score(&ones);
            assert!(
                (score - 1.0).abs() < 1e-6,
                "Model '{}' should return 1.0 for all-ones input, got {}",
                model.model_id(),
                score
            );
        }
    }

    #[test]
    fn sanitize_nan_dimension_produces_valid_score() {
        let model = Canonical4D::default();
        let score = model.compute_score(&[f64::NAN, 0.5, 0.5, 0.5]);
        assert!(score.is_finite(), "NaN dimension should produce finite score");
        assert!(score >= 0.0 && score <= 1.0);
    }
}
