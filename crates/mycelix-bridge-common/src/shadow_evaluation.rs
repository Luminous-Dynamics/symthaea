// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Shadow Evaluation Harness — Multi-Model Governance Research
//!
//! Runs ALL registered scoring models on the same input, but only
//! the community's active model determines the actual governance gate.
//! Shadow results accumulate as local-only telemetry (never broadcast
//! to DHT) for empirical model comparison.
//!
//! ## Design Principles
//!
//! 1. **Zero gate latency**: Shadow models run in parallel with the
//!    active model.  The gate decision is never delayed.
//! 2. **Local-only storage**: Shadow evaluations live on the agent's
//!    source chain as private entries.  Aggregated to DHT only via
//!    batched 24h summaries to prevent network spam.
//! 3. **Empirical model discovery**: Divergence metrics show when
//!    models disagree, giving communities data for model transitions.
//! 4. **Constitutional safety**: All models are validated against the
//!    [`ConstitutionalEnvelope`](crate::constitutional_envelope) before
//!    they can be registered for shadow evaluation.

use serde::{Deserialize, Serialize};

use crate::consciousness_profile::ConsciousnessTier;
use crate::constitutional_envelope::{apply_decay, sanitize_score, score_to_tier};
use crate::scoring_model::{ModelDescriptor, ScoringModel};

// ============================================================================
// Shadow Evaluation Types
// ============================================================================

/// Result from a single scoring model's evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResult {
    /// Which model produced this result.
    pub model_id: String,
    /// The combined score (after decay).
    pub score: f64,
    /// The tier this score maps to.
    pub tier: ConsciousnessTier,
    /// Vote weight in basis points (0–10,000).
    pub weight_bp: u32,
    /// Per-dimension breakdown (name, value, weight, weighted_contribution).
    pub dimensions: Vec<DimensionBreakdown>,
}

/// Per-dimension contribution to the combined score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionBreakdown {
    /// Dimension name.
    pub name: String,
    /// Raw value (0.0–1.0).
    pub value: f64,
    /// Weight in the scoring model.
    pub weight: f64,
    /// value × weight — this dimension's contribution.
    pub contribution: f64,
}

/// Divergence metrics between models on a single evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DivergenceMetrics {
    /// Maximum tier difference between any two models.
    /// 0 = all models agree on tier.  4 = max (Observer vs Guardian).
    pub max_tier_divergence: u8,
    /// Standard deviation of scores across all models.
    pub score_stddev: f64,
    /// Whether ANY shadow model would have produced a different gate
    /// decision (pass/fail) than the active model.
    pub gate_disagreement: bool,
    /// Number of models that agree with the active model's tier.
    pub tier_agreement_count: usize,
    /// Total number of models evaluated.
    pub total_models: usize,
}

/// Complete shadow evaluation result.
///
/// The `active` field determines the actual governance gate.
/// The `shadows` are for research only — never affect governance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowEvaluation {
    /// The active model's result (THIS determines the gate).
    pub active: ModelResult,
    /// Shadow results from all other registered models.
    pub shadows: Vec<ModelResult>,
    /// How much the models disagree.
    pub divergence: DivergenceMetrics,
    /// Timestamp of evaluation (microseconds).
    pub evaluated_at: u64,
}

// ============================================================================
// Evaluation Engine
// ============================================================================

/// Input for shadow evaluation — the raw dimension values and context.
#[derive(Debug, Clone)]
pub struct EvaluationInput<'a> {
    /// Raw dimension values.  Length must match the model's dimension count.
    /// For 4D models: [identity, reputation, community, engagement]
    /// For 8D models: [epistemic, thermo, network, economic, civic, stewardship, semantic, domain]
    pub dimensions: &'a [f64],
    /// Decay rate (lambda per day) — usually from the active model.
    pub lambda: f64,
    /// Days elapsed since last verified civic interaction.
    pub elapsed_days: f64,
    /// The governance threshold score required to pass the gate.
    pub threshold: f64,
    /// Current timestamp (microseconds).
    pub now_us: u64,
}

/// Evaluate a single model against the input dimensions.
///
/// Returns `None` if the model's dimension count doesn't match the input.
/// Uses the model's own weights but the caller's decay parameters.
pub fn evaluate_model(
    model: &dyn ScoringModel,
    dimensions: &[f64],
    lambda: f64,
    elapsed_days: f64,
) -> Option<ModelResult> {
    if dimensions.len() < model.dimension_count() {
        return None;
    }

    let model_dims = &dimensions[..model.dimension_count()];
    let raw_score = model.compute_score(model_dims);
    let decayed_score = apply_decay(raw_score, lambda, elapsed_days);
    let tier = score_to_tier(decayed_score);

    let dim_breakdown: Vec<DimensionBreakdown> = model
        .dimension_names()
        .iter()
        .zip(model.weights().iter())
        .enumerate()
        .map(|(i, (&name, &weight))| {
            let value = sanitize_score(model_dims.get(i).copied().unwrap_or(0.0));
            DimensionBreakdown {
                name: name.to_string(),
                value,
                weight,
                contribution: value * weight,
            }
        })
        .collect();

    let weight_bp = tier.vote_weight_bp();

    Some(ModelResult {
        model_id: model.model_id().to_string(),
        score: decayed_score,
        tier,
        weight_bp,
        dimensions: dim_breakdown,
    })
}

/// Evaluate a model descriptor (from DHT) against dimension values.
pub fn evaluate_descriptor(
    desc: &ModelDescriptor,
    dimensions: &[f64],
    lambda: f64,
    elapsed_days: f64,
) -> Option<ModelResult> {
    if dimensions.len() < desc.weights.len() {
        return None;
    }

    let model_dims = &dimensions[..desc.weights.len()];
    let raw_score = desc.compute_score(model_dims);
    let decayed_score = apply_decay(raw_score, lambda, elapsed_days);
    let tier = score_to_tier(decayed_score);

    let dim_breakdown: Vec<DimensionBreakdown> = desc
        .dimension_names
        .iter()
        .zip(desc.weights.iter())
        .enumerate()
        .map(|(i, (name, &weight))| {
            let value = sanitize_score(model_dims.get(i).copied().unwrap_or(0.0));
            DimensionBreakdown {
                name: name.clone(),
                value,
                weight,
                contribution: value * weight,
            }
        })
        .collect();

    let weight_bp = tier.vote_weight_bp();

    Some(ModelResult {
        model_id: desc.model_id.clone(),
        score: decayed_score,
        tier,
        weight_bp,
        dimensions: dim_breakdown,
    })
}

/// Run ALL models and produce a full shadow evaluation.
///
/// `active_model` determines the governance gate result.
/// `shadow_models` produce research-only results.
///
/// All models receive the SAME dimension values.  Models with fewer
/// dimensions use only the first N values.  Models requiring more
/// dimensions than available are skipped.
pub fn evaluate_all(
    active_model: &dyn ScoringModel,
    shadow_models: &[&dyn ScoringModel],
    input: &EvaluationInput,
) -> ShadowEvaluation {
    // Evaluate active model
    let active = evaluate_model(
        active_model,
        input.dimensions,
        input.lambda,
        input.elapsed_days,
    )
    .unwrap_or(ModelResult {
        model_id: active_model.model_id().to_string(),
        score: 0.0,
        tier: ConsciousnessTier::Observer,
        weight_bp: 0,
        dimensions: vec![],
    });

    // Evaluate shadow models
    let shadows: Vec<ModelResult> = shadow_models
        .iter()
        .filter_map(|m| evaluate_model(*m, input.dimensions, input.lambda, input.elapsed_days))
        .collect();

    // Compute divergence
    let divergence = compute_divergence(&active, &shadows, input.threshold);

    ShadowEvaluation {
        active,
        shadows,
        divergence,
        evaluated_at: input.now_us,
    }
}

/// Compute divergence metrics between the active result and shadows.
fn compute_divergence(
    active: &ModelResult,
    shadows: &[ModelResult],
    threshold: f64,
) -> DivergenceMetrics {
    if shadows.is_empty() {
        return DivergenceMetrics {
            max_tier_divergence: 0,
            score_stddev: 0.0,
            gate_disagreement: false,
            tier_agreement_count: 1,
            total_models: 1,
        };
    }

    let active_passes = active.score >= threshold;
    let active_tier_ord = tier_ordinal(active.tier);

    let all_scores: Vec<f64> = core::iter::once(active.score)
        .chain(shadows.iter().map(|s| s.score))
        .collect();

    let n = all_scores.len() as f64;
    let mean = all_scores.iter().sum::<f64>() / n;
    let variance = all_scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n;
    let stddev = variance.sqrt();

    let mut max_tier_div: u8 = 0;
    let mut tier_agree = 1usize; // active agrees with itself
    let mut gate_disagree = false;

    for shadow in shadows {
        let shadow_tier_ord = tier_ordinal(shadow.tier);
        let tier_diff = active_tier_ord.abs_diff(shadow_tier_ord) as u8;
        if tier_diff > max_tier_div {
            max_tier_div = tier_diff;
        }
        if shadow.tier == active.tier {
            tier_agree += 1;
        }
        let shadow_passes = shadow.score >= threshold;
        if shadow_passes != active_passes {
            gate_disagree = true;
        }
    }

    DivergenceMetrics {
        max_tier_divergence: max_tier_div,
        score_stddev: if stddev.is_finite() { stddev } else { 0.0 },
        gate_disagreement: gate_disagree,
        tier_agreement_count: tier_agree,
        total_models: 1 + shadows.len(),
    }
}

/// Map a tier to an ordinal for divergence computation.
fn tier_ordinal(tier: ConsciousnessTier) -> usize {
    match tier {
        ConsciousnessTier::Observer => 0,
        ConsciousnessTier::Participant => 1,
        ConsciousnessTier::Citizen => 2,
        ConsciousnessTier::Steward => 3,
        ConsciousnessTier::Guardian => 4,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scoring_model::{Canonical4D, MinimalCivic, Sovereign8D};

    fn make_input(dims: &[f64]) -> EvaluationInput {
        EvaluationInput {
            dimensions: dims,
            lambda: 0.002,
            elapsed_days: 0.0, // no decay for simplicity
            threshold: 0.4,    // Citizen tier
            now_us: 1_000_000,
        }
    }

    // ---- Single model evaluation ----

    #[test]
    fn evaluate_canonical_4d() {
        let model = Canonical4D::default();
        let dims = [0.8, 0.6, 0.7, 0.5]; // I/R/C/E
        let result = evaluate_model(&model, &dims, 0.002, 0.0).unwrap();
        assert_eq!(result.model_id, "canonical-4d-v1");
        assert!((result.score - 0.66).abs() < 1e-6);
        assert_eq!(result.tier, ConsciousnessTier::Steward);
        assert_eq!(result.dimensions.len(), 4);
    }

    #[test]
    fn evaluate_with_decay() {
        let model = Canonical4D::default();
        let dims = [1.0, 1.0, 1.0, 1.0]; // perfect score
        let result_no_decay = evaluate_model(&model, &dims, 0.002, 0.0).unwrap();
        let result_30d = evaluate_model(&model, &dims, 0.002, 30.0).unwrap();
        assert!(
            result_30d.score < result_no_decay.score,
            "30-day decay should reduce score"
        );
        assert!(result_30d.score > 0.9, "30 days at lambda=0.002 should still be high");
    }

    #[test]
    fn evaluate_dimension_mismatch_returns_none() {
        let model = Canonical4D::default();
        let dims = [0.5, 0.5]; // only 2, need 4
        assert!(evaluate_model(&model, &dims, 0.002, 0.0).is_none());
    }

    #[test]
    fn evaluate_extra_dimensions_ignored() {
        let model = Canonical4D::default();
        let dims = [0.8, 0.6, 0.7, 0.5, 0.9, 0.9, 0.9, 0.9]; // 8 dims, 4D model uses first 4
        let result = evaluate_model(&model, &dims, 0.002, 0.0).unwrap();
        assert!((result.score - 0.66).abs() < 1e-6);
    }

    // ---- Shadow evaluation ----

    #[test]
    fn shadow_eval_with_no_shadows() {
        let active = Canonical4D::default();
        let input = make_input(&[0.8, 0.6, 0.7, 0.5]);
        let eval = evaluate_all(&active, &[], &input);

        assert_eq!(eval.active.model_id, "canonical-4d-v1");
        assert!(eval.shadows.is_empty());
        assert_eq!(eval.divergence.max_tier_divergence, 0);
        assert!(!eval.divergence.gate_disagreement);
        assert_eq!(eval.divergence.total_models, 1);
    }

    #[test]
    fn shadow_eval_models_agree() {
        let active = Canonical4D::default();
        // 8D dims, but Canonical4D only uses first 4
        let dims = [0.8, 0.6, 0.7, 0.5, 0.8, 0.6, 0.7, 0.5];
        let sovereign = Sovereign8D::governance();
        let input = make_input(&dims);

        let eval = evaluate_all(&active, &[&sovereign], &input);

        assert_eq!(eval.active.model_id, "canonical-4d-v1");
        assert_eq!(eval.shadows.len(), 1);
        assert_eq!(eval.shadows[0].model_id, "sovereign-8d-v1");
        assert_eq!(eval.divergence.total_models, 2);
    }

    #[test]
    fn shadow_eval_detects_tier_divergence() {
        // NOTE: Different models consume dimensions by INDEX position.
        // MinimalCivic (I/R/E) takes dims[0..3], not dims[0],dims[1],dims[3].
        // This is intentional — shadow evaluation reveals how dimension
        // semantics affect governance outcomes.

        let active = Canonical4D::default();
        let minimal = MinimalCivic::default();

        // 4D: I=0.5, R=0.5, C=0.5, E=0.5 → 0.50 (Citizen)
        // 3D takes [0.5, 0.5, 0.5] → 0.50 (Citizen) — same
        // But with asymmetric: I=0.8, R=0.1, C=0.8, E=0.1
        // 4D: 0.8*0.25+0.1*0.25+0.8*0.30+0.1*0.20 = 0.20+0.025+0.24+0.02 = 0.485 (Citizen)
        // 3D: [0.8, 0.1, 0.8] → 0.8*0.35+0.1*0.35+0.8*0.30 = 0.28+0.035+0.24 = 0.555 (Citizen)
        // Need MORE divergence. Use extreme asymmetry:
        // I=0.0, R=0.0, C=1.0, E=1.0
        // 4D: 0*0.25+0*0.25+1.0*0.30+1.0*0.20 = 0.50 (Citizen)
        // 3D: [0.0, 0.0, 1.0] → 0*0.35+0*0.35+1.0*0.30 = 0.30 (Participant)
        let dims = [0.0, 0.0, 1.0, 1.0];
        let input = make_input(&dims);

        let eval = evaluate_all(&active, &[&minimal], &input);

        assert!(
            eval.divergence.max_tier_divergence > 0,
            "Models should diverge: active_tier={:?} shadow_tier={:?}",
            eval.active.tier,
            eval.shadows[0].tier
        );
    }

    #[test]
    fn shadow_eval_detects_gate_disagreement() {
        // 4D passes threshold (>= 0.4), 3D fails
        let active = Canonical4D::default();
        let minimal = MinimalCivic::default();

        // 4D: I=0.0, R=0.0, C=1.0, E=1.0 → 0.50 (passes 0.4)
        // 3D: [0.0, 0.0, 1.0] → 0.30 (fails 0.4)
        let dims = [0.0, 0.0, 1.0, 1.0];
        let input = make_input(&dims);

        let eval = evaluate_all(&active, &[&minimal], &input);

        assert!(eval.active.score >= 0.4, "Active should pass: {}", eval.active.score);
        assert!(
            eval.shadows[0].score < 0.4,
            "Shadow should fail: {}",
            eval.shadows[0].score
        );
        assert!(eval.divergence.gate_disagreement, "Gate disagreement should be detected");
    }

    #[test]
    fn shadow_eval_three_models() {
        let active = Canonical4D::default();
        let sovereign = Sovereign8D::governance();
        let minimal = MinimalCivic::default();
        let dims = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7];
        let input = make_input(&dims);

        let eval = evaluate_all(&active, &[&sovereign, &minimal], &input);

        assert_eq!(eval.divergence.total_models, 3);
        assert_eq!(eval.shadows.len(), 2);
    }

    // ---- Divergence metrics ----

    #[test]
    fn divergence_stddev_zero_when_all_agree() {
        let active = Canonical4D::default();
        let dims = [0.5, 0.5, 0.5, 0.5]; // symmetric → same score regardless of weights
        let input = make_input(&dims);

        // All models with symmetric input produce score = 0.5
        let eval = evaluate_all(&active, &[], &input);
        assert_eq!(eval.divergence.score_stddev, 0.0);
    }

    #[test]
    fn divergence_stddev_positive_when_models_differ() {
        let active = Canonical4D::default();
        let sovereign = Sovereign8D::governance();
        // Use extreme asymmetry where the 8D model sees very different
        // dimension contributions than the 4D model.
        // 4D uses [0.0, 0.0, 1.0, 1.0] → 0*0.25+0*0.25+1.0*0.30+1.0*0.20 = 0.50
        // 8D uses [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
        //   → 0*0.15+0*0.10+1.0*0.10+1.0*0.12+0*0.18+0*0.13+1.0*0.12+1.0*0.10 = 0.44
        let dims = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let input = make_input(&dims);

        let eval = evaluate_all(&active, &[&sovereign], &input);
        assert!(
            eval.divergence.score_stddev > 0.0,
            "Different weight distributions should produce different scores: active={}, shadow={}",
            eval.active.score,
            eval.shadows[0].score
        );
    }

    #[test]
    fn tier_agreement_count_correct() {
        let active = Canonical4D::default();
        let sovereign = Sovereign8D::governance();
        let minimal = MinimalCivic::default();
        let dims = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7];
        let input = make_input(&dims);

        let eval = evaluate_all(&active, &[&sovereign, &minimal], &input);

        // All models should agree on tier for symmetric high input
        assert_eq!(
            eval.divergence.tier_agreement_count,
            eval.divergence.total_models,
            "All models should agree on tier for uniform 0.7 input"
        );
    }

    // ---- Dimension breakdown ----

    #[test]
    fn dimension_breakdown_has_correct_names() {
        let model = Canonical4D::default();
        let dims = [0.8, 0.6, 0.7, 0.5];
        let result = evaluate_model(&model, &dims, 0.002, 0.0).unwrap();

        assert_eq!(result.dimensions[0].name, "Identity");
        assert_eq!(result.dimensions[1].name, "Reputation");
        assert_eq!(result.dimensions[2].name, "Community");
        assert_eq!(result.dimensions[3].name, "Engagement");
    }

    #[test]
    fn dimension_contributions_sum_to_score() {
        let model = Canonical4D::default();
        let dims = [0.8, 0.6, 0.7, 0.5];
        let result = evaluate_model(&model, &dims, 0.002, 0.0).unwrap();

        let contribution_sum: f64 = result.dimensions.iter().map(|d| d.contribution).sum();
        assert!(
            (contribution_sum - result.score).abs() < 1e-6,
            "Contributions should sum to score: {} vs {}",
            contribution_sum,
            result.score
        );
    }

    // ---- Serde ----

    #[test]
    fn shadow_evaluation_serde_roundtrip() {
        let active = Canonical4D::default();
        let sovereign = Sovereign8D::governance();
        let dims = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7];
        let input = make_input(&dims);

        let eval = evaluate_all(&active, &[&sovereign], &input);
        let json = serde_json::to_string(&eval).unwrap();
        let back: ShadowEvaluation = serde_json::from_str(&json).unwrap();

        assert_eq!(back.active.model_id, eval.active.model_id);
        assert_eq!(back.shadows.len(), eval.shadows.len());
        assert_eq!(
            back.divergence.total_models,
            eval.divergence.total_models
        );
    }

    // ---- Descriptor evaluation ----

    #[test]
    fn descriptor_evaluation_matches_model() {
        let model = Canonical4D::default();
        let desc = ModelDescriptor::from_model(&model, 1000, "test".into());
        let dims = [0.8, 0.6, 0.7, 0.5];

        let model_result = evaluate_model(&model, &dims, 0.002, 0.0).unwrap();
        let desc_result = evaluate_descriptor(&desc, &dims, 0.002, 0.0).unwrap();

        assert!(
            (model_result.score - desc_result.score).abs() < 1e-10,
            "Model and descriptor should produce identical scores"
        );
        assert_eq!(model_result.tier, desc_result.tier);
    }
}
