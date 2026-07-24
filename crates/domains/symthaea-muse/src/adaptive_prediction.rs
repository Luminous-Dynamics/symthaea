// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Calibrated, context-sensitive musical outcome prediction.
//!
//! V3 closed the symbolic prediction loop but used one fixed outcome vector per
//! action. This module turns accumulated prediction error into an explicit
//! online calibration model. It never learns notes and never relaxes theory
//! constraints: it only learns what measured symbolic effects a named action
//! tends to have in a named context.

use crate::cognitive_bridge::{
    CognitiveDecisionTrace, CognitiveSection, ObservedMusicalOutcome, PredictedMusicalOutcome,
    SymbolicAction, default_predicted_outcome,
};
use crate::intervention::{InterventionDescriptor, InterventionStrategy};
use crate::piece_recipe::RecipeDecision;
use serde::{Deserialize, Serialize};

pub const LEGACY_ADAPTIVE_OUTCOME_MODEL_VERSION: &str = "adaptive-outcome-v1";
pub const ADAPTIVE_OUTCOME_MODEL_VERSION: &str = "adaptive-outcome-v2";

/// Coarse texture context used without introducing a learned latent state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextureBand {
    Sparse,
    Chamber,
    Dense,
}

impl TextureBand {
    pub fn from_active_voices(active_voice_count: usize) -> Self {
        match active_voice_count {
            0 | 1 => Self::Sparse,
            2 | 3 => Self::Chamber,
            _ => Self::Dense,
        }
    }
}

/// Stable context key for one prediction calibration cell.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredictionContext {
    pub action: SymbolicAction,
    pub section: CognitiveSection,
    pub style_name: String,
    pub form_name: String,
    pub meter: u8,
    pub texture_band: TextureBand,
}

impl PredictionContext {
    pub fn new(
        action: SymbolicAction,
        section: CognitiveSection,
        style_name: impl Into<String>,
        form_name: impl Into<String>,
        meter: u8,
        texture_band: TextureBand,
    ) -> Self {
        Self {
            action,
            section,
            style_name: style_name.into(),
            form_name: form_name.into(),
            meter,
            texture_band,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct ChannelMoments {
    pub samples: u64,
    pub mean: f64,
    pub m2: f64,
}

impl ChannelMoments {
    fn observe(&mut self, value: f32) {
        self.samples += 1;
        let delta = value as f64 - self.mean;
        self.mean += delta / self.samples as f64;
        let delta_after = value as f64 - self.mean;
        self.m2 += delta * delta_after;
    }

    fn mean_f32(self) -> f32 {
        self.mean as f32
    }

    fn standard_deviation(self) -> f32 {
        if self.samples < 2 {
            0.5
        } else {
            (self.m2 / (self.samples - 1) as f64).max(0.0).sqrt() as f32
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct OutcomeMoments {
    pub tension: ChannelMoments,
    pub density: ChannelMoments,
    pub familiarity: ChannelMoments,
    pub tonal_displacement: ChannelMoments,
}

impl OutcomeMoments {
    pub fn samples(self) -> u64 {
        self.tension.samples
    }

    fn observe(&mut self, outcome: ObservedMusicalOutcome) {
        self.tension.observe(outcome.tension_delta);
        self.density.observe(outcome.density_delta);
        self.familiarity.observe(outcome.familiarity_delta);
        self.tonal_displacement
            .observe(outcome.tonal_displacement_delta);
    }

    fn mean(self) -> PredictedMusicalOutcome {
        PredictedMusicalOutcome {
            tension_delta: self.tension.mean_f32(),
            density_delta: self.density.mean_f32(),
            familiarity_delta: self.familiarity.mean_f32(),
            tonal_displacement_delta: self.tonal_displacement.mean_f32(),
        }
    }

    fn uncertainty(self) -> OutcomeUncertainty {
        OutcomeUncertainty {
            tension: self.tension.standard_deviation(),
            density: self.density.standard_deviation(),
            familiarity: self.familiarity.standard_deviation(),
            tonal_displacement: self.tonal_displacement.standard_deviation(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextOutcomeCalibration {
    pub context: PredictionContext,
    pub outcomes: OutcomeMoments,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionOutcomeCalibration {
    pub action: SymbolicAction,
    pub outcomes: OutcomeMoments,
}

/// Parameterized context for one concrete intervention strategy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterventionPredictionContext {
    pub base: PredictionContext,
    pub descriptor: InterventionDescriptor,
}

impl InterventionPredictionContext {
    pub fn new(base: PredictionContext, descriptor: InterventionDescriptor) -> Self {
        debug_assert_eq!(base.action, descriptor.action);
        Self { base, descriptor }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterventionOutcomeCalibration {
    pub context: InterventionPredictionContext,
    pub outcomes: OutcomeMoments,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StrategyOutcomeCalibration {
    pub action: SymbolicAction,
    pub strategy: InterventionStrategy,
    pub outcomes: OutcomeMoments,
}

/// Online calibration state. Vec-backed cells keep serialization stable and
/// avoid requiring ordering semantics from musical enums.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveOutcomeModel {
    pub model_version: String,
    pub exact_contexts: Vec<ContextOutcomeCalibration>,
    pub action_fallbacks: Vec<ActionOutcomeCalibration>,
    /// Exact parameterized interventions introduced in model v2.
    #[serde(default)]
    pub intervention_contexts: Vec<InterventionOutcomeCalibration>,
    /// Strategy-level fallback between exact intervention and action evidence.
    #[serde(default)]
    pub strategy_fallbacks: Vec<StrategyOutcomeCalibration>,
    /// Samples before an exact context becomes the primary evidence source.
    pub min_exact_context_samples: u64,
    /// Pseudo-observation weight retaining the hand-authored prior.
    pub prior_strength: f32,
}

impl Default for AdaptiveOutcomeModel {
    fn default() -> Self {
        Self {
            model_version: ADAPTIVE_OUTCOME_MODEL_VERSION.into(),
            exact_contexts: Vec::new(),
            action_fallbacks: Vec::new(),
            intervention_contexts: Vec::new(),
            strategy_fallbacks: Vec::new(),
            min_exact_context_samples: 4,
            prior_strength: 4.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredictionEvidenceSource {
    HandAuthoredPrior,
    ActionFallback,
    StrategyFallback,
    ExactContext,
    InterventionContext,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OutcomeUncertainty {
    pub tension: f32,
    pub density: f32,
    pub familiarity: f32,
    pub tonal_displacement: f32,
}

impl Default for OutcomeUncertainty {
    fn default() -> Self {
        Self {
            tension: 0.5,
            density: 0.5,
            familiarity: 0.5,
            tonal_displacement: 0.5,
        }
    }
}

/// Evidence explaining where one calibrated prediction came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredictionCalibrationEvidence {
    pub model_version: String,
    pub context: PredictionContext,
    pub source: PredictionEvidenceSource,
    pub exact_context_samples: u64,
    pub action_fallback_samples: u64,
    /// Sufficient statistics for independently reproducing the calibration.
    #[serde(default)]
    pub exact_context_moments: Option<OutcomeMoments>,
    #[serde(default)]
    pub action_fallback_moments: Option<OutcomeMoments>,
    pub prior: PredictedMusicalOutcome,
    pub calibrated: PredictedMusicalOutcome,
    pub uncertainty: OutcomeUncertainty,
}

/// Evidence for a prediction of one concrete, parameterized intervention.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterventionCalibrationEvidence {
    pub model_version: String,
    pub context: InterventionPredictionContext,
    pub source: PredictionEvidenceSource,
    pub intervention_context_samples: u64,
    pub strategy_fallback_samples: u64,
    pub action_fallback_samples: u64,
    #[serde(default)]
    pub intervention_context_moments: Option<OutcomeMoments>,
    #[serde(default)]
    pub strategy_fallback_moments: Option<OutcomeMoments>,
    #[serde(default)]
    pub action_fallback_moments: Option<OutcomeMoments>,
    pub prior: PredictedMusicalOutcome,
    pub calibrated: PredictedMusicalOutcome,
    pub uncertainty: OutcomeUncertainty,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptiveLearningError {
    MissingCalibrationEvidence,
    MissingObservedOutcome,
    NonFiniteOutcome,
    IncompatibleIntervention,
}

impl AdaptiveOutcomeModel {
    /// Reject persisted state whose semantics or sufficient statistics cannot
    /// be interpreted by this implementation.
    pub fn is_compatible(&self) -> bool {
        (self.model_version == ADAPTIVE_OUTCOME_MODEL_VERSION
            || self.model_version == LEGACY_ADAPTIVE_OUTCOME_MODEL_VERSION)
            && self.min_exact_context_samples > 0
            && self.prior_strength.is_finite()
            && self.prior_strength >= 0.0
            && self
                .exact_contexts
                .iter()
                .all(|entry| moments_are_valid(entry.outcomes))
            && self
                .action_fallbacks
                .iter()
                .all(|entry| moments_are_valid(entry.outcomes))
            && self.intervention_contexts.iter().all(|entry| {
                entry.context.descriptor.is_compatible()
                    && entry.context.base.action == entry.context.descriptor.action
                    && moments_are_valid(entry.outcomes)
            })
            && self
                .strategy_fallbacks
                .iter()
                .all(|entry| moments_are_valid(entry.outcomes))
    }

    /// Upgrade a valid v1 model in memory while retaining its action and
    /// coarse-context sufficient statistics. New parameterized cells begin
    /// empty and are learned only from v2 evidence.
    pub fn upgrade_legacy(&mut self) -> bool {
        if self.model_version == LEGACY_ADAPTIVE_OUTCOME_MODEL_VERSION && self.is_compatible() {
            self.model_version = ADAPTIVE_OUTCOME_MODEL_VERSION.into();
            true
        } else {
            false
        }
    }

    pub fn predict(&self, context: &PredictionContext) -> PredictionCalibrationEvidence {
        let prior = default_predicted_outcome(context.action);
        let action = self
            .action_fallbacks
            .iter()
            .find(|entry| entry.action == context.action)
            .map(|entry| entry.outcomes);
        let exact = self
            .exact_contexts
            .iter()
            .find(|entry| entry.context == *context)
            .map(|entry| entry.outcomes);

        let action_samples = action.map_or(0, OutcomeMoments::samples);
        let exact_samples = exact.map_or(0, OutcomeMoments::samples);
        let (source, calibrated, uncertainty) = if let Some(exact) = exact
            && exact_samples >= self.min_exact_context_samples
        {
            let fallback = action.map_or(prior, |moments| {
                shrink(
                    prior,
                    moments.mean(),
                    moments.samples(),
                    self.prior_strength,
                )
            });
            (
                PredictionEvidenceSource::ExactContext,
                shrink(fallback, exact.mean(), exact_samples, self.prior_strength),
                exact.uncertainty(),
            )
        } else if let Some(action) = action {
            (
                PredictionEvidenceSource::ActionFallback,
                shrink(prior, action.mean(), action_samples, self.prior_strength),
                action.uncertainty(),
            )
        } else {
            (
                PredictionEvidenceSource::HandAuthoredPrior,
                prior,
                OutcomeUncertainty::default(),
            )
        };

        PredictionCalibrationEvidence {
            model_version: self.model_version.clone(),
            context: context.clone(),
            source,
            exact_context_samples: exact_samples,
            action_fallback_samples: action_samples,
            exact_context_moments: exact,
            action_fallback_moments: action,
            prior,
            calibrated,
            uncertainty,
        }
    }

    pub fn predict_intervention(
        &self,
        context: &InterventionPredictionContext,
    ) -> InterventionCalibrationEvidence {
        let prior = default_predicted_outcome(context.base.action);
        let action = self
            .action_fallbacks
            .iter()
            .find(|entry| entry.action == context.base.action)
            .map(|entry| entry.outcomes);
        let strategy = self
            .strategy_fallbacks
            .iter()
            .find(|entry| {
                entry.action == context.base.action && entry.strategy == context.descriptor.strategy
            })
            .map(|entry| entry.outcomes);
        let exact = self
            .intervention_contexts
            .iter()
            .find(|entry| entry.context == *context)
            .map(|entry| entry.outcomes);

        let action_samples = action.map_or(0, OutcomeMoments::samples);
        let strategy_samples = strategy.map_or(0, OutcomeMoments::samples);
        let exact_samples = exact.map_or(0, OutcomeMoments::samples);
        let action_prediction = action.map_or(prior, |moments| {
            shrink(
                prior,
                moments.mean(),
                moments.samples(),
                self.prior_strength,
            )
        });
        let strategy_prediction = strategy.map_or(action_prediction, |moments| {
            shrink(
                action_prediction,
                moments.mean(),
                moments.samples(),
                self.prior_strength,
            )
        });
        let (source, calibrated, uncertainty) = if let Some(exact) = exact
            && exact_samples >= self.min_exact_context_samples
        {
            (
                PredictionEvidenceSource::InterventionContext,
                shrink(
                    strategy_prediction,
                    exact.mean(),
                    exact_samples,
                    self.prior_strength,
                ),
                exact.uncertainty(),
            )
        } else if let Some(strategy) = strategy {
            (
                PredictionEvidenceSource::StrategyFallback,
                strategy_prediction,
                strategy.uncertainty(),
            )
        } else if let Some(action) = action {
            (
                PredictionEvidenceSource::ActionFallback,
                action_prediction,
                action.uncertainty(),
            )
        } else {
            (
                PredictionEvidenceSource::HandAuthoredPrior,
                prior,
                OutcomeUncertainty::default(),
            )
        };

        InterventionCalibrationEvidence {
            model_version: self.model_version.clone(),
            context: context.clone(),
            source,
            intervention_context_samples: exact_samples,
            strategy_fallback_samples: strategy_samples,
            action_fallback_samples: action_samples,
            intervention_context_moments: exact,
            strategy_fallback_moments: strategy,
            action_fallback_moments: action,
            prior,
            calibrated,
            uncertainty,
        }
    }

    pub fn observe_intervention(
        &mut self,
        context: InterventionPredictionContext,
        outcome: ObservedMusicalOutcome,
    ) -> Result<(), AdaptiveLearningError> {
        if !outcome_is_finite(outcome) {
            return Err(AdaptiveLearningError::NonFiniteOutcome);
        }
        if !context.descriptor.is_compatible() || context.base.action != context.descriptor.action {
            return Err(AdaptiveLearningError::IncompatibleIntervention);
        }
        if let Some(entry) = self
            .intervention_contexts
            .iter_mut()
            .find(|entry| entry.context == context)
        {
            entry.outcomes.observe(outcome);
        } else {
            let mut outcomes = OutcomeMoments::default();
            outcomes.observe(outcome);
            self.intervention_contexts
                .push(InterventionOutcomeCalibration {
                    context: context.clone(),
                    outcomes,
                });
        }
        if let Some(entry) = self.strategy_fallbacks.iter_mut().find(|entry| {
            entry.action == context.base.action && entry.strategy == context.descriptor.strategy
        }) {
            entry.outcomes.observe(outcome);
        } else {
            let mut outcomes = OutcomeMoments::default();
            outcomes.observe(outcome);
            self.strategy_fallbacks.push(StrategyOutcomeCalibration {
                action: context.base.action,
                strategy: context.descriptor.strategy,
                outcomes,
            });
        }
        self.observe_action(context.base.action, outcome);
        Ok(())
    }

    pub fn observe(
        &mut self,
        context: PredictionContext,
        outcome: ObservedMusicalOutcome,
    ) -> Result<(), AdaptiveLearningError> {
        if !outcome_is_finite(outcome) {
            return Err(AdaptiveLearningError::NonFiniteOutcome);
        }
        if let Some(entry) = self
            .exact_contexts
            .iter_mut()
            .find(|entry| entry.context == context)
        {
            entry.outcomes.observe(outcome);
        } else {
            let mut outcomes = OutcomeMoments::default();
            outcomes.observe(outcome);
            self.exact_contexts.push(ContextOutcomeCalibration {
                context: context.clone(),
                outcomes,
            });
        }
        self.observe_action(context.action, outcome);
        Ok(())
    }

    fn observe_action(&mut self, action: SymbolicAction, outcome: ObservedMusicalOutcome) {
        if let Some(entry) = self
            .action_fallbacks
            .iter_mut()
            .find(|entry| entry.action == action)
        {
            entry.outcomes.observe(outcome);
        } else {
            let mut outcomes = OutcomeMoments::default();
            outcomes.observe(outcome);
            self.action_fallbacks
                .push(ActionOutcomeCalibration { action, outcomes });
        }
    }
}

/// Replace a trace's fixed action prior with the model's calibrated prediction.
pub fn calibrate_trace(
    model: &AdaptiveOutcomeModel,
    context: PredictionContext,
    trace: &mut CognitiveDecisionTrace,
) -> PredictionCalibrationEvidence {
    debug_assert_eq!(context.action, trace.proposal.action);
    let evidence = model.predict(&context);
    trace.predicted_outcome = evidence.calibrated;
    evidence
}

/// Calibrate a recipe decision and retain the model evidence beside it.
pub fn calibrate_decision(
    model: &AdaptiveOutcomeModel,
    context: PredictionContext,
    decision: &mut RecipeDecision,
) -> PredictionCalibrationEvidence {
    let evidence = calibrate_trace(model, context, &mut decision.trace);
    decision.prediction_calibration = Some(evidence.clone());
    evidence
}

/// Feed a completed decision's observed outcome back into the calibration
/// model. Rejected or unmeasured alternatives cannot train the model.
pub fn learn_from_decision(
    model: &mut AdaptiveOutcomeModel,
    decision: &RecipeDecision,
) -> Result<(), AdaptiveLearningError> {
    let calibration = decision
        .prediction_calibration
        .as_ref()
        .ok_or(AdaptiveLearningError::MissingCalibrationEvidence)?;
    let observed = decision
        .observed_outcome
        .ok_or(AdaptiveLearningError::MissingObservedOutcome)?;
    model.observe(calibration.context.clone(), observed)
}

fn shrink(
    prior: PredictedMusicalOutcome,
    observed_mean: PredictedMusicalOutcome,
    samples: u64,
    prior_strength: f32,
) -> PredictedMusicalOutcome {
    let weight = samples as f32 / (samples as f32 + prior_strength.max(0.0));
    blend(prior, observed_mean, weight)
}

fn blend(
    left: PredictedMusicalOutcome,
    right: PredictedMusicalOutcome,
    right_weight: f32,
) -> PredictedMusicalOutcome {
    let weight = right_weight.clamp(0.0, 1.0);
    let mix = |a: f32, b: f32| a * (1.0 - weight) + b * weight;
    PredictedMusicalOutcome {
        tension_delta: mix(left.tension_delta, right.tension_delta),
        density_delta: mix(left.density_delta, right.density_delta),
        familiarity_delta: mix(left.familiarity_delta, right.familiarity_delta),
        tonal_displacement_delta: mix(
            left.tonal_displacement_delta,
            right.tonal_displacement_delta,
        ),
    }
}

fn moments_are_valid(moments: OutcomeMoments) -> bool {
    let channels = [
        moments.tension,
        moments.density,
        moments.familiarity,
        moments.tonal_displacement,
    ];
    let samples = moments.samples();
    channels.iter().all(|channel| {
        channel.samples == samples
            && channel.mean.is_finite()
            && channel.m2.is_finite()
            && channel.m2 >= -f64::EPSILON * 16.0
    })
}

fn outcome_is_finite(outcome: ObservedMusicalOutcome) -> bool {
    outcome.tension_delta.is_finite()
        && outcome.density_delta.is_finite()
        && outcome.familiarity_delta.is_finite()
        && outcome.tonal_displacement_delta.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(section: CognitiveSection) -> PredictionContext {
        PredictionContext::new(
            SymbolicAction::IncreaseDensity,
            section,
            "Sonata",
            "Sonata",
            4,
            TextureBand::Chamber,
        )
    }

    fn observed(density_delta: f32) -> ObservedMusicalOutcome {
        ObservedMusicalOutcome {
            tension_delta: 0.05,
            density_delta,
            familiarity_delta: 0.0,
            tonal_displacement_delta: 0.0,
        }
    }

    #[test]
    fn unseen_context_uses_the_hand_authored_prior() {
        let model = AdaptiveOutcomeModel::default();
        let prediction = model.predict(&context(CognitiveSection::Development));
        assert_eq!(
            prediction.source,
            PredictionEvidenceSource::HandAuthoredPrior
        );
        assert_eq!(prediction.calibrated, prediction.prior);
    }

    #[test]
    fn action_fallback_learns_before_exact_context_is_mature() {
        let mut model = AdaptiveOutcomeModel::default();
        model
            .observe(context(CognitiveSection::Development), observed(0.10))
            .unwrap();
        let prediction = model.predict(&context(CognitiveSection::Recapitulation));
        assert_eq!(prediction.source, PredictionEvidenceSource::ActionFallback);
        assert!(prediction.calibrated.density_delta < prediction.prior.density_delta);
    }

    #[test]
    fn mature_exact_context_overrides_action_fallback_with_shrinkage() {
        let mut model = AdaptiveOutcomeModel::default();
        let context = context(CognitiveSection::Development);
        for _ in 0..4 {
            model.observe(context.clone(), observed(0.08)).unwrap();
        }
        let prediction = model.predict(&context);
        assert_eq!(prediction.source, PredictionEvidenceSource::ExactContext);
        assert_eq!(prediction.exact_context_samples, 4);
        assert!(prediction.calibrated.density_delta < 0.35);
        assert!(prediction.calibrated.density_delta > 0.08);
    }

    #[test]
    fn incompatible_persisted_models_are_detected() {
        let mut model = AdaptiveOutcomeModel::default();
        model.model_version = "unknown-model".into();
        assert!(!model.is_compatible());
        model.model_version = ADAPTIVE_OUTCOME_MODEL_VERSION.into();
        model.prior_strength = f32::NAN;
        assert!(!model.is_compatible());
    }

    #[test]
    fn non_finite_measurements_never_train_the_model() {
        let mut model = AdaptiveOutcomeModel::default();
        let mut invalid = observed(0.1);
        invalid.tension_delta = f32::NAN;
        assert_eq!(
            model.observe(context(CognitiveSection::Development), invalid),
            Err(AdaptiveLearningError::NonFiniteOutcome)
        );
        assert!(model.exact_contexts.is_empty());
    }

    fn intervention_context(strategy: InterventionStrategy) -> InterventionPredictionContext {
        let base = PredictionContext::new(
            SymbolicAction::ReturnOpeningMaterial,
            CognitiveSection::Recapitulation,
            "Sonata",
            "Sonata",
            4,
            TextureBand::Chamber,
        );
        let descriptor = InterventionDescriptor::new(
            SymbolicAction::ReturnOpeningMaterial,
            strategy,
            CognitiveSection::Exposition,
            CognitiveSection::Recapitulation,
            crate::intervention::ObligationClass::ReturnMotif,
            0,
            1.0,
            0.5,
            0.5,
            0.5,
            0.5,
            8,
            80,
        );
        InterventionPredictionContext::new(base, descriptor)
    }

    #[test]
    fn parameterized_learning_falls_back_by_strategy_before_exact_context() {
        let mut model = AdaptiveOutcomeModel::default();
        let literal = intervention_context(InterventionStrategy::Literal);
        model
            .observe_intervention(literal.clone(), observed(0.12))
            .unwrap();
        let mut related = literal;
        related.descriptor.baseline_density_bucket = 12;
        let prediction = model.predict_intervention(&related);
        assert_eq!(
            prediction.source,
            PredictionEvidenceSource::StrategyFallback
        );
        assert_eq!(prediction.strategy_fallback_samples, 1);
    }

    #[test]
    fn legacy_models_upgrade_without_inventing_parameterized_evidence() {
        let mut model = AdaptiveOutcomeModel::default();
        model.model_version = LEGACY_ADAPTIVE_OUTCOME_MODEL_VERSION.into();
        model
            .observe(context(CognitiveSection::Development), observed(0.1))
            .unwrap();
        assert!(model.upgrade_legacy());
        assert_eq!(model.model_version, ADAPTIVE_OUTCOME_MODEL_VERSION);
        assert!(model.intervention_contexts.is_empty());
        assert!(model.strategy_fallbacks.is_empty());
        assert_eq!(model.action_fallbacks[0].outcomes.samples(), 1);
    }
}
