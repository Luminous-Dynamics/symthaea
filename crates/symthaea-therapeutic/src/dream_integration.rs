// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Therapeutic Dream Integration
//!
//! Implements `DreamableAction` for therapeutic interventions, enabling the dream
//! engine to generate therapeutic counterfactuals: "What if the client had used
//! cognitive reappraisal instead of avoidance?"
//!
//! Wisdom fragments from therapeutic dreaming feed back into the RegulationEngine's
//! strategy selection via `incorporate_dream_wisdom()`.
//!
//! Science: Walker (2009) — sleep-dependent memory consolidation improves
//! emotional regulation. Stickgold & Walker (2013) — REM sleep facilitates
//! creative problem-solving in affect-laden contexts.

use crate::affect_regulation::{NeuromodDelta, RegulationStrategy};
use serde::{Deserialize, Serialize};
use symthaea_dream::DreamableAction;

/// A therapeutic action that can be dreamed about.
///
/// Represents a regulation strategy applied at a specific intensity with
/// its resulting neuromodulator delta. The dream engine can perturb the
/// strategy ordinal and intensity to explore counterfactual therapeutic paths.
///
/// # HDC encoding
///
/// The action encodes as a float vector:
/// `[strategy_ordinal, intensity, delta.dopamine, ..., delta.adenosine]`
/// (10 dimensions total), enabling the dream engine's similarity-based
/// world model to generalize across related interventions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TherapeuticAction {
    /// The regulation strategy applied (0-6 ordinal).
    pub strategy: RegulationStrategy,
    /// Intervention intensity (0.0-1.0), typically proportional to distress.
    pub intensity: f32,
    /// Resulting neuromodulator delta from `apply_strategy()`.
    pub neuromod_delta: NeuromodDelta,
    /// Client distress at time of action.
    pub client_distress: f32,
    /// Alliance quality at time of action.
    pub alliance_quality: f32,
}

impl TherapeuticAction {
    /// Create a new therapeutic action.
    pub fn new(
        strategy: RegulationStrategy,
        intensity: f32,
        neuromod_delta: NeuromodDelta,
        client_distress: f32,
        alliance_quality: f32,
    ) -> Self {
        Self {
            strategy,
            intensity,
            neuromod_delta,
            client_distress,
            alliance_quality,
        }
    }

    /// Encode this action as a float vector for the dream engine.
    fn to_vec(&self) -> Vec<f32> {
        vec![
            self.strategy_ordinal() as f32,
            self.intensity,
            self.neuromod_delta.dopamine,
            self.neuromod_delta.noradrenaline,
            self.neuromod_delta.serotonin,
            self.neuromod_delta.acetylcholine,
            self.neuromod_delta.gaba,
            self.neuromod_delta.oxytocin,
            self.neuromod_delta.glutamate,
            self.neuromod_delta.adenosine,
        ]
    }

    /// Strategy ordinal for dream engine compatibility.
    pub fn strategy_ordinal(&self) -> u8 {
        match self.strategy {
            RegulationStrategy::CognitiveReappraisal => 0,
            RegulationStrategy::DistressTolerance => 1,
            RegulationStrategy::Grounding => 2,
            RegulationStrategy::Defusion => 3,
            RegulationStrategy::Validation => 4,
            RegulationStrategy::Containment => 5,
            RegulationStrategy::ExposurePrep => 6,
        }
    }

    /// Reconstruct strategy from ordinal.
    fn strategy_from_ordinal(ordinal: u8) -> RegulationStrategy {
        match ordinal % 7 {
            0 => RegulationStrategy::CognitiveReappraisal,
            1 => RegulationStrategy::DistressTolerance,
            2 => RegulationStrategy::Grounding,
            3 => RegulationStrategy::Defusion,
            4 => RegulationStrategy::Validation,
            5 => RegulationStrategy::Containment,
            _ => RegulationStrategy::ExposurePrep,
        }
    }
}

impl DreamableAction for TherapeuticAction {
    /// Perturb the therapeutic action for counterfactual dreaming.
    ///
    /// - With 60% probability: switches to a different strategy (the core
    ///   counterfactual: "what if we had used a different approach?")
    /// - Always: perturbs intensity by ±20%
    /// - Recomputes neuromod delta for the new strategy/intensity
    fn perturb(&self, seed: u64) -> Self {
        use blake3::Hasher;

        let mut hasher = Hasher::new();
        hasher.update(&seed.to_le_bytes());
        hasher.update(&[self.strategy_ordinal()]);
        let hash = hasher.finalize();
        let bytes = hash.as_bytes();

        // Strategy perturbation: 60% chance of switching
        let switch = bytes[0] as f32 / 255.0 > 0.4;
        let new_strategy = if switch {
            let ordinal = (bytes[1] % 7) as u8;
            Self::strategy_from_ordinal(ordinal)
        } else {
            self.strategy
        };

        // Intensity perturbation: ±20%
        let intensity_noise = (bytes[2] as f32 / 255.0 * 2.0 - 1.0) * 0.2;
        let new_intensity = (self.intensity + intensity_noise).clamp(0.1, 1.0);

        // Recompute neuromod delta for the new strategy/intensity
        let mut engine = crate::affect_regulation::RegulationEngine::new();
        let new_delta = engine.apply_strategy(new_strategy, new_intensity);

        Self {
            strategy: new_strategy,
            intensity: new_intensity,
            neuromod_delta: new_delta,
            client_distress: self.client_distress,
            alliance_quality: self.alliance_quality,
        }
    }

    /// Predict outcome: simple model mapping neuromod delta to affect trajectory.
    ///
    /// The "state" is a float vector representing current client affect.
    /// Outcome is the predicted affect after applying this strategy's neuromod delta.
    ///
    /// Model: each neuromod transmitter nudges the state toward its target affect.
    /// 5-HT ↑ → valence ↑ (mood improvement)
    /// DA ↑ → motivation ↑
    /// Oxytocin ↑ → social bond ↑
    /// GABA ↑ → arousal ↓ (calming)
    /// NE ↓ → arousal ↓
    fn predict_outcome(&self, state: &[f32]) -> Vec<f32> {
        let d = &self.neuromod_delta;
        let scale = self.intensity * 0.3; // Therapeutic interventions are gentle

        state
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let nudge = match i % 6 {
                    0 => d.serotonin * scale,            // valence
                    1 => d.dopamine * scale,             // motivation
                    2 => d.oxytocin * scale,             // social bond
                    3 => -d.gaba * scale * 0.5,          // arousal ↓
                    4 => -d.noradrenaline * scale * 0.3, // threat ↓
                    _ => d.acetylcholine * scale * 0.2,  // cognitive clarity
                };
                (s + nudge).clamp(-1.0, 1.0)
            })
            .collect()
    }

    /// Magnitude: proportional to intensity and total neuromod delta.
    fn magnitude(&self) -> f32 {
        let delta_sum = self.to_vec()[2..].iter().map(|d| d.abs()).sum::<f32>();
        self.intensity * delta_sum
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::affect_regulation::RegulationEngine;

    fn make_action(strategy: RegulationStrategy, distress: f32) -> TherapeuticAction {
        let mut engine = RegulationEngine::new();
        let delta = engine.apply_strategy(strategy, distress);
        TherapeuticAction::new(strategy, distress, delta, distress, 0.5)
    }

    #[test]
    fn test_therapeutic_action_creation() {
        let action = make_action(RegulationStrategy::Validation, 0.6);
        assert_eq!(action.strategy_ordinal(), 4);
        assert!(action.neuromod_delta.oxytocin > 0.0);
    }

    #[test]
    fn test_to_vec_length() {
        let action = make_action(RegulationStrategy::Grounding, 0.5);
        let v = action.to_vec();
        assert_eq!(v.len(), 10); // ordinal + intensity + 8 transmitters
    }

    #[test]
    fn test_perturb_changes_action() {
        let action = make_action(RegulationStrategy::Validation, 0.5);
        // Try multiple seeds — at least one should change the strategy
        let mut changed = false;
        for seed in 0..20 {
            let perturbed = action.perturb(seed);
            if perturbed.strategy != action.strategy
                || (perturbed.intensity - action.intensity).abs() > 0.01
            {
                changed = true;
                break;
            }
        }
        assert!(changed, "perturb should change the action for some seeds");
    }

    #[test]
    fn test_perturb_intensity_bounded() {
        let action = make_action(RegulationStrategy::Grounding, 0.9);
        for seed in 0..50 {
            let perturbed = action.perturb(seed);
            assert!(
                perturbed.intensity >= 0.1 && perturbed.intensity <= 1.0,
                "intensity {} out of bounds",
                perturbed.intensity,
            );
        }
    }

    #[test]
    fn test_perturb_deterministic() {
        let action = make_action(RegulationStrategy::Defusion, 0.5);
        let a = action.perturb(42);
        let b = action.perturb(42);
        assert_eq!(a.strategy, b.strategy);
        assert!((a.intensity - b.intensity).abs() < f32::EPSILON);
    }

    #[test]
    fn test_predict_outcome_shifts_state() {
        let action = make_action(RegulationStrategy::Validation, 0.6);
        let state = vec![0.0; 12];
        let outcome = action.predict_outcome(&state);
        assert_eq!(outcome.len(), 12);
        // Validation boosts oxytocin → index 2 (social bond) should increase
        assert!(
            outcome[2] > 0.0,
            "Validation should boost social bond dimension"
        );
    }

    #[test]
    fn test_predict_outcome_grounding_calms() {
        let action = make_action(RegulationStrategy::Grounding, 0.7);
        let state = vec![0.5; 6];
        let outcome = action.predict_outcome(&state);
        // Grounding: GABA↑ → arousal↓ (index 3 should decrease)
        assert!(
            outcome[3] < 0.5,
            "Grounding should reduce arousal: {}",
            outcome[3],
        );
    }

    #[test]
    fn test_magnitude_scales_with_intensity() {
        let low = make_action(RegulationStrategy::Validation, 0.2);
        let high = make_action(RegulationStrategy::Validation, 0.8);
        assert!(
            high.magnitude() > low.magnitude(),
            "Higher intensity should produce higher magnitude",
        );
    }

    #[test]
    fn test_strategy_ordinal_roundtrip() {
        for strategy in RegulationStrategy::ALL {
            let action = make_action(strategy, 0.5);
            let ordinal = action.strategy_ordinal();
            let recovered = TherapeuticAction::strategy_from_ordinal(ordinal);
            assert_eq!(
                strategy, recovered,
                "ordinal roundtrip failed for {strategy:?}"
            );
        }
    }

    #[test]
    fn test_all_strategies_produce_nonzero_magnitude() {
        for strategy in RegulationStrategy::ALL {
            let action = make_action(strategy, 0.5);
            assert!(
                action.magnitude() > 0.0,
                "{strategy:?} should produce nonzero magnitude",
            );
        }
    }

    #[test]
    fn test_dreamable_trait_bounds() {
        // Verify TherapeuticAction satisfies DreamableAction trait bounds
        fn assert_dreamable<T: DreamableAction>() {}
        assert_dreamable::<TherapeuticAction>();
    }

    #[test]
    fn test_serialization_roundtrip() {
        let action = make_action(RegulationStrategy::CognitiveReappraisal, 0.6);
        let json = serde_json::to_string(&action).unwrap();
        let recovered: TherapeuticAction = serde_json::from_str(&json).unwrap();
        assert_eq!(recovered.strategy, action.strategy);
        assert!((recovered.intensity - action.intensity).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dream_counterfactual_scenario() {
        // Simulate what the dream engine does: record an event, perturb, compare
        let original = make_action(RegulationStrategy::DistressTolerance, 0.7);
        let state = vec![0.0, -0.5, 0.2, 0.8, 0.6, 0.3]; // distressed state

        let original_outcome = original.predict_outcome(&state);
        let counterfactual = original.perturb(42);
        let counterfactual_outcome = counterfactual.predict_outcome(&state);

        // Both outcomes should be bounded
        for v in &original_outcome {
            assert!((-1.0..=1.0).contains(v));
        }
        for v in &counterfactual_outcome {
            assert!((-1.0..=1.0).contains(v));
        }
    }
}
