// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! External action→sensation contingency runtime.
//!
//! This is a narrow adapter around the production [`SensorimotorEngine`]. It lets
//! callers provide the *actual* sensory consequence of an action instead of using
//! `EmbodiedConsciousnessAnalyzer::perform_action()`'s built-in synthetic sensation.
//!
//! The distinction matters for controlled contingency-reversal experiments: an
//! external environment must be able to change `A -> X` into `A -> Y` while the
//! action representation itself remains unchanged. This module adds no benchmark
//! scoring and grants no consciousness/evidence authority.

use super::embodied_cognition::{ActionPattern, SensoryPrediction, SensorimotorEngine};

/// A single externally supplied action/outcome learning step.
#[derive(Debug, Clone)]
pub struct ContingencyObservation {
    /// Prediction made *before* the actual outcome was revealed.
    pub prediction_before_outcome: Option<SensoryPrediction>,
    /// Actual externally supplied sensory consequence.
    pub actual_outcome: SensoryPrediction,
    /// Euclidean error over the three channels used by the production engine's
    /// own learning diagnostic (visual, tactile, proprioceptive), normalized
    /// by sqrt(3). `None` means no prior prediction existed yet.
    pub prediction_error: Option<f64>,
    /// Production engine's rolling average prediction error after learning.
    pub rolling_prediction_error: f64,
    /// Number of external outcomes incorporated by this runtime.
    pub observation_count: u64,
}

/// Production sensorimotor contingency learning exposed to an external world.
///
/// `enabled=false` is intentionally represented as absence of the engine rather
/// than a zero-valued dummy. That prevents an ablated embodiment path from
/// accidentally learning through a benchmark-only side channel.
pub struct ExternalContingencyRuntime {
    engine: Option<SensorimotorEngine>,
    observation_count: u64,
}

impl ExternalContingencyRuntime {
    pub fn new(enabled: bool, max_contingencies: usize) -> Self {
        Self {
            engine: enabled.then(|| SensorimotorEngine::new(max_contingencies)),
            observation_count: 0,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.engine.is_some()
    }

    /// Predict the sensory consequence of `action` without updating the model.
    pub fn predict(
        &self,
        action: &ActionPattern,
    ) -> Result<Option<SensoryPrediction>, &'static str> {
        let engine = self
            .engine
            .as_ref()
            .ok_or("embodied contingency runtime is disabled")?;
        Ok(engine.predict(action))
    }

    /// Reveal an externally generated sensory consequence and learn from it.
    ///
    /// The prediction is captured before the update so callers can score true
    /// prospective prediction rather than a post-hoc reconstruction.
    pub fn observe(
        &mut self,
        action: ActionPattern,
        actual_outcome: SensoryPrediction,
    ) -> Result<ContingencyObservation, &'static str> {
        let engine = self
            .engine
            .as_mut()
            .ok_or("embodied contingency runtime is disabled")?;

        let prediction_before_outcome = engine.predict(&action);
        let prediction_error = prediction_before_outcome
            .as_ref()
            .map(|pred| sensory_prediction_error(pred, &actual_outcome));

        engine.learn(
            action,
            actual_outcome.clone(),
            prediction_before_outcome.as_ref(),
        );
        self.observation_count += 1;

        Ok(ContingencyObservation {
            prediction_before_outcome,
            actual_outcome,
            prediction_error,
            rolling_prediction_error: engine.average_prediction_error(),
            observation_count: self.observation_count,
        })
    }
}

fn sensory_prediction_error(a: &SensoryPrediction, b: &SensoryPrediction) -> f64 {
    let sum_sq = (a.visual_change - b.visual_change).powi(2)
        + (a.tactile_expected - b.tactile_expected).powi(2)
        + (a.proprioceptive_change - b.proprioceptive_change).powi(2);
    (sum_sq / 3.0).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::embodied_cognition::{BodyPart, MovementType};
    use crate::hdc::binary_hv::BinaryHV;

    fn action(seed: u64, movement_type: MovementType) -> ActionPattern {
        ActionPattern {
            involved_parts: vec![BodyPart::RightHand],
            movement_type,
            intensity: 0.5,
            duration: 1,
            encoding: BinaryHV::random(seed),
        }
    }

    fn sensation(seed: u64, visual: f64, tactile: f64, proprio: f64) -> SensoryPrediction {
        SensoryPrediction {
            visual_change: visual,
            tactile_expected: tactile,
            proprioceptive_change: proprio,
            auditory_expected: 0.0,
            encoding: BinaryHV::random(seed),
        }
    }

    #[test]
    fn disabled_runtime_cannot_predict_or_learn() {
        let mut runtime = ExternalContingencyRuntime::new(false, 16);
        let a = action(1, MovementType::Reach);
        let x = sensation(11, 0.2, 0.8, 0.1);
        assert!(runtime.predict(&a).is_err());
        assert!(runtime.observe(a, x).is_err());
    }

    #[test]
    fn external_outcome_reversal_changes_prediction_without_changing_action() {
        let mut runtime = ExternalContingencyRuntime::new(true, 16);
        let a = action(1, MovementType::Reach);
        let x = sensation(11, 0.1, 0.9, 0.2);
        let y = sensation(12, 0.9, 0.1, 0.8);

        for _ in 0..24 {
            runtime.observe(a.clone(), x.clone()).unwrap();
        }
        let before = runtime.predict(&a).unwrap().expect("prediction after acquisition");
        assert!(sensory_prediction_error(&before, &x) < sensory_prediction_error(&before, &y));

        for _ in 0..48 {
            runtime.observe(a.clone(), y.clone()).unwrap();
        }
        let after = runtime.predict(&a).unwrap().expect("prediction after reversal");
        assert!(sensory_prediction_error(&after, &y) < sensory_prediction_error(&after, &x));
    }
}
