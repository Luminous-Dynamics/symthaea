// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Thin FEP-crate integration for the scientific world-model decision layer.

use symthaea_core::scientific_active_inference::{
    assess_test, ScientificActiveInferenceAssessment, ScientificActiveInferenceError,
    ScientificTest, ScientificTrajectoryForecast,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScientificFepPlanner {
    pub pragmatic_weight: f64,
    pub epistemic_weight: f64,
}

impl Default for ScientificFepPlanner {
    fn default() -> Self {
        Self { pragmatic_weight: 1.0, epistemic_weight: 1.0 }
    }
}

impl ScientificFepPlanner {
    pub const fn new(pragmatic_weight: f64, epistemic_weight: f64) -> Self {
        Self { pragmatic_weight, epistemic_weight }
    }

    pub fn assess(
        &self,
        forecasts: &[ScientificTrajectoryForecast],
        test: &ScientificTest,
    ) -> Result<ScientificActiveInferenceAssessment, ScientificActiveInferenceError> {
        if !self.pragmatic_weight.is_finite() || self.pragmatic_weight < 0.0
            || !self.epistemic_weight.is_finite() || self.epistemic_weight < 0.0
        {
            return Err(ScientificActiveInferenceError::InvalidWeight);
        }
        let mut assessment = assess_test(forecasts, test)?;
        assessment.expected_free_energy =
            self.pragmatic_weight * assessment.pragmatic_risk
                - self.epistemic_weight * assessment.epistemic_value_bits;
        Ok(assessment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};
    use symthaea_core::scientific_active_inference::ScientificOutcome;

    fn forecasts() -> Vec<ScientificTrajectoryForecast> {
        vec![
            ScientificTrajectoryForecast {
                model_id: "a".into(), lineage: "la".into(), prior: 0.5,
                horizon_seconds: 1.0, predicted_state: ContinuousHV::random(HDC_DIMENSION, 1),
                outcome_probabilities: vec![1.0, 0.0],
            },
            ScientificTrajectoryForecast {
                model_id: "b".into(), lineage: "lb".into(), prior: 0.5,
                horizon_seconds: 1.0, predicted_state: ContinuousHV::random(HDC_DIMENSION, 2),
                outcome_probabilities: vec![0.0, 1.0],
            },
        ]
    }

    #[test]
    fn weights_change_the_fep_decision_variable() {
        let test = ScientificTest {
            id: "test".into(), cost: 1.0,
            outcomes: vec![
                ScientificOutcome { id: "a".into(), utility: 1.0 },
                ScientificOutcome { id: "b".into(), utility: 0.0 },
            ],
        };
        let default = ScientificFepPlanner::default().assess(&forecasts(), &test).unwrap();
        let epistemic = ScientificFepPlanner::new(0.0, 1.0).assess(&forecasts(), &test).unwrap();
        assert!(epistemic.expected_free_energy < default.expected_free_energy);
    }
}
