// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Scientific world-model trajectories backed by Symthaea's CfC contract.
//!
//! This module consumes the existing TemporalPredictor trait rather than
//! implementing a second liquid-time-constant model.
//!
//! The scientific layer keeps symbolic/HDC state, continuous HDC dynamics, and
//! observations distinct. None of these quantities constitutes biological
//! evidence by itself.

use crate::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};
use crate::scientific_world_model::{ScientificHdcEncoder, ScientificState};
use crate::temporal::TemporalPredictor;
use serde::{Deserialize, Serialize};

/// A deterministic continuous projection of a scientific HDC state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScientificContinuousState {
    /// Source structural state.
    pub source: ScientificState,
    /// Continuous state supplied to a temporal predictor.
    pub vector: ContinuousHV,
}

/// One prediction made at a requested future horizon.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalPrediction {
    /// Requested horizon in seconds.
    pub horizon_seconds: f32,
    /// Predicted continuous state.
    pub state: ContinuousHV,
}

/// An observed trajectory point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalObservation {
    /// Elapsed time in seconds.
    pub time_seconds: f32,
    /// Observed continuous state.
    pub state: ContinuousHV,
}

/// Prediction/observation discrepancy at a matching time point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryResidual {
    /// Elapsed time in seconds.
    pub time_seconds: f32,
    /// Cosine similarity between prediction and observation.
    pub similarity: f32,
    /// One minus similarity; descriptive residual only.
    pub residual: f32,
}

/// Scientific trajectory adapter around an existing temporal predictor.
#[derive(Debug, Clone)]
pub struct ScientificTemporalAdapter<P> {
    predictor: P,
    encoder: ScientificHdcEncoder,
    seed: u64,
}

impl<P> ScientificTemporalAdapter<P>
where
    P: TemporalPredictor,
{
    /// Construct an adapter with the stable default representation seed.
    pub fn new(predictor: P) -> Self {
        Self::with_seed(predictor, 0x5343_4945_4E43_4557)
    }

    /// Construct an adapter with an explicit deterministic representation seed.
    pub fn with_seed(predictor: P, seed: u64) -> Self {
        Self {
            predictor,
            encoder: ScientificHdcEncoder::with_seed(seed),
            seed,
        }
    }

    /// Borrow the underlying predictor.
    pub fn predictor(&self) -> &P {
        &self.predictor
    }

    /// Mutably borrow the underlying predictor for observation/update calls.
    pub fn predictor_mut(&mut self) -> &mut P {
        &mut self.predictor
    }

    /// Project a scientific state into the continuous HDC space used by CfC.
    ///
    /// The projection is deterministic for a fixed encoder seed and state
    /// structure. It is a representation bridge, not a learned biological
    /// encoder.
    pub fn project(&self, state: &ScientificState) -> ScientificContinuousState {
        let mut continuous = Vec::with_capacity(state.relations.len() + 1);

        for relation in &state.relations {
            continuous.push(ContinuousHV::random(
                HDC_DIMENSION,
                self.continuous_seed(
                    "scientific.relation",
                    &format!("{}|{}|{}", relation.subject, relation.relation, relation.object),
                ),
            ));
        }

        if let Some(context) = &state.context {
            continuous.push(ContinuousHV::random(
                HDC_DIMENSION,
                self.continuous_seed("scientific.context", context),
            ));
        }

        if continuous.is_empty() {
            continuous.push(ContinuousHV::random(
                HDC_DIMENSION,
                self.continuous_seed("scientific.state", "entity-only"),
            ));
        }

        let refs: Vec<&ContinuousHV> = continuous.iter().collect();
        ScientificContinuousState {
            source: state.clone(),
            vector: ContinuousHV::bundle(&refs),
        }
    }

    /// Predict the scientific state representation at multiple horizons.
    pub fn predict(
        &self,
        state: &ScientificState,
        horizons_seconds: &[f32],
    ) -> Vec<TemporalPrediction> {
        let projected = self.project(state);
        horizons_seconds
            .iter()
            .copied()
            .filter(|h| h.is_finite() && *h > 0.0)
            .map(|horizon_seconds| TemporalPrediction {
                horizon_seconds,
                state: self.predictor.predict_at(&projected.vector, horizon_seconds),
            })
            .collect()
    }

    /// Observe a realized trajectory point and update the predictor.
    pub fn observe(&mut self, observation: &TemporalObservation, dt_seconds: f32) {
        self.predictor.observe(&observation.state, dt_seconds);
    }

    /// Compare predictions with observations at matching horizons.
    ///
    /// Residuals are descriptive diagnostics. They are not evidence-level
    /// qualification and do not promote a prediction into an observation.
    pub fn residuals(
        &self,
        predictions: &[TemporalPrediction],
        observations: &[TemporalObservation],
    ) -> Vec<TrajectoryResidual> {
        predictions
            .iter()
            .filter_map(|prediction| {
                observations
                    .iter()
                    .find(|observation| {
                        (observation.time_seconds - prediction.horizon_seconds).abs() < 1e-6
                    })
                    .map(|observation| {
                        let similarity = prediction.state.similarity(&observation.state);
                        TrajectoryResidual {
                            time_seconds: prediction.horizon_seconds,
                            similarity,
                            residual: 1.0 - similarity,
                        }
                    })
            })
            .collect()
    }

    fn continuous_seed(&self, namespace: &str, value: &str) -> u64 {
        let mut hash = self.seed ^ 0xcbf29ce484222325u64;
        for byte in namespace.bytes().chain(std::iter::once(0)).chain(value.bytes()) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    /// Expose the HDC encoder used for the source structural representation.
    pub fn encoder(&self) -> &ScientificHdcEncoder {
        &self.encoder
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scientific_world_model::ScientificRelation;

    struct TestPredictor;

    impl TemporalPredictor for TestPredictor {
        fn predict_at(&self, current_state: &ContinuousHV, horizon_seconds: f32) -> ContinuousHV {
            let scale = 1.0 / (1.0 + horizon_seconds);
            ContinuousHV::from_vec(
                current_state.values.iter().map(|value| value * scale).collect(),
            )
        }

        fn observe(&mut self, _state: &ContinuousHV, _dt_seconds: f32) {}
        fn domain(&self) -> &'static str { "test" }
        fn tau_base(&self) -> f32 { 1.0 }
    }

    fn state() -> ScientificState {
        ScientificHdcEncoder::new()
            .state(
                &[],
                &[ScientificRelation {
                    subject: "protease".into(),
                    relation: "cleaves".into(),
                    object: "target".into(),
                }],
                Some("cell"),
            )
            .unwrap()
    }

    #[test]
    fn projection_is_deterministic() {
        let adapter_a = ScientificTemporalAdapter::new(TestPredictor);
        let adapter_b = ScientificTemporalAdapter::new(TestPredictor);
        assert_eq!(adapter_a.project(&state()), adapter_b.project(&state()));
    }

    #[test]
    fn different_seeds_change_projection() {
        let a = ScientificTemporalAdapter::with_seed(TestPredictor, 1);
        let b = ScientificTemporalAdapter::with_seed(TestPredictor, 2);
        assert_ne!(a.project(&state()), b.project(&state()));
    }

    #[test]
    fn invalid_horizons_are_ignored() {
        let adapter = ScientificTemporalAdapter::new(TestPredictor);
        let predictions = adapter.predict(&state(), &[0.0, -1.0, f32::NAN, 1.0]);
        assert_eq!(predictions.len(), 1);
        assert_eq!(predictions[0].horizon_seconds, 1.0);
    }

    #[test]
    fn residual_is_zero_for_identical_prediction_and_observation() {
        let adapter = ScientificTemporalAdapter::new(TestPredictor);
        let predictions = adapter.predict(&state(), &[1.0]);
        let observations = vec![TemporalObservation {
            time_seconds: 1.0,
            state: predictions[0].state.clone(),
        }];
        let residuals = adapter.residuals(&predictions, &observations);
        assert_eq!(residuals.len(), 1);
        assert!(residuals[0].residual.abs() < f32::EPSILON);
    }
}
