// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Observation-only helpers for classic CfC.
//!
//! This is a deliberately conservative containment surface for TEMPORAL-STATE-000.
//! `CfCNetwork::predict_forward()` advances evolution state. Until the versioned
//! exact inference snapshot is implemented and qualified, callers that require a
//! read-only prediction can run the probe on a detached clone instead.
//!
//! The clone cost is intentional and must remain visible in performance/resource
//! accounting. This API is a correctness bridge, not an efficiency claim.

use ndarray::Array1;

use super::CfCNetwork;

impl CfCNetwork {
    /// Predict from the current classic-CfC state without mutating `self`.
    ///
    /// This clones the complete network and runs the ordinary mutating
    /// [`CfCNetwork::predict_forward`] on the detached copy. It therefore gives
    /// callers an observational-purity guarantee without relying on the legacy
    /// projected `read_state()` / `inject()` pair, which is not an exact restore
    /// for multilayer CfC networks.
    ///
    /// # Cost
    ///
    /// This copies the network, including parameters and optimizer/adaptation
    /// bookkeeping present in `Clone`. Callers must account for that copy cost
    /// separately from inference when comparing architectures or execution modes.
    ///
    /// # Scope
    ///
    /// This does not implement an exact snapshot API, historical-start training,
    /// checkpoint continuation, or online-adaptation rollback. It is only a safe
    /// observational prediction bridge for the current network state.
    pub fn predict_forward_observational(
        &self,
        input: &Array1<f32>,
        horizon: f32,
    ) -> anyhow::Result<Array1<f32>> {
        let mut detached = self.clone();
        detached.predict_forward(input, horizon)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::cfc::CfCNetworkConfig;

    #[test]
    fn observational_prediction_preserves_all_live_layer_states() {
        let config = CfCNetworkConfig {
            input_dim: 4,
            hidden_dim: 4,
            num_layers: 2,
            output_dim: 2,
            ..CfCNetworkConfig::default()
        };
        let mut live = CfCNetwork::new(config);
        live.set_state(vec![
            Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]),
            Array1::from_vec(vec![0.5, 0.6, 0.7, 0.8]),
        ]);

        let before = live.state();
        let input = Array1::from_vec(vec![0.9, -0.4, 0.2, 0.7]);
        let prediction = live
            .predict_forward_observational(&input, 0.25)
            .expect("observational CfC prediction");

        assert_eq!(prediction.len(), 2);
        assert_eq!(live.state(), before);
    }
}
