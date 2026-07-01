// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC encoder for gravcraft state.

use crate::types::GravcraftState;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

/// Encodes gravcraft state as a 16,384D hypervector.
pub struct GravcraftHdcEncoder {
    bases: Vec<ContinuousHV>,
    num_channels: usize,
}

impl GravcraftHdcEncoder {
    pub fn new(genesis: &GenesisSeed) -> Self {
        let num_channels = 16;
        let bases = (0..num_channels)
            .map(|i| genesis.hv(&format!("gravcraft::encoder::{}", i), HDC_DIMENSION))
            .collect();
        Self {
            bases,
            num_channels,
        }
    }

    /// Encode a gravcraft state as a ContinuousHV.
    pub fn encode(&self, state: &GravcraftState) -> ContinuousHV {
        let channels = state.to_channels();
        assert!(channels.len() <= self.num_channels);

        // Normalize channels to [-1, 1] range using tanh
        let weights: Vec<f32> = channels.iter().map(|&c| c.tanh()).collect();

        ContinuousHV::encode_weighted(&self.bases[..weights.len()], &weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_dimension() {
        let genesis = GenesisSeed::from_phrase("gravcraft encoder test");
        let encoder = GravcraftHdcEncoder::new(&genesis);
        let state = GravcraftState::default();
        let hv = encoder.encode(&state);
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_encoder_deterministic() {
        let genesis = GenesisSeed::from_phrase("deterministic");
        let encoder = GravcraftHdcEncoder::new(&genesis);
        // Use non-zero state to avoid zero-vector edge case
        let mut state = GravcraftState::default();
        state.position = [1.0, 2.0, 3.0];
        state.velocity = [0.1, 0.2, 0.3];
        state.metric_perturbation = 0.05;
        let hv1 = encoder.encode(&state);
        let hv2 = encoder.encode(&state);
        assert!(
            hv1.similarity(&hv2) > 0.999,
            "Same state should produce identical HV: sim={}",
            hv1.similarity(&hv2)
        );
    }
}
