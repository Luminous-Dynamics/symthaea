// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC encoding adapters for psychological stimuli.
//!
//! Each adapter converts a specific stimulus type into `ContinuousHV`
//! representations suitable for Symthaea's cognitive subsystems.

pub mod reward;
pub mod scenario;
pub mod semantic;
pub mod sequence;
pub mod spatial;

use symthaea_core::hdc::ContinuousHV;

/// Encode arbitrary stimuli into HDC hypervectors.
pub trait StimulusAdapter {
    /// The stimulus type this adapter encodes.
    type Stimulus;

    /// Encode a single stimulus into a `ContinuousHV` of given dimension.
    fn encode(&self, stimulus: &Self::Stimulus, dim: usize) -> ContinuousHV;

    /// Encode a sequence of stimuli, preserving order information.
    fn encode_sequence(&self, stimuli: &[Self::Stimulus], dim: usize) -> Vec<ContinuousHV> {
        stimuli.iter().map(|s| self.encode(s, dim)).collect()
    }
}
