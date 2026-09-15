// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Role-invariant cross-dimensional context for Holographic Liquid State.
//!
//! Exact equivariance to the full real bipolar role group places a strong
//! restriction on linear state mixing: any linear map commuting with every
//! independent sign-flip role must be diagonal. Cross-dimensional interaction
//! therefore has to enter through a role-invariant channel (or the symmetry has
//! to be weakened explicitly).
//!
//! `InvariantContextMixer` takes the first path. It gathers magnitudes from
//! cyclically offset coordinates and produces a context field:
//!
//! ```text
//! c_i = (1/K) * sum_k w[k,i] * (|h_(i+offset_k)| + |x_(i+offset_k)|) / 2
//! ```
//!
//! Since `|r ⊙ h| = |h|` for any bipolar role `r`,
//!
//! ```text
//! C(r ⊙ h, r ⊙ x) = C(h, x)
//! ```
//!
//! exactly in real arithmetic. The context can then modulate a signed local
//! channel without destroying binding equivariance.

use crate::continuous_hv::{ContinuousHV, UnitaryRole};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextMixerError {
    ZeroDimension,
    NoOffsets,
    DimensionMismatch { expected: usize, actual: usize },
}

impl fmt::Display for ContextMixerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "context mixer dimension must be non-zero"),
            Self::NoOffsets => write!(f, "context mixer requires at least one offset"),
            Self::DimensionMismatch { expected, actual } => write!(
                f,
                "context mixer dimension mismatch: expected {expected}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for ContextMixerError {}

/// O(KD) magnitude-context mixer with K cyclic receptive-field offsets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvariantContextMixer {
    dim: usize,
    offsets: Vec<usize>,
    weights: Vec<ContinuousHV>,
}

impl InvariantContextMixer {
    pub fn try_new(dim: usize, offsets: Vec<usize>, seed: u64) -> Result<Self, ContextMixerError> {
        if dim == 0 {
            return Err(ContextMixerError::ZeroDimension);
        }
        if offsets.is_empty() {
            return Err(ContextMixerError::NoOffsets);
        }

        let offsets: Vec<usize> = offsets.into_iter().map(|offset| offset % dim).collect();
        let weights = offsets
            .iter()
            .enumerate()
            .map(|(index, _)| {
                ContinuousHV::new_random(dim, seed.wrapping_add(index as u64 + 1))
            })
            .collect();

        Ok(Self {
            dim,
            offsets,
            weights,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn fan_in(&self) -> usize {
        self.offsets.len()
    }

    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// Compute role-invariant cross-dimensional context.
    pub fn mix(
        &self,
        state: &ContinuousHV,
        input: &ContinuousHV,
    ) -> Result<ContinuousHV, ContextMixerError> {
        self.check_dim(state.dim())?;
        self.check_dim(input.dim())?;

        let inv_k = 1.0 / self.offsets.len() as f32;
        let mut context = vec![0.0_f32; self.dim];

        for (i, value) in context.iter_mut().enumerate() {
            let mut sum = 0.0_f32;
            for (kernel, &offset) in self.offsets.iter().enumerate() {
                let source = (i + offset) % self.dim;
                let magnitude = 0.5 * (state.values[source].abs() + input.values[source].abs());
                sum += self.weights[kernel].values[i] * magnitude;
            }
            *value = sum * inv_k;
        }

        Ok(ContinuousHV::from_values(context))
    }

    /// Directly measure the maximum componentwise violation of role invariance.
    pub fn role_invariance_error(
        &self,
        role: &UnitaryRole,
        state: &ContinuousHV,
        input: &ContinuousHV,
    ) -> Result<f32, ContextMixerError> {
        self.check_dim(role.dim())?;
        let direct = self.mix(state, input)?;
        let transformed = self.mix(&role.bind(state), &role.bind(input))?;

        Ok(direct
            .values
            .iter()
            .zip(transformed.values.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max))
    }

    fn check_dim(&self, actual: usize) -> Result<(), ContextMixerError> {
        if actual == self.dim {
            Ok(())
        } else {
            Err(ContextMixerError::DimensionMismatch {
                expected: self.dim,
                actual,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_is_exactly_role_invariant() {
        let mixer = InvariantContextMixer::try_new(512, vec![1, 7, 31], 42).unwrap();
        let state = ContinuousHV::new_random(512, 43);
        let input = ContinuousHV::new_random(512, 44);
        let role = UnitaryRole::new(512, 45);
        let error = mixer
            .role_invariance_error(&role, &state, &input)
            .unwrap();
        assert_eq!(error, 0.0);
    }

    #[test]
    fn remote_magnitude_changes_local_context() {
        let mixer = InvariantContextMixer::try_new(16, vec![1], 7).unwrap();
        let state_a = ContinuousHV::new(16);
        let input = ContinuousHV::new(16);
        let mut state_b = state_a.clone();

        // For target i=0 and offset=1, source coordinate 1 is in the receptive field.
        state_b.values[1] = 1.0;
        let a = mixer.mix(&state_a, &input).unwrap();
        let b = mixer.mix(&state_b, &input).unwrap();
        assert_ne!(a.values[0], b.values[0]);
    }

    #[test]
    fn context_is_deterministic() {
        let a = InvariantContextMixer::try_new(128, vec![1, 3, 9], 99).unwrap();
        let b = InvariantContextMixer::try_new(128, vec![1, 3, 9], 99).unwrap();
        let state = ContinuousHV::new_random(128, 100);
        let input = ContinuousHV::new_random(128, 101);
        assert_eq!(a.mix(&state, &input).unwrap(), b.mix(&state, &input).unwrap());
    }
}
