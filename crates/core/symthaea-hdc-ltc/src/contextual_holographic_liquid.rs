// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Contextual Holographic Liquid State.
//!
//! This wrapper composes the theorem-bearing diagonal `HolographicLiquidCell`
//! with `InvariantContextMixer` without changing the diagonal cell itself.
//! Cross-dimensional magnitude context modulates the signed input channel:
//!
//! ```text
//! c       = C(|h|, |x|)
//! gain_i  = 1 + gamma * tanh(c_i)
//! x'_i    = gain_i * x_i
//! h'      = F(h, x', dt)
//! ```
//!
//! For any bipolar role `r`, `C(r⊙h, r⊙x) = C(h,x)`. Therefore the gain field
//! is unchanged while `x'` transforms as `r⊙x'`. Since the base HLS cell is
//! binding-equivariant, the composition is binding-equivariant as well.

use crate::config::fast_tanh;
use crate::continuous_hv::{ContinuousHV, UnitaryRole};
use crate::holographic_liquid::{HlsConfig, HlsError, HolographicLiquidCell};
use crate::invariant_context::{ContextMixerError, InvariantContextMixer};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug)]
pub enum ContextualHlsError {
    Hls(HlsError),
    Context(ContextMixerError),
    InvalidContextGain,
}

impl fmt::Display for ContextualHlsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hls(error) => write!(f, "{error}"),
            Self::Context(error) => write!(f, "{error}"),
            Self::InvalidContextGain => write!(f, "context gain must be finite"),
        }
    }
}

impl std::error::Error for ContextualHlsError {}

impl From<HlsError> for ContextualHlsError {
    fn from(value: HlsError) -> Self {
        Self::Hls(value)
    }
}

impl From<ContextMixerError> for ContextualHlsError {
    fn from(value: ContextMixerError) -> Self {
        Self::Context(value)
    }
}

/// Cross-dimensional role-invariant context composed with a binding-equivariant
/// HLS cell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualHolographicLiquidCell {
    base: HolographicLiquidCell,
    mixer: InvariantContextMixer,
    context_gain: f32,
}

impl ContextualHolographicLiquidCell {
    pub fn try_new(
        config: HlsConfig,
        offsets: Vec<usize>,
        context_gain: f32,
        seed: u64,
    ) -> Result<Self, ContextualHlsError> {
        if !context_gain.is_finite() {
            return Err(ContextualHlsError::InvalidContextGain);
        }
        let dim = config.dim;
        Ok(Self {
            base: HolographicLiquidCell::try_new(config, seed)?,
            mixer: InvariantContextMixer::try_new(dim, offsets, seed.wrapping_add(10_000))?,
            context_gain,
        })
    }

    pub fn state(&self) -> &ContinuousHV {
        self.base.state()
    }

    pub fn config(&self) -> &HlsConfig {
        self.base.config()
    }

    pub fn context_gain(&self) -> f32 {
        self.context_gain
    }

    pub fn mixer(&self) -> &InvariantContextMixer {
        &self.mixer
    }

    pub fn set_state(&mut self, state: ContinuousHV) -> Result<(), ContextualHlsError> {
        self.base.set_state(state)?;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.base.reset();
    }

    pub fn total_time(&self) -> f64 {
        self.base.total_time()
    }

    pub fn update_count(&self) -> u64 {
        self.base.update_count()
    }

    /// Compute the role-invariant context field for the current state/input.
    pub fn context(&self, input: &ContinuousHV) -> Result<ContinuousHV, ContextualHlsError> {
        Ok(self.mixer.mix(self.base.state(), input)?)
    }

    /// Compute the context-modulated input that will be sent to the base HLS cell.
    pub fn effective_input(
        &self,
        input: &ContinuousHV,
    ) -> Result<ContinuousHV, ContextualHlsError> {
        let context = self.context(input)?;
        let values = input
            .values
            .iter()
            .zip(context.values.iter())
            .map(|(&value, &ctx)| {
                let gain = 1.0 + self.context_gain * fast_tanh(ctx);
                value * gain
            })
            .collect();
        Ok(ContinuousHV::from_values(values))
    }

    /// Advance one contextual HLS event.
    pub fn step(
        &mut self,
        dt: f32,
        input: &ContinuousHV,
    ) -> Result<(), ContextualHlsError> {
        let effective = self.effective_input(input)?;
        self.base.step(dt, &effective)?;
        Ok(())
    }

    /// Maximum componentwise violation of the composed binding-equivariance law.
    pub fn binding_equivariance_error(
        &self,
        role: &UnitaryRole,
        input: &ContinuousHV,
        dt: f32,
    ) -> Result<f32, ContextualHlsError> {
        // Validate dimensions through the mixer before cloning transitions.
        let _ = self
            .mixer
            .role_invariance_error(role, self.base.state(), input)?;

        let mut direct = self.clone();
        direct.step(dt, input)?;
        let expected = role.bind(direct.state());

        let mut transformed = self.clone();
        transformed.set_state(role.bind(self.state()))?;
        transformed.step(dt, &role.bind(input))?;

        Ok(expected
            .values
            .iter()
            .zip(transformed.state().values.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::holographic_liquid::HlsActivation;

    fn config() -> HlsConfig {
        HlsConfig {
            dim: 256,
            activation: HlsActivation::Tanh,
            ..HlsConfig::default()
        }
    }

    #[test]
    fn contextual_cell_preserves_binding_equivariance() {
        let mut cell = ContextualHolographicLiquidCell::try_new(
            config(),
            vec![1, 7, 31],
            0.75,
            42,
        )
        .unwrap();
        cell.set_state(ContinuousHV::new_random(256, 43)).unwrap();
        let input = ContinuousHV::new_random(256, 44);
        let role = UnitaryRole::new(256, 45);
        let error = cell
            .binding_equivariance_error(&role, &input, 0.137)
            .unwrap();
        assert!(error <= 2e-6, "contextual equivariance error={error}");
    }

    #[test]
    fn remote_state_can_change_local_effective_input() {
        let mut cell = ContextualHolographicLiquidCell::try_new(
            HlsConfig {
                dim: 16,
                ..HlsConfig::default()
            },
            vec![1],
            1.0,
            7,
        )
        .unwrap();
        let mut input = ContinuousHV::new(16);
        input.values[0] = 1.0;

        let baseline = cell.effective_input(&input).unwrap();
        let mut changed_state = ContinuousHV::new(16);
        changed_state.values[1] = 2.0;
        cell.set_state(changed_state).unwrap();
        let changed = cell.effective_input(&input).unwrap();

        assert_ne!(baseline.values[0], changed.values[0]);
        assert_eq!(input.values[0], 1.0);
    }
}
