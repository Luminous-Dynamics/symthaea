use serde::{Deserialize, Serialize};

use crate::{NativeInteroceptiveState, ViabilityChannel, CHANNEL_COUNT};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct InteroceptiveDynamicsConfig {
    pub step_dt: f32,
    pub recovery_rate: f32,
    pub min_value: f32,
    pub max_value: f32,
}

impl Default for InteroceptiveDynamicsConfig {
    fn default() -> Self {
        Self {
            step_dt: 1.0,
            recovery_rate: 0.05,
            min_value: 0.0,
            max_value: 1.0,
        }
    }
}

impl InteroceptiveDynamicsConfig {
    pub fn validate(&self) {
        assert!(self.step_dt.is_finite() && self.step_dt > 0.0);
        assert!(self.recovery_rate.is_finite() && self.recovery_rate >= 0.0);
        assert!(self.min_value.is_finite());
        assert!(self.max_value.is_finite());
        assert!(self.min_value < self.max_value);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct InteroceptiveDrive {
    rates: [f32; CHANNEL_COUNT],
}

impl InteroceptiveDrive {
    pub const ZERO: Self = Self {
        rates: [0.0; CHANNEL_COUNT],
    };

    pub fn with_rate(mut self, channel: ViabilityChannel, rate: f32) -> Self {
        assert!(rate.is_finite());
        self.rates[channel.index()] = rate;
        self
    }

    #[inline]
    pub fn rate(&self, channel: ViabilityChannel) -> f32 {
        self.rates[channel.index()]
    }
}

impl Default for InteroceptiveDrive {
    fn default() -> Self {
        Self::ZERO
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeInteroceptiveModel {
    state: NativeInteroceptiveState,
    config: InteroceptiveDynamicsConfig,
    cycle: u64,
}

impl NativeInteroceptiveModel {
    pub fn new(state: NativeInteroceptiveState, config: InteroceptiveDynamicsConfig) -> Self {
        config.validate();
        Self {
            state,
            config,
            cycle: 0,
        }
    }

    #[inline]
    pub fn state(&self) -> &NativeInteroceptiveState {
        &self.state
    }

    #[inline]
    pub fn cycle(&self) -> u64 {
        self.cycle
    }

    pub fn step(&mut self, drive: InteroceptiveDrive) {
        let dt = self.config.step_dt;

        for channel in ViabilityChannel::ALL {
            let variable = self.state.get_mut(channel);
            let previous = variable.value;
            let restorative_rate =
                (variable.preferred_midpoint() - previous) * self.config.recovery_rate;
            let total_rate = restorative_rate + drive.rate(channel);
            let next = (previous + total_rate * dt)
                .clamp(self.config.min_value, self.config.max_value);

            variable.value = next;
            variable.velocity = (next - previous) / dt;
        }

        self.cycle = self.cycle.saturating_add(1);
    }
}

impl Default for NativeInteroceptiveModel {
    fn default() -> Self {
        Self::new(
            NativeInteroceptiveState::default(),
            InteroceptiveDynamicsConfig::default(),
        )
    }
}
