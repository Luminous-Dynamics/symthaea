use serde::{Deserialize, Serialize};

use crate::{NativeInteroceptiveState, ViabilityChannel, ViabilityVariable, CHANNEL_COUNT};

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
        assert!(
            self.recovery_rate * self.step_dt <= 1.0,
            "recovery gain per step must not exceed 1.0"
        );
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

/// Mechanically measured receipt for one native regulatory transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InteroceptiveStepReport {
    pub cycle_before: u64,
    pub cycle_after: u64,
    pub driven_channels: u8,
    pub restorative_channels: u8,
    pub clamped_channels: u8,
    pub changed_channels: u8,
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
    pub(crate) fn state_mut(&mut self) -> &mut NativeInteroceptiveState {
        &mut self.state
    }

    #[inline]
    pub fn config(&self) -> InteroceptiveDynamicsConfig {
        self.config
    }

    #[inline]
    pub fn cycle(&self) -> u64 {
        self.cycle
    }

    pub fn step(&mut self, drive: InteroceptiveDrive) -> InteroceptiveStepReport {
        let dt = self.config.step_dt;
        let cycle_before = self.cycle;
        let mut driven_channels = 0_u8;
        let mut restorative_channels = 0_u8;
        let mut clamped_channels = 0_u8;
        let mut changed_channels = 0_u8;

        for channel in ViabilityChannel::ALL {
            let variable = self.state.get_mut(channel);
            let previous = variable.value();
            let restorative_rate = restorative_rate(variable, self.config.recovery_rate);
            let external_rate = drive.rate(channel);

            if restorative_rate != 0.0 {
                restorative_channels = restorative_channels.saturating_add(1);
            }
            if external_rate != 0.0 {
                driven_channels = driven_channels.saturating_add(1);
            }

            let proposed = previous + (restorative_rate + external_rate) * dt;
            assert!(proposed.is_finite(), "native transition produced a non-finite value");
            let next = proposed.clamp(self.config.min_value, self.config.max_value);
            if next != proposed {
                clamped_channels = clamped_channels.saturating_add(1);
            }
            if next != previous {
                changed_channels = changed_channels.saturating_add(1);
            }

            variable.set_observation(next, (next - previous) / dt);
        }

        self.cycle = self.cycle.saturating_add(1);

        InteroceptiveStepReport {
            cycle_before,
            cycle_after: self.cycle,
            driven_channels,
            restorative_channels,
            clamped_channels,
            changed_channels,
        }
    }
}

fn restorative_rate(variable: &ViabilityVariable, recovery_rate: f32) -> f32 {
    if variable.value() < variable.preferred_low() {
        (variable.preferred_low() - variable.value()) * recovery_rate
    } else if variable.value() > variable.preferred_high() {
        (variable.preferred_high() - variable.value()) * recovery_rate
    } else {
        0.0
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
