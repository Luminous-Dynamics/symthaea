use serde::{de, Deserialize, Deserializer, Serialize};

use crate::{NativeInteroceptiveState, ViabilityChannel, ViabilityVariable, CHANNEL_COUNT};

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct InteroceptiveDynamicsConfig {
    pub step_dt: f32,
    pub recovery_rate: f32,
    pub min_value: f32,
    pub max_value: f32,
}

#[derive(Deserialize)]
struct InteroceptiveDynamicsConfigWire {
    step_dt: f32,
    recovery_rate: f32,
    min_value: f32,
    max_value: f32,
}

impl<'de> Deserialize<'de> for InteroceptiveDynamicsConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = InteroceptiveDynamicsConfigWire::deserialize(deserializer)?;
        let config = Self {
            step_dt: wire.step_dt,
            recovery_rate: wire.recovery_rate,
            min_value: wire.min_value,
            max_value: wire.max_value,
        };
        config.try_validate().map_err(de::Error::custom)?;
        Ok(config)
    }
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
    pub fn try_validate(&self) -> Result<(), String> {
        if !self.step_dt.is_finite() || self.step_dt <= 0.0 {
            return Err("step_dt must be finite and positive".into());
        }
        if !self.recovery_rate.is_finite() || self.recovery_rate < 0.0 {
            return Err("recovery_rate must be finite and non-negative".into());
        }
        if self.recovery_rate * self.step_dt > 1.0 {
            return Err("recovery gain per step must not exceed 1.0".into());
        }
        if !self.min_value.is_finite() || !self.max_value.is_finite() {
            return Err("model bounds must be finite".into());
        }
        if self.min_value >= self.max_value {
            return Err("min_value must be less than max_value".into());
        }
        Ok(())
    }

    /// Validate that one native state is representable inside this model's
    /// declared numerical domain. A state may be outside its *viable* band, but
    /// neither its current value nor its preferred/viable geometry may lie
    /// outside the model domain and then be silently collapsed by clamping.
    pub fn try_validate_state(&self, state: &NativeInteroceptiveState) -> Result<(), String> {
        self.try_validate()?;

        for channel in ViabilityChannel::ALL {
            let variable = state.get(channel);
            for (field, value) in [
                ("value", variable.value()),
                ("preferred_low", variable.preferred_low()),
                ("preferred_high", variable.preferred_high()),
                ("viable_low", variable.viable_low()),
                ("viable_high", variable.viable_high()),
            ] {
                if value < self.min_value || value > self.max_value {
                    return Err(format!(
                        "channel {} {field}={value} lies outside model domain [{}, {}]",
                        channel.stable_id(), self.min_value, self.max_value
                    ));
                }
            }
        }

        Ok(())
    }

    pub fn validate(&self) {
        self.try_validate()
            .unwrap_or_else(|error| panic!("{error}"));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct InteroceptiveDrive {
    rates: [f32; CHANNEL_COUNT],
}

#[derive(Deserialize)]
struct InteroceptiveDriveWire {
    rates: [f32; CHANNEL_COUNT],
}

impl<'de> Deserialize<'de> for InteroceptiveDrive {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = InteroceptiveDriveWire::deserialize(deserializer)?;
        if wire.rates.iter().any(|rate| !rate.is_finite()) {
            return Err(de::Error::custom("interoceptive drive rates must be finite"));
        }
        Ok(Self { rates: wire.rates })
    }
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

#[derive(Debug, Clone, PartialEq)]
pub struct NativeInteroceptiveModel {
    state: NativeInteroceptiveState,
    config: InteroceptiveDynamicsConfig,
    cycle: u64,
}

impl NativeInteroceptiveModel {
    pub fn try_new(
        state: NativeInteroceptiveState,
        config: InteroceptiveDynamicsConfig,
    ) -> Result<Self, String> {
        config.try_validate_state(&state)?;
        Ok(Self {
            state,
            config,
            cycle: 0,
        })
    }

    pub fn new(state: NativeInteroceptiveState, config: InteroceptiveDynamicsConfig) -> Self {
        Self::try_new(state, config).unwrap_or_else(|error| panic!("{error}"))
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
            assert!(
                proposed.is_finite(),
                "native transition produced a non-finite value"
            );
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
