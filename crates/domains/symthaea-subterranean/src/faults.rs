// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic sensor-fault injection for safety regression campaigns.

use crate::types::{NUM_STATE_CHANNELS, STATE_CHANNEL_RANGES, SubterraneanState};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SensorFault {
    Dropout { channel: usize },
    StuckAt { channel: usize, value: f64 },
    Bias { channel: usize, delta: f64 },
    OutOfRangeHigh { channel: usize },
    OutOfRangeLow { channel: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultInjectionError {
    InvalidChannel(usize),
}

impl std::fmt::Display for FaultInjectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidChannel(channel) => {
                write!(f, "sensor fault channel {channel} is out of range")
            }
        }
    }
}

impl std::error::Error for FaultInjectionError {}

impl SensorFault {
    pub fn apply(self, state: &mut SubterraneanState) -> Result<(), FaultInjectionError> {
        let channel = match self {
            Self::Dropout { channel }
            | Self::StuckAt { channel, .. }
            | Self::Bias { channel, .. }
            | Self::OutOfRangeHigh { channel }
            | Self::OutOfRangeLow { channel } => channel,
        };
        if channel >= NUM_STATE_CHANNELS {
            return Err(FaultInjectionError::InvalidChannel(channel));
        }
        let (minimum, maximum) = STATE_CHANNEL_RANGES[channel];
        match self {
            Self::Dropout { .. } => state.channels[channel] = f64::NAN,
            Self::StuckAt { value, .. } => state.channels[channel] = value,
            Self::Bias { delta, .. } => state.channels[channel] += delta,
            Self::OutOfRangeHigh { .. } => {
                state.channels[channel] = maximum + (maximum - minimum).abs().max(1.0)
            }
            Self::OutOfRangeLow { .. } => {
                state.channels[channel] = minimum - (maximum - minimum).abs().max(1.0)
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimedSensorFault {
    pub at_step: usize,
    pub fault: SensorFault,
}

#[derive(Debug, Clone, Default)]
pub struct SensorFaultCampaign {
    faults: Vec<TimedSensorFault>,
}

impl SensorFaultCampaign {
    pub fn new(faults: Vec<TimedSensorFault>) -> Self {
        Self { faults }
    }

    pub fn apply_step(
        &self,
        step: usize,
        state: &mut SubterraneanState,
    ) -> Result<usize, FaultInjectionError> {
        let mut applied = 0;
        for timed in &self.faults {
            if timed.at_step == step {
                timed.fault.apply(state)?;
                applied += 1;
            }
        }
        Ok(applied)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BATTERY_RATIO, GAS_RISK};

    #[test]
    fn deterministic_campaign_injects_only_at_requested_step() {
        let campaign = SensorFaultCampaign::new(vec![TimedSensorFault {
            at_step: 4,
            fault: SensorFault::Dropout { channel: GAS_RISK },
        }]);
        let mut state = SubterraneanState::home();
        assert_eq!(campaign.apply_step(3, &mut state).unwrap(), 0);
        assert!(state.channels[GAS_RISK].is_finite());
        assert_eq!(campaign.apply_step(4, &mut state).unwrap(), 1);
        assert!(state.channels[GAS_RISK].is_nan());
    }

    #[test]
    fn out_of_range_fault_is_detected_by_state_integrity() {
        let mut state = SubterraneanState::home();
        SensorFault::OutOfRangeHigh {
            channel: BATTERY_RATIO,
        }
        .apply(&mut state)
        .unwrap();
        assert!(!state.integrity_report().is_valid());
    }
}
