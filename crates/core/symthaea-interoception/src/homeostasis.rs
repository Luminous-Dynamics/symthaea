use serde::{Deserialize, Serialize};

use crate::{NativeInteroceptiveState, ViabilityChannel, CHANNEL_COUNT};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HomeostaticReport {
    pub weighted_deviation: f32,
    pub peak_deviation: f32,
    pub violated_channels: u8,
    pub channel_deviations: [f32; CHANNEL_COUNT],
}

impl HomeostaticReport {
    #[inline]
    pub fn is_within_viability(&self) -> bool {
        self.violated_channels == 0
    }
}

pub fn assess_homeostasis(state: &NativeInteroceptiveState) -> HomeostaticReport {
    let mut channel_deviations = [0.0; CHANNEL_COUNT];
    let mut weighted_sum = 0.0_f32;
    let mut weight_sum = 0.0_f32;
    let mut peak_deviation = 0.0_f32;
    let mut violated_channels = 0_u8;

    for channel in ViabilityChannel::ALL {
        let variable = state.get(channel);
        let deviation = variable.normalized_deviation().max(0.0);
        channel_deviations[channel.index()] = deviation;

        let weight = variable.precision * variable.importance;
        weighted_sum += deviation * weight;
        weight_sum += weight;
        peak_deviation = peak_deviation.max(deviation);

        if !variable.is_viable() {
            violated_channels = violated_channels.saturating_add(1);
        }
    }

    HomeostaticReport {
        weighted_deviation: if weight_sum > f32::EPSILON {
            weighted_sum / weight_sum
        } else {
            0.0
        },
        peak_deviation,
        violated_channels,
        channel_deviations,
    }
}
