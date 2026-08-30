use serde::{Deserialize, Serialize};

use crate::{assess_homeostasis, NativeInteroceptiveState, ViabilityChannel, CHANNEL_COUNT};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AllostaticConfig {
    pub horizon_steps: u16,
    pub dt: f32,
    pub discount: f32,
}

impl Default for AllostaticConfig {
    fn default() -> Self {
        Self {
            horizon_steps: 16,
            dt: 1.0,
            discount: 0.95,
        }
    }
}

impl AllostaticConfig {
    pub fn validate(&self) {
        assert!(self.horizon_steps > 0);
        assert!(self.dt.is_finite() && self.dt > 0.0);
        assert!(self.discount.is_finite() && self.discount >= 0.0 && self.discount <= 1.0);
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AllostaticReport {
    pub discounted_debt: f32,
    pub peak_projected_deviation: f32,
    pub terminal_deviation: f32,
    pub projected_viability_breaches: u16,
    pub channel_debt: [f32; CHANNEL_COUNT],
}

pub fn assess_allostasis(
    state: &NativeInteroceptiveState,
    config: AllostaticConfig,
) -> AllostaticReport {
    config.validate();

    let mut discounted_debt = 0.0_f32;
    let mut peak_projected_deviation = 0.0_f32;
    let mut terminal_deviation = 0.0_f32;
    let mut projected_viability_breaches = 0_u16;
    let mut channel_debt = [0.0_f32; CHANNEL_COUNT];
    let mut discount_weight = 1.0_f32;
    let mut total_discount_weight = 0.0_f32;

    for step in 1..=config.horizon_steps {
        let projected = state.predicted(config.dt * step as f32);
        let report = assess_homeostasis(&projected);

        discounted_debt += report.weighted_deviation * discount_weight;
        total_discount_weight += discount_weight;
        peak_projected_deviation = peak_projected_deviation.max(report.peak_deviation);
        projected_viability_breaches = projected_viability_breaches
            .saturating_add(report.violated_channels as u16);

        for channel in ViabilityChannel::ALL {
            channel_debt[channel.index()] +=
                report.channel_deviations[channel.index()] * discount_weight;
        }

        if step == config.horizon_steps {
            terminal_deviation = report.weighted_deviation;
        }

        discount_weight *= config.discount;
    }

    if total_discount_weight > f32::EPSILON {
        discounted_debt /= total_discount_weight;
        for debt in &mut channel_debt {
            *debt /= total_discount_weight;
        }
    }

    AllostaticReport {
        discounted_debt,
        peak_projected_deviation,
        terminal_deviation,
        projected_viability_breaches,
        channel_debt,
    }
}
