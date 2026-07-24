// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Single-species logistic growth: bounded growth toward a carrying capacity.

use crate::error::{ModelError, require_non_negative, require_positive};

/// Validated logistic-growth parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogisticModel {
    pub intrinsic_growth_rate: f64,
    pub carrying_capacity: f64,
}

impl LogisticModel {
    pub fn try_new(intrinsic_growth_rate: f64, carrying_capacity: f64) -> Result<Self, ModelError> {
        let model = Self {
            intrinsic_growth_rate,
            carrying_capacity,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        require_positive("intrinsic_growth_rate", self.intrinsic_growth_rate)?;
        require_positive("carrying_capacity", self.carrying_capacity)?;
        Ok(())
    }

    pub fn growth_rate(&self, population: f64) -> Result<f64, ModelError> {
        require_non_negative("population", population)?;
        Ok(growth_rate(
            population,
            self.intrinsic_growth_rate,
            self.carrying_capacity,
        ))
    }

    pub fn population(&self, initial: f64, time: f64) -> Result<f64, ModelError> {
        require_positive("initial_population", initial)?;
        crate::error::require_finite("time", time)?;
        Ok(population(
            initial,
            self.intrinsic_growth_rate,
            self.carrying_capacity,
            time,
        ))
    }

    pub fn time_to_reach(&self, initial: f64, target: f64) -> Option<f64> {
        time_to_reach(
            initial,
            self.intrinsic_growth_rate,
            self.carrying_capacity,
            target,
        )
    }
}

/// Instantaneous growth rate dN/dt = r·N·(1 − N/K).
pub fn growth_rate(n: f64, r: f64, k: f64) -> f64 {
    r * n * (1.0 - n / k)
}

/// Closed-form logistic solution.
pub fn population(n0: f64, r: f64, k: f64, t: f64) -> f64 {
    if n0 <= 0.0 {
        return 0.0;
    }
    k / (1.0 + ((k - n0) / n0) * (-r * t).exp())
}

/// Future time at which the population reaches `target`.
pub fn time_to_reach(n0: f64, r: f64, k: f64, target: f64) -> Option<f64> {
    if n0 <= 0.0 || k <= 0.0 || r <= 0.0 || target <= 0.0 {
        return None;
    }

    let lies_on_future_trajectory = (n0 < target && target < k) || (k < target && target < n0);
    if !lies_on_future_trajectory {
        return None;
    }

    let ratio = ((k - target) / target) / ((k - n0) / n0);
    if !ratio.is_finite() || ratio <= 0.0 {
        return None;
    }

    let t = -ratio.ln() / r;
    (t.is_finite() && t > 0.0).then_some(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validated_model_rejects_invalid_parameters() {
        assert!(LogisticModel::try_new(0.5, 100.0).is_ok());
        assert!(LogisticModel::try_new(0.0, 100.0).is_err());
        assert!(LogisticModel::try_new(0.5, -1.0).is_err());
    }

    #[test]
    fn initial_and_asymptote() {
        assert!((population(10.0, 1.0, 100.0, 0.0) - 10.0).abs() < 1e-12);
        assert!((population(10.0, 1.0, 100.0, 1000.0) - 100.0).abs() < 1e-6);
    }

    #[test]
    fn growth_is_zero_at_capacity_and_extinction() {
        assert!(growth_rate(100.0, 0.5, 100.0).abs() < 1e-12);
        assert!(growth_rate(0.0, 0.5, 100.0).abs() < 1e-12);
        let g_half = growth_rate(50.0, 0.5, 100.0);
        assert!(g_half > growth_rate(30.0, 0.5, 100.0));
        assert!(g_half > growth_rate(70.0, 0.5, 100.0));
    }

    #[test]
    fn time_to_reach_inverts_growth_trajectory() {
        let t = time_to_reach(10.0, 0.5, 100.0, 50.0).unwrap();
        assert!((population(10.0, 0.5, 100.0, t) - 50.0).abs() < 1e-9);
    }

    #[test]
    fn time_to_reach_rejects_past_target() {
        assert_eq!(time_to_reach(10.0, 0.5, 100.0, 5.0), None);
        assert_eq!(time_to_reach(200.0, 0.5, 100.0, 250.0), None);
    }

    #[test]
    fn time_to_reach_supports_decline_from_above_capacity() {
        let t = time_to_reach(200.0, 0.5, 100.0, 150.0).unwrap();
        assert!(t > 0.0);
        assert!((population(200.0, 0.5, 100.0, t) - 150.0).abs() < 1e-9);
    }
}
