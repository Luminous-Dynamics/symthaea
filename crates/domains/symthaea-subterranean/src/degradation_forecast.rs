// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Conservative, trend-based degradation-horizon forecasting.
//!
//! `MaintenanceMonitor` (see `maintenance.rs`) is reactive: it reports
//! today's health and today's disposition. `maintenance_window.rs`'s
//! planner needs a forward-looking horizon too -- "how many steps until
//! this crosses the service/abort threshold?" -- so it can return a
//! vehicle for scheduled service before an unscheduled failure forces it.
//!
//! [`DegradationForecaster`] extrapolates from a short rolling window of
//! per-component health observations to produce a [`DegradationForecast`].
//! Deliberately conservative in two ways:
//! 1. Predictions are driven by the single fastest-declining component,
//!    not an average across components -- a forecast that hides one
//!    badly-degrading part behind healthier ones is exactly the failure
//!    mode this module exists to prevent.
//! 2. The forecaster's own [`ForecastDisposition`] escalates on fixed,
//!    non-configurable horizons that are tighter than the caller-supplied
//!    `MaintenanceWindowPolicy` thresholds it's compared alongside in
//!    `maintenance_window.rs::MaintenanceWindowPlanner::assess` -- an
//!    operator who configures a lenient policy horizon still gets this
//!    independent, conservative backstop.

use crate::maintenance::{ComponentKind, NUM_COMPONENTS};
use serde::{Deserialize, Serialize};

/// Health threshold below which `MaintenanceMonitor::assessment` sets
/// `maintenance_due` -- mirrored here so this forecaster's own
/// `ForecastDisposition::ServiceSoon` escalates toward the same boundary
/// the reactive assessment already uses, not an independently invented one.
const SERVICE_HEALTH_THRESHOLD: f64 = 0.55;
/// Health threshold below which `MaintenanceMonitor::assessment` sets
/// `mission_abort_required`.
const ABORT_HEALTH_THRESHOLD: f64 = 0.22;
/// Fixed, non-configurable horizon (in observation steps) within which a
/// predicted service-threshold crossing escalates the disposition to
/// `ServiceSoon`. Deliberately tighter than `MaintenanceWindowPolicy`'s
/// default `service_return_horizon_steps` (10,000) -- see module docs.
const FORECAST_SERVICE_HORIZON_STEPS: u64 = 3_000;
/// Fixed, non-configurable horizon within which a predicted abort-threshold
/// crossing escalates the disposition to `AbortRisk`.
const FORECAST_ABORT_HORIZON_STEPS: u64 = 800;
/// Rolling window of recent health readings kept per component.
const TREND_WINDOW: usize = 8;

/// How urgently a [`DegradationForecast`] recommends acting, escalating
/// the same way `maintenance_window.rs::MaintenanceWindowDisposition` does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForecastDisposition {
    /// Fewer than two observations for the driving component: no trend can
    /// be computed yet, so no horizon claim is made.
    WarmingUp,
    /// Comfortable margin to both thresholds.
    Nominal,
    /// Extrapolated to cross the service threshold within
    /// `FORECAST_SERVICE_HORIZON_STEPS`.
    ServiceSoon,
    /// Extrapolated to cross the abort threshold within
    /// `FORECAST_ABORT_HORIZON_STEPS`.
    AbortRisk,
}

/// A forward-looking degradation horizon for the fastest-declining
/// component currently being tracked.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DegradationForecast {
    pub disposition: ForecastDisposition,
    /// The component driving this forecast, or `None` while `WarmingUp`.
    pub critical_component: Option<ComponentKind>,
    /// Current health of the driving component (the value the forecast's
    /// horizons are extrapolated from).
    pub predicted_minimum_health: f64,
    /// Extrapolated steps until the driving component's health crosses
    /// `SERVICE_HEALTH_THRESHOLD`, or `u64::MAX` if the trend never
    /// reaches it (flat or improving).
    pub steps_to_service: u64,
    /// Extrapolated steps until the driving component's health crosses
    /// `ABORT_HEALTH_THRESHOLD`, or `u64::MAX`.
    pub steps_to_abort: u64,
    /// How much of the trend window is filled for the driving component,
    /// in `[0.0, 1.0]`. Low confidence means the horizon above is being
    /// extrapolated from very little history.
    pub confidence: f64,
    /// Total observations recorded so far (across all components).
    pub observations: u32,
}

impl DegradationForecast {
    /// The forecast before any observations have been recorded.
    pub const fn warming_up() -> Self {
        Self {
            disposition: ForecastDisposition::WarmingUp,
            critical_component: None,
            predicted_minimum_health: 1.0,
            steps_to_service: u64::MAX,
            steps_to_abort: u64::MAX,
            confidence: 0.0,
            observations: 0,
        }
    }
}

/// Rolling per-component health trend tracker that produces
/// [`DegradationForecast`]s. Feed it a health reading per component (e.g.
/// from `MaintenanceMonitor::health`) at each step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationForecaster {
    windows: [[f64; TREND_WINDOW]; NUM_COMPONENTS],
    filled: [usize; NUM_COMPONENTS],
    observations: u32,
}

impl DegradationForecaster {
    pub fn new() -> Self {
        Self {
            windows: [[1.0; TREND_WINDOW]; NUM_COMPONENTS],
            filled: [0; NUM_COMPONENTS],
            observations: 0,
        }
    }

    /// Record one health reading per component, index-aligned with
    /// `ComponentKind::index`.
    pub fn observe(&mut self, health: [f64; NUM_COMPONENTS]) {
        for (component_index, &value) in health.iter().enumerate() {
            let value = if value.is_finite() {
                value.clamp(0.0, 1.0)
            } else {
                0.0
            };
            let window = &mut self.windows[component_index];
            window.rotate_left(1);
            window[TREND_WINDOW - 1] = value;
            self.filled[component_index] = (self.filled[component_index] + 1).min(TREND_WINDOW);
        }
        self.observations = self.observations.saturating_add(1);
    }

    /// Least-squares slope (health change per step) over the filled
    /// portion of one component's window. Positive means improving,
    /// negative means degrading. `None` if fewer than two readings are
    /// available yet.
    fn slope(&self, component_index: usize) -> Option<f64> {
        let filled = self.filled[component_index];
        if filled < 2 {
            return None;
        }
        let window = &self.windows[component_index][TREND_WINDOW - filled..];
        let n = filled as f64;
        let x_mean = (filled as f64 - 1.0) / 2.0;
        let y_mean = window.iter().sum::<f64>() / n;
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for (i, &y) in window.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }
        if denominator <= f64::EPSILON {
            return Some(0.0);
        }
        Some(numerator / denominator)
    }

    /// Steps until `current` crosses `threshold` if it keeps changing at
    /// `slope` per step. `0` if already at/below threshold, `u64::MAX` if
    /// the trend is flat/improving and will never reach it.
    fn steps_to_cross(current: f64, threshold: f64, slope: f64) -> u64 {
        if current <= threshold {
            return 0;
        }
        if slope >= -f64::EPSILON {
            return u64::MAX;
        }
        let steps = (current - threshold) / -slope;
        if steps.is_finite() && steps >= 0.0 {
            steps as u64
        } else {
            u64::MAX
        }
    }

    /// Produce a conservative forecast from the observations recorded so
    /// far, driven by the single fastest-declining component with enough
    /// history to compute a trend (see module docs for why not an average).
    pub fn forecast(&self) -> DegradationForecast {
        let mut worst: Option<(usize, f64, f64)> = None;
        for component in ComponentKind::ALL {
            let index = component.index();
            let Some(slope) = self.slope(index) else {
                continue;
            };
            let current = self.windows[index][TREND_WINDOW - 1];
            let is_worse = match worst {
                None => true,
                Some((_, _, worst_slope)) => slope < worst_slope,
            };
            if is_worse {
                worst = Some((index, current, slope));
            }
        }

        let Some((index, current, slope)) = worst else {
            return DegradationForecast {
                observations: self.observations,
                ..DegradationForecast::warming_up()
            };
        };

        let steps_to_service = Self::steps_to_cross(current, SERVICE_HEALTH_THRESHOLD, slope);
        let steps_to_abort = Self::steps_to_cross(current, ABORT_HEALTH_THRESHOLD, slope);
        let disposition = if steps_to_abort <= FORECAST_ABORT_HORIZON_STEPS {
            ForecastDisposition::AbortRisk
        } else if steps_to_service <= FORECAST_SERVICE_HORIZON_STEPS {
            ForecastDisposition::ServiceSoon
        } else {
            ForecastDisposition::Nominal
        };
        let confidence = (self.filled[index] as f64 / TREND_WINDOW as f64).clamp(0.0, 1.0);

        DegradationForecast {
            disposition,
            critical_component: ComponentKind::ALL.into_iter().find(|c| c.index() == index),
            predicted_minimum_health: current,
            steps_to_service,
            steps_to_abort,
            confidence,
            observations: self.observations,
        }
    }
}

impl Default for DegradationForecaster {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_observations_is_warming_up() {
        let forecaster = DegradationForecaster::new();
        let forecast = forecaster.forecast();
        assert_eq!(forecast.disposition, ForecastDisposition::WarmingUp);
        assert_eq!(forecast.critical_component, None);
        assert_eq!(forecast.observations, 0);
    }

    #[test]
    fn steady_decline_is_extrapolated_forward() {
        let mut forecaster = DegradationForecaster::new();
        // Cutter declines by 0.05/step, everything else stays perfect.
        let mut health = [1.0; NUM_COMPONENTS];
        for step in 0..TREND_WINDOW {
            health[ComponentKind::Cutter.index()] = 1.0 - 0.05 * (step as f64 + 1.0);
            forecaster.observe(health);
        }
        let forecast = forecaster.forecast();
        assert_eq!(forecast.critical_component, Some(ComponentKind::Cutter));
        assert!(forecast.steps_to_service < u64::MAX);
        assert!(forecast.steps_to_abort < u64::MAX);
        assert!(forecast.steps_to_abort >= forecast.steps_to_service);
        assert_eq!(forecast.confidence, 1.0);
    }

    #[test]
    fn flat_health_never_predicts_a_crossing() {
        let mut forecaster = DegradationForecaster::new();
        for _ in 0..TREND_WINDOW {
            forecaster.observe([1.0; NUM_COMPONENTS]);
        }
        let forecast = forecaster.forecast();
        assert_eq!(forecast.disposition, ForecastDisposition::Nominal);
        assert_eq!(forecast.steps_to_service, u64::MAX);
        assert_eq!(forecast.steps_to_abort, u64::MAX);
    }

    #[test]
    fn imminent_crossing_escalates_to_abort_risk() {
        let mut forecaster = DegradationForecaster::new();
        let mut health = [1.0; NUM_COMPONENTS];
        // Fast, steep decline that will cross both thresholds within the
        // fixed abort horizon.
        for step in 0..TREND_WINDOW {
            health[ComponentKind::ThermalPump.index()] = 0.9 - 0.3 * (step as f64 + 1.0);
            forecaster.observe(health);
        }
        let forecast = forecaster.forecast();
        assert_eq!(forecast.disposition, ForecastDisposition::AbortRisk);
        assert_eq!(
            forecast.critical_component,
            Some(ComponentKind::ThermalPump)
        );
    }

    #[test]
    fn worst_declining_component_drives_the_forecast_not_an_average() {
        let mut forecaster = DegradationForecaster::new();
        let mut health = [1.0; NUM_COMPONENTS];
        for step in 0..TREND_WINDOW {
            // Auger degrades fast; everything else (including Cutter)
            // stays flat. The forecast must be driven by Auger.
            health[ComponentKind::Auger.index()] = 1.0 - 0.08 * (step as f64 + 1.0);
            forecaster.observe(health);
        }
        let forecast = forecaster.forecast();
        assert_eq!(forecast.critical_component, Some(ComponentKind::Auger));
    }
}
