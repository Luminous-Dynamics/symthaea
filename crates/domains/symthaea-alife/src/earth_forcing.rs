// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phase 5a, per `ALIFE_PLAN_2026-07-08.md`: couples environmental forcing to a real domain
//! crate (`symthaea-earth-system`'s ice-albedo energy balance) instead of a synthetic sine wave,
//! for realism — while staying exogenous (an [`Organism`](crate::Organism) still cannot
//! influence it) and pure-std (no HDC/`EmbodimentBridge`/robotics coupling, matching this
//! crate's own non-goals).
//!
//! [`Environment`](crate::environment::Environment) (Phase 0) is left untouched — this is an
//! additional, alternative resource source, not a replacement.
//!
//! ## What's real physics vs. a documented modeling choice
//!
//! Real: [`symthaea_earth_system::IceAlbedoModel::net_radiation`] (Budyko 1969 ice-albedo
//! feedback — genuine nonlinear energy balance with bistable warm/snowball equilibria), forced
//! by a seasonal solar-constant cycle derived from Earth's actual orbital eccentricity range
//! (Berger 1978, ~6.9% peak-to-peak top-of-atmosphere insolation variation — not picked
//! freehand), integrated forward through real time via forward-Euler on a real mixed-layer
//! ocean heat capacity (Hartmann, *Global Physical Climatology*, 2nd ed., §2.7).
//!
//! Modeling choice, stated explicitly rather than left implicit: mapping surface temperature to
//! a `[0, 1]` "resource availability" proxy — habitability standing in for biological resource
//! abundance. This is an analogy (warmer, ice-free conditions ~ more biological productivity),
//! not a literal claim that Kelvin *is* nutrient concentration.
//!
//! ## Secular drift (Phase 7)
//!
//! [`EarthForcedEnvironment::with_secular_drift`] adds a slow, one-directional ramp to the
//! underlying solar constant on top of the seasonal cycle — a real, if simplified, stand-in for
//! a slowly deteriorating environment (as opposed to the bounded seasonal oscillation, which
//! always returns to the same mean). This is what makes an evolutionary-rescue experiment
//! possible (`tests/phase7_evolutionary_rescue.rs`): a threat that gets consistently worse, not
//! one that merely fluctuates around a fixed baseline.

use symthaea_earth_system::IceAlbedoModel;

/// Effective mixed-layer ocean heat capacity for a ~70 m reservoir — the standard 0-D/1-D
/// energy-balance choice for a model that must reproduce a real seasonal cycle timescale
/// (Hartmann, *Global Physical Climatology*, 2nd ed., §2.7). `C = ρ·c_p·depth`.
pub const MIXED_LAYER_HEAT_CAPACITY: f64 = 1000.0 * 4186.0 * 70.0; // ≈ 2.93e8 J/(m²·K)

/// Real seconds in a year — used so `seasonal_period_ticks` ticks span one real orbital period.
const SECONDS_PER_YEAR: f64 = 365.25 * 24.0 * 3600.0;

/// Earth's actual peak-to-peak top-of-atmosphere insolation variation from orbital eccentricity
/// (Berger 1978), as a fraction of the mean solar constant.
const ECCENTRICITY_INSOLATION_FRACTION: f64 = 0.069;

/// A resource signal driven by real, integrated ice-albedo climate physics rather than a
/// synthetic waveform. Stateful (temperature is real integrator state, not a closed-form lookup
/// of tick), so callers drive it with [`EarthForcedEnvironment::step`] inside a
/// `resource_for` closure — see `tests/phase5_earth_forcing.rs` for the pattern.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EarthForcedEnvironment {
    /// The real ice-albedo energy-balance model. Public so callers can perturb
    /// `solar_constant` etc. for scenario comparisons (see the Phase 5 ground-truth test).
    pub model: IceAlbedoModel,
    /// Current surface temperature (K) — real integrator state.
    pub temperature: f64,
    /// Peak-to-peak seasonal solar-constant variation (W/m²), derived from
    /// [`ECCENTRICITY_INSOLATION_FRACTION`], not picked freehand.
    pub seasonal_amplitude: f64,
    /// Ticks per full seasonal cycle.
    pub seasonal_period_ticks: f64,
    /// Real seconds of physical time one tick represents.
    pub dt_seconds: f64,
    /// Change in the underlying (pre-seasonal) solar constant per tick (W/m²) — zero by
    /// default (Phase 5a's behavior, unchanged). Negative values model a slowly dimming sun;
    /// positive values a slowly brightening one. See [`Self::with_secular_drift`].
    pub secular_drift_per_tick: f64,
    tick: u64,
}

impl EarthForcedEnvironment {
    /// Earth-like defaults, starting at the model's own warm stable equilibrium.
    /// `seasonal_period_ticks` ticks are calibrated to span one real year. No secular drift by
    /// default — use [`Self::with_secular_drift`] to add one.
    pub fn earth_like(seasonal_period_ticks: f64) -> Self {
        let model = IceAlbedoModel::earth();
        let temperature = model.warm_stable_temperature().unwrap_or(288.0);
        Self {
            seasonal_amplitude: model.solar_constant * ECCENTRICITY_INSOLATION_FRACTION,
            seasonal_period_ticks,
            dt_seconds: SECONDS_PER_YEAR / seasonal_period_ticks,
            secular_drift_per_tick: 0.0,
            temperature,
            model,
            tick: 0,
        }
    }

    /// Builder: add a secular solar-constant ramp (W/m² per tick) on top of the seasonal cycle.
    pub fn with_secular_drift(mut self, per_tick: f64) -> Self {
        self.secular_drift_per_tick = per_tick;
        self
    }

    /// Advance one tick: the underlying solar constant first drifts by
    /// `secular_drift_per_tick` (permanent, unlike the seasonal term), then a real forward-Euler
    /// step of `C·dT/dt = N(T)` runs under that new baseline plus the seasonal modulation,
    /// returning the resulting `[0, 1]` resource proxy.
    pub fn step(&mut self) -> f64 {
        self.model.solar_constant += self.secular_drift_per_tick;

        let phase = std::f64::consts::TAU * (self.tick as f64) / self.seasonal_period_ticks;
        let seasonal_offset = self.seasonal_amplitude * phase.sin();
        let mut forced_model = self.model;
        forced_model.solar_constant = self.model.solar_constant + seasonal_offset;

        let net_radiation = forced_model.net_radiation(self.temperature);
        let delta_temperature = net_radiation * self.dt_seconds / MIXED_LAYER_HEAT_CAPACITY;
        self.temperature += delta_temperature;
        self.tick += 1;

        self.resource_proxy()
    }

    /// Habitability proxy: 0 at/below the model's ice line, 1 at/above its warm line, linear in
    /// between — see the module doc's "documented modeling choice" note.
    fn resource_proxy(&self) -> f64 {
        ((self.temperature - self.model.t_ice) / (self.model.t_warm - self.model.t_ice))
            .clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn earth_like_stays_bounded_and_finite_over_many_ticks() {
        let mut env = EarthForcedEnvironment::earth_like(200.0);
        for _ in 0..5000u64 {
            let r = env.step();
            assert!(
                (0.0..=1.0).contains(&r),
                "resource proxy {r} out of range at temperature {}",
                env.temperature
            );
            assert!(env.temperature.is_finite(), "integrator diverged");
        }
    }

    #[test]
    fn earth_like_seasonal_forcing_actually_moves_temperature() {
        // Sanity check the seasonal term isn't a no-op -- confirms the forward-Euler integrator
        // is genuinely being forced, not just sitting at its initial equilibrium.
        let mut env = EarthForcedEnvironment::earth_like(200.0);
        let start = env.temperature;
        for _ in 0..1000u64 {
            env.step();
        }
        assert!(
            (env.temperature - start).abs() > 1e-6,
            "temperature never moved from its initial value: {start}"
        );
    }

    #[test]
    fn dimmed_sun_pushes_the_real_model_toward_the_frozen_branch() {
        // Matches symthaea-earth-system's own `snowball_state_is_stable_at_low_insolation`
        // fixture (S=600) -- confirms our forcing genuinely inherits that real bistability,
        // not a locally-reimplemented approximation of it.
        let mut env = EarthForcedEnvironment::earth_like(200.0);
        env.model.solar_constant = 600.0;
        for _ in 0..5000u64 {
            env.step();
        }
        assert!(
            env.temperature < env.model.t_ice,
            "expected the real model to have driven temperature to the frozen branch, got {}",
            env.temperature
        );
    }

    #[test]
    fn secular_drift_permanently_lowers_the_solar_constant() {
        let start = IceAlbedoModel::earth().solar_constant;
        let mut env = EarthForcedEnvironment::earth_like(200.0).with_secular_drift(-0.1);
        for _ in 0..1000u64 {
            env.step();
        }
        let expected = start - 0.1 * 1000.0;
        assert!(
            (env.model.solar_constant - expected).abs() < 1e-6,
            "expected solar_constant={expected} after 1000 ticks of -0.1/tick drift, got {}",
            env.model.solar_constant
        );
    }

    #[test]
    fn zero_drift_is_a_true_no_op_matching_phase5_behavior() {
        // Confirms with_secular_drift(0.0) (the earth_like default) reproduces Phase 5a's
        // existing behavior exactly -- backward compatibility, not just "close enough".
        let mut plain = EarthForcedEnvironment::earth_like(200.0);
        let mut explicit_zero = EarthForcedEnvironment::earth_like(200.0).with_secular_drift(0.0);
        for _ in 0..2000u64 {
            let a = plain.step();
            let b = explicit_zero.step();
            assert_eq!(a, b, "zero drift diverged from the undecorated constructor");
        }
    }
}
