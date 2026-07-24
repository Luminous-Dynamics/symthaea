// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Ice-albedo feedback in a 0-D energy balance.
//!
//! Net radiation is `N(T) = S(1-α(T))/4 − ε_eff·σ·T⁴`. Albedo is constant on
//! the frozen and warm branches and linear through the transition. That
//! piecewise structure permits a branch-aware equilibrium solver that detects
//! both sign-changing roots and tangential roots at saddle-node thresholds.

use crate::energy_balance::{STEFAN_BOLTZMANN, effective_emissivity_surface_temperature};
use crate::error::{ModelError, require_finite, require_fraction, require_positive};

const ROOT_FLUX_TOLERANCE: f64 = 1e-8;
const ROOT_TEMPERATURE_TOLERANCE: f64 = 1e-9;

/// A 0-D energy balance whose albedo depends on temperature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IceAlbedoModel {
    pub solar_constant: f64,
    /// Effective outgoing-longwave emissivity.
    pub emissivity: f64,
    pub albedo_ice: f64,
    pub albedo_warm: f64,
    pub t_ice: f64,
    pub t_warm: f64,
}

/// A temperature equilibrium and its local restoring classification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Equilibrium {
    pub temperature: f64,
    /// True only when perturbations on both sides are restoring.
    pub stable: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClimateLinearStability {
    Stable,
    Critical,
    Unstable,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClimateRecoveryDiagnostic {
    /// Local net-radiation slope, W m⁻² K⁻¹.
    pub net_radiation_slope: f64,
    pub stability: ClimateLinearStability,
    /// Stable-equilibrium e-folding time for the supplied heat capacity.
    pub e_folding_time_seconds: Option<f64>,
}

/// Analytic saddle-node candidate within the linear albedo-transition branch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SaddleNode {
    pub temperature: f64,
    pub solar_constant: f64,
}

/// All equilibria at one sampled value of the solar-control parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct EquilibriumSlice {
    pub solar_constant: f64,
    pub equilibria: Vec<Equilibrium>,
}

impl IceAlbedoModel {
    /// Construct and validate an ice-albedo model.
    pub fn try_new(
        solar_constant: f64,
        effective_olr_emissivity: f64,
        albedo_ice: f64,
        albedo_warm: f64,
        t_ice: f64,
        t_warm: f64,
    ) -> Result<Self, ModelError> {
        let model = Self {
            solar_constant,
            emissivity: effective_olr_emissivity,
            albedo_ice,
            albedo_warm,
            t_ice,
            t_warm,
        };
        model.validate()?;
        Ok(model)
    }

    /// Earth-like defaults.
    pub fn earth() -> IceAlbedoModel {
        IceAlbedoModel {
            solar_constant: 1361.0,
            emissivity: 0.62,
            albedo_ice: 0.60,
            albedo_warm: 0.30,
            t_ice: 263.0,
            t_warm: 283.0,
        }
    }

    /// Validate physical domains and transition ordering.
    pub fn validate(&self) -> Result<(), ModelError> {
        require_positive("solar_constant", self.solar_constant)?;
        require_positive("effective_olr_emissivity", self.emissivity)?;
        if self.emissivity > 1.0 {
            return Err(ModelError::OutOfRange {
                parameter: "effective_olr_emissivity",
                value: self.emissivity,
                min: f64::MIN_POSITIVE,
                max: 1.0,
            });
        }
        require_fraction("albedo_ice", self.albedo_ice)?;
        require_fraction("albedo_warm", self.albedo_warm)?;
        require_finite("t_ice", self.t_ice)?;
        require_finite("t_warm", self.t_warm)?;
        if self.t_ice >= self.t_warm {
            return Err(ModelError::InvalidOrdering {
                lower: "t_ice",
                lower_value: self.t_ice,
                upper: "t_warm",
                upper_value: self.t_warm,
            });
        }
        Ok(())
    }

    /// Temperature-dependent albedo.
    pub fn albedo(&self, t: f64) -> f64 {
        if t <= self.t_ice {
            self.albedo_ice
        } else if t >= self.t_warm {
            self.albedo_warm
        } else {
            let frac = (t - self.t_ice) / (self.t_warm - self.t_ice);
            self.albedo_ice + (self.albedo_warm - self.albedo_ice) * frac
        }
    }

    /// Net radiation `N(T)` (W/m²).
    pub fn net_radiation(&self, t: f64) -> f64 {
        let absorbed = self.solar_constant * (1.0 - self.albedo(t)) / 4.0;
        let emitted = self.emissivity * STEFAN_BOLTZMANN * t.powi(4);
        absorbed - emitted
    }

    /// Analytic derivative of net radiation away from transition kinks.
    pub fn net_radiation_derivative(&self, t: f64) -> f64 {
        let albedo_slope = if t > self.t_ice && t < self.t_warm {
            (self.albedo_warm - self.albedo_ice) / (self.t_warm - self.t_ice)
        } else {
            0.0
        };
        -self.solar_constant * albedo_slope / 4.0
            - 4.0 * self.emissivity * STEFAN_BOLTZMANN * t.powi(3)
    }

    /// Local linear recovery diagnostic for `C dT/dt = N(T)`.
    pub fn recovery_diagnostic(
        &self,
        temperature: f64,
        heat_capacity: f64,
    ) -> Result<ClimateRecoveryDiagnostic, ModelError> {
        self.validate()?;
        require_positive("temperature", temperature)?;
        require_positive("heat_capacity", heat_capacity)?;
        let slope = self.net_radiation_derivative(temperature);
        require_finite("net_radiation_slope", slope)?;
        let tolerance = 64.0 * f64::EPSILON * slope.abs().max(1.0);
        let stability = if slope.abs() <= tolerance {
            ClimateLinearStability::Critical
        } else if slope < 0.0 {
            ClimateLinearStability::Stable
        } else {
            ClimateLinearStability::Unstable
        };
        Ok(ClimateRecoveryDiagnostic {
            net_radiation_slope: slope,
            stability,
            e_folding_time_seconds: (stability == ClimateLinearStability::Stable)
                .then(|| -heat_capacity / slope),
        })
    }

    /// Find all non-negative equilibria. Invalid models return an empty set;
    /// use [`Self::try_equilibria`] when validation errors must be preserved.
    pub fn equilibria(&self) -> Vec<Equilibrium> {
        self.try_equilibria().unwrap_or_default()
    }

    /// Checked equilibrium solver over all non-negative temperatures.
    pub fn try_equilibria(&self) -> Result<Vec<Equilibrium>, ModelError> {
        self.equilibria_in(0.0, f64::MAX)
    }

    /// Find all equilibria in an explicit temperature interval.
    pub fn equilibria_in(
        &self,
        temperature_min: f64,
        temperature_max: f64,
    ) -> Result<Vec<Equilibrium>, ModelError> {
        self.validate()?;
        require_finite("temperature_min", temperature_min)?;
        require_finite("temperature_max", temperature_max)?;
        if temperature_min < 0.0 {
            return Err(ModelError::OutOfRange {
                parameter: "temperature_min",
                value: temperature_min,
                min: 0.0,
                max: f64::MAX,
            });
        }
        if temperature_min >= temperature_max {
            return Err(ModelError::InvalidOrdering {
                lower: "temperature_min",
                lower_value: temperature_min,
                upper: "temperature_max",
                upper_value: temperature_max,
            });
        }

        let mut roots = Vec::with_capacity(4);

        let cold = effective_emissivity_surface_temperature(
            self.solar_constant,
            self.albedo_ice,
            self.emissivity,
        );
        if cold <= self.t_ice + ROOT_TEMPERATURE_TOLERANCE {
            self.push_root(&mut roots, cold, temperature_min, temperature_max);
        }

        let mut transition_points = vec![self.t_ice, self.t_warm];
        let albedo_slope = (self.albedo_warm - self.albedo_ice) / (self.t_warm - self.t_ice);
        let stationary_cube =
            -self.solar_constant * albedo_slope / (16.0 * self.emissivity * STEFAN_BOLTZMANN);
        if stationary_cube > 0.0 {
            let stationary = stationary_cube.cbrt();
            if stationary > self.t_ice && stationary < self.t_warm {
                transition_points.push(stationary);
            }
        }
        transition_points.sort_by(f64::total_cmp);

        for &point in &transition_points {
            if self.net_radiation(point).abs() <= ROOT_FLUX_TOLERANCE {
                self.push_root(&mut roots, point, temperature_min, temperature_max);
            }
        }
        for pair in transition_points.windows(2) {
            let a = pair[0];
            let b = pair[1];
            let fa = self.net_radiation(a);
            let fb = self.net_radiation(b);
            if fa.signum() != fb.signum() {
                let root = self.bisect(a, b);
                self.push_root(&mut roots, root, temperature_min, temperature_max);
            }
        }

        let warm = effective_emissivity_surface_temperature(
            self.solar_constant,
            self.albedo_warm,
            self.emissivity,
        );
        if warm >= self.t_warm - ROOT_TEMPERATURE_TOLERANCE {
            self.push_root(&mut roots, warm, temperature_min, temperature_max);
        }

        roots.sort_by(|a, b| a.temperature.total_cmp(&b.temperature));
        Ok(roots)
    }

    /// Analytic transition-branch saddle node, if it lies inside the branch.
    ///
    /// This solves `N(T)=0` and `dN/dT=0` simultaneously for the model's
    /// piecewise-linear albedo branch. The returned solar constant is a control
    /// value; it need not equal `self.solar_constant`.
    pub fn transition_saddle_node(&self) -> Result<Option<SaddleNode>, ModelError> {
        self.validate()?;
        let albedo_slope = (self.albedo_warm - self.albedo_ice) / (self.t_warm - self.t_ice);
        if albedo_slope >= 0.0 {
            return Ok(None);
        }

        let intercept = 1.0 - self.albedo_ice + albedo_slope * self.t_ice;
        let temperature = 4.0 * intercept / (3.0 * albedo_slope);
        if !temperature.is_finite() || temperature <= self.t_ice || temperature >= self.t_warm {
            return Ok(None);
        }

        let solar_constant =
            -16.0 * self.emissivity * STEFAN_BOLTZMANN * temperature.powi(3) / albedo_slope;
        if !solar_constant.is_finite() || solar_constant <= 0.0 {
            return Ok(None);
        }
        Ok(Some(SaddleNode {
            temperature,
            solar_constant,
        }))
    }

    /// Sample equilibrium branches over an inclusive solar-constant interval.
    pub fn equilibrium_sweep(
        &self,
        solar_min: f64,
        solar_max: f64,
        samples: usize,
        temperature_min: f64,
        temperature_max: f64,
    ) -> Result<Vec<EquilibriumSlice>, ModelError> {
        self.validate()?;
        require_positive("solar_min", solar_min)?;
        require_positive("solar_max", solar_max)?;
        if solar_min >= solar_max {
            return Err(ModelError::InvalidOrdering {
                lower: "solar_min",
                lower_value: solar_min,
                upper: "solar_max",
                upper_value: solar_max,
            });
        }
        if samples < 2 {
            return Err(ModelError::OutOfRange {
                parameter: "samples",
                value: samples as f64,
                min: 2.0,
                max: f64::MAX,
            });
        }

        let mut sweep = Vec::with_capacity(samples);
        for index in 0..samples {
            let fraction = index as f64 / (samples - 1) as f64;
            let solar_constant = solar_min + fraction * (solar_max - solar_min);
            let model = Self {
                solar_constant,
                ..*self
            };
            sweep.push(EquilibriumSlice {
                solar_constant,
                equilibria: model.equilibria_in(temperature_min, temperature_max)?,
            });
        }
        Ok(sweep)
    }

    /// Warmest stable equilibrium temperature, if any.
    pub fn warm_stable_temperature(&self) -> Option<f64> {
        self.equilibria()
            .into_iter()
            .filter(|e| e.stable)
            .map(|e| e.temperature)
            .reduce(f64::max)
    }

    fn push_root(
        &self,
        roots: &mut Vec<Equilibrium>,
        temperature: f64,
        minimum: f64,
        maximum: f64,
    ) {
        if temperature < minimum - ROOT_TEMPERATURE_TOLERANCE
            || temperature > maximum + ROOT_TEMPERATURE_TOLERANCE
        {
            return;
        }
        if roots
            .iter()
            .any(|root| (root.temperature - temperature).abs() <= ROOT_TEMPERATURE_TOLERANCE)
        {
            return;
        }
        roots.push(self.classify(temperature));
    }

    fn bisect(&self, mut a: f64, mut b: f64) -> f64 {
        let mut fa = self.net_radiation(a);
        for _ in 0..100 {
            let mid = 0.5 * (a + b);
            let fm = self.net_radiation(mid);
            if fm.abs() <= ROOT_FLUX_TOLERANCE {
                return mid;
            }
            if fa.signum() == fm.signum() {
                a = mid;
                fa = fm;
            } else {
                b = mid;
            }
        }
        0.5 * (a + b)
    }

    fn classify(&self, t: f64) -> Equilibrium {
        let epsilon = (t.abs() * 1e-8).max(1e-5);
        let left = self.net_radiation((t - epsilon).max(0.0));
        let right = self.net_radiation(t + epsilon);
        Equilibrium {
            temperature: t,
            stable: left > 0.0 && right < 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn albedo_is_bounded_and_monotonic() {
        let m = IceAlbedoModel::earth();
        assert!((m.albedo(200.0) - 0.60).abs() < 1e-9);
        assert!((m.albedo(300.0) - 0.30).abs() < 1e-9);
        let mut last = m.albedo(200.0);
        let mut t = 200.0;
        while t <= 320.0 {
            let a = m.albedo(t);
            assert!(a <= last + 1e-12);
            last = a;
            t += 1.0;
        }
    }

    #[test]
    fn present_earth_has_a_habitable_stable_state() {
        let warm = IceAlbedoModel::earth()
            .warm_stable_temperature()
            .expect("warm equilibrium exists");
        assert!(warm > 273.15, "warm state: {warm} K");
        assert!(warm < 300.0, "warm state: {warm} K");
    }

    #[test]
    fn every_equilibrium_zeroes_net_radiation() {
        let m = IceAlbedoModel::earth();
        for e in m.equilibria() {
            assert!(m.net_radiation(e.temperature).abs() < 1e-6);
        }
    }

    #[test]
    fn equilibrium_interval_is_respected() {
        let m = IceAlbedoModel::earth();
        let roots = m.equilibria_in(270.0, 310.0).unwrap();
        assert!(
            roots
                .iter()
                .all(|root| (270.0..=310.0).contains(&root.temperature))
        );
    }

    #[test]
    fn tangential_transition_root_is_detected() {
        let m = IceAlbedoModel::try_new(
            1361.0,
            0.62,
            0.538_000_150_955_114,
            0.280_139_770_092_852_07,
            260.0,
            290.0,
        )
        .unwrap();
        let roots = m.equilibria_in(270.0, 280.0).unwrap();
        assert!(
            roots
                .iter()
                .any(|root| (root.temperature - 275.0).abs() < 1e-6),
            "roots={roots:?}"
        );
    }

    #[test]
    fn analytic_saddle_node_recovers_constructed_tangency() {
        let m = IceAlbedoModel::try_new(
            1361.0,
            0.62,
            0.538_000_150_955_114,
            0.280_139_770_092_852_07,
            260.0,
            290.0,
        )
        .unwrap();
        let saddle = m.transition_saddle_node().unwrap().unwrap();
        assert!((saddle.temperature - 275.0).abs() < 1e-9);
        assert!((saddle.solar_constant - 1361.0).abs() < 1e-8);

        let critical = IceAlbedoModel {
            solar_constant: saddle.solar_constant,
            ..m
        };
        assert!(critical.net_radiation(saddle.temperature).abs() < 1e-8);
        assert!(critical.net_radiation_derivative(saddle.temperature).abs() < 1e-10);
    }

    #[test]
    fn equilibrium_sweep_includes_both_control_endpoints() {
        let sweep = IceAlbedoModel::earth()
            .equilibrium_sweep(600.0, 1400.0, 5, 180.0, 330.0)
            .unwrap();
        assert_eq!(sweep.len(), 5);
        assert_eq!(sweep.first().unwrap().solar_constant, 600.0);
        assert_eq!(sweep.last().unwrap().solar_constant, 1400.0);
        assert!(sweep.iter().all(|slice| !slice.equilibria.is_empty()));
    }

    #[test]
    fn invalid_models_fail_closed() {
        let invalid = IceAlbedoModel {
            t_ice: 290.0,
            t_warm: 280.0,
            ..IceAlbedoModel::earth()
        };
        assert!(invalid.try_equilibria().is_err());
        assert!(invalid.equilibria().is_empty());
    }

    #[test]
    fn snowball_state_is_stable_at_low_insolation() {
        let m = IceAlbedoModel {
            solar_constant: 600.0,
            ..IceAlbedoModel::earth()
        };
        let stable: Vec<_> = m.equilibria().into_iter().filter(|e| e.stable).collect();
        assert!(!stable.is_empty());
        assert!(stable.iter().all(|e| e.temperature < 260.0));
    }

    #[test]
    fn linear_recovery_time_diverges_at_the_constructed_saddle() {
        let model = IceAlbedoModel::try_new(
            1361.0,
            0.62,
            0.538_000_150_955_114,
            0.280_139_770_092_852_07,
            260.0,
            290.0,
        )
        .unwrap();
        let saddle = model.transition_saddle_node().unwrap().unwrap();
        let critical = IceAlbedoModel {
            solar_constant: saddle.solar_constant,
            ..model
        };
        let diagnostic = critical
            .recovery_diagnostic(saddle.temperature, 4.0e8)
            .unwrap();
        assert_eq!(diagnostic.stability, ClimateLinearStability::Critical);
        assert!(diagnostic.e_folding_time_seconds.is_none());

        let warm = IceAlbedoModel::earth();
        let warm_temperature = warm.warm_stable_temperature().unwrap();
        let diagnostic = warm.recovery_diagnostic(warm_temperature, 4.0e8).unwrap();
        assert_eq!(diagnostic.stability, ClimateLinearStability::Stable);
        assert!(diagnostic.e_folding_time_seconds.unwrap() > 0.0);
    }
}
