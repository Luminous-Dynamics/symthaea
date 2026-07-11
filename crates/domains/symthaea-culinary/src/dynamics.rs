//! Phase 2 — cooking-process dynamics as *real* physics (not "CfC magically knows
//! thermodynamics"). Two closed-form models, each ground-truth-tested:
//!
//! - [`NewtonCooling`]: lumped-capacitance transient for a well-mixed lump (a
//!   stirred pot/sauce), built on `symthaea_thermofluids::convection_heat_rate`.
//!   Honest about its validity range via the Biot number.
//! - [`krieger_dougherty`]: emulsion relative viscosity, which **diverges at the
//!   same φ_max = random close packing** where Phase 1's emulsion validator breaks
//!   the emulsion — the two phases agree on the physics.

use crate::thresholds::RANDOM_CLOSE_PACKING;
use symthaea_thermofluids::thermal::convection_heat_rate;

/// A well-mixed thermal lump exchanging heat with a constant-temperature
/// environment by Newton's law of cooling. Lumped capacitance assumes a uniform
/// internal temperature — valid only when the Biot number is small (< ~0.1);
/// [`NewtonCooling::lumped_capacitance_valid`] checks that.
#[derive(Clone, Copy, Debug)]
pub struct NewtonCooling {
    /// Convective coefficient h (W·m⁻²·K⁻¹).
    pub h: f64,
    /// Surface area A (m²).
    pub area: f64,
    /// Mass m (kg).
    pub mass: f64,
    /// Specific heat c (J·kg⁻¹·K⁻¹).
    pub specific_heat: f64,
    /// Environment temperature (°C).
    pub t_env: f64,
}

impl NewtonCooling {
    /// Time constant τ = m·c / (h·A), seconds.
    pub fn tau(&self) -> f64 {
        self.mass * self.specific_heat / (self.h * self.area)
    }

    /// Closed-form temperature at time `t` s given start temperature `t0` °C:
    /// T(t) = T_env + (T0 − T_env)·e^(−t/τ).
    pub fn temperature_at(&self, t0: f64, t: f64) -> f64 {
        self.t_env + (t0 - self.t_env) * (-t / self.tau()).exp()
    }

    /// Time (s) to reach `target` °C from `t0` °C, or `None` if the environment
    /// can never take it there (target on the wrong side of T_env).
    pub fn time_to_reach(&self, t0: f64, target: f64) -> Option<f64> {
        let num = target - self.t_env;
        let den = t0 - self.t_env;
        if den == 0.0 || num / den <= 0.0 || num.abs() > den.abs() {
            return None;
        }
        Some(-self.tau() * (num / den).ln())
    }

    /// Explicit-Euler integration of the same ODE, using the thermofluids Newton
    /// heat-rate primitive for the flux. Used to validate any faster surrogate
    /// (Phase 2b CfC) against the closed form.
    pub fn simulate(&self, t0: f64, total_s: f64, dt: f64) -> f64 {
        let mut temp = t0;
        let mut t = 0.0;
        while t < total_s {
            let step = dt.min(total_s - t);
            // q = h·A·(T − T_env); dT = −q·dt / (m·c)
            let q = convection_heat_rate(self.h, self.area, temp - self.t_env);
            temp -= q * step / (self.mass * self.specific_heat);
            t += step;
        }
        temp
    }

    /// Biot number Bi = h·L_c / k for characteristic length `char_length_m` and
    /// thermal conductivity `conductivity` (W·m⁻¹·K⁻¹). Lumped capacitance is a
    /// good approximation when Bi < 0.1.
    pub fn biot_number(&self, conductivity: f64, char_length_m: f64) -> f64 {
        self.h * char_length_m / conductivity
    }

    /// Is the lumped-capacitance assumption defensible here (Bi < 0.1)?
    pub fn lumped_capacitance_valid(&self, conductivity: f64, char_length_m: f64) -> bool {
        self.biot_number(conductivity, char_length_m) < 0.1
    }
}

/// Krieger–Dougherty relative viscosity of a suspension/emulsion:
/// η_r = (1 − φ/φ_max)^(−[η]·φ_max), with intrinsic viscosity [η] ≈ 2.5 for
/// spheres and φ_max = random close packing. Diverges as φ → φ_max — the same
/// point at which Phase 1 declares the emulsion broken.
///
/// Returns `f64::INFINITY` at or above φ_max.
pub fn krieger_dougherty(phi: f64, intrinsic_viscosity: f64) -> f64 {
    let phi_max = RANDOM_CLOSE_PACKING;
    if phi <= 0.0 {
        return 1.0;
    }
    if phi >= phi_max {
        return f64::INFINITY;
    }
    (1.0 - phi / phi_max).powf(-intrinsic_viscosity * phi_max)
}

/// Krieger–Dougherty with the conventional hard-sphere intrinsic viscosity 2.5.
pub fn emulsion_relative_viscosity(phi: f64) -> f64 {
    krieger_dougherty(phi, 2.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sauce() -> NewtonCooling {
        // ~1 kg of water-like sauce, 0.05 m² surface, still-air-ish h.
        NewtonCooling {
            h: 15.0,
            area: 0.05,
            mass: 1.0,
            specific_heat: 4186.0,
            t_env: 20.0,
        }
    }

    #[test]
    fn cools_toward_environment_never_past_it() {
        let s = sauce();
        let hot = s.temperature_at(90.0, 0.0);
        let later = s.temperature_at(90.0, 10.0 * s.tau());
        assert!((hot - 90.0).abs() < 1e-9);
        assert!(later > s.t_env && later < 25.0, "got {later}");
    }

    #[test]
    fn one_tau_is_63_percent_of_the_way() {
        // After one time constant, ΔT has decayed by 1 − 1/e ≈ 63.2 %.
        let s = sauce();
        let t = s.temperature_at(90.0, s.tau());
        let expected = 20.0 + 70.0 * (1.0f64 / std::f64::consts::E);
        assert!((t - expected).abs() < 1e-6, "got {t}, expected {expected}");
    }

    #[test]
    fn euler_simulation_matches_closed_form() {
        // Validates the integrator a CfC surrogate would be checked against.
        let s = sauce();
        let closed = s.temperature_at(90.0, 600.0);
        let sim = s.simulate(90.0, 600.0, 0.05);
        assert!((closed - sim).abs() < 0.1, "closed={closed} sim={sim}");
    }

    #[test]
    fn biot_flags_a_thick_roast_as_not_lumped() {
        // A thick low-conductivity roast (k≈0.5, L≈0.05 m) is conduction-limited.
        let s = NewtonCooling {
            h: 15.0,
            area: 0.1,
            mass: 2.0,
            specific_heat: 3200.0,
            t_env: 160.0,
        };
        assert!(!s.lumped_capacitance_valid(0.5, 0.05));
    }

    #[test]
    fn viscosity_diverges_at_the_breaking_point() {
        assert_eq!(emulsion_relative_viscosity(0.0), 1.0);
        let mid = emulsion_relative_viscosity(0.5);
        let high = emulsion_relative_viscosity(0.70);
        assert!(high > mid && mid > 1.0, "mid={mid} high={high}");
        assert!(
            high > 50.0,
            "near the limit an emulsion is very thick: {high}"
        );
        assert!(emulsion_relative_viscosity(RANDOM_CLOSE_PACKING).is_infinite());
    }
}
