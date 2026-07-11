// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Ice-albedo feedback: a 0-D energy balance with temperature-dependent albedo.
//!
//! When the planet cools, ice spreads and raises albedo, reflecting more
//! sunlight and cooling it further — a positive feedback. This makes the energy
//! balance nonlinear and admits **multiple equilibria** (a warm state and a
//! frozen "snowball" state), the foundation of snowball-Earth theory
//! (Budyko 1969).
//!
//! Net radiation `N(T) = S(1-α(T))/4 − ε·σ·T⁴`. Equilibria are roots of `N`;
//! an equilibrium is stable when `dN/dT < 0`.

use crate::energy_balance::STEFAN_BOLTZMANN;

/// A 0-D energy balance whose albedo depends on temperature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IceAlbedoModel {
    /// Solar constant (W/m²).
    pub solar_constant: f64,
    /// Grey-atmosphere longwave emissivity.
    pub emissivity: f64,
    /// Albedo of a fully ice-covered planet (cold branch).
    pub albedo_ice: f64,
    /// Albedo of an ice-free planet (warm branch).
    pub albedo_warm: f64,
    /// At/below this temperature the planet is fully iced (K).
    pub t_ice: f64,
    /// At/above this temperature the planet is ice-free (K).
    pub t_warm: f64,
}

/// A temperature equilibrium and its stability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Equilibrium {
    pub temperature: f64,
    pub stable: bool,
}

impl IceAlbedoModel {
    /// Earth-like defaults: warm branch matches the grey-atmosphere Earth model.
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

    /// Temperature-dependent albedo: high when frozen, low when warm, linearly
    /// interpolated across the ice-line transition band.
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

    /// Net radiation `N(T) = S(1-α(T))/4 − ε·σ·T⁴` (W/m²).
    pub fn net_radiation(&self, t: f64) -> f64 {
        let absorbed = self.solar_constant * (1.0 - self.albedo(t)) / 4.0;
        let emitted = self.emissivity * STEFAN_BOLTZMANN * t.powi(4);
        absorbed - emitted
    }

    /// Find all equilibria (roots of `net_radiation`) in [210 K, 340 K] by
    /// scanning for sign changes and bisecting. Each is classified stable/
    /// unstable from the sign of `dN/dT`.
    pub fn equilibria(&self) -> Vec<Equilibrium> {
        let (lo, hi, step) = (180.0f64, 340.0f64, 0.25f64);
        let mut out = Vec::new();
        let mut t = lo;
        let mut prev_t = lo;
        let mut prev_n = self.net_radiation(lo);
        t += step;
        while t <= hi {
            let n = self.net_radiation(t);
            if prev_n == 0.0 {
                // Exact root at the sample point.
                out.push(self.classify(prev_t));
            } else if prev_n.signum() != n.signum() {
                let root = self.bisect(prev_t, t);
                out.push(self.classify(root));
            }
            prev_t = t;
            prev_n = n;
            t += step;
        }
        out
    }

    /// Warmest stable equilibrium temperature, if any (the habitable branch).
    pub fn warm_stable_temperature(&self) -> Option<f64> {
        self.equilibria()
            .into_iter()
            .filter(|e| e.stable)
            .map(|e| e.temperature)
            .fold(None, |acc, t| Some(acc.map_or(t, |a: f64| a.max(t))))
    }

    fn bisect(&self, mut a: f64, mut b: f64) -> f64 {
        let mut fa = self.net_radiation(a);
        for _ in 0..80 {
            let mid = 0.5 * (a + b);
            let fm = self.net_radiation(mid);
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
        // Stable if net radiation decreases through the root (restoring).
        let d = self.net_radiation(t + 0.01) - self.net_radiation(t - 0.01);
        Equilibrium {
            temperature: t,
            stable: d < 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn albedo_is_bounded_and_monotonic() {
        let m = IceAlbedoModel::earth();
        assert!((m.albedo(200.0) - 0.60).abs() < 1e-9); // fully iced
        assert!((m.albedo(300.0) - 0.30).abs() < 1e-9); // ice-free
        // Monotonically non-increasing with temperature.
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
        let m = IceAlbedoModel::earth();
        let warm = m
            .warm_stable_temperature()
            .expect("warm equilibrium exists");
        assert!(
            warm > 273.15,
            "warm state should be above freezing: {warm} K"
        );
        assert!(warm < 300.0, "warm state should be temperate: {warm} K");
    }

    #[test]
    fn every_equilibrium_zeroes_net_radiation() {
        let m = IceAlbedoModel::earth();
        for e in m.equilibria() {
            assert!(
                m.net_radiation(e.temperature).abs() < 1e-3,
                "N({}) = {}",
                e.temperature,
                m.net_radiation(e.temperature)
            );
        }
    }

    #[test]
    fn reducing_insolation_cools_the_warm_state() {
        // The ice-albedo feedback means a fainter sun gives a cooler (or lost)
        // habitable state. Compare warmest stable branch at high vs low S.
        let bright = IceAlbedoModel::earth();
        let faint = IceAlbedoModel {
            solar_constant: 1200.0,
            ..IceAlbedoModel::earth()
        };
        let tb = bright.warm_stable_temperature().unwrap();
        // Fainter sun: warm branch is cooler if it still exists.
        if let Some(tf) = faint.warm_stable_temperature() {
            assert!(tf < tb, "faint {tf} should be cooler than bright {tb}");
        }
    }

    #[test]
    fn snowball_state_is_stable_at_low_insolation() {
        // With a very faint sun the only stable state is frozen (< t_ice).
        let m = IceAlbedoModel {
            solar_constant: 600.0,
            ..IceAlbedoModel::earth()
        };
        let stable: Vec<_> = m.equilibria().into_iter().filter(|e| e.stable).collect();
        assert!(!stable.is_empty());
        assert!(
            stable.iter().all(|e| e.temperature < 260.0),
            "expected only a frozen stable state at S=600"
        );
    }
}
