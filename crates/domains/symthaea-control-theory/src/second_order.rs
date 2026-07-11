// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Standard second-order system `s² + 2ζωₙ·s + ωₙ²`.

/// Damping regime of a second-order system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Damping {
    Undamped,
    Underdamped,
    CriticallyDamped,
    Overdamped,
}

/// A second-order system defined by natural frequency and damping ratio.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SecondOrder {
    /// Natural frequency ωₙ (rad/s).
    pub natural_freq: f64,
    /// Damping ratio ζ.
    pub damping_ratio: f64,
}

impl SecondOrder {
    pub fn damping(&self) -> Damping {
        let z = self.damping_ratio;
        if z == 0.0 {
            Damping::Undamped
        } else if z < 1.0 {
            Damping::Underdamped
        } else if (z - 1.0).abs() < 1e-12 {
            Damping::CriticallyDamped
        } else {
            Damping::Overdamped
        }
    }

    /// Damped natural frequency `ωd = ωₙ√(1−ζ²)` (rad/s), 0 if not underdamped.
    pub fn damped_freq(&self) -> f64 {
        let z = self.damping_ratio;
        if z >= 1.0 {
            0.0
        } else {
            self.natural_freq * (1.0 - z * z).sqrt()
        }
    }

    /// Percent overshoot for a unit step (underdamped): `100·e^(−ζπ/√(1−ζ²))`.
    /// Zero for critically/over-damped systems.
    pub fn percent_overshoot(&self) -> f64 {
        let z = self.damping_ratio;
        if z <= 0.0 {
            100.0
        } else if z >= 1.0 {
            0.0
        } else {
            100.0 * (-z * std::f64::consts::PI / (1.0 - z * z).sqrt()).exp()
        }
    }

    /// 2% settling time `≈ 4/(ζωₙ)` (s).
    pub fn settling_time(&self) -> f64 {
        4.0 / (self.damping_ratio * self.natural_freq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn damping_regimes() {
        assert_eq!(
            SecondOrder {
                natural_freq: 1.0,
                damping_ratio: 0.0
            }
            .damping(),
            Damping::Undamped
        );
        assert_eq!(
            SecondOrder {
                natural_freq: 1.0,
                damping_ratio: 0.5
            }
            .damping(),
            Damping::Underdamped
        );
        assert_eq!(
            SecondOrder {
                natural_freq: 1.0,
                damping_ratio: 1.0
            }
            .damping(),
            Damping::CriticallyDamped
        );
        assert_eq!(
            SecondOrder {
                natural_freq: 1.0,
                damping_ratio: 2.0
            }
            .damping(),
            Damping::Overdamped
        );
    }

    #[test]
    fn overshoot_and_settling_known() {
        // ζ=0.5, ωn=1 → overshoot ≈ 16.30%, settling ≈ 8 s.
        let s = SecondOrder {
            natural_freq: 1.0,
            damping_ratio: 0.5,
        };
        assert!(
            (s.percent_overshoot() - 16.303).abs() < 0.01,
            "OS={}",
            s.percent_overshoot()
        );
        assert!((s.settling_time() - 8.0).abs() < 1e-9);
    }

    #[test]
    fn critically_damped_has_no_overshoot() {
        let s = SecondOrder {
            natural_freq: 2.0,
            damping_ratio: 1.0,
        };
        assert_eq!(s.percent_overshoot(), 0.0);
        assert_eq!(s.damped_freq(), 0.0);
    }
}
