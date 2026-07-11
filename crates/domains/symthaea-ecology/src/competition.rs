// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! The competitive Lotka-Volterra model — two species competing for a shared
//! resource.
//!
//! ```text
//! dN₁/dt = r₁·N₁·(1 − (N₁ + a₁₂·N₂)/K₁)
//! dN₂/dt = r₂·N₂·(1 − (N₂ + a₂₁·N₁)/K₂)
//! ```
//!
//! `a₁₂` is the effect of species 2 on species 1 (and vice versa). The
//! coexistence equilibrium and its stability (via the competitive-exclusion
//! conditions) are the classic results.

/// Competitive Lotka-Volterra parameters.
#[derive(Debug, Clone, Copy)]
pub struct Competition {
    /// Carrying capacities.
    pub k1: f64,
    pub k2: f64,
    /// Competition coefficients: `a12` = effect of sp.2 on sp.1, `a21` = vice versa.
    pub a12: f64,
    pub a21: f64,
}

impl Competition {
    /// The interior coexistence equilibrium `(N₁*, N₂*)`, where both isoclines
    /// cross with both populations positive. `None` if the isoclines are
    /// parallel (`a12·a21 = 1`) or the crossing is not in the positive quadrant
    /// (one species is competitively excluded).
    pub fn coexistence_equilibrium(&self) -> Option<(f64, f64)> {
        let denom = 1.0 - self.a12 * self.a21;
        if denom.abs() < 1e-12 {
            return None;
        }
        let n1 = (self.k1 - self.a12 * self.k2) / denom;
        let n2 = (self.k2 - self.a21 * self.k1) / denom;
        if n1 > 0.0 && n2 > 0.0 {
            Some((n1, n2))
        } else {
            None
        }
    }

    /// Whether a stable coexistence equilibrium exists. For the competitive
    /// model this is the classic condition **`a12·a21 < 1`** (intraspecific
    /// competition dominates interspecific) *and* an interior equilibrium
    /// exists.
    pub fn stable_coexistence(&self) -> bool {
        self.a12 * self.a21 < 1.0 && self.coexistence_equilibrium().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_weak_competition_coexists() {
        // Equal capacities, weak mutual competition → stable coexistence.
        let c = Competition {
            k1: 100.0,
            k2: 100.0,
            a12: 0.5,
            a21: 0.5,
        };
        let (n1, n2) = c.coexistence_equilibrium().unwrap();
        // N1* = N2* = (100 - 0.5·100)/(1 - 0.25) = 50/0.75 = 66.67.
        assert!((n1 - 200.0 / 3.0).abs() < 1e-9, "{n1}");
        assert!((n2 - 200.0 / 3.0).abs() < 1e-9);
        assert!(c.stable_coexistence());
    }

    #[test]
    fn strong_competition_excludes() {
        // a12·a21 = 2.25 > 1: an interior equilibrium (40, 40) still exists, but
        // it is an unstable saddle — the outcome is founder-controlled
        // competitive exclusion, so `stable_coexistence` is false.
        let c = Competition {
            k1: 100.0,
            k2: 100.0,
            a12: 1.5,
            a21: 1.5,
        };
        assert!(c.coexistence_equilibrium().is_some());
        assert!(!c.stable_coexistence());
    }
}
