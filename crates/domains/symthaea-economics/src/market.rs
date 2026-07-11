// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Linear market model: supply/demand equilibrium, elasticity, and surplus.

/// Linear demand curve `Qd = intercept − slope·P` (slope > 0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Demand {
    pub intercept: f64,
    pub slope: f64,
}

/// Linear supply curve `Qs = intercept + slope·P` (slope > 0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Supply {
    pub intercept: f64,
    pub slope: f64,
}

impl Demand {
    pub fn quantity_at(&self, price: f64) -> f64 {
        self.intercept - self.slope * price
    }
}

impl Supply {
    pub fn quantity_at(&self, price: f64) -> f64 {
        self.intercept + self.slope * price
    }
}

/// Market-clearing `(price, quantity)` where `Qd = Qs`. Returns `None` if the
/// equilibrium is economically meaningless (negative price or quantity).
pub fn equilibrium(demand: &Demand, supply: &Supply) -> Option<(f64, f64)> {
    let denom = demand.slope + supply.slope;
    if denom.abs() < 1e-15 {
        return None;
    }
    let price = (demand.intercept - supply.intercept) / denom;
    let quantity = demand.quantity_at(price);
    if price >= 0.0 && quantity >= 0.0 {
        Some((price, quantity))
    } else {
        None
    }
}

/// Point price-elasticity of demand `E = (dQ/dP)·(P/Q)` (reported as its
/// magnitude). For linear demand, `dQ/dP = −slope`.
pub fn price_elasticity_of_demand(demand: &Demand, price: f64) -> f64 {
    let q = demand.quantity_at(price);
    if q.abs() < 1e-15 {
        return f64::INFINITY;
    }
    ((-demand.slope) * (price / q)).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equilibrium_known() {
        // Qd = 100 - 2P, Qs = 20 + 2P → P* = 20, Q* = 60.
        let d = Demand {
            intercept: 100.0,
            slope: 2.0,
        };
        let s = Supply {
            intercept: 20.0,
            slope: 2.0,
        };
        let (p, q) = equilibrium(&d, &s).unwrap();
        assert!((p - 20.0).abs() < 1e-9, "p={p}");
        assert!((q - 60.0).abs() < 1e-9, "q={q}");
        // At equilibrium Qd == Qs.
        assert!((d.quantity_at(p) - s.quantity_at(p)).abs() < 1e-9);
    }

    #[test]
    fn elasticity_is_unit_at_midpoint() {
        // Qd = 100 - 2P. At P=25, Q=50 → E = |-2 * 25/50| = 1 (unit elastic).
        let d = Demand {
            intercept: 100.0,
            slope: 2.0,
        };
        assert!((price_elasticity_of_demand(&d, 25.0) - 1.0).abs() < 1e-9);
        // Lower price → inelastic (<1); higher price → elastic (>1).
        assert!(price_elasticity_of_demand(&d, 10.0) < 1.0);
        assert!(price_elasticity_of_demand(&d, 40.0) > 1.0);
    }
}
