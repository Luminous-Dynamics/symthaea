// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Inventory management: the economic order quantity model.

/// Economic order quantity `EOQ = √(2·D·S/H)` that minimizes total ordering +
/// holding cost, where D = annual demand, S = cost per order, H = holding cost
/// per unit per year.
pub fn economic_order_quantity(annual_demand: f64, order_cost: f64, holding_cost: f64) -> f64 {
    (2.0 * annual_demand * order_cost / holding_cost).sqrt()
}

/// Total annual cost at a given order quantity: `(D/Q)·S + (Q/2)·H`.
pub fn total_annual_cost(
    annual_demand: f64,
    order_cost: f64,
    holding_cost: f64,
    quantity: f64,
) -> f64 {
    (annual_demand / quantity) * order_cost + (quantity / 2.0) * holding_cost
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eoq_known() {
        // D=1000, S=10, H=2 → EOQ = √10000 = 100.
        assert!((economic_order_quantity(1000.0, 10.0, 2.0) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn eoq_minimizes_total_cost() {
        // Cost at the EOQ must be ≤ cost at nearby order sizes.
        let (d, s, h) = (1000.0, 10.0, 2.0);
        let eoq = economic_order_quantity(d, s, h);
        let at_eoq = total_annual_cost(d, s, h, eoq);
        assert!(at_eoq <= total_annual_cost(d, s, h, eoq * 0.8));
        assert!(at_eoq <= total_annual_cost(d, s, h, eoq * 1.25));
    }
}
