// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#![forbid(unsafe_code)]

//! # symthaea-economics
//!
//! A pure-`std`, auditable economics kernel with explicit validation and
//! numerical-failure semantics. The crate contains mathematical primitives;
//! agent cognition and simulation adapters belong in higher layers.

pub mod error;
pub mod finance;
pub mod game;
pub mod inequality;
pub mod market;

pub use error::{EconomicsError, Result};
pub use finance::{
    AmortizationPeriod, IrrAnalysis, IrrOptions, IrrStatus, amortization_schedule, annuity_payment,
    compound_interest, effective_annual_rate, future_value, irr, irr_analysis, mirr,
    nominal_annual_rate, npv, present_value,
};
pub use game::{Game2x2, MixedNash};
pub use inequality::{
    LorenzPoint, atkinson_index, gini, hoover_index, lorenz_curve, normalized_gini, theil_t,
};
pub use market::{
    Demand, Equilibrium, MarketSurplus, PriceOutcome, Supply, TaxedEquilibrium,
    arc_price_elasticity_of_demand, equilibrium, equilibrium_with_tax, market_at_price,
    market_surplus, price_elasticity_of_demand,
};

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn a_market_and_its_inequality() {
        let demand = Demand::new(120.0, 3.0).unwrap();
        let supply = Supply::new(0.0, 3.0).unwrap();
        let point = equilibrium(&demand, &supply).unwrap();
        assert!((point.price - 20.0).abs() < 1e-9);
        assert!((point.quantity - 60.0).abs() < 1e-9);
        assert!(gini(&[10.0, 10.0, 10.0, 70.0]).unwrap() > 0.4);
    }

    #[test]
    fn loan_total_interest_is_positive() {
        let payment = annuity_payment(10_000.0, 0.05 / 12.0, 60).unwrap();
        assert!(payment * 60.0 > 10_000.0);
    }
}
