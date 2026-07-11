// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-economics
//!
//! A self-contained economics & finance layer for Symthaea. Fills a confirmed
//! gap — the workspace had only behavioral-game fragments (Ultimatum, Public
//! Goods in psych-bench), no quantitative economics.
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link. All results are
//! closed-form and checked against textbook values.
//!
//! ## Scope
//!
//! - Finance: future/present value, NPV, IRR, compound interest, annuity/loan
//!   payments.
//! - Markets: linear supply/demand equilibrium, price elasticity of demand.
//! - Inequality: Gini coefficient.
//! - Game theory: 2×2 pure-strategy Nash equilibria, dominant strategies.
//!
//! ## Example
//!
//! ```
//! use symthaea_economics::finance::{npv, irr};
//! assert!((npv(0.10, &[-1000.0, 500.0, 500.0, 500.0]) - 243.426).abs() < 0.01);
//! assert!((irr(&[-1000.0, 600.0, 600.0]).unwrap() - 0.13066).abs() < 1e-4);
//! ```

pub mod finance;
pub mod game;
pub mod inequality;
pub mod market;

pub use finance::{annuity_payment, compound_interest, future_value, irr, npv, present_value};
pub use game::Game2x2;
pub use inequality::gini;
pub use market::{Demand, Supply, equilibrium, price_elasticity_of_demand};

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn a_market_and_its_inequality() {
        // A market clears, and a lopsided income vector is measured.
        let d = Demand {
            intercept: 120.0,
            slope: 3.0,
        };
        let s = Supply {
            intercept: 0.0,
            slope: 3.0,
        };
        let (p, q) = equilibrium(&d, &s).unwrap();
        assert!((p - 20.0).abs() < 1e-9 && (q - 60.0).abs() < 1e-9);
        assert!(gini(&[10.0, 10.0, 10.0, 70.0]) > 0.4); // concentrated
    }

    #[test]
    fn loan_total_interest_is_positive() {
        let pmt = annuity_payment(10_000.0, 0.05 / 12.0, 60);
        let total_paid = pmt * 60.0;
        assert!(total_paid > 10_000.0); // you pay back more than you borrowed
    }
}
