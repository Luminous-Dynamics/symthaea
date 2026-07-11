// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-statistics
//!
//! The statistics & probability layer the workspace was missing. Every
//! empirical domain (`symthaea-epidemiology`, `-physiology`, `-economics`) and
//! Symthaea's own calibration loop implicitly need it, yet there was no
//! foundational stats crate.
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link. Distribution CDFs
//! are built on **real special functions** ([`special`] — log-gamma, erf,
//! regularized incomplete gamma/beta), not table lookups, so Student's t and
//! chi-square are exact to machine tolerance. Every function is checked against
//! a known closed-form value.
//!
//! ## Layers
//! - [`descriptive`] — mean, variance, quantiles, covariance, correlation
//! - [`distributions`] — normal / binomial / Poisson / Student-t / chi-square
//!   (PDF, CDF, and the inverse-normal quantile)
//! - [`inference`] — one- and two-sample (Welch) t-tests, chi-square
//!   goodness-of-fit, confidence intervals
//! - [`regression`] — ordinary-least-squares simple linear regression
//! - [`bayes`] — Bayesian updating and binary-classifier / diagnostic metrics
//!   (sensitivity, specificity, PPV, likelihood ratios) — the calibration hook
//!
//! ## Example
//!
//! ```
//! use symthaea_statistics::bayes::posterior_positive;
//! // Rare disease (1%), sensitive (99%) and specific (95%) test:
//! // a positive result still only means ~17% chance of disease.
//! let ppv = posterior_positive(0.01, 0.99, 0.95).unwrap();
//! assert!((ppv - 0.1667).abs() < 1e-3);
//! ```

pub mod bayes;
pub mod descriptive;
pub mod distributions;
pub mod inference;
pub mod regression;
pub mod special;

pub use bayes::{Confusion, posterior, posterior_positive};
pub use descriptive::{correlation, covariance, mean, median, quantile, std_dev, variance};
pub use distributions::{
    binomial_pmf, chi_square_cdf, normal_cdf, normal_pdf, normal_quantile, poisson_pmf,
    students_t_cdf,
};
pub use inference::{
    ChiSquare, Interval, TTest, chi_square_gof, one_sample_t_test, welch_t_test,
    z_confidence_interval_mean,
};
pub use regression::{LinearFit, linear_regression};
