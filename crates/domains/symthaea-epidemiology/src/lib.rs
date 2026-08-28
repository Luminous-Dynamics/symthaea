// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-epidemiology
//!
//! A lightweight epidemiology domain for Symthaea.
//!
//! The crate currently has two deliberately separate capabilities:
//!
//! - [`sir`] — SIR compartment dynamics and closed-form epidemic quantities;
//! - [`surveillance`] — conservative statistical screening of **aggregate** time
//!   series using a robust historical baseline and explicit uncertainty-aware
//!   abstention.
//!
//! The surveillance module reports statistical change candidates only. It does
//! **not** diagnose disease, declare an outbreak, identify a pathogen, estimate
//! operational/public-health authority, or recommend a response. Source trust,
//! lineage/corroboration, persistence, competing hypotheses, and action authority
//! belong to later evidence/reasoning layers.
//!
//! ## SIR example
//!
//! ```
//! use symthaea_epidemiology::Sir;
//! let flu = Sir { beta: 0.3, gamma: 0.1 };      // R0 = 3
//! assert!((flu.basic_reproduction_number() - 3.0).abs() < 1e-12);
//! assert!((flu.herd_immunity_threshold() - 0.6667).abs() < 1e-3);
//! assert!(flu.final_size() > 0.9);
//! ```
//!
//! ## Aggregate surveillance example
//!
//! ```
//! use symthaea_epidemiology::{
//!     ScreeningDisposition, SurveillancePoint, SurveillanceScreenConfig,
//!     assess_latest_change,
//! };
//!
//! let history = [
//!     SurveillancePoint::observed(1, 8.0, 7.9, 8.1).unwrap(),
//!     SurveillancePoint::observed(2, 9.0, 8.9, 9.1).unwrap(),
//!     SurveillancePoint::observed(3, 10.0, 9.9, 10.1).unwrap(),
//!     SurveillancePoint::observed(4, 11.0, 10.9, 11.1).unwrap(),
//!     SurveillancePoint::observed(5, 12.0, 11.9, 12.1).unwrap(),
//! ];
//! let latest = SurveillancePoint::observed(6, 20.0, 19.0, 21.0).unwrap();
//! let config = SurveillanceScreenConfig::new(5, 3.0).unwrap();
//! let assessment = assess_latest_change(&history, latest, config).unwrap();
//!
//! assert!(matches!(
//!     assessment.disposition,
//!     ScreeningDisposition::ChangeCandidate(_)
//! ));
//! ```

pub mod sir;
pub mod surveillance;

pub use sir::{Sir, State};
pub use surveillance::{
    AbstentionReason, ChangeDirection, IntervalEstimate, RobustBaseline, ScreeningDisposition,
    SurveillanceAssessment, SurveillancePoint, SurveillanceScreenConfig, SurveillanceScreenError,
    assess_latest_change,
};
