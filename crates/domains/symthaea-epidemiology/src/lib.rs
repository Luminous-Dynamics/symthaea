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
//! [`surveillance_receipt`] adds an evidence-bearing wrapper that binds a result
//! to the exact screening algorithm identifier, caller-supplied configuration,
//! baseline time scope, latest timestamp, and a SHA-256 content commitment over
//! the complete ordered input series including explicit missingness. The complete
//! receipt also has its own canonical content identity, committing to every
//! configuration and returned-assessment field.
//!
//! The surveillance modules report statistical change candidates only. They do
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
//!     assess_latest_change_with_receipt,
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
//! let receipt = assess_latest_change_with_receipt(&history, latest, config).unwrap();
//!
//! assert_eq!(receipt.input_id().to_hex().len(), 64);
//! assert_eq!(receipt.id().to_hex().len(), 64);
//! assert!(matches!(
//!     receipt.assessment().disposition,
//!     ScreeningDisposition::ChangeCandidate(_)
//! ));
//! ```

pub mod sir;
pub mod surveillance;
pub mod surveillance_receipt;

pub use sir::{Sir, State};
pub use surveillance::{
    AbstentionReason, ChangeDirection, IntervalEstimate, RobustBaseline, ScreeningDisposition,
    SurveillanceAssessment, SurveillancePoint, SurveillanceScreenConfig, SurveillanceScreenError,
    assess_latest_change,
};
pub use surveillance_receipt::{
    BaselineTimeWindow, SURVEILLANCE_SCREEN_ALGORITHM_V1, SURVEILLANCE_SCREEN_INPUT_ID_DOMAIN_V1,
    SURVEILLANCE_SCREEN_RECEIPT_ID_DOMAIN_V1, SurveillanceScreenInputId, SurveillanceScreenReceipt,
    SurveillanceScreenReceiptId, assess_latest_change_with_receipt,
};
