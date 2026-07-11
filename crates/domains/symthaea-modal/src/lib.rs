// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-modal
//!
//! Propositional modal logic — the formal, testable core of metaphysics
//! (modality: necessity/possibility). Kripke-model evaluation plus bounded
//! validity / countermodel search across the classic systems **K / T / S4 / S5**.
//!
//! Second of the "hard" knowledge domains
//! (`symthaea/HARD_DOMAINS_PLAN_2026-07-07.md`). It extends the workspace's logic
//! substrate (DPLL/FOL engine, `symthaea-proof-audit`) into modal reasoning.
//! **Scope note:** this gives metaphysicians the *machinery* they argue with
//! (what follows necessarily from what), not answers to substantive metaphysical
//! questions — building the machinery is honest; "solving" metaphysics is not.
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link.
//!
//! ## Example
//!
//! ```
//! use symthaea_modal::kripke::{implies, necessarily, var};
//! use symthaea_modal::validity::{is_valid, System};
//! // The T axiom □p → p separates K from T:
//! let t = implies(necessarily(var("p")), var("p"));
//! assert!(!is_valid(&t, System::K));
//! assert!(is_valid(&t, System::T));
//! ```

pub mod kripke;
pub mod parse;
pub mod validity;

pub use kripke::{Formula, KripkeModel, and, implies, necessarily, not, or, possibly, var};
pub use parse::parse;
pub use validity::{System, find_countermodel, is_valid, is_valid_bounded};
