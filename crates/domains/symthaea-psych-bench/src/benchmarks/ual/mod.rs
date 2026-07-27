// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! UAL (Unlimited Associative Learning) Phase-1 probe packet.
//!
//! Implements the minimal P1/P2/P4a packet specified in
//! `symthaea/docs/SYMTHAEA_UAL_EXTENSION_DESIGN_2026-07-27.md` and
//! `symthaea/docs/SYMTHAEA_UAL_PHASE1_PROTOCOLS_2026-07-27.md`.
//!
//! **This is deliberately NOT a 15th Butlin indicator.** UAL is a
//! functional-capacity theory outside the frozen Butlin-14 denominator (see
//! the design doc, "Why UAL sits outside the frozen Butlin denominator") —
//! nothing here is registered in `benchmarks::butlin`, and no report from
//! this module may be summarized as "UAL demonstrated": per the design
//! doc's qualification requirement, even a clean P1/P2/P4a pass licenses at
//! most "initial compositional associative-learning profile demonstrated."
//!
//! - **P1** (`p1_reversal`): reversal learning, reusing
//!   `neuromod::reward_learning`'s existing Q-value/softmax mechanism.
//! - **P2** (`p2_second_order`): second-order conditioning via HDC
//!   bind-and-accumulate relational memory.
//! - **P4a** (`p4a_recombination`): held-out compositional recombination via
//!   a shared HDC value-integration memory.
//!
//! `common` holds shared low-level helpers (xorshift PRNG, softmax choice,
//! near-chance-similarity HV generation); `report` holds the mandatory
//! three-field UAL reporting types (functional outcome / internal
//! association formation / behavioral expression, with a schedule-status
//! qualifier).

pub mod common;
pub mod p1_reversal;
pub mod p2_second_order;
pub mod p4a_recombination;
pub mod report;

pub use p1_reversal::P1Reversal;
pub use p2_second_order::P2SecondOrder;
pub use p4a_recombination::P4aRecombination;
pub use report::{FunctionalOutcome, Presence, ScheduleStatus, UalProbeReport, UalSchedule};
