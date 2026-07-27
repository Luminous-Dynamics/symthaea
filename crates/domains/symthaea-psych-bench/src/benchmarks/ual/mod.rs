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
//! reporting types: functional outcome (`Demonstrated`/`NotDemonstrated`/
//! `Inconclusive` — fail-closed via `UalRuntimeQualification`), system-under-
//! test identity (`SystemUnderTest` — every probe here is a benchmark-local
//! reference mechanism, never `LiveSymthaea`), internal association
//! formation, behavioral expression, and a schedule-status qualifier.
//!
//! **Claim-integrity repair pass (2026-07-27)**: an independent review plus
//! this codebase's own direct verification found real defects that predated
//! this pass — a misleading `FullSymthaea` enum-variant name (fixed:
//! `CandidateHdcLearner`), a P1 metric bug and an unhandled no-manipulation
//! edge case (fixed), and a P2 mechanism whose schedule-dependence claim was
//! algebraically inert (fixed: mechanism simplified, claim retracted with an
//! executable proof test) plus a behavioral criterion that was mathematically
//! incapable of detecting success (fixed: retrieval-identity criterion).
//! P4a's own binding/unbinding semantics remain an open, disclosed
//! "prototype, under audit" caveat — see `p4a_recombination`'s module doc and
//! `hdc_binding_properties.rs`.

pub mod common;
pub mod hdc_binding_properties;
pub mod p1_reversal;
pub mod p2_second_order;
pub mod p4a_recombination;
pub mod report;

pub use p1_reversal::P1Reversal;
pub use p2_second_order::P2SecondOrder;
pub use p4a_recombination::P4aRecombination;
pub use report::{
    FunctionalOutcome, Presence, ScheduleStatus, SystemUnderTest, UalProbeReport,
    UalRuntimeQualification, UalSchedule,
};
