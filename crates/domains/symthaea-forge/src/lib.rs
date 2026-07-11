// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! `symthaea-forge`: a mutate -> verify -> benchmark -> select search loop
//! over a single target function's AST.
//!
//! This is Tier 2.1 of `DISCOVERY_AND_SELF_IMPROVEMENT_PLAN_2026-07-06.md`
//! -- the one item in that plan requiring genuinely new construction, not
//! a seam-fix. It answers the question "can Symthaea discover a better
//! *algorithm*, not just tune numeric constants?" with a real, if
//! deliberately scoped, yes: [`mutations`] applies AST-level structural
//! mutations (comparison-operator swaps, arithmetic-operator swaps, and
//! literal perturbation) via `syn`, not text/regex matching; [`fitness`]
//! enforces correctness (compile, then existing tests) strictly
//! pass/fail before a candidate is ever benchmarked; [`sandbox`] makes
//! the necessary in-place staging crash-safe; [`search`] ties it together
//! into a population/generations loop; and [`certificate`] records every
//! winning candidate's full evidence for human review -- nothing here
//! ever auto-applies a mutation to the real source tree.
//!
//! # What this deliberately does NOT do
//!
//! - No autonomous promotion. The search's output is a certificate + a
//!   proposed file written to an output directory, never an in-place edit
//!   of the real source. A human (today: whoever runs `forge-run` and
//!   reviews `report.md`) decides whether to apply it.
//! - No general-purpose program synthesis. The mutation operator set is
//!   small and specific (see [`mutations`]); this is a *local* search
//!   around an existing, working implementation, not code generation from
//!   a specification.
//! - No claim that every search run finds an improvement. Most won't --
//!   that is the expected, honest outcome for a well-written starting
//!   function with a small mutation operator set and a modest population/
//!   generation budget. The infrastructure being real does not imply every
//!   run being fruitful.

pub mod certificate;
pub mod fitness;
pub mod mutations;
pub mod sandbox;
pub mod search;

pub use certificate::ForgeCertificate;
pub use search::{ForgeConfig, SearchOutcome, run_search};
