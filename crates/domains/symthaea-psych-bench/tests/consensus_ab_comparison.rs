// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Retired evidence surface: historical Sequential-vs-Consensus A/B comparison.
//!
//! The original test claimed to compare two feedback-integration interventions.
//! That intervention is no longer reachable: consensus integration is always-on,
//! and the former boolean branch only changed `genesis_phrase`. Results produced
//! by that code therefore could not identify a causal consensus-vs-sequential
//! effect.
//!
//! This integration-test target is intentionally retained as a fail-closed
//! sentinel so the historical command cannot silently continue producing
//! publishable-looking numbers:
//!
//! ```text
//! cargo test -p symthaea-psych-bench \
//!   --features symthaea-backend \
//!   --test consensus_ab_comparison -- --ignored
//! ```
//!
//! The target may only become an A/B experiment again after a separately
//! qualified intervention-reachability theorem establishes two distinct runtime
//! mechanisms under an otherwise frozen execution contract.

const RETIRED_EVIDENCE_MARKER: &str = "RETIRED_EVIDENCE_SURFACE";
const RETIRED_REASON: &str =
    "consensus-vs-sequential intervention is unreachable because consensus is always-on";

/// Fail closed when somebody explicitly invokes the historical ignored A/B test.
///
/// Normal `cargo test` leaves this ignored. The old evidence-producing command
/// used `--ignored`, so it now fails loudly instead of emitting misleading A/B
/// metrics.
#[test]
#[ignore = "retired: no reachable consensus-vs-sequential intervention"]
fn consensus_ab_comparison_is_retired_until_intervention_reachability_is_requalified() {
    panic!("{RETIRED_EVIDENCE_MARKER}: {RETIRED_REASON}");
}
