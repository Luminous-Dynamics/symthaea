// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Raw per-partner interaction history, per `ALIFE_MULTIAGENT_GENESIS_PLAN_2026-07-25.md` (G0c).
//!
//! The central design discipline of Genesis v0: **never hand an agent a value we have already
//! socially interpreted.** No `trust`, `reputation`, `reliability`, or `cooperation_score` field
//! exists anywhere in this crate. [`InteractionRecord`] stores only factual counters, keyed by
//! [`crate::agent_id::AgentId`] in each organism's own ledger — a plain arithmetic history, not a
//! synthesized signal. A plausible-looking compression like `net_balance = received − given` was
//! deliberately rejected during design: it destroys real information (`gave 0, received 1` and
//! `gave 10, received 11` collapse to the same number, and those may be completely different
//! relationships). `net_balance` remains available as an **analysis-only** metric (see
//! [`InteractionRecord::net_balance`]) — it is never fed into an organism's own observation.

use serde::{Deserialize, Serialize};

/// Factual history with one specific partner. Every field is a raw, directly-observed counter —
/// nothing here is inferred or socially interpreted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct InteractionRecord {
    /// Total ever transferred *from* this organism *to* the partner.
    pub given_to_partner: f64,
    /// Total ever transferred *from* the partner *to* this organism.
    pub received_from_partner: f64,
    /// How many ticks this organism has been paired with this specific partner (regardless of
    /// whether a transfer occurred on any given tick).
    pub encounter_count: u32,
}

impl InteractionRecord {
    /// Analysis-only summary (`received − given`). Deliberately never fed into an organism's own
    /// observation — see module docs for why this compression is unsafe as a percept.
    pub fn net_balance(&self) -> f64 {
        self.received_from_partner - self.given_to_partner
    }
}

/// Maps `[0, ∞) → [0, 1)`, asymptotically saturating. Pure numeric range-compression so a raw,
/// unboundedly-growing counter (e.g. `encounter_count` after thousands of ticks) can be fed into
/// the FEP substrate's `[0,1]`-scale observation channels without swamping belief-update
/// gradients — **not** a social interpretation. The authoritative [`InteractionRecord`] stored on
/// each organism's ledger always stays exact and uncompressed; only the copy handed to
/// `perceive()` goes through this transform.
pub fn compress_for_observation(x: f64) -> f64 {
    let x = x.max(0.0);
    x / (x + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn net_balance_is_analysis_only_and_matches_definition() {
        let record = InteractionRecord {
            given_to_partner: 0.3,
            received_from_partner: 0.5,
            encounter_count: 2,
        };
        assert!((record.net_balance() - 0.2).abs() < 1e-12);
    }

    #[test]
    fn distinct_histories_with_identical_net_balance_stay_distinguishable_in_the_raw_record() {
        // The exact case the design conversation flagged: these two records have the same
        // net_balance (1.0) but are clearly different relationships. The raw record must keep
        // them distinguishable even though the analysis-only summary does not.
        let modest = InteractionRecord {
            given_to_partner: 0.0,
            received_from_partner: 1.0,
            encounter_count: 1,
        };
        let deep = InteractionRecord {
            given_to_partner: 10.0,
            received_from_partner: 11.0,
            encounter_count: 20,
        };
        assert_eq!(modest.net_balance(), deep.net_balance());
        assert_ne!(
            modest, deep,
            "raw records must not collapse to the same value"
        );
    }

    #[test]
    fn compress_for_observation_is_bounded_and_monotonic() {
        let mut prev = compress_for_observation(0.0);
        assert_eq!(prev, 0.0);
        for x in [1.0, 10.0, 100.0, 10_000.0] {
            let c = compress_for_observation(x);
            assert!(
                (0.0..1.0).contains(&c),
                "compressed value {c} out of [0,1) for x={x}"
            );
            assert!(c > prev, "compression must be strictly increasing: x={x}");
            prev = c;
        }
    }

    #[test]
    fn compress_for_observation_never_negative_for_negative_input() {
        // Defensive: this crate's counters should never go negative, but the transform itself
        // must not produce a nonsensical result if one ever did.
        assert_eq!(compress_for_observation(-5.0), 0.0);
    }
}
