// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Bridge: feed `symthaea-chronicle` historical data into the main crate's
//! [`TemporalReasoner`].
//!
//! `symthaea-chronicle` is a pure-`std`, dependency-free crate that owns the
//! genuinely-new *history* layer — dated events, causal chains, anachronism.
//! Its own interval reasoning is intentionally minimal. This bridge loads a
//! [`Chronicle`]'s events as [`TemporalInterval`]s inside the main crate's
//! [`TemporalReasoner`], so historical timelines gain the richer engine:
//! HDC-encoded intervals, the full Allen composition table (transitive
//! inference), and phenomenal-binding candidate search — while chronicle's
//! unique causal/anachronism layer is preserved and re-exposed.
//!
//! Point events (`start == end`) become a minimal one-unit interval `[y, y+1]`
//! because [`TemporalInterval`] requires `end > start`.

use anyhow::Result;
use symthaea_chronicle::{AllenRelation as ChronAllen, Chronicle};

use super::temporal_primitives::{AllenRelation, TemporalConfig, TemporalReasoner};

/// Map chronicle's pure-`std` Allen relation onto the main crate's
/// HDC-grounded [`AllenRelation`]. The two enums are structurally identical;
/// only the two disjoint-order names differ (`Before`/`After` ↔
/// `Precedes`/`PrecededBy`).
pub fn map_relation(r: ChronAllen) -> AllenRelation {
    match r {
        ChronAllen::Before => AllenRelation::Precedes,
        ChronAllen::After => AllenRelation::PrecededBy,
        ChronAllen::Meets => AllenRelation::Meets,
        ChronAllen::MetBy => AllenRelation::MetBy,
        ChronAllen::Overlaps => AllenRelation::Overlaps,
        ChronAllen::OverlappedBy => AllenRelation::OverlappedBy,
        ChronAllen::Starts => AllenRelation::Starts,
        ChronAllen::StartedBy => AllenRelation::StartedBy,
        ChronAllen::During => AllenRelation::During,
        ChronAllen::Contains => AllenRelation::Contains,
        ChronAllen::Finishes => AllenRelation::Finishes,
        ChronAllen::FinishedBy => AllenRelation::FinishedBy,
        ChronAllen::Equals => AllenRelation::Equals,
    }
}

/// A [`Chronicle`] whose events have been loaded into a [`TemporalReasoner`].
/// Allen-relation queries run on the main crate's engine; causal-chain and
/// anachronism queries delegate to the underlying chronicle.
pub struct HistoricalTemporalBridge {
    reasoner: TemporalReasoner,
    chronicle: Chronicle,
}

impl HistoricalTemporalBridge {
    /// Build a bridge from a chronicle, loading every event as an interval in a
    /// fresh reasoner. Point events become `[y, y+1]`.
    pub fn from_chronicle(chronicle: Chronicle) -> Result<Self> {
        let mut reasoner = TemporalReasoner::new(TemporalConfig::default());
        for name in chronicle.chronological() {
            if let Some(ev) = chronicle.get_event(&name) {
                let start = ev.start as f64;
                let end = if ev.end > ev.start {
                    ev.end as f64
                } else {
                    start + 1.0
                };
                reasoner.create_interval(name.clone(), start, end)?;
            }
        }
        Ok(Self {
            reasoner,
            chronicle,
        })
    }

    /// The Allen relation of event `a` to event `b`, computed by the main
    /// crate's reasoner (cached, with the inverse cached too).
    pub fn relation(&mut self, a: &str, b: &str) -> Result<AllenRelation> {
        self.reasoner.get_relation(a, b)
    }

    /// Transitive inference: given `a`→`b` and `b`→`c`, the set of Allen
    /// relations that can hold between `a` and `c` (via the composition table).
    pub fn infer(&mut self, a: &str, b: &str, c: &str) -> Result<Vec<AllenRelation>> {
        let r1 = self.reasoner.get_relation(a, b)?;
        let r2 = self.reasoner.get_relation(b, c)?;
        Ok(self.reasoner.compose(r1, r2))
    }

    /// Chronicle's causal chain from `cause` to `effect`, if any (unique to the
    /// history layer — the temporal reasoner has no causal graph).
    pub fn causal_chain(&self, cause: &str, effect: &str) -> Option<Vec<String>> {
        self.chronicle.causal_chain(cause, effect)
    }

    /// Whether referencing `entity` at `year` is anachronistic.
    pub fn is_anachronistic(&self, entity: &str, year: i64) -> Option<bool> {
        self.chronicle.is_anachronistic(entity, year)
    }

    /// Borrow the underlying reasoner (for HDC encoding, binding candidates…).
    pub fn reasoner(&self) -> &TemporalReasoner {
        &self.reasoner
    }

    /// Borrow the underlying chronicle.
    pub fn chronicle(&self) -> &Chronicle {
        &self.chronicle
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn history() -> Chronicle {
        let mut c = Chronicle::new();
        c.event("printing_press", 1440, None)
            .event("reformation", 1517, Some(1648))
            .event("enlightenment", 1685, Some(1815))
            .causation("printing_press", "reformation")
            .causation("reformation", "enlightenment")
            .entity("napoleon", 1769, 1821);
        c
    }

    #[test]
    fn interval_relation_via_main_engine() {
        let mut b = HistoricalTemporalBridge::from_chronicle(history()).unwrap();
        // reformation [1517,1648] ends before enlightenment [1685,1815] starts.
        assert_eq!(
            b.relation("reformation", "enlightenment").unwrap(),
            AllenRelation::Precedes
        );
        // …and the reasoner caches the inverse.
        assert_eq!(
            b.relation("enlightenment", "reformation").unwrap(),
            AllenRelation::PrecededBy
        );
    }

    #[test]
    fn engine_agrees_with_chronicle_for_proper_intervals() {
        let chron = history();
        let expected = map_relation(
            chron
                .allen_relation("reformation", "enlightenment")
                .unwrap(),
        );
        let mut b = HistoricalTemporalBridge::from_chronicle(chron).unwrap();
        assert_eq!(
            b.relation("reformation", "enlightenment").unwrap(),
            expected
        );
    }

    #[test]
    fn transitive_inference_over_timeline() {
        let mut b = HistoricalTemporalBridge::from_chronicle(history()).unwrap();
        // press ≺ reformation ≺ enlightenment ⟹ press ≺ enlightenment.
        let inferred = b
            .infer("printing_press", "reformation", "enlightenment")
            .unwrap();
        assert!(inferred.contains(&AllenRelation::Precedes), "{inferred:?}");
    }

    #[test]
    fn causal_and_anachronism_layers_preserved() {
        let b = HistoricalTemporalBridge::from_chronicle(history()).unwrap();
        assert_eq!(
            b.causal_chain("printing_press", "enlightenment"),
            Some(vec![
                "printing_press".to_string(),
                "reformation".to_string(),
                "enlightenment".to_string(),
            ])
        );
        // Napoleon (1769–1821) referenced at a smartphone-era year: anachronism.
        assert_eq!(b.is_anachronistic("napoleon", 2007), Some(true));
    }
}
