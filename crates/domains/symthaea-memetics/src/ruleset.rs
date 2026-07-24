// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Ruleset import — Warded Node design, Phase 5a
//! (`WARDED_NODE_DESIGN_2026-07-11.md`).
//!
//! [`MemeticImmuneSystem::vaccinate`] takes one pathogen signature at a time,
//! which is fine for signatures learned live but impractical for a guardian
//! who wants to protect a warded node with a starting set of known-bad
//! patterns from a source *they* trust (a friend, a school, a nonprofit
//! list). A [`Ruleset`] is a named, versioned, describable collection of
//! such signatures a guardian can bulk-import in one call.
//!
//! ## Scope (read before assuming more than this provides)
//!
//! This is deliberately **not** the full "Layer C" ruleset *curation and
//! distribution* problem from the Warded Node design — that also asks *who*
//! authors/publishes a shared ruleset and how a node decides to trust one,
//! which is a governance/social question, not an engineering one, and is
//! explicitly left for the project owner rather than defaulted here.
//! This module only solves the mechanical half: given a ruleset a guardian
//! already trusts, get every signature into the immune system in one call.
//!
//! **Not cryptographically signed.** Verifying a publisher's signature needs
//! a keypair/identity system, which would pull this crate into
//! `mycelix-identity` trust decisions it stays clear of by design (see
//! `CORE_SUBSTRATE.md`'s HDC-medium discipline — this crate is deliberately
//! mesh/identity-unaware). Import is trust-at-import-time: the same model
//! [`MemeticImmuneSystem::vaccinate`] already has (the caller fully vouches
//! for what they pass in). A signed variant is future work, not silently
//! promised here.
//!
//! **No file I/O here either.** `Ruleset` only derives `Serialize`/
//! `Deserialize` — reading bytes off disk, over a socket, wherever, and
//! picking a wire format (JSON, CBOR, ...) is the host application's job,
//! not this dependency-light crate's.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::binary_hv::BinaryHV;

/// One entry in a [`Ruleset`]: a known-bad pattern plus enough context for a
/// human to understand what it targets and where it came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RulesetEntry {
    /// The pathogen's HDC signature, as passed to
    /// [`MemeticImmuneSystem::vaccinate`](crate::MemeticImmuneSystem::vaccinate).
    pub signature: BinaryHV,
    /// Human-readable description of what this entry targets (e.g. "known
    /// grooming-pattern phrasing, reported 2026-07").
    pub description: String,
}

/// A named, versioned, describable collection of pathogen signatures a
/// guardian can bulk-import into a warded node's immune system.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Ruleset {
    /// Short human-readable name, e.g. `"family-safety-baseline"`.
    pub name: String,
    /// Free-form version string (semver, a date, whatever the source uses).
    pub version: String,
    /// Where this ruleset came from, for the guardian's own record — not
    /// verified by this crate (see module docs: no signature checking).
    pub source: String,
    pub entries: Vec<RulesetEntry>,
}

impl Ruleset {
    /// A new, empty ruleset with the given identifying metadata.
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        source: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            source: source.into(),
            entries: Vec::new(),
        }
    }

    /// Append one entry, builder-style.
    pub fn with_entry(mut self, signature: BinaryHV, description: impl Into<String>) -> Self {
        self.entries.push(RulesetEntry {
            signature,
            description: description.into(),
        });
        self
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::defense::MemeticImmuneSystem;
    use crate::meme::Meme;

    fn sample_ruleset() -> (BinaryHV, BinaryHV, Ruleset) {
        let a = BinaryHV::random(1);
        let b = BinaryHV::random(2);
        let ruleset = Ruleset::new("family-safety-baseline", "2026.07.11", "test-fixture")
            .with_entry(a.clone(), "known pattern A")
            .with_entry(b.clone(), "known pattern B");
        (a, b, ruleset)
    }

    #[test]
    fn empty_ruleset_vaccinates_nothing() {
        let mut immune = MemeticImmuneSystem::new(BinaryHV::random(9), 1.0);
        let empty = Ruleset::new("empty", "1.0", "nobody");
        assert!(empty.is_empty());
        let applied = immune.vaccinate_ruleset(&empty);
        assert_eq!(applied, 0);
        assert_eq!(immune.immune_memory_size(), 0);
    }

    #[test]
    fn ruleset_import_vaccinates_every_entry() {
        let (_, _, ruleset) = sample_ruleset();
        assert_eq!(ruleset.len(), 2);

        let mut immune = MemeticImmuneSystem::new(BinaryHV::random(9), 1.0);
        let applied = immune.vaccinate_ruleset(&ruleset);
        assert_eq!(applied, 2);
        assert_eq!(immune.immune_memory_size(), 2);
    }

    #[test]
    fn imported_signatures_are_genuinely_recognized_as_pathogens() {
        // Not just "the count went up" — a variant of an imported entry must
        // actually be rejected by the live screen(), same as a directly
        // vaccinate()'d pathogen.
        let (a, _, ruleset) = sample_ruleset();
        let mut immune = MemeticImmuneSystem::new(BinaryHV::random(9), 1.0);
        immune.vaccinate_ruleset(&ruleset);

        let variant = Meme::seed(1, a.add_noise(0.1, 3), 0.9);
        let outcome = immune.screen(&variant, crate::defense::GuardianPosture::Green, 0.0);
        assert!(
            !outcome.accepted,
            "an imported entry's variant must be rejected: {outcome:?}"
        );
        assert!(outcome.threat_match >= 0.7);
    }

    #[test]
    fn ruleset_survives_a_json_roundtrip() {
        // Proves the format is genuinely portable (a file a guardian could
        // download/share), not just an in-process convenience.
        let (a, b, ruleset) = sample_ruleset();

        let json = serde_json::to_string(&ruleset).expect("Ruleset must serialize to JSON");
        let restored: Ruleset = serde_json::from_str(&json).expect("must deserialize back");

        assert_eq!(restored.name, ruleset.name);
        assert_eq!(restored.version, ruleset.version);
        assert_eq!(restored.source, ruleset.source);
        assert_eq!(restored.entries.len(), 2);
        assert_eq!(restored.entries[0].signature, a);
        assert_eq!(restored.entries[1].signature, b);
        assert_eq!(restored.entries[0].description, "known pattern A");
    }

    #[test]
    fn builder_api_is_order_preserving() {
        let x = BinaryHV::random(10);
        let y = BinaryHV::random(11);
        let z = BinaryHV::random(12);
        let ruleset = Ruleset::new("n", "v", "s")
            .with_entry(x, "x")
            .with_entry(y, "y")
            .with_entry(z, "z");
        assert_eq!(ruleset.len(), 3);
        assert_eq!(ruleset.entries[0].description, "x");
        assert_eq!(ruleset.entries[1].description, "y");
        assert_eq!(ruleset.entries[2].description, "z");
    }
}
