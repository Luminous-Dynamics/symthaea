// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reproducibility for the PubChem network boundary (Phase A.1 hardening).
//!
//! A scientific audit result should not change merely because PubChem is
//! temporarily unavailable, nor should a later PubChem edit silently alter
//! what an old audit run is claimed to have found. This module makes a live
//! audit run reproducible offline:
//!
//! - [`CachedLookup`] persists exactly what was retrieved for one SMILES:
//!   the outcome, a retrieval timestamp, the source host, and a
//!   non-cryptographic hash of the raw response body (when there was one)
//!   for change detection.
//! - [`PubChemFixtureCache`] is a save/load-able collection of those,
//!   serialized as JSON -- a frozen fixture file.
//! - [`RecordingSource`] wraps a real live lookup and accumulates a cache as
//!   it goes, so a live run can also produce a fixture for later replay.
//! - [`ReplaySource`] answers every lookup purely from a loaded fixture,
//!   with zero network access -- a fresh query can produce new advisory
//!   evidence, but the original recorded run stays exactly reconstructible.
//!
//! The hash here is explicitly **not** a security digest -- it exists only
//! to answer "did the recorded content change since I cached it," which is
//! a reproducibility concern, not an authentication one. A fast,
//! non-cryptographic-but-collision-resistant-enough hash (SHA-256, chosen
//! for being already a dependency-free standard choice, not because this
//! needs cryptographic guarantees) is sufficient.

use crate::pubchem::{PubChemQueryOutcome, PubChemSource};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::path::Path;

pub const PUBCHEM_SOURCE_HOST: &str = "pubchem.ncbi.nlm.nih.gov";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedLookup {
    pub smiles: String,
    pub source_host: String,
    pub retrieved_at_unix_secs: u64,
    /// `None` for an `Unavailable` outcome with no response body (a
    /// transport failure, or a size-cap rejection before any body was
    /// fully read) -- there is nothing to hash.
    pub content_hash: Option<String>,
    pub outcome: PubChemQueryOutcome,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PubChemFixtureCache {
    pub entries: Vec<CachedLookup>,
}

impl PubChemFixtureCache {
    pub fn load_from_file(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("reading {}: {e}", path.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("parsing {}: {e}", path.display()))
    }

    pub fn save_to_file(&self, path: &Path) -> Result<(), String> {
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, text).map_err(|e| format!("writing {}: {e}", path.display()))
    }

    pub fn find(&self, smiles: &str) -> Option<&CachedLookup> {
        self.entries.iter().find(|e| e.smiles == smiles)
    }

    /// Replaces any existing entry for the same SMILES rather than
    /// duplicating it -- a fixture always has at most one entry per
    /// distinct compound.
    pub fn insert(&mut self, entry: CachedLookup) {
        match self.entries.iter_mut().find(|e| e.smiles == entry.smiles) {
            Some(existing) => *existing = entry,
            None => self.entries.push(entry),
        }
    }
}

pub fn content_hash(body: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(body.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn now_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Wraps a real live PubChem lookup and accumulates a [`PubChemFixtureCache`]
/// as it goes. Call [`RecordingSource::into_cache`] after a run to get the
/// fixture, then [`PubChemFixtureCache::save_to_file`] to freeze it.
#[derive(Default)]
pub struct RecordingSource {
    recorded: RefCell<PubChemFixtureCache>,
}

impl RecordingSource {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn into_cache(self) -> PubChemFixtureCache {
        self.recorded.into_inner()
    }
}

impl PubChemSource for RecordingSource {
    fn lookup(&self, smiles: &str) -> PubChemQueryOutcome {
        let (outcome, raw_body) = crate::pubchem::lookup_by_smiles_with_raw(smiles);
        let entry = CachedLookup {
            smiles: smiles.to_string(),
            source_host: PUBCHEM_SOURCE_HOST.to_string(),
            retrieved_at_unix_secs: now_unix_secs(),
            content_hash: raw_body.as_deref().map(content_hash),
            outcome: outcome.clone(),
        };
        self.recorded.borrow_mut().insert(entry);
        outcome
    }
}

/// Answers every lookup purely from a loaded fixture -- zero network access.
/// A SMILES not present in the fixture is `Unavailable` (a fixture gap is a
/// legitimate "we don't have advisory data for this," never silently
/// treated as agreement or as a certification input either way).
pub struct ReplaySource {
    cache: PubChemFixtureCache,
}

impl ReplaySource {
    pub fn from_cache(cache: PubChemFixtureCache) -> Self {
        Self { cache }
    }

    pub fn from_file(path: &Path) -> Result<Self, String> {
        Ok(Self::from_cache(PubChemFixtureCache::load_from_file(path)?))
    }
}

impl PubChemSource for ReplaySource {
    fn lookup(&self, smiles: &str) -> PubChemQueryOutcome {
        match self.cache.find(smiles) {
            Some(entry) => entry.outcome.clone(),
            None => PubChemQueryOutcome::Unavailable(format!(
                "'{smiles}' not present in replay fixture cache"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pubchem::PubChemRecord;

    fn sample_cache() -> PubChemFixtureCache {
        let mut cache = PubChemFixtureCache::default();
        cache.insert(CachedLookup {
            smiles: "CCO".to_string(),
            source_host: PUBCHEM_SOURCE_HOST.to_string(),
            retrieved_at_unix_secs: 1_800_000_000,
            content_hash: Some(content_hash(r#"{"fake":"body"}"#)),
            outcome: PubChemQueryOutcome::Found(PubChemRecord {
                cid: 702,
                molecular_formula: "C2H6O".to_string(),
                connectivity_smiles: Some("CCO".to_string()),
                iupac_name: Some("ethanol".to_string()),
            }),
        });
        cache
    }

    #[test]
    fn insert_replaces_existing_entry_for_same_smiles_not_duplicates() {
        let mut cache = sample_cache();
        assert_eq!(cache.entries.len(), 1);
        cache.insert(CachedLookup {
            smiles: "CCO".to_string(),
            source_host: PUBCHEM_SOURCE_HOST.to_string(),
            retrieved_at_unix_secs: 1_900_000_000,
            content_hash: None,
            outcome: PubChemQueryOutcome::NotFound,
        });
        assert_eq!(cache.entries.len(), 1, "must replace, not duplicate");
        assert_eq!(cache.entries[0].outcome, PubChemQueryOutcome::NotFound);
    }

    #[test]
    fn save_and_load_round_trips_exactly() {
        let cache = sample_cache();
        let dir = std::env::temp_dir().join(format!(
            "symthaea-process-discovery-cache-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fixture.json");
        cache.save_to_file(&path).unwrap();
        let loaded = PubChemFixtureCache::load_from_file(&path).unwrap();
        assert_eq!(loaded.entries.len(), cache.entries.len());
        assert_eq!(loaded.entries[0].smiles, cache.entries[0].smiles);
        assert_eq!(loaded.entries[0].outcome, cache.entries[0].outcome);
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn replay_source_answers_from_fixture_with_zero_network() {
        let source = ReplaySource::from_cache(sample_cache());
        let outcome = source.lookup("CCO");
        assert!(matches!(outcome, PubChemQueryOutcome::Found(_)));
    }

    #[test]
    fn replay_source_reports_unavailable_not_a_false_agreement_for_missing_entry() {
        let source = ReplaySource::from_cache(sample_cache());
        let outcome = source.lookup("CCCCCCCC"); // not in the fixture
        assert!(matches!(outcome, PubChemQueryOutcome::Unavailable(_)));
    }

    #[test]
    fn replaying_the_same_fixture_twice_is_deterministic() {
        let cache = sample_cache();
        let source_a = ReplaySource::from_cache(cache.clone());
        let source_b = ReplaySource::from_cache(cache);
        assert_eq!(source_a.lookup("CCO"), source_b.lookup("CCO"));
    }

    #[test]
    fn content_hash_is_stable_and_sensitive_to_change() {
        let a = content_hash("hello");
        let b = content_hash("hello");
        let c = content_hash("hello!");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
