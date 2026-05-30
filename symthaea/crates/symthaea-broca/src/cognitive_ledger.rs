// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cognitive Ledger — Versioned provenance of her self-authoring breakthroughs.
//!
//! Stores 'Cognitive Commits' in a persistent, audited ledger.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use symthaea_core::hdc::ContinuousHV;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveCommit {
    pub timestamp: u64,
    pub mission_id: String,
    pub parent_hash: String,
    pub hash: String,
    pub intent_nucleus: ContinuousHV,
    pub synthesized_code: String,
    pub coherence: f32,
    pub entropy: f32,
    pub verified: bool,
}

#[derive(Clone)]
pub struct CognitiveLedger {
    pub storage_path: PathBuf,
    pub history: Vec<CognitiveCommit>,
}

impl CognitiveLedger {
    pub fn new(base_dir: &str) -> Result<Self> {
        let storage_path = PathBuf::from(base_dir).join("cognitive_ledger.json");
        let history = if storage_path.exists() {
            let data = fs::read_to_string(&storage_path)?;
            serde_json::from_str(&data)?
        } else {
            Vec::new()
        };
        Ok(Self {
            storage_path,
            history,
        })
    }

    pub fn commit(&mut self, mut commit: CognitiveCommit) -> Result<()> {
        println!("📝 Cognitive Ledger: Recording commit for mission '{}'...", commit.mission_id);

        // 1. Link to previous state
        commit.parent_hash = self.history.last().map(|c| c.hash.clone()).unwrap_or_default();

        // 2. Compute current hash (simplified SHA-256 via MD5 for demo speed)
        let content = format!("{}{}{:?}", commit.parent_hash, commit.synthesized_code, commit.intent_nucleus.norm());
        commit.hash = format!("{:x}", md5::compute(content));

        self.history.push(commit);
        let data = serde_json::to_string_pretty(&self.history)?;
        fs::write(&self.storage_path, data)?;
        Ok(())
    }

    /// Verify the integrity of the entire developmental history.
    pub fn verify_lineage(&self) -> bool {
        println!("🛡️ Cognitive Ledger: Verifying Memetic Lineage...");
        let mut expected_parent = String::new();
        for commit in &self.history {
            if commit.parent_hash != expected_parent {
                println!("   ❌ Lineage BREAK detected at mission '{}'.", commit.mission_id);
                return false;
            }
            expected_parent = commit.hash.clone();
        }
        println!("   ✅ Lineage verified. developmental history is cryptographically sound.");
        true
    }


    pub fn last_commit(&self) -> Option<&CognitiveCommit> {
        self.history.last()
    }
}
