// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! CLI state file — tracks the user's acknowledgement and staged
//! local intents (DIDs they want to create, issuers they want to
//! classify) before the conductor round-trip is available.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CliState {
    /// ISO-8601 timestamp when user acknowledged the threat model.
    /// `None` means first run not complete yet.
    pub disclosure_acknowledged_at: Option<String>,

    /// Legal DIDs the user intends to create, queued locally until
    /// the conductor call fires.
    pub staged_legal_dids: Vec<StagedDid>,

    /// Issuer classifications staged locally.
    pub staged_issuer_classifications: Vec<StagedIssuerClassification>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StagedDid {
    pub label: Option<String>,
    pub staged_at: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StagedIssuerClassification {
    pub issuer_did: String,
    /// `sovereign` | `regulated` | `peer`
    pub tier: String,
    pub rationale: Option<String>,
    pub staged_at: String,
}

impl CliState {
    const FILE_NAME: &'static str = "state.json";

    pub fn load_or_default(dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let path = dir.join(Self::FILE_NAME);
        if !path.exists() {
            return Ok(CliState::default());
        }
        let bytes = fs::read(&path)?;
        let parsed: CliState = serde_json::from_slice(&bytes)?;
        Ok(parsed)
    }

    pub fn save(&self, dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        fs::create_dir_all(dir)?;
        let path = dir.join(Self::FILE_NAME);
        let bytes = serde_json::to_vec_pretty(self)?;
        fs::write(&path, bytes)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn tmp_dir(tag: &str) -> std::path::PathBuf {
        let mut p = env::temp_dir();
        p.push(format!(
            "lawful-id-test-{}-{}",
            tag,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        p
    }

    #[test]
    fn default_state_has_no_acknowledgement() {
        let s = CliState::default();
        assert!(s.disclosure_acknowledged_at.is_none());
        assert!(s.staged_legal_dids.is_empty());
    }

    #[test]
    fn load_missing_file_returns_default() {
        let dir = tmp_dir("load-missing");
        let s = CliState::load_or_default(&dir).expect("load");
        assert!(s.disclosure_acknowledged_at.is_none());
    }

    #[test]
    fn save_then_load_round_trips() {
        let dir = tmp_dir("roundtrip");
        let mut s = CliState::default();
        s.disclosure_acknowledged_at = Some("2026-04-18T12:00:00Z".to_string());
        s.staged_legal_dids.push(StagedDid {
            label: Some("passport holder".to_string()),
            staged_at: "2026-04-18T12:05:00Z".to_string(),
        });
        s.save(&dir).expect("save");
        let reloaded = CliState::load_or_default(&dir).expect("load");
        assert_eq!(
            reloaded.disclosure_acknowledged_at,
            s.disclosure_acknowledged_at
        );
        assert_eq!(reloaded.staged_legal_dids.len(), 1);
        assert_eq!(
            reloaded.staged_legal_dids[0].label.as_deref(),
            Some("passport holder")
        );
        // Cleanup.
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn staged_issuer_round_trips() {
        let dir = tmp_dir("issuer");
        let mut s = CliState::default();
        s.staged_issuer_classifications
            .push(StagedIssuerClassification {
                issuer_did: "did:web:state.gov".to_string(),
                tier: "sovereign".to_string(),
                rationale: Some("primary US sovereign identity issuer".to_string()),
                staged_at: "2026-04-18T12:10:00Z".to_string(),
            });
        s.save(&dir).expect("save");
        let reloaded = CliState::load_or_default(&dir).expect("load");
        assert_eq!(reloaded.staged_issuer_classifications.len(), 1);
        assert_eq!(reloaded.staged_issuer_classifications[0].tier, "sovereign");
        let _ = fs::remove_dir_all(&dir);
    }
}
