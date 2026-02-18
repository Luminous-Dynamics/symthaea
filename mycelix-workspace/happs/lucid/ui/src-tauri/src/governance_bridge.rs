//! Governance bridge for LUCID desktop application.
//!
//! Receives Phi-weighted voting data from the frontend (which calls Holochain
//! via TypeScript SDK) and caches it locally. Provides governance stats to
//! other Tauri modules (e.g., Symthaea bridge, mind bridge).
//!
//! Architecture: Frontend (TS SDK → Holochain) → Tauri command → local cache
//! No direct Holochain client — all zome calls go through the browser.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tauri::State;
use tokio::sync::Mutex;

// ============================================================================
// STATE
// ============================================================================

/// Governance state cached in the Tauri app.
pub struct GovernanceState {
    /// Cached Phi-weighted tallies by proposal ID.
    tallies: Arc<Mutex<HashMap<String, PhiTallyCache>>>,
}

impl GovernanceState {
    pub fn new() -> Self {
        Self {
            tallies: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

// ============================================================================
// TYPES
// ============================================================================

/// Cached Phi-weighted tally data from governance voting zome.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhiTallyCache {
    pub proposal_id: String,
    pub phi_votes_for: f64,
    pub phi_votes_against: f64,
    pub phi_abstentions: f64,
    pub raw_votes_for: u32,
    pub raw_votes_against: u32,
    pub raw_abstentions: u32,
    pub average_phi: f64,
    pub total_phi_weight: f64,
    pub eligible_voters: u32,
    pub quorum_reached: bool,
    pub approved: bool,
    /// Votes with real Phi data (Attested or Snapshot provenance).
    pub phi_enhanced_count: u32,
    /// Votes without Phi data (reputation-only).
    pub reputation_only_count: u32,
    /// Fraction of votes with verified consciousness data (0.0–1.0).
    pub phi_coverage: f64,
    /// Timestamp when this tally was last updated (ms since epoch).
    pub updated_at: u64,
}

/// Summary stats across all cached proposals.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GovernanceStats {
    /// Number of proposals with cached tallies.
    pub proposal_count: usize,
    /// Average phi_coverage across all proposals.
    pub avg_phi_coverage: f64,
    /// Total votes with attested Phi data.
    pub total_phi_enhanced: u32,
    /// Total votes without Phi data.
    pub total_reputation_only: u32,
    /// Overall phi coverage (total_phi_enhanced / total_votes).
    pub overall_phi_coverage: f64,
}

/// Error type for governance bridge commands.
#[derive(Debug, thiserror::Error)]
pub enum GovernanceError {
    #[error("Proposal not found: {0}")]
    ProposalNotFound(String),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

impl Serialize for GovernanceError {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

// ============================================================================
// COMMANDS
// ============================================================================

/// Cache a Phi-weighted tally received from the frontend.
///
/// Called by the frontend after it fetches a tally from the governance
/// voting zome via the TypeScript SDK.
#[tauri::command]
pub async fn cache_phi_tally(
    tally: PhiTallyCache,
    state: State<'_, GovernanceState>,
) -> Result<(), GovernanceError> {
    if tally.proposal_id.is_empty() {
        return Err(GovernanceError::InvalidInput(
            "proposal_id cannot be empty".into(),
        ));
    }
    if !(0.0..=1.0).contains(&tally.phi_coverage) {
        return Err(GovernanceError::InvalidInput(format!(
            "phi_coverage must be 0.0–1.0, got {}",
            tally.phi_coverage
        )));
    }

    let mut tallies = state.tallies.lock().await;
    tallies.insert(tally.proposal_id.clone(), tally);
    Ok(())
}

/// Get a cached Phi-weighted tally for a specific proposal.
#[tauri::command]
pub async fn get_phi_tally(
    proposal_id: String,
    state: State<'_, GovernanceState>,
) -> Result<PhiTallyCache, GovernanceError> {
    let tallies = state.tallies.lock().await;
    tallies
        .get(&proposal_id)
        .cloned()
        .ok_or_else(|| GovernanceError::ProposalNotFound(proposal_id))
}

/// Get aggregate governance stats across all cached proposals.
#[tauri::command]
pub async fn get_governance_stats(
    state: State<'_, GovernanceState>,
) -> Result<GovernanceStats, GovernanceError> {
    let tallies = state.tallies.lock().await;
    let proposal_count = tallies.len();

    if proposal_count == 0 {
        return Ok(GovernanceStats {
            proposal_count: 0,
            avg_phi_coverage: 0.0,
            total_phi_enhanced: 0,
            total_reputation_only: 0,
            overall_phi_coverage: 0.0,
        });
    }

    let mut total_phi_enhanced: u32 = 0;
    let mut total_reputation_only: u32 = 0;
    let mut sum_coverage: f64 = 0.0;

    for tally in tallies.values() {
        total_phi_enhanced += tally.phi_enhanced_count;
        total_reputation_only += tally.reputation_only_count;
        sum_coverage += tally.phi_coverage;
    }

    let total_votes = total_phi_enhanced + total_reputation_only;
    let overall_phi_coverage = if total_votes > 0 {
        total_phi_enhanced as f64 / total_votes as f64
    } else {
        0.0
    };

    Ok(GovernanceStats {
        proposal_count,
        avg_phi_coverage: sum_coverage / proposal_count as f64,
        total_phi_enhanced,
        total_reputation_only,
        overall_phi_coverage,
    })
}

/// Clear all cached tallies.
#[tauri::command]
pub async fn clear_governance_cache(
    state: State<'_, GovernanceState>,
) -> Result<usize, GovernanceError> {
    let mut tallies = state.tallies.lock().await;
    let count = tallies.len();
    tallies.clear();
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_tally_serde_roundtrip() {
        let tally = PhiTallyCache {
            proposal_id: "prop-001".into(),
            phi_votes_for: 12.5,
            phi_votes_against: 3.2,
            phi_abstentions: 1.0,
            raw_votes_for: 15,
            raw_votes_against: 5,
            raw_abstentions: 2,
            average_phi: 0.65,
            total_phi_weight: 16.7,
            eligible_voters: 30,
            quorum_reached: true,
            approved: true,
            phi_enhanced_count: 18,
            reputation_only_count: 4,
            phi_coverage: 0.818,
            updated_at: 1708000000000,
        };

        let json = serde_json::to_string(&tally).unwrap();
        let decoded: PhiTallyCache = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.proposal_id, "prop-001");
        assert!((decoded.phi_coverage - 0.818).abs() < 1e-6);
        assert_eq!(decoded.phi_enhanced_count, 18);
    }

    #[test]
    fn test_governance_stats_serde() {
        let stats = GovernanceStats {
            proposal_count: 5,
            avg_phi_coverage: 0.72,
            total_phi_enhanced: 45,
            total_reputation_only: 15,
            overall_phi_coverage: 0.75,
        };

        let json = serde_json::to_string(&stats).unwrap();
        assert!(json.contains("\"avgPhiCoverage\""));
        assert!(json.contains("\"overallPhiCoverage\""));
    }
}
