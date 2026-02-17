//! Holochain integration types and bridge helpers for mycelix-fl.
//!
//! Provides single-source-of-truth types for converting between the
//! mycelix-fl pipeline's internal representation and the Holochain zome's
//! entry types (from `federated_learning_integrity`).
//!
//! The coordinator's `pipeline.rs` imports from this module to ensure that
//! bridge conversions are defined once and used consistently.

use serde::{Deserialize, Serialize};

// ============================================================================
// Holochain Bridge Types
// ============================================================================

/// Detection summary for cross-boundary serialization.
///
/// Used in `ZomeAggregationReveal` to communicate which participants were
/// flagged as Byzantine during aggregation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionSummary {
    /// Map of participant_id → detection confidence score
    pub flagged_nodes: Vec<(String, f32)>,
    /// Names of detection layers that ran
    pub detection_layers_used: Vec<String>,
    /// Total number of participants checked
    pub total_checked: usize,
    /// Total number of participants flagged
    pub total_flagged: usize,
}

impl DetectionSummary {
    pub fn empty() -> Self {
        Self {
            flagged_nodes: Vec::new(),
            detection_layers_used: Vec::new(),
            total_checked: 0,
            total_flagged: 0,
        }
    }
}

/// Aggregation commitment — the zome-level representation of a commit phase entry.
///
/// Mirrors the `AggregationCommitment` integrity entry type but uses primitive
/// types compatible across both Holochain WASM and standard Rust contexts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZomeAggregationCommitment {
    /// Training round number
    pub round: u64,
    /// Agent public key of the committing validator (string form)
    pub aggregator: String,
    /// SHA-256 commitment hash of the aggregated result
    pub commitment_hash: String,
    /// Aggregation method used (e.g. "FedAvgHV")
    pub method: String,
    /// Number of gradients included in aggregation
    pub gradient_count: u32,
    /// Number of gradients excluded by Byzantine detection
    pub excluded_count: u32,
    /// Unix timestamp when commitment was created
    pub committed_at: i64,
    /// Trust score of the aggregating validator
    pub aggregator_trust_score: f32,
}

/// Aggregation reveal — the zome-level representation of a reveal phase entry.
///
/// Carries the actual aggregated result plus Shapley value attribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZomeAggregationReveal {
    /// Training round number
    pub round: u64,
    /// Agent public key of the revealing validator
    pub aggregator: String,
    /// Aggregated HV16 binary data (2KB)
    pub result_data: Vec<u8>,
    /// SHA-256 hash of result_data (must match prior commitment)
    pub result_hash: String,
    /// Detection summary for audit trail
    pub detection_summary: DetectionSummary,
    /// Shapley values: participant_id → contribution weight
    pub shapley_values: Vec<(String, f32)>,
    /// Unix timestamp when reveal was submitted
    pub revealed_at: i64,
}

// ============================================================================
// Bridge Helper Functions
// ============================================================================

/// Build a `ZomeAggregationCommitment` from pipeline result fields.
///
/// This is the single factory function for creating commitment bridge types.
/// Using this function ensures all fields are populated consistently.
#[allow(clippy::too_many_arguments)]
pub fn to_zome_commitment(
    round: u64,
    aggregator: &str,
    commitment_hash: &str,
    method: &str,
    gradient_count: u32,
    excluded_count: u32,
    aggregator_trust_score: f32,
    committed_at: i64,
) -> ZomeAggregationCommitment {
    ZomeAggregationCommitment {
        round,
        aggregator: aggregator.to_string(),
        commitment_hash: commitment_hash.to_string(),
        method: method.to_string(),
        gradient_count,
        excluded_count,
        committed_at,
        aggregator_trust_score,
    }
}

/// Build a `ZomeAggregationReveal` from pipeline result fields.
#[allow(clippy::too_many_arguments)]
pub fn to_zome_reveal(
    round: u64,
    aggregator: &str,
    result_data: Vec<u8>,
    result_hash: String,
    detection_summary: DetectionSummary,
    shapley_values: Vec<(String, f32)>,
    revealed_at: i64,
) -> ZomeAggregationReveal {
    ZomeAggregationReveal {
        round,
        aggregator: aggregator.to_string(),
        result_data,
        result_hash,
        detection_summary,
        shapley_values,
        revealed_at,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_zome_commitment_fields() {
        let c = to_zome_commitment(5, "agent1", "abc123", "FedAvgHV", 10, 2, 0.85, 1000);
        assert_eq!(c.round, 5);
        assert_eq!(c.aggregator, "agent1");
        assert_eq!(c.commitment_hash, "abc123");
        assert_eq!(c.method, "FedAvgHV");
        assert_eq!(c.gradient_count, 10);
        assert_eq!(c.excluded_count, 2);
        assert!((c.aggregator_trust_score - 0.85).abs() < 0.001);
        assert_eq!(c.committed_at, 1000);
    }

    #[test]
    fn test_to_zome_reveal_fields() {
        let summary = DetectionSummary {
            flagged_nodes: vec![("bad".to_string(), 0.9)],
            detection_layers_used: vec!["CosineFilter".to_string()],
            total_checked: 5,
            total_flagged: 1,
        };
        let r = to_zome_reveal(
            5,
            "agent1",
            vec![0xFF; 2048],
            "abc123".to_string(),
            summary,
            vec![("p1".to_string(), 0.5)],
            2000,
        );
        assert_eq!(r.round, 5);
        assert_eq!(r.result_data.len(), 2048);
        assert_eq!(r.detection_summary.total_flagged, 1);
        assert_eq!(r.shapley_values.len(), 1);
    }

    #[test]
    fn test_detection_summary_serialization() {
        let summary = DetectionSummary {
            flagged_nodes: vec![("node1".to_string(), 0.95)],
            detection_layers_used: vec!["Layer1".to_string(), "Layer2".to_string()],
            total_checked: 10,
            total_flagged: 1,
        };
        let json = serde_json::to_string(&summary).expect("serialize failed");
        let deser: DetectionSummary = serde_json::from_str(&json).expect("deserialize failed");
        assert_eq!(deser.total_checked, 10);
        assert_eq!(deser.total_flagged, 1);
        assert_eq!(deser.flagged_nodes.len(), 1);
    }

    #[test]
    fn test_detection_summary_empty() {
        let s = DetectionSummary::empty();
        assert_eq!(s.total_checked, 0);
        assert_eq!(s.flagged_nodes.len(), 0);
    }

    #[test]
    fn test_commitment_serialization_roundtrip() {
        let c = to_zome_commitment(1, "agg", "hash", "FedAvgHV", 5, 0, 0.9, 999);
        let json = serde_json::to_string(&c).expect("serialize");
        let deser: ZomeAggregationCommitment = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deser.round, 1);
        assert_eq!(deser.commitment_hash, "hash");
    }
}
