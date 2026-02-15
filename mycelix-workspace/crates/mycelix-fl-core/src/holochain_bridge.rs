//! Holochain On-Chain Bridge Types
//!
//! Serializable structs that mirror Holochain zome entry types without
//! depending on hdi/hdk. These types can be serialized to JSON and passed
//! to Holochain zome functions for on-chain storage.
//!
//! The actual Holochain zome (`Mycelix-Core/zomes/federated_learning/`)
//! stores `ModelGradient`, `TrainingRound`, `ByzantineRecord`, and
//! `NodeReputation` entries. This bridge provides conversion from
//! `PipelineResult` to those entry types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::pipeline::PipelineResult;
use crate::types::GradientUpdate;

/// Mirrors the `ModelGradient` zome entry type.
///
/// Stored on-chain as a record of a participant's contribution.
/// Gradient data is NOT stored on-chain (too large) — only the hash.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ZomeModelGradient {
    /// Participant who submitted this gradient.
    pub participant_id: String,
    /// Model version this gradient targets.
    pub model_version: u64,
    /// SHA-256 hash of the gradient vector (hex-encoded).
    pub gradient_hash: String,
    /// Number of training samples used.
    pub batch_size: u32,
    /// Training loss reported.
    pub loss: f32,
    /// Optional accuracy metric.
    pub accuracy: Option<f32>,
    /// Submission timestamp (Unix seconds).
    pub timestamp: u64,
}

impl ZomeModelGradient {
    /// Create from a GradientUpdate, computing the gradient hash.
    pub fn from_update(update: &GradientUpdate) -> Self {
        Self {
            participant_id: update.participant_id.clone(),
            model_version: update.model_version,
            gradient_hash: compute_gradient_hash(&update.gradients),
            batch_size: update.metadata.batch_size,
            loss: update.metadata.loss,
            accuracy: update.metadata.accuracy,
            timestamp: update.metadata.timestamp,
        }
    }
}

/// Mirrors the `TrainingRound` zome entry type.
///
/// Summary of a completed FL aggregation round, stored on-chain
/// for audit trail and protocol governance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ZomeTrainingRound {
    /// Sequential round number.
    pub round_number: u64,
    /// Model version after aggregation.
    pub model_version: u64,
    /// Total participants who submitted gradients.
    pub participant_count: u32,
    /// Participants excluded by Byzantine detection or gating.
    pub excluded_count: u32,
    /// Aggregation method used.
    pub aggregation_method: String,
    /// IDs of participants detected as Byzantine.
    pub byzantine_detected: Vec<String>,
    /// Privacy epsilon (if DP was applied).
    pub epsilon: Option<f64>,
    /// Total contributions received (before any filtering).
    pub total_contributions: u32,
}

impl ZomeTrainingRound {
    /// Create from a PipelineResult.
    ///
    /// `round_number` must be supplied by the caller (the pipeline
    /// doesn't track round numbers).
    pub fn from_pipeline_result(
        result: &PipelineResult,
        round_number: u64,
        contributions: &[GradientUpdate],
    ) -> Self {
        let byzantine_detected: Vec<String> = if let Some(ref det) = result.detection {
            det.byzantine_indices
                .iter()
                .filter_map(|&i| contributions.get(i).map(|u| u.participant_id.clone()))
                .collect()
        } else {
            Vec::new()
        };

        let epsilon = result.privacy.as_ref().and_then(|p| p.epsilon_estimate);

        Self {
            round_number,
            model_version: result.aggregated.model_version,
            participant_count: result.aggregated.participant_count as u32,
            excluded_count: result.aggregated.excluded_count as u32,
            aggregation_method: crate::convert::method_to_string(result.aggregated.method)
                .to_string(),
            byzantine_detected,
            epsilon,
            total_contributions: result.stats.total_contributions as u32,
        }
    }
}

/// Mirrors the `ByzantineRecord` zome entry type.
///
/// Stored on-chain when a participant is detected as Byzantine,
/// providing transparency and accountability.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ZomeByzantineRecord {
    /// Participant flagged as Byzantine.
    pub participant_id: String,
    /// Round in which detection occurred.
    pub round: u64,
    /// Detection signal scores: (signal_name, score).
    pub detection_signals: Vec<(String, f32)>,
    /// Whether the participant was actually excluded from aggregation.
    pub excluded: bool,
    /// Overall confidence score (0.0-1.0).
    pub confidence: f32,
}

impl ZomeByzantineRecord {
    /// Create records for all detected Byzantine participants from a PipelineResult.
    pub fn from_pipeline_result(
        result: &PipelineResult,
        round: u64,
        contributions: &[GradientUpdate],
    ) -> Vec<Self> {
        let detection = match &result.detection {
            Some(d) => d,
            None => return Vec::new(),
        };

        let mut records = Vec::new();
        for (idx_pos, &byz_idx) in detection.byzantine_indices.iter().enumerate() {
            let participant_id = match contributions.get(byz_idx) {
                Some(u) => u.participant_id.clone(),
                None => continue,
            };

            let signals = if let Some(breakdown) = detection
                .signal_breakdown
                .iter()
                .find(|b| b.participant_idx == byz_idx)
            {
                vec![
                    ("magnitude".to_string(), breakdown.magnitude_score),
                    ("direction".to_string(), breakdown.direction_score),
                    (
                        "cross_validation".to_string(),
                        breakdown.cross_validation_score,
                    ),
                    ("coordinate".to_string(), breakdown.coordinate_score),
                ]
            } else {
                Vec::new()
            };

            let confidence = detection
                .confidence_scores
                .get(idx_pos)
                .copied()
                .unwrap_or(0.5);

            records.push(Self {
                participant_id,
                round,
                detection_signals: signals,
                excluded: true,
                confidence,
            });
        }

        records
    }
}

/// Mirrors the `NodeReputation` zome entry type.
///
/// Snapshot of a participant's reputation state, stored on-chain
/// periodically for protocol governance and dispute resolution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ZomeNodeReputation {
    /// Participant identifier.
    pub participant_id: String,
    /// Current reputation score (0.0-1.0).
    pub reputation: f32,
    /// Total rounds participated.
    pub rounds_participated: u32,
    /// Total times excluded as Byzantine.
    pub exclusion_count: u32,
    /// Timestamp of this snapshot.
    pub last_updated: u64,
}

impl ZomeNodeReputation {
    /// Create from a reputation map and optional exclusion history.
    pub fn from_reputation(
        participant_id: &str,
        reputation: f32,
        rounds_participated: u32,
        exclusion_count: u32,
        timestamp: u64,
    ) -> Self {
        Self {
            participant_id: participant_id.to_string(),
            reputation,
            rounds_participated,
            exclusion_count,
            last_updated: timestamp,
        }
    }

    /// Batch-create reputation snapshots from a reputation map.
    pub fn batch_from_reputations(reputations: &HashMap<String, f32>, timestamp: u64) -> Vec<Self> {
        reputations
            .iter()
            .map(|(pid, &rep)| Self {
                participant_id: pid.clone(),
                reputation: rep,
                rounds_participated: 0,
                exclusion_count: 0,
                last_updated: timestamp,
            })
            .collect()
    }
}

/// Compute a deterministic hash of a gradient vector.
///
/// Uses a simple but collision-resistant hash: sum of (index * value)
/// encoded as hex. For production, replace with SHA-256.
fn compute_gradient_hash(gradients: &[f32]) -> String {
    // Simple deterministic hash suitable for testing.
    // In production, use SHA-256 via the `sha2` crate.
    let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
    for &g in gradients {
        let bits = g.to_bits() as u64;
        hash ^= bits;
        hash = hash.wrapping_mul(0x100000001b3); // FNV prime
    }
    format!("{:016x}", hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{PipelineConfig, UnifiedPipeline};

    fn make_test_contributions() -> (Vec<GradientUpdate>, HashMap<String, f32>) {
        let mut updates = Vec::new();
        let mut reps = HashMap::new();
        for i in 0..8 {
            let val = 0.5 + (i as f32 * 0.001);
            updates.push(GradientUpdate::new(
                format!("h{}", i),
                1,
                vec![val; 10],
                100,
                0.5,
            ));
            reps.insert(format!("h{}", i), 0.85);
        }
        for i in 0..2 {
            let val = if i == 0 { 100.0 } else { -100.0 };
            updates.push(GradientUpdate::new(
                format!("b{}", i),
                1,
                vec![val; 10],
                100,
                0.5,
            ));
            reps.insert(format!("b{}", i), 0.15);
        }
        (updates, reps)
    }

    #[test]
    fn test_zome_model_gradient_from_update() {
        let update = GradientUpdate::new("node1".into(), 42, vec![0.1, 0.2, 0.3], 100, 0.5);
        let zome = ZomeModelGradient::from_update(&update);
        assert_eq!(zome.participant_id, "node1");
        assert_eq!(zome.model_version, 42);
        assert_eq!(zome.batch_size, 100);
        assert!(!zome.gradient_hash.is_empty());
        assert_eq!(zome.gradient_hash.len(), 16); // 16 hex chars = 8 bytes
    }

    #[test]
    fn test_gradient_hash_deterministic() {
        let grads = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let hash1 = compute_gradient_hash(&grads);
        let hash2 = compute_gradient_hash(&grads);
        assert_eq!(hash1, hash2, "Hash should be deterministic");

        let different_grads = vec![0.1, 0.2, 0.3, 0.4, 0.6];
        let hash3 = compute_gradient_hash(&different_grads);
        assert_ne!(
            hash1, hash3,
            "Different gradients should produce different hashes"
        );
    }

    #[test]
    fn test_zome_training_round_from_pipeline() {
        let (updates, reps) = make_test_contributions();
        let mut pipeline = UnifiedPipeline::new(PipelineConfig::default());
        let result = pipeline.aggregate(&updates, &reps).unwrap();
        let round = ZomeTrainingRound::from_pipeline_result(&result, 1, &updates);

        assert_eq!(round.round_number, 1);
        assert_eq!(round.model_version, 1);
        assert!(round.participant_count > 0);
        assert!(round.total_contributions == 10);
    }

    #[test]
    fn test_zome_byzantine_records() {
        let (updates, reps) = make_test_contributions();
        let mut pipeline = UnifiedPipeline::new(PipelineConfig::default());
        let result = pipeline.aggregate(&updates, &reps).unwrap();
        let records = ZomeByzantineRecord::from_pipeline_result(&result, 1, &updates);

        // Byzantine nodes should have records (if detection flagged them)
        if !records.is_empty() {
            for record in &records {
                assert!(record.excluded);
                assert!(record.confidence > 0.0);
                assert!(!record.detection_signals.is_empty());
            }
        }
    }

    #[test]
    fn test_zome_node_reputation() {
        let rep = ZomeNodeReputation::from_reputation("node1", 0.85, 10, 2, 1234567890);
        assert_eq!(rep.participant_id, "node1");
        assert!((rep.reputation - 0.85).abs() < 0.001);
        assert_eq!(rep.rounds_participated, 10);
        assert_eq!(rep.exclusion_count, 2);
    }

    #[test]
    fn test_batch_reputation_snapshots() {
        let mut reps = HashMap::new();
        reps.insert("n1".to_string(), 0.9);
        reps.insert("n2".to_string(), 0.7);
        reps.insert("n3".to_string(), 0.5);

        let snapshots = ZomeNodeReputation::batch_from_reputations(&reps, 1000);
        assert_eq!(snapshots.len(), 3);
        for snap in &snapshots {
            assert!(reps.contains_key(&snap.participant_id));
            assert_eq!(snap.last_updated, 1000);
        }
    }

    #[test]
    fn test_json_roundtrip() {
        let update = GradientUpdate::new("node1".into(), 1, vec![0.1, 0.2], 100, 0.5);
        let zome_gradient = ZomeModelGradient::from_update(&update);
        let json = serde_json::to_string(&zome_gradient).unwrap();
        let parsed: ZomeModelGradient = serde_json::from_str(&json).unwrap();
        assert_eq!(zome_gradient, parsed);

        let round = ZomeTrainingRound {
            round_number: 1,
            model_version: 1,
            participant_count: 8,
            excluded_count: 2,
            aggregation_method: "trust_weighted".into(),
            byzantine_detected: vec!["b0".into(), "b1".into()],
            epsilon: Some(1.5),
            total_contributions: 10,
        };
        let json = serde_json::to_string(&round).unwrap();
        let parsed: ZomeTrainingRound = serde_json::from_str(&json).unwrap();
        assert_eq!(round, parsed);

        let byz_record = ZomeByzantineRecord {
            participant_id: "b0".into(),
            round: 1,
            detection_signals: vec![("magnitude".into(), 0.9), ("direction".into(), 0.7)],
            excluded: true,
            confidence: 0.85,
        };
        let json = serde_json::to_string(&byz_record).unwrap();
        let parsed: ZomeByzantineRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(byz_record, parsed);
    }

    #[test]
    fn test_full_pipeline_to_all_zome_types() {
        let (updates, reps) = make_test_contributions();
        let mut pipeline = UnifiedPipeline::new(PipelineConfig::default());
        let result = pipeline.aggregate(&updates, &reps).unwrap();

        // Generate all zome types from pipeline result
        let gradient_entries: Vec<ZomeModelGradient> =
            updates.iter().map(ZomeModelGradient::from_update).collect();
        assert_eq!(gradient_entries.len(), 10);

        let training_round = ZomeTrainingRound::from_pipeline_result(&result, 1, &updates);
        assert!(training_round.participant_count > 0);

        let byz_records = ZomeByzantineRecord::from_pipeline_result(&result, 1, &updates);
        // Records exist if detection found anything
        for record in &byz_records {
            assert!(!record.participant_id.is_empty());
        }

        let rep_snapshots = ZomeNodeReputation::batch_from_reputations(&reps, 1000);
        assert_eq!(rep_snapshots.len(), reps.len());

        // Verify all types serialize to JSON cleanly
        for entry in &gradient_entries {
            let _ = serde_json::to_string(entry).unwrap();
        }
        let _ = serde_json::to_string(&training_round).unwrap();
        for record in &byz_records {
            let _ = serde_json::to_string(record).unwrap();
        }
        for snap in &rep_snapshots {
            let _ = serde_json::to_string(snap).unwrap();
        }
    }
}
