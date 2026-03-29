// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Synchronization between Holochain DHT and external systems

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use mycelix_core_types::{KVector, TrustScore};

/// Sync status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncStatus {
    /// Not yet synced
    Pending,
    /// Currently syncing
    InProgress,
    /// Successfully synced
    Synced,
    /// Sync failed
    Failed,
    /// Sync conflict detected
    Conflict,
}

/// A trust record for syncing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustRecord {
    /// Agent ID
    pub agent_id: String,
    /// K-Vector trust representation
    pub k_vector: KVector,
    /// MATL trust score
    pub matl_score: TrustScore,
    /// Last update timestamp
    pub updated_at: i64,
    /// Source of this record
    pub source: TrustSource,
    /// Version for conflict resolution
    pub version: u64,
}

impl TrustRecord {
    /// Create a new trust record
    pub fn new(agent_id: String, k_vector: KVector, matl_score: TrustScore, source: TrustSource) -> Self {
        Self {
            agent_id,
            k_vector,
            matl_score,
            updated_at: current_timestamp(),
            source,
            version: 1,
        }
    }

    /// Update with new values
    pub fn update(&mut self, k_vector: KVector, matl_score: TrustScore) {
        self.k_vector = k_vector;
        self.matl_score = matl_score;
        self.updated_at = current_timestamp();
        self.version += 1;
    }
}

/// Source of a trust record
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrustSource {
    /// Local Holochain DHT
    HolochainDHT,
    /// External consensus system
    ExternalConsensus,
    /// PoGQ Oracle
    PoGQOracle,
    /// TCDM Tracker
    TCDMTracker,
    /// Manual override
    Manual,
}

/// Sync operation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncResult {
    /// Number of records synced
    pub synced_count: usize,
    /// Number of conflicts
    pub conflict_count: usize,
    /// Number of failures
    pub failure_count: usize,
    /// Records that had conflicts
    pub conflicts: Vec<SyncConflict>,
    /// When sync completed
    pub completed_at: i64,
    /// Total duration in milliseconds
    pub duration_ms: u64,
}

/// A sync conflict
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConflict {
    /// Agent ID with conflict
    pub agent_id: String,
    /// Local record
    pub local: TrustRecord,
    /// Remote record
    pub remote: TrustRecord,
    /// Resolved value (if auto-resolved)
    pub resolved: Option<TrustRecord>,
    /// Resolution strategy used
    pub strategy: ConflictResolution,
}

/// Conflict resolution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictResolution {
    /// Use local value
    KeepLocal,
    /// Use remote value
    KeepRemote,
    /// Use most recent value
    MostRecent,
    /// Average the values
    Average,
    /// Requires manual resolution
    Manual,
}

/// Trust store for managing trust records
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrustStore {
    /// Trust records by agent ID
    records: HashMap<String, TrustRecord>,
    /// Pending outbound sync
    outbound_pending: Vec<String>,
    /// Last sync timestamp
    last_sync: i64,
}

impl TrustStore {
    /// Create a new trust store
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a trust record
    pub fn get(&self, agent_id: &str) -> Option<&TrustRecord> {
        self.records.get(agent_id)
    }

    /// Get a mutable trust record
    pub fn get_mut(&mut self, agent_id: &str) -> Option<&mut TrustRecord> {
        self.records.get_mut(agent_id)
    }

    /// Insert or update a trust record
    pub fn upsert(&mut self, record: TrustRecord) {
        let agent_id = record.agent_id.clone();
        self.records.insert(agent_id.clone(), record);
        if !self.outbound_pending.contains(&agent_id) {
            self.outbound_pending.push(agent_id);
        }
    }

    /// Get all agent IDs
    pub fn agent_ids(&self) -> Vec<&String> {
        self.records.keys().collect()
    }

    /// Get all records
    pub fn all_records(&self) -> impl Iterator<Item = &TrustRecord> {
        self.records.values()
    }

    /// Get pending outbound records
    pub fn pending_outbound(&self) -> Vec<&TrustRecord> {
        self.outbound_pending
            .iter()
            .filter_map(|id| self.records.get(id))
            .collect()
    }

    /// Mark records as synced
    pub fn mark_synced(&mut self, agent_ids: &[String]) {
        self.outbound_pending.retain(|id| !agent_ids.contains(id));
        self.last_sync = current_timestamp();
    }

    /// Merge incoming records with conflict resolution
    pub fn merge_incoming(
        &mut self,
        incoming: Vec<TrustRecord>,
        strategy: ConflictResolution,
    ) -> Vec<SyncConflict> {
        let mut conflicts = Vec::new();

        for remote in incoming {
            if let Some(local) = self.records.get(&remote.agent_id) {
                // Potential conflict
                if local.version != remote.version || local.updated_at != remote.updated_at {
                    let resolved = resolve_conflict(local, &remote, strategy);
                    conflicts.push(SyncConflict {
                        agent_id: remote.agent_id.clone(),
                        local: local.clone(),
                        remote: remote.clone(),
                        resolved: Some(resolved.clone()),
                        strategy,
                    });
                    self.records.insert(remote.agent_id, resolved);
                }
            } else {
                // No conflict, just insert
                self.records.insert(remote.agent_id.clone(), remote);
            }
        }

        conflicts
    }

    /// Get store statistics
    pub fn stats(&self) -> TrustStoreStats {
        let record_count = self.records.len();
        let pending_count = self.outbound_pending.len();

        let avg_trust: f32 = if record_count > 0 {
            self.records.values().map(|r| r.matl_score.total).sum::<f32>() / record_count as f32
        } else {
            0.0
        };

        TrustStoreStats {
            record_count,
            pending_count,
            avg_trust,
            last_sync: self.last_sync,
        }
    }
}

/// Resolve a conflict between local and remote records
fn resolve_conflict(local: &TrustRecord, remote: &TrustRecord, strategy: ConflictResolution) -> TrustRecord {
    match strategy {
        ConflictResolution::KeepLocal => local.clone(),
        ConflictResolution::KeepRemote => remote.clone(),
        ConflictResolution::MostRecent => {
            if local.updated_at >= remote.updated_at {
                local.clone()
            } else {
                remote.clone()
            }
        }
        ConflictResolution::Average => {
            let k_vector = KVector::from_array([
                (local.k_vector.k_r + remote.k_vector.k_r) / 2.0,
                (local.k_vector.k_a + remote.k_vector.k_a) / 2.0,
                (local.k_vector.k_i + remote.k_vector.k_i) / 2.0,
                (local.k_vector.k_p + remote.k_vector.k_p) / 2.0,
                (local.k_vector.k_m + remote.k_vector.k_m) / 2.0,
                (local.k_vector.k_s + remote.k_vector.k_s) / 2.0,
                (local.k_vector.k_h + remote.k_vector.k_h) / 2.0,
                (local.k_vector.k_topo + remote.k_vector.k_topo) / 2.0,
            ]);
            let matl_score = TrustScore::new(
                (local.matl_score.pogq + remote.matl_score.pogq) / 2.0,
                (local.matl_score.tcdm + remote.matl_score.tcdm) / 2.0,
                (local.matl_score.entropy + remote.matl_score.entropy) / 2.0,
            );
            TrustRecord {
                agent_id: local.agent_id.clone(),
                k_vector,
                matl_score,
                updated_at: current_timestamp(),
                source: TrustSource::ExternalConsensus,
                version: local.version.max(remote.version) + 1,
            }
        }
        ConflictResolution::Manual => {
            // Keep local for now, will be resolved manually
            local.clone()
        }
    }
}

/// Trust store statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustStoreStats {
    pub record_count: usize,
    pub pending_count: usize,
    pub avg_trust: f32,
    pub last_sync: i64,
}

/// Get current timestamp
fn current_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_record(agent_id: &str, trust: f32) -> TrustRecord {
        let mut k = KVector::neutral();
        k.k_r = trust;
        TrustRecord::new(
            agent_id.to_string(),
            k,
            TrustScore::new(trust, trust, trust),
            TrustSource::HolochainDHT,
        )
    }

    #[test]
    fn test_trust_store_upsert() {
        let mut store = TrustStore::new();
        let record = create_test_record("agent-1", 0.8);
        store.upsert(record);

        assert!(store.get("agent-1").is_some());
        assert_eq!(store.pending_outbound().len(), 1);
    }

    #[test]
    fn test_merge_no_conflict() {
        let mut store = TrustStore::new();
        let incoming = vec![create_test_record("agent-1", 0.8)];

        let conflicts = store.merge_incoming(incoming, ConflictResolution::MostRecent);
        assert!(conflicts.is_empty());
        assert!(store.get("agent-1").is_some());
    }

    #[test]
    fn test_merge_with_conflict() {
        let mut store = TrustStore::new();
        store.upsert(create_test_record("agent-1", 0.6));

        let mut remote = create_test_record("agent-1", 0.8);
        remote.version = 2;

        let conflicts = store.merge_incoming(vec![remote], ConflictResolution::Average);
        assert_eq!(conflicts.len(), 1);

        // Should be averaged
        let resolved = store.get("agent-1").unwrap();
        assert!((resolved.matl_score.total - 0.7).abs() < 0.1);
    }

    #[test]
    fn test_mark_synced() {
        let mut store = TrustStore::new();
        store.upsert(create_test_record("agent-1", 0.8));
        store.upsert(create_test_record("agent-2", 0.7));

        assert_eq!(store.pending_outbound().len(), 2);

        store.mark_synced(&["agent-1".to_string()]);
        assert_eq!(store.pending_outbound().len(), 1);
    }
}
