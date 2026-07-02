// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Architectural Memory with versioned schema support.

use crate::evolutionary_scaffolder::EvolutionResult;
use anyhow::Result;
use rmp_serde::{from_slice, to_vec};
use serde::{Deserialize, Serialize};
use std::hash::DefaultHasher;
use std::path::Path;
use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_hdc_store::store::{HdcStore, StoreConfig};

/// Current schema version for EvolutionResult storage
#[allow(dead_code)]
const CURRENT_SCHEMA_VERSION: u32 = 1;

/// Versioned wrapper for long-term storage
#[derive(Debug, Clone, Serialize, Deserialize)]
enum VersionedEvolutionResult {
    V0(Vec<u8>),           // Legacy format (raw bytes)
    V1(EvolutionResultV1), // Current version
}

/// Current serialized form of EvolutionResult (V1)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EvolutionResultV1 {
    pub id: u64,
    pub success_score: f32,
    pub mutation_description: String,
    pub changed_files: Vec<String>,
    pub before_code: String,
    pub after_code: String,
    pub metrics: std::collections::HashMap<String, f32>,
    pub timestamp: u64,
}

impl From<&EvolutionResult> for EvolutionResultV1 {
    fn from(result: &EvolutionResult) -> Self {
        Self {
            id: result.id,
            success_score: result.success_score,
            mutation_description: result.mutation_description.clone(),
            changed_files: result.changed_files.clone(),
            before_code: result.before_code.clone(),
            after_code: result.after_code.clone(),
            metrics: result.metrics.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }
}

impl From<EvolutionResultV1> for EvolutionResult {
    fn from(v1: EvolutionResultV1) -> Self {
        EvolutionResult {
            id: v1.id,
            success_score: v1.success_score,
            mutation_description: v1.mutation_description,
            changed_files: v1.changed_files,
            before_code: v1.before_code,
            after_code: v1.after_code,
            metrics: v1.metrics,
        }
    }
}

/// Architectural Memory with versioned schema support
pub struct ArchitecturalMemory {
    vector_store: HdcStore,
    result_db: sled::Db,
    pub top_k: usize,
    pub min_success_threshold: f32,
}

impl std::fmt::Debug for ArchitecturalMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArchitecturalMemory")
            .field("top_k", &self.top_k)
            .field("min_success_threshold", &self.min_success_threshold)
            .finish()
    }
}

impl ArchitecturalMemory {
    pub fn new(store_path: impl AsRef<Path>) -> Result<Self> {
        let base = store_path.as_ref();
        std::fs::create_dir_all(base)?;
        let vector_store = HdcStore::create(
            base.join("architectural_vectors.hdc"),
            StoreConfig::default(),
        )
        .map_err(|e| anyhow::anyhow!("HdcStore create failed: {}", e))?;
        let result_db = sled::open(base.join("architectural_results"))?;

        Ok(Self {
            vector_store,
            result_db,
            top_k: 5,
            min_success_threshold: 0.65,
        })
    }

    /// Commit with automatic versioning
    pub fn commit_evolution(
        &mut self,
        result: &EvolutionResult,
        blueprint: &ContinuousHV,
    ) -> Result<()> {
        if result.success_score < self.min_success_threshold {
            return Ok(());
        }

        // Convert to current schema version
        let v1 = EvolutionResultV1::from(result);
        let versioned = VersionedEvolutionResult::V1(v1);

        let serialized = to_vec(&versioned)?;
        let memory_id = self.make_memory_id(result);

        // Store full versioned result
        self.result_db.insert(memory_id.to_be_bytes(), serialized)?;

        // Store bound vector for fast search
        let mutation_hv = self.encode_mutation(result);
        let bound_hv = blueprint.bind(&mutation_hv);
        let bhv = bound_hv.to_binary(0.0);
        self.vector_store
            .append(memory_id, &bhv)
            .map_err(|e| anyhow::anyhow!("HdcStore append failed: {}", e))?;

        Ok(())
    }

    /// Recall with automatic schema migration
    pub fn recall_best_patterns(
        &self,
        query_blueprint: &ContinuousHV,
    ) -> Result<Vec<EvolutionResult>> {
        let bquery = query_blueprint.to_binary(0.0);
        let candidates = self.vector_store.scan_similar(&bquery, self.top_k * 2);

        let mut results = Vec::new();

        for (id, _) in candidates {
            if let Some(bytes) = self.result_db.get(id.to_be_bytes())? {
                match from_slice::<VersionedEvolutionResult>(&bytes) {
                    Ok(VersionedEvolutionResult::V1(v1)) => {
                        results.push(EvolutionResult::from(v1));
                    }
                    Ok(VersionedEvolutionResult::V0(raw)) => {
                        if let Ok(old) = from_slice::<EvolutionResult>(&raw) {
                            results.push(old);
                        }
                    }
                    Err(_) => continue, // Skip corrupted entries
                }
            }
        }

        results.sort_by(|a, b| b.success_score.total_cmp(&a.success_score));
        results.truncate(self.top_k);

        Ok(results)
    }

    fn encode_mutation(&self, result: &EvolutionResult) -> ContinuousHV {
        // Simple hash-based seeding for mutation description
        let seed = result.id;
        ContinuousHV::random(16384, seed)
    }

    fn make_memory_id(&self, result: &EvolutionResult) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        result.mutation_description.hash(&mut hasher);
        result.success_score.to_bits().hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_test_result(success_score: f32, description: &str) -> EvolutionResult {
        EvolutionResult {
            id: 42,
            success_score,
            mutation_description: description.to_string(),
            changed_files: vec!["src/lib.rs".to_string()],
            before_code: "fn old() {}".to_string(),
            after_code: "fn new() {}".to_string(),
            metrics: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_commit_and_recall_basic() {
        let tmp = TempDir::new().unwrap();
        let mut memory = ArchitecturalMemory::new(tmp.path()).unwrap();

        let result = make_test_result(0.92, "Improved error handling in compiler");
        let blueprint = ContinuousHV::random(16384, 7);

        memory.commit_evolution(&result, &blueprint).unwrap();
        let recalled = memory.recall_best_patterns(&blueprint).unwrap();

        assert_eq!(recalled.len(), 1);
        assert_eq!(recalled[0].success_score, 0.92);
        assert!(recalled[0].mutation_description.contains("error handling"));
    }
}
