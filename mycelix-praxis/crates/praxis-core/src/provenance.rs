// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Provenance tracking for models and credentials

use crate::types::{ModelHash, ModelId};
use serde::{Deserialize, Serialize};

/// Provenance record for a machine learning model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProvenance {
    /// Unique identifier for this model version
    pub model_id: ModelId,

    /// Hash of model parameters
    pub model_hash: ModelHash,

    /// Hash of parent model (if this is derived via FL)
    pub parent_hash: Option<ModelHash>,

    /// Round ID if created via federated learning
    pub round_id: Option<String>,

    /// Number of participants who contributed
    pub contributor_count: Option<u32>,

    /// Aggregation method used
    pub aggregation_method: Option<String>,

    /// Timestamp of model creation
    pub created_at: i64,

    /// Additional metadata
    pub metadata: Option<serde_json::Value>,
}

impl ModelProvenance {
    /// Create a new provenance record with explicit timestamp
    /// For WASM/Holochain, use sys_time(); for native, use chrono::Utc::now().timestamp()
    pub fn new(model_id: ModelId, model_hash: ModelHash, created_at: i64) -> Self {
        Self {
            model_id,
            model_hash,
            parent_hash: None,
            round_id: None,
            contributor_count: None,
            aggregation_method: None,
            created_at,
            metadata: None,
        }
    }

    /// Create a provenance record with current timestamp (native builds only)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_now(model_id: ModelId, model_hash: ModelHash) -> Self {
        Self::new(model_id, model_hash, chrono::Utc::now().timestamp())
    }

    /// Set parent model information
    pub fn with_parent(mut self, parent_hash: ModelHash) -> Self {
        self.parent_hash = Some(parent_hash);
        self
    }

    /// Set federated learning round information
    pub fn with_fl_round(
        mut self,
        round_id: String,
        contributor_count: u32,
        aggregation_method: String,
    ) -> Self {
        self.round_id = Some(round_id);
        self.contributor_count = Some(contributor_count);
        self.aggregation_method = Some(aggregation_method);
        self
    }
}

/// Chain of provenance linking models together
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceChain {
    pub records: Vec<ModelProvenance>,
}

impl ProvenanceChain {
    /// Create a new provenance chain
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }

    /// Add a provenance record to the chain
    pub fn add(&mut self, record: ModelProvenance) {
        self.records.push(record);
    }

    /// Get the latest model in the chain
    pub fn latest(&self) -> Option<&ModelProvenance> {
        self.records.last()
    }

    /// Verify chain integrity (each model's parent matches previous model's hash)
    pub fn verify_integrity(&self) -> bool {
        for i in 1..self.records.len() {
            if let Some(parent_hash) = &self.records[i].parent_hash {
                if parent_hash != &self.records[i - 1].model_hash {
                    return false;
                }
            }
        }
        true
    }
}

impl Default for ProvenanceChain {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provenance_creation() {
        let prov = ModelProvenance::new(
            ModelId("model-1".to_string()),
            ModelHash("hash-1".to_string()),
            1234567890i64,
        );
        assert_eq!(prov.model_id.0, "model-1");
        assert!(prov.parent_hash.is_none());
    }

    #[test]
    fn test_provenance_chain() {
        let mut chain = ProvenanceChain::new();

        let prov1 = ModelProvenance::new(
            ModelId("model-1".to_string()),
            ModelHash("hash-1".to_string()),
            1234567890i64,
        );

        let prov2 = ModelProvenance::new(
            ModelId("model-2".to_string()),
            ModelHash("hash-2".to_string()),
            1234567891i64,
        )
        .with_parent(ModelHash("hash-1".to_string()));

        chain.add(prov1);
        chain.add(prov2);

        assert_eq!(chain.records.len(), 2);
        assert!(chain.verify_integrity());
    }
}
