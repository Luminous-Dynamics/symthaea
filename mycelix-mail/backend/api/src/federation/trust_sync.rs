// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Trust Synchronization
//!
//! Synchronize trust attestations across federated instances

use super::*;
use super::protocol::{FederationProtocol, FederationError};
use sqlx::PgPool;
use std::sync::Arc;

/// Service for synchronizing trust across federation
pub struct TrustSyncService {
    pool: PgPool,
    protocol: Arc<FederationProtocol>,
    config: TrustSyncConfig,
}

#[derive(Debug, Clone)]
pub struct TrustSyncConfig {
    /// How much to decay federated trust scores
    pub trust_decay_factor: f64,
    /// Maximum depth to follow trust chains
    pub max_chain_depth: u8,
    /// Minimum trust score to propagate
    pub min_propagation_score: f64,
    /// How long to cache federated trust
    pub cache_ttl_secs: u64,
}

impl Default for TrustSyncConfig {
    fn default() -> Self {
        Self {
            trust_decay_factor: 0.8,
            max_chain_depth: 3,
            min_propagation_score: 0.3,
            cache_ttl_secs: 3600,
        }
    }
}

impl TrustSyncService {
    pub fn new(pool: PgPool, protocol: Arc<FederationProtocol>, config: TrustSyncConfig) -> Self {
        Self { pool, protocol, config }
    }

    /// Get aggregated trust score including federated sources
    pub async fn get_aggregated_trust(
        &self,
        local_user_id: Uuid,
        target_identifier: &str,
    ) -> Result<AggregatedTrust, TrustSyncError> {
        // Get local trust first
        let local_trust = self.get_local_trust(local_user_id, target_identifier).await?;

        // Check cache for federated trust
        if let Some(cached) = self.get_cached_federated_trust(target_identifier).await? {
            return Ok(self.merge_trust(local_trust, cached));
        }

        // Query federated instances
        let federated_responses = self
            .protocol
            .query_federated_trust(target_identifier, None)
            .await;

        // Process federated attestations
        let mut federated_attestations = Vec::new();
        for (instance, response) in federated_responses {
            for attestation in response.attestations {
                // Apply trust decay based on distance
                let decayed_score = attestation.trust_level * self.config.trust_decay_factor;

                if decayed_score >= self.config.min_propagation_score {
                    federated_attestations.push(ProcessedAttestation {
                        source_instance: instance.clone(),
                        original_score: attestation.trust_level,
                        decayed_score,
                        context: attestation.context,
                        path_length: response.path.len() as u8,
                    });
                }
            }
        }

        // Cache the federated trust
        let federated_trust = FederatedTrustData {
            attestations: federated_attestations.clone(),
            aggregate_score: self.calculate_federated_aggregate(&federated_attestations),
            fetched_at: Utc::now(),
        };
        self.cache_federated_trust(target_identifier, &federated_trust).await?;

        Ok(self.merge_trust(local_trust, federated_trust))
    }

    /// Push trust update to federated instances
    pub async fn broadcast_trust_update(
        &self,
        from_user_id: Uuid,
        to_identifier: &str,
        trust_level: f64,
        context: &str,
    ) -> Result<BroadcastResult, TrustSyncError> {
        // Only broadcast significant trust changes
        if trust_level < self.config.min_propagation_score {
            return Ok(BroadcastResult {
                instances_notified: 0,
                instances_failed: 0,
                errors: vec![],
            });
        }

        // Create attestation
        let attestation = FederatedAttestation {
            from_user: format!("{}@{}", from_user_id, "local"),
            from_instance: "local".to_string(),
            trust_level,
            context: context.to_string(),
            created_at: Utc::now(),
            expires_at: None,
            signature: String::new(), // Will be signed by protocol
        };

        let update = FederationMessage::TrustUpdate(TrustUpdatePayload {
            attestation,
            action: TrustAction::Create,
        });

        // Broadcast to all connected instances with TrustSync capability
        let mut notified = 0;
        let mut failed = 0;
        let mut errors = Vec::new();

        // This is a simplified broadcast - real implementation would
        // handle this more elegantly with parallel requests
        for domain in self.get_trust_sync_instances().await {
            match self.protocol.send_message(&domain, &update).await {
                Ok(_) => notified += 1,
                Err(e) => {
                    failed += 1;
                    errors.push((domain, e.to_string()));
                }
            }
        }

        Ok(BroadcastResult {
            instances_notified: notified,
            instances_failed: failed,
            errors,
        })
    }

    /// Handle incoming trust update from federated instance
    pub async fn handle_trust_update(
        &self,
        from_instance: &str,
        update: &TrustUpdatePayload,
    ) -> Result<(), TrustSyncError> {
        // Validate the update
        if update.attestation.trust_level < 0.0 || update.attestation.trust_level > 1.0 {
            return Err(TrustSyncError::InvalidTrustLevel);
        }

        // Apply decay based on source instance trust
        let instance_trust = self.get_instance_trust(from_instance).await?;
        let effective_score = update.attestation.trust_level * instance_trust;

        // Store the federated attestation
        sqlx::query(
            r#"
            INSERT INTO federated_attestations (
                id, from_instance, from_user, to_user,
                trust_level, effective_score, context,
                created_at, expires_at, action
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (from_instance, from_user, to_user) DO UPDATE SET
                trust_level = EXCLUDED.trust_level,
                effective_score = EXCLUDED.effective_score,
                updated_at = NOW()
            "#,
        )
        .bind(Uuid::new_v4())
        .bind(from_instance)
        .bind(&update.attestation.from_user)
        .bind("target") // Would need to extract from attestation
        .bind(update.attestation.trust_level)
        .bind(effective_score)
        .bind(&update.attestation.context)
        .bind(update.attestation.created_at)
        .bind(update.attestation.expires_at)
        .bind(format!("{:?}", update.action))
        .execute(&self.pool)
        .await
        .map_err(|e| TrustSyncError::Database(e.to_string()))?;

        // Invalidate cache
        self.invalidate_cache(&update.attestation.from_user).await?;

        Ok(())
    }

    // Private helper methods

    async fn get_local_trust(
        &self,
        user_id: Uuid,
        target: &str,
    ) -> Result<LocalTrustData, TrustSyncError> {
        let result = sqlx::query_as::<_, LocalTrustRow>(
            r#"
            SELECT trust_score, context, last_interaction, attestation_count
            FROM trust_scores
            WHERE user_id = $1 AND target_identifier = $2
            "#,
        )
        .bind(user_id)
        .bind(target)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| TrustSyncError::Database(e.to_string()))?;

        Ok(result
            .map(|r| LocalTrustData {
                score: r.trust_score,
                context: r.context,
                last_interaction: r.last_interaction,
                attestation_count: r.attestation_count,
            })
            .unwrap_or_default())
    }

    async fn get_cached_federated_trust(
        &self,
        target: &str,
    ) -> Result<Option<FederatedTrustData>, TrustSyncError> {
        let result = sqlx::query_as::<_, CachedTrustRow>(
            r#"
            SELECT data, fetched_at
            FROM federated_trust_cache
            WHERE target_identifier = $1
              AND fetched_at > NOW() - INTERVAL '1 hour'
            "#,
        )
        .bind(target)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| TrustSyncError::Database(e.to_string()))?;

        Ok(result.map(|r| serde_json::from_value(r.data).ok()).flatten())
    }

    async fn cache_federated_trust(
        &self,
        target: &str,
        data: &FederatedTrustData,
    ) -> Result<(), TrustSyncError> {
        let json_data = serde_json::to_value(data)
            .map_err(|e| TrustSyncError::Serialization(e.to_string()))?;

        sqlx::query(
            r#"
            INSERT INTO federated_trust_cache (target_identifier, data, fetched_at)
            VALUES ($1, $2, NOW())
            ON CONFLICT (target_identifier) DO UPDATE SET
                data = EXCLUDED.data,
                fetched_at = NOW()
            "#,
        )
        .bind(target)
        .bind(json_data)
        .execute(&self.pool)
        .await
        .map_err(|e| TrustSyncError::Database(e.to_string()))?;

        Ok(())
    }

    async fn invalidate_cache(&self, target: &str) -> Result<(), TrustSyncError> {
        sqlx::query("DELETE FROM federated_trust_cache WHERE target_identifier = $1")
            .bind(target)
            .execute(&self.pool)
            .await
            .map_err(|e| TrustSyncError::Database(e.to_string()))?;
        Ok(())
    }

    async fn get_trust_sync_instances(&self) -> Vec<String> {
        // Would query known_instances with TrustSync capability
        vec![]
    }

    async fn get_instance_trust(&self, _instance: &str) -> Result<f64, TrustSyncError> {
        // Would look up instance reputation
        Ok(0.9)
    }

    fn calculate_federated_aggregate(&self, attestations: &[ProcessedAttestation]) -> f64 {
        if attestations.is_empty() {
            return 0.0;
        }

        let total_weight: f64 = attestations
            .iter()
            .map(|a| 1.0 / (a.path_length as f64 + 1.0))
            .sum();

        let weighted_sum: f64 = attestations
            .iter()
            .map(|a| a.decayed_score / (a.path_length as f64 + 1.0))
            .sum();

        weighted_sum / total_weight
    }

    fn merge_trust(&self, local: LocalTrustData, federated: FederatedTrustData) -> AggregatedTrust {
        // Weight local trust more heavily
        let local_weight = 0.7;
        let federated_weight = 0.3;

        let aggregate = if local.score > 0.0 && federated.aggregate_score > 0.0 {
            local.score * local_weight + federated.aggregate_score * federated_weight
        } else if local.score > 0.0 {
            local.score
        } else {
            federated.aggregate_score
        };

        AggregatedTrust {
            total_score: aggregate,
            local_score: local.score,
            federated_score: federated.aggregate_score,
            local_attestations: local.attestation_count,
            federated_attestations: federated.attestations.len(),
            sources: federated
                .attestations
                .into_iter()
                .map(|a| TrustSource {
                    instance: a.source_instance,
                    score: a.decayed_score,
                    path_length: a.path_length,
                })
                .collect(),
        }
    }
}

// ============================================================================
// Supporting Types
// ============================================================================

#[derive(Debug, Clone, Default)]
pub struct LocalTrustData {
    pub score: f64,
    pub context: Option<String>,
    pub last_interaction: Option<DateTime<Utc>>,
    pub attestation_count: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedTrustData {
    pub attestations: Vec<ProcessedAttestation>,
    pub aggregate_score: f64,
    pub fetched_at: DateTime<Utc>,
}

impl Default for FederatedTrustData {
    fn default() -> Self {
        Self {
            attestations: vec![],
            aggregate_score: 0.0,
            fetched_at: Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedAttestation {
    pub source_instance: String,
    pub original_score: f64,
    pub decayed_score: f64,
    pub context: String,
    pub path_length: u8,
}

#[derive(Debug, Clone)]
pub struct AggregatedTrust {
    pub total_score: f64,
    pub local_score: f64,
    pub federated_score: f64,
    pub local_attestations: i32,
    pub federated_attestations: usize,
    pub sources: Vec<TrustSource>,
}

#[derive(Debug, Clone)]
pub struct TrustSource {
    pub instance: String,
    pub score: f64,
    pub path_length: u8,
}

#[derive(Debug, Clone)]
pub struct BroadcastResult {
    pub instances_notified: u32,
    pub instances_failed: u32,
    pub errors: Vec<(String, String)>,
}

#[derive(Debug, sqlx::FromRow)]
struct LocalTrustRow {
    trust_score: f64,
    context: Option<String>,
    last_interaction: Option<DateTime<Utc>>,
    attestation_count: i32,
}

#[derive(Debug, sqlx::FromRow)]
struct CachedTrustRow {
    data: serde_json::Value,
    fetched_at: DateTime<Utc>,
}

#[derive(Debug, thiserror::Error)]
pub enum TrustSyncError {
    #[error("Database error: {0}")]
    Database(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Invalid trust level")]
    InvalidTrustLevel,

    #[error("Federation error: {0}")]
    Federation(#[from] FederationError),
}
