// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Query Engine
//!
//! Main query interface with filtering, sorting, and pagination

use crate::{
    Result,
    claims::DesciClaim,
    query::{ClaimIndex, QueryFilter, SortBy, SortOrder},
    storage::StorageBackend,
};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, instrument, warn};
use uuid::Uuid;

/// Query result
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Matched claims
    pub claims: Vec<DesciClaim>,

    /// Total number of matches (before pagination)
    pub total_count: usize,

    /// Current page (if paginated)
    pub page: usize,

    /// Execution time in milliseconds
    pub execution_time_ms: u64,

    /// Number of retrieval errors encountered (claims that could not be loaded)
    pub retrieval_errors: usize,
}

/// Query engine
pub struct QueryEngine {
    /// Index for fast lookups
    index: Arc<RwLock<ClaimIndex>>,

    /// Storage backend
    storage: Arc<dyn StorageBackend>,
}

impl QueryEngine {
    /// Create a new query engine
    pub fn new(storage: Arc<dyn StorageBackend>) -> Self {
        Self {
            index: Arc::new(RwLock::new(ClaimIndex::new())),
            storage,
        }
    }

    /// Add a claim to the index
    pub async fn add_claim(&self, claim: &DesciClaim) {
        let mut index = self.index.write().await;
        index.add_claim(claim);
    }

    /// Remove a claim from the index
    pub async fn remove_claim(&self, claim: &DesciClaim) {
        let mut index = self.index.write().await;
        index.remove_claim(claim);
    }

    /// Rebuild index from storage
    pub async fn rebuild_index(&self) -> Result<usize> {
        // This is a placeholder - in a real implementation, we'd need
        // a way to iterate over all claims in storage
        // For now, just clear the index
        let mut index = self.index.write().await;
        index.clear();
        Ok(0)
    }

    /// Execute a query
    #[instrument(skip(self, filter), fields(candidate_count, matched_count, errors))]
    pub async fn query(&self, filter: &QueryFilter) -> Result<QueryResult> {
        let start = std::time::Instant::now();

        // Get candidate claim IDs from index
        let candidate_ids = self.get_candidate_ids(filter).await?;
        debug!(
            candidate_count = candidate_ids.len(),
            "Retrieved candidate IDs from index"
        );

        // Retrieve claims from storage
        let mut claims = Vec::new();
        let mut retrieval_errors = 0usize;

        for id in &candidate_ids {
            match self.storage.retrieve(&id.to_string()).await {
                Ok(claim) => {
                    if self.matches_filter(&claim, filter) {
                        claims.push(claim);
                    }
                }
                Err(e) => {
                    // Log the error with context instead of silently continuing
                    retrieval_errors += 1;
                    warn!(
                        claim_id = %id,
                        error = %e,
                        error_count = retrieval_errors,
                        "Failed to retrieve claim from storage - claim may be corrupted or deleted"
                    );
                    // Continue processing other claims but track the error
                    continue;
                }
            }
        }

        // Log summary if there were errors
        if retrieval_errors > 0 {
            warn!(
                total_candidates = candidate_ids.len(),
                retrieval_errors = retrieval_errors,
                successful_retrievals = claims.len(),
                "Query completed with retrieval errors - some indexed claims could not be loaded"
            );
        }

        let total_count = claims.len();

        // Sort
        if let Some(sort_by) = filter.sort_by {
            let order = filter.sort_order.unwrap_or(SortOrder::Descending);
            self.sort_claims(&mut claims, sort_by, order);
        }

        // Apply pagination
        let offset = filter.offset.unwrap_or(0);
        let limit = filter.limit.unwrap_or(usize::MAX);

        let paginated_claims: Vec<_> = claims.into_iter().skip(offset).take(limit).collect();

        let execution_time_ms = start.elapsed().as_millis() as u64;

        debug!(
            matched_count = total_count,
            returned_count = paginated_claims.len(),
            retrieval_errors = retrieval_errors,
            execution_time_ms = execution_time_ms,
            "Query execution completed"
        );

        Ok(QueryResult {
            claims: paginated_claims,
            total_count,
            page: offset / limit.max(1),
            execution_time_ms,
            retrieval_errors,
        })
    }

    /// Get candidate claim IDs based on indexed filters
    async fn get_candidate_ids(&self, filter: &QueryFilter) -> Result<Vec<Uuid>> {
        let index = self.index.read().await;

        let mut candidates = index.all();

        // Apply category filter
        if let Some(category) = &filter.category {
            let category_matches = index.by_category(category);
            candidates.retain(|id| category_matches.contains(id));
        }

        // Apply tier filter
        if let Some(min_tier) = filter.min_tier {
            let tier_matches = index.by_min_tier(min_tier);
            candidates.retain(|id| tier_matches.contains(id));
        }

        // Apply keyword filters (all keywords must match)
        for keyword in &filter.keywords {
            let keyword_matches = index.by_keyword(keyword);
            candidates.retain(|id| keyword_matches.contains(id));
        }

        // Apply creator filter
        if let Some(creator) = &filter.creator {
            let creator_matches = index.by_creator(creator);
            candidates.retain(|id| creator_matches.contains(id));
        }

        Ok(candidates.into_iter().collect())
    }

    /// Check if claim matches all filter criteria
    fn matches_filter(&self, claim: &DesciClaim, filter: &QueryFilter) -> bool {
        // Check tier range
        if let Some(min_tier) = filter.min_tier {
            if claim.epistemic_tier < min_tier {
                return false;
            }
        }

        if let Some(max_tier) = filter.max_tier {
            if claim.epistemic_tier > max_tier {
                return false;
            }
        }

        // Check date range
        if let Some(from) = filter.date_from {
            if claim.created_at < from {
                return false;
            }
        }

        if let Some(to) = filter.date_to {
            if claim.created_at > to {
                return false;
            }
        }

        // Check minimum verifications
        if let Some(min_verifications) = filter.min_verifications {
            if claim.verifications.len() < min_verifications {
                return false;
            }
        }

        // Check license
        if let Some(license) = &filter.license {
            match &claim.content.license {
                Some(claim_license) if claim_license == license => (),
                _ => return false,
            }
        }

        true
    }

    /// Sort claims
    fn sort_claims(&self, claims: &mut [DesciClaim], by: SortBy, order: SortOrder) {
        match by {
            SortBy::CreatedAt => {
                claims.sort_by(|a, b| {
                    let cmp = a.created_at.cmp(&b.created_at);
                    match order {
                        SortOrder::Ascending => cmp,
                        SortOrder::Descending => cmp.reverse(),
                    }
                });
            }
            SortBy::UpdatedAt => {
                claims.sort_by(|a, b| {
                    let cmp = a.updated_at.cmp(&b.updated_at);
                    match order {
                        SortOrder::Ascending => cmp,
                        SortOrder::Descending => cmp.reverse(),
                    }
                });
            }
            SortBy::EpistemicTier => {
                claims.sort_by(|a, b| {
                    let cmp = a.epistemic_tier.cmp(&b.epistemic_tier);
                    match order {
                        SortOrder::Ascending => cmp,
                        SortOrder::Descending => cmp.reverse(),
                    }
                });
            }
            SortBy::VerificationCount => {
                claims.sort_by(|a, b| {
                    let cmp = a.verifications.len().cmp(&b.verifications.len());
                    match order {
                        SortOrder::Ascending => cmp,
                        SortOrder::Descending => cmp.reverse(),
                    }
                });
            }
            SortBy::Category => {
                claims.sort_by(|a, b| {
                    let cmp = a.content.category.cmp(&b.content.category);
                    match order {
                        SortOrder::Ascending => cmp,
                        SortOrder::Descending => cmp.reverse(),
                    }
                });
            }
        }
    }

    /// Get index statistics
    pub async fn stats(&self) -> crate::query::index::IndexStats {
        let index = self.index.read().await;
        index.stats()
    }

    /// Identify claims that hit E4/M3 status and need Quantum Archival.
    pub fn get_foundational_pending_anchors(&self) -> Result<Vec<Uuid>> {
        // Mock implementation: return a list of foundational IDs
        // In production, this would query the index for E4/M3 claims
        // that don't have a 'quantum_anchor_cid' yet.
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        claims::{ClaimContent, EpistemicTier},
        storage::MemoryStorage,
    };

    fn create_test_claim(category: &str, tier: EpistemicTier) -> DesciClaim {
        let content = ClaimContent {
            dataset_hash: "test".to_string(),
            description: "Test claim".to_string(),
            category: category.to_string(),
            keywords: vec!["test".to_string()],
            storage_ref: None,
            reproducibility_score: None,
            license: None,
        };

        DesciClaim::new(tier, content, "creator".to_string())
    }

    #[tokio::test]
    async fn test_query_engine_basic() {
        let storage = Arc::new(MemoryStorage::new());
        let engine = QueryEngine::new(storage.clone());

        let claim = create_test_claim("genomics", EpistemicTier::E2);
        storage.store(&claim).await.unwrap();
        engine.add_claim(&claim).await;

        let filter = QueryFilter::new();
        let result = engine.query(&filter).await.unwrap();

        assert_eq!(result.claims.len(), 1);
        assert_eq!(result.total_count, 1);
        assert_eq!(result.retrieval_errors, 0);
    }

    #[tokio::test]
    async fn test_query_with_category_filter() {
        let storage = Arc::new(MemoryStorage::new());
        let engine = QueryEngine::new(storage.clone());

        let claim1 = create_test_claim("genomics", EpistemicTier::E2);
        let claim2 = create_test_claim("climate", EpistemicTier::E2);

        storage.store(&claim1).await.unwrap();
        storage.store(&claim2).await.unwrap();
        engine.add_claim(&claim1).await;
        engine.add_claim(&claim2).await;

        let filter = QueryFilter::new().with_category("genomics".to_string());
        let result = engine.query(&filter).await.unwrap();

        assert_eq!(result.claims.len(), 1);
        assert_eq!(result.claims[0].content.category, "genomics");
    }

    #[tokio::test]
    async fn test_query_with_pagination() {
        let storage = Arc::new(MemoryStorage::new());
        let engine = QueryEngine::new(storage.clone());

        // Create 5 claims
        for i in 0..5 {
            let claim = create_test_claim("test", EpistemicTier::E0);
            storage.store(&claim).await.unwrap();
            engine.add_claim(&claim).await;
        }

        let filter = QueryFilter::new().with_limit(2).with_offset(0);
        let result = engine.query(&filter).await.unwrap();

        assert_eq!(result.claims.len(), 2);
        assert_eq!(result.total_count, 5);
    }

    #[tokio::test]
    async fn test_query_with_sorting() {
        let storage = Arc::new(MemoryStorage::new());
        let engine = QueryEngine::new(storage.clone());

        let claim1 = create_test_claim("test", EpistemicTier::E1);
        let claim2 = create_test_claim("test", EpistemicTier::E3);
        let claim3 = create_test_claim("test", EpistemicTier::E2);

        storage.store(&claim1).await.unwrap();
        storage.store(&claim2).await.unwrap();
        storage.store(&claim3).await.unwrap();
        engine.add_claim(&claim1).await;
        engine.add_claim(&claim2).await;
        engine.add_claim(&claim3).await;

        let filter = QueryFilter::new().with_sort(SortBy::EpistemicTier, SortOrder::Ascending);
        let result = engine.query(&filter).await.unwrap();

        assert_eq!(result.claims[0].epistemic_tier, EpistemicTier::E1);
        assert_eq!(result.claims[1].epistemic_tier, EpistemicTier::E2);
        assert_eq!(result.claims[2].epistemic_tier, EpistemicTier::E3);
    }
}
