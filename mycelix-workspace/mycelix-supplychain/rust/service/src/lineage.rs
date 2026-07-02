// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Lineage resolution and tracking
//!
//! This module provides functionality for resolving supply chain lineage,
//! including building full provenance trees from claims.

use crate::AppState;
use claim_model::{DkgClaim, EventType, SupplyEventVC};
use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use tracing::{debug, warn};

/// Maximum depth for lineage traversal to prevent infinite loops
const MAX_LINEAGE_DEPTH: usize = 100;

/// Maximum number of claims in a lineage tree
const MAX_LINEAGE_CLAIMS: usize = 1000;

/// Resolve previous claims for a given event
pub async fn resolve_previous_claims(
    state: &Arc<AppState>,
    vc: &SupplyEventVC,
) -> Option<Vec<String>> {
    match vc.credential_subject.event_type {
        EventType::Produced => {
            // Produced events have no parents
            None
        }
        EventType::Transformed => {
            // Look up parent batches
            if let Some(prev_batch_ids) = &vc.credential_subject.prev_batch_ids {
                let parent_claims = if let Some(ref db) = state.db {
                    // Use database
                    let mut all_parents = Vec::new();
                    for batch_id in prev_batch_ids {
                        if let Ok(batch_claims) = db.get_batch_claims(batch_id).await {
                            all_parents.extend(batch_claims.into_iter().map(|c| c.id));
                        }
                    }
                    all_parents
                } else {
                    // Use in-memory
                    let claims = state.claims.read().await;
                    claims
                        .values()
                        .filter(|claim| prev_batch_ids.contains(&claim.subject.batch_id))
                        .map(|claim| claim.id.clone())
                        .collect()
                };

                debug!(
                    "Found {} parent claims for TRANSFORMED event",
                    parent_claims.len()
                );
                if !parent_claims.is_empty() {
                    return Some(parent_claims);
                }
            }
            None
        }
        EventType::Shipped | EventType::Received => {
            // Find the most recent claim for this batch
            let batch_id = &vc.credential_subject.batch_id;

            if let Some(ref db) = state.db {
                // Use database
                if let Ok(mut batch_claims) = db.get_batch_claims(batch_id).await {
                    batch_claims.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
                    if let Some(latest) = batch_claims.first() {
                        debug!("Found previous claim {} for batch {}", latest.id, batch_id);
                        return Some(vec![latest.id.clone()]);
                    }
                }
            } else {
                // Use in-memory
                let claims = state.claims.read().await;
                let mut batch_claims: Vec<_> = claims
                    .values()
                    .filter(|claim| &claim.subject.batch_id == batch_id)
                    .collect();

                batch_claims.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

                if let Some(latest) = batch_claims.first() {
                    debug!("Found previous claim {} for batch {}", latest.id, batch_id);
                    return Some(vec![latest.id.clone()]);
                }
            }
            None
        }
        EventType::Certified => {
            // Find claims for this batch or product
            let batch_id = &vc.credential_subject.batch_id;

            let related_claims: Vec<String> = if let Some(ref db) = state.db {
                // Use database
                db.get_batch_claims(batch_id)
                    .await
                    .ok()
                    .map(|claims| claims.into_iter().map(|c| c.id).collect())
                    .unwrap_or_default()
            } else {
                // Use in-memory
                let claims = state.claims.read().await;
                claims
                    .values()
                    .filter(|claim| &claim.subject.batch_id == batch_id)
                    .map(|claim| claim.id.clone())
                    .collect()
            };

            if !related_claims.is_empty() {
                debug!(
                    "Found {} related claims for CERTIFIED event",
                    related_claims.len()
                );
                return Some(related_claims);
            }
            None
        }
    }
}

/// A node in the lineage tree
#[derive(Debug, Clone)]
pub struct LineageNode {
    /// The claim ID
    pub claim_id: String,
    /// Depth in the tree (0 = root)
    pub depth: usize,
    /// Parent claim IDs
    pub parents: Vec<String>,
}

/// Result of building a lineage tree
#[derive(Debug)]
pub struct LineageTree {
    /// Root claim ID
    pub root_id: String,
    /// All claim IDs in the tree (in traversal order)
    pub claim_ids: Vec<String>,
    /// Nodes with their relationships
    pub nodes: Vec<LineageNode>,
    /// Maximum depth reached
    pub max_depth: usize,
    /// Whether the tree was truncated due to limits
    pub truncated: bool,
}

/// Build a full lineage tree for a claim
///
/// This recursively traverses the `previous_claims` field of each claim
/// to build a complete provenance tree. The traversal uses BFS to ensure
/// breadth-first ordering and includes cycle detection to prevent infinite loops.
///
/// # Arguments
/// * `state` - Application state containing claims storage
/// * `claim_id` - The root claim ID to start traversal from
///
/// # Returns
/// A vector of all claim IDs in the lineage tree, ordered by traversal
pub async fn build_lineage_tree(state: &Arc<AppState>, claim_id: &str) -> Vec<String> {
    let tree = build_lineage_tree_detailed(state, claim_id).await;
    tree.claim_ids
}

/// Build a detailed lineage tree with node relationships
///
/// This provides full tree structure information including depth,
/// parent relationships, and truncation status.
pub async fn build_lineage_tree_detailed(state: &Arc<AppState>, claim_id: &str) -> LineageTree {
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<(String, usize)> = VecDeque::new();
    let mut claim_ids: Vec<String> = Vec::new();
    let mut nodes: Vec<LineageNode> = Vec::new();
    let mut max_depth: usize = 0;
    let mut truncated = false;

    // Start with the root claim
    queue.push_back((claim_id.to_string(), 0));
    visited.insert(claim_id.to_string());

    while let Some((current_id, depth)) = queue.pop_front() {
        // Check depth limit
        if depth > MAX_LINEAGE_DEPTH {
            warn!(
                claim_id = %current_id,
                depth = depth,
                "Lineage traversal reached maximum depth, truncating"
            );
            truncated = true;
            continue;
        }

        // Check claim count limit
        if claim_ids.len() >= MAX_LINEAGE_CLAIMS {
            warn!(
                claim_id = %current_id,
                total_claims = claim_ids.len(),
                "Lineage tree reached maximum size, truncating"
            );
            truncated = true;
            break;
        }

        // Get the claim
        let claim_opt = get_claim_by_id(state, &current_id).await;

        if let Some(claim) = claim_opt {
            claim_ids.push(current_id.clone());
            max_depth = max_depth.max(depth);

            // Get previous claims (parents in the lineage)
            let parents = claim.lineage.previous_claims.clone().unwrap_or_default();

            // Create the node
            nodes.push(LineageNode {
                claim_id: current_id.clone(),
                depth,
                parents: parents.clone(),
            });

            // Queue unvisited parents for traversal
            for parent_id in parents {
                if !visited.contains(&parent_id) {
                    visited.insert(parent_id.clone());
                    queue.push_back((parent_id, depth + 1));
                }
            }

            debug!(
                claim_id = %current_id,
                depth = depth,
                parent_count = nodes.last().map(|n| n.parents.len()).unwrap_or(0),
                "Traversed lineage node"
            );
        } else {
            // Claim not found locally - it may be in the DKG
            debug!(
                claim_id = %current_id,
                "Claim not found in local storage, skipping"
            );

            // Still add it to the tree to show the reference exists
            nodes.push(LineageNode {
                claim_id: current_id.clone(),
                depth,
                parents: vec![],
            });
            claim_ids.push(current_id);
            max_depth = max_depth.max(depth);
        }
    }

    debug!(
        root_id = %claim_id,
        total_claims = claim_ids.len(),
        max_depth = max_depth,
        truncated = truncated,
        "Built lineage tree"
    );

    LineageTree {
        root_id: claim_id.to_string(),
        claim_ids,
        nodes,
        max_depth,
        truncated,
    }
}

/// Build the upstream lineage (ancestors) for a claim
///
/// Returns all claims that are ancestors of the given claim,
/// ordered from newest to oldest (closest ancestors first).
pub async fn build_upstream_lineage(state: &Arc<AppState>, claim_id: &str) -> Vec<DkgClaim> {
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<String> = VecDeque::new();
    let mut ancestors: Vec<DkgClaim> = Vec::new();

    // Get the starting claim
    if let Some(root_claim) = get_claim_by_id(state, claim_id).await {
        // Add initial parents to queue
        if let Some(parents) = &root_claim.lineage.previous_claims {
            for parent_id in parents {
                if !visited.contains(parent_id) {
                    visited.insert(parent_id.clone());
                    queue.push_back(parent_id.clone());
                }
            }
        }
    }

    // BFS through ancestors
    while let Some(current_id) = queue.pop_front() {
        if ancestors.len() >= MAX_LINEAGE_CLAIMS {
            warn!("Upstream lineage reached maximum size, truncating");
            break;
        }

        if let Some(claim) = get_claim_by_id(state, &current_id).await {
            ancestors.push(claim.clone());

            // Queue this claim's parents
            if let Some(parents) = &claim.lineage.previous_claims {
                for parent_id in parents {
                    if !visited.contains(parent_id) {
                        visited.insert(parent_id.clone());
                        queue.push_back(parent_id.clone());
                    }
                }
            }
        }
    }

    ancestors
}

/// Build the downstream lineage (descendants) for a claim
///
/// Returns all claims that reference the given claim as an ancestor,
/// ordered from oldest to newest (closest descendants first).
pub async fn build_downstream_lineage(state: &Arc<AppState>, claim_id: &str) -> Vec<DkgClaim> {
    let mut descendants: Vec<DkgClaim> = Vec::new();

    // Get all claims and find those that reference this claim
    let all_claims = get_all_claims(state).await;

    // Build a set of claim IDs that are in the lineage
    let mut lineage_ids: HashSet<String> = HashSet::new();
    lineage_ids.insert(claim_id.to_string());

    // Iterate until no new descendants are found
    let mut found_new = true;
    while found_new {
        found_new = false;

        for claim in &all_claims {
            if lineage_ids.contains(&claim.id) {
                continue;
            }

            // Check if this claim references any of our lineage claims
            if let Some(parents) = &claim.lineage.previous_claims {
                for parent_id in parents {
                    if lineage_ids.contains(parent_id) {
                        lineage_ids.insert(claim.id.clone());
                        descendants.push(claim.clone());
                        found_new = true;
                        break;
                    }
                }
            }

            if descendants.len() >= MAX_LINEAGE_CLAIMS {
                warn!("Downstream lineage reached maximum size, truncating");
                return descendants;
            }
        }
    }

    // Sort by timestamp (oldest first for downstream)
    descendants.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

    descendants
}

/// Get the full provenance chain for a batch
///
/// This combines upstream and downstream lineage to provide
/// complete supply chain visibility for a batch.
pub async fn get_batch_provenance(state: &Arc<AppState>, batch_id: &str) -> Vec<DkgClaim> {
    let mut provenance: Vec<DkgClaim> = Vec::new();
    let mut visited: HashSet<String> = HashSet::new();

    // Get all claims for this batch
    let batch_claims = if let Some(ref db) = state.db {
        db.get_batch_claims(batch_id).await.unwrap_or_default()
    } else {
        let claims = state.claims.read().await;
        claims
            .values()
            .filter(|c| c.subject.batch_id == batch_id)
            .cloned()
            .collect()
    };

    // Add batch claims
    for claim in batch_claims {
        if !visited.contains(&claim.id) {
            visited.insert(claim.id.clone());

            // Get upstream for each batch claim
            let upstream = build_upstream_lineage(state, &claim.id).await;
            for ancestor in upstream {
                if !visited.contains(&ancestor.id) {
                    visited.insert(ancestor.id.clone());
                    provenance.push(ancestor);
                }
            }

            provenance.push(claim);
        }
    }

    // Sort by timestamp
    provenance.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

    provenance
}

/// Helper function to get a claim by ID from state
async fn get_claim_by_id(state: &Arc<AppState>, claim_id: &str) -> Option<DkgClaim> {
    if let Some(ref db) = state.db {
        db.get_claim(claim_id).await.ok().flatten()
    } else {
        let claims = state.claims.read().await;
        claims.get(claim_id).cloned()
    }
}

/// Helper function to get all claims from state
async fn get_all_claims(state: &Arc<AppState>) -> Vec<DkgClaim> {
    if let Some(ref db) = state.db {
        db.get_all_claims().await.unwrap_or_default()
    } else {
        let claims = state.claims.read().await;
        claims.values().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use claim_model::{Assertion, Evidence, Facility, Lineage, Subject};
    use std::collections::HashMap;
    use tokio::sync::RwLock;

    fn create_test_claim(
        id: &str,
        batch_id: &str,
        previous_claims: Option<Vec<String>>,
    ) -> DkgClaim {
        DkgClaim {
            id: id.to_string(),
            claim_type: "SupplyChainClaim".to_string(),
            issuer: "did:mycelix:org:test".to_string(),
            subject: Subject {
                batch_id: batch_id.to_string(),
                product_id: "SKU-001".to_string(),
            },
            assertion: Assertion {
                event_type: EventType::Produced,
                quantity: Some(100.0),
                unit: Some("kg".to_string()),
                facility_id: Some("FAC-001".to_string()),
            },
            evidence: Evidence {
                vc_jwt: "mock.jwt.token".to_string(),
                additional_documents: None,
            },
            lineage: Lineage {
                hash: "test-hash".to_string(),
                previous_claims,
            },
            timestamp: Utc::now(),
            confidence: Some(1.0),
            metadata: None,
        }
    }

    async fn create_test_state(claims: Vec<DkgClaim>) -> Arc<AppState> {
        let keypair = crypto::KeyPair::generate();
        let mut claims_map = HashMap::new();
        for claim in claims {
            claims_map.insert(claim.id.clone(), claim);
        }

        Arc::new(AppState {
            keypair,
            db: None,
            claims: RwLock::new(claims_map),
            pool: None,
        })
    }

    #[tokio::test]
    async fn test_build_lineage_tree_single_claim() {
        let claim = create_test_claim("claim-1", "BATCH-001", None);
        let state = create_test_state(vec![claim]).await;

        let tree = build_lineage_tree(&state, "claim-1").await;

        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0], "claim-1");
    }

    #[tokio::test]
    async fn test_build_lineage_tree_with_parents() {
        let parent1 = create_test_claim("parent-1", "BATCH-PARENT-1", None);
        let parent2 = create_test_claim("parent-2", "BATCH-PARENT-2", None);
        let child = create_test_claim(
            "child-1",
            "BATCH-CHILD",
            Some(vec!["parent-1".to_string(), "parent-2".to_string()]),
        );

        let state = create_test_state(vec![parent1, parent2, child]).await;

        let tree = build_lineage_tree(&state, "child-1").await;

        assert_eq!(tree.len(), 3);
        assert!(tree.contains(&"child-1".to_string()));
        assert!(tree.contains(&"parent-1".to_string()));
        assert!(tree.contains(&"parent-2".to_string()));
    }

    #[tokio::test]
    async fn test_build_lineage_tree_multi_level() {
        let grandparent = create_test_claim("grandparent", "BATCH-GP", None);
        let parent = create_test_claim("parent", "BATCH-P", Some(vec!["grandparent".to_string()]));
        let child = create_test_claim("child", "BATCH-C", Some(vec!["parent".to_string()]));

        let state = create_test_state(vec![grandparent, parent, child]).await;

        let tree = build_lineage_tree_detailed(&state, "child").await;

        assert_eq!(tree.claim_ids.len(), 3);
        assert_eq!(tree.max_depth, 2);
        assert!(!tree.truncated);
    }

    #[tokio::test]
    async fn test_build_lineage_tree_handles_cycles() {
        // Create claims that reference each other (cycle)
        let claim_a = create_test_claim("claim-a", "BATCH-A", Some(vec!["claim-b".to_string()]));
        let claim_b = create_test_claim("claim-b", "BATCH-B", Some(vec!["claim-a".to_string()]));

        let state = create_test_state(vec![claim_a, claim_b]).await;

        let tree = build_lineage_tree(&state, "claim-a").await;

        // Should not infinite loop, should include both claims exactly once
        assert_eq!(tree.len(), 2);
    }

    #[tokio::test]
    async fn test_build_upstream_lineage() {
        let grandparent = create_test_claim("grandparent", "BATCH-GP", None);
        let parent = create_test_claim("parent", "BATCH-P", Some(vec!["grandparent".to_string()]));
        let child = create_test_claim("child", "BATCH-C", Some(vec!["parent".to_string()]));

        let state = create_test_state(vec![grandparent, parent, child]).await;

        let upstream = build_upstream_lineage(&state, "child").await;

        assert_eq!(upstream.len(), 2);
        // Should contain parent and grandparent, but not child
        assert!(upstream.iter().any(|c| c.id == "parent"));
        assert!(upstream.iter().any(|c| c.id == "grandparent"));
        assert!(!upstream.iter().any(|c| c.id == "child"));
    }

    #[tokio::test]
    async fn test_build_downstream_lineage() {
        let parent = create_test_claim("parent", "BATCH-P", None);
        let child1 = create_test_claim("child-1", "BATCH-C1", Some(vec!["parent".to_string()]));
        let child2 = create_test_claim("child-2", "BATCH-C2", Some(vec!["parent".to_string()]));

        let state = create_test_state(vec![parent, child1, child2]).await;

        let downstream = build_downstream_lineage(&state, "parent").await;

        assert_eq!(downstream.len(), 2);
        assert!(downstream.iter().any(|c| c.id == "child-1"));
        assert!(downstream.iter().any(|c| c.id == "child-2"));
    }

    #[tokio::test]
    async fn test_missing_claim_handled() {
        let child = create_test_claim("child", "BATCH-C", Some(vec!["missing-parent".to_string()]));

        let state = create_test_state(vec![child]).await;

        let tree = build_lineage_tree(&state, "child").await;

        // Should include both the child and the reference to missing parent
        assert_eq!(tree.len(), 2);
        assert!(tree.contains(&"child".to_string()));
        assert!(tree.contains(&"missing-parent".to_string()));
    }
}
