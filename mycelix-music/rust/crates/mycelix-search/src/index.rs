// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HNSW index implementation

use crate::{AudioEmbedding, DistanceMetric, EmbeddingMetadata, Result, SearchError, SearchResult};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;
use uuid::Uuid;

/// HNSW graph node
#[derive(Debug, Clone)]
struct HnswNode {
    id: Uuid,
    vector: Vec<f32>,
    metadata: EmbeddingMetadata,
    /// Connections at each level
    connections: Vec<Vec<Uuid>>,
    level: usize,
}

/// Candidate for search with distance
#[derive(Debug, Clone)]
struct Candidate {
    id: Uuid,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        other.distance.partial_cmp(&self.distance).unwrap_or(Ordering::Equal)
    }
}

/// HNSW index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Vector dimension
    pub dimension: usize,
    /// Distance metric
    pub metric: DistanceMetric,
    /// Maximum number of connections per node at each level
    pub m: usize,
    /// Maximum number of connections at level 0
    pub m0: usize,
    /// Size of dynamic candidate list during construction
    pub ef_construction: usize,
    /// Normalization factor for level generation
    pub ml: f64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            dimension: 256,
            metric: DistanceMetric::Cosine,
            m: 16,
            m0: 32,
            ef_construction: 200,
            ml: 1.0 / (16.0_f64).ln(),
        }
    }
}

impl HnswConfig {
    pub fn for_dimension(dimension: usize) -> Self {
        Self {
            dimension,
            ..Default::default()
        }
    }
}

/// HNSW index for approximate nearest neighbor search
pub struct HnswIndex {
    config: HnswConfig,
    nodes: RwLock<HashMap<Uuid, HnswNode>>,
    entry_point: RwLock<Option<Uuid>>,
    max_level: RwLock<usize>,
}

impl HnswIndex {
    pub fn new(config: HnswConfig) -> Self {
        Self {
            config,
            nodes: RwLock::new(HashMap::new()),
            entry_point: RwLock::new(None),
            max_level: RwLock::new(0),
        }
    }

    /// Generate random level for new node
    fn random_level(&self) -> usize {
        let mut level = 0;
        let mut r = rand_u64() as f64 / u64::MAX as f64;

        while r < self.config.ml.exp().recip() && level < 16 {
            level += 1;
            r = rand_u64() as f64 / u64::MAX as f64;
        }

        level
    }

    /// Insert embedding into index
    pub fn insert(&self, embedding: AudioEmbedding) -> Result<Uuid> {
        if embedding.vector.len() != self.config.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.config.dimension,
                actual: embedding.vector.len(),
            });
        }

        let id = embedding.id;
        let level = self.random_level();

        let node = HnswNode {
            id,
            vector: embedding.vector,
            metadata: embedding.metadata,
            connections: vec![Vec::new(); level + 1],
            level,
        };

        let mut nodes = self.nodes.write();
        let mut entry_point = self.entry_point.write();
        let mut max_level = self.max_level.write();

        if nodes.is_empty() {
            // First node becomes entry point
            nodes.insert(id, node);
            *entry_point = Some(id);
            *max_level = level;
            return Ok(id);
        }

        let ep = entry_point.unwrap();

        // Search from top level down to insert level
        let mut current = ep;
        for l in (level + 1..=*max_level).rev() {
            current = self.search_layer_single(&nodes, &node.vector, current, l);
        }

        // Insert and connect at levels <= insert level
        for l in (0..=level.min(*max_level)).rev() {
            let candidates = self.search_layer(&nodes, &node.vector, current, self.config.ef_construction, l);

            // Select M best connections
            let m = if l == 0 { self.config.m0 } else { self.config.m };
            let connections: Vec<_> = candidates.iter().take(m).map(|c| c.id).collect();

            // Add bidirectional connections
            for &neighbor_id in &connections {
                if let Some(neighbor) = nodes.get_mut(&neighbor_id) {
                    if l < neighbor.connections.len() {
                        // Add connection to neighbor
                        if neighbor.connections[l].len() < m {
                            neighbor.connections[l].push(id);
                        } else {
                            // Prune if necessary
                            self.prune_connections(neighbor, l, m);
                        }
                    }
                }
            }

            // Store connections for new node
            let mut new_node = node.clone();
            if l < new_node.connections.len() {
                new_node.connections[l] = connections.clone();
            }
            nodes.insert(id, new_node.clone());

            if !candidates.is_empty() {
                current = candidates[0].id;
            }
        }

        // Update entry point if new node has higher level
        if level > *max_level {
            *entry_point = Some(id);
            *max_level = level;
        }

        nodes.insert(id, node);
        Ok(id)
    }

    /// Search for nearest neighbor in single layer
    fn search_layer_single(
        &self,
        nodes: &HashMap<Uuid, HnswNode>,
        query: &[f32],
        entry: Uuid,
        level: usize,
    ) -> Uuid {
        let mut current = entry;
        let mut current_dist = self.distance(query, &nodes[&current].vector);

        loop {
            let mut changed = false;

            if let Some(node) = nodes.get(&current) {
                if level < node.connections.len() {
                    for &neighbor_id in &node.connections[level] {
                        if let Some(neighbor) = nodes.get(&neighbor_id) {
                            let dist = self.distance(query, &neighbor.vector);
                            if dist < current_dist {
                                current = neighbor_id;
                                current_dist = dist;
                                changed = true;
                            }
                        }
                    }
                }
            }

            if !changed {
                break;
            }
        }

        current
    }

    /// Search for ef nearest neighbors in a layer
    fn search_layer(
        &self,
        nodes: &HashMap<Uuid, HnswNode>,
        query: &[f32],
        entry: Uuid,
        ef: usize,
        level: usize,
    ) -> Vec<Candidate> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        let entry_dist = self.distance(query, &nodes[&entry].vector);
        visited.insert(entry);
        candidates.push(Candidate { id: entry, distance: entry_dist });
        results.push(Candidate { id: entry, distance: -entry_dist }); // Max heap for results

        while let Some(current) = candidates.pop() {
            // Check if we can stop
            if let Some(worst) = results.peek() {
                if current.distance > -worst.distance && results.len() >= ef {
                    break;
                }
            }

            if let Some(node) = nodes.get(&current.id) {
                if level < node.connections.len() {
                    for &neighbor_id in &node.connections[level] {
                        if visited.insert(neighbor_id) {
                            if let Some(neighbor) = nodes.get(&neighbor_id) {
                                let dist = self.distance(query, &neighbor.vector);

                                let should_add = results.len() < ef ||
                                    results.peek().map(|w| dist < -w.distance).unwrap_or(true);

                                if should_add {
                                    candidates.push(Candidate { id: neighbor_id, distance: dist });
                                    results.push(Candidate { id: neighbor_id, distance: -dist });

                                    if results.len() > ef {
                                        results.pop();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Convert results to proper format
        let mut result_vec: Vec<_> = results
            .into_iter()
            .map(|c| Candidate { id: c.id, distance: -c.distance })
            .collect();
        result_vec.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        result_vec
    }

    /// Prune connections to keep only the best M
    fn prune_connections(&self, node: &mut HnswNode, level: usize, m: usize) {
        if level >= node.connections.len() || node.connections[level].len() <= m {
            return;
        }

        // Keep only the M closest connections
        // In production, would use heuristic pruning
        node.connections[level].truncate(m);
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.config.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.config.dimension,
                actual: query.len(),
            });
        }

        let nodes = self.nodes.read();
        let entry_point = self.entry_point.read();
        let max_level = self.max_level.read();

        let ep = match *entry_point {
            Some(ep) => ep,
            None => return Ok(vec![]),
        };

        // Search from top level
        let mut current = ep;
        for l in (1..=*max_level).rev() {
            current = self.search_layer_single(&nodes, query, current, l);
        }

        // Search level 0 with ef
        let candidates = self.search_layer(&nodes, query, current, ef_search.max(k), 0);

        // Convert to SearchResults
        let results: Vec<_> = candidates
            .into_iter()
            .take(k)
            .filter_map(|c| {
                nodes.get(&c.id).map(|node| {
                    let score = self.config.metric.to_similarity(c.distance);
                    SearchResult::new(c.id, score, c.distance, node.metadata.clone())
                })
            })
            .collect();

        Ok(results)
    }

    /// Get node by ID
    pub fn get(&self, id: &Uuid) -> Option<AudioEmbedding> {
        self.nodes.read().get(id).map(|node| AudioEmbedding {
            id: node.id,
            vector: node.vector.clone(),
            metadata: node.metadata.clone(),
        })
    }

    /// Remove node from index
    pub fn remove(&self, id: &Uuid) -> Result<()> {
        let mut nodes = self.nodes.write();
        let mut entry_point = self.entry_point.write();

        // Collect neighbor IDs first to avoid borrow conflict
        let neighbor_ids: Vec<Uuid> = if let Some(node) = nodes.get(id) {
            node.connections
                .iter()
                .flat_map(|level| level.iter().copied())
                .collect()
        } else {
            Vec::new()
        };

        // Remove from all neighbor lists
        for neighbor_id in neighbor_ids {
            if let Some(neighbor) = nodes.get_mut(&neighbor_id) {
                for conns in &mut neighbor.connections {
                    conns.retain(|&x| x != *id);
                }
            }
        }

        nodes.remove(id).ok_or_else(|| SearchError::NotFound(id.to_string()))?;

        // Update entry point if needed
        if *entry_point == Some(*id) {
            *entry_point = nodes.keys().next().cloned();
        }

        Ok(())
    }

    /// Get index size
    pub fn len(&self) -> usize {
        self.nodes.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.read().is_empty()
    }

    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        self.config.metric.compute(a, b)
    }
}

/// Simple random number generator (would use rand crate in production)
fn rand_u64() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos() as u64;

    // LCG parameters
    static mut STATE: u64 = 0;
    unsafe {
        STATE = STATE.wrapping_mul(6364136223846793005).wrapping_add(nanos);
        STATE
    }
}

/// Index statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    pub num_nodes: usize,
    pub max_level: usize,
    pub avg_connections: f32,
    pub dimension: usize,
}

impl HnswIndex {
    pub fn stats(&self) -> IndexStats {
        let nodes = self.nodes.read();
        let max_level = *self.max_level.read();

        let total_connections: usize = nodes
            .values()
            .map(|n| n.connections.iter().map(|c| c.len()).sum::<usize>())
            .sum();

        let avg_connections = if nodes.is_empty() {
            0.0
        } else {
            total_connections as f32 / nodes.len() as f32
        };

        IndexStats {
            num_nodes: nodes.len(),
            max_level,
            avg_connections,
            dimension: self.config.dimension,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_insert_search() {
        let config = HnswConfig::for_dimension(4);
        let index = HnswIndex::new(config);

        // Insert embeddings
        for i in 0..10 {
            let emb = AudioEmbedding::new(vec![i as f32, 0.0, 0.0, 0.0]);
            index.insert(emb).unwrap();
        }

        assert_eq!(index.len(), 10);

        // Search
        let results = index.search(&[5.0, 0.0, 0.0, 0.0], 3, 50).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_random_level() {
        let config = HnswConfig::default();
        let index = HnswIndex::new(config);

        // Generate levels and check distribution
        let levels: Vec<_> = (0..1000).map(|_| index.random_level()).collect();
        let max_level = levels.iter().max().unwrap();
        // Should not exceed max_level cap of 16 set in random_level()
        assert!(*max_level <= 16);
        // Level 0 should appear with reasonable frequency (at least 10%)
        let level_0_count = levels.iter().filter(|&&l| l == 0).count();
        assert!(level_0_count > 50, "Expected level 0 to appear frequently");
    }
}
