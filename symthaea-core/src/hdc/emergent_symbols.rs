//! # Emergent Symbol Grounding
//!
//! Detects when distributed HDC representations crystallize into discrete symbols.
//!
//! ## The Symbol Grounding Problem
//!
//! In cognitive science, the "symbol grounding problem" asks how meaningless symbols
//! acquire meaning. In HDC, we can detect this transition computationally:
//!
//! - **Distributed**: Fuzzy, overlapping representations (low crispness)
//! - **Symbolic**: Sharp, discrete clusters (high crispness)
//!
//! This module provides tools to:
//! 1. Cluster memories by HDC similarity
//! 2. Measure "crispness" of each cluster
//! 3. Identify when clusters crystallize into symbols
//! 4. Track symbol emergence over time
//!
//! ## Scientific Basis
//!
//! - Harnad (1990): Symbol Grounding Problem
//! - Steels (2008): Emergent symbol systems in language games
//! - Kanerva (2009): Hyperdimensional computing and symbol manipulation
//!
//! ## Example Usage
//!
//! ```rust
//! use symthaea::hdc::emergent_symbols::{SymbolGrounder, SymbolGrounderConfig};
//! use symthaea::hdc::unified_hv::ContinuousHV;
//!
//! let mut grounder = SymbolGrounder::new(SymbolGrounderConfig::default());
//!
//! // Add memories
//! let memories: Vec<ContinuousHV> = (0..100)
//!     .map(|i| ContinuousHV::random(1024, i))
//!     .collect();
//!
//! // Check for symbol emergence
//! let symbols = grounder.detect_symbols(&memories);
//! for symbol in symbols {
//!     println!("Symbol: crispness={:.3}, members={}", symbol.crispness, symbol.members);
//! }
//! ```

use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for symbol grounding detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolGrounderConfig {
    /// Minimum crispness to consider a cluster as a symbol (0.0-1.0)
    pub crispness_threshold: f32,

    /// Minimum cluster size to be considered
    pub min_cluster_size: usize,

    /// Similarity threshold for clustering (0.0-1.0)
    pub clustering_threshold: f32,

    /// Maximum number of symbols to track
    pub max_symbols: usize,

    /// Decay factor for symbol stability tracking
    pub stability_decay: f32,
}

impl Default for SymbolGrounderConfig {
    fn default() -> Self {
        Self {
            crispness_threshold: 0.7,
            min_cluster_size: 3,
            clustering_threshold: 0.5,
            max_symbols: 1000,
            stability_decay: 0.99,
        }
    }
}

/// A detected symbol (crystallized cluster)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol {
    /// Unique identifier
    pub id: u64,

    /// Prototype hypervector (cluster centroid)
    pub prototype: ContinuousHV,

    /// Optional human-readable label
    pub label: Option<String>,

    /// Number of members in this cluster
    pub members: usize,

    /// Crispness score (how discrete/symbolic this is)
    /// 1.0 = perfectly crisp, 0.0 = completely distributed
    pub crispness: f32,

    /// Intra-cluster variance (lower = more symbolic)
    pub variance: f32,

    /// How many times this symbol has been detected (stability)
    pub detection_count: u64,

    /// Average distance to nearest neighbor symbol
    pub isolation: f32,
}

impl Symbol {
    /// Check if this qualifies as a "true" symbol
    pub fn is_crystallized(&self, threshold: f32) -> bool {
        self.crispness >= threshold
    }

    /// Compute semantic similarity to another symbol
    pub fn similarity(&self, other: &Symbol) -> f32 {
        self.prototype.similarity(&other.prototype)
    }
}

/// Result of symbol detection analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolDetectionResult {
    /// All detected symbols
    pub symbols: Vec<Symbol>,

    /// Number of items that crystallized into symbols
    pub crystallized_count: usize,

    /// Number of items remaining distributed
    pub distributed_count: usize,

    /// Overall crystallization ratio (0.0-1.0)
    pub crystallization_ratio: f32,

    /// Average crispness across all clusters
    pub average_crispness: f32,

    /// Highest crispness achieved
    pub max_crispness: f32,

    /// Number of new symbols detected this round
    pub new_symbols: usize,

    /// Number of stable symbols (high detection count)
    pub stable_symbols: usize,
}

/// Symbol emergence event for tracking dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolEmergenceEvent {
    /// When this event occurred (step number)
    pub step: u64,

    /// Type of event
    pub event_type: SymbolEventType,

    /// Symbol involved
    pub symbol_id: u64,

    /// Crispness at event time
    pub crispness: f32,

    /// Cluster size at event time
    pub size: usize,
}

/// Types of symbol emergence events
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SymbolEventType {
    /// New symbol crystallized from distributed representation
    Crystallized,

    /// Symbol dissolved back into distributed representation
    Dissolved,

    /// Two symbols merged into one
    Merged { other_id: u64 },

    /// One symbol split into multiple
    Split { new_ids: Vec<u64> },

    /// Symbol was labeled by external agent
    Labeled { label: String },
}

/// Main symbol grounding detector
#[derive(Debug)]
pub struct SymbolGrounder {
    config: SymbolGrounderConfig,

    /// Currently tracked symbols
    symbols: HashMap<u64, Symbol>,

    /// Next symbol ID to assign
    next_id: u64,

    /// Event history
    events: Vec<SymbolEmergenceEvent>,

    /// Current step number
    step: u64,
}

impl SymbolGrounder {
    /// Create a new symbol grounder with given configuration
    pub fn new(config: SymbolGrounderConfig) -> Self {
        Self {
            config,
            symbols: HashMap::new(),
            next_id: 1,
            events: Vec::new(),
            step: 0,
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(SymbolGrounderConfig::default())
    }

    /// Detect symbols in a set of memories
    pub fn detect_symbols(&mut self, memories: &[ContinuousHV]) -> SymbolDetectionResult {
        self.step += 1;

        if memories.is_empty() {
            return SymbolDetectionResult {
                symbols: Vec::new(),
                crystallized_count: 0,
                distributed_count: 0,
                crystallization_ratio: 0.0,
                average_crispness: 0.0,
                max_crispness: 0.0,
                new_symbols: 0,
                stable_symbols: 0,
            };
        }

        // Step 1: Cluster memories by similarity
        let clusters = self.cluster_by_similarity(memories);

        // Step 2: Analyze each cluster for symbolization
        let mut detected_symbols = Vec::new();
        let mut crystallized_count = 0;
        let mut new_symbols_count = 0;

        for cluster in clusters {
            if cluster.len() < self.config.min_cluster_size {
                continue;
            }

            // Compute prototype (centroid)
            let prototype = self.compute_centroid(&cluster);

            // Compute crispness (average similarity to prototype)
            let crispness = self.compute_crispness(&cluster, &prototype);

            // Compute variance
            let variance = self.compute_variance(&cluster, &prototype);

            // Check if this matches an existing symbol
            let (symbol_id, is_new) = self.find_or_create_symbol(&prototype);

            let symbol = Symbol {
                id: symbol_id,
                prototype,
                label: self.symbols.get(&symbol_id).and_then(|s| s.label.clone()),
                members: cluster.len(),
                crispness,
                variance,
                detection_count: self.symbols.get(&symbol_id).map_or(1, |s| s.detection_count + 1),
                isolation: 0.0, // Will compute after all symbols detected
            };

            // Record crystallization events
            if crispness >= self.config.crispness_threshold {
                crystallized_count += cluster.len();

                if is_new {
                    new_symbols_count += 1;
                    self.events.push(SymbolEmergenceEvent {
                        step: self.step,
                        event_type: SymbolEventType::Crystallized,
                        symbol_id,
                        crispness,
                        size: cluster.len(),
                    });
                }
            }

            detected_symbols.push(symbol);
        }

        // Step 3: Compute isolation for each symbol
        for i in 0..detected_symbols.len() {
            let mut min_dist = f32::MAX;
            for j in 0..detected_symbols.len() {
                if i != j {
                    let sim = detected_symbols[i].similarity(&detected_symbols[j]);
                    let dist = 1.0 - sim;
                    if dist < min_dist {
                        min_dist = dist;
                    }
                }
            }
            detected_symbols[i].isolation = if min_dist == f32::MAX { 1.0 } else { min_dist };
        }

        // Step 4: Update tracked symbols
        for symbol in &detected_symbols {
            self.symbols.insert(symbol.id, symbol.clone());
        }

        // Trim to max symbols if needed
        self.trim_symbols();

        // Step 5: Compute statistics
        let total = memories.len();
        let distributed_count = total - crystallized_count;
        let crystallization_ratio = crystallized_count as f32 / total as f32;

        let average_crispness = if detected_symbols.is_empty() {
            0.0
        } else {
            detected_symbols.iter().map(|s| s.crispness).sum::<f32>() / detected_symbols.len() as f32
        };

        let max_crispness = detected_symbols.iter().map(|s| s.crispness).fold(0.0f32, f32::max);

        let stable_symbols = detected_symbols.iter()
            .filter(|s| s.detection_count >= 5 && s.crispness >= self.config.crispness_threshold)
            .count();

        SymbolDetectionResult {
            symbols: detected_symbols,
            crystallized_count,
            distributed_count,
            crystallization_ratio,
            average_crispness,
            max_crispness,
            new_symbols: new_symbols_count,
            stable_symbols,
        }
    }

    /// Cluster memories by HDC similarity using simple greedy algorithm
    fn cluster_by_similarity(&self, memories: &[ContinuousHV]) -> Vec<Vec<ContinuousHV>> {
        if memories.is_empty() {
            return Vec::new();
        }

        let mut clusters: Vec<Vec<ContinuousHV>> = Vec::new();
        let mut assigned = vec![false; memories.len()];

        for i in 0..memories.len() {
            if assigned[i] {
                continue;
            }

            // Start a new cluster with this memory
            let mut cluster = vec![memories[i].clone()];
            assigned[i] = true;

            // Find all similar memories
            for j in (i + 1)..memories.len() {
                if assigned[j] {
                    continue;
                }

                let sim = memories[i].similarity(&memories[j]);
                if sim >= self.config.clustering_threshold {
                    cluster.push(memories[j].clone());
                    assigned[j] = true;
                }
            }

            clusters.push(cluster);
        }

        clusters
    }

    /// Compute centroid of a cluster
    fn compute_centroid(&self, cluster: &[ContinuousHV]) -> ContinuousHV {
        if cluster.is_empty() {
            return ContinuousHV::zero(1);
        }

        let refs: Vec<&ContinuousHV> = cluster.iter().collect();
        ContinuousHV::bundle(&refs)
    }

    /// Compute crispness (average similarity to prototype)
    fn compute_crispness(&self, cluster: &[ContinuousHV], prototype: &ContinuousHV) -> f32 {
        if cluster.is_empty() {
            return 0.0;
        }

        let sum: f32 = cluster.iter()
            .map(|hv| prototype.similarity(hv).max(0.0)) // Only positive similarity
            .sum();

        sum / cluster.len() as f32
    }

    /// Compute variance within cluster
    fn compute_variance(&self, cluster: &[ContinuousHV], prototype: &ContinuousHV) -> f32 {
        if cluster.len() < 2 {
            return 0.0;
        }

        let mean_sim = self.compute_crispness(cluster, prototype);

        let sum_sq: f32 = cluster.iter()
            .map(|hv| {
                let sim = prototype.similarity(hv);
                (sim - mean_sim).powi(2)
            })
            .sum();

        (sum_sq / (cluster.len() - 1) as f32).sqrt()
    }

    /// Find existing symbol or create new one
    fn find_or_create_symbol(&mut self, prototype: &ContinuousHV) -> (u64, bool) {
        // Check if any existing symbol matches
        for (id, symbol) in &self.symbols {
            let sim = symbol.prototype.similarity(prototype);
            if sim >= self.config.clustering_threshold {
                return (*id, false);
            }
        }

        // Create new symbol
        let id = self.next_id;
        self.next_id += 1;
        (id, true)
    }

    /// Trim symbols to max capacity
    fn trim_symbols(&mut self) {
        if self.symbols.len() <= self.config.max_symbols {
            return;
        }

        // Remove symbols with lowest detection count
        let mut symbols: Vec<_> = self.symbols.iter().collect();
        symbols.sort_by(|a, b| a.1.detection_count.cmp(&b.1.detection_count));

        let to_remove = self.symbols.len() - self.config.max_symbols;
        let ids_to_remove: Vec<u64> = symbols.iter()
            .take(to_remove)
            .map(|(id, _)| **id)
            .collect();

        for id in ids_to_remove {
            self.symbols.remove(&id);
        }
    }

    /// Get all currently tracked symbols
    pub fn symbols(&self) -> &HashMap<u64, Symbol> {
        &self.symbols
    }

    /// Get crystallized symbols only
    pub fn crystallized_symbols(&self) -> Vec<&Symbol> {
        self.symbols.values()
            .filter(|s| s.crispness >= self.config.crispness_threshold)
            .collect()
    }

    /// Get stable symbols (high detection count + crystallized)
    pub fn stable_symbols(&self) -> Vec<&Symbol> {
        self.symbols.values()
            .filter(|s| s.detection_count >= 5 && s.crispness >= self.config.crispness_threshold)
            .collect()
    }

    /// Label a symbol
    pub fn label_symbol(&mut self, symbol_id: u64, label: String) -> bool {
        if let Some(symbol) = self.symbols.get_mut(&symbol_id) {
            symbol.label = Some(label.clone());
            self.events.push(SymbolEmergenceEvent {
                step: self.step,
                event_type: SymbolEventType::Labeled { label },
                symbol_id,
                crispness: symbol.crispness,
                size: symbol.members,
            });
            true
        } else {
            false
        }
    }

    /// Get event history
    pub fn events(&self) -> &[SymbolEmergenceEvent] {
        &self.events
    }

    /// Clear event history
    pub fn clear_events(&mut self) {
        self.events.clear();
    }

    /// Get statistics about symbol system
    pub fn statistics(&self) -> SymbolSystemStats {
        let total = self.symbols.len();
        let crystallized = self.crystallized_symbols().len();
        let stable = self.stable_symbols().len();
        let labeled = self.symbols.values().filter(|s| s.label.is_some()).count();

        let avg_crispness = if total > 0 {
            self.symbols.values().map(|s| s.crispness).sum::<f32>() / total as f32
        } else {
            0.0
        };

        let avg_members = if total > 0 {
            self.symbols.values().map(|s| s.members).sum::<usize>() as f32 / total as f32
        } else {
            0.0
        };

        SymbolSystemStats {
            total_symbols: total,
            crystallized_symbols: crystallized,
            stable_symbols: stable,
            labeled_symbols: labeled,
            average_crispness: avg_crispness,
            average_cluster_size: avg_members,
            total_events: self.events.len(),
            current_step: self.step,
        }
    }
}

/// Statistics about the symbol system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolSystemStats {
    pub total_symbols: usize,
    pub crystallized_symbols: usize,
    pub stable_symbols: usize,
    pub labeled_symbols: usize,
    pub average_crispness: f32,
    pub average_cluster_size: f32,
    pub total_events: usize,
    pub current_step: u64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_detection_basic() {
        let mut grounder = SymbolGrounder::default_config();

        // Create two distinct clusters
        let dim = 1024;

        // Cluster 1: All similar to each other
        let base1 = ContinuousHV::random(dim, 42);
        let cluster1: Vec<ContinuousHV> = (0..5)
            .map(|i| {
                let noise = ContinuousHV::random(dim, 100 + i).scale(0.1);
                base1.add(&noise)
            })
            .collect();

        // Cluster 2: All similar to each other but different from cluster 1
        let base2 = ContinuousHV::random(dim, 4242);
        let cluster2: Vec<ContinuousHV> = (0..5)
            .map(|i| {
                let noise = ContinuousHV::random(dim, 200 + i).scale(0.1);
                base2.add(&noise)
            })
            .collect();

        // Combine
        let mut memories = cluster1;
        memories.extend(cluster2);

        let result = grounder.detect_symbols(&memories);

        println!("Symbols detected: {}", result.symbols.len());
        println!("Crystallized: {}", result.crystallized_count);
        println!("Max crispness: {:.3}", result.max_crispness);

        // Should detect at least some structure
        assert!(result.symbols.len() >= 1);
    }

    #[test]
    fn test_crispness_calculation() {
        let grounder = SymbolGrounder::default_config();
        let dim = 1024;

        // Create a very crisp cluster (all nearly identical)
        let base = ContinuousHV::random(dim, 42);
        let cluster: Vec<ContinuousHV> = (0..5)
            .map(|i| {
                let noise = ContinuousHV::random(dim, 100 + i).scale(0.01);
                base.add(&noise)
            })
            .collect();

        let prototype = grounder.compute_centroid(&cluster);
        let crispness = grounder.compute_crispness(&cluster, &prototype);

        println!("Crisp cluster crispness: {:.3}", crispness);
        assert!(crispness > 0.9, "Crisp cluster should have high crispness");

        // Create a fuzzy cluster (less similar)
        let fuzzy: Vec<ContinuousHV> = (0..5)
            .map(|i| ContinuousHV::random(dim, i))
            .collect();

        let fuzzy_prototype = grounder.compute_centroid(&fuzzy);
        let fuzzy_crispness = grounder.compute_crispness(&fuzzy, &fuzzy_prototype);

        println!("Fuzzy cluster crispness: {:.3}", fuzzy_crispness);
        assert!(fuzzy_crispness < crispness, "Fuzzy cluster should have lower crispness");
    }

    #[test]
    fn test_symbol_emergence_tracking() {
        let mut grounder = SymbolGrounder::default_config();
        let dim = 1024;

        // First round: introduce a cluster
        let base = ContinuousHV::random(dim, 42);
        let cluster: Vec<ContinuousHV> = (0..5)
            .map(|i| {
                let noise = ContinuousHV::random(dim, 100 + i).scale(0.05);
                base.add(&noise)
            })
            .collect();

        let result1 = grounder.detect_symbols(&cluster);
        println!("Round 1: {} symbols, {} new", result1.symbols.len(), result1.new_symbols);

        // Second round: same cluster should be stable
        let result2 = grounder.detect_symbols(&cluster);
        println!("Round 2: {} symbols, {} new", result2.symbols.len(), result2.new_symbols);

        // New symbols should be 0 in second round
        assert_eq!(result2.new_symbols, 0, "Should recognize existing symbol");

        // Detection count should increase
        if let Some(symbol) = result2.symbols.first() {
            assert!(symbol.detection_count >= 2, "Detection count should increase");
        }
    }

    #[test]
    fn test_labeling() {
        let mut grounder = SymbolGrounder::default_config();
        let dim = 1024;

        let base = ContinuousHV::random(dim, 42);
        let cluster: Vec<ContinuousHV> = (0..5)
            .map(|i| {
                let noise = ContinuousHV::random(dim, 100 + i).scale(0.05);
                base.add(&noise)
            })
            .collect();

        let result = grounder.detect_symbols(&cluster);

        if let Some(symbol) = result.symbols.first() {
            let success = grounder.label_symbol(symbol.id, "apple".to_string());
            assert!(success, "Labeling should succeed");

            let updated = grounder.symbols().get(&symbol.id).unwrap();
            assert_eq!(updated.label, Some("apple".to_string()));
        }
    }

    #[test]
    fn test_statistics() {
        let mut grounder = SymbolGrounder::default_config();
        let dim = 1024;

        let base = ContinuousHV::random(dim, 42);
        let cluster: Vec<ContinuousHV> = (0..5)
            .map(|i| {
                let noise = ContinuousHV::random(dim, 100 + i).scale(0.05);
                base.add(&noise)
            })
            .collect();

        let _ = grounder.detect_symbols(&cluster);
        let stats = grounder.statistics();

        println!("Stats: {:?}", stats);
        assert!(stats.total_symbols >= 1);
        assert!(stats.current_step == 1);
    }
}
