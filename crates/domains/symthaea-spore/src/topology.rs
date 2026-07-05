// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness topology analysis for the Spore WASM kernel.
//!
//! Simplified port of `symthaea-consciousness-topology` (algebraic topology)
//! and `symthaea-field-dynamics` (wave packets / interference) for WASM targets.
//!
//! WASM-compatible: uses cycle counters instead of `std::time`.
//!
//! # Consciousness Space (7D)
//!
//! Each point lives in a 7-dimensional space:
//! `[Phi, Binding, Workspace, Attention, Recurrence, Embodiment, Knowledge]`
//!
//! Topological features (Betti numbers) reveal the *shape* of consciousness:
//! - β₀ = 1 → unified awareness; β₀ > 1 → fragmentation
//! - β₁ > 0 → cyclic/oscillatory patterns
//! - β₂ > 0 → enclosed voids in experience space

use serde::{Deserialize, Serialize};

// ── Points ──────────────────────────────────────────────────────────────

/// A sample in 7D consciousness space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessPoint {
    /// [Phi, Binding, Workspace, Attention, Recurrence, Embodiment, Knowledge]
    pub dimensions: [f64; 7],
    /// Cycle counter when this point was observed.
    pub cycle: u64,
}

impl ConsciousnessPoint {
    /// Euclidean distance in 7D consciousness space.
    pub fn distance(&self, other: &Self) -> f64 {
        self.dimensions
            .iter()
            .zip(other.dimensions.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }
}

// ── Betti Numbers ───────────────────────────────────────────────────────

/// Topological invariants of the consciousness point cloud.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BettiNumbers {
    /// β₀ — connected components (unity vs fragmentation).
    pub beta_0: usize,
    /// β₁ — 1-holes / loops (circular reasoning, oscillation).
    pub beta_1: usize,
    /// β₂ — 2-voids (enclosed regions of awareness).
    pub beta_2: usize,
    /// χ = β₀ − β₁ + β₂
    pub euler_characteristic: i64,
}

impl BettiNumbers {
    pub fn new(beta_0: usize, beta_1: usize, beta_2: usize) -> Self {
        let euler_characteristic = beta_0 as i64 - beta_1 as i64 + beta_2 as i64;
        Self {
            beta_0,
            beta_1,
            beta_2,
            euler_characteristic,
        }
    }
}

// ── Interpretation ──────────────────────────────────────────────────────

/// Human-legible interpretation of topology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyInterpretation {
    /// 1.0 for unified (β₀ = 1), 1/β₀ otherwise.
    pub unity: f64,
    /// (β₁ + 2·β₂) / 10.0 — higher = richer internal structure.
    pub complexity: f64,
    /// (β₀ − 1) / 10.0 when fragmented, else 0.
    pub fragmentation: f64,
}

impl TopologyInterpretation {
    pub fn from_betti(b: &BettiNumbers) -> Self {
        let unity = if b.beta_0 <= 1 {
            1.0
        } else {
            1.0 / b.beta_0 as f64
        };
        let complexity = (b.beta_1 + 2 * b.beta_2) as f64 / 10.0;
        let fragmentation = if b.beta_0 > 1 {
            (b.beta_0 - 1) as f64 / 10.0
        } else {
            0.0
        };
        Self {
            unity,
            complexity,
            fragmentation,
        }
    }
}

// ── Persistence ─────────────────────────────────────────────────────────

/// A birth-death pair from persistent homology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistencePair {
    /// Homological dimension (0 = component, 1 = loop, …).
    pub dimension: usize,
    /// Filtration value at which the feature appears.
    pub birth: f64,
    /// Filtration value at which the feature dies. `None` = persists forever.
    pub death: Option<f64>,
}

impl PersistencePair {
    /// Lifetime of this feature (infinite features → `f64::INFINITY`).
    pub fn persistence(&self) -> f64 {
        match self.death {
            Some(d) => d - self.birth,
            None => f64::INFINITY,
        }
    }
}

// ── Wave Packets ────────────────────────────────────────────────────────

/// A Gaussian wave packet in 7D consciousness space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WavePacket {
    /// Center position in 7D space.
    pub center: [f64; 7],
    /// Peak amplitude.
    pub amplitude: f64,
    /// Angular frequency.
    pub frequency: f64,
    /// Spatial width (σ) of the Gaussian envelope.
    pub width: f64,
    /// Cycle at which this packet was created.
    pub created_cycle: u64,
}

impl WavePacket {
    /// Evaluate the wave packet at a position and time.
    ///
    /// `value = amplitude × exp(−r²/(2σ²)) × cos(ω·t)`
    pub fn evaluate(&self, position: &[f64; 7], time: f64) -> f64 {
        let r_sq: f64 = self
            .center
            .iter()
            .zip(position.iter())
            .map(|(c, p)| (c - p) * (c - p))
            .sum();
        let envelope = (-r_sq / (2.0 * self.width * self.width)).exp();
        let wave = (self.frequency * time).cos();
        self.amplitude * envelope * wave
    }

    /// Energy ∝ amplitude² × σ^3.5  (7D Gaussian integral scaling).
    pub fn energy(&self) -> f64 {
        self.amplitude * self.amplitude * self.width.powf(3.5)
    }
}

// ── Interference ────────────────────────────────────────────────────────

/// Type of wave interference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterferenceType {
    Constructive,
    Destructive,
    Partial,
}

/// An observed interference event between wave packets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterferenceEvent {
    /// Where the interference occurs.
    pub position: [f64; 7],
    /// Classification of the interference.
    pub interference_type: InterferenceType,
    /// Magnitude of the combined amplitude.
    pub magnitude: f64,
    /// Cycle when detected.
    pub cycle: u64,
}

// ── Analysis result ─────────────────────────────────────────────────────

/// Complete topology analysis snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyAnalysis {
    pub betti: BettiNumbers,
    pub persistence_pairs: Vec<PersistencePair>,
    pub wave_packets: Vec<WavePacket>,
    pub interference_events: Vec<InterferenceEvent>,
    pub interpretation: TopologyInterpretation,
    pub points_analyzed: usize,
}

// ── Config ──────────────────────────────────────────────────────────────

/// Configuration for the topology analyzer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyConfig {
    /// Maximum points in sliding window.
    pub max_points: usize,
    /// Distance threshold for edge creation in the Rips complex.
    pub edge_threshold: f64,
    /// Rebuild the simplicial complex every N observations.
    pub rebuild_interval: usize,
}

impl Default for TopologyConfig {
    fn default() -> Self {
        Self {
            max_points: 100,
            edge_threshold: 0.5,
            rebuild_interval: 10,
        }
    }
}

// ── Stats ───────────────────────────────────────────────────────────────

/// Running statistics from the analyzer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyStats {
    pub total_observations: u64,
    pub current_window_size: usize,
    pub rebuilds: u64,
    pub last_rebuild_cycle: u64,
}

// ── Union-Find ──────────────────────────────────────────────────────────

/// Disjoint-set / union-find for connected-component counting.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // path halving
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
        true
    }

    fn components(&mut self) -> usize {
        let n = self.parent.len();
        let mut seen = std::collections::HashSet::new();
        for i in 0..n {
            seen.insert(self.find(i));
        }
        seen.len()
    }
}

// ── Analyzer ────────────────────────────────────────────────────────────

/// Main topology analysis system.
///
/// Feed it consciousness dimension snapshots via [`observe`] and retrieve
/// topological summaries via [`analyze`].
pub struct TopologyAnalyzer {
    config: TopologyConfig,
    points: Vec<ConsciousnessPoint>,
    /// Adjacency list for the current Rips complex.
    adjacency: Vec<Vec<usize>>,
    /// Number of edges in the current complex.
    edge_count: usize,
    /// Number of triangles (3-cliques) in the current complex.
    triangle_count: usize,
    /// Running counters.
    observations_since_rebuild: usize,
    total_observations: u64,
    rebuilds: u64,
    last_rebuild_cycle: u64,
}

impl TopologyAnalyzer {
    pub fn new(config: TopologyConfig) -> Self {
        Self {
            config,
            points: Vec::new(),
            adjacency: Vec::new(),
            edge_count: 0,
            triangle_count: 0,
            observations_since_rebuild: 0,
            total_observations: 0,
            rebuilds: 0,
            last_rebuild_cycle: 0,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(TopologyConfig::default())
    }

    /// Add a consciousness observation.
    pub fn observe(&mut self, dimensions: [f64; 7], cycle: u64) {
        let point = ConsciousnessPoint { dimensions, cycle };

        // Sliding window — evict oldest when full.
        if self.points.len() >= self.config.max_points {
            self.points.remove(0);
        }
        self.points.push(point);
        self.total_observations += 1;
        self.observations_since_rebuild += 1;

        // Periodically rebuild the simplicial complex.
        if self.observations_since_rebuild >= self.config.rebuild_interval {
            self.rebuild_complex();
            self.last_rebuild_cycle = cycle;
            self.observations_since_rebuild = 0;
            self.rebuilds += 1;
        }
    }

    /// Rebuild the Rips complex from the current point window.
    fn rebuild_complex(&mut self) {
        let n = self.points.len();
        self.adjacency = vec![Vec::new(); n];
        self.edge_count = 0;
        self.triangle_count = 0;

        // Build edges (undirected — store both directions for neighbor lookup).
        for i in 0..n {
            for j in (i + 1)..n {
                if self.points[i].distance(&self.points[j]) <= self.config.edge_threshold {
                    self.adjacency[i].push(j);
                    self.adjacency[j].push(i);
                    self.edge_count += 1;
                }
            }
        }

        // Count triangles (3-cliques): for each edge (i,j), count shared neighbors.
        for i in 0..n {
            for &j in &self.adjacency[i] {
                if j <= i {
                    continue;
                }
                for &k in &self.adjacency[j] {
                    if k <= j {
                        continue;
                    }
                    if self.adjacency[i].contains(&k) {
                        self.triangle_count += 1;
                    }
                }
            }
        }
    }

    /// Compute Betti numbers from the current complex.
    fn compute_betti(&self) -> BettiNumbers {
        let n = self.points.len();
        if n == 0 || self.adjacency.len() != n {
            return BettiNumbers::new(n.min(1), 0, 0);
        }

        // β₀ via union-find.
        let mut uf = UnionFind::new(n);
        for i in 0..n {
            for &j in &self.adjacency[i] {
                if j > i {
                    uf.union(i, j);
                }
            }
        }
        let beta_0 = uf.components();

        // β₁ ≈ edges − vertices + β₀  (Euler formula for 1-skeleton).
        let beta_1_raw = self.edge_count as i64 - n as i64 + beta_0 as i64;
        let beta_1 = beta_1_raw.max(0) as usize;

        // β₂ ≈ triangle count (simplified — each enclosed triangle is a potential void).
        let beta_2 = self.triangle_count;

        BettiNumbers::new(beta_0, beta_1, beta_2)
    }

    /// Compute persistence pairs (dimension-0 only for simplicity).
    ///
    /// Sorts all edges by length, processes via union-find. Each merge event
    /// produces a death for the younger component; the oldest component persists.
    fn compute_persistence(&self) -> Vec<PersistencePair> {
        let n = self.points.len();
        if n == 0 {
            return Vec::new();
        }

        // Collect all edges with their distances.
        let mut edges: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..n {
            for &j in &self.adjacency[i] {
                if j > i {
                    let d = self.points[i].distance(&self.points[j]);
                    edges.push((i, j, d));
                }
            }
        }
        edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        // Every vertex is born at distance 0.
        let mut birth: Vec<f64> = vec![0.0; n];
        let mut uf = UnionFind::new(n);
        let mut pairs = Vec::new();

        for (a, b, dist) in &edges {
            let ra = uf.find(*a);
            let rb = uf.find(*b);
            if ra != rb {
                // The younger component dies.
                let (dying_birth, _surviving_birth) = if birth[ra] >= birth[rb] {
                    (birth[ra], birth[rb])
                } else {
                    (birth[rb], birth[ra])
                };
                pairs.push(PersistencePair {
                    dimension: 0,
                    birth: dying_birth,
                    death: Some(*dist),
                });
                // Merge — keep the older birth value.
                uf.union(*a, *b);
                let root = uf.find(*a);
                birth[root] = birth[ra].min(birth[rb]);
            }
        }

        // The last surviving component persists forever.
        if n > 0 {
            pairs.push(PersistencePair {
                dimension: 0,
                birth: 0.0,
                death: None,
            });
        }

        pairs
    }

    /// Detect wave packets from recent amplitude increases.
    fn detect_wave_packets(&self) -> Vec<WavePacket> {
        let n = self.points.len();
        if n < 3 {
            return Vec::new();
        }

        let mut packets = Vec::new();

        // Look for local maxima in the norm of consciousness dimensions.
        for i in 1..(n - 1) {
            let norm_prev = norm7(&self.points[i - 1].dimensions);
            let norm_curr = norm7(&self.points[i].dimensions);
            let norm_next = norm7(&self.points[i + 1].dimensions);

            if norm_curr > norm_prev && norm_curr > norm_next {
                let amplitude = norm_curr - (norm_prev + norm_next) / 2.0;
                if amplitude > 0.05 {
                    packets.push(WavePacket {
                        center: self.points[i].dimensions,
                        amplitude,
                        frequency: std::f64::consts::TAU
                            / (self.points[i]
                                .cycle
                                .saturating_sub(self.points[i.saturating_sub(2)].cycle)
                                as f64)
                                .max(1.0),
                        width: self.config.edge_threshold,
                        created_cycle: self.points[i].cycle,
                    });
                }
            }
        }

        packets
    }

    /// Detect interference between wave packets.
    fn detect_interference(&self, packets: &[WavePacket]) -> Vec<InterferenceEvent> {
        let mut events = Vec::new();
        let latest_cycle = self.points.last().map(|p| p.cycle).unwrap_or(0);

        for i in 0..packets.len() {
            for j in (i + 1)..packets.len() {
                // Check overlap: midpoint between centers.
                let mut midpoint = [0.0f64; 7];
                for ((m, &ci), &cj) in midpoint
                    .iter_mut()
                    .zip(packets[i].center.iter())
                    .zip(packets[j].center.iter())
                {
                    *m = (ci + cj) / 2.0;
                }

                let time = latest_cycle as f64;
                let val_i = packets[i].evaluate(&midpoint, time);
                let val_j = packets[j].evaluate(&midpoint, time);
                let combined = val_i + val_j;
                let sum_abs = val_i.abs() + val_j.abs();

                if sum_abs < 1e-12 {
                    continue;
                }

                let ratio = combined.abs() / sum_abs;
                let interference_type = if ratio > 0.8 {
                    InterferenceType::Constructive
                } else if ratio < 0.2 {
                    InterferenceType::Destructive
                } else {
                    InterferenceType::Partial
                };

                events.push(InterferenceEvent {
                    position: midpoint,
                    interference_type,
                    magnitude: combined.abs(),
                    cycle: latest_cycle,
                });
            }
        }

        events
    }

    /// Run full topology analysis on the current point window.
    pub fn analyze(&mut self) -> TopologyAnalysis {
        // Ensure complex is up to date
        if self.adjacency.len() != self.points.len() && !self.points.is_empty() {
            self.rebuild_complex();
        }
        let betti = self.compute_betti();
        let persistence_pairs = self.compute_persistence();
        let wave_packets = self.detect_wave_packets();
        let interference_events = self.detect_interference(&wave_packets);
        let interpretation = TopologyInterpretation::from_betti(&betti);

        TopologyAnalysis {
            points_analyzed: self.points.len(),
            betti,
            persistence_pairs,
            wave_packets,
            interference_events,
            interpretation,
        }
    }

    /// Human-readable topology report.
    pub fn report(&mut self) -> String {
        let a = self.analyze();
        format!(
            "Topology Report ({} points)\n\
             ──────────────────────────\n\
             Betti numbers: β₀={}, β₁={}, β₂={}\n\
             Euler characteristic: χ={}\n\
             Unity: {:.2}, Complexity: {:.2}, Fragmentation: {:.2}\n\
             Persistent features: {} ({} infinite)\n\
             Wave packets: {}, Interference events: {}",
            a.points_analyzed,
            a.betti.beta_0,
            a.betti.beta_1,
            a.betti.beta_2,
            a.betti.euler_characteristic,
            a.interpretation.unity,
            a.interpretation.complexity,
            a.interpretation.fragmentation,
            a.persistence_pairs.len(),
            a.persistence_pairs
                .iter()
                .filter(|p| p.death.is_none())
                .count(),
            a.wave_packets.len(),
            a.interference_events.len(),
        )
    }

    /// Running statistics.
    pub fn stats(&self) -> TopologyStats {
        TopologyStats {
            total_observations: self.total_observations,
            current_window_size: self.points.len(),
            rebuilds: self.rebuilds,
            last_rebuild_cycle: self.last_rebuild_cycle,
        }
    }
}

/// Euclidean norm of a 7D vector.
fn norm7(v: &[f64; 7]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_distance() {
        let a = ConsciousnessPoint {
            dimensions: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            cycle: 0,
        };
        let b = ConsciousnessPoint {
            dimensions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            cycle: 1,
        };
        assert!((a.distance(&b) - 1.0).abs() < 1e-10);

        let c = ConsciousnessPoint {
            dimensions: [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            cycle: 2,
        };
        assert!((a.distance(&c) - 1.0).abs() < 1e-10);
        assert!((b.distance(&c) - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_betti_single_component() {
        // All points identical → β₀=1, β₁=0, β₂=0
        let mut analyzer = TopologyAnalyzer::new(TopologyConfig {
            max_points: 50,
            edge_threshold: 1.0,
            rebuild_interval: 1,
        });
        for i in 0..5 {
            analyzer.observe([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], i);
        }
        let a = analyzer.analyze();
        assert_eq!(a.betti.beta_0, 1, "identical points → one component");
        assert_eq!(
            a.betti.euler_characteristic,
            1 - a.betti.beta_1 as i64 + a.betti.beta_2 as i64
        );
    }

    #[test]
    fn test_betti_two_clusters() {
        // Two well-separated clusters → β₀=2
        let mut analyzer = TopologyAnalyzer::new(TopologyConfig {
            max_points: 50,
            edge_threshold: 0.3,
            rebuild_interval: 1,
        });
        // Cluster A near origin
        for i in 0..4 {
            analyzer.observe([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], i);
        }
        // Cluster B far away
        for i in 4..8 {
            analyzer.observe([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], i);
        }
        let a = analyzer.analyze();
        assert_eq!(a.betti.beta_0, 2, "two separated clusters → two components");
    }

    #[test]
    fn test_union_find() {
        let mut uf = UnionFind::new(5);
        assert_eq!(uf.components(), 5);
        uf.union(0, 1);
        assert_eq!(uf.components(), 4);
        uf.union(2, 3);
        assert_eq!(uf.components(), 3);
        uf.union(1, 3);
        assert_eq!(uf.components(), 2);
        // 0,1,2,3 connected; 4 alone
        assert_eq!(uf.find(0), uf.find(3));
        assert_ne!(uf.find(0), uf.find(4));
    }

    #[test]
    fn test_wave_packet_energy() {
        let wp = WavePacket {
            center: [0.0; 7],
            amplitude: 2.0,
            frequency: 1.0,
            width: 1.0,
            created_cycle: 0,
        };
        // energy = 2^2 * 1^3.5 = 4.0
        assert!((wp.energy() - 4.0).abs() < 1e-10);

        let wp2 = WavePacket {
            center: [0.0; 7],
            amplitude: 1.0,
            frequency: 1.0,
            width: 2.0,
            created_cycle: 0,
        };
        // energy = 1^2 * 2^3.5 = 11.3137…
        assert!((wp2.energy() - 2.0_f64.powf(3.5)).abs() < 1e-8);
    }

    #[test]
    fn test_interpretation_values() {
        let b = BettiNumbers::new(1, 0, 0);
        let interp = TopologyInterpretation::from_betti(&b);
        assert!((interp.unity - 1.0).abs() < 1e-10);
        assert!((interp.complexity - 0.0).abs() < 1e-10);
        assert!((interp.fragmentation - 0.0).abs() < 1e-10);

        let b2 = BettiNumbers::new(4, 3, 1);
        let interp2 = TopologyInterpretation::from_betti(&b2);
        assert!((interp2.unity - 0.25).abs() < 1e-10);
        assert!((interp2.complexity - 0.5).abs() < 1e-10); // (3 + 2*1)/10
        assert!((interp2.fragmentation - 0.3).abs() < 1e-10); // (4-1)/10
    }

    #[test]
    fn test_persistence_has_infinite_feature() {
        let mut analyzer = TopologyAnalyzer::new(TopologyConfig {
            max_points: 50,
            edge_threshold: 2.0,
            rebuild_interval: 1,
        });
        for i in 0..5 {
            analyzer.observe([i as f64 * 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], i);
        }
        let a = analyzer.analyze();
        let infinite = a
            .persistence_pairs
            .iter()
            .filter(|p| p.death.is_none())
            .count();
        assert!(infinite >= 1, "at least one infinite persistence feature");
    }
}
