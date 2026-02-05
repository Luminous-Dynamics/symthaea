//! # Topological Consciousness Router
//!
//! Uses persistent homology to detect stable topological features in
//! consciousness state space and route accordingly.
//!
//! ## Key Concepts
//!
//! - **Betti Numbers**: β₀ (components), β₁ (loops), β₂ (voids)
//! - **Persistence**: Features that persist across scales are fundamental
//! - **Topological Data Analysis**: Point cloud → simplicial complex → homology
//!
//! ## Mathematical Foundation
//!
//! - Vietoris-Rips complex from consciousness state point cloud
//! - Persistent homology via matrix reduction
//! - Betti curves and persistence barcodes

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

use super::{
    RoutingStrategy,
    InformationGeometricRouter, GeometricRouterConfig,
    LatentConsciousnessState,
};

// ═══════════════════════════════════════════════════════════════════════════
// SIMPLICIAL COMPLEX TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// A simplex in the consciousness simplicial complex
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Simplex {
    /// Vertex indices forming this simplex
    pub vertices: Vec<usize>,
    /// Dimension (0=point, 1=edge, 2=triangle, etc.)
    pub dimension: usize,
    /// Filtration value (radius at which simplex appears)
    pub filtration: f64,
    /// Whether this simplex is part of the boundary
    pub is_boundary: bool,
}

impl Simplex {
    /// Create a new simplex
    pub fn new(vertices: Vec<usize>, filtration: f64) -> Self {
        let dimension = if vertices.is_empty() { 0 } else { vertices.len() - 1 };
        Self {
            vertices,
            dimension,
            filtration,
            is_boundary: false,
        }
    }

    /// Get the boundary simplices (faces)
    pub fn boundary(&self) -> Vec<Simplex> {
        if self.dimension == 0 {
            return vec![];
        }

        let mut faces = Vec::with_capacity(self.vertices.len());
        for i in 0..self.vertices.len() {
            let mut face_vertices = self.vertices.clone();
            face_vertices.remove(i);
            faces.push(Simplex::new(face_vertices, self.filtration));
        }
        faces
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PERSISTENCE TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// A persistence interval (birth, death) for a topological feature
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PersistenceInterval {
    /// Birth time (filtration value when feature appears)
    pub birth: f64,
    /// Death time (filtration value when feature dies, infinity if never)
    pub death: f64,
    /// Dimension of the feature (0=component, 1=loop, 2=void)
    pub dimension: usize,
    /// Representative simplex index
    pub representative: Option<usize>,
}

impl PersistenceInterval {
    /// Compute the persistence (lifetime) of this feature
    pub fn persistence(&self) -> f64 {
        if self.death.is_infinite() {
            f64::INFINITY
        } else {
            self.death - self.birth
        }
    }

    /// Check if this is an essential feature (never dies)
    pub fn is_essential(&self) -> bool {
        self.death.is_infinite()
    }
}

/// Persistence diagram: collection of persistence intervals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceDiagram {
    /// All persistence intervals
    pub intervals: Vec<PersistenceInterval>,
    /// Maximum dimension computed
    pub max_dimension: usize,
    /// Total number of simplices processed
    pub num_simplices: usize,
}

impl PersistenceDiagram {
    /// Create an empty persistence diagram
    pub fn new(max_dimension: usize) -> Self {
        Self {
            intervals: Vec::new(),
            max_dimension,
            num_simplices: 0,
        }
    }

    /// Get intervals of a specific dimension
    pub fn intervals_dim(&self, dim: usize) -> Vec<&PersistenceInterval> {
        self.intervals.iter().filter(|i| i.dimension == dim).collect()
    }

    /// Compute Betti number at a given filtration value
    pub fn betti_number(&self, dim: usize, filtration: f64) -> usize {
        self.intervals
            .iter()
            .filter(|i| i.dimension == dim && i.birth <= filtration && i.death > filtration)
            .count()
    }

    /// Get total persistence in dimension d
    pub fn total_persistence(&self, dim: usize) -> f64 {
        self.intervals
            .iter()
            .filter(|i| i.dimension == dim && !i.death.is_infinite())
            .map(|i| i.persistence())
            .sum()
    }

    /// Get the most persistent features in dimension d
    pub fn most_persistent(&self, dim: usize, k: usize) -> Vec<&PersistenceInterval> {
        let mut intervals: Vec<_> = self.intervals_dim(dim);
        intervals.sort_by(|a, b| b.persistence().partial_cmp(&a.persistence()).unwrap());
        intervals.into_iter().take(k).collect()
    }

    /// Compute persistence entropy (topological complexity measure)
    pub fn persistence_entropy(&self, dim: usize) -> f64 {
        let total = self.total_persistence(dim);
        if total <= 0.0 {
            return 0.0;
        }

        let intervals: Vec<_> = self.intervals
            .iter()
            .filter(|i| i.dimension == dim && !i.death.is_infinite())
            .collect();

        let mut entropy = 0.0;
        for interval in intervals {
            let p = interval.persistence() / total;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }
        entropy
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// VIETORIS-RIPS COMPLEX
// ═══════════════════════════════════════════════════════════════════════════

/// Vietoris-Rips complex builder for consciousness states
#[derive(Debug, Clone)]
pub struct VietorisRipsComplex {
    /// Points (consciousness states as vectors)
    pub points: Vec<Vec<f64>>,
    /// All simplices in the complex
    pub simplices: Vec<Simplex>,
    /// Maximum dimension to compute
    pub max_dim: usize,
    /// Maximum filtration value
    pub max_filtration: f64,
    /// Number of filtration steps
    pub num_steps: usize,
}

impl VietorisRipsComplex {
    /// Create a new Vietoris-Rips complex builder
    pub fn new(max_dim: usize, max_filtration: f64, num_steps: usize) -> Self {
        Self {
            points: Vec::new(),
            simplices: Vec::new(),
            max_dim,
            max_filtration,
            num_steps,
        }
    }

    /// Add a point (consciousness state)
    pub fn add_point(&mut self, point: Vec<f64>) {
        self.points.push(point);
    }

    /// Compute Euclidean distance between two points
    fn distance(&self, i: usize, j: usize) -> f64 {
        let p1 = &self.points[i];
        let p2 = &self.points[j];
        let mut sum = 0.0;
        for k in 0..p1.len().min(p2.len()) {
            sum += (p1[k] - p2[k]).powi(2);
        }
        sum.sqrt()
    }

    /// Build the filtered simplicial complex
    pub fn build(&mut self) {
        self.simplices.clear();
        let n = self.points.len();
        if n == 0 {
            return;
        }

        // Precompute all pairwise distances
        let mut distances: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = self.distance(i, j);
                distances[i][j] = d;
                distances[j][i] = d;
            }
        }

        // Add 0-simplices (vertices) at filtration 0
        for i in 0..n {
            self.simplices.push(Simplex::new(vec![i], 0.0));
        }

        // Add higher-dimensional simplices based on filtration
        for dim in 1..=self.max_dim {
            self.add_simplices_of_dimension(dim, &distances);
        }

        // Sort by filtration value
        self.simplices.sort_by(|a, b| {
            a.filtration.partial_cmp(&b.filtration).unwrap()
        });
    }

    /// Add all simplices of a given dimension
    fn add_simplices_of_dimension(&mut self, dim: usize, distances: &[Vec<f64>]) {
        let n = self.points.len();
        if dim > n {
            return;
        }

        // Generate all (dim+1)-subsets of vertices
        let mut indices = vec![0usize; dim + 1];
        for i in 0..=dim {
            indices[i] = i;
        }

        loop {
            // Check if this subset forms a valid simplex
            let filt = self.simplex_filtration(&indices, distances);
            if filt <= self.max_filtration {
                self.simplices.push(Simplex::new(indices.clone(), filt));
            }

            // Generate next combination
            let mut k = dim;
            loop {
                indices[k] += 1;
                if indices[k] <= n - dim + k - 1 {
                    for j in (k + 1)..=dim {
                        indices[j] = indices[j - 1] + 1;
                    }
                    break;
                }
                if k == 0 {
                    return;
                }
                k -= 1;
            }
        }
    }

    /// Compute the filtration value for a simplex (max edge length)
    fn simplex_filtration(&self, vertices: &[usize], distances: &[Vec<f64>]) -> f64 {
        let mut max_dist = 0.0;
        for i in 0..vertices.len() {
            for j in (i + 1)..vertices.len() {
                let d = distances[vertices[i]][vertices[j]];
                if d > max_dist {
                    max_dist = d;
                }
            }
        }
        max_dist
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PERSISTENT HOMOLOGY
// ═══════════════════════════════════════════════════════════════════════════

/// Persistent homology computer using matrix reduction
pub struct PersistentHomology {
    /// The simplicial complex
    pub complex: VietorisRipsComplex,
    /// Boundary matrix (sparse representation)
    boundary_matrix: Vec<Vec<usize>>,
    /// Low values for reduced matrix
    low: Vec<Option<usize>>,
}

impl PersistentHomology {
    /// Create a new persistent homology computer
    pub fn new(complex: VietorisRipsComplex) -> Self {
        Self {
            complex,
            boundary_matrix: Vec::new(),
            low: Vec::new(),
        }
    }

    /// Compute persistent homology
    pub fn compute(&mut self) -> PersistenceDiagram {
        let n = self.complex.simplices.len();
        if n == 0 {
            return PersistenceDiagram::new(0);
        }

        self.build_boundary_matrix();
        self.reduce_matrix();
        self.extract_persistence()
    }

    /// Build the boundary matrix
    fn build_boundary_matrix(&mut self) {
        let n = self.complex.simplices.len();
        self.boundary_matrix = vec![Vec::new(); n];

        let mut simplex_map: HashMap<Vec<usize>, usize> = HashMap::new();
        for (i, s) in self.complex.simplices.iter().enumerate() {
            simplex_map.insert(s.vertices.clone(), i);
        }

        for (j, simplex) in self.complex.simplices.iter().enumerate() {
            if simplex.dimension == 0 {
                continue;
            }

            let faces = simplex.boundary();
            for face in faces {
                if let Some(&i) = simplex_map.get(&face.vertices) {
                    self.boundary_matrix[j].push(i);
                }
            }
            self.boundary_matrix[j].sort();
        }
    }

    /// Reduce the boundary matrix
    fn reduce_matrix(&mut self) {
        let n = self.boundary_matrix.len();
        self.low = vec![None; n];

        for j in 0..n {
            self.reduce_column(j);
        }
    }

    /// Reduce column j
    fn reduce_column(&mut self, j: usize) {
        while let Some(low_j) = self.get_low(j) {
            let mut found = None;
            for i in 0..j {
                if let Some(low_i) = self.low[i] {
                    if low_i == low_j {
                        found = Some(i);
                        break;
                    }
                }
            }

            if let Some(i) = found {
                self.add_column(i, j);
            } else {
                self.low[j] = Some(low_j);
                return;
            }
        }
        self.low[j] = None;
    }

    /// Get the lowest nonzero index in column j
    fn get_low(&self, j: usize) -> Option<usize> {
        self.boundary_matrix[j].last().copied()
    }

    /// Add column i to column j (mod 2)
    fn add_column(&mut self, i: usize, j: usize) {
        let col_i = self.boundary_matrix[i].clone();
        let col_j = &mut self.boundary_matrix[j];

        let mut result = Vec::new();
        let (mut pi, mut pj) = (0, 0);
        while pi < col_i.len() && pj < col_j.len() {
            if col_i[pi] < col_j[pj] {
                result.push(col_i[pi]);
                pi += 1;
            } else if col_i[pi] > col_j[pj] {
                result.push(col_j[pj]);
                pj += 1;
            } else {
                pi += 1;
                pj += 1;
            }
        }
        result.extend_from_slice(&col_i[pi..]);
        result.extend_from_slice(&col_j[pj..]);

        *col_j = result;
    }

    /// Extract persistence pairs from reduced matrix
    fn extract_persistence(&self) -> PersistenceDiagram {
        let mut diagram = PersistenceDiagram::new(self.complex.max_dim);
        diagram.num_simplices = self.complex.simplices.len();

        let n = self.boundary_matrix.len();
        let mut paired = vec![false; n];

        for j in 0..n {
            if let Some(i) = self.low[j] {
                paired[i] = true;
                paired[j] = true;

                let birth = self.complex.simplices[i].filtration;
                let death = self.complex.simplices[j].filtration;
                let dim = self.complex.simplices[i].dimension;

                if birth < death {
                    diagram.intervals.push(PersistenceInterval {
                        birth,
                        death,
                        dimension: dim,
                        representative: Some(i),
                    });
                }
            }
        }

        for i in 0..n {
            if !paired[i] && self.boundary_matrix[i].is_empty() {
                let birth = self.complex.simplices[i].filtration;
                let dim = self.complex.simplices[i].dimension;

                diagram.intervals.push(PersistenceInterval {
                    birth,
                    death: f64::INFINITY,
                    dimension: dim,
                    representative: Some(i),
                });
            }
        }

        diagram
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TOPOLOGICAL SIGNATURE
// ═══════════════════════════════════════════════════════════════════════════

/// Topological signature of a consciousness state region
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalSignature {
    /// Betti numbers at characteristic scales
    pub betti_numbers: Vec<Vec<usize>>,
    /// Persistence entropy per dimension
    pub persistence_entropy: Vec<f64>,
    /// Total persistence per dimension
    pub total_persistence: Vec<f64>,
    /// Number of essential features per dimension
    pub essential_features: Vec<usize>,
    /// Most persistent feature lifetimes
    pub top_lifetimes: Vec<Vec<f64>>,
    /// Wasserstein distance from reference topology
    pub topology_distance: f64,
}

impl TopologicalSignature {
    /// Compute signature from persistence diagram
    pub fn from_diagram(diagram: &PersistenceDiagram, num_scales: usize) -> Self {
        let max_dim = diagram.max_dimension;

        let max_filt = diagram.intervals
            .iter()
            .filter(|i| !i.death.is_infinite())
            .map(|i| i.death)
            .fold(0.0, f64::max);

        let scales: Vec<f64> = (0..num_scales)
            .map(|i| (i as f64 / num_scales as f64) * max_filt.max(1.0))
            .collect();

        let mut betti_numbers = Vec::new();
        for scale in &scales {
            let mut betti_at_scale = Vec::new();
            for dim in 0..=max_dim {
                betti_at_scale.push(diagram.betti_number(dim, *scale));
            }
            betti_numbers.push(betti_at_scale);
        }

        let persistence_entropy: Vec<f64> = (0..=max_dim)
            .map(|d| diagram.persistence_entropy(d))
            .collect();

        let total_persistence: Vec<f64> = (0..=max_dim)
            .map(|d| diagram.total_persistence(d))
            .collect();

        let essential_features: Vec<usize> = (0..=max_dim)
            .map(|d| diagram.intervals_dim(d).iter().filter(|i| i.is_essential()).count())
            .collect();

        let top_lifetimes: Vec<Vec<f64>> = (0..=max_dim)
            .map(|d| {
                diagram.most_persistent(d, 5)
                    .iter()
                    .filter(|i| !i.death.is_infinite())
                    .map(|i| i.persistence())
                    .collect()
            })
            .collect();

        Self {
            betti_numbers,
            persistence_entropy,
            total_persistence,
            essential_features,
            top_lifetimes,
            topology_distance: 0.0,
        }
    }

    /// Compute complexity score from topological features
    pub fn complexity_score(&self) -> f64 {
        let entropy_score: f64 = self.persistence_entropy.iter().sum();
        let betti_score: f64 = self.betti_numbers
            .iter()
            .flat_map(|b| b.iter())
            .map(|&b| b as f64)
            .sum::<f64>() / (self.betti_numbers.len().max(1) as f64);
        let essential_score: f64 = self.essential_features.iter().map(|&e| e as f64).sum();

        (entropy_score + betti_score + essential_score * 0.5) / 3.0
    }

    /// Compute stability score
    pub fn stability_score(&self) -> f64 {
        let persistence_score: f64 = self.total_persistence.iter().sum();
        let cohesion_score = 1.0 / (1.0 + self.essential_features.first().copied().unwrap_or(1) as f64);

        (persistence_score.tanh() + cohesion_score) / 2.0
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TOPOLOGICAL ROUTER
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for topological router
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalRouterConfig {
    /// Maximum dimension for homology computation
    pub max_dimension: usize,
    /// Maximum filtration radius
    pub max_filtration: f64,
    /// Number of filtration steps
    pub num_filtration_steps: usize,
    /// Minimum points needed for topology
    pub min_points: usize,
    /// Maximum points in sliding window
    pub max_points: usize,
    /// Complexity threshold for upgrading strategy
    pub complexity_threshold: f64,
    /// Stability threshold for downgrading strategy
    pub stability_threshold: f64,
    /// Weight for topological vs geometric routing
    pub topology_weight: f64,
}

impl Default for TopologicalRouterConfig {
    fn default() -> Self {
        Self {
            max_dimension: 2,
            max_filtration: 2.0,
            num_filtration_steps: 10,
            min_points: 5,
            max_points: 50,
            complexity_threshold: 2.0,
            stability_threshold: 0.7,
            topology_weight: 0.5,
        }
    }
}

/// Statistics for topological router
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TopologicalRouterStats {
    pub decisions_made: usize,
    pub homology_computations: usize,
    pub topology_upgrades: usize,
    pub stability_downgrades: usize,
    pub avg_complexity: f64,
    pub avg_stability: f64,
    pub total_simplices: usize,
    pub topological_transitions: usize,
}

/// Topological routing decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalRoutingDecision {
    pub strategy: RoutingStrategy,
    pub signature: TopologicalSignature,
    pub complexity: f64,
    pub stability: f64,
    pub transition_detected: bool,
    pub betti_at_decision: Vec<usize>,
    pub should_explore: bool,
}

/// Summary of topological router state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalRouterSummary {
    pub num_points: usize,
    pub current_betti: Vec<usize>,
    pub complexity: f64,
    pub stability: f64,
    pub decisions: usize,
    pub transitions: usize,
}

/// Topological Consciousness Router
pub struct TopologicalConsciousnessRouter {
    geometric_router: InformationGeometricRouter,
    config: TopologicalRouterConfig,
    state_history: VecDeque<Vec<f64>>,
    current_diagram: Option<PersistenceDiagram>,
    current_signature: Option<TopologicalSignature>,
    previous_signature: Option<TopologicalSignature>,
    stats: TopologicalRouterStats,
}

impl TopologicalConsciousnessRouter {
    /// Create a new topological router
    pub fn new(config: TopologicalRouterConfig) -> Self {
        Self {
            geometric_router: InformationGeometricRouter::new(GeometricRouterConfig::default()),
            config,
            state_history: VecDeque::with_capacity(100),
            current_diagram: None,
            current_signature: None,
            previous_signature: None,
            stats: TopologicalRouterStats::default(),
        }
    }

    /// Observe a new consciousness state
    pub fn observe_state(&mut self, state: &LatentConsciousnessState) {
        let point = vec![state.phi, state.integration, state.coherence, state.attention];

        self.state_history.push_back(point);
        while self.state_history.len() > self.config.max_points {
            self.state_history.pop_front();
        }

        self.geometric_router.observe_state(state);
    }

    /// Compute persistent homology on current state history
    fn compute_homology(&mut self) -> Option<PersistenceDiagram> {
        if self.state_history.len() < self.config.min_points {
            return None;
        }

        let mut complex = VietorisRipsComplex::new(
            self.config.max_dimension,
            self.config.max_filtration,
            self.config.num_filtration_steps,
        );

        for point in &self.state_history {
            complex.add_point(point.clone());
        }
        complex.build();

        let mut ph = PersistentHomology::new(complex);
        let diagram = ph.compute();

        self.stats.homology_computations += 1;
        self.stats.total_simplices += diagram.num_simplices;

        Some(diagram)
    }

    /// Detect topological transition
    fn detect_transition(&self, new_sig: &TopologicalSignature) -> bool {
        let Some(old_sig) = &self.previous_signature else {
            return false;
        };

        let mut betti_change = 0.0;
        for (old_betti, new_betti) in old_sig.betti_numbers.iter().zip(new_sig.betti_numbers.iter()) {
            for (o, n) in old_betti.iter().zip(new_betti.iter()) {
                betti_change += (*o as f64 - *n as f64).abs();
            }
        }

        let entropy_change: f64 = old_sig.persistence_entropy
            .iter()
            .zip(new_sig.persistence_entropy.iter())
            .map(|(o, n)| (o - n).abs())
            .sum();

        betti_change > 2.0 || entropy_change > 0.5
    }

    /// Route based on topological analysis
    pub fn route(&mut self, target: &LatentConsciousnessState) -> TopologicalRoutingDecision {
        let diagram = self.compute_homology();
        self.current_diagram = diagram.clone();

        let geo_decision = self.geometric_router.route(target);

        let signature = if let Some(ref diag) = diagram {
            TopologicalSignature::from_diagram(diag, 5)
        } else {
            TopologicalSignature {
                betti_numbers: vec![vec![1, 0, 0]],
                persistence_entropy: vec![0.0; 3],
                total_persistence: vec![0.0; 3],
                essential_features: vec![1, 0, 0],
                top_lifetimes: vec![vec![], vec![], vec![]],
                topology_distance: 0.0,
            }
        };

        let complexity = signature.complexity_score();
        let stability = signature.stability_score();

        let transition_detected = self.detect_transition(&signature);
        if transition_detected {
            self.stats.topological_transitions += 1;
        }

        let n = self.stats.decisions_made as f64;
        self.stats.avg_complexity = (self.stats.avg_complexity * n + complexity) / (n + 1.0);
        self.stats.avg_stability = (self.stats.avg_stability * n + stability) / (n + 1.0);

        let base_strategy = geo_decision.strategy;
        let strategy = self.select_strategy(base_strategy, &signature, complexity, stability, transition_detected);

        let betti_at_decision = if !signature.betti_numbers.is_empty() {
            signature.betti_numbers[signature.betti_numbers.len() / 2].clone()
        } else {
            vec![1, 0, 0]
        };

        let should_explore = complexity < 1.0 && stability > 0.5;

        self.previous_signature = self.current_signature.take();
        self.current_signature = Some(signature.clone());

        self.stats.decisions_made += 1;

        TopologicalRoutingDecision {
            strategy,
            signature,
            complexity,
            stability,
            transition_detected,
            betti_at_decision,
            should_explore,
        }
    }

    /// Select strategy based on topological features
    fn select_strategy(
        &mut self,
        base_strategy: RoutingStrategy,
        signature: &TopologicalSignature,
        complexity: f64,
        stability: f64,
        transition_detected: bool,
    ) -> RoutingStrategy {
        if transition_detected {
            self.stats.topology_upgrades += 1;
            return RoutingStrategy::FullDeliberation;
        }

        if complexity > self.config.complexity_threshold {
            self.stats.topology_upgrades += 1;
            return match base_strategy {
                RoutingStrategy::Reflexive | RoutingStrategy::FastPatterns => {
                    RoutingStrategy::HeuristicGuided
                }
                RoutingStrategy::HeuristicGuided => RoutingStrategy::StandardProcessing,
                RoutingStrategy::StandardProcessing => RoutingStrategy::FullDeliberation,
                other => other,
            };
        }

        if stability > self.config.stability_threshold {
            self.stats.stability_downgrades += 1;
            return match base_strategy {
                RoutingStrategy::FullDeliberation => RoutingStrategy::StandardProcessing,
                RoutingStrategy::StandardProcessing => RoutingStrategy::HeuristicGuided,
                RoutingStrategy::HeuristicGuided => RoutingStrategy::FastPatterns,
                other => other,
            };
        }

        if signature.essential_features.len() > 1 && signature.essential_features[1] > 0 {
            return RoutingStrategy::Ensemble;
        }

        if signature.essential_features.first().copied().unwrap_or(1) > 1 {
            return RoutingStrategy::StandardProcessing;
        }

        base_strategy
    }

    /// Get the current Betti numbers
    pub fn current_betti_numbers(&self) -> Vec<usize> {
        if let Some(ref sig) = self.current_signature {
            if !sig.betti_numbers.is_empty() {
                return sig.betti_numbers[sig.betti_numbers.len() / 2].clone();
            }
        }
        vec![1, 0, 0]
    }

    /// Check if we're in a topologically complex region
    pub fn is_complex_region(&self) -> bool {
        if let Some(ref sig) = self.current_signature {
            sig.complexity_score() > self.config.complexity_threshold
        } else {
            false
        }
    }

    /// Get summary of router state
    pub fn summary(&self) -> TopologicalRouterSummary {
        let (complexity, stability) = if let Some(ref sig) = self.current_signature {
            (sig.complexity_score(), sig.stability_score())
        } else {
            (0.0, 1.0)
        };

        TopologicalRouterSummary {
            num_points: self.state_history.len(),
            current_betti: self.current_betti_numbers(),
            complexity,
            stability,
            decisions: self.stats.decisions_made,
            transitions: self.stats.topological_transitions,
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> &TopologicalRouterStats {
        &self.stats
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplex_creation() {
        let simplex = Simplex::new(vec![0, 1, 2], 0.5);
        assert_eq!(simplex.dimension, 2);
        assert_eq!(simplex.filtration, 0.5);
    }

    #[test]
    fn test_simplex_boundary() {
        let triangle = Simplex::new(vec![0, 1, 2], 0.5);
        let boundary = triangle.boundary();

        assert_eq!(boundary.len(), 3);
    }

    #[test]
    fn test_persistence_interval() {
        let interval = PersistenceInterval {
            birth: 0.1,
            death: 0.5,
            dimension: 1,
            representative: Some(0),
        };

        assert!((interval.persistence() - 0.4).abs() < 0.001);
        assert!(!interval.is_essential());
    }

    #[test]
    fn test_topological_router_creation() {
        let router = TopologicalConsciousnessRouter::new(TopologicalRouterConfig::default());
        assert_eq!(router.stats.decisions_made, 0);
    }
}
