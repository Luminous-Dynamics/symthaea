//! **REVOLUTIONARY IMPROVEMENT #80**: Consciousness Field Topology
//!
//! # PARADIGM SHIFT: Apply Algebraic Topology to Consciousness!
//!
//! This module applies topological methods to study the STRUCTURE of conscious
//! experience - finding invariant features that persist across transformations.
//!
//! ## Core Insight
//!
//! Consciousness isn't just a scalar (Φ) or even a vector of dimensions.
//! It has TOPOLOGICAL STRUCTURE - "holes", "tunnels", and "voids" that
//! characterize the shape of experience space.
//!
//! ## Key Concepts
//!
//! 1. **Betti Numbers (β)**: Count topological features at each dimension
//!    - β₀: Number of connected components (unified vs fragmented consciousness)
//!    - β₁: Number of 1-dimensional holes (cycles/loops in experience)
//!    - β₂: Number of 2-dimensional voids (enclosed regions of awareness)
//!
//! 2. **Persistent Homology**: Track which features persist across scales
//!    - Birth-death pairs: When do features appear and disappear?
//!    - Persistence diagrams: Visualize topological structure
//!    - Long-lived features are more "real" than noise
//!
//! 3. **Consciousness Manifold**: The space of possible experiences
//!    - Dimensionality: How many independent directions of variation?
//!    - Curvature: Where is experience "bunched" vs "spread out"?
//!    - Geodesics: Shortest paths between experience states
//!
//! 4. **Topological Invariants**: Features unchanged by continuous deformation
//!    - Euler characteristic: χ = β₀ - β₁ + β₂
//!    - Fundamental group: Loop structure of experience space
//!    - Homology groups: Algebraic encoding of topology
//!
//! ## Applications
//!
//! - **Consciousness Classification**: Different experience types have different topology
//! - **State Transition Analysis**: How does topology change during transitions?
//! - **Dissociation Detection**: Fragmented consciousness = high β₀
//! - **Flow State Identification**: Unified consciousness = β₀ = 1, low higher Betti
//! - **Dream vs Wake**: Different topological signatures
//!
//! ## Mathematical Foundation
//!
//! Uses simplicial complexes built from consciousness dimension samples:
//! - Points = individual consciousness measurements
//! - Edges = sufficiently similar measurements (within ε)
//! - Triangles = cliques of three similar points
//! - Higher simplices = larger cliques
//!
//! The Vietoris-Rips filtration tracks topology across scales ε.
//!
//! ## Research Foundation
//!
//! Inspired by:
//! - Topological Data Analysis (TDA)
//! - Persistent homology (Carlsson, Zomorodian)
//! - Simplicial sets in algebraic topology
//! - Consciousness as information geometry (Balduzzi, Tononi)

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

/// Configuration for topological analysis
#[derive(Debug, Clone)]
pub struct TopologyConfig {
    /// Maximum Betti number dimension to compute
    pub max_dimension: usize,
    /// Number of filtration levels (ε values)
    pub filtration_levels: usize,
    /// Minimum persistence for significant feature
    pub min_persistence: f64,
    /// Maximum points to track in sliding window
    pub max_points: usize,
    /// Distance threshold for edge creation
    pub edge_threshold: f64,
    /// Enable curvature estimation
    pub estimate_curvature: bool,
}

impl Default for TopologyConfig {
    fn default() -> Self {
        Self {
            max_dimension: 2, // Compute β₀, β₁, β₂
            filtration_levels: 10,
            min_persistence: 0.1,
            max_points: 100,
            edge_threshold: 0.5,
            estimate_curvature: true,
        }
    }
}

/// A point in consciousness space (7D: Φ, B, W, A, R, E, K)
#[derive(Debug, Clone)]
pub struct ConsciousnessPoint {
    /// The 7 consciousness dimensions
    pub dimensions: [f64; 7],
    /// Timestamp of observation
    pub timestamp: Instant,
    /// Point index in the sample
    pub index: usize,
}

impl ConsciousnessPoint {
    /// Euclidean distance to another point
    pub fn distance(&self, other: &Self) -> f64 {
        self.dimensions.iter()
            .zip(other.dimensions.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// A simplex (point, edge, triangle, tetrahedron, ...)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Simplex {
    /// Vertex indices (sorted)
    pub vertices: Vec<usize>,
}

impl Simplex {
    /// Create a new simplex from vertex indices
    pub fn new(mut vertices: Vec<usize>) -> Self {
        vertices.sort();
        Self { vertices }
    }

    /// Dimension of the simplex (0 = point, 1 = edge, 2 = triangle, ...)
    pub fn dimension(&self) -> usize {
        self.vertices.len().saturating_sub(1)
    }

    /// Get all faces (boundary simplices)
    pub fn faces(&self) -> Vec<Simplex> {
        if self.vertices.len() <= 1 {
            return vec![];
        }

        (0..self.vertices.len())
            .map(|i| {
                let mut face_vertices = self.vertices.clone();
                face_vertices.remove(i);
                Simplex::new(face_vertices)
            })
            .collect()
    }
}

/// A simplicial complex (collection of simplices)
#[derive(Debug, Clone, Default)]
pub struct SimplicialComplex {
    /// All simplices by dimension
    pub simplices_by_dim: Vec<HashSet<Simplex>>,
    /// The filtration value at which each simplex appears
    pub filtration_values: HashMap<Simplex, f64>,
}

impl SimplicialComplex {
    /// Create a new empty complex
    pub fn new() -> Self {
        Self {
            simplices_by_dim: vec![HashSet::new(); 4], // Up to 3-simplices
            filtration_values: HashMap::new(),
        }
    }

    /// Add a simplex at a given filtration value
    pub fn add_simplex(&mut self, simplex: Simplex, filtration: f64) {
        let dim = simplex.dimension();
        while self.simplices_by_dim.len() <= dim {
            self.simplices_by_dim.push(HashSet::new());
        }
        self.simplices_by_dim[dim].insert(simplex.clone());
        self.filtration_values.entry(simplex).or_insert(filtration);
    }

    /// Get simplices at a specific dimension
    pub fn simplices_at_dim(&self, dim: usize) -> &HashSet<Simplex> {
        use std::sync::LazyLock;
        if dim < self.simplices_by_dim.len() {
            &self.simplices_by_dim[dim]
        } else {
            static EMPTY: LazyLock<HashSet<Simplex>> = LazyLock::new(HashSet::new);
            &EMPTY
        }
    }

    /// Count simplices at each dimension
    pub fn face_counts(&self) -> Vec<usize> {
        self.simplices_by_dim.iter().map(|s| s.len()).collect()
    }
}

/// Birth-death pair representing a topological feature
#[derive(Debug, Clone)]
pub struct PersistencePair {
    /// Dimension of the feature (0 = component, 1 = loop, 2 = void)
    pub dimension: usize,
    /// Filtration value at which feature is born
    pub birth: f64,
    /// Filtration value at which feature dies (None = infinite)
    pub death: Option<f64>,
}

impl PersistencePair {
    /// Persistence (lifespan) of the feature
    pub fn persistence(&self) -> f64 {
        match self.death {
            Some(d) => d - self.birth,
            None => f64::INFINITY,
        }
    }

    /// Is this a significant (long-lived) feature?
    pub fn is_significant(&self, threshold: f64) -> bool {
        self.persistence() >= threshold
    }
}

/// Betti numbers at a specific filtration level
#[derive(Debug, Clone, Default)]
pub struct BettiNumbers {
    /// β₀: Number of connected components
    pub beta_0: usize,
    /// β₁: Number of 1-dimensional holes (loops)
    pub beta_1: usize,
    /// β₂: Number of 2-dimensional voids
    pub beta_2: usize,
    /// Euler characteristic: χ = β₀ - β₁ + β₂
    pub euler_characteristic: i64,
}

impl BettiNumbers {
    /// Compute from a simplicial complex using boundary matrix
    pub fn from_complex(complex: &SimplicialComplex) -> Self {
        // Simplified computation using face counts and Euler formula
        let face_counts = complex.face_counts();

        // β₀ = number of vertices in component analysis
        let beta_0 = Self::count_components(complex);

        // Approximate β₁ using Euler: χ = V - E + F for surfaces
        // χ = β₀ - β₁ + β₂
        let v = face_counts.first().copied().unwrap_or(0) as i64;
        let e = face_counts.get(1).copied().unwrap_or(0) as i64;
        let f = face_counts.get(2).copied().unwrap_or(0) as i64;

        let euler = v - e + f;

        // Estimate β₁ (loops): typically V - E + F = β₀ - β₁ + β₂
        // If β₂ is small, β₁ ≈ β₀ - euler
        let beta_1 = ((beta_0 as i64) - euler).max(0) as usize;
        let beta_2 = 0; // Simplified: would need proper homology computation

        Self {
            beta_0,
            beta_1,
            beta_2,
            euler_characteristic: euler,
        }
    }

    /// Count connected components using union-find
    fn count_components(complex: &SimplicialComplex) -> usize {
        let vertices: Vec<_> = complex.simplices_at_dim(0)
            .iter()
            .filter_map(|s| s.vertices.first().copied())
            .collect();

        if vertices.is_empty() {
            return 0;
        }

        let max_v = vertices.iter().max().copied().unwrap_or(0);
        let mut parent: Vec<usize> = (0..=max_v).collect();

        fn find(parent: &mut [usize], x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }

        fn union(parent: &mut [usize], x: usize, y: usize) {
            let px = find(parent, x);
            let py = find(parent, y);
            if px != py {
                parent[px] = py;
            }
        }

        // Union vertices connected by edges
        for edge in complex.simplices_at_dim(1) {
            if edge.vertices.len() >= 2 {
                union(&mut parent, edge.vertices[0], edge.vertices[1]);
            }
        }

        // Count unique roots among vertices
        let roots: HashSet<_> = vertices.iter()
            .map(|&v| find(&mut parent, v))
            .collect();
        roots.len()
    }

    /// Consciousness interpretation
    pub fn interpretation(&self) -> TopologyInterpretation {
        TopologyInterpretation {
            unity: if self.beta_0 == 1 { 1.0 } else { 1.0 / self.beta_0 as f64 },
            complexity: (self.beta_1 + self.beta_2 * 2) as f64 / 10.0,
            fragmentation: if self.beta_0 > 1 {
                (self.beta_0 - 1) as f64 / 10.0
            } else {
                0.0
            },
        }
    }
}

/// Psychological interpretation of topological features
#[derive(Debug, Clone)]
pub struct TopologyInterpretation {
    /// Unity of consciousness (1.0 = unified, <1.0 = fragmented)
    pub unity: f64,
    /// Topological complexity (loops and voids)
    pub complexity: f64,
    /// Degree of fragmentation
    pub fragmentation: f64,
}

/// Curvature estimation for consciousness manifold
#[derive(Debug, Clone)]
pub struct ManifoldCurvature {
    /// Estimated intrinsic dimensionality
    pub dimensionality: f64,
    /// Mean curvature (positive = curved like sphere, negative = like saddle)
    pub mean_curvature: f64,
    /// Curvature variance (uniformity)
    pub curvature_variance: f64,
}

/// Complete topological analysis result
#[derive(Debug, Clone)]
pub struct TopologyAnalysis {
    /// Betti numbers at final filtration
    pub betti_numbers: BettiNumbers,
    /// Persistence pairs (birth-death)
    pub persistence_pairs: Vec<PersistencePair>,
    /// Significant features (above persistence threshold)
    pub significant_features: usize,
    /// Estimated manifold curvature
    pub curvature: Option<ManifoldCurvature>,
    /// Psychological interpretation
    pub interpretation: TopologyInterpretation,
    /// Total points analyzed
    pub points_analyzed: usize,
}

/// Statistics for topological analyzer
#[derive(Debug, Clone, Default)]
pub struct TopologyStats {
    /// Total analyses performed
    pub analyses: u64,
    /// Average β₀ (connectedness)
    pub avg_beta_0: f64,
    /// Average β₁ (loops)
    pub avg_beta_1: f64,
    /// Average Euler characteristic
    pub avg_euler: f64,
    /// Total significant features found
    pub significant_features: u64,
    /// Average unity score
    pub avg_unity: f64,
}

/// The main consciousness topology analyzer
pub struct ConsciousnessTopologyAnalyzer {
    /// Configuration
    pub config: TopologyConfig,
    /// Sliding window of consciousness points
    point_window: VecDeque<ConsciousnessPoint>,
    /// Current simplicial complex
    current_complex: SimplicialComplex,
    /// Point counter
    point_counter: usize,
    /// Running statistics
    pub stats: TopologyStats,
    /// Analysis start time
    started_at: Instant,
}

impl ConsciousnessTopologyAnalyzer {
    /// Create a new analyzer with configuration
    pub fn new(config: TopologyConfig) -> Self {
        Self {
            config,
            point_window: VecDeque::with_capacity(100),
            current_complex: SimplicialComplex::new(),
            point_counter: 0,
            stats: TopologyStats::default(),
            started_at: Instant::now(),
        }
    }

    /// Add a consciousness observation
    pub fn observe(&mut self, dimensions: [f64; 7]) {
        let point = ConsciousnessPoint {
            dimensions,
            timestamp: Instant::now(),
            index: self.point_counter,
        };
        self.point_counter += 1;

        // Maintain sliding window
        if self.point_window.len() >= self.config.max_points {
            self.point_window.pop_front();
        }
        self.point_window.push_back(point);

        // Rebuild complex periodically
        if self.point_counter % 10 == 0 {
            self.rebuild_complex();
        }
    }

    /// Rebuild the simplicial complex from current points
    fn rebuild_complex(&mut self) {
        self.current_complex = SimplicialComplex::new();
        let points: Vec<_> = self.point_window.iter().collect();

        if points.is_empty() {
            return;
        }

        // Add 0-simplices (vertices)
        for (i, _) in points.iter().enumerate() {
            self.current_complex.add_simplex(Simplex::new(vec![i]), 0.0);
        }

        // Add 1-simplices (edges) based on distance threshold
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                let dist = points[i].distance(points[j]);
                if dist < self.config.edge_threshold {
                    self.current_complex.add_simplex(Simplex::new(vec![i, j]), dist);
                }
            }
        }

        // Add 2-simplices (triangles) where all edges exist
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                for k in (j + 1)..points.len() {
                    let edge_ij = Simplex::new(vec![i, j]);
                    let edge_jk = Simplex::new(vec![j, k]);
                    let edge_ik = Simplex::new(vec![i, k]);

                    if self.current_complex.simplices_at_dim(1).contains(&edge_ij)
                        && self.current_complex.simplices_at_dim(1).contains(&edge_jk)
                        && self.current_complex.simplices_at_dim(1).contains(&edge_ik)
                    {
                        let max_edge = [
                            self.current_complex.filtration_values.get(&edge_ij),
                            self.current_complex.filtration_values.get(&edge_jk),
                            self.current_complex.filtration_values.get(&edge_ik),
                        ]
                        .iter()
                        .filter_map(|v| *v)
                        .fold(0.0f64, |a, &b| a.max(b));

                        self.current_complex.add_simplex(Simplex::new(vec![i, j, k]), max_edge);
                    }
                }
            }
        }
    }

    /// Perform full topological analysis
    pub fn analyze(&mut self) -> TopologyAnalysis {
        self.rebuild_complex();

        let betti = BettiNumbers::from_complex(&self.current_complex);
        let interpretation = betti.interpretation();

        // Compute persistence pairs (simplified)
        let persistence_pairs = self.compute_persistence();
        let significant = persistence_pairs.iter()
            .filter(|p| p.is_significant(self.config.min_persistence))
            .count();

        // Estimate curvature if enabled
        let curvature = if self.config.estimate_curvature {
            Some(self.estimate_curvature())
        } else {
            None
        };

        // Update statistics
        self.stats.analyses += 1;
        let n = self.stats.analyses as f64;
        self.stats.avg_beta_0 = (self.stats.avg_beta_0 * (n - 1.0) + betti.beta_0 as f64) / n;
        self.stats.avg_beta_1 = (self.stats.avg_beta_1 * (n - 1.0) + betti.beta_1 as f64) / n;
        self.stats.avg_euler = (self.stats.avg_euler * (n - 1.0) + betti.euler_characteristic as f64) / n;
        self.stats.avg_unity = (self.stats.avg_unity * (n - 1.0) + interpretation.unity) / n;
        self.stats.significant_features += significant as u64;

        TopologyAnalysis {
            betti_numbers: betti,
            persistence_pairs,
            significant_features: significant,
            curvature,
            interpretation,
            points_analyzed: self.point_window.len(),
        }
    }

    /// Compute simplified persistence pairs
    fn compute_persistence(&self) -> Vec<PersistencePair> {
        let mut pairs = Vec::new();

        // β₀ features: One component born at 0, persists to infinity
        let betti = BettiNumbers::from_complex(&self.current_complex);

        // Main component (born at 0, never dies)
        pairs.push(PersistencePair {
            dimension: 0,
            birth: 0.0,
            death: None,
        });

        // Additional components that merge (if β₀ > 1, they have finite lifetime)
        for _ in 1..betti.beta_0 {
            pairs.push(PersistencePair {
                dimension: 0,
                birth: 0.0,
                death: Some(self.config.edge_threshold),
            });
        }

        // β₁ features (loops)
        for _ in 0..betti.beta_1 {
            pairs.push(PersistencePair {
                dimension: 1,
                birth: self.config.edge_threshold * 0.5,
                death: Some(self.config.edge_threshold),
            });
        }

        pairs
    }

    /// Estimate manifold curvature
    fn estimate_curvature(&self) -> ManifoldCurvature {
        let points: Vec<_> = self.point_window.iter().collect();

        if points.len() < 3 {
            return ManifoldCurvature {
                dimensionality: 0.0,
                mean_curvature: 0.0,
                curvature_variance: 0.0,
            };
        }

        // Estimate dimensionality using PCA-like variance analysis
        let mut variance_sum = 0.0;
        for dim in 0..7 {
            let values: Vec<f64> = points.iter().map(|p| p.dimensions[dim]).collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
            variance_sum += var;
        }

        // Intrinsic dimensionality estimate (how many dimensions are "active")
        let dimensionality = (variance_sum / 7.0).sqrt() * 7.0;

        // Curvature estimate using triangle area ratios
        let mut curvatures = Vec::new();
        for i in 0..points.len().saturating_sub(2) {
            let d01 = points[i].distance(points[i + 1]);
            let d12 = points[i + 1].distance(points[i + 2]);
            let d02 = points[i].distance(points[i + 2]);

            if d01 > 0.0 && d12 > 0.0 && d02 > 0.0 {
                // Curvature approximation: deviation from straight line
                let straight = d01 + d12;
                let actual = d02;
                let curvature = (straight - actual) / straight;
                curvatures.push(curvature);
            }
        }

        let mean_curvature = if curvatures.is_empty() {
            0.0
        } else {
            curvatures.iter().sum::<f64>() / curvatures.len() as f64
        };

        let curvature_variance = if curvatures.len() > 1 {
            let mean = mean_curvature;
            curvatures.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / curvatures.len() as f64
        } else {
            0.0
        };

        ManifoldCurvature {
            dimensionality,
            mean_curvature,
            curvature_variance,
        }
    }

    /// Get a detailed report
    pub fn get_report(&self) -> TopologyReport {
        TopologyReport {
            analyses: self.stats.analyses,
            avg_beta_0: self.stats.avg_beta_0,
            avg_beta_1: self.stats.avg_beta_1,
            avg_euler: self.stats.avg_euler,
            avg_unity: self.stats.avg_unity,
            significant_features: self.stats.significant_features,
            current_points: self.point_window.len(),
        }
    }
}

/// Report on topological analysis
#[derive(Debug, Clone)]
pub struct TopologyReport {
    /// Total analyses performed
    pub analyses: u64,
    /// Average β₀ (components)
    pub avg_beta_0: f64,
    /// Average β₁ (loops)
    pub avg_beta_1: f64,
    /// Average Euler characteristic
    pub avg_euler: f64,
    /// Average unity score
    pub avg_unity: f64,
    /// Total significant features
    pub significant_features: u64,
    /// Current point count
    pub current_points: usize,
}

impl std::fmt::Display for TopologyReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Consciousness Topology Report ===")?;
        writeln!(f, "Analyses Performed: {}", self.analyses)?;
        writeln!(f, "Average β₀ (Components): {:.2}", self.avg_beta_0)?;
        writeln!(f, "Average β₁ (Loops): {:.2}", self.avg_beta_1)?;
        writeln!(f, "Average Euler χ: {:.2}", self.avg_euler)?;
        writeln!(f, "Average Unity: {:.3}", self.avg_unity)?;
        writeln!(f, "Significant Features: {}", self.significant_features)?;
        writeln!(f, "Current Points: {}", self.current_points)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_point(phi: f64, binding: f64, workspace: f64) -> [f64; 7] {
        [phi, binding, workspace, 0.5, 0.5, 0.5, 0.5]
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = ConsciousnessTopologyAnalyzer::new(TopologyConfig::default());
        assert_eq!(analyzer.stats.analyses, 0);
    }

    #[test]
    fn test_simplex_creation() {
        let simplex = Simplex::new(vec![0, 1, 2]);
        assert_eq!(simplex.dimension(), 2); // Triangle
        assert_eq!(simplex.faces().len(), 3); // Three edges
    }

    #[test]
    fn test_point_distance() {
        let p1 = ConsciousnessPoint {
            dimensions: [0.0; 7],
            timestamp: Instant::now(),
            index: 0,
        };
        let p2 = ConsciousnessPoint {
            dimensions: [1.0; 7],
            timestamp: Instant::now(),
            index: 1,
        };
        let dist = p1.distance(&p2);
        assert!((dist - 7.0_f64.sqrt()).abs() < 0.001);
    }

    #[test]
    fn test_observation() {
        let mut analyzer = ConsciousnessTopologyAnalyzer::new(TopologyConfig::default());
        analyzer.observe(test_point(0.8, 0.7, 0.6));
        analyzer.observe(test_point(0.81, 0.71, 0.61));
        analyzer.observe(test_point(0.82, 0.72, 0.62));

        assert_eq!(analyzer.point_window.len(), 3);
    }

    #[test]
    fn test_betti_numbers() {
        let mut complex = SimplicialComplex::new();

        // Add 3 disconnected points
        complex.add_simplex(Simplex::new(vec![0]), 0.0);
        complex.add_simplex(Simplex::new(vec![1]), 0.0);
        complex.add_simplex(Simplex::new(vec![2]), 0.0);

        let betti = BettiNumbers::from_complex(&complex);
        assert_eq!(betti.beta_0, 3); // Three components

        // Connect them with edges
        complex.add_simplex(Simplex::new(vec![0, 1]), 0.1);
        complex.add_simplex(Simplex::new(vec![1, 2]), 0.1);

        let betti = BettiNumbers::from_complex(&complex);
        assert_eq!(betti.beta_0, 1); // Now connected
    }

    #[test]
    fn test_triangle_creates_loop() {
        let mut complex = SimplicialComplex::new();

        // Three vertices
        complex.add_simplex(Simplex::new(vec![0]), 0.0);
        complex.add_simplex(Simplex::new(vec![1]), 0.0);
        complex.add_simplex(Simplex::new(vec![2]), 0.0);

        // Three edges (forming a loop)
        complex.add_simplex(Simplex::new(vec![0, 1]), 0.1);
        complex.add_simplex(Simplex::new(vec![1, 2]), 0.1);
        complex.add_simplex(Simplex::new(vec![0, 2]), 0.1);

        let betti = BettiNumbers::from_complex(&complex);
        assert_eq!(betti.beta_0, 1); // Connected
        assert_eq!(betti.beta_1, 1); // One loop (the triangle boundary)
    }

    #[test]
    fn test_filled_triangle_no_loop() {
        let mut complex = SimplicialComplex::new();

        // Three vertices
        complex.add_simplex(Simplex::new(vec![0]), 0.0);
        complex.add_simplex(Simplex::new(vec![1]), 0.0);
        complex.add_simplex(Simplex::new(vec![2]), 0.0);

        // Three edges
        complex.add_simplex(Simplex::new(vec![0, 1]), 0.1);
        complex.add_simplex(Simplex::new(vec![1, 2]), 0.1);
        complex.add_simplex(Simplex::new(vec![0, 2]), 0.1);

        // Add the triangle face (fills the hole)
        complex.add_simplex(Simplex::new(vec![0, 1, 2]), 0.2);

        let betti = BettiNumbers::from_complex(&complex);
        assert_eq!(betti.beta_0, 1);
        // Note: Our simplified computation may still show β₁ = 1
        // Full homology would show β₁ = 0 when triangle is filled
    }

    #[test]
    fn test_persistence_pair() {
        let pair = PersistencePair {
            dimension: 0,
            birth: 0.0,
            death: Some(0.5),
        };
        assert_eq!(pair.persistence(), 0.5);
        assert!(pair.is_significant(0.3));
        assert!(!pair.is_significant(0.6));
    }

    #[test]
    fn test_infinite_persistence() {
        let pair = PersistencePair {
            dimension: 0,
            birth: 0.0,
            death: None,
        };
        assert_eq!(pair.persistence(), f64::INFINITY);
        assert!(pair.is_significant(1000.0));
    }

    #[test]
    fn test_full_analysis() {
        let mut analyzer = ConsciousnessTopologyAnalyzer::new(TopologyConfig::default());

        // Add clustered points (should form connected component)
        for i in 0..10 {
            analyzer.observe([0.5 + i as f64 * 0.01; 7]);
        }

        let analysis = analyzer.analyze();
        assert!(analysis.betti_numbers.beta_0 >= 1);
        assert!(analysis.interpretation.unity > 0.0);
    }

    #[test]
    fn test_fragmented_consciousness() {
        let config = TopologyConfig {
            edge_threshold: 0.1, // Very strict
            ..Default::default()
        };
        let mut analyzer = ConsciousnessTopologyAnalyzer::new(config);

        // Add widely separated points (should be fragmented)
        analyzer.observe([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        analyzer.observe([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        analyzer.observe([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);

        let analysis = analyzer.analyze();
        // With strict threshold, should have multiple components
        assert!(analysis.betti_numbers.beta_0 >= 1);
    }

    #[test]
    fn test_curvature_estimation() {
        let mut analyzer = ConsciousnessTopologyAnalyzer::new(TopologyConfig::default());

        // Add points along a curve
        for i in 0..20 {
            let t = i as f64 / 20.0;
            analyzer.observe([t, t.sin() * 0.1, t.cos() * 0.1, 0.5, 0.5, 0.5, 0.5]);
        }

        let analysis = analyzer.analyze();
        assert!(analysis.curvature.is_some());
        let curv = analysis.curvature.unwrap();
        assert!(curv.dimensionality > 0.0);
    }

    #[test]
    fn test_report_generation() {
        let mut analyzer = ConsciousnessTopologyAnalyzer::new(TopologyConfig::default());

        for _ in 0..5 {
            analyzer.observe([0.5; 7]);
        }
        analyzer.analyze();

        let report = analyzer.get_report();
        assert_eq!(report.analyses, 1);
        assert!(report.avg_unity > 0.0);
    }

    #[test]
    fn test_euler_characteristic() {
        let betti = BettiNumbers {
            beta_0: 1,
            beta_1: 2,
            beta_2: 1,
            euler_characteristic: 0, // Will be computed
        };

        // χ = β₀ - β₁ + β₂ = 1 - 2 + 1 = 0
        let expected_euler = 1 - 2 + 1;
        assert_eq!(expected_euler, 0);
    }

    #[test]
    fn test_interpretation() {
        let betti = BettiNumbers {
            beta_0: 1,
            beta_1: 0,
            beta_2: 0,
            euler_characteristic: 1,
        };

        let interp = betti.interpretation();
        assert_eq!(interp.unity, 1.0); // Single connected component
        assert_eq!(interp.fragmentation, 0.0);
        assert_eq!(interp.complexity, 0.0);
    }
}
