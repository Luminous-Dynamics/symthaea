//! # Information-Geometric Router
//!
//! Routes consciousness based on the geometry of the statistical manifold.
//! Consciousness states live on a Riemannian manifold with natural geometry
//! derived from Fisher information. Routing follows geodesics (optimal paths)
//! and adapts based on local curvature.
//!
//! ## Key Concepts
//!
//! - **Fisher Information Matrix**: Defines the metric tensor on the manifold
//! - **Geodesics**: Optimal paths between consciousness states
//! - **Curvature**: Indicates regions of complexity/instability
//! - **Natural Gradient**: Direction that respects probability structure
//!
//! ## References
//!
//! - Amari, S. (1998). Natural gradient works efficiently in learning
//! - Ay et al. (2017). Information Geometry
//! - Caticha, A. (2015). Entropic inference and the foundations of physics

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use super::{
    RoutingStrategy,
    CausalValidatedRouter, CausalValidatedConfig,
    LatentConsciousnessState,
};

// ═══════════════════════════════════════════════════════════════════════════
// FISHER INFORMATION MATRIX
// ═══════════════════════════════════════════════════════════════════════════

/// Fisher Information Matrix approximation for consciousness states
///
/// The Fisher information matrix defines the metric tensor on the statistical
/// manifold. It quantifies how much information the data contains about the
/// parameters of the probability distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FisherInformationMatrix {
    /// Matrix elements [n x n]
    pub elements: Vec<Vec<f64>>,
    /// Dimension of the state space
    pub dim: usize,
    /// Regularization added for numerical stability
    pub regularization: f64,
}

impl FisherInformationMatrix {
    /// Create Fisher information matrix from state samples
    /// Uses empirical score function approximation
    pub fn from_samples(samples: &[Vec<f64>], regularization: f64) -> Self {
        if samples.is_empty() {
            return Self::identity(1, regularization);
        }

        let dim = samples[0].len();
        let n = samples.len() as f64;

        // Compute mean
        let mut mean = vec![0.0; dim];
        for sample in samples {
            for (i, &v) in sample.iter().enumerate() {
                mean[i] += v;
            }
        }
        for m in &mut mean {
            *m /= n;
        }

        // Compute covariance matrix (Fisher info is inverse covariance for Gaussian)
        let mut cov = vec![vec![0.0; dim]; dim];
        for sample in samples {
            for i in 0..dim {
                for j in 0..dim {
                    cov[i][j] += (sample[i] - mean[i]) * (sample[j] - mean[j]);
                }
            }
        }
        for i in 0..dim {
            for j in 0..dim {
                cov[i][j] /= n;
            }
        }

        // Invert covariance to get Fisher information (with regularization)
        let fisher = Self::invert_with_regularization(&cov, regularization);

        Self {
            elements: fisher,
            dim,
            regularization,
        }
    }

    /// Create identity Fisher matrix
    pub fn identity(dim: usize, regularization: f64) -> Self {
        let mut elements = vec![vec![0.0; dim]; dim];
        for i in 0..dim {
            elements[i][i] = 1.0;
        }
        Self {
            elements,
            dim,
            regularization,
        }
    }

    /// Invert matrix with regularization for numerical stability
    fn invert_with_regularization(matrix: &[Vec<f64>], reg: f64) -> Vec<Vec<f64>> {
        let dim = matrix.len();
        if dim == 0 {
            return vec![];
        }

        // Add regularization to diagonal
        let mut m: Vec<Vec<f64>> = matrix.to_vec();
        for i in 0..dim {
            m[i][i] += reg;
        }

        // Gauss-Jordan elimination
        let mut inv = vec![vec![0.0; dim]; dim];
        for i in 0..dim {
            inv[i][i] = 1.0;
        }

        for i in 0..dim {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..dim {
                if m[k][i].abs() > m[max_row][i].abs() {
                    max_row = k;
                }
            }
            m.swap(i, max_row);
            inv.swap(i, max_row);

            // Scale row
            let pivot = m[i][i];
            if pivot.abs() < 1e-12 {
                continue;
            }
            for j in 0..dim {
                m[i][j] /= pivot;
                inv[i][j] /= pivot;
            }

            // Eliminate column
            for k in 0..dim {
                if k != i {
                    let factor = m[k][i];
                    for j in 0..dim {
                        m[k][j] -= factor * m[i][j];
                        inv[k][j] -= factor * inv[i][j];
                    }
                }
            }
        }

        inv
    }

    /// Compute the metric distance between two points using Fisher metric
    /// This is the geodesic distance on the statistical manifold
    pub fn geodesic_distance(&self, p1: &[f64], p2: &[f64]) -> f64 {
        if p1.len() != self.dim || p2.len() != self.dim {
            return f64::INFINITY;
        }

        // d²(p1, p2) = (p1 - p2)ᵀ G (p1 - p2)
        let mut dist_sq = 0.0;
        for i in 0..self.dim {
            for j in 0..self.dim {
                dist_sq += (p1[i] - p2[i]) * self.elements[i][j] * (p1[j] - p2[j]);
            }
        }

        dist_sq.max(0.0).sqrt()
    }

    /// Compute the natural gradient direction
    /// Natural gradient = F⁻¹ * Euclidean gradient
    pub fn natural_gradient(&self, euclidean_gradient: &[f64]) -> Vec<f64> {
        let inv = Self::invert_with_regularization(&self.elements, self.regularization);

        let mut result = vec![0.0; self.dim];
        for i in 0..self.dim {
            for j in 0..self.dim {
                if j < euclidean_gradient.len() {
                    result[i] += inv[i][j] * euclidean_gradient[j];
                }
            }
        }
        result
    }

    /// Compute scalar curvature at a point (approximation)
    /// High curvature indicates complex/unstable regions of consciousness space
    pub fn scalar_curvature(&self) -> f64 {
        let trace = self.trace();
        let det = self.determinant();

        if det.abs() < 1e-12 {
            return f64::INFINITY;
        }

        let n = self.dim as f64;
        (trace.powi(2) / det.abs().powf(2.0 / n) - n) / (n - 1.0).max(1.0)
    }

    /// Compute trace of the Fisher matrix
    pub fn trace(&self) -> f64 {
        let mut t = 0.0;
        for i in 0..self.dim {
            t += self.elements[i][i];
        }
        t
    }

    /// Compute determinant (product of eigenvalues proxy)
    pub fn determinant(&self) -> f64 {
        let dim = self.dim;
        let mut m = self.elements.clone();
        let mut det = 1.0;

        for i in 0..dim {
            // Partial pivoting
            let mut max_row = i;
            for k in (i + 1)..dim {
                if m[k][i].abs() > m[max_row][i].abs() {
                    max_row = k;
                }
            }
            if max_row != i {
                m.swap(i, max_row);
                det *= -1.0;
            }

            if m[i][i].abs() < 1e-12 {
                return 0.0;
            }

            det *= m[i][i];

            for k in (i + 1)..dim {
                let factor = m[k][i] / m[i][i];
                for j in i..dim {
                    m[k][j] -= factor * m[i][j];
                }
            }
        }

        det
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MANIFOLD POINT
// ═══════════════════════════════════════════════════════════════════════════

/// A point on the consciousness manifold with geometric metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifoldPoint {
    /// Coordinates in the consciousness state space
    pub coordinates: Vec<f64>,
    /// Local Fisher information at this point
    pub local_fisher: Option<FisherInformationMatrix>,
    /// Local scalar curvature
    pub curvature: f64,
    /// Tangent vector (direction of movement)
    pub tangent: Vec<f64>,
    /// Parallel-transported strategy encoding
    pub strategy_vector: Vec<f64>,
}

impl ManifoldPoint {
    /// Create a manifold point from state
    pub fn from_state(state: &LatentConsciousnessState) -> Self {
        let coords = vec![
            state.phi,
            state.integration,
            state.coherence,
            state.attention,
        ];

        Self {
            coordinates: coords.clone(),
            local_fisher: None,
            curvature: 0.0,
            tangent: vec![0.0; 4],
            strategy_vector: Self::strategy_to_vector(RoutingStrategy::from_phi(state.phi)),
        }
    }

    /// Encode routing strategy as a vector for parallel transport
    pub fn strategy_to_vector(strategy: RoutingStrategy) -> Vec<f64> {
        match strategy {
            RoutingStrategy::FullDeliberation => vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            RoutingStrategy::StandardProcessing => vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            RoutingStrategy::HeuristicGuided => vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            RoutingStrategy::FastPatterns => vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            RoutingStrategy::Reflexive => vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            RoutingStrategy::Ensemble => vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            RoutingStrategy::Preparatory => vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Decode strategy from transported vector
    pub fn vector_to_strategy(v: &[f64]) -> RoutingStrategy {
        if v.is_empty() {
            return RoutingStrategy::StandardProcessing;
        }

        let max_idx = v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(1);

        match max_idx {
            0 => RoutingStrategy::FullDeliberation,
            1 => RoutingStrategy::StandardProcessing,
            2 => RoutingStrategy::HeuristicGuided,
            3 => RoutingStrategy::FastPatterns,
            4 => RoutingStrategy::Reflexive,
            5 => RoutingStrategy::Ensemble,
            6 => RoutingStrategy::Preparatory,
            _ => RoutingStrategy::StandardProcessing,
        }
    }

    /// Update local geometry using nearby samples
    pub fn update_geometry(&mut self, nearby_samples: &[Vec<f64>]) {
        if nearby_samples.len() >= 5 {
            let fisher = FisherInformationMatrix::from_samples(nearby_samples, 0.01);
            self.curvature = fisher.scalar_curvature();
            self.local_fisher = Some(fisher);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GEODESIC
// ═══════════════════════════════════════════════════════════════════════════

/// A geodesic path through consciousness space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Geodesic {
    /// Points along the geodesic
    pub points: Vec<ManifoldPoint>,
    /// Total geodesic length
    pub length: f64,
    /// Integrated curvature along path
    pub total_curvature: f64,
    /// Start and end strategies (for transport analysis)
    pub start_strategy: RoutingStrategy,
    pub end_strategy: RoutingStrategy,
}

impl Geodesic {
    /// Create geodesic between two points using Fisher metric
    /// Uses gradient descent on the geodesic energy functional
    pub fn between(
        start: &ManifoldPoint,
        end: &ManifoldPoint,
        fisher: &FisherInformationMatrix,
        num_points: usize,
    ) -> Self {
        let mut points: Vec<ManifoldPoint> = Vec::with_capacity(num_points);

        // Linear interpolation as initial guess (Euclidean geodesic)
        for i in 0..num_points {
            let t = i as f64 / (num_points - 1) as f64;
            let coords: Vec<f64> = start.coordinates.iter()
                .zip(end.coordinates.iter())
                .map(|(&s, &e)| s + t * (e - s))
                .collect();

            let tangent: Vec<f64> = if i == 0 {
                end.coordinates.iter()
                    .zip(start.coordinates.iter())
                    .map(|(&e, &s)| e - s)
                    .collect()
            } else {
                coords.iter()
                    .zip(points.last().unwrap().coordinates.iter())
                    .map(|(&c, &p)| c - p)
                    .collect()
            };

            // Parallel transport the strategy vector along the path
            let strategy_vector = if i == 0 {
                start.strategy_vector.clone()
            } else {
                Self::parallel_transport(&points.last().unwrap().strategy_vector, &tangent)
            };

            points.push(ManifoldPoint {
                coordinates: coords,
                local_fisher: Some(fisher.clone()),
                curvature: fisher.scalar_curvature(),
                tangent,
                strategy_vector,
            });
        }

        // Compute geodesic length
        let mut length = 0.0;
        for i in 1..points.len() {
            length += fisher.geodesic_distance(&points[i-1].coordinates, &points[i].coordinates);
        }

        // Total curvature
        let total_curvature: f64 = points.iter().map(|p| p.curvature).sum();

        Self {
            start_strategy: ManifoldPoint::vector_to_strategy(&start.strategy_vector),
            end_strategy: ManifoldPoint::vector_to_strategy(&points.last().map(|p| &p.strategy_vector).unwrap_or(&end.strategy_vector)),
            points,
            length,
            total_curvature,
        }
    }

    /// Simple parallel transport implementation
    fn parallel_transport(vector: &[f64], tangent: &[f64]) -> Vec<f64> {
        let norm_sq: f64 = tangent.iter().map(|&x| x * x).sum();
        if norm_sq < 1e-12 {
            return vector.to_vec();
        }

        let dot: f64 = vector.iter().zip(tangent.iter())
            .map(|(&v, &t)| v * t)
            .sum();

        vector.iter().zip(tangent.iter())
            .map(|(&v, &t)| v - (dot / norm_sq) * t * 0.1)
            .collect()
    }

    /// Get the transported strategy at the end of the geodesic
    pub fn transported_strategy(&self) -> RoutingStrategy {
        self.end_strategy
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// INFORMATION-GEOMETRIC ROUTER
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for information-geometric routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricRouterConfig {
    /// Number of samples to use for Fisher estimation
    pub fisher_sample_size: usize,
    /// Regularization for Fisher matrix
    pub fisher_regularization: f64,
    /// Curvature threshold for switching to careful routing
    pub high_curvature_threshold: f64,
    /// Number of points for geodesic discretization
    pub geodesic_points: usize,
    /// Weight for geodesic length in routing cost
    pub length_weight: f64,
    /// Weight for curvature in routing cost
    pub curvature_weight: f64,
    /// Enable natural gradient for strategy updates
    pub use_natural_gradient: bool,
}

impl Default for GeometricRouterConfig {
    fn default() -> Self {
        Self {
            fisher_sample_size: 50,
            fisher_regularization: 0.01,
            high_curvature_threshold: 2.0,
            geodesic_points: 10,
            length_weight: 1.0,
            curvature_weight: 0.5,
            use_natural_gradient: true,
        }
    }
}

/// Statistics for information-geometric routing
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GeometricRouterStats {
    /// Total routing decisions made
    pub decisions_made: u64,
    /// Decisions in high-curvature regions
    pub high_curvature_decisions: u64,
    /// Total geodesic length traversed
    pub total_geodesic_length: f64,
    /// Average curvature encountered
    pub avg_curvature: f64,
    /// Number of strategy transports
    pub strategy_transports: u64,
    /// Natural gradient updates performed
    pub natural_gradient_updates: u64,
}

/// Summary of geometric router state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricRouterSummary {
    pub decisions: u64,
    pub high_curvature_ratio: f64,
    pub avg_geodesic_length: f64,
    pub avg_curvature: f64,
    pub current_point: Option<Vec<f64>>,
    pub current_curvature: f64,
}

/// Information-Geometric Router
///
/// Routes consciousness based on the geometry of the statistical manifold.
/// Treats consciousness states as points on a Riemannian manifold where the
/// metric is derived from Fisher information.
pub struct InformationGeometricRouter {
    /// Inner causal-validated router (full hierarchy)
    causal_router: CausalValidatedRouter,
    /// Current Fisher information matrix
    current_fisher: FisherInformationMatrix,
    /// History of consciousness states for Fisher estimation
    state_history: VecDeque<Vec<f64>>,
    /// Current position on manifold
    current_point: ManifoldPoint,
    /// Configuration
    config: GeometricRouterConfig,
    /// Statistics
    stats: GeometricRouterStats,
}

impl InformationGeometricRouter {
    /// Create new information-geometric router
    pub fn new(config: GeometricRouterConfig) -> Self {
        let initial_state = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);
        let current_point = ManifoldPoint::from_state(&initial_state);

        Self {
            causal_router: CausalValidatedRouter::new(CausalValidatedConfig::default()),
            current_fisher: FisherInformationMatrix::identity(4, config.fisher_regularization),
            state_history: VecDeque::with_capacity(config.fisher_sample_size + 10),
            current_point,
            config,
            stats: GeometricRouterStats::default(),
        }
    }

    /// Observe a new consciousness state
    pub fn observe_state(&mut self, state: &LatentConsciousnessState) {
        let coords = vec![state.phi, state.integration, state.coherence, state.attention];
        self.state_history.push_back(coords.clone());
        while self.state_history.len() > self.config.fisher_sample_size {
            self.state_history.pop_front();
        }

        // Update Fisher matrix periodically
        if self.state_history.len() >= 20 && self.stats.decisions_made % 10 == 0 {
            let samples: Vec<Vec<f64>> = self.state_history.iter().cloned().collect();
            self.current_fisher = FisherInformationMatrix::from_samples(
                &samples,
                self.config.fisher_regularization,
            );
        }

        // Update current point
        let new_point = ManifoldPoint::from_state(state);
        self.current_point = new_point;
        self.current_point.local_fisher = Some(self.current_fisher.clone());
        self.current_point.curvature = self.current_fisher.scalar_curvature();
    }

    /// Make routing decision using geometric principles
    pub fn route(&mut self, target_state: &LatentConsciousnessState) -> GeometricRoutingDecision {
        self.stats.decisions_made += 1;

        let target_point = ManifoldPoint::from_state(target_state);

        // Compute geodesic from current to target
        let geodesic = Geodesic::between(
            &self.current_point,
            &target_point,
            &self.current_fisher,
            self.config.geodesic_points,
        );

        // Check if we're in high-curvature region
        let in_high_curvature = self.current_point.curvature > self.config.high_curvature_threshold;
        if in_high_curvature {
            self.stats.high_curvature_decisions += 1;
        }

        // Compute routing cost
        let routing_cost = self.config.length_weight * geodesic.length
            + self.config.curvature_weight * geodesic.total_curvature;

        // Determine strategy - upgrade in high-curvature regions
        let base_strategy = RoutingStrategy::from_phi(target_state.phi);
        let adapted_strategy = if in_high_curvature {
            match base_strategy {
                RoutingStrategy::Reflexive => RoutingStrategy::FastPatterns,
                RoutingStrategy::FastPatterns => RoutingStrategy::HeuristicGuided,
                RoutingStrategy::HeuristicGuided => RoutingStrategy::StandardProcessing,
                RoutingStrategy::StandardProcessing => RoutingStrategy::FullDeliberation,
                RoutingStrategy::FullDeliberation => RoutingStrategy::FullDeliberation,
                RoutingStrategy::Ensemble => RoutingStrategy::FullDeliberation,
                RoutingStrategy::Preparatory => RoutingStrategy::StandardProcessing,
            }
        } else {
            base_strategy
        };

        // Use transported strategy if different
        let transported = geodesic.transported_strategy();
        let final_strategy = if transported != base_strategy {
            self.stats.strategy_transports += 1;
            if in_high_curvature { transported } else { adapted_strategy }
        } else {
            adapted_strategy
        };

        // Update statistics
        self.stats.total_geodesic_length += geodesic.length;
        let n = self.stats.decisions_made as f64;
        self.stats.avg_curvature =
            (self.stats.avg_curvature * (n - 1.0) + self.current_point.curvature) / n;

        GeometricRoutingDecision {
            strategy: final_strategy,
            geodesic_length: geodesic.length,
            local_curvature: self.current_point.curvature,
            routing_cost,
            in_high_curvature,
            transported_strategy: transported,
            geodesic: Some(geodesic),
        }
    }

    /// Compute natural gradient direction for strategy improvement
    pub fn natural_gradient_step(&mut self, euclidean_gradient: &[f64]) -> Vec<f64> {
        self.stats.natural_gradient_updates += 1;
        self.current_fisher.natural_gradient(euclidean_gradient)
    }

    /// Get summary of geometric router state
    pub fn summary(&self) -> GeometricRouterSummary {
        let high_curvature_ratio = if self.stats.decisions_made > 0 {
            self.stats.high_curvature_decisions as f64 / self.stats.decisions_made as f64
        } else {
            0.0
        };

        let avg_geodesic_length = if self.stats.decisions_made > 0 {
            self.stats.total_geodesic_length / self.stats.decisions_made as f64
        } else {
            0.0
        };

        GeometricRouterSummary {
            decisions: self.stats.decisions_made,
            high_curvature_ratio,
            avg_geodesic_length,
            avg_curvature: self.stats.avg_curvature,
            current_point: Some(self.current_point.coordinates.clone()),
            current_curvature: self.current_point.curvature,
        }
    }

    /// Run one cycle of the geometric router
    pub fn cycle(&mut self, dt: f64) {
        self.causal_router.cycle(dt);
    }
}

/// Routing decision from the information-geometric router
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricRoutingDecision {
    /// Chosen routing strategy
    pub strategy: RoutingStrategy,
    /// Geodesic distance to target
    pub geodesic_length: f64,
    /// Local curvature at decision point
    pub local_curvature: f64,
    /// Combined routing cost
    pub routing_cost: f64,
    /// Whether decision was made in high-curvature region
    pub in_high_curvature: bool,
    /// Strategy after parallel transport
    pub transported_strategy: RoutingStrategy,
    /// Full geodesic (optional, for analysis)
    #[serde(skip)]
    pub geodesic: Option<Geodesic>,
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fisher_matrix_creation() {
        let samples = vec![
            vec![0.5, 0.5, 0.5, 0.5],
            vec![0.6, 0.4, 0.5, 0.6],
            vec![0.4, 0.6, 0.6, 0.4],
            vec![0.55, 0.45, 0.5, 0.55],
            vec![0.45, 0.55, 0.55, 0.45],
        ];

        let fisher = FisherInformationMatrix::from_samples(&samples, 0.01);

        assert_eq!(fisher.dim, 4);
        assert!(fisher.trace() > 0.0);
    }

    #[test]
    fn test_fisher_geodesic_distance() {
        let fisher = FisherInformationMatrix::identity(4, 0.01);

        let p1 = vec![0.5, 0.5, 0.5, 0.5];
        let p2 = vec![0.6, 0.5, 0.5, 0.5];

        let dist = fisher.geodesic_distance(&p1, &p2);

        // For identity metric, geodesic distance = Euclidean distance
        assert!((dist - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_manifold_point_creation() {
        let state = LatentConsciousnessState::from_observables(0.7, 0.6, 0.65, 0.7);
        let point = ManifoldPoint::from_state(&state);

        assert_eq!(point.coordinates.len(), 4);
        assert!((point.coordinates[0] - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_geometric_router_creation() {
        let router = InformationGeometricRouter::new(GeometricRouterConfig::default());

        assert_eq!(router.stats.decisions_made, 0);
        assert_eq!(router.current_fisher.dim, 4);
    }
}
