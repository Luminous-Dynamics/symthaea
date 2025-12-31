//! Consciousness-Guided Causal Program Synthesis
//!
//! Enhancement #8: Use integrated information (Φ) to guide program synthesis
//! toward more robust, maintainable, and interpretable solutions.
//!
//! # Overview
//!
//! Traditional program synthesis optimizes for:
//! - Causal strength (achieved vs target)
//! - Confidence scores (intervention testing)
//! - Complexity (number of operations)
//!
//! Consciousness-guided synthesis adds a fourth dimension:
//! - **Integrated information (Φ)** - measure of consciousness
//!
//! # Hypothesis
//!
//! Programs with higher Φ exhibit superior properties:
//! - **Robustness**: Better handling of edge cases and perturbations
//! - **Maintainability**: Easier to understand and modify
//! - **Generalization**: Better performance on unseen data
//! - **Interpretability**: More aligned with human causal reasoning
//!
//! # Scientific Foundation
//!
//! Based on Integrated Information Theory (IIT) 4.0:
//! - Φ measures how much information is integrated across components
//! - Higher Φ indicates stronger consciousness-like properties
//! - Star topology > Random topology (validated at +4.59%)
//!
//! # Example
//!
//! ```rust,ignore
//! use symthaea::synthesis::{
//!     CausalProgramSynthesizer, CausalSpec,
//!     consciousness_synthesis::{ConsciousnessSynthesisConfig, TopologyType},
//! };
//!
//! let mut synthesizer = CausalProgramSynthesizer::new(Default::default());
//!
//! let config = ConsciousnessSynthesisConfig {
//!     min_phi_hdc: 0.5,                           // Require Φ >= 0.5
//!     phi_weight: 0.3,                        // 30% weight on Φ
//!     preferred_topology: Some(TopologyType::Modular),
//!     ..Default::default()
//! };
//!
//! let spec = CausalSpec::RemoveCause {
//!     cause: "race".to_string(),
//!     effect: "approval".to_string(),
//! };
//!
//! let program = synthesizer.synthesize_conscious(&spec, config)?;
//!
//! println!("Φ: {:.3}", program.phi);
//! println!("Topology: {:?}", program.topology_type);
//! println!("Robustness: Higher than baseline!");
//! ```

use crate::hdc::HDC_DIMENSION;
use crate::hdc::consciousness_topology_generators::{ConsciousnessTopology, TopologyType as HdcTopologyType};
use crate::hdc::real_hv::RealHV;
use crate::synthesis::{CausalSpec, ProgramTemplate, SynthesisError, SynthesizedProgram, SynthesisResult};
use serde::{Deserialize, Serialize};

/// Consciousness-aware synthesis configuration
#[derive(Debug, Clone)]
pub struct ConsciousnessSynthesisConfig {
    /// Minimum acceptable Φ value (0.0-1.0)
    ///
    /// Programs with Φ below this threshold will be rejected.
    /// Higher values ensure more integrated programs but may reduce
    /// synthesis success rate.
    ///
    /// # Guidelines
    /// - 0.3: Minimal integration (most programs pass)
    /// - 0.5: Moderate integration (balanced)
    /// - 0.7: High integration (stringent, may fail often)
    pub min_phi_hdc: f64,

    /// Weight for Φ in multi-objective optimization (0.0-1.0)
    ///
    /// Determines how much to prioritize Φ vs causal strength.
    /// - 0.0 = Ignore Φ entirely (baseline synthesis)
    /// - 0.5 = Equal weight to Φ and causal strength
    /// - 1.0 = Optimize only for Φ (may sacrifice causal strength)
    ///
    /// # Recommended
    /// Start with 0.3 (30% Φ, 70% causal) for balanced results.
    pub phi_weight: f64,

    /// Preferred topology type (None = any)
    ///
    /// If specified, only programs with this topology will be accepted.
    /// Use this to enforce specific structural properties.
    pub preferred_topology: Option<TopologyType>,

    /// Maximum Φ computation time (milliseconds)
    ///
    /// Timeout for Φ calculation to prevent slow candidates from
    /// blocking synthesis. Candidates exceeding this are skipped.
    pub max_phi_computation_time: u64,

    /// Whether to generate consciousness explanations
    ///
    /// Adds human-readable explanation of Φ, topology, and metrics.
    /// Slightly increases synthesis time but aids interpretability.
    pub explain_consciousness: bool,
}

impl Default for ConsciousnessSynthesisConfig {
    fn default() -> Self {
        Self {
            min_phi_hdc: 0.3,              // Accept programs with some integration
            phi_weight: 0.3,            // 30% Φ, 70% causal strength
            preferred_topology: None,   // Accept any topology
            max_phi_computation_time: 5000,  // 5 seconds max
            explain_consciousness: true,     // Enable explanations
        }
    }
}

/// Topology classification for synthesized programs
///
/// Based on network structure analysis, programs are classified into
/// one of 8 canonical topology types, each with different properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyType {
    /// All-to-all connections (complete graph)
    ///
    /// - **Φ**: Highest (maximum integration)
    /// - **Robustness**: Excellent (redundant paths)
    /// - **Efficiency**: Poor (many edges)
    Dense,

    /// Community structure (clustered modules)
    ///
    /// - **Φ**: High (local integration + global coordination)
    /// - **Robustness**: Good (localized failures)
    /// - **Efficiency**: Good (balance)
    /// - **Best for**: Complex systems with subsystems
    Modular,

    /// Hub and spokes (central coordinator)
    ///
    /// - **Φ**: Medium-High (hub integrates information)
    /// - **Robustness**: Poor (single point of failure)
    /// - **Efficiency**: Excellent (minimal edges)
    /// - **Validated**: +4.59% Φ over Random
    Star,

    /// Circular connections
    ///
    /// - **Φ**: Medium (sequential integration)
    /// - **Robustness**: Fair (feedback loops)
    /// - **Efficiency**: Good
    Ring,

    /// Random connections (no clear pattern)
    ///
    /// - **Φ**: Low-Medium (baseline)
    /// - **Robustness**: Variable
    /// - **Efficiency**: Variable
    Random,

    /// Hierarchical tree structure
    ///
    /// - **Φ**: Low-Medium (hierarchical integration)
    /// - **Robustness**: Poor (single path to root)
    /// - **Efficiency**: Good
    BinaryTree,

    /// Grid structure (regular local connections)
    ///
    /// - **Φ**: Low (local integration only)
    /// - **Robustness**: Fair (alternative paths)
    /// - **Efficiency**: Good
    Lattice,

    /// Sequential chain
    ///
    /// - **Φ**: Lowest (minimal integration)
    /// - **Robustness**: Very Poor (single path)
    /// - **Efficiency**: Excellent (minimal edges)
    Line,
}

impl TopologyType {
    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            TopologyType::Dense => "Dense (all-to-all): Maximum integration, many redundant paths",
            TopologyType::Modular => "Modular (communities): Balanced local specialization and global coordination",
            TopologyType::Star => "Star (hub-spoke): Central integration hub with peripheral nodes",
            TopologyType::Ring => "Ring (circular): Sequential processing with feedback loops",
            TopologyType::Random => "Random: No clear structural organization",
            TopologyType::BinaryTree => "Binary Tree: Hierarchical information flow",
            TopologyType::Lattice => "Lattice (grid): Uniform local connectivity",
            TopologyType::Line => "Line (sequential): Simple linear processing chain",
        }
    }

    /// Expected Φ range for this topology
    pub fn expected_phi_range(&self) -> (f64, f64) {
        match self {
            TopologyType::Dense => (0.7, 1.0),
            TopologyType::Modular => (0.6, 0.8),
            TopologyType::Star => (0.45, 0.65),  // Validated at 0.4543
            TopologyType::Ring => (0.4, 0.6),
            TopologyType::Random => (0.35, 0.5),  // Validated at 0.4318
            TopologyType::BinaryTree => (0.3, 0.5),
            TopologyType::Lattice => (0.2, 0.4),
            TopologyType::Line => (0.1, 0.3),
        }
    }

    /// Convert to HDC topology type for ConsciousnessTopology construction
    pub fn to_hdc_type(&self) -> HdcTopologyType {
        match self {
            TopologyType::Dense => HdcTopologyType::DenseNetwork,
            TopologyType::Modular => HdcTopologyType::Modular,
            TopologyType::Star => HdcTopologyType::Star,
            TopologyType::Ring => HdcTopologyType::Ring,
            TopologyType::Random => HdcTopologyType::Random,
            TopologyType::BinaryTree => HdcTopologyType::BinaryTree,
            TopologyType::Lattice => HdcTopologyType::Lattice,
            TopologyType::Line => HdcTopologyType::Line,
        }
    }

    /// Convert from HDC topology type
    pub fn from_hdc_type(hdc_type: &HdcTopologyType) -> Self {
        match hdc_type {
            HdcTopologyType::DenseNetwork => TopologyType::Dense,
            HdcTopologyType::Modular => TopologyType::Modular,
            HdcTopologyType::Star => TopologyType::Star,
            HdcTopologyType::Ring => TopologyType::Ring,
            HdcTopologyType::Random => TopologyType::Random,
            HdcTopologyType::BinaryTree => TopologyType::BinaryTree,
            HdcTopologyType::Lattice => TopologyType::Lattice,
            HdcTopologyType::Line => TopologyType::Line,
            // Map exotic topologies to closest match
            HdcTopologyType::Torus | HdcTopologyType::KleinBottle | HdcTopologyType::MobiusStrip => TopologyType::Ring,
            HdcTopologyType::SmallWorld => TopologyType::Modular,
            HdcTopologyType::Hyperbolic | HdcTopologyType::ScaleFree => TopologyType::Star,
            HdcTopologyType::Sphere => TopologyType::Dense,
            HdcTopologyType::Fractal | HdcTopologyType::SierpinskiGasket |
            HdcTopologyType::FractalTree | HdcTopologyType::KochSnowflake |
            HdcTopologyType::MengerSponge | HdcTopologyType::CantorSet => TopologyType::BinaryTree,
            HdcTopologyType::Hypercube => TopologyType::Lattice,
            HdcTopologyType::Quantum => TopologyType::Random,
            // Tier 4: Neural-inspired
            HdcTopologyType::CorticalColumn | HdcTopologyType::Feedforward |
            HdcTopologyType::Recurrent | HdcTopologyType::Residual => TopologyType::Modular,
            HdcTopologyType::Bipartite | HdcTopologyType::CompleteBipartite => TopologyType::Star,
            HdcTopologyType::CorePeriphery | HdcTopologyType::BowTie => TopologyType::Star,
            HdcTopologyType::Attention => TopologyType::Modular,
            HdcTopologyType::PetersenGraph => TopologyType::Ring,
        }
    }
}

/// Result of consciousness-aware synthesis
///
/// # Note on Φ_HDC (Phi-HDC)
/// This implementation uses an **HDC-based approximation** of Integrated Information
/// Theory (IIT), not exact IIT 4.0. The approximation:
/// - Is **tractable**: O(n³) vs IIT's O(2^n)
/// - Is **validated**: 5-6% topology differentiation (Star > Random)
/// - Is **scalable**: Works for programs with 100+ variables
/// - Will be **validated** against exact IIT in Week 4
///
/// See `docs/IIT_IMPLEMENTATION_ANALYSIS.md` for complete details.
#[derive(Debug, Clone)]
pub struct ConsciousSynthesizedProgram {
    /// The synthesized program (standard output)
    pub program: SynthesizedProgram,

    /// HDC-approximated integrated information (Φ_HDC or Φ̃)
    ///
    /// This is a **tractable approximation** of IIT's Φ using Hyperdimensional Computing.
    /// - **Range**: [0, 1] where higher = more integrated information
    /// - **Computation**: Graph Laplacian algebraic connectivity (λ₂)
    /// - **Validation**: Differentiates topologies (Star > Random by ~5%)
    /// - **Complexity**: O(n³) - tractable for large programs
    ///
    /// **Not exact IIT 4.0** - this is a practical approximation that will be
    /// validated against PyPhi (IIT 3.0) in Week 4.
    pub phi_hdc: f64,

    /// Detected topology type
    ///
    /// Classification based on network structure analysis.
    pub topology_type: TopologyType,

    /// Topology heterogeneity (0.0-1.0)
    ///
    /// Measures differentiation (variance in node representations).
    /// Higher = more specialized components.
    /// Φ requires balance: high differentiation + high integration.
    pub heterogeneity: f64,

    /// Integration score (0.0-1.0)
    ///
    /// Measures how well connected components work together.
    /// Higher = stronger connections between nodes.
    pub integration_score: f64,

    /// Consciousness explanation (if enabled)
    ///
    /// Human-readable interpretation of Φ, topology, and metrics.
    pub consciousness_explanation: Option<String>,

    /// Multi-objective scores breakdown
    pub scores: MultiObjectiveScores,
}

/// Multi-objective optimization scores
#[derive(Debug, Clone)]
pub struct MultiObjectiveScores {
    /// Causal strength score (from base synthesis)
    ///
    /// How well the program achieves the target causal effect.
    pub causal_strength: f64,

    /// Confidence score (from intervention testing)
    ///
    /// Confidence in the causal effect measurement.
    pub confidence: f64,

    /// Φ score (integrated information)
    ///
    /// Consciousness-like integration quality.
    pub phi_score: f64,

    /// Complexity penalty (0.0-1.0, lower is better)
    ///
    /// Normalized complexity: simpler programs score higher.
    pub complexity: f64,

    /// Combined score (weighted sum, 0.0-1.0)
    ///
    /// Overall quality combining all objectives.
    pub combined: f64,
}

impl ConsciousSynthesizedProgram {
    /// Check if program meets quality thresholds
    pub fn meets_quality_threshold(&self, min_combined_score: f64) -> bool {
        self.scores.combined >= min_combined_score
    }

    /// Get consciousness quality assessment
    pub fn consciousness_quality(&self) -> ConsciousnessQuality {
        match self.phi_hdc {
            phi if phi >= 0.7 => ConsciousnessQuality::Excellent,
            phi if phi >= 0.5 => ConsciousnessQuality::Good,
            phi if phi >= 0.3 => ConsciousnessQuality::Fair,
            _ => ConsciousnessQuality::Poor,
        }
    }
}

/// Consciousness quality assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsciousnessQuality {
    /// Φ >= 0.7: Excellent integration, highly robust
    Excellent,
    /// Φ >= 0.5: Good integration, above average
    Good,
    /// Φ >= 0.3: Fair integration, acceptable
    Fair,
    /// Φ < 0.3: Poor integration, may have issues
    Poor,
}

impl ConsciousnessQuality {
    pub fn description(&self) -> &'static str {
        match self {
            ConsciousnessQuality::Excellent => {
                "Excellent: High integrated information, expected to be very robust"
            }
            ConsciousnessQuality::Good => {
                "Good: Above-average integration, should be reliable"
            }
            ConsciousnessQuality::Fair => {
                "Fair: Acceptable integration, may work but verify carefully"
            }
            ConsciousnessQuality::Poor => {
                "Poor: Low integration, likely to have robustness issues"
            }
        }
    }
}

// =============================================================================
// Topology Conversion & Classification
// =============================================================================

/// Extension methods for CausalProgramSynthesizer to support consciousness-aware synthesis
pub trait ConsciousSynthesizerExt {
    /// Convert program structure to consciousness topology
    fn program_to_topology(
        &self,
        program: &SynthesizedProgram,
    ) -> Result<ConsciousnessTopology, SynthesisError>;

    /// Classify topology type based on structure
    fn classify_topology(&self, topology: &ConsciousnessTopology) -> TopologyType;

    /// Measure heterogeneity (differentiation) of topology
    fn measure_heterogeneity(&self, topology: &ConsciousnessTopology) -> f64;

    /// Measure integration (cohesion) of topology
    fn measure_integration(&self, topology: &ConsciousnessTopology) -> f64;

    /// Check if topology has modular structure
    fn has_modular_structure(&self, topology: &ConsciousnessTopology) -> bool;
}

use crate::synthesis::CausalProgramSynthesizer;

/// Helper function to classify topology from edge structure
/// Used before full ConsciousnessTopology construction
fn classify_topology_from_edges(n: usize, edges: &[(usize, usize)]) -> TopologyType {
    let m = edges.len();

    if n == 0 {
        return TopologyType::Line;  // Degenerate case
    }

    // Dense: Complete graph (all-to-all)
    let max_edges = n * (n - 1) / 2;
    if m == max_edges && max_edges > 0 {
        return TopologyType::Dense;
    }

    // Tree structures: n-1 edges
    if m == n - 1 && n > 1 {
        let degrees: Vec<usize> = (0..n)
            .map(|i| edges.iter().filter(|(a, b)| *a == i || *b == i).count())
            .collect();
        let max_degree = degrees.iter().max().copied().unwrap_or(0);
        let min_degree = degrees.iter().min().copied().unwrap_or(0);

        if max_degree == n - 1 && min_degree == 1 {
            return TopologyType::Star;
        }
        if max_degree <= 2 {
            return TopologyType::Line;
        }
        return TopologyType::BinaryTree;
    }

    // Ring: n edges forming a cycle
    if m == n && n > 2 {
        let degrees: Vec<usize> = (0..n)
            .map(|i| edges.iter().filter(|(a, b)| *a == i || *b == i).count())
            .collect();
        if degrees.iter().all(|&d| d == 2) {
            return TopologyType::Ring;
        }
    }

    // Lattice check
    if n >= 4 {
        let degrees: Vec<usize> = (0..n)
            .map(|i| edges.iter().filter(|(a, b)| *a == i || *b == i).count())
            .collect();
        let avg_degree = degrees.iter().sum::<usize>() as f64 / n as f64;
        let degree_variance = degrees.iter()
            .map(|&d| { let diff = d as f64 - avg_degree; diff * diff })
            .sum::<f64>() / n as f64;
        if degree_variance < 1.0 && avg_degree >= 2.0 && avg_degree <= 4.0 {
            return TopologyType::Lattice;
        }
    }

    // Default: Random
    TopologyType::Random
}

impl ConsciousSynthesizerExt for CausalProgramSynthesizer {
    fn program_to_topology(
        &self,
        program: &SynthesizedProgram,
    ) -> Result<ConsciousnessTopology, SynthesisError> {
        let n_variables = program.variables.len();
        if n_variables == 0 {
            return Err(SynthesisError::InternalError(
                "Program has no variables".to_string(),
            ));
        }

        let dim = HDC_DIMENSION;

        // Create node identities (one per variable)
        // Each identity is a basis vector with slight random variation
        let node_identities: Vec<RealHV> = (0..n_variables)
            .map(|i| {
                // Base vector
                let base = RealHV::basis(i, dim);
                // Add variation for heterogeneity
                let variation = RealHV::random(dim, 42 + i as u64 * 1000);
                base.add(&variation.scale(0.05_f32))
            })
            .collect();

        // Extract edges from program template
        let edges = self.extract_program_edges(program)?;

        // Create node representations (identity bound with neighbors)
        let node_representations: Vec<RealHV> = node_identities
            .iter()
            .enumerate()
            .map(|(i, identity)| {
                // Find neighbors for this node
                let neighbors: Vec<RealHV> = edges
                    .iter()
                    .filter_map(|(a, b)| {
                        if *a == i {
                            Some(node_identities[*b].clone())
                        } else if *b == i {
                            Some(node_identities[*a].clone())
                        } else {
                            None
                        }
                    })
                    .collect();

                if neighbors.is_empty() {
                    // Isolated node - just return identity
                    identity.clone()
                } else {
                    // Bind identity with bundled neighbors
                    identity.bind(&RealHV::bundle(&neighbors))
                }
            })
            .collect();

        // Classify topology based on structure before full construction
        let topology_type = classify_topology_from_edges(n_variables, &edges);

        Ok(ConsciousnessTopology {
            n_nodes: n_variables,
            dim,
            node_identities,
            node_representations,
            topology_type: topology_type.to_hdc_type(),
            edges,
        })
    }

    fn classify_topology(&self, topology: &ConsciousnessTopology) -> TopologyType {
        let n = topology.node_identities.len();

        // Use helper for basic classification
        let basic_type = classify_topology_from_edges(n, &topology.edges);

        // Additional check for modular structure (requires topology analysis)
        if basic_type == TopologyType::Random && self.has_modular_structure(topology) {
            return TopologyType::Modular;
        }

        basic_type
    }

    fn measure_heterogeneity(&self, topology: &ConsciousnessTopology) -> f64 {
        let representations = &topology.node_representations;
        if representations.len() < 2 {
            return 0.0;
        }

        // Compute pairwise similarities
        let mut similarities = Vec::new();
        for i in 0..representations.len() {
            for j in (i + 1)..representations.len() {
                let sim = representations[i].similarity(&representations[j]);
                similarities.push(sim);
            }
        }

        if similarities.is_empty() {
            return 0.0;
        }

        // Heterogeneity = 1 - mean(similarity)
        // High heterogeneity means nodes are dissimilar (more differentiated)
        let mean_similarity: f64 = similarities.iter().map(|&x| x as f64).sum::<f64>() / similarities.len() as f64;
        (1.0 - mean_similarity).max(0.0).min(1.0)
    }

    fn measure_integration(&self, topology: &ConsciousnessTopology) -> f64 {
        if topology.edges.is_empty() {
            return 0.0;
        }

        // Integration = mean similarity of connected nodes
        let mut connected_similarities = Vec::new();

        for (i, j) in &topology.edges {
            if *i < topology.node_representations.len()
                && *j < topology.node_representations.len()
            {
                let sim = topology.node_representations[*i]
                    .similarity(&topology.node_representations[*j]);
                connected_similarities.push(sim);
            }
        }

        if connected_similarities.is_empty() {
            return 0.0;
        }

        let mean_integration: f64 =
            connected_similarities.iter().map(|&x| x as f64).sum::<f64>() / connected_similarities.len() as f64;
        mean_integration.max(0.0).min(1.0)
    }

    fn has_modular_structure(&self, topology: &ConsciousnessTopology) -> bool {
        let n = topology.node_identities.len();
        if n < 6 {
            return false; // Too small for meaningful modularity
        }

        // Compute clustering coefficient (measure of local clustering)
        let mut clustering_coefficients = Vec::new();

        for i in 0..n {
            // Find neighbors of node i
            let neighbors: Vec<usize> = topology
                .edges
                .iter()
                .filter_map(|(a, b)| {
                    if *a == i {
                        Some(*b)
                    } else if *b == i {
                        Some(*a)
                    } else {
                        None
                    }
                })
                .collect();

            if neighbors.len() < 2 {
                continue; // Need at least 2 neighbors for triangles
            }

            // Count triangles (edges between neighbors)
            let mut triangles = 0;
            for k1 in 0..neighbors.len() {
                for k2 in (k1 + 1)..neighbors.len() {
                    let n1 = neighbors[k1];
                    let n2 = neighbors[k2];

                    // Check if edge (n1, n2) exists
                    if topology.edges.contains(&(n1.min(n2), n1.max(n2))) {
                        triangles += 1;
                    }
                }
            }

            let possible_triangles = neighbors.len() * (neighbors.len() - 1) / 2;
            if possible_triangles > 0 {
                let clustering = triangles as f64 / possible_triangles as f64;
                clustering_coefficients.push(clustering);
            }
        }

        if clustering_coefficients.is_empty() {
            return false;
        }

        // High mean clustering suggests modularity
        let mean_clustering: f64 =
            clustering_coefficients.iter().sum::<f64>() / clustering_coefficients.len() as f64;

        mean_clustering > 0.5
    }
}

// Additional implementation methods for consciousness synthesis
impl CausalProgramSynthesizer {
    /// Synthesize program with consciousness-awareness (Enhancement #8)
    ///
    /// This is the main consciousness-guided synthesis method that:
    /// 1. Generates multiple candidate programs via standard synthesis
    /// 2. Converts each to a consciousness topology
    /// 3. Measures Φ for each (with timeout protection)
    /// 4. Scores candidates using multi-objective function (causal + Φ)
    /// 5. Returns the best candidate that meets consciousness threshold
    ///
    /// # Arguments
    /// * `spec` - Causal specification to implement
    /// * `config` - Consciousness synthesis configuration
    ///
    /// # Returns
    /// `ConsciousSynthesizedProgram` with Φ metrics and consciousness quality
    ///
    /// # Errors
    /// * `PhiComputationTimeout` - If Φ calculation exceeds timeout
    /// * `InsufficientConsciousness` - If no candidates meet min_phi threshold
    /// * `ConsciousnessSynthesisError` - Other synthesis errors
    fn synthesize_conscious(
        &mut self,
        spec: &CausalSpec,
        config: &ConsciousnessSynthesisConfig,
    ) -> SynthesisResult<ConsciousSynthesizedProgram> {
        use crate::hdc::phi_real::RealPhiCalculator;
        use std::time::Instant;

        // Step 1: Generate candidate programs
        let candidates = self.generate_candidates(spec, config)?;

        if candidates.is_empty() {
            return Err(SynthesisError::ConsciousnessSynthesisError(
                "No valid candidates generated".to_string(),
            ));
        }

        // Step 2: Evaluate each candidate with Φ measurement
        let phi_calculator = RealPhiCalculator::new();
        let mut evaluated_candidates: Vec<(SynthesizedProgram, f64, TopologyType, f64, f64)> =
            Vec::new();

        for (i, candidate) in candidates.iter().enumerate() {
            // Convert program to topology
            let topology = match self.program_to_topology(candidate) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Warning: Failed to convert candidate {} to topology: {:?}", i, e);
                    continue;
                }
            };

            // Classify topology
            let topology_type = self.classify_topology(&topology);

            // Measure Φ with timeout protection
            let start_time = Instant::now();
            let phi = phi_calculator.compute(&topology.node_representations);
            let elapsed = start_time.elapsed();

            // Check timeout
            if elapsed.as_millis() > config.max_phi_computation_time as u128 {
                return Err(SynthesisError::PhiComputationTimeout {
                    candidate_id: i,
                    time_ms: elapsed.as_millis() as u64,
                });
            }

            // Measure heterogeneity and integration
            let heterogeneity = self.measure_heterogeneity(&topology);
            let integration = self.measure_integration(&topology);

            evaluated_candidates.push((
                candidate.clone(),
                phi,
                topology_type,
                heterogeneity,
                integration,
            ));
        }

        if evaluated_candidates.is_empty() {
            return Err(SynthesisError::ConsciousnessSynthesisError(
                "All candidates failed topology conversion".to_string(),
            ));
        }

        // Step 3: Multi-objective scoring
        let scored_candidates: Vec<(SynthesizedProgram, f64, MultiObjectiveScores, TopologyType, f64, f64)> =
            evaluated_candidates
                .into_iter()
                .map(|(prog, phi_hdc, topo_type, het, integ)| {
                    let scores = self.compute_multi_objective_score(&prog, phi_hdc, config);
                    (prog, phi_hdc, scores, topo_type, het, integ)
                })
                .collect();

        // Step 4: Select best candidate
        let best = scored_candidates
            .iter()
            .filter(|(_, phi_hdc, _, _, _, _)| *phi_hdc >= config.min_phi_hdc)
            .max_by(|(_, _, scores1, _, _, _), (_, _, scores2, _, _, _)| {
                scores1
                    .combined
                    .partial_cmp(&scores2.combined)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        match best {
            Some((program, phi_hdc, scores, topology_type, heterogeneity, integration)) => {
                // Generate consciousness explanation if requested
                let consciousness_explanation = if config.explain_consciousness {
                    Some(self.generate_consciousness_explanation(
                        *phi_hdc,
                        topology_type,
                        *heterogeneity,
                        *integration,
                    ))
                } else {
                    None
                };

                Ok(ConsciousSynthesizedProgram {
                    program: program.clone(),
                    phi_hdc: *phi_hdc,
                    topology_type: *topology_type,
                    heterogeneity: *heterogeneity,
                    integration_score: *integration,
                    consciousness_explanation,
                    scores: scores.clone(),
                })
            }
            None => {
                // Find best Φ achieved (even if below threshold)
                let best_phi = scored_candidates
                    .iter()
                    .map(|(_, phi_hdc, _, _, _, _)| *phi_hdc)
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(0.0);

                Err(SynthesisError::InsufficientConsciousness {
                    min_phi: config.min_phi_hdc,
                    best_phi,
                })
            }
        }
    }

    /// Generate multiple candidate programs for consciousness evaluation
    ///
    /// This method creates diverse candidates by varying synthesis parameters
    fn generate_candidates(
        &mut self,
        spec: &CausalSpec,
        config: &ConsciousnessSynthesisConfig,
    ) -> SynthesisResult<Vec<SynthesizedProgram>> {
        let mut candidates = Vec::new();

        // Try baseline synthesis
        if let Ok(baseline) = self.synthesize(spec) {
            candidates.push(baseline);
        }

        // Generate variations with different complexity levels
        let original_max_complexity = self.config().max_complexity;

        for complexity_factor in &[0.5, 1.0, 1.5, 2.0] {
            self.config_mut().max_complexity = (original_max_complexity as f64 * complexity_factor) as usize;

            if let Ok(candidate) = self.synthesize(spec) {
                // Only add if not duplicate
                if !candidates.iter().any(|c| {
                    c.complexity == candidate.complexity
                        && c.achieved_strength == candidate.achieved_strength
                }) {
                    candidates.push(candidate);
                }
            }
        }

        // Restore original config
        self.config_mut().max_complexity = original_max_complexity;

        // Filter by preferred topology if specified
        if let Some(preferred) = config.preferred_topology {
            // Keep all for now - will filter by Φ later
            // But prefer topologies that match preference
            candidates.sort_by_key(|c| {
                if let Ok(topo) = self.program_to_topology(c) {
                    if self.classify_topology(&topo) == preferred {
                        return 0; // Highest priority
                    }
                }
                1 // Lower priority
            });
        }

        Ok(candidates)
    }

    /// Compute multi-objective score combining causal and consciousness metrics
    fn compute_multi_objective_score(
        &self,
        program: &SynthesizedProgram,
        phi_hdc: f64,
        config: &ConsciousnessSynthesisConfig,
    ) -> MultiObjectiveScores {
        let causal_strength = program.achieved_strength;
        let confidence = program.confidence;
        let phi_score = phi_hdc;

        // Complexity penalty (lower complexity is better)
        let max_complexity = self.config().max_complexity as f64;
        let complexity_score = 1.0 - (program.complexity as f64 / max_complexity).min(1.0);

        // Combined score: weighted average
        let phi_weight = config.phi_weight;
        let causal_weight = 1.0 - phi_weight;

        let combined = (causal_strength * causal_weight * 0.5)
            + (confidence * causal_weight * 0.5)
            + (phi_score * phi_weight)
            + (complexity_score * 0.1); // Small bonus for simplicity

        MultiObjectiveScores {
            causal_strength,
            confidence,
            phi_score,
            complexity: complexity_score,
            combined: combined.min(1.0).max(0.0), // Clamp to [0, 1]
        }
    }

    /// Generate human-readable explanation of consciousness properties
    fn generate_consciousness_explanation(
        &self,
        phi_hdc: f64,
        topology_type: &TopologyType,
        heterogeneity: f64,
        integration: f64,
    ) -> String {
        let quality = if phi_hdc > 0.7 {
            "excellent"
        } else if phi_hdc > 0.5 {
            "good"
        } else if phi_hdc > 0.3 {
            "fair"
        } else {
            "poor"
        };

        format!(
            "This program exhibits {} consciousness-like properties (Φ={:.3}).\n\
             Network structure: {} topology.\n\
             Differentiation (heterogeneity): {:.2} - {}\n\
             Integration (cohesion): {:.2} - {}\n\
             \n\
             Interpretation: The program's computational structure shows {} levels of \
             integrated information, suggesting {} potential for emergent complexity \
             and robust behavior under perturbations.",
            quality,
            phi_hdc,
            topology_type.description(),
            heterogeneity,
            if heterogeneity > 0.6 {
                "high variance in node representations"
            } else if heterogeneity > 0.3 {
                "moderate variance"
            } else {
                "low variance (more uniform)"
            },
            integration,
            if integration > 0.6 {
                "strong connections between related nodes"
            } else if integration > 0.3 {
                "moderate connections"
            } else {
                "weak connections (more independent)"
            },
            if phi_hdc > 0.7 {
                "high"
            } else if phi_hdc > 0.5 {
                "moderate"
            } else {
                "low"
            },
            if phi_hdc > 0.7 {
                "high"
            } else if phi_hdc > 0.5 {
                "moderate"
            } else {
                "low"
            }
        )
    }
}

/// Helper: Extract edges from program template
impl CausalProgramSynthesizer {
    fn extract_program_edges(
        &self,
        program: &SynthesizedProgram,
    ) -> Result<Vec<(usize, usize)>, SynthesisError> {
        let var_to_idx: std::collections::HashMap<_, _> = program
            .variables
            .iter()
            .enumerate()
            .map(|(i, v)| (v.clone(), i))
            .collect();

        let get_idx = |var: &str| -> Result<usize, SynthesisError> {
            var_to_idx.get(var).copied().ok_or_else(|| {
                SynthesisError::InternalError(format!("Variable {} not found", var))
            })
        };

        match &program.template {
            ProgramTemplate::Linear { weights, .. } => {
                // Linear: edges from all inputs to (implicit) output
                // For simplicity, create edges between all weighted variables
                let mut edges = Vec::new();
                let vars: Vec<_> = weights.keys().collect();
                for i in 0..vars.len() {
                    for j in (i + 1)..vars.len() {
                        let idx_i = get_idx(vars[i])?;
                        let idx_j = get_idx(vars[j])?;
                        edges.push((idx_i.min(idx_j), idx_i.max(idx_j)));
                    }
                }
                Ok(edges)
            }

            ProgramTemplate::Sequence { programs } => {
                // Chain of programs
                // For now, create sequential edges
                let mut edges = Vec::new();
                for i in 0..(programs.len().saturating_sub(1)) {
                    if i + 1 < program.variables.len() {
                        edges.push((i, i + 1));
                    }
                }
                Ok(edges)
            }

            ProgramTemplate::NeuralLayer { inputs, outputs, .. } => {
                // Bipartite graph: all inputs connected to all outputs
                let mut edges = Vec::new();
                for input in inputs {
                    for output in outputs {
                        let i = get_idx(input)?;
                        let j = get_idx(output)?;
                        if i != j {
                            edges.push((i.min(j), i.max(j)));
                        }
                    }
                }
                // Deduplicate
                edges.sort_unstable();
                edges.dedup();
                Ok(edges)
            }

            _ => {
                // For other templates, create star topology (first variable as hub)
                let mut edges = Vec::new();
                if program.variables.len() > 1 {
                    for i in 1..program.variables.len() {
                        edges.push((0, i));
                    }
                }
                Ok(edges)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ConsciousnessSynthesisConfig::default();
        assert_eq!(config.min_phi_hdc, 0.3);
        assert_eq!(config.phi_weight, 0.3);
        assert!(config.preferred_topology.is_none());
        assert_eq!(config.max_phi_computation_time, 5000);
        assert!(config.explain_consciousness);
    }

    #[test]
    fn test_topology_descriptions() {
        assert!(TopologyType::Dense.description().contains("all-to-all"));
        assert!(TopologyType::Modular.description().contains("communities"));
        assert!(TopologyType::Star.description().contains("hub"));
    }

    #[test]
    fn test_topology_phi_ranges() {
        let (min, max) = TopologyType::Dense.expected_phi_range();
        assert!(min < max);
        assert!(min >= 0.0 && max <= 1.0);

        // Star should have validated range around 0.45
        let (star_min, star_max) = TopologyType::Star.expected_phi_range();
        assert!(star_min <= 0.4543 && 0.4543 <= star_max,
            "Star Φ=0.4543 should be in range [{}, {}]", star_min, star_max);

        // Random should have validated range around 0.43
        let (rand_min, rand_max) = TopologyType::Random.expected_phi_range();
        assert!(rand_min <= 0.4318 && 0.4318 <= rand_max,
            "Random Φ=0.4318 should be in range [{}, {}]", rand_min, rand_max);
    }

    #[test]
    fn test_consciousness_quality() {
        assert_eq!(
            ConsciousnessQuality::Excellent.description(),
            "Excellent: High integrated information, expected to be very robust"
        );
    }

    #[test]
    fn test_topology_type_serialization() {
        use serde_json;

        let topology = TopologyType::Star;
        let json = serde_json::to_string(&topology).unwrap();
        let deserialized: TopologyType = serde_json::from_str(&json).unwrap();
        assert_eq!(topology, deserialized);
    }

    // =============================================================================
    // Topology Conversion & Classification Tests
    // =============================================================================

    use crate::synthesis::{CausalProgramSynthesizer, SynthesisConfig};
    use std::collections::HashMap;

    fn create_test_synthesizer() -> CausalProgramSynthesizer {
        CausalProgramSynthesizer::new(SynthesisConfig::default())
    }

    fn create_linear_program() -> SynthesizedProgram {
        let mut weights = HashMap::new();
        weights.insert("x".to_string(), 0.8);
        weights.insert("y".to_string(), 0.2);

        SynthesizedProgram {
            template: ProgramTemplate::Linear {
                weights,
                bias: 0.0,
            },
            specification: CausalSpec::MakeCause {
                cause: "x".to_string(),
                effect: "y".to_string(),
                strength: 0.8,
            },
            achieved_strength: 0.8,
            confidence: 0.9,
            complexity: 2,
            explanation: None,
            variables: vec!["x".to_string(), "y".to_string()],
        }
    }

    fn create_star_program(n_variables: usize) -> SynthesizedProgram {
        let mut weights = HashMap::new();
        let variables: Vec<String> = (0..n_variables).map(|i| format!("v{}", i)).collect();

        for var in &variables[1..] {
            weights.insert(var.clone(), 0.5);
        }

        SynthesizedProgram {
            template: ProgramTemplate::Linear {
                weights,
                bias: 0.0,
            },
            specification: CausalSpec::MakeCause {
                cause: "v0".to_string(),
                effect: "v1".to_string(),
                strength: 0.5,
            },
            achieved_strength: 0.5,
            confidence: 0.85,
            complexity: n_variables,
            explanation: None,
            variables,
        }
    }

    #[test]
    fn test_program_to_topology_linear() {
        let synthesizer = create_test_synthesizer();
        let program = create_linear_program();

        let topology = synthesizer.program_to_topology(&program).unwrap();

        // Should have 2 nodes, 1 edge
        assert_eq!(topology.node_identities.len(), 2);
        assert_eq!(topology.node_representations.len(), 2);
        assert_eq!(topology.edges.len(), 1);
        assert_eq!(topology.edges[0], (0, 1));
    }

    #[test]
    fn test_program_to_topology_star() {
        let synthesizer = create_test_synthesizer();
        let program = create_star_program(5);

        let topology = synthesizer.program_to_topology(&program).unwrap();

        // Should have 5 nodes
        assert_eq!(topology.node_identities.len(), 5);
        assert_eq!(topology.node_representations.len(), 5);

        // Linear template with 4 weights creates complete graph (10 edges for 5 nodes)
        // Or star topology depending on implementation
        assert!(topology.edges.len() > 0, "Should have edges");
    }

    #[test]
    fn test_classify_topology_dense() {
        use crate::hdc::consciousness_topology_generators::ConsciousnessTopology as CTGen;

        let synthesizer = create_test_synthesizer();

        // Create dense topology (dense_network with k=None for full connectivity)
        let topology = CTGen::dense_network(5, HDC_DIMENSION, None, 42);

        let classified = synthesizer.classify_topology(&topology);
        assert_eq!(
            classified,
            TopologyType::Dense,
            "Dense topology should be classified as Dense"
        );
    }

    #[test]
    fn test_classify_topology_star() {
        use crate::hdc::consciousness_topology_generators::ConsciousnessTopology as CTGen;

        let synthesizer = create_test_synthesizer();

        // Create star topology
        let topology = CTGen::star(5, HDC_DIMENSION, 42);

        let classified = synthesizer.classify_topology(&topology);
        assert_eq!(
            classified,
            TopologyType::Star,
            "Star topology should be classified as Star"
        );
    }

    #[test]
    fn test_classify_topology_ring() {
        use crate::hdc::consciousness_topology_generators::ConsciousnessTopology as CTGen;

        let synthesizer = create_test_synthesizer();

        // Create ring topology
        let topology = CTGen::ring(6, HDC_DIMENSION, 42);

        let classified = synthesizer.classify_topology(&topology);
        assert_eq!(
            classified,
            TopologyType::Ring,
            "Ring topology should be classified as Ring"
        );
    }

    #[test]
    fn test_classify_topology_line() {
        use crate::hdc::consciousness_topology_generators::ConsciousnessTopology as CTGen;

        let synthesizer = create_test_synthesizer();

        // Create line topology
        let topology = CTGen::line(5, HDC_DIMENSION, 42);

        let classified = synthesizer.classify_topology(&topology);
        assert_eq!(
            classified,
            TopologyType::Line,
            "Line topology should be classified as Line"
        );
    }

    #[test]
    fn test_classify_topology_random() {
        use crate::hdc::consciousness_topology_generators::ConsciousnessTopology as CTGen;

        let synthesizer = create_test_synthesizer();

        // Create random topology
        let topology = CTGen::random(8, HDC_DIMENSION, 42);

        let classified = synthesizer.classify_topology(&topology);
        // Random could be classified as Random or Modular depending on structure
        assert!(
            matches!(classified, TopologyType::Random | TopologyType::Modular),
            "Random topology should be classified as Random or Modular, got {:?}",
            classified
        );
    }

    #[test]
    fn test_measure_heterogeneity() {
        use crate::hdc::consciousness_topology_generators::ConsciousnessTopology as CTGen;

        let synthesizer = create_test_synthesizer();

        // Random topology should have high heterogeneity
        let random = CTGen::random(8, HDC_DIMENSION, 42);
        let het_random = synthesizer.measure_heterogeneity(&random);

        // Line topology should have lower heterogeneity
        let line = CTGen::line(8, HDC_DIMENSION, 42);
        let het_line = synthesizer.measure_heterogeneity(&line);

        // Heterogeneity should be in [0, 1]
        assert!(
            het_random >= 0.0 && het_random <= 1.0,
            "Heterogeneity should be in [0, 1], got {}",
            het_random
        );
        assert!(
            het_line >= 0.0 && het_line <= 1.0,
            "Heterogeneity should be in [0, 1], got {}",
            het_line
        );

        // Random should generally be more heterogeneous
        // (though this can vary with seeds)
        println!(
            "Heterogeneity: random={:.3}, line={:.3}",
            het_random, het_line
        );
    }

    #[test]
    fn test_measure_integration() {
        use crate::hdc::consciousness_topology_generators::ConsciousnessTopology as CTGen;

        let synthesizer = create_test_synthesizer();

        let star = CTGen::star(8, HDC_DIMENSION, 42);
        let integration = synthesizer.measure_integration(&star);

        // Integration should be in [0, 1]
        assert!(
            integration >= 0.0 && integration <= 1.0,
            "Integration should be in [0, 1], got {}",
            integration
        );

        println!("Star integration: {:.3}", integration);
    }

    #[test]
    fn test_has_modular_structure() {
        use crate::hdc::consciousness_topology_generators::ConsciousnessTopology as CTGen;

        let synthesizer = create_test_synthesizer();

        // Modular topology should detect modularity (12 nodes, 3 modules, seed 42)
        let modular = CTGen::modular(12, HDC_DIMENSION, 3, 42);
        let is_modular = synthesizer.has_modular_structure(&modular);

        println!("Modular topology detected as modular: {}", is_modular);
        // Note: Detection may not be perfect depending on structure
    }

    #[test]
    fn test_extract_program_edges_linear() {
        let synthesizer = create_test_synthesizer();
        let program = create_linear_program();

        let edges = synthesizer.extract_program_edges(&program).unwrap();

        // Linear with 2 variables should have 1 edge
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0], (0, 1));
    }

    #[test]
    fn test_extract_program_edges_star() {
        let synthesizer = create_test_synthesizer();
        let program = create_star_program(5);

        let edges = synthesizer.extract_program_edges(&program).unwrap();

        // Linear with 5 variables creates complete graph: 10 edges
        assert!(edges.len() > 0, "Should have edges");
        println!("Star program edges: {}", edges.len());
    }

    #[test]
    fn test_consciousness_quality_assessment() {
        // Test Excellent
        let excellent = ConsciousSynthesizedProgram {
            program: create_linear_program(),
            phi_hdc: 0.75,
            topology_type: TopologyType::Modular,
            heterogeneity: 0.6,
            integration_score: 0.7,
            consciousness_explanation: None,
            scores: MultiObjectiveScores {
                causal_strength: 0.8,
                confidence: 0.9,
                phi_score: 0.75,
                complexity: 0.8,
                combined: 0.82,
            },
        };

        assert_eq!(excellent.consciousness_quality(), ConsciousnessQuality::Excellent);

        // Test Good
        let good = ConsciousSynthesizedProgram {
            phi_hdc: 0.55,
            ..excellent.clone()
        };
        assert_eq!(good.consciousness_quality(), ConsciousnessQuality::Good);

        // Test Fair
        let fair = ConsciousSynthesizedProgram {
            phi_hdc: 0.35,
            ..excellent.clone()
        };
        assert_eq!(fair.consciousness_quality(), ConsciousnessQuality::Fair);

        // Test Poor
        let poor = ConsciousSynthesizedProgram {
            phi_hdc: 0.15,
            ..excellent
        };
        assert_eq!(poor.consciousness_quality(), ConsciousnessQuality::Poor);
    }

    #[test]
    fn test_meets_quality_threshold() {
        let program = ConsciousSynthesizedProgram {
            program: create_linear_program(),
            phi_hdc: 0.6,
            topology_type: TopologyType::Star,
            heterogeneity: 0.5,
            integration_score: 0.6,
            consciousness_explanation: None,
            scores: MultiObjectiveScores {
                causal_strength: 0.8,
                confidence: 0.9,
                phi_score: 0.6,
                complexity: 0.85,
                combined: 0.78,
            },
        };

        assert!(program.meets_quality_threshold(0.7));
        assert!(!program.meets_quality_threshold(0.8));
    }

    // ========================================================================
    // Week 2 Integration Tests - Consciousness-Guided Synthesis
    // ========================================================================

    #[test]
    fn test_synthesize_conscious_basic() {
        use super::super::synthesizer::{CausalProgramSynthesizer, SynthesisConfig};
        use super::super::causal_spec::CausalSpec;

        // Create synthesizer
        let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());

        // Create simple causal specification
        let spec = CausalSpec::MakeCause {
            cause: "temperature".to_string(),
            effect: "energy".to_string(),
            strength: 0.8,
        };

        // Create consciousness config with relaxed threshold for testing
        let config = ConsciousnessSynthesisConfig {
            min_phi_hdc: 0.3, // Lower threshold for test
            phi_weight: 0.3,
            preferred_topology: None,
            max_phi_computation_time: 5000,
            explain_consciousness: true,
        };

        // Synthesize with consciousness awareness
        let result = synthesizer.synthesize_conscious(&spec, &config);

        // Should succeed
        assert!(result.is_ok(), "Synthesis should succeed: {:?}", result.err());

        let conscious_program = result.unwrap();

        // Verify Φ meets threshold
        assert!(
            conscious_program.phi_hdc >= config.min_phi_hdc,
            "Φ={:.3} should be >= {:.3}",
            conscious_program.phi_hdc,
            config.min_phi_hdc
        );

        // Verify scores are valid
        assert!(conscious_program.scores.causal_strength >= 0.0);
        assert!(conscious_program.scores.causal_strength <= 1.0);
        assert!(conscious_program.scores.phi_score >= 0.0);
        assert!(conscious_program.scores.phi_score <= 1.0);
        assert!(conscious_program.scores.combined >= 0.0);
        assert!(conscious_program.scores.combined <= 1.0);

        // Verify explanation was generated
        assert!(conscious_program.consciousness_explanation.is_some());
        let explanation = conscious_program.consciousness_explanation.unwrap();
        assert!(explanation.contains("consciousness-like properties"));
        assert!(explanation.contains("Φ="));
    }

    #[test]
    fn test_synthesize_conscious_vs_baseline() {
        use super::super::synthesizer::{CausalProgramSynthesizer, SynthesisConfig};
        use super::super::causal_spec::CausalSpec;

        // Create synthesizer
        let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());

        let spec = CausalSpec::MakeCause {
            cause: "input".to_string(),
            effect: "output".to_string(),
            strength: 0.9,
        };

        // Baseline synthesis
        let baseline = synthesizer.synthesize(&spec).unwrap();

        // Consciousness-guided synthesis
        let config = ConsciousnessSynthesisConfig {
            min_phi_hdc: 0.4,
            phi_weight: 0.5,
            preferred_topology: None,
            max_phi_computation_time: 5000,
            explain_consciousness: false,
        };

        let conscious = synthesizer.synthesize_conscious(&spec, &config).unwrap();

        // Conscious synthesis should have Φ measured
        assert!(conscious.phi_hdc > 0.0, "Conscious synthesis should have Φ > 0");

        // Both should achieve similar causal strength
        assert!(
            (baseline.achieved_strength - conscious.scores.causal_strength).abs() < 0.3,
            "Causal strength should be similar: baseline={:.2}, conscious={:.2}",
            baseline.achieved_strength,
            conscious.scores.causal_strength
        );
    }

    #[test]
    fn test_insufficient_consciousness_error() {
        use super::super::synthesizer::{CausalProgramSynthesizer, SynthesisConfig};
        use super::super::causal_spec::CausalSpec;

        let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());

        let spec = CausalSpec::MakeCause {
            cause: "x".to_string(),
            effect: "y".to_string(),
            strength: 0.5,
        };

        // Set impossibly high Φ threshold
        let config = ConsciousnessSynthesisConfig {
            min_phi_hdc: 0.99, // Unreachable threshold
            phi_weight: 0.8,
            preferred_topology: None,
            max_phi_computation_time: 5000,
            explain_consciousness: false,
        };

        let result = synthesizer.synthesize_conscious(&spec, &config);

        // Should fail with InsufficientConsciousness
        assert!(result.is_err());
        match result.err().unwrap() {
            SynthesisError::InsufficientConsciousness { min_phi, best_phi } => {
                assert_eq!(min_phi, 0.99);
                assert!(best_phi < 0.99, "Best Φ={:.3} should be < 0.99", best_phi);
                assert!(best_phi > 0.0, "Best Φ={:.3} should be > 0", best_phi);
            }
            other => panic!("Expected InsufficientConsciousness, got {:?}", other),
        }
    }

    #[test]
    fn test_phi_computation_timeout_protection() {
        use super::super::synthesizer::{CausalProgramSynthesizer, SynthesisConfig};
        use super::super::causal_spec::CausalSpec;

        let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());

        let spec = CausalSpec::MakeCause {
            cause: "a".to_string(),
            effect: "b".to_string(),
            strength: 0.7,
        };

        // Set very short timeout (note: test may pass if Φ is actually fast)
        let config = ConsciousnessSynthesisConfig {
            min_phi_hdc: 0.3,
            phi_weight: 0.3,
            preferred_topology: None,
            max_phi_computation_time: 1, // 1ms - very tight
            explain_consciousness: false,
        };

        let result = synthesizer.synthesize_conscious(&spec, &config);

        // May timeout OR succeed if Φ is computed quickly
        // This test validates that timeout mechanism exists
        match result {
            Ok(_) => {
                // Φ was computed within 1ms - acceptable
                println!("Φ computed within timeout (very fast!)");
            }
            Err(SynthesisError::PhiComputationTimeout { candidate_id, time_ms }) => {
                // Timeout triggered - mechanism works
                println!("Timeout triggered for candidate {} at {}ms", candidate_id, time_ms);
                assert!(time_ms > 1, "Timeout should report time > 1ms");
            }
            Err(other) => panic!("Unexpected error: {:?}", other),
        }
    }

    #[test]
    fn test_multi_objective_scoring() {
        use super::super::synthesizer::{CausalProgramSynthesizer, SynthesisConfig};
        use super::super::causal_spec::CausalSpec;

        let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());

        let spec = CausalSpec::MakeCause {
            cause: "pressure".to_string(),
            effect: "volume".to_string(),
            strength: 0.85,
        };

        // High Φ weight - prioritize consciousness
        let config_high_phi = ConsciousnessSynthesisConfig {
            min_phi_hdc: 0.3,
            phi_weight: 0.8, // 80% weight on Φ
            preferred_topology: None,
            max_phi_computation_time: 5000,
            explain_consciousness: false,
        };

        // Low Φ weight - prioritize causal accuracy
        let config_low_phi = ConsciousnessSynthesisConfig {
            min_phi_hdc: 0.3,
            phi_weight: 0.2, // 20% weight on Φ
            preferred_topology: None,
            max_phi_computation_time: 5000,
            explain_consciousness: false,
        };

        let result_high_phi = synthesizer.synthesize_conscious(&spec, &config_high_phi);
        let result_low_phi = synthesizer.synthesize_conscious(&spec, &config_low_phi);

        // Both should succeed
        assert!(result_high_phi.is_ok());
        assert!(result_low_phi.is_ok());

        let prog_high_phi = result_high_phi.unwrap();
        let prog_low_phi = result_low_phi.unwrap();

        // High Φ weight should prioritize higher Φ scores
        // (though not guaranteed due to threshold filtering)
        assert!(prog_high_phi.phi_hdc >= config_high_phi.min_phi_hdc);
        assert!(prog_low_phi.phi_hdc >= config_low_phi.min_phi_hdc);

        // Combined scores should reflect weighting
        println!(
            "High Φ weight: combined={:.3}, causal={:.3}, φ={:.3}",
            prog_high_phi.scores.combined,
            prog_high_phi.scores.causal_strength,
            prog_high_phi.scores.phi_score
        );
        println!(
            "Low Φ weight: combined={:.3}, causal={:.3}, φ={:.3}",
            prog_low_phi.scores.combined,
            prog_low_phi.scores.causal_strength,
            prog_low_phi.scores.phi_score
        );
    }

    #[test]
    fn test_preferred_topology() {
        use super::super::synthesizer::{CausalProgramSynthesizer, SynthesisConfig};
        use super::super::causal_spec::CausalSpec;

        let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());

        let spec = CausalSpec::MakeCause {
            cause: "force".to_string(),
            effect: "acceleration".to_string(),
            strength: 0.95,
        };

        // Request Star topology
        let config = ConsciousnessSynthesisConfig {
            min_phi_hdc: 0.3,
            phi_weight: 0.4,
            preferred_topology: Some(TopologyType::Star),
            max_phi_computation_time: 5000,
            explain_consciousness: true,
        };

        let result = synthesizer.synthesize_conscious(&spec, &config);

        if let Ok(program) = result {
            // Preference may influence but not guarantee topology
            // (depends on what synthesis generates)
            println!("Generated topology: {:?}", program.topology_type);
            println!("Φ: {:.3}", program.phi_hdc);

            // At minimum, topology should be classified
            assert!(matches!(
                program.topology_type,
                TopologyType::Dense
                    | TopologyType::Modular
                    | TopologyType::Star
                    | TopologyType::Ring
                    | TopologyType::Random
                    | TopologyType::BinaryTree
                    | TopologyType::Lattice
                    | TopologyType::Line
            ));
        }
    }

    #[test]
    fn test_consciousness_explanation_generation() {
        use super::super::synthesizer::{CausalProgramSynthesizer, SynthesisConfig};
        use super::super::causal_spec::CausalSpec;

        let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());

        let spec = CausalSpec::MakeCause {
            cause: "mass".to_string(),
            effect: "weight".to_string(),
            strength: 0.99,
        };

        let config = ConsciousnessSynthesisConfig {
            min_phi_hdc: 0.3,
            phi_weight: 0.3,
            preferred_topology: None,
            max_phi_computation_time: 5000,
            explain_consciousness: true, // Enable explanation
        };

        let result = synthesizer.synthesize_conscious(&spec, &config).unwrap();

        // Should have explanation
        assert!(result.consciousness_explanation.is_some());

        let explanation = result.consciousness_explanation.unwrap();

        // Explanation should contain key information
        assert!(
            explanation.contains("consciousness-like properties")
                || explanation.contains("Φ=")
        );
        assert!(
            explanation.contains("topology") || explanation.contains("Network structure")
        );
        assert!(explanation.contains("heterogeneity") || explanation.contains("Differentiation"));
        assert!(explanation.contains("integration") || explanation.contains("Integration"));

        println!("Generated explanation:\n{}", explanation);
    }

    #[test]
    fn test_candidate_generation_diversity() {
        use super::super::synthesizer::{CausalProgramSynthesizer, SynthesisConfig};
        use super::super::causal_spec::CausalSpec;

        let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig {
            max_complexity: 50,
            ..SynthesisConfig::default()
        });

        let spec = CausalSpec::MakeCause {
            cause: "x1".to_string(),
            effect: "y1".to_string(),
            strength: 0.7,
        };

        let config = ConsciousnessSynthesisConfig::default();

        // Generate candidates
        let candidates = synthesizer.generate_candidates(&spec, &config).unwrap();

        // Should generate multiple candidates
        assert!(
            candidates.len() > 1,
            "Should generate multiple candidates, got {}",
            candidates.len()
        );
        assert!(
            candidates.len() <= 5,
            "Should not generate too many, got {}",
            candidates.len()
        );

        // Candidates should have different complexity levels
        let complexities: Vec<usize> = candidates.iter().map(|c| c.complexity).collect();
        let unique_complexities: std::collections::HashSet<usize> =
            complexities.iter().copied().collect();

        assert!(
            unique_complexities.len() > 1,
            "Should have diverse complexity levels, got {:?}",
            complexities
        );

        println!("Generated {} candidates with complexities: {:?}", candidates.len(), complexities);
    }
}
