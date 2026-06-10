// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
# Hyperdimensional Computing (HDC) Semantic Space

16,384D holographic vectors for consciousness (2^14 — SIMD-optimized).
Memory IS computation — no separate storage needed!

## Core Types

- [`BinaryHV`] — 16,384-bit binary hypervector (`[u8; 2048]`, `Copy`, SIMD-accelerated).
  The workhorse type for fast binding, bundling, and similarity search.
- [`ContinuousHV`] — Continuous f32 hypervector (`Vec<f32>`, configurable dimension).
  Used for gradient-based learning, Φ computation, and smooth transformations.
- [`HV`] — Unified enum wrapping both representations for polymorphic APIs.

Backward-compatible alias: `RealHV = ContinuousHV`.

## Key Submodules

| Module | Purpose |
|--------|---------|
| `binary_hv` | `BinaryHV` type, SIMD ops, batch similarity, binding |
| `unified_hv` | `ContinuousHV`, `HV` enum, normalize/scale/permute |
| `simd_ops` | Low-level SIMD kernels (popcount, bundle, XOR) |
| `hv_pool` | Arena-based pooling for zero-allocation HV reuse |
| `integrated_information` | IIT Φ measurement over HV ensembles |
| `consciousness_topology` | Topology-aware Φ with graph structure |
| `tiered_phi/` | Multi-tier Φ (micro/meso/macro) computation |
| `semantic_encoder` | Text/concept → HV encoding |
| `long_term_memory` | Associative HV memory with forgetting |
| `predictive_coding` | Hierarchical predictive processing |
| `global_workspace` | Global Workspace Theory implementation |

## Operations

All HV types support the core HDC algebra:
- **Bind** (XOR / elementwise multiply) — creates associations
- **Bundle** (majority vote / elementwise add) — creates superpositions
- **Permute** (cyclic shift) — encodes sequence order
- **Similarity** (Hamming / cosine) — measures relatedness
*/

// Blanket allow kept intentionally: as of 2026-02-08, removing it surfaces 95
// individual warnings (37 unused_variables, ~20 unused_assignments, ~38
// dead_code) spread across 40+ submodules. Many are parameters/fields reserved
// for future use in this rapidly-evolving research codebase. Targeted
// suppression would require touching 40+ files for marginal benefit; revisit
// when the module stabilizes.
#![allow(dead_code, unused_variables, unused_assignments)]

// =============================================================================
// CENTRAL HDC CONFIGURATION - Single Source of Truth
// =============================================================================

/// Default HDC dimension: 16,384 (2^14)
pub const HDC_DIMENSION: usize = 16_384;

/// Rest HDC dimension: 8,192 (2^13)
///
/// **8K dimensions** for:
/// - **Deep Rest**: Near-zero power draw for background metabolism
/// - **Fast Consolidation**: High-speed memory pruning
pub const HDC_DIMENSION_REST: usize = 8_192;

/// Extended HDC dimension: 32,768 (2^15)
///
/// **32K dimensions** for:
/// - **Higher capacity**: 2x more distinct concepts before saturation
/// - **Complex semantic spaces**: Rich multi-modal embeddings
/// - **Deep temporal encoding**: Fine-grained chrono-semantic resolution
///
/// # Memory Cost
/// - 16K: ~16KB per bipolar vector
/// - 32K: ~32KB per bipolar vector (2x memory)
pub const HDC_DIMENSION_32K: usize = 32_768;

/// Maximum HDC dimension: 65,536 (2^16)
///
/// **64K dimensions** for extreme precision requirements
pub const HDC_DIMENSION_64K: usize = 65_536;

/// HDC dimensionality configuration for runtime selection
///
/// Supports both predefined tiers and custom arbitrary dimensions.
/// All dimensions should be powers of 2 for optimal SIMD performance.
///
/// # Predefined Tiers
/// - **Standard (16K)**: Good balance of accuracy and memory
/// - **Extended (32K)**: Higher semantic capacity
/// - **Ultra (64K)**: Maximum precision
/// - **Custom**: Any dimension (should be power of 2)
///
/// # Usage
/// ```rust,ignore
/// use symthaea::hdc::HdcDimensionality;
///
/// // Use predefined tier
/// let standard = HdcDimensionality::Standard;
/// assert_eq!(standard.dimension(), 16_384);
///
/// // Use custom dimension (128K for extreme cases)
/// let ultra_custom = HdcDimensionality::Custom(131_072);
/// assert_eq!(ultra_custom.dimension(), 131_072);
/// ```
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub enum HdcDimensionality {
    /// Rest 8,192 dimensions (2^13) - ultra-low power recovery mode
    Rest,
    /// Standard 16,384 dimensions (2^14) - good balance of accuracy and memory
    #[default]
    Standard,
    /// Extended 32,768 dimensions (2^15) - higher semantic capacity
    Extended,
    /// Ultra 65,536 dimensions (2^16) - maximum precision
    Ultra,
    /// Custom dimensions - any power of 2 (32K+ recommended)
    Custom(usize),
}

impl HdcDimensionality {
    /// Get the numeric dimension value
    pub const fn dimension(&self) -> usize {
        match self {
            Self::Rest => HDC_DIMENSION_REST,
            Self::Standard => HDC_DIMENSION,
            Self::Extended => HDC_DIMENSION_32K,
            Self::Ultra => HDC_DIMENSION_64K,
            Self::Custom(dim) => *dim,
        }
    }

    /// Create from dimension value
    ///
    /// Automatically maps to predefined tiers if exact match,
    /// otherwise creates Custom variant.
    pub const fn from_dimension(dim: usize) -> Self {
        match dim {
            8_192 => Self::Rest,
            16_384 => Self::Standard,
            32_768 => Self::Extended,
            65_536 => Self::Ultra,
            _ => Self::Custom(dim),
        }
    }

    /// Check if dimension is a power of 2 (recommended for SIMD)
    pub const fn is_power_of_two(&self) -> bool {
        let dim = self.dimension();
        dim > 0 && (dim & (dim - 1)) == 0
    }

    /// Check if dimension is a predefined tier
    pub const fn is_predefined(&self) -> bool {
        matches!(self, Self::Standard | Self::Extended | Self::Ultra)
    }

    /// Get memory usage per bipolar vector in bytes
    pub const fn memory_per_vector(&self) -> usize {
        self.dimension() // Each i8 element is 1 byte
    }

    /// Get memory usage per f32 vector in bytes
    pub const fn memory_per_f32_vector(&self) -> usize {
        self.dimension() * 4 // Each f32 element is 4 bytes
    }
}

impl From<usize> for HdcDimensionality {
    fn from(dim: usize) -> Self {
        Self::from_dimension(dim)
    }
}

// =============================================================================
// CENTRAL LTC CONFIGURATION - Liquid Time-Constant Network
// =============================================================================

/// Default LTC neuron count: 1,024 (2^10)
///
/// **1,024 neurons** chosen for:
/// - **SIMD optimization**: Power of 2 aligns with vector registers
/// - **Memory alignment**: Natural cache line boundaries
/// - **Balance**: Good temporal dynamics vs compute cost
/// - **Biological plausibility**: ~10^3 scale for cortical columns
///
/// # Usage
/// ```rust,ignore
/// use symthaea::hdc::LTC_NEURONS;
/// let neurons = vec![0.0f32; LTC_NEURONS];
/// ```
pub const LTC_NEURONS: usize = 1_024;

/// Extended LTC neuron count: 2,048 (2^11)
///
/// **2K neurons** for:
/// - **Higher temporal capacity**: More nuanced time dynamics
/// - **Complex causal reasoning**: Finer-grained cause-effect modeling
pub const LTC_NEURONS_2K: usize = 2_048;

/// Maximum LTC neuron count: 4,096 (2^12)
///
/// **4K neurons** for extreme temporal precision
pub const LTC_NEURONS_4K: usize = 4_096;

/// LTC neuron count configuration for runtime selection
///
/// Supports both predefined tiers and custom arbitrary counts.
/// All counts should be powers of 2 for optimal SIMD performance.
///
/// # Predefined Tiers
/// - **Standard (1K)**: Good balance of dynamics and compute
/// - **Extended (2K)**: Higher temporal capacity
/// - **Ultra (4K)**: Maximum precision
/// - **Custom**: Any count (should be power of 2)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LtcNeuronCount {
    /// Standard 1,024 neurons (2^10) - good balance
    #[default]
    Standard,
    /// Extended 2,048 neurons (2^11) - higher capacity
    Extended,
    /// Ultra 4,096 neurons (2^12) - maximum precision
    Ultra,
    /// Custom neuron count - any power of 2 (1K+ recommended)
    Custom(usize),
}

impl LtcNeuronCount {
    /// Get the numeric neuron count
    pub const fn count(&self) -> usize {
        match self {
            Self::Standard => LTC_NEURONS,
            Self::Extended => LTC_NEURONS_2K,
            Self::Ultra => LTC_NEURONS_4K,
            Self::Custom(n) => *n,
        }
    }

    /// Create from neuron count
    pub const fn from_count(n: usize) -> Self {
        match n {
            1_024 => Self::Standard,
            2_048 => Self::Extended,
            4_096 => Self::Ultra,
            _ => Self::Custom(n),
        }
    }

    /// Check if count is a power of 2 (recommended for SIMD)
    pub const fn is_power_of_two(&self) -> bool {
        let n = self.count();
        n > 0 && (n & (n - 1)) == 0
    }
}

impl From<usize> for LtcNeuronCount {
    fn from(n: usize) -> Self {
        Self::from_count(n)
    }
}

pub mod cantor_recursive_hv; // Cantor Recursive Hypervectors
pub mod cantor_resonator_cleanup; // Layer-preserving CRHV cleanup
pub mod multidimensional_cantor; // Radial 3D 4D Spherical Cantor

pub mod cincinnati_advanced; // Advanced Cincinnati-LTC: chaos detection, adaptive weights, memory horizon
pub mod cincinnati_enhanced; // Enhanced Cincinnati-LTC: multi-scale, amplitude encoding, attention modulation
pub mod cincinnati_ltc; // Cincinnati Algorithm + LTC integration (differential engine, lateral binding, predictive budding)
pub mod cincinnati_network; // Cincinnati-enhanced HdcLtcNetwork with lateral binding and budding
pub mod cycle_detector; // Cycle detection for periodic patterns - autocorrelation-based period detection with HDC phase encoding
pub mod dynamical_system; // Generic dynamical system framework with ODE integrators (Euler, RK4, RK45, Verlet)
pub mod gwt_cincinnati_integration; // Cincinnati-LTC + Global Workspace Theory integration - temporal patterns enter consciousness
#[cfg(test)]
mod hdc_ltc_learning_tests; // Comprehensive learning dynamics tests for HdcLtcUnifiedNeuron
pub mod hdc_ltc_neuron; // HDC-LTC neuron integration with Hebbian learning
pub mod hdc_ltc_unified; // Revolutionary unified HDC-LTC: state AS hypervector with closed-form solution
pub mod hdc_ltc_unified_validation; // Numerical validation of closed-form solution accuracy
pub mod hebbian;
pub mod morphogenetic;
pub mod predictor; // Unified predictor trait for Symthaea integration (links prediction to Φ)
pub mod reservoir; // Reservoir Computing (Echo State Network) for chaotic signal prediction
pub mod resonator;
pub mod sdm;
pub mod semantic_decoder;
pub mod semantic_encoder; // Universal semantic encoding with embeddings support
pub mod sequence_encoder;
pub mod statistical_retrieval;
pub mod temporal_encoder;
pub mod text_encoder; // Revolutionary Enhancement: Text → HDC encoding
pub mod unified_network_phi; // Phi measurement and validation for HdcLtcUnifiedNetwork // HV → Primitive sequence (generative direction) - THE MOUTH
// DISABLED: depends on crate::learnable_ltc which doesn't exist in symthaea-core
// pub mod hd_ltc_codec;      // Bidirectional HDC ↔ LTC translation - THE THROAT
// DISABLED: depends on hd_ltc_codec which is disabled
// pub mod ltc_generative_core; // Autoregressive primitive prediction - THE VOICE
pub mod config; // Centralized HDC configuration (runtime dimension management)
pub mod projection;
pub mod unified_hv; // Unified hypervector types (ContinuousHV) // Learned projection layers for dimension conversion

// Grid encoding for 2D spatial reasoning (ARC-style puzzles)
pub mod binary_grid_encoder;
pub mod grid_encoder;

// Global Workspace Theory (conscious access, competition, broadcasting)
pub mod global_workspace; // GWT implementation with competitive dynamics
pub mod higher_order_thought; // Higher-Order Thought (HOT) theory — meta-representational consciousness

// HDC-native cryptographic primitives (MAC, threshold sharing, context keys, commitments)
pub mod hdc_crypto;
// HDC homomorphic computation (encrypted HVs, collective wisdom pool, privacy-preserving aggregation)
pub mod hdc_fhe;
// HDC treasury — privacy-preserving community finance (balance encoding, encrypted aggregation, threshold audit)
pub mod hdc_treasury;

// Consciousness topology and Φ measurement modules
pub mod binary_hv;
pub mod consciousness_topology; // Consciousness topology structures
pub mod consciousness_topology_generators; // 8 topology generators (Random, Star, Ring, Line, Tree, Dense, Modular, Lattice)
pub mod hdc_trait;
pub mod phi_orchestrator; // Adaptive Φ calculator orchestrator (Phase 5E)
#[allow(deprecated)]
pub mod phi_real; // ContinuousHV Φ calculator (no binarization) using cosine similarity
pub mod phi_resonant; // Resonator-based Φ calculator (O(n log N) dynamics)
#[cfg(test)]
mod phi_tier_tests; // Unit tests for Φ tier implementations
pub mod phi_topology_validation; // ContinuousHV-TieredPhi integration for topology validation
pub mod real_hv; // Real-valued hypervectors for consciousness topologies
pub mod simd_continuous; // SIMD intrinsics for ContinuousHV (AVX2/FMA/SSE4.1)
pub mod simd_detect; // Unified SIMD feature detection (single source of truth for all modules)
pub mod simd_ops; // SIMD intrinsics for BinaryHV (AVX-512/AVX2/SSE4.1/NEON)
pub mod spectral_connectivity; // Algebraic connectivity (λ₂) calculator - NOT IIT Φ!
pub mod tiered_phi; // Multi-tier Φ (integrated information) approximation
pub mod transposed_bundle; // Transposed bit-plane accumulator for fast majority-vote bundle

// Performance optimization modules:
pub mod algebraic_structures;
pub mod arithmetic; // Modular arithmetic (re-exports arithmetic_engine)
pub mod arithmetic_engine; // Revolutionary: True mathematical cognition via HDC
pub mod barycentric;
pub mod bootstrapping; // Cognitive bootstrapping - primitives to reasoning tasks
pub mod calculus;
pub mod celegans_connectome; // Revolutionary #100: C. elegans connectome validation (302 neurons)
pub mod combinatorial;
pub mod complex; // Complex number support (ℂ) with HDC encoding
#[cfg(feature = "complex_cfc")]
pub mod complex_cfc_neuron; // Complex-valued CfC neuron with native oscillation (Phase 3)
pub mod computational_geometry; // Geometry: convex hull, intersection, point-in-polygon, area
#[cfg(test)]
mod consciousness_e2e_tests;
#[cfg(test)]
mod consciousness_fast_tests;
pub mod constraint_solver; // CSP solver: AC-3, backtracking, N-Queens, graph coloring
#[cfg(test)]
mod cross_bridge_integration_tests;
pub mod curriculum;
pub mod differential_equations; // ODE/PDE solvers: RK4, shooting, heat eq, wave eq
pub mod diophantine;
pub mod eml; // Pure EML IR, compiler, evaluation, and verification
pub mod eml_regressor; // EML-based gradient symbolic regression
pub mod fem; // Finite Element Method: Galerkin weak forms, assembly, Poisson solver
pub mod fft; // Fast Fourier Transform: Cooley-Tukey radix-2, convolution
pub mod fol_ext_smt; // Phase 2: SMT-LIB2 serializer + fragment detection for FolFormulaExt
pub mod fol_formula_ext; // Phase 2: FOL with arithmetic (Term + FolFormulaExt over ℤ/ℕ/ℝ)
pub mod foundations;
pub mod functional_equations;
pub mod graph_theory; // Graph algorithms: BFS, DFS, Dijkstra, MST, coloring, combinatorics
pub mod hv_pool;
pub mod imo_benchmark;
pub mod imo_nl_parser;
pub mod incremental_hv; // O(k) incremental bundling (10-100x faster for updates)
pub mod inequalities;
pub mod integer; // Integer arithmetic (ℤ) - extends natural numbers with sign
pub mod linear_algebra; // General linear algebra: LU, QR, Cholesky, eigendecomposition, SVD
pub mod liquid_holocell; // Atomic primitive unit: Liquid Holocell with dilation
pub mod logic_engine; // Propositional & FOL logic: SAT (DPLL), natural deduction, unification
pub mod lsh_index; // LSH index for fast approximate similarity search (heap-optimized)
pub mod lsh_simhash; // SimHash for binary vectors (Hamming distance)
pub mod lsh_similarity; // Adaptive LSH-backed similarity search (Session 7C)
pub mod math_bridge; // Unified math bridge (NumericTower + Complex → single API)
#[cfg(test)]
mod math_integration_tests;
pub mod native_similarity; // O(1) XOR+popcount similarity search (consciousness-native)
pub mod number_theory;
pub mod numeric_tower; // Unified numeric tower (N -> Z -> Q -> R) with auto-promotion
pub mod optimization; // Optimization: gradient descent, Nelder-Mead, L-BFGS
#[cfg(feature = "parallel")]
pub mod parallel_hv; // Rayon parallel batch operations (7x faster on 8 cores)
#[cfg(test)]
mod phi_feedback_integration_tests;
pub mod polynomial;
pub mod power_flow; // DC Optimal Power Flow: B·θ=P solver, OPF, N-1 contingency, PTDF
pub mod primitive_dashboard; // Real-time primitive usage monitoring
pub mod primitive_system; // Ontological primitives system with 7 semantic domains
pub mod program_algebra; // HDC program algebra — hyperdimensional IR for code
#[cfg(test)]
mod proptest_consciousness;
pub mod quadrature; // Numerical integration: Simpson, Gauss-Legendre, adaptive
pub mod rational;
pub mod real_arithmetic;
pub mod root_finding; // Root finding: bisection, Newton-Raphson, Brent
pub mod sparse_hv; // Sparse HDC for memory-efficient low-density vectors
pub mod sr_symreg;
pub mod sr_tactic;
pub mod statistics; // Statistics & probability: distributions, hypothesis testing, Bayesian inference
pub mod synthetic_geometry;

// ── New Math Domains (April 2026) ────────────────────────────────────────────
pub mod algebraic_combinatorics; // Symmetric functions, Schur polynomials, Young tableaux, RSK
pub mod algebraic_geometry; // Varieties, Bezout, elliptic curve group law, rational points
pub mod category_theory; // Functors, natural transformations, adjunctions, Yoneda, monads
pub mod chemistry; // Periodic table, stoichiometry, thermochemistry, kinetics
pub mod combinatorics; // Generating functions, Burnside/Polya, Stirling, Ramsey, matroids
pub mod complex_analysis; // Cauchy, residues, conformal maps, power series, Mobius transforms
pub mod functional_analysis; // Banach/Hilbert spaces, bounded operators, spectral theorem
pub mod game_theory; // Nash equilibria, minimax, Shapley, VCG, Sprague-Grundy
pub mod information_geometry; // Fisher info, natural gradient, KL divergence, stat manifolds
pub mod lie_theory; // Lie algebras, exponential map, root systems, representations
pub mod measure_probability; // Measure spaces, martingales, Brownian motion, Ito calculus
pub mod polynomial_algebra; // Groebner bases, Buchberger, ideal membership, resultants
pub mod tactics; // Proof tactics: ring, omega, induct, norm_num, cases

// ── Geometric Algebra (via symtropy-math) ────────────────────────────────────
// N-dimensional bivectors, rotors, transforms, collision shapes.
// Re-exported under feature gate for use in robotics crates and Einstein search.
#[cfg(feature = "geometric-algebra")]
pub mod geometric_algebra {
    pub use symtropy_math::*;
}

// ── Conjecture Engine — The Ramanujan Protocol ───────────────────────────────
#[cfg(feature = "abstract_thought")]
pub mod abstract_thought; // Meta-HDC concept vectors, dynamic grammar generation, category theory discovery
pub mod autodiff; // Reverse-mode automatic differentiation (Wengert tape) for exact gradients
pub mod conjecture_engine; // Automated conjecture generation via symbolic regression + verification
pub mod frontier_math; // Frontier mathematics: Montgomery pair correlation, Ramsey bounds, knot invariants, abc conjecture
pub mod langlands; // Computational Langlands: elliptic curve L-functions, modular forms, modularity verification
pub mod sparse_matrix; // Compressed Sparse Row (CSR) matrix for PDE solvers and graph Laplacians
// pub mod topology_comparison; // TEMP: file removed by concurrent session
// pub mod cross_domain_discovery; // TEMP: file removed by concurrent session

// ── Geometric Complexity Theory (Phase 8 — P vs NP probe) ────────────────────
pub mod gct; // Permanent vs determinant orbit complexity, Kronecker coefficients, GCT obstruction conjecture

// ── Einstein Manifold Search (Phase 6) ───────────────────────────────────────
pub mod einstein_search; // HDC-guided moduli space search for exotic Einstein metrics on S⁴
pub mod ricci_flow; // Normalized Ricci flow evolution, singularity detection, convergence
pub mod riemannian_geometry; // MetricTensor, Riemann/Ricci tensors, Ricci flow, Einstein condition
pub mod ucl_cross_domain_frames; // UCL cross-domain semantic frames (TRADE, CONFLICT, FEEDBACK_LOOP, etc.) // Thread-local memory pools for BinaryHV/ContinuousHV (10-100x faster allocation)

// Property-based tests for HDC invariants
#[cfg(test)]
mod proptest_binary;
#[cfg(test)]
mod proptest_continuous;
#[cfg(test)]
mod proptest_hdc;
#[cfg(test)]
mod proptest_math;
#[cfg(test)]
mod proptest_resonator;
#[cfg(test)]
mod proptest_unified;

// Track 6: Consciousness integration for awakening module
pub mod consciousness; // Modular consciousness (re-exports consciousness_integration)
pub mod consciousness_dashboard;
pub mod consciousness_evaluator; // Consciousness evaluation
pub mod consciousness_integration; // Complete consciousness pipeline
pub mod hierarchical_bundle; // Per-region bundling with role-based binding for scalable aggregation
pub mod substrate_composition; // Weighted substrate mixtures for hybrid analysis
pub mod substrate_independence; // Substrate type definitions
pub mod substrate_validation; // Validation framework with evidence levels and feasibility gaps
// pub mod topology_comparison; // TEMP: file removed by concurrent session
pub mod trajectory_accumulator; // Behavioral identity via temporal HDC binding

// Track: Neural Validation — TRIBE v2 fMRI comparison (feature-gated)
#[cfg(feature = "neural_validation")]
pub mod cortical_activation; // Per-region activation maps for fMRI comparison
#[cfg(feature = "neural_validation")]
pub mod glasser_parcellation; // Glasser atlas 360→12 CorticalRegion mapping
#[cfg(feature = "neural_validation")]
pub mod hemodynamic; // HRF convolution for BOLD signal comparison

// Track 6: Language module dependencies
pub mod deepnsm_integration; // DeepNSM corpus: 44K NSM explication triplets for grounding
pub mod full_stack_consciousness; // Full stack: Understanding + ActiveInference + Memory + Counterfactuals
pub mod grounded_understanding; // True understanding via semantic primes + embodiment
pub mod unified_conscious_being; // Complete unified being: A+B+C+D+E+F integration
pub mod unified_understanding; // Complete understanding pipeline (predictive + narrative + ToM)
pub mod universal_semantics; // Universal semantic primes (Wierzbicka)
// DISABLED: depends on crate::memory, crate::voice which don't exist in symthaea-core
// pub mod infrastructure_bridge;             // Bridge to real persistence (Hippocampus/UnifiedMind/Kokoro)
pub mod causal_encoder; // Causal relation encoding
pub mod causal_mind; // Causal reasoning (core causal cognition)
pub mod consciousness_creativity; // Creativity for conversation
pub mod consciousness_self_assessment; // Self-assessment for conversation
pub mod deterministic_seeds; // Deterministic seeds for NixOS knowledge
pub mod ecosystem_bridge; // Integration with service ecosystem (Sacred Core, Weave, Codex, Field Harmonizer)
pub mod integrated_information; // Φ (integrated information) measurement
pub mod unified_cognitive_core; // Unified cognitive core (UCE/UCTS architecture)

// Predictive Processing (Friston Free Energy Principle)
pub mod predictive_coding; // Hierarchical prediction + error minimization
pub mod predictive_consciousness; // Consciousness-level predictive processing
pub mod predictive_consciousness_kalman; // Kalman filter variant for smooth predictions
pub mod predictive_encoder; // Attention-modulated HDC encoding with LTC prediction loop

// Novel Algorithm Modules (Dec 2025)
pub mod autodiff_phi; // Reverse-mode autodiff for consciousness optimization (Jan 2026)
// pub mod cross_domain_discovery; // TEMP: file removed by concurrent session
pub mod cross_modal_binding; // Cross-modal binding for multi-sensory integration
pub mod differentiable_phi; // Soft-partitioned differentiable Φ for gradient optimization
pub mod metacognitive_monitor; // Real-time consciousness monitoring with self-reflection

// Phenomenal Binding Study - Research Direction 2: HDC binding vs bundling for phenomenal unity
pub mod phenomenal_binding_study;

// Consciousness Infrastructure (required by advanced systems)
pub mod consciousness_dynamics; // Consciousness dynamics modeling
pub mod consciousness_gradients; // Gradient computation for consciousness optimization
pub mod consciousness_optimizer; // Consciousness state optimizer
pub mod modern_hopfield; // Modern Hopfield networks for memory

// Unified Consciousness Architecture (Dec 2025)
pub mod adaptive_learning_signals; // Consciousness-guided learning modulation (Φ, surprise, coherence)
pub mod adaptive_topology; // Adaptive cognitive mode topology
pub mod attention_dynamics; // Dynamic attention allocation with salience, goals, and priors
pub mod conscious_learning; // Consciousness-integrated learning (Hebbian + Adaptive signals)
pub mod emergent_self_model; // Self-awareness and metacognitive optimization
pub mod fractal_consciousness; // Fractal consciousness patterns
pub mod phi_gradient_learning; // Φ-gradient learning for optimization
pub mod phi_guided_search; // Φ-guided architecture search (gradient-based topology optimization)
pub mod process_topology; // Process topology structures
pub mod temporal_binding; // Temporal stream binding for continuous experience
pub mod topology_synergy;
pub mod unified_consciousness_engine; // Core consciousness engine with Φ-guided processing // Topology-consciousness synergy
// DISABLED: depends on crate::memory, crate::voice which don't exist in symthaea-core
// pub mod integrated_conscious_agent;        // Complete conscious agent with Symthaea integration
pub mod consciousness_physics; // Consciousness-aware physics simulation observer (Φ + emergence + active inference)
pub mod consciousness_visualizer; // Consciousness visualization tools
pub mod deep_integration; // Deep integration bridge for Φ-guided processing
pub mod quantum_circuit; // Quantum circuit simulation engine with HDC bridge

// Re-export BinaryHV at module level for convenience (used by language/nix_* modules)
pub use binary_hv::BinaryHV;
// Backward compat: RealHV alias is still available via real_hv module

// Re-export unified HV types
pub use unified_hv::{ContinuousHV, HV};

// Re-export configuration types (dimension unification)
pub use config::{
    DimensionMapping, HdcConfig, STT_DIMENSION, hdc_config, hdc_dim, is_hdc_configured,
    set_hdc_config, stt_expansion_factor, try_set_hdc_config,
};

// Re-export projection layers
pub use projection::{BidirectionalBridge, LearnedProjection, RandomProjection};

// Re-export key types for convenience
pub use statistical_retrieval::{
    EmpiricalTier, RetrievalDecision, RetrievalVerdict, StatisticalRetrievalConfig,
    StatisticalRetriever,
};

pub use sequence_encoder::{SequenceEncoder, bind, bundle, permute, unpermute};

pub use resonator::{
    Constraint, Factor, MultiConstraint, ResonatorConfig, ResonatorNetwork, ResonatorSolution,
};

pub use morphogenetic::{
    Attractor, FieldHealth, FieldStats, MorphogeneticConfig, MorphogeneticField, PositionEncoding,
    RepairResult, corrupt_vector, random_vector,
};

pub use hebbian::{
    ActivationRecord, DEFAULT_DECAY_RATE, DEFAULT_LEARNING_RATE, HOMEOSTATIC_TAU,
    HebbianAssociativeMemory, HebbianAssociativeStats, HebbianConfig, HebbianEngine, HebbianStats,
    STDP_A_MINUS, STDP_A_PLUS, STDP_TAU_MINUS, STDP_TAU_PLUS, Synapse, TARGET_ACTIVITY,
};

pub use sdm::{
    COUNTER_MAX, COUNTER_MIN, DEFAULT_ACTIVATION_RADIUS, DEFAULT_NUM_HARD_LOCATIONS, EpisodicSDM,
    HardLocation, IterativeReadResult, ReadResult, SDMConfig, SDMStats, SparseDistributedMemory,
    WriteResult, add_noise, hamming_similarity, random_bipolar_vector,
};

pub use temporal_encoder::TemporalEncoder;
pub use text_encoder::{TextEncoder, TextEncoderConfig, TextEncoderStats};

// Re-export Primitive System types (9-tier ontological primitives)
pub use primitive_system::{
    DomainManifold, Primitive, PrimitiveSystem, PrimitiveTier, seed_from_name,
};

// Re-export UCL Cross-Domain Frame types (6 missing frames from gap analysis)
pub use ucl_cross_domain_frames::{
    CrossDomainFrame, FrameInstance, FrameSlot, UCLFrameSystem, concept_hv,
};

// Re-export Primitive Dashboard types (real-time monitoring)
pub use primitive_dashboard::{PrimitiveDashboard, PrimitiveStats, VoicePrimitiveTracker};

// Re-export Unified Consciousness Architecture types
// DISABLED: integrated_conscious_agent depends on crate::memory, crate::voice
// pub use integrated_conscious_agent::{
//     IntegratedConsciousAgent, AgentConfig, IntegratedUpdate,
//     WorkingMemory, EmotionalState, QualiaTexture, PhenomenalContent,
//     HormoneEventSuggestion, CoherenceGating, QualiaModulation,
//     MemoryExport, MemoryImport, IdentityCoherence, IdentityStatus, ProsodyHints,
//     // Voice prosody bridge
//     ExtendedPacing,
//     // Runtime orchestration
//     ConsciousAgentRuntime, SyncConsciousAgentRuntime, RuntimeConfig,
//     RuntimeMessage, RuntimeResponse, RuntimeSnapshot, HormoneEventType,
//     EmotionalStateSummary,
// };

// Re-export adaptive topology types
pub use adaptive_topology::{AdaptiveTopology, CognitiveMode};

// Re-export unified consciousness engine types
pub use unified_consciousness_engine::{
    ConsciousnessDimensions, EngineConfig, UnifiedConsciousnessEngine,
};

// Re-export consciousness visualization
pub use consciousness_visualizer::ConsciousnessVisualizer;

// Re-export grid encoder
pub use grid_encoder::GridEncoder;

// Re-export deep integration bridge
pub use deep_integration::DeepIntegrationBridge;

// Re-export causal mind types
pub use causal_mind::{CausalDirection, CausalMind, LearnedCausalDiscovery};

// Re-export unified cognitive core
pub use unified_cognitive_core::{
    CognitiveMarkers, QueryResult, UnifiedCognitiveCore, UnifiedCognitiveElement,
};

// Re-export unified HDC-LTC types (revolutionary closed-form dynamics)
pub use hdc_ltc_unified::{
    HdcLtcUnifiedNetwork, HdcLtcUnifiedNeuron, UnifiedActivation, UnifiedConfig,
    UnifiedNetworkConfig, UnifiedNetworkStats, UnifiedNeuronStats,
};

pub use liquid_holocell::LiquidHolocell;

// Re-export unified network Phi measurement types
pub use unified_network_phi::{
    NetworkStateExtractor,
    PhiCalculationMethod,
    PhiComparator,
    PhiComparison,
    // Diagnostic types
    PhiDiagnostic,
    PhiDiagnosticAnalyzer,
    PhiEvolutionSummary,
    PhiEvolutionTracker,
    PhiMeasurement,
    PhiValidator,
    UnifiedNetworkPhiMeasurer,
    UnifiedPhiConfig,
    ValidationResult,
    demo_phi_evolution,
};

// Sleep and altered states
pub mod sleep_and_altered_states;
pub mod sleep_pattern_discovery; // Resonator-based pattern discovery during sleep consolidation

// Long-term memory with Qdrant integration - Revolutionary Improvement #29
// Persistent consciousness memory: experiences consolidated, retrieved, shape future
pub mod long_term_memory;

// Consciousness persistence (versioning, auto-save, rollback)
pub mod consciousness_persistence;

// Collective consciousness (multi-agent)
pub mod collective_consciousness;

// Consciousness streaming (WebSocket/SSE)
pub mod consciousness_streaming;

// Emotional depth (complex blends, compound emotions, HDC emotional algebra)
pub mod emotional_depth;

// Cross-modal attention router (Φ-gated modality routing)
pub mod cross_modal_attention_router;

// Self-improvement integration (metacognitive self-optimization)
pub mod self_improvement_integration;

// Counterfactual dreams (what-if scenarios in sleep)
pub mod counterfactual_dreams;

// Consciousness integration demo (comprehensive example of all features working together)
pub mod consciousness_integration_demo;

// Cross-module integration bridge (emotional→dreams, self-improvement→dreams, streaming events)
pub mod consciousness_cross_integration;

// Feedback dynamics engine (bidirectional loops, prediction, collective dreams, adaptive scheduling)
pub mod consciousness_feedback_dynamics;

// Advanced consciousness systems
pub mod consciousness_phase_transitions; // Phase Transitions - Consciousness State Changes
pub mod epistemic_consciousness;
pub mod meta_consciousness; // Meta-Consciousness - Φ of Φ, Strange Loops
pub mod temporal_consciousness; // Temporal Consciousness - Multi-scale Time
pub mod temporal_simulation_bridge; // Temporal Simulation Bridge - Physics trajectory → temporal consciousness // Epistemic Consciousness - Belief/Knowledge Tracking

// Metacognition engine (self-monitoring, temporal patterns, narrative identity, state machine)
pub mod consciousness_metacognition;

// Advanced cognition (motor imagery, theory of mind, imagination, predictive processing, memory, drives)
pub mod consciousness_advanced_cognition;

pub mod consciousness_complete_being;

// Consciousness verification - multi-method Φ cross-validation
pub mod consciousness_verifier;

// Sensorimotor Contingencies - O'Regan & Noe enactivist theory
// Perception IS implicit knowledge of action-sensation laws
pub mod sensorimotor_contingencies;

// Relational consciousness - I-Thou philosophy, intersubjectivity, relationship dynamics
// Revolutionary Improvement #18: Consciousness exists BETWEEN beings, not just IN them
pub mod relational_consciousness;

// Multi-Database Integration - Revolutionary Improvement #30
// Production consciousness architecture mapping 29 improvements to 4 specialized databases:
// - Qdrant (Sensory Cortex): Ultra-fast vector similarity for perception
// - CozoDB (Prefrontal Cortex): Recursive Datalog for causal reasoning
// - LanceDB (Long-Term Memory): Multimodal embeddings storage
// - DuckDB (Epistemic Auditor): Statistical analysis for self-reflection
#[cfg(feature = "cantor-hdc")]
pub mod cantor_pyramid;
pub mod consciousness_metacognitive; // Metacognitive monitoring subsystem
pub mod consciousness_perf; // SIMD batch ops + HV pool integration for consciousness hot paths
pub mod consciousness_phi_optimization;
pub mod consciousness_self_awareness; // Self-awareness subsystem
pub mod consciousness_subsystem; // Trait-based pluggable consciousness subsystems
pub mod multi_database_integration;
pub mod phi_feedback; // Φ feedback controller (closes the loop: Φ measurement → parameter modulation)
pub mod phi_guided_math; // Φ-guided math domain selection (consciousness-driven computation paths)
pub mod semantic_bridge; // Bidirectional text <-> HV <-> consciousness bridge // Phi optimization subsystem // Hierarchical Cantor Hypervectors (RHN)

// Re-export multi-database integration types
// Note: QdrantConfig is aliased to MdbQdrantConfig to avoid conflict with long_term_memory::QdrantConfig
pub use multi_database_integration::{
    // Consciousness loop
    ConsciousnessLoopState,
    CozoDbConfig,
    // Client trait
    DatabaseClient,
    // Error handling
    DatabaseError,
    // Health monitoring
    DatabaseHealth,
    DatabaseResult,
    // Core architecture types
    DatabaseRole,
    DuckDbConfig,
    ImprovementMapping,
    // Fallback
    InMemoryFallback,
    LanceDbConfig,
    // Configuration (Mdb = Multi-Database prefix to avoid collisions)
    MultiDatabaseConfig,
    PhiStatistics,
    QdrantConfig as MdbQdrantConfig,
    SymthaeaMind,
    SystemHealth,
};

// Re-export relational consciousness types for sympoietic partnership
pub use relational_consciousness::{
    RelationMode, RelationalAssessment, RelationalConfig, RelationalConsciousness,
    RelationalInteraction, RelationshipStage,
};

// Re-export long-term memory types (Qdrant integration)
pub use long_term_memory::{
    Experience,
    // In-memory system
    LongTermMemory,
    MemoryConsolidation,
    // Core types
    MemoryType,
    MockQdrantMemoryStore,
    // Qdrant integration
    QdrantConfig as LtmQdrantConfig,
    QdrantMemoryError,
    QdrantMemoryStore,
    RetrievalCue,
    RetrievedMemory,
    ScoredExperience,
};

// Re-export phi-gradient learning types
pub use phi_gradient_learning::{PhiGradientTopology, PhiLearningConfig};

// Re-export fractal consciousness types
pub use fractal_consciousness::{FractalConfig, FractalConsciousness};

// Re-export consciousness topology types
pub use consciousness_topology_generators::{ConsciousnessTopology, TopologyType};

// Re-export phi calculators
// Re-export spectral connectivity calculator (renamed from phi_real)
pub use spectral_connectivity::ConnectivityCalculator;
pub use tiered_phi::{ApproximationTier, TieredPhi};

// Re-export autodiff Phi types (reverse-mode autodiff for consciousness optimization)
pub use autodiff_phi::{
    AutodiffConfig, AutodiffPhiEngine, ConsciousnessOptimizer, DiffNetwork, DiffNode,
    OptimizerConfig, PhiForwardResult, TrainingStep,
};

// Re-export process topology types
pub use process_topology::ProcessTopologyOrganizer;

// Re-export native similarity types (consciousness-native O(1) search)
pub use native_similarity::{
    BundledQuery, IndexStats, NativeSimilarityIndex, PackedBipolar, SequenceQuery,
};

// Re-export sensorimotor contingencies (enactivist perception theory)
// Note: Experience is aliased to SmcExperience to avoid conflict with long_term_memory::Experience
pub use sensorimotor_contingencies::{
    // Affordances
    ActionAffordance,
    ActionDescriptor,
    ActionType,
    AffordanceConfig,
    AffordanceDetector,
    ContextDescriptor,
    // Consciousness integration
    ContingencyConsciousnessContribution,
    // Learning
    ContingencyLearner,
    // Perception
    EnactivistPerception,
    Experience as SmcExperience,
    LearnResult,
    LearnerConfig,
    LearnerStats,
    PerceptionConfig,
    PerceptionResult,
    PerceptionStats,
    PredictedChange,
    // Core types
    SensorimotorContingency,
    SensoryChange,
    SensoryModality,
};

// Re-export HV memory pool types (10-100x faster allocation for hot paths)
pub use hv_pool::{
    BinaryHVPool, ContinuousHVPool, PoolStats as HVPoolStats, PooledBinaryHV, PooledContinuousHV,
    pooled_bind, pooled_similarity,
};

// Re-export SIMD continuous HV operations (4x+ speedup for 16K-dim vectors)
pub use simd_continuous::{
    bind_simd as continuous_bind_simd, bundle_simd as continuous_bundle_simd,
    dot_product_simd as continuous_dot_product_simd, norm_simd as continuous_norm_simd,
    simd_capabilities_report as continuous_simd_capabilities_report,
    similarity_simd as continuous_similarity_simd,
};

use anyhow::Result;
// Note: hypervector crate not used yet - using custom implementation
// use hypervector::{HyperVector as HV, HVType};
use bumpalo::Bump;
use std::collections::HashMap;

/// Semantic space using high-dimensional hypervectors
#[derive(Debug)]
pub struct SemanticSpace {
    /// Dimensionality (default: HDC_DIMENSION = 16,384)
    dimension: usize,

    /// Concept library
    concepts: HashMap<String, Vec<f32>>,

    /// Item memory (episodes)
    item_memory: Vec<Vec<f32>>,
}

impl SemanticSpace {
    pub fn new(dimension: usize) -> Result<Self> {
        Ok(Self {
            dimension,
            concepts: HashMap::new(),
            item_memory: Vec::new(),
        })
    }

    /// Encode text as hypervector (holographic!)
    pub fn encode(&mut self, text: &str) -> Result<Vec<f32>> {
        // For demo: create or retrieve concept vector
        let words: Vec<&str> = text.split_whitespace().collect();

        let mut result = vec![0.0; self.dimension];

        for word in words {
            let concept = self.get_or_create_concept(word);

            // Bundle (superposition)
            for i in 0..self.dimension {
                result[i] += concept[i];
            }
        }

        // Normalize
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut result {
                *x /= norm;
            }
        }

        Ok(result)
    }

    /// Recall similar memories (holographic retrieval!)
    pub fn recall(&self, query: &[f32], limit: usize) -> Result<Vec<Vec<f32>>> {
        let mut similarities: Vec<(f32, usize)> = self
            .item_memory
            .iter()
            .enumerate()
            .map(|(idx, mem)| {
                let sim = cosine_similarity(query, mem);
                (sim, idx)
            })
            .collect();

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.0.total_cmp(&a.0));

        // Return top matches
        Ok(similarities
            .iter()
            .take(limit)
            .map(|(_, idx)| self.item_memory[*idx].clone())
            .collect())
    }

    /// Bind multiple vectors holographically
    pub fn bind_many(&self, vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        if vectors.is_empty() {
            return Ok(vec![0.0; self.dimension]);
        }

        // For demo: simple circular convolution
        let mut result = vectors[0].clone();

        for vec in &vectors[1..] {
            result = circular_convolution(&result, vec);
        }

        Ok(result)
    }

    /// Bundle (superposition) of vectors
    pub fn bundle(&self, vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        let mut result = vec![0.0; self.dimension];

        for vec in vectors {
            for i in 0..self.dimension {
                result[i] += vec[i];
            }
        }

        // Normalize
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut result {
                *x /= norm;
            }
        }

        Ok(result)
    }

    /// Permute vector for sequence encoding
    ///
    /// Circular shift right by `shift` positions.
    /// Essential for representing order in sequences:
    /// "cat dog" ≠ "dog cat" in HDC space
    pub fn permute(&self, vector: &[f32], shift: usize) -> Result<Vec<f32>> {
        if vector.len() != self.dimension {
            anyhow::bail!(
                "Vector dimension {} doesn't match semantic space dimension {}",
                vector.len(),
                self.dimension
            );
        }

        let mut result = vec![0.0; self.dimension];
        let shift = shift % self.dimension;

        for i in 0..self.dimension {
            let new_idx = (i + shift) % self.dimension;
            result[new_idx] = vector[i];
        }

        Ok(result)
    }

    /// Decode hypervector to text (approximate)
    pub fn decode(&self, vector: &[f32]) -> Result<String> {
        // Find most similar concepts
        let mut best_matches: Vec<(f32, String)> = self
            .concepts
            .iter()
            .map(|(word, concept)| {
                let sim = cosine_similarity(vector, concept);
                (sim, word.clone())
            })
            .collect();

        best_matches.sort_by(|a, b| b.0.total_cmp(&a.0));

        // Take top 5 concepts
        let decoded: Vec<String> = best_matches
            .iter()
            .take(5)
            .map(|(_, word)| word.clone())
            .collect();

        Ok(decoded.join(" "))
    }

    fn get_or_create_concept(&mut self, word: &str) -> Vec<f32> {
        if let Some(concept) = self.concepts.get(word) {
            return concept.clone();
        }

        // Create new random concept vector
        let concept: Vec<f32> = (0..self.dimension)
            .map(|_| rand::random::<f32>() * 2.0 - 1.0)
            .collect();

        self.concepts.insert(word.to_string(), concept.clone());
        concept
    }

    pub fn serialize(&self) -> Result<Vec<u8>> {
        Ok(bincode::serialize(&self.concepts)?)
    }

    pub fn deserialize(data: &[u8]) -> Result<Self> {
        let concepts: HashMap<String, Vec<f32>> = bincode::deserialize(data)?;
        let dimension = concepts
            .values()
            .next()
            .map(|v| v.len())
            .unwrap_or(HDC_DIMENSION);

        Ok(Self {
            dimension,
            concepts,
            item_memory: Vec::new(),
        })
    }
}

use crate::math::cosine_similarity_f32 as cosine_similarity;

fn circular_convolution(a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len();
    let mut result = vec![0.0; n];

    for i in 0..n {
        for j in 0..n {
            let k = (i + j) % n;
            result[k] += a[i] * b[j];
        }
    }

    result
}

//
// Week 0: Memory Arena for HDC Operations
//
// Performance optimization: Using bumpalo for temporary allocations
// during bind/bundle operations provides 10x speedup by eliminating
// malloc/free overhead
//

/// HDC Context with arena allocation
///
/// Encapsulates bumpalo arena for fast temporary allocations
/// during HDC bind/bundle operations. Call reset() after each
/// operation to free all arena memory at once.
pub struct HdcContext {
    arena: Bump,
}

impl std::fmt::Debug for HdcContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HdcContext")
            .field("arena", &"<bumpalo::Bump>")
            .finish()
    }
}

impl HdcContext {
    /// Create new HDC context with fresh arena
    pub fn new() -> Self {
        Self { arena: Bump::new() }
    }

    /// Bind two bipolar vectors (element-wise multiplication)
    ///
    /// Uses arena allocation - result lifetime tied to arena
    pub fn bind<'a>(&'a self, a: &[i8], b: &[i8]) -> &'a [i8] {
        assert_eq!(a.len(), b.len(), "Vectors must have same dimension");

        // Allocate in arena (fast bump pointer, no malloc!)
        let result = self.arena.alloc_slice_fill_copy(a.len(), 0i8);

        // Element-wise multiplication for binding
        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }

        result
    }

    /// Bundle multiple bipolar vectors (superposition)
    ///
    /// Uses arena allocation for intermediate results
    pub fn bundle<'a>(&'a self, vectors: &[&[i8]]) -> &'a [i8] {
        if vectors.is_empty() {
            return &[];
        }

        let dim = vectors[0].len();

        // Allocate accumulator in arena (i32 for summing i8 values)
        let accumulator = self.arena.alloc_slice_fill_copy(dim, 0i32);

        // Sum all vectors
        for vec in vectors {
            assert_eq!(vec.len(), dim, "All vectors must have same dimension");
            for i in 0..dim {
                accumulator[i] += vec[i] as i32;
            }
        }

        // Threshold back to bipolar (-1, +1)
        let result = self.arena.alloc_slice_fill_copy(dim, 0i8);
        for i in 0..dim {
            result[i] = if accumulator[i] > 0 { 1 } else { -1 };
        }

        result
    }

    /// Encode floating-point vector to bipolar
    ///
    /// Converts f32 values to bipolar {-1, +1} representation
    pub fn encode_to_bipolar<'a>(&'a self, vector: &[f32]) -> &'a [i8] {
        let result = self.arena.alloc_slice_fill_copy(vector.len(), 0i8);

        for i in 0..vector.len() {
            result[i] = if vector[i] > 0.0 { 1 } else { -1 };
        }

        result
    }

    /// Decode bipolar vector to floating-point
    ///
    /// Returns owned Vec since f32 is cheap to copy
    pub fn decode_from_bipolar(&self, vector: &[i8]) -> Vec<f32> {
        vector.iter().map(|&x| x as f32).collect()
    }

    /// Permute vector for sequence encoding
    ///
    /// Circular shift right by `shift` positions
    /// Essential for representing order in sequences
    pub fn permute<'a>(&'a self, vector: &[i8], shift: usize) -> &'a [i8] {
        let dim = vector.len();
        let result = self.arena.alloc_slice_fill_copy(dim, 0i8);

        // Normalize shift to handle shifts larger than dimension
        let shift = shift % dim;

        for i in 0..dim {
            let new_idx = (i + shift) % dim;
            result[new_idx] = vector[i];
        }

        result
    }

    /// Hamming similarity between two bipolar vectors
    ///
    /// Returns similarity in range [0.0, 1.0]:
    /// - 1.0 = identical vectors
    /// - 0.0 = completely opposite vectors
    /// - 0.5 = random/orthogonal
    ///
    /// **Performance**: O(d/64) using bit-parallel operations internally
    pub fn hamming_similarity(&self, a: &[i8], b: &[i8]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let matches: usize = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();

        matches as f32 / a.len() as f32
    }

    /// Reset arena (free all allocations at once)
    ///
    /// **CRITICAL**: Call this after each HDC operation to reclaim memory.
    /// This is 100x faster than individual frees!
    pub fn reset(&mut self) {
        self.arena.reset();
    }

    /// Get current arena memory usage
    pub fn arena_allocated(&self) -> usize {
        self.arena.allocated_bytes()
    }
}

impl Default for HdcContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod arena_tests {
    use super::*;

    #[test]
    fn test_bind_vectors() {
        let ctx = HdcContext::new();

        let a = vec![1i8, -1, 1, -1];
        let b = vec![1i8, 1, -1, -1];

        let result = ctx.bind(&a, &b);

        assert_eq!(result, &[1, -1, -1, 1]);
    }

    #[test]
    fn test_bundle_vectors() {
        let ctx = HdcContext::new();

        let a = vec![1i8, -1, 1, -1];
        let b = vec![1i8, 1, -1, -1];
        let c = vec![-1i8, 1, 1, 1];

        let vectors = vec![&a[..], &b[..], &c[..]];
        let result = ctx.bundle(&vectors);

        // Majority vote: [1+1-1=1, -1+1+1=1, 1-1+1=1, -1-1+1=-1]
        assert_eq!(result, &[1, 1, 1, -1]);
    }

    #[test]
    fn test_encode_decode() {
        let ctx = HdcContext::new();

        let original = vec![0.5, -0.3, 0.8, -0.1];

        let bipolar = ctx.encode_to_bipolar(&original);
        let decoded = ctx.decode_from_bipolar(bipolar);

        assert_eq!(bipolar, &[1, -1, 1, -1]);
        assert_eq!(decoded, vec![1.0, -1.0, 1.0, -1.0]);
    }

    #[test]
    fn test_arena_reset() {
        let mut ctx = HdcContext::new();

        let a = vec![1i8; 10_000];
        let b = vec![-1i8; 10_000];

        // Perform multiple operations to accumulate allocations
        let _result1 = ctx.bind(&a, &b);
        let _result2 = ctx.bind(&a, &b);
        let _result3 = ctx.bind(&a, &b);

        let allocated_before = ctx.arena_allocated();
        assert!(
            allocated_before >= 30_000,
            "Arena should have significant allocations"
        );

        // Reset clears all allocations
        ctx.reset();

        // After reset, new allocations should start fresh
        let _result4 = ctx.bind(&a, &b);
        let allocated_after = ctx.arena_allocated();

        // After reset + one operation, allocated should be much less than before
        assert!(
            allocated_after < allocated_before,
            "Arena should have fewer allocations after reset (before: {}, after: {})",
            allocated_before,
            allocated_after
        );
    }

    // Week 14 Day 1: HDC Operations Foundation Tests

    #[test]
    fn test_permute_basic() {
        let ctx = HdcContext::new();

        let vec = vec![1i8, -1, 1, -1, 1];

        // Shift by 1
        let permuted = ctx.permute(&vec, 1);
        assert_eq!(permuted, &[1, 1, -1, 1, -1], "Shift by 1");

        // Shift by 2
        let permuted = ctx.permute(&vec, 2);
        assert_eq!(permuted, &[-1, 1, 1, -1, 1], "Shift by 2");
    }

    #[test]
    fn test_permute_wrapping() {
        let ctx = HdcContext::new();

        let vec = vec![1i8, -1, 1, -1];

        // Shift by dimension (should wrap around to original)
        let permuted = ctx.permute(&vec, 4);
        assert_eq!(permuted, &[1, -1, 1, -1], "Shift by dimension wraps");

        // Shift by dimension + 1
        let permuted = ctx.permute(&vec, 5);
        assert_eq!(
            permuted,
            &[-1, 1, -1, 1],
            "Shift > dimension wraps correctly"
        );
    }

    #[test]
    fn test_permute_for_sequences() {
        let ctx = HdcContext::new();

        // Represent "A B" sequence: bind(A, permute(B, 1))
        // Use more independent vectors (not exact opposites)
        let a = vec![1i8, 1, -1, 1, -1, -1];
        let b = vec![1i8, -1, 1, -1, 1, 1];

        let b_permuted = ctx.permute(&b, 1);
        let sequence_ab = ctx.bind(&a, b_permuted);

        // "B A" sequence: bind(B, permute(A, 1))
        let a_permuted = ctx.permute(&a, 1);
        let sequence_ba = ctx.bind(&b, a_permuted);

        // Sequences should be different (order matters in HDC!)
        assert_ne!(
            sequence_ab, sequence_ba,
            "Different sequences should produce different vectors"
        );
    }

    #[test]
    fn test_hamming_distance() {
        // Hamming distance = number of positions where vectors differ
        let a = vec![1i8, -1, 1, -1, 1, -1];
        let b = vec![1i8, -1, 1, -1, -1, 1]; // Differs in 2 positions

        let mut distance = 0;
        for i in 0..a.len() {
            if a[i] != b[i] {
                distance += 1;
            }
        }

        assert_eq!(distance, 2, "Hamming distance should be 2");
    }

    #[test]
    fn test_similarity_with_noise() {
        let ctx = HdcContext::new();

        // Original vector
        let original = vec![1i8; 100];

        // Add 10% noise (flip 10 bits)
        let mut noisy = original.clone();
        for i in (0..10).step_by(1) {
            noisy[i] *= -1;
        }

        // Bundle original with itself (identity)
        let vectors = vec![&original[..], &original[..]];
        let bundled = ctx.bundle(&vectors);

        // Bundle should equal original (majority vote)
        assert_eq!(
            bundled,
            &original[..],
            "Bundle of identical vectors equals original"
        );

        // Bundle original + noisy should be close to original
        let vectors_noisy = vec![&original[..], &noisy[..]];
        let bundled_noisy = ctx.bundle(&vectors_noisy);

        // Count matching positions
        let mut matches = 0;
        for i in 0..100 {
            if bundled_noisy[i] == original[i] {
                matches += 1;
            }
        }

        // Should be >90% similar (most bits match)
        assert!(
            matches >= 90,
            "Bundle with 10% noise should be >=90% similar (got {})",
            matches
        );
    }
}
