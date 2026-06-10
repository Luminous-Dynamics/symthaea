// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mathematics curriculum for Symthaea's School system
//!
//! Defines progressive learning objectives for mathematical reasoning,
//! from basic integer arithmetic through formal verification and Fourier
//! analysis.  Two complementary representations are provided:
//!
//! 1. **`MathCurriculum`** — a self-contained, HDC-aware curriculum that
//!    tracks mastery, prerequisite chains, and encodes progress as a
//!    `BinaryHV` bundle.  This is the primary type for the math domain.
//!
//! 2. **`math_curriculum()`** / **`math_curriculum_advanced()`** — thin
//!    wrappers that expose the same objectives through the shared
//!    [`Curriculum`] builder used by the rest of the School system, so
//!    they can be loaded by [`School::add_curriculum`] without any
//!    additional glue.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::seed_from_name;

use super::curriculum::Curriculum;
use super::objective::{Difficulty, Domain, LearningObjective};

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════

const MATH_DOMAIN: &str = "Mathematics";
const CURRICULUM_ID: &str = "mathematics";
const ADVANCED_CURRICULUM_ID: &str = "mathematics-advanced";

// ── Tier ID slices (used for prerequisite wiring and tests) ────────────────

const TIER1_IDS: &[&str] = &[
    "math_integer_arithmetic",
    "math_order_of_operations",
    "math_fractions_decimals",
    "math_basic_exponents",
    "math_number_line",
    "math_intro_variables",
    "math_simple_equations",
    "math_inequalities",
    "math_coordinate_plane",
    "math_ratio_proportion",
];

const TIER2_IDS: &[&str] = &[
    "math_linear_systems",
    "math_polynomial_arithmetic",
    "math_quadratic_equations",
    "math_basic_functions",
    "math_combinatorics_intro",
    "math_basic_probability",
    "math_descriptive_statistics",
    "math_set_theory_basics",
    "math_logic_propositional",
    "math_graph_intro",
];

const TIER3_IDS: &[&str] = &[
    "math_matrices_linear_algebra",
    "math_eigenvalues_eigenvectors",
    "math_differentiation",
    "math_integration",
    "math_multivariable_calculus",
    "math_bayes_theorem",
    "math_hypothesis_testing",
    "math_complex_numbers",
    "math_recurrence_relations",
    "math_information_theory_basics",
];

const TIER4_IDS: &[&str] = &[
    "math_svd_matrix_decomp",
    "math_convex_optimization",
    "math_ode_systems",
    "math_proof_techniques",
    "math_constraint_satisfaction",
    "math_stochastic_processes",
    "math_abstract_algebra_groups",
    "math_topology_intro",
    "math_numerical_methods",
    "math_ml_math_foundations",
];

const TIER5_IDS: &[&str] = &[
    "math_fourier_analysis",
    "math_wavelet_transforms",
    "math_numerical_stability",
    "math_formal_verification",
    "math_category_theory",
    "math_manifold_learning",
    "math_functional_analysis",
    "math_measure_theory",
    "math_algorithmic_complexity",
    "math_symbolic_regression",
];

// ═══════════════════════════════════════════════════════════════════════════
// MATH DIFFICULTY ENUM
// ═══════════════════════════════════════════════════════════════════════════

/// Five-tier difficulty ladder for mathematical objectives.
///
/// Mirrors [`Difficulty`] but is self-contained within this module so
/// that `MathCurriculum` can be used without the rest of the School
/// machinery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MathDifficulty {
    /// Foundational arithmetic and algebra (no prerequisites).
    Beginner,
    /// Elementary algebra, basic probability, polynomial arithmetic.
    Elementary,
    /// Matrices, calculus, Bayes' theorem, eigenvalues.
    Intermediate,
    /// SVD, optimisation, ODEs, proof techniques, CSP.
    Advanced,
    /// Fourier analysis, numerical stability, formal verification.
    Expert,
}

impl MathDifficulty {
    /// Numeric representation suitable for scoring (0.0 – 1.0).
    pub fn as_f32(self) -> f32 {
        match self {
            MathDifficulty::Beginner => 0.1,
            MathDifficulty::Elementary => 0.3,
            MathDifficulty::Intermediate => 0.5,
            MathDifficulty::Advanced => 0.7,
            MathDifficulty::Expert => 0.9,
        }
    }

    /// Convert to the shared [`Difficulty`] type used by [`Curriculum`].
    pub fn to_difficulty(self) -> Difficulty {
        match self {
            MathDifficulty::Beginner => Difficulty::Beginner,
            MathDifficulty::Elementary => Difficulty::Elementary,
            MathDifficulty::Intermediate => Difficulty::Intermediate,
            MathDifficulty::Advanced => Difficulty::Advanced,
            MathDifficulty::Expert => Difficulty::Expert,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MATH OBJECTIVE
// ═══════════════════════════════════════════════════════════════════════════

/// A single mathematical learning objective with HDC encoding and mastery
/// state.
///
/// The `encoding` field is a deterministic [`BinaryHV`] derived from the
/// objective's `id` via [`seed_from_name`], so the same objective always
/// hashes to the same vector regardless of insertion order.
#[derive(Clone, Serialize, Deserialize)]
pub struct MathObjective {
    /// Stable numeric index (position in the objectives Vec).
    pub id: usize,

    /// Short identifier string, e.g. `"math_integer_arithmetic"`.
    pub name: String,

    /// Human-readable description of what is learned.
    pub description: String,

    /// Difficulty tier.
    pub difficulty: MathDifficulty,

    /// Prerequisite objective indices (indices into the owning
    /// `MathCurriculum::objectives` Vec).
    pub prerequisites: Vec<usize>,

    /// Deterministic 16 384-bit hypervector encoding for this concept.
    #[serde(skip)]
    pub encoding: BinaryHV,

    /// Whether the learner has mastered this objective.
    pub mastered: bool,
}

impl std::fmt::Debug for MathObjective {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MathObjective")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("difficulty", &self.difficulty)
            .field("prerequisites", &self.prerequisites)
            .field("mastered", &self.mastered)
            .finish()
    }
}

impl MathObjective {
    /// Construct a new objective.  The HDC encoding is computed
    /// deterministically from `name`.
    fn new(
        id: usize,
        name: &str,
        description: &str,
        difficulty: MathDifficulty,
        prerequisites: Vec<usize>,
    ) -> Self {
        let seed = seed_from_name(name);
        Self {
            id,
            name: name.to_string(),
            description: description.to_string(),
            difficulty,
            prerequisites,
            encoding: BinaryHV::random(seed),
            mastered: false,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MATH CURRICULUM
// ═══════════════════════════════════════════════════════════════════════════

/// Self-contained mathematics curriculum with 50+ objectives across five
/// difficulty tiers, prerequisite chains, mastery tracking, and HDC
/// progress encoding.
///
/// # Quick start
///
/// ```ignore
/// let mut mc = MathCurriculum::new();
///
/// // Mark an objective as mastered
/// mc.objectives[0].mastered = true;
///
/// // Find what is unlocked next
/// let mastered: HashSet<usize> = mc.objectives.iter()
///     .filter(|o| o.mastered)
///     .map(|o| o.id)
///     .collect();
/// let available = mc.next_objectives(&mastered);
///
/// // Encode all mastered objectives into a single BinaryHV
/// let progress_hv = mc.encode_progress();
/// ```
#[derive(Debug)]
pub struct MathCurriculum {
    /// All 50+ objectives in dependency order (Beginner → Expert).
    pub objectives: Vec<MathObjective>,
}

impl MathCurriculum {
    // ── Construction ────────────────────────────────────────────────────

    /// Create the full 50-objective mathematics curriculum.
    ///
    /// Objectives are indexed 0…49 in the order they appear; prerequisite
    /// lists contain those indices.  The tier layout is:
    ///
    /// | Tier | Indices | Count |
    /// |------|---------|-------|
    /// | Beginner (T1) | 0–9 | 10 |
    /// | Elementary (T2) | 10–19 | 10 |
    /// | Intermediate (T3) | 20–29 | 10 |
    /// | Advanced (T4) | 30–39 | 10 |
    /// | Expert (T5) | 40–49 | 10 |
    pub fn new() -> Self {
        // Prerequisite index ranges for each tier
        let t1: Vec<usize> = (0..10).collect();
        let t2: Vec<usize> = (10..20).collect();
        let t3: Vec<usize> = (20..30).collect();
        let t4: Vec<usize> = (30..40).collect();

        let objectives = vec![
            // ── Tier 1: Beginner ──────────────────────────────────────────
            MathObjective::new(
                0,
                "math_integer_arithmetic",
                "Perform addition, subtraction, multiplication, and division on integers, \
                 including handling negative numbers and zero.",
                MathDifficulty::Beginner,
                vec![],
            ),
            MathObjective::new(
                1,
                "math_order_of_operations",
                "Apply PEMDAS/BODMAS rules to evaluate arithmetic expressions with nested \
                 parentheses, exponents, and mixed operators.",
                MathDifficulty::Beginner,
                vec![],
            ),
            MathObjective::new(
                2,
                "math_fractions_decimals",
                "Convert between fractions, decimals, and percentages; perform arithmetic \
                 on rational numbers in all three forms.",
                MathDifficulty::Beginner,
                vec![],
            ),
            MathObjective::new(
                3,
                "math_basic_exponents",
                "Evaluate integer and fractional exponents; understand square roots, \
                 cube roots, and the laws of exponents.",
                MathDifficulty::Beginner,
                vec![0],
            ),
            MathObjective::new(
                4,
                "math_number_line",
                "Represent real numbers on a number line; understand absolute value, \
                 distance, and ordering.",
                MathDifficulty::Beginner,
                vec![],
            ),
            MathObjective::new(
                5,
                "math_intro_variables",
                "Introduce symbolic variables; evaluate algebraic expressions for given \
                 substitution values; combine like terms.",
                MathDifficulty::Beginner,
                vec![0],
            ),
            MathObjective::new(
                6,
                "math_simple_equations",
                "Solve single-variable linear equations using inverse operations; verify \
                 solutions by substitution.",
                MathDifficulty::Beginner,
                vec![5],
            ),
            MathObjective::new(
                7,
                "math_inequalities",
                "Solve and graph linear inequalities on a number line; understand \
                 direction-flipping when multiplying or dividing by negatives.",
                MathDifficulty::Beginner,
                vec![4, 6],
            ),
            MathObjective::new(
                8,
                "math_coordinate_plane",
                "Plot and read points in the Cartesian coordinate plane; compute distances \
                 and midpoints between two points.",
                MathDifficulty::Beginner,
                vec![4, 5],
            ),
            MathObjective::new(
                9,
                "math_ratio_proportion",
                "Set up and solve ratio and proportion problems; apply cross-multiplication \
                 and percentage change formulas.",
                MathDifficulty::Beginner,
                vec![2, 5],
            ),
            // ── Tier 2: Elementary ────────────────────────────────────────
            MathObjective::new(
                10,
                "math_linear_systems",
                "Solve 2×2 and 3×3 linear systems by substitution, elimination, \
                 and matrix row reduction.",
                MathDifficulty::Elementary,
                t1.clone(),
            ),
            MathObjective::new(
                11,
                "math_polynomial_arithmetic",
                "Add, subtract, multiply, and factor polynomials; apply the binomial \
                 theorem for small exponents.",
                MathDifficulty::Elementary,
                t1.clone(),
            ),
            MathObjective::new(
                12,
                "math_quadratic_equations",
                "Solve quadratic equations by factoring, completing the square, and the \
                 quadratic formula; interpret the discriminant.",
                MathDifficulty::Elementary,
                vec![11],
            ),
            MathObjective::new(
                13,
                "math_basic_functions",
                "Define functions as mappings; understand domain and range; compose \
                 functions; find inverses for simple cases.",
                MathDifficulty::Elementary,
                vec![6, 8],
            ),
            MathObjective::new(
                14,
                "math_combinatorics_intro",
                "Apply the multiplication principle, permutations, and combinations \
                 (nPr, nCr) to counting problems.",
                MathDifficulty::Elementary,
                t1.clone(),
            ),
            MathObjective::new(
                15,
                "math_basic_probability",
                "Compute classical, empirical, and conditional probability; apply \
                 the addition and multiplication rules.",
                MathDifficulty::Elementary,
                vec![14],
            ),
            MathObjective::new(
                16,
                "math_descriptive_statistics",
                "Compute mean, median, mode, variance, and standard deviation; \
                 interpret histograms, box plots, and scatter plots.",
                MathDifficulty::Elementary,
                t1.clone(),
            ),
            MathObjective::new(
                17,
                "math_set_theory_basics",
                "Operate on sets with union, intersection, complement, and Cartesian \
                 product; apply Venn diagrams to solve problems.",
                MathDifficulty::Elementary,
                t1.clone(),
            ),
            MathObjective::new(
                18,
                "math_logic_propositional",
                "Construct and evaluate propositional logic expressions; build truth \
                 tables; identify tautologies and contradictions.",
                MathDifficulty::Elementary,
                vec![17],
            ),
            MathObjective::new(
                19,
                "math_graph_intro",
                "Represent graphs as adjacency matrices and edge lists; compute degree \
                 sequences and identify simple graph properties.",
                MathDifficulty::Elementary,
                vec![17],
            ),
            // ── Tier 3: Intermediate ──────────────────────────────────────
            MathObjective::new(
                20,
                "math_matrices_linear_algebra",
                "Perform matrix multiplication, transposition, inversion; solve Ax = b \
                 via Gaussian elimination and LU decomposition.",
                MathDifficulty::Intermediate,
                t2.clone(),
            ),
            MathObjective::new(
                21,
                "math_eigenvalues_eigenvectors",
                "Compute eigenvalues and eigenvectors of square matrices; diagonalise \
                 symmetric matrices; apply spectral theorem.",
                MathDifficulty::Intermediate,
                vec![20],
            ),
            MathObjective::new(
                22,
                "math_differentiation",
                "Differentiate polynomials, trigonometric, exponential, and logarithmic \
                 functions using chain, product, and quotient rules.",
                MathDifficulty::Intermediate,
                t2.clone(),
            ),
            MathObjective::new(
                23,
                "math_integration",
                "Evaluate definite and indefinite integrals by substitution, integration \
                 by parts, and partial fractions; apply the Fundamental Theorem.",
                MathDifficulty::Intermediate,
                vec![22],
            ),
            MathObjective::new(
                24,
                "math_multivariable_calculus",
                "Compute partial derivatives, gradients, divergence, and curl; apply the \
                 chain rule and Jacobian to multivariable functions.",
                MathDifficulty::Intermediate,
                vec![22, 20],
            ),
            MathObjective::new(
                25,
                "math_bayes_theorem",
                "Apply Bayes' theorem to update prior beliefs with evidence; reason \
                 about posterior distributions in discrete settings.",
                MathDifficulty::Intermediate,
                vec![15],
            ),
            MathObjective::new(
                26,
                "math_hypothesis_testing",
                "Formulate null and alternative hypotheses; apply t-tests, chi-square, \
                 and ANOVA; interpret p-values and confidence intervals.",
                MathDifficulty::Intermediate,
                vec![16, 25],
            ),
            MathObjective::new(
                27,
                "math_complex_numbers",
                "Perform arithmetic on complex numbers in rectangular and polar form; \
                 compute roots of unity using De Moivre's theorem.",
                MathDifficulty::Intermediate,
                vec![12, 3],
            ),
            MathObjective::new(
                28,
                "math_recurrence_relations",
                "Solve linear recurrence relations by characteristic roots; identify \
                 closed forms for Fibonacci-style and divide-and-conquer recurrences.",
                MathDifficulty::Intermediate,
                vec![12, 18],
            ),
            MathObjective::new(
                29,
                "math_information_theory_basics",
                "Compute Shannon entropy, KL divergence, and mutual information; \
                 understand channel capacity and the noiseless coding theorem.",
                MathDifficulty::Intermediate,
                vec![15, 25],
            ),
            // ── Tier 4: Advanced ──────────────────────────────────────────
            MathObjective::new(
                30,
                "math_svd_matrix_decomp",
                "Compute singular-value decomposition, QR factorisation, and Cholesky \
                 decomposition; apply SVD to low-rank approximation and PCA.",
                MathDifficulty::Advanced,
                t3.clone(),
            ),
            MathObjective::new(
                31,
                "math_convex_optimization",
                "Identify convex sets and functions; apply gradient descent, Newton's \
                 method, and Lagrangian duality; understand KKT conditions.",
                MathDifficulty::Advanced,
                vec![24, 21],
            ),
            MathObjective::new(
                32,
                "math_ode_systems",
                "Solve first- and second-order ODEs analytically; analyse phase \
                 portraits of 2D autonomous systems; apply Laplace transforms.",
                MathDifficulty::Advanced,
                vec![23, 21],
            ),
            MathObjective::new(
                33,
                "math_proof_techniques",
                "Write rigorous mathematical proofs using direct proof, contrapositive, \
                 contradiction, induction, and structural induction.",
                MathDifficulty::Advanced,
                vec![18, 28],
            ),
            MathObjective::new(
                34,
                "math_constraint_satisfaction",
                "Model and solve constraint satisfaction problems (CSPs); apply \
                 arc-consistency, backtracking, and local search.",
                MathDifficulty::Advanced,
                vec![19, 18],
            ),
            MathObjective::new(
                35,
                "math_stochastic_processes",
                "Define and analyse discrete-time Markov chains; compute stationary \
                 distributions; understand Poisson processes and Brownian motion.",
                MathDifficulty::Advanced,
                vec![25, 29],
            ),
            MathObjective::new(
                36,
                "math_abstract_algebra_groups",
                "Understand groups, rings, and fields; apply Lagrange's theorem, \
                 homomorphisms, and quotient groups.",
                MathDifficulty::Advanced,
                vec![33, 17],
            ),
            MathObjective::new(
                37,
                "math_topology_intro",
                "Define topological spaces, open sets, and continuity; compute \
                 fundamental groups and understand connectedness and compactness.",
                MathDifficulty::Advanced,
                vec![33, 24],
            ),
            MathObjective::new(
                38,
                "math_numerical_methods",
                "Implement Newton–Raphson root finding, bisection, finite differences, \
                 Runge–Kutta integration, and Newton–Cotes quadrature.",
                MathDifficulty::Advanced,
                vec![22, 23, 20],
            ),
            MathObjective::new(
                39,
                "math_ml_math_foundations",
                "Derive gradient descent, backpropagation, cross-entropy loss, and \
                 softmax from first principles; understand the bias–variance tradeoff.",
                MathDifficulty::Advanced,
                vec![31, 30, 26],
            ),
            // ── Tier 5: Expert ────────────────────────────────────────────
            MathObjective::new(
                40,
                "math_fourier_analysis",
                "Derive the Fourier series and Fourier transform; apply DFT/FFT to \
                 signal processing; understand convolution theorem and Parseval's identity.",
                MathDifficulty::Expert,
                t4.clone(),
            ),
            MathObjective::new(
                41,
                "math_wavelet_transforms",
                "Construct wavelet bases (Haar, Daubechies); perform multi-resolution \
                 analysis; apply wavelets to compression and denoising.",
                MathDifficulty::Expert,
                vec![40],
            ),
            MathObjective::new(
                42,
                "math_numerical_stability",
                "Analyse floating-point rounding, condition numbers, and backward \
                 stability; apply pivoting strategies and iterative refinement.",
                MathDifficulty::Expert,
                vec![38, 30],
            ),
            MathObjective::new(
                43,
                "math_formal_verification",
                "Encode mathematical theorems in a proof assistant (Lean/Coq); prove \
                 properties of algorithms using type theory and dependent types.",
                MathDifficulty::Expert,
                vec![33, 36],
            ),
            MathObjective::new(
                44,
                "math_category_theory",
                "Define categories, functors, and natural transformations; understand \
                 limits, colimits, and adjoint functors.",
                MathDifficulty::Expert,
                vec![36, 37],
            ),
            MathObjective::new(
                45,
                "math_manifold_learning",
                "Analyse smooth manifolds, tangent spaces, and Riemannian metrics; \
                 apply geodesic distance and parallel transport to ML embeddings.",
                MathDifficulty::Expert,
                vec![37, 31],
            ),
            MathObjective::new(
                46,
                "math_functional_analysis",
                "Study Hilbert and Banach spaces, bounded linear operators, and spectral \
                 theory; apply to PDEs and kernel methods.",
                MathDifficulty::Expert,
                vec![40, 37],
            ),
            MathObjective::new(
                47,
                "math_measure_theory",
                "Define sigma-algebras, measures, and Lebesgue integration; understand \
                 convergence theorems and Radon–Nikodym derivative.",
                MathDifficulty::Expert,
                vec![33, 35],
            ),
            MathObjective::new(
                48,
                "math_algorithmic_complexity",
                "Classify problems in P, NP, and beyond; prove hardness via reductions; \
                 understand randomised and approximation algorithms.",
                MathDifficulty::Expert,
                vec![34, 33],
            ),
            MathObjective::new(
                49,
                "math_symbolic_regression",
                "Apply genetic programming and sparse regression (SINDy) to recover \
                 symbolic closed-form expressions from numerical data.",
                MathDifficulty::Expert,
                vec![39, 42],
            ),
        ];

        Self { objectives }
    }

    // ── Queries ──────────────────────────────────────────────────────────

    /// Return references to all objectives at a given difficulty tier.
    pub fn objectives_at_level(&self, difficulty: MathDifficulty) -> Vec<&MathObjective> {
        self.objectives
            .iter()
            .filter(|o| o.difficulty == difficulty)
            .collect()
    }

    /// Return `true` if every prerequisite for `objective_id` appears in
    /// `mastered_set`.
    pub fn prerequisites_met(&self, objective_id: usize, mastered_set: &HashSet<usize>) -> bool {
        if let Some(obj) = self.objectives.get(objective_id) {
            obj.prerequisites.iter().all(|p| mastered_set.contains(p))
        } else {
            false
        }
    }

    /// Return all objectives whose prerequisites are fully satisfied by
    /// `mastered_set` and which are not themselves already mastered.
    pub fn next_objectives(&self, mastered_set: &HashSet<usize>) -> Vec<&MathObjective> {
        self.objectives
            .iter()
            .filter(|o| !mastered_set.contains(&o.id))
            .filter(|o| self.prerequisites_met(o.id, mastered_set))
            .collect()
    }

    /// Fraction of objectives currently marked `mastered` (0.0 – 1.0).
    pub fn mastery_percentage(&self) -> f64 {
        if self.objectives.is_empty() {
            return 0.0;
        }
        let mastered = self.objectives.iter().filter(|o| o.mastered).count();
        mastered as f64 / self.objectives.len() as f64
    }

    /// Bundle all mastered objective encodings into a single [`BinaryHV`]
    /// via majority-vote bundling.
    ///
    /// Returns the zero vector when no objectives have been mastered yet.
    pub fn encode_progress(&self) -> BinaryHV {
        let mastered: Vec<&BinaryHV> = self
            .objectives
            .iter()
            .filter(|o| o.mastered)
            .map(|o| &o.encoding)
            .collect();

        if mastered.is_empty() {
            return BinaryHV::zero();
        }

        // Majority-vote bundle using bit-level counting.
        // For each bit position, count how many mastered HVs have it set.
        // If the count exceeds half the number of mastered HVs, the output
        // bit is 1; otherwise 0.
        let threshold = mastered.len() as u32;
        let mut counts = [0u32; 16384]; // one counter per bit (2048 bytes × 8 bits)

        for hv in &mastered {
            for (byte_idx, &byte) in hv.0.iter().enumerate() {
                for bit in 0..8u32 {
                    if byte & (1 << bit) != 0 {
                        counts[byte_idx * 8 + bit as usize] += 2;
                    }
                }
            }
        }

        let mut result = [0u8; 2048];
        for byte_idx in 0..2048 {
            let mut byte = 0u8;
            for bit in 0..8u32 {
                if counts[byte_idx * 8 + bit as usize] > threshold {
                    byte |= 1 << bit;
                }
            }
            result[byte_idx] = byte;
        }

        BinaryHV(result)
    }

    /// Update mastery based on MathService telemetry.
    ///
    /// Maps problem types to objective IDs and marks objectives as mastered
    /// when the corresponding solver has been exercised enough times (threshold = 3).
    /// This bridges the MathService solver feedback into the School's mastery tracking.
    pub fn update_from_telemetry(&mut self, by_type: &std::collections::HashMap<String, usize>) {
        // Problem type → objective IDs mapping
        let mappings: &[(&str, &[usize])] = &[
            ("LinearSystem", &[10]),           // math_linear_systems
            ("RootFinding", &[12]),            // math_quadratic_equations
            ("Integration", &[17]),            // math_integration
            ("MatrixAnalysis", &[20, 21]),     // math_matrices_linear_algebra, math_eigenvalues
            ("Statistics", &[16]),             // math_descriptive_statistics
            ("Optimization", &[31]),           // math_convex_optimization
            ("SignalAnalysis", &[40]),         // math_fourier_analysis
            ("Logic", &[18]),                  // math_logic_propositional
            ("ConstraintSatisfaction", &[34]), // math_constraint_satisfaction
            ("Geometry", &[8]),                // math_coordinate_plane
            ("GraphTheory", &[19]),            // math_graph_intro
            ("DifferentialEquation", &[32]),   // math_ode_systems
        ];

        let mastery_threshold = 3;
        for (type_name, obj_ids) in mappings {
            if let Some(&count) = by_type.get(*type_name) {
                if count >= mastery_threshold {
                    for &id in *obj_ids {
                        if let Some(obj) = self.objectives.get_mut(id) {
                            obj.mastered = true;
                        }
                    }
                }
            }
        }
    }
}

impl Default for MathCurriculum {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CURRICULUM BUILDER INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════
//
// The two functions below expose the same 50 objectives through the shared
// `Curriculum` + `LearningObjective` API so that `School::add_curriculum`
// can consume them directly.

/// Build the core mathematics curriculum (50 objectives across 5 tiers).
///
/// Integrates with the [`School`] system via [`Curriculum`].
pub fn math_curriculum() -> Curriculum {
    Curriculum::new(CURRICULUM_ID, "Mathematics")
        .with_description(
            "Progressive mathematics curriculum from integer arithmetic through \
             Fourier analysis and formal verification, organised in five difficulty \
             tiers with strict prerequisite ordering.",
        )
        .with_tags(&["math", "algebra", "calculus", "probability", "logic"])
        // ── Tier 1: Beginner ────────────────────────────────────────────
        .with_objective(
            LearningObjective::new("math_integer_arithmetic", "Integer Arithmetic")
                .with_description(
                    "Perform addition, subtraction, multiplication, and division on \
                     integers, including negative numbers and zero.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Beginner)
                .with_tags(&["math", "arithmetic", "integers", "foundations"])
                .with_estimated_minutes(20),
        )
        .with_objective(
            LearningObjective::new("math_order_of_operations", "Order of Operations")
                .with_description(
                    "Apply PEMDAS/BODMAS rules to evaluate arithmetic expressions with \
                     nested parentheses, exponents, and mixed operators.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Beginner)
                .with_tags(&["math", "arithmetic", "order-of-operations"])
                .with_estimated_minutes(15),
        )
        .with_objective(
            LearningObjective::new("math_fractions_decimals", "Fractions and Decimals")
                .with_description(
                    "Convert between fractions, decimals, and percentages; perform \
                     arithmetic on rational numbers in all three forms.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Beginner)
                .with_tags(&["math", "fractions", "decimals", "rational"])
                .with_estimated_minutes(20),
        )
        .with_objective(
            LearningObjective::new("math_basic_exponents", "Exponents and Roots")
                .with_description(
                    "Evaluate integer and fractional exponents; understand square roots, \
                     cube roots, and the laws of exponents.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Beginner)
                .with_prerequisites(&["math_integer_arithmetic"])
                .with_tags(&["math", "exponents", "roots", "powers"])
                .with_estimated_minutes(20),
        )
        .with_objective(
            LearningObjective::new("math_number_line", "The Number Line")
                .with_description(
                    "Represent real numbers on a number line; understand absolute value, \
                     distance, and ordering.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Beginner)
                .with_tags(&["math", "number-line", "absolute-value", "ordering"])
                .with_estimated_minutes(15),
        )
        .with_objective(
            LearningObjective::new("math_intro_variables", "Introduction to Variables")
                .with_description(
                    "Introduce symbolic variables; evaluate algebraic expressions for \
                     given substitution values; combine like terms.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Beginner)
                .with_prerequisites(&["math_integer_arithmetic"])
                .with_tags(&["math", "algebra", "variables", "expressions"])
                .with_estimated_minutes(20),
        )
        .with_objective(
            LearningObjective::new("math_simple_equations", "Simple Linear Equations")
                .with_description(
                    "Solve single-variable linear equations using inverse operations; \
                     verify solutions by substitution.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Beginner)
                .with_prerequisites(&["math_intro_variables"])
                .with_tags(&["math", "algebra", "equations", "linear"])
                .with_estimated_minutes(20),
        )
        .with_objective(
            LearningObjective::new("math_inequalities", "Inequalities")
                .with_description(
                    "Solve and graph linear inequalities on a number line; understand \
                     direction-flipping when multiplying or dividing by negatives.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Beginner)
                .with_prerequisites(&["math_number_line", "math_simple_equations"])
                .with_tags(&["math", "algebra", "inequalities"])
                .with_estimated_minutes(20),
        )
        .with_objective(
            LearningObjective::new("math_coordinate_plane", "The Coordinate Plane")
                .with_description(
                    "Plot and read points in the Cartesian coordinate plane; compute \
                     distances and midpoints between two points.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Beginner)
                .with_prerequisites(&["math_number_line", "math_intro_variables"])
                .with_tags(&["math", "geometry", "coordinates", "cartesian"])
                .with_estimated_minutes(20),
        )
        .with_objective(
            LearningObjective::new("math_ratio_proportion", "Ratios and Proportions")
                .with_description(
                    "Set up and solve ratio and proportion problems; apply \
                     cross-multiplication and percentage change formulas.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Beginner)
                .with_prerequisites(&["math_fractions_decimals", "math_intro_variables"])
                .with_tags(&["math", "arithmetic", "ratio", "proportion"])
                .with_estimated_minutes(20),
        )
        // ── Tier 2: Elementary ──────────────────────────────────────────
        .with_objective(
            LearningObjective::new("math_linear_systems", "Linear Systems of Equations")
                .with_description(
                    "Solve 2×2 and 3×3 linear systems by substitution, elimination, \
                     and matrix row reduction.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Elementary)
                .with_prerequisites(TIER1_IDS)
                .with_tags(&["math", "algebra", "linear-systems", "elimination"])
                .with_estimated_minutes(40),
        )
        .with_objective(
            LearningObjective::new("math_polynomial_arithmetic", "Polynomial Arithmetic")
                .with_description(
                    "Add, subtract, multiply, and factor polynomials; apply the binomial \
                     theorem for small exponents.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Elementary)
                .with_prerequisites(TIER1_IDS)
                .with_tags(&["math", "algebra", "polynomials", "factoring"])
                .with_estimated_minutes(40),
        )
        .with_objective(
            LearningObjective::new("math_quadratic_equations", "Quadratic Equations")
                .with_description(
                    "Solve quadratic equations by factoring, completing the square, and \
                     the quadratic formula; interpret the discriminant.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Elementary)
                .with_prerequisites(&["math_polynomial_arithmetic"])
                .with_tags(&["math", "algebra", "quadratics", "discriminant"])
                .with_estimated_minutes(40),
        )
        .with_objective(
            LearningObjective::new("math_basic_functions", "Functions and Inverses")
                .with_description(
                    "Define functions as mappings; understand domain and range; compose \
                     functions; find inverses for simple cases.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Elementary)
                .with_prerequisites(&["math_simple_equations", "math_coordinate_plane"])
                .with_tags(&["math", "functions", "domain", "range", "inverse"])
                .with_estimated_minutes(40),
        )
        .with_objective(
            LearningObjective::new("math_combinatorics_intro", "Combinatorics")
                .with_description(
                    "Apply the multiplication principle, permutations, and combinations \
                     (nPr, nCr) to counting problems.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Elementary)
                .with_prerequisites(TIER1_IDS)
                .with_tags(&["math", "combinatorics", "permutations", "combinations"])
                .with_estimated_minutes(40),
        )
        .with_objective(
            LearningObjective::new("math_basic_probability", "Basic Probability")
                .with_description(
                    "Compute classical, empirical, and conditional probability; apply \
                     the addition and multiplication rules.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Elementary)
                .with_prerequisites(&["math_combinatorics_intro"])
                .with_tags(&["math", "probability", "conditional", "events"])
                .with_estimated_minutes(40),
        )
        .with_objective(
            LearningObjective::new("math_descriptive_statistics", "Descriptive Statistics")
                .with_description(
                    "Compute mean, median, mode, variance, and standard deviation; \
                     interpret histograms, box plots, and scatter plots.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Elementary)
                .with_prerequisites(TIER1_IDS)
                .with_tags(&["math", "statistics", "mean", "variance", "distribution"])
                .with_estimated_minutes(40),
        )
        .with_objective(
            LearningObjective::new("math_set_theory_basics", "Set Theory")
                .with_description(
                    "Operate on sets with union, intersection, complement, and Cartesian \
                     product; apply Venn diagrams to solve problems.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Elementary)
                .with_prerequisites(TIER1_IDS)
                .with_tags(&["math", "sets", "union", "intersection", "venn"])
                .with_estimated_minutes(30),
        )
        .with_objective(
            LearningObjective::new("math_logic_propositional", "Propositional Logic")
                .with_description(
                    "Construct and evaluate propositional logic expressions; build truth \
                     tables; identify tautologies and contradictions.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Elementary)
                .with_prerequisites(&["math_set_theory_basics"])
                .with_tags(&["math", "logic", "truth-tables", "tautology"])
                .with_estimated_minutes(35),
        )
        .with_objective(
            LearningObjective::new("math_graph_intro", "Introduction to Graph Theory")
                .with_description(
                    "Represent graphs as adjacency matrices and edge lists; compute \
                     degree sequences and identify simple graph properties.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Elementary)
                .with_prerequisites(&["math_set_theory_basics"])
                .with_tags(&["math", "graph-theory", "adjacency", "degree"])
                .with_estimated_minutes(35),
        )
        // ── Tier 3: Intermediate ────────────────────────────────────────
        .with_objective(
            LearningObjective::new(
                "math_matrices_linear_algebra",
                "Matrices and Linear Algebra",
            )
            .with_description(
                "Perform matrix multiplication, transposition, inversion; solve Ax = b \
                     via Gaussian elimination and LU decomposition.",
            )
            .with_domain(Domain::Mathematics)
            .with_difficulty(Difficulty::Intermediate)
            .with_prerequisites(TIER2_IDS)
            .with_tags(&["math", "linear-algebra", "matrices", "gaussian"])
            .with_estimated_minutes(60),
        )
        .with_objective(
            LearningObjective::new(
                "math_eigenvalues_eigenvectors",
                "Eigenvalues and Eigenvectors",
            )
            .with_description(
                "Compute eigenvalues and eigenvectors of square matrices; diagonalise \
                     symmetric matrices; apply the spectral theorem.",
            )
            .with_domain(Domain::Mathematics)
            .with_difficulty(Difficulty::Intermediate)
            .with_prerequisites(&["math_matrices_linear_algebra"])
            .with_tags(&["math", "linear-algebra", "eigenvalues", "spectral"])
            .with_estimated_minutes(60),
        )
        .with_objective(
            LearningObjective::new("math_differentiation", "Differentiation")
                .with_description(
                    "Differentiate polynomials, trigonometric, exponential, and logarithmic \
                     functions using chain, product, and quotient rules.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Intermediate)
                .with_prerequisites(TIER2_IDS)
                .with_tags(&["math", "calculus", "differentiation", "derivatives"])
                .with_estimated_minutes(60),
        )
        .with_objective(
            LearningObjective::new("math_integration", "Integration")
                .with_description(
                    "Evaluate definite and indefinite integrals by substitution, integration \
                     by parts, and partial fractions; apply the Fundamental Theorem.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Intermediate)
                .with_prerequisites(&["math_differentiation"])
                .with_tags(&["math", "calculus", "integration", "antiderivatives"])
                .with_estimated_minutes(60),
        )
        .with_objective(
            LearningObjective::new("math_multivariable_calculus", "Multivariable Calculus")
                .with_description(
                    "Compute partial derivatives, gradients, divergence, and curl; apply \
                 the chain rule and Jacobian to multivariable functions.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Intermediate)
                .with_prerequisites(&["math_differentiation", "math_matrices_linear_algebra"])
                .with_tags(&[
                    "math",
                    "calculus",
                    "gradient",
                    "jacobian",
                    "partial-derivatives",
                ])
                .with_estimated_minutes(75),
        )
        .with_objective(
            LearningObjective::new("math_bayes_theorem", "Bayes' Theorem")
                .with_description(
                    "Apply Bayes' theorem to update prior beliefs with evidence; reason \
                     about posterior distributions in discrete settings.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Intermediate)
                .with_prerequisites(&["math_basic_probability"])
                .with_tags(&["math", "probability", "bayes", "posterior", "inference"])
                .with_estimated_minutes(45),
        )
        .with_objective(
            LearningObjective::new("math_hypothesis_testing", "Hypothesis Testing")
                .with_description(
                    "Formulate null and alternative hypotheses; apply t-tests, chi-square, \
                     and ANOVA; interpret p-values and confidence intervals.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Intermediate)
                .with_prerequisites(&["math_descriptive_statistics", "math_bayes_theorem"])
                .with_tags(&["math", "statistics", "hypothesis", "p-value", "anova"])
                .with_estimated_minutes(60),
        )
        .with_objective(
            LearningObjective::new("math_complex_numbers", "Complex Numbers")
                .with_description(
                    "Perform arithmetic on complex numbers in rectangular and polar form; \
                     compute roots of unity using De Moivre's theorem.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Intermediate)
                .with_prerequisites(&["math_quadratic_equations", "math_basic_exponents"])
                .with_tags(&["math", "complex", "polar", "de-moivre", "roots-of-unity"])
                .with_estimated_minutes(45),
        )
        .with_objective(
            LearningObjective::new("math_recurrence_relations", "Recurrence Relations")
                .with_description(
                    "Solve linear recurrence relations by characteristic roots; identify \
                     closed forms for Fibonacci-style and divide-and-conquer recurrences.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Intermediate)
                .with_prerequisites(&["math_quadratic_equations", "math_logic_propositional"])
                .with_tags(&["math", "recurrence", "fibonacci", "characteristic-roots"])
                .with_estimated_minutes(45),
        )
        .with_objective(
            LearningObjective::new("math_information_theory_basics", "Information Theory")
                .with_description(
                    "Compute Shannon entropy, KL divergence, and mutual information; \
                     understand channel capacity and the noiseless coding theorem.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Intermediate)
                .with_prerequisites(&["math_basic_probability", "math_bayes_theorem"])
                .with_tags(&["math", "information-theory", "entropy", "kl-divergence"])
                .with_estimated_minutes(50),
        )
        // ── Tier 4: Advanced ────────────────────────────────────────────
        .with_objective(
            LearningObjective::new("math_svd_matrix_decomp", "SVD and Matrix Decompositions")
                .with_description(
                    "Compute singular-value decomposition, QR factorisation, and Cholesky \
                     decomposition; apply SVD to low-rank approximation and PCA.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Advanced)
                .with_prerequisites(TIER3_IDS)
                .with_tags(&["math", "linear-algebra", "svd", "pca", "qr"])
                .with_estimated_minutes(75),
        )
        .with_objective(
            LearningObjective::new("math_convex_optimization", "Convex Optimisation")
                .with_description(
                    "Identify convex sets and functions; apply gradient descent, Newton's \
                     method, and Lagrangian duality; understand KKT conditions.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Advanced)
                .with_prerequisites(&[
                    "math_multivariable_calculus",
                    "math_eigenvalues_eigenvectors",
                ])
                .with_tags(&[
                    "math",
                    "optimisation",
                    "gradient-descent",
                    "kkt",
                    "lagrangian",
                ])
                .with_estimated_minutes(75),
        )
        .with_objective(
            LearningObjective::new("math_ode_systems", "Ordinary Differential Equations")
                .with_description(
                    "Solve first- and second-order ODEs analytically; analyse phase \
                     portraits of 2D autonomous systems; apply Laplace transforms.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Advanced)
                .with_prerequisites(&["math_integration", "math_eigenvalues_eigenvectors"])
                .with_tags(&["math", "ode", "differential-equations", "laplace"])
                .with_estimated_minutes(75),
        )
        .with_objective(
            LearningObjective::new("math_proof_techniques", "Mathematical Proof Techniques")
                .with_description(
                    "Write rigorous mathematical proofs using direct proof, contrapositive, \
                     contradiction, induction, and structural induction.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Advanced)
                .with_prerequisites(&["math_logic_propositional", "math_recurrence_relations"])
                .with_tags(&["math", "proofs", "induction", "logic", "rigour"])
                .with_estimated_minutes(60),
        )
        .with_objective(
            LearningObjective::new(
                "math_constraint_satisfaction",
                "Constraint Satisfaction Problems",
            )
            .with_description(
                "Model and solve constraint satisfaction problems (CSPs); apply \
                 arc-consistency, backtracking, and local search.",
            )
            .with_domain(Domain::Mathematics)
            .with_difficulty(Difficulty::Advanced)
            .with_prerequisites(&["math_graph_intro", "math_logic_propositional"])
            .with_tags(&[
                "math",
                "csp",
                "constraint",
                "backtracking",
                "arc-consistency",
            ])
            .with_estimated_minutes(60),
        )
        .with_objective(
            LearningObjective::new("math_stochastic_processes", "Stochastic Processes")
                .with_description(
                    "Define and analyse discrete-time Markov chains; compute stationary \
                     distributions; understand Poisson processes and Brownian motion.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Advanced)
                .with_prerequisites(&["math_bayes_theorem", "math_information_theory_basics"])
                .with_tags(&["math", "stochastic", "markov", "brownian", "poisson"])
                .with_estimated_minutes(75),
        )
        .with_objective(
            LearningObjective::new(
                "math_abstract_algebra_groups",
                "Abstract Algebra: Groups and Rings",
            )
            .with_description(
                "Understand groups, rings, and fields; apply Lagrange's theorem, \
                 homomorphisms, and quotient groups.",
            )
            .with_domain(Domain::Mathematics)
            .with_difficulty(Difficulty::Advanced)
            .with_prerequisites(&["math_proof_techniques", "math_set_theory_basics"])
            .with_tags(&["math", "abstract-algebra", "groups", "rings", "fields"])
            .with_estimated_minutes(75),
        )
        .with_objective(
            LearningObjective::new("math_topology_intro", "Introduction to Topology")
                .with_description(
                    "Define topological spaces, open sets, and continuity; compute \
                     fundamental groups and understand connectedness and compactness.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Advanced)
                .with_prerequisites(&["math_proof_techniques", "math_multivariable_calculus"])
                .with_tags(&["math", "topology", "continuity", "compactness", "homotopy"])
                .with_estimated_minutes(75),
        )
        .with_objective(
            LearningObjective::new("math_numerical_methods", "Numerical Methods")
                .with_description(
                    "Implement Newton–Raphson root finding, bisection, finite differences, \
                     Runge–Kutta integration, and Newton–Cotes quadrature.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Advanced)
                .with_prerequisites(&[
                    "math_differentiation",
                    "math_integration",
                    "math_matrices_linear_algebra",
                ])
                .with_tags(&[
                    "math",
                    "numerical",
                    "newton-raphson",
                    "runge-kutta",
                    "quadrature",
                ])
                .with_estimated_minutes(75),
        )
        .with_objective(
            LearningObjective::new("math_ml_math_foundations", "Mathematics for ML")
                .with_description(
                    "Derive gradient descent, backpropagation, cross-entropy loss, and \
                     softmax from first principles; understand the bias–variance tradeoff.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Advanced)
                .with_prerequisites(&[
                    "math_convex_optimization",
                    "math_svd_matrix_decomp",
                    "math_hypothesis_testing",
                ])
                .with_tags(&[
                    "math",
                    "ml",
                    "gradient-descent",
                    "backprop",
                    "bias-variance",
                ])
                .with_estimated_minutes(90),
        )
        // ── Tier 5: Expert ──────────────────────────────────────────────
        .with_objective(
            LearningObjective::new("math_fourier_analysis", "Fourier Analysis")
                .with_description(
                    "Derive the Fourier series and Fourier transform; apply DFT/FFT to \
                     signal processing; understand convolution theorem and Parseval's identity.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Expert)
                .with_prerequisites(TIER4_IDS)
                .with_tags(&["math", "fourier", "fft", "signal-processing", "convolution"])
                .with_estimated_minutes(90),
        )
        .with_objective(
            LearningObjective::new("math_wavelet_transforms", "Wavelet Transforms")
                .with_description(
                    "Construct wavelet bases (Haar, Daubechies); perform multi-resolution \
                     analysis; apply wavelets to compression and denoising.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Expert)
                .with_prerequisites(&["math_fourier_analysis"])
                .with_tags(&[
                    "math",
                    "wavelets",
                    "haar",
                    "multi-resolution",
                    "compression",
                ])
                .with_estimated_minutes(90),
        )
        .with_objective(
            LearningObjective::new("math_numerical_stability", "Numerical Stability")
                .with_description(
                    "Analyse floating-point rounding, condition numbers, and backward \
                     stability; apply pivoting strategies and iterative refinement.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Expert)
                .with_prerequisites(&["math_numerical_methods", "math_svd_matrix_decomp"])
                .with_tags(&[
                    "math",
                    "numerical",
                    "floating-point",
                    "condition-number",
                    "stability",
                ])
                .with_estimated_minutes(75),
        )
        .with_objective(
            LearningObjective::new("math_formal_verification", "Formal Verification")
                .with_description(
                    "Encode mathematical theorems in a proof assistant (Lean/Coq); prove \
                     properties of algorithms using type theory and dependent types.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Expert)
                .with_prerequisites(&["math_proof_techniques", "math_abstract_algebra_groups"])
                .with_tags(&["math", "formal-verification", "lean", "coq", "type-theory"])
                .with_estimated_minutes(120),
        )
        .with_objective(
            LearningObjective::new("math_category_theory", "Category Theory")
                .with_description(
                    "Define categories, functors, and natural transformations; understand \
                     limits, colimits, and adjoint functors.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Expert)
                .with_prerequisites(&["math_abstract_algebra_groups", "math_topology_intro"])
                .with_tags(&["math", "category-theory", "functors", "adjunctions"])
                .with_estimated_minutes(120),
        )
        .with_objective(
            LearningObjective::new("math_manifold_learning", "Manifold Learning")
                .with_description(
                    "Analyse smooth manifolds, tangent spaces, and Riemannian metrics; \
                     apply geodesic distance and parallel transport to ML embeddings.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Expert)
                .with_prerequisites(&["math_topology_intro", "math_convex_optimization"])
                .with_tags(&["math", "manifolds", "riemannian", "geodesic", "embedding"])
                .with_estimated_minutes(120),
        )
        .with_objective(
            LearningObjective::new("math_functional_analysis", "Functional Analysis")
                .with_description(
                    "Study Hilbert and Banach spaces, bounded linear operators, and spectral \
                     theory; apply to PDEs and kernel methods.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Expert)
                .with_prerequisites(&["math_fourier_analysis", "math_topology_intro"])
                .with_tags(&[
                    "math",
                    "functional-analysis",
                    "hilbert",
                    "banach",
                    "operator",
                ])
                .with_estimated_minutes(120),
        )
        .with_objective(
            LearningObjective::new("math_measure_theory", "Measure Theory")
                .with_description(
                    "Define sigma-algebras, measures, and Lebesgue integration; understand \
                     convergence theorems and the Radon–Nikodym derivative.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Expert)
                .with_prerequisites(&["math_proof_techniques", "math_stochastic_processes"])
                .with_tags(&[
                    "math",
                    "measure-theory",
                    "lebesgue",
                    "sigma-algebra",
                    "radon-nikodym",
                ])
                .with_estimated_minutes(120),
        )
        .with_objective(
            LearningObjective::new(
                "math_algorithmic_complexity",
                "Algorithmic Complexity Theory",
            )
            .with_description(
                "Classify problems in P, NP, and beyond; prove hardness via reductions; \
                 understand randomised and approximation algorithms.",
            )
            .with_domain(Domain::Mathematics)
            .with_difficulty(Difficulty::Expert)
            .with_prerequisites(&["math_constraint_satisfaction", "math_proof_techniques"])
            .with_tags(&["math", "complexity", "np", "reductions", "approximation"])
            .with_estimated_minutes(90),
        )
        .with_objective(
            LearningObjective::new("math_symbolic_regression", "Symbolic Regression")
                .with_description(
                    "Apply genetic programming and sparse regression (SINDy) to recover \
                     symbolic closed-form expressions from numerical data.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Expert)
                .with_prerequisites(&["math_ml_math_foundations", "math_numerical_stability"])
                .with_tags(&[
                    "math",
                    "symbolic-regression",
                    "sindy",
                    "genetic-programming",
                ])
                .with_estimated_minutes(90),
        )
        .build()
}

/// Build the advanced mathematics curriculum.
///
/// Requires the base `mathematics` curriculum and adds five cross-domain
/// synthesis objectives that connect mathematical theory to Symthaea's own
/// architecture.
pub fn math_curriculum_advanced() -> Curriculum {
    Curriculum::new(ADVANCED_CURRICULUM_ID, "Advanced Mathematics")
        .with_description(
            "Expert-level mathematical synthesis: HDC geometry, consciousness Phi \
             from information geometry, category-theoretic program semantics, \
             differential privacy, and algebraic topology of neural representations.",
        )
        .with_prerequisite_curriculum(CURRICULUM_ID)
        .with_tags(&["math", "advanced", "expert", "synthesis"])
        .with_objective(
            LearningObjective::new(
                "math_hdc_geometry",
                "Geometry of Hyperdimensional Computing",
            )
            .with_description(
                "Derive concentration-of-measure results for random binary hypervectors; \
                     understand why Hamming distance approximates cosine similarity at high \
                     dimension and its implications for HDC binding capacity.",
            )
            .with_domain(Domain::Mathematics)
            .with_difficulty(Difficulty::Expert)
            .with_tags(&[
                "math",
                "hdc",
                "measure-concentration",
                "high-dimensional",
                "geometry",
            ])
            .with_estimated_minutes(90),
        )
        .with_objective(
            LearningObjective::new(
                "math_information_geometry_phi",
                "Information Geometry and IIT Phi",
            )
            .with_description(
                "Derive integrated information Φ as a minimum-information partition; \
                 connect Fisher information geometry to consciousness measures; \
                 understand why Phi is NP-hard to compute exactly.",
            )
            .with_domain(Domain::Mathematics)
            .with_difficulty(Difficulty::Expert)
            .with_tags(&[
                "math",
                "iit",
                "phi",
                "information-geometry",
                "consciousness",
            ])
            .with_estimated_minutes(120),
        )
        .with_objective(
            LearningObjective::new(
                "math_categorical_program_semantics",
                "Categorical Program Semantics",
            )
            .with_description(
                "Model programming languages as Cartesian closed categories; \
                 understand denotational semantics, monads as computational effects, \
                 and the Curry–Howard–Lambek correspondence.",
            )
            .with_domain(Domain::Mathematics)
            .with_difficulty(Difficulty::Expert)
            .with_tags(&[
                "math",
                "category-theory",
                "denotational",
                "monads",
                "curry-howard",
            ])
            .with_estimated_minutes(120),
        )
        .with_objective(
            LearningObjective::new("math_differential_privacy", "Differential Privacy Theory")
                .with_description(
                    "Define (ε, δ)-differential privacy formally; prove the Laplace and \
                     Gaussian mechanisms; analyse privacy-utility tradeoffs via Rényi \
                     divergence.",
                )
                .with_domain(Domain::Mathematics)
                .with_difficulty(Difficulty::Expert)
                .with_tags(&[
                    "math",
                    "privacy",
                    "differential-privacy",
                    "laplace",
                    "renyi",
                ])
                .with_estimated_minutes(90),
        )
        .with_objective(
            LearningObjective::new(
                "math_algebraic_topology_neural",
                "Algebraic Topology of Neural Representations",
            )
            .with_description(
                "Apply persistent homology and Betti numbers to the activation \
                 manifolds of neural networks; connect topological data analysis \
                 to concept geometry in HDC.",
            )
            .with_domain(Domain::Mathematics)
            .with_difficulty(Difficulty::Expert)
            .with_tags(&[
                "math",
                "topology",
                "persistent-homology",
                "betti",
                "neural",
                "tda",
            ])
            .with_estimated_minutes(120),
        )
        .build()
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── MathCurriculum tests ─────────────────────────────────────────────

    #[test]
    fn test_math_curriculum_objective_count() {
        let mc = MathCurriculum::new();
        assert!(
            mc.objectives.len() >= 50,
            "expected at least 50 objectives, got {}",
            mc.objectives.len()
        );
    }

    #[test]
    fn test_all_indices_match_position() {
        let mc = MathCurriculum::new();
        for (pos, obj) in mc.objectives.iter().enumerate() {
            assert_eq!(
                obj.id, pos,
                "objective '{}' has id {} but is at position {}",
                obj.name, obj.id, pos
            );
        }
    }

    #[test]
    fn test_tier1_has_no_prerequisites_outside_tier() {
        let mc = MathCurriculum::new();
        let tier1_ids: HashSet<usize> = (0..10).collect();
        for obj in mc.objectives.iter().take(10) {
            for prereq in &obj.prerequisites {
                assert!(
                    tier1_ids.contains(prereq) || *prereq < obj.id,
                    "Tier 1 objective '{}' has cross-tier prerequisite index {}",
                    obj.name,
                    prereq
                );
            }
        }
    }

    #[test]
    fn test_prerequisites_are_valid_indices() {
        let mc = MathCurriculum::new();
        let total = mc.objectives.len();
        for obj in &mc.objectives {
            for prereq in &obj.prerequisites {
                assert!(
                    *prereq < total,
                    "objective '{}' has out-of-bounds prerequisite index {}",
                    obj.name,
                    prereq
                );
            }
        }
    }

    #[test]
    fn test_difficulty_filtering() {
        let mc = MathCurriculum::new();
        let beginners = mc.objectives_at_level(MathDifficulty::Beginner);
        let experts = mc.objectives_at_level(MathDifficulty::Expert);

        assert_eq!(beginners.len(), 10, "expected 10 Beginner objectives");
        assert_eq!(experts.len(), 10, "expected 10 Expert objectives");

        for obj in &beginners {
            assert_eq!(obj.difficulty, MathDifficulty::Beginner);
        }
        for obj in &experts {
            assert_eq!(obj.difficulty, MathDifficulty::Expert);
        }
    }

    #[test]
    fn test_mastery_percentage_empty() {
        let mc = MathCurriculum::new();
        assert!(
            (mc.mastery_percentage() - 0.0).abs() < f64::EPSILON,
            "expected 0.0 mastery on a fresh curriculum"
        );
    }

    #[test]
    fn test_mastery_percentage_full() {
        let mut mc = MathCurriculum::new();
        for obj in &mut mc.objectives {
            obj.mastered = true;
        }
        assert!(
            (mc.mastery_percentage() - 1.0).abs() < f64::EPSILON,
            "expected 1.0 mastery when all objectives mastered"
        );
    }

    #[test]
    fn test_prerequisites_met() {
        let mc = MathCurriculum::new();

        // Objective 0 has no prerequisites — always met.
        let empty: HashSet<usize> = HashSet::new();
        assert!(mc.prerequisites_met(0, &empty));

        // Objective 6 (math_simple_equations) requires 5 (math_intro_variables).
        assert!(!mc.prerequisites_met(6, &empty));
        let with_5: HashSet<usize> = [5].iter().copied().collect();
        assert!(mc.prerequisites_met(6, &with_5));
    }

    #[test]
    fn test_next_objectives_starts_with_tier1() {
        let mc = MathCurriculum::new();
        let mastered: HashSet<usize> = HashSet::new();
        let next = mc.next_objectives(&mastered);

        // All Tier 1 objectives should be available from the start.
        let next_ids: HashSet<usize> = next.iter().map(|o| o.id).collect();
        for id in 0..10 {
            // Objectives with intra-tier prerequisites may not all be available,
            // but at least the prerequisite-free Tier 1 objectives must be.
            let obj = &mc.objectives[id];
            if obj.prerequisites.is_empty() {
                assert!(
                    next_ids.contains(&id),
                    "Tier 1 objective '{}' (id={}) should be available with no mastery",
                    obj.name,
                    id
                );
            }
        }
    }

    #[test]
    fn test_encode_progress_zero_when_nothing_mastered() {
        let mc = MathCurriculum::new();
        let hv = mc.encode_progress();
        assert_eq!(
            hv,
            BinaryHV::zero(),
            "encode_progress should return zero vector when no objectives mastered"
        );
    }

    #[test]
    fn test_encode_progress_nonzero_when_mastered() {
        let mut mc = MathCurriculum::new();
        mc.objectives[0].mastered = true;
        mc.objectives[1].mastered = true;
        let hv = mc.encode_progress();
        assert_ne!(
            hv,
            BinaryHV::zero(),
            "encode_progress should return non-zero vector when objectives are mastered"
        );
    }

    // ── Curriculum builder tests ─────────────────────────────────────────

    #[test]
    fn test_math_curriculum_builder_count() {
        let c = math_curriculum();
        assert_eq!(
            c.objectives.len(),
            50,
            "expected 50 objectives in builder curriculum"
        );
    }

    #[test]
    fn test_math_curriculum_advanced_count() {
        let c = math_curriculum_advanced();
        assert_eq!(
            c.objectives.len(),
            5,
            "expected 5 objectives in advanced curriculum"
        );
    }

    #[test]
    fn test_all_builder_ids_unique() {
        let c = math_curriculum();
        let ids: HashSet<&str> = c.objectives.iter().map(|o| o.id.as_str()).collect();
        assert_eq!(
            ids.len(),
            c.objectives.len(),
            "duplicate objective IDs in base curriculum"
        );
    }

    #[test]
    fn test_advanced_requires_base_curriculum() {
        let adv = math_curriculum_advanced();
        assert!(
            adv.prerequisite_curricula
                .contains(&CURRICULUM_ID.to_string()),
            "advanced curriculum should require base mathematics curriculum"
        );
    }

    #[test]
    fn test_all_builder_objectives_have_tags() {
        let base = math_curriculum();
        let adv = math_curriculum_advanced();
        for obj in base.objectives.iter().chain(adv.objectives.iter()) {
            assert!(
                !obj.tags.is_empty(),
                "objective '{}' should have at least one tag",
                obj.id
            );
        }
    }

    #[test]
    fn test_all_builder_objectives_have_estimated_time() {
        let base = math_curriculum();
        let adv = math_curriculum_advanced();
        for obj in base.objectives.iter().chain(adv.objectives.iter()) {
            assert!(
                obj.estimated_minutes > 0,
                "objective '{}' should have a positive estimated_minutes",
                obj.id
            );
        }
    }

    #[test]
    fn test_no_id_collision_across_curricula() {
        let base = math_curriculum();
        let adv = math_curriculum_advanced();
        let base_ids: HashSet<&str> = base.objectives.iter().map(|o| o.id.as_str()).collect();
        let adv_ids: HashSet<&str> = adv.objectives.iter().map(|o| o.id.as_str()).collect();
        let overlap: Vec<&&str> = base_ids.intersection(&adv_ids).collect();
        assert!(
            overlap.is_empty(),
            "ID collision across curricula: {:?}",
            overlap
        );
    }

    #[test]
    fn test_tier5_builder_objectives_are_expert() {
        let c = math_curriculum();
        let tier5_set: HashSet<&str> = TIER5_IDS.iter().copied().collect();
        for obj in &c.objectives {
            if tier5_set.contains(obj.id.as_str()) {
                assert_eq!(
                    obj.difficulty,
                    Difficulty::Expert,
                    "objective '{}' should be Expert difficulty",
                    obj.id
                );
            }
        }
    }

    #[test]
    fn test_difficulty_as_f32_range() {
        for d in [
            MathDifficulty::Beginner,
            MathDifficulty::Elementary,
            MathDifficulty::Intermediate,
            MathDifficulty::Advanced,
            MathDifficulty::Expert,
        ] {
            let v = d.as_f32();
            assert!(
                (0.0..=1.0).contains(&v),
                "MathDifficulty::{:?}.as_f32() = {} is out of [0,1]",
                d,
                v
            );
        }
    }

    #[test]
    fn test_encodings_are_deterministic() {
        let mc1 = MathCurriculum::new();
        let mc2 = MathCurriculum::new();
        for (a, b) in mc1.objectives.iter().zip(mc2.objectives.iter()) {
            assert_eq!(
                a.encoding, b.encoding,
                "encoding for '{}' is not deterministic",
                a.name
            );
        }
    }

    #[test]
    fn test_encodings_are_distinct() {
        let mc = MathCurriculum::new();
        let encodings: HashSet<_> = mc.objectives.iter().map(|o| o.encoding).collect();
        assert_eq!(
            encodings.len(),
            mc.objectives.len(),
            "two or more objectives share the same HDC encoding"
        );
    }
}
