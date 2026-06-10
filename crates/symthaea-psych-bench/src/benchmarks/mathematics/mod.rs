// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mathematics benchmark suite.
//!
//! Thirteen benchmarks covering core mathematical cognition domains:
//! arithmetic, linear algebra, polynomial analysis, numerical integration,
//! matrix computation, statistical inference, Bayesian reasoning, logical
//! deduction, constraint satisfaction, proof construction, differential
//! equations, graph algorithms, and quantum circuit simulation.

pub mod arithmetic_word_problems;
pub mod bayesian_reasoning;
pub mod constraint_puzzles;
pub mod definite_integrals;
pub mod differential_equations;
pub mod graph_algorithms;
pub mod linear_system_solving;
pub mod logical_deduction;
pub mod matrix_operations;
pub mod polynomial_roots;
pub mod proof_construction;
pub mod quantum_circuits;
pub mod statistical_inference;

pub use arithmetic_word_problems::ArithmeticWordProblemsBenchmark;
pub use bayesian_reasoning::BayesianReasoningBenchmark;
pub use constraint_puzzles::ConstraintPuzzlesBenchmark;
pub use definite_integrals::DefiniteIntegralsBenchmark;
pub use differential_equations::DifferentialEquationsBenchmark;
pub use graph_algorithms::GraphAlgorithmsBenchmark;
pub use linear_system_solving::LinearSystemSolvingBenchmark;
pub use logical_deduction::LogicalDeductionBenchmark;
pub use matrix_operations::MatrixOperationsBenchmark;
pub use polynomial_roots::PolynomialRootsBenchmark;
pub use proof_construction::ProofConstructionBenchmark;
pub use quantum_circuits::QuantumCircuitsBenchmark;
pub use statistical_inference::StatisticalInferenceBenchmark;
