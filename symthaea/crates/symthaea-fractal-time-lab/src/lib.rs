// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Symthaea Fractal Time Lab
//!
//! Experimental computational testbed for:
//! 1. scale-recursive spectral structure,
//! 2. temporal symmetry breaking / subharmonic persistence,
//! 3. multi-scale integration survival.
//!
//! Epistemic status:
//! - Exploratory benchmark crate.
//! - Not physical proof of fractal time, quantum consciousness, or cosmological recurrence.
//! - Intended to compare hypotheses against explicit null models.

pub mod floquet_time_crystal;
pub mod hofstadter;
pub mod metrics;
pub mod multiscale_phi;
pub mod null_models;
pub mod report;
pub mod runner;
pub mod semantic_stream_diagnostics;
pub mod topological_analysis;

pub use metrics::{
    ExperimentScorecard, FractalMetric, IntegrationSurvivalScore, SelfSimilarityScore,
    SubharmonicScore, effect_size, mean, scorecards_to_csv, scorecards_to_json_array, std_dev,
};

pub use report::{claim_scope_markdown, scorecards_to_markdown_report};
pub use runner::{BenchmarkConfig, BenchmarkRun, run_all_benchmarks};
