//! Meta-Cognitive Modules — Symthaea thinking about itself
//!
//! This module contains consciousness-aware analysis features:
//! - Self-analysis: Symthaea reads and understands its own source code
//! - Phi code quality: IIT-based code quality metrics
//! - Code exploration: Active inference for codebase navigation
//! - Dream synthesis: Background creative code recombination

#[cfg(feature = "code_generation")]
pub mod self_analysis;

#[cfg(feature = "code_generation")]
pub mod phi_code_quality;

#[cfg(feature = "code_generation")]
pub mod code_explorer;

#[cfg(feature = "code_generation")]
pub mod dream_synthesis;
