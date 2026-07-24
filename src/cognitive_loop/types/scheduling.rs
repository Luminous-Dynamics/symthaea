// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Scheduling types — urgency and cycle state.

// CycleUrgency moved to the symthaea-cognitive-types crate (2026-07-12) — a pure
// data type referenced by CycleMetadata and 50+ call sites across the crate.
pub use symthaea_cognitive_types::CycleUrgency;

/// Read-only snapshot of shared cycle state, passed to extracted phase functions
/// to replace loose multi-parameter signatures.
#[derive(Debug, Clone)]
pub(crate) struct CycleState<'a> {
    pub compressed_state: &'a [f32],
    pub output: &'a [f32],
    pub prediction_error: f32,
    pub coherence: f32,
    pub unified_psi: f64,
    pub phi_attention_weight: f32,
    pub hv16_cached: &'a symthaea_core::hdc::BinaryHV,
    pub input: &'a str,
    pub urgency: CycleUrgency,
    pub attention_budget_exceeded: bool,
    pub predictive_budget_gated: bool,
    /// Whether a visual scene was recognized this cycle (for dream salience boost).
    #[cfg(feature = "vision-manifold")]
    pub scene_recognized: bool,
    /// Semantic embedding from neural encoder (for episodic memory similarity).
    #[cfg(feature = "semantic-encoder")]
    pub semantic_embedding: Option<Vec<f32>>,
}
