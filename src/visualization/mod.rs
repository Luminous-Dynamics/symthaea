// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Attention Visualization Module
//!
//! This module provides tools for debugging and interpreting Phi-gated attention
//! decisions in the Symthaea system.
//!
//! ## Overview
//!
//! Understanding how attention flows through a consciousness-aware system is crucial
//! for debugging, analysis, and interpretability. This module provides:
//!
//! - **AttentionSnapshot**: Captures a single moment of attention state
//! - **AttentionHistory**: Tracks attention patterns over time
//! - **AttentionVisualizer**: Tools for rendering attention data
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::visualization::{AttentionVisualizer, AttentionSnapshot};
//! use symthaea::attention::{PhiAttentionGate, PhiAttentionConfig};
//!
//! let mut visualizer = AttentionVisualizer::new();
//! let mut gate = PhiAttentionGate::default_gate();
//!
//! // Run attention computation
//! let result = gate.forward(&inputs, &phi_values);
//!
//! // Capture snapshot with input names
//! let snapshot = AttentionSnapshot::from_result(
//!     &result,
//!     &phi_values,
//!     vec!["visual", "auditory", "semantic"],
//!     gate.config().temperature,
//! );
//!
//! // Record in history
//! visualizer.record(snapshot);
//!
//! // Export for analysis
//! println!("{}", visualizer.history().to_ascii_heatmap());
//! let json = visualizer.history().to_json().unwrap();
//! ```

pub mod attention_viz;

pub use attention_viz::{
    AttentionFlowEdge, AttentionFlowGraph, AttentionHistory, AttentionSnapshot, AttentionSummary,
    AttentionVisualizer,
};
