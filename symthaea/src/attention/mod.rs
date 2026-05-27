// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Attention Module
//!
//! This module provides attention mechanisms including:
//! - Phi-guided attention for consciousness-aware information flow
//!
//! ## Key Types
//!
//! - [`PhiAttentionGate`] - IIT Phi-based attention gating
//! - [`PhiAttentionConfig`] - Configuration for Phi attention

pub mod phi_attention;

pub use phi_attention::{
    PhiAttentionConfig, PhiAttentionGate, PhiAttentionResult, compute_attention_weights,
};
