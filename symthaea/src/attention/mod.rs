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
    compute_attention_weights, softmax_with_temperature, PhiAttentionConfig, PhiAttentionGate,
    PhiAttentionResult,
};
