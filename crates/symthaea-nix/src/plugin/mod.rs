//! Integration with full Symthaea brain.
//!
//! This module provides the bridge types that connect symthaea-nix's NixOS
//! mind to the main Symthaea cognitive architecture:
//!
//! - `domain_plugin` — NixOS domain plugin for entity extraction and intent
//! - `actor_bridge` — NixOS state → Symthaea actor system messages
//! - `pipeline_integration` — NixOS cognition → Conscious Pipeline stages

pub mod domain_plugin;
pub mod actor_bridge;
pub mod pipeline_integration;

pub use domain_plugin::NixOsPlugin;
pub use actor_bridge::{
    NixActorBridge, NixActorMessage, NixMessageKind, NixActorState,
    NixActorRoles, HippocampusBridge, CausalMindBridge,
};
pub use pipeline_integration::{
    NixPipelineProcessor, NixPipelineResult, NixPipelineStage,
    NixConsciousnessQuadrant, NixPipelineHook,
};
