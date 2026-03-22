// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration with full Symthaea brain.
//!
//! This module provides the bridge types that connect symthaea-nix's NixOS
//! mind to the main Symthaea cognitive architecture:
//!
//! - `domain_plugin` — NixOS domain plugin for entity extraction and intent
//! - `actor_bridge` — NixOS state → Symthaea actor system messages
//! - `pipeline_integration` — NixOS cognition → Conscious Pipeline stages

pub mod actor_bridge;
pub mod domain_plugin;
pub mod pipeline_integration;

pub use actor_bridge::{
    CausalMindBridge, HippocampusBridge, NixActorBridge, NixActorMessage, NixActorRoles,
    NixActorState, NixMessageKind,
};
pub use domain_plugin::NixOsPlugin;
pub use pipeline_integration::{
    NixConsciousnessQuadrant, NixPipelineHook, NixPipelineProcessor, NixPipelineResult,
    NixPipelineStage,
};
