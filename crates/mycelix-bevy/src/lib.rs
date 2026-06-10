// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bevy SDK for building 3D/spatial Mycelix extensions.
//!
//! Provides a [`MycelixPlugin`] that adds consciousness state tracking,
//! data flow events, and cluster node components to a Bevy app. This is
//! a foundation crate -- it handles the ECS wiring so extension authors
//! can focus on 3D visualization and interaction.
//!
//! # Example
//!
//! ```rust,ignore
//! use bevy_app::prelude::*;
//! use bevy_ecs::prelude::*;
//! use mycelix_bevy::{MycelixPlugin, ConsciousnessResource, ClusterNode};
//!
//! fn main() {
//!     App::new()
//!         .add_plugins(MycelixPlugin)
//!         .add_systems(Startup, spawn_clusters)
//!         .run();
//! }
//!
//! fn spawn_clusters(mut commands: Commands) {
//!     commands.spawn(ClusterNode {
//!         cluster_id: "commons".into(),
//!         display_name: "Commons".into(),
//!         bio_name: "Mycelium".into(),
//!     });
//! }
//! ```

use bevy_app::prelude::*;
use bevy_color::Color;
use bevy_ecs::prelude::*;
use bevy_log::info;
use mycelix_cluster_manifest::{BridgeDirection, CivicTier, DataSensitivity};

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

/// Bevy plugin that registers Mycelix resources, events, and systems.
///
/// Adds:
/// - [`ConsciousnessResource`] (resource)
/// - [`DataFlowEvent`] (event)
/// - A system that logs tier changes each frame (debug builds only)
pub struct MycelixPlugin;

impl Plugin for MycelixPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ConsciousnessResource>()
            .add_event::<DataFlowEvent>()
            .add_systems(Update, log_tier_changes);
    }
}

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// Bevy resource wrapping the agent's consciousness state.
///
/// Updated at runtime from conductor bridge calls. Systems can read
/// this to gate UI elements or adjust visuals based on trust tier.
#[derive(Resource, Debug, Clone)]
pub struct ConsciousnessResource {
    /// Identity verification strength (0.0-1.0).
    pub identity: f64,
    /// Cross-hApp aggregated reputation (0.0-1.0).
    pub reputation: f64,
    /// Peer trust attestations (0.0-1.0).
    pub community: f64,
    /// Domain-specific engagement (0.0-1.0).
    pub engagement: f64,
    /// Derived trust tier.
    pub tier: CivicTier,
}

impl Default for ConsciousnessResource {
    fn default() -> Self {
        Self {
            identity: 0.0,
            reputation: 0.0,
            community: 0.0,
            engagement: 0.0,
            tier: CivicTier::Observer,
        }
    }
}

impl ConsciousnessResource {
    /// Combined score across all four dimensions.
    pub fn combined_score(&self) -> f64 {
        ((self.identity + self.reputation + self.community + self.engagement) / 4.0)
            .clamp(0.0, 1.0)
    }

    /// Recompute the tier from current dimension values.
    pub fn recompute_tier(&mut self) {
        let score = self.combined_score();
        self.tier = if score >= 0.8 {
            CivicTier::Guardian
        } else if score >= 0.6 {
            CivicTier::Steward
        } else if score >= 0.4 {
            CivicTier::Citizen
        } else if score >= 0.3 {
            CivicTier::Participant
        } else {
            CivicTier::Observer
        };
    }

    /// Check if the current tier meets a minimum requirement.
    pub fn meets_tier(&self, required: CivicTier) -> bool {
        self.tier >= required
    }
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// Component for representing a Mycelix cluster as an entity in 3D space.
///
/// Attach to an entity with a `Transform` and mesh to visualize a cluster
/// node in a network graph, globe, or other spatial layout.
#[derive(Component, Debug, Clone)]
pub struct ClusterNode {
    /// Cluster identifier (e.g., "commons", "health").
    pub cluster_id: String,
    /// Human-readable display name.
    pub display_name: String,
    /// Biological metaphor name (e.g., "Mycelium", "Homeostasis").
    pub bio_name: String,
}

/// Component for consciousness-driven color theming.
///
/// Attach to entities whose color should reflect the current trust tier.
/// A system can query for `DomainColor` entities and update their
/// material color based on the tier.
#[derive(Component, Debug, Clone)]
pub struct DomainColor {
    /// Base color for this domain element.
    pub base: Color,
    /// Glow/accent color.
    pub glow: Color,
    /// Whether to modulate opacity by consciousness score.
    pub consciousness_modulated: bool,
}

impl DomainColor {
    /// Create a new domain color pair.
    pub fn new(base: Color, glow: Color) -> Self {
        Self {
            base,
            glow,
            consciousness_modulated: false,
        }
    }

    /// Enable consciousness-based opacity modulation.
    pub fn with_modulation(mut self) -> Self {
        self.consciousness_modulated = true;
        self
    }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// Event fired when a data flow occurs between clusters via a bridge.
///
/// Systems can listen for these to animate data transfers in 3D,
/// update dashboards, or log sovereignty events.
#[derive(Event, Debug, Clone)]
pub struct DataFlowEvent {
    /// Source cluster ID.
    pub source_id: String,
    /// Target cluster ID.
    pub target_id: String,
    /// Direction of the flow.
    pub direction: BridgeDirection,
    /// Zome that initiated the flow.
    pub zome: String,
    /// Sensitivity of the data being transferred.
    pub sensitivity: DataSensitivity,
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// System that logs tier changes (debug only).
fn log_tier_changes(consciousness: Res<ConsciousnessResource>) {
    if consciousness.is_changed() && !consciousness.is_added() {
        info!(
            "Consciousness tier changed to {:?} (combined={:.3})",
            consciousness.tier,
            consciousness.combined_score()
        );
    }
}
