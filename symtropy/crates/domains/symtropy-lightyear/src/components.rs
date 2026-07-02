// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Networked components — the ECS state that Lightyear replicates.
//!
//! These are Bevy components that mirror Symtropy's physics state.
//! Lightyear handles serialization, prediction, rollback, and
//! interpolation for these automatically.

use bevy_ecs::prelude::*;
use bevy_reflect::Reflect;
use serde::{Deserialize, Serialize};

/// Replicated position (3D). Lightyear syncs this across peers.
#[derive(Component, Clone, Debug, Serialize, Deserialize, PartialEq, Reflect)]
pub struct NetPosition {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Replicated velocity. Used for prediction extrapolation.
#[derive(Component, Clone, Debug, Serialize, Deserialize, PartialEq, Reflect)]
pub struct NetVelocity {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Replicated rotation (quaternion).
#[derive(Component, Clone, Debug, Serialize, Deserialize, PartialEq, Reflect)]
pub struct NetRotation {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Replicated consciousness state for an entity.
#[derive(Component, Clone, Debug, Serialize, Deserialize, PartialEq, Reflect)]
pub struct NetConsciousness {
    /// Phi score (0.0 - 1.0).
    pub phi: f64,
    /// Energy budget remaining (Joules).
    pub energy: f64,
    /// Safety tier (0=Green, 1=Yellow, 2=Orange, 3=Red).
    pub safety_tier: u8,
    /// Whether this entity is in a sanctuary zone.
    pub in_sanctuary: bool,
}

/// Replicated player input (sent from client to authority peer).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PlayerInput {
    /// Movement direction (normalized).
    pub move_x: f64,
    pub move_y: f64,
    /// Whether the player is applying consciousness focus.
    pub focus: bool,
    /// Whether the player is attempting a harmony action.
    pub harmony: bool,
    /// Tick this input was generated on.
    pub tick: u64,
}

/// Authority marker — which peer is simulating this entity's physics.
#[derive(Component, Clone, Debug, Serialize, Deserialize, PartialEq, Reflect)]
pub struct NetAuthority {
    /// Peer ID of the authority.
    pub peer_id: u64,
    /// Whether authority can be transferred (e.g., proximity-based).
    pub transferable: bool,
}

/// Component identifying the spatial zone of an entity (Morton prefix).
#[derive(Component, Clone, Debug, Serialize, Deserialize, PartialEq, Reflect)]
pub struct SpatialZone(pub u64);

/// Update spatial authority and zones based on Morton codes.
pub fn update_spatial_authority<const D: usize>(
    mut query: Query<(&NetPosition, &NetAuthority, Option<&mut SpatialZone>)>,
) {
    // 1. Define world bounds (demo-fixed for now)
    let min = nalgebra::SVector::<f64, D>::from_element(-100.0);
    let max = nalgebra::SVector::<f64, D>::from_element(100.0);

    for (pos, _auth, zone) in query.iter_mut() {
        let mut p = nalgebra::SVector::<f64, D>::zeros();
        p[0] = pos.x;
        if D >= 2 {
            p[1] = pos.y;
        }
        if D >= 3 {
            p[2] = pos.z;
        }

        // 2. Compute Morton code
        let code = symtropy_physics::broadphase::morton_encode::<D>(&p, &min, &max);

        // 3. Extract 4-bit prefix for coarse spatial partitioning
        let prefix = symtropy_physics::broadphase::morton_prefix(code, 4);

        if let Some(mut z) = zone {
            z.0 = prefix;
        }

        // 4. In a real p2p setup, authority would follow the prefix
        // (e.g. peer_id = prefix % num_peers)
    }
}
