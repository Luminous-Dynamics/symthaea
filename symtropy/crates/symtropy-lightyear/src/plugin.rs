// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Symtropy Network Plugin — combines IrohIo + Lightyear + spatial authority.
//!
//! This is the single plugin games add to enable multiplayer:
//!
//! ```rust,ignore
//! App::new()
//!     .add_plugins(DefaultPlugins)
//!     .add_plugins(SymtropyNetPlugin)
//!     .run();
//! ```

use bevy_app::prelude::*;

use crate::iroh_io::IrohIoPlugin;
use crate::protocol;

/// Main networking plugin for Symtropy multiplayer.
///
/// Adds:
/// - IrohIo transport systems (PreUpdate/PostUpdate)
/// - Component registration for Lightyear replication
/// - Spatial authority zone management
///
/// Games should configure Lightyear separately (ClientPlugins/ServerPlugins)
/// and use this plugin alongside it.
pub struct SymtropyNetPlugin;

impl Plugin for SymtropyNetPlugin {
    fn build(&self, app: &mut App) {
        // Register networked component types
        protocol::register(app);

        // Add Iroh transport IO systems
        app.add_plugins(IrohIoPlugin);

        // Spatial authority management (FixedUpdate)
        app.add_systems(
            FixedUpdate,
            (
                crate::components::update_spatial_authority::<2>,
                crate::components::update_spatial_authority::<3>,
            ),
        );

        bevy_log::info!("Symtropy networking initialized (Lightyear + Iroh QUIC)");
    }
}
