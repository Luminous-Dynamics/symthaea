// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! # symtropy-bevy-scene
//!
//! Opinionated scene scaffolding for Bevy 0.18 apps. Adds a single plugin
//! that sets up sensible defaults so demos don't repeat the
//! camera/light/clear-color boilerplate.
//!
//! ## Quick start
//!
//! ```ignore
//! use bevy::prelude::*;
//! use symtropy_bevy_scene::{SymtropyScenePlugin, fixed_camera};
//!
//! fn main() {
//!     App::new()
//!         .add_plugins(DefaultPlugins)
//!         .add_plugins(SymtropyScenePlugin::default())
//!         .add_systems(Startup, setup)
//!         .run();
//! }
//!
//! fn setup(mut commands: Commands) {
//!     commands.spawn(fixed_camera(Vec3::new(0.0, 1.5, 6.0), Vec3::ZERO));
//!     // SymtropyScenePlugin already inserted ClearColor, GlobalAmbientLight,
//!     // and spawned a sun directional light.
//! }
//! ```
//!
//! ## What the plugin does
//!
//! - Inserts a dark `ClearColor` resource (cool indigo background).
//! - Inserts `GlobalAmbientLight` with cool tint and moderate brightness so
//!   shadowed sides don't go fully black.
//! - Spawns one `DirectionalLight` (the "sun") at a stage-lighting angle.
//!
//! Every value is overridable via `SymtropyScenePlugin::with_config`. If you
//! want zero defaults and only the helpers, use the `fixed_camera` /
//! `fixed_light` functions directly without the plugin.

use bevy::prelude::*;

/// Configuration for [`SymtropyScenePlugin`].
///
/// All fields have sensible defaults; override only what you need:
///
/// ```ignore
/// SymtropyScenePlugin::with_config(SymtropySceneConfig {
///     clear_color: Color::BLACK,
///     ..default()
/// })
/// ```
#[derive(Clone, Debug)]
pub struct SymtropySceneConfig {
    /// Window background colour. Default: dark cool indigo.
    pub clear_color: Color,
    /// Ambient light tint. Default: cool blue-white.
    pub ambient_color: Color,
    /// Ambient light intensity. Default: 200.0 (cd/m²).
    pub ambient_brightness: f32,
    /// Whether to spawn the default directional sun light. Default: `true`.
    pub spawn_sun: bool,
    /// Sun light intensity. Default: 8000 lux.
    pub sun_illuminance: f32,
    /// Sun light tint. Default: warm white.
    pub sun_color: Color,
    /// Sun rotation as Euler angles (XYZ radians). Default upper-front-right.
    pub sun_euler: Vec3,
}

impl Default for SymtropySceneConfig {
    fn default() -> Self {
        Self {
            clear_color: Color::srgb(0.04, 0.04, 0.06),
            ambient_color: Color::srgb(0.8, 0.85, 1.0),
            ambient_brightness: 200.0,
            spawn_sun: true,
            sun_illuminance: 8_000.0,
            sun_color: Color::linear_rgb(1.0, 0.98, 0.92),
            sun_euler: Vec3::new(-0.7, 0.5, 0.0),
        }
    }
}

/// Drop-in scene scaffolding plugin. See crate docs for what it sets up.
#[derive(Default, Clone)]
pub struct SymtropyScenePlugin {
    config: SymtropySceneConfig,
}

impl SymtropyScenePlugin {
    /// Create a plugin instance with custom configuration.
    pub fn with_config(config: SymtropySceneConfig) -> Self {
        Self { config }
    }
}

pub mod loader;

pub use loader::{SymtropyScene, SymtropySceneLoader};

impl Plugin for SymtropyScenePlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<SymtropyScene>()
            .init_asset_loader::<SymtropySceneLoader>();

        let cfg = self.config.clone();
        app.insert_resource(ClearColor(cfg.clear_color))
            .insert_resource(bevy::light::GlobalAmbientLight {
                color: cfg.ambient_color,
                brightness: cfg.ambient_brightness,
                ..default()
            });
        if cfg.spawn_sun {
            let sun = SunConfig {
                illuminance: cfg.sun_illuminance,
                color: cfg.sun_color,
                euler: cfg.sun_euler,
            };
            app.insert_resource(sun);
            app.add_systems(Startup, spawn_sun_system);
        }
    }
}

#[derive(Resource, Clone)]
struct SunConfig {
    illuminance: f32,
    color: Color,
    euler: Vec3,
}

fn spawn_sun_system(mut commands: Commands, sun: Res<SunConfig>) {
    commands.spawn((
        DirectionalLight {
            illuminance: sun.illuminance,
            color: sun.color,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(
            EulerRot::XYZ,
            sun.euler.x,
            sun.euler.y,
            sun.euler.z,
        )),
    ));
}

/// One-line camera spawn: `commands.spawn(fixed_camera(pos, target))`.
///
/// Returns a `Camera3d` + `Transform::looking_at(target, Vec3::Y)`.
/// For 2D scenes use bare `commands.spawn(Camera2d)` directly — this
/// helper is 3D-specific.
pub fn fixed_camera(position: Vec3, target: Vec3) -> impl Bundle {
    (
        Camera3d::default(),
        Transform::from_translation(position).looking_at(target, Vec3::Y),
    )
}

/// One-line directional light spawn for cases where you want a second sun
/// (e.g., a fill light) on top of `SymtropyScenePlugin`'s default sun.
pub fn fixed_light(illuminance: f32, color: Color, euler_xyz: Vec3) -> impl Bundle {
    (
        DirectionalLight {
            illuminance,
            color,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(
            EulerRot::XYZ,
            euler_xyz.x,
            euler_xyz.y,
            euler_xyz.z,
        )),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_default_is_sane() {
        let cfg = SymtropySceneConfig::default();
        assert!(cfg.spawn_sun);
        assert!(cfg.sun_illuminance > 0.0);
        assert!(cfg.ambient_brightness > 0.0);
    }

    #[test]
    fn plugin_default_constructs() {
        let _plugin = SymtropyScenePlugin::default();
    }

    #[test]
    fn plugin_with_config_constructs() {
        let _plugin = SymtropyScenePlugin::with_config(SymtropySceneConfig {
            spawn_sun: false,
            ..default()
        });
    }

    #[test]
    fn plugin_inserts_clear_color() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_plugins(bevy::asset::AssetPlugin::default());
        app.add_plugins(SymtropyScenePlugin::default());
        assert!(app.world().contains_resource::<ClearColor>());
    }

    #[test]
    fn plugin_inserts_ambient_light() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_plugins(bevy::asset::AssetPlugin::default());
        app.add_plugins(SymtropyScenePlugin::default());
        assert!(
            app.world()
                .contains_resource::<bevy::light::GlobalAmbientLight>()
        );
    }

    #[test]
    fn plugin_with_no_sun_skips_sun_resource() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_plugins(bevy::asset::AssetPlugin::default());
        app.add_plugins(SymtropyScenePlugin::with_config(SymtropySceneConfig {
            spawn_sun: false,
            ..default()
        }));
        assert!(!app.world().contains_resource::<SunConfig>());
    }
}
