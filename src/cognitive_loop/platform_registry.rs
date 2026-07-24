// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! The single source of truth for robotics platform bridge construction.
//!
//! `build_platform_registry()` collects every feature-enabled
//! [`PlatformPlugin`](symthaea_core::embodiment::PlatformPlugin) into a
//! [`PlatformRegistry`](symthaea_core::embodiment::PlatformRegistry).
//! `switch_embodiment` dispatches through this registry — there is no other
//! construction path. Adding a platform is: implement `PlatformPlugin` in the
//! platform crate, add the feature-gated `register` line here. (Before
//! 2026-07-12 dispatch was a hand-written per-platform `match` in
//! `accessors/system.rs` while 16 `plugin.rs` implementations sat dead —
//! the fork the 2026-07-10 robotics plan's Tier 2.1 closed.)
//!
//! Deliberately NOT registered:
//! - `phone` — `PhoneBridge::new(serial, w, h)` needs a live ADB device
//!   serial and screen geometry; it cannot be constructed from a genesis
//!   seed alone, so it attaches through its own explicit path rather than
//!   pretending a deviceless default is a body.

use symthaea_core::embodiment::PlatformRegistry;

/// Build the registry of all feature-enabled platform plugins.
///
/// Cheap to construct (a `Vec` of unit structs) — callers may build it
/// per-dispatch rather than caching.
pub fn build_platform_registry() -> PlatformRegistry {
    #[allow(unused_mut)]
    let mut registry = PlatformRegistry::new();

    #[cfg(feature = "humanoid")]
    registry.register(Box::new(symthaea_humanoid::plugin::HumanoidPlugin));
    // `flight` is a backward-compatible alias feature that enables
    // `multirotor`, so gating on the canonical name covers both.
    #[cfg(feature = "multirotor")]
    registry.register(Box::new(symthaea_multirotor::plugin::MultirotorPlugin));
    #[cfg(feature = "helicopter")]
    registry.register(Box::new(symthaea_helicopter::plugin::HelicopterPlugin));
    #[cfg(feature = "vehicle")]
    registry.register(Box::new(symthaea_vehicle::plugin::VehiclePlugin));
    #[cfg(feature = "auv")]
    registry.register(Box::new(symthaea_auv::plugin::AuvPlugin));
    #[cfg(feature = "manipulator")]
    registry.register(Box::new(symthaea_manipulator::plugin::ManipulatorPlugin));
    #[cfg(feature = "exoskeleton")]
    registry.register(Box::new(symthaea_exoskeleton::plugin::ExoskeletonPlugin));
    #[cfg(feature = "surgical")]
    registry.register(Box::new(symthaea_surgical::plugin::SurgicalPlugin));
    #[cfg(feature = "orbital")]
    registry.register(Box::new(symthaea_orbital::plugin::OrbitalPlugin));
    #[cfg(feature = "quadruped")]
    registry.register(Box::new(symthaea_quadruped::plugin::QuadrupedPlugin));
    #[cfg(feature = "subterranean")]
    registry.register(Box::new(symthaea_subterranean::plugin::SubterraneanPlugin));
    #[cfg(feature = "infrastructure")]
    registry.register(Box::new(
        symthaea_infrastructure::plugin::InfrastructurePlugin,
    ));
    #[cfg(feature = "scavenger")]
    registry.register(Box::new(symthaea_scavenger::plugin::ScavengerPlugin));
    #[cfg(feature = "agribot")]
    registry.register(Box::new(symthaea_agribot::plugin::AgribotPlugin));
    #[cfg(feature = "biota")]
    registry.register(Box::new(symthaea_biota::plugin::BiotaPlugin));
    #[cfg(feature = "clime")]
    registry.register(Box::new(symthaea_clime::plugin::ClimePlugin));

    registry
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every registered plugin must construct a working bridge whose
    /// self-reported platform and actuator count match the plugin's
    /// metadata. Under default features the registry is empty and this
    /// passes trivially; the CI per-platform feature legs exercise it.
    #[test]
    fn every_registered_plugin_constructs_a_consistent_bridge() {
        let registry = build_platform_registry();
        let genesis = symthaea_core::genesis::GenesisSeed::from_phrase("registry-test");
        for platform in registry.registered_platforms() {
            let bridge = registry
                .create_bridge(platform, &genesis)
                .expect("registered platform must construct a bridge");
            assert_eq!(
                bridge.platform(),
                platform,
                "bridge self-reports a different platform than its plugin"
            );
            assert!(bridge.num_actuators() > 0);
        }
    }

    #[test]
    fn no_duplicate_platform_registrations() {
        let registry = build_platform_registry();
        let mut platforms = registry.registered_platforms();
        let before = platforms.len();
        platforms.sort_by_key(|p| format!("{p:?}"));
        platforms.dedup();
        assert_eq!(before, platforms.len(), "duplicate plugin registration");
    }

    #[cfg(feature = "manipulator")]
    #[test]
    fn manipulator_is_registered() {
        use symthaea_core::embodiment::EmbodimentPlatform;
        assert!(
            build_platform_registry()
                .registered_platforms()
                .contains(&EmbodimentPlatform::Manipulator)
        );
    }

    #[cfg(feature = "multirotor")]
    #[test]
    fn multirotor_is_registered_as_quadrotor_variant() {
        use symthaea_core::embodiment::EmbodimentPlatform;
        assert!(
            build_platform_registry()
                .registered_platforms()
                .contains(&EmbodimentPlatform::Quadrotor)
        );
    }
}
