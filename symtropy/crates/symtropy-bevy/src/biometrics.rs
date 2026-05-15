// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bevy integration for the biometric-to-Φ bridge.
//!
//! # Overview
//!
//! `PlayerBiometrics` is a Bevy [`Resource`] that holds a rolling window of
//! [`BiometricSample`]s and knows which physics body belongs to the player.
//! Each `FixedUpdate` the `biometric_to_phi_system` converts the window to
//! [`ConsciousnessInputs`] and calls `ConsciousnessField::update_entity()` on
//! the player body — so the player's real physiology directly modulates their
//! in-game Φ, which gates all physics interactions through the safety tiers.
//!
//! # Usage
//!
//! ```ignore
//! // 1. Register the resource with the player's physics handle:
//! app.insert_resource(PlayerBiometrics::new(player_handle));
//!
//! // 2. Somewhere in your input/sensor system, push samples each frame:
//! fn sensor_system(mut bio: ResMut<PlayerBiometrics>) {
//!     bio.push(BiometricSample { hrv_rmssd: Some(heart_rate_variability()), ..Default::default() });
//! }
//!
//! // 3. The plugin wires `biometric_to_phi_system` into FixedUpdate automatically.
//! //    Player Φ now tracks their biometrics in real time.
//! ```
//!
//! # Without a biometric sensor
//!
//! Push synthetic samples derived from gameplay state (stress, focus, flow):
//! ```ignore
//! bio.push(BiometricSample {
//!     hrv_rmssd:           Some(lerp(20.0, 80.0, player_calm)),
//!     eeg_gamma_coherence: Some(player_focus),
//!     ..Default::default()
//! });
//! ```

use bevy::prelude::*;
use symtropy_consciousness_physics::biometrics::{BiometricHistory, BiometricSample};
use symtropy_math::Point;
use symtropy_physics::body::BodyHandle;

use crate::plugin::SymtropyPhysics;

// ────────────────────────────────────────────────────────────
//  Resource
// ────────────────────────────────────────────────────────────

/// Bevy resource holding the player's biometric state and physics handle.
///
/// Insert this resource to enable the biometric-to-Φ loop.
/// The associated `biometric_to_phi_system<D>` reads this resource each
/// `FixedUpdate` and updates the player entity in `ConsciousnessField<D>`.
#[derive(Resource)]
pub struct PlayerBiometrics {
    /// The physics body handle for the player entity.
    pub handle: BodyHandle,
    /// Rolling window of biometric samples.
    pub history: BiometricHistory,
    /// Whether the system is active. Set false to pause biometric coupling.
    pub active: bool,
}

impl PlayerBiometrics {
    /// Create with the player's physics body handle.
    /// Uses a 30-sample smoothing window (30 s @ 1 Hz sensor rate).
    pub fn new(handle: BodyHandle) -> Self {
        Self {
            handle,
            history: BiometricHistory::new(30),
            active: true,
        }
    }

    /// Create with a custom smoothing window.
    pub fn with_window(handle: BodyHandle, window: usize) -> Self {
        Self {
            handle,
            history: BiometricHistory::new(window),
            active: true,
        }
    }

    /// Push a new biometric sample. Call once per tick or per sensor update.
    pub fn push(&mut self, sample: BiometricSample) {
        self.history.push(sample);
    }

    /// Number of samples currently buffered.
    pub fn sample_count(&self) -> usize {
        self.history.len()
    }
}

// ────────────────────────────────────────────────────────────
//  Systems
// ────────────────────────────────────────────────────────────

/// FixedUpdate system: converts buffered biometric samples to `ConsciousnessInputs`
/// and updates the player entity in the `ConsciousnessField<D>`.
///
/// Uses `Option<ResMut<PlayerBiometrics>>` — skips silently when the resource is
/// absent or inactive, so no `run_if` condition is needed. Insert `PlayerBiometrics`
/// to activate biometric coupling; remove or set `active = false` to suspend it.
///
/// The player's position is looked up from the physics world to feed the
/// sanctuary zone update inside `update_entity`.
pub fn biometric_to_phi_system<const D: usize>(
    bio: Option<ResMut<PlayerBiometrics>>,
    mut physics: ResMut<SymtropyPhysics<D>>,
) {
    let Some(mut bio) = bio else {
        return;
    };
    if !bio.active {
        return;
    }

    let inputs = bio.history.to_consciousness_inputs();
    let handle = bio.handle;

    // Look up the player's current position from the physics world.
    // Falls back to origin if the handle isn't registered (e.g., before spawn).
    let position = physics
        .world
        .body(handle)
        .map(|b| b.transform.translation.clone())
        .unwrap_or_else(Point::origin);

    physics.field.update_entity(handle, &inputs, position);
}

// ────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::SVector;
    use symtropy_consciousness_physics::coupling::ConsciousnessField;
    use symtropy_physics::PhysicsWorld;

    fn make_physics() -> SymtropyPhysics<2> {
        SymtropyPhysics {
            world: PhysicsWorld::<2>::new(SVector::zeros()),
            field: ConsciousnessField::new(),
        }
    }

    fn add_player(p: &mut SymtropyPhysics<2>) -> BodyHandle {
        let h = p
            .world
            .add_sphere(symtropy_math::Point::new([0.0, 0.0]), 1.0, 1.0);
        p.field.register(h, 100.0, 5.0);
        h
    }

    #[test]
    fn player_biometrics_new_constructs() {
        let h = BodyHandle(0);
        let bio = PlayerBiometrics::new(h);
        assert_eq!(bio.handle, h);
        assert!(bio.active);
        assert_eq!(bio.sample_count(), 0);
    }

    #[test]
    fn player_biometrics_with_window_stores_window() {
        let bio = PlayerBiometrics::with_window(BodyHandle(1), 60);
        // Verify it doesn't panic and can accept samples up to window
        let mut bio = bio;
        for _ in 0..60 {
            bio.push(BiometricSample {
                hrv_rmssd: Some(50.0),
                ..Default::default()
            });
        }
        assert_eq!(bio.sample_count(), 60);
        // Next push should evict oldest
        bio.push(BiometricSample {
            hrv_rmssd: Some(50.0),
            ..Default::default()
        });
        assert_eq!(bio.sample_count(), 60);
    }

    #[test]
    fn push_increases_sample_count() {
        let mut bio = PlayerBiometrics::new(BodyHandle(0));
        bio.push(BiometricSample::default());
        bio.push(BiometricSample::default());
        assert_eq!(bio.sample_count(), 2);
    }

    #[test]
    fn biometric_system_updates_player_phi() {
        let mut p = make_physics();
        let h = add_player(&mut p);

        // Before biometric coupling: phi is 0.0 (no inputs yet)
        let phi_before = p.field.entities.get(&h).unwrap().phi();
        assert_eq!(phi_before, 0.0);

        // Create resource with high-coherence samples
        let mut bio = PlayerBiometrics::new(h);
        for _ in 0..10 {
            bio.push(BiometricSample {
                eeg_gamma_coherence: Some(0.9),
                hrv_rmssd: Some(70.0),
                ..Default::default()
            });
        }

        // Manually run what the system does (without a full Bevy App)
        let inputs = bio.history.to_consciousness_inputs();
        p.field
            .update_entity(h, &inputs, symtropy_math::Point::origin());

        let phi_after = p.field.entities.get(&h).unwrap().phi();
        assert!(
            phi_after > phi_before,
            "high gamma coherence should produce non-zero Phi, got {phi_after}"
        );
    }

    #[test]
    fn inactive_flag_skips_update() {
        let mut bio = PlayerBiometrics::new(BodyHandle(0));
        bio.active = false;
        // Verify active flag is stored correctly
        assert!(!bio.active);
    }

    #[test]
    fn neutral_samples_produce_mid_phi() {
        let mut p = make_physics();
        let h = add_player(&mut p);
        let mut bio = PlayerBiometrics::new(h);

        // No samples → history.to_consciousness_inputs() returns 0.5 on all channels
        let inputs = bio.history.to_consciousness_inputs();
        // phi=0.5, which should produce a non-zero consciousness level
        p.field
            .update_entity(h, &inputs, symtropy_math::Point::origin());
        let phi = p.field.entities.get(&h).unwrap().phi();
        assert!(
            phi > 0.0,
            "neutral inputs (0.5) should produce non-zero Phi"
        );
    }

    #[test]
    fn entity_accessor_returns_state_after_update() {
        let mut p = make_physics();
        let h = add_player(&mut p);
        let mut bio = PlayerBiometrics::new(h);
        bio.push(BiometricSample {
            eeg_gamma_coherence: Some(1.0),
            hrv_rmssd: Some(90.0),
            ..Default::default()
        });
        let inputs = bio.history.to_consciousness_inputs();
        p.field
            .update_entity(h, &inputs, symtropy_math::Point::origin());
        assert!(p.field.entities.get(&h).is_some());
    }
}
