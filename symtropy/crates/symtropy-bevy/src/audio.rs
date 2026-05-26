// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Audio integration for Symtropy using symthaea-muse.

use crate::plugin::SymtropyPhysics;
use bevy::prelude::*;
use symthaea_muse::{MuseConfig, MusicalState};

/// Component for entities that emit consciousness-driven audio.
#[derive(Component, Debug, Clone, Default, Reflect)]
#[reflect(Component)]
pub struct AudioEmitter {
    #[reflect(ignore)]
    pub config: MuseConfig,
}

pub struct RoboticAudioPlugin;

impl Plugin for RoboticAudioPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<AudioEmitter>().add_systems(
            Update,
            (
                phi_to_audio_system::<2>,
                phi_to_audio_system::<3>,
                phi_to_audio_system::<4>,
            ),
        );
    }
}

/// Update audio parameters based on current Φ and harmony levels.
pub fn phi_to_audio_system<const D: usize>(
    physics: Res<SymtropyPhysics<D>>,
    query: Query<(&crate::PhysicsBody, &AudioEmitter)>,
    // In a real implementation, this would interface with Bevy's Audio output
) {
    for (body, _emitter) in &query {
        let phi = physics.field.phi(body.handle);
        let harmony = physics
            .field
            .entities
            .get(&body.handle)
            .map(|e| e.harmony_activations)
            .unwrap_or([0.0; 9]);

        // Map to MusicalState (Muse expects 8 harmonies, we have 9 — drop the Pink/Contagion one for now)
        let mut muse_harmony = [0.0f32; 8];
        for i in 0..8 {
            muse_harmony[i] = harmony[i] as f32;
        }

        let _state = MusicalState {
            harmony_activations: muse_harmony,
            consciousness_level: phi as f32,
            ..Default::default()
        };

        // In the production version, we would render a short buffer
        // and stream it to the Bevy AudioSink.
    }
}
