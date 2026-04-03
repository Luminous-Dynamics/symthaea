// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Consciousness state management — the 3 sliders that drive everything.

use bevy::prelude::*;
use symthaea_muse::MusicalState;

/// The consciousness state resource — shared between audio and UI.
#[derive(Resource)]
pub struct ConsciousnessState {
    pub arousal: f32,
    pub valence: f32,
    pub phi: f32,
    /// The derived musical state sent to the synth engine.
    pub musical_state: MusicalState,
}

impl Default for ConsciousnessState {
    fn default() -> Self {
        let mut s = Self {
            arousal: 0.3,
            valence: 0.2,
            phi: 0.5,
            musical_state: MusicalState::default(),
        };
        s.update_musical_state();
        s
    }
}

impl ConsciousnessState {
    /// Recompute the musical state from V/A/Phi sliders.
    pub fn update_musical_state(&mut self) {
        self.musical_state.arousal = self.arousal;
        self.musical_state.valence = self.valence;
        self.musical_state.consciousness_level = self.phi;

        self.musical_state.dopamine = 0.3 + self.arousal * 0.5 + self.valence.max(0.0) * 0.2;
        self.musical_state.serotonin = 0.4 + self.valence.max(0.0) * 0.4;
        self.musical_state.noradrenaline = self.arousal * 0.5;
        self.musical_state.prediction_error = 0.1 + self.arousal * 0.2;

        self.musical_state.harmony_activations[0] = 0.5 + self.phi * 0.5;
        self.musical_state.harmony_activations[1] = 0.3 + self.valence.max(0.0) * 0.7;
        self.musical_state.harmony_activations[2] = 0.4 + self.phi * 0.3;
        self.musical_state.harmony_activations[3] = 0.2 + self.arousal * 0.5 + (-self.valence).max(0.0) * 0.3;
        self.musical_state.harmony_activations[4] = 0.3 + (1.0 - self.arousal) * 0.4;
        self.musical_state.harmony_activations[5] = 0.3 + self.valence.abs() * 0.3 + self.phi * 0.2;
        self.musical_state.harmony_activations[6] = self.arousal * 0.8;
        self.musical_state.harmony_activations[7] = (1.0 - self.arousal) * 0.6 + self.phi * 0.2;
    }
}

pub struct ConsciousnessPlugin;

impl Plugin for ConsciousnessPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ConsciousnessState>();
    }
}
