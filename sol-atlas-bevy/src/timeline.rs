// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bevy timeline system — keyboard scrubbing, auto-play, marker visibility.

use bevy::prelude::*;

/// Timeline state resource.
#[derive(Resource)]
pub struct TimelineState {
    pub year: u32,
    pub playing: bool,
    pub speed: f32,
    pub accumulator: f32,
}

impl Default for TimelineState {
    fn default() -> Self {
        Self {
            year: 0,
            playing: false,
            speed: 10.0, // years per second when auto-playing
            accumulator: 0.0,
        }
    }
}

/// Which timeline layer a marker belongs to (for visibility computation).
#[derive(Component, Clone, Copy)]
pub enum TimelineLayer {
    Fossil,
    Renewable,
    Nuclear,
    Vault(usize),
    Corridor(usize),
    Star, // always visible
}

/// Keyboard input for timeline scrubbing.
/// Left/Right: ±1 year. Shift+Left/Right: ±10. Space: toggle auto-play.
pub fn timeline_input_system(kb: Res<ButtonInput<KeyCode>>, mut state: ResMut<TimelineState>) {
    let shift = kb.pressed(KeyCode::ShiftLeft) || kb.pressed(KeyCode::ShiftRight);
    let step = if shift { 10 } else { 1 };

    if kb.pressed(KeyCode::ArrowRight) {
        state.year = state.year.saturating_add(step).min(500);
    }
    if kb.pressed(KeyCode::ArrowLeft) {
        state.year = state.year.saturating_sub(step);
    }
    if kb.just_pressed(KeyCode::Space) {
        state.playing = !state.playing;
    }
}

/// Auto-play: advance timeline when playing.
pub fn timeline_autoplay_system(time: Res<Time>, mut state: ResMut<TimelineState>) {
    if state.playing {
        state.accumulator += time.delta_secs() * state.speed;
        while state.accumulator >= 1.0 {
            state.year = state.year.saturating_add(1).min(500);
            state.accumulator -= 1.0;
        }
    }
}
