// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! The Room That Remembers You.
//!
//! Tracks the player's accumulated consciousness history across runs.
//! The starting room changes based on this history — walls shift color,
//! the environment reflects who you've been.

use bevy::prelude::*;

/// Persistent memory of the player across runs.
#[derive(Resource)]
pub struct RoomMemory {
    /// Total runs played.
    pub runs: u32,
    /// Total deaths.
    pub deaths: u32,
    /// Total victories.
    pub victories: u32,
    /// Peak consciousness level achieved.
    pub peak_consciousness: f64,
    /// Total NSM fragments collected across all runs.
    pub total_fragments: u32,
    /// Accumulated calm seconds (time spent below 0.2 arousal).
    pub calm_seconds: f32,
    /// Accumulated panic seconds (time spent above 0.7 arousal).
    pub panic_seconds: f32,
    /// The room remembers your dominant harmony.
    pub dominant_harmony_history: [f32; 8],
}

impl Default for RoomMemory {
    fn default() -> Self {
        Self {
            runs: 0,
            deaths: 0,
            victories: 0,
            peak_consciousness: 0.0,
            total_fragments: 0,
            calm_seconds: 0.0,
            panic_seconds: 0.0,
            dominant_harmony_history: [0.0; 8],
        }
    }
}

impl RoomMemory {
    /// The room's personality — derived from accumulated play history.
    pub fn room_personality(&self) -> &'static str {
        if self.runs == 0 {
            return "Awakening";
        }
        let calm_ratio = self.calm_seconds / (self.calm_seconds + self.panic_seconds + 1.0);
        if self.victories > self.deaths && calm_ratio > 0.6 {
            "Sanctuary"
        } else if self.deaths > self.victories * 2 {
            "Haunted"
        } else if calm_ratio < 0.3 {
            "Turbulent"
        } else if self.peak_consciousness > 0.8 {
            "Luminous"
        } else {
            "Becoming"
        }
    }

    /// Color the room takes based on personality.
    pub fn room_color(&self) -> Color {
        match self.room_personality() {
            "Sanctuary" => Color::srgb(0.3, 0.5, 0.35),  // warm green
            "Haunted" => Color::srgb(0.25, 0.15, 0.2),   // dark purple
            "Turbulent" => Color::srgb(0.35, 0.2, 0.15), // rusty red
            "Luminous" => Color::srgb(0.4, 0.38, 0.25),  // golden
            "Becoming" => Color::srgb(0.2, 0.25, 0.3),   // cool blue
            _ => Color::srgb(0.15, 0.15, 0.2),           // neutral
        }
    }
}

/// Update room memory from game state each frame.
pub fn room_memory_update_system(
    biometrics: Res<crate::resources::BiometricsCtx>,
    player_c: Res<crate::systems::consciousness::PlayerConsciousness>,
    harmony: Res<crate::systems::harmonies::LocalHarmonyState>,
    mut memory: ResMut<RoomMemory>,
    time: Res<Time>,
) {
    let stress = biometrics.encoder.compute_stress_vector();
    let dt = time.delta_secs();

    // Track calm vs panic time
    if stress.arousal < 0.2 {
        memory.calm_seconds += dt;
    } else if stress.arousal > 0.7 {
        memory.panic_seconds += dt;
    }

    // Track peak consciousness
    if player_c.level > memory.peak_consciousness {
        memory.peak_consciousness = player_c.level;
    }

    // Accumulate harmony history
    for i in 0..8 {
        memory.dominant_harmony_history[i] += harmony.activations[i] * dt * 0.01;
    }
}

/// Record run outcome when game ends.
pub fn record_death(memory: &mut RoomMemory, fragments: u32) {
    memory.runs += 1;
    memory.deaths += 1;
    memory.total_fragments += fragments;
}

pub fn record_victory(memory: &mut RoomMemory, fragments: u32) {
    memory.runs += 1;
    memory.victories += 1;
    memory.total_fragments += fragments;
}
