// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Eight Harmonies: the moral geometry made physical.
//!
//! The Eight Harmonies are not lore text — they are environmental states
//! that the player physically experiences through visuals and audio.
//! Each Harmony maps to a specific 16,384D HDC attractor state.
//!
//! When a local area achieves a Harmony (via TEND exchange, care work,
//! NPC cooperation), the environment transforms:
//! - Walls shift color (cold teal → warm gold)
//! - Audio resolves from dissonance to consonance
//! - Sacred Stillness rooms become Leviathan-proof safe zones

use bevy::prelude::*;

use crate::components::{Player, Tile};
use crate::resources::{BiometricsCtx, LeviathanState, SleepPhase};

/// The Eight Harmonies — mathematical attractors of consciousness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Harmony {
    /// Unison/Octave — base consonance. Unity of self.
    ResonantCoherence,
    /// Major 3rd — warmth. Care for all sentient beings.
    PanSentientFlourishing,
    /// Perfect 5th — stability. Deep understanding.
    IntegralWisdom,
    /// Minor 7th — tension. The creative edge of play.
    InfinitePlay,
    /// Perfect 4th — openness. Connection across boundaries.
    UniversalInterconnectedness,
    /// Major 6th — reciprocal warmth. Give and receive equally.
    SacredReciprocity,
    /// Ascending sequence — growth. Evolution toward complexity.
    EvolutionaryProgression,
    /// Drone/pedal — contemplation. The silence that knows.
    SacredStillness,
}

impl Harmony {
    pub const ALL: [Harmony; 8] = [
        Self::ResonantCoherence,
        Self::PanSentientFlourishing,
        Self::IntegralWisdom,
        Self::InfinitePlay,
        Self::UniversalInterconnectedness,
        Self::SacredReciprocity,
        Self::EvolutionaryProgression,
        Self::SacredStillness,
    ];

    /// Index into the harmony_activations array.
    pub fn index(&self) -> usize {
        match self {
            Self::ResonantCoherence => 0,
            Self::PanSentientFlourishing => 1,
            Self::IntegralWisdom => 2,
            Self::InfinitePlay => 3,
            Self::UniversalInterconnectedness => 4,
            Self::SacredReciprocity => 5,
            Self::EvolutionaryProgression => 6,
            Self::SacredStillness => 7,
        }
    }

    /// The color this harmony manifests as in the environment.
    pub fn color(&self) -> Color {
        match self {
            Self::ResonantCoherence => Color::srgb(0.9, 0.85, 0.6), // warm gold
            Self::PanSentientFlourishing => Color::srgb(0.4, 0.9, 0.5), // verdant green
            Self::IntegralWisdom => Color::srgb(0.5, 0.6, 0.9),     // deep blue
            Self::InfinitePlay => Color::srgb(0.9, 0.5, 0.8),       // playful magenta
            Self::UniversalInterconnectedness => Color::srgb(0.3, 0.8, 0.8), // teal
            Self::SacredReciprocity => Color::srgb(0.9, 0.7, 0.4),  // amber
            Self::EvolutionaryProgression => Color::srgb(0.6, 0.9, 0.3), // lime
            Self::SacredStillness => Color::srgb(0.95, 0.9, 0.7),   // soft white-gold
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::ResonantCoherence => "Resonant Coherence",
            Self::PanSentientFlourishing => "Pan-Sentient Flourishing",
            Self::IntegralWisdom => "Integral Wisdom",
            Self::InfinitePlay => "Infinite Play",
            Self::UniversalInterconnectedness => "Universal Interconnectedness",
            Self::SacredReciprocity => "Sacred Reciprocity",
            Self::EvolutionaryProgression => "Evolutionary Progression",
            Self::SacredStillness => "Sacred Stillness",
        }
    }
}

/// Tracks the dominant harmony state of the local environment around the player.
#[derive(Resource)]
pub struct LocalHarmonyState {
    /// Current harmony activations [0, 1] for each of the 8 harmonies.
    pub activations: [f32; 8],
    /// The dominant harmony (highest activation).
    pub dominant: Option<Harmony>,
    /// Whether the local area is in Sacred Stillness (Leviathan-proof).
    pub is_sanctuary: bool,
    /// Total harmony energy (sum of all activations).
    pub total_energy: f32,
}

impl Default for LocalHarmonyState {
    fn default() -> Self {
        Self {
            activations: [0.1; 8],
            dominant: None,
            is_sanctuary: false,
            total_energy: 0.8,
        }
    }
}

/// Update harmony state from biometrics and player behavior.
///
/// The harmonies are driven by the inverse of stress — calm, cooperative
/// behavior raises harmony activations. Panic and isolation lower them.
pub fn harmony_update_system(
    biometrics: Res<BiometricsCtx>,
    leviathan: Res<LeviathanState>,
    mut harmony: ResMut<LocalHarmonyState>,
    time: Res<Time>,
) {
    let stress = biometrics.encoder.compute_stress_vector();
    let load = biometrics.model.allostatic_load;
    let dt = time.delta_secs();

    // Base decay — harmonies fade without active cultivation
    for a in &mut harmony.activations {
        *a = (*a - 0.01 * dt).max(0.0);
    }

    // Calm behavior raises Sacred Stillness
    if stress.arousal < 0.2 && load < 0.3 {
        harmony.activations[7] = (harmony.activations[7] + 0.05 * dt).min(1.0);
    }

    // Moderate engagement raises Resonant Coherence
    if stress.arousal > 0.1 && stress.arousal < 0.5 && stress.valence > 0.0 {
        harmony.activations[0] = (harmony.activations[0] + 0.03 * dt).min(1.0);
    }

    // Exploration raises Evolutionary Progression
    if stress.dominance > 0.3 {
        harmony.activations[6] = (harmony.activations[6] + 0.02 * dt).min(1.0);
    }

    // High stress/panic raises Infinite Play (tension harmony)
    if stress.arousal > 0.7 {
        harmony.activations[3] = (harmony.activations[3] + 0.04 * dt).min(1.0);
    }

    // Leviathan danger suppresses all harmonies except Infinite Play
    if leviathan.phase == SleepPhase::Hunting {
        for i in [0, 1, 2, 4, 5, 6, 7] {
            harmony.activations[i] *= 1.0 - 0.5 * dt;
        }
    }

    // Compute dominant and total
    harmony.total_energy = harmony.activations.iter().sum();
    harmony.dominant = harmony
        .activations
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .and_then(|(max_idx, max_val)| {
            if *max_val > 0.3 {
                Some(Harmony::ALL[max_idx])
            } else {
                None
            }
        });

    // Sacred Stillness sanctuary: active when stillness > 0.6 and total energy > 2.0
    harmony.is_sanctuary = harmony.activations[7] > 0.6 && harmony.total_energy > 2.0;
}

/// Visual manifestation: tint nearby floor tiles with the dominant harmony color.
pub fn harmony_visual_system(
    harmony: Res<LocalHarmonyState>,
    player: Query<&Transform, With<Player>>,
    mut tiles: Query<(&Transform, &mut Sprite, &Tile), Without<Player>>,
    time: Res<Time>,
) {
    let Ok(player_tf) = player.single() else {
        return;
    };
    let player_pos = player_tf.translation.truncate();

    let Some(dominant) = harmony.dominant else {
        return;
    };
    let harmony_color = dominant.color();
    let intensity = harmony.activations[dominant.index()];

    // Tint tiles within radius based on harmony intensity
    let radius = 120.0 + intensity * 80.0; // 120-200px

    for (tile_tf, mut sprite, tile) in &mut tiles {
        if !tile.walkable {
            continue;
        }

        let tile_pos = tile_tf.translation.truncate();
        let dist = player_pos.distance(tile_pos);

        if dist < radius {
            let falloff = 1.0 - (dist / radius);
            let blend = falloff * intensity * 0.4; // max 40% tint

            // Pulse gently
            let pulse = 1.0 + (time.elapsed_secs() * 1.5 + dist * 0.01).sin() * 0.1;

            let base = Color::srgb(0.22, 0.22, 0.30); // floor color
            let Srgba {
                red: br,
                green: bg,
                blue: bb,
                ..
            } = base.to_srgba();
            let Srgba {
                red: hr,
                green: hg,
                blue: hb,
                ..
            } = harmony_color.to_srgba();

            sprite.color = Color::srgb(
                (br * (1.0 - blend) + hr * blend * pulse).clamp(0.0, 1.0),
                (bg * (1.0 - blend) + hg * blend * pulse).clamp(0.0, 1.0),
                (bb * (1.0 - blend) + hb * blend * pulse).clamp(0.0, 1.0),
            );
        }
    }
}

/// Sanctuary mechanic: Sacred Stillness rooms make you invisible to the Leviathan.
pub fn sanctuary_system(harmony: Res<LocalHarmonyState>, mut leviathan: ResMut<LeviathanState>) {
    if harmony.is_sanctuary {
        // Suppress noise accumulation while in sanctuary
        leviathan.noise_accumulator *= 0.95;
    }
}
