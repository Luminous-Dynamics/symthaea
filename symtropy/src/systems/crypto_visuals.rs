// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Visual feedback for physicalized cryptography.
//!
//! Makes the crypto infrastructure VISIBLE:
//! - FL defense: room tint shifts green→red based on aggregation quality
//! - DKG ceremony: extraction progress bar changes color based on ceremony state
//! - Epistemic fog: flashlight color shifts with E-level (warm→cool)
//!
//! All effects modify existing sprites — no new entity spawning needed.

use bevy::prelude::*;

use super::dkg_ceremony::DkgCeremonyState;
use super::epistemics::PlayerEpistemicState;
use super::fl_simulation::FlPool;
use crate::components::{ConsciousnessComp, CrewNpc, EpistemicTag, Flashlight, FusionCore, Player};

/// Tint the world based on FL defense quality.
/// Clean FL = normal colors. Degraded FL = red shift. Overwhelmed = deep red.
pub fn fl_defense_visual_system(pool: Res<FlPool>, mut clear_color: ResMut<ClearColor>) {
    if pool.round == 0 {
        return; // No FL rounds yet
    }

    let quality = pool.aggregation_quality;

    // Blend background from normal dark blue-gray toward red as quality drops
    let base_r = 0.02;
    let base_g = 0.02;
    let base_b = 0.04;

    let danger_r = 0.15 * (1.0 - quality);
    let danger_g = -0.01 * (1.0 - quality);

    clear_color.0 = Color::srgb(
        (base_r + danger_r).clamp(0.0, 0.3),
        (base_g + danger_g).max(0.0),
        base_b * quality,
    );
}

/// Visual feedback on the Fusion Core based on DKG ceremony state.
/// No ceremony: yellow. Active ceremony: pulsing blue. Succeeded: green. Failed: red.
pub fn dkg_ceremony_visual_system(
    ceremony: Res<DkgCeremonyState>,
    mut cores: Query<&mut Sprite, With<FusionCore>>,
    time: Res<Time>,
) {
    let Ok(mut sprite) = cores.single_mut() else {
        return;
    };

    if ceremony.ceremony_succeeded {
        // Green glow — ceremony succeeded, core unlocked
        sprite.color = Color::srgb(0.2, 1.0, 0.3);
    } else if ceremony.ceremony.is_some() {
        // Active ceremony — pulsing blue with progress
        let pulse = (time.elapsed_secs() * 4.0).sin().abs();
        let progress = ceremony.registered_names.len() as f32 / ceremony.threshold as f32;
        sprite.color = Color::srgb(0.2 * (1.0 - progress), 0.4 + pulse * 0.3, 0.8 + pulse * 0.2);
    } else {
        // Default: yellow
        sprite.color = Color::srgb(1.0, 0.9, 0.1);
    }
}

/// Visual feedback for epistemic state on the flashlight/player.
/// E0: dim amber. E1: warm yellow. E2: white. E3: cool cyan. E4: bright blue-white.
pub fn epistemic_visual_system(
    epistemic: Res<PlayerEpistemicState>,
    mut player_sprites: Query<&mut Sprite, With<Player>>,
) {
    let Ok(mut sprite) = player_sprites.single_mut() else {
        return;
    };

    // Player sprite color shifts with epistemic level
    let (r, g, b) = match epistemic.level {
        0 => (0.6, 0.4, 0.2), // E0: dim amber — you know nothing
        1 => (0.8, 0.7, 0.3), // E1: warm yellow — anecdotal
        2 => (0.2, 0.9, 1.0), // E2: cyan — observable (default player color)
        3 => (0.3, 0.8, 1.0), // E3: bright cyan — measurable
        _ => (0.5, 0.9, 1.0), // E4: blue-white — verified
    };

    // If coercion penalty active, desaturate
    if epistemic.coercion_active {
        sprite.color = Color::srgb(r * 0.5, g * 0.3, b * 0.3); // Reddish desaturation
    } else {
        sprite.color = Color::srgb(r, g, b);
    }
}

/// Show NPC participation readiness for DKG via sprite color.
/// Green: ready to participate. Yellow: marginal. Red: will defect.
pub fn npc_dkg_readiness_visual_system(
    ceremony: Res<DkgCeremonyState>,
    mut npcs: Query<(
        &mut Sprite,
        &CrewNpc,
        &ConsciousnessComp,
        &crate::components::NpcTrust,
    )>,
) {
    // Only show readiness when ceremony might start
    if ceremony.ceremony_succeeded {
        return;
    }

    for (mut sprite, npc, consciousness, trust) in &mut npcs {
        let score = consciousness.combined_score();
        let trust_ok = trust.trust >= 0.4;
        let consciousness_ok = score >= 0.3;
        let stress_ok = npc.caution < 0.7;

        if trust_ok && consciousness_ok && stress_ok {
            // Ready — green tint
            sprite.color = Color::srgb(0.3, 0.9, 0.3);
        } else if trust_ok || consciousness_ok {
            // Marginal — yellow
            sprite.color = Color::srgb(0.9, 0.8, 0.2);
        } else {
            // Will defect — red tint
            sprite.color = Color::srgb(0.9, 0.3, 0.2);
        }
    }
}
