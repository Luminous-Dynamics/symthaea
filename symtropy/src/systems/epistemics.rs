// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Epistemic Fog of War — visibility driven by real EmpiricalLevel.
//!
//! The player's flashlight radius is physically determined by their epistemic
//! confidence, using the REAL `mycelix_core_types::epistemic::EmpiricalLevel`.
//!
//! - E0 (Unverifiable): 60px — can barely see adjacent tiles
//! - E1 (Anecdotal): 100px — one room ahead
//! - E2 (Observable): 150px — comfortable exploration
//! - E3 (Measurable): 220px — strategic awareness
//! - E4 (CryptographicallyVerifiable): 300px — full room visibility
//!
//! Epistemic level advances by scavenging, verification, and FL contribution.
//! Coercion (taking NPC data without consent) degrades ALL epistemic tags,
//! shrinking the flashlight — you literally see less when you violate consent.

use bevy::prelude::*;

use crate::components::{ConsciousnessComp, CrewNpc, EpistemicTag, Flashlight, NpcTrust, Player};
use crate::resources::GovernanceLog;
use crate::systems::fl_simulation::FlPool;

/// Flashlight radius per epistemic level (pixels).
const RADIUS_PER_LEVEL: [f32; 5] = [
    60.0,  // E0: Unverifiable — barely see
    100.0, // E1: Anecdotal — one room
    150.0, // E2: Observable — comfortable
    220.0, // E3: Measurable — strategic
    300.0, // E4: CryptographicallyVerifiable — full visibility
];

/// Player's current epistemic level (aggregate of all their verified items).
/// This is a Resource because it affects the global flashlight, not per-entity.
#[derive(Resource)]
pub struct PlayerEpistemicState {
    /// Current aggregate epistemic level (0-4).
    pub level: u8,
    /// Number of items verified at each level.
    pub verified_counts: [u32; 5],
    /// Total items scavenged.
    pub total_scavenged: u32,
    /// Whether coercion penalty is active (decays over time).
    pub coercion_active: bool,
    /// Seconds until coercion penalty expires.
    pub coercion_cooldown: f32,
}

impl Default for PlayerEpistemicState {
    fn default() -> Self {
        Self {
            level: 0, // Start at E0 — you know nothing
            verified_counts: [0; 5],
            total_scavenged: 0,
            coercion_active: false,
            coercion_cooldown: 0.0,
        }
    }
}

impl PlayerEpistemicState {
    /// Recompute aggregate level from verified items.
    /// Level = highest E-level where you have at least 2 verified items.
    fn recompute_level(&mut self) {
        self.level = 0;
        for l in (0..5).rev() {
            if self.verified_counts[l] >= 2 {
                self.level = l as u8;
                break;
            }
        }
        // Coercion caps at E1 regardless of items
        if self.coercion_active {
            self.level = self.level.min(1);
        }
    }
}

/// Update flashlight radius based on epistemic level.
/// This is the core mechanic: knowledge = sight.
pub fn epistemic_flashlight_system(
    epistemic: Res<PlayerEpistemicState>,
    mut flashlights: Query<&mut Flashlight, With<Player>>,
    biometrics: Res<crate::resources::BiometricsCtx>,
) {
    let Ok(mut flashlight) = flashlights.single_mut() else {
        return;
    };

    let level = epistemic.level as usize;
    let base_radius = RADIUS_PER_LEVEL[level.min(4)];

    // Stress still modulates flicker, but base radius is epistemic
    let stress_flicker = biometrics.model.allostatic_load * 0.2;
    flashlight.base_radius = base_radius;
    flashlight.flicker = stress_flicker + biometrics.encoder.velocity_surprise() * 0.3;
}

/// Scavenge verification system — player verifies items to advance epistemic level.
/// NPCs can independently confirm items to advance from E1→E2 and E2→E3.
/// Contributing verified data to FL pool advances E3→E4.
pub fn epistemic_advancement_system(
    mut epistemic: ResMut<PlayerEpistemicState>,
    mut items: Query<&mut EpistemicTag>,
    npcs: Query<(&CrewNpc, &ConsciousnessComp)>,
    fl_pool: Res<FlPool>,
    mut log: ResMut<GovernanceLog>,
    time: Res<Time>,
    mut advancement_timer: Local<f32>,
) {
    // Tick coercion cooldown
    if epistemic.coercion_active {
        epistemic.coercion_cooldown -= time.delta_secs();
        if epistemic.coercion_cooldown <= 0.0 {
            epistemic.coercion_active = false;
            let msg = "Epistemic penalty expired — knowledge trust restored".to_string();
            log.push(time.elapsed_secs(), msg, 0);
        }
    }

    *advancement_timer += time.delta_secs();
    if *advancement_timer < 5.0 {
        return; // Check every 5 seconds
    }
    *advancement_timer = 0.0;

    // Count NPC confirmations (NPCs with high consciousness auto-verify items)
    let confirming_npcs: usize = npcs
        .iter()
        .filter(|(_, cp)| cp.combined_score() >= 0.4)
        .count();

    let mut advanced_any = false;

    for mut tag in &mut items {
        let current = tag.level();
        match current {
            // E0→E1: Player scavenged it (just having the item is E1)
            0 => {
                tag.advance();
                advanced_any = true;
            }
            // E1→E2: At least 2 NPCs can confirm (Observable)
            1 if confirming_npcs >= 2 => {
                tag.advance();
                advanced_any = true;
            }
            // E2→E3: All NPCs confirm (Measurable — independent replication)
            2 if confirming_npcs >= 3 => {
                tag.advance();
                advanced_any = true;
            }
            // E3→E4: Data contributed to FL and aggregation quality > 90% (Cryptographically Verifiable)
            3 if fl_pool.aggregation_quality > 0.9 && fl_pool.round > 0 => {
                tag.advance();
                advanced_any = true;
            }
            _ => {}
        }
    }

    if advanced_any {
        // Recount verified items
        epistemic.verified_counts = [0; 5];
        for tag in &items {
            let level = tag.level() as usize;
            if level < 5 {
                epistemic.verified_counts[level] += 1;
            }
        }
        epistemic.recompute_level();
    }
}

/// Coercion penalty — when player forces NPC data without consent,
/// ALL epistemic tags degrade and flashlight shrinks.
///
/// Triggered externally (from economy.rs when trust < 0.3 during data access).
/// This function is called as a one-shot, not a system.
pub fn apply_coercion_penalty(
    epistemic: &mut PlayerEpistemicState,
    items: &mut Query<&mut EpistemicTag>,
    log: &mut GovernanceLog,
    time_secs: f32,
) {
    // Degrade ALL items by one level
    let mut degraded = 0;
    for mut tag in items.iter_mut() {
        if tag.level() > 0 {
            tag.degrade();
            degraded += 1;
        }
    }

    // Activate coercion penalty (caps epistemic level at E1)
    epistemic.coercion_active = true;
    epistemic.coercion_cooldown = 30.0; // 30 seconds penalty

    // Recount
    epistemic.verified_counts = [0; 5];
    for tag in items.iter() {
        let level = tag.level() as usize;
        if level < 5 {
            epistemic.verified_counts[level] += 1;
        }
    }
    epistemic.recompute_level();

    let msg = format!(
        "COERCION DETECTED — {} items degraded, visibility reduced to E{}! (penalty: 30s)",
        degraded, epistemic.level,
    );
    eprintln!("[epistemics] {}", msg);
    log.push(time_secs, msg, 2);
}
