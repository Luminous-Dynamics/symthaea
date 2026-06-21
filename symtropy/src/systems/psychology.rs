// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Psychological needs system ported from the multiworld simulator.
//!
//! Adapts the sim's monthly-tick allostatic load / social satiation / engagement
//! model to real-time 64Hz game ticks. Runs every ~3 seconds (192 frames).
//!
//! References (from sim):
//! - McEwen (1998) "Protective and Damaging Effects of Stress Mediators" NEJM
//! - Dunbar (2010) "How Many Friends Does One Person Need?"
//! - Cacioppo & Patrick (2008) "Loneliness"
//! - Karasek (1979) demand-control model

use bevy::prelude::*;

use crate::components::{CrewNpc, NpcCollapsed, NpcTrust, Player, TendBalance};
use crate::resources::{EnergyWell, LeviathanState, PhysicsWorldRes, SleepPhase};
use symtropy_render_bridge::PhysicsBody;

/// Tick interval in seconds (~3s, maps sim's monthly tick to game time).
const PSYCHOLOGY_INTERVAL_SECS: f32 = 3.0;

/// Baseline stress accumulation per psychology tick.
const BASELINE_STRESS: f32 = 0.01;

/// Additional load from social isolation (no nearby agents).
const ISOLATION_LOAD: f32 = 0.02;

/// Additional load from danger (Leviathan stirring/awake).
const DANGER_LOAD: f32 = 0.04;

/// Natural load decay per psychology tick (rest, adaptation).
const LOAD_DECAY: f32 = 0.015;

/// Load reduction from harmony resonance (nearby resonant agents).
const HARMONY_LOAD_RELIEF: f32 = 0.02;

/// Load reduction from sanctuary zones.
const SANCTUARY_LOAD_RELIEF: f32 = 0.03;

/// Social satiation decay per tick.
const SOCIAL_DECAY: f32 = 0.015;

/// Social satiation bonus per nearby agent within harmony range.
const PROXIMITY_SOCIAL_BONUS: f32 = 0.02;

/// Isolation threshold — below this, stress accelerates.
const ISOLATION_THRESHOLD: f32 = 0.3;

/// Burnout threshold — above this, consciousness caps.
pub const BURNOUT_THRESHOLD: f32 = 0.8;

/// Engagement decay under high load + low social (escapism).
const ESCAPISM_DECAY: f32 = 0.02;

/// Engagement recovery when needs are met.
const ENGAGEMENT_RECOVERY: f32 = 0.008;

/// Psychological needs state for an NPC.
#[derive(Component)]
pub struct PsychologicalNeeds {
    /// Cumulative stress [0, 1]. Burnout at > 0.8.
    pub allostatic_load: f32,
    /// Relationship fulfillment [0, 1]. Below 0.3 = isolation.
    pub social_satiation: f32,
    /// Task participation [0, 1]. Low = disengaged / escapism.
    pub engagement: f32,
    /// Consecutive ticks with allostatic_load > COLLAPSE_THRESHOLD.
    /// At 3+, NPC collapses (T4-A).
    pub burnout_ticks: u32,
}

/// Load threshold for collapse (higher than burnout to avoid premature collapse).
const COLLAPSE_THRESHOLD: f32 = 0.9;
/// Consecutive ticks at collapse threshold before NPC collapses.
const COLLAPSE_TICKS_REQUIRED: u32 = 3;
/// Trust decay per psychology tick (T4-D).
const TRUST_DECAY_PER_TICK: f64 = 0.005;

impl Default for PsychologicalNeeds {
    fn default() -> Self {
        Self {
            allostatic_load: 0.15, // start slightly stressed (you're on a derelict ship)
            social_satiation: 0.5, // recently formed crew
            engagement: 0.8,       // high initial motivation
            burnout_ticks: 0,
        }
    }
}

/// Timer resource for psychology ticks.
#[derive(Resource)]
pub struct PsychologyTimer(pub f32);

impl Default for PsychologyTimer {
    fn default() -> Self {
        Self(0.0)
    }
}

/// Update psychological needs for all NPCs every ~3 seconds.
///
/// The player counts as a nearby agent for social calculations.
/// Stress equilibrium with 1+ companion: ~0.25. Isolated: runaway to burnout.
/// This is intentional — cooperation is thermodynamically mandatory.
pub fn psychology_tick_system(
    mut commands: Commands,
    mut npcs: Query<
        (
            Entity,
            &mut PsychologicalNeeds,
            &Transform,
            &PhysicsBody,
            &mut NpcTrust,
            Option<&NpcCollapsed>,
        ),
        With<CrewNpc>,
    >,
    other_agents: Query<&Transform, With<CrewNpc>>,
    player_query: Query<&Transform, With<crate::components::Player>>,
    leviathan: Res<LeviathanState>,
    harmony: Res<crate::systems::harmonies::LocalHarmonyState>,
    physics: Res<PhysicsWorldRes>,
    time: Res<Time>,
    mut timer: ResMut<PsychologyTimer>,
) {
    timer.0 += time.delta_secs();
    if timer.0 < PSYCHOLOGY_INTERVAL_SECS {
        return;
    }
    timer.0 = 0.0;

    let danger = match leviathan.phase {
        SleepPhase::Dormant => 0.0f32,
        SleepPhase::Stirring => 0.3,
        SleepPhase::Awake => 0.7,
        SleepPhase::Hunting => 1.0,
    };

    // Collect all agent positions INCLUDING the player for proximity checks
    let mut positions: Vec<Vec2> = other_agents
        .iter()
        .map(|t| t.translation.truncate())
        .collect();
    if let Ok(player_tf) = player_query.single() {
        positions.push(player_tf.translation.truncate());
    }

    for (entity, mut needs, transform, body, mut trust, collapsed) in &mut npcs {
        // Skip full psychology update for collapsed NPCs (they're in crisis).
        if collapsed.is_some() {
            continue;
        }

        let pos = transform.translation.truncate();

        // Count nearby agents including player (within harmony range)
        let harmony_range = physics.consciousness.constants.harmony_range as f32;
        let nearby_count = positions
            .iter()
            .filter(|p| {
                let d = (**p - pos).length();
                d > 2.0 && d < harmony_range
            })
            .count();

        let is_isolated = nearby_count == 0;
        let energy_frac = physics
            .consciousness
            .entities
            .get(&body.handle)
            .map(|e| e.energy.fraction_remaining() as f32)
            .unwrap_or(1.0);

        // --- Allostatic load ---
        // Accumulation
        needs.allostatic_load += BASELINE_STRESS;
        if is_isolated {
            needs.allostatic_load += ISOLATION_LOAD;
        }
        if danger > 0.1 {
            needs.allostatic_load += DANGER_LOAD * danger;
        }
        if energy_frac < 0.3 {
            needs.allostatic_load += 0.02; // energy depletion stress
        }

        // Decay and relief
        needs.allostatic_load -= LOAD_DECAY;
        if nearby_count > 0 {
            needs.allostatic_load -= HARMONY_LOAD_RELIEF * (nearby_count as f32).min(3.0);
        }
        if harmony.is_sanctuary {
            needs.allostatic_load -= SANCTUARY_LOAD_RELIEF;
        }
        needs.allostatic_load = needs.allostatic_load.clamp(0.0, 1.0);

        // --- Social satiation ---
        needs.social_satiation -= SOCIAL_DECAY;
        needs.social_satiation += PROXIMITY_SOCIAL_BONUS * (nearby_count as f32).min(3.0);
        needs.social_satiation = needs.social_satiation.clamp(0.0, 1.0);

        // --- Engagement ---
        if needs.allostatic_load > 0.5
            && needs.social_satiation < ISOLATION_THRESHOLD
            && danger < 0.3
        {
            needs.engagement -= ESCAPISM_DECAY;
        } else if danger > 0.5 {
            needs.engagement = (needs.engagement + 0.05).min(1.0);
        } else if needs.allostatic_load < 0.4 && needs.social_satiation > 0.4 {
            needs.engagement += ENGAGEMENT_RECOVERY;
        }
        needs.engagement = needs.engagement.clamp(0.0, 1.0);

        // --- T4-A: Burnout collapse detection ---
        if needs.allostatic_load > COLLAPSE_THRESHOLD {
            needs.burnout_ticks += 1;
            if needs.burnout_ticks >= COLLAPSE_TICKS_REQUIRED {
                eprintln!(
                    "[psychology] NPC collapsed from sustained burnout (load={:.2})",
                    needs.allostatic_load
                );
                needs.engagement = 0.0;
                commands.entity(entity).insert(NpcCollapsed {
                    recovery_tend_spent: 0,
                });
            }
        } else {
            needs.burnout_ticks = 0;
        }

        // --- T4-D: Trust decay ---
        trust.trust = (trust.trust - TRUST_DECAY_PER_TICK).max(0.0);
    }
}

// ============================================================================
// T4-B: NPC Collapse Recovery System
// ============================================================================

/// Resource tracking how many NPCs the player has saved this run.
#[derive(Resource, Default)]
pub struct CrewSavedCount(pub u32);

/// NPC collapse recovery: player must be nearby + near an energy well.
/// Each T press while conditions met adds 1 TEND toward the 3 required.
/// On recovery: NPC revives, crew saved count increases, Leviathan threshold drops.
pub fn npc_collapse_recovery_system(
    mut commands: Commands,
    mut collapsed_npcs: Query<
        (
            Entity,
            &Transform,
            &mut NpcCollapsed,
            &mut PsychologicalNeeds,
        ),
        With<CrewNpc>,
    >,
    player_query: Query<&Transform, With<Player>>,
    wells: Query<(&Transform, &EnergyWell)>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut player_tend: Query<&mut TendBalance, With<Player>>,
    mut saved: ResMut<CrewSavedCount>,
    mut leviathan: ResMut<LeviathanState>,
) {
    let Ok(player_tf) = player_query.single() else {
        return;
    };
    let player_pos = player_tf.translation.truncate();

    for (entity, npc_tf, mut collapsed, mut needs) in &mut collapsed_npcs {
        let npc_pos = npc_tf.translation.truncate();
        let dist_to_player = player_pos.distance(npc_pos);

        // Must be within interaction range.
        if dist_to_player > 60.0 {
            continue;
        }

        // Must be near an energy well.
        let near_well = wells.iter().any(|(well_tf, well)| {
            well.is_active() && npc_pos.distance(well_tf.translation.truncate()) < well.radius
        });
        if !near_well {
            continue;
        }

        // Player presses T to contribute TEND toward recovery.
        if keyboard.just_pressed(KeyCode::KeyT) {
            if let Ok(mut tend) = player_tend.single_mut() {
                if tend.can_spend(1) {
                    tend.balance -= 1;
                    collapsed.recovery_tend_spent += 1;
                    eprintln!(
                        "[psychology] Recovery TEND contributed ({}/3)",
                        collapsed.recovery_tend_spent
                    );
                }
            }
        }

        // Check if recovery threshold met.
        if collapsed.recovery_tend_spent >= 3 {
            eprintln!(
                "[psychology] NPC revived! Crew saved count: {}",
                saved.0 + 1
            );
            commands.entity(entity).remove::<NpcCollapsed>();
            needs.allostatic_load = 0.4;
            needs.engagement = 0.5;
            needs.burnout_ticks = 0;
            saved.0 += 1;

            // Reward: lower Leviathan threshold (easier sanctuary).
            leviathan.threshold = (leviathan.threshold - 0.5).max(5.0);
        }
    }
}

// ============================================================================
// T4-E: Visual NPC State
// ============================================================================

/// Update NPC sprite appearance based on psychological state.
pub fn npc_visual_state_system(
    mut npcs: Query<
        (
            &PsychologicalNeeds,
            Option<&mut Sprite>,
            Option<&MeshMaterial3d<StandardMaterial>>,
            Option<&NpcCollapsed>,
            &mut Transform,
        ),
        With<CrewNpc>,
    >,
    mut materials: ResMut<Assets<StandardMaterial>>,
    time: Res<Time>,
) {
    for (needs, opt_sprite, opt_mat, collapsed, mut transform) in &mut npcs {
        if collapsed.is_some() {
            // Collapsed: gray, small.
            if let Some(mut sprite) = opt_sprite {
                sprite.color = Color::srgb(0.3, 0.3, 0.3);
                sprite.custom_size = Some(Vec2::splat(12.0));
            }
            if let Some(mat_handle) = opt_mat {
                if let Some(mat) = materials.get_mut(&mat_handle.0) {
                    mat.base_color = Color::srgb(0.3, 0.3, 0.3);
                }
            }
            transform.scale = Vec3::splat(12.0 / 16.0);
            continue;
        }

        // High load: dim.
        let load_dim = if needs.allostatic_load > 0.6 {
            1.0 - (needs.allostatic_load - 0.6) * 0.8 // dims to ~0.68 at load=1.0
        } else {
            1.0
        };

        // Low engagement: shrink.
        let engagement_scale = if needs.engagement < 0.3 {
            12.0 + needs.engagement / 0.3 * 4.0 // 12-16px
        } else {
            16.0 // normal NPC size
        };

        // Low social: pulse (seeking attention).
        let social_pulse = if needs.social_satiation < 0.2 {
            0.8 + (time.elapsed_secs() * 3.0).sin().abs() * 0.2 // flicker between 0.8 and 1.0
        } else {
            1.0
        };

        let factor = load_dim * social_pulse;

        if let Some(mut sprite) = opt_sprite {
            let Srgba {
                red, green, blue, ..
            } = sprite.color.to_srgba();
            sprite.color = Color::srgb(
                (red * factor).clamp(0.0, 1.0),
                (green * factor).clamp(0.0, 1.0),
                (blue * factor).clamp(0.0, 1.0),
            );
            sprite.custom_size = Some(Vec2::splat(engagement_scale));
        }

        if let Some(mat_handle) = opt_mat {
            if let Some(mat) = materials.get_mut(&mat_handle.0) {
                let Srgba {
                    red, green, blue, ..
                } = mat.base_color.to_srgba();
                mat.base_color = Color::srgb(
                    (red * factor).clamp(0.0, 1.0),
                    (green * factor).clamp(0.0, 1.0),
                    (blue * factor).clamp(0.0, 1.0),
                );
            }
            transform.scale = Vec3::splat(engagement_scale / 16.0);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapse_threshold_consistent() {
        assert!(COLLAPSE_THRESHOLD > BURNOUT_THRESHOLD);
        assert!(COLLAPSE_TICKS_REQUIRED >= 2);
    }

    #[test]
    fn trust_decay_rate() {
        // At 3s ticks, trust decays ~0.1/minute.
        let decays_per_minute = 60.0 / PSYCHOLOGY_INTERVAL_SECS as f64;
        let total_decay = TRUST_DECAY_PER_TICK * decays_per_minute;
        assert!(
            total_decay > 0.05 && total_decay < 0.2,
            "Trust decay rate {total_decay}/min should be moderate"
        );
    }

    #[test]
    fn collapse_requires_sustained_burnout() {
        let mut needs = PsychologicalNeeds::default();
        // Single tick above threshold shouldn't collapse.
        needs.allostatic_load = 0.95;
        needs.burnout_ticks = 1;
        assert!(needs.burnout_ticks < COLLAPSE_TICKS_REQUIRED);
    }

    #[test]
    fn recovery_needs_three_tend() {
        let collapsed = NpcCollapsed {
            recovery_tend_spent: 2,
        };
        assert!(collapsed.recovery_tend_spent < 3);
        let recovered = NpcCollapsed {
            recovery_tend_spent: 3,
        };
        assert!(recovered.recovery_tend_spent >= 3);
    }

    #[test]
    fn default_needs_not_collapsed() {
        let needs = PsychologicalNeeds::default();
        assert!(needs.allostatic_load < COLLAPSE_THRESHOLD);
        assert!(needs.burnout_ticks == 0);
    }

    #[test]
    fn isolation_threshold_below_half() {
        assert!(ISOLATION_THRESHOLD < 0.5);
    }
}
