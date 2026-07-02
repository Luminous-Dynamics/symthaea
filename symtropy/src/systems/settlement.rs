// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Settlement systems for Firstlight Basin.
//!
//! Manages infrastructure health, metric coupling, and the repair loop.

use crate::components::*;
use crate::resources::*;
use bevy::prelude::*;

/// System that checks if the water crisis is resolved and triggers the first vote.
pub fn settlement_governance_trigger_system(
    metrics: Res<SettlementMetrics>,
    mut vote: ResMut<GovernanceVote>,
    mut next_state: ResMut<NextState<GamePhase>>,
    mut triggered: Local<bool>,
) {
    if !*triggered && metrics.water > 0.8 && metrics.power > 0.8 {
        info!("Water Crisis Resolved! Triggering First Public Vote.");
        crate::systems::chronicle_recorder::record_chronicle_event(
            "WaterCrisisResolved",
            serde_json::json!({
                "water_stability": metrics.water,
                "power_stability": metrics.power,
                "trust_level": metrics.trust,
            }),
        );
        *vote = GovernanceVote::new_water_crisis_vote();
        next_state.set(GamePhase::Council);
        *triggered = true;
    }
}

/// Simple UI system for the Council phase (voting).
pub fn council_ui_system(
    mut contexts: bevy_egui::EguiContexts,
    mut vote: ResMut<GovernanceVote>,
    mut next_state: ResMut<NextState<GamePhase>>,
    mut metrics: ResMut<SettlementMetrics>,
    mut log: ResMut<GovernanceLog>,
    mut active_ports: ResMut<crate::ports::ActivePorts>,
    time: Res<Time>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };

    bevy_egui::egui::Window::new("⚖️ Settlement Council")
        .anchor(bevy_egui::egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .collapsible(false)
        .resizable(false)
        .show(ctx, |ui| {
            ui.heading(&vote.question);
            ui.add_space(10.0);

            let mut new_selection = None;
            for (i, option) in vote.options.iter().enumerate() {
                if ui
                    .selectable_label(vote.selected_index == i, &option.label)
                    .clicked()
                {
                    new_selection = Some(i);
                }
                ui.label(&option.description);
                ui.small(&option.effect_text);
                ui.add_space(5.0);
            }

            if let Some(selection) = new_selection {
                vote.selected_index = selection;
            }

            ui.add_space(20.0);
            if ui.button("Cast Vote").clicked() {
                let selection = &vote.options[vote.selected_index];
                info!("Vote Cast: {}", selection.label);

                // Apply effects locally first
                match vote.selected_index {
                    0 => {
                        // Public Repair
                        metrics.trust += 0.2;
                        metrics.repair += 0.3;
                    }
                    1 => {
                        // Factory Overdrive
                        metrics.power += 0.2;
                        metrics.entropy += 0.1;
                    }
                    2 => {
                        // Perimeter Defense
                        metrics.safety += 0.3;
                        metrics.trust -= 0.05;
                    }
                    3 => {
                        // Archive Recovery
                        metrics.legitimacy += 0.2;
                        metrics.entropy -= 0.05;
                    }
                    _ => {}
                }

                // Cast via GovernancePort
                let vote_payload = crate::ports::Vote {
                    proposal_id: "water_crisis_001".to_string(),
                    voter_did: "did:key:z6MkLocalPlayer".to_string(),
                    approve: vote.selected_index == 0 || vote.selected_index == 3,
                };

                let tx_hash = if let Ok(receipt) = active_ports.governance.cast_vote(vote_payload) {
                    info!("Vote receipt confirmed: {}", receipt.tx_hash);
                    receipt.tx_hash
                } else {
                    "pending_local_fallback".to_string()
                };

                log.push(
                    time.elapsed_secs(),
                    format!("Decree: {}", selection.label),
                    1,
                );

                // Persist to Chronicle via ChroniclePort
                let _ = active_ports.chronicle.record_event(
                    "GovernanceVoteCast",
                    serde_json::json!({
                        "question": vote.question,
                        "selected_option": selection.label,
                        "tx_hash": tx_hash,
                        "metrics_after_vote": {
                            "power": metrics.power,
                            "water": metrics.water,
                            "trust": metrics.trust,
                            "safety": metrics.safety,
                        }
                    }),
                );

                vote.is_active = false;
                next_state.set(GamePhase::Playing);
            }
        });
}

/// System that updates global settlement metrics based on infrastructure health.
pub fn settlement_metric_update_system(
    mut metrics: ResMut<SettlementMetrics>,
    junctions: Query<&PowerJunction>,
    pumps: Query<&WaterPump>,
    time: Res<Time>,
) {
    // 1. Calculate Power Stability
    let total_junctions = junctions.iter().count() as f32;
    if total_junctions > 0.0 {
        let active_power: f32 = junctions
            .iter()
            .map(|j| if j.is_damaged { 0.2 } else { j.output })
            .sum();
        metrics.power = (active_power / total_junctions).clamp(0.0, 1.0);
    }

    // 2. Calculate Water Availability (depends on power)
    let total_pumps = pumps.iter().count() as f32;
    if total_pumps > 0.0 {
        let pump_efficiency: f32 = pumps
            .iter()
            .map(|p| {
                if !p.is_running || p.is_sabotaged {
                    0.0
                } else {
                    p.efficiency * metrics.power // Efficiency limited by power stability
                }
            })
            .sum();

        let target_water = pump_efficiency / total_pumps;
        // Water levels change gradually
        let delta = (target_water - metrics.water) * time.delta_secs() * 0.1;
        metrics.water = (metrics.water + delta).clamp(0.0, 1.0);
    }

    // 3. Systemic Entropy increases when power/water are low
    let entropy_pressure = (1.0 - metrics.power).max(1.0 - metrics.water);
    metrics.entropy = (metrics.entropy + entropy_pressure * time.delta_secs() * 0.01).min(1.0);

    // 4. Safety decreases as entropy rises
    metrics.safety = (1.0 - metrics.entropy).clamp(0.0, 1.0);
}

/// System for player interaction with infrastructure.
pub fn settlement_interaction_system(
    mouse: Res<ButtonInput<MouseButton>>,
    mut junctions: Query<(&mut PowerJunction, &Transform, &InteractionTarget)>,
    mut pumps: Query<(&mut WaterPump, &Transform, &InteractionTarget)>,
    player: Query<&Transform, With<Player>>,
    mut metrics: ResMut<SettlementMetrics>,
    mut log: ResMut<GovernanceLog>,
    mut active_ports: ResMut<crate::ports::ActivePorts>,
    time: Res<Time>,
) {
    let Ok(player_tf) = player.single() else {
        return;
    };

    if mouse.just_pressed(MouseButton::Left) {
        // Check junctions
        for (mut junction, tf, target) in &mut junctions {
            let dist = player_tf.translation.distance(tf.translation);
            if dist < target.radius && junction.is_damaged {
                junction.is_damaged = false;
                junction.output = 1.0;
                metrics.trust += 0.05;
                log.push(
                    time.elapsed_secs(),
                    "Power Junction Repaired".to_string(),
                    0,
                );

                // Persist event to Chronicle via ChroniclePort
                let _ = active_ports.chronicle.record_event(
                    "PowerJunctionRepaired",
                    serde_json::json!({
                        "actor": "player_local",
                        "junction_position": [tf.translation.x, tf.translation.y],
                    }),
                );

                info!("Repaired Power Junction!");
            }
        }

        // Check pumps
        for (mut pump, tf, target) in &mut pumps {
            let dist = player_tf.translation.distance(tf.translation);
            if dist < target.radius && (pump.is_sabotaged || !pump.is_running) {
                pump.is_sabotaged = false;
                pump.is_running = true;
                metrics.trust += 0.1;
                log.push(time.elapsed_secs(), "Water Pump Restored".to_string(), 0);

                // Persist event to Chronicle via ChroniclePort
                let _ = active_ports.chronicle.record_event(
                    "WaterPumpRestored",
                    serde_json::json!({
                        "actor": "player_local",
                        "pump_position": [tf.translation.x, tf.translation.y],
                    }),
                );

                info!("Restored Water Pump!");
            }
        }
    }
}

/// Modulates NPC behavior based on settlement metrics.
pub fn npc_settlement_reaction_system(
    metrics: Res<SettlementMetrics>,
    mut query: Query<(&mut CrewNpc, &mut MoveTarget)>,
) {
    for (mut npc, mut target) in &mut query {
        if metrics.water < 0.2 {
            // High stress/caution when water is critical
            npc.caution = (npc.caution + 0.01).min(1.0);
        } else if metrics.power < 0.3 {
            // NPCs move slower in low power
            target.speed = 50.0;
        } else {
            // Normal behavior
            npc.caution = (npc.caution - 0.001).max(0.3);
            target.speed = 100.0;
        }
    }
}
