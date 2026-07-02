// Copyright (C) 2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Playable Tutorial Scenario controller system.

use crate::components::{CrewNpc, NpcActionEvent, WaterPump, WorldFeedbackEvent};
use crate::resources::{SettlementMetrics, TutorialScenarioRes, TutorialStep};
use crate::systems::psychology::PsychologicalNeeds;
use bevy::prelude::*;

pub fn tutorial_scenario_system(
    tutorial: Option<ResMut<TutorialScenarioRes>>,
    mut water_pumps: Query<(Entity, &Transform, &mut WaterPump)>,
    npcs: Query<(Entity, &CrewNpc, &Transform)>,
    mut needs_query: Query<&mut PsychologicalNeeds>,
    _action_writer: MessageWriter<NpcActionEvent>,
    mut feedback_writer: MessageWriter<WorldFeedbackEvent>,
    mut settlement_metrics: Option<ResMut<SettlementMetrics>>,
) {
    let Some(mut tutorial) = tutorial else {
        return;
    };

    match tutorial.step {
        TutorialStep::Inactive => {}
        TutorialStep::PumpSabotaged => {
            // Find PR-4 position
            let mut pr4_pos = None;
            for (_, npc, tf) in &npcs {
                if npc.name.contains("PR-4") {
                    pr4_pos = Some(tf.translation.truncate());
                }
            }
            if let Some(pump_ent) = tutorial.pump_entity {
                if let Ok((_, tf, mut pump)) = water_pumps.get_mut(pump_ent) {
                    // Ensure the pump is sabotaged initially
                    pump.is_sabotaged = true;
                    // If PR-4 is near the pump, move to next stage
                    if let Some(p4_pos) = pr4_pos {
                        if p4_pos.distance(tf.translation.truncate()) < 35.0 {
                            tutorial.step = TutorialStep::PR4Repairing;
                            feedback_writer.write(WorldFeedbackEvent {
                                position: tf.translation.truncate(),
                                message: "FIELD DECK: PR-4 REPAIR TRACE ACTIVE".to_string(),
                                color: Color::srgb(0.9, 0.6, 0.2),
                            });
                        }
                    }
                }
            }
        }
        TutorialStep::PR4Repairing => {
            // Check if pump efficiency reaches 0.7 (partial repair)
            if let Some(pump_ent) = tutorial.pump_entity {
                if let Ok((_, tf, pump)) = water_pumps.get(pump_ent) {
                    if pump.efficiency >= 0.7 && !pump.is_sabotaged {
                        tutorial.step = TutorialStep::CoopRepairing;
                        feedback_writer.write(WorldFeedbackEvent {
                            position: tf.translation.truncate(),
                            message: "FIELD DECK: PUMP PARTIALLY ONLINE — CONTAMINATION REMAINS"
                                .to_string(),
                            color: Color::srgb(0.9, 0.6, 0.2),
                        });
                    }
                }
            }
        }
        TutorialStep::CoopRepairing => {
            // Wait for efficiency to reach 1.0
            if let Some(pump_ent) = tutorial.pump_entity {
                if let Ok((_, tf, pump)) = water_pumps.get(pump_ent) {
                    if pump.efficiency >= 1.0 {
                        tutorial.step = TutorialStep::LeoStressRising;

                        if let Some(ref mut sm) = settlement_metrics {
                            sm.water = 1.0;
                        }

                        feedback_writer.write(WorldFeedbackEvent {
                            position: tf.translation.truncate(),
                            message: "COOPERATIVE REPAIR COMPLETE — FLOW RESTORED".to_string(),
                            color: Color::srgb(0.2, 0.9, 0.4),
                        });

                        // Force Leo's stress high to trigger the next phase
                        for (entity, npc, _) in &npcs {
                            if npc.name.contains("Leo") {
                                if let Ok(mut needs) = needs_query.get_mut(entity) {
                                    needs.allostatic_load = 0.55;
                                }
                            }
                        }
                    }
                }
            }
        }
        TutorialStep::LeoStressRising => {
            // Check if Kael moves close to Leo
            let mut kael_pos = None;
            let mut leo_pos = None;
            let mut leo_entity = None;
            for (entity, npc, tf) in &npcs {
                if npc.name.contains("Kael") {
                    kael_pos = Some(tf.translation.truncate());
                } else if npc.name.contains("Leo") {
                    leo_pos = Some(tf.translation.truncate());
                    leo_entity = Some(entity);
                }
            }

            if let (Some(k_pos), Some(l_pos), Some(l_ent)) = (kael_pos, leo_pos, leo_entity) {
                if k_pos.distance(l_pos) < 50.0 {
                    tutorial.step = TutorialStep::LeoStabilizing;
                    feedback_writer.write(WorldFeedbackEvent {
                        position: l_pos,
                        message: "LEO STABILIZED — CARE PROXIMITY ACTIVE".to_string(),
                        color: Color::srgb(0.2, 0.9, 0.4),
                    });
                } else {
                    // Force Leo's stress to rise during this tutorial phase
                    if let Ok(mut needs) = needs_query.get_mut(l_ent) {
                        needs.allostatic_load = (needs.allostatic_load + 0.05).min(1.0);
                    }
                }
            }
        }
        TutorialStep::LeoStabilizing => {
            // Wait for Leo's stress to decay below 0.25 under Mira/Kael care
            let mut leo_stress = 0.0;
            let mut leo_pos = Vec2::ZERO;
            for (entity, npc, tf) in &npcs {
                if npc.name.contains("Leo") {
                    leo_pos = tf.translation.truncate();
                    if let Ok(needs) = needs_query.get(entity) {
                        leo_stress = needs.allostatic_load;
                    }
                }
            }
            if leo_stress < 0.25 {
                tutorial.step = TutorialStep::Completed;
                feedback_writer.write(WorldFeedbackEvent {
                    position: leo_pos,
                    message: "SETTLEMENT STABILIZED — CAUSALITY TUTORIAL RESOLVED".to_string(),
                    color: Color::srgb(0.2, 0.9, 0.4),
                });
            }
        }
        TutorialStep::Completed => {}
    }
}
