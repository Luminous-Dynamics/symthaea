// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! NPC dialogue: NPCs speak their consciousness bottleneck.
//!
//! When the player is near an NPC, a speech bubble appears above them
//! showing what they're experiencing — driven by the Master Equation.

use bevy::prelude::*;

use crate::components::{CrewNpc, Player};
use crate::systems::consciousness::NpcConsciousness;

/// Marker for NPC speech bubble text.
#[derive(Component)]
pub struct SpeechBubble {
    pub npc_entity: Entity,
    pub timer: f32,
}

/// Timer to throttle dialogue generation.
#[derive(Resource)]
pub struct DialogueTimer(pub f32);

impl Default for DialogueTimer {
    fn default() -> Self {
        Self(0.0)
    }
}

/// Generate dialogue from consciousness bottleneck + psychological needs.
fn bottleneck_to_dialogue(
    bottleneck: &str,
    level: f64,
    name: &str,
    needs: Option<&crate::systems::psychology::PsychologicalNeeds>,
) -> String {
    // 1. Role-specific overrides for high stress / allostatic burnout
    if name.contains("Kael") {
        if let Some(n) = needs {
            if n.allostatic_load > crate::systems::psychology::BURNOUT_THRESHOLD {
                return format!(
                    "{name}: \"The junction grids... they're completely fried. I can't keep them stable alone!\""
                );
            }
            if n.social_satiation < 0.15 {
                return format!(
                    "{name}: \"Does anyone even care about the life support status? I'm talking to empty rooms.\""
                );
            }
        }
        return match bottleneck {
            "phi" => format!(
                "{name}: \"If we don't fix the power loops, the consciousness coupling will collapse.\""
            ),
            "working_memory" => {
                format!("{name}: \"What was the sequence for the water pump purge again?\"")
            }
            "attention" => format!(
                "{name}: \"The voltage is fluctuating too fast. I can't focus on the schematics.\""
            ),
            _ => format!(
                "{name}: \"Checking the junction status. We need more voltage to keep the system online.\""
            ),
        };
    } else if name.contains("Mira") {
        if let Some(n) = needs {
            if n.allostatic_load > crate::systems::psychology::BURNOUT_THRESHOLD {
                return format!(
                    "{name}: \"So many wounded... the trauma feed is off the charts. I need a break.\""
                );
            }
            if n.social_satiation < 0.15 {
                return format!(
                    "{name}: \"I'm supposed to heal people, but everyone is shutting me out.\""
                );
            }
        }
        return match bottleneck {
            "phi" => format!(
                "{name}: \"Biometric stress is peaking. Please, everyone stay calm and breathe.\""
            ),
            "embodiment" => format!(
                "{name}: \"My hands are shaking... is this real feedback or phantom trauma?\""
            ),
            "knowledge" => format!(
                "{name}: \"I need clean water. If the pump is sabotaged, infection rates will spike.\""
            ),
            _ => format!(
                "{name}: \"Check your biometrics. Don't let your stress compromise the coupling.\""
            ),
        };
    } else if name.contains("Soren") {
        if let Some(n) = needs {
            if n.allostatic_load > crate::systems::psychology::BURNOUT_THRESHOLD {
                return format!(
                    "{name}: \"The logs... they're being erased. Everything we built... gone.\""
                );
            }
            if n.social_satiation < 0.15 {
                return format!(
                    "{name}: \"Isolated in the archive stacks. Only dead voices for company.\""
                );
            }
        }
        return match bottleneck {
            "phi" => format!(
                "{name}: \"This division of consciousness is documented in the ancient worldline files.\""
            ),
            "knowledge" => {
                format!("{name}: \"The Chronicle knows our choices. Nothing is forgotten.\"")
            }
            "broadcast" => format!(
                "{name}: \"I need to transmit this signal before the null ecology intercepts it.\""
            ),
            _ => format!(
                "{name}: \"I am scanning the historical anchors for precedents in the database.\""
            ),
        };
    } else if name.contains("Jack") {
        if let Some(n) = needs {
            if n.allostatic_load > crate::systems::psychology::BURNOUT_THRESHOLD {
                return format!(
                    "{name}: \"Drones everywhere! We're cornered! Get that door open!\""
                );
            }
            if n.social_satiation < 0.15 {
                return format!("{name}: \"Where is the backup team? We're on our own out here.\"");
            }
        }
        return match bottleneck {
            "attention" => {
                format!("{name}: \"Keep your eyes open. The null ecology is scanning this room.\"")
            }
            "working_memory" => {
                format!("{name}: \"I'm losing track of the patrol routes. Need to double back.\"")
            }
            _ => format!(
                "{name}: \"Secure the perimeter. I'll cover the pump area while you repair.\""
            ),
        };
    } else if name.contains("PR-4") {
        if let Some(n) = needs {
            if n.allostatic_load > crate::systems::psychology::BURNOUT_THRESHOLD {
                return format!(
                    "{name}: \"WARNING: Actuator thermal overload. Self-preservation protocol active.\""
                );
            }
        }
        return match bottleneck {
            "phi" => format!(
                "{name}: \"Error: Phi levels deviating from target parameters. Re-calibrating.\""
            ),
            "embodiment" => format!(
                "{name}: \"Chassis integrity at 98%. Ready for hardware maintenance task.\""
            ),
            "working_memory" => format!(
                "{name}: \"Buffer overflow: memory allocation failed. Repeating task list.\""
            ),
            _ => format!(
                "{name}: \"Beep. Ready to assist with power grid configuration and maintenance.\""
            ),
        };
    } else if name.contains("Nadia") {
        if let Some(n) = needs {
            if n.allostatic_load > crate::systems::psychology::BURNOUT_THRESHOLD {
                return format!(
                    "{name}: \"The trade-offs are too steep. We are sacrificing too much for this colony!\""
                );
            }
            if n.social_satiation < 0.15 {
                return format!(
                    "{name}: \"Nobody trusts my negotiations anymore. I'm talking to walls.\""
                );
            }
        }
        return match bottleneck {
            "phi" => format!(
                "{name}: \"We must balance resources. If we overdrive the factory, the entropy will kill us.\""
            ),
            "knowledge" => format!(
                "{name}: \"The factions are watching how we handle the water crisis. We need a vote.\""
            ),
            _ => format!(
                "{name}: \"We can secure a deal if we keep the power grid stable. Let's act logically.\""
            ),
        };
    } else if name.contains("Leo") {
        if let Some(n) = needs {
            if n.allostatic_load > crate::systems::psychology::BURNOUT_THRESHOLD {
                return format!(
                    "{name}: \"I don't know what to do! It's all falling apart and I'm scared!\""
                );
            }
            if n.social_satiation < 0.15 {
                return format!("{name}: \"Why is everyone ignoring me? I just want to help.\"");
            }
        }
        return match bottleneck {
            "phi" => format!(
                "{name}: \"Is my consciousness level high enough? I don't want to get caught by the Leviathan.\""
            ),
            "knowledge" => {
                format!("{name}: \"How do I use the repair tool? I'm trying my best...\"")
            }
            "working_memory" => {
                format!("{name}: \"Wait, what was Kael's instruction? I forgot...\"")
            }
            _ => format!("{name}: \"I'll watch the monitors. Tell me what needs doing, please.\""),
        };
    }

    // Fallback generic dialog for unknown NPCs
    if let Some(n) = needs {
        if n.allostatic_load > crate::systems::psychology::BURNOUT_THRESHOLD {
            return format!("{name}: \"I can't keep this up... everything hurts.\"");
        }
        if n.social_satiation < 0.15 {
            return format!("{name}: \"Does anyone even know I'm here?\"");
        }
        if n.engagement < 0.2 {
            return format!("{name}: \"...\"");
        }
    }

    if level > 0.7 {
        match bottleneck {
            "phi" => format!("{name}: \"I feel... integrated. Whole.\""),
            "broadcast" => format!("{name}: \"I can almost share what I know.\""),
            "knowledge" => format!("{name}: \"I understand more now.\""),
            "embodiment" => format!("{name}: \"My body feels real here.\""),
            _ => format!("{name}: \"Something is shifting inside me.\""),
        }
    } else if level > 0.4 {
        match bottleneck {
            "phi" => format!("{name}: \"Parts of me feel... disconnected.\""),
            "broadcast" => format!("{name}: \"I know something but can't express it.\""),
            "working_memory" => format!("{name}: \"I keep forgetting what I was doing.\""),
            "attention" => format!("{name}: \"I can't focus. Too much happening.\""),
            "knowledge" => format!("{name}: \"What IS this place? I need answers.\""),
            "embodiment" => format!("{name}: \"Am I really here? This doesn't feel real.\""),
            "recurrence" => format!("{name}: \"I can't think deeply enough about this.\""),
            "synchrony" => format!("{name}: \"We need to be together. Closer.\""),
            _ => format!("{name}: \"...\""),
        }
    } else {
        match bottleneck {
            "phi" => format!("{name}: \"I'm falling apart...\""),
            "broadcast" => format!("{name}: \"HELP— I can't— nobody hears—\""),
            "working_memory" => format!("{name}: \"Where... who...\""),
            "attention" => format!("{name}: \"Everything is noise. NOISE.\""),
            "knowledge" => format!("{name}: \"I don't understand anything anymore.\""),
            "embodiment" => format!("{name}: \"I can't feel my hands.\""),
            _ => format!("{name}: \"...please...\""),
        }
    }
}

/// Show NPC dialogue when player is nearby.
pub fn dialogue_system(
    player: Query<&Transform, With<Player>>,
    npcs: Query<(
        Entity,
        &Transform,
        &CrewNpc,
        Option<&NpcConsciousness>,
        Option<&crate::systems::psychology::PsychologicalNeeds>,
    )>,
    mut commands: Commands,
    existing_bubbles: Query<(Entity, &SpeechBubble)>,
    mut timer: ResMut<DialogueTimer>,
    time: Res<Time>,
) {
    timer.0 += time.delta_secs();
    if timer.0 < 3.0 {
        return;
    } // Update dialogue every 3 seconds
    timer.0 = 0.0;

    let Ok(player_tf) = player.single() else {
        return;
    };
    let player_pos = player_tf.translation.truncate();

    // Remove old bubbles
    for (entity, _) in &existing_bubbles {
        commands.entity(entity).despawn();
    }

    // Show dialogue for nearest NPC within range
    let mut closest: Option<(Entity, f32, String)> = None;
    for (entity, npc_tf, npc, consciousness, psych) in &npcs {
        let dist = player_pos.distance(npc_tf.translation.truncate());
        if dist < 80.0 {
            let dialogue = if let Some(c) = consciousness {
                bottleneck_to_dialogue(&c.bottleneck, c.level, &npc.name, psych)
            } else {
                format!("{}: \"...\"", npc.name)
            };
            if closest.is_none() || dist < closest.as_ref().unwrap().1 {
                closest = Some((entity, dist, dialogue));
            }
        }
    }

    if let Some((npc_entity, _, text)) = closest {
        // Find NPC position for bubble placement
        if let Ok((_, npc_tf, _, _, _)) = npcs.get(npc_entity) {
            commands.spawn((
                Text::new(text),
                TextFont {
                    font_size: 14.0,
                    ..default()
                },
                TextColor(Color::srgba(0.9, 0.9, 0.7, 0.9)),
                Node {
                    position_type: PositionType::Absolute,
                    top: Val::Px(680.0), // bottom of screen
                    left: Val::Px(12.0),
                    ..default()
                },
                SpeechBubble {
                    npc_entity,
                    timer: 0.0,
                },
            ));
        }
    }
}
