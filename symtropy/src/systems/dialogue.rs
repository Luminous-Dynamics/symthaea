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
    // Psychological needs override consciousness dialogue when extreme
    if let Some(n) = needs {
        if n.allostatic_load > crate::systems::psychology::BURNOUT_THRESHOLD {
            return format!("{name}: \"I can't keep this up... everything hurts.\"");
        }
        if n.social_satiation < 0.15 {
            return format!("{name}: \"Does anyone even know I'm here?\"");
        }
        if n.engagement < 0.2 {
            return format!("{name}: \"...\""); // disengaged — goes silent
        }
        // Moderate stress flavors
        if n.allostatic_load > 0.6 {
            return match bottleneck {
                "phi" => format!("{name}: \"Too much stress to think straight.\""),
                _ => format!("{name}: \"I need a moment. Just... a moment.\""),
            };
        }
        if n.social_satiation < 0.3 {
            return format!("{name}: \"Stay close. Please.\"");
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
