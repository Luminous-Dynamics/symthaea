// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Quadruped HUD: telemetry + consciousness + named gait mode.

use bevy::prelude::*;
use symthaea_quadruped::simulator::QuadrupedPhysicsSimulator;
use symthaea_quadruped::types::GaitType;
use symtropy_consciousness_physics::safety::SafetyTier;

use crate::resources::*;

#[derive(Component)]
pub struct QuadrupedHudText;

#[derive(Component)]
pub struct ConsciousnessHudText;

pub fn setup_hud(mut commands: Commands) {
    commands
        .spawn(Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            flex_direction: FlexDirection::Row,
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::FlexStart,
            padding: UiRect::all(Val::Px(12.0)),
            ..default()
        })
        .with_children(|root| {
            root.spawn(Node {
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(8.0)),
                ..default()
            })
            .with_child((
                Text::new(
                    "QUADRUPED\nposition: ---\nforward vel: ---\nbase height: ---\neffort: ---\nterrain: ---",
                ),
                TextFont {
                    font_size: FontSize::Px(16.0),
                    ..default()
                },
                TextColor(Color::srgb(0.85, 0.95, 1.0)),
                QuadrupedHudText,
            ));

            root.spawn(Node {
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(8.0)),
                ..default()
            })
            .with_child((
                Text::new("CONSCIOUSNESS\nΦ: ---\ntier: ---\ngait: ---\nPE: ---"),
                TextFont {
                    font_size: FontSize::Px(16.0),
                    ..default()
                },
                TextColor(Color::srgb(0.6, 1.0, 0.6)),
                ConsciousnessHudText,
            ));
        });
}

pub fn update_hud(
    q: Res<QuadrupedResources>,
    mut q_q: Query<&mut Text, (With<QuadrupedHudText>, Without<ConsciousnessHudText>)>,
    mut c_q: Query<&mut Text, (With<ConsciousnessHudText>, Without<QuadrupedHudText>)>,
) {
    let st = q.simulator.state();
    let fwd = st.base_linear_velocity[0];

    for mut t in &mut q_q {
        **t = format!(
            "QUADRUPED\n\
             position: ({:+.2}, {:+.2}, {:+.2}) m\n\
             forward vel: {:.2} m/s\n\
             base height: {:.2} m\n\
             effort: {:.2}   (12 actuators)\n\
             terrain roughness: {:.2}",
            st.base_position[0],
            st.base_position[1],
            st.base_position[2],
            fwd,
            st.base_position[2],
            q.last_effort,
            q.last_terrain_roughness,
        );
    }

    let tier_name = match q.current_safety {
        SafetyTier::Green => "GREEN  ●",
        SafetyTier::Yellow => "YELLOW ◐",
        SafetyTier::Orange => "ORANGE ◑",
        SafetyTier::Red => "RED    ○",
    };
    let gait_name = match q.current_gait {
        GaitType::Trot => "TROT     (2.0 Hz, 0.08 m step)",
        GaitType::Walk => "WALK     (1.0 Hz, 0.04 m step)",
        GaitType::Freeze => "FREEZE   (hold stance)",
        GaitType::Collapse => "COLLAPSE (ragdoll)",
    };

    for mut t in &mut c_q {
        **t = format!(
            "CONSCIOUSNESS\n\
             Φ: {:.3}\n\
             tier: {tier_name}\n\
             gait: {gait_name}\n\
             PE: {:.3}",
            q.current_phi, q.last_prediction_error,
        );
    }
}
