// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! HUD: balance telemetry + consciousness readout.

use bevy::prelude::*;
use symthaea_humanoid::simulator::HumanoidPhysicsSimulator;
use symtropy_consciousness_physics::safety::SafetyTier;

use crate::resources::*;

#[derive(Component)]
pub struct HumanoidHudText;

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
                Text::new("HUMANOID\nuprightness: ---\nroot height: ---\nhead height: ---\njoints effort: ---"),
                TextFont {
                    font_size: 16.0,
                    ..default()
                },
                TextColor(Color::srgb(0.75, 0.90, 1.0)),
                HumanoidHudText,
            ));

            root.spawn(Node {
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(8.0)),
                ..default()
            })
            .with_child((
                Text::new("CONSCIOUSNESS\nΦ: ---\ntier: ---\nmotor gain: ---\nPE: ---\npush: ---"),
                TextFont {
                    font_size: 16.0,
                    ..default()
                },
                TextColor(Color::srgb(0.6, 1.0, 0.6)),
                ConsciousnessHudText,
            ));
        });
}

pub fn update_hud(
    h: Res<HumanoidResources>,
    mut h_q: Query<&mut Text, (With<HumanoidHudText>, Without<ConsciousnessHudText>)>,
    mut c_q: Query<&mut Text, (With<ConsciousnessHudText>, Without<HumanoidHudText>)>,
) {
    let st = h.simulator.state();

    for mut t in &mut h_q {
        **t = format!(
            "HUMANOID\n\
             uprightness: {:.3}\n\
             root height: {:.2} m\n\
             head height: {:.2} m\n\
             joints effort: {:.2}   (21 actuators)",
            st.uprightness(),
            st.root_height,
            st.head_height,
            h.last_effort,
        );
    }

    let tier_name = match h.current_safety {
        SafetyTier::Green => "GREEN  ●",
        SafetyTier::Yellow => "YELLOW ◐",
        SafetyTier::Orange => "ORANGE ◑",
        SafetyTier::Red => "RED    ○",
    };
    let push_mag = (h.last_push_force[0].powi(2) + h.last_push_force[1].powi(2)).sqrt();

    for mut t in &mut c_q {
        **t = format!(
            "CONSCIOUSNESS\n\
             Φ: {:.3}\n\
             tier: {tier_name}\n\
             motor gain: {:.2}\n\
             PE: {:.3}\n\
             push: {:.1} N  ({:.0}% peak)",
            h.current_phi,
            h.current_motor_gain,
            h.last_prediction_error,
            push_mag,
            h.last_push_intensity * 100.0,
        );
    }
}
