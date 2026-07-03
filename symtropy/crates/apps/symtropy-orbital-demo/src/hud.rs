// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! HUD: orbital + consciousness telemetry.

use bevy::prelude::*;
use symthaea_orbital::simulator::OrbitalPhysicsSimulator;
use symtropy_consciousness_physics::safety::SafetyTier;

use crate::resources::*;

#[derive(Component)]
pub struct OrbitalHudText;

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
                    "ORBITAL\nee position: ---\ncomm window: ---\nsolar exposure: ---\nattitude drift: ---\njoint effort: ---",
                ),
                TextFont {
                    font_size: FontSize::Px(16.0),
                    ..default()
                },
                TextColor(Color::srgb(1.0, 1.0, 0.75)),
                OrbitalHudText,
            ));

            root.spawn(Node {
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(8.0)),
                ..default()
            })
            .with_child((
                Text::new("CONSCIOUSNESS\nΦ: ---\ntier: ---\nmotor gain: ---\nPE: ---"),
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
    orbital: Res<OrbitalResources>,
    mut orbital_q: Query<&mut Text, (With<OrbitalHudText>, Without<ConsciousnessHudText>)>,
    mut consc_q: Query<&mut Text, (With<ConsciousnessHudText>, Without<OrbitalHudText>)>,
) {
    let st = orbital.simulator.state();
    let drift = (st
        .spacecraft_angular_velocity
        .iter()
        .map(|v| v * v)
        .sum::<f64>())
    .sqrt();

    for mut t in &mut orbital_q {
        **t = format!(
            "ORBITAL\n\
             ee position: ({:+.2}, {:+.2}, {:+.2}) m\n\
             comm window: {:.0}%\n\
             solar exposure: {:.0}%\n\
             attitude drift: {:.4} rad/s\n\
             joint effort: {:.2}",
            st.ee_position[0],
            st.ee_position[1],
            st.ee_position[2],
            st.comm_window * 100.0,
            st.solar_exposure * 100.0,
            drift,
            orbital.last_effort,
        );
    }

    let tier_name = match orbital.current_safety {
        SafetyTier::Green => "GREEN  ●",
        SafetyTier::Yellow => "YELLOW ◐",
        SafetyTier::Orange => "ORANGE ◑",
        SafetyTier::Red => "RED    ○",
    };

    for mut t in &mut consc_q {
        **t = format!(
            "CONSCIOUSNESS\n\
             Φ: {:.3}\n\
             tier: {tier_name}\n\
             motor gain: {:.2}\n\
             PE: {:.3}",
            orbital.current_phi, orbital.current_motor_gain, orbital.last_prediction_error,
        );
    }
}
