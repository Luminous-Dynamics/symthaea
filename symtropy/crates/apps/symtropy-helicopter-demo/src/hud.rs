// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Helicopter HUD: left flight telemetry, right consciousness readout.

use bevy::prelude::*;
use symthaea_helicopter::simulator::HelicopterPhysicsSimulator;
use symtropy_consciousness_physics::safety::SafetyTier;

use crate::resources::*;

#[derive(Component)]
pub struct HelicopterHudText;

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
                    "HELICOPTER\naltitude: ---\noff-station: ---\nrotor rpm: ---\ncollective: ---",
                ),
                TextFont {
                    font_size: 16.0,
                    ..default()
                },
                TextColor(Color::srgb(1.0, 0.85, 0.85)),
                HelicopterHudText,
            ));

            root.spawn(Node {
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(8.0)),
                ..default()
            })
            .with_child((
                Text::new("CONSCIOUSNESS\nΦ: ---\ntier: ---\nmotor gain: ---\nPE: ---\nwind: ---"),
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
    heli: Res<HelicopterResources>,
    mut heli_q: Query<&mut Text, (With<HelicopterHudText>, Without<ConsciousnessHudText>)>,
    mut consc_q: Query<&mut Text, (With<ConsciousnessHudText>, Without<HelicopterHudText>)>,
) {
    let st = heli.simulator.state();
    let off = (st.position[0].powi(2) + st.position[1].powi(2)).sqrt();

    for mut t in &mut heli_q {
        **t = format!(
            "HELICOPTER\n\
             altitude: {:.2} m (target 20.00)\n\
             off-station: {:.2} m\n\
             rotor rpm: {:.0} (main) / {:.0} (tail)\n\
             collective: {:.2}",
            st.position[2], off, st.main_rotor_rpm, st.tail_rotor_rpm, heli.last_collective,
        );
    }

    let tier_name = match heli.current_safety {
        SafetyTier::Green => "GREEN  ●",
        SafetyTier::Yellow => "YELLOW ◐",
        SafetyTier::Orange => "ORANGE ◑",
        SafetyTier::Red => "RED    ○",
    };
    let wind_mag = (heli.last_wind_force[0].powi(2)
        + heli.last_wind_force[1].powi(2)
        + heli.last_wind_force[2].powi(2))
    .sqrt();

    for mut t in &mut consc_q {
        **t = format!(
            "CONSCIOUSNESS\n\
             Φ: {:.3}\n\
             tier: {tier_name}\n\
             motor gain: {:.2}\n\
             PE: {:.3}\n\
             wind: {:.1} N  ({:.1} m/s)",
            heli.current_phi,
            heli.current_motor_gain,
            heli.last_prediction_error,
            wind_mag,
            heli.last_wind_speed,
        );
    }
}
