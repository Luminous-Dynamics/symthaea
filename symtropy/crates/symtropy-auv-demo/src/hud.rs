// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! HUD: AUV telemetry + consciousness readout.

use bevy::prelude::*;
use symthaea_auv::simulator::AuvPhysicsSimulator;
use symtropy_consciousness_physics::safety::SafetyTier;

use crate::resources::*;

#[derive(Component)]
pub struct AuvHudText;

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
                Text::new("AUV\ndepth: ---\nposition: ---\nthruster effort: ---\nwaypoint: -/-"),
                TextFont {
                    font_size: 16.0,
                    ..default()
                },
                TextColor(Color::srgb(1.0, 0.92, 0.60)),
                AuvHudText,
            ));

            root.spawn(Node {
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(8.0)),
                ..default()
            })
            .with_child((
                Text::new(
                    "CONSCIOUSNESS\nΦ: ---\ntier: ---\nmotor gain: ---\nPE: ---\ncurrent: ---",
                ),
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
    auv: Res<AuvResources>,
    waypoints: Res<WaypointPath>,
    mut auv_q: Query<&mut Text, (With<AuvHudText>, Without<ConsciousnessHudText>)>,
    mut consc_q: Query<&mut Text, (With<ConsciousnessHudText>, Without<AuvHudText>)>,
) {
    let st = auv.simulator.state();

    for mut t in &mut auv_q {
        **t = format!(
            "AUV\n\
             depth: {:.2} m  (pressure {:.1} kPa)\n\
             position: ({:+.1}, {:+.1}) m\n\
             thruster effort: {:.2}\n\
             waypoint: {}/{}  (lap {})",
            st.depth,
            st.pressure,
            st.position[0],
            st.position[1],
            auv.last_thruster_effort,
            waypoints.current_index + 1,
            waypoints.waypoints.len(),
            waypoints.laps_completed,
        );
    }

    let tier_name = match auv.current_safety {
        SafetyTier::Green => "GREEN  ●",
        SafetyTier::Yellow => "YELLOW ◐",
        SafetyTier::Orange => "ORANGE ◑",
        SafetyTier::Red => "RED    ○",
    };
    let cur_mag = (auv.last_current_force[0].powi(2) + auv.last_current_force[1].powi(2)).sqrt();

    for mut t in &mut consc_q {
        **t = format!(
            "CONSCIOUSNESS\n\
             Φ: {:.3}\n\
             tier: {tier_name}\n\
             motor gain: {:.2}\n\
             PE: {:.3}\n\
             current: {:.1} N  ({:.0}% of peak)",
            auv.current_phi,
            auv.current_motor_gain,
            auv.last_prediction_error,
            cur_mag,
            auv.last_current_intensity * 100.0,
        );
    }
}
