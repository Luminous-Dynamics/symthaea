// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Real-time HUD for the flight demo.
//!
//! Top-left: flight telemetry (altitude, attitude, thrust, waypoint progress).
//! Top-right: consciousness readout (Φ, safety tier, motor gain, PE, gust).

use bevy::prelude::*;
use symthaea_multirotor::simulator::PhysicsSimulator;
use symtropy_consciousness_physics::safety::SafetyTier;

use crate::resources::*;

#[derive(Component)]
pub struct FlightHudText;

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
            // Left: flight telemetry
            root.spawn(Node {
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(8.0)),
                ..default()
            })
            .with_child((
                Text::new("FLIGHT\naltitude: ---\nroll/pitch: ---\nthrust: ---\nwaypoint: -/-"),
                TextFont {
                    font_size: FontSize::Px(16.0),
                    ..default()
                },
                TextColor(Color::srgb(0.85, 0.95, 1.0)),
                FlightHudText,
            ));

            // Right: consciousness side-channel
            root.spawn(Node {
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(8.0)),
                ..default()
            })
            .with_child((
                Text::new("CONSCIOUSNESS\nΦ: ---\ntier: ---\nmotor gain: ---\nPE: ---\ngust: ---"),
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
    quad: Res<QuadrotorState>,
    waypoints: Res<WaypointPath>,
    mut flight_q: Query<&mut Text, (With<FlightHudText>, Without<ConsciousnessHudText>)>,
    mut consc_q: Query<&mut Text, (With<ConsciousnessHudText>, Without<FlightHudText>)>,
) {
    let st = quad.simulator.state();
    let (roll, pitch, _yaw) = st.euler_angles();

    for mut t in &mut flight_q {
        **t = format!(
            "FLIGHT\n\
             altitude: {:.2} m\n\
             roll/pitch: {:+.2} / {:+.2} rad\n\
             thrust: {:.3} N\n\
             waypoint: {}/{}   (cycles {})",
            st.position[2],
            roll,
            pitch,
            quad.last_thrust,
            waypoints.current_index + 1,
            waypoints.waypoints.len(),
            waypoints.cycles_completed,
        );
    }

    let tier_name = match quad.current_safety {
        SafetyTier::Green => "GREEN  ●",
        SafetyTier::Yellow => "YELLOW ◐",
        SafetyTier::Orange => "ORANGE ◑",
        SafetyTier::Red => "RED    ○",
    };
    let gust_mag = (quad.last_gust_force[0].powi(2) + quad.last_gust_force[1].powi(2)).sqrt();

    for mut t in &mut consc_q {
        **t = format!(
            "CONSCIOUSNESS\n\
             Φ: {:.3}\n\
             tier: {tier_name}\n\
             motor gain: {:.2}\n\
             PE: {:.3}\n\
             gust: {:.3} N  ({:.0}% intensity)",
            quad.current_phi,
            quad.current_motor_gain,
            quad.last_prediction_error,
            gust_mag,
            quad.last_gust_intensity * 100.0,
        );
    }
}
