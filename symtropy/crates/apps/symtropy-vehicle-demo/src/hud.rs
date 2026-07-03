// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! HUD: flight-demo-style two-column text overlay.

use bevy::prelude::*;
use symthaea_vehicle::simulator::VehiclePhysicsSimulator;
use symtropy_consciousness_physics::safety::SafetyTier;

use crate::resources::*;

#[derive(Component)]
pub struct VehicleHudText;

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
                Text::new("VEHICLE\nspeed: ---\nslip F/R: ---\nthrottle/brake: ---\nsteering: ---\nwaypoint: -/-"),
                TextFont {
                    font_size: FontSize::Px(16.0),
                    ..default()
                },
                TextColor(Color::srgb(1.0, 0.85, 0.70)),
                VehicleHudText,
            ));

            root.spawn(Node {
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(8.0)),
                ..default()
            })
            .with_child((
                Text::new("CONSCIOUSNESS\nΦ: ---\ntier: ---\nmotor gain: ---\nPE: ---\nice μ: --- (intensity ---)"),
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
    vehicle: Res<VehicleResources>,
    waypoints: Res<WaypointPath>,
    mut veh_q: Query<&mut Text, (With<VehicleHudText>, Without<ConsciousnessHudText>)>,
    mut consc_q: Query<&mut Text, (With<ConsciousnessHudText>, Without<VehicleHudText>)>,
) {
    let st = vehicle.simulator.state();

    for mut t in &mut veh_q {
        **t = format!(
            "VEHICLE\n\
             speed: {:.2} m/s (target 8.00)\n\
             slip F/R: {:+.3} / {:+.3} rad\n\
             throttle/brake: {:.2} / {:.2}\n\
             steering: {:+.2}\n\
             waypoint: {}/{}  (lap {})",
            st.speed,
            st.tire_slip_front,
            st.tire_slip_rear,
            vehicle.last_throttle,
            vehicle.last_brake,
            vehicle.last_steering,
            waypoints.current_index + 1,
            waypoints.waypoints.len(),
            waypoints.laps_completed,
        );
    }

    let tier_name = match vehicle.current_safety {
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
             PE: {:.3}\n\
             ice μ: {:.2}  (intensity {:.0}%)",
            vehicle.current_phi,
            vehicle.current_motor_gain,
            vehicle.last_prediction_error,
            vehicle.current_friction,
            vehicle.current_ice_intensity * 100.0,
        );
    }
}
