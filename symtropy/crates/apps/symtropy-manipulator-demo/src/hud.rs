// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Real-time HUD overlay showing Phi metrics, ISO metrics, and throughput comparison.

use crate::resources::*;
use bevy::prelude::*;
use symthaea_manipulator::simulator::ManipulatorPhysicsSimulator;
use symtropy_consciousness_physics::safety::SafetyTier;

/// Marker for the Phi arm stats text.
#[derive(Component)]
pub struct PhiHudText;

/// Marker for the ISO arm stats text.
#[derive(Component)]
pub struct IsoHudText;

/// Marker for the throughput advantage text.
#[derive(Component)]
pub struct ThroughputHudText;

/// Spawn the HUD UI nodes.
pub fn setup_hud(mut commands: Commands) {
    // Root container — full screen
    commands.spawn(Node {
        width: Val::Percent(100.0),
        height: Val::Percent(100.0),
        flex_direction: FlexDirection::Column,
        justify_content: JustifyContent::SpaceBetween,
        ..default()
    }).with_children(|parent| {
        // Top row: Phi stats (left) + ISO stats (right)
        parent.spawn(Node {
            width: Val::Percent(100.0),
            flex_direction: FlexDirection::Row,
            justify_content: JustifyContent::SpaceBetween,
            padding: UiRect::all(Val::Px(12.0)),
            ..default()
        }).with_children(|row| {
            // Phi arm panel (left)
            row.spawn(Node {
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(8.0)),
                ..default()
            }).with_child((
                Text::new("ADAPTIVE SAFETY\nCertainty: ---\nTier: ---\nStiffness: ---\nPE: ---\nCycles: 0"),
                TextFont {
                    font_size: 16.0,
                    ..default()
                },
                TextColor(Color::srgb(0.0, 0.9, 0.3)),
                PhiHudText,
            ));

            // ISO arm panel (right)
            row.spawn(Node {
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(8.0)),
                ..default()
            }).with_child((
                Text::new("ISO/TS 15066 SSM\nStatus: ---\nSp: ---\nCycles: 0"),
                TextFont {
                    font_size: 16.0,
                    ..default()
                },
                TextColor(Color::srgb(0.9, 0.6, 0.1)),
                IsoHudText,
            ));
        });

        // Bottom center: throughput advantage
        parent.spawn(Node {
            width: Val::Percent(100.0),
            justify_content: JustifyContent::Center,
            padding: UiRect::all(Val::Px(16.0)),
            ..default()
        }).with_child((
            Text::new("THROUGHPUT ADVANTAGE: measuring..."),
            TextFont {
                font_size: 28.0,
                ..default()
            },
            TextColor(Color::srgb(1.0, 1.0, 1.0)),
            ThroughputHudText,
        ));
    });
}

/// Update Phi arm HUD text.
pub fn update_phi_hud(phi_arm: Res<PhiArmState>, mut query: Query<&mut Text, With<PhiHudText>>) {
    for mut text in &mut query {
        let tier_name = match phi_arm.current_safety {
            SafetyTier::Green => "GREEN",
            SafetyTier::Yellow => "YELLOW",
            SafetyTier::Orange => "ORANGE",
            SafetyTier::Red => "RED",
        };
        let tier_color = match phi_arm.current_safety {
            SafetyTier::Green => "●",
            SafetyTier::Yellow => "◐",
            SafetyTier::Orange => "◑",
            SafetyTier::Red => "○",
        };

        **text = format!(
            "ADAPTIVE SAFETY\n\
             Certainty: {:.3} {tier_color}\n\
             Tier: {tier_name}\n\
             Stiffness: {:.0}%\n\
             PE: {:.3}\n\
             Cycles: {} ({:.2}s avg)",
            phi_arm.current_phi,
            phi_arm.stiffness * 100.0,
            phi_arm.last_prediction_error,
            phi_arm.cycles_completed,
            phi_arm.mean_cycle_time(),
        );
    }
}

/// Update ISO arm HUD text.
pub fn update_iso_hud(iso: Res<IsoArmState>, mut query: Query<&mut Text, With<IsoHudText>>) {
    for mut text in &mut query {
        let status = if iso.is_stopped {
            "STOPPED ■"
        } else {
            "RUNNING ▶"
        };

        **text = format!(
            "ISO/TS 15066 SSM\n\
             Status: {status}\n\
             Sp: {:.2}m\n\
             Cycles: {} ({:.2}s avg)",
            iso.current_protective_distance,
            iso.cycles_completed,
            iso.mean_cycle_time(),
        );
    }
}

/// Update throughput advantage text.
pub fn update_throughput_hud(
    metrics: Res<ThroughputMetrics>,
    phi_arm: Res<PhiArmState>,
    iso: Res<IsoArmState>,
    mut query: Query<&mut Text, With<ThroughputHudText>>,
    mut color_query: Query<&mut TextColor, With<ThroughputHudText>>,
) {
    let both_have_cycles = phi_arm.cycles_completed >= 2 && iso.cycles_completed >= 2;

    for mut text in &mut query {
        if both_have_cycles {
            let sign = if metrics.advantage_percent >= 0.0 {
                "+"
            } else {
                ""
            };
            **text = format!(
                "THROUGHPUT ADVANTAGE: {sign}{:.1}%  (Adaptive: {} cycles | ISO: {} cycles)",
                metrics.advantage_percent, phi_arm.cycles_completed, iso.cycles_completed,
            );
        } else {
            **text = format!(
                "THROUGHPUT ADVANTAGE: measuring... (Adaptive: {} | ISO: {} cycles)",
                phi_arm.cycles_completed, iso.cycles_completed,
            );
        }
    }

    // Color: green if Phi wins, red if ISO wins
    for mut color in &mut color_query {
        if both_have_cycles && metrics.advantage_percent > 0.0 {
            *color = TextColor(Color::srgb(0.0, 1.0, 0.3));
        } else if both_have_cycles && metrics.advantage_percent < 0.0 {
            *color = TextColor(Color::srgb(1.0, 0.3, 0.0));
        } else {
            *color = TextColor(Color::srgb(0.7, 0.7, 0.7));
        }
    }
}
