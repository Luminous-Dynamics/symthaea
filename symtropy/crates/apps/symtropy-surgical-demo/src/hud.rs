// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! HUD: surgical telemetry + consciousness readout with safety level.

use bevy::prelude::*;
use symthaea_surgical::simulator::SurgicalPhysicsSimulator;
use symthaea_surgical::types::SurgicalSafetyLevel;
use symtropy_consciousness_physics::safety::SafetyTier;

use crate::resources::*;

#[derive(Component)]
pub struct SurgicalHudText;

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
                Text::new("SURGICAL\ntip: ---\ndist critical: ---\ntrocar compliance: ---\ntip force: ---\njaw: --- cautery: ---"),
                TextFont {
                    font_size: FontSize::Px(16.0),
                    ..default()
                },
                TextColor(Color::srgb(1.0, 0.85, 0.85)),
                SurgicalHudText,
            ));

            root.spawn(Node {
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(8.0)),
                ..default()
            })
            .with_child((
                Text::new("CONSCIOUSNESS\nΦ: ---\ntier: ---\nsafety level: ---\nPE: ---\n\nCAUTERY DUAL INTERLOCK\n  Φ channel: ---\n  HW channel: ---\n  combined: ---"),
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
    surg: Res<SurgicalResources>,
    mut surg_q: Query<&mut Text, (With<SurgicalHudText>, Without<ConsciousnessHudText>)>,
    mut consc_q: Query<&mut Text, (With<ConsciousnessHudText>, Without<SurgicalHudText>)>,
) {
    let st = surg.simulator.state();

    for mut t in &mut surg_q {
        **t = format!(
            "SURGICAL\n\
             tip: ({:+.1}, {:+.1}, {:+.1}) mm\n\
             dist critical: {:.2} mm\n\
             trocar compliance: {:.2}\n\
             tip force: {:.2} N\n\
             jaw: {:.2}   cautery: {:.2}",
            st.tip_position[0],
            st.tip_position[1],
            st.tip_position[2],
            st.critical_structure_distance,
            st.trocar_compliance,
            st.force_magnitude(),
            surg.last_jaw,
            surg.last_cautery,
        );
    }

    let tier_name = match surg.current_safety {
        SafetyTier::Green => "GREEN  ●",
        SafetyTier::Yellow => "YELLOW ◐",
        SafetyTier::Orange => "ORANGE ◑",
        SafetyTier::Red => "RED    ○",
    };
    let level_name = match surg.current_level {
        SurgicalSafetyLevel::FullControl => "FULL CONTROL  (100% torque)",
        SurgicalSafetyLevel::Reduced => "REDUCED       (40%)",
        SurgicalSafetyLevel::Freeze => "FREEZE        (0%)",
        SurgicalSafetyLevel::Retract => "RETRACT       (0%, pull back)",
    };

    // Dual-channel interlock display: show each channel's verdict and the
    // combined AND. Either channel saying "BLOCKED" blocks energy delivery.
    let ilock = &surg.last_interlock;
    let phi_label = if ilock.phi_channel {
        "ARMED"
    } else {
        "BLOCKED"
    };
    let hw_label = if ilock.hardware_channel {
        "OK"
    } else {
        "BLOCKED"
    };
    let combined_label = if ilock.combined {
        "ARMED ⚡"
    } else {
        "BLOCKED ■"
    };

    for mut t in &mut consc_q {
        **t = format!(
            "CONSCIOUSNESS\n\
             Φ: {:.3}\n\
             tier: {tier_name}\n\
             safety level: {level_name}\n\
             PE: {:.3}\n\
             \n\
             CAUTERY DUAL INTERLOCK\n\
               Φ channel: {phi_label}\n\
               HW channel: {hw_label}  (dist {:.1} mm, force {:.2} N)\n\
               combined:  {combined_label}",
            surg.current_phi, surg.last_prediction_error, ilock.hw_dist_mm, ilock.hw_force_n,
        );
    }
}
