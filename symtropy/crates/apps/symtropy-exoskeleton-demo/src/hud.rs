// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! HUD: exoskeleton telemetry + consciousness readout with assistance mode.

use bevy::prelude::*;
use symthaea_exoskeleton::simulator::ExoskeletonPhysicsSimulator;
use symthaea_exoskeleton::types::AssistanceMode;
use symtropy_consciousness_physics::safety::SafetyTier;

use crate::resources::*;

#[derive(Component)]
pub struct ExoHudText;

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
                    "EXOSKELETON\nCoP: ---\nGRF: ---\nexo effort: ---\nbattery SoC: ---\nhuman torques (rms): ---",
                ),
                TextFont {
                    font_size: 16.0,
                    ..default()
                },
                TextColor(Color::srgb(0.75, 0.90, 1.0)),
                ExoHudText,
            ));

            root.spawn(Node {
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(8.0)),
                ..default()
            })
            .with_child((
                Text::new("CONSCIOUSNESS\nΦ: ---\ntier: ---\nmode: ---\nPE: ---"),
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
    exo: Res<ExoskeletonResources>,
    mut exo_q: Query<&mut Text, (With<ExoHudText>, Without<ConsciousnessHudText>)>,
    mut consc_q: Query<&mut Text, (With<ConsciousnessHudText>, Without<ExoHudText>)>,
) {
    let st = exo.simulator.state();

    let rms: f64 = {
        let n = st.human_torques.len() as f64;
        (st.human_torques.iter().map(|x| x * x).sum::<f64>() / n).sqrt()
    };

    for mut t in &mut exo_q {
        **t = format!(
            "EXOSKELETON\n\
             CoP: ({:+.02}, {:+.02}) m\n\
             GRF: {:.1} N\n\
             exo effort: {:.2}\n\
             battery SoC: {:.1}%\n\
             human torques (rms): {:.2} N·m",
            st.center_of_pressure[0],
            st.center_of_pressure[1],
            st.ground_reaction_force,
            exo.last_exo_effort,
            st.battery_soc * 100.0,
            rms,
        );
    }

    let tier_name = match exo.current_safety {
        SafetyTier::Green => "GREEN  ●",
        SafetyTier::Yellow => "YELLOW ◐",
        SafetyTier::Orange => "ORANGE ◑",
        SafetyTier::Red => "RED    ○",
    };
    let mode_name = match exo.current_mode {
        AssistanceMode::Predictive => "PREDICTIVE   (100% torque)",
        AssistanceMode::Responsive => "RESPONSIVE   (60%)",
        AssistanceMode::Transparent => "TRANSPARENT  (20%)",
        AssistanceMode::GravityCompensation => "GRAVITY-COMP (0%)",
    };

    for mut t in &mut consc_q {
        **t = format!(
            "CONSCIOUSNESS\n\
             Φ: {:.3}\n\
             tier: {tier_name}\n\
             mode: {mode_name}\n\
             PE: {:.3}",
            exo.current_phi, exo.last_prediction_error,
        );
    }
}
