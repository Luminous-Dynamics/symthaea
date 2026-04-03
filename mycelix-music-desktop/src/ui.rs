// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! UI: consciousness-responsive background, large status display, keyboard controls.

use bevy::prelude::*;
use crate::audio::AudioPlaying;
use crate::consciousness::ConsciousnessState;

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_ui)
            .add_systems(Update, update_status)
            .add_systems(Update, update_background)
            .add_systems(Update, keyboard_controls);
    }
}

#[derive(Component)]
struct TitleText;

#[derive(Component)]
struct StatusText;

#[derive(Component)]
struct HelpText;

#[derive(Component)]
struct HarmonyBar(usize);

fn setup_ui(mut commands: Commands) {
    // Root container
    commands.spawn(Node {
        width: Val::Percent(100.0),
        height: Val::Percent(100.0),
        flex_direction: FlexDirection::Column,
        padding: UiRect::all(Val::Px(40.0)),
        ..default()
    }).with_children(|parent| {
        // Title
        parent.spawn((
            Text::new("Mycelix Music"),
            TextFont { font_size: 42.0, ..default() },
            TextColor(Color::srgb(0.5, 0.9, 0.8)),
            TitleText,
        ));

        parent.spawn((
            Text::new("Consciousness Instrument"),
            TextFont { font_size: 18.0, ..default() },
            TextColor(Color::srgb(0.4, 0.5, 0.6)),
        ));

        // Spacer
        parent.spawn(Node { height: Val::Px(30.0), ..default() });

        // Status text (updated each frame)
        parent.spawn((
            Text::new("Initializing..."),
            TextFont { font_size: 32.0, ..default() },
            TextColor(Color::srgb(0.8, 0.85, 0.95)),
            StatusText,
        ));

        // Spacer
        parent.spawn(Node { height: Val::Px(20.0), ..default() });

        // Harmony bars container
        let harmony_names = [
            "Coherence", "Flourishing", "Wisdom", "Play",
            "Interconnect", "Reciprocity", "Progress", "Stillness",
        ];
        let harmony_colors = [
            Color::srgb(1.0, 0.95, 0.8),
            Color::srgb(1.0, 0.7, 0.3),
            Color::srgb(0.3, 0.8, 0.5),
            Color::srgb(0.9, 0.4, 0.8),
            Color::srgb(0.3, 0.6, 1.0),
            Color::srgb(0.7, 0.5, 1.0),
            Color::srgb(1.0, 0.5, 0.3),
            Color::srgb(0.6, 0.9, 1.0),
        ];

        parent.spawn(Node {
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(4.0),
            ..default()
        }).with_children(|bars| {
            for i in 0..8 {
                bars.spawn(Node {
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(10.0),
                    ..default()
                }).with_children(|row| {
                    // Label
                    row.spawn((
                        Text::new(harmony_names[i]),
                        TextFont { font_size: 13.0, ..default() },
                        TextColor(Color::srgb(0.5, 0.5, 0.6)),
                    ));
                    // Bar background
                    row.spawn(Node {
                        width: Val::Px(200.0),
                        height: Val::Px(10.0),
                        ..default()
                    }).with_children(|bg| {
                        bg.spawn((
                            Node {
                                width: Val::Percent(50.0), // will be updated
                                height: Val::Percent(100.0),
                                ..default()
                            },
                            BackgroundColor(harmony_colors[i]),
                            HarmonyBar(i),
                        ));
                    });
                });
            }
        });

        // Spacer to push help to bottom
        parent.spawn(Node { flex_grow: 1.0, ..default() });

        // Help text
        parent.spawn((
            Text::new("A/D: Arousal    W/S: Valence    Q/E: Phi    Esc: Quit"),
            TextFont { font_size: 14.0, ..default() },
            TextColor(Color::srgb(0.35, 0.35, 0.45)),
            HelpText,
        ));
    });
}

fn update_status(
    consciousness: Res<ConsciousnessState>,
    playing: Res<AudioPlaying>,
    mut status_q: Query<(&mut Text, &mut TextColor), With<StatusText>>,
    mut title_q: Query<&mut TextColor, (With<TitleText>, Without<StatusText>)>,
    mut bar_q: Query<(&mut Node, &HarmonyBar), Without<StatusText>>,
) {
    let phi_tier = if consciousness.phi >= 0.8 { "Guardian" }
        else if consciousness.phi >= 0.6 { "Contributor" }
        else if consciousness.phi >= 0.4 { "Participant" }
        else if consciousness.phi >= 0.2 { "Observer" }
        else { "Dormant" };

    let emotion = {
        let a = consciousness.arousal;
        let v = consciousness.valence;
        if a > 0.6 && v > 0.3 { "Joy" }
        else if a > 0.6 && v < -0.3 { "Tension" }
        else if a < 0.4 && v > 0.3 { "Peace" }
        else if a < 0.4 && v < -0.3 { "Melancholy" }
        else { "Neutral" }
    };

    let audio = if playing.0 { "Playing" } else { "Stopped" };

    for (mut text, mut color) in &mut status_q {
        **text = format!(
            "{}\nV: {:+.2}   A: {:.2}   Phi: {:.2} ({})\nAudio: {}",
            emotion, consciousness.valence, consciousness.arousal,
            consciousness.phi, phi_tier, audio
        );
        // Color the status text based on valence
        let v = consciousness.valence;
        let r = 0.7 + v.max(0.0) * 0.2;
        let g = 0.8;
        let b = 0.9 - v.min(0.0).abs() * 0.2;
        *color = TextColor(Color::srgb(r, g, b));
    }

    // Title color from consciousness
    let hue_offset = consciousness.valence * 0.15;
    for mut color in &mut title_q {
        *color = TextColor(Color::srgb(0.5 + hue_offset, 0.9, 0.8 - hue_offset.abs()));
    }

    // Update harmony bars
    for (mut node, bar) in &mut bar_q {
        let activation = consciousness.musical_state.harmony_activations[bar.0];
        node.width = Val::Percent(activation * 100.0);
    }
}

fn update_background(
    consciousness: Res<ConsciousnessState>,
    mut clear_color: ResMut<ClearColor>,
) {
    let v = consciousness.valence;
    let a = consciousness.arousal;
    let p = consciousness.phi;

    let r = (0.04 + v.max(0.0) * 0.06 + a * 0.03 + p * 0.02).clamp(0.02, 0.15);
    let g = (0.03 + p * 0.04 + a * 0.02).clamp(0.02, 0.10);
    let b = (0.06 + (-v).max(0.0) * 0.08 + p * 0.03).clamp(0.03, 0.18);

    clear_color.0 = Color::srgb(r, g, b);
}

fn keyboard_controls(
    keys: Res<ButtonInput<KeyCode>>,
    mut consciousness: ResMut<ConsciousnessState>,
) {
    let step = 0.015;
    let mut changed = false;

    if keys.pressed(KeyCode::KeyD) { consciousness.arousal = (consciousness.arousal + step).min(0.95); changed = true; }
    if keys.pressed(KeyCode::KeyA) { consciousness.arousal = (consciousness.arousal - step).max(0.05); changed = true; }
    if keys.pressed(KeyCode::KeyW) { consciousness.valence = (consciousness.valence + step).min(0.9); changed = true; }
    if keys.pressed(KeyCode::KeyS) { consciousness.valence = (consciousness.valence - step).max(-0.9); changed = true; }
    if keys.pressed(KeyCode::KeyE) { consciousness.phi = (consciousness.phi + step).min(0.95); changed = true; }
    if keys.pressed(KeyCode::KeyQ) { consciousness.phi = (consciousness.phi - step).max(0.05); changed = true; }

    if changed { consciousness.update_musical_state(); }
}
