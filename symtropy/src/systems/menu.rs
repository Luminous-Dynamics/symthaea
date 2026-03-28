// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Main menu and loading screen.

use bevy::prelude::*;

use crate::resources::{DungeonSeed, GamePhase};

/// Marker for menu UI entities (despawned on transition).
#[derive(Component)]
pub struct MenuUi;

/// Marker for loading screen entities.
#[derive(Component)]
pub struct LoadingUi;

/// Spawn the main menu.
pub fn setup_menu(mut commands: Commands) {
    // Root container — full screen, centered
    commands.spawn((
        Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            ..default()
        },
        MenuUi,
    )).with_children(|parent| {
        // Title
        parent.spawn((
            Text::new("SYMTROPY"),
            TextFont { font_size: 64.0, ..default() },
            TextColor(Color::srgb(0.3, 0.9, 0.8)),
        ));

        // Subtitle
        parent.spawn((
            Text::new("The Room That Remembers You"),
            TextFont { font_size: 24.0, ..default() },
            TextColor(Color::srgb(0.6, 0.8, 0.7)),
            Node { margin: UiRect::bottom(Val::Px(8.0)), ..default() },
        ));

        // Lore hint
        parent.spawn((
            Text::new("This room knows your name. It has watched you fail before."),
            TextFont { font_size: 14.0, ..default() },
            TextColor(Color::srgb(0.4, 0.5, 0.45)),
            Node { margin: UiRect::bottom(Val::Px(50.0)), ..default() },
        ));

        // Menu options
        parent.spawn((
            Text::new("[N]  New Game — random dungeon"),
            TextFont { font_size: 20.0, ..default() },
            TextColor(Color::srgb(0.7, 0.9, 0.7)),
            Node { margin: UiRect::bottom(Val::Px(12.0)), ..default() },
        ));

        parent.spawn((
            Text::new("[R]  Replay last seed"),
            TextFont { font_size: 20.0, ..default() },
            TextColor(Color::srgb(0.6, 0.8, 0.6)),
            Node { margin: UiRect::bottom(Val::Px(12.0)), ..default() },
        ));

        parent.spawn((
            Text::new("[Esc]  Quit"),
            TextFont { font_size: 20.0, ..default() },
            TextColor(Color::srgb(0.5, 0.6, 0.5)),
            Node { margin: UiRect::bottom(Val::Px(40.0)), ..default() },
        ));

        // Footer
        parent.spawn((
            Text::new("Powered by Symthaea Consciousness Engine + Mycelix DHT"),
            TextFont { font_size: 12.0, ..default() },
            TextColor(Color::srgb(0.3, 0.4, 0.35)),
        ));
    });

    eprintln!("[symtropy] Main menu displayed");
}

/// Handle menu input.
pub fn menu_input_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut next_state: ResMut<NextState<GamePhase>>,
    mut seed: ResMut<DungeonSeed>,
) {
    if keyboard.just_pressed(KeyCode::KeyN) {
        // New random seed
        seed.0 = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(42);
        eprintln!("[symtropy] New game — seed: {}", seed.0);
        next_state.set(GamePhase::Loading);
    }

    if keyboard.just_pressed(KeyCode::KeyR) {
        // Replay with current seed
        eprintln!("[symtropy] Replay — seed: {}", seed.0);
        next_state.set(GamePhase::Loading);
    }

    if keyboard.just_pressed(KeyCode::Escape) {
        std::process::exit(0);
    }
}

/// Despawn menu UI when leaving MainMenu state.
pub fn cleanup_menu(
    mut commands: Commands,
    query: Query<Entity, With<MenuUi>>,
) {
    for entity in &query {
        commands.entity(entity).despawn();
    }
}

/// Spawn loading screen.
pub fn setup_loading(
    mut commands: Commands,
    seed: Res<DungeonSeed>,
) {
    commands.spawn((
        Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            ..default()
        },
        LoadingUi,
    )).with_children(|parent| {
        parent.spawn((
            Text::new("Generating dungeon..."),
            TextFont { font_size: 28.0, ..default() },
            TextColor(Color::srgb(0.5, 0.8, 0.7)),
        ));

        parent.spawn((
            Text::new(format!("Seed: {}", seed.0)),
            TextFont { font_size: 16.0, ..default() },
            TextColor(Color::srgb(0.4, 0.6, 0.5)),
            Node { margin: UiRect::top(Val::Px(12.0)), ..default() },
        ));
    });
}

/// Cleanup loading screen and transition to Playing.
pub fn cleanup_loading(
    mut commands: Commands,
    query: Query<Entity, With<LoadingUi>>,
) {
    for entity in &query {
        commands.entity(entity).despawn();
    }
}
