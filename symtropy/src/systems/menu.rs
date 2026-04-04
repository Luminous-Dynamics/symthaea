// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Symtropy Nexus — extensible engine launcher with living background.
//!
//! Reads from ExperienceRegistry to build the menu dynamically.
//! Background is a breathing mycelial network rendered via gizmos.

use bevy::prelude::*;

use crate::experience::ExperienceRegistry;
use crate::resources::{DungeonSeed, GamePhase};

/// Marker for Nexus UI entities (despawned on transition).
#[derive(Component)]
pub struct MenuUi;

/// Marker for loading screen entities.
#[derive(Component)]
pub struct LoadingUi;

/// Marker for the selection indicator text.
#[derive(Component)]
pub struct SelectionIndicator(pub usize);

/// Spawn the Symtropy Nexus launcher.
pub fn setup_menu(
    mut commands: Commands,
    registry: Res<ExperienceRegistry>,
) {
    // Deep space background
    commands.insert_resource(ClearColor(Color::linear_rgb(0.01, 0.008, 0.02)));

    // Root container — full screen, centered column
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
        // ═══ TITLE ═══════════════════════════════════════
        parent.spawn((
            Text::new("SYMTROPY"),
            TextFont { font_size: 56.0, ..default() },
            TextColor(Color::srgb(0.3, 0.9, 0.8)),
        ));

        parent.spawn((
            Text::new("consciousness-first technology"),
            TextFont { font_size: 16.0, ..default() },
            TextColor(Color::srgba(0.5, 0.7, 0.65, 0.6)),
            Node { margin: UiRect::bottom(Val::Px(40.0)), ..default() },
        ));

        // ═══ EXPERIENCE ENTRIES ══════════════════════════
        for (i, exp) in registry.experiences.iter().enumerate() {
            let is_selected = i == registry.selected;
            let alpha = if is_selected { 1.0 } else { 0.5 };
            let prefix = if is_selected { "▸ " } else { "  " };

            // Experience name
            parent.spawn((
                Text::new(format!("{}[{}]  {}", prefix, i + 1, exp.name)),
                TextFont { font_size: 22.0, ..default() },
                TextColor(Color::srgba(exp.icon_color[0], exp.icon_color[1], exp.icon_color[2], alpha)),
                Node { margin: UiRect::bottom(Val::Px(4.0)), ..default() },
                SelectionIndicator(i),
            ));

            // Subtitle
            parent.spawn((
                Text::new(format!("      {}", exp.subtitle)),
                TextFont { font_size: 13.0, ..default() },
                TextColor(Color::srgba(0.5, 0.6, 0.55, alpha * 0.6)),
                Node { margin: UiRect::bottom(Val::Px(16.0)), ..default() },
                SelectionIndicator(i),
            ));
        }

        // Settings option
        parent.spawn((
            Text::new("  [S]  Settings"),
            TextFont { font_size: 18.0, ..default() },
            TextColor(Color::srgba(0.4, 0.5, 0.45, 0.4)),
            Node { margin: UiRect::bottom(Val::Px(8.0)), ..default() },
        ));

        parent.spawn((
            Text::new("  [Esc]  Quit"),
            TextFont { font_size: 18.0, ..default() },
            TextColor(Color::srgba(0.4, 0.45, 0.4, 0.4)),
            Node { margin: UiRect::bottom(Val::Px(40.0)), ..default() },
        ));

        // ═══ FOOTER ═════════════════════════════════════
        parent.spawn((
            Text::new("Powered by Symthaea · Mycelix · Eight Harmonies"),
            TextFont { font_size: 11.0, ..default() },
            TextColor(Color::srgba(0.3, 0.4, 0.35, 0.4)),
        ));
    });

    eprintln!("[symtropy] Nexus displayed — {} experiences available", registry.experiences.len());
}

/// Nexus input — navigate and launch experiences.
pub fn menu_input_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut next_state: ResMut<NextState<GamePhase>>,
    mut seed: ResMut<DungeonSeed>,
    mut registry: ResMut<ExperienceRegistry>,
    mut indicators: Query<(&SelectionIndicator, &mut TextColor)>,
) {
    let count = registry.experiences.len();

    // Navigate: Up/Down arrows
    if keyboard.just_pressed(KeyCode::ArrowUp) && registry.selected > 0 {
        registry.selected -= 1;
    }
    if keyboard.just_pressed(KeyCode::ArrowDown) && registry.selected < count - 1 {
        registry.selected += 1;
    }

    // Quick-select: number keys
    if keyboard.just_pressed(KeyCode::Digit1) && count > 0 { registry.selected = 0; }
    if keyboard.just_pressed(KeyCode::Digit2) && count > 1 { registry.selected = 1; }
    if keyboard.just_pressed(KeyCode::Digit3) && count > 2 { registry.selected = 2; }

    // Update visual selection
    for (indicator, mut color) in indicators.iter_mut() {
        let exp = &registry.experiences[indicator.0];
        let is_selected = indicator.0 == registry.selected;
        let alpha = if is_selected { 1.0 } else { 0.4 };
        *color = TextColor(Color::srgba(
            exp.icon_color[0], exp.icon_color[1], exp.icon_color[2], alpha
        ));
    }

    // Launch: Enter or N
    if keyboard.just_pressed(KeyCode::Enter) || keyboard.just_pressed(KeyCode::KeyN) {
        let exp = &registry.experiences[registry.selected];
        eprintln!("[nexus] Launching: {}", exp.name);

        if exp.phase == GamePhase::Loading {
            seed.0 = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(42);
        }
        next_state.set(exp.phase);
    }

    // Replay (for The Room)
    if keyboard.just_pressed(KeyCode::KeyR) {
        eprintln!("[nexus] Replay — seed: {}", seed.0);
        next_state.set(GamePhase::Loading);
    }

    if keyboard.just_pressed(KeyCode::Escape) {
        std::process::exit(0);
    }
}

/// Living mycelial background — breathing network of nodes and connections.
pub fn nexus_background_system(
    mut gizmos: Gizmos,
    time: Res<Time>,
) {
    let t = time.elapsed_secs();
    let node_count = 60;

    // Generate deterministic node positions with slow drift
    for i in 0..node_count {
        let seed = i as f32 * 1.618;
        let base_x = (seed * 7.3).sin() * 400.0;
        let base_y = (seed * 3.7).cos() * 280.0;
        // Slow Brownian drift
        let drift_x = (t * 0.05 + seed * 2.1).sin() * 30.0;
        let drift_y = (t * 0.07 + seed * 1.3).cos() * 20.0;
        let x = base_x + drift_x;
        let y = base_y + drift_y;

        // Node glow
        let flicker = (t * 2.0 + seed).sin().abs() * 0.3 + 0.2;
        let node_color = Color::linear_rgba(0.0, 0.4 * flicker, 0.5 * flicker, flicker * 0.4);
        gizmos.circle_2d(Vec2::new(x, y), 2.0, node_color);

        // Connect to nearby nodes (deterministic pairs)
        for j in (i + 1)..node_count {
            let seed_j = j as f32 * 1.618;
            let jx = (seed_j * 7.3).sin() * 400.0 + (t * 0.05 + seed_j * 2.1).sin() * 30.0;
            let jy = (seed_j * 3.7).cos() * 280.0 + (t * 0.07 + seed_j * 1.3).cos() * 20.0;
            let dist = ((x - jx).powi(2) + (y - jy).powi(2)).sqrt();

            if dist < 120.0 && (i + j) % 3 == 0 {
                // Pulse along connection
                let pulse = (t * 0.3 + (i + j) as f32 * 0.5).sin().abs() * 0.15 + 0.05;
                let edge_color = Color::linear_rgba(0.0, 0.3 * pulse, 0.4 * pulse, pulse);
                gizmos.line_2d(Vec2::new(x, y), Vec2::new(jx, jy), edge_color);
            }
        }
    }
}

/// Despawn Nexus UI when leaving MainMenu state.
pub fn cleanup_menu(
    mut commands: Commands,
    query: Query<Entity, With<MenuUi>>,
) {
    for entity in &query {
        commands.entity(entity).despawn();
    }
}

/// Spawn loading screen.
pub fn setup_loading(mut commands: Commands, seed: Res<DungeonSeed>) {
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

/// Cleanup loading screen.
pub fn cleanup_loading(mut commands: Commands, query: Query<Entity, With<LoadingUi>>) {
    for entity in &query {
        commands.entity(entity).despawn();
    }
}
