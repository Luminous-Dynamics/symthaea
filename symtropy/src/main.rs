// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Symtropy: consciousness-driven survival game.
//!
//! Run:
//! ```sh
//! export LD_LIBRARY_PATH="$(cat /tmp/bevy_ld_library_path.txt)"
//! RUST_LOG=warn,symtropy=info ./target/debug/symtropy
//! ```

mod components;
mod plugin;
mod resources;
mod systems;

use bevy::prelude::*;
use bevy::render::settings::{RenderCreation, WgpuSettings};
use bevy::render::RenderPlugin;
use plugin::SymtropyPlugin;

fn main() {
    // Crash handler — writes to both stderr and a crash log file
    std::panic::set_hook(Box::new(|info| {
        let bt = std::backtrace::Backtrace::force_capture();
        let msg = format!(
            "\n=== SYMTROPY CRASH ===\n{info}\n\nBacktrace:\n{bt}\n======================\n"
        );
        eprintln!("{msg}");
        // Also write to file so we can inspect after window closes
        let _ = std::fs::write("/tmp/symtropy_panic.log", &msg);
    }));

    eprintln!("[symtropy] Starting...");
    eprintln!("[symtropy] DISPLAY={}", std::env::var("DISPLAY").unwrap_or_default());
    eprintln!("[symtropy] WAYLAND_DISPLAY={}", std::env::var("WAYLAND_DISPLAY").unwrap_or_default());

    // Configure wgpu to be more resilient — prefer Vulkan, fall back to GL
    let wgpu_settings = WgpuSettings {
        // Let wgpu choose the best available backend
        ..default()
    };

    // CLI flags
    let autostart = std::env::args().any(|a| a == "--autostart");
    let ai_player = std::env::args().any(|a| a == "--ai-player");

    let mut app = App::new();
    app.add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Symtropy: The Room That Remembers You".into(),
                        resolution: bevy::window::WindowResolution::new(960, 720),
                        present_mode: bevy::window::PresentMode::AutoVsync,
                        ..default()
                    }),
                    ..default()
                })
                .set(RenderPlugin {
                    render_creation: RenderCreation::Automatic(wgpu_settings),
                    ..default()
                })
                .set(ImagePlugin::default_nearest()),
        )
        .add_plugins(SymtropyPlugin)
        .add_systems(Startup, log_renderer_info);

    if autostart || ai_player {
        eprintln!("[symtropy] --autostart: skipping menu, starting game immediately");
        app.add_systems(Startup, |mut next: ResMut<bevy::prelude::NextState<crate::resources::GamePhase>>| {
            next.set(crate::resources::GamePhase::Loading);
        });
    }

    if ai_player {
        eprintln!("[symtropy] --ai-player: Symthaea is playing the game");
        let mut ai = crate::systems::ai_player::AiPlayer::new();
        ai.enabled = true;
        app.insert_resource(ai);
        app.add_systems(Update, crate::systems::ai_player::ai_player_system
            .run_if(bevy::prelude::in_state(crate::resources::GamePhase::Playing)));
    } else {
        app.insert_resource(crate::systems::ai_player::AiPlayer::new());
    }

    app.run();

    eprintln!("[symtropy] Clean exit.");
}

/// Log renderer info on startup to help debug GPU issues.
fn log_renderer_info() {
    eprintln!("[symtropy] Renderer initialized successfully");
    eprintln!("[symtropy] Window should be visible — if black, check GPU drivers");
}

// Window icon: use symtropy.desktop file for Linux desktop integration.
// Bevy 0.18 doesn't expose winit window icon API directly.
// Install: cp symtropy.desktop ~/.local/share/applications/
