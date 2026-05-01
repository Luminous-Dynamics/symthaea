// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Symtropy Engine — consciousness-first technology platform.
//!
//! Experiences:
//! - The Room That Remembers You (consciousness survival horror)
//! - Sol Atlas (civilizational planetary instrument)
//!
//! Run:
//! ```sh
//! export LD_LIBRARY_PATH="$(cat /tmp/bevy_ld_library_path.txt)"
//! RUST_LOG=warn,symtropy=info ./target/release/symtropy
//! ```

mod components;
pub mod experience;
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
    eprintln!(
        "[symtropy] DISPLAY={}",
        std::env::var("DISPLAY").unwrap_or_default()
    );
    eprintln!(
        "[symtropy] WAYLAND_DISPLAY={}",
        std::env::var("WAYLAND_DISPLAY").unwrap_or_default()
    );

    // Configure wgpu to be more resilient — prefer Vulkan, fall back to GL
    let wgpu_settings = WgpuSettings {
        // Let wgpu choose the best available backend
        ..default()
    };

    // CLI flags
    let autostart = std::env::args().any(|a| a == "--autostart");
    let ai_player = std::env::args().any(|a| a == "--ai-player");
    #[cfg(feature = "atlas")]
    let globe_mode = std::env::args().any(|a| a == "--globe");
    #[cfg(feature = "atlas")]
    let record_mode = std::env::args().any(|a| a == "--record");
    #[cfg(feature = "atlas")]
    let demo_mode = std::env::args().any(|a| a == "--demo");
    #[cfg(feature = "atlas")]
    let cinematic_mode = std::env::args().any(|a| a == "--cinematic");

    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Symtropy Engine".into(),
                    resolution: bevy::window::WindowResolution::new(1280, 720),
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
    // MSAA: Bevy 0.18 defaults to Sample4 via Msaa enum (set per-camera if needed)
    .add_plugins(SymtropyPlugin)
    .add_systems(Startup, log_renderer_info);

    if autostart || ai_player {
        eprintln!("[symtropy] --autostart: skipping menu, starting game immediately");
        app.add_systems(
            Startup,
            |mut next: ResMut<bevy::prelude::NextState<crate::resources::GamePhase>>| {
                next.set(crate::resources::GamePhase::Loading);
            },
        );
    }

    // --globe: enter globe view from main menu (requires --features atlas)
    #[cfg(feature = "atlas")]
    if globe_mode {
        eprintln!("[symtropy] --globe: entering Sol Atlas globe view from MainMenu");
        app.add_systems(
            Update,
            (|mut next: ResMut<bevy::prelude::NextState<crate::resources::GamePhase>>| {
                next.set(crate::resources::GamePhase::GlobeView);
            })
            .run_if(bevy::prelude::in_state(
                crate::resources::GamePhase::MainMenu,
            )),
        );
    }

    // --demo: auto-start demo director when globe view activates
    #[cfg(feature = "atlas")]
    if demo_mode && globe_mode {
        eprintln!("[symtropy] --demo: cinematic demo director enabled");
        app.add_systems(
            Update,
            (|mut director: ResMut<crate::systems::demo_director::DemoDirector>| {
                if !director.enabled {
                    director.enabled = true;
                    info!("[demo] Director started — 30s cinematic sequence");
                }
            })
            .run_if(bevy::prelude::in_state(
                crate::resources::GamePhase::GlobeView,
            )),
        );
    }

    // --record: auto-start frame capture when globe view activates
    #[cfg(feature = "atlas")]
    if record_mode && globe_mode {
        eprintln!("[symtropy] --record: auto-capturing frames to /tmp/terra-atlas-frames/");
        app.add_systems(
            Update,
            (|mut config: ResMut<sol_atlas_bevy::frame_capture::FrameCaptureConfig>| {
                if !config.active && config.frame_count == 0 {
                    config.active = true;
                    let _ = std::fs::create_dir_all(&config.output_dir);
                }
            })
            .run_if(bevy::prelude::in_state(
                crate::resources::GamePhase::GlobeView,
            )),
        );
    }

    // --cinematic: full-spectrum 90s cinematic demo
    #[cfg(feature = "atlas")]
    if cinematic_mode {
        eprintln!("[symtropy] --cinematic: full-spectrum cinematic (90s, 18 phases)");
        // Insert cinematic resources
        app.insert_resource(crate::systems::cinematic_director::CinematicDirector {
            enabled: true,
            ..Default::default()
        });
        app.insert_resource(crate::systems::cinematic_director::ScreenFade::default());
        app.insert_resource(crate::systems::cinematic_director::NarrationState::default());
        app.insert_resource(crate::systems::cinematic_director::VoiceNarration::default());
        // Frame capture with cinematic output dir
        app.insert_resource(sol_atlas_bevy::frame_capture::FrameCaptureConfig {
            output_dir: "/tmp/symtropy-cinematic-frames".into(),
            max_frames: 720,
            active: true,
            ..Default::default()
        });
        // AI player (starts disabled, cinematic enables it during dungeon phases)
        let ai = crate::systems::ai_player::AiPlayer::new();
        app.insert_resource(ai);
        // Auto-start Loading phase
        app.add_systems(
            Update,
            (|mut next: ResMut<bevy::prelude::NextState<crate::resources::GamePhase>>| {
                next.set(crate::resources::GamePhase::Loading);
            })
            .run_if(bevy::prelude::in_state(
                crate::resources::GamePhase::MainMenu,
            )),
        );
    }

    if ai_player {
        eprintln!("[symtropy] --ai-player: Symthaea is playing the game");
        let mut ai = crate::systems::ai_player::AiPlayer::new();
        ai.enabled = true;
        app.insert_resource(ai);
        app.add_systems(
            Update,
            crate::systems::ai_player::ai_player_system.run_if(bevy::prelude::in_state(
                crate::resources::GamePhase::Playing,
            )),
        );
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
