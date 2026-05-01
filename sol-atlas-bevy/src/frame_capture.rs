// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Frame capture system — saves rendered frames as PNGs for video/testing.
//!
//! Captures frames directly from the Bevy render pipeline, bypassing
//! the display server. Press F9 to toggle recording.
//!
//! Stitch frames into video: ffmpeg -framerate 24 -i frame_%05d.png -c:v libx264 demo.mp4

use bevy::prelude::*;
use bevy::render::view::screenshot::{save_to_disk, Screenshot};

/// Configuration for frame capture.
#[derive(Resource)]
pub struct FrameCaptureConfig {
    pub output_dir: String,
    pub interval_secs: f32,
    pub max_frames: u32,
    pub frame_count: u32,
    pub accumulator: f32,
    pub active: bool,
}

impl Default for FrameCaptureConfig {
    fn default() -> Self {
        Self {
            output_dir: "/tmp/terra-atlas-frames".into(),
            interval_secs: 1.0 / 4.0, // 4fps — gives async screenshots time to resolve
            max_frames: 720,          // 90 seconds at 8fps
            frame_count: 0,
            accumulator: 0.0,
            active: false,
        }
    }
}

/// System that captures frames. Press F9 to toggle.
pub fn frame_capture_system(
    mut commands: Commands,
    time: Res<Time>,
    mut config: ResMut<FrameCaptureConfig>,
    kb: Res<ButtonInput<KeyCode>>,
    windows: Query<Entity, With<Window>>,
) {
    if kb.just_pressed(KeyCode::F9) {
        config.active = !config.active;
        if config.active {
            config.frame_count = 0;
            config.accumulator = 0.0;
            let _ = std::fs::create_dir_all(&config.output_dir);
            info!("[capture] Recording started → {}", config.output_dir);
        } else {
            info!("[capture] Stopped. {} frames. Stitch: ffmpeg -framerate 24 -i {}/frame_%05d.png -c:v libx264 -pix_fmt yuv420p demo.mp4",
                config.frame_count, config.output_dir);
        }
    }

    if !config.active {
        return;
    }

    if config.max_frames > 0 && config.frame_count >= config.max_frames {
        config.active = false;
        info!("[capture] Max frames ({}). Stitch: ffmpeg -framerate 24 -i {}/frame_%05d.png -c:v libx264 -pix_fmt yuv420p demo.mp4",
            config.frame_count, config.output_dir);
        return;
    }

    config.accumulator += time.delta_secs();

    if config.accumulator >= config.interval_secs {
        config.accumulator -= config.interval_secs;

        let frame_path = format!("{}/frame_{:05}.png", config.output_dir, config.frame_count);

        // Capture from the primary window
        commands
            .spawn(Screenshot(bevy_camera::RenderTarget::Window(
                bevy::window::WindowRef::Primary,
            )))
            .observe(save_to_disk(frame_path));
        config.frame_count += 1;
    }
}
