// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Input capture: Bevy input events → behavioral biometrics encoder.

use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use symthaea_biometrics::input_telemetry::{KeystrokeSample, MouseSample};

use crate::resources::BiometricsCtx;

/// Capture mouse position and keyboard events into the biometrics encoder.
pub fn input_system(
    windows: Query<&Window, With<PrimaryWindow>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut biometrics: ResMut<BiometricsCtx>,
    time: Res<Time>,
) {
    let ts = (time.elapsed_secs_f64() * 1000.0) as u64;

    // Mouse position → normalized coordinates
    if let Ok(window) = windows.single() {
        if let Some(pos) = window.cursor_position() {
            let w = window.width().max(1.0);
            let h = window.height().max(1.0);
            biometrics.encoder.push_mouse(MouseSample {
                x: pos.x / w,
                y: pos.y / h,
                timestamp_ms: ts,
            });
        }
    }

    // Keystrokes
    for _key in keyboard.get_just_pressed() {
        biometrics.encoder.push_keystroke(KeystrokeSample {
            key_down: true,
            timestamp_ms: ts,
        });
    }
    for _key in keyboard.get_just_released() {
        biometrics.encoder.push_keystroke(KeystrokeSample {
            key_down: false,
            timestamp_ms: ts,
        });
    }

    // Tick the stress model
    let stress = biometrics.encoder.compute_stress_vector();
    biometrics.model.tick(&stress, time.delta_secs());
}
