// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Knowledge — Tauri v2 backend.
//!
//! Provides native capabilities: conductor lifecycle, platform info,
//! and cluster-specific IPC commands.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::Serialize;

#[derive(Clone, Serialize)]
struct PlatformInfo {
    mode: &'static str,
    platform: &'static str,
    arch: &'static str,
}

#[tauri::command]
fn get_platform_info() -> PlatformInfo {
    PlatformInfo {
        mode: "sovereign",
        platform: std::env::consts::OS,
        arch: std::env::consts::ARCH,
    }
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            get_platform_info,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Knowledge");
}
