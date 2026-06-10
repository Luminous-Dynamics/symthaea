// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sovereign Inoculation — Tauri native installer.
//!
//! Wraps the Leptos web UI with native system access:
//! - Direct app scanning (no WebSocket needed)
//! - Hardware detection via native APIs
//! - Filesystem access for config generation
//! - Direct SSH for remote installation
//! - USB device detection and ISO flashing

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod download;
mod flash;
mod usb;

// Re-export scanner types for Tauri commands
use symthaea_scan::scanner;

/// Scan the local system for installed apps and hardware info.
/// Called from the Leptos frontend via Tauri's invoke() bridge.
#[tauri::command]
fn scan_system() -> Result<serde_json::Value, String> {
    let result = scanner::scan();
    serde_json::to_value(&result).map_err(|e| e.to_string())
}

/// Get just hardware info (fast, no app scanning).
#[tauri::command]
fn get_hardware() -> Result<serde_json::Value, String> {
    let result = scanner::scan();
    serde_json::to_value(&result.hardware).map_err(|e| e.to_string())
}

/// Save generated NixOS config to a file.
#[tauri::command]
fn save_config(path: String, content: String) -> Result<(), String> {
    // Validate path — only allow .nix files in reasonable locations
    if !path.ends_with(".nix") {
        return Err("Config files must end in .nix".into());
    }
    std::fs::write(&path, &content).map_err(|e| format!("Failed to save: {e}"))
}

/// List all removable USB storage devices.
#[tauri::command]
fn list_usb_devices() -> Result<Vec<usb::UsbDevice>, String> {
    Ok(usb::list_devices())
}

/// Download the NixForHumanity ISO to a temp directory.
/// Returns the path to the downloaded ISO file.
/// Emits `download-progress` events with `{ downloaded, total, percent, status }`.
#[tauri::command]
async fn download_nixos_iso(app: tauri::AppHandle) -> Result<String, String> {
    let tmp = std::env::temp_dir();
    let path = download::download_iso(app, &tmp).await?;
    Ok(path.to_string_lossy().to_string())
}

/// Flash a downloaded ISO to a USB device.
/// Emits `flash-progress` events with `{ written, total, percent, status }`.
///
/// **Requires elevated privileges** (sudo on Linux/macOS, admin on Windows).
#[tauri::command]
async fn flash_usb_drive(
    app: tauri::AppHandle,
    iso_path: String,
    device: String,
) -> Result<(), String> {
    // Run the blocking flash operation on a dedicated thread
    let app_clone = app.clone();
    tokio::task::spawn_blocking(move || {
        flash::flash_iso(&app_clone, std::path::Path::new(&iso_path), &device)
    })
    .await
    .map_err(|e| format!("Flash task panicked: {}", e))?
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            scan_system,
            get_hardware,
            save_config,
            list_usb_devices,
            download_nixos_iso,
            flash_usb_drive,
        ])
        .run(tauri::generate_context!())
        .expect("error running Sovereign Inoculation");
}