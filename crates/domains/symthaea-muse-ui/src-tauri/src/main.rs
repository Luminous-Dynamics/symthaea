// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Muse Desktop — a Tauri v2 native wrapper giving Muse a real taskbar/tray
//! icon instead of two manually-run terminal commands + a browser tab.
//!
//! Deliberately does NOT embed `muse_studio`'s axum server in-process (that
//! would mean changing `symthaea-muse`'s bin-vs-lib boundary, real surgery
//! on a crate other work touches concurrently) — instead it spawns the
//! existing `muse_studio` binary as a child process on startup via `cargo
//! run` against the workspace (reuses whatever's already built/cached,
//! same as this whole project's manual dev workflow) and kills it when the
//! window closes. The webview loads the already-built `symthaea-muse-ui`
//! frontend (`frontendDist` in `tauri.conf.json`), which talks to the
//! spawned backend on `127.0.0.1:8400` exactly as it does today.
//!
//! Follows the tray-icon pattern already established in this monorepo by
//! `mycelix-workspace/happs/lucid/ui/src-tauri/src/main.rs` (Show/Quit menu
//! + spawning a background server from `setup()`) rather than inventing a
//! new one.

use std::path::PathBuf;
use std::process::Child;
use std::sync::Mutex;

use tauri::Manager;
use tauri::menu::{Menu, MenuItem};
use tauri::tray::TrayIconBuilder;

struct BackendProcess(Mutex<Option<Child>>);

/// The `symthaea` workspace root, computed from this crate's own manifest
/// path at compile time — `src-tauri` sits at
/// `crates/domains/symthaea-muse-ui/src-tauri`, four levels below
/// `symthaea/`.
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../..")
        .canonicalize()
        .expect("symthaea workspace root must exist relative to this crate")
}

fn spawn_backend() -> std::io::Result<Child> {
    std::process::Command::new("cargo")
        .args([
            "run",
            "-p",
            "symthaea-muse",
            "--bin",
            "muse_studio",
            "--features",
            "studio",
            "--release",
        ])
        .current_dir(workspace_root())
        .spawn()
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(BackendProcess(Mutex::new(None)))
        .setup(|app| {
            match spawn_backend() {
                Ok(child) => {
                    let state = app.state::<BackendProcess>();
                    *state.0.lock().unwrap() = Some(child);
                }
                Err(e) => {
                    eprintln!("muse-desktop: failed to spawn muse_studio backend: {e}");
                }
            }

            let quit = MenuItem::with_id(app, "quit", "Quit Muse", true, None::<&str>)?;
            let show = MenuItem::with_id(app, "show", "Show Muse", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&show, &quit])?;

            let _tray = TrayIconBuilder::new()
                .menu(&menu)
                .tooltip("Muse by Symthaea")
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "quit" => app.exit(0),
                    "show" => {
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                    }
                    _ => {}
                })
                .build(app)?;

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building the Muse Tauri app")
        .run(|app_handle, event| {
            // Kill the spawned backend when the app actually exits — an
            // orphaned `muse_studio` would otherwise keep holding port 8400.
            if let tauri::RunEvent::ExitRequested { .. } = event {
                let state = app_handle.state::<BackendProcess>();
                if let Some(mut child) = state.0.lock().unwrap().take() {
                    let _ = child.kill();
                }
            }
        });
}
