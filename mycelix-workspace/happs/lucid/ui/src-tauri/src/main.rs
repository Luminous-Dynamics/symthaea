// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! LUCID Desktop Application
//!
//! Tauri 2.0 native wrapper for the LUCID personal knowledge graph.
//!
//! ## Symthaea Integration
//!
//! This application integrates Symthaea for:
//! - 16,384-dimensional HDC embeddings (replacing transformers.js)
//! - E/N/M/H epistemic classification
//! - Thought coherence analysis
//! - Consciousness-aware semantic search

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod symthaea_bridge;
mod mind_bridge;
mod api_server;
mod zkp_bridge;
mod governance_bridge;

use symthaea_bridge::SymthaeaState;
use mind_bridge::LucidMindState;
use api_server::ApiState;
use zkp_bridge::ZkpState;
use governance_bridge::GovernanceState;
use std::sync::Arc;
use tauri::{
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    Emitter, Manager,
};

fn main() {
    // Initialize tracing for diagnostics
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("lucid=debug".parse().unwrap())
                .add_directive("symthaea=info".parse().unwrap()),
        )
        .init();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_clipboard_manager::init())
        // Manage Symthaea state
        .manage(SymthaeaState::new())
        // Manage LucidMind state (persistent working memory)
        .manage(LucidMindState::new())
        // Manage ZKP state for anonymous proofs
        .manage(ZkpState::new())
        // Manage governance state (cached Phi tallies from frontend)
        .manage(GovernanceState::new())
        .setup(|app| {
            // Create system tray menu
            let quit = MenuItem::with_id(app, "quit", "Quit LUCID", true, None::<&str>)?;
            let show = MenuItem::with_id(app, "show", "Show Window", true, None::<&str>)?;
            let capture = MenuItem::with_id(app, "capture", "Quick Capture", true, Some("CmdOrCtrl+Shift+L"))?;

            let menu = Menu::with_items(app, &[&capture, &show, &quit])?;

            // Build tray icon
            let _tray = TrayIconBuilder::new()
                .menu(&menu)
                .tooltip("LUCID - Personal Knowledge Graph")
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "quit" => {
                        app.exit(0);
                    }
                    "show" => {
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                    }
                    "capture" => {
                        // Emit quick capture event to frontend
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.emit("quick-capture", ());
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                    }
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        let app = tray.app_handle();
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                    }
                })
                .build(app)?;

            // Start local API server for browser extension
            let api_state = Arc::new(ApiState::new());
            let api_state_for_server = Arc::clone(&api_state);
            let api_state_for_symthaea = Arc::clone(&api_state);

            tauri::async_runtime::spawn(async move {
                tracing::info!(target: "lucid", "Starting local API server for browser extension...");
                if let Err(e) = api_server::start_api_server(api_state_for_server).await {
                    tracing::error!(target: "lucid", error = %e, "API server failed to start");
                }
            });

            // Initialize Symthaea in background
            let app_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                tracing::info!(target: "lucid", "Starting Symthaea initialization...");

                // Give the window time to load first
                tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

                // Get state and initialize
                let state = app_handle.state::<SymthaeaState>();
                match symthaea_bridge::initialize_symthaea(state).await {
                    Ok(response) => {
                        if response.success {
                            tracing::info!(
                                target: "lucid",
                                dimension = response.dimension,
                                "Symthaea ready for semantic search"
                            );

                            // Update API state
                            api_state_for_symthaea.set_symthaea_ready(true).await;

                            // Notify frontend that Symthaea is ready
                            if let Some(window) = app_handle.get_webview_window("main") {
                                let _ = window.emit("symthaea-ready", response);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            target: "lucid",
                            error = %e,
                            "Symthaea initialization failed, falling back to local search"
                        );
                    }
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // Original commands
            get_app_data_dir,
            show_notification,
            // Symthaea integration commands
            symthaea_bridge::initialize_symthaea,
            symthaea_bridge::symthaea_ready,
            symthaea_bridge::analyze_thought,
            symthaea_bridge::semantic_search,
            symthaea_bridge::embed_text,
            symthaea_bridge::batch_embed,
            symthaea_bridge::compute_similarity,
            symthaea_bridge::find_similar,
            symthaea_bridge::check_coherence,
            symthaea_bridge::classify_epistemic,
            symthaea_bridge::get_hdc_dimension,
            symthaea_bridge::get_symthaea_status,
            // LucidMind (ContinuousMind wrapper) commands
            mind_bridge::initialize_lucid_mind,
            mind_bridge::lucid_mind_ready,
            mind_bridge::seed_working_memory,
            mind_bridge::clear_working_memory,
            mind_bridge::analyze_with_context,
            mind_bridge::get_mind_consciousness_profile,
            mind_bridge::get_mind_state,
            mind_bridge::get_session_memory,
            mind_bridge::mind_tick,
            mind_bridge::take_consciousness_snapshot,
            mind_bridge::get_consciousness_evolution_summary,
            // Coherence feedback loop commands
            symthaea_bridge::auto_check_coherence,
            symthaea_bridge::backfill_embeddings,
            // ZKP (Zero-Knowledge Proof) commands
            zkp_bridge::generate_anonymous_belief_proof,
            zkp_bridge::generate_reputation_range_proof,
            zkp_bridge::verify_proof,
            zkp_bridge::create_value_commitment,
            zkp_bridge::hash_belief_content,
            zkp_bridge::zkp_ready,
            // Governance bridge commands (Phi-weighted tally caching)
            governance_bridge::cache_phi_tally,
            governance_bridge::get_phi_tally,
            governance_bridge::get_governance_stats,
            governance_bridge::clear_governance_cache,
        ])
        .run(tauri::generate_context!())
        .expect("error while running LUCID");
}

/// Get the application data directory
#[tauri::command]
fn get_app_data_dir(app: tauri::AppHandle) -> Result<String, String> {
    app.path()
        .app_data_dir()
        .map(|p| p.to_string_lossy().to_string())
        .map_err(|e| e.to_string())
}

/// Show a system notification
#[tauri::command]
fn show_notification(
    app: tauri::AppHandle,
    title: String,
    body: String,
) -> Result<(), String> {
    use tauri_plugin_notification::NotificationExt;

    app.notification()
        .builder()
        .title(&title)
        .body(&body)
        .show()
        .map_err(|e| e.to_string())
}
