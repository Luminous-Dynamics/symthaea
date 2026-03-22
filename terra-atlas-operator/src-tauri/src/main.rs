//! Terra Atlas Operator Console — Tauri v2 entry point.
//!
//! Provides a native desktop application for energy project operators,
//! connecting directly to a Holochain conductor for decentralized
//! project management, investment tracking, and consciousness scoring.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod conductor_bridge;
mod energy_bridge;

use conductor_bridge::{ConnectionStatus, ConductorConnection, ExternalConductor};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Application state shared across Tauri commands.
struct AppState {
    /// Conductor connection.
    conductor: Arc<ExternalConductor>,
}

// ─── Tauri Commands ─────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct HealthStatus {
    conductor: ConnectionStatus,
    version: String,
}

/// Health check — returns conductor connection status and app version.
#[tauri::command]
async fn health_check(state: tauri::State<'_, AppState>) -> Result<HealthStatus, String> {
    Ok(HealthStatus {
        conductor: state.conductor.status(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// Connect to the Holochain conductor.
#[tauri::command]
async fn connect_conductor(
    state: tauri::State<'_, AppState>,
    url: Option<String>,
) -> Result<ConnectionStatus, String> {
    // If a new URL is provided, we'd need to recreate the conductor
    // For now, just try connecting with the configured URL
    if let Some(ref _new_url) = url {
        // Future: recreate ExternalConductor with new URL
    }

    state.conductor.try_connect().await?;
    Ok(state.conductor.status())
}

/// Disconnect from the Holochain conductor.
#[tauri::command]
async fn disconnect_conductor(state: tauri::State<'_, AppState>) -> Result<(), String> {
    state.conductor.disconnect();
    Ok(())
}

/// Get all energy projects from the conductor.
#[tauri::command]
async fn get_projects(
    state: tauri::State<'_, AppState>,
) -> Result<Vec<energy_bridge::ProjectSummary>, String> {
    if !state.conductor.is_connected() {
        return Ok(Vec::new()); // Empty when disconnected
    }

    let result = state.conductor.call_zome(
        "energy",
        "projects_coordinator",
        "get_all_projects",
        vec![],
    );

    match result {
        Ok(zome_result) if zome_result.success => {
            // Deserialize the zome response into ProjectSummary vec
            serde_json::from_slice(&zome_result.payload)
                .map_err(|e| format!("Failed to parse projects: {}", e))
        }
        Ok(zome_result) => Err(zome_result.error.unwrap_or_else(|| "Unknown error".into())),
        Err(e) => {
            // Expected during Phase 2 — conductor connected but zome calls
            // need holochain_client. Return demo data instead.
            eprintln!("Zome call pending: {}", e);
            Ok(energy_bridge::demo_projects())
        }
    }
}

/// Get consciousness score for a specific project.
#[tauri::command]
async fn get_project_consciousness(
    project_id: String,
) -> Result<energy_bridge::ConsciousnessScore, String> {
    // Stub: returns default scores until conductor bridge is fully wired
    Ok(energy_bridge::ConsciousnessScore {
        project_id,
        phi_score: None,
        harmony_alignment: None,
        assessed_at: None,
    })
}

// ─── Entry Point ────────────────────────────────────────────────────────────

fn main() {
    let conductor_url =
        std::env::var("CONDUCTOR_URL").unwrap_or_else(|_| "ws://localhost:8888".to_string());

    let conductor = Arc::new(ExternalConductor::new(&conductor_url));

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_store::Builder::default().build())
        .manage(AppState {
            conductor,
        })
        .invoke_handler(tauri::generate_handler![
            health_check,
            connect_conductor,
            disconnect_conductor,
            get_projects,
            get_project_consciousness,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Terra Atlas Operator");
}
