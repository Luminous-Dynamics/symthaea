// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Terra Atlas Operator Console — Tauri v2 entry point.
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
    state: tauri::State<'_, AppState>,
    project_id: String,
) -> Result<energy_bridge::ConsciousnessScore, String> {
    if !state.conductor.is_connected() {
        return Ok(energy_bridge::ConsciousnessScore {
            project_id,
            phi_score: None,
            harmony_alignment: None,
            assessed_at: None,
        });
    }

    let result = state.conductor.call_zome(
        "energy",
        "energy_bridge",
        "get_latest_project_consciousness",
        serde_json::to_vec(&project_id).unwrap_or_default(),
    );

    match result {
        Ok(zome_result) if zome_result.success => {
            serde_json::from_slice(&zome_result.payload)
                .map_err(|e| format!("Failed to parse consciousness score: {}", e))
        }
        _ => Ok(energy_bridge::ConsciousnessScore {
            project_id,
            phi_score: None,
            harmony_alignment: None,
            assessed_at: None,
        }),
    }
}

/// Get full allocation dashboard for a project.
#[tauri::command]
async fn get_project_dashboard(
    state: tauri::State<'_, AppState>,
    project_id: String,
) -> Result<energy_bridge::AllocationDashboard, String> {
    if !state.conductor.is_connected() {
        return Ok(energy_bridge::AllocationDashboard {
            project_id,
            consciousness: None,
            impact: None,
            pending_pledges: 0,
            matched_pledges: 0,
            total_pledged: 0,
            net_humanity_benefit: 0.0,
            discovery_score: 0.0,
        });
    }

    let result = state.conductor.call_zome(
        "energy",
        "energy_bridge",
        "get_allocation_summary",
        serde_json::to_vec(&project_id).unwrap_or_default(),
    );

    match result {
        Ok(zome_result) if zome_result.success => {
            serde_json::from_slice(&zome_result.payload)
                .map_err(|e| format!("Failed to parse dashboard: {}", e))
        }
        _ => Ok(energy_bridge::AllocationDashboard {
            project_id,
            consciousness: None,
            impact: None,
            pending_pledges: 0,
            matched_pledges: 0,
            total_pledged: 0,
            net_humanity_benefit: 0.0,
            discovery_score: 0.0,
        }),
    }
}

/// Submit a consciousness-gated pledge toward a project.
#[tauri::command]
async fn submit_pledge(
    state: tauri::State<'_, AppState>,
    project_id: String,
    amount: u64,
    currency: String,
    harmony_intent: String,
    consciousness_score: f64,
    consciousness_tier: String,
    pledger_did: String,
) -> Result<String, String> {
    if !state.conductor.is_connected() {
        return Err("Not connected to conductor".into());
    }

    let input = serde_json::json!({
        "project_id": project_id,
        "pledger_did": pledger_did,
        "amount": amount,
        "currency": currency,
        "consciousness_score": consciousness_score,
        "consciousness_tier": consciousness_tier,
        "harmony_intent": harmony_intent,
    });

    let result = state.conductor.call_zome(
        "energy",
        "energy_bridge",
        "submit_pledge",
        serde_json::to_vec(&input).unwrap_or_default(),
    );

    match result {
        Ok(zome_result) if zome_result.success => Ok("Pledge submitted".into()),
        Ok(zome_result) => Err(zome_result.error.unwrap_or_else(|| "Pledge failed".into())),
        Err(e) => Err(format!("Conductor error: {}", e)),
    }
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
            get_project_dashboard,
            submit_pledge,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Terra Atlas Operator");
}
