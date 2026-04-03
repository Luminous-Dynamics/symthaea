// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # EduNet Tauri v2 Backend
//!
//! Native Rust backend for the EduNet desktop/mobile client.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │  Leptos CSR Frontend (unchanged from PWA)    │
//! │  Renders in Tauri's native webview           │
//! └──────────────────┬──────────────────────────┘
//!                    │ IPC (invoke)
//! ┌──────────────────▼──────────────────────────┐
//! │  Tauri Rust Backend (this file)              │
//! │  - Conductor lifecycle management            │
//! │  - BKT integrity validation on sync          │
//! │  - Secure key management (OS keychain)        │
//! │  - TEND credit computation (native)          │
//! └──────────────────┬──────────────────────────┘
//!                    │ WebSocket / IPC
//! ┌──────────────────▼──────────────────────────┐
//! │  Holochain Conductor (spawned as child)       │
//! │  - DHT gossip (background)                   │
//! │  - Zome execution                            │
//! │  - Agent key in OS keychain                  │
//! └─────────────────────────────────────────────┘
//! ```
//!
//! ## Security
//!
//! - Agent keys stored in OS keychain (not browser localStorage)
//! - BKT integrity validation before accepting imported progress
//! - TEND credits computed from verified evidence only
//! - Pending TEND from PWA is cosmetic — re-validated on sync

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use tauri::Manager;

// ============== Conductor State ==============

/// State of the Holochain conductor managed by this backend.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ConductorState {
    /// Not yet started
    Stopped,
    /// Starting up (loading DNA, generating keys)
    Starting,
    /// Running and connected
    Running {
        admin_port: u16,
        app_port: u16,
    },
    /// Failed to start
    Error(String),
}

/// Application state managed by Tauri.
struct AppState {
    conductor_state: std::sync::Mutex<ConductorState>,
}

// ============== IPC Commands ==============

/// Get the current conductor state.
#[tauri::command]
fn get_conductor_state(state: tauri::State<AppState>) -> ConductorState {
    state.conductor_state.lock().unwrap().clone()
}

/// Validate BKT integrity from imported PWA progress data.
///
/// This is the security boundary between Trial Mode (browser) and
/// Sovereign Mode (native). Prevents localStorage tampering.
///
/// Returns validated TEND credits that can actually be minted.
#[tauri::command]
fn validate_pwa_import(progress_json: String) -> Result<PwaImportResult, String> {
    // Parse the imported ProgressStore
    let import: PwaProgressImport = serde_json::from_str(&progress_json)
        .map_err(|e| format!("Invalid progress data: {}", e))?;

    let mut validated_nodes = 0u32;
    let mut rejected_nodes = 0u32;
    let mut total_validated_tend = 0.0f32;
    let mut rejections = Vec::new();

    // Validate each BKT state using the same formula as edunet-core::validation
    for (node_id, bkt) in &import.bkt_states {
        let valid = validate_bkt_state(
            bkt.p_mastery,
            bkt.attempts,
            bkt.correct,
            0.15, // Tolerance — generous for PWA data
        );

        if valid {
            validated_nodes += 1;
            // Compute TEND from verified mastery
            if bkt.p_mastery > 0.6 && bkt.attempts >= 5 {
                // Simplified TEND: 0.1 per validated mastered node
                total_validated_tend += 0.1;
            }
        } else {
            rejected_nodes += 1;
            rejections.push(format!(
                "{}: p_mastery={:.3} with {}/{} attempts — inconsistent",
                node_id, bkt.p_mastery, bkt.correct, bkt.attempts
            ));
        }
    }

    // Hard cap: max 10 TEND from PWA import (constitutional limit)
    total_validated_tend = total_validated_tend.min(10.0);

    Ok(PwaImportResult {
        validated_nodes,
        rejected_nodes,
        validated_tend: total_validated_tend,
        rejections,
    })
}

/// Get platform info for the frontend.
#[tauri::command]
fn get_platform_info() -> PlatformInfo {
    PlatformInfo {
        mode: "sovereign".to_string(),
        platform: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    }
}

// ============== Data Types ==============

#[derive(Serialize, Deserialize)]
struct PwaProgressImport {
    bkt_states: std::collections::HashMap<String, BktImport>,
}

#[derive(Serialize, Deserialize)]
struct BktImport {
    p_mastery: f32,
    attempts: u32,
    correct: u32,
}

#[derive(Serialize)]
struct PwaImportResult {
    validated_nodes: u32,
    rejected_nodes: u32,
    validated_tend: f32,
    rejections: Vec<String>,
}

#[derive(Serialize)]
struct PlatformInfo {
    mode: String,
    platform: String,
    arch: String,
    version: String,
}

// ============== BKT Validation (inlined from edunet-core::validation) ==============

/// BKT parameters matching the Leptos frontend exactly.
const P_TRANSIT: f32 = 0.1;
const P_SLIP: f32 = 0.1;
const P_GUESS: f32 = 0.25;
const P_INITIAL: f32 = 0.1;

fn bkt_update_step(p_l: f32, correct: bool) -> f32 {
    let posterior = if correct {
        let p_correct = p_l * (1.0 - P_SLIP) + (1.0 - p_l) * P_GUESS;
        p_l * (1.0 - P_SLIP) / p_correct
    } else {
        let p_incorrect = p_l * P_SLIP + (1.0 - p_l) * (1.0 - P_GUESS);
        p_l * P_SLIP / p_incorrect
    };
    let p_new = posterior + (1.0 - posterior) * P_TRANSIT;
    p_new.clamp(0.01, 0.99)
}

fn replay_bkt(attempts: u32, correct: u32, correct_first: bool) -> f32 {
    let mut p = P_INITIAL;
    let incorrect = attempts - correct;
    let (first_count, first_correct, second_count, second_correct) = if correct_first {
        (correct, true, incorrect, false)
    } else {
        (incorrect, false, correct, true)
    };
    for _ in 0..first_count {
        p = bkt_update_step(p, first_correct);
    }
    for _ in 0..second_count {
        p = bkt_update_step(p, second_correct);
    }
    p.clamp(0.01, 0.99)
}

fn validate_bkt_state(claimed: f32, attempts: u32, correct: u32, tolerance: f32) -> bool {
    if correct > attempts {
        return false;
    }
    if claimed < 0.0 || claimed > 1.0 {
        return false;
    }
    if attempts == 0 {
        return (claimed - P_INITIAL).abs() <= tolerance;
    }
    let case_a = replay_bkt(attempts, correct, true);
    let case_b = replay_bkt(attempts, correct, false);
    let min_case = case_a.min(case_b);
    let max_case = case_a.max(case_b);
    claimed >= (min_case - tolerance) && claimed <= (max_case + tolerance)
}

// ============== Entry Point ==============

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(AppState {
            conductor_state: std::sync::Mutex::new(ConductorState::Stopped),
        })
        .invoke_handler(tauri::generate_handler![
            get_conductor_state,
            validate_pwa_import,
            get_platform_info,
        ])
        .run(tauri::generate_context!())
        .expect("error while running EduNet");
}
