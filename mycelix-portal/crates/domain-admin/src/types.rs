// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Admin/IT types — system health, conductor status, RDP sessions.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SystemHealth {
    pub conductor_status: ConductorHealth,
    pub wasm_modules: Vec<WasmModule>,
    pub memory_usage_mb: f64,
    pub uptime_seconds: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConductorHealth {
    pub url: String,
    pub connected: bool,
    pub installed_apps: Vec<InstalledApp>,
    pub agent_count: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstalledApp {
    pub app_id: String,
    pub status: String,
    pub cell_count: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WasmModule {
    pub name: String,
    pub size_bytes: u64,
    pub last_compiled: i64,
}

/// RDP session viewer — architecture for web-based remote desktop.
/// The actual WebSocket relay connects to Symthaea's RDP server
/// and streams frames as Canvas2D draw commands.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RdpSession {
    pub session_id: String,
    pub target_host: String,
    pub width: u32,
    pub height: u32,
    pub connected: bool,
    pub fps: f32,
}
