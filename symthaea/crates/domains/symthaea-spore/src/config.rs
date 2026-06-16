// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Spore configuration.

use serde::{Deserialize, Serialize};

/// Configuration for the Spore consciousness kernel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SporeConfig {
    /// HDC dimension (default: 16,384 = full fidelity).
    /// Reduce for constrained environments (4,096 minimum recommended).
    pub hdc_dim: usize,

    /// Number of CfC neurons per layer.
    pub neurons_per_layer: usize,

    /// Number of network layers.
    pub network_layers: usize,

    /// Compute Phi every N cycles (1 = every cycle, 5 = every 5th).
    /// Higher values reduce CPU cost at the expense of consciousness resolution.
    pub phi_every_n_cycles: usize,

    /// Substrate type name (e.g. "SiliconDigital", "BiologicalNeurons").
    pub substrate: String,

    /// Target cycle rate in Hz. Engine will skip non-essential computations
    /// if falling behind this target.
    pub target_hz: f32,

    /// Number of warmup cycles during which a consciousness floor is applied.
    /// During warmup, the raw consciousness score is blended with a floor of 0.15
    /// so the Spore doesn't appear "dead" during initial interactions.
    /// After warmup, the floor fades linearly to 0.0 over the same number of cycles.
    /// Default: 10.
    pub consciousness_warmup_cycles: u64,

    /// Semantic memory ring buffer capacity (default: 500).
    /// Higher values retain more historical patterns at the cost of RAM.
    #[serde(default = "default_semantic_capacity")]
    pub semantic_memory_capacity: usize,

    /// Episodic memory priority queue capacity (default: 100).
    /// Higher values retain more high-Phi experiences for dream replay.
    #[serde(default = "default_episodic_capacity")]
    pub episodic_memory_capacity: usize,
}

fn default_semantic_capacity() -> usize {
    500
}
fn default_episodic_capacity() -> usize {
    100
}

impl Default for SporeConfig {
    fn default() -> Self {
        Self {
            hdc_dim: 16_384,
            neurons_per_layer: 64,
            network_layers: 3,
            phi_every_n_cycles: 1,
            substrate: "SiliconDigital".into(),
            target_hz: 50.0,
            consciousness_warmup_cycles: 10,
            semantic_memory_capacity: 500,
            episodic_memory_capacity: 100,
        }
    }
}

/// How the Spore shares consciousness state with the desktop Holon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HolonSyncMode {
    /// No bridge activity.
    Off,
    /// Heartbeats + consciousness vector only (no task delegation).
    PresenceOnly,
    /// Full sync: task delegation, knowledge shares, resuscitation.
    FullSync,
}

impl Default for HolonSyncMode {
    fn default() -> Self {
        Self::Off
    }
}

/// BLE mesh sharing mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BleMeshMode {
    /// No mesh activity.
    Off,
    /// Receive peer CVs, don't advertise own.
    PassiveDiscover,
    /// Receive + advertise own CV.
    ActiveShare,
}

impl Default for BleMeshMode {
    fn default() -> Self {
        Self::Off
    }
}

/// Pairing mode for Ed25519 device trust establishment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PairingMode {
    /// No pairing activity.
    Off,
    /// Accepting pairing requests from nearby devices.
    Discoverable,
    /// Already paired — only re-authenticate known devices.
    Paired,
}

impl Default for PairingMode {
    fn default() -> Self {
        Self::Off
    }
}

/// Controls all outbound sharing. Sovereign privacy by default.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharingConfig {
    pub holon_mode: HolonSyncMode,
    pub ble_mode: BleMeshMode,
    pub haptic_enabled: bool,
    pub telemetry_export: bool,
    #[serde(default)]
    pub pairing_mode: PairingMode,
}

impl Default for SharingConfig {
    fn default() -> Self {
        Self {
            holon_mode: HolonSyncMode::Off,
            ble_mode: BleMeshMode::Off,
            haptic_enabled: true,
            telemetry_export: false,
            pairing_mode: PairingMode::Off,
        }
    }
}
