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

/// Controls all outbound sharing. Sovereign privacy by default.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharingConfig {
    pub holon_mode: HolonSyncMode,
    pub ble_mode: BleMeshMode,
    pub haptic_enabled: bool,
    pub telemetry_export: bool,
}

impl Default for SharingConfig {
    fn default() -> Self {
        Self {
            holon_mode: HolonSyncMode::Off,
            ble_mode: BleMeshMode::Off,
            haptic_enabled: true,
            telemetry_export: false,
        }
    }
}
