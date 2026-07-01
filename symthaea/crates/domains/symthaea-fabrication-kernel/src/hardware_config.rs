// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Configuration layer for connecting real hardware to the fabrication kernel.
//!
//! Provides [`HardwareConfig`] as the single entry point for selecting a
//! printer backend (Mock, OctoPrint, Moonraker, or Custom) and optionally
//! enabling Cincinnati in-situ monitoring.  Configuration can be loaded from
//! environment variables, JSON, or constructed programmatically.

use crate::cincinnati_live::CincinnatiMonitorConfig;
use crate::printer_control::{MockPrinter, MoonrakerClient, OctoPrintClient, PrinterApi};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Printer backend
// ---------------------------------------------------------------------------

/// Which printer backend to instantiate.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PrinterBackend {
    /// Fully in-memory mock (no hardware required).
    #[default]
    Mock,
    /// OctoPrint REST API.
    OctoPrint { url: String },
    /// Moonraker (Klipper) JSON-RPC API.
    Moonraker { url: String },
    /// Generic HTTP printer endpoint.
    Custom { url: String },
}

// ---------------------------------------------------------------------------
// Printer config
// ---------------------------------------------------------------------------

/// Printer-specific configuration.
///
/// `Debug` is implemented manually (not derived) to redact `api_key` --
/// `Serialize`/`Deserialize` remain derived and unredacted, since the real
/// key must round-trip through config save/reload to actually work. The
/// risk this closes is `api_key` leaking into an unintended place like a
/// `tracing::debug!("{:?}", config)` diagnostic log, not the config file
/// itself (which necessarily needs the real key).
#[derive(Clone, Serialize, Deserialize)]
pub struct PrinterConfig {
    /// Which backend to use.
    pub backend: PrinterBackend,
    /// Optional API key (used by OctoPrint and Custom backends).
    pub api_key: Option<String>,
}

impl std::fmt::Debug for PrinterConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrinterConfig")
            .field("backend", &self.backend)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

impl Default for PrinterConfig {
    fn default() -> Self {
        Self {
            backend: PrinterBackend::Mock,
            api_key: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Cincinnati config
// ---------------------------------------------------------------------------

/// Cincinnati in-situ monitoring configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CincinnatiConfig {
    /// Whether Cincinnati monitoring is enabled.
    pub enabled: bool,
    /// Polling interval in milliseconds.
    #[serde(default = "default_poll_interval")]
    pub poll_interval_ms: u64,
    /// Z-score threshold for anomaly detection.
    #[serde(default = "default_anomaly_threshold")]
    pub anomaly_threshold: f32,
    /// Named sensor channels to monitor.
    #[serde(default)]
    pub sensor_channels: Vec<String>,
}

fn default_poll_interval() -> u64 {
    100
}

fn default_anomaly_threshold() -> f32 {
    2.5
}

impl Default for CincinnatiConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            poll_interval_ms: default_poll_interval(),
            anomaly_threshold: default_anomaly_threshold(),
            sensor_channels: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level hardware config
// ---------------------------------------------------------------------------

/// Unified configuration for all fabrication hardware connections.
///
/// # Examples
///
/// ```
/// use symthaea_fabrication_kernel::hardware_config::HardwareConfig;
///
/// // Default: simulation mode with MockPrinter, no Cincinnati.
/// let config = HardwareConfig::default();
/// assert!(config.simulation_mode);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Printer backend configuration.
    pub printer: PrinterConfig,
    /// Cincinnati in-situ monitoring configuration.
    pub cincinnati: CincinnatiConfig,
    /// When `true`, all hardware interactions are simulated.
    #[serde(default = "default_simulation_mode")]
    pub simulation_mode: bool,
}

fn default_simulation_mode() -> bool {
    true
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            printer: PrinterConfig::default(),
            cincinnati: CincinnatiConfig::default(),
            simulation_mode: true,
        }
    }
}

impl HardwareConfig {
    /// Load configuration from environment variables.
    ///
    /// | Variable | Default | Description |
    /// |----------|---------|-------------|
    /// | `FAB_PRINTER_BACKEND` | `"mock"` | `"mock"`, `"octoprint"`, `"moonraker"`, `"custom"` |
    /// | `FAB_PRINTER_URL` | `""` | Base URL for the printer backend |
    /// | `FAB_PRINTER_API_KEY` | *none* | API key for OctoPrint / Custom |
    /// | `FAB_CINCINNATI_ENABLED` | `"false"` | Enable Cincinnati monitoring |
    /// | `FAB_CINCINNATI_POLL_MS` | `"100"` | Polling interval in ms |
    /// | `FAB_CINCINNATI_THRESHOLD` | `"2.5"` | Anomaly z-score threshold |
    /// | `FAB_CINCINNATI_CHANNELS` | `""` | Comma-separated sensor channels |
    /// | `FAB_SIMULATION_MODE` | `"true"` | Enable simulation mode |
    pub fn from_env() -> Self {
        let backend_str = std::env::var("FAB_PRINTER_BACKEND")
            .unwrap_or_else(|_| "mock".to_string())
            .to_lowercase();
        let printer_url = std::env::var("FAB_PRINTER_URL").unwrap_or_default();
        let api_key = std::env::var("FAB_PRINTER_API_KEY").ok();

        let backend = match backend_str.as_str() {
            "octoprint" => PrinterBackend::OctoPrint { url: printer_url },
            "moonraker" => PrinterBackend::Moonraker { url: printer_url },
            "custom" => PrinterBackend::Custom { url: printer_url },
            _ => PrinterBackend::Mock,
        };

        let cincinnati_enabled = std::env::var("FAB_CINCINNATI_ENABLED")
            .map(|v| v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let poll_interval_ms = std::env::var("FAB_CINCINNATI_POLL_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default_poll_interval());

        let anomaly_threshold = std::env::var("FAB_CINCINNATI_THRESHOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default_anomaly_threshold());

        let sensor_channels: Vec<String> = std::env::var("FAB_CINCINNATI_CHANNELS")
            .ok()
            .map(|v| {
                v.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default();

        let simulation_mode = std::env::var("FAB_SIMULATION_MODE")
            .map(|v| !v.eq_ignore_ascii_case("false"))
            .unwrap_or(true);

        Self {
            printer: PrinterConfig { backend, api_key },
            cincinnati: CincinnatiConfig {
                enabled: cincinnati_enabled,
                poll_interval_ms,
                anomaly_threshold,
                sensor_channels,
            },
            simulation_mode,
        }
    }

    /// Factory method: create a boxed [`PrinterApi`] from the current config.
    ///
    /// In simulation mode, always returns a [`MockPrinter`] regardless of the
    /// configured backend.
    pub fn create_printer(&self) -> Box<dyn PrinterApi> {
        if self.simulation_mode {
            return Box::new(MockPrinter::new());
        }
        match &self.printer.backend {
            PrinterBackend::Mock => Box::new(MockPrinter::new()),
            PrinterBackend::OctoPrint { url } => {
                let key = self.printer.api_key.as_deref().unwrap_or("");
                Box::new(OctoPrintClient::new(url.as_str(), key))
            }
            PrinterBackend::Moonraker { url } => Box::new(MoonrakerClient::new(url.as_str())),
            PrinterBackend::Custom { url } => {
                // Custom backends fall back to OctoPrint protocol for now.
                let key = self.printer.api_key.as_deref().unwrap_or("");
                Box::new(OctoPrintClient::new(url.as_str(), key))
            }
        }
    }

    /// Create a [`CincinnatiMonitorConfig`] if Cincinnati monitoring is enabled.
    ///
    /// Returns `None` when `cincinnati.enabled` is `false`.
    pub fn create_monitor_config(&self) -> Option<CincinnatiMonitorConfig> {
        if !self.cincinnati.enabled {
            return None;
        }
        Some(CincinnatiMonitorConfig {
            poll_interval_ms: self.cincinnati.poll_interval_ms,
            anomaly_threshold: self.cincinnati.anomaly_threshold,
            sensor_channels: self.cincinnati.sensor_channels.clone(),
        })
    }

    /// Deserialize from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serialize to a pretty-printed JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("HardwareConfig should always serialize")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Env-var tests mutate process-wide state; serialize them.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn default_is_simulation() {
        let config = HardwareConfig::default();
        assert!(config.simulation_mode);
        assert_eq!(config.printer.backend, PrinterBackend::Mock);
        assert!(config.printer.api_key.is_none());
        assert!(!config.cincinnati.enabled);
        assert_eq!(config.cincinnati.poll_interval_ms, 100);
        assert!((config.cincinnati.anomaly_threshold - 2.5).abs() < f32::EPSILON);
        assert!(config.cincinnati.sensor_channels.is_empty());
    }

    #[test]
    fn printer_config_debug_redacts_api_key() {
        let config = PrinterConfig {
            backend: PrinterBackend::OctoPrint {
                url: "http://printer.local".into(),
            },
            api_key: Some("super-secret-key-do-not-log".into()),
        };
        let debug_output = format!("{config:?}");
        assert!(
            !debug_output.contains("super-secret-key-do-not-log"),
            "Debug output must not contain the raw API key, got: {debug_output}"
        );
        assert!(debug_output.contains("redacted"));
    }

    #[test]
    fn printer_config_debug_shows_none_when_absent() {
        let config = PrinterConfig {
            backend: PrinterBackend::Mock,
            api_key: None,
        };
        let debug_output = format!("{config:?}");
        assert!(debug_output.contains("None"));
    }

    #[test]
    fn printer_config_serialize_still_includes_real_key() {
        // Serialize (config save/reload) must NOT be redacted -- the real
        // key has to round-trip for the config to actually work.
        let config = PrinterConfig {
            backend: PrinterBackend::Custom {
                url: "http://printer.local".into(),
            },
            api_key: Some("super-secret-key-do-not-log".into()),
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("super-secret-key-do-not-log"));
        let round_tripped: PrinterConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(
            round_tripped.api_key.as_deref(),
            Some("super-secret-key-do-not-log")
        );
    }

    #[test]
    fn from_env_mock() {
        let _guard = ENV_LOCK.lock().unwrap();

        // Clear all env vars to ensure mock defaults.
        unsafe {
            std::env::remove_var("FAB_PRINTER_BACKEND");
            std::env::remove_var("FAB_PRINTER_URL");
            std::env::remove_var("FAB_PRINTER_API_KEY");
            std::env::remove_var("FAB_CINCINNATI_ENABLED");
            std::env::remove_var("FAB_CINCINNATI_POLL_MS");
            std::env::remove_var("FAB_CINCINNATI_THRESHOLD");
            std::env::remove_var("FAB_CINCINNATI_CHANNELS");
            std::env::remove_var("FAB_SIMULATION_MODE");
        }

        let config = HardwareConfig::from_env();
        assert_eq!(config.printer.backend, PrinterBackend::Mock);
        assert!(config.simulation_mode);
        assert!(!config.cincinnati.enabled);
    }

    #[test]
    fn create_printer_mock() {
        let config = HardwareConfig::default();
        let printer = config.create_printer();
        assert_eq!(printer.name(), "MockPrinter");
    }

    #[test]
    fn create_printer_octoprint_when_not_simulating() {
        let config = HardwareConfig {
            printer: PrinterConfig {
                backend: PrinterBackend::OctoPrint {
                    url: "http://octoprint.local".into(),
                },
                api_key: Some("secret".into()),
            },
            simulation_mode: false,
            ..Default::default()
        };
        let printer = config.create_printer();
        assert_eq!(printer.name(), "OctoPrint");
    }

    #[test]
    fn create_monitor_disabled() {
        let config = HardwareConfig::default();
        assert!(config.create_monitor_config().is_none());
    }

    #[test]
    fn create_monitor_enabled() {
        let config = HardwareConfig {
            cincinnati: CincinnatiConfig {
                enabled: true,
                poll_interval_ms: 200,
                anomaly_threshold: 3.5,
                sensor_channels: vec!["temp".into(), "pressure".into()],
            },
            ..Default::default()
        };
        let monitor = config.create_monitor_config().unwrap();
        assert_eq!(monitor.poll_interval_ms, 200);
        assert!((monitor.anomaly_threshold - 3.5).abs() < f32::EPSILON);
        assert_eq!(monitor.sensor_channels.len(), 2);
    }

    #[test]
    fn json_roundtrip() {
        let original = HardwareConfig {
            printer: PrinterConfig {
                backend: PrinterBackend::Moonraker {
                    url: "http://klipper.local:7125".into(),
                },
                api_key: None,
            },
            cincinnati: CincinnatiConfig {
                enabled: true,
                poll_interval_ms: 50,
                anomaly_threshold: 4.0,
                sensor_channels: vec!["temperature_nozzle".into()],
            },
            simulation_mode: false,
        };

        let json = original.to_json();
        let restored = HardwareConfig::from_json(&json).unwrap();

        assert_eq!(restored.printer.backend, original.printer.backend);
        assert_eq!(restored.simulation_mode, original.simulation_mode);
        assert_eq!(restored.cincinnati.enabled, original.cincinnati.enabled);
        assert_eq!(
            restored.cincinnati.poll_interval_ms,
            original.cincinnati.poll_interval_ms
        );
        assert!(
            (restored.cincinnati.anomaly_threshold - original.cincinnati.anomaly_threshold).abs()
                < f32::EPSILON
        );
        assert_eq!(
            restored.cincinnati.sensor_channels,
            original.cincinnati.sensor_channels
        );
    }

    #[test]
    fn from_env_with_channels() {
        let _guard = ENV_LOCK.lock().unwrap();

        unsafe {
            std::env::set_var("FAB_PRINTER_BACKEND", "moonraker");
            std::env::set_var("FAB_PRINTER_URL", "http://klipper:7125");
            std::env::remove_var("FAB_PRINTER_API_KEY");
            std::env::set_var("FAB_CINCINNATI_ENABLED", "true");
            std::env::set_var("FAB_CINCINNATI_POLL_MS", "250");
            std::env::set_var("FAB_CINCINNATI_THRESHOLD", "1.8");
            std::env::set_var(
                "FAB_CINCINNATI_CHANNELS",
                "temperature_nozzle, vibration_x, extrusion_pressure",
            );
            std::env::set_var("FAB_SIMULATION_MODE", "false");
        }

        let config = HardwareConfig::from_env();

        assert_eq!(
            config.printer.backend,
            PrinterBackend::Moonraker {
                url: "http://klipper:7125".into()
            }
        );
        assert!(config.printer.api_key.is_none());
        assert!(!config.simulation_mode);
        assert!(config.cincinnati.enabled);
        assert_eq!(config.cincinnati.poll_interval_ms, 250);
        assert!((config.cincinnati.anomaly_threshold - 1.8).abs() < f32::EPSILON);
        assert_eq!(config.cincinnati.sensor_channels.len(), 3);
        assert_eq!(config.cincinnati.sensor_channels[0], "temperature_nozzle");
        assert_eq!(config.cincinnati.sensor_channels[1], "vibration_x");
        assert_eq!(config.cincinnati.sensor_channels[2], "extrusion_pressure");

        // Cleanup.
        unsafe {
            std::env::remove_var("FAB_PRINTER_BACKEND");
            std::env::remove_var("FAB_PRINTER_URL");
            std::env::remove_var("FAB_CINCINNATI_ENABLED");
            std::env::remove_var("FAB_CINCINNATI_POLL_MS");
            std::env::remove_var("FAB_CINCINNATI_THRESHOLD");
            std::env::remove_var("FAB_CINCINNATI_CHANNELS");
            std::env::remove_var("FAB_SIMULATION_MODE");
        }
    }

    #[test]
    fn simulation_mode_forces_mock() {
        let config = HardwareConfig {
            printer: PrinterConfig {
                backend: PrinterBackend::Moonraker {
                    url: "http://klipper.local".into(),
                },
                api_key: None,
            },
            simulation_mode: true,
            ..Default::default()
        };
        // Even with Moonraker configured, simulation mode returns MockPrinter.
        let printer = config.create_printer();
        assert_eq!(printer.name(), "MockPrinter");
    }
}
