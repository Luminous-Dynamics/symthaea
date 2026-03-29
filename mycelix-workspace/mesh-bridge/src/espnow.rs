// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ESP-NOW transport — peer-to-peer WiFi without an access point.
//!
//! ESP-NOW is a connectionless protocol by Espressif that allows direct
//! device-to-device communication over WiFi radio at up to 250 bytes per frame.
//! No router, no AP, no internet — just raw peer-to-peer at ~200m range.
//!
//! # Why ESP-NOW for Mesh Consciousness
//!
//! - **Zero infrastructure**: Works between any two ESP32 devices
//! - **Low latency**: ~1ms vs ~500ms for LoRa — ideal for Local tier
//! - **No pairing required**: Broadcast mode reaches all nearby ESP-NOW devices
//! - **Coexists with WiFi**: Can run alongside normal WiFi station mode
//! - **Perfect for Soma**: Phone ↔ sensor ↔ gateway at walking distance
//!
//! # Architecture
//!
//! On Linux (RPi), ESP-NOW isn't native. We communicate with an ESP32
//! running an ESP-NOW bridge firmware via serial SLIP framing:
//!
//! ```text
//! RPi (mesh-bridge) ←→ Serial SLIP ←→ ESP32 (ESP-NOW bridge) ←→ Air
//! ```
//!
//! The ESP32 bridge firmware:
//! 1. Receives SLIP frames from serial
//! 2. Broadcasts them via ESP-NOW to all peers
//! 3. Receives ESP-NOW frames from air
//! 4. Sends them back via SLIP to serial
//!
//! # SLIP Framing (RFC 1055)
//!
//! ```text
//! [END=0xC0] [payload with ESC sequences] [END=0xC0]
//! ESC (0xDB) + ESC_END (0xDC) → literal 0xC0
//! ESC (0xDB) + ESC_ESC (0xDD) → literal 0xDB
//! ```
//!
//! # Environment Variables
//!
//! - `ESPNOW_PORT`: Serial port to ESP32 bridge (default: `/dev/ttyACM0`)
//! - `ESPNOW_BAUD`: Baud rate (default: 921600 — fast for low latency)
//! - `ESPNOW_CHANNEL`: WiFi channel 1-13 (default: 1)

use anyhow::{Context, Result};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::transport::MeshTransport;

/// SLIP framing constants (RFC 1055).
const SLIP_END: u8 = 0xC0;
const SLIP_ESC: u8 = 0xDB;
const SLIP_ESC_END: u8 = 0xDC;
const SLIP_ESC_ESC: u8 = 0xDD;

/// Maximum ESP-NOW payload size.
const ESPNOW_MAX_PAYLOAD: usize = 250;

/// Default baud rate — 921600 for minimum latency.
const DEFAULT_BAUD: u32 = 921_600;

/// ESP-NOW transport configuration.
#[derive(Debug, Clone)]
pub struct EspNowConfig {
    /// Serial port to the ESP32 bridge device.
    pub port: String,
    /// Baud rate.
    pub baud: u32,
    /// WiFi channel (1-13).
    pub channel: u8,
}

impl Default for EspNowConfig {
    fn default() -> Self {
        Self {
            port: "/dev/ttyACM0".into(),
            baud: DEFAULT_BAUD,
            channel: 1,
        }
    }
}

impl EspNowConfig {
    /// Load from environment variables.
    pub fn from_env() -> Self {
        Self {
            port: std::env::var("ESPNOW_PORT").unwrap_or_else(|_| "/dev/ttyACM0".into()),
            baud: std::env::var("ESPNOW_BAUD")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_BAUD),
            channel: std::env::var("ESPNOW_CHANNEL")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1),
        }
    }
}

/// SLIP-encode a payload for serial transmission to the ESP32 bridge.
pub fn slip_encode(data: &[u8]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(data.len() + 4);
    frame.push(SLIP_END);
    for &b in data {
        match b {
            SLIP_END => {
                frame.push(SLIP_ESC);
                frame.push(SLIP_ESC_END);
            }
            SLIP_ESC => {
                frame.push(SLIP_ESC);
                frame.push(SLIP_ESC_ESC);
            }
            _ => frame.push(b),
        }
    }
    frame.push(SLIP_END);
    frame
}

/// SLIP-decode a frame received from the ESP32 bridge.
/// Returns None if the frame is malformed.
pub fn slip_decode(frame: &[u8]) -> Option<Vec<u8>> {
    let mut data = Vec::with_capacity(frame.len());
    let mut in_escape = false;

    for &b in frame {
        if in_escape {
            match b {
                SLIP_ESC_END => data.push(SLIP_END),
                SLIP_ESC_ESC => data.push(SLIP_ESC),
                _ => return None, // Invalid escape sequence
            }
            in_escape = false;
        } else {
            match b {
                SLIP_END => {} // Frame delimiter, skip
                SLIP_ESC => in_escape = true,
                _ => data.push(b),
            }
        }
    }

    if in_escape {
        return None; // Incomplete escape
    }

    if data.is_empty() {
        None
    } else {
        Some(data)
    }
}

/// ESP-NOW mesh transport via serial SLIP to an ESP32 bridge.
///
/// The ESP32 runs minimal firmware that relays SLIP frames to/from
/// ESP-NOW broadcast. This gives Linux hosts (RPi, laptop) access
/// to ESP-NOW's peer-to-peer WiFi without native support.
#[cfg(feature = "espnow")]
pub struct EspNowTransport {
    port: Arc<Mutex<serialport::TTYPort>>,
    config: EspNowConfig,
}

#[cfg(feature = "espnow")]
impl EspNowTransport {
    /// Open the serial connection to the ESP32 bridge.
    pub fn new(config: EspNowConfig) -> Result<Self> {
        let port = serialport::new(&config.port, config.baud)
            .timeout(std::time::Duration::from_secs(5))
            .open_native()
            .context(format!(
                "Failed to open ESP-NOW bridge at {} @ {} baud",
                config.port, config.baud
            ))?;

        tracing::info!(
            "ESP-NOW bridge connected: {} @ {} baud, channel {}",
            config.port,
            config.baud,
            config.channel
        );

        Ok(Self {
            port: Arc::new(Mutex::new(port)),
            config,
        })
    }
}

#[cfg(feature = "espnow")]
#[async_trait::async_trait]
impl MeshTransport for EspNowTransport {
    fn name(&self) -> &str {
        "espnow"
    }

    async fn send(&self, frame: &[u8]) -> Result<()> {
        if frame.len() > ESPNOW_MAX_PAYLOAD {
            anyhow::bail!(
                "ESP-NOW payload {} bytes exceeds {} limit",
                frame.len(),
                ESPNOW_MAX_PAYLOAD
            );
        }

        let slip_frame = slip_encode(frame);
        let mut port = self.port.lock().await;
        use std::io::Write;
        port.write_all(&slip_frame)?;
        port.flush()?;

        tracing::debug!("ESP-NOW TX: {} bytes (SLIP: {} bytes)", frame.len(), slip_frame.len());
        Ok(())
    }

    async fn recv(&self, timeout_ms: u64) -> Result<Option<Vec<u8>>> {
        let mut port = self.port.lock().await;
        port.set_timeout(std::time::Duration::from_millis(timeout_ms))?;

        // Read until we get a complete SLIP frame
        let mut buf = Vec::with_capacity(512);
        let mut byte = [0u8; 1];
        let mut started = false;

        let deadline = std::time::Instant::now()
            + std::time::Duration::from_millis(timeout_ms);

        loop {
            if std::time::Instant::now() >= deadline {
                return Ok(None);
            }

            use std::io::Read;
            match port.read(&mut byte) {
                Ok(1) => {
                    if byte[0] == SLIP_END {
                        if started && !buf.is_empty() {
                            // End of frame
                            buf.push(SLIP_END);
                            return Ok(slip_decode(&buf));
                        }
                        // Start of frame
                        started = true;
                        buf.clear();
                        buf.push(SLIP_END);
                    } else if started {
                        buf.push(byte[0]);
                    }
                }
                Ok(_) => continue,
                Err(e) if e.kind() == std::io::ErrorKind::TimedOut
                    || e.kind() == std::io::ErrorKind::WouldBlock =>
                {
                    return Ok(None);
                }
                Err(e) => return Err(e.into()),
            }
        }
    }

    fn clone_box(&self) -> Box<dyn MeshTransport> {
        Box::new(Self {
            port: self.port.clone(),
            config: self.config.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slip_encode_simple() {
        let data = b"hello";
        let encoded = slip_encode(data);
        assert_eq!(encoded[0], SLIP_END);
        assert_eq!(encoded[encoded.len() - 1], SLIP_END);
        assert_eq!(&encoded[1..6], b"hello");
    }

    #[test]
    fn test_slip_encode_with_escapes() {
        let data = [0xC0, 0xDB, 0x42]; // END, ESC, 'B'
        let encoded = slip_encode(&data);
        // 0xC0 → ESC + ESC_END, 0xDB → ESC + ESC_ESC, 0x42 → 0x42
        assert_eq!(
            encoded,
            vec![SLIP_END, SLIP_ESC, SLIP_ESC_END, SLIP_ESC, SLIP_ESC_ESC, 0x42, SLIP_END]
        );
    }

    #[test]
    fn test_slip_roundtrip() {
        let data = vec![0xC0, 0xDB, 0x00, 0xFF, 0xC0, 0xDB];
        let encoded = slip_encode(&data);
        let decoded = slip_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_slip_roundtrip_normal() {
        let data = b"consciousness vector delta".to_vec();
        let encoded = slip_encode(&data);
        let decoded = slip_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_slip_decode_empty_frame() {
        let frame = [SLIP_END, SLIP_END]; // Empty payload
        assert!(slip_decode(&frame).is_none());
    }

    #[test]
    fn test_slip_decode_incomplete_escape() {
        let frame = [SLIP_END, 0x42, SLIP_ESC, SLIP_END]; // ESC at end without ESC_END/ESC_ESC
        assert!(slip_decode(&frame).is_none());
    }

    #[test]
    fn test_slip_decode_invalid_escape() {
        let frame = [SLIP_END, SLIP_ESC, 0x42, SLIP_END]; // ESC + invalid byte
        assert!(slip_decode(&frame).is_none());
    }

    #[test]
    fn test_espnow_max_payload() {
        assert_eq!(ESPNOW_MAX_PAYLOAD, 250);
    }

    #[test]
    fn test_espnow_config_default() {
        let config = EspNowConfig::default();
        assert_eq!(config.port, "/dev/ttyACM0");
        assert_eq!(config.baud, 921_600);
        assert_eq!(config.channel, 1);
    }

    #[test]
    fn test_slip_encode_no_special_bytes() {
        let data = vec![0x01, 0x02, 0x03];
        let encoded = slip_encode(&data);
        // No escaping needed: END + data + END
        assert_eq!(encoded.len(), data.len() + 2);
    }

    #[test]
    fn test_slip_encode_all_special_bytes() {
        let data = vec![SLIP_END; 10];
        let encoded = slip_encode(&data);
        // Each END becomes ESC + ESC_END (2 bytes) + framing ENDs
        assert_eq!(encoded.len(), 10 * 2 + 2);
    }
}
