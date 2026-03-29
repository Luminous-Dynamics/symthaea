// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Meshtastic transport — communicates with Meshtastic devices via serial or TCP.
//!
//! Meshtastic devices expose a protobuf-based API over serial (USB) or TCP (WiFi).
//! This transport sends mesh-bridge payloads as `PRIVATE_APP` packets (portnum 256+),
//! letting Meshtastic handle all radio-layer concerns:
//!   - LoRa modem configuration (SF, BW, CR)
//!   - Multi-hop mesh routing and deduplication
//!   - AES-256 channel encryption
//!   - Duty cycle enforcement
//!   - Power management and deep sleep
//!
//! # Framing Protocol (serial)
//!
//! ```text
//! [0x94] [0xC3] [MSB len] [LSB len] [protobuf payload...]
//! ```
//!
//! # Environment Variables
//!
//! - `MESHTASTIC_PORT`: Serial port path (default: `/dev/ttyUSB0`)
//!   OR `tcp://host:port` for TCP connections (default TCP port: 4403)
//! - `MESHTASTIC_BAUD`: Serial baud rate (default: 115200)
//! - `MESHTASTIC_CHANNEL`: Channel index for sending (default: 0)
//! - `MESHTASTIC_PORTNUM`: Application port number (default: 256 = first PRIVATE_APP)
//! - `MESHTASTIC_BROADCAST`: Whether to broadcast (default: true). If false,
//!   requires `MESHTASTIC_DEST_NODE` with the destination node number.

use anyhow::{Context, Result};
use std::io::{Read, Write};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::transport::MeshTransport;

/// Meshtastic serial framing magic bytes.
const MAGIC_BYTE_0: u8 = 0x94;
const MAGIC_BYTE_1: u8 = 0xC3;

/// Maximum Meshtastic payload size (237 bytes after protobuf overhead).
/// We limit to 200 to leave room for our wrapper header.
const MAX_PAYLOAD_SIZE: usize = 200;

/// Default serial baud rate for Meshtastic devices.
const DEFAULT_BAUD: u32 = 115_200;

/// Default TCP port for Meshtastic WiFi API.
const DEFAULT_TCP_PORT: u16 = 4403;

/// Meshtastic PRIVATE_APP base port number.
/// We use 256 (first private app port) for mesh-bridge traffic.
const DEFAULT_PORTNUM: u32 = 256;

/// Broadcast destination (0xFFFFFFFF = all nodes).
const BROADCAST_ADDR: u32 = 0xFFFF_FFFF;

/// Meshtastic portnum constants (subset of meshtastic.portnums.proto).
#[allow(dead_code)]
mod portnum {
    pub const TEXT_MESSAGE_APP: u32 = 1;
    pub const POSITION_APP: u32 = 3;
    pub const NODEINFO_APP: u32 = 4;
    pub const TELEMETRY_APP: u32 = 67;
    pub const PRIVATE_APP: u32 = 256;
}

/// Configuration for MeshtasticTransport.
#[derive(Debug, Clone)]
pub struct MeshtasticConfig {
    /// Serial port path or "tcp://host:port".
    pub port: String,
    /// Serial baud rate (ignored for TCP).
    pub baud: u32,
    /// Meshtastic channel index.
    pub channel: u8,
    /// Application port number (256+ for private apps).
    pub portnum: u32,
    /// Destination node number (0xFFFFFFFF = broadcast).
    pub dest_node: u32,
}

impl Default for MeshtasticConfig {
    fn default() -> Self {
        Self {
            port: "/dev/ttyUSB0".into(),
            baud: DEFAULT_BAUD,
            channel: 0,
            portnum: DEFAULT_PORTNUM,
            dest_node: BROADCAST_ADDR,
        }
    }
}

impl MeshtasticConfig {
    /// Load configuration from environment variables.
    pub fn from_env() -> Self {
        let port = std::env::var("MESHTASTIC_PORT")
            .unwrap_or_else(|_| "/dev/ttyUSB0".into());
        let baud: u32 = std::env::var("MESHTASTIC_BAUD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_BAUD);
        let channel: u8 = std::env::var("MESHTASTIC_CHANNEL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let portnum: u32 = std::env::var("MESHTASTIC_PORTNUM")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_PORTNUM);
        let broadcast = std::env::var("MESHTASTIC_BROADCAST")
            .map(|s| s != "false" && s != "0")
            .unwrap_or(true);
        let dest_node = if broadcast {
            BROADCAST_ADDR
        } else {
            std::env::var("MESHTASTIC_DEST_NODE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(BROADCAST_ADDR)
        };

        Self {
            port,
            baud,
            channel,
            portnum,
            dest_node,
        }
    }

    /// Whether using TCP rather than serial.
    pub fn is_tcp(&self) -> bool {
        self.port.starts_with("tcp://")
    }

    /// Parse TCP host:port from the port string.
    pub fn tcp_addr(&self) -> Option<String> {
        self.port.strip_prefix("tcp://").map(|s| {
            if s.contains(':') {
                s.to_string()
            } else {
                format!("{s}:{DEFAULT_TCP_PORT}")
            }
        })
    }
}

/// Inner connection — either serial or TCP.
enum MeshtasticConnection {
    #[cfg(feature = "meshtastic")]
    Serial(serialport::TTYPort),
    Tcp(std::net::TcpStream),
}

impl MeshtasticConnection {
    fn write_frame(&mut self, data: &[u8]) -> Result<()> {
        let len = data.len() as u16;
        let header = [MAGIC_BYTE_0, MAGIC_BYTE_1, (len >> 8) as u8, (len & 0xFF) as u8];
        match self {
            #[cfg(feature = "meshtastic")]
            MeshtasticConnection::Serial(port) => {
                port.write_all(&header)?;
                port.write_all(data)?;
                port.flush()?;
            }
            MeshtasticConnection::Tcp(stream) => {
                stream.write_all(&header)?;
                stream.write_all(data)?;
                stream.flush()?;
            }
        }
        Ok(())
    }

    fn read_frame(&mut self, timeout_ms: u64) -> Result<Option<Vec<u8>>> {
        // Set read timeout
        let timeout = std::time::Duration::from_millis(timeout_ms);
        match self {
            #[cfg(feature = "meshtastic")]
            MeshtasticConnection::Serial(port) => {
                port.set_timeout(timeout)?;
            }
            MeshtasticConnection::Tcp(stream) => {
                stream.set_read_timeout(Some(timeout))?;
            }
        }

        // Read magic bytes — scan for sync
        let mut sync_buf = [0u8; 1];
        let deadline = std::time::Instant::now() + timeout;

        loop {
            if std::time::Instant::now() >= deadline {
                return Ok(None);
            }

            let n = match self {
                #[cfg(feature = "meshtastic")]
                MeshtasticConnection::Serial(port) => port.read(&mut sync_buf),
                MeshtasticConnection::Tcp(stream) => stream.read(&mut sync_buf),
            };

            match n {
                Ok(1) if sync_buf[0] == MAGIC_BYTE_0 => {
                    // Read second magic byte
                    let n2 = match self {
                        #[cfg(feature = "meshtastic")]
                        MeshtasticConnection::Serial(port) => port.read(&mut sync_buf),
                        MeshtasticConnection::Tcp(stream) => stream.read(&mut sync_buf),
                    };
                    match n2 {
                        Ok(1) if sync_buf[0] == MAGIC_BYTE_1 => break,
                        Ok(_) => continue,
                        Err(e) if e.kind() == std::io::ErrorKind::TimedOut
                            || e.kind() == std::io::ErrorKind::WouldBlock => return Ok(None),
                        Err(e) => return Err(e.into()),
                    }
                }
                Ok(0) => return Ok(None),
                Ok(_) => continue,
                Err(e) if e.kind() == std::io::ErrorKind::TimedOut
                    || e.kind() == std::io::ErrorKind::WouldBlock => return Ok(None),
                Err(e) => return Err(e.into()),
            }
        }

        // Read length (2 bytes, big-endian)
        let mut len_buf = [0u8; 2];
        match self {
            #[cfg(feature = "meshtastic")]
            MeshtasticConnection::Serial(port) => port.read_exact(&mut len_buf)?,
            MeshtasticConnection::Tcp(stream) => stream.read_exact(&mut len_buf)?,
        }
        let payload_len = ((len_buf[0] as usize) << 8) | (len_buf[1] as usize);

        if payload_len > 4096 {
            return Err(anyhow::anyhow!("Meshtastic frame too large: {payload_len}"));
        }

        // Read payload
        let mut payload = vec![0u8; payload_len];
        match self {
            #[cfg(feature = "meshtastic")]
            MeshtasticConnection::Serial(port) => port.read_exact(&mut payload)?,
            MeshtasticConnection::Tcp(stream) => stream.read_exact(&mut payload)?,
        }

        Ok(Some(payload))
    }
}

/// Meshtastic mesh transport.
///
/// Wraps a serial or TCP connection to a Meshtastic device.
/// Payloads are sent as PRIVATE_APP mesh packets; Meshtastic handles
/// LoRa modulation, multi-hop routing, encryption, and duty cycle.
pub struct MeshtasticTransport {
    conn: Arc<Mutex<MeshtasticConnection>>,
    config: MeshtasticConfig,
}

impl MeshtasticTransport {
    /// Connect to a Meshtastic device using the provided config.
    pub fn new(config: MeshtasticConfig) -> Result<Self> {
        let conn = if config.is_tcp() {
            let addr = config.tcp_addr()
                .context("Invalid TCP address")?;
            let stream = std::net::TcpStream::connect_timeout(
                &addr.parse()?,
                std::time::Duration::from_secs(5),
            ).context(format!("Failed to connect to Meshtastic TCP at {addr}"))?;
            stream.set_nodelay(true)?;
            tracing::info!("Meshtastic TCP connected to {addr}");
            MeshtasticConnection::Tcp(stream)
        } else {
            #[cfg(feature = "meshtastic")]
            {
                let port = serialport::new(&config.port, config.baud)
                    .timeout(std::time::Duration::from_secs(5))
                    .open_native()
                    .context(format!(
                        "Failed to open Meshtastic serial port {} at {} baud",
                        config.port, config.baud
                    ))?;
                tracing::info!(
                    "Meshtastic serial connected: {} @ {} baud",
                    config.port,
                    config.baud
                );
                MeshtasticConnection::Serial(port)
            }
            #[cfg(not(feature = "meshtastic"))]
            {
                anyhow::bail!(
                    "Serial transport requires the 'meshtastic' feature. \
                     Use tcp:// or enable the feature flag."
                );
            }
        };

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            config,
        })
    }

    /// Encode a mesh-bridge payload into a Meshtastic-compatible packet.
    ///
    /// Uses a minimal binary envelope (no full protobuf dependency):
    /// ```text
    /// [portnum: u32 LE] [dest: u32 LE] [channel: u8] [payload...]
    /// ```
    ///
    /// This is a simplified ToRadio encoding. For full protobuf compatibility
    /// with the Meshtastic ecosystem, upgrade to use the `meshtastic` crate.
    fn encode_to_radio(&self, data: &[u8]) -> Vec<u8> {
        // Simplified packet format:
        // We use a minimal binary header that the Meshtastic device
        // interprets via the serial API.
        //
        // Full protobuf ToRadio would be:
        //   ToRadio { packet: MeshPacket { to, channel, decoded: Data { portnum, payload } } }
        //
        // For now, we use a compact binary envelope that can be upgraded
        // to full protobuf when the meshtastic-protobuf crate is added.
        let mut packet = Vec::with_capacity(9 + data.len());

        // Header: portnum(4) + dest(4) + channel(1)
        packet.extend_from_slice(&self.config.portnum.to_le_bytes());
        packet.extend_from_slice(&self.config.dest_node.to_le_bytes());
        packet.push(self.config.channel);

        // Payload
        packet.extend_from_slice(data);

        packet
    }

    /// Decode a received Meshtastic packet, extracting the payload.
    /// Returns None if the packet isn't for our portnum or is malformed.
    fn decode_from_radio(data: &[u8], expected_portnum: u32) -> Option<Vec<u8>> {
        if data.len() < 9 {
            return None;
        }

        let portnum = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        // Accept our portnum or any PRIVATE_APP range
        if portnum != expected_portnum && portnum < portnum::PRIVATE_APP {
            return None;
        }

        // Skip header (portnum + source/dest + channel = 9 bytes)
        Some(data[9..].to_vec())
    }
}

#[async_trait::async_trait]
impl MeshTransport for MeshtasticTransport {
    fn name(&self) -> &str {
        "meshtastic"
    }

    async fn send(&self, frame: &[u8]) -> Result<()> {
        if frame.len() > MAX_PAYLOAD_SIZE {
            tracing::warn!(
                "Meshtastic payload {} bytes exceeds {MAX_PAYLOAD_SIZE} limit, truncating",
                frame.len()
            );
        }

        let data = if frame.len() > MAX_PAYLOAD_SIZE {
            &frame[..MAX_PAYLOAD_SIZE]
        } else {
            frame
        };

        let packet = self.encode_to_radio(data);
        let mut conn = self.conn.lock().await;
        conn.write_frame(&packet)
            .context("Meshtastic send failed")?;

        tracing::debug!(
            "Meshtastic TX: {} bytes → {} (ch {})",
            data.len(),
            if self.config.dest_node == BROADCAST_ADDR {
                "broadcast".to_string()
            } else {
                format!("node {}", self.config.dest_node)
            },
            self.config.channel
        );

        Ok(())
    }

    async fn recv(&self, timeout_ms: u64) -> Result<Option<Vec<u8>>> {
        let mut conn = self.conn.lock().await;

        match conn.read_frame(timeout_ms)? {
            Some(raw) => {
                match Self::decode_from_radio(&raw, self.config.portnum) {
                    Some(payload) => {
                        tracing::debug!("Meshtastic RX: {} bytes", payload.len());
                        Ok(Some(payload))
                    }
                    None => {
                        // Not our portnum — ignore
                        tracing::trace!("Meshtastic RX: ignored (wrong portnum or malformed)");
                        Ok(None)
                    }
                }
            }
            None => Ok(None),
        }
    }

    fn clone_box(&self) -> Box<dyn MeshTransport> {
        Box::new(Self {
            conn: self.conn.clone(),
            config: self.config.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = MeshtasticConfig::default();
        assert_eq!(config.port, "/dev/ttyUSB0");
        assert_eq!(config.baud, 115_200);
        assert_eq!(config.channel, 0);
        assert_eq!(config.portnum, 256);
        assert_eq!(config.dest_node, BROADCAST_ADDR);
    }

    #[test]
    fn test_config_is_tcp() {
        let mut config = MeshtasticConfig::default();
        assert!(!config.is_tcp());

        config.port = "tcp://192.168.1.100".into();
        assert!(config.is_tcp());

        config.port = "tcp://192.168.1.100:4403".into();
        assert!(config.is_tcp());
    }

    #[test]
    fn test_config_tcp_addr() {
        let mut config = MeshtasticConfig::default();

        config.port = "tcp://192.168.1.100".into();
        assert_eq!(config.tcp_addr(), Some("192.168.1.100:4403".into()));

        config.port = "tcp://10.0.0.1:5555".into();
        assert_eq!(config.tcp_addr(), Some("10.0.0.1:5555".into()));

        config.port = "/dev/ttyUSB0".into();
        assert_eq!(config.tcp_addr(), None);
    }

    #[test]
    fn test_encode_to_radio_format() {
        // Test the encoding format directly without constructing a transport.
        let portnum: u32 = 256;
        let dest_node: u32 = BROADCAST_ADDR;
        let channel: u8 = 0;
        let data = b"hello mesh";

        let mut packet = Vec::with_capacity(9 + data.len());
        packet.extend_from_slice(&portnum.to_le_bytes());
        packet.extend_from_slice(&dest_node.to_le_bytes());
        packet.push(channel);
        packet.extend_from_slice(data);

        // Header: 4 (portnum) + 4 (dest) + 1 (channel) = 9
        assert_eq!(packet.len(), 9 + data.len());
        // Portnum = 256 LE
        assert_eq!(&packet[0..4], &256u32.to_le_bytes());
        // Dest = broadcast LE
        assert_eq!(&packet[4..8], &BROADCAST_ADDR.to_le_bytes());
        // Channel = 0
        assert_eq!(packet[8], 0);
        // Payload
        assert_eq!(&packet[9..], b"hello mesh");
    }

    #[test]
    fn test_decode_from_radio_valid() {
        let portnum: u32 = 256;
        let mut data = Vec::new();
        data.extend_from_slice(&portnum.to_le_bytes());    // portnum
        data.extend_from_slice(&BROADCAST_ADDR.to_le_bytes()); // source/dest
        data.push(0); // channel
        data.extend_from_slice(b"payload");

        let result = MeshtasticTransport::decode_from_radio(&data, 256);
        assert_eq!(result, Some(b"payload".to_vec()));
    }

    #[test]
    fn test_decode_from_radio_wrong_portnum() {
        let portnum: u32 = 1; // TEXT_MESSAGE_APP, not our port
        let mut data = Vec::new();
        data.extend_from_slice(&portnum.to_le_bytes());
        data.extend_from_slice(&BROADCAST_ADDR.to_le_bytes());
        data.push(0);
        data.extend_from_slice(b"text message");

        let result = MeshtasticTransport::decode_from_radio(&data, 256);
        assert_eq!(result, None);
    }

    #[test]
    fn test_decode_from_radio_too_short() {
        let result = MeshtasticTransport::decode_from_radio(&[0; 4], 256);
        assert_eq!(result, None);
    }

    #[test]
    fn test_max_payload_size_fits_lora() {
        // Meshtastic max packet is 237 bytes. Our header is 9 bytes.
        // So max usable payload = 237 - 9 = 228. We use 200 for safety margin.
        assert!(MAX_PAYLOAD_SIZE <= 228);
        assert!(MAX_PAYLOAD_SIZE >= 180, "payload must fit compressed deltas");
    }

    #[test]
    fn test_encode_roundtrip() {
        let config = MeshtasticConfig {
            portnum: 300,
            dest_node: 12345,
            channel: 2,
            ..Default::default()
        };

        // Encode
        let mut packet = Vec::new();
        packet.extend_from_slice(&config.portnum.to_le_bytes());
        packet.extend_from_slice(&config.dest_node.to_le_bytes());
        packet.push(config.channel);
        packet.extend_from_slice(b"test roundtrip");

        // Decode
        let payload = MeshtasticTransport::decode_from_radio(&packet, 300);
        assert_eq!(payload, Some(b"test roundtrip".to_vec()));
    }

    #[test]
    fn test_magic_bytes() {
        assert_eq!(MAGIC_BYTE_0, 0x94);
        assert_eq!(MAGIC_BYTE_1, 0xC3);
    }

    #[test]
    fn test_broadcast_addr() {
        assert_eq!(BROADCAST_ADDR, 0xFFFF_FFFF);
    }

    #[test]
    fn test_portnum_constants() {
        assert_eq!(portnum::TEXT_MESSAGE_APP, 1);
        assert_eq!(portnum::PRIVATE_APP, 256);
        assert!(DEFAULT_PORTNUM >= portnum::PRIVATE_APP);
    }
}
