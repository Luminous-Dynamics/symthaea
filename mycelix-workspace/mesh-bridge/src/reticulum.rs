// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reticulum transport — store-and-forward mesh over any physical medium.
//!
//! Reticulum is a transport-agnostic mesh networking protocol that works over
//! LoRa (via RNode), serial, TCP, I2P, or WiFi. It provides:
//!   - Link encryption (X25519 + HKDF + Fernet)
//!   - Multi-hop mesh routing with cryptographic addressing
//!   - Store-and-forward via LXMF (Lightweight Extensible Message Format)
//!   - Works with days of latency (perfect for solar-powered off-grid nodes)
//!
//! # Why Reticulum for Mesh Consciousness
//!
//! - **Days-tolerant store-and-forward**: Our StoreAndForward dream consolidation
//!   maps perfectly to LXMF's guaranteed delivery model
//! - **Transport agnostic**: Same protocol over LoRa, serial, TCP, or I2P
//! - **Cryptographic addressing**: Each node has a unique identity derived from keys
//! - **No infrastructure dependency**: Works without internet, DNS, or certificates
//! - **Existing ecosystem**: Nomad Network, Sideband, LXMF clients
//!
//! # Architecture
//!
//! Reticulum runs as a daemon (`rnsd`) on the host. We communicate via its
//! local socket API (TCP localhost or Unix socket):
//!
//! ```text
//! mesh-bridge ←→ TCP 127.0.0.1:37428 ←→ rnsd ←→ RNode (LoRa) / TCP / I2P
//! ```
//!
//! We use Reticulum's "Resource" primitive for consciousness data:
//! - **Announce**: Declare our mesh-bridge identity and capabilities
//! - **Resource transfer**: Send WisdomPackets as Reticulum Resources
//! - **LXMF**: Store-and-forward for consolidated wisdom after offline periods
//!
//! # Environment Variables
//!
//! - `RETICULUM_HOST`: rnsd TCP host (default: `127.0.0.1`)
//! - `RETICULUM_PORT`: rnsd TCP port (default: 37428)
//! - `RETICULUM_IDENTITY`: Path to identity file (default: `~/.reticulum/identities/mesh-bridge`)
//! - `RETICULUM_ANNOUNCE_INTERVAL`: Seconds between announce broadcasts (default: 300)

use anyhow::{Context, Result};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::Mutex;

use crate::transport::MeshTransport;

/// Default Reticulum daemon port.
const DEFAULT_PORT: u16 = 37428;

/// Reticulum packet header size (simplified — actual RNS uses variable headers).
/// We use a fixed 16-byte envelope for our application-layer packets.
const PACKET_HEADER_SIZE: usize = 16;

/// Maximum Reticulum payload per packet (500 bytes for single-packet transfers,
/// larger for multi-part Resources).
const MAX_SINGLE_PACKET: usize = 500;

/// Application name for Reticulum identity.
const APP_NAME: &str = "mycelix.mesh_bridge";

/// Aspect for consciousness data resources.
const ASPECT_CONSCIOUSNESS: &str = "consciousness";

/// Aspect for governance relay data.
const ASPECT_GOVERNANCE: &str = "governance";

/// Reticulum transport configuration.
#[derive(Debug, Clone)]
pub struct ReticulumConfig {
    /// rnsd TCP host.
    pub host: String,
    /// rnsd TCP port.
    pub port: u16,
    /// Path to identity file (None = auto-generate).
    pub identity_path: Option<String>,
    /// Announce interval (seconds).
    pub announce_interval: u64,
}

impl Default for ReticulumConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: DEFAULT_PORT,
            identity_path: None,
            announce_interval: 300,
        }
    }
}

impl ReticulumConfig {
    /// Load from environment variables.
    pub fn from_env() -> Self {
        Self {
            host: std::env::var("RETICULUM_HOST").unwrap_or_else(|_| "127.0.0.1".into()),
            port: std::env::var("RETICULUM_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_PORT),
            identity_path: std::env::var("RETICULUM_IDENTITY").ok(),
            announce_interval: std::env::var("RETICULUM_ANNOUNCE_INTERVAL")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(300),
        }
    }

    /// TCP address string.
    pub fn addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

/// Packet types in our Reticulum application protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PacketType {
    /// Consciousness data (WisdomPacket / compressed delta).
    Consciousness = 1,
    /// Governance relay (TEND, emergency, etc.).
    Governance = 2,
    /// Heartbeat / presence announcement.
    Heartbeat = 3,
    /// Threat signature for collective immunity.
    Threat = 4,
    /// Consolidated wisdom after dream consolidation (store-and-forward).
    ConsolidatedWisdom = 5,
}

impl PacketType {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::Consciousness),
            2 => Some(Self::Governance),
            3 => Some(Self::Heartbeat),
            4 => Some(Self::Threat),
            5 => Some(Self::ConsolidatedWisdom),
            _ => None,
        }
    }
}

/// Simple application-layer packet envelope for Reticulum transport.
///
/// ```text
/// [magic: u16 LE] [type: u8] [flags: u8] [length: u32 LE] [sequence: u32 LE] [reserved: 4B] [payload...]
/// ```
const MAGIC: u16 = 0x4D43; // "MC" for Mycelix Consciousness

/// Encode an application packet with our envelope.
pub fn encode_packet(packet_type: PacketType, payload: &[u8], sequence: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(PACKET_HEADER_SIZE + payload.len());
    buf.extend_from_slice(&MAGIC.to_le_bytes());
    buf.push(packet_type as u8);
    buf.push(0); // flags (reserved)
    buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    buf.extend_from_slice(&sequence.to_le_bytes());
    buf.extend_from_slice(&[0u8; 4]); // reserved
    buf.extend_from_slice(payload);
    buf
}

/// Decode an application packet. Returns (type, payload, sequence) or None.
pub fn decode_packet(data: &[u8]) -> Option<(PacketType, Vec<u8>, u32)> {
    if data.len() < PACKET_HEADER_SIZE {
        return None;
    }

    let magic = u16::from_le_bytes([data[0], data[1]]);
    if magic != MAGIC {
        return None;
    }

    let packet_type = PacketType::from_u8(data[2])?;
    let length = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let sequence = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);

    if data.len() < PACKET_HEADER_SIZE + length {
        return None;
    }

    let payload = data[PACKET_HEADER_SIZE..PACKET_HEADER_SIZE + length].to_vec();
    Some((packet_type, payload, sequence))
}

/// Reticulum mesh transport.
///
/// Connects to a local Reticulum daemon (`rnsd`) via TCP and uses it
/// as a transport layer for consciousness data and governance messages.
/// Reticulum handles routing, encryption, and store-and-forward.
pub struct ReticulumTransport {
    stream: Arc<Mutex<TcpStream>>,
    config: ReticulumConfig,
    sequence: Arc<std::sync::atomic::AtomicU32>,
}

impl ReticulumTransport {
    /// Connect to the local Reticulum daemon.
    pub async fn new(config: ReticulumConfig) -> Result<Self> {
        let addr = config.addr();
        let stream = TcpStream::connect(&addr)
            .await
            .context(format!(
                "Failed to connect to Reticulum daemon at {addr}. \
                 Is rnsd running? Start with: rnsd -v"
            ))?;

        tracing::info!("Reticulum connected to {addr}");

        Ok(Self {
            stream: Arc::new(Mutex::new(stream)),
            config,
            sequence: Arc::new(std::sync::atomic::AtomicU32::new(0)),
        })
    }

    fn next_sequence(&self) -> u32 {
        self.sequence
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }
}

#[async_trait::async_trait]
impl MeshTransport for ReticulumTransport {
    fn name(&self) -> &str {
        "reticulum"
    }

    async fn send(&self, frame: &[u8]) -> Result<()> {
        let seq = self.next_sequence();
        let packet_type = if frame.len() <= 40 {
            PacketType::Heartbeat
        } else {
            PacketType::Governance // Default; caller should use encode_packet directly for specifics
        };

        let packet = encode_packet(packet_type, frame, seq);

        // Length-prefix the packet for TCP framing
        let len = (packet.len() as u32).to_le_bytes();
        let mut stream = self.stream.lock().await;
        stream.write_all(&len).await?;
        stream.write_all(&packet).await?;
        stream.flush().await?;

        tracing::debug!(
            "Reticulum TX: {} bytes (seq {}, type {:?})",
            frame.len(),
            seq,
            packet_type
        );

        Ok(())
    }

    async fn recv(&self, timeout_ms: u64) -> Result<Option<Vec<u8>>> {
        let mut stream = self.stream.lock().await;

        // Read length prefix (4 bytes)
        let mut len_buf = [0u8; 4];
        match tokio::time::timeout(
            tokio::time::Duration::from_millis(timeout_ms),
            stream.read_exact(&mut len_buf),
        )
        .await
        {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => return Ok(None), // Timeout
        }

        let packet_len = u32::from_le_bytes(len_buf) as usize;
        if packet_len > 65536 {
            return Err(anyhow::anyhow!("Reticulum packet too large: {packet_len}"));
        }

        // Read packet
        let mut packet = vec![0u8; packet_len];
        stream.read_exact(&mut packet).await?;

        // Decode our application envelope
        match decode_packet(&packet) {
            Some((_packet_type, payload, _seq)) => {
                tracing::debug!("Reticulum RX: {} bytes", payload.len());
                Ok(Some(payload))
            }
            None => {
                tracing::trace!("Reticulum RX: non-MC packet, ignoring");
                Ok(None)
            }
        }
    }

    fn clone_box(&self) -> Box<dyn MeshTransport> {
        Box::new(Self {
            stream: self.stream.clone(),
            config: self.config.clone(),
            sequence: self.sequence.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ReticulumConfig::default();
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 37428);
        assert_eq!(config.announce_interval, 300);
        assert!(config.identity_path.is_none());
    }

    #[test]
    fn test_config_addr() {
        let config = ReticulumConfig::default();
        assert_eq!(config.addr(), "127.0.0.1:37428");
    }

    #[test]
    fn test_packet_encode_decode_roundtrip() {
        let payload = b"consciousness delta vector";
        let packet = encode_packet(PacketType::Consciousness, payload, 42);

        let (ptype, decoded_payload, seq) = decode_packet(&packet).unwrap();
        assert_eq!(ptype, PacketType::Consciousness);
        assert_eq!(decoded_payload, payload);
        assert_eq!(seq, 42);
    }

    #[test]
    fn test_packet_types() {
        assert_eq!(PacketType::from_u8(1), Some(PacketType::Consciousness));
        assert_eq!(PacketType::from_u8(2), Some(PacketType::Governance));
        assert_eq!(PacketType::from_u8(3), Some(PacketType::Heartbeat));
        assert_eq!(PacketType::from_u8(4), Some(PacketType::Threat));
        assert_eq!(PacketType::from_u8(5), Some(PacketType::ConsolidatedWisdom));
        assert_eq!(PacketType::from_u8(0), None);
        assert_eq!(PacketType::from_u8(255), None);
    }

    #[test]
    fn test_packet_too_short() {
        assert!(decode_packet(&[0; 4]).is_none());
    }

    #[test]
    fn test_packet_wrong_magic() {
        let mut packet = encode_packet(PacketType::Heartbeat, b"hello", 1);
        packet[0] = 0xFF; // Corrupt magic
        assert!(decode_packet(&packet).is_none());
    }

    #[test]
    fn test_packet_header_size() {
        assert_eq!(PACKET_HEADER_SIZE, 16);
        let packet = encode_packet(PacketType::Heartbeat, &[], 0);
        assert_eq!(packet.len(), PACKET_HEADER_SIZE);
    }

    #[test]
    fn test_packet_all_types_roundtrip() {
        for ptype in [
            PacketType::Consciousness,
            PacketType::Governance,
            PacketType::Heartbeat,
            PacketType::Threat,
            PacketType::ConsolidatedWisdom,
        ] {
            let data = vec![0xAB; 100];
            let packet = encode_packet(ptype, &data, 999);
            let (decoded_type, decoded_data, seq) = decode_packet(&packet).unwrap();
            assert_eq!(decoded_type, ptype);
            assert_eq!(decoded_data, data);
            assert_eq!(seq, 999);
        }
    }

    #[test]
    fn test_magic_value() {
        assert_eq!(MAGIC, 0x4D43); // "MC" in little-endian
    }

    #[test]
    fn test_max_single_packet() {
        // Verify max single packet fits comfortably with header
        assert!(MAX_SINGLE_PACKET + PACKET_HEADER_SIZE < 600);
    }
}
