// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CCSDS Space Packet transport — spacecraft communication framing.
//!
//! Implements the Consultative Committee for Space Data Systems (CCSDS)
//! Space Packet Protocol for satellite and interplanetary communication.
//!
//! # Architecture
//!
//! ```text
//! mesh-bridge ←→ TCP/serial ←→ Ground station modem ←→ S-band/UHF radio ←→ Space
//! ```
//!
//! The transport connects to a ground station modem (or software-defined radio)
//! via TCP or serial, framing payloads as CCSDS space packets.
//!
//! # Environment Variables
//!
//! - `CCSDS_HOST`: Ground station modem host (default: `127.0.0.1`)
//! - `CCSDS_PORT`: Ground station modem port (default: 10025)
//! - `CCSDS_APID`: Application Process ID (default: 100 = consciousness telemetry)
//! - `CCSDS_DEST`: Destination label for latency calc (default: "LEO")

use anyhow::{Context, Result};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::Mutex;

use crate::transport::MeshTransport;

/// CCSDS primary header size (6 bytes).
const HEADER_SIZE: usize = 6;

/// Default ground station port.
const DEFAULT_PORT: u16 = 10025;

/// CCSDS transport configuration.
#[derive(Debug, Clone)]
pub struct CcsdsConfig {
    pub host: String,
    pub port: u16,
    pub apid: u16,
    pub destination: String,
}

impl Default for CcsdsConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: DEFAULT_PORT,
            apid: 100,
            destination: "LEO".into(),
        }
    }
}

impl CcsdsConfig {
    pub fn from_env() -> Self {
        Self {
            host: std::env::var("CCSDS_HOST").unwrap_or_else(|_| "127.0.0.1".into()),
            port: std::env::var("CCSDS_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_PORT),
            apid: std::env::var("CCSDS_APID")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(100),
            destination: std::env::var("CCSDS_DEST").unwrap_or_else(|_| "LEO".into()),
        }
    }
}

/// Encode a CCSDS space packet (simplified primary header + payload).
pub fn encode_ccsds(apid: u16, sequence: u16, data: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(HEADER_SIZE + data.len());

    // Word 1: Version(3)=000 + Type(1)=0(TM) + SecHdr(1)=0 + APID(11)
    let word1: u16 = apid & 0x7FF;
    buf.extend_from_slice(&word1.to_be_bytes());

    // Word 2: SeqFlags(2)=11(standalone) + SeqCount(14)
    let word2: u16 = (0b11 << 14) | (sequence & 0x3FFF);
    buf.extend_from_slice(&word2.to_be_bytes());

    // Word 3: DataLength - 1
    let word3: u16 = if data.is_empty() { 0 } else { (data.len() - 1) as u16 };
    buf.extend_from_slice(&word3.to_be_bytes());

    buf.extend_from_slice(data);
    buf
}

/// Decode a CCSDS space packet. Returns (apid, sequence, payload).
pub fn decode_ccsds(buf: &[u8]) -> Option<(u16, u16, Vec<u8>)> {
    if buf.len() < HEADER_SIZE {
        return None;
    }

    let word1 = u16::from_be_bytes([buf[0], buf[1]]);
    let word2 = u16::from_be_bytes([buf[2], buf[3]]);
    let word3 = u16::from_be_bytes([buf[4], buf[5]]);

    let version = (word1 >> 13) & 0x7;
    if version != 0 {
        return None;
    }

    let apid = word1 & 0x7FF;
    let sequence = word2 & 0x3FFF;
    let data_len = (word3 as usize) + 1;

    if buf.len() < HEADER_SIZE + data_len {
        return None;
    }

    Some((apid, sequence, buf[HEADER_SIZE..HEADER_SIZE + data_len].to_vec()))
}

/// CCSDS space packet transport.
///
/// Connects to a ground station modem via TCP and exchanges
/// CCSDS-framed packets for satellite/spacecraft communication.
pub struct CcsdsTransport {
    stream: Arc<Mutex<TcpStream>>,
    config: CcsdsConfig,
    sequence: Arc<std::sync::atomic::AtomicU16>,
}

impl CcsdsTransport {
    pub async fn new(config: CcsdsConfig) -> Result<Self> {
        let addr = format!("{}:{}", config.host, config.port);
        let stream = TcpStream::connect(&addr)
            .await
            .context(format!("Failed to connect to ground station at {addr}"))?;

        tracing::info!(
            "CCSDS connected to {addr} (APID {}, dest: {})",
            config.apid,
            config.destination
        );

        Ok(Self {
            stream: Arc::new(Mutex::new(stream)),
            config,
            sequence: Arc::new(std::sync::atomic::AtomicU16::new(0)),
        })
    }

    fn next_seq(&self) -> u16 {
        self.sequence
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            & 0x3FFF // Wrap at 14 bits
    }
}

#[async_trait::async_trait]
impl MeshTransport for CcsdsTransport {
    fn name(&self) -> &str {
        "ccsds"
    }

    async fn send(&self, frame: &[u8]) -> Result<()> {
        let seq = self.next_seq();
        let packet = encode_ccsds(self.config.apid, seq, frame);

        // Length-prefix for TCP framing
        let len = (packet.len() as u32).to_be_bytes(); // Big-endian (CCSDS convention)
        let mut stream = self.stream.lock().await;
        stream.write_all(&len).await?;
        stream.write_all(&packet).await?;
        stream.flush().await?;

        tracing::debug!(
            "CCSDS TX: {} bytes (APID {}, seq {}, dest {})",
            frame.len(),
            self.config.apid,
            seq,
            self.config.destination
        );
        Ok(())
    }

    async fn recv(&self, timeout_ms: u64) -> Result<Option<Vec<u8>>> {
        let mut stream = self.stream.lock().await;

        let mut len_buf = [0u8; 4];
        match tokio::time::timeout(
            tokio::time::Duration::from_millis(timeout_ms),
            stream.read_exact(&mut len_buf),
        )
        .await
        {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => return Ok(None),
        }

        let packet_len = u32::from_be_bytes(len_buf) as usize;
        if packet_len > 65_541 { // CCSDS max: 6 header + 65,535 data
            return Err(anyhow::anyhow!("CCSDS packet too large: {packet_len}"));
        }

        let mut packet = vec![0u8; packet_len];
        stream.read_exact(&mut packet).await?;

        match decode_ccsds(&packet) {
            Some((_apid, _seq, payload)) => Ok(Some(payload)),
            None => Ok(None),
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
    fn test_ccsds_encode_decode_roundtrip() {
        let data = b"consciousness vector delta";
        let packet = encode_ccsds(100, 42, data);

        assert!(packet.len() >= HEADER_SIZE + data.len());

        let (apid, seq, decoded) = decode_ccsds(&packet).unwrap();
        assert_eq!(apid, 100);
        assert_eq!(seq, 42);
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_ccsds_version_zero() {
        let packet = encode_ccsds(0, 0, &[0x42]);
        assert_eq!(packet[0] >> 5, 0); // Version bits are 0
    }

    #[test]
    fn test_ccsds_apid_mask() {
        // APID is 11 bits (0-2047)
        let packet = encode_ccsds(2047, 0, &[0]);
        let (apid, _, _) = decode_ccsds(&packet).unwrap();
        assert_eq!(apid, 2047);

        // APID > 2047 should be masked
        let packet = encode_ccsds(4095, 0, &[0]);
        let (apid, _, _) = decode_ccsds(&packet).unwrap();
        assert_eq!(apid, 2047); // Masked to 11 bits
    }

    #[test]
    fn test_ccsds_sequence_wrap() {
        // Sequence is 14 bits (0-16383)
        let packet = encode_ccsds(100, 16383, &[0]);
        let (_, seq, _) = decode_ccsds(&packet).unwrap();
        assert_eq!(seq, 16383);
    }

    #[test]
    fn test_ccsds_standalone_flags() {
        let packet = encode_ccsds(100, 0, &[0]);
        let word2 = u16::from_be_bytes([packet[2], packet[3]]);
        let seq_flags = (word2 >> 14) & 0x3;
        assert_eq!(seq_flags, 0b11); // Standalone packet
    }

    #[test]
    fn test_ccsds_too_short() {
        assert!(decode_ccsds(&[0; 3]).is_none());
    }

    #[test]
    fn test_ccsds_wrong_version() {
        let mut packet = encode_ccsds(100, 0, &[0]);
        packet[0] |= 0b111 << 5; // Set version to non-zero
        assert!(decode_ccsds(&packet).is_none());
    }

    #[test]
    fn test_ccsds_header_size() {
        assert_eq!(HEADER_SIZE, 6);
        let packet = encode_ccsds(100, 0, &[]);
        assert_eq!(packet.len(), 6); // Header only for empty payload
    }

    #[test]
    fn test_ccsds_config_default() {
        let config = CcsdsConfig::default();
        assert_eq!(config.port, 10025);
        assert_eq!(config.apid, 100);
        assert_eq!(config.destination, "LEO");
    }

    #[test]
    fn test_ccsds_all_apids() {
        // Verify all Symthaea APIDs encode/decode correctly
        for &apid in &[100u16, 101, 200, 201, 300, 400, 500] {
            let packet = encode_ccsds(apid, 0, b"test");
            let (decoded_apid, _, _) = decode_ccsds(&packet).unwrap();
            assert_eq!(decoded_apid, apid);
        }
    }
}
