// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mesh transport abstraction.
//!
//! Implements `MeshTransport` trait for:
//! - Loopback (testing)
//! - LoRa SX1276 via SPI (feature: `lora`)
//! - B.A.T.M.A.N. advanced mesh over bat0 UDP (feature: `batman`)

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Trait for mesh network transports.
///
/// Mirrors Symthaea's MeshTransport trait but adapted for async Tokio.
#[async_trait::async_trait]
pub trait MeshTransport: Send + Sync {
    /// Human-readable transport name.
    fn name(&self) -> &str;

    /// Send a frame (may be a fragment, not a full payload).
    async fn send(&self, frame: &[u8]) -> Result<()>;

    /// Receive a frame. Blocks until data available or timeout.
    async fn recv(&self, timeout_ms: u64) -> Result<Option<Vec<u8>>>;

    /// Clone into a boxed trait object.
    fn clone_box(&self) -> Box<dyn MeshTransport>;
}

// =============================================================================
// Loopback Transport (testing)
// =============================================================================

/// In-memory loopback for integration tests.
/// Sent frames are available to recv on the same instance.
#[derive(Clone)]
pub struct LoopbackTransport {
    buffer: Arc<Mutex<Vec<Vec<u8>>>>,
}

impl LoopbackTransport {
    pub fn new() -> Self {
        Self {
            buffer: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[async_trait::async_trait]
impl MeshTransport for LoopbackTransport {
    fn name(&self) -> &str {
        "loopback"
    }

    async fn send(&self, frame: &[u8]) -> Result<()> {
        self.buffer.lock().await.push(frame.to_vec());
        Ok(())
    }

    async fn recv(&self, timeout_ms: u64) -> Result<Option<Vec<u8>>> {
        let deadline = tokio::time::Instant::now()
            + tokio::time::Duration::from_millis(timeout_ms);
        loop {
            {
                let mut buf = self.buffer.lock().await;
                if !buf.is_empty() {
                    return Ok(Some(buf.remove(0)));
                }
            }
            if tokio::time::Instant::now() >= deadline {
                return Ok(None);
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }
    }

    fn clone_box(&self) -> Box<dyn MeshTransport> {
        Box::new(self.clone())
    }
}

// =============================================================================
// LoRa SX1276 Transport (Linux SPI)
// =============================================================================

#[cfg(feature = "lora")]
pub struct LoraTransport {
    // SPI device handle
    spi: Arc<Mutex<spidev::Spidev>>,
    frequency_mhz: f64,
}

#[cfg(feature = "lora")]
impl LoraTransport {
    pub fn new(spi_path: &str, frequency_mhz: f64) -> Result<Self> {
        use spidev::{Spidev, SpidevOptions, SpiModeFlags};
        use std::io;

        let mut spi = Spidev::open(spi_path)?;
        let options = SpidevOptions::new()
            .bits_per_word(8)
            .max_speed_hz(1_000_000)
            .mode(SpiModeFlags::SPI_MODE_0)
            .build();
        spi.configure(&options)?;

        tracing::info!("LoRa SX1276 initialized on {spi_path} at {frequency_mhz} MHz");

        Ok(Self {
            spi: Arc::new(Mutex::new(spi)),
            frequency_mhz,
        })
    }
}

#[cfg(feature = "lora")]
#[async_trait::async_trait]
impl MeshTransport for LoraTransport {
    fn name(&self) -> &str {
        "lora-sx1276"
    }

    async fn send(&self, frame: &[u8]) -> Result<()> {
        // TODO: Implement SX1276 register writes for TX
        // For now, log that we would send
        tracing::debug!("LoRa TX: {} bytes at {} MHz", frame.len(), self.frequency_mhz);
        Ok(())
    }

    async fn recv(&self, _timeout_ms: u64) -> Result<Option<Vec<u8>>> {
        // TODO: Implement SX1276 register reads for RX with IRQ
        Ok(None)
    }

    fn clone_box(&self) -> Box<dyn MeshTransport> {
        Box::new(Self {
            spi: self.spi.clone(),
            frequency_mhz: self.frequency_mhz,
        })
    }
}

// =============================================================================
// B.A.T.M.A.N. Transport (UDP over bat0)
// =============================================================================

#[cfg(feature = "batman")]
pub struct BatmanTransport {
    socket: Arc<tokio::net::UdpSocket>,
    broadcast_addr: std::net::SocketAddr,
}

#[cfg(feature = "batman")]
impl BatmanTransport {
    pub async fn new(bind_port: u16, broadcast_port: u16) -> Result<Self> {
        let socket = tokio::net::UdpSocket::bind(format!("0.0.0.0:{bind_port}")).await?;
        socket.set_broadcast(true)?;
        let broadcast_addr: std::net::SocketAddr =
            format!("255.255.255.255:{broadcast_port}").parse()?;
        tracing::info!("B.A.T.M.A.N. transport on bat0, port {bind_port}");
        Ok(Self {
            socket: Arc::new(socket),
            broadcast_addr,
        })
    }
}

#[cfg(feature = "batman")]
#[async_trait::async_trait]
impl MeshTransport for BatmanTransport {
    fn name(&self) -> &str {
        "batman-udp"
    }

    async fn send(&self, frame: &[u8]) -> Result<()> {
        self.socket.send_to(frame, self.broadcast_addr).await?;
        Ok(())
    }

    async fn recv(&self, timeout_ms: u64) -> Result<Option<Vec<u8>>> {
        let mut buf = vec![0u8; 1024];
        match tokio::time::timeout(
            tokio::time::Duration::from_millis(timeout_ms),
            self.socket.recv_from(&mut buf),
        )
        .await
        {
            Ok(Ok((len, _addr))) => Ok(Some(buf[..len].to_vec())),
            Ok(Err(e)) => Err(e.into()),
            Err(_) => Ok(None), // timeout
        }
    }

    fn clone_box(&self) -> Box<dyn MeshTransport> {
        Box::new(Self {
            socket: self.socket.clone(),
            broadcast_addr: self.broadcast_addr,
        })
    }
}

// =============================================================================
// Transport Factory
// =============================================================================

/// Create transport based on available features and environment.
///
/// Set `MESH_TRANSPORT` to one of:
///   - `loopback` (default) — in-memory testing
///   - `lora` — direct SX1276 SPI (feature: `lora`)
///   - `batman` — B.A.T.M.A.N. UDP mesh (feature: `batman`)
///   - `meshtastic` — Meshtastic device via serial/TCP (feature: `meshtastic`)
///   - `espnow` — ESP-NOW peer-to-peer WiFi via serial bridge (feature: `espnow`)
///   - `reticulum` — Reticulum daemon store-and-forward mesh (feature: `reticulum`)
///   - `ccsds` — CCSDS space packet via ground station modem (feature: `ccsds`)
///   - `mqtt` — MQTT broker gateway (feature: `mqtt`)
pub fn create_transport() -> Result<Box<dyn MeshTransport>> {
    // Check env for transport preference
    let transport_type = std::env::var("MESH_TRANSPORT").unwrap_or_else(|_| "loopback".into());

    match transport_type.as_str() {
        #[cfg(feature = "lora")]
        "lora" => {
            let spi_path = std::env::var("LORA_SPI_PATH").unwrap_or_else(|_| "/dev/spidev0.0".into());
            let freq: f64 = std::env::var("LORA_FREQ_MHZ")
                .unwrap_or_else(|_| "868.0".into())
                .parse()
                .unwrap_or(868.0);
            Ok(Box::new(LoraTransport::new(&spi_path, freq)?))
        }
        #[cfg(feature = "batman")]
        "batman" => {
            let rt = tokio::runtime::Handle::current();
            let transport = rt.block_on(BatmanTransport::new(9735, 9735))?;
            Ok(Box::new(transport))
        }
        "meshtastic" => {
            let config = crate::meshtastic::MeshtasticConfig::from_env();
            let transport = crate::meshtastic::MeshtasticTransport::new(config)?;
            Ok(Box::new(transport))
        }
        #[cfg(feature = "espnow")]
        "espnow" => {
            let config = crate::espnow::EspNowConfig::from_env();
            let transport = crate::espnow::EspNowTransport::new(config)?;
            Ok(Box::new(transport))
        }
        #[cfg(feature = "reticulum")]
        "reticulum" => {
            let config = crate::reticulum::ReticulumConfig::from_env();
            let rt = tokio::runtime::Handle::current();
            let transport = rt.block_on(crate::reticulum::ReticulumTransport::new(config))?;
            Ok(Box::new(transport))
        }
        #[cfg(feature = "ccsds")]
        "ccsds" => {
            let config = crate::ccsds::CcsdsConfig::from_env();
            let rt = tokio::runtime::Handle::current();
            let transport = rt.block_on(crate::ccsds::CcsdsTransport::new(config))?;
            Ok(Box::new(transport))
        }
        #[cfg(feature = "mqtt")]
        "mqtt" => {
            let config = crate::mqtt::MqttConfig::from_env();
            let rt = tokio::runtime::Handle::current();
            let transport = rt.block_on(crate::mqtt::MqttTransport::new(config))?;
            Ok(Box::new(transport))
        }
        _ => {
            tracing::info!(
                "Using loopback transport (set MESH_TRANSPORT=lora|batman|meshtastic|mqtt for real mesh)"
            );
            Ok(Box::new(LoopbackTransport::new()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_loopback_roundtrip() {
        let transport = LoopbackTransport::new();
        transport.send(b"hello mesh").await.unwrap();
        let received = transport.recv(1000).await.unwrap();
        assert_eq!(received, Some(b"hello mesh".to_vec()));
    }

    #[tokio::test]
    async fn test_loopback_timeout() {
        let transport = LoopbackTransport::new();
        let received = transport.recv(50).await.unwrap();
        assert!(received.is_none());
    }

    #[tokio::test]
    async fn test_loopback_ordering() {
        let transport = LoopbackTransport::new();
        transport.send(b"first").await.unwrap();
        transport.send(b"second").await.unwrap();
        assert_eq!(transport.recv(100).await.unwrap(), Some(b"first".to_vec()));
        assert_eq!(transport.recv(100).await.unwrap(), Some(b"second".to_vec()));
    }

    #[tokio::test]
    async fn test_clone_box_shares_buffer() {
        let transport = LoopbackTransport::new();
        let cloned = transport.clone_box();

        // Send on original, receive on clone — shared Arc<Mutex<Vec>>
        transport.send(b"shared").await.unwrap();
        let received = cloned.recv(100).await.unwrap();
        assert_eq!(received, Some(b"shared".to_vec()));
    }

    #[tokio::test]
    async fn test_clone_box_bidirectional() {
        let transport = LoopbackTransport::new();
        let cloned = transport.clone_box();

        // Send on clone, receive on original
        cloned.send(b"from-clone").await.unwrap();
        let received = transport.recv(100).await.unwrap();
        assert_eq!(received, Some(b"from-clone".to_vec()));
    }

    #[test]
    fn test_create_transport_default() {
        // Without MESH_TRANSPORT env var, defaults to loopback
        std::env::remove_var("MESH_TRANSPORT");
        let t = create_transport().unwrap();
        assert_eq!(t.name(), "loopback");
    }

    #[test]
    fn test_create_transport_explicit_loopback() {
        std::env::set_var("MESH_TRANSPORT", "loopback");
        let t = create_transport().unwrap();
        assert_eq!(t.name(), "loopback");
        std::env::remove_var("MESH_TRANSPORT");
    }

    #[test]
    fn test_loopback_name() {
        let transport = LoopbackTransport::new();
        assert_eq!(transport.name(), "loopback");
    }

    #[tokio::test]
    async fn test_loopback_empty_frame() {
        let transport = LoopbackTransport::new();
        transport.send(b"").await.unwrap();
        let received = transport.recv(100).await.unwrap();
        assert_eq!(received, Some(vec![]));
    }

    #[tokio::test]
    async fn test_loopback_large_frame() {
        let transport = LoopbackTransport::new();
        let large = vec![0xABu8; 4096];
        transport.send(&large).await.unwrap();
        let received = transport.recv(100).await.unwrap();
        assert_eq!(received.unwrap().len(), 4096);
    }
}
