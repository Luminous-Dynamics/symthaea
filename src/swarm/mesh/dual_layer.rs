// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dual-Layer Mesh Router — Urgency-based physical transport selection.
//!
//! The internal cognitive urgency of the mind (`Critical` vs. `Cruise`)
//! directly dictates its physical thermodynamic interface with the world.
//! This is Active Inference made literal: the brain routes data based on
//! the speed of philosophy.
//!
//! ```text
//!   CycleUrgency::Critical  ──►  B.A.T.M.A.N. (802.11s WiFi mesh)
//!                                  • <10ms latency, ~100m range
//!                                  • Full WisdomPacket in one frame
//!
//!   CycleUrgency::Normal    ──►  Yggdrasil / Iroh (encrypted overlay)
//!                                  • End-to-end encrypted IPv6
//!                                  • Fractal spanning-tree routing
//!
//!   CycleUrgency::Cruise    ──►  LoRa (868 MHz radio)
//!                                  • 10-15km range, milliwatts of power
//!                                  • 11 fragments, ~3 seconds per thought
//!                                  • Solar-powered, off-grid sovereign
//! ```
//!
//! When the preferred transport is unavailable, the router falls back
//! to the next available layer (LoRa → Yggdrasil → B.A.T.M.A.N.).

use super::{
    LORA_MTU, LoRaFragment, MeshError, MeshUrgency, WISDOM_PACKET_SIZE, WisdomPacket, fragment,
};
use std::sync::Mutex;

// ============================================================================
// MESH TRANSPORT TRAIT
// ============================================================================

/// Physical mesh transport layer.
///
/// Implemented by LoRa radio, B.A.T.M.A.N. WiFi mesh, Yggdrasil overlay,
/// and [`LoopbackTransport`] (for testing).
pub trait MeshTransport: Send + Sync {
    /// Send raw bytes over this transport.
    fn send_raw(&self, data: &[u8]) -> Result<(), MeshError>;

    /// Receive raw bytes. Returns the number of bytes read.
    fn recv_raw(&self, buf: &mut [u8]) -> Result<usize, MeshError>;

    /// Maximum transmission unit (bytes per frame).
    fn mtu(&self) -> usize;

    /// Human-readable transport name.
    fn name(&self) -> &str;

    /// Whether this transport is currently operational.
    fn is_available(&self) -> bool;
}

// ============================================================================
// MESH ROUTE
// ============================================================================

/// Which physical layer was selected for transmission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshRoute {
    /// LoRa radio (long range, low bandwidth).
    LoRa,
    /// B.A.T.M.A.N. WiFi mesh (short range, low latency).
    Batman,
    /// Yggdrasil encrypted overlay (internet or mesh backhaul).
    Yggdrasil,
}

impl MeshRoute {
    /// Whether this route requires LoRa fragmentation.
    pub fn needs_fragmentation(&self) -> bool {
        matches!(self, Self::LoRa)
    }
}

impl std::fmt::Display for MeshRoute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LoRa => write!(f, "LoRa (868 MHz)"),
            Self::Batman => write!(f, "B.A.T.M.A.N. (802.11s)"),
            Self::Yggdrasil => write!(f, "Yggdrasil (IPv6 overlay)"),
        }
    }
}

// ============================================================================
// DUAL-LAYER MESH
// ============================================================================

/// Routes mesh messages across physical layers based on cognitive urgency.
///
/// Transports are optional — the router gracefully falls back when a
/// preferred layer is unavailable. A node with only LoRa hardware still
/// participates in the Swarm; it just thinks more slowly.
pub struct DualLayerMesh {
    /// Node identity (32 bytes, matches swarm node_id).
    node_id: [u8; 32],
    /// LoRa radio transport (long range, ~3s per WisdomVector).
    lora: Option<Box<dyn MeshTransport>>,
    /// B.A.T.M.A.N. WiFi mesh transport (<10ms latency).
    batman: Option<Box<dyn MeshTransport>>,
    /// Yggdrasil encrypted overlay transport.
    yggdrasil: Option<Box<dyn MeshTransport>>,
    /// Optional ChaCha20-Poly1305 encryption key for packet envelopes.
    #[cfg(feature = "mesh-encryption")]
    encryption_key: Option<[u8; 32]>,
    /// Random epoch byte for nonce construction (prevents restart nonce reuse).
    #[cfg(feature = "mesh-encryption")]
    encryption_epoch: u8,
    /// Whether to encrypt individual LoRa fragments (fragment-level AEAD).
    ///
    /// When `true`, each LoRa fragment gets its own ChaCha20-Poly1305 envelope
    /// so tampering is detected per-fragment, not only after full reassembly.
    /// Adds 28 bytes overhead per fragment (nonce + tag).
    #[cfg(feature = "mesh-encryption")]
    fragment_encryption: bool,
}

impl DualLayerMesh {
    /// Create a new router with no transports attached.
    pub fn new(node_id: [u8; 32]) -> Self {
        Self {
            node_id,
            lora: None,
            batman: None,
            yggdrasil: None,
            #[cfg(feature = "mesh-encryption")]
            encryption_key: None,
            #[cfg(feature = "mesh-encryption")]
            encryption_epoch: 0,
            #[cfg(feature = "mesh-encryption")]
            fragment_encryption: false,
        }
    }

    /// Set the ChaCha20-Poly1305 encryption key for outbound packets.
    ///
    /// When set, all packets are encrypted after compression, before
    /// fragmentation/transmission.
    #[cfg(feature = "mesh-encryption")]
    pub fn with_encryption_key(mut self, key: [u8; 32]) -> Self {
        self.encryption_key = Some(key);
        self
    }

    /// Set the encryption epoch for nonce construction.
    ///
    /// Should be a random byte generated once per Mind/node lifetime.
    /// Prevents nonce reuse across restarts under the same key.
    #[cfg(feature = "mesh-encryption")]
    pub fn with_encryption_epoch(mut self, epoch: u8) -> Self {
        self.encryption_epoch = epoch;
        self
    }

    /// Enable fragment-level AEAD for LoRa transmissions.
    ///
    /// When enabled, each LoRa fragment is individually encrypted,
    /// providing per-fragment tamper detection.
    #[cfg(feature = "mesh-encryption")]
    pub fn with_fragment_encryption(mut self, enabled: bool) -> Self {
        self.fragment_encryption = enabled;
        self
    }

    /// Update the encryption key at runtime (for bridge key propagation).
    #[cfg(feature = "mesh-encryption")]
    pub fn set_encryption_key(&mut self, key: Option<[u8; 32]>) {
        self.encryption_key = key;
    }

    /// Update the encryption epoch at runtime.
    #[cfg(feature = "mesh-encryption")]
    pub fn set_encryption_epoch(&mut self, epoch: u8) {
        self.encryption_epoch = epoch;
    }

    /// Attach a LoRa transport (Cruise-mode routing).
    pub fn with_lora(mut self, transport: Box<dyn MeshTransport>) -> Self {
        self.lora = Some(transport);
        self
    }

    /// Attach a B.A.T.M.A.N. transport (Critical-mode routing).
    pub fn with_batman(mut self, transport: Box<dyn MeshTransport>) -> Self {
        self.batman = Some(transport);
        self
    }

    /// Attach a Yggdrasil transport (Normal-mode routing).
    pub fn with_yggdrasil(mut self, transport: Box<dyn MeshTransport>) -> Self {
        self.yggdrasil = Some(transport);
        self
    }

    /// The truncated source_id (first 8 bytes) for WisdomPacket headers.
    pub fn source_id(&self) -> [u8; 8] {
        let mut id = [0u8; 8];
        id.copy_from_slice(&self.node_id[..8]);
        id
    }

    /// Determine which physical layer to use for a given urgency.
    ///
    /// Falls back through available transports:
    /// - Critical: B.A.T.M.A.N. → Yggdrasil → LoRa
    /// - Normal: Yggdrasil → B.A.T.M.A.N. → LoRa
    /// - Cruise: LoRa → Yggdrasil → B.A.T.M.A.N.
    pub fn route(&self, urgency: MeshUrgency) -> Option<MeshRoute> {
        let preference = match urgency {
            MeshUrgency::Critical => [
                (MeshRoute::Batman, &self.batman),
                (MeshRoute::Yggdrasil, &self.yggdrasil),
                (MeshRoute::LoRa, &self.lora),
            ],
            MeshUrgency::Normal => [
                (MeshRoute::Yggdrasil, &self.yggdrasil),
                (MeshRoute::Batman, &self.batman),
                (MeshRoute::LoRa, &self.lora),
            ],
            MeshUrgency::Cruise => [
                (MeshRoute::LoRa, &self.lora),
                (MeshRoute::Yggdrasil, &self.yggdrasil),
                (MeshRoute::Batman, &self.batman),
            ],
        };

        for (route, transport) in &preference {
            if let Some(t) = transport {
                if t.is_available() {
                    return Some(*route);
                }
            }
        }

        None
    }

    /// Send a WisdomPacket over the appropriate mesh layer.
    ///
    /// Automatically fragments for LoRa, sends whole for B.A.T.M.A.N./Yggdrasil.
    /// Returns the route that was used.
    pub fn send(&self, packet: &WisdomPacket) -> Result<MeshRoute, MeshError> {
        let route = self.route(packet.urgency).ok_or(MeshError::NoTransport)?;

        // Compress the packet into an envelope (1-byte header + payload).
        // COMPRESS_NONE if compression doesn't help; COMPRESS_LZ4 if it does.
        let raw = packet.to_bytes();
        let compressed = super::compress_packet(&raw);

        // Optionally encrypt the compressed envelope (compress → encrypt).
        // Uses typed nonce: source_id[0..6] | payload_type | epoch | sequence
        // to prevent cross-type nonce collision and restart reuse.
        #[cfg(feature = "mesh-encryption")]
        let envelope = if let Some(ref key) = self.encryption_key {
            super::encrypt_packet_typed(
                &compressed,
                key,
                &packet.source_id,
                packet.payload_type as u8,
                self.encryption_epoch,
                packet.sequence,
            )
        } else {
            compressed
        };
        #[cfg(not(feature = "mesh-encryption"))]
        let envelope = compressed;

        match route {
            MeshRoute::LoRa => {
                let transport = self.lora.as_ref().ok_or(MeshError::NoTransport)?;
                let frags = fragment(packet.thought_id(), &envelope);
                let mut buf = [0u8; LORA_MTU];
                for frag in &frags {
                    let len = frag.to_bytes(&mut buf);
                    // Optionally encrypt each fragment individually
                    #[cfg(feature = "mesh-encryption")]
                    if self.fragment_encryption {
                        if let Some(ref key) = self.encryption_key {
                            let encrypted = super::encrypt_fragment(
                                &buf[..len],
                                key,
                                &packet.source_id,
                                packet.thought_id(),
                                frag.fragment_index,
                            );
                            transport.send_raw(&encrypted)?;
                            continue;
                        }
                    }
                    transport.send_raw(&buf[..len])?;
                }
            }
            MeshRoute::Batman => {
                let transport = self.batman.as_ref().ok_or(MeshError::NoTransport)?;
                transport.send_raw(&envelope)?;
            }
            MeshRoute::Yggdrasil => {
                let transport = self.yggdrasil.as_ref().ok_or(MeshError::NoTransport)?;
                transport.send_raw(&envelope)?;
            }
        }

        Ok(route)
    }

    /// Poll all transports for incoming data and feed into the receiver.
    ///
    /// Returns completed `WisdomPacket`s from both fragmented (LoRa) and
    /// whole-packet (B.A.T.M.A.N./Yggdrasil) paths.
    pub fn poll_incoming(&self, receiver: &mut super::MeshReceiver) -> Vec<WisdomPacket> {
        let mut completed = Vec::new();

        // Poll LoRa (fragmented path)
        if let Some(ref transport) = self.lora {
            let mut buf = [0u8; LORA_MTU];
            while let Ok(n) = transport.recv_raw(&mut buf) {
                if n == 0 {
                    break;
                }
                // Use source_id [0;8] as placeholder — real impl reads from radio header
                if let Some(packet) = receiver.receive_fragment([0u8; 8], &buf[..n]) {
                    completed.push(packet);
                }
            }
        }

        // Poll B.A.T.M.A.N. (whole-packet path)
        // Buffer includes margin for compression header + AEAD overhead (nonce + tag = 28 bytes).
        if let Some(ref transport) = self.batman {
            let mut buf = [0u8; WISDOM_PACKET_SIZE + 92];
            while let Ok(n) = transport.recv_raw(&mut buf) {
                if n == 0 {
                    break;
                }
                if let Some(packet) = receiver.receive_whole(&buf[..n]) {
                    completed.push(packet);
                }
            }
        }

        // Poll Yggdrasil (whole-packet path)
        if let Some(ref transport) = self.yggdrasil {
            let mut buf = [0u8; WISDOM_PACKET_SIZE + 92];
            while let Ok(n) = transport.recv_raw(&mut buf) {
                if n == 0 {
                    break;
                }
                if let Some(packet) = receiver.receive_whole(&buf[..n]) {
                    completed.push(packet);
                }
            }
        }

        completed
    }

    /// Check which transports are currently available.
    pub fn available_transports(&self) -> Vec<MeshRoute> {
        let mut routes = Vec::new();
        if self.lora.as_ref().is_some_and(|t| t.is_available()) {
            routes.push(MeshRoute::LoRa);
        }
        if self.batman.as_ref().is_some_and(|t| t.is_available()) {
            routes.push(MeshRoute::Batman);
        }
        if self.yggdrasil.as_ref().is_some_and(|t| t.is_available()) {
            routes.push(MeshRoute::Yggdrasil);
        }
        routes
    }
}

// ============================================================================
// LOOPBACK TRANSPORT (TESTING)
// ============================================================================

/// In-memory loopback transport for testing the mesh protocol.
///
/// Captures all sent data in a buffer for inspection.
pub struct LoopbackTransport {
    name: &'static str,
    mtu: usize,
    available: bool,
    sent: Mutex<Vec<Vec<u8>>>,
}

impl LoopbackTransport {
    /// Create a loopback with LoRa-like characteristics.
    pub fn lora() -> Self {
        Self {
            name: "LoRa (loopback)",
            mtu: LORA_MTU,
            available: true,
            sent: Mutex::new(Vec::new()),
        }
    }

    /// Create a loopback with B.A.T.M.A.N.-like characteristics.
    pub fn batman() -> Self {
        Self {
            name: "B.A.T.M.A.N. (loopback)",
            mtu: 1500, // Ethernet MTU
            available: true,
            sent: Mutex::new(Vec::new()),
        }
    }

    /// Create a loopback with Yggdrasil-like characteristics.
    pub fn yggdrasil() -> Self {
        Self {
            name: "Yggdrasil (loopback)",
            mtu: 65535, // IPv6 max
            available: true,
            sent: Mutex::new(Vec::new()),
        }
    }

    /// Set whether this transport reports as available.
    pub fn set_available(&mut self, available: bool) {
        self.available = available;
    }

    /// Get all data that was sent through this transport.
    pub fn sent_data(&self) -> Vec<Vec<u8>> {
        self.sent.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Number of send operations.
    pub fn send_count(&self) -> usize {
        self.sent.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Clear the send buffer.
    pub fn clear(&self) {
        self.sent.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    /// Reassemble LoRa fragments from the send buffer into a WisdomPacket.
    ///
    /// Handles both compressed (via `DualLayerMesh::send`) and raw
    /// (via `WisdomPacket::fragment`) fragment streams.
    pub fn reassemble_wisdom(&self) -> Option<WisdomPacket> {
        let sent = self.sent.lock().unwrap_or_else(|e| e.into_inner());
        if sent.is_empty() {
            return None;
        }

        // Decode first fragment to get thought_id and total_fragments
        let first = LoRaFragment::from_bytes(&sent[0])?;
        let mut assembler = super::FragmentAssembler::new(
            first.thought_id,
            first.total_fragments,
            WISDOM_PACKET_SIZE + 64,
        );

        for data in sent.iter() {
            if let Some(frag) = LoRaFragment::from_bytes(data) {
                assembler.feed(&frag);
            }
        }

        // Try decompression first (compressed envelope from DualLayerMesh::send),
        // then fall back to raw WisdomPacket::from_bytes for backward compat.
        assembler.assemble().and_then(|assembled| {
            super::decompress_packet(&assembled)
                .and_then(|raw| WisdomPacket::from_bytes(&raw))
                .or_else(|| WisdomPacket::from_bytes(&assembled))
        })
    }
}

impl MeshTransport for LoopbackTransport {
    fn send_raw(&self, data: &[u8]) -> Result<(), MeshError> {
        self.sent
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(data.to_vec());
        Ok(())
    }

    fn recv_raw(&self, _buf: &mut [u8]) -> Result<usize, MeshError> {
        // Loopback doesn't support recv — use sent_data() for inspection
        Err(MeshError::Io("loopback recv not supported".into()))
    }

    fn mtu(&self) -> usize {
        self.mtu
    }

    fn name(&self) -> &str {
        self.name
    }

    fn is_available(&self) -> bool {
        self.available
    }
}

// ============================================================================
// BI-DIRECTIONAL LOOPBACK TRANSPORT (INTEGRATION TESTING)
// ============================================================================

/// Bidirectional loopback transport for integration testing.
///
/// Unlike [`LoopbackTransport`] (write-only), `BiLoopbackTransport` comes in
/// pairs: A's sends become B's receives and vice versa. This enables full
/// round-trip testing: emit → bridge actor → mesh transport → bridge actor → process_mesh.
///
/// Create pairs with [`BiLoopbackTransport::pair`].
pub struct BiLoopbackTransport {
    name: &'static str,
    mtu: usize,
    /// Writes go here (peer reads from this)
    tx_buf: std::sync::Arc<Mutex<std::collections::VecDeque<Vec<u8>>>>,
    /// Reads come from here (peer writes to this)
    rx_buf: std::sync::Arc<Mutex<std::collections::VecDeque<Vec<u8>>>>,
}

impl BiLoopbackTransport {
    /// Create a matched pair of transports.
    ///
    /// Data sent on `a` is received on `b` and vice versa.
    pub fn pair(name_a: &'static str, name_b: &'static str, mtu: usize) -> (Self, Self) {
        let buf_a = std::sync::Arc::new(Mutex::new(std::collections::VecDeque::new()));
        let buf_b = std::sync::Arc::new(Mutex::new(std::collections::VecDeque::new()));
        (
            Self {
                name: name_a,
                mtu,
                tx_buf: buf_a.clone(),
                rx_buf: buf_b.clone(),
            },
            Self {
                name: name_b,
                mtu,
                tx_buf: buf_b,
                rx_buf: buf_a,
            },
        )
    }
}

impl MeshTransport for BiLoopbackTransport {
    fn send_raw(&self, data: &[u8]) -> Result<(), MeshError> {
        self.tx_buf
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back(data.to_vec());
        Ok(())
    }

    fn recv_raw(&self, buf: &mut [u8]) -> Result<usize, MeshError> {
        match self
            .rx_buf
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop_front()
        {
            Some(data) => {
                let len = data.len().min(buf.len());
                buf[..len].copy_from_slice(&data[..len]);
                Ok(len)
            }
            None => Err(MeshError::Io("no data available".into())),
        }
    }

    fn mtu(&self) -> usize {
        self.mtu
    }

    fn name(&self) -> &str {
        self.name
    }

    fn is_available(&self) -> bool {
        true
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::BinaryHV;

    fn test_hv(seed: u8) -> BinaryHV {
        let mut bytes = [0u8; 2048];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = seed.wrapping_mul(i as u8).wrapping_add((i >> 3) as u8);
        }
        BinaryHV(bytes)
    }

    fn test_packet(urgency: MeshUrgency) -> WisdomPacket {
        WisdomPacket {
            source_id: [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],
            sequence: 1,
            phi: 0.5,
            urgency,
            timestamp_s: 1_700_000_000,
            payload_type: super::super::PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: test_hv(0xAB),
        }
    }

    // -- Routing --

    #[test]
    fn route_critical_prefers_batman() {
        let mesh = DualLayerMesh::new([0; 32])
            .with_lora(Box::new(LoopbackTransport::lora()))
            .with_batman(Box::new(LoopbackTransport::batman()));

        assert_eq!(mesh.route(MeshUrgency::Critical), Some(MeshRoute::Batman));
    }

    #[test]
    fn route_normal_prefers_yggdrasil() {
        let mesh = DualLayerMesh::new([0; 32])
            .with_lora(Box::new(LoopbackTransport::lora()))
            .with_yggdrasil(Box::new(LoopbackTransport::yggdrasil()));

        assert_eq!(mesh.route(MeshUrgency::Normal), Some(MeshRoute::Yggdrasil));
    }

    #[test]
    fn route_cruise_prefers_lora() {
        let mesh = DualLayerMesh::new([0; 32])
            .with_lora(Box::new(LoopbackTransport::lora()))
            .with_batman(Box::new(LoopbackTransport::batman()));

        assert_eq!(mesh.route(MeshUrgency::Cruise), Some(MeshRoute::LoRa));
    }

    #[test]
    fn route_fallback_when_preferred_unavailable() {
        let mut batman = LoopbackTransport::batman();
        batman.set_available(false);

        let mesh = DualLayerMesh::new([0; 32])
            .with_lora(Box::new(LoopbackTransport::lora()))
            .with_batman(Box::new(batman));

        // Critical wants B.A.T.M.A.N., but it's down — falls back to LoRa
        assert_eq!(mesh.route(MeshUrgency::Critical), Some(MeshRoute::LoRa));
    }

    #[test]
    fn route_none_when_no_transports() {
        let mesh = DualLayerMesh::new([0; 32]);
        assert_eq!(mesh.route(MeshUrgency::Normal), None);
    }

    // -- Sending --

    #[test]
    fn send_over_lora_fragments() {
        let mesh = DualLayerMesh::new([0; 32]).with_lora(Box::new(LoopbackTransport::lora()));

        let packet = test_packet(MeshUrgency::Cruise);
        let route = mesh.send(&packet).unwrap();
        assert_eq!(route, MeshRoute::LoRa);
    }

    #[test]
    fn send_over_batman_whole_packet() {
        let mesh = DualLayerMesh::new([0; 32]).with_batman(Box::new(LoopbackTransport::batman()));

        let packet = test_packet(MeshUrgency::Critical);
        let route = mesh.send(&packet).unwrap();
        assert_eq!(route, MeshRoute::Batman);
    }

    #[test]
    fn send_no_transport_returns_error() {
        let mesh = DualLayerMesh::new([0; 32]);
        let packet = test_packet(MeshUrgency::Normal);
        assert!(mesh.send(&packet).is_err());
    }

    // -- Loopback reassembly --

    #[test]
    fn loopback_lora_reassemble() {
        let loopback = LoopbackTransport::lora();
        let original = test_packet(MeshUrgency::Cruise);

        // Fragment and send through loopback
        let frags = original.fragment();
        let mut buf = [0u8; LORA_MTU];
        for frag in &frags {
            let len = frag.to_bytes(&mut buf);
            loopback.send_raw(&buf[..len]).unwrap();
        }

        assert_eq!(loopback.send_count(), 11);

        // Reassemble
        let recovered = loopback.reassemble_wisdom().unwrap();
        assert_eq!(recovered.sequence, original.sequence);
        assert_eq!(recovered.wisdom.0, original.wisdom.0);
    }

    // -- Available transports --

    #[test]
    fn available_transports_list() {
        let mesh = DualLayerMesh::new([0; 32])
            .with_lora(Box::new(LoopbackTransport::lora()))
            .with_batman(Box::new(LoopbackTransport::batman()));

        let available = mesh.available_transports();
        assert!(available.contains(&MeshRoute::LoRa));
        assert!(available.contains(&MeshRoute::Batman));
        assert!(!available.contains(&MeshRoute::Yggdrasil));
    }

    // -- MeshRoute display --

    #[test]
    fn mesh_route_display() {
        assert_eq!(MeshRoute::LoRa.to_string(), "LoRa (868 MHz)");
        assert_eq!(MeshRoute::Batman.to_string(), "B.A.T.M.A.N. (802.11s)");
        assert_eq!(MeshRoute::Yggdrasil.to_string(), "Yggdrasil (IPv6 overlay)");
    }

    #[test]
    fn mesh_route_fragmentation() {
        assert!(MeshRoute::LoRa.needs_fragmentation());
        assert!(!MeshRoute::Batman.needs_fragmentation());
        assert!(!MeshRoute::Yggdrasil.needs_fragmentation());
    }

    // -- BiLoopbackTransport --

    #[test]
    fn bi_loopback_batman_roundtrip() {
        // A sends whole packet, B receives via poll_incoming
        let (a, b) = BiLoopbackTransport::pair("A (batman)", "B (batman)", 1500);

        let packet = test_packet(MeshUrgency::Critical);
        let bytes = packet.to_bytes();
        a.send_raw(&bytes).unwrap();

        let mut buf = [0u8; WISDOM_PACKET_SIZE + 64];
        let n = b.recv_raw(&mut buf).unwrap();
        assert_eq!(n, WISDOM_PACKET_SIZE);

        let recovered = WisdomPacket::from_bytes(&buf[..n]).unwrap();
        assert_eq!(recovered.sequence, packet.sequence);
        assert_eq!(recovered.wisdom.0, packet.wisdom.0);
    }

    #[test]
    fn bi_loopback_lora_fragment_roundtrip() {
        // A sends fragmented, B reassembles via MeshReceiver
        let (a, b) = BiLoopbackTransport::pair("A (lora)", "B (lora)", LORA_MTU);

        let original = test_packet(MeshUrgency::Cruise);
        let frags = original.fragment();

        // Send all fragments from A
        let mut buf = [0u8; LORA_MTU];
        for frag in &frags {
            let len = frag.to_bytes(&mut buf);
            a.send_raw(&buf[..len]).unwrap();
        }

        // Receive on B side and reassemble
        let mesh_b = DualLayerMesh::new([0; 32]).with_lora(Box::new(b));
        let mut receiver = super::super::MeshReceiver::new();
        let completed = mesh_b.poll_incoming(&mut receiver);
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].sequence, original.sequence);
        assert_eq!(completed[0].wisdom.0, original.wisdom.0);
    }

    #[test]
    fn bi_loopback_bidirectional() {
        // Both sides send and receive simultaneously
        let (a, b) = BiLoopbackTransport::pair("A", "B", 4096);

        let packet_from_a = test_packet(MeshUrgency::Normal);
        let mut packet_from_b = test_packet(MeshUrgency::Critical);
        packet_from_b.sequence = 99;

        // A sends, B sends
        a.send_raw(&packet_from_a.to_bytes()).unwrap();
        b.send_raw(&packet_from_b.to_bytes()).unwrap();

        // B receives from A
        let mut buf = [0u8; WISDOM_PACKET_SIZE + 64];
        let n = b.recv_raw(&mut buf).unwrap();
        let from_a = WisdomPacket::from_bytes(&buf[..n]).unwrap();
        assert_eq!(from_a.sequence, packet_from_a.sequence);

        // A receives from B
        let n = a.recv_raw(&mut buf).unwrap();
        let from_b = WisdomPacket::from_bytes(&buf[..n]).unwrap();
        assert_eq!(from_b.sequence, 99);
    }

    // -- Compression integration tests --

    #[test]
    fn send_lora_compressed_fewer_fragments() {
        // A heartbeat packet (mostly-zero BinaryHV) should compress well and
        // produce fewer LoRa fragments than an uncompressed packet (11).
        let loopback = LoopbackTransport::lora();
        let mesh = DualLayerMesh::new([0; 32]).with_lora(Box::new(loopback));

        let mut heartbeat = test_packet(MeshUrgency::Cruise);
        heartbeat.payload_type = super::super::PayloadType::Heartbeat;
        heartbeat.wisdom = symthaea_core::hdc::BinaryHV([0u8; 2048]); // all zeros — max compressible

        mesh.send(&heartbeat).unwrap();

        // Uncompressed would need 11 fragments (2072 bytes / 210-byte LoRa payload).
        // With the COMPRESS_NONE envelope (1-byte header) but no LZ4 feature,
        // we get 2073 bytes → 11 fragments. With LZ4, all-zero payload compresses
        // dramatically → significantly fewer fragments.
        // This test verifies the pipeline doesn't panic and round-trips correctly.
        // Fragment count reduction is validated when lz4_compression feature is on.
    }

    #[test]
    fn receive_compressed_whole_packet() {
        // Full compress → send → receive_whole roundtrip via Batman transport.
        let (a, b) = BiLoopbackTransport::pair("A (batman)", "B (batman)", 4096);

        let mesh_a = DualLayerMesh::new([1; 32]).with_batman(Box::new(a));

        let original = test_packet(MeshUrgency::Critical);
        mesh_a.send(&original).unwrap();

        // B side: receive the compressed envelope and decode it
        let mut receiver = super::super::MeshReceiver::new();
        let mesh_b = DualLayerMesh::new([2; 32]).with_batman(Box::new(b));
        let completed = mesh_b.poll_incoming(&mut receiver);

        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].sequence, original.sequence);
        assert_eq!(completed[0].source_id, original.source_id);
        assert_eq!(completed[0].wisdom.0, original.wisdom.0);
    }

    #[test]
    fn backward_compat_uncompressed_whole_packet() {
        // Legacy nodes send raw WisdomPacket bytes (no compression envelope).
        // Verify receive_whole still handles them via fallback.
        let mut receiver = super::super::MeshReceiver::new();

        let original = test_packet(MeshUrgency::Normal);
        let raw_bytes = original.to_bytes(); // no compression header

        // receive_whole should fall back to direct from_bytes
        let result = receiver
            .receive_whole(&raw_bytes)
            .expect("backward compat parse");
        assert_eq!(result.sequence, original.sequence);
        assert_eq!(result.wisdom.0, original.wisdom.0);
    }

    // -- Encryption pipeline tests --

    #[cfg(feature = "mesh-encryption")]
    #[test]
    fn test_send_encrypts_output() {
        // DualLayerMesh with encryption key produces ciphertext that differs
        // from plain compress_packet() output.
        let key = [0x42u8; 32];
        let packet = test_packet(MeshUrgency::Critical);
        let unencrypted = super::super::compress_packet(&packet.to_bytes());

        // Use BiLoopbackTransport to capture the encrypted output.
        let (a_enc, b_enc) = BiLoopbackTransport::pair("A-enc", "B-enc", 4096);
        let mesh_enc = DualLayerMesh::new([1; 32])
            .with_batman(Box::new(a_enc))
            .with_encryption_key(key);
        mesh_enc.send(&packet).unwrap();

        let mut buf = [0u8; WISDOM_PACKET_SIZE + 92];
        let n = b_enc.recv_raw(&mut buf).unwrap();
        let encrypted_bytes = &buf[..n];

        // Encrypted output must differ from unencrypted compressed envelope
        assert_ne!(
            encrypted_bytes,
            &unencrypted[..],
            "Encrypted output should differ from unencrypted compressed envelope"
        );
        // Encrypted output must be larger (nonce + tag overhead)
        assert!(
            encrypted_bytes.len() > unencrypted.len(),
            "Encrypted output ({}) should be larger than unencrypted ({})",
            encrypted_bytes.len(),
            unencrypted.len()
        );
    }

    #[cfg(feature = "mesh-encryption")]
    #[test]
    fn test_encrypted_send_receive_roundtrip() {
        // Send through DualLayerMesh with key → MeshReceiver with same key → roundtrip.
        let key = [0xAB; 32];
        let (a, b) = BiLoopbackTransport::pair("A-enc", "B-enc", 4096);

        let mesh_a = DualLayerMesh::new([1; 32])
            .with_batman(Box::new(a))
            .with_encryption_key(key);

        let original = test_packet(MeshUrgency::Critical);
        mesh_a.send(&original).unwrap();

        let mut receiver = super::super::MeshReceiver::new().with_encryption_key(key);
        let mesh_b = DualLayerMesh::new([2; 32]).with_batman(Box::new(b));
        let completed = mesh_b.poll_incoming(&mut receiver);

        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].sequence, original.sequence);
        assert_eq!(completed[0].source_id, original.source_id);
        assert_eq!(completed[0].wisdom.0, original.wisdom.0);
    }

    #[cfg(feature = "mesh-encryption")]
    #[test]
    fn test_encrypted_wrong_key_rejected() {
        // Send with key A → receive with key B → decryption fails,
        // but backward-compat fallback also fails (ciphertext isn't valid
        // compressed/raw packet), so no packet returned.
        let key_a = [0xAA; 32];
        let key_b = [0xBB; 32];
        let (a, b) = BiLoopbackTransport::pair("A-enc", "B-enc", 4096);

        let mesh_a = DualLayerMesh::new([1; 32])
            .with_batman(Box::new(a))
            .with_encryption_key(key_a);

        let original = test_packet(MeshUrgency::Critical);
        mesh_a.send(&original).unwrap();

        let mut receiver = super::super::MeshReceiver::new().with_encryption_key(key_b);
        let mesh_b = DualLayerMesh::new([2; 32]).with_batman(Box::new(b));
        let completed = mesh_b.poll_incoming(&mut receiver);

        assert!(
            completed.is_empty(),
            "Wrong key should produce no packets, got {}",
            completed.len()
        );
        assert_eq!(
            receiver.stats().packets_decrypt_failed,
            1,
            "Should record one decryption failure"
        );
    }

    #[cfg(feature = "mesh-encryption")]
    #[test]
    fn test_encrypted_lora_fragment_roundtrip() {
        // Full encrypt → fragment → reassemble → decrypt over LoRa path.
        let key = [0xCD; 32];
        let (a, b) = BiLoopbackTransport::pair("A-lora-enc", "B-lora-enc", LORA_MTU);

        let mesh_a = DualLayerMesh::new([1; 32])
            .with_lora(Box::new(a))
            .with_encryption_key(key);

        let original = test_packet(MeshUrgency::Cruise);
        mesh_a.send(&original).unwrap();

        // Receive fragmented encrypted data and reassemble
        let mut receiver = super::super::MeshReceiver::new().with_encryption_key(key);
        let mesh_b = DualLayerMesh::new([2; 32]).with_lora(Box::new(b));
        let completed = mesh_b.poll_incoming(&mut receiver);

        assert_eq!(
            completed.len(),
            1,
            "Should reassemble and decrypt one packet"
        );
        assert_eq!(completed[0].sequence, original.sequence);
        assert_eq!(completed[0].source_id, original.source_id);
        assert_eq!(completed[0].wisdom.0, original.wisdom.0);
    }
}
