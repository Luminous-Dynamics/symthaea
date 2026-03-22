// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mesh network bridge interface for the Continuous Mind.

use super::ContinuousMind;

/// Size of the packet deduplication ring buffer (source_id + sequence pairs).
#[cfg(feature = "mesh")]
pub(super) const MESH_DEDUP_RING_SIZE: usize = 512;

/// Maximum age of a mesh packet (seconds) before it's rejected as stale.
#[cfg(feature = "mesh")]
pub(super) const MESH_MAX_PACKET_AGE_S: u32 = 60;

/// Duration of the bandwidth budget window.
#[cfg(feature = "mesh")]
pub(super) const MESH_BANDWIDTH_WINDOW: std::time::Duration = std::time::Duration::from_secs(10);

/// Initial (and reset) bandwidth budget: 100 KB per window.
#[cfg(feature = "mesh")]
pub(super) const MESH_BANDWIDTH_INITIAL: u64 = 100 * 1024;

/// Floor for AIMD multiplicative decrease: never below 25 KB.
#[cfg(feature = "mesh")]
pub(super) const MESH_BANDWIDTH_MIN: u64 = 25 * 1024;

/// Ceiling for AIMD additive increase: never above 200 KB.
#[cfg(feature = "mesh")]
pub(super) const MESH_BANDWIDTH_MAX: u64 = 200 * 1024;

/// Additive increase per window when mesh is healthy and not throttled.
#[cfg(feature = "mesh")]
pub(super) const MESH_BANDWIDTH_ADDITIVE_INCREASE: u64 = 10 * 1024;

/// Multiplicative decrease factor on throttle or low health.
#[cfg(feature = "mesh")]
pub(super) const MESH_BANDWIDTH_DECREASE_FACTOR: f64 = 0.5;

/// Capacity of the replay buffer for partition recovery.
#[cfg(feature = "mesh")]
pub(super) const MESH_REPLAY_BUFFER_CAPACITY: usize = 16;

#[cfg(feature = "mesh")]
impl ContinuousMind {
    /// Attach a mesh network bridge handle for physical radio consciousness exchange.
    ///
    /// Once attached, each `tick()` will automatically:
    /// 1. Flush `mesh_outbox` packets to the radio network via the bridge
    /// 2. Drain inbound wisdom packets into `mesh_inbox`
    ///
    /// The bridge actor must be spawned separately on a tokio runtime.
    pub fn set_mesh_bridge(&mut self, handle: crate::swarm::mesh::MeshBridgeHandle) {
        self.mesh_bridge = Some(handle);
    }

    /// Check if a mesh network bridge is attached and alive.
    pub fn has_mesh_bridge(&self) -> bool {
        self.mesh_bridge.as_ref().is_some_and(|h| h.is_alive())
    }

    /// Get a reference to the mesh peer registry.
    pub fn mesh_peers(&self) -> &crate::swarm::mesh::MeshPeerRegistry {
        &self.mesh_peers
    }

    /// Get a reference to the mesh telemetry counters.
    pub fn mesh_stats(&self) -> &crate::swarm::mesh::MeshStats {
        &self.mesh_stats
    }

    /// Build a structured mesh telemetry snapshot from current state.
    pub fn mesh_telemetry(&self) -> crate::swarm::mesh::MeshTelemetry {
        let peer_count = self.mesh_peers.peer_count();
        let avg_phi = self.mesh_peers.average_phi();
        let health_score = self.mesh_stats.health_score(peer_count);
        crate::swarm::mesh::MeshTelemetry {
            stats: self.mesh_stats.clone(),
            peer_count,
            avg_phi,
            health_score,
            moral_topology: self.cached_moral_topology.clone(),
        }
    }

    /// Cache a moral topology summary for inclusion in mesh telemetry gossip.
    pub fn set_cached_moral_topology(
        &mut self,
        summary: crate::hdc::moral_topology::MoralTopologySummary,
    ) {
        self.cached_moral_topology = Some(summary);
    }

    /// Attach a Hyperfeel engine for affective mesh payload processing.
    pub fn set_hyperfeel(&mut self, hf: crate::swarm::Hyperfeel) {
        self.hyperfeel = Some(hf);
    }

    /// Attach a sensor registry for physical environmental inputs.
    pub fn set_sensor_registry(&mut self, registry: crate::swarm::mesh::SensorRegistry) {
        self.sensor_registry = Some(registry);
    }

    /// Populate mesh telemetry fields in a CycleMetadata struct.
    pub fn populate_mesh_metadata(
        &self,
        metadata: &mut crate::cognitive_loop::types::CycleMetadata,
    ) {
        let peer_count = self.mesh_peers.peer_count();
        metadata.mesh.mesh_health_score = self.mesh_stats.health_score(peer_count);
        metadata.mesh.mesh_peer_count = peer_count as u32;
        metadata.mesh.mesh_bytes_sent = self.mesh_stats.bytes_sent;
        metadata.mesh.mesh_bytes_received = self.mesh_stats.bytes_received;
        metadata.mesh.mesh_compression_ratio = self.mesh_stats.compression_ratio();
        metadata.mesh.mesh_bandwidth_budget = self.mesh_bandwidth_budget;
        metadata.mesh.mesh_packets_throttled = self.mesh_stats.bandwidth_throttled;
        metadata.mesh.mesh_packets_decrypt_failed = self.mesh_stats.packets_decrypt_failed;
    }

    /// Check if the bandwidth budget allows sending `packet_bytes`.
    ///
    /// Returns `true` if the budget allows it (and deducts the bytes),
    /// `false` if throttled (increments `bandwidth_throttled` stat).
    pub(crate) fn mesh_bandwidth_check(&mut self, packet_bytes: u64) -> bool {
        let now = std::time::Instant::now();
        if now.duration_since(self.mesh_bandwidth_window_start) >= MESH_BANDWIDTH_WINDOW {
            // Window expired — adjust budget before resetting
            self.adjust_bandwidth_budget();
            self.mesh_bandwidth_window_start = now;
            self.mesh_bandwidth_window_bytes = 0;
            self.mesh_bandwidth_throttled_in_window = false;
        }
        if self.mesh_bandwidth_window_bytes + packet_bytes > self.mesh_bandwidth_budget {
            self.mesh_stats.bandwidth_throttled += 1;
            self.mesh_bandwidth_throttled_in_window = true;
            false
        } else {
            self.mesh_bandwidth_window_bytes += packet_bytes;
            true
        }
    }

    /// Register a single sensor (creates registry if needed).
    pub fn register_sensor(&mut self, sensor: Box<dyn crate::swarm::mesh::SensorInput>) {
        if self.sensor_registry.is_none() {
            self.sensor_registry = Some(crate::swarm::mesh::SensorRegistry::new(
                self.config.dimension,
            ));
        }
        if let Some(reg) = self.sensor_registry.as_mut() {
            reg.register(sensor);
        }
    }

    /// Emit a wisdom vector over the mesh network, gated by urgency.
    ///
    /// Emission frequency is controlled by the cognitive urgency level:
    /// - `Critical`: every call (~50Hz over B.A.T.M.A.N.)
    /// - `Normal`: ~once per second (~every 50 calls over Yggdrasil)
    /// - `Cruise`: ~once per 10 seconds (~every 500 calls over LoRa)
    ///
    /// If not enough ticks have elapsed since the last emission, this is a no-op.
    pub(crate) fn emit_wisdom(
        &mut self,
        wisdom_hv: symthaea_core::hdc::BinaryHV,
        urgency: crate::cognitive_loop::types::CycleUrgency,
        phi: f32,
    ) {
        use crate::cognitive_loop::types::CycleUrgency;
        use crate::swarm::mesh::{MeshOutbound, MeshUrgency, PayloadType, WisdomPacket};

        // Emission rate gating based on urgency
        let interval = match urgency {
            CycleUrgency::Critical => 1, // every cycle (~50Hz)
            CycleUrgency::Normal => 50,  // ~1/s at 50Hz
            CycleUrgency::Cruise => 500, // ~10s at 50Hz
        };

        let ticks_since = self.state.tick.saturating_sub(self.mesh_last_emit_tick);
        // Always allow the first emission (sequence == 0); thereafter gate by interval
        if ticks_since < interval && self.mesh_sequence > 0 {
            return;
        }

        // Bandwidth budget check
        if !self.mesh_bandwidth_check(crate::swarm::mesh::WISDOM_PACKET_SIZE as u64) {
            return;
        }

        self.mesh_last_emit_tick = self.state.tick;

        // Construct source_id from config dimension as a stand-in node identity
        // (real impl would use swarm node_id)
        let mut source_id = [0u8; 8];
        let dim_bytes = (self.config.dimension as u64).to_le_bytes();
        source_id.copy_from_slice(&dim_bytes);

        let timestamp_s = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as u32;

        let mut packet = WisdomPacket {
            source_id,
            sequence: self.mesh_sequence,
            phi,
            urgency: MeshUrgency::from(urgency),
            timestamp_s,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: crate::swarm::mesh::MESH_DEFAULT_TTL,
            wisdom: wisdom_hv,
        };

        self.sign_mesh_packet(&mut packet);

        // Push to replay buffer for partition recovery
        if self.mesh_replay_buffer.len() >= MESH_REPLAY_BUFFER_CAPACITY {
            self.mesh_replay_buffer.pop_front();
        }
        self.mesh_replay_buffer.push_back(packet.clone());

        self.mesh_stats.bytes_before_compression += crate::swarm::mesh::WISDOM_PACKET_SIZE as u64;
        self.mesh_stats.bytes_after_compression +=
            crate::swarm::mesh::compress_packet(&packet.to_bytes()).len() as u64;

        self.mesh_sequence = self.mesh_sequence.wrapping_add(1);
        self.mesh_outbox.push(MeshOutbound { packet });
        self.mesh_stats.wisdom_sent += 1;
        self.mesh_stats.bytes_sent += crate::swarm::mesh::WISDOM_PACKET_SIZE as u64;
        #[cfg(feature = "mesh-encryption")]
        if self.mesh_encryption_key.is_some() {
            self.mesh_stats.encrypted_packets_sent += 1;
        }

        tracing::trace!(
            target: "symthaea::mind::mesh",
            sequence = self.mesh_sequence.wrapping_sub(1),
            urgency = ?urgency,
            phi,
            "Emitted wisdom packet"
        );
    }

    /// Get the mesh source_id for this mind (first 8 bytes of node identity).
    pub(crate) fn mesh_source_id(&self) -> [u8; 8] {
        let mut source_id = [0u8; 8];
        let dim_bytes = (self.config.dimension as u64).to_le_bytes();
        source_id.copy_from_slice(&dim_bytes);
        source_id
    }

    /// Set the BLAKE3 key for packet authentication.
    ///
    /// When set, all emitted packets are signed with this key, and
    /// all received packets are verified (unsigned packets are rejected).
    /// Pass `None` to disable authentication (backward-compatible default).
    pub fn set_mesh_auth_key(&mut self, key: Option<[u8; 32]>) {
        self.mesh_auth_key = key;
    }

    /// Sign a WisdomPacket with the mesh auth key (if set).
    ///
    /// Computes a BLAKE3 keyed MAC over the serialized packet bytes
    /// and sets the `auth_mac` field. No-op if no key is configured.
    pub(crate) fn sign_mesh_packet(&self, packet: &mut crate::swarm::mesh::WisdomPacket) {
        if let Some(ref key) = self.mesh_auth_key {
            let bytes = packet.to_bytes();
            packet.auth_mac = crate::swarm::mesh::compute_packet_mac(&bytes, key);
        }
    }

    /// Adjust the dynamic bandwidth budget using AIMD after each window reset.
    ///
    /// - **Additive Increase**: If mesh is healthy (health > 0.5) and no throttle
    ///   occurred this window, increase budget by 10 KB (capped at 200 KB).
    /// - **Multiplicative Decrease**: If throttled or health < 0.3, halve the
    ///   budget (floored at 25 KB).
    /// - **Hold Steady**: health in [0.3, 0.5] or 0.0 (idle) -- no change.
    pub(crate) fn adjust_bandwidth_budget(&mut self) {
        let health = self.mesh_stats.health_score(self.mesh_peers.peer_count());
        if self.mesh_bandwidth_throttled_in_window || (health > 0.0 && health < 0.3) {
            // Multiplicative decrease
            let new = (self.mesh_bandwidth_budget as f64 * MESH_BANDWIDTH_DECREASE_FACTOR) as u64;
            self.mesh_bandwidth_budget = new.max(MESH_BANDWIDTH_MIN);
            self.mesh_stats.bandwidth_decreases += 1;
        } else if health > 0.5 {
            // Additive increase
            self.mesh_bandwidth_budget = (self.mesh_bandwidth_budget
                + MESH_BANDWIDTH_ADDITIVE_INCREASE)
                .min(MESH_BANDWIDTH_MAX);
            self.mesh_stats.bandwidth_increases += 1;
        }
        // health in [0.3, 0.5] or 0.0 (idle): hold steady
        self.mesh_stats.bandwidth_budget_current = self.mesh_bandwidth_budget;
    }

    /// Emit a moral topology gossip packet (every ~100 ticks).
    ///
    /// Shares this agent's moral topology summary with mesh peers,
    /// enabling cross-agent moral drift detection.
    pub(crate) fn emit_moral_topology(&mut self) {
        if self.mesh_bridge.is_none() {
            return;
        }

        let interval = 100u64;
        if self
            .state
            .tick
            .saturating_sub(self.mesh_moral_topology_last_tick)
            < interval
            && self.mesh_stats.moral_topology_sent > 0
        {
            return;
        }

        let summary = match &self.cached_moral_topology {
            Some(s) if s.scenario_count > 0 => s.clone(),
            _ => return, // Nothing to share yet
        };

        // Bandwidth budget check
        if !self.mesh_bandwidth_check(crate::swarm::mesh::WISDOM_PACKET_SIZE as u64) {
            return;
        }

        let source_id = self.mesh_source_id();

        let mut packet = crate::swarm::mesh::WisdomPacket::from_moral_topology(
            source_id,
            self.mesh_sequence,
            self.state.consciousness_level as f32,
            &summary,
        );

        self.sign_mesh_packet(&mut packet);

        self.mesh_stats.bytes_before_compression += crate::swarm::mesh::WISDOM_PACKET_SIZE as u64;
        self.mesh_stats.bytes_after_compression +=
            crate::swarm::mesh::compress_packet(&packet.to_bytes()).len() as u64;

        self.mesh_sequence = self.mesh_sequence.wrapping_add(1);
        self.mesh_outbox
            .push(crate::swarm::mesh::MeshOutbound { packet });
        self.mesh_stats.moral_topology_sent += 1;
        self.mesh_stats.bytes_sent += crate::swarm::mesh::WISDOM_PACKET_SIZE as u64;
        self.mesh_moral_topology_last_tick = self.state.tick;
        #[cfg(feature = "mesh-encryption")]
        if self.mesh_encryption_key.is_some() {
            self.mesh_stats.encrypted_packets_sent += 1;
        }

        tracing::trace!(
            target: "symthaea::mind::mesh",
            sequence = self.mesh_sequence.wrapping_sub(1),
            "Emitted moral topology packet"
        );
    }

    /// Emit a lightweight heartbeat packet over the mesh network.
    ///
    /// Heartbeats fire every 100 ticks (~2s at 50Hz), keeping the mind
    /// visible in the mesh peer registry even when wisdom emissions are
    /// throttled by low urgency. The heartbeat carries the current phi
    /// but no wisdom vector (zero BinaryHV).
    pub(crate) fn emit_heartbeat(&mut self) {
        use crate::swarm::mesh::{MeshOutbound, MeshUrgency, PayloadType, WisdomPacket};

        if self.mesh_bridge.is_none() {
            return;
        }

        let interval = 100u64;
        if self
            .state
            .tick
            .saturating_sub(self.mesh_heartbeat_last_tick)
            < interval
            && self.mesh_heartbeat_sequence > 0
        {
            return;
        }

        // Bandwidth budget check
        if !self.mesh_bandwidth_check(crate::swarm::mesh::WISDOM_PACKET_SIZE as u64) {
            return;
        }

        let source_id = self.mesh_source_id();

        let timestamp_s = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as u32;

        let mut packet = WisdomPacket {
            source_id,
            sequence: self.mesh_heartbeat_sequence,
            phi: self.state.consciousness_level as f32,
            urgency: MeshUrgency::Cruise,
            timestamp_s,
            payload_type: PayloadType::Heartbeat,
            auth_mac: 0,
            ttl: crate::swarm::mesh::MESH_DEFAULT_TTL,
            wisdom: symthaea_core::hdc::BinaryHV::zero(),
        };

        self.sign_mesh_packet(&mut packet);

        self.mesh_stats.bytes_before_compression += crate::swarm::mesh::WISDOM_PACKET_SIZE as u64;
        self.mesh_stats.bytes_after_compression +=
            crate::swarm::mesh::compress_packet(&packet.to_bytes()).len() as u64;

        self.mesh_outbox.push(MeshOutbound { packet });
        self.mesh_heartbeat_last_tick = self.state.tick;
        self.mesh_heartbeat_sequence = self.mesh_heartbeat_sequence.wrapping_add(1);
        self.mesh_stats.heartbeats_sent += 1;
        self.mesh_stats.bytes_sent += crate::swarm::mesh::WISDOM_PACKET_SIZE as u64;
        #[cfg(feature = "mesh-encryption")]
        if self.mesh_encryption_key.is_some() {
            self.mesh_stats.encrypted_packets_sent += 1;
        }

        tracing::trace!(
            target: "symthaea::mind::mesh",
            sequence = self.mesh_heartbeat_sequence.wrapping_sub(1),
            phi = self.state.consciousness_level as f32,
            "Emitted heartbeat packet"
        );
    }

    /// Set the ChaCha20-Poly1305 encryption key for mesh packets.
    ///
    /// When set, all outbound packets are encrypted after compression,
    /// and all inbound packets are decrypted before decompression.
    /// Pass `None` to disable encryption.
    ///
    /// The key is propagated to the bridge actor (if attached) so that
    /// `DualLayerMesh` and `MeshReceiver` pick it up on the next poll cycle.
    #[cfg(feature = "mesh-encryption")]
    pub fn set_mesh_encryption_key(&mut self, key: Option<[u8; 32]>) {
        self.mesh_encryption_key = key.map(crate::swarm::mesh::RotatingKeyPair::new);
        self.propagate_encryption_key();
    }

    /// Rotate the mesh encryption key with a grace period.
    ///
    /// During the grace period, both old and new keys are accepted for
    /// decryption. After `grace_ticks` cycles, the old key is discarded.
    /// The bridge actor is updated with the new key immediately.
    #[cfg(feature = "mesh-encryption")]
    pub fn rotate_mesh_key(&mut self, new_key: [u8; 32], grace_ticks: u64) {
        if let Some(ref mut pair) = self.mesh_encryption_key {
            pair.rotate(new_key, self.state.tick, grace_ticks);
        } else {
            // No previous key — just set the new one
            self.mesh_encryption_key = Some(crate::swarm::mesh::RotatingKeyPair::new(new_key));
        }
        self.propagate_encryption_key();
    }

    /// Propagate the current encryption key to the bridge actor.
    ///
    /// Uses `key_version()` as the epoch to ensure nonce uniqueness
    /// across key rotations.
    #[cfg(feature = "mesh-encryption")]
    fn propagate_encryption_key(&self) {
        if let Some(ref handle) = self.mesh_bridge {
            let (key, epoch) = self
                .mesh_encryption_key
                .as_ref()
                .map(|p| (Some(*p.current_key()), p.key_version()))
                .unwrap_or((None, self.mesh_encryption_epoch));
            handle.set_encryption_key(key);
            handle.set_encryption_epoch(epoch);
        }
    }

    /// Set the automatic key rotation interval (in ticks).
    ///
    /// When set to a value > 0, the mesh encryption key is automatically
    /// rotated every `ticks` cycles with a grace period of interval/4.
    /// Set to 0 to disable automatic rotation.
    #[cfg(feature = "mesh-encryption")]
    pub fn set_mesh_auto_rotate_interval(&mut self, ticks: u64) {
        self.mesh_auto_rotate_interval = ticks;
    }

    /// Initialize the per-peer X25519 key store.
    ///
    /// `secret_bytes` is the node's long-term X25519 secret key (32 bytes).
    /// Returns our public key (send to peers for key agreement).
    #[cfg(feature = "mesh-key-exchange")]
    pub fn init_peer_key_store(&mut self, secret_bytes: [u8; 32]) -> [u8; 32] {
        let store = crate::swarm::mesh::PeerKeyStore::new(secret_bytes);
        let public = store.public_key();
        self.mesh_peer_keys = Some(store);
        public
    }

    /// Perform X25519 key agreement with a peer.
    ///
    /// Derives a per-peer ChaCha20 symmetric key from the DH shared secret.
    /// Returns the derived key (also stored internally for lookup by source_id).
    #[cfg(feature = "mesh-key-exchange")]
    pub fn agree_with_peer(
        &mut self,
        peer_source_id: [u8; 8],
        peer_public: &[u8; 32],
    ) -> Option<[u8; 32]> {
        self.mesh_peer_keys
            .as_mut()
            .map(|store| store.agree(peer_source_id, peer_public))
    }

    /// Look up the derived symmetric key for a peer.
    #[cfg(feature = "mesh-key-exchange")]
    pub fn peer_encryption_key(&self, source_id: &[u8; 8]) -> Option<&[u8; 32]> {
        self.mesh_peer_keys
            .as_ref()
            .and_then(|store| store.peer_key(source_id))
    }

    /// Remove a peer's derived key (e.g., on expiry/disconnect).
    #[cfg(feature = "mesh-key-exchange")]
    pub fn remove_peer_key(&mut self, source_id: &[u8; 8]) {
        if let Some(ref mut store) = self.mesh_peer_keys {
            store.remove_peer(source_id);
        }
    }
}
