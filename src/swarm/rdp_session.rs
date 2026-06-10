// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! RDP Session Manager: PQC-authenticated, consciousness-gated session lifecycle.
//!
//! State machine:
//! ```text
//! Connecting → Handshaking → Authenticating → Active ⇄ Rekeying → Closing
//! ```
//!
//! ## Security Model
//!
//! 1. **PQC Handshake**: Ed25519 mutual auth + ML-KEM-768 key encapsulation
//!    → 32-byte ChaCha20-Poly1305 session key. Quantum-resistant.
//! 2. **Consciousness Gate**: Peer must present a ConsciousnessAttestation with
//!    phi ≥ `min_consciousness`. Stale certificates (>5 min) are rejected.
//! 3. **Re-attestation**: Every 60s, request fresh attestation. If peer phi
//!    drops below threshold → view-only mode (input forwarding paused).
//! 4. **Key Rotation**: Every 30 min, trigger async PQC re-handshake.
//!    Old key remains valid during grace period (~5s) to handle in-flight frames.
//!
//! ## Important: Async PQC Operations
//!
//! ML-KEM-768 encapsulation/decapsulation takes ~35μs — fast but must NOT
//! block the 50Hz capture tick. All PQC operations run via `tokio::spawn`
//! and communicate results via channels.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::rdp_protocol::{ControlMessage, RDP_PROTOCOL_VERSION, RdpFrame, RdpSessionConfig};

/// How often to request consciousness re-attestation (seconds).
const REATTESTATION_INTERVAL_SECS: u64 = 60;

/// How often to rotate session keys (seconds).
const KEY_ROTATION_INTERVAL_SECS: u64 = 1800; // 30 min

/// Grace period for old key after rotation (seconds).
const KEY_ROTATION_GRACE_SECS: u64 = 5;

/// Maximum sessions per manager.
const MAX_SESSIONS: usize = 8;

/// Session state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Waiting for transport connection.
    Connecting,
    /// PQC handshake in progress (Ed25519 + ML-KEM-768).
    Handshaking,
    /// Consciousness attestation exchange.
    Authenticating,
    /// Full bidirectional frame + input streaming.
    Active,
    /// View-only: frames stream but input forwarding is paused.
    /// Entered when peer's consciousness drops below threshold.
    ViewOnly,
    /// Async PQC re-handshake in progress (session remains active during this).
    Rekeying,
    /// Graceful shutdown initiated.
    Closing,
    /// Session terminated.
    Closed,
}

/// Permission level for the remote peer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SessionPermission {
    /// Can only view the screen (no input forwarding).
    ViewOnly,
    /// Can view and send input (full remote control).
    Interactive,
    /// Interactive + file transfer.
    Full,
}

/// A single RDP session with a remote peer.
pub struct RdpSession {
    /// Unique session identifier.
    pub session_id: String,
    /// Remote peer identifier (DID or device name).
    pub peer_id: String,
    /// Current state.
    pub state: SessionState,
    /// Negotiated configuration.
    pub config: RdpSessionConfig,
    /// Current permission level.
    pub permission: SessionPermission,
    /// Whether this side initiated the connection.
    pub is_initiator: bool,

    // ── Encryption ──────────────────────────────────────────────────
    /// Current session encryption key (32 bytes for ChaCha20-Poly1305).
    session_key: Option<[u8; 32]>,
    /// Previous session key (valid during grace period after rotation).
    prev_session_key: Option<[u8; 32]>,
    /// When the current key was established.
    key_established_at: Option<Instant>,
    /// When the previous key expires.
    prev_key_expires_at: Option<Instant>,
    /// Monotonic nonce counter for AEAD encryption.
    nonce_counter: u64,
    /// Per-session random source identifier (used by rdp_wire::seal for nonce construction).
    /// Generated once at session creation. Prevents cross-session nonce collisions
    /// even when two sessions happen to share the same key material (which should
    /// not occur, but defence in depth).
    source_id: [u8; 8],
    /// Per-session random epoch byte (prevents restart nonce reuse under the same key).
    epoch: u8,
    /// Sliding-window replay protection for inbound sealed envelopes (mutated
    /// by `open()` on each accepted message). Shared primitive — see
    /// `swarm::replay_window::ReplayWindow`.
    #[cfg(feature = "mesh-encryption")]
    replay_window: super::replay_window::ReplayWindow,

    // ── Consciousness ───────────────────────────────────────────────
    /// Last known peer consciousness level (phi).
    pub peer_phi: f32,
    /// Last attestation time.
    last_attestation: Option<Instant>,
    /// Whether re-attestation has been requested (waiting for response).
    attestation_pending: bool,

    // ── Frame tracking ──────────────────────────────────────────────
    /// Last frame ID sent.
    pub last_frame_sent: u64,
    /// Last frame ID acknowledged by peer.
    pub last_frame_acked: u64,
    /// Session start time.
    pub started_at: Instant,

    // ── Outbound message queue ──────────────────────────────────────
    outbound: Vec<RdpFrame>,
}

impl RdpSession {
    /// Create a new session in Connecting state.
    pub fn new(
        session_id: String,
        peer_id: String,
        config: RdpSessionConfig,
        is_initiator: bool,
    ) -> Self {
        // Generate per-session random source_id + epoch for AEAD nonce construction.
        // Using rand::random() avoids taking a Rng by value and matches the pattern
        // used elsewhere in swarm::mesh for epoch generation.
        let source_id: [u8; 8] = rand::random();
        let epoch: u8 = rand::random();

        Self {
            session_id,
            peer_id,
            state: SessionState::Connecting,
            config,
            permission: SessionPermission::ViewOnly, // Start view-only until authenticated
            is_initiator,
            session_key: None,
            prev_session_key: None,
            key_established_at: None,
            prev_key_expires_at: None,
            nonce_counter: 0,
            source_id,
            epoch,
            #[cfg(feature = "mesh-encryption")]
            replay_window: super::replay_window::ReplayWindow::new(),
            peer_phi: 0.0,
            last_attestation: None,
            attestation_pending: false,
            last_frame_sent: 0,
            last_frame_acked: 0,
            started_at: Instant::now(),
            outbound: Vec::new(),
        }
    }

    /// Advance to Handshaking state after transport connects.
    pub fn on_connected(&mut self) {
        if self.state == SessionState::Connecting {
            self.state = SessionState::Handshaking;
        }
    }

    /// Set the session key after PQC handshake completes.
    /// Advances from Handshaking to Authenticating.
    pub fn on_handshake_complete(&mut self, key: [u8; 32]) {
        if self.state == SessionState::Handshaking || self.state == SessionState::Rekeying {
            if self.state == SessionState::Rekeying {
                // Key rotation: save old key with grace period.
                self.prev_session_key = self.session_key;
                self.prev_key_expires_at =
                    Some(Instant::now() + Duration::from_secs(KEY_ROTATION_GRACE_SECS));
            }
            self.session_key = Some(key);
            self.key_established_at = Some(Instant::now());
            self.nonce_counter = 0;
            if self.state == SessionState::Handshaking {
                self.state = SessionState::Authenticating;
            } else {
                // Rekeying complete → back to Active.
                self.state = SessionState::Active;
            }
        }
    }

    /// Process a consciousness attestation from the peer.
    /// Advances from Authenticating to Active (or ViewOnly) on first attestation.
    pub fn on_attestation(&mut self, phi: f32, _state_hash: &[u8; 32], _signature: &[u8]) -> bool {
        self.peer_phi = phi;
        self.last_attestation = Some(Instant::now());
        self.attestation_pending = false;

        let passes_gate = phi >= self.config.min_consciousness;

        match self.state {
            SessionState::Authenticating => {
                if passes_gate {
                    self.state = SessionState::Active;
                    self.permission = SessionPermission::Interactive;
                    true
                } else {
                    // Below threshold → view-only (don't reject entirely).
                    self.state = SessionState::ViewOnly;
                    self.permission = SessionPermission::ViewOnly;
                    true
                }
            }
            SessionState::Active | SessionState::ViewOnly | SessionState::Rekeying => {
                // Re-attestation: update permission based on consciousness.
                if passes_gate {
                    if self.state == SessionState::ViewOnly {
                        self.state = SessionState::Active;
                    }
                    self.permission = SessionPermission::Interactive;
                } else {
                    if self.state == SessionState::Active {
                        self.state = SessionState::ViewOnly;
                    }
                    self.permission = SessionPermission::ViewOnly;
                }
                true
            }
            _ => false,
        }
    }

    /// Process a frame acknowledgment from the peer.
    pub fn on_frame_ack(&mut self, frame_id: u64) {
        if frame_id > self.last_frame_acked {
            self.last_frame_acked = frame_id;
        }
    }

    /// Queue an outbound frame.
    pub fn queue_frame(&mut self, frame: RdpFrame) {
        if let RdpFrame::Full(ref f) = frame {
            self.last_frame_sent = f.frame_id;
        } else if let RdpFrame::Delta(ref d) = frame {
            self.last_frame_sent = d.frame_id;
        }
        self.outbound.push(frame);
    }

    /// Drain outbound frames for transmission.
    pub fn drain_outbound(&mut self) -> Vec<RdpFrame> {
        std::mem::take(&mut self.outbound)
    }

    /// Check if input forwarding is allowed.
    pub fn can_forward_input(&self) -> bool {
        self.state == SessionState::Active && self.permission >= SessionPermission::Interactive
    }

    /// Check if the session is in a state where frames should be sent.
    pub fn should_send_frames(&self) -> bool {
        matches!(
            self.state,
            SessionState::Active | SessionState::ViewOnly | SessionState::Rekeying
        )
    }

    /// Get the current encryption key. Returns None if no key established.
    pub fn encryption_key(&self) -> Option<&[u8; 32]> {
        self.session_key.as_ref()
    }

    /// Allocate the next nonce for AEAD encryption.
    ///
    /// Uses `wrapping_add` to match the project convention (see
    /// `peer_registry.rs:242`, `rdp_codec.rs:318`, etc.) and to avoid a
    /// debug-build panic on overflow. In release builds the previous
    /// `+= 1` form silently wrapped, which is the same observable
    /// behavior as `wrapping_add`; the change makes the wrap explicit
    /// and prevents debug crashes.
    ///
    /// Note: u64 monotonic counter at 50 Hz wraps in ~5.8 billion years.
    /// At 30 fps RDP frame rate, the low 32 bits used in the AEAD nonce
    /// wrap in ~4.5 years of continuous operation. Real session lifetime
    /// is governed by `KEY_ROTATION_GRACE_SECS` rekey cadence (typically
    /// 30 minutes), so wraparound is not a security concern in practice.
    pub fn next_nonce(&mut self) -> u64 {
        let n = self.nonce_counter;
        self.nonce_counter = self.nonce_counter.wrapping_add(1);
        n
    }

    /// Seal a binary plaintext under the current session key using ChaCha20-Poly1305.
    ///
    /// Reuses the nonce-construction + AEAD path from `swarm::mesh::packet_crypto`
    /// so RDP wire frames share the same cryptographic hygiene (cross-type nonce
    /// separation via `payload_type`, per-session random `source_id` + `epoch`,
    /// monotonic sequence from `next_nonce()`).
    ///
    /// Wire format: `[ nonce (12 bytes) | ciphertext | poly1305 tag (16 bytes) ]`
    ///
    /// Returns `None` if the session has no key (handshake not complete) or
    /// the underlying AEAD call fails (should not happen with valid inputs).
    ///
    /// `payload_type` must differ between RdpFrame (`0x10`) and InputFrame (`0x11`)
    /// and any other stream type to prevent nonce collisions. See `rdp_wire` for
    /// the canonical assignments.
    #[cfg(feature = "mesh-encryption")]
    pub fn seal(&mut self, plaintext: &[u8], payload_type: u8) -> Option<Vec<u8>> {
        use chacha20poly1305::{ChaCha20Poly1305, KeyInit, Nonce, aead::Aead};
        let key = *self.session_key.as_ref()?;
        let seq = (self.next_nonce() & 0xFFFF_FFFF) as u32;

        // Nonce layout mirrors `swarm::mesh::packet_crypto::build_nonce` so
        // future reuse of those primitives remains straightforward:
        // `source_id[0..6] | payload_type | epoch | sequence[0..4]`.
        let mut nonce_bytes = [0u8; 12];
        nonce_bytes[..6].copy_from_slice(&self.source_id[..6]);
        nonce_bytes[6] = payload_type;
        nonce_bytes[7] = self.epoch;
        nonce_bytes[8..12].copy_from_slice(&seq.to_le_bytes());

        let cipher = ChaCha20Poly1305::new((&key).into());
        let nonce = Nonce::from(nonce_bytes);
        let ciphertext = cipher.encrypt(&nonce, plaintext).ok()?;

        let mut out = Vec::with_capacity(12 + ciphertext.len());
        out.extend_from_slice(&nonce_bytes);
        out.extend_from_slice(&ciphertext);
        Some(out)
    }

    /// Open a sealed binary envelope produced by `seal` (or by a peer's `seal` call).
    ///
    /// Performs three checks in order:
    ///
    /// 1. **Length**: envelope must be ≥ 28 bytes (12 nonce + 16 tag).
    /// 2. **AEAD verify**: ChaCha20-Poly1305 decrypt against the current
    ///    session key, falling back to the previous key during the
    ///    rekey grace period.
    /// 3. **Replay window**: the sequence number embedded in the nonce
    ///    (positions 8..12, little-endian u32) must be either strictly
    ///    higher than any previously accepted sequence for the same
    ///    `(source_id, payload_type)` stream, OR within the 64-message
    ///    sliding window AND not previously seen.
    ///
    /// Returns `None` on any failure. Mutates `self` to advance the replay
    /// window when a message is accepted.
    ///
    /// **Mutability note**: this used to be `&self` before the replay
    /// window was added in Phase I.A.5 Track 2.2. Callers must update to
    /// pass `&mut session`.
    #[cfg(feature = "mesh-encryption")]
    pub fn open(&mut self, envelope: &[u8]) -> Option<Vec<u8>> {
        use chacha20poly1305::{ChaCha20Poly1305, KeyInit, Nonce, aead::Aead};

        // (1) length check
        if envelope.len() < 12 + 16 {
            return None;
        }
        let (nonce_bytes, ciphertext) = envelope.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);

        // (2) AEAD verify with current key, then prev_session_key fallback.
        let plaintext = if let Some(key) = self.session_key.as_ref() {
            let cipher = ChaCha20Poly1305::new(key.into());
            if let Ok(pt) = cipher.decrypt(nonce, ciphertext) {
                Some(pt)
            } else if let Some(prev) = self.prev_session_key.as_ref() {
                let cipher = ChaCha20Poly1305::new(prev.into());
                cipher.decrypt(nonce, ciphertext).ok()
            } else {
                None
            }
        } else if let Some(prev) = self.prev_session_key.as_ref() {
            let cipher = ChaCha20Poly1305::new(prev.into());
            cipher.decrypt(nonce, ciphertext).ok()
        } else {
            None
        }?;

        // (3) Replay window: extract source_id (high 6 bytes of nonce
        // bytes 0..6, padded to u64) and payload_type (nonce byte 6) and
        // sequence (nonce bytes 8..12). Note: the seal path puts source_id
        // in nonce[0..6] (6 bytes), so we read those 6 bytes into the low
        // 48 bits of a u64 stream key. epoch (nonce[7]) is not part of
        // the replay window key — different epochs are different sessions
        // semantically, but a rekey within the same session does not
        // change source_id, so the window correctly persists across
        // rekey for the same logical stream.
        let mut source_id_u64 = 0u64;
        for (i, b) in nonce_bytes[..6].iter().enumerate() {
            source_id_u64 |= (*b as u64) << (i * 8);
        }
        let payload_type = nonce_bytes[6];
        let seq = u32::from_le_bytes([
            nonce_bytes[8],
            nonce_bytes[9],
            nonce_bytes[10],
            nonce_bytes[11],
        ]) as u64;

        if !self
            .replay_window
            .accept((source_id_u64, payload_type), seq)
        {
            // AEAD passed but the message is a replay or too old. Drop.
            return None;
        }

        Some(plaintext)
    }

    /// Initiate graceful close.
    pub fn close(&mut self, reason: &str) {
        self.state = SessionState::Closing;
        self.outbound
            .push(RdpFrame::Control(ControlMessage::Goodbye {
                reason: reason.to_string(),
            }));
    }

    /// Check if this session needs re-attestation.
    pub fn needs_reattestation(&self) -> bool {
        if self.attestation_pending {
            return false;
        }
        match self.last_attestation {
            Some(t) => t.elapsed() > Duration::from_secs(REATTESTATION_INTERVAL_SECS),
            None => false, // Initial attestation handled by state machine
        }
    }

    /// Check if this session needs key rotation.
    pub fn needs_key_rotation(&self) -> bool {
        if self.state == SessionState::Rekeying {
            return false; // Already rekeying
        }
        match self.key_established_at {
            Some(t) => t.elapsed() > Duration::from_secs(KEY_ROTATION_INTERVAL_SECS),
            None => false,
        }
    }

    /// Initiate key rotation (moves to Rekeying state).
    pub fn start_rekey(&mut self) {
        if self.state == SessionState::Active {
            self.state = SessionState::Rekeying;
        }
    }

    /// Request re-attestation from peer.
    pub fn request_attestation(&mut self) {
        self.attestation_pending = true;
        // The caller should send a ControlMessage requesting attestation.
    }

    /// Expire the previous key if grace period has passed.
    pub fn tick_key_expiry(&mut self) {
        if let Some(expires) = self.prev_key_expires_at {
            if Instant::now() > expires {
                self.prev_session_key = None;
                self.prev_key_expires_at = None;
            }
        }
    }

    /// Number of frames behind in acknowledgments (backpressure signal).
    pub fn frame_lag(&self) -> u64 {
        self.last_frame_sent.saturating_sub(self.last_frame_acked)
    }

    /// Session uptime.
    pub fn uptime(&self) -> Duration {
        self.started_at.elapsed()
    }
}

/// Manages multiple concurrent RDP sessions.
pub struct RdpSessionManager {
    sessions: HashMap<String, RdpSession>,
}

impl RdpSessionManager {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
        }
    }

    /// Create a new session. Returns the session_id.
    pub fn create_session(
        &mut self,
        peer_id: &str,
        config: RdpSessionConfig,
        is_initiator: bool,
    ) -> Option<String> {
        if self.sessions.len() >= MAX_SESSIONS {
            return None; // Cap reached
        }
        let session_id = format!(
            "rdp-{:016x}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64
        );
        let session = RdpSession::new(
            session_id.clone(),
            peer_id.to_string(),
            config,
            is_initiator,
        );
        self.sessions.insert(session_id.clone(), session);
        Some(session_id)
    }

    /// Get a session by ID.
    pub fn session(&self, session_id: &str) -> Option<&RdpSession> {
        self.sessions.get(session_id)
    }

    /// Get a mutable session by ID.
    pub fn session_mut(&mut self, session_id: &str) -> Option<&mut RdpSession> {
        self.sessions.get_mut(session_id)
    }

    /// Remove a closed session.
    pub fn remove_session(&mut self, session_id: &str) {
        self.sessions.remove(session_id);
    }

    /// Iterate over all active sessions.
    pub fn active_sessions(&self) -> impl Iterator<Item = &RdpSession> {
        self.sessions.values().filter(|s| s.should_send_frames())
    }

    /// Tick all sessions: check re-attestation, key rotation, key expiry.
    pub fn tick(&mut self) -> Vec<SessionAction> {
        let mut actions = Vec::new();

        for (id, session) in &mut self.sessions {
            // Expire old keys.
            session.tick_key_expiry();

            // Check re-attestation.
            if session.needs_reattestation() {
                session.request_attestation();
                actions.push(SessionAction::RequestAttestation {
                    session_id: id.clone(),
                });
            }

            // Check key rotation.
            if session.needs_key_rotation() {
                session.start_rekey();
                actions.push(SessionAction::StartRekey {
                    session_id: id.clone(),
                });
            }

            // Check backpressure.
            if session.frame_lag() > 5 {
                actions.push(SessionAction::Backpressure {
                    session_id: id.clone(),
                    lag: session.frame_lag(),
                });
            }
        }

        // Prune closed sessions.
        self.sessions.retain(|_, s| s.state != SessionState::Closed);

        actions
    }

    /// Number of active sessions.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Build a Hello message for initiating a connection.
    pub fn build_hello(client_name: &str) -> RdpFrame {
        RdpFrame::Control(ControlMessage::Hello {
            protocol_version: RDP_PROTOCOL_VERSION,
            client_name: client_name.to_string(),
        })
    }

    /// Build a Welcome response.
    pub fn build_welcome(
        server_name: &str,
        patch_cols: u16,
        patch_rows: u16,
        config: &RdpSessionConfig,
        consciousness_level: f32,
    ) -> RdpFrame {
        RdpFrame::Control(ControlMessage::Welcome {
            protocol_version: RDP_PROTOCOL_VERSION,
            server_name: server_name.to_string(),
            patch_cols,
            patch_rows,
            target_fps: config.target_fps,
            consciousness_level,
        })
    }
}

impl Default for RdpSessionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Actions the session manager requests from the host.
#[derive(Debug, Clone)]
pub enum SessionAction {
    /// Request consciousness re-attestation from peer.
    RequestAttestation { session_id: String },
    /// Start async PQC re-handshake for key rotation.
    StartRekey { session_id: String },
    /// Session has backpressure — consider reducing quality.
    Backpressure { session_id: String, lag: u64 },
}

/// Telemetry for monitoring session health.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionTelemetry {
    pub session_id: String,
    pub peer_id: String,
    pub state: String,
    pub permission: String,
    pub peer_phi: f32,
    pub frame_lag: u64,
    pub uptime_secs: u64,
    pub frames_sent: u64,
    pub frames_acked: u64,
    pub nonce_counter: u64,
}

impl RdpSession {
    /// Generate telemetry snapshot.
    pub fn telemetry(&self) -> SessionTelemetry {
        SessionTelemetry {
            session_id: self.session_id.clone(),
            peer_id: self.peer_id.clone(),
            state: format!("{:?}", self.state),
            permission: format!("{:?}", self.permission),
            peer_phi: self.peer_phi,
            frame_lag: self.frame_lag(),
            uptime_secs: self.uptime().as_secs(),
            frames_sent: self.last_frame_sent,
            frames_acked: self.last_frame_acked,
            nonce_counter: self.nonce_counter,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> RdpSessionConfig {
        RdpSessionConfig {
            min_consciousness: 0.1,
            ..RdpSessionConfig::default()
        }
    }

    #[test]
    fn test_session_state_machine_happy_path() {
        let mut s = RdpSession::new("s1".into(), "peer1".into(), default_config(), true);
        assert_eq!(s.state, SessionState::Connecting);

        s.on_connected();
        assert_eq!(s.state, SessionState::Handshaking);

        s.on_handshake_complete([42u8; 32]);
        assert_eq!(s.state, SessionState::Authenticating);
        assert!(s.encryption_key().is_some());

        let ok = s.on_attestation(0.5, &[0u8; 32], &[]);
        assert!(ok);
        assert_eq!(s.state, SessionState::Active);
        assert_eq!(s.permission, SessionPermission::Interactive);
        assert!(s.can_forward_input());
    }

    #[test]
    fn test_consciousness_gate_blocks_input() {
        let mut s = RdpSession::new("s1".into(), "peer1".into(), default_config(), true);
        s.on_connected();
        s.on_handshake_complete([42u8; 32]);

        // Attestation with phi below threshold (0.1).
        let ok = s.on_attestation(0.05, &[0u8; 32], &[]);
        assert!(ok);
        assert_eq!(s.state, SessionState::ViewOnly);
        assert!(!s.can_forward_input());
        assert!(s.should_send_frames()); // Still sends frames (view-only)
    }

    #[test]
    fn test_consciousness_recovery() {
        let mut s = RdpSession::new("s1".into(), "peer1".into(), default_config(), true);
        s.on_connected();
        s.on_handshake_complete([42u8; 32]);

        // Start below threshold.
        s.on_attestation(0.05, &[0u8; 32], &[]);
        assert_eq!(s.state, SessionState::ViewOnly);

        // Recover above threshold.
        s.on_attestation(0.3, &[0u8; 32], &[]);
        assert_eq!(s.state, SessionState::Active);
        assert!(s.can_forward_input());
    }

    #[test]
    fn test_key_rotation() {
        let mut s = RdpSession::new("s1".into(), "peer1".into(), default_config(), true);
        s.on_connected();
        s.on_handshake_complete([1u8; 32]);
        s.on_attestation(0.5, &[0u8; 32], &[]);
        assert_eq!(s.state, SessionState::Active);

        // Start rekey.
        s.start_rekey();
        assert_eq!(s.state, SessionState::Rekeying);
        assert!(s.should_send_frames()); // Still active during rekey

        // Complete rekey with new key.
        s.on_handshake_complete([2u8; 32]);
        assert_eq!(s.state, SessionState::Active);
        assert_eq!(s.encryption_key(), Some(&[2u8; 32]));
        // Old key still available during grace period.
        assert!(s.prev_session_key.is_some());
    }

    #[test]
    fn test_frame_lag_backpressure() {
        let mut s = RdpSession::new("s1".into(), "peer1".into(), default_config(), true);
        s.on_connected();
        s.on_handshake_complete([42u8; 32]);
        s.on_attestation(0.5, &[0u8; 32], &[]);

        s.last_frame_sent = 100;
        s.last_frame_acked = 90;
        assert_eq!(s.frame_lag(), 10);

        s.on_frame_ack(95);
        assert_eq!(s.frame_lag(), 5);
    }

    #[test]
    fn test_session_manager_cap() {
        let mut mgr = RdpSessionManager::new();
        for i in 0..MAX_SESSIONS {
            let id = mgr.create_session(&format!("peer{}", i), default_config(), true);
            assert!(id.is_some());
        }
        // Cap reached.
        let id = mgr.create_session("peer_overflow", default_config(), true);
        assert!(id.is_none());
    }

    #[test]
    fn test_nonce_counter_monotonic() {
        let mut s = RdpSession::new("s1".into(), "peer1".into(), default_config(), true);
        assert_eq!(s.next_nonce(), 0);
        assert_eq!(s.next_nonce(), 1);
        assert_eq!(s.next_nonce(), 2);
    }

    #[test]
    fn test_nonce_counter_wraparound() {
        // Verify wrapping_add behavior at u64::MAX. This used to be a
        // debug-build panic before the wrapping_add fix.
        let mut s = RdpSession::new("s1".into(), "peer1".into(), default_config(), true);
        // Manually seed the counter just below the boundary.
        s.nonce_counter = u64::MAX;
        assert_eq!(s.next_nonce(), u64::MAX);
        assert_eq!(s.next_nonce(), 0); // wrapped
        assert_eq!(s.next_nonce(), 1);
    }

    #[test]
    fn test_close_sends_goodbye() {
        let mut s = RdpSession::new("s1".into(), "peer1".into(), default_config(), true);
        s.on_connected();
        s.on_handshake_complete([42u8; 32]);
        s.on_attestation(0.5, &[0u8; 32], &[]);

        s.close("user requested");
        assert_eq!(s.state, SessionState::Closing);

        let outbound = s.drain_outbound();
        assert_eq!(outbound.len(), 1);
        match &outbound[0] {
            RdpFrame::Control(ControlMessage::Goodbye { reason }) => {
                assert!(reason.contains("user requested"));
            }
            _ => panic!("Expected Goodbye"),
        }
    }

    #[test]
    fn test_telemetry_snapshot() {
        let mut s = RdpSession::new("s1".into(), "peer1".into(), default_config(), true);
        s.on_connected();
        s.on_handshake_complete([42u8; 32]);
        s.on_attestation(0.42, &[0u8; 32], &[]);

        let t = s.telemetry();
        assert_eq!(t.session_id, "s1");
        assert_eq!(t.peer_id, "peer1");
        assert_eq!(t.state, "Active");
        assert!((t.peer_phi - 0.42).abs() < 0.01);
    }
}
