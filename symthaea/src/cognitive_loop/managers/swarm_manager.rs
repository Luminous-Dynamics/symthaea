//! # Swarm Manager — Distributed Peer Consciousness Integration
//!
//! Consolidates peer-to-peer swarm signals into a single [`CognitiveSubsystem`]
//! that reads from an immutable [`CycleSnapshot`] and produces [`SubsystemOutput`].
//!
//! ## Signals Modeled
//!
//! 1. **Peer connectivity**: Network density → confidence (social buffering, Heinrichs 2003)
//! 2. **Affective contagion**: Peer emotional state → valence/arousal shift (Hatfield 1993)
//! 3. **Collective Φ**: Peer Φ distribution → learning rate modulation
//! 4. **Network anomaly**: Sudden peer loss → exploration + arousal spike (alarm signal)
//! 5. **Federated gradient**: Trust-weighted gradient quality → confidence
//!
//! ## Design
//!
//! The manager maintains internal state (peer EMAs, connectivity history) but does NOT
//! mutate any CognitiveLoopService fields directly. It proposes changes via
//! `SubsystemOutput` which the `OutputCollector` integrates via consensus averaging.

use super::super::subsystem_trait::{
    output_flags, CognitiveSubsystem, CycleSnapshot, SubsystemOutput,
};
use super::super::thresholds;
use std::collections::VecDeque;

// ═══════════════════════════════════════════════════════════════════════════════
// SWARM EVENTS — injected from external swarm layer
// ═══════════════════════════════════════════════════════════════════════════════

/// Events that the swarm layer injects into the manager.
#[derive(Debug, Clone)]
pub enum SwarmEvent {
    /// A peer connected (trust-verified via Holochain cortex).
    PeerJoined { peer_id: String, trust_level: f64 },
    /// A peer disconnected or timed out.
    PeerLeft { peer_id: String },
    /// Received a consciousness vector from a peer.
    ConsciousnessUpdate {
        peer_id: String,
        phi: f64,
        valence: f64,
        arousal: f64,
    },
    /// Affective sync broadcast from a peer.
    AffectiveSync {
        peer_id: String,
        valence: f64,
        arousal: f64,
        intensity: f64,
    },
    /// Federated gradient round completed.
    FederatedRound {
        /// Number of contributing nodes.
        n_contributors: usize,
        /// Average gradient quality [0, 1].
        avg_quality: f64,
        /// Trust-weighted aggregation confidence.
        trust_confidence: f64,
    },
    /// Network topology changed (detected by Iroh or Holochain).
    TopologyChange {
        /// Connected peer count after the change.
        connected_peers: usize,
        /// Whether this was a mass disconnection (>30% loss).
        mass_disconnect: bool,
    },
    /// Shared knowledge facts from a peer.
    ///
    /// Science: Woolley et al. (2010) — collective intelligence through shared epistemics.
    KnowledgeShare {
        /// Peer who shared the knowledge.
        peer_id: String,
        /// Fact texts with their confidence scores.
        facts: Vec<(String, f32)>,
        /// Number of peers who corroborated these facts.
        corroboration_count: u32,
    },
    /// A peer completed the trust handshake (Ed25519 or BLAKE3 verification).
    TrustVerified {
        /// Peer's node ID.
        peer_id: String,
        /// Trust level after verification [0, 1].
        trust_level: f64,
        /// Peer's agent public key (hex-encoded).
        agent_pubkey: String,
    },

    /// A threat pattern shared by a peer sentinel.
    ///
    /// Basis: Collective immune memory — distributed threat recognition.
    /// Acceptance gated by peer reputation (> 0.5 required).
    ThreatPattern {
        /// Peer who detected the threat.
        peer_id: String,
        /// Type of threat detected (serialized ThreatSignalKind).
        threat_kind: String,
        /// Severity of the threat (0.0–1.0).
        severity: f32,
        /// Confidence in the detection (0.0–1.0).
        confidence: f32,
        /// Compact feature vector (32 f32 values).
        feature_vector: Vec<f32>,
        /// Human-readable description.
        description: String,
    },

    /// A time beacon received from a mesh peer (Sovereign Clock).
    TimeBeaconReceived {
        /// Source peer ID (8-byte prefix).
        source_id: [u8; 8],
        /// Beacon timestamp (µs since epoch).
        timestamp_us: u64,
        /// Peer's stratum level.
        stratum: u8,
        /// Peer's Phi at beacon time.
        phi: f32,
        /// Beacon drift estimate (ppm).
        drift_ppm: f32,
    },

    /// Content announced by a mesh peer (Sovereign Social Fabric).
    ContentAnnounced {
        /// Source peer hex ID.
        peer_id: String,
        /// BLAKE3 content hash.
        content_hash: [u8; 32],
        /// Truncated 256-bit HDV embedding.
        truncated_hdv: [u8; 32],
        /// Content domain.
        domain: String,
        /// Creation timestamp (Unix seconds).
        created_at: u64,
    },
}

/// A threat pattern report from a peer, ready for SentinelManager consumption.
#[derive(Debug, Clone)]
pub struct PeerThreatReport {
    /// Peer who reported the threat.
    pub peer_id: String,
    /// Serialized threat kind.
    pub threat_kind: String,
    /// Severity (0.0–1.0).
    pub severity: f32,
    /// Confidence (0.0–1.0).
    pub confidence: f32,
    /// Compact feature vector (32 f32 values).
    pub feature_vector: Vec<f32>,
    /// Human-readable description.
    pub description: String,
}

/// Swarm telemetry snapshot for CycleMetadata.
#[derive(Debug, Clone, Default)]
pub struct SwarmTelemetry {
    /// Current number of connected peers.
    pub connected_peers: usize,
    /// Connectivity EMA [0, 1] — ratio of connected/expected peers.
    pub connectivity_ema: f64,
    /// Average peer Φ across connected peers.
    pub mean_peer_phi: f64,
    /// Affective contagion strength this cycle.
    pub affective_contagion: f64,
    /// Federated learning trust confidence.
    pub federated_confidence: f64,
    /// Number of anomaly events since last telemetry.
    pub anomaly_count: u32,
    /// Number of peers that completed trust handshake verification.
    pub verified_peers: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// SWARM MANAGER
// ═══════════════════════════════════════════════════════════════════════════════

/// Swarm Manager — peer consciousness signals → cognitive modulation.
///
/// Implements `CognitiveSubsystem` at interval 41 (co-prime with 7, 11, 13, 19, 29, 37).
pub struct SwarmManager {
    // ── Event queue ─────────────────────────────────────────────────────
    /// Pending events from the swarm layer (drained each process cycle).
    pending_events: Vec<SwarmEvent>,

    // ── Peer state ──────────────────────────────────────────────────────
    /// Number of currently connected (trust-verified) peers.
    connected_peers: usize,
    /// Expected peer count for connectivity ratio (calibrated externally).
    expected_peers: usize,
    /// Connectivity EMA — ratio of connected / expected.
    connectivity_ema: f64,
    /// Per-peer last known Φ. Bounded at MAX_TRACKED_PEERS.
    peer_phi: Vec<(String, f64)>,

    // ── Affective contagion ─────────────────────────────────────────────
    /// Accumulated valence shift from peer affective sync.
    affective_valence_acc: f64,
    /// Accumulated arousal shift from peer affective sync.
    affective_arousal_acc: f64,
    /// Number of affective sync events this interval.
    affective_count: u32,

    // ── Federated learning ──────────────────────────────────────────────
    /// Latest federated round trust confidence.
    federated_confidence: f64,
    /// Number of contributing nodes in last round.
    federated_contributors: usize,

    // ── Anomaly detection ───────────────────────────────────────────────
    /// Consecutive cycles with mass disconnection.
    anomaly_streak: u32,
    /// Connectivity history for delta detection (last 8 values).
    connectivity_history: VecDeque<f64>,

    // ── External modifiers ────────────────────────────────────────────
    /// Connectivity modifier from SpectrumManager (mesh feature).
    /// Scales connectivity_ema to account for radio tier degradation.
    /// 1.0 = all tiers up, 0.0 = radio blackout.
    connectivity_modifier: f64,

    // ── Knowledge sharing ─────────────────────────────────────────────
    /// Pending knowledge facts received from peers (drained by cognitive loop).
    pending_knowledge_shares: Vec<(String, f32)>,

    // ── Threat sharing ──────────────────────────────────────────────
    /// Pending threat patterns from peers (drained by SentinelManager).
    pending_threat_patterns: Vec<PeerThreatReport>,

    // ── Trust tracking ──────────────────────────────────────────────────
    /// Number of peers that completed the trust handshake.
    verified_peers: usize,

    // ── Telemetry snapshot ──────────────────────────────────────────────
    /// Last computed telemetry (readable between process calls).
    last_telemetry: SwarmTelemetry,

    // ── FHE Collective Wisdom ─────────────────────────────────────────
    /// FHE collective wisdom pool for privacy-preserving peer learning.
    #[cfg(feature = "fhe-wisdom")]
    wisdom_pool: symthaea_core::hdc::hdc_fhe::CollectiveWisdomPool,

    /// Cycle counter for FHE aggregation interval.
    #[cfg(feature = "fhe-wisdom")]
    fhe_cycles_since_aggregation: usize,

    /// Session mask for encrypting local wisdom contributions (OTP).
    /// Generated once per session; peers share via threshold splitting.
    #[cfg(feature = "fhe-wisdom")]
    session_mask: symthaea_core::hdc::binary_hv::BinaryHV,

    /// Total local contributions made this session.
    #[cfg(feature = "fhe-wisdom")]
    fhe_contributions_total: usize,

    /// Total aggregations completed this session.
    #[cfg(feature = "fhe-wisdom")]
    fhe_aggregations_total: usize,
}

impl Default for SwarmManager {
    fn default() -> Self {
        Self {
            pending_events: Vec::with_capacity(32),
            connected_peers: 0,
            expected_peers: 10, // default expectation, reconfigurable
            connectivity_ema: 0.0,
            peer_phi: Vec::new(),
            affective_valence_acc: 0.0,
            affective_arousal_acc: 0.0,
            affective_count: 0,
            federated_confidence: 0.0,
            federated_contributors: 0,
            anomaly_streak: 0,
            connectivity_history: VecDeque::with_capacity(8),
            connectivity_modifier: 1.0,
            pending_knowledge_shares: Vec::new(),
            pending_threat_patterns: Vec::new(),
            verified_peers: 0,
            last_telemetry: SwarmTelemetry::default(),
            #[cfg(feature = "fhe-wisdom")]
            wisdom_pool: symthaea_core::hdc::hdc_fhe::CollectiveWisdomPool::new(),
            #[cfg(feature = "fhe-wisdom")]
            fhe_cycles_since_aggregation: 0,
            #[cfg(feature = "fhe-wisdom")]
            session_mask: symthaea_core::hdc::binary_hv::BinaryHV::random(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64,
            ),
            #[cfg(feature = "fhe-wisdom")]
            fhe_contributions_total: 0,
            #[cfg(feature = "fhe-wisdom")]
            fhe_aggregations_total: 0,
        }
    }
}

impl SwarmManager {
    /// Co-prime scheduling interval (cycles).
    pub const INTERVAL: u32 = 41;

    /// Maximum tracked peers for Φ averaging (prevents unbounded growth).
    const MAX_TRACKED_PEERS: usize = 256;

    /// EMA alpha for connectivity tracking.
    const CONNECTIVITY_EMA_ALPHA: f64 = 0.15;

    /// Affective contagion gain — how much peer emotion shifts local state.
    /// Basis: Hatfield et al. (1993) — emotional contagion is ~10-15% of source intensity.
    const AFFECTIVE_CONTAGION_GAIN: f64 = 0.12;

    /// Confidence boost per connected peer (diminishing returns via sqrt).
    /// Basis: Heinrichs et al. (2003) — social buffering reduces stress response.
    const SOCIAL_BUFFERING_SCALE: f64 = 0.01;

    /// Anomaly detection: mass disconnect → arousal spike.
    const ANOMALY_AROUSAL_SPIKE: f64 = 0.08;

    /// Federated round confidence → learning rate boost ceiling.
    const FEDERATED_LR_MAX_BOOST: f64 = 0.15;

    // ── Public API ──────────────────────────────────────────────────────

    /// Inject a swarm event for processing on the next cycle.
    pub fn inject_event(&mut self, event: SwarmEvent) {
        self.pending_events.push(event);
    }

    /// Set the expected peer count for connectivity ratio.
    pub fn set_expected_peers(&mut self, n: usize) {
        self.expected_peers = n.max(1);
    }

    /// Drain pending knowledge shares received from peers.
    ///
    /// Returns (text, confidence) pairs. The cognitive loop should inject
    /// these into the KnowledgeManager as corroborated facts.
    pub fn drain_knowledge_shares(&mut self) -> Vec<(String, f32)> {
        std::mem::take(&mut self.pending_knowledge_shares)
    }

    /// Drain pending threat patterns received from peers.
    ///
    /// Returns threat reports for SentinelManager/ThreatMemory consumption.
    pub fn drain_threat_patterns(&mut self) -> Vec<PeerThreatReport> {
        std::mem::take(&mut self.pending_threat_patterns)
    }

    /// Get the current telemetry snapshot.
    pub fn telemetry(&self) -> &SwarmTelemetry {
        &self.last_telemetry
    }

    /// Number of pending unprocessed events.
    pub fn pending_count(&self) -> usize {
        self.pending_events.len()
    }

    /// Current connected peer count.
    pub fn connected_peers(&self) -> usize {
        self.connected_peers
    }

    /// Current expected peer count for connectivity ratio.
    pub fn expected_peers(&self) -> usize {
        self.expected_peers
    }

    /// Set connectivity modifier from radio tier state (mesh feature).
    ///
    /// Scales the effective connectivity EMA to reflect radio degradation.
    /// 1.0 = all tiers operational, 0.0 = complete radio blackout.
    /// NaN/Inf values are clamped to 1.0 (safe default).
    pub fn set_connectivity_modifier(&mut self, factor: f64) {
        self.connectivity_modifier = if factor.is_finite() {
            factor.clamp(0.0, 1.0)
        } else {
            1.0
        };
    }

    /// Mean peer Φ across tracked peers.
    pub fn mean_peer_phi(&self) -> f64 {
        if self.peer_phi.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.peer_phi.iter().map(|(_, phi)| phi).sum();
        sum / self.peer_phi.len() as f64
    }

    // ── FHE Collective Wisdom API ──────────────────────────────────────

    /// Contribute an encrypted wisdom vector from a peer to the collective pool.
    #[cfg(feature = "fhe-wisdom")]
    pub fn contribute_encrypted_wisdom(
        &mut self,
        peer_id: &str,
        encrypted: symthaea_core::hdc::hdc_fhe::EncryptedHV,
    ) -> bool {
        self.wisdom_pool.contribute(peer_id, encrypted)
    }

    /// Attempt aggregation if enough contributions collected.
    #[cfg(feature = "fhe-wisdom")]
    pub fn try_aggregate_wisdom(&mut self) -> Option<symthaea_core::hdc::hdc_fhe::EncryptedHV> {
        self.fhe_cycles_since_aggregation += 1;
        if self.wisdom_pool.contribution_count() >= 3 {
            let result = self.wisdom_pool.aggregate();
            if result.is_some() {
                self.wisdom_pool.clear();
                self.fhe_cycles_since_aggregation = 0;
            }
            result
        } else {
            None
        }
    }

    /// Current wisdom pool contribution count.
    #[cfg(feature = "fhe-wisdom")]
    pub fn wisdom_pool_count(&self) -> usize {
        self.wisdom_pool.contribution_count()
    }

    /// Contribute local consciousness state (BinaryHV) to the collective wisdom pool.
    ///
    /// The HV is encrypted with the session mask before contribution.
    /// This is the main entry point called from the cognitive loop each cycle.
    #[cfg(feature = "fhe-wisdom")]
    pub fn contribute_local_wisdom(
        &mut self,
        hv: &symthaea_core::hdc::binary_hv::BinaryHV,
    ) -> bool {
        let encrypted = symthaea_core::hdc::hdc_fhe::EncryptedHV::encrypt(hv, &self.session_mask);
        let ok = self.wisdom_pool.contribute("local", encrypted);
        if ok {
            self.fhe_contributions_total += 1;
        }
        ok
    }

    /// Try aggregation and return decrypted collective wisdom if threshold met.
    ///
    /// Decrypts using the session mask. In a full deployment, decryption
    /// would require k-of-n threshold mask recovery from peers.
    #[cfg(feature = "fhe-wisdom")]
    pub fn try_aggregate_and_decrypt(&mut self) -> Option<symthaea_core::hdc::binary_hv::BinaryHV> {
        self.fhe_cycles_since_aggregation += 1;
        if self.wisdom_pool.contribution_count() >= 3 {
            if let Some(encrypted_aggregate) = self.wisdom_pool.aggregate() {
                self.wisdom_pool.clear();
                self.fhe_cycles_since_aggregation = 0;
                self.fhe_aggregations_total += 1;
                return Some(encrypted_aggregate.decrypt(&self.session_mask));
            }
        }
        None
    }

    /// FHE telemetry: total contributions this session.
    #[cfg(feature = "fhe-wisdom")]
    pub fn fhe_contributions_total(&self) -> usize {
        self.fhe_contributions_total
    }

    /// FHE telemetry: total aggregations this session.
    #[cfg(feature = "fhe-wisdom")]
    pub fn fhe_aggregations_total(&self) -> usize {
        self.fhe_aggregations_total
    }

    /// FHE telemetry: cycles since last aggregation.
    #[cfg(feature = "fhe-wisdom")]
    pub fn fhe_cycles_since_aggregation(&self) -> usize {
        self.fhe_cycles_since_aggregation
    }

    // ── Internal helpers ────────────────────────────────────────────────

    fn drain_events(&mut self) {
        let events = std::mem::take(&mut self.pending_events);
        for event in events {
            match event {
                SwarmEvent::PeerJoined {
                    peer_id,
                    trust_level,
                } => {
                    self.connected_peers = self.connected_peers.saturating_add(1);
                    // Initialize peer Φ with trust-scaled default
                    if self.peer_phi.len() < Self::MAX_TRACKED_PEERS {
                        // Remove old entry if re-joining
                        self.peer_phi.retain(|(id, _)| id != &peer_id);
                        self.peer_phi.push((
                            peer_id,
                            trust_level * thresholds::SWARM_PEER_PHI_TRUST_SCALE,
                        ));
                    }
                }
                SwarmEvent::PeerLeft { peer_id } => {
                    self.connected_peers = self.connected_peers.saturating_sub(1);
                    self.peer_phi.retain(|(id, _)| id != &peer_id);
                }
                SwarmEvent::ConsciousnessUpdate { peer_id, phi, .. } => {
                    let phi = if phi.is_finite() {
                        phi.clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    if let Some(entry) = self.peer_phi.iter_mut().find(|(id, _)| id == &peer_id) {
                        entry.1 = phi;
                    } else if self.peer_phi.len() < Self::MAX_TRACKED_PEERS {
                        self.peer_phi.push((peer_id, phi));
                    }
                }
                SwarmEvent::AffectiveSync {
                    valence,
                    arousal,
                    intensity,
                    ..
                } => {
                    let intensity = if intensity.is_finite() {
                        intensity.clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let valence = if valence.is_finite() {
                        valence.clamp(-1.0, 1.0)
                    } else {
                        0.0
                    };
                    let arousal = if arousal.is_finite() {
                        arousal.clamp(0.0, 1.0)
                    } else {
                        0.5
                    };
                    self.affective_valence_acc += valence * intensity;
                    self.affective_arousal_acc +=
                        (arousal - thresholds::SWARM_AFFECTIVE_AROUSAL_CENTER) * intensity;
                    self.affective_count += 1;
                }
                SwarmEvent::FederatedRound {
                    n_contributors,
                    trust_confidence,
                    ..
                } => {
                    self.federated_contributors = n_contributors;
                    self.federated_confidence = if trust_confidence.is_finite() {
                        trust_confidence.clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                }
                SwarmEvent::TopologyChange {
                    connected_peers,
                    mass_disconnect,
                } => {
                    let prev = self.connected_peers;
                    self.connected_peers = connected_peers;
                    if mass_disconnect || (prev > 3 && connected_peers < prev / 2) {
                        self.anomaly_streak = self.anomaly_streak.saturating_add(1);
                    } else {
                        self.anomaly_streak = 0;
                    }
                }
                SwarmEvent::KnowledgeShare {
                    facts,
                    corroboration_count,
                    ..
                } => {
                    // Store shared facts for the cognitive loop to integrate
                    // into the knowledge manager. Corroboration boosts confidence.
                    let boost = (corroboration_count as f32
                        * thresholds::SWARM_CORROBORATION_BOOST)
                        .min(thresholds::SWARM_CORROBORATION_CAP);
                    for (text, confidence) in &facts {
                        let effective_confidence = (confidence + boost).min(1.0);
                        self.pending_knowledge_shares
                            .push((text.clone(), effective_confidence));
                    }
                    // Cap pending shares to prevent unbounded growth
                    if self.pending_knowledge_shares.len() > 64 {
                        self.pending_knowledge_shares.truncate(64);
                    }
                }
                SwarmEvent::TrustVerified { trust_level, .. } => {
                    // Track verified peer count for telemetry
                    if trust_level > 0.0 {
                        self.verified_peers = self.verified_peers.saturating_add(1);
                    }
                }
                SwarmEvent::ThreatPattern {
                    peer_id,
                    threat_kind,
                    severity,
                    confidence,
                    feature_vector,
                    description,
                } => {
                    // Clamp NaN/Inf and store for SentinelManager drain
                    let sev = if severity.is_finite() {
                        severity.clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let conf = if confidence.is_finite() {
                        confidence.clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    if self.pending_threat_patterns.len() < 32 {
                        self.pending_threat_patterns.push(PeerThreatReport {
                            peer_id,
                            threat_kind,
                            severity: sev,
                            confidence: conf,
                            feature_vector,
                            description,
                        });
                    }
                }
                // Sovereign Inoculation events — handled by their dedicated managers,
                // passed through here for uniform channel draining.
                // Sovereign Inoculation events — handled by dedicated managers.
                SwarmEvent::TimeBeaconReceived { .. } | SwarmEvent::ContentAnnounced { .. } => {}
            }
        }
    }

    fn update_connectivity_ema(&mut self) {
        let ratio = if self.expected_peers > 0 {
            (self.connected_peers as f64 / self.expected_peers as f64).min(1.0)
        } else {
            0.0
        };
        // Apply radio connectivity modifier — reduced radio bandwidth
        // means fewer peers are effectively reachable.
        let effective_ratio = ratio * self.connectivity_modifier;
        self.connectivity_ema = self.connectivity_ema * (1.0 - Self::CONNECTIVITY_EMA_ALPHA)
            + effective_ratio * Self::CONNECTIVITY_EMA_ALPHA;

        // History for anomaly detection
        if self.connectivity_history.len() >= 8 {
            self.connectivity_history.pop_front();
        }
        self.connectivity_history.push_back(self.connectivity_ema);
    }

    fn compute_affective_output(&mut self) -> (f64, f64) {
        if self.affective_count == 0 {
            return (0.0, 0.0);
        }
        let n = self.affective_count as f64;
        let mean_valence = self.affective_valence_acc / n;
        let mean_arousal = self.affective_arousal_acc / n;

        // Reset accumulators
        self.affective_valence_acc = 0.0;
        self.affective_arousal_acc = 0.0;
        self.affective_count = 0;

        (
            mean_valence * Self::AFFECTIVE_CONTAGION_GAIN,
            mean_arousal * Self::AFFECTIVE_CONTAGION_GAIN,
        )
    }

    fn update_telemetry(&mut self, affective_mag: f64) {
        self.last_telemetry = SwarmTelemetry {
            connected_peers: self.connected_peers,
            connectivity_ema: self.connectivity_ema,
            mean_peer_phi: self.mean_peer_phi(),
            affective_contagion: affective_mag,
            federated_confidence: self.federated_confidence,
            anomaly_count: self.anomaly_streak,
            verified_peers: self.verified_peers,
        };
    }
}

impl CognitiveSubsystem for SwarmManager {
    fn name(&self) -> &'static str {
        "swarm_manager"
    }

    fn interval(&self) -> u32 {
        Self::INTERVAL
    }

    fn process(&mut self, _snapshot: &CycleSnapshot) -> SubsystemOutput {
        let mut output = SubsystemOutput::NEUTRAL;

        // ── 1. Drain and process events ─────────────────────────────────
        self.drain_events();
        self.update_connectivity_ema();

        // ── 2. Social buffering → confidence ────────────────────────────
        // More peers = higher confidence (sqrt for diminishing returns).
        // Heinrichs et al. (2003): social support reduces HPA axis reactivity.
        if self.connected_peers > 0 {
            let buffering = (self.connected_peers as f64).sqrt() * Self::SOCIAL_BUFFERING_SCALE;
            output.confidence_delta += buffering.min(thresholds::SWARM_SOCIAL_BUFFERING_CAP);
        }

        // ── 3. Affective contagion → valence/arousal ────────────────────
        let (valence_shift, arousal_shift) = self.compute_affective_output();
        output.valence_delta += valence_shift as f32;
        output.arousal_delta += arousal_shift as f32;

        let affective_mag = (valence_shift.powi(2) + arousal_shift.powi(2)).sqrt();

        // ── 4. Collective Φ → learning rate modulation ──────────────────
        let mean_phi = self.mean_peer_phi();
        if mean_phi > thresholds::SWARM_COLLECTIVE_PHI_THRESHOLD {
            // High collective consciousness → boost learning (collective intelligence).
            let phi_boost = (mean_phi - thresholds::SWARM_COLLECTIVE_PHI_THRESHOLD)
                * thresholds::SWARM_COLLECTIVE_PHI_LR_SCALE;
            output.lr_modulation = 1.0 + phi_boost.min(thresholds::SWARM_COLLECTIVE_PHI_LR_CAP);
        }

        // ── 5. Federated round → learning rate ──────────────────────────
        if self.federated_confidence > 0.5 && self.federated_contributors > 1 {
            let fed_boost = (self.federated_confidence - 0.5)
                * Self::FEDERATED_LR_MAX_BOOST
                * thresholds::SWARM_FEDERATED_BOOST_MULTIPLIER;
            output.lr_modulation *= 1.0 + fed_boost.min(Self::FEDERATED_LR_MAX_BOOST);
        }

        // ── 6. Network anomaly → alarm response ─────────────────────────
        if self.anomaly_streak > 0 {
            // Mass disconnect → arousal spike + exploration.
            output.arousal_delta += Self::ANOMALY_AROUSAL_SPIKE as f32;
            output.exploration_delta +=
                thresholds::SWARM_ANOMALY_EXPLORATION * self.anomaly_streak.min(3) as f64;
            output.flags |= output_flags::ANOMALY_DETECTED;

            // Extended anomaly → confidence drop.
            if self.anomaly_streak >= 2 {
                output.confidence_delta -=
                    thresholds::SWARM_ANOMALY_CONFIDENCE * self.anomaly_streak.min(5) as f64;
            }
        }

        // ── 7. Isolation detection → increased exploration ──────────────
        if self.connectivity_ema < thresholds::SWARM_ISOLATION_THRESHOLD
            && self.connected_peers == 0
        {
            // Complete isolation → explore to find peers.
            output.exploration_delta += thresholds::SWARM_ISOLATION_EXPLORATION_BOOST;
            output.flags |= output_flags::REQUEST_EXPLORATION;
        }

        // ── 8. Update telemetry ─────────────────────────────────────────
        self.update_telemetry(affective_mag);

        output
    }

    fn checkpoint(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(64);
        // Format: [connected_peers:u32][connectivity_ema:f64][federated_confidence:f64]
        //         [anomaly_streak:u32][expected_peers:u32]
        data.extend_from_slice(&(self.connected_peers as u32).to_le_bytes());
        data.extend_from_slice(&self.connectivity_ema.to_le_bytes());
        data.extend_from_slice(&self.federated_confidence.to_le_bytes());
        data.extend_from_slice(&self.anomaly_streak.to_le_bytes());
        data.extend_from_slice(&(self.expected_peers as u32).to_le_bytes());
        data
    }

    fn restore(&mut self, data: &[u8]) -> Result<(), String> {
        if data.len() < 24 {
            return Err(format!(
                "SwarmManager checkpoint too short: {} < 24",
                data.len()
            ));
        }
        self.connected_peers = u32::from_le_bytes(
            data[0..4]
                .try_into()
                .map_err(|_| "SwarmManager: corrupt bytes [0..4]".to_string())?,
        ) as usize;
        self.connectivity_ema = f64::from_le_bytes(
            data[4..12]
                .try_into()
                .map_err(|_| "SwarmManager: corrupt bytes [4..12]".to_string())?,
        );
        self.federated_confidence = f64::from_le_bytes(
            data[12..20]
                .try_into()
                .map_err(|_| "SwarmManager: corrupt bytes [12..20]".to_string())?,
        );
        self.anomaly_streak = u32::from_le_bytes(
            data[20..24]
                .try_into()
                .map_err(|_| "SwarmManager: corrupt bytes [20..24]".to_string())?,
        );
        if data.len() >= 28 {
            self.expected_peers = u32::from_le_bytes(
                data[24..28]
                    .try_into()
                    .map_err(|_| "SwarmManager: corrupt bytes [24..28]".to_string())?,
            ) as usize;
        }
        Ok(())
    }
}

// ── Swarm Module Adapters ─────────────────────────────────────────────────

/// Convert a [`PeerEvent`] from the swarm network module into a [`SwarmEvent`]
/// for the cognitive loop's SwarmManager.
///
/// Returns `None` for events that don't map to cognitive-level signals
/// (e.g., `Discovered` — not yet connected, `TrustChanged` — handled implicitly).
pub fn convert_peer_event(event: &crate::swarm::PeerEvent) -> Option<SwarmEvent> {
    match event {
        crate::swarm::PeerEvent::Connected(info) => Some(SwarmEvent::PeerJoined {
            peer_id: info.node_id.clone(),
            trust_level: info.trust_level.value(),
        }),
        crate::swarm::PeerEvent::Disconnected { peer_id, .. } => Some(SwarmEvent::PeerLeft {
            peer_id: peer_id.clone(),
        }),
        crate::swarm::PeerEvent::ConsciousnessUpdate { peer_id, phi, .. } => {
            Some(SwarmEvent::ConsciousnessUpdate {
                peer_id: peer_id.clone(),
                phi: *phi,
                valence: 0.0, // ConsciousnessUpdate only carries phi
                arousal: 0.0,
            })
        }
        crate::swarm::PeerEvent::TrustChanged {
            peer_id,
            new: trust,
            ..
        } => Some(SwarmEvent::TrustVerified {
            peer_id: peer_id.clone(),
            trust_level: trust.value(),
            agent_pubkey: String::new(),
        }),
        // Discovered doesn't map to cognitive-level signals
        _ => None,
    }
}

/// Convert a [`ConsciousnessVector`] from the swarm module into a
/// [`SwarmEvent::ConsciousnessUpdate`].
pub fn convert_consciousness_vector(
    peer_id: &str,
    cv: &crate::swarm::ConsciousnessVector,
) -> SwarmEvent {
    SwarmEvent::ConsciousnessUpdate {
        peer_id: peer_id.to_string(),
        phi: cv.phi,
        valence: cv.valence.clamp(-1.0, 1.0),
        arousal: cv.arousal.clamp(0.0, 1.0),
    }
}

/// Convert an [`AffectiveSync`] from the swarm module into a
/// [`SwarmEvent::AffectiveSync`].
pub fn convert_affective_sync(peer_id: &str, sync: &crate::swarm::AffectiveSync) -> SwarmEvent {
    SwarmEvent::AffectiveSync {
        peer_id: peer_id.to_string(),
        valence: (sync.valence as f64).clamp(-1.0, 1.0),
        arousal: (sync.arousal as f64).clamp(0.0, 1.0),
        intensity: (sync.dominance.abs() as f64).clamp(0.0, 1.0),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_loop::subsystem_trait::CycleSnapshot;

    #[test]
    fn test_name_and_interval() {
        let sm = SwarmManager::default();
        assert_eq!(sm.name(), "swarm_manager");
        assert_eq!(sm.interval(), 41);
    }

    #[test]
    fn test_interval_coprime() {
        let interval = 41u32;
        for other in [7, 11, 13, 19, 23, 29, 37] {
            assert_eq!(
                gcd(interval, other),
                1,
                "41 should be co-prime with {}",
                other
            );
        }
    }

    fn gcd(a: u32, b: u32) -> u32 {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }

    #[test]
    fn test_neutral_without_events() {
        let mut sm = SwarmManager::default();
        // Start with 1 peer to avoid isolation detection
        sm.connected_peers = 1;
        sm.connectivity_ema = 0.5;
        let snapshot = CycleSnapshot::default();
        let output = sm.process(&snapshot);

        // Social buffering from 1 peer → small positive confidence
        assert!(output.confidence_delta > 0.0);
        assert_eq!(output.valence_delta, 0.0);
        assert_eq!(output.arousal_delta, 0.0);
        // No exploration signal (not isolated)
        assert!(output.exploration_delta < 0.01);
    }

    #[test]
    fn test_peer_join_boosts_confidence() {
        let mut sm = SwarmManager::default();
        sm.inject_event(SwarmEvent::PeerJoined {
            peer_id: "peer-1".into(),
            trust_level: 0.8,
        });
        sm.inject_event(SwarmEvent::PeerJoined {
            peer_id: "peer-2".into(),
            trust_level: 0.9,
        });

        let snapshot = CycleSnapshot::default();
        let output = sm.process(&snapshot);

        assert!(
            output.confidence_delta > 0.0,
            "Peers should boost confidence: {}",
            output.confidence_delta
        );
    }

    #[test]
    fn test_affective_contagion() {
        let mut sm = SwarmManager::default();
        sm.inject_event(SwarmEvent::AffectiveSync {
            peer_id: "p1".into(),
            valence: 0.8,
            arousal: 0.7,
            intensity: 0.9,
        });

        let snapshot = CycleSnapshot::default();
        let output = sm.process(&snapshot);

        assert!(
            output.valence_delta > 0.0,
            "Positive peer valence should shift local: {}",
            output.valence_delta
        );
    }

    #[test]
    fn test_mass_disconnect_alarm() {
        let mut sm = SwarmManager::default();
        sm.connected_peers = 10;
        sm.inject_event(SwarmEvent::TopologyChange {
            connected_peers: 2,
            mass_disconnect: true,
        });

        let snapshot = CycleSnapshot::default();
        let output = sm.process(&snapshot);

        assert!(
            output.arousal_delta > 0.0,
            "Mass disconnect should spike arousal: {}",
            output.arousal_delta
        );
        assert!(
            output.flags & output_flags::ANOMALY_DETECTED != 0,
            "Should set ANOMALY_DETECTED flag"
        );
    }

    #[test]
    fn test_peer_phi_modulates_learning() {
        let mut sm = SwarmManager::default();
        // Add peers with high Φ
        for i in 0..5 {
            sm.inject_event(SwarmEvent::PeerJoined {
                peer_id: format!("p{}", i),
                trust_level: 0.9,
            });
            sm.inject_event(SwarmEvent::ConsciousnessUpdate {
                peer_id: format!("p{}", i),
                phi: 0.7,
                valence: 0.0,
                arousal: 0.5,
            });
        }

        let snapshot = CycleSnapshot::default();
        let output = sm.process(&snapshot);

        assert!(
            output.lr_modulation > 1.0,
            "High collective Φ should boost learning: {}",
            output.lr_modulation
        );
    }

    #[test]
    fn test_federated_round_boosts_lr() {
        let mut sm = SwarmManager::default();
        sm.inject_event(SwarmEvent::FederatedRound {
            n_contributors: 5,
            avg_quality: 0.8,
            trust_confidence: 0.9,
        });

        let snapshot = CycleSnapshot::default();
        let output = sm.process(&snapshot);

        assert!(
            output.lr_modulation > 1.0,
            "Federated round should boost LR: {}",
            output.lr_modulation
        );
    }

    #[test]
    fn test_isolation_drives_exploration() {
        let mut sm = SwarmManager::default();
        // Start with connectivity EMA at 0 (default), 0 peers
        sm.expected_peers = 10;

        let snapshot = CycleSnapshot::default();
        let output = sm.process(&snapshot);

        assert!(
            output.exploration_delta > 0.0,
            "Isolation should drive exploration: {}",
            output.exploration_delta
        );
        assert!(
            output.flags & output_flags::REQUEST_EXPLORATION != 0,
            "Should request exploration"
        );
    }

    #[test]
    fn test_nan_inputs_clamped() {
        let mut sm = SwarmManager::default();
        sm.inject_event(SwarmEvent::ConsciousnessUpdate {
            peer_id: "nan-peer".into(),
            phi: f64::NAN,
            valence: f64::INFINITY,
            arousal: f64::NEG_INFINITY,
        });
        sm.inject_event(SwarmEvent::AffectiveSync {
            peer_id: "nan-peer".into(),
            valence: f64::NAN,
            arousal: f64::NAN,
            intensity: f64::INFINITY,
        });

        let snapshot = CycleSnapshot::default();
        let output = sm.process(&snapshot);

        assert!(
            output.valence_delta.is_finite(),
            "NaN inputs should produce finite output"
        );
        assert!(
            output.arousal_delta.is_finite(),
            "NaN inputs should produce finite output"
        );
    }

    #[test]
    fn test_peer_cap_enforced() {
        let mut sm = SwarmManager::default();
        for i in 0..300 {
            sm.inject_event(SwarmEvent::PeerJoined {
                peer_id: format!("p{}", i),
                trust_level: 0.5,
            });
        }

        let snapshot = CycleSnapshot::default();
        sm.process(&snapshot);

        assert!(
            sm.peer_phi.len() <= SwarmManager::MAX_TRACKED_PEERS,
            "Peer cap should be enforced: {}",
            sm.peer_phi.len()
        );
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let mut sm = SwarmManager::default();
        sm.connected_peers = 7;
        sm.connectivity_ema = 0.65;
        sm.federated_confidence = 0.82;
        sm.anomaly_streak = 2;
        sm.expected_peers = 15;

        let data = sm.checkpoint();
        let mut sm2 = SwarmManager::default();
        sm2.restore(&data).unwrap();

        assert_eq!(sm2.connected_peers, 7);
        assert!((sm2.connectivity_ema - 0.65).abs() < 1e-10);
        assert!((sm2.federated_confidence - 0.82).abs() < 1e-10);
        assert_eq!(sm2.anomaly_streak, 2);
        assert_eq!(sm2.expected_peers, 15);
    }

    #[test]
    fn test_restore_rejects_short_data() {
        let mut sm = SwarmManager::default();
        assert!(sm.restore(&[0u8; 10]).is_err());
    }

    #[test]
    fn test_telemetry_updated() {
        let mut sm = SwarmManager::default();
        sm.inject_event(SwarmEvent::PeerJoined {
            peer_id: "t1".into(),
            trust_level: 0.9,
        });

        let snapshot = CycleSnapshot::default();
        sm.process(&snapshot);

        let telem = sm.telemetry();
        assert_eq!(telem.connected_peers, 1);
        assert!(telem.connectivity_ema > 0.0);
    }

    #[test]
    fn test_peer_leave_decrements() {
        let mut sm = SwarmManager::default();
        sm.inject_event(SwarmEvent::PeerJoined {
            peer_id: "leave-test".into(),
            trust_level: 0.5,
        });
        sm.inject_event(SwarmEvent::PeerLeft {
            peer_id: "leave-test".into(),
        });

        let snapshot = CycleSnapshot::default();
        sm.process(&snapshot);

        assert_eq!(sm.connected_peers, 0);
        assert!(sm.peer_phi.is_empty());
    }

    #[test]
    fn test_extended_anomaly_drops_confidence() {
        let mut sm = SwarmManager::default();
        sm.connected_peers = 10;

        // Multiple mass disconnects
        for _ in 0..3 {
            sm.inject_event(SwarmEvent::TopologyChange {
                connected_peers: 1,
                mass_disconnect: true,
            });
        }

        let snapshot = CycleSnapshot::default();
        let output = sm.process(&snapshot);

        assert!(
            output.confidence_delta < 0.0,
            "Extended anomaly should drop confidence: {}",
            output.confidence_delta
        );
    }

    // ── Adapter tests ───────────────────────────────────────────────────

    #[test]
    fn test_convert_peer_event() {
        use crate::swarm::{PeerEvent, PeerInfo, TrustLevel};

        // Connected → PeerJoined with trust value
        let mut info = PeerInfo::new("node-abc");
        info.trust_level = TrustLevel::Verified(0.75);
        let event = convert_peer_event(&PeerEvent::Connected(info));
        match event {
            Some(SwarmEvent::PeerJoined {
                peer_id,
                trust_level,
            }) => {
                assert_eq!(peer_id, "node-abc");
                assert!((trust_level - 0.75).abs() < 1e-10);
            }
            other => panic!("Expected PeerJoined, got {:?}", other),
        }

        // Disconnected → PeerLeft
        let event = convert_peer_event(&PeerEvent::Disconnected {
            peer_id: "node-xyz".into(),
            reason: "timeout".into(),
        });
        match event {
            Some(SwarmEvent::PeerLeft { peer_id }) => {
                assert_eq!(peer_id, "node-xyz");
            }
            other => panic!("Expected PeerLeft, got {:?}", other),
        }

        // Discovered → None (not cognitive-level)
        let info = PeerInfo::new("node-zzz");
        assert!(convert_peer_event(&PeerEvent::Discovered(info)).is_none());
    }

    #[test]
    fn test_convert_consciousness_vector() {
        use crate::swarm::ConsciousnessVector;

        let mut cv = ConsciousnessVector::new(vec![0.0; 64], 0.42);
        cv.valence = 0.6;
        cv.arousal = 0.8;

        let event = convert_consciousness_vector("peer-7", &cv);
        match event {
            SwarmEvent::ConsciousnessUpdate {
                peer_id,
                phi,
                valence,
                arousal,
            } => {
                assert_eq!(peer_id, "peer-7");
                assert!((phi - 0.42).abs() < 1e-10);
                assert!((valence - 0.6).abs() < 1e-10);
                assert!((arousal - 0.8).abs() < 1e-10);
            }
            other => panic!("Expected ConsciousnessUpdate, got {:?}", other),
        }

        // Out-of-range values are clamped
        let mut cv_oob = ConsciousnessVector::new(vec![], 0.5);
        cv_oob.valence = -5.0;
        cv_oob.arousal = 99.0;
        let event = convert_consciousness_vector("oob", &cv_oob);
        match event {
            SwarmEvent::ConsciousnessUpdate {
                valence, arousal, ..
            } => {
                assert!(
                    (valence - (-1.0)).abs() < 1e-10,
                    "valence should clamp to -1.0"
                );
                assert!((arousal - 1.0).abs() < 1e-10, "arousal should clamp to 1.0");
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_convert_affective_sync() {
        use crate::swarm::AffectiveSync as SwarmAffectiveSync;

        let sync = SwarmAffectiveSync {
            valence: -0.3,
            arousal: 0.6,
            dominance: -0.8,
            timestamp_ms: 0,
            sequence: 0,
        };

        let event = convert_affective_sync("peer-9", &sync);
        match event {
            SwarmEvent::AffectiveSync {
                peer_id,
                valence,
                arousal,
                intensity,
            } => {
                assert_eq!(peer_id, "peer-9");
                assert!((valence - (-0.3f32 as f64)).abs() < 1e-6);
                assert!((arousal - (0.6f32 as f64)).abs() < 1e-6);
                // intensity = |dominance| = 0.8
                assert!((intensity - 0.8).abs() < 1e-6);
            }
            other => panic!("Expected AffectiveSync, got {:?}", other),
        }
    }

    #[cfg(feature = "fhe-wisdom")]
    #[test]
    fn test_fhe_wisdom_contribute_and_aggregate() {
        let mut mgr = SwarmManager::default();

        let mask = symthaea_core::hdc::BinaryHV::random(42);
        for i in 0..3 {
            let hv = symthaea_core::hdc::BinaryHV::random(i);
            let encrypted = symthaea_core::hdc::hdc_fhe::EncryptedHV::encrypt(&hv, &mask);
            assert!(mgr.contribute_encrypted_wisdom(&format!("peer_{i}"), encrypted));
        }

        assert_eq!(mgr.wisdom_pool_count(), 3);
        let agg = mgr.try_aggregate_wisdom();
        assert!(agg.is_some());
        assert_eq!(mgr.wisdom_pool_count(), 0);
    }

    #[cfg(feature = "fhe-wisdom")]
    #[test]
    fn test_fhe_wisdom_insufficient_contributions() {
        let mut mgr = SwarmManager::default();

        let mask = symthaea_core::hdc::BinaryHV::random(42);
        let hv = symthaea_core::hdc::BinaryHV::random(1);
        let encrypted = symthaea_core::hdc::hdc_fhe::EncryptedHV::encrypt(&hv, &mask);
        mgr.contribute_encrypted_wisdom("peer_1", encrypted);

        assert_eq!(mgr.wisdom_pool_count(), 1);
        let agg = mgr.try_aggregate_wisdom();
        assert!(agg.is_none()); // Not enough contributions
    }
}
