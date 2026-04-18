// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
    /// Peer navigation estimate shared through the swarm service.
    PeerNavigationUpdate {
        peer_id: String,
        position_m: [f64; 3],
        position_sigma_m: f64,
        confidence: Option<f64>,
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

    /// A peer's Markov blanket state update (Kirchhoff et al. 2018).
    BlanketStateUpdate {
        peer_id: String,
        effective_permeability: f64,
        coalescence_ready: bool,
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

    /// Waste/circular economy event from peer or external system.
    ///
    /// Basis: Ellen MacArthur Foundation (2015) — circular economy as
    /// distributed feedback loop. High circularity signals system health;
    /// low material entropy signals concentration risk.
    #[cfg(feature = "circular")]
    WasteCircularityUpdate {
        /// Source peer or external system identifier.
        source_id: String,
        /// Material class being tracked (e.g., "plastic", "organic", "metal").
        material_class: String,
        /// Quantity in kg.
        quantity_kg: f32,
        /// Lifecycle stage: "collection", "processing", "remanufacturing", etc.
        lifecycle_stage: String,
        /// Circularity potential [0.0, 1.0] (1.0 = fully recyclable).
        circularity_potential: f32,
        /// Classification confidence [0.0, 1.0].
        confidence: f32,
    },

    /// Embodiment distress signal from a peer.
    ///
    /// Triggered when: high prediction_error + energy depletion + safety degradation.
    /// Basis: de Waal (2008) social alarm signals; Eisenberg (2000) empathic distress.
    DistressSignal {
        peer_id: String,
        prediction_error: f32,
        energy_depletion: f32,
        safety_level: u8,
        allostatic_load: f32,
        consciousness_level: f32,
    },

    /// Space situational awareness alert from mycelix-space or ground network.
    ///
    /// Basis: Kahneman (2011) — threat salience drives attentional capture.
    #[cfg(feature = "space-alerts")]
    SpaceAlert {
        /// Type of space alert.
        alert_type: SpaceAlertType,
        /// Severity (0.0–1.0): mapped from RiskLevel
        /// (Negligible=0.1, Low=0.3, Medium=0.5, High=0.8, Emergency=1.0).
        severity: f32,
        /// Collision probability or alert confidence.
        confidence: f32,
        /// Seconds until event (negative = past).
        time_to_event_seconds: f64,
        /// Human-readable description.
        description: String,
    },
    #[cfg(feature = "hypervisor")]
    SupervisoryHeartbeat {
        supervisor_id: String,
        epoch: u64,
        subordinate_count: usize,
        aggregated_phi: f64,
    },
    #[cfg(feature = "hypervisor")]
    SupervisoryEscalation {
        supervisor_id: String,
        reason: String,
        severity: f32,
    },
    #[cfg(feature = "hypervisor")]
    SupervisorElection {
        supervisor_id: String,
        phi_score: f64,
        epoch: u64,
    },
}

/// Type of space situational awareness alert.
///
/// Maps to mycelix-space conjunction screening, debris tracking,
/// and traffic coordination subsystems.
#[cfg(feature = "space-alerts")]
#[derive(Debug, Clone)]
pub enum SpaceAlertType {
    /// Conjunction predicted between tracked objects.
    ConjunctionWarning,
    /// Debris approaching operational region.
    DebrisProximity,
    /// Communication window opening with relay satellite.
    CommWindow,
    /// Unexpected orbital behavior detected.
    OrbitalAnomaly,
    /// Maneuver announced by peer operator.
    ManeuverAnnounced,
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
    /// Number of peers currently reporting usable navigation.
    pub navigation_ready_peers: usize,
    /// Mean peer-reported navigation sigma in meters.
    pub mean_peer_navigation_sigma_m: f64,
    /// Federated learning trust confidence.
    pub federated_confidence: f64,
    /// Number of anomaly events since last telemetry.
    pub anomaly_count: u32,
    /// Number of peers that completed trust handshake verification.
    pub verified_peers: usize,
    /// Number of active coalitions (Markov blanket merges, Friston 2013).
    pub coalition_count: usize,
    /// Number of conscious collectives (coalitions passing consciousness test).
    pub conscious_collective_count: usize,
    /// Total space alerts received (monotonic counter).
    #[cfg(feature = "space-alerts")]
    pub space_alert_count: u32,
    /// Severity of most recent space alert [0.0, 1.0].
    #[cfg(feature = "space-alerts")]
    pub last_space_alert_severity: f32,
    /// Total waste quantity tracked across network (kg).
    #[cfg(feature = "circular")]
    pub waste_total_kg: f32,
    /// Mean circularity potential across tracked waste [0.0, 1.0].
    #[cfg(feature = "circular")]
    pub waste_mean_circularity: f32,
    /// Number of waste events processed this interval.
    #[cfg(feature = "circular")]
    pub waste_events_processed: u32,
    /// Confidence EMA for waste classifications [0.0, 1.0].
    #[cfg(feature = "circular")]
    pub waste_confidence_ema: f64,
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
    /// Per-peer last known navigation sigma.
    peer_navigation_sigma: Vec<(String, f64)>,

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

    // ── Space alerts ─────────────────────────────────────────────────
    /// Count of space alerts received (monotonic).
    #[cfg(feature = "space-alerts")]
    space_alert_count: u32,
    /// Severity of most recent space alert.
    #[cfg(feature = "space-alerts")]
    last_space_alert_severity: f32,
    /// Accumulated arousal delta from space alerts this interval.
    #[cfg(feature = "space-alerts")]
    space_arousal_acc: f32,
    /// Accumulated valence delta from space alerts this interval.
    #[cfg(feature = "space-alerts")]
    space_valence_acc: f32,
    /// Accumulated confidence delta from space alerts this interval.
    #[cfg(feature = "space-alerts")]
    space_confidence_acc: f32,
    /// Accumulated exploration delta from space alerts this interval.
    #[cfg(feature = "space-alerts")]
    space_exploration_acc: f64,
    /// Accumulated LR modulation from space comm windows this interval.
    #[cfg(feature = "space-alerts")]
    space_lr_acc: f64,

    // ── Waste/Circular Economy tracking ─────────────────────────────────
    /// Running total of waste tracked (kg).
    #[cfg(feature = "circular")]
    waste_total_kg: f32,
    /// Running mean circularity potential [0.0, 1.0].
    #[cfg(feature = "circular")]
    waste_mean_circularity: f32,
    /// Events processed this interval.
    #[cfg(feature = "circular")]
    waste_events_processed: u32,
    /// Confidence EMA for waste classifications.
    #[cfg(feature = "circular")]
    waste_confidence_ema: f64,

    // ── Markov blanket coalescence (Friston 2013) ─────────────────────
    active_coalitions: Vec<crate::consciousness::fep_active_inference::SwarmCoalition>,
    coalition_cycle_counter: u32,
    peer_blanket_permeability: Vec<(String, f64)>,

    /// Swarm consciousness state (computed after coalition identification).
    pub swarm_consciousness: super::swarm_consciousness::SwarmConsciousness,

    // ── Distress tracking ──────────────────────────────────────────────
    /// Number of distress signals received (monotonic).
    distress_signals_received: u32,

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
            peer_navigation_sigma: Vec::new(),
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
            #[cfg(feature = "space-alerts")]
            space_alert_count: 0,
            #[cfg(feature = "space-alerts")]
            last_space_alert_severity: 0.0,
            #[cfg(feature = "space-alerts")]
            space_arousal_acc: 0.0,
            #[cfg(feature = "space-alerts")]
            space_valence_acc: 0.0,
            #[cfg(feature = "space-alerts")]
            space_confidence_acc: 0.0,
            #[cfg(feature = "space-alerts")]
            space_exploration_acc: 0.0,
            #[cfg(feature = "space-alerts")]
            space_lr_acc: 0.0,
            #[cfg(feature = "circular")]
            waste_total_kg: 0.0,
            #[cfg(feature = "circular")]
            waste_mean_circularity: 0.0,
            #[cfg(feature = "circular")]
            waste_events_processed: 0,
            #[cfg(feature = "circular")]
            waste_confidence_ema: 0.0,
            distress_signals_received: 0,
            active_coalitions: Vec::new(),
            coalition_cycle_counter: 0,
            peer_blanket_permeability: Vec::new(),
            swarm_consciousness: Default::default(),
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

    pub fn mean_peer_navigation_sigma_m(&self) -> f64 {
        if self.peer_navigation_sigma.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .peer_navigation_sigma
            .iter()
            .map(|(_, sigma)| sigma)
            .sum();
        sum / self.peer_navigation_sigma.len() as f64
    }

    pub fn navigation_ready_peers(&self) -> usize {
        self.peer_navigation_sigma
            .iter()
            .filter(|(_, sigma)| sigma.is_finite() && *sigma <= 25.0)
            .count()
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

    // ── Markov blanket coalescence API ──────────────────────────────────

    const COALITION_INTERVAL: u32 = 10;
    const COALITION_PERMEABILITY_THRESHOLD: f64 = 0.7;

    pub fn identify_coalitions(&mut self) -> usize {
        self.coalition_cycle_counter += 1;
        if self.coalition_cycle_counter < Self::COALITION_INTERVAL {
            return self.active_coalitions.len();
        }
        self.coalition_cycle_counter = 0;
        if self.peer_phi.len() < 2 {
            self.active_coalitions.clear();
            return 0;
        }
        let n = self.peer_phi.len();
        let mut edges = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let peer_i_blanket = self
                    .peer_blanket_permeability
                    .iter()
                    .find(|(id, _)| id == &self.peer_phi[i].0)
                    .map(|(_, p)| *p);
                let peer_j_blanket = self
                    .peer_blanket_permeability
                    .iter()
                    .find(|(id, _)| id == &self.peer_phi[j].0)
                    .map(|(_, p)| *p);
                let permeability = match (peer_i_blanket, peer_j_blanket) {
                    (Some(pi), Some(pj)) => pi.min(pj),
                    (Some(pi), None) | (None, Some(pi)) => {
                        let phi_proxy =
                            (1.0 - (self.peer_phi[i].1 - self.peer_phi[j].1).abs()).max(0.0);
                        (pi + phi_proxy) * 0.5
                    }
                    (None, None) => {
                        (1.0 - (self.peer_phi[i].1 - self.peer_phi[j].1).abs()).max(0.0)
                    }
                };
                edges.push((i, j, permeability));
            }
        }
        self.active_coalitions = crate::consciousness::fep_active_inference::identify_coalitions(
            &self.peer_phi,
            &edges,
            Self::COALITION_PERMEABILITY_THRESHOLD,
        );
        // Recompute swarm consciousness after coalition identification
        self.swarm_consciousness = super::swarm_consciousness::SwarmConsciousness::compute(
            &self.active_coalitions,
            self.mean_peer_phi(),
            None, // self_id not tracked here; set by the cognitive loop
        );
        self.active_coalitions.len()
    }

    pub fn coalitions(&self) -> &[crate::consciousness::fep_active_inference::SwarmCoalition] {
        &self.active_coalitions
    }

    /// Effective Phi for this agent, incorporating swarm delegation.
    ///
    /// Returns max(individual_phi, swarm-delegated Phi) when in a conscious collective.
    pub fn effective_phi(&self, individual_phi: f64) -> f64 {
        if self.swarm_consciousness.delegation.in_conscious_collective {
            let alpha = 0.3;
            let blended =
                alpha * self.swarm_consciousness.phi_swarm + (1.0 - alpha) * individual_phi;
            blended.max(individual_phi)
        } else {
            individual_phi
        }
    }

    pub fn conscious_collective_count(&self) -> usize {
        self.active_coalitions
            .iter()
            .filter(|c| c.is_conscious_collective())
            .count()
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
                    self.peer_navigation_sigma.retain(|(id, _)| id != &peer_id);
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
                SwarmEvent::PeerNavigationUpdate {
                    peer_id,
                    position_sigma_m,
                    ..
                } => {
                    let sigma = if position_sigma_m.is_finite() {
                        position_sigma_m.clamp(0.0, 1.0e9)
                    } else {
                        1.0e9
                    };
                    if let Some(entry) = self
                        .peer_navigation_sigma
                        .iter_mut()
                        .find(|(id, _)| id == &peer_id)
                    {
                        entry.1 = sigma;
                    } else if self.peer_navigation_sigma.len() < Self::MAX_TRACKED_PEERS {
                        self.peer_navigation_sigma.push((peer_id, sigma));
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

                SwarmEvent::BlanketStateUpdate {
                    peer_id,
                    effective_permeability,
                    ..
                } => {
                    let perm = if effective_permeability.is_finite() {
                        effective_permeability.clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    if let Some(entry) = self
                        .peer_blanket_permeability
                        .iter_mut()
                        .find(|(id, _)| id == &peer_id)
                    {
                        entry.1 = perm;
                    } else if self.peer_blanket_permeability.len() < Self::MAX_TRACKED_PEERS {
                        self.peer_blanket_permeability.push((peer_id, perm));
                    }
                }

                // Space situational awareness alerts — modulate arousal/valence/confidence.
                #[cfg(feature = "space-alerts")]
                SwarmEvent::SpaceAlert {
                    ref alert_type,
                    severity,
                    confidence,
                    time_to_event_seconds,
                    ..
                } => {
                    // NaN/Inf guard on inputs
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
                    let tte = if time_to_event_seconds.is_finite() {
                        time_to_event_seconds
                    } else {
                        0.0
                    };

                    self.space_alert_count = self.space_alert_count.saturating_add(1);
                    self.last_space_alert_severity = sev;

                    // Urgency factor: events closer in time are more urgent
                    let urgency = if tte > 0.0 {
                        (1.0 / (1.0 + tte / 3600.0)) as f32
                    } else {
                        1.0 // Already happening or past
                    };

                    match alert_type {
                        SpaceAlertType::ConjunctionWarning => {
                            // Conjunction → heightened alertness, search for alternatives
                            self.space_arousal_acc +=
                                thresholds::SPACE_CONJUNCTION_AROUSAL * sev * urgency;
                            self.space_exploration_acc +=
                                thresholds::SPACE_CONJUNCTION_EXPLORATION as f64 * conf as f64;
                        }
                        SpaceAlertType::DebrisProximity => {
                            // Debris → threat response (fight-or-flight analogue)
                            self.space_arousal_acc +=
                                thresholds::SPACE_DEBRIS_AROUSAL * sev * urgency;
                            self.space_valence_acc += thresholds::SPACE_DEBRIS_VALENCE * sev;
                            self.space_confidence_acc += thresholds::SPACE_DEBRIS_CONFIDENCE * sev;
                        }
                        SpaceAlertType::CommWindow => {
                            // Communication window → opportunity, boost learning
                            self.space_confidence_acc += thresholds::SPACE_COMM_CONFIDENCE * conf;
                            self.space_lr_acc +=
                                thresholds::SPACE_COMM_LR_BOOST as f64 * conf as f64;
                        }
                        SpaceAlertType::OrbitalAnomaly => {
                            // Anomaly → suspicious, elevate vigilance
                            self.anomaly_streak = self.anomaly_streak.saturating_add(1);
                            self.space_arousal_acc += thresholds::SPACE_ANOMALY_AROUSAL * sev;
                        }
                        SpaceAlertType::ManeuverAnnounced => {
                            // Maneuver announced → neutral information update
                            self.space_confidence_acc +=
                                thresholds::SPACE_MANEUVER_CONFIDENCE * conf;
                        }
                    }
                }

                // Waste/circular economy events — track material flows.
                #[cfg(feature = "circular")]
                SwarmEvent::WasteCircularityUpdate {
                    quantity_kg,
                    circularity_potential,
                    confidence,
                    ..
                } => {
                    let qty = if quantity_kg.is_finite() {
                        quantity_kg.max(0.0)
                    } else {
                        0.0
                    };
                    let circ = if circularity_potential.is_finite() {
                        circularity_potential.clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let conf = if confidence.is_finite() {
                        confidence.clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    self.waste_total_kg += qty;
                    self.waste_events_processed += 1;
                    // Running mean circularity
                    let n = self.waste_events_processed as f32;
                    self.waste_mean_circularity += (circ - self.waste_mean_circularity) / n;
                    // Confidence EMA (alpha = 0.1)
                    self.waste_confidence_ema = self.waste_confidence_ema * 0.9 + conf as f64 * 0.1;
                }

                SwarmEvent::DistressSignal {
                    prediction_error,
                    energy_depletion,
                    safety_level,
                    allostatic_load,
                    consciousness_level,
                    ..
                } => {
                    let pe = if prediction_error.is_finite() {
                        prediction_error.clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let energy = if energy_depletion.is_finite() {
                        energy_depletion.clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let load = if allostatic_load.is_finite() {
                        allostatic_load.clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let severity = energy * 0.4
                        + load * 0.3
                        + pe * 0.2
                        + (1.0 - consciousness_level.clamp(0.0, 1.0)) * 0.1;
                    if safety_level >= 3 && energy > 0.9 {
                        self.affective_arousal_acc += 0.15 * severity as f64;
                        self.affective_valence_acc -= 0.1 * severity as f64;
                    } else if safety_level >= 2 || energy > 0.7 {
                        self.affective_arousal_acc += 0.05 * severity as f64;
                        self.affective_valence_acc -= 0.05 * severity as f64;
                    }
                    self.distress_signals_received += 1;
                }
                #[cfg(feature = "hypervisor")]
                SwarmEvent::SupervisoryHeartbeat {
                    supervisor_id,
                    aggregated_phi,
                    ..
                } => {
                    let phi = if aggregated_phi.is_finite() {
                        aggregated_phi.clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let virt_id = format!("supervisor:{supervisor_id}");
                    if let Some(entry) = self.peer_phi.iter_mut().find(|(id, _)| id == &virt_id) {
                        entry.1 = phi;
                    } else if self.peer_phi.len() < Self::MAX_TRACKED_PEERS {
                        self.peer_phi.push((virt_id, phi));
                    }
                }
                #[cfg(feature = "hypervisor")]
                SwarmEvent::SupervisoryEscalation { severity, .. } => {
                    let sev = if severity.is_finite() {
                        severity.clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    self.affective_arousal_acc += 0.1 * sev as f64;
                    self.affective_valence_acc -= 0.05 * sev as f64;
                }
                #[cfg(feature = "hypervisor")]
                SwarmEvent::SupervisorElection { .. } => {}
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
            navigation_ready_peers: self.navigation_ready_peers(),
            mean_peer_navigation_sigma_m: self.mean_peer_navigation_sigma_m(),
            federated_confidence: self.federated_confidence,
            anomaly_count: self.anomaly_streak,
            verified_peers: self.verified_peers,
            coalition_count: self.active_coalitions.len(),
            conscious_collective_count: self.conscious_collective_count(),
            #[cfg(feature = "space-alerts")]
            space_alert_count: self.space_alert_count,
            #[cfg(feature = "space-alerts")]
            last_space_alert_severity: self.last_space_alert_severity,
            #[cfg(feature = "circular")]
            waste_total_kg: self.waste_total_kg,
            #[cfg(feature = "circular")]
            waste_mean_circularity: self.waste_mean_circularity,
            #[cfg(feature = "circular")]
            waste_events_processed: self.waste_events_processed,
            #[cfg(feature = "circular")]
            waste_confidence_ema: self.waste_confidence_ema,
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

        // ── 8. Space alerts → arousal/valence/confidence/exploration ────
        #[cfg(feature = "space-alerts")]
        {
            output.arousal_delta += self.space_arousal_acc;
            output.valence_delta += self.space_valence_acc;
            output.confidence_delta += self.space_confidence_acc as f64;
            output.exploration_delta += self.space_exploration_acc;
            if self.space_lr_acc > 0.0 {
                output.lr_modulation *= 1.0 + self.space_lr_acc;
            }
            // Reset accumulators
            self.space_arousal_acc = 0.0;
            self.space_valence_acc = 0.0;
            self.space_confidence_acc = 0.0;
            self.space_exploration_acc = 0.0;
            self.space_lr_acc = 0.0;
        }

        // ── 9. Coalition identification (Friston 2013) ──────────────────
        self.identify_coalitions();
        let conscious_count = self.conscious_collective_count();
        if conscious_count > 0 {
            output.confidence_delta +=
                (conscious_count as f64 * 0.02).min(thresholds::SWARM_SOCIAL_BUFFERING_CAP);
        }

        // ── 10. Update telemetry ────────────────────────────────────────
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

/// Convert a peer navigation estimate into a cognitive-layer swarm event.
pub fn convert_navigation_estimate(
    peer_id: &str,
    estimate: &positioning::PeerEstimate3D,
) -> SwarmEvent {
    SwarmEvent::PeerNavigationUpdate {
        peer_id: peer_id.to_string(),
        position_m: estimate.estimate.mean,
        position_sigma_m: estimate.estimate.uncertainty(),
        confidence: Some(estimate.confidence),
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
    fn test_navigation_updates_populate_telemetry() {
        let mut sm = SwarmManager::default();
        sm.inject_event(SwarmEvent::PeerNavigationUpdate {
            peer_id: "nav-peer".into(),
            position_m: [10.0, 0.0, 0.0],
            position_sigma_m: 4.0,
            confidence: Some(0.8),
        });

        sm.process(&CycleSnapshot::default());

        let telem = sm.telemetry();
        assert_eq!(telem.navigation_ready_peers, 1);
        assert!((telem.mean_peer_navigation_sigma_m - 4.0).abs() < 1e-6);
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

    #[test]
    fn test_convert_navigation_estimate() {
        let estimate = positioning::PeerEstimate3D {
            peer_id: "peer-9".into(),
            estimate: positioning::GaussianEstimate3D::from_diagonal_sigma([1.0, 2.0, 3.0], 6.0),
            confidence: 0.7,
            trust_weight: 1.0,
            timestamp_us: 0,
        };

        match convert_navigation_estimate("peer-9", &estimate) {
            SwarmEvent::PeerNavigationUpdate {
                peer_id,
                position_m,
                position_sigma_m,
                confidence,
            } => {
                assert_eq!(peer_id, "peer-9");
                assert_eq!(position_m, [1.0, 2.0, 3.0]);
                assert!((position_sigma_m - 6.0).abs() < 1e-6);
                assert_eq!(confidence, Some(0.7));
            }
            other => panic!("Expected PeerNavigationUpdate, got {:?}", other),
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

    // ── Space alert tests ───────────────────────────────────────────────

    #[cfg(feature = "space-alerts")]
    #[test]
    fn test_space_alert_conjunction_modulates_arousal() {
        let mut sm = SwarmManager::default();
        sm.connected_peers = 1;
        sm.connectivity_ema = 0.5;

        sm.inject_event(SwarmEvent::SpaceAlert {
            alert_type: SpaceAlertType::ConjunctionWarning,
            severity: 0.8,
            confidence: 0.9,
            time_to_event_seconds: 60.0,
            description: "Close approach predicted".into(),
        });

        let snapshot = CycleSnapshot::default();
        let output = sm.process(&snapshot);

        assert!(
            output.arousal_delta > 0.0,
            "Conjunction should boost arousal: {}",
            output.arousal_delta
        );
        assert!(
            output.exploration_delta > 0.0,
            "Conjunction should boost exploration: {}",
            output.exploration_delta
        );
    }

    #[cfg(feature = "space-alerts")]
    #[test]
    fn test_space_alert_debris_negative_valence() {
        let mut sm = SwarmManager::default();
        sm.connected_peers = 1;
        sm.connectivity_ema = 0.5;

        sm.inject_event(SwarmEvent::SpaceAlert {
            alert_type: SpaceAlertType::DebrisProximity,
            severity: 0.7,
            confidence: 0.85,
            time_to_event_seconds: 300.0,
            description: "Debris fragment approaching".into(),
        });

        let snapshot = CycleSnapshot::default();
        let output = sm.process(&snapshot);

        assert!(
            output.valence_delta < 0.0,
            "Debris should induce negative valence: {}",
            output.valence_delta
        );
        assert!(
            output.arousal_delta > 0.0,
            "Debris should boost arousal: {}",
            output.arousal_delta
        );
        // Confidence should decrease from debris uncertainty
        // (note: social buffering adds positive confidence, so check net effect
        // by comparing with the social buffering contribution alone)
        let mut sm_no_alert = SwarmManager::default();
        sm_no_alert.connected_peers = 1;
        sm_no_alert.connectivity_ema = 0.5;
        let baseline = sm_no_alert.process(&snapshot);
        assert!(
            output.confidence_delta < baseline.confidence_delta,
            "Debris should reduce confidence relative to baseline: {} vs {}",
            output.confidence_delta,
            baseline.confidence_delta
        );
    }

    #[cfg(feature = "space-alerts")]
    #[test]
    fn test_space_alert_comm_window_boosts_confidence() {
        let mut sm = SwarmManager::default();
        sm.connected_peers = 1;
        sm.connectivity_ema = 0.5;

        sm.inject_event(SwarmEvent::SpaceAlert {
            alert_type: SpaceAlertType::CommWindow,
            severity: 0.3,
            confidence: 0.95,
            time_to_event_seconds: 120.0,
            description: "Relay satellite in range".into(),
        });

        let snapshot = CycleSnapshot::default();
        let output = sm.process(&snapshot);

        assert!(
            output.confidence_delta > 0.0,
            "Comm window should boost confidence: {}",
            output.confidence_delta
        );
        assert!(
            output.lr_modulation > 1.0,
            "Comm window should boost learning rate: {}",
            output.lr_modulation
        );
    }

    #[cfg(feature = "space-alerts")]
    #[test]
    fn test_space_alert_urgency_decay() {
        let mut sm_near = SwarmManager::default();
        sm_near.connected_peers = 1;
        sm_near.connectivity_ema = 0.5;

        sm_near.inject_event(SwarmEvent::SpaceAlert {
            alert_type: SpaceAlertType::ConjunctionWarning,
            severity: 0.8,
            confidence: 0.9,
            time_to_event_seconds: 10.0, // Very close
            description: "Imminent".into(),
        });

        let mut sm_far = SwarmManager::default();
        sm_far.connected_peers = 1;
        sm_far.connectivity_ema = 0.5;

        sm_far.inject_event(SwarmEvent::SpaceAlert {
            alert_type: SpaceAlertType::ConjunctionWarning,
            severity: 0.8,
            confidence: 0.9,
            time_to_event_seconds: 36000.0, // 10 hours away
            description: "Distant".into(),
        });

        let snapshot = CycleSnapshot::default();
        let out_near = sm_near.process(&snapshot);
        let out_far = sm_far.process(&snapshot);

        assert!(
            out_near.arousal_delta > out_far.arousal_delta,
            "Near event should produce higher arousal than far: {} vs {}",
            out_near.arousal_delta,
            out_far.arousal_delta
        );
    }

    #[cfg(feature = "space-alerts")]
    #[test]
    fn test_space_alert_severity_clamped() {
        let mut sm = SwarmManager::default();
        sm.connected_peers = 1;
        sm.connectivity_ema = 0.5;

        // Inject with out-of-range values
        sm.inject_event(SwarmEvent::SpaceAlert {
            alert_type: SpaceAlertType::DebrisProximity,
            severity: 5.0,        // Over max
            confidence: f32::NAN, // NaN
            time_to_event_seconds: f64::INFINITY,
            description: "Bad data".into(),
        });

        let snapshot = CycleSnapshot::default();
        let output = sm.process(&snapshot);

        assert!(
            output.arousal_delta.is_finite(),
            "NaN/Inf inputs should produce finite arousal: {}",
            output.arousal_delta
        );
        assert!(
            output.valence_delta.is_finite(),
            "NaN/Inf inputs should produce finite valence: {}",
            output.valence_delta
        );
        assert!(
            output.confidence_delta.is_finite(),
            "NaN/Inf inputs should produce finite confidence: {}",
            output.confidence_delta
        );

        // Severity should have been clamped to 1.0
        assert_eq!(
            sm.last_space_alert_severity, 1.0,
            "Severity should be clamped to 1.0"
        );
    }
}
