//! # Radio Dispatcher — Multi-Band Tiered Radio Architecture
//!
//! Provides bandwidth-aware routing for Symthaea's mesh consciousness network
//! across heterogeneous radio tiers (Wi-Fi/BLE → LoRa/UHF → HF/NVIS).
//!
//! ## Five Capabilities
//!
//! 1. **Radio-Aware Bandwidth Profiles** — Per-tier AIMD parameters (MTU, duty cycle, latency)
//! 2. **Semantic Delta Compression** — XOR + RLE on BinaryHV diffs (2,048B → ~50-100B typical)
//! 3. **Payload Triage** — Urgency × size × tier routing via `PayloadClassifier`
//! 4. **Network Degradation → Safety** — Connectivity loss escalates `SafetyAgent` levels
//! 5. **Spectrum Manager** — SDR-as-sensory-modality `CognitiveSubsystem` (interval 53)
//!
//! ## Scientific Basis
//!
//! - Shannon (1948): Channel capacity theorem constrains information flow per tier
//! - Jacobson (1988): AIMD congestion control adapted for heterogeneous radio links
//! - Clark & Chalmers (1998): Extended mind thesis — radio network as cognitive extension
//! - Friston (2010): Active Inference over electromagnetic spectrum (prediction error = jamming)

use super::super::subsystem_trait::{
    output_flags, CognitiveSubsystem, CycleSnapshot, SubsystemOutput,
};
use std::collections::VecDeque;

// ═══════════════════════════════════════════════════════════════════════════════
// RADIO TIER — Physical layer abstraction
// ═══════════════════════════════════════════════════════════════════════════════

/// Physical radio tier corresponding to range/bandwidth trade-offs.
///
/// Each tier maps to a class of radio hardware with distinct constraints.
/// The cognitive loop selects tiers dynamically based on payload size,
/// urgency, and available connectivity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RadioTier {
    /// Wi-Fi mesh (802.11s) / BLE — high bandwidth, short range (~100m).
    /// Use for: full BinaryHV sync, DHT updates, rich data exchange.
    Local,
    /// LoRa (868/915 MHz) / UHF packet radio — low bandwidth, medium range (~15km).
    /// Use for: compressed consciousness vectors, alerts, peer discovery.
    Metro,
    /// HF NVIS / JS8Call — ultra-low bandwidth, over-the-horizon (~300km+).
    /// Use for: emergency alerts, cryptographic proofs, consensus heartbeats.
    Regional,
}

impl RadioTier {
    /// All tiers in descending bandwidth order.
    pub const ALL: [RadioTier; 3] = [RadioTier::Local, RadioTier::Metro, RadioTier::Regional];
}

/// Physical characteristics of a radio tier.
///
/// These parameters drive AIMD bandwidth control and payload routing decisions.
/// Values are based on real-world radio specifications:
/// - Local: 802.11s mesh (effective ~10 Mbps after overhead)
/// - Metro: LoRa SF7-SF12 at 125kHz (250 bps to 50 kbps, duty-cycle limited)
/// - Regional: JS8Call over HF NVIS (~50 bps effective, ionospheric variability)
#[derive(Debug, Clone, Copy)]
pub struct TierProfile {
    /// Maximum transmission unit in bytes.
    pub mtu: usize,
    /// AIMD bandwidth budget per 10-second window (bytes).
    pub bandwidth_budget: u64,
    /// Minimum AIMD budget floor (bytes).
    pub bandwidth_min: u64,
    /// Maximum AIMD budget ceiling (bytes).
    pub bandwidth_max: u64,
    /// AIMD additive increase per healthy window (bytes).
    pub additive_increase: u64,
    /// AIMD multiplicative decrease factor on congestion.
    pub decrease_factor: f64,
    /// Regulatory duty cycle limit (1.0 = unlimited, 0.01 = 1% LoRa EU).
    pub duty_cycle: f32,
    /// Expected one-way latency in milliseconds.
    pub latency_ms: u32,
    /// Expected packet delivery reliability (0.0–1.0).
    pub reliability: f32,
}

impl RadioTier {
    /// Get the physical profile for this tier.
    pub fn profile(self) -> TierProfile {
        match self {
            RadioTier::Local => TierProfile {
                mtu: 1500,
                bandwidth_budget: 1_000_000,    // 1 MB per 10s window
                bandwidth_min: 100_000,          // 100 KB floor
                bandwidth_max: 10_000_000,       // 10 MB ceiling
                additive_increase: 100_000,      // +100 KB per healthy window
                decrease_factor: 0.5,
                duty_cycle: 1.0,
                latency_ms: 5,
                reliability: 0.99,
            },
            RadioTier::Metro => TierProfile {
                mtu: 250,
                bandwidth_budget: 2_500,         // 2.5 KB per 10s (LoRa 1% duty)
                bandwidth_min: 500,              // 500 B floor
                bandwidth_max: 25_000,           // 25 KB ceiling
                additive_increase: 250,          // +250 B per healthy window
                decrease_factor: 0.5,
                duty_cycle: 0.01,                // 1% EU LoRa duty cycle
                latency_ms: 500,
                reliability: 0.85,
            },
            RadioTier::Regional => TierProfile {
                mtu: 50,
                bandwidth_budget: 250,           // 250 B per 10s (~50 bps)
                bandwidth_min: 50,               // 50 B floor
                bandwidth_max: 1_000,            // 1 KB ceiling
                additive_increase: 25,           // +25 B per healthy window
                decrease_factor: 0.5,
                duty_cycle: 0.5,                 // 50% (amateur HF)
                latency_ms: 5_000,
                reliability: 0.70,
            },
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DELTA COMPRESSOR — XOR + RLE for BinaryHV diffs
// ═══════════════════════════════════════════════════════════════════════════════

/// Compressed delta between two BinaryHV vectors.
///
/// Uses XOR to find changed bits, then run-length encodes the result.
/// For incremental cognitive state updates where few dimensions flip per cycle,
/// this typically compresses 2,048 bytes → 50-200 bytes.
#[derive(Debug, Clone)]
pub struct CompressedDelta {
    /// RLE-encoded XOR diff. Format: repeated (count: u16 LE, byte: u8) triples.
    /// A run of zeros means "unchanged", a run of non-zero means "these bits flipped".
    pub rle_data: Vec<u8>,
    /// Number of bytes that differ (for telemetry).
    pub changed_bytes: usize,
    /// Whether this is a full vector (not a delta) — used on reconnect.
    pub is_full: bool,
}

impl CompressedDelta {
    /// Compute XOR delta between two BinaryHV byte arrays, then RLE-compress.
    ///
    /// If the compressed result would be larger than the raw vector (high entropy),
    /// returns a full vector instead.
    pub fn from_diff(previous: &[u8; 2048], current: &[u8; 2048]) -> Self {
        // XOR to find changed bits
        let mut diff = [0u8; 2048];
        let mut changed_bytes = 0usize;
        for i in 0..2048 {
            diff[i] = previous[i] ^ current[i];
            if diff[i] != 0 {
                changed_bytes += 1;
            }
        }

        let rle_data = Self::rle_encode(&diff);

        // If RLE is larger than raw, just send full vector
        if rle_data.len() >= 2048 {
            return Self::full(current);
        }

        Self {
            rle_data,
            changed_bytes,
            is_full: false,
        }
    }

    /// Create a full (non-delta) compressed payload for initial sync or reconnect.
    pub fn full(data: &[u8; 2048]) -> Self {
        Self {
            rle_data: data.to_vec(),
            changed_bytes: 2048,
            is_full: true,
        }
    }

    /// Apply this delta to a previous BinaryHV to reconstruct the current one.
    ///
    /// For full vectors, ignores `previous` and returns the stored data directly.
    pub fn apply(&self, previous: &[u8; 2048]) -> Option<[u8; 2048]> {
        if self.is_full {
            if self.rle_data.len() != 2048 {
                return None;
            }
            let mut result = [0u8; 2048];
            result.copy_from_slice(&self.rle_data);
            return Some(result);
        }

        let diff = Self::rle_decode(&self.rle_data)?;
        if diff.len() != 2048 {
            return None;
        }

        let mut result = *previous;
        for i in 0..2048 {
            result[i] ^= diff[i];
        }
        Some(result)
    }

    /// Compressed size in bytes.
    pub fn wire_size(&self) -> usize {
        self.rle_data.len()
    }

    /// Compression ratio (1.0 = no compression, 0.0 = perfectly compressed).
    pub fn compression_ratio(&self) -> f64 {
        self.rle_data.len() as f64 / 2048.0
    }

    /// RLE encode: repeated (count_hi: u8, count_lo: u8, byte: u8) triples.
    /// Count is u16 LE to handle runs up to 65535.
    fn rle_encode(data: &[u8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(data.len() / 4);
        if data.is_empty() {
            return result;
        }

        let mut run_byte = data[0];
        let mut run_len: u16 = 1;

        for &b in &data[1..] {
            if b == run_byte && run_len < u16::MAX {
                run_len += 1;
            } else {
                // Emit run
                result.extend_from_slice(&run_len.to_le_bytes());
                result.push(run_byte);
                run_byte = b;
                run_len = 1;
            }
        }
        // Emit final run
        result.extend_from_slice(&run_len.to_le_bytes());
        result.push(run_byte);

        result
    }

    /// RLE decode: inverse of `rle_encode`.
    fn rle_decode(data: &[u8]) -> Option<Vec<u8>> {
        if data.len() % 3 != 0 {
            return None;
        }

        let mut result = Vec::with_capacity(2048);
        let mut i = 0;
        while i + 2 < data.len() {
            let count = u16::from_le_bytes([data[i], data[i + 1]]) as usize;
            let byte = data[i + 2];
            // Safety cap: prevent OOM from malicious data
            if result.len() + count > 65536 {
                return None;
            }
            result.extend(std::iter::repeat(byte).take(count));
            i += 3;
        }
        Some(result)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PAYLOAD CLASSIFIER — Urgency × Size → Tier routing
// ═══════════════════════════════════════════════════════════════════════════════

/// Classification of a mesh payload for tier routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadClass {
    /// Emergency alert — fits Regional tier (~40 bytes).
    Emergency,
    /// Peer discovery / heartbeat — fits Metro tier (~64 bytes).
    Discovery,
    /// Compressed consciousness delta — fits Metro tier if small enough.
    ConsciousnessDelta,
    /// Affective sync — fits Metro tier (~40 bytes).
    Affective,
    /// Full BinaryHV or large DHT entry — requires Local tier.
    BulkSync,
    /// Federated gradient — requires Local tier.
    Gradient,
}

impl PayloadClass {
    /// Minimum viable radio tier for this payload class.
    pub fn min_tier(self) -> RadioTier {
        match self {
            PayloadClass::Emergency => RadioTier::Regional,
            PayloadClass::Discovery => RadioTier::Metro,
            PayloadClass::Affective => RadioTier::Metro,
            PayloadClass::ConsciousnessDelta => RadioTier::Metro,
            PayloadClass::BulkSync => RadioTier::Local,
            PayloadClass::Gradient => RadioTier::Local,
        }
    }

    /// Whether this class requires Critical urgency to transmit.
    pub fn urgency_gate(self) -> Option<u8> {
        match self {
            PayloadClass::Emergency => Some(2), // Critical only
            _ => None,                          // Any urgency
        }
    }
}

/// Classifies payloads and selects the best available radio tier.
///
/// Routing decision: min_tier(payload_class) ∩ available_tiers ∩ urgency_gate.
/// Falls back to the highest available tier if the ideal tier is unavailable.
pub struct PayloadClassifier {
    /// Which tiers are currently operational.
    available_tiers: [bool; 3], // [Local, Metro, Regional]
}

impl Default for PayloadClassifier {
    fn default() -> Self {
        Self {
            available_tiers: [true, true, true], // All tiers available
        }
    }
}

impl PayloadClassifier {
    /// Update tier availability (called when radio hardware status changes).
    pub fn set_tier_available(&mut self, tier: RadioTier, available: bool) {
        self.available_tiers[tier as usize] = available;
    }

    /// Check if a specific tier is currently available.
    pub fn is_available(&self, tier: RadioTier) -> bool {
        self.available_tiers[tier as usize]
    }

    /// Count of available tiers.
    pub fn available_count(&self) -> usize {
        self.available_tiers.iter().filter(|&&a| a).count()
    }

    /// Classify a payload and select the best tier for transmission.
    ///
    /// Returns `None` if no suitable tier is available (complete radio blackout).
    pub fn route(
        &self,
        class: PayloadClass,
        payload_size: usize,
        urgency: u8,
    ) -> Option<RoutingDecision> {
        // Check urgency gate
        if let Some(min_urgency) = class.urgency_gate() {
            if urgency < min_urgency {
                return Some(RoutingDecision::Deferred {
                    reason: "urgency below gate",
                });
            }
        }

        let min_tier = class.min_tier();

        // Try tiers from min_tier upward (toward higher bandwidth)
        // RadioTier order: Local(0) > Metro(1) > Regional(2)
        // We want to find the best available tier at or above min_tier bandwidth
        let candidates: Vec<RadioTier> = RadioTier::ALL
            .iter()
            .copied()
            .filter(|&t| self.available_tiers[t as usize])
            .filter(|&t| (t as usize) <= (min_tier as usize))
            .collect();

        if candidates.is_empty() {
            // No tier at the required bandwidth level — try any available tier
            // but only if the payload can physically fit
            for &tier in &RadioTier::ALL {
                if self.available_tiers[tier as usize] {
                    let profile = tier.profile();
                    if payload_size <= profile.mtu {
                        return Some(RoutingDecision::Routed {
                            tier,
                            fragmented: false,
                            estimated_fragments: 1,
                        });
                    }
                }
            }
            return Some(RoutingDecision::Blocked {
                reason: "no tier with sufficient MTU available",
            });
        }

        // Select the best candidate (lowest index = highest bandwidth)
        let selected = candidates[0];
        let profile = selected.profile();

        let fragmented = payload_size > profile.mtu;
        let estimated_fragments = if fragmented {
            (payload_size + profile.mtu - 1) / profile.mtu
        } else {
            1
        };

        Some(RoutingDecision::Routed {
            tier: selected,
            fragmented,
            estimated_fragments,
        })
    }
}

/// Result of payload classification and tier selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingDecision {
    /// Payload routed to a specific tier.
    Routed {
        tier: RadioTier,
        fragmented: bool,
        estimated_fragments: usize,
    },
    /// Payload deferred (urgency too low for this class).
    Deferred { reason: &'static str },
    /// No suitable tier available.
    Blocked { reason: &'static str },
}

// ═══════════════════════════════════════════════════════════════════════════════
// NETWORK DEGRADATION — Connectivity → Safety escalation
// ═══════════════════════════════════════════════════════════════════════════════

/// Network health state derived from tier availability and packet loss.
///
/// Maps directly to SafetyAgent escalation levels:
/// - AllTiersUp → Green (full swarm sync)
/// - LocalDown → Yellow (consciousness vectors + alerts only)
/// - MetroOnly → Orange (emergency alerts + keepalive)
/// - Blackout → Red (autonomous mode, local-only cognition)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum NetworkHealth {
    /// All tiers operational.
    AllTiersUp,
    /// Local (Wi-Fi) down, Metro + Regional up.
    LocalDown,
    /// Only Metro (LoRa) available.
    MetroOnly,
    /// All radio down — autonomous mode.
    Blackout,
}

impl NetworkHealth {
    /// Derive from tier availability.
    pub fn from_tiers(local: bool, metro: bool, regional: bool) -> Self {
        match (local, metro, regional) {
            (true, _, _) => NetworkHealth::AllTiersUp,
            (false, true, _) | (false, false, true) => {
                if metro {
                    NetworkHealth::LocalDown
                } else {
                    NetworkHealth::MetroOnly
                }
            }
            (false, false, false) => NetworkHealth::Blackout,
        }
    }

    /// Suggested safety escalation level for this network state.
    ///
    /// This is advisory — the SafetyAgent makes the final determination
    /// by combining network health with consciousness metrics.
    pub fn safety_suggestion(self) -> u8 {
        match self {
            NetworkHealth::AllTiersUp => 0,  // Green
            NetworkHealth::LocalDown => 1,   // Yellow
            NetworkHealth::MetroOnly => 2,   // Orange
            NetworkHealth::Blackout => 3,    // Red
        }
    }

    /// Epistemic confidence discount for network degradation.
    ///
    /// When the network is degraded, peer consensus is less reliable,
    /// so epistemic confidence should be discounted.
    ///
    /// Basis: Woolley et al. (2010) — collective intelligence requires
    /// social sensitivity and equal contribution, both of which degrade
    /// with reduced connectivity.
    pub fn epistemic_discount(self) -> f64 {
        match self {
            NetworkHealth::AllTiersUp => 1.0,
            NetworkHealth::LocalDown => 0.85,
            NetworkHealth::MetroOnly => 0.6,
            NetworkHealth::Blackout => 0.3,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SPECTRUM MANAGER — SDR as Cognitive Subsystem
// ═══════════════════════════════════════════════════════════════════════════════

/// Spectrum occupancy observation from SDR hardware.
///
/// Injected by the SDR driver layer (when available) to inform
/// frequency selection and jamming detection.
#[derive(Debug, Clone)]
pub struct SpectrumObservation {
    /// Center frequency in Hz.
    pub frequency_hz: u64,
    /// Observed noise floor in dBm.
    pub noise_floor_dbm: f32,
    /// Signal-to-noise ratio in dB (negative = jammed).
    pub snr_db: f32,
    /// Whether this band appears jammed (SNR below threshold).
    pub jammed: bool,
}

/// Regulatory constraints for autonomous frequency selection.
///
/// Hard-gates the MCTS planner — the system MUST NOT hop to
/// frequencies outside these bounds regardless of prediction error.
///
/// Basis: FCC Part 15/97, ETSI EN 300 220 — legal operation boundaries.
#[derive(Debug, Clone)]
pub struct RegulatoryConstraints {
    /// Allowed frequency bands as (low_hz, high_hz) ranges.
    pub allowed_bands: Vec<(u64, u64)>,
    /// Maximum transmit power in dBm.
    pub max_power_dbm: f32,
    /// Region code (e.g., "US", "EU", "JP") for duty cycle rules.
    pub region: String,
}

impl Default for RegulatoryConstraints {
    fn default() -> Self {
        Self {
            // ISM bands (US FCC Part 15)
            allowed_bands: vec![
                (902_000_000, 928_000_000),   // 915 MHz ISM (US)
                (2_400_000_000, 2_500_000_000), // 2.4 GHz ISM
            ],
            max_power_dbm: 30.0, // 1W ERP
            region: "US".to_string(),
        }
    }
}

impl RegulatoryConstraints {
    /// EU regulatory profile (868 MHz + 2.4 GHz ISM).
    pub fn eu() -> Self {
        Self {
            allowed_bands: vec![
                (863_000_000, 870_000_000),   // 868 MHz ISM (EU)
                (2_400_000_000, 2_500_000_000), // 2.4 GHz ISM
            ],
            max_power_dbm: 14.0, // 25 mW ERP
            region: "EU".to_string(),
        }
    }

    /// Check if a frequency is within allowed bands.
    pub fn is_allowed(&self, frequency_hz: u64) -> bool {
        self.allowed_bands
            .iter()
            .any(|&(low, high)| frequency_hz >= low && frequency_hz <= high)
    }
}

/// Spectrum Manager — electromagnetic environment as sensory modality.
///
/// Implements `CognitiveSubsystem` at interval 53 (co-prime with all existing
/// manager intervals: 7, 11, 13, 19, 29, 37, 41).
///
/// Treats spectrum congestion as prediction error that drives frequency hopping
/// via the FEP Active Inference engine.
///
/// ## Signals Modeled
///
/// 1. **Bandwidth pressure**: Available capacity → confidence modulation
/// 2. **Jamming detection**: SNR collapse → arousal spike + exploration (frequency search)
/// 3. **Tier degradation**: Radio failures → safety escalation proposal
/// 4. **Spectrum prediction error**: Expected vs observed noise floor → surprise
pub struct SpectrumManager {
    // ── Tier state ───────────────────────────────────────────────────────
    /// Current tier availability.
    tier_available: [bool; 3],
    /// Per-tier packet loss EMA (0.0–1.0, lower is better).
    tier_loss_ema: [f64; 3],
    /// Per-tier current AIMD budget (bytes per window).
    tier_budget: [u64; 3],

    // ── Classifier ───────────────────────────────────────────────────────
    classifier: PayloadClassifier,

    // ── Spectrum state ───────────────────────────────────────────────────
    /// Pending spectrum observations from SDR (drained each process cycle).
    pending_observations: Vec<SpectrumObservation>,
    /// Jamming detection: consecutive cycles with jammed bands.
    jamming_streak: u32,
    /// Predicted noise floor EMA (for prediction error computation).
    predicted_noise_floor: f64,
    /// Regulatory constraints (hard gate on frequency selection).
    regulatory: RegulatoryConstraints,

    // ── Network health ───────────────────────────────────────────────────
    /// Current aggregate network health.
    network_health: NetworkHealth,
    /// Consecutive cycles at degraded health (for safety trend detection).
    degradation_streak: u32,

    // ── Delta compression state ──────────────────────────────────────────
    /// Last transmitted BinaryHV per peer (for delta computation).
    /// Keyed by first 8 bytes of source_id.
    peer_last_hv: Vec<([u8; 8], [u8; 2048])>,

    // ── Telemetry ────────────────────────────────────────────────────────
    last_telemetry: SpectrumTelemetry,
}

/// Telemetry snapshot for the Spectrum Manager.
#[derive(Debug, Clone, Default)]
pub struct SpectrumTelemetry {
    /// Current network health level.
    pub network_health: u8, // 0=AllUp, 1=LocalDown, 2=MetroOnly, 3=Blackout
    /// Per-tier availability [Local, Metro, Regional].
    pub tier_available: [bool; 3],
    /// Per-tier packet loss EMA.
    pub tier_loss_ema: [f64; 3],
    /// Consecutive jamming cycles.
    pub jamming_streak: u32,
    /// Spectrum prediction error (0.0–1.0).
    pub spectrum_prediction_error: f64,
    /// Average delta compression ratio across recent transmissions.
    pub avg_delta_compression: f64,
    /// Epistemic discount from network degradation.
    pub epistemic_discount: f64,
    /// Number of degradation streak cycles.
    pub degradation_streak: u32,
}

impl Default for SpectrumManager {
    fn default() -> Self {
        let profiles: Vec<TierProfile> = RadioTier::ALL.iter().map(|t| t.profile()).collect();
        Self {
            tier_available: [true, true, true],
            tier_loss_ema: [0.0; 3],
            tier_budget: [
                profiles[0].bandwidth_budget,
                profiles[1].bandwidth_budget,
                profiles[2].bandwidth_budget,
            ],
            classifier: PayloadClassifier::default(),
            pending_observations: Vec::with_capacity(16),
            jamming_streak: 0,
            predicted_noise_floor: -100.0, // dBm, typical quiet band
            regulatory: RegulatoryConstraints::default(),
            network_health: NetworkHealth::AllTiersUp,
            degradation_streak: 0,
            peer_last_hv: Vec::new(),
            last_telemetry: SpectrumTelemetry::default(),
        }
    }
}

// Re-export named constants from thresholds.rs for local use.
use super::super::thresholds::{
    RADIO_DEGRADATION_CONFIDENCE_DROP as DEGRADATION_CONFIDENCE_DROP,
    RADIO_JAMMING_AROUSAL_SPIKE as JAMMING_AROUSAL_SPIKE,
    RADIO_JAMMING_EXPLORATION_BOOST as JAMMING_EXPLORATION_BOOST,
    RADIO_JAMMING_SNR_THRESHOLD as JAMMING_SNR_THRESHOLD,
    RADIO_MAX_DELTA_PEERS as MAX_DELTA_PEERS,
    RADIO_NOISE_FLOOR_EMA_ALPHA as NOISE_FLOOR_EMA_ALPHA,
    RADIO_TIER_DEGRADED_LOSS as TIER_DEGRADED_LOSS,
    RADIO_TIER_LOSS_EMA_ALPHA as TIER_LOSS_EMA_ALPHA,
};

impl SpectrumManager {
    /// Create with specific regulatory constraints.
    pub fn with_regulatory(regulatory: RegulatoryConstraints) -> Self {
        Self {
            regulatory,
            ..Self::default()
        }
    }

    // ── Public API ───────────────────────────────────────────────────────

    /// Inject a spectrum observation from SDR hardware.
    pub fn inject_observation(&mut self, obs: SpectrumObservation) {
        self.pending_observations.push(obs);
    }

    /// Report a packet loss event on a specific tier.
    pub fn report_loss(&mut self, tier: RadioTier) {
        let idx = tier as usize;
        self.tier_loss_ema[idx] =
            self.tier_loss_ema[idx] * (1.0 - TIER_LOSS_EMA_ALPHA) + TIER_LOSS_EMA_ALPHA;
    }

    /// Report a successful packet delivery on a specific tier.
    pub fn report_success(&mut self, tier: RadioTier) {
        let idx = tier as usize;
        self.tier_loss_ema[idx] *= 1.0 - TIER_LOSS_EMA_ALPHA;
    }

    /// Set tier availability (called by radio hardware monitor).
    pub fn set_tier_available(&mut self, tier: RadioTier, available: bool) {
        let idx = tier as usize;
        self.tier_available[idx] = available;
        self.classifier.set_tier_available(tier, available);
        self.update_network_health();
    }

    /// Get the current network health.
    pub fn network_health(&self) -> NetworkHealth {
        self.network_health
    }

    /// Get the payload classifier for routing decisions.
    pub fn classifier(&self) -> &PayloadClassifier {
        &self.classifier
    }

    /// Get the current telemetry snapshot.
    pub fn telemetry(&self) -> &SpectrumTelemetry {
        &self.last_telemetry
    }

    /// Compute a compressed delta for a BinaryHV relative to a peer's last state.
    ///
    /// If no previous state exists for this peer, returns a full vector.
    /// Updates the peer's last-known state after compression.
    pub fn compress_delta(
        &mut self,
        peer_id: &[u8; 8],
        current_hv: &[u8; 2048],
    ) -> CompressedDelta {
        // Find peer's last HV
        let previous = self
            .peer_last_hv
            .iter()
            .find(|(id, _)| id == peer_id)
            .map(|(_, hv)| *hv);

        let delta = match previous {
            Some(prev) => CompressedDelta::from_diff(&prev, current_hv),
            None => CompressedDelta::full(current_hv),
        };

        // Update peer's last-known state
        if let Some(entry) = self.peer_last_hv.iter_mut().find(|(id, _)| id == peer_id) {
            entry.1 = *current_hv;
        } else if self.peer_last_hv.len() < MAX_DELTA_PEERS {
            self.peer_last_hv.push((*peer_id, *current_hv));
        }

        delta
    }

    /// Route a payload through the classifier.
    pub fn route(
        &self,
        class: PayloadClass,
        payload_size: usize,
        urgency: u8,
    ) -> Option<RoutingDecision> {
        self.classifier.route(class, payload_size, urgency)
    }

    /// Get the regulatory constraints.
    pub fn regulatory(&self) -> &RegulatoryConstraints {
        &self.regulatory
    }

    /// Total available bandwidth across all operational tiers (bytes per 10s window).
    ///
    /// Returns only the budgets of tiers that are currently available.
    /// Used by Broca cadence throttling to limit speech when bandwidth is low.
    pub fn available_bandwidth(&self) -> u64 {
        self.tier_budget
            .iter()
            .zip(self.tier_available.iter())
            .filter(|(_, &avail)| avail)
            .map(|(&budget, _)| budget)
            .sum()
    }

    /// Per-tier AIMD bandwidth budgets [Local, Metro, Regional].
    ///
    /// Returns the current budgets regardless of availability;
    /// consumers should check tier availability separately.
    pub fn tier_budgets(&self) -> &[u64; 3] {
        &self.tier_budget
    }

    /// Best available tier for governance traffic (highest reliability).
    ///
    /// Returns the most reliable operational tier for critical governance
    /// messages (votes, proposals). Governance traffic should prefer
    /// tiers with highest reliability even at lower bandwidth.
    pub fn best_tier_for_governance(&self) -> Option<RadioTier> {
        // Prefer Local > Metro > Regional (reliability ordering)
        for &tier in &RadioTier::ALL {
            if self.tier_available[tier as usize] {
                return Some(tier);
            }
        }
        None
    }

    // ── Internal ─────────────────────────────────────────────────────────

    fn update_network_health(&mut self) {
        let new_health = NetworkHealth::from_tiers(
            self.tier_available[0],
            self.tier_available[1],
            self.tier_available[2],
        );

        if new_health > NetworkHealth::AllTiersUp {
            self.degradation_streak = self.degradation_streak.saturating_add(1);
        } else {
            self.degradation_streak = 0;
        }

        self.network_health = new_health;
    }

    fn process_observations(&mut self) -> f64 {
        let observations = std::mem::take(&mut self.pending_observations);
        if observations.is_empty() {
            return 0.0;
        }

        let mut jammed_count = 0u32;
        let mut total_noise = 0.0f64;

        for obs in &observations {
            if obs.jammed || obs.snr_db < JAMMING_SNR_THRESHOLD {
                jammed_count += 1;
            }
            total_noise += obs.noise_floor_dbm as f64;
        }

        let mean_noise = total_noise / observations.len() as f64;

        // Update jamming streak
        if jammed_count > 0 {
            self.jamming_streak = self.jamming_streak.saturating_add(1);
        } else {
            self.jamming_streak = 0;
        }

        // Compute spectrum prediction error
        let noise_error = ((mean_noise - self.predicted_noise_floor).abs() / 50.0).min(1.0);

        // Update predicted noise floor via EMA
        self.predicted_noise_floor = self.predicted_noise_floor * (1.0 - NOISE_FLOOR_EMA_ALPHA)
            + mean_noise * NOISE_FLOOR_EMA_ALPHA;

        noise_error
    }

    fn update_telemetry(&mut self, spectrum_pe: f64) {
        self.last_telemetry = SpectrumTelemetry {
            network_health: self.network_health.safety_suggestion(),
            tier_available: self.tier_available,
            tier_loss_ema: self.tier_loss_ema,
            jamming_streak: self.jamming_streak,
            spectrum_prediction_error: spectrum_pe,
            avg_delta_compression: 0.0, // Updated externally when deltas are computed
            epistemic_discount: self.network_health.epistemic_discount(),
            degradation_streak: self.degradation_streak,
        };
    }

    /// Check if any tier has loss above the degradation threshold.
    fn any_tier_degraded(&self) -> bool {
        self.tier_loss_ema
            .iter()
            .zip(self.tier_available.iter())
            .any(|(&loss, &avail)| avail && loss > TIER_DEGRADED_LOSS)
    }
}

impl CognitiveSubsystem for SpectrumManager {
    fn name(&self) -> &'static str {
        "spectrum_manager"
    }

    fn interval(&self) -> u32 {
        53 // co-prime with 7, 11, 13, 19, 29, 37, 41
    }

    fn process(&mut self, _snapshot: &CycleSnapshot) -> SubsystemOutput {
        let mut output = SubsystemOutput::NEUTRAL;

        // ── 1. Process spectrum observations ─────────────────────────────
        let spectrum_pe = self.process_observations();

        // ── 2. Update network health from tier state ─────────────────────
        self.update_network_health();

        // ── 3. Jamming response ──────────────────────────────────────────
        if self.jamming_streak > 0 {
            // Jamming → arousal spike (threat detection) + exploration (frequency search)
            output.arousal_delta += JAMMING_AROUSAL_SPIKE as f32;
            output.exploration_delta +=
                JAMMING_EXPLORATION_BOOST * self.jamming_streak.min(3) as f64;
            output.flags |= output_flags::ANOMALY_DETECTED;
        }

        // ── 4. Network degradation → confidence/epistemic modulation ─────
        let health_level = self.network_health.safety_suggestion();
        if health_level > 0 {
            // Degraded network → reduced confidence in distributed cognition
            output.confidence_delta -= DEGRADATION_CONFIDENCE_DROP * health_level as f64;
        }

        // Blackout → escalate urgency (autonomous survival mode)
        if self.network_health == NetworkHealth::Blackout {
            output.flags |= output_flags::ESCALATE_URGENCY;
            // Isolation drives exploration (seeking new connections)
            output.exploration_delta += 0.05;
        }

        // ── 5. Tier loss → learning rate dampening ───────────────────────
        if self.any_tier_degraded() {
            // High packet loss → reduce learning rate (unreliable gradients)
            let max_loss = self
                .tier_loss_ema
                .iter()
                .cloned()
                .fold(0.0f64, f64::max);
            output.lr_modulation = 1.0 - (max_loss * 0.2).min(0.15);
        }

        // ── 6. Spectrum prediction error → surprise signal ───────────────
        if spectrum_pe > 0.5 {
            output.arousal_delta += (spectrum_pe as f32 * 0.05).min(0.08);
            output.flags |= output_flags::ANOMALY_DETECTED;
        }

        // ── 7. Update telemetry ──────────────────────────────────────────
        self.update_telemetry(spectrum_pe);

        output
    }

    fn checkpoint(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(64);
        // [tier_available: 3 bytes][tier_loss_ema: 3×f64][jamming_streak: u32]
        // [network_health: u8][degradation_streak: u32][predicted_noise: f64]
        for &a in &self.tier_available {
            data.push(if a { 1 } else { 0 });
        }
        for &loss in &self.tier_loss_ema {
            data.extend_from_slice(&loss.to_le_bytes());
        }
        data.extend_from_slice(&self.jamming_streak.to_le_bytes());
        data.push(self.network_health.safety_suggestion());
        data.extend_from_slice(&self.degradation_streak.to_le_bytes());
        data.extend_from_slice(&self.predicted_noise_floor.to_le_bytes());
        data
    }

    fn restore(&mut self, data: &[u8]) -> Result<(), String> {
        // Minimum: 3 + 24 + 4 + 1 + 4 + 8 = 44 bytes
        if data.len() < 44 {
            return Err(format!(
                "SpectrumManager checkpoint too short: {} < 44",
                data.len()
            ));
        }

        self.tier_available = [data[0] != 0, data[1] != 0, data[2] != 0];
        for i in 0..3 {
            let offset = 3 + i * 8;
            self.tier_loss_ema[i] = f64::from_le_bytes(
                data[offset..offset + 8]
                    .try_into()
                    .map_err(|_| "corrupt tier_loss_ema")?,
            );
        }
        self.jamming_streak = u32::from_le_bytes(
            data[27..31]
                .try_into()
                .map_err(|_| "corrupt jamming_streak")?,
        );
        // byte 31: network_health (reconstructed from tier_available instead)
        self.degradation_streak = u32::from_le_bytes(
            data[32..36]
                .try_into()
                .map_err(|_| "corrupt degradation_streak")?,
        );
        self.predicted_noise_floor = f64::from_le_bytes(
            data[36..44]
                .try_into()
                .map_err(|_| "corrupt predicted_noise")?,
        );

        // Sync classifier with restored tier availability
        for (i, &tier) in RadioTier::ALL.iter().enumerate() {
            self.classifier
                .set_tier_available(tier, self.tier_available[i]);
        }
        // Derive network_health from tiers but preserve the restored degradation_streak
        // (update_network_health would overwrite it based on current state).
        self.network_health = NetworkHealth::from_tiers(
            self.tier_available[0],
            self.tier_available[1],
            self.tier_available[2],
        );

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── RadioTier profiles ───────────────────────────────────────────────

    #[test]
    fn test_tier_profiles_bandwidth_ordering() {
        let local = RadioTier::Local.profile();
        let metro = RadioTier::Metro.profile();
        let regional = RadioTier::Regional.profile();

        assert!(local.bandwidth_budget > metro.bandwidth_budget);
        assert!(metro.bandwidth_budget > regional.bandwidth_budget);
        assert!(local.mtu > metro.mtu);
        assert!(metro.mtu > regional.mtu);
    }

    #[test]
    fn test_tier_profiles_latency_ordering() {
        let local = RadioTier::Local.profile();
        let metro = RadioTier::Metro.profile();
        let regional = RadioTier::Regional.profile();

        assert!(local.latency_ms < metro.latency_ms);
        assert!(metro.latency_ms < regional.latency_ms);
    }

    #[test]
    fn test_tier_profiles_reliability_ordering() {
        let local = RadioTier::Local.profile();
        let metro = RadioTier::Metro.profile();
        let regional = RadioTier::Regional.profile();

        assert!(local.reliability > metro.reliability);
        assert!(metro.reliability > regional.reliability);
    }

    // ── Delta compression ────────────────────────────────────────────────

    #[test]
    fn test_delta_identical_vectors() {
        let v = [0xABu8; 2048];
        let delta = CompressedDelta::from_diff(&v, &v);

        assert_eq!(delta.changed_bytes, 0);
        assert!(!delta.is_full);
        // All zeros → single RLE run of 2048 zeros → 3 bytes
        assert!(delta.wire_size() < 10);
        assert!(delta.compression_ratio() < 0.01);
    }

    #[test]
    fn test_delta_one_byte_changed() {
        let v1 = [0u8; 2048];
        let mut v2 = [0u8; 2048];
        v2[1024] = 0xFF;

        let delta = CompressedDelta::from_diff(&v1, &v2);
        assert_eq!(delta.changed_bytes, 1);
        assert!(!delta.is_full);
        // Should be much smaller than 2048
        assert!(delta.wire_size() < 20);

        // Apply and verify roundtrip
        let reconstructed = delta.apply(&v1).unwrap();
        assert_eq!(reconstructed, v2);
    }

    #[test]
    fn test_delta_full_vector() {
        let v = [0x42u8; 2048];
        let delta = CompressedDelta::full(&v);

        assert!(delta.is_full);
        assert_eq!(delta.wire_size(), 2048);

        let zero = [0u8; 2048];
        let reconstructed = delta.apply(&zero).unwrap();
        assert_eq!(reconstructed, v);
    }

    #[test]
    fn test_delta_roundtrip_sparse() {
        // Simulate two vectors with ~5% difference (sparse changes)
        let v1 = [0u8; 2048];
        let mut v2 = [0u8; 2048];
        // Change every 20th byte (102 changes out of 2048 = ~5%)
        for i in (0..2048).step_by(20) {
            v2[i] = 0xFF;
        }

        let delta = CompressedDelta::from_diff(&v1, &v2);
        let reconstructed = delta.apply(&v1).unwrap();
        assert_eq!(reconstructed, v2);
        // Sparse changes should compress well (mostly zero runs)
        assert!(
            delta.compression_ratio() < 0.8,
            "Sparse delta ratio {} should be < 0.8",
            delta.compression_ratio()
        );
    }

    #[test]
    fn test_delta_high_entropy_falls_back_to_full() {
        // Two completely random vectors — XOR is high entropy, RLE won't help
        let mut v1 = [0u8; 2048];
        let mut v2 = [0u8; 2048];
        for i in 0..2048 {
            v1[i] = (i * 7 + 3) as u8;
            v2[i] = (i * 13 + 11) as u8;
        }

        let delta = CompressedDelta::from_diff(&v1, &v2);
        // May or may not fall back to full depending on RLE expansion
        // But it should always be reconstructable
        let reconstructed = delta.apply(&v1).unwrap();
        assert_eq!(reconstructed, v2);
    }

    // ── Payload classifier ───────────────────────────────────────────────

    #[test]
    fn test_classifier_default_all_available() {
        let c = PayloadClassifier::default();
        assert_eq!(c.available_count(), 3);
    }

    #[test]
    fn test_classifier_emergency_routes_regional() {
        let c = PayloadClassifier::default();
        let decision = c.route(PayloadClass::Emergency, 40, 2).unwrap();
        match decision {
            RoutingDecision::Routed { tier, .. } => {
                // Emergency can go to any tier, but should prefer best available
                assert!(tier == RadioTier::Local || tier == RadioTier::Metro || tier == RadioTier::Regional);
            }
            _ => panic!("Expected Routed"),
        }
    }

    #[test]
    fn test_classifier_emergency_deferred_at_low_urgency() {
        let c = PayloadClassifier::default();
        let decision = c.route(PayloadClass::Emergency, 40, 1).unwrap();
        assert!(matches!(decision, RoutingDecision::Deferred { .. }));
    }

    #[test]
    fn test_classifier_bulk_routes_local() {
        let c = PayloadClassifier::default();
        let decision = c.route(PayloadClass::BulkSync, 2048, 1).unwrap();
        match decision {
            RoutingDecision::Routed { tier, fragmented, .. } => {
                assert_eq!(tier, RadioTier::Local);
                assert!(fragmented); // 2048 > 1500 MTU
            }
            _ => panic!("Expected Routed"),
        }
    }

    #[test]
    fn test_classifier_no_local_blocks_bulk() {
        let mut c = PayloadClassifier::default();
        c.set_tier_available(RadioTier::Local, false);

        let decision = c.route(PayloadClass::BulkSync, 2048, 1).unwrap();
        // Metro MTU is 250, Regional is 50 — neither can handle 2048
        assert!(matches!(decision, RoutingDecision::Blocked { .. }));
    }

    #[test]
    fn test_classifier_small_payload_fits_metro() {
        let c = PayloadClassifier::default();
        let decision = c.route(PayloadClass::ConsciousnessDelta, 100, 1).unwrap();
        match decision {
            RoutingDecision::Routed { tier, fragmented, .. } => {
                // Should route to best available tier (Local has highest bandwidth)
                assert_eq!(tier, RadioTier::Local);
                assert!(!fragmented);
            }
            _ => panic!("Expected Routed"),
        }
    }

    // ── Network health ───────────────────────────────────────────────────

    #[test]
    fn test_network_health_all_up() {
        assert_eq!(
            NetworkHealth::from_tiers(true, true, true),
            NetworkHealth::AllTiersUp
        );
    }

    #[test]
    fn test_network_health_local_down() {
        assert_eq!(
            NetworkHealth::from_tiers(false, true, true),
            NetworkHealth::LocalDown
        );
    }

    #[test]
    fn test_network_health_metro_only() {
        assert_eq!(
            NetworkHealth::from_tiers(false, false, true),
            NetworkHealth::MetroOnly
        );
    }

    #[test]
    fn test_network_health_blackout() {
        assert_eq!(
            NetworkHealth::from_tiers(false, false, false),
            NetworkHealth::Blackout
        );
    }

    #[test]
    fn test_network_health_ordering() {
        assert!(NetworkHealth::AllTiersUp < NetworkHealth::LocalDown);
        assert!(NetworkHealth::LocalDown < NetworkHealth::MetroOnly);
        assert!(NetworkHealth::MetroOnly < NetworkHealth::Blackout);
    }

    #[test]
    fn test_epistemic_discount_monotonic() {
        let healths = [
            NetworkHealth::AllTiersUp,
            NetworkHealth::LocalDown,
            NetworkHealth::MetroOnly,
            NetworkHealth::Blackout,
        ];
        for w in healths.windows(2) {
            assert!(
                w[0].epistemic_discount() >= w[1].epistemic_discount(),
                "{:?} discount should >= {:?} discount",
                w[0],
                w[1]
            );
        }
    }

    // ── Regulatory constraints ───────────────────────────────────────────

    #[test]
    fn test_regulatory_us_default() {
        let reg = RegulatoryConstraints::default();
        assert!(reg.is_allowed(915_000_000));
        assert!(reg.is_allowed(2_450_000_000));
        assert!(!reg.is_allowed(100_000_000)); // Not ISM
    }

    #[test]
    fn test_regulatory_eu() {
        let reg = RegulatoryConstraints::eu();
        assert!(reg.is_allowed(868_000_000));
        assert!(!reg.is_allowed(915_000_000)); // US only
    }

    // ── Spectrum Manager (CognitiveSubsystem) ────────────────────────────

    #[test]
    fn test_spectrum_manager_name_and_interval() {
        let sm = SpectrumManager::default();
        assert_eq!(sm.name(), "spectrum_manager");
        assert_eq!(sm.interval(), 53);
    }

    #[test]
    fn test_spectrum_manager_interval_coprime() {
        let interval = 53u32;
        for other in [7, 11, 13, 19, 23, 29, 37, 41, 47] {
            assert_eq!(
                gcd(interval, other),
                1,
                "53 should be co-prime with {}",
                other
            );
        }
    }

    fn gcd(a: u32, b: u32) -> u32 {
        if b == 0 { a } else { gcd(b, a % b) }
    }

    #[test]
    fn test_spectrum_manager_neutral_without_events() {
        let mut sm = SpectrumManager::default();
        let snapshot = CycleSnapshot::default();
        let output = sm.process(&snapshot);

        // No observations, all tiers up → neutral
        assert_eq!(output.arousal_delta, 0.0);
        assert_eq!(output.confidence_delta, 0.0);
    }

    #[test]
    fn test_spectrum_manager_jamming_response() {
        let mut sm = SpectrumManager::default();
        sm.inject_observation(SpectrumObservation {
            frequency_hz: 915_000_000,
            noise_floor_dbm: -30.0,
            snr_db: -10.0, // Below JAMMING_SNR_THRESHOLD
            jammed: true,
        });

        let snapshot = CycleSnapshot::default();
        let output = sm.process(&snapshot);

        assert!(output.arousal_delta > 0.0, "Jamming should spike arousal");
        assert!(
            output.exploration_delta > 0.0,
            "Jamming should drive exploration"
        );
        assert!(output.flags & output_flags::ANOMALY_DETECTED != 0);
    }

    #[test]
    fn test_spectrum_manager_blackout_escalates() {
        let mut sm = SpectrumManager::default();
        sm.set_tier_available(RadioTier::Local, false);
        sm.set_tier_available(RadioTier::Metro, false);
        sm.set_tier_available(RadioTier::Regional, false);

        let snapshot = CycleSnapshot::default();
        let output = sm.process(&snapshot);

        assert!(
            output.confidence_delta < 0.0,
            "Blackout should drop confidence"
        );
        assert!(output.flags & output_flags::ESCALATE_URGENCY != 0);
        assert_eq!(sm.network_health(), NetworkHealth::Blackout);
    }

    #[test]
    fn test_spectrum_manager_tier_loss_dampens_lr() {
        let mut sm = SpectrumManager::default();
        // Simulate high packet loss on Local tier
        for _ in 0..20 {
            sm.report_loss(RadioTier::Local);
        }

        let snapshot = CycleSnapshot::default();
        let output = sm.process(&snapshot);

        assert!(
            output.lr_modulation < 1.0,
            "High loss should dampen learning: {}",
            output.lr_modulation
        );
    }

    #[test]
    fn test_spectrum_manager_delta_compression() {
        let mut sm = SpectrumManager::default();
        let peer_id = [1u8; 8];

        // First call: full vector (no previous state)
        let hv1 = [0xAA; 2048];
        let delta1 = sm.compress_delta(&peer_id, &hv1);
        assert!(delta1.is_full);

        // Second call: delta (previous state exists)
        let mut hv2 = hv1;
        hv2[100] = 0xBB; // Change one byte
        let delta2 = sm.compress_delta(&peer_id, &hv2);
        assert!(!delta2.is_full);
        assert!(delta2.wire_size() < 100);
    }

    #[test]
    fn test_spectrum_manager_checkpoint_roundtrip() {
        let mut sm = SpectrumManager::default();
        // Set internal state directly (not via set_tier_available to avoid side effects)
        sm.tier_available = [false, true, true];
        sm.jamming_streak = 5;
        sm.degradation_streak = 3;
        sm.predicted_noise_floor = -85.0;
        sm.tier_loss_ema = [0.1, 0.25, 0.05];

        let data = sm.checkpoint();
        let mut sm2 = SpectrumManager::default();
        sm2.restore(&data).unwrap();

        assert_eq!(sm2.tier_available, [false, true, true]);
        assert_eq!(sm2.jamming_streak, 5);
        assert_eq!(sm2.degradation_streak, 3);
        assert!((sm2.predicted_noise_floor - (-85.0)).abs() < 1e-10);
        assert!((sm2.tier_loss_ema[1] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_spectrum_manager_restore_rejects_short() {
        let mut sm = SpectrumManager::default();
        assert!(sm.restore(&[0u8; 10]).is_err());
    }

    #[test]
    fn test_spectrum_manager_degradation_streak() {
        let mut sm = SpectrumManager::default();
        sm.set_tier_available(RadioTier::Local, false);
        // set_tier_available calls update_network_health which increments streak
        assert!(sm.degradation_streak >= 1);

        let streak_before = sm.degradation_streak;
        // Process cycle calls update_network_health again → streak grows
        let snapshot = CycleSnapshot::default();
        sm.process(&snapshot);
        assert!(sm.degradation_streak > streak_before);

        // Restore Local → streak resets
        sm.set_tier_available(RadioTier::Local, true);
        assert_eq!(sm.degradation_streak, 0);
    }

    #[test]
    fn test_spectrum_manager_telemetry() {
        let mut sm = SpectrumManager::default();
        sm.set_tier_available(RadioTier::Local, false);

        let snapshot = CycleSnapshot::default();
        sm.process(&snapshot);

        let telem = sm.telemetry();
        assert_eq!(telem.network_health, 1); // LocalDown
        assert!(!telem.tier_available[0]);
        assert!(telem.tier_available[1]);
        assert!(telem.epistemic_discount < 1.0);
    }

    // ── RLE encode/decode ────────────────────────────────────────────────

    #[test]
    fn test_rle_roundtrip_zeros() {
        let data = [0u8; 2048];
        let encoded = CompressedDelta::rle_encode(&data);
        let decoded = CompressedDelta::rle_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_rle_roundtrip_pattern() {
        let mut data = [0u8; 100];
        for i in 0..100 {
            data[i] = if i < 50 { 0xFF } else { 0x00 };
        }
        let encoded = CompressedDelta::rle_encode(&data);
        let decoded = CompressedDelta::rle_decode(&encoded).unwrap();
        assert_eq!(decoded, data.to_vec());
    }

    #[test]
    fn test_rle_empty() {
        let encoded = CompressedDelta::rle_encode(&[]);
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_rle_decode_invalid_length() {
        // Not a multiple of 3
        assert!(CompressedDelta::rle_decode(&[0, 0, 0, 1]).is_none());
    }

    // ── Regulatory ───────────────────────────────────────────────────────

    #[test]
    fn test_regulatory_hard_gate() {
        let reg = RegulatoryConstraints::default();
        // Military frequencies should be blocked
        assert!(!reg.is_allowed(300_000_000)); // UHF military
        assert!(!reg.is_allowed(50_000_000));  // VHF
    }
}
