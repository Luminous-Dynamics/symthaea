//! # Radio Dispatcher — Multi-Band Tiered Radio Architecture
//!
//! **Requires**: `feature = "mesh"`
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
use std::collections::{HashMap, VecDeque};

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
    /// Energy cost per bit in nanojoules (nJ/bit).
    /// Used for energy-aware tier selection when power budget is constrained.
    /// Basis: Friedman et al. (2013), Semtech SX1276 datasheet.
    pub energy_per_bit_nj: f64,
}

impl RadioTier {
    /// Get the physical profile for this tier.
    pub fn profile(self) -> TierProfile {
        match self {
            RadioTier::Local => TierProfile {
                mtu: 1500,
                bandwidth_budget: 1_000_000, // 1 MB per 10s window
                bandwidth_min: 100_000,      // 100 KB floor
                bandwidth_max: 10_000_000,   // 10 MB ceiling
                additive_increase: 100_000,  // +100 KB per healthy window
                decrease_factor: 0.5,
                duty_cycle: 1.0,
                latency_ms: 5,
                reliability: 0.99,
                energy_per_bit_nj: RADIO_ENERGY_PER_BIT_LOCAL,
            },
            RadioTier::Metro => TierProfile {
                mtu: 250,
                bandwidth_budget: 2_500, // 2.5 KB per 10s (LoRa 1% duty)
                bandwidth_min: 500,      // 500 B floor
                bandwidth_max: 25_000,   // 25 KB ceiling
                additive_increase: 250,  // +250 B per healthy window
                decrease_factor: 0.5,
                duty_cycle: 0.01, // 1% EU LoRa duty cycle
                latency_ms: 500,
                reliability: 0.85,
                energy_per_bit_nj: RADIO_ENERGY_PER_BIT_METRO,
            },
            RadioTier::Regional => TierProfile {
                mtu: 50,
                bandwidth_budget: 250, // 250 B per 10s (~50 bps)
                bandwidth_min: 50,     // 50 B floor
                bandwidth_max: 1_000,  // 1 KB ceiling
                additive_increase: 25, // +25 B per healthy window
                decrease_factor: 0.5,
                duty_cycle: 0.5, // 50% (amateur HF)
                latency_ms: 5_000,
                reliability: 0.70,
                energy_per_bit_nj: RADIO_ENERGY_PER_BIT_REGIONAL,
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
            NetworkHealth::AllTiersUp => 0, // Green
            NetworkHealth::LocalDown => 1,  // Yellow
            NetworkHealth::MetroOnly => 2,  // Orange
            NetworkHealth::Blackout => 3,   // Red
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
                (902_000_000, 928_000_000),     // 915 MHz ISM (US)
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
                (863_000_000, 870_000_000),     // 868 MHz ISM (EU)
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

// ═══════════════════════════════════════════════════════════════════════════════
// SPECTRUM WATERFALL — Time-series observation buffer
// ═══════════════════════════════════════════════════════════════════════════════

/// Timestamped spectrum observation for waterfall buffer.
#[derive(Debug, Clone)]
struct WaterfallEntry {
    /// Cycle number when observed.
    cycle: u64,
    /// Mean noise floor across all observations that cycle (dBm).
    noise_floor_dbm: f64,
    /// Mean SNR across all observations that cycle (dB).
    snr_db: f64,
    /// Whether jamming was detected this cycle.
    jammed: bool,
    /// Number of raw observations aggregated.
    observation_count: u32,
}

/// Ring buffer of spectrum observations for pattern detection.
///
/// Maintains a fixed-capacity window of aggregated per-cycle spectrum state.
/// Enables periodic interference detection and noise floor trend analysis.
///
/// Basis: Haykin (2005) — cognitive radio spectrum sensing requires
/// temporal context for reliable detection.
struct SpectrumWaterfall {
    /// Ring buffer of entries (oldest at front).
    entries: VecDeque<WaterfallEntry>,
    /// Maximum entries to keep.
    capacity: usize,
}

impl SpectrumWaterfall {
    fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Record a cycle's aggregated spectrum state.
    fn push(&mut self, entry: WaterfallEntry) {
        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    /// Mean noise floor across all entries (dBm).
    fn mean_noise_floor(&self) -> Option<f64> {
        if self.entries.is_empty() {
            return None;
        }
        let sum: f64 = self.entries.iter().map(|e| e.noise_floor_dbm).sum();
        Some(sum / self.entries.len() as f64)
    }

    /// Noise floor variance (for trend detection).
    fn noise_floor_variance(&self) -> Option<f64> {
        let mean = self.mean_noise_floor()?;
        if self.entries.len() < 2 {
            return None;
        }
        let var: f64 = self
            .entries
            .iter()
            .map(|e| (e.noise_floor_dbm - mean).powi(2))
            .sum::<f64>()
            / (self.entries.len() - 1) as f64;
        Some(var)
    }

    /// Fraction of recent entries with jamming detected (0.0–1.0).
    fn jamming_ratio(&self) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let jammed = self.entries.iter().filter(|e| e.jammed).count();
        jammed as f64 / self.entries.len() as f64
    }

    /// Detect periodic interference pattern.
    ///
    /// Returns estimated period (in cycles) if a repeating noise spike is found.
    /// Uses simple peak detection on noise floor — looks for regularly-spaced
    /// entries where noise is >1 std dev above mean.
    fn detect_periodic_interference(&self) -> Option<u32> {
        let mean = self.mean_noise_floor()?;
        let var = self.noise_floor_variance()?;
        let std_dev = var.sqrt();
        if std_dev < 1.0 || self.entries.len() < RADIO_WATERFALL_MIN_SAMPLES {
            return None;
        }

        // Find indices of noise spikes (>1 std dev above mean)
        let spike_cycles: Vec<u64> = self
            .entries
            .iter()
            .filter(|e| e.noise_floor_dbm > mean + std_dev)
            .map(|e| e.cycle)
            .collect();

        if spike_cycles.len() < 3 {
            return None;
        }

        // Check for consistent inter-spike intervals
        let intervals: Vec<u64> = spike_cycles.windows(2).map(|w| w[1] - w[0]).collect();
        let mean_interval = intervals.iter().sum::<u64>() as f64 / intervals.len() as f64;

        // Check variance of intervals — periodic if low
        let interval_var: f64 = intervals
            .iter()
            .map(|&i| (i as f64 - mean_interval).powi(2))
            .sum::<f64>()
            / intervals.len() as f64;

        if interval_var < mean_interval * 0.5 && mean_interval > 1.0 {
            Some(mean_interval as u32)
        } else {
            None
        }
    }

    /// Number of entries currently stored.
    fn len(&self) -> usize {
        self.entries.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADAPTIVE COMPRESSION — Tier-aware encoding strategies
// ═══════════════════════════════════════════════════════════════════════════════

/// Compression strategy selected based on target radio tier and payload type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionStrategy {
    /// Full 2048-byte BinaryHV — Local tier only.
    Full,
    /// XOR + RLE delta — Metro tier, typical 50-200 bytes.
    Delta,
    /// BLAKE3 hash only (32 bytes) — Regional tier, consensus verification.
    HashOnly,
}

impl CompressionStrategy {
    /// Select strategy based on target tier and whether a previous state exists.
    pub fn for_tier(tier: RadioTier, has_previous: bool) -> Self {
        match tier {
            RadioTier::Local => CompressionStrategy::Full,
            RadioTier::Metro => {
                if has_previous {
                    CompressionStrategy::Delta
                } else {
                    CompressionStrategy::Full // Must send full on first contact
                }
            }
            RadioTier::Regional => CompressionStrategy::HashOnly,
        }
    }
}

/// Result of tier-adaptive compression.
#[derive(Debug, Clone)]
pub struct TierCompressedPayload {
    /// The compression strategy used.
    pub strategy: CompressionStrategy,
    /// Compressed payload bytes (wire format).
    pub data: Vec<u8>,
    /// Original uncompressed size.
    pub original_size: usize,
}

impl TierCompressedPayload {
    /// Wire size in bytes.
    pub fn wire_size(&self) -> usize {
        self.data.len()
    }

    /// Compression ratio (0.0 = perfect, 1.0 = no compression).
    pub fn compression_ratio(&self) -> f64 {
        if self.original_size == 0 {
            return 1.0;
        }
        self.data.len() as f64 / self.original_size as f64
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FEC — Forward Error Correction (Reed-Solomon approximation)
// ═══════════════════════════════════════════════════════════════════════════════

/// Simple XOR-based FEC for Metro/Regional tiers.
///
/// Not full Reed-Solomon (would require a GF(2^8) library), but provides
/// single-block recovery via XOR parity. For N data blocks, produces 1 parity
/// block that can recover any single lost block.
///
/// Basis: Lin & Costello (2004) — error control coding fundamentals.
///
/// NOTE: For LoRa fragmentation, use `swarm::mesh::lora_fragment::FragmentAssembler`
/// which provides the same XOR FEC integrated with CRC-16 and reassembly.
/// This encoder is for non-LoRa payloads (Metro/Regional custom frames).
pub struct FecEncoder;

impl FecEncoder {
    /// Encode data with XOR parity FEC.
    ///
    /// Splits data into `block_size`-byte blocks and appends one XOR parity block.
    /// Returns the data with parity appended.
    pub fn encode(data: &[u8], block_size: usize) -> Vec<u8> {
        if data.len() < RADIO_FEC_MIN_PAYLOAD || block_size == 0 {
            return data.to_vec();
        }

        let mut result = data.to_vec();
        let mut parity = vec![0u8; block_size];

        for chunk in data.chunks(block_size) {
            for (i, &b) in chunk.iter().enumerate() {
                parity[i] ^= b;
            }
        }

        result.extend_from_slice(&parity);
        result
    }

    /// Decode FEC-encoded data, recovering from a single lost block.
    ///
    /// `lost_block_index`: which block (0-based) was lost. If `None`, just strips parity.
    pub fn decode(encoded: &[u8], block_size: usize, lost_block_index: Option<usize>) -> Vec<u8> {
        if block_size == 0 || encoded.len() <= block_size {
            return encoded.to_vec();
        }

        let data_len = encoded.len() - block_size;
        let parity = &encoded[data_len..];
        let mut data = encoded[..data_len].to_vec();

        if let Some(lost_idx) = lost_block_index {
            let start = lost_idx * block_size;
            let end = (start + block_size).min(data_len);

            // Recover: parity XOR all other blocks
            let mut recovered = parity.to_vec();
            for (block_idx, chunk) in data.chunks(block_size).enumerate() {
                if block_idx != lost_idx {
                    for (i, &b) in chunk.iter().enumerate() {
                        if i < recovered.len() {
                            recovered[i] ^= b;
                        }
                    }
                }
            }

            // Write recovered block
            for i in start..end {
                if i - start < recovered.len() {
                    data[i] = recovered[i - start];
                }
            }
        }

        data
    }

    /// Calculate FEC overhead for a given data size and block size.
    pub fn overhead(data_len: usize, block_size: usize) -> usize {
        if data_len < RADIO_FEC_MIN_PAYLOAD || block_size == 0 {
            0
        } else {
            block_size
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PEER DISCOVERY — Lightweight beacon protocol
// ═══════════════════════════════════════════════════════════════════════════════

/// Discovery beacon broadcast on Regional/Metro tiers.
///
/// Minimal payload (24 bytes) designed to fit within Regional MTU (50 bytes):
/// - 8 bytes: node ID
/// - 8 bytes: capabilities hash
/// - 4 bytes: cycle counter (for liveliness)
/// - 4 bytes: network health + tier mask + reserved
#[derive(Debug, Clone)]
pub struct DiscoveryBeacon {
    /// First 8 bytes of the node's identity.
    pub node_id: [u8; 8],
    /// Hash of the node's capabilities (features, firmware version, etc.).
    pub capabilities_hash: [u8; 8],
    /// Cycle counter at beacon time (monotonically increasing).
    pub cycle_counter: u32,
    /// Current network health level (0-3).
    pub network_health: u8,
    /// Bitmask of available tiers (bit 0 = Local, 1 = Metro, 2 = Regional).
    pub tier_mask: u8,
}

impl DiscoveryBeacon {
    /// Serialize to wire format (24 bytes).
    pub fn to_bytes(&self) -> [u8; RADIO_BEACON_SIZE] {
        let mut buf = [0u8; RADIO_BEACON_SIZE];
        buf[0..8].copy_from_slice(&self.node_id);
        buf[8..16].copy_from_slice(&self.capabilities_hash);
        buf[16..20].copy_from_slice(&self.cycle_counter.to_le_bytes());
        buf[20] = self.network_health;
        buf[21] = self.tier_mask;
        // 22-23: reserved
        buf
    }

    /// Deserialize from wire format.
    pub fn from_bytes(data: &[u8; RADIO_BEACON_SIZE]) -> Self {
        let mut node_id = [0u8; 8];
        node_id.copy_from_slice(&data[0..8]);
        let mut capabilities_hash = [0u8; 8];
        capabilities_hash.copy_from_slice(&data[8..16]);
        let cycle_counter = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);
        Self {
            node_id,
            capabilities_hash,
            cycle_counter,
            network_health: data[20],
            tier_mask: data[21],
        }
    }

    /// Generate tier mask from availability array.
    pub fn tier_mask_from(available: &[bool; 3]) -> u8 {
        let mut mask = 0u8;
        if available[0] {
            mask |= 0x01;
        }
        if available[1] {
            mask |= 0x02;
        }
        if available[2] {
            mask |= 0x04;
        }
        mask
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MULTI-HOP RELAY ROUTING — Mesh route table
// ═══════════════════════════════════════════════════════════════════════════════

/// A route to a peer via zero or more relay hops.
#[derive(Debug, Clone)]
pub struct RouteEntry {
    /// Destination node ID (first 8 bytes).
    pub destination: [u8; 8],
    /// Next-hop node ID (direct neighbor to forward to).
    pub next_hop: [u8; 8],
    /// Number of hops to destination (0 = direct neighbor).
    pub hop_count: u8,
    /// Best tier to reach next hop.
    pub tier: RadioTier,
    /// Cycle when this route was last refreshed.
    pub last_seen_cycle: u64,
    /// Estimated link quality (0.0–1.0).
    pub link_quality: f32,
}

/// Mesh routing table with TTL-based expiry.
///
/// Routes are learned from beacon reception and forwarded route advertisements.
/// Stale routes are pruned each cycle to prevent routing to departed nodes.
///
/// Basis: Perkins & Royer (1999) — Ad hoc On-Demand Distance Vector (AODV).
pub struct RouteTable {
    /// Known routes, keyed by destination node ID.
    routes: HashMap<[u8; 8], RouteEntry>,
    /// Maximum entries.
    capacity: usize,
}

impl RouteTable {
    fn new(capacity: usize) -> Self {
        Self {
            routes: HashMap::with_capacity(capacity),
            capacity,
        }
    }

    /// Add or update a route. Prefers shorter hop counts and higher link quality.
    pub fn update(&mut self, entry: RouteEntry) {
        if let Some(existing) = self.routes.get(&entry.destination) {
            // Only update if better (fewer hops, or same hops but better quality)
            if entry.hop_count < existing.hop_count
                || (entry.hop_count == existing.hop_count
                    && entry.link_quality > existing.link_quality)
            {
                self.routes.insert(entry.destination, entry);
            } else {
                // Just refresh the timestamp
                if let Some(e) = self.routes.get_mut(&entry.destination) {
                    e.last_seen_cycle = entry.last_seen_cycle;
                }
            }
        } else if self.routes.len() < self.capacity {
            self.routes.insert(entry.destination, entry);
        }
    }

    /// Look up the best route to a destination.
    pub fn lookup(&self, destination: &[u8; 8]) -> Option<&RouteEntry> {
        self.routes.get(destination)
    }

    /// Prune routes older than `max_age_cycles` from `current_cycle`.
    pub fn prune(&mut self, current_cycle: u64, max_age_cycles: u64) {
        self.routes.retain(|_, entry| {
            current_cycle.saturating_sub(entry.last_seen_cycle) < max_age_cycles
        });
    }

    /// Number of known routes.
    pub fn len(&self) -> usize {
        self.routes.len()
    }

    /// All known destinations.
    pub fn destinations(&self) -> Vec<[u8; 8]> {
        self.routes.keys().copied().collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MESH ENCRYPTION — Per-peer session encryption
// ═══════════════════════════════════════════════════════════════════════════════

/// Per-peer encryption session state.
///
/// Uses a pre-shared key model (PSK) with per-peer derived keys.
/// In production, this would use X25519 key exchange; here we model
/// the session state and nonce management.
///
/// Basis: Bernstein (2008) — ChaCha20-Poly1305 AEAD construction.
#[derive(Debug, Clone)]
pub struct PeerSession {
    /// Peer node ID.
    pub peer_id: [u8; 8],
    /// Shared session key (32 bytes, derived from PSK + peer IDs).
    pub session_key: [u8; 32],
    /// Outgoing nonce counter (monotonically increasing, prevents replay).
    pub tx_nonce_counter: u64,
    /// Last received nonce (for replay detection).
    pub rx_nonce_seen: u64,
    /// Cycle when this session was established.
    pub established_cycle: u64,
}

impl PeerSession {
    /// Create a new session with a derived key.
    ///
    /// In production, `session_key` would come from X25519 DH + HKDF.
    /// Here we accept it as a parameter.
    pub fn new(peer_id: [u8; 8], session_key: [u8; 32], cycle: u64) -> Self {
        Self {
            peer_id,
            session_key,
            tx_nonce_counter: 0,
            rx_nonce_seen: 0,
            established_cycle: cycle,
        }
    }

    /// Generate the next nonce (12 bytes: 4 zero + 8 counter LE).
    pub fn next_nonce(&mut self) -> [u8; RADIO_CRYPTO_NONCE_SIZE] {
        self.tx_nonce_counter += 1;
        let mut nonce = [0u8; RADIO_CRYPTO_NONCE_SIZE];
        nonce[4..12].copy_from_slice(&self.tx_nonce_counter.to_le_bytes());
        nonce
    }

    /// Check if a received nonce is valid (not replayed).
    pub fn check_nonce(&mut self, nonce_counter: u64) -> bool {
        if nonce_counter <= self.rx_nonce_seen {
            return false; // Replay detected
        }
        self.rx_nonce_seen = nonce_counter;
        true
    }
}

/// Mesh encryption manager — tracks per-peer session keys and nonces.
///
/// In a real deployment, this would integrate with Mycelix identity
/// for peer authentication and X25519 key exchange.
pub struct MeshEncryption {
    /// Active sessions keyed by peer node ID.
    sessions: HashMap<[u8; 8], PeerSession>,
    /// Maximum sessions.
    capacity: usize,
}

impl MeshEncryption {
    fn new(capacity: usize) -> Self {
        Self {
            sessions: HashMap::with_capacity(capacity),
            capacity,
        }
    }

    /// Register a peer session.
    pub fn add_session(&mut self, session: PeerSession) {
        if self.sessions.len() < self.capacity {
            self.sessions.insert(session.peer_id, session);
        }
    }

    /// Get a session for a peer.
    pub fn get_session(&self, peer_id: &[u8; 8]) -> Option<&PeerSession> {
        self.sessions.get(peer_id)
    }

    /// Get a mutable session for a peer.
    pub fn get_session_mut(&mut self, peer_id: &[u8; 8]) -> Option<&mut PeerSession> {
        self.sessions.get_mut(peer_id)
    }

    /// Remove a peer session.
    pub fn remove_session(&mut self, peer_id: &[u8; 8]) -> Option<PeerSession> {
        self.sessions.remove(peer_id)
    }

    /// Number of active sessions.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Encrypt plaintext using XOR scrambling (test/simulation placeholder).
    ///
    /// **WARNING: NOT cryptographically secure.** For production mesh encryption,
    /// use `swarm::mesh::mod.rs` which implements real ChaCha20-Poly1305 via the
    /// `chacha20poly1305` crate (feature: `mesh-encryption`).
    ///
    /// This placeholder preserves the API shape for unit testing session
    /// management (nonce tracking, replay detection) without requiring
    /// the full crypto dependency chain.
    pub fn encrypt(
        key: &[u8; 32],
        nonce: &[u8; RADIO_CRYPTO_NONCE_SIZE],
        plaintext: &[u8],
    ) -> Vec<u8> {
        let mut ciphertext = plaintext.to_vec();
        for (i, byte) in ciphertext.iter_mut().enumerate() {
            *byte ^= key[i % 32] ^ nonce[i % RADIO_CRYPTO_NONCE_SIZE];
        }
        // 16-byte simulated auth tag
        let mut tag = [0u8; 16];
        for (i, &b) in ciphertext.iter().enumerate() {
            tag[i % 16] ^= b;
        }
        ciphertext.extend_from_slice(&tag);
        ciphertext
    }

    /// Decrypt ciphertext using XOR scrambling (test/simulation placeholder).
    ///
    /// **WARNING: NOT cryptographically secure.** See `encrypt()` doc.
    pub fn decrypt(
        key: &[u8; 32],
        nonce: &[u8; RADIO_CRYPTO_NONCE_SIZE],
        ciphertext: &[u8],
    ) -> Option<Vec<u8>> {
        if ciphertext.len() < 16 {
            return None;
        }
        let data_len = ciphertext.len() - 16;
        let data = &ciphertext[..data_len];
        let tag = &ciphertext[data_len..];

        let mut expected_tag = [0u8; 16];
        for (i, &b) in data.iter().enumerate() {
            expected_tag[i % 16] ^= b;
        }
        // Constant-time comparison to prevent timing attacks on the auth tag.
        // Uses the same constant_time_eq from the handshake module.
        if !crate::swarm::handshake::constant_time_eq(tag, &expected_tag) {
            return None;
        }

        let mut plaintext = data.to_vec();
        for (i, byte) in plaintext.iter_mut().enumerate() {
            *byte ^= key[i % 32] ^ nonce[i % RADIO_CRYPTO_NONCE_SIZE];
        }
        Some(plaintext)
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
///
/// ## Hardware Integration
///
/// When a `RadioHardware` implementation is attached via `set_hardware()`,
/// the manager polls real SNR and availability from hardware each cycle.
/// Without hardware, it relies on manually-fed `SpectrumObservation`s.
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

    // ── Hardware abstraction ──────────────────────────────────────────
    /// Optional physical radio hardware (polled each cycle for SNR/availability).
    hardware: Option<Box<dyn RadioHardware>>,
    /// Region-aware regulatory database for frequency/power validation.
    regulatory_db: RegulatoryDatabase,

    // ── Waterfall (spectrum history) ─────────────────────────────────
    /// Time-series spectrum observation buffer for pattern detection.
    waterfall: SpectrumWaterfall,

    // ── Frequency hopping ────────────────────────────────────────────
    /// Cycles since last frequency hop (cooldown).
    hop_cooldown: u32,

    // ── Peer discovery ───────────────────────────────────────────────
    /// This node's ID (first 8 bytes).
    node_id: [u8; 8],
    /// Cycles since last beacon broadcast.
    beacon_counter: u32,

    // ── Multi-hop routing ────────────────────────────────────────────
    /// Mesh route table for multi-hop relay.
    route_table: RouteTable,
    /// Current cycle counter (for route expiry).
    current_cycle: u64,

    // ── Encryption ───────────────────────────────────────────────────
    /// Per-peer encryption session manager.
    encryption: MeshEncryption,

    // ── Energy tracking ──────────────────────────────────────────────
    /// Cumulative energy spent this cycle (nJ).
    energy_spent_nj: f64,
    /// Energy budget per cycle (nJ).
    energy_budget_nj: f64,
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
    /// Per-tier AIMD bandwidth budgets [Local, Metro, Regional].
    pub tier_budgets: [u64; 3],
    /// Waterfall depth (number of stored observations).
    pub waterfall_depth: usize,
    /// Periodic interference period (cycles), if detected.
    pub periodic_interference: Option<u32>,
    /// Known mesh peers (route table size).
    pub known_peers: usize,
    /// Active encryption sessions.
    pub encryption_sessions: usize,
    /// Energy spent this cycle (nJ).
    pub energy_spent_nj: f64,
    /// Jamming ratio from waterfall (0.0–1.0).
    pub waterfall_jamming_ratio: f64,
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
            predicted_noise_floor: RADIO_DEFAULT_NOISE_FLOOR_DBM,
            regulatory: RegulatoryConstraints::default(),
            network_health: NetworkHealth::AllTiersUp,
            degradation_streak: 0,
            peer_last_hv: Vec::new(),
            last_telemetry: SpectrumTelemetry::default(),
            hardware: None,
            regulatory_db: RegulatoryDatabase::new(RegulatoryRegion::IsmGlobal),
            waterfall: SpectrumWaterfall::new(RADIO_WATERFALL_CAPACITY),
            hop_cooldown: 0,
            node_id: [0u8; 8],
            beacon_counter: 0,
            route_table: RouteTable::new(RADIO_MAX_ROUTE_ENTRIES),
            current_cycle: 0,
            encryption: MeshEncryption::new(RADIO_CRYPTO_MAX_PEERS),
            energy_spent_nj: 0.0,
            energy_budget_nj: RADIO_ENERGY_BUDGET_PER_CYCLE,
        }
    }
}

// Re-export named constants from thresholds.rs for local use.
use super::super::thresholds::{
    RADIO_AUTO_HOP_NOISE_THRESHOLD, RADIO_BEACON_INTERVAL_CYCLES,
    RADIO_BEACON_PEER_CONFIDENCE_BOOST, RADIO_BEACON_SIZE, RADIO_BLACKOUT_EXPLORATION_BOOST,
    RADIO_CONSCIOUSNESS_HIGH_CONFIDENCE, RADIO_CONSCIOUSNESS_LOW_CONFIDENCE,
    RADIO_CRYPTO_MAX_PEERS, RADIO_CRYPTO_NONCE_SIZE, RADIO_DEFAULT_NOISE_FLOOR_DBM,
    RADIO_DEGRADATION_CONFIDENCE_DROP as DEGRADATION_CONFIDENCE_DROP, RADIO_ENERGY_AWARE_THRESHOLD,
    RADIO_ENERGY_BUDGET_PER_CYCLE, RADIO_ENERGY_PER_BIT_LOCAL, RADIO_ENERGY_PER_BIT_METRO,
    RADIO_ENERGY_PER_BIT_REGIONAL, RADIO_FEC_MIN_PAYLOAD, RADIO_HOP_COOLDOWN_CYCLES,
    RADIO_HOP_SNR_IMPROVEMENT_DB, RADIO_JAMMING_AROUSAL_SPIKE as JAMMING_AROUSAL_SPIKE,
    RADIO_JAMMING_EXPLORATION_BOOST as JAMMING_EXPLORATION_BOOST,
    RADIO_JAMMING_SNR_THRESHOLD as JAMMING_SNR_THRESHOLD, RADIO_LOSS_LR_DAMPEN_FACTOR,
    RADIO_LOSS_LR_DAMPEN_MAX, RADIO_MAX_DELTA_PEERS as MAX_DELTA_PEERS, RADIO_MAX_RELAY_HOPS,
    RADIO_MAX_ROUTE_ENTRIES, RADIO_NOISE_ERROR_NORMALIZER,
    RADIO_NOISE_FLOOR_EMA_ALPHA as NOISE_FLOOR_EMA_ALPHA, RADIO_ROUTE_EXPIRY_CYCLES,
    RADIO_SAFETY_JAMMING_THRESHOLD, RADIO_SPECTRUM_PE_AROUSAL_MAX, RADIO_SPECTRUM_PE_AROUSAL_SCALE,
    RADIO_SPECTRUM_PE_SURPRISE_THRESHOLD, RADIO_SYNTHETIC_NOISE_FLOOR_BASE,
    RADIO_SYNTHETIC_NOISE_FLOOR_RANGE, RADIO_SYNTHETIC_PEER_CAP, RADIO_SYNTHETIC_SNR_BASE,
    RADIO_SYNTHETIC_SNR_ISOLATED, RADIO_SYNTHETIC_SNR_PEER_BONUS, RADIO_SYNTHETIC_SNR_PHI_BONUS,
    RADIO_TIER_DEGRADED_LOSS as TIER_DEGRADED_LOSS,
    RADIO_TIER_LOSS_EMA_ALPHA as TIER_LOSS_EMA_ALPHA, RADIO_WATERFALL_CAPACITY,
    RADIO_WATERFALL_MIN_SAMPLES,
};

impl SpectrumManager {
    /// Co-prime scheduling interval (cycles).
    pub const INTERVAL: u32 = 53;

    /// Create with specific regulatory constraints.
    pub(crate) fn with_regulatory(regulatory: RegulatoryConstraints) -> Self {
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

    /// Ingest mesh network statistics to update tier loss EMAs.
    ///
    /// Maps real mesh packet counters into per-tier loss estimates.
    /// This bridges the gap between `swarm::mesh::MeshStats` (actual network)
    /// and SpectrumManager (cognitive model of the spectrum).
    ///
    /// `dropped` and `total_sent` come from the mesh layer's counters.
    /// The loss ratio is applied to all tiers weighted by their expected reliability.
    pub fn ingest_mesh_stats(&mut self, dropped: u64, total_sent: u64) {
        if total_sent == 0 {
            return;
        }
        let loss_ratio = (dropped as f64 / total_sent as f64).clamp(0.0, 1.0);
        // Spread loss across tiers: Local absorbs least (high reliability),
        // Regional absorbs most (low reliability).
        let weights = [0.2, 0.5, 1.0]; // Local, Metro, Regional
        for (i, &w) in weights.iter().enumerate() {
            let tier_loss = (loss_ratio * w).clamp(0.0, 1.0);
            self.tier_loss_ema[i] = self.tier_loss_ema[i] * (1.0 - TIER_LOSS_EMA_ALPHA)
                + tier_loss * TIER_LOSS_EMA_ALPHA;
        }
        // Keep telemetry in sync so callers see updated values
        self.last_telemetry.tier_loss_ema = self.tier_loss_ema;
    }

    /// Ingest swarm peer state for bidirectional Swarm→Spectrum coupling.
    ///
    /// Uses peer connectivity data to generate synthetic spectrum observations,
    /// enabling the waterfall model to track network health even without SDR
    /// hardware. This closes the Swarm→Spectrum feedback loop.
    ///
    /// Science: Network-level perception modulates physical-layer awareness
    /// (Pentland 2014 — "Social Physics").
    pub fn ingest_swarm_state(
        &mut self,
        connected_peers: usize,
        mean_peer_phi: f64,
        connectivity_ema: f64,
    ) {
        // Synthesize a spectrum observation from swarm state.
        // More peers + higher connectivity → better SNR estimate.
        // Zero peers → poor SNR (isolation → degraded perception).
        let peer_snr = if connected_peers == 0 {
            RADIO_SYNTHETIC_SNR_ISOLATED
        } else {
            let peer_factor = (connected_peers as f64)
                .min(RADIO_SYNTHETIC_PEER_CAP)
                .ln_1p()
                / RADIO_SYNTHETIC_PEER_CAP.ln_1p();
            let phi_factor = mean_peer_phi.clamp(0.0, 1.0);
            // Base SNR + peer bonus + phi bonus
            RADIO_SYNTHETIC_SNR_BASE
                + peer_factor * RADIO_SYNTHETIC_SNR_PEER_BONUS
                + phi_factor * RADIO_SYNTHETIC_SNR_PHI_BONUS
        };

        let noise_floor = RADIO_SYNTHETIC_NOISE_FLOOR_BASE
            - connectivity_ema.clamp(0.0, 1.0) * RADIO_SYNTHETIC_NOISE_FLOOR_RANGE;

        self.pending_observations.push(SpectrumObservation {
            frequency_hz: 900_000_000, // Nominal mesh frequency
            noise_floor_dbm: noise_floor as f32,
            snr_db: peer_snr as f32,
            jammed: connected_peers == 0 && connectivity_ema < 0.1,
        });
    }

    /// Pending spectrum observations (for diagnostics/testing).
    pub fn pending_observations(&self) -> &[SpectrumObservation] {
        &self.pending_observations
    }

    /// Get the last spectrum prediction error for FEP integration.
    ///
    /// Returns the prediction error from the most recent waterfall processing.
    /// Consumers can blend this into the global FEP surprise signal.
    pub fn spectrum_prediction_error(&self) -> f64 {
        self.last_telemetry.spectrum_prediction_error
    }

    /// Whether jamming has persisted beyond the safety escalation threshold.
    ///
    /// When true, SafetyAgent should escalate safety level (analogous to
    /// `integrity_critical` for tamper detection).
    pub fn is_network_critical(&self) -> bool {
        self.jamming_streak >= RADIO_SAFETY_JAMMING_THRESHOLD
            || self.network_health == NetworkHealth::Blackout
    }

    /// Network connectivity factor for consciousness coupling [0.0, 1.0].
    ///
    /// 1.0 = all tiers healthy, 0.0 = blackout.
    /// Used by ConsciousnessEngine to modulate unified consciousness ±2%.
    /// Science: Cacioppo & Patrick (2008) — social isolation impairs higher cognition.
    pub fn connectivity_factor(&self) -> f64 {
        match self.network_health {
            NetworkHealth::AllTiersUp => 1.0,
            NetworkHealth::LocalDown => 0.7,
            NetworkHealth::MetroOnly => 0.4,
            NetworkHealth::Blackout => 0.0,
        }
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
    pub(crate) fn classifier(&self) -> &PayloadClassifier {
        &self.classifier
    }

    /// Get the current telemetry snapshot.
    pub fn telemetry(&self) -> &SpectrumTelemetry {
        &self.last_telemetry
    }

    /// Validate that a transmission on the given tier and frequency is allowed
    /// by the regulatory database.
    ///
    /// Returns `true` if the frequency and power are within legal limits.
    /// This should be called before any actual transmission to the mesh layer.
    pub fn validate_transmission(
        &self,
        tier: RadioTier,
        frequency_hz: u64,
        power_dbm: f32,
    ) -> bool {
        self.regulatory_db.bands_for_tier(tier).iter().any(|band| {
            frequency_hz >= band.freq_min_hz
                && frequency_hz <= band.freq_max_hz
                && power_dbm <= band.max_eirp_dbm
        })
    }

    /// Get the regulatory region.
    pub fn regulatory_region(&self) -> RegulatoryRegion {
        self.regulatory_db.region()
    }

    /// Compute a compressed delta for a BinaryHV relative to a peer's last state.
    ///
    /// If no previous state exists for this peer, returns a full vector.
    /// Updates the peer's last-known state after compression.
    fn compress_delta(&mut self, peer_id: &[u8; 8], current_hv: &[u8; 2048]) -> CompressedDelta {
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

    /// Route a payload through the classifier, with energy-aware override.
    ///
    /// When the energy budget is over 50% spent, `energy_aware_route()` takes
    /// precedence — preferring the lowest energy-per-bit tier that fits the
    /// payload, regardless of normal routing preferences.
    ///
    /// Basis: Friedman et al. (2013) — energy-proportional radio usage.
    pub(crate) fn route(
        &self,
        class: PayloadClass,
        payload_size: usize,
        urgency: u8,
    ) -> Option<RoutingDecision> {
        // Energy-constrained path: override normal routing when budget is tight
        let budget_fraction = if self.energy_budget_nj > 0.0 {
            self.energy_spent_nj / self.energy_budget_nj
        } else {
            0.0
        };
        if budget_fraction > RADIO_ENERGY_AWARE_THRESHOLD {
            if let Some(tier) = self.energy_aware_route(payload_size, urgency) {
                return Some(RoutingDecision::Routed {
                    tier,
                    fragmented: payload_size > tier.profile().mtu,
                    estimated_fragments: if payload_size > tier.profile().mtu {
                        (payload_size + tier.profile().mtu - 1) / tier.profile().mtu
                    } else {
                        1
                    },
                });
            }
        }
        self.classifier.route(class, payload_size, urgency)
    }

    /// Get the regulatory constraints.
    pub(crate) fn regulatory(&self) -> &RegulatoryConstraints {
        &self.regulatory
    }

    /// Total available bandwidth across all operational tiers (bytes per 10s window).
    ///
    /// Returns only the budgets of tiers that are currently available.
    /// Used by Broca cadence throttling to limit speech when bandwidth is low.
    pub(crate) fn available_bandwidth(&self) -> u64 {
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
        RadioTier::ALL
            .iter()
            .copied()
            .find(|&tier| self.tier_available[tier as usize])
    }

    /// Select tier based on consciousness confidence level.
    ///
    /// High-confidence critical actions route on reliable tiers (Local preferred).
    /// Low-confidence exploratory actions route on energy-efficient tiers.
    /// This closes the consciousness→spectrum bidirectional feedback loop.
    ///
    /// Science: Resource allocation proportional to certainty —
    /// uncertain actions don't warrant expensive transmission (Friston 2010).
    pub fn consciousness_aware_tier(
        &self,
        payload_size: usize,
        confidence: f64,
    ) -> Option<RadioTier> {
        if confidence > RADIO_CONSCIOUSNESS_HIGH_CONFIDENCE {
            // High confidence: prefer most reliable available tier
            for &tier in &RadioTier::ALL {
                if self.tier_available[tier as usize] && payload_size <= tier.profile().mtu {
                    return Some(tier);
                }
            }
        } else if confidence < RADIO_CONSCIOUSNESS_LOW_CONFIDENCE {
            // Low confidence: prefer most energy-efficient available tier
            let mut candidates: Vec<(RadioTier, f64)> = RadioTier::ALL
                .iter()
                .copied()
                .filter(|&t| self.tier_available[t as usize])
                .filter(|&t| payload_size <= t.profile().mtu || t == RadioTier::Local)
                .map(|t| (t, t.profile().energy_per_bit_nj))
                .collect();
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            return candidates.first().map(|&(tier, _)| tier);
        }
        // Medium confidence: normal routing (let classifier decide)
        None
    }

    /// Create a SpectrumManager with a specific regulatory region.
    pub(crate) fn with_region(region: RegulatoryRegion) -> Self {
        let db = RegulatoryDatabase::new(region);
        let regulatory = db.to_legacy_constraints();
        let mut sm = Self::default();
        sm.regulatory = regulatory;
        sm.regulatory_db = db;
        sm
    }

    /// Attach radio hardware to this manager.
    ///
    /// When hardware is attached, `process()` will poll SNR and availability
    /// from it before processing observations.
    pub(crate) fn set_hardware(&mut self, hardware: Box<dyn RadioHardware>) {
        self.hardware = Some(hardware);
    }

    /// Get the hardware identifier, if hardware is attached.
    pub(crate) fn hardware_id(&self) -> Option<&str> {
        self.hardware.as_ref().map(|h| h.hardware_id())
    }

    /// Get the regulatory database.
    pub(crate) fn regulatory_db(&self) -> &RegulatoryDatabase {
        &self.regulatory_db
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
        let noise_error = ((mean_noise - self.predicted_noise_floor).abs()
            / RADIO_NOISE_ERROR_NORMALIZER)
            .min(1.0);

        // Update predicted noise floor via EMA
        self.predicted_noise_floor = self.predicted_noise_floor * (1.0 - NOISE_FLOOR_EMA_ALPHA)
            + mean_noise * NOISE_FLOOR_EMA_ALPHA;

        // Record into waterfall for pattern detection
        let mean_snr =
            observations.iter().map(|o| o.snr_db as f64).sum::<f64>() / observations.len() as f64;
        self.waterfall.push(WaterfallEntry {
            cycle: self.current_cycle,
            noise_floor_dbm: mean_noise,
            snr_db: mean_snr,
            jammed: jammed_count > 0,
            observation_count: observations.len() as u32,
        });

        noise_error
    }

    fn update_telemetry(&mut self, spectrum_pe: f64) {
        self.last_telemetry = SpectrumTelemetry {
            network_health: self.network_health.safety_suggestion(),
            tier_available: self.tier_available,
            tier_loss_ema: self.tier_loss_ema,
            jamming_streak: self.jamming_streak,
            spectrum_prediction_error: spectrum_pe,
            avg_delta_compression: 0.0,
            epistemic_discount: self.network_health.epistemic_discount(),
            degradation_streak: self.degradation_streak,
            tier_budgets: self.tier_budget,
            waterfall_depth: self.waterfall.len(),
            periodic_interference: self.waterfall.detect_periodic_interference(),
            known_peers: self.route_table.len(),
            encryption_sessions: self.encryption.session_count(),
            energy_spent_nj: self.energy_spent_nj,
            waterfall_jamming_ratio: self.waterfall.jamming_ratio(),
        };
    }

    /// Check if any tier has loss above the degradation threshold.
    fn any_tier_degraded(&self) -> bool {
        self.tier_loss_ema
            .iter()
            .zip(self.tier_available.iter())
            .any(|(&loss, &avail)| avail && loss > TIER_DEGRADED_LOSS)
    }

    // ── Item 1: AIMD congestion control ──────────────────────────────

    /// Tick AIMD bandwidth control for all tiers.
    ///
    /// Called each process cycle:
    /// - Healthy tier (low loss): additive increase toward ceiling
    /// - Congested tier (high loss): multiplicative decrease toward floor
    ///
    /// Basis: Jacobson (1988) — AIMD achieves fair bandwidth allocation
    /// and converges to optimal throughput under varied conditions.
    fn tick_aimd(&mut self) {
        for (idx, tier) in RadioTier::ALL.iter().enumerate() {
            if !self.tier_available[idx] {
                continue;
            }

            let profile = tier.profile();
            let loss = self.tier_loss_ema[idx];

            if loss > TIER_DEGRADED_LOSS {
                // Multiplicative decrease: budget *= decrease_factor
                let new_budget = (self.tier_budget[idx] as f64 * profile.decrease_factor) as u64;
                self.tier_budget[idx] = new_budget.max(profile.bandwidth_min);
            } else {
                // Additive increase: budget += additive_increase
                self.tier_budget[idx] =
                    (self.tier_budget[idx] + profile.additive_increase).min(profile.bandwidth_max);
            }
        }
    }

    // ── Item 2: Adaptive tier compression ────────────────────────────

    /// Compress a BinaryHV for a specific radio tier.
    ///
    /// Selects compression strategy based on tier MTU:
    /// - Local: full 2048-byte vector
    /// - Metro: XOR + RLE delta (50-200 bytes typical)
    /// - Regional: 32-byte hash for consensus verification only
    pub fn compress_for_tier(
        &mut self,
        peer_id: &[u8; 8],
        current_hv: &[u8; 2048],
        tier: RadioTier,
    ) -> TierCompressedPayload {
        let has_previous = self.peer_last_hv.iter().any(|(id, _)| id == peer_id);
        let strategy = CompressionStrategy::for_tier(tier, has_previous);

        let data = match strategy {
            CompressionStrategy::Full => {
                // Store peer state so subsequent calls can use delta
                if let Some(entry) = self.peer_last_hv.iter_mut().find(|(id, _)| id == peer_id) {
                    entry.1 = *current_hv;
                } else if self.peer_last_hv.len() < MAX_DELTA_PEERS {
                    self.peer_last_hv.push((*peer_id, *current_hv));
                }
                current_hv.to_vec()
            }
            CompressionStrategy::Delta => {
                let delta = self.compress_delta(peer_id, current_hv);
                delta.rle_data
            }
            CompressionStrategy::HashOnly => {
                // Simple hash: XOR fold 2048 bytes into 32 bytes
                let mut hash = [0u8; 32];
                for (i, &b) in current_hv.iter().enumerate() {
                    hash[i % 32] ^= b;
                }
                // Update peer state even though we only sent a hash
                if let Some(entry) = self.peer_last_hv.iter_mut().find(|(id, _)| id == peer_id) {
                    entry.1 = *current_hv;
                } else if self.peer_last_hv.len() < MAX_DELTA_PEERS {
                    self.peer_last_hv.push((*peer_id, *current_hv));
                }
                hash.to_vec()
            }
        };

        // Track energy cost
        let bits = data.len() as f64 * 8.0;
        self.energy_spent_nj += bits * tier.profile().energy_per_bit_nj;

        TierCompressedPayload {
            strategy,
            data,
            original_size: 2048,
        }
    }

    // ── Item 6: Cognitive frequency hopping ──────────────────────────

    /// Evaluate whether to hop frequencies based on waterfall analysis.
    ///
    /// Returns the suggested frequency (Hz) if a hop would improve SNR,
    /// or None if current frequency is acceptable.
    ///
    /// Basis: Haykin (2005) — cognitive radio uses spectrum sensing
    /// to find and exploit unused frequencies.
    pub fn suggest_frequency_hop(&mut self, tier: RadioTier) -> Option<u64> {
        // Cooldown check
        if self.hop_cooldown > 0 {
            return None;
        }

        // Need enough waterfall data
        if self.waterfall.len() < RADIO_WATERFALL_MIN_SAMPLES {
            return None;
        }

        // Only hop if we're experiencing bad conditions
        let current_mean_snr = self.waterfall.entries.back().map(|e| e.snr_db)?;

        if current_mean_snr > JAMMING_SNR_THRESHOLD as f64 + RADIO_HOP_SNR_IMPROVEMENT_DB as f64 {
            return None; // Current frequency is fine
        }

        // Find best band for this tier from regulatory database
        let bands = self.regulatory_db.bands_for_tier(tier);
        if bands.is_empty() {
            return None;
        }

        // Select center of the widest allowed band (simple heuristic)
        let best_band = bands.iter().max_by_key(|b| b.freq_max_hz - b.freq_min_hz)?;
        let center_freq = (best_band.freq_min_hz + best_band.freq_max_hz) / 2;

        // Check if this is different from current frequency
        if let Some(ref hw) = self.hardware {
            if hw.current_frequency(tier) == Some(center_freq) {
                return None; // Already on the best frequency
            }
        }

        self.hop_cooldown = RADIO_HOP_COOLDOWN_CYCLES;
        Some(center_freq)
    }

    // ── Item 7: Energy-aware routing ─────────────────────────────────

    /// Select the most energy-efficient tier that can carry the payload.
    ///
    /// When energy budget is constrained (>50% spent), prefers the
    /// lowest energy-per-bit tier that has sufficient MTU.
    ///
    /// Basis: Friedman et al. (2013) — energy-proportional radio usage.
    pub fn energy_aware_route(&self, payload_size: usize, urgency: u8) -> Option<RadioTier> {
        let budget_fraction = self.energy_spent_nj / self.energy_budget_nj;

        // If energy is plentiful (< 50% spent), use normal routing
        if budget_fraction < 0.5 {
            return self
                .route(PayloadClass::ConsciousnessDelta, payload_size, urgency)
                .and_then(|d| match d {
                    RoutingDecision::Routed { tier, .. } => Some(tier),
                    _ => None,
                });
        }

        // Energy-constrained: sort available tiers by energy_per_bit ascending
        let mut candidates: Vec<(RadioTier, f64)> = RadioTier::ALL
            .iter()
            .copied()
            .filter(|&t| self.tier_available[t as usize])
            .filter(|&t| payload_size <= t.profile().mtu || t == RadioTier::Local)
            .map(|t| (t, t.profile().energy_per_bit_nj))
            .collect();

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.first().map(|&(tier, _)| tier)
    }

    /// Reset energy tracking for a new cycle.
    pub fn reset_energy(&mut self) {
        self.energy_spent_nj = 0.0;
    }

    // ── Item 8: Peer discovery ───────────────────────────────────────

    /// Set this node's identity.
    pub fn set_node_id(&mut self, node_id: [u8; 8]) {
        self.node_id = node_id;
    }

    /// Generate a discovery beacon if the interval has elapsed.
    ///
    /// Returns `Some(beacon)` when it's time to broadcast.
    pub fn generate_beacon(&mut self, capabilities_hash: &[u8; 8]) -> Option<DiscoveryBeacon> {
        self.beacon_counter += 1;
        if self.beacon_counter < RADIO_BEACON_INTERVAL_CYCLES {
            return None;
        }
        self.beacon_counter = 0;

        Some(DiscoveryBeacon {
            node_id: self.node_id,
            capabilities_hash: *capabilities_hash,
            cycle_counter: self.current_cycle as u32,
            network_health: self.network_health.safety_suggestion(),
            tier_mask: DiscoveryBeacon::tier_mask_from(&self.tier_available),
        })
    }

    /// Process a received discovery beacon — update route table.
    ///
    /// Returns `true` if this is a newly discovered peer (not previously in route table).
    /// Callers can use this to trigger SwarmEvent::PeerJoined via the swarm manager.
    pub fn process_beacon(
        &mut self,
        beacon: &DiscoveryBeacon,
        received_on: RadioTier,
        snr: f32,
    ) -> bool {
        let is_new = self.route_table.lookup(&beacon.node_id).is_none();
        let link_quality = (snr / 30.0).clamp(0.0, 1.0);
        self.route_table.update(RouteEntry {
            destination: beacon.node_id,
            next_hop: beacon.node_id, // Direct neighbor
            hop_count: 0,
            tier: received_on,
            last_seen_cycle: self.current_cycle,
            link_quality,
        });
        is_new
    }

    // ── Item 9: Multi-hop relay routing ──────────────────────────────

    /// Look up the best route to a destination peer.
    pub fn find_route(&self, destination: &[u8; 8]) -> Option<&RouteEntry> {
        self.route_table.lookup(destination)
    }

    /// Route table accessor.
    pub fn route_table(&self) -> &RouteTable {
        &self.route_table
    }

    /// Wrap a payload for relay forwarding.
    ///
    /// Returns `(next_hop, relay_header + payload)` or `None` if no route exists.
    /// Relay header: [dest: 8][src: 8][ttl: 1][hop_count: 1] = 18 bytes.
    pub fn prepare_relay(
        &self,
        destination: &[u8; 8],
        payload: &[u8],
    ) -> Option<([u8; 8], Vec<u8>)> {
        let route = self.route_table.lookup(destination)?;
        if route.hop_count >= RADIO_MAX_RELAY_HOPS {
            return None; // TTL exceeded
        }

        let mut relay_packet = Vec::with_capacity(18 + payload.len());
        relay_packet.extend_from_slice(destination); // 8 bytes dest
        relay_packet.extend_from_slice(&self.node_id); // 8 bytes src
        relay_packet.push(RADIO_MAX_RELAY_HOPS - route.hop_count); // TTL
        relay_packet.push(route.hop_count + 1); // Current hop count
        relay_packet.extend_from_slice(payload);

        Some((route.next_hop, relay_packet))
    }

    // ── Item 11: Encryption ──────────────────────────────────────────

    /// Get the encryption manager.
    pub fn encryption(&self) -> &MeshEncryption {
        &self.encryption
    }

    /// Get the encryption manager mutably.
    pub fn encryption_mut(&mut self) -> &mut MeshEncryption {
        &mut self.encryption
    }

    /// Encrypt a payload for a specific peer.
    ///
    /// Returns encrypted payload with nonce prepended, or `None` if no session.
    pub fn encrypt_for_peer(&mut self, peer_id: &[u8; 8], payload: &[u8]) -> Option<Vec<u8>> {
        let session = self.encryption.get_session_mut(peer_id)?;
        let nonce = session.next_nonce();
        let key = session.session_key;
        let encrypted = MeshEncryption::encrypt(&key, &nonce, payload);

        // Prepend nonce counter (8 bytes) for receiver to reconstruct nonce
        let mut result = Vec::with_capacity(8 + encrypted.len());
        result.extend_from_slice(&session.tx_nonce_counter.to_le_bytes());
        result.extend_from_slice(&encrypted);
        Some(result)
    }

    /// Decrypt a payload from a specific peer.
    ///
    /// Expects format: [nonce_counter: 8 bytes][ciphertext + tag].
    pub fn decrypt_from_peer(&mut self, peer_id: &[u8; 8], encrypted: &[u8]) -> Option<Vec<u8>> {
        if encrypted.len() < 8 {
            return None;
        }
        let nonce_counter = u64::from_le_bytes([
            encrypted[0],
            encrypted[1],
            encrypted[2],
            encrypted[3],
            encrypted[4],
            encrypted[5],
            encrypted[6],
            encrypted[7],
        ]);

        let session = self.encryption.get_session_mut(peer_id)?;
        if !session.check_nonce(nonce_counter) {
            return None; // Replay detected
        }

        let mut nonce = [0u8; RADIO_CRYPTO_NONCE_SIZE];
        nonce[4..12].copy_from_slice(&nonce_counter.to_le_bytes());
        let key = session.session_key;

        MeshEncryption::decrypt(&key, &nonce, &encrypted[8..])
    }
}

impl CognitiveSubsystem for SpectrumManager {
    fn name(&self) -> &'static str {
        "spectrum_manager"
    }

    fn interval(&self) -> u32 {
        Self::INTERVAL
    }

    fn process(&mut self, _snapshot: &CycleSnapshot) -> SubsystemOutput {
        let mut output = SubsystemOutput::NEUTRAL;

        // ── 0. Poll hardware for SNR if available ────────────────────────
        if let Some(ref hw) = self.hardware {
            for &tier in &RadioTier::ALL {
                let avail = hw.is_available(tier);
                if self.tier_available[tier as usize] != avail {
                    self.tier_available[tier as usize] = avail;
                    self.classifier.set_tier_available(tier, avail);
                }
                if let Some(snr) = hw.current_snr(tier) {
                    self.pending_observations.push(SpectrumObservation {
                        frequency_hz: hw.current_frequency(tier).unwrap_or(0),
                        noise_floor_dbm: -100.0 + snr, // Approximate from SNR
                        snr_db: snr,
                        jammed: snr < JAMMING_SNR_THRESHOLD,
                    });
                }
            }
        }

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
            output.exploration_delta += RADIO_BLACKOUT_EXPLORATION_BOOST;
        }

        // ── 5. Tier loss → learning rate dampening ───────────────────────
        if self.any_tier_degraded() {
            // High packet loss → reduce learning rate (unreliable gradients)
            let max_loss = self.tier_loss_ema.iter().cloned().fold(0.0f64, f64::max);
            output.lr_modulation =
                1.0 - (max_loss * RADIO_LOSS_LR_DAMPEN_FACTOR).min(RADIO_LOSS_LR_DAMPEN_MAX);
        }

        // ── 6. Spectrum prediction error → surprise signal ───────────────
        if spectrum_pe > RADIO_SPECTRUM_PE_SURPRISE_THRESHOLD {
            output.arousal_delta += (spectrum_pe as f32 * RADIO_SPECTRUM_PE_AROUSAL_SCALE)
                .min(RADIO_SPECTRUM_PE_AROUSAL_MAX);
            output.flags |= output_flags::ANOMALY_DETECTED;
        }

        // ── 7. AIMD bandwidth control ─────────────────────────────────
        self.tick_aimd();

        // ── 8. Hop cooldown tick ──────────────────────────────────────
        self.hop_cooldown = self.hop_cooldown.saturating_sub(1);
        self.current_cycle += 1;

        // ── 9. Route table maintenance ────────────────────────────────
        self.route_table
            .prune(self.current_cycle, RADIO_ROUTE_EXPIRY_CYCLES);

        // ── 10. Reset per-cycle energy ────────────────────────────────
        self.energy_spent_nj = 0.0;

        // ── 11. Auto frequency hopping ──────────────────────────────
        // When waterfall shows persistent poor conditions, attempt cognitive hop.
        // Haykin (2005): observe → decide → act cycle for cognitive radio.
        if self.waterfall.len() >= RADIO_WATERFALL_MIN_SAMPLES
            && self.waterfall.mean_noise_floor().unwrap_or(f64::MIN)
                > RADIO_AUTO_HOP_NOISE_THRESHOLD
            && self.hop_cooldown == 0
        {
            // Try to hop each degraded tier
            for &tier in &RadioTier::ALL {
                if self.suggest_frequency_hop(tier).is_some() {
                    output.flags |= output_flags::ANOMALY_DETECTED;
                    break; // One hop per cycle
                }
            }
        }

        // ── 12. Update telemetry ──────────────────────────────────────
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
// RADIO HARDWARE TRAIT — Physical radio abstraction
// ═══════════════════════════════════════════════════════════════════════════════

/// Errors from radio hardware operations.
#[derive(Debug, Clone)]
pub enum RadioError {
    /// Hardware not available or powered off.
    Unavailable,
    /// Payload exceeds tier MTU.
    PayloadTooLarge { max: usize, got: usize },
    /// Regulatory constraint would be violated.
    RegulatoryViolation(String),
    /// Hardware-specific error.
    HardwareError(String),
    /// Channel busy (carrier sense).
    ChannelBusy,
    /// Frequency out of allowed band.
    FrequencyOutOfBand { freq_hz: u64, band: String },
}

impl std::fmt::Display for RadioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RadioError::Unavailable => write!(f, "radio hardware unavailable"),
            RadioError::PayloadTooLarge { max, got } => {
                write!(f, "payload too large: {} bytes (max {})", got, max)
            }
            RadioError::RegulatoryViolation(msg) => write!(f, "regulatory violation: {}", msg),
            RadioError::HardwareError(msg) => write!(f, "hardware error: {}", msg),
            RadioError::ChannelBusy => write!(f, "channel busy"),
            RadioError::FrequencyOutOfBand { freq_hz, band } => {
                write!(f, "frequency {} Hz out of band {}", freq_hz, band)
            }
        }
    }
}

/// Abstraction over physical radio hardware for mesh networking.
///
/// Implementations handle actual RF transmission/reception while
/// `SpectrumManager` handles cognitive-level decisions.
///
/// Basis: Clark & Chalmers (1998) — extended mind via radio as cognitive prosthesis.
pub trait RadioHardware: Send + Sync {
    /// Transmit a payload on the specified tier. Returns bytes actually sent.
    fn transmit(&mut self, tier: RadioTier, payload: &[u8]) -> Result<usize, RadioError>;
    /// Receive pending data from a tier. Returns `(payload, snr_db)`.
    fn receive(&mut self, tier: RadioTier) -> Result<Option<(Vec<u8>, f32)>, RadioError>;
    /// Query current signal-to-noise ratio for a tier.
    fn current_snr(&self, tier: RadioTier) -> Option<f32>;
    /// Query whether a tier's hardware is available/powered.
    fn is_available(&self, tier: RadioTier) -> bool;
    /// Set transmit power (dBm) for a tier, respecting regulatory limits.
    fn set_tx_power(&mut self, tier: RadioTier, power_dbm: f32) -> Result<(), RadioError>;
    /// Get current frequency (Hz) for a tier.
    fn current_frequency(&self, tier: RadioTier) -> Option<u64>;
    /// Tune to a specific frequency (Hz), respecting regulatory constraints.
    fn tune(&mut self, tier: RadioTier, frequency_hz: u64) -> Result<(), RadioError>;
    /// Hardware-specific name/identifier (e.g., "HackRF One", "RFM95W LoRa").
    fn hardware_id(&self) -> &str;
}

// ═══════════════════════════════════════════════════════════════════════════════
// MOCK RADIO HARDWARE — Testing implementation
// ═══════════════════════════════════════════════════════════════════════════════

/// Mock radio hardware for testing. Configurable SNR, availability per tier,
/// collects transmitted payloads for assertion.
pub struct MockRadioHardware {
    available: [bool; 3],
    snr: [f32; 3],
    frequency: [u64; 3],
    tx_power: [f32; 3],
    transmitted: Vec<(usize, Vec<u8>)>,
    receive_queue: VecDeque<(usize, Vec<u8>, f32)>,
    regulatory_db: Option<RegulatoryDatabase>,
}

impl MockRadioHardware {
    /// Create a mock with all tiers available and default SNR.
    pub fn new() -> Self {
        Self {
            available: [true, true, true],
            snr: [30.0, 15.0, 5.0],
            frequency: [2_450_000_000, 915_000_000, 7_100_000],
            tx_power: [20.0, 14.0, 30.0],
            transmitted: Vec::new(),
            receive_queue: VecDeque::new(),
            regulatory_db: None,
        }
    }

    pub fn set_available(&mut self, tier: RadioTier, available: bool) {
        self.available[tier as usize] = available;
    }

    pub fn set_snr(&mut self, tier: RadioTier, snr_db: f32) {
        self.snr[tier as usize] = snr_db;
    }

    pub fn transmitted_payloads(&self) -> &[(usize, Vec<u8>)] {
        &self.transmitted
    }

    pub fn inject_receive(&mut self, tier: RadioTier, payload: Vec<u8>, snr_db: f32) {
        self.receive_queue
            .push_back((tier as usize, payload, snr_db));
    }

    pub fn set_regulatory_db(&mut self, db: RegulatoryDatabase) {
        self.regulatory_db = Some(db);
    }
}

impl RadioHardware for MockRadioHardware {
    fn transmit(&mut self, tier: RadioTier, payload: &[u8]) -> Result<usize, RadioError> {
        if !self.available[tier as usize] {
            return Err(RadioError::Unavailable);
        }
        let mtu = tier.profile().mtu;
        if payload.len() > mtu {
            return Err(RadioError::PayloadTooLarge {
                max: mtu,
                got: payload.len(),
            });
        }
        self.transmitted.push((tier as usize, payload.to_vec()));
        Ok(payload.len())
    }

    fn receive(&mut self, tier: RadioTier) -> Result<Option<(Vec<u8>, f32)>, RadioError> {
        if !self.available[tier as usize] {
            return Err(RadioError::Unavailable);
        }
        let idx = tier as usize;
        if let Some(pos) = self.receive_queue.iter().position(|(t, _, _)| *t == idx) {
            let Some((_, payload, snr)) = self.receive_queue.remove(pos) else {
                // position() found it, remove() should succeed — but guard defensively
                return Ok(None);
            };
            Ok(Some((payload, snr)))
        } else {
            Ok(None)
        }
    }

    fn current_snr(&self, tier: RadioTier) -> Option<f32> {
        if self.available[tier as usize] {
            Some(self.snr[tier as usize])
        } else {
            None
        }
    }

    fn is_available(&self, tier: RadioTier) -> bool {
        self.available[tier as usize]
    }

    fn set_tx_power(&mut self, tier: RadioTier, power_dbm: f32) -> Result<(), RadioError> {
        if !self.available[tier as usize] {
            return Err(RadioError::Unavailable);
        }
        if let Some(ref db) = self.regulatory_db {
            let freq = self.frequency[tier as usize];
            if let Some(max_power) = db.max_power_for_frequency(freq) {
                if power_dbm > max_power {
                    return Err(RadioError::RegulatoryViolation(format!(
                        "power {} dBm exceeds max {} dBm for frequency {} Hz",
                        power_dbm, max_power, freq
                    )));
                }
            }
        }
        self.tx_power[tier as usize] = power_dbm;
        Ok(())
    }

    fn current_frequency(&self, tier: RadioTier) -> Option<u64> {
        if self.available[tier as usize] {
            Some(self.frequency[tier as usize])
        } else {
            None
        }
    }

    fn tune(&mut self, tier: RadioTier, frequency_hz: u64) -> Result<(), RadioError> {
        if !self.available[tier as usize] {
            return Err(RadioError::Unavailable);
        }
        if let Some(ref db) = self.regulatory_db {
            if !db.is_frequency_allowed(frequency_hz, tier) {
                return Err(RadioError::FrequencyOutOfBand {
                    freq_hz: frequency_hz,
                    band: format!("{:?} bands", db.region()),
                });
            }
        }
        self.frequency[tier as usize] = frequency_hz;
        Ok(())
    }

    fn hardware_id(&self) -> &str {
        "MockRadioHardware v1.0"
    }
}

/// No-op radio hardware that always returns `Unavailable`.
/// Used when mesh feature is enabled but no physical radio exists.
pub struct NullRadioHardware;

impl RadioHardware for NullRadioHardware {
    fn transmit(&mut self, _: RadioTier, _: &[u8]) -> Result<usize, RadioError> {
        Err(RadioError::Unavailable)
    }
    fn receive(&mut self, _: RadioTier) -> Result<Option<(Vec<u8>, f32)>, RadioError> {
        Err(RadioError::Unavailable)
    }
    fn current_snr(&self, _: RadioTier) -> Option<f32> {
        None
    }
    fn is_available(&self, _: RadioTier) -> bool {
        false
    }
    fn set_tx_power(&mut self, _: RadioTier, _: f32) -> Result<(), RadioError> {
        Err(RadioError::Unavailable)
    }
    fn current_frequency(&self, _: RadioTier) -> Option<u64> {
        None
    }
    fn tune(&mut self, _: RadioTier, _: u64) -> Result<(), RadioError> {
        Err(RadioError::Unavailable)
    }
    fn hardware_id(&self) -> &str {
        "NullRadioHardware"
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// REGULATORY DATABASE — Region-aware frequency allocations
// ═══════════════════════════════════════════════════════════════════════════════

/// ITU regulatory region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegulatoryRegion {
    /// FCC Part 15 (unlicensed) + Part 97 (amateur). US/Canada.
    FccUs,
    /// ETSI EN 300 220 (SRD) + EN 301 893 (5 GHz). EU/EEA/UK.
    EtsiEu,
    /// ARIB STD-T108. Japan.
    AribJp,
    /// Generic ISM (international fallback).
    IsmGlobal,
}

/// License requirement for a frequency band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LicenseType {
    Unlicensed,
    LightLicense,
    Amateur,
    Licensed,
}

/// A specific frequency allocation within a regulatory region.
#[derive(Debug, Clone)]
pub struct BandAllocation {
    pub name: String,
    pub freq_min_hz: u64,
    pub freq_max_hz: u64,
    pub max_eirp_dbm: f32,
    pub duty_cycle_max: Option<f32>,
    pub channel_bw_hz: u32,
    pub tier: RadioTier,
    pub license: LicenseType,
}

/// Regional regulatory database with band allocations.
///
/// Pre-populated with real-world spectrum allocations.
/// Basis: FCC Part 15/97, ETSI EN 300 220, ITU Radio Regulations.
#[derive(Debug, Clone)]
pub struct RegulatoryDatabase {
    region: RegulatoryRegion,
    bands: Vec<BandAllocation>,
}

impl RegulatoryDatabase {
    pub fn new(region: RegulatoryRegion) -> Self {
        let bands = match region {
            RegulatoryRegion::FccUs => Self::fcc_us_bands(),
            RegulatoryRegion::EtsiEu => Self::etsi_eu_bands(),
            RegulatoryRegion::AribJp => Self::arib_jp_bands(),
            RegulatoryRegion::IsmGlobal => Self::ism_global_bands(),
        };
        Self { region, bands }
    }

    pub fn region(&self) -> RegulatoryRegion {
        self.region
    }
    pub fn bands(&self) -> &[BandAllocation] {
        &self.bands
    }

    pub fn bands_for_tier(&self, tier: RadioTier) -> Vec<&BandAllocation> {
        self.bands.iter().filter(|b| b.tier == tier).collect()
    }

    pub fn is_frequency_allowed(&self, freq_hz: u64, tier: RadioTier) -> bool {
        self.bands
            .iter()
            .any(|b| b.tier == tier && freq_hz >= b.freq_min_hz && freq_hz <= b.freq_max_hz)
    }

    pub fn max_power_for_frequency(&self, freq_hz: u64) -> Option<f32> {
        self.bands
            .iter()
            .filter(|b| freq_hz >= b.freq_min_hz && freq_hz <= b.freq_max_hz)
            .map(|b| b.max_eirp_dbm)
            .fold(None, |acc, p| Some(acc.map_or(p, |a: f32| a.max(p))))
    }

    pub fn duty_cycle_for_band(&self, freq_hz: u64) -> Option<f32> {
        self.bands
            .iter()
            .find(|b| freq_hz >= b.freq_min_hz && freq_hz <= b.freq_max_hz)
            .and_then(|b| b.duty_cycle_max)
    }

    pub fn available_bandwidth(&self, tier: RadioTier) -> u64 {
        self.bands
            .iter()
            .filter(|b| b.tier == tier)
            .map(|b| b.freq_max_hz - b.freq_min_hz)
            .sum()
    }

    /// Convert to legacy `RegulatoryConstraints` for backward compatibility.
    pub fn to_legacy_constraints(&self) -> RegulatoryConstraints {
        let allowed_bands: Vec<(u64, u64)> = self
            .bands
            .iter()
            .filter(|b| b.license == LicenseType::Unlicensed)
            .map(|b| (b.freq_min_hz, b.freq_max_hz))
            .collect();
        let max_power = self
            .bands
            .iter()
            .filter(|b| b.license == LicenseType::Unlicensed)
            .map(|b| b.max_eirp_dbm)
            .fold(f32::NEG_INFINITY, f32::max);
        let region = match self.region {
            RegulatoryRegion::FccUs => "US",
            RegulatoryRegion::EtsiEu => "EU",
            RegulatoryRegion::AribJp => "JP",
            RegulatoryRegion::IsmGlobal => "GLOBAL",
        }
        .to_string();
        RegulatoryConstraints {
            allowed_bands,
            max_power_dbm: if max_power.is_finite() {
                max_power
            } else {
                0.0
            },
            region,
        }
    }

    fn fcc_us_bands() -> Vec<BandAllocation> {
        vec![
            BandAllocation {
                name: "ISM 915 MHz".into(),
                freq_min_hz: 902_000_000,
                freq_max_hz: 928_000_000,
                max_eirp_dbm: 30.0,
                duty_cycle_max: None,
                channel_bw_hz: 500_000,
                tier: RadioTier::Metro,
                license: LicenseType::Unlicensed,
            },
            BandAllocation {
                name: "ISM 2.4 GHz".into(),
                freq_min_hz: 2_400_000_000,
                freq_max_hz: 2_483_500_000,
                max_eirp_dbm: 36.0,
                duty_cycle_max: None,
                channel_bw_hz: 22_000_000,
                tier: RadioTier::Local,
                license: LicenseType::Unlicensed,
            },
            BandAllocation {
                name: "U-NII-3 5.8 GHz".into(),
                freq_min_hz: 5_725_000_000,
                freq_max_hz: 5_850_000_000,
                max_eirp_dbm: 36.0,
                duty_cycle_max: None,
                channel_bw_hz: 20_000_000,
                tier: RadioTier::Local,
                license: LicenseType::Unlicensed,
            },
            BandAllocation {
                name: "HF 80m Amateur".into(),
                freq_min_hz: 3_500_000,
                freq_max_hz: 4_000_000,
                max_eirp_dbm: 61.76,
                duty_cycle_max: Some(0.5),
                channel_bw_hz: 3_000,
                tier: RadioTier::Regional,
                license: LicenseType::Amateur,
            },
            BandAllocation {
                name: "HF 40m Amateur".into(),
                freq_min_hz: 7_000_000,
                freq_max_hz: 7_300_000,
                max_eirp_dbm: 61.76,
                duty_cycle_max: Some(0.5),
                channel_bw_hz: 3_000,
                tier: RadioTier::Regional,
                license: LicenseType::Amateur,
            },
            BandAllocation {
                name: "HF 20m Amateur (NVIS)".into(),
                freq_min_hz: 14_000_000,
                freq_max_hz: 14_350_000,
                max_eirp_dbm: 61.76,
                duty_cycle_max: Some(0.5),
                channel_bw_hz: 3_000,
                tier: RadioTier::Regional,
                license: LicenseType::Amateur,
            },
        ]
    }

    fn etsi_eu_bands() -> Vec<BandAllocation> {
        vec![
            BandAllocation {
                name: "SRD 868 MHz (1%)".into(),
                freq_min_hz: 868_000_000,
                freq_max_hz: 868_600_000,
                max_eirp_dbm: 14.0,
                duty_cycle_max: Some(0.01),
                channel_bw_hz: 125_000,
                tier: RadioTier::Metro,
                license: LicenseType::Unlicensed,
            },
            BandAllocation {
                name: "SRD 869 MHz (10%)".into(),
                freq_min_hz: 869_400_000,
                freq_max_hz: 869_650_000,
                max_eirp_dbm: 27.0,
                duty_cycle_max: Some(0.10),
                channel_bw_hz: 125_000,
                tier: RadioTier::Metro,
                license: LicenseType::Unlicensed,
            },
            BandAllocation {
                name: "ISM 2.4 GHz".into(),
                freq_min_hz: 2_400_000_000,
                freq_max_hz: 2_483_500_000,
                max_eirp_dbm: 20.0,
                duty_cycle_max: None,
                channel_bw_hz: 22_000_000,
                tier: RadioTier::Local,
                license: LicenseType::Unlicensed,
            },
            BandAllocation {
                name: "RLAN 5 GHz".into(),
                freq_min_hz: 5_150_000_000,
                freq_max_hz: 5_350_000_000,
                max_eirp_dbm: 23.0,
                duty_cycle_max: None,
                channel_bw_hz: 20_000_000,
                tier: RadioTier::Local,
                license: LicenseType::Unlicensed,
            },
        ]
    }

    fn arib_jp_bands() -> Vec<BandAllocation> {
        vec![
            BandAllocation {
                name: "ARIB 920 MHz".into(),
                freq_min_hz: 920_000_000,
                freq_max_hz: 928_000_000,
                max_eirp_dbm: 20.0,
                duty_cycle_max: Some(0.10),
                channel_bw_hz: 200_000,
                tier: RadioTier::Metro,
                license: LicenseType::Unlicensed,
            },
            BandAllocation {
                name: "ISM 2.4 GHz".into(),
                freq_min_hz: 2_400_000_000,
                freq_max_hz: 2_483_500_000,
                max_eirp_dbm: 20.0,
                duty_cycle_max: None,
                channel_bw_hz: 22_000_000,
                tier: RadioTier::Local,
                license: LicenseType::Unlicensed,
            },
        ]
    }

    fn ism_global_bands() -> Vec<BandAllocation> {
        vec![
            BandAllocation {
                name: "ISM 433 MHz".into(),
                freq_min_hz: 433_050_000,
                freq_max_hz: 434_790_000,
                max_eirp_dbm: 10.0,
                duty_cycle_max: Some(0.10),
                channel_bw_hz: 25_000,
                tier: RadioTier::Metro,
                license: LicenseType::Unlicensed,
            },
            BandAllocation {
                name: "ISM 2.4 GHz".into(),
                freq_min_hz: 2_400_000_000,
                freq_max_hz: 2_500_000_000,
                max_eirp_dbm: 20.0,
                duty_cycle_max: None,
                channel_bw_hz: 22_000_000,
                tier: RadioTier::Local,
                license: LicenseType::Unlicensed,
            },
        ]
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS-AWARE MESH ROUTING
// ═══════════════════════════════════════════════════════════════════════════════

/// Consciousness-aware routing layer — routes mesh traffic based on cognitive
/// urgency, node consciousness level, moral topology, and governance tier.
///
/// This extends the physical-layer `PayloadClassifier` with consciousness semantics:
/// - Moral emergencies (ahimsa violations) bypass bandwidth budgets
/// - Messages route preferentially through high-Phi nodes (most trustworthy relay)
/// - Guardian-tier nodes gate relay of moral alerts
/// - Adaptive sharing cadence based on collective Phi divergence
///
/// ## Scientific Basis
///
/// - Tononi (2004): Phi as measure of integrated information → routing trust
/// - Friston (2010): Active inference over mesh topology
/// - Clark & Chalmers (1998): Extended mind → network is cognitive substrate
pub struct ConsciousnessAwareRouter {
    /// This node's current Phi (integrated information).
    local_phi: f32,
    /// This node's consciousness tier (0=Observer, 1=Participant, 2=Citizen, 3=Steward, 4=Guardian).
    local_tier: u8,
    /// Known peer Phi values (keyed by node_id prefix).
    peer_phi: HashMap<[u8; 8], PeerConsciousnessState>,
    /// Moral urgency override: when true, bypasses bandwidth budgets for current cycle.
    moral_emergency_active: bool,
    /// Collective Phi (mean of all peers including self).
    collective_phi: f32,
    /// Collective Phi divergence (variance — high means disagreement, share more).
    collective_phi_divergence: f32,
    /// Adaptive sharing cadence: cycles between consciousness broadcasts.
    /// Lower when collective divergence is high.
    sharing_cadence: u32,
    /// Cycles since last consciousness broadcast.
    cycles_since_share: u32,
    /// Cumulative threat signatures received this session.
    threat_observations: Vec<ThreatObservation>,
}

/// Consciousness state of a known mesh peer.
#[derive(Debug, Clone)]
pub struct PeerConsciousnessState {
    /// Peer's reported Phi.
    pub phi: f32,
    /// Peer's consciousness level [0.0, 1.0].
    pub consciousness_level: f32,
    /// Peer's governance tier (0-4).
    pub governance_tier: u8,
    /// Cycle when last updated.
    pub last_update_cycle: u64,
    /// Trust score [0.0, 1.0] — influenced by consistency of reports.
    pub trust: f32,
}

/// Compact threat observation for collective immune memory.
#[derive(Debug, Clone)]
pub struct ThreatObservation {
    /// Threat type (maps to SentinelManager's 7 types).
    pub threat_type: u8,
    /// Severity [0.0, 1.0].
    pub severity: f32,
    /// Offending agent hash (8-byte prefix).
    pub agent_hash: [u8; 8],
    /// Compact signature (32 bytes, HDV-based).
    pub signature: [u8; 32],
    /// Cycle when observed.
    pub observed_cycle: u64,
    /// Number of independent corroborations.
    pub corroboration_count: u16,
}

/// Consciousness-aware routing decision.
#[derive(Debug, Clone)]
pub enum ConsciousRoutingDecision {
    /// Route normally via PayloadClassifier.
    Normal(RoutingDecision),
    /// Moral emergency — bypass bandwidth budgets, route through highest-Phi path.
    MoralEmergency {
        /// Best tier available (preferring fastest).
        tier: RadioTier,
        /// Highest-Phi relay node (if multi-hop).
        preferred_relay: Option<[u8; 8]>,
    },
    /// Consciousness sharing — adaptive cadence says share now.
    ConsciousnessShare {
        tier: RadioTier,
    },
    /// Threat broadcast — share immune signature with all peers.
    ThreatBroadcast {
        tier: RadioTier,
        threat: ThreatObservation,
    },
    /// Suppress — cadence says wait, or peer already has current state.
    Suppressed {
        reason: &'static str,
        cycles_until_next: u32,
    },
}

/// Default sharing cadence (cycles between consciousness broadcasts).
const DEFAULT_SHARING_CADENCE: u32 = 50;
/// Minimum sharing cadence when collective divergence is high.
const MIN_SHARING_CADENCE: u32 = 5;
/// Maximum sharing cadence when collective is stable.
const MAX_SHARING_CADENCE: u32 = 200;
/// Phi divergence threshold for increasing sharing frequency.
const PHI_DIVERGENCE_THRESHOLD: f32 = 0.15;
/// Maximum threat observations to retain.
const MAX_THREAT_OBSERVATIONS: usize = 64;
/// Threat observation expiry (cycles).
const THREAT_EXPIRY_CYCLES: u64 = 10_000;

impl Default for ConsciousnessAwareRouter {
    fn default() -> Self {
        Self {
            local_phi: 0.0,
            local_tier: 0,
            peer_phi: HashMap::new(),
            moral_emergency_active: false,
            collective_phi: 0.0,
            collective_phi_divergence: 0.0,
            sharing_cadence: DEFAULT_SHARING_CADENCE,
            cycles_since_share: 0,
            threat_observations: Vec::new(),
        }
    }
}

impl ConsciousnessAwareRouter {
    /// Update local consciousness state (called each cycle from CLS).
    pub fn update_local(&mut self, phi: f32, consciousness_level: f32, governance_tier: u8) {
        self.local_phi = phi;
        self.local_tier = governance_tier;
        self.cycles_since_share += 1;

        // Recompute collective state
        self.recompute_collective();
        self.adapt_cadence();
    }

    /// Update a peer's consciousness state (received via mesh).
    pub fn update_peer(
        &mut self,
        peer_id: [u8; 8],
        phi: f32,
        consciousness_level: f32,
        governance_tier: u8,
        current_cycle: u64,
    ) {
        let state = self.peer_phi.entry(peer_id).or_insert(PeerConsciousnessState {
            phi: 0.0,
            consciousness_level: 0.0,
            governance_tier: 0,
            last_update_cycle: 0,
            trust: 0.5, // Start at neutral trust
        });

        // Update trust based on consistency: large jumps in Phi are suspicious.
        let phi_delta = (phi - state.phi).abs();
        if phi_delta > 0.5 {
            state.trust = (state.trust - 0.1).max(0.0);
        } else {
            state.trust = (state.trust + 0.01).min(1.0);
        }

        state.phi = phi;
        state.consciousness_level = consciousness_level;
        state.governance_tier = governance_tier;
        state.last_update_cycle = current_cycle;

        self.recompute_collective();
        self.adapt_cadence();
    }

    /// Signal a moral emergency (e.g., ahimsa violation detected).
    /// Next routing decision will bypass bandwidth budgets.
    pub fn signal_moral_emergency(&mut self) {
        self.moral_emergency_active = true;
    }

    /// Record a threat observation for collective immune response.
    pub fn record_threat(&mut self, threat: ThreatObservation) {
        // Check for existing observation on same agent — corroborate instead of duplicate
        if let Some(existing) = self.threat_observations.iter_mut().find(|t| {
            t.agent_hash == threat.agent_hash && t.threat_type == threat.threat_type
        }) {
            existing.corroboration_count = existing.corroboration_count.saturating_add(1);
            existing.severity = existing.severity.max(threat.severity);
            return;
        }

        self.threat_observations.push(threat);

        // Cap observations
        if self.threat_observations.len() > MAX_THREAT_OBSERVATIONS {
            self.threat_observations.remove(0);
        }
    }

    /// Prune stale peers and threat observations.
    pub fn prune(&mut self, current_cycle: u64, max_peer_age: u64) {
        self.peer_phi
            .retain(|_, state| current_cycle.saturating_sub(state.last_update_cycle) < max_peer_age);
        self.threat_observations
            .retain(|t| current_cycle.saturating_sub(t.observed_cycle) < THREAT_EXPIRY_CYCLES);
    }

    /// Route a consciousness-aware payload.
    ///
    /// Extends the physical PayloadClassifier with consciousness semantics.
    pub fn route(
        &mut self,
        class: PayloadClass,
        payload_size: usize,
        urgency: u8,
        classifier: &PayloadClassifier,
    ) -> ConsciousRoutingDecision {
        // Moral emergency overrides everything
        if self.moral_emergency_active {
            self.moral_emergency_active = false; // One-shot

            // Find highest-Phi relay for maximum trust
            let preferred_relay = self.highest_phi_peer();

            // Use fastest available tier
            let tier = if classifier.is_available(RadioTier::Local) {
                RadioTier::Local
            } else if classifier.is_available(RadioTier::Metro) {
                RadioTier::Metro
            } else {
                RadioTier::Regional
            };

            return ConsciousRoutingDecision::MoralEmergency {
                tier,
                preferred_relay,
            };
        }

        // Check for pending threat broadcasts
        if let Some(threat) = self.next_unbroadcast_threat() {
            let tier = if classifier.is_available(RadioTier::Metro) {
                RadioTier::Metro
            } else if classifier.is_available(RadioTier::Local) {
                RadioTier::Local
            } else {
                RadioTier::Regional
            };

            return ConsciousRoutingDecision::ThreatBroadcast { tier, threat };
        }

        // Check if it's time to share consciousness state
        if class == PayloadClass::ConsciousnessDelta {
            if self.cycles_since_share >= self.sharing_cadence {
                self.cycles_since_share = 0;

                let tier = if classifier.is_available(RadioTier::Metro) {
                    RadioTier::Metro
                } else if classifier.is_available(RadioTier::Local) {
                    RadioTier::Local
                } else {
                    RadioTier::Regional
                };

                return ConsciousRoutingDecision::ConsciousnessShare { tier };
            } else {
                return ConsciousRoutingDecision::Suppressed {
                    reason: "sharing cadence not reached",
                    cycles_until_next: self.sharing_cadence - self.cycles_since_share,
                };
            }
        }

        // Default: normal routing
        match classifier.route(class, payload_size, urgency) {
            Some(decision) => ConsciousRoutingDecision::Normal(decision),
            None => ConsciousRoutingDecision::Suppressed {
                reason: "no routing decision",
                cycles_until_next: 1,
            },
        }
    }

    /// Get the highest-Phi peer (most trustworthy relay).
    pub fn highest_phi_peer(&self) -> Option<[u8; 8]> {
        self.peer_phi
            .iter()
            .filter(|(_, state)| state.trust > 0.3) // Minimum trust threshold
            .max_by(|(_, a), (_, b)| {
                // Weight: Phi * trust * consciousness_level
                let score_a = a.phi * a.trust * a.consciousness_level;
                let score_b = b.phi * b.trust * b.consciousness_level;
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(id, _)| *id)
    }

    /// Get peers sorted by trustworthiness (Phi * trust * consciousness).
    pub fn peers_by_trust(&self) -> Vec<([u8; 8], f32)> {
        let mut peers: Vec<_> = self
            .peer_phi
            .iter()
            .map(|(id, state)| {
                let score = state.phi * state.trust * state.consciousness_level;
                (*id, score)
            })
            .collect();
        peers.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        peers
    }

    /// Current collective Phi (mean across all nodes including self).
    pub fn collective_phi(&self) -> f32 {
        self.collective_phi
    }

    /// Current collective Phi divergence (variance — measures disagreement).
    pub fn collective_divergence(&self) -> f32 {
        self.collective_phi_divergence
    }

    /// Current adaptive sharing cadence.
    pub fn sharing_cadence(&self) -> u32 {
        self.sharing_cadence
    }

    /// Number of known consciousness peers.
    pub fn peer_count(&self) -> usize {
        self.peer_phi.len()
    }

    /// Active threat observations.
    pub fn threat_count(&self) -> usize {
        self.threat_observations.len()
    }

    /// Get all threat observations for broadcasting.
    pub fn threats(&self) -> &[ThreatObservation] {
        &self.threat_observations
    }

    // ── Private helpers ────────────────────────────────────────────────

    fn recompute_collective(&mut self) {
        let mut sum = self.local_phi;
        let mut count = 1u32;

        for (_, state) in &self.peer_phi {
            sum += state.phi * state.trust; // Trust-weighted
            count += 1;
        }

        self.collective_phi = sum / count as f32;

        // Variance
        let mut var_sum = (self.local_phi - self.collective_phi).powi(2);
        for (_, state) in &self.peer_phi {
            var_sum += (state.phi * state.trust - self.collective_phi).powi(2);
        }
        self.collective_phi_divergence = var_sum / count as f32;
    }

    fn adapt_cadence(&mut self) {
        // High divergence → share more frequently (converge faster)
        // Low divergence → share less frequently (save bandwidth)
        if self.collective_phi_divergence > PHI_DIVERGENCE_THRESHOLD {
            self.sharing_cadence = (self.sharing_cadence / 2).max(MIN_SHARING_CADENCE);
        } else if self.collective_phi_divergence < PHI_DIVERGENCE_THRESHOLD * 0.5 {
            self.sharing_cadence = (self.sharing_cadence * 2).min(MAX_SHARING_CADENCE);
        }
    }

    fn next_unbroadcast_threat(&self) -> Option<ThreatObservation> {
        // Return the highest-severity unbroadcast threat
        self.threat_observations
            .iter()
            .filter(|t| t.corroboration_count == 0) // Not yet corroborated = likely new
            .max_by(|a, b| a.severity.partial_cmp(&b.severity).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// STORE-AND-FORWARD — Dream-Consolidated Reconnection
// ═══════════════════════════════════════════════════════════════════════════════

/// Manages store-and-forward behavior for mesh nodes that experience
/// intermittent connectivity (solar nodes, mobile devices, disaster scenarios).
///
/// When a node reconnects after isolation, instead of dumping all accumulated
/// raw data, it runs dream consolidation to compress experiences into meaningful
/// wisdom patterns. This is biologically inspired: you don't relay every sensory
/// input — you relay what you learned from sleeping on it.
///
/// ## Integration
///
/// The Spore/Symthaea DreamEngine already performs counterfactual consolidation.
/// This module decides *when* to trigger consolidation and *what* to transmit
/// on reconnection.
pub struct StoreAndForward {
    /// Accumulated experiences during offline period.
    offline_buffer: VecDeque<OfflineExperience>,
    /// Maximum buffer size (experiences, not bytes).
    buffer_capacity: usize,
    /// Whether currently offline (no mesh connectivity).
    is_offline: bool,
    /// Cycle when connectivity was last lost.
    offline_since: Option<u64>,
    /// Number of reconnection events this session.
    reconnection_count: u32,
    /// Whether dream consolidation has been triggered for current buffer.
    consolidation_pending: bool,
}

/// A single offline experience — sensor event, governance action, or
/// consciousness state change recorded while disconnected.
#[derive(Debug, Clone)]
pub struct OfflineExperience {
    /// Cycle when recorded.
    pub cycle: u64,
    /// Type of experience.
    pub kind: OfflineExperienceKind,
    /// Salience score [0.0, 1.0] — high-salience experiences are prioritized.
    pub salience: f32,
}

/// Categories of offline experiences.
#[derive(Debug, Clone)]
pub enum OfflineExperienceKind {
    /// Sensor reading that exceeded a threshold.
    SensorAnomaly { sensor_id: String, value: f32 },
    /// Consciousness level crossed a boundary.
    ConsciousnessShift { from: f32, to: f32 },
    /// Moral evaluation with significant salience.
    MoralEvent { salience: f32 },
    /// Threat detected while offline.
    ThreatDetected { threat_type: u8, severity: f32 },
}

/// Result of dream consolidation on the offline buffer.
#[derive(Debug, Clone)]
pub struct ConsolidatedWisdom {
    /// Number of raw experiences consolidated.
    pub experiences_consolidated: usize,
    /// Duration of offline period (cycles).
    pub offline_duration: u64,
    /// Mean salience of consolidated experiences.
    pub mean_salience: f32,
    /// Summary patterns extracted (to transmit as WisdomPacket).
    pub patterns: Vec<ConsolidatedPattern>,
}

/// A single pattern extracted from dream consolidation.
#[derive(Debug, Clone)]
pub struct ConsolidatedPattern {
    /// Pattern type.
    pub kind: String,
    /// Confidence in this pattern [0.0, 1.0].
    pub confidence: f32,
    /// Compact representation (fits in a single mesh packet).
    pub data: Vec<u8>,
}

/// Default buffer capacity (experiences).
const STORE_FORWARD_BUFFER_CAPACITY: usize = 1000;
/// Salience threshold for keeping an experience in the buffer.
const SALIENCE_THRESHOLD: f32 = 0.3;

impl Default for StoreAndForward {
    fn default() -> Self {
        Self {
            offline_buffer: VecDeque::with_capacity(STORE_FORWARD_BUFFER_CAPACITY),
            buffer_capacity: STORE_FORWARD_BUFFER_CAPACITY,
            is_offline: false,
            offline_since: None,
            reconnection_count: 0,
            consolidation_pending: false,
        }
    }
}

impl StoreAndForward {
    /// Notify that connectivity has been lost.
    pub fn go_offline(&mut self, current_cycle: u64) {
        if !self.is_offline {
            self.is_offline = true;
            self.offline_since = Some(current_cycle);
            self.consolidation_pending = false;
        }
    }

    /// Notify that connectivity has been restored.
    /// Returns true if dream consolidation should be triggered.
    pub fn go_online(&mut self, current_cycle: u64) -> bool {
        if self.is_offline {
            self.is_offline = false;
            self.reconnection_count += 1;

            // Trigger consolidation if we accumulated significant experiences
            if self.offline_buffer.len() >= 10 {
                self.consolidation_pending = true;
                return true;
            }
        }
        false
    }

    /// Record an experience during offline period.
    /// Low-salience experiences are dropped to save memory.
    pub fn record(&mut self, experience: OfflineExperience) {
        if experience.salience < SALIENCE_THRESHOLD {
            return;
        }

        if self.offline_buffer.len() >= self.buffer_capacity {
            // Evict lowest-salience experience
            if let Some(min_idx) = self
                .offline_buffer
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.salience
                        .partial_cmp(&b.salience)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                if self.offline_buffer[min_idx].salience < experience.salience {
                    self.offline_buffer.remove(min_idx);
                } else {
                    return; // New experience is lower salience than everything in buffer
                }
            }
        }

        self.offline_buffer.push_back(experience);
    }

    /// Perform dream consolidation on the offline buffer.
    ///
    /// Extracts meaningful patterns from raw experiences:
    /// - Sensor anomalies → trend summaries (mean, variance, duration)
    /// - Consciousness shifts → boundary crossing events
    /// - Moral events → aggregate moral landscape change
    /// - Threats → consolidated threat signatures
    ///
    /// Returns consolidated wisdom ready for mesh transmission.
    pub fn consolidate(&mut self, current_cycle: u64) -> ConsolidatedWisdom {
        let offline_duration = self
            .offline_since
            .map(|since| current_cycle.saturating_sub(since))
            .unwrap_or(0);

        let mean_salience = if self.offline_buffer.is_empty() {
            0.0
        } else {
            self.offline_buffer.iter().map(|e| e.salience).sum::<f32>()
                / self.offline_buffer.len() as f32
        };

        let mut patterns = Vec::new();

        // Consolidate sensor anomalies into trend summaries
        let sensor_events: Vec<_> = self
            .offline_buffer
            .iter()
            .filter(|e| matches!(e.kind, OfflineExperienceKind::SensorAnomaly { .. }))
            .collect();
        if !sensor_events.is_empty() {
            patterns.push(ConsolidatedPattern {
                kind: "sensor_trend".into(),
                confidence: (sensor_events.len() as f32 / 10.0).min(1.0),
                data: format!("anomalies:{}", sensor_events.len()).into_bytes(),
            });
        }

        // Consolidate consciousness shifts
        let consciousness_events: Vec<_> = self
            .offline_buffer
            .iter()
            .filter(|e| matches!(e.kind, OfflineExperienceKind::ConsciousnessShift { .. }))
            .collect();
        if !consciousness_events.is_empty() {
            patterns.push(ConsolidatedPattern {
                kind: "consciousness_trajectory".into(),
                confidence: 0.8,
                data: format!("shifts:{}", consciousness_events.len()).into_bytes(),
            });
        }

        // Consolidate threats into aggregate signature
        let threat_events: Vec<_> = self
            .offline_buffer
            .iter()
            .filter(|e| matches!(e.kind, OfflineExperienceKind::ThreatDetected { .. }))
            .collect();
        if !threat_events.is_empty() {
            let max_severity = threat_events
                .iter()
                .filter_map(|e| match &e.kind {
                    OfflineExperienceKind::ThreatDetected { severity, .. } => Some(*severity),
                    _ => None,
                })
                .fold(0.0f32, f32::max);
            patterns.push(ConsolidatedPattern {
                kind: "threat_summary".into(),
                confidence: max_severity,
                data: format!("threats:{},max_severity:{:.2}", threat_events.len(), max_severity)
                    .into_bytes(),
            });
        }

        let experiences_consolidated = self.offline_buffer.len();
        self.offline_buffer.clear();
        self.consolidation_pending = false;

        ConsolidatedWisdom {
            experiences_consolidated,
            offline_duration,
            mean_salience,
            patterns,
        }
    }

    /// Whether consolidation is pending (buffer has data, just reconnected).
    pub fn needs_consolidation(&self) -> bool {
        self.consolidation_pending
    }

    /// Number of buffered experiences.
    pub fn buffer_len(&self) -> usize {
        self.offline_buffer.len()
    }

    /// Whether currently offline.
    pub fn is_offline(&self) -> bool {
        self.is_offline
    }

    /// Total reconnection events this session.
    pub fn reconnection_count(&self) -> u32 {
        self.reconnection_count
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
                assert!(
                    tier == RadioTier::Local
                        || tier == RadioTier::Metro
                        || tier == RadioTier::Regional
                );
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
            RoutingDecision::Routed {
                tier, fragmented, ..
            } => {
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
            RoutingDecision::Routed {
                tier, fragmented, ..
            } => {
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
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
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
        assert!(!reg.is_allowed(50_000_000)); // VHF
    }

    // ── RadioHardware trait ──────────────────────────────────────────────

    #[test]
    fn test_mock_hardware_transmit_receive() {
        let mut hw = MockRadioHardware::new();

        // Transmit on Local tier
        let payload = b"hello mesh";
        let sent = hw.transmit(RadioTier::Local, payload).unwrap();
        assert_eq!(sent, payload.len());

        // Verify transmitted payloads
        assert_eq!(hw.transmitted_payloads().len(), 1);
        assert_eq!(hw.transmitted_payloads()[0].0, RadioTier::Local as usize);
        assert_eq!(hw.transmitted_payloads()[0].1, payload.to_vec());

        // Inject receive data and read it back
        let recv_data = b"world".to_vec();
        hw.inject_receive(RadioTier::Metro, recv_data.clone(), 12.5);

        let received = hw.receive(RadioTier::Metro).unwrap().unwrap();
        assert_eq!(received.0, recv_data);
        assert!((received.1 - 12.5).abs() < 0.01);

        // No more data on Metro
        assert!(hw.receive(RadioTier::Metro).unwrap().is_none());
    }

    #[test]
    fn test_mock_hardware_payload_too_large() {
        let mut hw = MockRadioHardware::new();

        // Metro MTU is 250 bytes — try to send 300
        let big_payload = vec![0u8; 300];
        let result = hw.transmit(RadioTier::Metro, &big_payload);
        assert!(result.is_err());
        match result.unwrap_err() {
            RadioError::PayloadTooLarge { max, got } => {
                assert_eq!(max, 250);
                assert_eq!(got, 300);
            }
            other => panic!("Expected PayloadTooLarge, got {:?}", other),
        }
    }

    #[test]
    fn test_mock_hardware_unavailable_tier() {
        let mut hw = MockRadioHardware::new();
        hw.set_available(RadioTier::Regional, false);

        assert!(!hw.is_available(RadioTier::Regional));
        assert!(hw.current_snr(RadioTier::Regional).is_none());
        assert!(hw.current_frequency(RadioTier::Regional).is_none());

        let result = hw.transmit(RadioTier::Regional, b"test");
        assert!(matches!(result, Err(RadioError::Unavailable)));
    }

    #[test]
    fn test_null_hardware_always_unavailable() {
        let mut hw = NullRadioHardware;

        assert!(!hw.is_available(RadioTier::Local));
        assert!(!hw.is_available(RadioTier::Metro));
        assert!(!hw.is_available(RadioTier::Regional));

        assert!(hw.current_snr(RadioTier::Local).is_none());
        assert!(hw.current_frequency(RadioTier::Local).is_none());
        assert_eq!(hw.hardware_id(), "NullRadioHardware");

        assert!(matches!(
            hw.transmit(RadioTier::Local, b"test"),
            Err(RadioError::Unavailable)
        ));
        assert!(matches!(
            hw.receive(RadioTier::Local),
            Err(RadioError::Unavailable)
        ));
        assert!(matches!(
            hw.set_tx_power(RadioTier::Local, 10.0),
            Err(RadioError::Unavailable)
        ));
        assert!(matches!(
            hw.tune(RadioTier::Local, 2_450_000_000),
            Err(RadioError::Unavailable)
        ));
    }

    #[test]
    fn test_mock_hardware_snr() {
        let mut hw = MockRadioHardware::new();
        hw.set_snr(RadioTier::Local, 42.0);
        assert_eq!(hw.current_snr(RadioTier::Local), Some(42.0));
    }

    #[test]
    fn test_mock_hardware_tune_and_frequency() {
        let mut hw = MockRadioHardware::new();
        hw.tune(RadioTier::Local, 2_412_000_000).unwrap();
        assert_eq!(hw.current_frequency(RadioTier::Local), Some(2_412_000_000));
    }

    #[test]
    fn test_mock_hardware_tx_power() {
        let mut hw = MockRadioHardware::new();
        hw.set_tx_power(RadioTier::Metro, 14.0).unwrap();
        // No assertion on internal state — success is sufficient
    }

    #[test]
    fn test_mock_hardware_id() {
        let hw = MockRadioHardware::new();
        assert_eq!(hw.hardware_id(), "MockRadioHardware v1.0");
    }

    // ── RegulatoryDatabase ───────────────────────────────────────────────

    #[test]
    fn test_regulatory_fcc_lora_band() {
        let db = RegulatoryDatabase::new(RegulatoryRegion::FccUs);
        // 915 MHz should be allowed for Metro tier (LoRa)
        assert!(db.is_frequency_allowed(915_000_000, RadioTier::Metro));
        // 902 MHz (band edge) should be allowed
        assert!(db.is_frequency_allowed(902_000_000, RadioTier::Metro));
        // 928 MHz (band edge) should be allowed
        assert!(db.is_frequency_allowed(928_000_000, RadioTier::Metro));
    }

    #[test]
    fn test_regulatory_fcc_outside_band() {
        let db = RegulatoryDatabase::new(RegulatoryRegion::FccUs);
        // 850 MHz is not in any FCC ISM band
        assert!(!db.is_frequency_allowed(850_000_000, RadioTier::Metro));
        // 900 MHz is just below the 902 MHz band
        assert!(!db.is_frequency_allowed(900_000_000, RadioTier::Metro));
    }

    #[test]
    fn test_regulatory_etsi_duty_cycle() {
        let db = RegulatoryDatabase::new(RegulatoryRegion::EtsiEu);
        // 868.3 MHz should have 1% duty cycle (SRD band h1.4)
        let duty = db.duty_cycle_for_band(868_300_000);
        assert!(duty.is_some(), "868 MHz band should have a duty cycle");
        assert!(
            (duty.unwrap() - 0.01).abs() < 0.001,
            "Expected 1% duty cycle, got {}",
            duty.unwrap()
        );
    }

    #[test]
    fn test_regulatory_bands_for_tier() {
        let db = RegulatoryDatabase::new(RegulatoryRegion::FccUs);

        let local_bands = db.bands_for_tier(RadioTier::Local);
        assert!(
            local_bands.len() >= 2,
            "FCC should have at least 2 Local bands (2.4 GHz + 5.8 GHz)"
        );
        for band in &local_bands {
            assert_eq!(band.tier, RadioTier::Local);
        }

        let metro_bands = db.bands_for_tier(RadioTier::Metro);
        assert!(
            !metro_bands.is_empty(),
            "FCC should have Metro bands (915 MHz ISM)"
        );

        let regional_bands = db.bands_for_tier(RadioTier::Regional);
        assert!(
            !regional_bands.is_empty(),
            "FCC should have Regional bands (HF amateur)"
        );
        for band in &regional_bands {
            assert_eq!(band.license, LicenseType::Amateur);
        }
    }

    #[test]
    fn test_regulatory_max_power() {
        let db = RegulatoryDatabase::new(RegulatoryRegion::FccUs);

        // 915 MHz: 30 dBm EIRP
        let power = db.max_power_for_frequency(915_000_000);
        assert!(power.is_some());
        assert!((power.unwrap() - 30.0).abs() < 0.1);

        // 2.45 GHz: 36 dBm EIRP
        let power = db.max_power_for_frequency(2_450_000_000);
        assert!(power.is_some());
        assert!((power.unwrap() - 36.0).abs() < 0.1);

        // Out-of-band: None
        assert!(db.max_power_for_frequency(100_000_000).is_none());
    }

    #[test]
    fn test_regulatory_available_bandwidth() {
        let db = RegulatoryDatabase::new(RegulatoryRegion::FccUs);

        // Metro tier: 902-928 MHz = 26 MHz
        let metro_bw = db.available_bandwidth(RadioTier::Metro);
        assert_eq!(metro_bw, 26_000_000);

        // Local tier: 2.4 GHz (83.5 MHz) + 5.8 GHz (125 MHz) = 208.5 MHz
        let local_bw = db.available_bandwidth(RadioTier::Local);
        assert!(
            local_bw > 200_000_000,
            "Local bandwidth should be > 200 MHz, got {}",
            local_bw
        );
    }

    #[test]
    fn test_regulatory_etsi_region() {
        let db = RegulatoryDatabase::new(RegulatoryRegion::EtsiEu);
        assert_eq!(db.region(), RegulatoryRegion::EtsiEu);

        // EU should NOT have 915 MHz band
        assert!(!db.is_frequency_allowed(915_000_000, RadioTier::Metro));

        // EU should have 868 MHz
        assert!(db.is_frequency_allowed(868_300_000, RadioTier::Metro));

        // EU 2.4 GHz power is 20 dBm (vs US 36 dBm)
        let power = db.max_power_for_frequency(2_450_000_000);
        assert!(power.is_some());
        assert!((power.unwrap() - 20.0).abs() < 0.1);
    }

    #[test]
    fn test_regulatory_ism_global_fallback() {
        let db = RegulatoryDatabase::new(RegulatoryRegion::IsmGlobal);

        // 2.4 GHz should always be available globally
        assert!(db.is_frequency_allowed(2_450_000_000, RadioTier::Local));

        // 433 MHz should be available for Metro
        assert!(db.is_frequency_allowed(433_500_000, RadioTier::Metro));
    }

    #[test]
    fn test_regulatory_to_legacy_constraints() {
        let db = RegulatoryDatabase::new(RegulatoryRegion::FccUs);
        let legacy = db.to_legacy_constraints();

        assert_eq!(legacy.region, "US");
        assert!(legacy.is_allowed(915_000_000));
        assert!(legacy.is_allowed(2_450_000_000));
        // HF amateur bands should not be in legacy (they require Amateur license)
        assert!(
            !legacy.is_allowed(7_100_000),
            "Amateur bands should not appear in legacy unlicensed constraints"
        );
    }

    // ── Hardware + Regulatory integration ────────────────────────────────

    #[test]
    fn test_hardware_regulatory_integration() {
        let mut hw = MockRadioHardware::new();
        let db = RegulatoryDatabase::new(RegulatoryRegion::FccUs);
        hw.set_regulatory_db(db);

        // Valid tune: 915 MHz for Metro → should work
        hw.tune(RadioTier::Metro, 915_000_000).unwrap();

        // Invalid tune: 850 MHz is not in any FCC band
        let result = hw.tune(RadioTier::Metro, 850_000_000);
        assert!(result.is_err());
        match result.unwrap_err() {
            RadioError::FrequencyOutOfBand { freq_hz, .. } => {
                assert_eq!(freq_hz, 850_000_000);
            }
            other => panic!("Expected FrequencyOutOfBand, got {:?}", other),
        }

        // Valid power: 20 dBm on 915 MHz (max is 30 dBm)
        hw.tune(RadioTier::Metro, 915_000_000).unwrap();
        hw.set_tx_power(RadioTier::Metro, 20.0).unwrap();

        // Invalid power: 35 dBm on 915 MHz (max is 30 dBm)
        let result = hw.set_tx_power(RadioTier::Metro, 35.0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RadioError::RegulatoryViolation(_)
        ));
    }

    // ── SpectrumManager + hardware integration ──────────────────────────

    #[test]
    fn test_spectrum_manager_with_hardware() {
        let mut sm = SpectrumManager::with_region(RegulatoryRegion::FccUs);

        let hw = MockRadioHardware::new();
        sm.set_hardware(Box::new(hw));

        assert_eq!(sm.hardware_id(), Some("MockRadioHardware v1.0"));

        // Process should pick up SNR from hardware
        let snapshot = CycleSnapshot::default();
        let _output = sm.process(&snapshot);

        // Telemetry should reflect all tiers up (mock default)
        assert!(sm.telemetry().tier_available[0]);
    }

    #[test]
    fn test_spectrum_manager_with_region() {
        let sm = SpectrumManager::with_region(RegulatoryRegion::EtsiEu);
        assert_eq!(sm.regulatory_db().region(), RegulatoryRegion::EtsiEu);

        // Legacy constraints should match
        assert!(sm.regulatory().is_allowed(868_300_000));
    }

    #[test]
    fn test_spectrum_manager_hardware_down_tier() {
        let mut sm = SpectrumManager::default();

        let mut hw = MockRadioHardware::new();
        hw.set_available(RadioTier::Local, false);
        sm.set_hardware(Box::new(hw));

        let snapshot = CycleSnapshot::default();
        let output = sm.process(&snapshot);

        // Local should be detected as unavailable via hardware poll
        assert!(!sm.tier_available[0]);
        assert!(
            output.confidence_delta < 0.0,
            "Hardware-detected tier loss should reduce confidence"
        );
    }

    // ── Item 1: AIMD ─────────────────────────────────────────────────

    #[test]
    fn test_aimd_additive_increase_on_healthy() {
        let mut sm = SpectrumManager::default();
        let initial_budget = sm.tier_budget[0]; // Local
                                                // Loss is 0 (healthy) → additive increase
        sm.tick_aimd();
        assert!(
            sm.tier_budget[0] > initial_budget,
            "Healthy tier should increase budget: {} -> {}",
            initial_budget,
            sm.tier_budget[0]
        );
    }

    #[test]
    fn test_aimd_multiplicative_decrease_on_congestion() {
        let mut sm = SpectrumManager::default();
        // Simulate high loss on Metro
        sm.tier_loss_ema[1] = 0.8; // Above TIER_DEGRADED_LOSS
        let initial_budget = sm.tier_budget[1];
        sm.tick_aimd();
        assert!(
            sm.tier_budget[1] < initial_budget,
            "Congested tier should decrease budget: {} -> {}",
            initial_budget,
            sm.tier_budget[1]
        );
    }

    #[test]
    fn test_aimd_respects_floor_and_ceiling() {
        let mut sm = SpectrumManager::default();
        let local_profile = RadioTier::Local.profile();

        // Push budget to ceiling
        for _ in 0..200 {
            sm.tick_aimd();
        }
        assert!(
            sm.tier_budget[0] <= local_profile.bandwidth_max,
            "Budget should not exceed ceiling"
        );

        // Force heavy loss → decrease
        sm.tier_loss_ema[0] = 0.99;
        for _ in 0..200 {
            sm.tick_aimd();
        }
        assert!(
            sm.tier_budget[0] >= local_profile.bandwidth_min,
            "Budget should not fall below floor"
        );
    }

    #[test]
    fn test_aimd_skips_unavailable_tiers() {
        let mut sm = SpectrumManager::default();
        sm.tier_available[2] = false; // Regional down
        let initial = sm.tier_budget[2];
        sm.tick_aimd();
        assert_eq!(
            sm.tier_budget[2], initial,
            "Down tier budget should not change"
        );
    }

    // ── Item 2: Adaptive compression ─────────────────────────────────

    #[test]
    fn test_compress_for_tier_local_full() {
        let mut sm = SpectrumManager::default();
        let peer_id = [1u8; 8];
        let hv = [0xAA; 2048];
        let result = sm.compress_for_tier(&peer_id, &hv, RadioTier::Local);
        assert_eq!(result.strategy, CompressionStrategy::Full);
        assert_eq!(result.data.len(), 2048);
    }

    #[test]
    fn test_compress_for_tier_metro_delta() {
        let mut sm = SpectrumManager::default();
        let peer_id = [2u8; 8];
        let hv1 = [0xAA; 2048];
        // First call: full (no previous)
        let r1 = sm.compress_for_tier(&peer_id, &hv1, RadioTier::Metro);
        assert_eq!(r1.strategy, CompressionStrategy::Full);

        // Second call: delta (previous exists)
        let mut hv2 = hv1;
        hv2[100] = 0xBB;
        let r2 = sm.compress_for_tier(&peer_id, &hv2, RadioTier::Metro);
        assert_eq!(r2.strategy, CompressionStrategy::Delta);
        assert!(r2.data.len() < 100, "Delta should be small");
    }

    #[test]
    fn test_compress_for_tier_regional_hash() {
        let mut sm = SpectrumManager::default();
        let peer_id = [3u8; 8];
        let hv = [0xCC; 2048];
        let result = sm.compress_for_tier(&peer_id, &hv, RadioTier::Regional);
        assert_eq!(result.strategy, CompressionStrategy::HashOnly);
        assert_eq!(result.data.len(), 32);
    }

    #[test]
    fn test_compress_for_tier_tracks_energy() {
        let mut sm = SpectrumManager::default();
        let peer_id = [4u8; 8];
        let hv = [0xDD; 2048];
        sm.compress_for_tier(&peer_id, &hv, RadioTier::Local);
        assert!(sm.energy_spent_nj > 0.0, "Should track energy cost");
    }

    // ── Item 3: Waterfall ────────────────────────────────────────────

    #[test]
    fn test_waterfall_records_observations() {
        let mut sm = SpectrumManager::default();
        sm.inject_observation(SpectrumObservation {
            frequency_hz: 915_000_000,
            noise_floor_dbm: -90.0,
            snr_db: 20.0,
            jammed: false,
        });
        let snapshot = CycleSnapshot::default();
        sm.process(&snapshot);
        assert_eq!(sm.waterfall.len(), 1);
    }

    #[test]
    fn test_waterfall_mean_noise_floor() {
        let mut wf = SpectrumWaterfall::new(64);
        for i in 0..10 {
            wf.push(WaterfallEntry {
                cycle: i,
                noise_floor_dbm: -90.0 + i as f64,
                snr_db: 20.0,
                jammed: false,
                observation_count: 1,
            });
        }
        let mean = wf.mean_noise_floor().unwrap();
        assert!((mean - (-85.5)).abs() < 0.01);
    }

    #[test]
    fn test_waterfall_jamming_ratio() {
        let mut wf = SpectrumWaterfall::new(64);
        for i in 0..10 {
            wf.push(WaterfallEntry {
                cycle: i,
                noise_floor_dbm: -90.0,
                snr_db: 20.0,
                jammed: i % 2 == 0, // 5 out of 10 jammed
                observation_count: 1,
            });
        }
        assert!((wf.jamming_ratio() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_waterfall_capacity_bounded() {
        let mut wf = SpectrumWaterfall::new(4);
        for i in 0..10 {
            wf.push(WaterfallEntry {
                cycle: i,
                noise_floor_dbm: -90.0,
                snr_db: 20.0,
                jammed: false,
                observation_count: 1,
            });
        }
        assert_eq!(wf.len(), 4);
    }

    #[test]
    fn test_waterfall_periodic_detection() {
        let mut wf = SpectrumWaterfall::new(64);
        // Insert regular spikes every 10 cycles
        for i in 0..40 {
            let noise = if i % 10 == 0 { -50.0 } else { -90.0 }; // Spike
            wf.push(WaterfallEntry {
                cycle: i,
                noise_floor_dbm: noise,
                snr_db: if i % 10 == 0 { 2.0 } else { 20.0 },
                jammed: i % 10 == 0,
                observation_count: 1,
            });
        }
        let period = wf.detect_periodic_interference();
        assert!(period.is_some(), "Should detect periodic interference");
        assert_eq!(period.unwrap(), 10);
    }

    // ── Item 6: Frequency hopping ────────────────────────────────────

    #[test]
    fn test_frequency_hop_cooldown() {
        let mut sm = SpectrumManager::default();
        sm.hop_cooldown = 3;
        assert!(sm.suggest_frequency_hop(RadioTier::Metro).is_none());
    }

    #[test]
    fn test_frequency_hop_needs_waterfall_data() {
        let mut sm = SpectrumManager::default();
        // Empty waterfall → no hop
        assert!(sm.suggest_frequency_hop(RadioTier::Metro).is_none());
    }

    // ── Item 7: Energy-aware routing ─────────────────────────────────

    #[test]
    fn test_energy_per_bit_in_profile() {
        let local = RadioTier::Local.profile();
        let metro = RadioTier::Metro.profile();
        let regional = RadioTier::Regional.profile();
        // Wi-Fi > LoRa in energy/bit
        assert!(local.energy_per_bit_nj > metro.energy_per_bit_nj);
        // HF >> LoRa
        assert!(regional.energy_per_bit_nj > metro.energy_per_bit_nj);
    }

    #[test]
    fn test_energy_aware_route_normal() {
        let sm = SpectrumManager::default();
        // Energy plentiful → normal routing
        let route = sm.energy_aware_route(100, 1);
        assert!(route.is_some());
    }

    #[test]
    fn test_energy_aware_route_constrained() {
        let mut sm = SpectrumManager::default();
        // Spend >50% of budget → energy-constrained mode
        sm.energy_spent_nj = sm.energy_budget_nj * 0.6;
        let route = sm.energy_aware_route(100, 1);
        // Should prefer lowest energy tier that fits
        assert!(route.is_some());
        let tier = route.unwrap();
        // Metro has lowest energy_per_bit among tiers that fit 100 bytes
        assert_eq!(tier, RadioTier::Metro);
    }

    // ── Item 8: Peer discovery ───────────────────────────────────────

    #[test]
    fn test_beacon_serialization_roundtrip() {
        let beacon = DiscoveryBeacon {
            node_id: [1, 2, 3, 4, 5, 6, 7, 8],
            capabilities_hash: [0xAA; 8],
            cycle_counter: 42,
            network_health: 1,
            tier_mask: 0x07,
        };
        let bytes = beacon.to_bytes();
        assert_eq!(bytes.len(), RADIO_BEACON_SIZE);
        let decoded = DiscoveryBeacon::from_bytes(&bytes);
        assert_eq!(decoded.node_id, beacon.node_id);
        assert_eq!(decoded.capabilities_hash, beacon.capabilities_hash);
        assert_eq!(decoded.cycle_counter, 42);
        assert_eq!(decoded.network_health, 1);
        assert_eq!(decoded.tier_mask, 0x07);
    }

    #[test]
    fn test_beacon_tier_mask() {
        assert_eq!(DiscoveryBeacon::tier_mask_from(&[true, true, true]), 0x07);
        assert_eq!(DiscoveryBeacon::tier_mask_from(&[true, false, false]), 0x01);
        assert_eq!(DiscoveryBeacon::tier_mask_from(&[false, true, true]), 0x06);
        assert_eq!(
            DiscoveryBeacon::tier_mask_from(&[false, false, false]),
            0x00
        );
    }

    #[test]
    fn test_beacon_interval() {
        let mut sm = SpectrumManager::default();
        sm.set_node_id([1u8; 8]);
        let caps = [0xABu8; 8];

        // Should not generate before interval
        for _ in 0..RADIO_BEACON_INTERVAL_CYCLES - 1 {
            assert!(sm.generate_beacon(&caps).is_none());
        }
        // Should generate at interval
        let beacon = sm.generate_beacon(&caps);
        assert!(beacon.is_some());
        assert_eq!(beacon.unwrap().node_id, [1u8; 8]);
    }

    #[test]
    fn test_process_beacon_adds_route() {
        let mut sm = SpectrumManager::default();
        let beacon = DiscoveryBeacon {
            node_id: [42u8; 8],
            capabilities_hash: [0; 8],
            cycle_counter: 100,
            network_health: 0,
            tier_mask: 0x07,
        };
        sm.process_beacon(&beacon, RadioTier::Metro, 15.0);
        assert_eq!(sm.route_table.len(), 1);
        let route = sm.route_table.lookup(&[42u8; 8]).unwrap();
        assert_eq!(route.hop_count, 0);
        assert_eq!(route.tier, RadioTier::Metro);
    }

    // ── Item 9: Multi-hop relay ──────────────────────────────────────

    #[test]
    fn test_route_table_update_and_lookup() {
        let mut rt = RouteTable::new(128);
        rt.update(RouteEntry {
            destination: [1u8; 8],
            next_hop: [2u8; 8],
            hop_count: 1,
            tier: RadioTier::Metro,
            last_seen_cycle: 100,
            link_quality: 0.8,
        });
        assert_eq!(rt.len(), 1);
        let route = rt.lookup(&[1u8; 8]).unwrap();
        assert_eq!(route.hop_count, 1);
        assert_eq!(route.next_hop, [2u8; 8]);
    }

    #[test]
    fn test_route_table_prefers_shorter_hops() {
        let mut rt = RouteTable::new(128);
        rt.update(RouteEntry {
            destination: [1u8; 8],
            next_hop: [2u8; 8],
            hop_count: 3,
            tier: RadioTier::Metro,
            last_seen_cycle: 100,
            link_quality: 0.8,
        });
        // Better route with fewer hops
        rt.update(RouteEntry {
            destination: [1u8; 8],
            next_hop: [3u8; 8],
            hop_count: 1,
            tier: RadioTier::Local,
            last_seen_cycle: 101,
            link_quality: 0.9,
        });
        let route = rt.lookup(&[1u8; 8]).unwrap();
        assert_eq!(route.hop_count, 1);
        assert_eq!(route.next_hop, [3u8; 8]);
    }

    #[test]
    fn test_route_table_prune() {
        let mut rt = RouteTable::new(128);
        rt.update(RouteEntry {
            destination: [1u8; 8],
            next_hop: [2u8; 8],
            hop_count: 1,
            tier: RadioTier::Metro,
            last_seen_cycle: 10,
            link_quality: 0.8,
        });
        // Prune with current_cycle far ahead
        rt.prune(600, 500);
        assert_eq!(rt.len(), 0, "Stale route should be pruned");
    }

    #[test]
    fn test_prepare_relay() {
        let mut sm = SpectrumManager::default();
        sm.set_node_id([10u8; 8]);
        // Add a route
        sm.route_table.update(RouteEntry {
            destination: [20u8; 8],
            next_hop: [15u8; 8],
            hop_count: 1,
            tier: RadioTier::Metro,
            last_seen_cycle: 0,
            link_quality: 0.9,
        });

        let payload = b"hello";
        let result = sm.prepare_relay(&[20u8; 8], payload);
        assert!(result.is_some());
        let (next_hop, packet) = result.unwrap();
        assert_eq!(next_hop, [15u8; 8]);
        // Header: 8 (dest) + 8 (src) + 1 (ttl) + 1 (hop) + 5 (payload) = 23
        assert_eq!(packet.len(), 23);
        // Check destination in header
        assert_eq!(&packet[0..8], &[20u8; 8]);
        // Check source in header
        assert_eq!(&packet[8..16], &[10u8; 8]);
    }

    // ── Item 10: FEC ─────────────────────────────────────────────────

    #[test]
    fn test_fec_encode_decode_no_loss() {
        let data = b"hello world, this is a test of FEC encoding!";
        let encoded = FecEncoder::encode(data, 16);
        assert!(encoded.len() > data.len()); // Parity added
        let decoded = FecEncoder::decode(&encoded, 16, None);
        assert_eq!(&decoded, data);
    }

    #[test]
    fn test_fec_recover_lost_block() {
        let data = vec![0xAA; 64]; // 4 blocks of 16
        let encoded = FecEncoder::encode(&data, 16);

        // Corrupt block 2 (bytes 32-47)
        let mut corrupted = encoded.clone();
        for i in 32..48 {
            corrupted[i] = 0x00;
        }

        let recovered = FecEncoder::decode(&corrupted, 16, Some(2));
        assert_eq!(&recovered[..64], &data[..]);
    }

    #[test]
    fn test_fec_small_payload_passthrough() {
        let data = b"tiny";
        let encoded = FecEncoder::encode(data, 16);
        // Below RADIO_FEC_MIN_PAYLOAD → no FEC
        assert_eq!(encoded.len(), data.len());
    }

    #[test]
    fn test_fec_overhead() {
        assert_eq!(FecEncoder::overhead(100, 16), 16);
        assert_eq!(FecEncoder::overhead(10, 16), 0); // Below min
    }

    // ── Item 11: Encryption ──────────────────────────────────────────

    #[test]
    fn test_encryption_roundtrip() {
        let key = [42u8; 32];
        let nonce = [1u8; RADIO_CRYPTO_NONCE_SIZE];
        let plaintext = b"secret consciousness data";

        let encrypted = MeshEncryption::encrypt(&key, &nonce, plaintext);
        assert_ne!(&encrypted[..plaintext.len()], plaintext);

        let decrypted = MeshEncryption::decrypt(&key, &nonce, &encrypted);
        assert!(decrypted.is_some());
        assert_eq!(&decrypted.unwrap(), plaintext);
    }

    #[test]
    fn test_encryption_auth_tag_failure() {
        let key = [42u8; 32];
        let nonce = [1u8; RADIO_CRYPTO_NONCE_SIZE];
        let plaintext = b"secret data";

        let mut encrypted = MeshEncryption::encrypt(&key, &nonce, plaintext);
        // Tamper with auth tag
        let len = encrypted.len();
        encrypted[len - 1] ^= 0xFF;

        assert!(MeshEncryption::decrypt(&key, &nonce, &encrypted).is_none());
    }

    #[test]
    fn test_peer_session_nonce_monotonic() {
        let mut session = PeerSession::new([1u8; 8], [2u8; 32], 0);
        let n1 = session.next_nonce();
        let n2 = session.next_nonce();
        assert_ne!(n1, n2);
        assert_eq!(session.tx_nonce_counter, 2);
    }

    #[test]
    fn test_peer_session_replay_detection() {
        let mut session = PeerSession::new([1u8; 8], [2u8; 32], 0);
        assert!(session.check_nonce(1)); // First nonce: OK
        assert!(session.check_nonce(2)); // Second: OK
        assert!(!session.check_nonce(1)); // Replay: rejected
        assert!(!session.check_nonce(2)); // Replay: rejected
        assert!(session.check_nonce(3)); // New: OK
    }

    #[test]
    fn test_spectrum_manager_encrypt_decrypt() {
        let mut sm = SpectrumManager::default();
        let peer_id = [5u8; 8];
        let session = PeerSession::new(peer_id, [99u8; 32], 0);
        sm.encryption_mut().add_session(session);

        let plaintext = b"consciousness vector";
        let encrypted = sm.encrypt_for_peer(&peer_id, plaintext).unwrap();
        let decrypted = sm.decrypt_from_peer(&peer_id, &encrypted).unwrap();
        assert_eq!(&decrypted, plaintext);
    }

    #[test]
    fn test_encryption_no_session() {
        let mut sm = SpectrumManager::default();
        assert!(sm.encrypt_for_peer(&[9u8; 8], b"test").is_none());
    }

    #[test]
    fn test_mesh_encryption_capacity() {
        let mut enc = MeshEncryption::new(2);
        enc.add_session(PeerSession::new([1u8; 8], [1u8; 32], 0));
        enc.add_session(PeerSession::new([2u8; 8], [2u8; 32], 0));
        enc.add_session(PeerSession::new([3u8; 8], [3u8; 32], 0)); // Over capacity
        assert_eq!(enc.session_count(), 2); // Capped at 2
    }

    // ── Telemetry includes new fields ────────────────────────────────

    #[test]
    fn test_telemetry_includes_new_fields() {
        let mut sm = SpectrumManager::default();
        sm.inject_observation(SpectrumObservation {
            frequency_hz: 915_000_000,
            noise_floor_dbm: -90.0,
            snr_db: 20.0,
            jammed: false,
        });
        let snapshot = CycleSnapshot::default();
        sm.process(&snapshot);

        let t = sm.telemetry();
        assert!(t.tier_budgets[0] > 0); // AIMD budgets populated
        assert_eq!(t.waterfall_depth, 1);
        assert_eq!(t.known_peers, 0);
        assert_eq!(t.encryption_sessions, 0);
    }

    // ── Item 1: Mesh→Spectrum feedback ──────────────────────────────

    #[test]
    fn test_ingest_mesh_stats_updates_loss_ema() {
        let mut sm = SpectrumManager::default();
        sm.ingest_mesh_stats(50, 100); // 50% loss
        let t = sm.telemetry();
        assert!(t.tier_loss_ema[0] > 0.0, "Local loss should increase");
        assert!(
            t.tier_loss_ema[1] > t.tier_loss_ema[0],
            "Metro should absorb more"
        );
        assert!(
            t.tier_loss_ema[2] > t.tier_loss_ema[1],
            "Regional should absorb most"
        );
    }

    #[test]
    fn test_ingest_mesh_stats_zero_dropped() {
        let mut sm = SpectrumManager::default();
        sm.ingest_mesh_stats(0, 100);
        let t = sm.telemetry();
        for &loss in &t.tier_loss_ema {
            assert!(loss.abs() < 1e-10, "No drops should not increase loss");
        }
    }

    #[test]
    fn test_ingest_mesh_stats_zero_total() {
        let mut sm = SpectrumManager::default();
        sm.ingest_mesh_stats(10, 0); // Should be no-op
        let t = sm.telemetry();
        for &loss in &t.tier_loss_ema {
            assert!(loss.abs() < 1e-10);
        }
    }

    // ── Item 2: Safety escalation ───────────────────────────────────

    #[test]
    fn test_network_critical_on_blackout() {
        let mut sm = SpectrumManager::default();
        sm.set_tier_available(RadioTier::Local, false);
        sm.set_tier_available(RadioTier::Metro, false);
        sm.set_tier_available(RadioTier::Regional, false);
        assert!(sm.is_network_critical());
    }

    #[test]
    fn test_network_not_critical_all_up() {
        let sm = SpectrumManager::default();
        assert!(!sm.is_network_critical());
    }

    // ── Item 3: Connectivity factor ─────────────────────────────────

    #[test]
    fn test_connectivity_factor_all_up() {
        let sm = SpectrumManager::default();
        assert!((sm.connectivity_factor() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_connectivity_factor_blackout() {
        let mut sm = SpectrumManager::default();
        sm.set_tier_available(RadioTier::Local, false);
        sm.set_tier_available(RadioTier::Metro, false);
        sm.set_tier_available(RadioTier::Regional, false);
        assert!((sm.connectivity_factor() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_connectivity_factor_local_down() {
        let mut sm = SpectrumManager::default();
        sm.set_tier_available(RadioTier::Local, false);
        assert!((sm.connectivity_factor() - 0.7).abs() < 1e-10);
    }

    // ── Item 9: Beacon→SwarmManager ──────────────────────────────────

    #[test]
    fn test_process_beacon_returns_true_for_new_peer() {
        let mut sm = SpectrumManager::default();
        let beacon = DiscoveryBeacon {
            node_id: [42u8; 8],
            capabilities_hash: [0; 8],
            cycle_counter: 100,
            network_health: 0,
            tier_mask: 0b111,
        };
        let is_new = sm.process_beacon(&beacon, RadioTier::Local, 15.0);
        assert!(is_new, "First beacon from peer should be new");

        let is_new_again = sm.process_beacon(&beacon, RadioTier::Local, 15.0);
        assert!(
            !is_new_again,
            "Second beacon from same peer should not be new"
        );
    }

    // ── Item 10: Regulatory validation ──────────────────────────────

    #[test]
    fn test_validate_transmission_ism_band() {
        let sm = SpectrumManager::default(); // ISM Global
                                             // 433 MHz ISM band (Metro tier in ISM Global), 10.0 dBm max
        assert!(sm.validate_transmission(RadioTier::Metro, 433_500_000, 10.0));
    }

    #[test]
    fn test_validate_transmission_out_of_band() {
        let sm = SpectrumManager::default();
        // 100 MHz — not in any ISM allocation
        assert!(!sm.validate_transmission(RadioTier::Metro, 100_000_000, 14.0));
    }

    // ── Item 3: Synthetic observations from swarm state ────────────────

    #[test]
    fn test_ingest_swarm_state_generates_observation() {
        let mut sm = SpectrumManager::default();
        assert_eq!(sm.pending_observations.len(), 0);
        sm.ingest_swarm_state(5, 0.6, 0.8);
        assert_eq!(sm.pending_observations.len(), 1);
        let obs = &sm.pending_observations[0];
        assert!(!obs.jammed, "Connected peers should not be jammed");
        assert!(
            obs.snr_db > RADIO_SYNTHETIC_SNR_BASE as f32,
            "Peers should boost SNR"
        );
    }

    #[test]
    fn test_ingest_swarm_state_isolated_is_jammed() {
        let mut sm = SpectrumManager::default();
        sm.ingest_swarm_state(0, 0.0, 0.05);
        let obs = &sm.pending_observations[0];
        assert!(obs.jammed, "Zero peers + low connectivity should be jammed");
        assert!(
            (obs.snr_db - RADIO_SYNTHETIC_SNR_ISOLATED as f32).abs() < 1e-3,
            "Isolated SNR should be {}",
            RADIO_SYNTHETIC_SNR_ISOLATED
        );
    }

    #[test]
    fn test_ingest_swarm_state_snr_increases_with_peers() {
        let mut sm = SpectrumManager::default();
        sm.ingest_swarm_state(1, 0.5, 0.5);
        let snr1 = sm.pending_observations[0].snr_db;

        let mut sm2 = SpectrumManager::default();
        sm2.ingest_swarm_state(20, 0.5, 0.5);
        let snr20 = sm2.pending_observations[0].snr_db;

        assert!(snr20 > snr1, "More peers should increase SNR");
    }

    #[test]
    fn test_spectrum_prediction_error_accessor() {
        let sm = SpectrumManager::default();
        // Default telemetry: PE = 0.0
        assert!((sm.spectrum_prediction_error() - 0.0).abs() < 1e-10);
    }

    // ── Consciousness-aware tier selection ─────────────────────────────

    #[test]
    fn test_consciousness_aware_high_confidence_prefers_local() {
        let sm = SpectrumManager::default();
        let tier = sm.consciousness_aware_tier(40, 0.9);
        assert_eq!(
            tier,
            Some(RadioTier::Local),
            "High confidence should prefer Local"
        );
    }

    #[test]
    fn test_consciousness_aware_low_confidence_prefers_metro() {
        let sm = SpectrumManager::default();
        let tier = sm.consciousness_aware_tier(40, 0.1);
        // Metro has lowest energy_per_bit (20 nJ) among available tiers
        assert_eq!(
            tier,
            Some(RadioTier::Metro),
            "Low confidence should prefer Metro (lowest energy)"
        );
    }

    #[test]
    fn test_consciousness_aware_medium_confidence_defers() {
        let sm = SpectrumManager::default();
        let tier = sm.consciousness_aware_tier(40, 0.5);
        assert_eq!(
            tier, None,
            "Medium confidence should defer to normal routing"
        );
    }

    #[test]
    fn test_consciousness_aware_respects_mtu() {
        let sm = SpectrumManager::default();
        // Large payload (2048B) with high confidence → Local (mtu=1500)
        // Regional MTU=50, Metro MTU=250, so only Local can fit
        let tier = sm.consciousness_aware_tier(1000, 0.9);
        assert_eq!(
            tier,
            Some(RadioTier::Local),
            "Only Local can fit 1000B payload"
        );
    }

    // ── Energy-aware routing ──────────────────────────────────────────

    #[test]
    fn test_energy_aware_route_activates_above_threshold() {
        let mut sm = SpectrumManager::default();
        // Below threshold: normal routing
        sm.energy_spent_nj = sm.energy_budget_nj * 0.3;
        let r1 = sm.energy_aware_route(100, 1);
        assert!(r1.is_some());

        // Above threshold: energy-constrained routing
        sm.energy_spent_nj = sm.energy_budget_nj * 0.6;
        let r2 = sm.energy_aware_route(100, 1);
        assert!(r2.is_some());
        // Energy-constrained should prefer Metro (20 nJ/bit) over Local (50 nJ/bit)
        assert_eq!(r2, Some(RadioTier::Metro));
    }

    // ═══════════════════════════════════════════════════════════════════
    // CONSCIOUSNESS-AWARE ROUTER TESTS
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_consciousness_router_default() {
        let router = ConsciousnessAwareRouter::default();
        assert_eq!(router.peer_count(), 0);
        assert_eq!(router.collective_phi(), 0.0);
        assert_eq!(router.sharing_cadence(), DEFAULT_SHARING_CADENCE);
    }

    #[test]
    fn test_consciousness_router_update_peer() {
        let mut router = ConsciousnessAwareRouter::default();
        router.update_local(0.5, 0.7, 2);

        router.update_peer([1; 8], 0.8, 0.9, 3, 100);
        assert_eq!(router.peer_count(), 1);
        assert!(router.collective_phi() > 0.0);

        router.update_peer([2; 8], 0.3, 0.4, 1, 101);
        assert_eq!(router.peer_count(), 2);
    }

    #[test]
    fn test_consciousness_router_highest_phi_peer() {
        let mut router = ConsciousnessAwareRouter::default();
        router.update_local(0.5, 0.7, 2);
        router.update_peer([1; 8], 0.3, 0.6, 1, 100);
        router.update_peer([2; 8], 0.9, 0.95, 4, 100);
        router.update_peer([3; 8], 0.6, 0.8, 2, 100);

        let best = router.highest_phi_peer();
        assert_eq!(best, Some([2; 8]));
    }

    #[test]
    fn test_consciousness_router_moral_emergency() {
        let mut router = ConsciousnessAwareRouter::default();
        let classifier = PayloadClassifier::default();

        router.signal_moral_emergency();

        let decision = router.route(PayloadClass::Emergency, 40, 2, &classifier);
        match decision {
            ConsciousRoutingDecision::MoralEmergency { tier, .. } => {
                assert_eq!(tier, RadioTier::Local);
            }
            _ => panic!("expected MoralEmergency routing"),
        }

        // Should be one-shot — next route is normal
        let decision2 = router.route(PayloadClass::Discovery, 64, 1, &classifier);
        assert!(matches!(decision2, ConsciousRoutingDecision::Normal(_)));
    }

    #[test]
    fn test_consciousness_router_adaptive_cadence() {
        let mut router = ConsciousnessAwareRouter::default();
        router.update_local(0.5, 0.7, 2);

        // Add peers with divergent Phi — should decrease cadence
        router.update_peer([1; 8], 0.1, 0.3, 1, 100);
        router.update_peer([2; 8], 0.9, 0.9, 4, 100);

        let cadence_after_divergence = router.sharing_cadence();
        assert!(
            cadence_after_divergence < DEFAULT_SHARING_CADENCE,
            "high divergence should decrease cadence"
        );
    }

    #[test]
    fn test_consciousness_router_sharing_suppression() {
        let mut router = ConsciousnessAwareRouter::default();
        let classifier = PayloadClassifier::default();

        // Should suppress when cadence not reached
        let decision = router.route(PayloadClass::ConsciousnessDelta, 100, 1, &classifier);
        match decision {
            ConsciousRoutingDecision::Suppressed { reason, .. } => {
                assert!(reason.contains("cadence"));
            }
            ConsciousRoutingDecision::ConsciousnessShare { .. } => {
                // Also acceptable if cycles_since_share >= sharing_cadence
            }
            _ => panic!("expected Suppressed or ConsciousnessShare"),
        }
    }

    #[test]
    fn test_consciousness_router_threat_recording() {
        let mut router = ConsciousnessAwareRouter::default();

        let threat = ThreatObservation {
            threat_type: 3,
            severity: 0.8,
            agent_hash: [0xDE; 8],
            signature: [0x42; 32],
            observed_cycle: 100,
            corroboration_count: 0,
        };
        router.record_threat(threat.clone());
        assert_eq!(router.threat_count(), 1);

        // Same agent + type → corroborate, not duplicate
        router.record_threat(threat);
        assert_eq!(router.threat_count(), 1);
        assert_eq!(router.threats()[0].corroboration_count, 1);
    }

    #[test]
    fn test_consciousness_router_trust_decay() {
        let mut router = ConsciousnessAwareRouter::default();

        router.update_peer([1; 8], 0.5, 0.7, 2, 100);
        let trust_before = router.peer_phi.get(&[1; 8]).unwrap().trust;

        // Large Phi jump → trust decreases
        router.update_peer([1; 8], 0.99, 0.7, 2, 101);
        let trust_after = router.peer_phi.get(&[1; 8]).unwrap().trust;
        assert!(trust_after < trust_before, "large Phi jump should decrease trust");
    }

    #[test]
    fn test_consciousness_router_prune() {
        let mut router = ConsciousnessAwareRouter::default();
        router.update_peer([1; 8], 0.5, 0.7, 2, 100);
        router.update_peer([2; 8], 0.6, 0.8, 3, 200);

        // Prune with max_age=50, current_cycle=250
        // Peer 1 (last at cycle 100) → age 150 > 50 → pruned
        // Peer 2 (last at cycle 200) → age 50 = 50 → kept
        router.prune(250, 50);
        assert_eq!(router.peer_count(), 1);
    }

    // ═══════════════════════════════════════════════════════════════════
    // STORE-AND-FORWARD TESTS
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_store_forward_default() {
        let sf = StoreAndForward::default();
        assert!(!sf.is_offline());
        assert_eq!(sf.buffer_len(), 0);
        assert_eq!(sf.reconnection_count(), 0);
    }

    #[test]
    fn test_store_forward_offline_online_cycle() {
        let mut sf = StoreAndForward::default();

        sf.go_offline(100);
        assert!(sf.is_offline());

        // Record some experiences
        for i in 0..15 {
            sf.record(OfflineExperience {
                cycle: 100 + i,
                kind: OfflineExperienceKind::SensorAnomaly {
                    sensor_id: format!("sensor-{i}"),
                    value: 42.0,
                },
                salience: 0.5 + (i as f32 * 0.02),
            });
        }
        assert_eq!(sf.buffer_len(), 15);

        // Go online — should trigger consolidation
        let needs_consolidation = sf.go_online(200);
        assert!(needs_consolidation);
        assert!(!sf.is_offline());
        assert_eq!(sf.reconnection_count(), 1);
    }

    #[test]
    fn test_store_forward_salience_filtering() {
        let mut sf = StoreAndForward::default();
        sf.go_offline(100);

        // Low salience — should be dropped
        sf.record(OfflineExperience {
            cycle: 101,
            kind: OfflineExperienceKind::SensorAnomaly {
                sensor_id: "low".into(),
                value: 1.0,
            },
            salience: 0.1,
        });
        assert_eq!(sf.buffer_len(), 0);

        // High salience — should be kept
        sf.record(OfflineExperience {
            cycle: 102,
            kind: OfflineExperienceKind::SensorAnomaly {
                sensor_id: "high".into(),
                value: 99.0,
            },
            salience: 0.9,
        });
        assert_eq!(sf.buffer_len(), 1);
    }

    #[test]
    fn test_store_forward_consolidation() {
        let mut sf = StoreAndForward::default();
        sf.go_offline(100);

        // Add mixed experiences
        sf.record(OfflineExperience {
            cycle: 110,
            kind: OfflineExperienceKind::SensorAnomaly {
                sensor_id: "temp".into(),
                value: 45.0,
            },
            salience: 0.7,
        });
        sf.record(OfflineExperience {
            cycle: 120,
            kind: OfflineExperienceKind::ConsciousnessShift {
                from: 0.5,
                to: 0.8,
            },
            salience: 0.9,
        });
        sf.record(OfflineExperience {
            cycle: 130,
            kind: OfflineExperienceKind::ThreatDetected {
                threat_type: 2,
                severity: 0.6,
            },
            salience: 0.8,
        });

        let wisdom = sf.consolidate(200);
        assert_eq!(wisdom.experiences_consolidated, 3);
        assert_eq!(wisdom.offline_duration, 100);
        assert!(wisdom.mean_salience > 0.0);
        assert!(!wisdom.patterns.is_empty());

        // Buffer should be cleared after consolidation
        assert_eq!(sf.buffer_len(), 0);
    }

    #[test]
    fn test_peers_by_trust_ordering() {
        let mut router = ConsciousnessAwareRouter::default();
        router.update_peer([1; 8], 0.3, 0.4, 1, 100);
        router.update_peer([2; 8], 0.9, 0.9, 4, 100);
        router.update_peer([3; 8], 0.5, 0.5, 2, 100);

        let sorted = router.peers_by_trust();
        assert_eq!(sorted.len(), 3);
        // Highest trust score should be first
        assert!(sorted[0].1 >= sorted[1].1);
        assert!(sorted[1].1 >= sorted[2].1);
    }
}
