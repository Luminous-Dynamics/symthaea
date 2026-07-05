// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Radio tier abstractions, spectrum manager, and related types.

use super::super::super::subsystem_trait::{
    CognitiveSubsystem, CycleSnapshot, SubsystemOutput, output_flags,
};
use super::hardware::RegulatoryDatabase;
use super::transport::{
    CompressedDelta, DiscoveryBeacon, MeshEncryption, PeerSession, RouteEntry, RouteTable,
    TierCompressedPayload,
};
use crate::domain::{DomainProfile, DomainTransportClass};
use std::collections::{HashMap, VecDeque};

// Re-export named constants from thresholds.rs for local use.
use super::super::super::thresholds::{
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

use super::hardware::{RadioHardware, RegulatoryRegion};

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
    /// Satellite / spacecraft — extreme latency, high reliability with FEC.
    /// S-band (2 GHz), X-band (8 GHz), UHF (400 MHz), or optical laser link.
    /// Use for: LEO CubeSat relay, interplanetary store-and-forward,
    /// consciousness sharing across orbital distances.
    ///
    /// Latency: 1s (LEO) to 53 min (Jupiter). Bandwidth: 500 bps (deep space)
    /// to 2 Mbps (LEO relay). Requires CCSDS space packet framing, strong FEC
    /// (LDPC/turbo), and Doppler pre-compensation for non-GEO orbits.
    ///
    /// Science: Cerf & Kahn (1974) → Burleigh et al. (2003) DTN architecture.
    /// The store-and-forward dream consolidation is not an optimization for
    /// interplanetary — it's a requirement. You cannot have real-time consciousness
    /// sharing at light-speed delay. You share wisdom, not state.
    Interplanetary,
}

impl RadioTier {
    /// Terrestrial tiers in descending bandwidth order.
    pub const ALL: [RadioTier; 3] = [RadioTier::Local, RadioTier::Metro, RadioTier::Regional];

    /// All tiers including interplanetary.
    pub const ALL_WITH_SPACE: [RadioTier; 4] = [
        RadioTier::Local,
        RadioTier::Metro,
        RadioTier::Regional,
        RadioTier::Interplanetary,
    ];
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
            RadioTier::Interplanetary => TierProfile {
                mtu: 1024,                 // CCSDS max data field 65,535; practical 1 KB
                bandwidth_budget: 500,     // 500 B per 10s (deep space ~400 bps)
                bandwidth_min: 50,         // 50 B floor (critical telemetry)
                bandwidth_max: 200_000,    // 200 KB ceiling (LEO high-bandwidth pass)
                additive_increase: 50,     // +50 B per healthy window (conservative)
                decrease_factor: 0.25,     // Gentle decrease (passes are precious)
                duty_cycle: 1.0,           // No regulatory duty cycle in space
                latency_ms: 2_000,         // 2s LEO; actual varies 1s–3,180,000ms (Jupiter)
                reliability: 0.999,        // Strong FEC (LDPC/turbo) makes this high
                energy_per_bit_nj: 1000.0, // High power (5-400W TX), amortized
            },
        }
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
    available_tiers: [bool; 4], // [Local, Metro, Regional, Interplanetary]
    /// Domain policy that constrains valid routing choices.
    domain: DomainProfile,
}

impl Default for PayloadClassifier {
    fn default() -> Self {
        Self {
            available_tiers: [true, true, true, false],
            domain: DomainProfile::default(),
        }
    }
}

impl PayloadClassifier {
    pub(crate) fn tier_transport_class(tier: RadioTier) -> DomainTransportClass {
        match tier {
            RadioTier::Local => DomainTransportClass::LocalMesh,
            RadioTier::Metro => DomainTransportClass::MetroRelay,
            RadioTier::Regional => DomainTransportClass::RegionalRelay,
            RadioTier::Interplanetary => DomainTransportClass::InterplanetaryRelay,
        }
    }

    fn preferred_tiers_for_domain(&self) -> Vec<RadioTier> {
        self.domain
            .transport
            .priority_order()
            .into_iter()
            .map(|transport| match transport {
                DomainTransportClass::LocalMesh => RadioTier::Local,
                DomainTransportClass::MetroRelay => RadioTier::Metro,
                DomainTransportClass::RegionalRelay => RadioTier::Regional,
                DomainTransportClass::InterplanetaryRelay => RadioTier::Interplanetary,
            })
            .collect()
    }

    /// Update tier availability (called when radio hardware status changes).
    pub fn set_tier_available(&mut self, tier: RadioTier, available: bool) {
        self.available_tiers[tier as usize] = available;
    }

    /// Set the active domain policy for routing.
    pub fn set_domain_profile(&mut self, domain: DomainProfile) {
        self.domain = domain;
    }

    /// Get the current domain policy.
    pub fn domain_profile(&self) -> &DomainProfile {
        &self.domain
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
        let preferred_tiers = self.preferred_tiers_for_domain();

        // Try the domain's preferred tiers first, but only those currently
        // available and capable of meeting the nominal bandwidth requirement.
        let candidates: Vec<RadioTier> = preferred_tiers
            .iter()
            .copied()
            .filter(|&t| self.available_tiers[t as usize])
            .filter(|&t| {
                self.domain
                    .supports_transport(Self::tier_transport_class(t))
            })
            .filter(|&t| (t as usize) <= (min_tier as usize))
            .collect();

        if candidates.is_empty() {
            // No tier meets the nominal bandwidth preference.
            // In delay-tolerant domains, fall back to any allowed tier and
            // permit fragmentation/store-and-forward.
            for tier in preferred_tiers {
                if !self.available_tiers[tier as usize] {
                    continue;
                }
                let profile = tier.profile();
                let fragmented = payload_size > profile.mtu;

                if !fragmented || self.domain.transport.store_and_forward_required {
                    return Some(RoutingDecision::Routed {
                        tier,
                        fragmented,
                        estimated_fragments: if fragmented {
                            (payload_size + profile.mtu - 1) / profile.mtu
                        } else {
                            1
                        },
                    });
                }
            }
            return Some(RoutingDecision::Blocked {
                reason: "domain transport unavailable",
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
pub(super) struct WaterfallEntry {
    /// Cycle number when observed.
    pub(super) cycle: u64,
    /// Mean noise floor across all observations that cycle (dBm).
    pub(super) noise_floor_dbm: f64,
    /// Mean SNR across all observations that cycle (dB).
    pub(super) snr_db: f64,
    /// Whether jamming was detected this cycle.
    pub(super) jammed: bool,
    /// Number of raw observations aggregated.
    pub(super) observation_count: u32,
}

/// Ring buffer of spectrum observations for pattern detection.
///
/// Maintains a fixed-capacity window of aggregated per-cycle spectrum state.
/// Enables periodic interference detection and noise floor trend analysis.
///
/// Basis: Haykin (2005) — cognitive radio spectrum sensing requires
/// temporal context for reliable detection.
pub(super) struct SpectrumWaterfall {
    /// Ring buffer of entries (oldest at front).
    pub(super) entries: VecDeque<WaterfallEntry>,
    /// Maximum entries to keep.
    capacity: usize,
}

impl SpectrumWaterfall {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Record a cycle's aggregated spectrum state.
    pub(super) fn push(&mut self, entry: WaterfallEntry) {
        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    /// Mean noise floor across all entries (dBm).
    pub(super) fn mean_noise_floor(&self) -> Option<f64> {
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
    pub(super) fn jamming_ratio(&self) -> f64 {
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
    pub(super) fn detect_periodic_interference(&self) -> Option<u32> {
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
    pub(super) fn len(&self) -> usize {
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
            RadioTier::Regional | RadioTier::Interplanetary => CompressionStrategy::HashOnly,
        }
    }
}

// ── Role seeds for SpectrumObservation HDC encoding (perception_hv) ─────
// Four stable u64s identify each observation dimension. Role XOR Value
// gives a bound pair; bundling four pairs forms the observation HV.
const SPECTRUM_ROLE_FREQ: u64 = 0xF0E0_5EE0_0C0D_E001;
const SPECTRUM_ROLE_NOISE: u64 = 0xF0E0_5EE0_0C0D_E002;
const SPECTRUM_ROLE_SNR: u64 = 0xF0E0_5EE0_0C0D_E003;
const SPECTRUM_ROLE_JAMMED: u64 = 0xF0E0_5EE0_0C0D_E004;

/// Encode one spectrum observation as a BinaryHV via role-filler binding.
///
/// Quantization:
/// - Frequency → 1 MHz buckets (rounded)
/// - Noise floor → 2 dBm buckets
/// - SNR → 2 dB buckets
/// - Jammed → binary flag
///
/// Identical buckets produce identical HVs; different buckets produce
/// HDC-orthogonal HVs. This is a discrete LSH, not a continuous-similarity
/// embedding — sufficient for the cognitive loop to distinguish "clear RF
/// scene" from "jammed 2.4 GHz neighborhood".
fn encode_observation(obs: &SpectrumObservation) -> symthaea_core::hdc::BinaryHV {
    use symthaea_core::hdc::BinaryHV;
    let freq_bucket = obs.frequency_hz / 1_000_000;
    let noise_bucket = ((obs.noise_floor_dbm / 2.0) as i64) as u64;
    let snr_bucket = ((obs.snr_db / 2.0) as i64) as u64;
    let jammed_bit = obs.jammed as u64;

    let freq_pair = BinaryHV::random(SPECTRUM_ROLE_FREQ).bind(&BinaryHV::random(freq_bucket));
    let noise_pair = BinaryHV::random(SPECTRUM_ROLE_NOISE).bind(&BinaryHV::random(noise_bucket));
    let snr_pair = BinaryHV::random(SPECTRUM_ROLE_SNR).bind(&BinaryHV::random(snr_bucket));
    let jammed_pair = BinaryHV::random(SPECTRUM_ROLE_JAMMED).bind(&BinaryHV::random(jammed_bit));

    BinaryHV::bundle(&[freq_pair, noise_pair, snr_pair, jammed_pair])
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
    pub(super) tier_available: [bool; 3],
    /// Per-tier packet loss EMA (0.0–1.0, lower is better).
    tier_loss_ema: [f64; 3],
    /// Per-tier current AIMD budget (bytes per window).
    tier_budget: [u64; 3],

    // ── Classifier ───────────────────────────────────────────────────────
    classifier: PayloadClassifier,

    // ── Spectrum state ───────────────────────────────────────────────────
    /// Pending spectrum observations from SDR (drained each process cycle).
    pub(super) pending_observations: Vec<SpectrumObservation>,
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
    pub(super) degradation_streak: u32,

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
    pub(super) waterfall: SpectrumWaterfall,

    // ── Frequency hopping ────────────────────────────────────────────
    /// Cycles since last frequency hop (cooldown).
    pub(super) hop_cooldown: u32,

    // ── Peer discovery ───────────────────────────────────────────────
    /// This node's ID (first 8 bytes).
    node_id: [u8; 8],
    /// Cycles since last beacon broadcast.
    beacon_counter: u32,

    // ── Multi-hop routing ────────────────────────────────────────────
    /// Mesh route table for multi-hop relay.
    pub(super) route_table: RouteTable,
    /// Current cycle counter (for route expiry).
    current_cycle: u64,

    // ── Encryption ───────────────────────────────────────────────────
    /// Per-peer encryption session manager.
    encryption: MeshEncryption,

    // ── Energy tracking ──────────────────────────────────────────────
    /// Cumulative energy spent this cycle (nJ).
    pub(super) energy_spent_nj: f64,
    /// Energy budget per cycle (nJ).
    pub(super) energy_budget_nj: f64,
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

impl SpectrumManager {
    /// Create with a specific domain profile.
    pub fn with_domain_profile(domain: DomainProfile) -> Self {
        let mut manager = Self::default();
        manager.set_domain_profile(domain);
        manager
    }

    fn tier_allowed_by_domain(&self, tier: RadioTier) -> bool {
        self.classifier
            .domain_profile()
            .supports_transport(PayloadClassifier::tier_transport_class(tier))
    }

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
        if idx >= self.tier_loss_ema.len() {
            return;
        }
        self.tier_loss_ema[idx] =
            self.tier_loss_ema[idx] * (1.0 - TIER_LOSS_EMA_ALPHA) + TIER_LOSS_EMA_ALPHA;
    }

    /// Report a successful packet delivery on a specific tier.
    pub fn report_success(&mut self, tier: RadioTier) {
        let idx = tier as usize;
        if idx >= self.tier_loss_ema.len() {
            return;
        }
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

    /// Encode the currently-pending observations as a perception HV.
    ///
    /// Non-destructive — the tier's own `process_observations()` still drains
    /// them on its schedule. Perception consumes in parallel each cycle and
    /// simply reflects whatever the SDR layer has most recently reported.
    ///
    /// Returns `None` when the observation buffer is empty so the caller can
    /// skip the HDC bundle entirely.
    ///
    /// Encoding: role-filler binding — each observation produces four bound
    /// pairs (freq, noise, snr, jammed-flag), bundled into a per-observation
    /// HV, then all observation HVs are bundled together.
    pub fn perception_hv(&self) -> Option<symthaea_core::hdc::BinaryHV> {
        if self.pending_observations.is_empty() {
            return None;
        }
        let per_obs: Vec<symthaea_core::hdc::BinaryHV> = self
            .pending_observations
            .iter()
            .map(encode_observation)
            .collect();
        Some(symthaea_core::hdc::BinaryHV::bundle(&per_obs))
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
        if idx >= self.tier_available.len() {
            self.classifier.set_tier_available(tier, available);
            return;
        }
        self.tier_available[idx] = available;
        self.classifier.set_tier_available(tier, available);
        self.update_network_health();
    }

    /// Apply a domain profile to downstream routing decisions.
    pub fn set_domain_profile(&mut self, domain: DomainProfile) {
        self.classifier.set_domain_profile(domain);
    }

    /// Get the current domain profile used by the classifier.
    pub fn domain_profile(&self) -> &DomainProfile {
        self.classifier.domain_profile()
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
    pub(super) fn compress_delta(
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
            .filter(|&(_, &avail)| avail)
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
            .find(|&tier| self.tier_available[tier as usize] && self.tier_allowed_by_domain(tier))
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
                if self.tier_available[tier as usize]
                    && self.tier_allowed_by_domain(tier)
                    && payload_size <= tier.profile().mtu
                {
                    return Some(tier);
                }
            }
        } else if confidence < RADIO_CONSCIOUSNESS_LOW_CONFIDENCE {
            // Low confidence: prefer most energy-efficient available tier
            let mut candidates: Vec<(RadioTier, f64)> = RadioTier::ALL
                .iter()
                .copied()
                .filter(|&t| self.tier_available[t as usize])
                .filter(|&t| self.tier_allowed_by_domain(t))
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
    pub(super) fn tick_aimd(&mut self) {
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
            .filter(|&t| self.tier_allowed_by_domain(t))
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
