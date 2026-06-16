// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Radio, spectrum, mesh, frequency hopping, and energy budget constants.

// ═══════════════════════════════════════════════════════════════════════════════
// Radio / Spectrum Manager (mesh feature)
// Science: Haykin (2005) — cognitive radio, spectrum sensing and dynamic access.
// ═══════════════════════════════════════════════════════════════════════════════

/// SNR threshold below which a channel is considered jammed (dB).
/// Science: Haykin (2005) — cognitive radio spectrum sensing threshold.
pub const RADIO_JAMMING_SNR_THRESHOLD: f32 = 5.0;

/// Arousal spike when jamming is detected (threat response).
/// Science: LeDoux (2003) — amygdala rapid threat detection.
pub const RADIO_JAMMING_AROUSAL_SPIKE: f64 = 0.04;

/// Exploration boost per jamming streak cycle (frequency-hopping search).
/// Science: Berlyne (1960) — curiosity under uncertainty.
pub const RADIO_JAMMING_EXPLORATION_BOOST: f64 = 0.02;

/// Confidence drop per network degradation level.
/// Science: Slovic (1993) — trust asymmetry (harder to build than destroy).
pub const RADIO_DEGRADATION_CONFIDENCE_DROP: f64 = 0.03;

/// EMA alpha for tier packet-loss tracking (higher = faster response).
/// Science: Roberts (1959) — exponential smoothing for signal tracking.
pub const RADIO_TIER_LOSS_EMA_ALPHA: f64 = 0.15;

/// Tier loss EMA threshold above which a tier is considered degraded.
pub const RADIO_TIER_DEGRADED_LOSS: f64 = 0.5;

/// EMA alpha for predicted noise floor tracking.
pub const RADIO_NOISE_FLOOR_EMA_ALPHA: f64 = 0.1;

/// Maximum number of peers tracked for delta compression.
/// Prevents unbounded memory growth in dense networks.
pub const RADIO_MAX_DELTA_PEERS: usize = 64;

/// Total bandwidth (bytes/s) below which Broca speech rate is throttled.
/// Science: Shannon (1948) — channel capacity constrains information throughput.
pub const RADIO_BANDWIDTH_THROTTLE_THRESHOLD: u64 = 500;

/// Connectivity penalty when local tier is down (0.0-1.0 multiplier on swarm EMA).
/// Science: Dunbar (1998) — local network loss degrades social cognition.
pub const RADIO_CONNECTIVITY_PENALTY_LOCAL_DOWN: f64 = 0.7;

/// Connectivity penalty when only metro tier remains (greater degradation).
pub const RADIO_CONNECTIVITY_PENALTY_METRO_ONLY: f64 = 0.4;

/// Default noise floor estimate (dBm) for SpectrumManager initialization.
/// Science: Typical thermal noise floor for UHF receivers (Johnson-Nyquist noise).
pub const RADIO_DEFAULT_NOISE_FLOOR_DBM: f64 = -100.0;

/// Normalizer for spectrum prediction error computation.
/// Science: Maps absolute dBm deviation to [0,1] range (50 dBm = full surprise).
pub const RADIO_NOISE_ERROR_NORMALIZER: f64 = 50.0;

/// Exploration boost when all radio tiers are down (blackout).
/// Science: Isolation drives exploration to seek new connections (foraging theory).
pub const RADIO_BLACKOUT_EXPLORATION_BOOST: f64 = 0.05;

/// Factor applied to max tier loss for learning rate dampening.
/// Science: Unreliable communication → reduce gradient confidence (Jacobson 1988).
pub const RADIO_LOSS_LR_DAMPEN_FACTOR: f64 = 0.2;

/// Maximum LR dampening from tier loss (caps reduction at 15%).
/// Science: Even severe packet loss shouldn't halt learning entirely.
pub const RADIO_LOSS_LR_DAMPEN_MAX: f64 = 0.15;

/// Spectrum prediction error threshold to trigger surprise signal.
/// Science: Signal detection theory — only significant deviations are informative.
pub const RADIO_SPECTRUM_PE_SURPRISE_THRESHOLD: f64 = 0.5;

/// Maximum arousal contribution from spectrum prediction error.
/// Science: Bounded arousal prevents runaway excitation (Yerkes-Dodson 1908).
pub const RADIO_SPECTRUM_PE_AROUSAL_MAX: f32 = 0.08;

/// Arousal scale factor for spectrum prediction error.
/// Science: Linear mapping from PE magnitude to arousal delta.
pub const RADIO_SPECTRUM_PE_AROUSAL_SCALE: f32 = 0.05;

/// Waterfall ring buffer capacity (spectrum observations kept for pattern detection).
/// Science: ~64 observations at 53-cycle interval ≈ 3400 cycles of history.
pub const RADIO_WATERFALL_CAPACITY: usize = 64;

/// Minimum observations to compute waterfall statistics (mean, variance).
/// Science: Central limit theorem — need ≥8 samples for stable estimates.
pub const RADIO_WATERFALL_MIN_SAMPLES: usize = 8;

/// Frequency hop cooldown — minimum cycles between frequency changes.
/// Science: Avoid thrashing; allow SNR to stabilize after a hop (settle time).
pub const RADIO_HOP_COOLDOWN_CYCLES: u32 = 5;

/// SNR improvement threshold to trigger a frequency hop (dB).
/// Science: Only hop if predicted SNR gain exceeds measurement noise floor.
pub const RADIO_HOP_SNR_IMPROVEMENT_DB: f32 = 3.0;

/// Peer discovery beacon interval (cycles between beacon transmissions).
/// Science: Balance discovery latency vs duty cycle budget (Heinrichs 2003).
pub const RADIO_BEACON_INTERVAL_CYCLES: u32 = 100;

/// Peer discovery beacon payload size (bytes). Must fit Regional MTU (50 bytes).
pub const RADIO_BEACON_SIZE: usize = 24;

/// Maximum hops for multi-hop relay routing (TTL).
/// Science: Prevents routing loops; 4 hops covers typical mesh diameter.
pub const RADIO_MAX_RELAY_HOPS: u8 = 4;

/// Maximum routing table entries (prevents unbounded memory growth).
pub const RADIO_MAX_ROUTE_ENTRIES: usize = 128;

/// Route expiry in cycles (stale routes are pruned).
/// Science: Mobile mesh topology changes — stale routes cause packet loss.
pub const RADIO_ROUTE_EXPIRY_CYCLES: u64 = 500;

/// FEC overhead ratio for Metro tier (Reed-Solomon parity bytes / data bytes).
/// Science: RS(255,223) = 14% overhead, corrects up to 16 byte errors.
pub const RADIO_FEC_OVERHEAD_RATIO: f32 = 0.14;

/// FEC minimum payload size (bytes). Below this, FEC overhead exceeds benefit.
pub const RADIO_FEC_MIN_PAYLOAD: usize = 32;

/// Energy cost per bit for Wi-Fi (nJ/bit). Based on 802.11n measurements.
/// Science: Friedman et al. (2013) — measured Wi-Fi energy consumption.
pub const RADIO_ENERGY_PER_BIT_LOCAL: f64 = 50.0;

/// Energy cost per bit for LoRa (nJ/bit). Based on SX1276 datasheet.
/// Science: Semtech SX1276 datasheet — ~20 nJ/bit at SF7.
pub const RADIO_ENERGY_PER_BIT_METRO: f64 = 20.0;

/// Energy cost per bit for HF (nJ/bit). Based on 100W HF at ~50 bps.
/// Science: Amateur radio power measurements — high power, low throughput.
pub const RADIO_ENERGY_PER_BIT_REGIONAL: f64 = 2_000_000.0;

/// Energy budget per cycle (nJ). When exhausted, prefer lowest-energy tier.
/// Science: Bounded energy prevents thermal runaway in embedded systems.
pub const RADIO_ENERGY_BUDGET_PER_CYCLE: f64 = 100_000_000.0;

/// ChaCha20-Poly1305 nonce size (bytes). Per RFC 8439.
pub const RADIO_CRYPTO_NONCE_SIZE: usize = 12;

/// Maximum peers in the encryption key table.
pub const RADIO_CRYPTO_MAX_PEERS: usize = 64;

/// Safety-critical jamming threshold: consecutive jammed observations before escalation.
/// Science: Military EW doctrine — 3+ consecutive jammed scans = sustained threat (Poisel 2011).
pub const RADIO_SAFETY_JAMMING_THRESHOLD: u32 = 3;

/// Auto-hop noise floor threshold (dBm above ambient). If noise > ambient + this, auto-hop triggers.
/// Science: Adaptive frequency hopping in Bluetooth (IEEE 802.15.1) uses 10 dB threshold.
pub const RADIO_AUTO_HOP_NOISE_THRESHOLD: f64 = 10.0;

/// Confidence boost when a beacon confirms a new peer (~0.01 per new peer).
/// Science: Social buffering — connected nodes stabilize confidence (Heinrichs et al. 2003).
pub const RADIO_BEACON_PEER_CONFIDENCE_BOOST: f64 = 0.01;

/// Synthetic SNR for isolated nodes (0 connected peers) — very poor signal.
/// Science: Represents complete isolation, below jamming threshold.
pub const RADIO_SYNTHETIC_SNR_ISOLATED: f64 = 3.0;

/// Base synthetic SNR for connected nodes before peer/phi bonuses.
pub const RADIO_SYNTHETIC_SNR_BASE: f64 = 15.0;

/// Per-peer SNR bonus for synthetic observations (diminishing returns via cap).
pub const RADIO_SYNTHETIC_SNR_PEER_BONUS: f64 = 1.0;

/// Phi-based SNR bonus: collective consciousness coherence improves signal quality.
pub const RADIO_SYNTHETIC_SNR_PHI_BONUS: f64 = 5.0;

/// Maximum peers contributing to synthetic SNR bonus (diminishing returns cap).
pub const RADIO_SYNTHETIC_PEER_CAP: f64 = 10.0;

/// Base noise floor for synthetic observations (dBm).
pub const RADIO_SYNTHETIC_NOISE_FLOOR_BASE: f64 = -95.0;

/// Noise floor range for random variation in synthetic observations.
pub const RADIO_SYNTHETIC_NOISE_FLOOR_RANGE: f64 = 10.0;

/// Energy-aware routing activation threshold: fraction of energy budget spent
/// before switching to energy-efficient tier selection.
/// Science: Adaptive power management in WSN (Heinzelman et al. 2000, LEACH protocol).
pub const RADIO_ENERGY_AWARE_THRESHOLD: f64 = 0.5;

/// Exploration dampening when network is in full blackout (all tiers down).
/// Science: Conservation of resources under extreme stress (Hobfoll 1989).
pub const RADIO_BLACKOUT_STRATEGY_EXPLORATION_DAMPEN: f32 = 0.15;

/// Exploration dampening when network is degraded (metro-only).
/// Science: Moderate stress reduces exploratory behavior (Yerkes-Dodson 1908).
pub const RADIO_DEGRADED_STRATEGY_EXPLORATION_DAMPEN: f32 = 0.05;

/// NE baseline nudge during sustained jamming (≥3 consecutive cycles).
/// Science: Locus coeruleus NE response to sustained threat (Aston-Jones & Cohen 2005).
pub const RADIO_JAMMING_NE_NUDGE: f32 = 0.02;

/// DA baseline nudge on network recovery (jamming/degradation → healthy).
/// Science: Reward prediction error signal on threat resolution (Schultz 1997).
pub const RADIO_RECOVERY_DA_NUDGE: f32 = 0.015;

/// Minimum consecutive jamming cycles before neuromod coupling activates.
pub const RADIO_NEUROMOD_JAMMING_MIN_STREAK: u32 = 3;

/// Consciousness-aware tier selection: high-confidence threshold.
/// Above this, prefer reliable (Local) tier for important transmissions.
pub const RADIO_CONSCIOUSNESS_HIGH_CONFIDENCE: f64 = 0.7;

/// Consciousness-aware tier selection: low-confidence threshold.
/// Below this, prefer energy-efficient (Metro) tier to conserve resources.
pub const RADIO_CONSCIOUSNESS_LOW_CONFIDENCE: f64 = 0.3;
