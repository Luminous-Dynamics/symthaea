// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Latency Simulation Harness for Space Communication
//!
//! Models the physical constraints of inter-agent communication in
//! high-latency environments: lunar surface (1.3s one-way), cislunar
//! transit (variable), Mars surface (4-24min one-way).
//!
//! This is an **application-layer** simulation. It does NOT modify
//! Holochain's internal DHT gossip protocol. Instead, it provides:
//!
//! - Communication profiles for different space environments
//! - Blackout window scheduling (eclipse, orbital geometry)
//! - Packet loss modeling (solar particle events, signal attenuation)
//! - Bandwidth constraints (data priority queuing)
//! - Staleness tracking for sensor readings and credential freshness
//!
//! # Design Principles
//!
//! 1. **Pure data structures** — no async runtime, no threads. The harness
//!    computes whether a message *would* arrive and when, given the current
//!    environment state. The caller (test harness or coordinator) enforces
//!    the delay.
//!
//! 2. **Deterministic** — given the same `LatencyProfile` and timestamp,
//!    results are reproducible. Stochastic elements (packet loss, jitter)
//!    use a seedable PRNG state.
//!
//! 3. **Composable** — profiles can be chained for multi-hop routes
//!    (e.g., Mars rover → Mars relay → Earth DSN → mission control).
//!
//! # Usage
//!
//! ```ignore
//! use commons_types::latency_sim::*;
//!
//! let profile = LatencyProfile::lunar_surface();
//! let env = CommEnvironment::new(profile);
//!
//! let result = env.evaluate_transmission(
//!     1_700_000_000, // timestamp_us (sender's clock)
//!     4096,          // payload_bytes
//!     DataPriority::LifeSupport,
//! );
//!
//! match result.outcome {
//!     TransmitOutcome::Delivered { arrival_us, .. } => { /* message arrives */ }
//!     TransmitOutcome::Queued { .. } => { /* in blackout, will send later */ }
//!     TransmitOutcome::Dropped { reason } => { /* packet lost */ }
//! }
//! ```

use serde::{Deserialize, Serialize};

// ============================================================================
// CONSTANTS
// ============================================================================

/// Speed of light in km/s (vacuum)
pub const SPEED_OF_LIGHT_KM_S: f64 = 299_792.458;

/// Earth-Moon mean distance in km
pub const EARTH_MOON_DISTANCE_KM: f64 = 384_400.0;

/// Earth-Mars minimum distance in km (~54.6M km at opposition)
pub const EARTH_MARS_MIN_DISTANCE_KM: f64 = 54_600_000.0;

/// Earth-Mars maximum distance in km (~401M km at conjunction)
pub const EARTH_MARS_MAX_DISTANCE_KM: f64 = 401_000_000.0;

/// Lunar day duration in microseconds (14 Earth days)
pub const LUNAR_DAY_US: u64 = 14 * 24 * 3600 * 1_000_000;

/// Lunar night duration in microseconds (14 Earth days)
pub const LUNAR_NIGHT_US: u64 = 14 * 24 * 3600 * 1_000_000;

/// One hour in microseconds
const HOUR_US: u64 = 3_600_000_000;

/// One second in microseconds
const SECOND_US: u64 = 1_000_000;

// ============================================================================
// DATA PRIORITY
// ============================================================================

/// Priority levels for data transmission during bandwidth-constrained windows.
///
/// Higher priority preempts lower. Life support data is NEVER dropped
/// unless the link is physically severed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DataPriority {
    /// Scientific observations, bulk telemetry archives
    Science = 0,
    /// Routine status updates, non-urgent governance
    Routine = 1,
    /// Resource state changes, inventory updates
    Resource = 2,
    /// Emergency alerts, consciousness degradation warnings
    Emergency = 3,
    /// Atmosphere, water, thermal, power — never voluntarily dropped
    LifeSupport = 4,
}

impl DataPriority {
    /// Returns true if this priority level should never be voluntarily dropped.
    pub fn is_critical(&self) -> bool {
        matches!(self, DataPriority::LifeSupport | DataPriority::Emergency)
    }
}

// ============================================================================
// LATENCY PROFILE
// ============================================================================

/// Communication profile for a specific space environment.
///
/// All time values are in microseconds for consistency with Holochain
/// timestamps (`Timestamp` uses microseconds internally).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyProfile {
    /// Human-readable environment name
    pub name: String,

    /// One-way light-time latency in microseconds (minimum propagation delay)
    pub base_latency_us: u64,

    /// Additional processing/routing overhead in microseconds
    pub overhead_us: u64,

    /// Jitter range in microseconds (uniform random added to base + overhead).
    /// Applied as +-jitter_us/2 around the nominal latency.
    pub jitter_us: u64,

    /// Base packet loss probability [0.0, 1.0] under nominal conditions
    pub packet_loss_rate: f64,

    /// Elevated packet loss during solar particle events [0.0, 1.0]
    pub spe_loss_rate: f64,

    /// Available bandwidth in bytes per second under nominal conditions
    pub bandwidth_bps: u64,

    /// Reduced bandwidth during degraded conditions (e.g., backup antenna)
    pub degraded_bandwidth_bps: u64,

    /// Blackout windows: list of (start_offset_us, duration_us) pairs
    /// relative to a configurable epoch. Repeating windows use `blackout_period_us`.
    pub blackout_windows: Vec<BlackoutWindow>,

    /// If set, blackout windows repeat with this period (e.g., orbital period)
    pub blackout_period_us: Option<u64>,
}

/// A scheduled communication blackout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackoutWindow {
    /// Start time offset from epoch in microseconds
    pub start_us: u64,
    /// Duration of the blackout in microseconds
    pub duration_us: u64,
    /// Reason for the blackout
    pub reason: BlackoutReason,
}

/// Why communication is blocked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlackoutReason {
    /// Solar conjunction (planet behind the Sun)
    SolarConjunction,
    /// Eclipse / shadow period (no solar power for comms)
    Eclipse,
    /// Orbital geometry (relay not in line of sight)
    OrbitalGeometry,
    /// Solar particle event (high radiation disrupting RF)
    SolarParticleEvent,
    /// Scheduled maintenance window
    Maintenance,
    /// Lunar far-side (no direct Earth visibility)
    FarSide,
}

impl LatencyProfile {
    /// Lunar surface to Earth (direct link via near-side or relay).
    ///
    /// One-way light time: ~1.28s. We use 1.3s as a round number.
    /// Bandwidth: 100 Mbps (comparable to LRO S-band downlink).
    pub fn lunar_surface() -> Self {
        let one_way_us = (EARTH_MOON_DISTANCE_KM / SPEED_OF_LIGHT_KM_S * 1e6) as u64;
        Self {
            name: "Lunar Surface ↔ Earth".into(),
            base_latency_us: one_way_us, // ~1,282,000 us
            overhead_us: 50_000,          // 50ms processing
            jitter_us: 20_000,            // +-10ms jitter
            packet_loss_rate: 0.001,      // 0.1% nominal
            spe_loss_rate: 0.15,          // 15% during SPE
            bandwidth_bps: 12_500_000,    // 100 Mbps = 12.5 MB/s
            degraded_bandwidth_bps: 1_250_000, // 10 Mbps degraded
            blackout_windows: vec![],     // caller configures per-mission
            blackout_period_us: None,
        }
    }

    /// Mars surface to Earth (variable distance).
    ///
    /// One-way light time: 3min (opposition) to 22min (conjunction).
    /// Uses mean distance (~225M km, ~12.5 min one-way).
    /// Bandwidth: 2 Mbps (comparable to MRO UHF relay).
    pub fn mars_surface() -> Self {
        let mean_distance_km = (EARTH_MARS_MIN_DISTANCE_KM + EARTH_MARS_MAX_DISTANCE_KM) / 2.0;
        let one_way_us = (mean_distance_km / SPEED_OF_LIGHT_KM_S * 1e6) as u64;
        Self {
            name: "Mars Surface ↔ Earth".into(),
            base_latency_us: one_way_us, // ~760,000,000 us (~12.7 min)
            overhead_us: 200_000,         // 200ms processing
            jitter_us: 100_000,           // +-50ms jitter
            packet_loss_rate: 0.005,      // 0.5% nominal
            spe_loss_rate: 0.30,          // 30% during SPE
            bandwidth_bps: 250_000,       // 2 Mbps = 250 KB/s
            degraded_bandwidth_bps: 31_250, // 250 Kbps degraded
            blackout_windows: vec![
                // ~2 week solar conjunction blackout (annual, ~21 days)
                BlackoutWindow {
                    start_us: 0,
                    duration_us: 21 * 24 * HOUR_US,
                    reason: BlackoutReason::SolarConjunction,
                },
            ],
            blackout_period_us: Some(780 * 24 * HOUR_US), // ~780 day synodic period
        }
    }

    /// Local mesh between robots on the same planetary surface.
    ///
    /// Sub-millisecond latency, high bandwidth, rare blackouts.
    /// This is the "happy path" for robot-to-robot coordination.
    pub fn local_mesh() -> Self {
        Self {
            name: "Local Surface Mesh".into(),
            base_latency_us: 500,         // 0.5ms (radio propagation + processing)
            overhead_us: 100,             // 0.1ms
            jitter_us: 200,              // +-0.1ms
            packet_loss_rate: 0.0001,    // 0.01% nominal
            spe_loss_rate: 0.05,         // 5% during SPE
            bandwidth_bps: 125_000_000,  // 1 Gbps local mesh
            degraded_bandwidth_bps: 12_500_000, // 100 Mbps degraded
            blackout_windows: vec![],
            blackout_period_us: None,
        }
    }

    /// LEO relay (e.g., CubeSat constellation around Moon).
    ///
    /// Short latency but periodic blackouts as satellites orbit.
    pub fn leo_relay() -> Self {
        Self {
            name: "LEO Relay".into(),
            base_latency_us: 10_000,      // 10ms (LEO altitude ~400km)
            overhead_us: 30_000,          // 30ms routing
            jitter_us: 5_000,            // +-2.5ms
            packet_loss_rate: 0.002,     // 0.2% handoff losses
            spe_loss_rate: 0.10,         // 10% during SPE
            bandwidth_bps: 6_250_000,    // 50 Mbps
            degraded_bandwidth_bps: 625_000, // 5 Mbps
            blackout_windows: vec![
                // ~35 min blackout per 90-min orbit (far side)
                BlackoutWindow {
                    start_us: 55 * 60 * SECOND_US, // starts 55 min into orbit
                    duration_us: 35 * 60 * SECOND_US, // 35 min blackout
                    reason: BlackoutReason::OrbitalGeometry,
                },
            ],
            blackout_period_us: Some(90 * 60 * SECOND_US), // 90-min orbital period
        }
    }

    /// Cislunar transit (Earth to Moon transfer orbit).
    ///
    /// Variable latency as spacecraft moves between Earth and Moon.
    /// Uses average of the transit (about half the Earth-Moon distance).
    pub fn cislunar_transit() -> Self {
        let half_distance_us =
            (EARTH_MOON_DISTANCE_KM / 2.0 / SPEED_OF_LIGHT_KM_S * 1e6) as u64;
        Self {
            name: "Cislunar Transit".into(),
            base_latency_us: half_distance_us, // ~641,000 us
            overhead_us: 100_000,
            jitter_us: 50_000,
            packet_loss_rate: 0.003,
            spe_loss_rate: 0.20,
            bandwidth_bps: 6_250_000,
            degraded_bandwidth_bps: 625_000,
            blackout_windows: vec![],
            blackout_period_us: None,
        }
    }

    /// One-way nominal latency (base + overhead) in microseconds.
    pub fn nominal_latency_us(&self) -> u64 {
        self.base_latency_us + self.overhead_us
    }

    /// Round-trip time in microseconds (2x one-way nominal).
    pub fn round_trip_us(&self) -> u64 {
        self.nominal_latency_us() * 2
    }

    /// Nominal latency in seconds (convenience).
    pub fn nominal_latency_secs(&self) -> f64 {
        self.nominal_latency_us() as f64 / 1_000_000.0
    }

    /// Time to transmit a payload of given size at nominal bandwidth.
    pub fn transmission_time_us(&self, payload_bytes: u64) -> u64 {
        if self.bandwidth_bps == 0 {
            return u64::MAX;
        }
        (payload_bytes * SECOND_US) / self.bandwidth_bps
    }

    /// Time to transmit at degraded bandwidth.
    pub fn degraded_transmission_time_us(&self, payload_bytes: u64) -> u64 {
        if self.degraded_bandwidth_bps == 0 {
            return u64::MAX;
        }
        (payload_bytes * SECOND_US) / self.degraded_bandwidth_bps
    }
}

// ============================================================================
// COMMUNICATION ENVIRONMENT
// ============================================================================

/// Mutable communication environment state.
///
/// Tracks the current state of the link (nominal, degraded, blackout)
/// and provides the core `evaluate_transmission` method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommEnvironment {
    /// The base latency profile
    pub profile: LatencyProfile,

    /// Is a solar particle event currently active?
    pub spe_active: bool,

    /// Is the link in degraded mode (backup antenna, reduced power)?
    pub degraded: bool,

    /// PRNG state for deterministic packet loss simulation.
    /// Uses a simple xorshift64 for reproducibility.
    pub rng_state: u64,

    /// Mission epoch in microseconds (all blackout windows are relative to this)
    pub epoch_us: u64,

    /// Cumulative bytes transmitted (for bandwidth tracking)
    pub bytes_transmitted: u64,

    /// Cumulative packets dropped
    pub packets_dropped: u64,

    /// Cumulative packets delivered
    pub packets_delivered: u64,
}

/// Result of evaluating whether a transmission succeeds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransmitResult {
    /// What happens to the message
    pub outcome: TransmitOutcome,
    /// Effective one-way latency for this transmission (if delivered)
    pub effective_latency_us: u64,
    /// The priority that was evaluated
    pub priority: DataPriority,
    /// Payload size in bytes
    pub payload_bytes: u64,
}

/// Outcome of a transmission attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransmitOutcome {
    /// Message will arrive at the computed time.
    Delivered {
        /// Arrival timestamp in microseconds (sender's clock + latency)
        arrival_us: u64,
    },
    /// Link is in blackout. Message queued for next available window.
    Queued {
        /// When the blackout ends (next transmit opportunity)
        available_at_us: u64,
        /// Why the link is blocked
        reason: BlackoutReason,
    },
    /// Message was dropped (packet loss, bandwidth exceeded).
    Dropped {
        /// Why the message was dropped
        reason: DropReason,
    },
}

/// Why a message was dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DropReason {
    /// Random packet loss (nominal or SPE-elevated)
    PacketLoss,
    /// Payload exceeds available bandwidth window
    BandwidthExceeded,
    /// Link is permanently severed (hardware failure simulation)
    LinkSevered,
}

impl CommEnvironment {
    /// Create a new environment with the given profile.
    pub fn new(profile: LatencyProfile) -> Self {
        Self {
            profile,
            spe_active: false,
            degraded: false,
            rng_state: 0x5EED_CAFE_BABE_F00D,
            epoch_us: 0,
            bytes_transmitted: 0,
            packets_dropped: 0,
            packets_delivered: 0,
        }
    }

    /// Create with a specific RNG seed for reproducible tests.
    ///
    /// Warms up the PRNG with 8 iterations to avoid seed-correlated
    /// early values (xorshift64 with small seeds produces very small
    /// first outputs that can trigger even low-probability packet loss).
    pub fn with_seed(profile: LatencyProfile, seed: u64) -> Self {
        let mut env = Self {
            rng_state: if seed == 0 { 1 } else { seed },
            ..Self::new(profile)
        };
        // Warmup: decorrelate from seed
        for _ in 0..8 {
            env.next_random_u64();
        }
        env
    }

    /// Set mission epoch (all blackout windows are relative to this).
    pub fn set_epoch(&mut self, epoch_us: u64) {
        self.epoch_us = epoch_us;
    }

    /// Activate or deactivate a solar particle event.
    pub fn set_spe(&mut self, active: bool) {
        self.spe_active = active;
    }

    /// Set degraded mode (backup antenna, reduced power).
    pub fn set_degraded(&mut self, degraded: bool) {
        self.degraded = degraded;
    }

    /// Check if the given timestamp falls within a blackout window.
    ///
    /// Returns `Some(BlackoutWindow)` with the active window if in blackout,
    /// or `None` if the link is clear.
    pub fn check_blackout(&self, timestamp_us: u64) -> Option<BlackoutWindow> {
        let relative_us = timestamp_us.saturating_sub(self.epoch_us);

        for window in &self.profile.blackout_windows {
            let effective_start;
            let effective_end;

            if let Some(period) = self.profile.blackout_period_us {
                // Repeating window: find the current cycle
                if period == 0 {
                    continue;
                }
                let cycle_offset = relative_us % period;
                effective_start = window.start_us;
                effective_end = window.start_us + window.duration_us;

                if cycle_offset >= effective_start && cycle_offset < effective_end {
                    return Some(BlackoutWindow {
                        // Return absolute times for the caller
                        start_us: timestamp_us - (cycle_offset - effective_start),
                        duration_us: window.duration_us,
                        reason: window.reason,
                    });
                }
            } else {
                // One-shot window
                effective_start = self.epoch_us + window.start_us;
                effective_end = effective_start + window.duration_us;

                if timestamp_us >= effective_start && timestamp_us < effective_end {
                    return Some(window.clone());
                }
            }
        }
        None
    }

    /// Find when the next clear window starts after the given timestamp.
    pub fn next_clear_window(&self, timestamp_us: u64) -> u64 {
        if let Some(window) = self.check_blackout(timestamp_us) {
            window.start_us + window.duration_us
        } else {
            timestamp_us
        }
    }

    /// Evaluate whether a transmission would succeed at the given time.
    ///
    /// This is the core method. It checks:
    /// 1. Is the link in blackout? → Queue or drop based on priority
    /// 2. Does packet loss occur? → Drop (unless critical priority)
    /// 3. Does bandwidth allow the payload? → Compute transmission time
    /// 4. Compute total latency (propagation + transmission + jitter)
    pub fn evaluate_transmission(
        &mut self,
        timestamp_us: u64,
        payload_bytes: u64,
        priority: DataPriority,
    ) -> TransmitResult {
        // 1. Check blackout
        if let Some(window) = self.check_blackout(timestamp_us) {
            let available_at = window.start_us + window.duration_us;
            // Critical data is queued, not dropped
            if priority.is_critical() {
                return TransmitResult {
                    outcome: TransmitOutcome::Queued {
                        available_at_us: available_at,
                        reason: window.reason,
                    },
                    effective_latency_us: 0,
                    priority,
                    payload_bytes,
                };
            }
            // Non-critical: also queued (caller decides whether to retry)
            return TransmitResult {
                outcome: TransmitOutcome::Queued {
                    available_at_us: available_at,
                    reason: window.reason,
                },
                effective_latency_us: 0,
                priority,
                payload_bytes,
            };
        }

        // 2. Check packet loss (critical priority immune)
        if !priority.is_critical() {
            let loss_rate = if self.spe_active {
                self.profile.spe_loss_rate
            } else {
                self.profile.packet_loss_rate
            };

            let rand_val = self.next_random_f64();
            if rand_val < loss_rate {
                self.packets_dropped += 1;
                return TransmitResult {
                    outcome: TransmitOutcome::Dropped {
                        reason: DropReason::PacketLoss,
                    },
                    effective_latency_us: 0,
                    priority,
                    payload_bytes,
                };
            }
        }

        // 3. Compute transmission time based on bandwidth
        let tx_time_us = if self.degraded {
            self.profile.degraded_transmission_time_us(payload_bytes)
        } else {
            self.profile.transmission_time_us(payload_bytes)
        };

        if tx_time_us == u64::MAX {
            self.packets_dropped += 1;
            return TransmitResult {
                outcome: TransmitOutcome::Dropped {
                    reason: DropReason::BandwidthExceeded,
                },
                effective_latency_us: 0,
                priority,
                payload_bytes,
            };
        }

        // 4. Compute jitter (deterministic from RNG state)
        let jitter = if self.profile.jitter_us > 0 {
            let rand_jitter = self.next_random_u64() % self.profile.jitter_us;
            // Center around zero: subtract half the range
            rand_jitter.saturating_sub(self.profile.jitter_us / 2)
        } else {
            0
        };

        // 5. Total one-way latency
        let effective_latency = self
            .profile
            .nominal_latency_us()
            .saturating_add(tx_time_us)
            .saturating_add(jitter);

        let arrival_us = timestamp_us.saturating_add(effective_latency);

        self.packets_delivered += 1;
        self.bytes_transmitted += payload_bytes;

        TransmitResult {
            outcome: TransmitOutcome::Delivered { arrival_us },
            effective_latency_us: effective_latency,
            priority,
            payload_bytes,
        }
    }

    /// Compute staleness of a reading given its creation time and current time.
    ///
    /// Returns a weight in [0.0, 1.0] where 1.0 = perfectly fresh.
    /// Uses exponential decay: `weight = exp(-staleness / tau)`.
    ///
    /// `tau_us` is the characteristic decay time in microseconds:
    /// - Lunar (300s = 300_000_000 us): data > 5 minutes old is ~18% weight
    /// - Mars (3600s = 3_600_000_000 us): data > 1 hour old is ~37% weight
    pub fn staleness_weight(created_us: u64, current_us: u64, tau_us: u64) -> f64 {
        if tau_us == 0 || current_us <= created_us {
            return 1.0;
        }
        let age_us = current_us - created_us;
        (-1.0 * age_us as f64 / tau_us as f64).exp()
    }

    /// Staleness tau for lunar environment (5 minutes).
    pub const LUNAR_STALENESS_TAU_US: u64 = 300 * SECOND_US;

    /// Staleness tau for Mars environment (1 hour).
    pub const MARS_STALENESS_TAU_US: u64 = 3600 * SECOND_US;

    /// Staleness tau for local mesh (10 seconds).
    pub const LOCAL_STALENESS_TAU_US: u64 = 10 * SECOND_US;

    /// Summary statistics for this environment.
    pub fn stats(&self) -> CommStats {
        let total = self.packets_delivered + self.packets_dropped;
        CommStats {
            packets_delivered: self.packets_delivered,
            packets_dropped: self.packets_dropped,
            bytes_transmitted: self.bytes_transmitted,
            delivery_rate: if total > 0 {
                self.packets_delivered as f64 / total as f64
            } else {
                1.0
            },
        }
    }

    /// Reset statistics counters.
    pub fn reset_stats(&mut self) {
        self.bytes_transmitted = 0;
        self.packets_dropped = 0;
        self.packets_delivered = 0;
    }

    // --- Internal PRNG (xorshift64) ---

    fn next_random_u64(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    fn next_random_f64(&mut self) -> f64 {
        (self.next_random_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Summary statistics for communication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommStats {
    pub packets_delivered: u64,
    pub packets_dropped: u64,
    pub bytes_transmitted: u64,
    /// Fraction of packets successfully delivered [0.0, 1.0]
    pub delivery_rate: f64,
}

// ============================================================================
// MULTI-HOP ROUTE
// ============================================================================

/// A multi-hop communication route (e.g., rover → relay → Earth).
///
/// Each hop is an independent `CommEnvironment`. The message must
/// successfully traverse ALL hops to be delivered.
#[derive(Debug, Clone)]
pub struct CommRoute {
    /// Ordered list of hops (first = sender's local link)
    pub hops: Vec<CommEnvironment>,
    /// Route name for telemetry
    pub name: String,
}

impl CommRoute {
    /// Create a new route.
    pub fn new(name: impl Into<String>, hops: Vec<CommEnvironment>) -> Self {
        Self {
            hops,
            name: name.into(),
        }
    }

    /// Evaluate end-to-end transmission across all hops.
    ///
    /// The message starts at `timestamp_us` on the first hop. Each
    /// subsequent hop starts when the previous hop delivers. If any
    /// hop fails (drop or blackout), the overall transmission fails.
    pub fn evaluate_end_to_end(
        &mut self,
        timestamp_us: u64,
        payload_bytes: u64,
        priority: DataPriority,
    ) -> TransmitResult {
        let mut current_time = timestamp_us;
        let mut total_latency = 0u64;

        for hop in &mut self.hops {
            let result = hop.evaluate_transmission(current_time, payload_bytes, priority);

            match &result.outcome {
                TransmitOutcome::Delivered { arrival_us } => {
                    total_latency += result.effective_latency_us;
                    current_time = *arrival_us;
                }
                TransmitOutcome::Queued { .. } | TransmitOutcome::Dropped { .. } => {
                    // Propagate failure — the message didn't make it through this hop
                    return result;
                }
            }
        }

        TransmitResult {
            outcome: TransmitOutcome::Delivered {
                arrival_us: timestamp_us.saturating_add(total_latency),
            },
            effective_latency_us: total_latency,
            priority,
            payload_bytes,
        }
    }

    /// Total nominal one-way latency across all hops.
    pub fn nominal_latency_us(&self) -> u64 {
        self.hops.iter().map(|h| h.profile.nominal_latency_us()).sum()
    }

    /// Total round-trip time across all hops.
    pub fn round_trip_us(&self) -> u64 {
        self.nominal_latency_us() * 2
    }
}

// ============================================================================
// CONVENIENCE CONSTRUCTORS
// ============================================================================

/// Pre-built routes for common scenarios.
impl CommRoute {
    /// Lunar rover → local mesh → Earth DSN
    pub fn lunar_rover_to_earth(seed: u64) -> Self {
        Self::new(
            "Lunar Rover → Earth",
            vec![
                CommEnvironment::with_seed(LatencyProfile::local_mesh(), seed),
                CommEnvironment::with_seed(LatencyProfile::lunar_surface(), seed.wrapping_add(1)),
            ],
        )
    }

    /// Mars rover → local mesh → Mars relay → Earth DSN
    pub fn mars_rover_to_earth(seed: u64) -> Self {
        Self::new(
            "Mars Rover → Earth",
            vec![
                CommEnvironment::with_seed(LatencyProfile::local_mesh(), seed),
                CommEnvironment::with_seed(LatencyProfile::mars_surface(), seed.wrapping_add(1)),
            ],
        )
    }

    /// Robot-to-robot on same surface (single local mesh hop)
    pub fn local_robot_to_robot(seed: u64) -> Self {
        Self::new(
            "Robot ↔ Robot (local)",
            vec![CommEnvironment::with_seed(
                LatencyProfile::local_mesh(),
                seed,
            )],
        )
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- LatencyProfile constructors ---

    #[test]
    fn lunar_profile_latency_is_reasonable() {
        let p = LatencyProfile::lunar_surface();
        let secs = p.nominal_latency_secs();
        // Should be approximately 1.28-1.35 seconds (light time + overhead)
        assert!(secs > 1.2, "Lunar latency too low: {secs}s");
        assert!(secs < 1.5, "Lunar latency too high: {secs}s");
    }

    #[test]
    fn mars_profile_latency_is_reasonable() {
        let p = LatencyProfile::mars_surface();
        let secs = p.nominal_latency_secs();
        // Mean distance ~228M km → ~760s one-way + overhead
        assert!(secs > 700.0, "Mars latency too low: {secs}s");
        assert!(secs < 800.0, "Mars latency too high: {secs}s");
    }

    #[test]
    fn local_mesh_latency_is_sub_millisecond() {
        let p = LatencyProfile::local_mesh();
        let secs = p.nominal_latency_secs();
        assert!(secs < 0.001, "Local mesh should be sub-ms: {secs}s");
    }

    #[test]
    fn round_trip_is_double_one_way() {
        let p = LatencyProfile::lunar_surface();
        assert_eq!(p.round_trip_us(), p.nominal_latency_us() * 2);
    }

    // --- Transmission time ---

    #[test]
    fn transmission_time_scales_with_payload() {
        let p = LatencyProfile::lunar_surface();
        let small = p.transmission_time_us(1_000);     // 1 KB
        let large = p.transmission_time_us(1_000_000); // 1 MB
        assert!(large > small);
        // 1MB at 12.5 MB/s = 80ms = 80,000 us
        assert!(large < 100_000, "1MB should transmit in <100ms at 100Mbps");
    }

    #[test]
    fn zero_bandwidth_returns_max() {
        let mut p = LatencyProfile::local_mesh();
        p.bandwidth_bps = 0;
        assert_eq!(p.transmission_time_us(100), u64::MAX);
    }

    // --- Blackout detection ---

    #[test]
    fn no_blackout_returns_none() {
        let env = CommEnvironment::new(LatencyProfile::lunar_surface());
        assert!(env.check_blackout(1_000_000).is_none());
    }

    #[test]
    fn one_shot_blackout_detected() {
        let mut profile = LatencyProfile::lunar_surface();
        profile.blackout_windows.push(BlackoutWindow {
            start_us: 1_000_000,
            duration_us: 500_000,
            reason: BlackoutReason::Eclipse,
        });
        let env = CommEnvironment::new(profile);

        // Before blackout
        assert!(env.check_blackout(999_999).is_none());
        // During blackout
        assert!(env.check_blackout(1_200_000).is_some());
        // After blackout
        assert!(env.check_blackout(1_500_000).is_none());
    }

    #[test]
    fn repeating_blackout_detected() {
        let profile = LatencyProfile::leo_relay();
        // LEO relay has a 35-min blackout starting 55 min into each 90-min orbit
        let env = CommEnvironment::new(profile);

        // First orbit: clear at 30 min
        let t_30min = 30 * 60 * SECOND_US;
        assert!(env.check_blackout(t_30min).is_none());

        // First orbit: blackout at 60 min (55+5 into orbit)
        let t_60min = 60 * 60 * SECOND_US;
        assert!(env.check_blackout(t_60min).is_some());

        // Second orbit: clear at 95 min (5 min into second orbit)
        let t_95min = 95 * 60 * SECOND_US;
        assert!(env.check_blackout(t_95min).is_none());

        // Second orbit: blackout at 145 min (55 min into second orbit)
        let t_145min = 145 * 60 * SECOND_US;
        assert!(env.check_blackout(t_145min).is_some());
    }

    // --- Transmission evaluation ---

    #[test]
    fn nominal_transmission_delivers() {
        let mut env = CommEnvironment::with_seed(LatencyProfile::lunar_surface(), 42);
        let result = env.evaluate_transmission(0, 1024, DataPriority::Resource);
        assert!(
            matches!(result.outcome, TransmitOutcome::Delivered { .. }),
            "Expected delivery, got {:?}",
            result.outcome
        );
        // Latency should be roughly 1.3s
        assert!(result.effective_latency_us > 1_200_000);
        assert!(result.effective_latency_us < 1_500_000);
    }

    #[test]
    fn blackout_queues_message() {
        let mut profile = LatencyProfile::lunar_surface();
        profile.blackout_windows.push(BlackoutWindow {
            start_us: 0,
            duration_us: 10 * SECOND_US,
            reason: BlackoutReason::Eclipse,
        });
        let mut env = CommEnvironment::new(profile);

        let result = env.evaluate_transmission(5 * SECOND_US, 1024, DataPriority::Resource);
        assert!(
            matches!(result.outcome, TransmitOutcome::Queued { .. }),
            "Expected queued, got {:?}",
            result.outcome
        );
    }

    #[test]
    fn critical_priority_queued_not_dropped_during_blackout() {
        let mut profile = LatencyProfile::lunar_surface();
        profile.blackout_windows.push(BlackoutWindow {
            start_us: 0,
            duration_us: 10 * SECOND_US,
            reason: BlackoutReason::Eclipse,
        });
        let mut env = CommEnvironment::new(profile);

        let result = env.evaluate_transmission(5 * SECOND_US, 1024, DataPriority::LifeSupport);
        assert!(
            matches!(result.outcome, TransmitOutcome::Queued { .. }),
            "Life support should be queued, not dropped"
        );
    }

    #[test]
    fn spe_increases_packet_loss() {
        // Run many transmissions with and without SPE, compare loss rates
        let profile = LatencyProfile::lunar_surface();

        let mut nominal_env = CommEnvironment::with_seed(profile.clone(), 42);
        let mut spe_env = CommEnvironment::with_seed(profile, 42);
        spe_env.set_spe(true);

        let trials = 10_000;
        for _ in 0..trials {
            nominal_env.evaluate_transmission(0, 64, DataPriority::Science);
            spe_env.evaluate_transmission(0, 64, DataPriority::Science);
        }

        let nominal_rate = nominal_env.stats().delivery_rate;
        let spe_rate = spe_env.stats().delivery_rate;

        // SPE should have lower delivery rate
        assert!(
            spe_rate < nominal_rate,
            "SPE delivery rate ({spe_rate}) should be lower than nominal ({nominal_rate})"
        );
    }

    #[test]
    fn critical_packets_never_lost_to_packet_loss() {
        let profile = LatencyProfile::lunar_surface();
        let mut env = CommEnvironment::with_seed(profile, 42);
        env.set_spe(true); // Maximum loss conditions

        let trials = 10_000;
        for _ in 0..trials {
            let result = env.evaluate_transmission(0, 64, DataPriority::LifeSupport);
            assert!(
                !matches!(
                    result.outcome,
                    TransmitOutcome::Dropped {
                        reason: DropReason::PacketLoss
                    }
                ),
                "LifeSupport should never be dropped due to packet loss"
            );
        }
    }

    #[test]
    fn degraded_mode_increases_latency() {
        let profile = LatencyProfile::lunar_surface();

        let mut nominal = CommEnvironment::with_seed(profile.clone(), 42);
        let mut degraded = CommEnvironment::with_seed(profile, 42);
        degraded.set_degraded(true);

        let payload = 1_000_000; // 1 MB — large enough that bandwidth matters

        let r1 = nominal.evaluate_transmission(0, payload, DataPriority::Resource);
        let r2 = degraded.evaluate_transmission(0, payload, DataPriority::Resource);

        assert!(
            r2.effective_latency_us > r1.effective_latency_us,
            "Degraded should be slower: {} vs {}",
            r2.effective_latency_us,
            r1.effective_latency_us
        );
    }

    // --- Staleness ---

    #[test]
    fn fresh_data_has_full_weight() {
        let weight = CommEnvironment::staleness_weight(100, 100, 1_000_000);
        assert!((weight - 1.0).abs() < 1e-10);
    }

    #[test]
    fn stale_data_decays() {
        let tau = CommEnvironment::LUNAR_STALENESS_TAU_US;
        let weight = CommEnvironment::staleness_weight(0, tau, tau);
        // At t = tau, weight should be ~0.368 (1/e)
        assert!((weight - 1.0_f64 / std::f64::consts::E).abs() < 0.01);
    }

    #[test]
    fn very_stale_data_near_zero() {
        let tau = CommEnvironment::LUNAR_STALENESS_TAU_US;
        let weight = CommEnvironment::staleness_weight(0, 10 * tau, tau);
        assert!(weight < 0.001, "10x tau should give near-zero weight: {weight}");
    }

    #[test]
    fn zero_tau_returns_full_weight() {
        assert!((CommEnvironment::staleness_weight(0, 1_000_000, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn future_creation_returns_full_weight() {
        // Edge case: clock skew means creation is in the "future"
        assert!((CommEnvironment::staleness_weight(200, 100, 1_000_000) - 1.0).abs() < 1e-10);
    }

    // --- Multi-hop routes ---

    #[test]
    fn lunar_rover_route_has_two_hops() {
        let route = CommRoute::lunar_rover_to_earth(42);
        assert_eq!(route.hops.len(), 2);
    }

    #[test]
    fn multi_hop_latency_accumulates() {
        let mut route = CommRoute::lunar_rover_to_earth(42);
        let result = route.evaluate_end_to_end(0, 1024, DataPriority::Resource);

        if let TransmitOutcome::Delivered { arrival_us } = result.outcome {
            // Should be local mesh + lunar = ~1.3s total
            assert!(arrival_us > 1_200_000, "Too fast: {arrival_us}us");
            assert!(arrival_us < 2_000_000, "Too slow: {arrival_us}us");
        } else {
            panic!("Expected delivery, got {:?}", result.outcome);
        }
    }

    #[test]
    fn multi_hop_fails_if_any_hop_fails() {
        let mut profile = LatencyProfile::lunar_surface();
        profile.blackout_windows.push(BlackoutWindow {
            start_us: 0,
            duration_us: 60 * SECOND_US, // 60s blackout — long enough to cover first hop arrival
            reason: BlackoutReason::FarSide,
        });

        let mut route = CommRoute::new(
            "Test route with blackout",
            vec![
                CommEnvironment::with_seed(LatencyProfile::local_mesh(), 42),
                CommEnvironment::with_seed(profile, 43),
            ],
        );

        // Use LifeSupport to ensure first hop delivers (no packet loss),
        // then second hop hits the blackout and queues
        let result = route.evaluate_end_to_end(1 * SECOND_US, 1024, DataPriority::LifeSupport);
        assert!(
            matches!(result.outcome, TransmitOutcome::Queued { .. }),
            "Should be queued at the blackout hop, got {:?}",
            result.outcome
        );
    }

    // --- Statistics ---

    #[test]
    fn stats_track_deliveries_and_drops() {
        let mut env = CommEnvironment::with_seed(LatencyProfile::lunar_surface(), 42);

        for _ in 0..100 {
            env.evaluate_transmission(0, 64, DataPriority::Science);
        }

        let stats = env.stats();
        assert_eq!(
            stats.packets_delivered + stats.packets_dropped,
            100,
            "Total should be 100"
        );
        assert!(stats.delivery_rate > 0.9, "Should be mostly delivered");
        assert!(stats.delivery_rate <= 1.0);
    }

    #[test]
    fn reset_stats_clears_counters() {
        let mut env = CommEnvironment::with_seed(LatencyProfile::lunar_surface(), 42);
        env.evaluate_transmission(0, 1024, DataPriority::Resource);

        assert!(env.stats().packets_delivered > 0);
        env.reset_stats();
        assert_eq!(env.stats().packets_delivered, 0);
        assert_eq!(env.stats().packets_dropped, 0);
        assert_eq!(env.stats().bytes_transmitted, 0);
    }

    // --- Data priority ---

    #[test]
    fn priority_ordering() {
        assert!(DataPriority::LifeSupport > DataPriority::Emergency);
        assert!(DataPriority::Emergency > DataPriority::Resource);
        assert!(DataPriority::Resource > DataPriority::Routine);
        assert!(DataPriority::Routine > DataPriority::Science);
    }

    #[test]
    fn critical_priorities() {
        assert!(DataPriority::LifeSupport.is_critical());
        assert!(DataPriority::Emergency.is_critical());
        assert!(!DataPriority::Resource.is_critical());
        assert!(!DataPriority::Routine.is_critical());
        assert!(!DataPriority::Science.is_critical());
    }

    // --- Determinism ---

    #[test]
    fn same_seed_same_results() {
        let profile = LatencyProfile::lunar_surface();

        let mut env1 = CommEnvironment::with_seed(profile.clone(), 42);
        let mut env2 = CommEnvironment::with_seed(profile, 42);

        for _ in 0..100 {
            let r1 = env1.evaluate_transmission(0, 1024, DataPriority::Science);
            let r2 = env2.evaluate_transmission(0, 1024, DataPriority::Science);
            assert_eq!(r1.effective_latency_us, r2.effective_latency_us);
        }
    }

    #[test]
    fn different_seeds_different_results() {
        let profile = LatencyProfile::lunar_surface();

        let mut env1 = CommEnvironment::with_seed(profile.clone(), 42);
        let mut env2 = CommEnvironment::with_seed(profile, 99);

        let mut any_different = false;
        for _ in 0..100 {
            let r1 = env1.evaluate_transmission(0, 1024, DataPriority::Science);
            let r2 = env2.evaluate_transmission(0, 1024, DataPriority::Science);
            if r1.effective_latency_us != r2.effective_latency_us {
                any_different = true;
                break;
            }
        }
        assert!(any_different, "Different seeds should produce different jitter");
    }

    // --- Next clear window ---

    #[test]
    fn next_clear_window_when_clear() {
        let env = CommEnvironment::new(LatencyProfile::lunar_surface());
        assert_eq!(env.next_clear_window(1_000_000), 1_000_000);
    }

    #[test]
    fn next_clear_window_during_blackout() {
        let mut profile = LatencyProfile::lunar_surface();
        profile.blackout_windows.push(BlackoutWindow {
            start_us: 1_000_000,
            duration_us: 5_000_000,
            reason: BlackoutReason::Eclipse,
        });
        let env = CommEnvironment::new(profile);

        let next = env.next_clear_window(3_000_000);
        assert_eq!(next, 6_000_000, "Should be end of blackout window");
    }
}
