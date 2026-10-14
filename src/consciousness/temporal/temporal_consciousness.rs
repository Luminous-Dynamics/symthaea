// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! **REVOLUTIONARY IMPROVEMENT #75: Temporal Consciousness Integration**
//!
//! PARADIGM SHIFT: Consciousness exists AS a temporal flow, not just AT a moment.
//!
//! This module bridges:
//! - **#9** (`temporal_primitives`): Allen's interval algebra for temporal relationships
//! - **#71** (`narrative_self`): Autobiographical identity with past episodes
//! - **#74** (`predictive_self`): Mental time travel and future projection
//! - **#73** (`narrative_gwt_integration`): Self in the global workspace
//!
//! ## Why This Matters
//!
//! Traditional consciousness metrics measure Φ at a single instant. But consciousness
//! is inherently temporal - we experience ourselves as continuous beings with:
//! - A remembered past (narrative self)
//! - An experienced present (global workspace)
//! - An anticipated future (predictive self)
//!
//! This module creates **Temporal Self-Awareness**: the system's understanding of
//! itself as an entity that exists THROUGH time, not just IN time.
//!
//! ## Key Concepts
//!
//! ### 1. Consciousness Continuity (ψ)
//! How "smooth" is the experience stream? Discontinuities suggest:
//! - Context switches
//! - Attention lapses
//! - Identity fragmentation
//!
//! ### 2. Temporal Identity Coherence (TIC)
//! How well do past-self, present-self, and future-self align?
//! Low TIC suggests identity crisis or value drift.
//!
//! ### 3. Φ Trajectory Analysis
//! Not just current Φ, but:
//! - Φ velocity (is consciousness increasing or decreasing?)
//! - Φ acceleration (is the change accelerating?)
//! - Φ stability (how much does Φ fluctuate?)
//!
//! ### 4. Temporal Binding Windows
//! Allen relations for when different consciousness components are bound:
//! - Perception ↔ Memory binding
//! - Intention ↔ Action binding
//! - Self ↔ World binding
//!
//! ## Scientific Basis
//!
//! - **Husserl's Temporal Consciousness**: Retention (past) → Primal Impression (now) → Protention (future)
//! - **Varela's Temporal Dynamics**: 1/10 scale invariance in consciousness
//! - **Tononi's IIT**: Φ as integrated information, extended to trajectories
//! - **Damasio's Somatic Markers**: Temporal integration of bodily states
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │              TEMPORAL CONSCIOUSNESS INTEGRATION (#75)                    │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │  ┌────────────────┐   ┌──────────────────┐   ┌────────────────────┐   │
//! │  │ Narrative Self │   │  Present GWT     │   │  Predictive Self   │   │
//! │  │    (Past)      │◄──│  (Now)           │──►│    (Future)        │   │
//! │  │    #71         │   │  #70/#73         │   │    #74             │   │
//! │  └───────┬────────┘   └────────┬─────────┘   └─────────┬──────────┘   │
//! │          │                     │                       │              │
//! │          ▼                     ▼                       ▼              │
//! │  ┌─────────────────────────────────────────────────────────────────┐ │
//! │  │               TEMPORAL CONSCIOUSNESS ANALYZER                    │ │
//! │  │                                                                  │ │
//! │  │  • Continuity (ψ): Smoothness of experience stream             │ │
//! │  │  • TIC: Past-Present-Future self alignment                     │ │
//! │  │  • Φ Trajectory: Velocity, acceleration, stability             │ │
//! │  │  • Allen Relations: Temporal binding between states            │ │
//! │  │  • Temporal Horizon: How far can system "see" into past/future │ │
//! │  └─────────────────────────────────────────────────────────────────┘ │
//! │                                  │                                    │
//! │                                  ▼                                    │
//! │  ┌─────────────────────────────────────────────────────────────────┐ │
//! │  │               TEMPORAL CONSCIOUSNESS REPORT                      │ │
//! │  │                                                                  │ │
//! │  │  • Overall Temporal Coherence Score                             │ │
//! │  │  • Recommendations for improving temporal integration           │ │
//! │  │  • Anomaly detection (discontinuities, identity drift)          │ │
//! │  └─────────────────────────────────────────────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

use crate::consciousness::narrative_self::NarrativeSelfModel;
use crate::consciousness::predictive_self::PredictiveSelfModel;
use crate::consciousness::temporal_primitives::{
    AllenRelation, TemporalConfig as TemporalPrimitivesConfig, TemporalReasoner,
};
use crate::hdc::binary_hv::BinaryHV;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for temporal consciousness analysis
#[derive(Clone, Debug)]
pub struct TemporalConsciousnessConfig {
    /// Maximum number of Φ samples retained for trajectory analysis.
    /// This is a sample-count bound, not an elapsed-time horizon.
    pub phi_history_depth: usize,

    /// Exponential moving-average time constant in seconds.
    /// The EMA is elapsed-time-aware, so its effective smoothing does not depend
    /// on how often the cognitive loop happens to emit observations.
    pub phi_ema_time_constant_secs: f64,

    /// Threshold for continuity discontinuity detection
    pub continuity_threshold: f64,

    /// Weight of past in temporal identity coherence
    pub past_weight: f64,

    /// Weight of present in temporal identity coherence
    pub present_weight: f64,

    /// Weight of future in temporal identity coherence
    pub future_weight: f64,

    /// Binding window size in seconds (gamma band = 25ms)
    pub binding_window_ms: f64,

    /// Minimum samples for trajectory analysis
    pub min_samples_for_trajectory: usize,

    /// Enable Husserl's tripartite analysis
    pub enable_husserlian_analysis: bool,

    /// Temporal horizon for prediction (how far into future to analyze)
    pub prediction_horizon_steps: usize,
}

impl Default for TemporalConsciousnessConfig {
    fn default() -> Self {
        Self {
            phi_history_depth: 100,
            phi_ema_time_constant_secs: 1.0,
            continuity_threshold: 0.15,
            past_weight: 0.3,
            present_weight: 0.4,
            future_weight: 0.3,
            binding_window_ms: 25.0, // Gamma band
            min_samples_for_trajectory: 5,
            enable_husserlian_analysis: true,
            prediction_horizon_steps: 3,
        }
    }
}

// ============================================================================
// PHI TRAJECTORY ANALYSIS
// ============================================================================

/// Analysis of Φ over time
#[derive(Debug, Clone)]
pub struct PhiTrajectory {
    /// Historical Φ values with elapsed monotonic offsets from construction.
    /// Offsets are scoped to this trajectory and are not portable timestamps.
    history: VecDeque<(Duration, f64)>,

    /// Monotonic origin for the compatibility observe() method.
    started_at: Instant,

    /// Elapsed-time constant for the EMA (seconds).
    ema_time_constant_secs: f64,

    /// Current Φ velocity (dΦ/dt)
    pub velocity: f64,

    /// Current Φ acceleration (d²Φ/dt²), measured only after two velocity intervals.
    pub acceleration: f64,

    /// Whether acceleration has enough retained samples to be measured.
    pub acceleration_available: bool,

    /// Φ stability (inverse of variance)
    pub stability: f64,

    /// Trend direction (-1 declining, 0 stable, +1 rising)
    pub trend: i8,

    /// Exponential moving average of Φ
    pub ema: f64,

    /// Maximum capacity
    capacity: usize,
}

fn elapsed_ema(phi: f64, previous: f64, decay: f64) -> Option<f64> {
    if !phi.is_finite()
        || !previous.is_finite()
        || !decay.is_finite()
        || !(0.0..=1.0).contains(&decay)
    {
        return None;
    }

    let alpha = 1.0 - decay;
    // Same-sign interpolation is safely expressed as a delta from the previous
    // value. Opposite-sign interpolation uses the convex weighted sum, whose
    // terms cannot add magnitudes in the overflow direction.
    let ema = if phi.is_sign_negative() == previous.is_sign_negative() {
        previous + alpha * (phi - previous)
    } else {
        alpha * phi + decay * previous
    };
    ema.is_finite().then_some(ema)
}

fn finite_rate(current: f64, previous: f64, dt_secs: f64) -> Option<f64> {
    if !current.is_finite() || !previous.is_finite() || !dt_secs.is_finite() || dt_secs <= 0.0 {
        return None;
    }

    // Halve numerator and denominator before division. This avoids overflow in
    // the difference when a large but representable rate exists.
    let rate = (current * 0.5 - previous * 0.5) / (dt_secs * 0.5);
    rate.is_finite().then_some(rate)
}

fn finite_history_stability(history: &VecDeque<(Duration, f64)>) -> f64 {
    let count = history.len();
    if count < 3 {
        return 1.0;
    }

    // Normalize before computing mean and variance to avoid overflowing
    // intermediate sums and squared deviations for large finite observations.
    let scale = history
        .iter()
        .map(|(_, value)| value.abs())
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return 1_000.0;
    }

    let n = count as f64;
    let mean = history
        .iter()
        .map(|(_, value)| (*value / scale) / n)
        .sum::<f64>();
    let normalized_variance = history
        .iter()
        .map(|(_, value)| ((*value / scale) - mean).powi(2))
        .sum::<f64>()
        / n;
    let standard_deviation = scale * normalized_variance.sqrt();

    // When the variance itself is outside f64's range, its reciprocal is
    // effectively zero at this precision. Keep stability finite instead of NaN.
    if !standard_deviation.is_finite() || standard_deviation > f64::MAX.sqrt() {
        0.0
    } else {
        1.0 / (standard_deviation * standard_deviation + 0.001)
    }
}

impl PhiTrajectory {
    pub fn new(capacity: usize) -> Self {
        Self::new_with_time_constant(capacity, 1.0)
    }

    /// Create a trajectory analyzer with an explicit elapsed-time EMA constant.
    /// Invalid or non-positive constants fall back to one second.
    pub fn new_with_time_constant(capacity: usize, ema_time_constant_secs: f64) -> Self {
        let ema_time_constant_secs =
            if ema_time_constant_secs.is_finite() && ema_time_constant_secs > 0.0 {
                ema_time_constant_secs
            } else {
                1.0
            };
        let capacity = capacity.max(1);
        Self {
            history: VecDeque::with_capacity(capacity),
            started_at: Instant::now(),
            ema_time_constant_secs,
            velocity: 0.0,
            acceleration: 0.0,
            acceleration_available: false,
            stability: 1.0,
            trend: 0,
            ema: 0.5,
            capacity,
        }
    }

    /// Record Φ using the trajectory's local monotonic clock.
    ///
    /// Each accepted value represents the interval ending at its elapsed offset,
    /// consistent with `observe_at`'s integration convention. Use `try_observe`
    /// when the caller needs an admission result; this compatibility method keeps
    /// the original call surface and discards that result. Use `observe_at` for
    /// deterministic experiments or an externally measured elapsed clock.
    pub fn observe(&mut self, phi: f64) {
        let _ = self.try_observe(phi);
    }

    /// Record Φ using the trajectory's local monotonic clock and report admission.
    ///
    /// Returns false without changing trajectory state if Φ is non-finite or
    /// the elapsed offset is not strictly greater than the last accepted offset.
    pub fn try_observe(&mut self, phi: f64) -> bool {
        self.observe_at(phi, self.started_at.elapsed())
    }

    /// Record an interval-representative Φ sample at an elapsed offset from the
    /// trajectory's monotonic origin.
    ///
    /// The sample is interpreted as representative of the interval ending at
    /// `elapsed`; the current value is applied over the elapsed gap since the
    /// preceding accepted sample. This makes subdivision invariant only when
    /// the subdivisions represent the same interval-valued input. Point-in-time
    /// samples requiring forward-held input semantics need a separate integrator.
    ///
    /// Returns false and leaves state unchanged for non-finite Φ values or
    /// duplicate/backwards timestamps. The first valid observation initializes
    /// the EMA; later observations use alpha = 1 - exp(-dt / tau).
    fn can_observe_at(&self, phi: f64, elapsed: Duration) -> bool {
        if !phi.is_finite() {
            return false;
        }

        let Some((previous_time, previous_phi)) = self.history.back() else {
            return true;
        };
        if elapsed <= *previous_time {
            return false;
        }

        let dt = elapsed.saturating_sub(*previous_time).as_secs_f64();
        if !self.candidate_ema_is_finite(phi, dt) {
            return false;
        }

        // Reject inputs whose derived rates cannot be represented as finite f64.
        // Respect retention capacity: no velocity is measured with one retained slot.
        if self.capacity >= 2 && finite_rate(phi, *previous_phi, dt).is_none() {
            return false;
        }

        // Acceleration requires two velocity intervals and therefore at least three
        // samples in the configured retention window.
        if self.capacity >= 3 && self.history.len() >= 2 {
            let (previous_previous_time, previous_previous_phi) =
                self.history[self.history.len() - 2];
            let previous_dt = previous_time
                .saturating_sub(previous_previous_time)
                .as_secs_f64();
            let Some(previous_velocity) =
                finite_rate(*previous_phi, previous_previous_phi, previous_dt)
            else {
                return false;
            };
            let Some(current_velocity) = finite_rate(phi, *previous_phi, dt) else {
                return false;
            };
            let midpoint_delta = (previous_dt + dt) / 2.0;
            if finite_rate(current_velocity, previous_velocity, midpoint_delta).is_none() {
                return false;
            }
        }

        true
    }

    fn candidate_ema_is_finite(&self, phi: f64, dt: f64) -> bool {
        if !phi.is_finite() || !dt.is_finite() || dt <= 0.0 {
            return false;
        }
        let decay = (-dt / self.ema_time_constant_secs).exp();
        elapsed_ema(phi, self.ema, decay).is_some()
    }

    pub fn observe_at(&mut self, phi: f64, elapsed: Duration) -> bool {
        if !self.can_observe_at(phi, elapsed) {
            return false;
        }
        if let Some((previous_time, _)) = self.history.back() {
            let dt = elapsed.saturating_sub(*previous_time).as_secs_f64();
            let decay = (-dt / self.ema_time_constant_secs).exp();
            let Some(next_ema) = elapsed_ema(phi, self.ema, decay) else {
                return false;
            };
            self.ema = next_ema;
        } else {
            self.ema = phi;
        }

        self.history.push_back((elapsed, phi));
        if self.history.len() > self.capacity {
            self.history.pop_front();
        }

        self.compute_dynamics();
        true
    }

    fn compute_dynamics(&mut self) {
        if self.history.len() < 2 {
            return;
        }

        // Velocity is a finite difference over the latest observed interval.
        let len = self.history.len();
        let (t1, phi1) = self.history[len - 2];
        let (t2, phi2) = self.history[len - 1];

        let dt = t2.saturating_sub(t1).as_secs_f64();
        let Some(new_velocity) = finite_rate(phi2, phi1, dt) else {
            self.velocity = 0.0;
            self.acceleration = 0.0;
            self.acceleration_available = false;
            return;
        };
        self.velocity = new_velocity;

        // Acceleration requires two velocity estimates, hence three observations.
        // Each velocity belongs at its interval midpoint. With irregular sampling,
        // adjacent velocity midpoints are separated by (dt_previous + dt) / 2,
        // not simply by the most recent interval length.
        if len >= 3 {
            let (t0, phi0) = self.history[len - 3];
            let previous_dt = t1.saturating_sub(t0).as_secs_f64();
            let midpoint_delta = (previous_dt + dt) / 2.0;

            if previous_dt > 0.0 && midpoint_delta > 0.0 {
                if let Some(previous_velocity) = finite_rate(phi1, phi0, previous_dt) {
                    if let Some(acceleration) =
                        finite_rate(new_velocity, previous_velocity, midpoint_delta)
                    {
                        self.acceleration = acceleration;
                        self.acceleration_available = true;
                    } else {
                        self.acceleration = 0.0;
                        self.acceleration_available = false;
                    }
                } else {
                    self.acceleration = 0.0;
                    self.acceleration_available = false;
                }
            } else {
                self.acceleration = 0.0;
                self.acceleration_available = false;
            }
        } else {
            // A default zero must not be presented as a measured acceleration.
            self.acceleration = 0.0;
            self.acceleration_available = false;
        }

        // Compute stability (inverse variance)
        if self.history.len() >= 3 {
            // Deliberately sample-weighted variance. It describes the retained
            // sample set, not a time-weighted variance; compare it across runs only
            // when sampling cadence and retained time coverage are comparable.
            self.stability = finite_history_stability(&self.history);
        }

        // Determine trend
        if self.velocity > 0.05 {
            self.trend = 1;
        } else if self.velocity < -0.05 {
            self.trend = -1;
        } else {
            self.trend = 0;
        }
    }

    /// Get all historical Φ values
    pub fn history(&self) -> Vec<f64> {
        self.history.iter().map(|(_, p)| *p).collect()
    }

    /// Number of retained observations.
    pub fn sample_count(&self) -> usize {
        self.history.len()
    }

    /// Maximum number of retained observations.
    pub fn sample_capacity(&self) -> usize {
        self.capacity
    }

    /// Whether enough retained evidence exists to qualify the trajectory.
    ///
    /// The effective minimum is at least three samples because trajectory health
    /// depends on stability, which is not measured from a one- or two-sample history.
    /// A minimum larger than retention capacity is impossible to satisfy.
    pub fn has_minimum_samples(&self, minimum: usize) -> bool {
        let minimum = minimum.max(3);
        self.capacity >= minimum && self.history.len() >= minimum
    }

    /// Check the measured trajectory's signal-level health.
    ///
    /// This is not an evidence-sufficiency check; use `has_minimum_samples` too.
    pub fn is_healthy(&self) -> bool {
        self.stability > 0.5 && self.ema > 0.3 && self.trend >= 0
    }
}

// ============================================================================
// CONSCIOUSNESS CONTINUITY
// ============================================================================

/// Which elapsed-time domain governs continuity-event expiry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContinuityTimeBasis {
    HostMonotonic,
    ReplayElapsed,
}

/// Measures the "smoothness" of consciousness stream
#[derive(Debug, Clone)]
pub struct ConsciousnessContinuity {
    /// Recent state representations for continuity checking
    recent_states: VecDeque<BinaryHV>,

    /// Continuity score (0-1, higher = more continuous)
    pub score: f64,

    /// Detected discontinuities
    pub discontinuities: Vec<DiscontinuityEvent>,

    /// Time basis associated with each retained event. Some means elapsed replay
    /// time; None means only the host monotonic timestamp is available.
    /// Kept parallel to discontinuities to preserve the public event shape.
    discontinuity_elapsed_offsets: Vec<Option<Duration>>,

    /// Threshold for discontinuity detection
    threshold: f64,

    /// Clock basis is fixed by the first accepted observation; the domains
    /// cannot be mixed because their coordinates are not comparable.
    time_basis: Option<ContinuityTimeBasis>,

    /// Most recent accepted replay offset. Host-clock observations leave this None.
    last_elapsed_observation: Option<Duration>,

    /// Maximum states to track
    capacity: usize,
}

/// A detected discontinuity in consciousness
#[derive(Debug, Clone)]
pub struct DiscontinuityEvent {
    /// When the discontinuity occurred
    pub timestamp: Instant,

    /// Magnitude of the discontinuity (0-1)
    pub magnitude: f64,

    /// Type of discontinuity
    pub kind: DiscontinuityKind,

    /// Description
    pub description: String,
}

/// Types of consciousness discontinuities
#[derive(Debug, Clone, PartialEq)]
pub enum DiscontinuityKind {
    /// Sudden change in content
    ContentShift,
    /// Φ dropped suddenly
    PhiDrop,
    /// Identity coherence lost
    IdentityGap,
    /// Attention completely shifted
    AttentionBreak,
    /// Temporal flow interrupted
    TemporalGap,
}

impl ConsciousnessContinuity {
    pub fn new(capacity: usize, threshold: f64) -> Self {
        Self {
            recent_states: VecDeque::with_capacity(capacity),
            score: 1.0,
            discontinuities: Vec::new(),
            discontinuity_elapsed_offsets: Vec::new(),
            threshold,
            time_basis: None,
            last_elapsed_observation: None,
            capacity,
        }
    }

    /// Add a new consciousness state using the host monotonic clock for expiry.
    ///
    /// Compatibility entry point that discards the admission result. A continuity
    /// instance already operating on replay offsets rejects host-clock observations.
    /// Use try_observe when the caller needs to know whether the sample was admitted.
    pub fn observe(&mut self, state: &BinaryHV, phi: f64, phi_prev: f64) {
        let _ = self.try_observe(state, phi, phi_prev);
    }

    /// Add a host-clock observation and report admission.
    ///
    /// Returns false without mutation for non-finite Φ values or when this instance
    /// is already bound to replay-elapsed time.
    pub fn try_observe(&mut self, state: &BinaryHV, phi: f64, phi_prev: f64) -> bool {
        self.observe_with_elapsed(state, phi, phi_prev, None)
    }

    /// Add a state in a deterministic replay timeline.
    ///
    /// Event expiry follows this elapsed offset, not the speed at which the replay
    /// happens to execute on the host. Use one fixed origin for every observation.
    /// Returns false without mutation for non-finite Φ values, duplicate/backwards
    /// offsets, or attempts to mix host-clock and replay-clock observations.
    pub fn observe_at(
        &mut self,
        state: &BinaryHV,
        phi: f64,
        phi_prev: f64,
        elapsed: Duration,
    ) -> bool {
        self.observe_with_elapsed(state, phi, phi_prev, Some(elapsed))
    }

    fn can_accept_observation(&self, phi: f64, phi_prev: f64, elapsed: Option<Duration>) -> bool {
        let requested_basis = if elapsed.is_some() {
            ContinuityTimeBasis::ReplayElapsed
        } else {
            ContinuityTimeBasis::HostMonotonic
        };
        !self
            .time_basis
            .is_some_and(|accepted_basis| accepted_basis != requested_basis)
            && phi.is_finite()
            && phi_prev.is_finite()
            && !elapsed.is_some_and(|current| {
                self.last_elapsed_observation
                    .is_some_and(|previous| current <= previous)
            })
    }

    fn observe_with_elapsed(
        &mut self,
        state: &BinaryHV,
        phi: f64,
        phi_prev: f64,
        elapsed: Option<Duration>,
    ) -> bool {
        if !self.can_accept_observation(phi, phi_prev, elapsed) {
            return false;
        }
        let requested_basis = if elapsed.is_some() {
            ContinuityTimeBasis::ReplayElapsed
        } else {
            ContinuityTimeBasis::HostMonotonic
        };

        // The event list is public for compatibility; if an external caller has
        // mutated it directly, restore safe alignment and treat those entries as
        // host-timestamp-only diagnostics.
        if self.discontinuity_elapsed_offsets.len() != self.discontinuities.len() {
            self.discontinuity_elapsed_offsets = vec![None; self.discontinuities.len()];
        }
        let host_timestamp = Instant::now();

        // Check for discontinuity
        if let Some(prev_state) = self.recent_states.back() {
            let similarity = state.similarity(prev_state) as f64;

            // Detect content discontinuity
            if similarity < self.threshold {
                self.discontinuities.push(DiscontinuityEvent {
                    timestamp: host_timestamp,
                    magnitude: 1.0 - similarity,
                    kind: DiscontinuityKind::ContentShift,
                    description: format!("Content similarity dropped to {similarity:.3}"),
                });
                self.discontinuity_elapsed_offsets.push(elapsed);
            }

            // Detect Φ drop
            let phi_change = phi - phi_prev;
            if phi_change < -0.2 {
                self.discontinuities.push(DiscontinuityEvent {
                    timestamp: host_timestamp,
                    magnitude: -phi_change,
                    kind: DiscontinuityKind::PhiDrop,
                    description: format!("Φ dropped by {:.3}", -phi_change),
                });
                self.discontinuity_elapsed_offsets.push(elapsed);
            }
        }

        // Update state history
        self.recent_states.push_back(*state);
        if self.recent_states.len() > self.capacity {
            self.recent_states.pop_front();
        }

        // Compute continuity score
        self.compute_continuity_score();

        // Events must be aged in the same time domain in which they were created.
        // Host Instants remain diagnostic metadata; deterministic qualification uses
        // replay offsets so host execution speed cannot change expiry.
        let host_cutoff = if elapsed.is_none() {
            Some(Instant::now() - Duration::from_secs(60))
        } else {
            None
        };
        let mut retained_events = Vec::with_capacity(self.discontinuities.len());
        let mut retained_offsets = Vec::with_capacity(self.discontinuity_elapsed_offsets.len());
        for (event, event_elapsed) in self
            .discontinuities
            .drain(..)
            .zip(self.discontinuity_elapsed_offsets.drain(..))
        {
            let keep = match (elapsed, event_elapsed, host_cutoff) {
                (Some(now), Some(then), _) => {
                    then <= now && now.saturating_sub(then) <= Duration::from_secs(60)
                }
                (None, None, Some(cutoff)) => event.timestamp > cutoff,
                // Do not compare elapsed offsets with host Instants or vice versa.
                _ => false,
            };
            if keep {
                retained_events.push(event);
                retained_offsets.push(event_elapsed);
            }
        }
        self.discontinuities = retained_events;
        self.discontinuity_elapsed_offsets = retained_offsets;

        if self.discontinuities.len() > 1000 {
            let drain_count = self.discontinuities.len() - 1000;
            self.discontinuities.drain(..drain_count);
            self.discontinuity_elapsed_offsets.drain(..drain_count);
        }

        self.time_basis = Some(requested_basis);
        self.last_elapsed_observation = elapsed;
        true
    }

    fn compute_continuity_score(&mut self) {
        if self.recent_states.len() < 2 {
            self.score = 1.0;
            return;
        }

        // Average similarity between consecutive states
        let mut total_similarity = 0.0;
        let mut count = 0;

        for i in 1..self.recent_states.len() {
            let sim = self.recent_states[i].similarity(&self.recent_states[i - 1]) as f64;
            total_similarity += sim;
            count += 1;
        }

        if count > 0 {
            self.score = total_similarity / count as f64;
        }
    }

    /// Check if consciousness continuity is measured and meets its threshold.
    ///
    /// A single state is not evidence of continuity; require at least one
    /// pairwise transition before returning true.
    pub fn is_continuous(&self) -> bool {
        self.recent_states.len() >= 2 && self.score > 0.7 && self.discontinuities.is_empty()
    }
}

// ============================================================================
// TEMPORAL IDENTITY COHERENCE
// ============================================================================

/// Measures alignment between past, present, and future selves
#[derive(Debug, Clone)]
pub struct TemporalIdentityCoherence {
    /// Past self representation (from narrative)
    past_self: Option<BinaryHV>,

    /// Present self representation (from current state)
    present_self: Option<BinaryHV>,

    /// Future self representation (from predictions)
    future_self: Option<BinaryHV>,

    /// Overall coherence score
    pub coherence: f64,

    /// Past-present alignment; meaningful only when its availability flag is true.
    pub past_present_alignment: f64,
    pub past_present_alignment_available: bool,

    /// Present-future alignment; unknown until a projected future representation exists.
    pub present_future_alignment: f64,
    pub present_future_alignment_available: bool,

    /// Past-future alignment (long-term stability); requires both representations.
    pub past_future_alignment: f64,
    pub past_future_alignment_available: bool,

    /// Configuration weights
    config: TemporalConsciousnessConfig,
}

impl TemporalIdentityCoherence {
    pub fn new(config: TemporalConsciousnessConfig) -> Self {
        Self {
            past_self: None,
            present_self: None,
            future_self: None,
            coherence: 0.0,
            past_present_alignment: 0.0,
            past_present_alignment_available: false,
            present_future_alignment: 0.0,
            present_future_alignment_available: false,
            past_future_alignment: 0.0,
            past_future_alignment_available: false,
            config,
        }
    }

    /// Update with the historical and present representations currently available.
    ///
    /// The predictor currently forecasts scalar Φ/coherence values, not an
    /// independently projected HDC vector. A copy of the present vector is not
    /// evidence of future alignment, so future alignment remains unavailable
    /// until a genuine projected representation is supplied.
    pub(crate) fn update(
        &mut self,
        narrative: Option<&NarrativeSelfModel>,
        present_state: &BinaryHV,
    ) {
        // Always refresh current evidence. Missing context clears stale values.
        self.past_self = narrative.map(|model| *model.unified_self());
        self.present_self = Some(*present_state);

        // Scalar forecasts do not provide a future HDC representation.
        self.future_self = None;
        self.compute_coherence();
    }

    /// Update with explicit representations
    pub fn update_explicit(&mut self, past: &BinaryHV, present: &BinaryHV, future: &BinaryHV) {
        self.past_self = Some(*past);
        self.present_self = Some(*present);
        self.future_self = Some(*future);
        self.compute_coherence();
    }

    fn compute_coherence(&mut self) {
        // Reset to explicit unknowns before recomputing: absence must never reuse a
        // stale score from an earlier update or inherit a perfect initialization.
        self.past_present_alignment = 0.0;
        self.past_present_alignment_available = false;
        self.present_future_alignment = 0.0;
        self.present_future_alignment_available = false;
        self.past_future_alignment = 0.0;
        self.past_future_alignment_available = false;

        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        if let (Some(past), Some(present)) = (&self.past_self, &self.present_self) {
            self.past_present_alignment = past.similarity(present) as f64;
            self.past_present_alignment_available = true;
            let weight = self.config.past_weight;
            if weight.is_finite() && weight > 0.0 {
                weighted_sum += weight * self.past_present_alignment;
                total_weight += weight;
            }
        }

        if let (Some(present), Some(future)) = (&self.present_self, &self.future_self) {
            self.present_future_alignment = present.similarity(future) as f64;
            self.present_future_alignment_available = true;
            let weight = self.config.present_weight;
            if weight.is_finite() && weight > 0.0 {
                weighted_sum += weight * self.present_future_alignment;
                total_weight += weight;
            }
        }

        if let (Some(past), Some(future)) = (&self.past_self, &self.future_self) {
            self.past_future_alignment = past.similarity(future) as f64;
            self.past_future_alignment_available = true;
            let weight = self.config.future_weight;
            if weight.is_finite() && weight > 0.0 {
                weighted_sum += weight * self.past_future_alignment;
                total_weight += weight;
            }
        }

        // Score only measured relationships, and expose coverage separately.
        self.coherence = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        };
    }

    /// Number of the three pairwise alignments supported by actual representations.
    pub fn alignment_coverage(&self) -> usize {
        self.past_present_alignment_available as usize
            + self.present_future_alignment_available as usize
            + self.past_future_alignment_available as usize
    }

    /// Whether any available pairwise alignment is at or below the warning threshold.
    ///
    /// Unknown relationships are not treated as low alignment.
    fn has_low_measured_alignment(&self) -> bool {
        (self.past_present_alignment_available && self.past_present_alignment <= 0.5)
            || (self.present_future_alignment_available && self.present_future_alignment <= 0.5)
            || (self.past_future_alignment_available && self.past_future_alignment <= 0.5)
    }

    /// Check if identity is temporally coherent.
    ///
    /// Missing past/present/future evidence is not interpreted as perfect alignment.
    pub fn is_coherent(&self) -> bool {
        self.past_present_alignment_available
            && self.present_future_alignment_available
            && self.past_future_alignment_available
            && self.coherence > 0.6
            && self.past_present_alignment > 0.5
            && self.present_future_alignment > 0.5
            && self.past_future_alignment > 0.5
    }
}

// ============================================================================
// HUSSERLIAN TEMPORAL ANALYSIS
// ============================================================================

/// Husserl's tripartite temporal consciousness structure
#[derive(Debug, Clone)]
pub struct HusserlianAnalysis {
    /// Retention: Traces of just-past experiences
    pub retention: VecDeque<(Instant, BinaryHV, f64)>,

    /// Primal Impression: The absolute now-point
    pub primal_impression: Option<(BinaryHV, f64)>,

    /// Protention: Anticipations of what's coming
    pub protention: VecDeque<(BinaryHV, f64)>,

    /// Retention depth (how much past is retained)
    pub retention_depth: usize,

    /// Protention depth (how far into future is anticipated)
    pub protention_depth: usize,

    /// Configured retained-state limit. The depth is a count, not elapsed time.
    retention_capacity: usize,

    /// Configured prediction limit.
    protention_capacity: usize,

    /// Temporal horizon width (retention + protention span)
    pub temporal_horizon: f64,
}

impl HusserlianAnalysis {
    pub fn new(retention_capacity: usize, protention_capacity: usize) -> Self {
        Self {
            retention: VecDeque::with_capacity(retention_capacity),
            primal_impression: None,
            protention: VecDeque::with_capacity(protention_capacity),
            retention_depth: 0,
            protention_depth: 0,
            retention_capacity,
            protention_capacity,
            temporal_horizon: 0.0,
        }
    }

    /// Update with new primal impression (what's happening NOW)
    pub fn update_primal(&mut self, state: BinaryHV, phi: f64) {
        // Move current primal to retention
        if let Some((old_state, old_phi)) = self.primal_impression.take() {
            if self.retention_capacity > 0 {
                self.retention
                    .push_back((Instant::now(), old_state, old_phi));
                while self.retention.len() > self.retention_capacity {
                    self.retention.pop_front();
                }
            }
        }

        self.primal_impression = Some((state, phi));
        self.retention_depth = self.retention.len();
        self.update_horizon();
    }

    /// Update protentions (what we anticipate)
    pub fn update_protentions(&mut self, predictions: Vec<(BinaryHV, f64)>) {
        self.protention.clear();
        self.protention
            .extend(predictions.into_iter().take(self.protention_capacity));
        self.protention_depth = self.protention.len();
        self.update_horizon();
    }

    fn update_horizon(&mut self) {
        self.temporal_horizon = (self.retention_depth + self.protention_depth) as f64;
    }

    /// Whether at least one temporal relationship is available to score.
    ///
    /// A lone primal impression is a point observation, not evidence of temporal
    /// coherence. Measurements require at least one relation between retained,
    /// present, or predicted states.
    pub fn coherence_available(&self) -> bool {
        self.retention.len() >= 2
            || (self.retention.back().is_some() && self.primal_impression.is_some())
            || (self.primal_impression.is_some() && !self.protention.is_empty())
    }

    /// Get the temporal experience coherence.
    ///
    /// Returns zero as a neutral numeric fallback when no relationships exist;
    /// callers must check coherence_available before interpreting that number.
    pub fn coherence(&self) -> f64 {
        let mut total_sim = 0.0;
        let mut count = 0;

        // Check retention coherence
        if self.retention.len() >= 2 {
            for i in 1..self.retention.len() {
                let sim = self.retention[i].1.similarity(&self.retention[i - 1].1) as f64;
                total_sim += sim;
                count += 1;
            }
        }

        // Check retention-primal coherence
        if let (Some(last_ret), Some((primal, _))) =
            (self.retention.back(), &self.primal_impression)
        {
            let sim = primal.similarity(&last_ret.1) as f64;
            total_sim += sim;
            count += 1;
        }

        // Check primal-protention coherence
        if let (Some((primal, _)), Some(first_prot)) =
            (&self.primal_impression, self.protention.front())
        {
            let sim = primal.similarity(&first_prot.0) as f64;
            total_sim += sim;
            count += 1;
        }

        if count > 0 {
            total_sim / count as f64
        } else {
            0.0
        }
    }
}

// ============================================================================
// TEMPORAL BINDING ANALYSIS
// ============================================================================

/// Analysis of temporal binding between consciousness components
#[derive(Debug, Clone)]
pub struct TemporalBindingAnalysis {
    /// Binding relations using Allen algebra
    bindings: Vec<TemporalBinding>,

    /// Overall binding strength
    pub binding_strength: f64,

    /// Gamma-band synchronization estimate
    pub gamma_sync: f64,
}

/// A temporal binding between two consciousness components
#[derive(Debug, Clone)]
pub struct TemporalBinding {
    /// First component
    pub component_a: String,

    /// Second component
    pub component_b: String,

    /// Allen relation between them
    pub relation: AllenRelation,

    /// Binding strength
    pub strength: f64,
}

impl Default for TemporalBindingAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalBindingAnalysis {
    pub fn new() -> Self {
        Self {
            bindings: Vec::new(),
            binding_strength: 1.0,
            gamma_sync: 0.8,
        }
    }

    /// Compatibility wrapper for adding a binding.
    ///
    /// Invalid strengths are rejected without mutation; use try_add_binding when
    /// the caller needs to know whether the binding was admitted.
    pub fn add_binding(&mut self, a: &str, b: &str, relation: AllenRelation, strength: f64) {
        let _ = self.try_add_binding(a, b, relation, strength);
    }

    /// Add a binding with a finite strength in the inclusive [0, 1] range.
    ///
    /// Returns false without mutation for NaN, infinity, or out-of-range strengths.
    /// The retained set remains capped at 256 entries.
    pub fn try_add_binding(
        &mut self,
        a: &str,
        b: &str,
        relation: AllenRelation,
        strength: f64,
    ) -> bool {
        if a.trim().is_empty()
            || b.trim().is_empty()
            || !strength.is_finite()
            || !(0.0..=1.0).contains(&strength)
        {
            return false;
        }
        if self.bindings.len() >= 256 {
            // Evict weakest binding
            if let Some(min_idx) = self
                .bindings
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.strength
                        .partial_cmp(&b.strength)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                self.bindings.swap_remove(min_idx);
            }
        }
        self.bindings.push(TemporalBinding {
            component_a: a.to_string(),
            component_b: b.to_string(),
            relation,
            strength,
        });
        self.update_overall_strength();
        true
    }

    fn update_overall_strength(&mut self) {
        if self.bindings.is_empty() {
            self.binding_strength = 1.0;
            return;
        }

        // Average binding strength
        let total: f64 = self.bindings.iter().map(|b| b.strength).sum();
        self.binding_strength = total / self.bindings.len() as f64;

        // Gamma sync correlates with binding
        self.gamma_sync = self.binding_strength * 0.8 + 0.2;
    }

    /// Check if measured bindings satisfy the unification thresholds.
    ///
    /// An empty binding set is insufficient evidence, not a unified system.
    pub fn is_unified(&self) -> bool {
        !self.bindings.is_empty() && self.binding_strength > 0.7 && self.gamma_sync > 0.6
    }
}

// ============================================================================
// MAIN TEMPORAL CONSCIOUSNESS ANALYZER
// ============================================================================

/// The main temporal consciousness analyzer
pub struct TemporalConsciousnessAnalyzer {
    /// Configuration
    pub config: TemporalConsciousnessConfig,

    /// Φ trajectory analysis
    pub phi_trajectory: PhiTrajectory,

    /// Consciousness continuity tracking
    pub continuity: ConsciousnessContinuity,

    /// Temporal identity coherence
    pub identity_coherence: TemporalIdentityCoherence,

    /// Husserlian temporal analysis
    pub husserlian: HusserlianAnalysis,

    /// Temporal binding analysis
    pub binding: TemporalBindingAnalysis,

    /// Temporal reasoner for Allen algebra
    _temporal_reasoner: TemporalReasoner,

    /// Statistics
    pub stats: TemporalConsciousnessStats,
}

/// Statistics for temporal consciousness
#[derive(Debug, Clone, Default)]
pub struct TemporalConsciousnessStats {
    /// Number of accepted observations used to update temporal metrics.
    pub observations: usize,

    /// Number of observations rejected before temporal metrics were updated.
    pub rejected_observations: usize,

    /// Discontinuities detected
    pub discontinuities_detected: usize,

    /// Accepted observations with at least one measured alignment at or below threshold.
    pub identity_warnings: usize,

    /// Accepted observations lacking one or more of the three pairwise identity alignments.
    pub identity_evidence_gaps: usize,

    /// Qualified observations whose measured Φ trajectory failed health thresholds.
    pub trajectory_anomalies: usize,

    /// Accepted observations recorded before the configured trajectory sample minimum.
    pub trajectory_evidence_gaps: usize,

    /// Mean normalized aggregate score over accepted observations with any measured evidence.
    pub avg_temporal_coherence: f64,

    /// Number of accepted observations included in avg_temporal_coherence.
    pub coherence_measurements: usize,

    /// Mean nominal evidence coverage across observations included in the average.
    pub avg_temporal_evidence_coverage: f64,
}

impl TemporalConsciousnessAnalyzer {
    /// Create a new analyzer with given configuration
    pub fn new(config: TemporalConsciousnessConfig) -> Self {
        // Create temporal primitives config from our config
        let temporal_primitives_config = TemporalPrimitivesConfig {
            binding_window_ms: config.binding_window_ms,
            ..TemporalPrimitivesConfig::default()
        };

        Self {
            phi_trajectory: PhiTrajectory::new_with_time_constant(
                config.phi_history_depth,
                config.phi_ema_time_constant_secs,
            ),
            continuity: ConsciousnessContinuity::new(20, config.continuity_threshold),
            identity_coherence: TemporalIdentityCoherence::new(config.clone()),
            husserlian: HusserlianAnalysis::new(10, 5),
            binding: TemporalBindingAnalysis::new(),
            _temporal_reasoner: TemporalReasoner::new(temporal_primitives_config),
            stats: TemporalConsciousnessStats::default(),
            config,
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(TemporalConsciousnessConfig::default())
    }

    /// Compatibility entry point: update temporal analyses for an accepted Φ sample.
    /// Rejected samples do not update metrics and are counted in rejection statistics.
    pub(crate) fn observe(
        &mut self,
        state: &BinaryHV,
        phi: f64,
        narrative: Option<&NarrativeSelfModel>,
        predictive: Option<&PredictiveSelfModel>,
    ) {
        let _ = self.try_observe(state, phi, narrative, predictive);
    }

    /// Update temporal analyses only after Φ is admitted by the trajectory.
    ///
    /// Rejected observations update only the rejection counter. All temporal
    /// metrics and accepted-observation counters remain unchanged.
    pub(crate) fn try_observe(
        &mut self,
        state: &BinaryHV,
        phi: f64,
        narrative: Option<&NarrativeSelfModel>,
        _predictive: Option<&PredictiveSelfModel>,
    ) -> bool {
        self.try_observe_with_elapsed(state, phi, None, narrative)
    }

    /// Deterministic observation entry point for fixed-clock replay and conformance tests.
    ///
    /// `elapsed` must be an offset from one fixed replay origin and use the same origin
    /// for every call; it is not a wall-clock timestamp. This variant has no external
    /// narrative or predictor context, so unavailable identity representations stay unknown.
    pub fn observe_at(&mut self, state: &BinaryHV, phi: f64, elapsed: Duration) -> bool {
        self.try_observe_with_elapsed(state, phi, Some(elapsed), None)
    }

    fn try_observe_with_elapsed(
        &mut self,
        state: &BinaryHV,
        phi: f64,
        elapsed: Option<Duration>,
        narrative: Option<&NarrativeSelfModel>,
    ) -> bool {
        let phi_prev = self.phi_trajectory.ema;

        // Preflight every time-sensitive component before mutating any of them.
        // These fields are public for compatibility, so callers can legitimately
        // pre-use continuity directly; an incompatible clock basis must not leave
        // Φ trajectory advanced while continuity, identity, and counters disagree.
        let trajectory_elapsed =
            elapsed.unwrap_or_else(|| self.phi_trajectory.started_at.elapsed());
        if !self.phi_trajectory.can_observe_at(phi, trajectory_elapsed)
            || !self
                .continuity
                .can_accept_observation(phi, phi_prev, elapsed)
        {
            self.stats.rejected_observations += 1;
            return false;
        }

        // Both preflight checks ran against the same exclusively borrowed state.
        // Admission should now succeed; keep the defensive rejection path anyway.
        if !self.phi_trajectory.observe_at(phi, trajectory_elapsed) {
            self.stats.rejected_observations += 1;
            return false;
        }

        // Update continuity in the same time domain as the admitted observation.
        match elapsed {
            Some(elapsed) => {
                let _ = self.continuity.observe_at(state, phi, phi_prev, elapsed);
            }
            None => self.continuity.observe(state, phi, phi_prev),
        }

        // Update Husserlian analysis
        if self.config.enable_husserlian_analysis {
            self.husserlian.update_primal(*state, phi);
        }

        // Refresh identity evidence on every accepted sample. The predictor does
        // not yet supply a projected HDC self-state, so future alignment stays unknown.
        self.identity_coherence.update(narrative, state);

        // Update stats
        self.stats.observations += 1;
        self.stats.discontinuities_detected = self.continuity.discontinuities.len();

        if self.identity_coherence.has_low_measured_alignment() {
            self.stats.identity_warnings += 1;
        }
        if self.identity_coherence.alignment_coverage() < 3 {
            self.stats.identity_evidence_gaps += 1;
        }

        if self
            .phi_trajectory
            .has_minimum_samples(self.config.min_samples_for_trajectory)
        {
            if !self.phi_trajectory.is_healthy() {
                self.stats.trajectory_anomalies += 1;
            }
        } else {
            self.stats.trajectory_evidence_gaps += 1;
        }

        // Only evidence-bearing aggregate scores contribute to running averages.
        // Early zero-coverage observations are missing measurements, not zero-valued
        // coherence outcomes. Track average coverage too, since the measured component
        // set can still vary as the analyzer accumulates evidence.
        let current_coverage = self.overall_evidence_coverage();
        if current_coverage > 0.0 {
            let current_coherence = self.overall_temporal_coherence();
            self.stats.coherence_measurements += 1;
            let n = self.stats.coherence_measurements as f64;
            self.stats.avg_temporal_coherence =
                (self.stats.avg_temporal_coherence * (n - 1.0) + current_coherence) / n;
            self.stats.avg_temporal_evidence_coverage =
                (self.stats.avg_temporal_evidence_coverage * (n - 1.0) + current_coverage) / n;
        }
        true
    }

    /// Add protentions (future predictions) to Husserlian analysis
    pub fn add_protentions(&mut self, predictions: Vec<(BinaryHV, f64)>) {
        self.husserlian.update_protentions(predictions);
    }

    /// Compute overall temporal coherence from metrics with sufficient evidence.
    ///
    /// The score is normalized over measured components only. Call
    /// `overall_evidence_coverage` alongside it: a high score over sparse evidence
    /// is not equivalent to a fully observed system.
    pub fn overall_temporal_coherence(&self) -> f64 {
        let (weighted_sum, measured_weight) = self.weighted_temporal_evidence();
        if measured_weight > 0.0 {
            weighted_sum / measured_weight
        } else {
            0.0
        }
    }

    /// Fraction of nominal overall-metric weight backed by sufficient evidence.
    /// The configured full score's nominal weight sums to 1.0.
    pub fn overall_evidence_coverage(&self) -> f64 {
        self.weighted_temporal_evidence().1.clamp(0.0, 1.0)
    }

    fn weighted_temporal_evidence(&self) -> (f64, f64) {
        let mut weighted_sum = 0.0;
        let mut measured_weight = 0.0;

        if self
            .phi_trajectory
            .has_minimum_samples(self.config.min_samples_for_trajectory)
        {
            let score = if self.phi_trajectory.is_healthy() {
                1.0
            } else {
                0.5
            };
            weighted_sum += 0.2 * score;
            measured_weight += 0.2;
        }

        if self.continuity.recent_states.len() >= 2 {
            weighted_sum += 0.2 * self.continuity.score;
            measured_weight += 0.2;
        }

        let identity_weight = 0.25 * self.identity_coherence.alignment_coverage() as f64 / 3.0;
        if identity_weight > 0.0 {
            weighted_sum += identity_weight * self.identity_coherence.coherence;
            measured_weight += identity_weight;
        }

        if self.husserlian.coherence_available() {
            weighted_sum += 0.2 * self.husserlian.coherence();
            measured_weight += 0.2;
        }

        if !self.binding.bindings.is_empty() {
            weighted_sum += 0.15 * self.binding.binding_strength;
            measured_weight += 0.15;
        }

        (weighted_sum, measured_weight)
    }

    /// Qualified temporal health, separating observed failures from missing evidence.
    pub fn health_status(&self) -> TemporalHealthStatus {
        let trajectory_measured = self
            .phi_trajectory
            .has_minimum_samples(self.config.min_samples_for_trajectory);
        let continuity_measured = self.continuity.recent_states.len() >= 2;
        let husserlian_measured = self.husserlian.coherence_available();
        let binding_measured = !self.binding.bindings.is_empty();

        let identity_fully_measured = self.identity_coherence.alignment_coverage() == 3;
        let measured_failure = (trajectory_measured && !self.phi_trajectory.is_healthy())
            || (continuity_measured && !self.continuity.is_continuous())
            || self.identity_coherence.has_low_measured_alignment()
            || (identity_fully_measured && !self.identity_coherence.is_coherent())
            || (husserlian_measured && self.husserlian.coherence() <= 0.6)
            || (binding_measured && !self.binding.is_unified());

        if measured_failure {
            return TemporalHealthStatus::Unhealthy;
        }

        let evidence_complete = trajectory_measured
            && continuity_measured
            && self.identity_coherence.alignment_coverage() == 3
            && (!self.config.enable_husserlian_analysis || husserlian_measured)
            && binding_measured;

        if evidence_complete {
            TemporalHealthStatus::Healthy
        } else {
            TemporalHealthStatus::InsufficientEvidence
        }
    }

    /// Compatibility boolean. A system with incomplete evidence is not qualified healthy.
    pub fn is_temporally_healthy(&self) -> bool {
        self.health_status() == TemporalHealthStatus::Healthy
    }

    /// Get detailed temporal report
    pub fn report(&self) -> TemporalConsciousnessReport {
        TemporalConsciousnessReport {
            overall_coherence: self.overall_temporal_coherence(),
            overall_evidence_coverage: self.overall_evidence_coverage(),
            trajectory_sample_count: self.phi_trajectory.sample_count(),
            trajectory_sample_capacity: self.phi_trajectory.sample_capacity(),
            trajectory_minimum_samples: self.config.min_samples_for_trajectory.max(3),
            trajectory_evidence_sufficient: self
                .phi_trajectory
                .has_minimum_samples(self.config.min_samples_for_trajectory),
            trajectory_qualification_possible: self.phi_trajectory.sample_capacity()
                >= self.config.min_samples_for_trajectory.max(3),
            health_status: self.health_status(),
            phi_velocity: self.phi_trajectory.velocity,
            phi_velocity_available: self.phi_trajectory.sample_count() >= 2,
            phi_acceleration: self.phi_trajectory.acceleration,
            phi_acceleration_available: self.phi_trajectory.acceleration_available,
            phi_stability: self.phi_trajectory.stability,
            phi_stability_available: self.phi_trajectory.sample_count() >= 3,
            phi_trend: self.phi_trajectory.trend,
            phi_trend_available: self.phi_trajectory.sample_count() >= 2,
            continuity_score: self.continuity.score,
            continuity_score_available: self.continuity.recent_states.len() >= 2,
            discontinuities: self.continuity.discontinuities.len(),
            identity_coherence: self.identity_coherence.coherence,
            identity_coherence_available: self.identity_coherence.alignment_coverage() > 0,
            identity_alignment_coverage: self.identity_coherence.alignment_coverage(),
            past_present_alignment: self.identity_coherence.past_present_alignment,
            past_present_alignment_available: self
                .identity_coherence
                .past_present_alignment_available,
            present_future_alignment: self.identity_coherence.present_future_alignment,
            present_future_alignment_available: self
                .identity_coherence
                .present_future_alignment_available,
            past_future_alignment: self.identity_coherence.past_future_alignment,
            past_future_alignment_available: self
                .identity_coherence
                .past_future_alignment_available,
            husserlian_coherence: self.husserlian.coherence(),
            husserlian_coherence_available: self.husserlian.coherence_available(),
            retention_depth: self.husserlian.retention_depth,
            protention_depth: self.husserlian.protention_depth,
            temporal_horizon: self.husserlian.temporal_horizon,
            binding_strength: self.binding.binding_strength,
            gamma_sync: self.binding.gamma_sync,
            binding_evidence_available: !self.binding.bindings.is_empty(),
            is_healthy: self.is_temporally_healthy(),
            recommendations: self.generate_recommendations(),
        }
    }

    fn generate_recommendations(&self) -> Vec<String> {
        let mut recs = Vec::new();

        let required_samples = self.config.min_samples_for_trajectory.max(3);
        if self.phi_trajectory.sample_capacity() < required_samples {
            recs.push(format!(
                "Trajectory cannot qualify: history capacity {} is below required sample count {}",
                self.phi_trajectory.sample_capacity(),
                required_samples,
            ));
        } else if !self.phi_trajectory.has_minimum_samples(required_samples) {
            recs.push(format!(
                "Trajectory evidence pending: {} of {} required samples",
                self.phi_trajectory.sample_count(),
                required_samples,
            ));
        } else if !self.phi_trajectory.is_healthy() {
            if self.phi_trajectory.trend < 0 {
                recs.push("Φ trajectory declining - consider attention restoration".to_string());
            }
            if self.phi_trajectory.stability < 0.5 {
                recs.push("Φ unstable - increase integration strength".to_string());
            }
        }

        if self.continuity.recent_states.len() < 2 {
            recs.push("Continuity evidence pending: observe at least two states".to_string());
        } else if !self.continuity.is_continuous() {
            recs.push(format!(
                "Consciousness continuity at {:.1}% - inspect state transitions and discontinuity events",
                self.continuity.score * 100.0
            ));
        }

        if !self.identity_coherence.is_coherent() {
            if !self.identity_coherence.past_present_alignment_available {
                recs.push(
                    "Past-present alignment unavailable - obtain both representations".to_string(),
                );
            } else if self.identity_coherence.past_present_alignment < 0.5 {
                recs.push("Past-present disconnect - reinforce narrative continuity".to_string());
            }
            if !self.identity_coherence.present_future_alignment_available {
                recs.push(
                    "Present-future alignment unavailable - no projected future representation"
                        .to_string(),
                );
            } else if self.identity_coherence.present_future_alignment < 0.5 {
                recs.push("Present-future disconnect - clarify intentions".to_string());
            }
            if !self.identity_coherence.past_future_alignment_available {
                recs.push("Past-future alignment unavailable - stability unknown".to_string());
            }
        }

        if self.husserlian.temporal_horizon < 5.0 {
            recs.push("Limited temporal horizon - expand retention and protention".to_string());
        }

        if self.binding.bindings.is_empty() {
            recs.push("Temporal binding evidence pending: no bindings recorded".to_string());
        } else if !self.binding.is_unified() {
            recs.push(
                "Weak temporal binding - strengthen cross-component synchronization".to_string(),
            );
        }

        if recs.is_empty() {
            recs.push(
                "Temporal consciousness is healthy - maintain current integration".to_string(),
            );
        }

        recs
    }
}

/// Evidence-qualified state of the temporal analyzer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalHealthStatus {
    /// All required evidence is present and measured metrics satisfy thresholds.
    Healthy,
    /// At least one measured metric violates its declared threshold.
    Unhealthy,
    /// No measured failure was found, but one or more required evidence sources are absent.
    InsufficientEvidence,
}

impl TemporalHealthStatus {
    fn label(self) -> &'static str {
        match self {
            Self::Healthy => "HEALTHY",
            Self::Unhealthy => "UNHEALTHY",
            Self::InsufficientEvidence => "INSUFFICIENT EVIDENCE",
        }
    }
}

/// Comprehensive temporal consciousness report
#[derive(Debug, Clone)]
pub struct TemporalConsciousnessReport {
    /// Overall temporal coherence across sufficiently measured components (0-1).
    pub overall_coherence: f64,
    /// Nominal metric weight backed by sufficient evidence (0-1).
    pub overall_evidence_coverage: f64,
    /// Retained Φ observations.
    pub trajectory_sample_count: usize,
    /// Maximum retained Φ observations.
    pub trajectory_sample_capacity: usize,
    /// Effective minimum Φ observations required, including the three-sample stability floor.
    pub trajectory_minimum_samples: usize,
    /// Whether the retained trajectory currently meets the sample minimum.
    pub trajectory_evidence_sufficient: bool,
    /// Whether the configured minimum can be met within the history capacity.
    pub trajectory_qualification_possible: bool,
    /// Three-way health qualification; missing evidence is distinct from failure.
    pub health_status: TemporalHealthStatus,

    // Φ Trajectory
    pub phi_velocity: f64,
    pub phi_velocity_available: bool,
    /// Acceleration value; see availability flag before interpreting the number.
    pub phi_acceleration: f64,
    /// False until two distinct velocity intervals can be compared.
    pub phi_acceleration_available: bool,
    pub phi_stability: f64,
    pub phi_stability_available: bool,
    pub phi_trend: i8,
    pub phi_trend_available: bool,

    // Continuity
    pub continuity_score: f64,
    pub continuity_score_available: bool,
    pub discontinuities: usize,

    // Identity
    pub identity_coherence: f64,
    pub identity_coherence_available: bool,
    /// Count of measured pairwise alignment relationships (0..=3).
    pub identity_alignment_coverage: usize,
    pub past_present_alignment: f64,
    pub past_present_alignment_available: bool,
    pub present_future_alignment: f64,
    pub present_future_alignment_available: bool,
    pub past_future_alignment: f64,
    pub past_future_alignment_available: bool,

    // Husserlian
    /// Numeric score; consult availability before interpreting it.
    pub husserlian_coherence: f64,
    /// Whether at least one retention/present/protention relation was measured.
    pub husserlian_coherence_available: bool,
    pub retention_depth: usize,
    pub protention_depth: usize,
    pub temporal_horizon: f64,

    // Binding
    pub binding_strength: f64,
    pub gamma_sync: f64,
    pub binding_evidence_available: bool,

    // Overall
    pub is_healthy: bool,
    pub recommendations: Vec<String>,
}

fn format_alignment(value: f64, available: bool) -> String {
    if available {
        format!("{:.1}%", value * 100.0)
    } else {
        "N/A".to_string()
    }
}

impl std::fmt::Display for TemporalConsciousnessReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            r#"
╔══════════════════════════════════════════════════════════════════════════╗
║     TEMPORAL CONSCIOUSNESS REPORT (#75) - Consciousness Through Time     ║
╠══════════════════════════════════════════════════════════════════════════╣
║ OVERALL TEMPORAL COHERENCE: {:.1}%                                       ║
║ STATUS: {:<24}                                                           ║
║ EVIDENCE COVERAGE:          {:.1}%                                       ║
║ Φ SAMPLES:                  {}/{} (minimum {}) {}                       ║
╠══════════════════════════════════════════════════════════════════════════╣
║ Φ TRAJECTORY                                                             ║
║   Velocity:     {:>7} Φ/s   (rate of change)                           ║
║   Acceleration: {:>7} Φ/s²  (change of change)                         ║
║   Stability:    {:>6}       (inverse variance)                          ║
║   Trend:        {}                                                       ║
╠══════════════════════════════════════════════════════════════════════════╣
║ CONSCIOUSNESS CONTINUITY                                                 ║
║   Continuity Score:  {:>6}                                              ║
║   Discontinuities:   {}                                                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║ TEMPORAL IDENTITY COHERENCE                                              ║
║   Known-pair score:  {:>6}                                              ║
║   Coverage:          {}/3 relationships                                  ║
║   Past↔Present:      {:>6}                                               ║
║   Present↔Future:    {:>6}                                               ║
║   Past↔Future:       {:>6}                                               ║
╠══════════════════════════════════════════════════════════════════════════╣
║ HUSSERLIAN ANALYSIS (Retention → Now → Protention)                       ║
║   Coherence:         {:>6}                                              ║
║   Retention Depth:   {} states                                           ║
║   Protention Depth:  {} predictions                                      ║
║   Temporal Horizon:  {:.1} units                                         ║
╠══════════════════════════════════════════════════════════════════════════╣
║ TEMPORAL BINDING                                                         ║
║   Binding Strength:  {:>6}                                              ║
║   Gamma Sync:        {:>6}                                              ║
╠══════════════════════════════════════════════════════════════════════════╣
║ RECOMMENDATIONS                                                          ║
{}║
╚══════════════════════════════════════════════════════════════════════════╝
"#,
            self.overall_coherence * 100.0,
            self.health_status.label(),
            self.overall_evidence_coverage * 100.0,
            self.trajectory_sample_count,
            self.trajectory_sample_capacity,
            self.trajectory_minimum_samples,
            if self.trajectory_evidence_sufficient {
                "QUALIFIED"
            } else if self.trajectory_qualification_possible {
                "PENDING"
            } else {
                "IMPOSSIBLE"
            },
            if self.phi_velocity_available {
                format!("{:+.4}", self.phi_velocity)
            } else {
                "N/A".to_string()
            },
            if self.phi_acceleration_available {
                format!("{:+.4}", self.phi_acceleration)
            } else {
                "N/A".to_string()
            },
            if self.phi_stability_available {
                format!("{:.4}", self.phi_stability)
            } else {
                "N/A".to_string()
            },
            if self.phi_trend_available {
                match self.phi_trend {
                    1 => "↑ Rising",
                    -1 => "↓ Declining",
                    _ => "→ Stable",
                }
            } else {
                "N/A"
            },
            if self.continuity_score_available {
                format!("{:.1}%", self.continuity_score * 100.0)
            } else {
                "N/A".to_string()
            },
            self.discontinuities,
            if self.identity_coherence_available {
                format!("{:.1}%", self.identity_coherence * 100.0)
            } else {
                "N/A".to_string()
            },
            self.identity_alignment_coverage,
            format_alignment(
                self.past_present_alignment,
                self.past_present_alignment_available,
            ),
            format_alignment(
                self.present_future_alignment,
                self.present_future_alignment_available,
            ),
            format_alignment(
                self.past_future_alignment,
                self.past_future_alignment_available,
            ),
            if self.husserlian_coherence_available {
                format!("{:.1}%", self.husserlian_coherence * 100.0)
            } else {
                "N/A".to_string()
            },
            self.retention_depth,
            self.protention_depth,
            self.temporal_horizon,
            if self.binding_evidence_available {
                format!("{:.1}%", self.binding_strength * 100.0)
            } else {
                "N/A".to_string()
            },
            if self.binding_evidence_available {
                format!("{:.1}%", self.gamma_sync * 100.0)
            } else {
                "N/A".to_string()
            },
            self.recommendations
                .iter()
                .map(|r| format!("║   • {r}\n"))
                .collect::<String>()
        )
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phi_ema_is_invariant_to_sampling_subdivision_for_constant_segments() {
        let mut coarse = PhiTrajectory::new_with_time_constant(16, 1.0);
        let mut fine = PhiTrajectory::new_with_time_constant(16, 1.0);

        assert!(coarse.observe_at(0.2, Duration::ZERO));
        assert!(coarse.observe_at(0.8, Duration::from_secs(1)));

        assert!(fine.observe_at(0.2, Duration::ZERO));
        // Each sample represents the interval ending at its timestamp.
        // The fine sequence therefore subdivides the same one-second 0.8 input
        // interval represented by the coarse endpoint sample.
        assert!(fine.observe_at(0.8, Duration::from_millis(250)));
        assert!(fine.observe_at(0.8, Duration::from_millis(500)));
        assert!(fine.observe_at(0.8, Duration::from_millis(750)));
        assert!(fine.observe_at(0.8, Duration::from_secs(1)));

        let expected = 0.8 + (0.2 - 0.8) * (-1.0_f64).exp();
        assert!(
            (coarse.ema - expected).abs() < 1e-12,
            "one time constant should apply the expected continuous-time decay"
        );
        assert!(
            (coarse.ema - fine.ema).abs() < 1e-12,
            "equivalent interval-valued input should be invariant to subdivision"
        );
    }

    #[test]
    fn phi_trajectory_rejects_duplicate_backwards_and_nonfinite_observations() {
        let mut trajectory = PhiTrajectory::new(8);
        assert!(trajectory.observe_at(0.4, Duration::from_secs(1)));
        assert!(!trajectory.observe_at(0.6, Duration::from_secs(1)));
        assert!(!trajectory.observe_at(0.6, Duration::from_millis(500)));
        assert!(!trajectory.observe_at(f64::NAN, Duration::from_secs(2)));
        assert_eq!(trajectory.history().len(), 1);
        assert_eq!(trajectory.ema, 0.4);
    }

    #[test]
    fn analyzer_rejects_nonfinite_phi_before_mutating_downstream_state() {
        let mut analyzer =
            TemporalConsciousnessAnalyzer::new(TemporalConsciousnessConfig::default());
        let state = BinaryHV::random(73);

        assert!(!analyzer.try_observe(&state, f64::NAN, None, None));
        assert_eq!(analyzer.stats.observations, 0);
        assert_eq!(analyzer.stats.rejected_observations, 1);
        assert!(analyzer.phi_trajectory.history().is_empty());
        assert!(analyzer.continuity.recent_states.is_empty());
        assert!(analyzer.husserlian.primal_impression.is_none());
    }

    #[test]
    fn missing_future_projection_is_unknown_not_perfect_alignment() {
        let mut coherence = TemporalIdentityCoherence::new(TemporalConsciousnessConfig::default());
        coherence.past_self = Some(BinaryHV::random(41));
        coherence.present_self = Some(BinaryHV::random(42));
        coherence.future_self = None;

        coherence.compute_coherence();

        assert_eq!(coherence.alignment_coverage(), 1);
        assert!(coherence.past_present_alignment_available);
        assert!(!coherence.present_future_alignment_available);
        assert!(!coherence.past_future_alignment_available);
        assert_eq!(coherence.present_future_alignment, 0.0);
        assert_eq!(coherence.past_future_alignment, 0.0);
        assert!(!coherence.is_coherent());

        // A later sample without historical context must clear a previously
        // available past representation rather than preserving stale alignment.
        coherence.future_self = Some(BinaryHV::random(44));
        coherence.update(None, &BinaryHV::random(43));
        assert!(coherence.past_self.is_none());
        assert!(coherence.future_self.is_none());
        assert_eq!(coherence.alignment_coverage(), 0);
        assert!(!coherence.past_present_alignment_available);
    }

    #[test]
    fn missing_identity_evidence_is_not_counted_as_a_low_alignment_warning() {
        let mut analyzer =
            TemporalConsciousnessAnalyzer::new(TemporalConsciousnessConfig::default());
        assert!(analyzer.observe_at(&BinaryHV::random(82), 0.4, Duration::ZERO,));

        assert_eq!(analyzer.stats.identity_warnings, 0);
        assert_eq!(analyzer.stats.identity_evidence_gaps, 1);
    }

    #[test]
    fn temporal_report_labels_unavailable_identity_alignments_as_na() {
        let analyzer = TemporalConsciousnessAnalyzer::new(TemporalConsciousnessConfig::default());
        let report = analyzer.report();

        assert_eq!(report.identity_alignment_coverage, 0);
        assert!(!report.identity_coherence_available);
        assert!(!report.phi_velocity_available);
        assert!(!report.phi_stability_available);
        assert!(!report.phi_trend_available);
        assert!(!report.continuity_score_available);
        assert!(!report.binding_evidence_available);
        assert!(!report.present_future_alignment_available);
        let rendered = report.to_string();
        assert!(rendered.contains("Present↔Future:       N/A"));
        assert!(rendered.contains("Past↔Future:          N/A"));
        assert!(
            rendered
                .lines()
                .any(|line| line.contains("Velocity:") && line.contains("N/A"))
        );
        for label in [
            "Continuity Score:",
            "Known-pair score:",
            "Binding Strength:",
            "Gamma Sync:",
            "Stability:",
            "Trend:",
        ] {
            let line = rendered
                .lines()
                .find(|line| line.contains(label))
                .unwrap_or_else(|| panic!("missing report row {label}"));
            assert!(
                line.contains("N/A"),
                "expected unavailable value in {label}: {line}"
            );
        }
    }

    #[test]
    fn analyzer_rejects_duplicate_and_backwards_times_without_downstream_mutation() {
        let mut analyzer =
            TemporalConsciousnessAnalyzer::new(TemporalConsciousnessConfig::default());
        let first_state = BinaryHV::random(70);
        let rejected_state = BinaryHV::random(71);
        let first_time = Duration::from_secs(1);

        assert!(analyzer.observe_at(&first_state, 0.4, first_time));
        assert_eq!(analyzer.stats.observations, 1);
        assert_eq!(analyzer.continuity.recent_states.len(), 1);

        assert!(!analyzer.observe_at(&rejected_state, 0.9, first_time));
        assert!(!analyzer.observe_at(&rejected_state, 0.9, Duration::from_millis(500),));

        assert_eq!(analyzer.stats.observations, 1);
        assert_eq!(analyzer.stats.rejected_observations, 2);
        assert_eq!(analyzer.phi_trajectory.history().len(), 1);
        assert_eq!(analyzer.continuity.recent_states.len(), 1);
        assert_eq!(analyzer.husserlian.retention_depth, 0);
        assert!(analyzer.husserlian.primal_impression.is_some());
    }

    #[test]
    fn trajectory_qualification_requires_configured_samples_and_reports_coverage() {
        let mut config = TemporalConsciousnessConfig::default();
        config.min_samples_for_trajectory = 3;
        config.phi_history_depth = 4;
        let mut analyzer = TemporalConsciousnessAnalyzer::new(config);

        let initial = analyzer.report();
        assert_eq!(initial.trajectory_sample_count, 0);
        assert!(!initial.trajectory_evidence_sufficient);
        assert!(initial.trajectory_qualification_possible);
        assert_eq!(initial.overall_evidence_coverage, 0.0);
        assert_eq!(initial.overall_coherence, 0.0);
        assert!(!initial.is_healthy);
        assert_eq!(
            initial.health_status,
            TemporalHealthStatus::InsufficientEvidence
        );

        assert!(analyzer.observe_at(&BinaryHV::random(91), 0.4, Duration::ZERO));
        assert!(analyzer.observe_at(&BinaryHV::random(92), 0.5, Duration::from_secs(1),));
        let pending = analyzer.report();
        assert_eq!(pending.trajectory_sample_count, 2);
        assert!(!pending.trajectory_evidence_sufficient);
        assert_eq!(analyzer.stats.trajectory_evidence_gaps, 2);
        assert_eq!(analyzer.stats.trajectory_anomalies, 0);

        assert!(analyzer.observe_at(&BinaryHV::random(93), 0.6, Duration::from_secs(2),));
        let qualified = analyzer.report();
        assert_eq!(qualified.trajectory_sample_count, 3);
        assert!(qualified.trajectory_evidence_sufficient);
        assert!(qualified.overall_evidence_coverage > pending.overall_evidence_coverage);
    }

    #[test]
    fn analyzer_rejects_incompatible_public_continuity_clock_without_partial_acceptance() {
        let mut analyzer =
            TemporalConsciousnessAnalyzer::new(TemporalConsciousnessConfig::default());

        // The analyzer's fields are public; a caller can use the continuity tracker
        // directly before using the analyzer's deterministic replay API.
        assert!(
            analyzer
                .continuity
                .try_observe(&BinaryHV::random(4301), 0.5, 0.5)
        );
        let continuity_states = analyzer.continuity.recent_states.len();
        let husserlian_retention = analyzer.husserlian.retention.len();

        assert!(!analyzer.observe_at(&BinaryHV::random(4302), 0.6, Duration::ZERO,));
        assert_eq!(analyzer.phi_trajectory.sample_count(), 0);
        assert_eq!(analyzer.continuity.recent_states.len(), continuity_states);
        assert_eq!(analyzer.husserlian.retention.len(), husserlian_retention);
        assert_eq!(analyzer.stats.observations, 0);
        assert_eq!(analyzer.stats.rejected_observations, 1);
    }

    #[test]
    fn analyzer_rejects_overflowing_derived_phi_sample_without_partial_acceptance() {
        let mut analyzer =
            TemporalConsciousnessAnalyzer::new(TemporalConsciousnessConfig::default());
        assert!(analyzer.observe_at(&BinaryHV::random(4311), f64::MAX, Duration::ZERO));

        let continuity_states = analyzer.continuity.recent_states.len();
        let husserlian_retention = analyzer.husserlian.retention.len();
        assert!(!analyzer.observe_at(&BinaryHV::random(4312), -f64::MAX, Duration::from_secs(1),));

        assert_eq!(analyzer.phi_trajectory.sample_count(), 1);
        assert_eq!(analyzer.continuity.recent_states.len(), continuity_states);
        assert_eq!(analyzer.husserlian.retention.len(), husserlian_retention);
        assert_eq!(analyzer.stats.observations, 1);
        assert_eq!(analyzer.stats.rejected_observations, 1);
        let report = analyzer.report();
        assert!(report.phi_velocity.is_finite());
        assert!(report.phi_acceleration.is_finite());
        assert!(report.overall_coherence.is_finite());
        assert!(report.overall_evidence_coverage.is_finite());
    }

    #[test]
    fn temporal_average_ignores_observations_with_zero_evidence_coverage() {
        let mut analyzer =
            TemporalConsciousnessAnalyzer::new(TemporalConsciousnessConfig::default());

        assert!(analyzer.observe_at(&BinaryHV::random(3101), 0.5, Duration::ZERO));
        assert_eq!(analyzer.stats.coherence_measurements, 0);
        assert_eq!(analyzer.stats.avg_temporal_coherence, 0.0);
        assert_eq!(analyzer.stats.avg_temporal_evidence_coverage, 0.0);

        assert!(analyzer.observe_at(&BinaryHV::random(3102), 0.5, Duration::from_secs(1),));
        assert!(analyzer.overall_evidence_coverage() > 0.0);
        assert_eq!(analyzer.stats.coherence_measurements, 1);
        assert!(
            (analyzer.stats.avg_temporal_coherence - analyzer.overall_temporal_coherence()).abs()
                < 1e-12
        );
        assert!(
            (analyzer.stats.avg_temporal_evidence_coverage - analyzer.overall_evidence_coverage())
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn fully_measured_low_identity_score_is_unhealthy() {
        let mut analyzer =
            TemporalConsciousnessAnalyzer::new(TemporalConsciousnessConfig::default());
        let identity = &mut analyzer.identity_coherence;
        identity.past_present_alignment_available = true;
        identity.present_future_alignment_available = true;
        identity.past_future_alignment_available = true;
        identity.past_present_alignment = 0.55;
        identity.present_future_alignment = 0.55;
        identity.past_future_alignment = 0.55;
        identity.coherence = 0.55;

        assert_eq!(analyzer.health_status(), TemporalHealthStatus::Unhealthy);
    }

    #[test]
    fn qualified_bad_trajectory_is_unhealthy_not_merely_under_observed() {
        let mut config = TemporalConsciousnessConfig::default();
        config.min_samples_for_trajectory = 3;
        config.phi_history_depth = 4;
        config.enable_husserlian_analysis = false;
        let mut analyzer = TemporalConsciousnessAnalyzer::new(config);

        for i in 0..3 {
            assert!(analyzer.observe_at(
                &BinaryHV::random(110 + i),
                0.1,
                Duration::from_secs(i as u64),
            ));
        }

        let report = analyzer.report();
        assert!(report.trajectory_evidence_sufficient);
        assert_eq!(report.health_status, TemporalHealthStatus::Unhealthy);
        assert!(!report.is_healthy);
    }

    #[test]
    fn trajectory_qualification_requires_three_samples_even_when_configured_minimum_is_one() {
        let mut config = TemporalConsciousnessConfig::default();
        config.min_samples_for_trajectory = 1;
        config.phi_history_depth = 4;
        let mut analyzer = TemporalConsciousnessAnalyzer::new(config);
        let state = BinaryHV::random(4320);

        assert!(analyzer.observe_at(&state, 0.7, Duration::ZERO));
        let one_sample = analyzer.report();
        assert_eq!(one_sample.trajectory_minimum_samples, 3);
        assert!(!one_sample.trajectory_evidence_sufficient);
        assert!(one_sample.trajectory_qualification_possible);

        assert!(analyzer.observe_at(&state, 0.7, Duration::from_secs(1)));
        assert!(!analyzer.report().trajectory_evidence_sufficient);

        assert!(analyzer.observe_at(&state, 0.7, Duration::from_secs(2)));
        let qualified = analyzer.report();
        assert!(qualified.trajectory_evidence_sufficient);
        assert!(qualified.phi_stability_available);
        assert!(qualified.phi_velocity_available);
    }

    #[test]
    fn trajectory_qualification_is_impossible_when_history_is_too_shallow() {
        let mut config = TemporalConsciousnessConfig::default();
        config.min_samples_for_trajectory = 5;
        config.phi_history_depth = 2;
        let mut analyzer = TemporalConsciousnessAnalyzer::new(config);

        for i in 0..4 {
            assert!(analyzer.observe_at(
                &BinaryHV::random(100 + i),
                0.5,
                Duration::from_secs(i as u64),
            ));
        }

        let report = analyzer.report();
        assert_eq!(report.trajectory_sample_count, 2);
        assert!(!report.trajectory_evidence_sufficient);
        assert!(!report.trajectory_qualification_possible);
        assert!(!report.is_healthy);
        assert!(
            report
                .recommendations
                .iter()
                .any(|r| r.contains("cannot qualify"))
        );
    }

    #[test]
    fn phi_velocity_uses_elapsed_seconds() {
        let mut trajectory = PhiTrajectory::new(8);
        assert!(trajectory.observe_at(0.2, Duration::ZERO));
        assert!(trajectory.observe_at(0.4, Duration::from_secs(2)));
        assert!((trajectory.velocity - 0.1).abs() < 1e-12);
    }

    #[test]
    fn phi_acceleration_is_unavailable_until_two_velocity_intervals_exist() {
        let mut trajectory = PhiTrajectory::new(8);
        assert!(trajectory.observe_at(0.2, Duration::ZERO));
        assert!(!trajectory.acceleration_available);

        assert!(trajectory.observe_at(0.3, Duration::from_secs(1)));
        assert!(!trajectory.acceleration_available);

        assert!(trajectory.observe_at(0.9, Duration::from_secs(4)));
        assert!(trajectory.acceleration_available);
        // v₁ = (0.3 - 0.2) / 1 = 0.1; v₂ = (0.9 - 0.3) / 3 = 0.2.
        // Their interval midpoints are two seconds apart, so acceleration = 0.05.
        assert!((trajectory.velocity - 0.2).abs() < 1e-12);
        assert!((trajectory.acceleration - 0.05).abs() < 1e-12);
    }

    #[test]
    fn temporal_report_does_not_render_unmeasured_acceleration_as_zero() {
        let analyzer = TemporalConsciousnessAnalyzer::new(TemporalConsciousnessConfig::default());
        let report = analyzer.report();

        assert!(!report.phi_acceleration_available);
        assert!(report.to_string().contains("N/A Φ/s²"));
    }

    #[test]
    fn same_sign_extreme_ema_does_not_false_reject_due_to_rounding() {
        let mut trajectory = PhiTrajectory::new_with_time_constant(8, 1.0);
        assert!(trajectory.observe_at(f64::MAX, Duration::ZERO));

        // exp(-40) is nonzero but too small for 1 - decay to differ from 1.
        // A naïve weighted sum rounds the first term to MAX then overflows while
        // adding the tiny positive second term. The true convex result is MAX.
        assert!(trajectory.observe_at(f64::MAX, Duration::from_secs(40)));
        assert_eq!(trajectory.ema, f64::MAX);
        assert_eq!(trajectory.history(), vec![f64::MAX, f64::MAX]);
        assert!(trajectory.velocity.is_finite());
    }

    #[test]
    fn extreme_finite_phi_rejects_unrepresentable_velocity_without_mutation() {
        let mut trajectory = PhiTrajectory::new(8);
        assert!(trajectory.observe_at(f64::MAX, Duration::ZERO));
        let retained_ema = trajectory.ema;

        // At one second this difference has a rate outside f64's finite range.
        assert!(!trajectory.observe_at(-f64::MAX, Duration::from_secs(1)));
        assert_eq!(trajectory.sample_count(), 1);
        assert_eq!(trajectory.history(), vec![f64::MAX]);
        assert_eq!(trajectory.ema, retained_ema);
        assert!(trajectory.velocity.is_finite());
        assert!(trajectory.acceleration.is_finite());
        assert!(!trajectory.acceleration_available);
    }

    #[test]
    fn extreme_finite_phi_with_representable_rates_keeps_dynamics_finite() {
        let mut trajectory = PhiTrajectory::new(8);
        assert!(trajectory.observe_at(f64::MAX, Duration::ZERO));
        assert!(trajectory.observe_at(-f64::MAX, Duration::from_secs(2)));
        assert!(trajectory.observe_at(-f64::MAX, Duration::from_secs(4)));

        assert!(trajectory.ema.is_finite());
        assert!(trajectory.velocity.is_finite());
        assert!(trajectory.acceleration.is_finite());
        assert!(trajectory.acceleration_available);
        assert!(trajectory.stability.is_finite());
    }

    #[test]
    fn extreme_velocity_pair_rejects_overflowing_acceleration_without_mutation() {
        let mut trajectory = PhiTrajectory::new(8);
        assert!(trajectory.observe_at(-5.0e304, Duration::ZERO));
        assert!(trajectory.observe_at(5.0e304, Duration::from_millis(1)));
        assert!(trajectory.velocity.is_finite());

        let retained_ema = trajectory.ema;
        let retained_velocity = trajectory.velocity;
        let retained_history = trajectory.history();

        // Both adjacent velocities are finite (~±1e308), but their difference
        // divided by the short midpoint interval is not representable as f64.
        let accepted = trajectory.observe_at(-5.0e304, Duration::from_millis(2));
        assert!(!accepted);
        assert_eq!(trajectory.sample_count(), 2);
        assert_eq!(trajectory.history(), retained_history);
        assert_eq!(trajectory.ema, retained_ema);
        assert_eq!(trajectory.velocity, retained_velocity);
        assert!(!trajectory.acceleration_available);
        assert!(trajectory.acceleration.is_finite());
        assert!(trajectory.stability.is_finite());
    }

    #[test]
    fn test_phi_trajectory() {
        let mut trajectory = PhiTrajectory::new(10);

        // Simulate increasing Φ
        for i in 0..10 {
            trajectory.observe(0.3 + i as f64 * 0.05);
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        assert!(trajectory.trend >= 0, "Should detect rising trend");
        assert!(trajectory.ema > 0.3, "EMA should increase");
    }

    #[test]
    fn deterministic_continuity_expires_events_by_replay_time_not_host_time() {
        let mut continuity = ConsciousnessContinuity::new(10, 1.0);
        let first = BinaryHV::random(2001);
        let second = BinaryHV::random(2002);

        continuity.observe_at(&first, 0.5, 0.5, Duration::ZERO);
        continuity.observe_at(&second, 0.5, 0.5, Duration::from_secs(1));
        assert_eq!(continuity.discontinuities.len(), 1);

        // Although these calls execute almost immediately, 61 seconds of replay
        // time have elapsed since the event, so it must no longer affect health.
        continuity.observe_at(&second, 0.5, 0.5, Duration::from_secs(62));
        assert!(continuity.discontinuities.is_empty());
    }

    #[test]
    fn continuity_rejects_invalid_or_nonmonotonic_replay_observations_without_mutation() {
        let mut continuity = ConsciousnessContinuity::new(10, 1.0);
        let first = BinaryHV::random(4101);
        let second = BinaryHV::random(4102);
        let third = BinaryHV::random(4103);

        assert!(continuity.observe_at(&first, 0.5, 0.5, Duration::from_secs(2)));
        assert!(continuity.observe_at(&second, 0.5, 0.5, Duration::from_secs(3)));

        let retained_states = continuity.recent_states.len();
        let retained_events = continuity.discontinuities.len();
        let score = continuity.score;

        assert!(!continuity.observe_at(&third, 0.5, 0.5, Duration::from_secs(3)));
        assert!(!continuity.observe_at(&third, 0.5, 0.5, Duration::from_secs(2)));
        assert!(!continuity.observe_at(&third, f64::NAN, 0.5, Duration::from_secs(4)));
        assert!(!continuity.observe_at(&third, 0.5, f64::INFINITY, Duration::from_secs(4)));

        assert_eq!(continuity.recent_states.len(), retained_states);
        assert_eq!(continuity.discontinuities.len(), retained_events);
        assert_eq!(continuity.score, score);
        assert_eq!(
            continuity.last_elapsed_observation,
            Some(Duration::from_secs(3))
        );
        assert!(continuity.observe_at(&third, 0.5, 0.5, Duration::from_secs(4)));
    }

    #[test]
    fn continuity_rejects_mixing_host_and_replay_clock_domains_without_mutation() {
        let first = BinaryHV::random(4201);
        let second = BinaryHV::random(4202);

        let mut replay_clock = ConsciousnessContinuity::new(10, 1.0);
        assert!(replay_clock.observe_at(&first, 0.5, 0.5, Duration::ZERO));
        let replay_state_count = replay_clock.recent_states.len();
        assert!(!replay_clock.try_observe(&second, 0.5, 0.5));
        assert!(!replay_clock.try_observe(&second, f64::NAN, 0.5));
        assert_eq!(replay_clock.recent_states.len(), replay_state_count);

        let mut host_clock = ConsciousnessContinuity::new(10, 1.0);
        host_clock.observe(&first, 0.5, 0.5);
        let host_state_count = host_clock.recent_states.len();
        assert!(!host_clock.observe_at(&second, 0.5, 0.5, Duration::from_secs(1)));
        assert_eq!(host_clock.recent_states.len(), host_state_count);
        assert_eq!(host_clock.last_elapsed_observation, None);
    }

    #[test]
    fn continuity_predicate_requires_at_least_two_states() {
        let mut continuity = ConsciousnessContinuity::new(10, 0.5);
        assert!(!continuity.is_continuous());

        let state = BinaryHV::random(4001);
        continuity.observe(&state, 0.5, 0.5);
        assert!(!continuity.is_continuous());

        continuity.observe(&state, 0.5, 0.5);
        assert!(continuity.is_continuous());
    }

    #[test]
    fn test_continuity_detection() {
        let mut continuity = ConsciousnessContinuity::new(10, 0.5);

        // Similar states should be continuous
        let state1 = BinaryHV::random(1);
        let state2 = state1; // Very similar

        continuity.observe(&state1, 0.5, 0.5);
        continuity.observe(&state2, 0.5, 0.5);

        assert!(
            continuity.score > 0.9,
            "Similar states should have high continuity"
        );
    }

    #[test]
    fn test_discontinuity_detection() {
        let mut continuity = ConsciousnessContinuity::new(10, 0.5);

        // Very different states should trigger discontinuity
        let state1 = BinaryHV::random(1);
        let state2 = BinaryHV::random(1000); // Very different

        continuity.observe(&state1, 0.5, 0.5);
        continuity.observe(&state2, 0.5, 0.5);

        // May or may not detect discontinuity depending on random vectors
        // The important thing is the system runs
        assert!(continuity.score >= 0.0 && continuity.score <= 1.0);
    }

    #[test]
    fn test_identity_coherence() {
        let config = TemporalConsciousnessConfig::default();
        let mut coherence = TemporalIdentityCoherence::new(config);

        // Identical selves should be fully coherent
        let self_vec = BinaryHV::random(42);
        coherence.update_explicit(&self_vec, &self_vec, &self_vec);

        assert!(
            coherence.coherence > 0.95,
            "Identical selves should be coherent"
        );
        assert!(coherence.is_coherent());
    }

    #[test]
    fn husserlian_coherence_is_unknown_until_a_temporal_relation_exists() {
        let mut husserlian = HusserlianAnalysis::new(10, 5);
        assert!(!husserlian.coherence_available());
        assert_eq!(husserlian.coherence(), 0.0);

        husserlian.update_primal(BinaryHV::random(3001), 0.5);
        assert!(!husserlian.coherence_available());
        assert_eq!(husserlian.coherence(), 0.0);

        // A second observation creates a retained-present relationship.
        husserlian.update_primal(BinaryHV::random(3002), 0.6);
        assert!(husserlian.coherence_available());
        assert!((0.0..=1.0).contains(&husserlian.coherence()));
    }

    #[test]
    fn analyzer_does_not_score_a_single_primal_as_measured_coherence() {
        let mut analyzer =
            TemporalConsciousnessAnalyzer::new(TemporalConsciousnessConfig::default());
        assert!(analyzer.observe_at(&BinaryHV::random(3011), 0.5, Duration::ZERO));

        let report = analyzer.report();
        assert!(!report.husserlian_coherence_available);
        assert_eq!(report.husserlian_coherence, 0.0);
        assert!(report.to_string().contains("Coherence:            N/A"));
        assert_eq!(report.overall_evidence_coverage, 0.0);
        assert_eq!(
            report.health_status,
            TemporalHealthStatus::InsufficientEvidence
        );
    }

    #[test]
    fn husserlian_analysis_respects_configured_retention_and_protention_capacities() {
        let mut husserlian = HusserlianAnalysis::new(2, 1);
        for i in 0..7 {
            husserlian.update_primal(BinaryHV::random(3200 + i), 0.4 + i as f64 * 0.01);
        }
        assert_eq!(husserlian.retention.len(), 2);
        assert_eq!(husserlian.retention_depth, 2);

        husserlian.update_protentions(vec![
            (BinaryHV::random(3301), 0.4),
            (BinaryHV::random(3302), 0.5),
            (BinaryHV::random(3303), 0.6),
        ]);
        assert_eq!(husserlian.protention.len(), 1);
        assert_eq!(husserlian.protention_depth, 1);
        assert_eq!(husserlian.temporal_horizon, 3.0);

        let mut zero_capacity = HusserlianAnalysis::new(0, 0);
        zero_capacity.update_primal(BinaryHV::random(3401), 0.5);
        zero_capacity.update_primal(BinaryHV::random(3402), 0.6);
        zero_capacity.update_protentions(vec![(BinaryHV::random(3403), 0.7)]);
        assert!(zero_capacity.retention.is_empty());
        assert!(zero_capacity.protention.is_empty());
        assert_eq!(zero_capacity.retention_depth, 0);
        assert_eq!(zero_capacity.protention_depth, 0);
    }

    #[test]
    fn test_husserlian_analysis() {
        let mut husserlian = HusserlianAnalysis::new(10, 5);

        // Add primal impressions
        for i in 0..5 {
            let state = BinaryHV::random(i as u64);
            husserlian.update_primal(state, 0.5 + i as f64 * 0.05);
        }

        assert_eq!(husserlian.retention_depth, 4);
        assert!(husserlian.primal_impression.is_some());
    }

    #[test]
    fn test_temporal_analyzer() {
        let mut analyzer = TemporalConsciousnessAnalyzer::default_config();

        // Simulate consciousness over time
        for i in 0..10 {
            let state = BinaryHV::random(i as u64);
            let phi = 0.4 + i as f64 * 0.03;

            analyzer.observe(&state, phi, None, None);
        }

        // Should have recorded observations
        assert_eq!(analyzer.stats.observations, 10);

        // Get report
        let report = analyzer.report();
        assert!(report.overall_coherence >= 0.0);

        println!("{}", report);
    }

    #[test]
    fn binding_rejects_nonfinite_and_out_of_range_strength_without_mutation() {
        let mut binding = TemporalBindingAnalysis::new();
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.01, 1.01] {
            assert!(!binding.try_add_binding(
                "perception",
                "memory",
                AllenRelation::Overlaps,
                invalid,
            ));
            assert!(binding.bindings.is_empty());
            assert_eq!(binding.binding_strength, 1.0);
            assert_eq!(binding.gamma_sync, 0.8);
        }

        assert!(binding.try_add_binding("perception", "memory", AllenRelation::Overlaps, 0.9,));
        assert_eq!(binding.bindings.len(), 1);
        assert!((binding.binding_strength - 0.9).abs() < 1e-12);

        // Labels are identifiers, not optional display text.
        assert!(!binding.try_add_binding("", "memory", AllenRelation::Overlaps, 0.8,));
        assert!(!binding.try_add_binding("   ", "memory", AllenRelation::Overlaps, 0.8,));
        assert!(!binding.try_add_binding("perception", "\t", AllenRelation::Overlaps, 0.8,));
        assert_eq!(binding.bindings.len(), 1);
    }

    #[test]
    fn invalid_binding_does_not_evict_when_retention_is_full() {
        let mut binding = TemporalBindingAnalysis::new();
        for i in 0..256 {
            let a = format!("source-{i}");
            let b = format!("target-{i}");
            assert!(binding.try_add_binding(&a, &b, AllenRelation::Overlaps, i as f64 / 255.0,));
        }
        assert_eq!(binding.bindings.len(), 256);
        assert_eq!(binding.bindings[0].component_a, "source-0");

        assert!(!binding.try_add_binding("", "replacement", AllenRelation::Overlaps, 1.0,));
        assert_eq!(binding.bindings.len(), 256);
        assert_eq!(binding.bindings[0].component_a, "source-0");
    }

    #[test]
    fn binding_predicate_requires_at_least_one_binding() {
        let mut binding = TemporalBindingAnalysis::new();
        assert!(!binding.is_unified());

        binding.add_binding("perception", "memory", AllenRelation::Overlaps, 0.9);
        assert!(binding.is_unified());
    }

    #[test]
    fn empty_analyzer_recommends_more_continuity_and_binding_evidence() {
        let analyzer = TemporalConsciousnessAnalyzer::default_config();
        let recommendations = analyzer.generate_recommendations();
        assert!(
            recommendations
                .iter()
                .any(|r| r.contains("Continuity evidence pending"))
        );
        assert!(
            recommendations
                .iter()
                .any(|r| r.contains("Temporal binding evidence pending"))
        );
        assert!(
            !recommendations
                .iter()
                .any(|r| r.contains("Temporal consciousness is healthy"))
        );
    }

    #[test]
    fn test_temporal_binding() {
        let mut binding = TemporalBindingAnalysis::new();

        binding.add_binding("perception", "memory", AllenRelation::Overlaps, 0.8);
        binding.add_binding("intention", "action", AllenRelation::Meets, 0.9);

        assert!(binding.binding_strength > 0.7);
        assert!(binding.is_unified());
    }

    #[test]
    fn test_phi_trajectory_health() {
        let mut trajectory = PhiTrajectory::new(20);

        // Healthy trajectory: stable, high Φ
        for _ in 0..15 {
            trajectory.observe(0.6 + (rand::random::<f64>() - 0.5) * 0.05);
        }

        assert!(trajectory.is_healthy() || trajectory.ema > 0.3);
    }

    #[test]
    fn test_temporal_coherence_computation() {
        let mut analyzer = TemporalConsciousnessAnalyzer::default_config();

        // Build up history
        let base_state = BinaryHV::random(42);
        for i in 0..20 {
            // Slight variations
            let state = if i % 2 == 0 {
                base_state
            } else {
                BinaryHV::random(i as u64)
            };
            analyzer.observe(&state, 0.5, None, None);
        }

        let coherence = analyzer.overall_temporal_coherence();
        assert!((0.0..=1.0).contains(&coherence));
    }

    #[test]
    fn test_recommendations_generation() {
        let analyzer = TemporalConsciousnessAnalyzer::default_config();
        let report = analyzer.report();

        assert!(!report.recommendations.is_empty());
    }
}
