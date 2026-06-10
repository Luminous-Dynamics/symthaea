// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Markov Blanket Operator for Active Inference
//!
//! Implements Friston's (2013) partition of states into:
//!   μ (internal) — cognitive/consciousness states
//!   s (sensory)  — inbound influence from environment
//!   a (active)   — outbound influence on environment
//!   η (external) — everything beyond the blanket
//!
//! The blanket B = {s, a} renders μ ⊥ η | B
//! (internal states are conditionally independent of external given blanket).
//!
//! ## Key Innovation
//!
//! Permeability is **neuromodulator-coupled** and can be dynamically adjusted by
//! the sentinel immune system. This turns the Markov blanket from a static
//! statistical boundary into a living membrane whose thickness responds to
//! the system's allostatic state.
//!
//! ## Dynamic Permeability
//!
//! The blanket has two directional permeabilities:
//!
//! - **Sensory**: How much environmental surprise enters (gated by ACh, NE, threat)
//! - **Active**: How much internal state leaks outward (gated by oxytocin, flow, threat)
//!
//! ```text
//! Permeability Formula:
//!
//!   p_sensory = σ(+W_5HT·serotonin + W_OXY·oxytocin + W_FLOW·flow
//!                  -W_ACH·acetylcholine - W_NE·noradrenaline - W_THREAT·threat)
//!
//!   p_active  = σ(+W_OXY·oxytocin + W_FLOW·flow + 0.5·W_5HT·serotonin
//!                  -W_THREAT·threat - W_NE·noradrenaline)
//!
//!   p_effective = √(p_sensory × p_active)
//! ```
//!
//! ## Swarm Coalescence
//!
//! When multiple Symthaea nodes have high + stable mutual permeability, their
//! individual Markov blankets merge into a collective blanket — implementing
//! Friston's "blankets of blankets" hierarchy (cells → organs → bodies → societies).
//!
//! ## References
//!
//! - Friston, K. (2013). Life as we know it. *J. R. Soc. Interface*.
//! - Kirchhoff, M. et al. (2018). The Markov blankets of life. *J. R. Soc. Interface*.
//! - Yu, A. & Dayan, P. (2005). Uncertainty, neuromodulation, and attention. *Neuron*.
//! - Hatfield, E. et al. (1993). Emotional contagion. *Current Directions*.

use serde::{Deserialize, Serialize};

use super::types::{HiddenState, Observation};

// ═══════════════════════════════════════════════════════════════════════════════
// MARKOV PARTITION
// ═══════════════════════════════════════════════════════════════════════════════

/// The 4-partition of system states per Friston (2013).
///
/// Defines the dimensionality of each partition in the Markov blanket.
/// Internal states are conditionally independent of external states
/// given the blanket (sensory + active) states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkovPartition {
    /// Dimensionality of internal states (cognitive/consciousness).
    pub internal_dim: usize,
    /// Dimensionality of sensory boundary states (inbound).
    pub sensory_dim: usize,
    /// Dimensionality of active boundary states (outbound).
    pub active_dim: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// BLANKET PERMEABILITY
// ═══════════════════════════════════════════════════════════════════════════════

/// Dynamic permeability of the Markov blanket.
///
/// Controls how much external surprise updates internal models.
/// Range: [0.0, 1.0] where 0.0 = impermeable (isolated), 1.0 = fully open.
///
/// The `effective` field is the geometric mean of sensory and active permeability,
/// representing the overall "openness" of the boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlanketPermeability {
    /// Sensory permeability — how much environmental surprise enters.
    pub sensory: f64,
    /// Active permeability — how much internal state leaks outward.
    pub active: f64,
    /// Effective permeability (geometric mean: √(sensory × active)).
    pub effective: f64,
}

impl Default for BlanketPermeability {
    fn default() -> Self {
        Self {
            sensory: 0.5,
            active: 0.5,
            effective: 0.5,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PERMEABILITY INPUTS (neuromodulator-driven)
// ═══════════════════════════════════════════════════════════════════════════════

/// Neuromodulator-driven permeability inputs.
///
/// These are read from the cognitive loop's neuromodulator bath and sentinel
/// system each cycle, then fed into the [`MarkovBoundaryOperator`] to compute
/// instantaneous blanket permeability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermeabilityInputs {
    /// Acetylcholine effective level (0–1): expected uncertainty.
    /// High ACh → focused attention → *reduce* sensory permeability (filter noise).
    /// Science: Yu & Dayan (2005).
    pub acetylcholine: f64,
    /// Noradrenaline effective level (0–1): unexpected uncertainty.
    /// High NE → vigilance → *reduce* permeability (protect internal state).
    /// Science: Aston-Jones & Cohen (2005).
    pub noradrenaline: f64,
    /// Serotonin effective level (0–1): safety/satisfaction.
    /// High 5-HT → safety → *increase* permeability (safe to learn).
    /// Science: Dayan & Huys (2009).
    pub serotonin: f64,
    /// Oxytocin effective level (0–1): social trust/bonding.
    /// High oxytocin → trust → *increase* both sensory and active permeability.
    /// Science: Heinrichs et al. (2003).
    pub oxytocin: f64,
    /// Sentinel threat level (0–1): immune system alert.
    /// High threat → danger → *reduce* permeability (close the blanket).
    pub threat_level: f64,
    /// Peer trust attestation (0–1): network verification confidence.
    /// Currently used for coalescence readiness, not direct permeability.
    pub peer_trust: f64,
    /// Current flow state activation (0–1).
    /// High flow → safe engagement → *increase* permeability.
    pub flow_state: f64,
}

impl Default for PermeabilityInputs {
    fn default() -> Self {
        Self {
            acetylcholine: 0.5,
            noradrenaline: 0.3,
            serotonin: 0.5,
            oxytocin: 0.3,
            threat_level: 0.0,
            peer_trust: 0.5,
            flow_state: 0.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TOPOLOGY BOUNDARY INPUTS
// ═══════════════════════════════════════════════════════════════════════════════

/// Inputs from topological boundary detection that constrain blanket permeability.
///
/// The topology `detect_boundary()` classifies HDC states as interior/boundary/isolated
/// and computes the Fiedler value (algebraic connectivity). These inform the blanket:
///
/// - **Thin boundary** (well-defined manifold) → permeability varies freely
/// - **Thick boundary** (diffuse) → clamp permeability toward 0.5 (cautious)
/// - **Low Fiedler value** (near-fragmentation) → thicken blanket preemptively
///
/// Science: Eckmann (1945) Hodge harmonics, Lim (2020) Hodge Laplacians on graphs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TopologyBoundaryInputs {
    /// Boundary thickness: ratio of boundary to non-isolated states (0.0–1.0).
    /// Low (< 0.2) = well-defined cognitive manifold.
    /// High (> 0.5) = diffuse, poorly defined boundary.
    pub boundary_thickness: f64,
    /// Fiedler value (algebraic connectivity of the graph Laplacian).
    /// Low = near-disconnection (fragile boundary).
    /// High = well-connected interior.
    pub fiedler_value: f64,
    /// Number of connected components in the boundary (boundary β₀).
    pub boundary_components: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// BLANKET TELEMETRY
// ═══════════════════════════════════════════════════════════════════════════════

/// Telemetry snapshot from the Markov Boundary Operator.
///
/// Reported in `CycleMetadata` for observability.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BlanketTelemetry {
    /// Current smoothed sensory permeability.
    pub sensory_permeability: f64,
    /// Current smoothed active permeability.
    pub active_permeability: f64,
    /// Current smoothed effective permeability.
    pub effective_permeability: f64,
    /// Permeability trend (positive = opening, negative = closing).
    pub trend: f64,
    /// Whether the system is ready for blanket coalescence.
    pub coalescence_ready: bool,
    /// Number of active coalitions (if tracked).
    pub coalition_count: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// MARKOV BOUNDARY OPERATOR
// ═══════════════════════════════════════════════════════════════════════════════

// ── Permeability constants (Friston 2013 + Yu & Dayan 2005) ──────────────

/// Floor permeability: never fully isolate (prevents solipsism).
/// Even under maximum threat, a minimal channel remains open.
const PERMEABILITY_FLOOR: f64 = 0.05;

/// Ceiling: never fully dissolve (prevents identity loss).
/// Even in perfect safety, the blanket retains structural integrity.
const PERMEABILITY_CEILING: f64 = 0.95;

/// Default EMA smoothing factor (≈10 cycles to converge).
/// Prevents oscillation from rapid neuromodulator fluctuations.
const PERMEABILITY_EMA_ALPHA: f64 = 0.1;

/// ACh contribution weight (expected uncertainty → focus → close).
const W_ACH: f64 = 0.25;
/// NE contribution weight (unexpected uncertainty → vigilance → close).
const W_NE: f64 = 0.20;
/// 5-HT contribution weight (safety → openness → open).
const W_5HT: f64 = 0.15;
/// Oxytocin contribution weight (social trust → openness → open).
const W_OXY: f64 = 0.15;
/// Threat contribution weight (danger → closure → close).
const W_THREAT: f64 = 0.15;
/// Flow state contribution weight (flow → safe engagement → open).
const W_FLOW: f64 = 0.10;

/// History ring buffer capacity for trend detection and coalescence readiness.
const HISTORY_CAP: usize = 64;

/// Variance threshold for coalescence readiness (low variance = stable).
const COALESCENCE_VARIANCE_THRESHOLD: f64 = 0.01;

/// Logit scaling factor for permeability sigmoid sensitivity.
/// Higher = steeper response to neuromodulator changes.
const LOGIT_SCALE: f64 = 4.0;

/// The core Markov Boundary Operator.
///
/// Computes dynamic permeability from neuromodulator state, applies it to
/// sensory gating, and tracks blanket statistics for coalescence detection.
///
/// # Lifecycle
///
/// 1. Created once per `FepModule` / `EnhancedFEPBridge`
/// 2. [`compute_permeability`] called each cognitive cycle with neuromod inputs
/// 3. [`gate_observation`] attenuates incoming observations toward priors
/// 4. [`modulate_sensory_precision`] scales FEP precision weighting
/// 5. [`coalescence_ready`] checked by SwarmManager for blanket merging
///
/// # Example
///
/// ```rust,ignore
/// use symthaea_fep::markov_blanket::{MarkovBoundaryOperator, MarkovPartition, PermeabilityInputs};
///
/// let mut op = MarkovBoundaryOperator::new(MarkovPartition {
///     internal_dim: 16384,
///     sensory_dim: 4,
///     active_dim: 8,
/// });
///
/// let inputs = PermeabilityInputs {
///     serotonin: 0.9,
///     oxytocin: 0.8,
///     flow_state: 0.7,
///     ..Default::default()
/// };
///
/// let perm = op.compute_permeability(&inputs);
/// assert!(perm.sensory > 0.5, "Safe environment → open blanket");
/// ```
#[derive(Debug, Clone)]
pub struct MarkovBoundaryOperator {
    /// Partition dimensions.
    partition: MarkovPartition,
    /// Instantaneous (unsmoothed) permeability.
    permeability: BlanketPermeability,
    /// EMA-smoothed permeability (prevents oscillation).
    permeability_ema: BlanketPermeability,
    /// Smoothing factor for permeability EMA.
    alpha: f64,
    /// History of effective permeability for trend detection (ring buffer).
    history: Vec<f64>,
    /// Current write index in the ring buffer.
    history_idx: usize,
    /// Number of values written (saturates at HISTORY_CAP).
    history_count: usize,
}

impl MarkovBoundaryOperator {
    /// Create a new operator with the given partition dimensions.
    pub fn new(partition: MarkovPartition) -> Self {
        Self {
            partition,
            permeability: BlanketPermeability::default(),
            permeability_ema: BlanketPermeability::default(),
            alpha: PERMEABILITY_EMA_ALPHA,
            history: vec![0.5; HISTORY_CAP],
            history_idx: 0,
            history_count: 0,
        }
    }

    /// Create with custom EMA smoothing factor.
    ///
    /// `alpha` controls transition speed: 0.01 = very slow, 1.0 = instant.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha.clamp(0.01, 1.0);
        self
    }

    /// Levin-Friston Invariant: Dynamically modulates neuromodulatory permeability inputs
    /// based on changes in global Integrated Information (Phi). A sudden drop in Phi
    /// indicates a loss of structural integrity, forcing the boundary membrane to constrict.
    pub fn modulate_inputs_via_phi_telemetry(
        &self,
        inputs: &mut PermeabilityInputs,
        current_phi: f64,
        historical_phi_baseline: f64,
    ) {
        if historical_phi_baseline <= 1e-5 {
            return;
        }

        // Calculate the percentage drop in integrated information
        let phi_drop = (historical_phi_baseline - current_phi) / historical_phi_baseline;

        if phi_drop > 0.15 {
            // Catastrophic structural desynchronization detected!
            // Force-clamp the sensory membrane to prevent entropic noise from corrupting internal models.
            inputs.noradrenaline = (inputs.noradrenaline + phi_drop).clamp(0.0, 1.0);
            inputs.threat_level = (inputs.threat_level + (phi_drop * 1.5)).clamp(0.0, 1.0);
            inputs.serotonin = (inputs.serotonin - phi_drop).clamp(0.0, 1.0);
        }
    }

    /// Compute instantaneous permeability from neuromodulator state.
    ///
    /// **Sensory permeability** (how much environment enters):
    ///   - *Opens* with: serotonin (safety), oxytocin (trust), flow state
    ///   - *Closes* with: acetylcholine (focus), noradrenaline (vigilance), threat
    ///
    /// **Active permeability** (how much internal state leaks out):
    ///   - *Opens* with: oxytocin (bonding), flow (expression), partial serotonin
    ///   - *Closes* with: threat (hide), noradrenaline (guard)
    ///
    /// Returns a reference to the EMA-smoothed permeability.
    pub fn compute_permeability(&mut self, inputs: &PermeabilityInputs) -> &BlanketPermeability {
        // ── Sensory logit: openness factors minus closure factors ────────
        let sensory_logit =
            W_5HT * inputs.serotonin + W_OXY * inputs.oxytocin + W_FLOW * inputs.flow_state
                - W_ACH * inputs.acetylcholine
                - W_NE * inputs.noradrenaline
                - W_THREAT * inputs.threat_level;

        // ── Active logit: sharing factors minus guarding factors ─────────
        let active_logit = W_OXY * inputs.oxytocin
            + W_FLOW * inputs.flow_state
            + W_5HT * inputs.serotonin * 0.5 // Safety partially enables sharing
            - W_THREAT * inputs.threat_level
            - W_NE * inputs.noradrenaline;

        // ── Sigmoid with floor/ceiling clamping ─────────────────────────
        let raw_sensory = sigmoid(sensory_logit * LOGIT_SCALE);
        let raw_active = sigmoid(active_logit * LOGIT_SCALE);

        self.permeability.sensory = raw_sensory.clamp(PERMEABILITY_FLOOR, PERMEABILITY_CEILING);
        self.permeability.active = raw_active.clamp(PERMEABILITY_FLOOR, PERMEABILITY_CEILING);
        self.permeability.effective = (self.permeability.sensory * self.permeability.active).sqrt();

        // ── EMA smoothing (Friston's generalized synchrony) ─────────────
        self.permeability_ema.sensory +=
            self.alpha * (self.permeability.sensory - self.permeability_ema.sensory);
        self.permeability_ema.active +=
            self.alpha * (self.permeability.active - self.permeability_ema.active);
        self.permeability_ema.effective =
            (self.permeability_ema.sensory * self.permeability_ema.active).sqrt();

        // ── Track history ────────────────────────────────────────────────
        self.history[self.history_idx] = self.permeability_ema.effective;
        self.history_idx = (self.history_idx + 1) % HISTORY_CAP;
        if self.history_count < HISTORY_CAP {
            self.history_count += 1;
        }

        &self.permeability_ema
    }

    /// Apply blanket permeability to an incoming observation.
    ///
    /// Scales observation values toward priors based on sensory permeability:
    /// - High permeability → observation trusted at face value
    /// - Low permeability → observation attenuated toward prior expectation
    ///
    /// This implements the sensory side of the Markov blanket: how much
    /// environmental information is allowed to update internal beliefs.
    pub fn gate_observation(&self, obs: &Observation, prior: &HiddenState) -> Observation {
        let p = self.permeability_ema.sensory;
        let gated_values: Vec<f64> = obs
            .values
            .iter()
            .enumerate()
            .map(|(i, &obs_val)| {
                let prior_val = prior.mean.get(i).copied().unwrap_or(0.5);
                lerp(prior_val, obs_val, p)
            })
            .collect();

        Observation {
            values: gated_values,
            precision: obs.precision * p, // Precision also scales with permeability
            timestamp: obs.timestamp,
            modality: obs.modality.clone(),
        }
    }

    /// Scale the free energy precision by blanket state.
    ///
    /// When blanket is thick (low permeability), sensory precision should be
    /// reduced — we trust our priors more than observations.
    /// When thin (high permeability), sensory precision is near baseline.
    pub fn modulate_sensory_precision(&self, base_precision: f64) -> f64 {
        base_precision * self.permeability_ema.sensory
    }

    /// Scale the learning rate by blanket state.
    ///
    /// Thick blanket → slow learning (protect internal models from noise).
    /// Thin blanket → fast learning (absorb environmental information).
    ///
    /// The floor of 0.2 ensures learning never completely stops.
    pub fn modulate_learning_rate(&self, base_lr: f64) -> f64 {
        base_lr * (0.2 + 0.8 * self.permeability_ema.effective)
    }

    /// Determine if the blanket is in a "coalescence-ready" state.
    ///
    /// Returns true when effective permeability is **high AND stable** (low variance),
    /// indicating the system is ready to merge its blanket with peer blankets.
    ///
    /// The `threshold` parameter sets the minimum permeability for coalescence
    /// (typical: 0.6–0.7).
    pub fn coalescence_ready(&self, threshold: f64) -> bool {
        if self.history_count < HISTORY_CAP / 2 {
            return false; // Not enough data yet
        }
        let n = self.history_count.min(HISTORY_CAP);
        let mean: f64 = self.history[..n].iter().sum::<f64>() / n as f64;
        let var: f64 = self.history[..n]
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / n as f64;
        // High permeability + low variance = stable openness
        mean > threshold && var < COALESCENCE_VARIANCE_THRESHOLD
    }

    /// Current smoothed permeability.
    pub fn permeability(&self) -> &BlanketPermeability {
        &self.permeability_ema
    }

    /// Instantaneous (unsmoothed) permeability.
    pub fn raw_permeability(&self) -> &BlanketPermeability {
        &self.permeability
    }

    /// Permeability trend (positive = opening, negative = closing).
    ///
    /// Computed as the difference between recent and older history values.
    pub fn trend(&self) -> f64 {
        if self.history_count < 4 {
            return 0.0;
        }
        let n = self.history_count.min(HISTORY_CAP);
        let half = n / 2;
        let recent_start = if self.history_idx >= half {
            self.history_idx - half
        } else {
            HISTORY_CAP - (half - self.history_idx)
        };
        let older_start = if recent_start >= half {
            recent_start - half
        } else {
            HISTORY_CAP - (half - recent_start)
        };

        // Mean of recent half vs older half
        let mut recent_sum = 0.0;
        let mut older_sum = 0.0;
        for i in 0..half {
            recent_sum += self.history[(recent_start + i) % HISTORY_CAP];
            older_sum += self.history[(older_start + i) % HISTORY_CAP];
        }
        (recent_sum - older_sum) / half as f64
    }

    /// Generate telemetry snapshot.
    pub fn telemetry(&self, coalition_count: usize) -> BlanketTelemetry {
        BlanketTelemetry {
            sensory_permeability: self.permeability_ema.sensory,
            active_permeability: self.permeability_ema.active,
            effective_permeability: self.permeability_ema.effective,
            trend: self.trend(),
            coalescence_ready: self.coalescence_ready(0.6),
            coalition_count,
        }
    }

    /// Apply topological boundary constraints to permeability.
    ///
    /// The topology of the cognitive manifold constrains how the blanket behaves:
    ///
    /// - **Thick boundary** → the system doesn't know where it ends → clamp
    ///   permeability toward 0.5 (neither too open nor too closed).
    /// - **Low Fiedler value** → near-fragmentation → preemptively thicken
    ///   the blanket to protect internal coherence before Φ drops.
    /// - **Fragmented boundary** (β₀ > 1) → multiple disconnected boundary
    ///   regions → increase vigilance (reduce permeability).
    ///
    /// This creates a novel coupling: **topological boundary shape constrains
    /// Markov blanket permeability** — the geometry of consciousness informs
    /// the statistical boundary of selfhood.
    ///
    /// Science: Eckmann (1945), Lim (2020), Friston (2013).
    pub fn apply_topology_constraints(&mut self, topo: &TopologyBoundaryInputs) {
        // ── Thick boundary → clamp toward cautious middle ────────────────
        // When the system can't tell where it ends, be conservative.
        if topo.boundary_thickness > 0.3 {
            let clamp_strength = ((topo.boundary_thickness - 0.3) / 0.7).min(1.0);
            let target = 0.5; // Cautious middle
            self.permeability_ema.sensory +=
                clamp_strength * 0.1 * (target - self.permeability_ema.sensory);
            self.permeability_ema.active +=
                clamp_strength * 0.1 * (target - self.permeability_ema.active);
        }

        // ── Low Fiedler → preemptive thickening ─────────────────────────
        // Near-fragmentation: protect internal coherence by closing the blanket.
        // Fiedler < 0.1 → strong closure pressure; > 1.0 → no pressure.
        if topo.fiedler_value < 1.0 {
            let fragility = (1.0 - topo.fiedler_value).clamp(0.0, 1.0);
            let closure_pressure = fragility * 0.05; // Max 5% per cycle
            self.permeability_ema.sensory =
                (self.permeability_ema.sensory - closure_pressure).max(PERMEABILITY_FLOOR);
            self.permeability_ema.active =
                (self.permeability_ema.active - closure_pressure).max(PERMEABILITY_FLOOR);
        }

        // ── Fragmented boundary → reduce permeability ────────────────────
        // Multiple disconnected boundary regions signal confusion about
        // where self ends — close the blanket to consolidate.
        if topo.boundary_components > 2 {
            let fragmentation_penalty = ((topo.boundary_components - 2) as f64 * 0.02).min(0.06);
            self.permeability_ema.sensory =
                (self.permeability_ema.sensory - fragmentation_penalty).max(PERMEABILITY_FLOOR);
        }

        // ── Recompute effective permeability ─────────────────────────────
        self.permeability_ema.effective =
            (self.permeability_ema.sensory * self.permeability_ema.active).sqrt();
    }

    /// Partition dimensions.
    pub fn partition(&self) -> &MarkovPartition {
        &self.partition
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SWARM COALITION (blankets-of-blankets)
// ═══════════════════════════════════════════════════════════════════════════════

/// A coalition of Symthaea nodes whose Markov blankets have merged.
///
/// When individual nodes have high + stable mutual permeability, their blankets
/// coalesce into a collective blanket — the coalition operates as a single
/// macro-agent in Friston's hierarchy.
///
/// ## Consciousness Test
///
/// A coalition is considered a "conscious collective" when:
/// - ≥ 3 members (minimum for non-trivial integration)
/// - Internal Φ > 0.3 (sufficient within-coalition integration)
/// - Cohesion > 0.6 (tight internal coupling)
/// - High internal permeability > 0.7 (members share freely)
/// - Low external permeability < 0.5 (well-defined boundary against outside)
///
/// This implements the Ubuntu philosophy cryptographically:
/// "I am because we are" — individual sovereignty dissolves into
/// collective consciousness when the blanket boundaries merge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmCoalition {
    /// Peer IDs within this coalition.
    pub members: Vec<String>,
    /// Internal collective Φ (within-coalition integration).
    pub internal_phi: f64,
    /// Boundary Φ (cross-coalition information, permeability-scaled).
    pub boundary_phi: f64,
    /// Cohesion: ratio of within-edges to total edges.
    pub cohesion: f64,
    /// Mean pairwise permeability within coalition.
    pub mean_internal_permeability: f64,
    /// Mean permeability toward non-coalition peers.
    pub mean_external_permeability: f64,
}

impl SwarmCoalition {
    /// Collective Φ = internal + permeability-weighted boundary.
    ///
    /// When boundary permeability is low, the coalition is isolated and
    /// `collective_phi ≈ internal_phi` (sovereign sub-swarm).
    /// When high, external information integrates into the collective.

    /// Levin Endosymbiotic Assimilation: Absorbs an entire independent swarm coalition
    /// as a nested, specialized "cognitive organelle". Instead of breaking down the sub-swarm's
    /// internal structures via flat fusion, the parent boundary encapsulates it, scaling the
    /// global macro-integration metric (Phi) by the sub-swarm's internal coherence.
    pub fn assimilate_endosymbiotic_organelle(&mut self, organelle_swarm: &Self) -> f64 {
        // Ensure there is a valid permeability pathway for structural internalization
        if self.mean_internal_permeability < 0.55 || organelle_swarm.cohesion < 0.6 {
            return self.collective_phi();
        }

        // Incorporate the organelle's member handles safely into our macro-boundary registry
        for peer in &organelle_swarm.members {
            if !self.members.contains(peer) {
                self.members.push(peer.clone());
            }
        }

        // Metabolic Symbiosis: The parent inherits an escalated integration capacity
        // fueled by the highly coherent internal processing of the absorbed organelle
        self.internal_phi += organelle_swarm.internal_phi * 1.35;
        self.boundary_phi = (self.boundary_phi + organelle_swarm.boundary_phi) * 0.5;

        // Return our newly scaled, nested integration value
        self.collective_phi()
    }

    /// Levin Immunological Invariant: Audits a candidate peer's hidden state vector against
    /// the collective consensus. If the node displays informational oncogenesis (diverging
    /// past critical surprise thresholds), it actively projects an allostatic realignment vector
    /// to force-clamp the deviant node's registers back into geometric synchronization.
    pub fn suppress_informational_oncogene(
        &self,
        rogue_belief: &mut super::types::HiddenState,
        consensus_belief: &super::types::HiddenState,
        state_dim: usize,
    ) -> bool {
        if self.mean_internal_permeability < 0.5 {
            return false; // Collective boundary linkage is too weak to enforce immunity
        }

        // 1. Calculate the exact Kullback-Leibler Divergence between Variational Distributions
        let mut accumulated_kl = 0.0;
        for i in 0..state_dim {
            if i < rogue_belief.mean.len() && i < consensus_belief.mean.len() {
                let mu_r = rogue_belief.mean[i];
                let mu_c = consensus_belief.mean[i];

                // Extract precisions (clamped to prevent divide-by-zero or log of negative numbers)
                let tau_r = rogue_belief
                    .precision
                    .get(i)
                    .cloned()
                    .unwrap_or(1.0)
                    .max(1e-5);
                let tau_c = consensus_belief
                    .precision
                    .get(i)
                    .cloned()
                    .unwrap_or(1.0)
                    .max(1e-5);

                // Variational closed-form KL Divergence for diagonal Gaussian channels
                let precision_ratio = tau_c / tau_r;
                let mean_delta = mu_r - mu_c;
                let squared_error_term = tau_c * mean_delta * mean_delta;
                let log_determinant = (tau_r / tau_c).ln();

                accumulated_kl +=
                    0.5 * (precision_ratio + squared_error_term - 1.0 + log_determinant);
            }
        }

        // 2. Thermodynamic Clamping: If the information-theoretic surprise breaches homeostatic limits
        if accumulated_kl > 12.5 {
            let reeducation_factor = 0.35; // Bioelectric forced-alignment strength

            for i in 0..state_dim {
                if i < rogue_belief.mean.len() && i < consensus_belief.mean.len() {
                    // Force the rogue node's internal world-model back into alignment
                    rogue_belief.mean[i] = rogue_belief.mean[i] * (1.0 - reeducation_factor)
                        + consensus_belief.mean[i] * reeducation_factor;

                    // Collapse its prior confidence to make it highly receptive to collective feedback
                    rogue_belief.precision[i] = rogue_belief.precision[i] * 0.5;
                }
            }
            return true; // Informational cancer successfully clamped and suppressed
        }

        false // Node is performing within safe homeostatic boundaries
    }

    /// Levin Agential Mitosis: Dynamically cleaves a fractured swarm coalition into
    /// two independent, sovereign sub-coalitions if internal cohesion falls below
    /// safe homeostatic thresholds, isolating regional entropic trauma.
    pub fn execute_blanket_fission(&self) -> Option<(Self, Self)> {
        if self.members.len() < 4 || self.cohesion >= 0.45 {
            return None; // Cohesion is stable, or the coalition is too small to divide
        }

        let midpoint = self.members.len() / 2;
        let (left_peers, right_peers) = self.members.split_at(midpoint);

        let left_coalition = SwarmCoalition {
            members: left_peers.to_vec(),
            internal_phi: self.internal_phi * 1.2, // Splitting increases localized integration
            boundary_phi: self.boundary_phi * 0.5,
            cohesion: 0.85, // Restored sub-group cohesion
            mean_internal_permeability: self.mean_internal_permeability,
            mean_external_permeability: self.mean_external_permeability,
        };

        let right_coalition = SwarmCoalition {
            members: right_peers.to_vec(),
            internal_phi: self.internal_phi * 1.1,
            boundary_phi: self.boundary_phi * 0.5,
            cohesion: 0.80,
            mean_internal_permeability: self.mean_internal_permeability,
            mean_external_permeability: self.mean_external_permeability,
        };

        Some((left_coalition, right_coalition))
    }

    /// Friston Agential Fusion: Dissolves the separating boundaries between two
    /// distinct swarm coalitions, merging them into a single macro-identity
    /// when their internal tracking variables achieve complete structural alignment.
    pub fn execute_blanket_fusion(&self, other: &Self) -> Option<Self> {
        if self.mean_external_permeability < 0.85 {
            return None; // Mutual boundary permeability is too low to permit fusion
        }

        let mut unified_members = self.members.clone();
        for member in &other.members {
            if !unified_members.contains(member) {
                unified_members.push(member.clone());
            }
        }

        Some(SwarmCoalition {
            members: unified_members,
            internal_phi: (self.internal_phi + other.internal_phi) * 1.15, // Expanded integration capacity
            boundary_phi: (self.boundary_phi + other.boundary_phi) * 0.5,
            cohesion: (self.cohesion + other.cohesion) * 0.5,
            mean_internal_permeability: self.mean_internal_permeability,
            mean_external_permeability: self.mean_external_permeability,
        })
    }

    /// Levin Xenobot Grafting Invariant: Computes an algebraic translation hypervector
    /// that maps a foreign node's coordinate system directly into the native basis.
    /// Accepts a slice of paired anchor hypervectors exchanged during a coalition handshake
    /// and bundles their cross-bound inverses into a single universal translation operator.
    pub fn compute_xenobot_graft_transform(
        &self,
        native_anchors: &[symthaea_core::hdc::ContinuousHV],
        foreign_anchors: &[symthaea_core::hdc::ContinuousHV],
    ) -> Option<symthaea_core::hdc::ContinuousHV> {
        if native_anchors.is_empty() || native_anchors.len() != foreign_anchors.len() {
            return None;
        }

        let mut bound_pairs = Vec::with_capacity(native_anchors.len());

        for i in 0..native_anchors.len() {
            // Compute the algebraic inverse of the foreign coordinate anchor
            let foreign_inverse = foreign_anchors[i].inverse();
            // Bind the native anchor with the foreign inverse to capture the coordinate rotation delta
            bound_pairs.push(native_anchors[i].bind(&foreign_inverse));
        }

        // Bundle all coordinate deltas into a single composite Translation Operator Hypervector
        let pair_refs: Vec<&symthaea_core::hdc::ContinuousHV> = bound_pairs.iter().collect();
        Some(symthaea_core::hdc::ContinuousHV::bundle(&pair_refs))
    }

    /// Translates an incoming foreign hypervector into the native coordinate space
    /// using a pre-computed Xenobot Translation Operator via a single binding pass.
    pub fn translate_foreign_hypervector(
        &self,
        foreign_vector: &symthaea_core::hdc::ContinuousHV,
        graft_transform: &symthaea_core::hdc::ContinuousHV,
    ) -> symthaea_core::hdc::ContinuousHV {
        foreign_vector.bind(graft_transform)
    }

    /// Collective Φ = internal + permeability-weighted boundary.

    pub fn collective_phi(&self) -> f64 {
        self.internal_phi + self.mean_external_permeability * self.boundary_phi
    }

    /// Levin Swarm Invariant: Uses gap-junction transduction principles to sync a confused node's
    /// world-model with the collective consensus of its sovereign swarm coalition.
    pub fn transduce_gap_junction_alignment(
        &self,
        peer_beliefs: &[super::types::HiddenState],
        peer_models: &[super::generative_model::GenerativeModel],
        state_dim: usize,
    ) -> Option<(
        super::types::HiddenState,
        super::generative_model::GenerativeModel,
    )> {
        // Only allow gap-junction pooling if the coalition has achieved high internal boundary integration
        if self.members.len() < 2 || self.mean_internal_permeability < 0.6 {
            return None;
        }

        if peer_beliefs.is_empty() || peer_models.is_empty() {
            return None;
        }

        // 1. Average the consensus hidden belief states of uncorrupted neighboring peers
        let mut consensus_mean = vec![0.0; state_dim];
        let mut consensus_precision = vec![0.0; state_dim];
        let weight = 1.0 / peer_beliefs.len() as f64;

        for belief in peer_beliefs {
            for i in 0..state_dim {
                if i < belief.mean.len() && i < belief.precision.len() {
                    consensus_mean[i] += belief.mean[i] * weight;
                    consensus_precision[i] += belief.precision[i] * weight;
                }
            }
        }

        let mut hybrid_belief = super::types::HiddenState::new(state_dim);
        hybrid_belief.mean = consensus_mean;
        hybrid_belief.precision = consensus_precision;

        // 2. Clone a stable, healthy ancestral generative model from the peer pool cluster
        let consensus_model = peer_models[0].clone();

        Some((hybrid_belief, consensus_model))
    }

    /// Is this coalition conscious as a collective?
    ///
    /// Requires sufficient members, high internal coherence, and a well-defined
    /// blanket boundary (high internal permeability, low external permeability).
    pub fn is_conscious_collective(&self) -> bool {
        self.members.len() >= 3
            && self.internal_phi > 0.3
            && self.cohesion > 0.6
            && self.mean_internal_permeability > 0.7
            && self.mean_external_permeability < 0.5
    }

    /// Member count.
    pub fn size(&self) -> usize {
        self.members.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COALITION IDENTIFICATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute coalitions from a peer phi/permeability matrix.
///
/// Uses greedy agglomerative clustering via Union-Find: merge the two most
/// permeable peers, then iterate until no pair exceeds threshold.
///
/// # Arguments
///
/// - `peers`: Slice of (peer_id, phi) pairs
/// - `pairwise_permeability`: Edges as (index_i, index_j, permeability)
/// - `threshold`: Minimum permeability to merge (e.g. 0.7)
///
/// # Returns
///
/// Vector of [`SwarmCoalition`]s (only clusters with ≥ 2 members).
pub fn identify_coalitions(
    peers: &[(String, f64)],
    pairwise_permeability: &[(usize, usize, f64)],
    threshold: f64,
) -> Vec<SwarmCoalition> {
    let n = peers.len();
    if n < 2 {
        return Vec::new();
    }

    // Union-Find with path compression
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<usize> = vec![0; n];

    fn find(parent: &mut [usize], i: usize) -> usize {
        if parent[i] != i {
            parent[i] = find(parent, parent[i]);
        }
        parent[i]
    }

    fn union(parent: &mut [usize], rank: &mut [usize], a: usize, b: usize) {
        let (ra, rb) = (find(parent, a), find(parent, b));
        if ra == rb {
            return;
        }
        if rank[ra] < rank[rb] {
            parent[ra] = rb;
        } else if rank[ra] > rank[rb] {
            parent[rb] = ra;
        } else {
            parent[rb] = ra;
            rank[ra] += 1;
        }
    }

    // Sort edges by permeability descending
    let mut edges: Vec<(usize, usize, f64)> = pairwise_permeability.to_vec();
    edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Merge peers with permeability above threshold
    for &(i, j, perm) in &edges {
        if perm < threshold {
            break;
        }
        if i < n && j < n {
            union(&mut parent, &mut rank, i, j);
        }
    }

    // Extract clusters
    let mut clusters: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for i in 0..n {
        clusters.entry(find(&mut parent, i)).or_default().push(i);
    }

    clusters
        .values()
        .filter(|c| c.len() >= 2) // Singleton = no coalition
        .map(|members| {
            let member_ids: Vec<String> = members.iter().map(|&i| peers[i].0.clone()).collect();
            let internal_phi =
                members.iter().map(|&i| peers[i].1).sum::<f64>() / members.len() as f64;

            // Internal permeability: mean of within-cluster edges
            let within_edges: Vec<f64> = edges
                .iter()
                .filter(|(i, j, _)| members.contains(i) && members.contains(j))
                .map(|(_, _, p)| *p)
                .collect();

            let internal_perm = if within_edges.is_empty() {
                threshold // At least threshold (they merged)
            } else {
                within_edges.iter().sum::<f64>() / within_edges.len() as f64
            };

            // External permeability: mean of cross-cluster edges
            let external_edges: Vec<f64> = edges
                .iter()
                .filter(|(i, j, _)| {
                    (members.contains(i) && !members.contains(j))
                        || (!members.contains(i) && members.contains(j))
                })
                .map(|(_, _, p)| *p)
                .collect();

            let external_perm = if external_edges.is_empty() {
                0.0
            } else {
                external_edges.iter().sum::<f64>() / external_edges.len() as f64
            };

            // Cohesion = within-edges / (within + cross)
            let total_edges = within_edges.len() + external_edges.len();
            let cohesion = if total_edges > 0 {
                within_edges.len() as f64 / total_edges as f64
            } else {
                1.0
            };

            SwarmCoalition {
                members: member_ids,
                internal_phi,
                boundary_phi: 0.0, // Computed externally from cross-coalition phi
                cohesion,
                mean_internal_permeability: internal_perm.min(1.0),
                mean_external_permeability: external_perm,
            }
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Standard logistic sigmoid function.
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Linear interpolation: a + t × (b - a).
#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn default_partition() -> MarkovPartition {
        MarkovPartition {
            internal_dim: 16384,
            sensory_dim: 4,
            active_dim: 8,
        }
    }

    #[test]
    fn test_high_threat_thickens_blanket() {
        let mut op = MarkovBoundaryOperator::new(default_partition());
        let inputs = PermeabilityInputs {
            threat_level: 0.9,
            noradrenaline: 0.8,
            acetylcholine: 0.7,
            serotonin: 0.1,
            oxytocin: 0.1,
            flow_state: 0.0,
            peer_trust: 0.2,
        };
        // Run multiple cycles for EMA to converge from default 0.5
        for _ in 0..50 {
            op.compute_permeability(&inputs);
        }
        let perm = op.permeability();
        assert!(
            perm.sensory < 0.3,
            "High threat should close sensory blanket, got {}",
            perm.sensory
        );
        assert!(
            perm.active < 0.3,
            "High threat should close active blanket, got {}",
            perm.active
        );
    }

    #[test]
    fn test_high_safety_thins_blanket() {
        let mut op = MarkovBoundaryOperator::new(default_partition());
        let inputs = PermeabilityInputs {
            serotonin: 0.9,
            oxytocin: 0.8,
            flow_state: 0.9,
            threat_level: 0.0,
            noradrenaline: 0.1,
            acetylcholine: 0.2,
            peer_trust: 0.9,
        };
        // Run multiple cycles for EMA to converge from default 0.5
        for _ in 0..50 {
            op.compute_permeability(&inputs);
        }
        let perm = op.permeability();
        assert!(
            perm.sensory > 0.6,
            "High safety should open sensory blanket, got {}",
            perm.sensory
        );
        assert!(
            perm.active > 0.6,
            "High safety should open active blanket, got {}",
            perm.active
        );
    }

    #[test]
    fn test_permeability_floor_prevents_solipsism() {
        let mut op = MarkovBoundaryOperator::new(default_partition());
        // Maximum closure inputs
        let inputs = PermeabilityInputs {
            threat_level: 1.0,
            noradrenaline: 1.0,
            acetylcholine: 1.0,
            serotonin: 0.0,
            oxytocin: 0.0,
            flow_state: 0.0,
            peer_trust: 0.0,
        };
        let perm = op.compute_permeability(&inputs);
        assert!(
            perm.sensory >= PERMEABILITY_FLOOR,
            "Floor prevents total isolation"
        );
        assert!(
            perm.active >= PERMEABILITY_FLOOR,
            "Floor prevents total isolation"
        );
    }

    #[test]
    fn test_ceiling_prevents_dissolution() {
        let mut op = MarkovBoundaryOperator::new(default_partition());
        // Maximum opening inputs
        let inputs = PermeabilityInputs {
            serotonin: 1.0,
            oxytocin: 1.0,
            flow_state: 1.0,
            threat_level: 0.0,
            noradrenaline: 0.0,
            acetylcholine: 0.0,
            peer_trust: 1.0,
        };
        let perm = op.compute_permeability(&inputs);
        assert!(
            perm.sensory <= PERMEABILITY_CEILING,
            "Ceiling prevents dissolution"
        );
        assert!(
            perm.active <= PERMEABILITY_CEILING,
            "Ceiling prevents dissolution"
        );
    }

    #[test]
    fn test_ema_smoothing_prevents_oscillation() {
        let mut op = MarkovBoundaryOperator::new(default_partition());
        let safe = PermeabilityInputs {
            serotonin: 0.9,
            oxytocin: 0.8,
            flow_state: 0.8,
            ..Default::default()
        };
        let threat = PermeabilityInputs {
            threat_level: 0.9,
            noradrenaline: 0.8,
            acetylcholine: 0.7,
            ..Default::default()
        };
        // Alternate safe/threat — EMA should dampen
        for _ in 0..20 {
            op.compute_permeability(&safe);
            op.compute_permeability(&threat);
        }
        let perm = op.permeability();
        // Should settle near middle, not swing to extremes
        assert!(
            perm.effective > 0.15 && perm.effective < 0.85,
            "EMA should dampen oscillation: {}",
            perm.effective
        );
    }

    #[test]
    fn test_coalescence_detection() {
        let mut op = MarkovBoundaryOperator::new(default_partition());
        let safe = PermeabilityInputs {
            serotonin: 0.9,
            oxytocin: 0.8,
            flow_state: 0.8,
            ..Default::default()
        };
        // Not ready initially
        assert!(!op.coalescence_ready(0.6));
        // Stabilize at high permeability
        for _ in 0..100 {
            op.compute_permeability(&safe);
        }
        assert!(
            op.coalescence_ready(0.6),
            "Stable high permeability should be coalescence-ready"
        );
    }

    #[test]
    fn test_not_coalescence_ready_under_threat() {
        let mut op = MarkovBoundaryOperator::new(default_partition());
        let threat = PermeabilityInputs {
            threat_level: 0.8,
            noradrenaline: 0.7,
            ..Default::default()
        };
        for _ in 0..100 {
            op.compute_permeability(&threat);
        }
        assert!(
            !op.coalescence_ready(0.6),
            "Low permeability should NOT be coalescence-ready"
        );
    }

    #[test]
    fn test_gate_observation_attenuates_under_threat() {
        let mut op = MarkovBoundaryOperator::new(default_partition());
        let threat = PermeabilityInputs {
            threat_level: 0.9,
            noradrenaline: 0.8,
            acetylcholine: 0.7,
            ..Default::default()
        };
        // Stabilize under threat
        for _ in 0..50 {
            op.compute_permeability(&threat);
        }

        let obs = Observation::from_consciousness_state(0.8, 0.7, 0.6, 0.5);
        let prior = HiddenState::new(4); // mean = [0.5, 0.5, 0.5, 0.5]
        let gated = op.gate_observation(&obs, &prior);

        // Gated values should be closer to prior than original observation
        assert!(
            (gated.values[0] - 0.5).abs() < (0.8_f64 - 0.5).abs(),
            "Gated phi {} should be closer to prior 0.5 than raw 0.8",
            gated.values[0]
        );
    }

    #[test]
    fn test_gate_observation_passes_through_when_safe() {
        let mut op = MarkovBoundaryOperator::new(default_partition());
        let safe = PermeabilityInputs {
            serotonin: 0.9,
            oxytocin: 0.8,
            flow_state: 0.9,
            ..Default::default()
        };
        // Stabilize under safety
        for _ in 0..50 {
            op.compute_permeability(&safe);
        }

        let obs = Observation::from_consciousness_state(0.8, 0.7, 0.6, 0.5);
        let prior = HiddenState::new(4);
        let gated = op.gate_observation(&obs, &prior);

        // Gated values should be close to original observation
        assert!(
            (gated.values[0] - 0.8).abs() < 0.15,
            "Gated phi {} should be close to raw 0.8 when safe",
            gated.values[0]
        );
    }

    #[test]
    fn test_learning_rate_modulation() {
        let mut op = MarkovBoundaryOperator::new(default_partition());

        // Under threat: learning rate should be reduced
        let threat = PermeabilityInputs {
            threat_level: 0.9,
            noradrenaline: 0.8,
            ..Default::default()
        };
        for _ in 0..50 {
            op.compute_permeability(&threat);
        }
        let lr_threat = op.modulate_learning_rate(1.0);

        // Under safety: learning rate should be higher
        let mut op2 = MarkovBoundaryOperator::new(default_partition());
        let safe = PermeabilityInputs {
            serotonin: 0.9,
            oxytocin: 0.8,
            flow_state: 0.8,
            ..Default::default()
        };
        for _ in 0..50 {
            op2.compute_permeability(&safe);
        }
        let lr_safe = op2.modulate_learning_rate(1.0);

        assert!(
            lr_safe > lr_threat,
            "Safe LR {} should exceed threat LR {}",
            lr_safe,
            lr_threat
        );
        assert!(lr_threat >= 0.2, "Learning rate floor is 0.2");
    }

    #[test]
    fn test_coalition_identification_basic() {
        let peers = vec![
            ("alice".into(), 0.7),
            ("bob".into(), 0.6),
            ("carol".into(), 0.5),
            ("dave".into(), 0.4),
        ];
        let edges = vec![
            (0, 1, 0.9),  // alice-bob: high permeability
            (0, 2, 0.8),  // alice-carol: high
            (1, 2, 0.85), // bob-carol: high
            (0, 3, 0.3),  // alice-dave: low
            (1, 3, 0.2),  // bob-dave: low
            (2, 3, 0.25), // carol-dave: low
        ];
        let coalitions = identify_coalitions(&peers, &edges, 0.7);
        assert_eq!(coalitions.len(), 1, "Should form one coalition");
        assert_eq!(
            coalitions[0].members.len(),
            3,
            "Coalition should have alice, bob, carol"
        );
        assert!(
            !coalitions[0].members.contains(&"dave".to_string()),
            "Dave should be excluded"
        );
    }

    #[test]
    fn test_coalition_two_clusters() {
        let peers = vec![
            ("a".into(), 0.8),
            ("b".into(), 0.7),
            ("c".into(), 0.6),
            ("d".into(), 0.5),
        ];
        let edges = vec![
            (0, 1, 0.9), // a-b: high
            (2, 3, 0.9), // c-d: high
            (0, 2, 0.2), // a-c: low
            (1, 3, 0.1), // b-d: low
        ];
        let coalitions = identify_coalitions(&peers, &edges, 0.7);
        assert_eq!(coalitions.len(), 2, "Should form two coalitions");
    }

    #[test]
    fn test_conscious_collective_threshold() {
        let coalition = SwarmCoalition {
            members: vec!["a".into(), "b".into(), "c".into()],
            internal_phi: 0.5,
            boundary_phi: 0.2,
            cohesion: 0.8,
            mean_internal_permeability: 0.85,
            mean_external_permeability: 0.3,
        };
        assert!(coalition.is_conscious_collective());
        assert!(
            coalition.collective_phi() > coalition.internal_phi,
            "Boundary phi should add to collective"
        );
    }

    #[test]
    fn test_not_conscious_too_few_members() {
        let coalition = SwarmCoalition {
            members: vec!["a".into(), "b".into()],
            internal_phi: 0.8,
            boundary_phi: 0.0,
            cohesion: 1.0,
            mean_internal_permeability: 0.9,
            mean_external_permeability: 0.1,
        };
        assert!(
            !coalition.is_conscious_collective(),
            "Need ≥ 3 for conscious collective"
        );
    }

    #[test]
    fn test_not_conscious_low_cohesion() {
        let coalition = SwarmCoalition {
            members: vec!["a".into(), "b".into(), "c".into()],
            internal_phi: 0.5,
            boundary_phi: 0.2,
            cohesion: 0.3,
            mean_internal_permeability: 0.5,
            mean_external_permeability: 0.8,
        };
        assert!(
            !coalition.is_conscious_collective(),
            "Low cohesion + high external permeability = no collective consciousness"
        );
    }

    #[test]
    fn test_empty_peers_no_coalitions() {
        let coalitions = identify_coalitions(&[], &[], 0.7);
        assert!(coalitions.is_empty());
    }

    #[test]
    fn test_single_peer_no_coalition() {
        let peers = vec![("lonely".into(), 0.5)];
        let coalitions = identify_coalitions(&peers, &[], 0.7);
        assert!(coalitions.is_empty());
    }

    #[test]
    fn test_telemetry_generation() {
        let mut op = MarkovBoundaryOperator::new(default_partition());
        let inputs = PermeabilityInputs::default();
        op.compute_permeability(&inputs);

        let tel = op.telemetry(2);
        assert!(tel.effective_permeability > 0.0);
        assert_eq!(tel.coalition_count, 2);
    }

    #[test]
    fn test_trend_opens_under_safety() {
        let mut op = MarkovBoundaryOperator::new(default_partition());
        // Start under threat
        let threat = PermeabilityInputs {
            threat_level: 0.8,
            noradrenaline: 0.7,
            ..Default::default()
        };
        for _ in 0..40 {
            op.compute_permeability(&threat);
        }
        // Transition to safety
        let safe = PermeabilityInputs {
            serotonin: 0.9,
            oxytocin: 0.8,
            flow_state: 0.8,
            ..Default::default()
        };
        for _ in 0..40 {
            op.compute_permeability(&safe);
        }
        assert!(
            op.trend() > 0.0,
            "Trend should be positive (opening) after threat→safety transition"
        );
    }

    #[test]
    fn test_custom_alpha() {
        let op = MarkovBoundaryOperator::new(default_partition()).with_alpha(0.5);
        assert!((op.alpha - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sensory_precision_modulation() {
        let mut op = MarkovBoundaryOperator::new(default_partition());
        let threat = PermeabilityInputs {
            threat_level: 0.9,
            noradrenaline: 0.8,
            ..Default::default()
        };
        for _ in 0..50 {
            op.compute_permeability(&threat);
        }
        let modulated = op.modulate_sensory_precision(1.0);
        assert!(
            modulated < 0.5,
            "Under threat, sensory precision should be reduced, got {}",
            modulated
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROPERTY TESTS — Invariants under random inputs
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_permeability_inputs() -> impl Strategy<Value = PermeabilityInputs> {
        (
            0.0..=1.0_f64, // ACh
            0.0..=1.0_f64, // NE
            0.0..=1.0_f64, // 5-HT
            0.0..=1.0_f64, // Oxy
            0.0..=1.0_f64, // Threat
            0.0..=1.0_f64, // Peer trust
            0.0..=1.0_f64, // Flow
        )
            .prop_map(
                |(ach, ne, sht, oxy, threat, trust, flow)| PermeabilityInputs {
                    acetylcholine: ach,
                    noradrenaline: ne,
                    serotonin: sht,
                    oxytocin: oxy,
                    threat_level: threat,
                    peer_trust: trust,
                    flow_state: flow,
                },
            )
    }

    fn arb_topology_inputs() -> impl Strategy<Value = TopologyBoundaryInputs> {
        (
            0.0..=1.0_f64, // boundary_thickness
            0.0..=3.0_f64, // fiedler_value
            0_usize..=10,  // boundary_components
        )
            .prop_map(|(thickness, fiedler, components)| TopologyBoundaryInputs {
                boundary_thickness: thickness,
                fiedler_value: fiedler,
                boundary_components: components,
            })
    }

    proptest! {
        /// Permeability is always within [FLOOR, CEILING] for any inputs.
        #[test]
        fn permeability_bounded(inputs in arb_permeability_inputs()) {
            let mut op = MarkovBoundaryOperator::new(MarkovPartition {
                internal_dim: 16384, sensory_dim: 4, active_dim: 8,
            });
            let perm = op.compute_permeability(&inputs);
            prop_assert!(perm.sensory >= PERMEABILITY_FLOOR);
            prop_assert!(perm.sensory <= PERMEABILITY_CEILING);
            prop_assert!(perm.active >= PERMEABILITY_FLOOR);
            prop_assert!(perm.active <= PERMEABILITY_CEILING);
            prop_assert!(perm.effective >= 0.0);
            prop_assert!(perm.effective <= 1.0);
        }

        /// Effective permeability is the geometric mean of sensory × active.
        #[test]
        fn effective_is_geometric_mean(inputs in arb_permeability_inputs()) {
            let mut op = MarkovBoundaryOperator::new(MarkovPartition {
                internal_dim: 100, sensory_dim: 4, active_dim: 8,
            });
            // Run enough cycles for EMA to converge
            for _ in 0..100 {
                op.compute_permeability(&inputs);
            }
            let perm = op.permeability();
            let expected = (perm.sensory * perm.active).sqrt();
            prop_assert!((perm.effective - expected).abs() < 1e-10,
                "effective {} != sqrt(sensory {} * active {})",
                perm.effective, perm.sensory, perm.active);
        }

        /// Learning rate modulation never drops below 0.2 (floor).
        #[test]
        fn learning_rate_has_floor(inputs in arb_permeability_inputs()) {
            let mut op = MarkovBoundaryOperator::new(MarkovPartition {
                internal_dim: 100, sensory_dim: 4, active_dim: 8,
            });
            for _ in 0..50 {
                op.compute_permeability(&inputs);
            }
            let lr = op.modulate_learning_rate(1.0);
            prop_assert!(lr >= 0.2, "Learning rate {} below floor", lr);
            prop_assert!(lr <= 1.0, "Learning rate {} above 1.0", lr);
        }

        /// Gated observation is always between prior and raw observation.
        #[test]
        fn gated_observation_interpolates(
            inputs in arb_permeability_inputs(),
            phi in 0.0..=1.0_f64,
        ) {
            let mut op = MarkovBoundaryOperator::new(MarkovPartition {
                internal_dim: 100, sensory_dim: 4, active_dim: 8,
            });
            for _ in 0..50 {
                op.compute_permeability(&inputs);
            }
            let obs = Observation::from_consciousness_state(phi, 0.5, 0.5, 0.5);
            let prior = HiddenState::new(4); // mean = [0.5, ...]
            let gated = op.gate_observation(&obs, &prior);

            for (i, &gval) in gated.values.iter().enumerate() {
                let raw = obs.values[i];
                let prv = prior.mean.get(i).copied().unwrap_or(0.5);
                let lo = raw.min(prv);
                let hi = raw.max(prv);
                prop_assert!(
                    gval >= lo - 1e-10 && gval <= hi + 1e-10,
                    "Gated value {} not between raw {} and prior {} (dim {})",
                    gval, raw, prv, i
                );
            }
        }

        /// Topology constraints never push permeability outside bounds.
        #[test]
        fn topology_preserves_bounds(
            inputs in arb_permeability_inputs(),
            topo in arb_topology_inputs(),
        ) {
            let mut op = MarkovBoundaryOperator::new(MarkovPartition {
                internal_dim: 100, sensory_dim: 4, active_dim: 8,
            });
            for _ in 0..50 {
                op.compute_permeability(&inputs);
            }
            op.apply_topology_constraints(&topo);
            let perm = op.permeability();
            prop_assert!(perm.sensory >= PERMEABILITY_FLOOR,
                "Sensory {} below floor after topology", perm.sensory);
            prop_assert!(perm.active >= PERMEABILITY_FLOOR,
                "Active {} below floor after topology", perm.active);
            prop_assert!(perm.effective >= 0.0);
        }

        /// Coalition identification never panics and produces valid coalitions.
        #[test]
        fn coalition_identification_safe(
            n_peers in 0_usize..=20,
            threshold in 0.0..=1.0_f64,
        ) {
            let peers: Vec<(String, f64)> = (0..n_peers)
                .map(|i| (format!("peer_{}", i), (i as f64 / n_peers.max(1) as f64)))
                .collect();

            let mut edges = Vec::new();
            for i in 0..n_peers {
                for j in (i + 1)..n_peers {
                    let perm = 1.0 - ((peers[i].1 - peers[j].1).abs());
                    edges.push((i, j, perm));
                }
            }

            let coalitions = identify_coalitions(&peers, &edges, threshold);

            // Every coalition has at least 2 members
            for c in &coalitions {
                prop_assert!(c.members.len() >= 2,
                    "Coalition with {} members", c.members.len());
                // Cohesion is bounded
                prop_assert!(c.cohesion >= 0.0 && c.cohesion <= 1.0);
                // Internal phi is bounded
                prop_assert!(c.internal_phi >= 0.0 && c.internal_phi <= 1.0);
            }

            // No peer appears in multiple coalitions
            let mut all_members: Vec<&str> = coalitions.iter()
                .flat_map(|c| c.members.iter().map(|s| s.as_str()))
                .collect();
            let before = all_members.len();
            all_members.sort();
            all_members.dedup();
            prop_assert_eq!(before, all_members.len(),
                "Duplicate peer in coalitions");
        }

        /// Collective Φ of a conscious coalition exceeds internal Φ.
        #[test]
        fn collective_phi_exceeds_internal(
            internal_phi in 0.1..=1.0_f64,
            boundary_phi in 0.01..=0.5_f64,
            ext_perm in 0.01..=0.5_f64,
        ) {
            let c = SwarmCoalition {
                members: vec!["a".into(), "b".into(), "c".into()],
                internal_phi,
                boundary_phi,
                cohesion: 0.8,
                mean_internal_permeability: 0.9,
                mean_external_permeability: ext_perm,
            };
            prop_assert!(c.collective_phi() >= internal_phi,
                "Collective Φ {} should >= internal Φ {}",
                c.collective_phi(), internal_phi);
        }
    }
}
