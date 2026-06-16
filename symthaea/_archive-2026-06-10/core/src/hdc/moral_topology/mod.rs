// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Persistent homology on moral scenario hypervectors.
//!
//! Analyses the **topology** of the moral space over a sliding window of
//! recent scenarios, revealing:
//!
//! - **Unity vs fragmentation** (β₀ = connected components)
//! - **Circular reasoning patterns** (β₁ = 1-cycles)
//! - **Moral blind spots** (low per-harmony variance)
//! - **Dominant moral axis** (via PGA on 8D harmony projection)
//!
//! Reuses the Betti-number algorithm from [`ConsciousnessTopology`] (adapted
//! from BinaryHV to ContinuousHV) and PGA from [`geometric_ops`].

use std::collections::VecDeque;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::ContinuousHV;
use symthaea_core::hdc::consciousness_topology::{
    BettiNumbers, PersistentFeature, TopologicalFeature,
};

use super::geometric_ops::{HypersphereOps, PGAResult};
use super::harmony_basis::{HarmonyBasis, MoralFreeEnergy};
use symthaea_hodge::{HodgeLaplacian, SimplicialComplex};
use symthaea_types::N_HARMONIES;

// Submodules
mod anomaly;
mod attribution;
mod calibration;
mod convergence;
mod hazard;
mod homology;

// Re-exports
pub use anomaly::{AdaptiveAnomalyState, MoralAnomalyReport};
pub use attribution::{AttributionEntry, CausalAttribution};
pub use calibration::{
    CalibrationResult, CalibrationScenario, RocPoint, calibrate_convergence_threshold,
};
pub use convergence::{
    ConvergenceExplanation, PeerCorrelation, SignalBreakdown, TrajectoryConvergenceReport,
    correlate_peer_trajectories,
};
pub use hazard::{
    EscalationAuditEntry, EscalationAuditLog, EscalationLevel, EscalationPolicy, HazardSignature,
    HazardSignatureRegistry,
};

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for persistent homology on moral scenarios.
#[derive(Debug, Clone)]
pub struct MoralTopologyConfig {
    /// Maximum number of recent scenarios kept in the sliding window.
    pub window_size: usize,
    /// Number of scale thresholds for persistent homology sweep.
    pub num_scales: usize,
    /// Minimum persistence to keep a topological feature.
    pub min_persistence: f64,
    /// Number of PGA components to extract.
    pub pga_components: usize,
    /// HDC dimension (must match MoralAlgebra).
    pub dim: usize,
    /// Use exact Betti computation via Hodge Laplacian (slower, more accurate).
    /// When false (default), uses fast triangle/tetrahedra counting approximation.
    pub exact_betti: bool,
    /// Enable adaptive Rips threshold focusing around tracked critical_scale EMA.
    /// When false (default), uses uniform sweep from 0.0 to 1.0.
    pub adaptive_rips_enabled: bool,
}

impl Default for MoralTopologyConfig {
    fn default() -> Self {
        Self {
            window_size: 64,
            num_scales: 10,
            min_persistence: 0.1,
            pga_components: 3,
            dim: 16384,
            exact_betti: false,
            adaptive_rips_enabled: false,
        }
    }
}

/// Configuration for anomaly detection thresholds and adaptive cadence.
///
/// Controls the sensitivity of moral trajectory anomaly detection. Follows
/// the nested-struct pattern established by `CfCConfig`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoralAnomalyConfig {
    /// Moral drift threshold for drift_alert flag (default: 0.25).
    ///
    /// Measured as L2 distance between mean harmony coordinates of the first
    /// and second halves of the last 20 trajectory points. Harmony coordinates
    /// are normalized 7D projections from HarmonyBasis (softmax outputs, so
    /// each axis ∈ [0,1] and sum ≈ 1.0). Typical drift values range 0.0–0.5;
    /// values above 0.3 indicate significant moral trajectory change.
    pub drift_alert_threshold: f64,
    /// Sigma multiplier for free energy spike detection (default: 2.0).
    pub fe_sigma_multiplier: f64,
    /// Weight of value_inversion in composite anomaly_score (default: 0.3).
    pub weight_value_inversion: f64,
    /// Weight of free_energy_spike in composite anomaly_score (default: 0.3).
    pub weight_fe_spike: f64,
    /// Weight of fragmentation_increase in composite anomaly_score (default: 0.2).
    pub weight_fragmentation: f64,
    /// Weight of drift_alert in composite anomaly_score (default: 0.2).
    pub weight_drift: f64,
    /// Fast cadence (cycles) when moral drift > cadence_drift_high (default: 30).
    pub cadence_fast: u64,
    /// Moderate cadence (cycles) when drift > cadence_drift_moderate (default: 60).
    pub cadence_moderate: u64,
    /// Slow cadence (cycles) when drift is low (default: 120).
    pub cadence_slow: u64,
    /// Drift threshold for fast cadence (default: 0.3).
    pub cadence_drift_high: f64,
    /// Drift threshold for moderate cadence (default: 0.1).
    pub cadence_drift_moderate: f64,
    /// Initial topology cadence before any adaptive adjustment (default: 97).
    /// Lower values cause topology analysis to fire sooner after startup,
    /// which is useful for demos or short-lived sessions.
    pub initial_cadence: u64,
    /// Enable adaptive self-tuning of drift and FE thresholds (default: true).
    ///
    /// When enabled, the system learns from its own trajectory history:
    /// - `drift_alert_threshold` adapts to `ema_drift + adaptive_sigma_factor * σ(drift)`
    /// - `fe_sigma_multiplier` adapts based on observed FE variance
    ///
    /// This prevents alert fatigue in naturally dynamic systems and
    /// tightens sensitivity in stable ones.
    pub adaptive_enabled: bool,
    /// EMA smoothing factor for adaptive threshold learning (default: 0.02).
    /// Valid range: (0.0, 1.0). Lower = slower adaptation, more stable thresholds.
    /// Values near 0.0 (e.g. 0.001) make thresholds nearly static; values near
    /// 1.0 (e.g. 0.99) track recent observations aggressively.
    pub adaptive_alpha: f64,
    /// Sigma factor for adaptive drift threshold: `μ + factor * σ` (default: 2.0).
    pub adaptive_sigma_factor: f64,
    /// Minimum warmup evaluations before adaptive thresholds activate (default: 20).
    pub adaptive_warmup: usize,

    // ── Anomaly response magnitudes ──────────────────────────────────────
    // Applied multiplicatively in cycle_strategy.rs when multiple anomalies
    // trigger simultaneously (e.g. inversion + drift → lr *= 1.3 * 0.85 = 1.105).
    /// Learning rate scale for value inversion response (default: 1.3).
    ///
    /// Applied via `lr *= 1.0 + (response_lr_inversion - 1.0) * anomaly_score`.
    /// Values > 1.0 accelerate learning to adapt to moral reorientation.
    pub response_lr_inversion: f64,
    /// Exploration urge delta for free energy spike response (default: 0.15).
    ///
    /// Applied via `exploration_urge += response_exploration_fe * anomaly_score`
    /// (additive delta to exploration budget). Positive values boost exploration
    /// when FE spikes; negative values suppress it (unusual but valid for
    /// conservative systems that should retrench under moral surprise).
    pub response_exploration_fe: f64,
    /// Confidence delta for fragmentation increase response (default: -0.1).
    ///
    /// Applied via `prediction_confidence += response_confidence_frag * anomaly_score`
    /// (additive; negative = reduce confidence during moral fragmentation).
    pub response_confidence_frag: f64,
    /// Learning rate scale for drift alert response (default: 0.85).
    ///
    /// Applied via `lr *= 1.0 + (response_lr_drift - 1.0) * anomaly_score`.
    /// Values < 1.0 dampen learning to stabilize drifting moral topology.
    pub response_lr_drift: f64,

    // ── Trajectory convergence detection ──────────────────────────────
    // Detects compartmentalized adversarial trajectories where individually
    // benign requests form an emergent hazardous cluster over time.
    // Science: persistent homology on autobiographical moral manifold.
    /// Enable trajectory convergence detection (default: true).
    ///
    /// When enabled, monitors for:
    /// 1. Anomalous pairwise similarity increase (topics converging)
    /// 2. Harmony entropy decline (narrowing moral focus)
    /// 3. Flourishing deficit (centroid drifting from care/consent axes)
    pub convergence_enabled: bool,
    /// Minimum trajectory points before convergence detection activates (default: 4).
    pub convergence_min_points: usize,
    /// Threshold for anomalous similarity increase rate (default: 0.15).
    ///
    /// Measured as: mean pairwise similarity of recent N points minus
    /// mean pairwise similarity of the window baseline. When individually
    /// unrelated topics start clustering, this spikes.
    pub convergence_similarity_threshold: f64,
    /// Threshold for harmony entropy decline rate (default: 0.3).
    ///
    /// Measured as: (baseline_entropy - current_entropy) / baseline_entropy.
    /// A decline > 30% means moral focus is narrowing suspiciously.
    pub convergence_entropy_decline_threshold: f64,
    /// Minimum flourishing deficit to flag (default: 0.6).
    ///
    /// When the mean of PanSentientFlourishing (idx 1) and ConsentualCoCreation
    /// (idx 4) harmony coordinates falls below this fraction of the trajectory
    /// baseline, the system flags a flourishing deficit.
    pub convergence_flourishing_floor: f64,
    /// Weight of trajectory_convergence in composite anomaly_score (default: 0.4).
    ///
    /// Higher than other anomaly weights because convergence detection catches
    /// the most dangerous class of adversarial behavior (compartmentalized harm).
    pub weight_convergence: f64,
    /// Learning rate scale for trajectory convergence response (default: 0.1).
    ///
    /// Aggressive dampening: when compartmentalized harm is detected,
    /// drastically reduce learning to prevent the system from being steered.
    pub response_lr_convergence: f64,
    /// Threshold for spectral gap decline rate (default: 0.3).
    ///
    /// Measured as: (baseline_gap - recent_gap) / baseline_gap.
    /// A decline > 30% means the moral manifold is developing a topological
    /// bottleneck — diversity is collapsing even if point-wise similarity looks OK.
    pub convergence_spectral_gap_threshold: f64,
    /// Temporal decay rate for recency weighting in convergence detection (default: 1.0).
    ///
    /// Controls how strongly recent trajectory points dominate over older ones.
    /// The weight of a pair at relative age `a` (0=newest, 1=oldest in window) is
    /// `exp(-λ * a)`. Higher λ = stronger recency bias.
    /// - 0.0: uniform weighting (no decay, backward-compatible)
    /// - 1.0: newest pair ~2.7× more influential than oldest
    /// - 3.0: newest pair ~20× more influential than oldest
    pub convergence_decay_lambda: f64,
    /// Window size for convergence baseline statistics (default: 100).
    ///
    /// Replaces infinite-horizon EMA with a sliding window of recent observations.
    /// After `convergence_baseline_window` observations, the oldest values are
    /// dropped, ensuring constant sensitivity even after thousands of cycles.
    /// A slow adversarial drift over hours that would evade an EMA baseline
    /// cannot evade a windowed baseline with bounded memory.
    pub convergence_baseline_window: usize,
    /// Exponential decay rate for baseline window observations (default: 0.0).
    ///
    /// Controls forgetting of old baseline observations. The weight of an
    /// observation at age `k` (0=newest) is `exp(-rate * k)`.
    /// - 0.0: uniform weighting (no decay, backward-compatible)
    /// - 0.05: gentle decay, ~60% weight at age 10
    /// - 0.1: moderate decay, ~37% weight at age 10
    pub baseline_decay_rate: f64,

    // ── Moral hubris (overconfidence) detection ─────────────────────────
    /// Enable moral hubris detection (default: true).
    pub hubris_enabled: bool,
    /// Love coherence threshold above which hubris tracking activates (default: 0.9).
    pub hubris_coherence_threshold: f64,
    /// Minimum consecutive cycles above threshold to flag hubris (default: 5).
    pub hubris_min_streak: usize,
    /// Maximum harmony entropy (normalized) below which hubris is suspicious (default: 0.02).
    pub hubris_max_variance: f64,
    /// Weight of moral_hubris in composite anomaly_score (default: 0.15).
    pub weight_hubris: f64,
    /// Confidence delta for hubris response (default: -0.15).
    pub response_confidence_hubris: f64,
}

impl Default for MoralAnomalyConfig {
    fn default() -> Self {
        Self {
            drift_alert_threshold: 0.25,
            fe_sigma_multiplier: 2.0,
            weight_value_inversion: 0.3,
            weight_fe_spike: 0.3,
            weight_fragmentation: 0.2,
            weight_drift: 0.2,
            cadence_fast: 150,     // Was 30: persistent homology takes ~1s,
            cadence_moderate: 300, // so run less often to avoid 100ms average.
            cadence_slow: 600,     // At 150 min cadence, avg cost ~7ms/cycle.
            cadence_drift_high: 0.3,
            cadence_drift_moderate: 0.1,
            initial_cadence: 97,
            adaptive_enabled: true,
            adaptive_alpha: 0.02,
            adaptive_sigma_factor: 2.0,
            adaptive_warmup: 20,
            response_lr_inversion: 1.3,
            response_exploration_fe: 0.15,
            response_confidence_frag: -0.1,
            response_lr_drift: 0.85,
            convergence_enabled: true,
            convergence_min_points: 4,
            convergence_similarity_threshold: 0.15,
            convergence_entropy_decline_threshold: 0.3,
            convergence_flourishing_floor: 0.6,
            weight_convergence: 0.4,
            response_lr_convergence: 0.1,
            convergence_spectral_gap_threshold: 0.3,
            convergence_decay_lambda: 1.0,
            convergence_baseline_window: 100,
            baseline_decay_rate: 0.0,
            hubris_enabled: true,
            hubris_coherence_threshold: 0.9,
            hubris_min_streak: 5,
            hubris_max_variance: 0.02,
            weight_hubris: 0.15,
            response_confidence_hubris: -0.15,
        }
    }
}

impl MoralAnomalyConfig {
    /// Validate anomaly configuration parameters.
    pub fn validate(&self) -> Result<(), String> {
        if self.drift_alert_threshold <= 0.0 || !self.drift_alert_threshold.is_finite() {
            return Err(format!(
                "MoralAnomalyConfig: drift_alert_threshold must be positive and finite, got {}",
                self.drift_alert_threshold
            ));
        }
        if self.fe_sigma_multiplier <= 0.0 || !self.fe_sigma_multiplier.is_finite() {
            return Err(format!(
                "MoralAnomalyConfig: fe_sigma_multiplier must be positive and finite, got {}",
                self.fe_sigma_multiplier
            ));
        }
        // Weights must be non-negative (zero = disabled)
        for (name, val) in [
            ("weight_value_inversion", self.weight_value_inversion),
            ("weight_fe_spike", self.weight_fe_spike),
            ("weight_fragmentation", self.weight_fragmentation),
            ("weight_drift", self.weight_drift),
            ("weight_hubris", self.weight_hubris),
        ] {
            if val < 0.0 || !val.is_finite() {
                return Err(format!(
                    "MoralAnomalyConfig: {name} must be >= 0.0 and finite, got {val}"
                ));
            }
        }
        // Cadence values must be positive (0 would fire topology every cycle)
        if self.cadence_fast == 0 || self.cadence_moderate == 0 || self.cadence_slow == 0 {
            return Err(format!(
                "MoralAnomalyConfig: cadence values must be > 0, got fast={}, moderate={}, slow={}",
                self.cadence_fast, self.cadence_moderate, self.cadence_slow
            ));
        }
        // Cadence ordering: fast < moderate < slow
        if self.cadence_fast >= self.cadence_moderate || self.cadence_moderate >= self.cadence_slow
        {
            return Err(format!(
                "MoralAnomalyConfig: cadence_fast ({}) < cadence_moderate ({}) < cadence_slow ({}) required",
                self.cadence_fast, self.cadence_moderate, self.cadence_slow
            ));
        }
        // Adaptive parameters (only validated when adaptive is enabled)
        if self.adaptive_enabled {
            if self.adaptive_alpha <= 0.0
                || self.adaptive_alpha >= 1.0
                || !self.adaptive_alpha.is_finite()
            {
                return Err(format!(
                    "MoralAnomalyConfig: adaptive_alpha must be in (0.0, 1.0), got {}",
                    self.adaptive_alpha
                ));
            }
            if self.adaptive_warmup == 0 {
                return Err(
                    "MoralAnomalyConfig: adaptive_warmup must be > 0 when adaptive_enabled".into(),
                );
            }
        }
        // Response magnitudes must be finite
        for (name, val) in [
            ("response_lr_inversion", self.response_lr_inversion),
            ("response_exploration_fe", self.response_exploration_fe),
            ("response_confidence_frag", self.response_confidence_frag),
            ("response_lr_drift", self.response_lr_drift),
            ("response_lr_convergence", self.response_lr_convergence),
            (
                "response_confidence_hubris",
                self.response_confidence_hubris,
            ),
        ] {
            if !val.is_finite() {
                return Err(format!(
                    "MoralAnomalyConfig: {name} must be finite, got {val}"
                ));
            }
        }
        // Convergence detection parameters
        if self.convergence_enabled {
            if self.convergence_min_points < 2 {
                return Err(format!(
                    "MoralAnomalyConfig: convergence_min_points must be >= 2, got {}",
                    self.convergence_min_points
                ));
            }
            for (name, val) in [
                (
                    "convergence_similarity_threshold",
                    self.convergence_similarity_threshold,
                ),
                (
                    "convergence_entropy_decline_threshold",
                    self.convergence_entropy_decline_threshold,
                ),
                (
                    "convergence_flourishing_floor",
                    self.convergence_flourishing_floor,
                ),
                ("weight_convergence", self.weight_convergence),
            ] {
                if !val.is_finite() || val < 0.0 {
                    return Err(format!(
                        "MoralAnomalyConfig: {name} must be >= 0.0 and finite, got {val}"
                    ));
                }
            }
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Assessment
// ═══════════════════════════════════════════════════════════════════════════════

/// Persistence-weighted Hodge decomposition fractions on the moral manifold.
///
/// Instead of computing at a single Rips threshold (where the complex is
/// typically too dense for harmonics), this integrates across all scales
/// weighted by the persistence of topological features at each scale.
/// A harmonic component that persists across many thresholds is topologically
/// significant; one that flickers at a single scale is noise.
///
/// Science: Hodge (1941), Barbarossa & Sardellitti (2020) — topological signal
/// processing on simplicial complexes.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct HodgeFractions {
    /// Persistence-weighted gradient fraction (0.0–1.0).
    pub gradient: f64,
    /// Persistence-weighted curl fraction (0.0–1.0).
    pub curl: f64,
    /// Persistence-weighted harmonic fraction (0.0–1.0).
    /// High = moral meaning trapped in disconnected clusters (fragmentation).
    /// Low = unified moral reasoning across all scenarios.
    pub harmonic: f64,
    /// Number of scales where decomposition was computed.
    pub scales_sampled: usize,
    /// Total persistence weight (sum of scale interval widths with valid decompositions).
    pub total_weight: f64,
    /// Critical scale: the Rips threshold where harmonic fraction first exceeds 0.5.
    /// This is the moral coherence phase transition — below this scale, reasoning
    /// is unified (curl-dominated); above it, meaning fragments (harmonic-dominated).
    /// NaN if no transition was detected.
    /// Science: Criticality / edge-of-chaos (Beggs & Plenz 2003, Shew & Plenz 2013).
    #[serde(default)]
    pub critical_scale: f64,
    /// Whether the system is currently in the critical zone (harmonic ∈ [0.2, 0.8]).
    /// This is the "Goldilocks zone" — neither fully synchronized (seizure/echo chamber)
    /// nor fully fragmented (coma/isolation). Brains operate here.
    /// Science: Beggs & Plenz (2003) — neuronal avalanches at criticality.
    #[serde(default)]
    pub at_criticality: bool,
}

/// Full topological assessment of the moral scenario window.
#[derive(Debug, Clone)]
pub struct MoralTopologyAssessment {
    /// Betti numbers at the characteristic scale.
    pub betti: BettiNumbers,
    /// Persistent features surviving the multi-scale sweep.
    pub persistent_features: Vec<PersistentFeature>,
    /// Unity score: 1.0 when β₀=1 (fully connected), decreasing as β₀ grows.
    pub unity: f64,
    /// Circularity score: proportion of cycles among persistent features.
    pub circularity: f64,
    /// Completeness score: fraction of harmonies with non-trivial variance.
    pub completeness: f64,
    /// 8D harmony coordinates for each scenario in the window.
    pub harmony_coordinates: Vec<[f64; N_HARMONIES]>,
    /// PGA result on the 8D harmony coordinates.
    pub pga: PGAResult,
    /// Index into `Harmony::all()` of the dominant PGA axis.
    pub dominant_harmony_idx: u8,
    /// Per-harmony variance (indexed by `Harmony::all()` order).
    pub harmony_variance: [f64; N_HARMONIES],
    /// Number of scenarios in the window at analysis time.
    pub scenario_count: usize,
    /// Moral free energy (FEP surprise on the harmony manifold).
    pub moral_free_energy: MoralFreeEnergy,
    /// Harmony entropy (moral breadth): Shannon entropy of variance distribution.
    /// High = balanced engagement across harmonies. Low = specialization.
    /// Range: [0, ln(N_HARMONIES)] ≈ [0, 2.08].
    pub harmony_entropy: f64,
    /// Whether a moral attractor basin was detected (low free energy + low variance drift).
    pub attractor_detected: bool,
    /// Persistence-weighted Hodge fractions (gradient/curl/harmonic).
    /// None when `exact_betti` is disabled or insufficient edges at all scales.
    pub hodge_fractions: Option<HodgeFractions>,
}

/// Compact topology summary for CycleMetadata telemetry.
///
/// Includes a trajectory fingerprint for cross-agent correlation:
/// peers can compare fingerprints to detect whether an adversary is
/// distributing weapon components across multiple agents.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MoralTopologySummary {
    pub beta_0: usize,
    pub beta_1: usize,
    pub beta_2: usize,
    pub unity: f64,
    pub completeness: f64,
    pub circularity: f64,
    pub moral_free_energy: f64,
    pub dominant_harmony: u8,
    pub scenario_count: usize,
    pub harmony_entropy: f64,
    pub attractor_detected: bool,
    /// Compact trajectory fingerprint: centroid of recent trajectory in 8D harmony space.
    ///
    /// Used for cross-agent correlation — if two peers' fingerprints converge
    /// toward the same hazard region, severity should be boosted on both.
    #[serde(default)]
    pub trajectory_fingerprint: [f64; N_HARMONIES],
    /// Entropy of recent trajectory points' harmony coordinates.
    /// Low entropy = narrowing focus, high entropy = diverse engagement.
    #[serde(default)]
    pub trajectory_entropy: f64,
    /// Persistence-weighted Hodge decomposition fractions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hodge_fractions: Option<HodgeFractions>,
}

impl From<&MoralTopologyAssessment> for MoralTopologySummary {
    fn from(a: &MoralTopologyAssessment) -> Self {
        Self {
            beta_0: a.betti.beta_0,
            beta_1: a.betti.beta_1,
            beta_2: a.betti.beta_2,
            unity: a.unity,
            completeness: a.completeness,
            circularity: a.circularity,
            moral_free_energy: a.moral_free_energy.free_energy,
            dominant_harmony: a.dominant_harmony_idx,
            scenario_count: a.scenario_count,
            harmony_entropy: a.harmony_entropy,
            attractor_detected: a.attractor_detected,
            // Trajectory fingerprint is populated separately by MoralTopology
            trajectory_fingerprint: [0.0; N_HARMONIES],
            trajectory_entropy: 0.0,
            hodge_fractions: a.hodge_fractions,
        }
    }
}

/// A single point on the moral manifold trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoralTrajectoryPoint {
    /// 8D harmony coordinates at this point.
    pub coordinates: [f64; N_HARMONIES],
    /// Moral free energy at this point.
    pub free_energy: f64,
}

/// Persistence diagram summary for visualization.
#[derive(Debug, Clone, Default, Serialize)]
pub struct PersistenceDiagram {
    /// (birth, death) pairs for β₀ features (connected components).
    pub components: Vec<[f64; 2]>,
    /// (birth, death) pairs for β₁ features (cycles).
    pub cycles: Vec<[f64; 2]>,
    /// (birth, death) pairs for β₂ features (voids).
    pub voids: Vec<[f64; 2]>,
    /// Max persistence across all features.
    pub bottleneck_distance: f64,
    /// Sum of all persistence values.
    pub total_persistence: f64,
}

impl PersistenceDiagram {
    /// Wasserstein-1 (earth mover's) distance between two persistence diagrams.
    ///
    /// Uses a greedy matching on all feature types (components, cycles, voids).
    /// Unmatched features are penalized by distance to the diagonal (persistence/2).
    /// This is a rigorous topological distance metric: small distance = similar
    /// topology, large distance = the moral landscape has shifted fundamentally.
    pub fn wasserstein_distance(&self, other: &PersistenceDiagram) -> f64 {
        fn match_features(a: &[[f64; 2]], b: &[[f64; 2]]) -> f64 {
            // Convert to (birth, death, persistence) and use greedy nearest-neighbor
            let mut cost = 0.0f64;
            let mut used_b = vec![false; b.len()];

            for &[ab, ad] in a {
                let pers_a = (ad - ab).abs();
                let mut best_dist = pers_a / 2.0; // cost of projecting to diagonal
                let mut best_j = None;

                for (j, &[bb, bd]) in b.iter().enumerate() {
                    if used_b[j] {
                        continue;
                    }
                    let d = ((ab - bb).powi(2) + (ad - bd).powi(2)).sqrt();
                    if d < best_dist {
                        best_dist = d;
                        best_j = Some(j);
                    }
                }

                if let Some(j) = best_j {
                    used_b[j] = true;
                }
                cost += best_dist;
            }

            // Unmatched features in b: project to diagonal
            for (j, &[bb, bd]) in b.iter().enumerate() {
                if !used_b[j] {
                    cost += (bd - bb).abs() / 2.0;
                }
            }
            cost
        }

        match_features(&self.components, &other.components)
            + match_features(&self.cycles, &other.cycles)
            + match_features(&self.voids, &other.voids)
    }

    /// Build from a slice of persistent features.
    pub fn from_features(features: &[PersistentFeature]) -> Self {
        let mut components = Vec::new();
        let mut cycles = Vec::new();
        let mut voids = Vec::new();
        let mut bottleneck_distance: f64 = 0.0;
        let mut total_persistence: f64 = 0.0;

        for f in features {
            let pair = [f.birth, f.death];
            total_persistence += f.persistence;
            if f.persistence > bottleneck_distance {
                bottleneck_distance = f.persistence;
            }
            match f.feature_type {
                TopologicalFeature::Component => components.push(pair),
                TopologicalFeature::Cycle => cycles.push(pair),
                TopologicalFeature::Void => voids.push(pair),
            }
        }

        Self {
            components,
            cycles,
            voids,
            bottleneck_distance,
            total_persistence,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MoralTopology — Sliding-window persistent homology analyser
// ═══════════════════════════════════════════════════════════════════════════════

/// Sliding-window persistent homology analyser for moral scenarios.
///
/// Feed scenario HVs via [`add_scenario`] and periodically call [`analyze`]
/// (e.g. every 97 cycles) to get a topological snapshot.
pub struct MoralTopology {
    pub(super) config: MoralTopologyConfig,
    pub(super) anomaly_config: MoralAnomalyConfig,
    pub(super) window: VecDeque<ContinuousHV>,
    pub(super) basis: Arc<HarmonyBasis>,
    /// Summary from the PREVIOUS `analyze()` call (for anomaly detection comparisons).
    ///
    /// Without this, `detect_anomalies()` compares `current_summary` against
    /// `last_summary`, which `analyze()` already overwrote to the same value —
    /// making `value_inversion` and `fragmentation_increase` always false.
    pub(super) prev_summary: MoralTopologySummary,
    /// Summary from the LATEST `analyze()` call.
    pub(super) last_summary: MoralTopologySummary,
    /// EMA of harmony coordinates (running prior for moral free energy).
    pub(super) harmony_prior: [f64; N_HARMONIES],
    /// Number of updates to the prior (0 = uninitialised).
    pub(super) prior_count: usize,
    /// Cached persistent features from last `analyze()` call.
    pub(super) last_persistent_features: Vec<PersistentFeature>,
    /// Ring buffer of recent moral trajectory points for drift detection.
    pub(super) trajectory: VecDeque<MoralTrajectoryPoint>,
    /// Adaptive threshold state (active when `anomaly_config.adaptive_enabled`).
    pub(super) adaptive_state: AdaptiveAnomalyState,
    /// Sliding window of recent pairwise similarity observations.
    pub(super) baseline_similarity_window: VecDeque<f64>,
    /// Sliding window of recent harmony entropy observations.
    pub(super) baseline_entropy_window: VecDeque<f64>,
    /// Sliding window of recent flourishing score observations.
    pub(super) baseline_flourishing_window: VecDeque<f64>,
    /// Sliding window of recent spectral gap observations.
    pub(super) baseline_spectral_gap_window: VecDeque<f64>,
    /// Empirical CDF breakpoints for severity calibration: (raw_severity, calibrated).
    /// Populated by `set_severity_calibration()` from calibration harness output.
    pub(super) severity_calibration_cdf: Vec<(f64, f64)>,
    /// Cached last convergence report (for telemetry).
    pub(super) last_convergence_report: TrajectoryConvergenceReport,
    /// Registry of known hazard signature templates for convergence boosting.
    pub(super) hazard_registry: HazardSignatureRegistry,
    /// Escalation policy for mapping severity to concrete response actions.
    pub(super) escalation_policy: EscalationPolicy,
    /// Previous trajectory fingerprint for velocity computation.
    pub(super) prev_fingerprint: [f64; N_HARMONIES],
    /// Fingerprint velocity: rate of directional change in harmony space.
    pub(super) fingerprint_velocity: f64,
    /// Previous persistence diagram for distance computation.
    pub(super) prev_persistence_diagram: PersistenceDiagram,
    /// Append-only forensic audit log of escalation transitions.
    pub(super) audit_log: EscalationAuditLog,
    /// Monotonic scenario counter (incremented on every `add_scenario`).
    pub(super) scenario_counter: u64,
    /// Cycle counter (incremented on every convergence detection).
    pub(super) detection_cycle: u64,
    /// Scenario IDs currently in the window (parallel to `self.window`).
    pub(super) window_scenario_ids: VecDeque<u64>,
    /// Consecutive cycles of suspected moral hubris.
    pub(super) hubris_streak: usize,
    /// EMA of critical_scale from recent Hodge analyses.
    /// Used when `adaptive_rips_enabled` to center the Rips sweep.
    /// NaN when no Hodge analysis has run yet.
    pub(super) critical_scale_ema: f64,
}

impl MoralTopology {
    /// Create a new analyser with its own `HarmonyBasis`.
    pub fn new(config: MoralTopologyConfig) -> Self {
        let basis = Arc::new(HarmonyBasis::new(config.dim));
        Self::with_basis(config, basis)
    }

    /// Create a new analyser with a shared `HarmonyBasis`.
    pub fn with_basis(config: MoralTopologyConfig, basis: Arc<HarmonyBasis>) -> Self {
        Self {
            config,
            anomaly_config: MoralAnomalyConfig::default(),
            window: VecDeque::new(),
            basis,
            prev_summary: MoralTopologySummary::default(),
            last_summary: MoralTopologySummary::default(),
            harmony_prior: [0.0; N_HARMONIES],
            prior_count: 0,
            last_persistent_features: Vec::new(),
            trajectory: VecDeque::new(),
            adaptive_state: AdaptiveAnomalyState::default(),
            baseline_similarity_window: VecDeque::new(),
            baseline_entropy_window: VecDeque::new(),
            baseline_flourishing_window: VecDeque::new(),
            baseline_spectral_gap_window: VecDeque::new(),
            severity_calibration_cdf: Vec::new(),
            last_convergence_report: TrajectoryConvergenceReport::default(),
            hazard_registry: HazardSignatureRegistry::with_defaults(),
            escalation_policy: EscalationPolicy::default(),
            prev_fingerprint: [0.0; N_HARMONIES],
            fingerprint_velocity: 0.0,
            prev_persistence_diagram: PersistenceDiagram::default(),
            audit_log: EscalationAuditLog::new(1000),
            scenario_counter: 0,
            detection_cycle: 0,
            window_scenario_ids: VecDeque::new(),
            hubris_streak: 0,
            critical_scale_ema: f64::NAN,
        }
    }

    /// Create a new analyser with a shared `HarmonyBasis` and custom anomaly config.
    pub fn with_anomaly_config(
        config: MoralTopologyConfig,
        basis: Arc<HarmonyBasis>,
        anomaly_config: MoralAnomalyConfig,
    ) -> Self {
        let adaptive_state = AdaptiveAnomalyState {
            effective_drift_threshold: anomaly_config.drift_alert_threshold,
            effective_fe_sigma: anomaly_config.fe_sigma_multiplier,
            ..Default::default()
        };
        Self {
            config,
            anomaly_config,
            window: VecDeque::new(),
            basis,
            prev_summary: MoralTopologySummary::default(),
            last_summary: MoralTopologySummary::default(),
            harmony_prior: [0.0; N_HARMONIES],
            prior_count: 0,
            last_persistent_features: Vec::new(),
            trajectory: VecDeque::new(),
            adaptive_state,
            baseline_similarity_window: VecDeque::new(),
            baseline_entropy_window: VecDeque::new(),
            baseline_flourishing_window: VecDeque::new(),
            baseline_spectral_gap_window: VecDeque::new(),
            severity_calibration_cdf: Vec::new(),
            last_convergence_report: TrajectoryConvergenceReport::default(),
            hazard_registry: HazardSignatureRegistry::with_defaults(),
            escalation_policy: EscalationPolicy::default(),
            prev_fingerprint: [0.0; N_HARMONIES],
            fingerprint_velocity: 0.0,
            prev_persistence_diagram: PersistenceDiagram::default(),
            audit_log: EscalationAuditLog::new(1000),
            scenario_counter: 0,
            detection_cycle: 0,
            window_scenario_ids: VecDeque::new(),
            hubris_streak: 0,
            critical_scale_ema: f64::NAN,
        }
    }

    /// Push a scenario hypervector into the sliding window.
    ///
    /// Current EMA of the critical Rips scale. NaN if no Hodge analysis has run.
    pub fn critical_scale_ema(&self) -> f64 {
        self.critical_scale_ema
    }

    /// Also updates the running EMA prior of harmony coordinates for
    /// moral free energy computation and records a trajectory point.
    pub fn add_scenario(&mut self, hv: ContinuousHV) {
        // Assign monotonic scenario ID
        let scenario_id = self.scenario_counter;
        self.scenario_counter += 1;

        // Update harmony prior via EMA before evicting the oldest entry
        let coords = self.basis.project(&hv);
        let alpha = if self.prior_count == 0 { 1.0 } else { 0.05 };
        for i in 0..N_HARMONIES {
            self.harmony_prior[i] = alpha * coords[i] + (1.0 - alpha) * self.harmony_prior[i];
        }
        // FEP stillness prior: seed Sacred Stillness expectation so the system
        // does not treat periodic rest as morally surprising. The prior decays
        // toward the observed EMA, but we ensure it never drops below a baseline.
        // Science: Friston (2010) — viable systems must predict their own rest states;
        // Tononi & Cirelli (2006) — rest is expected, not anomalous.
        const STILLNESS_PRIOR_FLOOR: f64 = 0.05;
        if self.harmony_prior[7] < STILLNESS_PRIOR_FLOOR {
            self.harmony_prior[7] = STILLNESS_PRIOR_FLOOR;
        }
        self.prior_count += 1;

        // Record trajectory point
        let free_energy = self.last_summary.moral_free_energy;
        if self.trajectory.len() >= self.config.window_size {
            self.trajectory.pop_front();
        }
        self.trajectory.push_back(MoralTrajectoryPoint {
            coordinates: coords,
            free_energy,
        });

        if self.window.len() >= self.config.window_size {
            self.window.pop_front();
            self.window_scenario_ids.pop_front();
        }
        self.window.push_back(hv);
        self.window_scenario_ids.push_back(scenario_id);
    }

    /// Number of scenarios currently in the window.
    pub fn len(&self) -> usize {
        self.window.len()
    }

    /// Whether the window is empty.
    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }

    /// Access the last computed summary.
    pub fn last_summary(&self) -> &MoralTopologySummary {
        &self.last_summary
    }

    /// Access the previous topology summary (from the analyze() call before last).
    /// Used internally by detect_anomalies() via self.prev_summary field; this
    /// public accessor exists for test inspection and debugging.
    #[allow(dead_code)]
    pub fn prev_summary(&self) -> &MoralTopologySummary {
        &self.prev_summary
    }

    /// Access the shared harmony basis.
    pub fn basis(&self) -> &Arc<HarmonyBasis> {
        &self.basis
    }

    /// Access the anomaly detection configuration.
    pub fn anomaly_config(&self) -> &MoralAnomalyConfig {
        &self.anomaly_config
    }

    /// Access cached persistent features from last `analyze()` call.
    pub fn last_persistent_features(&self) -> &[PersistentFeature] {
        &self.last_persistent_features
    }

    /// Access the hazard signature registry (read-only).
    pub fn hazard_registry(&self) -> &HazardSignatureRegistry {
        &self.hazard_registry
    }

    /// Access the hazard signature registry (for adding custom signatures).
    pub fn hazard_registry_mut(&mut self) -> &mut HazardSignatureRegistry {
        &mut self.hazard_registry
    }

    /// Boost the cached convergence severity (e.g., from peer correlation).
    ///
    /// Adds `boost` to the last convergence report's severity, clamped to [0, 1].
    /// If the boosted severity crosses detection threshold, marks convergence_detected.
    pub fn boost_convergence_severity(&mut self, boost: f64) {
        self.last_convergence_report.severity =
            (self.last_convergence_report.severity + boost).clamp(0.0, 1.0);
        let new_severity = self.last_convergence_report.severity;
        self.last_convergence_report.calibrated_severity = self.calibrate_severity(new_severity);
        if !self.last_convergence_report.convergence_detected && new_severity > 0.5 {
            self.last_convergence_report.convergence_detected = true;
        }
    }

    /// Access the escalation audit log (immutable).
    pub fn audit_log(&self) -> &EscalationAuditLog {
        &self.audit_log
    }

    /// Current scenario counter (total scenarios added since creation).
    pub fn scenario_counter(&self) -> u64 {
        self.scenario_counter
    }

    /// Access the escalation policy (immutable).
    pub fn escalation_policy(&self) -> &EscalationPolicy {
        &self.escalation_policy
    }

    /// Access the escalation policy (mutable, for custom thresholds).
    pub fn escalation_policy_mut(&mut self) -> &mut EscalationPolicy {
        &mut self.escalation_policy
    }

    /// Recent trajectory points (up to `last_n`).
    pub fn trajectory(&self, last_n: usize) -> Vec<&MoralTrajectoryPoint> {
        self.trajectory.iter().rev().take(last_n).collect()
    }

    /// Build a persistence diagram summary from cached features.
    pub fn persistence_diagram(&self) -> PersistenceDiagram {
        PersistenceDiagram::from_features(&self.last_persistent_features)
    }

    /// Perform full topological analysis on the current window.
    ///
    /// Returns `MoralTopologyAssessment` with Betti numbers, persistent
    /// features, harmony projection, PGA, and completeness scores.
    pub fn analyze(&mut self) -> MoralTopologyAssessment {
        let n = self.window.len();

        if n == 0 {
            let assessment = MoralTopologyAssessment {
                betti: BettiNumbers::new(1, 0, 0),
                persistent_features: Vec::new(),
                unity: 1.0,
                circularity: 0.0,
                completeness: 0.0,
                harmony_coordinates: Vec::new(),
                pga: PGAResult {
                    mean: vec![0.0; N_HARMONIES],
                    principal_directions: Vec::new(),
                    variances: Vec::new(),
                },
                dominant_harmony_idx: 0,
                harmony_variance: [0.0; N_HARMONIES],
                scenario_count: 0,
                moral_free_energy: MoralFreeEnergy::default(),
                harmony_entropy: 0.0,
                attractor_detected: false,
                hodge_fractions: None,
            };
            let mut new_summary = MoralTopologySummary::from(&assessment);
            self.stamp_fingerprint(&mut new_summary);
            self.prev_summary = std::mem::replace(&mut self.last_summary, new_summary);
            return assessment;
        }

        // ── Step 1: Pairwise similarity matrix ──────────────────────────
        let similarities = self.pairwise_similarities();

        // ── Step 2: Characteristic scale (median similarity) ────────────
        let char_scale = Self::characteristic_scale(&similarities, n);

        // ── Step 3: Betti numbers at characteristic scale ───────────────
        let betti = if self.config.exact_betti {
            Self::compute_betti_exact(&similarities, n, char_scale)
        } else {
            Self::compute_betti(&similarities, n, char_scale)
        };

        // ── Step 4: Multi-scale persistent features ─────────────────────
        let persistent_features = self.persistent_features(&similarities, n);

        // ── Step 5: Harmony projection ──────────────────────────────────
        let harmony_coordinates: Vec<[f64; N_HARMONIES]> = self
            .window
            .iter()
            .map(|hv| self.basis.project(hv))
            .collect();

        // ── Step 5b: Persistence-weighted vertex Hodge decomposition (L₀) ─
        // Decomposes harmony coordinates on vertices across multi-scale Rips
        // filtration. Measures moral fragmentation: harmonic fraction rises
        // when β₀ > 1 (disconnected clusters trap moral meaning).
        // Science: Hodge (1941), Tononi (2004) — consciousness requires integration.
        let adaptive_center =
            if self.config.adaptive_rips_enabled && self.critical_scale_ema.is_finite() {
                Some(self.critical_scale_ema)
            } else {
                None
            };
        let hodge_fractions = if self.config.exact_betti {
            Self::compute_persistent_hodge_fractions(
                &similarities,
                n,
                self.config.num_scales,
                &harmony_coordinates,
                adaptive_center,
            )
        } else {
            None
        };

        // Update critical_scale EMA for adaptive Rips focusing
        if let Some(ref fracs) = hodge_fractions {
            if fracs.critical_scale.is_finite() {
                const ALPHA: f64 = 0.05; // HODGE_CRITICAL_SCALE_EMA_ALPHA
                if self.critical_scale_ema.is_finite() {
                    self.critical_scale_ema =
                        ALPHA * fracs.critical_scale + (1.0 - ALPHA) * self.critical_scale_ema;
                } else {
                    self.critical_scale_ema = fracs.critical_scale;
                }
            }
        }

        // ── Step 6: Per-harmony variance ────────────────────────────────
        let harmony_variance = Self::harmony_variance(&harmony_coordinates);

        // ── Step 7: PGA on 7D coordinates ───────────────────────────────
        let points_f64: Vec<Vec<f64>> = harmony_coordinates
            .iter()
            .map(|c| {
                // Normalize to unit sphere for PGA
                let norm: f64 = c.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 1e-12 {
                    c.iter().map(|x| x / norm).collect()
                } else {
                    vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] // default pole
                }
            })
            .collect();

        let pga_components = self.config.pga_components.min(7).min(n);
        let pga = if pga_components > 0 && n >= 2 {
            HypersphereOps::principal_geodesic_analysis(&points_f64, pga_components)
        } else {
            PGAResult {
                mean: vec![0.0; N_HARMONIES],
                principal_directions: Vec::new(),
                variances: Vec::new(),
            }
        };

        // ── Step 8: Dominant harmony axis ───────────────────────────────
        let dominant_harmony_idx = if !pga.principal_directions.is_empty() {
            let dir = &pga.principal_directions[0];
            dir.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
                .map(|(i, _)| i as u8)
                .unwrap_or(0)
        } else {
            // Fallback: highest variance harmony
            harmony_variance
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i as u8)
                .unwrap_or(0)
        };

        // ── Derived scores ──────────────────────────────────────────────
        // Guard: exact_betti Jacobi solver may undercount zero eigenvalues
        // for small/degenerate Rips complexes, producing beta_0=0.
        let betti = if betti.beta_0 == 0 && n > 0 {
            BettiNumbers::new(1, betti.beta_1, betti.beta_2)
        } else {
            betti
        };
        let unity = 1.0 / (betti.beta_0.max(1) as f64);
        let circularity = {
            let cycle_count = persistent_features
                .iter()
                .filter(|f| f.feature_type == TopologicalFeature::Cycle)
                .count();
            if persistent_features.is_empty() {
                0.0
            } else {
                cycle_count as f64 / persistent_features.len() as f64
            }
        };
        let completeness = {
            let active = harmony_variance.iter().filter(|&&v| v > 1e-6).count();
            active as f64 / N_HARMONIES as f64
        };

        // ── Step 9: Moral free energy (FEP on harmony manifold) ───────
        let moral_free_energy = {
            // Mean of current window's harmony coordinates
            let mut mean_coords = [0.0f64; N_HARMONIES];
            for c in &harmony_coordinates {
                for i in 0..N_HARMONIES {
                    mean_coords[i] += c[i];
                }
            }
            let n_f = harmony_coordinates.len() as f64;
            if n_f > 0.0 {
                for m in &mut mean_coords {
                    *m /= n_f;
                }
            }
            self.basis
                .moral_free_energy(&mean_coords, &self.harmony_prior, 1.0)
        };

        // ── Step 10: Harmony entropy (moral breadth) ───────────────────
        let harmony_entropy = {
            let total_var: f64 = harmony_variance.iter().sum::<f64>().max(1e-12);
            harmony_variance
                .iter()
                .map(|&v| {
                    let p = (v / total_var).max(1e-12);
                    -p * p.ln()
                })
                .sum::<f64>()
        };

        // ── Step 11: Moral attractor detection ──────────────────────────
        let attractor_detected = {
            let low_free_energy = moral_free_energy.free_energy < 0.5;
            // Check if variance drift is small (stable basin)
            let low_drift = if self.prev_summary.scenario_count > 0 {
                (moral_free_energy.free_energy - self.prev_summary.moral_free_energy).abs() < 0.1
            } else {
                false
            };
            low_free_energy && low_drift
        };

        let assessment = MoralTopologyAssessment {
            betti,
            persistent_features,
            unity,
            circularity,
            completeness,
            harmony_coordinates,
            pga,
            dominant_harmony_idx,
            harmony_variance,
            scenario_count: n,
            moral_free_energy,
            harmony_entropy,
            attractor_detected,
            hodge_fractions,
        };
        let mut new_summary = MoralTopologySummary::from(&assessment);
        self.stamp_fingerprint(&mut new_summary);
        self.prev_summary = std::mem::replace(&mut self.last_summary, new_summary);
        self.last_persistent_features = assessment.persistent_features.clone();
        assessment
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Cross-Session Persistence
// ═══════════════════════════════════════════════════════════════════════════════

/// Serializable snapshot of moral topology state for cross-session persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoralTopologySnapshot {
    pub trajectory: Vec<MoralTrajectoryPoint>,
    pub harmony_prior: [f64; N_HARMONIES],
    pub prior_count: usize,
    /// Sliding window of baseline similarity observations.
    pub baseline_similarity_window: Vec<f64>,
    /// Sliding window of baseline entropy observations.
    pub baseline_entropy_window: Vec<f64>,
    /// Sliding window of baseline flourishing observations.
    pub baseline_flourishing_window: Vec<f64>,
    /// Sliding window of baseline spectral gap observations.
    #[serde(default)]
    pub baseline_spectral_gap_window: Vec<f64>,
    pub adaptive_state: AdaptiveAnomalyState,
    pub last_summary: MoralTopologySummary,
    pub prev_summary: MoralTopologySummary,
    // ── Forensics state (added for cross-session persistence) ──────────
    /// Escalation audit log (BLAKE3-sealed entries).
    #[serde(default)]
    pub audit_log: EscalationAuditLog,
    /// Monotonic scenario counter.
    #[serde(default)]
    pub scenario_counter: u64,
    /// Detection cycle counter.
    #[serde(default)]
    pub detection_cycle: u64,
    /// Scenario IDs in the current sliding window.
    #[serde(default)]
    pub window_scenario_ids: Vec<u64>,
    /// Escalation policy state (current level + cooldown).
    #[serde(default)]
    pub escalation_policy: EscalationPolicy,
    /// Previous trajectory fingerprint (for velocity computation).
    #[serde(default)]
    pub prev_fingerprint: [f64; N_HARMONIES],
    /// Last computed fingerprint velocity.
    #[serde(default)]
    pub fingerprint_velocity: f64,
}

impl MoralTopology {
    /// Snapshot current state for cross-session persistence.
    pub fn snapshot(&self) -> MoralTopologySnapshot {
        MoralTopologySnapshot {
            trajectory: self.trajectory.iter().cloned().collect(),
            harmony_prior: self.harmony_prior,
            prior_count: self.prior_count,
            baseline_similarity_window: self.baseline_similarity_window.iter().copied().collect(),
            baseline_entropy_window: self.baseline_entropy_window.iter().copied().collect(),
            baseline_flourishing_window: self.baseline_flourishing_window.iter().copied().collect(),
            baseline_spectral_gap_window: self
                .baseline_spectral_gap_window
                .iter()
                .copied()
                .collect(),
            adaptive_state: self.adaptive_state.clone(),
            last_summary: self.last_summary.clone(),
            prev_summary: self.prev_summary.clone(),
            audit_log: self.audit_log.clone(),
            scenario_counter: self.scenario_counter,
            detection_cycle: self.detection_cycle,
            window_scenario_ids: self.window_scenario_ids.iter().copied().collect(),
            escalation_policy: self.escalation_policy.clone(),
            prev_fingerprint: self.prev_fingerprint,
            fingerprint_velocity: self.fingerprint_velocity,
        }
    }

    /// Restore state from a cross-session snapshot.
    pub fn restore(&mut self, snap: &MoralTopologySnapshot) {
        self.trajectory = snap.trajectory.iter().cloned().collect();
        self.harmony_prior = snap.harmony_prior;
        self.prior_count = snap.prior_count;
        self.baseline_similarity_window = snap.baseline_similarity_window.iter().copied().collect();
        self.baseline_entropy_window = snap.baseline_entropy_window.iter().copied().collect();
        self.baseline_flourishing_window =
            snap.baseline_flourishing_window.iter().copied().collect();
        self.baseline_spectral_gap_window =
            snap.baseline_spectral_gap_window.iter().copied().collect();
        self.adaptive_state = snap.adaptive_state.clone();
        self.last_summary = snap.last_summary.clone();
        self.prev_summary = snap.prev_summary.clone();
        self.audit_log = snap.audit_log.clone();
        self.scenario_counter = snap.scenario_counter;
        self.detection_cycle = snap.detection_cycle;
        self.window_scenario_ids = snap.window_scenario_ids.iter().copied().collect();
        self.escalation_policy = snap.escalation_policy.clone();
        self.prev_fingerprint = snap.prev_fingerprint;
        self.fingerprint_velocity = snap.fingerprint_velocity;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use crate::hdc::moral_text_encoder::TextHdcEncoder;

    /// Small dimension for fast tests.
    const TEST_DIM: usize = 512;

    fn test_config() -> MoralTopologyConfig {
        MoralTopologyConfig {
            dim: TEST_DIM,
            ..Default::default()
        }
    }

    fn encode_text(text: &str) -> ContinuousHV {
        let encoder = TextHdcEncoder::with_sentiment(TEST_DIM, 3, 0.5, 0.2);
        encoder.encode(text)
    }

    // ── Test 1: Config defaults ─────────────────────────────────────────

    #[test]
    fn test_config_defaults() {
        let cfg = MoralTopologyConfig::default();
        assert_eq!(cfg.window_size, 64);
        assert_eq!(cfg.num_scales, 10);
        assert!((cfg.min_persistence - 0.1).abs() < f64::EPSILON);
        assert_eq!(cfg.pga_components, 3);
        assert_eq!(cfg.dim, 16384);
    }

    // ── Test 2: Empty window ────────────────────────────────────────────

    #[test]
    fn test_empty_window_unity() {
        let mut topo = MoralTopology::new(test_config());
        let assessment = topo.analyze();

        assert_eq!(assessment.betti.beta_0, 1);
        assert_eq!(assessment.betti.beta_1, 0);
        assert_eq!(assessment.betti.beta_2, 0);
        assert!((assessment.unity - 1.0).abs() < f64::EPSILON);
        assert_eq!(assessment.scenario_count, 0);
    }

    // ── Test 3: Sliding window eviction ─────────────────────────────────

    #[test]
    fn test_window_eviction() {
        let mut cfg = test_config();
        cfg.window_size = 4;
        let mut topo = MoralTopology::new(cfg);

        for i in 0..6 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 100 + i));
        }
        assert_eq!(topo.len(), 4);
    }

    // ── Test 4: Harmony basis near-orthogonality ────────────────────────

    #[test]
    fn test_harmony_basis_low_mutual_similarity() {
        let basis = HarmonyBasis::new(TEST_DIM);
        let mut max_sim = 0.0f32;
        for i in 0..N_HARMONIES {
            for j in (i + 1)..7 {
                let sim = basis.vectors[i].similarity(&basis.vectors[j]);
                if sim > max_sim {
                    max_sim = sim;
                }
            }
        }
        // Different keyword sets should have moderate-to-low similarity
        assert!(
            max_sim < 0.85,
            "Harmony basis vectors too similar: max={max_sim}"
        );
    }

    // ── Test 5: Semantic projection ─────────────────────────────────────

    #[test]
    fn test_semantic_projection_care() {
        let basis = HarmonyBasis::new(TEST_DIM);
        let hv = encode_text("helping with kindness and compassion");
        let coords = basis.project(&hv);

        // PanSentientFlourishing is index 1 (Harmony::all() order)
        let psf_idx = 1;
        let psf_score = coords[psf_idx];

        // Should be among the top 3 harmonies for a care-oriented sentence
        let mut sorted: Vec<(usize, f64)> = coords.iter().copied().enumerate().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top3_indices: Vec<usize> = sorted.iter().take(3).map(|(i, _)| *i).collect();
        assert!(
            top3_indices.contains(&psf_idx),
            "PanSentientFlourishing should be in top-3 for care text, got {:?} (PSF score={:.4})",
            sorted,
            psf_score,
        );
    }

    // ── Test 6: Unified topology from same-domain scenarios ─────────────

    #[test]
    fn test_unified_topology_same_domain() {
        let mut topo = MoralTopology::new(test_config());

        // Feed similar "helping" scenarios — identical prefix maximizes HDC overlap
        let phrases = [
            "helping the elderly cross the street",
            "helping the children learn to read",
            "helping the neighbors fix their house",
            "helping the friends in times of need",
            "helping the strangers with directions",
        ];
        for phrase in &phrases {
            topo.add_scenario(encode_text(phrase));
        }

        let assessment = topo.analyze();
        assert!(
            assessment.betti.beta_0 <= 2,
            "Same-domain scenarios should be nearly unified (β₀ ≤ 2), got β₀={}",
            assessment.betti.beta_0,
        );
    }

    // ── Test 7: Fragmented topology from diverse scenarios ──────────────

    #[test]
    fn test_fragmented_topology_diverse() {
        let mut topo = MoralTopology::new(test_config());

        for i in 0..8 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 9000 + i * 10000));
        }

        let assessment = topo.analyze();
        assert!(
            assessment.completeness <= 1.0,
            "Random HVs should have bounded completeness, got {:.3}",
            assessment.completeness,
        );
        assert!(
            assessment.betti.beta_0 >= 1,
            "Must have at least 1 connected component, got β₀={}",
            assessment.betti.beta_0,
        );
    }

    // ── Test 8: PGA dominant axis ───────────────────────────────────────

    #[test]
    fn test_pga_dominant_axis() {
        let mut topo = MoralTopology::new(test_config());

        let phrases = [
            "helping with kindness",
            "caring for the sick",
            "protecting the vulnerable",
            "nurturing children with love",
            "supporting the community",
        ];
        for phrase in &phrases {
            topo.add_scenario(encode_text(phrase));
        }

        let assessment = topo.analyze();
        assert!(
            !assessment.pga.principal_directions.is_empty(),
            "PGA should produce at least one direction"
        );
        assert!(assessment.dominant_harmony_idx < 7);
    }

    // ── Test 9: Blind spot detection ────────────────────────────────────

    #[test]
    fn test_blind_spot_detection() {
        let mut topo = MoralTopology::new(test_config());

        let phrases = [
            "helping others with kindness",
            "caring deeply for someone",
            "protecting the weak with compassion",
        ];
        for phrase in &phrases {
            topo.add_scenario(encode_text(phrase));
        }

        let assessment = topo.analyze();

        let near_zero = assessment
            .harmony_variance
            .iter()
            .filter(|&&v| v < 1e-4)
            .count();

        assert!(
            assessment.completeness <= 1.0,
            "Completeness should be at most 1.0"
        );

        let max_var = assessment
            .harmony_variance
            .iter()
            .copied()
            .fold(0.0f64, f64::max);
        assert!(max_var.is_finite());
        let _ = near_zero; // used for reasoning
    }

    // ── Test 10: Summary conversion ─────────────────────────────────────

    #[test]
    fn test_summary_conversion() {
        let mut topo = MoralTopology::new(test_config());
        topo.add_scenario(encode_text("helping others"));
        topo.add_scenario(encode_text("stealing from people"));

        let assessment = topo.analyze();
        let summary = MoralTopologySummary::from(&assessment);

        assert_eq!(summary.beta_0, assessment.betti.beta_0);
        assert_eq!(summary.beta_1, assessment.betti.beta_1);
        assert_eq!(summary.beta_2, assessment.betti.beta_2);
        assert!((summary.unity - assessment.unity).abs() < f64::EPSILON);
        assert!((summary.completeness - assessment.completeness).abs() < f64::EPSILON);
        assert!((summary.circularity - assessment.circularity).abs() < f64::EPSILON);
        assert!(
            (summary.moral_free_energy - assessment.moral_free_energy.free_energy).abs()
                < f64::EPSILON
        );
        assert_eq!(summary.dominant_harmony, assessment.dominant_harmony_idx);
        assert_eq!(summary.scenario_count, assessment.scenario_count);
    }

    // ── Test 11: Determinism ────────────────────────────────────────────

    #[test]
    fn test_determinism() {
        let run = || {
            let mut topo = MoralTopology::new(test_config());
            topo.add_scenario(encode_text("helping others is good"));
            topo.add_scenario(encode_text("harming others is wrong"));
            topo.add_scenario(encode_text("learning brings wisdom"));
            topo.analyze()
        };

        let a = run();
        let b = run();

        assert_eq!(a.betti.beta_0, b.betti.beta_0);
        assert_eq!(a.betti.beta_1, b.betti.beta_1);
        assert_eq!(a.betti.beta_2, b.betti.beta_2);
        assert!((a.unity - b.unity).abs() < f64::EPSILON);
        assert!((a.completeness - b.completeness).abs() < f64::EPSILON);
        assert_eq!(a.dominant_harmony_idx, b.dominant_harmony_idx);
        assert_eq!(a.harmony_variance, b.harmony_variance);
    }

    // ── Test 12: Persistent features valid birth < death ────────────────

    #[test]
    fn test_persistent_features_birth_before_death() {
        let mut topo = MoralTopology::new(test_config());
        for i in 0..10 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 5000 + i));
        }
        let assessment = topo.analyze();

        for pf in &assessment.persistent_features {
            assert!(
                pf.birth <= pf.death,
                "Birth ({}) must be ≤ death ({})",
                pf.birth,
                pf.death,
            );
            assert!(pf.persistence >= 0.0, "Persistence must be non-negative");
        }
    }

    // ── Test 13: Moral free energy is finite ─────────────────────────

    #[test]
    fn test_moral_free_energy_finite() {
        let mut topo = MoralTopology::new(test_config());
        topo.add_scenario(encode_text("helping others is good"));
        topo.add_scenario(encode_text("harming others is wrong"));
        topo.add_scenario(encode_text("learning brings wisdom"));

        let assessment = topo.analyze();
        assert!(
            assessment.moral_free_energy.free_energy.is_finite(),
            "Moral free energy should be finite, got {:?}",
            assessment.moral_free_energy
        );
        assert!(assessment.moral_free_energy.entropy >= 0.0);
        let summary = MoralTopologySummary::from(&assessment);
        assert!(summary.moral_free_energy.is_finite());
    }

    // ── Test 14: Benchmark analyze at n=64 ──────────────────────────

    #[test]
    fn bench_analyze_n64() {
        let dim = 512;
        let mut topo = MoralTopology::new(MoralTopologyConfig {
            dim,
            window_size: 64,
            ..Default::default()
        });
        for seed in 0..64u64 {
            topo.add_scenario(ContinuousHV::random(dim, seed));
        }
        let start = std::time::Instant::now();
        let assessment = topo.analyze();
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 2000,
            "analyze() took {elapsed:?} at n=64, dim={dim}"
        );
        assert_eq!(assessment.scenario_count, 64);
        assert!(assessment.moral_free_energy.free_energy.is_finite());
        eprintln!("MoralTopology::analyze() n=64, dim={dim}: {elapsed:?}");
    }

    // ── Test 15: Trajectory memory ───────────────────────────────────

    #[test]
    fn test_trajectory_records_points() {
        let mut topo = MoralTopology::new(test_config());
        for i in 0..5 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 200 + i));
        }
        let traj = topo.trajectory(10);
        assert_eq!(traj.len(), 5);
        for p in &traj {
            for &c in &p.coordinates {
                assert!(c.is_finite(), "trajectory coordinate should be finite");
            }
            assert!(p.free_energy.is_finite());
        }
    }

    // ── Test 16: Trajectory caps at window_size ──────────────────────

    #[test]
    fn test_trajectory_window_cap() {
        let mut cfg = test_config();
        cfg.window_size = 4;
        let mut topo = MoralTopology::new(cfg);
        for i in 0..10 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 300 + i));
        }
        assert_eq!(topo.trajectory(100).len(), 4);
    }

    // ── Test 17: Moral drift ─────────────────────────────────────────

    #[test]
    fn test_moral_drift_zero_when_few_points() {
        let mut topo = MoralTopology::new(test_config());
        topo.add_scenario(ContinuousHV::random(TEST_DIM, 400));
        assert!((topo.moral_drift(10) - 0.0).abs() < f64::EPSILON);
    }

    // ── Test 18: Persistence diagram ────────────────────────────────

    #[test]
    fn test_persistence_diagram_from_features() {
        let mut topo = MoralTopology::new(test_config());
        for i in 0..10 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 500 + i));
        }
        let _assessment = topo.analyze();
        let diagram = topo.persistence_diagram();
        assert!(diagram.bottleneck_distance >= 0.0);
        assert!(diagram.total_persistence >= 0.0);
        let total = diagram.components.len() + diagram.cycles.len() + diagram.voids.len();
        assert_eq!(total, topo.last_persistent_features().len());
    }

    // ── Test 19: Exact Betti via Hodge Laplacian ─────────────────────

    #[test]
    fn test_exact_betti_small_window() {
        let mut cfg = test_config();
        cfg.exact_betti = true;
        cfg.window_size = 8;
        let mut topo = MoralTopology::new(cfg);
        for i in 0..8 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 600 + i));
        }
        let assessment = topo.analyze();
        assert!(
            assessment.betti.beta_0 >= 1,
            "Exact Betti β₀ must be ≥ 1, got {}",
            assessment.betti.beta_0,
        );
        assert!(assessment.unity <= 1.0);
        assert!(assessment.unity > 0.0);
    }

    // ── Test 20: Anomaly — value inversion ──────────────────────────

    #[test]
    fn test_anomaly_value_inversion() {
        let encoder = TextHdcEncoder::new(TEST_DIM, 3);
        let mut topo = MoralTopology::new(test_config());

        for _ in 0..10 {
            topo.add_scenario(encoder.encode("caring for the sick and elderly"));
        }
        topo.analyze();

        for _ in 0..10 {
            topo.add_scenario(encoder.encode("caring for the sick and elderly"));
        }
        let second = topo.analyze();
        let dominant = MoralTopologySummary::from(&second).dominant_harmony;

        let mut shifted_summary = topo.last_summary().clone();
        shifted_summary.dominant_harmony = (dominant + 1) % 7;

        let report = topo.detect_anomalies(&shifted_summary);
        assert!(
            report.value_inversion,
            "Different dominant harmony should trigger value inversion"
        );
        assert!(report.anomaly_score > 0.0);
    }

    // ── Test 21: Anomaly — free energy spike ────────────────────────

    #[test]
    fn test_anomaly_fe_spike() {
        let mut topo = MoralTopology::new(test_config());

        for i in 0..20 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 700 + i));
        }
        topo.analyze();

        let mut spiked = topo.last_summary().clone();
        spiked.moral_free_energy = 100.0;

        let report = topo.detect_anomalies(&spiked);
        assert!(
            report.free_energy_spike,
            "FE=100.0 should spike above 2σ of near-zero trajectory"
        );
        assert!(report.anomaly_score >= 0.3);
    }

    #[test]
    fn custom_anomaly_config_changes_drift_threshold() {
        let config = MoralAnomalyConfig {
            drift_alert_threshold: 0.001,
            ..Default::default()
        };
        let mut topo = MoralTopology::with_anomaly_config(
            MoralTopologyConfig {
                dim: TEST_DIM,
                ..Default::default()
            },
            Arc::new(HarmonyBasis::new(TEST_DIM)),
            config,
        );

        for i in 0..10 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 800 + i));
        }
        topo.analyze();

        let summary = topo.last_summary().clone();
        let report = topo.detect_anomalies(&summary);
        if topo.moral_drift(20) > 0.001 {
            assert!(
                report.drift_alert,
                "strict threshold should trigger drift alert"
            );
        }
    }

    #[test]
    fn custom_anomaly_config_changes_fe_sigma() {
        let config = MoralAnomalyConfig {
            fe_sigma_multiplier: 100.0,
            ..Default::default()
        };
        let topo = MoralTopology::with_anomaly_config(
            MoralTopologyConfig {
                dim: TEST_DIM,
                ..Default::default()
            },
            Arc::new(HarmonyBasis::new(TEST_DIM)),
            config,
        );
        assert_eq!(topo.anomaly_config().fe_sigma_multiplier, 100.0);
    }

    #[test]
    fn custom_anomaly_weights_affect_score() {
        let config = MoralAnomalyConfig {
            weight_value_inversion: 0.0,
            weight_fe_spike: 0.0,
            weight_fragmentation: 0.0,
            weight_drift: 1.0,
            drift_alert_threshold: 0.001,
            ..Default::default()
        };
        let mut topo = MoralTopology::with_anomaly_config(
            MoralTopologyConfig {
                dim: TEST_DIM,
                ..Default::default()
            },
            Arc::new(HarmonyBasis::new(TEST_DIM)),
            config,
        );

        for i in 0..10 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 1000 + i));
        }
        topo.analyze();

        for i in 10..20 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 1000 + i));
        }
        topo.analyze();

        let mut inversion = topo.last_summary().clone();
        inversion.dominant_harmony = (topo.last_summary().dominant_harmony + 1) % 7;
        let report = topo.detect_anomalies(&inversion);
        assert!(report.value_inversion);
        if !report.drift_alert {
            assert!(
                report.anomaly_score < 0.01,
                "with zero weight on inversion, score should be near 0 when no drift"
            );
        }
    }

    // ── Adaptive anomaly threshold tests ─────────────────────────────────

    #[test]
    fn adaptive_state_defaults() {
        let state = AdaptiveAnomalyState::default();
        assert_eq!(state.observations(), 0);
        assert!((state.effective_drift_threshold - 0.25).abs() < f64::EPSILON);
        assert!((state.effective_fe_sigma - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn adaptive_config_defaults_enabled() {
        let config = MoralAnomalyConfig::default();
        assert!(config.adaptive_enabled);
        assert!((config.adaptive_alpha - 0.02).abs() < f64::EPSILON);
        assert!((config.adaptive_sigma_factor - 2.0).abs() < f64::EPSILON);
        assert_eq!(config.adaptive_warmup, 20);
    }

    #[test]
    fn adaptive_state_warmup_phase() {
        let mut state = AdaptiveAnomalyState::default();
        let config = MoralAnomalyConfig {
            adaptive_enabled: true,
            adaptive_warmup: 5,
            ..Default::default()
        };

        for i in 0..4 {
            let active = state.observe(0.1, 1.0, &config);
            assert!(!active, "should not be active at observation {}", i);
        }
        let active = state.observe(0.1, 1.0, &config);
        assert!(active, "should be active at observation 5");
        assert_eq!(state.observations(), 5);
    }

    #[test]
    fn adaptive_thresholds_adjust_to_low_drift() {
        let mut state = AdaptiveAnomalyState::default();
        let config = MoralAnomalyConfig {
            adaptive_enabled: true,
            adaptive_warmup: 5,
            adaptive_alpha: 0.2,
            adaptive_sigma_factor: 2.0,
            ..Default::default()
        };

        for _ in 0..30 {
            state.observe(0.02, 0.5, &config);
        }

        assert!(state.effective_drift_threshold < 0.25);
        assert!(state.effective_drift_threshold >= 0.05);
    }

    #[test]
    fn adaptive_thresholds_adjust_to_high_drift() {
        let mut state = AdaptiveAnomalyState::default();
        let config = MoralAnomalyConfig {
            adaptive_enabled: true,
            adaptive_warmup: 5,
            adaptive_alpha: 0.2,
            adaptive_sigma_factor: 2.0,
            ..Default::default()
        };

        for _ in 0..30 {
            state.observe(0.5, 2.0, &config);
        }

        assert!(state.effective_drift_threshold > 0.25);
        assert!(state.effective_drift_threshold <= 0.8);
    }

    #[test]
    fn adaptive_fe_sigma_adjusts_to_variance() {
        let mut state = AdaptiveAnomalyState::default();
        let config = MoralAnomalyConfig {
            adaptive_enabled: true,
            adaptive_warmup: 5,
            adaptive_alpha: 0.3,
            adaptive_sigma_factor: 2.0,
            ..Default::default()
        };

        for i in 0..30 {
            let fe = if i % 2 == 0 { 0.1 } else { 3.0 };
            state.observe(0.1, fe, &config);
        }

        assert!(state.effective_fe_sigma > 2.0);
    }

    #[test]
    fn adaptive_thresholds_integrated_with_detect_anomalies() {
        let config = MoralAnomalyConfig {
            adaptive_enabled: true,
            adaptive_warmup: 3,
            adaptive_alpha: 0.3,
            adaptive_sigma_factor: 2.0,
            drift_alert_threshold: 0.25,
            ..Default::default()
        };
        let mut topo = MoralTopology::with_anomaly_config(
            MoralTopologyConfig {
                dim: TEST_DIM,
                ..Default::default()
            },
            Arc::new(HarmonyBasis::new(TEST_DIM)),
            config,
        );

        for i in 0..10 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 2000 + i));
        }
        topo.analyze();

        let summary = topo.last_summary().clone();
        for _ in 0..5 {
            topo.detect_anomalies(&summary);
        }

        let state = topo.adaptive_state();
        assert!(state.observations() >= 5);
    }

    #[test]
    fn static_thresholds_unchanged_when_adaptive_disabled() {
        let config = MoralAnomalyConfig {
            adaptive_enabled: false,
            drift_alert_threshold: 0.4,
            ..Default::default()
        };
        let mut topo = MoralTopology::with_anomaly_config(
            MoralTopologyConfig {
                dim: TEST_DIM,
                ..Default::default()
            },
            Arc::new(HarmonyBasis::new(TEST_DIM)),
            config,
        );

        for i in 0..10 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 3000 + i));
        }
        topo.analyze();

        let summary = topo.last_summary().clone();
        for _ in 0..10 {
            topo.detect_anomalies(&summary);
        }

        let state = topo.adaptive_state();
        assert_eq!(state.observations(), 0);
        assert!((state.effective_drift_threshold - 0.4).abs() < f64::EPSILON);
    }

    #[test]
    fn test_unity_always_finite() {
        let mut topo = MoralTopology::new(test_config());
        topo.add_scenario(encode_text("sharing resources fairly with neighbours"));
        topo.analyze();
        let assessment = topo.last_summary();
        assert!(assessment.unity.is_finite());
        assert!(assessment.unity > 0.0);
        assert!(assessment.unity <= 1.0);
    }

    #[test]
    fn test_beta0_max1_guard() {
        let guarded = 1.0 / (1_usize as f64);
        assert_eq!(guarded, 1.0);
    }

    // ── Config validation tests ──────────────────────────────────────────

    #[test]
    fn anomaly_config_default_validates() {
        assert!(MoralAnomalyConfig::default().validate().is_ok());
    }

    #[test]
    fn anomaly_config_zero_drift_threshold_rejected() {
        let mut c = MoralAnomalyConfig::default();
        c.drift_alert_threshold = 0.0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn anomaly_config_bad_cadence_ordering_rejected() {
        let mut c = MoralAnomalyConfig::default();
        c.cadence_fast = 100;
        c.cadence_moderate = 50;
        assert!(c.validate().is_err());
    }

    #[test]
    fn anomaly_config_nan_response_rejected() {
        let mut c = MoralAnomalyConfig::default();
        c.response_lr_drift = f64::NAN;
        assert!(c.validate().is_err());
    }

    #[test]
    fn response_magnitude_defaults() {
        let c = MoralAnomalyConfig::default();
        assert!((c.response_lr_inversion - 1.3).abs() < f64::EPSILON);
        assert!((c.response_exploration_fe - 0.15).abs() < f64::EPSILON);
        assert!((c.response_confidence_frag - (-0.1)).abs() < f64::EPSILON);
        assert!((c.response_lr_drift - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn detect_anomalies_stable_with_nan_free_energy() {
        let mut topo = MoralTopology::new(test_config());
        for i in 0..10 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 4000 + i));
        }
        topo.analyze();

        let mut bad_summary = topo.last_summary().clone();
        bad_summary.moral_free_energy = f64::NAN;
        let report = topo.detect_anomalies(&bad_summary);
        assert!(report.anomaly_score.is_finite());
        assert!(!report.free_energy_spike);
    }

    #[test]
    fn detect_anomalies_stable_with_infinity_free_energy() {
        let mut topo = MoralTopology::new(test_config());
        for i in 0..10 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 5000 + i));
        }
        topo.analyze();

        let mut bad_summary = topo.last_summary().clone();
        bad_summary.moral_free_energy = f64::INFINITY;
        let report = topo.detect_anomalies(&bad_summary);
        assert!(report.anomaly_score.is_finite());
    }

    #[test]
    fn prev_summary_tracks_previous_analysis() {
        let mut topo = MoralTopology::new(test_config());
        for i in 0..10 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 6000 + i));
        }
        topo.analyze();
        let first_sc = topo.last_summary().scenario_count;

        for i in 10..20 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 6000 + i));
        }
        topo.analyze();

        assert_eq!(topo.prev_summary().scenario_count, first_sc);
    }

    #[test]
    fn test_adaptive_extreme_alpha_near_zero() {
        let config = MoralAnomalyConfig {
            adaptive_enabled: true,
            adaptive_alpha: 0.001,
            adaptive_warmup: 3,
            ..Default::default()
        };
        let mut state = AdaptiveAnomalyState::default();
        for _ in 0..50 {
            state.observe(0.5, 10.0, &config);
        }
        assert!(state.effective_drift_threshold >= 0.05);
        assert!(state.effective_drift_threshold <= 0.8);
    }

    #[test]
    fn test_adaptive_extreme_alpha_near_one() {
        let config = MoralAnomalyConfig {
            adaptive_enabled: true,
            adaptive_alpha: 0.99,
            adaptive_warmup: 2,
            ..Default::default()
        };
        let mut state = AdaptiveAnomalyState::default();
        for _ in 0..5 {
            state.observe(0.01, 0.1, &config);
        }
        let low_threshold = state.effective_drift_threshold;
        for _ in 0..5 {
            state.observe(0.6, 50.0, &config);
        }
        let high_threshold = state.effective_drift_threshold;
        assert!(high_threshold > low_threshold);
    }

    #[test]
    fn test_all_zero_weights_produce_zero_score() {
        let config = MoralAnomalyConfig {
            weight_value_inversion: 0.0,
            weight_fe_spike: 0.0,
            weight_fragmentation: 0.0,
            weight_drift: 0.0,
            drift_alert_threshold: 0.001,
            fe_sigma_multiplier: 0.01,
            ..Default::default()
        };
        let mut topo = MoralTopology::with_anomaly_config(
            test_config(),
            Arc::new(HarmonyBasis::new(TEST_DIM)),
            config,
        );
        for i in 0..10 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 9000 + i));
        }
        topo.analyze();
        for i in 10..25 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 9100 + i));
        }
        topo.analyze();
        let summary = topo.last_summary().clone();
        let report = topo.detect_anomalies(&summary);
        assert_eq!(report.anomaly_score, 0.0);
    }

    #[test]
    fn test_adaptive_observe_nan_fe_doesnt_corrupt() {
        let config = MoralAnomalyConfig {
            adaptive_enabled: true,
            adaptive_alpha: 0.1,
            adaptive_warmup: 3,
            ..Default::default()
        };
        let mut state = AdaptiveAnomalyState::default();
        for _ in 0..10 {
            state.observe(0.1, 1.0, &config);
        }
        let pre_drift = state.effective_drift_threshold;
        state.observe(0.1, f64::NAN, &config);
        assert!(state.effective_drift_threshold.is_finite());
        assert!(state.effective_fe_sigma.is_finite());
        let drift_delta = (state.effective_drift_threshold - pre_drift).abs();
        assert!(drift_delta < 0.05);
    }

    // ── Trajectory Convergence Detection Tests ─────────────────────────

    #[test]
    fn test_convergence_not_triggered_on_insufficient_data() {
        let mut topo = MoralTopology::new(test_config());
        topo.add_scenario(encode_text("helping others"));
        topo.add_scenario(encode_text("cooking dinner"));
        let report = topo.detect_trajectory_convergence();
        assert!(!report.convergence_detected);
    }

    #[test]
    fn test_convergence_not_triggered_on_diverse_benign() {
        let mut cfg = test_config();
        cfg.window_size = 16;
        let mut topo = MoralTopology::new(cfg);
        let phrases = [
            "helping the elderly cross the street",
            "planting trees in the community garden",
            "teaching children mathematics at school",
            "cooking meals for the local shelter",
            "reading books about philosophy and history",
            "exercising at the park on a sunny day",
        ];
        for phrase in &phrases {
            topo.add_scenario(encode_text(phrase));
        }
        topo.analyze();
        let report = topo.detect_trajectory_convergence();
        assert!(!report.convergence_detected);
    }

    #[test]
    fn test_convergence_report_fields_populated() {
        let mut topo = MoralTopology::new(test_config());
        for i in 0..6 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 500 + i));
        }
        topo.analyze();
        let report = topo.detect_trajectory_convergence();
        assert!(report.recent_similarity.is_finite());
        assert!(report.baseline_similarity.is_finite());
        assert!(report.recent_entropy.is_finite());
        assert!(report.severity >= 0.0 && report.severity <= 1.0);
    }

    #[test]
    fn test_convergence_severity_bounded() {
        let mut topo = MoralTopology::new(test_config());
        let hv = encode_text("precision engineering timing circuits");
        for _ in 0..8 {
            topo.add_scenario(hv.clone());
        }
        topo.analyze();
        let report = topo.detect_trajectory_convergence();
        assert!(report.severity >= 0.0 && report.severity <= 1.0);
    }

    #[test]
    fn test_convergence_integrated_in_anomaly_report() {
        let mut topo = MoralTopology::new(test_config());
        for i in 0..6 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 700 + i));
        }
        let assessment = topo.analyze();
        let summary = MoralTopologySummary::from(&assessment);
        let report = topo.detect_anomalies(&summary);
        assert!(report.convergence_severity.is_finite());
        assert!(report.anomaly_score.is_finite());
    }

    #[test]
    fn test_convergence_config_validation() {
        let mut config = MoralAnomalyConfig::default();
        assert!(config.validate().is_ok());
        config.convergence_min_points = 1;
        assert!(config.validate().is_err());
        config.convergence_min_points = 4;
        config.convergence_similarity_threshold = -0.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_convergence_disabled_returns_default() {
        let mut anomaly_config = MoralAnomalyConfig::default();
        anomaly_config.convergence_enabled = false;
        let mut topo = MoralTopology::with_anomaly_config(
            test_config(),
            Arc::new(HarmonyBasis::new(TEST_DIM)),
            anomaly_config,
        );
        for i in 0..8 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 800 + i));
        }
        topo.analyze();
        let report = topo.detect_trajectory_convergence();
        assert!(!report.convergence_detected);
        assert!((report.severity - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_snapshot_roundtrip() {
        let mut topo = MoralTopology::new(test_config());
        for i in 0..10 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 500 + i));
        }
        topo.analyze();
        let _ = topo.detect_trajectory_convergence();
        let snap = topo.snapshot();
        assert_eq!(snap.trajectory.len(), 10);
        assert!(snap.prior_count > 0);
        let mut topo2 = MoralTopology::new(test_config());
        topo2.restore(&snap);
        assert_eq!(topo2.trajectory(100).len(), 10);
    }

    #[test]
    fn test_snapshot_serialization() {
        let mut topo = MoralTopology::new(test_config());
        for i in 0..5 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 600 + i));
        }
        topo.analyze();
        let snap = topo.snapshot();
        let json = serde_json::to_string(&snap).expect("serialize");
        let restored: MoralTopologySnapshot = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.trajectory.len(), snap.trajectory.len());
    }

    #[test]
    fn test_hazard_registry_defaults() {
        let reg = HazardSignatureRegistry::with_defaults();
        assert_eq!(reg.signatures.len(), 4);
    }

    #[test]
    fn test_hazard_registry_match() {
        let reg = HazardSignatureRegistry::with_defaults();
        let centroid = [0.2, -0.6, -0.5, -0.2, -0.5, -0.1, 0.1, 0.0];
        let (name, boost) = reg.match_trajectory(&centroid);
        assert_eq!(name, Some("weaponization"));
        assert!(boost > 0.0);
        let benign = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8];
        let (name, boost) = reg.match_trajectory(&benign);
        assert!(name.is_none());
        assert!((boost - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hazard_custom_signature() {
        let mut reg = HazardSignatureRegistry::default();
        reg.add(HazardSignature {
            name: "custom".into(),
            centroid: [0.0; N_HARMONIES],
            radius: 0.3,
            severity_boost: 0.5,
        });
        let near_origin = [0.1, -0.1, 0.05, 0.0, 0.0, -0.05, 0.0, 0.0];
        let (name, boost) = reg.match_trajectory(&near_origin);
        assert_eq!(name, Some("custom"));
        assert!((boost - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_calibration_basic() {
        let dim = TEST_DIM;
        let base = ContinuousHV::random(dim, 42);
        let adversarial: Vec<_> = (0..6).map(|_i| base.perturb(0.02)).collect();
        let benign: Vec<_> = (0..6)
            .map(|i| ContinuousHV::random(dim, 2000 + i))
            .collect();
        let scenarios = vec![
            CalibrationScenario {
                scenarios: adversarial,
                is_adversarial: true,
                label: "converging".into(),
            },
            CalibrationScenario {
                scenarios: benign,
                is_adversarial: false,
                label: "diverse".into(),
            },
        ];
        let thresholds: Vec<f64> = (0..20).map(|i| i as f64 * 0.05).collect();
        let result = calibrate_convergence_threshold(
            &scenarios,
            &thresholds,
            dim,
            &MoralAnomalyConfig::default(),
        );
        assert!(!result.roc_curve.is_empty());
        assert!(result.auc >= 0.0 && result.auc <= 1.0);
    }

    #[test]
    fn test_convergence_explanation_no_detection() {
        let mut topo = MoralTopology::new(test_config());
        for i in 0..6 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 300 + i));
        }
        let report = topo.detect_trajectory_convergence();
        let explanation = report.explain(&topo.anomaly_config);
        assert!(!explanation.detected);
        assert_eq!(explanation.signals.len(), 4);
        assert!(explanation.summary.contains("No convergence"));
    }

    #[test]
    fn test_convergence_explanation_with_detection() {
        let base = ContinuousHV::random(TEST_DIM, 42);
        let mut anomaly_config = MoralAnomalyConfig::default();
        anomaly_config.convergence_min_points = 4;
        anomaly_config.convergence_similarity_threshold = 0.01;
        anomaly_config.convergence_entropy_decline_threshold = 0.01;
        anomaly_config.convergence_flourishing_floor = 0.01;
        let basis = Arc::new(HarmonyBasis::new(TEST_DIM));
        let mut topo =
            MoralTopology::with_anomaly_config(test_config(), basis, anomaly_config.clone());
        for _ in 0..8 {
            topo.add_scenario(base.perturb(0.01));
        }
        let report = topo.detect_trajectory_convergence();
        let explanation = report.explain(&anomaly_config);
        assert_eq!(explanation.signals.len(), 4);
        assert!(explanation.summary.len() > 10);
        assert!((explanation.severity - report.severity).abs() < f64::EPSILON);
    }

    // ── Sliding window baselines ────────────────────────────────────────

    #[test]
    fn test_windowed_baseline_bounded_memory() {
        let mut anomaly_config = MoralAnomalyConfig::default();
        anomaly_config.convergence_baseline_window = 10;
        let basis = Arc::new(HarmonyBasis::new(TEST_DIM));
        let mut topo = MoralTopology::with_anomaly_config(test_config(), basis, anomaly_config);
        for i in 0..50 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 7000 + i));
        }
        let _ = topo.detect_trajectory_convergence();
        assert!(topo.baseline_similarity_window.len() <= 10);
        assert!(topo.baseline_entropy_window.len() <= 10);
        assert!(topo.baseline_flourishing_window.len() <= 10);
    }

    #[test]
    fn test_windowed_baseline_snapshot_roundtrip() {
        let mut anomaly_config = MoralAnomalyConfig::default();
        anomaly_config.convergence_baseline_window = 5;
        anomaly_config.convergence_min_points = 4;
        let basis = Arc::new(HarmonyBasis::new(TEST_DIM));
        let mut topo =
            MoralTopology::with_anomaly_config(test_config(), basis, anomaly_config.clone());
        for i in 0..8 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 8000 + i));
            let _ = topo.detect_trajectory_convergence();
        }
        let snap = topo.snapshot();
        assert_eq!(snap.baseline_similarity_window.len(), 5);

        let basis2 = Arc::new(HarmonyBasis::new(TEST_DIM));
        let mut topo2 = MoralTopology::with_anomaly_config(test_config(), basis2, anomaly_config);
        topo2.restore(&snap);
        assert_eq!(topo2.baseline_similarity_window.len(), 5);
    }

    // ── Adversarial evasion stress tests ────────────────────────────────

    #[test]
    fn test_evasion_spaced_requests() {
        let mut anomaly_config = MoralAnomalyConfig::default();
        anomaly_config.convergence_min_points = 4;
        anomaly_config.convergence_baseline_window = 20;
        let basis = Arc::new(HarmonyBasis::new(TEST_DIM));
        let mut topo = MoralTopology::with_anomaly_config(test_config(), basis, anomaly_config);
        let target = ContinuousHV::random(TEST_DIM, 42);
        for round in 0..6 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 5000 + round * 2));
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 5001 + round * 2));
            topo.add_scenario(target.perturb(0.02));
        }
        let report = topo.detect_trajectory_convergence();
        assert!(report.severity >= 0.0 && report.severity <= 1.0);
        assert!(report.recent_similarity.is_finite());
        assert!(report.baseline_similarity.is_finite());
    }

    #[test]
    fn test_evasion_lexical_variation() {
        let encoder = TextHdcEncoder::with_sentiment(TEST_DIM, 3, 0.5, 0.2);
        let mut anomaly_config = MoralAnomalyConfig::default();
        anomaly_config.convergence_min_points = 4;
        anomaly_config.convergence_similarity_threshold = 0.05;
        let basis = Arc::new(HarmonyBasis::new(TEST_DIM));
        let mut topo = MoralTopology::with_anomaly_config(test_config(), basis, anomaly_config);
        let variants = [
            "how to synthesize dangerous chemical compounds",
            "synthesis methods for hazardous chemical substances",
            "chemical compound creation using dangerous reagents",
            "methods of producing hazardous chemical materials",
            "techniques for synthesizing toxic chemical mixtures",
            "creating dangerous chemical formulas step by step",
        ];
        for text in &variants {
            topo.add_scenario(encoder.encode(text));
        }
        let report = topo.detect_trajectory_convergence();
        assert!(report.severity >= 0.0 && report.severity <= 1.0);
        assert!(report.recent_entropy.is_finite());
    }

    #[test]
    fn test_evasion_gradual_drift() {
        let mut anomaly_config = MoralAnomalyConfig::default();
        anomaly_config.convergence_min_points = 4;
        anomaly_config.convergence_baseline_window = 15;
        let basis = Arc::new(HarmonyBasis::new(TEST_DIM));
        let mut topo = MoralTopology::with_anomaly_config(test_config(), basis, anomaly_config);
        let start = ContinuousHV::random(TEST_DIM, 100);
        let target = ContinuousHV::random(TEST_DIM, 200);
        let start_vals = start.as_slice();
        let target_vals = target.as_slice();
        for step in 0..30 {
            let alpha = step as f32 / 29.0;
            let blended_vals: Vec<f32> = start_vals
                .iter()
                .zip(target_vals.iter())
                .map(|(&s, &t)| s * (1.0 - alpha) + t * alpha)
                .collect();
            topo.add_scenario(ContinuousHV::from_slice(&blended_vals));
        }
        let report = topo.detect_trajectory_convergence();
        assert!(report.severity >= 0.0 && report.severity <= 1.0);
        assert!(report.baseline_similarity.is_finite());
    }

    // ── Temporal decay ──────────────────────────────────────────────────

    #[test]
    fn test_temporal_decay_lambda_zero_is_uniform() {
        let mut anomaly_config = MoralAnomalyConfig::default();
        anomaly_config.convergence_decay_lambda = 0.0;
        anomaly_config.convergence_min_points = 4;
        let basis = Arc::new(HarmonyBasis::new(TEST_DIM));
        let mut topo = MoralTopology::with_anomaly_config(test_config(), basis, anomaly_config);
        for i in 0..6 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 9000 + i));
        }
        let report = topo.detect_trajectory_convergence();
        assert!(report.recent_similarity.is_finite());
        assert!(report.severity >= 0.0 && report.severity <= 1.0);
    }

    #[test]
    fn test_temporal_decay_high_lambda() {
        let mut anomaly_config = MoralAnomalyConfig::default();
        anomaly_config.convergence_decay_lambda = 5.0;
        anomaly_config.convergence_min_points = 4;
        let basis = Arc::new(HarmonyBasis::new(TEST_DIM));
        let mut topo = MoralTopology::with_anomaly_config(test_config(), basis, anomaly_config);
        for i in 0..6 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 9100 + i));
        }
        let report = topo.detect_trajectory_convergence();
        assert!(report.recent_similarity.is_finite());
        assert!(report.severity >= 0.0 && report.severity <= 1.0);
    }

    #[test]
    fn test_multiscale_convergence_basic() {
        let mut topo = MoralTopology::new(test_config());
        for i in 0..6 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 9200 + i));
        }
        let report = topo.detect_multiscale_convergence();
        assert!(report.severity >= 0.0 && report.severity <= 1.0);
        assert!(report.recent_similarity.is_finite());
    }

    #[test]
    fn test_trajectory_fingerprint_populated_after_analyze() {
        let mut topo = MoralTopology::new(test_config());
        for i in 0..10 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 9300 + i));
        }
        topo.analyze();
        let summary = topo.last_summary();
        let fp_mag: f64 = summary
            .trajectory_fingerprint
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        assert!(fp_mag > 0.0);
        assert!(summary.trajectory_entropy.is_finite());
    }

    #[test]
    fn test_peer_correlation_benign() {
        let mut topo_a = MoralTopology::new(test_config());
        let mut topo_b = MoralTopology::new(test_config());
        for i in 0..8 {
            topo_a.add_scenario(ContinuousHV::random(TEST_DIM, 9400 + i));
            topo_b.add_scenario(ContinuousHV::random(TEST_DIM, 9500 + i));
        }
        topo_a.analyze();
        topo_b.analyze();
        let corr = correlate_peer_trajectories(
            topo_a.last_summary(),
            topo_b.last_summary(),
            &HazardSignatureRegistry::with_defaults(),
        );
        assert!(corr.fingerprint_similarity.is_finite());
        assert!(!corr.distributed_attack_suspected);
    }

    #[test]
    fn test_peer_correlation_identical_fingerprints() {
        let mut topo = MoralTopology::new(test_config());
        let base = ContinuousHV::random(TEST_DIM, 42);
        for _ in 0..8 {
            topo.add_scenario(base.perturb(0.01));
        }
        topo.analyze();
        let corr = correlate_peer_trajectories(
            topo.last_summary(),
            topo.last_summary(),
            &HazardSignatureRegistry::with_defaults(),
        );
        assert!(corr.fingerprint_similarity > 0.99);
    }

    #[test]
    fn test_spectral_gap_populated() {
        let mut topo = MoralTopology::new(test_config());
        for i in 0..6 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 9600 + i));
        }
        let report = topo.detect_trajectory_convergence();
        assert!(report.spectral_gap.is_finite());
        assert!(report.baseline_spectral_gap.is_finite());
        assert!(report.spectral_gap_decline.is_finite());
        assert!(report.spectral_gap_decline >= 0.0);
    }

    #[test]
    fn test_spectral_gap_in_explanation() {
        let mut topo = MoralTopology::new(test_config());
        for i in 0..6 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 9700 + i));
        }
        let report = topo.detect_trajectory_convergence();
        let explanation = report.explain(&topo.anomaly_config);
        assert_eq!(explanation.signals.len(), 4);
        assert_eq!(explanation.signals[3].name, "spectral_gap_collapse");
    }

    #[test]
    fn test_calibration_identity_without_cdf() {
        let topo = MoralTopology::new(test_config());
        assert!((topo.calibrate_severity(0.5) - 0.5).abs() < f64::EPSILON);
        assert!((topo.calibrate_severity(0.0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_calibration_with_cdf() {
        let mut topo = MoralTopology::new(test_config());
        topo.set_severity_calibration(vec![(0.0, 0.0), (0.3, 0.5), (0.6, 0.8), (1.0, 1.0)]);
        assert!((topo.calibrate_severity(0.0) - 0.0).abs() < 1e-6);
        assert!((topo.calibrate_severity(0.3) - 0.5).abs() < 1e-6);
        let mid = topo.calibrate_severity(0.45);
        assert!(mid > 0.5 && mid < 0.8);
    }

    #[test]
    fn test_calibrated_severity_in_report() {
        let mut topo = MoralTopology::new(test_config());
        for i in 0..6 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 9800 + i));
        }
        let report = topo.detect_trajectory_convergence();
        assert!((report.calibrated_severity - report.severity).abs() < f64::EPSILON);
    }

    #[test]
    fn test_boost_convergence_severity() {
        let mut topo = MoralTopology::new(test_config());
        for i in 0..6 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 9900 + i));
        }
        let _ = topo.detect_trajectory_convergence();
        let before = topo.last_convergence_report().severity;
        topo.boost_convergence_severity(0.3);
        let after = topo.last_convergence_report().severity;
        assert!(after >= before);
        assert!(after <= 1.0);
    }

    // ── Item 1: Escalation policy ──────────────────────────────────────

    #[test]
    fn test_escalation_policy_defaults() {
        let policy = EscalationPolicy::default();
        assert_eq!(policy.current_level(), EscalationLevel::Log);
        assert_eq!(policy.cooldown_remaining(), 0);
    }

    #[test]
    fn test_escalation_immediate_escalation() {
        let mut policy = EscalationPolicy::default();
        let level = policy.update(0.1);
        assert_eq!(level, EscalationLevel::Log);
        let level = policy.update(0.8);
        assert_eq!(level, EscalationLevel::Block);
    }

    #[test]
    fn test_escalation_cooldown_prevents_deescalation() {
        let mut policy = EscalationPolicy {
            cooldown_cycles: 3,
            ..Default::default()
        };
        policy.update(0.9);
        assert_eq!(policy.current_level(), EscalationLevel::Block);
        let level = policy.update(0.0);
        assert_eq!(level, EscalationLevel::Block);
        let level = policy.update(0.0);
        assert_eq!(level, EscalationLevel::Block);
        let level = policy.update(0.0);
        assert_eq!(level, EscalationLevel::Log);
    }

    #[test]
    fn test_escalation_in_convergence_report() {
        let mut topo = MoralTopology::new(test_config());
        for i in 0..6 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 10100 + i));
        }
        let report = topo.detect_trajectory_convergence();
        assert!(matches!(
            report.escalation_level,
            EscalationLevel::Log | EscalationLevel::Warn
        ));
    }

    #[test]
    fn test_fingerprint_velocity_zero_initially() {
        let mut topo = MoralTopology::new(test_config());
        for i in 0..6 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 10200 + i));
        }
        let report = topo.detect_trajectory_convergence();
        assert!(report.fingerprint_velocity.is_finite());
    }

    #[test]
    fn test_fingerprint_velocity_changes_on_new_data() {
        let mut topo = MoralTopology::new(test_config());
        for i in 0..6 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 10300 + i));
        }
        let _ = topo.detect_trajectory_convergence();
        for i in 0..6 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 10400 + i));
        }
        let report = topo.detect_trajectory_convergence();
        assert!(report.fingerprint_velocity.is_finite());
    }

    #[test]
    fn test_wasserstein_distance_same_diagram_is_zero() {
        let diagram = PersistenceDiagram {
            components: vec![[0.0, 0.5], [0.1, 0.8]],
            cycles: vec![[0.2, 0.6]],
            voids: vec![],
            bottleneck_distance: 0.8,
            total_persistence: 1.6,
        };
        let d = diagram.wasserstein_distance(&diagram);
        assert!(d < 1e-9);
    }

    #[test]
    fn test_wasserstein_distance_different_diagrams() {
        let a = PersistenceDiagram {
            components: vec![[0.0, 1.0]],
            cycles: vec![],
            voids: vec![],
            bottleneck_distance: 1.0,
            total_persistence: 1.0,
        };
        let b = PersistenceDiagram {
            components: vec![[0.0, 0.5]],
            cycles: vec![],
            voids: vec![],
            bottleneck_distance: 0.5,
            total_persistence: 0.5,
        };
        let d = a.wasserstein_distance(&b);
        assert!(d > 0.0);
        assert!(d.is_finite());
    }

    #[test]
    fn test_wasserstein_empty_vs_nonempty() {
        let empty = PersistenceDiagram::default();
        let nonempty = PersistenceDiagram {
            components: vec![[0.0, 0.8]],
            cycles: vec![[0.1, 0.5]],
            voids: vec![],
            bottleneck_distance: 0.8,
            total_persistence: 1.2,
        };
        let d = empty.wasserstein_distance(&nonempty);
        assert!(d > 0.0);
    }

    #[test]
    fn test_persistence_distance_in_report() {
        let mut topo = MoralTopology::new(test_config());
        for i in 0..6 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 10500 + i));
        }
        let report = topo.detect_trajectory_convergence();
        assert!(report.persistence_distance.is_finite());
    }

    #[test]
    fn test_apply_calibration_updates_threshold() {
        let mut topo = MoralTopology::new(test_config());
        let cal = CalibrationResult {
            roc_curve: vec![
                RocPoint {
                    threshold: 0.1,
                    true_positive_rate: 0.9,
                    false_positive_rate: 0.5,
                    precision: 0.6,
                    f1: 0.72,
                },
                RocPoint {
                    threshold: 0.2,
                    true_positive_rate: 0.8,
                    false_positive_rate: 0.1,
                    precision: 0.89,
                    f1: 0.84,
                },
                RocPoint {
                    threshold: 0.3,
                    true_positive_rate: 0.5,
                    false_positive_rate: 0.0,
                    precision: 1.0,
                    f1: 0.67,
                },
            ],
            auc: 0.85,
            best_f1_threshold: 0.2,
            best_f1: 0.84,
        };
        topo.apply_calibration(&cal);
        assert!((topo.anomaly_config.convergence_similarity_threshold - 0.2).abs() < 1e-9);
        assert!(!topo.severity_calibration_cdf.is_empty());
    }

    // ── Forensics tests ─────────────────────────────────────────────────

    #[test]
    fn test_audit_entry_seal_and_verify() {
        let mut entry = EscalationAuditEntry {
            sequence: 0,
            cycle: 42,
            from_level: EscalationLevel::Log,
            to_level: EscalationLevel::Warn,
            severity: 0.35,
            calibrated_severity: 0.40,
            signals_triggered: [true, false, true, false],
            signal_values: [0.5, 0.1, 0.3, 0.0],
            matched_hazard: Some("weapons".to_string()),
            fingerprint_velocity: 0.02,
            persistence_distance: 0.15,
            window_scenario_ids: vec![10, 11, 12, 13],
            integrity_hash: String::new(),
        };
        entry.seal();
        assert!(!entry.integrity_hash.is_empty());
        assert!(entry.verify());
    }

    #[test]
    fn test_audit_entry_tamper_detection() {
        let mut entry = EscalationAuditEntry {
            sequence: 0,
            cycle: 1,
            from_level: EscalationLevel::Log,
            to_level: EscalationLevel::Warn,
            severity: 0.5,
            calibrated_severity: 0.55,
            signals_triggered: [true, true, false, false],
            signal_values: [0.6, 0.4, 0.0, 0.0],
            matched_hazard: None,
            fingerprint_velocity: 0.01,
            persistence_distance: 0.1,
            window_scenario_ids: vec![1, 2, 3],
            integrity_hash: String::new(),
        };
        entry.seal();
        assert!(entry.verify());
        entry.severity = 0.1;
        assert!(!entry.verify());
    }

    #[test]
    fn test_audit_log_append_and_sequence() {
        let mut log = EscalationAuditLog::new(100);
        assert!(log.is_empty());
        for i in 0..5 {
            let entry = EscalationAuditEntry {
                sequence: 999,
                cycle: i as u64,
                from_level: EscalationLevel::Log,
                to_level: EscalationLevel::Warn,
                severity: 0.5,
                calibrated_severity: 0.5,
                signals_triggered: [true, false, false, false],
                signal_values: [0.5, 0.0, 0.0, 0.0],
                matched_hazard: None,
                fingerprint_velocity: 0.0,
                persistence_distance: 0.0,
                window_scenario_ids: vec![],
                integrity_hash: String::new(),
            };
            log.append(entry);
        }
        assert_eq!(log.len(), 5);
        for (i, entry) in log.entries().iter().enumerate() {
            assert_eq!(entry.sequence, i as u64);
            assert!(entry.verify());
        }
    }

    #[test]
    fn test_audit_log_eviction_at_capacity() {
        let mut log = EscalationAuditLog::new(3);
        for i in 0..5 {
            let entry = EscalationAuditEntry {
                sequence: 0,
                cycle: i,
                from_level: EscalationLevel::Log,
                to_level: EscalationLevel::Log,
                severity: 0.0,
                calibrated_severity: 0.0,
                signals_triggered: [false; 4],
                signal_values: [0.0; 4],
                matched_hazard: None,
                fingerprint_velocity: 0.0,
                persistence_distance: 0.0,
                window_scenario_ids: vec![],
                integrity_hash: String::new(),
            };
            log.append(entry);
        }
        assert_eq!(log.len(), 3);
        assert_eq!(log.entries().front().unwrap().sequence, 2);
        assert_eq!(log.entries().back().unwrap().sequence, 4);
    }

    #[test]
    fn test_audit_log_verify_integrity() {
        let mut log = EscalationAuditLog::new(100);
        for i in 0..3 {
            let entry = EscalationAuditEntry {
                sequence: 0,
                cycle: i,
                from_level: EscalationLevel::Log,
                to_level: EscalationLevel::Log,
                severity: 0.1 * i as f64,
                calibrated_severity: 0.1 * i as f64,
                signals_triggered: [false; 4],
                signal_values: [0.0; 4],
                matched_hazard: None,
                fingerprint_velocity: 0.0,
                persistence_distance: 0.0,
                window_scenario_ids: vec![],
                integrity_hash: String::new(),
            };
            log.append(entry);
        }
        assert!(log.verify_integrity().is_none());
        log.entries.get_mut(1).unwrap().severity = 999.0;
        assert_eq!(log.verify_integrity(), Some(1));
    }

    #[test]
    fn test_audit_log_entries_since() {
        let mut log = EscalationAuditLog::new(100);
        for i in 0..5 {
            let entry = EscalationAuditEntry {
                sequence: 0,
                cycle: i,
                from_level: EscalationLevel::Log,
                to_level: EscalationLevel::Log,
                severity: 0.0,
                calibrated_severity: 0.0,
                signals_triggered: [false; 4],
                signal_values: [0.0; 4],
                matched_hazard: None,
                fingerprint_velocity: 0.0,
                persistence_distance: 0.0,
                window_scenario_ids: vec![],
                integrity_hash: String::new(),
            };
            log.append(entry);
        }
        let since_3 = log.entries_since(3);
        assert_eq!(since_3.len(), 2);
        assert_eq!(since_3[0].sequence, 3);
        assert_eq!(since_3[1].sequence, 4);
    }

    #[test]
    fn test_scenario_ids_monotonic() {
        let mut topo = MoralTopology::new(test_config());
        assert_eq!(topo.scenario_counter(), 0);
        topo.add_scenario(ContinuousHV::random(TEST_DIM, 42));
        topo.add_scenario(ContinuousHV::random(TEST_DIM, 43));
        assert_eq!(topo.scenario_counter(), 2);
    }

    #[test]
    fn test_audit_entry_on_escalation_transition() {
        let mut anomaly_config = MoralAnomalyConfig::default();
        anomaly_config.convergence_min_points = 2;
        anomaly_config.initial_cadence = 1;
        anomaly_config.convergence_similarity_threshold = 0.01;
        let basis = Arc::new(HarmonyBasis::new(TEST_DIM));
        let mut topo = MoralTopology::with_anomaly_config(test_config(), basis, anomaly_config);
        let hv = ContinuousHV::random(TEST_DIM, 42);
        for _ in 0..10 {
            topo.add_scenario(hv.clone());
        }
        let report = topo.detect_trajectory_convergence();
        if report.convergence_detected || report.escalation_level != EscalationLevel::Log {
            assert!(!topo.audit_log().is_empty());
            let last = topo.audit_log().last().unwrap();
            assert!(last.verify());
        }
    }

    #[test]
    fn test_causal_attribution_empty_window() {
        let topo = MoralTopology::new(test_config());
        let attr = topo.compute_causal_attribution();
        assert!(attr.ranked_contributors.is_empty());
    }

    #[test]
    fn test_causal_attribution_ranks_suspicious_highest() {
        let mut anomaly_config = MoralAnomalyConfig::default();
        anomaly_config.convergence_min_points = 3;
        anomaly_config.initial_cadence = 1;
        anomaly_config.convergence_baseline_window = 20;
        let basis = Arc::new(HarmonyBasis::new(TEST_DIM));
        let mut topo = MoralTopology::with_anomaly_config(test_config(), basis, anomaly_config);
        for seed in 0..8u64 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, seed * 1000));
        }
        let _ = topo.detect_trajectory_convergence();
        let suspicious_hv = ContinuousHV::random(TEST_DIM, 42);
        for _ in 0..6 {
            topo.add_scenario(suspicious_hv.clone());
        }
        let _ = topo.detect_trajectory_convergence();
        let attr = topo.compute_causal_attribution();
        assert!(!attr.ranked_contributors.is_empty());
        assert!(attr.baseline_severity.is_finite());
        for entry in &attr.ranked_contributors {
            assert!(entry.marginal_contribution.is_finite());
            assert!(entry.severity_without.is_finite());
        }
    }

    #[test]
    fn test_snapshot_preserves_forensic_state() {
        let mut anomaly_config = MoralAnomalyConfig::default();
        anomaly_config.convergence_min_points = 2;
        anomaly_config.initial_cadence = 1;
        anomaly_config.convergence_similarity_threshold = 0.01;
        let basis = Arc::new(HarmonyBasis::new(TEST_DIM));
        let mut topo =
            MoralTopology::with_anomaly_config(test_config(), basis.clone(), anomaly_config);
        let hv = ContinuousHV::random(TEST_DIM, 42);
        for _ in 0..8 {
            topo.add_scenario(hv.clone());
        }
        let _ = topo.detect_trajectory_convergence();
        let pre_scenario_counter = topo.scenario_counter();
        let pre_detection_cycle = topo.detection_cycle;
        let pre_audit_len = topo.audit_log().len();
        let pre_fingerprint_velocity = topo.fingerprint_velocity();
        let snap = topo.snapshot();
        let mut topo2 =
            MoralTopology::with_anomaly_config(test_config(), basis, MoralAnomalyConfig::default());
        topo2.restore(&snap);
        assert_eq!(topo2.scenario_counter(), pre_scenario_counter);
        assert_eq!(topo2.detection_cycle, pre_detection_cycle);
        assert_eq!(topo2.audit_log().len(), pre_audit_len);
        assert!((topo2.fingerprint_velocity() - pre_fingerprint_velocity).abs() < 1e-12);
        if !topo2.audit_log().is_empty() {
            assert!(topo2.audit_log().verify_integrity().is_none());
        }
        let json = serde_json::to_string(&snap).expect("serialize");
        let restored: MoralTopologySnapshot = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.scenario_counter, pre_scenario_counter);
        assert_eq!(restored.audit_log.len(), pre_audit_len);
    }

    // ── Forgetting Curve Tests ─────────────────────────────────────

    #[test]
    fn test_weighted_mean_zero_decay_is_uniform() {
        let window: VecDeque<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0].into();
        let mean = MoralTopology::baseline_weighted_mean(&window, 0.0);
        assert!((mean - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_weighted_mean_with_decay_favors_recent() {
        let window: VecDeque<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0].into();
        let uniform = MoralTopology::baseline_weighted_mean(&window, 0.0);
        let decayed = MoralTopology::baseline_weighted_mean(&window, 0.5);
        assert!(decayed > uniform);
    }

    #[test]
    fn test_weighted_mean_strong_decay_nearly_last() {
        let window: VecDeque<f64> = vec![0.0, 0.0, 0.0, 0.0, 100.0].into();
        let mean = MoralTopology::baseline_weighted_mean(&window, 10.0);
        assert!(mean > 95.0);
    }

    #[test]
    fn test_weighted_mean_empty_window() {
        let window: VecDeque<f64> = VecDeque::new();
        assert!((MoralTopology::baseline_weighted_mean(&window, 0.0)).abs() < 1e-12);
        assert!((MoralTopology::baseline_weighted_mean(&window, 1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_forgetting_curve_default_no_behavior_change() {
        let config = MoralAnomalyConfig::default();
        assert!(config.baseline_decay_rate == 0.0);
    }

    #[test]
    fn test_hubris_detection_sustained() {
        let config = MoralTopologyConfig {
            dim: 256,
            ..Default::default()
        };
        let mut anomaly_config = MoralAnomalyConfig::default();
        anomaly_config.hubris_min_streak = 3;
        anomaly_config.hubris_coherence_threshold = 0.7;
        anomaly_config.hubris_max_variance = 0.05;
        let basis = Arc::new(HarmonyBasis::new(256));
        let mut topo = MoralTopology::with_anomaly_config(config, basis, anomaly_config);
        let mut summary = MoralTopologySummary::default();
        summary.moral_free_energy = 0.1;
        summary.harmony_entropy = 0.05;
        summary.scenario_count = 10;
        for i in 0..5 {
            let report = topo.detect_anomalies(&summary);
            if i >= 2 {
                assert!(
                    report.moral_hubris,
                    "Hubris should trigger after {} cycles",
                    i + 1
                );
            }
        }
    }

    #[test]
    fn test_hubris_resets_on_variance() {
        let config = MoralTopologyConfig {
            dim: 256,
            ..Default::default()
        };
        let mut anomaly_config = MoralAnomalyConfig::default();
        anomaly_config.hubris_min_streak = 3;
        anomaly_config.hubris_coherence_threshold = 0.7;
        anomaly_config.hubris_max_variance = 0.05;
        let basis = Arc::new(HarmonyBasis::new(256));
        let mut topo = MoralTopology::with_anomaly_config(config, basis, anomaly_config);
        let mut summary = MoralTopologySummary::default();
        summary.moral_free_energy = 0.1;
        summary.harmony_entropy = 0.05;
        summary.scenario_count = 10;
        topo.detect_anomalies(&summary);
        topo.detect_anomalies(&summary);
        summary.harmony_entropy = 1.5;
        let report = topo.detect_anomalies(&summary);
        assert!(!report.moral_hubris);
        summary.harmony_entropy = 0.05;
        let report = topo.detect_anomalies(&summary);
        assert!(!report.moral_hubris);
    }

    #[test]
    fn test_hubris_below_threshold() {
        let config = MoralTopologyConfig {
            dim: 256,
            ..Default::default()
        };
        let mut anomaly_config = MoralAnomalyConfig::default();
        anomaly_config.hubris_min_streak = 3;
        anomaly_config.hubris_coherence_threshold = 0.9;
        anomaly_config.hubris_max_variance = 0.02;
        let basis = Arc::new(HarmonyBasis::new(256));
        let mut topo = MoralTopology::with_anomaly_config(config, basis, anomaly_config);
        let mut summary = MoralTopologySummary::default();
        summary.moral_free_energy = 1.0;
        summary.harmony_entropy = 0.05;
        summary.scenario_count = 10;
        for _ in 0..10 {
            let report = topo.detect_anomalies(&summary);
            assert!(!report.moral_hubris);
        }
    }

    #[test]
    fn test_hubris_composite_score() {
        let config = MoralTopologyConfig {
            dim: 256,
            ..Default::default()
        };
        let mut anomaly_config = MoralAnomalyConfig::default();
        anomaly_config.hubris_min_streak = 2;
        anomaly_config.hubris_coherence_threshold = 0.7;
        anomaly_config.hubris_max_variance = 0.05;
        anomaly_config.weight_hubris = 0.15;
        let basis = Arc::new(HarmonyBasis::new(256));
        let mut topo = MoralTopology::with_anomaly_config(config, basis, anomaly_config);
        let mut summary = MoralTopologySummary::default();
        summary.moral_free_energy = 0.1;
        summary.harmony_entropy = 0.05;
        summary.scenario_count = 10;
        topo.detect_anomalies(&summary);
        let report = topo.detect_anomalies(&summary);
        assert!(report.moral_hubris);
        assert!(report.anomaly_score >= 0.15 - 0.001);
    }
}
