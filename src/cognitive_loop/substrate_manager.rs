// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Substrate independence manager.
//!
//! Consolidates the 6 substrate-related fields from CognitiveLoopService
//! into a single cohesive manager. Handles feasibility computation,
//! validation overlays, speed/scale modulation, and runtime reconfiguration.

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use symthaea_core::hdc::substrate_independence::{
    CorticalRegion, SubstrateRequirements, SubstrateType,
};

use super::types::{EffectiveDimSource, RecurrentMaskEvent};

use super::config::CognitiveLoopConfig;
use super::thresholds::{
    SUBSTRATE_MIN_DIM_FRACTION, SUBSTRATE_OPS_PER_CYCLE, SUBSTRATE_SCALE_DIM_DIVISOR,
    SUBSTRATE_TRANSITION_HISTORY_CAP,
};

/// Default honest confidence for substrates not in the validation framework.
/// Matches EvidenceLevel::Theoretical.confidence() = 0.10.
const THEORETICAL_CONFIDENCE: f64 = 0.10;

/// Do not compare a per-cycle energy estimate to a watts budget until at least
/// one wall-clock second of monotonic observation has elapsed.
const ENERGY_RATE_WINDOW_MIN_SECONDS: f64 = 1.0;

/// Record of a substrate transition event.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SubstrateTransitionRecord {
    /// Cycle number when the transition occurred (0 = pre-run).
    pub cycle: u64,
    /// Previous substrate type.
    pub from: SubstrateType,
    /// New substrate type.
    pub to: SubstrateType,
    /// Feasibility before transition.
    pub old_feasibility: f64,
    /// Feasibility after transition.
    pub new_feasibility: f64,
}

/// Manages substrate-dependent consciousness feasibility, validation overlays,
/// and speed/scale modulation factors.
#[derive(Debug, Clone)]
pub(crate) struct SubstrateManager {
    /// Pre-computed substrate feasibility [0,1] from config.substrate_type.
    /// Scales Equation V2 consciousness to reflect substrate limitations.
    pub(crate) feasibility: f64,

    /// Pending substrate transition description for telemetry.
    /// Populated by `reconfigure_substrate()`/`reconfigure_composition()`,
    /// drained into `CycleMetadata.substrate_transition` once per cycle.
    pub(crate) pending_transition: Option<String>,

    /// Honest evidence confidence for the current substrate (0.0–0.95).
    /// From SubstrateValidationFramework: biological=0.95, silicon=0.10, etc.
    pub(crate) honest_confidence: f64,

    /// Effective feasibility after validation overlay blending.
    /// When overlay disabled: equals feasibility.
    /// When enabled: feasibility × (floor + (1 − floor) × honest_confidence).
    pub(crate) effective_feasibility: f64,

    /// CfC tau factor from substrate speed modulation [0.5, 2.0].
    /// 1.0 when speed modulation is disabled.
    pub(crate) tau_factor: f32,

    /// Scale pressure: log10(substrate_max_scale / bio_max_scale).
    /// Telemetry-only. 0.0 when speed modulation is disabled.
    pub(crate) scale_pressure: f32,

    /// Mirror of `CognitiveLoopConfig::effective_dim_fraction_override`, already
    /// validated and clamped to [0.0, 1.0]. Refreshed from config in `new()` and
    /// `recompute_substrate_dynamics()` so it cannot drift.
    pub(crate) dim_fraction_override: Option<f32>,

    /// Most recent recurrent-mask event, recorded by the mask site and drained
    /// by `telemetry()`.
    pub(crate) last_mask_event: Option<RecurrentMaskEvent>,

    /// Whether consciousness is still viable under energy constraints.
    /// False when energy budget is exhausted.
    pub(crate) consciousness_viable: bool,

    /// Total energy spent so far (joules).
    pub(crate) total_energy_spent: f64,

    /// Energy spent per cycle (joules).
    pub(crate) energy_per_cycle: f64,

    /// Throughput multiplier derived from substrate energy efficiency.
    /// Higher = more efficient substrate = more ops per joule.
    pub(crate) energy_throughput_multiplier: f32,

    /// Per-region substrate types (None = use global substrate for all).
    per_region_substrates: Option<HashMap<CorticalRegion, SubstrateType>>,

    /// Per-region feasibility scores.
    per_region_feasibility: HashMap<CorticalRegion, f32>,

    // ── Phase 3: Transition smoothing ────────────────────────────────────
    /// Target effective feasibility after transition (for EMA blending).
    target_effective_feasibility: f64,

    /// Target tau_factor after transition (for EMA blending).
    target_tau_factor: f32,

    /// Transition history log (ring buffer).
    transition_history: VecDeque<SubstrateTransitionRecord>,

    /// Current substrate type (for transition logging).
    current_substrate: SubstrateType,

    /// Energy spent in the most recent tick (for telemetry).
    #[cfg(feature = "jepa")]
    pub(crate) jepa_energy: f64,
    /// Modeled energy accumulated since the last closed monotonic-rate window.
    energy_rate_window_joules: f64,
    /// Monotonic start of the current energy-rate observation window.
    energy_rate_window_started_at: Option<Instant>,
    /// Average model-estimated energy rate over the most recently closed window (watts).
    estimated_energy_rate_watts: f64,
    /// Duration of the most recently closed energy-rate window (seconds).
    energy_rate_window_seconds: f64,
    last_energy_spent: f64,
}

impl SubstrateManager {
    /// Create a new substrate manager from config.
    pub fn new(config: &CognitiveLoopConfig) -> Self {
        let feasibility = if let Some(ref comp) = config.substrate_composition {
            comp.feasibility()
        } else {
            Self::requirements_for(&config.substrate_type).consciousness_feasibility()
        };

        // Compute energy per cycle from substrate energy_per_op
        let energy_per_op = config.substrate_type.energy_per_operation();
        let energy_per_cycle = energy_per_op * SUBSTRATE_OPS_PER_CYCLE;

        // Throughput multiplier: ratio of bio energy to this substrate's energy
        let bio_energy = SubstrateType::BiologicalNeurons.energy_per_operation();
        let energy_throughput_multiplier = (bio_energy / energy_per_op).clamp(0.1, 100.0) as f32;

        let mut mgr = Self {
            feasibility,
            pending_transition: None,
            honest_confidence: 0.0,
            effective_feasibility: feasibility,
            tau_factor: 1.0,
            scale_pressure: 0.0,
            dim_fraction_override: Self::validate_dim_override(
                config.effective_dim_fraction_override,
            ),
            last_mask_event: None,
            consciousness_viable: true,
            total_energy_spent: 0.0,
            energy_per_cycle,
            energy_throughput_multiplier,
            per_region_substrates: config.per_region_substrates.clone(),
            per_region_feasibility: HashMap::new(),
            target_effective_feasibility: feasibility,
            target_tau_factor: 1.0,
            transition_history: VecDeque::new(),
            current_substrate: config.substrate_type,
            last_energy_spent: 0.0,
            energy_rate_window_joules: 0.0,
            energy_rate_window_started_at: None,
            estimated_energy_rate_watts: 0.0,
            energy_rate_window_seconds: 0.0,
            #[cfg(feature = "jepa")]
            jepa_energy: 0.0,
        };
        mgr.recompute_effective_feasibility(config);
        mgr.recompute_substrate_dynamics(config);
        mgr.recompute_per_region_feasibility(config);
        mgr
    }

    /// Switch substrate type at runtime, recomputing consciousness feasibility.
    ///
    /// Returns (old_feasibility, new_feasibility) for telemetry.
    pub fn reconfigure_substrate(
        &mut self,
        config: &mut CognitiveLoopConfig,
        substrate: SubstrateType,
    ) -> (f64, f64) {
        let old = self.feasibility;
        let old_type = config.substrate_type;
        let canonical = substrate.canonical();
        self.feasibility = Self::requirements_for(&canonical).consciousness_feasibility();
        config.substrate_type = canonical;
        // Clear any stale composition — single-substrate mode now.
        config.substrate_composition = None;
        self.pending_transition = Some(format!(
            "{:?} -> {:?} ({:.3} -> {:.3})",
            old_type, canonical, old, self.feasibility
        ));
        self.recompute_effective_feasibility(config);
        self.recompute_substrate_dynamics(config);

        // Phase 3: recalculate energy for new substrate
        let energy_per_op = canonical.energy_per_operation();
        self.energy_per_cycle = energy_per_op * SUBSTRATE_OPS_PER_CYCLE;
        let bio_energy = SubstrateType::BiologicalNeurons.energy_per_operation();
        self.energy_throughput_multiplier = (bio_energy / energy_per_op).clamp(0.1, 100.0) as f32;

        // Phase 3: set transition targets (smoothing via tick_transition)
        self.target_effective_feasibility = self.effective_feasibility;
        self.target_tau_factor = self.tau_factor;

        // Phase 3: record transition history
        let record = SubstrateTransitionRecord {
            cycle: 0, // caller should set via reconfigure_substrate_at_cycle
            from: old_type,
            to: canonical,
            old_feasibility: old,
            new_feasibility: self.feasibility,
        };
        self.transition_history.push_back(record);
        if self.transition_history.len() > SUBSTRATE_TRANSITION_HISTORY_CAP {
            self.transition_history.pop_front();
        }
        self.current_substrate = canonical;

        (old, self.feasibility)
    }

    /// Reconfigure substrate with cycle number for transition history.
    pub fn reconfigure_substrate_at_cycle(
        &mut self,
        config: &mut CognitiveLoopConfig,
        substrate: SubstrateType,
        cycle: u64,
    ) -> (f64, f64) {
        let result = self.reconfigure_substrate(config, substrate);
        // Patch the cycle number on the most recent transition record
        if let Some(record) = self.transition_history.back_mut() {
            record.cycle = cycle;
        }
        result
    }

    /// Switch to a substrate composition at runtime, recomputing feasibility.
    pub fn reconfigure_composition(
        &mut self,
        config: &mut CognitiveLoopConfig,
        composition: symthaea_core::hdc::substrate_composition::SubstrateComposition,
    ) {
        let old_feas = self.feasibility;
        self.feasibility = composition.feasibility();
        self.pending_transition = Some(format!(
            "-> {} ({:.3} -> {:.3})",
            composition.name, old_feas, self.feasibility
        ));
        config.substrate_composition = Some(composition);
        self.recompute_effective_feasibility(config);
        self.recompute_substrate_dynamics(config);
    }

    /// Recompute effective feasibility from raw feasibility × validation overlay.
    /// Called after any substrate/composition change and at startup.
    pub fn recompute_effective_feasibility(&mut self, config: &CognitiveLoopConfig) {
        let framework =
            symthaea_core::hdc::substrate_validation::SubstrateValidationFramework::new();
        self.honest_confidence = if let Some(ref comp) = config.substrate_composition {
            let mut blended = 0.0f64;
            for (sub, &weight) in &comp.weights {
                let conf = match Self::substrate_validation_key(sub) {
                    Some(k) => framework.honest_feasibility(k),
                    None => THEORETICAL_CONFIDENCE,
                };
                blended += conf * weight as f64;
            }
            blended
        } else {
            match Self::substrate_validation_key(&config.substrate_type) {
                Some(k) => framework.honest_feasibility(k),
                None => THEORETICAL_CONFIDENCE,
            }
        };

        if config.enable_validation_overlay {
            let floor = config.validation_skepticism_floor;
            let confidence = self.honest_confidence;
            self.effective_feasibility = self.feasibility * (floor + (1.0 - floor) * confidence);
        } else {
            self.effective_feasibility = self.feasibility;
        }

        // Physics-grounded override: when quantum-consciousness feature is enabled,
        // blend the ab initio multi-theory composite score into effective_feasibility.
        // This grounds consciousness in the Schrödinger equation rather than heuristics.
        // See symthaea-quantum-chemistry/cognitive_loop_bridge.rs for the mapping.
        #[cfg(feature = "quantum-consciousness")]
        {
            let physics_score = symthaea_quantum_chemistry::cognitive_loop_bridge
                ::substrate_feasibility_from_physics(
                    &symthaea_quantum_chemistry::Molecule::water(), // Reference substrate
                    310.0, // Body temperature in Kelvin
                );
            // Blend: 50% physics-derived, 50% validation-framework-derived
            self.effective_feasibility = 0.5 * self.effective_feasibility + 0.5 * physics_score;
        }
    }

    /// Recompute substrate speed/scale modulation factors.
    /// Called after any substrate change and at startup.
    /// When a composition is set, weight-blends speed/scale from all components.
    pub fn recompute_substrate_dynamics(&mut self, config: &CognitiveLoopConfig) {
        // Refresh before any early return: the override must hold even on the
        // speed-modulation-disabled path, which zeroes scale_pressure.
        self.dim_fraction_override =
            Self::validate_dim_override(config.effective_dim_fraction_override);

        if !config.enable_substrate_speed_modulation {
            // Even without speed modulation, compute substrate-aware tau_factor
            // so that substrate type affects dynamics proportionally.
            // Without this, all substrates behave identically (tau=1.0).
            let bio_speed = SubstrateType::BiologicalNeurons.operation_speed();
            let sub_speed = if let Some(ref comp) = config.substrate_composition {
                let mut log_speed = 0.0f64;
                for (sub, &weight) in &comp.weights {
                    log_speed += (weight as f64) * sub.operation_speed().ln();
                }
                log_speed.exp()
            } else {
                config.substrate_type.operation_speed()
            };
            let log_ratio = (bio_speed / sub_speed).log10();
            self.tau_factor = (1.0 + 0.5 * log_ratio / 9.0).clamp(0.5, 2.0) as f32;
            self.scale_pressure = 0.0;
            return;
        }

        let bio_speed = SubstrateType::BiologicalNeurons.operation_speed();
        let bio_scale = SubstrateType::BiologicalNeurons.max_scale();

        let (sub_speed, sub_scale) = if let Some(ref comp) = config.substrate_composition {
            // Geometric mean (log-space blend) — physically meaningful when speeds
            // span 12 orders of magnitude.  exp(Σ wᵢ·ln(sᵢ)) avoids the slowest
            // component dominating a linear blend.
            let mut log_speed = 0.0f64;
            let mut log_scale = 0.0f64;
            for (sub, &weight) in &comp.weights {
                log_speed += (weight as f64) * sub.operation_speed().ln();
                log_scale += (weight as f64) * sub.max_scale().ln();
            }
            (log_speed.exp(), log_scale.exp())
        } else {
            (
                config.substrate_type.operation_speed(),
                config.substrate_type.max_scale(),
            )
        };

        // log_ratio > 0 when substrate is faster than biological
        let log_ratio = (bio_speed / sub_speed).log10();
        // Compress 12 orders of magnitude to [0.5, 2.0] tau factor
        self.tau_factor = (1.0 + 0.5 * log_ratio / 9.0).clamp(0.5, 2.0) as f32;

        self.scale_pressure = (sub_scale / bio_scale).log10() as f32;
    }

    /// Record this cycle's model-estimated energy and evaluate the configured watts
    /// budget only after a measured monotonic observation window has closed.
    ///
    /// `energy_per_cycle` and `total_energy_spent` are joules; a budget named
    /// `energy_budget_joules_per_sec` is watts (joules/second). Comparing those
    /// quantities directly is dimensionally invalid, so this gate compares the
    /// window's estimated joules divided by monotonic wall elapsed seconds.
    ///
    /// This remains a substrate-model estimate, not a measurement of electrical
    /// power drawn by the actual processor or robot.
    pub fn tick_energy(&mut self, config: &CognitiveLoopConfig) {
        self.tick_energy_at(config, Instant::now());
    }

    /// Deterministic time-injected form used by tests and controlled replay.
    ///
    /// Each call occurs at the start of a cognitive cycle. The previously
    /// accumulated cycle energy is compared with elapsed start-to-start wall time;
    /// the current cycle estimate is added only after the prior interval is closed.
    pub(crate) fn tick_energy_at(&mut self, config: &CognitiveLoopConfig, now: Instant) {
        if !config.enable_energy_budget {
            self.last_energy_spent = 0.0;
            self.energy_rate_window_joules = 0.0;
            self.energy_rate_window_started_at = None;
            self.estimated_energy_rate_watts = 0.0;
            self.energy_rate_window_seconds = 0.0;
            return;
        }

        // A malformed configured limit is a configuration fault, not something
        // to defer until a complete measurement window. Reject it before recording
        // energy or allowing another cycle to proceed under an invalid contract.
        if let Some(budget_watts) = config.energy_budget_joules_per_sec {
            if !budget_watts.is_finite() || budget_watts < 0.0 {
                self.consciousness_viable = false;
                self.last_energy_spent = 0.0;
                self.energy_rate_window_joules = 0.0;
                self.energy_rate_window_started_at = None;
                self.estimated_energy_rate_watts = 0.0;
                self.energy_rate_window_seconds = 0.0;
                return;
            }
        }

        // Joules per cognitive cycle come from the substrate's energy model.
        // Cognitive tau changes internal integration dynamics; it is not an
        // independent physical-energy multiplier. Wall-time throughput is captured
        // by the measured observation interval below.
        let modeled_energy_joules = self.energy_per_cycle;
        let next_total_energy = self.total_energy_spent + modeled_energy_joules;
        if !modeled_energy_joules.is_finite()
            || modeled_energy_joules < 0.0
            || !next_total_energy.is_finite()
        {
            // Non-finite/negative samples and cumulative overflow cannot safely feed
            // a rate budget: NaN comparisons otherwise evaluate false.
            self.consciousness_viable = false;
            self.last_energy_spent = 0.0;
            self.energy_rate_window_joules = 0.0;
            self.energy_rate_window_started_at = None;
            self.estimated_energy_rate_watts = 0.0;
            self.energy_rate_window_seconds = 0.0;
            return;
        }
        self.last_energy_spent = modeled_energy_joules;

        // The first call starts a fresh window. Do not include idle time between
        // manager construction and the first cognitive cycle in the denominator.
        let Some(window_started_at) = self.energy_rate_window_started_at else {
            self.energy_rate_window_started_at = Some(now);
            self.energy_rate_window_joules = modeled_energy_joules;
            self.total_energy_spent = next_total_energy;
            return;
        };

        let Some(elapsed) = now.checked_duration_since(window_started_at) else {
            // Reversed monotonic evidence invalidates the observation. Latch non-viable
            // and discard the partial window; do not account the current sample because
            // its relationship to that invalid interval cannot be established.
            self.consciousness_viable = false;
            self.last_energy_spent = 0.0;
            self.energy_rate_window_started_at = None;
            self.energy_rate_window_joules = 0.0;
            self.estimated_energy_rate_watts = 0.0;
            self.energy_rate_window_seconds = 0.0;
            return;
        };
        let elapsed_seconds = elapsed.as_secs_f64();
        if elapsed_seconds >= ENERGY_RATE_WINDOW_MIN_SECONDS {
            let estimated_rate_watts = self.energy_rate_window_joules / elapsed_seconds;
            if !estimated_rate_watts.is_finite() || estimated_rate_watts < 0.0 {
                self.consciousness_viable = false;
                self.last_energy_spent = 0.0;
                self.energy_rate_window_joules = 0.0;
                self.energy_rate_window_started_at = None;
                self.estimated_energy_rate_watts = 0.0;
                self.energy_rate_window_seconds = 0.0;
                return;
            }
            self.estimated_energy_rate_watts = estimated_rate_watts;
            self.energy_rate_window_seconds = elapsed_seconds;

            if let Some(budget_watts) = config.energy_budget_joules_per_sec {
                if estimated_rate_watts > budget_watts {
                    self.consciousness_viable = false;
                }
            }

            self.energy_rate_window_started_at = Some(now);
            self.energy_rate_window_joules = 0.0;
        }

        // Attribute this cycle's estimate after checking the previously completed
        // start-to-start interval, so the numerator and denominator refer to the
        // same completed cycles.
        let next_window_energy = self.energy_rate_window_joules + modeled_energy_joules;
        if !next_window_energy.is_finite() || next_window_energy < 0.0 {
            self.consciousness_viable = false;
            self.last_energy_spent = 0.0;
            self.energy_rate_window_joules = 0.0;
            self.energy_rate_window_started_at = None;
            self.estimated_energy_rate_watts = 0.0;
            self.energy_rate_window_seconds = 0.0;
            return;
        }
        self.energy_rate_window_joules = next_window_energy;
        self.total_energy_spent = next_total_energy;
    }
    /// Smooth transition dynamics via EMA blending each cycle.
    ///
    /// When `substrate_transition_alpha` < 1.0, effective_feasibility and
    /// tau_factor blend gradually toward their targets after a substrate switch.
    /// This models the imperfect fidelity of substrate transfer (Bostrom 2003).
    pub fn tick_transition(&mut self, config: &CognitiveLoopConfig) {
        let alpha = config.substrate_transition_alpha;
        if alpha >= 1.0 {
            return; // Instant switching — no smoothing needed
        }
        // EMA blend: current += alpha × (target − current)
        self.effective_feasibility +=
            alpha as f64 * (self.target_effective_feasibility - self.effective_feasibility);
        self.tau_factor += alpha * (self.target_tau_factor - self.tau_factor);
    }

    /// Validate and clamp a configured dimension-fraction override.
    ///
    /// Out-of-range values are clamped to [0.0, 1.0] with a warning rather than
    /// silently accepted — a fraction outside that range would index the mask
    /// boundary out of the state slice.
    pub(crate) fn validate_dim_override(raw: Option<f32>) -> Option<f32> {
        let f = raw?;
        if !f.is_finite() {
            tracing::warn!(
                value = f,
                "effective_dim_fraction_override is not finite — ignoring override"
            );
            return None;
        }
        if !(0.0..=1.0).contains(&f) {
            let clamped = f.clamp(0.0, 1.0);
            tracing::warn!(
                requested = f,
                clamped,
                "effective_dim_fraction_override out of [0.0, 1.0] — clamped"
            );
            return Some(clamped);
        }
        Some(f)
    }

    /// Compute effective HDC/CfC dimensionality fraction.
    ///
    /// Precedence:
    /// 1. `effective_dim_fraction_override`, when set — used verbatim, and
    ///    explicitly *not* floored at `SUBSTRATE_MIN_DIM_FRACTION`, so `0.0`
    ///    (full lesion) is reachable as a negative control.
    /// 2. Otherwise substrate scale pressure: 1.0 at or above biological scale
    ///    (positive scale_pressure), else `< 1.0` clamped to
    ///    [SUBSTRATE_MIN_DIM_FRACTION, 1.0].
    ///
    /// Note the default configuration reaches 1.0 by *two* independent routes:
    /// `SiliconDigital` has positive scale pressure, and speed modulation being
    /// off zeroes scale pressure outright. Both mean "no masking" — which is why
    /// exercising this mechanism at all requires the override.
    ///
    /// Science: Berry & Srivastava (2018) — HDC capacity scales with D^(5/3).
    pub fn effective_dim_fraction(&self) -> f32 {
        if let Some(f) = self.dim_fraction_override {
            return f;
        }
        if self.scale_pressure >= 0.0 {
            return 1.0;
        }
        (1.0 + self.scale_pressure / SUBSTRATE_SCALE_DIM_DIVISOR)
            .clamp(SUBSTRATE_MIN_DIM_FRACTION, 1.0)
    }

    /// Which controller supplied the value `effective_dim_fraction()` returns.
    ///
    /// This is the *configured* provider. The controller that actually applied a
    /// mask in a given cycle is `RecurrentMaskEvent::source`, which can differ
    /// (spectral entropy may dominate substrate pressure at the mask site).
    /// Returns `Disabled` when recurrent masking is switched off entirely, since
    /// in that case no fraction is applied to anything.
    pub fn effective_dim_source(&self, config: &CognitiveLoopConfig) -> EffectiveDimSource {
        if !config.enable_recurrent_dim_masking && !config.enable_spectral_entropy_masking {
            return EffectiveDimSource::Disabled;
        }
        if self.dim_fraction_override.is_some() {
            return EffectiveDimSource::FixedOverride;
        }
        EffectiveDimSource::SubstratePressure
    }

    /// Set the effective-dimension-fraction override at runtime.
    ///
    /// Validated and clamped identically to the construction path. Exists so a
    /// harness can contract and then expand *within one service instance* —
    /// measuring how long previously-masked dimensions take to be repopulated
    /// by the dynamics. That recovery time bounds the usable bandwidth of any
    /// controller built on this lever: expansion is not the inverse of
    /// contraction, because restored dimensions re-enter at zero.
    pub fn set_dim_fraction_override(&mut self, raw: Option<f32>) {
        self.dim_fraction_override = Self::validate_dim_override(raw);
    }

    /// Record a recurrent-mask event from the site that performed (or skipped)
    /// the mask. Drained by `telemetry()`.
    ///
    /// When both mask sites run in one cycle the later one supersedes; its
    /// `source` field identifies which controller's fraction was actually used.
    pub fn record_mask_event(&mut self, event: RecurrentMaskEvent) {
        self.last_mask_event = Some(event);
    }

    /// Access the transition history log.
    pub fn transition_history(&self) -> &VecDeque<SubstrateTransitionRecord> {
        &self.transition_history
    }

    /// Returns true when substrate feasibility is too low for full consciousness.
    /// Below threshold, expensive modules (reasoning, dream, cross-modal) should
    /// be skipped to focus resources on core perception-prediction.
    pub fn should_degrade_consciousness(&self) -> bool {
        self.effective_feasibility < 0.3 || !self.consciousness_viable
    }

    // ── Per-region substrate methods ──────────────────────────────────────

    /// Get the feasibility score for a specific cortical region.
    /// Returns the global effective feasibility when per-region is not configured.
    /// Get the feasibility score for a specific cortical region.
    /// Returns the global effective feasibility when per-region is not configured.
    pub fn region_feasibility(&self, region: CorticalRegion) -> f32 {
        self.per_region_feasibility
            .get(&region)
            .copied()
            .unwrap_or(self.effective_feasibility as f32)
    }

    /// Reconfigure a single region's substrate at runtime.
    pub fn reconfigure_region(&mut self, region: CorticalRegion, substrate: SubstrateType) {
        let map = self.per_region_substrates.get_or_insert_with(HashMap::new);
        map.insert(region, substrate.canonical());
        let feas =
            Self::requirements_for(&substrate.canonical()).consciousness_feasibility() as f32;
        self.per_region_feasibility.insert(region, feas);
        // Recompute aggregate effective feasibility from per-region scores
        self.recompute_aggregate_from_regions();
    }

    /// Recompute per-region feasibility scores from current substrate assignments.
    fn recompute_per_region_feasibility(&mut self, config: &CognitiveLoopConfig) {
        self.per_region_feasibility.clear();
        if let Some(ref map) = self.per_region_substrates {
            for (&region, substrate) in map {
                let feas = Self::requirements_for(&substrate.canonical())
                    .consciousness_feasibility() as f32;
                self.per_region_feasibility.insert(region, feas);
            }
            self.recompute_aggregate_from_regions();
        }
        // If no per-region map, per_region_feasibility stays empty and
        // region_feasibility() falls back to global effective_feasibility.
        let _ = config; // used for future validation overlay per-region
    }

    /// Recompute aggregate effective feasibility from per-region scores.
    ///
    /// Effective = weighted average (equal weights) of per-region feasibilities,
    /// with a cross-substrate communication penalty (0.95× per distinct substrate pair).
    fn recompute_aggregate_from_regions(&mut self) {
        if self.per_region_feasibility.is_empty() {
            return;
        }
        // Equal-weight average
        let sum: f32 = self.per_region_feasibility.values().sum();
        let count = self.per_region_feasibility.len() as f32;
        let avg = sum / count;

        // Cross-substrate communication penalty: count distinct substrate types
        let distinct_substrates: std::collections::HashSet<_> = self
            .per_region_substrates
            .as_ref()
            .map(|m| m.values().map(std::mem::discriminant).collect())
            .unwrap_or_default();
        let num_pairs = if distinct_substrates.len() > 1 {
            distinct_substrates.len() - 1
        } else {
            0
        };
        // 0.95× penalty per distinct substrate pair
        let penalty = 0.95_f32.powi(num_pairs as i32);

        self.effective_feasibility = (avg * penalty) as f64;
    }

    /// Map a canonical SubstrateType to its pre-built SubstrateRequirements profile.
    /// Unknown/future variants fall back to silicon_digital().
    pub(crate) fn requirements_for(substrate: &SubstrateType) -> SubstrateRequirements {
        match substrate.canonical() {
            SubstrateType::BiologicalNeurons => SubstrateRequirements::biological_neurons(),
            SubstrateType::SiliconDigital => SubstrateRequirements::silicon_digital(),
            SubstrateType::QuantumComputer => SubstrateRequirements::quantum_computer(),
            SubstrateType::PhotonicProcessor => SubstrateRequirements::photonic_processor(),
            SubstrateType::NeuromorphicChip => SubstrateRequirements::neuromorphic_chip(),
            SubstrateType::BiochemicalComputer => SubstrateRequirements::biochemical_computer(),
            SubstrateType::HybridSystem => SubstrateRequirements::hybrid_system(),
            SubstrateType::ExoticSubstrate => SubstrateRequirements::exotic_substrate(),
            SubstrateType::SpacecraftComputer => SubstrateRequirements::spacecraft_computer(),
            _ => SubstrateRequirements::silicon_digital(),
        }
    }

    /// Build a `SubstrateTelemetry` snapshot and drain the pending transition.
    pub fn telemetry(
        &mut self,
        config: &super::config::CognitiveLoopConfig,
    ) -> super::types::SubstrateTelemetry {
        let per_region = self
            .per_region_feasibility
            .iter()
            .map(|(region, &feas)| (region.as_str().to_string(), feas))
            .collect();
        let encoding_noise = if config.enable_substrate_encoding_noise && self.scale_pressure < 0.0
        {
            (-self.scale_pressure).min(7.0) / 70.0
        } else {
            0.0
        };
        super::types::SubstrateTelemetry {
            substrate_feasibility: self.effective_feasibility,
            substrate_transition: self.pending_transition.take(),
            substrate_feasibility_raw: self.feasibility,
            substrate_honest_confidence: self.honest_confidence,
            substrate_effective_feasibility: self.effective_feasibility,
            substrate_tau_factor: self.tau_factor,
            substrate_scale_pressure: self.scale_pressure,
            per_region_feasibility: per_region,
            substrate_encoding_noise: encoding_noise,
            total_energy_spent: self.total_energy_spent,
            energy_this_cycle: self.last_energy_spent,
            estimated_energy_rate_watts: self.estimated_energy_rate_watts,
            energy_rate_window_seconds: self.energy_rate_window_seconds,
            energy_throughput_multiplier: self.energy_throughput_multiplier,
            effective_dim_fraction: self.effective_dim_fraction(),
            effective_dim_source: self.effective_dim_source(config),
            recurrent_mask: self.last_mask_event.take(),
            transition_count: self.transition_history.len(),
        }
    }

    /// Map SubstrateType to validation framework key string.
    ///
    /// Substrates not in the framework (photonic, neuromorphic, biochemical, exotic)
    /// return None — callers should fall back to Theoretical confidence (0.10).
    pub(crate) fn substrate_validation_key(substrate: &SubstrateType) -> Option<&'static str> {
        match substrate.canonical() {
            SubstrateType::BiologicalNeurons => Some("biological"),
            SubstrateType::SiliconDigital => Some("silicon"),
            SubstrateType::QuantumComputer => Some("quantum"),
            SubstrateType::HybridSystem => Some("hybrid"),
            SubstrateType::SpacecraftComputer => Some("spacecraft"),
            _ => None,
        }
    }

    /// Binding capability [0,1] of the substrate running sensory processing.
    /// Uses per-region substrate (Sensory) when configured, else global.
    /// Modulates Phi binding computation — substrates with poor binding
    /// (e.g. biochemical) produce lower integrated information.
    pub(crate) fn binding_capability(&self, config: &CognitiveLoopConfig) -> f64 {
        let substrate = self.substrate_for_region(config, CorticalRegion::Sensory);
        Self::requirements_for(&substrate).binding_capability
    }

    /// Workspace capability [0,1] of the substrate running prefrontal processing.
    /// Uses per-region substrate (Prefrontal) when configured, else global.
    /// Modulates GWT broadcast — substrates with poor workspace support
    /// (e.g. exotic) require higher activation for conscious broadcast.
    pub(crate) fn workspace_capability(&self, config: &CognitiveLoopConfig) -> f64 {
        let substrate = self.substrate_for_region(config, CorticalRegion::Prefrontal);
        Self::requirements_for(&substrate).workspace_capability
    }

    /// Attention capability [0,1] of the substrate running prefrontal processing.
    /// Uses per-region substrate (Prefrontal) when configured, else global.
    /// Modulates phi-attention gate gain — substrates with poor attention
    /// (e.g. biochemical) produce weaker selective amplification.
    pub(crate) fn attention_capability(&self, config: &CognitiveLoopConfig) -> f64 {
        let substrate = self.substrate_for_region(config, CorticalRegion::Prefrontal);
        Self::requirements_for(&substrate).attention_capability
    }

    /// HOT (Higher-Order Thought) capability [0,1] of the substrate running
    /// prefrontal processing. Uses per-region substrate (Prefrontal) when
    /// configured, else global. Modulates the Recursion component in
    /// ConsciousnessEquationV2 — substrates with poor meta-representation
    /// (e.g. exotic) score lower on higher-order thought.
    pub(crate) fn hot_capability(&self, config: &CognitiveLoopConfig) -> f64 {
        let substrate = self.substrate_for_region(config, CorticalRegion::Prefrontal);
        Self::requirements_for(&substrate).hot_capability
    }

    /// Get the substrate type for a specific cortical region.
    /// Falls back to global substrate_type when per-region is not configured.
    fn substrate_for_region(
        &self,
        config: &CognitiveLoopConfig,
        region: CorticalRegion,
    ) -> SubstrateType {
        self.per_region_substrates
            .as_ref()
            .and_then(|m| m.get(&region).copied())
            .unwrap_or(config.substrate_type)
    }
}

// ── Delegation methods on CognitiveLoopService ──────────────────────────────
// These forward to SubstrateManager so that constructor.rs can call
// Self::requirements_for(...) and Self::substrate_validation_key(...).

#[allow(dead_code)] // Public API for external consumers; not called internally yet
impl super::CognitiveLoopService {
    /// Default honest confidence for substrates not in the validation framework.
    pub(crate) const THEORETICAL_CONFIDENCE: f64 = THEORETICAL_CONFIDENCE;

    /// Delegate to SubstrateManager::requirements_for.
    pub(crate) fn requirements_for(substrate: &SubstrateType) -> SubstrateRequirements {
        SubstrateManager::requirements_for(substrate)
    }

    /// Delegate to SubstrateManager::substrate_validation_key.
    pub(crate) fn substrate_validation_key(substrate: &SubstrateType) -> Option<&'static str> {
        SubstrateManager::substrate_validation_key(substrate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> CognitiveLoopConfig {
        CognitiveLoopConfig::default()
    }

    #[test]
    fn test_silicon_feasibility_approximately_071() {
        let config = default_config();
        assert_eq!(config.substrate_type, SubstrateType::SiliconDigital);
        let mgr = SubstrateManager::new(&config);
        // SiliconDigital feasibility should be ~0.71 (not 1.0)
        assert!(
            (mgr.feasibility - 0.71).abs() < 0.05,
            "SiliconDigital feasibility should be ~0.71, got {:.4}",
            mgr.feasibility
        );
        // Without validation overlay, effective == raw
        assert!(
            (mgr.effective_feasibility - mgr.feasibility).abs() < f64::EPSILON,
            "Without overlay, effective should equal raw"
        );
    }

    #[test]
    fn test_biological_feasibility_near_one() {
        let mut config = default_config();
        config.substrate_type = SubstrateType::BiologicalNeurons;
        let mgr = SubstrateManager::new(&config);
        assert!(
            mgr.feasibility > 0.9,
            "BiologicalNeurons feasibility should be near 1.0, got {:.4}",
            mgr.feasibility
        );
    }

    #[test]
    fn test_validation_overlay_reduces_feasibility() {
        let mut config = default_config();
        config.enable_validation_overlay = true;
        config.validation_skepticism_floor = 0.5;
        let mgr = SubstrateManager::new(&config);
        // SiliconDigital honest_confidence is ~0.10 (theoretical)
        // effective = raw × (0.5 + 0.5 × 0.10) = raw × 0.55
        assert!(
            mgr.effective_feasibility < mgr.feasibility,
            "Validation overlay should reduce feasibility: effective={:.4} vs raw={:.4}",
            mgr.effective_feasibility,
            mgr.feasibility
        );
    }

    #[test]
    fn test_validation_overlay_preserves_biological_confidence() {
        let mut config = default_config();
        config.substrate_type = SubstrateType::BiologicalNeurons;
        config.enable_validation_overlay = true;
        config.validation_skepticism_floor = 0.5;
        let mgr = SubstrateManager::new(&config);
        // Biological honest_confidence is 0.95
        // effective = raw × (0.5 + 0.5 × 0.95) = raw × 0.975
        assert!(
            mgr.effective_feasibility > 0.85,
            "Biological with overlay should remain high: {:.4}",
            mgr.effective_feasibility
        );
    }

    #[test]
    fn test_reconfigure_substrate_changes_feasibility() {
        let mut config = default_config();
        let mut mgr = SubstrateManager::new(&config);
        let initial = mgr.feasibility;

        let (old, new) = mgr.reconfigure_substrate(&mut config, SubstrateType::BiologicalNeurons);
        assert!((old - initial).abs() < f64::EPSILON);
        assert!(
            new > old,
            "Biological should have higher feasibility than Silicon"
        );
        assert!(
            mgr.pending_transition.is_some(),
            "reconfigure should set pending_transition (drained by telemetry())"
        );
    }

    #[test]
    fn test_reconfigure_produces_transition_string() {
        let mut config = default_config();
        let mut mgr = SubstrateManager::new(&config);
        mgr.reconfigure_substrate(&mut config, SubstrateType::QuantumComputer);

        assert!(
            mgr.pending_transition.is_some(),
            "reconfigure should set pending_transition"
        );
        let transition = mgr.pending_transition.as_ref().unwrap();
        assert!(
            transition.contains("SiliconDigital") && transition.contains("QuantumComputer"),
            "Transition should describe old->new: {transition}"
        );
    }

    #[test]
    fn test_telemetry_drains_transition() {
        let mut config = default_config();
        let mut mgr = SubstrateManager::new(&config);
        mgr.reconfigure_substrate(&mut config, SubstrateType::BiologicalNeurons);
        assert!(mgr.pending_transition.is_some());

        let telem = mgr.telemetry(&config);
        assert!(telem.substrate_transition.is_some());
        assert!(
            mgr.pending_transition.is_none(),
            "telemetry() should drain pending_transition"
        );

        let telem2 = mgr.telemetry(&config);
        assert!(
            telem2.substrate_transition.is_none(),
            "Second telemetry() should have no transition"
        );
    }

    #[test]
    fn test_speed_modulation_disabled_by_default() {
        let config = default_config();
        assert!(
            !config.enable_substrate_speed_modulation,
            "speed modulation should be off in the default config"
        );
        let mgr = SubstrateManager::new(&config);
        // recompute_substrate_dynamics's own doc comment: "Even without speed
        // modulation, compute substrate-aware tau_factor... Without this, all
        // substrates behave identically (tau=1.0)." Default substrate_type is
        // SiliconDigital (1ns/op) vs BiologicalNeurons (1ms/op) reference, so
        // tau_factor = 1.0 + 0.5*log10(1e6)/9 = 4/3, not 1.0 -- 1.0 would only
        // hold if the configured substrate_type were BiologicalNeurons itself.
        assert!(
            (mgr.tau_factor - 4.0 / 3.0).abs() < 1e-4,
            "tau_factor should reflect SiliconDigital's substrate-aware speed (4/3), got {}",
            mgr.tau_factor
        );
        assert!(
            mgr.scale_pressure.abs() < f32::EPSILON,
            "scale_pressure should be 0.0 when speed modulation disabled (unlike tau_factor, \
             this one really is a no-op per recompute_substrate_dynamics's early-return branch)"
        );
    }

    #[test]
    fn test_speed_modulation_enabled() {
        let mut config = default_config();
        config.enable_substrate_speed_modulation = true;
        config.substrate_type = SubstrateType::SiliconDigital;
        let mgr = SubstrateManager::new(&config);
        // Silicon is faster than biological → tau_factor > 1.0
        assert!(
            mgr.tau_factor > 1.0,
            "Faster substrate should have tau_factor > 1.0, got {:.4}",
            mgr.tau_factor
        );
        assert!(mgr.tau_factor <= 2.0, "tau_factor should be clamped to 2.0");
    }

    #[test]
    fn test_all_substrates_produce_finite_feasibility() {
        let substrates = [
            SubstrateType::BiologicalNeurons,
            SubstrateType::SiliconDigital,
            SubstrateType::QuantumComputer,
            SubstrateType::PhotonicProcessor,
            SubstrateType::NeuromorphicChip,
            SubstrateType::BiochemicalComputer,
            SubstrateType::HybridSystem,
            SubstrateType::ExoticSubstrate,
        ];
        for sub in &substrates {
            let mut config = default_config();
            config.substrate_type = *sub;
            let mgr = SubstrateManager::new(&config);
            assert!(
                mgr.feasibility.is_finite() && mgr.feasibility >= 0.0 && mgr.feasibility <= 1.0,
                "{:?} feasibility out of bounds: {:.4}",
                sub,
                mgr.feasibility
            );
            assert!(
                mgr.effective_feasibility.is_finite(),
                "{:?} effective_feasibility not finite",
                sub
            );
        }
    }

    #[test]
    fn test_all_substrates_with_overlay_finite() {
        let substrates = [
            SubstrateType::BiologicalNeurons,
            SubstrateType::SiliconDigital,
            SubstrateType::QuantumComputer,
            SubstrateType::PhotonicProcessor,
            SubstrateType::NeuromorphicChip,
            SubstrateType::BiochemicalComputer,
            SubstrateType::HybridSystem,
            SubstrateType::ExoticSubstrate,
        ];
        for sub in &substrates {
            let mut config = default_config();
            config.substrate_type = *sub;
            config.enable_validation_overlay = true;
            let mgr = SubstrateManager::new(&config);
            assert!(
                mgr.effective_feasibility.is_finite()
                    && mgr.effective_feasibility >= 0.0
                    && mgr.effective_feasibility <= 1.0,
                "{:?} effective_feasibility with overlay out of bounds: {:.4}",
                sub,
                mgr.effective_feasibility
            );
        }
    }

    #[test]
    fn test_aliases_map_to_canonical() {
        let mut config = default_config();
        let mut mgr = SubstrateManager::new(&config);

        mgr.reconfigure_substrate(&mut config, SubstrateType::Biological);
        assert_eq!(config.substrate_type, SubstrateType::BiologicalNeurons);

        mgr.reconfigure_substrate(&mut config, SubstrateType::Silicon);
        assert_eq!(config.substrate_type, SubstrateType::SiliconDigital);

        mgr.reconfigure_substrate(&mut config, SubstrateType::Quantum);
        assert_eq!(config.substrate_type, SubstrateType::QuantumComputer);
    }

    #[test]
    fn test_validation_key_returns_none_for_unknown_substrates() {
        // Photonic, neuromorphic, biochemical, exotic → None (fall back to THEORETICAL_CONFIDENCE)
        assert!(
            SubstrateManager::substrate_validation_key(&SubstrateType::PhotonicProcessor).is_none()
        );
        assert!(
            SubstrateManager::substrate_validation_key(&SubstrateType::NeuromorphicChip).is_none()
        );
        assert!(
            SubstrateManager::substrate_validation_key(&SubstrateType::BiochemicalComputer)
                .is_none()
        );
        assert!(
            SubstrateManager::substrate_validation_key(&SubstrateType::ExoticSubstrate).is_none()
        );
        // Known substrates → Some
        assert!(
            SubstrateManager::substrate_validation_key(&SubstrateType::BiologicalNeurons).is_some()
        );
        assert!(
            SubstrateManager::substrate_validation_key(&SubstrateType::SiliconDigital).is_some()
        );
    }

    #[test]
    fn test_reconfigure_substrate_clears_composition() {
        let mut config = default_config();
        let mut mgr = SubstrateManager::new(&config);

        // Set a composition
        let comp = symthaea_core::hdc::substrate_composition::SubstrateComposition::new(
            "test_comp".to_string(),
            vec![
                (SubstrateType::SiliconDigital, 0.6),
                (SubstrateType::BiologicalNeurons, 0.4),
            ],
        )
        .expect("valid composition");
        mgr.reconfigure_composition(&mut config, comp);
        assert!(config.substrate_composition.is_some());

        // Switch to single substrate — should clear composition
        mgr.reconfigure_substrate(&mut config, SubstrateType::QuantumComputer);
        assert!(
            config.substrate_composition.is_none(),
            "reconfigure_substrate should clear stale composition"
        );
    }

    #[test]
    fn test_composition_feasibility_blended() {
        let mut config = default_config();
        let mut mgr = SubstrateManager::new(&config);

        let comp = symthaea_core::hdc::substrate_composition::SubstrateComposition::new(
            "bio_silicon".to_string(),
            vec![
                (SubstrateType::BiologicalNeurons, 0.5),
                (SubstrateType::SiliconDigital, 0.5),
            ],
        )
        .expect("valid composition");
        mgr.reconfigure_composition(&mut config, comp);

        // Composition feasibility should be between pure bio and pure silicon
        let bio_feas = SubstrateManager::requirements_for(&SubstrateType::BiologicalNeurons)
            .consciousness_feasibility();
        let silicon_feas = SubstrateManager::requirements_for(&SubstrateType::SiliconDigital)
            .consciousness_feasibility();
        assert!(
            mgr.feasibility > silicon_feas.min(bio_feas) * 0.9,
            "Composition feasibility {:.4} too low",
            mgr.feasibility
        );
        assert!(
            mgr.feasibility <= bio_feas + 0.01,
            "Composition feasibility {:.4} exceeds biological {:.4}",
            mgr.feasibility,
            bio_feas
        );
    }

    #[test]
    fn test_slow_substrate_tau_below_one() {
        let mut config = default_config();
        config.enable_substrate_speed_modulation = true;
        config.substrate_type = SubstrateType::BiochemicalComputer;
        let mgr = SubstrateManager::new(&config);
        // Biochemical is ~1000× slower than biological → tau < 1.0
        assert!(
            mgr.tau_factor < 1.0,
            "Slow substrate should have tau < 1.0, got {:.4}",
            mgr.tau_factor
        );
        assert!(mgr.tau_factor >= 0.5, "tau_factor should be clamped to 0.5");
    }

    #[test]
    fn test_substrate_tau_sign_and_reference_values_are_explicit() {
        let cases = [
            (SubstrateType::PhotonicProcessor, 1.5_f32, "photonic faster than silicon"),
            (SubstrateType::SiliconDigital, 1.333_333_4_f32, "silicon faster than biological"),
            (SubstrateType::BiologicalNeurons, 1.0_f32, "biological reference"),
            (SubstrateType::BiochemicalComputer, 0.833_333_3_f32, "biochemical slower than biological"),
        ];
        let mut observed = Vec::new();

        for (substrate, expected, label) in cases {
            let mut config = default_config();
            config.enable_substrate_speed_modulation = true;
            config.substrate_type = substrate;
            let manager = SubstrateManager::new(&config);

            assert!(
                (manager.tau_factor - expected).abs() < 0.01,
                "{label}: expected tau near {expected}, observed {}",
                manager.tau_factor
            );
            observed.push((substrate, manager.tau_factor));
        }

        // operation_speed() is seconds per operation: smaller means faster.
        // Therefore faster substrates yield larger factors under the current formula.
        assert!(observed[0].1 > observed[1].1);
        assert!(observed[1].1 > observed[2].1);
        assert!(observed[2].1 > observed[3].1);
    }

    #[test]
    fn test_biological_tau_equals_one() {
        let mut config = default_config();
        config.enable_substrate_speed_modulation = true;
        config.substrate_type = SubstrateType::BiologicalNeurons;
        let mgr = SubstrateManager::new(&config);
        assert!(
            (mgr.tau_factor - 1.0).abs() < 0.01,
            "Biological reference should have tau ≈ 1.0, got {:.4}",
            mgr.tau_factor
        );
    }

    #[test]
    fn test_composition_speed_geometric_mean() {
        let mut config = default_config();
        config.enable_substrate_speed_modulation = true;
        let mut mgr = SubstrateManager::new(&config);

        // 50/50 bio + silicon composition
        let comp = symthaea_core::hdc::substrate_composition::SubstrateComposition::new(
            "hybrid".to_string(),
            vec![
                (SubstrateType::BiologicalNeurons, 0.5),
                (SubstrateType::SiliconDigital, 0.5),
            ],
        )
        .expect("valid composition");
        mgr.reconfigure_composition(&mut config, comp);

        // Geometric mean of bio and silicon speeds → intermediate tau
        // Should be between pure-bio tau (≈1.0) and pure-silicon tau (>1.0)
        assert!(
            mgr.tau_factor > 1.0,
            "50/50 bio+silicon should have tau > 1.0 (silicon contribution), got {:.4}",
            mgr.tau_factor
        );
        // But less than pure silicon
        let mut pure_silicon = default_config();
        pure_silicon.enable_substrate_speed_modulation = true;
        pure_silicon.substrate_type = SubstrateType::SiliconDigital;
        let pure_mgr = SubstrateManager::new(&pure_silicon);
        assert!(
            mgr.tau_factor < pure_mgr.tau_factor,
            "50/50 tau {:.4} should be less than pure silicon tau {:.4}",
            mgr.tau_factor,
            pure_mgr.tau_factor
        );
    }

    #[test]
    fn test_composition_overlay_blends_confidence() {
        let mut config = default_config();
        config.enable_validation_overlay = true;
        config.validation_skepticism_floor = 0.5;
        let mut mgr = SubstrateManager::new(&config);

        // 50/50 bio (conf=0.95) + silicon (conf=0.10)
        let comp = symthaea_core::hdc::substrate_composition::SubstrateComposition::new(
            "mixed".to_string(),
            vec![
                (SubstrateType::BiologicalNeurons, 0.5),
                (SubstrateType::SiliconDigital, 0.5),
            ],
        )
        .expect("valid composition");
        mgr.reconfigure_composition(&mut config, comp);

        // Blended confidence ≈ 0.5 × 0.95 + 0.5 × 0.10 = 0.525
        assert!(
            (mgr.honest_confidence - 0.525).abs() < 0.05,
            "Blended confidence should be ≈0.525, got {:.4}",
            mgr.honest_confidence
        );
        // Effective should be less than raw (overlay is on)
        assert!(
            mgr.effective_feasibility < mgr.feasibility,
            "Overlay should reduce: eff={:.4} raw={:.4}",
            mgr.effective_feasibility,
            mgr.feasibility
        );
    }

    #[test]
    fn test_scale_pressure_signs() {
        let mut config = default_config();
        config.enable_substrate_speed_modulation = true;

        // Silicon is more scalable → positive scale_pressure
        config.substrate_type = SubstrateType::SiliconDigital;
        let mgr = SubstrateManager::new(&config);
        assert!(
            mgr.scale_pressure > 0.0,
            "Silicon should have positive scale_pressure, got {:.4}",
            mgr.scale_pressure
        );

        // Biological is reference → scale_pressure ≈ 0
        config.substrate_type = SubstrateType::BiologicalNeurons;
        let mgr = SubstrateManager::new(&config);
        assert!(
            mgr.scale_pressure.abs() < 0.01,
            "Biological reference should have scale_pressure ≈ 0, got {:.4}",
            mgr.scale_pressure
        );
    }

    // ── Phase 3: Energy, Viability, Recovery ────────────────────────────

    #[test]
    fn test_energy_per_cycle_positive_for_all_substrates() {
        for sub in &[
            SubstrateType::BiologicalNeurons,
            SubstrateType::SiliconDigital,
            SubstrateType::QuantumComputer,
            SubstrateType::PhotonicProcessor,
            SubstrateType::NeuromorphicChip,
            SubstrateType::BiochemicalComputer,
            SubstrateType::HybridSystem,
            SubstrateType::ExoticSubstrate,
        ] {
            let mut cfg = default_config();
            cfg.substrate_type = *sub;
            let mgr = SubstrateManager::new(&cfg);
            assert!(
                mgr.energy_per_cycle > 0.0 && mgr.energy_per_cycle.is_finite(),
                "{sub:?} energy_per_cycle={:.2e}",
                mgr.energy_per_cycle
            );
        }
    }

    #[test]
    fn test_energy_accumulates_with_tick_energy() {
        let mut cfg = default_config();
        cfg.enable_energy_budget = true;
        cfg.energy_budget_joules_per_sec = Some(1e10); // large budget so viability stays true
        let mut mgr = SubstrateManager::new(&cfg);
        assert!((mgr.total_energy_spent - 0.0).abs() < f64::EPSILON);
        for _ in 0..10 {
            mgr.tick_energy(&cfg);
        }
        assert!(
            mgr.total_energy_spent > 0.0,
            "10 ticks should accumulate energy"
        );
        assert!(mgr.consciousness_viable, "large budget → still viable");
    }

    #[test]
    fn test_tick_energy_noop_without_budget() {
        // enable_energy_budget actually defaults to true (see
        // config/mod.rs:937, asserted explicitly at config/mod.rs:1905) --
        // this test's own premise needs it off, so set it explicitly rather
        // than relying on the default, matching the pattern already used by
        // test_energy_accumulates_with_tick_energy above for the true case.
        let mut cfg = default_config();
        cfg.enable_energy_budget = false;
        let mut mgr = SubstrateManager::new(&cfg);
        mgr.tick_energy(&cfg);
        assert!(
            (mgr.total_energy_spent - 0.0).abs() < f64::EPSILON,
            "tick_energy should be a no-op when budget disabled, got {}",
            mgr.total_energy_spent
        );
    }

    #[test]
    fn test_budget_exhaustion_kills_viability_after_wall_time_window() {
        let mut cfg = default_config();
        cfg.enable_energy_budget = true;
        let mgr_ref = SubstrateManager::new(&cfg);
        cfg.energy_budget_joules_per_sec = Some(mgr_ref.energy_per_cycle * 0.5);
        let mut mgr = SubstrateManager::new(&cfg);
        let window_start = std::time::Instant::now();
        mgr.energy_rate_window_started_at = Some(window_start);
        assert!(mgr.consciousness_viable);
        mgr.tick_energy_at(&cfg, window_start); // seed the first cycle's modeled energy

        mgr.tick_energy_at(&cfg, window_start + std::time::Duration::from_secs(1));
        assert!(
            !mgr.consciousness_viable,
            "Model-estimated average watts above the budget should trip the gate"
        );
        assert!(mgr.estimated_energy_rate_watts > cfg.energy_budget_joules_per_sec.unwrap());
        assert!((mgr.energy_rate_window_seconds - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_consciousness_viable_biological() {
        let mut cfg = default_config();
        cfg.substrate_type = SubstrateType::BiologicalNeurons;
        cfg.enable_validation_overlay = true;
        let mgr = SubstrateManager::new(&cfg);
        assert!(mgr.consciousness_viable);
    }

    #[test]
    fn test_throughput_multiplier_varies_by_substrate() {
        let cfg_silicon = default_config(); // SiliconDigital
        let mgr_silicon = SubstrateManager::new(&cfg_silicon);

        let mut cfg_bio = default_config();
        cfg_bio.substrate_type = SubstrateType::BiologicalNeurons;
        let mgr_bio = SubstrateManager::new(&cfg_bio);

        // Bio/bio ratio = 1.0, silicon should differ (lower energy → higher multiplier)
        assert!(
            (mgr_bio.energy_throughput_multiplier - 1.0).abs() < 0.01,
            "Bio reference should have multiplier ≈ 1.0, got {:.4}",
            mgr_bio.energy_throughput_multiplier
        );
        assert!(
            mgr_silicon.energy_throughput_multiplier != mgr_bio.energy_throughput_multiplier,
            "Different substrates should have different throughput multipliers"
        );
    }

    #[test]
    fn test_rate_budget_does_not_compare_joules_to_watts_before_window_closes() {
        let mut config = default_config();
        config.enable_energy_budget = true;
        let mut mgr = SubstrateManager::new(&config);
        let window_start = std::time::Instant::now();
        mgr.energy_rate_window_started_at = Some(window_start);
        config.energy_budget_joules_per_sec = Some(1e-20);

        mgr.tick_energy_at(&config, window_start + std::time::Duration::from_millis(100));
        assert!(
            mgr.total_energy_spent > config.energy_budget_joules_per_sec.unwrap(),
            "test requires cumulative joules to exceed the numeric watts budget"
        );
        assert!(
            mgr.consciousness_viable,
            "a numeric joule-vs-watt comparison must not trip the rate budget"
        );
        assert_eq!(mgr.estimated_energy_rate_watts, 0.0);

        mgr.tick_energy_at(&config, window_start + std::time::Duration::from_secs(1));
        assert!(
            !mgr.consciousness_viable,
            "a completed measured window above budget must trip the rate gate"
        );
        assert!(mgr.estimated_energy_rate_watts > config.energy_budget_joules_per_sec.unwrap());
    }

    #[test]
    fn test_invalid_energy_rate_budgets_fail_closed_before_first_window() {
        for invalid_budget in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut config = default_config();
            config.enable_energy_budget = true;
            config.energy_budget_joules_per_sec = Some(invalid_budget);

            let mut mgr = SubstrateManager::new(&config);
            mgr.tick_energy_at(&config, std::time::Instant::now());

            assert!(
                !mgr.consciousness_viable,
                "invalid watt budget {:?} must fail closed immediately",
                invalid_budget
            );
            assert_eq!(
                mgr.total_energy_spent, 0.0,
                "invalid configuration must not record an apparently valid energy sample"
            );
            assert_eq!(mgr.energy_rate_window_joules, 0.0);
            assert_eq!(mgr.energy_rate_window_started_at, None);
            assert_eq!(mgr.estimated_energy_rate_watts, 0.0);
            assert_eq!(mgr.energy_rate_window_seconds, 0.0);

            // The same invalid input must invalidate an already-open window too.
            let mut open_window_mgr = SubstrateManager::new(&config);
            let start = std::time::Instant::now();
            open_window_mgr.energy_rate_window_started_at = Some(start);
            open_window_mgr.energy_rate_window_joules = 123.0;
            open_window_mgr.tick_energy_at(
                &config,
                start + std::time::Duration::from_millis(500),
            );
            assert!(
                !open_window_mgr.consciousness_viable,
                "invalid watt budget {:?} must invalidate an open window immediately",
                invalid_budget
            );
            assert_eq!(open_window_mgr.energy_rate_window_joules, 0.0);
            assert_eq!(open_window_mgr.energy_rate_window_started_at, None);
            assert_eq!(open_window_mgr.total_energy_spent, 0.0);
        }
    }

    #[test]
    fn test_invalid_modeled_energy_fails_closed_before_accounting() {
        for energy_per_cycle in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            let mut config = default_config();
            config.enable_energy_budget = true;
            config.energy_budget_joules_per_sec = Some(1e9);

            let mut mgr = SubstrateManager::new(&config);
            mgr.energy_per_cycle = energy_per_cycle;
            let now = std::time::Instant::now();
            mgr.energy_rate_window_started_at = Some(now);
            mgr.energy_rate_window_joules = 0.25;

            mgr.tick_energy_at(&config, now + std::time::Duration::from_millis(500));

            assert!(
                !mgr.consciousness_viable,
                "invalid modeled energy {energy_per_cycle:?} J/cycle must fail closed"
            );
            assert_eq!(mgr.total_energy_spent, 0.0);
            assert_eq!(mgr.last_energy_spent, 0.0);
            assert_eq!(mgr.energy_rate_window_joules, 0.0);
            assert_eq!(mgr.energy_rate_window_started_at, None);
            assert_eq!(mgr.estimated_energy_rate_watts, 0.0);
        }
    }

    #[test]
    fn test_energy_accumulation_overflow_fails_closed() {
        let mut config = default_config();
        config.enable_energy_budget = true;
        config.energy_budget_joules_per_sec = Some(f64::MAX);

        // A finite sample that overflows the cumulative sum is still invalid evidence.
        let mut total_overflow = SubstrateManager::new(&config);
        total_overflow.total_energy_spent = f64::MAX;
        total_overflow.energy_per_cycle = f64::MAX;
        total_overflow.tau_factor = 1.0;
        total_overflow.tick_energy_at(&config, std::time::Instant::now());
        assert!(!total_overflow.consciousness_viable);
        assert_eq!(total_overflow.total_energy_spent, f64::MAX);
        assert_eq!(total_overflow.energy_rate_window_started_at, None);

        // Window accumulation can overflow independently before its rate closes.
        let mut window_overflow = SubstrateManager::new(&config);
        window_overflow.energy_per_cycle = 1.0e308;
        window_overflow.tau_factor = 1.0;
        window_overflow.energy_rate_window_joules = 1.0e308;
        let start = std::time::Instant::now();
        window_overflow.energy_rate_window_started_at = Some(start);
        window_overflow.tick_energy_at(
            &config,
            start + std::time::Duration::from_millis(500),
        );
        assert!(!window_overflow.consciousness_viable);
        assert_eq!(window_overflow.energy_rate_window_joules, 0.0);
        assert_eq!(window_overflow.energy_rate_window_started_at, None);
        assert_eq!(window_overflow.total_energy_spent, 0.0, "overflowing window sample must not be accounted");
    }

    #[test]
    fn test_energy_rate_telemetry_reports_the_closed_monotonic_window() {
        let mut config = default_config();
        config.enable_energy_budget = true;
        config.energy_budget_joules_per_sec = Some(1e10);
        let mut mgr = SubstrateManager::new(&config);
        let start = std::time::Instant::now();
        mgr.energy_rate_window_started_at = Some(start);

        mgr.tick_energy_at(&config, start);
        let expected_window_energy = mgr.energy_per_cycle;
        mgr.tick_energy_at(&config, start + std::time::Duration::from_secs(1));

        assert!((mgr.estimated_energy_rate_watts - expected_window_energy).abs() < 1e-18);
        assert!((mgr.energy_rate_window_seconds - 1.0).abs() < f64::EPSILON);
        let telemetry = mgr.telemetry(&config);
        assert_eq!(telemetry.estimated_energy_rate_watts, mgr.estimated_energy_rate_watts);
        assert_eq!(telemetry.energy_rate_window_seconds, 1.0);
    }

    #[test]
    fn test_energy_rate_uses_actual_elapsed_time_for_irregular_cycle_cadence() {
        let mut config = default_config();
        config.enable_energy_budget = true;
        config.energy_budget_joules_per_sec = Some(f64::MAX);

        let mut mgr = SubstrateManager::new(&config);
        mgr.tau_factor = 2.0;
        let energy_per_cycle = mgr.energy_per_cycle;
        let start = std::time::Instant::now();

        mgr.tick_energy_at(&config, start);
        mgr.tick_energy_at(&config, start + std::time::Duration::from_millis(250));
        mgr.tick_energy_at(&config, start + std::time::Duration::from_millis(1_500));

        let expected_watts = (2.0 * energy_per_cycle) / 1.5;
        assert!(
            (mgr.estimated_energy_rate_watts - expected_watts).abs() <= expected_watts.abs() * 1e-12,
            "expected modeled rate {expected_watts:e} W over 1.5 s, got {:e} W",
            mgr.estimated_energy_rate_watts
        );
        assert_eq!(mgr.energy_rate_window_seconds, 1.5);
        assert!(
            (mgr.total_energy_spent - 3.0 * energy_per_cycle).abs()
                <= (3.0 * energy_per_cycle).abs() * 1e-12,
            "cumulative modeled joules should account for all three ticks"
        );
        assert_eq!(
            mgr.energy_rate_window_joules, energy_per_cycle,
            "the cycle at the closing timestamp belongs to the next window"
        );
        assert!(mgr.consciousness_viable);
    }

    #[test]
    fn test_reversed_monotonic_energy_window_fails_closed() {
        let mut config = default_config();
        config.enable_energy_budget = true;
        config.energy_budget_joules_per_sec = Some(1e10);
        let mut mgr = SubstrateManager::new(&config);
        let now = std::time::Instant::now();
        mgr.energy_rate_window_started_at =
            Some(now + std::time::Duration::from_secs(1));

        mgr.tick_energy_at(&config, now);

        assert!(
            !mgr.consciousness_viable,
            "reversed monotonic evidence must not authorize continued viability"
        );
        assert_eq!(mgr.estimated_energy_rate_watts, 0.0);
        assert_eq!(mgr.energy_rate_window_seconds, 0.0);
        assert_eq!(mgr.total_energy_spent, 0.0, "sample tied to reversed timing must not be accounted");
        assert_eq!(mgr.energy_rate_window_joules, 0.0);
        assert_eq!(mgr.energy_rate_window_started_at, None);
        assert_eq!(mgr.last_energy_spent, 0.0);
    }

    #[test]
    fn test_modeled_energy_per_cycle_is_independent_of_cognitive_tau() {
        let mut config = default_config();
        config.enable_energy_budget = true;
        config.energy_budget_joules_per_sec = Some(f64::MAX);

        for tau in [0.5_f32, 1.0, 2.0] {
            let mut mgr = SubstrateManager::new(&config);
            mgr.tau_factor = tau;
            let expected_energy = mgr.energy_per_cycle;
            mgr.tick_energy_at(&config, std::time::Instant::now());

            assert_eq!(
                mgr.last_energy_spent, expected_energy,
                "cognitive tau={tau} must not masquerade as an energy multiplier"
            );
            assert_eq!(mgr.total_energy_spent, expected_energy);
            assert_eq!(mgr.energy_rate_window_joules, expected_energy);
            assert!(mgr.consciousness_viable);
        }
    }

    #[test]
    fn test_consciousness_collapse_gating() {
        let mut config = default_config();
        config.enable_validation_overlay = true;
        config.validation_skepticism_floor = 0.0; // maximally skeptical
        config.substrate_type = SubstrateType::ExoticSubstrate;
        let mgr = SubstrateManager::new(&config);
        // Exotic substrate with max skepticism should degrade
        assert!(
            mgr.should_degrade_consciousness() || mgr.effective_feasibility < 0.3,
            "Exotic substrate with zero floor should degrade: eff={:.4}, viable={}",
            mgr.effective_feasibility,
            mgr.consciousness_viable
        );
    }

    #[test]
    fn test_should_degrade_when_energy_rate_exceeds_budget() {
        let mut config = default_config();
        config.enable_energy_budget = true;
        config.energy_budget_joules_per_sec = Some(1e-20); // impossibly small modeled-watt budget
        let mut mgr = SubstrateManager::new(&config);
        let window_start = std::time::Instant::now();
        mgr.energy_rate_window_started_at = Some(window_start);
        assert!(!mgr.should_degrade_consciousness());
        mgr.tick_energy_at(&config, window_start); // seed the first cycle's modeled energy
        mgr.tick_energy_at(&config, window_start + std::time::Duration::from_secs(1));
        assert!(
            mgr.should_degrade_consciousness(),
            "Should degrade after the model-estimated energy rate exceeds the rate budget"
        );
    }

    #[test]
    fn test_should_not_degrade_biological() {
        let mut config = default_config();
        config.substrate_type = SubstrateType::BiologicalNeurons;
        config.enable_validation_overlay = true;
        config.validation_skepticism_floor = 0.5;
        let mgr = SubstrateManager::new(&config);
        assert!(
            !mgr.should_degrade_consciousness(),
            "Biological substrate should never degrade: eff={:.4}",
            mgr.effective_feasibility
        );
    }

    #[test]
    fn test_per_region_feasibility_fallback() {
        let cfg = default_config(); // no per-region config
        let mgr = SubstrateManager::new(&cfg);
        // With no per-region substrates, should fall back to global effective_feasibility
        let region_f = mgr.region_feasibility(CorticalRegion::Prefrontal);
        assert!(
            (region_f - mgr.effective_feasibility as f32).abs() < f32::EPSILON,
            "Without per-region config, should fall back to global: region={region_f:.4}, global={:.4}",
            mgr.effective_feasibility
        );
    }

    #[test]
    fn test_encoding_noise_telemetry_quantum() {
        let mut cfg = default_config();
        cfg.enable_substrate_encoding_noise = true;
        cfg.enable_substrate_speed_modulation = true;
        cfg.substrate_type = SubstrateType::QuantumComputer;
        let mut mgr = SubstrateManager::new(&cfg);
        // Quantum has scale_pressure ≈ -7.0 (10^4 qubits vs 10^11 neurons)
        assert!(
            mgr.scale_pressure < -5.0,
            "Quantum scale_pressure should be very negative: {:.2}",
            mgr.scale_pressure
        );
        let telem = mgr.telemetry(&cfg);
        // Encoding noise should be capped at 0.1 (7.0/70.0)
        assert!(
            telem.substrate_encoding_noise > 0.05,
            "Quantum should have significant encoding noise: {:.4}",
            telem.substrate_encoding_noise
        );
        assert!(
            telem.substrate_encoding_noise <= 0.1,
            "Encoding noise should be capped at 0.1: {:.4}",
            telem.substrate_encoding_noise
        );
    }

    #[test]
    fn test_encoding_noise_zero_for_biological() {
        let mut cfg = default_config();
        cfg.enable_substrate_encoding_noise = true;
        cfg.enable_substrate_speed_modulation = true;
        cfg.substrate_type = SubstrateType::BiologicalNeurons;
        let mut mgr = SubstrateManager::new(&cfg);
        // Bio is the reference — scale_pressure ≈ 0.0
        let telem = mgr.telemetry(&cfg);
        assert!(
            telem.substrate_encoding_noise < f32::EPSILON,
            "Biological should have zero encoding noise: {:.4}",
            telem.substrate_encoding_noise
        );
    }

    #[test]
    fn test_encoding_noise_zero_for_silicon() {
        let mut cfg = default_config();
        cfg.enable_substrate_encoding_noise = true;
        cfg.enable_substrate_speed_modulation = true;
        cfg.substrate_type = SubstrateType::SiliconDigital;
        let mut mgr = SubstrateManager::new(&cfg);
        // Silicon has positive scale_pressure (more units than bio)
        let telem = mgr.telemetry(&cfg);
        assert!(
            telem.substrate_encoding_noise < f32::EPSILON,
            "Silicon (positive scale_pressure) should have zero noise: {:.4}",
            telem.substrate_encoding_noise
        );
    }

    #[test]
    fn test_encoding_noise_disabled_by_default() {
        let mut cfg = default_config();
        // enable_substrate_encoding_noise is false by default
        cfg.enable_substrate_speed_modulation = true;
        cfg.substrate_type = SubstrateType::QuantumComputer;
        let mut mgr = SubstrateManager::new(&cfg);
        let telem = mgr.telemetry(&cfg);
        assert!(
            telem.substrate_encoding_noise < f32::EPSILON,
            "Encoding noise should be zero when disabled: {:.4}",
            telem.substrate_encoding_noise
        );
    }

    #[test]
    fn test_binding_capability_varies_by_substrate() {
        let bio_cfg = {
            let mut c = default_config();
            c.substrate_type = SubstrateType::BiologicalNeurons;
            c
        };
        let biochem_cfg = {
            let mut c = default_config();
            c.substrate_type = SubstrateType::BiochemicalComputer;
            c
        };
        let bio_mgr = SubstrateManager::new(&bio_cfg);
        let biochem_mgr = SubstrateManager::new(&biochem_cfg);
        assert!(
            bio_mgr.binding_capability(&bio_cfg) > biochem_mgr.binding_capability(&biochem_cfg),
            "Biological binding ({}) should exceed biochemical ({})",
            bio_mgr.binding_capability(&bio_cfg),
            biochem_mgr.binding_capability(&biochem_cfg),
        );
    }

    #[test]
    fn test_workspace_capability_varies_by_substrate() {
        let bio_cfg = {
            let mut c = default_config();
            c.substrate_type = SubstrateType::BiologicalNeurons;
            c
        };
        let exotic_cfg = {
            let mut c = default_config();
            c.substrate_type = SubstrateType::ExoticSubstrate;
            c
        };
        let bio_mgr = SubstrateManager::new(&bio_cfg);
        let exotic_mgr = SubstrateManager::new(&exotic_cfg);
        assert!(
            bio_mgr.workspace_capability(&bio_cfg) > exotic_mgr.workspace_capability(&exotic_cfg),
            "Biological workspace ({}) should exceed exotic ({})",
            bio_mgr.workspace_capability(&bio_cfg),
            exotic_mgr.workspace_capability(&exotic_cfg),
        );
    }

    #[test]
    fn test_attention_capability_varies_by_substrate() {
        let silicon_cfg = {
            let mut c = default_config();
            c.substrate_type = SubstrateType::SiliconDigital;
            c
        };
        let biochem_cfg = {
            let mut c = default_config();
            c.substrate_type = SubstrateType::BiochemicalComputer;
            c
        };
        let silicon_mgr = SubstrateManager::new(&silicon_cfg);
        let biochem_mgr = SubstrateManager::new(&biochem_cfg);
        assert!(
            silicon_mgr.attention_capability(&silicon_cfg)
                > biochem_mgr.attention_capability(&biochem_cfg),
            "Silicon attention ({}) should exceed biochemical ({})",
            silicon_mgr.attention_capability(&silicon_cfg),
            biochem_mgr.attention_capability(&biochem_cfg),
        );
    }

    #[test]
    fn test_all_capabilities_in_unit_range() {
        let substrates = [
            SubstrateType::BiologicalNeurons,
            SubstrateType::SiliconDigital,
            SubstrateType::QuantumComputer,
            SubstrateType::PhotonicProcessor,
            SubstrateType::NeuromorphicChip,
            SubstrateType::BiochemicalComputer,
            SubstrateType::HybridSystem,
            SubstrateType::ExoticSubstrate,
        ];
        for sub in &substrates {
            let mut cfg = default_config();
            cfg.substrate_type = *sub;
            let mgr = SubstrateManager::new(&cfg);
            let b = mgr.binding_capability(&cfg);
            let w = mgr.workspace_capability(&cfg);
            let a = mgr.attention_capability(&cfg);
            assert!(
                (0.0..=1.0).contains(&b) && (0.0..=1.0).contains(&w) && (0.0..=1.0).contains(&a),
                "{:?}: binding={}, workspace={}, attention={} — all must be in [0,1]",
                sub,
                b,
                w,
                a,
            );
        }
    }
}
