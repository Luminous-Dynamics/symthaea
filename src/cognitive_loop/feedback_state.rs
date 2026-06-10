// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Feedback Variable Integration — Consensus-based proposal system.
//!
//! All feedback variable mutations (`prediction_confidence`, `fep_lr_boost`,
//! `exploration_urge`, `adaptive_threshold_scale`) are routed through attributed
//! proposals. Each cycle, proposals are collected via helper methods and
//! integrated using consensus (averaged adds, geometric mean scales) to produce
//! noise-resistant values for the next cycle.

use std::fmt;

// ═══════════════════════════════════════════════════════════════════════════════
// PROPOSAL TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// A single proposed change to a feedback variable.
#[derive(Debug, Clone, Copy)]
pub enum FeedbackProposal {
    /// Additive delta: `value += delta`
    Add(f64),
    /// Multiplicative factor: `value *= factor`
    Scale(f64),
    /// Hard reset to a specific value (used sparingly: inference mode init, etc.)
    Set(f64),
}

/// Priority tier for feedback proposals.
///
/// Higher-priority proposals have more weight in consensus integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    /// Aesthetic/flow/harmony signals (weight 0.5x)
    Aesthetic = 0,
    /// Default: cognitive signals — binding, reasoning, surprise (weight 1.0x)
    Cognitive = 1,
    /// Homeostatic regulation — neuromod bath, FEP, sleep/wake (weight 2.0x)
    Homeostatic = 2,
    /// Safety-critical — agent vetoes, moral concerns, seizure protection (weight 3.0x)
    Safety = 3,
}

impl Priority {
    /// Base weight multiplier for this priority tier.
    pub fn weight(self) -> f64 {
        match self {
            Priority::Aesthetic => 0.5,
            Priority::Cognitive => 1.0,
            Priority::Homeostatic => 2.0,
            Priority::Safety => 3.0,
        }
    }

    /// Confidence-scaled weight: `base_weight × confidence`.
    ///
    /// A high-confidence Safety signal (conf=0.9) gets weight 2.7,
    /// while a low-confidence one (conf=0.3) gets weight 0.9 —
    /// potentially less than a high-confidence Cognitive signal (1.0).
    /// This prevents low-quality safety signals from drowning real cognitive work.
    pub fn weighted(self, confidence: f32) -> f64 {
        self.weight() * confidence.clamp(0.0, 1.0) as f64
    }
}

impl Default for Priority {
    fn default() -> Self {
        Priority::Cognitive
    }
}

/// An attributed proposal — who proposed what, at what priority.
#[derive(Debug, Clone)]
pub struct AttributedProposal {
    /// Subsystem that made this proposal (static str for zero-alloc attribution)
    pub source: &'static str,
    /// The proposal itself
    pub proposal: FeedbackProposal,
    /// Priority tier (higher = more weight in consensus)
    pub priority: Priority,
    /// Confidence in the proposal (0.0–1.0). Scales the priority weight.
    /// 1.0 = full weight (default), 0.5 = half weight.
    pub confidence: f32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROPOSAL COLLECTOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Collects proposals from subsystems during a cycle, then integrates them.
///
/// Reset at the start of each cycle via `clear()`.
#[derive(Debug, Clone)]
pub struct ProposalCollector {
    proposals: Vec<AttributedProposal>,
    /// Cached integration result — invalidated on every `propose()`, lazily
    /// recomputed on `integrate()`. Avoids O(n²) when helpers sync fields.
    cached: Option<(f64, f64, f64, f64)>, // (base, min, max, effective)
}

impl ProposalCollector {
    pub fn new() -> Self {
        Self {
            proposals: Vec::with_capacity(32),
            cached: None,
        }
    }

    /// Record a proposal from a named subsystem (default priority: Cognitive, confidence: 1.0).
    pub fn propose(&mut self, source: &'static str, proposal: FeedbackProposal) {
        self.proposals.push(AttributedProposal {
            source,
            proposal,
            priority: Priority::default(),
            confidence: 1.0,
        });
        self.cached = None;
    }

    /// Record a proposal with an explicit priority tier (confidence: 1.0).
    pub fn propose_with_priority(
        &mut self,
        source: &'static str,
        proposal: FeedbackProposal,
        priority: Priority,
    ) {
        self.proposals.push(AttributedProposal {
            source,
            proposal,
            priority,
            confidence: 1.0,
        });
        self.cached = None;
    }

    /// Record a proposal with confidence-scaled priority (#7 adaptive scaling).
    ///
    /// Effective weight = `priority.weight() × confidence`. A low-confidence
    /// Safety signal (conf=0.3) gets weight 0.9, potentially less than a
    /// high-confidence Cognitive signal (1.0).
    pub fn propose_weighted(
        &mut self,
        source: &'static str,
        proposal: FeedbackProposal,
        priority: Priority,
        confidence: f32,
    ) {
        self.proposals.push(AttributedProposal {
            source,
            proposal,
            priority,
            confidence: confidence.clamp(0.0, 1.0),
        });
        self.cached = None;
    }

    /// Number of proposals collected this cycle.
    pub fn len(&self) -> usize {
        self.proposals.len()
    }

    /// Whether no proposals have been collected.
    pub fn is_empty(&self) -> bool {
        self.proposals.is_empty()
    }

    /// Clear all proposals (called at cycle start).
    pub fn clear(&mut self) {
        self.proposals.clear();
        self.cached = None;
    }

    /// Iterate over proposals for inspection.
    pub fn proposals(&self) -> &[AttributedProposal] {
        &self.proposals
    }

    /// Count unique source names among proposals.
    #[allow(dead_code)]
    pub fn unique_sources(&self) -> usize {
        let mut seen: Vec<&'static str> = Vec::with_capacity(self.proposals.len());
        for ap in &self.proposals {
            if !seen.contains(&ap.source) {
                seen.push(ap.source);
            }
        }
        seen.len()
    }

    /// Dump all proposals as `(source, description)` pairs for trace logging.
    ///
    /// Each entry describes the proposal kind and magnitude, e.g.
    /// `("binding_strong", "Add(+0.0300)")`.
    pub fn dump_proposals(&self) -> Vec<(&'static str, String)> {
        self.proposals
            .iter()
            .map(|ap| {
                let desc = match ap.proposal {
                    FeedbackProposal::Add(d) => format!("Add({d:+.4})"),
                    FeedbackProposal::Scale(f) => format!("Scale({f:.4})"),
                    FeedbackProposal::Set(v) => format!("Set({v:.4})"),
                };
                (ap.source, desc)
            })
            .collect()
    }

    /// Compute the conflict ratio among Add proposals.
    /// Returns fraction of proposals that disagree in sign with the majority.
    /// 0.0 = unanimous, 0.5 = maximally conflicted, 1.0 only if single contrary.
    /// Science: conflicting proposals = subsystems disagree about direction.
    pub fn conflict_ratio(&self) -> f32 {
        let adds: Vec<f64> = self
            .proposals
            .iter()
            .filter_map(|ap| match ap.proposal {
                FeedbackProposal::Add(d) => Some(d),
                _ => None,
            })
            .collect();
        if adds.len() < 2 {
            return 0.0;
        }
        let positive = adds.iter().filter(|&&d| d > 0.0).count();
        let negative = adds.iter().filter(|&&d| d < 0.0).count();
        let minority = positive.min(negative);
        minority as f32 / adds.len() as f32
    }

    /// Integrate all proposals using priority-weighted consensus mode.
    ///
    /// - `Set` proposals: highest-priority Set wins (ties: last one)
    /// - `Add` proposals: priority-weighted average, then applied
    /// - `Scale` proposals: priority-weighted geometric mean, then applied
    ///
    /// When all proposals share the same priority, reduces to unweighted consensus.
    pub fn integrate(
        &mut self,
        base_value: f64,
        clamp_min: f64,
        clamp_max: f64,
    ) -> IntegrationResult {
        // Cache hit: same parameters → return cached effective value
        if let Some((cb, cmin, cmax, eff)) = self.cached {
            if (cb - base_value).abs() < 1e-15
                && (cmin - clamp_min).abs() < 1e-15
                && (cmax - clamp_max).abs() < 1e-15
            {
                return IntegrationResult {
                    effective: eff,
                    n_sets: self
                        .proposals
                        .iter()
                        .filter(|p| matches!(p.proposal, FeedbackProposal::Set(_)))
                        .count(),
                    n_adds: self
                        .proposals
                        .iter()
                        .filter(|p| matches!(p.proposal, FeedbackProposal::Add(_)))
                        .count(),
                    n_scales: self
                        .proposals
                        .iter()
                        .filter(|p| matches!(p.proposal, FeedbackProposal::Scale(_)))
                        .count(),
                };
            }
        }
        let mut sets: Vec<(&'static str, f64, Priority)> = Vec::new();
        let mut adds: Vec<(f64, f64)> = Vec::new(); // (delta, weight)
        let mut scales: Vec<(f64, f64)> = Vec::new(); // (factor, weight)

        for ap in &self.proposals {
            let w = ap.priority.weight() * ap.confidence.clamp(0.0, 1.0) as f64;
            match ap.proposal {
                FeedbackProposal::Set(v) => sets.push((ap.source, v, ap.priority)),
                FeedbackProposal::Add(d) => adds.push((d, w)),
                FeedbackProposal::Scale(f) => scales.push((f, w)),
            }
        }

        // Start from base or highest-priority Set (ties: last one wins)
        let mut value = if !sets.is_empty() {
            let max_pri = sets
                .iter()
                .map(|(_, _, p)| *p)
                .max()
                .expect("non-empty by guard");
            sets.iter()
                .rev()
                .find(|(_, _, p)| *p == max_pri)
                .map(|(_, v, _)| *v)
                .unwrap_or(base_value)
        } else {
            base_value
        };

        // Priority-weighted average of additive proposals
        if !adds.is_empty() {
            let total_w: f64 = adds.iter().map(|(_, w)| w).sum();
            if total_w > 0.0 {
                let weighted_sum: f64 = adds.iter().map(|(d, w)| d * w).sum();
                value += weighted_sum / total_w;
            }
        }
        // Priority-weighted geometric mean of multiplicative proposals
        if !scales.is_empty() {
            let valid: Vec<(f64, f64)> = scales
                .iter()
                .filter(|(f, _)| *f > 0.0 && f.is_finite())
                .copied()
                .collect();
            if !valid.is_empty() {
                let total_w: f64 = valid.iter().map(|(_, w)| w).sum();
                if total_w > 0.0 {
                    let wlog: f64 = valid.iter().map(|(f, w)| f.ln() * w).sum::<f64>();
                    let geo_mean = (wlog / total_w).exp();
                    if geo_mean.is_finite() {
                        value *= geo_mean;
                    }
                }
            }
        }

        // Guard: NaN from accumulated proposals would bypass clamp
        if !value.is_finite() {
            value = clamp_min;
        }
        value = value.clamp(clamp_min, clamp_max);

        // Cache for subsequent calls with same parameters
        self.cached = Some((base_value, clamp_min, clamp_max, value));

        IntegrationResult {
            effective: value,
            n_sets: sets.len(),
            n_adds: adds.len(),
            n_scales: scales.len(),
        }
    }

    /// Integrate without mutating (for read-only contexts like conflict_ratio callers).
    /// This is the uncached path — use sparingly.
    pub fn integrate_readonly(
        &self,
        base_value: f64,
        clamp_min: f64,
        clamp_max: f64,
    ) -> IntegrationResult {
        // Check cache first
        if let Some((cb, cmin, cmax, eff)) = self.cached {
            if (cb - base_value).abs() < 1e-15
                && (cmin - clamp_min).abs() < 1e-15
                && (cmax - clamp_max).abs() < 1e-15
            {
                return IntegrationResult {
                    effective: eff,
                    n_sets: self
                        .proposals
                        .iter()
                        .filter(|p| matches!(p.proposal, FeedbackProposal::Set(_)))
                        .count(),
                    n_adds: self
                        .proposals
                        .iter()
                        .filter(|p| matches!(p.proposal, FeedbackProposal::Add(_)))
                        .count(),
                    n_scales: self
                        .proposals
                        .iter()
                        .filter(|p| matches!(p.proposal, FeedbackProposal::Scale(_)))
                        .count(),
                };
            }
        }
        // Full compute without caching
        let mut sets_n = 0usize;
        let mut adds: Vec<(f64, f64)> = Vec::new();
        let mut scales: Vec<(f64, f64)> = Vec::new();
        let mut last_set: Option<(f64, Priority)> = None;

        for ap in &self.proposals {
            let w = ap.priority.weight();
            match ap.proposal {
                FeedbackProposal::Set(v) => {
                    sets_n += 1;
                    if last_set.map_or(true, |(_, p)| ap.priority >= p) {
                        last_set = Some((v, ap.priority));
                    }
                }
                FeedbackProposal::Add(d) => adds.push((d, w)),
                FeedbackProposal::Scale(f) => scales.push((f, w)),
            }
        }

        let mut value = last_set.map_or(base_value, |(v, _)| v);
        if !adds.is_empty() {
            let tw: f64 = adds.iter().map(|(_, w)| w).sum();
            if tw > 0.0 {
                value += adds.iter().map(|(d, w)| d * w).sum::<f64>() / tw;
            }
        }
        if !scales.is_empty() {
            let valid: Vec<(f64, f64)> = scales
                .iter()
                .filter(|(f, _)| *f > 0.0 && f.is_finite())
                .copied()
                .collect();
            if !valid.is_empty() {
                let tw: f64 = valid.iter().map(|(_, w)| w).sum();
                if tw > 0.0 {
                    let wlog: f64 = valid.iter().map(|(f, w)| f.ln() * w).sum::<f64>();
                    let gm = (wlog / tw).exp();
                    if gm.is_finite() {
                        value *= gm;
                    }
                }
            }
        }
        value = value.clamp(clamp_min, clamp_max);
        IntegrationResult {
            effective: value,
            n_sets: sets_n,
            n_adds: adds.len(),
            n_scales: scales.len(),
        }
    }
}

impl Default for ProposalCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of integrating proposals.
#[derive(Debug, Clone)]
pub struct IntegrationResult {
    pub effective: f64,
    pub n_sets: usize,
    pub n_adds: usize,
    pub n_scales: usize,
}

impl fmt::Display for IntegrationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "eff={:.4} ({}S+{}A+{}M)",
            self.effective, self.n_sets, self.n_adds, self.n_scales
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FEEDBACK STATE — top-level container
// ═══════════════════════════════════════════════════════════════════════════════

/// Consensus-integrated feedback values computed at cycle end.
///
/// These noise-resistant values (averaged adds, geometric mean scales)
/// are stored for deferred writeback at the start of the next cycle.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ConsensusResult {
    /// Consensus-integrated prediction_confidence for next cycle
    pub consensus_confidence: f64,
    /// Consensus-integrated fep_lr_boost value for next cycle
    pub consensus_lr: f64,
    /// Consensus-integrated exploration_urge for next cycle
    pub consensus_exploration: f64,
    /// Consensus-integrated adaptive_threshold_scale for next cycle
    pub consensus_threshold: f64,
}

/// Decoupled feedback state for the cognitive loop.
///
/// All feedback variable mutations are routed through attributed proposals.
/// At cycle end, proposals are integrated using consensus (averaged adds,
/// geometric mean scales) and the result is stored for deferred writeback
/// at the next cycle start.
#[derive(Debug, Clone)]
pub(crate) struct FeedbackState {
    /// Proposals for `prediction_confidence` (0.0–1.0)
    pub confidence: ProposalCollector,
    /// Proposals for `fep_lr_boost` (1.0–3.0)
    pub learning_rate: ProposalCollector,
    /// Proposals for `exploration_urge` (0.0–1.0)
    pub exploration: ProposalCollector,
    /// Proposals for `adaptive_threshold_scale` (0.5–2.0)
    pub threshold: ProposalCollector,

    // ── Cycle-start values for consensus computation ────────────────────
    cycle_start_confidence: f64,
    cycle_start_lr: f64,
    cycle_start_exploration: f64,
    cycle_start_threshold: f64,

    // ── Cycle-end integration results (for telemetry / debugging) ───────
    /// Last integrated confidence result
    pub last_confidence_integration: Option<IntegrationResult>,
    /// Last integrated LR result
    pub last_lr_integration: Option<IntegrationResult>,
    /// Last integrated exploration result
    pub last_exploration_integration: Option<IntegrationResult>,
    /// Last integrated threshold result
    pub last_threshold_integration: Option<IntegrationResult>,

    /// Last computed consensus result (set by `end_cycle()`).
    pub last_consensus: Option<ConsensusResult>,

    /// High-water mark of total feedback proposals in any single cycle.
    pub feedback_signals_high_water: u32,

    /// Number of channels dampened this cycle (0–4).
    pub feedback_dampened_count: u32,

    // ── Deferred consensus writeback ────────────────────────────────────
    /// Consensus confidence to apply via helper at the start of the next cycle.
    /// Stored by `store_consensus_for_next_cycle()`, consumed by `apply_pending_consensus()`.
    pending_consensus_confidence: Option<f64>,
    /// Consensus LR to apply via helper at the start of the next cycle.
    pending_consensus_lr: Option<f64>,
    /// Consensus exploration to apply via helper at the start of the next cycle.
    pending_consensus_exploration: Option<f64>,
    /// Consensus threshold to apply via helper at the start of the next cycle.
    pending_consensus_threshold: Option<f64>,
}

impl FeedbackState {
    pub fn new() -> Self {
        Self {
            confidence: ProposalCollector::new(),
            learning_rate: ProposalCollector::new(),
            exploration: ProposalCollector::new(),
            threshold: ProposalCollector::new(),
            cycle_start_confidence: 0.5,
            cycle_start_lr: 1.0,
            cycle_start_exploration: 0.0, // matches CuriosityDrive::default()
            cycle_start_threshold: 1.0,
            last_confidence_integration: None,
            last_lr_integration: None,
            last_exploration_integration: None,
            last_threshold_integration: None,
            last_consensus: None,
            feedback_signals_high_water: 0,
            feedback_dampened_count: 0,
            pending_consensus_confidence: None,
            pending_consensus_lr: None,
            pending_consensus_exploration: None,
            pending_consensus_threshold: None,
        }
    }

    /// Clear all collectors at cycle start.
    pub fn begin_cycle(&mut self) {
        self.confidence.clear();
        self.learning_rate.clear();
        self.exploration.clear();
        self.threshold.clear();
    }

    /// Integrate all feedback variables using consensus. Call at cycle end.
    ///
    /// Returns consensus-integrated values (averaged adds, geometric mean scales)
    /// for all 4 feedback variables.
    ///
    /// `consecutive_full_dampen`: count from carryover (for streak-based freeze).
    /// `in_flow`: whether the system is in flow state (for dampening relaxation).
    /// `flow_intensity`: flow intensity (0.0–1.0) for threshold widening.
    pub fn end_cycle_ext(
        &mut self,
        _current_confidence: f64,
        _current_lr: f64,
        _current_exploration: f64,
        _current_threshold: f64,
        consecutive_full_dampen: u32,
        in_flow: bool,
        flow_intensity: f32,
    ) -> ConsensusResult {
        let base_confidence = self.cycle_start_confidence;
        let base_lr = self.cycle_start_lr;
        let base_exploration = self.cycle_start_exploration;
        let base_threshold = self.cycle_start_threshold;

        // Update high-water mark of feedback signals per cycle
        let total_this_cycle = self.total_proposals() as u32;
        if total_this_cycle > self.feedback_signals_high_water {
            self.feedback_signals_high_water = total_this_cycle;
        }

        let conf_result = self.confidence.integrate(base_confidence, 0.0, 1.0);
        let lr_result = self.learning_rate.integrate(base_lr, 1.0, 3.0);
        let explore_result = self.exploration.integrate(base_exploration, 0.0, 1.0);
        let thresh_result = self.threshold.integrate(base_threshold, 0.5, 2.0);

        let consensus = ConsensusResult {
            consensus_confidence: conf_result.effective,
            consensus_lr: lr_result.effective,
            consensus_exploration: explore_result.effective,
            consensus_threshold: thresh_result.effective,
        };

        self.last_confidence_integration = Some(conf_result);
        self.last_lr_integration = Some(lr_result);
        self.last_exploration_integration = Some(explore_result);
        self.last_threshold_integration = Some(thresh_result);

        // Session 9 Item 2: Dampening streak freeze — if all 4 channels were dampened
        // for N+ consecutive cycles, freeze feedback to cycle-start values for 1 cycle.
        // Science: Turrigiano (2008) — homeostatic plasticity includes brief synaptic silencing.
        use super::thresholds::CONSENSUS_FREEZE_STREAK_THRESHOLD;
        if consecutive_full_dampen >= CONSENSUS_FREEZE_STREAK_THRESHOLD {
            self.feedback_dampened_count = 0; // Reset streak by not dampening
            self.last_consensus = Some(ConsensusResult {
                consensus_confidence: base_confidence,
                consensus_lr: base_lr,
                consensus_exploration: base_exploration,
                consensus_threshold: base_threshold,
            });
            return ConsensusResult {
                consensus_confidence: base_confidence,
                consensus_lr: base_lr,
                consensus_exploration: base_exploration,
                consensus_threshold: base_threshold,
            };
        }

        // Adaptive dampening: if consensus diverges too much from cycle-start on any
        // channel, dampen toward cycle-start. Prevents feedback runaway.
        // Kelso (1995): metastable dynamics resist abrupt transitions.
        //
        // Self-tightening: if most channels were dampened last cycle (>2), tighten threshold.
        // Session 9 Item 5: Flow relaxation — during flow, widen threshold to 50%.
        // Csikszentmihalyi (1990): flow requires reduced self-monitoring overhead.
        let divergence_threshold = if in_flow && flow_intensity > 0.5 {
            0.5 // Flow: trust the system's momentum
        } else if self.feedback_dampened_count > 2 {
            0.2 // Self-tightening after heavy dampening
        } else {
            0.3
        };
        // Low signal diversity → extra dampening blend (60% toward start).
        let diversity = self.signal_diversity();
        let blend_toward_start = if diversity < 0.3 { 0.6 } else { 0.5 };
        let mut dampened_count = 0u32;
        let dampen = |consensus_val: f64, start_val: f64, count: &mut u32| -> f64 {
            if start_val.abs() < 1e-10 {
                return consensus_val;
            }
            let ratio = (consensus_val - start_val).abs() / start_val.abs().max(0.1);
            if ratio > divergence_threshold {
                *count += 1;
                consensus_val * (1.0 - blend_toward_start) + start_val * blend_toward_start
            } else {
                consensus_val
            }
        };
        let consensus = ConsensusResult {
            consensus_confidence: dampen(
                consensus.consensus_confidence,
                base_confidence,
                &mut dampened_count,
            ),
            consensus_lr: dampen(consensus.consensus_lr, base_lr, &mut dampened_count),
            consensus_exploration: dampen(
                consensus.consensus_exploration,
                base_exploration,
                &mut dampened_count,
            ),
            consensus_threshold: dampen(
                consensus.consensus_threshold,
                base_threshold,
                &mut dampened_count,
            ),
        };
        self.feedback_dampened_count = dampened_count;

        self.last_consensus = Some(consensus);

        consensus
    }

    /// Cycle-start confidence snapshot.
    pub fn cycle_start_confidence(&self) -> f64 {
        self.cycle_start_confidence
    }
    /// Cycle-start LR snapshot.
    pub fn cycle_start_lr(&self) -> f64 {
        self.cycle_start_lr
    }
    /// Cycle-start exploration snapshot.
    pub fn cycle_start_exploration(&self) -> f64 {
        self.cycle_start_exploration
    }
    /// Cycle-start threshold snapshot.
    pub fn cycle_start_threshold(&self) -> f64 {
        self.cycle_start_threshold
    }

    /// Backwards-compatible `end_cycle` for tests — delegates to `end_cycle_ext`
    /// with no streak/flow context.
    #[cfg(test)]
    pub fn end_cycle(
        &mut self,
        current_confidence: f64,
        current_lr: f64,
        current_exploration: f64,
        current_threshold: f64,
    ) -> ConsensusResult {
        self.end_cycle_ext(
            current_confidence,
            current_lr,
            current_exploration,
            current_threshold,
            0,
            false,
            0.0,
        )
    }

    /// Record cycle-start values for consensus computation.
    ///
    /// Call at cycle start (after `begin_cycle()`) with the current field values.
    pub fn snapshot_cycle_start(
        &mut self,
        confidence: f64,
        lr: f64,
        exploration: f64,
        threshold: f64,
    ) {
        self.cycle_start_confidence = confidence;
        self.cycle_start_lr = lr;
        self.cycle_start_exploration = exploration;
        self.cycle_start_threshold = threshold;
    }

    /// Store consensus-smoothed values for deferred application at the next cycle start.
    ///
    /// Called at cycle end. The values are applied via helpers by
    /// `apply_pending_consensus()` at the next cycle start.
    pub fn store_consensus_for_next_cycle(&mut self, consensus: &ConsensusResult) {
        self.pending_consensus_confidence = Some(consensus.consensus_confidence);
        self.pending_consensus_lr = Some(consensus.consensus_lr);
        self.pending_consensus_exploration = Some(consensus.consensus_exploration);
        self.pending_consensus_threshold = Some(consensus.consensus_threshold);
    }

    /// Return any pending consensus values for the caller to apply via helpers.
    ///
    /// Call at cycle start (after `begin_cycle()` + `snapshot_cycle_start()`).
    /// Returns `(confidence, lr, exploration, threshold)` — the caller routes
    /// these through `set_confidence` / `set_lr` / `set_exploration` / `set_threshold`
    /// helpers, which emit the Set proposals themselves.
    pub fn apply_pending_consensus(
        &mut self,
    ) -> (Option<f64>, Option<f64>, Option<f64>, Option<f64>) {
        let confidence = self.pending_consensus_confidence.take();
        let lr = self.pending_consensus_lr.take();
        let exploration = self.pending_consensus_exploration.take();
        let threshold = self.pending_consensus_threshold.take();
        (confidence, lr, exploration, threshold)
    }

    /// How many total proposals were recorded this cycle.
    pub fn total_proposals(&self) -> usize {
        self.confidence.len()
            + self.learning_rate.len()
            + self.exploration.len()
            + self.threshold.len()
    }

    /// Return (dominant_source_name, concentration) in a single pass.
    /// Avoids the double Vec allocation from calling both methods separately.
    /// Concentration = fraction of total proposals from the dominant source (0.0 if none).
    fn dominant_source_info(&self) -> (&'static str, f32) {
        // Use a small inline buffer — typical subsystem count is <16 distinct sources
        let mut counts: [(&'static str, usize); 16] = [("", 0); 16];
        let mut n_sources = 0usize;
        let mut total = 0usize;
        for collector in [
            &self.confidence,
            &self.learning_rate,
            &self.exploration,
            &self.threshold,
        ] {
            for ap in collector.proposals() {
                total += 1;
                if let Some(entry) = counts[..n_sources]
                    .iter_mut()
                    .find(|(s, _)| *s == ap.source)
                {
                    entry.1 += 1;
                } else if n_sources < 16 {
                    counts[n_sources] = (ap.source, 1);
                    n_sources += 1;
                }
            }
        }
        if total == 0 {
            return ("", 0.0);
        }
        let (name, max_count) = counts[..n_sources]
            .iter()
            .max_by_key(|(_, c)| *c)
            .copied()
            .unwrap_or(("", 0));
        (name, max_count as f32 / total as f32)
    }

    /// Return the source name that contributed the most proposals this cycle.
    /// Ties broken arbitrarily. Returns "" if no proposals.
    pub fn dominant_source(&self) -> &'static str {
        self.dominant_source_info().0
    }

    /// Fraction of total proposals contributed by the dominant source.
    /// 0.0 if no proposals.
    pub fn dominant_source_concentration(&self) -> f32 {
        self.dominant_source_info().1
    }

    /// Average conflict ratio across all 4 channels.
    /// High conflict (>0.3) = subsystems disagree about direction.
    /// Science: Dayan & Daw (2008) — model-based/model-free disagreement signals meta-uncertainty.
    pub fn avg_conflict_ratio(&self) -> f32 {
        let ratios = [
            self.confidence.conflict_ratio(),
            self.learning_rate.conflict_ratio(),
            self.exploration.conflict_ratio(),
            self.threshold.conflict_ratio(),
        ];
        ratios.iter().sum::<f32>() / 4.0
    }

    /// Number of LR proposals collected this cycle.
    pub fn lr_proposal_count(&self) -> u32 {
        self.learning_rate.len() as u32
    }

    /// Count distinct source names across all 4 channels.
    /// Science: Dehaene (2014) — healthy cognition requires multi-source consensus.
    pub fn distinct_source_count(&self) -> usize {
        let mut seen: Vec<&'static str> = Vec::with_capacity(16);
        for collector in [
            &self.confidence,
            &self.learning_rate,
            &self.exploration,
            &self.threshold,
        ] {
            for ap in collector.proposals() {
                if !seen.contains(&ap.source) {
                    seen.push(ap.source);
                }
            }
        }
        seen.len()
    }

    // Feedback signal diversity: unique sources / total proposals.
    // High diversity (>0.7) = many subsystems contributing = healthy.
    // Low diversity (<0.3) = few subsystems dominating = potential bias.

    // ── Mid-cycle effective value accessors ─────────────────────────────

    /// Effective prediction_confidence from cycle-start + proposals so far.
    pub fn effective_confidence(&mut self) -> f64 {
        self.confidence
            .integrate(self.cycle_start_confidence, 0.01, 0.99)
            .effective
    }

    /// Effective fep_lr_boost from cycle-start + proposals so far.
    pub fn effective_lr_boost(&mut self) -> f64 {
        self.learning_rate
            .integrate(self.cycle_start_lr, 1.0, 3.0)
            .effective
    }

    /// Effective exploration_urge from cycle-start + proposals so far.
    pub fn effective_exploration(&mut self) -> f64 {
        self.exploration
            .integrate(self.cycle_start_exploration, 0.0, 1.0)
            .effective
    }

    /// Effective adaptive_threshold_scale from cycle-start + proposals so far.
    pub fn effective_threshold(&mut self) -> f64 {
        self.threshold
            .integrate(self.cycle_start_threshold, 0.5, 2.0)
            .effective
    }

    /// Compute a per-cycle feedback summary for diagnostics.
    pub fn feedback_summary(&self) -> FeedbackSummary {
        let total = self.total_proposals();
        let mut priority_counts = [0u32; 4];
        for collector in [
            &self.confidence,
            &self.learning_rate,
            &self.exploration,
            &self.threshold,
        ] {
            for ap in collector.proposals() {
                let idx = ap.priority as usize;
                if idx < 4 {
                    priority_counts[idx] += 1;
                }
            }
        }
        FeedbackSummary {
            total_proposals: total as u32,
            conflict_ratio: self.avg_conflict_ratio(),
            priority_counts,
            diversity: self.signal_diversity(),
        }
    }

    pub fn signal_diversity(&self) -> f32 {
        let total = self.total_proposals();
        if total == 0 {
            return 1.0; // No proposals = maximally diverse (vacuously)
        }
        let mut seen: Vec<&'static str> = Vec::with_capacity(total);
        for collector in [
            &self.confidence,
            &self.learning_rate,
            &self.exploration,
            &self.threshold,
        ] {
            for ap in collector.proposals() {
                if !seen.contains(&ap.source) {
                    seen.push(ap.source);
                }
            }
        }
        (seen.len() as f32 / total as f32).min(1.0)
    }
}

/// Per-cycle feedback diagnostics.
#[derive(Debug, Clone, Default)]
pub(crate) struct FeedbackSummary {
    /// Total proposals across all 4 channels
    pub total_proposals: u32,
    /// Average conflict ratio across channels
    pub conflict_ratio: f32,
    /// Proposal counts per priority tier: [Aesthetic, Cognitive, Homeostatic, Safety]
    pub priority_counts: [u32; 4],
    /// Signal diversity (unique sources / total proposals)
    pub diversity: f32,
}

impl Default for FeedbackState {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_collector_returns_base() {
        let mut collector = ProposalCollector::new();
        let result = collector.integrate(0.5, 0.0, 1.0);
        assert!((result.effective - 0.5).abs() < 1e-10);
        assert_eq!(result.n_sets, 0);
        assert_eq!(result.n_adds, 0);
        assert_eq!(result.n_scales, 0);
    }

    #[test]
    fn test_additive_proposals_averaged() {
        let mut collector = ProposalCollector::new();
        collector.propose("subsystem_a", FeedbackProposal::Add(0.1));
        collector.propose("subsystem_b", FeedbackProposal::Add(0.3));
        // Average of +0.1 and +0.3 = +0.2, applied to base 0.5 → 0.7
        let result = collector.integrate(0.5, 0.0, 1.0);
        assert!((result.effective - 0.7).abs() < 1e-10);
        assert_eq!(result.n_adds, 2);
    }

    #[test]
    fn test_multiplicative_proposals_geometric_mean() {
        let mut collector = ProposalCollector::new();
        // Two scale factors: 0.9 and 1.1 → geo mean = sqrt(0.99) ≈ 0.995
        collector.propose("subsystem_a", FeedbackProposal::Scale(0.9));
        collector.propose("subsystem_b", FeedbackProposal::Scale(1.1));
        let result = collector.integrate(1.0, 0.0, 2.0);
        let expected = (0.9f64 * 1.1).sqrt(); // sqrt(0.99)
        assert!((result.effective - expected).abs() < 1e-10);
        assert_eq!(result.n_scales, 2);
    }

    #[test]
    fn test_set_overrides_base() {
        let mut collector = ProposalCollector::new();
        collector.propose("init", FeedbackProposal::Set(0.3));
        collector.propose("override", FeedbackProposal::Set(0.7));
        // Last Set wins
        let result = collector.integrate(0.5, 0.0, 1.0);
        assert!((result.effective - 0.7).abs() < 1e-10);
        assert_eq!(result.n_sets, 2);
    }

    #[test]
    fn test_mixed_proposals() {
        let mut collector = ProposalCollector::new();
        collector.propose("base", FeedbackProposal::Set(0.5));
        collector.propose("boost_a", FeedbackProposal::Add(0.1));
        collector.propose("boost_b", FeedbackProposal::Add(0.1));
        collector.propose("scale_a", FeedbackProposal::Scale(1.2));
        // Set → 0.5, avg add → 0.1 → 0.6, geo_mean scale → 1.2 → 0.72
        let result = collector.integrate(0.0, 0.0, 1.0);
        assert!((result.effective - 0.72).abs() < 1e-10);
    }

    #[test]
    fn test_clamping_enforced() {
        let mut collector = ProposalCollector::new();
        collector.propose("huge", FeedbackProposal::Add(10.0));
        let result = collector.integrate(0.5, 0.0, 1.0);
        assert!((result.effective - 1.0).abs() < 1e-10);

        let mut collector2 = ProposalCollector::new();
        collector2.propose("negative", FeedbackProposal::Add(-10.0));
        let result2 = collector2.integrate(0.5, 0.0, 1.0);
        assert!((result2.effective - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_feedback_state_lifecycle() {
        let mut state = FeedbackState::new();

        // Begin cycle
        state.begin_cycle();
        assert_eq!(state.total_proposals(), 0);

        // Proposals during cycle
        state
            .confidence
            .propose("subsys_a", FeedbackProposal::Add(0.05));
        state
            .confidence
            .propose("subsys_b", FeedbackProposal::Scale(0.98));
        state
            .learning_rate
            .propose("fep", FeedbackProposal::Add(0.1));
        assert_eq!(state.total_proposals(), 3);

        // End cycle with current values
        state.end_cycle(0.6, 1.2, 0.4, 1.0);
        assert!(state.last_confidence_integration.is_some());
        assert!(state.last_lr_integration.is_some());

        // Next cycle clears
        state.begin_cycle();
        assert_eq!(state.total_proposals(), 0);
        // But last integration result persists
        assert!(state.last_confidence_integration.is_some());
    }

    #[test]
    fn test_lr_clamping_range() {
        let mut collector = ProposalCollector::new();
        collector.propose("huge", FeedbackProposal::Add(10.0));
        let result = collector.integrate(1.5, 1.0, 3.0);
        assert!((result.effective - 3.0).abs() < 1e-10);

        let mut collector2 = ProposalCollector::new();
        collector2.propose("negative", FeedbackProposal::Add(-10.0));
        let result2 = collector2.integrate(1.5, 1.0, 3.0);
        assert!((result2.effective - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_display_integration_result() {
        let result = IntegrationResult {
            effective: 0.7123,
            n_sets: 1,
            n_adds: 3,
            n_scales: 2,
        };
        let s = format!("{}", result);
        assert!(s.contains("0.7123"));
        assert!(s.contains("1S+3A+2M"));
    }

    #[test]
    fn test_proposal_collector_clear() {
        let mut collector = ProposalCollector::new();
        collector.propose("a", FeedbackProposal::Add(1.0));
        collector.propose("b", FeedbackProposal::Add(2.0));
        assert_eq!(collector.len(), 2);
        collector.clear();
        assert_eq!(collector.len(), 0);
        assert!(collector.proposals().is_empty());
    }

    #[test]
    fn test_dump_proposals_formats_correctly() {
        let mut collector = ProposalCollector::new();
        collector.propose("binding_strong", FeedbackProposal::Add(0.03));
        collector.propose("moral_harm", FeedbackProposal::Scale(0.85));
        collector.propose("init_reset", FeedbackProposal::Set(0.5));

        let dump = collector.dump_proposals();
        assert_eq!(dump.len(), 3);
        assert_eq!(dump[0].0, "binding_strong");
        assert!(dump[0].1.starts_with("Add("));
        assert_eq!(dump[1].0, "moral_harm");
        assert!(dump[1].1.starts_with("Scale("));
        assert_eq!(dump[2].0, "init_reset");
        assert!(dump[2].1.starts_with("Set("));
    }

    #[test]
    fn test_dump_proposals_empty_collector() {
        let collector = ProposalCollector::new();
        let dump = collector.dump_proposals();
        assert!(dump.is_empty());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // EDGE CASES: Scale proposals with non-positive values
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_zero_scale_factor_does_not_nan() {
        let mut collector = ProposalCollector::new();
        collector.propose("buggy", FeedbackProposal::Scale(0.0));
        let result = collector.integrate(1.0, 0.0, 2.0);
        assert!(result.effective.is_finite(), "Zero scale caused NaN/Inf");
        assert!(
            (result.effective - 1.0).abs() < 1e-10,
            "Zero scale should be filtered, leaving base"
        );
    }

    #[test]
    fn test_negative_scale_factor_does_not_nan() {
        let mut collector = ProposalCollector::new();
        collector.propose("buggy", FeedbackProposal::Scale(-0.5));
        collector.propose("valid", FeedbackProposal::Scale(1.1));
        let result = collector.integrate(1.0, 0.0, 2.0);
        assert!(
            result.effective.is_finite(),
            "Negative scale caused NaN/Inf"
        );
        // Only valid scale (1.1) should apply: geo_mean of [1.1] = 1.1
        assert!((result.effective - 1.1).abs() < 1e-10);
    }

    #[test]
    fn test_inf_scale_factor_filtered() {
        let mut collector = ProposalCollector::new();
        collector.propose("buggy", FeedbackProposal::Scale(f64::INFINITY));
        collector.propose("valid", FeedbackProposal::Scale(0.9));
        let result = collector.integrate(1.0, 0.0, 2.0);
        assert!(result.effective.is_finite());
        assert!((result.effective - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_all_invalid_scales_leaves_base() {
        let mut collector = ProposalCollector::new();
        collector.propose("a", FeedbackProposal::Scale(0.0));
        collector.propose("b", FeedbackProposal::Scale(-1.0));
        collector.propose("c", FeedbackProposal::Scale(f64::NAN));
        let result = collector.integrate(0.75, 0.0, 1.0);
        assert!(result.effective.is_finite());
        assert!(
            (result.effective - 0.75).abs() < 1e-10,
            "All invalid scales → base unchanged"
        );
        assert_eq!(result.n_scales, 3); // count is raw, not filtered
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // EDGE CASES: Conflict ratio
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_conflict_ratio_unanimous_positive() {
        let mut collector = ProposalCollector::new();
        collector.propose("a", FeedbackProposal::Add(0.1));
        collector.propose("b", FeedbackProposal::Add(0.2));
        collector.propose("c", FeedbackProposal::Add(0.05));
        assert!((collector.conflict_ratio() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_conflict_ratio_unanimous_negative() {
        let mut collector = ProposalCollector::new();
        collector.propose("a", FeedbackProposal::Add(-0.1));
        collector.propose("b", FeedbackProposal::Add(-0.2));
        assert!((collector.conflict_ratio() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_conflict_ratio_maximally_split() {
        let mut collector = ProposalCollector::new();
        collector.propose("a", FeedbackProposal::Add(0.1));
        collector.propose("b", FeedbackProposal::Add(-0.1));
        // 1 positive, 1 negative → minority=1, total=2 → 0.5
        assert!((collector.conflict_ratio() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_conflict_ratio_ignores_scale_and_set() {
        let mut collector = ProposalCollector::new();
        collector.propose("a", FeedbackProposal::Scale(0.9));
        collector.propose("b", FeedbackProposal::Set(0.5));
        // No Add proposals → conflict ratio = 0.0
        assert!((collector.conflict_ratio() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_conflict_ratio_single_add() {
        let mut collector = ProposalCollector::new();
        collector.propose("a", FeedbackProposal::Add(0.1));
        // < 2 adds → 0.0
        assert!((collector.conflict_ratio() - 0.0).abs() < 1e-6);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // EDGE CASES: Single proposals
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_single_add_applied_without_averaging() {
        let mut collector = ProposalCollector::new();
        collector.propose("only", FeedbackProposal::Add(0.1));
        let result = collector.integrate(0.5, 0.0, 1.0);
        // avg of [0.1] = 0.1
        assert!((result.effective - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_single_scale_applied_directly() {
        let mut collector = ProposalCollector::new();
        collector.propose("only", FeedbackProposal::Scale(2.0));
        let result = collector.integrate(0.3, 0.0, 1.0);
        // geo_mean of [2.0] = 2.0, 0.3 * 2.0 = 0.6
        assert!((result.effective - 0.6).abs() < 1e-10);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // EDGE CASES: Saturation (many proposals)
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_100_adds_averaged_correctly() {
        let mut collector = ProposalCollector::new();
        for i in 0..100 {
            // Alternate +0.01 and -0.01 → avg ≈ 0.0
            let delta = if i % 2 == 0 { 0.01 } else { -0.01 };
            collector.propose("subsys", FeedbackProposal::Add(delta));
        }
        let result = collector.integrate(0.5, 0.0, 1.0);
        assert!(
            (result.effective - 0.5).abs() < 1e-10,
            "100 balanced adds should cancel"
        );
        assert_eq!(result.n_adds, 100);
    }

    #[test]
    fn test_100_scales_geometric_mean() {
        let mut collector = ProposalCollector::new();
        // 50 × 1.02 and 50 × 0.98 → geo_mean ≈ (1.02^50 × 0.98^50)^(1/100)
        // = (1.02 × 0.98)^(50/100) = 0.9996^0.5 ≈ 0.9998
        for i in 0..100 {
            let factor = if i % 2 == 0 { 1.02 } else { 0.98 };
            collector.propose("subsys", FeedbackProposal::Scale(factor));
        }
        let result = collector.integrate(1.0, 0.0, 2.0);
        assert!(result.effective.is_finite());
        assert!(
            (result.effective - 1.0).abs() < 0.01,
            "Balanced scales should roughly cancel"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // EDGE CASES: FeedbackState dampening
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_streak_freeze_returns_base_values() {
        let mut state = FeedbackState::new();
        state.begin_cycle();
        state.snapshot_cycle_start(0.5, 1.0, 0.3, 1.0);
        state.confidence.propose("test", FeedbackProposal::Add(0.2));
        state
            .learning_rate
            .propose("test", FeedbackProposal::Add(0.5));

        // consecutive_full_dampen >= 3 → freeze to cycle-start
        let consensus = state.end_cycle_ext(0.5, 1.0, 0.3, 1.0, 3, false, 0.0);
        assert!(
            (consensus.consensus_confidence - 0.5).abs() < 1e-10,
            "Streak freeze should return base confidence"
        );
        assert!(
            (consensus.consensus_lr - 1.0).abs() < 1e-10,
            "Streak freeze should return base LR"
        );
        assert!((consensus.consensus_exploration - 0.3).abs() < 1e-10);
        assert!((consensus.consensus_threshold - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_streak_below_threshold_allows_integration() {
        let mut state = FeedbackState::new();
        state.begin_cycle();
        state.snapshot_cycle_start(0.5, 1.0, 0.3, 1.0);
        state.confidence.propose("test", FeedbackProposal::Add(0.1));

        // consecutive_full_dampen = 2 (below 3) → normal integration
        let consensus = state.end_cycle_ext(0.5, 1.0, 0.3, 1.0, 2, false, 0.0);
        // Should NOT be frozen — confidence should reflect the Add(0.1)
        assert!(
            consensus.consensus_confidence > 0.5,
            "Below-streak should allow integration"
        );
    }

    #[test]
    fn test_flow_relaxation_widens_divergence_threshold() {
        let mut state = FeedbackState::new();
        state.begin_cycle();
        state.snapshot_cycle_start(0.5, 1.0, 0.3, 1.0);
        // Large add that would trigger dampening under normal threshold (0.3)
        // but not under flow threshold (0.5)
        state.confidence.propose("test", FeedbackProposal::Add(0.2));

        // With flow: divergence_threshold = 0.5, ratio = |0.7-0.5|/0.5 = 0.4 < 0.5
        let consensus_flow = state.end_cycle_ext(0.5, 1.0, 0.3, 1.0, 0, true, 0.8);

        state.begin_cycle();
        state.snapshot_cycle_start(0.5, 1.0, 0.3, 1.0);
        state.confidence.propose("test", FeedbackProposal::Add(0.2));

        // Without flow: divergence_threshold = 0.3, ratio = 0.4 > 0.3 → dampened
        let consensus_noflow = state.end_cycle_ext(0.5, 1.0, 0.3, 1.0, 0, false, 0.0);

        // Flow should allow more divergence
        assert!(
            (consensus_flow.consensus_confidence - 0.7).abs()
                <= (consensus_noflow.consensus_confidence - 0.7).abs(),
            "Flow relaxation should allow the proposal through more easily"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // EDGE CASES: Signal diversity
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_signal_diversity_no_proposals() {
        let state = FeedbackState::new();
        assert!(
            (state.signal_diversity() - 1.0).abs() < 1e-6,
            "Empty → 1.0 (vacuously diverse)"
        );
    }

    #[test]
    fn test_signal_diversity_single_source_many_proposals() {
        let mut state = FeedbackState::new();
        state.confidence.propose("same", FeedbackProposal::Add(0.1));
        state.confidence.propose("same", FeedbackProposal::Add(0.2));
        state
            .learning_rate
            .propose("same", FeedbackProposal::Scale(1.1));
        state
            .exploration
            .propose("same", FeedbackProposal::Add(0.05));
        // 1 unique source / 4 total = 0.25
        assert!((state.signal_diversity() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_signal_diversity_all_unique() {
        let mut state = FeedbackState::new();
        state.confidence.propose("a", FeedbackProposal::Add(0.1));
        state.confidence.propose("b", FeedbackProposal::Add(0.2));
        state
            .learning_rate
            .propose("c", FeedbackProposal::Scale(1.1));
        // 3 unique / 3 total = 1.0
        assert!((state.signal_diversity() - 1.0).abs() < 1e-6);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // EDGE CASES: Dominant source
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_dominant_source_empty() {
        let state = FeedbackState::new();
        assert_eq!(state.dominant_source(), "");
        assert!((state.dominant_source_concentration() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_dominant_source_single_winner() {
        let mut state = FeedbackState::new();
        state
            .confidence
            .propose("dominant", FeedbackProposal::Add(0.1));
        state
            .confidence
            .propose("dominant", FeedbackProposal::Add(0.2));
        state
            .confidence
            .propose("minor", FeedbackProposal::Add(0.05));
        assert_eq!(state.dominant_source(), "dominant");
        // 2/3 ≈ 0.6667
        assert!((state.dominant_source_concentration() - 2.0 / 3.0).abs() < 1e-4);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // EDGE CASES: High-water mark
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_high_water_mark_tracks_maximum() {
        let mut state = FeedbackState::new();

        // Cycle 1: 3 proposals
        state.begin_cycle();
        state.snapshot_cycle_start(0.5, 1.0, 0.3, 1.0);
        state.confidence.propose("a", FeedbackProposal::Add(0.1));
        state.confidence.propose("b", FeedbackProposal::Add(0.2));
        state
            .learning_rate
            .propose("c", FeedbackProposal::Scale(1.1));
        state.end_cycle(0.5, 1.0, 0.3, 1.0);
        assert_eq!(state.feedback_signals_high_water, 3);

        // Cycle 2: 1 proposal (lower) → high-water stays at 3
        state.begin_cycle();
        state.snapshot_cycle_start(0.5, 1.0, 0.3, 1.0);
        state.confidence.propose("a", FeedbackProposal::Add(0.1));
        state.end_cycle(0.5, 1.0, 0.3, 1.0);
        assert_eq!(
            state.feedback_signals_high_water, 3,
            "High-water should not decrease"
        );

        // Cycle 3: 5 proposals → high-water updates to 5
        state.begin_cycle();
        state.snapshot_cycle_start(0.5, 1.0, 0.3, 1.0);
        for _ in 0..5 {
            state.confidence.propose("x", FeedbackProposal::Add(0.01));
        }
        state.end_cycle(0.5, 1.0, 0.3, 1.0);
        assert_eq!(state.feedback_signals_high_water, 5);
    }

    /// Verify that consensus writebacks routed through `store_consensus_for_next_cycle`
    /// + `apply_pending_consensus` produce values across two cycles.
    #[test]
    fn test_consensus_writeback_routed_through_proposals() {
        let mut state = FeedbackState::new();

        // Cycle 1: no pending consensus yet
        state.begin_cycle();
        let (conf, lr, explore, thresh) = state.apply_pending_consensus();
        assert!(conf.is_none()); // No pending on first cycle
        assert!(lr.is_none());
        assert!(explore.is_none());
        assert!(thresh.is_none());

        state.snapshot_cycle_start(0.5, 1.0, 0.3, 1.0);
        state
            .confidence
            .propose("test", FeedbackProposal::Add(0.05));
        state
            .learning_rate
            .propose("test", FeedbackProposal::Scale(1.1));
        state
            .exploration
            .propose("test", FeedbackProposal::Add(0.02));
        state
            .threshold
            .propose("test", FeedbackProposal::Scale(0.95));

        let consensus = state.end_cycle(0.55, 1.1, 0.32, 0.95);

        // Store consensus for next cycle (simulating what cycle_phase_output does)
        state.store_consensus_for_next_cycle(&consensus);

        // Cycle 2: consensus should be applied
        state.begin_cycle();
        let (conf, lr, explore, thresh) = state.apply_pending_consensus();
        // Consensus values should be pending
        assert!(conf.is_some(), "consensus confidence should be pending");
        assert!(lr.is_some(), "consensus lr should be pending");
        assert!(explore.is_some(), "consensus exploration should be pending");
        assert!(thresh.is_some(), "consensus threshold should be pending");

        // Values should be finite and within clamp ranges
        assert!(conf.unwrap().is_finite());
        assert!(lr.unwrap().is_finite());
        assert!(explore.unwrap().is_finite());
        assert!(thresh.unwrap().is_finite());
    }

    // ── Priority-aware integration tests ─────────────────────────────

    #[test]
    fn test_all_cognitive_same_as_default() {
        let mut d = ProposalCollector::new();
        d.propose("a", FeedbackProposal::Add(0.1));
        d.propose("b", FeedbackProposal::Scale(0.9));
        let mut e = ProposalCollector::new();
        e.propose_with_priority("a", FeedbackProposal::Add(0.1), Priority::Cognitive);
        e.propose_with_priority("b", FeedbackProposal::Scale(0.9), Priority::Cognitive);
        assert!(
            (d.integrate(0.5, 0.0, 1.0).effective - e.integrate(0.5, 0.0, 1.0).effective).abs()
                < 1e-10
        );
    }

    #[test]
    fn test_safety_outweighs_aesthetic_add() {
        let mut c = ProposalCollector::new();
        c.propose_with_priority("safety", FeedbackProposal::Add(0.2), Priority::Safety);
        c.propose_with_priority(
            "aesthetic",
            FeedbackProposal::Add(-0.2),
            Priority::Aesthetic,
        );
        assert!(
            c.integrate(0.5, 0.0, 1.0).effective > 0.5,
            "Safety should dominate"
        );
    }

    #[test]
    fn test_highest_priority_set_wins() {
        let mut c = ProposalCollector::new();
        c.propose_with_priority("aesthetic", FeedbackProposal::Set(0.2), Priority::Aesthetic);
        c.propose_with_priority("safety", FeedbackProposal::Set(0.8), Priority::Safety);
        assert!((c.integrate(0.5, 0.0, 1.0).effective - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_order_independence_with_mixed_priorities() {
        let mut fwd = ProposalCollector::new();
        fwd.propose_with_priority("a", FeedbackProposal::Add(0.1), Priority::Safety);
        fwd.propose_with_priority("b", FeedbackProposal::Add(-0.05), Priority::Aesthetic);
        fwd.propose_with_priority("c", FeedbackProposal::Scale(1.1), Priority::Homeostatic);
        fwd.propose_with_priority("d", FeedbackProposal::Scale(0.95), Priority::Cognitive);
        let mut rev = ProposalCollector::new();
        rev.propose_with_priority("d", FeedbackProposal::Scale(0.95), Priority::Cognitive);
        rev.propose_with_priority("c", FeedbackProposal::Scale(1.1), Priority::Homeostatic);
        rev.propose_with_priority("b", FeedbackProposal::Add(-0.05), Priority::Aesthetic);
        rev.propose_with_priority("a", FeedbackProposal::Add(0.1), Priority::Safety);
        let diff =
            (fwd.integrate(0.5, 0.0, 1.0).effective - rev.integrate(0.5, 0.0, 1.0).effective).abs();
        assert!(diff < 1e-10, "Order matters: diff={diff}");
    }

    #[test]
    fn test_feedback_summary_counts_priorities() {
        let mut state = FeedbackState::new();
        state.begin_cycle();
        state
            .confidence
            .propose_with_priority("s1", FeedbackProposal::Add(0.1), Priority::Safety);
        state
            .confidence
            .propose_with_priority("s2", FeedbackProposal::Add(0.05), Priority::Safety);
        state
            .learning_rate
            .propose("fep", FeedbackProposal::Add(0.1));
        state.exploration.propose_with_priority(
            "flow",
            FeedbackProposal::Scale(1.1),
            Priority::Aesthetic,
        );
        let summary = state.feedback_summary();
        assert_eq!(summary.total_proposals, 4);
        assert_eq!(summary.priority_counts[Priority::Safety as usize], 2);
        assert_eq!(summary.priority_counts[Priority::Cognitive as usize], 1);
        assert_eq!(summary.priority_counts[Priority::Aesthetic as usize], 1);
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        /// Generate random FeedbackProposal
        fn arb_proposal() -> impl Strategy<Value = FeedbackProposal> {
            prop_oneof![
                (-5.0..5.0f64).prop_map(FeedbackProposal::Add),
                (0.01..10.0f64).prop_map(FeedbackProposal::Scale),
                (-5.0..5.0f64).prop_map(FeedbackProposal::Set),
            ]
        }

        proptest! {
            /// LR integration always stays within [1.0, 3.0] regardless of proposals.
            #[test]
            fn prop_lr_integration_bounded(
                proposals in proptest::collection::vec(arb_proposal(), 0..50),
            ) {
                let mut collector = ProposalCollector::new();
                for p in proposals {
                    collector.propose("test", p);
                }
                let result = collector.integrate(1.0, 1.0, 3.0);
                prop_assert!(result.effective >= 1.0, "LR below min: {}", result.effective);
                prop_assert!(result.effective <= 3.0, "LR above max: {}", result.effective);
                prop_assert!(result.effective.is_finite(), "LR not finite: {}", result.effective);
            }

            /// Confidence integration always stays within [0.01, 0.99].
            #[test]
            fn prop_confidence_integration_bounded(
                proposals in proptest::collection::vec(arb_proposal(), 0..50),
            ) {
                let mut collector = ProposalCollector::new();
                for p in proposals {
                    collector.propose("test", p);
                }
                let result = collector.integrate(0.5, 0.01, 0.99);
                prop_assert!(result.effective >= 0.01, "Confidence below min: {}", result.effective);
                prop_assert!(result.effective <= 0.99, "Confidence above max: {}", result.effective);
                prop_assert!(result.effective.is_finite());
            }

            /// Exploration integration always stays within [0.0, 1.0].
            #[test]
            fn prop_exploration_integration_bounded(
                proposals in proptest::collection::vec(arb_proposal(), 0..50),
            ) {
                let mut collector = ProposalCollector::new();
                for p in proposals {
                    collector.propose("test", p);
                }
                let result = collector.integrate(0.3, 0.0, 1.0);
                prop_assert!(result.effective >= 0.0, "Exploration below min: {}", result.effective);
                prop_assert!(result.effective <= 1.0, "Exploration above max: {}", result.effective);
                prop_assert!(result.effective.is_finite());
            }

            /// Integrate is deterministic: same proposals -> same result.
            #[test]
            fn prop_integration_deterministic(
                proposals in proptest::collection::vec(arb_proposal(), 1..20),
                base in -10.0..10.0f64,
                min_val in -5.0..0.0f64,
                max_val in 1.0..10.0f64,
            ) {
                let mut c1 = ProposalCollector::new();
                let mut c2 = ProposalCollector::new();
                for p in &proposals {
                    c1.propose("test", *p);
                    c2.propose("test", *p);
                }
                let r1 = c1.integrate(base, min_val, max_val);
                let r2 = c2.integrate(base, min_val, max_val);
                prop_assert!((r1.effective - r2.effective).abs() < 1e-12,
                    "Non-deterministic: {} vs {}", r1.effective, r2.effective);
            }

            /// Conflict ratio is always in [0.0, 0.5].
            #[test]
            fn prop_conflict_ratio_bounded(
                proposals in proptest::collection::vec(arb_proposal(), 0..50),
            ) {
                let mut collector = ProposalCollector::new();
                for p in proposals {
                    collector.propose("test", p);
                }
                let ratio = collector.conflict_ratio();
                prop_assert!(ratio >= 0.0 && ratio <= 0.5, "Conflict ratio out of bounds: {}", ratio);
            }
        }
    }
}
