// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Governance Manager — Mycelix Governance ↔ Consciousness Bridge
//!
//! Implements [`CognitiveSubsystem`] to transform governance events from the Mycelix
//! distributed hApp into embodied cognitive signals: neuromodulatory nudges, confidence
//! deltas, exploration drives, and learning rate modulation.
//!
//! ## Design
//!
//! Governance events are injected via `inject_event()` / `inject_outcome()` from the
//! Mycelix bridge layer, then drained and processed during `process()` at interval 37
//! (co-prime with 7, 11, 13, 19, 23, 29). The manager never mutates CLS directly —
//! it produces `SubsystemOutput` proposals plus side-channel neuromod queues.
//!
//! ## Neuromodulatory Contagion (Phase 1)
//!
//! Governance events become embodied through neurochemistry:
//! - Emergency → NE baseline surge (Arnsten 2009)
//! - Reciprocity pledges → oxytocin injection (Zak 2012)
//! - Justice disputes → cortisol-proxy NE+5-HT shift (Sapolsky 2004)
//! - Aligned pass → DA phasic burst (Schultz 1997)
//! - Aligned fail → DA baseline dip (Schultz 1997)
//! - Reputation decline → 5-HT baseline dip (Crockett 2009)
//! - High collective phi → ECB baseline nudge

use std::collections::{HashMap, VecDeque};

use super::super::subsystem_trait::{
    CognitiveSubsystem, CycleSnapshot, SubsystemOutput, output_flags,
};
use super::super::thresholds;
use crate::mycelix::collective_identity::CommunityMode;
use crate::mycelix::epistemic_mesh::EpistemicMesh;

// ═══════════════════════════════════════════════════════════════════════════════
// EVENT TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// A governance event from the Mycelix network.
#[derive(Debug, Clone)]
pub struct GovernanceEvent {
    pub kind: GovernanceEventKind,
    pub proposal_id: Option<String>,
    pub timestamp_secs: u64,
}

/// The type of governance event.
#[derive(Debug, Clone)]
pub enum GovernanceEventKind {
    /// A new proposal was created in the governance system.
    ProposalCreated,
    /// A vote was cast, carrying the voter's Phi level and vote value.
    VoteCast { voter_phi: f64, vote_value: f64 },
    /// A tally completed with pass/fail result and collective Phi.
    TallyCompleted { passed: bool, collective_phi: f64 },
    /// An emergency was declared (escalated urgency).
    EmergencyDeclared,
    /// Reputation changed by a delta (positive or negative).
    ReputationChanged { delta: f64 },
    /// A reciprocity pledge was made with a given amount.
    ReciprocityPledge { amount: f64 },
    /// A justice dispute occurred; `involves_self` indicates personal involvement.
    JusticeDispute { involves_self: bool },
}

/// The outcome of a governance proposal, used for learning feedback (Phase 2).
#[derive(Debug, Clone)]
pub struct GovernanceOutcome {
    pub proposal_id: String,
    pub passed: bool,
    pub my_vote_aligned: Option<bool>,
    pub value_alignment_score: f64,
    pub harmonic_resonance: f64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// NEUROMOD QUEUE TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// A pending pharmacological injection (target transmitter, dose, half-life in cycles).
#[derive(Debug, Clone)]
pub(crate) struct PendingInjection {
    pub(crate) target: &'static str,
    pub(crate) dose: f32,
    pub(crate) half_life: u32,
}

/// A pending baseline nudge (target transmitter, delta).
#[derive(Debug, Clone)]
pub(crate) struct PendingBaseline {
    pub(crate) target: &'static str,
    pub(crate) nudge: f32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// GOVERNANCE MANAGER
// ═══════════════════════════════════════════════════════════════════════════════

/// Cognitive subsystem that translates Mycelix governance events into embodied
/// cognitive signals: neuromodulatory nudges, confidence deltas, and exploration drives.
pub struct GovernanceManager {
    /// Pending governance events (drained each `process()` call).
    pending_events: Vec<GovernanceEvent>,
    /// History of governance outcomes (capped at 64).
    outcome_history: VecDeque<GovernanceOutcome>,
    /// Exponential moving average of governance reward signal.
    reward_ema: f64,
    /// Pending pharmacological injections to apply after process().
    pending_injections: Vec<PendingInjection>,
    /// Pending baseline nudges to apply after process().
    pending_baselines: Vec<PendingBaseline>,
    /// Per-harmony accumulator for governance feedback.
    harmonic_deltas: [f64; 8],
    /// Predicted outcome alignment per proposal (for prediction error computation).
    predicted_outcomes: HashMap<String, (f64, u64)>, // (predicted_alignment, cycle_recorded)
    /// Latest reward signal for external consumption.
    latest_reward: Option<f32>,
    /// Completed outcomes ready for episodic recording, paired with governance PE.
    completed_outcomes: Vec<(GovernanceOutcome, f64)>,
    /// Running tally of reciprocity oxytocin this cycle (for cap enforcement).
    cycle_reciprocity_oxy: f32,
    /// Current cycle number (set from snapshot each process()).
    current_cycle: u64,
    /// Last collective Phi observed from a TallyCompleted event.
    last_collective_phi: f64,
    /// Latest epistemic mesh snapshot from the network (updated externally).
    epistemic_mesh: Option<EpistemicMesh>,
    /// Latest community mode from collective identity (updated externally).
    community_mode: Option<CommunityMode>,
    /// Snapshot of max |harmonic_delta| from last process() — survives take_harmonic_deltas().
    last_harmonic_delta_max: f64,
    /// Whether an external (multi-agent) mesh was set — prevents local fallback overwrite.
    external_mesh_set: bool,
    /// Accumulated confidence nudge from cross-coupling (drained in process()).
    confidence_nudge_acc: f64,
    /// Latest governance LR boost from prediction error (for telemetry).
    last_lr_boost: f64,
    /// Preferred radio tier for governance traffic (from SpectrumManager).
    #[cfg(feature = "mesh")]
    preferred_tier: Option<super::radio_dispatcher::RadioTier>,
    /// Cached financial health signals from Mycelix finance cluster.
    /// Updated periodically via `update_finance_health()`, not every cycle.
    finance_health: crate::consciousness::mycelix_bridge::FinanceHealthSignals,
}

impl Default for GovernanceManager {
    fn default() -> Self {
        Self {
            pending_events: Vec::new(),
            outcome_history: VecDeque::new(),
            reward_ema: 0.0,
            pending_injections: Vec::new(),
            pending_baselines: Vec::new(),
            harmonic_deltas: [0.0; 8],
            predicted_outcomes: HashMap::new(),
            latest_reward: None,
            completed_outcomes: Vec::new(),
            cycle_reciprocity_oxy: 0.0,
            current_cycle: 0,
            last_harmonic_delta_max: 0.0,
            last_collective_phi: 0.0,
            epistemic_mesh: None,
            community_mode: None,
            external_mesh_set: false,
            confidence_nudge_acc: 0.0,
            last_lr_boost: 1.0,
            #[cfg(feature = "mesh")]
            preferred_tier: None,
            finance_health: crate::consciousness::mycelix_bridge::FinanceHealthSignals::default(),
        }
    }
}

impl GovernanceManager {
    /// Co-prime scheduling interval (cycles).
    pub const INTERVAL: u32 = 37;

    /// Maximum stored outcomes.
    const MAX_OUTCOMES: usize = 64;

    /// Inject a governance event for processing in the next `process()` call.
    pub fn inject_event(&mut self, event: GovernanceEvent) {
        self.pending_events.push(event);
    }

    /// Inject a governance outcome for learning feedback.
    ///
    /// Clamps `value_alignment_score` to [0, 1] and `harmonic_resonance` to [-1, 1]
    /// on injection to prevent NaN/Inf propagation from malformed outcomes.
    pub fn inject_outcome(&mut self, outcome: GovernanceOutcome) {
        let mut o = outcome;
        // Guard against NaN/Inf — clamp or zero out
        o.value_alignment_score = if o.value_alignment_score.is_finite() {
            o.value_alignment_score.clamp(0.0, 1.0)
        } else {
            0.5 // safe default
        };
        o.harmonic_resonance = if o.harmonic_resonance.is_finite() {
            o.harmonic_resonance.clamp(-1.0, 1.0)
        } else {
            0.0 // safe default
        };
        self.outcome_history.push_back(o);
        while self.outcome_history.len() > Self::MAX_OUTCOMES {
            self.outcome_history.pop_front();
        }
    }

    /// Drain pending pharmacological injections. Called by the accessor layer.
    pub fn drain_injections(&mut self) -> Vec<PendingInjection> {
        std::mem::take(&mut self.pending_injections)
    }

    /// Drain pending baseline nudges. Called by the accessor layer.
    pub fn drain_baselines(&mut self) -> Vec<PendingBaseline> {
        std::mem::take(&mut self.pending_baselines)
    }

    /// Current reward EMA (for external telemetry).
    pub fn reward_ema(&self) -> f64 {
        self.reward_ema
    }

    /// Accumulate a confidence nudge from cross-coupling (drained in process()).
    /// Clamped to [-0.1, 0.1] per call, NaN-guarded.
    pub fn nudge_confidence(&mut self, delta: f64) {
        if delta.is_finite() {
            self.confidence_nudge_acc =
                (self.confidence_nudge_acc + delta.clamp(-0.1, 0.1)).clamp(-0.2, 0.2);
        }
    }

    /// Number of pending events.
    pub fn pending_event_count(&self) -> usize {
        self.pending_events.len()
    }

    /// Number of recorded outcomes.
    pub fn outcome_count(&self) -> usize {
        self.outcome_history.len()
    }

    /// Number of pending outcomes awaiting processing.
    pub fn pending_outcome_count(&self) -> usize {
        self.completed_outcomes.len()
    }

    /// Last collective Phi from a governance tally.
    pub fn last_collective_phi(&self) -> f64 {
        self.last_collective_phi
    }

    /// Update the epistemic mesh snapshot from network aggregation.
    ///
    /// External (multi-agent) meshes are marked as such and will NOT be
    /// overwritten by the local single-agent fallback in the cognitive cycle.
    /// Only call `set_local_epistemic_mesh()` for the cycle-internal fallback.
    pub fn set_epistemic_mesh(&mut self, mesh: EpistemicMesh) {
        self.epistemic_mesh = Some(mesh);
        self.external_mesh_set = true;
    }

    /// Set the local single-agent epistemic fallback. Only applied when no
    /// external mesh has been provided via `set_epistemic_mesh()`.
    pub fn set_local_epistemic_mesh(&mut self, mesh: EpistemicMesh) {
        if !self.external_mesh_set {
            self.epistemic_mesh = Some(mesh);
        }
    }

    /// Whether an external (multi-agent) epistemic mesh is active.
    pub fn has_external_mesh(&self) -> bool {
        self.external_mesh_set
    }

    /// Update the community mode from collective identity aggregation.
    pub fn set_community_mode(&mut self, mode: CommunityMode) {
        self.community_mode = Some(mode);
    }

    /// Current community mode, if known.
    pub fn community_mode(&self) -> Option<CommunityMode> {
        self.community_mode
    }

    /// Number of collective blind spots (0 if no mesh).
    pub fn blind_spot_count(&self) -> usize {
        self.epistemic_mesh
            .as_ref()
            .map(|m| m.blind_spots().len())
            .unwrap_or(0)
    }

    /// Maximum blind spot severity (0.0 if no blind spots).
    pub fn max_blind_spot_severity(&self) -> f64 {
        self.epistemic_mesh
            .as_ref()
            .map(|m| {
                m.blind_spots()
                    .iter()
                    .map(|bs| bs.severity)
                    .fold(0.0f64, f64::max)
            })
            .unwrap_or(0.0)
    }

    /// Number of agents in the epistemic mesh (0 if no mesh).
    pub fn epistemic_agent_count(&self) -> usize {
        self.epistemic_mesh
            .as_ref()
            .map(|m| m.agent_count())
            .unwrap_or(0)
    }

    /// Peek at harmonic deltas (non-consuming, for telemetry).
    pub fn harmonic_deltas(&self) -> &[f64; 8] {
        &self.harmonic_deltas
    }

    /// Max |harmonic delta| from last process() — survives take_harmonic_deltas().
    pub fn last_harmonic_delta_max(&self) -> f64 {
        self.last_harmonic_delta_max
    }

    /// LR boost from governance prediction error (1.0 = no boost).
    pub fn last_lr_boost(&self) -> f64 {
        self.last_lr_boost
    }

    /// Update cached financial health signals from Mycelix finance cluster.
    ///
    /// Called periodically (not every cycle) when fresh data arrives from
    /// the finance bridge. The `stress_index` is recomputed on update.
    pub fn update_finance_health(
        &mut self,
        signals: crate::consciousness::mycelix_bridge::FinanceHealthSignals,
    ) {
        let mut s = signals;
        s.stress_index = s.compute_stress_index();
        self.finance_health = s;
    }

    /// Current financial health signals (read-only snapshot).
    pub fn finance_health(&self) -> &crate::consciousness::mycelix_bridge::FinanceHealthSignals {
        &self.finance_health
    }

    /// Financial engagement modulation (0.0 = no effect, negative = dampening).
    ///
    /// Delegates to [`FinanceHealthSignals::engagement_modulation`].
    pub fn finance_engagement_modulation(&self) -> f32 {
        self.finance_health.engagement_modulation()
    }

    /// Set the preferred radio tier for governance traffic (from SpectrumManager).
    #[cfg(feature = "mesh")]
    pub fn set_preferred_tier(&mut self, tier: super::radio_dispatcher::RadioTier) {
        self.preferred_tier = Some(tier);
    }

    /// Get the preferred radio tier for governance traffic.
    #[cfg(feature = "mesh")]
    pub fn preferred_tier(&self) -> Option<super::radio_dispatcher::RadioTier> {
        self.preferred_tier
    }

    // ── Phase 2: Learning Methods ──────────────────────────────────────

    /// Maximum prediction entries (evict oldest on overflow).
    const PREDICTION_MAX_ENTRIES: usize = 128;

    /// Prediction TTL in cycles (evict stale predictions).
    /// 5000 cycles ≈ 100s at 50Hz — ample time for governance tallies.
    const PREDICTION_TTL_CYCLES: u64 = 5000;

    /// Record a predicted outcome alignment for a proposal.
    /// Called when casting a vote, recording what we expect to happen.
    pub fn predict_outcome(&mut self, proposal_id: String, predicted_alignment: f64) {
        // Evict stale entries
        self.predicted_outcomes.retain(|_, (_, cycle)| {
            self.current_cycle.saturating_sub(*cycle) < Self::PREDICTION_TTL_CYCLES
        });
        // Hard cap
        while self.predicted_outcomes.len() >= Self::PREDICTION_MAX_ENTRIES {
            // Remove the oldest entry
            if let Some(oldest_key) = self
                .predicted_outcomes
                .iter()
                .min_by_key(|(_, (_, cycle))| *cycle)
                .map(|(k, _)| k.clone())
            {
                self.predicted_outcomes.remove(&oldest_key);
            } else {
                break;
            }
        }
        self.predicted_outcomes
            .insert(proposal_id, (predicted_alignment, self.current_cycle));
    }

    /// Take the latest reward signal (consumed once).
    pub fn take_latest_reward(&mut self) -> Option<f32> {
        self.latest_reward.take()
    }

    /// Drain completed outcomes for episodic recording, with their governance prediction errors.
    pub fn drain_completed(&mut self) -> Vec<(GovernanceOutcome, f64)> {
        std::mem::take(&mut self.completed_outcomes)
    }

    /// Take harmonic deltas accumulated from governance outcomes.
    pub fn take_harmonic_deltas(&mut self) -> [f64; 8] {
        let deltas = self.harmonic_deltas;
        self.harmonic_deltas = [0.0; 8];
        deltas
    }

    /// Process a governance outcome for learning signals (Phase 2).
    fn process_outcome_learning(
        &mut self,
        outcome: &GovernanceOutcome,
        output: &mut SubsystemOutput,
    ) {
        // 1. Compute prediction error
        let predicted = self
            .predicted_outcomes
            .remove(&outcome.proposal_id)
            .map(|(pred, _)| pred)
            .unwrap_or(0.5); // default: uncertain
        let actual = outcome.value_alignment_score;
        let prediction_error = (predicted - actual).abs();

        // 2. Confidence delta from alignment
        match outcome.my_vote_aligned {
            Some(true) => output.confidence_delta += thresholds::GOV_ALIGNED_VOTE_CONFIDENCE,
            Some(false) => output.confidence_delta += thresholds::GOV_MISALIGNED_VOTE_CONFIDENCE,
            None => {}
        }

        // 3. LR modulation from governance PE (high PE → boost learning)
        let lr_boost = 1.0
            + (prediction_error * thresholds::GOV_PE_LR_SCALE).min(thresholds::GOV_PE_LR_MAX_BOOST);
        output.lr_modulation = output.lr_modulation.max(1.0) * lr_boost;
        self.last_lr_boost = lr_boost;

        // 4. Accumulate harmonic deltas from harmonic_resonance
        // Distribute resonance across all harmonies proportionally
        let per_harmony = outcome.harmonic_resonance / 8.0;
        let sign = if outcome.passed {
            1.0
        } else {
            thresholds::GOV_FAILED_OUTCOME_SIGN
        };
        for delta in &mut self.harmonic_deltas {
            *delta += per_harmony * sign;
        }

        // 5. Fragile consensus → boost exploration
        // (checked via TallyCompleted events, but outcomes can also flag it)

        // 6. Compute reward signal
        let reward = outcome.value_alignment_score
            * if outcome.passed {
                1.0
            } else {
                thresholds::GOV_FAILED_OUTCOME_SIGN
            };
        self.reward_ema = self.reward_ema * thresholds::GOV_REWARD_EMA_DECAY
            + reward * (1.0 - thresholds::GOV_REWARD_EMA_DECAY);
        self.latest_reward = Some(reward as f32);

        // 7. Store for episodic recording with governance PE
        self.completed_outcomes
            .push((outcome.clone(), prediction_error));
    }

    /// Queue a neuromod injection with floor check.
    fn queue_injection(&mut self, target: &'static str, dose: f32, half_life: u32) {
        if dose.abs() >= thresholds::GOV_NEUROMOD_FLOOR {
            self.pending_injections.push(PendingInjection {
                target,
                dose,
                half_life,
            });
        }
    }

    /// Queue a baseline nudge with floor check.
    fn queue_baseline(&mut self, target: &'static str, nudge: f32) {
        if nudge.abs() >= thresholds::GOV_NEUROMOD_FLOOR {
            self.pending_baselines
                .push(PendingBaseline { target, nudge });
        }
    }

    /// Process neuromodulatory effects of a single event (Phase 1).
    fn process_event_neuromod(&mut self, event: &GovernanceEvent) {
        match &event.kind {
            GovernanceEventKind::EmergencyDeclared => {
                // Arnsten 2009: acute stress → NE surge for vigilance
                self.queue_baseline("noradrenaline", thresholds::GOV_EMERGENCY_NE_NUDGE);
            }

            GovernanceEventKind::ReciprocityPledge { .. } => {
                // Zak 2012: reciprocity → oxytocin for social bonding
                let remaining_budget =
                    thresholds::GOV_RECIPROCITY_OXY_CAP - self.cycle_reciprocity_oxy;
                if remaining_budget > thresholds::GOV_NEUROMOD_FLOOR {
                    let dose = thresholds::GOV_RECIPROCITY_OXY_DOSE.min(remaining_budget);
                    self.cycle_reciprocity_oxy += dose;
                    self.queue_injection(
                        "oxytocin",
                        dose,
                        thresholds::GOV_RECIPROCITY_OXY_HALFLIFE,
                    );
                }
            }

            GovernanceEventKind::JusticeDispute { involves_self } => {
                if *involves_self {
                    // Sapolsky 2004: personal conflict → cortisol-proxy (NE↑, 5-HT↓)
                    self.queue_baseline("noradrenaline", thresholds::GOV_DISPUTE_NE_NUDGE);
                    self.queue_baseline("serotonin", thresholds::GOV_DISPUTE_SHT_NUDGE);
                }
            }

            GovernanceEventKind::TallyCompleted {
                passed,
                collective_phi,
            } => {
                // Track last collective phi for telemetry and consciousness coupling
                self.last_collective_phi = *collective_phi;

                // High collective phi → ECB baseline nudge (group coherence)
                if *collective_phi > thresholds::GOV_COLLECTIVE_PHI_HIGH {
                    self.queue_baseline("endocannabinoid", thresholds::GOV_COLLECTIVE_PHI_ECB);
                }

                // DA response depends on alignment (computed in outcomes, but
                // we can still give a baseline signal for the tally event itself)
                if *passed {
                    // Schultz 1997: positive outcome → phasic DA
                    self.queue_injection(
                        "dopamine",
                        thresholds::GOV_ALIGNED_PASS_DA_DOSE,
                        thresholds::GOV_ALIGNED_PASS_DA_HALFLIFE,
                    );
                }
            }

            GovernanceEventKind::ReputationChanged { delta } => {
                if *delta < -0.01 {
                    // Crockett 2009: social rejection → 5-HT dip
                    self.queue_baseline("serotonin", thresholds::GOV_REPUTATION_DECLINE_SHT);
                } else if *delta > 0.01 {
                    // Crockett 2009: social approval → 5-HT boost (symmetric)
                    self.queue_baseline("serotonin", thresholds::GOV_REPUTATION_GAIN_SHT);
                }
            }

            GovernanceEventKind::VoteCast { .. } | GovernanceEventKind::ProposalCreated => {
                // No neuromod effect for passive observation events
            }
        }
    }

    /// Process outcome-based neuromod effects (alignment-dependent DA).
    fn process_outcome_neuromod(&mut self, outcome: &GovernanceOutcome) {
        match outcome.my_vote_aligned {
            Some(true) if outcome.passed => {
                // Schultz 1997: prediction confirmed + reward → strong DA burst
                // (already got a base DA from TallyCompleted, this adds alignment bonus)
                self.queue_injection(
                    "dopamine",
                    thresholds::GOV_ALIGNED_PASS_DA_DOSE * 0.5,
                    thresholds::GOV_ALIGNED_PASS_DA_HALFLIFE,
                );
            }
            Some(true) if !outcome.passed => {
                // Schultz 1997: expected pass but failed → DA dip (negative PE)
                self.queue_baseline("dopamine", thresholds::GOV_ALIGNED_FAIL_DA_NUDGE);
            }
            Some(false) if outcome.passed => {
                // Voted against, it passed anyway → mild negative signal
                self.queue_baseline("dopamine", thresholds::GOV_ALIGNED_FAIL_DA_NUDGE * 0.5);
            }
            _ => {} // No vote or no alignment info → no DA modulation
        }
    }
}

impl CognitiveSubsystem for GovernanceManager {
    fn name(&self) -> &'static str {
        "governance_manager"
    }

    fn interval(&self) -> u32 {
        Self::INTERVAL
    }

    fn process(&mut self, snapshot: &CycleSnapshot) -> SubsystemOutput {
        let mut output = SubsystemOutput::NEUTRAL;

        // Update cycle counter from snapshot
        self.current_cycle = snapshot.cycle_number;

        // Reset per-cycle accumulators
        self.cycle_reciprocity_oxy = 0.0;

        // Drain and process events
        let events: Vec<GovernanceEvent> = std::mem::take(&mut self.pending_events);

        // Drain outcomes for processing (clear the history)
        let outcomes: Vec<GovernanceOutcome> = self.outcome_history.drain(..).collect();

        if events.is_empty() && outcomes.is_empty() {
            // Drain cross-coupling nudge even when idle
            if self.confidence_nudge_acc.abs() > 1e-10 {
                output.confidence_delta += self.confidence_nudge_acc;
                self.confidence_nudge_acc = 0.0;
            }
            return output;
        }

        let mut has_emergency = false;
        let mut tally_count = 0u32;

        for event in &events {
            // Phase 1: neuromod contagion
            self.process_event_neuromod(event);

            // Subsystem output signals
            match &event.kind {
                GovernanceEventKind::EmergencyDeclared => {
                    has_emergency = true;
                    output.arousal_delta += thresholds::GOV_EMERGENCY_AROUSAL;
                    output.flags |= output_flags::ESCALATE_URGENCY;
                }
                GovernanceEventKind::TallyCompleted { collective_phi, .. } => {
                    tally_count += 1;
                    // Fragile consensus → boost exploration
                    if *collective_phi < thresholds::GOV_FRAGILE_CONSENSUS_PHI {
                        output.exploration_delta += thresholds::GOV_FRAGILE_CONSENSUS_EXPLORE;
                        output.flags |= output_flags::REQUEST_EXPLORATION;
                    }
                }
                // High-phi voters boost confidence slightly
                GovernanceEventKind::VoteCast { voter_phi, .. }
                    if *voter_phi > thresholds::GOV_COLLECTIVE_PHI_HIGH =>
                {
                    output.confidence_delta += thresholds::GOV_HIGH_PHI_VOTER_CONFIDENCE;
                }
                _ => {}
            }
        }

        // Phase 1 + 2: Process outcomes for neuromod + learning
        for outcome in &outcomes {
            self.process_outcome_neuromod(outcome);
            self.process_outcome_learning(outcome, &mut output);
        }

        // Emergency raises arousal, suppresses exploration
        if has_emergency {
            output.exploration_delta -= thresholds::GOV_EMERGENCY_EXPLORE_SUPPRESS;
        }

        // Multiple tallies in one cycle → governance is active, mild LR boost
        if tally_count > 0 {
            output.lr_modulation = output.lr_modulation.max(1.0)
                * (1.0
                    + (tally_count as f64 * thresholds::GOV_TALLY_LR_SCALE)
                        .min(thresholds::GOV_TALLY_LR_MAX_BOOST));
        }

        // ── Epistemic mesh: blind spots → exploration boost ─────────────
        if let Some(ref mesh) = self.epistemic_mesh {
            if mesh.has_blind_spots() {
                // Collective ignorance drives exploration (Friston 2010)
                let max_severity = mesh
                    .blind_spots()
                    .iter()
                    .map(|bs| bs.severity)
                    .fold(0.0f64, f64::max);
                output.exploration_delta += max_severity * thresholds::GOV_BLIND_SPOT_EXPLORE_SCALE;
                output.flags |= output_flags::REQUEST_EXPLORATION;
            }
        }

        // ── Community mode: influence harmony weight deltas ─────────────
        if let Some(mode) = self.community_mode {
            // Community mode gently biases harmonic emphasis (±0.005 per cycle)
            let bias = thresholds::GOV_COMMUNITY_HARMONIC_BIAS;
            match mode {
                CommunityMode::Exploratory => {
                    self.harmonic_deltas[3] += bias; // InfinitePlay
                    self.harmonic_deltas[4] += bias; // UniversalInterconnectedness
                }
                CommunityMode::Protective => {
                    self.harmonic_deltas[1] += bias; // PanSentientFlourishing
                    self.harmonic_deltas[5] += bias; // SacredReciprocity
                }
                CommunityMode::Creative => {
                    self.harmonic_deltas[3] += bias; // InfinitePlay
                    self.harmonic_deltas[6] += bias; // EvolutionaryProgression
                }
                CommunityMode::Reflective => {
                    self.harmonic_deltas[7] += bias; // SacredStillness
                    self.harmonic_deltas[2] += bias; // IntegralWisdom
                }
            }
        }

        // Snapshot max |delta| for telemetry (survives take_harmonic_deltas)
        self.last_harmonic_delta_max = self
            .harmonic_deltas
            .iter()
            .map(|d| d.abs())
            .fold(0.0f64, f64::max);

        // Drain cross-coupling nudge
        if self.confidence_nudge_acc.abs() > 1e-10 {
            output.confidence_delta += self.confidence_nudge_acc;
            self.confidence_nudge_acc = 0.0;
        }

        output
    }

    fn checkpoint(&self) -> Vec<u8> {
        // Layout: [reward_ema: f64 = 8] = 8 bytes total.
        // Note: outcome_history is transient — it rebuilds from live governance events.
        let mut data = Vec::with_capacity(8);
        data.extend_from_slice(&self.reward_ema.to_le_bytes());
        data
    }

    fn restore(&mut self, data: &[u8]) -> Result<(), String> {
        if data.len() < 8 {
            return Err(format!(
                "GovernanceManager checkpoint too short: {} < 8",
                data.len()
            ));
        }
        self.reward_ema = f64::from_le_bytes(
            data[0..8]
                .try_into()
                .map_err(|_| "GovernanceManager: corrupt checkpoint bytes [0..8]".to_string())?,
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
    use crate::cognitive_loop::subsystem_trait::CycleSnapshot;

    fn default_snapshot() -> CycleSnapshot {
        CycleSnapshot::default()
    }

    fn make_event(kind: GovernanceEventKind) -> GovernanceEvent {
        GovernanceEvent {
            kind,
            proposal_id: None,
            timestamp_secs: 0,
        }
    }

    #[test]
    fn test_neutral_without_events() {
        let mut mgr = GovernanceManager::default();
        let output = mgr.process(&default_snapshot());
        assert!(
            output.confidence_delta.abs() < 1e-10,
            "No events should produce neutral output"
        );
        assert!(
            (output.lr_modulation - 1.0).abs() < 1e-10 || output.lr_modulation == 0.0,
            "No events should produce neutral LR"
        );
    }

    #[test]
    fn test_interval_is_37() {
        let mgr = GovernanceManager::default();
        assert_eq!(mgr.interval(), 37);
        // Verify co-prime with existing intervals
        for interval in [7u32, 11, 13, 19, 23, 29] {
            assert_ne!(37 % interval, 0, "37 should be co-prime with {}", interval);
            assert_ne!(interval % 37, 0, "{} should be co-prime with 37", interval);
        }
    }

    #[test]
    fn test_name() {
        let mgr = GovernanceManager::default();
        assert_eq!(mgr.name(), "governance_manager");
    }

    #[test]
    fn test_emergency_raises_ne() {
        let mut mgr = GovernanceManager::default();
        mgr.inject_event(make_event(GovernanceEventKind::EmergencyDeclared));
        let output = mgr.process(&default_snapshot());

        // Should have NE baseline nudge queued
        let baselines = mgr.drain_baselines();
        assert!(
            baselines.iter().any(|b| b.target == "noradrenaline"
                && (b.nudge - thresholds::GOV_EMERGENCY_NE_NUDGE).abs() < 1e-6),
            "Emergency should queue NE baseline nudge"
        );
        // Should escalate urgency
        assert!(
            output.flags & output_flags::ESCALATE_URGENCY != 0,
            "Emergency should escalate urgency"
        );
        assert!(output.arousal_delta > 0.0, "Emergency should raise arousal");
    }

    #[test]
    fn test_reciprocity_caps_oxytocin() {
        let mut mgr = GovernanceManager::default();
        // Inject 10 reciprocity pledges — should cap at GOV_RECIPROCITY_OXY_CAP
        for _ in 0..10 {
            mgr.inject_event(make_event(GovernanceEventKind::ReciprocityPledge {
                amount: 1.0,
            }));
        }
        mgr.process(&default_snapshot());

        let injections = mgr.drain_injections();
        let total_oxy: f32 = injections
            .iter()
            .filter(|i| i.target == "oxytocin")
            .map(|i| i.dose)
            .sum();
        assert!(
            (total_oxy - thresholds::GOV_RECIPROCITY_OXY_CAP).abs() < 1e-4,
            "Oxytocin should be capped at {}: got {}",
            thresholds::GOV_RECIPROCITY_OXY_CAP,
            total_oxy
        );
    }

    #[test]
    fn test_aligned_pass_da_burst() {
        let mut mgr = GovernanceManager::default();
        mgr.inject_event(make_event(GovernanceEventKind::TallyCompleted {
            passed: true,
            collective_phi: 0.8,
        }));
        mgr.inject_outcome(GovernanceOutcome {
            proposal_id: "p1".into(),
            passed: true,
            my_vote_aligned: Some(true),
            value_alignment_score: 0.9,
            harmonic_resonance: 0.7,
        });
        let output = mgr.process(&default_snapshot());

        let injections = mgr.drain_injections();
        let da_injections: Vec<_> = injections
            .iter()
            .filter(|i| i.target == "dopamine")
            .collect();
        assert!(
            !da_injections.is_empty(),
            "Aligned pass should produce DA injection"
        );
        assert!(
            output.confidence_delta > 0.0,
            "Aligned pass should boost confidence"
        );
    }

    #[test]
    fn test_misalignment_decreases_confidence() {
        let mut mgr = GovernanceManager::default();
        mgr.inject_outcome(GovernanceOutcome {
            proposal_id: "p2".into(),
            passed: true,
            my_vote_aligned: Some(false),
            value_alignment_score: 0.3,
            harmonic_resonance: 0.2,
        });
        let output = mgr.process(&default_snapshot());
        assert!(
            output.confidence_delta < 0.0,
            "Misalignment should decrease confidence: {}",
            output.confidence_delta
        );
    }

    #[test]
    fn test_floor_check_skips_tiny() {
        let mut mgr = GovernanceManager::default();
        // Reputation change with delta 0 should produce no nudges
        mgr.inject_event(make_event(GovernanceEventKind::ReputationChanged {
            delta: 0.0,
        }));
        mgr.process(&default_snapshot());

        let baselines = mgr.drain_baselines();
        // ReputationChanged with delta=0 (not negative) produces no nudge at all
        assert!(
            baselines.is_empty(),
            "Non-negative reputation change should produce no baseline nudges"
        );
    }

    #[test]
    fn test_fragile_consensus_boosts_exploration() {
        let mut mgr = GovernanceManager::default();
        mgr.inject_event(make_event(GovernanceEventKind::TallyCompleted {
            passed: true,
            collective_phi: 0.2, // Low collective phi = fragile
        }));
        let output = mgr.process(&default_snapshot());

        assert!(
            output.exploration_delta > 0.0,
            "Fragile consensus should boost exploration"
        );
        assert!(
            output.flags & output_flags::REQUEST_EXPLORATION != 0,
            "Fragile consensus should request exploration"
        );
    }

    #[test]
    fn test_dispute_ne_and_sht() {
        let mut mgr = GovernanceManager::default();
        mgr.inject_event(make_event(GovernanceEventKind::JusticeDispute {
            involves_self: true,
        }));
        mgr.process(&default_snapshot());

        let baselines = mgr.drain_baselines();
        assert!(
            baselines
                .iter()
                .any(|b| b.target == "noradrenaline" && b.nudge > 0.0),
            "Self-involved dispute should raise NE"
        );
        assert!(
            baselines
                .iter()
                .any(|b| b.target == "serotonin" && b.nudge < 0.0),
            "Self-involved dispute should lower 5-HT"
        );
    }

    #[test]
    fn test_outcome_history_capped() {
        let mut mgr = GovernanceManager::default();
        for i in 0..100 {
            mgr.inject_outcome(GovernanceOutcome {
                proposal_id: format!("p{}", i),
                passed: true,
                my_vote_aligned: Some(true),
                value_alignment_score: 0.5,
                harmonic_resonance: 0.5,
            });
        }
        assert_eq!(mgr.outcome_count(), GovernanceManager::MAX_OUTCOMES);
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let mut mgr = GovernanceManager::default();
        mgr.reward_ema = 0.42;
        let data = mgr.checkpoint();
        let mut mgr2 = GovernanceManager::default();
        mgr2.restore(&data).unwrap();
        assert!((mgr2.reward_ema - 0.42).abs() < 1e-10);
    }

    #[test]
    fn test_restore_rejects_short_data() {
        let mut mgr = GovernanceManager::default();
        assert!(mgr.restore(&[0u8; 4]).is_err());
    }

    #[test]
    fn test_non_self_dispute_no_neuromod() {
        let mut mgr = GovernanceManager::default();
        mgr.inject_event(make_event(GovernanceEventKind::JusticeDispute {
            involves_self: false,
        }));
        mgr.process(&default_snapshot());

        let baselines = mgr.drain_baselines();
        assert!(
            baselines.is_empty(),
            "Non-self dispute should produce no neuromod effect"
        );
    }

    #[test]
    fn test_epistemic_blind_spots_boost_exploration() {
        use crate::mycelix::epistemic_mesh::{EpistemicMesh, EpistemicSummary};
        use crate::mycelix::gis::IgnoranceType;

        let mut mgr = GovernanceManager::default();

        // Create a mesh with a severe blind spot
        let summaries = vec![
            EpistemicSummary {
                agent_id: "a1".into(),
                dominant_ignorance: IgnoranceType::KnownUnknown,
                domain_expertise: vec![],
                blind_spots: vec!["quantum".into()],
            },
            EpistemicSummary {
                agent_id: "a2".into(),
                dominant_ignorance: IgnoranceType::KnownUnknown,
                domain_expertise: vec![],
                blind_spots: vec!["quantum".into()],
            },
        ];
        mgr.set_epistemic_mesh(EpistemicMesh::new(summaries));

        // Need at least one event to avoid early return
        mgr.inject_event(make_event(GovernanceEventKind::ProposalCreated));
        let output = mgr.process(&default_snapshot());

        assert!(
            output.exploration_delta > 0.0,
            "Blind spots should boost exploration: {}",
            output.exploration_delta
        );
        assert!(output.flags & output_flags::REQUEST_EXPLORATION != 0);
    }

    #[test]
    fn test_community_mode_biases_harmonics() {
        let mut mgr = GovernanceManager::default();
        mgr.set_community_mode(CommunityMode::Protective);

        // Need at least one event
        mgr.inject_event(make_event(GovernanceEventKind::ProposalCreated));
        mgr.process(&default_snapshot());

        let deltas = mgr.take_harmonic_deltas();
        // Protective biases PanSentientFlourishing (1) and SacredReciprocity (5)
        assert!(
            deltas[1] > 0.0,
            "Protective mode should bias PanSentientFlourishing: {}",
            deltas[1]
        );
        assert!(
            deltas[5] > 0.0,
            "Protective mode should bias SacredReciprocity: {}",
            deltas[5]
        );
        // Other harmonies should be unaffected
        assert!(
            deltas[3].abs() < 1e-10,
            "InfinitePlay should not be biased in Protective mode"
        );
    }

    #[test]
    fn test_completed_outcomes_carry_prediction_error() {
        let mut mgr = GovernanceManager::default();
        // Predict outcome alignment
        mgr.predict_outcome("p1".into(), 0.9);
        // Actual outcome has very different alignment → high PE
        mgr.inject_outcome(GovernanceOutcome {
            proposal_id: "p1".into(),
            passed: true,
            my_vote_aligned: Some(true),
            value_alignment_score: 0.1,
            harmonic_resonance: 0.5,
        });
        mgr.process(&default_snapshot());

        let completed = mgr.drain_completed();
        assert_eq!(completed.len(), 1);
        let (_outcome, pe) = &completed[0];
        assert!(
            *pe > 0.5,
            "High prediction error expected: predicted=0.9, actual=0.1, got PE={}",
            pe
        );
    }

    #[test]
    fn test_high_collective_phi_ecb() {
        let mut mgr = GovernanceManager::default();
        mgr.inject_event(make_event(GovernanceEventKind::TallyCompleted {
            passed: false,
            collective_phi: 0.8,
        }));
        mgr.process(&default_snapshot());

        let baselines = mgr.drain_baselines();
        assert!(
            baselines
                .iter()
                .any(|b| b.target == "endocannabinoid" && b.nudge > 0.0),
            "High collective phi should nudge ECB baseline"
        );
    }
}
