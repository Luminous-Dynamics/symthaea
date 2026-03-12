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
    output_flags, CognitiveSubsystem, CycleSnapshot, SubsystemOutput,
};
use super::super::thresholds;

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
    /// Completed outcomes ready for episodic recording.
    completed_outcomes: Vec<GovernanceOutcome>,
    /// Running tally of reciprocity oxytocin this cycle (for cap enforcement).
    cycle_reciprocity_oxy: f32,
    /// Current cycle number (set from snapshot each process()).
    current_cycle: u64,
    /// Last collective Phi observed from a TallyCompleted event.
    last_collective_phi: f64,
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
            last_collective_phi: 0.0,
        }
    }
}

impl GovernanceManager {
    /// Maximum stored outcomes.
    const MAX_OUTCOMES: usize = 64;

    /// Inject a governance event for processing in the next `process()` call.
    pub fn inject_event(&mut self, event: GovernanceEvent) {
        self.pending_events.push(event);
    }

    /// Inject a governance outcome for learning feedback.
    pub fn inject_outcome(&mut self, outcome: GovernanceOutcome) {
        self.outcome_history.push_back(outcome);
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

    // ── Phase 2: Learning Methods ──────────────────────────────────────

    /// Maximum prediction entries (evict oldest on overflow).
    const PREDICTION_MAX_ENTRIES: usize = 128;

    /// Prediction TTL in cycles (evict stale predictions).
    const PREDICTION_TTL_CYCLES: u64 = 1000;

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

    /// Drain completed outcomes for episodic recording.
    pub fn drain_completed(&mut self) -> Vec<GovernanceOutcome> {
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
            Some(true) => output.confidence_delta += 0.02,
            Some(false) => output.confidence_delta -= 0.03,
            None => {}
        }

        // 3. LR modulation from governance PE (high PE → boost learning)
        let lr_boost = 1.0 + (prediction_error * 0.3).min(0.5);
        output.lr_modulation = output.lr_modulation.max(1.0) * lr_boost;

        // 4. Accumulate harmonic deltas from harmonic_resonance
        // Distribute resonance across all harmonies proportionally
        let per_harmony = outcome.harmonic_resonance / 8.0;
        let sign = if outcome.passed { 1.0 } else { -0.5 };
        for delta in &mut self.harmonic_deltas {
            *delta += per_harmony * sign;
        }

        // 5. Fragile consensus → boost exploration
        // (checked via TallyCompleted events, but outcomes can also flag it)

        // 6. Compute reward signal
        let reward = outcome.value_alignment_score * if outcome.passed { 1.0 } else { -0.5 };
        self.reward_ema = self.reward_ema * 0.9 + reward * 0.1;
        self.latest_reward = Some(reward as f32);

        // 7. Store for episodic recording
        self.completed_outcomes.push(outcome.clone());
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
                if *collective_phi > 0.5 {
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
                if *delta < 0.0 {
                    // Crockett 2009: social rejection → 5-HT dip
                    self.queue_baseline("serotonin", thresholds::GOV_REPUTATION_DECLINE_SHT);
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
        37 // co-prime with 7, 11, 13, 19, 23, 29
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
                    output.arousal_delta += 0.1;
                    output.flags |= output_flags::ESCALATE_URGENCY;
                }
                GovernanceEventKind::TallyCompleted { collective_phi, .. } => {
                    tally_count += 1;
                    // Fragile consensus → boost exploration
                    if *collective_phi < 0.3 {
                        output.exploration_delta += 0.05;
                        output.flags |= output_flags::REQUEST_EXPLORATION;
                    }
                }
                GovernanceEventKind::VoteCast { voter_phi, .. } => {
                    // High-phi voters boost confidence slightly
                    if *voter_phi > 0.5 {
                        output.confidence_delta += 0.005;
                    }
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
            output.exploration_delta -= 0.1;
        }

        // Multiple tallies in one cycle → governance is active, mild LR boost
        if tally_count > 0 {
            output.lr_modulation =
                output.lr_modulation.max(1.0) * (1.0 + (tally_count as f64 * 0.02).min(0.1));
        }

        output
    }

    fn checkpoint(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(16);
        data.extend_from_slice(&self.reward_ema.to_le_bytes());
        data.extend_from_slice(&(self.outcome_history.len() as u32).to_le_bytes());
        data
    }

    fn restore(&mut self, data: &[u8]) -> Result<(), String> {
        if data.len() < 12 {
            return Err(format!(
                "GovernanceManager checkpoint too short: {} < 12",
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
