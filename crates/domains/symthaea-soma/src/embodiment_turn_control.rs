// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Causal, one-decision-per-event gate for reciprocal conversational control.
//!
//! This module does not detect turns. It qualifies candidate WAIT/RESPOND/
//! CONTINUE/YIELD decisions emitted by a detector before they are converted
//! into presentation events. Ambiguous candidates abstain; they are never
//! silently converted into WAIT or RESPOND.
//!
//! Each candidate event ID is consumed once, including an abstained candidate.
//! This prevents repeated late fallbacks from converting one pause into many
//! apparently independent decisions.

use crate::embodiment_event::TurnState;
use std::collections::HashSet;
use std::str::FromStr;

pub const MAX_TURN_DECISIONS_PER_SESSION: usize = 65_536;

#[derive(Debug, Clone, PartialEq)]
pub struct TurnCandidate {
    event_id: String,
    sequence: u64,
    clock_domain_id: String,
    observed_at_monotonic_ns: String,
    expires_at_monotonic_ns: String,
    action: TurnState,
    confidence: f32,
    calibration_id: String,
}

impl TurnCandidate {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        event_id: impl Into<String>,
        sequence: u64,
        clock_domain_id: impl Into<String>,
        observed_at_monotonic_ns: impl Into<String>,
        expires_at_monotonic_ns: impl Into<String>,
        action: TurnState,
        confidence: f32,
        calibration_id: impl Into<String>,
    ) -> Result<Self, TurnControlError> {
        let event_id = event_id.into();
        let clock_domain_id = clock_domain_id.into();
        let observed_at_monotonic_ns = observed_at_monotonic_ns.into();
        let expires_at_monotonic_ns = expires_at_monotonic_ns.into();
        let calibration_id = calibration_id.into();
        validate_identifier(&event_id)?;
        validate_identifier(&clock_domain_id)?;
        validate_identifier(&calibration_id)?;
        parse_monotonic_ns(&observed_at_monotonic_ns)?;
        parse_monotonic_ns(&expires_at_monotonic_ns)?;
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(TurnControlError::InvalidConfidence);
        }
        Ok(Self {
            event_id,
            sequence,
            clock_domain_id,
            observed_at_monotonic_ns,
            expires_at_monotonic_ns,
            action,
            confidence,
            calibration_id,
        })
    }

    pub fn event_id(&self) -> &str { &self.event_id }
    pub fn sequence(&self) -> u64 { self.sequence }
    pub fn action(&self) -> TurnState { self.action }
    pub fn confidence(&self) -> f32 { self.confidence }
    pub fn calibration_id(&self) -> &str { &self.calibration_id }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TurnControlThresholds {
    wait: f32,
    respond: f32,
    continue_floor: f32,
    yield_floor: f32,
}

impl TurnControlThresholds {
    pub fn new(
        wait: f32,
        respond: f32,
        continue_floor: f32,
        yield_floor: f32,
    ) -> Result<Self, TurnControlError> {
        let values = [wait, respond, continue_floor, yield_floor];
        if values.iter().any(|value| !value.is_finite() || !(0.0..=1.0).contains(value)) {
            return Err(TurnControlError::InvalidThreshold);
        }
        // Taking the floor from the partner is the highest-cost false action.
        if yield_floor < respond || respond < continue_floor {
            return Err(TurnControlError::InvalidThresholdOrdering);
        }
        Ok(Self { wait, respond, continue_floor, yield_floor })
    }

    pub fn threshold_for(self, action: TurnState) -> f32 {
        match action {
            TurnState::Wait => self.wait,
            TurnState::Respond => self.respond,
            TurnState::Continue => self.continue_floor,
            TurnState::Yield => self.yield_floor,
        }
    }
}

impl Default for TurnControlThresholds {
    fn default() -> Self {
        Self {
            wait: 0.55,
            respond: 0.82,
            continue_floor: 0.76,
            yield_floor: 0.94,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TurnDecision {
    event_id: String,
    sequence: u64,
    action: TurnState,
    confidence: f32,
    required_confidence: f32,
    calibration_id: String,
    decided_at_monotonic_ns: String,
    decision_latency_ns: u64,
}

impl TurnDecision {
    pub fn event_id(&self) -> &str { &self.event_id }
    pub fn sequence(&self) -> u64 { self.sequence }
    pub fn action(&self) -> TurnState { self.action }
    pub fn confidence(&self) -> f32 { self.confidence }
    pub fn required_confidence(&self) -> f32 { self.required_confidence }
    pub fn calibration_id(&self) -> &str { &self.calibration_id }
    pub fn decided_at_monotonic_ns(&self) -> &str { &self.decided_at_monotonic_ns }
    pub fn decision_latency_ns(&self) -> u64 { self.decision_latency_ns }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TurnAbstention {
    event_id: String,
    sequence: u64,
    proposed_action: TurnState,
    confidence: f32,
    required_confidence: f32,
    calibration_id: String,
}

impl TurnAbstention {
    pub fn event_id(&self) -> &str { &self.event_id }
    pub fn sequence(&self) -> u64 { self.sequence }
    pub fn proposed_action(&self) -> TurnState { self.proposed_action }
    pub fn confidence(&self) -> f32 { self.confidence }
    pub fn required_confidence(&self) -> f32 { self.required_confidence }
    pub fn calibration_id(&self) -> &str { &self.calibration_id }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TurnControlOutcome {
    Emitted(TurnDecision),
    Abstained(TurnAbstention),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnControlError {
    InvalidIdentifier,
    InvalidTimestamp,
    InvalidConfidence,
    InvalidThreshold,
    InvalidThresholdOrdering,
    WrongClockDomain,
    FutureObservation,
    CandidateExpired,
    ClockRegression,
    SequenceRegression,
    SequenceGap,
    DuplicateEvent,
    CapacityReached,
    InvalidTransition { from: Option<TurnState>, to: TurnState },
}

/// Session-scoped gate. A candidate is consumed whether emitted or abstained,
/// ensuring the same event cannot be retried with a later fallback timestamp.
#[derive(Debug, Clone)]
pub struct TurnDecisionGate {
    clock_domain_id: String,
    thresholds: TurnControlThresholds,
    max_decisions: usize,
    consumed_event_ids: HashSet<String>,
    last_sequence: Option<u64>,
    last_candidate_observed_at: Option<u64>,
    last_decision_clock: Option<u64>,
    current_action: Option<TurnState>,
}

impl TurnDecisionGate {
    pub fn new(
        clock_domain_id: impl Into<String>,
        thresholds: TurnControlThresholds,
    ) -> Result<Self, TurnControlError> {
        Self::with_capacity(clock_domain_id, thresholds, MAX_TURN_DECISIONS_PER_SESSION)
    }

    pub fn with_capacity(
        clock_domain_id: impl Into<String>,
        thresholds: TurnControlThresholds,
        max_decisions: usize,
    ) -> Result<Self, TurnControlError> {
        let clock_domain_id = clock_domain_id.into();
        validate_identifier(&clock_domain_id)?;
        if max_decisions == 0 || max_decisions > MAX_TURN_DECISIONS_PER_SESSION {
            return Err(TurnControlError::CapacityReached);
        }
        Ok(Self {
            clock_domain_id,
            thresholds,
            max_decisions,
            consumed_event_ids: HashSet::with_capacity(max_decisions.min(1024)),
            last_sequence: None,
            last_candidate_observed_at: None,
            last_decision_clock: None,
            current_action: None,
        })
    }

    pub fn current_action(&self) -> Option<TurnState> { self.current_action }
    pub fn consumed_count(&self) -> usize { self.consumed_event_ids.len() }

    /// Evaluate one detector candidate using a caller-supplied timestamp from
    /// the gate's exact monotonic clock domain. No wall/game clock is accepted.
    pub fn evaluate(
        &mut self,
        candidate: TurnCandidate,
        now_clock_domain_id: &str,
        now_monotonic_ns: &str,
    ) -> Result<TurnControlOutcome, TurnControlError> {
        validate_identifier(now_clock_domain_id)?;
        let now = parse_monotonic_ns(now_monotonic_ns)?;
        if now_clock_domain_id != self.clock_domain_id
            || candidate.clock_domain_id != self.clock_domain_id
        {
            return Err(TurnControlError::WrongClockDomain);
        }
        if self.consumed_event_ids.contains(&candidate.event_id) {
            return Err(TurnControlError::DuplicateEvent);
        }
        if self.consumed_event_ids.len() >= self.max_decisions {
            return Err(TurnControlError::CapacityReached);
        }
        if let Some(previous_now) = self.last_decision_clock {
            if now < previous_now {
                return Err(TurnControlError::ClockRegression);
            }
        }

        let observed = parse_monotonic_ns(&candidate.observed_at_monotonic_ns)?;
        let expires = parse_monotonic_ns(&candidate.expires_at_monotonic_ns)?;
        if observed > now {
            return Err(TurnControlError::FutureObservation);
        }
        if expires <= now {
            return Err(TurnControlError::CandidateExpired);
        }
        if self.last_candidate_observed_at.is_some_and(|previous| observed < previous) {
            return Err(TurnControlError::ClockRegression);
        }
        if let Some(previous_sequence) = self.last_sequence {
            if candidate.sequence <= previous_sequence {
                return Err(TurnControlError::SequenceRegression);
            }
            if candidate.sequence != previous_sequence.saturating_add(1) {
                return Err(TurnControlError::SequenceGap);
            }
        }
        if !transition_allowed(self.current_action, candidate.action) {
            return Err(TurnControlError::InvalidTransition {
                from: self.current_action,
                to: candidate.action,
            });
        }

        // Consume the candidate before returning either decision or abstention.
        // A low-confidence event must not be re-emitted later as a fallback.
        self.consumed_event_ids.insert(candidate.event_id.clone());
        self.last_sequence = Some(candidate.sequence);
        self.last_candidate_observed_at = Some(observed);
        self.last_decision_clock = Some(now);

        let required_confidence = self.thresholds.threshold_for(candidate.action);
        if candidate.confidence < required_confidence {
            return Ok(TurnControlOutcome::Abstained(TurnAbstention {
                event_id: candidate.event_id,
                sequence: candidate.sequence,
                proposed_action: candidate.action,
                confidence: candidate.confidence,
                required_confidence,
                calibration_id: candidate.calibration_id,
            }));
        }

        self.current_action = Some(candidate.action);
        Ok(TurnControlOutcome::Emitted(TurnDecision {
            event_id: candidate.event_id,
            sequence: candidate.sequence,
            action: candidate.action,
            confidence: candidate.confidence,
            required_confidence,
            calibration_id: candidate.calibration_id,
            decided_at_monotonic_ns: now_monotonic_ns.to_owned(),
            decision_latency_ns: now.saturating_sub(observed),
        }))
    }
}

fn transition_allowed(from: Option<TurnState>, to: TurnState) -> bool {
    match (from, to) {
        (None, TurnState::Wait | TurnState::Respond) => true,
        (Some(TurnState::Wait), TurnState::Wait | TurnState::Respond | TurnState::Yield) => true,
        (Some(TurnState::Respond), TurnState::Respond | TurnState::Continue | TurnState::Yield | TurnState::Wait) => true,
        (Some(TurnState::Continue), TurnState::Continue | TurnState::Yield | TurnState::Wait) => true,
        (Some(TurnState::Yield), TurnState::Yield | TurnState::Wait | TurnState::Respond) => true,
        _ => false,
    }
}

fn validate_identifier(value: &str) -> Result<(), TurnControlError> {
    if value.is_empty()
        || value.len() > 128
        || !value.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
    {
        return Err(TurnControlError::InvalidIdentifier);
    }
    Ok(())
}

fn parse_monotonic_ns(value: &str) -> Result<u64, TurnControlError> {
    if value.is_empty()
        || (value.len() > 1 && value.starts_with('0'))
        || !value.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(TurnControlError::InvalidTimestamp);
    }
    u64::from_str(value).map_err(|_| TurnControlError::InvalidTimestamp)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLOCK: &str = "clock:turn-control-epoch-1";

    fn candidate(
        id: &str,
        sequence: u64,
        observed: &str,
        expiry: &str,
        action: TurnState,
        confidence: f32,
    ) -> TurnCandidate {
        TurnCandidate::new(
            id, sequence, CLOCK, observed, expiry, action, confidence, "calibration-v1",
        ).unwrap()
    }

    fn gate() -> TurnDecisionGate {
        TurnDecisionGate::new(CLOCK, TurnControlThresholds::default()).unwrap()
    }

    #[test]
    fn respond_candidate_emits_once_with_client_clock_latency() {
        let mut gate = gate();
        let result = gate.evaluate(
            candidate("event-1", 1, "100", "1000", TurnState::Respond, 0.9),
            CLOCK,
            "220",
        ).unwrap();
        let TurnControlOutcome::Emitted(decision) = result else { panic!("expected decision"); };
        assert_eq!(decision.action(), TurnState::Respond);
        assert_eq!(decision.decision_latency_ns(), 120);
        assert_eq!(decision.required_confidence(), 0.82);
        assert_eq!(gate.current_action(), Some(TurnState::Respond));
    }

    #[test]
    fn low_confidence_yield_abstains_and_is_consumed() {
        let mut gate = gate();
        let result = gate.evaluate(
            candidate("event-1", 1, "100", "1000", TurnState::Yield, 0.93),
            CLOCK,
            "200",
        ).unwrap();
        let TurnControlOutcome::Abstained(abstention) = result else { panic!("expected abstention"); };
        assert_eq!(abstention.required_confidence(), 0.94);
        assert_eq!(gate.current_action(), None);
        assert_eq!(gate.consumed_count(), 1);
        assert_eq!(
            gate.evaluate(
                candidate("event-1", 1, "100", "1000", TurnState::Yield, 1.0),
                CLOCK,
                "210",
            ),
            Err(TurnControlError::DuplicateEvent)
        );
    }

    #[test]
    fn yield_uses_a_higher_threshold_than_continue_or_respond() {
        let thresholds = TurnControlThresholds::default();
        assert!(thresholds.threshold_for(TurnState::Yield) > thresholds.threshold_for(TurnState::Respond));
        assert!(thresholds.threshold_for(TurnState::Respond) > thresholds.threshold_for(TurnState::Continue));
    }

    #[test]
    fn invalid_threshold_ordering_is_rejected() {
        assert_eq!(
            TurnControlThresholds::new(0.5, 0.7, 0.8, 0.9),
            Err(TurnControlError::InvalidThresholdOrdering)
        );
    }

    #[test]
    fn candidate_confidence_must_be_finite_and_bounded() {
        assert_eq!(
            TurnCandidate::new("event-1", 1, CLOCK, "1", "2", TurnState::Respond, f32::NAN, "cal-v1"),
            Err(TurnControlError::InvalidConfidence)
        );
        assert!(TurnCandidate::new("event-1", 1, CLOCK, "1", "2", TurnState::Respond, 1.1, "cal-v1").is_err());
    }

    #[test]
    fn mismatched_clock_domains_are_rejected() {
        let mut gate = gate();
        assert_eq!(
            gate.evaluate(candidate("event-1", 1, "100", "1000", TurnState::Wait, 0.9), "clock:other", "200"),
            Err(TurnControlError::WrongClockDomain)
        );
    }

    #[test]
    fn future_observations_and_expired_candidates_are_rejected() {
        let mut future_gate = gate();
        assert_eq!(
            future_gate.evaluate(candidate("event-1", 1, "201", "1000", TurnState::Wait, 0.9), CLOCK, "200"),
            Err(TurnControlError::FutureObservation)
        );
        let mut expired_gate = gate();
        assert_eq!(
            expired_gate.evaluate(candidate("event-1", 1, "100", "200", TurnState::Wait, 0.9), CLOCK, "200"),
            Err(TurnControlError::CandidateExpired)
        );
    }

    #[test]
    fn clock_regression_and_sequence_gaps_are_rejected() {
        let mut gate = gate();
        gate.evaluate(candidate("event-1", 1, "100", "1000", TurnState::Wait, 0.9), CLOCK, "200").unwrap();
        assert_eq!(
            gate.evaluate(candidate("event-2", 2, "190", "1000", TurnState::Respond, 0.9), CLOCK, "199"),
            Err(TurnControlError::ClockRegression)
        );
        assert_eq!(
            gate.evaluate(candidate("event-2", 3, "210", "1000", TurnState::Respond, 0.9), CLOCK, "220"),
            Err(TurnControlError::SequenceGap)
        );
    }

    #[test]
    fn sequence_regression_is_rejected() {
        let mut gate = gate();
        gate.evaluate(candidate("event-1", 1, "100", "1000", TurnState::Wait, 0.9), CLOCK, "200").unwrap();
        assert_eq!(
            gate.evaluate(candidate("event-2", 1, "210", "1000", TurnState::Respond, 0.9), CLOCK, "220"),
            Err(TurnControlError::SequenceRegression)
        );
    }

    #[test]
    fn impossible_turn_transition_is_rejected() {
        let mut gate = gate();
        gate.evaluate(candidate("event-1", 1, "100", "1000", TurnState::Wait, 0.9), CLOCK, "200").unwrap();
        assert!(matches!(
            gate.evaluate(candidate("event-2", 2, "210", "1000", TurnState::Continue, 0.9), CLOCK, "220"),
            Err(TurnControlError::InvalidTransition { .. })
        ));
    }

    #[test]
    fn valid_reciprocal_sequence_is_wait_respond_continue_yield() {
        let mut gate = gate();
        for (id, sequence, observed, now, action, confidence) in [
            ("event-1", 1, "100", "200", TurnState::Wait, 0.9),
            ("event-2", 2, "210", "300", TurnState::Respond, 0.9),
            ("event-3", 3, "310", "400", TurnState::Continue, 0.9),
            ("event-4", 4, "410", "500", TurnState::Yield, 0.98),
        ] {
            assert!(matches!(
                gate.evaluate(candidate(id, sequence, observed, "1000", action, confidence), CLOCK, now),
                Ok(TurnControlOutcome::Emitted(_))
            ));
        }
        assert_eq!(gate.current_action(), Some(TurnState::Yield));
    }

    #[test]
    fn abstention_consumes_the_sequence_without_changing_current_action() {
        let mut gate = gate();
        gate.evaluate(candidate("event-1", 1, "100", "1000", TurnState::Wait, 0.9), CLOCK, "200").unwrap();
        let result = gate.evaluate(
            candidate("event-2", 2, "210", "1000", TurnState::Respond, 0.5),
            CLOCK,
            "220",
        ).unwrap();
        assert!(matches!(result, TurnControlOutcome::Abstained(_)));
        assert_eq!(gate.current_action(), Some(TurnState::Wait));
        assert_eq!(gate.consumed_count(), 2);
    }

    #[test]
    fn decision_gate_is_bounded() {
        let mut gate = TurnDecisionGate::with_capacity(CLOCK, TurnControlThresholds::default(), 1).unwrap();
        gate.evaluate(candidate("event-1", 1, "100", "1000", TurnState::Wait, 0.9), CLOCK, "200").unwrap();
        assert_eq!(
            gate.evaluate(candidate("event-2", 2, "210", "1000", TurnState::Respond, 0.9), CLOCK, "220"),
            Err(TurnControlError::CapacityReached)
        );
    }

    #[test]
    fn timestamps_reject_noncanonical_decimal_values() {
        assert_eq!(
            TurnCandidate::new("event-1", 1, CLOCK, "0100", "1000", TurnState::Wait, 0.9, "cal-v1"),
            Err(TurnControlError::InvalidTimestamp)
        );
    }
}
