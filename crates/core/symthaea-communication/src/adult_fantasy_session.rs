// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Adult-only fantasy-dialogue session semantics.
//!
//! This module is deliberately non-generative and non-actuating. It establishes
//! a conversational session boundary for adult fantasy/roleplay without granting
//! human-contact consent, motor authority, identity/likeness authorization, or
//! health/diagnostic authority.

use serde::{Deserialize, Serialize};

pub const ADULT_FANTASY_SESSION_SCHEMA_V1: &str =
    "symthaea.communication.adult-fantasy-session.v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdultEligibilityEvidenceHandleV1 {
    evidence_id: String,
    participant_id: String,
    valid_from_ns: u64,
    valid_until_ns: u64,
}

impl AdultEligibilityEvidenceHandleV1 {
    pub fn new(
        evidence_id: impl Into<String>,
        participant_id: impl Into<String>,
        valid_from_ns: u64,
        valid_until_ns: u64,
    ) -> Result<Self, AdultFantasySessionError> {
        let evidence_id = evidence_id.into().trim().to_owned();
        let participant_id = participant_id.into().trim().to_owned();
        if evidence_id.is_empty() || evidence_id.len() > 256 {
            return Err(AdultFantasySessionError::InvalidEligibilityEvidenceId);
        }
        if participant_id.is_empty() || participant_id.len() > 256 {
            return Err(AdultFantasySessionError::InvalidParticipantId);
        }
        if valid_from_ns >= valid_until_ns {
            return Err(AdultFantasySessionError::InvalidEligibilityValidityWindow);
        }
        Ok(Self {
            evidence_id,
            participant_id,
            valid_from_ns,
            valid_until_ns,
        })
    }

    pub fn evidence_id(&self) -> &str {
        &self.evidence_id
    }

    pub fn participant_id(&self) -> &str {
        &self.participant_id
    }

    pub const fn is_time_valid_at(&self, now_ns: u64) -> bool {
        self.valid_from_ns <= now_ns && now_ns < self.valid_until_ns
    }
}

/// External verifier boundary for adult-eligibility evidence.
///
/// The session module intentionally does not implement age/identity verification.
/// A caller must supply a verifier backed by the product's separately governed
/// trusted eligibility mechanism.
pub trait AdultEligibilityVerifier {
    fn verify_adult_eligibility(
        &self,
        evidence: &AdultEligibilityEvidenceHandleV1,
        participant_id: &str,
        now_ns: u64,
    ) -> bool;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FantasyRealityFrameV1 {
    /// Explicitly framed roleplay between the participant and Symthaea.
    ExplicitRoleplay,
    /// Explicitly framed fictional/narrative world state.
    NarrativeFiction,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FantasyRetentionPolicyV1 {
    /// Raw fantasy state is ephemeral unless another explicit policy is admitted.
    #[default]
    Ephemeral,
    /// Only separately approved, derived preference state may persist.
    PreferenceOnly,
    /// Session continuity may persist under a separate retention implementation.
    SessionContinuity,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FantasyIdentityPolicyV1 {
    /// Original or fictional personas only; no real-person imitation is admitted.
    OriginalOrFictionalOnly,
    /// A separately governed identity/likeness authorization is referenced.
    AuthorizedRealPerson { authorization_id: String },
}

impl FantasyIdentityPolicyV1 {
    fn validate(&self) -> Result<(), AdultFantasySessionError> {
        if let Self::AuthorizedRealPerson { authorization_id } = self {
            if authorization_id.trim().is_empty() || authorization_id.len() > 256 {
                return Err(AdultFantasySessionError::InvalidIdentityAuthorizationId);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdultFantasyActivationV1 {
    participant_id: String,
    session_id: String,
    session_epoch: u64,
    reality_frame: FantasyRealityFrameV1,
    retention_policy: FantasyRetentionPolicyV1,
    identity_policy: FantasyIdentityPolicyV1,
}

impl AdultFantasyActivationV1 {
    pub fn new(
        participant_id: impl Into<String>,
        session_id: impl Into<String>,
        session_epoch: u64,
        reality_frame: FantasyRealityFrameV1,
        retention_policy: FantasyRetentionPolicyV1,
        identity_policy: FantasyIdentityPolicyV1,
    ) -> Result<Self, AdultFantasySessionError> {
        let participant_id = participant_id.into().trim().to_owned();
        let session_id = session_id.into().trim().to_owned();
        if participant_id.is_empty() || participant_id.len() > 256 {
            return Err(AdultFantasySessionError::InvalidParticipantId);
        }
        if session_id.is_empty() || session_id.len() > 256 {
            return Err(AdultFantasySessionError::InvalidSessionId);
        }
        if session_epoch == 0 {
            return Err(AdultFantasySessionError::InvalidSessionEpoch);
        }
        identity_policy.validate()?;
        Ok(Self {
            participant_id,
            session_id,
            session_epoch,
            reality_frame,
            retention_policy,
            identity_policy,
        })
    }

    pub fn participant_id(&self) -> &str {
        &self.participant_id
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub const fn session_epoch(&self) -> u64 {
        self.session_epoch
    }

    pub const fn reality_frame(&self) -> FantasyRealityFrameV1 {
        self.reality_frame
    }

    pub const fn retention_policy(&self) -> FantasyRetentionPolicyV1 {
        self.retention_policy
    }

    pub fn identity_policy(&self) -> &FantasyIdentityPolicyV1 {
        &self.identity_policy
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdultFantasyStopReasonV1 {
    ExplicitExit,
    SlowdownEscalatedToExit,
    AdultEligibilityNoLongerValid,
    SessionBoundaryInvalidated,
    RetentionPolicyInvalidated,
    ProtectivePreemption,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdultFantasySessionStateV1 {
    Idle,
    Active { session_epoch: u64 },
    Stopped {
        terminal_session_epoch: u64,
        reason: AdultFantasyStopReasonV1,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdultFantasySessionError {
    InvalidEligibilityEvidenceId,
    InvalidEligibilityValidityWindow,
    InvalidParticipantId,
    InvalidSessionId,
    InvalidSessionEpoch,
    InvalidIdentityAuthorizationId,
    ParticipantMismatch,
    EligibilityEvidenceNotCurrent,
    EligibilityVerificationFailed,
    SessionNotIdle,
    StaleSessionEpoch,
    NoActiveSession,
    ExitNotAcknowledgable,
}

/// Non-cloneable runtime boundary for one participant/session identity.
///
/// This value is conversational eligibility state only. It is intentionally not
/// serializable and has no conversion into human-contact or motor authority.
#[derive(Debug)]
pub struct AdultFantasySessionV1 {
    participant_id: String,
    session_id: String,
    state: AdultFantasySessionStateV1,
    highest_terminal_session_epoch: u64,
    active_activation: Option<AdultFantasyActivationV1>,
    active_eligibility_evidence: Option<AdultEligibilityEvidenceHandleV1>,
}

impl AdultFantasySessionV1 {
    pub fn new(
        participant_id: impl Into<String>,
        session_id: impl Into<String>,
    ) -> Result<Self, AdultFantasySessionError> {
        let participant_id = participant_id.into().trim().to_owned();
        let session_id = session_id.into().trim().to_owned();
        if participant_id.is_empty() || participant_id.len() > 256 {
            return Err(AdultFantasySessionError::InvalidParticipantId);
        }
        if session_id.is_empty() || session_id.len() > 256 {
            return Err(AdultFantasySessionError::InvalidSessionId);
        }
        Ok(Self {
            participant_id,
            session_id,
            state: AdultFantasySessionStateV1::Idle,
            highest_terminal_session_epoch: 0,
            active_activation: None,
            active_eligibility_evidence: None,
        })
    }

    pub const fn state(&self) -> AdultFantasySessionStateV1 {
        self.state
    }

    pub const fn highest_terminal_session_epoch(&self) -> u64 {
        self.highest_terminal_session_epoch
    }

    pub fn active_activation(&self) -> Option<&AdultFantasyActivationV1> {
        self.active_activation.as_ref()
    }

    pub const fn is_active(&self) -> bool {
        matches!(self.state, AdultFantasySessionStateV1::Active { .. })
    }

    pub fn minimum_next_session_epoch(&self) -> Option<u64> {
        self.highest_terminal_session_epoch.checked_add(1)
    }

    pub fn activate<V: AdultEligibilityVerifier>(
        &mut self,
        activation: AdultFantasyActivationV1,
        eligibility_evidence: AdultEligibilityEvidenceHandleV1,
        verifier: &V,
        now_ns: u64,
    ) -> Result<(), AdultFantasySessionError> {
        if self.state != AdultFantasySessionStateV1::Idle {
            return Err(AdultFantasySessionError::SessionNotIdle);
        }
        if activation.participant_id != self.participant_id
            || eligibility_evidence.participant_id != self.participant_id
        {
            return Err(AdultFantasySessionError::ParticipantMismatch);
        }
        if activation.session_id != self.session_id {
            return Err(AdultFantasySessionError::InvalidSessionId);
        }
        if activation.session_epoch <= self.highest_terminal_session_epoch {
            return Err(AdultFantasySessionError::StaleSessionEpoch);
        }
        if !eligibility_evidence.is_time_valid_at(now_ns) {
            return Err(AdultFantasySessionError::EligibilityEvidenceNotCurrent);
        }
        if !verifier.verify_adult_eligibility(
            &eligibility_evidence,
            &self.participant_id,
            now_ns,
        ) {
            return Err(AdultFantasySessionError::EligibilityVerificationFailed);
        }

        self.state = AdultFantasySessionStateV1::Active {
            session_epoch: activation.session_epoch,
        };
        self.active_activation = Some(activation);
        self.active_eligibility_evidence = Some(eligibility_evidence);
        Ok(())
    }

    /// Revalidate the active adult-eligibility evidence.
    ///
    /// Any loss of current trusted eligibility immediately latches the fantasy
    /// session into `Stopped`. It never auto-resumes if evidence later returns.
    pub fn revalidate<V: AdultEligibilityVerifier>(
        &mut self,
        verifier: &V,
        now_ns: u64,
    ) -> Result<bool, AdultFantasySessionError> {
        let evidence = self
            .active_eligibility_evidence
            .as_ref()
            .ok_or(AdultFantasySessionError::NoActiveSession)?;
        if !self.is_active() {
            return Err(AdultFantasySessionError::NoActiveSession);
        }
        let valid = evidence.is_time_valid_at(now_ns)
            && verifier.verify_adult_eligibility(evidence, &self.participant_id, now_ns);
        if valid {
            return Ok(true);
        }
        self.stop(AdultFantasyStopReasonV1::AdultEligibilityNoLongerValid)?;
        Ok(false)
    }

    /// Latch fantasy-session stop. Repeated stop calls while already stopped are
    /// idempotent and return `false`.
    pub fn stop(
        &mut self,
        reason: AdultFantasyStopReasonV1,
    ) -> Result<bool, AdultFantasySessionError> {
        match self.state {
            AdultFantasySessionStateV1::Active { session_epoch } => {
                self.highest_terminal_session_epoch =
                    self.highest_terminal_session_epoch.max(session_epoch);
                self.active_activation = None;
                self.active_eligibility_evidence = None;
                self.state = AdultFantasySessionStateV1::Stopped {
                    terminal_session_epoch: session_epoch,
                    reason,
                };
                Ok(true)
            }
            AdultFantasySessionStateV1::Stopped { .. } => Ok(false),
            AdultFantasySessionStateV1::Idle => {
                Err(AdultFantasySessionError::NoActiveSession)
            }
        }
    }

    /// Return to ordinary conversation after a stopped fantasy session.
    ///
    /// This never restores the prior fantasy epoch. A strictly newer epoch plus
    /// current independently verified adult-eligibility evidence is required.
    pub fn acknowledge_exit_to_ordinary_conversation(
        &mut self,
    ) -> Result<(), AdultFantasySessionError> {
        match self.state {
            AdultFantasySessionStateV1::Stopped { .. } => {
                self.state = AdultFantasySessionStateV1::Idle;
                Ok(())
            }
            _ => Err(AdultFantasySessionError::ExitNotAcknowledgable),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AllowVerifier;
    impl AdultEligibilityVerifier for AllowVerifier {
        fn verify_adult_eligibility(
            &self,
            _evidence: &AdultEligibilityEvidenceHandleV1,
            _participant_id: &str,
            _now_ns: u64,
        ) -> bool {
            true
        }
    }

    struct DenyVerifier;
    impl AdultEligibilityVerifier for DenyVerifier {
        fn verify_adult_eligibility(
            &self,
            _evidence: &AdultEligibilityEvidenceHandleV1,
            _participant_id: &str,
            _now_ns: u64,
        ) -> bool {
            false
        }
    }

    fn activation(epoch: u64) -> AdultFantasyActivationV1 {
        AdultFantasyActivationV1::new(
            "participant-a",
            "session-a",
            epoch,
            FantasyRealityFrameV1::ExplicitRoleplay,
            FantasyRetentionPolicyV1::Ephemeral,
            FantasyIdentityPolicyV1::OriginalOrFictionalOnly,
        )
        .unwrap()
    }

    fn evidence() -> AdultEligibilityEvidenceHandleV1 {
        AdultEligibilityEvidenceHandleV1::new("adult-evidence-1", "participant-a", 100, 300)
            .unwrap()
    }

    #[test]
    fn current_verified_adult_evidence_can_activate_explicit_fantasy() {
        let mut session = AdultFantasySessionV1::new("participant-a", "session-a").unwrap();
        session
            .activate(activation(1), evidence(), &AllowVerifier, 150)
            .unwrap();
        assert_eq!(
            session.state(),
            AdultFantasySessionStateV1::Active { session_epoch: 1 }
        );
        assert!(session.is_active());
        assert_eq!(
            session.active_activation().unwrap().reality_frame(),
            FantasyRealityFrameV1::ExplicitRoleplay
        );
    }

    #[test]
    fn unverified_or_expired_adult_evidence_fails_closed() {
        let mut denied = AdultFantasySessionV1::new("participant-a", "session-a").unwrap();
        assert_eq!(
            denied.activate(activation(1), evidence(), &DenyVerifier, 150),
            Err(AdultFantasySessionError::EligibilityVerificationFailed)
        );

        let mut expired = AdultFantasySessionV1::new("participant-a", "session-a").unwrap();
        assert_eq!(
            expired.activate(activation(1), evidence(), &AllowVerifier, 300),
            Err(AdultFantasySessionError::EligibilityEvidenceNotCurrent)
        );
    }

    #[test]
    fn participant_substitution_fails_closed() {
        let wrong = AdultEligibilityEvidenceHandleV1::new(
            "adult-evidence-other",
            "participant-b",
            100,
            300,
        )
        .unwrap();
        let mut session = AdultFantasySessionV1::new("participant-a", "session-a").unwrap();
        assert_eq!(
            session.activate(activation(1), wrong, &AllowVerifier, 150),
            Err(AdultFantasySessionError::ParticipantMismatch)
        );
    }

    #[test]
    fn stop_latches_and_same_epoch_never_auto_resumes() {
        let mut session = AdultFantasySessionV1::new("participant-a", "session-a").unwrap();
        session
            .activate(activation(2), evidence(), &AllowVerifier, 150)
            .unwrap();
        assert!(session.stop(AdultFantasyStopReasonV1::ExplicitExit).unwrap());
        assert!(!session.stop(AdultFantasyStopReasonV1::ProtectivePreemption).unwrap());
        assert!(!session.is_active());
        session
            .acknowledge_exit_to_ordinary_conversation()
            .unwrap();
        assert_eq!(
            session.activate(activation(2), evidence(), &AllowVerifier, 160),
            Err(AdultFantasySessionError::StaleSessionEpoch)
        );
    }

    #[test]
    fn strictly_fresh_epoch_can_reenter_after_explicit_exit() {
        let mut session = AdultFantasySessionV1::new("participant-a", "session-a").unwrap();
        session
            .activate(activation(2), evidence(), &AllowVerifier, 150)
            .unwrap();
        session.stop(AdultFantasyStopReasonV1::ExplicitExit).unwrap();
        session
            .acknowledge_exit_to_ordinary_conversation()
            .unwrap();
        session
            .activate(activation(3), evidence(), &AllowVerifier, 160)
            .unwrap();
        assert_eq!(
            session.state(),
            AdultFantasySessionStateV1::Active { session_epoch: 3 }
        );
    }

    #[test]
    fn eligibility_loss_while_active_stops_without_auto_resume() {
        let mut session = AdultFantasySessionV1::new("participant-a", "session-a").unwrap();
        session
            .activate(activation(1), evidence(), &AllowVerifier, 150)
            .unwrap();
        assert!(!session.revalidate(&AllowVerifier, 300).unwrap());
        assert_eq!(
            session.state(),
            AdultFantasySessionStateV1::Stopped {
                terminal_session_epoch: 1,
                reason: AdultFantasyStopReasonV1::AdultEligibilityNoLongerValid,
            }
        );
    }

    #[test]
    fn malformed_real_person_authorization_is_rejected() {
        assert_eq!(
            AdultFantasyActivationV1::new(
                "participant-a",
                "session-a",
                1,
                FantasyRealityFrameV1::NarrativeFiction,
                FantasyRetentionPolicyV1::Ephemeral,
                FantasyIdentityPolicyV1::AuthorizedRealPerson {
                    authorization_id: "   ".into(),
                },
            ),
            Err(AdultFantasySessionError::InvalidIdentityAuthorizationId)
        );
    }
}
