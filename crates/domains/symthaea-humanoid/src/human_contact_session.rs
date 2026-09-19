// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Latching runtime lifecycle for scoped human-contact consent.
//!
//! This module is deliberately non-actuating. It consumes semantic consent scopes
//! and tracks whether one exact consent epoch is currently eligible for downstream
//! authority evaluation. Revocation destroys that eligibility and cannot be undone
//! in place; a strictly newer consent epoch is required before contact may become
//! eligible again.

use crate::human_contact_consent::HumanContactConsentScopeV1;

/// Why a live human-contact consent epoch was invalidated.
///
/// Reasons are audit semantics only. No reason can restore authority or select a
/// motor behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanContactRevocationReason {
    ExplicitWithdrawal,
    ConsentEvidenceLost,
    AuthorityInvalidated,
    ProtectivePreemption,
    SessionReset,
    RelevantConfigurationChanged,
}

/// Runtime consent lifecycle state.
///
/// `WithdrawRequired` is latching: acknowledging completion of withdrawal moves
/// to `Idle`, never back to `Active`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanContactConsentSessionState {
    Idle,
    Active {
        consent_epoch: u64,
    },
    WithdrawRequired {
        revoked_consent_epoch: u64,
        reason: HumanContactRevocationReason,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanContactSessionError {
    EmptyParticipant,
    EmptySession,
    ParticipantMismatch,
    SessionMismatch,
    ScopeNotLive,
    SessionNotIdle,
    StaleConsentEpoch,
    NoActiveConsent,
    WithdrawalNotRequired,
}

/// Non-cloneable runtime lifecycle for one exact participant/session pair.
///
/// This value is not a motor permit. It only proves that the consent lifecycle has
/// not been revoked and that the admitted semantic scope belongs to a fresh epoch.
#[derive(Debug)]
pub struct HumanContactAuthoritySessionV1 {
    participant_id: String,
    session_id: String,
    state: HumanContactConsentSessionState,
    highest_terminal_consent_epoch: u64,
}

impl HumanContactAuthoritySessionV1 {
    pub fn new(
        participant_id: impl Into<String>,
        session_id: impl Into<String>,
    ) -> Result<Self, HumanContactSessionError> {
        let participant_id = participant_id.into().trim().to_owned();
        let session_id = session_id.into().trim().to_owned();
        if participant_id.is_empty() {
            return Err(HumanContactSessionError::EmptyParticipant);
        }
        if session_id.is_empty() {
            return Err(HumanContactSessionError::EmptySession);
        }
        Ok(Self {
            participant_id,
            session_id,
            state: HumanContactConsentSessionState::Idle,
            highest_terminal_consent_epoch: 0,
        })
    }

    pub fn participant_id(&self) -> &str {
        &self.participant_id
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub const fn state(&self) -> HumanContactConsentSessionState {
        self.state
    }

    pub const fn highest_terminal_consent_epoch(&self) -> u64 {
        self.highest_terminal_consent_epoch
    }

    /// Lowest epoch that could be fresh after all prior terminal transitions.
    pub fn minimum_next_consent_epoch(&self) -> Option<u64> {
        self.highest_terminal_consent_epoch.checked_add(1)
    }

    /// Admit one semantic scope into the runtime lifecycle.
    ///
    /// Admission is reject-only. It authenticates nothing and grants no motor
    /// authority; later layers must still verify authentic consent, qualification,
    /// live authority evidence, whole-body intent, and final safety.
    pub fn admit_scope(
        &mut self,
        scope: &HumanContactConsentScopeV1,
        now_ns: u64,
    ) -> Result<(), HumanContactSessionError> {
        if self.state != HumanContactConsentSessionState::Idle {
            return Err(HumanContactSessionError::SessionNotIdle);
        }
        if scope.participant_id() != self.participant_id {
            return Err(HumanContactSessionError::ParticipantMismatch);
        }
        if scope.session_id() != self.session_id {
            return Err(HumanContactSessionError::SessionMismatch);
        }
        if !scope.is_live_at(now_ns) {
            return Err(HumanContactSessionError::ScopeNotLive);
        }
        if scope.consent_epoch() <= self.highest_terminal_consent_epoch {
            return Err(HumanContactSessionError::StaleConsentEpoch);
        }

        self.state = HumanContactConsentSessionState::Active {
            consent_epoch: scope.consent_epoch(),
        };
        Ok(())
    }

    /// Latch revocation of the currently active consent epoch.
    ///
    /// Returns `true` when this call newly latched revocation. Repeated stop calls
    /// while withdrawal is already required are idempotent and return `false`.
    pub fn revoke_active(
        &mut self,
        reason: HumanContactRevocationReason,
    ) -> Result<bool, HumanContactSessionError> {
        match self.state {
            HumanContactConsentSessionState::Active { consent_epoch } => {
                self.highest_terminal_consent_epoch =
                    self.highest_terminal_consent_epoch.max(consent_epoch);
                self.state = HumanContactConsentSessionState::WithdrawRequired {
                    revoked_consent_epoch: consent_epoch,
                    reason,
                };
                Ok(true)
            }
            HumanContactConsentSessionState::WithdrawRequired { .. } => Ok(false),
            HumanContactConsentSessionState::Idle => {
                Err(HumanContactSessionError::NoActiveConsent)
            }
        }
    }

    /// Record that bounded withdrawal/disengagement has completed.
    ///
    /// This never restores the revoked epoch. The session returns to `Idle`, from
    /// which only a strictly newer live consent epoch can be admitted.
    pub fn acknowledge_withdrawal_complete(
        &mut self,
    ) -> Result<(), HumanContactSessionError> {
        match self.state {
            HumanContactConsentSessionState::WithdrawRequired { .. } => {
                self.state = HumanContactConsentSessionState::Idle;
                Ok(())
            }
            _ => Err(HumanContactSessionError::WithdrawalNotRequired),
        }
    }

    pub const fn is_active(&self) -> bool {
        matches!(self.state, HumanContactConsentSessionState::Active { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::human_contact_consent::{
        HumanBodyRegion, HumanContactClass, HumanContactRobotSite,
        HumanContactConsentScopeV1,
    };

    fn site(name: &str) -> HumanContactRobotSite {
        HumanContactRobotSite::new(name).unwrap()
    }

    fn scope(
        participant: &str,
        session: &str,
        consent_epoch: u64,
        revocation_epoch: u64,
        valid_from_ns: u64,
        valid_until_ns: u64,
    ) -> HumanContactConsentScopeV1 {
        HumanContactConsentScopeV1::new(
            participant,
            session,
            consent_epoch,
            revocation_epoch,
            HumanContactClass::Social,
            [HumanBodyRegion::Hand],
            [site("right_hand")],
            valid_from_ns,
            valid_until_ns,
        )
        .unwrap()
    }

    #[test]
    fn live_scope_can_enter_active_state() {
        let mut session = HumanContactAuthoritySessionV1::new("p", "s").unwrap();
        session.admit_scope(&scope("p", "s", 1, 0, 100, 200), 150).unwrap();
        assert_eq!(
            session.state(),
            HumanContactConsentSessionState::Active { consent_epoch: 1 }
        );
        assert!(session.is_active());
    }

    #[test]
    fn explicit_revocation_latches_and_removes_active_eligibility() {
        let mut session = HumanContactAuthoritySessionV1::new("p", "s").unwrap();
        session.admit_scope(&scope("p", "s", 2, 1, 100, 200), 150).unwrap();
        assert!(session
            .revoke_active(HumanContactRevocationReason::ExplicitWithdrawal)
            .unwrap());
        assert_eq!(
            session.state(),
            HumanContactConsentSessionState::WithdrawRequired {
                revoked_consent_epoch: 2,
                reason: HumanContactRevocationReason::ExplicitWithdrawal,
            }
        );
        assert!(!session.is_active());
    }

    #[test]
    fn repeated_stop_is_idempotent_while_revocation_is_latched() {
        let mut session = HumanContactAuthoritySessionV1::new("p", "s").unwrap();
        session.admit_scope(&scope("p", "s", 2, 1, 100, 200), 150).unwrap();
        assert!(session
            .revoke_active(HumanContactRevocationReason::ExplicitWithdrawal)
            .unwrap());
        assert!(!session
            .revoke_active(HumanContactRevocationReason::ProtectivePreemption)
            .unwrap());
        assert_eq!(
            session.state(),
            HumanContactConsentSessionState::WithdrawRequired {
                revoked_consent_epoch: 2,
                reason: HumanContactRevocationReason::ExplicitWithdrawal,
            }
        );
    }

    #[test]
    fn withdrawal_acknowledgement_never_auto_resumes_old_epoch() {
        let mut session = HumanContactAuthoritySessionV1::new("p", "s").unwrap();
        let old_scope = scope("p", "s", 2, 1, 100, 300);
        session.admit_scope(&old_scope, 150).unwrap();
        session
            .revoke_active(HumanContactRevocationReason::ExplicitWithdrawal)
            .unwrap();
        session.acknowledge_withdrawal_complete().unwrap();
        assert_eq!(session.state(), HumanContactConsentSessionState::Idle);
        assert_eq!(session.highest_terminal_consent_epoch(), 2);
        assert_eq!(
            session.admit_scope(&old_scope, 160),
            Err(HumanContactSessionError::StaleConsentEpoch)
        );
    }

    #[test]
    fn strictly_fresh_epoch_can_be_admitted_after_revocation() {
        let mut session = HumanContactAuthoritySessionV1::new("p", "s").unwrap();
        session
            .admit_scope(&scope("p", "s", 2, 1, 100, 300), 150)
            .unwrap();
        session
            .revoke_active(HumanContactRevocationReason::ExplicitWithdrawal)
            .unwrap();
        session.acknowledge_withdrawal_complete().unwrap();
        session
            .admit_scope(&scope("p", "s", 3, 2, 150, 400), 200)
            .unwrap();
        assert_eq!(
            session.state(),
            HumanContactConsentSessionState::Active { consent_epoch: 3 }
        );
    }

    #[test]
    fn wrong_participant_or_session_fails_closed() {
        let mut authority = HumanContactAuthoritySessionV1::new("p", "s").unwrap();
        assert_eq!(
            authority.admit_scope(&scope("other", "s", 1, 0, 100, 200), 150),
            Err(HumanContactSessionError::ParticipantMismatch)
        );
        assert_eq!(
            authority.admit_scope(&scope("p", "other", 1, 0, 100, 200), 150),
            Err(HumanContactSessionError::SessionMismatch)
        );
    }

    #[test]
    fn expired_or_revoked_scope_cannot_become_active() {
        let mut authority = HumanContactAuthoritySessionV1::new("p", "s").unwrap();
        assert_eq!(
            authority.admit_scope(&scope("p", "s", 1, 0, 100, 150), 150),
            Err(HumanContactSessionError::ScopeNotLive)
        );
        assert_eq!(
            authority.admit_scope(&scope("p", "s", 2, 2, 100, 200), 150),
            Err(HumanContactSessionError::ScopeNotLive)
        );
    }

    #[test]
    fn new_scope_cannot_replace_an_active_scope_in_place() {
        let mut authority = HumanContactAuthoritySessionV1::new("p", "s").unwrap();
        authority
            .admit_scope(&scope("p", "s", 1, 0, 100, 200), 150)
            .unwrap();
        assert_eq!(
            authority.admit_scope(&scope("p", "s", 2, 1, 100, 300), 160),
            Err(HumanContactSessionError::SessionNotIdle)
        );
    }
}
