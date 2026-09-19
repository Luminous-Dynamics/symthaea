// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bound surface-contact authority state machine.
//!
//! This module deliberately separates planner/contact intent from hard surface
//! authority. Legacy contact booleans are not an authority source here.

use serde::{Deserialize, Serialize};

use crate::multi_contact::ContactSite;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContactAuthorityModeV1 {
    Inactive,
    Scheduled,
    Acquiring,
    Established,
    Releasing,
    Lost,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContactEvidenceClassV1 {
    SimulatorDerived,
    HardwareEstimated,
    HardwareMeasured,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ContactAuthorityPolicyV1 {
    pub maximum_evidence_age_s: f64,
    pub maximum_future_skew_s: f64,
}

impl ContactAuthorityPolicyV1 {
    pub fn validate(&self) -> bool {
        self.maximum_evidence_age_s.is_finite()
            && self.maximum_evidence_age_s > 0.0
            && self.maximum_future_skew_s.is_finite()
            && self.maximum_future_skew_s >= 0.0
    }

    pub fn policy_id(&self) -> Option<String> {
        self.validate().then(|| {
            format!(
                "contact-authority-v1:max-age:{:.17e}:max-future-skew:{:.17e}",
                self.maximum_evidence_age_s, self.maximum_future_skew_s
            )
        })
    }
}

/// Sealed runtime token representing a complete contact-establishment lineage.
///
/// Fields are private and this type intentionally does not implement
/// `Deserialize`. External callers therefore cannot manufacture an Established
/// token from JSON or by filling public identity strings. A later verified
/// adapter must construct this token from the exact typed evidence chain.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EstablishedContactEvidenceV1 {
    site: ContactSite,
    model_id: String,
    sampled_at_s: f64,
    geometry_id: String,
    interaction_id: String,
    support_normal_policy_id: String,
    support_normal_evidence_id: String,
    support_normal_verification_id: String,
    contact_acceleration_evidence_id: String,
    source_class: ContactEvidenceClassV1,
}

impl EstablishedContactEvidenceV1 {
    pub fn site(&self) -> ContactSite {
        self.site
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn sampled_at_s(&self) -> f64 {
        self.sampled_at_s
    }

    pub fn geometry_id(&self) -> &str {
        &self.geometry_id
    }

    pub fn interaction_id(&self) -> &str {
        &self.interaction_id
    }

    pub fn support_normal_policy_id(&self) -> &str {
        &self.support_normal_policy_id
    }

    pub fn support_normal_evidence_id(&self) -> &str {
        &self.support_normal_evidence_id
    }

    pub fn support_normal_verification_id(&self) -> &str {
        &self.support_normal_verification_id
    }

    pub fn contact_acceleration_evidence_id(&self) -> &str {
        &self.contact_acceleration_evidence_id
    }

    pub fn source_class(&self) -> ContactEvidenceClassV1 {
        self.source_class
    }

    fn validate(&self) -> bool {
        self.sampled_at_s.is_finite()
            && self.sampled_at_s >= 0.0
            && nonempty(&self.model_id)
            && nonempty(&self.geometry_id)
            && nonempty(&self.interaction_id)
            && nonempty(&self.support_normal_policy_id)
            && nonempty(&self.support_normal_evidence_id)
            && nonempty(&self.support_normal_verification_id)
            && nonempty(&self.contact_acceleration_evidence_id)
    }

    pub fn lineage_id(&self) -> Option<String> {
        self.validate().then(|| {
            format!(
                "contact-establishment-v1:site:{:?}:model:{}:sampled-at:{:.17e}:geometry:{}:interaction:{}:normal-policy:{}:normal-evidence:{}:normal-verification:{}:contact-acceleration:{}:source:{:?}",
                self.site,
                self.model_id,
                self.sampled_at_s,
                self.geometry_id,
                self.interaction_id,
                self.support_normal_policy_id,
                self.support_normal_evidence_id,
                self.support_normal_verification_id,
                self.contact_acceleration_evidence_id,
                self.source_class,
            )
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContactAuthorityTransitionKindV1 {
    Schedule,
    BeginAcquisition,
    Establish,
    RefreshEstablished,
    BeginRelease,
    CompleteRelease,
    Invalidate,
    ResetLost,
    EvidenceExpired,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContactAuthorityTransitionV1 {
    pub sequence: u64,
    pub establishment_epoch: u64,
    pub site: ContactSite,
    pub from: ContactAuthorityModeV1,
    pub to: ContactAuthorityModeV1,
    pub kind: ContactAuthorityTransitionKindV1,
    pub at_s: f64,
    pub policy_id: String,
    pub reason_id: String,
    pub evidence_lineage_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContactAuthorityError {
    InvalidPolicy,
    InvalidTime,
    TimeRegression,
    MissingReasonIdentity,
    InvalidEvidence,
    EvidenceSiteMismatch,
    EvidenceTooOld,
    EvidenceTooFarInFuture,
    IllegalTransition,
    RefreshModelMismatch,
    RefreshGeometryMismatch,
    RefreshSupportNormalPolicyMismatch,
    RefreshEvidenceClassMismatch,
    SequenceOverflow,
    EpochOverflow,
}

/// Runtime contact-authority state. It is intentionally not deserializable: a
/// persisted mode bit must not be able to manufacture `Established` authority.
#[derive(Debug, Clone)]
pub struct ContactAuthorityStateMachineV1 {
    site: ContactSite,
    policy: ContactAuthorityPolicyV1,
    mode: ContactAuthorityModeV1,
    transition_sequence: u64,
    establishment_epoch: u64,
    last_transition_at_s: f64,
    last_reason_id: String,
    established_evidence: Option<EstablishedContactEvidenceV1>,
}

impl ContactAuthorityStateMachineV1 {
    pub fn new(
        site: ContactSite,
        policy: ContactAuthorityPolicyV1,
        initialized_at_s: f64,
    ) -> Result<Self, ContactAuthorityError> {
        if !policy.validate() {
            return Err(ContactAuthorityError::InvalidPolicy);
        }
        if !valid_time(initialized_at_s) {
            return Err(ContactAuthorityError::InvalidTime);
        }
        Ok(Self {
            site,
            policy,
            mode: ContactAuthorityModeV1::Inactive,
            transition_sequence: 0,
            establishment_epoch: 0,
            last_transition_at_s: initialized_at_s,
            last_reason_id: "initialized".to_string(),
            established_evidence: None,
        })
    }

    pub const fn site(&self) -> ContactSite {
        self.site
    }

    pub const fn mode(&self) -> ContactAuthorityModeV1 {
        self.mode
    }

    pub const fn transition_sequence(&self) -> u64 {
        self.transition_sequence
    }

    pub const fn establishment_epoch(&self) -> u64 {
        self.establishment_epoch
    }

    pub const fn last_transition_at_s(&self) -> f64 {
        self.last_transition_at_s
    }

    pub fn last_reason_id(&self) -> &str {
        &self.last_reason_id
    }

    pub const fn policy(&self) -> ContactAuthorityPolicyV1 {
        self.policy
    }

    pub fn policy_id(&self) -> String {
        self.policy
            .policy_id()
            .expect("validated ContactAuthorityPolicyV1")
    }

    pub fn established_evidence(&self) -> Option<&EstablishedContactEvidenceV1> {
        (self.mode == ContactAuthorityModeV1::Established)
            .then_some(self.established_evidence.as_ref())
            .flatten()
    }

    pub fn surface_wrench_eligible_at(&self, now_s: f64) -> bool {
        valid_time(now_s)
            && now_s >= self.last_transition_at_s
            && self.mode == ContactAuthorityModeV1::Established
            && self
                .established_evidence
                .as_ref()
                .is_some_and(|evidence| self.evidence_fresh_at(evidence, now_s).is_ok())
    }

    pub fn schedule(
        &mut self,
        now_s: f64,
        intent_id: &str,
    ) -> Result<ContactAuthorityTransitionV1, ContactAuthorityError> {
        if self.mode != ContactAuthorityModeV1::Inactive {
            return Err(ContactAuthorityError::IllegalTransition);
        }
        self.transition(
            ContactAuthorityModeV1::Scheduled,
            ContactAuthorityTransitionKindV1::Schedule,
            now_s,
            intent_id,
            None,
            false,
        )
    }

    pub fn begin_acquisition(
        &mut self,
        now_s: f64,
        acquisition_id: &str,
    ) -> Result<ContactAuthorityTransitionV1, ContactAuthorityError> {
        if self.mode != ContactAuthorityModeV1::Scheduled {
            return Err(ContactAuthorityError::IllegalTransition);
        }
        self.transition(
            ContactAuthorityModeV1::Acquiring,
            ContactAuthorityTransitionKindV1::BeginAcquisition,
            now_s,
            acquisition_id,
            None,
            false,
        )
    }

    pub fn establish(
        &mut self,
        now_s: f64,
        reason_id: &str,
        evidence: EstablishedContactEvidenceV1,
    ) -> Result<ContactAuthorityTransitionV1, ContactAuthorityError> {
        if self.mode != ContactAuthorityModeV1::Acquiring {
            return Err(ContactAuthorityError::IllegalTransition);
        }
        self.validate_establishment_evidence(&evidence, now_s)?;
        let lineage = evidence
            .lineage_id()
            .ok_or(ContactAuthorityError::InvalidEvidence)?;
        let next_epoch = self
            .establishment_epoch
            .checked_add(1)
            .ok_or(ContactAuthorityError::EpochOverflow)?;
        let transition = self.transition(
            ContactAuthorityModeV1::Established,
            ContactAuthorityTransitionKindV1::Establish,
            now_s,
            reason_id,
            Some(lineage),
            false,
        )?;
        self.establishment_epoch = next_epoch;
        self.established_evidence = Some(evidence);
        Ok(ContactAuthorityTransitionV1 {
            establishment_epoch: self.establishment_epoch,
            ..transition
        })
    }

    pub fn refresh_established(
        &mut self,
        now_s: f64,
        reason_id: &str,
        evidence: EstablishedContactEvidenceV1,
    ) -> Result<ContactAuthorityTransitionV1, ContactAuthorityError> {
        if self.mode != ContactAuthorityModeV1::Established {
            return Err(ContactAuthorityError::IllegalTransition);
        }
        self.validate_establishment_evidence(&evidence, now_s)?;
        let current = self
            .established_evidence
            .as_ref()
            .ok_or(ContactAuthorityError::InvalidEvidence)?;
        if current.model_id != evidence.model_id {
            return Err(ContactAuthorityError::RefreshModelMismatch);
        }
        if current.geometry_id != evidence.geometry_id {
            return Err(ContactAuthorityError::RefreshGeometryMismatch);
        }
        if current.support_normal_policy_id != evidence.support_normal_policy_id {
            return Err(ContactAuthorityError::RefreshSupportNormalPolicyMismatch);
        }
        if current.source_class != evidence.source_class {
            return Err(ContactAuthorityError::RefreshEvidenceClassMismatch);
        }
        let lineage = evidence
            .lineage_id()
            .ok_or(ContactAuthorityError::InvalidEvidence)?;
        let transition = self.transition(
            ContactAuthorityModeV1::Established,
            ContactAuthorityTransitionKindV1::RefreshEstablished,
            now_s,
            reason_id,
            Some(lineage),
            false,
        )?;
        self.established_evidence = Some(evidence);
        Ok(transition)
    }

    pub fn begin_release(
        &mut self,
        now_s: f64,
        release_id: &str,
    ) -> Result<ContactAuthorityTransitionV1, ContactAuthorityError> {
        if self.mode != ContactAuthorityModeV1::Established {
            return Err(ContactAuthorityError::IllegalTransition);
        }
        self.transition(
            ContactAuthorityModeV1::Releasing,
            ContactAuthorityTransitionKindV1::BeginRelease,
            now_s,
            release_id,
            None,
            true,
        )
    }

    pub fn complete_release(
        &mut self,
        now_s: f64,
        reason_id: &str,
    ) -> Result<ContactAuthorityTransitionV1, ContactAuthorityError> {
        if self.mode != ContactAuthorityModeV1::Releasing {
            return Err(ContactAuthorityError::IllegalTransition);
        }
        self.transition(
            ContactAuthorityModeV1::Inactive,
            ContactAuthorityTransitionKindV1::CompleteRelease,
            now_s,
            reason_id,
            None,
            true,
        )
    }

    pub fn invalidate(
        &mut self,
        now_s: f64,
        reason_id: &str,
    ) -> Result<ContactAuthorityTransitionV1, ContactAuthorityError> {
        if matches!(
            self.mode,
            ContactAuthorityModeV1::Inactive | ContactAuthorityModeV1::Lost
        ) {
            return Err(ContactAuthorityError::IllegalTransition);
        }
        self.transition(
            ContactAuthorityModeV1::Lost,
            ContactAuthorityTransitionKindV1::Invalidate,
            now_s,
            reason_id,
            None,
            true,
        )
    }

    pub fn reset_lost(
        &mut self,
        now_s: f64,
        reason_id: &str,
    ) -> Result<ContactAuthorityTransitionV1, ContactAuthorityError> {
        if self.mode != ContactAuthorityModeV1::Lost {
            return Err(ContactAuthorityError::IllegalTransition);
        }
        self.transition(
            ContactAuthorityModeV1::Inactive,
            ContactAuthorityTransitionKindV1::ResetLost,
            now_s,
            reason_id,
            None,
            true,
        )
    }

    /// Reconcile time-sensitive authority. If current Established evidence has
    /// become stale or is otherwise no longer fresh, authority is revoked in
    /// this call rather than waiting for a later planner/controller phase.
    pub fn reconcile_time(
        &mut self,
        now_s: f64,
    ) -> Result<Option<ContactAuthorityTransitionV1>, ContactAuthorityError> {
        self.validate_transition_time(now_s)?;
        if self.mode != ContactAuthorityModeV1::Established {
            return Ok(None);
        }
        let fresh = self
            .established_evidence
            .as_ref()
            .is_some_and(|evidence| self.evidence_fresh_at(evidence, now_s).is_ok());
        if fresh {
            return Ok(None);
        }
        self.transition(
            ContactAuthorityModeV1::Lost,
            ContactAuthorityTransitionKindV1::EvidenceExpired,
            now_s,
            "evidence-expired",
            None,
            true,
        )
        .map(Some)
    }

    fn validate_establishment_evidence(
        &self,
        evidence: &EstablishedContactEvidenceV1,
        now_s: f64,
    ) -> Result<(), ContactAuthorityError> {
        if !evidence.validate() {
            return Err(ContactAuthorityError::InvalidEvidence);
        }
        if evidence.site != self.site {
            return Err(ContactAuthorityError::EvidenceSiteMismatch);
        }
        self.evidence_fresh_at(evidence, now_s)
    }

    fn evidence_fresh_at(
        &self,
        evidence: &EstablishedContactEvidenceV1,
        now_s: f64,
    ) -> Result<(), ContactAuthorityError> {
        if !valid_time(now_s) {
            return Err(ContactAuthorityError::InvalidTime);
        }
        let future = evidence.sampled_at_s - now_s;
        if future > self.policy.maximum_future_skew_s {
            return Err(ContactAuthorityError::EvidenceTooFarInFuture);
        }
        let age = now_s - evidence.sampled_at_s;
        if age > self.policy.maximum_evidence_age_s {
            return Err(ContactAuthorityError::EvidenceTooOld);
        }
        Ok(())
    }

    fn validate_transition_time(&self, now_s: f64) -> Result<(), ContactAuthorityError> {
        if !valid_time(now_s) {
            return Err(ContactAuthorityError::InvalidTime);
        }
        if now_s < self.last_transition_at_s {
            return Err(ContactAuthorityError::TimeRegression);
        }
        Ok(())
    }

    fn transition(
        &mut self,
        to: ContactAuthorityModeV1,
        kind: ContactAuthorityTransitionKindV1,
        now_s: f64,
        reason_id: &str,
        evidence_lineage_id: Option<String>,
        clear_evidence: bool,
    ) -> Result<ContactAuthorityTransitionV1, ContactAuthorityError> {
        self.validate_transition_time(now_s)?;
        if !nonempty(reason_id) {
            return Err(ContactAuthorityError::MissingReasonIdentity);
        }
        let sequence = self
            .transition_sequence
            .checked_add(1)
            .ok_or(ContactAuthorityError::SequenceOverflow)?;
        let from = self.mode;
        self.mode = to;
        self.transition_sequence = sequence;
        self.last_transition_at_s = now_s;
        self.last_reason_id.clear();
        self.last_reason_id.push_str(reason_id);
        if clear_evidence {
            self.established_evidence = None;
        }
        Ok(ContactAuthorityTransitionV1 {
            sequence,
            establishment_epoch: self.establishment_epoch,
            site: self.site,
            from,
            to,
            kind,
            at_s: now_s,
            policy_id: self.policy_id(),
            reason_id: reason_id.to_string(),
            evidence_lineage_id,
        })
    }
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_time(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> ContactAuthorityPolicyV1 {
        ContactAuthorityPolicyV1 {
            maximum_evidence_age_s: 0.050,
            maximum_future_skew_s: 0.002,
        }
    }

    fn evidence(sampled_at_s: f64) -> EstablishedContactEvidenceV1 {
        EstablishedContactEvidenceV1 {
            site: ContactSite::RightFoot,
            model_id: "model-a".into(),
            sampled_at_s,
            geometry_id: "geometry-a".into(),
            interaction_id: "interaction-a".into(),
            support_normal_policy_id: "normal-policy-a".into(),
            support_normal_evidence_id: "normal-evidence-a".into(),
            support_normal_verification_id: "normal-verify-a".into(),
            contact_acceleration_evidence_id: "bias-a".into(),
            source_class: ContactEvidenceClassV1::SimulatorDerived,
        }
    }

    fn acquiring(now_s: f64) -> ContactAuthorityStateMachineV1 {
        let mut state = ContactAuthorityStateMachineV1::new(
            ContactSite::RightFoot,
            policy(),
            now_s,
        )
        .unwrap();
        state.schedule(now_s, "plan-1").unwrap();
        state.begin_acquisition(now_s, "acquire-1").unwrap();
        state
    }

    fn established(now_s: f64) -> ContactAuthorityStateMachineV1 {
        let mut state = acquiring(now_s);
        state
            .establish(now_s, "evidence-established", evidence(now_s))
            .unwrap();
        state
    }

    #[test]
    fn scheduled_and_acquiring_never_grant_surface_authority() {
        let mut state = ContactAuthorityStateMachineV1::new(
            ContactSite::RightFoot,
            policy(),
            1.0,
        )
        .unwrap();
        state.schedule(1.0, "plan-1").unwrap();
        assert_eq!(state.mode(), ContactAuthorityModeV1::Scheduled);
        assert!(!state.surface_wrench_eligible_at(1.0));
        state.begin_acquisition(1.0, "acquire-1").unwrap();
        assert_eq!(state.mode(), ContactAuthorityModeV1::Acquiring);
        assert!(!state.surface_wrench_eligible_at(1.0));
    }

    #[test]
    fn cannot_skip_acquisition_to_established() {
        let mut inactive = ContactAuthorityStateMachineV1::new(
            ContactSite::RightFoot,
            policy(),
            1.0,
        )
        .unwrap();
        assert_eq!(
            inactive.establish(1.0, "bad", evidence(1.0)),
            Err(ContactAuthorityError::IllegalTransition)
        );
        inactive.schedule(1.0, "plan-1").unwrap();
        assert_eq!(
            inactive.establish(1.0, "bad", evidence(1.0)),
            Err(ContactAuthorityError::IllegalTransition)
        );
    }

    #[test]
    fn incomplete_stale_future_and_cross_site_evidence_fail_closed() {
        let mut state = acquiring(1.0);
        let mut empty = evidence(1.0);
        empty.geometry_id.clear();
        assert_eq!(
            state.establish(1.0, "empty", empty),
            Err(ContactAuthorityError::InvalidEvidence)
        );

        let mut stale = acquiring(1.0);
        assert_eq!(
            stale.establish(1.0, "stale", evidence(0.90)),
            Err(ContactAuthorityError::EvidenceTooOld)
        );

        let mut future = acquiring(1.0);
        assert_eq!(
            future.establish(1.0, "future", evidence(1.01)),
            Err(ContactAuthorityError::EvidenceTooFarInFuture)
        );

        let mut wrong_site = acquiring(1.0);
        let mut left = evidence(1.0);
        left.site = ContactSite::LeftFoot;
        assert_eq!(
            wrong_site.establish(1.0, "wrong-site", left),
            Err(ContactAuthorityError::EvidenceSiteMismatch)
        );
    }

    #[test]
    fn established_requires_complete_fresh_evidence() {
        let mut state = acquiring(1.0);
        let transition = state
            .establish(1.0, "evidence-established", evidence(1.0))
            .unwrap();
        assert_eq!(transition.to, ContactAuthorityModeV1::Established);
        assert_eq!(transition.establishment_epoch, 1);
        assert_eq!(state.establishment_epoch(), 1);
        assert_eq!(state.last_reason_id(), "evidence-established");
        assert!(state.surface_wrench_eligible_at(1.0));
        assert_eq!(
            state.established_evidence().unwrap().source_class(),
            ContactEvidenceClassV1::SimulatorDerived
        );
    }

    #[test]
    fn authority_does_not_exist_before_the_establish_transition_time() {
        let state = established(1.0);
        assert!(!state.surface_wrench_eligible_at(0.999));
        assert!(state.surface_wrench_eligible_at(1.0));
    }

    #[test]
    fn evidence_lineage_binds_the_exact_sample_time() {
        let first = evidence(1.0).lineage_id().unwrap();
        let second = evidence(1.001).lineage_id().unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn release_and_invalidation_revoke_authority_immediately() {
        let mut releasing = established(1.0);
        releasing.begin_release(1.01, "release-1").unwrap();
        assert_eq!(releasing.mode(), ContactAuthorityModeV1::Releasing);
        assert!(!releasing.surface_wrench_eligible_at(1.01));
        assert!(releasing.established_evidence().is_none());

        let mut lost = established(1.0);
        lost.invalidate(1.01, "contact-lost").unwrap();
        assert_eq!(lost.mode(), ContactAuthorityModeV1::Lost);
        assert!(!lost.surface_wrench_eligible_at(1.01));
        assert!(lost.established_evidence().is_none());
    }

    #[test]
    fn reconcile_time_revokes_evidence_on_the_first_stale_update() {
        let mut state = established(1.0);
        assert!(state.reconcile_time(1.049).unwrap().is_none());
        let transition = state.reconcile_time(1.051).unwrap().unwrap();
        assert_eq!(
            transition.kind,
            ContactAuthorityTransitionKindV1::EvidenceExpired
        );
        assert_eq!(state.mode(), ContactAuthorityModeV1::Lost);
        assert!(!state.surface_wrench_eligible_at(1.051));
    }

    #[test]
    fn refresh_rejects_static_or_epistemic_lineage_drift() {
        let mut state = established(1.0);

        let mut cross_model = evidence(1.01);
        cross_model.model_id = "model-b".into();
        assert_eq!(
            state.refresh_established(1.01, "refresh", cross_model),
            Err(ContactAuthorityError::RefreshModelMismatch)
        );

        let mut cross_geometry = evidence(1.01);
        cross_geometry.geometry_id = "geometry-b".into();
        assert_eq!(
            state.refresh_established(1.01, "refresh", cross_geometry),
            Err(ContactAuthorityError::RefreshGeometryMismatch)
        );

        let mut cross_policy = evidence(1.01);
        cross_policy.support_normal_policy_id = "normal-policy-b".into();
        assert_eq!(
            state.refresh_established(1.01, "refresh", cross_policy),
            Err(ContactAuthorityError::RefreshSupportNormalPolicyMismatch)
        );

        let mut cross_class = evidence(1.01);
        cross_class.source_class = ContactEvidenceClassV1::HardwareMeasured;
        assert_eq!(
            state.refresh_established(1.01, "refresh", cross_class),
            Err(ContactAuthorityError::RefreshEvidenceClassMismatch)
        );
    }

    #[test]
    fn refresh_allows_current_dynamic_evidence_to_advance_within_epoch() {
        let mut state = established(1.0);
        let epoch = state.establishment_epoch();
        let mut refreshed = evidence(1.01);
        refreshed.interaction_id = "interaction-b".into();
        refreshed.support_normal_evidence_id = "normal-evidence-b".into();
        refreshed.support_normal_verification_id = "normal-verify-b".into();
        refreshed.contact_acceleration_evidence_id = "bias-b".into();
        let transition = state
            .refresh_established(1.01, "fresh-frame", refreshed.clone())
            .unwrap();
        assert_eq!(
            transition.kind,
            ContactAuthorityTransitionKindV1::RefreshEstablished
        );
        assert_eq!(state.establishment_epoch(), epoch);
        assert_eq!(
            state.established_evidence().unwrap().interaction_id(),
            refreshed.interaction_id()
        );
        assert!(state.surface_wrench_eligible_at(1.01));
    }

    #[test]
    fn new_acquisition_increments_epoch_and_sequence() {
        let mut state = established(1.0);
        let first_epoch = state.establishment_epoch();
        let first_sequence = state.transition_sequence();
        state.invalidate(1.01, "lost").unwrap();
        state.reset_lost(1.01, "reset").unwrap();
        state.schedule(1.01, "plan-2").unwrap();
        state.begin_acquisition(1.01, "acquire-2").unwrap();
        let transition = state
            .establish(1.01, "re-established", evidence(1.01))
            .unwrap();
        assert_eq!(state.establishment_epoch(), first_epoch + 1);
        assert!(state.transition_sequence() > first_sequence);
        assert_eq!(transition.establishment_epoch, state.establishment_epoch());
    }

    #[test]
    fn lost_requires_explicit_reset_before_reacquisition() {
        let mut state = established(1.0);
        state.invalidate(1.01, "lost").unwrap();
        assert_eq!(
            state.schedule(1.01, "plan-2"),
            Err(ContactAuthorityError::IllegalTransition)
        );
        state.reset_lost(1.01, "operator-reset").unwrap();
        state.schedule(1.01, "plan-2").unwrap();
        assert_eq!(state.mode(), ContactAuthorityModeV1::Scheduled);
    }

    #[test]
    fn freshness_policy_has_no_hidden_identity() {
        let first = ContactAuthorityPolicyV1 {
            maximum_evidence_age_s: 0.05,
            maximum_future_skew_s: 0.002,
        };
        let second = ContactAuthorityPolicyV1 {
            maximum_evidence_age_s: 0.06,
            maximum_future_skew_s: 0.002,
        };
        assert_ne!(first.policy_id(), second.policy_id());
        assert!(
            !ContactAuthorityPolicyV1 {
                maximum_evidence_age_s: 0.0,
                maximum_future_skew_s: 0.0,
            }
            .validate()
        );
    }

    #[test]
    fn transition_time_cannot_move_backward() {
        let mut state = ContactAuthorityStateMachineV1::new(
            ContactSite::RightFoot,
            policy(),
            1.0,
        )
        .unwrap();
        assert_eq!(
            state.schedule(0.99, "plan-1"),
            Err(ContactAuthorityError::TimeRegression)
        );
    }
}
