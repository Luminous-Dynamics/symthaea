// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Operator-approved operational Reach authority.
//!
//! Qualification protocol selection and operator authority must not be independent
//! facts. A generic operator source could otherwise be combined with a different,
//! weaker-but-valid qualification protocol. This facade requires upstream operator
//! approval evidence to bind the exact Reach subject, exact operational protocol
//! fingerprint, and exact operational scope before qualification authority may
//! cross the motor boundary.
//!
//! This module does **not** authenticate an operator or verify signatures. The
//! wrapped `HumanoidAuthoritySourceSnapshot` is still an already-established
//! upstream authority source. Xenia/Mycelix (or another authority service) should
//! own authentication and signature/revocation verification. This layer makes the
//! resulting policy/scope binding explicit and non-substitutable inside the
//! humanoid execution API.

use crate::execution_authority_scope::HumanoidScopedSkillAuthorityReceipt;
use crate::qualification::HumanoidQualificationSubject;
use crate::reach_operational_promotion::{
    HumanoidReachOperationalProtocolArtifact,
    HumanoidReachOperationalProtocolPolicy,
    issue_humanoid_reach_operational_protocol_authority_receipt,
};
use crate::skill_authority_receipt::HumanoidAuthoritySourceSnapshot;
use crate::skill_permit::HumanoidSkillExecutionPermit;

pub use crate::reach_operational_promotion::{
    HUMANOID_REACH_OPERATIONAL_PROTOCOL_ARTIFACT_SCHEMA_VERSION,
    HUMANOID_REACH_OPERATIONAL_PROTOCOL_POLICY_SCHEMA_VERSION,
    HUMANOID_REACH_PROTOCOL_EPISODE_STAGE_SCHEMA_VERSION,
    HUMANOID_REACH_PROTOCOL_STEP_STAGE_SCHEMA_VERSION,
    HumanoidReachEpisodeCampaignAssessment,
    HumanoidReachEpisodeCampaignFailureKind,
    HumanoidReachEpisodeScenarioAssessment,
    HumanoidReachEpisodeScenarioFailureKind,
    HumanoidReachEpisodeScenarioRequirement,
    HumanoidReachOperationalEpisodeCampaignPolicy,
    HumanoidReachOperationalEpisodeCase,
    HumanoidReachOperationalProtocolArtifact as HumanoidReachQualifiedOperationalProtocolArtifact,
    HumanoidReachOperationalProtocolPolicy as HumanoidReachQualifiedOperationalProtocolPolicy,
    HumanoidReachOperationalProtocolPromotionFailure,
    HumanoidReachProtocolBoundEpisodeStageArtifact,
    HumanoidReachProtocolBoundStepStageArtifact,
    HumanoidReachProtocolEpisodeStageIssueFailure,
    HumanoidReachProtocolStepStageIssueFailure,
    HumanoidReachStageProtocolRequirement,
    assess_humanoid_reach_episode_campaign,
    issue_humanoid_reach_protocol_bound_episode_stage,
    issue_humanoid_reach_protocol_bound_step_stage,
    promote_humanoid_reach_protocol_to_operational,
};

pub const HUMANOID_REACH_OPERATOR_PROTOCOL_APPROVAL_SCHEMA_VERSION: u32 = 1;

/// Upstream operator authority explicitly bound to one exact operational Reach
/// protocol and one exact operational scope.
///
/// Fields are private so callers cannot mutate the binding after construction.
/// Construction still assumes `operator_source` was authenticated/authorized by
/// an upstream authority system; this crate only verifies structural lineage and
/// freshness.
pub struct HumanoidReachOperatorProtocolApproval {
    schema_version: u32,
    approval_id: String,
    subject_fingerprint: u64,
    protocol_policy_fingerprint: u64,
    operational_scope_id: String,
    approved_at_s: f64,
    valid_until_s: f64,
    operator_source: HumanoidAuthoritySourceSnapshot,
    approval_fingerprint: u64,
}

impl std::fmt::Debug for HumanoidReachOperatorProtocolApproval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidReachOperatorProtocolApproval")
            .field("approval_id", &self.approval_id)
            .field("subject_fingerprint", &self.subject_fingerprint)
            .field("protocol_policy_fingerprint", &self.protocol_policy_fingerprint)
            .field("operational_scope_id", &self.operational_scope_id)
            .field("approved_at_s", &self.approved_at_s)
            .field("valid_until_s", &self.valid_until_s)
            .field("operator_scale", &self.operator_source.scale)
            .field("approval_fingerprint", &self.approval_fingerprint)
            .finish()
    }
}

impl HumanoidReachOperatorProtocolApproval {
    /// Bind already-established upstream operator authority to the exact policy
    /// and deployment scope. This is provenance binding, not authentication.
    #[allow(clippy::too_many_arguments)]
    pub fn bind_upstream(
        subject: &HumanoidQualificationSubject,
        policy: &HumanoidReachOperationalProtocolPolicy,
        operational_scope_id: impl Into<String>,
        approval_id: impl Into<String>,
        operator_source: HumanoidAuthoritySourceSnapshot,
        approved_at_s: f64,
        valid_until_s: f64,
    ) -> Option<Self> {
        let operational_scope_id = operational_scope_id.into();
        let approval_id = approval_id.into();
        let policy_fingerprint = policy.fingerprint();
        if !subject.validate()
            || !policy.validate_for(subject)
            || policy_fingerprint == 0
            || !valid_id(&operational_scope_id)
            || !valid_id(&approval_id)
            || !approved_at_s.is_finite()
            || approved_at_s < 0.0
            || !valid_until_s.is_finite()
            || valid_until_s < approved_at_s
            || !operator_source.validate_at(approved_at_s)
            || valid_until_s > operator_source.valid_until_s
        {
            return None;
        }

        let mut approval = Self {
            schema_version: HUMANOID_REACH_OPERATOR_PROTOCOL_APPROVAL_SCHEMA_VERSION,
            approval_id,
            subject_fingerprint: subject.fingerprint(),
            protocol_policy_fingerprint: policy_fingerprint,
            operational_scope_id,
            approved_at_s,
            valid_until_s,
            operator_source,
            approval_fingerprint: 0,
        };
        approval.approval_fingerprint = fingerprint_operator_approval(&approval);
        approval
            .validate_at(subject, policy, approval.approved_at_s)
            .then_some(approval)
    }

    pub fn approval_id(&self) -> &str {
        &self.approval_id
    }

    pub const fn subject_fingerprint(&self) -> u64 {
        self.subject_fingerprint
    }

    pub const fn protocol_policy_fingerprint(&self) -> u64 {
        self.protocol_policy_fingerprint
    }

    pub fn operational_scope_id(&self) -> &str {
        &self.operational_scope_id
    }

    pub const fn approved_at_s(&self) -> f64 {
        self.approved_at_s
    }

    pub const fn valid_until_s(&self) -> f64 {
        self.valid_until_s
    }

    pub const fn approval_fingerprint(&self) -> u64 {
        self.approval_fingerprint
    }

    pub fn validate_at(
        &self,
        subject: &HumanoidQualificationSubject,
        policy: &HumanoidReachOperationalProtocolPolicy,
        now_s: f64,
    ) -> bool {
        self.schema_version == HUMANOID_REACH_OPERATOR_PROTOCOL_APPROVAL_SCHEMA_VERSION
            && subject.validate()
            && policy.validate_for(subject)
            && self.subject_fingerprint != 0
            && self.subject_fingerprint == subject.fingerprint()
            && self.protocol_policy_fingerprint != 0
            && self.protocol_policy_fingerprint == policy.fingerprint()
            && valid_id(&self.approval_id)
            && valid_id(&self.operational_scope_id)
            && self.approved_at_s.is_finite()
            && self.approved_at_s >= 0.0
            && self.valid_until_s.is_finite()
            && self.valid_until_s >= self.approved_at_s
            && now_s.is_finite()
            && now_s >= self.approved_at_s
            && now_s <= self.valid_until_s
            && self.operator_source.validate_at(now_s)
            && self.valid_until_s <= self.operator_source.valid_until_s
            && self.approval_fingerprint != 0
            && self.approval_fingerprint == fingerprint_operator_approval(self)
    }

    fn derived_operator_source(&self, now_s: f64) -> Option<HumanoidAuthoritySourceSnapshot> {
        if !now_s.is_finite() || now_s < self.approved_at_s || now_s > self.valid_until_s {
            return None;
        }
        let evaluated_at_s = self.operator_source.evaluated_at_s.max(self.approved_at_s);
        let valid_until_s = self.operator_source.valid_until_s.min(self.valid_until_s);
        let source = HumanoidAuthoritySourceSnapshot {
            evidence_id: format!(
                "reach-operator-policy-approved:{:016x}",
                self.approval_fingerprint
            ),
            scale: self.operator_source.scale,
            evaluated_at_s,
            valid_until_s,
        };
        source.validate_at(now_s).then_some(source)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachPolicyApprovedAuthorityIssueFailure {
    InvalidQualificationArtifact,
    QualificationPolicyMismatch,
    InvalidOperatorApproval,
    DerivedOperatorSourceInvalid,
    InnerAuthorityIssue,
}

/// Preferred public operational Reach authority issuer.
///
/// The operational scope comes from the approved operator-policy binding itself;
/// there is no independent scope string to substitute at the final call site.
#[allow(clippy::too_many_arguments)]
pub fn issue_humanoid_reach_policy_approved_operational_authority_receipt(
    permit: &HumanoidSkillExecutionPermit<'_>,
    qualification: &HumanoidReachOperationalProtocolArtifact,
    policy: &HumanoidReachOperationalProtocolPolicy,
    operator_approval: &HumanoidReachOperatorProtocolApproval,
    physical: HumanoidAuthoritySourceSnapshot,
    epistemic: HumanoidAuthoritySourceSnapshot,
    cognitive: HumanoidAuthoritySourceSnapshot,
    now_s: f64,
    now_unix_millis: u64,
) -> Result<HumanoidScopedSkillAuthorityReceipt, HumanoidReachPolicyApprovedAuthorityIssueFailure> {
    if !qualification.validate_at(&qualification.subject, now_unix_millis) {
        return Err(HumanoidReachPolicyApprovedAuthorityIssueFailure::InvalidQualificationArtifact);
    }
    let policy_fingerprint = policy.fingerprint();
    if !policy.validate_for(&qualification.subject)
        || policy_fingerprint == 0
        || qualification.policy_fingerprint != policy_fingerprint
        || qualification.policy_id != policy.policy_id
    {
        return Err(HumanoidReachPolicyApprovedAuthorityIssueFailure::QualificationPolicyMismatch);
    }
    if !operator_approval.validate_at(&qualification.subject, policy, now_s) {
        return Err(HumanoidReachPolicyApprovedAuthorityIssueFailure::InvalidOperatorApproval);
    }
    let operator = operator_approval
        .derived_operator_source(now_s)
        .ok_or(HumanoidReachPolicyApprovedAuthorityIssueFailure::DerivedOperatorSourceInvalid)?;

    issue_humanoid_reach_operational_protocol_authority_receipt(
        permit,
        qualification,
        operator,
        physical,
        epistemic,
        cognitive,
        operator_approval.operational_scope_id.clone(),
        now_s,
        now_unix_millis,
    )
    .map_err(|_| HumanoidReachPolicyApprovedAuthorityIssueFailure::InnerAuthorityIssue)
}

fn fingerprint_operator_approval(approval: &HumanoidReachOperatorProtocolApproval) -> u64 {
    if approval.schema_version != HUMANOID_REACH_OPERATOR_PROTOCOL_APPROVAL_SCHEMA_VERSION
        || !valid_id(&approval.approval_id)
        || approval.subject_fingerprint == 0
        || approval.protocol_policy_fingerprint == 0
        || !valid_id(&approval.operational_scope_id)
        || !approval.approved_at_s.is_finite()
        || approval.approved_at_s < 0.0
        || !approval.valid_until_s.is_finite()
        || approval.valid_until_s < approval.approved_at_s
        || !valid_id(&approval.operator_source.evidence_id)
        || !approval.operator_source.scale.is_finite()
        || !(0.0..=1.0).contains(&approval.operator_source.scale)
        || !approval.operator_source.evaluated_at_s.is_finite()
        || !approval.operator_source.valid_until_s.is_finite()
        || approval.operator_source.valid_until_s < approval.operator_source.evaluated_at_s
        || approval.valid_until_s > approval.operator_source.valid_until_s
    {
        return 0;
    }
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    feed_u64(&mut hash, approval.schema_version as u64);
    feed_bytes(&mut hash, approval.approval_id.as_bytes());
    feed_u64(&mut hash, approval.subject_fingerprint);
    feed_u64(&mut hash, approval.protocol_policy_fingerprint);
    feed_bytes(&mut hash, approval.operational_scope_id.as_bytes());
    feed_u64(&mut hash, approval.approved_at_s.to_bits());
    feed_u64(&mut hash, approval.valid_until_s.to_bits());
    feed_bytes(&mut hash, approval.operator_source.evidence_id.as_bytes());
    feed_u64(&mut hash, approval.operator_source.scale.to_bits() as u64);
    feed_u64(&mut hash, approval.operator_source.evaluated_at_s.to_bits());
    feed_u64(&mut hash, approval.operator_source.valid_until_s.to_bits());
    if hash == 0 { 1 } else { hash }
}

fn valid_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && value.len() <= 256
        && value
            .bytes()
            .all(|byte| byte.is_ascii_graphic() && !byte.is_ascii_whitespace())
}

fn feed_u64(hash: &mut u64, value: u64) {
    for byte in value.to_le_bytes() {
        *hash ^= byte as u64;
        *hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
}

fn feed_bytes(hash: &mut u64, bytes: &[u8]) {
    feed_u64(hash, bytes.len() as u64);
    for byte in bytes {
        *hash ^= *byte as u64;
        *hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_authority_scope::HumanoidExecutionPurpose;
    use crate::morphology::HumanoidMorphology;
    use crate::reach_operational_promotion::HumanoidReachStageProtocolRequirement;
    use crate::types::{ActuationMode, HumanoidTask};

    fn subject() -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dexterous53,
            HumanoidTask::Reach,
            ActuationMode::NormalizedTorque,
            "operator-policy-binding-test-v1",
        )
    }

    fn policy() -> HumanoidReachOperationalProtocolPolicy {
        HumanoidReachOperationalProtocolPolicy {
            schema_version: HUMANOID_REACH_OPERATIONAL_PROTOCOL_POLICY_SCHEMA_VERSION,
            policy_id: "approved-policy-v1".into(),
            subject_fingerprint: subject().fingerprint(),
            simulation: HumanoidReachStageProtocolRequirement {
                purpose: HumanoidExecutionPurpose::SimulationQualification,
                step_protocol_fingerprint: 1,
                episode_protocol_fingerprint: 2,
            },
            hil: HumanoidReachStageProtocolRequirement {
                purpose: HumanoidExecutionPurpose::HilQualification,
                step_protocol_fingerprint: 3,
                episode_protocol_fingerprint: 4,
            },
            physical: HumanoidReachStageProtocolRequirement {
                purpose: HumanoidExecutionPurpose::PhysicalQualification,
                step_protocol_fingerprint: 5,
                episode_protocol_fingerprint: 6,
            },
            maximum_simulation_stage_age_millis: 10_000,
            maximum_hil_stage_age_millis: 10_000,
            maximum_physical_stage_age_millis: 10_000,
            maximum_stage_pair_skew_millis: 1_000,
            operational_artifact_validity_millis: 5_000,
        }
    }

    fn operator_source() -> HumanoidAuthoritySourceSnapshot {
        HumanoidAuthoritySourceSnapshot {
            evidence_id: "xenia-operator-grant:test-v1".into(),
            scale: 1.0,
            evaluated_at_s: 1.0,
            valid_until_s: 10.0,
        }
    }

    #[test]
    fn approval_binds_exact_policy_and_scope() {
        let approval = HumanoidReachOperatorProtocolApproval::bind_upstream(
            &subject(),
            &policy(),
            "robot-cell-a",
            "approval-a",
            operator_source(),
            2.0,
            8.0,
        )
        .unwrap();
        assert!(approval.validate_at(&subject(), &policy(), 3.0));
        assert_eq!(approval.operational_scope_id(), "robot-cell-a");
        assert_ne!(approval.approval_fingerprint(), 0);
    }

    #[test]
    fn changed_policy_invalidates_existing_approval() {
        let approval = HumanoidReachOperatorProtocolApproval::bind_upstream(
            &subject(),
            &policy(),
            "robot-cell-a",
            "approval-a",
            operator_source(),
            2.0,
            8.0,
        )
        .unwrap();
        let mut changed = policy();
        changed.maximum_stage_pair_skew_millis = 2_000;
        assert!(!approval.validate_at(&subject(), &changed, 3.0));
    }

    #[test]
    fn approval_cannot_outlive_operator_source() {
        assert!(HumanoidReachOperatorProtocolApproval::bind_upstream(
            &subject(),
            &policy(),
            "robot-cell-a",
            "approval-a",
            operator_source(),
            2.0,
            11.0,
        )
        .is_none());
    }
}
