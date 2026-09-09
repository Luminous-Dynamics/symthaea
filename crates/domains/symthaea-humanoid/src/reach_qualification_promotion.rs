// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Promotion from coverage-complete Reach qualification evidence to operational
//! qualification authority.
//!
//! The promotion chain is explicit and non-circular:
//!
//! TrialProtocol -> Simulation -> HIL -> Physical -> Operational qualification.
//!
//! Each stage is independently re-assessed from exact lineage-bound trials. A
//! public caller cannot label an arbitrary authority receipt `Operational`; the
//! final scoping constructor is crate-internal and is reached here only after a
//! valid operational qualification artifact is checked against the live permit.
//!
//! These artifacts are internal engineering qualification evidence, not legal or
//! product-safety certification.

use crate::execution_authority_scope::{
    HumanoidExecutionAuthorityScopeIssueFailure, HumanoidExecutionPurpose,
    HumanoidScopedSkillAuthorityReceipt, scope_verified_operational_authority_receipt,
};
use crate::qualification::HumanoidQualificationSubject;
use crate::reach_qualification_lineage::{
    HumanoidReachLineageBoundTrial, HumanoidReachLineageCampaignAssessment,
    HumanoidReachLineageCampaignPolicy, assess_lineage_humanoid_reach_qualification_campaign,
};
use crate::skill_authority_receipt::{
    HumanoidAuthoritySourceSnapshot, HumanoidSkillAuthorityEvidence,
    HumanoidSkillAuthorityReceiptIssueFailure, issue_humanoid_skill_authority_receipt,
};
use crate::skill_permit::HumanoidSkillExecutionPermit;
use crate::types::HumanoidTask;

pub const HUMANOID_REACH_QUALIFICATION_STAGE_ARTIFACT_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_REACH_OPERATIONAL_QUALIFICATION_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_REACH_OPERATIONAL_PROMOTION_POLICY_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachQualificationStageArtifact {
    pub schema_version: u32,
    pub subject: HumanoidQualificationSubject,
    pub subject_fingerprint: u64,
    pub execution_purpose: HumanoidExecutionPurpose,
    pub campaign_id: String,
    pub campaign_fingerprint: u64,
    pub lineage_policy_fingerprint: u64,
    pub corpus_lineage_fingerprint: u64,
    pub issued_unix_millis: u64,
    pub artifact_fingerprint: u64,
}

impl HumanoidReachQualificationStageArtifact {
    pub fn validate(&self) -> bool {
        self.schema_version == HUMANOID_REACH_QUALIFICATION_STAGE_ARTIFACT_SCHEMA_VERSION
            && self.subject.validate()
            && self.subject.task == HumanoidTask::Reach
            && self.subject_fingerprint != 0
            && self.subject_fingerprint == self.subject.fingerprint()
            && self.execution_purpose.is_qualification()
            && valid_id(&self.campaign_id)
            && self.campaign_fingerprint != 0
            && self.lineage_policy_fingerprint != 0
            && self.corpus_lineage_fingerprint != 0
            && self.issued_unix_millis != 0
            && self.artifact_fingerprint != 0
            && self.artifact_fingerprint == fingerprint_stage_artifact(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachQualificationStageIssueFailure {
    InvalidSubject,
    WrongCampaignPurpose,
    CampaignNotPromotionEligible,
    InvalidIssueTime,
    InvalidArtifactFingerprint,
}

/// Re-run a lineage campaign and issue an immutable artifact only when the exact
/// corpus is promotion eligible.
pub fn issue_humanoid_reach_qualification_stage_artifact(
    subject: &HumanoidQualificationSubject,
    policy: &HumanoidReachLineageCampaignPolicy,
    trials: &[HumanoidReachLineageBoundTrial],
    issued_unix_millis: u64,
) -> Result<HumanoidReachQualificationStageArtifact, HumanoidReachQualificationStageIssueFailure> {
    if !subject.validate() || subject.task != HumanoidTask::Reach {
        return Err(HumanoidReachQualificationStageIssueFailure::InvalidSubject);
    }
    if !policy.campaign.required_execution_purpose.is_qualification() {
        return Err(HumanoidReachQualificationStageIssueFailure::WrongCampaignPurpose);
    }
    if issued_unix_millis == 0 {
        return Err(HumanoidReachQualificationStageIssueFailure::InvalidIssueTime);
    }
    let assessment = assess_lineage_humanoid_reach_qualification_campaign(subject, policy, trials);
    if !assessment.promotion_eligible {
        return Err(HumanoidReachQualificationStageIssueFailure::CampaignNotPromotionEligible);
    }
    artifact_from_assessment(subject, policy, assessment, issued_unix_millis)
}

fn artifact_from_assessment(
    subject: &HumanoidQualificationSubject,
    policy: &HumanoidReachLineageCampaignPolicy,
    assessment: HumanoidReachLineageCampaignAssessment,
    issued_unix_millis: u64,
) -> Result<HumanoidReachQualificationStageArtifact, HumanoidReachQualificationStageIssueFailure> {
    let mut artifact = HumanoidReachQualificationStageArtifact {
        schema_version: HUMANOID_REACH_QUALIFICATION_STAGE_ARTIFACT_SCHEMA_VERSION,
        subject: subject.clone(),
        subject_fingerprint: subject.fingerprint(),
        execution_purpose: policy.campaign.required_execution_purpose,
        campaign_id: policy.campaign.campaign_id.clone(),
        campaign_fingerprint: assessment.base.campaign_fingerprint,
        lineage_policy_fingerprint: assessment.lineage_policy_fingerprint,
        corpus_lineage_fingerprint: assessment.corpus_lineage_fingerprint,
        issued_unix_millis,
        artifact_fingerprint: 0,
    };
    artifact.artifact_fingerprint = fingerprint_stage_artifact(&artifact);
    if !artifact.validate() {
        return Err(HumanoidReachQualificationStageIssueFailure::InvalidArtifactFingerprint);
    }
    Ok(artifact)
}

/// Explicit freshness/promotion policy for converting three qualification stages
/// into an operational Reach qualification artifact. There is intentionally no
/// Default implementation.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachOperationalPromotionPolicy {
    pub schema_version: u32,
    pub policy_id: String,
    pub subject_fingerprint: u64,
    pub maximum_simulation_stage_age_millis: u64,
    pub maximum_hil_stage_age_millis: u64,
    pub maximum_physical_stage_age_millis: u64,
    pub operational_artifact_validity_millis: u64,
}

impl HumanoidReachOperationalPromotionPolicy {
    pub fn validate_for(&self, subject: &HumanoidQualificationSubject) -> bool {
        self.schema_version == HUMANOID_REACH_OPERATIONAL_PROMOTION_POLICY_SCHEMA_VERSION
            && valid_id(&self.policy_id)
            && subject.validate()
            && subject.task == HumanoidTask::Reach
            && self.subject_fingerprint != 0
            && self.subject_fingerprint == subject.fingerprint()
            && self.maximum_simulation_stage_age_millis > 0
            && self.maximum_hil_stage_age_millis > 0
            && self.maximum_physical_stage_age_millis > 0
            && self.operational_artifact_validity_millis > 0
    }

    pub fn fingerprint(&self) -> u64 {
        if !valid_id(&self.policy_id)
            || self.subject_fingerprint == 0
            || self.maximum_simulation_stage_age_millis == 0
            || self.maximum_hil_stage_age_millis == 0
            || self.maximum_physical_stage_age_millis == 0
            || self.operational_artifact_validity_millis == 0
        {
            return 0;
        }
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        feed_u64(&mut hash, self.schema_version as u64);
        feed_bytes(&mut hash, self.policy_id.as_bytes());
        feed_u64(&mut hash, self.subject_fingerprint);
        feed_u64(&mut hash, self.maximum_simulation_stage_age_millis);
        feed_u64(&mut hash, self.maximum_hil_stage_age_millis);
        feed_u64(&mut hash, self.maximum_physical_stage_age_millis);
        feed_u64(&mut hash, self.operational_artifact_validity_millis);
        if hash == 0 { 1 } else { hash }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachOperationalQualificationArtifact {
    pub schema_version: u32,
    pub subject: HumanoidQualificationSubject,
    pub subject_fingerprint: u64,
    pub promotion_policy_id: String,
    pub promotion_policy_fingerprint: u64,
    pub simulation_stage_fingerprint: u64,
    pub hil_stage_fingerprint: u64,
    pub physical_stage_fingerprint: u64,
    pub issued_unix_millis: u64,
    pub valid_until_unix_millis: u64,
    pub artifact_fingerprint: u64,
}

impl HumanoidReachOperationalQualificationArtifact {
    pub fn validate_at(&self, subject: &HumanoidQualificationSubject, now_unix_millis: u64) -> bool {
        self.schema_version == HUMANOID_REACH_OPERATIONAL_QUALIFICATION_SCHEMA_VERSION
            && subject.validate()
            && subject.task == HumanoidTask::Reach
            && self.subject == *subject
            && self.subject_fingerprint == subject.fingerprint()
            && self.subject_fingerprint != 0
            && valid_id(&self.promotion_policy_id)
            && self.promotion_policy_fingerprint != 0
            && self.simulation_stage_fingerprint != 0
            && self.hil_stage_fingerprint != 0
            && self.physical_stage_fingerprint != 0
            && self.issued_unix_millis != 0
            && self.valid_until_unix_millis >= self.issued_unix_millis
            && now_unix_millis >= self.issued_unix_millis
            && now_unix_millis <= self.valid_until_unix_millis
            && self.artifact_fingerprint != 0
            && self.artifact_fingerprint == fingerprint_operational_artifact(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachOperationalPromotionFailure {
    InvalidSubject,
    InvalidPolicy,
    InvalidStageArtifact,
    StagePurposeMismatch,
    StageSubjectMismatch,
    StageOrderInvalid,
    StageEvidenceStale,
    InvalidPromotionTime,
    ExpiryOverflow,
    InvalidArtifactFingerprint,
}

pub fn promote_humanoid_reach_to_operational_qualification(
    subject: &HumanoidQualificationSubject,
    policy: &HumanoidReachOperationalPromotionPolicy,
    simulation: &HumanoidReachQualificationStageArtifact,
    hil: &HumanoidReachQualificationStageArtifact,
    physical: &HumanoidReachQualificationStageArtifact,
    now_unix_millis: u64,
) -> Result<HumanoidReachOperationalQualificationArtifact, HumanoidReachOperationalPromotionFailure> {
    if !subject.validate() || subject.task != HumanoidTask::Reach {
        return Err(HumanoidReachOperationalPromotionFailure::InvalidSubject);
    }
    if !policy.validate_for(subject) || policy.fingerprint() == 0 {
        return Err(HumanoidReachOperationalPromotionFailure::InvalidPolicy);
    }
    if now_unix_millis == 0 {
        return Err(HumanoidReachOperationalPromotionFailure::InvalidPromotionTime);
    }
    for stage in [simulation, hil, physical] {
        if !stage.validate() {
            return Err(HumanoidReachOperationalPromotionFailure::InvalidStageArtifact);
        }
        if stage.subject != *subject || stage.subject_fingerprint != subject.fingerprint() {
            return Err(HumanoidReachOperationalPromotionFailure::StageSubjectMismatch);
        }
        if stage.issued_unix_millis > now_unix_millis {
            return Err(HumanoidReachOperationalPromotionFailure::StageOrderInvalid);
        }
    }
    if simulation.execution_purpose != HumanoidExecutionPurpose::SimulationQualification
        || hil.execution_purpose != HumanoidExecutionPurpose::HilQualification
        || physical.execution_purpose != HumanoidExecutionPurpose::PhysicalQualification
    {
        return Err(HumanoidReachOperationalPromotionFailure::StagePurposeMismatch);
    }
    if !(simulation.issued_unix_millis <= hil.issued_unix_millis
        && hil.issued_unix_millis <= physical.issued_unix_millis)
    {
        return Err(HumanoidReachOperationalPromotionFailure::StageOrderInvalid);
    }
    if now_unix_millis.saturating_sub(simulation.issued_unix_millis)
        > policy.maximum_simulation_stage_age_millis
        || now_unix_millis.saturating_sub(hil.issued_unix_millis)
            > policy.maximum_hil_stage_age_millis
        || now_unix_millis.saturating_sub(physical.issued_unix_millis)
            > policy.maximum_physical_stage_age_millis
    {
        return Err(HumanoidReachOperationalPromotionFailure::StageEvidenceStale);
    }

    let valid_until_unix_millis = now_unix_millis
        .checked_add(policy.operational_artifact_validity_millis)
        .ok_or(HumanoidReachOperationalPromotionFailure::ExpiryOverflow)?;
    let mut artifact = HumanoidReachOperationalQualificationArtifact {
        schema_version: HUMANOID_REACH_OPERATIONAL_QUALIFICATION_SCHEMA_VERSION,
        subject: subject.clone(),
        subject_fingerprint: subject.fingerprint(),
        promotion_policy_id: policy.policy_id.clone(),
        promotion_policy_fingerprint: policy.fingerprint(),
        simulation_stage_fingerprint: simulation.artifact_fingerprint,
        hil_stage_fingerprint: hil.artifact_fingerprint,
        physical_stage_fingerprint: physical.artifact_fingerprint,
        issued_unix_millis: now_unix_millis,
        valid_until_unix_millis,
        artifact_fingerprint: 0,
    };
    artifact.artifact_fingerprint = fingerprint_operational_artifact(&artifact);
    if !artifact.validate_at(subject, now_unix_millis) {
        return Err(HumanoidReachOperationalPromotionFailure::InvalidArtifactFingerprint);
    }
    Ok(artifact)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachOperationalAuthorityIssueFailure {
    InvalidQualificationArtifact,
    PermitSubjectMismatch,
    TimeDomainInvalid,
    SourceValidityTooShort,
    Inner(HumanoidSkillAuthorityReceiptIssueFailure),
    Scope(HumanoidExecutionAuthorityScopeIssueFailure),
}

/// Derive an operational Reach receipt from a valid operational qualification
/// artifact plus the other four current restrictive authority sources.
///
/// `qualification=1.0` is created here, not supplied by the caller. Its local
/// validity cannot extend beyond the remaining wall-clock life of the promotion
/// artifact. The final scoped receipt is Operational + QualifiedCapability.
#[allow(clippy::too_many_arguments)]
pub fn issue_humanoid_reach_operational_authority_receipt(
    permit: &HumanoidSkillExecutionPermit<'_>,
    qualification: &HumanoidReachOperationalQualificationArtifact,
    operator: HumanoidAuthoritySourceSnapshot,
    physical: HumanoidAuthoritySourceSnapshot,
    epistemic: HumanoidAuthoritySourceSnapshot,
    cognitive: HumanoidAuthoritySourceSnapshot,
    operational_scope_id: impl Into<String>,
    now_s: f64,
    now_unix_millis: u64,
) -> Result<HumanoidScopedSkillAuthorityReceipt, HumanoidReachOperationalAuthorityIssueFailure> {
    if !qualification.validate_at(&qualification.subject, now_unix_millis) {
        return Err(HumanoidReachOperationalAuthorityIssueFailure::InvalidQualificationArtifact);
    }
    let subjects = permit
        .requirements()
        .iter()
        .map(|requirement| requirement.request.subject_fingerprint)
        .collect::<Vec<_>>();
    if subjects.as_slice() != &[qualification.subject_fingerprint]
        || permit.morphology() != qualification.subject.morphology
        || permit.actuation_mode() != qualification.subject.actuation_mode
        || permit.backend_profile_id() != qualification.subject.backend_profile_id
    {
        return Err(HumanoidReachOperationalAuthorityIssueFailure::PermitSubjectMismatch);
    }
    if !now_s.is_finite() || now_s < 0.0 || now_unix_millis == 0 {
        return Err(HumanoidReachOperationalAuthorityIssueFailure::TimeDomainInvalid);
    }
    let remaining_millis = qualification
        .valid_until_unix_millis
        .checked_sub(now_unix_millis)
        .ok_or(HumanoidReachOperationalAuthorityIssueFailure::InvalidQualificationArtifact)?;
    let remaining_s = remaining_millis as f64 / 1000.0;
    if !remaining_s.is_finite() || remaining_s <= 0.0 {
        return Err(HumanoidReachOperationalAuthorityIssueFailure::SourceValidityTooShort);
    }
    let qualification_valid_until_s = now_s + remaining_s;
    if !qualification_valid_until_s.is_finite() || qualification_valid_until_s <= now_s {
        return Err(HumanoidReachOperationalAuthorityIssueFailure::SourceValidityTooShort);
    }
    let qualification_source = HumanoidAuthoritySourceSnapshot {
        evidence_id: format!("reach-qualified:{:016x}", qualification.artifact_fingerprint),
        scale: 1.0,
        evaluated_at_s: now_s,
        valid_until_s: qualification_valid_until_s,
    };
    let evidence = HumanoidSkillAuthorityEvidence {
        operator,
        qualification: qualification_source,
        physical,
        epistemic,
        cognitive,
    };
    let inner = issue_humanoid_skill_authority_receipt(permit, evidence, now_s)
        .map_err(HumanoidReachOperationalAuthorityIssueFailure::Inner)?;
    scope_verified_operational_authority_receipt(inner, permit, operational_scope_id, now_s)
        .map_err(HumanoidReachOperationalAuthorityIssueFailure::Scope)
}

fn fingerprint_stage_artifact(artifact: &HumanoidReachQualificationStageArtifact) -> u64 {
    if artifact.schema_version != HUMANOID_REACH_QUALIFICATION_STAGE_ARTIFACT_SCHEMA_VERSION
        || artifact.subject_fingerprint == 0
        || !valid_id(&artifact.campaign_id)
        || artifact.campaign_fingerprint == 0
        || artifact.lineage_policy_fingerprint == 0
        || artifact.corpus_lineage_fingerprint == 0
        || artifact.issued_unix_millis == 0
        || !artifact.execution_purpose.is_qualification()
    {
        return 0;
    }
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    feed_u64(&mut hash, artifact.schema_version as u64);
    feed_u64(&mut hash, artifact.subject_fingerprint);
    feed_u64(&mut hash, purpose_id(artifact.execution_purpose));
    feed_bytes(&mut hash, artifact.campaign_id.as_bytes());
    feed_u64(&mut hash, artifact.campaign_fingerprint);
    feed_u64(&mut hash, artifact.lineage_policy_fingerprint);
    feed_u64(&mut hash, artifact.corpus_lineage_fingerprint);
    feed_u64(&mut hash, artifact.issued_unix_millis);
    if hash == 0 { 1 } else { hash }
}

fn fingerprint_operational_artifact(artifact: &HumanoidReachOperationalQualificationArtifact) -> u64 {
    if artifact.schema_version != HUMANOID_REACH_OPERATIONAL_QUALIFICATION_SCHEMA_VERSION
        || artifact.subject_fingerprint == 0
        || !valid_id(&artifact.promotion_policy_id)
        || artifact.promotion_policy_fingerprint == 0
        || artifact.simulation_stage_fingerprint == 0
        || artifact.hil_stage_fingerprint == 0
        || artifact.physical_stage_fingerprint == 0
        || artifact.issued_unix_millis == 0
        || artifact.valid_until_unix_millis < artifact.issued_unix_millis
    {
        return 0;
    }
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    feed_u64(&mut hash, artifact.schema_version as u64);
    feed_u64(&mut hash, artifact.subject_fingerprint);
    feed_bytes(&mut hash, artifact.promotion_policy_id.as_bytes());
    feed_u64(&mut hash, artifact.promotion_policy_fingerprint);
    feed_u64(&mut hash, artifact.simulation_stage_fingerprint);
    feed_u64(&mut hash, artifact.hil_stage_fingerprint);
    feed_u64(&mut hash, artifact.physical_stage_fingerprint);
    feed_u64(&mut hash, artifact.issued_unix_millis);
    feed_u64(&mut hash, artifact.valid_until_unix_millis);
    if hash == 0 { 1 } else { hash }
}

fn purpose_id(purpose: HumanoidExecutionPurpose) -> u64 {
    match purpose {
        HumanoidExecutionPurpose::SimulationQualification => 1,
        HumanoidExecutionPurpose::HilQualification => 2,
        HumanoidExecutionPurpose::PhysicalQualification => 3,
        HumanoidExecutionPurpose::Operational => 4,
    }
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
    use crate::morphology::HumanoidMorphology;
    use crate::types::ActuationMode;

    fn subject() -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dexterous53,
            HumanoidTask::Reach,
            ActuationMode::NormalizedTorque,
            "reach-promotion-test-v1",
        )
    }

    fn stage(purpose: HumanoidExecutionPurpose, issued: u64, fp_seed: u64) -> HumanoidReachQualificationStageArtifact {
        let mut artifact = HumanoidReachQualificationStageArtifact {
            schema_version: HUMANOID_REACH_QUALIFICATION_STAGE_ARTIFACT_SCHEMA_VERSION,
            subject: subject(),
            subject_fingerprint: subject().fingerprint(),
            execution_purpose: purpose,
            campaign_id: format!("campaign-{fp_seed}"),
            campaign_fingerprint: 100 + fp_seed,
            lineage_policy_fingerprint: 200 + fp_seed,
            corpus_lineage_fingerprint: 300 + fp_seed,
            issued_unix_millis: issued,
            artifact_fingerprint: 0,
        };
        artifact.artifact_fingerprint = fingerprint_stage_artifact(&artifact);
        artifact
    }

    fn policy() -> HumanoidReachOperationalPromotionPolicy {
        HumanoidReachOperationalPromotionPolicy {
            schema_version: HUMANOID_REACH_OPERATIONAL_PROMOTION_POLICY_SCHEMA_VERSION,
            policy_id: "reach-operational-promotion-test-v1".into(),
            subject_fingerprint: subject().fingerprint(),
            maximum_simulation_stage_age_millis: 10_000,
            maximum_hil_stage_age_millis: 10_000,
            maximum_physical_stage_age_millis: 10_000,
            operational_artifact_validity_millis: 5_000,
        }
    }

    #[test]
    fn complete_ordered_chain_can_promote() {
        let result = promote_humanoid_reach_to_operational_qualification(
            &subject(),
            &policy(),
            &stage(HumanoidExecutionPurpose::SimulationQualification, 1_000, 1),
            &stage(HumanoidExecutionPurpose::HilQualification, 2_000, 2),
            &stage(HumanoidExecutionPurpose::PhysicalQualification, 3_000, 3),
            4_000,
        );
        assert!(result.is_ok());
        let artifact = result.unwrap();
        assert!(artifact.validate_at(&subject(), 4_000));
    }

    #[test]
    fn missing_stage_semantics_cannot_promote() {
        let result = promote_humanoid_reach_to_operational_qualification(
            &subject(),
            &policy(),
            &stage(HumanoidExecutionPurpose::SimulationQualification, 1_000, 1),
            &stage(HumanoidExecutionPurpose::SimulationQualification, 2_000, 2),
            &stage(HumanoidExecutionPurpose::PhysicalQualification, 3_000, 3),
            4_000,
        );
        assert_eq!(result, Err(HumanoidReachOperationalPromotionFailure::StagePurposeMismatch));
    }

    #[test]
    fn stale_stage_cannot_promote() {
        let mut strict = policy();
        strict.maximum_simulation_stage_age_millis = 100;
        let result = promote_humanoid_reach_to_operational_qualification(
            &subject(),
            &strict,
            &stage(HumanoidExecutionPurpose::SimulationQualification, 1_000, 1),
            &stage(HumanoidExecutionPurpose::HilQualification, 3_800, 2),
            &stage(HumanoidExecutionPurpose::PhysicalQualification, 3_900, 3),
            4_000,
        );
        assert_eq!(result, Err(HumanoidReachOperationalPromotionFailure::StageEvidenceStale));
    }
}
