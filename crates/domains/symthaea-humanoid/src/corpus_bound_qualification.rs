// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Corpus-bound positive humanoid qualification.
//!
//! A strong content commitment is not useful if it authenticates an arbitrarily
//! weak test policy. This module therefore joins three facts in one verifier-owned
//! transition:
//!
//! 1. the exact embodied-control corpus is committed with BLAKE3;
//! 2. the certificate is recomputed from that same corpus and criteria;
//! 3. the criteria are at least as strict as the current motor-authority default
//!    certification floor before current release/subject qualification is run.
//!
//! The result remains process-local and non-Serde. This is still internal
//! qualification evidence, not legal/product safety certification.

use crate::embodied_certification::{
    EmbodiedCertificationCriteria, EmbodiedControlCase, certify_embodied_control,
};
use crate::embodied_corpus_commitment::{
    EmbodiedControlCorpusCommitmentError, EmbodiedControlCorpusCommitmentV1,
    commit_embodied_control_corpus,
};
use crate::execution::HumanoidAuthorityEnvelope;
use crate::qualification::{HumanoidQualificationEnvelope, HumanoidQualificationSubject};
use crate::release_pipeline::{HumanoidReleasePipelinePolicy, HumanoidReleasePipelineReport};
use crate::verified_qualification::{
    VerifiedHumanoidQualification, verify_current_humanoid_qualification,
};

#[derive(Debug)]
pub struct CorpusBoundVerifiedHumanoidQualification {
    verified: VerifiedHumanoidQualification,
    corpus: EmbodiedControlCorpusCommitmentV1,
    criteria: EmbodiedCertificationCriteria,
}

impl CorpusBoundVerifiedHumanoidQualification {
    pub fn qualification(&self) -> &VerifiedHumanoidQualification {
        &self.verified
    }

    pub const fn corpus_commitment(&self) -> EmbodiedControlCorpusCommitmentV1 {
        self.corpus
    }

    pub const fn criteria(&self) -> EmbodiedCertificationCriteria {
        self.criteria
    }

    pub fn restrict_authority(
        &self,
        authority: HumanoidAuthorityEnvelope,
    ) -> HumanoidAuthorityEnvelope {
        self.verified.restrict_authority(authority)
    }
}

#[derive(Debug)]
pub enum CorpusBoundQualificationError {
    CriteriaBelowMotorAuthorityFloor,
    Corpus(EmbodiedControlCorpusCommitmentError),
    Qualification(HumanoidQualificationEnvelope),
}

/// Require a criteria set that is valid and no weaker than the current default
/// motor-authority certification floor. Stricter thresholds are allowed.
pub fn criteria_meet_motor_authority_floor(
    criteria: EmbodiedCertificationCriteria,
) -> bool {
    let floor = EmbodiedCertificationCriteria::default();
    let unit_interval = |value: f64| value.is_finite() && (0.0..=1.0).contains(&value);
    let non_negative = |value: f64| value.is_finite() && value >= 0.0;

    if !unit_interval(criteria.maximum_fallback_rate)
        || !unit_interval(criteria.maximum_budget_miss_rate)
        || !unit_interval(criteria.maximum_fall_rate)
        || !unit_interval(criteria.minimum_recovery_rate)
        || !non_negative(criteria.maximum_terrain_height_std_m)
        || !non_negative(criteria.maximum_terrain_evidence_age_s)
    {
        return false;
    }

    criteria.maximum_fallback_rate <= floor.maximum_fallback_rate
        && criteria.maximum_budget_miss_rate <= floor.maximum_budget_miss_rate
        && criteria.maximum_fall_rate <= floor.maximum_fall_rate
        && criteria.minimum_recovery_rate >= floor.minimum_recovery_rate
        && criteria.maximum_terrain_height_std_m <= floor.maximum_terrain_height_std_m
        && criteria.maximum_terrain_evidence_age_s <= floor.maximum_terrain_evidence_age_s
        && (!floor.require_solver_derived_case || criteria.require_solver_derived_case)
        && (!floor.require_floating_base_case || criteria.require_floating_base_case)
        && (!floor.require_upper_body_contact_case || criteria.require_upper_body_contact_case)
}

/// Recompute certification from the exact committed corpus and issue a positive
/// process-local witness only when the criteria meet the motor-authority floor
/// and the ordinary current qualification verifier also succeeds.
pub fn verify_current_humanoid_qualification_from_corpus(
    subject: &HumanoidQualificationSubject,
    policy: &HumanoidReleasePipelinePolicy,
    release: &HumanoidReleasePipelineReport,
    cases: &[EmbodiedControlCase],
    criteria: EmbodiedCertificationCriteria,
    now_unix_millis: u64,
) -> Result<CorpusBoundVerifiedHumanoidQualification, CorpusBoundQualificationError> {
    if !criteria_meet_motor_authority_floor(criteria) {
        return Err(CorpusBoundQualificationError::CriteriaBelowMotorAuthorityFloor);
    }

    let corpus = commit_embodied_control_corpus(cases, criteria)
        .map_err(CorpusBoundQualificationError::Corpus)?;
    let certificate = certify_embodied_control(cases, criteria);
    let verified = verify_current_humanoid_qualification(
        subject,
        policy,
        release,
        &certificate,
        now_unix_millis,
    )
    .map_err(CorpusBoundQualificationError::Qualification)?;

    Ok(CorpusBoundVerifiedHumanoidQualification {
        verified,
        corpus,
        criteria,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::full_dynamics::DynamicsFidelity;
    use crate::hierarchical::HierarchicalHumanoidController;
    use crate::morphology::HumanoidMorphology;
    use crate::qualification::MOTOR_AUTHORITY_REQUIRED_STAGES;
    use crate::release_pipeline::{ReleaseStageEvidence, evaluate_release_pipeline};
    use crate::types::{ActuationMode, HumanoidCommand, HumanoidState, HumanoidTask};

    fn subject() -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dmc21,
            HumanoidTask::Stand,
            ActuationMode::NormalizedTorque,
            "corpus-bound-test-v1",
        )
    }

    fn release_for(
        subject: &HumanoidQualificationSubject,
        generated_at: u64,
    ) -> HumanoidReleasePipelineReport {
        let fingerprint = subject.fingerprint();
        let stages = MOTOR_AUTHORITY_REQUIRED_STAGES
            .iter()
            .copied()
            .enumerate()
            .map(|(index, stage)| ReleaseStageEvidence {
                stage,
                artifact_id: format!("artifact-{index}"),
                artifact_sha256: format!("{:064x}", index + 1),
                producer_id: format!("producer-{index}"),
                passed: true,
                generated_unix_millis: generated_at,
                subject_fingerprint: fingerprint,
            })
            .collect();
        evaluate_release_pipeline(
            "corpus-bound-release-1",
            &HumanoidReleasePipelinePolicy::default(),
            stages,
            generated_at,
        )
    }

    fn qualified_case(subject: &HumanoidQualificationSubject) -> EmbodiedControlCase {
        let state = HumanoidState::default_for(HumanoidMorphology::Dmc21);
        let zero = HumanoidCommand::zero();
        let (_, mut report) = HierarchicalHumanoidController::new(HumanoidMorphology::Dmc21)
            .synthesize(HumanoidTask::Stand, &state, &zero, &zero, 1.0, 0.0);

        // Construct a deterministic synthetic qualification fixture satisfying
        // the current default evidence categories. These are test facts only,
        // not physical safety claims or production thresholds.
        report.inverse_dynamics_fallback = false;
        report.contact_dynamics_fallback = false;
        report.contact_solver_budget_missed = false;
        report.floating_base_model_available = true;
        report.floating_base_dynamics_converged = true;
        report.floating_base_dynamics_fallback = false;
        report.floating_base_solver_budget_missed = false;

        EmbodiedControlCase {
            scenario_id: "qualified-case".into(),
            subject_fingerprint: subject.fingerprint(),
            dynamics_fidelity: DynamicsFidelity::SolverDerived,
            active_contacts: 3,
            terrain_height_std_m: 0.01,
            terrain_evidence_age_s: 0.01,
            report,
            fell: false,
            recovered: false,
        }
    }

    #[test]
    fn default_floor_corpus_produces_bound_positive_witness() {
        let subject = subject();
        let policy = HumanoidReleasePipelinePolicy::default();
        let release = release_for(&subject, 1_000);
        let cases = vec![qualified_case(&subject)];
        let verified = verify_current_humanoid_qualification_from_corpus(
            &subject,
            &policy,
            &release,
            &cases,
            EmbodiedCertificationCriteria::default(),
            1_000,
        )
        .unwrap();
        assert!(verified.corpus_commitment().validate());
        assert!(verified.qualification().audit_record().qualified);
        assert_eq!(verified.corpus_commitment().case_count, 1);
    }

    #[test]
    fn weaker_criteria_cannot_produce_corpus_bound_witness() {
        let subject = subject();
        let policy = HumanoidReleasePipelinePolicy::default();
        let release = release_for(&subject, 1_000);
        let cases = vec![qualified_case(&subject)];
        let mut weak = EmbodiedCertificationCriteria::default();
        weak.maximum_fall_rate += 0.10;
        assert!(matches!(
            verify_current_humanoid_qualification_from_corpus(
                &subject,
                &policy,
                &release,
                &cases,
                weak,
                1_000,
            ),
            Err(CorpusBoundQualificationError::CriteriaBelowMotorAuthorityFloor)
        ));
    }

    #[test]
    fn materially_changed_corpus_gets_new_bound_identity() {
        let subject = subject();
        let policy = HumanoidReleasePipelinePolicy::default();
        let release = release_for(&subject, 1_000);
        let first_cases = vec![qualified_case(&subject)];
        let mut second_cases = vec![qualified_case(&subject)];
        second_cases[0].terrain_evidence_age_s = 0.02;

        let first = verify_current_humanoid_qualification_from_corpus(
            &subject,
            &policy,
            &release,
            &first_cases,
            EmbodiedCertificationCriteria::default(),
            1_000,
        )
        .unwrap();
        let second = verify_current_humanoid_qualification_from_corpus(
            &subject,
            &policy,
            &release,
            &second_cases,
            EmbodiedCertificationCriteria::default(),
            1_000,
        )
        .unwrap();
        assert_ne!(
            first.corpus_commitment().digest,
            second.corpus_commitment().digest
        );
    }

    #[test]
    fn stale_release_evidence_cannot_be_rescued_by_strong_corpus_identity() {
        let subject = subject();
        let policy = HumanoidReleasePipelinePolicy::default();
        let release = release_for(&subject, 1_000);
        let cases = vec![qualified_case(&subject)];
        let stale_now = 1_000 + policy.maximum_evidence_age_millis + 1;
        assert!(matches!(
            verify_current_humanoid_qualification_from_corpus(
                &subject,
                &policy,
                &release,
                &cases,
                EmbodiedCertificationCriteria::default(),
                stale_now,
            ),
            Err(CorpusBoundQualificationError::Qualification(_))
        ));
    }
}
