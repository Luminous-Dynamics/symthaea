// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Promotion-facing recursive validation for measured Grasp qualification evidence.
//!
//! Individual evidence objects are intentionally useful in isolation, but any future
//! stage-promotion/import boundary must revalidate the complete lineage rather than
//! trusting matching cached digests. This module provides that single fail-closed
//! entry point without granting motor authority or enabling ObjectContact lowering.

use crate::grasp_contact_evidence::HumanoidGraspContactPolicy;
use crate::grasp_controller_qualification::{
    HumanoidGraspControllerCandidate, HumanoidGraspControllerQualificationPolicy,
};
use crate::grasp_measurement_coverage::{
    HumanoidGraspMeasurementCoveragePolicy, HumanoidGraspQualificationMeasurementChain,
    HumanoidMeasuredGraspControllerQualificationTrial,
};
use crate::grasp_retention_evidence::HumanoidGraspRetentionPolicy;
use crate::qualification::HumanoidQualificationSubject;
use crate::types::HumanoidTask;

/// Revalidate one complete measured-controller qualification lineage.
///
/// This is the required validation shape for any future serialization/import or
/// Simulation -> HIL -> Physical promotion layer. It deliberately accepts every
/// upstream semantic object instead of only their digests so each layer can
/// independently recompute and prove its own identity.
#[allow(clippy::too_many_arguments)]
pub fn validate_complete_measured_grasp_qualification(
    subject: &HumanoidQualificationSubject,
    candidate: &HumanoidGraspControllerCandidate,
    controller_policy: &HumanoidGraspControllerQualificationPolicy,
    contact_policy: &HumanoidGraspContactPolicy,
    retention_policy: &HumanoidGraspRetentionPolicy,
    measurement_chain: &HumanoidGraspQualificationMeasurementChain,
    measurement_policy: &HumanoidGraspMeasurementCoveragePolicy,
    measured_trial: &HumanoidMeasuredGraspControllerQualificationTrial,
) -> bool {
    let required_purpose = controller_policy.required_execution_purpose();

    subject.validate()
        && subject.task == HumanoidTask::Grasp
        && candidate.validate()
        && controller_policy.validate_for(subject, candidate, contact_policy, retention_policy)
        && contact_policy.validate_for(subject)
        && retention_policy.validate_for(subject, contact_policy)
        && measurement_chain.validate()
        // Promotion/import must independently prove the measurement environment
        // corresponds to the exact campaign purpose; constructor history is not
        // accepted as evidence of this relation.
        && measurement_chain.kind().admits(required_purpose)
        && measurement_policy.validate_for(
            subject,
            candidate,
            controller_policy,
            contact_policy,
            retention_policy,
            measurement_chain,
        )
        && measured_trial.trial().validate_for(subject, controller_policy)
        && measured_trial.trial().execution_purpose() == required_purpose
        && measured_trial.measurement().validate_for(
            measured_trial.trial(),
            measurement_policy,
            measurement_chain,
        )
        && measured_trial.validate_for(measurement_policy, measurement_chain)
}

/// Stronger predicate for evidence that is both structurally valid and accepted.
///
/// Structural validity and qualification success remain separate facts. Promotion
/// code should generally require this predicate; audit/import tools may instead
/// call `validate_complete_measured_grasp_qualification` to inspect valid failures.
#[allow(clippy::too_many_arguments)]
pub fn complete_measured_grasp_qualification_accepted(
    subject: &HumanoidQualificationSubject,
    candidate: &HumanoidGraspControllerCandidate,
    controller_policy: &HumanoidGraspControllerQualificationPolicy,
    contact_policy: &HumanoidGraspContactPolicy,
    retention_policy: &HumanoidGraspRetentionPolicy,
    measurement_chain: &HumanoidGraspQualificationMeasurementChain,
    measurement_policy: &HumanoidGraspMeasurementCoveragePolicy,
    measured_trial: &HumanoidMeasuredGraspControllerQualificationTrial,
) -> bool {
    validate_complete_measured_grasp_qualification(
        subject,
        candidate,
        controller_policy,
        contact_policy,
        retention_policy,
        measurement_chain,
        measurement_policy,
        measured_trial,
    ) && measured_trial.qualification_evidence_accepted()
}
