// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent integrity verification for prepared contact-authority mapping evidence.
//!
//! HUM-WRENCH-001D5B4 deliberately does not call the D5B3 preparer or its
//! private validator. It independently re-derives the static D4 mapping
//! identities and full prepared lineage from the sealed descriptor's public
//! read-only surface. Successful verification is diagnostic evidence only; it
//! does not construct D4 Established evidence, inspect/mutate a runtime
//! authority state machine, or invoke a controller / QP.

use serde::{Deserialize, Serialize};

use crate::contact_authority::ContactEvidenceClassV1;
use crate::mujoco_complete_contact_verifier::CompleteContactEvidenceVerificationV1;
use crate::mujoco_prepared_contact_authority::PreparedContactAuthorityEvidenceV1;
use crate::multi_contact::ContactSite;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreparedContactAuthorityVerificationError {
    InvalidSubjectIdentity,
    SourceClassMismatch,
    InvalidUpstreamVerification,
    UpstreamVerificationSubjectMismatch,
    ModelSubjectMismatch,
    ContactFrameMismatch,
    PreparedLineageMismatch,
}

/// Independently re-derived integrity receipt for one sealed D5B3 descriptor.
///
/// This receipt is serializable/deserializable because it is diagnostic
/// evidence rather than a capability. Replaying it cannot establish contact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreparedContactAuthorityVerificationV1 {
    pub site: ContactSite,
    pub sampled_at_s: f64,
    pub model_id: String,
    pub model_signature: u64,
    pub support_site_name: String,
    pub source_class: ContactEvidenceClassV1,
    pub recomputed_model_subject_id: String,
    pub recomputed_contact_frame_id: String,
    pub recomputed_prepared_lineage_id: String,
    pub upstream_verification_confirmed: bool,
    pub producer_model_subject_matches: bool,
    pub producer_contact_frame_matches: bool,
    pub producer_prepared_lineage_matches: bool,
}

impl PreparedContactAuthorityVerificationV1 {
    pub const fn verified(&self) -> bool {
        self.upstream_verification_confirmed
            && self.producer_model_subject_matches
            && self.producer_contact_frame_matches
            && self.producer_prepared_lineage_matches
    }
}

/// Independently verify one sealed D5B3 prepared-authority descriptor.
pub fn verify_prepared_contact_authority_evidence_v1(
    prepared: &PreparedContactAuthorityEvidenceV1,
) -> Result<PreparedContactAuthorityVerificationV1, PreparedContactAuthorityVerificationError> {
    verify_components(
        prepared.site(),
        prepared.sampled_at_s(),
        prepared.model_id(),
        prepared.model_signature(),
        prepared.physical_geom_name(),
        prepared.physical_geom_id(),
        prepared.support_site_name(),
        prepared.support_site_id(),
        prepared.immutable_geometry_id(),
        prepared.model_subject_id(),
        prepared.contact_frame_id(),
        prepared.interaction_id(),
        prepared.support_normal_policy_id(),
        prepared.support_normal_evidence_id(),
        prepared.support_normal_verification_id(),
        prepared.contact_kinematics_evidence_id(),
        prepared.complete_evidence_id(),
        prepared.verification(),
        prepared.prepared_lineage_id(),
        prepared.source_class(),
    )
}

#[allow(clippy::too_many_arguments)]
fn verify_components(
    site: ContactSite,
    sampled_at_s: f64,
    model_id: &str,
    model_signature: u64,
    physical_geom_name: &str,
    physical_geom_id: usize,
    support_site_name: &str,
    support_site_id: usize,
    immutable_geometry_id: &str,
    producer_model_subject_id: &str,
    producer_contact_frame_id: &str,
    interaction_id: &str,
    support_normal_policy_id: &str,
    support_normal_evidence_id: &str,
    support_normal_verification_id: &str,
    contact_kinematics_evidence_id: &str,
    complete_evidence_id: &str,
    upstream_verification: &CompleteContactEvidenceVerificationV1,
    producer_prepared_lineage_id: &str,
    source_class: ContactEvidenceClassV1,
) -> Result<PreparedContactAuthorityVerificationV1, PreparedContactAuthorityVerificationError> {
    if !sampled_at_s.is_finite()
        || sampled_at_s < 0.0
        || [
            model_id,
            physical_geom_name,
            support_site_name,
            immutable_geometry_id,
            producer_model_subject_id,
            producer_contact_frame_id,
            interaction_id,
            support_normal_policy_id,
            support_normal_evidence_id,
            support_normal_verification_id,
            contact_kinematics_evidence_id,
            complete_evidence_id,
            producer_prepared_lineage_id,
        ]
        .iter()
        .any(|value| !nonempty(value))
    {
        return Err(PreparedContactAuthorityVerificationError::InvalidSubjectIdentity);
    }
    if source_class != ContactEvidenceClassV1::SimulatorDerived {
        return Err(PreparedContactAuthorityVerificationError::SourceClassMismatch);
    }
    if !upstream_verification.verified() {
        return Err(PreparedContactAuthorityVerificationError::InvalidUpstreamVerification);
    }
    if upstream_verification.model_signature != model_signature
        || upstream_verification.model_id != model_id
        || upstream_verification.sampled_at_s.to_bits() != sampled_at_s.to_bits()
        || upstream_verification.support_site_name != support_site_name
        || upstream_verification.source_class != source_class
        || upstream_verification.recomputed_contact_kinematics_lineage_id
            != contact_kinematics_evidence_id
        || upstream_verification.recomputed_complete_lineage_id != complete_evidence_id
    {
        return Err(
            PreparedContactAuthorityVerificationError::UpstreamVerificationSubjectMismatch,
        );
    }

    let recomputed_model_subject_id = independent_model_subject_id(model_id, model_signature);
    if recomputed_model_subject_id != producer_model_subject_id {
        return Err(PreparedContactAuthorityVerificationError::ModelSubjectMismatch);
    }

    let recomputed_contact_frame_id = independent_contact_frame_id(
        site,
        physical_geom_name,
        physical_geom_id,
        support_site_name,
        support_site_id,
        immutable_geometry_id,
    );
    if recomputed_contact_frame_id != producer_contact_frame_id {
        return Err(PreparedContactAuthorityVerificationError::ContactFrameMismatch);
    }

    let recomputed_prepared_lineage_id = independent_prepared_lineage_id(
        site,
        sampled_at_s,
        &recomputed_model_subject_id,
        &recomputed_contact_frame_id,
        interaction_id,
        support_normal_policy_id,
        support_normal_evidence_id,
        support_normal_verification_id,
        contact_kinematics_evidence_id,
        complete_evidence_id,
        source_class,
    );
    if recomputed_prepared_lineage_id != producer_prepared_lineage_id {
        return Err(PreparedContactAuthorityVerificationError::PreparedLineageMismatch);
    }

    Ok(PreparedContactAuthorityVerificationV1 {
        site,
        sampled_at_s,
        model_id: model_id.to_string(),
        model_signature,
        support_site_name: support_site_name.to_string(),
        source_class,
        recomputed_model_subject_id,
        recomputed_contact_frame_id,
        recomputed_prepared_lineage_id,
        upstream_verification_confirmed: true,
        producer_model_subject_matches: true,
        producer_contact_frame_matches: true,
        producer_prepared_lineage_matches: true,
    })
}

/// Independent copy of D5B3's static model-subject canonicalization.
fn independent_model_subject_id(model_id: &str, model_signature: u64) -> String {
    format!(
        "mujoco-model-subject-v1:model:{}:signature:{model_signature:016x}",
        independent_component(model_id),
    )
}

/// Independent copy of D5B3's static contact-frame canonicalization.
fn independent_contact_frame_id(
    site: ContactSite,
    physical_geom_name: &str,
    physical_geom_id: usize,
    support_site_name: &str,
    support_site_id: usize,
    immutable_geometry_id: &str,
) -> String {
    format!(
        "mujoco-contact-frame-v1:site:{site:?}:physical-geom:{}:{}:support-site:{}:{}:geometry:{}",
        independent_component(physical_geom_name),
        physical_geom_id,
        independent_component(support_site_name),
        support_site_id,
        independent_component(immutable_geometry_id),
    )
}

#[allow(clippy::too_many_arguments)]
fn independent_prepared_lineage_id(
    site: ContactSite,
    sampled_at_s: f64,
    model_subject_id: &str,
    contact_frame_id: &str,
    interaction_id: &str,
    support_normal_policy_id: &str,
    support_normal_evidence_id: &str,
    support_normal_verification_id: &str,
    contact_kinematics_evidence_id: &str,
    complete_evidence_id: &str,
    source_class: ContactEvidenceClassV1,
) -> String {
    format!(
        "prepared-contact-authority-evidence-v1:site:{site:?}:time-bits:{:016x}:model-subject:{}:contact-frame:{}:interaction:{}:normal-policy:{}:normal-evidence:{}:normal-verification:{}:contact-kinematics:{}:complete-evidence:{}:source:{source_class:?}",
        sampled_at_s.to_bits(),
        independent_component(model_subject_id),
        independent_component(contact_frame_id),
        independent_component(interaction_id),
        independent_component(support_normal_policy_id),
        independent_component(support_normal_evidence_id),
        independent_component(support_normal_verification_id),
        independent_component(contact_kinematics_evidence_id),
        independent_component(complete_evidence_id),
    )
}

fn independent_component(value: &str) -> String {
    format!("{}:{}", value.len(), value)
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::morphology::HumanoidMorphology;
    use crate::mujoco_complete_contact_evidence::bind_complete_mujoco_contact_evidence_v1;
    use crate::mujoco_contact_interaction::{
        MuJoCoContactInteractionError, extract_mujoco_contact_interaction,
    };
    use crate::mujoco_contact_kinematics::extract_verified_mujoco_contact_bias_evidence_v1;
    use crate::mujoco_contact_patch::extract_mujoco_foot_patch_set;
    use crate::mujoco_prepared_contact_authority::prepare_contact_authority_evidence_v1;
    use crate::mujoco_support_normal::{SupportNormalPolicyV1, qualify_support_normals};
    use crate::mujoco_verified_support::bind_verified_mujoco_surface_support_v1;
    use crate::simulator::{HumanoidPhysicsSimulator, MuJoCoHumanoidSimulator};
    use crate::types::HumanoidCommand;

    fn live_prepared_evidence() -> PreparedContactAuthorityEvidenceV1 {
        let mut sim = MuJoCoHumanoidSimulator::for_morphology(HumanoidMorphology::Dmc21).unwrap();
        sim.step(&HumanoidCommand::zero(), 0.0);
        let dynamics = sim.floating_base_dynamics_snapshot().unwrap();
        let dynamics_model_id = dynamics.model_id.clone();
        let model = Arc::clone(sim.model_arc());
        let patches = extract_mujoco_foot_patch_set(
            model.as_ref(),
            sim.data_mut(),
            &dynamics_model_id,
        )
        .unwrap();
        let policy = SupportNormalPolicyV1 {
            minimum_abs_alignment: 0.0,
        };

        let mut support = None;
        for site in [ContactSite::RightFoot, ContactSite::LeftFoot] {
            let interaction = match extract_mujoco_contact_interaction(
                model.as_ref(),
                sim.data_mut(),
                &patches,
                site,
            ) {
                Ok(record) => record,
                Err(MuJoCoContactInteractionError::NoActiveContact)
                | Err(MuJoCoContactInteractionError::NoActiveSupportContact) => continue,
                Err(error) => panic!("unexpected interaction extraction failure: {error:?}"),
            };
            let region = qualify_support_normals(&interaction, &patches, policy).unwrap();
            if region.surface_support_eligible() {
                support = Some(
                    bind_verified_mujoco_surface_support_v1(&patches, &interaction, &region)
                        .unwrap(),
                );
                break;
            }
        }
        let support = support.expect("expected generated DMC21 areal support");
        let bias = extract_verified_mujoco_contact_bias_evidence_v1(
            model.as_ref(),
            sim.data_mut(),
            dynamics,
        )
        .unwrap();
        let complete = bind_complete_mujoco_contact_evidence_v1(&support, &bias).unwrap();
        prepare_contact_authority_evidence_v1(&complete).unwrap()
    }

    #[test]
    fn independently_rederives_valid_prepared_mapping() {
        let prepared = live_prepared_evidence();
        let verification = verify_prepared_contact_authority_evidence_v1(&prepared).unwrap();
        assert!(verification.verified());
        assert_eq!(
            verification.recomputed_model_subject_id,
            prepared.model_subject_id()
        );
        assert_eq!(
            verification.recomputed_contact_frame_id,
            prepared.contact_frame_id()
        );
        assert_eq!(
            verification.recomputed_prepared_lineage_id,
            prepared.prepared_lineage_id()
        );
    }

    #[test]
    fn model_subject_identity_changes_with_model_id_or_signature() {
        let prepared = live_prepared_evidence();
        let base = independent_model_subject_id(prepared.model_id(), prepared.model_signature());
        assert_ne!(
            base,
            independent_model_subject_id("different-model", prepared.model_signature())
        );
        assert_ne!(
            base,
            independent_model_subject_id(
                prepared.model_id(),
                prepared.model_signature().wrapping_add(1),
            )
        );
    }

    #[test]
    fn contact_frame_identity_changes_for_every_static_mapping_component() {
        let prepared = live_prepared_evidence();
        let base = independent_contact_frame_id(
            prepared.site(),
            prepared.physical_geom_name(),
            prepared.physical_geom_id(),
            prepared.support_site_name(),
            prepared.support_site_id(),
            prepared.immutable_geometry_id(),
        );
        let other_site = match prepared.site() {
            ContactSite::RightFoot => ContactSite::LeftFoot,
            _ => ContactSite::RightFoot,
        };
        assert_ne!(
            base,
            independent_contact_frame_id(
                other_site,
                prepared.physical_geom_name(),
                prepared.physical_geom_id(),
                prepared.support_site_name(),
                prepared.support_site_id(),
                prepared.immutable_geometry_id(),
            )
        );
        assert_ne!(
            base,
            independent_contact_frame_id(
                prepared.site(),
                "different-geom",
                prepared.physical_geom_id(),
                prepared.support_site_name(),
                prepared.support_site_id(),
                prepared.immutable_geometry_id(),
            )
        );
        assert_ne!(
            base,
            independent_contact_frame_id(
                prepared.site(),
                prepared.physical_geom_name(),
                prepared.physical_geom_id().wrapping_add(1),
                prepared.support_site_name(),
                prepared.support_site_id(),
                prepared.immutable_geometry_id(),
            )
        );
        assert_ne!(
            base,
            independent_contact_frame_id(
                prepared.site(),
                prepared.physical_geom_name(),
                prepared.physical_geom_id(),
                "different-support-site",
                prepared.support_site_id(),
                prepared.immutable_geometry_id(),
            )
        );
        assert_ne!(
            base,
            independent_contact_frame_id(
                prepared.site(),
                prepared.physical_geom_name(),
                prepared.physical_geom_id(),
                prepared.support_site_name(),
                prepared.support_site_id().wrapping_add(1),
                prepared.immutable_geometry_id(),
            )
        );
        assert_ne!(
            base,
            independent_contact_frame_id(
                prepared.site(),
                prepared.physical_geom_name(),
                prepared.physical_geom_id(),
                prepared.support_site_name(),
                prepared.support_site_id(),
                "different-geometry-id",
            )
        );
    }

    #[test]
    fn dynamic_evidence_changes_only_full_prepared_identity() {
        let prepared = live_prepared_evidence();
        let model_subject = independent_model_subject_id(
            prepared.model_id(),
            prepared.model_signature(),
        );
        let contact_frame = independent_contact_frame_id(
            prepared.site(),
            prepared.physical_geom_name(),
            prepared.physical_geom_id(),
            prepared.support_site_name(),
            prepared.support_site_id(),
            prepared.immutable_geometry_id(),
        );
        let base = independent_prepared_lineage_id(
            prepared.site(),
            prepared.sampled_at_s(),
            &model_subject,
            &contact_frame,
            prepared.interaction_id(),
            prepared.support_normal_policy_id(),
            prepared.support_normal_evidence_id(),
            prepared.support_normal_verification_id(),
            prepared.contact_kinematics_evidence_id(),
            prepared.complete_evidence_id(),
            prepared.source_class(),
        );
        let changed = independent_prepared_lineage_id(
            prepared.site(),
            prepared.sampled_at_s(),
            &model_subject,
            &contact_frame,
            "different-interaction",
            prepared.support_normal_policy_id(),
            prepared.support_normal_evidence_id(),
            prepared.support_normal_verification_id(),
            prepared.contact_kinematics_evidence_id(),
            prepared.complete_evidence_id(),
            prepared.source_class(),
        );
        assert_eq!(model_subject, prepared.model_subject_id());
        assert_eq!(contact_frame, prepared.contact_frame_id());
        assert_ne!(base, changed);
    }

    #[test]
    fn mismatched_upstream_verification_and_empty_dynamic_identity_fail_closed() {
        let prepared = live_prepared_evidence();
        let mut mismatched = prepared.verification().clone();
        mismatched.model_id.push_str("-different");
        assert!(matches!(
            verify_components(
                prepared.site(),
                prepared.sampled_at_s(),
                prepared.model_id(),
                prepared.model_signature(),
                prepared.physical_geom_name(),
                prepared.physical_geom_id(),
                prepared.support_site_name(),
                prepared.support_site_id(),
                prepared.immutable_geometry_id(),
                prepared.model_subject_id(),
                prepared.contact_frame_id(),
                prepared.interaction_id(),
                prepared.support_normal_policy_id(),
                prepared.support_normal_evidence_id(),
                prepared.support_normal_verification_id(),
                prepared.contact_kinematics_evidence_id(),
                prepared.complete_evidence_id(),
                &mismatched,
                prepared.prepared_lineage_id(),
                prepared.source_class(),
            ),
            Err(PreparedContactAuthorityVerificationError::UpstreamVerificationSubjectMismatch)
        ));

        assert!(matches!(
            verify_components(
                prepared.site(),
                prepared.sampled_at_s(),
                prepared.model_id(),
                prepared.model_signature(),
                prepared.physical_geom_name(),
                prepared.physical_geom_id(),
                prepared.support_site_name(),
                prepared.support_site_id(),
                prepared.immutable_geometry_id(),
                prepared.model_subject_id(),
                prepared.contact_frame_id(),
                "",
                prepared.support_normal_policy_id(),
                prepared.support_normal_evidence_id(),
                prepared.support_normal_verification_id(),
                prepared.contact_kinematics_evidence_id(),
                prepared.complete_evidence_id(),
                prepared.verification(),
                prepared.prepared_lineage_id(),
                prepared.source_class(),
            ),
            Err(PreparedContactAuthorityVerificationError::InvalidSubjectIdentity)
        ));
    }
}
