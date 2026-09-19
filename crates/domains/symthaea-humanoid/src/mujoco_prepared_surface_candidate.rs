// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Final sealed MuJoCo surface-contact candidate before D4 runtime authority.
//!
//! HUM-WRENCH-001D5B5 keeps the full typed D5B contact evidence together with
//! its exact D5B3 prepared authority mapping. Minting reruns both independent
//! D5B2 and D5B4 verifiers and then cross-links every static and dynamic subject
//! identity. The result is still pre-authority: it does not construct D4
//! Established evidence, inspect/mutate a contact-authority state machine, or
//! invoke inverse dynamics / a QP.

use serde::Serialize;

use crate::contact_authority::ContactEvidenceClassV1;
use crate::mujoco_complete_contact_evidence::CompleteContactEstablishmentEvidenceV1;
use crate::mujoco_complete_contact_verifier::{
    CompleteContactEvidenceVerificationError, CompleteContactEvidenceVerificationV1,
    verify_complete_contact_evidence_v1,
};
use crate::mujoco_prepared_contact_authority::PreparedContactAuthorityEvidenceV1;
use crate::mujoco_prepared_contact_verifier::{
    PreparedContactAuthorityVerificationError, PreparedContactAuthorityVerificationV1,
    verify_prepared_contact_authority_evidence_v1,
};
use crate::multi_contact::ContactSite;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreparedSurfaceContactCandidateError {
    CompleteVerificationFailed(CompleteContactEvidenceVerificationError),
    PreparedVerificationFailed(PreparedContactAuthorityVerificationError),
    CompleteVerificationNotConfirmed,
    PreparedVerificationNotConfirmed,
    SiteMismatch,
    SampleTimeMismatch,
    ModelIdentityMismatch,
    ModelSignatureMismatch,
    PhysicalGeomNameMismatch,
    PhysicalGeomIdMismatch,
    SupportSiteNameMismatch,
    SupportSiteIdMismatch,
    ImmutableGeometryMismatch,
    InteractionIdentityMismatch,
    SupportNormalPolicyMismatch,
    SupportNormalEvidenceMismatch,
    SupportNormalVerificationMismatch,
    ContactKinematicsMismatch,
    CompleteEvidenceMismatch,
    SourceClassMismatch,
    InvalidResult,
}

/// Full typed pre-authority surface-contact proposition.
///
/// Fields are private and this type intentionally does not implement
/// `Deserialize`. External callers therefore cannot pair unrelated typed and
/// prepared evidence or replay a serialized object as a future authority input.
#[derive(Debug, Clone, Serialize)]
pub struct PreparedSurfaceContactCandidateV1 {
    complete: CompleteContactEstablishmentEvidenceV1,
    prepared: PreparedContactAuthorityEvidenceV1,
    complete_verification: CompleteContactEvidenceVerificationV1,
    prepared_verification: PreparedContactAuthorityVerificationV1,
    candidate_lineage_id: String,
    source_class: ContactEvidenceClassV1,
}

impl PreparedSurfaceContactCandidateV1 {
    pub fn complete(&self) -> &CompleteContactEstablishmentEvidenceV1 {
        &self.complete
    }

    pub fn prepared(&self) -> &PreparedContactAuthorityEvidenceV1 {
        &self.prepared
    }

    pub fn complete_verification(&self) -> &CompleteContactEvidenceVerificationV1 {
        &self.complete_verification
    }

    pub fn prepared_verification(&self) -> &PreparedContactAuthorityVerificationV1 {
        &self.prepared_verification
    }

    pub fn candidate_lineage_id(&self) -> &str {
        &self.candidate_lineage_id
    }

    pub const fn source_class(&self) -> ContactEvidenceClassV1 {
        self.source_class
    }

    fn validate(&self) -> bool {
        self.source_class == ContactEvidenceClassV1::SimulatorDerived
            && self.complete_verification.verified()
            && self.prepared_verification.verified()
            && validate_link_view(&CandidateLinkView::from_pair(&self.complete, &self.prepared))
                .is_ok()
            && self.candidate_lineage_id
                == candidate_lineage_id(
                    &self.complete,
                    &self.prepared,
                    &self.complete_verification,
                    &self.prepared_verification,
                    self.source_class,
                )
    }
}

/// Seal one exact typed D5B + prepared D5B3 pair only after independently
/// re-verifying both representations and cross-linking every retained identity.
#[allow(dead_code, reason = "HUM-WRENCH-001D5C will become the first non-test consumer")]
pub(crate) fn seal_prepared_surface_contact_candidate_v1(
    complete: &CompleteContactEstablishmentEvidenceV1,
    prepared: &PreparedContactAuthorityEvidenceV1,
) -> Result<PreparedSurfaceContactCandidateV1, PreparedSurfaceContactCandidateError> {
    let complete_verification = verify_complete_contact_evidence_v1(complete)
        .map_err(PreparedSurfaceContactCandidateError::CompleteVerificationFailed)?;
    if !complete_verification.verified() {
        return Err(PreparedSurfaceContactCandidateError::CompleteVerificationNotConfirmed);
    }

    let prepared_verification = verify_prepared_contact_authority_evidence_v1(prepared)
        .map_err(PreparedSurfaceContactCandidateError::PreparedVerificationFailed)?;
    if !prepared_verification.verified() {
        return Err(PreparedSurfaceContactCandidateError::PreparedVerificationNotConfirmed);
    }

    validate_link_view(&CandidateLinkView::from_pair(complete, prepared))?;

    let source_class = ContactEvidenceClassV1::SimulatorDerived;
    let candidate_lineage_id = candidate_lineage_id(
        complete,
        prepared,
        &complete_verification,
        &prepared_verification,
        source_class,
    );
    let result = PreparedSurfaceContactCandidateV1 {
        complete: complete.clone(),
        prepared: prepared.clone(),
        complete_verification,
        prepared_verification,
        candidate_lineage_id,
        source_class,
    };
    if result.validate() {
        Ok(result)
    } else {
        Err(PreparedSurfaceContactCandidateError::InvalidResult)
    }
}

#[derive(Debug, Clone)]
struct CandidateLinkView {
    complete_site: ContactSite,
    prepared_site: ContactSite,
    complete_sample_bits: u64,
    prepared_sample_bits: u64,
    complete_model_id: String,
    prepared_model_id: String,
    complete_model_signature: u64,
    prepared_model_signature: u64,
    complete_physical_geom_name: String,
    prepared_physical_geom_name: String,
    complete_physical_geom_id: usize,
    prepared_physical_geom_id: usize,
    complete_support_site_name: String,
    prepared_support_site_name: String,
    complete_support_site_id: usize,
    prepared_support_site_id: usize,
    complete_immutable_geometry_id: String,
    prepared_immutable_geometry_id: String,
    complete_interaction_id: String,
    prepared_interaction_id: String,
    complete_support_normal_policy_id: String,
    prepared_support_normal_policy_id: String,
    complete_support_normal_evidence_id: String,
    prepared_support_normal_evidence_id: String,
    complete_support_normal_verification_id: String,
    prepared_support_normal_verification_id: String,
    complete_contact_kinematics_id: String,
    prepared_contact_kinematics_id: String,
    complete_evidence_id: String,
    prepared_complete_evidence_id: String,
    complete_source_class: ContactEvidenceClassV1,
    prepared_source_class: ContactEvidenceClassV1,
}

impl CandidateLinkView {
    fn from_pair(
        complete: &CompleteContactEstablishmentEvidenceV1,
        prepared: &PreparedContactAuthorityEvidenceV1,
    ) -> Self {
        let support = complete.support();
        Self {
            complete_site: complete.site(),
            prepared_site: prepared.site(),
            complete_sample_bits: complete.sampled_at_s().to_bits(),
            prepared_sample_bits: prepared.sampled_at_s().to_bits(),
            complete_model_id: complete.model_id().to_string(),
            prepared_model_id: prepared.model_id().to_string(),
            complete_model_signature: complete.model_signature(),
            prepared_model_signature: prepared.model_signature(),
            complete_physical_geom_name: complete.physical_geom_name().to_string(),
            prepared_physical_geom_name: prepared.physical_geom_name().to_string(),
            complete_physical_geom_id: complete.physical_geom_id(),
            prepared_physical_geom_id: prepared.physical_geom_id(),
            complete_support_site_name: complete.support_site_name().to_string(),
            prepared_support_site_name: prepared.support_site_name().to_string(),
            complete_support_site_id: complete.support_site_id(),
            prepared_support_site_id: prepared.support_site_id(),
            complete_immutable_geometry_id: support.placed_patch().geometry.geometry_id.clone(),
            prepared_immutable_geometry_id: prepared.immutable_geometry_id().to_string(),
            complete_interaction_id: support.interaction_limits().interaction_id.clone(),
            prepared_interaction_id: prepared.interaction_id().to_string(),
            complete_support_normal_policy_id: support.support_normal_policy_id().to_string(),
            prepared_support_normal_policy_id: prepared.support_normal_policy_id().to_string(),
            complete_support_normal_evidence_id: support
                .support_normal_evidence_lineage_id()
                .to_string(),
            prepared_support_normal_evidence_id: prepared
                .support_normal_evidence_id()
                .to_string(),
            complete_support_normal_verification_id: support
                .support_normal_verification_lineage_id()
                .to_string(),
            prepared_support_normal_verification_id: prepared
                .support_normal_verification_id()
                .to_string(),
            complete_contact_kinematics_id: complete
                .contact_kinematics_lineage_id()
                .to_string(),
            prepared_contact_kinematics_id: prepared
                .contact_kinematics_evidence_id()
                .to_string(),
            complete_evidence_id: complete.complete_lineage_id().to_string(),
            prepared_complete_evidence_id: prepared.complete_evidence_id().to_string(),
            complete_source_class: complete.source_class(),
            prepared_source_class: prepared.source_class(),
        }
    }
}

fn validate_link_view(
    view: &CandidateLinkView,
) -> Result<(), PreparedSurfaceContactCandidateError> {
    if view.complete_site != view.prepared_site {
        return Err(PreparedSurfaceContactCandidateError::SiteMismatch);
    }
    if view.complete_sample_bits != view.prepared_sample_bits {
        return Err(PreparedSurfaceContactCandidateError::SampleTimeMismatch);
    }
    if view.complete_model_id != view.prepared_model_id {
        return Err(PreparedSurfaceContactCandidateError::ModelIdentityMismatch);
    }
    if view.complete_model_signature != view.prepared_model_signature {
        return Err(PreparedSurfaceContactCandidateError::ModelSignatureMismatch);
    }
    if view.complete_physical_geom_name != view.prepared_physical_geom_name {
        return Err(PreparedSurfaceContactCandidateError::PhysicalGeomNameMismatch);
    }
    if view.complete_physical_geom_id != view.prepared_physical_geom_id {
        return Err(PreparedSurfaceContactCandidateError::PhysicalGeomIdMismatch);
    }
    if view.complete_support_site_name != view.prepared_support_site_name {
        return Err(PreparedSurfaceContactCandidateError::SupportSiteNameMismatch);
    }
    if view.complete_support_site_id != view.prepared_support_site_id {
        return Err(PreparedSurfaceContactCandidateError::SupportSiteIdMismatch);
    }
    if view.complete_immutable_geometry_id != view.prepared_immutable_geometry_id {
        return Err(PreparedSurfaceContactCandidateError::ImmutableGeometryMismatch);
    }
    if view.complete_interaction_id != view.prepared_interaction_id {
        return Err(PreparedSurfaceContactCandidateError::InteractionIdentityMismatch);
    }
    if view.complete_support_normal_policy_id != view.prepared_support_normal_policy_id {
        return Err(PreparedSurfaceContactCandidateError::SupportNormalPolicyMismatch);
    }
    if view.complete_support_normal_evidence_id != view.prepared_support_normal_evidence_id {
        return Err(PreparedSurfaceContactCandidateError::SupportNormalEvidenceMismatch);
    }
    if view.complete_support_normal_verification_id != view.prepared_support_normal_verification_id {
        return Err(PreparedSurfaceContactCandidateError::SupportNormalVerificationMismatch);
    }
    if view.complete_contact_kinematics_id != view.prepared_contact_kinematics_id {
        return Err(PreparedSurfaceContactCandidateError::ContactKinematicsMismatch);
    }
    if view.complete_evidence_id != view.prepared_complete_evidence_id {
        return Err(PreparedSurfaceContactCandidateError::CompleteEvidenceMismatch);
    }
    if view.complete_source_class != ContactEvidenceClassV1::SimulatorDerived
        || view.prepared_source_class != ContactEvidenceClassV1::SimulatorDerived
        || view.complete_source_class != view.prepared_source_class
    {
        return Err(PreparedSurfaceContactCandidateError::SourceClassMismatch);
    }
    Ok(())
}

fn candidate_lineage_id(
    complete: &CompleteContactEstablishmentEvidenceV1,
    prepared: &PreparedContactAuthorityEvidenceV1,
    complete_verification: &CompleteContactEvidenceVerificationV1,
    prepared_verification: &PreparedContactAuthorityVerificationV1,
    source_class: ContactEvidenceClassV1,
) -> String {
    format!(
        "prepared-surface-contact-candidate-v1:site:{:?}:time-bits:{:016x}:complete:{}:prepared:{}:d5b2-kinematics:{}:d5b2-complete:{}:d5b4-model-subject:{}:d5b4-contact-frame:{}:d5b4-prepared:{}:source:{source_class:?}",
        complete.site(),
        complete.sampled_at_s().to_bits(),
        component(complete.complete_lineage_id()),
        component(prepared.prepared_lineage_id()),
        component(&complete_verification.recomputed_contact_kinematics_lineage_id),
        component(&complete_verification.recomputed_complete_lineage_id),
        component(&prepared_verification.recomputed_model_subject_id),
        component(&prepared_verification.recomputed_contact_frame_id),
        component(&prepared_verification.recomputed_prepared_lineage_id),
    )
}

fn component(value: &str) -> String {
    format!("{}:{}", value.len(), value)
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

    fn live_pair() -> (
        CompleteContactEstablishmentEvidenceV1,
        PreparedContactAuthorityEvidenceV1,
    ) {
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
        let prepared = prepare_contact_authority_evidence_v1(&complete).unwrap();
        (complete, prepared)
    }

    #[test]
    fn exact_typed_and_prepared_pair_seals() {
        let (complete, prepared) = live_pair();
        let candidate = seal_prepared_surface_contact_candidate_v1(&complete, &prepared).unwrap();
        assert_eq!(
            candidate.complete().complete_lineage_id(),
            complete.complete_lineage_id()
        );
        assert_eq!(
            candidate.prepared().prepared_lineage_id(),
            prepared.prepared_lineage_id()
        );
        assert!(candidate.complete_verification().verified());
        assert!(candidate.prepared_verification().verified());
        assert_eq!(candidate.source_class(), ContactEvidenceClassV1::SimulatorDerived);
        assert!(!candidate.candidate_lineage_id().is_empty());
        assert_eq!(
            candidate.complete().contact_jacobian(),
            complete.contact_jacobian()
        );
        assert_eq!(candidate.complete().contact_bias(), complete.contact_bias());
    }

    #[test]
    fn static_subject_mismatches_fail_closed() {
        let (complete, prepared) = live_pair();
        let base = CandidateLinkView::from_pair(&complete, &prepared);

        let mut changed = base.clone();
        changed.prepared_site = if base.complete_site == ContactSite::RightFoot {
            ContactSite::LeftFoot
        } else {
            ContactSite::RightFoot
        };
        assert_eq!(
            validate_link_view(&changed),
            Err(PreparedSurfaceContactCandidateError::SiteMismatch)
        );

        let mut changed = base.clone();
        changed.prepared_sample_bits = base.prepared_sample_bits.wrapping_add(1);
        assert_eq!(
            validate_link_view(&changed),
            Err(PreparedSurfaceContactCandidateError::SampleTimeMismatch)
        );

        let mut changed = base.clone();
        changed.prepared_model_id.push_str("-different");
        assert_eq!(
            validate_link_view(&changed),
            Err(PreparedSurfaceContactCandidateError::ModelIdentityMismatch)
        );

        let mut changed = base.clone();
        changed.prepared_model_signature = base.prepared_model_signature.wrapping_add(1);
        assert_eq!(
            validate_link_view(&changed),
            Err(PreparedSurfaceContactCandidateError::ModelSignatureMismatch)
        );

        let mut changed = base.clone();
        changed.prepared_physical_geom_name.push_str("-different");
        assert_eq!(
            validate_link_view(&changed),
            Err(PreparedSurfaceContactCandidateError::PhysicalGeomNameMismatch)
        );

        let mut changed = base.clone();
        changed.prepared_physical_geom_id = base.prepared_physical_geom_id.wrapping_add(1);
        assert_eq!(
            validate_link_view(&changed),
            Err(PreparedSurfaceContactCandidateError::PhysicalGeomIdMismatch)
        );

        let mut changed = base.clone();
        changed.prepared_support_site_name.push_str("-different");
        assert_eq!(
            validate_link_view(&changed),
            Err(PreparedSurfaceContactCandidateError::SupportSiteNameMismatch)
        );

        let mut changed = base.clone();
        changed.prepared_support_site_id = base.prepared_support_site_id.wrapping_add(1);
        assert_eq!(
            validate_link_view(&changed),
            Err(PreparedSurfaceContactCandidateError::SupportSiteIdMismatch)
        );

        let mut changed = base;
        changed.prepared_immutable_geometry_id.push_str("-different");
        assert_eq!(
            validate_link_view(&changed),
            Err(PreparedSurfaceContactCandidateError::ImmutableGeometryMismatch)
        );
    }

    #[test]
    fn dynamic_subject_mismatches_fail_closed() {
        let (complete, prepared) = live_pair();
        let base = CandidateLinkView::from_pair(&complete, &prepared);

        let mut changed = base.clone();
        changed.prepared_interaction_id.push_str("-different");
        assert_eq!(
            validate_link_view(&changed),
            Err(PreparedSurfaceContactCandidateError::InteractionIdentityMismatch)
        );

        let mut changed = base.clone();
        changed.prepared_support_normal_policy_id.push_str("-different");
        assert_eq!(
            validate_link_view(&changed),
            Err(PreparedSurfaceContactCandidateError::SupportNormalPolicyMismatch)
        );

        let mut changed = base.clone();
        changed.prepared_support_normal_evidence_id.push_str("-different");
        assert_eq!(
            validate_link_view(&changed),
            Err(PreparedSurfaceContactCandidateError::SupportNormalEvidenceMismatch)
        );

        let mut changed = base.clone();
        changed
            .prepared_support_normal_verification_id
            .push_str("-different");
        assert_eq!(
            validate_link_view(&changed),
            Err(PreparedSurfaceContactCandidateError::SupportNormalVerificationMismatch)
        );

        let mut changed = base.clone();
        changed.prepared_contact_kinematics_id.push_str("-different");
        assert_eq!(
            validate_link_view(&changed),
            Err(PreparedSurfaceContactCandidateError::ContactKinematicsMismatch)
        );

        let mut changed = base;
        changed.prepared_complete_evidence_id.push_str("-different");
        assert_eq!(
            validate_link_view(&changed),
            Err(PreparedSurfaceContactCandidateError::CompleteEvidenceMismatch)
        );
    }
}
