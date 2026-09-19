// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Prepared mapping from independently verified D5B evidence into D4-shaped identities.
//!
//! HUM-WRENCH-001D5B3 remains pre-authority. It verifies the sealed D5B token,
//! derives the exact static and dynamic identities that a later D5C adapter may
//! map into D4, and emits another sealed read-only descriptor. It does not
//! construct `EstablishedContactEvidenceV1`, call a contact-authority state
//! machine, or touch inverse dynamics / a QP.

use serde::Serialize;

use crate::contact_authority::ContactEvidenceClassV1;
use crate::mujoco_complete_contact_evidence::CompleteContactEstablishmentEvidenceV1;
use crate::mujoco_complete_contact_verifier::{
    CompleteContactEvidenceVerificationError, CompleteContactEvidenceVerificationV1,
    verify_complete_contact_evidence_v1,
};
use crate::multi_contact::ContactSite;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreparedContactAuthorityEvidenceError {
    IndependentVerificationFailed(CompleteContactEvidenceVerificationError),
    VerificationNotConfirmed,
    SourceClassMismatch,
    MissingIdentity,
    InvalidResult,
}

/// Sealed pre-authority descriptor for the exact evidence fields D5C will map
/// into D4. Private fields and no `Deserialize` prevent callers from filling a
/// D4-shaped string bundle without passing the typed D5B/D5B2 chain.
#[derive(Debug, Clone, Serialize)]
pub struct PreparedContactAuthorityEvidenceV1 {
    site: ContactSite,
    sampled_at_s: f64,
    model_id: String,
    model_signature: u64,
    physical_geom_name: String,
    physical_geom_id: usize,
    support_site_name: String,
    support_site_id: usize,
    immutable_geometry_id: String,
    model_subject_id: String,
    contact_frame_id: String,
    interaction_id: String,
    support_normal_policy_id: String,
    support_normal_evidence_id: String,
    support_normal_verification_id: String,
    contact_kinematics_evidence_id: String,
    complete_evidence_id: String,
    verification: CompleteContactEvidenceVerificationV1,
    prepared_lineage_id: String,
    source_class: ContactEvidenceClassV1,
}

impl PreparedContactAuthorityEvidenceV1 {
    pub const fn site(&self) -> ContactSite {
        self.site
    }

    pub const fn sampled_at_s(&self) -> f64 {
        self.sampled_at_s
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub const fn model_signature(&self) -> u64 {
        self.model_signature
    }

    pub fn physical_geom_name(&self) -> &str {
        &self.physical_geom_name
    }

    pub const fn physical_geom_id(&self) -> usize {
        self.physical_geom_id
    }

    pub fn support_site_name(&self) -> &str {
        &self.support_site_name
    }

    pub const fn support_site_id(&self) -> usize {
        self.support_site_id
    }

    pub fn immutable_geometry_id(&self) -> &str {
        &self.immutable_geometry_id
    }

    pub fn model_subject_id(&self) -> &str {
        &self.model_subject_id
    }

    pub fn contact_frame_id(&self) -> &str {
        &self.contact_frame_id
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

    pub fn contact_kinematics_evidence_id(&self) -> &str {
        &self.contact_kinematics_evidence_id
    }

    pub fn complete_evidence_id(&self) -> &str {
        &self.complete_evidence_id
    }

    pub fn verification(&self) -> &CompleteContactEvidenceVerificationV1 {
        &self.verification
    }

    pub fn prepared_lineage_id(&self) -> &str {
        &self.prepared_lineage_id
    }

    pub const fn source_class(&self) -> ContactEvidenceClassV1 {
        self.source_class
    }

    fn validate(&self) -> bool {
        self.sampled_at_s.is_finite()
            && self.sampled_at_s >= 0.0
            && self.source_class == ContactEvidenceClassV1::SimulatorDerived
            && self.verification.verified()
            && nonempty(&self.model_id)
            && nonempty(&self.physical_geom_name)
            && nonempty(&self.support_site_name)
            && nonempty(&self.immutable_geometry_id)
            && nonempty(&self.model_subject_id)
            && nonempty(&self.contact_frame_id)
            && nonempty(&self.interaction_id)
            && nonempty(&self.support_normal_policy_id)
            && nonempty(&self.support_normal_evidence_id)
            && nonempty(&self.support_normal_verification_id)
            && nonempty(&self.contact_kinematics_evidence_id)
            && nonempty(&self.complete_evidence_id)
            && nonempty(&self.prepared_lineage_id)
            && self.model_subject_id
                == model_subject_id(&self.model_id, self.model_signature)
            && self.contact_frame_id
                == contact_frame_id(
                    self.site,
                    &self.physical_geom_name,
                    self.physical_geom_id,
                    &self.support_site_name,
                    self.support_site_id,
                    &self.immutable_geometry_id,
                )
            && self.verification.model_signature == self.model_signature
            && self.verification.model_id == self.model_id
            && self.verification.sampled_at_s.to_bits() == self.sampled_at_s.to_bits()
            && self.verification.support_site_name == self.support_site_name
            && self.verification.recomputed_contact_kinematics_lineage_id
                == self.contact_kinematics_evidence_id
            && self.verification.recomputed_complete_lineage_id == self.complete_evidence_id
            && self.prepared_lineage_id
                == prepared_lineage_id(
                    self.site,
                    self.sampled_at_s,
                    &self.model_subject_id,
                    &self.contact_frame_id,
                    &self.interaction_id,
                    &self.support_normal_policy_id,
                    &self.support_normal_evidence_id,
                    &self.support_normal_verification_id,
                    &self.contact_kinematics_evidence_id,
                    &self.complete_evidence_id,
                    self.source_class,
                )
    }
}

/// Prepare a sealed D4-shaped evidence descriptor only after independent D5B2
/// verification succeeds. This function does not construct or transition D4.
#[allow(dead_code, reason = "HUM-WRENCH-001D5C will become the first non-test consumer")]
pub(crate) fn prepare_contact_authority_evidence_v1(
    evidence: &CompleteContactEstablishmentEvidenceV1,
) -> Result<PreparedContactAuthorityEvidenceV1, PreparedContactAuthorityEvidenceError> {
    let verification = verify_complete_contact_evidence_v1(evidence)
        .map_err(PreparedContactAuthorityEvidenceError::IndependentVerificationFailed)?;
    if !verification.verified() {
        return Err(PreparedContactAuthorityEvidenceError::VerificationNotConfirmed);
    }
    if evidence.source_class() != ContactEvidenceClassV1::SimulatorDerived {
        return Err(PreparedContactAuthorityEvidenceError::SourceClassMismatch);
    }

    let support = evidence.support();
    let immutable_geometry_id = support.placed_patch().geometry.geometry_id.clone();
    let interaction_id = support.interaction_limits().interaction_id.clone();
    let support_normal_policy_id = support.support_normal_policy_id().to_string();
    let support_normal_evidence_id = support.support_normal_evidence_lineage_id().to_string();
    let support_normal_verification_id =
        support.support_normal_verification_lineage_id().to_string();
    let contact_kinematics_evidence_id = evidence.contact_kinematics_lineage_id().to_string();
    let complete_evidence_id = evidence.complete_lineage_id().to_string();

    if [
        immutable_geometry_id.as_str(),
        interaction_id.as_str(),
        support_normal_policy_id.as_str(),
        support_normal_evidence_id.as_str(),
        support_normal_verification_id.as_str(),
        contact_kinematics_evidence_id.as_str(),
        complete_evidence_id.as_str(),
    ]
    .iter()
    .any(|value| !nonempty(value))
    {
        return Err(PreparedContactAuthorityEvidenceError::MissingIdentity);
    }

    let model_subject_id = model_subject_id(evidence.model_id(), evidence.model_signature());
    let contact_frame_id = contact_frame_id(
        evidence.site(),
        evidence.physical_geom_name(),
        evidence.physical_geom_id(),
        evidence.support_site_name(),
        evidence.support_site_id(),
        &immutable_geometry_id,
    );
    let prepared_lineage_id = prepared_lineage_id(
        evidence.site(),
        evidence.sampled_at_s(),
        &model_subject_id,
        &contact_frame_id,
        &interaction_id,
        &support_normal_policy_id,
        &support_normal_evidence_id,
        &support_normal_verification_id,
        &contact_kinematics_evidence_id,
        &complete_evidence_id,
        evidence.source_class(),
    );

    let result = PreparedContactAuthorityEvidenceV1 {
        site: evidence.site(),
        sampled_at_s: evidence.sampled_at_s(),
        model_id: evidence.model_id().to_string(),
        model_signature: evidence.model_signature(),
        physical_geom_name: evidence.physical_geom_name().to_string(),
        physical_geom_id: evidence.physical_geom_id(),
        support_site_name: evidence.support_site_name().to_string(),
        support_site_id: evidence.support_site_id(),
        immutable_geometry_id,
        model_subject_id,
        contact_frame_id,
        interaction_id,
        support_normal_policy_id,
        support_normal_evidence_id,
        support_normal_verification_id,
        contact_kinematics_evidence_id,
        complete_evidence_id,
        verification,
        prepared_lineage_id,
        source_class: ContactEvidenceClassV1::SimulatorDerived,
    };
    if result.validate() {
        Ok(result)
    } else {
        Err(PreparedContactAuthorityEvidenceError::InvalidResult)
    }
}

fn model_subject_id(model_id: &str, model_signature: u64) -> String {
    format!(
        "mujoco-model-subject-v1:model:{}:signature:{model_signature:016x}",
        component(model_id),
    )
}

fn contact_frame_id(
    site: ContactSite,
    physical_geom_name: &str,
    physical_geom_id: usize,
    support_site_name: &str,
    support_site_id: usize,
    immutable_geometry_id: &str,
) -> String {
    format!(
        "mujoco-contact-frame-v1:site:{site:?}:physical-geom:{}:{}:support-site:{}:{}:geometry:{}",
        component(physical_geom_name),
        physical_geom_id,
        component(support_site_name),
        support_site_id,
        component(immutable_geometry_id),
    )
}

#[allow(clippy::too_many_arguments)]
fn prepared_lineage_id(
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
        component(model_subject_id),
        component(contact_frame_id),
        component(interaction_id),
        component(support_normal_policy_id),
        component(support_normal_evidence_id),
        component(support_normal_verification_id),
        component(contact_kinematics_evidence_id),
        component(complete_evidence_id),
    )
}

fn component(value: &str) -> String {
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
    use crate::mujoco_support_normal::{SupportNormalPolicyV1, qualify_support_normals};
    use crate::mujoco_verified_support::bind_verified_mujoco_surface_support_v1;
    use crate::simulator::{HumanoidPhysicsSimulator, MuJoCoHumanoidSimulator};
    use crate::types::HumanoidCommand;

    fn live_complete_evidence() -> CompleteContactEstablishmentEvidenceV1 {
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
        bind_complete_mujoco_contact_evidence_v1(&support, &bias).unwrap()
    }

    #[test]
    fn valid_independently_verified_token_prepares_d4_mapping() {
        let evidence = live_complete_evidence();
        let prepared = prepare_contact_authority_evidence_v1(&evidence).unwrap();
        assert_eq!(prepared.site(), evidence.site());
        assert_eq!(prepared.sampled_at_s().to_bits(), evidence.sampled_at_s().to_bits());
        assert_eq!(prepared.model_id(), evidence.model_id());
        assert_eq!(prepared.model_signature(), evidence.model_signature());
        assert_eq!(prepared.physical_geom_name(), evidence.physical_geom_name());
        assert_eq!(prepared.physical_geom_id(), evidence.physical_geom_id());
        assert_eq!(prepared.support_site_name(), evidence.support_site_name());
        assert_eq!(prepared.support_site_id(), evidence.support_site_id());
        assert_eq!(
            prepared.contact_kinematics_evidence_id(),
            evidence.contact_kinematics_lineage_id()
        );
        assert_eq!(prepared.complete_evidence_id(), evidence.complete_lineage_id());
        assert!(prepared.verification().verified());
        assert_eq!(prepared.source_class(), ContactEvidenceClassV1::SimulatorDerived);
    }

    #[test]
    fn static_model_subject_binds_model_and_signature() {
        let base = model_subject_id("model-a", 7);
        assert_ne!(base, model_subject_id("model-b", 7));
        assert_ne!(base, model_subject_id("model-a", 8));
    }

    #[test]
    fn static_contact_frame_binds_concrete_mapping_and_geometry() {
        let base = contact_frame_id(ContactSite::RightFoot, "r_foot_g", 3, "r_foot_site", 5, "geom-a");
        assert_ne!(base, contact_frame_id(ContactSite::LeftFoot, "r_foot_g", 3, "r_foot_site", 5, "geom-a"));
        assert_ne!(base, contact_frame_id(ContactSite::RightFoot, "other", 3, "r_foot_site", 5, "geom-a"));
        assert_ne!(base, contact_frame_id(ContactSite::RightFoot, "r_foot_g", 4, "r_foot_site", 5, "geom-a"));
        assert_ne!(base, contact_frame_id(ContactSite::RightFoot, "r_foot_g", 3, "other-site", 5, "geom-a"));
        assert_ne!(base, contact_frame_id(ContactSite::RightFoot, "r_foot_g", 3, "r_foot_site", 6, "geom-a"));
        assert_ne!(base, contact_frame_id(ContactSite::RightFoot, "r_foot_g", 3, "r_foot_site", 5, "geom-b"));
    }

    #[test]
    fn dynamic_evidence_does_not_change_static_ids() {
        let model_static = model_subject_id("model-a", 7);
        let frame_static = contact_frame_id(ContactSite::RightFoot, "r_foot_g", 3, "r_foot_site", 5, "geom-a");
        let first = prepared_lineage_id(
            ContactSite::RightFoot,
            1.0,
            &model_static,
            &frame_static,
            "interaction-a",
            "policy-a",
            "normal-a",
            "verify-a",
            "kin-a",
            "complete-a",
            ContactEvidenceClassV1::SimulatorDerived,
        );
        let second = prepared_lineage_id(
            ContactSite::RightFoot,
            1.1,
            &model_static,
            &frame_static,
            "interaction-b",
            "policy-a",
            "normal-b",
            "verify-b",
            "kin-b",
            "complete-b",
            ContactEvidenceClassV1::SimulatorDerived,
        );
        assert_ne!(first, second);
        assert_eq!(model_static, model_subject_id("model-a", 7));
        assert_eq!(
            frame_static,
            contact_frame_id(ContactSite::RightFoot, "r_foot_g", 3, "r_foot_site", 5, "geom-a")
        );
    }
}
