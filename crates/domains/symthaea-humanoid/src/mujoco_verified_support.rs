// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sealed typed binding of qualified MuJoCo surface-support evidence.
//!
//! HUM-WRENCH-001D5A binds exact 001C geometry/placement, 001D current
//! interaction evidence, 001D2 support-normal evidence, and an independently
//! re-derived 001D3 verification result into one read-only simulator-support
//! token. It deliberately contains no contact-acceleration evidence and cannot
//! establish D4 contact authority by itself.

use serde::Serialize;

use crate::contact_authority::ContactEvidenceClassV1;
use crate::contact_patch::{
    ContactPatchGeometrySource, ContactPatchPlacementSource, PlacedContactPatchV1,
};
use crate::contact_wrench::{ContactInteractionLimitSource, ContactInteractionLimitsV1};
use crate::mujoco_contact_interaction::{
    MuJoCoContactDimensionalityV1, MuJoCoContactInteractionRecordV1,
};
use crate::mujoco_contact_patch::MuJoCoFootPatchSetV1;
use crate::mujoco_support_normal::NormalQualifiedSupportRegionV1;
use crate::mujoco_support_normal_verifier::{
    SupportNormalEvidenceVerificationError, SupportNormalEvidenceVerificationV1,
    verify_support_normal_evidence_v1,
};
use crate::multi_contact::ContactSite;

const TIME_TOLERANCE_SCALE: f64 = 1.0e-9;
const AREA_EPS: f64 = 1.0e-12;

/// Sealed simulator-support evidence. Fields are private and the type is
/// serialize-only so external callers cannot deserialize or fill a support token.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VerifiedSurfaceSupportEvidenceV1 {
    site: ContactSite,
    model_id: String,
    model_signature: u64,
    sampled_at_s: f64,
    physical_geom_name: String,
    physical_geom_id: usize,
    support_site_name: String,
    support_site_id: usize,
    placed_patch: PlacedContactPatchV1,
    interaction_limits: ContactInteractionLimitsV1,
    interaction_dimensionality: MuJoCoContactDimensionalityV1,
    support_normal_policy_id: String,
    normal_qualified_hull_vertices_local_xy_m: Vec<[f64; 2]>,
    normal_qualified_area_m2: f64,
    verification: SupportNormalEvidenceVerificationV1,
    support_normal_evidence_lineage_id: String,
    support_normal_verification_lineage_id: String,
    support_lineage_id: String,
    source_class: ContactEvidenceClassV1,
}

impl VerifiedSurfaceSupportEvidenceV1 {
    pub fn site(&self) -> ContactSite {
        self.site
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn model_signature(&self) -> u64 {
        self.model_signature
    }

    pub fn sampled_at_s(&self) -> f64 {
        self.sampled_at_s
    }

    pub fn physical_geom_name(&self) -> &str {
        &self.physical_geom_name
    }

    pub fn physical_geom_id(&self) -> usize {
        self.physical_geom_id
    }

    pub fn support_site_name(&self) -> &str {
        &self.support_site_name
    }

    pub fn support_site_id(&self) -> usize {
        self.support_site_id
    }

    pub fn placed_patch(&self) -> &PlacedContactPatchV1 {
        &self.placed_patch
    }

    pub fn interaction_limits(&self) -> &ContactInteractionLimitsV1 {
        &self.interaction_limits
    }

    pub fn interaction_dimensionality(&self) -> MuJoCoContactDimensionalityV1 {
        self.interaction_dimensionality
    }

    pub fn support_normal_policy_id(&self) -> &str {
        &self.support_normal_policy_id
    }

    pub fn normal_qualified_hull_vertices_local_xy_m(&self) -> &[[f64; 2]] {
        &self.normal_qualified_hull_vertices_local_xy_m
    }

    pub fn normal_qualified_area_m2(&self) -> f64 {
        self.normal_qualified_area_m2
    }

    pub fn verification(&self) -> &SupportNormalEvidenceVerificationV1 {
        &self.verification
    }

    pub fn support_normal_evidence_lineage_id(&self) -> &str {
        &self.support_normal_evidence_lineage_id
    }

    pub fn support_normal_verification_lineage_id(&self) -> &str {
        &self.support_normal_verification_lineage_id
    }

    pub fn support_lineage_id(&self) -> &str {
        &self.support_lineage_id
    }

    pub fn source_class(&self) -> ContactEvidenceClassV1 {
        self.source_class
    }

    pub fn surface_support_eligible(&self) -> bool {
        self.verification.surface_support_eligible
            && self.normal_qualified_hull_vertices_local_xy_m.len() >= 3
            && self.normal_qualified_area_m2 > AREA_EPS
    }

    fn validate(&self) -> bool {
        self.placed_patch.validate()
            && self.interaction_limits.validate()
            && self.placed_patch.geometry.site == self.site
            && self.interaction_limits.site == self.site
            && same_sample_time(self.placed_patch.sampled_at_s, self.sampled_at_s)
            && same_sample_time(self.interaction_limits.sampled_at_s, self.sampled_at_s)
            && !self.model_id.trim().is_empty()
            && !self.physical_geom_name.trim().is_empty()
            && !self.support_site_name.trim().is_empty()
            && !self.support_normal_policy_id.trim().is_empty()
            && self.support_normal_policy_id == self.verification.policy_id
            && !self.support_normal_evidence_lineage_id.trim().is_empty()
            && !self.support_normal_verification_lineage_id.trim().is_empty()
            && !self.support_lineage_id.trim().is_empty()
            && self.normal_qualified_hull_vertices_local_xy_m.len()
                == self.verification.recomputed_hull_vertices_local_xy_m.len()
            && points_close(
                &self.normal_qualified_hull_vertices_local_xy_m,
                &self.verification.recomputed_hull_vertices_local_xy_m,
            )
            && close(
                self.normal_qualified_area_m2,
                self.verification.recomputed_area_m2,
            )
            && self.surface_support_eligible()
            && self.source_class == ContactEvidenceClassV1::SimulatorDerived
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifiedSurfaceSupportError {
    InvalidPatchSet,
    InvalidInteractionRecord,
    InvalidNormalRegion,
    IndependentVerificationFailed(SupportNormalEvidenceVerificationError),
    SiteMismatch,
    ModelMismatch,
    TimeMismatch,
    FootGeometryMismatch,
    GeometryIdentityMismatch,
    InteractionIdentityMismatch,
    ContactIndexMismatch,
    SimulatorSourceMismatch,
    DegenerateQualifiedSupport,
    InvalidResult,
}

/// Bind exact typed simulator support evidence into a sealed support token.
///
/// This is crate-internal on purpose. The token is an intermediate capability
/// for a later D5B adapter; it is not contact-establishment or QP authority.
#[allow(dead_code, reason = "HUM-WRENCH-001D5B will become the first non-test consumer")]
pub(crate) fn bind_verified_mujoco_surface_support_v1(
    patches: &MuJoCoFootPatchSetV1,
    interaction: &MuJoCoContactInteractionRecordV1,
    normal_region: &NormalQualifiedSupportRegionV1,
) -> Result<VerifiedSurfaceSupportEvidenceV1, VerifiedSurfaceSupportError> {
    if !patches.validate() {
        return Err(VerifiedSurfaceSupportError::InvalidPatchSet);
    }
    if !interaction.validate() {
        return Err(VerifiedSurfaceSupportError::InvalidInteractionRecord);
    }
    if !normal_region.validate() {
        return Err(VerifiedSurfaceSupportError::InvalidNormalRegion);
    }

    let verification = verify_support_normal_evidence_v1(normal_region)
        .map_err(VerifiedSurfaceSupportError::IndependentVerificationFailed)?;

    if !verification.surface_support_eligible {
        return Err(VerifiedSurfaceSupportError::DegenerateQualifiedSupport);
    }

    if interaction.site != normal_region.site {
        return Err(VerifiedSurfaceSupportError::SiteMismatch);
    }
    if interaction.model_id != patches.model_id
        || normal_region.model_id != patches.model_id
        || interaction.model_signature != patches.model_signature
        || normal_region.model_signature != patches.model_signature
    {
        return Err(VerifiedSurfaceSupportError::ModelMismatch);
    }
    if !same_sample_time(interaction.sampled_at_s, patches.sampled_at_s)
        || !same_sample_time(normal_region.sampled_at_s, patches.sampled_at_s)
    {
        return Err(VerifiedSurfaceSupportError::TimeMismatch);
    }

    let patch_record = patches
        .record_for_site(interaction.site)
        .ok_or(VerifiedSurfaceSupportError::SiteMismatch)?;
    if patch_record.geom_id != interaction.foot_geom_id
        || patch_record.physical_geom_name != interaction.foot_geom_name
        || patch_record.placed_patch.geometry.site != interaction.site
    {
        return Err(VerifiedSurfaceSupportError::FootGeometryMismatch);
    }
    if normal_region.source_geometry_id != patch_record.placed_patch.geometry.geometry_id {
        return Err(VerifiedSurfaceSupportError::GeometryIdentityMismatch);
    }
    if normal_region.source_interaction_id != interaction.limits.interaction_id {
        return Err(VerifiedSurfaceSupportError::InteractionIdentityMismatch);
    }
    if normal_region.source_contact_indices != interaction.contact_indices {
        return Err(VerifiedSurfaceSupportError::ContactIndexMismatch);
    }

    if patch_record.placed_patch.geometry.source != ContactPatchGeometrySource::SimulatorGeometry
        || patch_record.placed_patch.placement_source
            != ContactPatchPlacementSource::SimulatorKinematics
        || interaction.limits.source != ContactInteractionLimitSource::SimulatorContactPair
    {
        return Err(VerifiedSurfaceSupportError::SimulatorSourceMismatch);
    }

    let normal_evidence_lineage_id = normal_evidence_lineage_id(normal_region);
    let verification_lineage_id = verification_lineage_id(normal_region, &verification);
    let support_lineage_id = support_lineage_id(
        &patches.model_id,
        &patch_record.physical_geom_name,
        patch_record.geom_id,
        &patch_record.support_site_name,
        patch_record.support_site_id,
        &patch_record.placed_patch.geometry.geometry_id,
        &interaction.limits.interaction_id,
        &verification_lineage_id,
        patches.model_signature,
        interaction.site,
        interaction.sampled_at_s,
    );

    let result = VerifiedSurfaceSupportEvidenceV1 {
        site: interaction.site,
        model_id: patches.model_id.clone(),
        model_signature: patches.model_signature,
        sampled_at_s: interaction.sampled_at_s,
        physical_geom_name: patch_record.physical_geom_name.clone(),
        physical_geom_id: patch_record.geom_id,
        support_site_name: patch_record.support_site_name.clone(),
        support_site_id: patch_record.support_site_id,
        placed_patch: patch_record.placed_patch.clone(),
        interaction_limits: interaction.limits.clone(),
        interaction_dimensionality: interaction.dimensionality,
        support_normal_policy_id: normal_region.policy_id.clone(),
        normal_qualified_hull_vertices_local_xy_m: verification
            .recomputed_hull_vertices_local_xy_m
            .clone(),
        normal_qualified_area_m2: verification.recomputed_area_m2,
        verification,
        support_normal_evidence_lineage_id: normal_evidence_lineage_id,
        support_normal_verification_lineage_id: verification_lineage_id,
        support_lineage_id,
        source_class: ContactEvidenceClassV1::SimulatorDerived,
    };

    if result.validate() {
        Ok(result)
    } else {
        Err(VerifiedSurfaceSupportError::InvalidResult)
    }
}

fn normal_evidence_lineage_id(region: &NormalQualifiedSupportRegionV1) -> String {
    format!(
        "support-normal-evidence-v1:model:{}:sig:{:016x}:site:{:?}:time-bits:{:016x}:geometry:{}:interaction:{}:policy:{}:source-contacts:{}:admitted:{}:rejected:{}:hull:{}:area-bits:{:016x}",
        component(&region.model_id),
        region.model_signature,
        region.site,
        region.sampled_at_s.to_bits(),
        component(&region.source_geometry_id),
        component(&region.source_interaction_id),
        component(&region.policy_id),
        usize_list(&region.source_contact_indices),
        usize_list(&region.admitted_contact_indices),
        usize_list(&region.rejected_contact_indices),
        xy_bits_list(&region.hull_vertices_local_xy_m),
        region.area_m2.to_bits(),
    )
}

fn verification_lineage_id(
    region: &NormalQualifiedSupportRegionV1,
    verification: &SupportNormalEvidenceVerificationV1,
) -> String {
    format!(
        "support-normal-verification-v1:source:{}:policy:{}:contacts:{}:admitted:{}:rejected:{}:hull:{}:area-bits:{:016x}:surface:{}",
        component(&normal_evidence_lineage_id(region)),
        component(&verification.policy_id),
        verification.source_contact_count,
        verification.admitted_contact_count,
        verification.rejected_contact_count,
        xy_bits_list(&verification.recomputed_hull_vertices_local_xy_m),
        verification.recomputed_area_m2.to_bits(),
        verification.surface_support_eligible,
    )
}

#[allow(clippy::too_many_arguments)]
fn support_lineage_id(
    model_id: &str,
    physical_geom_name: &str,
    physical_geom_id: usize,
    support_site_name: &str,
    support_site_id: usize,
    geometry_id: &str,
    interaction_id: &str,
    verification_id: &str,
    model_signature: u64,
    site: ContactSite,
    sampled_at_s: f64,
) -> String {
    format!(
        "verified-surface-support-v1:model:{}:sig:{model_signature:016x}:site:{site:?}:time-bits:{:016x}:physical-geom:{}:{}:support-site:{}:{}:geometry:{}:interaction:{}:verification:{}",
        component(model_id),
        sampled_at_s.to_bits(),
        component(physical_geom_name),
        physical_geom_id,
        component(support_site_name),
        support_site_id,
        component(geometry_id),
        component(interaction_id),
        component(verification_id),
    )
}

fn component(value: &str) -> String {
    format!("{}:{}", value.len(), value)
}

fn usize_list(values: &[usize]) -> String {
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn xy_bits_list(values: &[[f64; 2]]) -> String {
    values
        .iter()
        .map(|[x, y]| format!("{:016x}.{:016x}", x.to_bits(), y.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

fn same_sample_time(left: f64, right: f64) -> bool {
    let tolerance = TIME_TOLERANCE_SCALE * (1.0 + left.abs().max(right.abs()));
    (left - right).abs() <= tolerance
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1.0e-12 * (1.0 + left.abs().max(right.abs()))
}

fn points_close(left: &[[f64; 2]], right: &[[f64; 2]]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| close(left[0], right[0]) && close(left[1], right[1]))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::morphology::HumanoidMorphology;
    use crate::mujoco_contact_interaction::{
        MuJoCoContactInteractionError, extract_mujoco_contact_interaction,
    };
    use crate::mujoco_contact_patch::extract_mujoco_foot_patch_set;
    use crate::mujoco_support_normal::{SupportNormalPolicyV1, qualify_support_normals};
    use crate::simulator::{HumanoidPhysicsSimulator, MuJoCoHumanoidSimulator};
    use crate::types::HumanoidCommand;

    fn live_bundle() -> (
        MuJoCoFootPatchSetV1,
        MuJoCoContactInteractionRecordV1,
        NormalQualifiedSupportRegionV1,
    ) {
        let mut sim = MuJoCoHumanoidSimulator::for_morphology(HumanoidMorphology::Dmc21).unwrap();
        sim.step(&HumanoidCommand::zero(), 0.0);
        let model = Arc::clone(sim.model_arc());
        let patches = extract_mujoco_foot_patch_set(
            model.as_ref(),
            sim.data_mut(),
            "generated-dmc21-verified-support-v1",
        )
        .unwrap();
        let policy = SupportNormalPolicyV1 {
            minimum_abs_alignment: 0.0,
        };

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
                return (patches, interaction, region);
            }
        }
        panic!("expected at least one non-degenerate generated DMC21 support region");
    }

    #[test]
    fn generated_support_chain_binds_to_sealed_simulator_token() {
        let (patches, interaction, region) = live_bundle();
        let token = bind_verified_mujoco_surface_support_v1(&patches, &interaction, &region)
            .unwrap();
        let patch_record = patches.record_for_site(interaction.site).unwrap();
        assert_eq!(token.site(), interaction.site);
        assert_eq!(token.source_class(), ContactEvidenceClassV1::SimulatorDerived);
        assert!(token.surface_support_eligible());
        assert_eq!(token.physical_geom_name(), patch_record.physical_geom_name);
        assert_eq!(token.physical_geom_id(), patch_record.geom_id);
        assert_eq!(token.support_site_name(), patch_record.support_site_name);
        assert_eq!(token.support_site_id(), patch_record.support_site_id);
        assert_eq!(
            token.interaction_limits().interaction_id,
            interaction.limits.interaction_id
        );
        assert_eq!(
            token.placed_patch().geometry.geometry_id,
            region.source_geometry_id
        );
        assert_eq!(token.support_normal_policy_id(), token.verification().policy_id);
    }

    #[test]
    fn model_geometry_interaction_and_contact_index_mismatch_fail_closed() {
        let (patches, interaction, region) = live_bundle();

        let mut wrong_model = region.clone();
        wrong_model.model_signature ^= 1;
        assert_eq!(
            bind_verified_mujoco_surface_support_v1(&patches, &interaction, &wrong_model),
            Err(VerifiedSurfaceSupportError::ModelMismatch)
        );

        let mut wrong_geometry = region.clone();
        wrong_geometry.source_geometry_id.push_str("-other");
        assert_eq!(
            bind_verified_mujoco_surface_support_v1(&patches, &interaction, &wrong_geometry),
            Err(VerifiedSurfaceSupportError::GeometryIdentityMismatch)
        );

        let mut wrong_interaction = region.clone();
        wrong_interaction.source_interaction_id.push_str("-other");
        assert_eq!(
            bind_verified_mujoco_surface_support_v1(&patches, &interaction, &wrong_interaction),
            Err(VerifiedSurfaceSupportError::InteractionIdentityMismatch)
        );

        let mut changed_interaction = interaction.clone();
        changed_interaction.contact_indices[0] += 1000;
        changed_interaction.contact_indices.sort_unstable();
        assert!(changed_interaction.validate());
        assert_eq!(
            bind_verified_mujoco_surface_support_v1(&patches, &changed_interaction, &region),
            Err(VerifiedSurfaceSupportError::ContactIndexMismatch)
        );
    }

    #[test]
    fn tampered_normal_evidence_is_rejected_by_independent_verifier() {
        let (patches, interaction, mut region) = live_bundle();
        region.area_m2 += 0.01;
        assert!(matches!(
            bind_verified_mujoco_surface_support_v1(&patches, &interaction, &region),
            Err(VerifiedSurfaceSupportError::IndependentVerificationFailed(_))
        ));
    }

    #[test]
    fn strong_support_requires_non_degenerate_verified_hull() {
        let verification = SupportNormalEvidenceVerificationV1 {
            policy_id: "fixture".into(),
            source_contact_count: 2,
            admitted_contact_count: 2,
            rejected_contact_count: 0,
            recomputed_hull_vertices_local_xy_m: vec![[0.0, 0.0], [1.0, 0.0]],
            recomputed_area_m2: 0.0,
            surface_support_eligible: false,
        };
        assert!(!verification.surface_support_eligible);
        assert!(verification.recomputed_hull_vertices_local_xy_m.len() < 3);
    }

    #[test]
    fn lineage_binds_exact_hull_shape() {
        // Both rectangles have area 2 m²; the lineage still differs by exact vertices.
        let first = vec![[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]];
        let second = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0]];
        assert_ne!(xy_bits_list(&first), xy_bits_list(&second));
    }

    #[test]
    fn lineage_changes_with_time_interaction_policy_and_support_site_identity() {
        let (patches, interaction, region) = live_bundle();
        let verification = verify_support_normal_evidence_v1(&region).unwrap();
        let patch_record = patches.record_for_site(interaction.site).unwrap();
        let make = |time: f64, interaction_id: &str, policy_region: &NormalQualifiedSupportRegionV1, support_site_name: &str| {
            let mut verification_for_policy = verification.clone();
            verification_for_policy.policy_id = policy_region.policy_id.clone();
            support_lineage_id(
                &patches.model_id,
                &patch_record.physical_geom_name,
                patch_record.geom_id,
                support_site_name,
                patch_record.support_site_id,
                &region.source_geometry_id,
                interaction_id,
                &verification_lineage_id(policy_region, &verification_for_policy),
                patches.model_signature,
                interaction.site,
                time,
            )
        };
        let base = make(
            interaction.sampled_at_s,
            &interaction.limits.interaction_id,
            &region,
            &patch_record.support_site_name,
        );
        let different_time = make(
            interaction.sampled_at_s + 0.001,
            &interaction.limits.interaction_id,
            &region,
            &patch_record.support_site_name,
        );
        let different_interaction = make(
            interaction.sampled_at_s,
            "another-interaction",
            &region,
            &patch_record.support_site_name,
        );
        let mut different_policy_region = region.clone();
        different_policy_region.policy_id.push_str("-other");
        let different_policy = make(
            interaction.sampled_at_s,
            &interaction.limits.interaction_id,
            &different_policy_region,
            &patch_record.support_site_name,
        );
        let different_support_site = make(
            interaction.sampled_at_s,
            &interaction.limits.interaction_id,
            &region,
            "different-support-site",
        );
        assert_ne!(base, different_time);
        assert_ne!(base, different_interaction);
        assert_ne!(base, different_policy);
        assert_ne!(base, different_support_site);
    }
}
