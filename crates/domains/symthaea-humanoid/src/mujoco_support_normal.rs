// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bearing support-normal admissibility for MuJoCo plantar contacts.
//!
//! HUM-WRENCH-001D establishes which live contacts are geometrically plantar and
//! what contact-pair limits MuJoCo resolved. This module adds a distinct policy:
//! which of those contact reaction axes are sufficiently aligned with the exact
//! sole normal to contribute to an areal support/COP region.

use serde::{Deserialize, Serialize};

use crate::mujoco_contact_interaction::MuJoCoContactInteractionRecordV1;
use crate::mujoco_contact_patch::MuJoCoFootPatchSetV1;
use crate::multi_contact::ContactSite;

const NORMAL_EPS: f64 = 1.0e-12;
const REGION_AREA_EPS: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SupportNormalPolicyV1 {
    /// Minimum absolute cosine alignment between MuJoCo contact normal and sole normal.
    pub minimum_abs_alignment: f64,
}

impl SupportNormalPolicyV1 {
    pub fn validate(&self) -> bool {
        self.minimum_abs_alignment.is_finite()
            && (0.0..=1.0).contains(&self.minimum_abs_alignment)
    }

    /// Deterministic policy identity: changing the threshold changes the subject.
    pub fn policy_id(&self) -> Option<String> {
        self.validate().then(|| {
            format!(
                "support-normal-v1:min-abs-dot:{:.17e}",
                self.minimum_abs_alignment
            )
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SupportNormalPointEvidenceV1 {
    pub contact_index: usize,
    pub local_xy_m: [f64; 2],
    pub contact_normal_world: [f64; 3],
    pub sole_normal_world: [f64; 3],
    pub absolute_alignment: f64,
    pub admissible: bool,
}

impl SupportNormalPointEvidenceV1 {
    pub fn validate(&self) -> bool {
        self.local_xy_m.iter().all(|value| value.is_finite())
            && self
                .contact_normal_world
                .iter()
                .all(|value| value.is_finite())
            && self
                .sole_normal_world
                .iter()
                .all(|value| value.is_finite())
            && self.absolute_alignment.is_finite()
            && (0.0..=1.0 + 1.0e-12).contains(&self.absolute_alignment)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalQualifiedSupportRegionV1 {
    pub site: ContactSite,
    pub model_id: String,
    pub model_signature: u64,
    pub sampled_at_s: f64,
    pub source_geometry_id: String,
    pub source_interaction_id: String,
    pub policy_id: String,
    pub minimum_abs_alignment: f64,
    pub source_contact_indices: Vec<usize>,
    pub admitted_contact_indices: Vec<usize>,
    pub rejected_contact_indices: Vec<usize>,
    pub point_evidence: Vec<SupportNormalPointEvidenceV1>,
    pub hull_vertices_local_xy_m: Vec<[f64; 2]>,
    pub area_m2: f64,
}

impl NormalQualifiedSupportRegionV1 {
    pub fn validate(&self) -> bool {
        !self.model_id.trim().is_empty()
            && !self.source_geometry_id.trim().is_empty()
            && !self.source_interaction_id.trim().is_empty()
            && !self.policy_id.trim().is_empty()
            && self.sampled_at_s.is_finite()
            && self.sampled_at_s >= 0.0
            && self.minimum_abs_alignment.is_finite()
            && (0.0..=1.0).contains(&self.minimum_abs_alignment)
            && !self.source_contact_indices.is_empty()
            && strictly_increasing(&self.source_contact_indices)
            && strictly_increasing_or_empty(&self.admitted_contact_indices)
            && strictly_increasing_or_empty(&self.rejected_contact_indices)
            && self.point_evidence.len() == self.source_contact_indices.len()
            && self
                .point_evidence
                .iter()
                .all(SupportNormalPointEvidenceV1::validate)
            && self
                .admitted_contact_indices
                .iter()
                .all(|index| self.source_contact_indices.contains(index))
            && self
                .rejected_contact_indices
                .iter()
                .all(|index| self.source_contact_indices.contains(index))
            && self
                .admitted_contact_indices
                .iter()
                .all(|index| !self.rejected_contact_indices.contains(index))
            && self.admitted_contact_indices.len() + self.rejected_contact_indices.len()
                == self.source_contact_indices.len()
            && self
                .hull_vertices_local_xy_m
                .iter()
                .flat_map(|point| point.iter())
                .all(|value| value.is_finite())
            && self.area_m2.is_finite()
            && self.area_m2 >= 0.0
    }

    pub fn surface_support_eligible(&self) -> bool {
        self.hull_vertices_local_xy_m.len() >= 3 && self.area_m2 > REGION_AREA_EPS
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SupportNormalQualificationError {
    InvalidPolicy,
    InvalidInteractionRecord,
    InvalidPatchSet,
    SiteMismatch,
    ModelMismatch,
    TimeMismatch,
    GeometryBindingMismatch,
    ContactVectorLengthMismatch,
    InvalidContactNormal,
    InvalidResult,
}

pub fn qualify_support_normals(
    interaction: &MuJoCoContactInteractionRecordV1,
    patches: &MuJoCoFootPatchSetV1,
    policy: SupportNormalPolicyV1,
) -> Result<NormalQualifiedSupportRegionV1, SupportNormalQualificationError> {
    if !policy.validate() {
        return Err(SupportNormalQualificationError::InvalidPolicy);
    }
    if !interaction.validate() {
        return Err(SupportNormalQualificationError::InvalidInteractionRecord);
    }
    if !patches.validate() {
        return Err(SupportNormalQualificationError::InvalidPatchSet);
    }
    if interaction.model_id != patches.model_id
        || interaction.model_signature != patches.model_signature
    {
        return Err(SupportNormalQualificationError::ModelMismatch);
    }
    if !same_sample_time(interaction.sampled_at_s, patches.sampled_at_s) {
        return Err(SupportNormalQualificationError::TimeMismatch);
    }

    let patch_record = patches
        .record_for_site(interaction.site)
        .ok_or(SupportNormalQualificationError::SiteMismatch)?;
    if patch_record.geom_id != interaction.foot_geom_id
        || patch_record.physical_geom_name != interaction.foot_geom_name
        || patch_record.placed_patch.geometry.site != interaction.site
    {
        return Err(SupportNormalQualificationError::GeometryBindingMismatch);
    }

    let region = &interaction.active_region;
    let count = interaction.contact_indices.len();
    if region.contact_points_local_xy_m.len() != count
        || region.contact_normals_world.len() != count
    {
        return Err(SupportNormalQualificationError::ContactVectorLengthMismatch);
    }

    let sole_normal = patch_record.placed_patch.normal_world;
    let mut point_evidence = Vec::with_capacity(count);
    let mut admitted_contact_indices = Vec::new();
    let mut rejected_contact_indices = Vec::new();
    let mut admitted_points = Vec::new();

    for position in 0..count {
        let contact_index = interaction.contact_indices[position];
        let local_xy_m = region.contact_points_local_xy_m[position];
        let contact_normal_world = region.contact_normals_world[position];
        let absolute_alignment = absolute_normal_alignment(contact_normal_world, sole_normal)?;
        let admissible = absolute_alignment + 1.0e-12 >= policy.minimum_abs_alignment;
        if admissible {
            admitted_contact_indices.push(contact_index);
            admitted_points.push(local_xy_m);
        } else {
            rejected_contact_indices.push(contact_index);
        }
        point_evidence.push(SupportNormalPointEvidenceV1 {
            contact_index,
            local_xy_m,
            contact_normal_world,
            sole_normal_world: sole_normal,
            absolute_alignment,
            admissible,
        });
    }

    let hull_vertices_local_xy_m = convex_hull(admitted_points);
    let area_m2 = polygon_area(&hull_vertices_local_xy_m);
    let result = NormalQualifiedSupportRegionV1 {
        site: interaction.site,
        model_id: interaction.model_id.clone(),
        model_signature: interaction.model_signature,
        sampled_at_s: interaction.sampled_at_s,
        source_geometry_id: patch_record.placed_patch.geometry.geometry_id.clone(),
        source_interaction_id: interaction.limits.interaction_id.clone(),
        policy_id: policy.policy_id().ok_or(SupportNormalQualificationError::InvalidPolicy)?,
        minimum_abs_alignment: policy.minimum_abs_alignment,
        source_contact_indices: interaction.contact_indices.clone(),
        admitted_contact_indices,
        rejected_contact_indices,
        point_evidence,
        hull_vertices_local_xy_m,
        area_m2,
    };
    if result.validate() {
        Ok(result)
    } else {
        Err(SupportNormalQualificationError::InvalidResult)
    }
}

fn absolute_normal_alignment(
    contact_normal_world: [f64; 3],
    sole_normal_world: [f64; 3],
) -> Result<f64, SupportNormalQualificationError> {
    if contact_normal_world.iter().any(|value| !value.is_finite())
        || sole_normal_world.iter().any(|value| !value.is_finite())
    {
        return Err(SupportNormalQualificationError::InvalidContactNormal);
    }
    let contact_norm = norm3(contact_normal_world);
    let sole_norm = norm3(sole_normal_world);
    if contact_norm <= NORMAL_EPS || sole_norm <= NORMAL_EPS {
        return Err(SupportNormalQualificationError::InvalidContactNormal);
    }
    Ok((dot3(contact_normal_world, sole_normal_world) / (contact_norm * sole_norm))
        .abs()
        .clamp(0.0, 1.0))
}

fn strictly_increasing(values: &[usize]) -> bool {
    !values.is_empty() && values.windows(2).all(|pair| pair[0] < pair[1])
}

fn strictly_increasing_or_empty(values: &[usize]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn convex_hull(mut points: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
    points.sort_by(|left, right| {
        left[0]
            .total_cmp(&right[0])
            .then_with(|| left[1].total_cmp(&right[1]))
    });
    points.dedup_by(|left, right| left == right);
    if points.len() <= 2 {
        return points;
    }

    let mut lower = Vec::new();
    for point in &points {
        while lower.len() >= 2
            && cross2(lower[lower.len() - 2], lower[lower.len() - 1], *point) <= 0.0
        {
            lower.pop();
        }
        lower.push(*point);
    }
    let mut upper = Vec::new();
    for point in points.iter().rev() {
        while upper.len() >= 2
            && cross2(upper[upper.len() - 2], upper[upper.len() - 1], *point) <= 0.0
        {
            upper.pop();
        }
        upper.push(*point);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

fn polygon_area(points: &[[f64; 2]]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    let twice_area = (0..points.len())
        .map(|index| {
            let next = (index + 1) % points.len();
            points[index][0] * points[next][1] - points[next][0] * points[index][1]
        })
        .sum::<f64>();
    0.5 * twice_area.abs()
}

fn same_sample_time(left: f64, right: f64) -> bool {
    let tolerance = 1.0e-9 * (1.0 + left.abs().max(right.abs()));
    (left - right).abs() <= tolerance
}

fn norm3(vector: [f64; 3]) -> f64 {
    dot3(vector, vector).sqrt()
}

fn dot3(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross2(origin: [f64; 2], left: [f64; 2], right: [f64; 2]) -> f64 {
    (left[0] - origin[0]) * (right[1] - origin[1])
        - (left[1] - origin[1]) * (right[0] - origin[0])
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
    use crate::simulator::{HumanoidPhysicsSimulator, MuJoCoHumanoidSimulator};
    use crate::types::HumanoidCommand;

    #[test]
    fn sign_flipped_normals_have_identical_alignment() {
        let sole = [0.0, 0.0, 1.0];
        let positive = absolute_normal_alignment([0.0, 0.0, 1.0], sole).unwrap();
        let negative = absolute_normal_alignment([0.0, 0.0, -1.0], sole).unwrap();
        assert!((positive - 1.0).abs() < 1.0e-12);
        assert!((negative - positive).abs() < 1.0e-12);
    }

    #[test]
    fn vertical_wall_normal_is_not_plantar_aligned() {
        let alignment = absolute_normal_alignment([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap();
        assert_eq!(alignment, 0.0);
        assert!(alignment < 0.8);
    }

    #[test]
    fn zero_and_nonfinite_normals_fail_closed() {
        assert_eq!(
            absolute_normal_alignment([0.0; 3], [0.0, 0.0, 1.0]),
            Err(SupportNormalQualificationError::InvalidContactNormal)
        );
        assert_eq!(
            absolute_normal_alignment([f64::NAN, 0.0, 1.0], [0.0, 0.0, 1.0]),
            Err(SupportNormalQualificationError::InvalidContactNormal)
        );
    }

    #[test]
    fn policy_identity_changes_with_threshold() {
        let low = SupportNormalPolicyV1 {
            minimum_abs_alignment: 0.7,
        };
        let high = SupportNormalPolicyV1 {
            minimum_abs_alignment: 0.8,
        };
        assert_ne!(low.policy_id(), high.policy_id());
    }

    #[test]
    fn invalid_policy_fails_closed() {
        for value in [-0.1, 1.1, f64::NAN] {
            let policy = SupportNormalPolicyV1 {
                minimum_abs_alignment: value,
            };
            assert!(!policy.validate());
            assert!(policy.policy_id().is_none());
        }
    }

    #[test]
    fn generated_floor_contacts_are_normal_qualifiable() {
        let mut sim = MuJoCoHumanoidSimulator::for_morphology(HumanoidMorphology::Dmc21).unwrap();
        sim.step(&HumanoidCommand::zero(), 0.0);
        let model = Arc::clone(sim.model_arc());
        let patches = extract_mujoco_foot_patch_set(
            model.as_ref(),
            sim.data_mut(),
            "generated-dmc21-support-normal-v1",
        )
        .unwrap();
        let policy = SupportNormalPolicyV1 {
            minimum_abs_alignment: 0.8,
        };

        let mut qualified_any = false;
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
            let qualified = qualify_support_normals(&interaction, &patches, policy).unwrap();
            assert_eq!(
                qualified.admitted_contact_indices.len(),
                interaction.contact_indices.len()
            );
            assert!(qualified
                .point_evidence
                .iter()
                .all(|point| point.absolute_alignment >= 0.8));
            qualified_any = true;
        }
        assert!(qualified_any, "expected at least one active floor-support foot");
    }
}
