// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MuJoCo contact-pair and active-support evidence for qualified humanoid foot patches.
//!
//! Immutable support geometry, time-varying placement, material/contact-pair limits,
//! and the *currently active* support region are distinct propositions. A physical
//! sole may be large while only its toe or edge is touching; downstream COP authority
//! must not silently expand that instantaneous contact into the whole sole.

use std::sync::Arc;

use mujoco_rs::prelude::*;
use serde::{Deserialize, Serialize};

use crate::contact_patch::PlacedContactPatchV1;
use crate::contact_wrench::{ContactInteractionLimitSource, ContactInteractionLimitsV1};
use crate::mujoco_contact_patch::MuJoCoFootPatchSetV1;
use crate::multi_contact::ContactSite;

const REGION_AREA_EPS: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MuJoCoSupportContactToleranceV1 {
    /// Allowed difference between resolved friction parameters across point contacts.
    pub friction_abs: f64,
    /// XY tolerance when deciding whether a contact point lies on the declared patch.
    pub patch_xy_m: f64,
    /// Absolute distance from the declared support plane admitted as plantar support.
    pub support_plane_m: f64,
}

impl Default for MuJoCoSupportContactToleranceV1 {
    fn default() -> Self {
        Self {
            friction_abs: 1.0e-12,
            patch_xy_m: 1.0e-6,
            support_plane_m: 2.0e-3,
        }
    }
}

impl MuJoCoSupportContactToleranceV1 {
    pub fn validate(&self) -> bool {
        [self.friction_abs, self.patch_xy_m, self.support_plane_m]
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MuJoCoContactDimensionalityV1 {
    NormalOnly,
    Tangential,
    Torsional,
    Rolling,
}

impl MuJoCoContactDimensionalityV1 {
    pub const fn from_dim(dim: i32) -> Option<Self> {
        match dim {
            1 => Some(Self::NormalOnly),
            3 => Some(Self::Tangential),
            4 => Some(Self::Torsional),
            6 => Some(Self::Rolling),
            _ => None,
        }
    }

    pub const fn dim(self) -> i32 {
        match self {
            Self::NormalOnly => 1,
            Self::Tangential => 3,
            Self::Torsional => 4,
            Self::Rolling => 6,
        }
    }

    pub const fn grants_sliding(self) -> bool {
        !matches!(self, Self::NormalOnly)
    }

    pub const fn grants_torsion(self) -> bool {
        matches!(self, Self::Torsional | Self::Rolling)
    }

    pub const fn grants_rolling(self) -> bool {
        matches!(self, Self::Rolling)
    }
}

/// Conservative scalar projection used while `ContactInteractionLimitsV1` owns
/// one sliding coefficient. The original anisotropic pair remains in evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlidingFrictionProjectionV1 {
    MinimumResolvedTangential,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MuJoCoActiveContactRegionV1 {
    pub contact_positions_world_m: Vec<[f64; 3]>,
    pub contact_points_local_xy_m: Vec<[f64; 2]>,
    /// Signed offsets from the physical support plane along the patch normal.
    pub contact_plane_offsets_m: Vec<f64>,
    /// MuJoCo contact-frame normal axes in world coordinates.
    pub contact_normals_world: Vec<[f64; 3]>,
    /// Convex hull of current plantar contact points in the patch-local plane.
    pub hull_vertices_local_xy_m: Vec<[f64; 2]>,
    pub area_m2: f64,
}

impl MuJoCoActiveContactRegionV1 {
    pub fn validate(&self) -> bool {
        let count = self.contact_positions_world_m.len();
        count > 0
            && self.contact_points_local_xy_m.len() == count
            && self.contact_plane_offsets_m.len() == count
            && self.contact_normals_world.len() == count
            && self
                .contact_positions_world_m
                .iter()
                .flat_map(|point| point.iter())
                .all(|value| value.is_finite())
            && self
                .contact_points_local_xy_m
                .iter()
                .flat_map(|point| point.iter())
                .all(|value| value.is_finite())
            && self
                .contact_plane_offsets_m
                .iter()
                .all(|value| value.is_finite())
            && self
                .contact_normals_world
                .iter()
                .flat_map(|normal| normal.iter())
                .all(|value| value.is_finite())
            && self
                .hull_vertices_local_xy_m
                .iter()
                .flat_map(|point| point.iter())
                .all(|value| value.is_finite())
            && self.area_m2.is_finite()
            && self.area_m2 >= 0.0
    }

    /// A non-degenerate polygon is required before the strong surface-COP path
    /// may treat the current contact as an areal support region.
    pub fn surface_support_eligible(&self) -> bool {
        self.hull_vertices_local_xy_m.len() >= 3 && self.area_m2 > REGION_AREA_EPS
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MuJoCoContactInteractionRecordV1 {
    pub site: ContactSite,
    pub model_id: String,
    pub model_signature: u64,
    pub sampled_at_s: f64,
    pub foot_geom_id: usize,
    pub foot_geom_name: String,
    pub environment_geom_id: usize,
    pub environment_geom_name: String,
    /// Exact live-contact indices admitted as plantar support.
    pub contact_indices: Vec<usize>,
    /// Active foot-geom contacts that were valid constraints but did not lie on
    /// the qualified plantar patch/plane. These cannot create support authority.
    pub non_supporting_contact_indices: Vec<usize>,
    pub dimensionality: MuJoCoContactDimensionalityV1,
    /// Resolved MuJoCo `[tangent1, tangent2, spin, roll1, roll2]` coefficients.
    pub resolved_friction: [f64; 5],
    pub sliding_projection: SlidingFrictionProjectionV1,
    pub extraction_tolerance: MuJoCoSupportContactToleranceV1,
    pub active_region: MuJoCoActiveContactRegionV1,
    pub limits: ContactInteractionLimitsV1,
}

impl MuJoCoContactInteractionRecordV1 {
    pub fn validate(&self) -> bool {
        !self.model_id.trim().is_empty()
            && self.sampled_at_s.is_finite()
            && self.sampled_at_s >= 0.0
            && !self.foot_geom_name.trim().is_empty()
            && !self.environment_geom_name.trim().is_empty()
            && !self.contact_indices.is_empty()
            && self.contact_indices.windows(2).all(|pair| pair[0] < pair[1])
            && self
                .non_supporting_contact_indices
                .windows(2)
                .all(|pair| pair[0] < pair[1])
            && self.active_region.validate()
            && self.active_region.contact_positions_world_m.len() == self.contact_indices.len()
            && self.extraction_tolerance.validate()
            && self
                .active_region
                .contact_plane_offsets_m
                .iter()
                .all(|offset| offset.abs() <= self.extraction_tolerance.support_plane_m + 1.0e-12)
            && self
                .resolved_friction
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
            && self.limits.validate()
            && self.limits.site == self.site
            && self.limits.source == ContactInteractionLimitSource::SimulatorContactPair
            && same_sample_time(self.limits.sampled_at_s, self.sampled_at_s)
            && (!self.dimensionality.grants_sliding()
                || self.limits.sliding_friction_coefficient
                    <= self.resolved_friction[0].min(self.resolved_friction[1]) + 1.0e-12)
            && (self.dimensionality.grants_sliding()
                || self.limits.sliding_friction_coefficient == 0.0)
            && (self.dimensionality.grants_torsion()
                || self.limits.torsional_friction_radius_m == 0.0)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MuJoCoContactInteractionError {
    EmptyModelIdentity,
    ModelMismatch,
    PatchSetMismatch,
    InvalidPatchSet,
    InvalidTolerance,
    UnsupportedSite,
    MissingFootPatchRecord,
    NoActiveContact,
    NoActiveSupportContact,
    InvalidContactGeom,
    InvalidContactPoint,
    MissingEnvironmentName,
    AmbiguousEnvironmentGeom,
    UnsupportedContactDimensionality(i32),
    InvalidResolvedFriction,
    InconsistentContactDimensionality,
    InconsistentInteractionLimits,
    InvalidActiveRegion,
    InvalidResult,
}

#[derive(Debug, Clone, Copy)]
struct ContactSample {
    index: usize,
    environment_geom_id: usize,
    dim: i32,
    friction: [f64; 5],
    position_world_m: [f64; 3],
    normal_world: [f64; 3],
    local_xy_m: [f64; 2],
    plane_offset_m: f64,
}

pub fn extract_mujoco_contact_interaction(
    model: &MjModel,
    data: &MjData<Arc<MjModel>>,
    patches: &MuJoCoFootPatchSetV1,
    site: ContactSite,
) -> Result<MuJoCoContactInteractionRecordV1, MuJoCoContactInteractionError> {
    extract_mujoco_contact_interaction_with_tolerance(
        model,
        data,
        patches,
        site,
        MuJoCoSupportContactToleranceV1::default(),
    )
}

/// Extract exact current MuJoCo pair limits plus the actual plantar contact hull.
pub fn extract_mujoco_contact_interaction_with_tolerance(
    model: &MjModel,
    data: &MjData<Arc<MjModel>>,
    patches: &MuJoCoFootPatchSetV1,
    site: ContactSite,
    tolerance: MuJoCoSupportContactToleranceV1,
) -> Result<MuJoCoContactInteractionRecordV1, MuJoCoContactInteractionError> {
    if patches.model_id.trim().is_empty() {
        return Err(MuJoCoContactInteractionError::EmptyModelIdentity);
    }
    if !patches.validate() {
        return Err(MuJoCoContactInteractionError::InvalidPatchSet);
    }
    if !tolerance.validate() {
        return Err(MuJoCoContactInteractionError::InvalidTolerance);
    }
    if model.signature() != data.model().signature() {
        return Err(MuJoCoContactInteractionError::ModelMismatch);
    }
    if patches.model_signature != model.signature()
        || !same_sample_time(patches.sampled_at_s, data.time())
    {
        return Err(MuJoCoContactInteractionError::PatchSetMismatch);
    }
    if !matches!(site, ContactSite::RightFoot | ContactSite::LeftFoot) {
        return Err(MuJoCoContactInteractionError::UnsupportedSite);
    }

    let patch_record = patches
        .record_for_site(site)
        .ok_or(MuJoCoContactInteractionError::MissingFootPatchRecord)?;
    let foot_geom_id = patch_record.geom_id;
    if model.id_to_name(MjtObj::mjOBJ_GEOM, foot_geom_id)
        != Some(patch_record.physical_geom_name.as_str())
    {
        return Err(MuJoCoContactInteractionError::PatchSetMismatch);
    }

    let patch = &patch_record.placed_patch;
    let mut samples = Vec::new();
    let mut non_supporting_contact_indices = Vec::new();
    let mut any_active_foot_contact = false;

    for (index, contact) in data.contact().iter().enumerate() {
        if contact.exclude != 0 || contact.efc_address < 0 {
            continue;
        }
        let foot_is_first = contact.geom1 == foot_geom_id as i32;
        let foot_is_second = contact.geom2 == foot_geom_id as i32;
        if !foot_is_first && !foot_is_second {
            continue;
        }
        any_active_foot_contact = true;

        let position_world_m = contact.pos;
        let normal_world = [contact.frame[0], contact.frame[1], contact.frame[2]];
        if position_world_m.iter().any(|value| !value.is_finite())
            || normal_world.iter().any(|value| !value.is_finite())
        {
            return Err(MuJoCoContactInteractionError::InvalidContactPoint);
        }
        let (local_xy_m, plane_offset_m) = project_contact_to_patch(patch, position_world_m);
        if !point_in_convex_patch(
            &patch.geometry.vertices_local_xy_m,
            local_xy_m,
            tolerance.patch_xy_m,
        ) || plane_offset_m.abs() > tolerance.support_plane_m
        {
            non_supporting_contact_indices.push(index);
            continue;
        }

        let environment_raw = if foot_is_first {
            contact.geom2
        } else {
            contact.geom1
        };
        if environment_raw < 0 || environment_raw as usize == foot_geom_id {
            return Err(MuJoCoContactInteractionError::InvalidContactGeom);
        }
        samples.push(ContactSample {
            index,
            environment_geom_id: environment_raw as usize,
            dim: contact.dim,
            friction: contact.friction,
            position_world_m,
            normal_world,
            local_xy_m,
            plane_offset_m,
        });
    }

    if !any_active_foot_contact {
        return Err(MuJoCoContactInteractionError::NoActiveContact);
    }
    if samples.is_empty() {
        return Err(MuJoCoContactInteractionError::NoActiveSupportContact);
    }

    let (environment_geom_id, contact_indices, dimensionality, friction) =
        aggregate_samples(&samples, tolerance.friction_abs)?;
    let environment_geom_name = model
        .id_to_name(MjtObj::mjOBJ_GEOM, environment_geom_id)
        .ok_or(MuJoCoContactInteractionError::MissingEnvironmentName)?
        .to_string();

    let active_region = build_active_region(&samples)?;
    let sliding_friction_coefficient = if dimensionality.grants_sliding() {
        friction[0].min(friction[1])
    } else {
        0.0
    };
    let torsional_friction_radius_m = if dimensionality.grants_torsion() {
        friction[2]
    } else {
        0.0
    };
    let sampled_at_s = data.time();
    let interaction_id = format!(
        "{}:sig:{:016x}:site:{site:?}:foot:{}:{}:env:{}:{}:dim:{}:friction:{:.17e}:{:.17e}:{:.17e}:{:.17e}:{:.17e}:projection:min-tangent-v1",
        patches.model_id,
        model.signature(),
        patch_record.physical_geom_name,
        foot_geom_id,
        environment_geom_name,
        environment_geom_id,
        dimensionality.dim(),
        friction[0],
        friction[1],
        friction[2],
        friction[3],
        friction[4],
    );
    let limits = ContactInteractionLimitsV1 {
        site,
        sliding_friction_coefficient,
        torsional_friction_radius_m,
        source: ContactInteractionLimitSource::SimulatorContactPair,
        interaction_id,
        sampled_at_s,
        confidence: 1.0,
    };

    non_supporting_contact_indices.sort_unstable();
    non_supporting_contact_indices.dedup();
    let result = MuJoCoContactInteractionRecordV1 {
        site,
        model_id: patches.model_id.clone(),
        model_signature: model.signature(),
        sampled_at_s,
        foot_geom_id,
        foot_geom_name: patch_record.physical_geom_name.clone(),
        environment_geom_id,
        environment_geom_name,
        contact_indices,
        non_supporting_contact_indices,
        dimensionality,
        resolved_friction: friction,
        sliding_projection: SlidingFrictionProjectionV1::MinimumResolvedTangential,
        extraction_tolerance: tolerance,
        active_region,
        limits,
    };
    if result.validate() {
        Ok(result)
    } else {
        Err(MuJoCoContactInteractionError::InvalidResult)
    }
}

fn aggregate_samples(
    samples: &[ContactSample],
    friction_tolerance: f64,
) -> Result<
    (usize, Vec<usize>, MuJoCoContactDimensionalityV1, [f64; 5]),
    MuJoCoContactInteractionError,
> {
    let first = samples
        .first()
        .ok_or(MuJoCoContactInteractionError::NoActiveSupportContact)?;
    if !friction_tolerance.is_finite() || friction_tolerance < 0.0 {
        return Err(MuJoCoContactInteractionError::InvalidResolvedFriction);
    }
    let dimensionality = MuJoCoContactDimensionalityV1::from_dim(first.dim)
        .ok_or(MuJoCoContactInteractionError::UnsupportedContactDimensionality(first.dim))?;
    validate_friction(first.friction)?;

    let mut indices = Vec::with_capacity(samples.len());
    for sample in samples {
        if sample.environment_geom_id != first.environment_geom_id {
            return Err(MuJoCoContactInteractionError::AmbiguousEnvironmentGeom);
        }
        if sample.dim != first.dim {
            return Err(MuJoCoContactInteractionError::InconsistentContactDimensionality);
        }
        MuJoCoContactDimensionalityV1::from_dim(sample.dim)
            .ok_or(MuJoCoContactInteractionError::UnsupportedContactDimensionality(sample.dim))?;
        validate_friction(sample.friction)?;
        if !friction_matches(first.friction, sample.friction, friction_tolerance) {
            return Err(MuJoCoContactInteractionError::InconsistentInteractionLimits);
        }
        indices.push(sample.index);
    }
    indices.sort_unstable();
    indices.dedup();

    Ok((
        first.environment_geom_id,
        indices,
        dimensionality,
        first.friction,
    ))
}

fn build_active_region(
    samples: &[ContactSample],
) -> Result<MuJoCoActiveContactRegionV1, MuJoCoContactInteractionError> {
    if samples.is_empty() {
        return Err(MuJoCoContactInteractionError::NoActiveSupportContact);
    }
    let contact_positions_world_m = samples
        .iter()
        .map(|sample| sample.position_world_m)
        .collect::<Vec<_>>();
    let contact_points_local_xy_m = samples
        .iter()
        .map(|sample| sample.local_xy_m)
        .collect::<Vec<_>>();
    let contact_plane_offsets_m = samples
        .iter()
        .map(|sample| sample.plane_offset_m)
        .collect::<Vec<_>>();
    let contact_normals_world = samples
        .iter()
        .map(|sample| sample.normal_world)
        .collect::<Vec<_>>();
    let hull_vertices_local_xy_m = convex_hull(contact_points_local_xy_m.clone());
    let area_m2 = polygon_area(&hull_vertices_local_xy_m);
    let region = MuJoCoActiveContactRegionV1 {
        contact_positions_world_m,
        contact_points_local_xy_m,
        contact_plane_offsets_m,
        contact_normals_world,
        hull_vertices_local_xy_m,
        area_m2,
    };
    if region.validate() {
        Ok(region)
    } else {
        Err(MuJoCoContactInteractionError::InvalidActiveRegion)
    }
}

fn project_contact_to_patch(
    patch: &PlacedContactPatchV1,
    point_world_m: [f64; 3],
) -> ([f64; 2], f64) {
    let relative = [
        point_world_m[0] - patch.origin_world_m[0],
        point_world_m[1] - patch.origin_world_m[1],
        point_world_m[2] - patch.origin_world_m[2],
    ];
    (
        [
            dot3(relative, patch.tangent_x_world),
            dot3(relative, patch.tangent_y_world),
        ],
        dot3(relative, patch.normal_world),
    )
}

fn point_in_convex_patch(vertices: &[[f64; 2]], point: [f64; 2], tolerance: f64) -> bool {
    vertices.len() >= 3
        && (0..vertices.len()).all(|index| {
            let a = vertices[index];
            let b = vertices[(index + 1) % vertices.len()];
            cross2(a, b, point) >= -tolerance
        })
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

fn validate_friction(friction: [f64; 5]) -> Result<(), MuJoCoContactInteractionError> {
    if friction
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0)
    {
        Ok(())
    } else {
        Err(MuJoCoContactInteractionError::InvalidResolvedFriction)
    }
}

fn friction_matches(left: [f64; 5], right: [f64; 5], tolerance: f64) -> bool {
    left.into_iter()
        .zip(right)
        .all(|(left, right)| (left - right).abs() <= tolerance)
}

fn same_sample_time(left: f64, right: f64) -> bool {
    let tolerance = 1.0e-9 * (1.0 + left.abs().max(right.abs()));
    (left - right).abs() <= tolerance
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
    use super::*;
    use crate::morphology::HumanoidMorphology;
    use crate::mujoco_contact_patch::extract_mujoco_foot_patch_set;
    use crate::simulator::{HumanoidPhysicsSimulator, MuJoCoHumanoidSimulator};
    use crate::types::HumanoidCommand;

    const MODEL_ID: &str = "generated-dmc21-contact-interaction-v2";

    fn sample(index: usize, env: usize, dim: i32, friction: [f64; 5], xy: [f64; 2]) -> ContactSample {
        ContactSample {
            index,
            environment_geom_id: env,
            dim,
            friction,
            position_world_m: [xy[0], xy[1], 0.0],
            normal_world: [0.0, 0.0, 1.0],
            local_xy_m: xy,
            plane_offset_m: 0.0,
        }
    }

    #[test]
    fn dimensionality_grants_only_declared_authority() {
        assert!(!MuJoCoContactDimensionalityV1::NormalOnly.grants_sliding());
        assert!(MuJoCoContactDimensionalityV1::Tangential.grants_sliding());
        assert!(!MuJoCoContactDimensionalityV1::Tangential.grants_torsion());
        assert!(MuJoCoContactDimensionalityV1::Torsional.grants_torsion());
        assert!(!MuJoCoContactDimensionalityV1::Torsional.grants_rolling());
        assert!(MuJoCoContactDimensionalityV1::Rolling.grants_rolling());
    }

    #[test]
    fn active_region_uses_current_contact_hull_not_whole_sole() {
        let friction = [0.8, 0.8, 0.03, 0.0, 0.0];
        let samples = [
            sample(0, 7, 3, friction, [-0.02, -0.01]),
            sample(1, 7, 3, friction, [0.02, -0.01]),
            sample(2, 7, 3, friction, [0.02, 0.01]),
            sample(3, 7, 3, friction, [-0.02, 0.01]),
        ];
        let region = build_active_region(&samples).unwrap();
        assert!(region.surface_support_eligible());
        assert!((region.area_m2 - 0.0008).abs() < 1.0e-12);
        assert!(region.area_m2 < 0.0392);
    }

    #[test]
    fn one_or_two_points_do_not_gain_areal_cop_authority() {
        let friction = [0.8, 0.8, 0.0, 0.0, 0.0];
        let one = [sample(0, 7, 3, friction, [0.0, 0.0])];
        let two = [
            sample(0, 7, 3, friction, [-0.01, 0.0]),
            sample(1, 7, 3, friction, [0.01, 0.0]),
        ];
        assert!(!build_active_region(&one).unwrap().surface_support_eligible());
        assert!(!build_active_region(&two).unwrap().surface_support_eligible());
    }

    #[test]
    fn multiple_consistent_contacts_retain_all_indices() {
        let friction = [0.8, 0.7, 0.03, 0.01, 0.01];
        let samples = [
            sample(2, 7, 4, friction, [-0.01, 0.0]),
            sample(5, 7, 4, friction, [0.01, 0.0]),
        ];
        let (_, indices, dim, resolved) = aggregate_samples(&samples, 1.0e-12).unwrap();
        assert_eq!(indices, vec![2, 5]);
        assert_eq!(dim, MuJoCoContactDimensionalityV1::Torsional);
        assert_eq!(resolved, friction);
    }

    #[test]
    fn differing_environment_geoms_fail_closed() {
        let friction = [0.8, 0.8, 0.03, 0.0, 0.0];
        let samples = [
            sample(0, 7, 4, friction, [-0.01, 0.0]),
            sample(1, 8, 4, friction, [0.01, 0.0]),
        ];
        assert_eq!(
            aggregate_samples(&samples, 1.0e-12),
            Err(MuJoCoContactInteractionError::AmbiguousEnvironmentGeom)
        );
    }

    #[test]
    fn inconsistent_resolved_friction_fails_closed() {
        let first = [0.8, 0.8, 0.03, 0.0, 0.0];
        let second = [0.7, 0.7, 0.03, 0.0, 0.0];
        let samples = [
            sample(0, 7, 4, first, [-0.01, 0.0]),
            sample(1, 7, 4, second, [0.01, 0.0]),
        ];
        assert_eq!(
            aggregate_samples(&samples, 1.0e-12),
            Err(MuJoCoContactInteractionError::InconsistentInteractionLimits)
        );
    }

    #[test]
    fn unsupported_contact_dimensionality_fails_closed() {
        let samples = [sample(
            0,
            7,
            2,
            [0.8, 0.8, 0.0, 0.0, 0.0],
            [0.0, 0.0],
        )];
        assert_eq!(
            aggregate_samples(&samples, 1.0e-12),
            Err(MuJoCoContactInteractionError::UnsupportedContactDimensionality(2))
        );
    }

    #[test]
    fn generated_default_contact_is_tangential_and_grants_no_torsion() {
        let mut sim = MuJoCoHumanoidSimulator::for_morphology(HumanoidMorphology::Dmc21).unwrap();
        sim.step(&HumanoidCommand::zero(), 0.0);
        let model = Arc::clone(sim.model_arc());
        let patches =
            extract_mujoco_foot_patch_set(model.as_ref(), sim.data_mut(), MODEL_ID).unwrap();

        let mut extracted = Vec::new();
        for site in [ContactSite::RightFoot, ContactSite::LeftFoot] {
            match extract_mujoco_contact_interaction(
                model.as_ref(),
                sim.data_mut(),
                &patches,
                site,
            ) {
                Ok(record) => extracted.push(record),
                Err(MuJoCoContactInteractionError::NoActiveContact)
                | Err(MuJoCoContactInteractionError::NoActiveSupportContact) => {}
                Err(error) => panic!("unexpected MuJoCo interaction extraction failure: {error:?}"),
            }
        }
        assert!(
            !extracted.is_empty(),
            "generated standing humanoid should have at least one active plantar foot contact"
        );
        for record in extracted {
            assert_eq!(record.environment_geom_name, "floor");
            assert_eq!(record.dimensionality, MuJoCoContactDimensionalityV1::Tangential);
            assert!(record.limits.sliding_friction_coefficient > 0.0);
            assert_eq!(record.limits.torsional_friction_radius_m, 0.0);
            assert!(record.resolved_friction[2] > 0.0);
            assert!(!record.contact_indices.is_empty());
            assert!(record.active_region.validate());
            assert!(record
                .active_region
                .contact_plane_offsets_m
                .iter()
                .all(|offset| offset.abs() <= record.extraction_tolerance.support_plane_m));
            assert_eq!(
                record.limits.source,
                ContactInteractionLimitSource::SimulatorContactPair
            );
        }
    }

    #[test]
    fn stale_patch_set_time_is_rejected() {
        let mut sim = MuJoHumanoidSimulator::for_morphology(HumanoidMorphology::Dmc21).unwrap();
        let model = Arc::clone(sim.model_arc());
        let patches =
            extract_mujoco_foot_patch_set(model.as_ref(), sim.data_mut(), MODEL_ID).unwrap();
        sim.step(&HumanoidCommand::zero(), 0.0);
        assert_eq!(
            extract_mujoco_contact_interaction(
                model.as_ref(),
                sim.data_mut(),
                &patches,
                ContactSite::RightFoot,
            ),
            Err(MuJoCoContactInteractionError::PatchSetMismatch)
        );
    }
}
