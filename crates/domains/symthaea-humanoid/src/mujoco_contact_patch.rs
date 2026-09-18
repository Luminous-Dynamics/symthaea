// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MuJoCo extraction for exact physical foot-patch geometry and placement.
//!
//! Geometry and placement are distinct from contact-pair interaction limits.
//! This module deliberately emits no friction proposition.

use std::sync::Arc;

use mujoco_rs::prelude::*;
use serde::{Deserialize, Serialize};

use crate::contact_patch::{
    ContactPatchGeometrySource, ContactPatchGeometryV1, ContactPatchPlacementSource,
    PlacedContactPatchV1,
};
use crate::multi_contact::ContactSite;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MuJoCoContactPatchError {
    EmptyModelIdentity,
    ModelMismatch,
    UnsupportedSite,
    MissingBody,
    MissingPhysicalGeom,
    MissingSupportSite,
    WrongGeomBody,
    WrongSupportSiteBody,
    UnsupportedGeomType,
    InvalidGeomSize,
    MissingWorldPose,
    InvalidPlacedPatch,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MuJoCoFootPatchRecordV1 {
    pub site: ContactSite,
    pub body_name: String,
    pub physical_geom_name: String,
    pub support_site_name: String,
    pub body_id: usize,
    pub geom_id: usize,
    pub support_site_id: usize,
    pub placed_patch: PlacedContactPatchV1,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MuJoCoFootPatchSetV1 {
    pub model_id: String,
    pub model_signature: u64,
    pub sampled_at_s: f64,
    pub records: Vec<MuJoCoFootPatchRecordV1>,
}

impl MuJoCoFootPatchSetV1 {
    pub fn validate(&self) -> bool {
        !self.model_id.trim().is_empty()
            && self.sampled_at_s.is_finite()
            && self.sampled_at_s >= 0.0
            && self.records.len() == 2
            && self.records.iter().all(|record| {
                record.placed_patch.validate()
                    && record.placed_patch.geometry.site == record.site
                    && same_sample_time(record.placed_patch.sampled_at_s, self.sampled_at_s)
                    && !record.body_name.trim().is_empty()
                    && !record.physical_geom_name.trim().is_empty()
                    && !record.support_site_name.trim().is_empty()
            })
            && self
                .records
                .iter()
                .any(|record| record.site == ContactSite::RightFoot)
            && self
                .records
                .iter()
                .any(|record| record.site == ContactSite::LeftFoot)
    }

    pub fn record_for_site(&self, site: ContactSite) -> Option<&MuJoCoFootPatchRecordV1> {
        self.records.iter().find(|record| record.site == site)
    }

    pub fn patch_for_site(&self, site: ContactSite) -> Option<&PlacedContactPatchV1> {
        self.record_for_site(site).map(|record| &record.placed_patch)
    }
}

#[derive(Debug, Clone, Copy)]
struct FootPatchMapping {
    site: ContactSite,
    body_name: &'static str,
    geom_name: &'static str,
    support_site_name: &'static str,
}

const FOOT_MAPPINGS: [FootPatchMapping; 2] = [
    FootPatchMapping {
        site: ContactSite::RightFoot,
        body_name: "right_foot",
        geom_name: "r_foot_g",
        support_site_name: "r_foot_site",
    },
    FootPatchMapping {
        site: ContactSite::LeftFoot,
        body_name: "left_foot",
        geom_name: "l_foot_g",
        support_site_name: "l_foot_site",
    },
];

/// Extract both physical foot support patches from the exact current MuJoCo state.
pub fn extract_mujoco_foot_patch_set(
    model: &MjModel,
    data: &MjData<Arc<MjModel>>,
    model_id: &str,
) -> Result<MuJoCoFootPatchSetV1, MuJoCoContactPatchError> {
    validate_model_binding(model, data, model_id)?;

    let sampled_at_s = data.time();
    let records = FOOT_MAPPINGS
        .iter()
        .copied()
        .map(|mapping| extract_mapping(model, data, model_id, mapping))
        .collect::<Result<Vec<_>, _>>()?;

    let set = MuJoCoFootPatchSetV1 {
        model_id: model_id.to_string(),
        model_signature: model.signature(),
        sampled_at_s,
        records,
    };
    if set.validate() {
        Ok(set)
    } else {
        Err(MuJoCoContactPatchError::InvalidPlacedPatch)
    }
}

/// Extract one supported foot patch. Unsupported contact sites fail closed.
pub fn extract_mujoco_patch_for_site(
    model: &MjModel,
    data: &MjData<Arc<MjModel>>,
    model_id: &str,
    site: ContactSite,
) -> Result<PlacedContactPatchV1, MuJoCoContactPatchError> {
    validate_model_binding(model, data, model_id)?;
    let mapping = FOOT_MAPPINGS
        .iter()
        .copied()
        .find(|mapping| mapping.site == site)
        .ok_or(MuJoCoContactPatchError::UnsupportedSite)?;
    Ok(extract_mapping(model, data, model_id, mapping)?.placed_patch)
}

fn validate_model_binding(
    model: &MjModel,
    data: &MjData<Arc<MjModel>>,
    model_id: &str,
) -> Result<(), MuJoCoContactPatchError> {
    if model_id.trim().is_empty() {
        return Err(MuJoCoContactPatchError::EmptyModelIdentity);
    }
    if model.signature() != data.model().signature() {
        return Err(MuJoCoContactPatchError::ModelMismatch);
    }
    Ok(())
}

fn extract_mapping(
    model: &MjModel,
    data: &MjData<Arc<MjModel>>,
    model_id: &str,
    mapping: FootPatchMapping,
) -> Result<MuJoCoFootPatchRecordV1, MuJoCoContactPatchError> {
    let body_id = model
        .name_to_id(MjtObj::mjOBJ_BODY, mapping.body_name)
        .ok_or(MuJoCoContactPatchError::MissingBody)?;
    let geom_id = model
        .name_to_id(MjtObj::mjOBJ_GEOM, mapping.geom_name)
        .ok_or(MuJoCoContactPatchError::MissingPhysicalGeom)?;
    let support_site_id = model
        .name_to_id(MjtObj::mjOBJ_SITE, mapping.support_site_name)
        .ok_or(MuJoCoContactPatchError::MissingSupportSite)?;

    if model.geom_bodyid().get(geom_id).copied() != Some(body_id as i32) {
        return Err(MuJoCoContactPatchError::WrongGeomBody);
    }
    if model.site_bodyid().get(support_site_id).copied() != Some(body_id as i32) {
        return Err(MuJoCoContactPatchError::WrongSupportSiteBody);
    }
    if model.geom_type().get(geom_id).copied() != Some(MjtGeom::mjGEOM_BOX) {
        return Err(MuJoCoContactPatchError::UnsupportedGeomType);
    }

    let [half_x, half_y, half_z] = model
        .geom_size()
        .get(geom_id)
        .copied()
        .ok_or(MuJoCoContactPatchError::InvalidGeomSize)?;
    if [half_x, half_y, half_z]
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(MuJoCoContactPatchError::InvalidGeomSize);
    }

    let center = data
        .geom_xpos()
        .get(geom_id)
        .copied()
        .ok_or(MuJoCoContactPatchError::MissingWorldPose)?;
    let rotation = data
        .geom_xmat()
        .get(geom_id)
        .copied()
        .ok_or(MuJoCoContactPatchError::MissingWorldPose)?;
    if center.iter().any(|value| !value.is_finite())
        || rotation.iter().any(|value| !value.is_finite())
    {
        return Err(MuJoCoContactPatchError::MissingWorldPose);
    }

    // MuJoCo xmat is row-major. For a local-to-world rotation matrix, the
    // columns are the local basis axes expressed in world coordinates.
    let tangent_x_world = [rotation[0], rotation[3], rotation[6]];
    let tangent_y_world = [rotation[1], rotation[4], rotation[7]];
    let normal_world = [rotation[2], rotation[5], rotation[8]];

    // geom_xpos is the box centre; support is the physical lower face.
    let origin_world_m = [
        center[0] - half_z * normal_world[0],
        center[1] - half_z * normal_world[1],
        center[2] - half_z * normal_world[2],
    ];

    let geometry = ContactPatchGeometryV1 {
        site: mapping.site,
        vertices_local_xy_m: vec![
            [-half_x, -half_y],
            [half_x, -half_y],
            [half_x, half_y],
            [-half_x, half_y],
        ],
        source: ContactPatchGeometrySource::SimulatorGeometry,
        geometry_id: format!(
            "{model_id}:sig:{:016x}:geom:{}:box:{half_x:.17e}:{half_y:.17e}:{half_z:.17e}",
            model.signature(),
            mapping.geom_name,
        ),
    };

    let placed_patch = PlacedContactPatchV1 {
        geometry,
        origin_world_m,
        tangent_x_world,
        tangent_y_world,
        normal_world,
        sampled_at_s: data.time(),
        placement_source: ContactPatchPlacementSource::SimulatorKinematics,
        confidence: 1.0,
    };
    if !placed_patch.validate() {
        return Err(MuJoCoContactPatchError::InvalidPlacedPatch);
    }

    Ok(MuJoCoFootPatchRecordV1 {
        site: mapping.site,
        body_name: mapping.body_name.to_string(),
        physical_geom_name: mapping.geom_name.to_string(),
        support_site_name: mapping.support_site_name.to_string(),
        body_id,
        geom_id,
        support_site_id,
        placed_patch,
    })
}

fn same_sample_time(left: f64, right: f64) -> bool {
    let tolerance = 1.0e-9 * (1.0 + left.abs().max(right.abs()));
    (left - right).abs() <= tolerance
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphology::HumanoidMorphology;
    use crate::simulator::{HumanoidPhysicsSimulator, MuJoCoHumanoidSimulator};
    use crate::types::HumanoidCommand;

    const MODEL_ID: &str = "generated-dmc21-foot-patch-test-v2";

    fn generated() -> MuJoCoHumanoidSimulator {
        MuJoCoHumanoidSimulator::for_morphology(HumanoidMorphology::Dmc21).unwrap()
    }

    fn extract(sim: &mut MuJoCoHumanoidSimulator) -> MuJoCoFootPatchSetV1 {
        let model = Arc::clone(sim.model_arc());
        extract_mujoco_foot_patch_set(model.as_ref(), sim.data_mut(), MODEL_ID).unwrap()
    }

    #[test]
    fn generated_feet_use_physical_geom_extents_not_site_markers() {
        let mut sim = generated();
        let set = extract(&mut sim);
        assert!(set.validate());
        for site in [ContactSite::RightFoot, ContactSite::LeftFoot] {
            let patch = set.patch_for_site(site).unwrap();
            assert!((patch.geometry.area_m2().unwrap() - 0.0392).abs() < 1.0e-12);
            let max_x = patch
                .geometry
                .vertices_local_xy_m
                .iter()
                .map(|vertex| vertex[0].abs())
                .fold(0.0, f64::max);
            let max_y = patch
                .geometry
                .vertices_local_xy_m
                .iter()
                .map(|vertex| vertex[1].abs())
                .fold(0.0, f64::max);
            assert!((max_x - 0.14).abs() < 1.0e-12);
            assert!((max_y - 0.07).abs() < 1.0e-12);
        }

        let model = Arc::clone(sim.model_arc());
        let site_id = model
            .name_to_id(MjtObj::mjOBJ_SITE, "r_foot_site")
            .unwrap();
        assert!((model.site_size()[site_id][0] - 0.142).abs() < 1.0e-12);
        assert_ne!(model.site_size()[site_id][0], 0.14);
    }

    #[test]
    fn geometry_identity_is_stable_while_sample_time_advances() {
        let mut sim = generated();
        let before = extract(&mut sim);
        let geometry_id = before
            .patch_for_site(ContactSite::RightFoot)
            .unwrap()
            .geometry
            .geometry_id
            .clone();
        let time_before = before.sampled_at_s;

        sim.step(&HumanoidCommand::zero(), 0.0);
        let after = extract(&mut sim);
        assert_eq!(
            after
                .patch_for_site(ContactSite::RightFoot)
                .unwrap()
                .geometry
                .geometry_id,
            geometry_id
        );
        assert!(after.sampled_at_s > time_before);
    }

    #[test]
    fn support_plane_is_bottom_face_not_box_center() {
        let mut sim = generated();
        let model = Arc::clone(sim.model_arc());
        let geom_id = model.name_to_id(MjtObj::mjOBJ_GEOM, "r_foot_g").unwrap();
        let center = sim.data_mut().geom_xpos()[geom_id];
        let set = extract(&mut sim);
        let patch = set.patch_for_site(ContactSite::RightFoot).unwrap();
        let delta = [
            center[0] - patch.origin_world_m[0],
            center[1] - patch.origin_world_m[1],
            center[2] - patch.origin_world_m[2],
        ];
        let along_normal = delta[0] * patch.normal_world[0]
            + delta[1] * patch.normal_world[1]
            + delta[2] * patch.normal_world[2];
        assert!((along_normal - 0.025).abs() < 1.0e-12);
    }

    #[test]
    fn unsupported_contact_sites_fail_closed() {
        let mut sim = generated();
        let model = Arc::clone(sim.model_arc());
        assert_eq!(
            extract_mujoco_patch_for_site(
                model.as_ref(),
                sim.data_mut(),
                MODEL_ID,
                ContactSite::RightHand,
            ),
            Err(MuJoCoContactPatchError::UnsupportedSite)
        );
    }

    #[test]
    fn cross_model_kinematics_are_rejected() {
        let mut dmc = generated();
        let other = MuJoCoHumanoidSimulator::for_morphology(HumanoidMorphology::WithNeckWrist)
            .unwrap();
        let other_model = Arc::clone(other.model_arc());
        assert_eq!(
            extract_mujoco_foot_patch_set(other_model.as_ref(), dmc.data_mut(), MODEL_ID),
            Err(MuJoCoContactPatchError::ModelMismatch)
        );
    }

    #[test]
    fn geometry_extractor_emits_no_interaction_limits() {
        let mut sim = generated();
        let set = extract(&mut sim);
        let right = set.patch_for_site(ContactSite::RightFoot).unwrap();
        assert!(right.geometry.geometry_id.contains("r_foot_g"));
        assert_eq!(right.geometry.source, ContactPatchGeometrySource::SimulatorGeometry);
    }
}
