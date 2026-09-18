// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit contact-patch geometry and world placement.
//!
//! A planted foot or hand is a surface, not a single point. This module keeps
//! geometry separate from contact activation, friction, and material interaction
//! so support-area and future wrench/COP constraints cannot silently infer one
//! physical proposition from another.

use serde::{Deserialize, Serialize};

use crate::multi_contact::{ContactSite, MultiContactFrame};

const MAX_PATCH_VERTICES: usize = 16;
const GEOMETRY_EPS: f64 = 1.0e-10;
const FRAME_EPS: f64 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContactPatchGeometrySource {
    /// Declared morphology geometry. This is a model declaration, not a measurement.
    MorphologyDeclaration,
    /// Geometry extracted from a simulator/model asset.
    SimulatorGeometry,
    /// Geometry obtained from an explicit hardware calibration procedure.
    HardwareCalibration,
}

/// Immutable local support-surface geometry.
///
/// Friction is deliberately absent: interaction limits depend on the pair of
/// contacting materials/surfaces and must be supplied as separate evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContactPatchGeometryV1 {
    pub site: ContactSite,
    /// Counter-clockwise convex polygon in the contact-local tangent plane.
    pub vertices_local_xy_m: Vec<[f64; 2]>,
    pub source: ContactPatchGeometrySource,
    pub geometry_id: String,
}

impl ContactPatchGeometryV1 {
    pub fn validate(&self) -> bool {
        let vertices = &self.vertices_local_xy_m;
        if !(3..=MAX_PATCH_VERTICES).contains(&vertices.len())
            || self.geometry_id.trim().is_empty()
            || vertices
                .iter()
                .flat_map(|vertex| vertex.iter())
                .any(|value| !value.is_finite())
        {
            return false;
        }

        // Require a deterministic CCW convex boundary. This rejects repeated,
        // collinear, self-intersecting, and unordered vertex sets rather than
        // silently repairing them into a different physical patch.
        if signed_polygon_area(vertices) <= GEOMETRY_EPS {
            return false;
        }
        (0..vertices.len()).all(|index| {
            let a = vertices[index];
            let b = vertices[(index + 1) % vertices.len()];
            let c = vertices[(index + 2) % vertices.len()];
            cross2(a, b, c) > GEOMETRY_EPS
        })
    }

    pub fn area_m2(&self) -> Option<f64> {
        self.validate()
            .then(|| signed_polygon_area(&self.vertices_local_xy_m))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContactPatchPlacementSource {
    /// Pose produced by simulator kinematics at the stated sample instant.
    SimulatorKinematics,
    /// Pose estimated from hardware sensing/state estimation.
    HardwareEstimate,
    /// Declared nominal pose for static fixtures/bench geometry.
    DeclaredNominal,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlacedContactPatchV1 {
    pub geometry: ContactPatchGeometryV1,
    pub origin_world_m: [f64; 3],
    pub tangent_x_world: [f64; 3],
    pub tangent_y_world: [f64; 3],
    pub normal_world: [f64; 3],
    pub sampled_at_s: f64,
    pub placement_source: ContactPatchPlacementSource,
    pub confidence: f64,
}

impl PlacedContactPatchV1 {
    pub fn validate(&self) -> bool {
        self.geometry.validate()
            && self.origin_world_m.iter().all(|value| value.is_finite())
            && self.tangent_x_world.iter().all(|value| value.is_finite())
            && self.tangent_y_world.iter().all(|value| value.is_finite())
            && self.normal_world.iter().all(|value| value.is_finite())
            && self.sampled_at_s.is_finite()
            && self.sampled_at_s >= 0.0
            && self.confidence.is_finite()
            && (0.0..=1.0).contains(&self.confidence)
            && (norm3(self.tangent_x_world) - 1.0).abs() <= FRAME_EPS
            && (norm3(self.tangent_y_world) - 1.0).abs() <= FRAME_EPS
            && (norm3(self.normal_world) - 1.0).abs() <= FRAME_EPS
            && dot3(self.tangent_x_world, self.tangent_y_world).abs() <= FRAME_EPS
            && dot3(self.tangent_x_world, self.normal_world).abs() <= FRAME_EPS
            && dot3(self.tangent_y_world, self.normal_world).abs() <= FRAME_EPS
            && dot3(
                cross3(self.tangent_x_world, self.tangent_y_world),
                self.normal_world,
            ) >= 1.0 - FRAME_EPS
    }

    pub fn vertices_world_m(&self) -> Option<Vec<[f64; 3]>> {
        self.validate().then(|| {
            self.geometry
                .vertices_local_xy_m
                .iter()
                .map(|[x, y]| {
                    [
                        self.origin_world_m[0]
                            + x * self.tangent_x_world[0]
                            + y * self.tangent_y_world[0],
                        self.origin_world_m[1]
                            + x * self.tangent_x_world[1]
                            + y * self.tangent_y_world[1],
                        self.origin_world_m[2]
                            + x * self.tangent_x_world[2]
                            + y * self.tangent_y_world[2],
                    ]
                })
                .collect()
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SupportRegionError {
    InvalidContactFrame,
    InvalidPatch,
    DuplicatePatchSite,
    MissingActivePatch,
    SampleTimeMismatch,
}

/// Ground-projected convex support-region area from explicit physical patches.
///
/// Every active contact must have exactly one valid, same-time patch placement.
/// Extra patches for inactive contacts are ignored and therefore cannot create
/// fictitious support. Missing geometry fails closed instead of falling back to
/// point contacts.
pub fn support_region_area_m2(
    frame: &MultiContactFrame,
    patches: &[PlacedContactPatchV1],
) -> Result<f64, SupportRegionError> {
    if !frame.validate() {
        return Err(SupportRegionError::InvalidContactFrame);
    }

    for (index, patch) in patches.iter().enumerate() {
        if !patch.validate() {
            return Err(SupportRegionError::InvalidPatch);
        }
        if patches[..index]
            .iter()
            .any(|earlier| earlier.geometry.site == patch.geometry.site)
        {
            return Err(SupportRegionError::DuplicatePatchSite);
        }
    }

    let mut points = Vec::<[f64; 2]>::new();
    for contact in frame.active() {
        let patch = patches
            .iter()
            .find(|patch| patch.geometry.site == contact.site)
            .ok_or(SupportRegionError::MissingActivePatch)?;
        if !same_sample_time(frame.timestamp, patch.sampled_at_s) {
            return Err(SupportRegionError::SampleTimeMismatch);
        }
        let world_vertices = patch
            .vertices_world_m()
            .ok_or(SupportRegionError::InvalidPatch)?;
        points.extend(
            world_vertices
                .into_iter()
                .map(|vertex| [vertex[0], vertex[1]]),
        );
    }

    if points.is_empty() {
        return Ok(0.0);
    }
    Ok(convex_hull_area(points))
}

fn same_sample_time(left: f64, right: f64) -> bool {
    let tolerance = 1.0e-9 * (1.0 + left.abs().max(right.abs()));
    (left - right).abs() <= tolerance
}

fn convex_hull_area(mut points: Vec<[f64; 2]>) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    points.sort_by(|left, right| {
        left[0]
            .total_cmp(&right[0])
            .then_with(|| left[1].total_cmp(&right[1]))
    });
    points.dedup_by(|left, right| left == right);
    if points.len() < 3 {
        return 0.0;
    }

    let mut lower = Vec::<[f64; 2]>::new();
    for point in &points {
        while lower.len() >= 2
            && cross2(lower[lower.len() - 2], lower[lower.len() - 1], *point) <= 0.0
        {
            lower.pop();
        }
        lower.push(*point);
    }
    let mut upper = Vec::<[f64; 2]>::new();
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
    signed_polygon_area(&lower).abs()
}

fn signed_polygon_area(points: &[[f64; 2]]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    let twice_area = (0..points.len())
        .map(|index| {
            let next = (index + 1) % points.len();
            points[index][0] * points[next][1] - points[next][0] * points[index][1]
        })
        .sum::<f64>();
    0.5 * twice_area
}

fn cross2(origin: [f64; 2], left: [f64; 2], right: [f64; 2]) -> f64 {
    (left[0] - origin[0]) * (right[1] - origin[1])
        - (left[1] - origin[1]) * (right[0] - origin[0])
}

fn dot3(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn norm3(vector: [f64; 3]) -> f64 {
    dot3(vector, vector).sqrt()
}

fn cross3(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contact::{ContactFrame, ContactSource, FootContact};

    fn rectangle(site: ContactSite, half_x: f64, half_y: f64) -> ContactPatchGeometryV1 {
        ContactPatchGeometryV1 {
            site,
            vertices_local_xy_m: vec![
                [-half_x, -half_y],
                [half_x, -half_y],
                [half_x, half_y],
                [-half_x, half_y],
            ],
            source: ContactPatchGeometrySource::MorphologyDeclaration,
            geometry_id: format!("synthetic-{site:?}"),
        }
    }

    fn placed(site: ContactSite, x: f64, y: f64) -> PlacedContactPatchV1 {
        PlacedContactPatchV1 {
            geometry: rectangle(site, 0.10, 0.04),
            origin_world_m: [x, y, 0.0],
            tangent_x_world: [1.0, 0.0, 0.0],
            tangent_y_world: [0.0, 1.0, 0.0],
            normal_world: [0.0, 0.0, 1.0],
            sampled_at_s: 1.0,
            placement_source: ContactPatchPlacementSource::SimulatorKinematics,
            confidence: 1.0,
        }
    }

    fn foot(active: bool, x: f64, y: f64) -> FootContact {
        FootContact {
            in_contact: active,
            point_world_m: [x, y, 0.0],
            force_world_n: [0.0, 0.0, if active { 100.0 } else { 0.0 }],
            torque_world_nm: [0.0; 3],
            center_of_pressure_world_m: [x, y],
            confidence: 1.0,
        }
    }

    fn frame(right_active: bool, left_active: bool) -> MultiContactFrame {
        MultiContactFrame::from_feet(&ContactFrame {
            timestamp: 1.0,
            source: ContactSource::SolverWrench,
            right: foot(right_active, 0.0, -0.10),
            left: foot(left_active, 0.0, 0.10),
        })
    }

    #[test]
    fn geometry_identity_does_not_embed_interaction_friction() {
        let geometry = rectangle(ContactSite::RightFoot, 0.10, 0.04);
        assert!(geometry.validate());
        assert_eq!(geometry.area_m2(), Some(0.016));
    }

    #[test]
    fn one_active_foot_has_nonzero_physical_support_area() {
        let patches = [
            placed(ContactSite::RightFoot, 0.0, -0.10),
            placed(ContactSite::LeftFoot, 0.0, 0.10),
        ];
        let area = support_region_area_m2(&frame(true, false), &patches).unwrap();
        assert!((area - 0.016).abs() < 1.0e-12);
    }

    #[test]
    fn two_feet_expand_support_region_beyond_individual_patch() {
        let patches = [
            placed(ContactSite::RightFoot, 0.0, -0.10),
            placed(ContactSite::LeftFoot, 0.0, 0.10),
        ];
        let area = support_region_area_m2(&frame(true, true), &patches).unwrap();
        assert!(area > 0.016);
    }

    #[test]
    fn missing_active_patch_fails_closed() {
        let patches = [placed(ContactSite::LeftFoot, 0.0, 0.10)];
        assert_eq!(
            support_region_area_m2(&frame(true, false), &patches),
            Err(SupportRegionError::MissingActivePatch)
        );
    }

    #[test]
    fn inactive_patch_cannot_create_support() {
        let patches = [
            placed(ContactSite::RightFoot, 0.0, -0.10),
            placed(ContactSite::LeftFoot, 10.0, 10.0),
        ];
        let area = support_region_area_m2(&frame(true, false), &patches).unwrap();
        assert!((area - 0.016).abs() < 1.0e-12);
    }

    #[test]
    fn stale_patch_placement_is_rejected() {
        let mut patch = placed(ContactSite::RightFoot, 0.0, -0.10);
        patch.sampled_at_s = 0.5;
        assert_eq!(
            support_region_area_m2(&frame(true, false), &[patch]),
            Err(SupportRegionError::SampleTimeMismatch)
        );
    }

    #[test]
    fn malformed_or_clockwise_geometry_is_not_repaired() {
        let mut geometry = rectangle(ContactSite::RightFoot, 0.10, 0.04);
        geometry.vertices_local_xy_m.reverse();
        assert!(!geometry.validate());
    }

    #[test]
    fn placement_frame_must_be_right_handed_orthonormal() {
        let mut patch = placed(ContactSite::RightFoot, 0.0, 0.0);
        patch.tangent_y_world = [1.0, 0.0, 0.0];
        assert!(!patch.validate());
    }
}
