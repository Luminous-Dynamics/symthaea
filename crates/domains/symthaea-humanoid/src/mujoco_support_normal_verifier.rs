// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent internal-consistency verification for HUM-WRENCH-001D2 evidence.
//!
//! This module deliberately re-derives the small amount of policy/vector/hull
//! arithmetic encoded in `NormalQualifiedSupportRegionV1`. It does not call
//! MuJoCo and does not establish that simulator contact normals are physically
//! true. Its authority is limited to checking that a serialized evidence record
//! is internally consistent with its own raw fields and declared policy.

use serde::{Deserialize, Serialize};

use crate::mujoco_support_normal::{
    NormalQualifiedSupportRegionV1, SupportNormalPolicyV1,
};

const NORMAL_EPS: f64 = 1.0e-12;
const NUMERIC_TOLERANCE: f64 = 1.0e-12;
const REGION_AREA_EPS: f64 = 1.0e-12;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SupportNormalEvidenceVerificationV1 {
    pub policy_id: String,
    pub source_contact_count: usize,
    pub admitted_contact_count: usize,
    pub rejected_contact_count: usize,
    pub recomputed_hull_vertices_local_xy_m: Vec<[f64; 2]>,
    pub recomputed_area_m2: f64,
    pub surface_support_eligible: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportNormalEvidenceVerificationError {
    InvalidSourceRecord,
    InvalidPolicy,
    PolicyIdentityMismatch,
    PointContactIndexMismatch,
    InvalidNormal,
    AlignmentMismatch,
    AdmissibilityMismatch,
    AdmittedIndicesMismatch,
    RejectedIndicesMismatch,
    HullMismatch,
    AreaMismatch,
    SurfaceEligibilityMismatch,
}

/// Independently re-derive every authority-relevant relationship carried by a
/// HUM-WRENCH-001D2 support-normal record.
///
/// Passing this verifier means only that the evidence is internally consistent.
/// It does not qualify the chosen alignment threshold or the physical truth of
/// any simulator-derived contact normal.
pub fn verify_support_normal_evidence_v1(
    region: &NormalQualifiedSupportRegionV1,
) -> Result<SupportNormalEvidenceVerificationV1, SupportNormalEvidenceVerificationError> {
    if !region.validate() {
        return Err(SupportNormalEvidenceVerificationError::InvalidSourceRecord);
    }

    let policy = SupportNormalPolicyV1 {
        minimum_abs_alignment: region.minimum_abs_alignment,
    };
    if !policy.validate() {
        return Err(SupportNormalEvidenceVerificationError::InvalidPolicy);
    }
    if policy.policy_id().as_deref() != Some(region.policy_id.as_str()) {
        return Err(SupportNormalEvidenceVerificationError::PolicyIdentityMismatch);
    }

    if region.point_evidence.len() != region.source_contact_indices.len() {
        return Err(SupportNormalEvidenceVerificationError::InvalidSourceRecord);
    }

    let mut expected_admitted = Vec::new();
    let mut expected_rejected = Vec::new();
    let mut admitted_points = Vec::new();

    for (position, point) in region.point_evidence.iter().enumerate() {
        if point.contact_index != region.source_contact_indices[position] {
            return Err(SupportNormalEvidenceVerificationError::PointContactIndexMismatch);
        }

        let recomputed_alignment = independent_absolute_alignment(
            point.contact_normal_world,
            point.sole_normal_world,
        )?;
        if !close(recomputed_alignment, point.absolute_alignment) {
            return Err(SupportNormalEvidenceVerificationError::AlignmentMismatch);
        }

        let expected_admissible =
            recomputed_alignment + NUMERIC_TOLERANCE >= region.minimum_abs_alignment;
        if point.admissible != expected_admissible {
            return Err(SupportNormalEvidenceVerificationError::AdmissibilityMismatch);
        }

        if expected_admissible {
            expected_admitted.push(point.contact_index);
            admitted_points.push(point.local_xy_m);
        } else {
            expected_rejected.push(point.contact_index);
        }
    }

    if expected_admitted != region.admitted_contact_indices {
        return Err(SupportNormalEvidenceVerificationError::AdmittedIndicesMismatch);
    }
    if expected_rejected != region.rejected_contact_indices {
        return Err(SupportNormalEvidenceVerificationError::RejectedIndicesMismatch);
    }

    let recomputed_hull = independent_convex_hull(admitted_points);
    if !point_vectors_close(&recomputed_hull, &region.hull_vertices_local_xy_m) {
        return Err(SupportNormalEvidenceVerificationError::HullMismatch);
    }

    let recomputed_area_m2 = independent_polygon_area(&recomputed_hull);
    if !close(recomputed_area_m2, region.area_m2) {
        return Err(SupportNormalEvidenceVerificationError::AreaMismatch);
    }

    let expected_surface_support =
        recomputed_hull.len() >= 3 && recomputed_area_m2 > REGION_AREA_EPS;
    if expected_surface_support != region.surface_support_eligible() {
        return Err(SupportNormalEvidenceVerificationError::SurfaceEligibilityMismatch);
    }

    Ok(SupportNormalEvidenceVerificationV1 {
        policy_id: region.policy_id.clone(),
        source_contact_count: region.source_contact_indices.len(),
        admitted_contact_count: expected_admitted.len(),
        rejected_contact_count: expected_rejected.len(),
        recomputed_hull_vertices_local_xy_m: recomputed_hull,
        recomputed_area_m2,
        surface_support_eligible: expected_surface_support,
    })
}

fn independent_absolute_alignment(
    contact_normal_world: [f64; 3],
    sole_normal_world: [f64; 3],
) -> Result<f64, SupportNormalEvidenceVerificationError> {
    if contact_normal_world.iter().any(|value| !value.is_finite())
        || sole_normal_world.iter().any(|value| !value.is_finite())
    {
        return Err(SupportNormalEvidenceVerificationError::InvalidNormal);
    }
    let contact_norm = independent_norm3(contact_normal_world);
    let sole_norm = independent_norm3(sole_normal_world);
    if contact_norm <= NORMAL_EPS || sole_norm <= NORMAL_EPS {
        return Err(SupportNormalEvidenceVerificationError::InvalidNormal);
    }
    Ok((independent_dot3(contact_normal_world, sole_normal_world)
        / (contact_norm * sole_norm))
        .abs()
        .clamp(0.0, 1.0))
}

fn independent_convex_hull(mut points: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
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
            && independent_cross2(lower[lower.len() - 2], lower[lower.len() - 1], *point) <= 0.0
        {
            lower.pop();
        }
        lower.push(*point);
    }

    let mut upper = Vec::new();
    for point in points.iter().rev() {
        while upper.len() >= 2
            && independent_cross2(upper[upper.len() - 2], upper[upper.len() - 1], *point) <= 0.0
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

fn independent_polygon_area(points: &[[f64; 2]]) -> f64 {
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

fn point_vectors_close(left: &[[f64; 2]], right: &[[f64; 2]]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            close(left[0], right[0]) && close(left[1], right[1])
        })
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs()
        <= NUMERIC_TOLERANCE * (1.0 + left.abs().max(right.abs()))
}

fn independent_norm3(vector: [f64; 3]) -> f64 {
    independent_dot3(vector, vector).sqrt()
}

fn independent_dot3(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn independent_cross2(origin: [f64; 2], left: [f64; 2], right: [f64; 2]) -> f64 {
    (left[0] - origin[0]) * (right[1] - origin[1])
        - (left[1] - origin[1]) * (right[0] - origin[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_contact::ContactSite;
    use crate::mujoco_support_normal::SupportNormalPointEvidenceV1;

    fn valid_region() -> NormalQualifiedSupportRegionV1 {
        let minimum_abs_alignment = 0.8;
        let policy = SupportNormalPolicyV1 {
            minimum_abs_alignment,
        };
        let local_points = [
            [-0.10, -0.04],
            [0.10, -0.04],
            [0.10, 0.04],
            [-0.10, 0.04],
        ];
        let source_contact_indices = vec![2, 4, 6, 8];
        let point_evidence = source_contact_indices
            .iter()
            .copied()
            .zip(local_points)
            .enumerate()
            .map(|(position, (contact_index, local_xy_m))| SupportNormalPointEvidenceV1 {
                contact_index,
                local_xy_m,
                contact_normal_world: if position % 2 == 0 {
                    [0.0, 0.0, 1.0]
                } else {
                    [0.0, 0.0, -1.0]
                },
                sole_normal_world: [0.0, 0.0, 1.0],
                absolute_alignment: 1.0,
                admissible: true,
            })
            .collect::<Vec<_>>();
        let admitted_points = point_evidence
            .iter()
            .map(|point| point.local_xy_m)
            .collect::<Vec<_>>();
        let hull_vertices_local_xy_m = independent_convex_hull(admitted_points);
        let area_m2 = independent_polygon_area(&hull_vertices_local_xy_m);

        NormalQualifiedSupportRegionV1 {
            site: ContactSite::RightFoot,
            model_id: "synthetic-model-v1".to_string(),
            model_signature: 42,
            sampled_at_s: 1.0,
            source_geometry_id: "synthetic-geometry-v1".to_string(),
            source_interaction_id: "synthetic-interaction-v1".to_string(),
            policy_id: policy.policy_id().unwrap(),
            minimum_abs_alignment,
            source_contact_indices: source_contact_indices.clone(),
            admitted_contact_indices: source_contact_indices,
            rejected_contact_indices: Vec::new(),
            point_evidence,
            hull_vertices_local_xy_m,
            area_m2,
        }
    }

    #[test]
    fn valid_record_is_independently_rederived() {
        let region = valid_region();
        assert!(region.validate());
        let verification = verify_support_normal_evidence_v1(&region).unwrap();
        assert_eq!(verification.source_contact_count, 4);
        assert_eq!(verification.admitted_contact_count, 4);
        assert_eq!(verification.rejected_contact_count, 0);
        assert!(verification.surface_support_eligible);
        assert!(close(verification.recomputed_area_m2, 0.016));
    }

    #[test]
    fn policy_id_tampering_is_rejected() {
        let mut region = valid_region();
        region.policy_id.push_str(":tampered");
        assert!(region.validate());
        assert_eq!(
            verify_support_normal_evidence_v1(&region),
            Err(SupportNormalEvidenceVerificationError::PolicyIdentityMismatch)
        );
    }

    #[test]
    fn stored_alignment_tampering_is_rejected() {
        let mut region = valid_region();
        region.point_evidence[0].absolute_alignment = 0.9;
        assert!(region.validate());
        assert_eq!(
            verify_support_normal_evidence_v1(&region),
            Err(SupportNormalEvidenceVerificationError::AlignmentMismatch)
        );
    }

    #[test]
    fn admissibility_flag_tampering_is_rejected() {
        let mut region = valid_region();
        region.point_evidence[0].admissible = false;
        assert!(region.validate());
        assert_eq!(
            verify_support_normal_evidence_v1(&region),
            Err(SupportNormalEvidenceVerificationError::AdmissibilityMismatch)
        );
    }

    #[test]
    fn point_contact_index_tampering_is_rejected() {
        let mut region = valid_region();
        region.point_evidence[0].contact_index = 3;
        assert!(region.validate());
        assert_eq!(
            verify_support_normal_evidence_v1(&region),
            Err(SupportNormalEvidenceVerificationError::PointContactIndexMismatch)
        );
    }

    #[test]
    fn admitted_vector_tampering_is_rejected() {
        let mut region = valid_region();
        region.admitted_contact_indices = vec![2, 4, 6];
        region.rejected_contact_indices = vec![8];
        assert!(region.validate());
        assert_eq!(
            verify_support_normal_evidence_v1(&region),
            Err(SupportNormalEvidenceVerificationError::AdmittedIndicesMismatch)
        );
    }

    #[test]
    fn rejected_vector_tampering_is_rejected() {
        let mut region = valid_region();
        region.admitted_contact_indices = vec![2, 4, 8];
        region.rejected_contact_indices = vec![6];
        assert!(region.validate());
        assert_eq!(
            verify_support_normal_evidence_v1(&region),
            Err(SupportNormalEvidenceVerificationError::AdmittedIndicesMismatch)
        );
    }

    #[test]
    fn hull_tampering_is_rejected() {
        let mut region = valid_region();
        region.hull_vertices_local_xy_m[0][0] += 0.01;
        assert!(region.validate());
        assert_eq!(
            verify_support_normal_evidence_v1(&region),
            Err(SupportNormalEvidenceVerificationError::HullMismatch)
        );
    }

    #[test]
    fn area_tampering_is_rejected() {
        let mut region = valid_region();
        region.area_m2 += 0.01;
        assert!(region.validate());
        assert_eq!(
            verify_support_normal_evidence_v1(&region),
            Err(SupportNormalEvidenceVerificationError::AreaMismatch)
        );
    }

    #[test]
    fn sign_flipped_equivalent_normals_remain_valid() {
        let mut region = valid_region();
        region.point_evidence[0].contact_normal_world = [0.0, 0.0, -1.0];
        assert!(verify_support_normal_evidence_v1(&region).is_ok());
    }

    #[test]
    fn two_point_record_is_valid_but_not_areal_support() {
        let mut region = valid_region();
        region.source_contact_indices.truncate(2);
        region.admitted_contact_indices.truncate(2);
        region.point_evidence.truncate(2);
        region.hull_vertices_local_xy_m = independent_convex_hull(
            region
                .point_evidence
                .iter()
                .map(|point| point.local_xy_m)
                .collect(),
        );
        region.area_m2 = 0.0;
        assert!(region.validate());
        let verification = verify_support_normal_evidence_v1(&region).unwrap();
        assert!(!verification.surface_support_eligible);
        assert!(!region.surface_support_eligible());
    }
}
