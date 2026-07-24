// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bounded manufacturability evidence for minimum opposing-surface distance.
//!
//! The analysis casts inward rays from triangle centroids and records the first
//! opposing surface. It is a conservative screening oracle, not an exact medial
//! axis or signed-distance proof. Budget exhaustion and unresolved rays fail
//! closed at the authority boundary.

use crate::mesh::TriangleMesh;

/// Policy for bounded local-thickness screening.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MinimumFeaturePolicy {
    /// Smallest permitted opposing-surface distance, in millimetres.
    pub minimum_wall_thickness_mm: f32,
    /// Offset from the source face before ray testing, in millimetres.
    pub ray_origin_epsilon_mm: f32,
    /// Barycentric and parallelism tolerance.
    pub intersection_epsilon: f32,
    /// Hard bound on ray/triangle narrow-phase tests.
    pub max_ray_triangle_tests: usize,
}

impl Default for MinimumFeaturePolicy {
    fn default() -> Self {
        Self {
            minimum_wall_thickness_mm: 0.4,
            ray_origin_epsilon_mm: 1.0e-5,
            intersection_epsilon: 1.0e-7,
            max_ray_triangle_tests: 20_000_000,
        }
    }
}

/// Bounded thickness evidence for a closed mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct MinimumFeatureReport {
    pub rays_cast: usize,
    pub ray_triangle_tests: usize,
    pub minimum_observed_thickness_mm: Option<f32>,
    pub thin_source_triangles: Vec<usize>,
    pub unresolved_source_triangles: Vec<usize>,
    pub truncated: bool,
}

impl MinimumFeatureReport {
    pub fn passes(&self) -> bool {
        !self.truncated
            && self.thin_source_triangles.is_empty()
            && self.unresolved_source_triangles.is_empty()
    }
}

/// Input or bounded-work failure.
#[derive(Debug, Clone, PartialEq)]
pub enum MinimumFeatureError {
    InvalidPolicy(&'static str),
    InvalidTriangle { triangle: usize },
}

/// Screen a closed mesh for local opposing-surface distances.
pub fn analyze_minimum_features(
    mesh: &TriangleMesh,
    policy: MinimumFeaturePolicy,
) -> Result<MinimumFeatureReport, MinimumFeatureError> {
    validate_policy(policy)?;
    let mut report = MinimumFeatureReport {
        rays_cast: 0,
        ray_triangle_tests: 0,
        minimum_observed_thickness_mm: None,
        thin_source_triangles: Vec::new(),
        unresolved_source_triangles: Vec::new(),
        truncated: false,
    };

    for (source_index, source) in mesh.indices.iter().enumerate() {
        let source_vertices = triangle_vertices(mesh, source_index, *source)?;
        let Some(outward) = triangle_normal(source_vertices, policy.intersection_epsilon) else {
            return Err(MinimumFeatureError::InvalidTriangle {
                triangle: source_index,
            });
        };
        let centroid = [
            (source_vertices[0][0] + source_vertices[1][0] + source_vertices[2][0]) / 3.0,
            (source_vertices[0][1] + source_vertices[1][1] + source_vertices[2][1]) / 3.0,
            (source_vertices[0][2] + source_vertices[1][2] + source_vertices[2][2]) / 3.0,
        ];
        let direction = [-outward[0], -outward[1], -outward[2]];
        let origin = [
            centroid[0] + direction[0] * policy.ray_origin_epsilon_mm,
            centroid[1] + direction[1] * policy.ray_origin_epsilon_mm,
            centroid[2] + direction[2] * policy.ray_origin_epsilon_mm,
        ];
        report.rays_cast += 1;
        let mut nearest = None::<f32>;

        for (candidate_index, candidate) in mesh.indices.iter().enumerate() {
            if candidate_index == source_index {
                continue;
            }
            if report.ray_triangle_tests >= policy.max_ray_triangle_tests {
                report.truncated = true;
                return Ok(report);
            }
            report.ray_triangle_tests += 1;
            let candidate_vertices = triangle_vertices(mesh, candidate_index, *candidate)?;
            if let Some(distance) = ray_triangle_distance(
                origin,
                direction,
                candidate_vertices,
                policy.intersection_epsilon,
            ) && distance > policy.ray_origin_epsilon_mm
                && nearest.is_none_or(|current| distance < current)
            {
                nearest = Some(distance);
            }
        }

        if let Some(distance) = nearest {
            let corrected_distance = distance + policy.ray_origin_epsilon_mm;
            report.minimum_observed_thickness_mm = Some(
                report
                    .minimum_observed_thickness_mm
                    .map_or(corrected_distance, |current| {
                        current.min(corrected_distance)
                    }),
            );
            if corrected_distance + policy.intersection_epsilon < policy.minimum_wall_thickness_mm {
                report.thin_source_triangles.push(source_index);
            }
        } else {
            report.unresolved_source_triangles.push(source_index);
        }
    }

    Ok(report)
}

fn validate_policy(policy: MinimumFeaturePolicy) -> Result<(), MinimumFeatureError> {
    for (name, value) in [
        (
            "minimum_wall_thickness_mm",
            policy.minimum_wall_thickness_mm,
        ),
        ("ray_origin_epsilon_mm", policy.ray_origin_epsilon_mm),
        ("intersection_epsilon", policy.intersection_epsilon),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(MinimumFeatureError::InvalidPolicy(name));
        }
    }
    if policy.max_ray_triangle_tests == 0 {
        return Err(MinimumFeatureError::InvalidPolicy("max_ray_triangle_tests"));
    }
    Ok(())
}

fn triangle_vertices(
    mesh: &TriangleMesh,
    triangle_index: usize,
    triangle: [u32; 3],
) -> Result<[[f32; 3]; 3], MinimumFeatureError> {
    let indices = [
        triangle[0] as usize,
        triangle[1] as usize,
        triangle[2] as usize,
    ];
    if indices.iter().any(|index| *index >= mesh.vertices.len()) {
        return Err(MinimumFeatureError::InvalidTriangle {
            triangle: triangle_index,
        });
    }
    let vertices = [
        mesh.vertices[indices[0]],
        mesh.vertices[indices[1]],
        mesh.vertices[indices[2]],
    ];
    if vertices
        .iter()
        .flatten()
        .any(|component| !component.is_finite())
    {
        return Err(MinimumFeatureError::InvalidTriangle {
            triangle: triangle_index,
        });
    }
    Ok(vertices)
}

fn triangle_normal(vertices: [[f32; 3]; 3], epsilon: f32) -> Option<[f32; 3]> {
    let a = vertices[0];
    let b = vertices[1];
    let c = vertices[2];
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let normal = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if !length.is_finite() || length <= epsilon {
        return None;
    }
    Some([normal[0] / length, normal[1] / length, normal[2] / length])
}

fn ray_triangle_distance(
    origin: [f32; 3],
    direction: [f32; 3],
    triangle: [[f32; 3]; 3],
    epsilon: f32,
) -> Option<f32> {
    let edge1 = subtract(triangle[1], triangle[0]);
    let edge2 = subtract(triangle[2], triangle[0]);
    let p = cross(direction, edge2);
    let determinant = dot(edge1, p);
    if determinant.abs() <= epsilon {
        return None;
    }
    let inverse = 1.0 / determinant;
    let t = subtract(origin, triangle[0]);
    let u = dot(t, p) * inverse;
    if u < -epsilon || u > 1.0 + epsilon {
        return None;
    }
    let q = cross(t, edge1);
    let v = dot(direction, q) * inverse;
    if v < -epsilon || u + v > 1.0 + epsilon {
        return None;
    }
    let distance = dot(edge2, q) * inverse;
    (distance > epsilon && distance.is_finite()).then_some(distance)
}

fn subtract(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csg::{CSGNode, Transform3D};
    use crate::mesh::resolve_to_mesh;

    #[test]
    fn unit_cube_passes_default_wall_threshold() {
        let mesh = resolve_to_mesh(&CSGNode::cube());
        let report = analyze_minimum_features(&mesh, MinimumFeaturePolicy::default()).unwrap();
        assert!(report.passes(), "{report:#?}");
        assert!((report.minimum_observed_thickness_mm.unwrap() - 1.0).abs() < 1.0e-3);
    }

    #[test]
    fn thin_plate_is_rejected() {
        let mesh = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
            scale: [1.0, 1.0, 0.2],
            ..Default::default()
        }));
        let report = analyze_minimum_features(&mesh, MinimumFeaturePolicy::default()).unwrap();
        assert!(!report.passes());
        assert!(!report.thin_source_triangles.is_empty());
        assert!(report.minimum_observed_thickness_mm.unwrap() < 0.21);
    }

    #[test]
    fn analysis_budget_exhaustion_is_explicit() {
        let mesh = resolve_to_mesh(&CSGNode::cube());
        let report = analyze_minimum_features(
            &mesh,
            MinimumFeaturePolicy {
                max_ray_triangle_tests: 1,
                ..MinimumFeaturePolicy::default()
            },
        )
        .unwrap();
        assert!(report.truncated);
        assert!(!report.passes());
    }
}
