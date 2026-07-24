// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bounded triangle self-intersection evidence for fabrication meshes.
//!
//! The implementation uses a sweep-and-prune broad phase followed by
//! triangle/triangle narrow-phase tests. Adjacent faces that share a complete
//! geometric edge are ignored; contacts confined to a shared vertex are not
//! reported as self-intersections.

use crate::mesh::TriangleMesh;

/// Default maximum number of broad-phase candidate pairs examined per mesh.
pub const DEFAULT_SELF_INTERSECTION_PAIR_BUDGET: usize = 2_000_000;

/// Result of a bounded self-intersection scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfIntersectionReport {
    /// Non-adjacent triangle index pairs whose interiors overlap.
    pub triangle_pairs: Vec<[usize; 2]>,
    /// Number of broad-phase candidates examined by the narrow phase.
    pub candidate_pairs_tested: usize,
    /// True when the pair budget was exhausted before the scan completed.
    pub truncated: bool,
}

impl SelfIntersectionReport {
    pub fn is_clear(&self) -> bool {
        !self.truncated && self.triangle_pairs.is_empty()
    }
}

#[derive(Clone, Copy)]
struct Triangle {
    points: [[f32; 3]; 3],
    bbox_min: [f32; 3],
    bbox_max: [f32; 3],
}

/// Find non-adjacent triangle self-intersections with a bounded work budget.
pub fn find_self_intersections(
    mesh: &TriangleMesh,
    epsilon: f32,
    pair_budget: usize,
) -> SelfIntersectionReport {
    let mut triangles: Vec<(usize, Triangle)> = mesh
        .indices
        .iter()
        .enumerate()
        .filter_map(|(triangle_index, triangle)| {
            let indices = [
                triangle[0] as usize,
                triangle[1] as usize,
                triangle[2] as usize,
            ];
            if indices.iter().any(|index| *index >= mesh.vertices.len()) {
                return None;
            }
            let points = [
                mesh.vertices[indices[0]],
                mesh.vertices[indices[1]],
                mesh.vertices[indices[2]],
            ];
            if points
                .iter()
                .flatten()
                .any(|component| !component.is_finite())
            {
                return None;
            }
            Some((triangle_index, Triangle::new(points)))
        })
        .collect();

    triangles.sort_by(|(_, left), (_, right)| {
        left.bbox_min[0]
            .partial_cmp(&right.bbox_min[0])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut report = SelfIntersectionReport {
        triangle_pairs: Vec::new(),
        candidate_pairs_tested: 0,
        truncated: false,
    };

    'outer: for left_position in 0..triangles.len() {
        let (left_index, left) = triangles[left_position];
        for &(right_index, right) in triangles.iter().skip(left_position + 1) {
            if right.bbox_min[0] > left.bbox_max[0] + epsilon {
                break;
            }
            if !aabb_overlaps(&left, &right, epsilon) {
                continue;
            }
            if report.candidate_pairs_tested >= pair_budget {
                report.truncated = true;
                break 'outer;
            }
            report.candidate_pairs_tested += 1;

            let shared_points = shared_points(&left.points, &right.points, epsilon);
            if shared_points.len() >= 2 {
                // Manifold neighboring faces share a complete edge. Duplicate
                // and non-manifold triangles are reported by topology checks.
                continue;
            }

            if triangles_intersect(&left.points, &right.points, &shared_points, epsilon) {
                report.triangle_pairs.push([left_index, right_index]);
            }
        }
    }

    report
}

impl Triangle {
    fn new(points: [[f32; 3]; 3]) -> Self {
        let mut bbox_min = points[0];
        let mut bbox_max = points[0];
        for point in points.iter().skip(1) {
            for axis in 0..3 {
                bbox_min[axis] = bbox_min[axis].min(point[axis]);
                bbox_max[axis] = bbox_max[axis].max(point[axis]);
            }
        }
        Self {
            points,
            bbox_min,
            bbox_max,
        }
    }
}

fn aabb_overlaps(left: &Triangle, right: &Triangle, epsilon: f32) -> bool {
    (0..3).all(|axis| {
        left.bbox_min[axis] <= right.bbox_max[axis] + epsilon
            && right.bbox_min[axis] <= left.bbox_max[axis] + epsilon
    })
}

fn shared_points(left: &[[f32; 3]; 3], right: &[[f32; 3]; 3], epsilon: f32) -> Vec<[f32; 3]> {
    let mut shared = Vec::new();
    for left_point in left {
        if right
            .iter()
            .any(|right_point| distance_squared(*left_point, *right_point) <= epsilon * epsilon)
        {
            shared.push(*left_point);
        }
    }
    shared
}

fn triangles_intersect(
    left: &[[f32; 3]; 3],
    right: &[[f32; 3]; 3],
    shared: &[[f32; 3]],
    epsilon: f32,
) -> bool {
    let left_normal = cross(sub(left[1], left[0]), sub(left[2], left[0]));
    let right_normal = cross(sub(right[1], right[0]), sub(right[2], right[0]));
    let left_len = norm(left_normal);
    let right_len = norm(right_normal);
    if left_len <= epsilon || right_len <= epsilon {
        return false;
    }

    let left_unit = scale(left_normal, 1.0 / left_len);
    let right_unit = scale(right_normal, 1.0 / right_len);
    let right_distances = [
        dot(left_unit, sub(right[0], left[0])),
        dot(left_unit, sub(right[1], left[0])),
        dot(left_unit, sub(right[2], left[0])),
    ];
    let left_distances = [
        dot(right_unit, sub(left[0], right[0])),
        dot(right_unit, sub(left[1], right[0])),
        dot(right_unit, sub(left[2], right[0])),
    ];

    if strictly_one_sided(&right_distances, epsilon) || strictly_one_sided(&left_distances, epsilon)
    {
        return false;
    }

    if norm(cross(left_unit, right_unit)) <= epsilon {
        if right_distances
            .iter()
            .any(|distance| distance.abs() > epsilon)
        {
            return false;
        }
        return coplanar_triangles_overlap(left, right, shared, left_unit, epsilon);
    }

    let mut intersections = Vec::new();
    collect_edge_triangle_hits(left, right, epsilon, &mut intersections);
    collect_edge_triangle_hits(right, left, epsilon, &mut intersections);
    deduplicate_points(&mut intersections, epsilon);
    intersections.into_iter().any(|point| {
        !shared
            .iter()
            .any(|shared_point| distance_squared(point, *shared_point) <= epsilon * epsilon)
    })
}

fn collect_edge_triangle_hits(
    source: &[[f32; 3]; 3],
    target: &[[f32; 3]; 3],
    epsilon: f32,
    output: &mut Vec<[f32; 3]>,
) {
    for edge in [[0usize, 1usize], [1, 2], [2, 0]] {
        if let Some(point) =
            segment_triangle_intersection(source[edge[0]], source[edge[1]], target, epsilon)
        {
            output.push(point);
        }
    }
}

fn segment_triangle_intersection(
    start: [f32; 3],
    end: [f32; 3],
    triangle: &[[f32; 3]; 3],
    epsilon: f32,
) -> Option<[f32; 3]> {
    let direction = sub(end, start);
    let edge1 = sub(triangle[1], triangle[0]);
    let edge2 = sub(triangle[2], triangle[0]);
    let p = cross(direction, edge2);
    let determinant = dot(edge1, p);
    if determinant.abs() <= epsilon {
        return None;
    }
    let inverse = 1.0 / determinant;
    let tvec = sub(start, triangle[0]);
    let u = dot(tvec, p) * inverse;
    if u < -epsilon || u > 1.0 + epsilon {
        return None;
    }
    let q = cross(tvec, edge1);
    let v = dot(direction, q) * inverse;
    if v < -epsilon || u + v > 1.0 + epsilon {
        return None;
    }
    let t = dot(edge2, q) * inverse;
    if t < -epsilon || t > 1.0 + epsilon {
        return None;
    }
    Some(add(start, scale(direction, t.clamp(0.0, 1.0))))
}

fn coplanar_triangles_overlap(
    left: &[[f32; 3]; 3],
    right: &[[f32; 3]; 3],
    shared: &[[f32; 3]],
    normal: [f32; 3],
    epsilon: f32,
) -> bool {
    let axis = dominant_axis(normal);
    let left_2d = left.map(|point| project(point, axis));
    let right_2d = right.map(|point| project(point, axis));

    for (point_3d, point_2d) in left.iter().zip(left_2d.iter()) {
        if !is_shared_point(*point_3d, shared, epsilon)
            && point_strictly_in_triangle_2d(*point_2d, &right_2d, epsilon)
        {
            return true;
        }
    }
    for (point_3d, point_2d) in right.iter().zip(right_2d.iter()) {
        if !is_shared_point(*point_3d, shared, epsilon)
            && point_strictly_in_triangle_2d(*point_2d, &left_2d, epsilon)
        {
            return true;
        }
    }

    for left_edge in [[0usize, 1usize], [1, 2], [2, 0]] {
        for right_edge in [[0usize, 1usize], [1, 2], [2, 0]] {
            if segments_properly_intersect_2d(
                left_2d[left_edge[0]],
                left_2d[left_edge[1]],
                right_2d[right_edge[0]],
                right_2d[right_edge[1]],
                epsilon,
            ) {
                return true;
            }
        }
    }
    false
}

fn point_strictly_in_triangle_2d(point: [f32; 2], triangle: &[[f32; 2]; 3], epsilon: f32) -> bool {
    let o1 = orient_2d(triangle[0], triangle[1], point);
    let o2 = orient_2d(triangle[1], triangle[2], point);
    let o3 = orient_2d(triangle[2], triangle[0], point);
    (o1 > epsilon && o2 > epsilon && o3 > epsilon)
        || (o1 < -epsilon && o2 < -epsilon && o3 < -epsilon)
}

fn segments_properly_intersect_2d(
    a: [f32; 2],
    b: [f32; 2],
    c: [f32; 2],
    d: [f32; 2],
    epsilon: f32,
) -> bool {
    let ab_c = orient_2d(a, b, c);
    let ab_d = orient_2d(a, b, d);
    let cd_a = orient_2d(c, d, a);
    let cd_b = orient_2d(c, d, b);
    ab_c * ab_d < -epsilon * epsilon && cd_a * cd_b < -epsilon * epsilon
}

fn orient_2d(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

fn dominant_axis(normal: [f32; 3]) -> usize {
    let absolute = [normal[0].abs(), normal[1].abs(), normal[2].abs()];
    if absolute[0] >= absolute[1] && absolute[0] >= absolute[2] {
        0
    } else if absolute[1] >= absolute[2] {
        1
    } else {
        2
    }
}

fn project(point: [f32; 3], omitted_axis: usize) -> [f32; 2] {
    match omitted_axis {
        0 => [point[1], point[2]],
        1 => [point[0], point[2]],
        _ => [point[0], point[1]],
    }
}

fn is_shared_point(point: [f32; 3], shared: &[[f32; 3]], epsilon: f32) -> bool {
    shared
        .iter()
        .any(|shared_point| distance_squared(point, *shared_point) <= epsilon * epsilon)
}

fn strictly_one_sided(distances: &[f32; 3], epsilon: f32) -> bool {
    distances.iter().all(|distance| *distance > epsilon)
        || distances.iter().all(|distance| *distance < -epsilon)
}

fn deduplicate_points(points: &mut Vec<[f32; 3]>, epsilon: f32) {
    let mut unique = Vec::new();
    for point in points.drain(..) {
        if !unique
            .iter()
            .any(|existing| distance_squared(point, *existing) <= epsilon * epsilon)
        {
            unique.push(point);
        }
    }
    *points = unique;
}

fn add(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn sub(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn scale(value: [f32; 3], scalar: f32) -> [f32; 3] {
    [value[0] * scalar, value[1] * scalar, value[2] * scalar]
}

fn dot(left: [f32; 3], right: [f32; 3]) -> f32 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn norm(value: [f32; 3]) -> f32 {
    dot(value, value).sqrt()
}

fn distance_squared(left: [f32; 3], right: [f32; 3]) -> f32 {
    let difference = sub(left, right);
    dot(difference, difference)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csg::{CSGNode, Transform3D};
    use crate::mesh::resolve_to_mesh;

    #[test]
    fn closed_cube_has_no_self_intersections() {
        let cube = resolve_to_mesh(&CSGNode::cube());
        let report = find_self_intersections(&cube, 1.0e-5, 100_000);
        assert!(
            report.is_clear(),
            "unexpected pairs: {:?}",
            report.triangle_pairs
        );
    }

    #[test]
    fn overlapping_disconnected_cubes_are_detected() {
        let mut left = resolve_to_mesh(&CSGNode::cube());
        let right = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
            translate: [0.25, 0.25, 0.25],
            ..Default::default()
        }));
        left.merge(&right);
        let report = find_self_intersections(&left, 1.0e-5, 100_000);
        assert!(!report.triangle_pairs.is_empty());
        assert!(!report.truncated);
    }

    #[test]
    fn disjoint_cubes_are_clear() {
        let mut left = resolve_to_mesh(&CSGNode::cube());
        let right = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
            translate: [2.0, 0.0, 0.0],
            ..Default::default()
        }));
        left.merge(&right);
        assert!(find_self_intersections(&left, 1.0e-5, 100_000).is_clear());
    }

    #[test]
    fn pair_budget_fails_closed() {
        let mut mesh = resolve_to_mesh(&CSGNode::cube());
        let second = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
            translate: [0.1, 0.1, 0.1],
            ..Default::default()
        }));
        mesh.merge(&second);
        let report = find_self_intersections(&mesh, 1.0e-5, 0);
        assert!(report.truncated);
        assert!(!report.is_clear());
    }
}
