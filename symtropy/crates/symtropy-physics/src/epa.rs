// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EPA (Expanding Polytope Algorithm) for penetration depth and contact normal.
//!
//! Takes the GJK termination simplex (which contains the origin) and expands it
//! into the full Minkowski difference to find the closest face to the origin.
//! That face's normal and distance give the penetration normal and depth.
//!
//! Implemented for 2D (edge-based) and 3D (face-based). 4D falls back to
//! bounding-sphere approximation (ND EPA is combinatorially infeasible).

use nalgebra::SVector;
use symtropy_math::Shape;

const MAX_EPA_ITERATIONS: usize = 32;
const EPA_TOLERANCE: f64 = 1e-6;

/// EPA result: penetration normal and depth.
#[derive(Clone, Debug)]
pub struct EpaResult<const D: usize> {
    /// Contact normal pointing from A to B.
    pub normal: SVector<f64, D>,
    /// Penetration depth (positive = overlapping).
    pub depth: f64,
}

/// Compute penetration depth and normal using EPA.
///
/// `simplex` must contain the origin (from a successful GJK intersection test).
/// Returns None if the simplex is degenerate.
pub fn penetration<const D: usize>(
    shape_a: &dyn Shape<D>,
    pos_a: &SVector<f64, D>,
    shape_b: &dyn Shape<D>,
    pos_b: &SVector<f64, D>,
    simplex: &[SVector<f64, D>],
) -> Option<EpaResult<D>> {
    if D == 2 {
        epa_2d(shape_a, pos_a, shape_b, pos_b, simplex)
    } else if D == 3 {
        epa_3d(shape_a, pos_a, shape_b, pos_b, simplex)
    } else {
        // Fallback: approximate from bounding spheres
        let (_, ra) = shape_a.bounding_sphere();
        let (_, rb) = shape_b.bounding_sphere();
        let delta = pos_b - pos_a;
        let dist = delta.norm();
        if dist < 1e-15 {
            let mut normal = SVector::zeros();
            normal[0] = 1.0;
            return Some(EpaResult {
                normal,
                depth: ra + rb,
            });
        }
        Some(EpaResult {
            normal: delta / dist,
            depth: (ra + rb - dist).max(0.0),
        })
    }
}

/// Support point on the Minkowski difference.
fn mink_support<const D: usize>(
    a: &dyn Shape<D>,
    pa: &SVector<f64, D>,
    b: &dyn Shape<D>,
    pb: &SVector<f64, D>,
    dir: &SVector<f64, D>,
) -> SVector<f64, D> {
    (a.support(dir) + pa) - (b.support(&-dir) + pb)
}

// ═══════════════════════════════════════════════════════════════════════════
// 2D EPA: Edge-based polytope expansion
// ═══════════════════════════════════════════════════════════════════════════

/// 2D EPA edge.
struct Edge2D<const D: usize> {
    a: SVector<f64, D>,
    b: SVector<f64, D>,
    normal: SVector<f64, D>,
    distance: f64,
}

fn epa_2d<const D: usize>(
    shape_a: &dyn Shape<D>,
    pos_a: &SVector<f64, D>,
    shape_b: &dyn Shape<D>,
    pos_b: &SVector<f64, D>,
    simplex: &[SVector<f64, D>],
) -> Option<EpaResult<D>> {
    // Build initial polygon from simplex (must have 2-3 points)
    let mut polytope: Vec<SVector<f64, D>> = simplex.to_vec();

    // Ensure winding order (counter-clockwise)
    if polytope.len() >= 3 {
        ensure_ccw_2d(&mut polytope);
    } else if polytope.len() < 2 {
        return None;
    }

    for _ in 0..MAX_EPA_ITERATIONS {
        // Find the closest edge to the origin
        let (edge_idx, closest) = find_closest_edge_2d(&polytope)?;

        // Get support point in the edge normal direction
        let support = mink_support(shape_a, pos_a, shape_b, pos_b, &closest.normal);
        let new_dist = support.dot(&closest.normal);

        // If the new point doesn't extend the polytope significantly, we're done
        if (new_dist - closest.distance).abs() < EPA_TOLERANCE {
            return Some(EpaResult {
                normal: closest.normal,
                depth: closest.distance,
            });
        }

        // Insert the new point between the edge's endpoints
        polytope.insert(edge_idx + 1, support);
    }

    // Max iterations — return best estimate
    let (_, closest) = find_closest_edge_2d(&polytope)?;
    Some(EpaResult {
        normal: closest.normal,
        depth: closest.distance,
    })
}

fn find_closest_edge_2d<const D: usize>(
    polytope: &[SVector<f64, D>],
) -> Option<(usize, Edge2D<D>)> {
    let n = polytope.len();
    if n < 2 {
        return None;
    }

    let mut best_idx = 0;
    let mut best_dist = f64::MAX;
    let mut best_normal: SVector<f64, D> = SVector::zeros();
    let mut best_a = polytope[0];
    let mut best_b = polytope[0];

    for i in 0..n {
        let a = polytope[i];
        let b = polytope[(i + 1) % n];
        let edge = b - a;

        // 2D outward normal: perpendicular to edge, pointing away from origin
        let mut normal: SVector<f64, D> = SVector::zeros();
        if D >= 2 {
            normal[0] = edge[1];
            normal[1] = -edge[0];
        }
        let len = normal.norm();
        if len < 1e-15 {
            continue;
        }
        normal /= len;

        // Ensure normal points outward (away from origin)
        if normal.dot(&a) < 0.0 {
            normal = -normal;
        }

        let dist = normal.dot(&a);
        if dist < best_dist {
            best_dist = dist;
            best_normal = normal;
            best_idx = i;
            best_a = a;
            best_b = b;
        }
    }

    Some((
        best_idx,
        Edge2D {
            a: best_a,
            b: best_b,
            normal: best_normal,
            distance: best_dist,
        },
    ))
}

fn ensure_ccw_2d<const D: usize>(polytope: &mut Vec<SVector<f64, D>>) {
    if polytope.len() < 3 || D < 2 {
        return;
    }
    // Cross product of first two edges
    let a = polytope[0];
    let b = polytope[1];
    let c = polytope[2];
    let ab = b - a;
    let ac = c - a;
    let cross = ab[0] * ac[1] - ab[1] * ac[0];
    if cross < 0.0 {
        polytope.reverse();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3D EPA: Face-based polytope expansion
// ═══════════════════════════════════════════════════════════════════════════

struct Face3D<const D: usize> {
    indices: [usize; 3],
    normal: SVector<f64, D>,
    distance: f64,
}

fn epa_3d<const D: usize>(
    shape_a: &dyn Shape<D>,
    pos_a: &SVector<f64, D>,
    shape_b: &dyn Shape<D>,
    pos_b: &SVector<f64, D>,
    simplex: &[SVector<f64, D>],
) -> Option<EpaResult<D>> {
    if simplex.len() < 4 || D < 3 {
        // Not a full tetrahedron — fall back to bounding sphere approx
        let (_, ra) = shape_a.bounding_sphere();
        let (_, rb) = shape_b.bounding_sphere();
        let delta = pos_b - pos_a;
        let dist = delta.norm();
        if dist < 1e-15 {
            let mut n = SVector::zeros();
            n[0] = 1.0;
            return Some(EpaResult {
                normal: n,
                depth: ra + rb,
            });
        }
        return Some(EpaResult {
            normal: delta / dist,
            depth: (ra + rb - dist).max(0.0),
        });
    }

    let mut vertices: Vec<SVector<f64, D>> = simplex.to_vec();

    // Build initial tetrahedron faces (4 triangles)
    let mut faces: Vec<Face3D<D>> = Vec::new();
    let face_indices = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
    for indices in &face_indices {
        if let Some(face) = make_face_3d(&vertices, *indices) {
            faces.push(face);
        }
    }

    if faces.is_empty() {
        return None;
    }

    for _ in 0..MAX_EPA_ITERATIONS {
        // Find closest face to origin
        let (face_idx, _) = faces
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.distance.partial_cmp(&b.distance).unwrap())?;

        let closest_normal = faces[face_idx].normal;
        let closest_dist = faces[face_idx].distance;

        // Get support point in face normal direction
        let support = mink_support(shape_a, pos_a, shape_b, pos_b, &closest_normal);
        let new_dist = support.dot(&closest_normal);

        if (new_dist - closest_dist).abs() < EPA_TOLERANCE {
            return Some(EpaResult {
                normal: closest_normal,
                depth: closest_dist,
            });
        }

        // Add new vertex
        let new_idx = vertices.len();
        vertices.push(support);

        // Remove faces visible from the new point and collect horizon edges
        let mut horizon: Vec<[usize; 2]> = Vec::new();
        let mut i = 0;
        while i < faces.len() {
            let face = &faces[i];
            let to_point = support - vertices[face.indices[0]];
            if face.normal.dot(&to_point) > 0.0 {
                // Face is visible — collect its edges for the horizon
                let fi = face.indices;
                add_horizon_edge(&mut horizon, [fi[0], fi[1]]);
                add_horizon_edge(&mut horizon, [fi[1], fi[2]]);
                add_horizon_edge(&mut horizon, [fi[2], fi[0]]);
                faces.swap_remove(i);
            } else {
                i += 1;
            }
        }

        // Create new faces from horizon edges to the new point
        for edge in &horizon {
            if let Some(face) = make_face_3d(&vertices, [edge[0], edge[1], new_idx]) {
                faces.push(face);
            }
        }

        if faces.is_empty() {
            return Some(EpaResult {
                normal: closest_normal,
                depth: closest_dist,
            });
        }
    }

    // Return best found
    let (_, best) = faces
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.distance.partial_cmp(&b.distance).unwrap())?;

    Some(EpaResult {
        normal: best.normal,
        depth: best.distance,
    })
}

fn make_face_3d<const D: usize>(
    vertices: &[SVector<f64, D>],
    indices: [usize; 3],
) -> Option<Face3D<D>> {
    if D < 3 {
        return None;
    }
    let a = vertices[indices[0]];
    let b = vertices[indices[1]];
    let c = vertices[indices[2]];
    let ab = b - a;
    let ac = c - a;

    // Cross product for 3D
    let mut normal: SVector<f64, D> = SVector::zeros();
    if D >= 3 {
        normal[0] = ab[1] * ac[2] - ab[2] * ac[1];
        normal[1] = ab[2] * ac[0] - ab[0] * ac[2];
        normal[2] = ab[0] * ac[1] - ab[1] * ac[0];
    }

    let len = normal.norm();
    if len < 1e-15 {
        return None;
    }
    normal /= len;

    // Ensure normal points away from origin
    let dist = normal.dot(&a);
    if dist < 0.0 {
        normal = -normal;
    }
    let dist = dist.abs();

    Some(Face3D {
        indices,
        normal,
        distance: dist,
    })
}

/// Add edge to horizon, removing if already present (shared edge = interior).
fn add_horizon_edge(horizon: &mut Vec<[usize; 2]>, edge: [usize; 2]) {
    // Check if reverse edge exists (shared between two visible faces)
    let reverse = [edge[1], edge[0]];
    if let Some(pos) = horizon.iter().position(|e| *e == reverse) {
        horizon.swap_remove(pos); // Shared edge — remove (interior)
    } else {
        horizon.push(edge);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gjk;
    use symtropy_math::{ConvexHull, Point, Sphere};

    #[test]
    fn epa_2d_spheres() {
        let a = Sphere::<2>::new(Point::origin(), 1.0);
        let b = Sphere::<2>::new(Point::origin(), 1.0);
        let pa = SVector::from([0.0, 0.0]);
        let pb = SVector::from([1.0, 0.0]);

        let gjk_result = gjk::intersects(&a, &pa, &b, &pb);
        assert!(gjk_result.intersecting);

        let epa = penetration(&a, &pa, &b, &pb, &gjk_result.simplex).unwrap();
        // Two unit spheres 1.0 apart: penetration = 2*1.0 - 1.0 = 1.0
        assert!(
            (epa.depth - 1.0).abs() < 0.1,
            "depth = {}, expected ~1.0",
            epa.depth
        );
        // Normal should point roughly along x axis
        assert!(epa.normal[0].abs() > 0.5, "normal x = {}", epa.normal[0]);
    }

    #[test]
    fn epa_2d_boxes() {
        let a = ConvexHull::<2>::unit_cube();
        let b = ConvexHull::<2>::unit_cube();
        let pa = SVector::from([0.0, 0.0]);
        let pb = SVector::from([1.0, 0.0]);

        let gjk_result = gjk::intersects(&a, &pa, &b, &pb);
        assert!(gjk_result.intersecting);

        let epa = penetration(&a, &pa, &b, &pb, &gjk_result.simplex).unwrap();
        // Two unit cubes (half-extent 1) at distance 1: overlap = 2*1 - 1 = 1
        assert!(
            (epa.depth - 1.0).abs() < 0.2,
            "depth = {}, expected ~1.0",
            epa.depth
        );
    }

    #[test]
    fn epa_3d_spheres() {
        let a = Sphere::<3>::new(Point::origin(), 1.0);
        let b = Sphere::<3>::new(Point::origin(), 1.0);
        let pa = SVector::from([0.0, 0.0, 0.0]);
        let pb = SVector::from([0.5, 0.0, 0.0]);

        let gjk_result = gjk::intersects(&a, &pa, &b, &pb);
        assert!(gjk_result.intersecting);

        let epa = penetration(&a, &pa, &b, &pb, &gjk_result.simplex).unwrap();
        // Penetration = 2*1.0 - 0.5 = 1.5
        assert!(
            (epa.depth - 1.5).abs() < 0.2,
            "depth = {}, expected ~1.5",
            epa.depth
        );
    }

    #[test]
    fn epa_3d_sphere_vs_box() {
        let sphere = Sphere::<3>::new(Point::origin(), 1.0);
        let cube = ConvexHull::<3>::unit_cube();
        let ps = SVector::from([0.0, 0.0, 0.0]);
        let pc = SVector::from([1.5, 0.0, 0.0]);

        let gjk_result = gjk::intersects(&sphere, &ps, &cube, &pc);
        assert!(gjk_result.intersecting);

        let epa = penetration(&sphere, &ps, &cube, &pc, &gjk_result.simplex).unwrap();
        // sphere(r=1) at 0 + cube(half=1) at 1.5: overlap = 1+1 - 1.5 = 0.5
        assert!(epa.depth > 0.0, "depth should be positive");
        assert!(epa.depth < 1.5, "depth {} too large", epa.depth);
    }

    #[test]
    fn epa_normal_direction() {
        let a = Sphere::<2>::new(Point::origin(), 1.0);
        let b = Sphere::<2>::new(Point::origin(), 1.0);
        let pa = SVector::from([0.0, 0.0]);
        let pb = SVector::from([0.5, 0.0]); // B is to the right

        let gjk_result = gjk::intersects(&a, &pa, &b, &pb);
        let epa = penetration(&a, &pa, &b, &pb, &gjk_result.simplex).unwrap();
        // Normal should point from A to B (positive x)
        assert!(
            epa.normal[0] > 0.0,
            "normal should point from A to B, got {:?}",
            epa.normal
        );
    }

    #[test]
    fn epa_4d_fallback() {
        let a = Sphere::<4>::new(Point::origin(), 1.0);
        let b = Sphere::<4>::new(Point::origin(), 1.0);
        let pa = SVector::from([0.0, 0.0, 0.0, 0.0]);
        let pb = SVector::from([0.5, 0.0, 0.0, 0.0]);

        let gjk_result = gjk::intersects(&a, &pa, &b, &pb);
        let epa = penetration(&a, &pa, &b, &pb, &gjk_result.simplex).unwrap();
        assert!(epa.depth > 0.0);
    }

    #[test]
    fn epa_depth_positive() {
        // Property: EPA depth is always non-negative for intersecting shapes
        let offsets = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5];
        for &offset in &offsets {
            let a = Sphere::<3>::unit();
            let b = Sphere::<3>::unit();
            let pa = SVector::zeros();
            let pb = SVector::from([offset, 0.0, 0.0]);

            let gjk_result = gjk::intersects(&a, &pa, &b, &pb);
            if gjk_result.intersecting {
                let epa = penetration(&a, &pa, &b, &pb, &gjk_result.simplex).unwrap();
                assert!(
                    epa.depth >= 0.0,
                    "negative depth {} at offset {}",
                    epa.depth,
                    offset
                );
            }
        }
    }
}
