// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Generative design module — multi-objective fitness evaluation
//!
//! Connects CSG tree generation to real structural analysis via the
//! [`AnalyticalBackend`] and mesh validation pipeline, enabling
//! evolutionary / generative design loops with physics-grounded fitness.

use crate::analytical::{AnalyticalBackend, CrossSection, MaterialProperties};
use crate::csg::CSGNode;
use crate::mesh::{TriangleMesh, resolve_to_mesh};
use crate::validate::validate_mesh;

/// Default transverse load (Newtons) for structural fitness evaluation.
///
/// 1 kN represents a moderate load suitable for small structural parts
/// (brackets, mounts, hinges). Callers needing different loads should
/// use [`structural_fitness_with_load`] instead.
const DEFAULT_TRANSVERSE_LOAD: f64 = 1000.0;

/// Default beam length (meters) derived from mesh bounding box extent.
///
/// Falls back to 0.1 m if the mesh has no vertices.
fn beam_length_from_mesh(mesh: &TriangleMesh) -> f64 {
    if mesh.vertices.is_empty() {
        return 0.1;
    }
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut min_z = f32::INFINITY;
    let mut max_z = f32::NEG_INFINITY;
    for v in &mesh.vertices {
        min_x = min_x.min(v[0]);
        max_x = max_x.max(v[0]);
        min_y = min_y.min(v[1]);
        max_y = max_y.max(v[1]);
        min_z = min_z.min(v[2]);
        max_z = max_z.max(v[2]);
    }
    let dx = (max_x - min_x) as f64;
    let dy = (max_y - min_y) as f64;
    let dz = (max_z - min_z) as f64;
    // Use the longest axis as the effective beam length
    let longest = dx.max(dy).max(dz);
    if longest < 1e-10 { 0.1 } else { longest }
}

/// Cross-section approximation from mesh bounding box.
///
/// Uses the two shorter bounding-box dimensions as a rectangular cross-section,
/// which is a conservative approximation for arbitrary geometry.
fn cross_section_from_mesh(mesh: &TriangleMesh) -> CrossSection {
    if mesh.vertices.is_empty() {
        return CrossSection::Rectangle {
            width: 0.01,
            height: 0.01,
        };
    }
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for v in &mesh.vertices {
        for i in 0..3 {
            min[i] = min[i].min(v[i]);
            max[i] = max[i].max(v[i]);
        }
    }
    let mut extents = [
        (max[0] - min[0]) as f64,
        (max[1] - min[1]) as f64,
        (max[2] - min[2]) as f64,
    ];
    extents.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // The two shorter dimensions form the cross-section
    let width = extents[0].max(0.001);
    let height = extents[1].max(0.001);
    CrossSection::Rectangle { width, height }
}

/// Built-in multi-objective fitness function using FEA and mesh validation.
///
/// Evaluates a CSG tree across three objectives:
/// 1. **Structural strength**: `safety_factor` from [`AnalyticalBackend`] (higher = better)
/// 2. **Manufacturability**: watertight + no degenerates from [`validate_mesh`]
/// 3. **Material efficiency**: penalizes excessive geometry relative to `target_triangle_count`
///
/// Returns a weighted score in `[0.0, 1.0]`:
/// `0.4 * structural + 0.4 * manufacturability + 0.2 * efficiency`
///
/// Uses steel material with a 1 kN transverse load and bounding-box-derived
/// beam geometry. For custom load parameters, see [`structural_fitness_with_load`].
pub fn structural_fitness(node: &CSGNode, target_triangle_count: usize) -> f64 {
    structural_fitness_with_load(node, target_triangle_count, DEFAULT_TRANSVERSE_LOAD)
}

/// Like [`structural_fitness`] but with a configurable transverse load (Newtons).
pub fn structural_fitness_with_load(
    node: &CSGNode,
    target_triangle_count: usize,
    transverse_force: f64,
) -> f64 {
    // 1. Resolve CSG tree to triangle mesh
    let mesh = resolve_to_mesh(node);

    // Handle degenerate case: empty mesh gets zero fitness
    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
        return 0.0;
    }

    // 2. Validate the mesh
    let report = validate_mesh(&mesh);

    // 3. Structural score via AnalyticalBackend
    let beam_length = beam_length_from_mesh(&mesh);
    let cross_section = cross_section_from_mesh(&mesh);
    let material = MaterialProperties::steel();
    let backend = AnalyticalBackend::new(material, cross_section, beam_length);
    let result = backend.analyze(0.0, transverse_force);

    // Normalize safety_factor to [0, 1]: sf=1 maps to 0.5, sf>=10 maps to ~1.0
    // Uses 1 - 1/(1 + sf/5) so the curve rises steeply then saturates.
    let structural_score = if result.safety_factor.is_finite() && result.safety_factor > 0.0 {
        1.0 - 1.0 / (1.0 + result.safety_factor / 5.0)
    } else {
        0.0
    };

    // 4. Manufacturability score
    let manufacturability_score = if report.is_watertight && report.degenerate_triangles.is_empty()
    {
        1.0
    } else if report.is_watertight {
        0.5
    } else {
        0.0
    };

    // 5. Material efficiency: penalize excessive geometry
    let target = if target_triangle_count == 0 {
        1
    } else {
        target_triangle_count
    };
    let efficiency_score = 1.0 - (report.triangle_count as f64 / (target as f64 * 2.0)).min(1.0);

    // 6. Weighted sum
    0.4 * structural_score + 0.4 * manufacturability_score + 0.2 * efficiency_score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_has_nonzero_fitness() {
        let fitness = structural_fitness(&CSGNode::cube(), 12);
        assert!(fitness > 0.0, "cube fitness should be > 0, got {}", fitness);
        assert!(fitness <= 1.0, "fitness should be <= 1.0, got {}", fitness);
    }

    #[test]
    fn sphere_has_nonzero_fitness() {
        let fitness = structural_fitness(&CSGNode::sphere(), 200);
        assert!(
            fitness > 0.0,
            "sphere fitness should be > 0, got {}",
            fitness
        );
    }

    #[test]
    fn cylinder_has_nonzero_fitness() {
        let fitness = structural_fitness(&CSGNode::cylinder(), 100);
        assert!(
            fitness > 0.0,
            "cylinder fitness should be > 0, got {}",
            fitness
        );
    }

    #[test]
    fn cone_does_not_panic() {
        let fitness = structural_fitness(&CSGNode::cone(), 50);
        assert!(fitness.is_finite());
    }

    #[test]
    fn torus_does_not_panic() {
        let fitness = structural_fitness(&CSGNode::torus(), 500);
        assert!(fitness.is_finite());
    }

    #[test]
    fn boolean_union_does_not_panic() {
        let tree = CSGNode::cube().union(CSGNode::sphere());
        let fitness = structural_fitness(&tree, 300);
        assert!(fitness.is_finite());
        assert!(fitness > 0.0);
    }

    #[test]
    fn complex_union_tree_does_not_panic() {
        // Boolean subtract/intersect use recursive BSP which can overflow
        // the stack on complex meshes, so we test a multi-union tree instead.
        let tree = CSGNode::cube()
            .union(CSGNode::cylinder())
            .union(CSGNode::cone());
        let fitness = structural_fitness(&tree, 200);
        assert!(fitness.is_finite());
        assert!(fitness > 0.0);
    }

    #[test]
    fn transformed_node_does_not_panic() {
        use crate::csg::Transform3D;
        let tree = CSGNode::cube().with_transform(Transform3D {
            scale: [2.0, 0.5, 1.0],
            ..Default::default()
        });
        let fitness = structural_fitness(&tree, 12);
        assert!(fitness.is_finite());
        assert!(fitness > 0.0);
    }

    #[test]
    fn zero_target_triangle_count_does_not_panic() {
        let fitness = structural_fitness(&CSGNode::cube(), 0);
        assert!(fitness.is_finite());
    }

    #[test]
    fn large_target_gives_higher_efficiency() {
        // A cube has 12 triangles. target=12 means ratio=12/24=0.5, efficiency=0.5.
        // target=1000 means ratio=12/2000≈0.006, efficiency≈0.994.
        let f_small = structural_fitness(&CSGNode::cube(), 12);
        let f_large = structural_fitness(&CSGNode::cube(), 1000);
        // Efficiency is only 0.2 weight, so difference may be small but should exist
        assert!(
            f_large >= f_small,
            "larger target should give >= fitness: {} vs {}",
            f_large,
            f_small
        );
    }

    #[test]
    fn custom_load_works() {
        let low = structural_fitness_with_load(&CSGNode::cube(), 12, 100.0);
        let high = structural_fitness_with_load(&CSGNode::cube(), 12, 1_000_000.0);
        // Higher load => lower safety factor => lower structural score
        assert!(
            low >= high,
            "low load should give >= fitness than high load: {} vs {}",
            low,
            high
        );
    }

    #[test]
    fn cube_watertight_gets_manufacturability_bonus() {
        // Cube is watertight with no degenerates => manufacturability = 1.0
        let fitness = structural_fitness(&CSGNode::cube(), 12);
        // manufacturability contributes 0.4 * 1.0 = 0.4 minimum
        assert!(
            fitness >= 0.4,
            "watertight cube should have >= 0.4 fitness, got {}",
            fitness
        );
    }
}
