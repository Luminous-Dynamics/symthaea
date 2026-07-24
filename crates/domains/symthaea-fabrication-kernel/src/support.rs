// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Local overhang analysis and bounded sacrificial support synthesis.
//!
//! This module provides geometry evidence and a conservative column-support
//! plan. It does not claim process-complete support optimization: interfaces,
//! branching trees, thermal distortion, trapped support, and removal access
//! remain machine/material-specific concerns.

use crate::csg::CSGNode;
use crate::mesh::TriangleMesh;
use std::collections::HashMap;

/// Conservative support-planning parameters in millimetres and degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SupportConfig {
    pub build_plate_z_mm: f32,
    /// Faces whose outward normal points downward beyond this angle from the
    /// vertical axis are support candidates. Typical FDM value: 45 degrees.
    pub max_overhang_from_vertical_degrees: f32,
    pub contact_tolerance_mm: f32,
    pub interface_gap_mm: f32,
    pub column_width_mm: f32,
    pub column_pitch_mm: f32,
    pub max_columns: usize,
}

impl Default for SupportConfig {
    fn default() -> Self {
        Self {
            build_plate_z_mm: 0.0,
            max_overhang_from_vertical_degrees: 45.0,
            contact_tolerance_mm: 0.05,
            interface_gap_mm: 0.2,
            column_width_mm: 0.8,
            column_pitch_mm: 2.0,
            max_columns: 256,
        }
    }
}

/// One vertical sacrificial support column.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SupportColumn {
    pub center_xy_mm: [f32; 2],
    pub bottom_z_mm: f32,
    pub top_z_mm: f32,
    pub width_mm: f32,
}

impl SupportColumn {
    pub fn height_mm(&self) -> f32 {
        (self.top_z_mm - self.bottom_z_mm).max(0.0)
    }

    pub fn to_csg(self) -> CSGNode {
        CSGNode::cube()
            .scale(
                self.width_mm as f64,
                self.width_mm as f64,
                self.height_mm() as f64,
            )
            .translate(
                self.center_xy_mm[0] as f64,
                self.center_xy_mm[1] as f64,
                ((self.bottom_z_mm + self.top_z_mm) * 0.5) as f64,
            )
    }
}

/// Evidence and synthesized geometry for local overhangs.
#[derive(Debug, Clone, PartialEq)]
pub struct SupportPlan {
    pub overhang_triangles: Vec<usize>,
    pub unsupported_surface_area_mm2: f32,
    pub columns: Vec<SupportColumn>,
    /// True when candidates exceeded the configured support-column budget.
    pub truncated: bool,
}

impl SupportPlan {
    pub fn requires_support(&self) -> bool {
        !self.overhang_triangles.is_empty()
    }

    pub fn is_complete(&self) -> bool {
        !self.truncated
    }

    /// Build a CSG union of all planned support columns.
    pub fn to_csg(&self) -> Option<CSGNode> {
        let mut columns = self.columns.iter().copied().map(SupportColumn::to_csg);
        let first = columns.next()?;
        Some(columns.fold(first, CSGNode::union))
    }
}

/// Analyze local downward faces and synthesize a bounded column-support plan.
pub fn plan_column_supports(mesh: &TriangleMesh, config: SupportConfig) -> SupportPlan {
    let angle = config
        .max_overhang_from_vertical_degrees
        .clamp(0.0, 89.9)
        .to_radians();
    let downward_threshold = -angle.cos();
    let pitch = config.column_pitch_mm.max(config.column_width_mm).max(0.01);
    let width = config.column_width_mm.max(0.01);
    let plate = config.build_plate_z_mm;

    let mut overhang_triangles = Vec::new();
    let mut unsupported_surface_area_mm2 = 0.0;
    let mut cells: HashMap<(i64, i64), SupportColumn> = HashMap::new();
    let mut truncated = false;

    for (triangle_index, triangle) in mesh.indices.iter().enumerate() {
        let indices = [
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        ];
        if indices.iter().any(|index| *index >= mesh.vertices.len()) {
            continue;
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
            continue;
        }

        let cross = cross(sub(points[1], points[0]), sub(points[2], points[0]));
        let doubled_area = norm(cross);
        if doubled_area <= f32::EPSILON {
            continue;
        }
        let normal_z = cross[2] / doubled_area;
        let lowest_z = points
            .iter()
            .map(|point| point[2])
            .fold(f32::INFINITY, f32::min);
        if normal_z >= downward_threshold
            || lowest_z <= plate + config.contact_tolerance_mm.max(0.0)
        {
            continue;
        }

        let top_z = lowest_z - config.interface_gap_mm.max(0.0);
        if top_z <= plate + f32::EPSILON {
            continue;
        }

        overhang_triangles.push(triangle_index);
        unsupported_surface_area_mm2 += doubled_area * 0.5;
        let centroid = [
            (points[0][0] + points[1][0] + points[2][0]) / 3.0,
            (points[0][1] + points[1][1] + points[2][1]) / 3.0,
        ];
        let key = (
            (centroid[0] / pitch).floor() as i64,
            (centroid[1] / pitch).floor() as i64,
        );
        if let Some(existing) = cells.get_mut(&key) {
            existing.top_z_mm = existing.top_z_mm.max(top_z);
            continue;
        }
        if cells.len() >= config.max_columns {
            truncated = true;
            continue;
        }
        cells.insert(
            key,
            SupportColumn {
                center_xy_mm: centroid,
                bottom_z_mm: plate,
                top_z_mm: top_z,
                width_mm: width,
            },
        );
    }

    let mut columns: Vec<_> = cells.into_values().collect();
    columns.sort_by(|left, right| {
        left.center_xy_mm[0]
            .partial_cmp(&right.center_xy_mm[0])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                left.center_xy_mm[1]
                    .partial_cmp(&right.center_xy_mm[1])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    SupportPlan {
        overhang_triangles,
        unsupported_surface_area_mm2,
        columns,
        truncated,
    }
}

fn sub(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn cross(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn norm(value: [f32; 3]) -> f32 {
    (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csg::{CSGNode, Transform3D};
    use crate::mesh::resolve_to_mesh;

    #[test]
    fn cube_on_build_plate_needs_no_support() {
        let mesh = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
            translate: [0.0, 0.0, 0.5],
            ..Default::default()
        }));
        let plan = plan_column_supports(&mesh, SupportConfig::default());
        assert!(!plan.requires_support());
        assert!(plan.columns.is_empty());
    }

    #[test]
    fn floating_cube_generates_local_columns() {
        let mesh = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
            translate: [0.0, 0.0, 2.0],
            ..Default::default()
        }));
        let plan = plan_column_supports(&mesh, SupportConfig::default());
        assert!(plan.requires_support());
        assert!(!plan.columns.is_empty());
        assert!(plan.unsupported_surface_area_mm2 > 0.9);
        assert!(plan.to_csg().is_some());
        assert!(plan.columns.iter().all(|column| column.height_mm() > 0.0));
    }

    #[test]
    fn vertical_walls_are_not_overhangs() {
        let mesh = TriangleMesh {
            vertices: vec![[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 2.0]],
            normals: vec![[1.0, 0.0, 0.0]; 3],
            indices: vec![[0, 1, 2]],
        };
        assert!(!plan_column_supports(&mesh, SupportConfig::default()).requires_support());
    }

    #[test]
    fn column_budget_reports_truncation() {
        let mesh = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
            translate: [0.0, 0.0, 2.0],
            ..Default::default()
        }));
        let plan = plan_column_supports(
            &mesh,
            SupportConfig {
                column_pitch_mm: 0.01,
                max_columns: 0,
                ..SupportConfig::default()
            },
        );
        assert!(plan.truncated);
        assert!(plan.columns.is_empty());
        assert!(plan.requires_support());
    }
}
