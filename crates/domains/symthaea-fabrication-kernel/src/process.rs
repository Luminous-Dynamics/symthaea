// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Process-preparation authority between closed geometry and slicing.
//!
//! A topologically valid solid is not automatically positioned or supported for
//! a manufacturing process. This module grants a narrower capability only after
//! build-plate placement, component policy, and bounded support planning pass.

use crate::mesh::TriangleMesh;
use crate::support::{SupportConfig, SupportPlan, plan_column_supports};
use crate::validate::{FabricationReadyMesh, ValidationReport};

/// Process-level placement and support policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FabricationProcessPolicy {
    pub build_plate_z_mm: f32,
    pub placement_tolerance_mm: f32,
    pub require_single_component: bool,
    pub allow_sacrificial_supports: bool,
    pub support: SupportConfig,
}

impl Default for FabricationProcessPolicy {
    fn default() -> Self {
        let support = SupportConfig::default();
        Self {
            build_plate_z_mm: support.build_plate_z_mm,
            placement_tolerance_mm: 0.05,
            require_single_component: true,
            allow_sacrificial_supports: true,
            support,
        }
    }
}

/// Process-preparation violations that prevent trusted slicing.
#[derive(Debug, Clone, PartialEq)]
pub enum ProcessViolation {
    NonFinitePolicy(&'static str),
    GeometryBelowBuildPlate { vertex: usize, z_mm: f32 },
    MultipleComponents { count: usize },
    SupportRequiredButDisabled,
    SupportPlanTruncated,
}

/// Evidence collected while preparing a validated solid for a process.
#[derive(Debug, Clone, PartialEq)]
pub struct ProcessPreparationReport {
    pub minimum_z_mm: f32,
    pub maximum_z_mm: f32,
    pub support_plan: SupportPlan,
    pub violations: Vec<ProcessViolation>,
}

impl ProcessPreparationReport {
    pub fn is_ready(&self) -> bool {
        self.violations.is_empty()
    }
}

/// Error returned before process authority can be granted.
#[derive(Debug, Clone)]
pub enum ProcessPreparationError {
    Geometry(ValidationReport),
    Process(ProcessPreparationReport),
}

/// Closed-solid mesh that is positioned and support-planned for slicing.
#[derive(Debug, Clone)]
pub struct ProcessPreparedMesh {
    geometry: FabricationReadyMesh,
    policy: FabricationProcessPolicy,
    report: ProcessPreparationReport,
}

impl ProcessPreparedMesh {
    pub fn try_new(
        mesh: TriangleMesh,
        policy: FabricationProcessPolicy,
    ) -> Result<Self, ProcessPreparationError> {
        let geometry =
            FabricationReadyMesh::try_new(mesh).map_err(ProcessPreparationError::Geometry)?;
        Self::from_fabrication_ready(geometry, policy)
    }

    pub fn from_fabrication_ready(
        geometry: FabricationReadyMesh,
        mut policy: FabricationProcessPolicy,
    ) -> Result<Self, ProcessPreparationError> {
        let mut violations = Vec::new();
        for (name, value) in [
            ("build_plate_z_mm", policy.build_plate_z_mm),
            ("placement_tolerance_mm", policy.placement_tolerance_mm),
        ] {
            if !value.is_finite() {
                violations.push(ProcessViolation::NonFinitePolicy(name));
            }
        }
        if !policy.placement_tolerance_mm.is_finite() {
            policy.placement_tolerance_mm = 0.0;
        }
        policy.support.build_plate_z_mm = policy.build_plate_z_mm;

        let mesh = geometry.mesh();
        let mut minimum_z_mm = f32::INFINITY;
        let mut maximum_z_mm = f32::NEG_INFINITY;
        for (vertex_index, vertex) in mesh.vertices.iter().enumerate() {
            minimum_z_mm = minimum_z_mm.min(vertex[2]);
            maximum_z_mm = maximum_z_mm.max(vertex[2]);
            if vertex[2] < policy.build_plate_z_mm - policy.placement_tolerance_mm.max(0.0) {
                violations.push(ProcessViolation::GeometryBelowBuildPlate {
                    vertex: vertex_index,
                    z_mm: vertex[2],
                });
            }
        }

        if policy.require_single_component && geometry.report().connected_components != 1 {
            violations.push(ProcessViolation::MultipleComponents {
                count: geometry.report().connected_components,
            });
        }

        let support_plan = plan_column_supports(mesh, policy.support);
        if support_plan.requires_support() && !policy.allow_sacrificial_supports {
            violations.push(ProcessViolation::SupportRequiredButDisabled);
        }
        if support_plan.truncated {
            violations.push(ProcessViolation::SupportPlanTruncated);
        }

        let report = ProcessPreparationReport {
            minimum_z_mm,
            maximum_z_mm,
            support_plan,
            violations,
        };
        if !report.is_ready() {
            return Err(ProcessPreparationError::Process(report));
        }
        Ok(Self {
            geometry,
            policy,
            report,
        })
    }

    pub fn mesh(&self) -> &TriangleMesh {
        self.geometry.mesh()
    }

    pub fn geometry(&self) -> &FabricationReadyMesh {
        &self.geometry
    }

    pub fn policy(&self) -> &FabricationProcessPolicy {
        &self.policy
    }

    pub fn report(&self) -> &ProcessPreparationReport {
        &self.report
    }

    pub fn into_geometry(self) -> FabricationReadyMesh {
        self.geometry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csg::{CSGNode, Transform3D};
    use crate::mesh::resolve_to_mesh;

    #[test]
    fn cube_on_plate_gains_process_authority_without_support() {
        let mesh = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
            translate: [0.0, 0.0, 0.5],
            ..Default::default()
        }));
        let prepared = ProcessPreparedMesh::try_new(mesh, FabricationProcessPolicy::default())
            .expect("plate-aligned cube should be process-ready");
        assert!(!prepared.report().support_plan.requires_support());
        assert!(prepared.report().is_ready());
    }

    #[test]
    fn floating_cube_requires_and_receives_support_plan() {
        let mesh = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
            translate: [0.0, 0.0, 2.0],
            ..Default::default()
        }));
        let prepared = ProcessPreparedMesh::try_new(mesh, FabricationProcessPolicy::default())
            .expect("default policy permits complete sacrificial support plans");
        assert!(prepared.report().support_plan.requires_support());
        assert!(!prepared.report().support_plan.columns.is_empty());
    }

    #[test]
    fn support_requirement_fails_when_supports_are_disabled() {
        let mesh = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
            translate: [0.0, 0.0, 2.0],
            ..Default::default()
        }));
        let result = ProcessPreparedMesh::try_new(
            mesh,
            FabricationProcessPolicy {
                allow_sacrificial_supports: false,
                ..FabricationProcessPolicy::default()
            },
        );
        assert!(matches!(
            result,
            Err(ProcessPreparationError::Process(report))
                if report.violations.contains(&ProcessViolation::SupportRequiredButDisabled)
        ));
    }

    #[test]
    fn geometry_below_plate_fails_closed() {
        let mesh = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
            translate: [0.0, 0.0, 0.0],
            ..Default::default()
        }));
        let result = ProcessPreparedMesh::try_new(mesh, FabricationProcessPolicy::default());
        assert!(matches!(
            result,
            Err(ProcessPreparationError::Process(report))
                if report.violations.iter().any(|violation| matches!(violation, ProcessViolation::GeometryBelowBuildPlate { .. }))
        ));
    }

    #[test]
    fn disconnected_shells_fail_single_component_policy() {
        let mut left = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
            translate: [0.0, 0.0, 0.5],
            ..Default::default()
        }));
        let right = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
            translate: [2.0, 0.0, 0.5],
            ..Default::default()
        }));
        left.merge(&right);
        let result = ProcessPreparedMesh::try_new(left, FabricationProcessPolicy::default());
        assert!(matches!(
            result,
            Err(ProcessPreparationError::Process(report))
                if report.violations.contains(&ProcessViolation::MultipleComponents { count: 2 })
        ));
    }
}
