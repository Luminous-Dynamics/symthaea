// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Capability-bearing manufacturability qualification.
//!
//! [`ManufacturingReadyMesh`] proves that a mesh crossed closed-solid
//! validation, process preparation, and bounded minimum-feature screening under
//! retained policies. It still does not certify a material/process pair or real
//! hardware operation.

use crate::manufacturability::{
    MinimumFeatureError, MinimumFeaturePolicy, MinimumFeatureReport, analyze_minimum_features,
};
use crate::mesh::TriangleMesh;
use crate::process::{FabricationProcessPolicy, ProcessPreparationError, ProcessPreparedMesh};

/// Failure before manufacturability authority can be granted.
#[derive(Debug, Clone)]
pub enum ManufacturingQualificationError {
    Process(ProcessPreparationError),
    Analysis(MinimumFeatureError),
    MinimumFeature(MinimumFeatureReport),
}

/// Process-prepared mesh that passed one retained minimum-feature policy.
#[derive(Debug, Clone)]
pub struct ManufacturingReadyMesh {
    process: ProcessPreparedMesh,
    minimum_feature_policy: MinimumFeaturePolicy,
    minimum_feature_report: MinimumFeatureReport,
}

impl ManufacturingReadyMesh {
    pub fn try_new(
        mesh: TriangleMesh,
        process_policy: FabricationProcessPolicy,
        minimum_feature_policy: MinimumFeaturePolicy,
    ) -> Result<Self, ManufacturingQualificationError> {
        let process = ProcessPreparedMesh::try_new(mesh, process_policy)
            .map_err(ManufacturingQualificationError::Process)?;
        Self::from_process_prepared(process, minimum_feature_policy)
    }

    pub fn from_process_prepared(
        process: ProcessPreparedMesh,
        minimum_feature_policy: MinimumFeaturePolicy,
    ) -> Result<Self, ManufacturingQualificationError> {
        let minimum_feature_report =
            analyze_minimum_features(process.mesh(), minimum_feature_policy)
                .map_err(ManufacturingQualificationError::Analysis)?;
        if !minimum_feature_report.passes() {
            return Err(ManufacturingQualificationError::MinimumFeature(
                minimum_feature_report,
            ));
        }
        Ok(Self {
            process,
            minimum_feature_policy,
            minimum_feature_report,
        })
    }

    pub fn mesh(&self) -> &TriangleMesh {
        self.process.mesh()
    }

    pub fn process(&self) -> &ProcessPreparedMesh {
        &self.process
    }

    pub fn minimum_feature_policy(&self) -> &MinimumFeaturePolicy {
        &self.minimum_feature_policy
    }

    pub fn minimum_feature_report(&self) -> &MinimumFeatureReport {
        &self.minimum_feature_report
    }

    pub fn into_process_prepared(self) -> ProcessPreparedMesh {
        self.process
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csg::{CSGNode, Transform3D};
    use crate::mesh::resolve_to_mesh;

    #[test]
    fn plate_aligned_cube_gains_manufacturing_authority() {
        let mesh = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
            translate: [0.0, 0.0, 0.5],
            ..Default::default()
        }));
        let ready = ManufacturingReadyMesh::try_new(
            mesh,
            FabricationProcessPolicy::default(),
            MinimumFeaturePolicy::default(),
        )
        .unwrap();
        assert!(ready.minimum_feature_report().passes());
        assert_eq!(
            ready.process().policy(),
            &FabricationProcessPolicy::default()
        );
    }

    #[test]
    fn thin_part_cannot_gain_manufacturing_authority() {
        let mesh = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
            scale: [1.0, 1.0, 0.2],
            translate: [0.0, 0.0, 0.1],
            ..Default::default()
        }));
        let result = ManufacturingReadyMesh::try_new(
            mesh,
            FabricationProcessPolicy::default(),
            MinimumFeaturePolicy::default(),
        );
        assert!(matches!(
            result,
            Err(ManufacturingQualificationError::MinimumFeature(report))
                if !report.thin_source_triangles.is_empty()
        ));
    }
}
