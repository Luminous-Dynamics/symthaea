// Copyright (C) 2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Multi-Physics Virtual Geometry (MPVG) prototype layout and clustering.
//!
//! Extends Bevy's virtual geometry/meshlet layout to store thermodynamic,
//! acoustic, and structural stress parameters directly inside vertex buffer strides,
//! enabling unified, high-fidelity visual and physical simulation loops.

use nalgebra::SVector;
use serde::{Deserialize, Serialize};

/// Custom vertex structure storing visual attributes alongside physical parameters.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[repr(C)]
pub struct MultiPhysicsVertex {
    // --- Visual Attributes (12 + 12 + 8 = 32 bytes) ---
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],

    // --- Physical Substrate Attributes (16 bytes) ---
    /// Mass density of the material at this vertex point (kg/m^3).
    pub mass_density: f32,
    /// Young's modulus / elasticity representing structural stiffness (Pa).
    pub elastic_modulus: f32,
    /// Heat conductivity coefficient (W/(m*K)).
    pub thermal_conductivity: f32,
    /// Acoustic impedance matching coefficient (Pa*s/m).
    pub acoustic_impedance: f32,
}

/// A cluster of vertices representing a single virtual geometry meshlet.
/// Designed to map directly to hardware-accelerated bounding volumes or GPU mesh shaders.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiPhysicsMeshlet {
    /// Range of vertices in the parent mesh index buffer.
    pub vertex_offset: u32,
    pub vertex_count: u32,
    /// Range of local triangle indices.
    pub index_offset: u32,
    pub index_count: u32,

    // --- Aggregated Physics Bounds (Cached for fast query and island integration) ---
    /// Center of mass for this specific cluster.
    pub center_of_mass: SVector<f32, 3>,
    /// Total mass computed from vertex volumes/densities.
    pub total_mass: f32,
    /// Average elastic stiffness tensor representation for fast stress analysis.
    pub average_elasticity: f32,
}

/// A compiled, virtualized multi-physics geometry mesh ready for zero-copy streaming.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiPhysicsMeshletMesh {
    /// Flat array of all multi-physics vertices.
    pub vertices: Vec<MultiPhysicsVertex>,
    /// Flat array of 8-bit or 32-bit local indices mapping to triangles.
    pub indices: Vec<u8>,
    /// List of cooked meshlet clusters.
    pub meshlets: Vec<MultiPhysicsMeshlet>,
}

impl MultiPhysicsMeshletMesh {
    /// Constructs a basic test mesh containing a single cluster from raw data.
    pub fn build_prototype(vertices: Vec<MultiPhysicsVertex>, indices: Vec<u8>) -> Self {
        let vertex_count = vertices.len() as u32;
        let index_count = indices.len() as u32;

        // Compute aggregated physical attributes for the prototype cluster
        let mut mass_sum = 0.0;
        let mut com_accumulator = SVector::<f32, 3>::zeros();
        let mut elasticity_sum = 0.0;

        for v in &vertices {
            // Assume equal volume allocation per vertex for centroid mass weighting
            let weight = v.mass_density.max(0.001);
            mass_sum += weight;
            com_accumulator += SVector::from(v.position) * weight;
            elasticity_sum += v.elastic_modulus;
        }

        let center_of_mass = if mass_sum > 0.0 {
            com_accumulator / mass_sum
        } else {
            SVector::zeros()
        };

        let average_elasticity = if vertex_count > 0 {
            elasticity_sum / vertex_count as f32
        } else {
            0.0
        };

        let prototype_meshlet = MultiPhysicsMeshlet {
            vertex_offset: 0,
            vertex_count,
            index_offset: 0,
            index_count,
            center_of_mass,
            total_mass: mass_sum,
            average_elasticity,
        };

        Self {
            vertices,
            indices,
            meshlets: vec![prototype_meshlet],
        }
    }

    /// Helper to cook/serialize the virtual geometry mesh to a raw byte stream
    /// suitable for deterministic sandboxed Nix outputs or zero-copy VRAM uploads.
    pub fn cook_binary(&self) -> Result<Vec<u8>, bincode::Error> {
        bincode::serialize(self)
    }

    /// Deserializes a cooked multi-physics mesh stream.
    pub fn load_binary(bytes: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prototype_cooking_and_aggregation() {
        let v1 = MultiPhysicsVertex {
            position: [0.0, 0.0, 0.0],
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 0.0],
            mass_density: 7800.0,   // Steel density
            elastic_modulus: 200e9, // Steel Young's Modulus
            thermal_conductivity: 50.0,
            acoustic_impedance: 46.0e6,
        };

        let v2 = MultiPhysicsVertex {
            position: [1.0, 0.0, 0.0],
            normal: [0.0, 0.0, 1.0],
            uv: [1.0, 0.0],
            mass_density: 2700.0,  // Aluminum density
            elastic_modulus: 70e9, // Aluminum Young's Modulus
            thermal_conductivity: 200.0,
            acoustic_impedance: 17.0e6,
        };

        let vertices = vec![v1, v2];
        let indices = vec![0, 1];

        // Build prototype multi-physics mesh
        let mesh = MultiPhysicsMeshletMesh::build_prototype(vertices, indices);

        assert_eq!(mesh.meshlets.len(), 1);
        let meshlet = &mesh.meshlets[0];

        // Check center of mass calculation (weighted toward heavier steel vertex at 0.0)
        assert!(
            meshlet.center_of_mass[0] < 0.5,
            "Center of mass must be weighted toward steel"
        );
        assert_eq!(meshlet.average_elasticity, 135e9); // (200e9 + 70e9) / 2

        // Cook to binary stream
        let cooked_bytes = mesh.cook_binary().expect("Failed to serialize");
        assert!(
            !cooked_bytes.is_empty(),
            "Cooked byte stream should not be empty"
        );

        // Load binary back
        let loaded_mesh =
            MultiPhysicsMeshletMesh::load_binary(&cooked_bytes).expect("Failed to deserialize");
        assert_eq!(loaded_mesh.vertices.len(), 2);
        assert_eq!(loaded_mesh.vertices[0].mass_density, 7800.0);
    }
}
