// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Native Bevy 0.20 adapter for the renderer-neutral Mycelial Presence model.
//!
//! This crate is intentionally excluded from the root workspace so that its
//! Bevy/wgpu dependency graph cannot force a migration of other renderers.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
use symthaea_canvas::{
    EmbodimentSnapshot, MycelialTubeMesh, PresenceMode, TubeTessellationConfig,
    TubeTessellationError, build_mycelial_model, tessellate_mycelial_model,
};

/// Build the canonical model and translate its bounded CPU tube mesh into a
/// Bevy mesh asset. No renderer-specific state is added to the model.
pub fn build_bevy_filament_mesh(
    snapshot: &EmbodimentSnapshot,
    mode: PresenceMode,
    config: TubeTessellationConfig,
) -> Result<Mesh, TubeTessellationError> {
    let model = build_mycelial_model(snapshot, mode);
    let mesh = tessellate_mycelial_model(&model, config)?;
    Ok(into_bevy_mesh(mesh))
}

/// Convert a validated renderer-neutral tube mesh into a Bevy 3D mesh.
pub fn into_bevy_mesh(source: MycelialTubeMesh) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, source.positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, source.normals);
    mesh.insert_indices(Indices::U32(source.indices));
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::mesh::VertexAttributeValues;

    #[test]
    fn converts_canonical_tube_mesh_to_bevy_mesh_assets() {
        let snapshot = EmbodimentSnapshot::default();
        let mesh = build_bevy_filament_mesh(
            &snapshot,
            PresenceMode::Ambient,
            TubeTessellationConfig::default(),
        )
        .expect("canonical model should tessellate");

        let positions = mesh.attribute(Mesh::ATTRIBUTE_POSITION);
        let normals = mesh.attribute(Mesh::ATTRIBUTE_NORMAL);
        assert!(matches!(positions, Some(VertexAttributeValues::Float32x3(values)) if !values.is_empty()));
        assert!(matches!(normals, Some(VertexAttributeValues::Float32x3(values)) if !values.is_empty()));
        assert!(mesh.indices().is_some());
        assert_eq!(mesh.primitive_topology(), PrimitiveTopology::TriangleList);
    }

    #[test]
    fn generated_mesh_contains_only_finite_vertices() {
        let mesh = build_bevy_filament_mesh(
            &EmbodimentSnapshot::default(),
            PresenceMode::Cognitive,
            TubeTessellationConfig::default(),
        )
        .expect("canonical model should tessellate");
        let Some(VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("positions are required");
        };
        assert!(positions.iter().flatten().all(|value| value.is_finite()));
    }
}
