// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! DeSci Evidence Plugin — 3D visualization of the Earth Evidence Mesh.
//!
//! Renders EvidencePackets as holographic markers on the globe,
//! color-coded by Epistemic Domain (Sapphire/Crimson).

use crate::holographic_material::HolographicMaterial;
use bevy::prelude::*;
use mycelix_desci_core::meta::EmpiricalAxis;
use mycelix_earth::EvidencePacket;
use sol_atlas_core::geo;
use sol_atlas_core::types::*;

pub struct DeSciEvidencePlugin;

impl Plugin for DeSciEvidencePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, spawn_evidence_markers);
    }
}

/// Component for evidence markers on the globe.
#[derive(Component)]
pub struct EvidenceMarker {
    pub packet_id: uuid::Uuid,
}

/// Scan for new EvidencePackets and spawn holographic markers.
pub fn spawn_evidence_markers(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<HolographicMaterial>>,
    // In production, this would be a Resource containing a stream of packets
    new_packets: Query<(Entity, &EvidencePacket), Added<EvidencePacket>>,
) {
    let marker_mesh = meshes.add(Sphere::new(1.0).mesh().uv(16, 16));

    for (_entity, packet) in &new_packets {
        // Mock coordinates for v0 (should be extracted from packet.aoi_hash)
        let lat = 45.0;
        let lon = 15.0;
        let pos = geo::lat_lon_to_xyz(lat, lon, 1.02);

        // Determine Epistemic Color
        // Sapphire Blue for Empirical (E3+)
        // Crimson Red for Somatic (N2+)
        let color = if packet.lem.empirical >= EmpiricalAxis::E3CryptographicallyProven {
            LinearRgba::new(0.0, 0.4, 0.8, 0.8) // Sapphire Blue
        } else if packet.somatic_witnesses.len() > 0 {
            LinearRgba::new(0.8, 0.08, 0.24, 0.8) // Crimson Red
        } else {
            LinearRgba::new(0.0, 0.84, 0.78, 0.6) // Mycelix Cyan
        };

        // Create Holographic Material
        let material = materials.add(HolographicMaterial {
            base_color: color,
            glow_intensity: 2.5,
            pulse_speed: if packet.somatic_witnesses.len() > 0 {
                2.0
            } else {
                0.5
            },
            ..default()
        });

        commands.spawn((
            Mesh3d(marker_mesh.clone()),
            MeshMaterial3d(material),
            Transform::from_xyz(pos[0], pos[1], pos[2]).with_scale(Vec3::splat(0.02)),
            EvidenceMarker {
                packet_id: packet.id,
            },
        ));

        info!(
            "🔮 [Sol Atlas] Holographic Evidence Marker spawned for packet {}",
            packet.id
        );
    }
}
