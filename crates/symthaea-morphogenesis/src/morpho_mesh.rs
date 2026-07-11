// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Morphogenetic Mesh and MEA Telemetry Ingest.
//!
//! Facilitates the transition from synthetic grids to real-world biological
//! datasets (MEA, voltage-sensitive dye imaging). Uses graph-based coordinate
//! encoding where hypervector similarity mirrors biological proximity.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::unified_hv::ContinuousHV;

/// A Multi-Electrode Array (MEA) data packet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeaPacket {
    /// Timestamp in milliseconds.
    pub timestamp_ms: u64,
    /// Voltage readings from each electrode.
    pub electrode_voltages: Vec<f32>,
}

/// Node in the organic morphogenetic mesh.
#[derive(Debug, Clone)]
pub struct MeshNode {
    pub id: usize,
    pub x: f32,
    pub y: f32,
    /// Neighbors in the gap-junction graph.
    pub neighbors: Vec<usize>,
    /// Unique random hypervector for this node.
    pub base_hv: ContinuousHV,
    /// Smoothed spatial hypervector (proximity-aware).
    pub spatial_hv: ContinuousHV,
}

/// Organic Mesh Adapter for Biological Telemetry.
pub struct MorphoMeshAdapter {
    pub dim: usize,
    pub nodes: Vec<MeshNode>,
    /// Prototype for hyperpolarized state (Voltage > 0).
    pub hyper_prototype: ContinuousHV,
    /// Prototype for depolarized state (Voltage < 0).
    pub depol_prototype: ContinuousHV,
}

impl MorphoMeshAdapter {
    /// Create a new mesh adapter from electrode coordinates.
    pub fn new_from_mea(
        dim: usize,
        coords: &[(f32, f32)],
        adjacency_threshold: f32,
        seed: u64,
        hyper_prototype: ContinuousHV,
        depol_prototype: ContinuousHV,
    ) -> Self {
        let n = coords.len();
        let mut nodes = Vec::with_capacity(n);

        // 1. Generate base random vectors for each electrode
        for i in 0..n {
            let (x, y) = coords[i];
            let base_hv = ContinuousHV::random(dim, seed.wrapping_add(i as u64));
            nodes.push(MeshNode {
                id: i,
                x,
                y,
                neighbors: Vec::new(),
                base_hv,
                spatial_hv: ContinuousHV::zero(dim), // Placeholder
            });
        }

        // 2. Build gap-junction adjacency graph based on physical proximity
        for i in 0..n {
            for j in (i + 1)..n {
                let dist =
                    ((nodes[i].x - nodes[j].x).powi(2) + (nodes[i].y - nodes[j].y).powi(2)).sqrt();
                if dist <= adjacency_threshold {
                    nodes[i].neighbors.push(j);
                    nodes[j].neighbors.push(i);
                }
            }
        }

        // 3. Compute distance-preserving spatial hypervectors
        // c_i = normalize(base_i + sum(neighbors_j))
        let mut spatial_hvs = Vec::with_capacity(n);
        for i in 0..n {
            let mut bundle_set = vec![&nodes[i].base_hv];
            for &neigh_idx in &nodes[i].neighbors {
                bundle_set.push(&nodes[neigh_idx].base_hv);
            }
            spatial_hvs.push(ContinuousHV::bundle(&bundle_set).normalize());
        }

        for i in 0..n {
            nodes[i].spatial_hv = spatial_hvs[i].clone();
        }

        Self {
            dim,
            nodes,
            hyper_prototype,
            depol_prototype,
        }
    }

    /// Map a raw MEA voltage packet into a unified tissue hypervector.
    ///
    /// H_tissue = sum( spatial_i ⊗ encode(voltage_i) )
    pub fn ingest_mea_packet(&self, packet: &MeaPacket) -> ContinuousHV {
        let mut cell_hvs = Vec::with_capacity(self.nodes.len());

        for (i, &v) in packet.electrode_voltages.iter().enumerate() {
            if i >= self.nodes.len() {
                break;
            }

            // Map scalar voltage to discrete state prototype
            let state = if v >= 0.0 {
                &self.hyper_prototype
            } else {
                &self.depol_prototype
            };

            // h_cell = spatial ⊗ state
            let cell_hv = self.nodes[i].spatial_hv.bind(state);
            cell_hvs.push(cell_hv);
        }

        let refs: Vec<&ContinuousHV> = cell_hvs.iter().collect();
        ContinuousHV::bundle(&refs).normalize()
    }

    /// Get all spatial coordinate hypervectors.
    pub fn spatial_coordinates(&self) -> Vec<ContinuousHV> {
        self.nodes.iter().map(|n| n.spatial_hv.clone()).collect()
    }
}
