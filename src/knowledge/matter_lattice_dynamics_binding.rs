// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Public boundary for the batched lattice-dynamics execution graph.
//!
//! The raw graph remains crate-private so callers have one admission path. A
//! solver execution may contain one or many q-point samples; the final
//! dynamical-stability claim is a separate aggregate and may not alias a batch
//! claim. Those invariants are enforced by the private graph implementation.

use symthaea_evidence_plane::external_receipt::ExternalEvidenceBundle;

pub use super::matter_lattice_dynamics_graph::{
    LatticeDynamicsAggregationBinding, LatticeDynamicsAggregationReceipt,
    LatticeDynamicsExecutionGraphBinding, LatticeDynamicsExecutionGraphReceipt,
    LatticeDynamicsGraphError, LatticeQPointBatchBinding, LatticeQPointBatchReceipt,
    LatticeQPointSampleBinding, LatticeQPointSampleReceipt,
};

/// The only public lattice-dynamics graph admission path.
pub fn bind_lattice_dynamics_execution_graph(
    evidence_bound: super::matter_crystal_evidence_binding::EvidenceBoundCrystalPhaseMatterClaim,
    graph: LatticeDynamicsExecutionGraphReceipt,
    bundle: &ExternalEvidenceBundle,
) -> Result<LatticeDynamicsExecutionGraphBinding, LatticeDynamicsGraphError> {
    super::matter_lattice_dynamics_graph::bind_lattice_dynamics_execution_graph(
        evidence_bound,
        graph,
        bundle,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_surface_exposes_batch_not_one_process_per_qpoint_types() {
        let _sample = LatticeQPointSampleReceipt {
            q_point_id: "q0".into(),
            q_fractional_bits: [0.0_f64.to_bits(); 3],
            dynamical_matrix_artifact_id: "dynmat:q0".into(),
        };
        let _ = std::mem::size_of::<LatticeQPointBatchReceipt>();
        let _ = std::mem::size_of::<LatticeQPointBatchBinding>();
    }
}
