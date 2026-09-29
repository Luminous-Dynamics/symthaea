// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Engineering reasoning facade for Symthaea.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_broca::{BrocaConfig, BrocaGenerator, ThoughtChannels};
use symthaea_causal_reasoning::causal_calculus::StructuralCausalModel;
use symthaea_digital_twin::TwinState;
use symthaea_fabrication_kernel::autonomy_loop::{AutonomyEvent, AutonomyLoop};
use symthaea_fabrication_kernel::cincinnati_live::{AnomalyAlert, CincinnatiMonitor};
use symthaea_fabrication_kernel::csg::CSGNode;
use symthaea_fabrication_kernel::{GeometricThought, TriangleMesh};
use symthaea_formal_safety::{EvidenceKind, ProofObligation, SafetyCase};
use symthaea_harmonies::{AlignmentResult, EightHarmonies};
use symthaea_materials::{MaterialAgingModel, MaterialProperty};
use symthaea_memory::{
    Episode, EpisodicMemory, EpisodicReplayConfig, MemoryCoordinator, SemanticMemory,
};
use symthaea_sim_bridge::{
    AmygdalaInterlock, EngineeringDomain, MetricEncoder, SimulationRegistry, SimulationRequest,
    SurpriseMonitor,
};
use symthaea_swarm::{SwarmAggregator, SwarmMessage, SwarmProofMsg, SwarmStateMsg};
use symthaea_workspace::GlobalWorkspace;

pub use symthaea_digital_twin as digital_twin;
pub use symthaea_formal_safety as formal_safety;
pub use symthaea_memory as memory;
pub use symthaea_sim_bridge as sim_bridge;
pub mod engineering_identity;
pub mod engineering_relation;
pub use engineering_identity::EngineeringObjectId;
pub use engineering_relation::{EngineeringRelation, EngineeringRelationKind, RelationFamily};
pub mod scientific_lineage;
pub use scientific_lineage::{AuthorityCeiling, QualificationProjection, ScientificLineageGraph};

/// Debug-friendly wrapper for the fabrication autonomy loop.
pub struct DebugFabricationLoop(pub AutonomyLoop);
impl std::fmt::Debug for DebugFabricationLoop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AutonomyLoop")
            .field("state", &self.0.state())
            .finish()
    }
}
