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
pub use scientific_lineage::{
    AuthorityCeiling, QualificationProjection, QualificationProjectionError,
    ScientificLineageGraph, QUALIFICATION_POLICY, QUALIFICATION_PROJECTION_SCHEMA,
};

/// Debug-friendly wrapper for the fabrication autonomy loop.
pub struct DebugFabricationLoop(pub AutonomyLoop);
impl std::fmt::Debug for DebugFabricationLoop {