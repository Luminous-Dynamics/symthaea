//! Fabrication manager for advanced manufacturing integration.
//!
//! Requires feature: `advanced-manufacturing`

use serde::{Deserialize, Serialize};

/// Events produced by the fabrication subsystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FabricationEvent {
    /// What happened.
    pub kind: FabricationEventKind,
    /// Cycle at which the event occurred.
    pub cycle: u64,
}

/// Kinds of fabrication events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FabricationEventKind {
    /// A new fabrication job was queued.
    JobQueued,
    /// A fabrication job completed.
    JobCompleted,
}

/// Telemetry snapshot from the fabrication subsystem.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FabricationTelemetry {
    /// Number of jobs completed.
    pub jobs_completed: u64,
}

/// Manager for fabrication processes within the cognitive loop.
#[derive(Debug)]
pub struct FabricationManager {
    telemetry: FabricationTelemetry,
}

impl FabricationManager {
    /// Create a new fabrication manager.
    pub fn new() -> Self {
        Self {
            telemetry: FabricationTelemetry::default(),
        }
    }

    /// Get current telemetry.
    pub fn telemetry(&self) -> &FabricationTelemetry {
        &self.telemetry
    }
}
