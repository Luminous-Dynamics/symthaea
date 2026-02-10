//! Φ-Gated Action Execution
//!
//! All NixOS actions are routed through the consciousness-gated executor.
//! Sub-modules produce [`NixOSCommand`] values; the executor checks Φ
//! thresholds, handles rollback, and records outcomes for episodic memory.

pub mod executor;
pub mod phi_gate;
pub mod config_writer;
pub mod flake_ops;
pub mod service_manager;
pub mod generation_manager;
pub mod gc_manager;
pub mod plan_executor;

pub use executor::{
    NixOSCommand, NixOSExecutor, ExecutionResult, ExecutionRecord,
    SafetyLevel, ChannelOperation, FlakeOperation,
};
pub use phi_gate::{get_nixos_rollback, classify_command_destructiveness};
pub use config_writer::{ConfigWriter, ConfigPatch, WriteResult};
pub use flake_ops::{FlakeOps, FlakeMetadata, FlakeCheckResult};
pub use service_manager::{ServiceManager, ServiceStatus};
pub use generation_manager::{GenerationManager, Generation, GenerationDiff};
pub use gc_manager::{GcManager, GcAnalysis, GcRecommendation};
pub use plan_executor::{PlanExecutor, PlanStep, PlanExecutionResult, StepStatus};
