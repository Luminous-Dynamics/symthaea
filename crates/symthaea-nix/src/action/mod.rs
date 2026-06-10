// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Φ-Gated Action Execution
//!
//! All NixOS actions are routed through the consciousness-gated executor.
//! Sub-modules produce `NixOSCommand` values; the executor checks Φ
//! thresholds, handles rollback, and records outcomes for episodic memory.

pub mod config_writer;
pub mod executor;
pub mod flake_ops;
pub mod gc_manager;
pub mod generation_manager;
pub mod phi_gate;
pub mod plan_executor;
pub mod service_manager;

pub use config_writer::{ConfigPatch, ConfigWriter, WriteResult};
pub use executor::{
    ChannelOperation, ExecutionRecord, ExecutionResult, FlakeOperation, NixOSCommand,
    NixOSExecutor, SafetyLevel,
};
pub use flake_ops::{FlakeCheckResult, FlakeMetadata, FlakeOps};
pub use gc_manager::{GcAnalysis, GcManager, GcRecommendation};
pub use generation_manager::{Generation, GenerationDiff, GenerationManager};
pub use phi_gate::{classify_command_destructiveness, get_nixos_rollback};
pub use plan_executor::{PlanExecutionResult, PlanExecutor, PlanStep, StepStatus};
pub use service_manager::{ServiceManager, ServiceStatus};
