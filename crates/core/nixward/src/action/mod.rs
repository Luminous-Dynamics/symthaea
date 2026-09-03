// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! NixOS action execution.
//!
//! Nixward separates **authority** from **cognitive telemetry**. Sub-modules
//! produce `NixOSCommand` values; `execution_context` describes who/what may
//! authorize an action and carries optional cognition metrics. `ContextualExecutor`
//! is the preferred migration path: it enforces explicit authority and exact
//! action binding before delegating to the legacy executor for mature rollback
//! and outcome handling.

pub mod config_writer;
pub mod context_executor;
pub mod execution_context;
pub mod executor;
pub mod flake_ops;
pub mod gc_manager;
pub mod generation_manager;
pub mod phi_gate;
pub mod plan_executor;
pub mod service_manager;

pub use config_writer::{ConfigPatch, ConfigWriter, WriteResult};
pub use context_executor::{ContextExecutionRecord, ContextualExecutor};
pub use execution_context::{
    AuthorityContext, AuthoritySource, CognitiveContext, ExecutionContext, PhiMeasurement,
    EXECUTION_CONTEXT_SCHEMA_VERSION,
};
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
