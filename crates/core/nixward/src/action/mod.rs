// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! NixOS action execution, rollback, and authority migration.
//!
//! The legacy executor still contains historical Phi-threshold confirmation
//! semantics during migration. New governed action work should use the typed
//! action-intent/authorization records in `authorization` and must not treat
//! Phi/confidence as execution authority.

pub mod approver_evidence;
pub mod authorization;
pub mod config_writer;
pub mod daemon_incarnation;
pub mod executor;
pub mod flake_ops;
pub mod gc_manager;
pub mod generation_manager;
pub mod local_approval;
pub mod local_approval_ipc;
pub mod local_approval_store;
pub mod local_approval_submission;
pub mod phi_gate;
pub mod plan_executor;
pub mod service_manager;
pub mod temporal;

pub use approver_evidence::{
    ApproverEvidenceErrorV1, ApproverEvidenceProfileV1, ApproverEvidenceRefV1,
    LocalUnixPeerCredentialEvidenceV1, VerifiedLocalUnixPeerCredentialV1,
    xenia_evidence_ref_v1,
};
pub use authorization::{
    NixActionDescriptorV1, NixActionIntentV1, NixActionScopeV1,
    NixAuthorizationDecisionV1, NixAuthorizationErrorV1, NixAuthorizationProfileV1,
    NixExecutionAuthorizationRecordV1, NixExecutionReceiptV1, NixMechanicalResultV1,
    NixPostconditionStatusV1,
};
pub use config_writer::{ConfigPatch, ConfigWriter, WriteResult};
pub use daemon_incarnation::{DaemonApprovalContextErrorV1, LiveDaemonIncarnationV1};
pub use executor::{
    ChannelOperation, ExecutionRecord, ExecutionResult, FlakeOperation, NixOSCommand,
    NixOSExecutor, SafetyLevel,
};
pub use flake_ops::{FlakeCheckResult, FlakeMetadata, FlakeOps};
pub use gc_manager::{GcAnalysis, GcManager, GcRecommendation};
pub use generation_manager::{Generation, GenerationDiff, GenerationManager};
pub use local_approval::{
    LocalApprovalDecisionKindV1, LocalApprovalErrorV1, LocalNixApprovalDecisionV1,
    PendingNixApprovalRequestV1, digest_display,
};
pub use local_approval_ipc::LocalApprovalIpcErrorV1;
#[cfg(target_os = "linux")]
pub use local_approval_ipc::observe_linux_unix_peer_v1;
pub use local_approval_store::{
    ConsumedLocalApprovalDecisionV1, LocalApprovalRequestStoreErrorV1,
    LocalApprovalRequestStoreV1, PendingRequestInstallV1,
};
pub use local_approval_submission::{
    LocalApprovalAdmissionErrorV1, LocalApprovalSubmissionV1,
    admit_verified_local_submission_v1,
};
pub use phi_gate::{classify_command_destructiveness, get_nixos_rollback};
pub use plan_executor::{PlanExecutionResult, PlanExecutor, PlanStep, StepStatus};
pub use service_manager::{ServiceManager, ServiceStatus};
pub use temporal::{
    EvidenceCurrentnessV1, EvidenceTemporalEvaluationV1, EvidenceTemporalStatusV1,
    EvidenceWindowMillisV1, NixTimeErrorV1, UnixMillisV1, UnixSecondsV1,
};
