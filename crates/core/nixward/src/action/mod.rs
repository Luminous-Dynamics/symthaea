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
pub mod local_approval_projection;
#[cfg(target_os = "linux")]
pub mod local_approval_runtime;
#[cfg(target_os = "linux")]
pub mod local_approval_socket;
pub mod local_approval_store;
pub mod local_approval_submission;
pub mod post_state;
pub mod phi_gate;
pub mod plan_executor;
pub mod service_manager;
pub mod service_domain;
pub mod service_effect;
pub mod service_state;
#[cfg(feature = "systemd-observer")]
pub mod systemd_definition;
#[cfg(feature = "systemd-observer")]
pub mod systemd_observer;
#[cfg(feature = "systemd-mutation")]
pub mod systemd_mutation;
pub(crate) mod systemd_transport;
pub mod temporal;

pub use approver_evidence::{
    ApproverEvidenceErrorV1, ApproverEvidenceProfileV1, ApproverEvidenceRefV1,
    RequiredApprovalProfileV1,
    LocalUnixPeerCredentialEvidenceV1, VerifiedLocalUnixPeerCredentialV1,
    xenia_evidence_ref_v1,
};
pub use authorization::{
    NixActionDescriptorV1, NixActionIntentV1, NixActionScopeV1,
    NixAuthorizationDecisionV1, NixAuthorizationErrorV1, NixAuthorizationProfileV1,
    NixExecutionAuthorizationRecordV1, NixExecutionReceiptV1, NixLocalExecutionAuthorityV1,
    NixMechanicalResultV1,
    NixPostconditionStatusV1,
};
pub use post_state::{
    NixPostStateClaimV1, NixPostStateErrorV1, NixPostStateReceiptV1,
    NixPostStateStabilityEvidenceV1, NixPostStateStabilitySampleV1,
    NixPostconditionAssessmentV1, NixServicePostStateExpectationV1,
    NixServicePostStateObservationV1, NixSystemdJobEvidenceV1, NixSystemdJobTypeV1,
    NixSystemdUnitDefinitionIdentityV1, NixVerifiedPostStateObservationV1,
    NixVerifiedPostStateStabilityEvidenceV1,
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
    PendingNixApprovalRequestV1, digest_display, operator_visible_action_for_command,
};
pub use local_approval_ipc::LocalApprovalIpcErrorV1;
pub use local_approval_projection::{
    LocalApprovalProjectionErrorV1, PendingNixApprovalProjectionV1,
};
#[cfg(target_os = "linux")]
pub use local_approval_ipc::observe_linux_unix_peer_v1;
#[cfg(target_os = "linux")]
pub use local_approval_runtime::{
    InstalledLocalApprovalRequestV1, LocalApprovalRuntimeErrorV1, LocalApprovalRuntimeV1,
};
#[cfg(target_os = "linux")]
pub use local_approval_socket::{
    LOCAL_APPROVAL_MAX_FRAME_BYTES_V1, LOCAL_APPROVAL_PROTOCOL_V2,
    LOCAL_APPROVAL_SOCKET_FILENAME_V1, LocalApprovalAckStatusV1, LocalApprovalAckV1,
    LocalApprovalSocketErrorV1, LocalApprovalSocketServerV1, LocalApprovalWireRequestV2,
    default_local_approval_runtime_dir_v1, submit_local_approval_v2,
};
pub use local_approval_store::{
    ConsumedLocalApprovalDecisionV1, LocalApprovalRequestStoreErrorV1,
    LocalApprovalRequestStoreV1, PendingRequestCurrentnessV1, PendingRequestInstallV1,
};
pub use local_approval_submission::{
    LocalApprovalAdmissionErrorV1, LocalApprovalSubmissionV1, LocalApprovalSubmissionV2,
};
pub use phi_gate::{classify_command_destructiveness, get_nixos_rollback};
pub use plan_executor::{PlanExecutionResult, PlanExecutor, PlanStep, StepStatus};
pub use service_manager::{ServiceManager, ServiceStatus};
pub use service_effect::{
    NixServiceEffectContextErrorV1, NixServiceEffectContextV1,
};
pub use service_domain::{NixServiceOperationErrorV1, NixServiceOperationKindV1, NixServiceOperationV1};
#[cfg(feature = "systemd-observer")]
pub use systemd_definition::{
    NixDefinitionFileContentDigestV1, NixSystemdDefinitionContentCommitmentV1,
    NixSystemdDefinitionContentErrorV1, NixVerifiedSystemdDefinitionContentCommitmentV1,
};
#[cfg(feature = "systemd-observer")]
pub use systemd_observer::{
    NixSystemdJobHandleV1, NixSystemdJobRemovedWatcherV1, NixSystemdObserverErrorV1,
    NixSystemdReadOnlyObserverV1,
};
#[cfg(feature = "systemd-mutation")]
pub use systemd_mutation::{
    NixSystemdLifecycleMutationTransportV1, NixSystemdMutationTransportErrorV1,
};
pub use service_state::{
    NixServiceEnablementEvidenceV1,
    NixServiceOperationCapabilitiesV1,
    ServiceLoadStateV1,
    NixServiceObservedStateV1, NixServiceStateErrorV1, ServiceActiveStateV1,
    ServiceUnitFileStateV1,
};
pub use temporal::{
    EvidenceCurrentnessV1, EvidenceTemporalEvaluationV1, EvidenceTemporalStatusV1,
    EvidenceWindowMillisV1, NixTimeErrorV1, UnixMillisV1, UnixSecondsV1,
};
