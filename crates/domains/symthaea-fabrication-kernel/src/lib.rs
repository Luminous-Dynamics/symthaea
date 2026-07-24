// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea Fabrication Kernel
//!
//! HDC-to-Mesh bridge: geometric primitives encoded as hypervectors,
//! CSG boolean operations, bounded mesh repair, manufacturability qualification,
//! revocation-aware provenance, role-aware release quorum, timed machine-session authority, signed operator commands, multi-gateway quorum, verified disaster recovery, immutable incident evidence, release-candidate certification, threshold promotion ceremonies, gateway membership rotation, fenced partition leases, Merkle transparency logs, independent checkpoint witnesses and gossip, reproducible artifact provenance, cross-region quorum, evidence-bound rollback, compromised-signer containment, immutable gateway tombstones, hardware rollout revocation, post-rollback requalification, quorum-derived clock evidence, monotonic authority epochs, explicit policy migration, offline recovery-key ceremonies, anchored evidence compaction, secure upgrade handoff, post-activation probation, automatic rollback triggers, hardware reauthorization, retention and continuity evidence, evidence-complete upgrade finalization, packaged STL/3MF export, and simulation backends.

#[cfg(feature = "analytical")]
pub mod analytical;
pub mod artifact_provenance;
pub mod artifact_set;
pub mod attestation;
pub mod audit;
pub mod audit_evidence;
pub mod authority_epoch;
pub mod automatic_rollback;
pub mod autonomy_loop;
pub mod blueprint;
pub mod bsp;
pub mod building;
pub mod clock;
pub mod clock_continuity;
pub mod containment_replay;
pub mod containment_state;
pub mod crypto_digest;
pub mod csg;
pub mod design_loop;
pub mod evidence_anchor;
pub mod evidence_compaction;
pub mod evidence_retention;
pub mod execution_guard;
pub mod export;
pub mod fault_injection;
pub mod gateway_consensus;
pub mod gateway_consensus_tracker;
pub mod gateway_decommission;
pub mod gateway_decommission_tracker;
pub mod gateway_membership;
pub mod gateway_recovery;
pub mod gateway_replay;
pub mod gateway_state;
pub mod gateway_store;
pub mod gateway_tombstone;
pub mod gateway_tombstone_registry;
#[cfg(feature = "analytical")]
pub mod generative;
pub mod governance;
pub mod import;
pub mod incident;
pub mod incident_ledger;
pub mod infill;
pub mod intersection;
pub mod key_continuity;
pub mod lease_authority;
pub mod machine;
pub mod manufacturability;
pub mod manufacturing;
pub mod material_handling;
pub mod mesh;
pub mod policy_migration;
pub mod policy_migration_tracker;
pub mod post_rollback_requalification;
pub mod post_rollback_requalification_tracker;
pub mod primitives;
pub mod printer_control;
pub mod process;
pub mod promotion_replay;
pub mod provenance;
pub mod qualification;
pub mod reconciliation;
pub mod recovery;
pub mod recovery_key;
pub mod region_quorum;
pub mod region_quorum_tracker;
pub mod release;
pub mod release_assurance;
pub mod release_certification;
pub mod release_lineage;
pub mod release_promotion;
pub mod release_rollback;
pub mod repair;
pub mod replay;
pub mod resilience_state;
pub mod rollback_replay;
pub mod rollback_transparency;
pub mod rollout;
pub mod rollout_revocation;
pub mod rollout_revocation_tracker;
pub mod rotation;
pub mod session;
pub mod signer_compromise;
pub mod signer_compromise_tracker;
pub mod simulator;
pub mod slicer;
pub mod submission;
pub mod submission_coordinator;
pub mod submission_ledger;
pub mod support;
pub mod telemetry;
pub mod telemetry_tracker;
pub mod thought;
pub mod threshold;
pub mod toolpath;
pub mod transparency;
pub mod transparency_checkpoint;
pub mod transparency_witness;
pub mod transparency_witness_tracker;
pub mod trust;
pub mod units;
pub mod upgrade_bundle;
pub mod upgrade_finalization;
pub mod upgrade_handoff;
pub mod upgrade_operational_bundle;
pub mod upgrade_operational_replay;
pub mod upgrade_operational_state;
pub mod upgrade_probation;
pub mod upgrade_probation_tracker;
pub mod upgrade_replay;
pub mod upgrade_state;
pub mod upgrade_tracker;
pub mod validate;
pub mod witness_gossip;
pub mod witness_gossip_tracker;

pub mod cincinnati_live;
pub mod defect_prediction;
pub mod delegation;
pub mod hardware_config;
pub mod hardware_reauthorization;
pub mod hardware_reauthorization_tracker;
pub mod nurbs;
pub mod operational_replay;
pub mod operator_command;
pub mod operator_command_tracker;
pub mod package;
pub mod step_import;

pub use automatic_rollback::{
    AuthorizedAutomaticRollback, AutomaticRollbackError, AutomaticRollbackPolicy,
    AutomaticRollbackTrigger, RollbackTriggerKind, UpgradeHealthSignal,
    authorize_automatic_rollback, digest_automatic_rollback_trigger, digest_upgrade_health_signal,
    evaluate_automatic_rollback,
};
pub use clock_continuity::{
    ClockContinuityError, ClockContinuityPolicy, VerifiedClockContinuity, digest_clock_continuity,
    verify_clock_continuity,
};
pub use evidence_retention::{
    AuthorizedEvidenceRetentionPolicy, EvidenceClass, EvidenceDescriptor, EvidenceLegalHold,
    EvidenceRetentionAction, EvidenceRetentionDecision, EvidenceRetentionError,
    EvidenceRetentionPolicy, EvidenceRetentionRule, authorize_evidence_retention_policy,
    digest_evidence_retention_decision, digest_evidence_retention_policy,
    evaluate_evidence_retention,
};
pub use hardware_reauthorization::{
    HardwareReauthorizationError, HardwareReauthorizationPolicy, HardwareReauthorizationSigner,
    HardwareReauthorizationStatement, HardwareReauthorizationVerifier,
    SignedHardwareReauthorization, VerifiedHardwareReauthorization,
    digest_hardware_reauthorization_statement, sign_hardware_reauthorization,
    verify_hardware_reauthorization,
};
pub use hardware_reauthorization_tracker::{
    HardwareReauthorizationRecord, HardwareReauthorizationTracker,
    HardwareReauthorizationTrackingError, digest_hardware_reauthorization_tracker,
};
pub use key_continuity::{
    KeyContinuityError, KeyContinuityPolicy, VerifiedKeyContinuity, digest_key_continuity,
    verify_key_continuity,
};
pub use upgrade_operational_bundle::{
    UpgradeOperationalBundleError, UpgradeOperationalBundleLimits,
    UpgradeOperationalEvidenceBundle, build_upgrade_operational_evidence_bundle,
    decode_upgrade_operational_evidence_bundle, digest_upgrade_operational_evidence_bundle,
    encode_upgrade_operational_evidence_bundle, verify_upgrade_operational_evidence_bundle,
};
pub use upgrade_operational_replay::{
    UpgradeOperationalReplayContract, UpgradeOperationalReplayError,
    UpgradeOperationalReplayMismatch, UpgradeOperationalReplayVerificationReport,
    build_upgrade_operational_replay_contract, digest_upgrade_operational_replay_contract,
    verify_upgrade_operational_replay_contract,
};
pub use upgrade_operational_state::{
    FabricationUpgradeOperationalState, UpgradeOperationalEvidenceDigests,
    UpgradeOperationalStateError, digest_upgrade_operational_state,
    verify_upgrade_operational_state_successor,
};
pub use upgrade_probation::{
    AuthorizedUpgradeProbationClearance, UpgradeProbationError, UpgradeProbationEvidence,
    UpgradeProbationObservation, UpgradeProbationPolicy, authorize_upgrade_probation_clearance,
    build_upgrade_probation_evidence, digest_upgrade_probation_evidence,
    digest_upgrade_probation_observation,
};
pub use upgrade_probation_tracker::{
    UpgradeProbationTracker, UpgradeProbationTrackingError, digest_upgrade_probation_tracker,
};

pub use artifact_provenance::{
    ArtifactProvenanceError, ArtifactProvenancePolicy, ArtifactProvenanceSigner,
    ArtifactProvenanceStatement, ArtifactProvenanceVerifier, ArtifactProvenanceViolation,
    ProvenanceInput, SignedArtifactProvenance, VerifiedArtifactProvenance,
    build_artifact_provenance_statement, digest_artifact_provenance_statement,
    sign_artifact_provenance, verify_artifact_provenance,
};
pub use artifact_set::{
    ArtifactSetError, ReleaseArtifact, ReleaseArtifactSet, build_release_artifact_set,
    digest_release_artifact_set, verify_release_artifact_set,
};
pub use attestation::{
    AttestationBuildError, AttestationPolicy, AttestationTrustContext,
    AttestationVerificationReport, AttestationViolation, AttestedFabricationManifest,
    DetachedSignature, ManifestSignatureVerifier, ManifestSigner, SignatureAlgorithm,
    VerifiedAttestation, attest_fabrication_manifest, verify_attestation_authority,
    verify_attestation_authority_with_trust, verify_attested_manifest,
    verify_attested_manifest_with_trust,
};
pub use audit::{
    AuditAction, AuditAppendError, AuditEvent, AuditJournal, AuditVerificationReport,
    AuditViolation, compute_audit_event_hash, digest_audit_journal,
};
pub use audit_evidence::{
    AuditAnchor, AuditAnchorError, AuditAnchorSigner, AuditAnchorVerifier, AuditSegment,
    AuditSegmentError, AuditSegmentVerificationReport, SignedAuditAnchor, VerifiedAuditAnchor,
    digest_audit_anchor, export_audit_segment, sign_audit_anchor, verify_audit_segment,
    verify_signed_audit_anchor,
};
pub use authority_epoch::{
    AUTHORITY_EPOCH_SCHEMA, AuthorityEpochError, AuthorityEpochTracker,
    AuthorityEpochTrackingError, AuthorityEpochVector, digest_authority_epoch,
    digest_authority_epoch_tracker,
};
pub use bsp::{csg_intersect, csg_subtract, csg_union};
pub use clock::{
    CLOCK_OBSERVATION_SCHEMA, ClockEpochTracker, ClockObservation, ClockObservationVerifier,
    ClockQuorumPolicy, ClockTrackingError, ClockViolation, VerifiedClockWindow,
    canonical_clock_observation_bytes, digest_clock_epoch_tracker, digest_clock_observation,
    verify_clock_quorum,
};
pub use containment_replay::{
    ContainmentReplayContract, ContainmentReplayError, ContainmentReplayMismatch,
    ContainmentReplayVerificationReport, build_containment_replay_contract,
    digest_containment_replay_contract, verify_containment_replay_contract,
};
pub use containment_state::{
    ContainmentStateError, FabricationContainmentState, digest_containment_state,
    verify_containment_state_successor,
};
pub use crypto_digest::{DigestParseError, Sha256, Sha256Digest, sha256};
pub use csg::{BooleanOp, CSGNode, Primitive, Transform3D};
pub use delegation::{
    DelegatedReleaseAuthorization, DelegationBuildError, DelegationGrantBody, DelegationSigner,
    DelegationVerifier, DelegationViolation, SignedDelegationGrant,
    authorize_release_with_delegations, digest_delegation_grants, sign_delegation_grant,
};
pub use evidence_anchor::{
    AuthorizedEvidenceCompactionAnchor, EvidenceAnchorError, EvidenceAnchorPolicy,
    EvidenceAnchorTracker, EvidenceAnchorTrackingError, EvidenceCompactionAnchor,
    authorize_evidence_compaction_anchor, build_evidence_compaction_anchor,
    digest_evidence_compaction_anchor, verify_evidence_compaction_anchor,
};
pub use evidence_compaction::{
    COMPACTED_EVIDENCE_SCHEMA, CompactedEvidence, EVIDENCE_RECORD_SCHEMA, EvidenceCompactionError,
    EvidenceCompactionPolicy, EvidenceCompactionTracker, EvidenceCompactionTrackingError,
    EvidenceJournal, EvidenceRecord, compact_evidence, digest_compacted_evidence,
    digest_evidence_compaction_tracker, empty_evidence_head,
};
pub use execution_guard::{
    ContainmentAction, ExecutionCheckpointError, ExecutionGuard, ExecutionGuardCheckpoint,
    ExecutionGuardPolicy, ExecutionTelemetry, GuardDecision, GuardViolation,
    digest_execution_checkpoint,
};
#[allow(deprecated)]
pub use export::export_3mf;
pub use export::{
    GovernedPackageBuildError, export_3mf_model_xml, export_3mf_package,
    export_3mf_package_with_attestation, export_3mf_package_with_governance,
    export_3mf_package_with_manifest, export_stl,
};
pub use gateway_consensus::{
    GatewayConsensusError, GatewayConsensusPolicy, GatewayConsensusViolation, GatewayEndorsement,
    GatewayEndorsementSigner, GatewayEndorsementVerifier, SignedGatewayEndorsement,
    VerifiedGatewayConsensus, canonical_gateway_endorsement_bytes, digest_gateway_endorsement,
    endorse_gateway_state, verify_gateway_consensus,
};
pub use gateway_consensus_tracker::{
    AcceptedGatewayConsensus, GatewayConsensusTracker, GatewayConsensusTrackingError,
};
pub use gateway_decommission::{
    AuthorizedGatewayDecommission, GatewayDecommissionError, GatewayDecommissionPlan,
    GatewayDecommissionPolicy, authorize_gateway_decommission, build_gateway_decommission_plan,
    digest_gateway_decommission_plan,
};
pub use gateway_decommission_tracker::{
    GatewayDecommissionTracker, GatewayDecommissionTrackingError, GatewayRetirementRecord,
    GatewayRetirementStage, digest_gateway_decommission_tracker,
};
pub use gateway_membership::{
    AuthorizedGatewayMembership, GatewayMember, GatewayMembership, GatewayMembershipError,
    GatewayMembershipPolicy, GatewayMembershipTransition, authorize_membership_transition,
    build_membership_transition, digest_gateway_membership, digest_membership_transition,
};
pub use gateway_recovery::{
    GatewayConsensusEvidence, GatewayRecoveryBundle, GatewayRecoveryCheckpoint,
    GatewayRecoveryError, digest_recovery_bundle,
};
pub use gateway_replay::{
    GatewayReplayContract, GatewayReplayError, GatewayReplayMismatch,
    GatewayReplayVerificationReport, build_gateway_replay_contract, digest_gateway_replay_contract,
    digest_reconciliation_report, verify_gateway_replay_contract,
};
pub use gateway_state::{
    FabricationGatewayState, GatewayEvidenceDigests, GatewayStateEnvelope, GatewayStateError,
    verify_gateway_state_successor,
};
pub use gateway_store::{GatewayStateStore, GatewayStoreError};
pub use gateway_tombstone::{
    AuthorizedGatewayTombstone, GatewayTombstone, GatewayTombstoneError,
    authorize_gateway_tombstone, build_gateway_tombstone, digest_gateway_tombstone,
};
pub use gateway_tombstone_registry::{
    GatewayTombstoneRecord, GatewayTombstoneRegistry, GatewayTombstoneRegistryError,
    digest_gateway_tombstone_registry,
};
pub use governance::{FabricationGovernance, GovernanceError};
pub use import::{
    StlError, StlParseLimits, parse_ascii_stl, parse_ascii_stl_with_limits, parse_binary_stl,
    parse_binary_stl_with_limits, parse_stl, parse_stl_with_limits,
};
pub use incident::{
    IncidentBundle, IncidentBundleError, IncidentBundleSigner, IncidentBundleVerifier,
    IncidentKind, IncidentVerificationViolation, SignedIncidentBundle, VerifiedIncidentBundle,
    digest_incident_bundle, sign_incident_bundle, verify_incident_bundle,
};
pub use incident_ledger::{
    IncidentLedger, IncidentLedgerAction, IncidentLedgerError, IncidentLedgerEvent,
    IncidentLedgerVerificationReport, IncidentLedgerViolation, digest_incident_ledger,
};
pub use infill::{
    InfillConfig, InfillError, InfillPattern, clip_segment_to_layer, generate_infill,
    generate_infill_for_layer, point_in_layer_material, try_generate_infill,
    try_generate_infill_for_layer,
};
pub use intersection::{
    DEFAULT_SELF_INTERSECTION_PAIR_BUDGET, SelfIntersectionReport, find_self_intersections,
};
pub use lease_authority::{
    AcceptedPartitionLease, AuthorizedPartitionLease, LeaseAuthorityTracker, LeaseTrackingError,
    PartitionLease, PartitionLeaseError, PartitionLeasePolicy, authorize_partition_lease,
    digest_partition_lease,
};
pub use machine::{
    MachineCapabilities, MachineNegotiationViolation, MachineProfile, MachineSession,
    MachineSessionWindow, MachineValidationReport, MachineViolation, MachineViolationReason,
    NegotiatedMachine, TimedMachineSession, ValidatedGCode, digest_timed_machine_session,
    negotiate_machine_profile, negotiate_machine_profile_at, submit_validated_gcode,
    validate_gcode_for_machine,
};
pub use manufacturability::{
    MinimumFeatureError, MinimumFeaturePolicy, MinimumFeatureReport, analyze_minimum_features,
};
pub use mesh::{TessellationPolicy, TriangleMesh, resolve_to_mesh, resolve_to_mesh_with_policy};
pub use operational_replay::{
    OperationalFabricationReplayContract, OperationalReplayError, OperationalReplayMismatch,
    OperationalReplayVerificationReport, build_operational_replay_contract,
    digest_fault_injection_matrix, digest_operational_replay_contract,
    verify_operational_replay_contract,
};
pub use operator_command::{
    OperatorCommand, OperatorCommandError, OperatorCommandExpectation, OperatorCommandKind,
    OperatorCommandPolicy, OperatorCommandSigner, OperatorCommandVerifier,
    OperatorCommandViolation, SignedOperatorCommand, VerifiedOperatorCommand,
    canonical_operator_command_bytes, digest_operator_command, sign_operator_command,
    verify_operator_command,
};
pub use operator_command_tracker::{
    AppliedOperatorCommand, OperatorCommandTracker, OperatorCommandTrackingError,
    OperatorExecutionState,
};
pub use package::{
    Inspected3mfPackage, PackageError, PackageInspectionLimits, inspect_3mf_package,
    verify_attested_3mf_package, verify_governed_3mf_package,
};
pub use policy_migration::{
    AuthorizedPolicyMigration, PolicyBinding, PolicyInvariantBinding, PolicyInvariantDisposition,
    PolicyInvariantMigration, PolicyMigrationError, PolicyMigrationPlan, PolicyMigrationPolicy,
    authorize_policy_migration, digest_policy_binding, digest_policy_migration_plan,
};
pub use policy_migration_tracker::{
    PolicyMigrationRecord, PolicyMigrationTracker, PolicyMigrationTrackingError,
    digest_policy_migration_tracker, empty_policy_migration_record_digest,
};
pub use post_rollback_requalification::{
    AuthorizedPostRollbackRequalification, PostRollbackRequalificationError,
    PostRollbackRequalificationEvidence, PostRollbackRequalificationPolicy,
    authorize_post_rollback_requalification, build_post_rollback_requalification_evidence,
    digest_post_rollback_requalification,
};
pub use post_rollback_requalification_tracker::{
    PostRollbackRequalificationRecord, PostRollbackRequalificationTracker,
    PostRollbackRequalificationTrackingError, digest_post_rollback_requalification_tracker,
};
pub use primitives::*;
pub use printer_control::{
    MockPrinter, MoonrakerClient, OctoPrintClient, PrinterApi, PrinterError, PrinterStatus,
    TemperatureReading, printer_from_url,
};
pub use process::{
    FabricationProcessPolicy, ProcessPreparationError, ProcessPreparationReport,
    ProcessPreparedMesh, ProcessViolation,
};
pub use promotion_replay::{
    PromotionReplayContract, PromotionReplayError, PromotionReplayMismatch,
    PromotionReplayVerificationReport, build_promotion_replay_contract,
    digest_promotion_replay_contract, verify_promotion_replay_contract,
};
pub use provenance::{
    FabricationManifest, FabricationManifestError, GeometryFingerprintError,
    GeometryFingerprintPolicy, ManifestMismatch, ManifestVerificationReport, StableFingerprint,
    build_fabrication_manifest, canonical_manifest_bytes, digest_fabrication_manifest,
    fingerprint_gcode_program, fingerprint_machine_profile, fingerprint_mesh_geometry,
    fingerprint_minimum_feature_policy, fingerprint_minimum_feature_report,
    fingerprint_process_policy, fingerprint_process_report, fingerprint_slice_config,
    fingerprint_slice_layers, fingerprint_toolpath_config, verify_fabrication_manifest,
};
pub use qualification::{ManufacturingQualificationError, ManufacturingReadyMesh};
pub use recovery::{
    InterruptedPrintEvidence, RecoveryAuthorizationError, RecoveryPolicy, reauthorize_print_restart,
};
pub use recovery_key::{
    AuthorizedRecoveryActivation, RecoveryActivationPolicy, RecoveryActivationRequest,
    RecoveryActivationTracker, RecoveryKeyError, RecoveryKeySet, RecoveryParticipant,
    RecoveryScope, RecoveryTrackingError, authorize_recovery_activation,
    digest_recovery_activation_request, digest_recovery_activation_tracker,
    digest_recovery_key_set,
};
pub use region_quorum::{
    RegionalQuorumError, RegionalQuorumEvidence, RegionalQuorumPolicy, RegionalWeight,
    build_regional_quorum_evidence, digest_regional_quorum_evidence,
    validate_regional_quorum_evidence,
};
pub use region_quorum_tracker::{
    RegionalQuorumTracker, RegionalQuorumTrackingError, digest_regional_quorum_tracker,
};
pub use release::{
    ReleaseAuthority, ReleaseAuthorization, ReleaseEvaluationReport, ReleasePolicy,
    ReleasePolicyError, ReleaseQuorumRequirement, ReleaseSignerBinding, ReleaseViolation,
    SignerRole, authorize_release, canonical_release_policy_bytes, digest_release_policy,
};
pub use release_assurance::{
    AssuredReleasePromotion, ReleaseAssuranceError, ReleaseAssuranceEvidence,
    ReleaseAssurancePolicy, authorize_release_assurance, build_release_assurance_evidence,
    digest_release_assurance,
};
pub use release_certification::{
    CertifiedReleaseCandidate, ReleaseCandidateEvidence, ReleaseCandidateSigner,
    ReleaseCandidateVerifier, ReleaseCertificationError, ReleaseCertificationPolicy,
    ReleaseCertificationViolation, ReleaseEvidenceMismatch, ReleaseEvidenceVerificationReport,
    SignedReleaseCandidate, digest_release_candidate, sign_release_candidate,
    verify_release_candidate, verify_release_candidate_evidence,
};
pub use release_lineage::{
    ReleaseLineage, ReleaseLineageAction, ReleaseLineageError, ReleaseLineageEvent,
    digest_release_lineage,
};
pub use release_promotion::{
    AuthorizedReleasePromotion, ReleasePromotionError, ReleasePromotionEvidence,
    ReleasePromotionPolicy, authorize_release_promotion, build_release_promotion_evidence,
    digest_release_promotion,
};
pub use release_rollback::{
    AuthorizedReleaseRollback, ReleaseRollbackError, ReleaseRollbackEvidence,
    ReleaseRollbackPolicy, authorize_release_rollback, build_release_rollback_evidence,
    digest_release_rollback,
};
pub use repair::{
    MeshRepairError, MeshRepairPolicy, MeshRepairReport, MeshRepairResult, repair_mesh,
};
pub use replay::{
    AlgorithmVersion, FabricationReplayContract, GovernedFabricationReplayContract,
    GovernedReplayMismatch, GovernedReplayVerificationReport, ReplayContractError,
    ReplayEnvironment, ReplayMismatch, ReplayVerificationReport, build_governed_replay_contract,
    build_replay_contract, digest_governed_replay_contract, digest_replay_contract,
    verify_governed_replay_contract, verify_replay_contract,
};
pub use resilience_state::{
    ReleaseResilienceState, ReleaseResilienceStateError, digest_release_resilience_state,
    verify_release_resilience_successor,
};
pub use rollback_replay::{
    RollbackReplayContract, RollbackReplayError, RollbackReplayMismatch,
    RollbackReplayVerificationReport, build_rollback_replay_contract,
    digest_rollback_replay_contract, verify_rollback_replay_contract,
};
pub use rollback_transparency::{
    RollbackTransparencyError, VerifiedRollbackTransparency, publish_release_rollback,
    verify_release_rollback_transparency,
};
pub use rollout::{
    AuthorizedRolloutAdvance, RolloutAdvance, RolloutError, RolloutObservation, RolloutPhase,
    RolloutPlan, RolloutTracker, RolloutTrackingError, authorize_rollout_advance,
    digest_rollout_advance, digest_rollout_observation, digest_rollout_plan,
};
pub use rollout_revocation::{
    AuthorizedRolloutRevocation, RolloutRevocationError, RolloutRevocationEvidence,
    RolloutRevocationScope, authorize_rollout_revocation, build_rollout_revocation_evidence,
    digest_rollout_revocation,
};
pub use rollout_revocation_tracker::{
    RolloutRevocationRecord, RolloutRevocationTracker, RolloutRevocationTrackingError,
    digest_rollout_revocation_tracker,
};
pub use session::{MachineSessionLease, MachineSessionTracker, SessionTrackingError};
pub use signer_compromise::{
    AuthorizedSignerCompromise, CompromisedSignerIdentity, SignerCompromiseError,
    SignerCompromiseNotice, SignerCompromisePolicy, authorize_signer_compromise,
    build_signer_compromise_notice, digest_signer_compromise_notice,
};
pub use signer_compromise_tracker::{
    CompromiseContainmentRecord, SignerCompromiseTracker, SignerCompromiseTrackingError,
    digest_signer_compromise_tracker,
};
pub use simulator::{ForceHV, PhysicsBackend, SimState};
pub use slicer::{
    Contour, Point2, Segment2, SliceConfig, SliceError, SliceLayer, slice_fabrication_ready,
    slice_manufacturing_ready, slice_mesh, slice_mesh_at_z, try_slice_mesh, try_slice_mesh_at_z,
};
pub use step_import::{
    StepFile, StepParseError, StepParseLimits, parse_step_subset, parse_step_subset_with_limits,
};
pub use submission::{
    AuthorizedPrintJob, GovernedAuthorizedPrintJob, GovernedPrintAuthorizationError,
    GovernedSubmittedJobReceipt, PrintAuthorizationError, SubmissionError, SubmittedJobReceipt,
    authorize_governed_print_job, authorize_print_job, submit_authorized_job,
    submit_governed_authorized_job,
};
pub use submission_ledger::{
    SubmissionDisposition, SubmissionIntent, SubmissionLedger, SubmissionLedgerAction,
    SubmissionLedgerError, SubmissionLedgerEvent, SubmissionLedgerVerificationReport,
    SubmissionLedgerViolation,
};
pub use support::{SupportColumn, SupportConfig, SupportPlan, plan_column_supports};
pub use telemetry_tracker::{MachineTelemetryTracker, TelemetryTrackingError};
pub use thought::GeometricThought;
pub use threshold::{
    SignedThresholdApproval, ThresholdApproval, ThresholdApprovalSigner, ThresholdApprovalVerifier,
    ThresholdCeremonyError, ThresholdCeremonyPolicy, ThresholdCeremonyViolation,
    VerifiedThresholdCeremony, canonical_threshold_approval_bytes, digest_threshold_approval,
    sign_threshold_approval, verify_threshold_ceremony, verify_threshold_ceremony_with_containment,
};
pub use toolpath::{
    GCodeCommand, GCodeGenerationError, GCodeProgram, ToolpathConfig, generate_gcode,
    try_generate_gcode,
};
pub use transparency::{
    InclusionProofNode, ProofSide, TransparencyEntry, TransparencyError,
    TransparencyInclusionProof, TransparencyLog, digest_transparency_entry,
    digest_transparency_log, verify_transparency_inclusion,
};
pub use transparency_checkpoint::{
    SignedTransparencyCheckpoint, TransparencyCheckpoint, TransparencyCheckpointError,
    TransparencyCheckpointSigner, TransparencyCheckpointTracker,
    TransparencyCheckpointTrackingError, TransparencyCheckpointVerifier,
    TransparencyCheckpointViolation, VerifiedTransparencyCheckpoint,
    digest_transparency_checkpoint, sign_transparency_checkpoint, verify_transparency_checkpoint,
};
pub use transparency_witness::{
    SignedTransparencyWitness, TransparencyWitnessError, TransparencyWitnessPolicy,
    TransparencyWitnessSigner, TransparencyWitnessStatement, TransparencyWitnessVerifier,
    TransparencyWitnessViolation, VerifiedTransparencyWitnessQuorum,
    digest_transparency_witness_statement, sign_transparency_witness,
    verify_transparency_witness_quorum,
};
pub use transparency_witness_tracker::{
    TransparencyWitnessTracker, TransparencyWitnessTrackingError, WitnessObservationState,
    digest_transparency_witness_tracker,
};
pub use trust::{
    KeyEligibility, KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot,
    TrustSnapshotError, TrustSnapshotTracker, TrustSnapshotTrackingError,
    canonical_trust_snapshot_bytes, digest_trust_snapshot,
};
pub use units::{CANONICAL_GEOMETRY_UNIT, Meters, Millimeters, Newtons, Pascals, UnitError};
pub use upgrade_bundle::{
    UpgradeBundleError, UpgradeBundleLimits, UpgradeEvidenceBundle, build_upgrade_evidence_bundle,
    decode_upgrade_evidence_bundle, digest_upgrade_evidence_bundle, encode_upgrade_evidence_bundle,
    verify_upgrade_evidence_bundle,
};
pub use upgrade_finalization::{
    AuthorizedUpgradeFinalization, UpgradeFinalizationError, UpgradeFinalizationEvidence,
    UpgradeFinalizationPolicy, authorize_upgrade_finalization, build_upgrade_finalization_evidence,
    digest_upgrade_finalization_evidence,
};
pub use upgrade_handoff::{
    AuthorizedUpgradeHandoff, UpgradeEndpoint, UpgradeHandoffError, UpgradeHandoffPlan,
    UpgradeHandoffPolicy, authorize_upgrade_handoff, digest_upgrade_endpoint,
    digest_upgrade_handoff_plan,
};
pub use upgrade_replay::{
    UpgradeReplayContract, UpgradeReplayError, UpgradeReplayMismatch,
    UpgradeReplayVerificationReport, build_upgrade_replay_contract, digest_policy_migration_set,
    digest_upgrade_replay_contract, verify_upgrade_replay_contract,
};
pub use upgrade_state::{
    FabricationUpgradeState, UpgradeEvidenceDigests, UpgradeStateError,
    digest_upgrade_evidence_set, digest_upgrade_state, verify_upgrade_state_successor,
};
pub use upgrade_tracker::{
    UpgradeHandoffTracker, UpgradeRecord, UpgradeStage, UpgradeTrackingError,
    digest_upgrade_tracker, empty_upgrade_record_digest,
};
pub use validate::{
    EdgeTopologyReport, FabricationReadyMesh, ValidationReport, analyze_edge_topology,
    compute_signed_volume, find_duplicate_triangles, validate_mesh,
};
pub use witness_gossip::{
    SignedWitnessGossip, VerifiedWitnessEquivocation, VerifiedWitnessGossip,
    WitnessEquivocationProof, WitnessGossipError, WitnessGossipPolicy, WitnessGossipSigner,
    WitnessGossipStatement, WitnessGossipVerifier, digest_witness_equivocation_proof,
    digest_witness_gossip_statement, prove_witness_equivocation, sign_witness_gossip,
    verify_witness_gossip,
};
pub use witness_gossip_tracker::{
    WitnessEquivocationRecord, WitnessGossipObservationRecord, WitnessGossipTracker,
    WitnessGossipTrackingError, digest_witness_gossip_tracker,
};

pub use cincinnati_live::{
    AnomalyAlert, AnomalyType, ChannelStats, CincinnatiMonitor, CincinnatiMonitorConfig,
    SensorReading,
};
