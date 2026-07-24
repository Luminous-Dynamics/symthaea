// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! File-oriented tooling for the frozen cognition study.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Write};
use std::path::Path;
use symthaea_muse::analysis_crosscheck::{
    AnalysisAgreementTolerance, AnalysisCrosscheckReport, NormalizedPrimaryAnalysis,
    crosscheck_primary_analyses, seal_normalized_primary_analysis, validate_analysis_crosscheck,
};
use symthaea_muse::blinded_study::{
    ArmArtifactBinding, BlindedSchedule, BlindingCodebook, build_blinded_schedule,
    validate_blinded_schedule,
};
use symthaea_muse::cohort_registry::{
    PilotCohortRegistry, seal_pilot_cohort_registry, validate_pilot_cohort_registry,
};
use symthaea_muse::confirmatory_amendment_control::{
    ConfirmatoryAmendmentLedger, ConfirmatoryAuthoritySnapshot, seal_confirmatory_amendment_ledger,
    seal_confirmatory_authority_snapshot, validate_confirmatory_amendment_ledger,
};
use symthaea_muse::confirmatory_analysis::{ConfirmatoryAnalysisPlan, analyze_confirmatory_study};
use symthaea_muse::confirmatory_analysis_execution::{
    ConfirmatoryAnalysisCommandEvidence, ConfirmatoryAnalysisDeviation,
    ConfirmatoryAnalysisExecutionRecord, build_confirmatory_analysis_execution,
    validate_confirmatory_analysis_execution,
};
use symthaea_muse::confirmatory_cohort_registry::{
    ConfirmatoryCohortRegistry, seal_confirmatory_cohort_registry,
    validate_confirmatory_cohort_registry,
};
use symthaea_muse::confirmatory_collection_close::{
    ConfirmatoryCloseSignoff, ConfirmatoryCollectionCloseReason,
    ConfirmatoryCollectionCloseReceipt, build_confirmatory_collection_close,
    validate_confirmatory_collection_close,
};
use symthaea_muse::confirmatory_collection_monitor::{
    ConfirmatoryCollectionSnapshot, ConfirmatorySessionStatus,
    build_confirmatory_collection_snapshot, validate_confirmatory_collection_snapshot,
};
use symthaea_muse::confirmatory_collection_protocol::{
    ConfirmatoryCollectionProtocol, seal_confirmatory_collection_protocol,
    validate_confirmatory_collection_protocol,
};
use symthaea_muse::confirmatory_final_release::{
    ConfirmatoryFinalReleaseBundle, build_confirmatory_final_release,
    validate_confirmatory_final_release,
};
use symthaea_muse::confirmatory_publication::{
    ConfirmatoryPublicationRecord, seal_confirmatory_publication, validate_confirmatory_publication,
};
use symthaea_muse::confirmatory_readiness::{
    ConfirmatoryDryRunEvidence, ConfirmatoryReadinessReport, HumanStudyGovernanceEvidence,
    IndependentReproductionReadiness, WorkspaceValidationEvidence,
    build_confirmatory_readiness_report, seal_confirmatory_dry_run, seal_human_study_governance,
    seal_independent_reproduction_readiness, seal_workspace_validation_evidence,
    validate_confirmatory_readiness_report,
};
use symthaea_muse::confirmatory_readiness_release::{
    ConfirmatoryReadinessReleaseBundle, build_confirmatory_readiness_release,
    validate_confirmatory_readiness_release,
};
use symthaea_muse::confirmatory_unblinding::{
    ConfirmatoryUnblindingAuthorization, ConfirmatoryUnblindingReceipt,
    build_confirmatory_unblinding_receipt, validate_confirmatory_unblinding_receipt,
};
use symthaea_muse::evidence_digest::{canonical_json_sha256, decode_hex_32};
use symthaea_muse::experiment_manifest::FrozenStudyManifest;
use symthaea_muse::external_review_completion::{
    ExternalReviewCompletionEvidence, build_external_review_completion,
    validate_external_review_completion,
};
use symthaea_muse::external_review_package::{
    ExternalReviewPackage, ReviewEvidenceIndex, build_external_review_package,
    seal_review_evidence_index, validate_external_review_package,
};
use symthaea_muse::external_review_protocol::{
    FrozenExternalReviewProtocol, seal_external_review_protocol, validate_external_review_protocol,
};
use symthaea_muse::external_review_resolution::{
    ExternalReviewResolutionLedger, seal_external_review_resolution_ledger,
    validate_external_review_resolution_ledger,
};
use symthaea_muse::external_review_response::{
    ExternalReviewResponse, seal_external_review_response, validate_external_review_response,
};
use symthaea_muse::family_clustered_analysis::{
    FamilyClusteredAnalysisPlan, analyze_family_clustered,
};
use symthaea_muse::methodology_plan::FrozenMethodologyPlan;
use symthaea_muse::participant_evidence::{
    ParticipantEvidenceEnvelope, compile_participant_dataset, seal_participant_evidence,
};
use symthaea_muse::participant_schedule::{
    ParticipantCohortSpec, ParticipantScheduleBook, build_participant_schedule,
    validate_participant_schedule,
};
use symthaea_muse::pilot_collection::{
    PilotCollectionEnvelope, PilotSessionSubmission, seal_pilot_collection,
    validate_pilot_collection,
};
use symthaea_muse::pilot_monitoring::{
    PilotOperationalSnapshot, build_pilot_operational_snapshot, validate_pilot_operational_snapshot,
};
use symthaea_muse::pilot_protocol::{
    FrozenPilotProtocol, PilotAmendmentLedger, seal_pilot_amendment_ledger,
    validate_pilot_amendment_ledger,
};
use symthaea_muse::pilot_report::{
    ConfirmatorySampleSizeRecommendation, PilotReviewReport, seal_pilot_review_report,
    seal_sample_size_recommendation, validate_pilot_review_report,
};
use symthaea_muse::pilot_schedule::{
    PilotCohortSpec, PilotParticipantScheduleBook, PilotScheduleAudit,
    build_pilot_participant_schedule, validate_pilot_participant_schedule,
};
use symthaea_muse::policy_budget_evidence::{
    PolicyBudgetEvidenceBundle, seal_policy_budget_evidence, validate_policy_budget_evidence,
};
use symthaea_muse::post_publication_audit::{
    PostPublicationAuditLedger, PostPublicationClaimChange, PostPublicationEventKind,
    append_post_publication_event, new_post_publication_audit, validate_post_publication_audit,
};
use symthaea_muse::ranked_preference_analysis::{
    RankedPreferenceAnalysisPlan, analyze_ranked_preference,
};
use symthaea_muse::replication_execution::{
    ReplicationSiteExecutionRecord, seal_replication_execution, validate_replication_execution,
};
use symthaea_muse::replication_orchestration::{
    ReplicationLifecyclePhase, ReplicationOrchestrationLog, append_replication_transition,
    new_replication_orchestration, validate_replication_orchestration,
};
use symthaea_muse::replication_package::{
    ReplicationPackageEntry, ReplicationSitePackage, build_replication_package,
    validate_replication_package,
};
use symthaea_muse::replication_protocol::{
    FrozenReplicationProtocol, seal_replication_protocol, validate_replication_protocol,
};
use symthaea_muse::replication_site_registry::{
    ReplicationSiteRegistry, seal_replication_site_registry, validate_replication_site_registry,
};
use symthaea_muse::replication_synthesis::{
    PublishedSourcePrimaryResult, ReplicationSynthesisRecord, synthesize_replications,
    validate_replication_synthesis,
};
use symthaea_muse::reproducibility_attestation::{
    IndependentReproductionAttestation, seal_reproduction_attestation,
    validate_reproduction_attestation,
};
use symthaea_muse::research_archive::{
    ResearchArchiveManifest, ResearchArchivePlan, seal_research_archive, validate_research_archive,
};
use symthaea_muse::research_release_promotion::{
    ResearchReleasePromotionRecord, ResearchReleasePromotionRequest,
    evaluate_research_release_promotion, validate_research_release_promotion,
};
use symthaea_muse::research_revision_governance::{
    ResearchRevisionProposal, seal_research_revision_proposal, validate_research_revision_proposal,
};
use symthaea_muse::stewardship_governance::{
    ResearchStewardshipCharter, seal_stewardship_charter, validate_stewardship_charter,
};
use symthaea_muse::stewardship_release::{
    StewardshipReleaseBundle, build_stewardship_release, validate_stewardship_release,
};
use symthaea_muse::structural_evidence::{
    StructuralEvidenceBundle, compile_structural_outcomes, seal_structural_evidence,
};
use symthaea_muse::study_artifact::{
    ArtifactProductionPlan, StudyArtifactBundle, seal_study_artifacts,
    validate_study_artifact_bundle,
};
use symthaea_muse::study_collection::{StudyCollectionDraft, seal_runner_collection};
use symthaea_muse::study_evidence::{
    CompiledStudyDataset, RawStudyEvidence, compile_study_dataset, seal_raw_evidence,
};
use symthaea_muse::study_operations_release::{
    StudyOperationsReleaseBundle, build_study_operations_release, validate_study_operations_release,
};
use symthaea_muse::study_orchestration::{
    StudyAuthorityBinding, StudyLifecyclePhase, StudyOrchestrationLog, append_study_transition,
    new_study_orchestration, upgrade_legacy_study_orchestration, validate_study_orchestration,
};
use symthaea_muse::study_release::{
    StudyReleaseBundle, StudyReleasePlan, seal_study_release, validate_study_release,
};
use symthaea_muse::study_runner::{
    RunnerProtocol, StudyRunnerPackage, StudySessionEvent, StudySessionLog, append_session_event,
    build_pilot_runner_package, build_runner_package, compile_listener_block, new_session_log,
    validate_pilot_runner_package, validate_runner_package, validate_session_log,
};
use symthaea_muse::temporal_confirmatory::{
    FrozenTemporalRecord, TemporalConfirmatoryPlan, analyze_temporal_confirmatory,
};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let Some(command) = args.get(1).map(String::as_str) else {
        print_usage();
        return Err(invalid_input("missing command"));
    };
    match command {
        "digest-json" => {
            require_len(&args, 3)?;
            let value: serde_json::Value = read_json(&args[2])?;
            println!("{}", canonical_json_sha256(&value)?);
        }
        "validate-manifest" => {
            require_len(&args, 3)?;
            let manifest: FrozenStudyManifest = read_json(&args[2])?;
            let issues = manifest.validate();
            write_stdout_json(&issues)?;
            fail_if_issues(!issues.is_empty(), "manifest validation failed")?;
        }
        "build-schedule" => {
            require_len(&args, 7)?;
            let manifest: FrozenStudyManifest = read_json(&args[2])?;
            let artifacts: Vec<ArmArtifactBinding> = read_json(&args[3])?;
            let secret_hex = std::fs::read_to_string(&args[4])?;
            let secret_key = decode_hex_32(secret_hex.trim()).ok_or_else(|| {
                invalid_input("secret key file must contain exactly 64 hex characters")
            })?;
            let (schedule, codebook) = build_blinded_schedule(&manifest, &artifacts, secret_key)
                .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[5], &schedule)?;
            write_json(&args[6], &codebook)?;
        }
        "validate-schedule" => {
            require_len(&args, 5)?;
            let manifest: FrozenStudyManifest = read_json(&args[2])?;
            let schedule: BlindedSchedule = read_json(&args[3])?;
            let codebook: BlindingCodebook = read_json(&args[4])?;
            let issues = validate_blinded_schedule(&manifest, &schedule, Some(&codebook));
            write_stdout_json(&issues)?;
            fail_if_issues(!issues.is_empty(), "schedule validation failed")?;
        }
        "seal-artifact-bundle" => {
            require_len(&args, 8)?;
            let manifest: FrozenStudyManifest = read_json(&args[2])?;
            let methodology: FrozenMethodologyPlan = read_json(&args[3])?;
            let schedule: BlindedSchedule = read_json(&args[4])?;
            let plan: ArtifactProductionPlan = read_json(&args[5])?;
            let bundle = seal_study_artifacts(
                &manifest,
                &methodology,
                &schedule,
                &plan,
                Path::new(&args[6]),
            )
            .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[7], &bundle)?;
        }
        "validate-artifact-bundle" => {
            require_len(&args, 9)?;
            let manifest: FrozenStudyManifest = read_json(&args[2])?;
            let methodology: FrozenMethodologyPlan = read_json(&args[3])?;
            let schedule: BlindedSchedule = read_json(&args[4])?;
            let plan: ArtifactProductionPlan = read_json(&args[5])?;
            let bundle: StudyArtifactBundle = read_json(&args[6])?;
            let issues = validate_study_artifact_bundle(
                &manifest,
                &methodology,
                &schedule,
                &plan,
                &bundle,
                Path::new(&args[7]),
            );
            write_json(&args[8], &issues)?;
            fail_if_issues(!issues.is_empty(), "study artifact validation failed")?;
        }
        "build-runner-package" => {
            require_len(&args, 8)?;
            let schedule: BlindedSchedule = read_json(&args[2])?;
            let participant_schedule: ParticipantScheduleBook = read_json(&args[3])?;
            let artifacts: StudyArtifactBundle = read_json(&args[4])?;
            let protocol: RunnerProtocol = read_json(&args[6])?;
            let package = build_runner_package(
                &schedule,
                &participant_schedule,
                &artifacts,
                &args[5],
                protocol,
            )
            .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[7], &package)?;
        }
        "validate-runner-package" => {
            require_len(&args, 7)?;
            let schedule: BlindedSchedule = read_json(&args[2])?;
            let participant_schedule: ParticipantScheduleBook = read_json(&args[3])?;
            let artifacts: StudyArtifactBundle = read_json(&args[4])?;
            let package: StudyRunnerPackage = read_json(&args[5])?;
            let issues =
                validate_runner_package(&package, &schedule, &participant_schedule, &artifacts);
            write_json(&args[6], &issues)?;
            fail_if_issues(!issues.is_empty(), "runner package validation failed")?;
        }
        "init-session" => {
            require_len(&args, 4)?;
            let package: StudyRunnerPackage = read_json(&args[2])?;
            write_json(&args[3], &new_session_log(&package))?;
        }
        "append-session-event" => {
            require_len(&args, 8)?;
            let package: StudyRunnerPackage = read_json(&args[2])?;
            let mut log: StudySessionLog = read_json(&args[3])?;
            let event: StudySessionEvent = read_json(&args[4])?;
            let server_received_unix_ms = args[5]
                .parse::<u64>()
                .map_err(|_| invalid_input("SERVER_UNIX_MS must be an unsigned integer"))?;
            let client_elapsed_ms = args[6]
                .parse::<u64>()
                .map_err(|_| invalid_input("CLIENT_ELAPSED_MS must be an unsigned integer"))?;
            append_session_event(
                &package,
                &mut log,
                server_received_unix_ms,
                client_elapsed_ms,
                event,
            )
            .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[7], &log)?;
        }
        "validate-session" => {
            require_len(&args, 5)?;
            let package: StudyRunnerPackage = read_json(&args[2])?;
            let log: StudySessionLog = read_json(&args[3])?;
            let issues = validate_session_log(&package, &log, true);
            write_json(&args[4], &issues)?;
            fail_if_issues(!issues.is_empty(), "study session validation failed")?;
        }
        "compile-session" => {
            require_len(&args, 5)?;
            let package: StudyRunnerPackage = read_json(&args[2])?;
            let log: StudySessionLog = read_json(&args[3])?;
            let block = compile_listener_block(&package, &log)
                .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[4], &block)?;
        }
        "seal-runner-collection" => {
            require_len(&args, 8)?;
            let manifest: FrozenStudyManifest = read_json(&args[2])?;
            let schedule: BlindedSchedule = read_json(&args[3])?;
            let participant_schedule: ParticipantScheduleBook = read_json(&args[4])?;
            let artifacts: StudyArtifactBundle = read_json(&args[5])?;
            let draft: StudyCollectionDraft = read_json(&args[6])?;
            let envelope = seal_runner_collection(
                &manifest,
                &schedule,
                &participant_schedule,
                &artifacts,
                &draft,
            )
            .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[7], &envelope)?;
        }
        "seal-release" => {
            require_len(&args, 5)?;
            let plan: StudyReleasePlan = read_json(&args[2])?;
            let bundle = seal_study_release(&plan, Path::new(&args[3]))
                .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[4], &bundle)?;
        }
        "validate-release" => {
            require_len(&args, 6)?;
            let plan: StudyReleasePlan = read_json(&args[2])?;
            let bundle: StudyReleaseBundle = read_json(&args[3])?;
            let issues = validate_study_release(&plan, &bundle, Path::new(&args[4]));
            write_json(&args[5], &issues)?;
            fail_if_issues(!issues.is_empty(), "study release validation failed")?;
        }
        "seal-evidence" => {
            require_len(&args, 4)?;
            let mut evidence: RawStudyEvidence = read_json(&args[2])?;
            seal_raw_evidence(&mut evidence)?;
            write_json(&args[3], &evidence)?;
        }
        "compile-evidence" => {
            require_len(&args, 7)?;
            let manifest: FrozenStudyManifest = read_json(&args[2])?;
            let schedule: BlindedSchedule = read_json(&args[3])?;
            let codebook: BlindingCodebook = read_json(&args[4])?;
            let evidence: RawStudyEvidence = read_json(&args[5])?;
            let dataset = compile_study_dataset(&manifest, &schedule, &codebook, &evidence)
                .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[6], &dataset)?;
        }
        "analyze" => {
            require_len(&args, 6)?;
            let manifest: FrozenStudyManifest = read_json(&args[2])?;
            let dataset: CompiledStudyDataset = read_json(&args[3])?;
            let plan: ConfirmatoryAnalysisPlan = read_json(&args[4])?;
            let report = analyze_confirmatory_study(&manifest, &dataset, &plan);
            write_json(&args[5], &report)?;
            fail_if_issues(
                !report.issues.is_empty(),
                "confirmatory analysis validation failed",
            )?;
        }
        "validate-methodology" => {
            require_len(&args, 4)?;
            let manifest: FrozenStudyManifest = read_json(&args[2])?;
            let methodology: FrozenMethodologyPlan = read_json(&args[3])?;
            let issues = methodology.validate(&manifest);
            write_stdout_json(&issues)?;
            fail_if_issues(!issues.is_empty(), "methodology validation failed")?;
        }
        "build-participant-schedule" => {
            require_len(&args, 9)?;
            let manifest: FrozenStudyManifest = read_json(&args[2])?;
            let schedule: BlindedSchedule = read_json(&args[3])?;
            let codebook: BlindingCodebook = read_json(&args[4])?;
            let cohort: ParticipantCohortSpec = read_json(&args[5])?;
            let secret_hex = std::fs::read_to_string(&args[6])?;
            let secret_key = decode_hex_32(secret_hex.trim()).ok_or_else(|| {
                invalid_input("secret key file must contain exactly 64 hex characters")
            })?;
            let (participant_schedule, audit) =
                build_participant_schedule(&manifest, &schedule, &codebook, &cohort, secret_key)
                    .map_err(|issues| {
                        invalid_input(serde_json::to_string_pretty(&issues).unwrap())
                    })?;
            write_json(&args[7], &participant_schedule)?;
            write_json(&args[8], &audit)?;
        }
        "validate-participant-schedule" => {
            require_len(&args, 6)?;
            let manifest: FrozenStudyManifest = read_json(&args[2])?;
            let schedule: BlindedSchedule = read_json(&args[3])?;
            let cohort: ParticipantCohortSpec = read_json(&args[4])?;
            let participant_schedule: ParticipantScheduleBook = read_json(&args[5])?;
            let issues = validate_participant_schedule(
                &manifest,
                &schedule,
                &cohort,
                &participant_schedule,
                None,
            );
            write_stdout_json(&issues)?;
            fail_if_issues(!issues.is_empty(), "participant schedule validation failed")?;
        }
        "seal-participant-evidence" => {
            require_len(&args, 7)?;
            let manifest: FrozenStudyManifest = read_json(&args[2])?;
            let schedule: BlindedSchedule = read_json(&args[3])?;
            let participant_schedule: ParticipantScheduleBook = read_json(&args[4])?;
            let evidence: RawStudyEvidence = read_json(&args[5])?;
            let envelope =
                seal_participant_evidence(&manifest, &schedule, &participant_schedule, evidence)?;
            write_json(&args[6], &envelope)?;
        }
        "compile-participant-evidence" => {
            require_len(&args, 9)?;
            let manifest: FrozenStudyManifest = read_json(&args[2])?;
            let schedule: BlindedSchedule = read_json(&args[3])?;
            let codebook: BlindingCodebook = read_json(&args[4])?;
            let cohort: ParticipantCohortSpec = read_json(&args[5])?;
            let participant_schedule: ParticipantScheduleBook = read_json(&args[6])?;
            let envelope: ParticipantEvidenceEnvelope = read_json(&args[7])?;
            let dataset = compile_participant_dataset(
                &manifest,
                &schedule,
                &codebook,
                &cohort,
                &participant_schedule,
                &envelope,
            )
            .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[8], &dataset)?;
        }
        "seal-structural-evidence" => {
            require_len(&args, 4)?;
            let mut bundle: StructuralEvidenceBundle = read_json(&args[2])?;
            seal_structural_evidence(&mut bundle)?;
            write_json(&args[3], &bundle)?;
        }
        "compile-structural-evidence" => {
            require_len(&args, 7)?;
            let manifest: FrozenStudyManifest = read_json(&args[2])?;
            let schedule: BlindedSchedule = read_json(&args[3])?;
            let methodology: FrozenMethodologyPlan = read_json(&args[4])?;
            let bundle: StructuralEvidenceBundle = read_json(&args[5])?;
            let outcomes = compile_structural_outcomes(&manifest, &schedule, &methodology, &bundle)
                .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[6], &outcomes)?;
        }
        "seal-policy-budget" => {
            require_len(&args, 4)?;
            let mut bundle: PolicyBudgetEvidenceBundle = read_json(&args[2])?;
            seal_policy_budget_evidence(&mut bundle)?;
            write_json(&args[3], &bundle)?;
        }
        "validate-policy-budget" => {
            require_len(&args, 5)?;
            let manifest: FrozenStudyManifest = read_json(&args[2])?;
            let methodology: FrozenMethodologyPlan = read_json(&args[3])?;
            let bundle: PolicyBudgetEvidenceBundle = read_json(&args[4])?;
            let issues = validate_policy_budget_evidence(&manifest, &methodology, &bundle);
            write_stdout_json(&issues)?;
            fail_if_issues(!issues.is_empty(), "policy budget validation failed")?;
        }
        "analyze-family-clustered" => {
            require_len(&args, 7)?;
            let manifest: FrozenStudyManifest = read_json(&args[2])?;
            let methodology: FrozenMethodologyPlan = read_json(&args[3])?;
            let dataset: CompiledStudyDataset = read_json(&args[4])?;
            let plan: FamilyClusteredAnalysisPlan = read_json(&args[5])?;
            let report = analyze_family_clustered(&manifest, &methodology, &dataset, &plan);
            write_json(&args[6], &report)?;
            fail_if_issues(
                !report.issues.is_empty(),
                "family-clustered analysis failed",
            )?;
        }
        "analyze-ranked-preference" => {
            require_len(&args, 11)?;
            let manifest: FrozenStudyManifest = read_json(&args[2])?;
            let methodology: FrozenMethodologyPlan = read_json(&args[3])?;
            let schedule: BlindedSchedule = read_json(&args[4])?;
            let codebook: BlindingCodebook = read_json(&args[5])?;
            let cohort: ParticipantCohortSpec = read_json(&args[6])?;
            let participant_schedule: ParticipantScheduleBook = read_json(&args[7])?;
            let envelope: ParticipantEvidenceEnvelope = read_json(&args[8])?;
            let plan: RankedPreferenceAnalysisPlan = read_json(&args[9])?;
            let report = analyze_ranked_preference(
                &manifest,
                &methodology,
                &schedule,
                &codebook,
                &cohort,
                &participant_schedule,
                &envelope,
                &plan,
            );
            write_json(&args[10], &report)?;
            fail_if_issues(
                !report.issues.is_empty(),
                "ranked preference analysis failed",
            )?;
        }
        "validate-pilot-protocol" => {
            require_len(&args, 5)?;
            let manifest: FrozenStudyManifest = read_json(&args[2])?;
            let methodology: FrozenMethodologyPlan = read_json(&args[3])?;
            let protocol: FrozenPilotProtocol = read_json(&args[4])?;
            let issues = protocol.validate(&manifest, &methodology);
            write_stdout_json(&issues)?;
            fail_if_issues(!issues.is_empty(), "pilot protocol validation failed")?;
        }
        "build-pilot-schedule" => {
            require_len(&args, 11)?;
            let manifest: FrozenStudyManifest = read_json(&args[2])?;
            let methodology: FrozenMethodologyPlan = read_json(&args[3])?;
            let schedule: BlindedSchedule = read_json(&args[4])?;
            let codebook: BlindingCodebook = read_json(&args[5])?;
            let protocol: FrozenPilotProtocol = read_json(&args[6])?;
            let cohort: PilotCohortSpec = read_json(&args[7])?;
            let protocol_issues = protocol.validate(&manifest, &methodology);
            fail_if_issues(
                !protocol_issues.is_empty(),
                "pilot protocol validation failed",
            )?;
            let secret_hex = std::fs::read_to_string(&args[8])?;
            let secret_key = decode_hex_32(secret_hex.trim()).ok_or_else(|| {
                invalid_input("pilot secret key file must contain exactly 64 hex characters")
            })?;
            let (pilot_schedule, audit) = build_pilot_participant_schedule(
                &manifest, &schedule, &codebook, &protocol, &cohort, secret_key,
            )
            .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[9], &pilot_schedule)?;
            write_json(&args[10], &audit)?;
        }
        "validate-pilot-schedule" => {
            require_len(&args, 11)?;
            let manifest: FrozenStudyManifest = read_json(&args[2])?;
            let methodology: FrozenMethodologyPlan = read_json(&args[3])?;
            let schedule: BlindedSchedule = read_json(&args[4])?;
            let codebook: BlindingCodebook = read_json(&args[5])?;
            let protocol: FrozenPilotProtocol = read_json(&args[6])?;
            let cohort: PilotCohortSpec = read_json(&args[7])?;
            let pilot_schedule: PilotParticipantScheduleBook = read_json(&args[8])?;
            let audit: PilotScheduleAudit = read_json(&args[9])?;
            let mut issues = protocol
                .validate(&manifest, &methodology)
                .into_iter()
                .map(|issue| format!("protocol: {issue:?}"))
                .collect::<Vec<_>>();
            issues.extend(
                validate_pilot_participant_schedule(
                    &manifest,
                    &schedule,
                    &codebook,
                    &protocol,
                    &cohort,
                    &pilot_schedule,
                    Some(&audit),
                )
                .into_iter()
                .map(|issue| format!("schedule: {issue:?}")),
            );
            write_json(&args[10], &issues)?;
            fail_if_issues(!issues.is_empty(), "pilot schedule validation failed")?;
        }
        "build-pilot-runner-package" => {
            require_len(&args, 8)?;
            let schedule: BlindedSchedule = read_json(&args[2])?;
            let pilot_schedule: PilotParticipantScheduleBook = read_json(&args[3])?;
            let artifacts: StudyArtifactBundle = read_json(&args[4])?;
            let protocol: RunnerProtocol = read_json(&args[6])?;
            let package = build_pilot_runner_package(
                &schedule,
                &pilot_schedule,
                &artifacts,
                &args[5],
                protocol,
            )
            .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[7], &package)?;
        }
        "validate-pilot-runner-package" => {
            require_len(&args, 7)?;
            let schedule: BlindedSchedule = read_json(&args[2])?;
            let pilot_schedule: PilotParticipantScheduleBook = read_json(&args[3])?;
            let artifacts: StudyArtifactBundle = read_json(&args[4])?;
            let package: StudyRunnerPackage = read_json(&args[5])?;
            let issues =
                validate_pilot_runner_package(&package, &schedule, &pilot_schedule, &artifacts);
            write_json(&args[6], &issues)?;
            fail_if_issues(!issues.is_empty(), "pilot runner package validation failed")?;
        }
        "seal-pilot-cohort-registry" => {
            require_len(&args, 4)?;
            let mut registry: PilotCohortRegistry = read_json(&args[2])?;
            seal_pilot_cohort_registry(&mut registry)?;
            write_json(&args[3], &registry)?;
        }
        "validate-pilot-cohort-registry" => {
            require_len(&args, 7)?;
            let protocol: FrozenPilotProtocol = read_json(&args[2])?;
            let cohort: PilotCohortSpec = read_json(&args[3])?;
            let pilot_schedule: PilotParticipantScheduleBook = read_json(&args[4])?;
            let registry: PilotCohortRegistry = read_json(&args[5])?;
            let issues =
                validate_pilot_cohort_registry(&protocol, &cohort, &pilot_schedule, &registry);
            write_json(&args[6], &issues)?;
            fail_if_issues(
                !issues.is_empty(),
                "pilot cohort registry validation failed",
            )?;
        }
        "seal-pilot-collection" => {
            require_len(&args, 10)?;
            let protocol: FrozenPilotProtocol = read_json(&args[2])?;
            let schedule: BlindedSchedule = read_json(&args[3])?;
            let pilot_schedule: PilotParticipantScheduleBook = read_json(&args[4])?;
            let artifacts: StudyArtifactBundle = read_json(&args[5])?;
            let registry: PilotCohortRegistry = read_json(&args[6])?;
            let sessions: Vec<PilotSessionSubmission> = read_json(&args[8])?;
            let envelope = seal_pilot_collection(
                &protocol,
                &schedule,
                &pilot_schedule,
                &artifacts,
                &registry,
                args[7].clone(),
                sessions,
            )
            .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[9], &envelope)?;
        }
        "validate-pilot-collection" => {
            require_len(&args, 9)?;
            let protocol: FrozenPilotProtocol = read_json(&args[2])?;
            let schedule: BlindedSchedule = read_json(&args[3])?;
            let pilot_schedule: PilotParticipantScheduleBook = read_json(&args[4])?;
            let artifacts: StudyArtifactBundle = read_json(&args[5])?;
            let registry: PilotCohortRegistry = read_json(&args[6])?;
            let envelope: PilotCollectionEnvelope = read_json(&args[7])?;
            let issues = validate_pilot_collection(
                &protocol,
                &schedule,
                &pilot_schedule,
                &artifacts,
                &registry,
                &envelope,
            );
            write_json(&args[8], &issues)?;
            fail_if_issues(!issues.is_empty(), "pilot collection validation failed")?;
        }
        "build-pilot-snapshot" => {
            require_len(&args, 6)?;
            let protocol: FrozenPilotProtocol = read_json(&args[2])?;
            let collection: PilotCollectionEnvelope = read_json(&args[3])?;
            let snapshot = build_pilot_operational_snapshot(
                &protocol,
                args[4].clone(),
                &collection.operational_records,
            )
            .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[5], &snapshot)?;
        }
        "validate-pilot-snapshot" => {
            require_len(&args, 5)?;
            let protocol: FrozenPilotProtocol = read_json(&args[2])?;
            let snapshot: PilotOperationalSnapshot = read_json(&args[3])?;
            let issues = validate_pilot_operational_snapshot(&protocol, &snapshot);
            write_json(&args[4], &issues)?;
            fail_if_issues(!issues.is_empty(), "pilot snapshot validation failed")?;
        }
        "seal-pilot-amendments" => {
            require_len(&args, 4)?;
            let mut ledger: PilotAmendmentLedger = read_json(&args[2])?;
            seal_pilot_amendment_ledger(&mut ledger)?;
            write_json(&args[3], &ledger)?;
        }
        "validate-pilot-amendments" => {
            require_len(&args, 4)?;
            let ledger: PilotAmendmentLedger = read_json(&args[2])?;
            let issues = validate_pilot_amendment_ledger(&ledger);
            write_json(&args[3], &issues)?;
            fail_if_issues(!issues.is_empty(), "pilot amendment validation failed")?;
        }
        "seal-sample-size-recommendation" => {
            require_len(&args, 4)?;
            let mut recommendation: ConfirmatorySampleSizeRecommendation = read_json(&args[2])?;
            seal_sample_size_recommendation(&mut recommendation)?;
            write_json(&args[3], &recommendation)?;
        }
        "seal-pilot-report" => {
            require_len(&args, 4)?;
            let mut report: PilotReviewReport = read_json(&args[2])?;
            seal_pilot_review_report(&mut report)?;
            write_json(&args[3], &report)?;
        }
        "validate-pilot-report" => {
            require_len(&args, 7)?;
            let protocol: FrozenPilotProtocol = read_json(&args[2])?;
            let ledger: PilotAmendmentLedger = read_json(&args[3])?;
            let snapshot: PilotOperationalSnapshot = read_json(&args[4])?;
            let report: PilotReviewReport = read_json(&args[5])?;
            let issues = validate_pilot_review_report(&protocol, &ledger, &snapshot, &report);
            write_json(&args[6], &issues)?;
            fail_if_issues(!issues.is_empty(), "pilot report validation failed")?;
        }
        "init-orchestration" => {
            require_len(&args, 4)?;
            write_json(&args[3], &new_study_orchestration(args[2].clone()))?;
        }
        "append-orchestration" => {
            require_len(&args, 5)?;
            let mut log: StudyOrchestrationLog = read_json(&args[2])?;
            let request: OrchestrationTransitionRequest = read_json(&args[3])?;
            append_study_transition(
                &mut log,
                request.to,
                request.recorded_at_utc,
                request.operator_id,
                request.authorization_sha256,
                request.added_authorities,
            )
            .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[4], &log)?;
        }
        "upgrade-orchestration-v11" | "upgrade-orchestration-v12" => {
            require_len(&args, 4)?;
            let mut log: StudyOrchestrationLog = read_json(&args[2])?;
            upgrade_legacy_study_orchestration(&mut log)
                .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[3], &log)?;
        }
        "validate-orchestration" => {
            require_len(&args, 4)?;
            let log: StudyOrchestrationLog = read_json(&args[2])?;
            let issues = validate_study_orchestration(&log);
            write_json(&args[3], &issues)?;
            fail_if_issues(!issues.is_empty(), "study orchestration validation failed")?;
        }
        "seal-normalized-analysis" => {
            require_len(&args, 4)?;
            let mut analysis: NormalizedPrimaryAnalysis = read_json(&args[2])?;
            seal_normalized_primary_analysis(&mut analysis)?;
            write_json(&args[3], &analysis)?;
        }
        "crosscheck-analyses" => {
            require_len(&args, 6)?;
            let rust: NormalizedPrimaryAnalysis = read_json(&args[2])?;
            let external: NormalizedPrimaryAnalysis = read_json(&args[3])?;
            let tolerance: AnalysisAgreementTolerance = read_json(&args[4])?;
            let report = crosscheck_primary_analyses(&rust, &external, tolerance)
                .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[5], &report)?;
            fail_if_issues(!report.passed, "independent analyses did not agree")?;
        }
        "validate-analysis-crosscheck" => {
            require_len(&args, 6)?;
            let rust: NormalizedPrimaryAnalysis = read_json(&args[2])?;
            let external: NormalizedPrimaryAnalysis = read_json(&args[3])?;
            let report: AnalysisCrosscheckReport = read_json(&args[4])?;
            let issues = validate_analysis_crosscheck(&rust, &external, &report);
            write_json(&args[5], &issues)?;
            fail_if_issues(
                !issues.is_empty() || !report.passed,
                "analysis crosscheck failed",
            )?;
        }
        "build-operations-release" => {
            require_len(&args, 17)?;
            let base_release: StudyReleaseBundle = read_json(&args[2])?;
            let protocol: FrozenPilotProtocol = read_json(&args[3])?;
            let pilot_schedule: PilotParticipantScheduleBook = read_json(&args[4])?;
            let registry: PilotCohortRegistry = read_json(&args[5])?;
            let collection: PilotCollectionEnvelope = read_json(&args[6])?;
            let snapshot: PilotOperationalSnapshot = read_json(&args[7])?;
            let ledger: PilotAmendmentLedger = read_json(&args[8])?;
            let report: PilotReviewReport = read_json(&args[9])?;
            let orchestration: StudyOrchestrationLog = read_json(&args[10])?;
            let crosscheck: AnalysisCrosscheckReport = read_json(&args[11])?;
            let attestation: IndependentReproductionAttestation = read_json(&args[12])?;
            let bundle = build_study_operations_release(
                &base_release,
                &protocol,
                &pilot_schedule,
                &registry,
                &collection,
                &snapshot,
                &ledger,
                &report,
                &orchestration,
                &crosscheck,
                &attestation,
                args[13].clone(),
                args[14].clone(),
                args[15].clone(),
            )?;
            write_json(&args[16], &bundle)?;
        }
        "validate-operations-release" => {
            require_len(&args, 15)?;
            let base_release: StudyReleaseBundle = read_json(&args[2])?;
            let protocol: FrozenPilotProtocol = read_json(&args[3])?;
            let pilot_schedule: PilotParticipantScheduleBook = read_json(&args[4])?;
            let registry: PilotCohortRegistry = read_json(&args[5])?;
            let collection: PilotCollectionEnvelope = read_json(&args[6])?;
            let snapshot: PilotOperationalSnapshot = read_json(&args[7])?;
            let ledger: PilotAmendmentLedger = read_json(&args[8])?;
            let report: PilotReviewReport = read_json(&args[9])?;
            let orchestration: StudyOrchestrationLog = read_json(&args[10])?;
            let crosscheck: AnalysisCrosscheckReport = read_json(&args[11])?;
            let attestation: IndependentReproductionAttestation = read_json(&args[12])?;
            let bundle: StudyOperationsReleaseBundle = read_json(&args[13])?;
            let issues = validate_study_operations_release(
                &base_release,
                &protocol,
                &pilot_schedule,
                &registry,
                &collection,
                &snapshot,
                &ledger,
                &report,
                &orchestration,
                &crosscheck,
                &attestation,
                &bundle,
            );
            write_json(&args[14], &issues)?;
            fail_if_issues(!issues.is_empty(), "operations release validation failed")?;
        }
        "seal-reproduction-attestation" => {
            require_len(&args, 4)?;
            let mut attestation: IndependentReproductionAttestation = read_json(&args[2])?;
            seal_reproduction_attestation(&mut attestation)?;
            write_json(&args[3], &attestation)?;
        }
        "validate-reproduction-attestation" => {
            require_len(&args, 6)?;
            let release: StudyReleaseBundle = read_json(&args[2])?;
            let crosscheck: AnalysisCrosscheckReport = read_json(&args[3])?;
            let attestation: IndependentReproductionAttestation = read_json(&args[4])?;
            let issues = validate_reproduction_attestation(&release, &crosscheck, &attestation);
            write_json(&args[5], &issues)?;
            fail_if_issues(
                !issues.is_empty(),
                "reproduction attestation validation failed",
            )?;
        }
        "seal-external-review-protocol" => {
            require_len(&args, 4)?;
            let mut protocol: FrozenExternalReviewProtocol = read_json(&args[2])?;
            seal_external_review_protocol(&mut protocol)?;
            write_json(&args[3], &protocol)?;
        }
        "validate-external-review-protocol" => {
            require_len(&args, 5)?;
            let operations: StudyOperationsReleaseBundle = read_json(&args[2])?;
            let protocol: FrozenExternalReviewProtocol = read_json(&args[3])?;
            let issues = validate_external_review_protocol(&operations, &protocol);
            write_json(&args[4], &issues)?;
            fail_if_issues(
                !issues.is_empty(),
                "external review protocol validation failed",
            )?;
        }
        "seal-review-evidence-index" => {
            require_len(&args, 4)?;
            let mut index: ReviewEvidenceIndex = read_json(&args[2])?;
            seal_review_evidence_index(&mut index)?;
            write_json(&args[3], &index)?;
        }
        "build-external-review-package" => {
            require_len(&args, 8)?;
            let protocol: FrozenExternalReviewProtocol = read_json(&args[2])?;
            let index: ReviewEvidenceIndex = read_json(&args[3])?;
            let package = build_external_review_package(
                &protocol,
                &index,
                &args[4],
                args[5].clone(),
                args[6].clone(),
            )
            .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[7], &package)?;
        }
        "validate-external-review-package" => {
            require_len(&args, 6)?;
            let protocol: FrozenExternalReviewProtocol = read_json(&args[2])?;
            let index: ReviewEvidenceIndex = read_json(&args[3])?;
            let package: ExternalReviewPackage = read_json(&args[4])?;
            let issues = validate_external_review_package(&protocol, &index, &package);
            write_json(&args[5], &issues)?;
            fail_if_issues(
                !issues.is_empty(),
                "external review package validation failed",
            )?;
        }
        "seal-external-review-response" => {
            require_len(&args, 4)?;
            let mut response: ExternalReviewResponse = read_json(&args[2])?;
            seal_external_review_response(&mut response)?;
            write_json(&args[3], &response)?;
        }
        "validate-external-review-response" => {
            require_len(&args, 6)?;
            let protocol: FrozenExternalReviewProtocol = read_json(&args[2])?;
            let package: ExternalReviewPackage = read_json(&args[3])?;
            let response: ExternalReviewResponse = read_json(&args[4])?;
            let issues = validate_external_review_response(&protocol, &package, &response);
            write_json(&args[5], &issues)?;
            fail_if_issues(
                !issues.is_empty(),
                "external review response validation failed",
            )?;
        }
        "seal-external-review-resolution" => {
            require_len(&args, 4)?;
            let mut ledger: ExternalReviewResolutionLedger = read_json(&args[2])?;
            seal_external_review_resolution_ledger(&mut ledger)?;
            write_json(&args[3], &ledger)?;
        }
        "validate-external-review-resolution" => {
            require_len(&args, 6)?;
            let protocol: FrozenExternalReviewProtocol = read_json(&args[2])?;
            let responses: Vec<ExternalReviewResponse> = read_json(&args[3])?;
            let ledger: ExternalReviewResolutionLedger = read_json(&args[4])?;
            let issues = validate_external_review_resolution_ledger(&protocol, &responses, &ledger);
            write_json(&args[5], &issues)?;
            fail_if_issues(
                !issues.is_empty(),
                "external review resolution validation failed",
            )?;
        }
        "build-external-review-completion" => {
            require_len(&args, 9)?;
            let protocol: FrozenExternalReviewProtocol = read_json(&args[2])?;
            let index: ReviewEvidenceIndex = read_json(&args[3])?;
            let packages: Vec<ExternalReviewPackage> = read_json(&args[4])?;
            let responses: Vec<ExternalReviewResponse> = read_json(&args[5])?;
            let resolution: ExternalReviewResolutionLedger = read_json(&args[6])?;
            let completion = build_external_review_completion(
                &protocol,
                &index,
                &packages,
                &responses,
                &resolution,
                args[7].clone(),
            )
            .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[8], &completion)?;
        }
        "validate-external-review-completion" => {
            require_len(&args, 4)?;
            let completion: ExternalReviewCompletionEvidence = read_json(&args[2])?;
            let issues = validate_external_review_completion(&completion);
            write_json(&args[3], &issues)?;
            fail_if_issues(
                !issues.is_empty(),
                "external review completion validation failed",
            )?;
        }
        "seal-confirmatory-authority" => {
            require_len(&args, 4)?;
            let mut snapshot: ConfirmatoryAuthoritySnapshot = read_json(&args[2])?;
            seal_confirmatory_authority_snapshot(&mut snapshot)?;
            write_json(&args[3], &snapshot)?;
        }
        "seal-confirmatory-amendments" => {
            require_len(&args, 4)?;
            let mut ledger: ConfirmatoryAmendmentLedger = read_json(&args[2])?;
            seal_confirmatory_amendment_ledger(&mut ledger)?;
            write_json(&args[3], &ledger)?;
        }
        "validate-confirmatory-amendments" => {
            require_len(&args, 4)?;
            let ledger: ConfirmatoryAmendmentLedger = read_json(&args[2])?;
            let issues = validate_confirmatory_amendment_ledger(&ledger);
            write_json(&args[3], &issues)?;
            fail_if_issues(
                !issues.is_empty(),
                "confirmatory amendment validation failed",
            )?;
        }
        "seal-workspace-validation" => {
            require_len(&args, 4)?;
            let mut evidence: WorkspaceValidationEvidence = read_json(&args[2])?;
            seal_workspace_validation_evidence(&mut evidence)?;
            write_json(&args[3], &evidence)?;
        }
        "seal-human-study-governance" => {
            require_len(&args, 4)?;
            let mut evidence: HumanStudyGovernanceEvidence = read_json(&args[2])?;
            seal_human_study_governance(&mut evidence)?;
            write_json(&args[3], &evidence)?;
        }
        "seal-confirmatory-dry-run" => {
            require_len(&args, 4)?;
            let mut evidence: ConfirmatoryDryRunEvidence = read_json(&args[2])?;
            seal_confirmatory_dry_run(&mut evidence)?;
            write_json(&args[3], &evidence)?;
        }
        "seal-independent-reproduction-readiness" => {
            require_len(&args, 4)?;
            let mut evidence: IndependentReproductionReadiness = read_json(&args[2])?;
            seal_independent_reproduction_readiness(&mut evidence)?;
            write_json(&args[3], &evidence)?;
        }
        "build-confirmatory-readiness" => {
            require_len(&args, 15)?;
            let pilot: PilotReviewReport = read_json(&args[3])?;
            let completion: ExternalReviewCompletionEvidence = read_json(&args[4])?;
            let amendments: ConfirmatoryAmendmentLedger = read_json(&args[5])?;
            let workspace: WorkspaceValidationEvidence = read_json(&args[6])?;
            let governance: HumanStudyGovernanceEvidence = read_json(&args[7])?;
            let dry_run: ConfirmatoryDryRunEvidence = read_json(&args[8])?;
            let reproduction: IndependentReproductionReadiness = read_json(&args[9])?;
            let report = build_confirmatory_readiness_report(
                args[2].clone(),
                &pilot,
                &completion,
                &amendments,
                &workspace,
                &governance,
                &dry_run,
                &reproduction,
                args[10].clone(),
                args[11].clone(),
                args[12].clone(),
                args[13].clone(),
            )?;
            write_json(&args[14], &report)?;
        }
        "validate-confirmatory-readiness" => {
            require_len(&args, 4)?;
            let report: ConfirmatoryReadinessReport = read_json(&args[2])?;
            let issues = validate_confirmatory_readiness_report(&report);
            write_json(&args[3], &issues)?;
            fail_if_issues(
                !issues.is_empty(),
                "confirmatory readiness validation failed",
            )?;
        }
        "build-confirmatory-readiness-release" => {
            require_len(&args, 20)?;
            let operations: StudyOperationsReleaseBundle = read_json(&args[2])?;
            let protocol: FrozenExternalReviewProtocol = read_json(&args[3])?;
            let index: ReviewEvidenceIndex = read_json(&args[4])?;
            let packages: Vec<ExternalReviewPackage> = read_json(&args[5])?;
            let responses: Vec<ExternalReviewResponse> = read_json(&args[6])?;
            let resolution: ExternalReviewResolutionLedger = read_json(&args[7])?;
            let completion: ExternalReviewCompletionEvidence = read_json(&args[8])?;
            let amendments: ConfirmatoryAmendmentLedger = read_json(&args[9])?;
            let workspace: WorkspaceValidationEvidence = read_json(&args[10])?;
            let governance: HumanStudyGovernanceEvidence = read_json(&args[11])?;
            let dry_run: ConfirmatoryDryRunEvidence = read_json(&args[12])?;
            let reproduction: IndependentReproductionReadiness = read_json(&args[13])?;
            let readiness: ConfirmatoryReadinessReport = read_json(&args[14])?;
            let bundle = build_confirmatory_readiness_release(
                &operations,
                &protocol,
                &index,
                &packages,
                &responses,
                &resolution,
                &completion,
                &amendments,
                &workspace,
                &governance,
                &dry_run,
                &reproduction,
                &readiness,
                args[15].clone(),
                args[16].clone(),
                args[17].clone(),
                args[18].clone(),
            )?;
            write_json(&args[19], &bundle)?;
        }
        "validate-confirmatory-readiness-release" => {
            require_len(&args, 17)?;
            let operations: StudyOperationsReleaseBundle = read_json(&args[2])?;
            let protocol: FrozenExternalReviewProtocol = read_json(&args[3])?;
            let index: ReviewEvidenceIndex = read_json(&args[4])?;
            let packages: Vec<ExternalReviewPackage> = read_json(&args[5])?;
            let responses: Vec<ExternalReviewResponse> = read_json(&args[6])?;
            let resolution: ExternalReviewResolutionLedger = read_json(&args[7])?;
            let completion: ExternalReviewCompletionEvidence = read_json(&args[8])?;
            let amendments: ConfirmatoryAmendmentLedger = read_json(&args[9])?;
            let workspace: WorkspaceValidationEvidence = read_json(&args[10])?;
            let governance: HumanStudyGovernanceEvidence = read_json(&args[11])?;
            let dry_run: ConfirmatoryDryRunEvidence = read_json(&args[12])?;
            let reproduction: IndependentReproductionReadiness = read_json(&args[13])?;
            let readiness: ConfirmatoryReadinessReport = read_json(&args[14])?;
            let bundle: ConfirmatoryReadinessReleaseBundle = read_json(&args[15])?;
            let issues = validate_confirmatory_readiness_release(
                &operations,
                &protocol,
                &index,
                &packages,
                &responses,
                &resolution,
                &completion,
                &amendments,
                &workspace,
                &governance,
                &dry_run,
                &reproduction,
                &readiness,
                &bundle,
            );
            write_json(&args[16], &issues)?;
            fail_if_issues(
                !issues.is_empty(),
                "confirmatory readiness release validation failed",
            )?;
        }
        "seal-confirmatory-collection-protocol" => {
            require_len(&args, 4)?;
            let mut protocol: ConfirmatoryCollectionProtocol = read_json(&args[2])?;
            seal_confirmatory_collection_protocol(&mut protocol)
                .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[3], &protocol)?;
        }
        "validate-confirmatory-collection-protocol" => {
            require_len(&args, 4)?;
            let protocol: ConfirmatoryCollectionProtocol = read_json(&args[2])?;
            let issues = validate_confirmatory_collection_protocol(&protocol);
            write_json(&args[3], &issues)?;
            fail_if_issues(
                !issues.is_empty(),
                "confirmatory collection protocol validation failed",
            )?;
        }
        "seal-confirmatory-cohort-registry" => {
            require_len(&args, 4)?;
            let mut registry: ConfirmatoryCohortRegistry = read_json(&args[2])?;
            seal_confirmatory_cohort_registry(&mut registry)?;
            write_json(&args[3], &registry)?;
        }
        "validate-confirmatory-cohort-registry" => {
            require_len(&args, 7)?;
            let cohort: ParticipantCohortSpec = read_json(&args[2])?;
            let schedule: ParticipantScheduleBook = read_json(&args[3])?;
            let protocol: ConfirmatoryCollectionProtocol = read_json(&args[4])?;
            let registry: ConfirmatoryCohortRegistry = read_json(&args[5])?;
            let issues = validate_confirmatory_cohort_registry(
                &cohort,
                &schedule,
                &protocol.protocol_sha256,
                &registry,
            );
            write_json(&args[6], &issues)?;
            fail_if_issues(
                !issues.is_empty(),
                "confirmatory cohort registry validation failed",
            )?;
        }
        "build-confirmatory-collection-snapshot" => {
            require_len(&args, 5)?;
            let protocol: ConfirmatoryCollectionProtocol = read_json(&args[2])?;
            let request: ConfirmatorySnapshotBuildRequest = read_json(&args[3])?;
            let snapshot = build_confirmatory_collection_snapshot(
                &protocol,
                request.snapshot_sequence,
                request.recorded_at_utc,
                request.previous_snapshot_sha256,
                request.sessions,
                request.integrity_incident_open,
                request.governance_abort_order_sha256,
                request.frozen_deadline_reached,
            )
            .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[4], &snapshot)?;
        }
        "validate-confirmatory-collection-snapshot" => {
            require_len(&args, 6)?;
            let protocol: ConfirmatoryCollectionProtocol = read_json(&args[2])?;
            let snapshot: ConfirmatoryCollectionSnapshot = read_json(&args[3])?;
            let previous: Option<ConfirmatoryCollectionSnapshot> = if args[4] == "-" {
                None
            } else {
                Some(read_json(&args[4])?)
            };
            let issues =
                validate_confirmatory_collection_snapshot(&protocol, &snapshot, previous.as_ref());
            write_json(&args[5], &issues)?;
            fail_if_issues(
                !issues.is_empty(),
                "confirmatory collection snapshot validation failed",
            )?;
        }
        "build-confirmatory-collection-close" => {
            require_len(&args, 10)?;
            let protocol: ConfirmatoryCollectionProtocol = read_json(&args[2])?;
            let cohort: ParticipantCohortSpec = read_json(&args[3])?;
            let schedule: ParticipantScheduleBook = read_json(&args[4])?;
            let registry: ConfirmatoryCohortRegistry = read_json(&args[5])?;
            let snapshot: ConfirmatoryCollectionSnapshot = read_json(&args[6])?;
            let evidence: ParticipantEvidenceEnvelope = read_json(&args[7])?;
            let request: ConfirmatoryCloseBuildRequest = read_json(&args[8])?;
            let receipt = build_confirmatory_collection_close(
                &protocol,
                &cohort,
                &schedule,
                &registry,
                &snapshot,
                &evidence,
                request.closed_at_utc,
                request.close_reason,
                request.signoffs,
            )
            .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[9], &receipt)?;
        }
        "validate-confirmatory-collection-close" => {
            require_len(&args, 10)?;
            let protocol: ConfirmatoryCollectionProtocol = read_json(&args[2])?;
            let cohort: ParticipantCohortSpec = read_json(&args[3])?;
            let schedule: ParticipantScheduleBook = read_json(&args[4])?;
            let registry: ConfirmatoryCohortRegistry = read_json(&args[5])?;
            let snapshot: ConfirmatoryCollectionSnapshot = read_json(&args[6])?;
            let evidence: ParticipantEvidenceEnvelope = read_json(&args[7])?;
            let receipt: ConfirmatoryCollectionCloseReceipt = read_json(&args[8])?;
            let issues = validate_confirmatory_collection_close(
                &protocol, &cohort, &schedule, &registry, &snapshot, &evidence, &receipt,
            );
            write_json(&args[9], &issues)?;
            fail_if_issues(
                !issues.is_empty(),
                "confirmatory collection close validation failed",
            )?;
        }
        "build-confirmatory-unblinding" => {
            require_len(&args, 9)?;
            let manifest: FrozenStudyManifest = read_json(&args[2])?;
            let schedule: BlindedSchedule = read_json(&args[3])?;
            let codebook: BlindingCodebook = read_json(&args[4])?;
            let secret_hex = std::fs::read_to_string(&args[5])?;
            let secret_key = decode_hex_32(secret_hex.trim()).ok_or_else(|| {
                invalid_input("secret key file must contain exactly 64 hex characters")
            })?;
            let close: ConfirmatoryCollectionCloseReceipt = read_json(&args[6])?;
            let request: ConfirmatoryUnblindingBuildRequest = read_json(&args[7])?;
            let receipt = build_confirmatory_unblinding_receipt(
                &manifest,
                &schedule,
                &codebook,
                secret_key,
                request.key_reveal_file_sha256,
                &close,
                request.unblinded_at_utc,
                request.authorizations,
            )
            .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[8], &receipt)?;
        }
        "validate-confirmatory-unblinding" => {
            require_len(&args, 9)?;
            let manifest: FrozenStudyManifest = read_json(&args[2])?;
            let schedule: BlindedSchedule = read_json(&args[3])?;
            let codebook: BlindingCodebook = read_json(&args[4])?;
            let secret_hex = std::fs::read_to_string(&args[5])?;
            let secret_key = decode_hex_32(secret_hex.trim()).ok_or_else(|| {
                invalid_input("secret key file must contain exactly 64 hex characters")
            })?;
            let close: ConfirmatoryCollectionCloseReceipt = read_json(&args[6])?;
            let receipt: ConfirmatoryUnblindingReceipt = read_json(&args[7])?;
            let issues = validate_confirmatory_unblinding_receipt(
                &manifest, &schedule, &codebook, secret_key, &close, &receipt,
            );
            write_json(&args[8], &issues)?;
            fail_if_issues(
                !issues.is_empty(),
                "confirmatory unblinding validation failed",
            )?;
        }
        "build-confirmatory-analysis-execution" => {
            require_len(&args, 10)?;
            let close: ConfirmatoryCollectionCloseReceipt = read_json(&args[2])?;
            let unblinding: ConfirmatoryUnblindingReceipt = read_json(&args[3])?;
            let dataset: CompiledStudyDataset = read_json(&args[4])?;
            let rust: NormalizedPrimaryAnalysis = read_json(&args[5])?;
            let external: NormalizedPrimaryAnalysis = read_json(&args[6])?;
            let crosscheck: AnalysisCrosscheckReport = read_json(&args[7])?;
            let request: ConfirmatoryAnalysisExecutionRequest = read_json(&args[8])?;
            let record = build_confirmatory_analysis_execution(
                &close,
                &unblinding,
                &dataset,
                request.frozen_analysis_plan_sha256,
                &rust,
                &external,
                &crosscheck,
                request.commands,
                request.deviations,
                request.executed_at_utc,
            )
            .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[9], &record)?;
        }
        "validate-confirmatory-analysis-execution" => {
            require_len(&args, 10)?;
            let close: ConfirmatoryCollectionCloseReceipt = read_json(&args[2])?;
            let unblinding: ConfirmatoryUnblindingReceipt = read_json(&args[3])?;
            let dataset: CompiledStudyDataset = read_json(&args[4])?;
            let rust: NormalizedPrimaryAnalysis = read_json(&args[5])?;
            let external: NormalizedPrimaryAnalysis = read_json(&args[6])?;
            let crosscheck: AnalysisCrosscheckReport = read_json(&args[7])?;
            let record: ConfirmatoryAnalysisExecutionRecord = read_json(&args[8])?;
            let issues = validate_confirmatory_analysis_execution(
                &close,
                &unblinding,
                &dataset,
                &rust,
                &external,
                &crosscheck,
                &record,
            );
            write_json(&args[9], &issues)?;
            fail_if_issues(
                !issues.is_empty(),
                "confirmatory analysis execution validation failed",
            )?;
        }
        "seal-confirmatory-publication" => {
            require_len(&args, 7)?;
            let manifest: FrozenStudyManifest = read_json(&args[2])?;
            let methodology: FrozenMethodologyPlan = read_json(&args[3])?;
            let analysis: ConfirmatoryAnalysisExecutionRecord = read_json(&args[4])?;
            let mut publication: ConfirmatoryPublicationRecord = read_json(&args[5])?;
            seal_confirmatory_publication(&manifest, &methodology, &analysis, &mut publication)
                .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[6], &publication)?;
        }
        "validate-confirmatory-publication" => {
            require_len(&args, 7)?;
            let manifest: FrozenStudyManifest = read_json(&args[2])?;
            let methodology: FrozenMethodologyPlan = read_json(&args[3])?;
            let analysis: ConfirmatoryAnalysisExecutionRecord = read_json(&args[4])?;
            let publication: ConfirmatoryPublicationRecord = read_json(&args[5])?;
            let issues =
                validate_confirmatory_publication(&manifest, &methodology, &analysis, &publication);
            write_json(&args[6], &issues)?;
            fail_if_issues(
                !issues.is_empty(),
                "confirmatory publication validation failed",
            )?;
        }
        "init-post-publication-audit" => {
            require_len(&args, 4)?;
            let publication: ConfirmatoryPublicationRecord = read_json(&args[2])?;
            let ledger = new_post_publication_audit(&publication)
                .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[3], &ledger)?;
        }
        "append-post-publication-event" => {
            require_len(&args, 5)?;
            let mut ledger: PostPublicationAuditLedger = read_json(&args[2])?;
            let request: PostPublicationEventRequest = read_json(&args[3])?;
            append_post_publication_event(
                &mut ledger,
                request.event_kind,
                request.recorded_at_utc,
                request.public_notice_uri,
                request.reason,
                request.authority_id,
                request.authorization_sha256,
                request.claim_changes,
                request.supporting_evidence_sha256,
            )
            .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[4], &ledger)?;
        }
        "validate-post-publication-audit" => {
            require_len(&args, 4)?;
            let ledger: PostPublicationAuditLedger = read_json(&args[2])?;
            let issues = validate_post_publication_audit(&ledger);
            write_json(&args[3], &issues)?;
            fail_if_issues(
                !issues.is_empty(),
                "post-publication audit validation failed",
            )?;
        }
        "build-confirmatory-final-release" => {
            require_len(&args, 13)?;
            let readiness: ConfirmatoryReadinessReleaseBundle = read_json(&args[2])?;
            let protocol: ConfirmatoryCollectionProtocol = read_json(&args[3])?;
            let close: ConfirmatoryCollectionCloseReceipt = read_json(&args[4])?;
            let unblinding: ConfirmatoryUnblindingReceipt = read_json(&args[5])?;
            let analysis: ConfirmatoryAnalysisExecutionRecord = read_json(&args[6])?;
            let publication: ConfirmatoryPublicationRecord = read_json(&args[7])?;
            let audit: PostPublicationAuditLedger = read_json(&args[8])?;
            let study_release: StudyReleaseBundle = read_json(&args[9])?;
            let orchestration: StudyOrchestrationLog = read_json(&args[10])?;
            let request: ConfirmatoryFinalReleaseRequest = read_json(&args[11])?;
            let bundle = build_confirmatory_final_release(
                &readiness,
                &protocol,
                &close,
                &unblinding,
                &analysis,
                &publication,
                &audit,
                &study_release,
                &orchestration,
                request.public_release_uri,
                request.released_at_utc,
            )
            .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[12], &bundle)?;
        }
        "validate-confirmatory-final-release" => {
            require_len(&args, 13)?;
            let readiness: ConfirmatoryReadinessReleaseBundle = read_json(&args[2])?;
            let protocol: ConfirmatoryCollectionProtocol = read_json(&args[3])?;
            let close: ConfirmatoryCollectionCloseReceipt = read_json(&args[4])?;
            let unblinding: ConfirmatoryUnblindingReceipt = read_json(&args[5])?;
            let analysis: ConfirmatoryAnalysisExecutionRecord = read_json(&args[6])?;
            let publication: ConfirmatoryPublicationRecord = read_json(&args[7])?;
            let audit: PostPublicationAuditLedger = read_json(&args[8])?;
            let study_release: StudyReleaseBundle = read_json(&args[9])?;
            let orchestration: StudyOrchestrationLog = read_json(&args[10])?;
            let bundle: ConfirmatoryFinalReleaseBundle = read_json(&args[11])?;
            let issues = validate_confirmatory_final_release(
                &readiness,
                &protocol,
                &close,
                &unblinding,
                &analysis,
                &publication,
                &audit,
                &study_release,
                &orchestration,
                &bundle,
            );
            write_json(&args[12], &issues)?;
            fail_if_issues(
                !issues.is_empty(),
                "confirmatory final release validation failed",
            )?;
        }
        "seal-replication-protocol" => {
            require_len(&args, 5)?;
            let source: ConfirmatoryFinalReleaseBundle = read_json(&args[2])?;
            let mut protocol: FrozenReplicationProtocol = read_json(&args[3])?;
            seal_replication_protocol(&source, &mut protocol)
                .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[4], &protocol)?;
        }
        "validate-replication-protocol" => {
            require_len(&args, 5)?;
            let source: ConfirmatoryFinalReleaseBundle = read_json(&args[2])?;
            let protocol: FrozenReplicationProtocol = read_json(&args[3])?;
            let issues = validate_replication_protocol(&source, &protocol);
            write_json(&args[4], &issues)?;
            fail_if_issues(!issues.is_empty(), "replication protocol validation failed")?;
        }
        "seal-replication-site-registry" => {
            require_len(&args, 5)?;
            let protocol: FrozenReplicationProtocol = read_json(&args[2])?;
            let mut registry: ReplicationSiteRegistry = read_json(&args[3])?;
            seal_replication_site_registry(&protocol, &mut registry)
                .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[4], &registry)?;
        }
        "validate-replication-site-registry" => {
            require_len(&args, 5)?;
            let protocol: FrozenReplicationProtocol = read_json(&args[2])?;
            let registry: ReplicationSiteRegistry = read_json(&args[3])?;
            let issues = validate_replication_site_registry(&protocol, &registry);
            write_json(&args[4], &issues)?;
            fail_if_issues(
                !issues.is_empty(),
                "replication site registry validation failed",
            )?;
        }
        "build-replication-package" => {
            require_len(&args, 9)?;
            let protocol: FrozenReplicationProtocol = read_json(&args[2])?;
            let registry: ReplicationSiteRegistry = read_json(&args[3])?;
            let entries: Vec<ReplicationPackageEntry> = read_json(&args[5])?;
            let package = build_replication_package(
                &protocol,
                &registry,
                args[4].clone(),
                entries,
                args[6].clone(),
                args[7].clone(),
            )
            .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[8], &package)?;
        }
        "validate-replication-package" => {
            require_len(&args, 6)?;
            let protocol: FrozenReplicationProtocol = read_json(&args[2])?;
            let registry: ReplicationSiteRegistry = read_json(&args[3])?;
            let package: ReplicationSitePackage = read_json(&args[4])?;
            let issues = validate_replication_package(&protocol, &registry, &package);
            write_json(&args[5], &issues)?;
            fail_if_issues(!issues.is_empty(), "replication package validation failed")?;
        }
        "seal-replication-execution" => {
            require_len(&args, 7)?;
            let protocol: FrozenReplicationProtocol = read_json(&args[2])?;
            let registry: ReplicationSiteRegistry = read_json(&args[3])?;
            let package: ReplicationSitePackage = read_json(&args[4])?;
            let mut record: ReplicationSiteExecutionRecord = read_json(&args[5])?;
            seal_replication_execution(&protocol, &registry, &package, &mut record)
                .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[6], &record)?;
        }
        "validate-replication-execution" => {
            require_len(&args, 7)?;
            let protocol: FrozenReplicationProtocol = read_json(&args[2])?;
            let registry: ReplicationSiteRegistry = read_json(&args[3])?;
            let package: ReplicationSitePackage = read_json(&args[4])?;
            let record: ReplicationSiteExecutionRecord = read_json(&args[5])?;
            let issues = validate_replication_execution(&protocol, &registry, &package, &record);
            write_json(&args[6], &issues)?;
            fail_if_issues(
                !issues.is_empty(),
                "replication execution validation failed",
            )?;
        }
        "synthesize-replications" => {
            require_len(&args, 7)?;
            let protocol: FrozenReplicationProtocol = read_json(&args[2])?;
            let registry: ReplicationSiteRegistry = read_json(&args[3])?;
            let records: Vec<ReplicationSiteExecutionRecord> = read_json(&args[4])?;
            let request: ReplicationSynthesisRequest = read_json(&args[5])?;
            let synthesis = synthesize_replications(
                &protocol,
                &registry,
                request.source_result,
                &records,
                request.analysis_implementation_sha256,
                request.independent_analysis_sha256,
                request.crosscheck_sha256,
                request.public_release_uri,
                request.completed_at_utc,
            )
            .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[6], &synthesis)?;
        }
        "validate-replication-synthesis" => {
            require_len(&args, 7)?;
            let protocol: FrozenReplicationProtocol = read_json(&args[2])?;
            let registry: ReplicationSiteRegistry = read_json(&args[3])?;
            let records: Vec<ReplicationSiteExecutionRecord> = read_json(&args[4])?;
            let synthesis: ReplicationSynthesisRecord = read_json(&args[5])?;
            let issues = validate_replication_synthesis(&protocol, &registry, &records, &synthesis);
            write_json(&args[6], &issues)?;
            fail_if_issues(
                !issues.is_empty(),
                "replication synthesis validation failed",
            )?;
        }
        "init-replication-orchestration" => {
            require_len(&args, 4)?;
            let protocol: FrozenReplicationProtocol = read_json(&args[2])?;
            write_json(&args[3], &new_replication_orchestration(&protocol)?)?;
        }
        "append-replication-transition" => {
            require_len(&args, 5)?;
            let mut log: ReplicationOrchestrationLog = read_json(&args[2])?;
            let request: ReplicationTransitionRequest = read_json(&args[3])?;
            append_replication_transition(
                &mut log,
                request.to,
                request.actor_id,
                request.authority_sha256,
                request.recorded_at_utc,
            )
            .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[4], &log)?;
        }
        "validate-replication-orchestration" => {
            require_len(&args, 4)?;
            let log: ReplicationOrchestrationLog = read_json(&args[2])?;
            let issues = validate_replication_orchestration(&log);
            write_json(&args[3], &issues)?;
            fail_if_issues(
                !issues.is_empty(),
                "replication orchestration validation failed",
            )?;
        }
        "seal-research-revision" => {
            require_len(&args, 5)?;
            let source: ConfirmatoryFinalReleaseBundle = read_json(&args[2])?;
            let mut proposal: ResearchRevisionProposal = read_json(&args[3])?;
            seal_research_revision_proposal(&source, &mut proposal)
                .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[4], &proposal)?;
        }
        "validate-research-revision" => {
            require_len(&args, 5)?;
            let source: ConfirmatoryFinalReleaseBundle = read_json(&args[2])?;
            let proposal: ResearchRevisionProposal = read_json(&args[3])?;
            let issues = validate_research_revision_proposal(&source, &proposal);
            write_json(&args[4], &issues)?;
            fail_if_issues(!issues.is_empty(), "research revision validation failed")?;
        }
        "seal-stewardship-charter" => {
            require_len(&args, 5)?;
            let source: ConfirmatoryFinalReleaseBundle = read_json(&args[2])?;
            let mut charter: ResearchStewardshipCharter = read_json(&args[3])?;
            seal_stewardship_charter(&source, &mut charter)
                .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[4], &charter)?;
        }
        "validate-stewardship-charter" => {
            require_len(&args, 5)?;
            let source: ConfirmatoryFinalReleaseBundle = read_json(&args[2])?;
            let charter: ResearchStewardshipCharter = read_json(&args[3])?;
            let issues = validate_stewardship_charter(&source, &charter);
            write_json(&args[4], &issues)?;
            fail_if_issues(!issues.is_empty(), "stewardship charter validation failed")?;
        }
        "seal-research-archive" => {
            require_len(&args, 5)?;
            let plan: ResearchArchivePlan = read_json(&args[2])?;
            let manifest = seal_research_archive(Path::new(&args[3]), &plan)
                .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[4], &manifest)?;
        }
        "validate-research-archive" => {
            require_len(&args, 6)?;
            let plan: ResearchArchivePlan = read_json(&args[2])?;
            let manifest: ResearchArchiveManifest = read_json(&args[3])?;
            let issues = validate_research_archive(Path::new(&args[4]), &plan, &manifest);
            write_json(&args[5], &issues)?;
            fail_if_issues(!issues.is_empty(), "research archive validation failed")?;
        }
        "evaluate-research-release-promotion" => {
            require_len(&args, 8)?;
            let source: ConfirmatoryFinalReleaseBundle = read_json(&args[2])?;
            let synthesis: ReplicationSynthesisRecord = read_json(&args[3])?;
            let archive: ResearchArchiveManifest = read_json(&args[4])?;
            let charter: ResearchStewardshipCharter = read_json(&args[5])?;
            let request: ResearchReleasePromotionRequest = read_json(&args[6])?;
            let record = evaluate_research_release_promotion(
                &source, &synthesis, &archive, &charter, &request,
            )
            .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[7], &record)?;
        }
        "validate-research-release-promotion" => {
            require_len(&args, 9)?;
            let source: ConfirmatoryFinalReleaseBundle = read_json(&args[2])?;
            let synthesis: ReplicationSynthesisRecord = read_json(&args[3])?;
            let archive: ResearchArchiveManifest = read_json(&args[4])?;
            let charter: ResearchStewardshipCharter = read_json(&args[5])?;
            let request: ResearchReleasePromotionRequest = read_json(&args[6])?;
            let record: ResearchReleasePromotionRecord = read_json(&args[7])?;
            let issues = validate_research_release_promotion(
                &source, &synthesis, &archive, &charter, &request, &record,
            );
            write_json(&args[8], &issues)?;
            fail_if_issues(
                !issues.is_empty(),
                "research release promotion validation failed",
            )?;
        }
        "build-stewardship-release" => {
            require_len(&args, 14)?;
            let source: ConfirmatoryFinalReleaseBundle = read_json(&args[2])?;
            let protocol: FrozenReplicationProtocol = read_json(&args[3])?;
            let registry: ReplicationSiteRegistry = read_json(&args[4])?;
            let packages: Vec<ReplicationSitePackage> = read_json(&args[5])?;
            let executions: Vec<ReplicationSiteExecutionRecord> = read_json(&args[6])?;
            let synthesis: ReplicationSynthesisRecord = read_json(&args[7])?;
            let orchestration: ReplicationOrchestrationLog = read_json(&args[8])?;
            let charter: ResearchStewardshipCharter = read_json(&args[9])?;
            let archive: ResearchArchiveManifest = read_json(&args[10])?;
            let promotion: ResearchReleasePromotionRecord = read_json(&args[11])?;
            let request: StewardshipReleaseRequest = read_json(&args[12])?;
            let bundle = build_stewardship_release(
                &source,
                &protocol,
                &registry,
                &packages,
                &executions,
                &synthesis,
                &orchestration,
                &charter,
                &archive,
                &promotion,
                request.revision_governance_policy_sha256,
                request.security_review_sha256,
                request.source_revision,
                request.workspace_tree_sha256,
                request.public_release_uri,
                request.released_at_utc,
            )
            .map_err(|issues| invalid_input(serde_json::to_string_pretty(&issues).unwrap()))?;
            write_json(&args[13], &bundle)?;
        }
        "validate-stewardship-release" => {
            require_len(&args, 14)?;
            let source: ConfirmatoryFinalReleaseBundle = read_json(&args[2])?;
            let protocol: FrozenReplicationProtocol = read_json(&args[3])?;
            let registry: ReplicationSiteRegistry = read_json(&args[4])?;
            let packages: Vec<ReplicationSitePackage> = read_json(&args[5])?;
            let executions: Vec<ReplicationSiteExecutionRecord> = read_json(&args[6])?;
            let synthesis: ReplicationSynthesisRecord = read_json(&args[7])?;
            let orchestration: ReplicationOrchestrationLog = read_json(&args[8])?;
            let charter: ResearchStewardshipCharter = read_json(&args[9])?;
            let archive: ResearchArchiveManifest = read_json(&args[10])?;
            let promotion: ResearchReleasePromotionRecord = read_json(&args[11])?;
            let bundle: StewardshipReleaseBundle = read_json(&args[12])?;
            let issues = validate_stewardship_release(
                &source,
                &protocol,
                &registry,
                &packages,
                &executions,
                &synthesis,
                &orchestration,
                &charter,
                &archive,
                &promotion,
                &bundle,
            );
            write_json(&args[13], &issues)?;
            fail_if_issues(!issues.is_empty(), "stewardship release validation failed")?;
        }
        "analyze-temporal" => {
            require_len(&args, 5)?;
            let records: Vec<FrozenTemporalRecord> = read_json(&args[2])?;
            let plan: TemporalConfirmatoryPlan = read_json(&args[3])?;
            let report = analyze_temporal_confirmatory(&records, &plan);
            write_json(&args[4], &report)?;
            fail_if_issues(
                !report.issues.is_empty(),
                "temporal analysis validation failed",
            )?;
        }
        _ => {
            print_usage();
            return Err(invalid_input(format!("unknown command: {command}")));
        }
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct OrchestrationTransitionRequest {
    to: StudyLifecyclePhase,
    recorded_at_utc: String,
    operator_id: String,
    authorization_sha256: String,
    added_authorities: Vec<StudyAuthorityBinding>,
}

#[derive(Debug, Deserialize)]
struct ConfirmatorySnapshotBuildRequest {
    snapshot_sequence: u32,
    recorded_at_utc: String,
    previous_snapshot_sha256: String,
    sessions: Vec<ConfirmatorySessionStatus>,
    integrity_incident_open: bool,
    governance_abort_order_sha256: Option<String>,
    frozen_deadline_reached: bool,
}

#[derive(Debug, Deserialize)]
struct ConfirmatoryCloseBuildRequest {
    closed_at_utc: String,
    close_reason: ConfirmatoryCollectionCloseReason,
    signoffs: Vec<ConfirmatoryCloseSignoff>,
}

#[derive(Debug, Deserialize)]
struct ConfirmatoryUnblindingBuildRequest {
    key_reveal_file_sha256: String,
    unblinded_at_utc: String,
    authorizations: Vec<ConfirmatoryUnblindingAuthorization>,
}

#[derive(Debug, Deserialize)]
struct ConfirmatoryAnalysisExecutionRequest {
    frozen_analysis_plan_sha256: String,
    commands: Vec<ConfirmatoryAnalysisCommandEvidence>,
    deviations: Vec<ConfirmatoryAnalysisDeviation>,
    executed_at_utc: String,
}

#[derive(Debug, Deserialize)]
struct PostPublicationEventRequest {
    event_kind: PostPublicationEventKind,
    recorded_at_utc: String,
    public_notice_uri: String,
    reason: String,
    authority_id: String,
    authorization_sha256: String,
    claim_changes: Vec<PostPublicationClaimChange>,
    supporting_evidence_sha256: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ConfirmatoryFinalReleaseRequest {
    public_release_uri: String,
    released_at_utc: String,
}

#[derive(Debug, Deserialize)]
struct ReplicationSynthesisRequest {
    source_result: PublishedSourcePrimaryResult,
    analysis_implementation_sha256: String,
    independent_analysis_sha256: String,
    crosscheck_sha256: String,
    public_release_uri: String,
    completed_at_utc: String,
}

#[derive(Debug, Deserialize)]
struct ReplicationTransitionRequest {
    to: ReplicationLifecyclePhase,
    actor_id: String,
    authority_sha256: String,
    recorded_at_utc: String,
}

#[derive(Debug, Deserialize)]
struct StewardshipReleaseRequest {
    revision_governance_policy_sha256: String,
    security_review_sha256: String,
    source_revision: String,
    workspace_tree_sha256: String,
    public_release_uri: String,
    released_at_utc: String,
}

fn require_len(args: &[String], expected: usize) -> Result<(), Box<dyn Error>> {
    if args.len() == expected {
        Ok(())
    } else {
        print_usage();
        Err(invalid_input(format!(
            "expected {} arguments, found {}",
            expected - 1,
            args.len().saturating_sub(1)
        )))
    }
}

fn read_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T, Box<dyn Error>> {
    Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
}

fn write_json<T: Serialize>(path: impl AsRef<Path>, value: &T) -> Result<(), Box<dyn Error>> {
    let mut writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn write_stdout_json<T: Serialize>(value: &T) -> Result<(), Box<dyn Error>> {
    let stdout = io::stdout();
    let mut lock = stdout.lock();
    serde_json::to_writer_pretty(&mut lock, value)?;
    lock.write_all(b"\n")?;
    Ok(())
}

fn fail_if_issues(has_issues: bool, message: &str) -> Result<(), Box<dyn Error>> {
    if has_issues {
        Err(invalid_input(message))
    } else {
        Ok(())
    }
}

fn invalid_input(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}

fn print_usage() {
    eprintln!(
        "usage:\n  cognitive_study digest-json FILE.json\n  cognitive_study validate-manifest MANIFEST.json\n  cognitive_study validate-methodology MANIFEST.json METHODOLOGY.json\n  cognitive_study build-schedule MANIFEST.json ARTIFACTS.json SECRET_KEY_FILE SCHEDULE.json CODEBOOK.json\n  cognitive_study validate-schedule MANIFEST.json SCHEDULE.json CODEBOOK.json\n  cognitive_study seal-artifact-bundle MANIFEST.json METHODOLOGY.json SCHEDULE.json PLAN.json ARTIFACT_ROOT BUNDLE.json\n  cognitive_study validate-artifact-bundle MANIFEST.json METHODOLOGY.json SCHEDULE.json PLAN.json BUNDLE.json ARTIFACT_ROOT ISSUES.json\n  cognitive_study build-runner-package SCHEDULE.json PARTICIPANT_SCHEDULE.json ARTIFACT_BUNDLE.json BLOCK_ID PROTOCOL.json PACKAGE.json\n  cognitive_study validate-runner-package SCHEDULE.json PARTICIPANT_SCHEDULE.json ARTIFACT_BUNDLE.json PACKAGE.json ISSUES.json\n  cognitive_study init-session PACKAGE.json SESSION.json\n  cognitive_study append-session-event PACKAGE.json SESSION.json EVENT.json SERVER_UNIX_MS CLIENT_ELAPSED_MS UPDATED_SESSION.json\n  cognitive_study validate-session PACKAGE.json SESSION.json ISSUES.json\n  cognitive_study compile-session PACKAGE.json SESSION.json LISTENER_BLOCK.json\n  cognitive_study seal-runner-collection MANIFEST.json SCHEDULE.json PARTICIPANT_SCHEDULE.json ARTIFACT_BUNDLE.json COLLECTION.json ENVELOPE.json\n  cognitive_study seal-release RELEASE_PLAN.json RELEASE_ROOT RELEASE_BUNDLE.json\n  cognitive_study validate-release RELEASE_PLAN.json RELEASE_BUNDLE.json RELEASE_ROOT ISSUES.json\n  cognitive_study build-participant-schedule MANIFEST.json SCHEDULE.json CODEBOOK.json COHORT.json SECRET_KEY_FILE PARTICIPANT_SCHEDULE.json PARTICIPANT_AUDIT.json\n  cognitive_study validate-participant-schedule MANIFEST.json SCHEDULE.json COHORT.json PARTICIPANT_SCHEDULE.json\n  cognitive_study seal-structural-evidence STRUCTURAL_DRAFT.json STRUCTURAL_SEALED.json\n  cognitive_study compile-structural-evidence MANIFEST.json SCHEDULE.json METHODOLOGY.json STRUCTURAL.json OUTCOMES.json\n  cognitive_study seal-policy-budget BUDGET_DRAFT.json BUDGET_SEALED.json\n  cognitive_study validate-policy-budget MANIFEST.json METHODOLOGY.json BUDGET.json\n  cognitive_study seal-participant-evidence MANIFEST.json SCHEDULE.json PARTICIPANT_SCHEDULE.json RAW.json ENVELOPE.json\n  cognitive_study compile-participant-evidence MANIFEST.json SCHEDULE.json CODEBOOK.json COHORT.json PARTICIPANT_SCHEDULE.json ENVELOPE.json DATASET.json\n  cognitive_study compile-evidence MANIFEST.json SCHEDULE.json CODEBOOK.json EVIDENCE.json DATASET.json\n  cognitive_study analyze MANIFEST.json DATASET.json PLAN.json REPORT.json\n  cognitive_study analyze-family-clustered MANIFEST.json METHODOLOGY.json DATASET.json PLAN.json REPORT.json\n  cognitive_study analyze-ranked-preference MANIFEST.json METHODOLOGY.json SCHEDULE.json CODEBOOK.json COHORT.json PARTICIPANT_SCHEDULE.json ENVELOPE.json PLAN.json REPORT.json\n  cognitive_study seal-external-review-protocol DRAFT.json SEALED.json
  cognitive_study validate-external-review-protocol OPERATIONS_RELEASE.json PROTOCOL.json ISSUES.json
  cognitive_study seal-review-evidence-index DRAFT.json SEALED.json
  cognitive_study build-external-review-package PROTOCOL.json INDEX.json REVIEWER_ID ISSUED_AT_UTC INSTRUCTIONS_SHA256 PACKAGE.json
  cognitive_study validate-external-review-package PROTOCOL.json INDEX.json PACKAGE.json ISSUES.json
  cognitive_study seal-external-review-response DRAFT.json SEALED.json
  cognitive_study validate-external-review-response PROTOCOL.json PACKAGE.json RESPONSE.json ISSUES.json
  cognitive_study seal-external-review-resolution DRAFT.json SEALED.json
  cognitive_study validate-external-review-resolution PROTOCOL.json RESPONSES.json LEDGER.json ISSUES.json
  cognitive_study build-external-review-completion PROTOCOL.json INDEX.json PACKAGES.json RESPONSES.json RESOLUTION.json COMPLETED_AT_UTC COMPLETION.json
  cognitive_study validate-external-review-completion COMPLETION.json ISSUES.json
  cognitive_study seal-confirmatory-authority DRAFT.json SEALED.json
  cognitive_study seal-confirmatory-amendments DRAFT.json SEALED.json
  cognitive_study validate-confirmatory-amendments LEDGER.json ISSUES.json
  cognitive_study seal-workspace-validation DRAFT.json SEALED.json
  cognitive_study seal-human-study-governance DRAFT.json SEALED.json
  cognitive_study seal-confirmatory-dry-run DRAFT.json SEALED.json
  cognitive_study seal-independent-reproduction-readiness DRAFT.json SEALED.json
  cognitive_study build-confirmatory-readiness OPERATIONS_RELEASE_SHA256 PILOT_REPORT.json REVIEW_COMPLETION.json AMENDMENTS.json WORKSPACE.json GOVERNANCE.json DRY_RUN.json REPRODUCTION.json DECIDED_AT_UTC AUTHORITY RECEIPT_URI RECEIPT_SHA256 REPORT.json
  cognitive_study validate-confirmatory-readiness REPORT.json ISSUES.json
  cognitive_study build-confirmatory-readiness-release OPERATIONS.json PROTOCOL.json INDEX.json PACKAGES.json RESPONSES.json RESOLUTION.json COMPLETION.json AMENDMENTS.json WORKSPACE.json GOVERNANCE.json DRY_RUN.json REPRODUCTION.json READINESS.json SOURCE_SHA256 FLAKE_LOCK_SHA256 TOOLCHAIN_SHA256 TIMESTAMP_SHA256 RELEASE.json
  cognitive_study validate-confirmatory-readiness-release OPERATIONS.json PROTOCOL.json INDEX.json PACKAGES.json RESPONSES.json RESOLUTION.json COMPLETION.json AMENDMENTS.json WORKSPACE.json GOVERNANCE.json DRY_RUN.json REPRODUCTION.json READINESS.json RELEASE.json ISSUES.json
  cognitive_study analyze-temporal RECORDS.json PLAN.json REPORT.json
  cognitive_study validate-pilot-protocol MANIFEST.json METHODOLOGY.json PILOT_PROTOCOL.json
  cognitive_study build-pilot-schedule MANIFEST.json METHODOLOGY.json SCHEDULE.json CODEBOOK.json PILOT_PROTOCOL.json COHORT.json PILOT_SECRET_KEY_FILE PILOT_SCHEDULE.json PILOT_AUDIT.json
  cognitive_study validate-pilot-schedule MANIFEST.json METHODOLOGY.json SCHEDULE.json CODEBOOK.json PILOT_PROTOCOL.json COHORT.json PILOT_SCHEDULE.json PILOT_AUDIT.json ISSUES.json
  cognitive_study build-pilot-runner-package SCHEDULE.json PILOT_SCHEDULE.json ARTIFACT_BUNDLE.json BLOCK_ID PROTOCOL.json PACKAGE.json
  cognitive_study validate-pilot-runner-package SCHEDULE.json PILOT_SCHEDULE.json ARTIFACT_BUNDLE.json PACKAGE.json ISSUES.json
  cognitive_study seal-pilot-cohort-registry DRAFT.json SEALED.json
  cognitive_study validate-pilot-cohort-registry PILOT_PROTOCOL.json COHORT.json PILOT_SCHEDULE.json REGISTRY.json ISSUES.json
  cognitive_study seal-pilot-collection PILOT_PROTOCOL.json SCHEDULE.json PILOT_SCHEDULE.json ARTIFACTS.json REGISTRY.json COLLECTED_AT_UTC SESSIONS.json COLLECTION.json
  cognitive_study validate-pilot-collection PILOT_PROTOCOL.json SCHEDULE.json PILOT_SCHEDULE.json ARTIFACTS.json REGISTRY.json COLLECTION.json ISSUES.json
  cognitive_study build-pilot-snapshot PILOT_PROTOCOL.json COLLECTION.json OBSERVED_AT_UTC SNAPSHOT.json
  cognitive_study validate-pilot-snapshot PILOT_PROTOCOL.json SNAPSHOT.json ISSUES.json
  cognitive_study seal-pilot-amendments DRAFT.json SEALED.json
  cognitive_study validate-pilot-amendments LEDGER.json ISSUES.json
  cognitive_study seal-sample-size-recommendation DRAFT.json SEALED.json
  cognitive_study seal-pilot-report DRAFT.json SEALED.json
  cognitive_study validate-pilot-report PILOT_PROTOCOL.json AMENDMENTS.json SNAPSHOT.json REPORT.json ISSUES.json
  cognitive_study init-orchestration ORCHESTRATION_ID LOG.json
  cognitive_study append-orchestration LOG.json REQUEST.json UPDATED_LOG.json
  cognitive_study upgrade-orchestration-v12 LEGACY_LOG.json UPGRADED_LOG.json
  cognitive_study upgrade-orchestration-v11 LEGACY_LOG.json UPGRADED_LOG.json  # legacy alias
  cognitive_study validate-orchestration LOG.json ISSUES.json
  cognitive_study seal-normalized-analysis DRAFT.json SEALED.json
  cognitive_study crosscheck-analyses RUST.json EXTERNAL.json TOLERANCE.json REPORT.json
  cognitive_study validate-analysis-crosscheck RUST.json EXTERNAL.json REPORT.json ISSUES.json
  cognitive_study seal-reproduction-attestation DRAFT.json SEALED.json
  cognitive_study validate-reproduction-attestation RELEASE.json CROSSCHECK.json ATTESTATION.json ISSUES.json
  cognitive_study build-operations-release BASE_RELEASE.json PILOT_PROTOCOL.json PILOT_SCHEDULE.json REGISTRY.json COLLECTION.json SNAPSHOT.json AMENDMENTS.json PILOT_REPORT.json ORCHESTRATION.json CROSSCHECK.json ATTESTATION.json SOURCE_SHA256 NIX_LOCK_SHA256 TOOLCHAIN_SHA256 OPERATIONS_RELEASE.json
  cognitive_study validate-operations-release BASE_RELEASE.json PILOT_PROTOCOL.json PILOT_SCHEDULE.json REGISTRY.json COLLECTION.json SNAPSHOT.json AMENDMENTS.json PILOT_REPORT.json ORCHESTRATION.json CROSSCHECK.json ATTESTATION.json OPERATIONS_RELEASE.json ISSUES.json
  cognitive_study seal-confirmatory-collection-protocol DRAFT.json SEALED.json
  cognitive_study validate-confirmatory-collection-protocol PROTOCOL.json ISSUES.json
  cognitive_study seal-confirmatory-cohort-registry DRAFT.json SEALED.json
  cognitive_study validate-confirmatory-cohort-registry COHORT.json PARTICIPANT_SCHEDULE.json PROTOCOL.json REGISTRY.json ISSUES.json
  cognitive_study build-confirmatory-collection-snapshot PROTOCOL.json REQUEST.json SNAPSHOT.json
  cognitive_study validate-confirmatory-collection-snapshot PROTOCOL.json SNAPSHOT.json PREVIOUS_OR_DASH ISSUES.json
  cognitive_study build-confirmatory-collection-close PROTOCOL.json COHORT.json PARTICIPANT_SCHEDULE.json REGISTRY.json SNAPSHOT.json EVIDENCE.json REQUEST.json RECEIPT.json
  cognitive_study validate-confirmatory-collection-close PROTOCOL.json COHORT.json PARTICIPANT_SCHEDULE.json REGISTRY.json SNAPSHOT.json EVIDENCE.json RECEIPT.json ISSUES.json
  cognitive_study build-confirmatory-unblinding MANIFEST.json SCHEDULE.json CODEBOOK.json SECRET_KEY_FILE CLOSE.json REQUEST.json RECEIPT.json
  cognitive_study validate-confirmatory-unblinding MANIFEST.json SCHEDULE.json CODEBOOK.json SECRET_KEY_FILE CLOSE.json RECEIPT.json ISSUES.json
  cognitive_study build-confirmatory-analysis-execution CLOSE.json UNBLINDING.json DATASET.json RUST.json EXTERNAL.json CROSSCHECK.json REQUEST.json RECORD.json
  cognitive_study validate-confirmatory-analysis-execution CLOSE.json UNBLINDING.json DATASET.json RUST.json EXTERNAL.json CROSSCHECK.json RECORD.json ISSUES.json
  cognitive_study seal-confirmatory-publication MANIFEST.json METHODOLOGY.json ANALYSIS.json DRAFT.json SEALED.json
  cognitive_study validate-confirmatory-publication MANIFEST.json METHODOLOGY.json ANALYSIS.json PUBLICATION.json ISSUES.json
  cognitive_study init-post-publication-audit PUBLICATION.json LEDGER.json
  cognitive_study append-post-publication-event LEDGER.json REQUEST.json UPDATED_LEDGER.json
  cognitive_study validate-post-publication-audit LEDGER.json ISSUES.json
  cognitive_study build-confirmatory-final-release READINESS.json PROTOCOL.json CLOSE.json UNBLINDING.json ANALYSIS.json PUBLICATION.json AUDIT.json STUDY_RELEASE.json ORCHESTRATION.json REQUEST.json BUNDLE.json
  cognitive_study validate-confirmatory-final-release READINESS.json PROTOCOL.json CLOSE.json UNBLINDING.json ANALYSIS.json PUBLICATION.json AUDIT.json STUDY_RELEASE.json ORCHESTRATION.json BUNDLE.json ISSUES.json
  cognitive_study seal-replication-protocol SOURCE_FINAL.json DRAFT.json SEALED.json
  cognitive_study validate-replication-protocol SOURCE_FINAL.json PROTOCOL.json ISSUES.json
  cognitive_study seal-replication-site-registry PROTOCOL.json DRAFT.json SEALED.json
  cognitive_study validate-replication-site-registry PROTOCOL.json REGISTRY.json ISSUES.json
  cognitive_study build-replication-package PROTOCOL.json REGISTRY.json SITE_ID ENTRIES.json ISSUED_BY ISSUED_AT_UTC PACKAGE.json
  cognitive_study validate-replication-package PROTOCOL.json REGISTRY.json PACKAGE.json ISSUES.json
  cognitive_study seal-replication-execution PROTOCOL.json REGISTRY.json PACKAGE.json DRAFT.json SEALED.json
  cognitive_study validate-replication-execution PROTOCOL.json REGISTRY.json PACKAGE.json RECORD.json ISSUES.json
  cognitive_study synthesize-replications PROTOCOL.json REGISTRY.json RECORDS.json REQUEST.json SYNTHESIS.json
  cognitive_study validate-replication-synthesis PROTOCOL.json REGISTRY.json RECORDS.json SYNTHESIS.json ISSUES.json
  cognitive_study init-replication-orchestration PROTOCOL.json LOG.json
  cognitive_study append-replication-transition LOG.json REQUEST.json UPDATED_LOG.json
  cognitive_study validate-replication-orchestration LOG.json ISSUES.json
  cognitive_study seal-research-revision SOURCE_FINAL.json DRAFT.json SEALED.json
  cognitive_study validate-research-revision SOURCE_FINAL.json PROPOSAL.json ISSUES.json
  cognitive_study seal-stewardship-charter SOURCE_FINAL.json DRAFT.json SEALED.json
  cognitive_study validate-stewardship-charter SOURCE_FINAL.json CHARTER.json ISSUES.json
  cognitive_study seal-research-archive PLAN.json ARCHIVE_ROOT MANIFEST.json
  cognitive_study validate-research-archive PLAN.json MANIFEST.json ARCHIVE_ROOT ISSUES.json
  cognitive_study evaluate-research-release-promotion SOURCE_FINAL.json SYNTHESIS.json ARCHIVE.json CHARTER.json REQUEST.json RECORD.json
  cognitive_study validate-research-release-promotion SOURCE_FINAL.json SYNTHESIS.json ARCHIVE.json CHARTER.json REQUEST.json RECORD.json ISSUES.json
  cognitive_study build-stewardship-release SOURCE_FINAL.json PROTOCOL.json REGISTRY.json PACKAGES.json EXECUTIONS.json SYNTHESIS.json ORCHESTRATION.json CHARTER.json ARCHIVE.json PROMOTION.json REQUEST.json BUNDLE.json
  cognitive_study validate-stewardship-release SOURCE_FINAL.json PROTOCOL.json REGISTRY.json PACKAGES.json EXECUTIONS.json SYNTHESIS.json ORCHESTRATION.json CHARTER.json ARCHIVE.json PROMOTION.json BUNDLE.json ISSUES.json"
    );
}
