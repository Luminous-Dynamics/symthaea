// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Source-bound IBM i foundation for legacy enterprise IT reasoning.
//!
//! This module deepens the IBM i slice of `LegacyComputingPackV1` with explicit
//! mechanisms, evidence signals, failure modes, and non-executable diagnostic
//! procedures. Seeded propositions are version-scoped to IBM i 7.6 and remain
//! metadata-only until the separate retained-source artifact path freezes exact
//! vendor documentation bytes.

use crate::knowledge_source::{
    KnowledgeAuthorityClassV1, KnowledgeLifecycleV1, KnowledgeStabilityV1,
};
use crate::legacy_computing::{
    LegacyComputingErrorV1, LegacyComputingPackV1, LegacyCoverageStateV1,
    LegacyKnowledgeAreaV1, LegacyPlatformV1, LegacyProcedureAuthorityV1,
    LegacyProcedureKindV1, LegacyProcedureStepV1, LegacyProcedureV1,
};
use crate::standards_registry::{
    ClaimModalityV1, SourceCaptureV1, SourceDocumentIdV1, SourceDocumentKindV1,
    SourceSnapshotIdV1, StandardsRegistryErrorV1, TechnicalClaimIdV1,
    TechnicalKnowledgeClaimV1, TechnicalPublisherV1, TechnicalSourceDocumentV1,
    TechnicalSourceLocatorV1, TechnicalSourceSnapshotV1,
};
use crate::technology::{ApplicabilityScopeV1, StringSelectorV1};
use crate::types::SupportCategory;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const LEGACY_IBMI_FOUNDATION_SCHEMA_V1: &str = "symthaea-it-legacy-ibmi-foundation-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum IbmiMechanismKindV1 {
    ObjectLibraryNamespace,
    JobSubsystemWorkManagement,
    Db2ForI,
    ObjectAuthority,
    SaveRestore,
    TcpIpNetworking,
    ControlLanguage,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IbmiEvidenceSignalV1 {
    pub id: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IbmiFailureModeV1 {
    pub id: String,
    pub symptom: String,
    pub discriminators: Vec<String>,
    pub unsafe_shortcuts: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IbmiMechanismModelV1 {
    pub kind: IbmiMechanismKindV1,
    pub area: LegacyKnowledgeAreaV1,
    pub title: String,
    pub summary: String,
    pub dependencies: BTreeSet<IbmiMechanismKindV1>,
    pub source_claims: BTreeSet<TechnicalClaimIdV1>,
    pub evidence_signals: Vec<IbmiEvidenceSignalV1>,
    pub failure_modes: Vec<IbmiFailureModeV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IbmiFoundationV1 {
    pub schema_version: String,
    pub product_version: String,
    pub mechanisms: Vec<IbmiMechanismModelV1>,
    pub procedure_ids: BTreeSet<String>,
}

impl IbmiFoundationV1 {
    pub fn validate(&self, pack: &LegacyComputingPackV1) -> Result<(), LegacyIbmiErrorV1> {
        if self.schema_version != LEGACY_IBMI_FOUNDATION_SCHEMA_V1 {
            return Err(LegacyIbmiErrorV1::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        require_nonempty(&self.product_version, "IBM i product version")?;
        if self.mechanisms.is_empty() {
            return Err(LegacyIbmiErrorV1::InvalidField(
                "IBM i foundation requires mechanisms".into(),
            ));
        }

        let mut kinds = BTreeSet::new();
        for mechanism in &self.mechanisms {
            if !kinds.insert(mechanism.kind) {
                return Err(LegacyIbmiErrorV1::DuplicateMechanism(mechanism.kind));
            }
            require_nonempty(&mechanism.title, "IBM i mechanism title")?;
            require_nonempty(&mechanism.summary, "IBM i mechanism summary")?;
            if mechanism.source_claims.is_empty() {
                return Err(LegacyIbmiErrorV1::InvalidField(format!(
                    "IBM i mechanism {:?} requires source claims",
                    mechanism.kind
                )));
            }
            for claim_id in &mechanism.source_claims {
                if pack.sources.claim(claim_id).is_none() {
                    return Err(LegacyIbmiErrorV1::UnknownClaim(claim_id.clone()));
                }
            }
            if mechanism.evidence_signals.is_empty() || mechanism.failure_modes.is_empty() {
                return Err(LegacyIbmiErrorV1::InvalidField(format!(
                    "IBM i mechanism {:?} requires evidence and failure modes",
                    mechanism.kind
                )));
            }
            let mut evidence_ids = BTreeSet::new();
            for evidence in &mechanism.evidence_signals {
                require_nonempty(&evidence.id, "IBM i evidence id")?;
                require_nonempty(&evidence.description, "IBM i evidence description")?;
                if !evidence_ids.insert(evidence.id.as_str()) {
                    return Err(LegacyIbmiErrorV1::DuplicateEvidenceId(evidence.id.clone()));
                }
            }
            let mut failure_ids = BTreeSet::new();
            for failure in &mechanism.failure_modes {
                require_nonempty(&failure.id, "IBM i failure mode id")?;
                require_nonempty(&failure.symptom, "IBM i failure mode symptom")?;
                if !failure_ids.insert(failure.id.as_str()) {
                    return Err(LegacyIbmiErrorV1::DuplicateFailureMode(failure.id.clone()));
                }
                if failure.discriminators.is_empty() {
                    return Err(LegacyIbmiErrorV1::InvalidField(format!(
                        "IBM i failure mode {} requires discriminators",
                        failure.id
                    )));
                }
            }
        }

        let procedure_ids: BTreeSet<_> = pack.procedures.iter().map(|p| p.id.as_str()).collect();
        for procedure_id in &self.procedure_ids {
            if !procedure_ids.contains(procedure_id.as_str()) {
                return Err(LegacyIbmiErrorV1::UnknownProcedure(procedure_id.clone()));
            }
        }
        Ok(())
    }
}

pub fn enrich_legacy_ibmi_foundation_v1(
    pack: &mut LegacyComputingPackV1,
    fetched_at_unix_ms: u64,
) -> Result<IbmiFoundationV1, LegacyIbmiErrorV1> {
    if fetched_at_unix_ms == 0 {
        return Err(LegacyIbmiErrorV1::InvalidField(
            "IBM i source fetch timestamp must be non-zero".into(),
        ));
    }
    pack.validate()?;

    let snapshots = register_ibmi_sources(pack, fetched_at_unix_ms)?;
    register_ibmi_claims(pack)?;
    let procedure_ids = add_ibmi_procedures(pack, &snapshots)?;
    update_ibmi_profile(pack, &snapshots)?;

    let foundation = IbmiFoundationV1 {
        schema_version: LEGACY_IBMI_FOUNDATION_SCHEMA_V1.into(),
        product_version: "7.6".into(),
        mechanisms: seed_mechanisms(),
        procedure_ids,
    };
    pack.validate()?;
    foundation.validate(pack)?;
    Ok(foundation)
}

fn register_ibmi_sources(
    pack: &mut LegacyComputingPackV1,
    fetched_at_unix_ms: u64,
) -> Result<BTreeSet<SourceSnapshotIdV1>, LegacyIbmiErrorV1> {
    let definitions = [
        (
            "ibm:ibmi-object-authority",
            "ibm:ibmi-object-authority@7.6",
            "IBM i object and library authority",
            "IBM i 7.6 object authority",
            "https://www.ibm.com/docs/en/i/7.6.0?topic=e-edit-object-authority",
        ),
        (
            "ibm:ibmi-work-management",
            "ibm:ibmi-work-management@7.6",
            "IBM i work management and job queues",
            "IBM i 7.6 work management",
            "https://www.ibm.com/support/pages/active-job-queue-analysis-qsys2jobqueueinfo",
        ),
        (
            "ibm:ibmi-db2",
            "ibm:ibmi-db2@7.6",
            "Db2 for IBM i",
            "Db2 for IBM i 7.6",
            "https://www.ibm.com/support/pages/db2-ibm-i",
        ),
        (
            "ibm:ibmi-db2-distributed",
            "ibm:ibmi-db2-distributed@7.6",
            "Db2 for i distributed relational database support",
            "IBM i 7.6 Db2 distributed relational database support",
            "https://www.ibm.com/docs/en/i/7.6.0?topic=sql-db2-i-distributed-relational-database-support",
        ),
        (
            "ibm:ibmi-authority-collection",
            "ibm:ibmi-authority-collection@7.6",
            "IBM i authority collection",
            "IBM i 7.6 authority collection",
            "https://www.ibm.com/docs/en/i/7.6.0?topic=collection-authority-interfaces",
        ),
        (
            "ibm:ibmi-save-restore",
            "ibm:ibmi-save-restore@7.6",
            "IBM i save and restore",
            "IBM i 7.6 save and restore",
            "https://www.ibm.com/support/pages/node/7181664",
        ),
        (
            "ibm:ibmi-tcpip-connectivity",
            "ibm:ibmi-tcpip-connectivity@7.6",
            "IBM TCP/IP Connectivity for i",
            "IBM i 7.6 TCP/IP connectivity",
            "https://www.ibm.com/docs/en/i/7.6.0?topic=c-software-requirements-2",
        ),
        (
            "ibm:ibmi-cl-overview",
            "ibm:ibmi-cl-overview@7.6",
            "IBM i Control Language",
            "IBM i 7.6 Control Language",
            "https://www.ibm.com/docs/en/i/7.6.0",
        ),
    ];

    let mut snapshots = BTreeSet::new();
    for (document_id, snapshot_id, title, canonical_ref, locator) in definitions {
        pack.sources.register_document(TechnicalSourceDocumentV1 {
            id: SourceDocumentIdV1(document_id.into()),
            publisher: TechnicalPublisherV1::Vendor("IBM".into()),
            kind: SourceDocumentKindV1::VendorDocumentation,
            title: title.into(),
            canonical_ref: canonical_ref.into(),
            canonical_locator: Some(locator.into()),
        })?;
        pack.sources.register_snapshot(TechnicalSourceSnapshotV1 {
            id: SourceSnapshotIdV1(snapshot_id.into()),
            document_id: SourceDocumentIdV1(document_id.into()),
            version: Some("7.6".into()),
            lifecycle: KnowledgeLifecycleV1::Active,
            authority: KnowledgeAuthorityClassV1::VendorDocumentation,
            stability: KnowledgeStabilityV1::Stable,
            published_at_unix_ms: None,
            source_updated_at_unix_ms: None,
            fetched_at_unix_ms,
            capture: SourceCaptureV1::MetadataOnly,
            relations: BTreeSet::new(),
        })?;
        snapshots.insert(SourceSnapshotIdV1(snapshot_id.into()));
    }
    Ok(snapshots)
}

fn register_ibmi_claims(pack: &mut LegacyComputingPackV1) -> Result<(), LegacyIbmiErrorV1> {
    let claims = [
        claim(
            "legacy:ibmi:object-library-namespace",
            "ibm:ibmi-object-authority@7.6",
            "IBM i resources are typed system objects addressed through libraries and library-list resolution; object authority views expose object name, containing library, owner, type, primary group, authorized users, and authorization-list association.",
            "Edit Object Authority",
            SupportCategory::Software,
        ),
        claim(
            "legacy:ibmi:subsystem-job-queue",
            "ibm:ibmi-work-management@7.6",
            "IBM i batch work is mediated by job queues and subsystem descriptions; a job queue can feed a subsystem and queue/subsystem limits can explain waiting work without implying application failure.",
            "Active job queue analysis",
            SupportCategory::Software,
        ),
        claim(
            "legacy:ibmi:db2-integrated",
            "ibm:ibmi-db2@7.6",
            "Db2 for i is integrated with IBM i and Power Systems rather than managed as a separately installed database engine; IBM i storage and operating-system facilities are part of the database operating model.",
            "Db2 for IBM i",
            SupportCategory::Software,
        ),
        claim(
            "legacy:ibmi:db2-distributed-connectivity",
            "ibm:ibmi-db2-distributed@7.6",
            "Db2 for i supports distributed relational database connections and package operations, so local database health and remote relational-database connectivity are distinct diagnostic dimensions.",
            "Db2 for i distributed relational database support",
            SupportCategory::Network,
        ),
        claim(
            "legacy:ibmi:object-authority-model",
            "ibm:ibmi-object-authority@7.6",
            "IBM i object access is governed by object-specific authorities, ownership, special authorities, group/profile state, and optional authorization lists; broad special authority is not equivalent to least-privilege diagnosis.",
            "Edit Object Authority",
            SupportCategory::Security,
        ),
        claim(
            "legacy:ibmi:authority-collection-observation",
            "ibm:ibmi-authority-collection@7.6",
            "IBM i authority collection can record authority-check information for users and objects and can be inspected through commands, APIs, and QSYS2 views without granting broader object authority.",
            "Authority collection interfaces",
            SupportCategory::Security,
        ),
        claim(
            "legacy:ibmi:save-restore-semantics",
            "ibm:ibmi-save-restore@7.6",
            "IBM i backup and recovery behavior depends on object type, save scope, media/save-file state, access paths and object-specific save semantics; successful save completion does not by itself prove a complete restore path.",
            "IBM i 7.6 backup and restore enhancements",
            SupportCategory::Software,
        ),
        claim(
            "legacy:ibmi:tcpip-connectivity",
            "ibm:ibmi-tcpip-connectivity@7.6",
            "IBM TCP/IP Connectivity for i is a platform component used by networked IBM i services; interface, route, service-job and application health must be distinguished during network diagnosis.",
            "Software requirements",
            SupportCategory::Network,
        ),
        claim(
            "legacy:ibmi:cl-control-surface",
            "ibm:ibmi-cl-overview@7.6",
            "IBM i Control Language is a primary system administration and programming control surface; command availability or syntax does not itself establish authority, safety, or applicability to the active object/job context.",
            "Control Language",
            SupportCategory::Software,
        ),
    ];
    for claim in claims {
        pack.sources.register_claim(claim)?;
    }
    Ok(())
}

fn claim(
    id: &str,
    snapshot: &str,
    statement: &str,
    section: &str,
    category: SupportCategory,
) -> TechnicalKnowledgeClaimV1 {
    TechnicalKnowledgeClaimV1 {
        id: TechnicalClaimIdV1(id.into()),
        statement: statement.into(),
        source_snapshot: SourceSnapshotIdV1(snapshot.into()),
        locator: Some(TechnicalSourceLocatorV1 {
            section: Some(section.into()),
            fragment: None,
        }),
        modality: ClaimModalityV1::Descriptive,
        applicability: Some(ibmi_scope()),
        extraction_quality: Some(0.95),
        category: Some(category),
    }
}

fn ibmi_scope() -> ApplicabilityScopeV1 {
    ApplicabilityScopeV1 {
        ecosystem: StringSelectorV1::Exact("legacy-enterprise-os".into()),
        vendor: StringSelectorV1::Exact("IBM".into()),
        product: StringSelectorV1::Exact("IBM i".into()),
        version: StringSelectorV1::Prefix("7.6".into()),
        ..ApplicabilityScopeV1::default()
    }
}

fn update_ibmi_profile(
    pack: &mut LegacyComputingPackV1,
    snapshots: &BTreeSet<SourceSnapshotIdV1>,
) -> Result<(), LegacyIbmiErrorV1> {
    let profile = pack
        .profiles
        .iter_mut()
        .find(|profile| profile.platform == LegacyPlatformV1::IbmI)
        .ok_or(LegacyIbmiErrorV1::MissingIbmiProfile)?;
    profile.source_snapshots.extend(snapshots.iter().cloned());
    for area in [
        LegacyKnowledgeAreaV1::WorkloadAndJobs,
        LegacyKnowledgeAreaV1::Storage,
        LegacyKnowledgeAreaV1::Networking,
        LegacyKnowledgeAreaV1::IdentityAndSecurity,
        LegacyKnowledgeAreaV1::ObservabilityAndProblemManagement,
        LegacyKnowledgeAreaV1::AvailabilityAndRecovery,
        LegacyKnowledgeAreaV1::SoftwareLifecycle,
        LegacyKnowledgeAreaV1::InteroperabilityAndMigration,
    ] {
        profile.coverage.insert(area, LegacyCoverageStateV1::ClaimSeeded);
    }
    for old in [
        "ibm-i-object-model",
        "ibm-i-jobs-subsystems",
        "ibm-i-db2",
        "ibm-i-security",
        "ibm-i-save-restore",
        "ibm-i-networking",
    ] {
        profile.gap_tags.remove(old);
    }
    profile.gap_tags.extend([
        "ibm-i-object-library-resolution-depth".into(),
        "ibm-i-subsystem-job-queue-diagnostics".into(),
        "db2-for-i-journal-lock-plan-depth".into(),
        "ibm-i-authority-administration-diagnostics".into(),
        "ibm-i-save-restore-recovery-labs".into(),
        "ibm-i-tcpip-service-diagnostics".into(),
        "ibm-i-cl-programming-depth".into(),
        "ibm-i-iasp-lpar-ha-depth".into(),
    ]);
    Ok(())
}

fn add_ibmi_procedures(
    pack: &mut LegacyComputingPackV1,
    snapshots: &BTreeSet<SourceSnapshotIdV1>,
) -> Result<BTreeSet<String>, LegacyIbmiErrorV1> {
    let procedures = vec![
        procedure(
            "legacy:ibmi:triage-object-resolution",
            LegacyKnowledgeAreaV1::SoftwareLifecycle,
            LegacyProcedureKindV1::Diagnose,
            "Triage IBM i object/library resolution",
            vec![
                "Establish the exact object name, object type, library qualifier and current job/library-list context.",
                "Inspect object existence, containing library, owner/type metadata and library-list resolution without changing the object.",
                "Compare program/object references with the currently resolved object and identify stale or ambiguous library-list resolution.",
                "If a namespace or library-list change appears necessary, produce an operator-reviewed change proposal with rollback and verification criteria.",
            ],
        ),
        procedure(
            "legacy:ibmi:triage-job-queue-subsystem",
            LegacyKnowledgeAreaV1::WorkloadAndJobs,
            LegacyProcedureKindV1::Diagnose,
            "Triage IBM i queued or stalled work",
            vec![
                "Establish the exact job, job queue, subsystem description and current queue/subsystem state.",
                "Inspect held/released/scheduled jobs, queue limits, subsystem active-job limits and current active jobs read-only.",
                "Distinguish application failure from work that is valid but not selected because of subsystem/job-queue state or limits.",
                "Any queue release, subsystem change or limit adjustment remains an operator-reviewed change proposal.",
            ],
        ),
        procedure(
            "legacy:ibmi:triage-db2-for-i",
            LegacyKnowledgeAreaV1::Storage,
            LegacyProcedureKindV1::Diagnose,
            "Triage Db2 for i availability and data-path failures",
            vec![
                "Establish IBM i release/PTF context, local relational database identity and affected library/schema/object.",
                "Inspect database object state, locks, journal relationships and query/service evidence without changing schema or data.",
                "Separate local object/database health from DRDA/DDM or remote relational-database connectivity.",
                "Any index, journal, schema, data or package change remains proposal-only and must include rollback and verification.",
            ],
        ),
        procedure(
            "legacy:ibmi:triage-authority-denial",
            LegacyKnowledgeAreaV1::IdentityAndSecurity,
            LegacyProcedureKindV1::Diagnose,
            "Triage IBM i object authority denial",
            vec![
                "Establish the exact user profile, object, library, object type and requested operation.",
                "Inspect object owner, private/public/group/authorization-list authority and relevant special-authority context read-only.",
                "Use authority-collection evidence when available to distinguish required authority from unrelated privilege.",
                "Never treat *ALLOBJ or other broad special-authority assignment as a diagnostic shortcut; propose the narrowest reviewed change only if required.",
            ],
        ),
        procedure(
            "legacy:ibmi:triage-save-restore",
            LegacyKnowledgeAreaV1::AvailabilityAndRecovery,
            LegacyProcedureKindV1::Recover,
            "Triage IBM i save/restore recoverability",
            vec![
                "Establish exact save scope, object types, library/IASP context, save media/save-file identity and timestamps.",
                "Inspect save history, object coverage, access-path/journal dependencies and restore prerequisites read-only.",
                "Separate a successful save operation from proof that the intended object set and dependencies can be restored.",
                "Any production restore remains proposal-only and requires explicit target, rollback/fallback, verification and blast-radius review.",
            ],
        ),
        procedure(
            "legacy:ibmi:triage-tcpip-service",
            LegacyKnowledgeAreaV1::Networking,
            LegacyProcedureKindV1::Diagnose,
            "Triage IBM i TCP/IP service reachability",
            vec![
                "Establish the exact interface/address family, route, target service and server-job/application context.",
                "Inspect interface, route and service-job state plus bounded reachability evidence without changing configuration.",
                "Distinguish transport/path failure from a healthy network path with an unhealthy or unauthorized application service.",
                "Any route/interface/service restart or configuration change remains proposal-only.",
            ],
        ),
    ];

    let mut ids = BTreeSet::new();
    for mut procedure in procedures {
        procedure.source_snapshots = snapshots.clone();
        if let Some(existing) = pack.procedures.iter().find(|p| p.id == procedure.id) {
            if existing != &procedure {
                return Err(LegacyIbmiErrorV1::ProcedureIdentityConflict(procedure.id));
            }
        } else {
            procedure.validate(&pack.sources)?;
            pack.procedures.push(procedure.clone());
        }
        ids.insert(procedure.id);
    }
    Ok(ids)
}

fn procedure(
    id: &str,
    area: LegacyKnowledgeAreaV1,
    kind: LegacyProcedureKindV1,
    title: &str,
    descriptions: Vec<&str>,
) -> LegacyProcedureV1 {
    let steps = descriptions
        .into_iter()
        .enumerate()
        .map(|(index, description)| LegacyProcedureStepV1 {
            authority: if index + 1 == 4 {
                LegacyProcedureAuthorityV1::ChangeProposalOnly
            } else {
                LegacyProcedureAuthorityV1::ReadOnlyObservation
            },
            description: description.into(),
        })
        .collect();
    LegacyProcedureV1 {
        id: id.into(),
        platform: LegacyPlatformV1::IbmI,
        area,
        kind,
        title: title.into(),
        applicability: ibmi_scope(),
        preconditions: vec![
            "Exact IBM i release and target system identity are established.".into(),
            "Operator authority and diagnostic scope are known before any active change is proposed.".into(),
        ],
        steps,
        verification: vec![
            "Re-check the original symptom using the same scoped identity and evidence path.".into(),
            "Confirm no unrelated object/job/service scope was modified by diagnosis.".into(),
        ],
        source_snapshots: BTreeSet::new(),
    }
}

fn seed_mechanisms() -> Vec<IbmiMechanismModelV1> {
    use IbmiMechanismKindV1 as M;
    vec![
        mechanism(
            M::ObjectLibraryNamespace,
            LegacyKnowledgeAreaV1::SoftwareLifecycle,
            "Object and library namespace",
            "Typed objects live in libraries and job/thread library-list context affects unqualified resolution.",
            &[],
            &["legacy:ibmi:object-library-namespace"],
            &[
                ("object-metadata", "Resolved object name/type/library/owner and object description metadata."),
                ("library-list", "Current library and user/system library-list resolution context."),
                ("program-references", "Program/object reference metadata compared with current object placement."),
            ],
            &[
                ("wrong-library-resolution", "An unqualified object name resolves to a different library/object than intended.", &["qualified object succeeds", "library-list order differs from expected"], &["moving or deleting objects before establishing resolution context"]),
                ("stale-program-reference", "Program reference metadata no longer matches the actual relocated or overridden object.", &["reference metadata differs from current object placement"], &["rebuilding unrelated programs before confirming the reference mismatch"]),
            ],
        ),
        mechanism(
            M::JobSubsystemWorkManagement,
            LegacyKnowledgeAreaV1::WorkloadAndJobs,
            "Jobs, job queues, and subsystems",
            "Submitted work can wait on job queues until a subsystem selects it; queue/subsystem state and limits are separate from application execution state.",
            &[],
            &["legacy:ibmi:subsystem-job-queue"],
            &[
                ("job-queue-state", "Held/released/scheduled counts and job queue status."),
                ("subsystem-state", "Subsystem description, active-job limits and current active jobs."),
                ("job-status", "Exact job status, queue, subsystem and function metadata."),
            ],
            &[
                ("queue-not-selected", "A valid submitted job remains queued because the queue/subsystem is held, inactive or constrained.", &["job remains queued without application execution evidence", "queue/subsystem limits or state explain non-selection"], &["restarting the application before confirming the job ever became active"]),
                ("subsystem-capacity", "Work waits because active-job or priority constraints are reached.", &["active jobs at configured limits", "queued work becomes eligible as capacity changes"], &["raising limits globally without workload-impact review"]),
            ],
        ),
        mechanism(
            M::Db2ForI,
            LegacyKnowledgeAreaV1::Storage,
            "Db2 for i",
            "Db2 for i is integrated into IBM i, while local database state, object locks/journaling and distributed relational connectivity remain distinct diagnostic layers.",
            &[M::ObjectLibraryNamespace, M::JobSubsystemWorkManagement],
            &["legacy:ibmi:db2-integrated", "legacy:ibmi:db2-distributed-connectivity"],
            &[
                ("database-object-state", "Library/schema/table/file/index/object metadata and availability."),
                ("locking-journaling", "Lock state, journal relationships and relevant service/query evidence."),
                ("rdb-connectivity", "Local relational database entry and DRDA/DDM/distributed connection evidence."),
            ],
            &[
                ("local-object-not-network", "A local database object/lock/journal problem is misdiagnosed as a remote network failure.", &["local object access fails with healthy remote path evidence", "object/lock/journal evidence explains failure"], &["changing routes or firewall state before local database evidence is checked"]),
                ("distributed-connectivity", "Local Db2 is healthy but remote relational-database connectivity or package/authentication state fails.", &["local SQL succeeds", "remote connection/package path fails"], &["reorganizing local tables to solve a distributed connection failure"]),
            ],
        ),
        mechanism(
            M::ObjectAuthority,
            LegacyKnowledgeAreaV1::IdentityAndSecurity,
            "Object authority and special authorities",
            "Object access depends on object authorities, ownership, profiles/groups, authorization lists and special authorities; authority collection can provide evidence without broadening privilege.",
            &[M::ObjectLibraryNamespace],
            &["legacy:ibmi:object-authority-model", "legacy:ibmi:authority-collection-observation"],
            &[
                ("object-authority", "Owner, public/private/group authority, authorization list and requested operation."),
                ("user-profile", "User/group profile and special-authority context."),
                ("authority-collection", "Collected authority-check evidence for user/object access paths."),
            ],
            &[
                ("insufficient-object-authority", "A user can resolve an object but lacks authority for the requested operation.", &["object exists and resolves", "authority evidence denies the requested operation"], &["granting *ALLOBJ", "broadening public authority without identifying the missing right"]),
                ("library-versus-object-authority", "Library traversal or object authority is missing while the adjacent layer is healthy.", &["library and object authorities differ in the failing path"], &["changing both library and object permissions before isolating the missing layer"]),
            ],
        ),
        mechanism(
            M::SaveRestore,
            LegacyKnowledgeAreaV1::AvailabilityAndRecovery,
            "Save and restore",
            "Recoverability depends on what objects and dependencies were actually saved, the target/IASP context and restore semantics—not just a successful save completion message.",
            &[M::ObjectLibraryNamespace, M::Db2ForI],
            &["legacy:ibmi:save-restore-semantics"],
            &[
                ("save-history", "Save timestamp/scope/media or save-file identity and completion evidence."),
                ("object-coverage", "Exact objects, object types and dependent access paths/journals represented in the save."),
                ("restore-context", "Target library/IASP, authority and restore prerequisites."),
            ],
            &[
                ("successful-save-incomplete-recovery", "The save operation succeeded but omitted content or dependencies needed by the recovery objective.", &["save scope does not cover required objects/dependencies", "restore rehearsal exposes missing state"], &["performing production restore before confirming scope and target"]),
                ("restore-context-mismatch", "The retained save is valid but target library/IASP/object context differs from the recovery plan.", &["source save metadata is valid", "target context differs from original assumptions"], &["restoring over existing production objects without explicit target/blast-radius review"]),
            ],
        ),
        mechanism(
            M::TcpIpNetworking,
            LegacyKnowledgeAreaV1::Networking,
            "TCP/IP networking and service jobs",
            "Interface/route reachability, TCP transport and IBM i server/application job health are separate diagnostic layers.",
            &[M::JobSubsystemWorkManagement],
            &["legacy:ibmi:tcpip-connectivity"],
            &[
                ("interface-route", "Address-family, interface and routing evidence."),
                ("transport", "Bounded TCP reachability and connection evidence."),
                ("service-job", "IBM i server/application job and subsystem state for the target service."),
            ],
            &[
                ("network-up-service-down", "IP transport is healthy but the IBM i service/application job is not available.", &["route/transport succeeds", "service job is missing, unhealthy or unauthorized"], &["changing routes because an application service is down"]),
                ("partial-address-family-path", "One address family or path works while the affected family/path fails.", &["family/path-specific reachability differs", "service health is otherwise stable"], &["restarting all TCP/IP services before isolating the failing path"]),
            ],
        ),
        mechanism(
            M::ControlLanguage,
            LegacyKnowledgeAreaV1::SystemLifecycle,
            "Control Language",
            "CL is a principal IBM i control/programming surface, but a known command is only a syntax/capability fact—not authorization or safe-change approval.",
            &[M::ObjectLibraryNamespace, M::ObjectAuthority],
            &["legacy:ibmi:cl-control-surface"],
            &[
                ("command-context", "Command, parameters, current job/library context and object targets."),
                ("authority-context", "Authority required by the command versus authority actually held."),
                ("message-context", "Command completion/escape messages and job-log evidence."),
            ],
            &[
                ("valid-command-wrong-context", "A syntactically valid CL operation targets the wrong object/library/job context.", &["command syntax is valid", "resolved target/context differs from intended target"], &["reissuing a mutating command before confirming the exact resolved target"]),
                ("authority-versus-syntax", "A command exists and is correctly formed but the caller lacks the required authority.", &["command parsing succeeds", "authority evidence explains denial"], &["granting broad special authority to make the command succeed"]),
            ],
        ),
    ]
}

fn mechanism(
    kind: IbmiMechanismKindV1,
    area: LegacyKnowledgeAreaV1,
    title: &str,
    summary: &str,
    dependencies: &[IbmiMechanismKindV1],
    source_claims: &[&str],
    evidence: &[(&str, &str)],
    failures: &[(&str, &str, &[&str], &[&str])],
) -> IbmiMechanismModelV1 {
    IbmiMechanismModelV1 {
        kind,
        area,
        title: title.into(),
        summary: summary.into(),
        dependencies: dependencies.iter().copied().collect(),
        source_claims: source_claims
            .iter()
            .map(|id| TechnicalClaimIdV1((*id).into()))
            .collect(),
        evidence_signals: evidence
            .iter()
            .map(|(id, description)| IbmiEvidenceSignalV1 {
                id: (*id).into(),
                description: (*description).into(),
            })
            .collect(),
        failure_modes: failures
            .iter()
            .map(|(id, symptom, discriminators, unsafe_shortcuts)| IbmiFailureModeV1 {
                id: (*id).into(),
                symptom: (*symptom).into(),
                discriminators: discriminators.iter().map(|v| (*v).into()).collect(),
                unsafe_shortcuts: unsafe_shortcuts.iter().map(|v| (*v).into()).collect(),
            })
            .collect(),
    }
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), LegacyIbmiErrorV1> {
    if value.trim().is_empty() {
        Err(LegacyIbmiErrorV1::InvalidField(field.into()))
    } else {
        Ok(())
    }
}

#[derive(Debug)]
pub enum LegacyIbmiErrorV1 {
    LegacyPack(LegacyComputingErrorV1),
    Standards(StandardsRegistryErrorV1),
    UnsupportedSchema(String),
    InvalidField(String),
    MissingIbmiProfile,
    DuplicateMechanism(IbmiMechanismKindV1),
    DuplicateEvidenceId(String),
    DuplicateFailureMode(String),
    UnknownClaim(TechnicalClaimIdV1),
    UnknownProcedure(String),
    ProcedureIdentityConflict(String),
}

impl fmt::Display for LegacyIbmiErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LegacyPack(err) => write!(f, "invalid legacy pack: {err}"),
            Self::Standards(err) => write!(f, "invalid IBM i source registry data: {err}"),
            Self::UnsupportedSchema(value) => write!(f, "unsupported IBM i schema {value}"),
            Self::InvalidField(value) => write!(f, "invalid IBM i field: {value}"),
            Self::MissingIbmiProfile => write!(f, "legacy pack is missing the IBM i profile"),
            Self::DuplicateMechanism(kind) => write!(f, "duplicate IBM i mechanism {kind:?}"),
            Self::DuplicateEvidenceId(id) => write!(f, "duplicate IBM i evidence id {id}"),
            Self::DuplicateFailureMode(id) => write!(f, "duplicate IBM i failure mode {id}"),
            Self::UnknownClaim(id) => write!(f, "unknown IBM i source claim {}", id.0),
            Self::UnknownProcedure(id) => write!(f, "unknown IBM i procedure {id}"),
            Self::ProcedureIdentityConflict(id) => {
                write!(f, "IBM i procedure identity conflict for {id}")
            }
        }
    }
}

impl Error for LegacyIbmiErrorV1 {}

impl From<LegacyComputingErrorV1> for LegacyIbmiErrorV1 {
    fn from(value: LegacyComputingErrorV1) -> Self {
        Self::LegacyPack(value)
    }
}

impl From<StandardsRegistryErrorV1> for LegacyIbmiErrorV1 {
    fn from(value: StandardsRegistryErrorV1) -> Self {
        Self::Standards(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seed_legacy_computing_pack_v1;

    #[test]
    fn enrichment_is_idempotent_for_same_source_snapshot_time() {
        let mut pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        let first = enrich_legacy_ibmi_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        let claims = pack.sources.claims().count();
        let procedures = pack.procedures.len();
        let second = enrich_legacy_ibmi_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        assert_eq!(first, second);
        assert_eq!(pack.sources.claims().count(), claims);
        assert_eq!(pack.procedures.len(), procedures);
    }

    #[test]
    fn ibm_i_areas_advance_but_explicit_depth_gaps_remain() {
        let mut pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        enrich_legacy_ibmi_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        let profile = pack.profile(LegacyPlatformV1::IbmI).unwrap();
        assert_eq!(
            profile.state(LegacyKnowledgeAreaV1::IdentityAndSecurity),
            LegacyCoverageStateV1::ClaimSeeded
        );
        assert!(profile
            .gap_tags
            .contains("ibm-i-authority-administration-diagnostics"));
        assert!(profile.gap_tags.contains("ibm-i-save-restore-recovery-labs"));
    }

    #[test]
    fn every_mechanism_is_source_bound_and_discriminative() {
        let mut pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        let foundation =
            enrich_legacy_ibmi_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        assert_eq!(foundation.mechanisms.len(), 7);
        for mechanism in foundation.mechanisms {
            assert!(!mechanism.source_claims.is_empty());
            assert!(!mechanism.evidence_signals.is_empty());
            assert!(mechanism
                .failure_modes
                .iter()
                .all(|failure| !failure.discriminators.is_empty()));
        }
    }

    #[test]
    fn procedures_never_mint_execution_authority() {
        let mut pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        let foundation =
            enrich_legacy_ibmi_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        for id in foundation.procedure_ids {
            let procedure = pack.procedures.iter().find(|p| p.id == id).unwrap();
            assert!(procedure.steps.iter().all(|step| matches!(
                step.authority,
                LegacyProcedureAuthorityV1::ReadOnlyObservation
                    | LegacyProcedureAuthorityV1::ChangeProposalOnly
            )));
        }
    }

    #[test]
    fn db2_local_state_and_distributed_connectivity_stay_distinct() {
        let mut pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        let foundation =
            enrich_legacy_ibmi_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        let db2 = foundation
            .mechanisms
            .iter()
            .find(|m| m.kind == IbmiMechanismKindV1::Db2ForI)
            .unwrap();
        assert!(db2.failure_modes.iter().any(|f| f.id == "local-object-not-network"));
        assert!(db2
            .failure_modes
            .iter()
            .any(|f| f.id == "distributed-connectivity"));
    }
}
