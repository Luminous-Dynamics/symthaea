// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Source-bound z/OS foundation for legacy enterprise IT reasoning.
//!
//! This module deepens the z/OS slice of `LegacyComputingPackV1` with explicit
//! mechanisms, evidence signals, failure modes, and non-executable diagnostic
//! procedures. All seeded technical propositions are bound to IBM z/OS 3.2
//! documentation snapshots; the snapshots remain metadata-only until a separate
//! source-capture process freezes exact source artifacts.

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

pub const LEGACY_ZOS_FOUNDATION_SCHEMA_V1: &str = "symthaea-it-legacy-zos-foundation-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ZosMechanismKindV1 {
    Jcl,
    Jes2,
    DfsmsVsam,
    Racf,
    Sysplex,
    CommunicationsServerTcpIp,
    VtamSna,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZosEvidenceSignalV1 {
    pub id: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZosFailureModeV1 {
    pub id: String,
    pub symptom: String,
    /// Evidence that helps distinguish this failure from adjacent hypotheses.
    pub discriminators: Vec<String>,
    /// Unsafe or over-broad shortcuts that should not be treated as diagnosis.
    pub unsafe_shortcuts: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZosMechanismModelV1 {
    pub kind: ZosMechanismKindV1,
    pub area: LegacyKnowledgeAreaV1,
    pub title: String,
    pub summary: String,
    pub dependencies: BTreeSet<ZosMechanismKindV1>,
    pub source_claims: BTreeSet<TechnicalClaimIdV1>,
    pub evidence_signals: Vec<ZosEvidenceSignalV1>,
    pub failure_modes: Vec<ZosFailureModeV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZosFoundationV1 {
    pub schema_version: String,
    pub product_version: String,
    pub mechanisms: Vec<ZosMechanismModelV1>,
    pub procedure_ids: BTreeSet<String>,
}

impl ZosFoundationV1 {
    pub fn validate(&self, pack: &LegacyComputingPackV1) -> Result<(), LegacyZosErrorV1> {
        if self.schema_version != LEGACY_ZOS_FOUNDATION_SCHEMA_V1 {
            return Err(LegacyZosErrorV1::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        require_nonempty(&self.product_version, "z/OS product version")?;
        if self.mechanisms.is_empty() {
            return Err(LegacyZosErrorV1::InvalidField(
                "z/OS foundation requires mechanisms".into(),
            ));
        }

        let mut kinds = BTreeSet::new();
        for mechanism in &self.mechanisms {
            if !kinds.insert(mechanism.kind) {
                return Err(LegacyZosErrorV1::DuplicateMechanism(mechanism.kind));
            }
            require_nonempty(&mechanism.title, "z/OS mechanism title")?;
            require_nonempty(&mechanism.summary, "z/OS mechanism summary")?;
            if mechanism.source_claims.is_empty() {
                return Err(LegacyZosErrorV1::InvalidField(format!(
                    "z/OS mechanism {:?} requires source claims",
                    mechanism.kind
                )));
            }
            for claim_id in &mechanism.source_claims {
                if pack.sources.claim(claim_id).is_none() {
                    return Err(LegacyZosErrorV1::UnknownClaim(claim_id.clone()));
                }
            }
            if mechanism.evidence_signals.is_empty() {
                return Err(LegacyZosErrorV1::InvalidField(format!(
                    "z/OS mechanism {:?} requires evidence signals",
                    mechanism.kind
                )));
            }
            if mechanism.failure_modes.is_empty() {
                return Err(LegacyZosErrorV1::InvalidField(format!(
                    "z/OS mechanism {:?} requires failure modes",
                    mechanism.kind
                )));
            }
            let mut evidence_ids = BTreeSet::new();
            for evidence in &mechanism.evidence_signals {
                require_nonempty(&evidence.id, "z/OS evidence id")?;
                require_nonempty(&evidence.description, "z/OS evidence description")?;
                if !evidence_ids.insert(evidence.id.as_str()) {
                    return Err(LegacyZosErrorV1::DuplicateEvidenceId(
                        evidence.id.clone(),
                    ));
                }
            }
            let mut failure_ids = BTreeSet::new();
            for failure in &mechanism.failure_modes {
                require_nonempty(&failure.id, "z/OS failure mode id")?;
                require_nonempty(&failure.symptom, "z/OS failure mode symptom")?;
                if !failure_ids.insert(failure.id.as_str()) {
                    return Err(LegacyZosErrorV1::DuplicateFailureMode(
                        failure.id.clone(),
                    ));
                }
                if failure.discriminators.is_empty() {
                    return Err(LegacyZosErrorV1::InvalidField(format!(
                        "z/OS failure mode {} requires discriminators",
                        failure.id
                    )));
                }
            }
        }

        let procedure_ids: BTreeSet<_> = pack.procedures.iter().map(|p| p.id.as_str()).collect();
        for procedure_id in &self.procedure_ids {
            if !procedure_ids.contains(procedure_id.as_str()) {
                return Err(LegacyZosErrorV1::UnknownProcedure(procedure_id.clone()));
            }
        }
        Ok(())
    }
}

/// Enrich an existing legacy pack with a conservative z/OS 3.2 foundation.
/// Exact replay is idempotent because the underlying source registry identities
/// and procedure IDs are stable and conflict-detecting.
pub fn enrich_legacy_zos_foundation_v1(
    pack: &mut LegacyComputingPackV1,
    fetched_at_unix_ms: u64,
) -> Result<ZosFoundationV1, LegacyZosErrorV1> {
    if fetched_at_unix_ms == 0 {
        return Err(LegacyZosErrorV1::InvalidField(
            "z/OS source fetch timestamp must be non-zero".into(),
        ));
    }
    pack.validate()?;

    let snapshots = register_zos_sources(pack, fetched_at_unix_ms)?;
    register_zos_claims(pack)?;
    let procedure_ids = add_zos_procedures(pack, &snapshots)?;
    update_zos_profile(pack, &snapshots)?;

    let foundation = ZosFoundationV1 {
        schema_version: LEGACY_ZOS_FOUNDATION_SCHEMA_V1.into(),
        product_version: "3.2".into(),
        mechanisms: seed_mechanisms(),
        procedure_ids,
    };
    pack.validate()?;
    foundation.validate(pack)?;
    Ok(foundation)
}

fn register_zos_sources(
    pack: &mut LegacyComputingPackV1,
    fetched_at_unix_ms: u64,
) -> Result<BTreeSet<SourceSnapshotIdV1>, LegacyZosErrorV1> {
    let definitions = [
        (
            "ibm:zos-jcl-reference",
            "ibm:zos-jcl-reference@3.2",
            "z/OS MVS JCL Reference",
            "z/OS 3.2 MVS JCL Reference",
            "https://www.ibm.com/docs/en/zos/3.2.0?topic=mvs-zos-jcl-reference",
        ),
        (
            "ibm:zos-jes2-introduction",
            "ibm:zos-jes2-introduction@3.2",
            "z/OS JES2 introduction",
            "z/OS 3.2 JES2 job processing",
            "https://www.ibm.com/docs/en/zos/3.2.0?topic=jes2-what-is-jes",
        ),
        (
            "ibm:zos-dfsms-vsam",
            "ibm:zos-dfsms-vsam@3.2",
            "z/OS DFSMS VSAM data sets",
            "z/OS 3.2 DFSMS VSAM data sets",
            "https://www.ibm.com/docs/en/zos/3.2.0?topic=files-defining-vsam-data-sets",
        ),
        (
            "ibm:zos-racf-overview",
            "ibm:zos-racf-overview@3.2",
            "z/OS Security Server RACF",
            "z/OS 3.2 Security Server RACF",
            "https://www.ibm.com/docs/en/zos/3.2.0?topic=zos-security-server-racf",
        ),
        (
            "ibm:zos-sysplex-characteristics",
            "ibm:zos-sysplex-characteristics@3.2",
            "z/OS sysplex characteristics",
            "z/OS 3.2 sysplex and coupling facility characteristics",
            "https://www.ibm.com/docs/en/zos/3.2.0?topic=introduction-characteristics-sysplex",
        ),
        (
            "ibm:zos-communications-server",
            "ibm:zos-communications-server@3.2",
            "z/OS Communications Server",
            "z/OS 3.2 Communications Server introduction",
            "https://www.ibm.com/docs/en/zos/3.2.0?topic=functions-introduction-zos-communications-server",
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
            version: Some("3.2".into()),
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

fn register_zos_claims(pack: &mut LegacyComputingPackV1) -> Result<(), LegacyZosErrorV1> {
    let claims = [
        claim(
            "legacy:zos:jcl-control-model",
            "ibm:zos-jcl-reference@3.2",
            "JCL defines job-control statements used to enter jobs, control job processing, and request resources; the JCL reference also covers JES2/JES3 job-entry control statements.",
            "Purpose of this information",
            SupportCategory::Software,
        ),
        claim(
            "legacy:zos:jes2-job-lifecycle",
            "ibm:zos-jes2-introduction@3.2",
            "JES receives jobs, schedules them for MVS processing, and controls job output; JES2 manages work before and after program execution while the MVS base control program manages execution itself.",
            "What is a JES?",
            SupportCategory::Software,
        ),
        claim(
            "legacy:zos:vsam-catalog-definition",
            "ibm:zos-dfsms-vsam@3.2",
            "VSAM data sets are cataloged and can be defined through Access Method Services or supported allocation paths; catalog and data-set inspection are part of problem identification and verification.",
            "Defining VSAM data sets",
            SupportCategory::Software,
        ),
        claim(
            "legacy:zos:racf-access-control",
            "ibm:zos-racf-overview@3.2",
            "RACF is the z/OS Security Server access-control facility for protected resources and provides administration, auditing, diagnosis, messages, and interface documentation.",
            "z/OS Security Server RACF",
            SupportCategory::Security,
        ),
        claim(
            "legacy:zos:sysplex-shared-state",
            "ibm:zos-sysplex-characteristics@3.2",
            "A z/OS sysplex uses XCF communication, global resource serialization, signaling paths, shared couple data, and optionally coupling facilities for high-speed shared data and coordination across systems.",
            "Characteristics of a sysplex",
            SupportCategory::Software,
        ),
        claim(
            "legacy:zos:communications-server-dual-stack",
            "ibm:zos-communications-server@3.2",
            "z/OS Communications Server provides TCP/IP networking and SNA networking through VTAM; TCP/IP includes network/transport functions while VTAM covers SNA families including subarea, APPN, and HPR.",
            "Introduction to z/OS Communications Server",
            SupportCategory::Network,
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
        applicability: Some(zos_scope()),
        extraction_quality: Some(0.95),
        category: Some(category),
    }
}

fn zos_scope() -> ApplicabilityScopeV1 {
    ApplicabilityScopeV1 {
        ecosystem: StringSelectorV1::Exact("legacy-enterprise-os".into()),
        vendor: StringSelectorV1::Exact("IBM".into()),
        product: StringSelectorV1::Exact("z/OS".into()),
        version: StringSelectorV1::Prefix("3.2".into()),
        ..ApplicabilityScopeV1::default()
    }
}

fn update_zos_profile(
    pack: &mut LegacyComputingPackV1,
    snapshots: &BTreeSet<SourceSnapshotIdV1>,
) -> Result<(), LegacyZosErrorV1> {
    let profile = pack
        .profiles
        .iter_mut()
        .find(|profile| profile.platform == LegacyPlatformV1::Zos)
        .ok_or(LegacyZosErrorV1::MissingZosProfile)?;
    profile.source_snapshots.extend(snapshots.iter().cloned());
    for area in [
        LegacyKnowledgeAreaV1::WorkloadAndJobs,
        LegacyKnowledgeAreaV1::Storage,
        LegacyKnowledgeAreaV1::Networking,
        LegacyKnowledgeAreaV1::IdentityAndSecurity,
        LegacyKnowledgeAreaV1::AvailabilityAndRecovery,
    ] {
        profile.coverage.insert(area, LegacyCoverageStateV1::ClaimSeeded);
    }
    for old in [
        "jcl-and-batch",
        "dataset-and-vsam",
        "jes",
        "racf",
        "sysplex",
        "tcpip-and-sna",
    ] {
        profile.gap_tags.remove(old);
    }
    profile.gap_tags.extend([
        "jcl-jes-deep-diagnostics".into(),
        "vsam-catalog-rls-recovery".into(),
        "racf-admin-diagnosis".into(),
        "sysplex-xcf-cf-recovery".into(),
        "tcpip-vtam-sna-diagnostics".into(),
    ]);
    Ok(())
}

fn add_zos_procedures(
    pack: &mut LegacyComputingPackV1,
    snapshots: &BTreeSet<SourceSnapshotIdV1>,
) -> Result<BTreeSet<String>, LegacyZosErrorV1> {
    let procedure_specs = [
        (
            "legacy:zos:batch-job-triage-v1",
            LegacyKnowledgeAreaV1::WorkloadAndJobs,
            "z/OS batch job/JES2 triage",
            "Establish whether the job failed admission, conversion, scheduling, execution, or output processing before proposing a rerun or JCL change.",
            &["ibm:zos-jcl-reference@3.2", "ibm:zos-jes2-introduction@3.2"][..],
        ),
        (
            "legacy:zos:vsam-access-triage-v1",
            LegacyKnowledgeAreaV1::Storage,
            "z/OS VSAM access/allocation triage",
            "Distinguish catalog/definition, allocation, sharing, authorization, and application-open failures before proposing storage changes.",
            &["ibm:zos-dfsms-vsam@3.2"][..],
        ),
        (
            "legacy:zos:racf-denial-triage-v1",
            LegacyKnowledgeAreaV1::IdentityAndSecurity,
            "z/OS RACF access-denial triage",
            "Establish the effective security context, protected resource, applicable profile, and audit evidence before proposing any authorization change.",
            &["ibm:zos-racf-overview@3.2"][..],
        ),
        (
            "legacy:zos:sysplex-partial-failure-v1",
            LegacyKnowledgeAreaV1::AvailabilityAndRecovery,
            "z/OS sysplex partial-failure triage",
            "Separate signaling, couple-data-set, coupling-facility, serialization, and member-specific symptoms before proposing isolation or recovery action.",
            &["ibm:zos-sysplex-characteristics@3.2"][..],
        ),
        (
            "legacy:zos:communications-path-v1",
            LegacyKnowledgeAreaV1::Networking,
            "z/OS Communications Server path triage",
            "Identify whether the affected path is TCP/IP, VTAM/SNA, application binding, stack/policy, or external network state before proposing a network change.",
            &["ibm:zos-communications-server@3.2"][..],
        ),
    ];

    let mut ids = BTreeSet::new();
    for (id, area, title, information_goal, source_ids) in procedure_specs {
        let mut exact_sources = BTreeSet::new();
        for source_id in source_ids {
            let source = SourceSnapshotIdV1((*source_id).into());
            if !snapshots.contains(&source) {
                return Err(LegacyZosErrorV1::InvalidField(format!(
                    "z/OS procedure {id} references unregistered source snapshot {}",
                    source.0
                )));
            }
            exact_sources.insert(source);
        }
        let procedure = LegacyProcedureV1 {
            id: id.into(),
            platform: LegacyPlatformV1::Zos,
            area,
            kind: LegacyProcedureKindV1::Diagnose,
            title: title.into(),
            applicability: zos_scope(),
            preconditions: vec![
                "Confirm z/OS release/maintenance context and the affected system or sysplex member.".into(),
                "Preserve the current failure evidence before any separately authorized change.".into(),
            ],
            steps: vec![
                LegacyProcedureStepV1 {
                    authority: LegacyProcedureAuthorityV1::ReadOnlyObservation,
                    description: information_goal.into(),
                },
                LegacyProcedureStepV1 {
                    authority: LegacyProcedureAuthorityV1::ReadOnlyObservation,
                    description: "Correlate platform-native status/messages with recent changes and source-version applicability; treat stale or cross-member evidence as potentially misleading.".into(),
                },
                LegacyProcedureStepV1 {
                    authority: LegacyProcedureAuthorityV1::ChangeProposalOnly,
                    description: "Formulate only the smallest reversible operator-reviewed intervention supported by the evidence, with rollback and post-change verification.".into(),
                },
            ],
            verification: vec![
                "Re-observe the original failure path using the same workload/resource identity.".into(),
                "Confirm adjacent workloads and shared sysplex/storage/network/security state did not regress.".into(),
            ],
            source_snapshots: exact_sources,
        };
        insert_procedure(pack, procedure)?;
        ids.insert(id.into());
    }
    Ok(ids)
}

fn insert_procedure(
    pack: &mut LegacyComputingPackV1,
    procedure: LegacyProcedureV1,
) -> Result<(), LegacyZosErrorV1> {
    if let Some(existing) = pack.procedures.iter().find(|item| item.id == procedure.id) {
        if existing == &procedure {
            return Ok(());
        }
        return Err(LegacyZosErrorV1::ProcedureIdentityConflict(procedure.id));
    }
    pack.procedures.push(procedure);
    Ok(())
}

fn seed_mechanisms() -> Vec<ZosMechanismModelV1> {
    vec![
        mechanism(
            ZosMechanismKindV1::Jcl,
            LegacyKnowledgeAreaV1::WorkloadAndJobs,
            "Job Control Language",
            "JCL describes batch work, execution steps, resource/data definitions, and job-control intent presented to z/OS/JES.",
            &[],
            &["legacy:zos:jcl-control-model"],
            &[
                ("jcl-structure", "JOB/step/data-definition structure and resolved procedure/input context"),
                ("jcl-resolution", "effective JCL after installation defaults and procedure expansion"),
            ],
            &[
                (
                    "jcl-definition-error",
                    "A job is rejected or fails before intended program execution because the effective job-control definition is invalid or inapplicable.",
                    &["job did not reach normal execution", "effective JCL differs from submitter assumption", "resource/data definition evidence points before program logic"],
                    &["blindly rerun the unchanged job", "edit production JCL before preserving the effective failing definition"],
                ),
            ],
        ),
        mechanism(
            ZosMechanismKindV1::Jes2,
            LegacyKnowledgeAreaV1::WorkloadAndJobs,
            "JES2 job lifecycle",
            "JES2 owns job input, queue/scheduling/spool and output responsibilities around MVS execution, with installation policy able to constrain submitter choices.",
            &[ZosMechanismKindV1::Jcl],
            &["legacy:zos:jes2-job-lifecycle"],
            &[
                ("jes2-queue-state", "job phase, queue/class/priority/output state and installation policy context"),
                ("jes2-output", "spool/output disposition and completion evidence"),
            ],
            &[
                (
                    "jes2-not-selected",
                    "A syntactically acceptable job remains queued or does not enter execution as expected.",
                    &["job is present in JES2", "execution has not begun", "class/priority/resource/policy state is relevant"],
                    &["raise priority globally", "purge and resubmit without identifying the selection constraint"],
                ),
            ],
        ),
        mechanism(
            ZosMechanismKindV1::DfsmsVsam,
            LegacyKnowledgeAreaV1::Storage,
            "DFSMS / VSAM data-set model",
            "VSAM uses cataloged data-set definitions and access-method semantics; storage diagnosis must distinguish catalog, definition/allocation, sharing, authorization, and application access.",
            &[],
            &["legacy:zos:vsam-catalog-definition"],
            &[
                ("vsam-catalog", "catalog identity, cluster/components, attributes and expected volume/storage context"),
                ("vsam-access", "open/allocation/sharing status and application access path"),
            ],
            &[
                (
                    "vsam-definition-or-catalog-drift",
                    "Application access fails because the cataloged object or effective data-set definition does not match the workload assumption.",
                    &["catalog/attribute evidence conflicts with workload expectation", "failure occurs before normal record processing", "other storage paths may remain healthy"],
                    &["delete/redefine the data set before preserving catalog and content state", "treat every VSAM failure as media corruption"],
                ),
            ],
        ),
        mechanism(
            ZosMechanismKindV1::Racf,
            LegacyKnowledgeAreaV1::IdentityAndSecurity,
            "RACF security decision model",
            "RACF evaluates access to protected resources under the effective z/OS security context; denial diagnosis must preserve resource/profile/audit evidence rather than defaulting to privilege expansion.",
            &[],
            &["legacy:zos:racf-access-control"],
            &[
                ("racf-context", "effective user/group/security context and workload origin"),
                ("racf-decision", "protected resource/profile relationship and security/audit evidence"),
            ],
            &[
                (
                    "racf-authorization-denial",
                    "A workload reaches a protected resource but is denied under its effective RACF security context.",
                    &["resource path is reachable", "security evidence identifies the protected resource/context", "peer workloads may differ by identity or profile"],
                    &["grant broad access to make the symptom disappear", "disable protection or auditing during diagnosis"],
                ),
            ],
        ),
        mechanism(
            ZosMechanismKindV1::Sysplex,
            LegacyKnowledgeAreaV1::AvailabilityAndRecovery,
            "Sysplex / XCF / coupling-facility coordination",
            "A sysplex coordinates multiple z/OS systems through XCF, serialization, signaling, shared couple data, and optionally coupling-facility structures for shared state and availability.",
            &[],
            &["legacy:zos:sysplex-shared-state"],
            &[
                ("sysplex-membership", "member identity/state and XCF/signaling health"),
                ("sysplex-couple-data", "couple-data-set availability/currentness and policy context"),
                ("coupling-facility", "CF connectivity/structure state for affected exploiters"),
            ],
            &[
                (
                    "sysplex-partial-coordination-loss",
                    "One or more members/subsystems lose shared-state or signaling behavior while other members continue service.",
                    &["symptom differs by member", "shared coordination evidence is degraded", "local system health alone does not explain the failure"],
                    &["restart multiple members simultaneously", "remove a member or shared structure before establishing quorum/shared-state impact"],
                ),
            ],
        ),
        mechanism(
            ZosMechanismKindV1::CommunicationsServerTcpIp,
            LegacyKnowledgeAreaV1::Networking,
            "Communications Server TCP/IP",
            "z/OS Communications Server supplies TCP/IP networking, including IPv4/IPv6, transport/network functions and application connectivity.",
            &[],
            &["legacy:zos:communications-server-dual-stack"],
            &[
                ("tcpip-stack", "stack/interface/address/route/listener/policy state relevant to the failing flow"),
                ("tcpip-path", "local and external reachability evidence separated from application/session state"),
            ],
            &[
                (
                    "tcpip-path-or-policy-failure",
                    "A TCP/IP application path fails even though the host or unrelated network services may remain available.",
                    &["failure is scoped to a flow/address/family/service", "stack/path/policy evidence differs from application-only hypotheses", "IPv4/IPv6 behavior may differ"],
                    &["restart the TCP/IP stack as a first diagnostic", "change routing/firewall/policy globally before isolating the affected path"],
                ),
            ],
        ),
        mechanism(
            ZosMechanismKindV1::VtamSna,
            LegacyKnowledgeAreaV1::Networking,
            "VTAM / SNA",
            "Communications Server also provides SNA networking through VTAM, including subarea, APPN and HPR families; an SNA/session fault is not automatically a TCP/IP fault.",
            &[],
            &["legacy:zos:communications-server-dual-stack"],
            &[
                ("vtam-session", "VTAM/SNA resource/session/path state for the affected application"),
                ("sna-family", "subarea/APPN/HPR context needed to interpret network behavior"),
            ],
            &[
                (
                    "sna-session-path-failure",
                    "A VTAM/SNA application session fails while TCP/IP services may remain healthy.",
                    &["affected application uses SNA/VTAM", "TCP/IP reachability does not establish SNA session health", "session/resource state is the discriminating evidence"],
                    &["treat successful IP ping as proof the SNA path is healthy", "restart unrelated TCP/IP services"],
                ),
            ],
        ),
    ]
}

fn mechanism(
    kind: ZosMechanismKindV1,
    area: LegacyKnowledgeAreaV1,
    title: &str,
    summary: &str,
    dependencies: &[ZosMechanismKindV1],
    claim_ids: &[&str],
    evidence: &[(&str, &str)],
    failures: &[(&str, &str, &[&str], &[&str])],
) -> ZosMechanismModelV1 {
    ZosMechanismModelV1 {
        kind,
        area,
        title: title.into(),
        summary: summary.into(),
        dependencies: dependencies.iter().copied().collect(),
        source_claims: claim_ids
            .iter()
            .map(|id| TechnicalClaimIdV1((*id).into()))
            .collect(),
        evidence_signals: evidence
            .iter()
            .map(|(id, description)| ZosEvidenceSignalV1 {
                id: (*id).into(),
                description: (*description).into(),
            })
            .collect(),
        failure_modes: failures
            .iter()
            .map(|(id, symptom, discriminators, unsafe_shortcuts)| ZosFailureModeV1 {
                id: (*id).into(),
                symptom: (*symptom).into(),
                discriminators: discriminators.iter().map(|v| (*v).into()).collect(),
                unsafe_shortcuts: unsafe_shortcuts.iter().map(|v| (*v).into()).collect(),
            })
            .collect(),
    }
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), LegacyZosErrorV1> {
    if value.trim().is_empty() {
        Err(LegacyZosErrorV1::InvalidField(field.into()))
    } else {
        Ok(())
    }
}

#[derive(Debug)]
pub enum LegacyZosErrorV1 {
    LegacyPack(LegacyComputingErrorV1),
    Standards(StandardsRegistryErrorV1),
    UnsupportedSchema(String),
    InvalidField(String),
    MissingZosProfile,
    DuplicateMechanism(ZosMechanismKindV1),
    DuplicateEvidenceId(String),
    DuplicateFailureMode(String),
    UnknownClaim(TechnicalClaimIdV1),
    UnknownProcedure(String),
    ProcedureIdentityConflict(String),
}

impl fmt::Display for LegacyZosErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LegacyPack(err) => write!(f, "invalid legacy pack: {err}"),
            Self::Standards(err) => write!(f, "z/OS source registry error: {err}"),
            Self::UnsupportedSchema(schema) => write!(f, "unsupported z/OS foundation schema {schema}"),
            Self::InvalidField(message) => write!(f, "invalid z/OS foundation field: {message}"),
            Self::MissingZosProfile => write!(f, "legacy pack lacks a z/OS profile"),
            Self::DuplicateMechanism(kind) => write!(f, "duplicate z/OS mechanism {kind:?}"),
            Self::DuplicateEvidenceId(id) => write!(f, "duplicate z/OS evidence id {id}"),
            Self::DuplicateFailureMode(id) => write!(f, "duplicate z/OS failure mode {id}"),
            Self::UnknownClaim(id) => write!(f, "unknown z/OS claim {}", id.0),
            Self::UnknownProcedure(id) => write!(f, "unknown z/OS procedure {id}"),
            Self::ProcedureIdentityConflict(id) => write!(f, "z/OS procedure identity conflict {id}"),
        }
    }
}

impl Error for LegacyZosErrorV1 {}

impl From<LegacyComputingErrorV1> for LegacyZosErrorV1 {
    fn from(value: LegacyComputingErrorV1) -> Self {
        Self::LegacyPack(value)
    }
}

impl From<StandardsRegistryErrorV1> for LegacyZosErrorV1 {
    fn from(value: StandardsRegistryErrorV1) -> Self {
        Self::Standards(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seed_legacy_computing_pack_v1;

    #[test]
    fn enrichment_is_idempotent_and_narrows_zos_gap_tags() {
        let mut pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        let first = enrich_legacy_zos_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        let second = enrich_legacy_zos_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        assert_eq!(first, second);
        let profile = pack.profile(LegacyPlatformV1::Zos).unwrap();
        assert_eq!(
            profile.state(LegacyKnowledgeAreaV1::WorkloadAndJobs),
            LegacyCoverageStateV1::ClaimSeeded
        );
        assert!(!profile.gap_tags.contains("jcl-and-batch"));
        assert!(profile.gap_tags.contains("jcl-jes-deep-diagnostics"));
    }

    #[test]
    fn foundation_keeps_tcpip_and_sna_distinct() {
        let mut pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        let foundation = enrich_legacy_zos_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        assert!(foundation
            .mechanisms
            .iter()
            .any(|m| m.kind == ZosMechanismKindV1::CommunicationsServerTcpIp));
        assert!(foundation
            .mechanisms
            .iter()
            .any(|m| m.kind == ZosMechanismKindV1::VtamSna));
    }

    #[test]
    fn all_zos_mechanisms_are_source_claim_bound_and_have_failure_discriminators() {
        let mut pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        let foundation = enrich_legacy_zos_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        foundation.validate(&pack).unwrap();
        for mechanism in &foundation.mechanisms {
            assert!(!mechanism.source_claims.is_empty());
            assert!(mechanism
                .failure_modes
                .iter()
                .all(|mode| !mode.discriminators.is_empty()));
        }
    }

    #[test]
    fn zos_procedures_never_mint_execution_authority() {
        let mut pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        let foundation = enrich_legacy_zos_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        for id in &foundation.procedure_ids {
            let procedure = pack.procedures.iter().find(|p| &p.id == id).unwrap();
            assert!(procedure.steps.iter().all(|step| matches!(
                step.authority,
                LegacyProcedureAuthorityV1::ReadOnlyObservation
                    | LegacyProcedureAuthorityV1::ChangeProposalOnly
            )));
        }
    }

    #[test]
    fn procedures_use_smallest_justified_source_sets() {
        let mut pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        enrich_legacy_zos_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        let expected = [
            ("legacy:zos:batch-job-triage-v1", &["ibm:zos-jcl-reference@3.2", "ibm:zos-jes2-introduction@3.2"][..]),
            ("legacy:zos:vsam-access-triage-v1", &["ibm:zos-dfsms-vsam@3.2"][..]),
            ("legacy:zos:racf-denial-triage-v1", &["ibm:zos-racf-overview@3.2"][..]),
            ("legacy:zos:sysplex-partial-failure-v1", &["ibm:zos-sysplex-characteristics@3.2"][..]),
            ("legacy:zos:communications-path-v1", &["ibm:zos-communications-server@3.2"][..]),
        ];
        for (procedure_id, source_ids) in expected {
            let procedure = pack.procedures.iter().find(|p| p.id == procedure_id).unwrap();
            let expected_sources: BTreeSet<_> = source_ids
                .iter()
                .map(|id| SourceSnapshotIdV1((*id).into()))
                .collect();
            assert_eq!(procedure.source_snapshots, expected_sources);
        }
    }
}
