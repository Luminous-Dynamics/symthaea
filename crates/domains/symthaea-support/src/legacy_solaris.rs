// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Source-bound Oracle Solaris foundation for legacy enterprise IT reasoning.
//!
//! This module deepens the Oracle Solaris 11.4 slice of `LegacyComputingPackV1`
//! with explicit ZFS, SMF, Zones, FMA, DTrace, IPMP/networking, and IPS/boot-
//! environment mechanisms. It remains advisory and non-executable; source
//! snapshots remain metadata-only until retained-artifact qualification freezes
//! exact source bytes.

use crate::knowledge_source::{
    KnowledgeAuthorityClassV1, KnowledgeLifecycleV1, KnowledgeStabilityV1,
};
use crate::legacy_computing::{
    LegacyComputingErrorV1, LegacyComputingPackV1, LegacyCoverageStateV1,
    LegacyKnowledgeAreaV1, LegacyPlatformV1, LegacyProcedureAuthorityV1,
    LegacyProcedureKindV1, LegacyProcedureStepV1, LegacyProcedureV1,
};
use crate::legacy_platform_identity::legacy_platform_scope_v1;
use crate::standards_registry::{
    ClaimModalityV1, SourceCaptureV1, SourceDocumentIdV1, SourceDocumentKindV1,
    SourceSnapshotIdV1, StandardsRegistryErrorV1, TechnicalClaimIdV1,
    TechnicalKnowledgeClaimV1, TechnicalPublisherV1, TechnicalSourceDocumentV1,
    TechnicalSourceLocatorV1, TechnicalSourceSnapshotV1,
};
use crate::types::SupportCategory;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const LEGACY_SOLARIS_FOUNDATION_SCHEMA_V1: &str =
    "symthaea-it-legacy-solaris-foundation-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SolarisMechanismKindV1 {
    Zfs,
    Smf,
    Zones,
    Fma,
    Dtrace,
    NetworkIpmp,
    IpsBootEnvironments,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolarisEvidenceSignalV1 {
    pub id: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolarisFailureModeV1 {
    pub id: String,
    pub symptom: String,
    pub discriminators: Vec<String>,
    pub unsafe_shortcuts: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolarisMechanismModelV1 {
    pub kind: SolarisMechanismKindV1,
    pub area: LegacyKnowledgeAreaV1,
    pub title: String,
    pub summary: String,
    pub dependencies: BTreeSet<SolarisMechanismKindV1>,
    pub source_claims: BTreeSet<TechnicalClaimIdV1>,
    pub evidence_signals: Vec<SolarisEvidenceSignalV1>,
    pub failure_modes: Vec<SolarisFailureModeV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolarisFoundationV1 {
    pub schema_version: String,
    pub product_version: String,
    pub mechanisms: Vec<SolarisMechanismModelV1>,
    pub procedure_ids: BTreeSet<String>,
}

impl SolarisFoundationV1 {
    pub fn validate(&self, pack: &LegacyComputingPackV1) -> Result<(), LegacySolarisErrorV1> {
        if self.schema_version != LEGACY_SOLARIS_FOUNDATION_SCHEMA_V1 {
            return Err(LegacySolarisErrorV1::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        if self.product_version.trim().is_empty() || self.mechanisms.is_empty() {
            return Err(LegacySolarisErrorV1::InvalidField(
                "Solaris foundation requires product version and mechanisms".into(),
            ));
        }

        let mut kinds = BTreeSet::new();
        for mechanism in &self.mechanisms {
            if !kinds.insert(mechanism.kind) {
                return Err(LegacySolarisErrorV1::DuplicateMechanism(mechanism.kind));
            }
            if mechanism.title.trim().is_empty()
                || mechanism.summary.trim().is_empty()
                || mechanism.source_claims.is_empty()
                || mechanism.evidence_signals.is_empty()
                || mechanism.failure_modes.is_empty()
            {
                return Err(LegacySolarisErrorV1::InvalidField(format!(
                    "Solaris mechanism {:?} is incomplete",
                    mechanism.kind
                )));
            }
            for claim_id in &mechanism.source_claims {
                if pack.sources.claim(claim_id).is_none() {
                    return Err(LegacySolarisErrorV1::UnknownClaim(claim_id.clone()));
                }
            }
            let mut evidence_ids = BTreeSet::new();
            for signal in &mechanism.evidence_signals {
                if signal.id.trim().is_empty() || signal.description.trim().is_empty() {
                    return Err(LegacySolarisErrorV1::InvalidField(
                        "Solaris evidence signal fields must be non-empty".into(),
                    ));
                }
                if !evidence_ids.insert(signal.id.as_str()) {
                    return Err(LegacySolarisErrorV1::DuplicateEvidenceId(signal.id.clone()));
                }
            }
            let mut failure_ids = BTreeSet::new();
            for failure in &mechanism.failure_modes {
                if failure.id.trim().is_empty()
                    || failure.symptom.trim().is_empty()
                    || failure.discriminators.is_empty()
                {
                    return Err(LegacySolarisErrorV1::InvalidField(
                        "Solaris failure mode requires id, symptom, and discriminators".into(),
                    ));
                }
                if !failure_ids.insert(failure.id.as_str()) {
                    return Err(LegacySolarisErrorV1::DuplicateFailureMode(
                        failure.id.clone(),
                    ));
                }
            }
        }

        let procedure_ids: BTreeSet<_> = pack.procedures.iter().map(|p| p.id.as_str()).collect();
        for id in &self.procedure_ids {
            if !procedure_ids.contains(id.as_str()) {
                return Err(LegacySolarisErrorV1::UnknownProcedure(id.clone()));
            }
        }
        Ok(())
    }
}

pub fn enrich_legacy_solaris_foundation_v1(
    pack: &mut LegacyComputingPackV1,
    fetched_at_unix_ms: u64,
) -> Result<SolarisFoundationV1, LegacySolarisErrorV1> {
    if fetched_at_unix_ms == 0 {
        return Err(LegacySolarisErrorV1::InvalidField(
            "Solaris source fetch timestamp must be non-zero".into(),
        ));
    }
    pack.validate()?;
    let snapshots = register_solaris_sources(pack, fetched_at_unix_ms)?;
    register_solaris_claims(pack)?;
    let procedure_ids = add_solaris_procedures(pack, &snapshots)?;
    update_solaris_profile(pack, &snapshots)?;

    let foundation = SolarisFoundationV1 {
        schema_version: LEGACY_SOLARIS_FOUNDATION_SCHEMA_V1.into(),
        product_version: "11.4".into(),
        mechanisms: seed_mechanisms(),
        procedure_ids,
    };
    pack.validate()?;
    foundation.validate(pack)?;
    Ok(foundation)
}

fn register_solaris_sources(
    pack: &mut LegacyComputingPackV1,
    fetched_at_unix_ms: u64,
) -> Result<BTreeSet<SourceSnapshotIdV1>, LegacySolarisErrorV1> {
    let definitions = [
        (
            "oracle:solaris-zfs",
            "oracle:solaris-zfs@11.4",
            "Managing Oracle Solaris ZFS File Systems",
            "Oracle Solaris 11.4 ZFS administration",
            "https://docs.oracle.com/en/operating-systems/solaris/oracle-solaris/11.4/manage-zfs/managing-oracle-solaris-zfs-file-systems.html",
        ),
        (
            "oracle:solaris-smf",
            "oracle:solaris-smf@11.4",
            "Managing System Services in Oracle Solaris 11.4",
            "Oracle Solaris 11.4 Service Management Facility",
            "https://docs.oracle.com/cd/E37838_01/html/E60998/index.html",
        ),
        (
            "oracle:solaris-zones",
            "oracle:solaris-zones@11.4",
            "Creating and Using Oracle Solaris Zones",
            "Oracle Solaris 11.4 zones administration",
            "https://docs.oracle.com/en/operating-systems/solaris/oracle-solaris/11.4/use-zones/index.html",
        ),
        (
            "oracle:solaris-fma",
            "oracle:solaris-fma@11.4",
            "Managing Faults, Defects, and Alerts in Oracle Solaris 11.4",
            "Oracle Solaris 11.4 Fault Management Architecture",
            "https://docs.oracle.com/cd/E37838_01/html/E61036/index.html",
        ),
        (
            "oracle:solaris-dtrace",
            "oracle:solaris-dtrace@11.4",
            "Oracle Solaris 11.4 DTrace Guide",
            "Oracle Solaris 11.4 dynamic tracing",
            "https://docs.oracle.com/en/operating-systems/solaris/oracle-solaris/11.4/dtrace-guide/index.html",
        ),
        (
            "oracle:solaris-network-ipmp",
            "oracle:solaris-network-ipmp@11.4",
            "Administering TCP/IP Networks, IPMP, and IP Tunnels",
            "Oracle Solaris 11.4 network and IPMP administration",
            "https://docs.oracle.com/cd/E37838_01/html/E60991/ipmpov.html",
        ),
        (
            "oracle:solaris-ips-be",
            "oracle:solaris-ips-be@11.4",
            "Updating Systems and Adding Software in Oracle Solaris 11.4",
            "Oracle Solaris 11.4 IPS and boot environments",
            "https://docs.oracle.com/en/operating-systems/solaris/oracle-solaris/11.4/update-sys-add-sw/image-packaging-system.html",
        ),
    ];

    let mut snapshots = BTreeSet::new();
    for (document_id, snapshot_id, title, canonical_ref, locator) in definitions {
        pack.sources.register_document(TechnicalSourceDocumentV1 {
            id: SourceDocumentIdV1(document_id.into()),
            publisher: TechnicalPublisherV1::Vendor("Oracle".into()),
            kind: SourceDocumentKindV1::VendorDocumentation,
            title: title.into(),
            canonical_ref: canonical_ref.into(),
            canonical_locator: Some(locator.into()),
        })?;
        pack.sources.register_snapshot(TechnicalSourceSnapshotV1 {
            id: SourceSnapshotIdV1(snapshot_id.into()),
            document_id: SourceDocumentIdV1(document_id.into()),
            version: Some("11.4".into()),
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

fn register_solaris_claims(pack: &mut LegacyComputingPackV1) -> Result<(), LegacySolarisErrorV1> {
    let claims = [
        claim(
            "legacy:solaris:zfs-dataset-model",
            "oracle:solaris-zfs@11.4",
            "Oracle Solaris ZFS organizes storage through pools and datasets; dataset is a generic term covering file systems, snapshots, clones, or volumes, with hierarchical properties and administrative state that must be distinguished from underlying device health.",
            "Managing Oracle Solaris ZFS File Systems",
            SupportCategory::Software,
        ),
        claim(
            "legacy:solaris:smf-service-graph",
            "oracle:solaris-smf@11.4",
            "SMF manages system and application services with explicit service states, dependencies, restarters, configuration repositories, and troubleshooting state such as offline, degraded, or maintenance.",
            "Introduction to the Service Management Facility",
            SupportCategory::Software,
        ),
        claim(
            "legacy:solaris:zones-virtualization",
            "oracle:solaris-zones@11.4",
            "Oracle Solaris Zones provide isolated operating-system environments under a global zone; zone configuration, runtime state, resource usage, SMF integration, and network mode are distinct diagnostic layers.",
            "Zones administration and monitoring",
            SupportCategory::Software,
        ),
        claim(
            "legacy:solaris:fma-diagnostic-lifecycle",
            "oracle:solaris-fma@11.4",
            "Oracle Solaris FMA receives structured error and information reports, diagnoses faults/defects/alerts, emits suspect lists, and uses response agents; an FMA diagnosis is evidence with its own lifecycle rather than an unconditional proof of the current root cause.",
            "Fault Management Overview",
            SupportCategory::Hardware,
        ),
        claim(
            "legacy:solaris:dtrace-dynamic-instrumentation",
            "oracle:solaris-dtrace@11.4",
            "DTrace dynamically enables probes in the kernel or user processes and removes instrumentation after tracing; probe observations can localize behavior but must be interpreted in workload and provider context.",
            "About DTrace and Getting Started",
            SupportCategory::Software,
        ),
        claim(
            "legacy:solaris:ipmp-interface-group",
            "oracle:solaris-network-ipmp@11.4",
            "IPMP groups multiple underlying IP interfaces behind an IPMP interface and distributes data addresses across active interfaces; group health and an individual link's state are not interchangeable.",
            "IPMP Support in Oracle Solaris",
            SupportCategory::Network,
        ),
        claim(
            "legacy:solaris:ips-boot-environment",
            "oracle:solaris-ips-be@11.4",
            "IPS manages packages and images, while a boot environment is a bootable instance of an image; package operations may create a new boot environment, so installed package state and the currently active boot environment must be distinguished.",
            "Image Packaging System and Boot Environments",
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
        applicability: Some(legacy_platform_scope_v1(LegacyPlatformV1::Solaris)),
        extraction_quality: Some(0.95),
        category: Some(category),
    }
}

fn seed_mechanisms() -> Vec<SolarisMechanismModelV1> {
    vec![
        mechanism(
            SolarisMechanismKindV1::Zfs,
            LegacyKnowledgeAreaV1::Storage,
            "ZFS pool/dataset state",
            "Separate pool/device health from dataset, mount, property, snapshot, quota, and boot-environment state.",
            &[],
            &["legacy:solaris:zfs-dataset-model"],
            &[("zpool-state", "Current pool/vdev health and error state"), ("dataset-state", "Dataset properties, mount state, space, snapshots, and inheritance")],
            &[("dataset-vs-device", "A dataset is unavailable or full while the pool/device layer remains healthy", &["Compare pool/vdev health with dataset/mount/property state"], &["Replace storage devices before establishing device failure"])],
        ),
        mechanism(
            SolarisMechanismKindV1::Smf,
            LegacyKnowledgeAreaV1::WorkloadAndJobs,
            "SMF service dependency graph",
            "Reason about service state, dependency satisfaction, restarters, logs, and maintenance/offline/degraded transitions rather than treating a stopped process as the entire service model.",
            &[],
            &["legacy:solaris:smf-service-graph"],
            &[("service-state", "SMF instance state and transition reason"), ("dependency-state", "Dependency and restarter state"), ("service-log", "Service-specific diagnostic log evidence")],
            &[("dependency-unsatisfied", "A service is offline because a dependency is unsatisfied", &["Inspect dependency graph and restarter before restarting the service"], &["Repeatedly clear maintenance or restart dependencies without identifying the failed dependency"])],
        ),
        mechanism(
            SolarisMechanismKindV1::Zones,
            LegacyKnowledgeAreaV1::VirtualizationAndPartitioning,
            "Zones virtualization boundary",
            "Separate global-zone state, zone configuration, zone runtime state, resource controls, networking mode, and services inside a zone.",
            &[SolarisMechanismKindV1::Smf],
            &["legacy:solaris:zones-virtualization"],
            &[("zoneadm-state", "Zone lifecycle/runtime state"), ("zonecfg-state", "Persistent zone configuration"), ("zonestat-state", "Zone resource-consumption evidence")],
            &[("zone-local-vs-global", "A workload fails in one zone while the global zone and peer zones remain healthy", &["Compare zone runtime/config/resource/network context before blaming the host"], &["Restart the global zone to repair a single-zone problem"])],
        ),
        mechanism(
            SolarisMechanismKindV1::Fma,
            LegacyKnowledgeAreaV1::ObservabilityAndProblemManagement,
            "Fault Management Architecture",
            "Treat FMA reports, diagnoses, suspect lists, UUIDs, and repair lifecycle as structured evidence that must be correlated with current symptoms.",
            &[],
            &["legacy:solaris:fma-diagnostic-lifecycle"],
            &[("fma-active", "Current active faults/defects/alerts and suspect list"), ("fma-history", "Fault lifecycle and historical UUID/event state")],
            &[("stale-diagnosis", "An older FMA diagnosis is present but does not align with the current failure window", &["Compare FMA lifecycle/timestamps with current affected resource and symptom"], &["Replace the historical suspect component without current corroboration"])],
        ),
        mechanism(
            SolarisMechanismKindV1::Dtrace,
            LegacyKnowledgeAreaV1::ObservabilityAndProblemManagement,
            "DTrace dynamic instrumentation",
            "Use provider/probe-specific tracing as bounded observational evidence while distinguishing observation from diagnosis and accounting for probe scope and workload context.",
            &[],
            &["legacy:solaris:dtrace-dynamic-instrumentation"],
            &[("probe-selection", "Provider/probe predicates and enabled scope"), ("trace-output", "Timestamped traced events tied to the active workload")],
            &[("trace-overgeneralization", "A trace observes one path or process and is generalized to the entire system", &["Bind probe/provider/process/time scope to the claim being tested"], &["Enable broad high-volume tracing without an information goal or resource bound"])],
        ),
        mechanism(
            SolarisMechanismKindV1::NetworkIpmp,
            LegacyKnowledgeAreaV1::Networking,
            "Solaris networking and IPMP",
            "Separate datalink/interface/address/route state from IPMP group health and from zone-visible network state.",
            &[],
            &["legacy:solaris:ipmp-interface-group"],
            &[("ipmp-group", "IPMP group/interface and underlying member health"), ("route-address", "Address, route, and path state for affected traffic")],
            &[("member-vs-group", "One IPMP member fails while the group continues forwarding", &["Inspect group-level failover/data-address placement before declaring outage"], &["Reconfigure all interfaces because one member reports a fault"])],
        ),
        mechanism(
            SolarisMechanismKindV1::IpsBootEnvironments,
            LegacyKnowledgeAreaV1::SoftwareLifecycle,
            "IPS image and boot-environment lifecycle",
            "Distinguish repository/publisher/package constraints, installed image state, inactive boot environments, and the currently active boot environment.",
            &[SolarisMechanismKindV1::Zfs],
            &["legacy:solaris:ips-boot-environment"],
            &[("ips-state", "Publisher, package, dependency, and image state"), ("be-state", "Active/inactive boot environments and package versions")],
            &[("inactive-be-confusion", "Expected package/version exists only in an inactive boot environment", &["Establish active BE and image/package state before changing publishers or packages"], &["Remove/reinstall packages before checking which BE is active"])],
        ),
    ]
}

fn mechanism(
    kind: SolarisMechanismKindV1,
    area: LegacyKnowledgeAreaV1,
    title: &str,
    summary: &str,
    dependencies: &[SolarisMechanismKindV1],
    claims: &[&str],
    evidence: &[(&str, &str)],
    failures: &[(&str, &str, &[&str], &[&str])],
) -> SolarisMechanismModelV1 {
    SolarisMechanismModelV1 {
        kind,
        area,
        title: title.into(),
        summary: summary.into(),
        dependencies: dependencies.iter().copied().collect(),
        source_claims: claims
            .iter()
            .map(|id| TechnicalClaimIdV1((*id).into()))
            .collect(),
        evidence_signals: evidence
            .iter()
            .map(|(id, description)| SolarisEvidenceSignalV1 {
                id: (*id).into(),
                description: (*description).into(),
            })
            .collect(),
        failure_modes: failures
            .iter()
            .map(|(id, symptom, discriminators, unsafe_shortcuts)| SolarisFailureModeV1 {
                id: (*id).into(),
                symptom: (*symptom).into(),
                discriminators: discriminators.iter().map(|v| (*v).into()).collect(),
                unsafe_shortcuts: unsafe_shortcuts.iter().map(|v| (*v).into()).collect(),
            })
            .collect(),
    }
}

fn add_solaris_procedures(
    pack: &mut LegacyComputingPackV1,
    snapshots: &BTreeSet<SourceSnapshotIdV1>,
) -> Result<BTreeSet<String>, LegacySolarisErrorV1> {
    let defs = [
        ("legacy:solaris:zfs-triage", LegacyKnowledgeAreaV1::Storage, LegacyProcedureKindV1::Diagnose, "Solaris ZFS layered triage", "Establish pool/vdev and dataset state separately before proposing storage changes."),
        ("legacy:solaris:smf-triage", LegacyKnowledgeAreaV1::WorkloadAndJobs, LegacyProcedureKindV1::Diagnose, "Solaris SMF dependency triage", "Inspect service state, dependencies, restarter, logs, and configuration before proposing state changes."),
        ("legacy:solaris:zones-triage", LegacyKnowledgeAreaV1::VirtualizationAndPartitioning, LegacyProcedureKindV1::Diagnose, "Solaris Zones scoped triage", "Separate global-zone health from affected-zone configuration, runtime, resources, networking, and in-zone service state."),
        ("legacy:solaris:fma-triage", LegacyKnowledgeAreaV1::ObservabilityAndProblemManagement, LegacyProcedureKindV1::Diagnose, "Solaris FMA evidence triage", "Correlate current FMA diagnoses and lifecycle with the incident window before proposing component repair/replacement."),
        ("legacy:solaris:dtrace-plan", LegacyKnowledgeAreaV1::ObservabilityAndProblemManagement, LegacyProcedureKindV1::Diagnose, "Solaris DTrace bounded diagnostic plan", "Define provider/probe, process/path, duration, and information goal before separately authorizing dynamic instrumentation."),
        ("legacy:solaris:ipmp-triage", LegacyKnowledgeAreaV1::Networking, LegacyProcedureKindV1::Diagnose, "Solaris IPMP/network triage", "Compare IPMP group/member, datalink, address, route, and zone-visible path state before proposing network changes."),
        ("legacy:solaris:ips-be-triage", LegacyKnowledgeAreaV1::SoftwareLifecycle, LegacyProcedureKindV1::Diagnose, "Solaris IPS/boot-environment triage", "Establish active boot environment, image, publisher, package, dependency, and update state before proposing package changes."),
    ];

    let mut ids = BTreeSet::new();
    for (id, area, kind, title, goal) in defs {
        let procedure = LegacyProcedureV1 {
            id: id.into(),
            platform: LegacyPlatformV1::Solaris,
            area,
            kind,
            title: title.into(),
            applicability: legacy_platform_scope_v1(LegacyPlatformV1::Solaris),
            preconditions: vec![
                "Confirm Oracle Solaris 11.4 platform/version applicability and preserve current incident evidence.".into(),
                "This advisory procedure does not authorize runtime or configuration changes.".into(),
            ],
            steps: vec![
                LegacyProcedureStepV1 {
                    authority: LegacyProcedureAuthorityV1::ReadOnlyObservation,
                    description: goal.into(),
                },
                LegacyProcedureStepV1 {
                    authority: LegacyProcedureAuthorityV1::ReadOnlyObservation,
                    description: "Compare fresh evidence with historical/stale evidence and adjacent mechanism layers before choosing a root-cause hypothesis.".into(),
                },
                LegacyProcedureStepV1 {
                    authority: LegacyProcedureAuthorityV1::ChangeProposalOnly,
                    description: "If a state-changing diagnostic or remediation is justified, formulate the smallest reversible operator-reviewed proposal with rollback and verification.".into(),
                },
            ],
            verification: vec![
                "Re-observe the original symptom and directly affected Solaris mechanism after any separately authorized intervention.".into(),
                "Verify adjacent ZFS/SMF/zone/FMA/network/boot-environment state did not regress.".into(),
            ],
            source_snapshots: snapshots.clone(),
        };
        insert_procedure_idempotent(pack, procedure)?;
        ids.insert(id.into());
    }
    Ok(ids)
}

fn insert_procedure_idempotent(
    pack: &mut LegacyComputingPackV1,
    procedure: LegacyProcedureV1,
) -> Result<(), LegacySolarisErrorV1> {
    if let Some(existing) = pack.procedures.iter().find(|p| p.id == procedure.id) {
        if existing != &procedure {
            return Err(LegacySolarisErrorV1::ProcedureConflict(procedure.id));
        }
        return Ok(());
    }
    pack.procedures.push(procedure);
    Ok(())
}

fn update_solaris_profile(
    pack: &mut LegacyComputingPackV1,
    snapshots: &BTreeSet<SourceSnapshotIdV1>,
) -> Result<(), LegacySolarisErrorV1> {
    let profile = pack
        .profiles
        .iter_mut()
        .find(|p| p.platform == LegacyPlatformV1::Solaris)
        .ok_or(LegacySolarisErrorV1::MissingSolarisProfile)?;
    profile.source_snapshots.extend(snapshots.iter().cloned());

    for area in [
        LegacyKnowledgeAreaV1::SystemLifecycle,
        LegacyKnowledgeAreaV1::WorkloadAndJobs,
        LegacyKnowledgeAreaV1::Storage,
        LegacyKnowledgeAreaV1::Networking,
        LegacyKnowledgeAreaV1::AvailabilityAndRecovery,
        LegacyKnowledgeAreaV1::VirtualizationAndPartitioning,
        LegacyKnowledgeAreaV1::SoftwareLifecycle,
    ] {
        let current = profile.state(area);
        if current < LegacyCoverageStateV1::ClaimSeeded {
            profile.coverage.insert(area, LegacyCoverageStateV1::ClaimSeeded);
        }
    }

    for retired in ["zones-failure-scenarios", "dtrace-diagnostics"] {
        profile.gap_tags.remove(retired);
    }
    for gap in [
        "zfs-recovery-labs",
        "smf-dependency-diagnostics",
        "zones-resource-network-diagnostics",
        "fma-hardware-fault-labs",
        "dtrace-provider-safety-depth",
        "ipmp-network-virtualization-depth",
        "ips-be-recovery-labs",
    ] {
        profile.gap_tags.insert(gap.into());
    }
    Ok(())
}

#[derive(Debug)]
pub enum LegacySolarisErrorV1 {
    Standards(StandardsRegistryErrorV1),
    Computing(LegacyComputingErrorV1),
    UnsupportedSchema(String),
    InvalidField(String),
    DuplicateMechanism(SolarisMechanismKindV1),
    DuplicateEvidenceId(String),
    DuplicateFailureMode(String),
    UnknownClaim(TechnicalClaimIdV1),
    UnknownProcedure(String),
    ProcedureConflict(String),
    MissingSolarisProfile,
}

impl fmt::Display for LegacySolarisErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Standards(err) => write!(f, "Solaris source registry error: {err}"),
            Self::Computing(err) => write!(f, "Solaris legacy-pack error: {err}"),
            Self::UnsupportedSchema(schema) => write!(f, "unsupported Solaris schema {schema}"),
            Self::InvalidField(message) => write!(f, "invalid Solaris foundation: {message}"),
            Self::DuplicateMechanism(kind) => write!(f, "duplicate Solaris mechanism {kind:?}"),
            Self::DuplicateEvidenceId(id) => write!(f, "duplicate Solaris evidence id {id}"),
            Self::DuplicateFailureMode(id) => write!(f, "duplicate Solaris failure mode {id}"),
            Self::UnknownClaim(id) => write!(f, "unknown Solaris claim {}", id.0),
            Self::UnknownProcedure(id) => write!(f, "unknown Solaris procedure {id}"),
            Self::ProcedureConflict(id) => write!(f, "conflicting Solaris procedure {id}"),
            Self::MissingSolarisProfile => write!(f, "legacy pack is missing Solaris profile"),
        }
    }
}

impl Error for LegacySolarisErrorV1 {}

impl From<StandardsRegistryErrorV1> for LegacySolarisErrorV1 {
    fn from(value: StandardsRegistryErrorV1) -> Self {
        Self::Standards(value)
    }
}

impl From<LegacyComputingErrorV1> for LegacySolarisErrorV1 {
    fn from(value: LegacyComputingErrorV1) -> Self {
        Self::Computing(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seed_legacy_computing_pack_v1;

    fn pack() -> LegacyComputingPackV1 {
        seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap()
    }

    #[test]
    fn enrichment_is_idempotent_and_preserves_stronger_coverage() {
        let mut pack = pack();
        let before_observability = pack
            .profile(LegacyPlatformV1::Solaris)
            .unwrap()
            .state(LegacyKnowledgeAreaV1::ObservabilityAndProblemManagement);
        let first = enrich_legacy_solaris_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        let second = enrich_legacy_solaris_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.mechanisms.len(), 7);
        let after = pack.profile(LegacyPlatformV1::Solaris).unwrap();
        assert_eq!(
            after.state(LegacyKnowledgeAreaV1::ObservabilityAndProblemManagement),
            before_observability
        );
        assert!(!after.gap_tags.contains("zones-failure-scenarios"));
        assert!(!after.gap_tags.contains("dtrace-diagnostics"));
        assert!(after.gap_tags.contains("solaris-10-legacy"));
        assert!(after.gap_tags.contains("sparc-platform-depth"));
    }

    #[test]
    fn every_mechanism_is_source_bound_and_discriminative() {
        let mut pack = pack();
        let foundation = enrich_legacy_solaris_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
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
    fn zfs_and_boot_environment_are_distinct_mechanisms() {
        let mut pack = pack();
        let foundation = enrich_legacy_solaris_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        let zfs = foundation
            .mechanisms
            .iter()
            .find(|m| m.kind == SolarisMechanismKindV1::Zfs)
            .unwrap();
        let be = foundation
            .mechanisms
            .iter()
            .find(|m| m.kind == SolarisMechanismKindV1::IpsBootEnvironments)
            .unwrap();
        assert_ne!(zfs.kind, be.kind);
        assert!(be.dependencies.contains(&SolarisMechanismKindV1::Zfs));
    }

    #[test]
    fn advisory_procedures_never_mint_execution_authority() {
        let mut pack = pack();
        let foundation = enrich_legacy_solaris_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
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
    fn dtrace_is_not_silently_treated_as_passive_execution() {
        let mut pack = pack();
        enrich_legacy_solaris_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        let procedure = pack
            .procedures
            .iter()
            .find(|p| p.id == "legacy:solaris:dtrace-plan")
            .unwrap();
        assert!(procedure
            .steps
            .iter()
            .any(|step| step.authority == LegacyProcedureAuthorityV1::ChangeProposalOnly));
    }
}
