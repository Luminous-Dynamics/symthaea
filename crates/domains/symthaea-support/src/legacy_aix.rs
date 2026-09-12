// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Source-bound AIX foundation for legacy enterprise IT reasoning.
//!
//! Deepens AIX 7.3 with explicit ODM/device, LVM, SRC, error-log, NIM,
//! LPAR, and PowerHA mechanism models. The layer is advisory only; source
//! snapshots remain metadata-only until separately retained and digest-bound.

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
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

pub const LEGACY_AIX_FOUNDATION_SCHEMA_V1: &str = "symthaea-it-legacy-aix-foundation-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AixMechanismKindV1 {
    DeviceConfigurationOdm,
    LvmStorage,
    SrcSubsystems,
    ErrorLogging,
    Nim,
    LogicalPartitions,
    PowerHa,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AixEvidenceSignalV1 {
    pub id: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AixFailureModeV1 {
    pub id: String,
    pub symptom: String,
    pub discriminators: Vec<String>,
    pub unsafe_shortcuts: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AixMechanismModelV1 {
    pub kind: AixMechanismKindV1,
    pub area: LegacyKnowledgeAreaV1,
    pub title: String,
    pub summary: String,
    pub dependencies: BTreeSet<AixMechanismKindV1>,
    pub source_claims: BTreeSet<TechnicalClaimIdV1>,
    pub evidence_signals: Vec<AixEvidenceSignalV1>,
    pub failure_modes: Vec<AixFailureModeV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AixFoundationV1 {
    pub schema_version: String,
    pub product_version: String,
    pub mechanisms: Vec<AixMechanismModelV1>,
    pub procedure_ids: BTreeSet<String>,
}

impl AixFoundationV1 {
    pub fn validate(&self, pack: &LegacyComputingPackV1) -> Result<(), LegacyAixErrorV1> {
        if self.schema_version != LEGACY_AIX_FOUNDATION_SCHEMA_V1 {
            return Err(LegacyAixErrorV1::UnsupportedSchema(self.schema_version.clone()));
        }
        nonempty(&self.product_version, "AIX product version")?;
        if self.mechanisms.is_empty() {
            return Err(LegacyAixErrorV1::InvalidField("missing mechanisms".into()));
        }
        let mut kinds = BTreeSet::new();
        for model in &self.mechanisms {
            if !kinds.insert(model.kind) {
                return Err(LegacyAixErrorV1::DuplicateMechanism(model.kind));
            }
            if model.source_claims.is_empty()
                || model.evidence_signals.is_empty()
                || model.failure_modes.is_empty()
            {
                return Err(LegacyAixErrorV1::InvalidField(format!(
                    "mechanism {:?} lacks claims/evidence/failure modes",
                    model.kind
                )));
            }
            for claim in &model.source_claims {
                if pack.sources.claim(claim).is_none() {
                    return Err(LegacyAixErrorV1::UnknownClaim(claim.clone()));
                }
            }
            if model
                .failure_modes
                .iter()
                .any(|failure| failure.discriminators.is_empty())
            {
                return Err(LegacyAixErrorV1::InvalidField(format!(
                    "mechanism {:?} has non-discriminative failure mode",
                    model.kind
                )));
            }
        }
        for id in &self.procedure_ids {
            if !pack.procedures.iter().any(|procedure| procedure.id == *id) {
                return Err(LegacyAixErrorV1::UnknownProcedure(id.clone()));
            }
        }
        Ok(())
    }
}

pub fn enrich_legacy_aix_foundation_v1(
    pack: &mut LegacyComputingPackV1,
    fetched_at_unix_ms: u64,
) -> Result<AixFoundationV1, LegacyAixErrorV1> {
    if fetched_at_unix_ms == 0 {
        return Err(LegacyAixErrorV1::InvalidField(
            "source fetch timestamp must be non-zero".into(),
        ));
    }
    pack.validate()?;
    let snapshots = register_sources(pack, fetched_at_unix_ms)?;
    register_claims(pack)?;
    let procedure_ids = register_procedures(pack, &snapshots)?;
    promote_profile(pack, &snapshots)?;
    let foundation = AixFoundationV1 {
        schema_version: LEGACY_AIX_FOUNDATION_SCHEMA_V1.into(),
        product_version: "7.3".into(),
        mechanisms: mechanisms(),
        procedure_ids,
    };
    pack.validate()?;
    foundation.validate(pack)?;
    Ok(foundation)
}

fn register_sources(
    pack: &mut LegacyComputingPackV1,
    fetched_at_unix_ms: u64,
) -> Result<BTreeSet<SourceSnapshotIdV1>, LegacyAixErrorV1> {
    let definitions = [
        ("ibm:aix-odm-device-config", "ibm:aix-odm-device-config@7.3", "AIX device configuration database", "AIX 7.3 ODM device configuration database", "https://www.ibm.com/docs/en/aix/7.3.0?topic=subsystem-device-configuration-database-overview", "7.3"),
        ("ibm:aix-lvm-storage", "ibm:aix-lvm-storage@7.3", "AIX logical volume storage concepts", "AIX 7.3 LVM storage concepts", "https://www.ibm.com/docs/en/aix/7.3.0?topic=storage-logical-volume-concepts", "7.3"),
        ("ibm:aix-src", "ibm:aix-src@7.3", "AIX System Resource Controller", "AIX 7.3 System Resource Controller", "https://www.ibm.com/docs/en/aix/7.3.0?topic=concepts-system-resource-controller", "7.3"),
        ("ibm:aix-error-log", "ibm:aix-error-log@7.3", "AIX error-logging facility", "AIX 7.3 error-logging facility", "https://www.ibm.com/docs/en/aix/7.3.0?topic=overview-error-logging-facility", "7.3"),
        ("ibm:aix-nim", "ibm:aix-nim@7.3", "AIX Network Installation Management", "AIX 7.3 Network Installation Management", "https://www.ibm.com/docs/en/aix/7.3.0?topic=installing-network-installation-management", "7.3"),
        ("ibm:aix-lpar", "ibm:aix-lpar@7.3", "AIX logical partitions", "AIX 7.3 logical partitions", "https://www.ibm.com/docs/en/aix/7.3.0?topic=concepts-logical-partitions", "7.3"),
        ("ibm:powerha-aix-7210sp1", "ibm:powerha-aix-7210sp1@2026-04", "PowerHA SystemMirror 7.2.10 SP1 fix information", "PowerHA 7.2.10 SP1 / AIX 7.3 recommended levels", "https://www.ibm.com/support/pages/powerha-fix-information-powerha-7210-service-pack-1", "7.2.10 SP1 / 2026-04"),
    ];
    let mut snapshots = BTreeSet::new();
    for (doc_id, snap_id, title, canonical_ref, locator, version) in definitions {
        pack.sources.register_document(TechnicalSourceDocumentV1 {
            id: SourceDocumentIdV1(doc_id.into()),
            publisher: TechnicalPublisherV1::Vendor("IBM".into()),
            kind: SourceDocumentKindV1::VendorDocumentation,
            title: title.into(),
            canonical_ref: canonical_ref.into(),
            canonical_locator: Some(locator.into()),
        })?;
        pack.sources.register_snapshot(TechnicalSourceSnapshotV1 {
            id: SourceSnapshotIdV1(snap_id.into()),
            document_id: SourceDocumentIdV1(doc_id.into()),
            version: Some(version.into()),
            lifecycle: KnowledgeLifecycleV1::Active,
            authority: KnowledgeAuthorityClassV1::VendorDocumentation,
            stability: KnowledgeStabilityV1::Stable,
            published_at_unix_ms: None,
            source_updated_at_unix_ms: None,
            fetched_at_unix_ms,
            capture: SourceCaptureV1::MetadataOnly,
            relations: BTreeSet::new(),
        })?;
        snapshots.insert(SourceSnapshotIdV1(snap_id.into()));
    }
    Ok(snapshots)
}

fn register_claims(pack: &mut LegacyComputingPackV1) -> Result<(), LegacyAixErrorV1> {
    let claims = [
        claim("legacy:aix:odm-device-state", "ibm:aix-odm-device-config@7.3", "AIX device configuration uses ODM-backed predefined and customized object classes; configured device identity is distinct from current runtime path availability.", "Device Configuration Database Overview", SupportCategory::Hardware, &[]),
        claim("legacy:aix:lvm-layering", "ibm:aix-lvm-storage@7.3", "AIX LVM separates physical volumes, volume groups, physical/logical partitions and logical volumes, so filesystem symptoms must be localized within that hierarchy.", "Logical volume storage concepts", SupportCategory::Software, &[]),
        claim("legacy:aix:src-subsystem-control", "ibm:aix-src@7.3", "AIX SRC supplies subsystem/subserver lifecycle and status control; SRC state is distinct from raw process existence and end-to-end application health.", "System resource controller", SupportCategory::Software, &[]),
        claim("legacy:aix:error-log", "ibm:aix-error-log@7.3", "The AIX error-log facility records hardware and software failures as fault evidence that must be correlated with current system state and failure timing.", "Error-logging facility", SupportCategory::Hardware, &[]),
        claim("legacy:aix:nim-model", "ibm:aix-nim@7.3", "AIX NIM models masters, clients, networks and resources for BOS/software operations; NIM management state is separate from firmware/SMS boot and running-client state.", "Network Installation Management", SupportCategory::Software, &["nim"]),
        claim("legacy:aix:lpar-resource-boundary", "ibm:aix-lpar@7.3", "AIX logical partitions isolate OS environments while assigning processor, memory and I/O resources; a guest symptom can originate in partition or virtual-I/O state as well as inside AIX.", "Logical partitions", SupportCategory::Hardware, &["lpar"]),
        claim("legacy:aix:powerha-version-context", "ibm:powerha-aix-7210sp1@2026-04", "IBM PowerHA 7.2.10 SP1 guidance identifies tested/recommended AIX 7.3 maintenance levels, so cluster reasoning must bind both AIX and PowerHA maintenance context.", "Recommended levels", SupportCategory::Software, &["powerha-systemmirror", "powerha:7.2.10-sp1"]),
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
    required_features: &[&str],
) -> TechnicalKnowledgeClaimV1 {
    let mut applicability = aix_scope();
    applicability
        .required_features
        .extend(required_features.iter().map(|value| (*value).into()));
    TechnicalKnowledgeClaimV1 {
        id: TechnicalClaimIdV1(id.into()),
        statement: statement.into(),
        source_snapshot: SourceSnapshotIdV1(snapshot.into()),
        locator: Some(TechnicalSourceLocatorV1 {
            section: Some(section.into()),
            fragment: None,
        }),
        modality: ClaimModalityV1::Descriptive,
        applicability: Some(applicability),
        extraction_quality: Some(0.95),
        category: Some(category),
    }
}

/// Canonical V1 legacy ecosystem is shared with the base pack. Product/vendor
/// selectors provide isolation between AIX, IBM i, z/OS, Solaris, and HP-UX.
fn aix_scope() -> ApplicabilityScopeV1 {
    ApplicabilityScopeV1 {
        ecosystem: StringSelectorV1::Exact("legacy-enterprise-os".into()),
        vendor: StringSelectorV1::Exact("IBM".into()),
        product: "AIX".into(),
        version: StringSelectorV1::Prefix("7.3".into()),
        ..ApplicabilityScopeV1::default()
    }
}

fn promote_profile(
    pack: &mut LegacyComputingPackV1,
    snapshots: &BTreeSet<SourceSnapshotIdV1>,
) -> Result<(), LegacyAixErrorV1> {
    let profile = pack
        .profiles
        .iter_mut()
        .find(|profile| profile.platform == LegacyPlatformV1::Aix)
        .ok_or(LegacyAixErrorV1::MissingAixProfile)?;
    profile.source_snapshots.extend(snapshots.iter().cloned());
    for area in [
        LegacyKnowledgeAreaV1::SystemLifecycle,
        LegacyKnowledgeAreaV1::WorkloadAndJobs,
        LegacyKnowledgeAreaV1::Storage,
        LegacyKnowledgeAreaV1::ObservabilityAndProblemManagement,
        LegacyKnowledgeAreaV1::AvailabilityAndRecovery,
        LegacyKnowledgeAreaV1::VirtualizationAndPartitioning,
        LegacyKnowledgeAreaV1::SoftwareLifecycle,
    ] {
        if profile.state(area) < LegacyCoverageStateV1::ClaimSeeded {
            profile.coverage.insert(area, LegacyCoverageStateV1::ClaimSeeded);
        }
    }
    for old in ["lpar-hmc", "nim", "powerha"] {
        profile.gap_tags.remove(old);
    }
    profile.gap_tags.extend([
        "aix-odm-cfgmgr-device-diagnostics".into(),
        "aix-lvm-jfs2-multipath-recovery".into(),
        "aix-src-subsystem-diagnostics".into(),
        "aix-errpt-hardware-correlation".into(),
        "aix-nim-mksysb-alt-disk-depth".into(),
        "aix-lpar-hmc-powervm-runtime-depth".into(),
        "aix-powerha-cluster-failure-labs".into(),
    ]);
    Ok(())
}

fn register_procedures(
    pack: &mut LegacyComputingPackV1,
    snapshots: &BTreeSet<SourceSnapshotIdV1>,
) -> Result<BTreeSet<String>, LegacyAixErrorV1> {
    let definitions = [
        ("legacy:aix:triage-device-odm", LegacyKnowledgeAreaV1::SystemLifecycle, LegacyProcedureKindV1::Diagnose, "Triage AIX device/ODM state", "Compare ODM identity/attributes/parentage with current device/path and partition evidence before proposing reconfiguration.", "ibm:aix-odm-device-config@7.3"),
        ("legacy:aix:triage-lvm-storage", LegacyKnowledgeAreaV1::Storage, LegacyProcedureKindV1::Diagnose, "Triage AIX LVM storage", "Map filesystem through LV/VG/PV/path state and distinguish logical allocation from device/path failure before proposing storage changes.", "ibm:aix-lvm-storage@7.3"),
        ("legacy:aix:triage-src-subsystem", LegacyKnowledgeAreaV1::WorkloadAndJobs, LegacyProcedureKindV1::Diagnose, "Triage AIX SRC subsystem", "Compare SRC registration/status, process state, dependencies and functional service health before proposing start/stop/refresh actions.", "ibm:aix-src@7.3"),
        ("legacy:aix:triage-error-log", LegacyKnowledgeAreaV1::ObservabilityAndProblemManagement, LegacyProcedureKindV1::Diagnose, "Correlate AIX error-log evidence", "Correlate current error-log identifiers/resources/timestamps with the failure window and independent subsystem evidence before proposing repair.", "ibm:aix-error-log@7.3"),
        ("legacy:aix:triage-nim", LegacyKnowledgeAreaV1::SoftwareLifecycle, LegacyProcedureKindV1::Diagnose, "Triage AIX NIM operation", "Separate NIM object/resource/control state from firmware/SMS boot, network transport, installation, and running-client state before proposing NIM changes.", "ibm:aix-nim@7.3"),
        ("legacy:aix:triage-lpar", LegacyKnowledgeAreaV1::VirtualizationAndPartitioning, LegacyProcedureKindV1::Diagnose, "Triage AIX LPAR resource boundary", "Compare saved/current partition allocation, VIOS-backed resources, and AIX guest observations before proposing DLPAR/profile changes.", "ibm:aix-lpar@7.3"),
        ("legacy:aix:triage-powerha", LegacyKnowledgeAreaV1::AvailabilityAndRecovery, LegacyProcedureKindV1::Recover, "Triage AIX PowerHA partial failure", "Bind AIX/PowerHA levels and inspect cluster/node/resource-group/network/storage state across members before proposing movement or restart.", "ibm:powerha-aix-7210sp1@2026-04"),
    ];
    let mut ids = BTreeSet::new();
    for (id, area, kind, title, diagnostic, source_snapshot) in definitions {
        let source_snapshot = SourceSnapshotIdV1(source_snapshot.into());
        if !snapshots.contains(&source_snapshot) {
            return Err(LegacyAixErrorV1::InvalidField(format!(
                "AIX procedure {id} references unregistered source snapshot {}",
                source_snapshot.0
            )));
        }
        let mut procedure = procedure(id, area, kind, title, diagnostic);
        procedure.source_snapshots = [source_snapshot].into_iter().collect();
        if let Some(existing) = pack.procedures.iter().find(|p| p.id == procedure.id) {
            if existing != &procedure {
                return Err(LegacyAixErrorV1::ProcedureIdentityConflict(procedure.id));
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
    diagnostic: &str,
) -> LegacyProcedureV1 {
    LegacyProcedureV1 {
        id: id.into(),
        platform: LegacyPlatformV1::Aix,
        area,
        kind,
        title: title.into(),
        applicability: aix_scope(),
        preconditions: vec![
            "Establish exact AIX release/TL/SP, target LPAR, and relevant optional subsystem versions/features.".into(),
            "Preserve operator authority boundaries; diagnosis does not authorize mutation.".into(),
        ],
        steps: vec![
            LegacyProcedureStepV1 {
                authority: LegacyProcedureAuthorityV1::ReadOnlyObservation,
                description: "Establish failure window, exact affected resource, current topology, and recent change context.".into(),
            },
            LegacyProcedureStepV1 {
                authority: LegacyProcedureAuthorityV1::ReadOnlyObservation,
                description: diagnostic.into(),
            },
            LegacyProcedureStepV1 {
                authority: LegacyProcedureAuthorityV1::ReadOnlyObservation,
                description: "Check source/version/feature applicability and separate stale or indirect evidence from current direct evidence.".into(),
            },
            LegacyProcedureStepV1 {
                authority: LegacyProcedureAuthorityV1::ChangeProposalOnly,
                description: "If evidence supports repair, propose the smallest reversible operator-reviewed change with rollback, blast-radius, and verification criteria.".into(),
            },
        ],
        verification: vec![
            "Re-observe the original symptom against the same system/LPAR/resource identity.".into(),
            "Confirm unrelated workloads, paths, cluster members, and dependencies did not regress.".into(),
        ],
        source_snapshots: BTreeSet::new(),
    }
}

fn mechanisms() -> Vec<AixMechanismModelV1> {
    use AixMechanismKindV1 as M;
    vec![
        model(M::DeviceConfigurationOdm, LegacyKnowledgeAreaV1::SystemLifecycle, "ODM-backed device configuration", "Configured ODM state, attributes, parentage, device methods, and runtime path availability are distinct layers.", &[], &["legacy:aix:odm-device-state"], &["odm-object", "runtime-path", "driver-method"], &[
            ("defined-not-available", "Device is defined but current runtime path is unavailable.", &["ODM object exists", "runtime path fails"], &["delete ODM object before path diagnosis"]),
            ("stale-device-attributes", "Customized configuration differs from current topology.", &["parent/location/attribute mismatch"], &["broad reconfiguration before preserving state"]),
        ]),
        model(M::LvmStorage, LegacyKnowledgeAreaV1::Storage, "LVM storage hierarchy", "PV/VG/partition/LV/filesystem state and underlying device paths must remain separate diagnostic layers.", &[M::DeviceConfigurationOdm], &["legacy:aix:lvm-layering"], &["pv-vg-lv", "mirror-state", "filesystem-state"], &[
            ("logical-not-physical", "Logical/filesystem state is mistaken for physical media failure.", &["PV/path healthy", "LV/filesystem evidence explains symptom"], &["replace disk before isolating LVM layer"]),
            ("path-loss-versus-vg", "Path/device loss produces downstream VG/PV symptoms.", &["path changed before LVM symptom"], &["force VG changes before path diagnosis"]),
        ]),
        model(M::SrcSubsystems, LegacyKnowledgeAreaV1::WorkloadAndJobs, "System Resource Controller", "SRC managed state, process existence, and application function are independent observations.", &[], &["legacy:aix:src-subsystem-control"], &["src-status", "process-state", "functional-health"], &[
            ("process-up-src-down", "Daemon exists while SRC registration/state is wrong.", &["process present", "SRC state differs"], &["kill process before SRC/dependency diagnosis"]),
            ("src-up-service-unhealthy", "SRC active while end-to-end service is unhealthy.", &["SRC active", "functional check fails"], &["treat SRC active as service proof"]),
        ]),
        model(M::ErrorLogging, LegacyKnowledgeAreaV1::ObservabilityAndProblemManagement, "AIX error logging", "Error-log records are time-scoped evidence, not automatic unique root causes.", &[M::DeviceConfigurationOdm], &["legacy:aix:error-log"], &["errlog-record", "failure-window", "corroboration"], &[
            ("stale-error-anchor", "Old error record is incorrectly used as current root cause.", &["record predates failure window", "fresh evidence differs"], &["repair named resource from stale log alone"]),
            ("symptom-not-root", "Logged resource is downstream of an upstream failure.", &["upstream evidence precedes logged error"], &["first error wins reasoning"]),
        ]),
        model(M::Nim, LegacyKnowledgeAreaV1::SoftwareLifecycle, "Network Installation Management", "NIM object/resource/control state, firmware/SMS boot, transfer/install, and running AIX are distinct stages.", &[M::LvmStorage], &["legacy:aix:nim-model"], &["nim-objects", "resource-state", "boot-stage"], &[
            ("nim-object-versus-network", "NIM objects are valid while firmware/SMS networking fails.", &["NIM resources valid", "failure before AIX runtime"], &["change running AIX network for SMS failure"]),
            ("resource-level-mismatch", "NIM resource level/composition mismatches target operation.", &["resource metadata differs"], &["rebuild all resources before isolating mismatch"]),
        ]),
        model(M::LogicalPartitions, LegacyKnowledgeAreaV1::VirtualizationAndPartitioning, "LPAR resource boundary", "Saved profile, current dynamic allocation, VIOS backing, and AIX guest view are separate topology facts.", &[M::DeviceConfigurationOdm], &["legacy:aix:lpar-resource-boundary"], &["lpar-profile", "runtime-allocation", "guest-view"], &[
            ("guest-healthy-resource-missing", "Guest OS healthy while assigned/backing resource is absent.", &["AIX otherwise healthy", "partition resource differs"], &["reconfigure guest before checking LPAR/VIOS"]),
            ("profile-runtime-drift", "Saved profile differs from current DLPAR allocation.", &["profile/runtime mismatch"], &["reapply profile without workload impact review"]),
        ]),
        model(M::PowerHa, LegacyKnowledgeAreaV1::AvailabilityAndRecovery, "PowerHA SystemMirror", "Cluster/resource-group health spans nodes, dependencies, topology, and exact AIX/PowerHA maintenance context.", &[M::LogicalPartitions, M::LvmStorage, M::SrcSubsystems], &["legacy:aix:powerha-version-context"], &["cluster-state", "dependency-state", "version-level"], &[
            ("local-up-cluster-degraded", "Local node healthy while cluster/resource group degraded.", &["local checks pass", "cross-node state degraded"], &["restart local service before cluster diagnosis"]),
            ("version-context-drift", "Cluster behavior changes across incompatible or changed maintenance context.", &["level changed near failure", "behavior differs across members"], &["ad hoc cluster upgrades/downgrades"]),
        ]),
    ]
}

fn model(
    kind: AixMechanismKindV1,
    area: LegacyKnowledgeAreaV1,
    title: &str,
    summary: &str,
    dependencies: &[AixMechanismKindV1],
    source_claims: &[&str],
    evidence_ids: &[&str],
    failures: &[(&str, &str, &[&str], &[&str])],
) -> AixMechanismModelV1 {
    AixMechanismModelV1 {
        kind,
        area,
        title: title.into(),
        summary: summary.into(),
        dependencies: dependencies.iter().copied().collect(),
        source_claims: source_claims
            .iter()
            .map(|value| TechnicalClaimIdV1((*value).into()))
            .collect(),
        evidence_signals: evidence_ids
            .iter()
            .map(|value| AixEvidenceSignalV1 {
                id: (*value).into(),
                description: format!("AIX {} evidence", value.replace('-', " ")),
            })
            .collect(),
        failure_modes: failures
            .iter()
            .map(|(id, symptom, discriminators, unsafe_shortcuts)| AixFailureModeV1 {
                id: (*id).into(),
                symptom: (*symptom).into(),
                discriminators: discriminators.iter().map(|value| (*value).into()).collect(),
                unsafe_shortcuts: unsafe_shortcuts.iter().map(|value| (*value).into()).collect(),
            })
            .collect(),
    }
}

fn nonempty(value: &str, field: &'static str) -> Result<(), LegacyAixErrorV1> {
    if value.trim().is_empty() {
        Err(LegacyAixErrorV1::InvalidField(field.into()))
    } else {
        Ok(())
    }
}

#[derive(Debug)]
pub enum LegacyAixErrorV1 {
    LegacyPack(LegacyComputingErrorV1),
    Standards(StandardsRegistryErrorV1),
    UnsupportedSchema(String),
    InvalidField(String),
    MissingAixProfile,
    DuplicateMechanism(AixMechanismKindV1),
    UnknownClaim(TechnicalClaimIdV1),
    UnknownProcedure(String),
    ProcedureIdentityConflict(String),
}

impl fmt::Display for LegacyAixErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LegacyPack(err) => write!(f, "invalid legacy pack: {err}"),
            Self::Standards(err) => write!(f, "invalid AIX source data: {err}"),
            Self::UnsupportedSchema(value) => write!(f, "unsupported AIX schema {value}"),
            Self::InvalidField(value) => write!(f, "invalid AIX field: {value}"),
            Self::MissingAixProfile => write!(f, "legacy pack is missing AIX profile"),
            Self::DuplicateMechanism(kind) => write!(f, "duplicate AIX mechanism {kind:?}"),
            Self::UnknownClaim(id) => write!(f, "unknown AIX claim {}", id.0),
            Self::UnknownProcedure(id) => write!(f, "unknown AIX procedure {id}"),
            Self::ProcedureIdentityConflict(id) => write!(f, "AIX procedure identity conflict {id}"),
        }
    }
}

impl Error for LegacyAixErrorV1 {}
impl From<LegacyComputingErrorV1> for LegacyAixErrorV1 {
    fn from(value: LegacyComputingErrorV1) -> Self { Self::LegacyPack(value) }
}
impl From<StandardsRegistryErrorV1> for LegacyAixErrorV1 {
    fn from(value: StandardsRegistryErrorV1) -> Self { Self::Standards(value) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        KnowledgeQueryPurposeV1, SupportKnowledgeQueryV1, SupportKnowledgeSourceV1,
        TechnologyIdentityV1, seed_legacy_computing_pack_v1,
    };

    #[test]
    fn enrichment_is_idempotent_and_coverage_is_monotonic() {
        let mut pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        let before = pack
            .profile(LegacyPlatformV1::Aix)
            .unwrap()
            .state(LegacyKnowledgeAreaV1::ObservabilityAndProblemManagement);
        let first = enrich_legacy_aix_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        let claims = pack.sources.claims().count();
        let second = enrich_legacy_aix_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        assert_eq!(first, second);
        assert_eq!(pack.sources.claims().count(), claims);
        assert_eq!(
            pack.profile(LegacyPlatformV1::Aix)
                .unwrap()
                .state(LegacyKnowledgeAreaV1::ObservabilityAndProblemManagement),
            before
        );
    }

    #[test]
    fn optional_subsystem_claims_do_not_auto_apply() {
        let mut pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        enrich_legacy_aix_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        let technology = TechnologyIdentityV1 {
            ecosystem: Some("legacy-enterprise-os".into()),
            vendor: Some("IBM".into()),
            product: "AIX".into(),
            edition: None,
            version: Some("7.3".into()),
            build: None,
            architecture: None,
            platform: None,
            profile: None,
            observed_features: BTreeSet::new(),
        };
        let query = SupportKnowledgeQueryV1 {
            text: "PowerHA recommended levels".into(),
            limit: 10,
            category: Some(SupportCategory::Software),
            technology: Some(technology),
            purpose: KnowledgeQueryPurposeV1::LocalReasoning,
        };
        let hits = pack.sources.search_support_knowledge(&query).unwrap();
        let hit = hits
            .iter()
            .find(|hit| hit.source_id.contains("powerha-version-context"))
            .unwrap();
        assert!(hit.technology.is_none());
    }

    #[test]
    fn mechanisms_are_source_bound_and_procedures_non_authoritative() {
        let mut pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        let foundation = enrich_legacy_aix_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        assert_eq!(foundation.mechanisms.len(), 7);
        assert!(foundation.mechanisms.iter().all(|model| !model.source_claims.is_empty()));
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
    fn procedures_use_mechanism_scoped_source_snapshots() {
        let mut pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        enrich_legacy_aix_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        let expected = [
            ("legacy:aix:triage-device-odm", "ibm:aix-odm-device-config@7.3"),
            ("legacy:aix:triage-lvm-storage", "ibm:aix-lvm-storage@7.3"),
            ("legacy:aix:triage-src-subsystem", "ibm:aix-src@7.3"),
            ("legacy:aix:triage-error-log", "ibm:aix-error-log@7.3"),
            ("legacy:aix:triage-nim", "ibm:aix-nim@7.3"),
            ("legacy:aix:triage-lpar", "ibm:aix-lpar@7.3"),
            ("legacy:aix:triage-powerha", "ibm:powerha-aix-7210sp1@2026-04"),
        ];
        for (procedure_id, snapshot_id) in expected {
            let procedure = pack.procedures.iter().find(|p| p.id == procedure_id).unwrap();
            assert_eq!(procedure.source_snapshots.len(), 1);
            assert!(procedure
                .source_snapshots
                .contains(&SourceSnapshotIdV1(snapshot_id.into())));
        }
    }

    #[test]
    fn base_and_enriched_aix_share_canonical_ecosystem() {
        let scope = aix_scope();
        let identity = TechnologyIdentityV1 {
            ecosystem: Some("legacy-enterprise-os".into()),
            vendor: Some("IBM".into()),
            product: "AIX".into(),
            edition: None,
            version: Some("7.3".into()),
            build: None,
            architecture: None,
            platform: None,
            profile: None,
            observed_features: BTreeSet::new(),
        };
        assert_eq!(
            scope.assess(&identity).unwrap().status,
            crate::ApplicabilityStatusV1::Applicable
        );
    }
}
