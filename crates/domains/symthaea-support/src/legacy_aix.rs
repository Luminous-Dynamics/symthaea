// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Source-bound AIX foundation for legacy enterprise IT reasoning.
//!
//! This module deepens the AIX 7.3 slice of `LegacyComputingPackV1` with
//! explicit device/ODM, storage, subsystem, error-log, NIM, LPAR, and PowerHA
//! mechanisms. It remains advisory and non-executable. Source snapshots are
//! metadata-only until the separate retained-artifact path freezes exact bytes.

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
        require_nonempty(&self.product_version, "AIX product version")?;
        if self.mechanisms.is_empty() {
            return Err(LegacyAixErrorV1::InvalidField(
                "AIX foundation requires mechanisms".into(),
            ));
        }

        let mut kinds = BTreeSet::new();
        for mechanism in &self.mechanisms {
            if !kinds.insert(mechanism.kind) {
                return Err(LegacyAixErrorV1::DuplicateMechanism(mechanism.kind));
            }
            require_nonempty(&mechanism.title, "AIX mechanism title")?;
            require_nonempty(&mechanism.summary, "AIX mechanism summary")?;
            if mechanism.source_claims.is_empty()
                || mechanism.evidence_signals.is_empty()
                || mechanism.failure_modes.is_empty()
            {
                return Err(LegacyAixErrorV1::InvalidField(format!(
                    "AIX mechanism {:?} requires source claims, evidence, and failure modes",
                    mechanism.kind
                )));
            }
            for claim in &mechanism.source_claims {
                if pack.sources.claim(claim).is_none() {
                    return Err(LegacyAixErrorV1::UnknownClaim(claim.clone()));
                }
            }
            let mut evidence_ids = BTreeSet::new();
            for evidence in &mechanism.evidence_signals {
                require_nonempty(&evidence.id, "AIX evidence id")?;
                require_nonempty(&evidence.description, "AIX evidence description")?;
                if !evidence_ids.insert(evidence.id.as_str()) {
                    return Err(LegacyAixErrorV1::DuplicateEvidenceId(evidence.id.clone()));
                }
            }
            let mut failure_ids = BTreeSet::new();
            for failure in &mechanism.failure_modes {
                require_nonempty(&failure.id, "AIX failure mode id")?;
                require_nonempty(&failure.symptom, "AIX failure mode symptom")?;
                if !failure_ids.insert(failure.id.as_str()) {
                    return Err(LegacyAixErrorV1::DuplicateFailureMode(failure.id.clone()));
                }
                if failure.discriminators.is_empty() {
                    return Err(LegacyAixErrorV1::InvalidField(format!(
                        "AIX failure mode {} requires discriminators",
                        failure.id
                    )));
                }
            }
        }

        let known_procedures: BTreeSet<_> = pack.procedures.iter().map(|p| p.id.as_str()).collect();
        for id in &self.procedure_ids {
            if !known_procedures.contains(id.as_str()) {
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
            "AIX source fetch timestamp must be non-zero".into(),
        ));
    }
    pack.validate()?;
    let snapshots = register_aix_sources(pack, fetched_at_unix_ms)?;
    register_aix_claims(pack)?;
    let procedure_ids = add_aix_procedures(pack, &snapshots)?;
    update_aix_profile(pack, &snapshots)?;

    let foundation = AixFoundationV1 {
        schema_version: LEGACY_AIX_FOUNDATION_SCHEMA_V1.into(),
        product_version: "7.3".into(),
        mechanisms: seed_mechanisms(),
        procedure_ids,
    };
    pack.validate()?;
    foundation.validate(pack)?;
    Ok(foundation)
}

fn register_aix_sources(
    pack: &mut LegacyComputingPackV1,
    fetched_at_unix_ms: u64,
) -> Result<BTreeSet<SourceSnapshotIdV1>, LegacyAixErrorV1> {
    let definitions = [
        (
            "ibm:aix-odm-device-config",
            "ibm:aix-odm-device-config@7.3",
            "AIX device configuration database",
            "AIX 7.3 ODM device configuration database",
            "https://www.ibm.com/docs/en/aix/7.3.0?topic=subsystem-device-configuration-database-overview",
            "7.3",
        ),
        (
            "ibm:aix-lvm-storage",
            "ibm:aix-lvm-storage@7.3",
            "AIX logical volume storage concepts",
            "AIX 7.3 LVM storage concepts",
            "https://www.ibm.com/docs/en/aix/7.3.0?topic=storage-logical-volume-concepts",
            "7.3",
        ),
        (
            "ibm:aix-src",
            "ibm:aix-src@7.3",
            "AIX System Resource Controller",
            "AIX 7.3 System Resource Controller",
            "https://www.ibm.com/docs/en/aix/7.3.0?topic=concepts-system-resource-controller",
            "7.3",
        ),
        (
            "ibm:aix-error-log",
            "ibm:aix-error-log@7.3",
            "AIX error-logging facility",
            "AIX 7.3 error-logging facility",
            "https://www.ibm.com/docs/en/aix/7.3.0?topic=overview-error-logging-facility",
            "7.3",
        ),
        (
            "ibm:aix-nim",
            "ibm:aix-nim@7.3",
            "AIX Network Installation Management",
            "AIX 7.3 Network Installation Management",
            "https://www.ibm.com/docs/en/aix/7.3.0?topic=installing-network-installation-management",
            "7.3",
        ),
        (
            "ibm:aix-lpar",
            "ibm:aix-lpar@7.3",
            "AIX logical partitions",
            "AIX 7.3 logical partitions",
            "https://www.ibm.com/docs/en/aix/7.3.0?topic=concepts-logical-partitions",
            "7.3",
        ),
        (
            "ibm:powerha-aix-7210sp1",
            "ibm:powerha-aix-7210sp1@2026-04",
            "PowerHA SystemMirror 7.2.10 SP1 fix information",
            "PowerHA SystemMirror 7.2.10 SP1 / AIX 7.3 recommended levels",
            "https://www.ibm.com/support/pages/powerha-fix-information-powerha-7210-service-pack-1",
            "7.2.10 SP1 / 2026-04",
        ),
    ];

    let mut snapshots = BTreeSet::new();
    for (document_id, snapshot_id, title, canonical_ref, locator, version) in definitions {
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
        snapshots.insert(SourceSnapshotIdV1(snapshot_id.into()));
    }
    Ok(snapshots)
}

fn register_aix_claims(pack: &mut LegacyComputingPackV1) -> Result<(), LegacyAixErrorV1> {
    let claims = [
        claim(
            "legacy:aix:odm-device-state",
            "ibm:aix-odm-device-config@7.3",
            "AIX device configuration uses ODM-backed predefined and customized object classes; a device can be defined in configuration data independently of whether the current runtime device path is healthy or available.",
            "Device Configuration Database Overview",
            SupportCategory::Hardware,
        ),
        claim(
            "legacy:aix:lvm-layering",
            "ibm:aix-lvm-storage@7.3",
            "AIX LVM separates physical volumes, volume groups, physical/logical partitions, and logical volumes; storage diagnosis must preserve these layers instead of treating every filesystem symptom as a physical-disk failure.",
            "Logical volume storage concepts",
            SupportCategory::Software,
        ),
        claim(
            "legacy:aix:src-subsystem-control",
            "ibm:aix-src@7.3",
            "AIX SRC provides a common control and status interface for subsystems and subservers; subsystem state is distinct from generic process existence and can include subsystem-specific refresh, trace, and notification behavior.",
            "System resource controller",
            SupportCategory::Software,
        ),
        claim(
            "legacy:aix:error-log",
            "ibm:aix-error-log@7.3",
            "The AIX error-logging facility records hardware and software failures for fault detection and corrective action, providing evidence that is distinct from ordinary application or syslog messages.",
            "Error-logging facility",
            SupportCategory::Hardware,
        ),
        claim(
            "legacy:aix:nim-model",
            "ibm:aix-nim@7.3",
            "AIX NIM models a master, clients, resources, and networks so BOS/software installation and maintenance can be managed centrally; NIM object/resource state is separate from the live operating state of a client.",
            "Network Installation Management",
            SupportCategory::Software,
        ),
        claim(
            "legacy:aix:lpar-resource-boundary",
            "ibm:aix-lpar@7.3",
            "AIX logical partitions isolate operating-system environments while assigning processor, memory, boot, network, and I/O resources; an AIX guest symptom can therefore originate in partition/resource assignment as well as inside the guest OS.",
            "Logical partitions",
            SupportCategory::Hardware,
        ),
        claim(
            "legacy:aix:powerha-version-context",
            "ibm:powerha-aix-7210sp1@2026-04",
            "Current IBM PowerHA 7.2.10 SP1 guidance lists multiple AIX 7.3 TL/SP levels as tested/recommended, so PowerHA diagnosis and remediation must bind both AIX and PowerHA maintenance levels rather than assuming any AIX 7.3 combination is equivalent.",
            "Recommended levels",
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
        applicability: Some(aix_scope()),
        extraction_quality: Some(0.95),
        category: Some(category),
    }
}

fn aix_scope() -> ApplicabilityScopeV1 {
    ApplicabilityScopeV1 {
        ecosystem: StringSelectorV1::Exact("legacy-enterprise-unix".into()),
        vendor: StringSelectorV1::Exact("IBM".into()),
        product: StringSelectorV1::Exact("AIX".into()),
        version: StringSelectorV1::Prefix("7.3".into()),
        ..ApplicabilityScopeV1::default()
    }
}

fn update_aix_profile(
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

fn add_aix_procedures(
    pack: &mut LegacyComputingPackV1,
    snapshots: &BTreeSet<SourceSnapshotIdV1>,
) -> Result<BTreeSet<String>, LegacyAixErrorV1> {
    let procedures = vec![
        procedure(
            "legacy:aix:triage-device-odm",
            LegacyKnowledgeAreaV1::SystemLifecycle,
            LegacyProcedureKindV1::Diagnose,
            "Triage AIX device/ODM configuration state",
            &[
                "Establish exact AIX TL/SP, LPAR identity, device logical name, parent path and expected hardware relationship.",
                "Compare predefined/customized ODM device state, attributes, parent/location data and current device availability read-only.",
                "Distinguish a configuration-database mismatch from an absent path, failed hardware, or driver/runtime problem.",
                "Any cfgmgr/device-definition/attribute mutation remains an operator-reviewed proposal with rollback and post-change verification.",
            ],
        ),
        procedure(
            "legacy:aix:triage-lvm-storage",
            LegacyKnowledgeAreaV1::Storage,
            LegacyProcedureKindV1::Diagnose,
            "Triage AIX LVM storage failure",
            &[
                "Establish the exact filesystem/logical-volume/volume-group/physical-volume chain and current mount/use context.",
                "Inspect PV/VG/LV state, partition mappings, mirrors and relevant filesystem evidence without changing allocation.",
                "Separate logical-volume/filesystem problems from missing paths, physical-volume failures, or stale device configuration.",
                "Any varyon/import/mirror/LV/filesystem/storage change remains proposal-only with data-protection and rollback review.",
            ],
        ),
        procedure(
            "legacy:aix:triage-src-subsystem",
            LegacyKnowledgeAreaV1::WorkloadAndJobs,
            LegacyProcedureKindV1::Diagnose,
            "Triage AIX SRC-managed subsystem",
            &[
                "Establish exact subsystem/subserver identity, SRC group, expected service state and dependent resources.",
                "Inspect SRC status, process state, subsystem configuration and recent termination/notification evidence read-only.",
                "Distinguish an SRC registration/state problem from a daemon process crash or dependency failure.",
                "Any start/stop/refresh/configuration operation remains an operator-reviewed proposal.",
            ],
        ),
        procedure(
            "legacy:aix:triage-error-log",
            LegacyKnowledgeAreaV1::ObservabilityAndProblemManagement,
            LegacyProcedureKindV1::Diagnose,
            "Correlate AIX error-log evidence",
            &[
                "Establish the failure window, affected resource and exact system/LPAR identity.",
                "Inspect current AIX error-log entries and correlate identifiers/resources/timestamps with application, storage, and platform evidence.",
                "Treat an error-log record as evidence to interpret, not automatic proof that its named component is the unique root cause.",
                "Any repair or component reset derived from the evidence remains proposal-only until the causal chain is established.",
            ],
        ),
        procedure(
            "legacy:aix:triage-nim",
            LegacyKnowledgeAreaV1::SoftwareLifecycle,
            LegacyProcedureKindV1::Diagnose,
            "Triage AIX NIM operation",
            &[
                "Establish exact NIM master/client identity, operation, object/resource assignments, AIX levels, and control ownership.",
                "Inspect NIM object/resource/network state and the client boot/runtime stage without changing allocations.",
                "Distinguish NIM database/resource problems from firmware/SMS network boot, client OS, or general network failures.",
                "Any resource allocation, install, migration, mksysb, SPOT or lpp_source mutation remains proposal-only.",
            ],
        ),
        procedure(
            "legacy:aix:triage-lpar",
            LegacyKnowledgeAreaV1::VirtualizationAndPartitioning,
            LegacyProcedureKindV1::Diagnose,
            "Triage AIX LPAR resource boundary",
            &[
                "Establish exact managed-system/LPAR identity and current processor, memory, virtual-I/O and boot/network assignments.",
                "Compare AIX guest observations with partition/HMC/PowerVM evidence without assuming the guest sees the full physical topology.",
                "Distinguish guest OS failure from unavailable or changed partition resources and VIOS-backed dependencies.",
                "Any DLPAR/profile/virtual-I/O/HMC change remains proposal-only with peer-impact and rollback review.",
            ],
        ),
        procedure(
            "legacy:aix:triage-powerha",
            LegacyKnowledgeAreaV1::AvailabilityAndRecovery,
            LegacyProcedureKindV1::Recover,
            "Triage AIX PowerHA partial failure",
            &[
                "Establish exact AIX TL/SP, PowerHA release/SP, cluster/node/resource-group identities and recent topology/change history.",
                "Inspect node, cluster, network, storage and resource-group state across members rather than relying on one healthy node.",
                "Separate local AIX health from cluster coordination, dependency, storage/network or version-compatibility problems.",
                "Any resource-group movement, cluster restart, DLPAR or storage/network change remains proposal-only with quorum/blast-radius/rollback review.",
            ],
        ),
    ];

    let mut ids = BTreeSet::new();
    for mut procedure in procedures {
        procedure.source_snapshots = snapshots.clone();
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
    descriptions: &[&str],
) -> LegacyProcedureV1 {
    let last = descriptions.len().saturating_sub(1);
    LegacyProcedureV1 {
        id: id.into(),
        platform: LegacyPlatformV1::Aix,
        area,
        kind,
        title: title.into(),
        applicability: aix_scope(),
        preconditions: vec![
            "Exact AIX release/TL/SP and target LPAR identity are established.".into(),
            "Diagnostic scope and operator authority are known before any active change is proposed.".into(),
        ],
        steps: descriptions
            .iter()
            .enumerate()
            .map(|(index, description)| LegacyProcedureStepV1 {
                authority: if index == last {
                    LegacyProcedureAuthorityV1::ChangeProposalOnly
                } else {
                    LegacyProcedureAuthorityV1::ReadOnlyObservation
                },
                description: (*description).into(),
            })
            .collect(),
        verification: vec![
            "Re-test the original symptom against the same scoped AIX/LPAR identity.".into(),
            "Confirm diagnosis or proposed change does not rely on stale topology or unrelated subsystem evidence.".into(),
        ],
        source_snapshots: BTreeSet::new(),
    }
}

fn seed_mechanisms() -> Vec<AixMechanismModelV1> {
    use AixMechanismKindV1 as M;
    vec![
        mechanism(
            M::DeviceConfigurationOdm,
            LegacyKnowledgeAreaV1::SystemLifecycle,
            "ODM-backed device configuration",
            "AIX maintains predefined and customized device objects whose configured state, attributes, parentage, driver methods and runtime availability are distinct diagnostic layers.",
            &[],
            &["legacy:aix:odm-device-state"],
            &[
                ("odm-device", "Predefined/customized ODM device identity, attributes, parent/location and current state."),
                ("runtime-path", "Current operating-system visibility and path/device availability."),
                ("driver-method", "Configured driver/method relationship and relevant configuration evidence."),
            ],
            &[
                ("defined-not-available", "A device remains defined in ODM but is not currently available through the expected runtime path.", &["ODM object exists", "runtime availability/path evidence fails"], &["deleting ODM objects before preserving attributes and verifying hardware/path state"]),
                ("stale-device-attributes", "Customized configuration no longer matches current adapter/storage topology.", &["configured attributes or parent/location differ from current topology evidence"], &["running broad reconfiguration without identifying affected devices and rollback"]),
            ],
        ),
        mechanism(
            M::LvmStorage,
            LegacyKnowledgeAreaV1::Storage,
            "LVM storage hierarchy",
            "AIX storage separates physical volumes, volume groups, partitions and logical volumes; filesystem symptoms must be localized within that hierarchy and underlying path state.",
            &[M::DeviceConfigurationOdm],
            &["legacy:aix:lvm-layering"],
            &[
                ("pv-vg-lv", "Physical-volume, volume-group, logical-volume and partition mapping/state."),
                ("mirror-state", "Copy/mirror allocation and stale/synchronized state where applicable."),
                ("filesystem-state", "Filesystem/mount/log relationship to the affected logical volume."),
            ],
            &[
                ("logical-not-physical", "An LV/filesystem allocation/state problem is mistaken for physical media failure.", &["physical path/PV remains healthy", "LV/filesystem state explains symptom"], &["replacing a disk before isolating the failed LVM layer"]),
                ("path-loss-versus-vg", "A storage path/device configuration problem makes a PV/VG appear unhealthy while LVM metadata is not the originating fault.", &["device/path evidence changed", "LVM symptom follows missing/degraded path"], &["forcing volume-group changes before restoring or understanding path state"]),
            ],
        ),
        mechanism(
            M::SrcSubsystems,
            LegacyKnowledgeAreaV1::WorkloadAndJobs,
            "System Resource Controller",
            "SRC manages subsystem and subserver lifecycle/status through a common interface, so service state, registered subsystem state and raw process state must not be conflated.",
            &[],
            &["legacy:aix:src-subsystem-control"],
            &[
                ("src-status", "Subsystem/subserver status and group identity from SRC."),
                ("process-state", "Current process/PID existence and termination behavior."),
                ("src-config", "Subsystem registration/configuration and notification/refresh characteristics."),
            ],
            &[
                ("process-up-src-down", "A daemon process exists but SRC state/configuration is inconsistent with the expected managed service.", &["process exists", "SRC status/registration differs"], &["killing the process before establishing SRC ownership and dependencies"]),
                ("src-up-service-unhealthy", "SRC reports an active subsystem while its application-level function is unhealthy.", &["SRC active", "functional/service evidence fails"], &["assuming active SRC status proves end-to-end service health"]),
            ],
        ),
        mechanism(
            M::ErrorLogging,
            LegacyKnowledgeAreaV1::ObservabilityAndProblemManagement,
            "AIX error logging",
            "AIX has a dedicated error-logging facility for hardware/software failures whose records should be temporally and causally correlated with other evidence rather than treated as self-proving root cause.",
            &[M::DeviceConfigurationOdm],
            &["legacy:aix:error-log"],
            &[
                ("errlog-record", "Error identifier, resource, timestamp, class/type and detail data."),
                ("failure-window", "Temporal relation between error-log records and the reported symptom/change window."),
                ("corroboration", "Matching device, LVM, service, platform or application evidence."),
            ],
            &[
                ("stale-error-anchor", "An older persistent error record is incorrectly assumed to explain a newer unrelated incident.", &["error timestamp predates failure window", "fresh evidence points elsewhere"], &["replacing/restarting the named component based on stale errpt history alone"]),
                ("symptom-not-root", "The error log records a downstream symptom produced by an upstream path/resource problem.", &["upstream evidence precedes/correlates with the logged error", "named resource is otherwise locally healthy"], &["treating the first error identifier as unique root cause without causal correlation"]),
            ],
        ),
        mechanism(
            M::Nim,
            LegacyKnowledgeAreaV1::SoftwareLifecycle,
            "Network Installation Management",
            "NIM represents masters, clients, resources and networks as management objects for BOS/software installation; NIM control/resource state and client firmware/boot/runtime state are separate layers.",
            &[M::LvmStorage],
            &["legacy:aix:nim-model"],
            &[
                ("nim-objects", "Master/client/network/resource object definitions and current allocations/control ownership."),
                ("resource-state", "lpp_source/SPOT/mksysb or other resource identity, level and availability."),
                ("boot-stage", "Whether failure occurs in firmware/SMS network boot, NIM transfer/install, or running AIX."),
            ],
            &[
                ("nim-object-versus-network", "NIM object/resource configuration is valid but firmware/SMS boot networking fails before AIX is running.", &["NIM objects/resources validate", "failure occurs before AIX runtime networking"], &["changing running AIX network configuration to solve firmware/SMS boot failure"]),
                ("resource-level-mismatch", "Client install/update fails because NIM resource levels or composition do not match the intended operation.", &["resource metadata differs from target/install expectation"], &["rebuilding all NIM resources before identifying the mismatched resource"]),
            ],
        ),
        mechanism(
            M::LogicalPartitions,
            LegacyKnowledgeAreaV1::VirtualizationAndPartitioning,
            "LPAR resource boundary",
            "AIX runs within an isolated logical partition whose processors, memory, I/O and virtualized resources can change independently of guest OS configuration.",
            &[M::DeviceConfigurationOdm],
            &["legacy:aix:lpar-resource-boundary"],
            &[
                ("lpar-profile", "Current versus profile processor/memory/I/O/virtual-adapter assignments."),
                ("guest-view", "AIX-visible CPU/memory/device/path observations."),
                ("vios-dependency", "Virtual-I/O backing/path relationship where applicable."),
            ],
            &[
                ("guest-healthy-resource-missing", "The AIX kernel is running but an expected virtual/physical resource is absent or changed at the partition/VIOS layer.", &["guest OS otherwise healthy", "partition/resource assignment differs"], &["reconfiguring AIX device state before confirming partition/VIOS resource assignment"]),
                ("profile-runtime-drift", "Saved LPAR profile and current dynamic allocation differ, misleading capacity/topology assumptions.", &["profile and current allocation differ"], &["activating/reapplying profiles without checking running workload impact"]),
            ],
        ),
        mechanism(
            M::PowerHa,
            LegacyKnowledgeAreaV1::AvailabilityAndRecovery,
            "PowerHA SystemMirror",
            "PowerHA cluster/resource-group health spans multiple nodes, networks, storage dependencies and version combinations; one healthy AIX node is insufficient evidence of cluster health.",
            &[M::LogicalPartitions, M::LvmStorage, M::SrcSubsystems],
            &["legacy:aix:powerha-version-context"],
            &[
                ("cluster-state", "Cluster/node/resource-group state across members."),
                ("dependency-state", "Network/storage/application-controller dependency evidence."),
                ("version-level", "AIX TL/SP and PowerHA release/SP compatibility context."),
            ],
            &[
                ("local-up-cluster-degraded", "An AIX node is locally healthy while cluster coordination or a resource group is degraded.", &["local OS checks pass", "cross-node/resource-group state is degraded"], &["restarting healthy local services before examining cluster/resource-group state"]),
                ("version-context-drift", "A cluster problem appears after AIX/PowerHA maintenance changes whose combination differs from tested/recommended levels.", &["maintenance level changed near failure", "cluster behavior differs across nodes/levels"], &["upgrading or downgrading cluster nodes ad hoc without a coordinated compatibility/rollback plan"]),
            ],
        ),
    ]
}

fn mechanism(
    kind: AixMechanismKindV1,
    area: LegacyKnowledgeAreaV1,
    title: &str,
    summary: &str,
    dependencies: &[AixMechanismKindV1],
    source_claims: &[&str],
    evidence: &[(&str, &str)],
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
            .map(|id| TechnicalClaimIdV1((*id).into()))
            .collect(),
        evidence_signals: evidence
            .iter()
            .map(|(id, description)| AixEvidenceSignalV1 {
                id: (*id).into(),
                description: (*description).into(),
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

fn require_nonempty(value: &str, field: &'static str) -> Result<(), LegacyAixErrorV1> {
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
    DuplicateEvidenceId(String),
    DuplicateFailureMode(String),
    UnknownClaim(TechnicalClaimIdV1),
    UnknownProcedure(String),
    ProcedureIdentityConflict(String),
}

impl fmt::Display for LegacyAixErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LegacyPack(err) => write!(f, "invalid legacy pack: {err}"),
            Self::Standards(err) => write!(f, "invalid AIX source registry data: {err}"),
            Self::UnsupportedSchema(value) => write!(f, "unsupported AIX schema {value}"),
            Self::InvalidField(value) => write!(f, "invalid AIX field: {value}"),
            Self::MissingAixProfile => write!(f, "legacy pack is missing the AIX profile"),
            Self::DuplicateMechanism(kind) => write!(f, "duplicate AIX mechanism {kind:?}"),
            Self::DuplicateEvidenceId(id) => write!(f, "duplicate AIX evidence id {id}"),
            Self::DuplicateFailureMode(id) => write!(f, "duplicate AIX failure mode {id}"),
            Self::UnknownClaim(id) => write!(f, "unknown AIX source claim {}", id.0),
            Self::UnknownProcedure(id) => write!(f, "unknown AIX procedure {id}"),
            Self::ProcedureIdentityConflict(id) => {
                write!(f, "AIX procedure identity conflict for {id}")
            }
        }
    }
}

impl Error for LegacyAixErrorV1 {}

impl From<LegacyComputingErrorV1> for LegacyAixErrorV1 {
    fn from(value: LegacyComputingErrorV1) -> Self {
        Self::LegacyPack(value)
    }
}

impl From<StandardsRegistryErrorV1> for LegacyAixErrorV1 {
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
        let first = enrich_legacy_aix_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        let claims = pack.sources.claims().count();
        let procedures = pack.procedures.len();
        let second = enrich_legacy_aix_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        assert_eq!(first, second);
        assert_eq!(pack.sources.claims().count(), claims);
        assert_eq!(pack.procedures.len(), procedures);
    }

    #[test]
    fn coverage_promotion_is_monotonic_and_narrows_real_gaps() {
        let mut pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        assert_eq!(
            pack.profile(LegacyPlatformV1::Aix)
                .unwrap()
                .state(LegacyKnowledgeAreaV1::ObservabilityAndProblemManagement),
            LegacyCoverageStateV1::ProcedureSeeded
        );
        enrich_legacy_aix_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        let profile = pack.profile(LegacyPlatformV1::Aix).unwrap();
        assert_eq!(
            profile.state(LegacyKnowledgeAreaV1::ObservabilityAndProblemManagement),
            LegacyCoverageStateV1::ProcedureSeeded
        );
        assert_eq!(
            profile.state(LegacyKnowledgeAreaV1::VirtualizationAndPartitioning),
            LegacyCoverageStateV1::ClaimSeeded
        );
        assert!(!profile.gap_tags.contains("nim"));
        assert!(!profile.gap_tags.contains("lpar-hmc"));
        assert!(!profile.gap_tags.contains("powerha"));
        assert!(profile.gap_tags.contains("aix-powerha-cluster-failure-labs"));
        assert!(profile.gap_tags.contains("aix-networking"));
    }

    #[test]
    fn all_mechanisms_are_source_bound_and_discriminative() {
        let mut pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        let foundation = enrich_legacy_aix_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
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
        let foundation = enrich_legacy_aix_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
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
    fn odm_configuration_and_runtime_device_state_stay_distinct() {
        let mut pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        let foundation = enrich_legacy_aix_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        let odm = foundation
            .mechanisms
            .iter()
            .find(|m| m.kind == AixMechanismKindV1::DeviceConfigurationOdm)
            .unwrap();
        assert!(odm
            .failure_modes
            .iter()
            .any(|failure| failure.id == "defined-not-available"));
    }
}
