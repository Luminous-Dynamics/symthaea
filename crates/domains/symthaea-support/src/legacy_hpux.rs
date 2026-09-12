// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Source-bound HP-UX 11i v3 foundation for legacy enterprise IT reasoning.
//!
//! This module deepens the HP-UX slice of `LegacyComputingPackV1` with explicit
//! LVM/VxFS, native multipathing/persistent DSF, Serviceguard, vPars/Integrity VM,
//! nPartitions, Ignite-UX, and Software Distributor mechanisms. It remains
//! advisory and non-executable; source snapshots remain metadata-only until the
//! retained-artifact path freezes exact source bytes.

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
use crate::technology::ApplicabilityScopeV1;
use crate::types::SupportCategory;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const LEGACY_HPUX_FOUNDATION_SCHEMA_V1: &str = "symthaea-it-legacy-hpux-foundation-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum HpuxMechanismKindV1 {
    LvmVxfs,
    NativeMultipathPersistentDsf,
    Serviceguard,
    VparsIntegrityVm,
    Npartitions,
    IgniteUx,
    SoftwareDistributor,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HpuxEvidenceSignalV1 {
    pub id: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HpuxFailureModeV1 {
    pub id: String,
    pub symptom: String,
    pub discriminators: Vec<String>,
    pub unsafe_shortcuts: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HpuxMechanismModelV1 {
    pub kind: HpuxMechanismKindV1,
    pub area: LegacyKnowledgeAreaV1,
    pub title: String,
    pub summary: String,
    pub dependencies: BTreeSet<HpuxMechanismKindV1>,
    pub source_claims: BTreeSet<TechnicalClaimIdV1>,
    pub evidence_signals: Vec<HpuxEvidenceSignalV1>,
    pub failure_modes: Vec<HpuxFailureModeV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HpuxFoundationV1 {
    pub schema_version: String,
    pub product_version: String,
    pub mechanisms: Vec<HpuxMechanismModelV1>,
    pub procedure_ids: BTreeSet<String>,
}

impl HpuxFoundationV1 {
    pub fn validate(&self, pack: &LegacyComputingPackV1) -> Result<(), LegacyHpuxErrorV1> {
        if self.schema_version != LEGACY_HPUX_FOUNDATION_SCHEMA_V1 {
            return Err(LegacyHpuxErrorV1::UnsupportedSchema(self.schema_version.clone()));
        }
        if self.product_version.trim().is_empty() || self.mechanisms.is_empty() {
            return Err(LegacyHpuxErrorV1::InvalidField(
                "HP-UX foundation requires product version and mechanisms".into(),
            ));
        }
        let mut kinds = BTreeSet::new();
        for mechanism in &self.mechanisms {
            if !kinds.insert(mechanism.kind) {
                return Err(LegacyHpuxErrorV1::DuplicateMechanism(mechanism.kind));
            }
            if mechanism.title.trim().is_empty()
                || mechanism.summary.trim().is_empty()
                || mechanism.source_claims.is_empty()
                || mechanism.evidence_signals.is_empty()
                || mechanism.failure_modes.is_empty()
            {
                return Err(LegacyHpuxErrorV1::InvalidField(format!(
                    "HP-UX mechanism {:?} is incomplete",
                    mechanism.kind
                )));
            }
            for claim_id in &mechanism.source_claims {
                if pack.sources.claim(claim_id).is_none() {
                    return Err(LegacyHpuxErrorV1::UnknownClaim(claim_id.clone()));
                }
            }
            let mut evidence_ids = BTreeSet::new();
            for signal in &mechanism.evidence_signals {
                if signal.id.trim().is_empty() || signal.description.trim().is_empty() {
                    return Err(LegacyHpuxErrorV1::InvalidField(
                        "HP-UX evidence fields must be non-empty".into(),
                    ));
                }
                if !evidence_ids.insert(signal.id.as_str()) {
                    return Err(LegacyHpuxErrorV1::DuplicateEvidenceId(signal.id.clone()));
                }
            }
            let mut failure_ids = BTreeSet::new();
            for failure in &mechanism.failure_modes {
                if failure.id.trim().is_empty()
                    || failure.symptom.trim().is_empty()
                    || failure.discriminators.is_empty()
                {
                    return Err(LegacyHpuxErrorV1::InvalidField(
                        "HP-UX failure mode requires id, symptom, and discriminators".into(),
                    ));
                }
                if !failure_ids.insert(failure.id.as_str()) {
                    return Err(LegacyHpuxErrorV1::DuplicateFailureMode(failure.id.clone()));
                }
            }
        }
        let procedure_ids: BTreeSet<_> = pack.procedures.iter().map(|p| p.id.as_str()).collect();
        for id in &self.procedure_ids {
            if !procedure_ids.contains(id.as_str()) {
                return Err(LegacyHpuxErrorV1::UnknownProcedure(id.clone()));
            }
        }
        Ok(())
    }
}

pub fn enrich_legacy_hpux_foundation_v1(
    pack: &mut LegacyComputingPackV1,
    fetched_at_unix_ms: u64,
) -> Result<HpuxFoundationV1, LegacyHpuxErrorV1> {
    if fetched_at_unix_ms == 0 {
        return Err(LegacyHpuxErrorV1::InvalidField(
            "HP-UX source fetch timestamp must be non-zero".into(),
        ));
    }
    pack.validate()?;
    let snapshots = register_hpux_sources(pack, fetched_at_unix_ms)?;
    register_hpux_claims(pack)?;
    let procedure_ids = add_hpux_procedures(pack, &snapshots)?;
    update_hpux_profile(pack, &snapshots)?;
    let foundation = HpuxFoundationV1 {
        schema_version: LEGACY_HPUX_FOUNDATION_SCHEMA_V1.into(),
        product_version: "11i v3".into(),
        mechanisms: seed_mechanisms(),
        procedure_ids,
    };
    pack.validate()?;
    foundation.validate(pack)?;
    Ok(foundation)
}

fn register_hpux_sources(
    pack: &mut LegacyComputingPackV1,
    fetched_at_unix_ms: u64,
) -> Result<BTreeSet<SourceSnapshotIdV1>, LegacyHpuxErrorV1> {
    let definitions = [
        (
            "hpe:hpux-alletra-lvm-vxfs",
            "hpe:hpux-alletra-lvm-vxfs@sd00003479",
            "HPE Alletra Storage MP B10000: HP-UX Implementation Guide",
            "HP-UX 11i v3 LVM/VxFS layered storage examples",
            "https://support.hpe.com/hpesc/public/docDisplay?docId=sd00003479en_us&docLocale=en_US",
            "11i v3",
        ),
        (
            "hpe:hpux-primera-multipath",
            "hpe:hpux-primera-multipath@sd00001339",
            "HPE Primera HP-UX Implementation Guide",
            "HP-UX 11i v3 persistent DSF and native multipathing",
            "https://support.hpe.com/hpesc/public/docDisplay?docId=sd00001339en_us&docLocale=en_US",
            "11i v3",
        ),
        (
            "hpe:hpux-serviceguard-storage",
            "hpe:hpux-serviceguard-storage@2025-12",
            "HPE Serviceguard Enterprise Cluster Master Toolkit User Guide",
            "Serviceguard on HP-UX 11i v3, December 2025",
            "https://support.hpe.com/hpesc/public/docDisplay?docId=sd00007156en_us&docLocale=en_US",
            "2025-12",
        ),
        (
            "hpe:hpux-vpars-integrityvm",
            "hpe:hpux-vpars-integrityvm@6.1",
            "HP-UX vPars and Integrity VM V6.1",
            "HP-UX vPars and Integrity VM 6.1",
            "https://support.hpe.com/hpesc/public/api/document/c03233037",
            "6.1",
        ),
        (
            "hpe:hpux-superdome-partitioning",
            "hpe:hpux-superdome-partitioning@c03607734",
            "HP Superdome 2 Partitioning Administrator",
            "Superdome 2 nPartitions and virtual partition administration",
            "https://support.hpe.com/hpesc/public/api/document/c03607734",
            "Superdome 2",
        ),
        (
            "hpe:hpux-install-update",
            "hpe:hpux-install-update@2025-05",
            "HP-UX 11i v3 Installation and Update Guide",
            "HP-UX 11i v3 Installation and Update Guide, May 2025",
            "https://support.hpe.com/hpesc/public/api/document/dp00006246en_us",
            "2025-05",
        ),
    ];
    let mut snapshots = BTreeSet::new();
    for (doc_id, snap_id, title, canonical_ref, locator, version) in definitions {
        pack.sources.register_document(TechnicalSourceDocumentV1 {
            id: SourceDocumentIdV1(doc_id.into()),
            publisher: TechnicalPublisherV1::Vendor("HPE".into()),
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

fn register_hpux_claims(pack: &mut LegacyComputingPackV1) -> Result<(), LegacyHpuxErrorV1> {
    let claims = [
        claim(
            "legacy:hpux:lvm-vxfs-layering",
            "hpe:hpux-alletra-lvm-vxfs@sd00003479",
            "HP-UX 11i v3 storage examples place LVM physical/volume-group/logical-volume state below VxFS filesystem state; filesystem symptoms and LUN/path health therefore occupy different diagnostic layers.",
            "Dynamic LUN expansion and LVM/VxFS examples",
            SupportCategory::Software,
            &[],
        ),
        claim(
            "legacy:hpux:native-multipath-persistent-dsf",
            "hpe:hpux-primera-multipath@sd00001339",
            "HP-UX 11i v3 uses native multipathing and persistent device special files for path-independent LUN identity; individual LUN-path state and persistent device identity must be distinguished.",
            "Setting up multipathing software—HP-UX 11i v3",
            SupportCategory::Hardware,
            &[],
        ),
        claim(
            "legacy:hpux:serviceguard-cluster-storage",
            "hpe:hpux-serviceguard-storage@2025-12",
            "Serviceguard adds cluster/package/failover and I/O-fencing semantics above local HP-UX and storage state; a locally healthy node or volume does not establish package availability or fencing correctness.",
            "Serviceguard support for ASM on HP-UX 11i v3 onwards",
            SupportCategory::Software,
            &["serviceguard"],
        ),
        claim(
            "legacy:hpux:vpars-integrityvm-boundary",
            "hpe:hpux-vpars-integrityvm@6.1",
            "vPars and Integrity VM 6.1 use a Virtual Server Platform to manage virtual servers and virtual/physical I/O, creating a host/guest/resource boundary distinct from HP-UX state inside the guest.",
            "Introduction and common manageability",
            SupportCategory::Hardware,
            &["vpars-integrity-vm:6.1"],
        ),
        claim(
            "legacy:hpux:npartitions-boundary",
            "hpe:hpux-superdome-partitioning@c03607734",
            "Superdome 2 nPartitions are managed through complex/OA partition configuration and firmware context; partition configuration and hardware-complex state are distinct from the HP-UX guest running inside a partition.",
            "Introduction and partition management",
            SupportCategory::Hardware,
            &["npartitions"],
        ),
        claim(
            "legacy:hpux:ignite-recovery-context",
            "hpe:hpux-install-update@2025-05",
            "The HP-UX 11i v3 installation/update lifecycle includes cold installation, recovery of customized data, storage/volume-group recovery context, and explicit post-install/update verification; successful installation alone does not prove full application recoverability.",
            "Installation, recovery, and post-install verification",
            SupportCategory::Software,
            &["ignite-ux"],
        ),
        claim(
            "legacy:hpux:software-distributor-verification",
            "hpe:hpux-install-update@2025-05",
            "HP-UX update procedures require distinguishing software selection/install state from post-install verification and configuration; installed depot/product state alone does not establish application correctness.",
            "Updating and verifying HP-UX software",
            SupportCategory::Software,
            &[],
        ),
    ];
    for claim in claims {
        pack.sources.register_claim(claim)?;
    }
    Ok(())
}

fn hpux_scope(required_features: &[&str]) -> ApplicabilityScopeV1 {
    let mut scope = legacy_platform_scope_v1(LegacyPlatformV1::HpUx);
    scope
        .required_features
        .extend(required_features.iter().map(|value| (*value).into()));
    scope
}

fn claim(
    id: &str,
    snapshot: &str,
    statement: &str,
    section: &str,
    category: SupportCategory,
    required_features: &[&str],
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
        applicability: Some(hpux_scope(required_features)),
        extraction_quality: Some(0.95),
        category: Some(category),
    }
}

fn seed_mechanisms() -> Vec<HpuxMechanismModelV1> {
    use HpuxMechanismKindV1 as M;
    vec![
        model(
            M::LvmVxfs,
            LegacyKnowledgeAreaV1::Storage,
            "LVM and VxFS layered storage",
            "Separate LUN/path/PV/VG/LV state from VxFS filesystem/mount/capacity state before attributing storage symptoms.",
            &[],
            &["legacy:hpux:lvm-vxfs-layering"],
            &[("lvm-state", "PV/VG/LV allocation, availability, mirror and extent state"), ("vxfs-state", "Filesystem/mount/capacity/journal state above the LV")],
            &[("filesystem-vs-lun", "Filesystem failure is incorrectly equated with physical LUN failure", &["Map symptom through filesystem/LV/VG/PV/path layers"], &["Replace/detach storage before isolating the failed layer"])],
        ),
        model(
            M::NativeMultipathPersistentDsf,
            LegacyKnowledgeAreaV1::Storage,
            "Native multipathing and persistent DSF identity",
            "Distinguish persistent LUN identity from individual hardware paths, legacy DSFs, path-state transitions, and multipath policy.",
            &[],
            &["legacy:hpux:native-multipath-persistent-dsf"],
            &[("persistent-dsf", "Persistent device identity/WWID and current DSF"), ("lunpaths", "Per-path state and hardware path mapping"), ("multipath-policy", "Load-balance/failover policy and current path set")],
            &[("single-path-not-lun-loss", "One LUN path fails while other paths and persistent device remain healthy", &["Compare WWID/persistent DSF with all LUN paths"], &["Remove/recreate persistent device identity because one path failed"]), ("stale-legacy-dsf", "Legacy path naming is mistaken for persistent device identity", &["Compare legacy versus persistent DSF and current WWID"], &["Delete device files before preserving path/application mappings"])],
        ),
        model(
            M::Serviceguard,
            LegacyKnowledgeAreaV1::AvailabilityAndRecovery,
            "Serviceguard cluster/package/fencing model",
            "Separate local-node health from package/resource/dependency state, shared storage activation, fencing, and failover eligibility.",
            &[M::LvmVxfs, M::NativeMultipathPersistentDsf],
            &["legacy:hpux:serviceguard-cluster-storage"],
            &[("cluster-package", "Cluster/node/package/resource state across members"), ("fencing-storage", "Shared-volume activation/fencing and path health"), ("dependency-state", "Package dependency and failover eligibility")],
            &[("local-up-package-down", "Local HP-UX node healthy while package/resources are degraded", &["Compare cluster/package/resource state across nodes"], &["Restart local application before cluster/fencing diagnosis"]), ("storage-up-fencing-wrong", "Shared storage reachable but package fencing/activation semantics are not satisfied", &["Distinguish path reachability from exclusive activation/fencing"], &["Force volume activation on multiple nodes"] )],
        ),
        model(
            M::VparsIntegrityVm,
            LegacyKnowledgeAreaV1::VirtualizationAndPartitioning,
            "vPars and Integrity VM host/guest boundary",
            "Separate VSP host state, VM/vPar definition, virtual/physical I/O backing, NPIV/network assignments, and HP-UX guest observations.",
            &[M::NativeMultipathPersistentDsf],
            &["legacy:hpux:vpars-integrityvm-boundary"],
            &[("vsp-state", "Virtual Server Platform and virtual-server state"), ("io-backing", "Virtual/direct I/O backing and assignment"), ("guest-state", "Guest-visible HP-UX state")],
            &[("guest-vs-backing", "Guest OS reports missing resource while host-side backing/assignment changed", &["Compare VSP definition/backing with guest view"], &["Reconfigure guest before checking host assignment"] )],
        ),
        model(
            M::Npartitions,
            LegacyKnowledgeAreaV1::VirtualizationAndPartitioning,
            "Superdome nPartition and complex boundary",
            "Separate OA/complex firmware and nPartition configuration/resources from the HP-UX instance inside the partition.",
            &[],
            &["legacy:hpux:npartitions-boundary"],
            &[("complex-state", "OA/complex and firmware status"), ("npar-config", "Partition cell/blade/I/O assignment and lifecycle"), ("guest-state", "HP-UX state inside the partition")],
            &[("partition-vs-guest", "HP-UX guest symptom originates in partition/firmware/resource state", &["Compare complex/nPartition state with guest observations"], &["Modify guest configuration before checking partition topology"] )],
        ),
        model(
            M::IgniteUx,
            LegacyKnowledgeAreaV1::AvailabilityAndRecovery,
            "Ignite-UX installation and recovery context",
            "Treat installation/image restoration, storage identity, customized-data recovery, and post-install verification as separate recovery stages.",
            &[M::LvmVxfs, M::NativeMultipathPersistentDsf],
            &["legacy:hpux:ignite-recovery-context"],
            &[("image-install", "Installation/recovery image and target state"), ("storage-identity", "Target DSFs/VGs/filesystems after recovery"), ("post-verify", "Recovered configuration/data/application verification")],
            &[("booted-not-recovered", "Recovered host boots but required data/config/application state is incomplete", &["Compare recovery objective with restored volumes/files/config and app verification"], &["Declare recovery complete because the kernel booted"] )],
        ),
        model(
            M::SoftwareDistributor,
            LegacyKnowledgeAreaV1::SoftwareLifecycle,
            "Software Distributor/update verification",
            "Separate depot/selection/install state, dependency/update result, verification, reboot/activation context, and application post-update health.",
            &[],
            &["legacy:hpux:software-distributor-verification"],
            &[("software-state", "Selected/installed software and update transaction state"), ("verification-state", "Post-install software verification and configuration"), ("app-health", "Application behavior after update")],
            &[("installed-not-healthy", "Package/update is installed but verification or application configuration fails", &["Compare install result, verification, configuration, and app health"], &["Reinstall the entire bundle before isolating failed verification/config"] )],
        ),
    ]
}

fn model(
    kind: HpuxMechanismKindV1,
    area: LegacyKnowledgeAreaV1,
    title: &str,
    summary: &str,
    dependencies: &[HpuxMechanismKindV1],
    claims: &[&str],
    evidence: &[(&str, &str)],
    failures: &[(&str, &str, &[&str], &[&str])],
) -> HpuxMechanismModelV1 {
    HpuxMechanismModelV1 {
        kind,
        area,
        title: title.into(),
        summary: summary.into(),
        dependencies: dependencies.iter().copied().collect(),
        source_claims: claims.iter().map(|id| TechnicalClaimIdV1((*id).into())).collect(),
        evidence_signals: evidence
            .iter()
            .map(|(id, description)| HpuxEvidenceSignalV1 {
                id: (*id).into(),
                description: (*description).into(),
            })
            .collect(),
        failure_modes: failures
            .iter()
            .map(|(id, symptom, discriminators, unsafe_shortcuts)| HpuxFailureModeV1 {
                id: (*id).into(),
                symptom: (*symptom).into(),
                discriminators: discriminators.iter().map(|v| (*v).into()).collect(),
                unsafe_shortcuts: unsafe_shortcuts.iter().map(|v| (*v).into()).collect(),
            })
            .collect(),
    }
}

fn add_hpux_procedures(
    pack: &mut LegacyComputingPackV1,
    snapshots: &BTreeSet<SourceSnapshotIdV1>,
) -> Result<BTreeSet<String>, LegacyHpuxErrorV1> {
    let defs = [
        ("legacy:hpux:lvm-vxfs-triage", LegacyKnowledgeAreaV1::Storage, LegacyProcedureKindV1::Diagnose, "HP-UX LVM/VxFS layered triage", "Map symptom through VxFS/LV/VG/PV/LUN/path layers before proposing storage changes.", &[][..], "hpe:hpux-alletra-lvm-vxfs@sd00003479"),
        ("legacy:hpux:multipath-triage", LegacyKnowledgeAreaV1::Storage, LegacyProcedureKindV1::Diagnose, "HP-UX native multipath triage", "Establish WWID/persistent DSF and compare every LUN path, path state, and multipath policy before proposing path/device changes.", &[][..], "hpe:hpux-primera-multipath@sd00001339"),
        ("legacy:hpux:serviceguard-triage", LegacyKnowledgeAreaV1::AvailabilityAndRecovery, LegacyProcedureKindV1::Recover, "HP-UX Serviceguard partial-failure triage", "Compare cluster/node/package/resource/dependency/fencing/storage state across members before proposing package movement or restart.", &["serviceguard"][..], "hpe:hpux-serviceguard-storage@2025-12"),
        ("legacy:hpux:vpars-vm-triage", LegacyKnowledgeAreaV1::VirtualizationAndPartitioning, LegacyProcedureKindV1::Diagnose, "HP-UX vPars/Integrity VM triage", "Compare VSP virtual-server definition and I/O backing with guest-visible state before proposing host/guest changes.", &["vpars-integrity-vm:6.1"][..], "hpe:hpux-vpars-integrityvm@6.1"),
        ("legacy:hpux:npar-triage", LegacyKnowledgeAreaV1::VirtualizationAndPartitioning, LegacyProcedureKindV1::Diagnose, "HP-UX nPartition triage", "Compare OA/complex firmware and partition topology/resources with the HP-UX guest before proposing partition or guest changes.", &["npartitions"][..], "hpe:hpux-superdome-partitioning@c03607734"),
        ("legacy:hpux:ignite-recovery-triage", LegacyKnowledgeAreaV1::AvailabilityAndRecovery, LegacyProcedureKindV1::Recover, "HP-UX Ignite-UX recovery verification", "Compare recovery objective, image/install state, persistent storage identity, restored customized data, and application verification before declaring recovery complete.", &["ignite-ux"][..], "hpe:hpux-install-update@2025-05"),
        ("legacy:hpux:software-update-triage", LegacyKnowledgeAreaV1::SoftwareLifecycle, LegacyProcedureKindV1::Diagnose, "HP-UX software update verification", "Separate selected/installed update state from verification, activation/reboot context, configuration, and application health.", &[][..], "hpe:hpux-install-update@2025-05"),
    ];
    let mut ids = BTreeSet::new();
    for (id, area, kind, title, diagnostic, required_features, source_snapshot) in defs {
        let source_snapshot = SourceSnapshotIdV1(source_snapshot.into());
        if !snapshots.contains(&source_snapshot) {
            return Err(LegacyHpuxErrorV1::InvalidField(format!(
                "HP-UX procedure {id} references unregistered source snapshot {}",
                source_snapshot.0
            )));
        }
        let procedure = LegacyProcedureV1 {
            id: id.into(),
            platform: LegacyPlatformV1::HpUx,
            area,
            kind,
            title: title.into(),
            applicability: hpux_scope(required_features),
            preconditions: vec![
                "Establish exact HP-UX 11i v3 update level, Integrity platform/partition identity, and relevant optional-product features.".into(),
                "Preserve evidence and operator authority; this advisory procedure does not authorize mutation.".into(),
            ],
            steps: vec![
                LegacyProcedureStepV1 {
                    authority: LegacyProcedureAuthorityV1::ReadOnlyObservation,
                    description: "Establish the failure window, affected resource identity, current topology, and recent change context.".into(),
                },
                LegacyProcedureStepV1 {
                    authority: LegacyProcedureAuthorityV1::ReadOnlyObservation,
                    description: diagnostic.into(),
                },
                LegacyProcedureStepV1 {
                    authority: LegacyProcedureAuthorityV1::ReadOnlyObservation,
                    description: "Separate stale/historical evidence from current direct evidence and verify source/version/feature applicability.".into(),
                },
                LegacyProcedureStepV1 {
                    authority: LegacyProcedureAuthorityV1::ChangeProposalOnly,
                    description: "If evidence supports intervention, propose the smallest reversible operator-reviewed change with blast-radius, rollback, and verification criteria.".into(),
                },
            ],
            verification: vec![
                "Re-observe the original symptom against the same host/partition/cluster/storage identity after any separately authorized intervention.".into(),
                "Verify adjacent storage paths, cluster members, virtual resources, and unrelated workloads did not regress.".into(),
            ],
            source_snapshots: [source_snapshot].into_iter().collect(),
        };
        insert_procedure_idempotent(pack, procedure)?;
        ids.insert(id.into());
    }
    Ok(ids)
}

fn insert_procedure_idempotent(
    pack: &mut LegacyComputingPackV1,
    procedure: LegacyProcedureV1,
) -> Result<(), LegacyHpuxErrorV1> {
    if let Some(existing) = pack.procedures.iter().find(|p| p.id == procedure.id) {
        if existing != &procedure {
            return Err(LegacyHpuxErrorV1::ProcedureConflict(procedure.id));
        }
        return Ok(());
    }
    procedure.validate(&pack.sources)?;
    pack.procedures.push(procedure);
    Ok(())
}

fn update_hpux_profile(
    pack: &mut LegacyComputingPackV1,
    snapshots: &BTreeSet<SourceSnapshotIdV1>,
) -> Result<(), LegacyHpuxErrorV1> {
    let profile = pack
        .profiles
        .iter_mut()
        .find(|p| p.platform == LegacyPlatformV1::HpUx)
        .ok_or(LegacyHpuxErrorV1::MissingHpuxProfile)?;
    profile.source_snapshots.extend(snapshots.iter().cloned());
    for area in [
        LegacyKnowledgeAreaV1::SystemLifecycle,
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
    for retired in ["serviceguard", "vpars-npars", "ignite-ux", "san-multipath"] {
        profile.gap_tags.remove(retired);
    }
    for gap in [
        "hpux-lvm-vxfs-recovery-depth",
        "hpux-native-multipath-san-failure-labs",
        "hpux-serviceguard-failure-labs",
        "hpux-vpars-integrityvm-runtime-depth",
        "hpux-npar-firmware-hardware-depth",
        "hpux-ignite-recovery-labs",
        "hpux-software-distributor-patch-depth",
    ] {
        profile.gap_tags.insert(gap.into());
    }
    Ok(())
}

#[derive(Debug)]
pub enum LegacyHpuxErrorV1 {
    Standards(StandardsRegistryErrorV1),
    Computing(LegacyComputingErrorV1),
    UnsupportedSchema(String),
    InvalidField(String),
    DuplicateMechanism(HpuxMechanismKindV1),
    DuplicateEvidenceId(String),
    DuplicateFailureMode(String),
    UnknownClaim(TechnicalClaimIdV1),
    UnknownProcedure(String),
    ProcedureConflict(String),
    MissingHpuxProfile,
}

impl fmt::Display for LegacyHpuxErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Standards(err) => write!(f, "HP-UX source registry error: {err}"),
            Self::Computing(err) => write!(f, "HP-UX legacy-pack error: {err}"),
            Self::UnsupportedSchema(schema) => write!(f, "unsupported HP-UX schema {schema}"),
            Self::InvalidField(message) => write!(f, "invalid HP-UX foundation: {message}"),
            Self::DuplicateMechanism(kind) => write!(f, "duplicate HP-UX mechanism {kind:?}"),
            Self::DuplicateEvidenceId(id) => write!(f, "duplicate HP-UX evidence id {id}"),
            Self::DuplicateFailureMode(id) => write!(f, "duplicate HP-UX failure mode {id}"),
            Self::UnknownClaim(id) => write!(f, "unknown HP-UX claim {}", id.0),
            Self::UnknownProcedure(id) => write!(f, "unknown HP-UX procedure {id}"),
            Self::ProcedureConflict(id) => write!(f, "conflicting HP-UX procedure {id}"),
            Self::MissingHpuxProfile => write!(f, "legacy pack is missing HP-UX profile"),
        }
    }
}

impl Error for LegacyHpuxErrorV1 {}

impl From<StandardsRegistryErrorV1> for LegacyHpuxErrorV1 {
    fn from(value: StandardsRegistryErrorV1) -> Self {
        Self::Standards(value)
    }
}

impl From<LegacyComputingErrorV1> for LegacyHpuxErrorV1 {
    fn from(value: LegacyComputingErrorV1) -> Self {
        Self::Computing(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::technology::ApplicabilityStatusV1;
    use crate::{legacy_platform_identity_v1, seed_legacy_computing_pack_v1};

    fn pack() -> LegacyComputingPackV1 {
        seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap()
    }

    #[test]
    fn enrichment_is_idempotent_and_monotonic() {
        let mut pack = pack();
        let before_obs = pack
            .profile(LegacyPlatformV1::HpUx)
            .unwrap()
            .state(LegacyKnowledgeAreaV1::ObservabilityAndProblemManagement);
        let first = enrich_legacy_hpux_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        let second = enrich_legacy_hpux_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.mechanisms.len(), 7);
        let profile = pack.profile(LegacyPlatformV1::HpUx).unwrap();
        assert_eq!(profile.state(LegacyKnowledgeAreaV1::ObservabilityAndProblemManagement), before_obs);
        assert!(profile.gap_tags.contains("hpux-networking"));
        assert!(profile.gap_tags.contains("security"));
        for retired in ["serviceguard", "vpars-npars", "ignite-ux", "san-multipath"] {
            assert!(!profile.gap_tags.contains(retired));
        }
    }

    #[test]
    fn every_mechanism_is_source_bound_and_discriminative() {
        let mut pack = pack();
        let foundation = enrich_legacy_hpux_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        for mechanism in foundation.mechanisms {
            assert!(!mechanism.source_claims.is_empty());
            assert!(!mechanism.evidence_signals.is_empty());
            assert!(mechanism.failure_modes.iter().all(|f| !f.discriminators.is_empty()));
        }
    }

    #[test]
    fn optional_products_require_positive_feature_evidence() {
        let mut pack = pack();
        enrich_legacy_hpux_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        let plain_hpux = legacy_platform_identity_v1(LegacyPlatformV1::HpUx);
        for claim_id in [
            "legacy:hpux:serviceguard-cluster-storage",
            "legacy:hpux:vpars-integrityvm-boundary",
            "legacy:hpux:npartitions-boundary",
            "legacy:hpux:ignite-recovery-context",
        ] {
            let claim = pack.sources.claim(&TechnicalClaimIdV1(claim_id.into())).unwrap();
            let assessment = claim.applicability.as_ref().unwrap().assess(&plain_hpux).unwrap();
            assert_eq!(assessment.status, ApplicabilityStatusV1::Indeterminate);
        }
    }

    #[test]
    fn single_multipath_failure_is_not_lun_loss() {
        let mut pack = pack();
        let foundation = enrich_legacy_hpux_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        let multipath = foundation
            .mechanisms
            .iter()
            .find(|m| m.kind == HpuxMechanismKindV1::NativeMultipathPersistentDsf)
            .unwrap();
        assert!(multipath.failure_modes.iter().any(|f| f.id == "single-path-not-lun-loss"));
    }

    #[test]
    fn advisory_procedures_never_mint_execution_authority() {
        let mut pack = pack();
        let foundation = enrich_legacy_hpux_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
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
    fn advisory_procedures_use_mechanism_scoped_source_snapshots() {
        let mut pack = pack();
        enrich_legacy_hpux_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        let expected = [
            ("legacy:hpux:lvm-vxfs-triage", "hpe:hpux-alletra-lvm-vxfs@sd00003479"),
            ("legacy:hpux:multipath-triage", "hpe:hpux-primera-multipath@sd00001339"),
            ("legacy:hpux:serviceguard-triage", "hpe:hpux-serviceguard-storage@2025-12"),
            ("legacy:hpux:vpars-vm-triage", "hpe:hpux-vpars-integrityvm@6.1"),
            ("legacy:hpux:npar-triage", "hpe:hpux-superdome-partitioning@c03607734"),
            ("legacy:hpux:ignite-recovery-triage", "hpe:hpux-install-update@2025-05"),
            ("legacy:hpux:software-update-triage", "hpe:hpux-install-update@2025-05"),
        ];
        for (procedure_id, snapshot_id) in expected {
            let procedure = pack.procedures.iter().find(|p| p.id == procedure_id).unwrap();
            assert_eq!(procedure.source_snapshots.len(), 1);
            assert!(procedure
                .source_snapshots
                .contains(&SourceSnapshotIdV1(snapshot_id.into())));
        }
    }
}
