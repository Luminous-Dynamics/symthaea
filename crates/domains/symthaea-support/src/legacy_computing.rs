// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Source-bound legacy enterprise computing knowledge pack.
//!
//! This module starts filling the `LegacyComputing` IT-domain gap without
//! turning old vendor documentation into unversioned folklore. It composes the
//! existing technical standards/source registry and keeps platform identity,
//! version applicability, coverage state, and operational procedure semantics
//! explicit.
//!
//! V1 is deliberately advisory. Procedures contain observations and proposal
//! semantics, never executable command strings or authority tokens.

use crate::knowledge_source::{
    KnowledgeAuthorityClassV1, KnowledgeLifecycleV1, KnowledgeStabilityV1,
};
use crate::standards_registry::{
    ClaimModalityV1, SourceCaptureV1, SourceDocumentIdV1, SourceDocumentKindV1,
    SourceSnapshotIdV1, StandardsRegistryErrorV1, TechnicalClaimIdV1,
    TechnicalKnowledgeClaimV1, TechnicalPublisherV1, TechnicalSourceDocumentV1,
    TechnicalSourceLocatorV1, TechnicalSourceSnapshotV1, TechnicalStandardsRegistryV1,
};
use crate::technology::{ApplicabilityScopeV1, StringSelectorV1};
use crate::types::SupportCategory;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const LEGACY_COMPUTING_PACK_SCHEMA_V1: &str = "symthaea-it-legacy-computing-pack-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum LegacyPlatformV1 {
    Zos,
    IbmI,
    Aix,
    Solaris,
    HpUx,
}

impl LegacyPlatformV1 {
    pub const ALL: [Self; 5] = [Self::Zos, Self::IbmI, Self::Aix, Self::Solaris, Self::HpUx];

    pub fn product_name(self) -> &'static str {
        match self {
            Self::Zos => "z/OS",
            Self::IbmI => "IBM i",
            Self::Aix => "AIX",
            Self::Solaris => "Oracle Solaris",
            Self::HpUx => "HP-UX",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum LegacyKnowledgeAreaV1 {
    SystemLifecycle,
    WorkloadAndJobs,
    Storage,
    Networking,
    IdentityAndSecurity,
    ObservabilityAndProblemManagement,
    AvailabilityAndRecovery,
    VirtualizationAndPartitioning,
    SoftwareLifecycle,
    InteroperabilityAndMigration,
}

impl LegacyKnowledgeAreaV1 {
    pub const ALL: [Self; 10] = [
        Self::SystemLifecycle,
        Self::WorkloadAndJobs,
        Self::Storage,
        Self::Networking,
        Self::IdentityAndSecurity,
        Self::ObservabilityAndProblemManagement,
        Self::AvailabilityAndRecovery,
        Self::VirtualizationAndPartitioning,
        Self::SoftwareLifecycle,
        Self::InteroperabilityAndMigration,
    ];
}

/// Repository knowledge maturity only. None of these states is a competence claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LegacyCoverageStateV1 {
    Unmapped,
    SourceMapped,
    ClaimSeeded,
    ProcedureSeeded,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyPlatformProfileV1 {
    pub platform: LegacyPlatformV1,
    pub version_family: String,
    pub source_snapshots: BTreeSet<SourceSnapshotIdV1>,
    pub coverage: BTreeMap<LegacyKnowledgeAreaV1, LegacyCoverageStateV1>,
    pub gap_tags: BTreeSet<String>,
}

impl LegacyPlatformProfileV1 {
    pub fn state(&self, area: LegacyKnowledgeAreaV1) -> LegacyCoverageStateV1 {
        self.coverage
            .get(&area)
            .copied()
            .unwrap_or(LegacyCoverageStateV1::Unmapped)
    }

    pub fn unmapped_areas(&self) -> Vec<LegacyKnowledgeAreaV1> {
        LegacyKnowledgeAreaV1::ALL
            .into_iter()
            .filter(|area| self.state(*area) == LegacyCoverageStateV1::Unmapped)
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LegacyProcedureKindV1 {
    Observe,
    Diagnose,
    Recover,
    Migrate,
    Verify,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LegacyProcedureAuthorityV1 {
    ReadOnlyObservation,
    ChangeProposalOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyProcedureStepV1 {
    pub authority: LegacyProcedureAuthorityV1,
    pub description: String,
}

/// Advisory procedure. It carries no executable command, capability, or executor hook.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyProcedureV1 {
    pub id: String,
    pub platform: LegacyPlatformV1,
    pub area: LegacyKnowledgeAreaV1,
    pub kind: LegacyProcedureKindV1,
    pub title: String,
    pub applicability: ApplicabilityScopeV1,
    pub preconditions: Vec<String>,
    pub steps: Vec<LegacyProcedureStepV1>,
    pub verification: Vec<String>,
    pub source_snapshots: BTreeSet<SourceSnapshotIdV1>,
}

impl LegacyProcedureV1 {
    pub fn validate(
        &self,
        registry: &TechnicalStandardsRegistryV1,
    ) -> Result<(), LegacyComputingErrorV1> {
        require_nonempty(&self.id, "procedure id")?;
        require_nonempty(&self.title, "procedure title")?;
        self.applicability
            .validate()
            .map_err(|err| LegacyComputingErrorV1::InvalidApplicability(err.to_string()))?;
        if self.preconditions.is_empty() {
            return Err(LegacyComputingErrorV1::InvalidField(
                "legacy procedure requires preconditions".into(),
            ));
        }
        if self.steps.is_empty() {
            return Err(LegacyComputingErrorV1::InvalidField(
                "legacy procedure requires steps".into(),
            ));
        }
        if self.verification.is_empty() {
            return Err(LegacyComputingErrorV1::InvalidField(
                "legacy procedure requires verification".into(),
            ));
        }
        if self.source_snapshots.is_empty() {
            return Err(LegacyComputingErrorV1::InvalidField(
                "legacy procedure requires source snapshots".into(),
            ));
        }
        for value in self
            .preconditions
            .iter()
            .chain(self.verification.iter())
            .chain(self.steps.iter().map(|step| &step.description))
        {
            require_nonempty(value, "procedure text")?;
        }
        for snapshot in &self.source_snapshots {
            if registry.snapshot(snapshot).is_none() {
                return Err(LegacyComputingErrorV1::UnknownSourceSnapshot(
                    snapshot.clone(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LegacyComputingPackV1 {
    pub schema_version: String,
    pub sources: TechnicalStandardsRegistryV1,
    pub profiles: Vec<LegacyPlatformProfileV1>,
    pub procedures: Vec<LegacyProcedureV1>,
}

impl LegacyComputingPackV1 {
    pub fn validate(&self) -> Result<(), LegacyComputingErrorV1> {
        if self.schema_version != LEGACY_COMPUTING_PACK_SCHEMA_V1 {
            return Err(LegacyComputingErrorV1::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        self.sources.validate_relations()?;

        let mut platforms = BTreeSet::new();
        for profile in &self.profiles {
            if !platforms.insert(profile.platform) {
                return Err(LegacyComputingErrorV1::DuplicatePlatform(profile.platform));
            }
            require_nonempty(&profile.version_family, "version family")?;
            for area in LegacyKnowledgeAreaV1::ALL {
                if !profile.coverage.contains_key(&area) {
                    return Err(LegacyComputingErrorV1::MissingCoverageArea {
                        platform: profile.platform,
                        area,
                    });
                }
            }
            for snapshot in &profile.source_snapshots {
                if self.sources.snapshot(snapshot).is_none() {
                    return Err(LegacyComputingErrorV1::UnknownSourceSnapshot(
                        snapshot.clone(),
                    ));
                }
            }
            if profile.gap_tags.iter().any(|tag| tag.trim().is_empty()) {
                return Err(LegacyComputingErrorV1::InvalidField(
                    "empty legacy gap tag".into(),
                ));
            }
        }
        for platform in LegacyPlatformV1::ALL {
            if !platforms.contains(&platform) {
                return Err(LegacyComputingErrorV1::MissingPlatform(platform));
            }
        }

        let mut procedure_ids = BTreeSet::new();
        for procedure in &self.procedures {
            if !procedure_ids.insert(procedure.id.as_str()) {
                return Err(LegacyComputingErrorV1::DuplicateProcedure(
                    procedure.id.clone(),
                ));
            }
            procedure.validate(&self.sources)?;
        }
        Ok(())
    }

    pub fn profile(&self, platform: LegacyPlatformV1) -> Option<&LegacyPlatformProfileV1> {
        self.profiles.iter().find(|profile| profile.platform == platform)
    }
}

pub fn seed_legacy_computing_pack_v1(
    fetched_at_unix_ms: u64,
) -> Result<LegacyComputingPackV1, LegacyComputingErrorV1> {
    if fetched_at_unix_ms == 0 {
        return Err(LegacyComputingErrorV1::InvalidField(
            "source fetch timestamp must be non-zero".into(),
        ));
    }

    let mut sources = TechnicalStandardsRegistryV1::new();
    register_source(
        &mut sources,
        "ibm:zos-system-level",
        "ibm:zos-system-level@3.2",
        "IBM",
        "z/OS System-Level",
        "z/OS 3.2 system-level documentation",
        "https://www.ibm.com/docs/en/zos/3.2.0?topic=zos-system-level",
        "3.2",
        KnowledgeLifecycleV1::Active,
        fetched_at_unix_ms,
    )?;
    register_source(
        &mut sources,
        "ibm:ibmi-docs",
        "ibm:ibmi-docs@7.6",
        "IBM",
        "IBM i documentation",
        "IBM i 7.6 documentation",
        "https://www.ibm.com/docs/en/i/7.6.0",
        "7.6",
        KnowledgeLifecycleV1::Active,
        fetched_at_unix_ms,
    )?;
    register_source(
        &mut sources,
        "ibm:aix-os-management",
        "ibm:aix-os-management@7.3",
        "IBM",
        "AIX operating system management",
        "AIX 7.3 operating system management",
        "https://www.ibm.com/docs/en/aix/7.3.0?topic=operating-system-management",
        "7.3",
        KnowledgeLifecycleV1::Active,
        fetched_at_unix_ms,
    )?;
    register_source(
        &mut sources,
        "oracle:solaris-docs",
        "oracle:solaris-docs@11.4",
        "Oracle",
        "Oracle Solaris 11.4 Documentation Library",
        "Oracle Solaris 11.4",
        "https://docs.oracle.com/en/operating-systems/solaris/oracle-solaris/11.4/",
        "11.4",
        KnowledgeLifecycleV1::Active,
        fetched_at_unix_ms,
    )?;
    register_source(
        &mut sources,
        "hpe:hpux-admin-overview",
        "hpe:hpux-admin-overview@11iv3",
        "HPE",
        "HP-UX System Administration - Overview",
        "HP-UX 11i v3 system administration",
        "https://support.hpe.com/hpesc/public/docDisplay?docId=c02921284&docLocale=en_US",
        "11i v3",
        KnowledgeLifecycleV1::Unknown,
        fetched_at_unix_ms,
    )?;
    register_source(
        &mut sources,
        "hpe:hpux-install-update",
        "hpe:hpux-install-update@11iv3-2025-05",
        "HPE",
        "HP-UX 11i v3 Installation and Update Guide",
        "HP-UX 11i v3 Installation and Update Guide, May 2025",
        "https://support.hpe.com/hpesc/public/api/document/dp00006246en_us",
        "11i v3 / May 2025",
        KnowledgeLifecycleV1::Unknown,
        fetched_at_unix_ms,
    )?;

    register_claim(
        &mut sources,
        "legacy:zos:system-level-scope",
        "ibm:zos-system-level@3.2",
        "z/OS system-level documentation covers migration, installation, problem management, and IBM Health Checker guidance.",
        "Description",
        scope(LegacyPlatformV1::Zos, "IBM", "z/OS", "3.2"),
    )?;
    register_claim(
        &mut sources,
        "legacy:aix:os-management-scope",
        "ibm:aix-os-management@7.3",
        "AIX operating system management documentation covers processes, files, backup and restore, physical and logical storage, paging, devices, system resources, and shell administration.",
        "Operating system management",
        scope(LegacyPlatformV1::Aix, "IBM", "AIX", "7.3"),
    )?;
    register_claim(
        &mut sources,
        "legacy:solaris:admin-library-scope",
        "oracle:solaris-docs@11.4",
        "Oracle Solaris 11.4 documentation includes system administration, networking, ZFS and storage, zones virtualization, DTrace and fault management, security, and migration guidance.",
        "Documentation Library",
        scope(LegacyPlatformV1::Solaris, "Oracle", "Oracle Solaris", "11.4"),
    )?;
    register_claim(
        &mut sources,
        "legacy:hpux:admin-overview-scope",
        "hpe:hpux-admin-overview@11iv3",
        "HP-UX system administration documentation covers logical volume management and enterprise virtualization concepts.",
        "Product features",
        scope(LegacyPlatformV1::HpUx, "HPE", "HP-UX", "11i v3"),
    )?;
    register_claim(
        &mut sources,
        "legacy:hpux:install-verification",
        "hpe:hpux-install-update@11iv3-2025-05",
        "HP-UX 11i v3 installation and update guidance includes post-install verification, backup, diagnostics, and update troubleshooting.",
        "Post-install/update tasks and troubleshooting",
        scope(LegacyPlatformV1::HpUx, "HPE", "HP-UX", "11i v3"),
    )?;

    let profiles = vec![
        profile(
            LegacyPlatformV1::Zos,
            "3.2",
            &["ibm:zos-system-level@3.2"],
            &[
                LegacyKnowledgeAreaV1::SystemLifecycle,
                LegacyKnowledgeAreaV1::ObservabilityAndProblemManagement,
                LegacyKnowledgeAreaV1::SoftwareLifecycle,
            ],
            &["jcl-and-batch", "dataset-and-vsam", "jes", "racf", "sysplex", "tcpip-and-sna"],
        ),
        profile(
            LegacyPlatformV1::IbmI,
            "7.6",
            &["ibm:ibmi-docs@7.6"],
            &[
                LegacyKnowledgeAreaV1::SystemLifecycle,
                LegacyKnowledgeAreaV1::AvailabilityAndRecovery,
                LegacyKnowledgeAreaV1::SoftwareLifecycle,
            ],
            &["object-library-model", "jobs-and-subsystems", "db2-for-i", "cl", "security-authority", "save-restore"],
        ),
        profile(
            LegacyPlatformV1::Aix,
            "7.3",
            &["ibm:aix-os-management@7.3"],
            &[
                LegacyKnowledgeAreaV1::SystemLifecycle,
                LegacyKnowledgeAreaV1::WorkloadAndJobs,
                LegacyKnowledgeAreaV1::Storage,
                LegacyKnowledgeAreaV1::IdentityAndSecurity,
                LegacyKnowledgeAreaV1::AvailabilityAndRecovery,
                LegacyKnowledgeAreaV1::SoftwareLifecycle,
            ],
            &["lpar-hmc", "nim", "powerha", "aix-networking", "smit-deep-procedures"],
        ),
        profile(
            LegacyPlatformV1::Solaris,
            "11.4",
            &["oracle:solaris-docs@11.4"],
            &[
                LegacyKnowledgeAreaV1::SystemLifecycle,
                LegacyKnowledgeAreaV1::WorkloadAndJobs,
                LegacyKnowledgeAreaV1::Storage,
                LegacyKnowledgeAreaV1::Networking,
                LegacyKnowledgeAreaV1::IdentityAndSecurity,
                LegacyKnowledgeAreaV1::ObservabilityAndProblemManagement,
                LegacyKnowledgeAreaV1::AvailabilityAndRecovery,
                LegacyKnowledgeAreaV1::VirtualizationAndPartitioning,
                LegacyKnowledgeAreaV1::SoftwareLifecycle,
                LegacyKnowledgeAreaV1::InteroperabilityAndMigration,
            ],
            &["solaris-10-legacy", "sparc-platform-depth", "zones-failure-scenarios", "dtrace-diagnostics"],
        ),
        profile(
            LegacyPlatformV1::HpUx,
            "11i v3",
            &[
                "hpe:hpux-admin-overview@11iv3",
                "hpe:hpux-install-update@11iv3-2025-05",
            ],
            &[
                LegacyKnowledgeAreaV1::SystemLifecycle,
                LegacyKnowledgeAreaV1::Storage,
                LegacyKnowledgeAreaV1::AvailabilityAndRecovery,
                LegacyKnowledgeAreaV1::VirtualizationAndPartitioning,
                LegacyKnowledgeAreaV1::SoftwareLifecycle,
            ],
            &["serviceguard", "vpars-npars", "hpux-networking", "security", "ignite-ux", "san-multipath"],
        ),
    ];

    let procedures = vec![
        first_response_procedure(
            LegacyPlatformV1::Zos,
            "3.2",
            "IBM",
            "z/OS",
            "legacy:zos:first-response",
            "ibm:zos-system-level@3.2",
        ),
        first_response_procedure(
            LegacyPlatformV1::IbmI,
            "7.6",
            "IBM",
            "IBM i",
            "legacy:ibmi:first-response",
            "ibm:ibmi-docs@7.6",
        ),
        first_response_procedure(
            LegacyPlatformV1::Aix,
            "7.3",
            "IBM",
            "AIX",
            "legacy:aix:first-response",
            "ibm:aix-os-management@7.3",
        ),
        first_response_procedure(
            LegacyPlatformV1::Solaris,
            "11.4",
            "Oracle",
            "Oracle Solaris",
            "legacy:solaris:first-response",
            "oracle:solaris-docs@11.4",
        ),
        first_response_procedure(
            LegacyPlatformV1::HpUx,
            "11i v3",
            "HPE",
            "HP-UX",
            "legacy:hpux:first-response",
            "hpe:hpux-install-update@11iv3-2025-05",
        ),
    ];

    let mut pack = LegacyComputingPackV1 {
        schema_version: LEGACY_COMPUTING_PACK_SCHEMA_V1.into(),
        sources,
        profiles,
        procedures,
    };

    // A seeded advisory procedure raises only the matching problem-management area.
    for procedure in &pack.procedures {
        if let Some(profile) = pack
            .profiles
            .iter_mut()
            .find(|profile| profile.platform == procedure.platform)
        {
            profile.coverage.insert(
                LegacyKnowledgeAreaV1::ObservabilityAndProblemManagement,
                LegacyCoverageStateV1::ProcedureSeeded,
            );
        }
    }

    pack.validate()?;
    Ok(pack)
}

fn register_source(
    registry: &mut TechnicalStandardsRegistryV1,
    document_id: &str,
    snapshot_id: &str,
    vendor: &str,
    title: &str,
    canonical_ref: &str,
    locator: &str,
    version: &str,
    lifecycle: KnowledgeLifecycleV1,
    fetched_at_unix_ms: u64,
) -> Result<(), LegacyComputingErrorV1> {
    registry.register_document(TechnicalSourceDocumentV1 {
        id: SourceDocumentIdV1(document_id.into()),
        publisher: TechnicalPublisherV1::Vendor(vendor.into()),
        kind: SourceDocumentKindV1::VendorDocumentation,
        title: title.into(),
        canonical_ref: canonical_ref.into(),
        canonical_locator: Some(locator.into()),
    })?;
    registry.register_snapshot(TechnicalSourceSnapshotV1 {
        id: SourceSnapshotIdV1(snapshot_id.into()),
        document_id: SourceDocumentIdV1(document_id.into()),
        version: Some(version.into()),
        lifecycle,
        authority: KnowledgeAuthorityClassV1::VendorDocumentation,
        stability: KnowledgeStabilityV1::Stable,
        published_at_unix_ms: None,
        source_updated_at_unix_ms: None,
        fetched_at_unix_ms,
        capture: SourceCaptureV1::MetadataOnly,
        relations: BTreeSet::new(),
    })?;
    Ok(())
}

fn register_claim(
    registry: &mut TechnicalStandardsRegistryV1,
    id: &str,
    snapshot: &str,
    statement: &str,
    section: &str,
    applicability: ApplicabilityScopeV1,
) -> Result<(), LegacyComputingErrorV1> {
    registry.register_claim(TechnicalKnowledgeClaimV1 {
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
        category: Some(SupportCategory::Software),
    })?;
    Ok(())
}

fn profile(
    platform: LegacyPlatformV1,
    version_family: &str,
    snapshots: &[&str],
    source_mapped: &[LegacyKnowledgeAreaV1],
    gaps: &[&str],
) -> LegacyPlatformProfileV1 {
    let mut coverage = LegacyKnowledgeAreaV1::ALL
        .into_iter()
        .map(|area| (area, LegacyCoverageStateV1::Unmapped))
        .collect::<BTreeMap<_, _>>();
    for area in source_mapped {
        coverage.insert(*area, LegacyCoverageStateV1::SourceMapped);
    }
    LegacyPlatformProfileV1 {
        platform,
        version_family: version_family.into(),
        source_snapshots: snapshots
            .iter()
            .map(|id| SourceSnapshotIdV1((*id).into()))
            .collect(),
        coverage,
        gap_tags: gaps.iter().map(|gap| (*gap).into()).collect(),
    }
}

fn scope(
    _platform: LegacyPlatformV1,
    vendor: &str,
    product: &str,
    version_prefix: &str,
) -> ApplicabilityScopeV1 {
    ApplicabilityScopeV1 {
        ecosystem: StringSelectorV1::Exact("legacy-enterprise-os".into()),
        vendor: StringSelectorV1::Exact(vendor.into()),
        product: StringSelectorV1::Exact(product.into()),
        version: StringSelectorV1::Prefix(version_prefix.into()),
        ..ApplicabilityScopeV1::default()
    }
}

fn first_response_procedure(
    platform: LegacyPlatformV1,
    version: &str,
    vendor: &str,
    product: &str,
    id: &str,
    source_snapshot: &str,
) -> LegacyProcedureV1 {
    LegacyProcedureV1 {
        id: id.into(),
        platform,
        area: LegacyKnowledgeAreaV1::ObservabilityAndProblemManagement,
        kind: LegacyProcedureKindV1::Diagnose,
        title: format!("{} non-destructive first-response triage", platform.product_name()),
        applicability: scope(platform, vendor, product, version),
        preconditions: vec![
            "Establish exact platform, release family, architecture, and maintenance context before applying version-specific knowledge.".into(),
            "Preserve operator authority boundaries; this advisory procedure does not authorize changes.".into(),
        ],
        steps: vec![
            LegacyProcedureStepV1 {
                authority: LegacyProcedureAuthorityV1::ReadOnlyObservation,
                description: "Establish the failure window, affected workloads, recent changes, and whether the symptom is system-wide or scoped to one subsystem.".into(),
            },
            LegacyProcedureStepV1 {
                authority: LegacyProcedureAuthorityV1::ReadOnlyObservation,
                description: "Collect platform-native health, problem-management, storage, network, workload, and availability evidence without changing system state.".into(),
            },
            LegacyProcedureStepV1 {
                authority: LegacyProcedureAuthorityV1::ReadOnlyObservation,
                description: "Check source applicability and maintenance level before interpreting version-sensitive behavior.".into(),
            },
            LegacyProcedureStepV1 {
                authority: LegacyProcedureAuthorityV1::ChangeProposalOnly,
                description: "If evidence supports a repair, formulate the smallest reversible operator-reviewed change with explicit rollback and verification criteria.".into(),
            },
        ],
        verification: vec![
            "Re-observe the original symptom and the directly affected subsystem after any separately authorized intervention.".into(),
            "Verify that unrelated workloads, storage, network, and availability indicators did not regress.".into(),
        ],
        source_snapshots: BTreeSet::from([SourceSnapshotIdV1(source_snapshot.into())]),
    }
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), LegacyComputingErrorV1> {
    if value.trim().is_empty() {
        Err(LegacyComputingErrorV1::InvalidField(field.into()))
    } else {
        Ok(())
    }
}

#[derive(Debug)]
pub enum LegacyComputingErrorV1 {
    Standards(StandardsRegistryErrorV1),
    UnsupportedSchema(String),
    InvalidField(String),
    InvalidApplicability(String),
    MissingPlatform(LegacyPlatformV1),
    DuplicatePlatform(LegacyPlatformV1),
    MissingCoverageArea {
        platform: LegacyPlatformV1,
        area: LegacyKnowledgeAreaV1,
    },
    DuplicateProcedure(String),
    UnknownSourceSnapshot(SourceSnapshotIdV1),
}

impl fmt::Display for LegacyComputingErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Standards(err) => write!(f, "legacy source registry error: {err}"),
            Self::UnsupportedSchema(schema) => write!(f, "unsupported legacy pack schema {schema}"),
            Self::InvalidField(field) => write!(f, "invalid legacy computing field: {field}"),
            Self::InvalidApplicability(message) => write!(f, "invalid legacy applicability: {message}"),
            Self::MissingPlatform(platform) => write!(f, "missing legacy platform {platform:?}"),
            Self::DuplicatePlatform(platform) => write!(f, "duplicate legacy platform {platform:?}"),
            Self::MissingCoverageArea { platform, area } => {
                write!(f, "missing legacy coverage cell {platform:?}/{area:?}")
            }
            Self::DuplicateProcedure(id) => write!(f, "duplicate legacy procedure {id}"),
            Self::UnknownSourceSnapshot(id) => write!(f, "unknown legacy source snapshot {}", id.0),
        }
    }
}

impl Error for LegacyComputingErrorV1 {}

impl From<StandardsRegistryErrorV1> for LegacyComputingErrorV1 {
    fn from(value: StandardsRegistryErrorV1) -> Self {
        Self::Standards(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge_source::{KnowledgeQueryPurposeV1, SupportKnowledgeQueryV1, SupportKnowledgeSourceV1};
    use crate::technology::TechnologyIdentityV1;

    #[test]
    fn seed_pack_represents_every_platform_and_area_without_claiming_completion() {
        let pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        assert_eq!(pack.profiles.len(), LegacyPlatformV1::ALL.len());
        for platform in LegacyPlatformV1::ALL {
            let profile = pack.profile(platform).unwrap();
            assert_eq!(profile.coverage.len(), LegacyKnowledgeAreaV1::ALL.len());
        }
        assert!(!pack
            .profile(LegacyPlatformV1::Zos)
            .unwrap()
            .unmapped_areas()
            .is_empty());
        assert!(pack
            .profile(LegacyPlatformV1::Solaris)
            .unwrap()
            .unmapped_areas()
            .is_empty());
    }

    #[test]
    fn source_claims_are_version_applicable_not_timeless() {
        let mut pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        let query = SupportKnowledgeQueryV1 {
            text: "AIX storage backup processes".into(),
            limit: 10,
            category: Some(SupportCategory::Software),
            technology: Some(TechnologyIdentityV1 {
                ecosystem: Some("legacy-enterprise-os".into()),
                vendor: Some("IBM".into()),
                product: "AIX".into(),
                edition: None,
                version: Some("7.3.4".into()),
                build: None,
                architecture: Some("POWER".into()),
                platform: None,
                profile: None,
                observed_features: BTreeSet::new(),
            }),
            purpose: KnowledgeQueryPurposeV1::LocalReasoning,
        };
        let hits = pack.sources.search_support_knowledge(&query).unwrap();
        assert!(hits.iter().any(|hit| hit.title.contains("AIX")));
    }

    #[test]
    fn wrong_platform_does_not_inherit_vendor_claim() {
        let mut pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        let query = SupportKnowledgeQueryV1 {
            text: "AIX storage backup processes".into(),
            limit: 10,
            category: Some(SupportCategory::Software),
            technology: Some(TechnologyIdentityV1 {
                ecosystem: Some("legacy-enterprise-os".into()),
                vendor: Some("Oracle".into()),
                product: "Oracle Solaris".into(),
                edition: None,
                version: Some("11.4".into()),
                build: None,
                architecture: Some("SPARC".into()),
                platform: None,
                profile: None,
                observed_features: BTreeSet::new(),
            }),
            purpose: KnowledgeQueryPurposeV1::LocalReasoning,
        };
        let hits = pack.sources.search_support_knowledge(&query).unwrap();
        assert!(!hits.iter().any(|hit| hit.title.contains("AIX operating system management")));
    }

    #[test]
    fn procedures_are_advisory_and_source_bound() {
        let pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        for procedure in &pack.procedures {
            procedure.validate(&pack.sources).unwrap();
            assert!(procedure
                .steps
                .iter()
                .all(|step| matches!(
                    step.authority,
                    LegacyProcedureAuthorityV1::ReadOnlyObservation
                        | LegacyProcedureAuthorityV1::ChangeProposalOnly
                )));
        }
    }
}
