// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Source-bound networking depth for AIX 7.3 and IBM i 7.6.
//!
//! This module fills two concrete legacy-networking gaps without assuming generic
//! Unix semantics. It models AIX TCP/IP configuration and interface-option
//! precedence separately from IBM i TCP/IP lifecycle and multi-protocol
//! communications. All procedures are advisory; no executable command or
//! authority token is introduced.

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

pub const LEGACY_NETWORKING_DEPTH_SCHEMA_V1: &str = "symthaea-it-legacy-aix-ibmi-networking-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum LegacyNetworkMechanismKindV1 {
    AixTcpIpControlPlane,
    AixNetworkOptionPrecedence,
    IbmiTcpIpLifecycle,
    IbmiCommunicationsProtocols,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyNetworkEvidenceSignalV1 {
    pub id: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyNetworkFailureModeV1 {
    pub id: String,
    pub symptom: String,
    pub discriminators: Vec<String>,
    pub unsafe_shortcuts: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyNetworkMechanismModelV1 {
    pub kind: LegacyNetworkMechanismKindV1,
    pub platform: LegacyPlatformV1,
    pub title: String,
    pub summary: String,
    pub source_claims: BTreeSet<TechnicalClaimIdV1>,
    pub evidence_signals: Vec<LegacyNetworkEvidenceSignalV1>,
    pub failure_modes: Vec<LegacyNetworkFailureModeV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyNetworkingDepthV1 {
    pub schema_version: String,
    pub mechanisms: Vec<LegacyNetworkMechanismModelV1>,
    pub procedure_ids: BTreeSet<String>,
}

impl LegacyNetworkingDepthV1 {
    pub fn validate(&self, pack: &LegacyComputingPackV1) -> Result<(), LegacyNetworkingDepthErrorV1> {
        if self.schema_version != LEGACY_NETWORKING_DEPTH_SCHEMA_V1 {
            return Err(LegacyNetworkingDepthErrorV1::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        if self.mechanisms.len() != 4 {
            return Err(LegacyNetworkingDepthErrorV1::InvalidField(
                "legacy networking V1 requires four mechanism models".into(),
            ));
        }
        let mut kinds = BTreeSet::new();
        for mechanism in &self.mechanisms {
            if !kinds.insert(mechanism.kind) {
                return Err(LegacyNetworkingDepthErrorV1::DuplicateMechanism(
                    mechanism.kind,
                ));
            }
            if !matches!(mechanism.platform, LegacyPlatformV1::Aix | LegacyPlatformV1::IbmI) {
                return Err(LegacyNetworkingDepthErrorV1::InvalidField(
                    "legacy networking V1 mechanism has unsupported platform".into(),
                ));
            }
            if mechanism.title.trim().is_empty()
                || mechanism.summary.trim().is_empty()
                || mechanism.source_claims.is_empty()
                || mechanism.evidence_signals.is_empty()
                || mechanism.failure_modes.is_empty()
            {
                return Err(LegacyNetworkingDepthErrorV1::InvalidField(format!(
                    "network mechanism {:?} is incomplete",
                    mechanism.kind
                )));
            }
            for claim in &mechanism.source_claims {
                if pack.sources.claim(claim).is_none() {
                    return Err(LegacyNetworkingDepthErrorV1::UnknownClaim(claim.clone()));
                }
            }
        }
        let procedure_ids: BTreeSet<_> = pack.procedures.iter().map(|p| p.id.as_str()).collect();
        for id in &self.procedure_ids {
            if !procedure_ids.contains(id.as_str()) {
                return Err(LegacyNetworkingDepthErrorV1::UnknownProcedure(id.clone()));
            }
        }
        Ok(())
    }
}

pub fn enrich_legacy_aix_ibmi_networking_v1(
    pack: &mut LegacyComputingPackV1,
    fetched_at_unix_ms: u64,
) -> Result<LegacyNetworkingDepthV1, LegacyNetworkingDepthErrorV1> {
    if fetched_at_unix_ms == 0 {
        return Err(LegacyNetworkingDepthErrorV1::InvalidField(
            "legacy networking source timestamp must be non-zero".into(),
        ));
    }
    pack.validate()?;
    let snapshots = register_sources(pack, fetched_at_unix_ms)?;
    register_claims(pack)?;
    let procedure_ids = register_procedures(pack, &snapshots)?;
    promote_profiles(pack, &snapshots)?;
    let foundation = LegacyNetworkingDepthV1 {
        schema_version: LEGACY_NETWORKING_DEPTH_SCHEMA_V1.into(),
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
) -> Result<BTreeMap<LegacyPlatformV1, BTreeSet<SourceSnapshotIdV1>>, LegacyNetworkingDepthErrorV1> {
    let definitions = [
        (
            LegacyPlatformV1::Aix,
            "ibm:aix-tcpip-configuration",
            "ibm:aix-tcpip-configuration@7.3",
            "AIX 7.3 Configuration of TCP/IP",
            "AIX 7.3 TCP/IP configuration",
            "https://www.ibm.com/docs/en/aix/7.3.0?topic=protocol-configuration-tcpip",
            "7.3",
        ),
        (
            LegacyPlatformV1::Aix,
            "ibm:aix-interface-network-options",
            "ibm:aix-interface-network-options@7.3",
            "AIX 7.3 Interface-specific network options",
            "AIX 7.3 interface-specific TCP/IP network options",
            "https://www.ibm.com/docs/en/aix/7.3.0?topic=interfaces-interface-specific-network-options",
            "7.3",
        ),
        (
            LegacyPlatformV1::IbmI,
            "ibm:ibmi-tcpip-setup",
            "ibm:ibmi-tcpip-setup@7.6",
            "IBM i 7.6 TCP/IP setup",
            "IBM i 7.6 TCP/IP setup and configuration",
            "https://www.ibm.com/docs/en/i/7.6.0?topic=networking-tcpip-setup",
            "7.6",
        ),
        (
            LegacyPlatformV1::IbmI,
            "ibm:ibmi-communications",
            "ibm:ibmi-communications@7.6",
            "IBM i 7.6 communications",
            "IBM i 7.6 communications and protocol configuration",
            "https://www.ibm.com/docs/en/i/7.6.0?topic=communications-getting-started-i",
            "7.6",
        ),
    ];

    let mut snapshots = BTreeMap::<LegacyPlatformV1, BTreeSet<SourceSnapshotIdV1>>::new();
    for (platform, doc_id, snapshot_id, title, canonical_ref, locator, version) in definitions {
        pack.sources.register_document(TechnicalSourceDocumentV1 {
            id: SourceDocumentIdV1(doc_id.into()),
            publisher: TechnicalPublisherV1::Vendor("IBM".into()),
            kind: SourceDocumentKindV1::VendorDocumentation,
            title: title.into(),
            canonical_ref: canonical_ref.into(),
            canonical_locator: Some(locator.into()),
        })?;
        pack.sources.register_snapshot(TechnicalSourceSnapshotV1 {
            id: SourceSnapshotIdV1(snapshot_id.into()),
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
        snapshots
            .entry(platform)
            .or_default()
            .insert(SourceSnapshotIdV1(snapshot_id.into()));
    }
    Ok(snapshots)
}

fn register_claims(pack: &mut LegacyComputingPackV1) -> Result<(), LegacyNetworkingDepthErrorV1> {
    let claims = [
        claim(
            LegacyPlatformV1::Aix,
            "legacy:aix:tcpip-control-plane",
            "ibm:aix-tcpip-configuration@7.3",
            "AIX TCP/IP configuration separates adapter/interface addressing, host identity, routing, name resolution, service daemons and gateway behavior; a healthy interface alone does not establish end-to-end network service.",
            "Configuration of TCP/IP",
        ),
        claim(
            LegacyPlatformV1::Aix,
            "legacy:aix:network-option-precedence",
            "ibm:aix-interface-network-options@7.3",
            "AIX interface-specific network options can override system-wide network options, while per-socket application settings can override interface-specific values; observed performance must therefore be localized to the effective option layer.",
            "Interface-specific network options",
        ),
        claim(
            LegacyPlatformV1::IbmI,
            "legacy:ibmi:tcpip-lifecycle",
            "ibm:ibmi-tcpip-setup@7.6",
            "IBM i TCP/IP operation separates stack activation, interfaces/routes, and TCP/IP application services; an active TCP/IP stack does not by itself prove that the required interface or server job is active.",
            "TCP/IP setup",
        ),
        claim(
            LegacyPlatformV1::IbmI,
            "legacy:ibmi:communications-protocols",
            "ibm:ibmi-communications@7.6",
            "IBM i communications supports TCP/IP alongside APPC, APPN, HPR and other communications models whose configuration objects and session state are distinct; healthy TCP/IP does not establish non-TCP/IP application connectivity.",
            "Getting started with IBM i communications",
        ),
    ];
    for claim in claims {
        pack.sources.register_claim(claim)?;
    }
    Ok(())
}

fn claim(
    platform: LegacyPlatformV1,
    id: &str,
    snapshot: &str,
    statement: &str,
    section: &str,
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
        applicability: Some(legacy_platform_scope_v1(platform)),
        extraction_quality: Some(0.95),
        category: Some(SupportCategory::Network),
    }
}

fn register_procedures(
    pack: &mut LegacyComputingPackV1,
    snapshots: &BTreeMap<LegacyPlatformV1, BTreeSet<SourceSnapshotIdV1>>,
) -> Result<BTreeSet<String>, LegacyNetworkingDepthErrorV1> {
    let definitions = [
        (
            LegacyPlatformV1::Aix,
            "legacy:aix:network-layer-triage",
            "AIX TCP/IP layer triage",
            "Compare adapter/interface/address state, route selection, name resolution, daemon/service state and relevant network-option precedence before proposing configuration changes.",
        ),
        (
            LegacyPlatformV1::IbmI,
            "legacy:ibmi:network-layer-triage",
            "IBM i communications layer triage",
            "Separate TCP/IP stack, interface/route, server-job and protocol-specific communications state before proposing stack, interface or service changes.",
        ),
    ];
    let mut ids = BTreeSet::new();
    for (platform, id, title, diagnostic) in definitions {
        let procedure = LegacyProcedureV1 {
            id: id.into(),
            platform,
            area: LegacyKnowledgeAreaV1::Networking,
            kind: LegacyProcedureKindV1::Diagnose,
            title: title.into(),
            applicability: legacy_platform_scope_v1(platform),
            preconditions: vec![
                "Establish exact platform/version and affected network/service identity before diagnosis.".into(),
                "Preserve operator authority boundaries; this procedure does not authorize network mutation.".into(),
            ],
            steps: vec![
                LegacyProcedureStepV1 {
                    authority: LegacyProcedureAuthorityV1::ReadOnlyObservation,
                    description: "Establish failure window, interface/link state, addressing, routes, name-resolution path and recent network changes.".into(),
                },
                LegacyProcedureStepV1 {
                    authority: LegacyProcedureAuthorityV1::ReadOnlyObservation,
                    description: diagnostic.into(),
                },
                LegacyProcedureStepV1 {
                    authority: LegacyProcedureAuthorityV1::ReadOnlyObservation,
                    description: "Use a healthy control path only as evidence for that path/protocol; do not generalize it to unrelated services or protocol families.".into(),
                },
                LegacyProcedureStepV1 {
                    authority: LegacyProcedureAuthorityV1::ChangeProposalOnly,
                    description: "If a repair is supported, propose the smallest reversible network change with blast-radius, rollback and post-change verification.".into(),
                },
            ],
            verification: vec![
                "Re-test the original affected path and protocol against the same endpoint identity.".into(),
                "Verify unrelated interfaces, routes, services and protocol families did not regress.".into(),
            ],
            source_snapshots: snapshots.get(&platform).cloned().unwrap_or_default(),
        };
        insert_procedure(pack, procedure)?;
        ids.insert(id.into());
    }
    Ok(ids)
}

fn insert_procedure(
    pack: &mut LegacyComputingPackV1,
    procedure: LegacyProcedureV1,
) -> Result<(), LegacyNetworkingDepthErrorV1> {
    if let Some(existing) = pack.procedures.iter().find(|existing| existing.id == procedure.id) {
        if existing != &procedure {
            return Err(LegacyNetworkingDepthErrorV1::ProcedureConflict(procedure.id));
        }
        return Ok(());
    }
    procedure.validate(&pack.sources)?;
    pack.procedures.push(procedure);
    Ok(())
}

fn promote_profiles(
    pack: &mut LegacyComputingPackV1,
    snapshots: &BTreeMap<LegacyPlatformV1, BTreeSet<SourceSnapshotIdV1>>,
) -> Result<(), LegacyNetworkingDepthErrorV1> {
    for platform in [LegacyPlatformV1::Aix, LegacyPlatformV1::IbmI] {
        let profile = pack
            .profiles
            .iter_mut()
            .find(|profile| profile.platform == platform)
            .ok_or(LegacyNetworkingDepthErrorV1::MissingPlatform(platform))?;
        if let Some(platform_snapshots) = snapshots.get(&platform) {
            profile.source_snapshots.extend(platform_snapshots.iter().cloned());
        }
        if profile.state(LegacyKnowledgeAreaV1::Networking) < LegacyCoverageStateV1::ClaimSeeded {
            profile
                .coverage
                .insert(LegacyKnowledgeAreaV1::Networking, LegacyCoverageStateV1::ClaimSeeded);
        }
        match platform {
            LegacyPlatformV1::Aix => {
                profile.gap_tags.remove("aix-networking");
                profile
                    .gap_tags
                    .insert("aix-routing-dns-daemon-performance-network-labs".into());
            }
            LegacyPlatformV1::IbmI => {
                profile
                    .gap_tags
                    .insert("ibmi-routing-dns-server-job-network-labs".into());
                profile
                    .gap_tags
                    .insert("ibmi-appn-hpr-interoperability-depth".into());
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}

fn mechanisms() -> Vec<LegacyNetworkMechanismModelV1> {
    use LegacyNetworkMechanismKindV1 as K;
    vec![
        model(
            K::AixTcpIpControlPlane,
            LegacyPlatformV1::Aix,
            "AIX TCP/IP control-plane layers",
            "Interface/address, routing, name resolution, daemon/service and gateway state are distinct diagnostic layers.",
            &["legacy:aix:tcpip-control-plane"],
            &[
                ("interface-state", "Current adapter/interface/address state"),
                ("route-state", "Route table, selected gateway and remote-network path"),
                ("name-resolution", "Local resolver and name-server path"),
                ("daemon-service", "inetd/server daemon and application service state"),
            ],
            &[
                ("interface-up-route-broken", "Interface is up but the selected remote route/gateway is wrong or missing.", &["local interface works", "remote route/path differs"], &["cycle adapter before checking route"]),
                ("network-up-name-broken", "IP reachability succeeds while name resolution fails.", &["address control succeeds", "resolver path fails"], &["restart all networking for DNS-only failure"]),
            ],
        ),
        model(
            K::AixNetworkOptionPrecedence,
            LegacyPlatformV1::Aix,
            "AIX network-option precedence",
            "Application socket options, interface-specific network options and system-wide values have explicit precedence and must not be conflated.",
            &["legacy:aix:network-option-precedence"],
            &[
                ("global-options", "System-wide network option values"),
                ("interface-options", "Interface-specific overrides and use_isno state"),
                ("socket-options", "Application-level socket overrides"),
            ],
            &[
                ("global-value-not-effective", "A system-wide value appears correct but a narrower interface/socket override controls the affected traffic.", &["effective behavior differs by interface/app", "override exists"], &["change global tuning before finding effective layer"]),
            ],
        ),
        model(
            K::IbmiTcpIpLifecycle,
            LegacyPlatformV1::IbmI,
            "IBM i TCP/IP lifecycle layers",
            "TCP/IP stack activation, interface activation/routes and application server jobs are separate operational states.",
            &["legacy:ibmi:tcpip-lifecycle"],
            &[
                ("stack-state", "TCP/IP stack and IPv4/IPv6 activation state"),
                ("interface-route", "Interface AUTOSTART/current state and route configuration"),
                ("server-job", "Target TCP/IP application server job/autostart state"),
            ],
            &[
                ("stack-up-interface-down", "TCP/IP is active while the required interface is inactive.", &["stack active", "interface inactive"], &["restart the entire stack first"]),
                ("interface-up-server-down", "Network path is healthy while the target TCP/IP server job is not running.", &["interface/path healthy", "server job absent"], &["change routes before checking server job"]),
            ],
        ),
        model(
            K::IbmiCommunicationsProtocols,
            LegacyPlatformV1::IbmI,
            "IBM i multi-protocol communications",
            "TCP/IP and APPC/APPN/HPR use distinct communication configuration/session state; success in one protocol family is not proof for another.",
            &["legacy:ibmi:communications-protocols"],
            &[
                ("protocol-dependency", "Actual application protocol family and configuration objects"),
                ("session-state", "Protocol-specific session/path state"),
                ("tcpip-control", "Independent TCP/IP control-path result"),
            ],
            &[
                ("tcpip-good-appn-bad", "TCP/IP tests pass while an APPN/HPR-dependent application path fails.", &["application dependency is non-TCP/IP", "protocol-specific session fails"], &["declare network healthy because ping works"]),
            ],
        ),
    ]
}

fn model(
    kind: LegacyNetworkMechanismKindV1,
    platform: LegacyPlatformV1,
    title: &str,
    summary: &str,
    claims: &[&str],
    evidence: &[(&str, &str)],
    failures: &[(&str, &str, &[&str], &[&str])],
) -> LegacyNetworkMechanismModelV1 {
    LegacyNetworkMechanismModelV1 {
        kind,
        platform,
        title: title.into(),
        summary: summary.into(),
        source_claims: claims
            .iter()
            .map(|id| TechnicalClaimIdV1((*id).into()))
            .collect(),
        evidence_signals: evidence
            .iter()
            .map(|(id, description)| LegacyNetworkEvidenceSignalV1 {
                id: (*id).into(),
                description: (*description).into(),
            })
            .collect(),
        failure_modes: failures
            .iter()
            .map(|(id, symptom, discriminators, unsafe_shortcuts)| LegacyNetworkFailureModeV1 {
                id: (*id).into(),
                symptom: (*symptom).into(),
                discriminators: discriminators.iter().map(|value| (*value).into()).collect(),
                unsafe_shortcuts: unsafe_shortcuts.iter().map(|value| (*value).into()).collect(),
            })
            .collect(),
    }
}

#[derive(Debug)]
pub enum LegacyNetworkingDepthErrorV1 {
    Standards(StandardsRegistryErrorV1),
    Computing(LegacyComputingErrorV1),
    UnsupportedSchema(String),
    InvalidField(String),
    DuplicateMechanism(LegacyNetworkMechanismKindV1),
    UnknownClaim(TechnicalClaimIdV1),
    UnknownProcedure(String),
    ProcedureConflict(String),
    MissingPlatform(LegacyPlatformV1),
}

impl fmt::Display for LegacyNetworkingDepthErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Standards(err) => write!(f, "legacy networking source error: {err}"),
            Self::Computing(err) => write!(f, "legacy networking pack error: {err}"),
            Self::UnsupportedSchema(schema) => write!(f, "unsupported legacy networking schema {schema}"),
            Self::InvalidField(message) => write!(f, "invalid legacy networking foundation: {message}"),
            Self::DuplicateMechanism(kind) => write!(f, "duplicate legacy networking mechanism {kind:?}"),
            Self::UnknownClaim(id) => write!(f, "unknown legacy networking claim {}", id.0),
            Self::UnknownProcedure(id) => write!(f, "unknown legacy networking procedure {id}"),
            Self::ProcedureConflict(id) => write!(f, "conflicting legacy networking procedure {id}"),
            Self::MissingPlatform(platform) => write!(f, "legacy networking pack missing platform {platform:?}"),
        }
    }
}

impl Error for LegacyNetworkingDepthErrorV1 {}

impl From<StandardsRegistryErrorV1> for LegacyNetworkingDepthErrorV1 {
    fn from(value: StandardsRegistryErrorV1) -> Self {
        Self::Standards(value)
    }
}

impl From<LegacyComputingErrorV1> for LegacyNetworkingDepthErrorV1 {
    fn from(value: LegacyComputingErrorV1) -> Self {
        Self::Computing(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::technology::ApplicabilityStatusV1;
    use crate::{build_legacy_five_platform_portfolio_v1, legacy_platform_identity_v1};

    fn pack() -> LegacyComputingPackV1 {
        build_legacy_five_platform_portfolio_v1(1_800_000_000_000)
            .unwrap()
            .0
    }

    #[test]
    fn networking_enrichment_is_idempotent_and_promotes_only_target_platforms() {
        let mut pack = pack();
        let first = enrich_legacy_aix_ibmi_networking_v1(&mut pack, 1_800_000_000_100).unwrap();
        let second = enrich_legacy_aix_ibmi_networking_v1(&mut pack, 1_800_000_000_100).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.mechanisms.len(), 4);
        for platform in [LegacyPlatformV1::Aix, LegacyPlatformV1::IbmI] {
            assert!(pack.profile(platform).unwrap().state(LegacyKnowledgeAreaV1::Networking)
                >= LegacyCoverageStateV1::ClaimSeeded);
        }
        assert!(!pack.profile(LegacyPlatformV1::Aix).unwrap().gap_tags.contains("aix-networking"));
        assert!(pack.profile(LegacyPlatformV1::HpUx).unwrap().gap_tags.contains("hpux-networking"));
    }

    #[test]
    fn claims_are_product_isolated_despite_shared_ibm_vendor() {
        let mut pack = pack();
        enrich_legacy_aix_ibmi_networking_v1(&mut pack, 1_800_000_000_100).unwrap();
        let aix = legacy_platform_identity_v1(LegacyPlatformV1::Aix);
        let ibmi = legacy_platform_identity_v1(LegacyPlatformV1::IbmI);
        let aix_claim = pack
            .sources
            .claim(&TechnicalClaimIdV1("legacy:aix:tcpip-control-plane".into()))
            .unwrap();
        let ibmi_claim = pack
            .sources
            .claim(&TechnicalClaimIdV1("legacy:ibmi:tcpip-lifecycle".into()))
            .unwrap();
        assert_eq!(
            aix_claim.applicability.as_ref().unwrap().assess(&aix).unwrap().status,
            ApplicabilityStatusV1::Applicable
        );
        assert_eq!(
            aix_claim.applicability.as_ref().unwrap().assess(&ibmi).unwrap().status,
            ApplicabilityStatusV1::Inapplicable
        );
        assert_eq!(
            ibmi_claim.applicability.as_ref().unwrap().assess(&aix).unwrap().status,
            ApplicabilityStatusV1::Inapplicable
        );
    }

    #[test]
    fn procedures_never_cross_into_execution_authority() {
        let mut pack = pack();
        let depth = enrich_legacy_aix_ibmi_networking_v1(&mut pack, 1_800_000_000_100).unwrap();
        for id in depth.procedure_ids {
            let procedure = pack.procedures.iter().find(|procedure| procedure.id == id).unwrap();
            assert!(procedure.steps.iter().all(|step| matches!(
                step.authority,
                LegacyProcedureAuthorityV1::ReadOnlyObservation
                    | LegacyProcedureAuthorityV1::ChangeProposalOnly
            )));
        }
    }

    #[test]
    fn aix_option_precedence_and_ibmi_service_lifecycle_remain_distinct() {
        let mut pack = pack();
        let depth = enrich_legacy_aix_ibmi_networking_v1(&mut pack, 1_800_000_000_100).unwrap();
        assert!(depth.mechanisms.iter().any(|mechanism| {
            mechanism.kind == LegacyNetworkMechanismKindV1::AixNetworkOptionPrecedence
                && mechanism.failure_modes.iter().any(|failure| failure.id == "global-value-not-effective")
        }));
        assert!(depth.mechanisms.iter().any(|mechanism| {
            mechanism.kind == LegacyNetworkMechanismKindV1::IbmiTcpIpLifecycle
                && mechanism.failure_modes.iter().any(|failure| failure.id == "interface-up-server-down")
        }));
    }
}
