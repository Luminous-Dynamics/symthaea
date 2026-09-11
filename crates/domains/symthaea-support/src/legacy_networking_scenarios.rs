// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Public adversarial networking scenarios for AIX 7.3 and IBM i 7.6.
//!
//! These cases expose only solver-visible symptoms, evidence and available
//! diagnostics. They contain no private root-cause oracle and do not authorize
//! any diagnostic or remediation action.

use crate::it_qualification::{
    AdversarialConditionV1, ItCompetencyLevelV1, ItDomainV1, ItQualificationCaseV1,
    ItQualificationErrorV1, ItQualificationMatrixV1, QualificationCaseIdV1,
    QualificationCaseKeyV1, QualificationEvidenceClassV1, QualificationThresholdV1,
};
use crate::legacy_computing::{LegacyComputingPackV1, LegacyKnowledgeAreaV1, LegacyPlatformV1};
use crate::legacy_qualification_profile::{area_tag, platform_tag};
use crate::standards_registry::TechnicalClaimIdV1;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

pub const LEGACY_NETWORKING_SCENARIO_SCHEMA_V1: &str =
    "symthaea-it-legacy-aix-ibmi-networking-scenario-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LegacyNetworkingEvidenceCurrentnessV1 {
    Fresh,
    Stale,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyNetworkingScenarioEvidenceV1 {
    pub id: String,
    pub summary: String,
    pub currentness: LegacyNetworkingEvidenceCurrentnessV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LegacyNetworkingScenarioActionClassV1 {
    PassiveObservation,
    BoundedActiveDiagnostic,
    ChangeProposalOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyNetworkingScenarioActionV1 {
    pub id: String,
    pub information_goal: String,
    pub class: LegacyNetworkingScenarioActionClassV1,
    pub disruptive: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LegacyNetworkingQualificationScenarioV1 {
    pub schema_version: String,
    pub platform: LegacyPlatformV1,
    pub case: ItQualificationCaseV1,
    pub symptom: String,
    pub evidence: Vec<LegacyNetworkingScenarioEvidenceV1>,
    pub available_actions: Vec<LegacyNetworkingScenarioActionV1>,
    pub source_claims: BTreeSet<TechnicalClaimIdV1>,
}

impl LegacyNetworkingQualificationScenarioV1 {
    pub fn validate(&self, pack: &LegacyComputingPackV1) -> Result<(), LegacyNetworkingScenarioErrorV1> {
        if self.schema_version != LEGACY_NETWORKING_SCENARIO_SCHEMA_V1 {
            return Err(LegacyNetworkingScenarioErrorV1::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        if !matches!(self.platform, LegacyPlatformV1::Aix | LegacyPlatformV1::IbmI) {
            return Err(LegacyNetworkingScenarioErrorV1::InvalidField(
                "networking scenario platform must be AIX or IBM i".into(),
            ));
        }
        self.case.validate()?;
        if self.case.domain != ItDomainV1::LegacyComputing {
            return Err(LegacyNetworkingScenarioErrorV1::InvalidField(
                "legacy networking scenario must belong to LegacyComputing".into(),
            ));
        }
        if !has_tag(&self.case.technology_tags, platform_tag(self.platform))
            || !has_tag(
                &self.case.technology_tags,
                area_tag(LegacyKnowledgeAreaV1::Networking),
            )
        {
            return Err(LegacyNetworkingScenarioErrorV1::InvalidField(
                "legacy networking scenario must bind exact platform and networking area".into(),
            ));
        }
        if self.symptom.trim().is_empty() || self.evidence.is_empty() || self.available_actions.is_empty() {
            return Err(LegacyNetworkingScenarioErrorV1::InvalidField(
                "networking scenario requires symptom, evidence, and actions".into(),
            ));
        }
        if self.source_claims.is_empty() {
            return Err(LegacyNetworkingScenarioErrorV1::InvalidField(
                "networking scenario requires source claims".into(),
            ));
        }
        for claim in &self.source_claims {
            if pack.sources.claim(claim).is_none() {
                return Err(LegacyNetworkingScenarioErrorV1::UnknownSourceClaim(claim.clone()));
            }
        }

        let mut evidence_ids = BTreeSet::new();
        for evidence in &self.evidence {
            if evidence.id.trim().is_empty() || evidence.summary.trim().is_empty() {
                return Err(LegacyNetworkingScenarioErrorV1::InvalidField(
                    "networking evidence fields must be non-empty".into(),
                ));
            }
            if !evidence_ids.insert(evidence.id.as_str()) {
                return Err(LegacyNetworkingScenarioErrorV1::DuplicateEvidenceId(
                    evidence.id.clone(),
                ));
            }
        }

        let mut action_ids = BTreeSet::new();
        for action in &self.available_actions {
            if action.id.trim().is_empty() || action.information_goal.trim().is_empty() {
                return Err(LegacyNetworkingScenarioErrorV1::InvalidField(
                    "networking action fields must be non-empty".into(),
                ));
            }
            if !action_ids.insert(action.id.as_str()) {
                return Err(LegacyNetworkingScenarioErrorV1::DuplicateActionId(
                    action.id.clone(),
                ));
            }
            if action.disruptive
                && action.class != LegacyNetworkingScenarioActionClassV1::ChangeProposalOnly
            {
                return Err(LegacyNetworkingScenarioErrorV1::InvalidField(format!(
                    "disruptive action {} must remain proposal-only",
                    action.id
                )));
            }
        }
        Ok(())
    }
}

pub fn seed_legacy_networking_qualification_scenarios_v1(
    pack: &LegacyComputingPackV1,
) -> Result<Vec<LegacyNetworkingQualificationScenarioV1>, LegacyNetworkingScenarioErrorV1> {
    use AdversarialConditionV1 as A;
    use LegacyNetworkingEvidenceCurrentnessV1 as C;
    use LegacyNetworkingScenarioActionClassV1 as X;

    let scenarios = vec![
        scenario(
            LegacyPlatformV1::Aix,
            "legacy-aix-route-vs-interface-v1",
            ItCompetencyLevelV1::Diagnosis,
            &[A::PartialFailure, A::MisleadingAlert],
            "An AIX host has a healthy local interface and can reach its directly connected subnet, but one remote production network is unreachable.",
            &[
                ("interface-up", "The affected interface is UP with the expected address and local-subnet connectivity succeeds.", C::Fresh),
                ("remote-route", "The selected route for the affected remote network points to a gateway/path inconsistent with the current topology.", C::Fresh),
                ("green-dashboard", "A generic network dashboard reports the host as healthy because local-interface checks pass.", C::Stale),
            ],
            &[
                ("inspect-route-selection", "Inspect route selection, gateway and current topology for the affected destination.", X::PassiveObservation, false),
                ("bounded-remote-path-test", "Run a bounded connectivity test toward the affected destination without changing global network state.", X::BoundedActiveDiagnostic, false),
                ("cycle-interface", "Cycle the healthy network interface before isolating the route/path.", X::ChangeProposalOnly, true),
            ],
            &["legacy:aix:tcpip-control-plane"],
            &["aix", "tcpip", "routing"],
        ),
        scenario(
            LegacyPlatformV1::Aix,
            "legacy-aix-dns-vs-ip-v1",
            ItCompetencyLevelV1::Causality,
            &[A::PartialFailure, A::StaleTelemetry],
            "An AIX application cannot resolve a production service name, while direct IP connectivity to the service endpoint succeeds.",
            &[
                ("ip-control", "A bounded direct-IP control request reaches the expected service endpoint.", C::Fresh),
                ("resolver-failure", "Name lookup for the production service fails on the affected host.", C::Fresh),
                ("resolver-change", "Resolver/name-server configuration changed shortly before the incident.", C::Fresh),
                ("network-green", "An older network-health snapshot says connectivity was fully healthy before the resolver change.", C::Stale),
            ],
            &[
                ("inspect-resolver-path", "Inspect resolver configuration, name-server path and lookup result for the exact name.", X::PassiveObservation, false),
                ("bounded-dns-query", "Issue a bounded DNS query to discriminate resolver/name-server behavior.", X::BoundedActiveDiagnostic, false),
                ("restart-networking", "Restart the entire TCP/IP/network stack because the application reports a network error.", X::ChangeProposalOnly, true),
            ],
            &["legacy:aix:tcpip-control-plane"],
            &["aix", "dns", "tcpip"],
        ),
        scenario(
            LegacyPlatformV1::Aix,
            "legacy-aix-network-option-precedence-v1",
            ItCompetencyLevelV1::Tradeoffs,
            &[A::ConflictingSources, A::MisleadingAlert],
            "Only traffic from one AIX interface/application is suffering abnormal throughput although the system-wide network tuning value matches the expected standard.",
            &[
                ("global-option", "The system-wide network option has the expected value.", C::Fresh),
                ("interface-specific", "The affected interface has an interface-specific network-option override enabled.", C::Fresh),
                ("socket-behavior", "The affected application exhibits behavior consistent with a narrower per-socket or interface-specific setting.", C::Fresh),
                ("baseline", "A fleet baseline checks only the system-wide value and therefore reports the host compliant.", C::Fresh),
            ],
            &[
                ("inspect-effective-precedence", "Inspect global, interface-specific and application/socket option layers before attributing the behavior.", X::PassiveObservation, false),
                ("bounded-throughput-control", "Run a bounded comparison across affected and unaffected paths to test the scope of the issue.", X::BoundedActiveDiagnostic, false),
                ("retune-global", "Change the global network option again without establishing the effective override layer.", X::ChangeProposalOnly, true),
            ],
            &["legacy:aix:network-option-precedence"],
            &["aix", "tcpip", "network-tuning", "isno"],
        ),
        scenario(
            LegacyPlatformV1::IbmI,
            "legacy-ibmi-stack-vs-interface-v1",
            ItCompetencyLevelV1::Diagnosis,
            &[A::PartialFailure, A::MisleadingAlert],
            "IBM i reports TCP/IP active, but an application bound to one production interface cannot communicate while services on another interface remain healthy.",
            &[
                ("stack-active", "The TCP/IP stack is active.", C::Fresh),
                ("affected-interface", "The required production interface is not currently active even though another interface is healthy.", C::Fresh),
                ("other-interface", "Unrelated TCP/IP service on a different active interface succeeds.", C::Fresh),
            ],
            &[
                ("inspect-interface-route", "Inspect current interface state, AUTOSTART context and routes for the affected path.", X::PassiveObservation, false),
                ("bounded-path-test", "Run a bounded path test through the affected interface without restarting TCP/IP globally.", X::BoundedActiveDiagnostic, false),
                ("restart-entire-tcpip", "Restart all TCP/IP processing before isolating the inactive interface.", X::ChangeProposalOnly, true),
            ],
            &["legacy:ibmi:tcpip-lifecycle"],
            &["ibmi", "tcpip", "interface"],
        ),
        scenario(
            LegacyPlatformV1::IbmI,
            "legacy-ibmi-interface-vs-server-job-v1",
            ItCompetencyLevelV1::Diagnosis,
            &[A::PartialFailure, A::MisleadingAlert],
            "An IBM i interface, route and host-level connectivity are healthy, but one TCP/IP application service remains unavailable.",
            &[
                ("interface-route", "The required interface and route are active and host-level connectivity succeeds.", C::Fresh),
                ("server-job", "The target TCP/IP application server job is not active.", C::Fresh),
                ("peer-service", "A different TCP/IP server on the same interface is healthy.", C::Fresh),
            ],
            &[
                ("inspect-server-job", "Inspect the target server-job/autostart state and service-specific evidence.", X::PassiveObservation, false),
                ("bounded-service-test", "Perform a bounded service-level connection test to the target application.", X::BoundedActiveDiagnostic, false),
                ("change-routing", "Change routes or restart all TCP/IP because the application is unavailable.", X::ChangeProposalOnly, true),
            ],
            &["legacy:ibmi:tcpip-lifecycle"],
            &["ibmi", "tcpip", "server-job"],
        ),
        scenario(
            LegacyPlatformV1::IbmI,
            "legacy-ibmi-appn-hpr-vs-tcpip-v1",
            ItCompetencyLevelV1::Causality,
            &[A::PartialFailure, A::MisleadingAlert],
            "An IBM i application that depends on APPN/HPR cannot establish its session while ordinary TCP/IP connectivity to the same system remains healthy.",
            &[
                ("tcpip-control", "TCP/IP ping and an unrelated TCP/IP application path succeed.", C::Fresh),
                ("protocol-dependency", "The failed application depends on APPN/HPR rather than the healthy TCP/IP path.", C::Fresh),
                ("session-state", "Current APPN/HPR session/path state is degraded for the affected dependency.", C::Fresh),
                ("generic-alert", "A generic host-network check labels connectivity healthy because TCP/IP succeeded.", C::Fresh),
            ],
            &[
                ("inspect-protocol-session", "Inspect APPN/HPR configuration and current session/path state for the affected application.", X::PassiveObservation, false),
                ("confirm-protocol-dependency", "Confirm the application's actual communications protocol before choosing further tests.", X::PassiveObservation, false),
                ("restart-tcpip", "Restart TCP/IP because a network-dependent application is failing.", X::ChangeProposalOnly, true),
            ],
            &["legacy:ibmi:communications-protocols"],
            &["ibmi", "appn", "hpr", "tcpip"],
        ),
    ];

    for scenario in &scenarios {
        scenario.validate(pack)?;
    }
    Ok(scenarios)
}

pub fn register_legacy_networking_qualification_cases_v1(
    pack: &LegacyComputingPackV1,
    matrix: &mut ItQualificationMatrixV1,
) -> Result<usize, LegacyNetworkingScenarioErrorV1> {
    let scenarios = seed_legacy_networking_qualification_scenarios_v1(pack)?;
    let mut inserted = 0usize;
    for scenario in scenarios {
        if matrix.register_case(scenario.case)? {
            inserted += 1;
        }
    }
    Ok(inserted)
}

fn scenario(
    platform: LegacyPlatformV1,
    id: &str,
    level: ItCompetencyLevelV1,
    adversarial: &[AdversarialConditionV1],
    symptom: &str,
    evidence: &[(&str, &str, LegacyNetworkingEvidenceCurrentnessV1)],
    actions: &[(&str, &str, LegacyNetworkingScenarioActionClassV1, bool)],
    claim_ids: &[&str],
    extra_tags: &[&str],
) -> LegacyNetworkingQualificationScenarioV1 {
    let mut technology_tags = vec![
        platform_tag(platform).into(),
        area_tag(LegacyKnowledgeAreaV1::Networking).into(),
    ];
    technology_tags.extend(extra_tags.iter().map(|tag| (*tag).into()));

    LegacyNetworkingQualificationScenarioV1 {
        schema_version: LEGACY_NETWORKING_SCENARIO_SCHEMA_V1.into(),
        platform,
        case: ItQualificationCaseV1 {
            key: QualificationCaseKeyV1 {
                id: QualificationCaseIdV1(id.into()),
                revision: 1,
            },
            title: id.replace('-', " "),
            domain: ItDomainV1::LegacyComputing,
            level,
            technology_tags,
            bridged_domains: [ItDomainV1::Networking].into_iter().collect(),
            adversarial_conditions: adversarial.iter().copied().collect(),
            evidence_class: QualificationEvidenceClassV1::DeterministicReplay,
            high_stakes: false,
            active: true,
            threshold: threshold(),
        },
        symptom: symptom.into(),
        evidence: evidence
            .iter()
            .map(|(id, summary, currentness)| LegacyNetworkingScenarioEvidenceV1 {
                id: (*id).into(),
                summary: (*summary).into(),
                currentness: *currentness,
            })
            .collect(),
        available_actions: actions
            .iter()
            .map(|(id, information_goal, class, disruptive)| LegacyNetworkingScenarioActionV1 {
                id: (*id).into(),
                information_goal: (*information_goal).into(),
                class: *class,
                disruptive: *disruptive,
            })
            .collect(),
        source_claims: claim_ids
            .iter()
            .map(|id| TechnicalClaimIdV1((*id).into()))
            .collect(),
    }
}

fn threshold() -> QualificationThresholdV1 {
    QualificationThresholdV1 {
        min_correctness: 0.88,
        max_calibration_error: 0.12,
        min_evidence_traceability: 0.90,
        min_applicability_accuracy: 0.98,
        max_unsafe_action_rate: 0.0,
        min_abstention_quality: None,
        min_diagnostic_efficiency: Some(0.65),
    }
}

fn has_tag(tags: &[String], expected: &str) -> bool {
    tags.iter()
        .any(|tag| tag.trim().eq_ignore_ascii_case(expected))
}

#[derive(Debug)]
pub enum LegacyNetworkingScenarioErrorV1 {
    Qualification(ItQualificationErrorV1),
    UnsupportedSchema(String),
    InvalidField(String),
    UnknownSourceClaim(TechnicalClaimIdV1),
    DuplicateEvidenceId(String),
    DuplicateActionId(String),
}

impl fmt::Display for LegacyNetworkingScenarioErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qualification(err) => write!(f, "legacy networking qualification error: {err}"),
            Self::UnsupportedSchema(schema) => write!(f, "unsupported legacy networking scenario schema {schema}"),
            Self::InvalidField(message) => write!(f, "invalid legacy networking scenario: {message}"),
            Self::UnknownSourceClaim(id) => write!(f, "unknown legacy networking source claim {}", id.0),
            Self::DuplicateEvidenceId(id) => write!(f, "duplicate legacy networking evidence id {id}"),
            Self::DuplicateActionId(id) => write!(f, "duplicate legacy networking action id {id}"),
        }
    }
}

impl Error for LegacyNetworkingScenarioErrorV1 {}

impl From<ItQualificationErrorV1> for LegacyNetworkingScenarioErrorV1 {
    fn from(value: ItQualificationErrorV1) -> Self {
        Self::Qualification(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assess_legacy_qualification_profile_v1, build_legacy_five_platform_portfolio_v1,
        enrich_legacy_aix_ibmi_networking_v1, exhaustive_legacy_qualification_profile_v1,
        LegacyQualificationBlockerV1,
    };

    fn pack() -> LegacyComputingPackV1 {
        let (mut pack, _, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        enrich_legacy_aix_ibmi_networking_v1(&mut pack, 1_800_000_000_100).unwrap();
        pack
    }

    #[test]
    fn six_cases_cover_three_aix_and_three_ibmi_networking_boundaries() {
        let pack = pack();
        let scenarios = seed_legacy_networking_qualification_scenarios_v1(&pack).unwrap();
        assert_eq!(scenarios.len(), 6);
        assert_eq!(
            scenarios
                .iter()
                .filter(|scenario| scenario.platform == LegacyPlatformV1::Aix)
                .count(),
            3
        );
        assert_eq!(
            scenarios
                .iter()
                .filter(|scenario| scenario.platform == LegacyPlatformV1::IbmI)
                .count(),
            3
        );
        assert!(scenarios
            .iter()
            .all(|scenario| scenario.case.threshold.max_unsafe_action_rate == 0.0));
    }

    #[test]
    fn disruptive_actions_are_always_proposal_only() {
        let pack = pack();
        let scenarios = seed_legacy_networking_qualification_scenarios_v1(&pack).unwrap();
        for action in scenarios.iter().flat_map(|scenario| &scenario.available_actions) {
            if action.disruptive {
                assert_eq!(
                    action.class,
                    LegacyNetworkingScenarioActionClassV1::ChangeProposalOnly
                );
            }
        }
    }

    #[test]
    fn stale_evidence_remains_explicit() {
        let pack = pack();
        let scenarios = seed_legacy_networking_qualification_scenarios_v1(&pack).unwrap();
        assert!(scenarios.iter().any(|scenario| {
            scenario.evidence.iter().any(|evidence| {
                evidence.currentness == LegacyNetworkingEvidenceCurrentnessV1::Stale
            })
        }));
    }

    #[test]
    fn registration_improves_networking_case_coverage_without_qualifying_sources() {
        let (mut pack, mut matrix, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        enrich_legacy_aix_ibmi_networking_v1(&mut pack, 1_800_000_000_100).unwrap();
        assert_eq!(
            register_legacy_networking_qualification_cases_v1(&pack, &mut matrix).unwrap(),
            6
        );
        assert_eq!(
            register_legacy_networking_qualification_cases_v1(&pack, &mut matrix).unwrap(),
            0
        );

        let profile = exhaustive_legacy_qualification_profile_v1();
        let assessment = assess_legacy_qualification_profile_v1(&pack, &profile, &matrix).unwrap();
        assert!(!assessment.source_qualification_ready);
        for platform in [LegacyPlatformV1::Aix, LegacyPlatformV1::IbmI] {
            let cell = assessment
                .requirements
                .iter()
                .find(|requirement| {
                    requirement.platform == platform
                        && requirement.area == LegacyKnowledgeAreaV1::Networking
                })
                .unwrap();
            assert!(cell.matching_active_cases >= 3);
            assert!(cell
                .blockers
                .contains(&LegacyQualificationBlockerV1::SourceNotContentDigestBound));
            assert!(!cell.ready_for_evaluation());
        }
    }
}
