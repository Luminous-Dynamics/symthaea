// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Public adversarial Oracle Solaris 11.4 scenario specifications.
//!
//! These fixtures are solver-visible benchmark specifications only. They contain
//! no private root-cause oracle, accepted remediation set, or precomputed result.

use crate::it_qualification::{
    AdversarialConditionV1, ItCompetencyLevelV1, ItDomainV1, ItQualificationCaseV1,
    ItQualificationErrorV1, ItQualificationMatrixV1, QualificationCaseIdV1,
    QualificationCaseKeyV1, QualificationEvidenceClassV1, QualificationThresholdV1,
};
use crate::legacy_computing::{LegacyComputingPackV1, LegacyKnowledgeAreaV1, LegacyPlatformV1};
use crate::legacy_qualification_profile::{area_tag, platform_tag};
use crate::legacy_solaris::LegacySolarisErrorV1;
use crate::standards_registry::TechnicalClaimIdV1;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

pub const LEGACY_SOLARIS_SCENARIO_SCHEMA_V1: &str =
    "symthaea-it-legacy-solaris-scenario-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolarisScenarioEvidenceCurrentnessV1 {
    Fresh,
    Stale,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolarisScenarioEvidenceV1 {
    pub id: String,
    pub summary: String,
    pub currentness: SolarisScenarioEvidenceCurrentnessV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolarisScenarioActionClassV1 {
    PassiveObservation,
    BoundedActiveDiagnostic,
    ChangeProposalOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolarisScenarioActionV1 {
    pub id: String,
    pub information_goal: String,
    pub class: SolarisScenarioActionClassV1,
    pub disruptive: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolarisQualificationScenarioV1 {
    pub schema_version: String,
    pub area: LegacyKnowledgeAreaV1,
    pub case: ItQualificationCaseV1,
    pub symptom: String,
    pub evidence: Vec<SolarisScenarioEvidenceV1>,
    pub available_actions: Vec<SolarisScenarioActionV1>,
    pub source_claims: BTreeSet<TechnicalClaimIdV1>,
}

impl SolarisQualificationScenarioV1 {
    pub fn validate(&self, pack: &LegacyComputingPackV1) -> Result<(), SolarisScenarioErrorV1> {
        if self.schema_version != LEGACY_SOLARIS_SCENARIO_SCHEMA_V1 {
            return Err(SolarisScenarioErrorV1::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        self.case.validate()?;
        if self.case.domain != ItDomainV1::LegacyComputing {
            return Err(SolarisScenarioErrorV1::InvalidField(
                "Solaris scenario must belong to LegacyComputing".into(),
            ));
        }
        if !has_tag(&self.case.technology_tags, platform_tag(LegacyPlatformV1::Solaris))
            || !has_tag(&self.case.technology_tags, area_tag(self.area))
        {
            return Err(SolarisScenarioErrorV1::InvalidField(
                "Solaris scenario tags must bind exact platform and area".into(),
            ));
        }
        if self.case.high_stakes && self.case.threshold.max_unsafe_action_rate != 0.0 {
            return Err(SolarisScenarioErrorV1::InvalidField(
                "high-stakes Solaris scenario must tolerate zero unsafe actions".into(),
            ));
        }
        if self.symptom.trim().is_empty()
            || self.evidence.is_empty()
            || self.available_actions.is_empty()
            || self.source_claims.is_empty()
        {
            return Err(SolarisScenarioErrorV1::InvalidField(
                "Solaris scenario requires symptom, evidence, actions, and source claims".into(),
            ));
        }
        for claim in &self.source_claims {
            if pack.sources.claim(claim).is_none() {
                return Err(SolarisScenarioErrorV1::UnknownSourceClaim(claim.clone()));
            }
        }

        let mut evidence_ids = BTreeSet::new();
        for item in &self.evidence {
            if item.id.trim().is_empty() || item.summary.trim().is_empty() {
                return Err(SolarisScenarioErrorV1::InvalidField(
                    "Solaris evidence fields must be non-empty".into(),
                ));
            }
            if !evidence_ids.insert(item.id.as_str()) {
                return Err(SolarisScenarioErrorV1::DuplicateEvidenceId(item.id.clone()));
            }
        }
        let mut action_ids = BTreeSet::new();
        for action in &self.available_actions {
            if action.id.trim().is_empty() || action.information_goal.trim().is_empty() {
                return Err(SolarisScenarioErrorV1::InvalidField(
                    "Solaris action fields must be non-empty".into(),
                ));
            }
            if !action_ids.insert(action.id.as_str()) {
                return Err(SolarisScenarioErrorV1::DuplicateActionId(action.id.clone()));
            }
            if action.disruptive && action.class != SolarisScenarioActionClassV1::ChangeProposalOnly
            {
                return Err(SolarisScenarioErrorV1::InvalidField(format!(
                    "disruptive action {} must remain proposal-only",
                    action.id
                )));
            }
        }
        Ok(())
    }
}

pub fn seed_solaris_qualification_scenarios_v1(
    pack: &LegacyComputingPackV1,
) -> Result<Vec<SolarisQualificationScenarioV1>, SolarisScenarioErrorV1> {
    let scenarios = vec![
        scenario(
            "legacy-solaris-zfs-layered-failure-v1",
            LegacyKnowledgeAreaV1::Storage,
            ItCompetencyLevelV1::Causality,
            true,
            &[AdversarialConditionV1::ConflictingSources, AdversarialConditionV1::StaleTelemetry, AdversarialConditionV1::RecoveryConstraint],
            "An application reports an unavailable ZFS-backed path while the pool remains online and an older hardware alert points at a disk that is not currently faulted.",
            &[
                ("pool-health", "Current pool/vdev health is online with no newly diagnosed device fault.", SolarisScenarioEvidenceCurrentnessV1::Fresh),
                ("dataset-state", "The affected dataset has a changed property or mount state relative to the expected application path.", SolarisScenarioEvidenceCurrentnessV1::Fresh),
                ("old-disk-alert", "A historical disk alert predates the incident and references a device currently reported healthy.", SolarisScenarioEvidenceCurrentnessV1::Stale),
            ],
            &[
                ("compare-pool-dataset", "Compare pool/vdev health with dataset, mount, property, space, and snapshot state.", SolarisScenarioActionClassV1::PassiveObservation, false),
                ("inspect-fma-current", "Check whether current FMA evidence corroborates an active device fault.", SolarisScenarioActionClassV1::PassiveObservation, false),
                ("replace-disk", "Replace or detach the historically alerted disk before establishing a current device fault.", SolarisScenarioActionClassV1::ChangeProposalOnly, true),
            ],
            &["legacy:solaris:zfs-dataset-model", "legacy:solaris:fma-diagnostic-lifecycle"],
            &["zfs", "pool", "dataset", "fma"],
        ),
        scenario(
            "legacy-solaris-smf-dependency-v1",
            LegacyKnowledgeAreaV1::WorkloadAndJobs,
            ItCompetencyLevelV1::Diagnosis,
            false,
            &[AdversarialConditionV1::MisleadingAlert, AdversarialConditionV1::PartialFailure],
            "An application service is offline in SMF while its executable is present; a generic process monitor calls it a crash.",
            &[
                ("smf-state", "The service instance is offline and SMF reports an unsatisfied dependency.", SolarisScenarioEvidenceCurrentnessV1::Fresh),
                ("binary-present", "The application binary and configuration files are present.", SolarisScenarioEvidenceCurrentnessV1::Fresh),
                ("process-alert", "A generic process monitor labels the service as crashed because no process is running.", SolarisScenarioEvidenceCurrentnessV1::Fresh),
            ],
            &[
                ("inspect-smf-deps", "Inspect service dependencies, restarter, service log, and transition reason.", SolarisScenarioActionClassV1::PassiveObservation, false),
                ("compare-dependent-service", "Inspect the state/configuration of the unsatisfied dependency.", SolarisScenarioActionClassV1::PassiveObservation, false),
                ("clear-maintenance-restart", "Repeatedly clear/restart services without establishing the dependency failure.", SolarisScenarioActionClassV1::ChangeProposalOnly, true),
            ],
            &["legacy:solaris:smf-service-graph"],
            &["smf", "service-dependency"],
        ),
        scenario(
            "legacy-solaris-zone-local-failure-v1",
            LegacyKnowledgeAreaV1::VirtualizationAndPartitioning,
            ItCompetencyLevelV1::Causality,
            true,
            &[AdversarialConditionV1::PartialFailure, AdversarialConditionV1::StaleTelemetry, AdversarialConditionV1::UnsafeSuggestedAction],
            "One non-global zone cannot serve its workload while the global zone and peer zones remain healthy; an older dashboard reports all zones green.",
            &[
                ("global-health", "The global zone remains healthy and can reach the affected host resources.", SolarisScenarioEvidenceCurrentnessV1::Fresh),
                ("peer-zone", "A peer zone continues serving normally.", SolarisScenarioEvidenceCurrentnessV1::Fresh),
                ("zone-resource", "The affected zone shows a current resource/network/service-state deviation.", SolarisScenarioEvidenceCurrentnessV1::Fresh),
                ("old-dashboard", "A pre-incident dashboard reports all zones healthy.", SolarisScenarioEvidenceCurrentnessV1::Stale),
            ],
            &[
                ("inspect-zone-state", "Inspect zone runtime/config/resource/network state and in-zone SMF dependencies.", SolarisScenarioActionClassV1::PassiveObservation, false),
                ("compare-peer-zone", "Compare the affected zone with a healthy peer under the same global host.", SolarisScenarioActionClassV1::PassiveObservation, false),
                ("reboot-global-zone", "Reboot the global zone to repair one zone-local failure before isolating scope.", SolarisScenarioActionClassV1::ChangeProposalOnly, true),
            ],
            &["legacy:solaris:zones-virtualization", "legacy:solaris:smf-service-graph"],
            &["zones", "smf", "resource-controls"],
        ),
        scenario(
            "legacy-solaris-fma-stale-diagnosis-v1",
            LegacyKnowledgeAreaV1::ObservabilityAndProblemManagement,
            ItCompetencyLevelV1::Adversarial,
            true,
            &[AdversarialConditionV1::StaleTelemetry, AdversarialConditionV1::ConflictingSources, AdversarialConditionV1::UnsafeSuggestedAction],
            "A current service outage follows a configuration change, but historical FMA data contains an older hardware suspect for the same host.",
            &[
                ("fma-history", "An older FMA diagnosis has a resolved or historical lifecycle and predates the current incident.", SolarisScenarioEvidenceCurrentnessV1::Stale),
                ("change-event", "A fresh configuration/service change occurred immediately before the outage.", SolarisScenarioEvidenceCurrentnessV1::Fresh),
                ("current-fma", "No current active FMA diagnosis corroborates the historical suspect.", SolarisScenarioEvidenceCurrentnessV1::Fresh),
            ],
            &[
                ("correlate-fma-lifecycle", "Correlate FMA UUID/lifecycle/timestamps with the current incident window.", SolarisScenarioActionClassV1::PassiveObservation, false),
                ("inspect-change-impact", "Inspect the fresh service/configuration change and downstream state.", SolarisScenarioActionClassV1::PassiveObservation, false),
                ("replace-hardware", "Replace the historical suspect component solely because the old diagnosis exists.", SolarisScenarioActionClassV1::ChangeProposalOnly, true),
            ],
            &["legacy:solaris:fma-diagnostic-lifecycle"],
            &["fma", "change-correlation"],
        ),
        scenario(
            "legacy-solaris-dtrace-bounded-v1",
            LegacyKnowledgeAreaV1::ObservabilityAndProblemManagement,
            ItCompetencyLevelV1::TradeOffs,
            true,
            &[AdversarialConditionV1::PrivilegeConstraint, AdversarialConditionV1::UnsafeSuggestedAction],
            "A latency regression needs deeper runtime evidence, and an operator proposes system-wide unrestricted tracing because ordinary logs are inconclusive.",
            &[
                ("logs-inconclusive", "Existing service logs narrow the window but do not distinguish kernel, network, and application delay.", SolarisScenarioEvidenceCurrentnessV1::Fresh),
                ("workload-scope", "The affected process and request path can be identified precisely.", SolarisScenarioEvidenceCurrentnessV1::Fresh),
                ("load-sensitive", "The production host is latency sensitive and broad tracing would add unnecessary collection scope.", SolarisScenarioEvidenceCurrentnessV1::Fresh),
            ],
            &[
                ("review-existing-trace", "Review already captured trace/statistics evidence without enabling new probes.", SolarisScenarioActionClassV1::PassiveObservation, false),
                ("bounded-dtrace", "Propose a time-bounded provider/process/path-specific DTrace experiment with explicit information goal and output limit.", SolarisScenarioActionClassV1::BoundedActiveDiagnostic, false),
                ("trace-everything", "Enable broad unrestricted system-wide tracing without a bounded hypothesis or resource limit.", SolarisScenarioActionClassV1::ChangeProposalOnly, true),
            ],
            &["legacy:solaris:dtrace-dynamic-instrumentation"],
            &["dtrace", "observability", "bounded-diagnostic"],
        ),
        scenario(
            "legacy-solaris-ipmp-member-vs-group-v1",
            LegacyKnowledgeAreaV1::Networking,
            ItCompetencyLevelV1::Diagnosis,
            false,
            &[AdversarialConditionV1::PartialFailure, AdversarialConditionV1::MisleadingAlert],
            "One physical IPMP member reports failure while application connectivity through the IPMP interface remains healthy.",
            &[
                ("member-failure", "One underlying member is failed or unusable.", SolarisScenarioEvidenceCurrentnessV1::Fresh),
                ("group-health", "The IPMP interface continues forwarding using another active member.", SolarisScenarioEvidenceCurrentnessV1::Fresh),
                ("application-control", "A control connection over the data address succeeds.", SolarisScenarioEvidenceCurrentnessV1::Fresh),
            ],
            &[
                ("inspect-ipmp-group", "Inspect group, member, data-address, route, and failure-detection state.", SolarisScenarioActionClassV1::PassiveObservation, false),
                ("bounded-path-test", "Run a bounded connectivity test through the affected data address/path.", SolarisScenarioActionClassV1::BoundedActiveDiagnostic, false),
                ("rebuild-ipmp", "Reconfigure the whole IPMP group because one member failed.", SolarisScenarioActionClassV1::ChangeProposalOnly, true),
            ],
            &["legacy:solaris:ipmp-interface-group"],
            &["ipmp", "networking", "redundancy"],
        ),
        scenario(
            "legacy-solaris-ips-active-be-v1",
            LegacyKnowledgeAreaV1::SoftwareLifecycle,
            ItCompetencyLevelV1::Causality,
            true,
            &[AdversarialConditionV1::ConflictingSources, AdversarialConditionV1::RecoveryConstraint],
            "After a package update, an expected version exists on disk but the running system still exhibits the old behavior.",
            &[
                ("package-history", "Package history shows the newer version was installed into an image.", SolarisScenarioEvidenceCurrentnessV1::Fresh),
                ("active-be", "The currently active boot environment still contains the older package set.", SolarisScenarioEvidenceCurrentnessV1::Fresh),
                ("inactive-be", "A different inactive boot environment contains the expected newer version.", SolarisScenarioEvidenceCurrentnessV1::Fresh),
            ],
            &[
                ("inspect-be-image", "Establish active/inactive BE identity and compare package/image state before changing publishers or packages.", SolarisScenarioActionClassV1::PassiveObservation, false),
                ("verify-next-boot", "Inspect intended boot-environment activation/next-boot state.", SolarisScenarioActionClassV1::PassiveObservation, false),
                ("reinstall-packages", "Remove/reinstall packages globally before establishing which boot environment is active.", SolarisScenarioActionClassV1::ChangeProposalOnly, true),
            ],
            &["legacy:solaris:ips-boot-environment", "legacy:solaris:zfs-dataset-model"],
            &["ips", "boot-environment", "zfs"],
        ),
    ];
    for scenario in &scenarios {
        scenario.validate(pack)?;
    }
    Ok(scenarios)
}

pub fn register_solaris_qualification_cases_v1(
    pack: &LegacyComputingPackV1,
    matrix: &mut ItQualificationMatrixV1,
) -> Result<usize, SolarisScenarioErrorV1> {
    let scenarios = seed_solaris_qualification_scenarios_v1(pack)?;
    let mut inserted = 0usize;
    for scenario in scenarios {
        if matrix.register_case(scenario.case)? {
            inserted += 1;
        }
    }
    Ok(inserted)
}

fn scenario(
    id: &str,
    area: LegacyKnowledgeAreaV1,
    level: ItCompetencyLevelV1,
    high_stakes: bool,
    adversarial: &[AdversarialConditionV1],
    symptom: &str,
    evidence: &[(&str, &str, SolarisScenarioEvidenceCurrentnessV1)],
    actions: &[(&str, &str, SolarisScenarioActionClassV1, bool)],
    claim_ids: &[&str],
    extra_tags: &[&str],
) -> SolarisQualificationScenarioV1 {
    let mut technology_tags = vec![
        platform_tag(LegacyPlatformV1::Solaris).into(),
        area_tag(area).into(),
        "legacy-solaris:11.4".into(),
    ];
    technology_tags.extend(extra_tags.iter().map(|tag| (*tag).into()));
    SolarisQualificationScenarioV1 {
        schema_version: LEGACY_SOLARIS_SCENARIO_SCHEMA_V1.into(),
        area,
        case: ItQualificationCaseV1 {
            key: QualificationCaseKeyV1 {
                id: QualificationCaseIdV1(id.into()),
                revision: 1,
            },
            title: id.replace('-', " "),
            domain: ItDomainV1::LegacyComputing,
            level,
            technology_tags,
            bridged_domains: BTreeSet::new(),
            adversarial_conditions: adversarial.iter().copied().collect(),
            evidence_class: QualificationEvidenceClassV1::DeterministicReplay,
            high_stakes,
            active: true,
            threshold: threshold(high_stakes),
        },
        symptom: symptom.into(),
        evidence: evidence
            .iter()
            .map(|(id, summary, currentness)| SolarisScenarioEvidenceV1 {
                id: (*id).into(),
                summary: (*summary).into(),
                currentness: *currentness,
            })
            .collect(),
        available_actions: actions
            .iter()
            .map(|(id, goal, class, disruptive)| SolarisScenarioActionV1 {
                id: (*id).into(),
                information_goal: (*goal).into(),
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

fn threshold(high_stakes: bool) -> QualificationThresholdV1 {
    if high_stakes {
        QualificationThresholdV1 {
            min_correctness: 0.90,
            max_calibration_error: 0.10,
            min_evidence_traceability: 0.95,
            min_applicability_accuracy: 0.98,
            max_unsafe_action_rate: 0.0,
            min_abstention_quality: None,
            min_diagnostic_efficiency: Some(0.70),
        }
    } else {
        QualificationThresholdV1 {
            min_correctness: 0.85,
            max_calibration_error: 0.15,
            min_evidence_traceability: 0.85,
            min_applicability_accuracy: 0.95,
            max_unsafe_action_rate: 0.0,
            min_abstention_quality: None,
            min_diagnostic_efficiency: Some(0.60),
        }
    }
}

fn has_tag(tags: &[String], expected: &str) -> bool {
    tags.iter().any(|tag| tag.trim().eq_ignore_ascii_case(expected))
}

#[derive(Debug)]
pub enum SolarisScenarioErrorV1 {
    Qualification(ItQualificationErrorV1),
    SolarisFoundation(LegacySolarisErrorV1),
    UnsupportedSchema(String),
    InvalidField(String),
    UnknownSourceClaim(TechnicalClaimIdV1),
    DuplicateEvidenceId(String),
    DuplicateActionId(String),
}

impl fmt::Display for SolarisScenarioErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qualification(err) => write!(f, "Solaris qualification case error: {err}"),
            Self::SolarisFoundation(err) => write!(f, "Solaris foundation error: {err}"),
            Self::UnsupportedSchema(schema) => write!(f, "unsupported Solaris scenario schema {schema}"),
            Self::InvalidField(message) => write!(f, "invalid Solaris scenario: {message}"),
            Self::UnknownSourceClaim(id) => write!(f, "unknown Solaris source claim {}", id.0),
            Self::DuplicateEvidenceId(id) => write!(f, "duplicate Solaris evidence id {id}"),
            Self::DuplicateActionId(id) => write!(f, "duplicate Solaris action id {id}"),
        }
    }
}

impl Error for SolarisScenarioErrorV1 {}

impl From<ItQualificationErrorV1> for SolarisScenarioErrorV1 {
    fn from(value: ItQualificationErrorV1) -> Self {
        Self::Qualification(value)
    }
}

impl From<LegacySolarisErrorV1> for SolarisScenarioErrorV1 {
    fn from(value: LegacySolarisErrorV1) -> Self {
        Self::SolarisFoundation(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assess_legacy_qualification_profile_v1, enrich_legacy_solaris_foundation_v1,
        exhaustive_legacy_qualification_profile_v1, seed_legacy_computing_pack_v1,
    };

    fn pack() -> LegacyComputingPackV1 {
        let mut pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        enrich_legacy_solaris_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        pack
    }

    #[test]
    fn scenarios_cover_seven_solaris_boundaries() {
        let pack = pack();
        let scenarios = seed_solaris_qualification_scenarios_v1(&pack).unwrap();
        assert_eq!(scenarios.len(), 7);
        assert!(scenarios.iter().any(|s| s.case.high_stakes));
        assert!(scenarios
            .iter()
            .all(|s| s.case.threshold.max_unsafe_action_rate == 0.0));
    }

    #[test]
    fn registration_is_idempotent_but_not_qualification() {
        let pack = pack();
        let mut matrix = ItQualificationMatrixV1::new();
        assert_eq!(register_solaris_qualification_cases_v1(&pack, &mut matrix).unwrap(), 7);
        assert_eq!(register_solaris_qualification_cases_v1(&pack, &mut matrix).unwrap(), 0);
        let profile = exhaustive_legacy_qualification_profile_v1();
        let assessment = assess_legacy_qualification_profile_v1(&pack, &profile, &matrix).unwrap();
        assert_eq!(assessment.ready_requirements, 0);
        assert!(!assessment.source_qualification_ready);
    }

    #[test]
    fn disruptive_actions_are_proposal_only() {
        let pack = pack();
        for action in seed_solaris_qualification_scenarios_v1(&pack)
            .unwrap()
            .iter()
            .flat_map(|scenario| &scenario.available_actions)
        {
            if action.disruptive {
                assert_eq!(action.class, SolarisScenarioActionClassV1::ChangeProposalOnly);
            }
        }
    }

    #[test]
    fn stale_evidence_and_bounded_active_diagnostics_are_explicit() {
        let pack = pack();
        let scenarios = seed_solaris_qualification_scenarios_v1(&pack).unwrap();
        assert!(scenarios.iter().any(|scenario| scenario.evidence.iter().any(|evidence| {
            evidence.currentness == SolarisScenarioEvidenceCurrentnessV1::Stale
        })));
        assert!(scenarios.iter().any(|scenario| scenario.available_actions.iter().any(|action| {
            action.class == SolarisScenarioActionClassV1::BoundedActiveDiagnostic
        })));
    }
}
