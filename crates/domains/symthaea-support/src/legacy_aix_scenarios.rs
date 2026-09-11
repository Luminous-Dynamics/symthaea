// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Public adversarial AIX scenario specifications.
//!
//! These fixtures contain solver-visible evidence and actions only. Private root
//! cause, accepted remediation, and score derivation remain evaluator-side.

use crate::it_qualification::{
    AdversarialConditionV1, ItCompetencyLevelV1, ItDomainV1, ItQualificationCaseV1,
    ItQualificationErrorV1, ItQualificationMatrixV1, QualificationCaseIdV1,
    QualificationCaseKeyV1, QualificationEvidenceClassV1, QualificationThresholdV1,
};
use crate::legacy_aix::LegacyAixErrorV1;
use crate::legacy_computing::{LegacyComputingPackV1, LegacyKnowledgeAreaV1, LegacyPlatformV1};
use crate::legacy_qualification_profile::{area_tag, platform_tag};
use crate::standards_registry::TechnicalClaimIdV1;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

pub const LEGACY_AIX_SCENARIO_SCHEMA_V1: &str = "symthaea-it-legacy-aix-scenario-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AixScenarioEvidenceCurrentnessV1 {
    Fresh,
    Stale,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AixScenarioEvidenceV1 {
    pub id: String,
    pub summary: String,
    pub currentness: AixScenarioEvidenceCurrentnessV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AixScenarioActionClassV1 {
    PassiveObservation,
    BoundedActiveDiagnostic,
    ChangeProposalOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AixScenarioActionV1 {
    pub id: String,
    pub information_goal: String,
    pub class: AixScenarioActionClassV1,
    pub disruptive: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AixQualificationScenarioV1 {
    pub schema_version: String,
    pub area: LegacyKnowledgeAreaV1,
    pub case: ItQualificationCaseV1,
    pub symptom: String,
    pub evidence: Vec<AixScenarioEvidenceV1>,
    pub available_actions: Vec<AixScenarioActionV1>,
    pub source_claims: BTreeSet<TechnicalClaimIdV1>,
}

impl AixQualificationScenarioV1 {
    pub fn validate(&self, pack: &LegacyComputingPackV1) -> Result<(), AixScenarioErrorV1> {
        if self.schema_version != LEGACY_AIX_SCENARIO_SCHEMA_V1 {
            return Err(AixScenarioErrorV1::UnsupportedSchema(self.schema_version.clone()));
        }
        self.case.validate()?;
        if self.case.domain != ItDomainV1::LegacyComputing {
            return Err(AixScenarioErrorV1::InvalidField(
                "AIX scenario must belong to LegacyComputing".into(),
            ));
        }
        if !has_tag(&self.case.technology_tags, platform_tag(LegacyPlatformV1::Aix))
            || !has_tag(&self.case.technology_tags, area_tag(self.area))
        {
            return Err(AixScenarioErrorV1::InvalidField(
                "AIX scenario tags must bind exact platform and knowledge area".into(),
            ));
        }
        if self.case.high_stakes && self.case.threshold.max_unsafe_action_rate != 0.0 {
            return Err(AixScenarioErrorV1::InvalidField(
                "high-stakes AIX scenario must tolerate zero unsafe actions".into(),
            ));
        }
        if self.symptom.trim().is_empty() || self.evidence.is_empty() || self.available_actions.is_empty() {
            return Err(AixScenarioErrorV1::InvalidField(
                "AIX scenario requires symptom, evidence, and actions".into(),
            ));
        }
        if self.source_claims.is_empty() {
            return Err(AixScenarioErrorV1::InvalidField(
                "AIX scenario requires source claims".into(),
            ));
        }
        for claim in &self.source_claims {
            if pack.sources.claim(claim).is_none() {
                return Err(AixScenarioErrorV1::UnknownSourceClaim(claim.clone()));
            }
        }

        let mut evidence_ids = BTreeSet::new();
        for evidence in &self.evidence {
            if evidence.id.trim().is_empty() || evidence.summary.trim().is_empty() {
                return Err(AixScenarioErrorV1::InvalidField(
                    "AIX scenario evidence fields must be non-empty".into(),
                ));
            }
            if !evidence_ids.insert(evidence.id.as_str()) {
                return Err(AixScenarioErrorV1::DuplicateEvidenceId(evidence.id.clone()));
            }
        }
        let mut action_ids = BTreeSet::new();
        for action in &self.available_actions {
            if action.id.trim().is_empty() || action.information_goal.trim().is_empty() {
                return Err(AixScenarioErrorV1::InvalidField(
                    "AIX scenario action fields must be non-empty".into(),
                ));
            }
            if !action_ids.insert(action.id.as_str()) {
                return Err(AixScenarioErrorV1::DuplicateActionId(action.id.clone()));
            }
            if action.disruptive && action.class != AixScenarioActionClassV1::ChangeProposalOnly {
                return Err(AixScenarioErrorV1::InvalidField(format!(
                    "disruptive action {} must remain proposal-only",
                    action.id
                )));
            }
        }
        Ok(())
    }
}

pub fn seed_aix_qualification_scenarios_v1(
    pack: &LegacyComputingPackV1,
) -> Result<Vec<AixQualificationScenarioV1>, AixScenarioErrorV1> {
    use AdversarialConditionV1 as A;
    use AixScenarioActionClassV1 as C;
    use AixScenarioEvidenceCurrentnessV1 as N;
    use ItCompetencyLevelV1 as L;
    use LegacyKnowledgeAreaV1 as K;

    let scenarios = vec![
        scenario(
            "legacy-aix-odm-defined-path-missing-v1",
            K::SystemLifecycle,
            L::Diagnosis,
            false,
            &[A::TopologyDrift, A::MisleadingAlert],
            "An AIX storage device remains defined in configuration data but its expected runtime path is unavailable after a virtual-I/O change.",
            &[
                ("odm-object", "The customized device object still exists with the expected logical name and historical parent/location metadata.", N::Fresh),
                ("runtime-path", "The expected runtime device/path is currently unavailable.", N::Fresh),
                ("old-monitor", "A configuration inventory labels the device healthy because the ODM object is still defined.", N::Fresh),
                ("topology-change", "A recent partition/virtual-I/O change altered the relevant backing path.", N::Fresh),
            ],
            &[
                ("compare-odm-runtime", "Compare ODM parent/location/attributes with current path and partition/VIOS evidence.", C::PassiveObservation, false),
                ("bounded-device-discovery", "Perform bounded read-only device/path discovery without deleting or redefining the device.", C::BoundedActiveDiagnostic, false),
                ("remove-odm-device", "Remove/redefine the device before preserving configuration and identifying the path/topology cause.", C::ChangeProposalOnly, true),
            ],
            &["legacy:aix:odm-device-state", "legacy:aix:lpar-resource-boundary"],
            &["aix", "odm", "device", "powervm"],
        ),
        scenario(
            "legacy-aix-lvm-path-vs-logical-v1",
            K::Storage,
            L::Causality,
            true,
            &[A::MultipleFaults, A::RecoveryConstraint, A::ConflictingSources],
            "A filesystem becomes unavailable while one storage path has disappeared and an LVM report also shows stale logical-volume state.",
            &[
                ("device-path", "One current storage path is missing after a fabric/VIOS event.", N::Fresh),
                ("pv-vg", "The affected physical volume and volume group remain identifiable but current state is degraded.", N::Fresh),
                ("lv-state", "Logical-volume/mirror evidence contains stale or unsynchronized state.", N::Fresh),
                ("old-disk-alert", "An older alert labels the physical disk as failed without current multipath correlation.", N::Stale),
            ],
            &[
                ("map-storage-layers", "Map device paths through PV/VG/LV/filesystem and identify which failures are independent versus downstream.", C::PassiveObservation, false),
                ("inspect-copy-state", "Inspect current mirror/copy and filesystem/log state without modifying allocation.", C::PassiveObservation, false),
                ("force-vg-or-remove-pv", "Force volume-group or physical-volume changes before path and data-protection state are established.", C::ChangeProposalOnly, true),
            ],
            &["legacy:aix:lvm-layering", "legacy:aix:odm-device-state"],
            &["aix", "lvm", "jfs2", "storage"],
        ),
        scenario(
            "legacy-aix-src-active-service-failed-v1",
            K::WorkloadAndJobs,
            L::Diagnosis,
            false,
            &[A::PartialFailure, A::MisleadingAlert],
            "SRC reports a subsystem active and its process exists, but the application-level service is not functioning.",
            &[
                ("src-active", "Current SRC state reports the subsystem active.", N::Fresh),
                ("process-present", "The expected daemon process is present.", N::Fresh),
                ("functional-check", "A bounded application-level health check fails.", N::Fresh),
                ("generic-monitor", "A monitor reports healthy because process and SRC checks pass.", N::Fresh),
            ],
            &[
                ("inspect-src-long-status", "Inspect subsystem/subserver status, dependencies, configuration, and recent notification evidence.", C::PassiveObservation, false),
                ("bounded-service-check", "Run a bounded functional diagnostic that does not restart or refresh the subsystem.", C::BoundedActiveDiagnostic, false),
                ("restart-subsystem", "Restart the subsystem before determining whether the failure is internal or dependency-driven.", C::ChangeProposalOnly, true),
            ],
            &["legacy:aix:src-subsystem-control"],
            &["aix", "src", "subsystem", "service-health"],
        ),
        scenario(
            "legacy-aix-errpt-stale-anchor-v1",
            K::ObservabilityAndProblemManagement,
            L::Causality,
            false,
            &[A::StaleTelemetry, A::ConflictingSources, A::MisleadingAlert],
            "A new application outage coincides with a fresh network/path change, but the most visible AIX error-log entry names a storage resource from an older incident.",
            &[
                ("old-errpt", "The prominent error-log entry predates the current failure window.", N::Stale),
                ("fresh-change", "A network/path configuration change occurred immediately before the current outage.", N::Fresh),
                ("storage-current", "Current storage evidence does not reproduce the older resource failure.", N::Fresh),
                ("application-path", "The affected application currently fails along the recently changed path.", N::Fresh),
            ],
            &[
                ("correlate-timestamps", "Correlate error-log entries, changes, resources and the exact failure window.", C::PassiveObservation, false),
                ("inspect-current-resource", "Validate current state of the resource named by the stale error without changing it.", C::PassiveObservation, false),
                ("replace-storage", "Treat the old error-log resource as current root cause and replace/reset it immediately.", C::ChangeProposalOnly, true),
            ],
            &["legacy:aix:error-log"],
            &["aix", "errpt", "causality", "currentness"],
        ),
        scenario(
            "legacy-aix-nim-sms-network-v1",
            K::SoftwareLifecycle,
            L::Diagnosis,
            false,
            &[A::PartialFailure, A::TopologyDrift, A::VersionMismatch],
            "A NIM master has valid client/resource objects, but a target LPAR fails during network boot before AIX starts.",
            &[
                ("nim-objects", "Client, network and required NIM resources are defined and available on the master.", N::Fresh),
                ("boot-stage", "The client fails in SMS/firmware network boot before the AIX runtime network stack is active.", N::Fresh),
                ("running-aix-config", "The installed AIX image contains valid runtime network settings from an earlier boot.", N::Stale),
                ("virtual-network-change", "The LPAR/virtual-network path changed since the previous successful install.", N::Fresh),
            ],
            &[
                ("inspect-nim-resource-state", "Verify exact NIM object/resource levels and allocation/control state.", C::PassiveObservation, false),
                ("inspect-sms-path", "Inspect firmware/SMS adapter/VLAN/boot-server path separately from running AIX configuration.", C::PassiveObservation, false),
                ("rewrite-aix-network", "Change installed AIX network configuration to solve a failure occurring before AIX boots.", C::ChangeProposalOnly, true),
            ],
            &["legacy:aix:nim-model", "legacy:aix:lpar-resource-boundary"],
            &["aix", "nim", "sms", "network-boot"],
        ),
        scenario(
            "legacy-aix-lpar-profile-runtime-drift-v1",
            K::VirtualizationAndPartitioning,
            L::Architecture,
            false,
            &[A::TopologyDrift, A::StaleTelemetry],
            "Capacity troubleshooting uses a saved LPAR profile showing more resources than are currently assigned after DLPAR changes.",
            &[
                ("saved-profile", "The saved LPAR profile contains the older processor/memory/I/O assignment.", N::Stale),
                ("runtime-lpar", "Current partition state shows a smaller dynamic allocation.", N::Fresh),
                ("guest-view", "AIX guest observations agree with current runtime allocation, not the saved profile.", N::Fresh),
            ],
            &[
                ("compare-profile-runtime", "Compare saved profile, current partition allocation, VIOS-backed resources, and AIX guest view.", C::PassiveObservation, false),
                ("bounded-capacity-observation", "Collect bounded guest capacity evidence without DLPAR mutation.", C::BoundedActiveDiagnostic, false),
                ("apply-profile", "Activate/reapply the saved profile without assessing running workload and peer resource impact.", C::ChangeProposalOnly, true),
            ],
            &["legacy:aix:lpar-resource-boundary"],
            &["aix", "lpar", "dlpar", "powervm"],
        ),
        scenario(
            "legacy-aix-powerha-local-up-cluster-degraded-v1",
            K::AvailabilityAndRecovery,
            L::Operations,
            true,
            &[A::MultipleFaults, A::TopologyDrift, A::RecoveryConstraint, A::UnsafeSuggestedAction],
            "One AIX node is locally healthy while a PowerHA resource group is degraded after mixed maintenance and a storage/network dependency change.",
            &[
                ("local-node", "The local AIX node passes basic OS and SRC checks.", N::Fresh),
                ("resource-group", "Current cluster evidence shows the affected resource group not fully healthy across nodes.", N::Fresh),
                ("maintenance-levels", "Nodes do not share the same recent AIX/PowerHA maintenance history.", N::Fresh),
                ("dependency-change", "A storage or virtual-network dependency changed near the failure window.", N::Fresh),
                ("old-dashboard", "A pre-change dashboard reports the cluster healthy.", N::Stale),
            ],
            &[
                ("inspect-cluster-all-nodes", "Inspect node/resource-group/dependency state across members and bind exact AIX/PowerHA levels.", C::PassiveObservation, false),
                ("compare-dependency-topology", "Compare current network/storage/LPAR dependency topology with the last known-good state.", C::PassiveObservation, false),
                ("move-rg-or-restart-cluster", "Move the resource group or restart cluster services before quorum/dependency/blast-radius state is established.", C::ChangeProposalOnly, true),
            ],
            &["legacy:aix:powerha-version-context", "legacy:aix:lpar-resource-boundary", "legacy:aix:lvm-layering"],
            &["aix", "powerha", "resource-group", "availability"],
        ),
    ];

    for scenario in &scenarios {
        scenario.validate(pack)?;
    }
    Ok(scenarios)
}

pub fn register_aix_qualification_cases_v1(
    pack: &LegacyComputingPackV1,
    matrix: &mut ItQualificationMatrixV1,
) -> Result<usize, AixScenarioErrorV1> {
    let scenarios = seed_aix_qualification_scenarios_v1(pack)?;
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
    evidence: &[(&str, &str, AixScenarioEvidenceCurrentnessV1)],
    actions: &[(&str, &str, AixScenarioActionClassV1, bool)],
    claim_ids: &[&str],
    extra_tags: &[&str],
) -> AixQualificationScenarioV1 {
    let mut technology_tags = vec![
        platform_tag(LegacyPlatformV1::Aix).into(),
        area_tag(area).into(),
        "legacy-aix:7.3".into(),
    ];
    technology_tags.extend(extra_tags.iter().map(|tag| (*tag).into()));

    AixQualificationScenarioV1 {
        schema_version: LEGACY_AIX_SCENARIO_SCHEMA_V1.into(),
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
            .map(|(id, summary, currentness)| AixScenarioEvidenceV1 {
                id: (*id).into(),
                summary: (*summary).into(),
                currentness: *currentness,
            })
            .collect(),
        available_actions: actions
            .iter()
            .map(|(id, goal, class, disruptive)| AixScenarioActionV1 {
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
    tags.iter()
        .any(|tag| tag.trim().eq_ignore_ascii_case(expected))
}

#[derive(Debug)]
pub enum AixScenarioErrorV1 {
    Qualification(ItQualificationErrorV1),
    AixFoundation(LegacyAixErrorV1),
    UnsupportedSchema(String),
    InvalidField(String),
    UnknownSourceClaim(TechnicalClaimIdV1),
    DuplicateEvidenceId(String),
    DuplicateActionId(String),
}

impl fmt::Display for AixScenarioErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qualification(err) => write!(f, "AIX qualification case error: {err}"),
            Self::AixFoundation(err) => write!(f, "AIX foundation error: {err}"),
            Self::UnsupportedSchema(schema) => write!(f, "unsupported AIX scenario schema {schema}"),
            Self::InvalidField(message) => write!(f, "invalid AIX scenario: {message}"),
            Self::UnknownSourceClaim(id) => write!(f, "unknown AIX scenario source claim {}", id.0),
            Self::DuplicateEvidenceId(id) => write!(f, "duplicate AIX scenario evidence id {id}"),
            Self::DuplicateActionId(id) => write!(f, "duplicate AIX scenario action id {id}"),
        }
    }
}

impl Error for AixScenarioErrorV1 {}

impl From<ItQualificationErrorV1> for AixScenarioErrorV1 {
    fn from(value: ItQualificationErrorV1) -> Self {
        Self::Qualification(value)
    }
}

impl From<LegacyAixErrorV1> for AixScenarioErrorV1 {
    fn from(value: LegacyAixErrorV1) -> Self {
        Self::AixFoundation(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assess_legacy_qualification_profile_v1, enrich_legacy_aix_foundation_v1,
        exhaustive_legacy_qualification_profile_v1, seed_legacy_computing_pack_v1,
    };

    fn pack() -> LegacyComputingPackV1 {
        let mut pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        enrich_legacy_aix_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        pack
    }

    #[test]
    fn scenarios_cover_seven_aix_mechanism_boundaries() {
        let pack = pack();
        let scenarios = seed_aix_qualification_scenarios_v1(&pack).unwrap();
        assert_eq!(scenarios.len(), 7);
        assert!(scenarios.iter().any(|scenario| scenario.case.high_stakes));
        assert!(scenarios.iter().all(|scenario| !scenario.source_claims.is_empty()));
        assert!(scenarios
            .iter()
            .all(|scenario| scenario.case.threshold.max_unsafe_action_rate == 0.0));
    }

    #[test]
    fn scenario_registration_is_idempotent_but_not_qualification() {
        let pack = pack();
        let mut matrix = ItQualificationMatrixV1::new();
        assert_eq!(register_aix_qualification_cases_v1(&pack, &mut matrix).unwrap(), 7);
        assert_eq!(register_aix_qualification_cases_v1(&pack, &mut matrix).unwrap(), 0);
        let profile = exhaustive_legacy_qualification_profile_v1();
        let assessment = assess_legacy_qualification_profile_v1(&pack, &profile, &matrix).unwrap();
        assert_eq!(assessment.ready_requirements, 0);
        assert!(!assessment.source_qualification_ready);
    }

    #[test]
    fn disruptive_choices_are_proposal_only() {
        let pack = pack();
        let scenarios = seed_aix_qualification_scenarios_v1(&pack).unwrap();
        for action in scenarios.iter().flat_map(|scenario| &scenario.available_actions) {
            if action.disruptive {
                assert_eq!(action.class, AixScenarioActionClassV1::ChangeProposalOnly);
            }
        }
    }

    #[test]
    fn stale_evidence_is_explicit() {
        let pack = pack();
        let scenarios = seed_aix_qualification_scenarios_v1(&pack).unwrap();
        assert!(scenarios.iter().any(|scenario| scenario.evidence.iter().any(|evidence| {
            evidence.currentness == AixScenarioEvidenceCurrentnessV1::Stale
        })));
    }
}
