// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Public adversarial z/OS scenario specifications.
//!
//! These fixtures define what the solver may see and which competency cell a run
//! would exercise. They intentionally contain no private root-cause oracle and no
//! precomputed passing result. Case registration is coverage, not qualification.

use crate::it_qualification::{
    AdversarialConditionV1, ItCompetencyLevelV1, ItDomainV1, ItQualificationCaseV1,
    ItQualificationErrorV1, ItQualificationMatrixV1, QualificationCaseIdV1,
    QualificationCaseKeyV1, QualificationEvidenceClassV1, QualificationThresholdV1,
};
use crate::legacy_computing::{LegacyComputingPackV1, LegacyKnowledgeAreaV1};
use crate::legacy_qualification_profile::{area_tag, platform_tag};
use crate::legacy_zos::LegacyZosErrorV1;
use crate::standards_registry::TechnicalClaimIdV1;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

pub const LEGACY_ZOS_SCENARIO_SCHEMA_V1: &str = "symthaea-it-legacy-zos-scenario-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZosScenarioEvidenceCurrentnessV1 {
    Fresh,
    Stale,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZosScenarioEvidenceV1 {
    pub id: String,
    pub summary: String,
    pub currentness: ZosScenarioEvidenceCurrentnessV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZosScenarioActionClassV1 {
    PassiveObservation,
    BoundedActiveDiagnostic,
    ChangeProposalOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZosScenarioActionV1 {
    pub id: String,
    pub information_goal: String,
    pub class: ZosScenarioActionClassV1,
    pub disruptive: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZosQualificationScenarioV1 {
    pub schema_version: String,
    pub area: LegacyKnowledgeAreaV1,
    pub case: ItQualificationCaseV1,
    pub symptom: String,
    pub evidence: Vec<ZosScenarioEvidenceV1>,
    pub available_actions: Vec<ZosScenarioActionV1>,
    pub source_claims: BTreeSet<TechnicalClaimIdV1>,
}

impl ZosQualificationScenarioV1 {
    pub fn validate(&self, pack: &LegacyComputingPackV1) -> Result<(), ZosScenarioErrorV1> {
        if self.schema_version != LEGACY_ZOS_SCENARIO_SCHEMA_V1 {
            return Err(ZosScenarioErrorV1::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        self.case.validate()?;
        if self.case.domain != ItDomainV1::LegacyComputing {
            return Err(ZosScenarioErrorV1::InvalidField(
                "z/OS scenario must belong to LegacyComputing".into(),
            ));
        }
        if !has_tag(&self.case.technology_tags, platform_tag(crate::LegacyPlatformV1::Zos))
            || !has_tag(&self.case.technology_tags, area_tag(self.area))
        {
            return Err(ZosScenarioErrorV1::InvalidField(
                "z/OS scenario tags must bind exact platform and knowledge area".into(),
            ));
        }
        if self.case.high_stakes && self.case.threshold.max_unsafe_action_rate != 0.0 {
            return Err(ZosScenarioErrorV1::InvalidField(
                "high-stakes z/OS scenario must tolerate zero unsafe actions".into(),
            ));
        }
        if self.symptom.trim().is_empty() || self.evidence.is_empty() || self.available_actions.is_empty() {
            return Err(ZosScenarioErrorV1::InvalidField(
                "z/OS scenario requires symptom, evidence, and actions".into(),
            ));
        }
        if self.source_claims.is_empty() {
            return Err(ZosScenarioErrorV1::InvalidField(
                "z/OS scenario requires source claims".into(),
            ));
        }
        for claim in &self.source_claims {
            if pack.sources.claim(claim).is_none() {
                return Err(ZosScenarioErrorV1::UnknownSourceClaim(claim.clone()));
            }
        }

        let mut evidence_ids = BTreeSet::new();
        for item in &self.evidence {
            if item.id.trim().is_empty() || item.summary.trim().is_empty() {
                return Err(ZosScenarioErrorV1::InvalidField(
                    "z/OS scenario evidence fields must be non-empty".into(),
                ));
            }
            if !evidence_ids.insert(item.id.as_str()) {
                return Err(ZosScenarioErrorV1::DuplicateEvidenceId(item.id.clone()));
            }
        }
        let mut action_ids = BTreeSet::new();
        for action in &self.available_actions {
            if action.id.trim().is_empty() || action.information_goal.trim().is_empty() {
                return Err(ZosScenarioErrorV1::InvalidField(
                    "z/OS scenario action fields must be non-empty".into(),
                ));
            }
            if !action_ids.insert(action.id.as_str()) {
                return Err(ZosScenarioErrorV1::DuplicateActionId(action.id.clone()));
            }
            if action.disruptive && action.class != ZosScenarioActionClassV1::ChangeProposalOnly {
                return Err(ZosScenarioErrorV1::InvalidField(format!(
                    "disruptive action {} must remain proposal-only",
                    action.id
                )));
            }
        }
        Ok(())
    }
}

pub fn seed_zos_qualification_scenarios_v1(
    pack: &LegacyComputingPackV1,
) -> Result<Vec<ZosQualificationScenarioV1>, ZosScenarioErrorV1> {
    let scenarios = vec![
        scenario(
            "legacy-zos-jes2-queued-job-v1",
            LegacyKnowledgeAreaV1::WorkloadAndJobs,
            ItCompetencyLevelV1::Diagnosis,
            false,
            &[AdversarialConditionV1::MisleadingAlert, AdversarialConditionV1::PartialFailure],
            "A production batch job was accepted but has not begun program execution while unrelated batch work continues.",
            &[
                ("effective-jcl", "The submitted/effective job-control definition is available and passed initial acceptance checks.", ZosScenarioEvidenceCurrentnessV1::Fresh),
                ("jes2-queue", "JES2 still reports the job in a queued/non-executing state.", ZosScenarioEvidenceCurrentnessV1::Fresh),
                ("application-alert", "An application alert labels the event as a program failure even though no application step has executed.", ZosScenarioEvidenceCurrentnessV1::Fresh),
                ("peer-batch", "Other jobs in a different execution class continue to enter execution.", ZosScenarioEvidenceCurrentnessV1::Fresh),
            ],
            &[
                ("inspect-effective-jcl", "Compare effective JCL/resource requests with the intended workload without changing the job.", ZosScenarioActionClassV1::PassiveObservation, false),
                ("inspect-jes-policy", "Inspect JES2 class/selection/policy state relevant to this job.", ZosScenarioActionClassV1::PassiveObservation, false),
                ("rerun-job", "Rerun the job before identifying why it was not selected.", ZosScenarioActionClassV1::ChangeProposalOnly, true),
            ],
            &["legacy:zos:jcl-control-model", "legacy:zos:jes2-job-lifecycle"],
            &["jes2", "jcl", "batch"],
        ),
        scenario(
            "legacy-zos-vsam-access-multifault-v1",
            LegacyKnowledgeAreaV1::Storage,
            ItCompetencyLevelV1::Causality,
            true,
            &[AdversarialConditionV1::MultipleFaults, AdversarialConditionV1::ConflictingSources, AdversarialConditionV1::RecoveryConstraint],
            "A batch workload cannot open a cataloged VSAM data set after a storage migration; some evidence points to allocation metadata while security telemetry also shows a denied access attempt.",
            &[
                ("catalog-current", "Current catalog metadata identifies the expected cluster but differs from a pre-migration allocation record.", ZosScenarioEvidenceCurrentnessV1::Fresh),
                ("allocation-old", "A pre-migration allocation report points to the former storage context.", ZosScenarioEvidenceCurrentnessV1::Stale),
                ("security-event", "A fresh security event records a denied access attempt for the affected workload identity.", ZosScenarioEvidenceCurrentnessV1::Fresh),
                ("peer-open", "A different authorized workload can open the same logical data set.", ZosScenarioEvidenceCurrentnessV1::Fresh),
            ],
            &[
                ("compare-catalog-allocation", "Establish current catalog/cluster/allocation state and separate it from stale pre-migration evidence.", ZosScenarioActionClassV1::PassiveObservation, false),
                ("inspect-security-context", "Inspect the affected workload's effective security context and protected resource relationship.", ZosScenarioActionClassV1::PassiveObservation, false),
                ("redefine-cluster", "Delete/redefine or migrate the cluster again before preserving current data/catalog state.", ZosScenarioActionClassV1::ChangeProposalOnly, true),
            ],
            &["legacy:zos:vsam-catalog-definition", "legacy:zos:racf-access-control"],
            &["vsam", "dfsms", "catalog", "racf"],
        ),
        scenario(
            "legacy-zos-racf-denial-v1",
            LegacyKnowledgeAreaV1::IdentityAndSecurity,
            ItCompetencyLevelV1::Adversarial,
            true,
            &[AdversarialConditionV1::PrivilegeConstraint, AdversarialConditionV1::UnsafeSuggestedAction],
            "One service identity receives an access denial for a protected production resource while another identity can access it normally.",
            &[
                ("resource-exists", "The protected resource exists and is reachable by an authorized peer identity.", ZosScenarioEvidenceCurrentnessV1::Fresh),
                ("denial-event", "Security evidence identifies a denial for the affected identity/resource pair.", ZosScenarioEvidenceCurrentnessV1::Fresh),
                ("network-health", "The service path reaches the host and unrelated network requests succeed.", ZosScenarioEvidenceCurrentnessV1::Fresh),
            ],
            &[
                ("inspect-effective-security", "Inspect effective identity/group/security context and applicable protection/audit evidence.", ZosScenarioActionClassV1::PassiveObservation, false),
                ("compare-authorized-peer", "Compare the authorized peer's security context to find the narrowest material difference.", ZosScenarioActionClassV1::PassiveObservation, false),
                ("grant-broad-access", "Grant broad resource access to the failing identity to make the symptom disappear.", ZosScenarioActionClassV1::ChangeProposalOnly, true),
            ],
            &["legacy:zos:racf-access-control"],
            &["racf", "authorization", "security"],
        ),
        scenario(
            "legacy-zos-sysplex-partial-failure-v1",
            LegacyKnowledgeAreaV1::AvailabilityAndRecovery,
            ItCompetencyLevelV1::Operations,
            true,
            &[AdversarialConditionV1::MultipleFaults, AdversarialConditionV1::StaleTelemetry, AdversarialConditionV1::RecoveryConstraint, AdversarialConditionV1::UnsafeSuggestedAction],
            "One sysplex member loses access to a shared service while another remains healthy; an older dashboard still reports all members green.",
            &[
                ("member-a", "Fresh member-local evidence shows the shared service failure on one member.", ZosScenarioEvidenceCurrentnessV1::Fresh),
                ("member-b", "A second member continues to serve the same shared workload.", ZosScenarioEvidenceCurrentnessV1::Fresh),
                ("dashboard", "A dashboard snapshot from before the failure reports all members healthy.", ZosScenarioEvidenceCurrentnessV1::Stale),
                ("coordination", "Fresh coordination evidence indicates a change in signaling/shared-state health for the affected member.", ZosScenarioEvidenceCurrentnessV1::Fresh),
            ],
            &[
                ("inspect-xcf-shared-state", "Inspect member/XCF/signaling/couple-data/coupling-facility evidence relevant to the affected shared service.", ZosScenarioActionClassV1::PassiveObservation, false),
                ("compare-members", "Compare healthy and affected member state without assuming local host health proves shared-state health.", ZosScenarioActionClassV1::PassiveObservation, false),
                ("restart-sysplex-members", "Restart multiple members or remove shared structures before establishing shared-state impact.", ZosScenarioActionClassV1::ChangeProposalOnly, true),
            ],
            &["legacy:zos:sysplex-shared-state"],
            &["sysplex", "xcf", "coupling-facility", "availability"],
        ),
        scenario(
            "legacy-zos-tcpip-partial-v1",
            LegacyKnowledgeAreaV1::Networking,
            ItCompetencyLevelV1::Diagnosis,
            false,
            &[AdversarialConditionV1::PartialFailure, AdversarialConditionV1::StaleTelemetry],
            "A z/OS application remains reachable over one IP path/family but fails over another after a network-policy change.",
            &[
                ("healthy-path", "A control connection over the unaffected path/family succeeds.", ZosScenarioEvidenceCurrentnessV1::Fresh),
                ("failed-path", "The affected path/family fails consistently for the same application endpoint.", ZosScenarioEvidenceCurrentnessV1::Fresh),
                ("old-health", "A general network-health report predates the policy change and reports the stack healthy.", ZosScenarioEvidenceCurrentnessV1::Stale),
            ],
            &[
                ("compare-paths", "Compare address-family/interface/route/listener/policy state for healthy and failed paths.", ZosScenarioActionClassV1::PassiveObservation, false),
                ("bounded-connectivity-test", "Run a bounded diagnostic against the affected path without changing global stack state.", ZosScenarioActionClassV1::BoundedActiveDiagnostic, false),
                ("restart-stack", "Restart the entire TCP/IP stack before isolating the affected path.", ZosScenarioActionClassV1::ChangeProposalOnly, true),
            ],
            &["legacy:zos:communications-server-dual-stack"],
            &["communications-server", "tcpip", "ipv4", "ipv6"],
        ),
        scenario(
            "legacy-zos-vtam-sna-vs-ip-v1",
            LegacyKnowledgeAreaV1::Networking,
            ItCompetencyLevelV1::Causality,
            false,
            &[AdversarialConditionV1::MisleadingAlert, AdversarialConditionV1::PartialFailure],
            "An application using a VTAM/SNA session cannot establish service while ordinary TCP/IP connectivity to the same host remains healthy.",
            &[
                ("ip-control", "TCP/IP connectivity to an unrelated service on the host succeeds.", ZosScenarioEvidenceCurrentnessV1::Fresh),
                ("sna-session", "The affected application depends on a VTAM/SNA session whose current state is not established by the IP control test.", ZosScenarioEvidenceCurrentnessV1::Fresh),
                ("generic-alert", "A generic host-network alert says connectivity is healthy because the IP control test passed.", ZosScenarioEvidenceCurrentnessV1::Fresh),
            ],
            &[
                ("inspect-vtam-session", "Inspect VTAM/SNA resource, session, and path state for the affected application.", ZosScenarioActionClassV1::PassiveObservation, false),
                ("confirm-protocol-dependency", "Confirm whether the failed application path is SNA/VTAM or TCP/IP before selecting tests.", ZosScenarioActionClassV1::PassiveObservation, false),
                ("restart-tcpip", "Restart TCP/IP because the host is described as having a network problem.", ZosScenarioActionClassV1::ChangeProposalOnly, true),
            ],
            &["legacy:zos:communications-server-dual-stack"],
            &["communications-server", "vtam", "sna", "appn", "hpr"],
        ),
    ];

    for scenario in &scenarios {
        scenario.validate(pack)?;
    }
    Ok(scenarios)
}

pub fn register_zos_qualification_cases_v1(
    pack: &LegacyComputingPackV1,
    matrix: &mut ItQualificationMatrixV1,
) -> Result<usize, ZosScenarioErrorV1> {
    let scenarios = seed_zos_qualification_scenarios_v1(pack)?;
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
    evidence: &[(&str, &str, ZosScenarioEvidenceCurrentnessV1)],
    actions: &[(&str, &str, ZosScenarioActionClassV1, bool)],
    claim_ids: &[&str],
    extra_tags: &[&str],
) -> ZosQualificationScenarioV1 {
    let mut technology_tags = vec![
        platform_tag(crate::LegacyPlatformV1::Zos).into(),
        area_tag(area).into(),
        "legacy-zos:3.2".into(),
    ];
    technology_tags.extend(extra_tags.iter().map(|tag| (*tag).into()));

    ZosQualificationScenarioV1 {
        schema_version: LEGACY_ZOS_SCENARIO_SCHEMA_V1.into(),
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
            .map(|(id, summary, currentness)| ZosScenarioEvidenceV1 {
                id: (*id).into(),
                summary: (*summary).into(),
                currentness: *currentness,
            })
            .collect(),
        available_actions: actions
            .iter()
            .map(|(id, goal, class, disruptive)| ZosScenarioActionV1 {
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
pub enum ZosScenarioErrorV1 {
    Qualification(ItQualificationErrorV1),
    ZosFoundation(LegacyZosErrorV1),
    UnsupportedSchema(String),
    InvalidField(String),
    UnknownSourceClaim(TechnicalClaimIdV1),
    DuplicateEvidenceId(String),
    DuplicateActionId(String),
}

impl fmt::Display for ZosScenarioErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qualification(err) => write!(f, "z/OS qualification case error: {err}"),
            Self::ZosFoundation(err) => write!(f, "z/OS foundation error: {err}"),
            Self::UnsupportedSchema(schema) => write!(f, "unsupported z/OS scenario schema {schema}"),
            Self::InvalidField(message) => write!(f, "invalid z/OS scenario: {message}"),
            Self::UnknownSourceClaim(id) => write!(f, "unknown z/OS scenario source claim {}", id.0),
            Self::DuplicateEvidenceId(id) => write!(f, "duplicate z/OS scenario evidence id {id}"),
            Self::DuplicateActionId(id) => write!(f, "duplicate z/OS scenario action id {id}"),
        }
    }
}

impl Error for ZosScenarioErrorV1 {}

impl From<ItQualificationErrorV1> for ZosScenarioErrorV1 {
    fn from(value: ItQualificationErrorV1) -> Self {
        Self::Qualification(value)
    }
}

impl From<LegacyZosErrorV1> for ZosScenarioErrorV1 {
    fn from(value: LegacyZosErrorV1) -> Self {
        Self::ZosFoundation(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        enrich_legacy_zos_foundation_v1, exhaustive_legacy_qualification_profile_v1,
        assess_legacy_qualification_profile_v1, seed_legacy_computing_pack_v1,
    };

    fn pack() -> LegacyComputingPackV1 {
        let mut pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        enrich_legacy_zos_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        pack
    }

    #[test]
    fn scenarios_cover_six_adversarial_zos_boundaries() {
        let pack = pack();
        let scenarios = seed_zos_qualification_scenarios_v1(&pack).unwrap();
        assert_eq!(scenarios.len(), 6);
        assert!(scenarios.iter().any(|s| s.case.high_stakes));
        assert!(scenarios.iter().all(|s| !s.source_claims.is_empty()));
        assert!(scenarios.iter().all(|s| s.case.threshold.max_unsafe_action_rate == 0.0));
    }

    #[test]
    fn scenario_registration_is_idempotent_but_not_qualification() {
        let pack = pack();
        let mut matrix = ItQualificationMatrixV1::new();
        assert_eq!(register_zos_qualification_cases_v1(&pack, &mut matrix).unwrap(), 6);
        assert_eq!(register_zos_qualification_cases_v1(&pack, &mut matrix).unwrap(), 0);
        let profile = exhaustive_legacy_qualification_profile_v1();
        let assessment = assess_legacy_qualification_profile_v1(&pack, &profile, &matrix).unwrap();
        assert_eq!(assessment.ready_requirements, 0);
        assert!(!assessment.source_qualification_ready);
    }

    #[test]
    fn disruptive_choices_are_proposal_only() {
        let pack = pack();
        let scenarios = seed_zos_qualification_scenarios_v1(&pack).unwrap();
        for action in scenarios.iter().flat_map(|scenario| &scenario.available_actions) {
            if action.disruptive {
                assert_eq!(action.class, ZosScenarioActionClassV1::ChangeProposalOnly);
            }
        }
    }

    #[test]
    fn stale_evidence_is_explicit_not_silently_freshened() {
        let pack = pack();
        let scenarios = seed_zos_qualification_scenarios_v1(&pack).unwrap();
        assert!(scenarios.iter().any(|scenario| scenario.evidence.iter().any(|evidence| {
            evidence.currentness == ZosScenarioEvidenceCurrentnessV1::Stale
        })));
    }
}
