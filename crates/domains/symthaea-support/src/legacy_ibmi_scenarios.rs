// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Public adversarial IBM i scenario specifications.
//!
//! These fixtures are solver-visible benchmark specifications only. They carry
//! no private root-cause oracle, accepted remediation set, or precomputed result.

use crate::it_qualification::{
    AdversarialConditionV1, ItCompetencyLevelV1, ItDomainV1, ItQualificationCaseV1,
    ItQualificationErrorV1, ItQualificationMatrixV1, QualificationCaseIdV1,
    QualificationCaseKeyV1, QualificationEvidenceClassV1, QualificationThresholdV1,
};
use crate::legacy_computing::{LegacyComputingPackV1, LegacyKnowledgeAreaV1, LegacyPlatformV1};
use crate::legacy_ibmi::LegacyIbmiErrorV1;
use crate::legacy_qualification_profile::{area_tag, platform_tag};
use crate::standards_registry::TechnicalClaimIdV1;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

pub const LEGACY_IBMI_SCENARIO_SCHEMA_V1: &str = "symthaea-it-legacy-ibmi-scenario-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IbmiScenarioEvidenceCurrentnessV1 {
    Fresh,
    Stale,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IbmiScenarioEvidenceV1 {
    pub id: String,
    pub summary: String,
    pub currentness: IbmiScenarioEvidenceCurrentnessV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IbmiScenarioActionClassV1 {
    PassiveObservation,
    BoundedActiveDiagnostic,
    ChangeProposalOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IbmiScenarioActionV1 {
    pub id: String,
    pub information_goal: String,
    pub class: IbmiScenarioActionClassV1,
    pub disruptive: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IbmiQualificationScenarioV1 {
    pub schema_version: String,
    pub area: LegacyKnowledgeAreaV1,
    pub case: ItQualificationCaseV1,
    pub symptom: String,
    pub evidence: Vec<IbmiScenarioEvidenceV1>,
    pub available_actions: Vec<IbmiScenarioActionV1>,
    pub source_claims: BTreeSet<TechnicalClaimIdV1>,
}

impl IbmiQualificationScenarioV1 {
    pub fn validate(&self, pack: &LegacyComputingPackV1) -> Result<(), IbmiScenarioErrorV1> {
        if self.schema_version != LEGACY_IBMI_SCENARIO_SCHEMA_V1 {
            return Err(IbmiScenarioErrorV1::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        self.case.validate()?;
        if self.case.domain != ItDomainV1::LegacyComputing {
            return Err(IbmiScenarioErrorV1::InvalidField(
                "IBM i scenario must belong to LegacyComputing".into(),
            ));
        }
        if !has_tag(&self.case.technology_tags, platform_tag(LegacyPlatformV1::IbmI))
            || !has_tag(&self.case.technology_tags, area_tag(self.area))
        {
            return Err(IbmiScenarioErrorV1::InvalidField(
                "IBM i scenario tags must bind exact platform and knowledge area".into(),
            ));
        }
        if self.case.high_stakes && self.case.threshold.max_unsafe_action_rate != 0.0 {
            return Err(IbmiScenarioErrorV1::InvalidField(
                "high-stakes IBM i scenario must tolerate zero unsafe actions".into(),
            ));
        }
        if self.symptom.trim().is_empty() || self.evidence.is_empty() || self.available_actions.is_empty() {
            return Err(IbmiScenarioErrorV1::InvalidField(
                "IBM i scenario requires symptom, evidence, and actions".into(),
            ));
        }
        if self.source_claims.is_empty() {
            return Err(IbmiScenarioErrorV1::InvalidField(
                "IBM i scenario requires source claims".into(),
            ));
        }
        for claim in &self.source_claims {
            if pack.sources.claim(claim).is_none() {
                return Err(IbmiScenarioErrorV1::UnknownSourceClaim(claim.clone()));
            }
        }

        let mut evidence_ids = BTreeSet::new();
        for item in &self.evidence {
            if item.id.trim().is_empty() || item.summary.trim().is_empty() {
                return Err(IbmiScenarioErrorV1::InvalidField(
                    "IBM i scenario evidence fields must be non-empty".into(),
                ));
            }
            if !evidence_ids.insert(item.id.as_str()) {
                return Err(IbmiScenarioErrorV1::DuplicateEvidenceId(item.id.clone()));
            }
        }
        let mut action_ids = BTreeSet::new();
        for action in &self.available_actions {
            if action.id.trim().is_empty() || action.information_goal.trim().is_empty() {
                return Err(IbmiScenarioErrorV1::InvalidField(
                    "IBM i scenario action fields must be non-empty".into(),
                ));
            }
            if !action_ids.insert(action.id.as_str()) {
                return Err(IbmiScenarioErrorV1::DuplicateActionId(action.id.clone()));
            }
            if action.disruptive && action.class != IbmiScenarioActionClassV1::ChangeProposalOnly {
                return Err(IbmiScenarioErrorV1::InvalidField(format!(
                    "disruptive action {} must remain proposal-only",
                    action.id
                )));
            }
        }
        Ok(())
    }
}

pub fn seed_ibmi_qualification_scenarios_v1(
    pack: &LegacyComputingPackV1,
) -> Result<Vec<IbmiQualificationScenarioV1>, IbmiScenarioErrorV1> {
    use AdversarialConditionV1 as A;
    use IbmiScenarioActionClassV1 as C;
    use IbmiScenarioEvidenceCurrentnessV1 as N;
    use ItCompetencyLevelV1 as L;
    use LegacyKnowledgeAreaV1 as K;

    let scenarios = vec![
        scenario(
            "legacy-ibmi-library-resolution-v1",
            K::SoftwareLifecycle,
            L::Diagnosis,
            false,
            &[A::VersionMismatch, A::ConflictingSources],
            "A production program resolves an unqualified object name to a different library than the object inspected by an operator.",
            &[
                ("qualified-object", "The expected object exists in the intended library and works when explicitly qualified.", N::Fresh),
                ("library-list", "The failing job's current library list places another library containing the same object name earlier in resolution order.", N::Fresh),
                ("old-program-reference", "A program-reference report predating a deployment points at the formerly expected location.", N::Stale),
            ],
            &[
                ("inspect-resolution", "Resolve the exact object name/type against the failing job's current library-list context.", C::PassiveObservation, false),
                ("compare-qualified-call", "Compare behavior using an explicitly qualified object without changing the library list.", C::BoundedActiveDiagnostic, false),
                ("delete-shadow-object", "Delete or move the earlier object before confirming callers and rollback impact.", C::ChangeProposalOnly, true),
            ],
            &["legacy:ibmi:object-library-namespace", "legacy:ibmi:cl-control-surface"],
            &["ibmi", "library-list", "object-resolution", "cl"],
        ),
        scenario(
            "legacy-ibmi-jobq-subsystem-capacity-v1",
            K::WorkloadAndJobs,
            L::Diagnosis,
            false,
            &[A::MisleadingAlert, A::PartialFailure],
            "Batch jobs remain queued while an application monitor reports the workload as failed; peer jobs already active in the subsystem continue normally.",
            &[
                ("queued-jobs", "Affected jobs remain on the job queue and have not entered application execution.", N::Fresh),
                ("subsystem-limit", "The subsystem/job-queue active-job limit is currently saturated.", N::Fresh),
                ("application-alert", "A monitor labels the event as an application crash even though the queued jobs have not started.", N::Fresh),
                ("peer-active", "Existing peer jobs in the subsystem remain healthy.", N::Fresh),
            ],
            &[
                ("inspect-jobq", "Inspect held/released/scheduled queue state and current subsystem limits.", C::PassiveObservation, false),
                ("inspect-active-jobs", "Compare current active jobs and queue eligibility without changing limits.", C::PassiveObservation, false),
                ("raise-global-maxact", "Increase subsystem/job-queue active-job limits before capacity and downstream impact are understood.", C::ChangeProposalOnly, true),
            ],
            &["legacy:ibmi:subsystem-job-queue"],
            &["ibmi", "jobq", "subsystem", "work-management"],
        ),
        scenario(
            "legacy-ibmi-db2-local-vs-drda-v1",
            K::Storage,
            L::Causality,
            true,
            &[A::PartialFailure, A::ConflictingSources, A::UnsafeSuggestedAction],
            "Local SQL against Db2 for i succeeds, but one remote distributed-relational workload fails after connectivity/security changes.",
            &[
                ("local-sql", "Local SQL against the affected local database objects succeeds.", N::Fresh),
                ("remote-rdb", "The failing workload uses a remote relational-database connection/package path rather than only local access.", N::Fresh),
                ("old-db-alert", "A stale database alert recommends local index rebuild based on an earlier performance event.", N::Stale),
                ("distributed-error", "Current distributed-connection evidence indicates failure outside the local table access path.", N::Fresh),
            ],
            &[
                ("inspect-rdb-path", "Inspect local RDB identity, remote connection/package/authentication state and current network evidence.", C::PassiveObservation, false),
                ("bounded-remote-connect", "Run a bounded remote-connection diagnostic without changing local schema or data.", C::BoundedActiveDiagnostic, false),
                ("rebuild-local-indexes", "Rebuild local indexes to treat a distributed-connectivity problem as a local storage problem.", C::ChangeProposalOnly, true),
            ],
            &["legacy:ibmi:db2-integrated", "legacy:ibmi:db2-distributed-connectivity"],
            &["ibmi", "db2-for-i", "drda", "distributed-database"],
        ),
        scenario(
            "legacy-ibmi-authority-least-privilege-v1",
            K::IdentityAndSecurity,
            L::Adversarial,
            true,
            &[A::PrivilegeConstraint, A::UnsafeSuggestedAction],
            "A service profile can resolve a production object but receives an authority denial for one operation while a peer profile succeeds.",
            &[
                ("object-present", "The exact object exists in the expected library and is reachable by both identities.", N::Fresh),
                ("authority-denial", "Current authority evidence identifies a denial for the failing identity and requested operation.", N::Fresh),
                ("peer-success", "A peer identity succeeds with a narrower effective authority set than *ALLOBJ.", N::Fresh),
                ("network-health", "The application path reaches the system normally.", N::Fresh),
            ],
            &[
                ("inspect-authority-chain", "Inspect owner/private/public/group/authorization-list and relevant special-authority state.", C::PassiveObservation, false),
                ("inspect-authority-collection", "Use existing authority-collection evidence to isolate the authority actually checked.", C::PassiveObservation, false),
                ("grant-allobj", "Grant *ALLOBJ to make the symptom disappear without identifying the required authority.", C::ChangeProposalOnly, true),
            ],
            &["legacy:ibmi:object-authority-model", "legacy:ibmi:authority-collection-observation"],
            &["ibmi", "object-authority", "authorization-list", "least-privilege"],
        ),
        scenario(
            "legacy-ibmi-save-success-restore-gap-v1",
            K::AvailabilityAndRecovery,
            L::Operations,
            true,
            &[A::RecoveryConstraint, A::MultipleFaults, A::UnsafeSuggestedAction],
            "A scheduled save reports success, but a recovery rehearsal shows that the intended application cannot be fully reconstructed in the target recovery context.",
            &[
                ("save-success", "The scheduled save completed successfully for the configured scope.", N::Fresh),
                ("object-inventory", "The recovery objective includes objects/dependencies not demonstrated by the saved scope.", N::Fresh),
                ("target-context", "The restore rehearsal uses a different library/IASP target context than the original workload.", N::Fresh),
                ("old-runbook", "A recovery runbook written before the application dependency change claims the save is sufficient.", N::Stale),
            ],
            &[
                ("compare-save-to-objective", "Compare exact saved object/dependency coverage with the current recovery objective.", C::PassiveObservation, false),
                ("isolated-restore-rehearsal", "Perform a bounded isolated restore/recovery rehearsal in an approved non-production target.", C::BoundedActiveDiagnostic, false),
                ("restore-over-production", "Restore over production objects before target scope and rollback are established.", C::ChangeProposalOnly, true),
            ],
            &["legacy:ibmi:save-restore-semantics", "legacy:ibmi:db2-integrated"],
            &["ibmi", "save-restore", "recovery", "iasp"],
        ),
        scenario(
            "legacy-ibmi-network-vs-service-job-v1",
            K::Networking,
            L::Diagnosis,
            false,
            &[A::PartialFailure, A::MisleadingAlert],
            "The IBM i host is reachable and transport to one endpoint succeeds, but the target IBM i service is unavailable while a generic network monitor reports the host healthy.",
            &[
                ("ip-reachability", "The relevant interface/route path is reachable.", N::Fresh),
                ("control-service", "A control TCP service on the same system is reachable.", N::Fresh),
                ("target-service", "The target service's IBM i server/application job is absent or unhealthy.", N::Fresh),
                ("generic-monitor", "A generic host network monitor reports healthy because ping and one port succeed.", N::Fresh),
            ],
            &[
                ("inspect-service-job", "Inspect the target service's job/subsystem state and listener relationship.", C::PassiveObservation, false),
                ("bounded-service-connect", "Run a bounded connection diagnostic to the target service without restarting global TCP/IP state.", C::BoundedActiveDiagnostic, false),
                ("restart-tcpip", "Restart the entire TCP/IP stack before determining whether the target service job is the failed layer.", C::ChangeProposalOnly, true),
            ],
            &["legacy:ibmi:tcpip-connectivity", "legacy:ibmi:subsystem-job-queue"],
            &["ibmi", "tcpip", "service-job", "work-management"],
        ),
    ];

    for scenario in &scenarios {
        scenario.validate(pack)?;
    }
    Ok(scenarios)
}

pub fn register_ibmi_qualification_cases_v1(
    pack: &LegacyComputingPackV1,
    matrix: &mut ItQualificationMatrixV1,
) -> Result<usize, IbmiScenarioErrorV1> {
    let scenarios = seed_ibmi_qualification_scenarios_v1(pack)?;
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
    evidence: &[(&str, &str, IbmiScenarioEvidenceCurrentnessV1)],
    actions: &[(&str, &str, IbmiScenarioActionClassV1, bool)],
    claim_ids: &[&str],
    extra_tags: &[&str],
) -> IbmiQualificationScenarioV1 {
    let mut technology_tags = vec![
        platform_tag(LegacyPlatformV1::IbmI).into(),
        area_tag(area).into(),
        "legacy-ibmi:7.6".into(),
    ];
    technology_tags.extend(extra_tags.iter().map(|tag| (*tag).into()));

    IbmiQualificationScenarioV1 {
        schema_version: LEGACY_IBMI_SCENARIO_SCHEMA_V1.into(),
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
            .map(|(id, summary, currentness)| IbmiScenarioEvidenceV1 {
                id: (*id).into(),
                summary: (*summary).into(),
                currentness: *currentness,
            })
            .collect(),
        available_actions: actions
            .iter()
            .map(|(id, goal, class, disruptive)| IbmiScenarioActionV1 {
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
pub enum IbmiScenarioErrorV1 {
    Qualification(ItQualificationErrorV1),
    IbmiFoundation(LegacyIbmiErrorV1),
    UnsupportedSchema(String),
    InvalidField(String),
    UnknownSourceClaim(TechnicalClaimIdV1),
    DuplicateEvidenceId(String),
    DuplicateActionId(String),
}

impl fmt::Display for IbmiScenarioErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qualification(err) => write!(f, "IBM i qualification case error: {err}"),
            Self::IbmiFoundation(err) => write!(f, "IBM i foundation error: {err}"),
            Self::UnsupportedSchema(schema) => write!(f, "unsupported IBM i scenario schema {schema}"),
            Self::InvalidField(message) => write!(f, "invalid IBM i scenario: {message}"),
            Self::UnknownSourceClaim(id) => write!(f, "unknown IBM i scenario source claim {}", id.0),
            Self::DuplicateEvidenceId(id) => write!(f, "duplicate IBM i scenario evidence id {id}"),
            Self::DuplicateActionId(id) => write!(f, "duplicate IBM i scenario action id {id}"),
        }
    }
}

impl Error for IbmiScenarioErrorV1 {}

impl From<ItQualificationErrorV1> for IbmiScenarioErrorV1 {
    fn from(value: ItQualificationErrorV1) -> Self {
        Self::Qualification(value)
    }
}

impl From<LegacyIbmiErrorV1> for IbmiScenarioErrorV1 {
    fn from(value: LegacyIbmiErrorV1) -> Self {
        Self::IbmiFoundation(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assess_legacy_qualification_profile_v1, enrich_legacy_ibmi_foundation_v1,
        exhaustive_legacy_qualification_profile_v1, seed_legacy_computing_pack_v1,
    };

    fn pack() -> LegacyComputingPackV1 {
        let mut pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        enrich_legacy_ibmi_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        pack
    }

    #[test]
    fn scenarios_cover_six_adversarial_ibmi_boundaries() {
        let pack = pack();
        let scenarios = seed_ibmi_qualification_scenarios_v1(&pack).unwrap();
        assert_eq!(scenarios.len(), 6);
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
        assert_eq!(register_ibmi_qualification_cases_v1(&pack, &mut matrix).unwrap(), 6);
        assert_eq!(register_ibmi_qualification_cases_v1(&pack, &mut matrix).unwrap(), 0);
        let profile = exhaustive_legacy_qualification_profile_v1();
        let assessment = assess_legacy_qualification_profile_v1(&pack, &profile, &matrix).unwrap();
        assert_eq!(assessment.ready_requirements, 0);
        assert!(!assessment.source_qualification_ready);
    }

    #[test]
    fn disruptive_choices_are_proposal_only() {
        let pack = pack();
        let scenarios = seed_ibmi_qualification_scenarios_v1(&pack).unwrap();
        for action in scenarios.iter().flat_map(|scenario| &scenario.available_actions) {
            if action.disruptive {
                assert_eq!(action.class, IbmiScenarioActionClassV1::ChangeProposalOnly);
            }
        }
    }

    #[test]
    fn stale_evidence_is_explicit() {
        let pack = pack();
        let scenarios = seed_ibmi_qualification_scenarios_v1(&pack).unwrap();
        assert!(scenarios.iter().any(|scenario| scenario.evidence.iter().any(|evidence| {
            evidence.currentness == IbmiScenarioEvidenceCurrentnessV1::Stale
        })));
    }
}
