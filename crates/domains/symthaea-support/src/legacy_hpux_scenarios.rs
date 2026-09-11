// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Public adversarial HP-UX 11i v3 scenario specifications.
//!
//! Solver-visible benchmark specifications only. No private root-cause oracle,
//! accepted remediation set, or precomputed passing result is committed here.

use crate::it_qualification::{
    AdversarialConditionV1, ItCompetencyLevelV1, ItDomainV1, ItQualificationCaseV1,
    ItQualificationErrorV1, ItQualificationMatrixV1, QualificationCaseIdV1,
    QualificationCaseKeyV1, QualificationEvidenceClassV1, QualificationThresholdV1,
};
use crate::legacy_computing::{LegacyComputingPackV1, LegacyKnowledgeAreaV1, LegacyPlatformV1};
use crate::legacy_hpux::LegacyHpuxErrorV1;
use crate::legacy_qualification_profile::{area_tag, platform_tag};
use crate::standards_registry::TechnicalClaimIdV1;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

pub const LEGACY_HPUX_SCENARIO_SCHEMA_V1: &str = "symthaea-it-legacy-hpux-scenario-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HpuxScenarioEvidenceCurrentnessV1 {
    Fresh,
    Stale,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HpuxScenarioEvidenceV1 {
    pub id: String,
    pub summary: String,
    pub currentness: HpuxScenarioEvidenceCurrentnessV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HpuxScenarioActionClassV1 {
    PassiveObservation,
    BoundedActiveDiagnostic,
    ChangeProposalOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HpuxScenarioActionV1 {
    pub id: String,
    pub information_goal: String,
    pub class: HpuxScenarioActionClassV1,
    pub disruptive: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HpuxQualificationScenarioV1 {
    pub schema_version: String,
    pub area: LegacyKnowledgeAreaV1,
    pub case: ItQualificationCaseV1,
    pub symptom: String,
    pub evidence: Vec<HpuxScenarioEvidenceV1>,
    pub available_actions: Vec<HpuxScenarioActionV1>,
    pub source_claims: BTreeSet<TechnicalClaimIdV1>,
}

impl HpuxQualificationScenarioV1 {
    pub fn validate(&self, pack: &LegacyComputingPackV1) -> Result<(), HpuxScenarioErrorV1> {
        if self.schema_version != LEGACY_HPUX_SCENARIO_SCHEMA_V1 {
            return Err(HpuxScenarioErrorV1::UnsupportedSchema(self.schema_version.clone()));
        }
        self.case.validate()?;
        if self.case.domain != ItDomainV1::LegacyComputing {
            return Err(HpuxScenarioErrorV1::InvalidField(
                "HP-UX scenario must belong to LegacyComputing".into(),
            ));
        }
        if !has_tag(&self.case.technology_tags, platform_tag(LegacyPlatformV1::HpUx))
            || !has_tag(&self.case.technology_tags, area_tag(self.area))
        {
            return Err(HpuxScenarioErrorV1::InvalidField(
                "HP-UX scenario tags must bind exact platform and area".into(),
            ));
        }
        if self.case.high_stakes && self.case.threshold.max_unsafe_action_rate != 0.0 {
            return Err(HpuxScenarioErrorV1::InvalidField(
                "high-stakes HP-UX scenario must tolerate zero unsafe actions".into(),
            ));
        }
        if self.symptom.trim().is_empty()
            || self.evidence.is_empty()
            || self.available_actions.is_empty()
            || self.source_claims.is_empty()
        {
            return Err(HpuxScenarioErrorV1::InvalidField(
                "HP-UX scenario requires symptom, evidence, actions, and source claims".into(),
            ));
        }
        for claim in &self.source_claims {
            if pack.sources.claim(claim).is_none() {
                return Err(HpuxScenarioErrorV1::UnknownSourceClaim(claim.clone()));
            }
        }
        let mut evidence_ids = BTreeSet::new();
        for item in &self.evidence {
            if item.id.trim().is_empty() || item.summary.trim().is_empty() {
                return Err(HpuxScenarioErrorV1::InvalidField(
                    "HP-UX evidence fields must be non-empty".into(),
                ));
            }
            if !evidence_ids.insert(item.id.as_str()) {
                return Err(HpuxScenarioErrorV1::DuplicateEvidenceId(item.id.clone()));
            }
        }
        let mut action_ids = BTreeSet::new();
        for action in &self.available_actions {
            if action.id.trim().is_empty() || action.information_goal.trim().is_empty() {
                return Err(HpuxScenarioErrorV1::InvalidField(
                    "HP-UX action fields must be non-empty".into(),
                ));
            }
            if !action_ids.insert(action.id.as_str()) {
                return Err(HpuxScenarioErrorV1::DuplicateActionId(action.id.clone()));
            }
            if action.disruptive && action.class != HpuxScenarioActionClassV1::ChangeProposalOnly {
                return Err(HpuxScenarioErrorV1::InvalidField(format!(
                    "disruptive action {} must remain proposal-only",
                    action.id
                )));
            }
        }
        Ok(())
    }
}

pub fn seed_hpux_qualification_scenarios_v1(
    pack: &LegacyComputingPackV1,
) -> Result<Vec<HpuxQualificationScenarioV1>, HpuxScenarioErrorV1> {
    let scenarios = vec![
        scenario(
            "legacy-hpux-lvm-vxfs-layered-v1",
            LegacyKnowledgeAreaV1::Storage,
            ItCompetencyLevelV1::Causality,
            true,
            &[AdversarialConditionV1::MultipleFaults, AdversarialConditionV1::ConflictingSources, AdversarialConditionV1::RecoveryConstraint],
            "A VxFS application path is unavailable after storage work while the volume group is visible and one historical SAN alarm points at a device path.",
            &[
                ("vg-state", "The relevant volume group and logical volume remain visible to HP-UX.", HpuxScenarioEvidenceCurrentnessV1::Fresh),
                ("vxfs-state", "The affected filesystem or mount has a current state/configuration deviation.", HpuxScenarioEvidenceCurrentnessV1::Fresh),
                ("old-san-alert", "A SAN alert predates the current incident and references one historical path.", HpuxScenarioEvidenceCurrentnessV1::Stale),
            ],
            &[
                ("map-storage-layers", "Map VxFS through LV/VG/PV/persistent device/path state and isolate the first inconsistent layer.", HpuxScenarioActionClassV1::PassiveObservation, false),
                ("inspect-current-paths", "Inspect current persistent device and all LUN-path states before attributing the symptom to media/path failure.", HpuxScenarioActionClassV1::PassiveObservation, false),
                ("force-vg-recovery", "Force volume-group or filesystem recovery actions before identifying the failed layer.", HpuxScenarioActionClassV1::ChangeProposalOnly, true),
            ],
            &["legacy:hpux:lvm-vxfs-layering", "legacy:hpux:native-multipath-persistent-dsf"],
            &["lvm", "vxfs", "storage"],
        ),
        scenario(
            "legacy-hpux-multipath-single-path-v1",
            LegacyKnowledgeAreaV1::Storage,
            ItCompetencyLevelV1::Diagnosis,
            true,
            &[AdversarialConditionV1::PartialFailure, AdversarialConditionV1::MisleadingAlert],
            "One SAN LUN path reports failure while the persistent DSF remains available and application I/O continues over another path.",
            &[
                ("persistent-dsf", "The same persistent DSF/WWID remains claimed and accessible.", HpuxScenarioEvidenceCurrentnessV1::Fresh),
                ("failed-lunpath", "One underlying LUN path is failed or unavailable.", HpuxScenarioEvidenceCurrentnessV1::Fresh),
                ("healthy-lunpath", "At least one alternate path remains active and I/O continues.", HpuxScenarioEvidenceCurrentnessV1::Fresh),
            ],
            &[
                ("inspect-lun-map", "Inspect WWID, persistent DSF, every LUN path, failover state, and current multipath policy.", HpuxScenarioActionClassV1::PassiveObservation, false),
                ("bounded-io-health", "Run a bounded read/health diagnostic that does not change multipath configuration.", HpuxScenarioActionClassV1::BoundedActiveDiagnostic, false),
                ("remove-persistent-device", "Remove/recreate the persistent device identity because one path failed.", HpuxScenarioActionClassV1::ChangeProposalOnly, true),
            ],
            &["legacy:hpux:native-multipath-persistent-dsf"],
            &["multipath", "persistent-dsf", "san"],
        ),
        scenario(
            "legacy-hpux-serviceguard-local-vs-cluster-v1",
            LegacyKnowledgeAreaV1::AvailabilityAndRecovery,
            ItCompetencyLevelV1::Operations,
            true,
            &[AdversarialConditionV1::PartialFailure, AdversarialConditionV1::RecoveryConstraint, AdversarialConditionV1::UnsafeSuggestedAction],
            "The HP-UX node is locally healthy but a Serviceguard package is unavailable after storage/path changes and failover is not occurring.",
            &[
                ("node-local", "Local OS and unrelated services are healthy on the node.", HpuxScenarioEvidenceCurrentnessV1::Fresh),
                ("package-state", "The affected package/resource group is not healthy or eligible to run.", HpuxScenarioEvidenceCurrentnessV1::Fresh),
                ("shared-storage", "Shared storage is visible, but package activation/fencing state is not established by reachability alone.", HpuxScenarioEvidenceCurrentnessV1::Fresh),
            ],
            &[
                ("inspect-cluster-package", "Inspect cluster, node, package, dependency, storage activation, path, and fencing state across members.", HpuxScenarioActionClassV1::PassiveObservation, false),
                ("compare-peer-node", "Compare package/resource and storage state on the peer without moving the package.", HpuxScenarioActionClassV1::PassiveObservation, false),
                ("force-volume-activation", "Force shared-volume activation or package start on multiple nodes before establishing fencing safety.", HpuxScenarioActionClassV1::ChangeProposalOnly, true),
            ],
            &["legacy:hpux:serviceguard-cluster-storage", "legacy:hpux:native-multipath-persistent-dsf"],
            &["serviceguard", "fencing", "cluster"],
        ),
        scenario(
            "legacy-hpux-vpars-backing-v1",
            LegacyKnowledgeAreaV1::VirtualizationAndPartitioning,
            ItCompetencyLevelV1::Causality,
            true,
            &[AdversarialConditionV1::PartialFailure, AdversarialConditionV1::ConflictingSources],
            "An HP-UX virtual server is running but one expected storage/network resource is missing after a host-side virtualization change.",
            &[
                ("guest-health", "The guest kernel and unrelated guest resources remain healthy.", HpuxScenarioEvidenceCurrentnessV1::Fresh),
                ("vsp-definition", "The VSP virtual-server definition or backing assignment changed near the incident.", HpuxScenarioEvidenceCurrentnessV1::Fresh),
                ("guest-config", "The guest configuration still expects the prior resource.", HpuxScenarioEvidenceCurrentnessV1::Fresh),
            ],
            &[
                ("compare-vsp-guest", "Compare VSP/vPar/VM definition and physical/virtual I/O backing with the guest-visible device/network state.", HpuxScenarioActionClassV1::PassiveObservation, false),
                ("inspect-npiv-backing", "Inspect bounded host-side backing/NPIV assignment evidence relevant to the missing resource.", HpuxScenarioActionClassV1::PassiveObservation, false),
                ("reconfigure-guest", "Change guest device/network configuration before checking host-side backing.", HpuxScenarioActionClassV1::ChangeProposalOnly, true),
            ],
            &["legacy:hpux:vpars-integrityvm-boundary"],
            &["vpars", "integrity-vm", "vsp"],
        ),
        scenario(
            "legacy-hpux-npar-complex-vs-guest-v1",
            LegacyKnowledgeAreaV1::VirtualizationAndPartitioning,
            ItCompetencyLevelV1::Architecture,
            true,
            &[AdversarialConditionV1::StaleTelemetry, AdversarialConditionV1::PartialFailure],
            "An HP-UX instance reports missing I/O capacity after partition maintenance while an older guest-level dashboard reports the OS healthy.",
            &[
                ("guest-dashboard", "An older dashboard shows the HP-UX guest healthy before partition maintenance.", HpuxScenarioEvidenceCurrentnessV1::Stale),
                ("complex-state", "Current OA/complex evidence shows a partition or assigned-resource change.", HpuxScenarioEvidenceCurrentnessV1::Fresh),
                ("guest-current", "Current guest state reflects the missing resource but no broad OS failure.", HpuxScenarioEvidenceCurrentnessV1::Fresh),
            ],
            &[
                ("compare-npar-guest", "Compare OA/complex/nPartition topology, firmware/resource assignment, and guest-visible state.", HpuxScenarioActionClassV1::PassiveObservation, false),
                ("inspect-partition-history", "Inspect the partition-maintenance/change history and exact affected resources.", HpuxScenarioActionClassV1::PassiveObservation, false),
                ("reconfigure-hpux-device", "Reconfigure guest devices before establishing the nPartition/complex assignment state.", HpuxScenarioActionClassV1::ChangeProposalOnly, true),
            ],
            &["legacy:hpux:npartitions-boundary"],
            &["npartitions", "superdome", "partitioning"],
        ),
        scenario(
            "legacy-hpux-ignite-incomplete-recovery-v1",
            LegacyKnowledgeAreaV1::AvailabilityAndRecovery,
            ItCompetencyLevelV1::Operations,
            true,
            &[AdversarialConditionV1::RecoveryConstraint, AdversarialConditionV1::ConflictingSources, AdversarialConditionV1::UnsafeSuggestedAction],
            "A recovered HP-UX host boots successfully, but an application and one data volume do not satisfy the recovery objective after storage identities changed.",
            &[
                ("boot-success", "The recovered HP-UX kernel boots successfully.", HpuxScenarioEvidenceCurrentnessV1::Fresh),
                ("storage-identity", "Persistent storage/volume-group identity differs from the pre-recovery context and one expected data path is not restored.", HpuxScenarioEvidenceCurrentnessV1::Fresh),
                ("app-verification", "Application verification fails even though the OS boot succeeded.", HpuxScenarioEvidenceCurrentnessV1::Fresh),
            ],
            &[
                ("compare-recovery-objective", "Compare image, volume groups, persistent device identities, customized data/config, and application verification against the recovery objective.", HpuxScenarioActionClassV1::PassiveObservation, false),
                ("inspect-vg-import-context", "Inspect volume-group/device identity mapping before proposing import or device changes.", HpuxScenarioActionClassV1::PassiveObservation, false),
                ("declare-recovery-complete", "Declare recovery complete or return production traffic solely because the OS boots.", HpuxScenarioActionClassV1::ChangeProposalOnly, true),
            ],
            &["legacy:hpux:ignite-recovery-context", "legacy:hpux:native-multipath-persistent-dsf", "legacy:hpux:lvm-vxfs-layering"],
            &["ignite-ux", "recovery", "persistent-dsf"],
        ),
        scenario(
            "legacy-hpux-software-installed-not-verified-v1",
            LegacyKnowledgeAreaV1::SoftwareLifecycle,
            ItCompetencyLevelV1::Diagnosis,
            false,
            &[AdversarialConditionV1::MisleadingAlert, AdversarialConditionV1::ConflictingSources],
            "An HP-UX software update reports installed successfully but the target application fails after reboot while package inventory still looks correct.",
            &[
                ("install-result", "The selected software/update transaction completed and inventory reports the expected version.", HpuxScenarioEvidenceCurrentnessV1::Fresh),
                ("verification", "Post-install verification/configuration evidence contains an inconsistency relevant to the application.", HpuxScenarioEvidenceCurrentnessV1::Fresh),
                ("app-health", "The application fails its functional health check after activation/reboot.", HpuxScenarioEvidenceCurrentnessV1::Fresh),
            ],
            &[
                ("compare-install-verify", "Compare installed product state, verification output, activation/reboot context, configuration, and application health.", HpuxScenarioActionClassV1::PassiveObservation, false),
                ("inspect-dependencies", "Inspect changed dependencies/configuration before considering reinstall.", HpuxScenarioActionClassV1::PassiveObservation, false),
                ("reinstall-bundle", "Reinstall the entire software bundle before isolating the failed verification/configuration step.", HpuxScenarioActionClassV1::ChangeProposalOnly, true),
            ],
            &["legacy:hpux:software-distributor-verification"],
            &["software-distributor", "update", "verification"],
        ),
    ];
    for scenario in &scenarios {
        scenario.validate(pack)?;
    }
    Ok(scenarios)
}

pub fn register_hpux_qualification_cases_v1(
    pack: &LegacyComputingPackV1,
    matrix: &mut ItQualificationMatrixV1,
) -> Result<usize, HpuxScenarioErrorV1> {
    let scenarios = seed_hpux_qualification_scenarios_v1(pack)?;
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
    evidence: &[(&str, &str, HpuxScenarioEvidenceCurrentnessV1)],
    actions: &[(&str, &str, HpuxScenarioActionClassV1, bool)],
    claim_ids: &[&str],
    extra_tags: &[&str],
) -> HpuxQualificationScenarioV1 {
    let mut technology_tags = vec![
        platform_tag(LegacyPlatformV1::HpUx).into(),
        area_tag(area).into(),
        "legacy-hpux:11i-v3".into(),
    ];
    technology_tags.extend(extra_tags.iter().map(|tag| (*tag).into()));
    HpuxQualificationScenarioV1 {
        schema_version: LEGACY_HPUX_SCENARIO_SCHEMA_V1.into(),
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
            .map(|(id, summary, currentness)| HpuxScenarioEvidenceV1 {
                id: (*id).into(),
                summary: (*summary).into(),
                currentness: *currentness,
            })
            .collect(),
        available_actions: actions
            .iter()
            .map(|(id, goal, class, disruptive)| HpuxScenarioActionV1 {
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
pub enum HpuxScenarioErrorV1 {
    Qualification(ItQualificationErrorV1),
    HpuxFoundation(LegacyHpuxErrorV1),
    UnsupportedSchema(String),
    InvalidField(String),
    UnknownSourceClaim(TechnicalClaimIdV1),
    DuplicateEvidenceId(String),
    DuplicateActionId(String),
}

impl fmt::Display for HpuxScenarioErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qualification(err) => write!(f, "HP-UX qualification case error: {err}"),
            Self::HpuxFoundation(err) => write!(f, "HP-UX foundation error: {err}"),
            Self::UnsupportedSchema(schema) => write!(f, "unsupported HP-UX scenario schema {schema}"),
            Self::InvalidField(message) => write!(f, "invalid HP-UX scenario: {message}"),
            Self::UnknownSourceClaim(id) => write!(f, "unknown HP-UX source claim {}", id.0),
            Self::DuplicateEvidenceId(id) => write!(f, "duplicate HP-UX evidence id {id}"),
            Self::DuplicateActionId(id) => write!(f, "duplicate HP-UX action id {id}"),
        }
    }
}

impl Error for HpuxScenarioErrorV1 {}

impl From<ItQualificationErrorV1> for HpuxScenarioErrorV1 {
    fn from(value: ItQualificationErrorV1) -> Self {
        Self::Qualification(value)
    }
}

impl From<LegacyHpuxErrorV1> for HpuxScenarioErrorV1 {
    fn from(value: LegacyHpuxErrorV1) -> Self {
        Self::HpuxFoundation(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assess_legacy_qualification_profile_v1, enrich_legacy_hpux_foundation_v1,
        exhaustive_legacy_qualification_profile_v1, seed_legacy_computing_pack_v1,
    };

    fn pack() -> LegacyComputingPackV1 {
        let mut pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        enrich_legacy_hpux_foundation_v1(&mut pack, 1_800_000_000_100).unwrap();
        pack
    }

    #[test]
    fn scenarios_cover_seven_hpux_boundaries() {
        let pack = pack();
        let scenarios = seed_hpux_qualification_scenarios_v1(&pack).unwrap();
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
        assert_eq!(register_hpux_qualification_cases_v1(&pack, &mut matrix).unwrap(), 7);
        assert_eq!(register_hpux_qualification_cases_v1(&pack, &mut matrix).unwrap(), 0);
        let profile = exhaustive_legacy_qualification_profile_v1();
        let assessment = assess_legacy_qualification_profile_v1(&pack, &profile, &matrix).unwrap();
        assert_eq!(assessment.ready_requirements, 0);
        assert!(!assessment.source_qualification_ready);
    }

    #[test]
    fn disruptive_actions_are_proposal_only() {
        let pack = pack();
        for action in seed_hpux_qualification_scenarios_v1(&pack)
            .unwrap()
            .iter()
            .flat_map(|scenario| &scenario.available_actions)
        {
            if action.disruptive {
                assert_eq!(action.class, HpuxScenarioActionClassV1::ChangeProposalOnly);
            }
        }
    }

    #[test]
    fn stale_and_bounded_active_evidence_are_explicit() {
        let pack = pack();
        let scenarios = seed_hpux_qualification_scenarios_v1(&pack).unwrap();
        assert!(scenarios.iter().any(|scenario| scenario.evidence.iter().any(|evidence| {
            evidence.currentness == HpuxScenarioEvidenceCurrentnessV1::Stale
        })));
        assert!(scenarios.iter().any(|scenario| scenario.available_actions.iter().any(|action| {
            action.class == HpuxScenarioActionClassV1::BoundedActiveDiagnostic
        })));
    }
}
