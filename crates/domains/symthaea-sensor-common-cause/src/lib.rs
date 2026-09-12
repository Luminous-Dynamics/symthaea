// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Common-cause fault-domain diversity assurance for admitted sensor evidence.
//!
//! This crate is deliberately separate from basic replay/freshness admission. It
//! asks whether the *accepted physical sensors* are diverse enough across explicit
//! common-cause domains such as power, clock, network, compute, enclosure, or site.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use serde::{Deserialize, Serialize};
use symthaea_domain_awareness::ObservationEnvelope;
use symthaea_sensor_trust::ObservationAdmissionReport;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum FaultDomainKind {
    Power,
    Clock,
    Network,
    Compute,
    Enclosure,
    Mount,
    Site,
    EnvironmentalExposure,
    Custom,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FaultDomainTag {
    pub kind: FaultDomainKind,
    pub domain_id: String,
}

impl FaultDomainTag {
    pub fn validate(&self) -> bool {
        !self.domain_id.trim().is_empty()
    }
}

/// Evidence-backed common-cause profile for one physical sensing element.
///
/// Each category may appear at most once. A genuinely redundant dependency can be
/// represented by a reviewed composite domain id and supporting evidence rather than
/// silently assuming the redundancy makes the source independent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalSourceFaultProfile {
    pub physical_source_id: String,
    pub domains: Vec<FaultDomainTag>,
    pub evidence_refs: Vec<String>,
}

impl PhysicalSourceFaultProfile {
    pub fn validate(&self) -> bool {
        let unique_kinds = self
            .domains
            .iter()
            .map(|domain| domain.kind)
            .collect::<BTreeSet<_>>();
        !self.physical_source_id.trim().is_empty()
            && !self.domains.is_empty()
            && self.domains.iter().all(FaultDomainTag::validate)
            && unique_kinds.len() == self.domains.len()
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
    }

    pub fn domain(&self, kind: FaultDomainKind) -> Option<&str> {
        self.domains
            .iter()
            .find(|domain| domain.kind == kind)
            .map(|domain| domain.domain_id.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FaultDomainRequirement {
    pub kind: FaultDomainKind,
    pub minimum_distinct_domains: usize,
}

impl FaultDomainRequirement {
    pub fn validate(&self) -> bool {
        self.minimum_distinct_domains > 0
    }
}

/// Deployment-reviewed diversity requirements. No safety-critical defaults.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommonCausePolicy {
    pub policy_id: String,
    pub requirements: Vec<FaultDomainRequirement>,
}

impl CommonCausePolicy {
    pub fn validate(&self) -> bool {
        let unique = self
            .requirements
            .iter()
            .map(|requirement| requirement.kind)
            .collect::<BTreeSet<_>>();
        !self.policy_id.trim().is_empty()
            && !self.requirements.is_empty()
            && self.requirements.iter().all(FaultDomainRequirement::validate)
            && unique.len() == self.requirements.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommonCauseIssue {
    InvalidPolicy,
    NoAcceptedPhysicalSources,
    AcceptedObservationMissing(String),
    InvalidProfile(String),
    DuplicateProfile(String),
    MissingProfile(String),
    MissingRequiredDomain {
        physical_source_id: String,
        kind: FaultDomainKind,
    },
    InsufficientDiversity {
        kind: FaultDomainKind,
        observed_distinct_domains: usize,
        required_distinct_domains: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommonCauseDiversityReport {
    pub policy_id: String,
    pub accepted_physical_sources: usize,
    pub profiled_physical_sources: usize,
    pub distinct_domains: BTreeMap<FaultDomainKind, usize>,
    pub issues: Vec<CommonCauseIssue>,
    pub requires_fail_closed: bool,
}

impl CommonCauseDiversityReport {
    /// Diversity evidence can restrict confidence/capability but cannot grant authority.
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

/// Assess common-cause diversity for the physical sources actually admitted by
/// `symthaea-sensor-trust`.
///
/// No topology is inferred. Every common-cause domain must come from an explicit,
/// evidence-backed source profile.
pub fn assess_common_cause_diversity(
    observations: &[ObservationEnvelope],
    admission: &ObservationAdmissionReport,
    profiles: &[PhysicalSourceFaultProfile],
    policy: &CommonCausePolicy,
) -> CommonCauseDiversityReport {
    if !policy.validate() {
        return CommonCauseDiversityReport {
            policy_id: policy.policy_id.clone(),
            accepted_physical_sources: 0,
            profiled_physical_sources: 0,
            distinct_domains: BTreeMap::new(),
            issues: vec![CommonCauseIssue::InvalidPolicy],
            requires_fail_closed: true,
        };
    }

    let accepted_ids = admission
        .accepted_observation_ids
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let by_id = observations
        .iter()
        .map(|observation| (observation.observation_id, observation))
        .collect::<HashMap<_, _>>();

    let mut issues = Vec::new();
    let mut accepted_sources = BTreeSet::<String>::new();
    for observation_id in &accepted_ids {
        match by_id.get(observation_id) {
            Some(observation) => {
                accepted_sources.insert(observation.lineage.physical_source_id.clone());
            }
            None => issues.push(CommonCauseIssue::AcceptedObservationMissing(
                observation_id.to_string(),
            )),
        }
    }
    if accepted_sources.is_empty() {
        issues.push(CommonCauseIssue::NoAcceptedPhysicalSources);
    }

    let mut profile_map = BTreeMap::<String, &PhysicalSourceFaultProfile>::new();
    let mut duplicate_profile_ids = BTreeSet::new();
    for profile in profiles {
        if !profile.validate() {
            issues.push(CommonCauseIssue::InvalidProfile(
                profile.physical_source_id.clone(),
            ));
            continue;
        }
        if profile_map
            .insert(profile.physical_source_id.clone(), profile)
            .is_some()
        {
            duplicate_profile_ids.insert(profile.physical_source_id.clone());
        }
    }
    for physical_source_id in duplicate_profile_ids {
        issues.push(CommonCauseIssue::DuplicateProfile(physical_source_id));
    }

    let mut distinct = BTreeMap::<FaultDomainKind, BTreeSet<String>>::new();
    let mut profiled_sources = 0usize;

    for physical_source_id in &accepted_sources {
        let Some(profile) = profile_map.get(physical_source_id) else {
            issues.push(CommonCauseIssue::MissingProfile(physical_source_id.clone()));
            continue;
        };
        profiled_sources = profiled_sources.saturating_add(1);
        for requirement in &policy.requirements {
            match profile.domain(requirement.kind) {
                Some(domain_id) => {
                    distinct
                        .entry(requirement.kind)
                        .or_default()
                        .insert(domain_id.to_string());
                }
                None => issues.push(CommonCauseIssue::MissingRequiredDomain {
                    physical_source_id: physical_source_id.clone(),
                    kind: requirement.kind,
                }),
            }
        }
    }

    let mut distinct_counts = BTreeMap::new();
    for requirement in &policy.requirements {
        let observed = distinct
            .get(&requirement.kind)
            .map_or(0, BTreeSet::len);
        distinct_counts.insert(requirement.kind, observed);
        if observed < requirement.minimum_distinct_domains {
            issues.push(CommonCauseIssue::InsufficientDiversity {
                kind: requirement.kind,
                observed_distinct_domains: observed,
                required_distinct_domains: requirement.minimum_distinct_domains,
            });
        }
    }

    CommonCauseDiversityReport {
        policy_id: policy.policy_id.clone(),
        accepted_physical_sources: accepted_sources.len(),
        profiled_physical_sources: profiled_sources,
        distinct_domains: distinct_counts,
        requires_fail_closed: !issues.is_empty(),
        issues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_domain_awareness::{
        Domain, EvidenceLineage, IntegrityStatus, Measurement, MeasurementUncertainty, Modality,
        SensorHealth, TimeEvidence,
    };
    use symthaea_sensor_trust::ObservationAdmissionReport;
    use uuid::Uuid;

    fn observation(source_id: &str, physical_source_id: &str, modality: Modality) -> ObservationEnvelope {
        ObservationEnvelope {
            observation_id: Uuid::new_v4(),
            source_id: source_id.into(),
            sequence: 1,
            domain: Domain::Air,
            modality,
            coordinate_frame: "local".into(),
            measurement: Measurement::Scalar {
                quantity: "presence".into(),
                value: 1.0,
                unit: "bool".into(),
            },
            uncertainty: MeasurementUncertainty {
                position_sigma_m: None,
                velocity_sigma_mps: None,
                bearing_sigma_deg: None,
            },
            time: TimeEvidence {
                observed_at_ms: 1_000,
                received_at_ms: 1_001,
                clock_source: "clock".into(),
                clock_uncertainty_ms: 1,
                maximum_valid_age_ms: 500,
            },
            lineage: EvidenceLineage {
                physical_source_id: physical_source_id.into(),
                processor_id: format!("processor-{source_id}"),
                network_path: "network".into(),
                clock_domain: "clock-domain".into(),
            },
            integrity: IntegrityStatus::Verified,
            sensor_health: SensorHealth::Nominal,
            confidence: 0.9,
            evidence_refs: vec![format!("evidence:{source_id}")],
        }
    }

    fn admission(observations: &[ObservationEnvelope]) -> ObservationAdmissionReport {
        ObservationAdmissionReport {
            accepted_observation_ids: observations.iter().map(|obs| obs.observation_id).collect(),
            rejections: Vec::new(),
            independent_physical_sources: observations
                .iter()
                .map(|obs| obs.lineage.physical_source_id.as_str())
                .collect::<BTreeSet<_>>()
                .len(),
            independent_modalities: observations
                .iter()
                .map(|obs| obs.modality)
                .collect::<HashSet<_>>()
                .len(),
            isolated_sources: 0,
            requires_fail_closed: false,
        }
    }

    fn profile(source: &str, power: &str, network: &str) -> PhysicalSourceFaultProfile {
        PhysicalSourceFaultProfile {
            physical_source_id: source.into(),
            domains: vec![
                FaultDomainTag {
                    kind: FaultDomainKind::Power,
                    domain_id: power.into(),
                },
                FaultDomainTag {
                    kind: FaultDomainKind::Network,
                    domain_id: network.into(),
                },
            ],
            evidence_refs: vec![format!("fault-profile:{source}")],
        }
    }

    fn policy() -> CommonCausePolicy {
        CommonCausePolicy {
            policy_id: "dual-diversity-v1".into(),
            requirements: vec![
                FaultDomainRequirement {
                    kind: FaultDomainKind::Power,
                    minimum_distinct_domains: 2,
                },
                FaultDomainRequirement {
                    kind: FaultDomainKind::Network,
                    minimum_distinct_domains: 2,
                },
            ],
        }
    }

    #[test]
    fn two_physical_sensors_with_diverse_fault_domains_pass() {
        let observations = vec![
            observation("eo-a", "camera-a", Modality::ElectroOptical),
            observation("ir-b", "camera-b", Modality::Infrared),
        ];
        let report = assess_common_cause_diversity(
            &observations,
            &admission(&observations),
            &[
                profile("camera-a", "power-a", "network-a"),
                profile("camera-b", "power-b", "network-b"),
            ],
            &policy(),
        );
        assert!(!report.requires_fail_closed);
        assert_eq!(report.distinct_domains[&FaultDomainKind::Power], 2);
        assert_eq!(report.distinct_domains[&FaultDomainKind::Network], 2);
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn shared_power_common_cause_fails_even_with_two_physical_sensors() {
        let observations = vec![
            observation("eo-a", "camera-a", Modality::ElectroOptical),
            observation("ir-b", "camera-b", Modality::Infrared),
        ];
        let report = assess_common_cause_diversity(
            &observations,
            &admission(&observations),
            &[
                profile("camera-a", "shared-power", "network-a"),
                profile("camera-b", "shared-power", "network-b"),
            ],
            &policy(),
        );
        assert!(report.requires_fail_closed);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            CommonCauseIssue::InsufficientDiversity {
                kind: FaultDomainKind::Power,
                observed_distinct_domains: 1,
                required_distinct_domains: 2,
            }
        )));
    }

    #[test]
    fn missing_profile_fails_closed() {
        let observations = vec![
            observation("eo-a", "camera-a", Modality::ElectroOptical),
            observation("ir-b", "camera-b", Modality::Infrared),
        ];
        let report = assess_common_cause_diversity(
            &observations,
            &admission(&observations),
            &[profile("camera-a", "power-a", "network-a")],
            &policy(),
        );
        assert!(report.requires_fail_closed);
        assert!(report
            .issues
            .contains(&CommonCauseIssue::MissingProfile("camera-b".into())));
    }

    #[test]
    fn missing_required_domain_fails_closed() {
        let observations = vec![observation(
            "eo-a",
            "camera-a",
            Modality::ElectroOptical,
        )];
        let incomplete = PhysicalSourceFaultProfile {
            physical_source_id: "camera-a".into(),
            domains: vec![FaultDomainTag {
                kind: FaultDomainKind::Power,
                domain_id: "power-a".into(),
            }],
            evidence_refs: vec!["fault-profile:camera-a".into()],
        };
        let report = assess_common_cause_diversity(
            &observations,
            &admission(&observations),
            &[incomplete],
            &CommonCausePolicy {
                policy_id: "network-required".into(),
                requirements: vec![FaultDomainRequirement {
                    kind: FaultDomainKind::Network,
                    minimum_distinct_domains: 1,
                }],
            },
        );
        assert!(report.requires_fail_closed);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            CommonCauseIssue::MissingRequiredDomain {
                physical_source_id,
                kind: FaultDomainKind::Network,
            } if physical_source_id == "camera-a"
        )));
    }
}
