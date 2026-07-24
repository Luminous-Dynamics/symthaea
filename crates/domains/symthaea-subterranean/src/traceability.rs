// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Requirement-to-verification traceability with deterministic completeness gates.

use crate::requirements::{RequirementId, RequirementRegistry, VerificationMethod};
use crate::scenario_manifest::ScenarioManifest;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationArtifactKind {
    DeterministicTest,
    RuntimeInvariant,
    AnalysisReport,
    EvidenceField,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceLink {
    pub requirement: RequirementId,
    pub artifact_kind: VerificationArtifactKind,
    pub artifact_id: String,
    pub scenario_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceabilityMatrix {
    links: Vec<TraceLink>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceabilityReport {
    pub linked_requirements: Vec<RequirementId>,
    pub missing_requirements: Vec<RequirementId>,
    pub duplicate_links: Vec<String>,
    pub unknown_scenarios: Vec<String>,
    pub method_mismatches: Vec<RequirementId>,
}

impl TraceabilityReport {
    pub fn passes(&self) -> bool {
        self.missing_requirements.is_empty()
            && self.duplicate_links.is_empty()
            && self.unknown_scenarios.is_empty()
            && self.method_mismatches.is_empty()
    }
}

impl TraceabilityMatrix {
    pub fn canonical() -> Self {
        use RequirementId::*;
        use VerificationArtifactKind::{DeterministicTest, EvidenceField, RuntimeInvariant};
        let links = [
            (SafeCommandBounds, RuntimeInvariant, "INV-CMD-001"),
            (HazardPreemption, RuntimeInvariant, "INV-SAF-001"),
            (
                ReturnReserveProtection,
                DeterministicTest,
                "long_horizon.return_reserve",
            ),
            (
                SensorQuorum,
                DeterministicTest,
                "survivability.critical_sensor_quorum",
            ),
            (ActuatorIsolation, RuntimeInvariant, "INV-ACT-001"),
            (
                ThermalPowerDerating,
                DeterministicTest,
                "survivability.thermal_power",
            ),
            (
                OperatorReplayResistance,
                DeterministicTest,
                "authority.replay_resistance",
            ),
            (
                RecoveryQuorum,
                DeterministicTest,
                "authority.recovery_quorum",
            ),
            (
                UpdateRollback,
                DeterministicTest,
                "authority.update_rollback",
            ),
            (
                PartitionReconciliation,
                DeterministicTest,
                "survivability.partition_reconciliation",
            ),
            (
                EvidenceCompleteness,
                EvidenceField,
                "SafetyEvidenceRecord.certification",
            ),
            (
                CheckpointContinuity,
                DeterministicTest,
                "survivability.checkpoint_replay",
            ),
        ]
        .into_iter()
        .map(|(requirement, artifact_kind, artifact_id)| TraceLink {
            requirement,
            artifact_kind,
            artifact_id: artifact_id.to_string(),
            scenario_id: None,
        })
        .collect();
        Self { links }
    }

    pub fn from_links(links: Vec<TraceLink>) -> Self {
        Self { links }
    }

    pub fn links(&self) -> &[TraceLink] {
        &self.links
    }

    pub fn push(&mut self, link: TraceLink) {
        self.links.push(link);
    }

    pub fn links_for(&self, requirement: RequirementId) -> Vec<&TraceLink> {
        self.links
            .iter()
            .filter(|link| link.requirement == requirement)
            .collect()
    }

    pub fn validate(
        &self,
        registry: &RequirementRegistry,
        scenarios: &[ScenarioManifest],
    ) -> TraceabilityReport {
        let known_scenarios: BTreeSet<&str> = scenarios
            .iter()
            .map(|scenario| scenario.scenario_id.as_str())
            .collect();
        let mut by_requirement: BTreeMap<RequirementId, usize> = BTreeMap::new();
        let mut link_keys = BTreeSet::new();
        let mut duplicate_links = Vec::new();
        let mut unknown_scenarios = Vec::new();
        let mut method_mismatches = Vec::new();

        for link in &self.links {
            *by_requirement.entry(link.requirement).or_default() += 1;
            let key = format!(
                "{}::{:?}::{}::{}",
                link.requirement.code(),
                link.artifact_kind,
                link.artifact_id,
                link.scenario_id.as_deref().unwrap_or("")
            );
            if !link_keys.insert(key.clone()) {
                duplicate_links.push(key);
            }
            if let Some(scenario_id) = link.scenario_id.as_deref() {
                if !known_scenarios.contains(scenario_id) {
                    unknown_scenarios.push(scenario_id.to_string());
                }
            }
            if let Some(definition) = registry.definition(link.requirement) {
                let method_matches = match definition.verification {
                    VerificationMethod::RuntimeInvariant => {
                        link.artifact_kind == VerificationArtifactKind::RuntimeInvariant
                    }
                    VerificationMethod::DeterministicTest => matches!(
                        link.artifact_kind,
                        VerificationArtifactKind::DeterministicTest
                            | VerificationArtifactKind::RuntimeInvariant
                    ),
                    VerificationMethod::Analysis => {
                        link.artifact_kind == VerificationArtifactKind::AnalysisReport
                    }
                    VerificationMethod::Inspection => matches!(
                        link.artifact_kind,
                        VerificationArtifactKind::EvidenceField
                            | VerificationArtifactKind::AnalysisReport
                    ),
                };
                if !method_matches {
                    method_mismatches.push(link.requirement);
                }
            }
        }

        let mut linked_requirements = Vec::new();
        let mut missing_requirements = Vec::new();
        for requirement in registry.release_blocking_ids() {
            if by_requirement
                .get(&requirement)
                .copied()
                .unwrap_or_default()
                == 0
            {
                missing_requirements.push(requirement);
            } else {
                linked_requirements.push(requirement);
            }
        }
        linked_requirements.sort();
        missing_requirements.sort();
        duplicate_links.sort();
        duplicate_links.dedup();
        unknown_scenarios.sort();
        unknown_scenarios.dedup();
        method_mismatches.sort();
        method_mismatches.dedup();

        TraceabilityReport {
            linked_requirements,
            missing_requirements,
            duplicate_links,
            unknown_scenarios,
            method_mismatches,
        }
    }
}

impl Default for TraceabilityMatrix {
    fn default() -> Self {
        Self::canonical()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_matrix_covers_all_release_requirements() {
        let registry = RequirementRegistry::canonical();
        let report = TraceabilityMatrix::canonical().validate(&registry, &[]);
        assert!(report.passes(), "{report:?}");
    }

    #[test]
    fn omitted_requirement_blocks_completeness() {
        let registry = RequirementRegistry::canonical();
        let mut links = TraceabilityMatrix::canonical().links().to_vec();
        links.retain(|link| link.requirement != RequirementId::UpdateRollback);
        let report = TraceabilityMatrix::from_links(links).validate(&registry, &[]);
        assert_eq!(
            report.missing_requirements,
            vec![RequirementId::UpdateRollback]
        );
    }
}
