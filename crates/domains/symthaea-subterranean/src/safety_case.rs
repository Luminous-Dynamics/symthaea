// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Structured claim-argument-evidence safety case.

use crate::fault_tree::{FaultTreeEvaluation, TopEvent};
use crate::requirements::{RequirementId, RequirementRegistry};
use crate::traceability::{TraceabilityMatrix, VerificationArtifactKind};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const SAFETY_CASE_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClaimDisposition {
    Supported,
    Unsupported,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceReference {
    pub artifact_kind: VerificationArtifactKind,
    pub artifact_id: String,
    pub scenario_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetyClaim {
    pub requirement: RequirementId,
    pub claim: String,
    pub argument: String,
    pub evidence: Vec<EvidenceReference>,
    pub disposition: ClaimDisposition,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetyCase {
    pub schema_version: u16,
    pub system: String,
    pub claims: Vec<SafetyClaim>,
    pub residual_top_events: Vec<TopEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetyCaseAssessment {
    pub supported_claims: usize,
    pub unsupported_requirements: Vec<RequirementId>,
    pub rejected_requirements: Vec<RequirementId>,
    pub duplicate_requirements: Vec<RequirementId>,
    pub residual_top_events: Vec<TopEvent>,
}

impl SafetyCaseAssessment {
    pub fn release_eligible(&self) -> bool {
        self.unsupported_requirements.is_empty()
            && self.rejected_requirements.is_empty()
            && self.duplicate_requirements.is_empty()
            && self.residual_top_events.is_empty()
    }
}

impl SafetyCase {
    pub fn assemble(
        registry: &RequirementRegistry,
        traceability: &TraceabilityMatrix,
        fault_evaluation: &FaultTreeEvaluation,
    ) -> Self {
        let claims = registry
            .requirements()
            .iter()
            .map(|definition| {
                let links = traceability.links_for(definition.id);
                let evidence: Vec<EvidenceReference> = links
                    .iter()
                    .map(|link| EvidenceReference {
                        artifact_kind: link.artifact_kind,
                        artifact_id: link.artifact_id.clone(),
                        scenario_id: link.scenario_id.clone(),
                    })
                    .collect();
                let disposition = if evidence.is_empty() {
                    ClaimDisposition::Unsupported
                } else {
                    ClaimDisposition::Supported
                };
                SafetyClaim {
                    requirement: definition.id,
                    claim: definition.title.clone(),
                    argument: format!(
                        "Requirement {} is enforced or verified through {} bounded artifact(s).",
                        definition.id.code(),
                        evidence.len()
                    ),
                    evidence,
                    disposition,
                }
            })
            .collect();
        Self {
            schema_version: SAFETY_CASE_SCHEMA_VERSION,
            system: "symthaea-subterranean".to_string(),
            claims,
            residual_top_events: fault_evaluation.active_top_events.clone(),
        }
    }

    pub fn assess(&self, registry: &RequirementRegistry) -> SafetyCaseAssessment {
        let mut counts = BTreeMap::new();
        let mut supported_claims = 0usize;
        let mut rejected_requirements = Vec::new();
        let mut unsupported = BTreeSet::new();
        for requirement in registry.release_blocking_ids() {
            unsupported.insert(requirement);
        }
        for claim in &self.claims {
            *counts.entry(claim.requirement).or_insert(0usize) += 1;
            match claim.disposition {
                ClaimDisposition::Supported if !claim.evidence.is_empty() => {
                    supported_claims += 1;
                    unsupported.remove(&claim.requirement);
                }
                ClaimDisposition::Rejected => {
                    rejected_requirements.push(claim.requirement);
                    unsupported.remove(&claim.requirement);
                }
                ClaimDisposition::Unsupported | ClaimDisposition::Supported => {}
            }
        }
        let mut duplicate_requirements: Vec<RequirementId> = counts
            .into_iter()
            .filter_map(|(requirement, count)| (count > 1).then_some(requirement))
            .collect();
        let mut unsupported_requirements: Vec<RequirementId> = unsupported.into_iter().collect();
        rejected_requirements.sort();
        rejected_requirements.dedup();
        duplicate_requirements.sort();
        unsupported_requirements.sort();
        let mut residual_top_events = self.residual_top_events.clone();
        residual_top_events.sort();
        residual_top_events.dedup();
        SafetyCaseAssessment {
            supported_claims,
            unsupported_requirements,
            rejected_requirements,
            duplicate_requirements,
            residual_top_events,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fault_tree::FaultTreeModel;
    use std::collections::BTreeSet;

    #[test]
    fn canonical_case_is_release_eligible_when_no_top_event_is_active() {
        let registry = RequirementRegistry::canonical();
        let traceability = TraceabilityMatrix::canonical();
        let faults = FaultTreeModel::canonical().evaluate(&BTreeSet::new());
        let case = SafetyCase::assemble(&registry, &traceability, &faults);
        assert!(case.assess(&registry).release_eligible());
    }

    #[test]
    fn residual_top_event_blocks_release_eligibility() {
        let registry = RequirementRegistry::canonical();
        let traceability = TraceabilityMatrix::canonical();
        let faults = FaultTreeEvaluation {
            active_top_events: vec![TopEvent::ThermalRunaway],
            active_basic_faults: Vec::new(),
        };
        let case = SafetyCase::assemble(&registry, &traceability, &faults);
        assert!(!case.assess(&registry).release_eligible());
    }
}
