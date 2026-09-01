// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Combined effect attribution + checkpoint continuity qualification.
//!
//! Checkpoint consistency is an integrity/continuity dimension. It must not
//! silently upgrade observation coverage, exact-effect attribution, or evidence
//! authenticity. External signatures/witnesses remain a separate assurance
//! layer beyond this deterministic composition.

use symthaea_cogsec_evidence::{
    CheckpointBuildError, CheckpointVerificationReport, CheckpointedEffectBoundEvidence,
    EvidenceCheckpoint, verify_checkpoint_chain,
};
use symthaea_evidence_plane::EvidenceCounters;

use crate::{
    EffectBoundScenarioQualificationReport, ObserverOnlyScenario, OwnerAwareScenario,
    qualify_observer_effect_bound_scenario, qualify_owner_aware_effect_bound_scenario,
};

/// Combined structural qualification including checkpoint continuity.
#[derive(Debug, Clone, PartialEq)]
pub struct IntegrityBoundScenarioQualificationReport {
    /// Observation/count/attribution/exact-effect qualification.
    pub effect_bound: EffectBoundScenarioQualificationReport,
    /// Deterministic checkpoint/hash-chain verification.
    pub integrity: CheckpointVerificationReport,
}

impl IntegrityBoundScenarioQualificationReport {
    /// Whether the base scenario supports scoped observation coverage.
    pub fn qualifies_for_scoped_observation_coverage(&self) -> bool {
        self.effect_bound.qualifies_for_scoped_observation_coverage()
    }

    /// Whether mapped attribution and exact-effect equality qualify.
    ///
    /// This deliberately does not imply checkpoint continuity.
    pub fn qualifies_for_effect_bound_mapped_attribution(&self) -> bool {
        self.effect_bound
            .qualifies_for_effect_bound_mapped_attribution()
    }

    /// Whether the supplied checkpoint continuation is internally consistent.
    ///
    /// This is not an authenticity/signature claim.
    pub fn checkpoint_chain_is_consistent(&self) -> bool {
        self.integrity.chain_is_consistent()
    }

    /// Whether mapped exact-effect attribution is also bound into a consistent
    /// checkpoint continuation.
    pub fn qualifies_for_integrity_bound_mapped_attribution(&self) -> bool {
        self.qualifies_for_effect_bound_mapped_attribution()
            && self.checkpoint_chain_is_consistent()
    }

    /// Whether all-stage exact-effect attribution and checkpoint continuity both qualify.
    pub fn qualifies_for_integrity_bound_strong_attribution(&self) -> bool {
        self.effect_bound
            .qualifies_for_effect_bound_strong_attribution()
            && self.checkpoint_chain_is_consistent()
    }
}

/// Qualify one observer-only checkpointed snapshot against an optional external anchor.
pub fn qualify_checkpointed_observer_scenario(
    scenario: &ObserverOnlyScenario,
    item: &CheckpointedEffectBoundEvidence,
    anchor: Option<&EvidenceCheckpoint>,
    measured: Option<&EvidenceCounters>,
) -> Result<IntegrityBoundScenarioQualificationReport, CheckpointBuildError> {
    Ok(IntegrityBoundScenarioQualificationReport {
        effect_bound: qualify_observer_effect_bound_scenario(
            scenario,
            &item.snapshot,
            measured,
        ),
        integrity: verify_checkpoint_chain(std::slice::from_ref(item), anchor)?,
    })
}

/// Qualify one owner-aware checkpointed snapshot against an optional external anchor.
pub fn qualify_checkpointed_owner_aware_scenario(
    scenario: &OwnerAwareScenario,
    item: &CheckpointedEffectBoundEvidence,
    anchor: Option<&EvidenceCheckpoint>,
    measured: Option<&EvidenceCounters>,
) -> Result<IntegrityBoundScenarioQualificationReport, CheckpointBuildError> {
    Ok(IntegrityBoundScenarioQualificationReport {
        effect_bound: qualify_owner_aware_effect_bound_scenario(
            scenario,
            &item.snapshot,
            measured,
        ),
        integrity: verify_checkpoint_chain(std::slice::from_ref(item), anchor)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use symthaea_cogsec_evidence::{
        CheckpointViolation, DerivedShadowMetrics, EffectBindingReport, ReconciliationReport,
    };

    use crate::{AttributionReport, ScenarioQualificationReport};

    fn clean_effect_bound_report() -> EffectBoundScenarioQualificationReport {
        EffectBoundScenarioQualificationReport {
            scenario: ScenarioQualificationReport {
                scenario_id: "test".into(),
                evidence: ReconciliationReport {
                    metrics: DerivedShadowMetrics::default(),
                    violations: Vec::new(),
                },
                event_count_violations: Vec::new(),
                contract_violations: Vec::new(),
                attribution: AttributionReport::default(),
            },
            effects: EffectBindingReport::default(),
        }
    }

    #[test]
    fn checkpoint_failure_does_not_redefine_effect_attribution() {
        let report = IntegrityBoundScenarioQualificationReport {
            effect_bound: clean_effect_bound_report(),
            integrity: CheckpointVerificationReport {
                violations: vec![CheckpointViolation::CheckpointRootMismatch {
                    checkpoint_index: 4,
                }],
            },
        };

        assert!(report.qualifies_for_effect_bound_mapped_attribution());
        assert!(!report.checkpoint_chain_is_consistent());
        assert!(!report.qualifies_for_integrity_bound_mapped_attribution());
    }

    #[test]
    fn clean_effect_and_integrity_dimensions_compose_without_weighting() {
        let report = IntegrityBoundScenarioQualificationReport {
            effect_bound: clean_effect_bound_report(),
            integrity: CheckpointVerificationReport::default(),
        };

        assert!(report.qualifies_for_integrity_bound_mapped_attribution());
        assert!(report.qualifies_for_integrity_bound_strong_attribution());
    }
}
