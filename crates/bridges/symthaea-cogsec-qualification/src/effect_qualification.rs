// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Combined scenario + portable exact-effect qualification.
//!
//! This façade prevents first-runtime qualification from accidentally using the
//! older v1 attribution predicate while forgetting the additive exact-effect
//! sidecar. Coverage, mapped attribution, taxonomy completeness, and effect
//! equality remain separate claims; none is collapsed into a weighted score.

use symthaea_cogsec_evidence::{
    EffectBindingReport, EffectBoundEvidenceSnapshot, validate_effect_bound_snapshot,
};
use symthaea_evidence_plane::EvidenceCounters;

use crate::{
    ObserverOnlyScenario, OwnerAwareScenario, ScenarioContract, ScenarioQualificationReport,
    qualify_shadow_scenario,
};

/// Combined result for one typed scenario and one effect-bound evidence snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct EffectBoundScenarioQualificationReport {
    /// Existing count/manifest/event/counter/attribution qualification.
    pub scenario: ScenarioQualificationReport,
    /// Additive portable exact-effect structural verification.
    pub effects: EffectBindingReport,
}

impl EffectBoundScenarioQualificationReport {
    /// Whether the base scenario supports its scoped observation-coverage claim.
    ///
    /// Exact-effect equality is intentionally not required for a pure coverage
    /// claim; callers must use the attribution predicates below when claiming
    /// that observed mutations are the same effects CogSec evaluated.
    pub fn qualifies_for_scoped_observation_coverage(&self) -> bool {
        self.scenario.qualifies_for_scoped_observation_coverage()
    }

    /// Whether every currently mapped attribution stage is consistent **and**
    /// every paired observed mutation has the exact evaluated effect commitment.
    pub fn qualifies_for_effect_bound_mapped_attribution(&self) -> bool {
        self.scenario.qualifies_for_mapped_attribution()
            && self.effects.effect_bindings_are_consistent()
    }

    /// Whether all-stage strong attribution qualifies and exact-effect bindings
    /// are structurally consistent.
    ///
    /// Existing taxonomy limitations such as unmapped working-state/dream stages
    /// still block this stronger claim even when digest bindings themselves pass.
    pub fn qualifies_for_effect_bound_strong_attribution(&self) -> bool {
        self.scenario.qualifies_for_strong_attribution()
            && self.effects.effect_bindings_are_consistent()
    }
}

fn qualify_effect_bound_contract(
    contract: &ScenarioContract,
    snapshot: &EffectBoundEvidenceSnapshot,
    measured: Option<&EvidenceCounters>,
) -> EffectBoundScenarioQualificationReport {
    EffectBoundScenarioQualificationReport {
        scenario: qualify_shadow_scenario(contract, &snapshot.base, measured),
        effects: validate_effect_bound_snapshot(snapshot),
    }
}

/// Qualify one early observer-only scenario against effect-bound portable evidence.
///
/// The typed scenario profile guarantees that owner-issued `ResourceVersion` is
/// not claimed during the first #206 observer-only runtime tranche.
pub fn qualify_observer_effect_bound_scenario(
    scenario: &ObserverOnlyScenario,
    snapshot: &EffectBoundEvidenceSnapshot,
    measured: Option<&EvidenceCounters>,
) -> EffectBoundScenarioQualificationReport {
    qualify_effect_bound_contract(scenario.contract(), snapshot, measured)
}

/// Qualify one later owner-aware scenario against effect-bound portable evidence.
pub fn qualify_owner_aware_effect_bound_scenario(
    scenario: &OwnerAwareScenario,
    snapshot: &EffectBoundEvidenceSnapshot,
    measured: Option<&EvidenceCounters>,
) -> EffectBoundScenarioQualificationReport {
    qualify_effect_bound_contract(scenario.contract(), snapshot, measured)
}

#[cfg(test)]
mod tests {
    use super::*;

    use symthaea_cogsec_evidence::{
        DerivedShadowMetrics, EffectBindingViolation, EventId, ReconciliationReport,
        ShadowEventKind,
    };

    use crate::{AttributionLimitation, AttributionReport};

    fn scenario_report(attribution: AttributionReport) -> ScenarioQualificationReport {
        ScenarioQualificationReport {
            scenario_id: "test".into(),
            evidence: ReconciliationReport {
                metrics: DerivedShadowMetrics::default(),
                violations: Vec::new(),
            },
            event_count_violations: Vec::new(),
            contract_violations: Vec::new(),
            attribution,
        }
    }

    #[test]
    fn effect_violation_blocks_effect_bound_attribution_without_redefining_coverage() {
        let report = EffectBoundScenarioQualificationReport {
            scenario: scenario_report(AttributionReport::default()),
            effects: EffectBindingReport {
                violations: vec![EffectBindingViolation::SnapshotSchemaMismatch { found: 99 }],
            },
        };

        assert!(report.qualifies_for_scoped_observation_coverage());
        assert!(!report.qualifies_for_effect_bound_mapped_attribution());
        assert!(!report.qualifies_for_effect_bound_strong_attribution());
    }

    #[test]
    fn taxonomy_limitation_blocks_strong_but_not_mapped_effect_attribution() {
        let mut attribution = AttributionReport::default();
        attribution
            .limitations
            .insert(AttributionLimitation::UnmappedMutationStage {
                event_id: EventId {
                    ledger_epoch: 1,
                    sequence: 1,
                },
                kind: ShadowEventKind::WorkingStateInfluenceEvaluated,
            });

        let report = EffectBoundScenarioQualificationReport {
            scenario: scenario_report(attribution),
            effects: EffectBindingReport::default(),
        };

        assert!(report.qualifies_for_effect_bound_mapped_attribution());
        assert!(!report.qualifies_for_effect_bound_strong_attribution());
    }

    #[test]
    fn fully_clean_dimensions_allow_strong_effect_bound_attribution() {
        let report = EffectBoundScenarioQualificationReport {
            scenario: scenario_report(AttributionReport::default()),
            effects: EffectBindingReport::default(),
        };
        assert!(report.qualifies_for_effect_bound_strong_attribution());
    }
}
