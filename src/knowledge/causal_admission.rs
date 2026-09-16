// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Explicit admission policy for semantic causal hypotheses.
//!
//! This module does **not** decide whether a causal claim is true. It answers a
//! narrower question: whether a hypothesis is *eligible for downstream causal-DAG
//! admission under a caller-supplied policy*. Keeping this gate explicit prevents
//! reports, simulations, repeated copies, or high-confidence prose from becoming
//! structural causal assumptions by accident.

use super::causal_hypothesis::{
    CausalEvidenceProfile, CausalHypothesisId, CausalHypothesisStore,
};
use super::claim_evidence::{EpistemicLedger, EvidenceKind, EvidencePolarity};
use super::epistemic_vector::{
    ClaimUncertaintyAssessment, UncertaintyDimension, UncertaintyValue,
};
use super::evidence_independence::EvidenceIndependenceAnalyzer;
use std::collections::BTreeSet;

/// Caller-owned admission criteria.
///
/// There is intentionally no `Default`: causal admission thresholds are policy,
/// not universal scientific constants, and must be chosen explicitly by the
/// consumer/domain.
#[derive(Debug, Clone, PartialEq)]
pub struct CausalAdmissionPolicy {
    pub min_supporting_interventions: usize,
    pub min_supporting_replications: usize,
    /// Minimum number of distinct *declared ultimate provenance roots* among
    /// supporting intervention/replication evidence.
    pub min_distinct_interventional_roots: usize,
    pub reject_on_interventional_contradiction: bool,
    pub require_mechanism_claim: bool,
    pub max_epistemic_uncertainty: Option<UncertaintyValue>,
    pub max_aleatoric_uncertainty: Option<UncertaintyValue>,
    pub max_ontological_uncertainty: Option<UncertaintyValue>,
    pub max_distribution_shift_uncertainty: Option<UncertaintyValue>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CausalAdmissionFailure {
    InsufficientInterventions { required: usize, observed: usize },
    InsufficientReplications { required: usize, observed: usize },
    InsufficientDistinctInterventionalRoots { required: usize, observed: usize },
    InterventionalContradiction { observed: usize },
    MissingMechanismClaim,
    MissingUncertaintyAssessment,
    AssessmentForDifferentClaim,
    MissingUncertaintyDimension(UncertaintyDimension),
    UncertaintyAboveLimit {
        dimension: UncertaintyDimension,
        observed: UncertaintyValue,
        limit: UncertaintyValue,
    },
}

/// Auditable result of applying one explicit policy to one hypothesis.
#[derive(Debug, Clone, PartialEq)]
pub struct CausalAdmissionDecision {
    pub hypothesis_id: CausalHypothesisId,
    pub eligible: bool,
    pub failures: Vec<CausalAdmissionFailure>,
    pub evidence_profile: CausalEvidenceProfile,
    /// Distinct declared source roots among supporting intervention/replication records.
    pub distinct_interventional_roots: usize,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CausalAdmissionGate;

impl CausalAdmissionGate {
    /// Evaluate eligibility under `policy` without mutating any hypothesis,
    /// evidence record, confidence value, or causal graph.
    pub fn evaluate(
        store: &CausalHypothesisStore,
        ledger: &EpistemicLedger,
        hypothesis_id: CausalHypothesisId,
        policy: &CausalAdmissionPolicy,
        uncertainty: Option<&ClaimUncertaintyAssessment>,
    ) -> Option<CausalAdmissionDecision> {
        let hypothesis = store.hypothesis(hypothesis_id)?;
        let profile = store.evidence_profile(ledger, hypothesis_id)?;
        let distinct_interventional_roots =
            distinct_supporting_interventional_roots(ledger, hypothesis.claim_id);

        let mut failures = Vec::new();

        if profile.supporting_interventions < policy.min_supporting_interventions {
            failures.push(CausalAdmissionFailure::InsufficientInterventions {
                required: policy.min_supporting_interventions,
                observed: profile.supporting_interventions,
            });
        }
        if profile.supporting_replications < policy.min_supporting_replications {
            failures.push(CausalAdmissionFailure::InsufficientReplications {
                required: policy.min_supporting_replications,
                observed: profile.supporting_replications,
            });
        }
        if distinct_interventional_roots < policy.min_distinct_interventional_roots {
            failures.push(
                CausalAdmissionFailure::InsufficientDistinctInterventionalRoots {
                    required: policy.min_distinct_interventional_roots,
                    observed: distinct_interventional_roots,
                },
            );
        }

        let contradicting_interventions =
            profile.contradicting_interventions + profile.contradicting_replications;
        if policy.reject_on_interventional_contradiction && contradicting_interventions > 0 {
            failures.push(CausalAdmissionFailure::InterventionalContradiction {
                observed: contradicting_interventions,
            });
        }

        if policy.require_mechanism_claim && hypothesis.mechanism_claim_ids.is_empty() {
            failures.push(CausalAdmissionFailure::MissingMechanismClaim);
        }

        let limits = [
            (
                UncertaintyDimension::Epistemic,
                policy.max_epistemic_uncertainty,
            ),
            (
                UncertaintyDimension::Aleatoric,
                policy.max_aleatoric_uncertainty,
            ),
            (
                UncertaintyDimension::Ontological,
                policy.max_ontological_uncertainty,
            ),
            (
                UncertaintyDimension::DistributionShift,
                policy.max_distribution_shift_uncertainty,
            ),
        ];
        if limits.iter().any(|(_, limit)| limit.is_some()) {
            match uncertainty {
                None => failures.push(CausalAdmissionFailure::MissingUncertaintyAssessment),
                Some(assessment) if assessment.claim_id != hypothesis.claim_id => {
                    failures.push(CausalAdmissionFailure::AssessmentForDifferentClaim)
                }
                Some(assessment) => {
                    for (dimension, limit) in limits {
                        let Some(limit) = limit else {
                            continue;
                        };
                        match assessment.vector.get(dimension) {
                            None => failures.push(
                                CausalAdmissionFailure::MissingUncertaintyDimension(dimension),
                            ),
                            Some(observed) if observed.get() > limit.get() => failures.push(
                                CausalAdmissionFailure::UncertaintyAboveLimit {
                                    dimension,
                                    observed,
                                    limit,
                                },
                            ),
                            Some(_) => {}
                        }
                    }
                }
            }
        }

        Some(CausalAdmissionDecision {
            hypothesis_id,
            eligible: failures.is_empty(),
            failures,
            evidence_profile: profile,
            distinct_interventional_roots,
        })
    }
}

/// Count distinct declared ultimate roots only among supporting intervention and
/// replication records. Reports or simulations cannot satisfy this criterion.
fn distinct_supporting_interventional_roots(
    ledger: &EpistemicLedger,
    claim_id: super::claim_evidence::ClaimId,
) -> usize {
    let report =
        EvidenceIndependenceAnalyzer::analyze(ledger, claim_id, EvidencePolarity::Supports);
    let mut roots = BTreeSet::new();

    for lineage in report.lineages {
        let Some(evidence) = ledger.evidence(lineage.evidence_id) else {
            continue;
        };
        if matches!(
            evidence.kind,
            EvidenceKind::Intervention | EvidenceKind::Replication
        ) {
            roots.extend(lineage.root_ids);
        }
    }

    roots.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::causal_hypothesis::{CausalSign, CausalHypothesisStore};
    use super::super::claim_evidence::{ClaimKind, EvidencePolarity};
    use super::super::entity_event::EntityEventStore;
    use super::super::extraction::EntityType;
    use super::super::epistemic_vector::EpistemicVector;

    struct Fixture {
        entities: EntityEventStore,
        ledger: EpistemicLedger,
        cause: super::super::entity_event::EntityId,
        effect: super::super::entity_event::EntityId,
        claim: super::super::claim_evidence::ClaimId,
    }

    fn fixture() -> Fixture {
        let mut entities = EntityEventStore::new();
        let cause = entities
            .create_entity("temperature", EntityType::Quantity, 1)
            .unwrap();
        let effect = entities
            .create_entity("evaporation", EntityType::Process, 1)
            .unwrap();
        let mut ledger = EpistemicLedger::new();
        let claim = ledger.add_claim(
            "temperature increases evaporation",
            ClaimKind::Causal,
            Some("physics".into()),
            None,
            1,
        );
        Fixture {
            entities,
            ledger,
            cause,
            effect,
            claim,
        }
    }

    fn policy() -> CausalAdmissionPolicy {
        CausalAdmissionPolicy {
            min_supporting_interventions: 1,
            min_supporting_replications: 1,
            min_distinct_interventional_roots: 2,
            reject_on_interventional_contradiction: true,
            require_mechanism_claim: false,
            max_epistemic_uncertainty: None,
            max_aleatoric_uncertainty: None,
            max_ontological_uncertainty: None,
            max_distribution_shift_uncertainty: None,
        }
    }

    fn hypothesis(f: &Fixture, mechanisms: Vec<super::super::claim_evidence::ClaimId>) -> (CausalHypothesisStore, CausalHypothesisId) {
        let mut store = CausalHypothesisStore::new();
        let id = store
            .add_hypothesis(
                &f.entities,
                &f.ledger,
                f.cause,
                f.effect,
                f.claim,
                CausalSign::Increases,
                None,
                mechanisms,
                2,
            )
            .unwrap();
        (store, id)
    }

    fn root(f: &mut Fixture, label: &str) -> super::super::claim_evidence::ProvenanceId {
        f.ledger
            .add_provenance(label, None, None, 2, vec![])
            .unwrap()
    }

    fn derived(
        f: &mut Fixture,
        label: &str,
        parents: Vec<super::super::claim_evidence::ProvenanceId>,
    ) -> super::super::claim_evidence::ProvenanceId {
        f.ledger
            .add_provenance(label, None, None, 2, parents)
            .unwrap()
    }

    fn evidence(
        f: &mut Fixture,
        kind: EvidenceKind,
        polarity: EvidencePolarity,
        provenance: super::super::claim_evidence::ProvenanceId,
    ) -> super::super::claim_evidence::EvidenceId {
        f.ledger
            .add_evidence(f.claim, kind, polarity, provenance, 3, None, None)
            .unwrap()
    }

    #[test]
    fn report_only_claim_is_not_admissible_under_interventional_policy() {
        let mut f = fixture();
        let source = root(&mut f, "paper");
        evidence(&mut f, EvidenceKind::Report, EvidencePolarity::Supports, source);
        let (store, id) = hypothesis(&f, vec![]);

        let decision = CausalAdmissionGate::evaluate(&store, &f.ledger, id, &policy(), None)
            .expect("hypothesis exists");
        assert!(!decision.eligible);
        assert_eq!(decision.evidence_profile.supporting_reports, 1);
        assert!(decision.failures.iter().any(|failure| matches!(
            failure,
            CausalAdmissionFailure::InsufficientInterventions { .. }
        )));
    }

    #[test]
    fn copied_interventional_records_do_not_satisfy_root_diversity() {
        let mut f = fixture();
        let original = root(&mut f, "original-experiment");
        let copy_a = derived(&mut f, "copy-a", vec![original]);
        let copy_b = derived(&mut f, "copy-b", vec![original]);
        evidence(
            &mut f,
            EvidenceKind::Intervention,
            EvidencePolarity::Supports,
            copy_a,
        );
        evidence(
            &mut f,
            EvidenceKind::Replication,
            EvidencePolarity::Supports,
            copy_b,
        );
        let (store, id) = hypothesis(&f, vec![]);

        let decision = CausalAdmissionGate::evaluate(&store, &f.ledger, id, &policy(), None)
            .expect("hypothesis exists");
        assert_eq!(decision.distinct_interventional_roots, 1);
        assert!(!decision.eligible);
        assert!(decision.failures.iter().any(|failure| matches!(
            failure,
            CausalAdmissionFailure::InsufficientDistinctInterventionalRoots {
                required: 2,
                observed: 1
            }
        )));
    }

    #[test]
    fn explicitly_satisfied_policy_is_eligible_not_declared_true() {
        let mut f = fixture();
        let experiment = root(&mut f, "experiment-a");
        let replication = root(&mut f, "experiment-b");
        evidence(
            &mut f,
            EvidenceKind::Intervention,
            EvidencePolarity::Supports,
            experiment,
        );
        evidence(
            &mut f,
            EvidenceKind::Replication,
            EvidencePolarity::Supports,
            replication,
        );
        let (store, id) = hypothesis(&f, vec![]);

        let decision = CausalAdmissionGate::evaluate(&store, &f.ledger, id, &policy(), None)
            .expect("hypothesis exists");
        assert!(decision.eligible);
        assert!(decision.failures.is_empty());
        assert_eq!(decision.distinct_interventional_roots, 2);
    }

    #[test]
    fn contradictory_intervention_fails_closed_when_policy_requires_it() {
        let mut f = fixture();
        let experiment = root(&mut f, "experiment-a");
        let replication = root(&mut f, "experiment-b");
        let contradiction = root(&mut f, "experiment-c");
        evidence(
            &mut f,
            EvidenceKind::Intervention,
            EvidencePolarity::Supports,
            experiment,
        );
        evidence(
            &mut f,
            EvidenceKind::Replication,
            EvidencePolarity::Supports,
            replication,
        );
        evidence(
            &mut f,
            EvidenceKind::Intervention,
            EvidencePolarity::Contradicts,
            contradiction,
        );
        let (store, id) = hypothesis(&f, vec![]);

        let decision = CausalAdmissionGate::evaluate(&store, &f.ledger, id, &policy(), None)
            .expect("hypothesis exists");
        assert!(!decision.eligible);
        assert!(decision.failures.iter().any(|failure| matches!(
            failure,
            CausalAdmissionFailure::InterventionalContradiction { observed: 1 }
        )));
    }

    #[test]
    fn uncertainty_limits_require_explicit_assessment_and_dimension() {
        let mut f = fixture();
        let experiment = root(&mut f, "experiment-a");
        let replication = root(&mut f, "experiment-b");
        let e1 = evidence(
            &mut f,
            EvidenceKind::Intervention,
            EvidencePolarity::Supports,
            experiment,
        );
        let e2 = evidence(
            &mut f,
            EvidenceKind::Replication,
            EvidencePolarity::Supports,
            replication,
        );
        let (store, id) = hypothesis(&f, vec![]);
        let mut p = policy();
        p.max_ontological_uncertainty = Some(UncertaintyValue::new(0.4).unwrap());

        let missing = CausalAdmissionGate::evaluate(&store, &f.ledger, id, &p, None).unwrap();
        assert!(missing.failures.contains(&CausalAdmissionFailure::MissingUncertaintyAssessment));

        let vector = EpistemicVector::new()
            .with(UncertaintyDimension::Ontological, 0.7)
            .unwrap();
        let assessment = ClaimUncertaintyAssessment::new(
            &f.ledger,
            f.claim,
            vector,
            vec![e1, e2],
            4,
        )
        .unwrap();
        let high = CausalAdmissionGate::evaluate(
            &store,
            &f.ledger,
            id,
            &p,
            Some(&assessment),
        )
        .unwrap();
        assert!(!high.eligible);
        assert!(high.failures.iter().any(|failure| matches!(
            failure,
            CausalAdmissionFailure::UncertaintyAboveLimit {
                dimension: UncertaintyDimension::Ontological,
                ..
            }
        )));
    }
}
