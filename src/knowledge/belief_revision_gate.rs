// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Proposal-only gate for evidence-grounded epistemic-support revisions.
//!
//! EKM-023 separates epistemic support from accessibility, retention, and
//! consolidation. This module evaluates whether a proposed epistemic-support
//! delta is admissible under an explicit caller policy. It inspects the actual
//! ledger evidence, declared provenance ancestry, contradiction state, causal
//! evidence kind, uncertainty freshness, and calibration state.
//!
//! There is deliberately no apply/update method here.

use super::claim_evidence::{
    ClaimId, ClaimKind, EpistemicLedger, EvidenceId, EvidenceKind, EvidencePolarity, ProvenanceId,
};
use super::epistemic_vector::{ClaimUncertaintyAssessment, UncertaintyDimension};
use super::knowledge_weight_routing::{
    KnowledgeWeightAuthorityRouter, KnowledgeWeightDimension, KnowledgeWeightError,
    KnowledgeWeightRoutingFailure, KnowledgeWeightSource, KnowledgeWeightUpdateProposal,
};
use std::collections::{BTreeSet, HashSet};

#[derive(Debug, Clone, PartialEq)]
pub struct EpistemicRevisionProposal {
    pub claim_id: ClaimId,
    pub update: KnowledgeWeightUpdateProposal,
}

impl EpistemicRevisionProposal {
    pub fn new(
        claim_id: ClaimId,
        delta: f32,
        basis_evidence_ids: Vec<EvidenceId>,
        rationale: impl Into<String>,
    ) -> Result<Self, KnowledgeWeightError> {
        Ok(Self {
            claim_id,
            update: KnowledgeWeightUpdateProposal::new(
                KnowledgeWeightDimension::EpistemicSupport,
                KnowledgeWeightSource::AdmittedEvidence,
                delta,
                basis_evidence_ids,
                rationale,
            )?,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibrationSnapshot {
    pub sample_count: u64,
    pub ece: f64,
}

impl CalibrationSnapshot {
    pub fn new(sample_count: u64, ece: f64) -> Result<Self, BeliefRevisionPolicyError> {
        if !ece.is_finite() || !(0.0..=1.0).contains(&ece) {
            return Err(BeliefRevisionPolicyError::InvalidCalibrationEce(ece));
        }
        Ok(Self { sample_count, ece })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BeliefRevisionPolicy {
    max_abs_delta: f32,
    min_declared_provenance_roots: usize,
    require_calibration: bool,
    min_calibration_samples: u64,
    max_calibration_ece: f64,
    require_uncertainty_assessment: bool,
    require_current_uncertainty: bool,
    block_strengthen_with_unresolved_contradictions: bool,
    require_intervention_for_causal_strengthen: bool,
    strengthen_uncertainty_caps: Vec<(UncertaintyDimension, f64)>,
}

impl BeliefRevisionPolicy {
    pub fn new(
        max_abs_delta: f32,
        min_declared_provenance_roots: usize,
        require_calibration: bool,
        min_calibration_samples: u64,
        max_calibration_ece: f64,
    ) -> Result<Self, BeliefRevisionPolicyError> {
        if !max_abs_delta.is_finite() || !(0.0..=1.0).contains(&max_abs_delta) {
            return Err(BeliefRevisionPolicyError::InvalidMaxAbsDelta(max_abs_delta));
        }
        if !max_calibration_ece.is_finite() || !(0.0..=1.0).contains(&max_calibration_ece) {
            return Err(BeliefRevisionPolicyError::InvalidCalibrationEce(
                max_calibration_ece,
            ));
        }
        Ok(Self {
            max_abs_delta,
            min_declared_provenance_roots,
            require_calibration,
            min_calibration_samples,
            max_calibration_ece,
            require_uncertainty_assessment: false,
            require_current_uncertainty: false,
            block_strengthen_with_unresolved_contradictions: false,
            require_intervention_for_causal_strengthen: false,
            strengthen_uncertainty_caps: Vec::new(),
        })
    }

    pub fn with_uncertainty_requirements(
        mut self,
        require_assessment: bool,
        require_current: bool,
    ) -> Self {
        self.require_uncertainty_assessment = require_assessment;
        self.require_current_uncertainty = require_current;
        self
    }

    pub fn block_strengthen_with_unresolved_contradictions(mut self, block: bool) -> Self {
        self.block_strengthen_with_unresolved_contradictions = block;
        self
    }

    pub fn require_intervention_for_causal_strengthen(mut self, require: bool) -> Self {
        self.require_intervention_for_causal_strengthen = require;
        self
    }

    pub fn with_strengthen_uncertainty_cap(
        mut self,
        dimension: UncertaintyDimension,
        maximum: f64,
    ) -> Result<Self, BeliefRevisionPolicyError> {
        if !maximum.is_finite() || !(0.0..=1.0).contains(&maximum) {
            return Err(BeliefRevisionPolicyError::InvalidUncertaintyCap(maximum));
        }
        if let Some(existing) = self
            .strengthen_uncertainty_caps
            .iter_mut()
            .find(|(candidate, _)| *candidate == dimension)
        {
            existing.1 = maximum;
        } else {
            self.strengthen_uncertainty_caps.push((dimension, maximum));
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BeliefRevisionPolicyError {
    InvalidMaxAbsDelta(f32),
    InvalidCalibrationEce(f64),
    InvalidUncertaintyCap(f64),
}

#[derive(Debug, Clone, PartialEq)]
pub enum BeliefRevisionFailure {
    WeightRoutingDenied(Vec<KnowledgeWeightRoutingFailure>),
    UnknownClaim(ClaimId),
    UnknownEvidence(EvidenceId),
    EvidenceForDifferentClaim {
        evidence_id: EvidenceId,
        expected_claim: ClaimId,
        actual_claim: ClaimId,
    },
    PositiveDeltaLacksSupportingEvidence,
    PositiveDeltaIncludesContradictingEvidence,
    NegativeDeltaLacksContradictingEvidence,
    NegativeDeltaIncludesSupportingEvidence,
    DeclaredProvenanceRootsBelowMinimum {
        required: usize,
        actual: usize,
    },
    DeltaExceedsPolicy {
        maximum: f32,
        actual: f32,
    },
    CalibrationMissing,
    CalibrationSamplesBelowMinimum {
        required: u64,
        actual: u64,
    },
    CalibrationEceAboveMaximum {
        maximum: f64,
        actual: f64,
    },
    UncertaintyAssessmentMissing,
    UncertaintyAssessmentForDifferentClaim {
        expected_claim: ClaimId,
        actual_claim: ClaimId,
    },
    UncertaintyAssessmentPredatesEvidence {
        assessment_cycle: u64,
        latest_evidence_cycle: u64,
    },
    RequiredUncertaintyDimensionUnassessed(UncertaintyDimension),
    UncertaintyAboveMaximum {
        dimension: UncertaintyDimension,
        maximum: f64,
        actual: f64,
    },
    UnresolvedContradictionBlocksStrengthen {
        count: usize,
    },
    CausalStrengthenLacksInterventionalBasis,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BeliefRevisionDecision {
    eligible: bool,
    declared_provenance_root_count: usize,
    failures: Vec<BeliefRevisionFailure>,
}

impl BeliefRevisionDecision {
    pub fn eligible(&self) -> bool {
        self.eligible
    }

    pub fn declared_provenance_root_count(&self) -> usize {
        self.declared_provenance_root_count
    }

    pub fn failures(&self) -> &[BeliefRevisionFailure] {
        &self.failures
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BeliefRevisionGate;

impl BeliefRevisionGate {
    pub fn evaluate(
        ledger: &EpistemicLedger,
        proposal: &EpistemicRevisionProposal,
        policy: &BeliefRevisionPolicy,
        calibration: Option<CalibrationSnapshot>,
        uncertainty: Option<&ClaimUncertaintyAssessment>,
    ) -> BeliefRevisionDecision {
        let mut failures = Vec::new();

        let routing = KnowledgeWeightAuthorityRouter::evaluate(&proposal.update);
        if !routing.eligible() {
            failures.push(BeliefRevisionFailure::WeightRoutingDenied(
                routing.failures().to_vec(),
            ));
        }

        let claim = match ledger.claim(proposal.claim_id) {
            Some(claim) => claim,
            None => {
                failures.push(BeliefRevisionFailure::UnknownClaim(proposal.claim_id));
                return BeliefRevisionDecision {
                    eligible: false,
                    declared_provenance_root_count: 0,
                    failures,
                };
            }
        };

        let delta = proposal.update.delta.get();
        if delta.abs() > policy.max_abs_delta {
            failures.push(BeliefRevisionFailure::DeltaExceedsPolicy {
                maximum: policy.max_abs_delta,
                actual: delta.abs(),
            });
        }

        let mut supporting = 0usize;
        let mut contradicting = 0usize;
        let mut interventional_support = 0usize;
        let mut provenance_roots = BTreeSet::new();

        for evidence_id in &proposal.update.basis_evidence_ids {
            let Some(evidence) = ledger.evidence(*evidence_id) else {
                failures.push(BeliefRevisionFailure::UnknownEvidence(*evidence_id));
                continue;
            };
            if evidence.claim_id != proposal.claim_id {
                failures.push(BeliefRevisionFailure::EvidenceForDifferentClaim {
                    evidence_id: *evidence_id,
                    expected_claim: proposal.claim_id,
                    actual_claim: evidence.claim_id,
                });
                continue;
            }

            match evidence.polarity {
                EvidencePolarity::Supports => {
                    supporting += 1;
                    if matches!(evidence.kind, EvidenceKind::Intervention | EvidenceKind::Replication)
                    {
                        interventional_support += 1;
                    }
                }
                EvidencePolarity::Contradicts => contradicting += 1,
                EvidencePolarity::Contextualizes => {}
            }
            provenance_roots.extend(declared_roots(ledger, evidence.provenance_id));
        }

        if delta > 0.0 {
            if supporting == 0 {
                failures.push(BeliefRevisionFailure::PositiveDeltaLacksSupportingEvidence);
            }
            if contradicting > 0 {
                failures.push(BeliefRevisionFailure::PositiveDeltaIncludesContradictingEvidence);
            }
        } else if delta < 0.0 {
            if contradicting == 0 {
                failures.push(BeliefRevisionFailure::NegativeDeltaLacksContradictingEvidence);
            }
            if supporting > 0 {
                failures.push(BeliefRevisionFailure::NegativeDeltaIncludesSupportingEvidence);
            }
        }

        if provenance_roots.len() < policy.min_declared_provenance_roots {
            failures.push(BeliefRevisionFailure::DeclaredProvenanceRootsBelowMinimum {
                required: policy.min_declared_provenance_roots,
                actual: provenance_roots.len(),
            });
        }

        if delta != 0.0 && policy.require_calibration {
            match calibration {
                None => failures.push(BeliefRevisionFailure::CalibrationMissing),
                Some(snapshot) => {
                    if snapshot.sample_count < policy.min_calibration_samples {
                        failures.push(BeliefRevisionFailure::CalibrationSamplesBelowMinimum {
                            required: policy.min_calibration_samples,
                            actual: snapshot.sample_count,
                        });
                    }
                    if snapshot.ece > policy.max_calibration_ece {
                        failures.push(BeliefRevisionFailure::CalibrationEceAboveMaximum {
                            maximum: policy.max_calibration_ece,
                            actual: snapshot.ece,
                        });
                    }
                }
            }
        }

        if delta > 0.0 && policy.block_strengthen_with_unresolved_contradictions {
            let contradiction_count = ledger
                .evidence_for_claim(proposal.claim_id)
                .into_iter()
                .filter(|record| record.polarity == EvidencePolarity::Contradicts)
                .count();
            if contradiction_count > 0 {
                failures.push(BeliefRevisionFailure::UnresolvedContradictionBlocksStrengthen {
                    count: contradiction_count,
                });
            }
        }

        if delta > 0.0
            && claim.kind == ClaimKind::Causal
            && policy.require_intervention_for_causal_strengthen
            && interventional_support == 0
        {
            failures.push(BeliefRevisionFailure::CausalStrengthenLacksInterventionalBasis);
        }

        if policy.require_uncertainty_assessment
            || policy.require_current_uncertainty
            || !policy.strengthen_uncertainty_caps.is_empty()
        {
            match uncertainty {
                None => failures.push(BeliefRevisionFailure::UncertaintyAssessmentMissing),
                Some(assessment) => {
                    if assessment.claim_id != proposal.claim_id {
                        failures.push(BeliefRevisionFailure::UncertaintyAssessmentForDifferentClaim {
                            expected_claim: proposal.claim_id,
                            actual_claim: assessment.claim_id,
                        });
                    } else {
                        if policy.require_current_uncertainty {
                            if let Some(latest_evidence_cycle) = ledger
                                .evidence_for_claim(proposal.claim_id)
                                .into_iter()
                                .map(|record| record.observed_at_cycle)
                                .max()
                            {
                                if latest_evidence_cycle > assessment.assessed_at_cycle {
                                    failures.push(
                                        BeliefRevisionFailure::UncertaintyAssessmentPredatesEvidence {
                                            assessment_cycle: assessment.assessed_at_cycle,
                                            latest_evidence_cycle,
                                        },
                                    );
                                }
                            }
                        }

                        if delta > 0.0 {
                            for (dimension, maximum) in &policy.strengthen_uncertainty_caps {
                                match assessment.vector.get(*dimension) {
                                    None => failures.push(
                                        BeliefRevisionFailure::RequiredUncertaintyDimensionUnassessed(
                                            *dimension,
                                        ),
                                    ),
                                    Some(value) if value.get() > *maximum => failures.push(
                                        BeliefRevisionFailure::UncertaintyAboveMaximum {
                                            dimension: *dimension,
                                            maximum: *maximum,
                                            actual: value.get(),
                                        },
                                    ),
                                    Some(_) => {}
                                }
                            }
                        }
                    }
                }
            }
        }

        BeliefRevisionDecision {
            eligible: failures.is_empty(),
            declared_provenance_root_count: provenance_roots.len(),
            failures,
        }
    }
}

fn declared_roots(ledger: &EpistemicLedger, start: ProvenanceId) -> BTreeSet<ProvenanceId> {
    fn walk(
        ledger: &EpistemicLedger,
        current: ProvenanceId,
        visited: &mut HashSet<ProvenanceId>,
        roots: &mut BTreeSet<ProvenanceId>,
    ) {
        if !visited.insert(current) {
            return;
        }
        let Some(record) = ledger.provenance(current) else {
            return;
        };
        if record.parent_ids.is_empty() {
            roots.insert(current);
            return;
        }
        for parent in &record.parent_ids {
            walk(ledger, *parent, visited, roots);
        }
    }

    let mut visited = HashSet::new();
    let mut roots = BTreeSet::new();
    walk(ledger, start, &mut visited, &mut roots);
    roots
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{EpistemicVector, UncertaintyDimension};

    fn base_policy() -> BeliefRevisionPolicy {
        BeliefRevisionPolicy::new(0.20, 1, true, 10, 0.20).unwrap()
    }

    fn root(ledger: &mut EpistemicLedger, label: &str) -> ProvenanceId {
        ledger
            .add_provenance(label, None, None, 1, vec![])
            .unwrap()
    }

    fn add_evidence(
        ledger: &mut EpistemicLedger,
        claim: ClaimId,
        provenance: ProvenanceId,
        kind: EvidenceKind,
        polarity: EvidencePolarity,
        cycle: u64,
    ) -> EvidenceId {
        ledger
            .add_evidence(claim, kind, polarity, provenance, cycle, None, None)
            .unwrap()
    }

    #[test]
    fn supporting_evidence_can_pass_shadow_revision_gate() {
        let mut ledger = EpistemicLedger::new();
        let source = root(&mut ledger, "lab-a");
        let claim = ledger.add_claim("X predicts Y", ClaimKind::Predictive, None, None, 1);
        let evidence = add_evidence(
            &mut ledger,
            claim,
            source,
            EvidenceKind::Measurement,
            EvidencePolarity::Supports,
            2,
        );
        let proposal = EpistemicRevisionProposal::new(claim, 0.10, vec![evidence], "new measurement")
            .unwrap();
        let calibration = CalibrationSnapshot::new(100, 0.10).unwrap();
        let decision = BeliefRevisionGate::evaluate(
            &ledger,
            &proposal,
            &base_policy(),
            Some(calibration),
            None,
        );
        assert!(decision.eligible(), "{:?}", decision.failures());
    }

    #[test]
    fn zero_sample_zero_ece_is_not_treated_as_calibrated() {
        let mut ledger = EpistemicLedger::new();
        let source = root(&mut ledger, "lab-a");
        let claim = ledger.add_claim("X predicts Y", ClaimKind::Predictive, None, None, 1);
        let evidence = add_evidence(
            &mut ledger,
            claim,
            source,
            EvidenceKind::Measurement,
            EvidencePolarity::Supports,
            2,
        );
        let proposal = EpistemicRevisionProposal::new(claim, 0.10, vec![evidence], "measurement")
            .unwrap();
        let calibration = CalibrationSnapshot::new(0, 0.0).unwrap();
        let decision = BeliefRevisionGate::evaluate(
            &ledger,
            &proposal,
            &base_policy(),
            Some(calibration),
            None,
        );
        assert!(decision.failures().contains(
            &BeliefRevisionFailure::CalibrationSamplesBelowMinimum {
                required: 10,
                actual: 0,
            }
        ));
    }

    #[test]
    fn positive_revision_cannot_use_contradicting_basis() {
        let mut ledger = EpistemicLedger::new();
        let source = root(&mut ledger, "lab-a");
        let claim = ledger.add_claim("X predicts Y", ClaimKind::Predictive, None, None, 1);
        let evidence = add_evidence(
            &mut ledger,
            claim,
            source,
            EvidenceKind::Measurement,
            EvidencePolarity::Contradicts,
            2,
        );
        let proposal = EpistemicRevisionProposal::new(claim, 0.10, vec![evidence], "bad direction")
            .unwrap();
        let decision = BeliefRevisionGate::evaluate(
            &ledger,
            &proposal,
            &base_policy(),
            Some(CalibrationSnapshot::new(100, 0.10).unwrap()),
            None,
        );
        assert!(decision
            .failures()
            .contains(&BeliefRevisionFailure::PositiveDeltaLacksSupportingEvidence));
        assert!(decision.failures().contains(
            &BeliefRevisionFailure::PositiveDeltaIncludesContradictingEvidence
        ));
    }

    #[test]
    fn copied_reports_do_not_satisfy_two_root_policy() {
        let mut ledger = EpistemicLedger::new();
        let original = root(&mut ledger, "original");
        let copy_a = ledger
            .add_provenance("copy-a", None, None, 2, vec![original])
            .unwrap();
        let copy_b = ledger
            .add_provenance("copy-b", None, None, 2, vec![original])
            .unwrap();
        let claim = ledger.add_claim("X exists", ClaimKind::Descriptive, None, None, 1);
        let a = add_evidence(
            &mut ledger,
            claim,
            copy_a,
            EvidenceKind::Report,
            EvidencePolarity::Supports,
            3,
        );
        let b = add_evidence(
            &mut ledger,
            claim,
            copy_b,
            EvidenceKind::Report,
            EvidencePolarity::Supports,
            3,
        );
        let policy = BeliefRevisionPolicy::new(0.20, 2, false, 0, 1.0).unwrap();
        let proposal = EpistemicRevisionProposal::new(claim, 0.10, vec![a, b], "two copies")
            .unwrap();
        let decision = BeliefRevisionGate::evaluate(&ledger, &proposal, &policy, None, None);
        assert_eq!(decision.declared_provenance_root_count(), 1);
        assert!(decision.failures().contains(
            &BeliefRevisionFailure::DeclaredProvenanceRootsBelowMinimum {
                required: 2,
                actual: 1,
            }
        ));
    }

    #[test]
    fn causal_strengthen_can_require_interventional_basis() {
        let mut ledger = EpistemicLedger::new();
        let source = root(&mut ledger, "paper");
        let claim = ledger.add_claim("A causes B", ClaimKind::Causal, None, None, 1);
        let report = add_evidence(
            &mut ledger,
            claim,
            source,
            EvidenceKind::Report,
            EvidencePolarity::Supports,
            2,
        );
        let policy = BeliefRevisionPolicy::new(0.20, 1, false, 0, 1.0)
            .unwrap()
            .require_intervention_for_causal_strengthen(true);
        let proposal = EpistemicRevisionProposal::new(claim, 0.10, vec![report], "causal report")
            .unwrap();
        let denied = BeliefRevisionGate::evaluate(&ledger, &proposal, &policy, None, None);
        assert!(denied.failures().contains(
            &BeliefRevisionFailure::CausalStrengthenLacksInterventionalBasis
        ));

        let intervention = add_evidence(
            &mut ledger,
            claim,
            source,
            EvidenceKind::Intervention,
            EvidencePolarity::Supports,
            3,
        );
        let proposal = EpistemicRevisionProposal::new(
            claim,
            0.10,
            vec![intervention],
            "interventional support",
        )
        .unwrap();
        let admitted = BeliefRevisionGate::evaluate(&ledger, &proposal, &policy, None, None);
        assert!(admitted.eligible(), "{:?}", admitted.failures());
    }

    #[test]
    fn stale_uncertainty_assessment_blocks_revision_when_policy_requires_current() {
        let mut ledger = EpistemicLedger::new();
        let first_source = root(&mut ledger, "first");
        let second_source = root(&mut ledger, "second");
        let claim = ledger.add_claim("X predicts Y", ClaimKind::Predictive, None, None, 1);
        let first = add_evidence(
            &mut ledger,
            claim,
            first_source,
            EvidenceKind::Observation,
            EvidencePolarity::Supports,
            2,
        );
        let assessment = ClaimUncertaintyAssessment::new(
            &ledger,
            claim,
            EpistemicVector::new()
                .with(UncertaintyDimension::Epistemic, 0.2)
                .unwrap(),
            vec![first],
            3,
        )
        .unwrap();
        let later = add_evidence(
            &mut ledger,
            claim,
            second_source,
            EvidenceKind::Measurement,
            EvidencePolarity::Supports,
            7,
        );
        let proposal = EpistemicRevisionProposal::new(claim, 0.10, vec![later], "newer evidence")
            .unwrap();
        let policy = BeliefRevisionPolicy::new(0.20, 1, false, 0, 1.0)
            .unwrap()
            .with_uncertainty_requirements(true, true);
        let decision = BeliefRevisionGate::evaluate(
            &ledger,
            &proposal,
            &policy,
            None,
            Some(&assessment),
        );
        assert!(decision.failures().contains(
            &BeliefRevisionFailure::UncertaintyAssessmentPredatesEvidence {
                assessment_cycle: 3,
                latest_evidence_cycle: 7,
            }
        ));
    }

    #[test]
    fn current_uncertainty_requirement_alone_requires_assessment() {
        let mut ledger = EpistemicLedger::new();
        let source = root(&mut ledger, "measurement");
        let claim = ledger.add_claim("X exists", ClaimKind::Descriptive, None, None, 1);
        let evidence = add_evidence(
            &mut ledger,
            claim,
            source,
            EvidenceKind::Measurement,
            EvidencePolarity::Supports,
            2,
        );
        let proposal = EpistemicRevisionProposal::new(claim, 0.10, vec![evidence], "measurement")
            .unwrap();
        let policy = BeliefRevisionPolicy::new(0.20, 1, false, 0, 1.0)
            .unwrap()
            .with_uncertainty_requirements(false, true);
        let decision = BeliefRevisionGate::evaluate(&ledger, &proposal, &policy, None, None);
        assert!(decision
            .failures()
            .contains(&BeliefRevisionFailure::UncertaintyAssessmentMissing));
    }

    #[test]
    fn ontological_uncertainty_cap_can_block_strengthening() {
        let mut ledger = EpistemicLedger::new();
        let source = root(&mut ledger, "measurement");
        let claim = ledger.add_claim("X exists", ClaimKind::Descriptive, None, None, 1);
        let evidence = add_evidence(
            &mut ledger,
            claim,
            source,
            EvidenceKind::Measurement,
            EvidencePolarity::Supports,
            2,
        );
        let assessment = ClaimUncertaintyAssessment::new(
            &ledger,
            claim,
            EpistemicVector::new()
                .with(UncertaintyDimension::Ontological, 0.8)
                .unwrap(),
            vec![evidence],
            2,
        )
        .unwrap();
        let proposal = EpistemicRevisionProposal::new(claim, 0.10, vec![evidence], "measurement")
            .unwrap();
        let policy = BeliefRevisionPolicy::new(0.20, 1, false, 0, 1.0)
            .unwrap()
            .with_uncertainty_requirements(true, false)
            .with_strengthen_uncertainty_cap(UncertaintyDimension::Ontological, 0.4)
            .unwrap();
        let decision = BeliefRevisionGate::evaluate(
            &ledger,
            &proposal,
            &policy,
            None,
            Some(&assessment),
        );
        assert!(decision.failures().contains(
            &BeliefRevisionFailure::UncertaintyAboveMaximum {
                dimension: UncertaintyDimension::Ontological,
                maximum: 0.4,
                actual: 0.8,
            }
        ));
    }
}
