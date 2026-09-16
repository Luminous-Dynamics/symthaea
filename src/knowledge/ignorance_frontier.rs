// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Diagnostic ignorance frontier for explicit knowledge gaps.
//!
//! The frontier describes missing, conflicting, stale, or provenance-concentrated
//! epistemic support for an explicitly supplied claim set. It does not browse,
//! run experiments, rank research priorities, change confidence, or trigger
//! autonomous learning.

use super::claim_evidence::{ClaimId, ClaimKind, EpistemicLedger, EvidencePolarity};
use super::epistemic_vector::{ClaimUncertaintyAssessment, UncertaintyDimension};
use super::evidence_independence::EvidenceIndependenceAnalyzer;
use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fmt;

const UNCERTAINTY_DIMENSIONS: [UncertaintyDimension; 4] = [
    UncertaintyDimension::Epistemic,
    UncertaintyDimension::Aleatoric,
    UncertaintyDimension::Ontological,
    UncertaintyDimension::DistributionShift,
];

/// One explicit gap in the support or uncertainty accounting for a claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KnowledgeGap {
    /// The claim has no attached evidence records at all.
    NoEvidence,
    /// Evidence exists, but none of it is marked as supporting the claim.
    NoSupportingEvidence,
    /// Supporting records share declared ultimate provenance roots.
    ///
    /// This indicates source concentration only; distinct roots would still not
    /// prove statistical or institutional independence.
    SharedSupportingProvenanceAncestry {
        shared_pair_count: usize,
        supporting_evidence_count: usize,
        distinct_root_count: usize,
    },
    /// Contradicting evidence exists and remains unresolved in the ledger.
    ContradictoryEvidence { count: usize },
    /// A causal claim has no supporting Intervention/Replication record.
    NoInterventionalSupport,
    /// No multidimensional uncertainty assessment has been recorded.
    NoUncertaintyAssessment,
    /// One uncertainty component is explicitly unassessed (`None`, not zero).
    UnassessedUncertaintyDimension(UncertaintyDimension),
    /// An assessment exists but names no evidence basis.
    AssessmentHasNoEvidenceBasis,
    /// New evidence arrived after the latest supplied uncertainty assessment.
    AssessmentPredatesEvidence {
        assessment_cycle: u64,
        latest_evidence_cycle: u64,
    },
}

/// Descriptive ignorance profile for one claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimIgnoranceProfile {
    pub claim_id: ClaimId,
    pub evidence_count: usize,
    pub supporting_evidence_count: usize,
    pub supporting_distinct_provenance_root_count: usize,
    pub contradicting_evidence_count: usize,
    pub interventional_support_count: usize,
    pub assessed_uncertainty_dimension_count: usize,
    pub gaps: Vec<KnowledgeGap>,
}

/// Frontier report over the exact claim scope supplied by the caller.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IgnoranceFrontierReport {
    pub profiles: Vec<ClaimIgnoranceProfile>,
    pub total_gap_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IgnoranceFrontierError {
    UnknownClaim(ClaimId),
    DuplicateClaim(ClaimId),
    AmbiguousAssessment {
        claim_id: ClaimId,
        assessed_at_cycle: u64,
    },
}

impl fmt::Display for IgnoranceFrontierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownClaim(id) => write!(f, "unknown claim id {}", id.0),
            Self::DuplicateClaim(id) => write!(f, "claim {} appears more than once", id.0),
            Self::AmbiguousAssessment {
                claim_id,
                assessed_at_cycle,
            } => write!(
                f,
                "claim {} has multiple uncertainty assessments at cycle {}",
                claim_id.0, assessed_at_cycle
            ),
        }
    }
}

impl Error for IgnoranceFrontierError {}

/// Pure diagnostic builder for claim-scoped ignorance accounting.
pub struct IgnoranceFrontier;

impl IgnoranceFrontier {
    /// Inspect an explicit claim scope using the latest supplied uncertainty
    /// assessment for each claim.
    ///
    /// Historical assessments may be supplied together; the highest cycle wins.
    /// Equal-cycle duplicates fail closed instead of relying on input order.
    pub fn inspect(
        ledger: &EpistemicLedger,
        claim_ids: &[ClaimId],
        assessments: &[ClaimUncertaintyAssessment],
    ) -> Result<IgnoranceFrontierReport, IgnoranceFrontierError> {
        let mut seen_claims = HashSet::with_capacity(claim_ids.len());
        for claim_id in claim_ids.iter().copied() {
            if !seen_claims.insert(claim_id) {
                return Err(IgnoranceFrontierError::DuplicateClaim(claim_id));
            }
            if ledger.claim(claim_id).is_none() {
                return Err(IgnoranceFrontierError::UnknownClaim(claim_id));
            }
        }

        let scope: HashSet<ClaimId> = claim_ids.iter().copied().collect();
        let mut latest_assessment: BTreeMap<ClaimId, &ClaimUncertaintyAssessment> = BTreeMap::new();
        for assessment in assessments {
            if !scope.contains(&assessment.claim_id) {
                continue;
            }
            match latest_assessment.get(&assessment.claim_id).copied() {
                None => {
                    latest_assessment.insert(assessment.claim_id, assessment);
                }
                Some(existing) if assessment.assessed_at_cycle > existing.assessed_at_cycle => {
                    latest_assessment.insert(assessment.claim_id, assessment);
                }
                Some(existing) if assessment.assessed_at_cycle == existing.assessed_at_cycle => {
                    return Err(IgnoranceFrontierError::AmbiguousAssessment {
                        claim_id: assessment.claim_id,
                        assessed_at_cycle: assessment.assessed_at_cycle,
                    });
                }
                Some(_) => {}
            }
        }

        let mut profiles = Vec::with_capacity(claim_ids.len());
        for claim_id in claim_ids.iter().copied() {
            let claim = ledger
                .claim(claim_id)
                .ok_or(IgnoranceFrontierError::UnknownClaim(claim_id))?;
            let evidence = ledger.evidence_for_claim(claim_id);
            let evidence_count = evidence.len();
            let supporting_evidence_count = evidence
                .iter()
                .filter(|record| record.polarity == EvidencePolarity::Supports)
                .count();
            let contradicting_evidence_count = evidence
                .iter()
                .filter(|record| record.polarity == EvidencePolarity::Contradicts)
                .count();
            let interventional_support_count = ledger.interventional_support_count(claim_id);
            let supporting_provenance = EvidenceIndependenceAnalyzer::analyze(
                ledger,
                claim_id,
                EvidencePolarity::Supports,
            );
            let supporting_distinct_provenance_root_count = supporting_provenance.distinct_root_count;

            let mut gaps = Vec::new();
            if evidence_count == 0 {
                gaps.push(KnowledgeGap::NoEvidence);
            } else if supporting_evidence_count == 0 {
                gaps.push(KnowledgeGap::NoSupportingEvidence);
            }
            if supporting_provenance.has_shared_ancestry() {
                gaps.push(KnowledgeGap::SharedSupportingProvenanceAncestry {
                    shared_pair_count: supporting_provenance.shared_ancestry.len(),
                    supporting_evidence_count,
                    distinct_root_count: supporting_provenance.distinct_root_count,
                });
            }
            if contradicting_evidence_count > 0 {
                gaps.push(KnowledgeGap::ContradictoryEvidence {
                    count: contradicting_evidence_count,
                });
            }
            if claim.kind == ClaimKind::Causal && interventional_support_count == 0 {
                gaps.push(KnowledgeGap::NoInterventionalSupport);
            }

            let assessed_uncertainty_dimension_count =
                if let Some(assessment) = latest_assessment.get(&claim_id).copied() {
                    for dimension in UNCERTAINTY_DIMENSIONS {
                        if assessment.vector.get(dimension).is_none() {
                            gaps.push(KnowledgeGap::UnassessedUncertaintyDimension(dimension));
                        }
                    }
                    if assessment.basis_evidence_ids.is_empty() {
                        gaps.push(KnowledgeGap::AssessmentHasNoEvidenceBasis);
                    }
                    if let Some(latest_evidence_cycle) = evidence
                        .iter()
                        .map(|record| record.observed_at_cycle)
                        .max()
                    {
                        if latest_evidence_cycle > assessment.assessed_at_cycle {
                            gaps.push(KnowledgeGap::AssessmentPredatesEvidence {
                                assessment_cycle: assessment.assessed_at_cycle,
                                latest_evidence_cycle,
                            });
                        }
                    }
                    assessment.vector.assessed_dimension_count()
                } else {
                    gaps.push(KnowledgeGap::NoUncertaintyAssessment);
                    0
                };

            profiles.push(ClaimIgnoranceProfile {
                claim_id,
                evidence_count,
                supporting_evidence_count,
                supporting_distinct_provenance_root_count,
                contradicting_evidence_count,
                interventional_support_count,
                assessed_uncertainty_dimension_count,
                gaps,
            });
        }

        let total_gap_count = profiles.iter().map(|profile| profile.gaps.len()).sum();
        Ok(IgnoranceFrontierReport {
            profiles,
            total_gap_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{EpistemicVector, EvidenceKind, UncertaintyDimension};

    fn source(ledger: &mut EpistemicLedger, label: &str) -> crate::knowledge::ProvenanceId {
        ledger
            .add_provenance(label, None, None, 1, vec![])
            .unwrap()
    }

    #[test]
    fn empty_causal_claim_exposes_missing_evidence_and_assessment() {
        let mut ledger = EpistemicLedger::new();
        let claim = ledger.add_claim("A causes B", ClaimKind::Causal, None, None, 1);

        let report = IgnoranceFrontier::inspect(&ledger, &[claim], &[]).unwrap();
        let profile = &report.profiles[0];
        assert!(profile.gaps.contains(&KnowledgeGap::NoEvidence));
        assert!(profile
            .gaps
            .contains(&KnowledgeGap::NoInterventionalSupport));
        assert!(profile
            .gaps
            .contains(&KnowledgeGap::NoUncertaintyAssessment));
        assert_eq!(profile.interventional_support_count, 0);
        assert_eq!(profile.supporting_distinct_provenance_root_count, 0);
    }

    #[test]
    fn report_only_causal_support_remains_on_ignorance_frontier() {
        let mut ledger = EpistemicLedger::new();
        let provenance = source(&mut ledger, "paper");
        let claim = ledger.add_claim("A causes B", ClaimKind::Causal, None, None, 1);
        ledger
            .add_evidence(
                claim,
                EvidenceKind::Report,
                EvidencePolarity::Supports,
                provenance,
                2,
                None,
                None,
            )
            .unwrap();

        let report = IgnoranceFrontier::inspect(&ledger, &[claim], &[]).unwrap();
        let profile = &report.profiles[0];
        assert!(!profile.gaps.contains(&KnowledgeGap::NoEvidence));
        assert!(profile
            .gaps
            .contains(&KnowledgeGap::NoInterventionalSupport));
        assert_eq!(profile.supporting_distinct_provenance_root_count, 1);
    }

    #[test]
    fn copied_supporting_reports_surface_provenance_concentration() {
        let mut ledger = EpistemicLedger::new();
        let original = source(&mut ledger, "original");
        let copy_a = ledger
            .add_provenance("copy-a", None, None, 2, vec![original])
            .unwrap();
        let copy_b = ledger
            .add_provenance("copy-b", None, None, 2, vec![original])
            .unwrap();
        let claim = ledger.add_claim("X exists", ClaimKind::Descriptive, None, None, 1);
        for provenance in [copy_a, copy_b] {
            ledger
                .add_evidence(
                    claim,
                    EvidenceKind::Report,
                    EvidencePolarity::Supports,
                    provenance,
                    3,
                    None,
                    None,
                )
                .unwrap();
        }

        let report = IgnoranceFrontier::inspect(&ledger, &[claim], &[]).unwrap();
        let profile = &report.profiles[0];
        assert_eq!(profile.supporting_evidence_count, 2);
        assert_eq!(profile.supporting_distinct_provenance_root_count, 1);
        assert!(profile.gaps.contains(
            &KnowledgeGap::SharedSupportingProvenanceAncestry {
                shared_pair_count: 1,
                supporting_evidence_count: 2,
                distinct_root_count: 1,
            }
        ));
    }

    #[test]
    fn separately_rooted_support_is_not_called_independent_or_a_gap() {
        let mut ledger = EpistemicLedger::new();
        let a = source(&mut ledger, "a");
        let b = source(&mut ledger, "b");
        let claim = ledger.add_claim("X exists", ClaimKind::Descriptive, None, None, 1);
        for provenance in [a, b] {
            ledger
                .add_evidence(
                    claim,
                    EvidenceKind::Report,
                    EvidencePolarity::Supports,
                    provenance,
                    2,
                    None,
                    None,
                )
                .unwrap();
        }

        let report = IgnoranceFrontier::inspect(&ledger, &[claim], &[]).unwrap();
        let profile = &report.profiles[0];
        assert_eq!(profile.supporting_distinct_provenance_root_count, 2);
        assert!(!profile.gaps.iter().any(|gap| matches!(
            gap,
            KnowledgeGap::SharedSupportingProvenanceAncestry { .. }
        )));
    }

    #[test]
    fn contradiction_is_preserved_as_a_gap_not_resolved_automatically() {
        let mut ledger = EpistemicLedger::new();
        let a = source(&mut ledger, "source-a");
        let b = source(&mut ledger, "source-b");
        let claim = ledger.add_claim("X predicts Y", ClaimKind::Predictive, None, None, 1);
        ledger
            .add_evidence(
                claim,
                EvidenceKind::Measurement,
                EvidencePolarity::Supports,
                a,
                2,
                None,
                None,
            )
            .unwrap();
        ledger
            .add_evidence(
                claim,
                EvidenceKind::Measurement,
                EvidencePolarity::Contradicts,
                b,
                3,
                None,
                None,
            )
            .unwrap();

        let report = IgnoranceFrontier::inspect(&ledger, &[claim], &[]).unwrap();
        assert!(report.profiles[0]
            .gaps
            .contains(&KnowledgeGap::ContradictoryEvidence { count: 1 }));
    }

    #[test]
    fn zero_assessed_uncertainty_is_not_unassessed() {
        let mut ledger = EpistemicLedger::new();
        let provenance = source(&mut ledger, "measurement");
        let claim = ledger.add_claim("X exists", ClaimKind::Descriptive, None, None, 1);
        let evidence = ledger
            .add_evidence(
                claim,
                EvidenceKind::Measurement,
                EvidencePolarity::Supports,
                provenance,
                2,
                None,
                None,
            )
            .unwrap();
        let vector = EpistemicVector::new()
            .with(UncertaintyDimension::Epistemic, 0.0)
            .unwrap();
        let assessment = ClaimUncertaintyAssessment::new(
            &ledger,
            claim,
            vector,
            vec![evidence],
            2,
        )
        .unwrap();

        let report = IgnoranceFrontier::inspect(&ledger, &[claim], &[assessment]).unwrap();
        let gaps = &report.profiles[0].gaps;
        assert!(!gaps.contains(&KnowledgeGap::UnassessedUncertaintyDimension(
            UncertaintyDimension::Epistemic
        )));
        assert!(gaps.contains(&KnowledgeGap::UnassessedUncertaintyDimension(
            UncertaintyDimension::Aleatoric
        )));
        assert_eq!(report.profiles[0].assessed_uncertainty_dimension_count, 1);
    }

    #[test]
    fn new_evidence_marks_older_uncertainty_assessment_stale() {
        let mut ledger = EpistemicLedger::new();
        let first_source = source(&mut ledger, "first");
        let second_source = source(&mut ledger, "second");
        let claim = ledger.add_claim("X predicts Y", ClaimKind::Predictive, None, None, 1);
        let first = ledger
            .add_evidence(
                claim,
                EvidenceKind::Observation,
                EvidencePolarity::Supports,
                first_source,
                2,
                None,
                None,
            )
            .unwrap();
        let assessment = ClaimUncertaintyAssessment::new(
            &ledger,
            claim,
            EpistemicVector::new(),
            vec![first],
            3,
        )
        .unwrap();
        ledger
            .add_evidence(
                claim,
                EvidenceKind::Observation,
                EvidencePolarity::Contradicts,
                second_source,
                7,
                None,
                None,
            )
            .unwrap();

        let report = IgnoranceFrontier::inspect(&ledger, &[claim], &[assessment]).unwrap();
        assert!(report.profiles[0]
            .gaps
            .contains(&KnowledgeGap::AssessmentPredatesEvidence {
                assessment_cycle: 3,
                latest_evidence_cycle: 7,
            }));
    }

    #[test]
    fn duplicate_claim_scope_fails_closed() {
        let mut ledger = EpistemicLedger::new();
        let claim = ledger.add_claim("test", ClaimKind::Descriptive, None, None, 1);
        assert_eq!(
            IgnoranceFrontier::inspect(&ledger, &[claim, claim], &[]).unwrap_err(),
            IgnoranceFrontierError::DuplicateClaim(claim)
        );
    }
}