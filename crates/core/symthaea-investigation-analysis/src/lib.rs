// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic dependency and negative-search analysis for bounded investigations.
//!
//! This crate performs no source discovery, network access, persistence, model inference,
//! source-reliability scoring, or truth adjudication. It consumes explicit dependency and
//! search-coverage observations and returns candidate-only descriptive summaries.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use symthaea_investigation::{
    EvidenceRef, FrontierRef, InvestigationAuthorityScopeV1, InvestigationError, ProfileRef,
    SearchEvidenceRef,
};

pub const ANALYSIS_PROFILE_V1: &str = "symthaea:deterministic-osint-analysis:v1";

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArtifactProjectionRef(EvidenceRef);

impl ArtifactProjectionRef {
    pub fn new(value: impl Into<String>) -> Result<Self, InvestigationError> {
        EvidenceRef::new(value).map(Self)
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DependencyGroupRef(EvidenceRef);

impl DependencyGroupRef {
    pub fn new(value: impl Into<String>) -> Result<Self, InvestigationError> {
        EvidenceRef::new(value).map(Self)
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnalyzerError {
    Investigation(InvestigationError),
    DuplicateArtifact(String),
    NegativeSearchHasResults(usize),
    IncoherentNegativeSearch {
        finding: NegativeFindingProjectionV1,
        coverage: SearchCoverageProjectionV1,
    },
}

impl fmt::Display for AnalyzerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Investigation(error) => error.fmt(f),
            Self::DuplicateArtifact(id) => write!(f, "artifact appears more than once: {id}"),
            Self::NegativeSearchHasResults(count) => {
                write!(f, "negative-search observation has {count} result(s)")
            }
            Self::IncoherentNegativeSearch { finding, coverage } => write!(
                f,
                "negative finding {finding:?} is incoherent with coverage {coverage:?}"
            ),
        }
    }
}

impl Error for AnalyzerError {}

impl From<InvestigationError> for AnalyzerError {
    fn from(value: InvestigationError) -> Self {
        Self::Investigation(value)
    }
}

/// Read-only dependency state projected into Symthaea analysis.
///
/// These variants are descriptive inputs only. They are not canonical Mycelix EPI objects and
/// do not establish source independence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DependencyProjectionStateV1 {
    /// The artifact belongs to an explicitly observed lineage/dependency group.
    ObservedLineageGroup(DependencyGroupRef),
    /// A declared analysis profile assessed this artifact disjoint within its exact scope.
    DeclaredDisjointWithinScope(ProfileRef),
    /// No dependency was detected under the declared scope/profile.
    NoDependencyDetectedWithinScope(ProfileRef),
    /// Dependency lineage remains unresolved.
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DependencyObservationV1 {
    pub artifact_ref: ArtifactProjectionRef,
    pub state: DependencyProjectionStateV1,
}

/// Exact membership of one observed dependency group.
///
/// Group membership is descriptive lineage only; the group is not a proven independent source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservedDependencyGroupV1 {
    pub group_ref: DependencyGroupRef,
    pub artifact_refs: Vec<ArtifactProjectionRef>,
}

/// Scoped dependency assessment retained without strengthening it to global independence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScopedArtifactDependencyAssessmentV1 {
    pub artifact_ref: ArtifactProjectionRef,
    pub scope_profile_ref: ProfileRef,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CorroborationSummaryV1 {
    pub frontier_ref: FrontierRef,
    pub analysis_profile_ref: ProfileRef,
    pub inspected_artifacts: Vec<ArtifactProjectionRef>,
    pub observed_dependency_groups: Vec<ObservedDependencyGroupV1>,
    pub declared_disjoint_within_scope: Vec<ScopedArtifactDependencyAssessmentV1>,
    pub no_dependency_detected_within_scope: Vec<ScopedArtifactDependencyAssessmentV1>,
    pub unresolved_lineage: Vec<ArtifactProjectionRef>,
}

impl CorroborationSummaryV1 {
    pub fn asserting_artifact_count(&self) -> usize {
        self.inspected_artifacts.len()
    }

    pub fn observed_dependency_group_count(&self) -> usize {
        self.observed_dependency_groups.len()
    }

    pub fn unresolved_lineage_count(&self) -> usize {
        self.unresolved_lineage.len()
    }

    pub fn authority_scope(&self) -> InvestigationAuthorityScopeV1 {
        InvestigationAuthorityScopeV1::CandidateAnalysisOnly
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CorroborationAnalyzerV1;

impl CorroborationAnalyzerV1 {
    pub fn summarize(
        frontier_ref: FrontierRef,
        analysis_profile_ref: ProfileRef,
        observations: Vec<DependencyObservationV1>,
    ) -> Result<CorroborationSummaryV1, AnalyzerError> {
        let mut seen_artifacts = BTreeSet::new();
        let mut inspected_artifacts = Vec::with_capacity(observations.len());
        let mut group_members: BTreeMap<DependencyGroupRef, Vec<ArtifactProjectionRef>> =
            BTreeMap::new();
        let mut declared_disjoint = Vec::new();
        let mut no_dependency_detected = Vec::new();
        let mut unresolved = Vec::new();

        for observation in observations {
            if !seen_artifacts.insert(observation.artifact_ref.clone()) {
                return Err(AnalyzerError::DuplicateArtifact(
                    observation.artifact_ref.as_str().to_string(),
                ));
            }

            inspected_artifacts.push(observation.artifact_ref.clone());
            match observation.state {
                DependencyProjectionStateV1::ObservedLineageGroup(group_ref) => {
                    group_members
                        .entry(group_ref)
                        .or_default()
                        .push(observation.artifact_ref);
                }
                DependencyProjectionStateV1::DeclaredDisjointWithinScope(scope_profile_ref) => {
                    declared_disjoint.push(ScopedArtifactDependencyAssessmentV1 {
                        artifact_ref: observation.artifact_ref,
                        scope_profile_ref,
                    });
                }
                DependencyProjectionStateV1::NoDependencyDetectedWithinScope(scope_profile_ref) => {
                    no_dependency_detected.push(ScopedArtifactDependencyAssessmentV1 {
                        artifact_ref: observation.artifact_ref,
                        scope_profile_ref,
                    });
                }
                DependencyProjectionStateV1::Unknown => {
                    unresolved.push(observation.artifact_ref);
                }
            }
        }

        let observed_dependency_groups = group_members
            .into_iter()
            .map(|(group_ref, artifact_refs)| ObservedDependencyGroupV1 {
                group_ref,
                artifact_refs,
            })
            .collect();

        Ok(CorroborationSummaryV1 {
            frontier_ref,
            analysis_profile_ref,
            inspected_artifacts,
            observed_dependency_groups,
            declared_disjoint_within_scope: declared_disjoint,
            no_dependency_detected_within_scope: no_dependency_detected,
            unresolved_lineage: unresolved,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchCoverageProjectionV1 {
    UnknownCoverage,
    KnownPartial,
    PaginationExhaustedUnderProfile,
    ExhaustiveWithinDeclaredFiniteCorpus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NegativeFindingProjectionV1 {
    NoMatchObservedUnderSearchProfile,
    AbsentFromExactFiniteCorpusCommitment,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NegativeSearchObservationV1 {
    pub search_evidence_ref: SearchEvidenceRef,
    pub frontier_ref: FrontierRef,
    pub result_count: usize,
    pub coverage: SearchCoverageProjectionV1,
    pub finding: NegativeFindingProjectionV1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NegativeSearchInterpretationV1 {
    UnresolvedDueToUnknownCoverage,
    UnresolvedDueToPartialCoverage,
    NoMatchWithinPaginationProfile,
    AbsentWithinExactFiniteCorpus,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NegativeSearchInterpretationRecordV1 {
    pub search_evidence_ref: SearchEvidenceRef,
    pub frontier_ref: FrontierRef,
    pub interpretation: NegativeSearchInterpretationV1,
}

impl NegativeSearchInterpretationRecordV1 {
    pub fn authority_scope(&self) -> InvestigationAuthorityScopeV1 {
        InvestigationAuthorityScopeV1::CandidateAnalysisOnly
    }
}

pub fn interpret_negative_search(
    observation: NegativeSearchObservationV1,
) -> Result<NegativeSearchInterpretationRecordV1, AnalyzerError> {
    if observation.result_count != 0 {
        return Err(AnalyzerError::NegativeSearchHasResults(
            observation.result_count,
        ));
    }

    let interpretation = match (observation.finding, observation.coverage) {
        (
            NegativeFindingProjectionV1::NoMatchObservedUnderSearchProfile,
            SearchCoverageProjectionV1::UnknownCoverage,
        ) => NegativeSearchInterpretationV1::UnresolvedDueToUnknownCoverage,
        (
            NegativeFindingProjectionV1::NoMatchObservedUnderSearchProfile,
            SearchCoverageProjectionV1::KnownPartial,
        ) => NegativeSearchInterpretationV1::UnresolvedDueToPartialCoverage,
        (
            NegativeFindingProjectionV1::NoMatchObservedUnderSearchProfile,
            SearchCoverageProjectionV1::PaginationExhaustedUnderProfile,
        ) => NegativeSearchInterpretationV1::NoMatchWithinPaginationProfile,
        (
            NegativeFindingProjectionV1::NoMatchObservedUnderSearchProfile,
            SearchCoverageProjectionV1::ExhaustiveWithinDeclaredFiniteCorpus,
        )
        | (
            NegativeFindingProjectionV1::AbsentFromExactFiniteCorpusCommitment,
            SearchCoverageProjectionV1::ExhaustiveWithinDeclaredFiniteCorpus,
        ) => NegativeSearchInterpretationV1::AbsentWithinExactFiniteCorpus,
        (finding, coverage) => {
            return Err(AnalyzerError::IncoherentNegativeSearch { finding, coverage })
        }
    };

    Ok(NegativeSearchInterpretationRecordV1 {
        search_evidence_ref: observation.search_evidence_ref,
        frontier_ref: observation.frontier_ref,
        interpretation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(id: &str) -> ArtifactProjectionRef {
        ArtifactProjectionRef::new(format!("artifact:{id}")).unwrap()
    }

    fn group(id: &str) -> DependencyGroupRef {
        DependencyGroupRef::new(format!("dependency-group:{id}")).unwrap()
    }

    fn frontier() -> FrontierRef {
        FrontierRef::new("frontier:F2").unwrap()
    }

    fn profile() -> ProfileRef {
        ProfileRef::new("analysis:dependency:v1").unwrap()
    }

    #[test]
    fn three_reports_in_one_lineage_preserve_exact_group_membership() {
        let summary = CorroborationAnalyzerV1::summarize(
            frontier(),
            profile(),
            vec![
                DependencyObservationV1 {
                    artifact_ref: artifact("A1"),
                    state: DependencyProjectionStateV1::ObservedLineageGroup(group("G1")),
                },
                DependencyObservationV1 {
                    artifact_ref: artifact("A2"),
                    state: DependencyProjectionStateV1::ObservedLineageGroup(group("G1")),
                },
                DependencyObservationV1 {
                    artifact_ref: artifact("A3"),
                    state: DependencyProjectionStateV1::ObservedLineageGroup(group("G1")),
                },
            ],
        )
        .unwrap();

        assert_eq!(summary.asserting_artifact_count(), 3);
        assert_eq!(summary.observed_dependency_group_count(), 1);
        assert_eq!(summary.observed_dependency_groups[0].group_ref.as_str(), "dependency-group:G1");
        assert_eq!(summary.observed_dependency_groups[0].artifact_refs.len(), 3);
        assert_eq!(summary.unresolved_lineage_count(), 0);
        assert_eq!(
            summary.authority_scope(),
            InvestigationAuthorityScopeV1::CandidateAnalysisOnly
        );
    }

    #[test]
    fn two_groups_preserve_membership_without_implying_independence() {
        let summary = CorroborationAnalyzerV1::summarize(
            frontier(),
            profile(),
            vec![
                DependencyObservationV1 {
                    artifact_ref: artifact("A1"),
                    state: DependencyProjectionStateV1::ObservedLineageGroup(group("G1")),
                },
                DependencyObservationV1 {
                    artifact_ref: artifact("A2"),
                    state: DependencyProjectionStateV1::ObservedLineageGroup(group("G1")),
                },
                DependencyObservationV1 {
                    artifact_ref: artifact("A3"),
                    state: DependencyProjectionStateV1::ObservedLineageGroup(group("G2")),
                },
            ],
        )
        .unwrap();

        assert_eq!(summary.observed_dependency_group_count(), 2);
        assert_eq!(summary.observed_dependency_groups[0].artifact_refs.len(), 2);
        assert_eq!(summary.observed_dependency_groups[1].artifact_refs.len(), 1);
    }

    #[test]
    fn duplicate_artifact_assignment_rejects_even_if_group_matches() {
        let result = CorroborationAnalyzerV1::summarize(
            frontier(),
            profile(),
            vec![
                DependencyObservationV1 {
                    artifact_ref: artifact("A1"),
                    state: DependencyProjectionStateV1::ObservedLineageGroup(group("G1")),
                },
                DependencyObservationV1 {
                    artifact_ref: artifact("A1"),
                    state: DependencyProjectionStateV1::ObservedLineageGroup(group("G1")),
                },
            ],
        );

        assert_eq!(
            result,
            Err(AnalyzerError::DuplicateArtifact("artifact:A1".into()))
        );
    }

    #[test]
    fn unresolved_lineage_does_not_become_an_independent_group() {
        let summary = CorroborationAnalyzerV1::summarize(
            frontier(),
            profile(),
            vec![
                DependencyObservationV1 {
                    artifact_ref: artifact("A1"),
                    state: DependencyProjectionStateV1::ObservedLineageGroup(group("G1")),
                },
                DependencyObservationV1 {
                    artifact_ref: artifact("A2"),
                    state: DependencyProjectionStateV1::Unknown,
                },
            ],
        )
        .unwrap();

        assert_eq!(summary.observed_dependency_group_count(), 1);
        assert_eq!(summary.unresolved_lineage_count(), 1);
    }

    #[test]
    fn no_dependency_detected_preserves_exact_scope_and_is_not_a_group() {
        let summary = CorroborationAnalyzerV1::summarize(
            frontier(),
            profile(),
            vec![DependencyObservationV1 {
                artifact_ref: artifact("A1"),
                state: DependencyProjectionStateV1::NoDependencyDetectedWithinScope(
                    ProfileRef::new("dependency-scope:limited").unwrap(),
                ),
            }],
        )
        .unwrap();

        assert_eq!(summary.asserting_artifact_count(), 1);
        assert_eq!(summary.observed_dependency_group_count(), 0);
        assert_eq!(summary.no_dependency_detected_within_scope.len(), 1);
        assert_eq!(
            summary.no_dependency_detected_within_scope[0]
                .scope_profile_ref
                .as_str(),
            "dependency-scope:limited"
        );
    }

    #[test]
    fn declared_disjoint_preserves_exact_scope() {
        let summary = CorroborationAnalyzerV1::summarize(
            frontier(),
            profile(),
            vec![DependencyObservationV1 {
                artifact_ref: artifact("A1"),
                state: DependencyProjectionStateV1::DeclaredDisjointWithinScope(
                    ProfileRef::new("dependency-scope:fixture").unwrap(),
                ),
            }],
        )
        .unwrap();

        assert_eq!(summary.observed_dependency_group_count(), 0);
        assert_eq!(summary.declared_disjoint_within_scope.len(), 1);
        assert_eq!(
            summary.declared_disjoint_within_scope[0]
                .scope_profile_ref
                .as_str(),
            "dependency-scope:fixture"
        );
    }

    #[test]
    fn empty_artifact_set_does_not_invent_corroboration() {
        let summary = CorroborationAnalyzerV1::summarize(frontier(), profile(), vec![]).unwrap();
        assert_eq!(summary.asserting_artifact_count(), 0);
        assert_eq!(summary.observed_dependency_group_count(), 0);
        assert_eq!(summary.unresolved_lineage_count(), 0);
    }

    fn negative_search(
        coverage: SearchCoverageProjectionV1,
        finding: NegativeFindingProjectionV1,
    ) -> NegativeSearchObservationV1 {
        NegativeSearchObservationV1 {
            search_evidence_ref: SearchEvidenceRef::new("search:S1").unwrap(),
            frontier_ref: FrontierRef::new("frontier:F2").unwrap(),
            result_count: 0,
            coverage,
            finding,
        }
    }

    #[test]
    fn zero_results_with_unknown_coverage_remain_unresolved() {
        let interpreted = interpret_negative_search(negative_search(
            SearchCoverageProjectionV1::UnknownCoverage,
            NegativeFindingProjectionV1::NoMatchObservedUnderSearchProfile,
        ))
        .unwrap();

        assert_eq!(
            interpreted.interpretation,
            NegativeSearchInterpretationV1::UnresolvedDueToUnknownCoverage
        );
    }

    #[test]
    fn zero_results_with_partial_coverage_remain_unresolved() {
        let interpreted = interpret_negative_search(negative_search(
            SearchCoverageProjectionV1::KnownPartial,
            NegativeFindingProjectionV1::NoMatchObservedUnderSearchProfile,
        ))
        .unwrap();

        assert_eq!(
            interpreted.interpretation,
            NegativeSearchInterpretationV1::UnresolvedDueToPartialCoverage
        );
    }

    #[test]
    fn pagination_exhaustion_stays_scoped_to_the_profile() {
        let interpreted = interpret_negative_search(negative_search(
            SearchCoverageProjectionV1::PaginationExhaustedUnderProfile,
            NegativeFindingProjectionV1::NoMatchObservedUnderSearchProfile,
        ))
        .unwrap();

        assert_eq!(
            interpreted.interpretation,
            NegativeSearchInterpretationV1::NoMatchWithinPaginationProfile
        );
    }

    #[test]
    fn exact_finite_exhaustive_search_can_establish_only_corpus_absence() {
        for finding in [
            NegativeFindingProjectionV1::NoMatchObservedUnderSearchProfile,
            NegativeFindingProjectionV1::AbsentFromExactFiniteCorpusCommitment,
        ] {
            let interpreted = interpret_negative_search(negative_search(
                SearchCoverageProjectionV1::ExhaustiveWithinDeclaredFiniteCorpus,
                finding,
            ))
            .unwrap();

            assert_eq!(
                interpreted.interpretation,
                NegativeSearchInterpretationV1::AbsentWithinExactFiniteCorpus
            );
            assert_eq!(
                interpreted.authority_scope(),
                InvestigationAuthorityScopeV1::CandidateAnalysisOnly
            );
        }
    }

    #[test]
    fn finite_corpus_absence_claim_with_unknown_coverage_rejects() {
        let result = interpret_negative_search(negative_search(
            SearchCoverageProjectionV1::UnknownCoverage,
            NegativeFindingProjectionV1::AbsentFromExactFiniteCorpusCommitment,
        ));

        assert_eq!(
            result,
            Err(AnalyzerError::IncoherentNegativeSearch {
                finding: NegativeFindingProjectionV1::AbsentFromExactFiniteCorpusCommitment,
                coverage: SearchCoverageProjectionV1::UnknownCoverage,
            })
        );
    }

    #[test]
    fn negative_search_with_positive_result_count_rejects() {
        let mut observation = negative_search(
            SearchCoverageProjectionV1::UnknownCoverage,
            NegativeFindingProjectionV1::NoMatchObservedUnderSearchProfile,
        );
        observation.result_count = 1;

        assert_eq!(
            interpret_negative_search(observation),
            Err(AnalyzerError::NegativeSearchHasResults(1))
        );
    }

    #[test]
    fn interpretation_vocabulary_contains_no_world_absence_or_truth_state() {
        let variants = [
            NegativeSearchInterpretationV1::UnresolvedDueToUnknownCoverage,
            NegativeSearchInterpretationV1::UnresolvedDueToPartialCoverage,
            NegativeSearchInterpretationV1::NoMatchWithinPaginationProfile,
            NegativeSearchInterpretationV1::AbsentWithinExactFiniteCorpus,
        ];

        for variant in variants {
            let rendered = format!("{variant:?}");
            assert!(!rendered.contains("World"));
            assert!(!rendered.contains("True"));
            assert!(!rendered.contains("False"));
        }
    }
}
