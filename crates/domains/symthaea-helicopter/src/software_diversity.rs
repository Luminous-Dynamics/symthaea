// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Software-diversity assurance for redundant control lanes.
//!
//! Nominally separate lanes can share the same defect through source lineage,
//! algorithms, compilers, dependencies, teams, or training data. This module
//! quantifies declared diversity without claiming statistical independence.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DiversityDimension {
    SourceLineage,
    Algorithm,
    Language,
    Compiler,
    DependencyGraph,
    DevelopmentTeam,
    VerificationTeam,
    TrainingData,
    HardwareArchitecture,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SoftwareLaneIdentity {
    pub lane_id: String,
    pub source_lineage: String,
    pub algorithm_family: String,
    pub language: String,
    pub compiler_family: String,
    pub dependency_digest: String,
    pub development_team: String,
    pub verification_team: String,
    pub training_data_digest: Option<String>,
    pub hardware_architecture: String,
    pub evidence_ids: BTreeMap<DiversityDimension, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SoftwareDiversityPolicy {
    pub minimum_lanes: usize,
    pub required_distinct_dimensions: BTreeSet<DiversityDimension>,
    pub minimum_distinct_dimensions_per_pair: usize,
    pub prohibit_shared_development_and_verification_team: bool,
    pub require_evidence_for: BTreeSet<DiversityDimension>,
}

impl Default for SoftwareDiversityPolicy {
    fn default() -> Self {
        Self {
            minimum_lanes: 2,
            required_distinct_dimensions: BTreeSet::from([
                DiversityDimension::SourceLineage,
                DiversityDimension::Algorithm,
                DiversityDimension::DependencyGraph,
                DiversityDimension::DevelopmentTeam,
            ]),
            minimum_distinct_dimensions_per_pair: 5,
            prohibit_shared_development_and_verification_team: true,
            require_evidence_for: BTreeSet::from([
                DiversityDimension::SourceLineage,
                DiversityDimension::Algorithm,
                DiversityDimension::DependencyGraph,
                DiversityDimension::DevelopmentTeam,
                DiversityDimension::VerificationTeam,
            ]),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SoftwareDiversityStatus {
    Pass,
    Fail,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SoftwareDiversityIssue {
    EmptyIdentity,
    DuplicateLane(String),
    InsufficientLanes {
        required: usize,
        observed: usize,
    },
    MissingEvidence {
        lane_id: String,
        dimension: DiversityDimension,
    },
    SharedRequiredDimension {
        left: String,
        right: String,
        dimension: DiversityDimension,
    },
    InsufficientPairwiseDiversity {
        left: String,
        right: String,
        observed: usize,
        required: usize,
    },
    TeamIndependenceViolation {
        lane_id: String,
    },
    SharedVerificationTeam {
        left: String,
        right: String,
    },
    MissingTrainingDataIdentity(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PairwiseDiversityAssessment {
    pub left_lane_id: String,
    pub right_lane_id: String,
    pub distinct_dimensions: Vec<DiversityDimension>,
    pub shared_dimensions: Vec<DiversityDimension>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SoftwareDiversityReport {
    pub status: SoftwareDiversityStatus,
    pub lane_count: usize,
    pub pairwise: Vec<PairwiseDiversityAssessment>,
    pub issues: Vec<SoftwareDiversityIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SoftwareDiversityError {
    InvalidPolicy,
}

#[derive(Debug, Clone)]
pub struct SoftwareDiversityAssessor {
    policy: SoftwareDiversityPolicy,
}

impl SoftwareDiversityAssessor {
    pub fn new(policy: SoftwareDiversityPolicy) -> Result<Self, SoftwareDiversityError> {
        if policy.minimum_lanes < 2
            || policy.minimum_distinct_dimensions_per_pair == 0
            || policy.minimum_distinct_dimensions_per_pair > all_dimensions().len()
        {
            return Err(SoftwareDiversityError::InvalidPolicy);
        }
        Ok(Self { policy })
    }

    pub fn assess(&self, lanes: &[SoftwareLaneIdentity]) -> SoftwareDiversityReport {
        let mut issues = Vec::new();
        let mut ids = BTreeSet::new();
        for lane in lanes {
            if lane.lane_id.trim().is_empty()
                || lane.source_lineage.trim().is_empty()
                || lane.algorithm_family.trim().is_empty()
                || lane.language.trim().is_empty()
                || lane.compiler_family.trim().is_empty()
                || lane.dependency_digest.trim().is_empty()
                || lane.development_team.trim().is_empty()
                || lane.verification_team.trim().is_empty()
                || lane.hardware_architecture.trim().is_empty()
            {
                issues.push(SoftwareDiversityIssue::EmptyIdentity);
            }
            if !ids.insert(lane.lane_id.as_str()) {
                issues.push(SoftwareDiversityIssue::DuplicateLane(lane.lane_id.clone()));
            }
            if self
                .policy
                .prohibit_shared_development_and_verification_team
                && lane.development_team == lane.verification_team
            {
                issues.push(SoftwareDiversityIssue::TeamIndependenceViolation {
                    lane_id: lane.lane_id.clone(),
                });
            }
            for dimension in &self.policy.require_evidence_for {
                if lane
                    .evidence_ids
                    .get(dimension)
                    .is_none_or(|id| id.trim().is_empty())
                {
                    issues.push(SoftwareDiversityIssue::MissingEvidence {
                        lane_id: lane.lane_id.clone(),
                        dimension: *dimension,
                    });
                }
            }
            if self
                .policy
                .required_distinct_dimensions
                .contains(&DiversityDimension::TrainingData)
                && lane
                    .training_data_digest
                    .as_ref()
                    .is_none_or(|id| id.trim().is_empty())
            {
                issues.push(SoftwareDiversityIssue::MissingTrainingDataIdentity(
                    lane.lane_id.clone(),
                ));
            }
        }
        if lanes.len() < self.policy.minimum_lanes {
            issues.push(SoftwareDiversityIssue::InsufficientLanes {
                required: self.policy.minimum_lanes,
                observed: lanes.len(),
            });
        }

        let mut pairwise = Vec::new();
        for (index, left) in lanes.iter().enumerate() {
            for right in lanes.iter().skip(index + 1) {
                let mut distinct = Vec::new();
                let mut shared = Vec::new();
                for dimension in all_dimensions() {
                    if dimension_value(left, dimension) != dimension_value(right, dimension) {
                        distinct.push(dimension);
                    } else {
                        shared.push(dimension);
                        if self
                            .policy
                            .required_distinct_dimensions
                            .contains(&dimension)
                        {
                            issues.push(SoftwareDiversityIssue::SharedRequiredDimension {
                                left: left.lane_id.clone(),
                                right: right.lane_id.clone(),
                                dimension,
                            });
                        }
                    }
                }
                if distinct.len() < self.policy.minimum_distinct_dimensions_per_pair {
                    issues.push(SoftwareDiversityIssue::InsufficientPairwiseDiversity {
                        left: left.lane_id.clone(),
                        right: right.lane_id.clone(),
                        observed: distinct.len(),
                        required: self.policy.minimum_distinct_dimensions_per_pair,
                    });
                }
                if left.verification_team == right.verification_team {
                    issues.push(SoftwareDiversityIssue::SharedVerificationTeam {
                        left: left.lane_id.clone(),
                        right: right.lane_id.clone(),
                    });
                }
                pairwise.push(PairwiseDiversityAssessment {
                    left_lane_id: left.lane_id.clone(),
                    right_lane_id: right.lane_id.clone(),
                    distinct_dimensions: distinct,
                    shared_dimensions: shared,
                });
            }
        }

        let status = if issues.iter().any(is_failure) {
            SoftwareDiversityStatus::Fail
        } else if issues.is_empty() {
            SoftwareDiversityStatus::Pass
        } else {
            SoftwareDiversityStatus::Incomplete
        };
        SoftwareDiversityReport {
            status,
            lane_count: lanes.len(),
            pairwise,
            issues,
        }
    }
}

fn all_dimensions() -> [DiversityDimension; 9] {
    [
        DiversityDimension::SourceLineage,
        DiversityDimension::Algorithm,
        DiversityDimension::Language,
        DiversityDimension::Compiler,
        DiversityDimension::DependencyGraph,
        DiversityDimension::DevelopmentTeam,
        DiversityDimension::VerificationTeam,
        DiversityDimension::TrainingData,
        DiversityDimension::HardwareArchitecture,
    ]
}

fn dimension_value(lane: &SoftwareLaneIdentity, dimension: DiversityDimension) -> &str {
    match dimension {
        DiversityDimension::SourceLineage => &lane.source_lineage,
        DiversityDimension::Algorithm => &lane.algorithm_family,
        DiversityDimension::Language => &lane.language,
        DiversityDimension::Compiler => &lane.compiler_family,
        DiversityDimension::DependencyGraph => &lane.dependency_digest,
        DiversityDimension::DevelopmentTeam => &lane.development_team,
        DiversityDimension::VerificationTeam => &lane.verification_team,
        DiversityDimension::TrainingData => lane.training_data_digest.as_deref().unwrap_or("none"),
        DiversityDimension::HardwareArchitecture => &lane.hardware_architecture,
    }
}

fn is_failure(issue: &SoftwareDiversityIssue) -> bool {
    matches!(
        issue,
        SoftwareDiversityIssue::SharedRequiredDimension { .. }
            | SoftwareDiversityIssue::InsufficientPairwiseDiversity { .. }
            | SoftwareDiversityIssue::TeamIndependenceViolation { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lane(id: &str, primary: bool) -> SoftwareLaneIdentity {
        let mut evidence_ids = BTreeMap::new();
        for dimension in SoftwareDiversityPolicy::default().require_evidence_for {
            evidence_ids.insert(dimension, format!("evidence-{id}-{dimension:?}"));
        }
        SoftwareLaneIdentity {
            lane_id: id.into(),
            source_lineage: if primary {
                "rust-primary"
            } else {
                "cpp-oracle"
            }
            .into(),
            algorithm_family: if primary {
                "hdc-ltc"
            } else {
                "classical-state-space"
            }
            .into(),
            language: if primary { "Rust" } else { "C++" }.into(),
            compiler_family: if primary { "LLVM" } else { "GCC" }.into(),
            dependency_digest: format!("sha256:deps-{id}"),
            development_team: format!("dev-{id}"),
            verification_team: format!("verify-{id}"),
            training_data_digest: Some(format!("sha256:data-{id}")),
            hardware_architecture: if primary { "x86_64" } else { "aarch64" }.into(),
            evidence_ids,
        }
    }

    #[test]
    fn genuinely_diverse_pair_passes() {
        let report = SoftwareDiversityAssessor::new(SoftwareDiversityPolicy::default())
            .unwrap()
            .assess(&[lane("a", true), lane("b", false)]);
        assert_eq!(report.status, SoftwareDiversityStatus::Pass);
    }

    #[test]
    fn cloned_lane_fails() {
        let left = lane("a", true);
        let mut right = left.clone();
        right.lane_id = "b".into();
        let report = SoftwareDiversityAssessor::new(SoftwareDiversityPolicy::default())
            .unwrap()
            .assess(&[left, right]);
        assert_eq!(report.status, SoftwareDiversityStatus::Fail);
    }
}
