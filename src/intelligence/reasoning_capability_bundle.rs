// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Portable input bundle for one reasoning capability lane.
//!
//! Benchmark runners can emit this common bundle rather than inventing domain-specific summary
//! formats. Qualification is still performed from the full episodes + receipts; a precomputed
//! score is never accepted as authority.

use super::reasoning_capability_matrix::{
    build_capability_lane, BaselineMetric, CapabilityLaneDescriptor, CapabilityLaneReport,
    CapabilityMatrixError,
};
use super::reasoning_qualification::{ReasoningEpisode, ReasoningQualificationReceipt};
use serde::{Deserialize, Serialize};
use std::fmt;

pub const REASONING_CAPABILITY_LANE_BUNDLE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapabilityLaneBundle {
    pub schema_version: u32,
    pub descriptor: CapabilityLaneDescriptor,
    pub episodes: Vec<ReasoningEpisode>,
    pub receipts: Vec<ReasoningQualificationReceipt>,
    pub baselines: Vec<BaselineMetric>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CapabilityLaneBundleError {
    UnsupportedSchema(u32),
    Matrix(CapabilityMatrixError),
}

impl fmt::Display for CapabilityLaneBundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchema(version) => {
                write!(f, "unsupported capability lane bundle schema version {version}")
            }
            Self::Matrix(err) => write!(f, "capability lane qualification failed: {err}"),
        }
    }
}

impl std::error::Error for CapabilityLaneBundleError {}

impl From<CapabilityMatrixError> for CapabilityLaneBundleError {
    fn from(value: CapabilityMatrixError) -> Self {
        Self::Matrix(value)
    }
}

impl CapabilityLaneBundle {
    pub fn new(
        descriptor: CapabilityLaneDescriptor,
        episodes: Vec<ReasoningEpisode>,
        receipts: Vec<ReasoningQualificationReceipt>,
        baselines: Vec<BaselineMetric>,
    ) -> Self {
        Self {
            schema_version: REASONING_CAPABILITY_LANE_BUNDLE_SCHEMA_VERSION,
            descriptor,
            episodes,
            receipts,
            baselines,
        }
    }

    /// Rebuild the lane report from primary episode + receipt evidence every time.
    pub fn qualify(&self) -> Result<CapabilityLaneReport, CapabilityLaneBundleError> {
        if self.schema_version != REASONING_CAPABILITY_LANE_BUNDLE_SCHEMA_VERSION {
            return Err(CapabilityLaneBundleError::UnsupportedSchema(
                self.schema_version,
            ));
        }
        Ok(build_capability_lane(
            self.descriptor.clone(),
            &self.episodes,
            &self.receipts,
            self.baselines.clone(),
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intelligence::{
        evaluate_episode, CapabilityLaneDescriptor, ContaminationStatus, EpisodeJudgment,
        HoldoutPolicy, ReasoningDomain, ReasoningEpisode, ReasoningOutcome, ReasoningProblemRef,
        ResourceBudget, ResourceUsage,
    };

    fn bundle() -> CapabilityLaneBundle {
        let episode = match ReasoningEpisode::new(
            "subject-a",
            "config-a",
            ReasoningDomain::Mathematics,
            ReasoningProblemRef {
                benchmark: "math-unit".into(),
                benchmark_version: "v1".into(),
                split: "holdout".into(),
                problem_id: "m1".into(),
                problem_hash: "blake3:m1".into(),
            },
            vec![],
            vec![],
            vec![],
            ReasoningOutcome::Asserted {
                value: "42".into(),
                confidence: 0.9,
            },
            ResourceUsage::default(),
        ) {
            Ok(value) => value,
            Err(err) => panic!("episode must validate: {err}"),
        };
        let receipt = match evaluate_episode(
            &episode,
            "eval:m1",
            &EpisodeJudgment {
                exact_correct: Some(true),
                task_score: None,
            },
        ) {
            Ok(value) => value,
            Err(err) => panic!("receipt must validate: {err}"),
        };
        let descriptor = CapabilityLaneDescriptor {
            lane_id: "math-unit".into(),
            domain: ReasoningDomain::Mathematics,
            benchmark: "math-unit".into(),
            benchmark_version: "v1".into(),
            split: "holdout".into(),
            holdout_policy: HoldoutPolicy::FrozenUnseen,
            contamination_policy_id: "ledger:v1".into(),
            contamination_status: ContaminationStatus::Controlled,
            resource_budget: ResourceBudget::default(),
        };
        CapabilityLaneBundle::new(descriptor, vec![episode], vec![receipt], vec![])
    }

    #[test]
    fn bundle_requalifies_from_primary_evidence() {
        let report = bundle().qualify();
        let report = match report {
            Ok(value) => value,
            Err(err) => panic!("bundle should qualify: {err}"),
        };
        assert_eq!(report.capability.exact_accuracy, Some(1.0));
    }

    #[test]
    fn unsupported_bundle_schema_fails_closed() {
        let mut value = bundle();
        value.schema_version += 1;
        assert!(matches!(
            value.qualify(),
            Err(CapabilityLaneBundleError::UnsupportedSchema(_))
        ));
    }

    #[test]
    fn tampered_receipt_is_rejected_during_requalification() {
        let mut value = bundle();
        value.receipts[0].exact_correct = Some(false);
        assert!(value.qualify().is_err());
    }
}
