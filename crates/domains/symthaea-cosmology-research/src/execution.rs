// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fail-closed executable contract for DE-001A0/A1.
//!
//! This binary performs no cosmology. It encodes the protocol that a later
//! scientific runner must satisfy before DE-001A1 can be considered valid.

use serde::{Deserialize, Serialize};
use symthaea_cosmology_research::identity::{GitObjectId, Sha256Digest};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionOutcome {
    Pass,
    Negative,
    Indeterminate,
    Invalid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InvocationPolicy {
    Forbidden,
    Required,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum A1Blocker {
    EnvironmentNotQualified,
    ArtifactIntegrityNotPassed,
    ParameterPointNotFrozen,
    InvalidSpecification,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactHashCheck {
    pub label: String,
    pub expected_sha256: Sha256Digest,
    pub observed_sha256: Sha256Digest,
}

impl ArtifactHashCheck {
    pub fn matches(&self) -> bool {
        !self.label.trim().is_empty() && self.expected_sha256 == self.observed_sha256
    }
}

/// Receipt for DE-001A0. A hash mismatch is an integrity failure, not a
/// scientific NEGATIVE result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct A0IntegrityReceipt {
    pub subject_commit: GitObjectId,
    pub reference_manifest_blob: GitObjectId,
    pub checks: Vec<ArtifactHashCheck>,
    pub unregistered_artifact_consumed: bool,
    pub postflight_immutable: bool,
}

impl A0IntegrityReceipt {
    pub fn effective_outcome(&self) -> ExecutionOutcome {
        if self.checks.is_empty()
            || self.unregistered_artifact_consumed
            || !self.postflight_immutable
            || self.checks.iter().any(|check| !check.matches())
        {
            ExecutionOutcome::Invalid
        } else {
            ExecutionOutcome::Pass
        }
    }

    pub fn is_pass(&self) -> bool {
        self.effective_outcome() == ExecutionOutcome::Pass
    }
}

/// Frozen A1 protocol. `parameter_point_sha256=None` is an explicit blocker;
/// implementations must not reconstruct the point from derived outputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct A1FixedPointSpec {
    pub subject_commit: GitObjectId,
    pub reference_manifest_blob: GitObjectId,
    pub closure_manifest_blob: GitObjectId,
    pub parameter_point_sha256: Option<Sha256Digest>,
    pub expected_chi2_bao: f64,
    pub max_absolute_delta: f64,
    pub required_evaluation_count: u32,
    pub minimizer_policy: InvocationPolicy,
    pub sampler_policy: InvocationPolicy,
}

impl A1FixedPointSpec {
    pub fn validate(&self) -> bool {
        self.parameter_point_sha256.is_some()
            && self.expected_chi2_bao.is_finite()
            && self.max_absolute_delta.is_finite()
            && self.max_absolute_delta >= 0.0
            && self.required_evaluation_count == 1
            && self.minimizer_policy == InvocationPolicy::Forbidden
            && self.sampler_policy == InvocationPolicy::Forbidden
    }

    pub fn authorize(
        &self,
        environment_qualified: bool,
        a0_passed: bool,
    ) -> Result<(), A1Blocker> {
        if self.parameter_point_sha256.is_none() {
            return Err(A1Blocker::ParameterPointNotFrozen);
        }
        if !self.validate() {
            return Err(A1Blocker::InvalidSpecification);
        }
        if !environment_qualified {
            return Err(A1Blocker::EnvironmentNotQualified);
        }
        if !a0_passed {
            return Err(A1Blocker::ArtifactIntegrityNotPassed);
        }
        Ok(())
    }
}

/// Receipt for one fixed-point A1 likelihood evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct A1FixedPointReceipt {
    pub subject_commit: GitObjectId,
    pub environment_receipt_sha256: Sha256Digest,
    pub a0_receipt_sha256: Sha256Digest,
    pub parameter_point_sha256: Sha256Digest,
    pub result_bundle_sha256: Sha256Digest,
    pub observed_chi2_bao: f64,
    pub evaluation_count: u32,
    pub minimizer_invoked: bool,
    pub sampler_invoked: bool,
    pub parameter_mutation_detected: bool,
    pub environment_qualification_passed: bool,
    pub a0_integrity_passed: bool,
    pub postflight_immutable: bool,
}

impl A1FixedPointReceipt {
    pub fn effective_outcome(&self, spec: &A1FixedPointSpec) -> ExecutionOutcome {
        if !spec.validate()
            || !self.environment_qualification_passed
            || !self.a0_integrity_passed
            || !self.postflight_immutable
            || self.minimizer_invoked
            || self.sampler_invoked
            || self.parameter_mutation_detected
            || self.evaluation_count != spec.required_evaluation_count
            || !self.observed_chi2_bao.is_finite()
            || spec.parameter_point_sha256.as_ref() != Some(&self.parameter_point_sha256)
            || self.subject_commit != spec.subject_commit
        {
            return ExecutionOutcome::Invalid;
        }

        let delta = (self.observed_chi2_bao - spec.expected_chi2_bao).abs();
        if delta <= spec.max_absolute_delta {
            ExecutionOutcome::Pass
        } else {
            ExecutionOutcome::Negative
        }
    }

    pub const fn licenses_scientific_claim(&self) -> bool {
        false
    }
}

fn main() {
    println!("DE-001A0/A1 execution contract only; scientific execution disabled");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> Sha256Digest {
        Sha256Digest::parse(&byte.to_string().repeat(64)).unwrap()
    }

    fn git(byte: char) -> GitObjectId {
        GitObjectId::parse(&byte.to_string().repeat(40)).unwrap()
    }

    fn spec() -> A1FixedPointSpec {
        A1FixedPointSpec {
            subject_commit: git('a'),
            reference_manifest_blob: git('b'),
            closure_manifest_blob: git('c'),
            parameter_point_sha256: Some(digest('d')),
            expected_chi2_bao: 10.282_299,
            max_absolute_delta: 0.01,
            required_evaluation_count: 1,
            minimizer_policy: InvocationPolicy::Forbidden,
            sampler_policy: InvocationPolicy::Forbidden,
        }
    }

    fn receipt() -> A1FixedPointReceipt {
        A1FixedPointReceipt {
            subject_commit: git('a'),
            environment_receipt_sha256: digest('e'),
            a0_receipt_sha256: digest('f'),
            parameter_point_sha256: digest('d'),
            result_bundle_sha256: digest('1'),
            observed_chi2_bao: 10.282_299,
            evaluation_count: 1,
            minimizer_invoked: false,
            sampler_invoked: false,
            parameter_mutation_detected: false,
            environment_qualification_passed: true,
            a0_integrity_passed: true,
            postflight_immutable: true,
        }
    }

    #[test]
    fn a0_hash_mismatch_is_invalid_not_negative() {
        let receipt = A0IntegrityReceipt {
            subject_commit: git('a'),
            reference_manifest_blob: git('b'),
            checks: vec![ArtifactHashCheck {
                label: "dataset-mean".into(),
                expected_sha256: digest('c'),
                observed_sha256: digest('d'),
            }],
            unregistered_artifact_consumed: false,
            postflight_immutable: true,
        };
        assert_eq!(receipt.effective_outcome(), ExecutionOutcome::Invalid);
    }

    #[test]
    fn clean_a0_receipt_passes() {
        let receipt = A0IntegrityReceipt {
            subject_commit: git('a'),
            reference_manifest_blob: git('b'),
            checks: vec![ArtifactHashCheck {
                label: "dataset-mean".into(),
                expected_sha256: digest('c'),
                observed_sha256: digest('c'),
            }],
            unregistered_artifact_consumed: false,
            postflight_immutable: true,
        };
        assert!(receipt.is_pass());
    }

    #[test]
    fn a1_is_blocked_until_parameter_point_is_frozen() {
        let mut blocked = spec();
        blocked.parameter_point_sha256 = None;
        assert_eq!(
            blocked.authorize(true, true),
            Err(A1Blocker::ParameterPointNotFrozen)
        );
    }

    #[test]
    fn a1_requires_environment_and_a0_pass() {
        let spec = spec();
        assert_eq!(
            spec.authorize(false, true),
            Err(A1Blocker::EnvironmentNotQualified)
        );
        assert_eq!(
            spec.authorize(true, false),
            Err(A1Blocker::ArtifactIntegrityNotPassed)
        );
        assert_eq!(spec.authorize(true, true), Ok(()));
    }

    #[test]
    fn exact_single_fixed_point_evaluation_can_pass() {
        let spec = spec();
        let receipt = receipt();
        assert_eq!(receipt.effective_outcome(&spec), ExecutionOutcome::Pass);
        assert!(!receipt.licenses_scientific_claim());
    }

    #[test]
    fn optimizer_or_sampler_use_invalidates_a1() {
        let spec = spec();

        let mut optimizer = receipt();
        optimizer.minimizer_invoked = true;
        assert_eq!(optimizer.effective_outcome(&spec), ExecutionOutcome::Invalid);

        let mut sampler = receipt();
        sampler.sampler_invoked = true;
        assert_eq!(sampler.effective_outcome(&spec), ExecutionOutcome::Invalid);
    }

    #[test]
    fn extra_evaluation_or_parameter_mutation_invalidates_a1() {
        let spec = spec();

        let mut extra = receipt();
        extra.evaluation_count = 2;
        assert_eq!(extra.effective_outcome(&spec), ExecutionOutcome::Invalid);

        let mut mutated = receipt();
        mutated.parameter_mutation_detected = true;
        assert_eq!(mutated.effective_outcome(&spec), ExecutionOutcome::Invalid);
    }

    #[test]
    fn clean_out_of_tolerance_result_is_negative_not_invalid() {
        let spec = spec();
        let mut receipt = receipt();
        receipt.observed_chi2_bao = 10.40;
        assert_eq!(receipt.effective_outcome(&spec), ExecutionOutcome::Negative);
    }
}
