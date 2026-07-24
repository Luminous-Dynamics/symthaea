// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence that all four policies received the same candidate set and budget.

use crate::cognitive_experiment::CognitivePolicyArm;
use crate::evidence_digest::canonical_json_sha256;
use crate::experiment_manifest::{FrozenStudyManifest, StudySplit};
use crate::methodology_plan::{CandidateSetMode, FrozenMethodologyPlan};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const POLICY_BUDGET_EVIDENCE_VERSION: &str = "symthaea-muse-policy-budget-evidence-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArmBudgetUsage {
    pub arm: CognitivePolicyArm,
    pub candidate_set_sha256: String,
    pub candidates_evaluated: usize,
    pub theory_validations: usize,
    pub policy_evaluations: usize,
    pub policy_binary_sha256: String,
    pub policy_state_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixtureBudgetEvidence {
    pub fixture_id: String,
    pub candidate_set_sha256: String,
    pub candidate_count: usize,
    pub allowed_operators_sha256: String,
    pub compute_environment_sha256: String,
    pub arms: Vec<ArmBudgetUsage>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyBudgetEvidenceBundle {
    pub evidence_version: String,
    pub manifest_sha256: String,
    pub methodology_sha256: String,
    pub evidence_sha256: String,
    pub fixtures: Vec<FixtureBudgetEvidence>,
}

#[derive(Serialize)]
struct PolicyBudgetCommitment<'a> {
    evidence_version: &'a str,
    manifest_sha256: &'a str,
    methodology_sha256: &'a str,
    fixtures: &'a [FixtureBudgetEvidence],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyBudgetIssue {
    WrongVersion {
        found: String,
    },
    InvalidMethodology,
    SerializationFailed {
        field: String,
    },
    DigestMismatch {
        field: String,
    },
    DuplicateFixture {
        fixture_id: String,
    },
    UnknownFixture {
        fixture_id: String,
    },
    MissingFixture {
        fixture_id: String,
    },
    CandidateSetModeNotShared,
    CandidateCountMismatch {
        fixture_id: String,
        found: usize,
        expected: usize,
    },
    OperatorSetMismatch {
        fixture_id: String,
    },
    ComputeEnvironmentMismatch {
        fixture_id: String,
    },
    DuplicateArm {
        fixture_id: String,
        arm: CognitivePolicyArm,
    },
    MissingArm {
        fixture_id: String,
        arm: CognitivePolicyArm,
    },
    CandidateSetMismatch {
        fixture_id: String,
        arm: CognitivePolicyArm,
    },
    UnequalCandidateEvaluation {
        fixture_id: String,
        arm: CognitivePolicyArm,
        found: usize,
    },
    TheoryBudgetExceeded {
        fixture_id: String,
        arm: CognitivePolicyArm,
        found: usize,
    },
    PolicyBudgetExceeded {
        fixture_id: String,
        arm: CognitivePolicyArm,
        found: usize,
    },
    InvalidPolicyDigest {
        fixture_id: String,
        arm: CognitivePolicyArm,
        field: String,
    },
    MissingSymthaeaCheckpoint {
        fixture_id: String,
    },
    SymthaeaCheckpointMismatch {
        fixture_id: String,
    },
}

pub fn policy_budget_commitment(
    bundle: &PolicyBudgetEvidenceBundle,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&PolicyBudgetCommitment {
        evidence_version: &bundle.evidence_version,
        manifest_sha256: &bundle.manifest_sha256,
        methodology_sha256: &bundle.methodology_sha256,
        fixtures: &bundle.fixtures,
    })
}

pub fn seal_policy_budget_evidence(
    bundle: &mut PolicyBudgetEvidenceBundle,
) -> Result<(), serde_json::Error> {
    bundle.evidence_sha256 = policy_budget_commitment(bundle)?;
    Ok(())
}

pub fn validate_policy_budget_evidence(
    manifest: &FrozenStudyManifest,
    methodology: &FrozenMethodologyPlan,
    bundle: &PolicyBudgetEvidenceBundle,
) -> Vec<PolicyBudgetIssue> {
    let mut issues = Vec::new();
    if bundle.evidence_version != POLICY_BUDGET_EVIDENCE_VERSION {
        issues.push(PolicyBudgetIssue::WrongVersion {
            found: bundle.evidence_version.clone(),
        });
    }
    if !methodology.validate(manifest).is_empty() {
        issues.push(PolicyBudgetIssue::InvalidMethodology);
    }
    if methodology.policy_budget.candidate_set_mode != CandidateSetMode::SharedAcrossArms {
        issues.push(PolicyBudgetIssue::CandidateSetModeNotShared);
    }
    verify_digest(
        "manifest_sha256",
        canonical_json_sha256(manifest),
        &bundle.manifest_sha256,
        &mut issues,
    );
    verify_digest(
        "methodology_sha256",
        canonical_json_sha256(methodology),
        &bundle.methodology_sha256,
        &mut issues,
    );
    verify_digest(
        "evidence_sha256",
        policy_budget_commitment(bundle),
        &bundle.evidence_sha256,
        &mut issues,
    );

    let confirmatory: BTreeSet<_> = manifest
        .fixtures
        .iter()
        .filter(|fixture| fixture.split == StudySplit::Confirmatory)
        .map(|fixture| fixture.key.fixture_id.as_str())
        .collect();
    let mut seen_fixtures = BTreeSet::new();
    for fixture in &bundle.fixtures {
        if !seen_fixtures.insert(fixture.fixture_id.as_str()) {
            issues.push(PolicyBudgetIssue::DuplicateFixture {
                fixture_id: fixture.fixture_id.clone(),
            });
        }
        if !confirmatory.contains(fixture.fixture_id.as_str()) {
            issues.push(PolicyBudgetIssue::UnknownFixture {
                fixture_id: fixture.fixture_id.clone(),
            });
        }
        if fixture.candidate_count != methodology.policy_budget.candidates_per_fixture {
            issues.push(PolicyBudgetIssue::CandidateCountMismatch {
                fixture_id: fixture.fixture_id.clone(),
                found: fixture.candidate_count,
                expected: methodology.policy_budget.candidates_per_fixture,
            });
        }
        if fixture.allowed_operators_sha256 != methodology.policy_budget.allowed_operators_sha256 {
            issues.push(PolicyBudgetIssue::OperatorSetMismatch {
                fixture_id: fixture.fixture_id.clone(),
            });
        }
        if fixture.compute_environment_sha256
            != methodology.policy_budget.compute_environment_sha256
        {
            issues.push(PolicyBudgetIssue::ComputeEnvironmentMismatch {
                fixture_id: fixture.fixture_id.clone(),
            });
        }
        let mut arms = BTreeMap::new();
        for usage in &fixture.arms {
            if arms.insert(usage.arm, usage).is_some() {
                issues.push(PolicyBudgetIssue::DuplicateArm {
                    fixture_id: fixture.fixture_id.clone(),
                    arm: usage.arm,
                });
            }
            if usage.candidate_set_sha256 != fixture.candidate_set_sha256 {
                issues.push(PolicyBudgetIssue::CandidateSetMismatch {
                    fixture_id: fixture.fixture_id.clone(),
                    arm: usage.arm,
                });
            }
            if usage.candidates_evaluated != fixture.candidate_count {
                issues.push(PolicyBudgetIssue::UnequalCandidateEvaluation {
                    fixture_id: fixture.fixture_id.clone(),
                    arm: usage.arm,
                    found: usage.candidates_evaluated,
                });
            }
            if usage.theory_validations > methodology.policy_budget.max_theory_validations_per_arm {
                issues.push(PolicyBudgetIssue::TheoryBudgetExceeded {
                    fixture_id: fixture.fixture_id.clone(),
                    arm: usage.arm,
                    found: usage.theory_validations,
                });
            }
            if usage.policy_evaluations > methodology.policy_budget.max_policy_evaluations_per_arm {
                issues.push(PolicyBudgetIssue::PolicyBudgetExceeded {
                    fixture_id: fixture.fixture_id.clone(),
                    arm: usage.arm,
                    found: usage.policy_evaluations,
                });
            }
            if !is_sha256(&usage.policy_binary_sha256) {
                issues.push(PolicyBudgetIssue::InvalidPolicyDigest {
                    fixture_id: fixture.fixture_id.clone(),
                    arm: usage.arm,
                    field: "policy_binary_sha256".into(),
                });
            }
            if usage
                .policy_state_sha256
                .as_ref()
                .is_some_and(|digest| !is_sha256(digest))
            {
                issues.push(PolicyBudgetIssue::InvalidPolicyDigest {
                    fixture_id: fixture.fixture_id.clone(),
                    arm: usage.arm,
                    field: "policy_state_sha256".into(),
                });
            }
        }
        for arm in CognitivePolicyArm::ALL {
            if !arms.contains_key(&arm) {
                issues.push(PolicyBudgetIssue::MissingArm {
                    fixture_id: fixture.fixture_id.clone(),
                    arm,
                });
            }
        }
        if let Some(symthaea) = arms.get(&CognitivePolicyArm::Symthaea) {
            match symthaea.policy_state_sha256.as_deref() {
                None => issues.push(PolicyBudgetIssue::MissingSymthaeaCheckpoint {
                    fixture_id: fixture.fixture_id.clone(),
                }),
                Some(value) if value != methodology.model_checkpoint.checkpoint_sha256 => {
                    issues.push(PolicyBudgetIssue::SymthaeaCheckpointMismatch {
                        fixture_id: fixture.fixture_id.clone(),
                    });
                }
                Some(_) => {}
            }
        }
    }
    for fixture_id in confirmatory {
        if !seen_fixtures.contains(fixture_id) {
            issues.push(PolicyBudgetIssue::MissingFixture {
                fixture_id: fixture_id.into(),
            });
        }
    }
    issues
}

fn verify_digest(
    field: &str,
    result: Result<String, serde_json::Error>,
    expected: &str,
    issues: &mut Vec<PolicyBudgetIssue>,
) {
    match result {
        Ok(value) if value == expected => {}
        Ok(_) => issues.push(PolicyBudgetIssue::DigestMismatch {
            field: field.into(),
        }),
        Err(_) => issues.push(PolicyBudgetIssue::SerializationFailed {
            field: field.into(),
        }),
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    #[test]
    fn commitment_detects_budget_record_tampering() {
        let mut bundle = PolicyBudgetEvidenceBundle {
            evidence_version: POLICY_BUDGET_EVIDENCE_VERSION.into(),
            manifest_sha256: DIGEST.into(),
            methodology_sha256: DIGEST.into(),
            evidence_sha256: String::new(),
            fixtures: vec![FixtureBudgetEvidence {
                fixture_id: "fixture-1".into(),
                candidate_set_sha256: DIGEST.into(),
                candidate_count: 5,
                allowed_operators_sha256: DIGEST.into(),
                compute_environment_sha256: DIGEST.into(),
                arms: Vec::new(),
            }],
        };
        seal_policy_budget_evidence(&mut bundle).unwrap();
        let sealed = bundle.evidence_sha256.clone();
        bundle.fixtures[0].candidate_count += 1;
        assert_ne!(policy_budget_commitment(&bundle).unwrap(), sealed);
    }
}
