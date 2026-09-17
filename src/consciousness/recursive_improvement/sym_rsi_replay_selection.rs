// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fail-closed replay policy promotion for SYM-RSI-001.
//!
//! Unknown replay branches are epistemically missing data, not cheap evaluations.
//! A policy is therefore eligible for replay-based promotion only if every action it
//! takes across the frozen replay corpus is historically supported and reaches a
//! recorded terminal state. Selection among eligible candidates uses the bounded
//! replay objective and preserves the incumbent on exact ties.
//!
//! Policy selection is additionally restricted to the preregistered training-replay
//! split. Held-out replay, fresh execution, and OOD worlds are evaluation evidence,
//! never selection evidence.

use super::replay_policy::{select_replay_policy, PolicySelectionError, ReplayPolicyScore};
use super::sym_rsi_experiment::{
    EvaluationSplit, ExperimentHarnessError, SymRsiExperimentManifest,
};
use super::sym_rsi_replay_corpus::{
    score_policy_on_replay_worlds, ReplayCorpusError, ReplayCorpusEvaluation, ReplayFixtureWorld,
};
use super::sym_rsi_runner::FixedHashPolicy;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const SYM_RSI_001_REPLAY_SELECTION_SCHEMA: &str =
    "symthaea.sym-rsi-001.replay-selection.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixedHashCandidateSpec {
    pub policy_id: String,
    pub salt: u64,
}

impl FixedHashCandidateSpec {
    pub fn policy(&self) -> FixedHashPolicy {
        FixedHashPolicy::new(self.policy_id.clone(), self.salt)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplayIneligibility {
    UnsupportedAction,
    IncompleteCoverage,
    DidNotReachRecordedTerminal,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayCandidateAssessment {
    pub spec: FixedHashCandidateSpec,
    pub evaluation: ReplayCorpusEvaluation,
    pub objective: f64,
    pub eligible: bool,
    pub ineligibility: Option<ReplayIneligibility>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplaySelectionReceipt {
    pub schema: String,
    pub experiment_id: String,
    pub preregistration_digest: String,
    pub subject_digest: String,
    pub environment_digest: String,
    pub incumbent_policy_id: String,
    pub selected_policy_id: String,
    pub beta_cost: f64,
    pub beta_parallelism: f64,
    pub full_historical_support_required: bool,
    pub selection_split: EvaluationSplit,
    pub incumbent_objective: f64,
    pub selected_objective: f64,
    pub strict_replay_improvement: bool,
    pub assessments: Vec<ReplayCandidateAssessment>,
    pub evidence_digest: String,
}

impl ReplaySelectionReceipt {
    pub fn selected_spec(&self) -> Option<&FixedHashCandidateSpec> {
        self.assessments
            .iter()
            .find(|assessment| assessment.spec.policy_id == self.selected_policy_id)
            .map(|assessment| &assessment.spec)
    }
}

pub fn select_fixed_hash_policy_from_replay(
    manifest: &SymRsiExperimentManifest,
    incumbent_policy_id: &str,
    candidates: &[FixedHashCandidateSpec],
    worlds: &[ReplayFixtureWorld],
) -> Result<ReplaySelectionReceipt, ReplaySelectionError> {
    manifest
        .validate()
        .map_err(ReplaySelectionError::ManifestInvalid)?;
    if candidates.is_empty() {
        return Err(ReplaySelectionError::EmptyCandidates);
    }

    let mut policy_ids = BTreeSet::new();
    let mut salts = BTreeSet::new();
    for candidate in candidates {
        if candidate.policy_id.trim().is_empty() {
            return Err(ReplaySelectionError::EmptyPolicyId);
        }
        if !policy_ids.insert(candidate.policy_id.as_str()) {
            return Err(ReplaySelectionError::DuplicatePolicyId(
                candidate.policy_id.clone(),
            ));
        }
        if !salts.insert(candidate.salt) {
            return Err(ReplaySelectionError::DuplicateSalt(candidate.salt));
        }
    }
    if !candidates
        .iter()
        .any(|candidate| candidate.policy_id == incumbent_policy_id)
    {
        return Err(ReplaySelectionError::IncumbentMissing(
            incumbent_policy_id.to_owned(),
        ));
    }

    if worlds.is_empty() {
        return Err(ReplaySelectionError::EmptyReplayWorlds);
    }
    if let Some(world) = worlds
        .iter()
        .find(|world| world.split != EvaluationSplit::TrainingReplay)
    {
        return Err(ReplaySelectionError::NonTrainingReplayWorld {
            domain_id: world.domain.id().to_owned(),
            split: world.split,
            seed: world.seed,
        });
    }

    let mut assessments = Vec::with_capacity(candidates.len());
    let mut eligible_scores: Vec<ReplayPolicyScore> = Vec::new();

    for candidate in candidates {
        let policy = candidate.policy();
        let evaluation = score_policy_on_replay_worlds(&policy, worlds)
            .map_err(ReplaySelectionError::ReplayCorpus)?;
        let score = evaluation.as_policy_score();
        let objective = score.objective(manifest.beta_cost, manifest.beta_parallelism);
        if !objective.is_finite() {
            return Err(ReplaySelectionError::NonFiniteObjective(
                candidate.policy_id.clone(),
            ));
        }

        let ineligibility = if evaluation.unsupported_worlds > 0 {
            Some(ReplayIneligibility::UnsupportedAction)
        } else if evaluation.replay_coverage != 1.0 {
            Some(ReplayIneligibility::IncompleteCoverage)
        } else if evaluation.terminal_worlds != evaluation.world_count {
            Some(ReplayIneligibility::DidNotReachRecordedTerminal)
        } else {
            None
        };
        let eligible = ineligibility.is_none();
        if eligible {
            eligible_scores.push(score);
        }

        assessments.push(ReplayCandidateAssessment {
            spec: candidate.clone(),
            evaluation,
            objective,
            eligible,
            ineligibility,
        });
    }

    let incumbent_assessment = assessments
        .iter()
        .find(|assessment| assessment.spec.policy_id == incumbent_policy_id)
        .expect("incumbent presence checked above");
    if !incumbent_assessment.eligible {
        return Err(ReplaySelectionError::IncumbentNotFullySupported);
    }
    if eligible_scores.is_empty() {
        return Err(ReplaySelectionError::NoEligibleCandidates);
    }

    let selected = select_replay_policy(
        incumbent_policy_id,
        &eligible_scores,
        manifest.beta_cost,
        manifest.beta_parallelism,
    )
    .map_err(ReplaySelectionError::PolicySelection)?;
    let selected_assessment = assessments
        .iter()
        .find(|assessment| assessment.spec.policy_id == selected.policy_id)
        .expect("eligible score came from an assessment");

    let strict_replay_improvement = selected.policy_id != incumbent_policy_id;
    let evidence_digest = selection_evidence_digest(
        manifest,
        incumbent_policy_id,
        &selected.policy_id,
        &assessments,
    );

    Ok(ReplaySelectionReceipt {
        schema: SYM_RSI_001_REPLAY_SELECTION_SCHEMA.into(),
        experiment_id: manifest.experiment_id.clone(),
        preregistration_digest: manifest.preregistration_digest.clone(),
        subject_digest: manifest.subject_digest.clone(),
        environment_digest: manifest.environment_digest.clone(),
        incumbent_policy_id: incumbent_policy_id.to_owned(),
        selected_policy_id: selected.policy_id.clone(),
        beta_cost: manifest.beta_cost,
        beta_parallelism: manifest.beta_parallelism,
        full_historical_support_required: true,
        selection_split: EvaluationSplit::TrainingReplay,
        incumbent_objective: incumbent_assessment.objective,
        selected_objective: selected_assessment.objective,
        strict_replay_improvement,
        assessments,
        evidence_digest,
    })
}

fn selection_evidence_digest(
    manifest: &SymRsiExperimentManifest,
    incumbent_policy_id: &str,
    selected_policy_id: &str,
    assessments: &[ReplayCandidateAssessment],
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-001.replay-selection-evidence.v1\0");
    for value in [
        manifest.experiment_id.as_str(),
        manifest.preregistration_digest.as_str(),
        manifest.subject_digest.as_str(),
        manifest.environment_digest.as_str(),
        incumbent_policy_id,
        selected_policy_id,
    ] {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    hasher.update(&manifest.beta_cost.to_bits().to_le_bytes());
    hasher.update(&manifest.beta_parallelism.to_bits().to_le_bytes());
    hasher.update(&[0]); // EvaluationSplit::TrainingReplay domain tag.
    for assessment in assessments {
        hasher.update(&(assessment.spec.policy_id.len() as u64).to_le_bytes());
        hasher.update(assessment.spec.policy_id.as_bytes());
        hasher.update(&assessment.spec.salt.to_le_bytes());
        hasher.update(&assessment.objective.to_bits().to_le_bytes());
        hasher.update(&assessment.evaluation.replay_coverage.to_bits().to_le_bytes());
        hasher.update(&(assessment.evaluation.unsupported_worlds as u64).to_le_bytes());
        hasher.update(&(assessment.evaluation.terminal_worlds as u64).to_le_bytes());
        hasher.update(&[u8::from(assessment.eligible)]);
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplaySelectionError {
    ManifestInvalid(ExperimentHarnessError),
    EmptyCandidates,
    EmptyPolicyId,
    DuplicatePolicyId(String),
    DuplicateSalt(u64),
    IncumbentMissing(String),
    EmptyReplayWorlds,
    NonTrainingReplayWorld {
        domain_id: String,
        split: EvaluationSplit,
        seed: u64,
    },
    IncumbentNotFullySupported,
    NoEligibleCandidates,
    NonFiniteObjective(String),
    ReplayCorpus(ReplayCorpusError),
    PolicySelection(PolicySelectionError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::recursive_improvement::{
        canonical_sym_rsi_001_fixture_manifest, merge_observed_traces, run_fixture_policy,
        EvaluationSplit, FixtureDomainKind,
    };

    #[test]
    fn unsupported_candidate_is_ineligible_not_cheap() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let collector_specs = [
            FixedHashCandidateSpec {
                policy_id: "incumbent".into(),
                salt: 7,
            },
            FixedHashCandidateSpec {
                policy_id: "collector-b".into(),
                salt: 11,
            },
        ];
        let mut traces = Vec::new();
        for spec in &collector_specs {
            let mut policy = spec.policy();
            traces.push(
                run_fixture_policy(
                    &manifest,
                    FixtureDomainKind::BranchingSearch,
                    EvaluationSplit::TrainingReplay,
                    1,
                    &mut policy,
                )
                .unwrap(),
            );
        }
        let world = merge_observed_traces(&traces).unwrap();

        let mut unseen = None;
        for salt in 100..10_000 {
            let spec = FixedHashCandidateSpec {
                policy_id: format!("unseen-{salt}"),
                salt,
            };
            let policy = spec.policy();
            let eval = score_policy_on_replay_worlds(&policy, std::slice::from_ref(&world)).unwrap();
            if eval.unsupported_worlds > 0 {
                unseen = Some(spec);
                break;
            }
        }
        let unseen = unseen.expect("fixture should expose at least one unobserved branch");

        let candidates = vec![collector_specs[0].clone(), unseen.clone()];
        let receipt = select_fixed_hash_policy_from_replay(
            &manifest,
            "incumbent",
            &candidates,
            &[world],
        )
        .unwrap();
        let unseen_assessment = receipt
            .assessments
            .iter()
            .find(|a| a.spec.policy_id == unseen.policy_id)
            .unwrap();
        assert!(!unseen_assessment.eligible);
        assert_eq!(
            unseen_assessment.ineligibility,
            Some(ReplayIneligibility::UnsupportedAction)
        );
        assert_eq!(receipt.selected_policy_id, "incumbent");
    }

    #[test]
    fn selected_policy_is_always_fully_supported() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let candidates = vec![
            FixedHashCandidateSpec {
                policy_id: "incumbent".into(),
                salt: 7,
            },
            FixedHashCandidateSpec {
                policy_id: "candidate".into(),
                salt: 11,
            },
        ];
        let mut traces = Vec::new();
        for spec in &candidates {
            let mut policy = spec.policy();
            traces.push(
                run_fixture_policy(
                    &manifest,
                    FixtureDomainKind::RuggedOptimization,
                    EvaluationSplit::TrainingReplay,
                    1,
                    &mut policy,
                )
                .unwrap(),
            );
        }
        let world = merge_observed_traces(&traces).unwrap();
        let receipt = select_fixed_hash_policy_from_replay(
            &manifest,
            "incumbent",
            &candidates,
            &[world],
        )
        .unwrap();
        let selected = receipt
            .assessments
            .iter()
            .find(|a| a.spec.policy_id == receipt.selected_policy_id)
            .unwrap();
        assert!(selected.eligible);
        assert_eq!(selected.evaluation.replay_coverage, 1.0);
        assert_eq!(
            selected.evaluation.terminal_worlds,
            selected.evaluation.world_count
        );
        assert_eq!(receipt.selection_split, EvaluationSplit::TrainingReplay);
        assert!(receipt.evidence_digest.starts_with("blake3:"));
    }

    #[test]
    fn duplicate_behavior_is_rejected_even_under_different_id() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let candidates = vec![
            FixedHashCandidateSpec {
                policy_id: "incumbent".into(),
                salt: 7,
            },
            FixedHashCandidateSpec {
                policy_id: "alias".into(),
                salt: 7,
            },
        ];
        assert_eq!(
            select_fixed_hash_policy_from_replay(&manifest, "incumbent", &candidates, &[]),
            Err(ReplaySelectionError::DuplicateSalt(7))
        );
    }

    #[test]
    fn held_out_replay_is_rejected_as_selection_evidence() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let spec = FixedHashCandidateSpec {
            policy_id: "incumbent".into(),
            salt: 7,
        };
        let mut policy = spec.policy();
        let trace = run_fixture_policy(
            &manifest,
            FixtureDomainKind::BranchingSearch,
            EvaluationSplit::HeldOutReplay,
            101,
            &mut policy,
        )
        .unwrap();
        let world = merge_observed_traces(&[trace]).unwrap();
        assert_eq!(
            select_fixed_hash_policy_from_replay(
                &manifest,
                "incumbent",
                &[spec],
                &[world],
            ),
            Err(ReplaySelectionError::NonTrainingReplayWorld {
                domain_id: FixtureDomainKind::BranchingSearch.id().into(),
                split: EvaluationSplit::HeldOutReplay,
                seed: 101,
            })
        );
    }
}
