// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Frozen candidate family and replay-corpus acquisition for SYM-RSI-001.
//!
//! Candidate identities, behavior salts, domains, seed partitions, and collection
//! procedure are fixed before any measured comparison. Training and held-out replay
//! corpora are acquired independently from real deterministic fixture executions.
//! The held-out corpus is never passed to the training-only selector.

use super::sym_rsi_experiment::{EvaluationSplit, SymRsiExperimentManifest};
use super::sym_rsi_fixtures::{
    canonical_sym_rsi_001_fixture_manifest, FixtureDomainKind,
};
use super::sym_rsi_replay_corpus::{merge_observed_traces, ReplayCorpusError, ReplayFixtureWorld};
use super::sym_rsi_replay_selection::FixedHashCandidateSpec;
use super::sym_rsi_runner::{run_fixture_policy, FixtureRunnerError};
use serde::{Deserialize, Serialize};

pub const SYM_RSI_001_CANDIDATE_FAMILY_SCHEMA: &str =
    "symthaea.sym-rsi-001.candidate-family.v1";
pub const SYM_RSI_001_REPLAY_CORPUS_SCHEMA: &str =
    "symthaea.sym-rsi-001.replay-corpus-acquisition.v1";
pub const SYM_RSI_001_INCUMBENT_POLICY_ID: &str = "sym-rsi-fixed-hash-incumbent-v1";

const CANONICAL_POLICY_SPECS: [(&str, u64); 8] = [
    (SYM_RSI_001_INCUMBENT_POLICY_ID, 7),
    ("sym-rsi-fixed-hash-c01-v1", 11),
    ("sym-rsi-fixed-hash-c02-v1", 17),
    ("sym-rsi-fixed-hash-c03-v1", 23),
    ("sym-rsi-fixed-hash-c04-v1", 31),
    ("sym-rsi-fixed-hash-c05-v1", 47),
    ("sym-rsi-fixed-hash-c06-v1", 61),
    ("sym-rsi-fixed-hash-c07-v1", 89),
];

pub fn canonical_fixed_hash_candidate_family() -> Vec<FixedHashCandidateSpec> {
    CANONICAL_POLICY_SPECS
        .into_iter()
        .map(|(policy_id, salt)| FixedHashCandidateSpec {
            policy_id: policy_id.into(),
            salt,
        })
        .collect()
}

pub fn canonical_candidate_family_digest() -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-001.candidate-family.v1\0");
    for (policy_id, salt) in CANONICAL_POLICY_SPECS {
        hasher.update(&(policy_id.len() as u64).to_le_bytes());
        hasher.update(policy_id.as_bytes());
        hasher.update(&salt.to_le_bytes());
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayCorpusAcquisitionReceipt {
    pub schema: String,
    pub experiment_id: String,
    pub preregistration_digest: String,
    pub subject_digest: String,
    pub environment_digest: String,
    pub candidate_family_schema: String,
    pub candidate_family_digest: String,
    pub split: EvaluationSplit,
    pub domain_count: usize,
    pub seed_count_per_domain: usize,
    pub world_count: usize,
    pub trajectory_count: usize,
    pub collector_policy_ids: Vec<String>,
    pub evidence_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenReplayCorpus {
    pub receipt: ReplayCorpusAcquisitionReceipt,
    pub worlds: Vec<ReplayFixtureWorld>,
}

impl FrozenReplayCorpus {
    pub fn split(&self) -> EvaluationSplit {
        self.receipt.split
    }

    pub fn worlds(&self) -> &[ReplayFixtureWorld] {
        &self.worlds
    }
}

/// Acquire the frozen training or held-out replay corpus using exactly the
/// canonical candidate family. Fresh-execution and OOD seeds are intentionally
/// inaccessible through this API so they cannot be consumed during replay setup.
pub fn acquire_canonical_replay_corpus(
    manifest: &SymRsiExperimentManifest,
    split: EvaluationSplit,
) -> Result<FrozenReplayCorpus, ReplayAcquisitionError> {
    ensure_canonical_manifest(manifest)?;
    if !matches!(
        split,
        EvaluationSplit::TrainingReplay | EvaluationSplit::HeldOutReplay
    ) {
        return Err(ReplayAcquisitionError::ForbiddenAcquisitionSplit(split));
    }

    let candidates = canonical_fixed_hash_candidate_family();
    let mut worlds = Vec::new();
    let mut trajectory_bindings = Vec::new();
    let mut seed_count_per_domain = None;

    for domain in FixtureDomainKind::ALL {
        let domain_spec = manifest
            .domains
            .iter()
            .find(|spec| spec.domain_id == domain.id())
            .ok_or_else(|| ReplayAcquisitionError::MissingDomain(domain.id().into()))?;
        let seeds = match split {
            EvaluationSplit::TrainingReplay => &domain_spec.seeds.training_replay,
            EvaluationSplit::HeldOutReplay => &domain_spec.seeds.held_out_replay,
            _ => unreachable!("split restricted above"),
        };

        match seed_count_per_domain {
            None => seed_count_per_domain = Some(seeds.len()),
            Some(expected) if expected != seeds.len() => {
                return Err(ReplayAcquisitionError::UnequalSeedCount {
                    domain_id: domain.id().into(),
                    expected,
                    observed: seeds.len(),
                });
            }
            Some(_) => {}
        }

        for &seed in seeds {
            let mut traces = Vec::with_capacity(candidates.len());
            for candidate in &candidates {
                let mut policy = candidate.policy();
                let trace = run_fixture_policy(manifest, domain, split, seed, &mut policy)
                    .map_err(ReplayAcquisitionError::Runner)?;
                trajectory_bindings.push(TrajectoryBinding {
                    domain_id: domain.id().into(),
                    seed,
                    policy_id: candidate.policy_id.clone(),
                    final_evidence_digest: trace.evidence_digest.clone(),
                    evaluator_calls: trace.evaluator_calls,
                });
                traces.push(trace);
            }
            worlds.push(
                merge_observed_traces(&traces).map_err(ReplayAcquisitionError::ReplayCorpus)?,
            );
        }
    }

    let evidence_digest = replay_corpus_evidence_digest(
        manifest,
        split,
        &candidates,
        &trajectory_bindings,
    );
    let collector_policy_ids = candidates
        .iter()
        .map(|candidate| candidate.policy_id.clone())
        .collect::<Vec<_>>();
    let seed_count_per_domain = seed_count_per_domain.unwrap_or(0);

    Ok(FrozenReplayCorpus {
        receipt: ReplayCorpusAcquisitionReceipt {
            schema: SYM_RSI_001_REPLAY_CORPUS_SCHEMA.into(),
            experiment_id: manifest.experiment_id.clone(),
            preregistration_digest: manifest.preregistration_digest.clone(),
            subject_digest: manifest.subject_digest.clone(),
            environment_digest: manifest.environment_digest.clone(),
            candidate_family_schema: SYM_RSI_001_CANDIDATE_FAMILY_SCHEMA.into(),
            candidate_family_digest: canonical_candidate_family_digest(),
            split,
            domain_count: FixtureDomainKind::ALL.len(),
            seed_count_per_domain,
            world_count: worlds.len(),
            trajectory_count: trajectory_bindings.len(),
            collector_policy_ids,
            evidence_digest,
        },
        worlds,
    })
}

fn ensure_canonical_manifest(
    manifest: &SymRsiExperimentManifest,
) -> Result<(), ReplayAcquisitionError> {
    manifest
        .validate()
        .map_err(ReplayAcquisitionError::ManifestInvalid)?;
    let expected = canonical_sym_rsi_001_fixture_manifest(
        manifest.preregistration_digest.clone(),
        manifest.subject_digest.clone(),
        manifest.environment_digest.clone(),
    );
    if manifest != &expected {
        return Err(ReplayAcquisitionError::ManifestNotCanonical);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TrajectoryBinding {
    domain_id: String,
    seed: u64,
    policy_id: String,
    final_evidence_digest: String,
    evaluator_calls: u64,
}

fn replay_corpus_evidence_digest(
    manifest: &SymRsiExperimentManifest,
    split: EvaluationSplit,
    candidates: &[FixedHashCandidateSpec],
    trajectories: &[TrajectoryBinding],
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-001.replay-corpus-acquisition.v1\0");
    for value in [
        manifest.experiment_id.as_str(),
        manifest.preregistration_digest.as_str(),
        manifest.subject_digest.as_str(),
        manifest.environment_digest.as_str(),
        canonical_candidate_family_digest().as_str(),
    ] {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    hasher.update(&[split_tag(split)]);
    for candidate in candidates {
        hasher.update(&(candidate.policy_id.len() as u64).to_le_bytes());
        hasher.update(candidate.policy_id.as_bytes());
        hasher.update(&candidate.salt.to_le_bytes());
    }
    for trajectory in trajectories {
        for value in [
            trajectory.domain_id.as_str(),
            trajectory.policy_id.as_str(),
            trajectory.final_evidence_digest.as_str(),
        ] {
            hasher.update(&(value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
        hasher.update(&trajectory.seed.to_le_bytes());
        hasher.update(&trajectory.evaluator_calls.to_le_bytes());
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn split_tag(split: EvaluationSplit) -> u8 {
    match split {
        EvaluationSplit::TrainingReplay => 0,
        EvaluationSplit::HeldOutReplay => 1,
        EvaluationSplit::FreshExecution => 2,
        EvaluationSplit::OutOfDistribution => 3,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayAcquisitionError {
    ManifestInvalid(super::sym_rsi_experiment::ExperimentHarnessError),
    ManifestNotCanonical,
    ForbiddenAcquisitionSplit(EvaluationSplit),
    MissingDomain(String),
    UnequalSeedCount {
        domain_id: String,
        expected: usize,
        observed: usize,
    },
    Runner(FixtureRunnerError),
    ReplayCorpus(ReplayCorpusError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_family_is_frozen_and_unique() {
        let family = canonical_fixed_hash_candidate_family();
        assert_eq!(family.len(), 8);
        assert_eq!(family[0].policy_id, SYM_RSI_001_INCUMBENT_POLICY_ID);
        assert_eq!(family[0].salt, 7);
        let mut ids = family
            .iter()
            .map(|candidate| candidate.policy_id.as_str())
            .collect::<Vec<_>>();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), family.len());
        let mut salts = family.iter().map(|candidate| candidate.salt).collect::<Vec<_>>();
        salts.sort_unstable();
        salts.dedup();
        assert_eq!(salts.len(), family.len());
        assert!(canonical_candidate_family_digest().starts_with("blake3:"));
    }

    #[test]
    fn training_corpus_uses_only_frozen_training_seeds_and_all_collectors() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let corpus = acquire_canonical_replay_corpus(&manifest, EvaluationSplit::TrainingReplay)
            .unwrap();
        assert_eq!(corpus.receipt.domain_count, 3);
        assert_eq!(corpus.receipt.seed_count_per_domain, 8);
        assert_eq!(corpus.receipt.world_count, 24);
        assert_eq!(corpus.receipt.trajectory_count, 24 * 8);
        assert_eq!(corpus.worlds.len(), 24);
        assert!(corpus
            .worlds
            .iter()
            .all(|world| world.split == EvaluationSplit::TrainingReplay));
        assert!(corpus.receipt.evidence_digest.starts_with("blake3:"));
    }

    #[test]
    fn held_out_corpus_is_separate_and_smaller() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let corpus = acquire_canonical_replay_corpus(&manifest, EvaluationSplit::HeldOutReplay)
            .unwrap();
        assert_eq!(corpus.receipt.seed_count_per_domain, 4);
        assert_eq!(corpus.receipt.world_count, 12);
        assert_eq!(corpus.receipt.trajectory_count, 12 * 8);
        assert!(corpus
            .worlds
            .iter()
            .all(|world| world.split == EvaluationSplit::HeldOutReplay));
    }

    #[test]
    fn fresh_and_ood_splits_cannot_be_preconsumed_as_replay_corpora() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        assert_eq!(
            acquire_canonical_replay_corpus(&manifest, EvaluationSplit::FreshExecution),
            Err(ReplayAcquisitionError::ForbiddenAcquisitionSplit(
                EvaluationSplit::FreshExecution
            ))
        );
        assert_eq!(
            acquire_canonical_replay_corpus(&manifest, EvaluationSplit::OutOfDistribution),
            Err(ReplayAcquisitionError::ForbiddenAcquisitionSplit(
                EvaluationSplit::OutOfDistribution
            ))
        );
    }

    #[test]
    fn modified_manifest_starts_a_different_lineage_instead_of_reusing_corpus_api() {
        let mut manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        manifest.beta_cost = 0.051;
        assert_eq!(
            acquire_canonical_replay_corpus(&manifest, EvaluationSplit::TrainingReplay),
            Err(ReplayAcquisitionError::ManifestNotCanonical)
        );
    }
}
