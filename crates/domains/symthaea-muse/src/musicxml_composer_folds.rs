// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic, replayable composer-disjoint fold generation.
//!
//! This module creates a MUSE out-of-distribution evaluation protocol from a
//! pinned source sample inventory. It does not download or mutate corpus data.
//! The official source benchmark splits remain separate; generated folds are
//! a new, explicitly labelled MUSE evaluation track.

use crate::evidence_digest::{canonical_json_sha256, sha256_hex};
use crate::musicxml_corpus_qualification::{
    CorpusQualificationIssueV1, CorpusRegistrationV1, CorpusSampleManifestV1, CorpusSampleV1,
    CorpusSplitRoleV1, CorpusTrackRegistrationV1, MUSICXML_CORPUS_SAMPLE_MANIFEST_SCHEMA_V1,
    validate_corpus_registration, validate_corpus_sample_manifest,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const COMPOSER_FOLD_PLAN_SCHEMA_V1: &str = "muse-composer-disjoint-fold-plan-v1";
pub const COMPOSER_FOLD_ALGORITHM_V1: &str =
    "muse-largest-first-work-balanced-seeded-tie-order-v1";
pub const MIN_COMPOSER_FOLDS_V1: usize = 2;
pub const MAX_COMPOSER_FOLDS_V1: usize = 10;
pub const MAX_COMPOSER_FOLD_SAMPLES_V1: usize = 100_000;

/// One item from the acquired source inventory, before assigning any split.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComposerFoldCandidateV1 {
    pub sample_id: String,
    pub composer_group_id: String,
    pub work_group_id: String,
    pub input_relative_path: String,
    pub reference_relative_path: String,
    pub input_sha256: String,
    pub reference_sha256: String,
}

/// The public seed is already a SHA-256 commitment. It is part of the frozen
/// protocol, not a private randomness source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComposerFoldConfigV1 {
    pub fold_count: usize,
    pub seed_sha256: String,
    pub algorithm_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComposerFoldGroupAssignmentV1 {
    pub composer_group_id: String,
    pub fold_index: usize,
    pub unique_work_count: usize,
    pub sample_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComposerFoldPlanV1 {
    pub schema: String,
    pub config: ComposerFoldConfigV1,
    pub corpus_id: String,
    pub release_id: String,
    pub track_id: String,
    pub source_revision_sha256: String,
    pub source_inventory_sha256: String,
    pub registration_sha256: String,
    /// Sorted by composer_group_id for canonical display. Fold assignment is
    /// determined by the frozen seeded largest-first balancing algorithm.
    pub assignments: Vec<ComposerFoldGroupAssignmentV1>,
    /// One full manifest per fold. Every source sample is Test exactly once
    /// and Train in all other folds.
    pub fold_manifests: Vec<CorpusSampleManifestV1>,
    pub fold_manifest_sha256: Vec<String>,
    pub plan_sha256: String,
}

#[derive(Serialize)]
struct ComposerFoldPlanCommitmentV1<'a> {
    schema: &'a str,
    config: &'a ComposerFoldConfigV1,
    corpus_id: &'a str,
    release_id: &'a str,
    track_id: &'a str,
    source_revision_sha256: &'a str,
    source_inventory_sha256: &'a str,
    registration_sha256: &'a str,
    assignments: &'a [ComposerFoldGroupAssignmentV1],
    fold_manifests: &'a [CorpusSampleManifestV1],
    fold_manifest_sha256: &'a [String],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComposerFoldIssueV1 {
    InvalidFoldCount { count: usize },
    InvalidSeedDigest,
    WrongAlgorithmVersion,
    EmptySourceInventory,
    SourceInventoryTooLarge { count: usize, maximum: usize },
    InvalidRegistration(Vec<CorpusQualificationIssueV1>),
    UnknownTrack { track_id: String },
    UnsupportedSplitPolicy,
    TrainingNotAllowed,
    EmptySampleId,
    DuplicateSampleId { sample_id: String },
    NonCanonicalSampleId { sample_id: String },
    MissingComposerGroup { sample_id: String },
    MissingWorkGroup { sample_id: String },
    NonCanonicalComposerGroup { sample_id: String },
    NonCanonicalWorkGroup { sample_id: String },
    WorkMappedToMultipleComposers {
        work_group_id: String,
        first_composer: String,
        second_composer: String,
    },
    InsufficientComposerGroups { count: usize, folds: usize },
    InvalidFoldManifest {
        fold_index: usize,
        issues: Vec<CorpusQualificationIssueV1>,
    },
    SerializationFailed,
}

#[derive(Debug, Clone)]
struct ComposerGroup {
    work_ids: BTreeSet<String>,
    sample_count: usize,
    stable_order_digest: String,
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn inventory_commitment(
    candidates: &[ComposerFoldCandidateV1],
) -> Result<String, ComposerFoldIssueV1> {
    let mut ordered = candidates.to_vec();
    ordered.sort_by(|left, right| left.sample_id.cmp(&right.sample_id));
    canonical_json_sha256(&ordered).map_err(|_| ComposerFoldIssueV1::SerializationFailed)
}

fn stable_composer_order_digest(
    seed_sha256: &str,
    composer_group_id: &str,
) -> String {
    let material = format!(
        "muse-composer-disjoint-fold-order-v1\0{seed_sha256}\0{composer_group_id}"
    );
    sha256_hex(material.as_bytes())
}

fn plan_commitment(plan: &ComposerFoldPlanV1) -> Result<String, ComposerFoldIssueV1> {
    canonical_json_sha256(&ComposerFoldPlanCommitmentV1 {
        schema: &plan.schema,
        config: &plan.config,
        corpus_id: &plan.corpus_id,
        release_id: &plan.release_id,
        track_id: &plan.track_id,
        source_revision_sha256: &plan.source_revision_sha256,
        source_inventory_sha256: &plan.source_inventory_sha256,
        registration_sha256: &plan.registration_sha256,
        assignments: &plan.assignments,
        fold_manifests: &plan.fold_manifests,
        fold_manifest_sha256: &plan.fold_manifest_sha256,
    })
    .map_err(|_| ComposerFoldIssueV1::SerializationFailed)
}

/// Generate a balanced composer-held-out K-fold plan.
///
/// Balance objective, in order:
/// 1. minimize the number of unique works already assigned to a fold;
/// 2. minimize the number of sample rows already assigned;
/// 3. minimize the number of composer groups already assigned;
/// 4. choose the lowest fold index.
///
/// Composer groups are processed largest-first by unique-work count, then by
/// sample count, then by a domain-separated digest derived from the frozen
/// seed and composer ID. The greedy assignment is deterministic and preserves
/// all works/performances for one composer in a single held-out fold.
pub fn generate_composer_disjoint_fold_plan(
    registration: &CorpusRegistrationV1,
    track_id: &str,
    candidates: &[ComposerFoldCandidateV1],
    config: &ComposerFoldConfigV1,
) -> Result<ComposerFoldPlanV1, Vec<ComposerFoldIssueV1>> {
    let mut issues = Vec::new();

    if !(MIN_COMPOSER_FOLDS_V1..=MAX_COMPOSER_FOLDS_V1).contains(&config.fold_count) {
        issues.push(ComposerFoldIssueV1::InvalidFoldCount {
            count: config.fold_count,
        });
    }
    if !valid_sha256(&config.seed_sha256) {
        issues.push(ComposerFoldIssueV1::InvalidSeedDigest);
    }
    if config.algorithm_version != COMPOSER_FOLD_ALGORITHM_V1 {
        issues.push(ComposerFoldIssueV1::WrongAlgorithmVersion);
    }
    if candidates.is_empty() {
        issues.push(ComposerFoldIssueV1::EmptySourceInventory);
    }
    if candidates.len() > MAX_COMPOSER_FOLD_SAMPLES_V1 {
        issues.push(ComposerFoldIssueV1::SourceInventoryTooLarge {
            count: candidates.len(),
            maximum: MAX_COMPOSER_FOLD_SAMPLES_V1,
        });
    }

    let registration_issues = validate_corpus_registration(registration);
    if !registration_issues.is_empty() {
        issues.push(ComposerFoldIssueV1::InvalidRegistration(registration_issues));
    }
    let Some(track) = registration
        .tracks
        .iter()
        .find(|track| track.track_id == track_id)
    else {
        issues.push(ComposerFoldIssueV1::UnknownTrack {
            track_id: track_id.to_string(),
        });
        return Err(issues);
    };
    if !track.split_policy.work_disjoint
        || !track.split_policy.composer_disjoint
        || !track.split_policy.exact_artifact_disjoint
    {
        issues.push(ComposerFoldIssueV1::UnsupportedSplitPolicy);
    }
    if !track.training_allowed {
        issues.push(ComposerFoldIssueV1::TrainingNotAllowed);
    }

    let mut sample_ids = BTreeSet::new();
    let mut work_composers: BTreeMap<String, String> = BTreeMap::new();
    let mut groups: BTreeMap<String, ComposerGroup> = BTreeMap::new();

    for candidate in candidates {
        if candidate.sample_id.trim().is_empty() {
            issues.push(ComposerFoldIssueV1::EmptySampleId);
        } else {
            if candidate.sample_id != candidate.sample_id.trim() {
                issues.push(ComposerFoldIssueV1::NonCanonicalSampleId {
                    sample_id: candidate.sample_id.clone(),
                });
            }
            if !sample_ids.insert(candidate.sample_id.clone()) {
                issues.push(ComposerFoldIssueV1::DuplicateSampleId {
                    sample_id: candidate.sample_id.clone(),
                });
            }
        }

        let composer = candidate.composer_group_id.trim();
        if composer.is_empty() {
            issues.push(ComposerFoldIssueV1::MissingComposerGroup {
                sample_id: candidate.sample_id.clone(),
            });
            continue;
        }
        if composer != candidate.composer_group_id {
            issues.push(ComposerFoldIssueV1::NonCanonicalComposerGroup {
                sample_id: candidate.sample_id.clone(),
            });
            continue;
        }

        let work = candidate.work_group_id.trim();
        if work.is_empty() {
            issues.push(ComposerFoldIssueV1::MissingWorkGroup {
                sample_id: candidate.sample_id.clone(),
            });
            continue;
        }
        if work != candidate.work_group_id {
            issues.push(ComposerFoldIssueV1::NonCanonicalWorkGroup {
                sample_id: candidate.sample_id.clone(),
            });
            continue;
        }

        match work_composers.get(work) {
            Some(first) if first != composer => {
                issues.push(ComposerFoldIssueV1::WorkMappedToMultipleComposers {
                    work_group_id: work.to_string(),
                    first_composer: first.clone(),
                    second_composer: composer.to_string(),
                });
            }
            Some(_) => {}
            None => {
                work_composers.insert(work.to_string(), composer.to_string());
            }
        }

        let group = groups
            .entry(composer.to_string())
            .or_insert_with(|| ComposerGroup {
                work_ids: BTreeSet::new(),
                sample_count: 0,
                stable_order_digest: stable_composer_order_digest(
                    &config.seed_sha256,
                    composer,
                ),
            });
        group.work_ids.insert(work.to_string());
        group.sample_count = group.sample_count.saturating_add(1);
    }

    if groups.len() < config.fold_count {
        issues.push(ComposerFoldIssueV1::InsufficientComposerGroups {
            count: groups.len(),
            folds: config.fold_count,
        });
    }
    if !issues.is_empty() {
        return Err(issues);
    }

    let mut ordered_groups: Vec<_> = groups.iter().collect();
    ordered_groups.sort_by(|(left_id, left), (right_id, right)| {
        right
            .work_ids
            .len()
            .cmp(&left.work_ids.len())
            .then_with(|| right.sample_count.cmp(&left.sample_count))
            .then_with(|| left.stable_order_digest.cmp(&right.stable_order_digest))
            .then_with(|| left_id.cmp(right_id))
    });

    let mut fold_work_loads = vec![0usize; config.fold_count];
    let mut fold_sample_loads = vec![0usize; config.fold_count];
    let mut fold_composer_loads = vec![0usize; config.fold_count];
    let mut composer_to_fold = BTreeMap::new();

    for (composer_id, group) in ordered_groups {
        let fold_index = (0..config.fold_count)
            .min_by_key(|index| {
                (
                    fold_work_loads[*index],
                    fold_sample_loads[*index],
                    fold_composer_loads[*index],
                    *index,
                )
            })
            .ok_or_else(|| vec![ComposerFoldIssueV1::InvalidFoldCount {
                count: config.fold_count,
            }])?;
        composer_to_fold.insert(composer_id.clone(), fold_index);
        fold_work_loads[fold_index] =
            fold_work_loads[fold_index].saturating_add(group.work_ids.len());
        fold_sample_loads[fold_index] =
            fold_sample_loads[fold_index].saturating_add(group.sample_count);
        fold_composer_loads[fold_index] =
            fold_composer_loads[fold_index].saturating_add(1);
    }

    let assignments: Vec<_> = groups
        .iter()
        .map(|(composer_group_id, group)| ComposerFoldGroupAssignmentV1 {
            composer_group_id: composer_group_id.clone(),
            fold_index: composer_to_fold[composer_group_id],
            unique_work_count: group.work_ids.len(),
            sample_count: group.sample_count,
        })
        .collect();

    let mut fold_manifests = Vec::with_capacity(config.fold_count);
    let mut fold_manifest_sha256 = Vec::with_capacity(config.fold_count);
    for fold_index in 0..config.fold_count {
        let fold_id = format!("composer-fold-{fold_index:02}");
        let mut samples: Vec<CorpusSampleV1> = candidates
            .iter()
            .map(|candidate| CorpusSampleV1 {
                sample_id: candidate.sample_id.clone(),
                fold_id: fold_id.clone(),
                split: if composer_to_fold[&candidate.composer_group_id] == fold_index {
                    CorpusSplitRoleV1::Test
                } else {
                    CorpusSplitRoleV1::Train
                },
                composer_group_id: Some(candidate.composer_group_id.clone()),
                work_group_id: Some(candidate.work_group_id.clone()),
                input_relative_path: candidate.input_relative_path.clone(),
                reference_relative_path: candidate.reference_relative_path.clone(),
                input_sha256: candidate.input_sha256.clone(),
                reference_sha256: candidate.reference_sha256.clone(),
            })
            .collect();
        samples.sort_by(|left, right| left.sample_id.cmp(&right.sample_id));

        let manifest = CorpusSampleManifestV1 {
            schema: MUSICXML_CORPUS_SAMPLE_MANIFEST_SCHEMA_V1.into(),
            corpus_id: registration.corpus_id.clone(),
            release_id: registration.release_id.clone(),
            track_id: track_id.to_string(),
            samples,
        };
        let manifest_issues = validate_corpus_sample_manifest(registration, track, &manifest);
        if !manifest_issues.is_empty() {
            issues.push(ComposerFoldIssueV1::InvalidFoldManifest {
                fold_index,
                issues: manifest_issues,
            });
            continue;
        }
        let digest = canonical_json_sha256(&manifest)
            .map_err(|_| vec![ComposerFoldIssueV1::SerializationFailed])?;
        fold_manifest_sha256.push(digest);
        fold_manifests.push(manifest);
    }

    if !issues.is_empty() || fold_manifests.len() != config.fold_count {
        return Err(issues);
    }

    let source_inventory_sha256 =
        inventory_commitment(candidates).map_err(|issue| vec![issue])?;
    let registration_sha256 = canonical_json_sha256(registration)
        .map_err(|_| vec![ComposerFoldIssueV1::SerializationFailed])?;
    let source_revision_sha256 = canonical_json_sha256(&(
        registration.source_revision_id.as_deref(),
        &registration.source_uri,
    ))
    .map_err(|_| vec![ComposerFoldIssueV1::SerializationFailed])?;

    let mut plan = ComposerFoldPlanV1 {
        schema: COMPOSER_FOLD_PLAN_SCHEMA_V1.into(),
        config: config.clone(),
        corpus_id: registration.corpus_id.clone(),
        release_id: registration.release_id.clone(),
        track_id: track_id.to_string(),
        source_revision_sha256,
        source_inventory_sha256,
        registration_sha256,
        assignments,
        fold_manifests,
        fold_manifest_sha256,
        plan_sha256: String::new(),
    };
    plan.plan_sha256 = plan_commitment(&plan).map_err(|issue| vec![issue])?;
    Ok(plan)
}

/// Recompute every assignment and manifest. The supplied plan is valid only
/// when all fields and the domain-separated plan commitment match exactly.
pub fn validate_composer_disjoint_fold_plan_replay(
    registration: &CorpusRegistrationV1,
    track_id: &str,
    candidates: &[ComposerFoldCandidateV1],
    config: &ComposerFoldConfigV1,
    supplied: &ComposerFoldPlanV1,
) -> bool {
    generate_composer_disjoint_fold_plan(registration, track_id, candidates, config)
        .is_ok_and(|expected| expected == *supplied)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::musicxml_corpus_qualification::{
        CorpusLicenseV1, CorpusUseScopeV1, RightsEvidenceStatusV1,
        candidate_musicxml_corpus_registry_v1,
    };

    fn registration() -> CorpusRegistrationV1 {
        candidate_musicxml_corpus_registry_v1()
            .into_iter()
            .find(|value| value.corpus_id == "ossq-omr")
            .unwrap()
    }

    fn candidates(composers: usize) -> Vec<ComposerFoldCandidateV1> {
        let mut values = Vec::new();
        for composer_index in 0..composers {
            for work_index in 0..2 {
                for take_index in 0..2 {
                    let id = format!("c{composer_index:02}-w{work_index:02}-t{take_index:02}");
                    values.push(ComposerFoldCandidateV1 {
                        sample_id: id.clone(),
                        composer_group_id: format!("composer-{composer_index:02}"),
                        work_group_id: format!("work-{composer_index:02}-{work_index:02}"),
                        input_relative_path: format!("images/{id}.png"),
                        reference_relative_path: format!("scores/{id}.xml"),
                        input_sha256: sha256_hex(format!("image:{id}").as_bytes()),
                        reference_sha256: sha256_hex(format!("score:{id}").as_bytes()),
                    });
                }
            }
        }
        values
    }

    fn config(folds: usize, seed: char) -> ComposerFoldConfigV1 {
        ComposerFoldConfigV1 {
            fold_count: folds,
            seed_sha256: seed.to_string().repeat(64),
            algorithm_version: COMPOSER_FOLD_ALGORITHM_V1.into(),
        }
    }

    fn make_plan(
        registration: &CorpusRegistrationV1,
        candidates: &[ComposerFoldCandidateV1],
        config: &ComposerFoldConfigV1,
    ) -> ComposerFoldPlanV1 {
        generate_composer_disjoint_fold_plan(
            registration,
            "omr-composer-heldout",
            candidates,
            config,
        )
        .unwrap()
    }

    #[test]
    fn same_seed_replays_byte_identical_plan() {
        let registration = registration();
        let candidates = candidates(15);
        let config = config(5, 'a');
        let first = make_plan(&registration, &candidates, &config);
        let second = make_plan(&registration, &candidates, &config);
        assert_eq!(first, second);
        assert!(validate_composer_disjoint_fold_plan_replay(
            &registration,
            "omr-composer-heldout",
            &candidates,
            &config,
            &first,
        ));
        assert_eq!(first.fold_manifests.len(), 5);
        assert_eq!(first.fold_manifest_sha256.len(), 5);
    }

    #[test]
    fn every_sample_is_test_once_and_train_in_other_folds() {
        let registration = registration();
        let candidates = candidates(15);
        let plan = make_plan(&registration, &candidates, &config(5, 'b'));
        for candidate in &candidates {
            let roles: Vec<_> = plan
                .fold_manifests
                .iter()
                .map(|manifest| {
                    manifest.samples
                        .iter()
                        .find(|sample| sample.sample_id == candidate.sample_id)
                        .unwrap()
                        .split
                })
                .collect();
            assert_eq!(
                roles.iter().filter(|role| **role == CorpusSplitRoleV1::Test).count(),
                1
            );
            assert_eq!(
                roles.iter().filter(|role| **role == CorpusSplitRoleV1::Train).count(),
                4
            );
        }
    }

    #[test]
    fn all_works_and_performances_for_one_composer_stay_in_one_fold() {
        let registration = registration();
        let candidates = candidates(15);
        let plan = make_plan(&registration, &candidates, &config(5, 'c'));
        for assignment in &plan.assignments {
            for manifest in &plan.fold_manifests {
                let composer_rows: Vec<_> = manifest.samples.iter()
                    .filter(|sample| sample.composer_group_id.as_deref()
                        == Some(assignment.composer_group_id.as_str()))
                    .collect();
                let expected = if manifest.fold_id == format!("composer-fold-{:02}", assignment.fold_index) {
                    CorpusSplitRoleV1::Test
                } else {
                    CorpusSplitRoleV1::Train
                };
                assert!(composer_rows.iter().all(|sample| sample.split == expected));
            }
        }
    }

    #[test]
    fn fold_loads_are_balanced_and_deterministic() {
        let registration = registration();
        let candidates = candidates(17);
        let plan = make_plan(&registration, &candidates, &config(5, 'd'));
        let loads: Vec<_> = plan.fold_manifests.iter().map(|manifest| {
            let test_samples = manifest.samples.iter()
                .filter(|sample| sample.split == CorpusSplitRoleV1::Test)
                .count();
            let test_works: BTreeSet<_> = manifest.samples.iter()
                .filter(|sample| sample.split == CorpusSplitRoleV1::Test)
                .filter_map(|sample| sample.work_group_id.as_deref())
                .collect();
            (test_works.len(), test_samples)
        }).collect();
        assert!(loads.iter().map(|load| load.0).max().unwrap()
            - loads.iter().map(|load| load.0).min().unwrap() <= 1);
        assert!(loads.iter().map(|load| load.1).max().unwrap()
            - loads.iter().map(|load| load.1).min().unwrap() <= 4);
    }

    #[test]
    fn different_seed_changes_tie_order_and_plan_commitment() {
        let registration = registration();
        let candidates = candidates(10);
        let first = make_plan(&registration, &candidates, &config(5, 'a'));
        let second = make_plan(&registration, &candidates, &config(5, 'b'));
        assert_ne!(first.plan_sha256, second.plan_sha256);
        assert_ne!(first.assignments, second.assignments);
    }

    #[test]
    fn work_cannot_be_claimed_by_two_composers() {
        let registration = registration();
        let mut candidates = candidates(5);
        candidates[0].work_group_id = candidates[1].work_group_id.clone();
        candidates[0].composer_group_id = "composer-a".into();
        candidates[1].composer_group_id = "composer-b".into();
        assert!(generate_composer_disjoint_fold_plan(
            &registration,
            "omr-composer-heldout",
            &candidates,
            &config(2, 'e'),
        )
        .unwrap_err()
        .contains(&ComposerFoldIssueV1::WorkMappedToMultipleComposers {
            work_group_id: candidates[1].work_group_id.clone(),
            first_composer: "composer-a".into(),
            second_composer: "composer-b".into(),
        }));
    }

    #[test]
    fn unsupported_track_policy_and_training_prohibition_fail_closed() {
        let mut registration = registration();
        let track = registration.tracks.iter_mut()
            .find(|track| track.track_id == "omr-composer-heldout").unwrap();
        track.split_policy.composer_disjoint = false;
        assert!(generate_composer_disjoint_fold_plan(
            &registration,
            "omr-composer-heldout",
            &candidates(5),
            &config(5, 'f'),
        )
        .unwrap_err()
        .contains(&ComposerFoldIssueV1::UnsupportedSplitPolicy));

        let mut registration = registration();
        let track = registration.tracks.iter_mut()
            .find(|track| track.track_id == "omr-composer-heldout").unwrap();
        track.training_allowed = false;
        assert!(generate_composer_disjoint_fold_plan(
            &registration,
            "omr-composer-heldout",
            &candidates(5),
            &config(5, 'f'),
        )
        .unwrap_err()
        .contains(&ComposerFoldIssueV1::TrainingNotAllowed));
    }

    #[test]
    fn invalid_fold_count_seed_and_group_count_are_refused() {
        let registration = registration();
        let candidates = candidates(3);
        assert!(generate_composer_disjoint_fold_plan(
            &registration,
            "omr-composer-heldout",
            &candidates,
            &config(1, 'a'),
        )
        .unwrap_err()
        .contains(&ComposerFoldIssueV1::InvalidFoldCount { count: 1 }));

        let mut invalid_seed = config(2, 'a');
        invalid_seed.seed_sha256 = "not-a-digest".into();
        assert!(generate_composer_disjoint_fold_plan(
            &registration,
            "omr-composer-heldout",
            &candidates,
            &invalid_seed,
        )
        .unwrap_err()
        .contains(&ComposerFoldIssueV1::InvalidSeedDigest));

        assert!(generate_composer_disjoint_fold_plan(
            &registration,
            "omr-composer-heldout",
            &candidates(1),
            &config(2, 'a'),
        )
        .unwrap_err()
        .contains(&ComposerFoldIssueV1::InsufficientComposerGroups {
            count: 1,
            folds: 2,
        }));
    }

    #[test]
    fn invalid_path_digest_and_duplicate_sample_fail_closed() {
        let registration = registration();
        let mut candidates = candidates(5);
        candidates[1].sample_id = candidates[0].sample_id.clone();
        assert!(generate_composer_disjoint_fold_plan(
            &registration,
            "omr-composer-heldout",
            &candidates,
            &config(5, 'a'),
        )
        .unwrap_err()
        .contains(&ComposerFoldIssueV1::DuplicateSampleId {
            sample_id: candidates[0].sample_id.clone(),
        }));

        let mut candidates = candidates(5);
        candidates[0].input_relative_path = "../escape.png".into();
        assert!(generate_composer_disjoint_fold_plan(
            &registration,
            "omr-composer-heldout",
            &candidates,
            &config(5, 'a'),
        )
        .unwrap_err()
        .iter()
        .any(|issue| matches!(
            issue,
            ComposerFoldIssueV1::InvalidFoldManifest { .. }
        )));
    }

    #[test]
    fn plan_replay_rejects_changed_inventory_or_manifest() {
        let registration = registration();
        let mut candidates = candidates(10);
        let config = config(5, 'a');
        let mut plan = make_plan(&registration, &candidates, &config);
        plan.fold_manifests[0].samples[0].split = CorpusSplitRoleV1::Test;
        assert!(!validate_composer_disjoint_fold_plan_replay(
            &registration,
            "omr-composer-heldout",
            &candidates,
            &config,
            &plan,
        ));
        let clean = make_plan(&registration, &candidates, &config);
        candidates[0].input_sha256 = "f".repeat(64);
        assert!(!validate_composer_disjoint_fold_plan_replay(
            &registration,
            "omr-composer-heldout",
            &candidates,
            &config,
            &clean,
        ));
    }

    #[test]
    fn source_sample_order_does_not_change_assignment_or_commitment() {
        let registration = registration();
        let mut candidates = candidates(12);
        let config = config(5, 'b');
        let original = make_plan(&registration, &candidates, &config);
        candidates.reverse();
        let reordered = make_plan(&registration, &candidates, &config);
        assert_eq!(original, reordered);
    }
}
