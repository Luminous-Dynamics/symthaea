//! Event-grouped validation splits to prevent image/circle leakage.
//!
//! Each row is a sample (for example, an image, geometric trace, or feature
//! set). Every sample from one physical event and its derivatives must share
//! the same leakage_group_id. Folds are made over groups, never over rows.

use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkSample {
    pub sample_id: String,
    /// Canonical event identifier for reporting and audit.
    pub event_id: String,
    /// Canonical split group tying every image/derivative to one physical event.
    pub leakage_group_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventFold {
    pub fold_index: usize,
    pub train_sample_indices: Vec<usize>,
    pub validation_sample_indices: Vec<usize>,
    pub train_group_count: usize,
    pub validation_group_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventFoldPlan {
    pub seed: u64,
    pub sample_count: usize,
    pub leakage_group_count: usize,
    pub folds: Vec<EventFold>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventSplitError {
    EmptySamples,
    EmptySampleId { index: usize },
    EmptyEventId { index: usize },
    EmptyLeakageGroupId { index: usize },
    DuplicateSampleId(String),
    TooFewGroups { requested_folds: usize, available_groups: usize },
    InvalidFoldCount,
    EmptyFold { fold_index: usize },
    DuplicateSampleIndex { fold_index: usize, sample_index: usize },
    MissingValidationSample { sample_index: usize },
    ValidationSampleInMultipleFolds { sample_index: usize },
    TrainValidationOverlap { fold_index: usize, sample_index: usize },
    InvalidTrainComplement { fold_index: usize, sample_index: usize },
    LeakageAcrossFolds { leakage_group_id: String },
}

impl std::fmt::Display for EventSplitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptySamples => write!(f, "benchmark requires at least one sample"),
            Self::EmptySampleId { index } => write!(f, "sample {index} has an empty sample_id"),
            Self::EmptyEventId { index } => write!(f, "sample {index} has an empty event_id"),
            Self::EmptyLeakageGroupId { index } => write!(f, "sample {index} has an empty leakage_group_id"),
            Self::DuplicateSampleId(id) => write!(f, "duplicate sample_id: {id}"),
            Self::TooFewGroups { requested_folds, available_groups } => write!(
                f,
                "cannot make {requested_folds} folds from {available_groups} leakage groups"
            ),
            Self::InvalidFoldCount => write!(f, "fold count must be at least two"),
            Self::EmptyFold { fold_index } => write!(f, "fold {fold_index} has an empty train or validation partition"),
            Self::DuplicateSampleIndex { fold_index, sample_index } => write!(
                f, "fold {fold_index} repeats sample index {sample_index}"
            ),
            Self::MissingValidationSample { sample_index } => write!(
                f, "sample index {sample_index} never appears in validation"
            ),
            Self::ValidationSampleInMultipleFolds { sample_index } => write!(
                f, "sample index {sample_index} appears in multiple validation folds"
            ),
            Self::TrainValidationOverlap { fold_index, sample_index } => write!(
                f, "fold {fold_index} leaks sample index {sample_index} into train and validation"
            ),
            Self::InvalidTrainComplement { fold_index, sample_index } => write!(
                f, "fold {fold_index} train set is not the validation complement at sample {sample_index}"
            ),
            Self::LeakageAcrossFolds { leakage_group_id } => write!(
                f, "leakage group {leakage_group_id} crosses validation folds"
            ),
        }
    }
}

impl std::error::Error for EventSplitError {}

/// Deterministic FNV-1a tie-break hash; this is not a cryptographic digest.
fn tie_break_hash(seed: u64, value: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64 ^ seed;
    for byte in value.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Build approximately sample-balanced K-folds while keeping whole event
/// groups together. Large groups are greedily assigned first; equal-sized
/// groups use a deterministic seeded tie-break.
///
/// This is for evaluation partitioning, not randomization for inference.
/// Do not use its fold balance as evidence of population representativeness.
pub fn grouped_event_kfold(
    samples: &[BenchmarkSample],
    fold_count: usize,
    seed: u64,
) -> Result<EventFoldPlan, EventSplitError> {
    validate_samples(samples)?;
    if fold_count < 2 {
        return Err(EventSplitError::InvalidFoldCount);
    }

    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, sample) in samples.iter().enumerate() {
        groups
            .entry(sample.leakage_group_id.clone())
            .or_default()
            .push(index);
    }
    if fold_count > groups.len() {
        return Err(EventSplitError::TooFewGroups {
            requested_folds: fold_count,
            available_groups: groups.len(),
        });
    }

    let mut ordered_groups: Vec<(String, Vec<usize>)> = groups.into_iter().collect();
    ordered_groups.sort_by(|(first_id, first_rows), (second_id, second_rows)| {
        second_rows
            .len()
            .cmp(&first_rows.len())
            .then_with(|| {
                tie_break_hash(seed, first_id).cmp(&tie_break_hash(seed, second_id))
            })
            .then_with(|| first_id.cmp(second_id))
    });

    let mut fold_groups: Vec<Vec<(String, Vec<usize>)>> =
        (0..fold_count).map(|_| Vec::new()).collect();
    let mut fold_loads = vec![0_usize; fold_count];

    for group in ordered_groups {
        let target_fold = (0..fold_count)
            .min_by_key(|fold_index| {
                (
                    fold_loads[*fold_index],
                    fold_groups[*fold_index].len(),
                    *fold_index,
                )
            })
            .expect("fold_count >= 2");
        fold_loads[target_fold] += group.1.len();
        fold_groups[target_fold].push(group);
    }

    let mut folds = Vec::with_capacity(fold_count);
    for (fold_index, groups_for_fold) in fold_groups.iter().enumerate() {
        let mut validation_sample_indices: Vec<usize> = groups_for_fold
            .iter()
            .flat_map(|(_, indices)| indices.iter().copied())
            .collect();
        validation_sample_indices.sort_unstable();

        let validation_set: HashSet<usize> =
            validation_sample_indices.iter().copied().collect();
        let train_sample_indices: Vec<usize> = (0..samples.len())
            .filter(|index| !validation_set.contains(index))
            .collect();

        let validation_group_count = groups_for_fold.len();
        let all_group_count = samples
            .iter()
            .map(|sample| sample.leakage_group_id.as_str())
            .collect::<HashSet<_>>()
            .len();
        let train_group_count = all_group_count - validation_group_count;

        folds.push(EventFold {
            fold_index,
            train_sample_indices,
            validation_sample_indices,
            train_group_count,
            validation_group_count,
        });
    }

    let plan = EventFoldPlan {
        seed,
        sample_count: samples.len(),
        leakage_group_count: samples
            .iter()
            .map(|sample| sample.leakage_group_id.as_str())
            .collect::<HashSet<_>>()
            .len(),
        folds,
    };
    validate_event_fold_plan(samples, &plan)?;
    Ok(plan)
}

fn validate_samples(samples: &[BenchmarkSample]) -> Result<(), EventSplitError> {
    if samples.is_empty() {
        return Err(EventSplitError::EmptySamples);
    }
    let mut identifiers = HashSet::with_capacity(samples.len());
    for (index, sample) in samples.iter().enumerate() {
        if sample.sample_id.trim().is_empty() {
            return Err(EventSplitError::EmptySampleId { index });
        }
        if sample.event_id.trim().is_empty() {
            return Err(EventSplitError::EmptyEventId { index });
        }
        if sample.leakage_group_id.trim().is_empty() {
            return Err(EventSplitError::EmptyLeakageGroupId { index });
        }
        if !identifiers.insert(sample.sample_id.as_str()) {
            return Err(EventSplitError::DuplicateSampleId(
                sample.sample_id.clone(),
            ));
        }
    }
    Ok(())
}

/// Audit a plan for complete validation coverage, train complement correctness
/// and the invariant that one event leakage group occurs in one validation fold.
pub fn validate_event_fold_plan(
    samples: &[BenchmarkSample],
    plan: &EventFoldPlan,
) -> Result<(), EventSplitError> {
    validate_samples(samples)?;
    if plan.folds.len() < 2 || plan.sample_count != samples.len() {
        return Err(EventSplitError::InvalidFoldCount);
    }

    let expected_group_count: HashSet<&str> = samples
        .iter()
        .map(|sample| sample.leakage_group_id.as_str())
        .collect();
    if plan.leakage_group_count != expected_group_count.len() {
        return Err(EventSplitError::InvalidFoldCount);
    }

    let mut validation_counts = vec![0_usize; samples.len()];
    let mut group_fold: HashMap<&str, usize> = HashMap::new();

    for (expected_fold_index, fold) in plan.folds.iter().enumerate() {
        let fold_index = fold.fold_index;
        if fold_index != expected_fold_index {
            return Err(EventSplitError::InvalidFoldCount);
        }
        if fold.train_sample_indices.is_empty() || fold.validation_sample_indices.is_empty() {
            return Err(EventSplitError::EmptyFold { fold_index });
        }

        let mut train_seen = HashSet::new();
        for index in &fold.train_sample_indices {
            if *index >= samples.len() || !train_seen.insert(*index) {
                return Err(EventSplitError::DuplicateSampleIndex {
                    fold_index,
                    sample_index: *index,
                });
            }
        }

        let mut validation_seen = HashSet::new();
        for index in &fold.validation_sample_indices {
            if *index >= samples.len() || !validation_seen.insert(*index) {
                return Err(EventSplitError::DuplicateSampleIndex {
                    fold_index,
                    sample_index: *index,
                });
            }
            validation_counts[*index] += 1;
            if validation_counts[*index] > 1 {
                return Err(EventSplitError::ValidationSampleInMultipleFolds {
                    sample_index: *index,
                });
            }

            let group = samples[*index].leakage_group_id.as_str();
            if group_fold.insert(group, fold_index).is_some_and(|previous| previous != fold_index) {
                return Err(EventSplitError::LeakageAcrossFolds {
                    leakage_group_id: group.to_owned(),
                });
            }
        }

        for sample_index in 0..samples.len() {
            let in_train = train_seen.contains(&sample_index);
            let in_validation = validation_seen.contains(&sample_index);
            if in_train && in_validation {
                return Err(EventSplitError::TrainValidationOverlap {
                    fold_index,
                    sample_index,
                });
            }
            if in_train == in_validation {
                return Err(EventSplitError::InvalidTrainComplement {
                    fold_index,
                    sample_index,
                });
            }
        }
    }

    for (sample_index, count) in validation_counts.iter().copied().enumerate() {
        if count == 0 {
            return Err(EventSplitError::MissingValidationSample { sample_index });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Vec<BenchmarkSample> {
        vec![
            BenchmarkSample { sample_id: "a-1".into(), event_id: "event-a".into(), leakage_group_id: "physical-a".into() },
            BenchmarkSample { sample_id: "a-2".into(), event_id: "event-a".into(), leakage_group_id: "physical-a".into() },
            BenchmarkSample { sample_id: "b-1".into(), event_id: "event-b".into(), leakage_group_id: "physical-b".into() },
            BenchmarkSample { sample_id: "c-1".into(), event_id: "event-c".into(), leakage_group_id: "physical-c".into() },
            BenchmarkSample { sample_id: "c-2".into(), event_id: "event-c".into(), leakage_group_id: "physical-c".into() },
            BenchmarkSample { sample_id: "c-3".into(), event_id: "event-c".into(), leakage_group_id: "physical-c".into() },
            BenchmarkSample { sample_id: "d-1".into(), event_id: "event-d".into(), leakage_group_id: "physical-d".into() },
        ]
    }

    #[test]
    fn grouped_folds_keep_all_event_derivatives_together() {
        let samples = fixture();
        let plan = grouped_event_kfold(&samples, 3, 42).expect("valid grouped folds");
        validate_event_fold_plan(&samples, &plan).expect("no event leakage");

        for sample in &samples {
            let validation_fold = plan
                .folds
                .iter()
                .find(|fold| {
                    fold.validation_sample_indices
                        .iter()
                        .any(|index| samples[*index].sample_id == sample.sample_id)
                })
                .expect("sample is held out once");
            assert!(validation_fold.validation_sample_indices.iter().all(|index| {
                samples[*index].leakage_group_id != sample.leakage_group_id
                    || samples[*index].event_id == sample.event_id
            }));
        }
    }

    #[test]
    fn grouped_folds_are_deterministic_for_a_frozen_seed() {
        let samples = fixture();
        assert_eq!(
            grouped_event_kfold(&samples, 3, 20261118),
            grouped_event_kfold(&samples, 3, 20261118)
        );
    }

    #[test]
    fn invalid_duplicate_and_missing_group_metadata_fail_closed() {
        assert_eq!(
            grouped_event_kfold(&[], 2, 1),
            Err(EventSplitError::EmptySamples)
        );
        let duplicate = vec![
            BenchmarkSample { sample_id: "same".into(), event_id: "a".into(), leakage_group_id: "a".into() },
            BenchmarkSample { sample_id: "same".into(), event_id: "b".into(), leakage_group_id: "b".into() },
        ];
        assert_eq!(
            grouped_event_kfold(&duplicate, 2, 1),
            Err(EventSplitError::DuplicateSampleId("same".into()))
        );
        let missing_group = vec![
            BenchmarkSample { sample_id: "x".into(), event_id: "x".into(), leakage_group_id: " ".into() },
            BenchmarkSample { sample_id: "y".into(), event_id: "y".into(), leakage_group_id: "y".into() },
        ];
        assert_eq!(
            grouped_event_kfold(&missing_group, 2, 1),
            Err(EventSplitError::EmptyLeakageGroupId { index: 0 })
        );
    }

    #[test]
    fn fold_count_must_be_between_two_and_group_count() {
        let samples = fixture();
        assert_eq!(
            grouped_event_kfold(&samples, 1, 42),
            Err(EventSplitError::InvalidFoldCount)
        );
        assert_eq!(
            grouped_event_kfold(&samples, 5, 42),
            Err(EventSplitError::TooFewGroups {
                requested_folds: 5,
                available_groups: 4
            })
        );
    }

    #[test]
    fn manual_leakage_across_folds_is_detected() {
        let samples = fixture();
        let mut plan = grouped_event_kfold(&samples, 3, 42).expect("valid grouped folds");
        let first_fold = 0;
        let second_fold = 1;
        let leaking_group_sample = plan.folds[first_fold].validation_sample_indices[0];
        let leaking_group = samples[leaking_group_sample].leakage_group_id.clone();
        let moved_sample = samples
            .iter()
            .enumerate()
            .find(|(index, sample)| {
                *index != leaking_group_sample && sample.leakage_group_id == leaking_group
            })
            .map(|(index, _)| index);
        if let Some(moved_sample) = moved_sample {
            plan.folds[second_fold].validation_sample_indices.push(moved_sample);
            plan.folds[second_fold].train_sample_indices.retain(|index| *index != moved_sample);
        } else {
            let other = plan.folds[second_fold].validation_sample_indices[0];
            plan.folds[second_fold].validation_sample_indices.push(leaking_group_sample);
            plan.folds[second_fold].train_sample_indices.retain(|index| *index != leaking_group_sample);
            plan.folds[second_fold].train_sample_indices.push(other);
        }
        assert!(matches!(
            validate_event_fold_plan(&samples, &plan),
            Err(EventSplitError::ValidationSampleInMultipleFolds { .. })
                | Err(EventSplitError::LeakageAcrossFolds { .. })
        ));
    }

    #[test]
    fn fold_partitions_are_exact_train_validation_complements() {
        let samples = fixture();
        let plan = grouped_event_kfold(&samples, 3, 42).expect("valid grouped folds");
        for fold in &plan.folds {
            assert_eq!(
                fold.train_sample_indices.len() + fold.validation_sample_indices.len(),
                samples.len()
            );
        }
    }
}
