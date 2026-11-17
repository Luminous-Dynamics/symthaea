//! Leakage-controlled chronological window partitioning.
//!
//! Windows may overlap in historical context, but their target intervals must
//! not cross into the next partition. Splits are computed per entity from
//! distinct forecast origins. Windows sharing an origin stay in one partition.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ForecastWindow {
    pub window_id: u64,
    pub entity_id: u64,
    /// Inclusive beginning of the historical input interval.
    pub history_start_us: i64,
    /// Forecast decision time; target interval begins here.
    pub forecast_origin_us: i64,
    /// Exclusive end of the target interval.
    pub target_end_exclusive_us: i64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EntitySplitBoundaries {
    pub entity_id: u64,
    /// First forecast origin assigned to validation.
    pub validation_start_us: i64,
    /// First forecast origin assigned to test.
    pub test_start_us: i64,
    /// Latest permitted train target end, exclusive.
    pub train_target_cutoff_us: i64,
    /// Latest permitted validation target end, exclusive.
    pub validation_target_cutoff_us: i64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PurgedSplitPlan {
    pub train_window_ids: Vec<u64>,
    pub validation_window_ids: Vec<u64>,
    pub test_window_ids: Vec<u64>,
    pub purged_train_window_ids: Vec<u64>,
    pub purged_validation_window_ids: Vec<u64>,
    pub boundaries: Vec<EntitySplitBoundaries>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SplitError {
    ZeroWindowId,
    ZeroEntityId,
    DuplicateWindowId { window_id: u64 },
    InvalidWindowInterval { window_id: u64 },
    InvalidSplitRatios { train_basis_points: u16, validation_basis_points: u16 },
    InsufficientDistinctOrigins { entity_id: u64, origins: usize },
    EmbargoUnderflow { entity_id: u64, boundary_us: i64, embargo_us: u64 },
    EmptyPartition { entity_id: u64, partition: &'static str },
    EmptyInput,
}

impl fmt::Display for SplitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroWindowId => write!(f, "forecast window ID must be nonzero"),
            Self::ZeroEntityId => write!(f, "forecast window entity ID must be nonzero"),
            Self::DuplicateWindowId { window_id } => write!(f, "duplicate forecast window ID {window_id}"),
            Self::InvalidWindowInterval { window_id } => write!(f, "forecast window {window_id} has an invalid history/origin/target interval"),
            Self::InvalidSplitRatios { train_basis_points, validation_basis_points } => write!(f, "invalid chronological split ratios: train={train_basis_points}bp, validation={validation_basis_points}bp"),
            Self::InsufficientDistinctOrigins { entity_id, origins } => write!(f, "entity {entity_id} has only {origins} distinct forecast origins; at least three are required"),
            Self::EmbargoUnderflow { entity_id, boundary_us, embargo_us } => write!(f, "embargo {embargo_us} underflows boundary {boundary_us} for entity {entity_id}"),
            Self::EmptyPartition { entity_id, partition } => write!(f, "entity {entity_id} has no eligible {partition} windows after purging"),
            Self::EmptyInput => write!(f, "at least one forecast window is required"),
        }
    }
}

impl std::error::Error for SplitError {}

/// Construct per-entity chronological partitions with a target-horizon purge.
///
/// Ratios are in basis points (10_000 = 100%). train + validation must be
/// strictly less than 10_000; test receives the remainder. Boundary selection
/// is based on distinct forecast origins, so tied origins cannot be split
/// between partitions.
///
/// A train window is retained only when
/// target_end_exclusive_us <= validation_start_us - embargo_us.
/// A validation window is retained only when
/// target_end_exclusive_us <= test_start_us - embargo_us.
/// This prevents target labels in one split from appearing at or after the
/// next split's forecast origin. Historical input intervals may overlap.
pub fn purged_chronological_split(
    windows: &[ForecastWindow],
    train_basis_points: u16,
    validation_basis_points: u16,
    embargo_us: u64,
) -> Result<PurgedSplitPlan, SplitError> {
    if windows.is_empty() {
        return Err(SplitError::EmptyInput);
    }
    if train_basis_points == 0
        || validation_basis_points == 0
        || u32::from(train_basis_points) + u32::from(validation_basis_points) >= 10_000
    {
        return Err(SplitError::InvalidSplitRatios {
            train_basis_points,
            validation_basis_points,
        });
    }

    let mut seen_ids = BTreeSet::new();
    let mut by_entity = BTreeMap::<u64, Vec<ForecastWindow>>::new();
    for window in windows {
        if window.window_id == 0 {
            return Err(SplitError::ZeroWindowId);
        }
        if window.entity_id == 0 {
            return Err(SplitError::ZeroEntityId);
        }
        if !seen_ids.insert(window.window_id) {
            return Err(SplitError::DuplicateWindowId { window_id: window.window_id });
        }
        if window.history_start_us >= window.forecast_origin_us
            || window.forecast_origin_us >= window.target_end_exclusive_us
        {
            return Err(SplitError::InvalidWindowInterval { window_id: window.window_id });
        }
        by_entity.entry(window.entity_id).or_default().push(*window);
    }

    let mut plan = PurgedSplitPlan::default();
    for (entity_id, mut entity_windows) in by_entity {
        entity_windows.sort_unstable_by_key(|window| {
            (window.forecast_origin_us, window.window_id)
        });
        let origins: Vec<i64> = entity_windows.iter()
            .map(|window| window.forecast_origin_us)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        if origins.len() < 3 {
            return Err(SplitError::InsufficientDistinctOrigins {
                entity_id,
                origins: origins.len(),
            });
        }

        let n = origins.len() as u128;
        let train_cut_index = usize::try_from(
            n * u128::from(train_basis_points) / 10_000,
        ).expect("origin count fits usize");
        let validation_cut_index = usize::try_from(
            n * u128::from(u32::from(train_basis_points) + u32::from(validation_basis_points))
                / 10_000,
        ).expect("origin count fits usize");
        if train_cut_index == 0
            || validation_cut_index <= train_cut_index
            || validation_cut_index >= origins.len()
        {
            return Err(SplitError::InsufficientDistinctOrigins {
                entity_id,
                origins: origins.len(),
            });
        }

        let validation_start_us = origins[train_cut_index];
        let test_start_us = origins[validation_cut_index];
        let embargo = i64::try_from(embargo_us).map_err(|_| SplitError::EmbargoUnderflow {
            entity_id,
            boundary_us: validation_start_us,
            embargo_us,
        })?;
        let train_target_cutoff_us = validation_start_us.checked_sub(embargo)
            .ok_or(SplitError::EmbargoUnderflow {
                entity_id,
                boundary_us: validation_start_us,
                embargo_us,
            })?;
        let validation_target_cutoff_us = test_start_us.checked_sub(embargo)
            .ok_or(SplitError::EmbargoUnderflow {
                entity_id,
                boundary_us: test_start_us,
                embargo_us,
            })?;

        let before = (
            plan.train_window_ids.len(),
            plan.validation_window_ids.len(),
            plan.test_window_ids.len(),
            plan.purged_train_window_ids.len(),
            plan.purged_validation_window_ids.len(),
        );
        for window in &entity_windows {
            if window.forecast_origin_us < validation_start_us {
                if window.target_end_exclusive_us <= train_target_cutoff_us {
                    plan.train_window_ids.push(window.window_id);
                } else {
                    plan.purged_train_window_ids.push(window.window_id);
                }
            } else if window.forecast_origin_us < test_start_us {
                if window.target_end_exclusive_us <= validation_target_cutoff_us {
                    plan.validation_window_ids.push(window.window_id);
                } else {
                    plan.purged_validation_window_ids.push(window.window_id);
                }
            } else {
                plan.test_window_ids.push(window.window_id);
            }
        }
        let after = (
            plan.train_window_ids.len(),
            plan.validation_window_ids.len(),
            plan.test_window_ids.len(),
            plan.purged_train_window_ids.len(),
            plan.purged_validation_window_ids.len(),
        );
        if after.0 == before.0 {
            return Err(SplitError::EmptyPartition { entity_id, partition: "train" });
        }
        if after.1 == before.1 {
            return Err(SplitError::EmptyPartition { entity_id, partition: "validation" });
        }
        if after.2 == before.2 {
            return Err(SplitError::EmptyPartition { entity_id, partition: "test" });
        }

        plan.boundaries.push(EntitySplitBoundaries {
            entity_id,
            validation_start_us,
            test_start_us,
            train_target_cutoff_us,
            validation_target_cutoff_us,
        });
    }

    plan.train_window_ids.sort_unstable();
    plan.validation_window_ids.sort_unstable();
    plan.test_window_ids.sort_unstable();
    plan.purged_train_window_ids.sort_unstable();
    plan.purged_validation_window_ids.sort_unstable();
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(id: u64, entity: u64, origin: i64, horizon: i64) -> ForecastWindow {
        ForecastWindow {
            window_id: id,
            entity_id: entity,
            history_start_us: origin - 10,
            forecast_origin_us: origin,
            target_end_exclusive_us: origin + horizon,
        }
    }

    fn sample_windows(horizon: i64) -> Vec<ForecastWindow> {
        (1..=10).map(|id| window(id, 1, i64::try_from(id).unwrap() * 10, horizon)).collect()
    }

    #[test]
    fn overlapping_prediction_horizons_are_purged_at_split_boundaries() {
        let plan = purged_chronological_split(&sample_windows(15), 6_000, 2_000, 0).unwrap();
        assert_eq!(plan.train_window_ids, vec![1, 2, 3, 4, 5]);
        assert_eq!(plan.purged_train_window_ids, vec![6]);
        assert_eq!(plan.validation_window_ids, vec![7]);
        assert_eq!(plan.purged_validation_window_ids, vec![8]);
        assert_eq!(plan.test_window_ids, vec![9, 10]);
    }

    #[test]
    fn target_ending_exactly_at_next_origin_is_allowed() {
        let plan = purged_chronological_split(&sample_windows(10), 6_000, 2_000, 0).unwrap();
        assert!(plan.train_window_ids.contains(&6));
        assert!(plan.validation_window_ids.contains(&8));
    }

    #[test]
    fn embargo_shrinks_the_eligible_target_interval() {
        let plan = purged_chronological_split(&sample_windows(10), 6_000, 2_000, 5).unwrap();
        assert_eq!(plan.boundaries[0].train_target_cutoff_us, 65);
        assert_eq!(plan.boundaries[0].validation_target_cutoff_us, 85);
        assert!(!plan.train_window_ids.contains(&6));
        assert!(!plan.validation_window_ids.contains(&8));
    }

    #[test]
    fn tied_forecast_origins_are_never_split_across_partitions() {
        let mut windows = vec![
            window(1, 1, 10, 5), window(2, 1, 20, 5),
            window(3, 1, 30, 5), window(4, 1, 40, 5),
            window(5, 1, 50, 5), window(6, 1, 50, 5),
            window(7, 1, 60, 5), window(8, 1, 70, 5),
            window(9, 1, 80, 5),
        ];
        windows.reverse();
        let plan = purged_chronological_split(&windows, 6_000, 2_000, 0).unwrap();
        let fifty_ids = [5, 6];
        let in_validation = fifty_ids.iter().all(|id| plan.validation_window_ids.contains(id));
        let in_train = fifty_ids.iter().all(|id| plan.train_window_ids.contains(id));
        assert!(in_validation || in_train);
    }

    #[test]
    fn output_is_independent_of_input_permutation() {
        let windows = sample_windows(15);
        let first = purged_chronological_split(&windows, 6_000, 2_000, 0).unwrap();
        let mut reversed = windows.clone();
        reversed.reverse();
        let second = purged_chronological_split(&reversed, 6_000, 2_000, 0).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn entities_receive_independent_chronological_boundaries() {
        let mut windows = sample_windows(5);
        windows.extend((1..=8).map(|id| window(100 + id, 2, i64::try_from(id).unwrap() * 100, 5)));
        let plan = purged_chronological_split(&windows, 6_000, 2_000, 0).unwrap();
        assert_eq!(plan.boundaries.len(), 2);
        assert_eq!(plan.boundaries[0].entity_id, 1);
        assert_eq!(plan.boundaries[1].entity_id, 2);
    }

    #[test]
    fn invalid_windows_ratios_and_duplicate_ids_fail_closed() {
        let invalid = ForecastWindow {
            window_id: 1, entity_id: 1, history_start_us: 10,
            forecast_origin_us: 10, target_end_exclusive_us: 20,
        };
        assert_eq!(
            purged_chronological_split(&[invalid], 6_000, 2_000, 0),
            Err(SplitError::InvalidWindowInterval { window_id: 1 })
        );
        assert!(matches!(
            purged_chronological_split(&sample_windows(5), 8_000, 3_000, 0),
            Err(SplitError::InvalidSplitRatios { .. })
        ));
        let mut duplicate = sample_windows(5);
        duplicate[1].window_id = duplicate[0].window_id;
        assert_eq!(
            purged_chronological_split(&duplicate, 6_000, 2_000, 0),
            Err(SplitError::DuplicateWindowId { window_id: 1 })
        );
    }

    #[test]
    fn insufficient_origins_and_empty_after_purge_are_reported() {
        assert_eq!(
            purged_chronological_split(&[], 6_000, 2_000, 0),
            Err(SplitError::EmptyInput)
        );
        let few = vec![window(1, 1, 10, 2), window(2, 1, 20, 2)];
        assert_eq!(
            purged_chronological_split(&few, 6_000, 2_000, 0),
            Err(SplitError::InsufficientDistinctOrigins { entity_id: 1, origins: 2 })
        );
        let long_horizon = sample_windows(100);
        assert!(matches!(
            purged_chronological_split(&long_horizon, 6_000, 2_000, 0),
            Err(SplitError::EmptyPartition { .. })
        ));
    }
}
