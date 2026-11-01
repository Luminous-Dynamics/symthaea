//! Train-only standardization tied to an explicit purged split plan.
//!
//! Fitting is restricted to samples whose window IDs occur in the split plan's
//! train partition. Values from validation, test and purged windows are never
//! read into the moments. Repeated raw observations across overlapping train
//! windows are deduplicated by (entity, feature, observation ID).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use super::splits::{ForecastWindow, PurgedSplitPlan};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindowFeatureValue {
    pub window_id: u64,
    pub entity_id: u64,
    pub feature_id: u32,
    /// Stable raw observation identity across overlapping windows.
    pub observation_id: u64,
    pub value: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeatureScale {
    pub count: u64,
    pub mean: f64,
    /// Sample standard deviation (ddof=1); zero for one or constant sample.
    pub sample_std: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrainOnlyStandardizer {
    scales: BTreeMap<(u64, u32), FeatureScale>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ScalerError {
    DuplicateWindowId { window_id: u64 },
    UnknownWindowId { window_id: u64 },
    WindowEntityMismatch { window_id: u64 },
    ZeroEntityId,
    ZeroFeatureId,
    ZeroObservationId,
    NonFiniteTrainingValue { window_id: u64 },
    ConflictingDuplicateObservation { entity_id: u64, feature_id: u32, observation_id: u64 },
    EmptyTrainingData,
    InvalidSplitPlan,
    UnknownFeature { entity_id: u64, feature_id: u32 },
    NonFiniteTransformValue,
    ArithmeticOverflow,
}

impl fmt::Display for ScalerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateWindowId { window_id } => write!(f, "duplicate window ID {window_id}"),
            Self::UnknownWindowId { window_id } => write!(f, "feature sample references unknown window {window_id}"),
            Self::WindowEntityMismatch { window_id } => write!(f, "feature sample entity disagrees with window {window_id}"),
            Self::ZeroEntityId => write!(f, "feature sample entity ID must be nonzero"),
            Self::ZeroFeatureId => write!(f, "feature ID must be nonzero"),
            Self::ZeroObservationId => write!(f, "raw observation ID must be nonzero"),
            Self::NonFiniteTrainingValue { window_id } => write!(f, "training value in window {window_id} is not finite"),
            Self::ConflictingDuplicateObservation { entity_id, feature_id, observation_id } => write!(f, "duplicate raw observation {observation_id} has conflicting values for entity {entity_id}, feature {feature_id}"),
            Self::EmptyTrainingData => write!(f, "no unique numeric observations were found in train windows"),
            Self::InvalidSplitPlan => write!(f, "split plan contains duplicate or unknown window IDs"),
            Self::UnknownFeature { entity_id, feature_id } => write!(f, "no train-only scale for entity {entity_id}, feature {feature_id}"),
            Self::NonFiniteTransformValue => write!(f, "transform value must be finite"),
            Self::ArithmeticOverflow => write!(f, "feature moment computation overflowed"),
        }
    }
}
impl std::error::Error for ScalerError {}

#[derive(Clone, Copy, Debug, Default)]
struct RunningMoments {
    count: u64,
    mean: f64,
    m2: f64,
}

impl RunningMoments {
    fn push(&mut self, value: f64) -> Result<(), ScalerError> {
        let next_count = self.count.checked_add(1).ok_or(ScalerError::ArithmeticOverflow)?;
        let delta = value - self.mean;
        let next_mean = self.mean + delta / next_count as f64;
        let delta_after = value - next_mean;
        let next_m2 = self.m2 + delta * delta_after;
        if !next_mean.is_finite() || !next_m2.is_finite() {
            return Err(ScalerError::ArithmeticOverflow);
        }
        self.count = next_count;
        self.mean = next_mean;
        self.m2 = next_m2;
        Ok(())
    }

    fn finish(self) -> Result<FeatureScale, ScalerError> {
        if self.count == 0 {
            return Err(ScalerError::EmptyTrainingData);
        }
        let sample_std = if self.count <= 1 {
            0.0
        } else {
            (self.m2 / (self.count - 1) as f64).sqrt()
        };
        if !sample_std.is_finite() {
            return Err(ScalerError::ArithmeticOverflow);
        }
        Ok(FeatureScale {
            count: self.count,
            mean: self.mean,
            sample_std,
        })
    }
}

impl TrainOnlyStandardizer {
    /// Fit per-entity/per-feature statistics using only the plan's training
    /// windows. Samples from validation, test or purged windows are skipped
    /// before their numeric values are inspected.
    pub fn fit(
        windows: &[ForecastWindow],
        plan: &PurgedSplitPlan,
        samples: &[WindowFeatureValue],
    ) -> Result<Self, ScalerError> {
        let mut window_by_id = BTreeMap::<u64, u64>::new();
        for window in windows {
            if window_by_id.insert(window.window_id, window.entity_id).is_some() {
                return Err(ScalerError::DuplicateWindowId { window_id: window.window_id });
            }
        }

        let mut partition_ids = BTreeSet::new();
        for id in plan.train_window_ids.iter()
            .chain(&plan.validation_window_ids)
            .chain(&plan.test_window_ids)
            .chain(&plan.purged_train_window_ids)
            .chain(&plan.purged_validation_window_ids)
        {
            if !partition_ids.insert(*id) || !window_by_id.contains_key(id) {
                return Err(ScalerError::InvalidSplitPlan);
            }
        }
        if partition_ids.len() != windows.len() {
            return Err(ScalerError::InvalidSplitPlan);
        }

        let train_ids: BTreeSet<u64> = plan.train_window_ids.iter().copied().collect();
        let mut unique_values = BTreeMap::<(u64, u32, u64), f64>::new();
        for sample in samples {
            let expected_entity = window_by_id.get(&sample.window_id)
                .ok_or(ScalerError::UnknownWindowId { window_id: sample.window_id })?;
            if *expected_entity != sample.entity_id {
                return Err(ScalerError::WindowEntityMismatch { window_id: sample.window_id });
            }
            if !train_ids.contains(&sample.window_id) {
                // Deliberately do not inspect val/test/purged numeric values.
                continue;
            }
            if sample.entity_id == 0 {
                return Err(ScalerError::ZeroEntityId);
            }
            if sample.feature_id == 0 {
                return Err(ScalerError::ZeroFeatureId);
            }
            if sample.observation_id == 0 {
                return Err(ScalerError::ZeroObservationId);
            }
            if !sample.value.is_finite() {
                return Err(ScalerError::NonFiniteTrainingValue { window_id: sample.window_id });
            }

            let key = (sample.entity_id, sample.feature_id, sample.observation_id);
            match unique_values.get(&key) {
                None => { unique_values.insert(key, sample.value); }
                Some(existing) if existing.to_bits() == sample.value.to_bits() => {}
                Some(_) => {
                    return Err(ScalerError::ConflictingDuplicateObservation {
                        entity_id: sample.entity_id,
                        feature_id: sample.feature_id,
                        observation_id: sample.observation_id,
                    });
                }
            }
        }
        if unique_values.is_empty() {
            return Err(ScalerError::EmptyTrainingData);
        }

        // BTreeMap iteration gives a deterministic entity/feature/observation
        // order independent of input vector order.
        let mut moments = BTreeMap::<(u64, u32), RunningMoments>::new();
        for ((entity_id, feature_id, _observation_id), value) in unique_values {
            moments.entry((entity_id, feature_id)).or_default().push(value)?;
        }
        let scales = moments.into_iter()
            .map(|(key, values)| values.finish().map(|scale| (key, scale)))
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        Ok(Self { scales })
    }

    pub fn scale(&self, entity_id: u64, feature_id: u32) -> Option<FeatureScale> {
        self.scales.get(&(entity_id, feature_id)).copied()
    }

    /// Match pandas-style sample standard deviation. A zero/one-sample scale
    /// subtracts the mean without division, yielding zero for constant data.
    pub fn transform(
        &self,
        entity_id: u64,
        feature_id: u32,
        value: f64,
    ) -> Result<f64, ScalerError> {
        if !value.is_finite() {
            return Err(ScalerError::NonFiniteTransformValue);
        }
        let scale = self.scales.get(&(entity_id, feature_id))
            .ok_or(ScalerError::UnknownFeature { entity_id, feature_id })?;
        let transformed = if scale.sample_std == 0.0 {
            value - scale.mean
        } else {
            (value - scale.mean) / scale.sample_std
        };
        if !transformed.is_finite() {
            return Err(ScalerError::ArithmeticOverflow);
        }
        Ok(transformed)
    }

    pub fn features(&self) -> impl Iterator<Item = ((u64, u32), FeatureScale)> + '_ {
        self.scales.iter().map(|(key, value)| (*key, *value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn windows() -> Vec<ForecastWindow> {
        (1..=10).map(|id| ForecastWindow {
            window_id: id,
            entity_id: 1,
            history_start_us: i64::try_from(id).unwrap() * 10 - 10,
            forecast_origin_us: i64::try_from(id).unwrap() * 10,
            target_end_exclusive_us: i64::try_from(id).unwrap() * 10 + 10,
        }).collect()
    }

    fn plan(windows: &[ForecastWindow]) -> PurgedSplitPlan {
        super::super::splits::purged_chronological_split(windows, 6_000, 2_000, 0).unwrap()
    }

    fn sample(window_id: u64, entity_id: u64, feature_id: u32, observation_id: u64, value: f64) -> WindowFeatureValue {
        WindowFeatureValue { window_id, entity_id, feature_id, observation_id, value }
    }

    #[test]
    fn validation_and_test_values_do_not_affect_fitted_statistics() {
        let windows = windows();
        let plan = plan(&windows);
        let samples = vec![
            sample(1, 1, 1, 101, 1.0),
            sample(2, 1, 1, 102, 3.0),
            sample(7, 1, 1, 107, 1000.0),
            sample(9, 1, 1, 109, -9000.0),
        ];
        let scaler = TrainOnlyStandardizer::fit(&windows, &plan, &samples).unwrap();
        let stats = scaler.scale(1, 1).unwrap();
        assert_eq!(stats.count, 2);
        assert_eq!(stats.mean, 2.0);
        assert!((stats.sample_std - 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn repeated_raw_observation_in_overlapping_train_windows_is_counted_once() {
        let windows = windows();
        let plan = plan(&windows);
        let samples = vec![
            sample(1, 1, 1, 101, 1.0),
            sample(2, 1, 1, 101, 1.0),
            sample(3, 1, 1, 102, 3.0),
        ];
        let scaler = TrainOnlyStandardizer::fit(&windows, &plan, &samples).unwrap();
        assert_eq!(scaler.scale(1, 1).unwrap().count, 2);
        assert_eq!(scaler.scale(1, 1).unwrap().mean, 2.0);
    }

    #[test]
    fn conflicting_duplicate_observation_values_fail_closed() {
        let windows = windows();
        let plan = plan(&windows);
        let samples = vec![
            sample(1, 1, 1, 101, 1.0),
            sample(2, 1, 1, 101, 2.0),
        ];
        assert_eq!(
            TrainOnlyStandardizer::fit(&windows, &plan, &samples),
            Err(ScalerError::ConflictingDuplicateObservation {
                entity_id: 1, feature_id: 1, observation_id: 101,
            })
        );
    }

    #[test]
    fn statistics_are_separate_by_entity_and_feature() {
        let mut windows = windows();
        windows.extend((1..=10).map(|id| ForecastWindow {
            window_id: 100 + id,
            entity_id: 2,
            history_start_us: i64::try_from(id).unwrap() * 100 - 10,
            forecast_origin_us: i64::try_from(id).unwrap() * 100,
            target_end_exclusive_us: i64::try_from(id).unwrap() * 100 + 10,
        }));
        let plan = plan(&windows);
        let samples = vec![
            sample(1, 1, 1, 1, 1.0), sample(2, 1, 1, 2, 3.0),
            sample(1, 1, 2, 3, 10.0), sample(2, 1, 2, 4, 20.0),
            sample(101, 2, 1, 5, 100.0), sample(102, 2, 1, 6, 300.0),
        ];
        let scaler = TrainOnlyStandardizer::fit(&windows, &plan, &samples).unwrap();
        assert_eq!(scaler.scale(1, 1).unwrap().mean, 2.0);
        assert_eq!(scaler.scale(1, 2).unwrap().mean, 15.0);
        assert_eq!(scaler.scale(2, 1).unwrap().mean, 200.0);
        assert_eq!(scaler.scale(2, 2), None);
    }

    #[test]
    fn constant_and_single_sample_features_transform_without_nan() {
        let windows = windows();
        let plan = plan(&windows);
        let scaler = TrainOnlyStandardizer::fit(&windows, &plan, &[
            sample(1, 1, 1, 1, 7.0),
            sample(2, 1, 1, 2, 7.0),
            sample(3, 1, 2, 3, 5.0),
        ]).unwrap();
        assert_eq!(scaler.scale(1, 1).unwrap().sample_std, 0.0);
        assert_eq!(scaler.transform(1, 1, 7.0).unwrap(), 0.0);
        assert_eq!(scaler.transform(1, 2, 5.0).unwrap(), 0.0);
    }

    #[test]
    fn scaler_is_deterministic_under_training_sample_permutation() {
        let windows = windows();
        let plan = plan(&windows);
        let mut samples = vec![
            sample(1, 1, 1, 1, 1.0),
            sample(2, 1, 1, 2, 3.0),
            sample(3, 1, 1, 3, 5.0),
        ];
        let left = TrainOnlyStandardizer::fit(&windows, &plan, &samples).unwrap();
        samples.reverse();
        let right = TrainOnlyStandardizer::fit(&windows, &plan, &samples).unwrap();
        assert_eq!(left, right);
    }

    #[test]
    fn unknown_windows_wrong_entities_and_unknown_features_fail_closed() {
        let windows = windows();
        let plan = plan(&windows);
        assert_eq!(
            TrainOnlyStandardizer::fit(&windows, &plan, &[sample(99, 1, 1, 1, 1.0)]),
            Err(ScalerError::UnknownWindowId { window_id: 99 })
        );
        assert_eq!(
            TrainOnlyStandardizer::fit(&windows, &plan, &[sample(1, 2, 1, 1, 1.0)]),
            Err(ScalerError::WindowEntityMismatch { window_id: 1 })
        );
        let scaler = TrainOnlyStandardizer::fit(&windows, &plan, &[sample(1, 1, 1, 1, 1.0)]).unwrap();
        assert_eq!(
            scaler.transform(1, 2, 1.0),
            Err(ScalerError::UnknownFeature { entity_id: 1, feature_id: 2 })
        );
    }

    #[test]
    fn non_finite_validation_values_are_ignored_but_training_values_rejected() {
        let windows = windows();
        let plan = plan(&windows);
        let scaler = TrainOnlyStandardizer::fit(&windows, &plan, &[
            sample(1, 1, 1, 1, 1.0),
            sample(7, 1, 1, 2, f64::NAN),
        ]).unwrap();
        assert_eq!(scaler.scale(1, 1).unwrap().count, 1);
        assert_eq!(
            TrainOnlyStandardizer::fit(&windows, &plan, &[sample(1, 1, 1, 1, f64::NAN)]),
            Err(ScalerError::NonFiniteTrainingValue { window_id: 1 })
        );
    }
}
