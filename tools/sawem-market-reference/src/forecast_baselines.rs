//! Leakage-safe, executable probabilistic baselines for electricity prices.
//!
//! This module constructs two simple, auditable benchmarks:
//! - persistence with an empirical distribution of contiguous historical price changes;
//! - training-window seasonal-bucket climatology.
//!
//! The caller supplies an immutable baseline configuration/build digest and a
//! training-manifest digest. Only observations whose intervals ended and whose
//! source was available by the declared training cutoff are eligible. This
//! module does not authenticate feeds or claim these baselines are optimal.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForecastBaselineMethod {
    PersistenceWithEmpiricalIncrements,
    SeasonalBucketClimatology,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HistoricalPriceObservation<'a> {
    pub record_id: &'a str,
    pub source_ref: &'a str,
    pub source_sha256: &'a str,
    pub series_id: &'a str,
    pub seasonal_bucket_id: &'a str,
    pub valid_period_start_utc: i64,
    pub valid_period_end_utc: i64,
    pub available_at_utc: i64,
    pub realized_price_minor_zar_per_mwh: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForecastBaselineQuery<'a> {
    pub baseline_model_id: &'a str,
    pub baseline_model_sha256: &'a str,
    pub series_id: &'a str,
    pub seasonal_bucket_id: &'a str,
    pub target_period_start_utc: i64,
    pub target_period_end_utc: i64,
    pub decision_time_utc: i64,
    /// Freeze the historical estimation boundary separately from decision time.
    pub training_cutoff_utc: i64,
    pub minimum_training_samples: u32,
    pub method: ForecastBaselineMethod,
    pub training_manifest_source_ref: &'a str,
    pub training_manifest_sha256: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BaselineTrainingSource<'a> {
    pub record_id: &'a str,
    pub source_ref: &'a str,
    pub source_sha256: &'a str,
    pub valid_period_start_utc: i64,
    pub valid_period_end_utc: i64,
    pub available_at_utc: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForecastBaselineReceipt<'a> {
    pub baseline_model_id: &'a str,
    pub baseline_model_sha256: &'a str,
    pub method: ForecastBaselineMethod,
    pub series_id: &'a str,
    pub seasonal_bucket_id: &'a str,
    pub target_period_start_utc: i64,
    pub target_period_end_utc: i64,
    pub forecast_issued_at_utc: i64,
    pub forecast_available_at_utc: i64,
    pub training_cutoff_utc: i64,
    pub q10_minor_zar_per_mwh: i64,
    pub q50_minor_zar_per_mwh: i64,
    pub q90_minor_zar_per_mwh: i64,
    pub training_sample_count: u32,
    pub future_or_post_cutoff_excluded_count: u32,
    pub training_manifest_source_ref: &'a str,
    pub training_manifest_sha256: &'a str,
    pub training_sources: Vec<BaselineTrainingSource<'a>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForecastBaselineResult<'a> {
    Determined(ForecastBaselineReceipt<'a>),
    Indeterminate { reason: &'static str },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForecastBaselineError {
    InvalidEvidenceReference,
    InvalidSourceDigest,
    InvalidModelDigest,
    InvalidTrainingManifestDigest,
    InvalidTargetPeriod,
    InvalidTrainingCutoff,
    InvalidMinimumSampleCount,
    DuplicateRecordId,
    DuplicateHistoricalPeriod,
    ArithmeticOverflow,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn nearest_rank_quantile(values: &[i64], quantile_bps: u16) -> Result<i64, ForecastBaselineError> {
    if values.is_empty() || quantile_bps == 0 || quantile_bps > 10_000 {
        return Err(ForecastBaselineError::InvalidMinimumSampleCount);
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let numerator = (sorted.len() as u128)
        .checked_mul(u128::from(quantile_bps))
        .ok_or(ForecastBaselineError::ArithmeticOverflow)?;
    let rank = numerator
        .checked_add(9_999)
        .ok_or(ForecastBaselineError::ArithmeticOverflow)?
        / 10_000;
    let index = usize::try_from(rank.saturating_sub(1))
        .map_err(|_| ForecastBaselineError::ArithmeticOverflow)?;
    sorted
        .get(index)
        .copied()
        .ok_or(ForecastBaselineError::ArithmeticOverflow)
}

fn validate_query(query: ForecastBaselineQuery<'_>) -> Result<(), ForecastBaselineError> {
    if !nonempty(query.baseline_model_id)
        || !nonempty(query.series_id)
        || !nonempty(query.seasonal_bucket_id)
        || !nonempty(query.training_manifest_source_ref)
    {
        return Err(ForecastBaselineError::InvalidEvidenceReference);
    }
    if !valid_sha256(query.baseline_model_sha256) {
        return Err(ForecastBaselineError::InvalidModelDigest);
    }
    if !valid_sha256(query.training_manifest_sha256) {
        return Err(ForecastBaselineError::InvalidTrainingManifestDigest);
    }
    if query.target_period_start_utc >= query.target_period_end_utc
        || query.decision_time_utc >= query.target_period_start_utc
    {
        return Err(ForecastBaselineError::InvalidTargetPeriod);
    }
    if query.training_cutoff_utc > query.decision_time_utc {
        return Err(ForecastBaselineError::InvalidTrainingCutoff);
    }
    if query.minimum_training_samples == 0 {
        return Err(ForecastBaselineError::InvalidMinimumSampleCount);
    }
    Ok(())
}

/// Construct a forecast from one of the frozen simple baselines. Future-known
/// or post-training-cutoff records are counted and excluded, never silently
/// promoted to the historical information set.
pub fn build_forecast_baseline<'a>(
    query: ForecastBaselineQuery<'a>,
    history: &'a [HistoricalPriceObservation<'a>],
) -> Result<ForecastBaselineResult<'a>, ForecastBaselineError> {
    validate_query(query)?;

    for (index, item) in history.iter().copied().enumerate() {
        if !nonempty(item.record_id)
            || !nonempty(item.source_ref)
            || !nonempty(item.series_id)
            || !nonempty(item.seasonal_bucket_id)
        {
            return Err(ForecastBaselineError::InvalidEvidenceReference);
        }
        if !valid_sha256(item.source_sha256) {
            return Err(ForecastBaselineError::InvalidSourceDigest);
        }
        if item.valid_period_start_utc >= item.valid_period_end_utc
            || item.available_at_utc < 0
        {
            return Err(ForecastBaselineError::InvalidTargetPeriod);
        }
        if history[..index]
            .iter()
            .any(|previous| previous.record_id == item.record_id)
        {
            return Err(ForecastBaselineError::DuplicateRecordId);
        }
        if history[..index].iter().any(|previous| {
            previous.series_id == item.series_id
                && previous.valid_period_start_utc == item.valid_period_start_utc
                && previous.valid_period_end_utc == item.valid_period_end_utc
        }) {
            return Err(ForecastBaselineError::DuplicateHistoricalPeriod);
        }
    }

    let matching_series = history
        .iter()
        .copied()
        .filter(|item| item.series_id == query.series_id)
        .collect::<Vec<_>>();
    let future_or_post_cutoff_excluded_count = u32::try_from(
        matching_series
            .iter()
            .filter(|item| {
                item.valid_period_end_utc > query.training_cutoff_utc
                    || item.available_at_utc > query.training_cutoff_utc
            })
            .count(),
    )
    .map_err(|_| ForecastBaselineError::ArithmeticOverflow)?;

    let mut eligible = matching_series
        .into_iter()
        .filter(|item| {
            item.valid_period_end_utc <= query.training_cutoff_utc
                && item.available_at_utc <= query.training_cutoff_utc
        })
        .collect::<Vec<_>>();
    eligible.sort_by_key(|item| (item.valid_period_start_utc, item.record_id));

    let (q10, q50, q90, used): (i64, i64, i64, Vec<&HistoricalPriceObservation<'a>>) =
        match query.method {
            ForecastBaselineMethod::PersistenceWithEmpiricalIncrements => {
                let Some(latest) = eligible.iter().max_by_key(|item| item.valid_period_end_utc) else {
                    return Ok(ForecastBaselineResult::Indeterminate {
                        reason: "no price observation is eligible by the training cutoff",
                    });
                };
                let mut increments = Vec::new();
                let mut used_points = Vec::new();
                for pair in eligible.windows(2) {
                    if pair[0].valid_period_end_utc != pair[1].valid_period_start_utc {
                        continue;
                    }
                    let change = i128::from(pair[1].realized_price_minor_zar_per_mwh)
                        .checked_sub(i128::from(pair[0].realized_price_minor_zar_per_mwh))
                        .ok_or(ForecastBaselineError::ArithmeticOverflow)?;
                    increments.push(
                        i64::try_from(change).map_err(|_| ForecastBaselineError::ArithmeticOverflow)?,
                    );
                    used_points.push(pair[0]);
                    used_points.push(pair[1]);
                }
                if increments.len() < query.minimum_training_samples as usize {
                    return Ok(ForecastBaselineResult::Indeterminate {
                        reason: "too few contiguous historical price changes for persistence baseline",
                    });
                }
                let delta_q10 = nearest_rank_quantile(&increments, 1_000)?;
                let delta_q50 = nearest_rank_quantile(&increments, 5_000)?;
                let delta_q90 = nearest_rank_quantile(&increments, 9_000)?;
                let latest_price = i128::from(latest.realized_price_minor_zar_per_mwh);
                let q10 = i64::try_from(
                    latest_price
                        .checked_add(i128::from(delta_q10))
                        .ok_or(ForecastBaselineError::ArithmeticOverflow)?,
                )
                .map_err(|_| ForecastBaselineError::ArithmeticOverflow)?;
                let q50 = i64::try_from(
                    latest_price
                        .checked_add(i128::from(delta_q50))
                        .ok_or(ForecastBaselineError::ArithmeticOverflow)?,
                )
                .map_err(|_| ForecastBaselineError::ArithmeticOverflow)?;
                let q90 = i64::try_from(
                    latest_price
                        .checked_add(i128::from(delta_q90))
                        .ok_or(ForecastBaselineError::ArithmeticOverflow)?,
                )
                .map_err(|_| ForecastBaselineError::ArithmeticOverflow)?;
                used_points.push(*latest);
                used_points.sort_by_key(|item| (item.valid_period_start_utc, item.record_id));
                used_points.dedup_by_key(|item| item.record_id);
                (q10, q50, q90, used_points)
            }
            ForecastBaselineMethod::SeasonalBucketClimatology => {
                let selected = eligible
                    .iter()
                    .copied()
                    .filter(|item| item.seasonal_bucket_id == query.seasonal_bucket_id)
                    .collect::<Vec<_>>();
                if selected.len() < query.minimum_training_samples as usize {
                    return Ok(ForecastBaselineResult::Indeterminate {
                        reason: "too few training observations in the target seasonal bucket",
                    });
                }
                let values = selected
                    .iter()
                    .map(|item| item.realized_price_minor_zar_per_mwh)
                    .collect::<Vec<_>>();
                (
                    nearest_rank_quantile(&values, 1_000)?,
                    nearest_rank_quantile(&values, 5_000)?,
                    nearest_rank_quantile(&values, 9_000)?,
                    selected,
                )
            }
        };

    if q10 > q50 || q50 > q90 {
        return Err(ForecastBaselineError::ArithmeticOverflow);
    }

    let training_sources = used
        .iter()
        .map(|item| BaselineTrainingSource {
            record_id: item.record_id,
            source_ref: item.source_ref,
            source_sha256: item.source_sha256,
            valid_period_start_utc: item.valid_period_start_utc,
            valid_period_end_utc: item.valid_period_end_utc,
            available_at_utc: item.available_at_utc,
        })
        .collect::<Vec<_>>();
    let training_sample_count = u32::try_from(training_sources.len())
        .map_err(|_| ForecastBaselineError::ArithmeticOverflow)?;

    Ok(ForecastBaselineResult::Determined(ForecastBaselineReceipt {
        baseline_model_id: query.baseline_model_id,
        baseline_model_sha256: query.baseline_model_sha256,
        method: query.method,
        series_id: query.series_id,
        seasonal_bucket_id: query.seasonal_bucket_id,
        target_period_start_utc: query.target_period_start_utc,
        target_period_end_utc: query.target_period_end_utc,
        forecast_issued_at_utc: query.decision_time_utc,
        forecast_available_at_utc: query.decision_time_utc,
        training_cutoff_utc: query.training_cutoff_utc,
        q10_minor_zar_per_mwh: q10,
        q50_minor_zar_per_mwh: q50,
        q90_minor_zar_per_mwh: q90,
        training_sample_count,
        future_or_post_cutoff_excluded_count,
        training_manifest_source_ref: query.training_manifest_source_ref,
        training_manifest_sha256: query.training_manifest_sha256,
        training_sources,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE_SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const MODEL_SHA: &str = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
    const MANIFEST_SHA: &str = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";

    fn observation(
        id: &'static str,
        start: i64,
        end: i64,
        price: i64,
        available: i64,
        bucket: &'static str,
    ) -> HistoricalPriceObservation<'static> {
        HistoricalPriceObservation {
            record_id: id,
            source_ref: id,
            source_sha256: SOURCE_SHA,
            series_id: "SAWEM/SMP",
            seasonal_bucket_id: bucket,
            valid_period_start_utc: start,
            valid_period_end_utc: end,
            available_at_utc: available,
            realized_price_minor_zar_per_mwh: price,
        }
    }

    fn query(method: ForecastBaselineMethod) -> ForecastBaselineQuery<'static> {
        ForecastBaselineQuery {
            baseline_model_id: "baseline/v1",
            baseline_model_sha256: MODEL_SHA,
            series_id: "SAWEM/SMP",
            seasonal_bucket_id: "weekday/18:00",
            target_period_start_utc: 120,
            target_period_end_utc: 135,
            decision_time_utc: 110,
            training_cutoff_utc: 105,
            minimum_training_samples: 6,
            method,
            training_manifest_source_ref: "training-manifest/001",
            training_manifest_sha256: MANIFEST_SHA,
        }
    }

    #[test]
    fn persistence_with_empirical_increments_uses_only_contiguous_training_history() {
        let history = [
            observation("p0", 0, 15, 100, 15, "weekday/18:00"),
            observation("p1", 15, 30, 110, 30, "weekday/18:00"),
            observation("p2", 30, 45, 105, 45, "weekday/18:00"),
            observation("p3", 45, 60, 120, 60, "weekday/18:00"),
            observation("p4", 60, 75, 115, 75, "weekday/18:00"),
            observation("p5", 75, 90, 130, 90, "weekday/18:00"),
            observation("p6", 90, 105, 125, 105, "weekday/18:00"),
        ];
        let result = build_forecast_baseline(
            query(ForecastBaselineMethod::PersistenceWithEmpiricalIncrements),
            &history,
        )
        .unwrap();
        let ForecastBaselineResult::Determined(result) = result else {
            panic!("six contiguous historical price changes are available");
        };
        assert_eq!(result.q10_minor_zar_per_mwh, 120);
        assert_eq!(result.q50_minor_zar_per_mwh, 120);
        assert_eq!(result.q90_minor_zar_per_mwh, 140);
        assert_eq!(result.forecast_issued_at_utc, 110);
        assert_eq!(result.forecast_available_at_utc, 110);
        assert_eq!(result.future_or_post_cutoff_excluded_count, 0);
    }

    #[test]
    fn climatology_uses_training_only_observations_in_the_target_bucket() {
        let history = [
            observation("bucket-a", 0, 15, 100, 15, "weekday/18:00"),
            observation("bucket-b", 15, 30, 150, 30, "weekday/18:00"),
            observation("bucket-c", 30, 45, 200, 45, "weekday/18:00"),
            observation("other-bucket", 45, 60, 999, 60, "weekend/18:00"),
            observation("too-late", 90, 105, 1_500, 106, "weekday/18:00"),
        ];
        let result = build_forecast_baseline(
            query(ForecastBaselineMethod::SeasonalBucketClimatology),
            &history,
        )
        .unwrap();
        let ForecastBaselineResult::Determined(result) = result else {
            panic!("three target-bucket training observations exist");
        };
        assert_eq!(result.q10_minor_zar_per_mwh, 100);
        assert_eq!(result.q50_minor_zar_per_mwh, 150);
        assert_eq!(result.q90_minor_zar_per_mwh, 200);
        assert_eq!(result.training_sample_count, 3);
        assert_eq!(result.future_or_post_cutoff_excluded_count, 1);
    }

    #[test]
    fn future_known_and_post_cutoff_prices_never_enter_a_baseline() {
        let history = [
            observation("p0", 0, 15, 100, 15, "weekday/18:00"),
            observation("p1", 15, 30, 110, 30, "weekday/18:00"),
            observation("p2", 30, 45, 105, 45, "weekday/18:00"),
            observation("p3", 45, 60, 120, 60, "weekday/18:00"),
            observation("p4", 60, 75, 115, 75, "weekday/18:00"),
            observation("p5", 75, 90, 130, 90, "weekday/18:00"),
            observation("after-cutoff", 105, 120, 999, 121, "weekday/18:00"),
        ];
        let result = build_forecast_baseline(
            query(ForecastBaselineMethod::PersistenceWithEmpiricalIncrements),
            &history,
        )
        .unwrap();
        let ForecastBaselineResult::Indeterminate { reason } = result else {
            panic!("only five contiguous increments qualify");
        };
        assert_eq!(
            reason,
            "too few contiguous historical price changes for persistence baseline"
        );
    }

    #[test]
    fn missing_training_history_is_indeterminate_not_a_zero_price_forecast() {
        assert_eq!(
            build_forecast_baseline(
                query(ForecastBaselineMethod::PersistenceWithEmpiricalIncrements),
                &[],
            ),
            Ok(ForecastBaselineResult::Indeterminate {
                reason: "no price observation is eligible by the training cutoff",
            })
        );
    }

    #[test]
    fn invalid_source_digests_duplicate_ids_and_duplicate_periods_are_rejected() {
        let bad_digest = [HistoricalPriceObservation {
            source_sha256: "bad",
            ..observation("bad", 0, 15, 100, 15, "weekday/18:00")
        }];
        assert_eq!(
            build_forecast_baseline(query(ForecastBaselineMethod::SeasonalBucketClimatology), &bad_digest),
            Err(ForecastBaselineError::InvalidSourceDigest)
        );

        let duplicate_id = [
            observation("same", 0, 15, 100, 15, "weekday/18:00"),
            observation("same", 15, 30, 110, 30, "weekday/18:00"),
        ];
        assert_eq!(
            build_forecast_baseline(query(ForecastBaselineMethod::SeasonalBucketClimatology), &duplicate_id),
            Err(ForecastBaselineError::DuplicateRecordId)
        );

        let duplicate_period = [
            observation("first", 0, 15, 100, 15, "weekday/18:00"),
            observation("second", 0, 15, 110, 16, "weekday/18:00"),
        ];
        assert_eq!(
            build_forecast_baseline(query(ForecastBaselineMethod::SeasonalBucketClimatology), &duplicate_period),
            Err(ForecastBaselineError::DuplicateHistoricalPeriod)
        );
    }

    #[test]
    fn baseline_cannot_be_built_after_the_decision_or_for_a_past_target() {
        assert_eq!(
            build_forecast_baseline(
                ForecastBaselineQuery {
                    training_cutoff_utc: 111,
                    ..query(ForecastBaselineMethod::SeasonalBucketClimatology)
                },
                &[],
            ),
            Err(ForecastBaselineError::InvalidTrainingCutoff)
        );
        assert_eq!(
            build_forecast_baseline(
                ForecastBaselineQuery {
                    target_period_start_utc: 100,
                    target_period_end_utc: 115,
                    ..query(ForecastBaselineMethod::SeasonalBucketClimatology)
                },
                &[],
            ),
            Err(ForecastBaselineError::InvalidTargetPeriod)
        );
    }
}
