//! Leakage-free as-of replay for market-price forecast vintages.
//!
//! A replay at decision time T may use only records available to the decision
//! maker by T. Forecast issue time, source availability time and target valid
//! time are kept distinct. A realized observation supersedes a forecast only
//! when the observation's target interval has ended and the observation was
//! available by T. The module selects evidence vintages; it does not predict
//! prices, authenticate source bytes, or establish forecast calibration.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PriceVintageKind<'a> {
    Forecast {
        issued_at_utc: i64,
        model_id: &'a str,
        model_sha256: &'a str,
    },
    Realized {
        observed_at_utc: i64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PriceVintage<'a> {
    pub record_id: &'a str,
    pub source_ref: &'a str,
    pub source_sha256: &'a str,
    pub series_id: &'a str,
    pub valid_period_start_utc: i64,
    pub valid_period_end_utc: i64,
    /// When this exact record became available to the decision workflow.
    /// This is not interchangeable with issue time or target valid time.
    pub available_at_utc: i64,
    pub value_minor_zar_per_mwh: i64,
    pub kind: PriceVintageKind<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AsOfReplayTarget<'a> {
    pub series_id: &'a str,
    pub valid_period_start_utc: i64,
    pub valid_period_end_utc: i64,
    /// A target can be scoped to one model family/version identity. Realized
    /// observations are common ground truth and are not filtered by model_id.
    pub model_id: Option<&'a str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectedVintageKind {
    Forecast,
    Realized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectedPriceVintage<'a> {
    pub target_index: usize,
    pub record_index: usize,
    pub record_id: &'a str,
    pub kind: SelectedVintageKind,
    pub value_minor_zar_per_mwh: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsOfReplayAssessment<'a> {
    pub decision_time_utc: i64,
    /// One entry per target that has a known eligible vintage, in target order.
    pub selected: Vec<SelectedPriceVintage<'a>>,
    /// Required target indices that had no eligible, known-at-decision vintage.
    pub missing_target_indices: Vec<usize>,
    /// Records whose source availability time is later than the decision time.
    pub future_known_excluded_count: u32,
    /// Eligible but older vintages superseded by a later vintage for a target.
    pub superseded_vintage_count: u32,
}

impl AsOfReplayAssessment<'_> {
    pub fn is_complete(&self) -> bool {
        self.missing_target_indices.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsOfReplayError {
    InvalidEvidenceReference,
    InvalidSourceDigest,
    InvalidModelDigest,
    InvalidTargetPeriod,
    InvalidVintagePeriod,
    InvalidVintageChronology,
    DuplicateRecordId,
    DuplicateTarget,
    AmbiguousLatestVintage,
    ModelScopeRequired,
    ArithmeticOverflow,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_vintage(vintage: PriceVintage<'_>) -> Result<(), AsOfReplayError> {
    if !nonempty(vintage.record_id) || !nonempty(vintage.source_ref) || !nonempty(vintage.series_id) {
        return Err(AsOfReplayError::InvalidEvidenceReference);
    }
    if !valid_sha256(vintage.source_sha256) {
        return Err(AsOfReplayError::InvalidSourceDigest);
    }
    if vintage.valid_period_start_utc >= vintage.valid_period_end_utc {
        return Err(AsOfReplayError::InvalidVintagePeriod);
    }

    match vintage.kind {
        PriceVintageKind::Forecast {
            issued_at_utc,
            model_id,
            model_sha256,
        } => {
            if !nonempty(model_id) {
                return Err(AsOfReplayError::InvalidEvidenceReference);
            }
            if !valid_sha256(model_sha256) {
                return Err(AsOfReplayError::InvalidModelDigest);
            }
            if issued_at_utc > vintage.available_at_utc {
                return Err(AsOfReplayError::InvalidVintageChronology);
            }
        }
        PriceVintageKind::Realized { observed_at_utc } => {
            if vintage.valid_period_end_utc > observed_at_utc
                || observed_at_utc > vintage.available_at_utc
            {
                return Err(AsOfReplayError::InvalidVintageChronology);
            }
        }
    }
    Ok(())
}

fn target_matches(vintage: PriceVintage<'_>, target: AsOfReplayTarget<'_>) -> bool {
    if vintage.series_id != target.series_id
        || vintage.valid_period_start_utc != target.valid_period_start_utc
        || vintage.valid_period_end_utc != target.valid_period_end_utc
    {
        return false;
    }
    match vintage.kind {
        PriceVintageKind::Forecast { model_id, .. } => target
            .model_id
            .map(|expected| expected == model_id)
            .unwrap_or(true),
        PriceVintageKind::Realized { .. } => true,
    }
}

fn priority_and_vintage_time(vintage: PriceVintage<'_>) -> (u8, i64) {
    match vintage.kind {
        PriceVintageKind::Forecast { issued_at_utc, .. } => (1, issued_at_utc),
        PriceVintageKind::Realized { observed_at_utc } => (2, observed_at_utc),
    }
}

/// Select the latest forecast vintage available at decision time, or a realized
/// observation if the interval has ended and the observation was also available.
/// The latest forecast is selected by issue time; a realized observation has
/// higher priority than a forecast for the same exact target interval.
pub fn replay_prices_as_of<'a>(
    decision_time_utc: i64,
    targets: &[AsOfReplayTarget<'_>],
    vintages: &'a [PriceVintage<'a>],
) -> Result<AsOfReplayAssessment<'a>, AsOfReplayError> {
    for (index, target) in targets.iter().enumerate() {
        if !nonempty(target.series_id)
            || target.valid_period_start_utc >= target.valid_period_end_utc
            || target.model_id.map(nonempty) == Some(false)
        {
            return Err(AsOfReplayError::InvalidTargetPeriod);
        }
        for previous in &targets[..index] {
            if previous.series_id == target.series_id
                && previous.valid_period_start_utc == target.valid_period_start_utc
                && previous.valid_period_end_utc == target.valid_period_end_utc
                && previous.model_id == target.model_id
            {
                return Err(AsOfReplayError::DuplicateTarget);
            }
        }
    }

    for (index, vintage) in vintages.iter().copied().enumerate() {
        validate_vintage(vintage)?;
        if vintages[..index]
            .iter()
            .any(|previous| previous.record_id == vintage.record_id)
        {
            return Err(AsOfReplayError::DuplicateRecordId);
        }
    }

    // Never select whichever forecast engine happened to publish last when
    // the target does not name a model. Model outputs are separate evidence,
    // not interchangeable revisions of one series.
    for target in targets
        .iter()
        .copied()
        .filter(|target| target.model_id.is_none())
    {
        let has_realized_observation = vintages.iter().copied().any(|vintage| {
            target_matches(vintage, target)
                && vintage.available_at_utc <= decision_time_utc
                && matches!(vintage.kind, PriceVintageKind::Realized { .. })
        });
        // A known realized price is model-independent. If no such observation
        // exists, competing forecast engines require an explicit target scope.
        if has_realized_observation {
            continue;
        }
        let mut forecast_model: Option<&str> = None;
        for vintage in vintages.iter().copied() {
            if !target_matches(vintage, target) || vintage.available_at_utc > decision_time_utc {
                continue;
            }
            if let PriceVintageKind::Forecast { model_id, .. } = vintage.kind {
                match forecast_model {
                    Some(existing) if existing != model_id => {
                        return Err(AsOfReplayError::ModelScopeRequired);
                    }
                    None => forecast_model = Some(model_id),
                    _ => {}
                }
            }
        }
    }

    let future_known_excluded_count = u32::try_from(
        vintages
            .iter()
            .filter(|vintage| vintage.available_at_utc > decision_time_utc)
            .count(),
    )
    .map_err(|_| AsOfReplayError::ArithmeticOverflow)?;

    let mut selected = Vec::new();
    let mut missing_target_indices = Vec::new();
    let mut superseded_vintage_count = 0_u32;

    for (target_index, target) in targets.iter().copied().enumerate() {
        let mut best_index: Option<usize> = None;
        let mut best_rank: Option<(u8, i64, i64)> = None;
        let mut best_tie_count = 0_u32;
        let mut eligible_count = 0_u32;

        for (record_index, vintage) in vintages.iter().copied().enumerate() {
            if !target_matches(vintage, target) || vintage.available_at_utc > decision_time_utc {
                continue;
            }

            let (priority, vintage_time) = priority_and_vintage_time(vintage);
            if priority == 2 && vintage_time > decision_time_utc {
                continue;
            }
            eligible_count = eligible_count.saturating_add(1);
            let rank = (priority, vintage_time, vintage.available_at_utc);

            match best_rank {
                None => {
                    best_index = Some(record_index);
                    best_rank = Some(rank);
                    best_tie_count = 1;
                }
                Some(current) if rank > current => {
                    best_index = Some(record_index);
                    best_rank = Some(rank);
                    best_tie_count = 1;
                }
                Some(current) if rank == current => {
                    best_tie_count = best_tie_count.saturating_add(1);
                }
                _ => {}
            }
        }

        if best_tie_count > 1 {
            return Err(AsOfReplayError::AmbiguousLatestVintage);
        }

        if let Some(record_index) = best_index {
            let vintage = vintages[record_index];
            selected.push(SelectedPriceVintage {
                target_index,
                record_index,
                record_id: vintage.record_id,
                kind: match vintage.kind {
                    PriceVintageKind::Forecast { .. } => SelectedVintageKind::Forecast,
                    PriceVintageKind::Realized { .. } => SelectedVintageKind::Realized,
                },
                value_minor_zar_per_mwh: vintage.value_minor_zar_per_mwh,
            });
            superseded_vintage_count = superseded_vintage_count
                .checked_add(eligible_count.saturating_sub(1))
                .ok_or(AsOfReplayError::ArithmeticOverflow)?;
        } else {
            missing_target_indices.push(target_index);
        }
    }

    Ok(AsOfReplayAssessment {
        decision_time_utc,
        selected,
        missing_target_indices,
        future_known_excluded_count,
        superseded_vintage_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE_SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const MODEL_SHA: &str = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

    fn target(model_id: Option<&'static str>) -> AsOfReplayTarget<'static> {
        AsOfReplayTarget {
            series_id: "SAWEM/SMP",
            valid_period_start_utc: 100,
            valid_period_end_utc: 200,
            model_id,
        }
    }

    fn forecast(
        id: &'static str,
        issued: i64,
        available: i64,
        value: i64,
        model: &'static str,
    ) -> PriceVintage<'static> {
        PriceVintage {
            record_id: id,
            source_ref: id,
            source_sha256: SOURCE_SHA,
            series_id: "SAWEM/SMP",
            valid_period_start_utc: 100,
            valid_period_end_utc: 200,
            available_at_utc: available,
            value_minor_zar_per_mwh: value,
            kind: PriceVintageKind::Forecast {
                issued_at_utc: issued,
                model_id: model,
                model_sha256: MODEL_SHA,
            },
        }
    }

    fn realized(
        id: &'static str,
        observed: i64,
        available: i64,
        value: i64,
    ) -> PriceVintage<'static> {
        PriceVintage {
            record_id: id,
            source_ref: id,
            source_sha256: SOURCE_SHA,
            series_id: "SAWEM/SMP",
            valid_period_start_utc: 100,
            valid_period_end_utc: 200,
            available_at_utc: available,
            value_minor_zar_per_mwh: value,
            kind: PriceVintageKind::Realized {
                observed_at_utc: observed,
            },
        }
    }

    #[test]
    fn selects_latest_forecast_known_at_decision_and_excludes_future_vintages() {
        let vintages = [
            forecast("fc-early", 70, 80, 500, "chronos-v1"),
            forecast("fc-latest-known", 90, 95, 600, "chronos-v1"),
            forecast("fc-future", 101, 102, 99_999, "chronos-v1"),
            realized("actual-not-yet-published", 200, 220, 750),
        ];
        let result = replay_prices_as_of(100, &[target(Some("chronos-v1"))], &vintages).unwrap();
        assert!(result.is_complete());
        assert_eq!(result.selected[0].record_id, "fc-latest-known");
        assert_eq!(result.selected[0].value_minor_zar_per_mwh, 600);
        assert_eq!(result.future_known_excluded_count, 2);
        assert_eq!(result.superseded_vintage_count, 1);
    }

    #[test]
    fn realized_price_supersedes_forecast_only_after_period_and_publication() {
        let vintages = [
            forecast("fc", 70, 80, 500, "chronos-v1"),
            realized("actual", 205, 210, 700),
        ];
        let result = replay_prices_as_of(220, &[target(Some("chronos-v1"))], &vintages).unwrap();
        assert!(result.is_complete());
        assert_eq!(result.selected[0].record_id, "actual");
        assert_eq!(result.selected[0].kind, SelectedVintageKind::Realized);
        assert_eq!(result.selected[0].value_minor_zar_per_mwh, 700);
        assert_eq!(result.superseded_vintage_count, 1);
    }

    #[test]
    fn future_known_realized_data_does_not_fill_a_missing_historical_target() {
        let vintages = [realized("future-actual", 205, 210, 700)];
        let result = replay_prices_as_of(150, &[target(None)], &vintages).unwrap();
        assert!(!result.is_complete());
        assert_eq!(result.selected.len(), 0);
        assert_eq!(result.missing_target_indices, vec![0]);
        assert_eq!(result.future_known_excluded_count, 1);
    }

    #[test]
    fn model_scoped_targets_do_not_mix_forecast_engines() {
        let vintages = [
            forecast("model-a", 90, 95, 600, "chronos-v1"),
            forecast("model-b", 92, 96, 900, "homer-v1"),
        ];
        let result = replay_prices_as_of(
            100,
            &[target(Some("chronos-v1")), target(Some("homer-v1"))],
            &vintages,
        )
        .unwrap();
        assert!(result.is_complete());
        assert_eq!(result.selected[0].value_minor_zar_per_mwh, 600);
        assert_eq!(result.selected[1].value_minor_zar_per_mwh, 900);
    }

    #[test]
    fn unscoped_target_with_multiple_forecast_engines_is_rejected() {
        let vintages = [
            forecast("model-a", 90, 95, 600, "chronos-v1"),
            forecast("model-b", 92, 96, 900, "homer-v1"),
        ];
        assert_eq!(
            replay_prices_as_of(100, &[target(None)], &vintages),
            Err(AsOfReplayError::ModelScopeRequired)
        );
    }

    #[test]
    fn forecast_issued_before_decision_but_published_after_is_excluded() {
        let vintages = [
            forecast("earlier-known", 70, 80, 500, "chronos-v1"),
            forecast("late-published", 90, 101, 900, "chronos-v1"),
        ];
        let result = replay_prices_as_of(100, &[target(Some("chronos-v1"))], &vintages).unwrap();
        assert_eq!(result.selected[0].record_id, "earlier-known");
        assert_eq!(result.selected[0].value_minor_zar_per_mwh, 500);
        assert_eq!(result.future_known_excluded_count, 1);
        assert_eq!(result.superseded_vintage_count, 0);
    }

    #[test]
    fn known_realized_price_is_model_independent_even_with_multiple_forecasts() {
        let vintages = [
            forecast("chronos", 90, 95, 600, "chronos-v1"),
            forecast("homer", 92, 96, 900, "homer-v1"),
            realized("actual", 205, 210, 700),
        ];
        let result = replay_prices_as_of(220, &[target(None)], &vintages).unwrap();
        assert!(result.is_complete());
        assert_eq!(result.selected[0].record_id, "actual");
        assert_eq!(result.selected[0].kind, SelectedVintageKind::Realized);
        assert_eq!(result.future_known_excluded_count, 0);
    }

    #[test]
    fn ambiguous_latest_forecasts_fail_instead_of_choosing_arbitrarily() {
        let vintages = [
            forecast("one", 90, 95, 600, "chronos-v1"),
            forecast("two", 90, 95, 601, "chronos-v1"),
        ];
        assert_eq!(
            replay_prices_as_of(100, &[target(Some("chronos-v1"))], &vintages),
            Err(AsOfReplayError::AmbiguousLatestVintage)
        );
    }

    #[test]
    fn invalid_source_and_model_digests_are_rejected() {
        let bad_source = [PriceVintage {
            source_sha256: "not-a-digest",
            ..forecast("bad-source", 90, 95, 600, "chronos-v1")
        }];
        assert_eq!(
            replay_prices_as_of(100, &[target(Some("chronos-v1"))], &bad_source),
            Err(AsOfReplayError::InvalidSourceDigest)
        );

        let bad_model = [PriceVintage {
            kind: PriceVintageKind::Forecast {
                issued_at_utc: 90,
                model_id: "chronos-v1",
                model_sha256: "bad-model-digest",
            },
            ..forecast("bad-model", 90, 95, 600, "chronos-v1")
        }];
        assert_eq!(
            replay_prices_as_of(100, &[target(Some("chronos-v1"))], &bad_model),
            Err(AsOfReplayError::InvalidModelDigest)
        );
    }

    #[test]
    fn observed_period_cannot_be_known_before_it_has_ended() {
        let invalid = [realized("early-actual", 190, 195, 700)];
        assert_eq!(
            replay_prices_as_of(200, &[target(None)], &invalid),
            Err(AsOfReplayError::InvalidVintageChronology)
        );
    }

    #[test]
    fn duplicate_record_ids_and_duplicate_targets_fail_closed() {
        let duplicate_records = [
            forecast("same", 90, 95, 600, "chronos-v1"),
            forecast("same", 91, 96, 610, "chronos-v1"),
        ];
        assert_eq!(
            replay_prices_as_of(100, &[target(Some("chronos-v1"))], &duplicate_records),
            Err(AsOfReplayError::DuplicateRecordId)
        );
        assert_eq!(
            replay_prices_as_of(100, &[target(None), target(None)], &[]),
            Err(AsOfReplayError::DuplicateTarget)
        );
    }
}
