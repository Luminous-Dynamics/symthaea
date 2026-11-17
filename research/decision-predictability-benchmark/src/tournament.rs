//! Offline evaluator for the CSV contract emitted by the independent chaos
//! reference generator.
//!
//! This first tournament lane is deliberately narrow: one chronological
//! continuation per generated trajectory, with a 60/10/30 train/validation/
//! test split, one fixed forecast origin, recursive free-running predictions,
//! and no test-time assimilation. It is a diagnostic, not a held-out
//! equation-family tournament and not a source of independent statistical
//! replicates.

use crate::baseline::{FittedBaseline, fit_linear_ar, fit_mean, fit_persistence};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;

const TRAIN_PERCENT: usize = 60;
const DEVELOPMENT_PERCENT: usize = 70;
const MIN_TOTAL_SAMPLES: usize = 100;
const AR_ORDERS: [usize; 4] = [1, 2, 4, 8];
const RIDGE_PENALTY: f64 = 0.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    X,
    Y,
    Z,
}

impl Channel {
    const ALL: [Self; 3] = [Self::X, Self::Y, Self::Z];

    fn as_str(self) -> &'static str {
        match self {
            Self::X => "x",
            Self::Y => "y",
            Self::Z => "z",
        }
    }

    fn index(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FixtureTrajectory {
    pub fixture_id: String,
    pub regime: String,
    pub values: Vec<[f64; 3]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TournamentError {
    EmptyInput,
    MissingRecords,
    InvalidHeader,
    FieldCount {
        line: usize,
        actual: usize,
    },
    InvalidStep {
        line: usize,
    },
    InvalidNumber {
        line: usize,
        field: &'static str,
    },
    NonFiniteValue {
        line: usize,
        field: &'static str,
    },
    EmptyFixtureId {
        line: usize,
    },
    RegimeMismatch {
        line: usize,
    },
    NonContiguousStep {
        line: usize,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for TournamentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(formatter, "fixture CSV input is empty"),
            Self::MissingRecords => write!(formatter, "fixture CSV contains no sample records"),
            Self::InvalidHeader => write!(
                formatter,
                "fixture CSV header does not match the v0 contract"
            ),
            Self::FieldCount { line, actual } => {
                write!(formatter, "line {line} has {actual} fields; expected 6")
            }
            Self::InvalidStep { line } => {
                write!(formatter, "line {line} has an invalid step index")
            }
            Self::InvalidNumber { line, field } => {
                write!(
                    formatter,
                    "line {line} has an invalid numeric field {field}"
                )
            }
            Self::NonFiniteValue { line, field } => {
                write!(
                    formatter,
                    "line {line} has a non-finite numeric field {field}"
                )
            }
            Self::EmptyFixtureId { line } => {
                write!(formatter, "line {line} has an empty fixture ID")
            }
            Self::RegimeMismatch { line } => {
                write!(
                    formatter,
                    "line {line} changes the regime for an existing fixture"
                )
            }
            Self::NonContiguousStep {
                line,
                expected,
                actual,
            } => write!(
                formatter,
                "line {line} has step {actual}; expected contiguous step {expected}"
            ),
        }
    }
}

impl Error for TournamentError {}

/// Parse the header/comment/CSV format emitted by chaos-fixtures.
///
/// The parser preserves fixture encounter order and requires step indices to
/// start at zero and increase by exactly one within each fixture. FNV digests
/// in the generator are intentionally not treated as authentication.
pub fn parse_fixture_csv(input: &str) -> Result<Vec<FixtureTrajectory>, TournamentError> {
    if input.trim().is_empty() {
        return Err(TournamentError::EmptyInput);
    }

    let mut fixtures: Vec<FixtureTrajectory> = Vec::new();
    let mut fixture_indices: HashMap<String, usize> = HashMap::new();
    let mut header_seen = false;
    let mut record_count = 0usize;

    for (zero_based_line, raw_line) in input.lines().enumerate() {
        let line_number = zero_based_line + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if line == "fixture,regime,step,x,y,z" {
            if header_seen || record_count > 0 {
                return Err(TournamentError::InvalidHeader);
            }
            header_seen = true;
            continue;
        }

        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != 6 {
            return Err(TournamentError::FieldCount {
                line: line_number,
                actual: fields.len(),
            });
        }

        let fixture_id = fields[0].trim();
        if fixture_id.is_empty() {
            return Err(TournamentError::EmptyFixtureId { line: line_number });
        }
        let regime = fields[1].trim();
        let step = fields[2]
            .trim()
            .parse::<usize>()
            .map_err(|_| TournamentError::InvalidStep { line: line_number })?;

        let x = parse_value(fields[3], line_number, "x")?;
        let y = parse_value(fields[4], line_number, "y")?;
        let z = parse_value(fields[5], line_number, "z")?;

        let fixture_index = if let Some(index) = fixture_indices.get(fixture_id).copied() {
            let existing = &fixtures[index];
            if existing.regime != regime {
                return Err(TournamentError::RegimeMismatch { line: line_number });
            }
            let expected_step = existing.values.len();
            if step != expected_step {
                return Err(TournamentError::NonContiguousStep {
                    line: line_number,
                    expected: expected_step,
                    actual: step,
                });
            }
            index
        } else {
            if step != 0 {
                return Err(TournamentError::NonContiguousStep {
                    line: line_number,
                    expected: 0,
                    actual: step,
                });
            }
            let index = fixtures.len();
            fixtures.push(FixtureTrajectory {
                fixture_id: fixture_id.to_owned(),
                regime: regime.to_owned(),
                values: Vec::new(),
            });
            fixture_indices.insert(fixture_id.to_owned(), index);
            index
        };

        fixtures[fixture_index].values.push([x, y, z]);
        record_count += 1;
    }

    if record_count == 0 {
        return Err(TournamentError::MissingRecords);
    }
    Ok(fixtures)
}

fn parse_value(value: &str, line: usize, field: &'static str) -> Result<f64, TournamentError> {
    let parsed = value
        .trim()
        .parse::<f64>()
        .map_err(|_| TournamentError::InvalidNumber { line, field })?;
    if !parsed.is_finite() {
        return Err(TournamentError::NonFiniteValue { line, field });
    }
    Ok(parsed)
}

#[derive(Clone, Debug)]
struct ModelSpec {
    name: String,
    order: Option<usize>,
    validation_nrmse: Option<f64>,
}

#[derive(Clone, Debug)]
struct SuccessfulForecast {
    spec: ModelSpec,
    predictions: Vec<f64>,
}

#[derive(Clone, Debug)]
struct FailedForecast {
    spec: ModelSpec,
    status: &'static str,
}

#[derive(Clone, Copy, Debug)]
struct ErrorMetrics {
    mae: f64,
    rmse: f64,
    nrmse: f64,
}

/// Evaluate every fixture/channel and return machine-readable CSV.
///
/// Split contract: first 60% training, next 10% validation, final 30% test.
/// AR order is selected only on the validation continuation from {1,2,4,8};
/// the selected order and all fixed-order arms are then refit on the first
/// 70%. Every test forecast is recursive/free-running from that one as-of
/// origin. No later test observation is fed back into a model.
///
/// The result carries no confidence interval: lead times within a single
/// deterministic trajectory are correlated, and this protocol supplies only
/// one trajectory per fixture.
pub fn evaluate_fixture_csv(input: &str) -> Result<String, TournamentError> {
    let fixtures = parse_fixture_csv(input)?;
    let mut output = String::new();
    output.push_str("# protocol=chronological_continuation_diagnostic_v0\n");
    output.push_str(&format!(
        "# source_csv_fnv1a_non_security={:016x}\n",
        fnv1a_non_security(input.as_bytes())
    ));
    output.push_str("# split=60_percent_train,10_percent_validation,30_percent_test\n");
    output.push_str(
        "# forecast=recursive_free_running_from_one_fixed_as_of_origin;no_test_assimilation\n",
    );
    output.push_str(
        "# ar_selection=minimum_full_validation_rollout_nrmse_over_orders_1_2_4_8;no_test_tuning\n",
    );
    output.push_str("# nrmse=rmse_divided_by_development_population_standard_deviation\n");
    output.push_str("# scope=within_trajectory_same_regime_continuation_only;not_held_out_family_generalization\n");
    output.push_str("# uncertainty=no_confidence_intervals;forecast_leads_are_correlated\n");
    output.push_str("fixture,regime,channel,model,ar_order,train_n,validation_n,test_n,origin_step,validation_nrmse,lead_first,lead_last,observations,mae,rmse,nrmse,skill_vs_persistence,status\n");

    for fixture in &fixtures {
        for channel in Channel::ALL {
            evaluate_series(fixture, channel, &mut output);
        }
    }
    Ok(output)
}

fn evaluate_series(fixture: &FixtureTrajectory, channel: Channel, output: &mut String) {
    let values: Vec<f64> = fixture
        .values
        .iter()
        .map(|row| row[channel.index()])
        .collect();
    let sample_count = values.len();

    if sample_count < MIN_TOTAL_SAMPLES {
        write_status_row(
            output,
            fixture,
            channel,
            "none",
            None,
            0,
            0,
            sample_count,
            sample_count.saturating_sub(1),
            None,
            "insufficient_samples",
        );
        return;
    }

    let train_end = sample_count * TRAIN_PERCENT / 100;
    let development_end = sample_count * DEVELOPMENT_PERCENT / 100;
    if train_end < 2 || development_end <= train_end || development_end >= sample_count {
        write_status_row(
            output,
            fixture,
            channel,
            "none",
            None,
            train_end,
            development_end.saturating_sub(train_end),
            sample_count.saturating_sub(development_end),
            development_end.saturating_sub(1),
            None,
            "invalid_split",
        );
        return;
    }

    let training = &values[..train_end];
    let validation = &values[train_end..development_end];
    let development = &values[..development_end];
    let test = &values[development_end..];

    let validation_scale = population_std(training);
    let development_scale = population_std(development);
    if !validation_scale.is_finite()
        || validation_scale <= 0.0
        || !development_scale.is_finite()
        || development_scale <= 0.0
    {
        write_status_row(
            output,
            fixture,
            channel,
            "none",
            None,
            train_end,
            validation.len(),
            test.len(),
            development_end.saturating_sub(1),
            None,
            "constant_or_unscaled_channel",
        );
        return;
    }

    let mut validation_specs = Vec::new();
    for (name, order) in [("persistence", None), ("mean", None)] {
        let score = validation_score(name, order, training, validation, validation_scale);
        validation_specs.push(ModelSpec {
            name: name.to_owned(),
            order,
            validation_nrmse: score,
        });
    }

    let mut eligible_ar_orders = Vec::new();
    for order in AR_ORDERS {
        let score = validation_score("ar", Some(order), training, validation, validation_scale);
        if let Some(score) = score {
            eligible_ar_orders.push((order, score));
        }
        validation_specs.push(ModelSpec {
            name: format!("ar{order}"),
            order: Some(order),
            validation_nrmse: score,
        });
    }

    let selected_ar = eligible_ar_orders.into_iter().min_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });

    let mut final_specs = vec![validation_specs[0].clone(), validation_specs[1].clone()];
    final_specs.extend(validation_specs.into_iter().skip(2));
    if let Some((order, score)) = selected_ar {
        final_specs.push(ModelSpec {
            name: "ar_tuned".to_owned(),
            order: Some(order),
            validation_nrmse: Some(score),
        });
    } else {
        final_specs.push(ModelSpec {
            name: "ar_tuned".to_owned(),
            order: None,
            validation_nrmse: None,
        });
    }

    let mut successful = Vec::new();
    let mut failed = Vec::new();

    for spec in final_specs {
        if spec.name == "ar_tuned" && spec.order.is_none() {
            failed.push(FailedForecast {
                spec,
                status: "no_eligible_ar_order",
            });
            continue;
        }

        let fitted = match fit_spec(&spec, development) {
            Ok(model) => model,
            Err(_) => {
                failed.push(FailedForecast {
                    spec,
                    status: "fit_error",
                });
                continue;
            }
        };

        match fitted.forecast(development, test.len()) {
            Ok(predictions) => successful.push(SuccessfulForecast { spec, predictions }),
            Err(_) => failed.push(FailedForecast {
                spec,
                status: "forecast_error",
            }),
        }
    }

    let persistence_rmse = successful
        .iter()
        .find(|forecast| forecast.spec.name == "persistence")
        .map(|forecast| {
            horizon_bins(forecast.predictions.len())
                .into_iter()
                .map(|(first, last)| {
                    let start = first - 1;
                    let end = last.min(forecast.predictions.len());
                    let truth = &test[start..end];
                    let prediction = &forecast.predictions[start..end];
                    (
                        first,
                        last,
                        error_metrics(prediction, truth, development_scale).rmse,
                    )
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    for forecast in &successful {
        for (first, last) in horizon_bins(forecast.predictions.len()) {
            let start = first - 1;
            let end = last.min(forecast.predictions.len());
            if start >= end {
                continue;
            }

            let truth = &test[start..end];
            let prediction = &forecast.predictions[start..end];
            let metrics = error_metrics(prediction, truth, development_scale);
            let reference_rmse = persistence_rmse
                .iter()
                .find(|(reference_first, reference_last, _)| {
                    *reference_first == first && *reference_last == last
                })
                .map(|(_, _, rmse)| *rmse);
            let skill = reference_rmse
                .filter(|reference| *reference > 0.0 && reference.is_finite())
                .map(|reference| 1.0 - metrics.rmse / reference);

            write_metric_row(
                output,
                fixture,
                channel,
                &forecast.spec,
                train_end,
                validation.len(),
                test.len(),
                development_end.saturating_sub(1),
                first,
                end,
                metrics,
                skill,
            );
        }
    }

    for failure in failed {
        write_status_row(
            output,
            fixture,
            channel,
            &failure.spec.name,
            failure.spec.order,
            train_end,
            validation.len(),
            test.len(),
            development_end.saturating_sub(1),
            failure.spec.validation_nrmse,
            failure.status,
        );
    }
}

fn validation_score(
    name: &str,
    order: Option<usize>,
    history: &[f64],
    validation: &[f64],
    scale: f64,
) -> Option<f64> {
    let fitted = fit_named_model(name, order, history).ok()?;
    let forecast = fitted.forecast(history, validation.len()).ok()?;
    let metrics = error_metrics(&forecast, validation, scale);
    metrics.nrmse.is_finite().then_some(metrics.nrmse)
}

fn fit_spec(
    spec: &ModelSpec,
    history: &[f64],
) -> Result<FittedBaseline, crate::baseline::BaselineError> {
    fit_named_model(&spec.name, spec.order, history)
}

fn fit_named_model(
    name: &str,
    order: Option<usize>,
    history: &[f64],
) -> Result<FittedBaseline, crate::baseline::BaselineError> {
    match name {
        "persistence" => fit_persistence(history),
        "mean" => fit_mean(history),
        "ar" | "ar_tuned" | "ar1" | "ar2" | "ar4" | "ar8" => fit_linear_ar(
            history,
            order.ok_or(crate::baseline::BaselineError::InvalidAutoregressionOrder)?,
            RIDGE_PENALTY,
        ),
        _ => Err(crate::baseline::BaselineError::NumericalFailure),
    }
}

fn population_std(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let max_abs = values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if max_abs == 0.0 || !max_abs.is_finite() {
        return 0.0;
    }

    let normalized_mean =
        values.iter().map(|value| value / max_abs).sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let difference = value / max_abs - normalized_mean;
            difference * difference
        })
        .sum::<f64>()
        / values.len() as f64;
    max_abs * variance.sqrt()
}

fn error_metrics(prediction: &[f64], truth: &[f64], scale: f64) -> ErrorMetrics {
    if prediction.is_empty()
        || prediction.len() != truth.len()
        || !scale.is_finite()
        || scale <= 0.0
    {
        return ErrorMetrics {
            mae: f64::NAN,
            rmse: f64::NAN,
            nrmse: f64::NAN,
        };
    }

    let max_error = prediction
        .iter()
        .zip(truth.iter())
        .map(|(estimate, actual)| (estimate - actual).abs())
        .fold(0.0_f64, f64::max);
    if !max_error.is_finite() {
        return ErrorMetrics {
            mae: f64::NAN,
            rmse: f64::NAN,
            nrmse: f64::NAN,
        };
    }
    if max_error == 0.0 {
        return ErrorMetrics {
            mae: 0.0,
            rmse: 0.0,
            nrmse: 0.0,
        };
    }

    let normalized_abs = prediction
        .iter()
        .zip(truth.iter())
        .map(|(estimate, actual)| (estimate - actual).abs() / max_error);
    let count = prediction.len() as f64;
    let mae = max_error * (normalized_abs.clone().sum::<f64>() / count);
    let rmse = max_error * (normalized_abs.map(|value| value * value).sum::<f64>() / count).sqrt();
    let nrmse = rmse / scale;

    ErrorMetrics { mae, rmse, nrmse }
}

/// Return inclusive 1-based lead-time bins, clipped to the actual horizon.
fn horizon_bins(horizon: usize) -> Vec<(usize, usize)> {
    const BINS: [(usize, usize); 8] = [
        (1, 1),
        (2, 5),
        (6, 10),
        (11, 20),
        (21, 50),
        (51, 100),
        (101, 200),
        (201, usize::MAX),
    ];
    BINS.into_iter()
        .filter_map(|(first, last)| (first <= horizon).then_some((first, last.min(horizon))))
        .collect()
}

fn write_status_row(
    output: &mut String,
    fixture: &FixtureTrajectory,
    channel: Channel,
    model: &str,
    order: Option<usize>,
    train_n: usize,
    validation_n: usize,
    test_n: usize,
    origin_step: usize,
    validation_nrmse: Option<f64>,
    status: &str,
) {
    let fields = vec![
        fixture.fixture_id.clone(),
        fixture.regime.clone(),
        channel.as_str().to_owned(),
        model.to_owned(),
        order.map(|value| value.to_string()).unwrap_or_default(),
        train_n.to_string(),
        validation_n.to_string(),
        test_n.to_string(),
        origin_step.to_string(),
        format_optional(validation_nrmse),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        status.to_owned(),
    ];
    output.push_str(&fields.join(","));
    output.push('\n');
}

fn write_metric_row(
    output: &mut String,
    fixture: &FixtureTrajectory,
    channel: Channel,
    spec: &ModelSpec,
    train_n: usize,
    validation_n: usize,
    test_n: usize,
    origin_step: usize,
    lead_first: usize,
    lead_last: usize,
    metrics: ErrorMetrics,
    skill: Option<f64>,
) {
    let fields = vec![
        fixture.fixture_id.clone(),
        fixture.regime.clone(),
        channel.as_str().to_owned(),
        spec.name.clone(),
        spec.order
            .map(|value| value.to_string())
            .unwrap_or_default(),
        train_n.to_string(),
        validation_n.to_string(),
        test_n.to_string(),
        origin_step.to_string(),
        format_optional(spec.validation_nrmse),
        lead_first.to_string(),
        lead_last.to_string(),
        (lead_last - lead_first + 1).to_string(),
        format!("{:.12e}", metrics.mae),
        format!("{:.12e}", metrics.rmse),
        format!("{:.12e}", metrics.nrmse),
        format_optional(skill),
        "ok".to_owned(),
    ];
    output.push_str(&fields.join(","));
    output.push('\n');
}

fn format_optional(value: Option<f64>) -> String {
    value
        .filter(|number| number.is_finite())
        .map(|number| format!("{number:.12e}"))
        .unwrap_or_default()
}

fn fnv1a_non_security(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn csv_for(values: &[f64]) -> String {
        let mut input =
            String::from("# suite=symthaea-chaos-reference-suite-v0\nfixture,regime,step,x,y,z\n");
        for (step, value) in values.iter().enumerate() {
            input.push_str(&format!(
                "ar_fixture,known_regime,{step},{value:.17e},1.0,2.0\n"
            ));
        }
        input
    }

    #[test]
    fn parses_generator_comments_header_and_multiple_fixtures() {
        let input = "# seed=7\nfixture,regime,step,x,y,z\na,stable,0,1,2,3\na,stable,1,2,3,4\nb,periodic,0,5,6,7\n";
        let fixtures = parse_fixture_csv(input).unwrap();
        assert_eq!(fixtures.len(), 2);
        assert_eq!(fixtures[0].fixture_id, "a");
        assert_eq!(fixtures[0].values.len(), 2);
        assert_eq!(fixtures[1].regime, "periodic");
    }

    #[test]
    fn rejects_noncontiguous_steps_and_regime_changes() {
        let gap = "fixture,regime,step,x,y,z\na,stable,0,1,2,3\na,stable,2,2,3,4\n";
        assert_eq!(
            parse_fixture_csv(gap),
            Err(TournamentError::NonContiguousStep {
                line: 3,
                expected: 1,
                actual: 2,
            })
        );

        let mismatch = "fixture,regime,step,x,y,z\na,stable,0,1,2,3\na,chaotic,1,2,3,4\n";
        assert_eq!(
            parse_fixture_csv(mismatch),
            Err(TournamentError::RegimeMismatch { line: 3 })
        );
    }

    #[test]
    fn rejects_nonfinite_and_malformed_numeric_fields() {
        let nonfinite = "fixture,regime,step,x,y,z\na,stable,0,NaN,2,3\n";
        assert_eq!(
            parse_fixture_csv(nonfinite),
            Err(TournamentError::NonFiniteValue {
                line: 2,
                field: "x",
            })
        );
        let malformed = "fixture,regime,step,x,y,z\na,stable,0,1,2,oops\n";
        assert_eq!(
            parse_fixture_csv(malformed),
            Err(TournamentError::InvalidNumber {
                line: 2,
                field: "z",
            })
        );
    }

    #[test]
    fn evaluation_is_chronological_free_running_and_reports_all_baseline_arms() {
        let mut values = vec![0.2, -0.1];
        for _ in 0..148 {
            let index = values.len();
            let next = 0.1 + 0.72 * values[index - 1] - 0.15 * values[index - 2];
            values.push(next);
        }
        let result = evaluate_fixture_csv(&csv_for(&values)).unwrap();
        assert!(result.contains("# split=60_percent_train,10_percent_validation,30_percent_test"));
        assert!(result.contains(
            "# forecast=recursive_free_running_from_one_fixed_as_of_origin;no_test_assimilation"
        ));
        assert!(result.contains("ar_tuned"));
        assert!(result.contains("persistence"));
        assert!(result.contains("mean"));
        assert!(result.contains("skill_vs_persistence"));
        assert!(result.contains(",ok\n"));
    }

    #[test]
    fn constant_channels_are_not_misreported_as_successful_forecasts() {
        let values: Vec<f64> = (0..120).map(|index| (index as f64 * 0.07).sin()).collect();
        let result = evaluate_fixture_csv(&csv_for(&values)).unwrap();
        assert!(result.lines().any(|line| {
            line.starts_with("ar_fixture,known_regime,y,none,")
                && line.ends_with("constant_or_unscaled_channel")
        }));
        assert!(result.lines().any(|line| {
            line.starts_with("ar_fixture,known_regime,z,none,")
                && line.ends_with("constant_or_unscaled_channel")
        }));
    }

    #[test]
    fn horizon_bins_are_fixed_and_clipped() {
        assert_eq!(horizon_bins(12), vec![(1, 1), (2, 5), (6, 10), (11, 12)]);
    }

    #[test]
    fn source_digest_is_deterministic_and_non_security() {
        assert_eq!(
            fnv1a_non_security(b"fixture"),
            fnv1a_non_security(b"fixture")
        );
        assert_ne!(fnv1a_non_security(b"fixture"), fnv1a_non_security(b"other"));
    }
}
