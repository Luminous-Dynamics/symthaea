#![forbid(unsafe_code)]

//! Finite-sample calibration and benchmark-integrity primitives for CMAQ.
//!
//! The crate deliberately separates empirical calibration from theorem-level
//! validity. Exact binomial intervals summarize finite Monte Carlo samples;
//! they do not prove the conditional-expectation assumptions of an e-process.

use std::collections::HashSet;

use sha2::{Digest, Sha256};
use statrs::distribution::{Beta, ContinuousCDF};

/// Schema version for all metrics emitted by this crate.
pub mod scenarios;

pub const METRICS_SCHEMA_VERSION: &str = "CMAQ-059-METRICS-V1";
/// Domain separator for reproducible paired-stream seeds.
pub const SEED_SCHEMA_VERSION: &str = "CMAQ-059-PAIRED-STREAM-SEED-V1";

const SEED_DOMAIN: &[u8] = b"LUMINOUS-DYNAMICS/CMAQ-059/PAIRED-STREAM-SEED/V1\0";
const MAX_EXACT_COUNT: u64 = 1_u64 << 53;
const MAX_IDENTIFIER_BYTES: usize = 512;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CalibrationError {
    ZeroTrials,
    SuccessesExceedTrials,
    CountExceedsExactFloatRange,
    InvalidConfidence,
    InvalidFamilySize,
    EmptyNullFamily,
    EmptyIdentifier,
    IdentifierTooLong,
    DuplicateNullCell,
    InvalidNominalAlpha,
    InvalidTolerance,
    EmptyDetectionTrials,
    InvalidHorizon,
    ChangePointOutsideHorizon,
    AlarmOutsideHorizon,
    MixedDetectionCells,
    InvalidDistributionParameters,
    QuantileFailure,
}

impl std::fmt::Display for CalibrationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for CalibrationError {}

#[derive(Clone, Debug, PartialEq)]
pub struct BinomialInterval {
    pub successes: u64,
    pub trials: u64,
    pub estimate: f64,
    pub confidence: f64,
    pub lower: f64,
    pub upper: f64,
}

/// Computes the two-sided Clopper–Pearson interval.
///
/// For x successes in n independent Bernoulli trials, the endpoints are beta
/// quantiles. Boundary cases are handled explicitly so x=0 and x=n remain
/// well-defined. The caller is responsible for the independence unit: in CMAQ
/// that unit is one independently generated full-horizon replicate, not an
/// overlapping clip or a repeated time step from the same stream.
pub fn clopper_pearson(
    successes: u64,
    trials: u64,
    confidence: f64,
) -> Result<BinomialInterval, CalibrationError> {
    validate_binomial(successes, trials)?;
    validate_confidence(confidence)?;

    let tail = (1.0 - confidence) / 2.0;
    let lower = if successes == 0 {
        0.0
    } else {
        beta_quantile(successes as f64, (trials - successes + 1) as f64, tail)?
    };
    let upper = if successes == trials {
        1.0
    } else {
        beta_quantile(
            (successes + 1) as f64,
            (trials - successes) as f64,
            1.0 - tail,
        )?
    };

    Ok(BinomialInterval {
        successes,
        trials,
        estimate: successes as f64 / trials as f64,
        confidence,
        lower,
        upper,
    })
}

/// Computes a one-sided exact binomial upper confidence bound.
///
/// This is used with a Bonferroni allocation across all registered null cells.
/// If family_confidence is 0.95 and family_size is J, each cell receives tail
/// probability (1 - 0.95) / J. The resulting family has at least the declared
/// simultaneous coverage without assuming independence between cells.
pub fn clopper_pearson_upper(
    successes: u64,
    trials: u64,
    family_confidence: f64,
    family_size: usize,
) -> Result<f64, CalibrationError> {
    validate_binomial(successes, trials)?;
    validate_confidence(family_confidence)?;
    if family_size == 0 {
        return Err(CalibrationError::InvalidFamilySize);
    }

    if successes == trials {
        return Ok(1.0);
    }
    let per_cell_tail = (1.0 - family_confidence) / family_size as f64;
    let quantile = 1.0 - per_cell_tail;
    beta_quantile(
        (successes + 1) as f64,
        (trials - successes) as f64,
        quantile,
    )
}

fn validate_binomial(successes: u64, trials: u64) -> Result<(), CalibrationError> {
    if trials == 0 {
        return Err(CalibrationError::ZeroTrials);
    }
    if successes > trials {
        return Err(CalibrationError::SuccessesExceedTrials);
    }
    if trials > MAX_EXACT_COUNT {
        return Err(CalibrationError::CountExceedsExactFloatRange);
    }
    Ok(())
}

fn validate_confidence(confidence: f64) -> Result<(), CalibrationError> {
    if !confidence.is_finite() || confidence <= 0.0 || confidence >= 1.0 {
        return Err(CalibrationError::InvalidConfidence);
    }
    Ok(())
}

fn beta_quantile(shape_a: f64, shape_b: f64, probability: f64) -> Result<f64, CalibrationError> {
    let distribution =
        Beta::new(shape_a, shape_b).map_err(|_| CalibrationError::InvalidDistributionParameters)?;
    let value = distribution
        .try_inverse_cdf(probability)
        .map_err(|_| CalibrationError::QuantileFailure)?;
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(CalibrationError::QuantileFailure)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NullCalibrationCell {
    /// Unique scenario/horizon/alpha cell identifier within an arm.
    pub cell_id: String,
    pub arm_id: String,
    pub nominal_alpha: f64,
    pub false_alarms: u64,
    pub trials: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NullCellSummary {
    pub cell_id: String,
    pub arm_id: String,
    pub nominal_alpha: f64,
    pub observed_rate: f64,
    pub pointwise_interval: BinomialInterval,
    pub simultaneous_upper: f64,
    /// Empirical screen only; not a theorem or deployment qualification.
    pub within_tolerance: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NullFamilyReport {
    pub schema_version: &'static str,
    pub family_confidence: f64,
    pub calibration_tolerance: f64,
    pub family_size: usize,
    pub all_cells_within_tolerance: bool,
    pub cells: Vec<NullCellSummary>,
}

/// Reports each null cell with a pointwise interval and a familywise upper
/// bound. The cell list must be frozen before running the benchmark.
pub fn summarize_null_family(
    cells: &[NullCalibrationCell],
    family_confidence: f64,
    calibration_tolerance: f64,
) -> Result<NullFamilyReport, CalibrationError> {
    if cells.is_empty() {
        return Err(CalibrationError::EmptyNullFamily);
    }
    validate_confidence(family_confidence)?;
    if !calibration_tolerance.is_finite()
        || calibration_tolerance < 0.0
        || calibration_tolerance >= 1.0
    {
        return Err(CalibrationError::InvalidTolerance);
    }

    let mut seen = HashSet::with_capacity(cells.len());
    let family_size = cells.len();
    let mut summaries = Vec::with_capacity(family_size);
    let mut all_cells_within_tolerance = true;

    for cell in cells {
        validate_identifier(&cell.cell_id)?;
        validate_identifier(&cell.arm_id)?;
        if !seen.insert((cell.cell_id.clone(), cell.arm_id.clone())) {
            return Err(CalibrationError::DuplicateNullCell);
        }
        if !cell.nominal_alpha.is_finite() || cell.nominal_alpha <= 0.0 || cell.nominal_alpha >= 1.0
        {
            return Err(CalibrationError::InvalidNominalAlpha);
        }

        let pointwise_interval =
            clopper_pearson(cell.false_alarms, cell.trials, family_confidence)?;
        let simultaneous_upper = clopper_pearson_upper(
            cell.false_alarms,
            cell.trials,
            family_confidence,
            family_size,
        )?;
        let within_tolerance =
            simultaneous_upper <= (cell.nominal_alpha + calibration_tolerance).min(1.0);
        all_cells_within_tolerance &= within_tolerance;

        summaries.push(NullCellSummary {
            cell_id: cell.cell_id.clone(),
            arm_id: cell.arm_id.clone(),
            nominal_alpha: cell.nominal_alpha,
            observed_rate: pointwise_interval.estimate,
            pointwise_interval,
            simultaneous_upper,
            within_tolerance,
        });
    }

    Ok(NullFamilyReport {
        schema_version: METRICS_SCHEMA_VERSION,
        family_confidence,
        calibration_tolerance,
        family_size,
        all_cells_within_tolerance,
        cells: summaries,
    })
}

fn validate_identifier(value: &str) -> Result<(), CalibrationError> {
    if value.trim().is_empty() {
        return Err(CalibrationError::EmptyIdentifier);
    }
    if value.len() > MAX_IDENTIFIER_BYTES {
        return Err(CalibrationError::IdentifierTooLong);
    }
    Ok(())
}

/// One independent full-horizon alternative replicate.
///
/// change_point is the number of initial, pre-change observations. Therefore
/// an alarm at sequence <= change_point is an early alarm, not a detection.
/// A miss or early alarm receives restricted delay horizon + 1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DetectionTrial {
    pub horizon: u64,
    pub change_point: u64,
    pub first_alarm: Option<u64>,
}

impl DetectionTrial {
    pub fn restricted_delay(&self) -> Result<u64, CalibrationError> {
        self.validate()?;
        match self.first_alarm {
            Some(sequence) if sequence > self.change_point => Ok(sequence - self.change_point),
            _ => self
                .horizon
                .checked_add(1)
                .ok_or(CalibrationError::InvalidHorizon),
        }
    }

    pub fn is_early_alarm(&self) -> Result<bool, CalibrationError> {
        self.validate()?;
        Ok(matches!(
            self.first_alarm,
            Some(sequence) if sequence <= self.change_point
        ))
    }

    pub fn is_detection(&self) -> Result<bool, CalibrationError> {
        self.validate()?;
        Ok(matches!(
            self.first_alarm,
            Some(sequence) if sequence > self.change_point
        ))
    }

    fn validate(&self) -> Result<(), CalibrationError> {
        if self.horizon == 0 {
            return Err(CalibrationError::InvalidHorizon);
        }
        if self.change_point >= self.horizon {
            return Err(CalibrationError::ChangePointOutsideHorizon);
        }
        if let Some(sequence) = self.first_alarm {
            if sequence == 0 || sequence > self.horizon {
                return Err(CalibrationError::AlarmOutsideHorizon);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DetectionSummary {
    pub schema_version: &'static str,
    pub trials: u64,
    pub detections: u64,
    pub early_alarms: u64,
    pub unconditional_power: f64,
    pub power_interval: BinomialInterval,
    pub restricted_mean_delay: f64,
    pub restricted_delay_p50: u64,
    pub restricted_delay_p90: u64,
    pub restricted_delay_p95: u64,
    pub horizon: u64,
    pub change_point: u64,
}

/// Summarizes power and restricted delay without dropping early alarms or
/// misses. All trials in a cell must share one horizon and one change point.
pub fn summarize_detection_trials(
    trials: &[DetectionTrial],
    confidence: f64,
) -> Result<DetectionSummary, CalibrationError> {
    if trials.is_empty() {
        return Err(CalibrationError::EmptyDetectionTrials);
    }
    validate_confidence(confidence)?;

    let horizon = trials[0].horizon;
    let change_point = trials[0].change_point;
    let mut detections = 0_u64;
    let mut early_alarms = 0_u64;
    let mut delays = Vec::with_capacity(trials.len());

    for trial in trials {
        trial.validate()?;
        if trial.horizon != horizon || trial.change_point != change_point {
            return Err(CalibrationError::MixedDetectionCells);
        }
        if trial.is_detection()? {
            detections += 1;
        }
        if trial.is_early_alarm()? {
            early_alarms += 1;
        }
        delays.push(trial.restricted_delay()?);
    }

    let trial_count = trials.len() as u64;
    let power_interval = clopper_pearson(detections, trial_count, confidence)?;
    let restricted_mean_delay =
        delays.iter().map(|delay| *delay as f64).sum::<f64>() / trial_count as f64;

    let mut sorted = delays;
    sorted.sort_unstable();

    Ok(DetectionSummary {
        schema_version: METRICS_SCHEMA_VERSION,
        trials: trial_count,
        detections,
        early_alarms,
        unconditional_power: power_interval.estimate,
        power_interval,
        restricted_mean_delay,
        restricted_delay_p50: nearest_rank(&sorted, 0.50),
        restricted_delay_p90: nearest_rank(&sorted, 0.90),
        restricted_delay_p95: nearest_rank(&sorted, 0.95),
        horizon,
        change_point,
    })
}

fn nearest_rank(sorted: &[u64], quantile: f64) -> u64 {
    debug_assert!(!sorted.is_empty());
    let rank = ((quantile * sorted.len() as f64).ceil() as usize)
        .max(1)
        .min(sorted.len());
    sorted[rank - 1]
}

/// Derives a stable seed digest for one paired source stream.
///
/// The same (experiment, scenario, replicate) key is deliberately used by
/// every algorithm arm. Arm ID is intentionally excluded from this key:
/// paired arms must see identical underlying streams. This digest is a seed
/// schedule, not a random number generator and not proof that source
/// replicates are independent.
pub fn paired_stream_seed(
    experiment_id: &str,
    scenario_id: &str,
    replicate_id: u64,
) -> Result<[u8; 32], CalibrationError> {
    validate_identifier(experiment_id)?;
    validate_identifier(scenario_id)?;

    let mut hasher = Sha256::new();
    hasher.update(SEED_DOMAIN);
    update_length_prefixed(&mut hasher, experiment_id.as_bytes());
    update_length_prefixed(&mut hasher, scenario_id.as_bytes());
    hasher.update(replicate_id.to_be_bytes());
    let digest = hasher.finalize();

    let mut output = [0_u8; 32];
    output.copy_from_slice(&digest);
    Ok(output)
}

fn update_length_prefixed(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value);
}

pub fn seed_hex(seed: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in seed {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_alarm_clopper_pearson_upper_is_exact_boundary_case() {
        let interval = clopper_pearson(0, 10_000, 0.95).unwrap();
        assert_eq!(interval.lower, 0.0);
        assert_eq!(interval.estimate, 0.0);
        // Two-sided 95% Clopper–Pearson uses 0.025 in the upper tail;
        // the one-sided 95% "rule of three" value (~0.00030) is different.
        assert!(interval.upper > 0.00036);
        assert!(interval.upper < 0.00038);
    }

    #[test]
    fn all_alarm_clopper_pearson_upper_is_one() {
        let interval = clopper_pearson(10_000, 10_000, 0.95).unwrap();
        assert!(interval.lower > 0.999);
        assert_eq!(interval.upper, 1.0);
    }

    #[test]
    fn pointwise_reference_interval_contains_observed_five_percent_rate() {
        let interval = clopper_pearson(500, 10_000, 0.95).unwrap();
        assert_eq!(interval.estimate, 0.05);
        assert!(interval.lower < 0.05);
        assert!(interval.upper > 0.05);
        assert!(interval.lower > 0.044);
        assert!(interval.upper < 0.056);
    }

    #[test]
    fn simultaneous_upper_bound_is_no_smaller_than_pointwise_one_sided_bound() {
        let single = clopper_pearson_upper(500, 10_000, 0.95, 1).unwrap();
        let family = clopper_pearson_upper(500, 10_000, 0.95, 12).unwrap();
        assert!(family >= single);
        assert!(family < 0.06);
    }

    #[test]
    fn null_family_report_uses_registered_family_size_and_tolerance() {
        let cells = vec![
            NullCalibrationCell {
                cell_id: "iid-equal-H128-a05".into(),
                arm_id: "cmaq-058".into(),
                nominal_alpha: 0.05,
                false_alarms: 400,
                trials: 10_000,
            },
            NullCalibrationCell {
                cell_id: "heteroskedastic-equal-H128-a05".into(),
                arm_id: "cmaq-058".into(),
                nominal_alpha: 0.05,
                false_alarms: 420,
                trials: 10_000,
            },
        ];
        let report = summarize_null_family(&cells, 0.95, 0.01).unwrap();
        assert_eq!(report.family_size, 2);
        assert!(report.all_cells_within_tolerance);
        assert!(
            report
                .cells
                .iter()
                .all(|cell| { cell.simultaneous_upper >= cell.pointwise_interval.upper - 1e-12 })
        );
    }

    #[test]
    fn null_family_flags_a_material_calibration_exceedance() {
        let cells = vec![NullCalibrationCell {
            cell_id: "shifted-null-control".into(),
            arm_id: "test-arm".into(),
            nominal_alpha: 0.05,
            false_alarms: 900,
            trials: 10_000,
        }];
        let report = summarize_null_family(&cells, 0.95, 0.005).unwrap();
        assert!(!report.all_cells_within_tolerance);
        assert!(!report.cells[0].within_tolerance);
    }

    #[test]
    fn restricted_delay_counts_misses_and_prechange_alarms_as_h_plus_one() {
        let post_change = DetectionTrial {
            horizon: 10,
            change_point: 3,
            first_alarm: Some(5),
        };
        let early = DetectionTrial {
            horizon: 10,
            change_point: 3,
            first_alarm: Some(2),
        };
        let missed = DetectionTrial {
            horizon: 10,
            change_point: 3,
            first_alarm: None,
        };
        assert_eq!(post_change.restricted_delay().unwrap(), 2);
        assert_eq!(early.restricted_delay().unwrap(), 11);
        assert_eq!(missed.restricted_delay().unwrap(), 11);
        assert!(early.is_early_alarm().unwrap());
        assert!(!early.is_detection().unwrap());
        assert!(post_change.is_detection().unwrap());
    }

    #[test]
    fn detection_summary_keeps_early_alarms_and_misses_in_denominator() {
        let trials = vec![
            DetectionTrial {
                horizon: 10,
                change_point: 3,
                first_alarm: Some(5),
            },
            DetectionTrial {
                horizon: 10,
                change_point: 3,
                first_alarm: Some(2),
            },
            DetectionTrial {
                horizon: 10,
                change_point: 3,
                first_alarm: None,
            },
        ];
        let summary = summarize_detection_trials(&trials, 0.95).unwrap();
        assert_eq!(summary.trials, 3);
        assert_eq!(summary.detections, 1);
        assert_eq!(summary.early_alarms, 1);
        assert!((summary.unconditional_power - 1.0 / 3.0).abs() < 1e-12);
        assert!((summary.restricted_mean_delay - 8.0).abs() < 1e-12);
        assert_eq!(summary.restricted_delay_p50, 11);
        assert_eq!(summary.restricted_delay_p90, 11);
        assert_eq!(summary.restricted_delay_p95, 11);
    }

    #[test]
    fn seed_schedule_is_deterministic_and_scenario_scoped_not_arm_scoped() {
        let first = paired_stream_seed("cmaq-059-v1", "conditional-equality", 7).unwrap();
        let repeated = paired_stream_seed("cmaq-059-v1", "conditional-equality", 7).unwrap();
        let next_replicate = paired_stream_seed("cmaq-059-v1", "conditional-equality", 8).unwrap();
        let next_scenario = paired_stream_seed("cmaq-059-v1", "serial-equality", 7).unwrap();
        assert_eq!(first, repeated);
        assert_ne!(first, next_replicate);
        assert_ne!(first, next_scenario);
        assert_eq!(seed_hex(&first).len(), 64);
    }

    #[test]
    fn invalid_counts_confidence_and_detection_cells_fail_closed() {
        assert_eq!(
            clopper_pearson(0, 0, 0.95),
            Err(CalibrationError::ZeroTrials)
        );
        assert_eq!(
            clopper_pearson(11, 10, 0.95),
            Err(CalibrationError::SuccessesExceedTrials)
        );
        assert_eq!(
            clopper_pearson(1, 10, 1.0),
            Err(CalibrationError::InvalidConfidence)
        );
        assert_eq!(
            clopper_pearson_upper(1, 10, 0.95, 0),
            Err(CalibrationError::InvalidFamilySize)
        );
        assert_eq!(
            DetectionTrial {
                horizon: 10,
                change_point: 10,
                first_alarm: None
            }
            .restricted_delay(),
            Err(CalibrationError::ChangePointOutsideHorizon)
        );
    }
}
