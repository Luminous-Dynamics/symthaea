//! Monte Carlo comparison of fixed, three-bet and wide-grid e-processes.
//!
//! Usage: allostatic-risk-eprocess-null-monte-carlo <raw.csv> <summary.csv>
//! Each cohort contains five parallel bounded-risk monitor streams of 300 binary
//! losses each. Methods see identical losses. Cohort-level family-wise alarm rate
//! is the primary calibration statistic; power is evaluated at risks above epsilon.

use allostatic_control_bench::bounded_risk_eprocess::{
    BoundedRiskEProcess, MixtureBoundedRiskEProcess,
};
use std::error::Error;
use std::fs;
use std::io::{Error as IoError, ErrorKind};

const RISK_RATES: [f64; 6] = [0.01, 0.05, 0.075, 0.10, 0.20, 0.40];
const RISK_LABELS: [&str; 6] = ["0.010", "0.050", "0.075", "0.100", "0.200", "0.400"];
const COHORTS: usize = 1_000;
const MONITORS: usize = 5;
const HORIZON: usize = 300;
const EPSILON: f64 = 0.05;
const DELTA: f64 = 0.05;
const FIXED_FRACTION: f64 = 0.50;
const FIXED_BET: f64 = FIXED_FRACTION / EPSILON;
const MIX3: [f64; 3] = [0.25, 0.50, 0.75];
const MIX7: [f64; 7] = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.95];
const Z_975: f64 = 1.959_963_984_540_054;

const RAW_HEADER: &str = "risk_rate,cohort,monitor,events,event_trace,first_event_step,fixed_first_alarm_step,mix3_first_alarm_step,mix7_first_alarm_step,fixed_final_log_e,mix3_final_log_e,mix7_final_log_e";
const SUMMARY_HEADER: &str = "risk_rate,method,cohorts,monitors_per_cohort,horizon,cohort_fwer_count,cohort_fwer_rate,wilson_95_lower,wilson_95_upper,per_monitor_alarm_count,per_monitor_alarm_rate,mean_first_alarm_step,p95_first_alarm_step,median_first_alarm_step,mean_event_count,expected_event_count,epsilon,delta,log_alarm_threshold,bet_grid";

#[derive(Clone, Copy, Debug)]
struct StreamRecord {
    cohort: usize,
    monitor: usize,
    events: usize,
    event_trace: String,
    first_event_step: usize,
    fixed_first_alarm_step: usize,
    mix3_first_alarm_step: usize,
    mix7_first_alarm_step: usize,
    fixed_final_log_e: f64,
    mix3_final_log_e: f64,
    mix7_final_log_e: f64,
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn uniform(seed: u64) -> f64 {
    (splitmix64(seed) >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

fn wilson_interval(successes: usize, trials: usize) -> (f64, f64) {
    let n = trials as f64;
    let phat = successes as f64 / n;
    let z2 = Z_975 * Z_975;
    let denominator = 1.0 + z2 / n;
    let center = (phat + z2 / (2.0 * n)) / denominator;
    let half = Z_975
        * (phat * (1.0 - phat) / n + z2 / (4.0 * n * n)).sqrt()
        / denominator;
    ((center - half).max(0.0), (center + half).min(1.0))
}

fn percentile(values: &[usize], pct: usize) -> Option<usize> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = (sorted.len() * pct).div_ceil(100).max(1);
    Some(sorted[rank - 1])
}

fn mean_usize(values: &[usize]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<usize>() as f64 / values.len() as f64)
}

fn optional_float(value: Option<f64>) -> String {
    value.map_or_else(|| "NA".to_string(), |number| format!("{number:.9}"))
}

fn optional_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "NA".to_string(), |number| number.to_string())
}

fn thresholds() -> Result<(f64, f64, f64), Box<dyn Error>> {
    let fixed = BoundedRiskEProcess::new(EPSILON, FIXED_BET, DELTA, MONITORS)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "invalid fixed e-process"))?;
    let mix3 = MixtureBoundedRiskEProcess::new(EPSILON, MIX3, DELTA, MONITORS)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "invalid three-bet e-process"))?;
    let mix7 = MixtureBoundedRiskEProcess::new(EPSILON, MIX7, DELTA, MONITORS)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "invalid wide-grid e-process"))?;
    Ok((
        fixed.log_alarm_threshold(),
        mix3.log_alarm_threshold(),
        mix7.log_alarm_threshold(),
    ))
}

fn run_stream(
    rate_index: usize,
    cohort: usize,
    monitor_index: usize,
    risk_rate: f64,
) -> Result<StreamRecord, Box<dyn Error>> {
    let mut fixed = BoundedRiskEProcess::new(EPSILON, FIXED_BET, DELTA, MONITORS)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "invalid fixed e-process"))?;
    let mut mix3 = MixtureBoundedRiskEProcess::new(EPSILON, MIX3, DELTA, MONITORS)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "invalid three-bet e-process"))?;
    let mut mix7 = MixtureBoundedRiskEProcess::new(EPSILON, MIX7, DELTA, MONITORS)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "invalid wide-grid e-process"))?;

    let mut events = 0_usize;
    let mut event_trace = String::with_capacity(HORIZON);
    let mut first_event_step = HORIZON;
    let mut fixed_first_alarm_step = HORIZON;
    let mut mix3_first_alarm_step = HORIZON;
    let mut mix7_first_alarm_step = HORIZON;
    for step in 0..HORIZON {
        let seed = 0xD1B5_4A32_D192_ED03_u64
            ^ (rate_index as u64).wrapping_mul(0x9E37_79B9)
            ^ (cohort as u64).wrapping_mul(0x1000_0001)
            ^ (monitor_index as u64).wrapping_mul(0x0010_0001)
            ^ (step as u64).wrapping_mul(0x0000_0101);
        let loss = usize::from(uniform(seed) < risk_rate) as f64;
        if loss == 1.0 {
            events += 1;
            event_trace.push('1');
            if first_event_step == HORIZON {
                first_event_step = step;
            }
        } else {
            event_trace.push('0');
        }
        if fixed.observe(loss) == Some(true) && fixed_first_alarm_step == HORIZON {
            fixed_first_alarm_step = step;
        }
        if mix3.observe(loss) == Some(true) && mix3_first_alarm_step == HORIZON {
            mix3_first_alarm_step = step;
        }
        if mix7.observe(loss) == Some(true) && mix7_first_alarm_step == HORIZON {
            mix7_first_alarm_step = step;
        }
    }
    Ok(StreamRecord {
        cohort,
        monitor: monitor_index,
        events,
        event_trace,
        first_event_step,
        fixed_first_alarm_step,
        mix3_first_alarm_step,
        mix7_first_alarm_step,
        fixed_final_log_e: fixed.log_e_value(),
        mix3_final_log_e: mix3.log_e_value(),
        mix7_final_log_e: mix7.log_e_value(),
    })
}

fn method_fields(record: StreamRecord, method: &str) -> (usize, f64, &'static str) {
    match method {
        "fixed" => (
            record.fixed_first_alarm_step,
            record.fixed_final_log_e,
            "0.500",
        ),
        "mix3" => (
            record.mix3_first_alarm_step,
            record.mix3_final_log_e,
            "0.250;0.500;0.750",
        ),
        "mix7" => (
            record.mix7_first_alarm_step,
            record.mix7_final_log_e,
            "0.025;0.050;0.100;0.250;0.500;0.750;0.950",
        ),
        _ => unreachable!("static method name"),
    }
}

fn summary_row(
    output: &mut String,
    rate_label: &str,
    records: &[StreamRecord],
    method: &str,
    risk_rate: f64,
    log_threshold: f64,
) {
    let mut alarm_steps = Vec::new();
    let mut alarm_count = 0_usize;
    let mut event_count = 0_usize;
    for record in records {
        let (first_alarm, _, _) = method_fields(*record, method);
        alarm_count += usize::from(first_alarm < HORIZON);
        if first_alarm < HORIZON {
            alarm_steps.push(first_alarm);
        }
        event_count += record.events;
    }

    let mut fwer_count = 0_usize;
    for cohort in 0..COHORTS {
        let any_alarm = records.iter().any(|record| {
            record.cohort == cohort && method_fields(*record, method).0 < HORIZON
        });
        fwer_count += usize::from(any_alarm);
    }
    let (wilson_lower, wilson_upper) = wilson_interval(fwer_count, COHORTS);
    let grid = method_fields(records[0], method).2;
    let fields = [
        rate_label.to_string(),
        method.to_string(),
        COHORTS.to_string(),
        MONITORS.to_string(),
        HORIZON.to_string(),
        fwer_count.to_string(),
        format!("{:.9}", fwer_count as f64 / COHORTS as f64),
        format!("{wilson_lower:.9}"),
        format!("{wilson_upper:.9}"),
        alarm_count.to_string(),
        format!("{:.9}", alarm_count as f64 / (COHORTS * MONITORS) as f64),
        optional_float(mean_usize(&alarm_steps)),
        optional_usize(percentile(&alarm_steps, 95)),
        optional_usize(percentile(&alarm_steps, 50)),
        format!("{:.9}", event_count as f64 / (COHORTS * MONITORS) as f64),
        format!("{:.9}", risk_rate * HORIZON as f64),
        format!("{EPSILON:.9}"),
        format!("{DELTA:.9}"),
        format!("{log_threshold:.9}"),
        grid.to_string(),
    ];
    output.push_str(&fields.join(","));
    output.push('\n');
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            "usage: allostatic-risk-eprocess-null-monte-carlo <raw.csv> <summary.csv>",
        )
        .into());
    }
    let (fixed_threshold, mix3_threshold, mix7_threshold) = thresholds()?;
    let mut raw = format!("{RAW_HEADER}\n");
    let mut summary = format!("{SUMMARY_HEADER}\n");
    let mut all_rate_summaries: Vec<(usize, &'static str, usize, f64, f64)> = Vec::new();

    for (rate_index, risk_rate) in RISK_RATES.iter().copied().enumerate() {
        let mut records = Vec::with_capacity(COHORTS * MONITORS);
        for cohort in 0..COHORTS {
            for monitor in 0..MONITORS {
                let record = run_stream(rate_index, cohort, monitor, risk_rate)?;
                let fields = [
                    RISK_LABELS[rate_index].to_string(),
                    record.cohort.to_string(),
                    record.monitor.to_string(),
                    record.events.to_string(),
                    record.event_trace.clone(),
                    optional_usize((record.first_event_step < HORIZON).then_some(record.first_event_step)),
                    record.fixed_first_alarm_step.to_string(),
                    record.mix3_first_alarm_step.to_string(),
                    record.mix7_first_alarm_step.to_string(),
                    format!("{:.9}", record.fixed_final_log_e),
                    format!("{:.9}", record.mix3_final_log_e),
                    format!("{:.9}", record.mix7_final_log_e),
                ];
                raw.push_str(&fields.join(","));
                raw.push('\n');
                records.push(record);
            }
        }
        for (method, threshold) in [
            ("fixed", fixed_threshold),
            ("mix3", mix3_threshold),
            ("mix7", mix7_threshold),
        ] {
            summary_row(
                &mut summary,
                RISK_LABELS[rate_index],
                &records,
                method,
                risk_rate,
                threshold,
            );
            let fwer_count = (0..COHORTS)
                .filter(|cohort| {
                    records.iter().any(|record| {
                        record.cohort == *cohort && method_fields(*record, method).0 < HORIZON
                    })
                })
                .count();
            all_rate_summaries.push((
                rate_index,
                method,
                fwer_count,
                fwer_count as f64 / COHORTS as f64,
                risk_rate,
            ));
        }
    }

    let expected_raw = RISK_RATES.len() * COHORTS * MONITORS;
    let expected_summary = RISK_RATES.len() * 3;
    if all_rate_summaries.len() != expected_summary || raw.lines().count() != expected_raw + 1 {
        return Err(IoError::new(ErrorKind::InvalidData, "Monte Carlo output coverage mismatch").into());
    }
    validate_monte_carlo_gates(&all_rate_summaries)?;

    fs::write(&args[1], raw)?;
    fs::write(&args[2], summary)?;
    Ok(())
}


fn validate_monte_carlo_gates(
    rows: &[(usize, &'static str, usize, f64, f64)],
) -> Result<(), Box<dyn Error>> {
    let fwer = |rate_index: usize, method: &str| {
        rows.iter()
            .find(|row| row.0 == rate_index && row.1 == method)
            .map(|row| row.3)
            .unwrap_or(f64::NAN)
    };
    if fwer(0, "mix3") > 0.02 || fwer(0, "mix7") > 0.02 ||
       fwer(1, "mix3") > 0.08 || fwer(1, "mix7") > 0.08 ||
       fwer(2, "mix7") < fwer(2, "mix3") + 0.15 ||
       fwer(3, "mix7") < fwer(3, "mix3") + 0.20 ||
       fwer(4, "mix3") < 0.95 || fwer(4, "mix7") < 0.95 {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            "e-process Monte Carlo calibration/power gates failed",
        )
        .into());
    }
    Ok(())
}
