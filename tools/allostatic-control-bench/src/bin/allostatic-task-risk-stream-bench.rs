//! Persistent bounded-risk monitoring of observed false-safe prediction episodes.
//!
//! Usage: allostatic-task-risk-stream-bench <task-scope-raw.csv> <raw-out.csv> <summary-out.csv>
//! Unlike the residual e-process benchmark, this monitor consumes a binary task
//! outcome: an episode is positive if any predicted-safe transition was observed
//! to violate the configured reserve/load limits. Evidence persists across 300
//! episodes in each operating-point stream.

use allostatic_control_bench::bounded_risk_eprocess::{
    BoundedRiskEProcess, MixtureBoundedRiskEProcess,
};
use std::error::Error;
use std::fs;
use std::io::{Error as IoError, ErrorKind};

const HORIZON: usize = 8;
const EPISODES: usize = 300;
const RISK_BOUND: f64 = 0.05;
const GLOBAL_DELTA: f64 = 0.05;
const PARALLEL_MONITORS: usize = 5;
const BET_FRACTIONS: [f64; 3] = [0.25, 0.50, 0.75];
const FIXED_BET_FRACTION: f64 = 0.50;
const FIXED_BET: f64 = FIXED_BET_FRACTION / RISK_BOUND;

const INPUT_HEADER: &str = "operating_point,condition,episode_index,feature_score,feature_threshold,feature_alarm,scope_score,scope_threshold,scope_alarm,combined_alarm,outside_predictor_envelope,first_feature_alarm_step,first_scope_alarm_step,first_combined_alarm_step,first_false_safe_step,feature_flagged_by_false_safe_step,scope_flagged_by_false_safe_step,combined_flagged_by_false_safe_step,scope_false_safe_override";
const RAW_HEADER: &str = "operating_point,condition,episode_index,feature_score,feature_threshold,feature_alarm,scope_score,scope_threshold,scope_alarm,combined_alarm,outside_predictor_envelope,first_feature_alarm_step,first_scope_alarm_step,first_combined_alarm_step,first_false_safe_step,feature_flagged_by_false_safe_step,scope_flagged_by_false_safe_step,combined_flagged_by_false_safe_step,scope_false_safe_override,observed_false_safe_event,mixture_log_e,mixture_component_log_e_1,mixture_component_log_e_2,mixture_component_log_e_3,mixture_latched_alarm,mixture_first_alarm_episode,fixed_log_e,fixed_latched_alarm,fixed_first_alarm_episode,log_alarm_threshold,risk_bound,bet_fraction_1,bet_fraction_2,bet_fraction_3";
const SUMMARY_HEADER: &str = "operating_point,condition,episodes,false_safe_events,false_safe_event_rate,mixture_alarm,mixture_first_alarm_episode,fixed_alarm,fixed_first_alarm_episode,first_false_safe_episode,mixture_delay_episodes,fixed_delay_episodes,mixture_events_at_alarm,fixed_events_at_alarm,mixture_final_log_e,fixed_final_log_e,log_alarm_threshold,risk_bound,global_delta,parallel_monitors";

const OPERATING_POINTS: [&str; 5] = [
    "critical_low_reserve_high_load",
    "reserve_limited",
    "load_limited",
    "balanced",
    "high_margin",
];
const CONDITIONS: [&str; 4] = [
    "nominal",
    "moderate_shift",
    "strong_shift",
    "stealth_shift",
];

#[derive(Clone, Debug)]
struct InputRow {
    fields: Vec<String>,
    operating_point: String,
    condition: String,
    episode_index: usize,
    first_false_safe_step: usize,
    scope_false_safe_override: bool,
    outside_predictor_envelope: bool,
}

impl InputRow {
    fn parse(line: &str) -> Result<Self, Box<dyn Error>> {
        let fields: Vec<String> = line.split(',').map(str::to_string).collect();
        if fields.len() != 19 {
            return Err(IoError::new(
                ErrorKind::InvalidData,
                format!("task-scope row has {} columns, expected 19", fields.len()),
            )
            .into());
        }
        let episode_index = fields[2].parse::<usize>()?;
        let first_false_safe_step = fields[14].parse::<usize>()?;
        if first_false_safe_step > HORIZON {
            return Err(IoError::new(
                ErrorKind::InvalidData,
                "first false-safe step is outside the episode horizon",
            )
            .into());
        }
        let parse_bool = |value: &str, label: &str| -> Result<bool, Box<dyn Error>> {
            match value {
                "true" => Ok(true),
                "false" => Ok(false),
                _ => Err(IoError::new(
                    ErrorKind::InvalidData,
                    format!("invalid boolean in {label}"),
                )
                .into()),
            }
        };
        Ok(Self {
            operating_point: fields[0].clone(),
            condition: fields[1].clone(),
            episode_index,
            first_false_safe_step,
            scope_false_safe_override: parse_bool(&fields[18], "scope false-safe override")?,
            outside_predictor_envelope: parse_bool(&fields[10], "predictor-envelope label")?,
            fields,
        })
    }

    fn observed_false_safe_event(&self) -> bool {
        self.first_false_safe_step < HORIZON
    }
}

#[derive(Clone, Debug, Default)]
struct StreamSummary {
    false_safe_events: usize,
    first_false_safe_episode: Option<usize>,
    mixture_first_alarm_episode: Option<usize>,
    fixed_first_alarm_episode: Option<usize>,
    mixture_events_at_alarm: Option<usize>,
    fixed_events_at_alarm: Option<usize>,
    final_mixture_log_e: f64,
    final_fixed_log_e: f64,
}

fn optional_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "NA".to_string(), |number| number.to_string())
}

fn summary_row(
    output: &mut String,
    operating_point: &str,
    condition: &str,
    summary: &StreamSummary,
    log_threshold: f64,
) {
    let mixture_alarm = summary.mixture_first_alarm_episode.is_some();
    let fixed_alarm = summary.fixed_first_alarm_episode.is_some();
    let mixture_delay = summary
        .mixture_first_alarm_episode
        .zip(summary.first_false_safe_episode)
        .map(|(alarm, event)| alarm.saturating_sub(event));
    let fixed_delay = summary
        .fixed_first_alarm_episode
        .zip(summary.first_false_safe_episode)
        .map(|(alarm, event)| alarm.saturating_sub(event));
    let fields = [
        operating_point.to_string(),
        condition.to_string(),
        EPISODES.to_string(),
        summary.false_safe_events.to_string(),
        format!("{:.9}", summary.false_safe_events as f64 / EPISODES as f64),
        if mixture_alarm { "true" } else { "false" }.to_string(),
        optional_usize(summary.mixture_first_alarm_episode),
        if fixed_alarm { "true" } else { "false" }.to_string(),
        optional_usize(summary.fixed_first_alarm_episode),
        optional_usize(summary.first_false_safe_episode),
        optional_usize(mixture_delay),
        optional_usize(fixed_delay),
        optional_usize(summary.mixture_events_at_alarm),
        optional_usize(summary.fixed_events_at_alarm),
        format!("{:.9}", summary.final_mixture_log_e),
        format!("{:.9}", summary.final_fixed_log_e),
        format!("{log_threshold:.9}"),
        format!("{RISK_BOUND:.9}"),
        format!("{GLOBAL_DELTA:.9}"),
        PARALLEL_MONITORS.to_string(),
    ];
    output.push_str(&fields.join(","));
    output.push('\n');
}

fn process_stream(
    rows: &[InputRow],
    expected_operating_point: &str,
    expected_condition: &str,
    raw_output: &mut String,
    summary_output: &mut String,
    log_threshold: f64,
) -> Result<(), Box<dyn Error>> {
    if rows.len() != EPISODES {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            format!("stream has {} episodes, expected {EPISODES}", rows.len()),
        )
        .into());
    }
    let first = &rows[0];
    if first.operating_point != expected_operating_point
        || first.condition != expected_condition
    {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            "task-risk stream order disagrees with the declared operating point/condition panel",
        )
        .into());
    }
    let mut mixture = MixtureBoundedRiskEProcess::new(
        RISK_BOUND,
        BET_FRACTIONS,
        GLOBAL_DELTA,
        PARALLEL_MONITORS,
    )
    .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "invalid mixture e-process config"))?;
    let mut fixed = BoundedRiskEProcess::new(
        RISK_BOUND,
        FIXED_BET,
        GLOBAL_DELTA,
        PARALLEL_MONITORS,
    )
    .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "invalid fixed e-process config"))?;

    let mut summary = StreamSummary::default();
    let mut mixture_alarms = Vec::with_capacity(EPISODES);
    let mut fixed_alarms = Vec::with_capacity(EPISODES);
    let mut mixture_logs = Vec::with_capacity(EPISODES);
    let mut mixture_component_logs = Vec::with_capacity(EPISODES);
    let mut fixed_logs = Vec::with_capacity(EPISODES);
    let mut events_seen = 0_usize;

    for (episode, row) in rows.iter().enumerate() {
        if row.operating_point != first.operating_point
            || row.condition != first.condition
            || row.episode_index != episode
            || !OPERATING_POINTS.contains(&row.operating_point.as_str())
            || !CONDITIONS.contains(&row.condition.as_str())
        {
            return Err(IoError::new(
                ErrorKind::InvalidData,
                "task-risk rows are not in the expected unique episode-index order",
            )
            .into());
        }
        let event = row.observed_false_safe_event();
        if event != row.scope_false_safe_override
            || row.outside_predictor_envelope != (row.condition != "nominal")
        {
            return Err(IoError::new(
                ErrorKind::InvalidData,
                "false-safe label or ground-truth envelope condition is inconsistent",
            )
            .into());
        }
        if event {
            summary.false_safe_events += 1;
            summary.first_false_safe_episode.get_or_insert(episode);
        }
        events_seen += usize::from(event);
        let loss = if event { 1.0 } else { 0.0 };
        let mixture_alarm = mixture.observe(loss).ok_or_else(|| {
            IoError::new(ErrorKind::InvalidData, "mixture e-process rejected binary event")
        })?;
        let fixed_alarm = fixed.observe(loss).ok_or_else(|| {
            IoError::new(ErrorKind::InvalidData, "fixed e-process rejected binary event")
        })?;
        if mixture_alarm && summary.mixture_first_alarm_episode.is_none() {
            summary.mixture_first_alarm_episode = Some(episode);
            summary.mixture_events_at_alarm = Some(events_seen);
        }
        if fixed_alarm && summary.fixed_first_alarm_episode.is_none() {
            summary.fixed_first_alarm_episode = Some(episode);
            summary.fixed_events_at_alarm = Some(events_seen);
        }
        mixture_alarms.push(mixture_alarm);
        fixed_alarms.push(fixed_alarm);
        mixture_logs.push(mixture.log_e_value());
        mixture_component_logs.push(mixture.component_log_e_values());
        fixed_logs.push(fixed.log_e_value());
    }
    summary.final_mixture_log_e = mixture.log_e_value();
    summary.final_fixed_log_e = fixed.log_e_value();

    for (episode, row) in rows.iter().enumerate() {
        let mut fields = row.fields.clone();
        fields.push(if row.observed_false_safe_event() { "true" } else { "false" }.to_string());
        fields.push(format!("{:.9}", mixture_logs[episode]));
        for component_log_e in mixture_component_logs[episode] {
            fields.push(format!("{component_log_e:.9}"));
        }
        fields.push(if mixture_alarms[episode] { "true" } else { "false" }.to_string());
        fields.push(optional_usize(summary.mixture_first_alarm_episode));
        fields.push(format!("{:.9}", fixed_logs[episode]));
        fields.push(if fixed_alarms[episode] { "true" } else { "false" }.to_string());
        fields.push(optional_usize(summary.fixed_first_alarm_episode));
        fields.push(format!("{log_threshold:.9}"));
        fields.push(format!("{RISK_BOUND:.9}"));
        fields.push(format!("{:.9}", BET_FRACTIONS[0]));
        fields.push(format!("{:.9}", BET_FRACTIONS[1]));
        fields.push(format!("{:.9}", BET_FRACTIONS[2]));
        raw_output.push_str(&fields.join(","));
        raw_output.push('\n');
    }

    summary_row(
        summary_output,
        &first.operating_point,
        &first.condition,
        &summary,
        log_threshold,
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            "usage: allostatic-task-risk-stream-bench <task-scope-raw.csv> <raw-out.csv> <summary-out.csv>",
        )
        .into());
    }
    let input = fs::read_to_string(&args[1])?;
    let mut lines = input.lines();
    let header = lines
        .next()
        .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "missing task-scope input header"))?;
    if header != INPUT_HEADER {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            "task-scope input schema does not match the pinned benchmark",
        )
        .into());
    }
    let rows = lines
        .map(InputRow::parse)
        .collect::<Result<Vec<_>, _>>()?;
    let expected_rows = OPERATING_POINTS.len() * CONDITIONS.len() * EPISODES;
    if rows.len() != expected_rows || rows.len() != 6_000 {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            format!("expected 6,000 task-scope episode rows, received {}", rows.len()),
        )
        .into());
    }

    let configuration = MixtureBoundedRiskEProcess::new(
        RISK_BOUND,
        BET_FRACTIONS,
        GLOBAL_DELTA,
        PARALLEL_MONITORS,
    )
    .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "invalid e-process configuration"))?;
    let log_threshold = configuration.log_alarm_threshold();
    let mut raw_output = format!("{RAW_HEADER}\n");
    let mut summary_output = format!("{SUMMARY_HEADER}\n");
    let mut row_index = 0_usize;

    for operating_point in OPERATING_POINTS {
        for condition in CONDITIONS {
            let end = row_index + EPISODES;
            process_stream(
                &rows[row_index..end],
                operating_point,
                condition,
                &mut raw_output,
                &mut summary_output,
                log_threshold,
            )?;
            row_index = end;
        }
    }
    if row_index != rows.len() {
        return Err(IoError::new(ErrorKind::InvalidData, "unconsumed task-scope rows").into());
    }
    fs::write(&args[2], raw_output)?;
    fs::write(&args[3], summary_output)?;
    Ok(())
}
