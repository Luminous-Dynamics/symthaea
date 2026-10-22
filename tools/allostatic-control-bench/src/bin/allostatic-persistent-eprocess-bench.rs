//! Compare one persistent e-process with per-episode resets under the same global
//! false-alarm allocation across 20 state/condition evaluation streams.
//!
//! Usage: allostatic-persistent-eprocess-bench <sequential-raw.csv> <raw.csv> <summary.csv>
//! A persistent monitor keeps log-evidence across episode boundaries. The reset
//! comparator uses summable alpha spending over 300 episodes. Both use the same
//! bounded residual loss and global delta.

use allostatic_control_bench::bounded_risk_eprocess::MixtureBoundedRiskEProcess;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::io::{Error as IoError, ErrorKind};

const HORIZON: usize = 8;
const EPISODES_PER_STREAM: usize = 300;
const TOTAL_STREAMS: usize = 20;
const RISK_BOUND: f64 = 0.15;
const RESIDUAL_SCALE: f64 = 0.05;
const GLOBAL_DELTA: f64 = 0.05;
const BET_FRACTIONS: [f64; 3] = [0.25, 0.5, 0.75];

const INPUT_HEADER: &str = "operating_point,condition,episode_seed,step,residual_score,pointwise_threshold,pointwise_step_alarm,pointwise_first_alarm_step,cusum_statistic,cusum_threshold,cusum_latched_alarm,cusum_first_alarm_step,ground_truth_outside_predictor_envelope,all_episode_scores_below_pointwise_threshold";
const RAW_HEADER: &str = "operating_point,condition,episode_seed,step,residual_score,pointwise_threshold,pointwise_step_alarm,pointwise_first_alarm_step,cusum_statistic,cusum_threshold,cusum_latched_alarm,cusum_first_alarm_step,ground_truth_outside_predictor_envelope,all_episode_scores_below_pointwise_threshold,bounded_risk_loss,episode_budget_index,persistent_log_e_value,persistent_log_threshold,persistent_alarm,persistent_first_alarm_stream_step,persistent_component_log_e_1,persistent_component_log_e_2,persistent_component_log_e_3,reset_alpha_log_e_value,reset_alpha_log_threshold,reset_alpha_alarm,reset_alpha_first_alarm_step,episode_alpha_budget,reset_component_log_e_1,reset_component_log_e_2,reset_component_log_e_3";
const SUMMARY_HEADER: &str = "operating_point,condition,episodes,persistent_alarm_stream,persistent_false_alarm_stream,persistent_missed_shift_stream,persistent_first_alarm_episode,persistent_first_alarm_step,persistent_final_log_e_value,persistent_max_log_e_value,persistent_mean_bounded_loss,reset_alpha_alarm_episodes,reset_alpha_false_alarm_episodes,reset_alpha_missed_shift_episodes,reset_alpha_mean_detection_step,reset_alpha_episodes_all_scores_below_pointwise,reset_alpha_caught_below_pointwise,reset_alpha_budget_sum,max_episode_budget_index,reset_alpha_mean_final_log_e";

const CONDITIONS: [&str; 4] = [
    "nominal",
    "gradual_shift",
    "abrupt_shift",
    "stealth_shift",
];

#[derive(Clone, Debug)]
struct InputRow {
    fields: Vec<String>,
    operating_point: String,
    condition: String,
    seed: u64,
    step: usize,
    residual_score: f64,
    pointwise_threshold: f64,
    all_scores_below_pointwise: bool,
}

impl InputRow {
    fn parse(line: &str) -> Result<Self, Box<dyn Error>> {
        let fields: Vec<String> = line.split(',').map(str::to_string).collect();
        if fields.len() != 14 {
            return Err(IoError::new(
                ErrorKind::InvalidData,
                format!("sequential input row has {} columns, expected 14", fields.len()),
            )
            .into());
        }
        let seed = fields[2].parse::<u64>()?;
        let step = fields[3].parse::<usize>()?;
        let residual_score = fields[4].parse::<f64>()?;
        let pointwise_threshold = fields[5].parse::<f64>()?;
        let all_scores_below_pointwise = match fields[13].as_str() {
            "true" => true,
            "false" => false,
            _ => {
                return Err(
                    IoError::new(ErrorKind::InvalidData, "invalid episode-wide pointwise flag")
                        .into(),
                )
            }
        };
        if !residual_score.is_finite()
            || !(0.0..=1.0).contains(&residual_score)
            || !pointwise_threshold.is_finite()
            || pointwise_threshold < 0.0
        {
            return Err(IoError::new(
                ErrorKind::InvalidData,
                "input residual or pointwise threshold is outside its valid range",
            )
            .into());
        }
        Ok(Self {
            operating_point: fields[0].clone(),
            condition: fields[1].clone(),
            seed,
            step,
            residual_score,
            pointwise_threshold,
            all_scores_below_pointwise,
            fields,
        })
    }
}

#[derive(Clone, Debug, Default)]
struct StreamSummary {
    episodes: usize,
    persistent_final_log_e_value: f64,
    persistent_max_log_e_value: f64,
    persistent_first_alarm_stream_step: Option<usize>,
    persistent_mean_bounded_loss: f64,
    reset_alpha_alarm_episodes: usize,
    reset_alpha_false_alarm_episodes: usize,
    reset_alpha_missed_shift_episodes: usize,
    reset_alpha_detection_steps: Vec<usize>,
    reset_alpha_episodes_all_scores_below_pointwise: usize,
    reset_alpha_caught_below_pointwise: usize,
    reset_alpha_budget_sum: f64,
    reset_alpha_final_log_e_values: Vec<f64>,
}

fn optional_float(value: Option<f64>) -> String {
    value.map_or_else(|| "NA".to_string(), |number| format!("{number:.9}"))
}

fn optional_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "NA".to_string(), |number| number.to_string())
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn mean_steps(values: &[usize]) -> Option<f64> {
    mean(&values.iter().map(|step| *step as f64).collect::<Vec<_>>())
}

fn summary_row(
    output: &mut String,
    operating_point: &str,
    condition: &str,
    summary: &StreamSummary,
) {
    let persistent_alarmed = summary.persistent_first_alarm_stream_step.is_some();
    let shifted = condition != "nominal";
    let persistent_false_alarm = !shifted && persistent_alarmed;
    let persistent_missed_shift = shifted && !persistent_alarmed;
    let fields = [
        operating_point.to_string(),
        condition.to_string(),
        summary.episodes.to_string(),
        persistent_alarmed.to_string(),
        persistent_false_alarm.to_string(),
        persistent_missed_shift.to_string(),
        optional_usize(
            summary
                .persistent_first_alarm_stream_step
                .map(|step| step / HORIZON),
        ),
        optional_usize(
            summary
                .persistent_first_alarm_stream_step
                .map(|step| step % HORIZON),
        ),
        format!("{:.9}", summary.persistent_final_log_e_value),
        format!("{:.9}", summary.persistent_max_log_e_value),
        format!(
            "{:.9}",
            summary.persistent_mean_bounded_loss / (summary.episodes * HORIZON) as f64
        ),
        summary.reset_alpha_alarm_episodes.to_string(),
        summary.reset_alpha_false_alarm_episodes.to_string(),
        summary.reset_alpha_missed_shift_episodes.to_string(),
        optional_float(mean_steps(&summary.reset_alpha_detection_steps)),
        summary
            .reset_alpha_episodes_all_scores_below_pointwise
            .to_string(),
        summary.reset_alpha_caught_below_pointwise.to_string(),
        format!("{:.12}", summary.reset_alpha_budget_sum),
        summary.episodes.to_string(),
        optional_float(mean(&summary.reset_alpha_final_log_e_values)),
    ];
    output.push_str(&fields.join(","));
    output.push('\n');
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            "usage: allostatic-persistent-eprocess-bench <sequential-raw.csv> <raw.csv> <summary.csv>",
        )
        .into());
    }
    let input = fs::read_to_string(&args[1])?;
    let mut lines = input.lines();
    if lines.next() != Some(INPUT_HEADER) {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            "input header does not match the pinned sequential-scope schema",
        )
        .into());
    }
    let rows = lines
        .map(InputRow::parse)
        .collect::<Result<Vec<_>, _>>()?;
    if rows.len() != 48_000 {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            format!("expected 48,000 input step rows, received {}", rows.len()),
        )
        .into());
    }

    let mut stream_indices: BTreeMap<(String, String), Vec<usize>> = BTreeMap::new();
    for (index, row) in rows.iter().enumerate() {
        stream_indices
            .entry((row.operating_point.clone(), row.condition.clone()))
            .or_default()
            .push(index);
    }
    if stream_indices.len() != TOTAL_STREAMS {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            "input does not contain exactly 20 state/condition streams",
        )
        .into());
    }

    let persistent_threshold = (TOTAL_STREAMS as f64 / GLOBAL_DELTA).ln();
    let mut output_rows: Vec<Vec<String>> = vec![Vec::new(); rows.len()];
    let mut summaries = BTreeMap::new();

    for ((operating_point, condition), indices) in stream_indices {
        let expected_stream_rows = EPISODES_PER_STREAM * HORIZON;
        if indices.len() != expected_stream_rows {
            return Err(IoError::new(
                ErrorKind::InvalidData,
                format!("stream {operating_point}/{condition} has {} rows, expected {expected_stream_rows}", indices.len()),
            )
            .into());
        }

        let mut persistent = MixtureBoundedRiskEProcess::new(
            RISK_BOUND,
            BET_FRACTIONS,
            GLOBAL_DELTA,
            TOTAL_STREAMS,
        )
        .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "invalid persistent e-process"))?;

        let mut summary = StreamSummary::default();
        let mut first_persistent_alarm = None;
        let mut stream_loss_sum = 0.0_f64;
        let mut stream_max_log_e = 0.0_f64;

        for episode in 0..EPISODES_PER_STREAM {
            let episode_indices = &indices[episode * HORIZON..(episode + 1) * HORIZON];
            let mut reset_alpha = MixtureBoundedRiskEProcess::new_with_alpha_spending(
                RISK_BOUND,
                BET_FRACTIONS,
                GLOBAL_DELTA,
                TOTAL_STREAMS,
                episode as u64 + 1,
            )
            .ok_or_else(|| {
                IoError::new(ErrorKind::InvalidInput, "invalid reset alpha-spending process")
            })?;
            let mut first_reset_alarm_step = HORIZON;
            let mut all_scores_below_pointwise = true;
            let mut episode_rows = Vec::with_capacity(HORIZON);
            let expected_seed = 50_000 + episode as u64;

            for (step, index) in episode_indices.iter().copied().enumerate() {
                let row = &rows[index];
                if row.operating_point != operating_point
                    || row.condition != condition
                    || row.seed != expected_seed
                    || row.step != step
                {
                    return Err(IoError::new(
                        ErrorKind::InvalidData,
                        "stream row order must be 300 chronological eight-step episodes",
                    )
                    .into());
                }

                let bounded_loss = (row.residual_score / RESIDUAL_SCALE).clamp(0.0, 1.0);
                let persistent_alarm = persistent.observe(bounded_loss).ok_or_else(|| {
                    IoError::new(ErrorKind::InvalidData, "persistent e-process rejected loss")
                })?;
                let persistent_log_e_value = persistent.log_e_value();
                stream_loss_sum += bounded_loss;
                stream_max_log_e = stream_max_log_e.max(persistent_log_e_value);

                if persistent_alarm && first_persistent_alarm.is_none() {
                    first_persistent_alarm = Some(episode * HORIZON + step);
                }

                let reset_alarm = reset_alpha.observe(bounded_loss).ok_or_else(|| {
                    IoError::new(ErrorKind::InvalidData, "reset alpha monitor rejected loss")
                })?;
                if reset_alarm && first_reset_alarm_step == HORIZON {
                    first_reset_alarm_step = step;
                }
                all_scores_below_pointwise &= row.residual_score <= row.pointwise_threshold;

                let episode_budget_index = episode as u64 + 1;
                let episode_alpha_budget =
                    GLOBAL_DELTA / (TOTAL_STREAMS as f64
                        * episode_budget_index as f64
                        * (episode_budget_index + 1) as f64);
                let mut fields = row.fields.clone();
                fields.push(format!("{bounded_loss:.9}"));
                fields.push(episode_budget_index.to_string());
                fields.push(format!("{persistent_log_e_value:.9}"));
                fields.push(format!("{persistent_threshold:.9}"));
                fields.push(persistent_alarm.to_string());
                fields.push("NA".to_string());
                for component_log in persistent.component_log_e_values() {
                    fields.push(format!("{component_log:.9}"));
                }
                fields.push(format!("{:.9}", reset_alpha.log_e_value()));
                fields.push(format!("{:.9}", reset_alpha.log_alarm_threshold()));
                fields.push(reset_alarm.to_string());
                fields.push(first_reset_alarm_step.to_string());
                fields.push(format!("{episode_alpha_budget:.12}"));
                for component_log in reset_alpha.component_log_e_values() {
                    fields.push(format!("{component_log:.9}"));
                }
                output_rows[index] = fields;
                episode_rows.push(index);
            }

            for index in episode_rows {
                output_rows[index][19] = optional_usize(first_persistent_alarm);
                output_rows[index][26] = first_reset_alarm_step.to_string();
            }

            let reset_detected = first_reset_alarm_step < HORIZON;
            summary.episodes += 1;
            summary.reset_alpha_alarm_episodes += usize::from(reset_detected);
            summary.reset_alpha_false_alarm_episodes +=
                usize::from(condition == "nominal" && reset_detected);
            summary.reset_alpha_missed_shift_episodes +=
                usize::from(condition != "nominal" && !reset_detected);
            if reset_detected {
                summary.reset_alpha_detection_steps.push(first_reset_alarm_step);
            }
            summary.reset_alpha_episodes_all_scores_below_pointwise +=
                usize::from(all_scores_below_pointwise);
            summary.reset_alpha_caught_below_pointwise +=
                usize::from(all_scores_below_pointwise && reset_detected);
            summary.reset_alpha_budget_sum +=
                GLOBAL_DELTA / (TOTAL_STREAMS as f64
                    * (episode as f64 + 1.0)
                    * (episode as f64 + 2.0));
            summary.reset_alpha_final_log_e_values.push(reset_alpha.log_e_value());
        }

        summary.persistent_final_log_e_value = persistent.log_e_value();
        summary.persistent_max_log_e_value = stream_max_log_e;
        summary.persistent_first_alarm_stream_step = first_persistent_alarm;
        summary.persistent_mean_bounded_loss = stream_loss_sum;
        summaries.insert((operating_point, condition), summary);
    }

    if output_rows.iter().any(Vec::is_empty)
        || summaries.len() != TOTAL_STREAMS
        || summaries
            .values()
            .any(|summary| summary.episodes != EPISODES_PER_STREAM)
    {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            "persistent/reset benchmark did not cover every stream and row",
        )
        .into());
    }

    let mut raw_output = format!("{RAW_HEADER}\n");
    for fields in output_rows {
        raw_output.push_str(&fields.join(","));
        raw_output.push('\n');
    }
    let mut summary_output = format!("{SUMMARY_HEADER}\n");
    for ((operating_point, condition), summary) in summaries {
        summary_row(&mut summary_output, &operating_point, &condition, &summary);
    }
    fs::write(&args[2], raw_output)?;
    fs::write(&args[3], summary_output)?;
    Ok(())
}
