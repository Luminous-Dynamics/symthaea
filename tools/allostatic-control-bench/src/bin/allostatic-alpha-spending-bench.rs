//! Compare per-episode e-process resets with lifetime alpha-spending resets.
//!
//! Usage: allostatic-alpha-spending-bench <sequential-raw.csv> <raw.csv> <summary.csv>
//! It consumes the exact CUSUM trace and interleaves conditions by episode index,
//! so spending ranks are not correlated with synthetic condition labels.

use allostatic_control_bench::bounded_risk_eprocess::MixtureBoundedRiskEProcess;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::io::{Error as IoError, ErrorKind};

const HORIZON: usize = 8;
const EPISODES_PER_CONDITION: usize = 300;
const CONDITIONS_PER_STATE: usize = 4;
const TOTAL_CASES_PER_STATE: u64 =
    (EPISODES_PER_CONDITION * CONDITIONS_PER_STATE) as u64;
const BUDGET_HORIZON: u64 = EPISODES_PER_CONDITION as u64;
const CALIBRATED_RESIDUAL_SCALE: f64 = 0.05;
const RISK_BOUND: f64 = 0.15;
const GLOBAL_DELTA: f64 = 0.05;
const PARALLEL_MONITORS: usize = 5;
const BET_FRACTIONS: [f64; 3] = [0.25, 0.5, 0.75];

const INPUT_HEADER: &str = "operating_point,condition,episode_seed,step,residual_score,pointwise_threshold,pointwise_step_alarm,pointwise_first_alarm_step,cusum_statistic,cusum_threshold,cusum_latched_alarm,cusum_first_alarm_step,ground_truth_outside_predictor_envelope,all_episode_scores_below_pointwise_threshold";
const RAW_HEADER: &str = "operating_point,condition,episode_seed,step,residual_score,pointwise_threshold,pointwise_step_alarm,pointwise_first_alarm_step,cusum_statistic,cusum_threshold,cusum_latched_alarm,cusum_first_alarm_step,ground_truth_outside_predictor_envelope,all_episode_scores_below_pointwise_threshold,bounded_risk_loss,episode_budget_index,episode_false_alarm_budget,spending_log_e_value,spending_log_threshold,spending_alarm,spending_first_alarm_step,reset_log_e_value,reset_log_threshold,reset_alarm,reset_first_alarm_step,spending_component_log_e_1,spending_component_log_e_2,spending_component_log_e_3,reset_component_log_e_1,reset_component_log_e_2,reset_component_log_e_3";
const SUMMARY_HEADER: &str = "operating_point,condition,episodes,spending_alarms,spending_false_alarms,spending_missed_shifts,reset_alarms,reset_false_alarms,reset_missed_shifts,spending_mean_detection_step,reset_mean_detection_step,episodes_all_scores_below_pointwise,spending_caught_below_pointwise,reset_caught_below_pointwise,alpha_budget_spent_sum,max_episode_budget_index,mean_final_spending_log_e,max_final_spending_log_e,mean_final_reset_log_e";

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
        let residual_score = fields[4].parse::<f64>()?;
        let pointwise_threshold = fields[5].parse::<f64>()?;
        let seed = fields[2].parse::<u64>()?;
        let step = fields[3].parse::<usize>()?;
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
struct Cell {
    episodes: usize,
    spending_alarms: usize,
    spending_false_alarms: usize,
    spending_missed_shifts: usize,
    reset_alarms: usize,
    reset_false_alarms: usize,
    reset_missed_shifts: usize,
    spending_detection_steps: Vec<usize>,
    reset_detection_steps: Vec<usize>,
    all_scores_below_pointwise: usize,
    spending_caught_below_pointwise: usize,
    reset_caught_below_pointwise: usize,
    alpha_budget_spent_sum: f64,
    max_episode_budget_index: u64,
    final_spending_log_e_values: Vec<f64>,
    final_reset_log_e_values: Vec<f64>,
}

impl Cell {
    #[allow(clippy::too_many_arguments)]
    fn observe(
        &mut self,
        condition: &str,
        spending_first_alarm: usize,
        reset_first_alarm: usize,
        all_below_pointwise: bool,
        episode_budget_index: u64,
        spending_log_e_value: f64,
        reset_log_e_value: f64,
    ) {
        let spending_detected = spending_first_alarm < HORIZON;
        let reset_detected = reset_first_alarm < HORIZON;
        let shifted = condition != "nominal";

        self.episodes += 1;
        self.spending_alarms += usize::from(spending_detected);
        self.reset_alarms += usize::from(reset_detected);
        self.spending_false_alarms += usize::from(!shifted && spending_detected);
        self.reset_false_alarms += usize::from(!shifted && reset_detected);
        self.spending_missed_shifts += usize::from(shifted && !spending_detected);
        self.reset_missed_shifts += usize::from(shifted && !reset_detected);
        if spending_detected {
            self.spending_detection_steps.push(spending_first_alarm);
        }
        if reset_detected {
            self.reset_detection_steps.push(reset_first_alarm);
        }
        self.all_scores_below_pointwise += usize::from(all_below_pointwise);
        self.spending_caught_below_pointwise +=
            usize::from(all_below_pointwise && spending_detected);
        self.reset_caught_below_pointwise +=
            usize::from(all_below_pointwise && reset_detected);
        self.alpha_budget_spent_sum +=
            GLOBAL_DELTA / (PARALLEL_MONITORS as f64
                * episode_budget_index as f64
                * (episode_budget_index + 1) as f64);
        self.max_episode_budget_index = self.max_episode_budget_index.max(episode_budget_index);
        self.final_spending_log_e_values.push(spending_log_e_value);
        self.final_reset_log_e_values.push(reset_log_e_value);
    }
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn mean_steps(values: &[usize]) -> Option<f64> {
    mean(&values.iter().map(|step| *step as f64).collect::<Vec<_>>())
}

fn optional_float(value: Option<f64>) -> String {
    value.map_or_else(|| "NA".to_string(), |number| format!("{number:.9}"))
}

fn summary_row(
    output: &mut String,
    operating_point: &str,
    condition: &str,
    cell: &Cell,
) {
    let fields = [
        operating_point.to_string(),
        condition.to_string(),
        cell.episodes.to_string(),
        cell.spending_alarms.to_string(),
        cell.spending_false_alarms.to_string(),
        cell.spending_missed_shifts.to_string(),
        cell.reset_alarms.to_string(),
        cell.reset_false_alarms.to_string(),
        cell.reset_missed_shifts.to_string(),
        optional_float(mean_steps(&cell.spending_detection_steps)),
        optional_float(mean_steps(&cell.reset_detection_steps)),
        cell.all_scores_below_pointwise.to_string(),
        cell.spending_caught_below_pointwise.to_string(),
        cell.reset_caught_below_pointwise.to_string(),
        format!("{:.12}", cell.alpha_budget_spent_sum),
        cell.max_episode_budget_index.to_string(),
        optional_float(mean(&cell.final_spending_log_e_values)),
        optional_float(cell.final_spending_log_e_values.iter().copied().reduce(f64::max)),
        optional_float(mean(&cell.final_reset_log_e_values)),
    ];
    output.push_str(&fields.join(","));
    output.push('\n');
}

fn process_episode(
    rows: &[InputRow],
    episode_budget_index: u64,
    raw_output: &mut String,
    cells: &mut BTreeMap<(String, String), Cell>,
) -> Result<(), Box<dyn Error>> {
    if rows.len() != HORIZON {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            format!("episode contains {} rows; expected {HORIZON}", rows.len()),
        )
        .into());
    }
    let first = &rows[0];
    let spending_threshold = allostatic_control_bench::bounded_risk_eprocess::
        alpha_spending_log_threshold(GLOBAL_DELTA, PARALLEL_MONITORS, episode_budget_index)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "invalid episode alpha allocation"))?;
    let mut spending_monitor = MixtureBoundedRiskEProcess::new_with_alpha_spending(
        RISK_BOUND,
        BET_FRACTIONS,
        GLOBAL_DELTA,
        PARALLEL_MONITORS,
        episode_budget_index,
    )
    .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "invalid alpha-spending monitor"))?;
    let mut reset_monitor = MixtureBoundedRiskEProcess::new(
        RISK_BOUND,
        BET_FRACTIONS,
        GLOBAL_DELTA,
        PARALLEL_MONITORS,
    )
    .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "invalid per-episode monitor"))?;

    let mut spending_first_alarm = HORIZON;
    let mut reset_first_alarm = HORIZON;
    let mut all_scores_below_pointwise = true;
    let mut output_rows = Vec::with_capacity(HORIZON);

    for (expected_step, row) in rows.iter().enumerate() {
        if row.operating_point != first.operating_point
            || row.condition != first.condition
            || row.seed != first.seed
            || row.step != expected_step
        {
            return Err(IoError::new(
                ErrorKind::InvalidData,
                "episode rows are not contiguous or do not share their state/condition/seed",
            )
            .into());
        }

        let bounded_loss = (row.residual_score / CALIBRATED_RESIDUAL_SCALE).clamp(0.0, 1.0);
        let spending_alarm = spending_monitor.observe(bounded_loss).ok_or_else(|| {
            IoError::new(ErrorKind::InvalidData, "alpha-spending monitor rejected bounded loss")
        })?;
        let reset_alarm = reset_monitor.observe(bounded_loss).ok_or_else(|| {
            IoError::new(ErrorKind::InvalidData, "per-episode monitor rejected bounded loss")
        })?;
        if spending_alarm && spending_first_alarm == HORIZON {
            spending_first_alarm = expected_step;
        }
        if reset_alarm && reset_first_alarm == HORIZON {
            reset_first_alarm = expected_step;
        }
        all_scores_below_pointwise &= row.residual_score <= row.pointwise_threshold;
        output_rows.push((
            row,
            bounded_loss,
            spending_monitor.log_e_value(),
            spending_alarm,
            reset_monitor.log_e_value(),
            reset_alarm,
            spending_monitor.component_log_e_values(),
            reset_monitor.component_log_e_values(),
        ));
    }

    if rows
        .iter()
        .any(|row| row.all_scores_below_pointwise != all_scores_below_pointwise)
    {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            "source episode-wide pointwise flags disagree with recomputed scores",
        )
        .into());
    }

    for (
        row,
        bounded_loss,
        spending_log_e,
        spending_alarm,
        reset_log_e,
        reset_alarm,
        spending_components,
        reset_components,
    ) in &output_rows
    {
        let mut fields = row.fields.clone();
        fields.push(format!("{bounded_loss:.9}"));
        fields.push(episode_budget_index.to_string());
        fields.push(format!("{:.12}", GLOBAL_DELTA / (
            PARALLEL_MONITORS as f64 * episode_budget_index as f64
                * (episode_budget_index + 1) as f64
        )));
        fields.push(format!("{spending_log_e:.9}"));
        fields.push(format!("{spending_threshold:.9}"));
        fields.push(if *spending_alarm { "true" } else { "false" }.to_string());
        fields.push(spending_first_alarm.to_string());
        fields.push(format!("{reset_log_e:.9}"));
        fields.push(format!("{:.9}", (PARALLEL_MONITORS as f64 / GLOBAL_DELTA).ln()));
        fields.push(if *reset_alarm { "true" } else { "false" }.to_string());
        fields.push(reset_first_alarm.to_string());
        for component_log in spending_components {
            fields.push(format!("{component_log:.9}"));
        }
        for component_log in reset_components {
            fields.push(format!("{component_log:.9}"));
        }
        raw_output.push_str(&fields.join(","));
        raw_output.push('\n');
    }

    cells
        .entry((first.operating_point.clone(), first.condition.clone()))
        .or_default()
        .observe(
            &first.condition,
            spending_first_alarm,
            reset_first_alarm,
            all_scores_below_pointwise,
            episode_budget_index,
            spending_monitor.log_e_value(),
            reset_monitor.log_e_value(),
        );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            "usage: allostatic-alpha-spending-bench <sequential-raw.csv> <raw.csv> <summary.csv>",
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

    let mut raw_output = format!("{RAW_HEADER}\n");
    let mut cells: BTreeMap<(String, String), Cell> = BTreeMap::new();
    let mut episode_indices: BTreeMap<String, u64> = BTreeMap::new();
    let mut expected_state: Option<String> = None;
    let mut current_seed: Option<u64> = None;

    for chunk in rows.chunks_exact(HORIZON) {
        let first = &chunk[0];
        if expected_state.as_deref() != Some(first.operating_point.as_str()) {
            if episode_indices
                .get(&first.operating_point)
                .copied()
                .unwrap_or(0)
                != 0
            {
                return Err(IoError::new(
                    ErrorKind::InvalidData,
                    "operating points are not contiguous in input",
                )
                .into());
            }
            expected_state = Some(first.operating_point.clone());
            current_seed = None;
        }

        let ordinal = episode_indices.entry(first.operating_point.clone()).or_default();
        *ordinal += 1;
        if *ordinal > TOTAL_CASES_PER_STATE {
            return Err(IoError::new(
                ErrorKind::InvalidData,
                "too many episodes for a state; alpha allocation panel exceeded",
            )
            .into());
        }
        let condition_position = ((*ordinal - 1) % CONDITIONS_PER_STATE as u64) as usize;
        let seed_position = (*ordinal - 1) / CONDITIONS_PER_STATE as u64;
        let expected_condition = CONDITIONS[condition_position];
        let expected_seed = 50_000 + seed_position;
        if first.condition != expected_condition || first.seed != expected_seed {
            return Err(IoError::new(
                ErrorKind::InvalidData,
                "input must interleave all four conditions for each episode seed",
            )
            .into());
        }

        if current_seed.is_some_and(|seed| first.seed < seed) {
            return Err(IoError::new(
                ErrorKind::InvalidData,
                "episode seed moved backwards within an operating point",
            )
            .into());
        }
        current_seed = Some(first.seed);
        let budget_index = first.seed.checked_sub(49_999).ok_or_else(|| {
            IoError::new(ErrorKind::InvalidData, "episode seed precedes alpha-spending epoch")
        })?;
        if budget_index > BUDGET_HORIZON {
            return Err(IoError::new(
                ErrorKind::InvalidData,
                "episode seed exceeds the predeclared alpha-spending horizon",
            )
            .into());
        }
        process_episode(chunk, budget_index, &mut raw_output, &mut cells)?;
    }

    if episode_indices.len() != 5
        || episode_indices.values().any(|count| *count != TOTAL_CASES_PER_STATE)
        || cells.len() != 20
        || cells.values().any(|cell| cell.episodes != EPISODES_PER_CONDITION)
    {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            "alpha-spending outputs do not cover the 5 × 4 × 300 episode panel",
        )
        .into());
    }

    let full_spend = GLOBAL_DELTA * (1.0 - 1.0 / (BUDGET_HORIZON + 1) as f64);
    let mut summary = format!("{SUMMARY_HEADER}\n");
    for ((operating_point, condition), cell) in cells {
        summary_row(&mut summary, &operating_point, &condition, &cell);
    }
    // Four conditions are paired counterfactual replays for each episode index,
    // not four simultaneous runtime streams. The operational index advances once
    // per episode, and the same budget is used for its paired shift controls.
    if !full_spend.is_finite() || full_spend >= GLOBAL_DELTA {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            "lifetime alpha-spending sum exceeded its declared global delta",
        )
        .into());
    }
    fs::write(&args[2], raw_output)?;
    fs::write(&args[3], summary)?;
    Ok(())
}
