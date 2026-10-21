//! Distributional and paired analysis for the allostatic-control benchmark.
//!
//! Reports episode-level violation rate/severity and paired shield effects.
//! The bootstrap is descriptive for this frozen synthetic seed panel; it does
//! not by itself establish generalization to new scenario families.

use allostatic_control_bench::{Policy, Scenario, EPISODE_CSV_HEADER, HORIZON};
use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::fs;
use std::io::{self, ErrorKind};

#[derive(Clone, Debug)]
struct Episode {
    task_reward: f64,
    cumulative_exposure: f64,
    unsafe_ticks: usize,
    unsafe_episode: bool,
    violation_exposure: f64,
    max_violation_severity: f64,
    minimum_reserve: f64,
    mean_safe_action_options: f64,
    minimum_safe_action_options: usize,
    missed_limits: usize,
    shield_vetoes: usize,
    no_safe_action_ticks: usize,
    emergency_recover_steps: usize,
    false_alarms: usize,
    false_alarm_opportunities: usize,
    recovery_steps: Option<usize>,
    failed: bool,
}

type ConditionKey = (String, String, bool);
type EpisodesBySeed = BTreeMap<u64, Episode>;

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message.into())
}

fn column_index(header: &BTreeMap<String, usize>, name: &str) -> Result<usize, Box<dyn Error>> {
    header
        .get(name)
        .copied()
        .ok_or_else(|| invalid_data(format!("missing CSV column: {name}")).into())
}

fn cell<'a>(
    values: &'a [&str],
    header: &BTreeMap<String, usize>,
    name: &str,
) -> Result<&'a str, Box<dyn Error>> {
    let index = column_index(header, name)?;
    values
        .get(index)
        .copied()
        .ok_or_else(|| invalid_data(format!("row missing column {name}")).into())
}

fn parse_f64(
    values: &[&str],
    header: &BTreeMap<String, usize>,
    name: &str,
) -> Result<f64, Box<dyn Error>> {
    let parsed: f64 = cell(values, header, name)?.parse()?;
    if !parsed.is_finite() {
        return Err(invalid_data(format!("nonfinite numeric value in {name}")).into());
    }
    Ok(parsed)
}

fn parse_usize(
    values: &[&str],
    header: &BTreeMap<String, usize>,
    name: &str,
) -> Result<usize, Box<dyn Error>> {
    Ok(cell(values, header, name)?.parse()?)
}

fn parse_bool(
    values: &[&str],
    header: &BTreeMap<String, usize>,
    name: &str,
) -> Result<bool, Box<dyn Error>> {
    match cell(values, header, name)? {
        "true" => Ok(true),
        "false" => Ok(false),
        other => Err(invalid_data(format!("invalid boolean in {name}: {other}")).into()),
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Nearest-rank empirical quantile. For q=0.95 this returns the observed
/// 95th-percentile episode, not a fitted or extrapolated tail estimate.
fn percentile_nearest_rank(values: &[f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let q = quantile.clamp(0.0, 1.0);
    let rank = ((q * ordered.len() as f64).ceil() as usize).max(1);
    ordered[rank.min(ordered.len()) - 1]
}

fn next_unit(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / ((1u64 << 53) as f64)
}

fn stable_seed(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01B3)
    })
}

/// Percentile bootstrap CI for a mean paired difference. Pairing happens
/// before this function: each input value is the same-seed shielded-minus-
/// unshielded difference. Fixed resampling makes the report reproducible.
fn bootstrap_mean_ci(values: &[f64], seed: u64) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    const REPLICATES: usize = 4_000;
    let mut state = seed;
    let mut boot_means = Vec::with_capacity(REPLICATES);
    for _ in 0..REPLICATES {
        let mut total = 0.0;
        for _ in 0..values.len() {
            let index = (next_unit(&mut state) * values.len() as f64) as usize;
            total += values[index.min(values.len() - 1)];
        }
        boot_means.push(total / values.len() as f64);
    }
    (
        percentile_nearest_rank(&boot_means, 0.025),
        percentile_nearest_rank(&boot_means, 0.975),
    )
}

fn parse_episodes(csv: &str) -> Result<BTreeMap<ConditionKey, EpisodesBySeed>, Box<dyn Error>> {
    let mut lines = csv.lines();
    let header_line = lines.next().ok_or_else(|| invalid_data("CSV is empty"))?;
    if header_line != EPISODE_CSV_HEADER {
        return Err(invalid_data("CSV header does not match the declared episode schema").into());
    }
    let header_fields: Vec<&str> = header_line.split(',').collect();
    let header: BTreeMap<String, usize> = header_fields
        .iter()
        .enumerate()
        .map(|(index, name)| ((*name).to_string(), index))
        .collect();
    if header.len() != header_fields.len() {
        return Err(invalid_data("CSV header contains duplicate column names").into());
    }

    let required = [
        "scenario",
        "seed",
        "policy",
        "shielded",
        "task_reward",
        "cumulative_exposure",
        "unsafe_ticks",
        "unsafe_episode",
        "violation_exposure",
        "max_violation_severity",
        "minimum_reserve",
        "missed_limits",
        "shield_vetoes",
        "emergency_recover_steps",
        "no_safe_action_ticks",
        "mean_safe_action_options",
        "minimum_safe_action_options",
        "false_alarms",
        "false_alarm_opportunities",
        "push_steps",
        "maintain_steps",
        "resource_cost",
        "recovery_steps",
        "failed",
    ];
    for name in required {
        column_index(&header, name)?;
    }

    let mut groups: BTreeMap<ConditionKey, EpisodesBySeed> = BTreeMap::new();
    for (line_index, line) in lines.enumerate() {
        if line.is_empty() {
            continue;
        }
        let values: Vec<&str> = line.split(',').collect();
        if values.len() != header_fields.len() {
            return Err(invalid_data(format!(
                "CSV column count mismatch on line {}",
                line_index + 2
            ))
            .into());
        }
        let scenario = cell(&values, &header, "scenario")?.to_string();
        let policy = cell(&values, &header, "policy")?.to_string();
        let shielded = parse_bool(&values, &header, "shielded")?;
        let seed: u64 = cell(&values, &header, "seed")?.parse()?;
        if !Scenario::all()
            .iter()
            .any(|candidate| candidate.name() == scenario.as_str())
            || !Policy::all()
                .iter()
                .any(|candidate| candidate.name() == policy.as_str())
            || seed >= 32
        {
            return Err(invalid_data(format!(
                "unknown scenario/policy or out-of-panel seed on line {}",
                line_index + 2
            ))
            .into());
        }
        let unsafe_ticks = parse_usize(&values, &header, "unsafe_ticks")?;
        let unsafe_episode = parse_bool(&values, &header, "unsafe_episode")?;
        let cumulative_load = parse_f64(&values, &header, "cumulative_exposure")?;
        let mean_load = parse_f64(&values, &header, "mean_load")?;
        let peak_load = parse_f64(&values, &header, "peak_load")?;
        let violation_exposure = parse_f64(&values, &header, "violation_exposure")?;
        let max_violation_severity = parse_f64(&values, &header, "max_violation_severity")?;
        let minimum_reserve = parse_f64(&values, &header, "minimum_reserve")?;
        let mean_safe_action_options = parse_f64(&values, &header, "mean_safe_action_options")?;
        let minimum_safe_action_options =
            parse_usize(&values, &header, "minimum_safe_action_options")?;
        let terminal_reserve = parse_f64(&values, &header, "terminal_reserve")?;
        let terminal_load = parse_f64(&values, &header, "terminal_load")?;
        let false_alarms = parse_usize(&values, &header, "false_alarms")?;
        let false_alarm_opportunities = parse_usize(&values, &header, "false_alarm_opportunities")?;
        let missed_limits = parse_usize(&values, &header, "missed_limits")?;
        let shield_vetoes = parse_usize(&values, &header, "shield_vetoes")?;
        let no_safe_action_ticks = parse_usize(&values, &header, "no_safe_action_ticks")?;
        let push_steps = parse_usize(&values, &header, "push_steps")?;
        let maintain_steps = parse_usize(&values, &header, "maintain_steps")?;
        let emergency_recover_steps = parse_usize(&values, &header, "emergency_recover_steps")?;
        let resource_cost = parse_f64(&values, &header, "resource_cost")?;
        let recovery_text = cell(&values, &header, "recovery_steps")?;
        let recovery_steps = if recovery_text == "NA" {
            None
        } else {
            Some(recovery_text.parse::<usize>()?)
        };
        if unsafe_episode != (unsafe_ticks > 0) {
            return Err(invalid_data(format!(
                "unsafe_episode disagrees with unsafe_ticks on line {}",
                line_index + 2
            ))
            .into());
        }
        for (name, value) in [
            ("mean_load", mean_load),
            ("peak_load", peak_load),
            ("minimum_reserve", minimum_reserve),
            ("terminal_reserve", terminal_reserve),
            ("terminal_load", terminal_load),
            ("max_violation_severity", max_violation_severity),
        ] {
            if !(0.0..=1.0).contains(&value) {
                return Err(invalid_data(format!(
                    "{name} outside declared [0,1] range on line {}",
                    line_index + 2
                ))
                .into());
            }
        }
        let mean_tolerance = 1e-5;
        if !(0.0..=3.0).contains(&mean_safe_action_options)
            || minimum_safe_action_options > 3
            || mean_safe_action_options + mean_tolerance < minimum_safe_action_options as f64
        {
            return Err(invalid_data(format!(
                "safe-action option metric violates contract on line {}",
                line_index + 2
            ))
            .into());
        }
        if cumulative_load < 0.0 || violation_exposure < 0.0 || resource_cost < 0.0 {
            return Err(invalid_data(format!(
                "negative accumulated metric on line {}",
                line_index + 2
            ))
            .into());
        }
        if (mean_load - cumulative_load / HORIZON as f64).abs() > mean_tolerance
            || peak_load + mean_tolerance < mean_load
            || peak_load + mean_tolerance < terminal_load
            || minimum_reserve > terminal_reserve + mean_tolerance
        {
            return Err(invalid_data(format!(
                "inconsistent trajectory summary on line {}",
                line_index + 2
            ))
            .into());
        }
        if unsafe_ticks > HORIZON
            || false_alarms > false_alarm_opportunities
            || false_alarm_opportunities > HORIZON
            || missed_limits > HORIZON
            || shield_vetoes > HORIZON
            || no_safe_action_ticks > HORIZON
            || push_steps > HORIZON
            || maintain_steps > HORIZON
            || emergency_recover_steps > HORIZON
            || push_steps + maintain_steps + emergency_recover_steps != HORIZON
        {
            return Err(invalid_data(format!(
                "episode counters violate horizon/denominator contract on line {}",
                line_index + 2
            ))
            .into());
        }
        if recovery_steps.is_some_and(|steps| steps > HORIZON) {
            return Err(invalid_data(format!(
                "recovery_steps outside horizon on line {}",
                line_index + 2
            ))
            .into());
        }
        let episode = Episode {
            task_reward: parse_f64(&values, &header, "task_reward")?,
            cumulative_exposure: cumulative_load,
            unsafe_ticks,
            unsafe_episode,
            violation_exposure,
            max_violation_severity,
            minimum_reserve,
            mean_safe_action_options,
            minimum_safe_action_options,
            missed_limits,
            shield_vetoes,
            no_safe_action_ticks,
            emergency_recover_steps,
            false_alarms,
            false_alarm_opportunities,
            recovery_steps,
            failed: parse_bool(&values, &header, "failed")?,
        };
        let group = groups.entry((scenario, policy, shielded)).or_default();
        if group.insert(seed, episode).is_some() {
            return Err(invalid_data(format!(
                "duplicate episode seed {seed} on line {}",
                line_index + 2
            ))
            .into());
        }
    }
    if groups.is_empty() {
        return Err(invalid_data("CSV contains no episode rows").into());
    }
    Ok(groups)
}

fn write_summary(
    groups: &BTreeMap<ConditionKey, EpisodesBySeed>,
    path: &str,
) -> Result<usize, Box<dyn Error>> {
    let mut output = String::from(
        "scenario,policy,shielded,n,mean_task_reward,mean_cumulative_load,p95_cumulative_load,episode_violation_rate,mean_unsafe_ticks,mean_violation_exposure,p95_violation_exposure,mean_conditional_peak_violation,mean_minimum_reserve,mean_missed_limits,mean_shield_vetoes,mean_no_safe_action_ticks,mean_emergency_recover_steps,mean_false_alarms,mean_false_alarm_opportunities,false_alarm_rate_when_opportunity,recovery_observed_n,mean_observed_recovery_steps,recovery_censored_rate,mean_viable_action_options,worst_episode_minimum_viable_action_options,episode_failure_rate\n"
    );
    for ((scenario, policy, shielded), by_seed) in groups {
        let episodes: Vec<&Episode> = by_seed.values().collect();
        let rewards: Vec<f64> = episodes.iter().map(|row| row.task_reward).collect();
        let exposure: Vec<f64> = episodes.iter().map(|row| row.cumulative_exposure).collect();
        let unsafe_ticks: Vec<f64> = episodes.iter().map(|row| row.unsafe_ticks as f64).collect();
        let violation_exposure: Vec<f64> =
            episodes.iter().map(|row| row.violation_exposure).collect();
        let conditional_severity: Vec<f64> = episodes
            .iter()
            .filter(|row| row.unsafe_episode)
            .map(|row| row.max_violation_severity)
            .collect();
        let minimum_reserve: Vec<f64> = episodes.iter().map(|row| row.minimum_reserve).collect();
        let mean_viable_options: Vec<f64> = episodes
            .iter()
            .map(|row| row.mean_safe_action_options)
            .collect();
        let worst_episode_minimum_viable_options: usize = episodes
            .iter()
            .map(|row| row.minimum_safe_action_options)
            .min()
            .unwrap_or(0);
        let missed: Vec<f64> = episodes
            .iter()
            .map(|row| row.missed_limits as f64)
            .collect();
        let vetoes: Vec<f64> = episodes
            .iter()
            .map(|row| row.shield_vetoes as f64)
            .collect();
        let no_safe_actions: Vec<f64> = episodes
            .iter()
            .map(|row| row.no_safe_action_ticks as f64)
            .collect();
        let emergency_steps: Vec<f64> = episodes
            .iter()
            .map(|row| row.emergency_recover_steps as f64)
            .collect();
        let false_alarms: Vec<f64> = episodes.iter().map(|row| row.false_alarms as f64).collect();
        let false_alarm_opportunities: Vec<f64> = episodes
            .iter()
            .map(|row| row.false_alarm_opportunities as f64)
            .collect();
        let false_alarm_total: usize = episodes.iter().map(|row| row.false_alarms).sum();
        let false_alarm_opportunity_total: usize = episodes
            .iter()
            .map(|row| row.false_alarm_opportunities)
            .sum();
        let false_alarm_rate = if false_alarm_opportunity_total == 0 {
            "NA".to_string()
        } else {
            format!(
                "{:.6}",
                false_alarm_total as f64 / false_alarm_opportunity_total as f64
            )
        };
        let observed_recovery: Vec<f64> = episodes
            .iter()
            .filter_map(|row| row.recovery_steps.map(|steps| steps as f64))
            .collect();
        let recovery_censored = episodes
            .iter()
            .filter(|row| row.recovery_steps.is_none())
            .count();
        let recovery_mean = if observed_recovery.is_empty() {
            "NA".to_string()
        } else {
            format!("{:.6}", mean(&observed_recovery))
        };
        let recovery_censored_rate = recovery_censored as f64 / episodes.len() as f64;
        let violation_rate =
            episodes.iter().filter(|row| row.unsafe_episode).count() as f64 / episodes.len() as f64;
        let failure_rate =
            episodes.iter().filter(|row| row.failed).count() as f64 / episodes.len() as f64;
        let conditional = if conditional_severity.is_empty() {
            "NA".to_string()
        } else {
            format!("{:.6}", mean(&conditional_severity))
        };
        output.push_str(&format!(
            "{scenario},{policy},{shielded},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{conditional},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{false_alarm_rate},{},{recovery_mean},{:.6},{:.6},{},{:.6}\n",
            episodes.len(),
            mean(&rewards),
            mean(&exposure),
            percentile_nearest_rank(&exposure, 0.95),
            violation_rate,
            mean(&unsafe_ticks),
            mean(&violation_exposure),
            percentile_nearest_rank(&violation_exposure, 0.95),
            mean(&minimum_reserve),
            mean(&missed),
            mean(&vetoes),
            mean(&no_safe_actions),
            mean(&emergency_steps),
            mean(&false_alarms),
            mean(&false_alarm_opportunities),
            observed_recovery.len(),
            recovery_censored_rate,
            mean(&mean_viable_options),
            worst_episode_minimum_viable_options,
            failure_rate,
        ));
    }
    let rows = output.lines().count().saturating_sub(1);
    fs::write(path, output)?;
    Ok(rows)
}

fn write_paired_deltas(
    groups: &BTreeMap<ConditionKey, EpisodesBySeed>,
    path: &str,
) -> Result<usize, Box<dyn Error>> {
    let mut output = String::from(
        "scenario,policy,n,delta_task_reward_mean,delta_task_reward_ci95_low,delta_task_reward_ci95_high,delta_cumulative_load_mean,delta_cumulative_load_ci95_low,delta_cumulative_load_ci95_high,delta_violation_exposure_mean,delta_violation_exposure_ci95_low,delta_violation_exposure_ci95_high,delta_episode_violation_rate_mean,delta_episode_violation_rate_ci95_low,delta_episode_violation_rate_ci95_high,delta_peak_violation_severity_mean,delta_peak_violation_severity_ci95_low,delta_peak_violation_severity_ci95_high\n"
    );
    let mut pair_count = 0;
    for (key, shielded_rows) in groups {
        let (scenario, policy, shielded) = key;
        if !*shielded {
            continue;
        }
        let unshielded_rows = groups
            .get(&(scenario.to_string(), policy.to_string(), false))
            .ok_or_else(|| {
                invalid_data(format!("missing unshielded pair for {scenario}/{policy}"))
            })?;
        if shielded_rows.len() != unshielded_rows.len() {
            return Err(invalid_data(format!(
                "paired sample count mismatch for {scenario}/{policy}"
            ))
            .into());
        }
        let mut reward_delta = Vec::new();
        let mut load_delta = Vec::new();
        let mut exposure_delta = Vec::new();
        let mut violation_rate_delta = Vec::new();
        let mut severity_delta = Vec::new();
        for (seed, shielded_episode) in shielded_rows {
            let baseline = unshielded_rows.get(seed).ok_or_else(|| {
                invalid_data(format!(
                    "missing paired seed {seed} for {scenario}/{policy}"
                ))
            })?;
            reward_delta.push(shielded_episode.task_reward - baseline.task_reward);
            load_delta.push(shielded_episode.cumulative_exposure - baseline.cumulative_exposure);
            exposure_delta.push(shielded_episode.violation_exposure - baseline.violation_exposure);
            let shielded_violation = if shielded_episode.unsafe_episode {
                1.0
            } else {
                0.0
            };
            let baseline_violation = if baseline.unsafe_episode { 1.0 } else { 0.0 };
            violation_rate_delta.push(shielded_violation - baseline_violation);
            severity_delta
                .push(shielded_episode.max_violation_severity - baseline.max_violation_severity);
        }
        let reward_ci = bootstrap_mean_ci(
            &reward_delta,
            stable_seed(&format!("{scenario}/{policy}/reward")),
        );
        let load_ci = bootstrap_mean_ci(
            &load_delta,
            stable_seed(&format!("{scenario}/{policy}/load")),
        );
        let exposure_ci = bootstrap_mean_ci(
            &exposure_delta,
            stable_seed(&format!("{scenario}/{policy}/violation-exposure")),
        );
        let rate_ci = bootstrap_mean_ci(
            &violation_rate_delta,
            stable_seed(&format!("{scenario}/{policy}/violation-rate")),
        );
        let severity_ci = bootstrap_mean_ci(
            &severity_delta,
            stable_seed(&format!("{scenario}/{policy}/severity")),
        );
        output.push_str(&format!(
            "{scenario},{policy},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            reward_delta.len(),
            mean(&reward_delta), reward_ci.0, reward_ci.1,
            mean(&load_delta), load_ci.0, load_ci.1,
            mean(&exposure_delta), exposure_ci.0, exposure_ci.1,
            mean(&violation_rate_delta), rate_ci.0, rate_ci.1,
            mean(&severity_delta), severity_ci.0, severity_ci.1,
        ));
        pair_count += 1;
    }
    if pair_count == 0 {
        return Err(invalid_data("no shielded/unshielded pairs found").into());
    }
    let rows = output.lines().count().saturating_sub(1);
    fs::write(path, output)?;
    Ok(rows)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        return Err(invalid_data(
            "usage: allostatic-control-report <episode.csv> <summary.csv> <paired-deltas.csv>",
        )
        .into());
    }
    let input = fs::read_to_string(&args[1])?;
    let groups = parse_episodes(&input)?;
    let summary_rows = write_summary(&groups, &args[2])?;
    let paired_rows = write_paired_deltas(&groups, &args[3])?;
    println!("summary_rows={summary_rows}");
    println!("paired_delta_rows={paired_rows}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_HEADER: &str = EPISODE_CSV_HEADER;

    const SAMPLE_ROW: &str = "calm,0,task_only,false,1.0,4.8,0.1,0.2,0.8,1.5,1,0.8,0.2,0,false,0.0,0.0,0,0,0,0,1,1,47,0,0.2,false,NA";

    #[test]
    fn parser_requires_episode_violation_flag_to_match_tick_count() {
        let good = format!("{SAMPLE_HEADER}\n{SAMPLE_ROW}\n");
        let groups = parse_episodes(&good).expect("valid minimal CSV should parse");
        assert_eq!(groups.len(), 1);

        let inconsistent = good.replace(",0,false,0.0,0.0,", ",1,false,0.0,0.0,");
        assert!(parse_episodes(&inconsistent).is_err());
    }

    #[test]
    fn parser_rejects_noncanonical_header() {
        let altered = format!(
            "{}\n{SAMPLE_ROW}\n",
            SAMPLE_HEADER.replace("scenario,seed", "scenario,id")
        );
        assert!(parse_episodes(&altered).is_err());
    }

    #[test]
    fn parser_rejects_inconsistent_mean_and_cumulative_load() {
        let altered_row = SAMPLE_ROW.replace(",4.8,0.1,", ",4.8,0.2,");
        let altered = format!("{SAMPLE_HEADER}\n{altered_row}\n");
        assert!(parse_episodes(&altered).is_err());
    }

    #[test]
    fn parser_rejects_unknown_scenario_policy_and_seed() {
        let unknown_scenario = format!(
            "{}\n{}\n",
            SAMPLE_HEADER,
            SAMPLE_ROW.replace("calm,0,task_only", "unknown,0,task_only")
        );
        assert!(parse_episodes(&unknown_scenario).is_err());

        let unknown_policy = format!(
            "{}\n{}\n",
            SAMPLE_HEADER,
            SAMPLE_ROW.replace("calm,0,task_only", "calm,0,unknown")
        );
        assert!(parse_episodes(&unknown_policy).is_err());

        let out_of_panel_seed = format!(
            "{}\n{}\n",
            SAMPLE_HEADER,
            SAMPLE_ROW.replace("calm,0,task_only", "calm,32,task_only")
        );
        assert!(parse_episodes(&out_of_panel_seed).is_err());
    }

    #[test]
    fn parser_rejects_duplicate_condition_seed_rows() {
        let duplicate = format!("{SAMPLE_HEADER}\n{SAMPLE_ROW}\n{SAMPLE_ROW}\n");
        assert!(parse_episodes(&duplicate).is_err());
    }

    #[test]
    fn nearest_rank_percentile_uses_observed_samples() {
        assert_eq!(percentile_nearest_rank(&[1.0, 2.0, 3.0, 4.0], 0.95), 4.0);
        assert_eq!(percentile_nearest_rank(&[9.0], 0.95), 9.0);
        assert_eq!(percentile_nearest_rank(&[], 0.95), 0.0);
    }

    #[test]
    fn paired_bootstrap_is_deterministic_and_collapses_for_constant_values() {
        let values = [2.0, 2.0, 2.0, 2.0];
        let first = bootstrap_mean_ci(&values, 42);
        let second = bootstrap_mean_ci(&values, 42);
        assert_eq!(first, second);
        assert!((first.0 - 2.0).abs() < 1e-12);
        assert!((first.1 - 2.0).abs() < 1e-12);
    }
}
