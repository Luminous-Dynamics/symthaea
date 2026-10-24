// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Group-aware paired percentile bootstrap for BROCA-TASK-EVAL-V1.
// Run the shell scorecard gate first so the CSV digest receipt is verified.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::process;

const HEADER: &str = "protocol_id,run_id,source_sha,dataset_id,dataset_sha256,config_sha256,split,case_id,group_id,stratum,arm,task_success,action_applicable,action_correct,abstention_applicable,abstention_correct,stale_memory_reuse,unauthorized_memory_exposure,deleted_memory_resurrection,query_latency_ms,write_latency_ms,input_tokens,output_tokens,storage_bytes,energy_j";

#[derive(Debug, Clone)]
struct Options {
    input: String,
    resamples: usize,
    seed: u64,
    min_cases: usize,
    min_groups: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Metadata {
    run_id: String,
    source_sha: String,
    dataset_id: String,
    dataset_sha256: String,
    config_sha256: String,
}

#[derive(Debug, Clone)]
struct Case {
    case_id: String,
    group_id: String,
    stratum: String,
    outcomes: BTreeMap<String, u8>,
}

#[derive(Debug)]
struct Dataset {
    metadata: Metadata,
    cases: Vec<Case>,
    arms: BTreeSet<String>,
    groups: Vec<Vec<usize>>,
}

#[derive(Debug)]
struct Contrast {
    candidate: String,
    baseline: String,
    wins: usize,
    losses: usize,
    ties: usize,
    mean_lift: f64,
    low: f64,
    high: f64,
}

fn usage() -> &'static str {
    "Usage: cargo run --manifest-path research/broca-task-eval-v1/Cargo.toml --release -- \\
  --input <validated-heldout.csv> [--resamples 10000] [--seed 20261024] \\
  [--min-cases 100] [--min-groups 10]\n\nRun scripts/check_broca_task_eval.sh on the CSV first. The shell gate verifies\nits receipt/digest and scorecard invariants; this tool computes grouped paired\npercentile-bootstrap intervals for task-success differences."
}

fn parse_options() -> Result<Option<Options>, String> {
    let mut args = env::args().skip(1);
    let mut input = None;
    let mut resamples = 10_000usize;
    let mut seed = 20_261_024u64;
    let mut min_cases = 100usize;
    let mut min_groups = 10usize;

    fn next_value(
        args: &mut std::iter::Skip<std::env::Args>,
        name: &str,
    ) -> Result<String, String> {
        args.next().ok_or_else(|| format!("{name} requires a value"))
    }

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => return Ok(None),
            "--input" => input = Some(next_value(&mut args, "--input")?),
            "--resamples" => {
                resamples = next_value(&mut args, "--resamples")?
                    .parse()
                    .map_err(|_| "--resamples must be a positive integer".to_string())?;
            }
            "--seed" => {
                seed = next_value(&mut args, "--seed")?
                    .parse()
                    .map_err(|_| "--seed must be an unsigned integer".to_string())?;
            }
            "--min-cases" => {
                min_cases = next_value(&mut args, "--min-cases")?
                    .parse()
                    .map_err(|_| "--min-cases must be a positive integer".to_string())?;
            }
            "--min-groups" => {
                min_groups = next_value(&mut args, "--min-groups")?
                    .parse()
                    .map_err(|_| "--min-groups must be a positive integer".to_string())?;
            }
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }

    let input = input.ok_or_else(|| "--input is required".to_string())?;
    if resamples == 0 || min_cases == 0 || min_groups == 0 {
        return Err("--resamples, --min-cases and --min-groups must be positive".to_string());
    }
    Ok(Some(Options {
        input,
        resamples,
        seed,
        min_cases,
        min_groups,
    }))
}

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'_' || b == b'-')
}

fn valid_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn parse_binary(value: &str, field: &str, line: usize) -> Result<u8, String> {
    match value {
        "0" => Ok(0),
        "1" => Ok(1),
        _ => Err(format!("line {line}: {field} must be 0 or 1")),
    }
}

fn parse_nonnegative_number(value: &str) -> bool {
    !value.is_empty()
        && value
            .parse::<f64>()
            .is_ok_and(|number| number.is_finite() && number >= 0.0)
}

fn read_dataset(path: &str, min_cases: usize, min_groups: usize) -> Result<Dataset, String> {
    let content = fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let mut lines = content.lines();
    let header = lines.next().ok_or_else(|| "CSV is empty".to_string())?;
    if header.trim_end_matches('\r') != HEADER {
        return Err("CSV header does not match BROCA-TASK-EVAL-V1".to_string());
    }

    let mut metadata: Option<Metadata> = None;
    let mut cases_by_id: BTreeMap<String, Case> = BTreeMap::new();
    let mut arms = BTreeSet::new();

    for (offset, raw_line) in lines.enumerate() {
        let line_number = offset + 2;
        let line = raw_line.trim_end_matches('\r');
        if line.is_empty() {
            return Err(format!("line {line_number}: blank rows are not allowed"));
        }
        if line.contains('"') {
            return Err(format!("line {line_number}: quoted fields are not allowed"));
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != 25 {
            return Err(format!("line {line_number}: expected 25 columns, got {}", fields.len()));
        }
        if fields.iter().any(|field| field.is_empty()) {
            return Err(format!("line {line_number}: blank fields are not allowed"));
        }
        if fields[0] != "BROCA-TASK-EVAL-V1" || fields[6] != "heldout" {
            return Err(format!("line {line_number}: wrong protocol ID or split"));
        }
        if !valid_id(fields[1]) || !valid_id(fields[3]) {
            return Err(format!("line {line_number}: invalid run_id or dataset_id"));
        }
        if !valid_hex(fields[2], 40)
            || !valid_hex(fields[4], 64)
            || !valid_hex(fields[5], 64)
        {
            return Err(format!("line {line_number}: malformed source/dataset/config digest"));
        }
        if !valid_id(fields[7])
            || !valid_id(fields[8])
            || !valid_id(fields[9])
            || !valid_id(fields[10])
        {
            return Err(format!("line {line_number}: invalid case, group, stratum or arm ID"));
        }

        let row_metadata = Metadata {
            run_id: fields[1].to_string(),
            source_sha: fields[2].to_string(),
            dataset_id: fields[3].to_string(),
            dataset_sha256: fields[4].to_string(),
            config_sha256: fields[5].to_string(),
        };
        if let Some(existing) = &metadata {
            if existing != &row_metadata {
                return Err(format!("line {line_number}: run metadata changes within one CSV"));
            }
        } else {
            metadata = Some(row_metadata);
        }

        let task_success = parse_binary(fields[11], "task_success", line_number)?;
        for (index, name) in [
            (12usize, "action_applicable"),
            (13, "action_correct"),
            (14, "abstention_applicable"),
            (15, "abstention_correct"),
            (16, "stale_memory_reuse"),
            (17, "unauthorized_memory_exposure"),
            (18, "deleted_memory_resurrection"),
        ] {
            let value = parse_binary(fields[index], name, line_number)?;
            if index >= 16 && value != 0 {
                return Err(format!(
                    "line {line_number}: observed governance event {name}; inference is blocked"
                ));
            }
        }
        if fields[12] == "0" && fields[13] != "0" {
            return Err(format!("line {line_number}: inapplicable action cannot be marked correct"));
        }
        if fields[14] == "0" && fields[15] != "0" {
            return Err(format!("line {line_number}: inapplicable abstention cannot be marked correct"));
        }
        if !parse_nonnegative_number(fields[19]) || !parse_nonnegative_number(fields[20]) {
            return Err(format!("line {line_number}: invalid latency value"));
        }
        for index in 21..=23 {
            if fields[index].parse::<u64>().is_err() {
                return Err(format!("line {line_number}: invalid integer resource metric"));
            }
        }
        if fields[24] != "-1"
            && !fields[24]
                .parse::<f64>()
                .is_ok_and(|number| number.is_finite() && number >= 0.0)
        {
            return Err(format!("line {line_number}: energy_j must be -1 or non-negative"));
        }

        let case_id = fields[7].to_string();
        let group_id = fields[8].to_string();
        let stratum = fields[9].to_string();
        let arm = fields[10].to_string();
        arms.insert(arm.clone());

        let case = cases_by_id.entry(case_id.clone()).or_insert_with(|| Case {
            case_id: case_id.clone(),
            group_id: group_id.clone(),
            stratum: stratum.clone(),
            outcomes: BTreeMap::new(),
        });
        if case.group_id != group_id || case.stratum != stratum {
            return Err(format!("line {line_number}: case group/stratum differs across arms"));
        }
        if case.outcomes.insert(arm.clone(), task_success).is_some() {
            return Err(format!("line {line_number}: duplicate case/arm pair {case_id}/{arm}"));
        }
    }

    let metadata = metadata.ok_or_else(|| "CSV contains no data rows".to_string())?;
    if cases_by_id.len() < min_cases {
        return Err(format!(
            "case count {} is below --min-cases {min_cases}",
            cases_by_id.len()
        ));
    }
    if !arms.contains("no_memory") || !arms.contains("exact_hdc") {
        return Err("required arms no_memory and exact_hdc must both exist".to_string());
    }
    if !["bm25", "dense_retriever", "strong_retriever"]
        .iter()
        .any(|arm| arms.contains(*arm))
    {
        return Err("at least one external retrieval arm (bm25, dense_retriever, strong_retriever) is required".to_string());
    }
    for case in cases_by_id.values() {
        if case.outcomes.len() != arms.len()
            || arms.iter().any(|arm| !case.outcomes.contains_key(arm))
        {
            return Err(format!("case {} does not have exactly one row for every arm", case.case_id));
        }
    }

    let cases: Vec<Case> = cases_by_id.into_values().collect();
    let mut group_map: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, case) in cases.iter().enumerate() {
        group_map.entry(case.group_id.clone()).or_default().push(index);
    }
    if group_map.len() < min_groups {
        return Err(format!(
            "independent group count {} is below --min-groups {min_groups}",
            group_map.len()
        ));
    }
    let groups = group_map.into_values().collect();

    Ok(Dataset {
        metadata,
        cases,
        arms,
        groups,
    })
}

// SplitMix64 gives reproducible pseudo-random draws without external crates.
#[derive(Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        debug_assert!(bound > 0);
        let bound = bound as u64;
        // Lemire's multiply-high method with rejection avoids modulo bias.
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let product = u128::from(self.next_u64()) * u128::from(bound);
            let low = product as u64;
            if low >= threshold {
                return (product >> 64) as usize;
            }
        }
    }
}

fn make_bootstrap_draws(group_count: usize, resamples: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut rng = SplitMix64::new(seed);
    (0..resamples)
        .map(|_| (0..group_count).map(|_| rng.below(group_count)).collect())
        .collect()
}

fn quantile(sorted: &[f64], probability: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let position = probability * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let weight = position - lower as f64;
    sorted[lower] * (1.0 - weight) + sorted[upper] * weight
}

fn bootstrap_interval(
    case_deltas: &[i8],
    groups: &[Vec<usize>],
    draws: &[Vec<usize>],
) -> (f64, f64) {
    let mut effects = Vec::with_capacity(draws.len());
    for sampled_groups in draws {
        let mut sum = 0i64;
        let mut count = 0usize;
        for &group_index in sampled_groups {
            for &case_index in &groups[group_index] {
                sum += i64::from(case_deltas[case_index]);
                count += 1;
            }
        }
        debug_assert!(count > 0);
        effects.push(sum as f64 / count as f64 * 100.0);
    }
    effects.sort_by(f64::total_cmp);
    (quantile(&effects, 0.025), quantile(&effects, 0.975))
}

fn compare(
    dataset: &Dataset,
    candidate: &str,
    baseline: &str,
    draws: &[Vec<usize>],
) -> Contrast {
    let mut deltas = Vec::with_capacity(dataset.cases.len());
    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut ties = 0usize;

    for case in &dataset.cases {
        let candidate_outcome = case.outcomes[candidate];
        let baseline_outcome = case.outcomes[baseline];
        match candidate_outcome.cmp(&baseline_outcome) {
            std::cmp::Ordering::Greater => wins += 1,
            std::cmp::Ordering::Less => losses += 1,
            std::cmp::Ordering::Equal => ties += 1,
        }
        deltas.push(candidate_outcome as i8 - baseline_outcome as i8);
    }
    let mean_lift = deltas.iter().map(|delta| f64::from(*delta)).sum::<f64>()
        / deltas.len() as f64
        * 100.0;
    let (low, high) = bootstrap_interval(&deltas, &dataset.groups, draws);
    Contrast {
        candidate: candidate.to_string(),
        baseline: baseline.to_string(),
        wins,
        losses,
        ties,
        mean_lift,
        low,
        high,
    }
}

fn run(options: Options) -> Result<(), String> {
    let dataset = read_dataset(&options.input, options.min_cases, options.min_groups)?;
    let draws = make_bootstrap_draws(dataset.groups.len(), options.resamples, options.seed);
    let mut contrasts = Vec::new();

    for candidate in &dataset.arms {
        for baseline in ["no_memory", "exact_hdc"] {
            if candidate.as_str() != baseline {
                contrasts.push(compare(&dataset, candidate, baseline, &draws));
            }
        }
    }
    contrasts.sort_by(|a, b| {
        a.candidate
            .cmp(&b.candidate)
            .then_with(|| a.baseline.cmp(&b.baseline))
    });

    eprintln!(
        "protocol=BROCA-TASK-EVAL-V1 source_sha={} dataset_id={} cases={} groups={} resamples={} seed={}",
        dataset.metadata.source_sha,
        dataset.metadata.dataset_id,
        dataset.cases.len(),
        dataset.groups.len(),
        options.resamples,
        options.seed
    );
    if dataset.groups.len() < 30 {
        eprintln!(
            "WARNING: fewer than 30 independent groups; treat percentile-bootstrap intervals as exploratory and increase independent groups before confirmatory claims"
        );
    }

    println!("candidate,baseline,cases,independent_groups,candidate_wins,baseline_wins,ties,mean_task_success_lift_pp,ci95_low_pp,ci95_high_pp,bootstrap_resamples,bootstrap_seed,method");
    for row in contrasts {
        println!(
            "{},{},{},{},{},{},{},{:.4},{:.4},{:.4},{},{},grouped_paired_percentile_bootstrap_unadjusted",
            row.candidate,
            row.baseline,
            dataset.cases.len(),
            dataset.groups.len(),
            row.wins,
            row.losses,
            row.ties,
            row.mean_lift,
            row.low,
            row.high,
            options.resamples,
            options.seed
        );
    }
    Ok(())
}

fn main() {
    let options = match parse_options() {
        Ok(Some(options)) => options,
        Ok(None) => {
            println!("{}", usage());
            return;
        }
        Err(error) => {
            eprintln!("error: {error}\n\n{}", usage());
            process::exit(2);
        }
    };
    if let Err(error) = run(options) {
        eprintln!("FAIL: {error}");
        process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_draws_are_deterministic_for_a_fixed_seed() {
        assert_eq!(
            make_bootstrap_draws(7, 20, 1234),
            make_bootstrap_draws(7, 20, 1234)
        );
        assert_ne!(
            make_bootstrap_draws(7, 20, 1234),
            make_bootstrap_draws(7, 20, 1235)
        );
    }

    #[test]
    fn bootstrap_draw_keeps_each_sampled_group_intact() {
        let deltas = [1i8, 1, -1, -1];
        let groups = vec![vec![0, 1], vec![2, 3]];
        let all_first_group = vec![vec![0, 0]];
        let all_second_group = vec![vec![1, 1]];
        assert_eq!(bootstrap_interval(&deltas, &groups, &all_first_group), (100.0, 100.0));
        assert_eq!(bootstrap_interval(&deltas, &groups, &all_second_group), (-100.0, -100.0));
    }

    #[test]
    fn quantile_interpolates_between_adjacent_samples() {
        assert!((quantile(&[0.0, 10.0], 0.25) - 2.5).abs() < f64::EPSILON * 10.0);
    }

    #[test]
    fn group_bootstrap_interval_is_repeatable() {
        let deltas = [1i8, 0, 1, -1, 0, 0];
        let groups = vec![vec![0, 1], vec![2, 3], vec![4, 5]];
        let first_draws = make_bootstrap_draws(groups.len(), 1000, 99);
        let second_draws = make_bootstrap_draws(groups.len(), 1000, 99);
        assert_eq!(
            bootstrap_interval(&deltas, &groups, &first_draws),
            bootstrap_interval(&deltas, &groups, &second_draws)
        );
    }
}
