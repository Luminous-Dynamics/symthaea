// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Seed-clustered confidence and discrete temporal-capacity frontier analysis.
//!
//! This program consumes temporal_associative_capacity JSONL output. Queries
//! within one bundled memory are NOT treated as independent replicates. The
//! statistical unit is the independent corpus/model seed.
//!
//! Run:
//!   cargo run --release --manifest-path crates/core/symthaea-hdc-ltc/Cargo.toml \
//!     --example temporal_capacity_frontier -- --input temporal-capacity.jsonl \
//!     --out temporal-frontier.jsonl
//!
//! K95 and T90 are evaluated only on tested grid points. They are qualified
//! using one-sided 95% exact Clopper-Pearson lower bounds on the fraction of
//! independent seeds meeting a per-seed retrieval threshold.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};

const SCHEMA_VERSION: &str = "symthaea.temporal-capacity-frontier/0.1";
const INPUT_SCHEMA_VERSION: &str = "symthaea.temporal-associative-capacity/0.1";
const ALPHA: f64 = 0.05;
const K95_PER_SEED_ACCURACY: f64 = 0.95;
const T90_PER_SEED_ACCURACY: f64 = 0.90;
const REQUIRED_SEED_RELIABILITY: f64 = 0.95;

#[derive(Debug)]
struct Args {
    input: String,
    output: Option<String>,
    min_seeds: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            input: String::new(),
            output: None,
            min_seeds: 59,
        }
    }
}

#[derive(Debug, Deserialize)]
struct SummaryInput {
    record_type: String,
    schema_version: String,
    source_revision: Option<String>,
    cohort_id: String,
    seed: u64,
    dimension: usize,
    memory_fan_in: usize,
    condition: String,
    gap_profile: String,
    horizon_seconds: f64,
    event_count: usize,
    query_count: usize,
    correct_count: usize,
    valid_query_count: usize,
    tie_count: usize,
    invalid_count: usize,
    accuracy: f64,
    chance_accuracy: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CohortKey {
    source_revision: Option<String>,
    dimension: usize,
    memory_fan_in: usize,
    condition: String,
    gap_profile: String,
    horizon_bits: u64,
    event_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct K95FamilyKey {
    source_revision: Option<String>,
    dimension: usize,
    condition: String,
    gap_profile: String,
    horizon_bits: u64,
    event_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct T90FamilyKey {
    source_revision: Option<String>,
    dimension: usize,
    condition: String,
    gap_profile: String,
    memory_fan_in: usize,
    event_count: usize,
}

#[derive(Debug, Default)]
struct CohortData {
    seed_accuracy: BTreeMap<u64, f64>,
    chance_accuracy: Option<f64>,
    cohort_ids: BTreeSet<String>,
}

#[derive(Serialize)]
#[serde(tag = "record_type", rename_all = "snake_case")]
enum OutputRecord {
    CohortReliability(CohortReliabilityRecord),
    K95(K95Record),
    T90(T90Record),
}

#[derive(Serialize)]
struct CohortReliabilityRecord {
    schema_version: &'static str,
    source_revision: Option<String>,
    dimension: usize,
    memory_fan_in: usize,
    condition: String,
    gap_profile: String,
    horizon_seconds: f64,
    event_count: usize,
    cohort_ids: Vec<String>,
    independent_seed_count: usize,
    min_seed_count_gate: usize,
    mean_seed_accuracy: f64,
    min_seed_accuracy: f64,
    max_seed_accuracy: f64,
    seeds_meeting_95_percent_accuracy: usize,
    seed_reliability_lower_95_for_95_percent_accuracy: f64,
    seeds_meeting_90_percent_accuracy: usize,
    seed_reliability_lower_95_for_90_percent_accuracy: f64,
    chance_accuracy: f64,
    confirmatory_sample_gate_met: bool,
}

#[derive(Serialize)]
struct K95Record {
    schema_version: &'static str,
    source_revision: Option<String>,
    dimension: usize,
    condition: String,
    gap_profile: String,
    horizon_seconds: f64,
    event_count: usize,
    target_per_seed_accuracy: f64,
    target_seed_reliability: f64,
    familywise_confidence_level: f64,
    per_level_alpha: f64,
    tested_level_count: usize,
    multiplicity_correction: &'static str,
    minimum_seeds_if_all_pass: usize,
    tested_fan_in_levels: Vec<usize>,
    common_seed_count: usize,
    paired_seed_sets: bool,
    confirmatory_sample_gate_met: bool,
    qualified_k95: Option<usize>,
    interpretation: &'static str,
}

#[derive(Serialize)]
struct T90Record {
    schema_version: &'static str,
    source_revision: Option<String>,
    dimension: usize,
    condition: String,
    gap_profile: String,
    memory_fan_in: usize,
    event_count: usize,
    target_per_seed_accuracy: f64,
    target_seed_reliability: f64,
    familywise_confidence_level: f64,
    per_level_alpha: f64,
    tested_level_count: usize,
    multiplicity_correction: &'static str,
    minimum_seeds_if_all_pass: usize,
    tested_horizon_seconds: Vec<f64>,
    common_seed_count: usize,
    paired_seed_sets: bool,
    confirmatory_sample_gate_met: bool,
    qualified_t90_seconds: Option<f64>,
    interpretation: &'static str,
}

fn valid_git_revision(revision: &str) -> bool {
    matches!(revision.len(), 40 | 64)
        && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut args = Args::default();
    let mut iter = std::env::args().skip(1);

    while let Some(flag) = iter.next() {
        let value = match flag.as_str() {
            "--help" | "-h" => {
                println!(
                    "temporal_capacity_frontier --input results.jsonl [--out frontier.jsonl] \
                     [--min-seeds 59]"
                );
                std::process::exit(0);
            }
            "--input" | "--out" | "--min-seeds" => {
                iter.next().ok_or_else(|| format!("missing value for {flag}"))?
            }
            _ => return Err(format!("unknown argument: {flag}").into()),
        };

        match flag.as_str() {
            "--input" => args.input = value,
            "--out" => args.output = Some(value),
            "--min-seeds" => args.min_seeds = value.parse()?,
            _ => unreachable!("flag validated above"),
        }
    }

    if args.input.is_empty() {
        return Err("--input is required".into());
    }
    if args.min_seeds < 2 || args.min_seeds > 10_000 {
        return Err("--min-seeds must be in 2..=10000".into());
    }
    if args.output.as_deref() == Some(args.input.as_str()) {
        return Err("input and output paths must differ".into());
    }
    Ok(args)
}

fn validate_summary(row: &SummaryInput, line_number: usize) -> Result<(), Box<dyn Error>> {
    let count_sum = row.valid_query_count.checked_add(row.invalid_count);
    let correct_tie_sum = row.correct_count.checked_add(row.tie_count);
    if row.record_type != "summary" || row.schema_version != INPUT_SCHEMA_VERSION {
        return Err(format!("line {line_number}: wrong record type or schema version").into());
    }
    if row.cohort_id.is_empty()
        || row.dimension < 8
        || row.memory_fan_in < 2
        || row.query_count != row.memory_fan_in
        || row.query_count == 0
        || row.correct_count > row.query_count
        || count_sum != Some(row.query_count)
        || correct_tie_sum.is_none_or(|sum| sum > row.valid_query_count)
        || !row.accuracy.is_finite()
        || !(0.0..=1.0).contains(&row.accuracy)
        || !row.chance_accuracy.is_finite()
        || !(0.0..=1.0).contains(&row.chance_accuracy)
        || !row.horizon_seconds.is_finite()
        || row.horizon_seconds < 0.0
    {
        return Err(format!("line {line_number}: invalid summary fields").into());
    }
    let reproduced_accuracy = row.correct_count as f64 / row.query_count as f64;
    if (reproduced_accuracy - row.accuracy).abs() > 1.0e-12 {
        return Err(format!("line {line_number}: accuracy does not reproduce from counts").into());
    }
    let revision = row
        .source_revision
        .as_deref()
        .ok_or_else(|| format!("line {line_number}: source revision is required for frontier analysis"))?;
    if !valid_git_revision(revision) {
        return Err(format!("line {line_number}: invalid source revision").into());
    }
    if !matches!(
        row.condition.as_str(),
        "static_hdc_bundle"
            | "static_mismatched_pairing_control"
            | "static_random_query_control"
            | "hdc_ltc_zero_input"
            | "hdc_ltc_distractor"
    ) {
        return Err(format!("line {line_number}: unknown condition").into());
    }
    if !matches!(row.gap_profile.as_str(), "not_applicable" | "regular" | "irregular") {
        return Err(format!("line {line_number}: unknown gap profile").into());
    }
    Ok(())
}

fn load_summaries(path: &str) -> Result<BTreeMap<CohortKey, CohortData>, Box<dyn Error>> {
    let input = BufReader::new(File::open(path)?);
    let mut cohorts: BTreeMap<CohortKey, CohortData> = BTreeMap::new();

    for (index, line) in input.lines().enumerate() {
        let line_number = index + 1;
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(&line)?;
        let record_type = value
            .get("record_type")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("line {line_number}: missing record_type"))?;
        if record_type == "query" {
            continue;
        }
        if record_type != "summary" {
            return Err(format!("line {line_number}: unexpected record type {record_type}").into());
        }

        let row: SummaryInput = serde_json::from_value(value)?;
        validate_summary(&row, line_number)?;
        let key = CohortKey {
            source_revision: row.source_revision,
            dimension: row.dimension,
            memory_fan_in: row.memory_fan_in,
            condition: row.condition,
            gap_profile: row.gap_profile,
            horizon_bits: row.horizon_seconds.to_bits(),
            event_count: row.event_count,
        };
        let cohort = cohorts.entry(key).or_default();
        if let Some(previous) = cohort.chance_accuracy {
            if (previous - row.chance_accuracy).abs() > 1.0e-12 {
                return Err(format!("line {line_number}: chance accuracy changed within cohort").into());
            }
        } else {
            cohort.chance_accuracy = Some(row.chance_accuracy);
        }
        if cohort.seed_accuracy.insert(row.seed, row.accuracy).is_some() {
            return Err(format!("line {line_number}: duplicate seed in cohort").into());
        }
        cohort.cohort_ids.insert(row.cohort_id);
    }

    if cohorts.is_empty() {
        return Err("input contains no summary records".into());
    }
    Ok(cohorts)
}

// Lanczos approximation for log(Gamma(z)), used by the regularized incomplete
// beta function in the exact Clopper-Pearson confidence calculation.
fn ln_gamma(z: f64) -> f64 {
    const COEFFICIENTS: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if z < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * z).sin().ln()
            - ln_gamma(1.0 - z);
    }

    let z = z - 1.0;
    let mut x = COEFFICIENTS[0];
    for (i, coefficient) in COEFFICIENTS.iter().enumerate().skip(1) {
        x += coefficient / (z + i as f64);
    }
    let t = z + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
}

fn beta_continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITERATIONS: usize = 300;
    const EPSILON: f64 = 3.0e-14;
    const MIN_DENOMINATOR: f64 = 1.0e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < MIN_DENOMINATOR {
        d = MIN_DENOMINATOR;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_ITERATIONS {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;
        let mut aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < MIN_DENOMINATOR {
            d = MIN_DENOMINATOR;
        }
        c = 1.0 + aa / c;
        if c.abs() < MIN_DENOMINATOR {
            c = MIN_DENOMINATOR;
        }
        d = 1.0 / d;
        h *= d * c;

        aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < MIN_DENOMINATOR {
            d = MIN_DENOMINATOR;
        }
        c = 1.0 + aa / c;
        if c.abs() < MIN_DENOMINATOR {
            c = MIN_DENOMINATOR;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() <= EPSILON {
            break;
        }
    }
    h
}

fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    let log_bt = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b)
        + a * x.ln()
        + b * (-x).ln_1p();
    let bt = log_bt.exp();

    if x < (a + 1.0) / (a + b + 2.0) {
        (bt * beta_continued_fraction(a, b, x) / a).clamp(0.0, 1.0)
    } else {
        (1.0 - bt * beta_continued_fraction(b, a, 1.0 - x) / b).clamp(0.0, 1.0)
    }
}

// One-sided exact Clopper-Pearson lower confidence limit for a binomial
// success probability. For k successes from n independent seeds:
// Beta^{-1}(alpha; k, n-k+1). k=0 has lower bound 0.
fn clopper_pearson_lower(successes: usize, trials: usize, alpha: f64) -> f64 {
    if trials == 0 || successes == 0 {
        return 0.0;
    }
    if successes > trials || !(0.0 < alpha && alpha < 1.0) {
        return f64::NAN;
    }

    let a = successes as f64;
    let b = (trials - successes + 1) as f64;
    let mut low = 0.0;
    let mut high = 1.0;
    for _ in 0..100 {
        let mid = (low + high) * 0.5;
        if regularized_incomplete_beta(mid, a, b) < alpha {
            low = mid;
        } else {
            high = mid;
        }
    }
    (low + high) * 0.5
}

fn reliability(
    accuracies: &[f64],
    target_accuracy: f64,
    alpha: f64,
) -> Result<(usize, f64), Box<dyn Error>> {
    if accuracies.is_empty() {
        return Err("cannot compute seed reliability from zero seeds".into());
    }
    let successes = accuracies
        .iter()
        .filter(|&&accuracy| accuracy >= target_accuracy)
        .count();
    Ok((
        successes,
        clopper_pearson_lower(successes, accuracies.len(), alpha),
    ))
}

fn minimum_seeds_for_zero_failures(alpha: f64, target_reliability: f64) -> usize {
    (alpha.ln() / target_reliability.ln()).ceil() as usize
}

fn seed_set(data: &CohortData) -> BTreeSet<u64> {
    data.seed_accuracy.keys().copied().collect()
}

fn horizon_from_bits(bits: u64) -> f64 {
    f64::from_bits(bits)
}

fn analyze(
    cohorts: &BTreeMap<CohortKey, CohortData>,
    min_seeds: usize,
) -> Result<Vec<OutputRecord>, Box<dyn Error>> {
    let mut records = Vec::new();

    for (key, data) in cohorts {
        let accuracies: Vec<f64> = data.seed_accuracy.values().copied().collect();
        let (passing_95, lower_95) =
            reliability(&accuracies, K95_PER_SEED_ACCURACY, ALPHA)?;
        let (passing_90, lower_90) =
            reliability(&accuracies, T90_PER_SEED_ACCURACY, ALPHA)?;
        let count = accuracies.len();
        let mean_accuracy = accuracies.iter().sum::<f64>() / count as f64;
        let min_accuracy = accuracies.iter().copied().fold(f64::INFINITY, f64::min);
        let max_accuracy = accuracies.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        records.push(OutputRecord::CohortReliability(CohortReliabilityRecord {
            schema_version: SCHEMA_VERSION,
            source_revision: key.source_revision.clone(),
            dimension: key.dimension,
            memory_fan_in: key.memory_fan_in,
            condition: key.condition.clone(),
            gap_profile: key.gap_profile.clone(),
            horizon_seconds: horizon_from_bits(key.horizon_bits),
            event_count: key.event_count,
            cohort_ids: data.cohort_ids.iter().cloned().collect(),
            independent_seed_count: count,
            min_seed_count_gate: min_seeds,
            mean_seed_accuracy: mean_accuracy,
            min_seed_accuracy: min_accuracy,
            max_seed_accuracy: max_accuracy,
            seeds_meeting_95_percent_accuracy: passing_95,
            seed_reliability_lower_95_for_95_percent_accuracy: lower_95,
            seeds_meeting_90_percent_accuracy: passing_90,
            seed_reliability_lower_95_for_90_percent_accuracy: lower_90,
            chance_accuracy: data.chance_accuracy.unwrap_or(0.0),
            confirmatory_sample_gate_met: count >= min_seeds,
        }));
    }

    let mut k95_families: BTreeMap<K95FamilyKey, BTreeMap<usize, &CohortData>> =
        BTreeMap::new();
    let mut t90_families: BTreeMap<T90FamilyKey, BTreeMap<u64, &CohortData>> =
        BTreeMap::new();

    for (key, data) in cohorts {
        let k_key = K95FamilyKey {
            source_revision: key.source_revision.clone(),
            dimension: key.dimension,
            condition: key.condition.clone(),
            gap_profile: key.gap_profile.clone(),
            horizon_bits: key.horizon_bits,
            event_count: key.event_count,
        };
        k95_families
            .entry(k_key)
            .or_default()
            .insert(key.memory_fan_in, data);

        if key.horizon_bits != 0
            && matches!(key.condition.as_str(), "hdc_ltc_zero_input" | "hdc_ltc_distractor")
        {
            let t_key = T90FamilyKey {
                source_revision: key.source_revision.clone(),
                dimension: key.dimension,
                condition: key.condition.clone(),
                gap_profile: key.gap_profile.clone(),
                memory_fan_in: key.memory_fan_in,
                event_count: key.event_count,
            };
            t90_families
                .entry(t_key)
                .or_default()
                .insert(key.horizon_bits, data);
        }
    }

    for (family, levels) in k95_families {
        let tested_levels: Vec<usize> = levels.keys().copied().collect();
        let tested_level_count = levels.len();
        let seed_sets: Vec<BTreeSet<u64>> = levels.values().map(|data| seed_set(data)).collect();
        let common_seeds = seed_sets.first().cloned().unwrap_or_default();
        let paired = seed_sets.iter().all(|set| *set == common_seeds);
        let familywise_alpha = ALPHA;
        let per_level_alpha = familywise_alpha / tested_level_count as f64;
        let minimum_seeds_if_all_pass =
            minimum_seeds_for_zero_failures(per_level_alpha, REQUIRED_SEED_RELIABILITY);
        let required_seed_count = min_seeds.max(minimum_seeds_if_all_pass);
        let sample_gate = common_seeds.len() >= required_seed_count && tested_level_count >= 2;
        let mut qualified_k95 = None;

        if paired && sample_gate {
            for (fan_in, data) in &levels {
                let accuracies: Vec<f64> = common_seeds
                    .iter()
                    .map(|seed| data.seed_accuracy[seed])
                    .collect();
                let (_, lower) = reliability(&accuracies, K95_PER_SEED_ACCURACY, per_level_alpha)?;
                if lower >= REQUIRED_SEED_RELIABILITY {
                    qualified_k95 = Some(*fan_in);
                }
            }
        }

        records.push(OutputRecord::K95(K95Record {
            schema_version: SCHEMA_VERSION,
            source_revision: family.source_revision,
            dimension: family.dimension,
            condition: family.condition,
            gap_profile: family.gap_profile,
            horizon_seconds: horizon_from_bits(family.horizon_bits),
            event_count: family.event_count,
            target_per_seed_accuracy: K95_PER_SEED_ACCURACY,
            target_seed_reliability: REQUIRED_SEED_RELIABILITY,
            familywise_confidence_level: 1.0 - familywise_alpha,
            per_level_alpha,
            tested_level_count,
            multiplicity_correction: "Bonferroni within the fixed D/condition/gap/horizon/event family",
            minimum_seeds_if_all_pass,
            tested_fan_in_levels: tested_levels,
            common_seed_count: common_seeds.len(),
            paired_seed_sets: paired,
            confirmatory_sample_gate_met: sample_gate && paired,
            qualified_k95,
            interpretation: "maximum tested fan-in meeting the per-seed accuracy and seed-reliability targets using one-sided exact Clopper-Pearson lower bounds with Bonferroni family-wise alpha; null means no tested point qualified",
        }));
    }

    for (family, levels) in t90_families {
        let mut tested: Vec<(f64, u64)> = levels
            .keys()
            .map(|&bits| (horizon_from_bits(bits), bits))
            .collect();
        tested.sort_by(|a, b| a.0.total_cmp(&b.0));
        let tested_horizons: Vec<f64> = tested.iter().map(|(horizon, _)| *horizon).collect();
        let tested_level_count = levels.len();
        let seed_sets: Vec<BTreeSet<u64>> = levels.values().map(|data| seed_set(data)).collect();
        let common_seeds = seed_sets.first().cloned().unwrap_or_default();
        let paired = seed_sets.iter().all(|set| *set == common_seeds);
        let familywise_alpha = ALPHA;
        let per_level_alpha = familywise_alpha / tested_level_count as f64;
        let minimum_seeds_if_all_pass =
            minimum_seeds_for_zero_failures(per_level_alpha, REQUIRED_SEED_RELIABILITY);
        let required_seed_count = min_seeds.max(minimum_seeds_if_all_pass);
        let sample_gate = common_seeds.len() >= required_seed_count && tested_level_count >= 2;
        let mut qualified_t90_seconds = None;

        if paired && sample_gate {
            for (horizon, bits) in &tested {
                let data = levels[bits];
                let accuracies: Vec<f64> = common_seeds
                    .iter()
                    .map(|seed| data.seed_accuracy[seed])
                    .collect();
                let (_, lower) = reliability(&accuracies, T90_PER_SEED_ACCURACY, per_level_alpha)?;
                if lower >= REQUIRED_SEED_RELIABILITY {
                    qualified_t90_seconds = Some(*horizon);
                }
            }
        }

        records.push(OutputRecord::T90(T90Record {
            schema_version: SCHEMA_VERSION,
            source_revision: family.source_revision,
            dimension: family.dimension,
            condition: family.condition,
            gap_profile: family.gap_profile,
            memory_fan_in: family.memory_fan_in,
            event_count: family.event_count,
            target_per_seed_accuracy: T90_PER_SEED_ACCURACY,
            target_seed_reliability: REQUIRED_SEED_RELIABILITY,
            familywise_confidence_level: 1.0 - familywise_alpha,
            per_level_alpha,
            tested_level_count,
            multiplicity_correction: "Bonferroni within the fixed D/condition/gap/fan-in/event family",
            minimum_seeds_if_all_pass,
            tested_horizon_seconds: tested_horizons,
            common_seed_count: common_seeds.len(),
            paired_seed_sets: paired,
            confirmatory_sample_gate_met: sample_gate && paired,
            qualified_t90_seconds,
            interpretation: "maximum tested horizon meeting the per-seed accuracy and seed-reliability targets using one-sided exact Clopper-Pearson lower bounds with Bonferroni family-wise alpha; null means no tested point qualified",
        }));
    }

    Ok(records)
}

fn write_records(writer: &mut impl Write, records: &[OutputRecord]) -> Result<(), Box<dyn Error>> {
    for record in records {
        serde_json::to_writer(&mut *writer, record)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let cohorts = load_summaries(&args.input)?;
    let records = analyze(&cohorts, args.min_seeds)?;

    let output: Box<dyn Write> = match args.output.as_deref() {
        Some(path) if path != "-" => Box::new(File::create(path)?),
        _ => Box::new(io::stdout()),
    };
    let mut writer = BufWriter::new(output);
    write_records(&mut writer, &records)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clopper_pearson_all_successes_requires_about_59_seeds_for_95_95() {
        let thirty = clopper_pearson_lower(30, 30, ALPHA);
        let fifty_nine = clopper_pearson_lower(59, 59, ALPHA);
        let two_level_alpha = ALPHA / 2.0;
        assert!((thirty - 0.904966).abs() < 1.0e-5);
        assert!(thirty < REQUIRED_SEED_RELIABILITY);
        assert!(fifty_nine >= REQUIRED_SEED_RELIABILITY);
        assert_eq!(minimum_seeds_for_zero_failures(ALPHA, 0.95), 59);
        assert_eq!(minimum_seeds_for_zero_failures(two_level_alpha, 0.95), 72);
        assert_eq!(minimum_seeds_for_zero_failures(ALPHA / 3.0, 0.95), 80);
    }

    #[test]
    fn clopper_pearson_known_mixed_sample_is_bounded() {
        let lower = clopper_pearson_lower(57, 60, ALPHA);
        assert!(lower > 0.80 && lower < 0.95);
        assert_eq!(clopper_pearson_lower(0, 60, ALPHA), 0.0);
    }

    #[test]
    fn paired_seed_sets_require_exact_identity() {
        let a: BTreeSet<u64> = [0, 1, 2, 3].into_iter().collect();
        let b: BTreeSet<u64> = [0, 1, 2, 3].into_iter().collect();
        let c: BTreeSet<u64> = [0, 1, 2, 4].into_iter().collect();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn analysis_rejects_underpowered_and_qualifies_paired_familywise_frontiers() {
        let revision = "a".repeat(40);
        let mut cohorts = BTreeMap::new();

        for fan_in in [4usize, 8] {
            for horizon in [1.0f64, 2.0] {
                let key = CohortKey {
                    source_revision: Some(revision.clone()),
                    dimension: 1_024,
                    memory_fan_in: fan_in,
                    condition: "hdc_ltc_zero_input".to_owned(),
                    gap_profile: "irregular".to_owned(),
                    horizon_bits: horizon.to_bits(),
                    event_count: 4,
                };
                let mut data = CohortData {
                    chance_accuracy: Some(1.0 / fan_in as f64),
                    ..CohortData::default()
                };
                for seed in 0..59u64 {
                    data.seed_accuracy.insert(seed, 1.0);
                    data.cohort_ids.insert(format!("seed-{seed}"));
                }
                cohorts.insert(key, data);
            }
        }

        // 59/59 succeeds for one pointwise interval, but must not qualify a
        // two-level family after Bonferroni halves alpha (72 are required).
        let exploratory = analyze(&cohorts, 59).expect("exploratory frontier should analyze");
        assert!(exploratory.iter().all(|record| match record {
            OutputRecord::K95(row) => row.qualified_k95.is_none(),
            OutputRecord::T90(row) => row.qualified_t90_seconds.is_none(),
            OutputRecord::CohortReliability(_) => true,
        }));

        for data in cohorts.values_mut() {
            for seed in 59..80u64 {
                data.seed_accuracy.insert(seed, 1.0);
                data.cohort_ids.insert(format!("seed-{seed}"));
            }
        }

        let records = analyze(&cohorts, 59).expect("paired Bonferroni-adjusted frontier should analyze");
        let k95 = records.iter().find_map(|record| match record {
            OutputRecord::K95(row)
                if row.dimension == 1_024
                    && row.horizon_seconds == 1.0
                    && row.condition == "hdc_ltc_zero_input" =>
            {
                row.qualified_k95
            }
            _ => None,
        });
        let t90 = records.iter().find_map(|record| match record {
            OutputRecord::T90(row)
                if row.dimension == 1_024
                    && row.memory_fan_in == 4
                    && row.condition == "hdc_ltc_zero_input" =>
            {
                row.qualified_t90_seconds
            }
            _ => None,
        });

        assert_eq!(k95, Some(8));
        assert_eq!(t90, Some(2.0));
        let required = records.iter().find_map(|record| match record {
            OutputRecord::K95(row) if row.horizon_seconds == 1.0 => {
                Some(row.minimum_seeds_if_all_pass)
            }
            _ => None,
        });
        assert_eq!(required, Some(72));
    }

    #[test]
    fn horizon_bits_round_trip_exactly() {
        let horizon = 0.123_456_789_f64;
        assert_eq!(horizon_from_bits(horizon.to_bits()), horizon);
    }
}
