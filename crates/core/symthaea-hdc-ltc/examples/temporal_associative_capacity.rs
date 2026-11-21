// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Temporal associative-retrieval capacity experiment.
//!
//! This is a research-only passive-retention benchmark, not a learned episodic
//! memory system. It measures whether a key/value bundle remains decodable
//! after the existing HDC-LTC neuron evolves under zero input or distractors.
//!
//! Run:
//!   cargo run --release --manifest-path crates/core/symthaea-hdc-ltc/Cargo.toml \
//!     --example temporal_associative_capacity -- --dims 1024,4096 --loads 4,8,16
//!
//! Output is JSON Lines: one row per query plus one summary row per cohort.
//! Raw query outcomes are retained so summaries can be independently rebuilt.

use serde::Serialize;
use std::error::Error;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::time::Instant;
use symthaea_hdc_ltc::{ContinuousHV, HdcLtcUnifiedNeuron, NeuronConfig};

const SCHEMA_VERSION: &str = "symthaea.temporal-associative-capacity/0.1";
const DEFAULT_DIMS: &[usize] = &[1_024, 4_096];
const DEFAULT_LOADS: &[usize] = &[4, 8, 16];
const DEFAULT_HORIZONS: &[f64] = &[0.0, 1.0, 10.0];
const DEFAULT_EVENTS: &[usize] = &[1, 8];
const DEFAULT_SEEDS: usize = 3;
const TIE_EPSILON: f32 = 1.0e-7;
const NEURON_MIN_DT_SECONDS: f32 = 1.0e-9;
const MAX_CODEBOOK_COORDINATES: usize = 8_388_608;
const MAX_EVENT_INPUT_COORDINATES: usize = 8_388_608;

#[derive(Debug)]
struct Args {
    dims: Vec<usize>,
    loads: Vec<usize>,
    horizons: Vec<f64>,
    events: Vec<usize>,
    seed_count: usize,
    output: Option<String>,
    source_revision: Option<String>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            dims: DEFAULT_DIMS.to_vec(),
            loads: DEFAULT_LOADS.to_vec(),
            horizons: DEFAULT_HORIZONS.to_vec(),
            events: DEFAULT_EVENTS.to_vec(),
            seed_count: DEFAULT_SEEDS,
            output: None,
            source_revision: None,
        }
    }
}

#[derive(Serialize)]
#[serde(tag = "record_type", rename_all = "snake_case")]
enum OutputRecord {
    Query(QueryRecord),
    Summary(SummaryRecord),
}

#[derive(Serialize)]
struct QueryRecord {
    schema_version: &'static str,
    source_revision: Option<String>,
    cohort_id: String,
    seed: u64,
    dimension: usize,
    memory_fan_in: usize,
    condition: &'static str,
    gap_profile: &'static str,
    horizon_seconds: f64,
    event_count: usize,
    query_index: usize,
    predicted_index: Option<usize>,
    correct: bool,
    tie: bool,
    valid: bool,
    target_similarity: Option<f32>,
    best_distractor_similarity: Option<f32>,
    target_margin: Option<f32>,
    chance_accuracy: f64,
}

#[derive(Serialize)]
struct SummaryRecord {
    schema_version: &'static str,
    source_revision: Option<String>,
    cohort_id: String,
    seed: u64,
    dimension: usize,
    memory_fan_in: usize,
    condition: &'static str,
    gap_profile: &'static str,
    horizon_seconds: f64,
    event_count: usize,
    gap_schedule_seconds: Vec<f64>,
    gap_coefficient_of_variation: Option<f64>,
    input_seeds: Vec<String>,
    query_count: usize,
    valid_query_count: usize,
    correct_count: usize,
    tie_count: usize,
    invalid_count: usize,
    accuracy: f64,
    chance_accuracy: f64,
    mean_target_similarity: Option<f64>,
    mean_best_distractor_similarity: Option<f64>,
    mean_target_margin: Option<f64>,
    plug_in_fano_information_lower_bound_bits: f64,
    initial_state_norm: f64,
    final_state_norm: f64,
    api_update_count: u64,
    reported_logical_seconds: f64,
    logical_horizon_error_seconds: f64,
    memory_preparation_wall_ns: u128,
    model_setup_wall_ns: u128,
    input_preparation_wall_ns: u128,
    evolution_wall_ns: u128,
    decode_wall_ns: u128,
    interpretation: &'static str,
}

struct PreparedMemory {
    keys: Vec<ContinuousHV>,
    values: Vec<ContinuousHV>,
    bundle: ContinuousHV,
    mismatched_bundle: ContinuousHV,
    random_query_keys: Vec<ContinuousHV>,
}

struct EvolutionOutcome {
    state: ContinuousHV,
    gap_schedule_seconds: Vec<f64>,
    gap_coefficient_of_variation: Option<f64>,
    input_seeds: Vec<String>,
    api_update_count: u64,
    reported_logical_seconds: f64,
    model_setup_wall_ns: u128,
    input_preparation_wall_ns: u128,
    evolution_wall_ns: u128,
}

struct Scores {
    predicted_index: Option<usize>,
    target_similarity: Option<f32>,
    best_distractor_similarity: Option<f32>,
    target_margin: Option<f32>,
    correct: bool,
    tie: bool,
    valid: bool,
}

#[derive(Default)]
struct Accumulator {
    valid_queries: usize,
    correct: usize,
    ties: usize,
    invalid: usize,
    target_similarity_sum: f64,
    distractor_similarity_sum: f64,
    margin_sum: f64,
}

fn parse_list<T>(value: &str, name: &str) -> Result<Vec<T>, Box<dyn Error>>
where
    T: std::str::FromStr,
    T::Err: Error + 'static,
{
    let parsed: Result<Vec<T>, T::Err> = value.split(',').map(str::parse).collect();
    let values = parsed?;
    if values.is_empty() {
        return Err(format!("{name} must not be empty").into());
    }
    Ok(values)
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
                    "temporal_associative_capacity [--dims 1024,4096] [--loads 4,8,16] \
                     [--horizons 0,1,10] [--events 1,8] [--seeds 3] \
                     [--source-revision <40-or-64-char-git-sha>] [--out results.jsonl]"
                );
                std::process::exit(0);
            }
            "--dims" | "--loads" | "--horizons" | "--events" | "--seeds" | "--out" | "--source-revision" => {
                iter.next()
                    .ok_or_else(|| format!("missing value for {flag}"))?
            }
            _ => return Err(format!("unknown argument: {flag}").into()),
        };

        match flag.as_str() {
            "--dims" => args.dims = parse_list(&value, "dimensions")?,
            "--loads" => args.loads = parse_list(&value, "memory loads")?,
            "--horizons" => args.horizons = parse_list(&value, "horizons")?,
            "--events" => args.events = parse_list(&value, "event counts")?,
            "--seeds" => args.seed_count = value.parse()?,
            "--out" => args.output = Some(value),
            "--source-revision" => args.source_revision = Some(value),
            _ => unreachable!("flag was validated above"),
        }
    }

    if args.dims.iter().any(|&d| d < 8 || d > 262_144) {
        return Err("dimensions must be in 8..=262144".into());
    }
    if args.loads.iter().any(|&n| n < 2 || n > 512) {
        return Err("memory loads must be in 2..=512".into());
    }
    if args.horizons.iter().any(|t| !t.is_finite() || *t < 0.0) {
        return Err("horizons must be finite and non-negative".into());
    }
    if args.events.iter().any(|&n| n == 0 || n > 4_096) {
        return Err("event counts must be in 1..=4096".into());
    }
    if args.seed_count == 0 || args.seed_count > 10_000 {
        return Err("seed count must be in 1..=10000".into());
    }
    if args
        .source_revision
        .as_deref()
        .is_some_and(|revision| !valid_git_revision(revision))
    {
        return Err("source revision must be a 40- or 64-character hexadecimal Git SHA".into());
    }
    // The experiment holds several full codebooks concurrently, plus an
    // optional distractor input schedule. Bound both allocation surfaces.
    if args.dims.iter().any(|&dim| {
        args.loads
            .iter()
            .any(|&load| dim.saturating_mul(load) > MAX_CODEBOOK_COORDINATES)
    }) {
        return Err("dimension × memory fan-in exceeds the codebook allocation envelope".into());
    }
    if args.dims.iter().any(|&dim| {
        args.events
            .iter()
            .any(|&events| dim.saturating_mul(events) > MAX_EVENT_INPUT_COORDINATES)
    }) {
        return Err("dimension × event count exceeds the distractor allocation envelope".into());
    }

    Ok(args)
}

fn derive_seed(root: u64, domain: u64, index: u64) -> u64 {
    root
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(domain.wrapping_mul(0xD1B5_4A32_D192_ED03))
        .wrapping_add(index.wrapping_mul(0x94D0_49BB_1331_11EB))
}

fn bipolar_random(dim: usize, seed: u64) -> ContinuousHV {
    let random = ContinuousHV::new_random(dim, seed);
    ContinuousHV::from_values(
        random
            .values
            .into_iter()
            .map(|value| if value >= 0.0 { 1.0 } else { -1.0 })
            .collect(),
    )
}

fn prepare_memory(dim: usize, fan_in: usize, seed: u64) -> PreparedMemory {
    let keys: Vec<ContinuousHV> = (0..fan_in)
        .map(|i| bipolar_random(dim, derive_seed(seed, 1, i as u64)))
        .collect();
    let values: Vec<ContinuousHV> = (0..fan_in)
        .map(|i| bipolar_random(dim, derive_seed(seed, 2, i as u64)))
        .collect();

    let bound: Vec<ContinuousHV> = keys
        .iter()
        .zip(values.iter())
        .map(|(key, value)| key.bind(value))
        .collect();
    let refs: Vec<&ContinuousHV> = bound.iter().collect();
    let bundle = ContinuousHV::bundle(&refs);

    // Cyclically mispair every key with the next value. This is a deterministic
    // negative control: the query keys are unchanged, but none names its target.
    let mismatched: Vec<ContinuousHV> = keys
        .iter()
        .enumerate()
        .map(|(index, key)| key.bind(&values[(index + 1) % fan_in]))
        .collect();
    let mismatched_refs: Vec<&ContinuousHV> = mismatched.iter().collect();
    let mismatched_bundle = ContinuousHV::bundle(&mismatched_refs);
    let random_query_keys: Vec<ContinuousHV> = (0..fan_in)
        .map(|i| bipolar_random(dim, derive_seed(seed, 6, i as u64)))
        .collect();

    PreparedMemory {
        keys,
        values,
        bundle,
        mismatched_bundle,
        random_query_keys,
    }
}

fn make_gaps(
    horizon_seconds: f64,
    event_count: usize,
    gap_profile: &'static str,
    seed: u64,
) -> Vec<f32> {
    if horizon_seconds == 0.0 || event_count == 0 {
        return Vec::new();
    }

    if gap_profile == "regular" {
        let dt = horizon_seconds / event_count as f64;
        let mut gaps = vec![dt as f32; event_count];
        let prefix: f64 = gaps[..event_count - 1].iter().map(|&v| v as f64).sum();
        gaps[event_count - 1] = (horizon_seconds - prefix) as f32;
        return gaps;
    }

    // Deterministic positive weights create irregular, causal intervals.
    let mut rng = derive_seed(seed, 3, event_count as u64);
    let mut weights = Vec::with_capacity(event_count);
    for _ in 0..event_count {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let unit = ((rng >> 40) as f64) / ((1_u64 << 24) as f64);
        weights.push(0.25 + unit);
    }
    let total_weight: f64 = weights.iter().sum();
    let mut gaps = Vec::with_capacity(event_count);
    let mut elapsed = 0.0_f64;
    for (index, weight) in weights.iter().enumerate() {
        let dt = if index + 1 == event_count {
            horizon_seconds - elapsed
        } else {
            horizon_seconds * weight / total_weight
        };
        gaps.push(dt as f32);
        elapsed += dt;
    }
    gaps
}

fn coefficient_of_variation(values: &[f32]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mean = values.iter().map(|&value| value as f64).sum::<f64>()
        / values.len() as f64;
    if !mean.is_finite() || mean <= 0.0 {
        return None;
    }
    let variance = values
        .iter()
        .map(|&value| {
            let delta = value as f64 - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64;
    Some(variance.sqrt() / mean)
}

fn evolve_state(
    dim: usize,
    memory: &ContinuousHV,
    seed: u64,
    horizon_seconds: f64,
    event_count: usize,
    gap_profile: &'static str,
    condition: &'static str,
) -> Result<EvolutionOutcome, Box<dyn Error>> {
    if condition == "static_hdc_bundle"
        || condition == "static_mismatched_pairing_control"
        || condition == "static_random_query_control"
        || horizon_seconds == 0.0
    {
        return Ok(EvolutionOutcome {
            state: memory.clone(),
            gap_schedule_seconds: Vec::new(),
            gap_coefficient_of_variation: None,
            input_seeds: Vec::new(),
            api_update_count: 0,
            reported_logical_seconds: 0.0,
            model_setup_wall_ns: 0,
            input_preparation_wall_ns: 0,
            evolution_wall_ns: 0,
        });
    }

    if condition != "hdc_ltc_zero_input" && condition != "hdc_ltc_distractor" {
        return Err(format!("unsupported evolution condition: {condition}").into());
    }
    if event_count == 0 {
        return Err("a positive horizon requires at least one evolution event".into());
    }

    let setup_start = Instant::now();
    let config = NeuronConfig {
        dim,
        ..NeuronConfig::default()
    };
    let mut neuron = HdcLtcUnifiedNeuron::new(config, derive_seed(seed, 4, 0));
    neuron.set_state(memory.clone());
    let model_setup_wall_ns = setup_start.elapsed().as_nanos();

    // Input generation is a distinct measured phase. This prevents random
    // distractor construction from being reported as neuron evolution cost.
    let input_prep_start = Instant::now();
    let zero_input = (condition == "hdc_ltc_zero_input")
        .then(|| ContinuousHV::new(dim));
    let gaps = make_gaps(horizon_seconds, event_count, gap_profile, seed);
    let input_seed_values: Vec<u64> = if condition == "hdc_ltc_distractor" {
        (0..event_count)
            .map(|index| derive_seed(seed, 5, index as u64))
            .collect()
    } else {
        Vec::new()
    };
    // Hex strings preserve all 64 seed bits in JSON consumers that use
    // IEEE-754 numbers (and therefore cannot exactly represent all u64s).
    let input_seeds: Vec<String> = input_seed_values
        .iter()
        .map(|seed| format!("{seed:016x}"))
        .collect();
    let distractors: Vec<ContinuousHV> = input_seed_values
        .iter()
        .map(|&input_seed| bipolar_random(dim, input_seed))
        .collect();
    let gap_schedule_seconds: Vec<f64> = gaps.iter().map(|&dt| dt as f64).collect();
    let gap_coefficient_of_variation = coefficient_of_variation(&gaps);
    let input_preparation_wall_ns = input_prep_start.elapsed().as_nanos();

    if gaps.len() != event_count
        || gaps
            .iter()
            .any(|dt| !dt.is_finite() || *dt < NEURON_MIN_DT_SECONDS)
    {
        return Err(
            "invalid time-gap schedule: every interval must be finite and at least 1e-9 seconds"
                .into(),
        );
    }
    if condition == "hdc_ltc_distractor" && distractors.len() != event_count {
        return Err("distractor input schedule does not match the interval count".into());
    }

    let start = Instant::now();
    for (index, dt) in gaps.iter().enumerate() {
        let input = if let Some(zero) = zero_input.as_ref() {
            zero
        } else {
            &distractors[index]
        };
        neuron.evolve_closed_form(*dt, input);
    }
    // Include the state snapshot copy in the evolution-to-decode handoff cost.
    let state = neuron.state().clone();
    let evolution_wall_ns = start.elapsed().as_nanos();

    Ok(EvolutionOutcome {
        state,
        gap_schedule_seconds,
        gap_coefficient_of_variation,
        input_seeds,
        api_update_count: neuron.update_count(),
        reported_logical_seconds: neuron.total_time(),
        model_setup_wall_ns,
        input_preparation_wall_ns,
        evolution_wall_ns,
    })
}

fn decode(
    memory: &PreparedMemory,
    state: &ContinuousHV,
    query_index: usize,
    query_key: &ContinuousHV,
) -> Scores {
    let query = query_key.bind(state);
    let mut best_index = None;
    let mut best_score = f32::NEG_INFINITY;
    let mut second_best = f32::NEG_INFINITY;
    let mut best_distractor = f32::NEG_INFINITY;

    for (candidate_index, candidate) in memory.values.iter().enumerate() {
        let score = query.similarity(candidate);
        if !score.is_finite() {
            return Scores {
                predicted_index: None,
                target_similarity: None,
                best_distractor_similarity: None,
                target_margin: None,
                correct: false,
                tie: false,
                valid: false,
            };
        }

        if candidate_index != query_index && score > best_distractor {
            best_distractor = score;
        }
        if score > best_score {
            second_best = best_score;
            best_score = score;
            best_index = Some(candidate_index);
        } else if score > second_best {
            second_best = score;
        }
    }

    let target_similarity = query.similarity(&memory.values[query_index]);
    let tie = (best_score - second_best).abs() <= TIE_EPSILON;
    let target_margin = target_similarity - best_distractor;
    let valid = target_similarity.is_finite()
        && best_distractor.is_finite()
        && target_margin.is_finite();
    let correct = valid && best_index == Some(query_index) && !tie;

    Scores {
        predicted_index: best_index,
        target_similarity: valid.then_some(target_similarity),
        best_distractor_similarity: valid.then_some(best_distractor),
        target_margin: valid.then_some(target_margin),
        correct,
        tie,
        valid,
    }
}

fn binary_entropy(probability: f64) -> f64 {
    if probability <= 0.0 || probability >= 1.0 {
        0.0
    } else {
        -probability * probability.log2() - (1.0 - probability) * (1.0 - probability).log2()
    }
}

fn plug_in_fano_lower_bound_bits(error_rate: f64, classes: usize) -> f64 {
    if classes <= 1 {
        return 0.0;
    }
    let log_classes = (classes as f64).log2();
    let conditional_entropy_bound =
        binary_entropy(error_rate) + error_rate * ((classes - 1) as f64).log2();
    (log_classes - conditional_entropy_bound).max(0.0)
}

fn cohort_id(
    seed: u64,
    dimension: usize,
    fan_in: usize,
    condition: &'static str,
    gap_profile: &'static str,
    horizon_seconds: f64,
    event_count: usize,
) -> String {
    format!(
        "s{seed}-d{dimension}-l{fan_in}-{condition}-{gap_profile}-t{:016x}-e{event_count}",
        horizon_seconds.to_bits()
    )
}

fn run_cohort(
    writer: &mut impl Write,
    source_revision: Option<&str>,
    memory: &PreparedMemory,
    memory_preparation_wall_ns: u128,
    seed: u64,
    dim: usize,
    fan_in: usize,
    horizon_seconds: f64,
    requested_events: usize,
    gap_profile: &'static str,
    condition: &'static str,
) -> Result<(), Box<dyn Error>> {
    let event_count = if horizon_seconds == 0.0
        || condition == "static_hdc_bundle"
        || condition == "static_mismatched_pairing_control"
        || condition == "static_random_query_control"
    {
        0
    } else {
        requested_events
    };
    let id = cohort_id(
        seed,
        dim,
        fan_in,
        condition,
        gap_profile,
        horizon_seconds,
        event_count,
    );

    let initial_bundle = if condition == "static_mismatched_pairing_control" {
        &memory.mismatched_bundle
    } else {
        &memory.bundle
    };
    let evolution = evolve_state(
        dim,
        initial_bundle,
        seed,
        horizon_seconds,
        event_count,
        gap_profile,
        condition,
    )?;
    let state = &evolution.state;

    let decode_start = Instant::now();
    let mut accumulator = Accumulator::default();
    let mut query_results = Vec::with_capacity(fan_in);

    for query_index in 0..fan_in {
        let query_key = if condition == "static_random_query_control" {
            &memory.random_query_keys[query_index]
        } else {
            &memory.keys[query_index]
        };
        let scores = decode(memory, state, query_index, query_key);
        if scores.valid {
            accumulator.valid_queries += 1;
            accumulator.target_similarity_sum +=
                scores.target_similarity.expect("valid score has target") as f64;
            accumulator.distractor_similarity_sum += scores
                .best_distractor_similarity
                .expect("valid score has distractor") as f64;
            accumulator.margin_sum +=
                scores.target_margin.expect("valid score has margin") as f64;
        } else {
            accumulator.invalid += 1;
        }
        if scores.correct {
            accumulator.correct += 1;
        }
        if scores.tie {
            accumulator.ties += 1;
        }
        query_results.push((query_index, scores));
    }

    let decode_wall_ns = decode_start.elapsed().as_nanos();

    for (query_index, scores) in query_results {
        write_record(
            writer,
            &OutputRecord::Query(QueryRecord {
                schema_version: SCHEMA_VERSION,
                source_revision: source_revision.map(str::to_owned),
                cohort_id: id.clone(),
                seed,
                dimension: dim,
                memory_fan_in: fan_in,
                condition,
                gap_profile,
                horizon_seconds,
                event_count,
                query_index,
                predicted_index: scores.predicted_index,
                correct: scores.correct,
                tie: scores.tie,
                valid: scores.valid,
                target_similarity: scores.target_similarity,
                best_distractor_similarity: scores.best_distractor_similarity,
                target_margin: scores.target_margin,
                chance_accuracy: 1.0 / fan_in as f64,
            }),
        )?;
    }

    let accuracy = accumulator.correct as f64 / fan_in as f64;
    let mean = |sum: f64| {
        (accumulator.valid_queries > 0)
            .then_some(sum / accumulator.valid_queries as f64)
    };

    write_record(
        writer,
        &OutputRecord::Summary(SummaryRecord {
            schema_version: SCHEMA_VERSION,
            source_revision: source_revision.map(str::to_owned),
            cohort_id: id,
            seed,
            dimension: dim,
            memory_fan_in: fan_in,
            condition,
            gap_profile,
            horizon_seconds,
            event_count,
            gap_schedule_seconds: evolution.gap_schedule_seconds,
            gap_coefficient_of_variation: evolution.gap_coefficient_of_variation,
            input_seeds: evolution.input_seeds,
            query_count: fan_in,
            valid_query_count: accumulator.valid_queries,
            correct_count: accumulator.correct,
            tie_count: accumulator.ties,
            invalid_count: accumulator.invalid,
            accuracy,
            chance_accuracy: 1.0 / fan_in as f64,
            mean_target_similarity: mean(accumulator.target_similarity_sum),
            mean_best_distractor_similarity: mean(accumulator.distractor_similarity_sum),
            mean_target_margin: mean(accumulator.margin_sum),
            plug_in_fano_information_lower_bound_bits: plug_in_fano_lower_bound_bits(
                1.0 - accuracy,
                fan_in,
            ),
            initial_state_norm: initial_bundle.norm() as f64,
            final_state_norm: state.norm() as f64,
            api_update_count: evolution.api_update_count,
            reported_logical_seconds: evolution.reported_logical_seconds,
            logical_horizon_error_seconds: (evolution.reported_logical_seconds - horizon_seconds).abs(),
            memory_preparation_wall_ns,
            model_setup_wall_ns: evolution.model_setup_wall_ns,
            input_preparation_wall_ns: evolution.input_preparation_wall_ns,
            evolution_wall_ns: evolution.evolution_wall_ns,
            decode_wall_ns,
            interpretation: "finite-sample plug-in Fano diagnostic; not a confidence-certified information bound",
        }),
    )?;
    Ok(())
}

fn write_record(writer: &mut impl Write, record: &OutputRecord) -> Result<(), Box<dyn Error>> {
    serde_json::to_writer(&mut *writer, record)?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let output: Box<dyn Write> = match args.output.as_deref() {
        Some(path) if path != "-" => Box::new(File::create(path)?),
        _ => Box::new(io::stdout()),
    };
    let mut writer = BufWriter::new(output);

    for &dim in &args.dims {
        for &fan_in in &args.loads {
            for seed_index in 0..args.seed_count {
                let seed = seed_index as u64;
                let memory_prep_start = Instant::now();
                let memory = prepare_memory(dim, fan_in, seed);
                let memory_preparation_wall_ns = memory_prep_start.elapsed().as_nanos();

                // Static controls are emitted once per corpus, not duplicated
                // for each horizon or event schedule.
                for condition in [
                    "static_hdc_bundle",
                    "static_mismatched_pairing_control",
                    "static_random_query_control",
                ] {
                    run_cohort(
                        &mut writer,
                        args.source_revision.as_deref(),
                        &memory,
                        memory_preparation_wall_ns,
                        seed,
                        dim,
                        fan_in,
                        0.0,
                        0,
                        "not_applicable",
                        condition,
                    )?;
                }

                // The zero-horizon HDC-LTC path must be an exact identity
                // shortcut: same state, zero updates and same decode as static.
                run_cohort(
                    &mut writer,
                    args.source_revision.as_deref(),
                    &memory,
                    memory_preparation_wall_ns,
                    seed,
                    dim,
                    fan_in,
                    0.0,
                    0,
                    "not_applicable",
                    "hdc_ltc_zero_input",
                )?;

                for &horizon_seconds in args.horizons.iter().filter(|&&h| h > 0.0) {
                    for &events in &args.events {
                        for gap_profile in ["regular", "irregular"].into_iter().take(
                            if events > 1 { 2 } else { 1 },
                        ) {
                            for condition in [
                                "hdc_ltc_zero_input",
                                "hdc_ltc_distractor",
                            ] {
                                run_cohort(
                                    &mut writer,
                                    args.source_revision.as_deref(),
                                    &memory,
                                    memory_preparation_wall_ns,
                                    seed,
                                    dim,
                                    fan_in,
                                    horizon_seconds,
                                    events,
                                    gap_profile,
                                    condition,
                                )?;
                            }
                        }
                    }
                }
            }
        }
    }

    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn git_revision_validation_accepts_only_full_sha_lengths() {
        assert!(valid_git_revision(&"a".repeat(40)));
        assert!(valid_git_revision(&"b".repeat(64)));
        assert!(!valid_git_revision(&"c".repeat(39)));
        assert!(!valid_git_revision(&"g".repeat(40)));
    }

    #[test]
    fn seeded_codebooks_and_bundles_are_reproducible() {
        let first = prepare_memory(1_024, 8, 42);
        let second = prepare_memory(1_024, 8, 42);

        assert_eq!(first.keys, second.keys);
        assert_eq!(first.values, second.values);
        assert_eq!(first.bundle, second.bundle);
        assert_eq!(first.mismatched_bundle, second.mismatched_bundle);
        assert_eq!(first.random_query_keys, second.random_query_keys);
    }

    #[test]
    fn negative_controls_are_separate_from_the_incumbent_bundle() {
        let memory = prepare_memory(1_024, 8, 7);

        assert_ne!(memory.bundle, memory.mismatched_bundle);
        assert!(memory
            .keys
            .iter()
            .zip(memory.random_query_keys.iter())
            .all(|(key, random)| key != random));
    }

    #[test]
    fn regular_and_irregular_gaps_are_positive_and_cover_the_horizon() {
        for profile in ["regular", "irregular"] {
            let gaps = make_gaps(12.75, 17, profile, 91);
            assert_eq!(gaps.len(), 17);
            assert!(gaps.iter().all(|dt| dt.is_finite() && *dt > 0.0));

            let elapsed: f64 = gaps.iter().map(|&dt| dt as f64).sum();
            assert!((elapsed - 12.75).abs() < 1.0e-5);
        }
    }

    #[test]
    fn cohort_ids_preserve_exact_horizon_identity() {
        let first = cohort_id(1, 1024, 8, "hdc_ltc_zero_input", "regular", 1.000_000_1, 4);
        let second = cohort_id(1, 1024, 8, "hdc_ltc_zero_input", "regular", 1.000_000_2, 4);
        assert_ne!(first, second);
    }

    #[test]
    fn gap_schedule_descriptors_are_replayable() {
        let regular = make_gaps(4.0, 4, "regular", 12);
        let irregular = make_gaps(4.0, 4, "irregular", 12);

        assert_eq!(coefficient_of_variation(&regular), Some(0.0));
        assert!(coefficient_of_variation(&irregular).expect("nonempty gaps") > 0.0);
    }

    #[test]
    fn zero_horizon_is_an_exact_identity_without_solver_updates() {
        let memory = prepare_memory(1_024, 4, 13);
        let outcome = evolve_state(
            1_024,
            &memory.bundle,
            13,
            0.0,
            0,
            "not_applicable",
            "hdc_ltc_zero_input",
        )
        .expect("zero-horizon identity path should be accepted");

        assert_eq!(outcome.state, memory.bundle);
        assert_eq!(outcome.api_update_count, 0);
        assert_eq!(outcome.reported_logical_seconds, 0.0);
        assert_eq!(outcome.evolution_wall_ns, 0);
    }

    #[test]
    fn hdc_ltc_post_step_norm_cap_is_reportable() {
        let memory = prepare_memory(256, 4, 23);
        let initial_norm = memory.bundle.norm();
        assert!(initial_norm > 5.0);

        // At the minimum accepted dt, state decay is negligible. The post-step
        // norm limit must therefore be the operation that bounds the state.
        let outcome = evolve_state(
            256,
            &memory.bundle,
            23,
            NEURON_MIN_DT_SECONDS as f64,
            1,
            "regular",
            "hdc_ltc_zero_input",
        )
        .expect("one minimum positive interval should evolve");

        assert_eq!(outcome.api_update_count, 1);
        assert!(
            (outcome.reported_logical_seconds - NEURON_MIN_DT_SECONDS as f64).abs() < 1.0e-15
        );
        assert!((outcome.state.norm() - 5.0).abs() < 1.0e-4);
    }

    #[test]
    fn static_recall_and_both_negative_controls_behave_as_expected() {
        const DIM: usize = 1_024;
        const FAN_IN: usize = 8;
        const SEEDS: u64 = 50;
        let mut static_correct = 0usize;
        let mut random_query_correct = 0usize;
        let mut mispaired_original_correct = 0usize;
        let total = SEEDS as usize * FAN_IN;

        for seed in 0..SEEDS {
            let memory = prepare_memory(DIM, FAN_IN, seed);
            for query_index in 0..FAN_IN {
                let expected_key = &memory.keys[query_index];

                let static_scores = decode(&memory, &memory.bundle, query_index, expected_key);
                static_correct += usize::from(static_scores.valid && static_scores.correct);

                let random_scores = decode(
                    &memory,
                    &memory.bundle,
                    query_index,
                    &memory.random_query_keys[query_index],
                );
                random_query_correct +=
                    usize::from(random_scores.valid && random_scores.correct);

                let mispaired_scores =
                    decode(&memory, &memory.mismatched_bundle, query_index, expected_key);
                mispaired_original_correct +=
                    usize::from(mispaired_scores.valid && mispaired_scores.correct);
            }
        }

        // This deterministic fixture is a positive control, not a universal
        // capacity claim. The random-query arm should be statistically near
        // 1/FAN_IN; the deliberately wrong association should not recover
        // the original semantic labels.
        assert_eq!(static_correct, total);
        let random_query_accuracy = random_query_correct as f64 / total as f64;
        assert!(
            (0.04..=0.22).contains(&random_query_accuracy),
            "random-query accuracy {random_query_accuracy:.3} is not near chance {}",
            1.0 / FAN_IN as f64
        );
        assert_eq!(mispaired_original_correct, 0);
    }

    #[test]
    fn evolution_rejects_intervals_below_the_neuron_time_floor() {
        let memory = prepare_memory(256, 4, 19);
        let result = evolve_state(
            256,
            &memory.bundle,
            19,
            1.0e-10,
            1,
            "regular",
            "hdc_ltc_zero_input",
        );
        assert!(result.is_err());
    }

    #[test]
    fn fano_diagnostic_is_finite_and_within_label_entropy() {
        for classes in [2, 4, 8, 16] {
            let maximum = (classes as f64).log2();
            for error_rate in [0.0, 0.1, 0.25, 0.5, 1.0] {
                let bound = plug_in_fano_lower_bound_bits(error_rate, classes);
                assert!(bound.is_finite());
                assert!((0.0..=maximum).contains(&bound));
            }
            assert_eq!(plug_in_fano_lower_bound_bits(0.0, classes), maximum);
            let always_wrong_bound =
                (maximum - ((classes - 1) as f64).log2()).max(0.0);
            assert!(
                (plug_in_fano_lower_bound_bits(1.0, classes) - always_wrong_bound).abs()
                    < 1.0e-12
            );
        }
    }
}
