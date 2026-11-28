//! Paired F1 navigation campaign across identical procedurally generated maps.
//!
//! This is a zero-shot comparison harness, not a training run or performance
//! claim. Run from the adapter workspace with cargo run --release --example paired_f1.

use std::error::Error;
use std::time::{Duration, Instant};

use symthaea_gamebench::{
    Action, Cell, F1EpisodeReport, GreedyManhattanCandidate, GridSpec,
    InteractiveCandidate, Observation, Position, RandomLegalCandidate, evaluate_f1,
};
use symthaea_gamebench_fep_adapter::{
    FepNavigationCandidate, HdcLtcSpatialCandidate, HdcSpatialCandidate,
};

const SIDE: u16 = 7;
const MAP_COUNT: u64 = 16;
const ACTION_LIMIT: u32 = 196;

#[derive(Debug)]
struct EpisodeRecord {
    success: bool,
    candidate_protocol_valid: bool,
    actions: u32,
    optimal_actions: Option<u32>,
    excess_actions: Option<i64>,
    decision_latencies_ns: Vec<u128>,
    wall_time: Duration,
}

#[derive(Debug, Default)]
struct Aggregate {
    episodes: u64,
    successes: u64,
    protocol_valid: u64,
    total_actions: u64,
    successful_episodes: u64,
    total_excess_actions: i64,
    decision_latencies_ns: Vec<u128>,
    wall_times_ns: Vec<u128>,
    missing_optimum: u64,
}

fn make_map(seed: u64) -> GridSpec {
    let mut rng = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next_u64 = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };

    let goal = Position::new(SIDE - 1, SIDE - 1);
    let start = Position::new(0, 0);
    let mut cells = vec![Cell::Wall; usize::from(SIDE) * usize::from(SIDE)];
    let mut x = 0_u16;
    let mut y = 0_u16;
    cells[0] = Cell::Floor;

    // Carve a guaranteed monotone route. Additional random floor cells create
    // branches/dead ends; the exact evaluator computes each map's optimum.
    while x < goal.x || y < goal.y {
        let move_east = if x == goal.x {
            false
        } else if y == goal.y {
            true
        } else {
            next_u64() & 1 == 0
        };

        if move_east {
            x += 1;
        } else {
            y += 1;
        }
        let index = usize::from(y) * usize::from(SIDE) + usize::from(x);
        cells[index] = Cell::Floor;
    }

    for cell in &mut cells {
        if *cell == Cell::Wall && next_u64() % 100 < 13 {
            *cell = Cell::Floor;
        }
    }
    let goal_index = usize::from(goal.y) * usize::from(SIDE) + usize::from(goal.x);
    cells[goal_index] = Cell::Goal;

    GridSpec::new(SIDE, SIDE, cells, start, goal, ACTION_LIMIT)
        .expect("procedurally generated F1 map must be valid")
}

fn run_policy<C>(policy_name: &str, candidate: &mut C) -> Aggregate
where
    C: InteractiveCandidate<Observation, Action>,
    C::Error: Error,
{
    let mut aggregate = Aggregate::default();

    for map_seed in 1..=MAP_COUNT {
        let spec = make_map(map_seed);
        let started = Instant::now();
        let report: F1EpisodeReport = evaluate_f1(spec, candidate);
        let wall_time = started.elapsed();

        let record = EpisodeRecord {
            success: report.success,
            candidate_protocol_valid: report.candidate_protocol_valid,
            actions: report.actions_attempted,
            optimal_actions: report.optimal_actions,
            excess_actions: report.excess_actions,
            decision_latencies_ns: report.decision_latency_ns,
            wall_time,
        };

        aggregate.episodes += 1;
        aggregate.successes += u64::from(record.success);
        aggregate.protocol_valid += u64::from(record.candidate_protocol_valid);
        aggregate.total_actions += u64::from(record.actions);
        aggregate.decision_latencies_ns.extend(record.decision_latencies_ns.iter().copied());
        aggregate.wall_times_ns.push(record.wall_time.as_nanos());
        if record.success {
            aggregate.successful_episodes += 1;
            if let Some(excess) = record.excess_actions {
                aggregate.total_excess_actions += excess;
            }
        }
        if record.optimal_actions.is_none() {
            aggregate.missing_optimum += 1;
        }

        println!(
            "episode,{policy_name},{map_seed},{},{},{},{},{},{},{}",
            u8::from(record.success),
            u8::from(record.candidate_protocol_valid),
            record.actions,
            record.optimal_actions.map_or(String::from(""), |value| value.to_string()),
            record.excess_actions.map_or(String::from(""), |value| value.to_string()),
            percentile(&record.decision_latencies_ns, 0.50),
            percentile(&record.decision_latencies_ns, 0.95),
            record.wall_time.as_nanos(),
        );
    }

    aggregate
}

fn percentile(values: &[u128], fraction: f64) -> u128 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let index = ((sorted.len() - 1) as f64 * fraction).ceil() as usize;
    sorted[index.min(sorted.len() - 1)]
}

fn print_summary(name: &str, aggregate: &mut Aggregate) {
    let denominator = aggregate.episodes.max(1) as f64;
    let success_rate = aggregate.successes as f64 / denominator;
    let protocol_valid_rate = aggregate.protocol_valid as f64 / denominator;
    let mean_actions = aggregate.total_actions as f64 / denominator;
    let mean_excess = if aggregate.successful_episodes == 0 {
        f64::NAN
    } else {
        aggregate.total_excess_actions as f64 / aggregate.successful_episodes as f64
    };
    let p50_decision_ns = percentile(&aggregate.decision_latencies_ns, 0.50);
    let p95_decision_ns = percentile(&aggregate.decision_latencies_ns, 0.95);
    let p95_episode_ns = percentile(&aggregate.wall_times_ns, 0.95);

    println!(
        "summary,{name},{},{:.4},{:.4},{:.2},{:.3},{},{},{}",
        aggregate.episodes,
        success_rate,
        protocol_valid_rate,
        mean_actions,
        mean_excess,
        p50_decision_ns,
        p95_decision_ns,
        p95_episode_ns
    );
    if aggregate.missing_optimum > 0 {
        eprintln!(
            "warning: {name} had {} episodes without a reachable exact reference",
            aggregate.missing_optimum
        );
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("record,policy,map_seed,success,protocol_valid,actions,optimal_actions,excess_actions,p50_decision_ns,p95_decision_ns,episode_wall_ns");
    println!("summary_header,policy,episodes,success_rate,protocol_valid_rate,mean_actions,mean_excess_actions,p50_decision_ns,p95_decision_ns,p95_episode_wall_ns");

    let mut random = RandomLegalCandidate::new(0x5241_4E44_4F4D);
    let mut random_results = run_policy("random_legal", &mut random);
    let mut greedy = GreedyManhattanCandidate;
    let mut greedy_results = run_policy("greedy_manhattan", &mut greedy);
    let mut hdc = HdcSpatialCandidate::new(SIDE, SIDE, 0x4844_435F_5350)?;
    let mut hdc_results = run_policy("spatial_hdc", &mut hdc);
    let mut hdc_ltc = HdcLtcSpatialCandidate::new(SIDE, SIDE, 0x4844_435F_4C54)?;
    let mut hdc_ltc_results = run_policy("hdc_ltc", &mut hdc_ltc);
    let mut fep = FepNavigationCandidate::new(SIDE, SIDE, 0x4645_505F_4149)?;
    let mut fep_results = run_policy("fep_active_inference", &mut fep);

    print_summary("random_legal", &mut random_results);
    print_summary("greedy_manhattan", &mut greedy_results);
    print_summary("spatial_hdc", &mut hdc_results);
    print_summary("hdc_ltc", &mut hdc_ltc_results);
    print_summary("fep_active_inference", &mut fep_results);

    Ok(())
}
