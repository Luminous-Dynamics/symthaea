//! Deterministic finite-horizon viable-action frontier across operating states.
//!
//! Usage: allostatic-multistep-viability-bench <raw.csv>
//! Synthetic results test search contracts and runtime accounting, not empirical
//! safety or performance on a physical system.

use allostatic_control_bench::multistep_viability::{
    evaluate_horizon_action_set_with_budget, HorizonSearchStats, HorizonViableActionSet,
    MAX_EXPANDED_SEARCH_NODES,
};
use allostatic_control_bench::robust_viability::{
    evaluate_action_set, ActionEffectBounds, DynamicsBounds, Interval, StateEnvelope,
    ViabilityLimits,
};
use allostatic_control_bench::Action;
use std::error::Error;
use std::fs;
use std::io::{Error as IoError, ErrorKind};

const HORIZONS: [usize; 5] = [1, 2, 3, 5, 8];
const PROFILE_COUNT: u64 = 24;
const HEADER: &str = "operating_point,profile_seed,horizon,push_certified,maintain_certified,emergency_recover_certified,certified_action_count,empty_viable_set,expanded_nodes,transition_evaluations,max_depth_reached";

#[derive(Clone, Copy, Debug)]
struct OperatingPoint {
    name: &'static str,
    reserve: f64,
    load: f64,
}

const OPERATING_POINTS: [OperatingPoint; 5] = [
    OperatingPoint {
        name: "critical_low_reserve_high_load",
        reserve: 0.22,
        load: 0.80,
    },
    OperatingPoint {
        name: "reserve_limited",
        reserve: 0.20,
        load: 0.45,
    },
    OperatingPoint {
        name: "load_limited",
        reserve: 0.50,
        load: 0.78,
    },
    OperatingPoint {
        name: "balanced",
        reserve: 0.40,
        load: 0.55,
    },
    OperatingPoint {
        name: "high_margin",
        reserve: 0.65,
        load: 0.30,
    },
];

fn deterministic_trit(seed: u64, dimension: u64) -> f64 {
    let mut value = seed.wrapping_add(
        dimension
            .wrapping_add(1)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15),
    );
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    [-1.0, 0.0, 1.0][(value % 3) as usize]
}

fn interval(lower: f64, upper: f64) -> Result<Interval, Box<dyn Error>> {
    Interval::new(lower, upper)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid interval").into())
}

fn shifted_effect(
    reserve: (f64, f64),
    load: (f64, f64),
    seed: u64,
    action_index: u64,
) -> Result<ActionEffectBounds, Box<dyn Error>> {
    let reserve_shift = deterministic_trit(seed, action_index * 2) * 0.008;
    let load_shift = deterministic_trit(seed, action_index * 2 + 1) * 0.010;
    let reserve_interval = interval(reserve.0 + reserve_shift, reserve.1 + reserve_shift)?;
    let load_interval = interval(load.0 + load_shift, load.1 + load_shift)?;
    ActionEffectBounds::new(reserve_interval, load_interval)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid action-effect envelope").into())
}

fn dynamics_for(seed: u64) -> Result<DynamicsBounds, Box<dyn Error>> {
    let push = shifted_effect((-0.030, -0.005), (0.030, 0.070), seed, 0)?;
    let maintain = shifted_effect((0.010, 0.040), (-0.010, 0.030), seed, 1)?;
    let emergency_recover = shifted_effect((0.025, 0.060), (-0.160, -0.080), seed, 2)?;
    Ok(DynamicsBounds::new(push, maintain, emergency_recover))
}

fn state_for(point: OperatingPoint) -> Result<StateEnvelope, Box<dyn Error>> {
    StateEnvelope::new(
        interval(point.reserve, point.reserve)?,
        interval(point.load, point.load)?,
    )
    .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid operating state").into())
}

fn same_action_set(left: HorizonViableActionSet, right: HorizonViableActionSet) -> bool {
    [Action::Push, Action::Maintain, Action::EmergencyRecover]
        .iter()
        .all(|action| left.contains(*action) == right.contains(*action))
}

fn subset_of(shorter: HorizonViableActionSet, longer: HorizonViableActionSet) -> bool {
    [Action::Push, Action::Maintain, Action::EmergencyRecover]
        .iter()
        .all(|action| !longer.contains(*action) || shorter.contains(*action))
}

fn csv_bool(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

fn row(
    output: &mut String,
    point: OperatingPoint,
    seed: u64,
    horizon: usize,
    actions: HorizonViableActionSet,
    stats: HorizonSearchStats,
) {
    output.push_str(&format!(
        "{},{},{},{},{},{},{},{},{},{},{}\n",
        point.name,
        seed,
        horizon,
        csv_bool(actions.push()),
        csv_bool(actions.maintain()),
        csv_bool(actions.emergency_recover()),
        actions.count(),
        csv_bool(actions.count() == 0),
        stats.expanded_nodes,
        stats.transition_evaluations,
        stats.max_depth_reached,
    ));
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            "usage: allostatic-multistep-viability-bench <raw.csv>",
        )
        .into());
    }

    let limits = ViabilityLimits::new(0.15, 0.88)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid viability limits"))?;
    let mut output = format!("{HEADER}\n");
    let mut row_count = 0_usize;

    for point in OPERATING_POINTS {
        let state = state_for(point)?;
        for seed in 0..PROFILE_COUNT {
            let dynamics = dynamics_for(seed)?;
            let one_step = evaluate_action_set(state, dynamics, limits, true);
            let mut previous: Option<HorizonViableActionSet> = None;

            for horizon in HORIZONS {
                let evaluation = evaluate_horizon_action_set_with_budget(
                    state,
                    dynamics,
                    limits,
                    true,
                    horizon,
                    MAX_EXPANDED_SEARCH_NODES,
                )
                .ok_or_else(|| {
                    IoError::new(
                        ErrorKind::InvalidData,
                        format!(
                            "horizon search did not finish within budget: state={} seed={seed} horizon={horizon}",
                            point.name
                        ),
                    )
                })?;

                if horizon == 1 && !same_action_set(evaluation.action_set, one_step) {
                    return Err(IoError::new(
                        ErrorKind::InvalidData,
                        format!("horizon-1 mismatch at operating point {}", point.name),
                    )
                    .into());
                }
                if previous.is_some_and(|shorter| !subset_of(shorter, evaluation.action_set)) {
                    return Err(IoError::new(
                        ErrorKind::InvalidData,
                        format!(
                            "horizon nesting failure: state={} seed={seed} horizon={horizon}",
                            point.name
                        ),
                    )
                    .into());
                }
                if evaluation.stats.expanded_nodes > MAX_EXPANDED_SEARCH_NODES
                    || evaluation.stats.transition_evaluations > MAX_EXPANDED_SEARCH_NODES
                {
                    return Err(IoError::new(
                        ErrorKind::InvalidData,
                        "search statistics exceed the declared per-evaluation budget",
                    )
                    .into());
                }

                row(
                    &mut output,
                    point,
                    seed,
                    horizon,
                    evaluation.action_set,
                    evaluation.stats,
                );
                row_count += 1;
                previous = Some(evaluation.action_set);
            }
        }
    }

    if row_count != 600 {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            format!("expected 600 horizon rows, generated {row_count}"),
        )
        .into());
    }
    fs::write(&args[1], output)?;
    Ok(())
}
