//! Compare rectangular outer bounds with correlated zonotope certificates.
//!
//! Usage: allostatic-joint-uncertainty-bench <raw.csv> <summary.csv>
//! The fixtures use a synthetic coupled reserve/load safety rule. This is a
//! geometry/accounting benchmark, not a physical controller or an SLS-MPC
//! reproduction.

use allostatic_control_bench::joint_viability::{
    action_name, box_certifies, box_viability_margins, corner_oracle_safe,
    zonotope_certifies, zonotope_viability_margins, ActionEffectZonotope, Delta2D,
    JointViabilityLimits, StatePoint,
};
use allostatic_control_bench::Action;
use std::error::Error;
use std::fs;
use std::io::{Error as IoError, ErrorKind};

const PROFILE_COUNT: u64 = 64;
const ACTIONS: [Action; 3] = [Action::Push, Action::Maintain, Action::EmergencyRecover];
const RAW_HEADER: &str = "operating_point,profile_seed,action,state_reserve,state_load,center_reserve_delta,center_load_delta,generator_count,corner_count,box_certified,zonotope_certified,oracle_safe,box_false_certificate,zonotope_false_certificate,box_false_veto,zonotope_false_veto,box_minimum_slack,zonotope_minimum_slack";
const SUMMARY_HEADER: &str = "operating_point,action,profiles,oracle_safe_actions,box_certified_actions,zonotope_certified_actions,box_false_certificates,zonotope_false_certificates,box_false_vetoes,zonotope_false_vetoes,box_false_veto_rate,zonotope_false_veto_rate,box_action_availability,zonotope_action_availability,availability_gain_actions,mean_box_minimum_slack,mean_zonotope_minimum_slack";

#[derive(Clone, Copy, Debug)]
struct OperatingPoint {
    name: &'static str,
    reserve: f64,
    load: f64,
}

const OPERATING_POINTS: [OperatingPoint; 5] = [
    OperatingPoint {
        name: "critical_low_reserve_high_load",
        reserve: 0.32,
        load: 0.55,
    },
    OperatingPoint {
        name: "reserve_limited",
        reserve: 0.28,
        load: 0.45,
    },
    OperatingPoint {
        name: "load_limited",
        reserve: 0.45,
        load: 0.76,
    },
    OperatingPoint {
        name: "balanced",
        reserve: 0.48,
        load: 0.48,
    },
    OperatingPoint {
        name: "high_margin",
        reserve: 0.70,
        load: 0.28,
    },
];

#[derive(Clone, Copy, Debug, Default)]
struct Counters {
    profiles: usize,
    oracle_safe: usize,
    box_certified: usize,
    zonotope_certified: usize,
    box_false_certificates: usize,
    zonotope_false_certificates: usize,
    box_false_vetoes: usize,
    zonotope_false_vetoes: usize,
    sum_box_minimum_slack: f64,
    sum_zonotope_minimum_slack: f64,
}

impl Counters {
    fn observe(
        &mut self,
        oracle_safe: bool,
        box_certified: bool,
        zonotope_certified: bool,
        box_minimum_slack: f64,
        zonotope_minimum_slack: f64,
    ) {
        self.profiles += 1;
        self.oracle_safe += usize::from(oracle_safe);
        self.box_certified += usize::from(box_certified);
        self.zonotope_certified += usize::from(zonotope_certified);
        self.box_false_certificates += usize::from(box_certified && !oracle_safe);
        self.zonotope_false_certificates += usize::from(zonotope_certified && !oracle_safe);
        self.box_false_vetoes += usize::from(oracle_safe && !box_certified);
        self.zonotope_false_vetoes += usize::from(oracle_safe && !zonotope_certified);
        self.sum_box_minimum_slack += box_minimum_slack;
        self.sum_zonotope_minimum_slack += zonotope_minimum_slack;
    }

    fn accumulate(&mut self, other: Self) {
        self.profiles += other.profiles;
        self.oracle_safe += other.oracle_safe;
        self.box_certified += other.box_certified;
        self.zonotope_certified += other.zonotope_certified;
        self.box_false_certificates += other.box_false_certificates;
        self.zonotope_false_certificates += other.zonotope_false_certificates;
        self.box_false_vetoes += other.box_false_vetoes;
        self.zonotope_false_vetoes += other.zonotope_false_vetoes;
        self.sum_box_minimum_slack += other.sum_box_minimum_slack;
        self.sum_zonotope_minimum_slack += other.sum_zonotope_minimum_slack;
    }
}

fn deterministic_unit(seed: u64, lane: u64) -> f64 {
    let mut value = seed.wrapping_add(
        lane.wrapping_add(1)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15),
    );
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    (value >> 11) as f64 / ((1_u64 << 53) as f64)
}

fn deterministic_trit(seed: u64, lane: u64) -> f64 {
    let bucket = (deterministic_unit(seed, lane) * 5.0).floor() as i32;
    (bucket.clamp(0, 4) - 2) as f64 / 2.0
}

fn limits() -> JointViabilityLimits {
    JointViabilityLimits::new(0.15, 0.88, 0.5, 0.0)
        .expect("constant joint-viability limits are valid")
}

fn state_for(point: OperatingPoint, seed: u64) -> Result<StatePoint, Box<dyn Error>> {
    let reserve = point.reserve + deterministic_trit(seed, 501) * 0.01;
    let load = point.load + deterministic_trit(seed, 502) * 0.01;
    StatePoint::new(reserve, load)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid synthetic state").into())
}

fn action_center(action: Action) -> (f64, f64) {
    match action {
        Action::Push => (-0.005, 0.020),
        Action::Maintain => (0.015, -0.005),
        Action::EmergencyRecover => (0.025, -0.050),
    }
}

fn base_generator(action: Action) -> (f64, f64) {
    match action {
        Action::Push => (0.030, 0.060),
        Action::Maintain => (0.020, 0.040),
        Action::EmergencyRecover => (0.025, 0.050),
    }
}

fn effects_for(seed: u64, action: Action) -> Result<ActionEffectZonotope, Box<dyn Error>> {
    let action_index = match action {
        Action::Push => 0_u64,
        Action::Maintain => 1_u64,
        Action::EmergencyRecover => 2_u64,
    };
    let (base_center_reserve, base_center_load) = action_center(action);
    let center = Delta2D::new(
        base_center_reserve + deterministic_trit(seed, 100 + action_index) * 0.002,
        base_center_load + deterministic_trit(seed, 200 + action_index) * 0.004,
    )
    .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid action-effect center"))?;

    let (base_r, base_l) = base_generator(action);
    let generator_count = 1 + ((seed + action_index) % 3) as usize;
    let mut generators = Vec::with_capacity(generator_count);
    for index in 0..generator_count {
        let lane = 1_000 + action_index * 100 + index as u64 * 2;
        let reserve = base_r + deterministic_trit(seed, lane) * 0.003;
        let load = base_l + deterministic_trit(seed, lane + 1) * 0.006;
        generators.push(
            Delta2D::new(reserve, load)
                .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid generator"))?,
        );
    }

    ActionEffectZonotope::new(center, generators)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid action-effect zonotope").into())
}

fn csv_bool(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

fn raw_row(
    output: &mut String,
    point: OperatingPoint,
    seed: u64,
    action: Action,
    state: StatePoint,
    effects: &ActionEffectZonotope,
    box_certified: bool,
    zonotope_certified: bool,
    oracle_safe: bool,
    box_minimum_slack: f64,
    zonotope_minimum_slack: f64,
) {
    output.push_str(&format!(
        "{},{},{},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{},{},{},{},{},{:.9},{:.9}\n",
        point.name,
        seed,
        action_name(action),
        state.reserve(),
        state.load(),
        effects.center().reserve(),
        effects.center().load(),
        effects.generators().len(),
        effects.corner_count(),
        csv_bool(box_certified),
        csv_bool(zonotope_certified),
        csv_bool(oracle_safe),
        csv_bool(box_certified && !oracle_safe),
        csv_bool(zonotope_certified && !oracle_safe),
        csv_bool(oracle_safe && !box_certified),
        csv_bool(oracle_safe && !zonotope_certified),
        box_minimum_slack,
        zonotope_minimum_slack,
    ));
}

fn rate(numerator: usize, denominator: usize) -> String {
    if denominator == 0 {
        "NA".to_owned()
    } else {
        format!("{:.6}", numerator as f64 / denominator as f64)
    }
}

fn summary_row(output: &mut String, point_name: &str, action: Action, counters: Counters) {
    let mean_box = counters.sum_box_minimum_slack / counters.profiles as f64;
    let mean_zonotope = counters.sum_zonotope_minimum_slack / counters.profiles as f64;
    output.push_str(&format!(
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.9},{:.9}\n",
        point_name,
        action_name(action),
        counters.profiles,
        counters.oracle_safe,
        counters.box_certified,
        counters.zonotope_certified,
        counters.box_false_certificates,
        counters.zonotope_false_certificates,
        counters.box_false_vetoes,
        counters.zonotope_false_vetoes,
        rate(counters.box_false_vetoes, counters.oracle_safe),
        rate(counters.zonotope_false_vetoes, counters.oracle_safe),
        rate(counters.box_certified, counters.profiles),
        rate(counters.zonotope_certified, counters.profiles),
        counters.zonotope_certified.saturating_sub(counters.box_certified),
        mean_box,
        mean_zonotope,
    ));
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            "usage: allostatic-joint-uncertainty-bench <raw.csv> <summary.csv>",
        )
        .into());
    }

    let mut raw = format!("{RAW_HEADER}\n");
    let mut summary = format!("{SUMMARY_HEADER}\n");
    let mut buckets: Vec<Vec<Counters>> = (0..OPERATING_POINTS.len())
        .map(|_| vec![Counters::default(); ACTIONS.len()])
        .collect();
    let mut pooled = vec![Counters::default(); ACTIONS.len()];
    let mut rows = 0_usize;

    for (point_index, point) in OPERATING_POINTS.iter().copied().enumerate() {
        for seed in 0..PROFILE_COUNT {
            let state = state_for(point, seed)?;
            if !state.is_viable(limits()) {
                return Err(IoError::new(
                    ErrorKind::InvalidData,
                    format!("fixture generated an unsafe initial state: {} seed={seed}", point.name),
                )
                .into());
            }

            for (action_index, action) in ACTIONS.iter().copied().enumerate() {
                let effects = effects_for(seed, action)?;
                let box_margins = box_viability_margins(state, &effects, limits());
                let zonotope_margins = zonotope_viability_margins(state, &effects, limits());
                let box_certificate = box_certifies(state, &effects, limits());
                let zonotope_certificate = zonotope_certifies(state, &effects, limits());
                let oracle_safe = corner_oracle_safe(state, &effects, limits());

                if box_certificate && !oracle_safe {
                    return Err(IoError::new(
                        ErrorKind::InvalidData,
                        format!("box false certificate: state={} seed={seed} action={}", point.name, action_name(action)),
                    )
                    .into());
                }
                if zonotope_certificate != oracle_safe {
                    return Err(IoError::new(
                        ErrorKind::InvalidData,
                        format!("zonotope/oracle disagreement: state={} seed={seed} action={}", point.name, action_name(action)),
                    )
                    .into());
                }
                if box_margins.minimum() > zonotope_margins.minimum() + 1e-10 {
                    return Err(IoError::new(
                        ErrorKind::InvalidData,
                        "box support bound unexpectedly tighter than the zonotope support bound",
                    )
                    .into());
                }

                raw_row(
                    &mut raw,
                    point,
                    seed,
                    action,
                    state,
                    &effects,
                    box_certificate,
                    zonotope_certificate,
                    oracle_safe,
                    box_margins.minimum(),
                    zonotope_margins.minimum(),
                );
                buckets[point_index][action_index].observe(
                    oracle_safe,
                    box_certificate,
                    zonotope_certificate,
                    box_margins.minimum(),
                    zonotope_margins.minimum(),
                );
                pooled[action_index].observe(
                    oracle_safe,
                    box_certificate,
                    zonotope_certificate,
                    box_margins.minimum(),
                    zonotope_margins.minimum(),
                );
                rows += 1;
            }
        }
    }

    for (point_index, point) in OPERATING_POINTS.iter().enumerate() {
        for (action_index, action) in ACTIONS.iter().copied().enumerate() {
            summary_row(
                &mut summary,
                point.name,
                action,
                buckets[point_index][action_index],
            );
        }
    }
    for action_index in 0..ACTIONS.len() {
        summary_row(&mut summary, "all_states", ACTIONS[action_index], pooled[action_index]);
    }

    if rows != OPERATING_POINTS.len() * PROFILE_COUNT as usize * ACTIONS.len() {
        return Err(IoError::new(ErrorKind::InvalidData, "raw frontier row count mismatch").into());
    }

    fs::write(&args[1], &raw)?;
    fs::write(&args[2], &summary)?;
    println!("Joint-uncertainty raw results: {}", args[1]);
    println!("Joint-uncertainty summary: {}", args[2]);
    println!("Evaluation rows: {rows}");
    Ok(())
}
