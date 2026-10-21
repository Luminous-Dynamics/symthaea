//! Emit a paired operating-state frontier for point, global, and action-specific safety filters.
//!
//! Usage: allostatic-model-mismatch-bench <raw.csv> <summary.csv>
//! The generated profiles are deterministic synthetic falsification fixtures,
//! not independent physical trials or evidence of empirical conformal coverage.

use allostatic_control_bench::calibration::{
    ActionSpecificCalibratedEnvelopeRadii, CalibratedEnvelopeRadii,
};
use allostatic_control_bench::model_mismatch::{
    action_specific_calibration_panel, calibrate_action_specific_mismatch_panel,
    calibrate_mismatch_panel, calibration_panel, point_and_action_specific_robust_vs_true,
    point_and_robust_vs_true, synthetic_mismatch_profile,
};
use allostatic_control_bench::robust_viability::{Interval, StateEnvelope, ViabilityLimits};
use allostatic_control_bench::Action;
use std::error::Error;
use std::fs;
use std::io::{Error as IoError, ErrorKind};

const STRESS_GRID: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
const RAW_HEADER: &str = "panel,operating_point,reserve,load,episode,stress,action,point_certified,global_robust_certified,action_specific_certified,true_safe,point_false_certification,global_robust_false_certification,action_specific_false_certification,global_robust_false_veto,action_specific_false_veto";
const SUMMARY_HEADER: &str = "panel,operating_point,reserve,load,profiles,candidate_actions,point_certified_actions,global_robust_certified_actions,action_specific_certified_actions,true_safe_actions,point_false_certifications,global_robust_false_certifications,action_specific_false_certifications,global_robust_false_vetoes,action_specific_false_vetoes,point_false_cert_rate,global_robust_false_cert_rate,action_specific_false_cert_rate,global_robust_false_veto_rate,action_specific_false_veto_rate,point_action_fraction,global_robust_action_fraction,action_specific_action_fraction";

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

#[derive(Clone, Copy, Debug)]
struct CalibrationModels {
    global: CalibratedEnvelopeRadii,
    action_specific: ActionSpecificCalibratedEnvelopeRadii,
}

#[derive(Clone, Copy, Debug, Default)]
struct PanelCounters {
    profiles: usize,
    candidate_actions: usize,
    point_certified: usize,
    global_robust_certified: usize,
    action_specific_certified: usize,
    true_safe: usize,
    point_false_certifications: usize,
    global_robust_false_certifications: usize,
    action_specific_false_certifications: usize,
    global_robust_false_vetoes: usize,
    action_specific_false_vetoes: usize,
}

impl PanelCounters {
    fn accumulate(&mut self, other: Self) {
        self.profiles += other.profiles;
        self.candidate_actions += other.candidate_actions;
        self.point_certified += other.point_certified;
        self.global_robust_certified += other.global_robust_certified;
        self.action_specific_certified += other.action_specific_certified;
        self.true_safe += other.true_safe;
        self.point_false_certifications += other.point_false_certifications;
        self.global_robust_false_certifications += other.global_robust_false_certifications;
        self.action_specific_false_certifications += other.action_specific_false_certifications;
        self.global_robust_false_vetoes += other.global_robust_false_vetoes;
        self.action_specific_false_vetoes += other.action_specific_false_vetoes;
    }
}

fn action_name(action: Action) -> &'static str {
    match action {
        Action::Push => "push",
        Action::Maintain => "maintain",
        Action::EmergencyRecover => "emergency_recover",
    }
}

fn rate(numerator: usize, denominator: usize) -> String {
    if denominator == 0 {
        "NA".to_string()
    } else {
        format!("{:.6}", numerator as f64 / denominator as f64)
    }
}

fn summarize_panel(
    panel: &str,
    operating_point: &str,
    reserve: &str,
    load: &str,
    counters: PanelCounters,
    summary: &mut String,
) {
    summary.push_str(&format!(
        "{panel},{operating_point},{reserve},{load},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
        counters.profiles,
        counters.candidate_actions,
        counters.point_certified,
        counters.global_robust_certified,
        counters.action_specific_certified,
        counters.true_safe,
        counters.point_false_certifications,
        counters.global_robust_false_certifications,
        counters.action_specific_false_certifications,
        counters.global_robust_false_vetoes,
        counters.action_specific_false_vetoes,
        rate(counters.point_false_certifications, counters.point_certified),
        rate(
            counters.global_robust_false_certifications,
            counters.global_robust_certified
        ),
        rate(
            counters.action_specific_false_certifications,
            counters.action_specific_certified
        ),
        rate(counters.global_robust_false_vetoes, counters.true_safe),
        rate(counters.action_specific_false_vetoes, counters.true_safe),
        rate(counters.point_certified, counters.candidate_actions),
        rate(
            counters.global_robust_certified,
            counters.candidate_actions
        ),
        rate(
            counters.action_specific_certified,
            counters.candidate_actions
        ),
    ));
}

fn state_for(point: OperatingPoint) -> Result<StateEnvelope, Box<dyn Error>> {
    StateEnvelope::new(
        Interval::point(point.reserve)
            .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid reserve point"))?,
        Interval::point(point.load)
            .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid load point"))?,
    )
    .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid operating state").into())
}

fn run_panel(
    panel: &str,
    point: OperatingPoint,
    first_seed: u64,
    profile_count: usize,
    mismatch_scale: f64,
    models: CalibrationModels,
    raw: &mut String,
) -> Result<PanelCounters, Box<dyn Error>> {
    let state = state_for(point)?;
    let limits = ViabilityLimits::new(0.15, 0.88)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid viability limits"))?;
    let actions = [Action::Push, Action::Maintain, Action::EmergencyRecover];
    let is_in_envelope = panel == "within_envelope";
    let mut counters = PanelCounters {
        profiles: profile_count,
        ..PanelCounters::default()
    };

    for offset in 0..profile_count {
        let mismatch = synthetic_mismatch_profile(first_seed + offset as u64, mismatch_scale)
            .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid mismatch scale"))?;
        for stress in STRESS_GRID {
            for action in actions {
                let (point_certified, global_robust_certified, true_safe) =
                    point_and_robust_vs_true(
                        state,
                        action,
                        stress,
                        mismatch,
                        models.global,
                        limits,
                    )
                    .ok_or_else(|| {
                        IoError::new(ErrorKind::InvalidData, "invalid global comparison inputs")
                    })?;
                let (specific_point_certified, action_specific_certified, specific_true_safe) =
                    point_and_action_specific_robust_vs_true(
                        state,
                        action,
                        stress,
                        mismatch,
                        models.action_specific,
                        limits,
                    )
                    .ok_or_else(|| {
                        IoError::new(
                            ErrorKind::InvalidData,
                            "invalid action-specific comparison inputs",
                        )
                    })?;
                if point_certified != specific_point_certified || true_safe != specific_true_safe {
                    return Err(IoError::new(
                        ErrorKind::InvalidData,
                        "global and action-specific comparisons disagree on point prediction or true state",
                    )
                    .into());
                }

                let point_false = point_certified && !true_safe;
                let global_robust_false = global_robust_certified && !true_safe;
                let action_specific_false = action_specific_certified && !true_safe;
                let global_robust_false_veto = !global_robust_certified && true_safe;
                let action_specific_false_veto = !action_specific_certified && true_safe;

                if is_in_envelope && (global_robust_false || action_specific_false) {
                    return Err(IoError::new(
                        ErrorKind::InvalidData,
                        format!(
                            "robust false certification within envelope: point={} reserve={:.2} load={:.2} profile={} stress={stress} action={}",
                            point.name,
                            point.reserve,
                            point.load,
                            offset,
                            action_name(action)
                        ),
                    )
                    .into());
                }

                counters.candidate_actions += 1;
                counters.point_certified += usize::from(point_certified);
                counters.global_robust_certified += usize::from(global_robust_certified);
                counters.action_specific_certified += usize::from(action_specific_certified);
                counters.true_safe += usize::from(true_safe);
                counters.point_false_certifications += usize::from(point_false);
                counters.global_robust_false_certifications += usize::from(global_robust_false);
                counters.action_specific_false_certifications += usize::from(action_specific_false);
                counters.global_robust_false_vetoes += usize::from(global_robust_false_veto);
                counters.action_specific_false_vetoes += usize::from(action_specific_false_veto);

                raw.push_str(&format!(
                    "{panel},{},{:.2},{:.2},{},{stress:.2},{},{},{},{},{},{},{},{},{},{}\n",
                    point.name,
                    point.reserve,
                    point.load,
                    offset,
                    action_name(action),
                    point_certified,
                    global_robust_certified,
                    action_specific_certified,
                    true_safe,
                    point_false,
                    global_robust_false,
                    action_specific_false,
                    global_robust_false_veto,
                    action_specific_false_veto,
                ));
            }
        }
    }
    Ok(counters)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            "usage: allostatic-model-mismatch-bench <raw.csv> <summary.csv>",
        )
        .into());
    }

    let global_calibration = calibration_panel();
    let global = calibrate_mismatch_panel(&global_calibration, &STRESS_GRID, 0.90)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "global calibration panel failed"))?;
    let action_specific_calibration = action_specific_calibration_panel();
    let action_specific =
        calibrate_action_specific_mismatch_panel(&action_specific_calibration, &STRESS_GRID, 0.95)
            .ok_or_else(|| {
                IoError::new(
                    ErrorKind::InvalidData,
                    "action-specific calibration panel failed",
                )
            })?;
    let models = CalibrationModels {
        global,
        action_specific,
    };

    let mut raw = format!("{RAW_HEADER}\n");
    let mut summary = format!("{SUMMARY_HEADER}\n");

    for (panel, first_seed, profile_count, mismatch_scale) in
        [("within_envelope", 1_000, 300, 1.0), ("out_of_envelope", 5_000, 32, 2.0)]
    {
        let mut aggregate = PanelCounters::default();
        for point in OPERATING_POINTS {
            let counters = run_panel(
                panel,
                point,
                first_seed,
                profile_count,
                mismatch_scale,
                models,
                &mut raw,
            )?;
            summarize_panel(
                panel,
                point.name,
                &format!("{:.2}", point.reserve),
                &format!("{:.2}", point.load),
                counters,
                &mut summary,
            );
            aggregate.accumulate(counters);
        }
        summarize_panel(panel, "all_states", "NA", "NA", aggregate, &mut summary);
    }

    fs::write(&args[1], raw)?;
    fs::write(&args[2], summary)?;
    Ok(())
}
