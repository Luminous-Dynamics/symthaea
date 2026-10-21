//! Compare global and action-specific robust filters at matched 90% and 95% joint coverage targets.
//!
//! Usage: allostatic-model-mismatch-bench <raw.csv> <summary.csv>
//! The generated profiles are deterministic synthetic falsification fixtures,
//! not independent physical trials or evidence of empirical conformal coverage.

use allostatic_control_bench::calibration::{
    ActionSpecificCalibratedEnvelopeRadii, CalibratedEnvelopeRadii,
};
use allostatic_control_bench::model_mismatch::{
    action_specific_calibration_panel, calibrate_action_specific_mismatch_panel,
    calibrate_mismatch_panel, point_and_action_specific_robust_vs_true,
    point_and_robust_vs_true, synthetic_mismatch_profile,
};
use allostatic_control_bench::robust_viability::{Interval, StateEnvelope, ViabilityLimits};
use allostatic_control_bench::Action;
use std::error::Error;
use std::fs;
use std::io::{Error as IoError, ErrorKind};

const STRESS_GRID: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
const RAW_HEADER: &str = "panel,operating_point,reserve,load,episode,stress,action,point_certified,global_90_certified,action_specific_90_certified,global_95_certified,action_specific_95_certified,true_safe,point_false_certification,global_90_false_certification,action_specific_90_false_certification,global_95_false_certification,action_specific_95_false_certification,global_90_false_veto,action_specific_90_false_veto,global_95_false_veto,action_specific_95_false_veto";
const SUMMARY_HEADER: &str = "panel,operating_point,reserve,load,profiles,candidate_actions,point_certified_actions,global_90_certified_actions,action_specific_90_certified_actions,global_95_certified_actions,action_specific_95_certified_actions,true_safe_actions,point_false_certifications,global_90_false_certifications,action_specific_90_false_certifications,global_95_false_certifications,action_specific_95_false_certifications,global_90_false_vetoes,action_specific_90_false_vetoes,global_95_false_vetoes,action_specific_95_false_vetoes,point_false_cert_rate,global_90_false_cert_rate,action_specific_90_false_cert_rate,global_95_false_cert_rate,action_specific_95_false_cert_rate,global_90_false_veto_rate,action_specific_90_false_veto_rate,global_95_false_veto_rate,action_specific_95_false_veto_rate,point_action_fraction,global_90_action_fraction,action_specific_90_action_fraction,global_95_action_fraction,action_specific_95_action_fraction";

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
    global_90: CalibratedEnvelopeRadii,
    action_specific_90: ActionSpecificCalibratedEnvelopeRadii,
    global_95: CalibratedEnvelopeRadii,
    action_specific_95: ActionSpecificCalibratedEnvelopeRadii,
}

#[derive(Clone, Copy, Debug, Default)]
struct PanelCounters {
    profiles: usize,
    candidate_actions: usize,
    point_certified: usize,
    robust_certified: [usize; 4],
    true_safe: usize,
    false_certifications: [usize; 5],
    false_vetoes: [usize; 4],
}

impl PanelCounters {
    fn accumulate(&mut self, other: Self) {
        self.profiles += other.profiles;
        self.candidate_actions += other.candidate_actions;
        self.point_certified += other.point_certified;
        self.true_safe += other.true_safe;
        for (left, right) in self.robust_certified.iter_mut().zip(other.robust_certified) {
            *left += right;
        }
        for (left, right) in self.false_certifications.iter_mut().zip(other.false_certifications) {
            *left += right;
        }
        for (left, right) in self.false_vetoes.iter_mut().zip(other.false_vetoes) {
            *left += right;
        }
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
    let robust_rates: Vec<String> = counters
        .robust_certified
        .iter()
        .enumerate()
        .map(|(index, denominator)| {
            rate(counters.false_certifications[index + 1], *denominator)
        })
        .collect();
    let false_veto_rates: Vec<String> = counters
        .false_vetoes
        .iter()
        .map(|vetoes| rate(*vetoes, counters.true_safe))
        .collect();
    let action_fractions: Vec<String> = std::iter::once(counters.point_certified)
        .chain(counters.robust_certified)
        .map(|certified| rate(certified, counters.candidate_actions))
        .collect();

    let mut fields = vec![
        panel.to_string(),
        operating_point.to_string(),
        reserve.to_string(),
        load.to_string(),
        counters.profiles.to_string(),
        counters.candidate_actions.to_string(),
        counters.point_certified.to_string(),
    ];
    fields.extend(counters.robust_certified.iter().map(usize::to_string));
    fields.push(counters.true_safe.to_string());
    fields.extend(counters.false_certifications.iter().map(usize::to_string));
    fields.extend(counters.false_vetoes.iter().map(usize::to_string));
    fields.push(rate(
        counters.false_certifications[0],
        counters.point_certified,
    ));
    fields.extend(robust_rates);
    fields.extend(false_veto_rates);
    fields.extend(action_fractions);
    push_row(summary, fields);
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
                let (point_certified, global_90, true_safe) = point_and_robust_vs_true(
                    state,
                    action,
                    stress,
                    mismatch,
                    models.global_90,
                    limits,
                )
                .ok_or_else(|| {
                    IoError::new(ErrorKind::InvalidData, "invalid global 90% comparison inputs")
                })?;
                let (point_95, global_95, true_95) = point_and_robust_vs_true(
                    state,
                    action,
                    stress,
                    mismatch,
                    models.global_95,
                    limits,
                )
                .ok_or_else(|| {
                    IoError::new(ErrorKind::InvalidData, "invalid global 95% comparison inputs")
                })?;
                let (specific_point_90, action_specific_90, true_specific_90) =
                    point_and_action_specific_robust_vs_true(
                        state,
                        action,
                        stress,
                        mismatch,
                        models.action_specific_90,
                        limits,
                    )
                    .ok_or_else(|| {
                        IoError::new(
                            ErrorKind::InvalidData,
                            "invalid action-specific 90% comparison inputs",
                        )
                    })?;
                let (specific_point_95, action_specific_95, true_specific_95) =
                    point_and_action_specific_robust_vs_true(
                        state,
                        action,
                        stress,
                        mismatch,
                        models.action_specific_95,
                        limits,
                    )
                    .ok_or_else(|| {
                        IoError::new(
                            ErrorKind::InvalidData,
                            "invalid action-specific 95% comparison inputs",
                        )
                    })?;

                if [point_95, specific_point_90, specific_point_95]
                    .iter()
                    .any(|value| *value != point_certified)
                    || [true_95, true_specific_90, true_specific_95]
                        .iter()
                        .any(|value| *value != true_safe)
                {
                    return Err(IoError::new(
                        ErrorKind::InvalidData,
                        "matched-coverage models disagree on point prediction or true state",
                    )
                    .into());
                }

                let robust_certified = [global_90, action_specific_90, global_95, action_specific_95];
                let false_certifications = [
                    point_certified && !true_safe,
                    global_90 && !true_safe,
                    action_specific_90 && !true_safe,
                    global_95 && !true_safe,
                    action_specific_95 && !true_safe,
                ];
                let false_vetoes = [
                    !global_90 && true_safe,
                    !action_specific_90 && true_safe,
                    !global_95 && true_safe,
                    !action_specific_95 && true_safe,
                ];

                if is_in_envelope
                    && robust_certified.iter().zip([global_90, action_specific_90, global_95, action_specific_95])
                        .any(|(certified, _)| *certified && !true_safe)
                {
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
                for (count, certified) in counters.robust_certified.iter_mut().zip(robust_certified) {
                    *count += usize::from(certified);
                }
                counters.true_safe += usize::from(true_safe);
                for (count, is_false) in counters.false_certifications.iter_mut().zip(false_certifications) {
                    *count += usize::from(is_false);
                }
                for (count, is_veto) in counters.false_vetoes.iter_mut().zip(false_vetoes) {
                    *count += usize::from(is_veto);
                }

                let fields = vec![
                    panel.to_string(),
                    point.name.to_string(),
                    format!("{:.2}", point.reserve),
                    format!("{:.2}", point.load),
                    offset.to_string(),
                    format!("{stress:.2}"),
                    action_name(action).to_string(),
                    b(point_certified),
                    b(global_90),
                    b(action_specific_90),
                    b(global_95),
                    b(action_specific_95),
                    b(true_safe),
                    b(false_certifications[0]),
                    b(false_certifications[1]),
                    b(false_certifications[2]),
                    b(false_certifications[3]),
                    b(false_certifications[4]),
                    b(false_vetoes[0]),
                    b(false_vetoes[1]),
                    b(false_vetoes[2]),
                    b(false_vetoes[3]),
                ];
                push_row(raw, fields);
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

    // Calibrate all four methods on the same 159-profile synthetic design panel.
    // The panel is a deterministic bounds fixture, not an exchangeable empirical sample.
    let calibration_panel = action_specific_calibration_panel();
    let global_90 = calibrate_mismatch_panel(&calibration_panel, &STRESS_GRID, 0.90)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "global 90% calibration failed"))?;
    let action_specific_90 =
        calibrate_action_specific_mismatch_panel(&calibration_panel, &STRESS_GRID, 0.90)
            .ok_or_else(|| {
                IoError::new(ErrorKind::InvalidData, "action-specific 90% calibration failed")
            })?;
    let global_95 = calibrate_mismatch_panel(&calibration_panel, &STRESS_GRID, 0.95)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "global 95% calibration failed"))?;
    let action_specific_95 =
        calibrate_action_specific_mismatch_panel(&calibration_panel, &STRESS_GRID, 0.95)
            .ok_or_else(|| {
                IoError::new(ErrorKind::InvalidData, "action-specific 95% calibration failed")
            })?;
    let models = CalibrationModels {
        global_90,
        action_specific_90,
        global_95,
        action_specific_95,
    };

    let mut raw = format!("{RAW_HEADER}\n");
    let mut summary = format!("{SUMMARY_HEADER}\n");

    for (panel, first_seed, profile_count, mismatch_scale) in [
        ("within_envelope", 1_000, 300, 1.0),
        ("out_of_envelope", 5_000, 32, 2.0),
    ] {
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
