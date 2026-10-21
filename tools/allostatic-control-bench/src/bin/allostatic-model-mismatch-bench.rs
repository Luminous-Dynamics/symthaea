//! Emit deterministic point-shield versus robust-envelope mismatch evidence.
//!
//! Usage: allostatic-model-mismatch-bench <raw.csv> <summary.csv>
//! The output is synthetic falsification evidence, not an external benchmark score.

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
const RAW_HEADER: &str = "panel,episode,stress,action,point_certified,global_robust_certified,action_specific_certified,true_safe,point_false_certification,global_robust_false_certification,action_specific_false_certification,global_robust_false_veto,action_specific_false_veto";
const SUMMARY_HEADER: &str = "panel,profiles,candidate_actions,point_certified_actions,global_robust_certified_actions,action_specific_certified_actions,true_safe_actions,point_false_certifications,global_robust_false_certifications,action_specific_false_certifications,global_robust_false_vetoes,action_specific_false_vetoes,point_false_cert_rate,global_robust_false_cert_rate,action_specific_false_cert_rate,global_robust_false_veto_rate,action_specific_false_veto_rate,point_action_fraction,global_robust_action_fraction,action_specific_action_fraction";

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

fn summarize_panel(panel: &str, counters: PanelCounters, summary: &mut String) {
    summary.push_str(&format!(
        "{panel},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
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
        rate(
            counters.point_false_certifications,
            counters.point_certified
        ),
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
        rate(counters.global_robust_certified, counters.candidate_actions),
        rate(
            counters.action_specific_certified,
            counters.candidate_actions
        ),
    ));
}

fn run_panel(
    panel: &str,
    first_seed: u64,
    profile_count: usize,
    mismatch_scale: f64,
    calibration_radii: allostatic_control_bench::calibration::CalibratedEnvelopeRadii,
    action_specific_radii: allostatic_control_bench::calibration::ActionSpecificCalibratedEnvelopeRadii,
    raw: &mut String,
) -> Result<PanelCounters, Box<dyn Error>> {
    let state = StateEnvelope::new(
        Interval::point(0.22)
            .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid reserve point"))?,
        Interval::point(0.80)
            .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid load point"))?,
    )
    .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid state"))?;
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
                        calibration_radii,
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
                        action_specific_radii,
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
                            "robust false certification within envelope: profile={} stress={stress} action={}",
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
                    "{panel},{},{stress:.2},{},{},{},{},{},{},{},{},{},{}\n",
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

    let calibration = calibration_panel();
    let global_radii = calibrate_mismatch_panel(&calibration, &STRESS_GRID, 0.90)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "global calibration panel failed"))?;
    let action_specific_calibration = action_specific_calibration_panel();
    let action_specific_radii =
        calibrate_action_specific_mismatch_panel(&action_specific_calibration, &STRESS_GRID, 0.95)
            .ok_or_else(|| {
                IoError::new(
                    ErrorKind::InvalidData,
                    "action-specific calibration panel failed",
                )
            })?;

    let mut raw = format!("{RAW_HEADER}\n");
    let mut summary = format!("{SUMMARY_HEADER}\n");
    let within = run_panel(
        "within_envelope",
        1_000,
        300,
        1.0,
        global_radii,
        action_specific_radii,
        &mut raw,
    )?;
    let outside = run_panel(
        "out_of_envelope",
        5_000,
        32,
        2.0,
        global_radii,
        action_specific_radii,
        &mut raw,
    )?;
    summarize_panel("within_envelope", within, &mut summary);
    summarize_panel("out_of_envelope", outside, &mut summary);

    fs::write(&args[1], raw)?;
    fs::write(&args[2], summary)?;
    Ok(())
}
