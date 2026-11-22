use cetb_rcaf::groupwise_null_calibration::{
    calibrate_groupwise_null, GroupwiseCalibrationConfig,
    StratifiedHeldOutEvaluationObservation, StratifiedNullCalibrationObservation,
};
use cetb_rcaf::held_out_null_calibration::{
    HeldOutEvaluationObservation, NullCalibrationObservation,
};
use std::error::Error;
use std::fs;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn Error>> {
    let groups = [
        ("low_motion_single_echo", 1.0_f64),
        ("high_motion_single_echo", 2.0_f64),
        ("multi_echo", 3.0_f64),
    ];
    let mut calibration = Vec::new();
    let mut evaluation = Vec::new();

    for (stratum_id, scale) in groups {
        for index in 0..99 {
            let value = (index + 1) as f64 / 100.0 * scale;
            calibration.push(StratifiedNullCalibrationObservation {
                stratum_id: stratum_id.to_string(),
                observation: NullCalibrationObservation {
                    replicate_id: format!("cal-{stratum_id}-{index:03}"),
                    statistics: [value, value * 0.8, value * 0.6],
                },
            });
        }
        for index in 0..100 {
            let value = (index + 1) as f64 / 100.0 * scale;
            let signal = (0.90 + index as f64 / 1000.0) * scale;
            evaluation.push(StratifiedHeldOutEvaluationObservation {
                stratum_id: stratum_id.to_string(),
                observation: HeldOutEvaluationObservation {
                    replicate_id: format!("eval-{stratum_id}-{index:03}"),
                    null_statistics: [value, value * 0.8, value * 0.6],
                    signal_statistics: [signal, signal * 0.95, signal * 0.9],
                    signal_directional_cosine: if index % 10 == 0 { -0.2 } else { 0.8 },
                },
            });
        }
    }

    // New acquisition domain: deliberately no calibration threshold is available for it.
    for index in 0..10 {
        evaluation.push(StratifiedHeldOutEvaluationObservation {
            stratum_id: "unseen_scanner".to_string(),
            observation: HeldOutEvaluationObservation {
                replicate_id: format!("eval-unseen-{index:03}"),
                null_statistics: [100.0; 3],
                signal_statistics: [100.0; 3],
                signal_directional_cosine: 1.0,
            },
        });
    }

    let report = calibrate_groupwise_null(
        &calibration,
        &evaluation,
        &GroupwiseCalibrationConfig {
            calibration_split_id: "site-calibration-v1".to_string(),
            evaluation_split_id: "site-evaluation-v1".to_string(),
            expected_strata: groups.iter().map(|(name, _)| (*name).to_string()).collect(),
            alpha: 0.05,
            minimum_direction_cosine: 0.0,
        },
    )?;
    if report.strata.len() != 3
        || report.estimators.len() != 3
        || report.unseen_evaluation_strata.len() != 1
    {
        return Err("groupwise calibration report has unexpected dimensions".into());
    }
    let rendered = report.to_json_pretty();

    if let Some(path) = std::env::args_os().nth(1) {
        fs::write(path, rendered)?;
    } else {
        io::stdout().write_all(rendered.as_bytes())?;
    }
    Ok(())
}
