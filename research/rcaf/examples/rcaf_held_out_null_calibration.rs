use cetb_rcaf::held_out_null_calibration::{
    calibrate_and_evaluate_held_out_null, HeldOutEvaluationObservation, NullCalibrationObservation,
};
use std::error::Error;
use std::fs;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn Error>> {
    // Independent synthetic null calibration split: 99 replicates yield 5% finite-sample
    // resolution at alpha=0.05. The sorted null statistic spans 0.01..0.99 for estimator A.
    let calibration = (0..99)
        .map(|index| {
            let base = (index + 1) as f64 / 100.0;
            NullCalibrationObservation {
                replicate_id: format!("cal-{index:03}"),
                statistics: [base, base * 0.8, base * 0.6],
            }
        })
        .collect::<Vec<_>>();

    // Separate held-out evaluation split, with nuisance-only nulls and paired signal trials.
    let evaluation = (0..100)
        .map(|index| {
            let base = (index + 1) as f64 / 100.0;
            let signal = 0.90 + index as f64 / 1000.0;
            HeldOutEvaluationObservation {
                replicate_id: format!("eval-{index:03}"),
                null_statistics: [base, base * 0.8, base * 0.6],
                signal_statistics: [signal, signal * 0.95, signal * 0.9],
                signal_directional_cosine: if index % 10 == 0 { -0.2 } else { 0.8 },
            }
        })
        .collect::<Vec<_>>();

    let report = calibrate_and_evaluate_held_out_null(
        &calibration,
        &evaluation,
        "synthetic-calibration-split-v1",
        "synthetic-evaluation-split-v1",
        0.05,
        0.0,
    )?;
    if report.estimators.len() != 3
        || report.calibration_replicates != 99
        || report.evaluation_replicates != 100
    {
        return Err("held-out null calibration report has unexpected dimensions".into());
    }
    let rendered = report.to_json_pretty();
    if let Some(path) = std::env::args_os().nth(1) {
        fs::write(path, rendered)?;
    } else {
        io::stdout().write_all(rendered.as_bytes())?;
    }
    Ok(())
}
