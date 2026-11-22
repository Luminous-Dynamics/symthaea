use cetb_rcaf::weighted_covariate_shift::{
    evaluate_weighted_covariate_shift, RatioProvenance, WeightedCovariateShiftConfig,
    WeightedHeldOutEvaluationObservation, WeightedNullCalibrationObservation,
};
use std::error::Error;
use std::fs;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn Error>> {
    let calibration = (0..99)
        .map(|index| {
            let score = (index + 1) as f64 / 100.0;
            // Synthetic, analytically supplied density-ratio weights with moderate continuous
            // variation. They are known only within this constructed fixture.
            let weight = 0.5 + index as f64 / 98.0;
            WeightedNullCalibrationObservation {
                replicate_id: format!("cal-{index:03}"),
                statistics: [score, score * 0.8, score * 0.6],
                importance_weight: weight,
            }
        })
        .collect::<Vec<_>>();

    let evaluation = (0..100)
        .map(|index| {
            let score = (index + 1) as f64 / 100.0;
            let importance_weight = if index == 99 { 10.0 } else { 1.0 };
            WeightedHeldOutEvaluationObservation {
                replicate_id: format!("eval-{index:03}"),
                null_statistics: [score, score * 0.8, score * 0.6],
                signal_statistics: [0.99, 0.99 * 0.8, 0.99 * 0.6],
                signal_directional_cosine: if index % 10 == 0 { -0.2 } else { 0.8 },
                importance_weight,
            }
        })
        .collect::<Vec<_>>();

    let report = evaluate_weighted_covariate_shift(
        &calibration,
        &evaluation,
        &WeightedCovariateShiftConfig {
            calibration_split_id: "known-ratio-calibration-v1".to_string(),
            evaluation_split_id: "known-ratio-evaluation-v1".to_string(),
            ratio_provenance: RatioProvenance::KnownDensityRatio,
            alpha: 0.05,
            minimum_direction_cosine: 0.0,
            minimum_effective_sample_size: 50.0,
            source_support_attested: true,
            conditional_score_invariance_attested: true,
        },
    )?;

    if report.estimators.len() != 3
        || report.calibration_replicates != 99
        || report.evaluation_replicates != 100
        || report.abstentions.len() != 3
        || report.estimators.iter().any(|item| {
            item.finite_threshold_count != 99 || item.abstained_evaluation_count != 1
        })
    {
        return Err("weighted covariate-shift report has unexpected dimensions or abstentions".into());
    }

    let rendered = report.to_json_pretty();
    if let Some(path) = std::env::args_os().nth(1) {
        fs::write(path, rendered)?;
    } else {
        io::stdout().write_all(rendered.as_bytes())?;
    }
    Ok(())
}
