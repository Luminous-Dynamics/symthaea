use std::error::Error;

use spectra_event_metric_calibration::{
    GaussianSignalModel, SimulationDesign, evaluate, simulate,
};

const D_PRIME_GRID: [f64; 6] = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0];
const CRITERION_GRID: [f64; 3] = [-1.0, 0.0, 1.0];

fn main() -> Result<(), Box<dyn Error>> {
    let seed_count = std::env::args()
        .nth(1)
        .map(|value| value.parse::<u64>())
        .transpose()?
        .unwrap_or(256);

    println!(
        "track,seed,d_prime,criterion,prevalence,estimated_d_prime,estimated_criterion,auc,expected_auc,ap,random_ap_baseline,accuracy,expected_accuracy,balanced_accuracy,expected_balanced_accuracy,hit_rate,expected_hit_rate,false_alarm_rate,expected_false_alarm_rate"
    );

    for (track, signal_trials, noise_trials) in [
        ("balanced", 512usize, 512usize),
        ("rare_5pct", 128usize, 2_432usize),
    ] {
        for d_prime in D_PRIME_GRID {
            for criterion in CRITERION_GRID {
                let model = GaussianSignalModel::new(d_prime, criterion)?;
                for replicate in 0..seed_count {
                    // Reuse each replicate seed across all model cells. The
                    // generator uses separate signal/noise streams and a
                    // model-independent trial permutation.
                    let seed = 0x5EED_2026_u64.wrapping_add(replicate);
                    let design = SimulationDesign {
                        signal_trials,
                        noise_trials,
                        seed,
                    };
                    let bundle = simulate(model, design)?;
                    let report = evaluate(&bundle.candidate, &bundle.evaluator)?;
                    println!(
                        "{},{},{:.6},{:.6},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8}",
                        track,
                        report.seed,
                        report.injected_d_prime,
                        report.injected_criterion_c,
                        report.prevalence,
                        report.signal_detection.d_prime,
                        report.signal_detection.criterion_c,
                        report.roc_auc,
                        report.expected_roc_auc,
                        report.average_precision,
                        report.random_ranking_ap_baseline,
                        report.accuracy,
                        report.expected_accuracy,
                        report.balanced_accuracy,
                        report.expected_balanced_accuracy,
                        report.signal_detection.hit_rate,
                        report.expected_hit_rate,
                        report.signal_detection.false_alarm_rate,
                        report.expected_false_alarm_rate,
                    );
                }
            }
        }
    }

    Ok(())
}
