use std::collections::BTreeMap;
use std::error::Error;

use spectra_event_metric_calibration::{
    GaussianSignalModel, SimulationDesign, evaluate, simulate,
};

const D_PRIME_GRID: [f64; 6] = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0];
const CRITERION_GRID: [f64; 3] = [-1.0, 0.0, 1.0];

#[derive(Debug, Default, Clone, Copy)]
struct CellAccumulator {
    count: u64,
    d_prime_sum: f64,
    d_prime_error_sq: f64,
    criterion_sum: f64,
    criterion_error_sq: f64,
    auc_sum: f64,
    expected_auc_sum: f64,
    auc_error_sq: f64,
    ap_sum: f64,
    prevalence_sum: f64,
    accuracy_sum: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let seed_count = std::env::args()
        .nth(1)
        .map(|value| value.parse::<u64>())
        .transpose()?
        .unwrap_or(256);
    if seed_count == 0 {
        return Err("seed count must be positive".into());
    }

    let mut summary: BTreeMap<(&str, i32, i32), CellAccumulator> = BTreeMap::new();
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
                    // Reuse replicate seeds across cells. Signal, noise, and
                    // shuffle streams are independent of d-prime and criterion.
                    let seed = 0x5EED_2026_u64.wrapping_add(replicate);
                    let design = SimulationDesign { signal_trials, noise_trials, seed };
                    let bundle = simulate(model, design)?;
                    let report = evaluate(&bundle.candidate, &bundle.evaluator)?;
                    println!(
                        "{},{},{:.6},{:.6},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8}",
                        track, report.seed, report.injected_d_prime, report.injected_criterion_c,
                        report.prevalence, report.signal_detection.d_prime,
                        report.signal_detection.criterion_c, report.roc_auc, report.expected_roc_auc,
                        report.average_precision, report.random_ranking_ap_baseline, report.accuracy,
                        report.expected_accuracy, report.balanced_accuracy,
                        report.expected_balanced_accuracy, report.signal_detection.hit_rate,
                        report.expected_hit_rate, report.signal_detection.false_alarm_rate,
                        report.expected_false_alarm_rate,
                    );

                    let key = (track, (d_prime * 1000.0).round() as i32,
                        (criterion * 1000.0).round() as i32);
                    let cell = summary.entry(key).or_default();
                    cell.count += 1;
                    cell.d_prime_sum += report.signal_detection.d_prime;
                    cell.d_prime_error_sq += (report.signal_detection.d_prime - d_prime).powi(2);
                    cell.criterion_sum += report.signal_detection.criterion_c;
                    cell.criterion_error_sq +=
                        (report.signal_detection.criterion_c - criterion).powi(2);
                    cell.auc_sum += report.roc_auc;
                    cell.expected_auc_sum += report.expected_roc_auc;
                    cell.auc_error_sq += (report.roc_auc - report.expected_roc_auc).powi(2);
                    cell.ap_sum += report.average_precision;
                    cell.prevalence_sum += report.prevalence;
                    cell.accuracy_sum += report.accuracy;
                }
            }
        }
    }

    // Aggregate reports go to stderr so stdout remains a machine-readable CSV.
    eprintln!("# Aggregate metric-calibration report");
    eprintln!("track,d_prime,criterion,n,mean_d_prime,d_prime_bias,d_prime_rmse,mean_criterion,criterion_bias,criterion_rmse,mean_auc,auc_bias_vs_analytic,auc_rmse,mean_ap,random_ap_baseline,mean_accuracy");
    for ((track, d_key, c_key), cell) in &summary {
        let n = cell.count as f64;
        let d_prime = *d_key as f64 / 1000.0;
        let criterion = *c_key as f64 / 1000.0;
        let mean_d_prime = cell.d_prime_sum / n;
        let mean_criterion = cell.criterion_sum / n;
        let mean_auc = cell.auc_sum / n;
        let mean_expected_auc = cell.expected_auc_sum / n;
        eprintln!(
            "{},{:.3},{:.3},{},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8}",
            track, d_prime, criterion, cell.count, mean_d_prime, mean_d_prime - d_prime,
            (cell.d_prime_error_sq / n).sqrt(), mean_criterion, mean_criterion - criterion,
            (cell.criterion_error_sq / n).sqrt(), mean_auc, mean_auc - mean_expected_auc,
            (cell.auc_error_sq / n).sqrt(), cell.ap_sum / n, cell.prevalence_sum / n,
            cell.accuracy_sum / n,
        );
    }

    eprintln!("# Criterion-invariance diagnostic");
    eprintln!("track,d_prime,estimated_d_prime_spread_across_criteria,estimated_auc_spread_across_criteria");
    for track in ["balanced", "rare_5pct"] {
        for d_prime in D_PRIME_GRID {
            let d_key = (d_prime * 1000.0).round() as i32;
            let cells: Vec<_> = CRITERION_GRID.iter().filter_map(|criterion| {
                summary.get(&(track, d_key, (*criterion * 1000.0).round() as i32))
            }).collect();
            if cells.len() != CRITERION_GRID.len() { continue; }
            let d_means: Vec<_> = cells.iter().map(|cell| cell.d_prime_sum / cell.count as f64).collect();
            let auc_means: Vec<_> = cells.iter().map(|cell| cell.auc_sum / cell.count as f64).collect();
            let d_spread = d_means.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - d_means.iter().copied().fold(f64::INFINITY, f64::min);
            let auc_spread = auc_means.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - auc_means.iter().copied().fold(f64::INFINITY, f64::min);
            eprintln!("{},{:.3},{:.8},{:.8}", track, d_prime, d_spread, auc_spread);
        }
    }

    eprintln!("# Monotonicity diagnostic");
    eprintln!("track,criterion,nondecreasing_d_prime_steps_of_5,nondecreasing_auc_steps_of_5");
    for track in ["balanced", "rare_5pct"] {
        for criterion in CRITERION_GRID {
            let c_key = (criterion * 1000.0).round() as i32;
            let cells: Vec<_> = D_PRIME_GRID.iter().filter_map(|d_prime| {
                summary.get(&(track, (*d_prime * 1000.0).round() as i32, c_key))
            }).collect();
            let mut d_steps = 0;
            let mut auc_steps = 0;
            for pair in cells.windows(2) {
                if pair[1].d_prime_sum / pair[1].count as f64
                    >= pair[0].d_prime_sum / pair[0].count as f64 { d_steps += 1; }
                if pair[1].auc_sum / pair[1].count as f64
                    >= pair[0].auc_sum / pair[0].count as f64 { auc_steps += 1; }
            }
            eprintln!("{},{:.3},{}/5,{}/5", track, criterion, d_steps, auc_steps);
        }
    }

    Ok(())
}