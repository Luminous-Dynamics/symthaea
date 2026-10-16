//! Deterministic synthetic replay smoke benchmark.
//!
//! This is a test-fixture generator, not empirical evidence or a market simulator validated
//! against venue behavior. Each event has a latent probability; outcomes and horizon signals are
//! drawn from deterministic, independent streams. Forecast methods see only signals available at
//! their horizon. An explicit future-only oracle stream is a negative control for cutoff leakage.

use std::collections::BTreeMap;

use symthaea_futures_market_integrity::replay::{
    paired_cluster_bootstrap, replay_horizons, ForecastObservation, ForecastTarget,
    PairedForecastComparison, ReplayError, ReplayExclusionReason,
};

const EVENT_COUNT: usize = 60;
const HORIZONS: [(&str, u64); 5] = [
    ("90d", 11_000),
    ("30d", 17_000),
    ("7d", 19_300),
    ("1d", 19_900),
    ("close", 19_990),
];
const RESOLUTION_OFFSET: u64 = 20_000;
const MAX_STALENESS_SECONDS: u64 = 120;

fn main() {
    run().expect("synthetic replay benchmark should be internally consistent");
}

fn run() -> Result<(), ReplayError> {
    let mut observations = Vec::new();
    let mut targets = Vec::new();
    let mut cluster_by_target = BTreeMap::new();

    for event_index in 0..EVENT_COUNT {
        let event_id = format!("synthetic-event-{event_index:03}");
        let cluster_id = format!("synthetic-parent-{:03}", event_index / 2);
        let latent_yes_probability = 0.1 + 0.8 * deterministic_unit(event_index, 0);
        let outcome = deterministic_unit(event_index, 1) < latent_yes_probability;
        let event_base = 1_000_000 + event_index as u64 * 100_000;
        let mut positive_signals = 0usize;

        for (horizon_index, (horizon_id, cutoff_offset)) in HORIZONS.iter().enumerate() {
            let cutoff = event_base + *cutoff_offset;
            let target = ForecastTarget::new(
                event_id.clone(),
                *horizon_id,
                cutoff,
                event_base + RESOLUTION_OFFSET,
                outcome,
                MAX_STALENESS_SECONDS,
            )?;
            cluster_by_target.insert((event_id.clone(), (*horizon_id).to_owned()), cluster_id.clone());
            targets.push(target);

            // Each signal is drawn from the latent event probability, not from the realized
            // outcome. Forecast methods only aggregate signals observed by this horizon.
            let public_signal = deterministic_unit(event_index, 10 + horizon_index as u64)
                < latent_yes_probability;
            positive_signals += usize::from(public_signal);
            let signal_count = (horizon_index + 1) as f64;
            let market_probability =
                (positive_signals as f64 / signal_count).clamp(0.02, 0.98);
            let shrinkage_probability =
                ((positive_signals as f64 + 1.0) / (signal_count + 2.0)).clamp(0.02, 0.98);

            observations.push(ForecastObservation::new(
                event_id.clone(),
                "market",
                cutoff,
                market_probability,
            )?);
            observations.push(ForecastObservation::new(
                event_id.clone(),
                "base_rate",
                cutoff,
                0.5,
            )?);

            // Deliberate missing snapshots create explicit coverage loss.
            if (event_index * 3 + horizon_index) % 23 != 0 {
                observations.push(ForecastObservation::new(
                    event_id.clone(),
                    "shrinkage_candidate",
                    cutoff,
                    shrinkage_probability,
                )?);
            }

            // Negative control: an outcome-revealing forecast is timestamped after this cutoff.
            // It is never valid for this target, and is too stale to be reused by later horizons.
            observations.push(ForecastObservation::new(
                event_id.clone(),
                "oracle_future_only",
                cutoff + 1,
                if outcome { 0.999 } else { 0.001 },
            )?);
        }
    }

    let report = replay_horizons(&observations, &targets)?;
    println!("horizon,method,eligible,total,coverage,mean_brier,mean_log_loss,paired_brier_difference,ci95_low,ci95_high,bootstrap_share_candidate_better,clusters");

    for (horizon, _) in HORIZONS {
        let negative_controls = report
            .excluded
            .iter()
            .filter(|row| row.horizon_id == horizon && row.forecaster_id == "oracle_future_only")
            .collect::<Vec<_>>();
        assert_eq!(
            negative_controls.len(),
            EVENT_COUNT,
            "the future-only oracle control must be excluded for every event/horizon"
        );
        assert!(
            negative_controls.iter().all(|row| matches!(
                row.reason,
                ReplayExclusionReason::NoForecastAtOrBeforeCutoff
                    | ReplayExclusionReason::ForecastTooStale
            )),
            "the future-only oracle control may only be excluded for explicit temporal reasons"
        );
        eprintln!(
            "NEGATIVE_CONTROL horizon={horizon} oracle_forecasts_excluded={} expected={EVENT_COUNT}",
            negative_controls.len()
        );
    }

    for (horizon_index, (horizon, _)) in HORIZONS.iter().enumerate() {
        let horizon_targets = targets
            .iter()
            .filter(|target| target.horizon_id() == *horizon)
            .count();

        let horizon_scores = report
            .scored
            .iter()
            .filter(|score| score.horizon_id == *horizon)
            .collect::<Vec<_>>();
        let mut metric_rows = Vec::new();
        for method in ["base_rate", "market", "shrinkage_candidate"] {
            let scores = horizon_scores
                .iter()
                .filter(|score| score.forecaster_id == method)
                .collect::<Vec<_>>();
            if scores.is_empty() {
                continue;
            }
            let count = scores.len();
            let mean_brier =
                scores.iter().map(|score| score.brier_score).sum::<f64>() / count as f64;
            let mean_log_loss =
                scores.iter().map(|score| score.log_loss).sum::<f64>() / count as f64;
            let coverage = count as f64 / horizon_targets as f64;
            metric_rows.push((method, count, coverage, mean_brier, mean_log_loss));
        }

        let market_by_event = horizon_scores
            .iter()
            .filter(|score| score.forecaster_id == "market")
            .map(|score| (score.event_id.clone(), *score))
            .collect::<BTreeMap<_, _>>();
        let candidate_scores = horizon_scores
            .iter()
            .filter(|score| score.forecaster_id == "shrinkage_candidate");
        let mut paired = Vec::new();
        for candidate in candidate_scores {
            let Some(baseline) = market_by_event.get(&candidate.event_id) else {
                continue;
            };
            let key = (candidate.event_id.clone(), candidate.horizon_id.clone());
            let cluster = cluster_by_target
                .get(&key)
                .expect("every synthetic target has a parent cluster");
            paired.push(PairedForecastComparison::new(
                candidate.event_id.clone(),
                candidate.horizon_id.clone(),
                cluster.clone(),
                candidate.outcome,
                baseline.probability,
                candidate.probability,
            )?);
        }
        let bootstrap =
            paired_cluster_bootstrap(&paired, 10_000, 0xC0FFEE + horizon_index as u64)?;

        for (method, count, coverage, brier, log_loss) in metric_rows {
            if method == "shrinkage_candidate" {
                println!(
                    "{horizon},{method},{count},{horizon_targets},{coverage:.6},{brier:.6},{log_loss:.6},{:.6},{:.6},{:.6},{:.6},{}",
                    bootstrap.mean_brier_difference,
                    bootstrap.lower_95_brier_difference,
                    bootstrap.upper_95_brier_difference,
                    bootstrap.bootstrap_share_candidate_better,
                    bootstrap.cluster_count
                );
            } else {
                println!(
                    "{horizon},{method},{count},{horizon_targets},{coverage:.6},{brier:.6},{log_loss:.6},,,,,{}",
                    bootstrap.cluster_count
                );
            }
        }
    }
    Ok(())
}

fn deterministic_unit(event_index: usize, salt: u64) -> f64 {
    // SplitMix64-derived deterministic draws for an offline fixture, never for cryptography.
    let mut value = (event_index as u64 + 1)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ salt.wrapping_mul(0xD1B5_4A32_D192_ED03);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    (value >> 11) as f64 / ((1_u64 << 53) as f64)
}
