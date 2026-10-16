//! Deterministic synthetic replay smoke benchmark.
//!
//! This is a test fixture generator, not empirical evidence or a market simulator validated
//! against venue behavior. Forecast probabilities are intentionally constructed from synthetic
//! outcomes so the evaluator can be checked against known structure.

use std::collections::BTreeMap;

use symthaea_futures_market_integrity::replay::{
    paired_cluster_bootstrap, replay_horizons, ForecastObservation, ForecastTarget,
    PairedForecastComparison, ReplayError,
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
const BASE_CORRECT_PROBABILITY: [f64; 5] = [0.53, 0.59, 0.67, 0.77, 0.86];

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
        let outcome = (event_index * 7 + event_index / 3) % 11 < 5;
        let event_base = 1_000_000 + event_index as u64 * 100_000;

        for (horizon_index, (horizon_id, cutoff_offset)) in HORIZONS.iter().enumerate() {
            let cutoff = event_base + cutoff_offset;
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

            let market_correct = synthetic_correct_probability(event_index, horizon_index);
            let candidate_delta = if event_index % 5 == 0 { -0.06 } else { 0.04 };
            let symthaea_correct = (market_correct + candidate_delta).clamp(0.05, 0.95);
            let market_probability = probability_for_outcome(market_correct, outcome);
            let symthaea_probability = probability_for_outcome(symthaea_correct, outcome);

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

            // Deliberate missing snapshots create explicit missing/stale coverage loss.
            if (event_index * 3 + horizon_index) % 23 != 0 {
                observations.push(ForecastObservation::new(
                    event_id.clone(),
                    "symthaea_synthetic",
                    cutoff,
                    symthaea_probability,
                )?);
            }

            // Negative-control quote: an outcome-revealing market observation after this cutoff.
            // It must never affect this horizon's score. Later horizons have their own later quote.
            observations.push(ForecastObservation::new(
                event_id.clone(),
                "market",
                cutoff + 1,
                if outcome { 0.999 } else { 0.001 },
            )?);
        }
    }

    let report = replay_horizons(&observations, &targets)?;
    println!("horizon,method,eligible,total,coverage,mean_brier,mean_log_loss,paired_brier_difference,ci95_low,ci95_high,bootstrap_share_candidate_better,clusters");

    for (horizon_index, (horizon, _)) in HORIZONS.iter().enumerate() {
        let horizon_targets = targets
            .iter()
            .filter(|target| target.horizon_id() == *horizon)
            .count();
        let methods = ["base_rate", "market", "symthaea_synthetic"];

        let horizon_scores = report
            .scored
            .iter()
            .filter(|score| score.horizon_id == *horizon)
            .collect::<Vec<_>>();
        let mut metric_rows = Vec::new();
        for method in methods {
            let scores = horizon_scores
                .iter()
                .filter(|score| score.forecaster_id == method)
                .collect::<Vec<_>>();
            if scores.is_empty() {
                continue;
            }
            let count = scores.len();
            let mean_brier = scores.iter().map(|score| score.brier_score).sum::<f64>() / count as f64;
            let mean_log_loss = scores.iter().map(|score| score.log_loss).sum::<f64>() / count as f64;
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
            .filter(|score| score.forecaster_id == "symthaea_synthetic");
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
        let bootstrap = paired_cluster_bootstrap(&paired, 10_000, 0xC0FFEE + horizon_index as u64)?;

        for (method, count, coverage, brier, log_loss) in metric_rows {
            if method == "symthaea_synthetic" {
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

fn synthetic_correct_probability(event_index: usize, horizon_index: usize) -> f64 {
    let regime_adjustment = match event_index % 7 {
        0 => -0.10,
        1 => -0.05,
        2 => 0.00,
        3 => 0.02,
        4 => 0.04,
        5 => 0.06,
        _ => 0.08,
    };
    (BASE_CORRECT_PROBABILITY[horizon_index] + regime_adjustment).clamp(0.05, 0.95)
}

fn probability_for_outcome(probability_of_correct_outcome: f64, outcome: bool) -> f64 {
    if outcome {
        probability_of_correct_outcome
    } else {
        1.0 - probability_of_correct_outcome
    }
}
