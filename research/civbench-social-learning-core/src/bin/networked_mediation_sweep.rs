// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Network density × algorithmic mediation sensitivity sweep.
//!
//! Each condition uses the same seed range, with graph topology generated
//! deterministically from the seed and shortcut probability. Output is a
//! synthetic mechanism sensitivity study, not a fit to human data.

use civbench_social_learning_core::networked_mediation::{
    simulate_networked_mediation, NetworkedMediationConfig, RecommenderObjective,
};

const REPLICATES: u64 = 256;
const SHORTCUT_PROBABILITIES: [f64; 4] = [0.0, 0.02, 0.10, 0.50];
const MEDIATION_FRACTIONS: [f64; 5] = [0.0, 0.25, 0.50, 0.75, 1.0];
const RECOMMENDER_OBJECTIVES: [(&str, RecommenderObjective, f64); 2] = [
    ("performance_only", RecommenderObjective::PerformanceOnly, 0.0),
    ("performance_plus_lineage_diversity", RecommenderObjective::PerformanceDiversity { novelty_weight: 0.5 }, 0.5),
];

#[derive(Default)]
struct OnlineStats {
    n: u64,
    sum: f64,
    sum_squares: f64,
}

impl OnlineStats {
    fn push(&mut self, value: f64) {
        self.n += 1;
        self.sum += value;
        self.sum_squares += value * value;
    }

    fn mean(&self) -> Option<f64> {
        (self.n > 0).then(|| self.sum / self.n as f64)
    }

    fn ci95_half_width(&self) -> Option<f64> {
        if self.n < 2 {
            return None;
        }
        let n = self.n as f64;
        let variance = ((self.sum_squares - self.sum * self.sum / n) / (n - 1.0)).max(0.0);
        Some(1.96 * variance.sqrt() / n.sqrt())
    }
}

fn fmt(value: Option<f64>) -> String {
    value.map(|number| format!("{number:.6}")).unwrap_or_default()
}

fn main() {
    println!(
        "shortcut_probability,mediation_fraction,recommender_objective,novelty_weight,generation,replicates,actual_network_density,mean_quality,mean_quality_ci95,max_quality,knowledge_coverage,seed_lineage_prevalence,unique_nonzero_lineages,effective_lineage_diversity,lineage_hhi,individual_innovations,observations,successful_acquisitions,peer_adoptions,mediated_adoptions,peer_acquisitions,mediated_acquisitions"
    );

    for shortcut_probability in SHORTCUT_PROBABILITIES {
        for mediation_fraction in MEDIATION_FRACTIONS {
            for (objective_name, recommender_objective, novelty_weight) in RECOMMENDER_OBJECTIVES {
            let mut density = OnlineStats::default();
            let mut mean_quality: Vec<OnlineStats> = (0..8).map(|_| OnlineStats::default()).collect();
            let mut max_quality: Vec<OnlineStats> = (0..8).map(|_| OnlineStats::default()).collect();
            let mut coverage: Vec<OnlineStats> = (0..8).map(|_| OnlineStats::default()).collect();
            let mut seed_prevalence: Vec<OnlineStats> = (0..8).map(|_| OnlineStats::default()).collect();
            let mut unique_lineages: Vec<OnlineStats> = (0..8).map(|_| OnlineStats::default()).collect();
            let mut effective_diversity: Vec<OnlineStats> = (0..8).map(|_| OnlineStats::default()).collect();
            let mut lineage_hhi: Vec<OnlineStats> = (0..8).map(|_| OnlineStats::default()).collect();
            let mut innovations: Vec<OnlineStats> = (0..8).map(|_| OnlineStats::default()).collect();
            let mut observations: Vec<OnlineStats> = (0..8).map(|_| OnlineStats::default()).collect();
            let mut acquisitions: Vec<OnlineStats> = (0..8).map(|_| OnlineStats::default()).collect();
            let mut peer_adoptions: Vec<OnlineStats> = (0..8).map(|_| OnlineStats::default()).collect();
            let mut mediated_adoptions: Vec<OnlineStats> = (0..8).map(|_| OnlineStats::default()).collect();
            let mut peer_acquisitions: Vec<OnlineStats> = (0..8).map(|_| OnlineStats::default()).collect();
            let mut mediated_acquisitions: Vec<OnlineStats> = (0..8).map(|_| OnlineStats::default()).collect();

            for seed in 0..REPLICATES {
                let config = NetworkedMediationConfig {
                    shortcut_probability,
                    mediation_fraction,
                    recommender_objective,
                    ..NetworkedMediationConfig::default()
                };
                let run = simulate_networked_mediation(config, seed)
                    .expect("predeclared mediation sweep condition must validate");
                density.push(run.graph.density());
                for metrics in run.metrics() {
                    let generation = metrics.generation;
                    let n = config.population_size as f64;
                    mean_quality[generation].push(metrics.mean_quality);
                    max_quality[generation].push(f64::from(metrics.max_quality));
                    coverage[generation].push(metrics.knowledge_coverage);
                    seed_prevalence[generation].push(metrics.seed_lineage_prevalence);
                    unique_lineages[generation].push(metrics.unique_nonzero_lineages as f64);
                    effective_diversity[generation].push(metrics.effective_lineage_diversity);
                    lineage_hhi[generation].push(metrics.lineage_hhi);
                    innovations[generation].push(metrics.individual_innovations as f64 / n);
                    observations[generation].push(metrics.observations as f64 / n);
                    acquisitions[generation].push(metrics.successful_acquisitions as f64 / n);
                    peer_adoptions[generation].push(metrics.peer_adoptions as f64 / n);
                    mediated_adoptions[generation].push(metrics.mediated_adoptions as f64 / n);
                    peer_acquisitions[generation].push(metrics.peer_acquisitions as f64 / n);
                    mediated_acquisitions[generation].push(metrics.mediated_acquisitions as f64 / n);
                }
            }

            for generation in 0..8 {
                println!(
                    "{:.6},{:.6},{},{:.6},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                    shortcut_probability,
                    mediation_fraction,
                    objective_name,
                    novelty_weight,
                    generation,
                    REPLICATES,
                    fmt(density.mean()),
                    fmt(mean_quality[generation].mean()),
                    fmt(mean_quality[generation].ci95_half_width()),
                    fmt(max_quality[generation].mean()),
                    fmt(coverage[generation].mean()),
                    fmt(seed_prevalence[generation].mean()),
                    fmt(unique_lineages[generation].mean()),
                    fmt(effective_diversity[generation].mean()),
                    fmt(lineage_hhi[generation].mean()),
                    fmt(innovations[generation].mean()),
                    fmt(observations[generation].mean()),
                    fmt(acquisitions[generation].mean()),
                    fmt(peer_adoptions[generation].mean()),
                    fmt(mediated_adoptions[generation].mean()),
                    fmt(peer_acquisitions[generation].mean()),
                    fmt(mediated_acquisitions[generation].mean()),
                );
            }
            }
        }
    }
}
