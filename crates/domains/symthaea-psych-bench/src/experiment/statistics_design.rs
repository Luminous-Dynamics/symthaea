// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Nuisance-topology guards for SYM-ARCH-002A2.
//!
//! A representation/learner/stream realization may be genuinely nested inside
//! one generated environment, or the same realization may be reused across many
//! environments. Those designs have different dependence structures and must not
//! share an uncertainty procedure silently.
//!
//! This module is the fail-closed scientific entry point around the lower-level
//! v1 statistics implementation:
//!
//! - `NestedIndependent` requires nuisance identities to be globally unique;
//! - `CrossedShared` requires the same complete nuisance grid in every environment;
//! - crossed uncertainty uses a two-way environment × nuisance bootstrap;
//! - v1 prospective power refuses crossed designs until a crossed power simulator
//!   is implemented and separately validated.

use crate::experiment::{ExperimentManifest, PairedEstimate};
use crate::experiment_statistics::{
    NestedEnvironmentResult, ProspectivePowerConfig, ProspectivePowerPlan,
    prospective_power_from_dev,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NuisanceTopology {
    /// Every nuisance realization belongs to exactly one environment.
    NestedIndependent,
    /// The same nuisance identities form a complete balanced grid in every environment.
    CrossedShared,
}

fn normalized_environment_grid(
    results: &[NestedEnvironmentResult],
) -> Result<Vec<BTreeMap<String, f64>>, String> {
    if results.len() < 3 {
        return Err("at least three independent environments are required".into());
    }

    let mut environment_ids = BTreeSet::new();
    let mut grids = Vec::with_capacity(results.len());
    for environment in results {
        environment.validate()?;
        let environment_id = environment.environment_digest.to_ascii_lowercase();
        if !environment_ids.insert(environment_id) {
            return Err("duplicate environment digest would create pseudoreplication".into());
        }

        let mut grid = BTreeMap::new();
        for run in &environment.paired_runs {
            let nuisance_id = run.nuisance_digest.to_ascii_lowercase();
            if grid.insert(nuisance_id, run.delta()).is_some() {
                return Err("duplicate nuisance identity within one environment".into());
            }
        }
        grids.push(grid);
    }
    Ok(grids)
}

/// Validate the dependence topology before scientific inference.
///
/// This is intentionally stricter than `NestedEnvironmentResult::validate`, which
/// can validate only one environment at a time and therefore cannot know whether
/// a nuisance realization has been reused across environments.
pub fn validate_nuisance_topology(
    results: &[NestedEnvironmentResult],
    topology: NuisanceTopology,
) -> Result<(), String> {
    let grids = normalized_environment_grid(results)?;

    match topology {
        NuisanceTopology::NestedIndependent => {
            let mut globally_seen = BTreeSet::new();
            for grid in &grids {
                for nuisance_id in grid.keys() {
                    if !globally_seen.insert(nuisance_id.clone()) {
                        return Err(
                            "nested-independent nuisance design reuses a nuisance identity across environments"
                                .into(),
                        );
                    }
                }
            }
        }
        NuisanceTopology::CrossedShared => {
            let reference: BTreeSet<&str> = grids[0].keys().map(String::as_str).collect();
            if reference.len() < 2 {
                return Err(
                    "crossed-shared inference requires at least two nuisance identities".into(),
                );
            }
            for grid in grids.iter().skip(1) {
                let observed: BTreeSet<&str> = grid.keys().map(String::as_str).collect();
                if observed != reference {
                    return Err(
                        "crossed-shared nuisance design requires the same complete nuisance grid in every environment"
                            .into(),
                    );
                }
            }
        }
    }

    Ok(())
}

fn percentile(sorted: &[f64], probability: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    let position = probability.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let weight = position - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

/// Equal-cell estimate with a two-way crossed environment × nuisance bootstrap.
///
/// Each bootstrap replicate independently resamples environment identities and
/// nuisance identities with replacement, then evaluates the Cartesian product of
/// those sampled clusters. Reusing a nuisance draw across all sampled environments
/// preserves the crossed dependence that would be destroyed by independently
/// resampling nuisance runs inside each environment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CrossedHierarchicalEstimate {
    pub n_environments: usize,
    pub n_nuisance_identities: usize,
    pub total_cells: usize,
    pub mean_delta: f64,
    pub ci95_low: f64,
    pub ci95_high: f64,
    pub bootstrap_resamples: usize,
}

pub fn crossed_environment_nuisance_delta_percentile(
    results: &[NestedEnvironmentResult],
    n_resamples: usize,
    seed: u64,
) -> Result<CrossedHierarchicalEstimate, String> {
    validate_nuisance_topology(results, NuisanceTopology::CrossedShared)?;
    if n_resamples < 200 {
        return Err("crossed hierarchical bootstrap requires at least 200 resamples".into());
    }

    let grids = normalized_environment_grid(results)?;
    let nuisance_ids: Vec<String> = grids[0].keys().cloned().collect();
    let n_environments = grids.len();
    let n_nuisance_identities = nuisance_ids.len();
    let total_cells = n_environments
        .checked_mul(n_nuisance_identities)
        .ok_or_else(|| "crossed cell count overflow".to_string())?;

    let observed_sum: f64 = grids
        .iter()
        .flat_map(|grid| nuisance_ids.iter().map(move |id| grid[id]))
        .sum();
    let mean_delta = observed_sum / total_cells as f64;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut bootstrap = Vec::with_capacity(n_resamples);
    for _ in 0..n_resamples {
        let sampled_environments: Vec<usize> = (0..n_environments)
            .map(|_| rng.gen_range(0..n_environments))
            .collect();
        let sampled_nuisance: Vec<usize> = (0..n_nuisance_identities)
            .map(|_| rng.gen_range(0..n_nuisance_identities))
            .collect();

        let mut sum = 0.0;
        for &environment_index in &sampled_environments {
            let grid = &grids[environment_index];
            for &nuisance_index in &sampled_nuisance {
                sum += grid[&nuisance_ids[nuisance_index]];
            }
        }
        bootstrap.push(sum / total_cells as f64);
    }

    bootstrap.sort_by(|left, right| left.total_cmp(right));
    Ok(CrossedHierarchicalEstimate {
        n_environments,
        n_nuisance_identities,
        total_cells,
        mean_delta,
        ci95_low: percentile(&bootstrap, 0.025),
        ci95_high: percentile(&bootstrap, 0.975),
        bootstrap_resamples: n_resamples,
    })
}

/// Convert a crossed estimate into the common paired-estimate shape when only the
/// mean and interval are needed by downstream SESOI classification.
pub fn crossed_as_paired_estimate(estimate: &CrossedHierarchicalEstimate) -> PairedEstimate {
    PairedEstimate {
        n_pairs: estimate.n_environments,
        mean_delta: estimate.mean_delta,
        ci95_low: estimate.ci95_low,
        ci95_high: estimate.ci95_high,
    }
}

/// Topology-checked entry point for v1 prospective power planning.
///
/// The existing power simulator is valid only for genuinely nested nuisance runs.
/// Crossed designs fail closed here rather than silently receiving the nested
/// simulator. A dedicated crossed power simulator should be a separately reviewed
/// follow-up because it must reproduce both environment and shared-nuisance cluster
/// uncertainty inside each simulated study.
pub fn prospective_power_from_dev_topology_checked(
    dev_manifest: &ExperimentManifest,
    dev_results: &[NestedEnvironmentResult],
    config: &ProspectivePowerConfig,
    topology: NuisanceTopology,
) -> Result<ProspectivePowerPlan, String> {
    validate_nuisance_topology(dev_results, topology)?;
    match topology {
        NuisanceTopology::NestedIndependent => {
            prospective_power_from_dev(dev_manifest, dev_results, config)
        }
        NuisanceTopology::CrossedShared => Err(
            "v1 prospective power does not support crossed nuisance designs; use a separately validated crossed power simulator"
                .into(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment::{
        EXPERIMENT_MANIFEST_SCHEMA_V1, SeedManifest, StreamNamespace, TuningStatus,
    };
    use crate::experiment_statistics::{PairedRunResult, PowerDirection};

    fn digest(value: u64) -> String {
        format!("{value:064x}")
    }

    fn environment(id: u64, nuisance: &[(u64, f64)]) -> NestedEnvironmentResult {
        NestedEnvironmentResult {
            environment_digest: digest(10_000 + id),
            paired_runs: nuisance
                .iter()
                .map(|(nuisance_id, delta)| PairedRunResult {
                    nuisance_digest: digest(*nuisance_id),
                    candidate: 0.5 + delta,
                    control: 0.5,
                })
                .collect(),
        }
    }

    #[test]
    fn nested_topology_rejects_cross_environment_nuisance_reuse() {
        let results = vec![
            environment(1, &[(1, 0.1)]),
            environment(2, &[(1, 0.2)]),
            environment(3, &[(3, 0.3)]),
        ];
        assert!(
            validate_nuisance_topology(&results, NuisanceTopology::NestedIndependent).is_err()
        );
    }

    #[test]
    fn crossed_topology_requires_a_complete_shared_grid() {
        let good = vec![
            environment(1, &[(1, 0.10), (2, 0.20)]),
            environment(2, &[(1, 0.00), (2, 0.10)]),
            environment(3, &[(1, -0.10), (2, 0.00)]),
        ];
        validate_nuisance_topology(&good, NuisanceTopology::CrossedShared).unwrap();

        let bad = vec![
            environment(1, &[(1, 0.10), (2, 0.20)]),
            environment(2, &[(1, 0.00), (3, 0.10)]),
            environment(3, &[(1, -0.10), (2, 0.00)]),
        ];
        assert!(validate_nuisance_topology(&bad, NuisanceTopology::CrossedShared).is_err());
    }

    #[test]
    fn crossed_bootstrap_is_deterministic_and_equal_cell_weighted() {
        let results = vec![
            environment(1, &[(1, 0.10), (2, 0.20), (3, 0.30)]),
            environment(2, &[(1, 0.00), (2, 0.10), (3, 0.20)]),
            environment(3, &[(1, -0.10), (2, 0.00), (3, 0.10)]),
            environment(4, &[(1, 0.20), (2, 0.30), (3, 0.40)]),
        ];
        let first = crossed_environment_nuisance_delta_percentile(&results, 500, 42).unwrap();
        let second = crossed_environment_nuisance_delta_percentile(&results, 500, 42).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.n_environments, 4);
        assert_eq!(first.n_nuisance_identities, 3);
        assert_eq!(first.total_cells, 12);
        assert!((first.mean_delta - 0.15).abs() < 1e-12);
        assert!(first.ci95_low <= first.mean_delta);
        assert!(first.mean_delta <= first.ci95_high);
    }

    #[test]
    fn crossed_design_fails_closed_in_v1_power_planner() {
        let results = vec![
            environment(1, &[(1, 0.08), (2, 0.12)]),
            environment(2, &[(1, 0.06), (2, 0.10)]),
            environment(3, &[(1, 0.07), (2, 0.11)]),
        ];
        let manifest = ExperimentManifest {
            schema: EXPERIMENT_MANIFEST_SCHEMA_V1.into(),
            experiment_id: "crossed-power-test".into(),
            experiment_version: "v1".into(),
            code_revision: "deadbeef".into(),
            preregistration_hash: digest(100),
            generator_hash: digest(101),
            stream_namespace: StreamNamespace::Dev,
            tuning_status: TuningStatus::Exploratory,
            prior_results_observed: true,
            seed_manifest: SeedManifest {
                environment_seeds: vec![1, 2, 3],
                representation_seeds: vec![1, 2],
                learner_seeds: vec![],
                stream_seeds: vec![],
            },
            primary_hypothesis: "crossed power is fail-closed in v1".into(),
            primary_comparator: "matched control".into(),
            sesoi: 0.05,
        };
        let config = ProspectivePowerConfig {
            environment_counts: vec![3, 5],
            runs_per_environment: 2,
            simulation_trials: 100,
            bootstrap_resamples: 100,
            target_power: 0.80,
            sesoi: 0.05,
            planning_effect: 0.10,
            residual_scale: 1.0,
            direction: PowerDirection::MeaningfulGain,
            seed: 7,
        };

        let error = prospective_power_from_dev_topology_checked(
            &manifest,
            &results,
            &config,
            NuisanceTopology::CrossedShared,
        )
        .unwrap_err();
        assert!(error.contains("does not support crossed nuisance designs"));
    }
}
