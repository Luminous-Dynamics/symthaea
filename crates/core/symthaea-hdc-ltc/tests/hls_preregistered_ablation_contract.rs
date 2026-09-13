// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::HashSet;
use symthaea_hdc_ltc::{
    ExactLearningAblationPlan, run_exact_learning_ablation,
};

#[test]
fn public_smoke_plan_is_deterministic_paired_and_disjoint() {
    let plan = ExactLearningAblationPlan::smoke();
    let train = plan.train_world_seeds.iter().copied().collect::<HashSet<_>>();
    let test = plan.test_world_seeds.iter().copied().collect::<HashSet<_>>();
    assert_eq!(train.len(), plan.train_world_seeds.len());
    assert_eq!(test.len(), plan.test_world_seeds.len());
    assert!(train.is_disjoint(&test));

    let first = run_exact_learning_ablation(&plan).unwrap();
    let second = run_exact_learning_ablation(&plan).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.held_out_worlds.len(), plan.test_world_seeds.len());
    for (comparison, seed) in first.held_out_worlds.iter().zip(&plan.test_world_seeds) {
        assert_eq!(comparison.seed, *seed);
        assert_eq!(comparison.frozen.query_count, comparison.trained.query_count);
        assert!(comparison.mean_loss_delta.is_finite());
        assert!(comparison.accuracy_delta.is_finite());
    }
}

#[test]
fn research_v0_plan_is_frozen_to_declared_scale_and_seed_counts() {
    let plan = ExactLearningAblationPlan::research_v0();
    assert_eq!(plan.hls_config.dim, 512);
    assert!(plan.hls_config.state_norm_limit.is_infinite());
    assert_eq!(plan.benchmark_template.entities, 16);
    assert_eq!(plan.benchmark_template.objects, 32);
    assert_eq!(plan.benchmark_template.locations, 8);
    assert_eq!(plan.benchmark_template.events, 1_000);
    assert_eq!(plan.train_world_seeds.len(), 12);
    assert_eq!(plan.test_world_seeds.len(), 24);
    assert_eq!(plan.train_world_seeds[0], 10_001);
    assert_eq!(*plan.train_world_seeds.last().unwrap(), 10_012);
    assert_eq!(plan.test_world_seeds[0], 20_001);
    assert_eq!(*plan.test_world_seeds.last().unwrap(), 20_024);
}
