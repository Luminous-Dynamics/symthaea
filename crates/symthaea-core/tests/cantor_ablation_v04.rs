// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use std::time::Instant;
use symthaea_core::hdc::cantor_pyramid::{
    BundleMode, CantorHdcConfig, CantorRouter, HashRouter, PyramidCantorVector, RandomRouter,
    SemanticRouter,
};
use symthaea_core::hdc::unified_hv::ContinuousHV;

/// HCH Ablation v0.4
///
/// Scope:
/// 1. Multi-seed statistics (Mean, StdDev)
/// 2. Cantor-64K Control (Fair vs Flat-64K)
/// 3. Router Comparison (Hash vs Random vs Semantic)
/// 4. Higher object load stress test (N=512)

struct TrialResult {
    accuracy: f32,
    margin: f32,
    latency_ms: f32,
}

fn run_trial(
    seed: u64,
    config: CantorHdcConfig,
    n_objects: usize,
    router: &dyn CantorRouter,
) -> TrialResult {
    let start = Instant::now();
    let leaf_dim = config.leaf_dim;
    let mut pyramid = PyramidCantorVector::new(config, None);

    // Codebook size = 2 * n_objects for a bit of retrieval challenge
    let codebook_size = (n_objects * 2).max(100);
    let codebook: Vec<ContinuousHV> = (0..codebook_size)
        .map(|i| ContinuousHV::random(leaf_dim, seed + 1000 + i as u64))
        .collect();

    let mut stored = Vec::new();
    for i in 0..n_objects {
        let role = ContinuousHV::random(leaf_dim, seed + 2000 + i as u64);
        let val_idx = i % codebook_size;
        let binding = role.bind(&codebook[val_idx]);

        let leaf_idx = router.route(&role, &ContinuousHV::zero(leaf_dim), config.branching);
        let leaf = pyramid.find_node(1, leaf_idx).unwrap().clone();
        pyramid.bundle_at_node(&leaf, &binding);
        stored.push((role, val_idx, leaf_idx));
    }

    let mut hits = 0;
    let mut margin = 0.0;
    for (role, correct_idx, leaf_idx) in &stored {
        let leaf = pyramid.find_node(1, *leaf_idx).unwrap();
        let recovered = ContinuousHV::from_slice(pyramid.node_data(leaf)).bind(&role.inverse());

        let mut best_sim = -1.0;
        let mut second_best_sim = -1.0;
        let mut best_idx = 0;

        for (idx, cand) in codebook.iter().enumerate() {
            let sim = recovered.similarity(cand);
            if sim > best_sim {
                second_best_sim = best_sim;
                best_sim = sim;
                best_idx = idx;
            } else if sim > second_best_sim {
                second_best_sim = sim;
            }
        }

        if best_idx == *correct_idx {
            hits += 1;
        }
        margin += best_sim - second_best_sim;
    }

    TrialResult {
        accuracy: hits as f32 / n_objects as f32,
        margin: margin / n_objects as f32,
        latency_ms: start.elapsed().as_secs_f32() * 1000.0,
    }
}

fn run_flat_trial(seed: u64, dim: usize, n_objects: usize) -> TrialResult {
    let start = Instant::now();
    let config = CantorHdcConfig {
        total_dim: dim,
        levels: 1,
        branching: 1,
        leaf_dim: dim,
        bundle_mode: BundleMode::UnitNormalize,
    };
    let mut pyramid = PyramidCantorVector::new(config, None);
    let node = pyramid.find_node(0, 0).unwrap().clone();

    let codebook_size = (n_objects * 2).max(100);
    let codebook: Vec<ContinuousHV> = (0..codebook_size)
        .map(|i| ContinuousHV::random(dim, seed + 3000 + i as u64))
        .collect();

    let mut stored = Vec::new();
    for i in 0..n_objects {
        let role = ContinuousHV::random(dim, seed + 4000 + i as u64);
        let val_idx = i % codebook_size;
        let binding = role.bind(&codebook[val_idx]);
        pyramid.bundle_at_node(&node, &binding);
        stored.push((role, val_idx));
    }

    let mut hits = 0;
    let mut margin = 0.0;
    for (role, correct_idx) in &stored {
        let recovered = ContinuousHV::from_slice(pyramid.node_data(&node)).bind(&role.inverse());

        let mut best_sim = -1.0;
        let mut second_best_sim = -1.0;
        let mut best_idx = 0;

        for (idx, cand) in codebook.iter().enumerate() {
            let sim = recovered.similarity(cand);
            if sim > best_sim {
                second_best_sim = best_sim;
                best_sim = sim;
                best_idx = idx;
            } else if sim > second_best_sim {
                second_best_sim = sim;
            }
        }

        if best_idx == *correct_idx {
            hits += 1;
        }
        margin += best_sim - second_best_sim;
    }

    TrialResult {
        accuracy: hits as f32 / n_objects as f32,
        margin: margin / n_objects as f32,
        latency_ms: start.elapsed().as_secs_f32() * 1000.0,
    }
}

#[test]
fn test_hch_v04_multi_seed_statistics() {
    let n_seeds = 5;
    let n_objects = 256;

    let config_16k = CantorHdcConfig {
        total_dim: 16384,
        levels: 2,
        branching: 16,
        leaf_dim: 1024,
        bundle_mode: BundleMode::UnitNormalize,
    };

    println!("\n--- HCH v0.4 Multi-Seed Statistics (N=256 objects, 5 trials) ---");
    println!("Model      | Accuracy (Mean ± Std) | Margin (Mean ± Std) | Latency (Mean)");
    println!("-----------|-----------------------|---------------------|--------------");

    let mut flat_accs = Vec::new();
    let mut flat_margins = Vec::new();
    let mut flat_times = Vec::new();

    let mut cantor_accs = Vec::new();
    let mut cantor_margins = Vec::new();
    let mut cantor_times = Vec::new();

    let hash_router = HashRouter;

    for seed in 0..n_seeds as u64 {
        let flat = run_flat_trial(seed, 16384, n_objects);
        flat_accs.push(flat.accuracy);
        flat_margins.push(flat.margin);
        flat_times.push(flat.latency_ms);

        let cantor = run_trial(seed, config_16k, n_objects, &hash_router);
        cantor_accs.push(cantor.accuracy);
        cantor_margins.push(cantor.margin);
        cantor_times.push(cantor.latency_ms);
    }

    fn stats(v: &[f32]) -> (f32, f32) {
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let variance = v.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32;
        (mean, variance.sqrt())
    }

    let (f_acc_m, f_acc_s) = stats(&flat_accs);
    let (f_mar_m, f_mar_s) = stats(&flat_margins);
    let f_time = flat_times.iter().sum::<f32>() / n_seeds as f32;

    let (c_acc_m, c_acc_s) = stats(&cantor_accs);
    let (c_mar_m, c_mar_s) = stats(&cantor_margins);
    let c_time = cantor_times.iter().sum::<f32>() / n_seeds as f32;

    println!(
        "Flat-16K   | {:.1}% ± {:.1}%         | {:.4} ± {:.4}     | {:.1}ms",
        f_acc_m * 100.0,
        f_acc_s * 100.0,
        f_mar_m,
        f_mar_s,
        f_time
    );
    println!(
        "Cantor-16K | {:.1}% ± {:.1}%         | {:.4} ± {:.4}     | {:.1}ms",
        c_acc_m * 100.0,
        c_acc_s * 100.0,
        c_mar_m,
        c_mar_s,
        c_time
    );
}

#[test]
fn test_hch_v04_scaling_64k() {
    let seed = 42;
    let n_objects = 512; // Higher load for 64K

    println!("\n--- HCH v0.4 Scaling 64K (N=512 objects) ---");
    println!("Model      | Accuracy | Margin | Latency");
    println!("-----------|----------|--------|--------");

    // Flat 64K
    let flat_64 = run_flat_trial(seed, 65536, n_objects);
    println!(
        "Flat-64K   | {:.1}%    | {:.4} | {:.1}ms",
        flat_64.accuracy * 100.0,
        flat_64.margin,
        flat_64.latency_ms
    );

    // Cantor 64K (16 leaves of 4096D)
    let config_64 = CantorHdcConfig {
        total_dim: 65536,
        levels: 2,
        branching: 16,
        leaf_dim: 4096,
        bundle_mode: BundleMode::UnitNormalize,
    };
    let hash_router = HashRouter;
    let cantor_64 = run_trial(seed, config_64, n_objects, &hash_router);
    println!(
        "Cantor-64K | {:.1}%    | {:.4} | {:.1}ms",
        cantor_64.accuracy * 100.0,
        cantor_64.margin,
        cantor_64.latency_ms
    );
}

#[test]
fn test_hch_v04_router_comparison() {
    let seed = 123;
    let n_objects = 256;
    let config = CantorHdcConfig {
        total_dim: 16384,
        levels: 2,
        branching: 16,
        leaf_dim: 1024,
        bundle_mode: BundleMode::UnitNormalize,
    };

    println!("\n--- HCH v0.4 Router Comparison (N=256) ---");
    println!("Router     | Accuracy | Margin | Latency");
    println!("-----------|----------|--------|--------");

    let hash_router = HashRouter;
    let hash_res = run_trial(seed, config, n_objects, &hash_router);
    println!(
        "Hash       | {:.1}%    | {:.4} | {:.1}ms",
        hash_res.accuracy * 100.0,
        hash_res.margin,
        hash_res.latency_ms
    );

    let random_router = RandomRouter { seed };
    let rand_res = run_trial(seed, config, n_objects, &random_router);
    println!(
        "Random     | {:.1}%    | {:.4} | {:.1}ms",
        rand_res.accuracy * 100.0,
        rand_res.margin,
        rand_res.latency_ms
    );

    let semantic_router = SemanticRouter;
    let sem_res = run_trial(seed, config, n_objects, &semantic_router);
    println!(
        "Semantic v0| {:.1}%    | {:.4} | {:.1}ms",
        sem_res.accuracy * 100.0,
        sem_res.margin,
        sem_res.latency_ms
    );
}
