// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use std::time::Instant;
use symthaea_core::hdc::cantor_pyramid::{
    BundleMode, CantorHdcConfig, CantorRouter, HashRouter, PyramidCantorVector,
};
use symthaea_core::hdc::unified_hv::ContinuousHV;

/// HCH Ablation v0.3
///
/// Focus:
/// 1. codebook retrieval
/// 2. similarity margins
/// 3. same-leaf saturation
/// 4. unknown-leaf routing
/// 5. separate quality vs latency benchmarks

#[test]
fn test_codebook_retrieval_margins() {
    let dim = 1024;
    let n_objects = 100;

    // Create a codebook of 1000 candidate values
    let codebook: Vec<ContinuousHV> = (0..1000)
        .map(|i| ContinuousHV::random(dim, 5000 + i as u64))
        .collect();

    let mut config = CantorHdcConfig::default();
    config.total_dim = 16384;
    config.leaf_dim = dim;
    config.branching = 16;
    config.levels = 2;
    config.bundle_mode = BundleMode::UnitNormalize;

    let mut pyramid = PyramidCantorVector::new(config, None);
    let router = HashRouter;

    // Store N objects
    let mut stored_indices = Vec::new();
    for i in 0..n_objects {
        let role = ContinuousHV::random(dim, 100 + i as u64);
        let val_idx = i % 1000;
        let val = &codebook[val_idx];
        let binding = role.bind(val);

        let leaf_idx = router.route(&role, &ContinuousHV::zero(dim), config.branching);
        let leaf = pyramid.find_node(1, leaf_idx).unwrap().clone();
        pyramid.bundle_at_node(&leaf, &binding);
        stored_indices.push((role, val_idx));
    }

    println!("\n--- Codebook Retrieval Margins (HCH v0.3) ---");
    let mut top1_hits = 0;
    let mut avg_margin = 0.0;

    for (role, correct_idx) in &stored_indices {
        let leaf_idx = router.route(role, &ContinuousHV::zero(dim), config.branching);
        let leaf = pyramid.find_node(1, leaf_idx).unwrap();
        let recovered = ContinuousHV::from_slice(pyramid.node_data(leaf)).bind(&role.inverse());

        let mut best_sim = -1.0;
        let mut second_best_sim = -1.0;
        let mut best_idx = 0;

        for (idx, candidate) in codebook.iter().enumerate() {
            let sim = recovered.similarity(candidate);
            if sim > best_sim {
                second_best_sim = best_sim;
                best_sim = sim;
                best_idx = idx;
            } else if sim > second_best_sim {
                second_best_sim = sim;
            }
        }

        if best_idx == *correct_idx {
            top1_hits += 1;
        }
        avg_margin += best_sim - second_best_sim;
    }

    println!("Top-1 Accuracy: {}/{}", top1_hits, n_objects);
    println!(
        "Avg Similarity Margin: {:.4}",
        avg_margin / n_objects as f32
    );

    assert!(top1_hits > 0, "Should have at least some hits");
}

#[test]
fn test_same_leaf_saturation_v03() {
    let dim = 1024;
    let n_objects_list = [10, 50, 100, 200];

    println!("\n--- Same-Leaf Saturation v0.3 ---");
    println!("N Objects | Top-1 Accuracy | Avg Margin");
    println!("----------|----------------|-----------");

    for &n in &n_objects_list {
        let config = CantorHdcConfig {
            total_dim: dim, // Just a single leaf test
            levels: 1,
            branching: 1,
            leaf_dim: dim,
            bundle_mode: BundleMode::UnitNormalize,
        };
        let mut pyramid = PyramidCantorVector::new(config, None);
        let leaf = pyramid.find_node(0, 0).unwrap().clone();

        let codebook: Vec<ContinuousHV> = (0..500)
            .map(|i| ContinuousHV::random(dim, 6000 + i as u64))
            .collect();

        let mut stored = Vec::new();
        for i in 0..n {
            let role = ContinuousHV::random(dim, 7000 + i as u64);
            let val_idx = i % 500;
            let binding = role.bind(&codebook[val_idx]);
            pyramid.bundle_at_node(&leaf, &binding);
            stored.push((role, val_idx));
        }

        let mut hits = 0;
        let mut margin = 0.0;
        for (role, correct_idx) in &stored {
            let recovered =
                ContinuousHV::from_slice(pyramid.node_data(&leaf)).bind(&role.inverse());
            let mut best_sim = -1.0;
            let mut second_best_sim = -1.0;
            let mut best_idx = 0;

            for (idx, candidate) in codebook.iter().enumerate() {
                let sim = recovered.similarity(candidate);
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
        println!(
            "{:9} | {:14} | {:.4}",
            n,
            format!("{}/{}", hits, n),
            margin / n as f32
        );
    }
}

#[test]
fn test_fair_quality_comparison() {
    let total_objects = 256;
    let dim = 16384;
    let leaf_dim = 1024; // 16 leaves

    println!("\n--- Fair Representation Quality Comparison (N=256 objects) ---");
    println!("Config     | Top-1 Acc | Avg Margin | Time (ms)");
    println!("-----------|-----------|------------|----------");

    // 1. Flat Baseline
    {
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

        let codebook: Vec<ContinuousHV> = (0..512)
            .map(|i| ContinuousHV::random(dim, 8000 + i as u64))
            .collect();
        let mut stored = Vec::new();
        for i in 0..total_objects {
            let role = ContinuousHV::random(dim, 9000 + i as u64);
            let val_idx = i % 512;
            let binding = role.bind(&codebook[val_idx]);
            pyramid.bundle_at_node(&node, &binding);
            stored.push((role, val_idx));
        }

        let mut hits = 0;
        let mut margin = 0.0;
        for (role, correct_idx) in &stored {
            let recovered =
                ContinuousHV::from_slice(pyramid.node_data(&node)).bind(&role.inverse());
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
        println!(
            "Flat-16K   | {:9} | {:.4}     | {:?}",
            format!("{}/{}", hits, total_objects),
            margin / total_objects as f32,
            start.elapsed()
        );
    }

    // 2. Cantor (HCH)
    {
        let start = Instant::now();
        let config = CantorHdcConfig {
            total_dim: dim,
            levels: 2,
            branching: 16,
            leaf_dim: leaf_dim,
            bundle_mode: BundleMode::UnitNormalize,
        };
        let mut pyramid = PyramidCantorVector::new(config, None);
        let router = HashRouter;

        let codebook: Vec<ContinuousHV> = (0..512)
            .map(|i| ContinuousHV::random(leaf_dim, 8000 + i as u64))
            .collect();
        let mut stored = Vec::new();
        for i in 0..total_objects {
            let role = ContinuousHV::random(leaf_dim, 9000 + i as u64);
            let val_idx = i % 512;
            let binding = role.bind(&codebook[val_idx]);

            let leaf_idx = router.route(&role, &ContinuousHV::zero(leaf_dim), 16);
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
        println!(
            "Cantor-16K | {:9} | {:.4}     | {:?}",
            format!("{}/{}", hits, total_objects),
            margin / total_objects as f32,
            start.elapsed()
        );
    }
}

#[test]
fn test_broadcast_loss_reconstruction_accuracy() {
    let config = CantorHdcConfig {
        total_dim: 16384,
        levels: 4,
        branching: 4,
        leaf_dim: 256,
        bundle_mode: BundleMode::UnitNormalize,
    };
    let mut pyramid = PyramidCantorVector::new(config, None);

    // Codebook for L3 (256D)
    let codebook: Vec<ContinuousHV> = (0..100)
        .map(|i| ContinuousHV::random(256, 1111 + i as u64))
        .collect();

    // Store 10 objects in L3,0
    for i in 0..10 {
        let role = ContinuousHV::random(256, 2222 + i as u64);
        let binding = role.bind(&codebook[i]);
        let leaf = pyramid.find_node(3, 0).unwrap().clone();
        pyramid.bundle_at_node(&leaf, &binding);
    }

    println!("\n--- Broadcast Loss Reconstruction Accuracy ---");
    println!("Level | Top-1 Accuracy | Avg Margin");
    println!("------|----------------|-----------");

    for level in (0..4).rev() {
        if level < 3 {
            pyramid.broadcast_up(level);
        }

        let node = pyramid.find_node(level, 0).unwrap();
        let node_dim = node.range.len();
        let data = pyramid.node_data(node);
        let hv = ContinuousHV::from_slice(data);

        let mut hits = 0;
        let mut margin = 0.0;

        for i in 0..10 {
            let role = ContinuousHV::random(256, 2222 + i as u64);
            let recovered = hv.bind(&role.dilate(node_dim).inverse());

            let mut best_sim = -1.0;
            let mut second_best_sim = -1.0;
            let mut best_idx = 0;

            for (idx, cand) in codebook.iter().enumerate() {
                let sim = recovered.similarity(&cand.dilate(node_dim));
                if sim > best_sim {
                    second_best_sim = best_sim;
                    best_sim = sim;
                    best_idx = idx;
                } else if sim > second_best_sim {
                    second_best_sim = sim;
                }
            }

            if best_idx == i {
                hits += 1;
            }
            margin += best_sim - second_best_sim;
        }
        println!(
            "L{}    | {:14} | {:.4}",
            level,
            format!("{}/10", hits),
            margin / 10.0
        );
    }
}
