// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use std::time::Instant;
use symthaea_core::hdc::cantor_pyramid::{CantorHdcConfig, PyramidCantorVector};
use symthaea_core::hdc::unified_hv::ContinuousHV;

/// HCH Ablation v0.2
///
/// Scope:
/// - same-leaf saturation
/// - unknown-leaf retrieval
/// - Flat-16K vs Cantor-16K vs Flat-64K vs Cantor-64K
/// - broadcast loss metrics
/// - latency benchmarks

#[test]
fn test_same_leaf_saturation() {
    let leaf_dims = [256, 512, 1024, 2048];
    println!("\n--- Same-Leaf Saturation ---");
    println!("Leaf Dim | N=10 | N=50 | N=100");
    println!("---------|------|------|------");

    for &dim in &leaf_dims {
        let config = CantorHdcConfig {
            total_dim: 16384,
            levels: 2,
            branching: 16384 / dim,
            leaf_dim: dim,
            ..CantorHdcConfig::default()
        };

        print!("{:8} | ", dim);
        for &n in &[10, 50, 100] {
            let mut pyramid = PyramidCantorVector::new(config, None);
            let leaf = pyramid.find_node(1, 0).unwrap().clone();

            let target_r = ContinuousHV::random(dim, 777);
            let target_v = ContinuousHV::random(dim, 888);
            let target_b = target_r.bind(&target_v);
            pyramid.bundle_at_node(&leaf, &target_b);

            for i in 0..n {
                let r = ContinuousHV::random(dim, 1000 + i as u64);
                let v = ContinuousHV::random(dim, 2000 + i as u64);
                pyramid.bundle_at_node(&leaf, &r.bind(&v));
            }

            let recovered =
                ContinuousHV::from_slice(pyramid.node_data(&leaf)).bind(&target_r.inverse());
            let sim = recovered.similarity(&target_v);
            print!("{:.4} | ", sim);
        }
        println!();
    }
}

#[test]
fn test_unknown_leaf_retrieval() {
    let config = CantorHdcConfig {
        total_dim: 16384,
        levels: 2,
        branching: 16,
        leaf_dim: 1024,
        ..CantorHdcConfig::default()
    };
    let mut pyramid = PyramidCantorVector::new(config, None);

    // Store 16 distinct objects in 16 different leaves
    let mut targets = Vec::new();
    for i in 0..16 {
        let r = ContinuousHV::random(1024, 100 + i as u64);
        let v = ContinuousHV::random(1024, 200 + i as u64);
        let b = r.bind(&v);
        let leaf = pyramid.find_node(1, i).unwrap().clone();
        pyramid.bundle_at_node(&leaf, &b);
        targets.push((r, v));
    }

    println!("\n--- Unknown-Leaf Retrieval ---");
    let mut top1_hits = 0;
    for (i, (r, v)) in targets.iter().enumerate() {
        let mut best_sim = -1.0;
        let mut best_leaf = 0;

        for leaf_idx in 0..16 {
            let leaf = pyramid.find_node(1, leaf_idx).unwrap();
            let recovered = ContinuousHV::from_slice(pyramid.node_data(leaf)).bind(&r.inverse());
            let sim = recovered.similarity(v);
            if sim > best_sim {
                best_sim = sim;
                best_leaf = leaf_idx;
            }
        }

        if best_leaf == i {
            top1_hits += 1;
        }
    }
    println!("Top-1 Accuracy: {}/16", top1_hits);
    assert_eq!(
        top1_hits, 16,
        "Should perfectly retrieve from correct leaf when orthogonal"
    );
}

#[test]
fn test_flat_vs_cantor_scaling() {
    let configs = [
        ("Flat-16K", 16384, false),
        ("Cantor-16K", 16384, true),
        ("Flat-64K", 65536, false),
        ("Cantor-64K", 65536, true),
    ];

    println!("\n--- Flat vs Cantor Scaling (N=100 bundles) ---");
    println!("Config     | Similarity | Latency (ms)");
    println!("-----------|------------|-------------");

    for (name, dim, hierarchical) in configs {
        let start = Instant::now();
        let config = CantorHdcConfig {
            total_dim: dim,
            levels: if hierarchical { 4 } else { 1 },
            branching: 4,
            leaf_dim: dim / 64, // For hierarchical
            ..CantorHdcConfig::default()
        };

        let mut pyramid = PyramidCantorVector::new(config, None);
        let target_r = ContinuousHV::random(dim, 42);
        let target_v = ContinuousHV::random(dim, 43);
        let target_b = target_r.bind(&target_v);

        if hierarchical {
            let leaf = pyramid.find_node(3, 0).unwrap().clone();
            let target_b_leaf = ContinuousHV::from_slice(&target_b.as_slice()[0..config.leaf_dim]);
            pyramid.bundle_at_node(&leaf, &target_b_leaf);

            for i in 0..100 {
                let r = ContinuousHV::random(config.leaf_dim, 1000 + i as u64);
                let v = ContinuousHV::random(config.leaf_dim, 2000 + i as u64);
                let leaf_idx = i % 64;
                let l = pyramid.find_node(3, leaf_idx).unwrap().clone();
                pyramid.bundle_at_node(&l, &r.bind(&v));
            }

            let leaf = pyramid.find_node(3, 0).unwrap();
            let target_r_leaf = ContinuousHV::from_slice(&target_r.as_slice()[0..config.leaf_dim]);
            let target_v_leaf = ContinuousHV::from_slice(&target_v.as_slice()[0..config.leaf_dim]);
            let recovered =
                ContinuousHV::from_slice(pyramid.node_data(leaf)).bind(&target_r_leaf.inverse());
            let sim = recovered.similarity(&target_v_leaf);
            println!("{:10} | {:.4}     | {:?}", name, sim, start.elapsed());
        } else {
            let l0 = pyramid.find_node(0, 0).unwrap().clone();
            pyramid.bundle_at_node(&l0, &target_b);

            for i in 0..100 {
                let r = ContinuousHV::random(dim, 1000 + i as u64);
                let v = ContinuousHV::random(dim, 2000 + i as u64);
                pyramid.bundle_at_node(&l0, &r.bind(&v));
            }

            let recovered =
                ContinuousHV::from_slice(pyramid.node_data(&l0)).bind(&target_r.inverse());
            let sim = recovered.similarity(&target_v);
            println!("{:10} | {:.4}     | {:?}", name, sim, start.elapsed());
        }
    }
}

#[test]
fn test_broadcast_loss_metrics() {
    let config = CantorHdcConfig {
        total_dim: 16384,
        levels: 4,
        branching: 4,
        leaf_dim: 256,
        ..CantorHdcConfig::default()
    };
    let mut pyramid = PyramidCantorVector::new(config, None);

    // 1. Store a unique signal in ONE leaf (L3,0)
    let signal = ContinuousHV::random(256, 999);
    let leaf = pyramid.find_node(3, 0).unwrap().clone();
    pyramid.bundle_at_node(&leaf, &signal);

    println!("\n--- Broadcast Loss Metrics ---");
    println!("Level | Similarity to L3 Signal");
    println!("------|-----------------------");

    // 2. Broadcast up step by step
    for level in (0..3).rev() {
        // Find nodes at 'level' that are ancestors of L3,0
        // In our deterministic tiling, L3,0 is a child of L2,0, which is child of L1,0, which is child of L0,0
        pyramid.broadcast_up(level); // Actually this broadcasts ALL children of this node

        let node = pyramid.find_node(level, 0).unwrap();
        let data = pyramid.node_data(node);
        let hv = ContinuousHV::from_slice(data);

        // Signal is 256D, level node might be larger. Dilate signal to match.
        let sim = hv.similarity(&signal.dilate(node.range.len()));
        println!("L{}    | {:.4}", level, sim);
    }
}
