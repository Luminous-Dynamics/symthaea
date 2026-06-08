// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use symthaea_core::hdc::cantor_pyramid::{
    BundleMode, CantorHdcConfig, CantorRouter, HashRouter, HypercubeRouter, LoadBalancedHashRouter,
    PrefixMaxRouter, PrototypeRouter, PyramidCantorVector, RandomRouter,
};
use symthaea_core::hdc::unified_hv::ContinuousHV;

/// HCH Ablation v0.5
///
/// Scope:
/// 1. Comprehensive Router Comparison (Load distribution, Top-1/3 Accuracy)
/// 2. Oracle bound establish
/// 3. Margin-based abstention calibration
/// 4. Load entropy metrics

struct V05Result {
    top1: f32,
    top3: f32,
    margin: f32,
    load_entropy: f32,
    max_load: usize,
    abstain_acc: f32,  // Accuracy when margin >= threshold
    abstain_rate: f32, // % of queries that abstain
}

fn calculate_entropy(counts: &[usize]) -> f32 {
    let total: usize = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let mut entropy = 0.0;
    for &c in counts {
        if c > 0 {
            let p = c as f32 / total as f32;
            entropy -= p * p.log2();
        }
    }
    entropy
}

fn run_v05_trial(
    seed: u64,
    config: CantorHdcConfig,
    n_objects: usize,
    router: &dyn CantorRouter,
    abstain_threshold: f32,
) -> V05Result {
    let leaf_dim = config.leaf_dim;
    let mut pyramid = PyramidCantorVector::new(config, None);
    let codebook_size = 500;
    let codebook: Vec<ContinuousHV> = (0..codebook_size)
        .map(|i| ContinuousHV::random(leaf_dim, seed + 1000 + i as u64))
        .collect();

    let mut stored = Vec::new();
    let mut leaf_counts = vec![0; config.branching];

    for i in 0..n_objects {
        let role = ContinuousHV::random(leaf_dim, seed + 2000 + i as u64);
        let val_idx = i % codebook_size;
        let binding = role.bind(&codebook[val_idx]);

        let leaf_idx =
            router.route_and_record(&role, &ContinuousHV::zero(leaf_dim), config.branching);
        leaf_counts[leaf_idx] += 1;

        let leaf = pyramid.find_node(1, leaf_idx).unwrap().clone();
        pyramid.bundle_at_node(&leaf, &binding);
        stored.push((role, val_idx, leaf_idx));
    }

    let mut hits1 = 0;
    let mut hits3 = 0;
    let mut total_margin = 0.0;
    let mut abstain_hits = 0;
    let mut abstain_count = 0;
    let mut non_abstain_total = 0;

    for (role, correct_idx, leaf_idx) in &stored {
        let leaf = pyramid.find_node(1, *leaf_idx).unwrap();
        let recovered = ContinuousHV::from_slice(pyramid.node_data(leaf)).bind(&role.inverse());

        let mut sims: Vec<(usize, f32)> = codebook
            .iter()
            .enumerate()
            .map(|(idx, cand)| (idx, recovered.similarity(cand)))
            .collect();

        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let margin = sims[0].1 - sims[1].1;
        total_margin += margin;

        if sims[0].0 == *correct_idx {
            hits1 += 1;
        }
        if sims.iter().take(3).any(|(idx, _)| idx == correct_idx) {
            hits3 += 1;
        }

        if margin < abstain_threshold {
            abstain_count += 1;
        } else {
            non_abstain_total += 1;
            if sims[0].0 == *correct_idx {
                abstain_hits += 1;
            }
        }
    }

    V05Result {
        top1: hits1 as f32 / n_objects as f32,
        top3: hits3 as f32 / n_objects as f32,
        margin: total_margin / n_objects as f32,
        load_entropy: calculate_entropy(&leaf_counts),
        max_load: *leaf_counts.iter().max().unwrap_or(&0),
        abstain_acc: if non_abstain_total > 0 {
            abstain_hits as f32 / non_abstain_total as f32
        } else {
            0.0
        },
        abstain_rate: abstain_count as f32 / n_objects as f32,
    }
}

#[test]
#[ignore = "full HCH router ablation benchmark; run explicitly with --ignored --nocapture"]
fn test_hch_v05_comprehensive_router_comparison() {
    let seed = 42;
    let n_objects = 256;
    let config = CantorHdcConfig {
        total_dim: 16384,
        levels: 2,
        branching: 16,
        leaf_dim: 1024,
        bundle_mode: BundleMode::UnitNormalize,
    };
    let abstain_tau = 0.05;

    println!(
        "\n--- HCH v0.5 Router Comparison (N=256, τ={:.2}) ---",
        abstain_tau
    );
    println!("Router         | Top-1 | Top-3 | Margin | Entrp | MaxL | AbsAcc | AbsRate");
    println!("---------------|-------|-------|--------|-------|------|--------|--------");

    let routers: Vec<(&str, Box<dyn CantorRouter>)> = vec![
        ("Random", Box::new(RandomRouter { seed })),
        ("Hash", Box::new(HashRouter)),
        (
            "Hypercube",
            Box::new(HypercubeRouter {
                dimensions: 4,
                seed,
            }),
        ),
        ("LB-Hash-2", Box::new(LoadBalancedHashRouter::new(16, 2))),
        ("LB-Hash-4", Box::new(LoadBalancedHashRouter::new(16, 4))),
        ("PrefixMax", Box::new(PrefixMaxRouter)),
        (
            "Prototype",
            Box::new(PrototypeRouter {
                leaf_keys: (0..16)
                    .map(|i| ContinuousHV::random(1024, seed + 5000 + i as u64))
                    .collect(),
            }),
        ),
    ];

    for (name, router) in routers {
        let res = run_v05_trial(seed, config, n_objects, router.as_ref(), abstain_tau);
        println!(
            "{:14} | {:.1}% | {:.1}% | {:.4} | {:.2}  | {:4} | {:.1}%  | {:.1}%",
            name,
            res.top1 * 100.0,
            res.top3 * 100.0,
            res.margin,
            res.load_entropy,
            res.max_load,
            res.abstain_acc * 100.0,
            res.abstain_rate * 100.0
        );
    }

    // Oracle-ish bound.
    // We simulate Oracle by just giving each object its own leaf (1 object per leaf stress)
    // or routing perfectly to an un-saturated leaf.
    // For this test, we'll just run one trial with a very large number of leaves to see the limit.
    let oracle_config = CantorHdcConfig {
        total_dim: 16384 * 4,
        levels: 2,
        branching: 64,
        leaf_dim: 1024,
        bundle_mode: BundleMode::UnitNormalize,
    };
    let res_oracle = run_v05_trial(seed, oracle_config, n_objects, &HashRouter, abstain_tau);
    println!(
        "{:14} | {:.1}% | {:.1}% | {:.4} | {:.2}  | {:4} | {:.1}%  | {:.1}%",
        "Oracle-Approx",
        res_oracle.top1 * 100.0,
        res_oracle.top3 * 100.0,
        res_oracle.margin,
        res_oracle.load_entropy,
        res_oracle.max_load,
        res_oracle.abstain_acc * 100.0,
        res_oracle.abstain_rate * 100.0
    );
}

#[test]
fn test_hch_v05_hash_vs_hypercube_smoke() {
    let seed = 7;
    let n_objects = 16;
    let config = CantorHdcConfig {
        total_dim: 4096,
        levels: 2,
        branching: 8,
        leaf_dim: 512,
        bundle_mode: BundleMode::UnitNormalize,
    };
    let abstain_tau = 0.05;

    let hash = run_v05_trial(seed, config, n_objects, &HashRouter, abstain_tau);
    let hypercube = run_v05_trial(
        seed,
        config,
        n_objects,
        &HypercubeRouter {
            dimensions: 3,
            seed,
        },
        abstain_tau,
    );

    assert!(hash.top3 >= hash.top1);
    assert!(hypercube.top3 >= hypercube.top1);
    assert!(hash.load_entropy > 0.0);
    assert!(hypercube.load_entropy > 0.0);
    assert!(hash.max_load <= n_objects);
    assert!(hypercube.max_load <= n_objects);
}

#[test]
fn test_hch_v05_hypercube_coordinate_geometry() {
    let router = HypercubeRouter {
        dimensions: 4,
        seed: 17,
    };
    let role = ContinuousHV::random(1024, 1234);
    let context = ContinuousHV::random(1024, 5678);

    let idx = router.route(&role, &context, 16);
    let neighbors = HypercubeRouter::hamming_neighbors(idx, 4);

    assert!(idx < 16);
    assert_eq!(router.node_count(), 16);
    assert_eq!(neighbors.len(), 4);
    assert!(neighbors.iter().all(|neighbor| *neighbor < 16));
    assert!(neighbors.iter().all(|neighbor| {
        let diff = idx ^ *neighbor;
        diff.count_ones() == 1
    }));
}

#[test]
#[ignore = "full HCH abstention sweep; run explicitly with --ignored --nocapture"]
fn test_hch_v05_abstention_curve() {
    let seed = 99;
    let n_objects = 128;
    let config = CantorHdcConfig {
        total_dim: 16384,
        levels: 2,
        branching: 16,
        leaf_dim: 1024,
        bundle_mode: BundleMode::UnitNormalize,
    };
    let router = LoadBalancedHashRouter::new(16, 2);

    println!("\n--- HCH v0.5 Abstention Calibration (N=128) ---");
    println!("Threshold τ | Accuracy | Abstain Rate");
    println!("------------|----------|-------------");

    for &tau in &[0.0, 0.02, 0.05, 0.1, 0.15] {
        let res = run_v05_trial(seed, config, n_objects, &router, tau);
        println!(
            "{:.2}        | {:.1}%    | {:.1}%",
            tau,
            res.abstain_acc * 100.0,
            res.abstain_rate * 100.0
        );
    }
}
