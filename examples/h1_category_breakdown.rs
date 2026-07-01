// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! H1 Category Breakdown Analysis
//!
//! Analyzes which phenomenal/computational categories show the strongest
//! topological differences to understand what drives the H1 effect.

#[cfg(feature = "neural-bridge")]
use std::path::Path;
#[cfg(feature = "neural-bridge")]
use std::time::Instant;

use anyhow::Result;

#[cfg(feature = "neural-bridge")]
use symthaea::perception::ConsciousnessProbeV2;

#[cfg(feature = "neural-bridge")]
use symthaea_core::hdc::consciousness_topology::{ConsciousnessTopology, TopologyConfig};

fn main() -> Result<()> {
    #[cfg(not(feature = "neural-bridge"))]
    {
        println!("This example requires the 'neural-bridge' feature.");
        println!(
            "Run with: cargo run --example h1_category_breakdown --features neural-bridge --release"
        );
        Ok(())
    }

    #[cfg(feature = "neural-bridge")]
    run_category_analysis()
}

#[cfg(feature = "neural-bridge")]
fn run_category_analysis() -> Result<()> {
    println!("\n");
    println!("================================================================");
    println!("   H1 CATEGORY BREAKDOWN ANALYSIS");
    println!("   Which categories drive the phenomenal-computational difference?");
    println!("================================================================\n");

    let probe_path = Path::new("models/neural_bridge/probe_weights_bge_m3.npy");
    if !probe_path.exists() {
        println!("ERROR: Probe weights not found");
        return Ok(());
    }

    println!("Loading BGE-M3 model...");
    let load_start = Instant::now();
    let mut probe = ConsciousnessProbeV2::load_with_probe(probe_path)?;
    println!("  Loaded in {:.2}s\n", load_start.elapsed().as_secs_f64());

    // Phenomenal concepts by category
    let qualia_by_category: Vec<(&str, Vec<&str>)> = vec![
        (
            "visual",
            vec![
                "The subjective experience of seeing red",
                "The experience of deep blue in the evening sky",
                "The bright flash of yellow in a sunflower",
                "The vivid green of fresh spring leaves",
                "The pure white of fresh snow",
                "The deep purple of ripe grapes",
                "The orange glow of a candle flame",
                "The soft pink of cherry blossoms",
            ],
        ),
        (
            "auditory",
            vec![
                "The felt quality of hearing a musical note",
                "The vibration I feel when humming a tune",
                "The deep bass rumble I feel in my chest",
                "The high-pitched ring of a bell",
                "The crackling sound of fire",
                "The gentle whisper of wind through trees",
                "The rhythmic patter of rain on a roof",
                "The low hum of distant traffic",
            ],
        ),
        (
            "tactile",
            vec![
                "The raw sensation of pressure on my skin",
                "The soft texture of velvet under my fingers",
                "The prickly sensation of touching a cactus",
                "The rough grain of sandpaper under my hand",
                "The smooth coolness of polished marble",
                "The fuzzy softness of a peach skin",
                "The silky flow of water through my fingers",
            ],
        ),
        (
            "gustatory",
            vec![
                "The taste of sweetness on my tongue",
                "The bitter taste that lingers after coffee",
                "The sour pucker from biting a lemon",
                "The salty taste of ocean spray",
                "The creamy texture of chocolate melting",
                "The tangy zip of citrus on my tongue",
                "The umami richness of aged cheese",
                "The spicy burn of hot peppers",
                "The metallic taste of blood",
            ],
        ),
        (
            "olfactory",
            vec![
                "The smell of roses filling my awareness",
                "The rich aroma of fresh bread baking",
                "The crisp scent of pine needles in the forest",
                "The earthy smell of rain on dry soil",
                "The floral scent of lavender",
                "The sharp smell of vinegar",
                "The woody scent of cedar",
                "The sweet fragrance of honeysuckle",
            ],
        ),
        (
            "thermal",
            vec![
                "The feeling of warmth spreading through my body",
                "The cool sensation of water on my face",
                "The sensation of cold metal against my palm",
                "The burning heat of touching something hot",
                "The icy chill of a winter breeze",
            ],
        ),
        (
            "pain",
            vec![
                "What it is like to feel pain",
                "The sharp sting of a paper cut",
                "The dull ache of a sore muscle",
                "The throbbing pulse of a headache",
                "The sharp pinch of a needle",
            ],
        ),
    ];

    // Computational concepts by category
    let computation_by_category: Vec<(&str, Vec<&str>)> = vec![
        (
            "algorithms",
            vec![
                "Binary search tree traversal algorithms",
                "Graph traversal using depth-first search",
                "Breadth-first search queue exploration",
                "Dijkstra shortest path computation",
                "Floyd-Warshall all-pairs shortest paths",
                "Topological sort dependency ordering",
                "Bellman-Ford negative edge relaxation",
                "Kruskal minimum spanning tree edges",
                "Prim minimum spanning tree growth",
                "A-star heuristic path finding",
                "Tarjan strongly connected components",
            ],
        ),
        (
            "data_structures",
            vec![
                "Hash table collision resolution strategies",
                "Linked list node insertion and deletion",
                "Stack push and pop operations",
                "Heap data structure heapify operation",
                "Red-black tree rotation balancing",
                "Trie prefix tree string matching",
                "Binary heap priority queue operations",
                "Queue enqueue and dequeue operations",
                "AVL tree height balancing rotation",
                "Bloom filter probabilistic membership",
                "Skip list logarithmic search structure",
                "Union-find disjoint set operations",
                "B-tree disk-optimized node splitting",
                "Segment tree range query aggregation",
                "Fenwick tree prefix sum updates",
                "Circular buffer wraparound indexing",
                "LRU cache eviction policy",
                "Deque double-ended insertion",
                "Suffix array substring indexing",
                "Treap randomized priority balancing",
                "Splay tree self-adjusting access",
            ],
        ),
        (
            "sorting",
            vec![
                "Quicksort partition and pivot selection",
                "Merge sort divide and conquer strategy",
                "Insertion sort element placement",
                "Radix sort digit-by-digit ordering",
                "Counting sort frequency distribution",
                "Selection sort minimum finding",
                "Heapsort extract-max iteration",
                "Shell sort gap sequence comparison",
                "Bubble sort adjacent element swapping",
            ],
        ),
        (
            "memory",
            vec![
                "Memory allocation and deallocation in systems",
                "Garbage collection memory management",
                "Array index bounds checking",
            ],
        ),
        (
            "optimization",
            vec![
                "Dynamic programming optimization techniques",
                "Memoization cache lookup optimization",
            ],
        ),
        (
            "other_comp",
            vec![
                "Recursive function evaluation in programming",
                "Type inference in static analysis",
                "Compiler lexical analysis and tokenization",
                "Network packet routing algorithms",
            ],
        ),
    ];

    let topo_config = TopologyConfig {
        min_persistence: 0.05,
        max_scale: 1.0,
        num_scales: 20,
        detect_cycles: true,
        detect_voids: true,
    };

    let analyze_concept = |probe: &mut ConsciousnessProbeV2, text: &str| -> Result<f64> {
        let hv = probe.concept_to_hv(text)?;
        let mut topology = ConsciousnessTopology::new(topo_config.clone());
        topology.add_state(hv);
        for shift in 1..5 {
            let permuted = hv.permute(shift * 100);
            topology.add_state(permuted);
        }
        let assessment = topology.analyze(0.5);
        Ok(assessment.unity_score)
    };

    // Analyze phenomenal categories
    println!("================================================================");
    println!("   PHENOMENAL CATEGORIES");
    println!("================================================================\n");

    let mut qualia_category_means: Vec<(&str, f64, f64, usize)> = Vec::new();

    for (category, concepts) in &qualia_by_category {
        let mut scores = Vec::new();
        for concept in concepts {
            match analyze_concept(&mut probe, concept) {
                Ok(score) => scores.push(score),
                Err(e) => println!("  Error processing '{}': {}", concept, e),
            }
        }

        if !scores.is_empty() {
            let mean = scores.iter().sum::<f64>() / scores.len() as f64;
            let std = (scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / scores.len() as f64)
                .sqrt();
            qualia_category_means.push((category, mean, std, scores.len()));
            println!(
                "{:12} (n={:2}): Mean Unity = {:.4} (+/- {:.4})",
                category,
                scores.len(),
                mean,
                std
            );
        }
    }

    // Analyze computational categories
    println!("\n================================================================");
    println!("   COMPUTATIONAL CATEGORIES");
    println!("================================================================\n");

    let mut comp_category_means: Vec<(&str, f64, f64, usize)> = Vec::new();

    for (category, concepts) in &computation_by_category {
        let mut scores = Vec::new();
        for concept in concepts {
            match analyze_concept(&mut probe, concept) {
                Ok(score) => scores.push(score),
                Err(e) => println!("  Error processing '{}': {}", concept, e),
            }
        }

        if !scores.is_empty() {
            let mean = scores.iter().sum::<f64>() / scores.len() as f64;
            let std = (scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / scores.len() as f64)
                .sqrt();
            comp_category_means.push((category, mean, std, scores.len()));
            println!(
                "{:15} (n={:2}): Mean Unity = {:.4} (+/- {:.4})",
                category,
                scores.len(),
                mean,
                std
            );
        }
    }

    // Summary comparison
    println!("\n================================================================");
    println!("   CATEGORY RANKING (by mean unity)");
    println!("================================================================\n");

    let mut all_categories: Vec<(&str, &str, f64, usize)> = Vec::new();
    for (cat, mean, _, n) in &qualia_category_means {
        all_categories.push((cat, "phenomenal", *mean, *n));
    }
    for (cat, mean, _, n) in &comp_category_means {
        all_categories.push((cat, "computational", *mean, *n));
    }

    all_categories.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    println!(
        "{:4} {:15} {:12} {:>6} {:>6}",
        "Rank", "Category", "Type", "Unity", "n"
    );
    println!("{:-<50}", "");
    for (i, (cat, typ, mean, n)) in all_categories.iter().enumerate() {
        let marker = if *typ == "phenomenal" { "*" } else { " " };
        println!(
            "{:4} {:15} {:12} {:>6.4} {:>6}{}",
            i + 1,
            cat,
            typ,
            mean,
            n,
            marker
        );
    }
    println!("\n* = phenomenal category");

    // Overall statistics
    let qualia_overall_mean: f64 = qualia_category_means
        .iter()
        .map(|(_, m, _, n)| m * (*n as f64))
        .sum::<f64>()
        / qualia_category_means
            .iter()
            .map(|(_, _, _, n)| *n)
            .sum::<usize>() as f64;

    let comp_overall_mean: f64 = comp_category_means
        .iter()
        .map(|(_, m, _, n)| m * (*n as f64))
        .sum::<f64>()
        / comp_category_means
            .iter()
            .map(|(_, _, _, n)| *n)
            .sum::<usize>() as f64;

    println!("\n================================================================");
    println!("   OVERALL COMPARISON");
    println!("================================================================\n");

    println!("Phenomenal overall mean:    {:.4}", qualia_overall_mean);
    println!("Computational overall mean: {:.4}", comp_overall_mean);
    println!(
        "Difference:                 {:.4}",
        qualia_overall_mean - comp_overall_mean
    );

    // Identify strongest drivers
    println!("\n================================================================");
    println!("   KEY INSIGHTS");
    println!("================================================================\n");

    // Find highest and lowest phenomenal categories
    let highest_qualia = qualia_category_means
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let lowest_qualia = qualia_category_means
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let highest_comp = comp_category_means
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let lowest_comp = comp_category_means
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    if let (Some(hq), Some(lq)) = (highest_qualia, lowest_qualia) {
        println!(
            "Phenomenal range: {} ({:.4}) to {} ({:.4})",
            hq.0, hq.1, lq.0, lq.1
        );
    }

    if let (Some(hc), Some(lc)) = (highest_comp, lowest_comp) {
        println!(
            "Computational range: {} ({:.4}) to {} ({:.4})",
            hc.0, hc.1, lc.0, lc.1
        );
    }

    println!("\n================================================================");
    println!("   CATEGORY BREAKDOWN COMPLETE");
    println!("================================================================\n");

    Ok(())
}