//! Ablation: encoder robustness under progressive class-boundary obfuscation.
//!
//! Sweeps noise_level from 0.0 → 1.0 in 0.1 steps, regenerates the corpus at
//! each step, runs HDBSCAN, and reports purity. The slope of the resulting
//! curve tells us how much of the encoder's 1.000 purity on clean fixtures
//! actually comes from multi-field composition vs. easy channel differences.
//!
//! Interpretation:
//!
//!   - Flat curve (purity stays high) → encoder composes all field channels,
//!     robust to missing/scrambled discriminators. Good sign for real data.
//!
//!   - Cliff at low noise (purity collapses fast) → encoder leans hard on
//!     provider+event_id. Real data will be messy and the thesis needs more
//!     encoding channels BEFORE we burn weeks on a real corpus run.
//!
//!   - Linear decay → graceful degradation. Expected behavior.
//!
//! Usage:
//!   cargo run -p symthaea-logparse --example ablation_noise --release

use std::collections::HashMap;
use symthaea_logparse::cluster::{hdbscan_cluster, nearest_centroid, purity};
use symthaea_logparse::encoder::{Hdv, bundle, encode};
use symthaea_logparse::fixtures::generate_noisy_corpus;

fn main() {
    const N_PER_CLASS: usize = 40;
    const SEED: u64 = 0xAB1A7105;
    const STEPS: usize = 11; // 0.0, 0.1, ..., 1.0

    println!("=== Encoder robustness ablation ===");
    println!(
        "n_per_class={N_PER_CLASS}, classes=5, total={} events per run",
        N_PER_CLASS * 5
    );
    println!(
        "{:>8}  {:>10}  {:>10}  {:>10}  {:>10}",
        "noise", "nc_purity", "hd_purity", "clusters", "noise_pts"
    );

    let mut results: Vec<(f32, f32, f32)> = Vec::new();

    for step in 0..STEPS {
        let noise = step as f32 / (STEPS - 1) as f32;
        let corpus = generate_noisy_corpus(N_PER_CLASS, SEED, noise);
        let hvs: Vec<Hdv> = corpus.iter().map(encode).collect();
        let labels: Vec<String> = corpus.iter().map(|e| e.label.clone().unwrap()).collect();
        let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

        // Nearest-centroid baseline: knows the labels, so this is an upper
        // bound on "how separable are the class distributions as the encoder
        // sees them". If THIS collapses, the encoder has lost signal.
        let mut by_label: HashMap<&str, Vec<Hdv>> = HashMap::new();
        for (hv, gt) in hvs.iter().zip(label_refs.iter()) {
            by_label.entry(*gt).or_default().push(hv.clone());
        }
        let mut ordered: Vec<&str> = by_label.keys().copied().collect();
        ordered.sort();
        let centroids: Vec<Hdv> = ordered.iter().map(|l| bundle(&by_label[*l])).collect();
        let nc_assign = nearest_centroid(&hvs, &centroids);
        let nc_p = purity(&nc_assign, &label_refs);

        // HDBSCAN: no label access, measures unsupervised separability.
        let hd_assign = hdbscan_cluster(&hvs, Some(5)).expect("hdbscan");
        let hd_p = purity(&hd_assign, &label_refs);
        let noise_pts = hd_assign.iter().filter(|&&c| c == -1).count();
        let distinct: std::collections::HashSet<_> =
            hd_assign.iter().filter(|&&c| c != -1).collect();

        let hd_display = if hd_p.is_nan() {
            "  NaN  ".to_string()
        } else {
            format!("{hd_p:>10.3}")
        };

        println!(
            "{:>8.2}  {:>10.3}  {}  {:>10}  {:>10}",
            noise,
            nc_p,
            hd_display,
            distinct.len(),
            noise_pts
        );

        results.push((noise, nc_p, if hd_p.is_nan() { 0.0 } else { hd_p }));
    }

    // Interpret the curve.
    println!("\n=== Interpretation ===");
    let (_, nc_clean, hd_clean) = results[0];
    let (_, nc_halfway, hd_halfway) = results[5];
    let (_, nc_max, hd_max) = results[STEPS - 1];

    println!("Clean (noise=0.0):    nc={nc_clean:.3}  hd={hd_clean:.3}");
    println!("Halfway (noise=0.5):  nc={nc_halfway:.3}  hd={hd_halfway:.3}");
    println!("Max noise (noise=1.0): nc={nc_max:.3}  hd={hd_max:.3}");
    println!("Chance (1/5 classes): nc=0.200  hd=0.200");

    let nc_halflife = results
        .iter()
        .position(|(_, p, _)| *p < nc_clean * 0.75)
        .map(|i| results[i].0);
    if let Some(hl) = nc_halflife {
        println!(
            "\nNearest-centroid 25% loss at noise={hl:.2} — channel leakage \
             this far and the encoder loses >=25% of its separation."
        );
    } else {
        println!(
            "\nNearest-centroid held >=75% purity across the entire sweep — encoder is robust."
        );
    }

    let hd_halflife = results
        .iter()
        .position(|(_, _, p)| *p < hd_clean * 0.75)
        .map(|i| results[i].0);
    if let Some(hl) = hd_halflife {
        println!(
            "HDBSCAN 25% loss at noise={hl:.2} — unsupervised density-\
             separation breaks here. This is the honest number for real data."
        );
    } else {
        println!(
            "HDBSCAN held >=75% purity across the entire sweep — unsupervised separability is robust."
        );
    }
}
