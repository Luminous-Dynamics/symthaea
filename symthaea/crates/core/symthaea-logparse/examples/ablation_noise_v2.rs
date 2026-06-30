//! Ablation v2: full-event contamination.
//!
//! Unlike `ablation_noise` (which only swaps provider/event_id/component +
//! 2 donor fields), v2 replaces the entire event body with a donor from a
//! random other class, keeping only the ground-truth label.
//!
//! Expected: purity decays monotonically toward chance (0.20 for 5 classes).
//! If it doesn't, something is wrong with the experiment or the encoder is
//! leaking signal through a channel we didn't model.

use std::collections::HashMap;
use symthaea_logparse::cluster::{hdbscan_cluster, nearest_centroid, purity};
use symthaea_logparse::encoder::{Hdv, bundle, encode};
use symthaea_logparse::fixtures::generate_noisy_corpus_v2;

fn main() {
    const N_PER_CLASS: usize = 40;
    const SEED: u64 = 0xAB1A7105;
    const STEPS: usize = 11;

    println!("=== Encoder robustness ablation v2 (full-event contamination) ===");
    println!(
        "{} events per run, 5 classes, chance=0.200",
        N_PER_CLASS * 5
    );
    println!(
        "{:>8}  {:>10}  {:>10}  {:>10}  {:>10}",
        "noise", "nc_purity", "hd_purity", "hd_clusters", "hd_noise"
    );

    let mut results: Vec<(f32, f32, f32)> = Vec::new();

    for step in 0..STEPS {
        let noise = step as f32 / (STEPS - 1) as f32;
        let corpus = generate_noisy_corpus_v2(N_PER_CLASS, SEED, noise);
        let hvs: Vec<Hdv> = corpus.iter().map(encode).collect();
        let labels: Vec<String> = corpus.iter().map(|e| e.label.clone().unwrap()).collect();
        let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

        let mut by_label: HashMap<&str, Vec<Hdv>> = HashMap::new();
        for (hv, gt) in hvs.iter().zip(label_refs.iter()) {
            by_label.entry(*gt).or_default().push(hv.clone());
        }
        let mut ordered: Vec<&str> = by_label.keys().copied().collect();
        ordered.sort();
        let centroids: Vec<Hdv> = ordered.iter().map(|l| bundle(&by_label[*l])).collect();
        let nc_assign = nearest_centroid(&hvs, &centroids);
        let nc_p = purity(&nc_assign, &label_refs);

        let hd_assign = hdbscan_cluster(&hvs, Some(5)).expect("hdbscan");
        let hd_p = purity(&hd_assign, &label_refs);
        let hd_noise = hd_assign.iter().filter(|&&c| c == -1).count();
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
            hd_noise
        );

        results.push((noise, nc_p, if hd_p.is_nan() { 0.0 } else { hd_p }));
    }

    println!("\n=== Interpretation ===");
    let (_, nc0, hd0) = results[0];
    let (_, nc10, hd10) = results[STEPS - 1];
    let nc_decay = nc0 - nc10;
    let hd_decay = hd0 - hd10;

    println!("Clean (noise=0.0):  nc={nc0:.3}  hd={hd0:.3}");
    println!("Scrambled (noise=1.0): nc={nc10:.3}  hd={hd10:.3}");
    println!("Chance level:       nc=0.200  hd=0.200");
    println!("Total decay:        nc={nc_decay:.3}  hd={hd_decay:.3}");

    // Sanity: at noise=1.0, purity should be near chance.
    if nc10 < 0.40 {
        println!("\nnc purity near chance at max noise — as expected.");
    } else {
        println!(
            "\nnc purity stayed at {nc10:.3} at max noise. Either the encoder \
             has a systematic class bias, or the donor selection is leaking \
             class information. INVESTIGATE."
        );
    }

    // Check monotonicity — any reversal indicates a noise-model artifact.
    let mut monotone = true;
    for w in results.windows(2) {
        if w[1].1 > w[0].1 + 0.10 {
            monotone = false;
            break;
        }
    }
    if monotone {
        println!("nc purity is (approximately) monotone decreasing — clean adversarial curve.");
    } else {
        println!("nc purity has a non-monotone segment — investigate the donor sampler.");
    }
}
