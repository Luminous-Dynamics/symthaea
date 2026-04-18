//! End-to-end pipeline test on synthetic fixtures.
//!
//! This is NOT the Phase 1 kill-criterion test — that requires a real labeled
//! Evtx corpus (see `memory/project_msp_wedge.md`). This is a pipeline sanity
//! check: does generate → encode → HDBSCAN → purity actually wire together,
//! and does it clear a much lower bar on synthetic data that is designed to
//! be separable?
//!
//! Failure here means the pipeline is broken. Success here means the pipeline
//! works — it does NOT mean the thesis is validated.

use symthaea_logparse::cluster::{hdbscan_cluster, nearest_centroid, purity};
use symthaea_logparse::encoder::{bundle, encode, Hdv};
use symthaea_logparse::fixtures::generate_synthetic_corpus;
use std::collections::HashMap;

/// Nearest-centroid baseline on synthetic fixtures.
///
/// Each class bundles its own events into a centroid, then every event is
/// assigned to its nearest centroid. On cleanly-separable synthetic data
/// this should hit near-perfect purity — if it doesn't, the encoder is
/// broken.
#[test]
fn synthetic_nearest_centroid_high_purity() {
    let corpus = generate_synthetic_corpus(20, 0xC0FFEE);
    let hvs: Vec<Hdv> = corpus.iter().map(encode).collect();
    let labels: Vec<String> = corpus
        .iter()
        .map(|e| e.label.clone().unwrap())
        .collect();
    let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

    // Build centroids in a stable order.
    let mut by_label: HashMap<&str, Vec<Hdv>> = HashMap::new();
    for (hv, gt) in hvs.iter().zip(label_refs.iter()) {
        by_label.entry(*gt).or_default().push(hv.clone());
    }
    let mut ordered: Vec<&str> = by_label.keys().copied().collect();
    ordered.sort();
    let centroids: Vec<Hdv> = ordered.iter().map(|l| bundle(&by_label[*l])).collect();

    let assignments = nearest_centroid(&hvs, &centroids);
    let p = purity(&assignments, &label_refs);
    assert!(
        p >= 0.90,
        "nearest-centroid purity on synthetic fixtures should be >= 0.90, got {p}"
    );
}

/// HDBSCAN on synthetic fixtures. Stricter test — this exercises the real
/// clusterer, not just a label-aware baseline.
///
/// On 5 cleanly-separable classes with 20 events each, HDBSCAN with a
/// reasonable min_cluster_size should recover something close to the ground
/// truth. We assert a lower bar (0.70) than nearest-centroid because HDBSCAN
/// can legitimately mark borderline points as noise, which lowers purity
/// even when the dense cores are correctly separated.
#[test]
fn synthetic_hdbscan_reasonable_purity() {
    let corpus = generate_synthetic_corpus(20, 0xBEEF);
    let hvs: Vec<Hdv> = corpus.iter().map(encode).collect();
    let labels: Vec<String> = corpus
        .iter()
        .map(|e| e.label.clone().unwrap())
        .collect();
    let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

    let assignments = hdbscan_cluster(&hvs, Some(5)).expect("hdbscan should run");
    let p = purity(&assignments, &label_refs);

    let noise = assignments.iter().filter(|&&c| c == -1).count();
    let n_clusters: std::collections::HashSet<_> =
        assignments.iter().filter(|&&c| c != -1).collect();

    eprintln!(
        "hdbscan synthetic: purity={p:.3}, clusters={}, noise={}/{}",
        n_clusters.len(),
        noise,
        assignments.len()
    );

    // Pipeline sanity: at least one cluster was found, and non-noise points
    // cluster with >= 0.70 purity. Remember: NOT the Phase 1 kill criterion.
    assert!(
        n_clusters.len() >= 2,
        "HDBSCAN should find >= 2 clusters on 5-class synthetic data"
    );
    if !p.is_nan() {
        assert!(
            p >= 0.70,
            "HDBSCAN purity on synthetic fixtures should be >= 0.70, got {p}"
        );
    }
}

/// Event count and label invariants.
#[test]
fn fixture_invariants() {
    let corpus = generate_synthetic_corpus(10, 1);
    assert_eq!(corpus.len(), 50);
    assert!(corpus.iter().all(|e| e.label.is_some()));
    let classes: std::collections::HashSet<_> =
        corpus.iter().map(|e| e.label.as_ref().unwrap()).collect();
    assert_eq!(classes.len(), 5);
}
