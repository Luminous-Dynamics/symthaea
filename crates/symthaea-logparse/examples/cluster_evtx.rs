//! Phase 1 benchmark runner.
//!
//! Two modes:
//!
//!   # Synthetic mode (works today, pipeline sanity only):
//!   cargo run -p symthaea-logparse --example cluster_evtx -- --synthetic
//!
//!   # Real corpus mode (the actual Phase 1 kill criterion — requires a
//!   # labeled .evtx corpus to be staged):
//!   cargo run -p symthaea-logparse --example cluster_evtx -- /path/to/corpus/
//!
//! Expected corpus layout:
//!   /path/to/corpus/
//!     labels.csv      # filename,label  (one row per .evtx file)
//!     <file1>.evtx
//!     <file2>.evtx
//!     ...
//!
//! Synthetic mode does NOT validate the thesis — it just runs the full
//! pipeline against fixtures designed to be separable. Use it to confirm the
//! encoder + HDBSCAN wire together correctly before pointing real data at it.

use std::collections::HashMap;
use std::path::PathBuf;
use symthaea_logparse::cluster::{hdbscan_cluster, nearest_centroid, purity};
use symthaea_logparse::encoder::{Hdv, bundle, encode};
use symthaea_logparse::fixtures::generate_synthetic_corpus;
use symthaea_logparse::{LogEvent, evtx_source};

const KILL_CRITERION: f32 = 0.50;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arg = std::env::args()
        .nth(1)
        .ok_or("usage:\n  cluster_evtx --synthetic\n  cluster_evtx <corpus_dir>")?;

    let (events, mode_label) = if arg == "--synthetic" {
        let corpus = generate_synthetic_corpus(40, 0xC0FFEE);
        println!("[synthetic] generated {} events", corpus.len());
        (corpus, "synthetic (pipeline sanity only)")
    } else {
        let corpus_dir = PathBuf::from(&arg);
        let events = load_real_corpus(&corpus_dir)?;
        (events, "real corpus")
    };

    if events.is_empty() {
        eprintln!("no events to cluster");
        return Ok(());
    }

    let hvs: Vec<Hdv> = events.iter().map(encode).collect();
    let gt: Vec<String> = events
        .iter()
        .map(|e| e.label.clone().unwrap_or_else(|| "<unlabeled>".into()))
        .collect();
    let gt_refs: Vec<&str> = gt.iter().map(|s| s.as_str()).collect();

    // ------------------------------------------------------------------
    // SUPERVISED UPPER BOUND: nearest-centroid against known labels.
    // If this is low, the encoder itself cannot separate the classes and
    // no amount of clustering parameter tuning will save the thesis. If
    // this is high but HDBSCAN is low, the failure is parameter selection.
    // ------------------------------------------------------------------
    println!("[baseline] computing nearest-centroid upper bound...");
    let mut by_label: HashMap<&str, Vec<Hdv>> = HashMap::new();
    for (hv, gt) in hvs.iter().zip(gt_refs.iter()) {
        by_label.entry(*gt).or_default().push(hv.clone());
    }
    let mut ordered: Vec<&str> = by_label.keys().copied().collect();
    ordered.sort();
    let centroids: Vec<Hdv> = ordered.iter().map(|l| bundle(&by_label[*l])).collect();
    let nc_assignments = nearest_centroid(&hvs, &centroids);
    let nc_purity = purity(&nc_assignments, &gt_refs);
    println!("[baseline] nearest-centroid purity (supervised): {nc_purity:.3}");

    // ------------------------------------------------------------------
    // HDBSCAN SWEEP: try several min_cluster values to separate
    // "parameter failure" from "encoder failure".
    // ------------------------------------------------------------------
    let sweep: Vec<usize> = vec![10, 20, 40, 80, (hvs.len() / 20).max(5)];

    let mut best_p = f32::NEG_INFINITY;
    let mut best_cfg: (usize, usize, usize) = (0, 0, 0); // (min_cluster, clusters, noise)

    println!("\n[sweep] HDBSCAN min_cluster ablation:");
    println!(
        "{:>12}  {:>10}  {:>10}  {:>10}",
        "min_cluster", "clusters", "noise", "purity"
    );
    for &mc in &sweep {
        let labels = hdbscan_cluster(&hvs, Some(mc))?;
        let noise = labels.iter().filter(|&&c| c == -1).count();
        let distinct: std::collections::HashSet<_> = labels.iter().filter(|&&c| c != -1).collect();
        let p = purity(&labels, &gt_refs);
        println!(
            "{:>12}  {:>10}  {:>10}  {:>10.3}",
            mc,
            distinct.len(),
            noise,
            p
        );
        if p > best_p && !p.is_nan() {
            best_p = p;
            best_cfg = (mc, distinct.len(), noise);
        }
    }

    println!("\n=== Phase 1 result ({mode_label}) ===");
    println!("events:              {}", hvs.len());
    println!("ground-truth:        {} classes", count_classes(&gt));
    println!("nc upper bound:      {nc_purity:.3} (supervised)");
    println!(
        "best hdbscan:        {:.3} at min_cluster={} ({} clusters, {} noise)",
        best_p, best_cfg.0, best_cfg.1, best_cfg.2
    );
    if arg == "--synthetic" {
        println!(
            "\nNOTE: synthetic mode does NOT validate the thesis. Real corpus \
             required for Phase 1 kill-criterion evaluation."
        );
    } else {
        let nc_verdict = if nc_purity >= KILL_CRITERION {
            "PASS"
        } else {
            "FAIL"
        };
        let hd_verdict = if best_p >= KILL_CRITERION {
            "PASS"
        } else {
            "FAIL"
        };
        let chance = 1.0 / count_classes(&gt) as f32;
        println!("\nchance level (1/n_classes): {chance:.3}");
        println!("kill criterion (>= {KILL_CRITERION:.2}):");
        println!("  nearest-centroid (upper bound): {nc_verdict}");
        println!("  HDBSCAN (unsupervised):         {hd_verdict}");

        if nc_purity >= KILL_CRITERION && best_p < KILL_CRITERION {
            println!(
                "\nInterpretation: encoder CAN separate the classes (nc >= {KILL_CRITERION:.2}), \
                 but HDBSCAN cannot recover them unsupervised. Failure mode is \
                 density-overlap in hypervector space, not encoder inadequacy. \
                 Phase 2 should explore supervised probing (logistic regression \
                 on bipolar HVs) before adding more encoding channels."
            );
        } else if nc_purity < KILL_CRITERION {
            println!(
                "\nInterpretation: even the supervised upper bound is below \
                 the kill criterion. The encoder genuinely cannot separate \
                 these MITRE tactics. Phase 2 must add encoding channels OR \
                 the thesis needs reframing (e.g. anomaly detection against \
                 a learned baseline rather than multi-class clustering)."
            );
        }
    }

    // Use the best hdbscan labels for the per-class breakdown.
    let labels = hdbscan_cluster(&hvs, Some(best_cfg.0))?;

    // Per-class recall breakdown — useful for diagnosing which event families
    // the encoder struggles with.
    println!("\n=== Per-class breakdown ===");
    let mut class_stats: HashMap<&str, (usize, usize)> = HashMap::new(); // (total, assigned)
    for (lbl, &c) in gt_refs.iter().zip(labels.iter()) {
        let entry = class_stats.entry(*lbl).or_insert((0, 0));
        entry.0 += 1;
        if c != -1 {
            entry.1 += 1;
        }
    }
    let mut class_vec: Vec<_> = class_stats.into_iter().collect();
    class_vec.sort_by_key(|(k, _)| *k);
    for (k, (total, assigned)) in class_vec {
        let pct = 100.0 * assigned as f32 / total as f32;
        println!("  {k:<20}  {assigned:>4}/{total:<4}  ({pct:.0}% assigned)");
    }

    Ok(())
}

fn load_real_corpus(corpus_dir: &PathBuf) -> Result<Vec<LogEvent>, Box<dyn std::error::Error>> {
    // Hard caps to keep HDBSCAN tractable. HDBSCAN is O(n² · d) and 16,384D
    // vectors mean n above ~3000 stops being interactive on a single host.
    //
    // MAX_PER_FILE prevents any single ETW-trace .evtx from dominating the
    // corpus — EVTX-ATTACK-SAMPLES has one 29K-event file that would
    // otherwise constitute 80% of the labeled sample.
    //
    // MAX_TOTAL is a stratified downsample by label AFTER per-file capping.
    const MAX_PER_FILE: usize = 50;
    const MAX_TOTAL: usize = 1000;

    let labels_path = corpus_dir.join("labels.csv");
    if !labels_path.exists() {
        return Err(format!(
            "No labels.csv in {}. See memory/project_msp_wedge.md.",
            corpus_dir.display()
        )
        .into());
    }
    let labels_csv = std::fs::read_to_string(&labels_path)?;
    let mut file_labels: HashMap<String, String> = HashMap::new();
    for line in labels_csv.lines().skip(1) {
        if let Some((f, l)) = line.split_once(',') {
            file_labels.insert(f.trim().to_string(), l.trim().to_string());
        }
    }

    let mut by_label: HashMap<String, Vec<LogEvent>> = HashMap::new();
    let mut total_parsed: usize = 0;
    let mut total_kept: usize = 0;

    for entry in std::fs::read_dir(corpus_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("evtx") {
            continue;
        }
        let fname = path.file_name().unwrap().to_string_lossy().to_string();
        let Some(label) = file_labels.get(&fname) else {
            eprintln!("skip {fname}: no label");
            continue;
        };
        let parsed = evtx_source::parse_evtx_file(&path)?;
        let parsed_len = parsed.len();
        total_parsed += parsed_len;

        // Per-file cap: deterministic stride subsampling so we don't lose
        // class signal from high-volume ETW traces entirely.
        let capped: Vec<LogEvent> = if parsed_len <= MAX_PER_FILE {
            parsed
        } else {
            let stride = parsed_len / MAX_PER_FILE;
            parsed
                .into_iter()
                .step_by(stride.max(1))
                .take(MAX_PER_FILE)
                .collect()
        };

        let kept = capped.len();
        total_kept += kept;
        if parsed_len != kept {
            println!("{fname}: {parsed_len} events -> {kept} (capped), label={label}");
        } else {
            println!("{fname}: {kept} events, label={label}");
        }

        let bucket = by_label.entry(label.clone()).or_default();
        for mut ev in capped {
            ev.label = Some(label.clone());
            bucket.push(ev);
        }
    }

    println!("\n[parse] total_parsed={total_parsed}, kept_after_per_file_cap={total_kept}");

    // Stratified downsample across labels to MAX_TOTAL. Deterministic: we
    // stride-sample inside each label's bucket so repeated runs are
    // reproducible.
    let n_labels = by_label.len().max(1);
    let per_label_budget = MAX_TOTAL / n_labels;

    let mut events = Vec::new();
    for (label, bucket) in by_label.iter() {
        let take = bucket.len().min(per_label_budget);
        let stride = (bucket.len() / take.max(1)).max(1);
        let sampled: Vec<LogEvent> = bucket.iter().step_by(stride).take(take).cloned().collect();
        println!(
            "[sample] {label}: {} -> {} events",
            bucket.len(),
            sampled.len()
        );
        events.extend(sampled);
    }
    println!(
        "[sample] final corpus: {} events across {} labels",
        events.len(),
        n_labels
    );
    Ok(events)
}

fn count_classes(gt: &[String]) -> usize {
    let set: std::collections::HashSet<&str> = gt.iter().map(|s| s.as_str()).collect();
    set.len()
}
