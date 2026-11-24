//! Supervised probe — Phase 2 priority 2.
//!
//! Trains a one-vs-rest logistic regression over the 16,384D bipolar
//! hypervectors produced by the encoder. Real-corpus evaluation uses an
//! 80/20 split grouped by source EVTX file, not by individual event, so events
//! from one file cannot leak across train and test.
//!
//! Purpose: measure the encoder's **supervised ceiling**. Nearest-centroid
//! gave 0.451 purity (below the 0.50 kill criterion), but NC averages
//! every dimension equally — a learned classifier can weight discriminative
//! dimensions up and noise dimensions down. If the probe gives a
//! significantly higher number than NC, the encoder has signal the NC
//! baseline couldn't exploit. If the probe plateaus near 0.50, the signal
//! itself is the limit and richer encoding channels become mandatory.
//!
//! Usage:
//!   cargo run -p symthaea-logparse --example supervised_probe --release -- /tmp/evtx-staged
//!   cargo run -p symthaea-logparse --example supervised_probe --release -- --synthetic

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use symthaea_logparse::encoder::{Hdv, encode};
use symthaea_logparse::fixtures::generate_synthetic_corpus;
use symthaea_logparse::probe::{
    LogisticProbe, TrainConfig, group_kfold_splits, group_stratified_split,
};
use symthaea_logparse::{LogEvent, evtx_source};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arg = std::env::args()
        .nth(1)
        .ok_or("usage:\n  supervised_probe --synthetic\n  supervised_probe <corpus_dir>")?;

    let (events, group_ids, mode_label) = if arg == "--synthetic" {
        let corpus = generate_synthetic_corpus(100, 0xC0FFEE);
        let groups = (0..corpus.len())
            .map(|i| format!("synthetic-event-{i:06}"))
            .collect();
        (corpus, groups, "synthetic (one event per group)")
    } else {
        let corpus_dir = PathBuf::from(&arg);
        let (events, groups) = load_real_corpus(&corpus_dir)?;
        (events, groups, "real EVTX-ATTACK-SAMPLES (file-disjoint)")
    };

    println!("[probe] mode: {mode_label}");
    println!("[probe] encoding {} events...", events.len());
    let hvs: Vec<Hdv> = events.iter().map(encode).collect();
    let labels: Vec<String> = events
        .iter()
        .map(|e| e.label.clone().unwrap_or_else(|| "<unlabeled>".into()))
        .collect();

    // Show per-class distribution
    let mut dist: HashMap<&str, usize> = HashMap::new();
    for l in labels.iter() {
        *dist.entry(l.as_str()).or_insert(0) += 1;
    }
    let mut dist_vec: Vec<_> = dist.iter().collect();
    dist_vec.sort_by_key(|(k, _)| *k);
    println!("[probe] per-class distribution:");
    for (k, n) in &dist_vec {
        println!("  {k:<22} {n}");
    }

    // 80/20 stratified split by provenance group. For real corpora the
    // group is the source EVTX file name; synthetic mode uses one event/group.
    let split = group_stratified_split(&hvs, &labels, &group_ids, 0.8, 0xFACADE)?;
    let train_groups: HashSet<&str> = split.train_group_ids.iter().map(String::as_str).collect();
    let test_groups: HashSet<&str> = split.test_group_ids.iter().map(String::as_str).collect();
    if !train_groups.is_disjoint(&test_groups) {
        return Err("provenance group leakage detected between train and test".into());
    }
    let train_x = split.train_hvs;
    let train_y = split.train_labels;
    let test_x = split.test_hvs;
    let test_y = split.test_labels;
    println!(
        "\n[probe] train={} events/{} groups, test={} events/{} groups, classes={}",
        train_x.len(),
        train_groups.len(),
        test_x.len(),
        test_groups.len(),
        dist.len()
    );

    let cfg = TrainConfig {
        epochs: 40,
        learning_rate: 0.05,
        l2: 1e-4,
        batch_size: 32,
        seed: 0xDEADBEEF,
    };
    println!(
        "[probe] config: epochs={} lr={} l2={} batch={}",
        cfg.epochs, cfg.learning_rate, cfg.l2, cfg.batch_size
    );

    println!("\n[probe] training...");
    let probe = LogisticProbe::train(&train_x, &train_y, cfg);

    let train_acc = probe.accuracy(&train_x, &train_y);
    let test_acc = probe.accuracy(&test_x, &test_y);

    println!("\n=== Supervised probe result ({mode_label}) ===");
    println!("train accuracy:    {train_acc:.3}");
    println!("test accuracy:     {test_acc:.3}");

    // Group-disjoint K-fold evaluation is the main uncertainty check. Each
    // source EVTX file appears in exactly one held-out fold, never in that
    // fold's training set. If the corpus has too few independent files per
    // label, fail explicitly rather than silently reverting to event splits.
    let grouped_folds = group_kfold_splits(&hvs, &labels, &group_ids, 5, 0x5EED)?;
    let mut grouped_fold_accuracies = Vec::with_capacity(grouped_folds.len());
    println!("\n=== Source-file-disjoint 5-fold CV ===");
    for (fold_index, fold) in grouped_folds.iter().enumerate() {
        let train_groups: HashSet<&str> =
            fold.train_group_ids.iter().map(String::as_str).collect();
        let test_groups: HashSet<&str> =
            fold.test_group_ids.iter().map(String::as_str).collect();
        if !train_groups.is_disjoint(&test_groups) {
            return Err(format!("source-file leakage in grouped fold {}", fold_index + 1).into());
        }
        let fold_probe = LogisticProbe::train(&fold.train_hvs, &fold.train_labels, cfg);
        let fold_accuracy = fold_probe.accuracy(&fold.test_hvs, &fold.test_labels);
        println!(
            "  fold {}: train={} events/{} files, test={} events/{} files, accuracy={:.3}",
            fold_index + 1,
            fold.train_hvs.len(),
            train_groups.len(),
            fold.test_hvs.len(),
            test_groups.len(),
            fold_accuracy
        );
        grouped_fold_accuracies.push(fold_accuracy);
    }
    let grouped_mean = grouped_fold_accuracies.iter().sum::<f32>()
        / grouped_fold_accuracies.len() as f32;
    let grouped_variance = grouped_fold_accuracies
        .iter()
        .map(|value| (value - grouped_mean).powi(2))
        .sum::<f32>()
        / grouped_fold_accuracies.len() as f32;
    println!(
        "  file-disjoint mean ± population SD: {:.3} ± {:.3}",
        grouped_mean,
        grouped_variance.sqrt()
    );

    println!("chance level:      {:.3}", 1.0 / dist.len() as f32);
    println!("historical NC event-level reference: 0.451 (not directly comparable)");
    println!("historical HDBSCAN event-level reference: 0.565 (not directly comparable)");
    println!("Phase 1 cutoff:    0.500");

    if test_acc >= 0.70 {
        println!(
            "\nInterpretation: the encoder has substantial supervised signal \
             that nearest-centroid could not exploit. A classifier path is \
             viable — Phase 2 priority 3 (richer encoding) is a nice-to-have, \
             not a blocker."
        );
    } else if test_acc >= 0.55 {
        println!(
            "\nInterpretation: the encoder has modest supervised signal — the \
             learned classifier beats NC but doesn't clear 0.70. Richer \
             encoding channels would likely push it higher."
        );
    } else if test_acc >= 0.45 {
        println!(
            "\nInterpretation: the supervised probe is at parity with NC (~0.45). \
             A linear classifier cannot extract class signal the NC baseline missed. \
             The encoder's information bottleneck is upstream. Phase 2 priority 3 \
             (richer encoding channels) is now MANDATORY, not optional."
        );
    } else {
        println!(
            "\nInterpretation: the supervised probe is BELOW nearest-centroid, \
             which should not happen. Investigate training configuration."
        );
    }

    // Per-class accuracy breakdown on test set
    println!("\n=== Per-class test accuracy ===");
    let test_preds = probe.predict_batch(&test_x);
    let class_names = &probe.class_names;
    let name_idx: HashMap<&str, usize> = class_names
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i))
        .collect();

    let mut per_class: HashMap<&str, (usize, usize)> = HashMap::new(); // (correct, total)
    for (lbl, &pred) in test_y.iter().zip(test_preds.iter()) {
        let entry = per_class.entry(lbl.as_str()).or_insert((0, 0));
        entry.1 += 1;
        if name_idx.get(lbl.as_str()) == Some(&pred) {
            entry.0 += 1;
        }
    }
    let mut per_class_vec: Vec<_> = per_class.into_iter().collect();
    per_class_vec.sort_by_key(|(k, _)| *k);
    for (k, (c, t)) in per_class_vec {
        let a = c as f32 / t as f32;
        println!("  {k:<22} {c:>3}/{t:<3}  ({a:.2})");
    }

    Ok(())
}

fn load_real_corpus(
    corpus_dir: &PathBuf,
) -> Result<(Vec<LogEvent>, Vec<String>), Box<dyn std::error::Error>> {
    const MAX_PER_FILE: usize = 50;
    const MAX_TOTAL: usize = 1000;

    let labels_path = corpus_dir.join("labels.csv");
    if !labels_path.exists() {
        return Err(format!("No labels.csv in {}", corpus_dir.display()).into());
    }
    let labels_csv = std::fs::read_to_string(&labels_path)?;
    let mut file_labels: HashMap<String, String> = HashMap::new();
    for line in labels_csv.lines().skip(1) {
        if let Some((f, l)) = line.split_once(',') {
            file_labels.insert(f.trim().to_string(), l.trim().to_string());
        }
    }

    let mut by_label: HashMap<String, Vec<(LogEvent, String)>> = HashMap::new();
    for entry in std::fs::read_dir(corpus_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("evtx") {
            continue;
        }
        let fname = path.file_name().unwrap().to_string_lossy().to_string();
        let Some(label) = file_labels.get(&fname) else {
            continue;
        };
        let parsed = evtx_source::parse_evtx_file(&path)?;
        let capped: Vec<LogEvent> = if parsed.len() <= MAX_PER_FILE {
            parsed
        } else {
            let stride = parsed.len() / MAX_PER_FILE;
            parsed
                .into_iter()
                .step_by(stride.max(1))
                .take(MAX_PER_FILE)
                .collect()
        };
        let bucket = by_label.entry(label.clone()).or_default();
        for mut ev in capped {
            ev.label = Some(label.clone());
            // Provenance only: this group ID is used by the split logic, not
            // encoded as a model feature.
            bucket.push((ev, fname.clone()));
        }
    }

    let n_labels = by_label.len().max(1);
    let per_label_budget = MAX_TOTAL / n_labels;

    let mut events = Vec::new();
    let mut group_ids = Vec::new();
    for (label, bucket) in by_label.iter() {
        let take = bucket.len().min(per_label_budget);
        let stride = (bucket.len() / take.max(1)).max(1);
        let sampled: Vec<(LogEvent, String)> =
            bucket.iter().step_by(stride).take(take).cloned().collect();
        for (event, group_id) in sampled {
            events.push(event);
            group_ids.push(group_id);
        }
        println!(
            "[sample] {label}: {} events from {} source files",
            events.iter().filter(|event| event.label.as_deref() == Some(label)).count(),
            bucket.iter().map(|(_, group)| group).collect::<HashSet<_>>().len()
        );
    }
    if events.len() != group_ids.len() {
        return Err("internal corpus provenance alignment failure".into());
    }
    Ok((events, group_ids))
}
