//! Phase 2 experiment runner.
//!
//! Runs the three experiments we owe ourselves after the Apr 14 probe
//! result of 0.815 on the single 80/20 split:
//!
//!   1. **Scale-up**: train on the full sbousseaden corpus (post per-file
//!      cap), not the 1000-event downsample. Does the test accuracy move?
//!
//!   2. **5-fold cross-validation**: stratified 5-fold CV on sbousseaden.
//!      Turn the single-split number into a mean ± std confidence interval.
//!
//!   3. **Cross-corpus OOD**: train on sbousseaden, test on OTRF
//!      Security-Datasets (Mordor-format JSONL). Measures whether the
//!      encoder learned "what credential access looks like" or "what
//!      sbousseaden's curation looks like."
//!
//! Usage:
//!   cargo run -p symthaea-logparse --example phase2_experiments --release -- \
//!     /tmp/evtx-staged /tmp/otrf-staged

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use symthaea_logparse::encoder::{encode, Hdv};
use symthaea_logparse::probe::{LogisticProbe, TrainConfig};
use symthaea_logparse::{evtx_source, otrf_source, LogEvent};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let evtx_dir = std::env::args().nth(1).ok_or("need evtx corpus dir")?;
    let otrf_dir = std::env::args().nth(2).ok_or("need otrf corpus dir")?;

    let shared_labels: Vec<&str> = vec![
        "credential_access",
        "defense_evasion",
        "discovery",
        "execution",
        "lateral_movement",
        "persistence",
        "privilege_escalation",
    ];
    // OTRF has no command_and_control samples, so cross-corpus tests run on 7 classes.

    // ----------------------------------------------------------------
    // Load sbousseaden (per-file cap 50, but no global total cap now)
    // ----------------------------------------------------------------
    println!("[load] sbousseaden corpus from {evtx_dir}");
    let sbou_events = load_evtx_corpus(&PathBuf::from(&evtx_dir), 50, None, &shared_labels)?;
    let sbou_labels: Vec<String> = sbou_events
        .iter()
        .map(|e| e.label.clone().unwrap())
        .collect();
    let sbou_hvs: Vec<Hdv> = sbou_events.iter().map(encode).collect();
    print_distribution("sbousseaden", &sbou_labels);

    // ----------------------------------------------------------------
    // Load OTRF (walk the staged dirs, parse every .json file)
    // ----------------------------------------------------------------
    println!("\n[load] OTRF corpus from {otrf_dir}");
    let otrf_events = load_otrf_corpus(&PathBuf::from(&otrf_dir), &shared_labels, 50)?;
    let otrf_labels: Vec<String> = otrf_events
        .iter()
        .map(|e| e.label.clone().unwrap())
        .collect();
    let otrf_hvs: Vec<Hdv> = otrf_events.iter().map(encode).collect();
    print_distribution("OTRF", &otrf_labels);

    let cfg = TrainConfig {
        epochs: 40,
        learning_rate: 0.05,
        l2: 1e-4,
        batch_size: 32,
        seed: 0xDEADBEEF,
    };

    // ----------------------------------------------------------------
    // Experiment 1: Scale-up single 80/20 split on full sbousseaden
    // ----------------------------------------------------------------
    println!("\n=== Experiment 1: scale-up (sbousseaden full corpus, 80/20 split) ===");
    let (tr_hv, tr_y, te_hv, te_y) =
        symthaea_logparse::probe::stratified_split(&sbou_hvs, &sbou_labels, 0.8, 0xFACADE);
    println!(
        "  train={}, test={}, classes={}",
        tr_hv.len(),
        te_hv.len(),
        shared_labels.len()
    );
    let probe1 = LogisticProbe::train(&tr_hv, &tr_y, cfg);
    let train_acc = probe1.accuracy(&tr_hv, &tr_y);
    let test_acc = probe1.accuracy(&te_hv, &te_y);
    println!("  train accuracy: {train_acc:.3}");
    println!("  test accuracy:  {test_acc:.3}");
    let scale_up_probe = probe1;

    // ----------------------------------------------------------------
    // Experiment 2: 5-fold stratified cross-validation on sbousseaden
    // ----------------------------------------------------------------
    println!("\n=== Experiment 2: 5-fold stratified CV on sbousseaden ===");
    let fold_accs = run_kfold(&sbou_hvs, &sbou_labels, 5, cfg);
    let mean = fold_accs.iter().sum::<f32>() / fold_accs.len() as f32;
    let var = fold_accs.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / fold_accs.len() as f32;
    let std = var.sqrt();
    println!("  per-fold test accuracy: {:?}", fold_accs);
    println!("  mean ± std: {mean:.3} ± {std:.3}");
    println!(
        "  min/max:    {:.3} / {:.3}",
        fold_accs.iter().cloned().fold(f32::INFINITY, f32::min),
        fold_accs.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    // ----------------------------------------------------------------
    // Experiment 3: Cross-corpus OOD — train on sbousseaden, test on OTRF
    // ----------------------------------------------------------------
    println!("\n=== Experiment 3: cross-corpus OOD ===");
    println!("  train corpus: sbousseaden ({} events)", sbou_hvs.len());
    println!("  test  corpus: OTRF        ({} events)", otrf_hvs.len());
    let ood_test_acc = scale_up_probe.accuracy(&otrf_hvs, &otrf_labels);
    println!("  OOD test accuracy: {ood_test_acc:.3}");
    println!("  (reference: same-distribution test was {test_acc:.3})");

    // Per-class OOD breakdown
    println!("\n=== Per-class OOD breakdown ===");
    let preds = scale_up_probe.predict_batch(&otrf_hvs);
    let class_names = &scale_up_probe.class_names;
    let name_idx: HashMap<&str, usize> = class_names
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i))
        .collect();
    let mut per_class: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for (lbl, &pred) in otrf_labels.iter().zip(preds.iter()) {
        let entry = per_class.entry(lbl.as_str()).or_insert((0, 0));
        entry.1 += 1;
        if name_idx.get(lbl.as_str()) == Some(&pred) {
            entry.0 += 1;
        }
    }
    for (k, (c, t)) in per_class {
        let a = if t > 0 { c as f32 / t as f32 } else { f32::NAN };
        println!("  {k:<22} {c:>4}/{t:<4}  ({a:.2})");
    }

    // ----------------------------------------------------------------
    // Summary
    // ----------------------------------------------------------------
    println!("\n=== Phase 2 summary ===");
    println!("  sbousseaden single split:    {test_acc:.3}");
    println!("  sbousseaden 5-fold CV:       {mean:.3} ± {std:.3}");
    println!("  OOD (train sbou, test otrf): {ood_test_acc:.3}");
    println!("  chance (7 classes):          {:.3}", 1.0 / 7.0);
    println!("  Phase 1 cutoff:              0.500");

    Ok(())
}

fn print_distribution(name: &str, labels: &[String]) {
    let mut dist: BTreeMap<&str, usize> = BTreeMap::new();
    for l in labels {
        *dist.entry(l.as_str()).or_insert(0) += 1;
    }
    println!(
        "[dist] {name}: {} events across {} classes",
        labels.len(),
        dist.len()
    );
    for (k, n) in dist {
        println!("  {k:<22} {n}");
    }
}

fn load_evtx_corpus(
    corpus_dir: &PathBuf,
    max_per_file: usize,
    max_total: Option<usize>,
    keep_labels: &[&str],
) -> Result<Vec<LogEvent>, Box<dyn std::error::Error>> {
    let labels_path = corpus_dir.join("labels.csv");
    let labels_csv = std::fs::read_to_string(&labels_path)?;
    let mut file_labels: HashMap<String, String> = HashMap::new();
    for line in labels_csv.lines().skip(1) {
        if let Some((f, l)) = line.split_once(',') {
            file_labels.insert(f.trim().to_string(), l.trim().to_string());
        }
    }

    let keep_set: std::collections::HashSet<&str> = keep_labels.iter().copied().collect();
    let mut by_label: BTreeMap<String, Vec<LogEvent>> = BTreeMap::new();

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
        if !keep_set.contains(label.as_str()) {
            continue;
        }
        let parsed = evtx_source::parse_evtx_file(&path)?;
        let capped: Vec<LogEvent> = if parsed.len() <= max_per_file {
            parsed
        } else {
            let stride = parsed.len() / max_per_file;
            parsed
                .into_iter()
                .step_by(stride.max(1))
                .take(max_per_file)
                .collect()
        };
        let bucket = by_label.entry(label.clone()).or_default();
        for mut ev in capped {
            ev.label = Some(label.clone());
            bucket.push(ev);
        }
    }

    let events = if let Some(cap) = max_total {
        let n_labels = by_label.len().max(1);
        let per_label_budget = cap / n_labels;
        let mut out = Vec::new();
        for bucket in by_label.values() {
            let take = bucket.len().min(per_label_budget);
            let stride = (bucket.len() / take.max(1)).max(1);
            out.extend(bucket.iter().step_by(stride).take(take).cloned());
        }
        out
    } else {
        by_label.into_values().flatten().collect()
    };
    Ok(events)
}

fn load_otrf_corpus(
    corpus_dir: &PathBuf,
    keep_labels: &[&str],
    max_per_dir: usize,
) -> Result<Vec<LogEvent>, Box<dyn std::error::Error>> {
    let labels_path = corpus_dir.join("labels.csv");
    let labels_csv = std::fs::read_to_string(&labels_path)?;
    let mut dir_labels: HashMap<String, String> = HashMap::new();
    for line in labels_csv.lines().skip(1) {
        if let Some((f, l)) = line.split_once(',') {
            dir_labels.insert(f.trim().to_string(), l.trim().to_string());
        }
    }
    let keep_set: std::collections::HashSet<&str> = keep_labels.iter().copied().collect();

    let mut out = Vec::new();
    for entry in std::fs::read_dir(corpus_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let dname = path.file_name().unwrap().to_string_lossy().to_string();
        let Some(label) = dir_labels.get(&dname) else {
            continue;
        };
        if !keep_set.contains(label.as_str()) {
            continue;
        }
        let parsed = otrf_source::parse_jsonl_tree(&path)?;
        let capped: Vec<LogEvent> = if parsed.len() <= max_per_dir {
            parsed
        } else {
            let stride = parsed.len() / max_per_dir;
            parsed
                .into_iter()
                .step_by(stride.max(1))
                .take(max_per_dir)
                .collect()
        };
        for mut ev in capped {
            ev.label = Some(label.clone());
            out.push(ev);
        }
    }
    Ok(out)
}

/// Stratified 5-fold cross-validation. Splits each class's samples into k
/// contiguous chunks after a deterministic shuffle, then trains k models.
fn run_kfold(hvs: &[Hdv], labels: &[String], k: usize, cfg: TrainConfig) -> Vec<f32> {
    use symthaea_logparse::probe::LogisticProbe;

    // Group indices by label
    let mut by_label: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, l) in labels.iter().enumerate() {
        by_label.entry(l.clone()).or_default().push(i);
    }

    // Deterministic shuffle per label
    let mut state: u64 = 0xDEADBEEF;
    fn xorshift(s: &mut u64) -> u64 {
        let mut x = *s;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *s = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    for (_, idxs) in by_label.iter_mut() {
        for i in (1..idxs.len()).rev() {
            let j = (xorshift(&mut state) as usize) % (i + 1);
            idxs.swap(i, j);
        }
    }

    // Assign each index to a fold via round-robin within each label
    let n = hvs.len();
    let mut fold_of = vec![0usize; n];
    for (_, idxs) in by_label.iter() {
        for (position, &idx) in idxs.iter().enumerate() {
            fold_of[idx] = position % k;
        }
    }

    let mut accs = Vec::with_capacity(k);
    for fold in 0..k {
        let mut train_hvs = Vec::new();
        let mut train_y = Vec::new();
        let mut test_hvs = Vec::new();
        let mut test_y = Vec::new();
        for i in 0..n {
            if fold_of[i] == fold {
                test_hvs.push(hvs[i].clone());
                test_y.push(labels[i].clone());
            } else {
                train_hvs.push(hvs[i].clone());
                train_y.push(labels[i].clone());
            }
        }
        println!(
            "  fold {}: train={}, test={}",
            fold + 1,
            train_hvs.len(),
            test_hvs.len()
        );
        let probe = LogisticProbe::train(&train_hvs, &train_y, cfg);
        let acc = probe.accuracy(&test_hvs, &test_y);
        println!("  fold {}: test accuracy = {:.3}", fold + 1, acc);
        accs.push(acc);
    }
    accs
}
