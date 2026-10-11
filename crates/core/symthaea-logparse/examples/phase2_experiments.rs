//! Phase 2 experiment runner.
//!
//! Runs the three experiments we owe ourselves after the Apr 14 probe
//! result of 0.815 on the single 80/20 split:
//!
//!   1. **Scale-up**: train on the full sbousseaden corpus (post per-file
//!      cap), not the 1000-event downsample. Does the test accuracy move?
//!
//!   2. **Grouped 5-fold cross-validation**: keep each EVTX source file entirely
//!      within one fold, so neighboring events from one capture cannot appear
//!      in both train and test.
//!
//!   3. **Cross-corpus OOD**: train on sbousseaden, test on OTRF
//!      Security-Datasets (Mordor-format JSONL). Measures whether the
//!      encoder learned "what credential access looks like" or "what
//!      sbousseaden's curation looks like."
//!
//! Usage:
//!   cargo run -p symthaea-logparse --example phase2_experiments --release -- \
//!     /tmp/evtx-staged /tmp/otrf-staged [--domain-invariant-ablation]

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use symthaea_logparse::encoder::{
    DOMAIN_INVARIANT_ENCODER_VERSION, Hdv, encode, encode_domain_invariant,
};
use symthaea_logparse::probe::{
    classification_report, selective_accuracy_curve, ClassificationReport, LogisticProbe,
    TrainConfig,
};
use symthaea_logparse::{LogEvent, evtx_source, otrf_source};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let evtx_dir = std::env::args().nth(1).ok_or("need evtx corpus dir")?;
    let otrf_dir = std::env::args().nth(2).ok_or("need otrf corpus dir")?;
    let run_domain_invariant_ablation =
        std::env::args().nth(3).as_deref() == Some("--domain-invariant-ablation");

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
    let (sbou_events, sbou_groups) =
        load_evtx_corpus(&PathBuf::from(&evtx_dir), 50, None, &shared_labels)?;
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
    println!("\n=== Experiment 1: scale-up (sbousseaden full corpus, grouped 80/20 split) ===");
    let (tr_hv, tr_y, te_hv, te_y) =
        symthaea_logparse::probe::grouped_stratified_split(
            &sbou_hvs, &sbou_labels, &sbou_groups, 0.8, 0xFACADE,
        )?;
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
    println!("\n=== Experiment 2: grouped 5-fold CV on sbousseaden ===");
    let fold_accs = run_grouped_kfold(&sbou_hvs, &sbou_labels, &sbou_groups, 5, cfg)?;
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

    // Cross-corpus evaluation includes class-level precision/recall/F1 and
    // a confusion matrix. OTRF is not a benign baseline, so this is not an
    // operational benign false-positive rate.
    println!("\n=== Raw-encoder OOD classification report ===");
    let raw_ood_details: Vec<(usize, f32)> = otrf_hvs
        .iter()
        .map(|hv| scale_up_probe.predict_with_margin(hv))
        .collect();
    let raw_ood_predictions: Vec<usize> =
        raw_ood_details.iter().map(|(class, _)| *class).collect();
    let raw_ood_margins: Vec<f32> = raw_ood_details.iter().map(|(_, margin)| *margin).collect();
    let raw_ood_report =
        classification_report(&otrf_labels, &raw_ood_predictions, &scale_up_probe.class_names)?;
    print_classification_report("raw encoder OOD", &raw_ood_report);
    print_selective_curve(
        "raw encoder OOD",
        &otrf_labels,
        &raw_ood_predictions,
        &raw_ood_margins,
        &scale_up_probe.class_names,
    )?;

    // ----------------------------------------------------------------
    // Summary
    // ----------------------------------------------------------------
    println!("\n=== Phase 2 summary ===");
    println!("  sbousseaden single split:    {test_acc:.3}");
    println!("  sbousseaden 5-fold CV:       {mean:.3} ± {std:.3}");
    println!("  OOD (train sbou, test otrf): {ood_test_acc:.3}");
    println!("  chance (7 classes):          {:.3}", 1.0 / 7.0);
    println!("  Phase 1 cutoff:              0.500");

    if run_domain_invariant_ablation {
        println!("\n=== Optional ablation: domain-invariant encoder ===");
        println!("  This is a paired experiment using the same source split seed.");
        println!("  It is not a new headline result until run on the staged corpora.");
        println!("  ablation encoder version: {DOMAIN_INVARIANT_ENCODER_VERSION}");

        let sbou_di_hvs: Vec<Hdv> = sbou_events.iter().map(encode_domain_invariant).collect();
        let otrf_di_hvs: Vec<Hdv> = otrf_events.iter().map(encode_domain_invariant).collect();
        let (tr_hv, tr_y, te_hv, te_y) =
            symthaea_logparse::probe::grouped_stratified_split(
                &sbou_di_hvs, &sbou_labels, &sbou_groups, 0.8, 0xFACADE,
            )?;
        let di_probe = LogisticProbe::train(&tr_hv, &tr_y, cfg);
        let di_in_distribution = di_probe.accuracy(&te_hv, &te_y);
        let di_ood_details: Vec<(usize, f32)> = otrf_di_hvs
            .iter()
            .map(|hv| di_probe.predict_with_margin(hv))
            .collect();
        let di_ood_predictions: Vec<usize> =
            di_ood_details.iter().map(|(class, _)| *class).collect();
        let di_ood_margins: Vec<f32> = di_ood_details.iter().map(|(_, margin)| *margin).collect();
        let di_ood_report =
            classification_report(&otrf_labels, &di_ood_predictions, &di_probe.class_names)?;

        println!("  baseline raw in-distribution accuracy: {test_acc:.3}");
        println!("  ablation in-distribution accuracy:      {di_in_distribution:.3}");
        println!("  baseline raw OOD accuracy:              {ood_test_acc:.3}");
        println!("  ablation OOD accuracy:                  {:.3}", di_ood_report.accuracy);
        println!("  baseline raw OOD macro recall:          {:.3}", raw_ood_report.macro_recall);
        println!("  ablation OOD macro recall:              {:.3}", di_ood_report.macro_recall);
        println!("  baseline raw OOD macro F1:              {:.3}", raw_ood_report.macro_f1);
        println!("  ablation OOD macro F1:                  {:.3}", di_ood_report.macro_f1);
        print_classification_report("domain-invariant OOD", &di_ood_report);
        print_selective_curve(
            "domain-invariant OOD",
            &otrf_labels,
            &di_ood_predictions,
            &di_ood_margins,
            &di_probe.class_names,
        )?;
        println!(
            "{}",
            concat!(
                "  Class-level FPs measure confusion among these labels, ",
                "not benign operational false positives."
            )
        );
    }

    Ok(())
}

fn print_selective_curve(
    name: &str,
    labels: &[String],
    predictions: &[usize],
    margins: &[f32],
    class_names: &[String],
) -> Result<(), String> {
    let points = selective_accuracy_curve(
        labels,
        predictions,
        margins,
        class_names,
        &[0.50, 0.75, 0.90, 1.00],
    )?;
    println!("  {name} uncalibrated margin-ranked selective evaluation:");
    println!(
        "  {:>9} {:>9} {:>10} {:>12} {:>11} {:>17}",
        "target", "coverage", "accepted", "min margin", "accuracy", "macro recall"
    );
    for point in points {
        println!(
            "  {:>8.0}% {:>8.0}% {:>10} {:>12.4} {:>11.3} {:>17.3}",
            point.target_coverage * 100.0,
            point.actual_coverage * 100.0,
            point.accepted_samples,
            point.margin_threshold,
            point.accepted_accuracy,
            point.accepted_macro_recall
        );
    }
    println!(
        "  {}",
        concat!(
            "Margin ranking is not calibrated confidence; ",
            "this curve does not select a deployment threshold."
        )
    );
    Ok(())
}

fn print_classification_report(name: &str, report: &ClassificationReport) {
    println!(
        "  {name}: n={}, accuracy={:.3}, macro precision={:.3}, macro recall={:.3}, macro F1={:.3}",
        report.samples,
        report.accuracy,
        report.macro_precision,
        report.macro_recall,
        report.macro_f1
    );
    println!(
        "  {:<22} {:>6} {:>6} {:>5} {:>5} {:>5} {:>9} {:>9} {:>9}",
        "class", "support", "pred", "TP", "FP", "FN", "precision", "recall", "F1"
    );
    for (class, metrics) in &report.per_class {
        println!(
            "  {class:<22} {:>6} {:>6} {:>5} {:>5} {:>5} {:>9.3} {:>9.3} {:>9.3}",
            metrics.support,
            metrics.predicted,
            metrics.true_positive,
            metrics.false_positive,
            metrics.false_negative,
            metrics.precision,
            metrics.recall,
            metrics.f1
        );
    }

    println!("  Confusion matrix (rows=actual, columns=predicted):");
    print!("  {:<22}", "actual");
    for class in report.confusion_matrix.keys() {
        print!(" {:>7}", class);
    }
    println!();
    for (actual, row) in &report.confusion_matrix {
        print!("  {actual:<22}");
        for predicted in report.confusion_matrix.keys() {
            print!(" {:>7}", row[predicted]);
        }
        println!();
    }
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
) -> Result<(Vec<LogEvent>, Vec<String>), Box<dyn std::error::Error>> {
    let labels_path = corpus_dir.join("labels.csv");
    let labels_csv = std::fs::read_to_string(&labels_path)?;
    let mut file_labels: HashMap<String, String> = HashMap::new();
    for line in labels_csv.lines().skip(1) {
        if let Some((f, l)) = line.split_once(',') {
            file_labels.insert(f.trim().to_string(), l.trim().to_string());
        }
    }

    let keep_set: std::collections::HashSet<&str> = keep_labels.iter().copied().collect();
    let mut by_label: BTreeMap<String, Vec<(LogEvent, String)>> = BTreeMap::new();
    let mut paths: Vec<PathBuf> = std::fs::read_dir(corpus_dir)?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<Result<_, _>>()?;
    paths.sort();

    for path in paths {
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
            let stride = parsed.len() / max_per_file.max(1);
            parsed
                .into_iter()
                .step_by(stride.max(1))
                .take(max_per_file)
                .collect()
        };
        let bucket = by_label.entry(label.clone()).or_default();
        for mut ev in capped {
            ev.label = Some(label.clone());
            bucket.push((ev, format!("evtx:{fname}")));
        }
    }

    let rows: Vec<(LogEvent, String)> = if let Some(cap) = max_total {
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
    let (events, groups): (Vec<LogEvent>, Vec<String>) = rows.into_iter().unzip();
    Ok((events, groups))
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
    let mut paths: Vec<PathBuf> = std::fs::read_dir(corpus_dir)?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<Result<_, _>>()?;
    paths.sort();

    let mut out: Vec<LogEvent> = Vec::new();
    for path in paths {
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
            let stride = parsed.len() / max_per_dir.max(1);
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

/// Stratified k-fold cross-validation with source groups kept intact.
fn run_grouped_kfold(
    hvs: &[Hdv],
    labels: &[String],
    groups: &[String],
    k: usize,
    cfg: TrainConfig,
) -> Result<Vec<f32>, String> {
    if hvs.len() != labels.len() {
        return Err("grouped CV feature/label length mismatch".to_string());
    }
    let fold_of =
        symthaea_logparse::probe::grouped_fold_assignment(labels, groups, k, 0xDEADBEEF)?;

    let mut accs = Vec::with_capacity(k);
    for fold in 0..k {
        let mut train_hvs = Vec::new();
        let mut train_y = Vec::new();
        let mut test_hvs = Vec::new();
        let mut test_y = Vec::new();
        for i in 0..hvs.len() {
            if fold_of[i] == fold {
                test_hvs.push(hvs[i].clone());
                test_y.push(labels[i].clone());
            } else {
                train_hvs.push(hvs[i].clone());
                train_y.push(labels[i].clone());
            }
        }
        println!(
            "  grouped fold {}: train={}, test={}",
            fold + 1,
            train_hvs.len(),
            test_hvs.len()
        );
        let probe = LogisticProbe::train(&train_hvs, &train_y, cfg);
        let accuracy = probe.accuracy(&test_hvs, &test_y);
        println!("  grouped fold {}: test accuracy = {:.3}", fold + 1, accuracy);
        accs.push(accuracy);
    }
    Ok(accs)
}

