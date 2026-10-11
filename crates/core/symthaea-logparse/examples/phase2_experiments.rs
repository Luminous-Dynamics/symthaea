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
//!     [--scenario-level-evaluation]

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use symthaea_logparse::encoder::{
    DOMAIN_INVARIANT_ENCODER_VERSION, Hdv, encode, encode_domain_invariant,
};
use symthaea_logparse::probe::{
    classification_report, ClassificationReport, LogisticProbe, TrainConfig,
};
use symthaea_logparse::scenario::{
    aggregate_ordered_scenarios, aggregate_scenarios, validate_matching_scenario_modes,
    ORDERED_SCENARIO_ENCODER_VERSION, ScenarioEmbedding, ScenarioEncodingMode,
};
use symthaea_logparse::{LogEvent, evtx_source, otrf_source};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let evtx_dir = std::env::args().nth(1).ok_or("need evtx corpus dir")?;
    let otrf_dir = std::env::args().nth(2).ok_or("need otrf corpus dir")?;
    let experiment_flags: Vec<String> = std::env::args().skip(3).collect();
    let run_domain_invariant_ablation = experiment_flags
        .iter()
        .any(|flag| flag == "--domain-invariant-ablation");
    let run_scenario_level_evaluation = experiment_flags
        .iter()
        .any(|flag| flag == "--scenario-level-evaluation");

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
    let (otrf_events, otrf_groups) =
        load_otrf_corpus(&PathBuf::from(&otrf_dir), &shared_labels, 50)?;
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
    match run_grouped_kfold(&sbou_hvs, &sbou_labels, &sbou_groups, 5, 0xDEADBEEF, cfg) {
        Ok(fold_accs) if !fold_accs.is_empty() => {
            let mean = fold_accs.iter().sum::<f32>() / fold_accs.len() as f32;
            let var = fold_accs
                .iter()
                .map(|a| (a - mean).powi(2))
                .sum::<f32>()
                / fold_accs.len() as f32;
            let std = var.sqrt();
            println!("  per-fold test accuracy: {:?}", fold_accs);
            println!("  mean ± std: {mean:.3} ± {std:.3}");
            println!(
                "  min/max:    {:.3} / {:.3}",
                fold_accs.iter().cloned().fold(f32::INFINITY, f32::min),
                fold_accs.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            );
        }
        Ok(_) => println!("  grouped CV unavailable: no fold metrics produced"),
        Err(error) => println!(
            "  grouped CV unavailable under the strict capture-group contract: {error}"
        ),
    }

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
    let raw_ood_predictions = scale_up_probe.predict_batch(&otrf_hvs);
    let raw_ood_report =
        classification_report(&otrf_labels, &raw_ood_predictions, &scale_up_probe.class_names)?;
    print_classification_report("raw encoder OOD", &raw_ood_report);

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
        let di_ood_predictions = di_probe.predict_batch(&otrf_di_hvs);
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
        println!(
            "{}",
            concat!(
                "  Class-level FPs measure confusion among these labels, ",
                "not benign operational false positives."
            )
        );
    }

    if run_scenario_level_evaluation {
        println!("\n=== Capture/scenario-level evaluation ===");
        println!("  One vector per labeled source file or scenario directory.");
        println!("  Compare an orderless bundle with relative record-position encoding.");
        println!(
            "  Ordered scenario encoder version: {ORDERED_SCENARIO_ENCODER_VERSION}"
        );
        println!("  Relative record order is not proof of timestamp order or causality.");

        let sbou_raw_scenarios = aggregate_scenarios(&sbou_hvs, &sbou_labels, &sbou_groups)?;
        let otrf_raw_scenarios = aggregate_scenarios(&otrf_hvs, &otrf_labels, &otrf_groups)?;
        run_scenario_classifier(
            "raw encoder / orderless bundle",
            &sbou_raw_scenarios,
            &otrf_raw_scenarios,
            cfg,
        )?;

        let sbou_raw_ordered =
            aggregate_ordered_scenarios(&sbou_hvs, &sbou_labels, &sbou_groups)?;
        let otrf_raw_ordered =
            aggregate_ordered_scenarios(&otrf_hvs, &otrf_labels, &otrf_groups)?;
        run_scenario_classifier(
            "raw encoder / relative record position",
            &sbou_raw_ordered,
            &otrf_raw_ordered,
            cfg,
        )?;

        if run_domain_invariant_ablation {
            let sbou_di_hvs: Vec<Hdv> =
                sbou_events.iter().map(encode_domain_invariant).collect();
            let otrf_di_hvs: Vec<Hdv> =
                otrf_events.iter().map(encode_domain_invariant).collect();
            let sbou_di_scenarios =
                aggregate_scenarios(&sbou_di_hvs, &sbou_labels, &sbou_groups)?;
            let otrf_di_scenarios =
                aggregate_scenarios(&otrf_di_hvs, &otrf_labels, &otrf_groups)?;
            run_scenario_classifier(
                "domain-invariant encoder / orderless bundle",
                &sbou_di_scenarios,
                &otrf_di_scenarios,
                cfg,
            )?;

            let sbou_di_ordered =
                aggregate_ordered_scenarios(&sbou_di_hvs, &sbou_labels, &sbou_groups)?;
            let otrf_di_ordered =
                aggregate_ordered_scenarios(&otrf_di_hvs, &otrf_labels, &otrf_groups)?;
            run_scenario_classifier(
                "domain-invariant encoder / relative record position",
                &sbou_di_ordered,
                &otrf_di_ordered,
                cfg,
            )?;
        }
    }

    Ok(())
}

fn run_scenario_classifier(
    name: &str,
    source_scenarios: &[ScenarioEmbedding],
    held_out_scenarios: &[ScenarioEmbedding],
    cfg: TrainConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let expected_mode =
        validate_matching_scenario_modes(source_scenarios, held_out_scenarios)?;
    let mode_name = match expected_mode {
        ScenarioEncodingMode::OrderlessBundle => "orderless-bundle",
        ScenarioEncodingMode::RelativeRecordPosition => "relative-record-position",
    };
    println!("  scenario encoding mode: {mode_name}");

    let source_hvs: Vec<Hdv> = source_scenarios
        .iter()
        .map(|scenario| scenario.vector.clone())
        .collect();
    let source_labels: Vec<String> = source_scenarios
        .iter()
        .map(|scenario| scenario.label.clone())
        .collect();
    let source_groups: Vec<String> = source_scenarios
        .iter()
        .map(|scenario| scenario.scenario_id.clone())
        .collect();
    let held_out_hvs: Vec<Hdv> = held_out_scenarios
        .iter()
        .map(|scenario| scenario.vector.clone())
        .collect();
    let held_out_labels: Vec<String> = held_out_scenarios
        .iter()
        .map(|scenario| scenario.label.clone())
        .collect();

    println!("\n--- Scenario-level model: {name} ---");
    println!(
        "  source captures/scenarios: {}, held-out corpus scenarios: {}",
        source_scenarios.len(),
        held_out_scenarios.len()
    );
    println!(
        "  source event count represented by bundles: {}",
        source_scenarios
            .iter()
            .map(|scenario| scenario.event_count)
            .sum::<usize>()
    );
    print_distribution("source scenarios by label", &source_labels);
    print_distribution("OTRF scenarios by label", &held_out_labels);

    // Same three split seeds are reused for every scenario encoder so results
    // are paired across representation variants.
    let scenario_repeat_seeds = [0xDEADBEEF, 0x51CE_0002, 0x51CE_0003];
    match run_repeated_grouped_kfold(
        &source_hvs,
        &source_labels,
        &source_groups,
        5,
        &scenario_repeat_seeds,
        cfg,
    ) {
        Ok(repeat_means) if !repeat_means.is_empty() => {
            let mean = repeat_means.iter().sum::<f32>() / repeat_means.len() as f32;
            let variance = repeat_means
                .iter()
                .map(|accuracy| (accuracy - mean).powi(2))
                .sum::<f32>()
                / repeat_means.len() as f32;
            println!(
                "  repeated grouped scenario 5-fold accuracy: {mean:.3} ± {:.3} (repeat means)",
                variance.sqrt()
            );
            println!("  per-repeat means: {repeat_means:?}");
        }
        Ok(_) => println!("  grouped scenario CV unavailable: no repeat metrics produced"),
        Err(error) => println!(
            "  grouped scenario CV unavailable under the strict capture-group contract: {error}"
        ),
    }

    let (train_hvs, train_labels, test_hvs, test_labels) =
        symthaea_logparse::probe::grouped_stratified_split(
            &source_hvs,
            &source_labels,
            &source_groups,
            0.8,
            0x51CE_0001,
        )?;
    let probe = LogisticProbe::train(&train_hvs, &train_labels, cfg);
    let held_out_predictions = probe.predict_batch(&test_hvs);
    let held_out_report =
        classification_report(&test_labels, &held_out_predictions, &probe.class_names)?;
    print_classification_report("source-capture holdout", &held_out_report);

    let ood_predictions = probe.predict_batch(&held_out_hvs);
    let ood_report = classification_report(
        &held_out_labels,
        &ood_predictions,
        &probe.class_names,
    )?;
    print_classification_report("separate-corpus scenario-level OOD", &ood_report);
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
) -> Result<(Vec<LogEvent>, Vec<String>), Box<dyn std::error::Error>> {
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
    let mut groups: Vec<String> = Vec::new();
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
        let group_id = format!("otrf:{dname}");
        for mut ev in capped {
            ev.label = Some(label.clone());
            out.push(ev);
            groups.push(group_id.clone());
        }
    }
    Ok((out, groups))
}

/// Stratified k-fold cross-validation with source groups kept intact.
fn run_repeated_grouped_kfold(
    hvs: &[Hdv],
    labels: &[String],
    groups: &[String],
    k: usize,
    seeds: &[u64],
    cfg: TrainConfig,
) -> Result<Vec<f32>, String> {
    if seeds.is_empty() {
        return Err("repeated grouped CV needs at least one split seed".to_string());
    }
    let mut repeat_means = Vec::with_capacity(seeds.len());
    for (index, &seed) in seeds.iter().enumerate() {
        println!(
            "\n  grouped CV repeat {} of {} (split seed={seed:#x})",
            index + 1,
            seeds.len()
        );
        let fold_accuracies = run_grouped_kfold(hvs, labels, groups, k, seed, cfg)?;
        if fold_accuracies.is_empty() {
            return Err("grouped CV returned no fold metrics".to_string());
        }
        let mean = fold_accuracies.iter().sum::<f32>() / fold_accuracies.len() as f32;
        println!("  repeat {} mean accuracy: {mean:.3}", index + 1);
        repeat_means.push(mean);
    }
    Ok(repeat_means)
}

fn run_grouped_kfold(
    hvs: &[Hdv],
    labels: &[String],
    groups: &[String],
    k: usize,
    seed: u64,
    cfg: TrainConfig,
) -> Result<Vec<f32>, String> {
    if hvs.len() != labels.len() {
        return Err("grouped CV feature/label length mismatch".to_string());
    }
    let fold_of = symthaea_logparse::probe::grouped_fold_assignment(labels, groups, k, seed)?;

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

