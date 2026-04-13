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
use symthaea_logparse::cluster::{hdbscan_cluster, purity};
use symthaea_logparse::encoder::{encode, Hdv};
use symthaea_logparse::fixtures::generate_synthetic_corpus;
use symthaea_logparse::{evtx_source, LogEvent};

const KILL_CRITERION: f32 = 0.50;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arg = std::env::args().nth(1).ok_or(
        "usage:\n  cluster_evtx --synthetic\n  cluster_evtx <corpus_dir>",
    )?;

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

    println!("[cluster] running HDBSCAN on {} × 16384D hypervectors...", hvs.len());
    let min_cluster = (hvs.len() / 20).max(5);
    let labels = hdbscan_cluster(&hvs, Some(min_cluster))?;

    let noise = labels.iter().filter(|&&c| c == -1).count();
    let distinct: std::collections::HashSet<_> =
        labels.iter().filter(|&&c| c != -1).collect();
    let p = purity(&labels, &gt_refs);

    println!("\n=== Phase 1 result ({mode_label}) ===");
    println!("events:          {}", hvs.len());
    println!("ground-truth:    {} classes", count_classes(&gt));
    println!("min_cluster:     {min_cluster}");
    println!("clusters found:  {}", distinct.len());
    println!("noise points:    {}/{}", noise, hvs.len());
    println!("purity:          {p:.3}");
    if arg == "--synthetic" {
        println!(
            "\nNOTE: synthetic mode does NOT validate the thesis. Real corpus \
             required for Phase 1 kill-criterion evaluation."
        );
    } else {
        let verdict = if p >= KILL_CRITERION { "PASS" } else { "FAIL" };
        println!("\nkill criterion (>= {KILL_CRITERION:.2}): {verdict}");
    }

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

    let mut events = Vec::new();
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
        println!("{fname}: {} events, label={label}", parsed.len());
        for mut ev in parsed {
            ev.label = Some(label.clone());
            events.push(ev);
        }
    }
    Ok(events)
}

fn count_classes(gt: &[String]) -> usize {
    let set: std::collections::HashSet<&str> = gt.iter().map(|s| s.as_str()).collect();
    set.len()
}
