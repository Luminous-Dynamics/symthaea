//! Phase 1 benchmark runner (skeleton).
//!
//! Usage (once a labeled corpus is staged):
//!   cargo run -p symthaea-logparse --example cluster_evtx -- /path/to/corpus/
//!
//! Expected corpus layout:
//!   /path/to/corpus/
//!     labels.csv      # filename,label  (one row per .evtx file)
//!     <file1>.evtx
//!     <file2>.evtx
//!     ...
//!
//! What this does:
//!   1. Parse every .evtx file, attach the corpus label to every event
//!   2. Encode each event to a 16,384D hypervector
//!   3. (STUB) Run clustering — currently just nearest-centroid against
//!      per-label centroids as a sanity baseline
//!   4. Compute cluster purity against the ground-truth labels
//!   5. Print result. If purity >= 0.50, Phase 1 kill criterion is passed.
//!
//! TODO (next session):
//!   - Swap nearest-centroid for real HDBSCAN (linfa-clustering or the
//!     `hdbscan` crate)
//!   - Add the DFIR.training corpus staging script
//!   - Record per-provider purity breakdown (which event classes cluster
//!     well, which don't — this tells us whether the encoder needs more
//!     role channels)

use std::collections::HashMap;
use std::path::PathBuf;
use symthaea_logparse::encoder::{bundle, encode, Hdv};
use symthaea_logparse::{cluster, evtx_source, LogEvent};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let corpus_dir = std::env::args()
        .nth(1)
        .ok_or("usage: cluster_evtx <corpus_dir>")?;
    let corpus_dir = PathBuf::from(corpus_dir);

    // Load labels.csv (filename,label)
    let labels_path = corpus_dir.join("labels.csv");
    if !labels_path.exists() {
        eprintln!(
            "No labels.csv in {}. Phase 1 spike needs a labeled corpus. \
             See memory/project_msp_wedge.md for DFIR.training staging plan.",
            corpus_dir.display()
        );
        return Ok(());
    }
    let labels_csv = std::fs::read_to_string(&labels_path)?;
    let mut file_labels: HashMap<String, String> = HashMap::new();
    for line in labels_csv.lines().skip(1) {
        if let Some((f, l)) = line.split_once(',') {
            file_labels.insert(f.trim().to_string(), l.trim().to_string());
        }
    }

    // Parse + encode every file.
    let mut all_events: Vec<LogEvent> = Vec::new();
    let mut all_hvs: Vec<Hdv> = Vec::new();
    let mut gt: Vec<String> = Vec::new();

    for entry in std::fs::read_dir(&corpus_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("evtx") {
            continue;
        }
        let fname = path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let Some(label) = file_labels.get(&fname) else {
            eprintln!("skip {fname}: no label");
            continue;
        };

        let events = evtx_source::parse_evtx_file(&path)?;
        println!("{fname}: {} events, label={label}", events.len());

        for mut ev in events {
            ev.label = Some(label.clone());
            all_hvs.push(encode(&ev));
            gt.push(label.clone());
            all_events.push(ev);
        }
    }

    if all_events.is_empty() {
        eprintln!("no events parsed");
        return Ok(());
    }

    // Nearest-centroid baseline: build one centroid per ground-truth label
    // by bundling all HVs of that class.
    let mut by_label: HashMap<&str, Vec<Hdv>> = HashMap::new();
    for (hv, gt) in all_hvs.iter().zip(gt.iter()) {
        by_label.entry(gt.as_str()).or_default().push(hv.clone());
    }
    let labels_in_order: Vec<&str> = by_label.keys().copied().collect();
    let centroids: Vec<Hdv> = labels_in_order
        .iter()
        .map(|l| bundle(&by_label[*l]))
        .collect();

    let assignments = cluster::nearest_centroid(&all_hvs, &centroids);
    // Map assignments back to predicted labels, then to i32 buckets for purity.
    let gt_refs: Vec<&str> = gt.iter().map(|s| s.as_str()).collect();
    let purity = cluster::purity(&assignments, &gt_refs);

    println!("\n=== Phase 1 spike result ===");
    println!("events:    {}", all_events.len());
    println!("classes:   {}", labels_in_order.len());
    println!("purity:    {purity:.3}");
    println!(
        "kill criterion (>= 0.50): {}",
        if purity >= 0.50 { "PASS" } else { "FAIL" }
    );
    println!(
        "\nNote: this is the nearest-centroid BASELINE, not HDBSCAN. \
         The real spike is the centroid-free clustering run — TODO."
    );

    Ok(())
}
