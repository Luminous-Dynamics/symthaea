use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum SourceMode {
    ScriptedDemo,
    LiveSimulation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DashboardEvent {
    pub event_id: String,
    pub event_type: String,
    pub label: String,
    pub description: String,
    pub severity: String,
    pub absolute_frame_index: u64,
    pub history_offset: u64,
    pub causal_role: String,
    pub metric_impacts: Vec<(String, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RegimeInterval {
    pub r#type: String,
    pub start_frame: u64,
    pub end_frame: u64,
    pub duration_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TimeWaterfallFrame {
    pub frame_index: u64,
    pub sim_time_s: f64,
    pub source_mode: SourceMode,
    pub metrics: [f64; 7],
    pub confidence: f64,
    pub anomaly_bits: u8,
    pub anomaly_flags: Vec<String>,
    pub is_chronicle: bool,
    pub events: Vec<DashboardEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExportSummary {
    pub event_count: usize,
    pub peak_fep_prediction_error: f64,
    pub min_phi_integration: f64,
    pub peak_anomaly_score: f64,
    pub peak_memory_pressure: f64,
    pub peak_mip_instability: f64,
    pub perturbation_frame: Option<u64>,
    pub recovery_frame: Option<u64>,
    pub first_chronicle_marked_frame: Option<u64>,
    pub durability_commit_frame: Option<u64>,
    pub chronicle_marked_frame_count: usize,
    pub durable_record_event_count: usize,
    pub frames_to_recovery: Option<u64>,
    pub seconds_to_recovery: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EvidenceBundle {
    pub schema_version: String,
    pub exported_at: String,
    pub source_mode: SourceMode,
    pub scenario: String,
    pub frame_order: String,
    pub history_len: usize,
    pub metric_names: Vec<String>,
    pub anomaly_bit_legend: std::collections::HashMap<String, String>,
    pub frames: Vec<TimeWaterfallFrame>,
    pub intervals: Vec<RegimeInterval>,
    pub summary: ExportSummary,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: time_waterfall_analyze <path_to_export_json>");
        std::process::exit(1);
    }

    let path = Path::new(&args[1]);
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error opening file: {}", e);
            std::process::exit(1);
        }
    };

    let mut content = String::new();
    if let Err(e) = file.read_to_string(&mut content) {
        eprintln!("Error reading file: {}", e);
        std::process::exit(1);
    }

    let bundle: EvidenceBundle = match serde_json::from_str(&content) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("JSON Schema Validation Error: {}", e);
            std::process::exit(1);
        }
    };

    // Analyze the story
    let classification = if bundle.summary.recovery_frame.is_some()
        && bundle
            .intervals
            .iter()
            .any(|i| i.r#type == "false_green_diagnostic")
    {
        "recovered_after_false_green"
    } else {
        "stable"
    };

    // Perform strict schema validation checks
    let mut validation_errors = Vec::new();

    // 1. Verify chronological frames order
    for idx in 1..bundle.frames.len() {
        if bundle.frames[idx].frame_index <= bundle.frames[idx - 1].frame_index {
            validation_errors.push(format!(
                "Frame ordering violation at index {}: index {} <= previous index {}",
                idx,
                bundle.frames[idx].frame_index,
                bundle.frames[idx - 1].frame_index
            ));
        }
    }

    // 2. Verify events absolute_frame_index vs history_offset presence
    for f in &bundle.frames {
        for e in &f.events {
            if e.absolute_frame_index == 0 {
                validation_errors.push(format!(
                    "Event {} missing valid absolute_frame_index",
                    e.event_id
                ));
            }
        }
    }

    // 3. Verify durability_commit_frame is not confused with first_chronicle_marked_frame
    if let (Some(chron), Some(dur)) = (
        bundle.summary.first_chronicle_marked_frame,
        bundle.summary.durability_commit_frame,
    ) {
        if chron >= dur {
            validation_errors.push(format!(
                "Semantics conflict: first_chronicle_marked_frame ({}) should precede durability_commit_frame ({})",
                chron, dur
            ));
        }
    }

    // 4. Verify intervals have positive duration and correct boundaries
    for (idx, interval) in bundle.intervals.iter().enumerate() {
        if interval.start_frame > interval.end_frame {
            validation_errors.push(format!(
                "Interval {} (type {}) start frame {} > end frame {}",
                idx, interval.r#type, interval.start_frame, interval.end_frame
            ));
        }
        if interval.duration_s <= 0.0 {
            validation_errors.push(format!(
                "Interval {} (type {}) has invalid duration: {}",
                idx, interval.r#type, interval.duration_s
            ));
        }
    }

    if !validation_errors.is_empty() {
        eprintln!(
            "❌ Schema validation failed with {} errors:",
            validation_errors.len()
        );
        for err in &validation_errors {
            eprintln!("  - {}", err);
        }
        std::process::exit(2);
    }

    let perturbation_frame_str = bundle
        .summary
        .perturbation_frame
        .map(|f| format!("frame {}", f))
        .unwrap_or_else(|| "none".to_string());

    let mut pred_contr_str = "none".to_string();
    let mut false_green_str = "none".to_string();
    let mut recovery_str = "none".to_string();

    for i in &bundle.intervals {
        match i.r#type.as_str() {
            "prediction_contradiction" => {
                pred_contr_str = format!("{}–{}", i.start_frame, i.end_frame);
            }
            "false_green_diagnostic" => {
                false_green_str = format!("{}–{}", i.start_frame, i.end_frame);
            }
            "recovery_mode" => {
                recovery_str = format!("{}–{}", i.start_frame, i.end_frame);
            }
            _ => {}
        }
    }

    let durability_commit_str = bundle
        .summary
        .durability_commit_frame
        .map(|f| f.to_string())
        .unwrap_or_else(|| "none".to_string());

    let recovery_time_str = bundle
        .summary
        .seconds_to_recovery
        .map(|s| format!("{:.1}s", s))
        .unwrap_or_else(|| "none".to_string());

    // Determine audit recommendation
    let recommended_audit = if bundle
        .intervals
        .iter()
        .any(|i| i.r#type == "false_green_diagnostic")
    {
        "inspect false-green interval"
    } else if bundle.summary.peak_fep_prediction_error > 0.6 {
        "inspect contradiction interval"
    } else {
        "none"
    };

    println!("schema: {}", bundle.schema_version);
    println!("scenario: {}", bundle.scenario);
    println!("frames: {}", bundle.frames.len());
    println!("classification: {}", classification);
    println!("perturbation: {}", perturbation_frame_str);
    println!("prediction_contradiction: {}", pred_contr_str);
    println!("false_green: {}", false_green_str);
    println!("recovery: {}", recovery_str);
    println!("durability_commit: {}", durability_commit_str);
    println!("peak_fep: {:.2}", bundle.summary.peak_fep_prediction_error);
    println!("min_phi: {:.2}", bundle.summary.min_phi_integration);
    println!("recovery_time: {}", recovery_time_str);
    println!("recommended_audit: {}", recommended_audit);
    println!("✓ Headless schema validation passed cleanly.");
}
