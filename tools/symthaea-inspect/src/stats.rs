/*!
 * Statistics and analysis of traces
 */

use anyhow::Result;
use std::path::Path;
use std::collections::HashMap;

use crate::trace::{Trace, EventType, WorkspaceIgnitionData, RouterSelectionData};

pub fn show_stats(trace_path: &Path, detailed: bool) -> Result<()> {
    let trace = Trace::load(trace_path)?;

    println!("📊 Trace Statistics");
    println!();
    println!("Session: {}", trace.session_id);
    println!("Started: {}", trace.timestamp_start);
    if let Some(end) = &trace.timestamp_end {
        println!("Ended: {}", end);
    }
    println!();

    // Basic counts
    println!("Events:");
    println!("  Total: {}", trace.events.len());

    let mut type_counts: HashMap<EventType, usize> = HashMap::new();
    for event in &trace.events {
        *type_counts.entry(event.event_type.clone()).or_insert(0) += 1;
    }

    for (event_type, count) in &type_counts {
        println!("  {}: {}", event_type, count);
    }
    println!();

    // Φ statistics
    let phi_measurements: Vec<f64> = trace.events.iter()
        .filter(|e| e.event_type == EventType::WorkspaceIgnition)
        .filter_map(|e| {
            e.data.as_ref().and_then(|d| {
                serde_json::from_value::<WorkspaceIgnitionData>(d.clone()).ok()
            })
        })
        .map(|d| d.phi)
        .collect();

    if !phi_measurements.is_empty() {
        println!("Φ (Integrated Information):");
        let avg = phi_measurements.iter().sum::<f64>() / phi_measurements.len() as f64;
        let max = phi_measurements.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = phi_measurements.iter().cloned().fold(f64::INFINITY, f64::min);

        println!("  Average: {:.3}", avg);
        println!("  Max: {:.3}", max);
        println!("  Min: {:.3}", min);

        if detailed {
            // Calculate standard deviation
            let variance = phi_measurements.iter()
                .map(|&x| (x - avg).powi(2))
                .sum::<f64>() / phi_measurements.len() as f64;
            let stddev = variance.sqrt();

            println!("  Std Dev: {:.3}", stddev);

            // Histogram
            println!();
            println!("  Distribution:");
            print_histogram(&phi_measurements, 10);
        }

        println!();
    }

    // Router statistics
    let router_selections: Vec<RouterSelectionData> = trace.events.iter()
        .filter(|e| e.event_type == EventType::RouterSelection)
        .filter_map(|e| {
            e.data.as_ref().and_then(|d| {
                serde_json::from_value::<RouterSelectionData>(d.clone()).ok()
            })
        })
        .collect();

    if !router_selections.is_empty() {
        println!("Router Selection:");

        let mut router_counts: HashMap<String, usize> = HashMap::new();
        let mut router_confidences: HashMap<String, Vec<f64>> = HashMap::new();

        for selection in &router_selections {
            *router_counts.entry(selection.selected_router.clone()).or_insert(0) += 1;
            router_confidences.entry(selection.selected_router.clone())
                .or_insert_with(Vec::new)
                .push(selection.confidence);
        }

        for (router, count) in &router_counts {
            let avg_confidence = if let Some(confidences) = router_confidences.get(router) {
                confidences.iter().sum::<f64>() / confidences.len() as f64
            } else {
                0.0
            };

            println!("  {}: {} times (avg confidence: {:.2}%)",
                router, count, avg_confidence * 100.0);
        }

        println!();
    }

    // Performance
    if let Some(summary) = &trace.summary {
        println!("Performance:");
        println!("  Duration: {} ms", summary.duration_ms);
        if summary.total_events > 0 {
            let avg_event_time = summary.duration_ms as f64 / summary.total_events as f64;
            println!("  Avg event time: {:.2} ms", avg_event_time);
        }
        println!();
    }

    // Errors and security
    let error_count = type_counts.get(&EventType::ErrorOccurred).unwrap_or(&0);
    if *error_count > 0 {
        println!("⚠️  Errors: {}", error_count);
    }

    if let Some(summary) = &trace.summary {
        if summary.security_denials > 0 {
            println!("🔒 Security denials: {}", summary.security_denials);
        }
    }

    Ok(())
}

fn print_histogram(data: &[f64], bins: usize) {
    if data.is_empty() {
        return;
    }

    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    if range == 0.0 {
        println!("    All values are {:.3}", min);
        return;
    }

    let bin_width = range / bins as f64;
    let mut counts = vec![0usize; bins];

    for &value in data {
        let bin = ((value - min) / bin_width).floor() as usize;
        let bin = bin.min(bins - 1); // Handle edge case where value == max
        counts[bin] += 1;
    }

    let max_count = counts.iter().max().unwrap_or(&1);
    let scale = 40.0 / *max_count as f64; // 40 chars wide

    for (i, count) in counts.iter().enumerate() {
        let bin_start = min + i as f64 * bin_width;
        let bin_end = bin_start + bin_width;
        let bar_width = (*count as f64 * scale) as usize;
        let bar = "█".repeat(bar_width);

        println!("    [{:.2} - {:.2}): {} ({})",
            bin_start, bin_end, bar, count);
    }
}
