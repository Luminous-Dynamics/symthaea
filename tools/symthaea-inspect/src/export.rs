/*!
 * Export trace metrics to various formats
 */

use anyhow::{Context, Result, bail};
use std::path::Path;
use std::io::Write;

use crate::trace::{Trace, EventType, WorkspaceIgnitionData};

pub fn export_metric(
    trace_path: &Path,
    metric: &str,
    format: &str,
    output: Option<&Path>,
) -> Result<()> {
    let trace = Trace::load(trace_path)?;

    match metric.to_lowercase().as_str() {
        "phi" => export_phi(&trace, format, output),
        "free_energy" => export_free_energy(&trace, format, output),
        "confidence" => export_confidence(&trace, format, output),
        "router" => export_router_selections(&trace, format, output),
        _ => bail!("Unknown metric: {}. Available: phi, free_energy, confidence, router", metric),
    }
}

fn export_phi(trace: &Trace, format: &str, output: Option<&Path>) -> Result<()> {
    let mut data = Vec::new();

    for event in &trace.events {
        if event.event_type == EventType::WorkspaceIgnition {
            if let Some(event_data) = &event.data {
                if let Ok(ws_data) = serde_json::from_value::<WorkspaceIgnitionData>(event_data.clone()) {
                    data.push((event.timestamp.clone(), ws_data.phi));
                }
            }
        }
    }

    match format {
        "csv" => export_csv(&data, &["timestamp", "phi"], output),
        "json" => export_json(&data, output),
        "jsonl" => export_jsonl(&data, output),
        _ => bail!("Unknown format: {}. Available: csv, json, jsonl", format),
    }
}

fn export_free_energy(trace: &Trace, format: &str, output: Option<&Path>) -> Result<()> {
    let mut data = Vec::new();

    for event in &trace.events {
        if event.event_type == EventType::WorkspaceIgnition {
            if let Some(event_data) = &event.data {
                if let Ok(ws_data) = serde_json::from_value::<WorkspaceIgnitionData>(event_data.clone()) {
                    data.push((event.timestamp.clone(), ws_data.free_energy));
                }
            }
        }
    }

    match format {
        "csv" => export_csv(&data, &["timestamp", "free_energy"], output),
        "json" => export_json(&data, output),
        "jsonl" => export_jsonl(&data, output),
        _ => bail!("Unknown format: {}. Available: csv, json, jsonl", format),
    }
}

fn export_confidence(trace: &Trace, format: &str, output: Option<&Path>) -> Result<()> {
    let mut data = Vec::new();

    for event in &trace.events {
        if event.event_type == EventType::RouterSelection {
            if let Some(event_data) = &event.data {
                if let Ok(router_data) = serde_json::from_value::<crate::trace::RouterSelectionData>(event_data.clone()) {
                    data.push((event.timestamp.clone(), router_data.confidence));
                }
            }
        }
    }

    match format {
        "csv" => export_csv(&data, &["timestamp", "confidence"], output),
        "json" => export_json(&data, output),
        "jsonl" => export_jsonl(&data, output),
        _ => bail!("Unknown format: {}. Available: csv, json, jsonl", format),
    }
}

fn export_router_selections(trace: &Trace, format: &str, output: Option<&Path>) -> Result<()> {
    let mut data = Vec::new();

    for event in &trace.events {
        if event.event_type == EventType::RouterSelection {
            if let Some(event_data) = &event.data {
                if let Ok(router_data) = serde_json::from_value::<crate::trace::RouterSelectionData>(event_data.clone()) {
                    data.push((
                        event.timestamp.clone(),
                        router_data.selected_router,
                        router_data.confidence,
                    ));
                }
            }
        }
    }

    match format {
        "csv" => {
            let rows: Vec<Vec<String>> = data.iter().map(|(ts, router, conf)| {
                vec![ts.clone(), router.clone(), conf.to_string()]
            }).collect();

            if let Some(path) = output {
                let mut wtr = csv::Writer::from_path(path)?;
                wtr.write_record(&["timestamp", "router", "confidence"])?;
                for row in rows {
                    wtr.write_record(&row)?;
                }
                wtr.flush()?;
            } else {
                let mut wtr = csv::Writer::from_writer(std::io::stdout());
                wtr.write_record(&["timestamp", "router", "confidence"])?;
                for row in rows {
                    wtr.write_record(&row)?;
                }
                wtr.flush()?;
            }
            Ok(())
        }
        "json" => export_json(&data, output),
        "jsonl" => export_jsonl(&data, output),
        _ => bail!("Unknown format: {}. Available: csv, json, jsonl", format),
    }
}

fn export_csv<T: serde::Serialize>(
    data: &[(String, T)],
    headers: &[&str],
    output: Option<&Path>,
) -> Result<()> {
    if let Some(path) = output {
        let mut wtr = csv::Writer::from_path(path)?;
        wtr.write_record(headers)?;

        for (timestamp, value) in data {
            wtr.write_record(&[
                timestamp.clone(),
                serde_json::to_string(value)?,
            ])?;
        }

        wtr.flush()?;
    } else {
        let mut wtr = csv::Writer::from_writer(std::io::stdout());
        wtr.write_record(headers)?;

        for (timestamp, value) in data {
            wtr.write_record(&[
                timestamp.clone(),
                serde_json::to_string(value)?,
            ])?;
        }

        wtr.flush()?;
    }

    Ok(())
}

fn export_json<T: serde::Serialize>(
    data: &T,
    output: Option<&Path>,
) -> Result<()> {
    let json = serde_json::to_string_pretty(data)?;

    if let Some(path) = output {
        std::fs::write(path, json)?;
    } else {
        println!("{}", json);
    }

    Ok(())
}

fn export_jsonl<T: serde::Serialize>(
    data: &[(String, T)],
    output: Option<&Path>,
) -> Result<()> {
    let lines: Vec<String> = data.iter()
        .map(|(ts, value)| {
            serde_json::json!({
                "timestamp": ts,
                "value": value,
            }).to_string()
        })
        .collect();

    let content = lines.join("\n");

    if let Some(path) = output {
        std::fs::write(path, content)?;
    } else {
        println!("{}", content);
    }

    Ok(())
}
