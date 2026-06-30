//! Windows Event Log (.evtx) ingestion via the `evtx` crate.
//!
//! This runs on Linux — we parse offline corpora (DFIR.training, public
//! incident samples) during the Phase 1 spike. Live ETW ingestion is a Phase 2+
//! concern and will require a Windows-side collector agent.

use crate::event::{LogEvent, Severity, Source};
use crate::{LogParseError, Result};
use chrono::{DateTime, Utc};
use std::collections::BTreeMap;
use std::path::Path;

/// Parse an .evtx file into a vector of normalized `LogEvent`s.
///
/// Returns events in file order. Errors on individual records are logged to
/// stderr and skipped — a corrupt record should not kill the whole corpus run.
pub fn parse_evtx_file(path: &Path) -> Result<Vec<LogEvent>> {
    let mut parser =
        evtx::EvtxParser::from_path(path).map_err(|e| LogParseError::Evtx(e.to_string()))?;

    let mut out = Vec::new();
    for record in parser.records_json_value() {
        match record {
            Ok(rec) => {
                if let Some(ev) = record_to_log_event(&rec.data) {
                    out.push(ev);
                }
            }
            Err(e) => {
                eprintln!("evtx: skipping bad record: {e}");
            }
        }
    }
    Ok(out)
}

/// Convert a parsed Evtx JSON record into our normalized shape.
///
/// The shape of `data` is what the `evtx` crate produces: a JSON tree mirroring
/// the Event XML (System, EventData, RenderingInfo). We pick out the fields
/// Phase 1 clustering actually needs and drop the rest.
fn record_to_log_event(data: &serde_json::Value) -> Option<LogEvent> {
    let event = data.get("Event")?;
    let system = event.get("System")?;

    let provider = system
        .get("Provider")
        .and_then(|p| p.get("#attributes"))
        .and_then(|a| a.get("Name"))
        .and_then(|n| n.as_str())
        .unwrap_or("")
        .to_string();

    let event_id = system.get("EventID").and_then(extract_u32).unwrap_or(0);

    let level = system.get("Level").and_then(extract_u32).unwrap_or(4);
    let severity = evtx_level_to_severity(level);

    let timestamp = system
        .get("TimeCreated")
        .and_then(|t| t.get("#attributes"))
        .and_then(|a| a.get("SystemTime"))
        .and_then(|s| s.as_str())
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(Utc::now);

    let component = system
        .get("Channel")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();

    let host = system
        .get("Computer")
        .and_then(|c| c.as_str())
        .map(|s| s.to_string());

    let mut fields = BTreeMap::new();
    if let Some(event_data) = event.get("EventData") {
        flatten_event_data(event_data, "", &mut fields);
    }

    // Synthesize a message from the top few fields. Evtx records often have
    // no rendered message unless the provider manifest is registered.
    let message = fields
        .iter()
        .take(3)
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(" ");

    Some(LogEvent {
        timestamp,
        source: Source::WindowsEvent,
        severity,
        component,
        provider,
        event_id,
        message,
        fields,
        host,
        label: None,
    })
}

fn extract_u32(v: &serde_json::Value) -> Option<u32> {
    if let Some(n) = v.as_u64() {
        return Some(n as u32);
    }
    if let Some(obj) = v.as_object() {
        if let Some(inner) = obj.get("#text").and_then(|x| x.as_u64()) {
            return Some(inner as u32);
        }
    }
    v.as_str().and_then(|s| s.parse().ok())
}

fn flatten_event_data(v: &serde_json::Value, prefix: &str, out: &mut BTreeMap<String, String>) {
    match v {
        serde_json::Value::Object(m) => {
            for (k, val) in m {
                let key = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                flatten_event_data(val, &key, out);
            }
        }
        serde_json::Value::Array(a) => {
            for (i, val) in a.iter().enumerate() {
                flatten_event_data(val, &format!("{prefix}[{i}]"), out);
            }
        }
        serde_json::Value::String(s) => {
            out.insert(prefix.to_string(), s.clone());
        }
        other => {
            out.insert(prefix.to_string(), other.to_string());
        }
    }
}

/// Map Evtx numeric Level (0-5) to our Severity enum.
/// 0 = LogAlways, 1 = Critical, 2 = Error, 3 = Warning, 4 = Info, 5 = Verbose
fn evtx_level_to_severity(level: u32) -> Severity {
    match level {
        0 => Severity::Info,
        1 => Severity::Critical,
        2 => Severity::Error,
        3 => Severity::Warning,
        4 => Severity::Info,
        5 => Severity::Debug,
        _ => Severity::Info,
    }
}
