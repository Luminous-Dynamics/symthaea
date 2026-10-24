//! OTRF Security-Datasets (Mordor format) ingestion.
//!
//! The OTRF repo stores Windows attack samples as zipped JSON Lines files,
//! each line being a Splunk-style parsed Windows event. Fields mostly mirror
//! the Evtx System section (SourceName, Channel, Hostname, TimeCreated,
//! EventID, Level, Message) plus arbitrary Mordor extension fields.
//!
//! This is the **out-of-distribution corpus** for the supervised probe — a
//! probe trained on sbousseaden/EVTX-ATTACK-SAMPLES (native .evtx) tested
//! here measures cross-collection-methodology generalization.

use crate::event::{LogEvent, Severity, Source};
use crate::{LogParseError, Result};
use chrono::{DateTime, Utc};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Parse a single JSON Lines file from an unzipped OTRF sample.
pub fn parse_jsonl_file(path: &Path) -> Result<Vec<LogEvent>> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<serde_json::Value>(&line) {
            Ok(v) => {
                if let Some(ev) = json_to_log_event(&v) {
                    out.push(ev);
                }
            }
            Err(e) => eprintln!("otrf: skip bad json line: {e}"),
        }
    }
    Ok(out)
}

fn json_to_log_event(v: &serde_json::Value) -> Option<LogEvent> {
    let obj = v.as_object()?;

    let provider = obj
        .get("SourceName")
        .or_else(|| obj.get("ProviderName"))
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();

    let component = obj
        .get("Channel")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();

    let host = obj
        .get("Hostname")
        .or_else(|| obj.get("Computer"))
        .and_then(|x| x.as_str())
        .map(|s| s.to_string());

    // Never narrow a 64-bit input with `as u32`: an oversized EventID
    // could wrap into a legitimate security event ID (e.g. 4624).
    let event_id = obj
        .get("EventID")
        .and_then(parse_event_id)
        .unwrap_or(0);

    let level_str = obj.get("Level").and_then(|x| {
        x.as_str()
            .map(|s| s.to_string())
            .or_else(|| x.as_u64().map(|n| n.to_string()))
    });
    let severity = level_str
        .as_deref()
        .and_then(parse_level)
        .unwrap_or(Severity::Info);

    let timestamp = obj
        .get("TimeCreated")
        .or_else(|| obj.get("@timestamp"))
        .and_then(|t| t.as_str())
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(Utc::now);

    let message = obj
        .get("Message")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();

    // Collect the rest as fields. Skip the System-level keys we already
    // extracted to avoid double-encoding them.
    let skip_keys: &[&str] = &[
        "SourceName",
        "ProviderName",
        "Channel",
        "Hostname",
        "Computer",
        "EventID",
        "Level",
        "TimeCreated",
        "@timestamp",
        "Message",
    ];
    let mut fields = BTreeMap::new();
    for (k, val) in obj {
        if skip_keys.contains(&k.as_str()) {
            continue;
        }
        let sval = match val {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Null => continue,
            other => other.to_string(),
        };
        if sval.len() > 256 {
            // Mordor/Splunk include huge blobs; truncate for the encoder so
            // hash collisions don't obscure short-field structure. The helper
            // stops at a UTF-8 boundary instead of panicking on non-ASCII data.
            fields.insert(k.clone(), truncate_utf8(&sval, 256).to_string());
        } else {
            fields.insert(k.clone(), sval);
        }
    }

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

/// Parse an EventID without allowing oversized values to wrap into known IDs.
fn parse_event_id(value: &serde_json::Value) -> Option<u32> {
    if let Some(number) = value.as_u64() {
        return u32::try_from(number).ok();
    }
    value.as_str().and_then(|text| text.parse::<u32>().ok())
}

/// Return the longest prefix of `value` that fits within `max_bytes`
/// without splitting a UTF-8 code point.
fn truncate_utf8(value: &str, max_bytes: usize) -> &str {
    let mut end = value.len().min(max_bytes);
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

fn parse_level(s: &str) -> Option<Severity> {
    // OTRF uses both numeric strings ("4", "2") and a few name strings.
    if let Ok(n) = s.parse::<u32>() {
        return Some(match n {
            0 => Severity::Info,
            1 => Severity::Critical,
            2 => Severity::Error,
            3 => Severity::Warning,
            4 => Severity::Info,
            5 => Severity::Debug,
            _ => Severity::Info,
        });
    }
    match s.to_lowercase().as_str() {
        "critical" => Some(Severity::Critical),
        "error" => Some(Severity::Error),
        "warning" => Some(Severity::Warning),
        "info" | "informational" => Some(Severity::Info),
        "debug" | "verbose" => Some(Severity::Debug),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_truncation_preserves_utf8_boundaries() {
        // A 3-byte character makes byte offset 256 land inside a code point.
        let long = "漢".repeat(100);
        let value = serde_json::json!({
            "SourceName": "Microsoft-Windows-Security-Auditing",
            "Channel": "Security",
            "EventID": 4624,
            "LongField": long,
        });
        let event = json_to_log_event(&value).expect("object should parse");
        let truncated = &event.fields["LongField"];
        assert!(truncated.len() <= 256);
        assert_eq!(truncated.len(), 255);
        assert!(truncated.chars().all(|c| c == '漢'));
    }

    #[test]
    fn event_id_parsing_rejects_values_that_would_wrap() {
        let overflow = u32::MAX as u64 + 4624;
        assert_eq!(parse_event_id(&serde_json::json!(4624)), Some(4624));
        assert_eq!(parse_event_id(&serde_json::json!("4624")), Some(4624));
        assert_eq!(parse_event_id(&serde_json::json!(overflow)), None);
        assert_eq!(parse_event_id(&serde_json::json!(overflow.to_string())), None);

        for event_id in [
            serde_json::json!(overflow),
            serde_json::json!(overflow.to_string()),
        ] {
            let value = serde_json::json!({
                "SourceName": "Microsoft-Windows-Security-Auditing",
                "Channel": "Security",
                "EventID": event_id,
            });
            let event = json_to_log_event(&value).expect("object should parse");
            assert_eq!(event.event_id, 0, "invalid IDs fail closed to unknown");
        }
    }

    #[test]
    fn utf8_truncation_handles_short_and_zero_limits() {
        assert_eq!(truncate_utf8("security", 256), "security");
        assert_eq!(truncate_utf8("漢字", 0), "");
        assert_eq!(truncate_utf8("漢字", 4), "漢");
    }
}

/// Convenience wrapper: parse every `.json` file under a directory tree.
pub fn parse_jsonl_tree(root: &Path) -> Result<Vec<LogEvent>> {
    let mut out = Vec::new();
    visit(root, &mut out)?;
    Ok(out)
}

fn visit(dir: &Path, out: &mut Vec<LogEvent>) -> Result<()> {
    // read_dir order is filesystem-dependent. Sorting is necessary for
    // deterministic corpus construction and seeded training runs.
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<std::io::Result<_>>()?;
    paths.sort();

    for path in paths {
        if path.is_dir() {
            visit(&path, out)?;
        } else if path.extension().and_then(|s| s.to_str()) == Some("json") {
            match parse_jsonl_file(&path) {
                Ok(mut events) => out.append(&mut events),
                Err(e) => eprintln!("otrf: {}: {e}", path.display()),
            }
        }
    }
    Ok(())
}
