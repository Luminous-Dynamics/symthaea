//! Syslog ingestion via `syslog_loose` (handles both RFC5424 and RFC3164).
//!
//! Input is a path to a newline-delimited log file. Streaming ingestion is a
//! Phase 2+ concern — the spike only needs offline corpus parsing.

use crate::Result;
use crate::event::{LogEvent, Severity, Source};
use chrono::Utc;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use syslog_loose::{SyslogSeverity, Variant};

pub fn parse_syslog_file(path: &Path) -> Result<Vec<LogEvent>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        out.push(parse_syslog_line(&line));
    }
    Ok(out)
}

pub fn parse_syslog_line(line: &str) -> LogEvent {
    let msg = syslog_loose::parse_message(line, Variant::Either);

    let severity = msg.severity.map(map_severity).unwrap_or(Severity::Info);

    let timestamp = msg
        .timestamp
        .map(|t| t.with_timezone(&Utc))
        .unwrap_or_else(Utc::now);

    let component = msg.appname.map(|s| s.to_string()).unwrap_or_default();
    let provider = msg
        .facility
        .map(|f| f.as_str().to_string())
        .unwrap_or_default();
    let host = msg.hostname.map(|h| h.to_string());

    let mut fields = BTreeMap::new();
    for sd in &msg.structured_data {
        for (k, v) in &sd.params {
            fields.insert(format!("{}.{}", sd.id, k), v.to_string());
        }
    }
    if let Some(procid) = &msg.procid {
        fields.insert("procid".to_string(), format!("{procid}"));
    }
    if let Some(msgid) = msg.msgid {
        fields.insert("msgid".to_string(), msgid.to_string());
    }

    LogEvent {
        timestamp,
        source: Source::Syslog,
        severity,
        component,
        provider,
        event_id: 0,
        message: msg.msg.to_string(),
        fields,
        host,
        label: None,
    }
}

fn map_severity(s: SyslogSeverity) -> Severity {
    match s {
        SyslogSeverity::SEV_EMERG => Severity::Emergency,
        SyslogSeverity::SEV_ALERT => Severity::Alert,
        SyslogSeverity::SEV_CRIT => Severity::Critical,
        SyslogSeverity::SEV_ERR => Severity::Error,
        SyslogSeverity::SEV_WARNING => Severity::Warning,
        SyslogSeverity::SEV_NOTICE => Severity::Notice,
        SyslogSeverity::SEV_INFO => Severity::Info,
        SyslogSeverity::SEV_DEBUG => Severity::Debug,
    }
}
