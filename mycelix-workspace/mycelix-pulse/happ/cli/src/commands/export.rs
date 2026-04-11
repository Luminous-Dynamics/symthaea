// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use std::fs::File;
use std::io::Write;

use crate::client::MycellixClient;
use crate::types::MailMessage;

/// Export messages to file in various formats
pub async fn handle_export(
    client: &MycellixClient,
    format: &str,
    output: String,
    since: Option<String>,
) -> Result<()> {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("                      MESSAGE EXPORT");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("📦 Output File:  {}", output);
    println!("📄 Format:       {}", format_name(format));
    if let Some(ref date) = since {
        println!("📅 Filter:       Messages since {}", date);
    }
    println!();

    // 1. Fetch all messages
    println!("📬 Fetching inbox messages...");
    let inbox_messages = client
        .get_inbox()
        .await
        .context("Failed to fetch inbox messages")?;

    println!("📭 Fetching sent messages...");
    let sent_messages = client
        .get_sent()
        .await
        .context("Failed to fetch sent messages")?;

    // 2. Combine messages
    let mut all_messages = inbox_messages;
    all_messages.extend(sent_messages);

    println!();
    println!("📊 Found {} total message(s)", all_messages.len());

    // 3. Apply date filter if specified
    if let Some(since_date) = since {
        let filter_timestamp = parse_date(&since_date)?;
        let before_count = all_messages.len();
        all_messages.retain(|msg| msg.timestamp >= filter_timestamp);
        let after_count = all_messages.len();
        println!(
            "🔍 Filtered to {} message(s) since {}",
            after_count, since_date
        );
        if before_count > after_count {
            println!(
                "   (Excluded {} older message(s))",
                before_count - after_count
            );
        }
    }

    // Handle no messages to export
    if all_messages.is_empty() {
        println!();
        println!("⚠️  No messages to export!");
        println!();
        println!("💡 Tips:");
        println!("   • Check if you have any messages in your inbox or sent folder");
        println!("   • Try removing the --since filter to export all messages");
        return Ok(());
    }

    println!();
    println!(
        "💾 Writing {} message(s) to {}...",
        all_messages.len(),
        output
    );

    // 4. Export to file
    match format {
        "json" => export_json(&all_messages, &output)?,
        "mbox" => export_mbox(&all_messages, &output)?,
        "csv" => export_csv(&all_messages, &output)?,
        _ => bail!("Unsupported export format: {}", format),
    }

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("                     EXPORT COMPLETE");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("✅ Successfully exported {} message(s)", all_messages.len());
    println!("📁 File: {}", output);
    println!("📊 Format: {}", format_name(format));
    println!();
    println!("💡 You can now open this file with your preferred application");

    Ok(())
}

/// Export messages to JSON format
fn export_json(messages: &[MailMessage], output: &str) -> Result<()> {
    let json =
        serde_json::to_string_pretty(messages).context("Failed to serialize messages to JSON")?;

    let mut file =
        File::create(output).context(format!("Failed to create output file: {}", output))?;

    file.write_all(json.as_bytes())
        .context("Failed to write JSON data")?;

    Ok(())
}

/// Export messages to MBOX format (standard Unix mailbox format)
fn export_mbox(messages: &[MailMessage], output: &str) -> Result<()> {
    let mut file =
        File::create(output).context(format!("Failed to create output file: {}", output))?;

    for msg in messages {
        // MBOX format starts each message with "From " line
        let from_line = format!(
            "From {} {}\n",
            msg.from_did,
            format_timestamp_mbox(msg.timestamp)
        );
        file.write_all(from_line.as_bytes())
            .context("Failed to write MBOX from line")?;

        // Headers
        let headers = format!(
            "From: {}\nTo: {}\nDate: {}\nSubject: {}\n",
            msg.from_did,
            msg.to_did,
            format_timestamp_rfc2822(msg.timestamp),
            decrypt_subject(&msg.subject_encrypted)
        );
        file.write_all(headers.as_bytes())
            .context("Failed to write MBOX headers")?;

        // Body (placeholder - would fetch from CID in real implementation)
        let body = format!("\n[Body CID: {}]\n\n", msg.body_cid);
        file.write_all(body.as_bytes())
            .context("Failed to write MBOX body")?;
    }

    Ok(())
}

/// Export messages to CSV format
fn export_csv(messages: &[MailMessage], output: &str) -> Result<()> {
    let mut file =
        File::create(output).context(format!("Failed to create output file: {}", output))?;

    // CSV header
    let header = "Timestamp,Date,From,To,Subject,BodyCID,Tier,ThreadID\n";
    file.write_all(header.as_bytes())
        .context("Failed to write CSV header")?;

    // CSV rows
    for msg in messages {
        let subject = decrypt_subject(&msg.subject_encrypted);
        let thread_id = msg.thread_id.as_deref().unwrap_or("");
        let tier = format!("{:?}", msg.epistemic_tier);

        let row = format!(
            "{},{},{},{},{},{},{},{}\n",
            msg.timestamp,
            format_timestamp_iso8601(msg.timestamp),
            escape_csv(&msg.from_did),
            escape_csv(&msg.to_did),
            escape_csv(&subject),
            escape_csv(&msg.body_cid),
            tier,
            escape_csv(thread_id)
        );
        file.write_all(row.as_bytes())
            .context("Failed to write CSV row")?;
    }

    Ok(())
}

/// Parse date string to Unix timestamp
///
/// Supports multiple date formats:
/// - Unix timestamp (e.g., "1609459200")
/// - ISO 8601 / RFC 3339 (e.g., "2021-01-01T00:00:00Z")
/// - Date only (e.g., "2021-01-01")
/// - Date with time (e.g., "2021-01-01 12:30:00")
/// - Relative formats (e.g., "today", "yesterday", "7d", "1w", "1m")
fn parse_date(date_str: &str) -> Result<i64> {
    let date_str = date_str.trim();

    // Try parsing as Unix timestamp first
    if let Ok(ts) = date_str.parse::<i64>() {
        return Ok(ts);
    }

    // Handle relative date formats
    let now = Utc::now();
    match date_str.to_lowercase().as_str() {
        "today" => {
            let today = now.date_naive().and_hms_opt(0, 0, 0).unwrap();
            return Ok(DateTime::<Utc>::from_naive_utc_and_offset(today, Utc).timestamp());
        }
        "yesterday" => {
            let yesterday = (now - chrono::Duration::days(1)).date_naive().and_hms_opt(0, 0, 0).unwrap();
            return Ok(DateTime::<Utc>::from_naive_utc_and_offset(yesterday, Utc).timestamp());
        }
        _ => {}
    }

    // Handle relative duration formats (e.g., "7d", "1w", "1m", "2h")
    if date_str.len() >= 2 {
        let (num_str, unit) = date_str.split_at(date_str.len() - 1);
        if let Ok(num) = num_str.parse::<i64>() {
            let duration = match unit.to_lowercase().as_str() {
                "h" => Some(chrono::Duration::hours(num)),
                "d" => Some(chrono::Duration::days(num)),
                "w" => Some(chrono::Duration::weeks(num)),
                "m" => Some(chrono::Duration::days(num * 30)), // Approximate month
                _ => None,
            };
            if let Some(dur) = duration {
                return Ok((now - dur).timestamp());
            }
        }
    }

    // Try parsing as RFC 3339 / ISO 8601 with timezone
    if let Ok(dt) = DateTime::parse_from_rfc3339(date_str) {
        return Ok(dt.with_timezone(&Utc).timestamp());
    }

    // Try parsing as ISO 8601 date-time without timezone (assume UTC)
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(date_str, "%Y-%m-%dT%H:%M:%S") {
        return Ok(DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc).timestamp());
    }

    // Try parsing as date with space-separated time
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(date_str, "%Y-%m-%d %H:%M:%S") {
        return Ok(DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc).timestamp());
    }

    // Try parsing as date only (start of day)
    if let Ok(date) = chrono::NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
        let dt = date.and_hms_opt(0, 0, 0).unwrap();
        return Ok(DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc).timestamp());
    }

    bail!(
        "Failed to parse date: '{}'. Supported formats:\n  \
        - Unix timestamp (e.g., 1609459200)\n  \
        - ISO 8601 (e.g., 2021-01-01T00:00:00Z)\n  \
        - Date only (e.g., 2021-01-01)\n  \
        - Date with time (e.g., 2021-01-01 12:30:00)\n  \
        - Relative (e.g., today, yesterday, 7d, 1w, 1m)",
        date_str
    )
}

/// Format timestamp for MBOX "From " line
fn format_timestamp_mbox(ts: i64) -> String {
    let dt = DateTime::<Utc>::from_timestamp(ts, 0).unwrap_or_else(|| Utc::now());
    dt.format("%a %b %d %H:%M:%S %Y").to_string()
}

/// Format timestamp as RFC 2822 (for email headers)
fn format_timestamp_rfc2822(ts: i64) -> String {
    let dt = DateTime::<Utc>::from_timestamp(ts, 0).unwrap_or_else(|| Utc::now());
    dt.to_rfc2822()
}

/// Format timestamp as ISO 8601 (for CSV)
fn format_timestamp_iso8601(ts: i64) -> String {
    let dt = DateTime::<Utc>::from_timestamp(ts, 0).unwrap_or_else(|| Utc::now());
    dt.to_rfc3339()
}

/// Escape CSV field (add quotes if contains comma, newline, or quote)
fn escape_csv(field: &str) -> String {
    if field.contains(',') || field.contains('\n') || field.contains('"') {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

/// Format name for display
fn format_name(format: &str) -> &str {
    match format {
        "json" => "JSON (Machine-readable)",
        "mbox" => "MBOX (Standard Unix mailbox)",
        "csv" => "CSV (Spreadsheet-friendly)",
        _ => format,
    }
}

/// Decrypt subject (placeholder implementation)
fn decrypt_subject(encrypted: &[u8]) -> String {
    if let Ok(s) = String::from_utf8(encrypted.to_vec()) {
        if s.starts_with("ENC:") {
            s[4..].to_string()
        } else {
            s
        }
    } else {
        "<encrypted>".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::EpistemicTier;

    #[test]
    fn test_escape_csv() {
        assert_eq!(escape_csv("simple"), "simple");
        assert_eq!(escape_csv("has,comma"), "\"has,comma\"");
        assert_eq!(escape_csv("has\"quote"), "\"has\"\"quote\"");
        assert_eq!(escape_csv("has\nnewline"), "\"has\nnewline\"");
    }

    #[test]
    fn test_format_name() {
        assert_eq!(format_name("json"), "JSON (Machine-readable)");
        assert_eq!(format_name("mbox"), "MBOX (Standard Unix mailbox)");
        assert_eq!(format_name("csv"), "CSV (Spreadsheet-friendly)");
    }

    #[test]
    fn test_format_timestamp_iso8601() {
        let ts = 1609459200; // 2021-01-01 00:00:00 UTC
        let formatted = format_timestamp_iso8601(ts);
        assert!(formatted.contains("2021"));
        assert!(formatted.contains("T"));
    }

    #[test]
    fn test_format_timestamp_rfc2822() {
        let ts = 1609459200; // 2021-01-01 00:00:00 UTC
        let formatted = format_timestamp_rfc2822(ts);
        assert!(formatted.contains("2021"));
        assert!(formatted.contains("GMT") || formatted.contains("+0000"));
    }

    #[test]
    fn test_decrypt_subject() {
        let encrypted = b"ENC:Test Subject";
        let decrypted = decrypt_subject(encrypted);
        assert_eq!(decrypted, "Test Subject");

        let plain = b"Plain Subject";
        let result = decrypt_subject(plain);
        assert_eq!(result, "Plain Subject");
    }

    #[test]
    fn test_parse_date_unix_timestamp() {
        let result = parse_date("1609459200");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1609459200);
    }

    #[test]
    fn test_parse_date_iso8601() {
        // ISO 8601 with timezone
        let result = parse_date("2021-01-01T00:00:00Z");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1609459200);

        // ISO 8601 without timezone (assume UTC)
        let result = parse_date("2021-01-01T12:30:00");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1609504200);
    }

    #[test]
    fn test_parse_date_date_only() {
        let result = parse_date("2021-01-01");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1609459200);
    }

    #[test]
    fn test_parse_date_with_time() {
        let result = parse_date("2021-01-01 12:30:00");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1609504200);
    }

    #[test]
    fn test_parse_date_relative() {
        // These tests verify parsing works, but exact values depend on current time
        let result = parse_date("today");
        assert!(result.is_ok());

        let result = parse_date("yesterday");
        assert!(result.is_ok());

        let result = parse_date("7d");
        assert!(result.is_ok());

        let result = parse_date("1w");
        assert!(result.is_ok());

        let result = parse_date("1m");
        assert!(result.is_ok());

        let result = parse_date("24h");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_date_invalid() {
        let result = parse_date("invalid");
        assert!(result.is_err());

        let result = parse_date("not-a-date");
        assert!(result.is_err());
    }
}
