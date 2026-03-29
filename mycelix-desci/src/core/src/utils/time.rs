// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Time utilities
//!
//! Functions for time formatting, parsing, and relative time calculations

use chrono::{DateTime, Duration, Utc};
use crate::error::{Error, Result};

/// Format a timestamp as ISO 8601 / RFC 3339
pub fn format_iso8601(dt: &DateTime<Utc>) -> String {
    dt.to_rfc3339()
}

/// Format a timestamp as a human-readable string
pub fn format_human(dt: &DateTime<Utc>) -> String {
    dt.format("%Y-%m-%d %H:%M:%S UTC").to_string()
}

/// Format a timestamp in a compact format
pub fn format_compact(dt: &DateTime<Utc>) -> String {
    dt.format("%Y%m%d_%H%M%S").to_string()
}

/// Format a timestamp for display (date only)
pub fn format_date(dt: &DateTime<Utc>) -> String {
    dt.format("%Y-%m-%d").to_string()
}

/// Format a timestamp for display (time only)
pub fn format_time(dt: &DateTime<Utc>) -> String {
    dt.format("%H:%M:%S").to_string()
}

/// Parse an ISO 8601 / RFC 3339 timestamp
pub fn parse_iso8601(s: &str) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| Error::Validation(format!("Invalid ISO 8601 timestamp: {}", e)))
}

/// Get the current UTC timestamp
pub fn now() -> DateTime<Utc> {
    Utc::now()
}

/// Get the Unix epoch (1970-01-01 00:00:00 UTC)
pub fn epoch() -> DateTime<Utc> {
    DateTime::UNIX_EPOCH
}

/// Convert timestamp to Unix timestamp (seconds since epoch)
pub fn to_unix_timestamp(dt: &DateTime<Utc>) -> i64 {
    dt.timestamp()
}

/// Convert Unix timestamp to DateTime
pub fn from_unix_timestamp(timestamp: i64) -> DateTime<Utc> {
    DateTime::from_timestamp(timestamp, 0).unwrap_or(epoch())
}

/// Format duration in human-readable form
pub fn format_duration(duration: Duration) -> String {
    let total_seconds = duration.num_seconds().abs();

    if total_seconds == 0 {
        return "0 seconds".to_string();
    }

    let days = total_seconds / 86400;
    let hours = (total_seconds % 86400) / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;

    let mut parts = Vec::new();

    if days > 0 {
        parts.push(format!("{} day{}", days, if days == 1 { "" } else { "s" }));
    }
    if hours > 0 {
        parts.push(format!("{} hour{}", hours, if hours == 1 { "" } else { "s" }));
    }
    if minutes > 0 {
        parts.push(format!("{} minute{}", minutes, if minutes == 1 { "" } else { "s" }));
    }
    if seconds > 0 || parts.is_empty() {
        parts.push(format!("{} second{}", seconds, if seconds == 1 { "" } else { "s" }));
    }

    parts.join(", ")
}

/// Format relative time (e.g., "2 hours ago", "in 3 days")
pub fn format_relative(dt: &DateTime<Utc>) -> String {
    let now = Utc::now();
    let duration = now.signed_duration_since(*dt);

    format_relative_duration(duration)
}

/// Format a duration as relative time
pub fn format_relative_duration(duration: Duration) -> String {
    let total_seconds = duration.num_seconds();
    let abs_seconds = total_seconds.abs();

    let (value, unit) = if abs_seconds < 60 {
        (abs_seconds, "second")
    } else if abs_seconds < 3600 {
        (abs_seconds / 60, "minute")
    } else if abs_seconds < 86400 {
        (abs_seconds / 3600, "hour")
    } else if abs_seconds < 604800 {
        (abs_seconds / 86400, "day")
    } else if abs_seconds < 2592000 {
        (abs_seconds / 604800, "week")
    } else if abs_seconds < 31536000 {
        (abs_seconds / 2592000, "month")
    } else {
        (abs_seconds / 31536000, "year")
    };

    let plural = if value == 1 { "" } else { "s" };

    if total_seconds < 0 {
        format!("in {} {}{}", value, unit, plural)
    } else if total_seconds < 5 {
        "just now".to_string()
    } else {
        format!("{} {}{} ago", value, unit, plural)
    }
}

/// Calculate time elapsed since a timestamp
pub fn elapsed_since(dt: &DateTime<Utc>) -> Duration {
    Utc::now().signed_duration_since(*dt)
}

/// Calculate time until a timestamp
pub fn time_until(dt: &DateTime<Utc>) -> Duration {
    dt.signed_duration_since(Utc::now())
}

/// Check if a timestamp is in the past
pub fn is_past(dt: &DateTime<Utc>) -> bool {
    *dt < Utc::now()
}

/// Check if a timestamp is in the future
pub fn is_future(dt: &DateTime<Utc>) -> bool {
    *dt > Utc::now()
}

/// Check if a timestamp is within a given range
pub fn is_within_range(
    dt: &DateTime<Utc>,
    start: &DateTime<Utc>,
    end: &DateTime<Utc>,
) -> bool {
    dt >= start && dt <= end
}

/// Add duration to a timestamp
pub fn add_duration(dt: &DateTime<Utc>, duration: Duration) -> DateTime<Utc> {
    *dt + duration
}

/// Subtract duration from a timestamp
pub fn subtract_duration(dt: &DateTime<Utc>, duration: Duration) -> DateTime<Utc> {
    *dt - duration
}

/// Create a duration from seconds
pub fn seconds(secs: i64) -> Duration {
    Duration::seconds(secs)
}

/// Create a duration from minutes
pub fn minutes(mins: i64) -> Duration {
    Duration::minutes(mins)
}

/// Create a duration from hours
pub fn hours(hrs: i64) -> Duration {
    Duration::hours(hrs)
}

/// Create a duration from days
pub fn days(days: i64) -> Duration {
    Duration::days(days)
}

/// Create a duration from weeks
pub fn weeks(weeks: i64) -> Duration {
    Duration::weeks(weeks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_iso8601() {
        let dt = DateTime::from_timestamp(1234567890, 0).unwrap();
        let formatted = format_iso8601(&dt);
        assert!(formatted.contains("2009-02-13"));
    }

    #[test]
    fn test_format_human() {
        let dt = DateTime::from_timestamp(1234567890, 0).unwrap();
        let formatted = format_human(&dt);
        assert!(formatted.contains("2009-02-13"));
        assert!(formatted.contains("UTC"));
    }

    #[test]
    fn test_format_compact() {
        let dt = DateTime::from_timestamp(1234567890, 0).unwrap();
        let formatted = format_compact(&dt);
        assert!(formatted.contains("20090213"));
    }

    #[test]
    fn test_format_date() {
        let dt = DateTime::from_timestamp(1234567890, 0).unwrap();
        assert_eq!(format_date(&dt), "2009-02-13");
    }

    #[test]
    fn test_format_time() {
        let dt = DateTime::from_timestamp(1234567890, 0).unwrap();
        let formatted = format_time(&dt);
        assert!(formatted.contains(":"));
    }

    #[test]
    fn test_parse_iso8601_valid() {
        let iso = "2009-02-13T23:31:30Z";
        let dt = parse_iso8601(iso).unwrap();
        assert_eq!(dt.timestamp(), 1234567890);
    }

    #[test]
    fn test_parse_iso8601_invalid() {
        assert!(parse_iso8601("not a date").is_err());
    }

    #[test]
    fn test_unix_timestamp_conversion() {
        let timestamp = 1234567890;
        let dt = from_unix_timestamp(timestamp);
        assert_eq!(to_unix_timestamp(&dt), timestamp);
    }

    #[test]
    fn test_format_duration_seconds() {
        let duration = Duration::seconds(45);
        assert_eq!(format_duration(duration), "45 seconds");
    }

    #[test]
    fn test_format_duration_minutes() {
        let duration = Duration::minutes(5);
        assert_eq!(format_duration(duration), "5 minutes");
    }

    #[test]
    fn test_format_duration_hours() {
        let duration = Duration::hours(2);
        assert_eq!(format_duration(duration), "2 hours");
    }

    #[test]
    fn test_format_duration_days() {
        let duration = Duration::days(3);
        assert_eq!(format_duration(duration), "3 days");
    }

    #[test]
    fn test_format_duration_mixed() {
        let duration = Duration::days(1) + Duration::hours(2) + Duration::minutes(30);
        let formatted = format_duration(duration);
        assert!(formatted.contains("1 day"));
        assert!(formatted.contains("2 hours"));
        assert!(formatted.contains("30 minutes"));
    }

    #[test]
    fn test_format_duration_singular() {
        let duration = Duration::days(1) + Duration::hours(1) + Duration::minutes(1) + Duration::seconds(1);
        let formatted = format_duration(duration);
        assert!(formatted.contains("1 day,"));
        assert!(formatted.contains("1 hour,"));
        assert!(formatted.contains("1 minute,"));
        assert!(formatted.contains("1 second"));
        assert!(!formatted.contains("days"));
        assert!(!formatted.contains("hours"));
        assert!(!formatted.contains("minutes"));
        assert!(!formatted.contains("seconds"));
    }

    #[test]
    fn test_format_relative_past() {
        let past = Utc::now() - Duration::hours(2);
        let formatted = format_relative(&past);
        assert!(formatted.contains("ago") || formatted == "just now");
    }

    #[test]
    fn test_format_relative_future() {
        let future = Utc::now() + Duration::hours(2);
        let formatted = format_relative(&future);
        assert!(formatted.contains("in"));
    }

    #[test]
    fn test_format_relative_just_now() {
        let now = Utc::now();
        let formatted = format_relative(&now);
        assert_eq!(formatted, "just now");
    }

    #[test]
    fn test_format_relative_duration_seconds() {
        let duration = Duration::seconds(30);
        let formatted = format_relative_duration(duration);
        assert!(formatted.contains("30 seconds ago"));
    }

    #[test]
    fn test_format_relative_duration_minutes() {
        let duration = Duration::minutes(15);
        let formatted = format_relative_duration(duration);
        assert!(formatted.contains("15 minutes ago"));
    }

    #[test]
    fn test_format_relative_duration_hours() {
        let duration = Duration::hours(3);
        let formatted = format_relative_duration(duration);
        assert!(formatted.contains("3 hours ago"));
    }

    #[test]
    fn test_format_relative_duration_days() {
        let duration = Duration::days(5);
        let formatted = format_relative_duration(duration);
        assert!(formatted.contains("5 days ago"));
    }

    #[test]
    fn test_format_relative_duration_future() {
        let duration = Duration::hours(-2);
        let formatted = format_relative_duration(duration);
        assert!(formatted.contains("in 2 hours"));
    }

    #[test]
    fn test_is_past() {
        let past = Utc::now() - Duration::hours(1);
        assert!(is_past(&past));

        let future = Utc::now() + Duration::hours(1);
        assert!(!is_past(&future));
    }

    #[test]
    fn test_is_future() {
        let future = Utc::now() + Duration::hours(1);
        assert!(is_future(&future));

        let past = Utc::now() - Duration::hours(1);
        assert!(!is_future(&past));
    }

    #[test]
    fn test_is_within_range() {
        let start = Utc::now() - Duration::hours(2);
        let end = Utc::now() + Duration::hours(2);
        let middle = Utc::now();

        assert!(is_within_range(&middle, &start, &end));

        let before = Utc::now() - Duration::hours(3);
        assert!(!is_within_range(&before, &start, &end));

        let after = Utc::now() + Duration::hours(3);
        assert!(!is_within_range(&after, &start, &end));
    }

    #[test]
    fn test_add_duration() {
        let dt = Utc::now();
        let duration = Duration::hours(2);
        let result = add_duration(&dt, duration);
        assert_eq!(result.timestamp(), dt.timestamp() + 7200);
    }

    #[test]
    fn test_subtract_duration() {
        let dt = Utc::now();
        let duration = Duration::hours(2);
        let result = subtract_duration(&dt, duration);
        assert_eq!(result.timestamp(), dt.timestamp() - 7200);
    }

    #[test]
    fn test_duration_helpers() {
        assert_eq!(seconds(60).num_seconds(), 60);
        assert_eq!(minutes(5).num_minutes(), 5);
        assert_eq!(hours(2).num_hours(), 2);
        assert_eq!(days(7).num_days(), 7);
        assert_eq!(weeks(2).num_weeks(), 2);
    }

    #[test]
    fn test_elapsed_since() {
        let past = Utc::now() - Duration::seconds(100);
        let elapsed = elapsed_since(&past);
        // Allow for small timing variations
        assert!(elapsed.num_seconds() >= 99 && elapsed.num_seconds() <= 101);
    }

    #[test]
    fn test_time_until() {
        let future = Utc::now() + Duration::seconds(100);
        let until = time_until(&future);
        // Allow for small timing variations
        assert!(until.num_seconds() >= 99 && until.num_seconds() <= 101);
    }
}
