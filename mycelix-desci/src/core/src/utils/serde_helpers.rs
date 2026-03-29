// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Serialization helpers
//!
//! Custom serializers, deserializers, and formatting utilities

use crate::error::{Error, Result};
use serde::{de::DeserializeOwned, Serialize};
use std::path::Path;

/// Serialize to JSON with pretty printing
pub fn to_json_pretty<T: Serialize>(value: &T) -> Result<String> {
    serde_json::to_string_pretty(value)
        .map_err(|e| Error::SerializationError(format!("JSON serialization failed: {}", e)))
}

/// Serialize to compact JSON
pub fn to_json<T: Serialize>(value: &T) -> Result<String> {
    serde_json::to_string(value)
        .map_err(|e| Error::SerializationError(format!("JSON serialization failed: {}", e)))
}

/// Deserialize from JSON string
pub fn from_json<T: DeserializeOwned>(json: &str) -> Result<T> {
    serde_json::from_str(json)
        .map_err(|e| Error::SerializationError(format!("JSON deserialization failed: {}", e)))
}

/// Serialize to JSON and write to file
pub fn to_json_file<T: Serialize, P: AsRef<Path>>(value: &T, path: P) -> Result<()> {
    let json = to_json_pretty(value)?;
    std::fs::write(&path, json).map_err(|e| {
        Error::IoError(format!(
            "Failed to write JSON to {}: {}",
            path.as_ref().display(),
            e
        ))
    })
}

/// Read and deserialize JSON from file
pub fn from_json_file<T: DeserializeOwned, P: AsRef<Path>>(path: P) -> Result<T> {
    let content = std::fs::read_to_string(&path).map_err(|e| {
        Error::IoError(format!(
            "Failed to read JSON from {}: {}",
            path.as_ref().display(),
            e
        ))
    })?;
    from_json(&content)
}

/// Serialize to binary using bincode
pub fn to_binary<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    bincode::serialize(value)
        .map_err(|e| Error::SerializationError(format!("Binary serialization failed: {}", e)))
}

/// Deserialize from binary using bincode
pub fn from_binary<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    bincode::deserialize(bytes)
        .map_err(|e| Error::SerializationError(format!("Binary deserialization failed: {}", e)))
}

/// Serialize to binary and write to file
pub fn to_binary_file<T: Serialize, P: AsRef<Path>>(value: &T, path: P) -> Result<()> {
    let bytes = to_binary(value)?;
    std::fs::write(&path, bytes).map_err(|e| {
        Error::IoError(format!(
            "Failed to write binary to {}: {}",
            path.as_ref().display(),
            e
        ))
    })
}

/// Read and deserialize binary from file
pub fn from_binary_file<T: DeserializeOwned, P: AsRef<Path>>(path: P) -> Result<T> {
    let bytes = std::fs::read(&path).map_err(|e| {
        Error::IoError(format!(
            "Failed to read binary from {}: {}",
            path.as_ref().display(),
            e
        ))
    })?;
    from_binary(&bytes)
}

/// Format bytes as human-readable size
pub fn format_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB", "PB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    if unit_idx == 0 {
        format!("{} {}", bytes, UNITS[0])
    } else {
        format!("{:.2} {}", size, UNITS[unit_idx])
    }
}

/// Format a vector as a comma-separated list
pub fn format_list<T: std::fmt::Display>(items: &[T]) -> String {
    items
        .iter()
        .map(|item| item.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Format a vector as a bullet list
pub fn format_bullet_list<T: std::fmt::Display>(items: &[T]) -> String {
    items
        .iter()
        .map(|item| format!("• {}", item))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Truncate string with ellipsis
pub fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else if max_len <= 3 {
        "...".to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

/// Custom serializer for Option<String> that skips None values
pub mod option_string {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &Option<String>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match value {
            Some(s) => serializer.serialize_some(s),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Option::<String>::deserialize(deserializer)
    }
}

/// Custom serializer for DateTime that uses RFC3339 format
pub mod datetime_rfc3339 {
    use chrono::{DateTime, Utc};
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(dt: &DateTime<Utc>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&dt.to_rfc3339())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<DateTime<Utc>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        DateTime::parse_from_rfc3339(&s)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(serde::de::Error::custom)
    }
}

/// Custom serializer for Option<DateTime> using RFC3339
pub mod option_datetime_rfc3339 {
    use chrono::{DateTime, Utc};
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(dt: &Option<DateTime<Utc>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match dt {
            Some(dt) => serializer.serialize_str(&dt.to_rfc3339()),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<DateTime<Utc>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Option::<String>::deserialize(deserializer)?
            .map(|s| {
                DateTime::parse_from_rfc3339(&s)
                    .map(|dt| dt.with_timezone(&Utc))
                    .map_err(serde::de::Error::custom)
            })
            .transpose()
    }
}

/// Custom serializer for Vec<u8> as hex string
pub mod hex_bytes {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&hex::encode(bytes))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        hex::decode(&s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use tempfile::NamedTempFile;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct TestData {
        name: String,
        value: i32,
    }

    #[test]
    fn test_json_serialization() {
        let data = TestData {
            name: "test".to_string(),
            value: 42,
        };

        let json = to_json(&data).unwrap();
        assert!(json.contains("test"));
        assert!(json.contains("42"));
    }

    #[test]
    fn test_json_pretty_serialization() {
        let data = TestData {
            name: "test".to_string(),
            value: 42,
        };

        let json = to_json_pretty(&data).unwrap();
        assert!(json.contains('\n')); // Pretty printed has newlines
        assert!(json.contains("test"));
    }

    #[test]
    fn test_json_deserialization() {
        let json = r#"{"name":"test","value":42}"#;
        let data: TestData = from_json(json).unwrap();

        assert_eq!(data.name, "test");
        assert_eq!(data.value, 42);
    }

    #[test]
    fn test_json_roundtrip() {
        let original = TestData {
            name: "roundtrip".to_string(),
            value: 123,
        };

        let json = to_json(&original).unwrap();
        let restored: TestData = from_json(&json).unwrap();

        assert_eq!(original, restored);
    }

    #[test]
    fn test_json_file_write_read() {
        let data = TestData {
            name: "file_test".to_string(),
            value: 999,
        };

        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Write
        to_json_file(&data, path).unwrap();

        // Read
        let restored: TestData = from_json_file(path).unwrap();
        assert_eq!(data, restored);
    }

    #[test]
    fn test_binary_serialization() {
        let data = TestData {
            name: "binary".to_string(),
            value: 42,
        };

        let bytes = to_binary(&data).unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_binary_deserialization() {
        let data = TestData {
            name: "binary".to_string(),
            value: 42,
        };

        let bytes = to_binary(&data).unwrap();
        let restored: TestData = from_binary(&bytes).unwrap();

        assert_eq!(data, restored);
    }

    #[test]
    fn test_binary_file_write_read() {
        let data = TestData {
            name: "binary_file".to_string(),
            value: 777,
        };

        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Write
        to_binary_file(&data, path).unwrap();

        // Read
        let restored: TestData = from_binary_file(path).unwrap();
        assert_eq!(data, restored);
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(500), "500 B");
        assert_eq!(format_size(1024), "1.00 KB");
        assert_eq!(format_size(1536), "1.50 KB");
        assert_eq!(format_size(1024 * 1024), "1.00 MB");
        assert_eq!(format_size(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(format_size(1024_u64.pow(4)), "1.00 TB");
    }

    #[test]
    fn test_format_list() {
        let items = vec!["apple", "banana", "cherry"];
        assert_eq!(format_list(&items), "apple, banana, cherry");

        let empty: Vec<String> = vec![];
        assert_eq!(format_list(&empty), "");
    }

    #[test]
    fn test_format_bullet_list() {
        let items = vec!["first", "second", "third"];
        let result = format_bullet_list(&items);

        assert!(result.contains("• first"));
        assert!(result.contains("• second"));
        assert!(result.contains("• third"));
    }

    #[test]
    fn test_truncate_string_short() {
        let s = "hello";
        assert_eq!(truncate_string(s, 10), "hello");
    }

    #[test]
    fn test_truncate_string_exact() {
        let s = "hello";
        assert_eq!(truncate_string(s, 5), "hello");
    }

    #[test]
    fn test_truncate_string_long() {
        let s = "hello world";
        assert_eq!(truncate_string(s, 8), "hello...");
    }

    #[test]
    fn test_truncate_string_very_short() {
        let s = "hello world";
        assert_eq!(truncate_string(s, 3), "...");
        assert_eq!(truncate_string(s, 2), "...");
        assert_eq!(truncate_string(s, 1), "...");
    }

    #[test]
    fn test_datetime_rfc3339_serialization() {
        use chrono::{DateTime, Utc};

        #[derive(Serialize, Deserialize)]
        struct TestStruct {
            #[serde(with = "super::datetime_rfc3339")]
            timestamp: DateTime<Utc>,
        }

        let data = TestStruct {
            timestamp: Utc::now(),
        };

        let json = to_json(&data).unwrap();
        let restored: TestStruct = from_json(&json).unwrap();

        // Allow small time difference due to serialization
        let diff = (data.timestamp - restored.timestamp)
            .num_milliseconds()
            .abs();
        assert!(diff < 1000); // Less than 1 second difference
    }

    #[test]
    fn test_hex_bytes_serialization() {
        #[derive(Serialize, Deserialize, PartialEq, Debug)]
        struct TestStruct {
            #[serde(with = "super::hex_bytes")]
            data: Vec<u8>,
        }

        let data = TestStruct {
            data: vec![0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0],
        };

        let json = to_json(&data).unwrap();
        assert!(json.contains("123456789abcdef0"));

        let restored: TestStruct = from_json(&json).unwrap();
        assert_eq!(data, restored);
    }

    #[test]
    fn test_invalid_json_error() {
        let invalid_json = r#"{"name":"test", invalid}"#;
        let result: Result<TestData> = from_json(invalid_json);
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_file_error() {
        let result: Result<TestData> = from_json_file("/nonexistent/path.json");
        assert!(result.is_err());
    }
}
