// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CSV Parser for Legacy Academic Records
//!
//! Parses CSV files containing student academic records and maps them
//! to the AcademicCredential format.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Field mapping configuration for CSV import
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldMapping {
    /// CSV column for student ID
    pub student_id: String,
    /// CSV column for first name
    pub first_name: String,
    /// CSV column for last name
    pub last_name: String,
    /// CSV column for degree name
    pub degree_name: String,
    /// CSV column for major/field of study
    pub major: String,
    /// CSV column for conferral date (ISO 8601 or common formats)
    pub conferral_date: String,
    /// Optional: CSV column for GPA
    pub gpa: Option<String>,
    /// Optional: CSV column for honors (comma-separated)
    pub honors: Option<String>,
    /// Optional: CSV column for minor
    pub minor: Option<String>,
    /// Optional: CSV column for email (for DID generation)
    pub email: Option<String>,
    /// Optional: CSV column for date of birth
    pub birth_date: Option<String>,
}

impl Default for FieldMapping {
    fn default() -> Self {
        Self {
            student_id: "student_id".to_string(),
            first_name: "first_name".to_string(),
            last_name: "last_name".to_string(),
            degree_name: "degree_name".to_string(),
            major: "major".to_string(),
            conferral_date: "conferral_date".to_string(),
            gpa: Some("gpa".to_string()),
            honors: Some("honors".to_string()),
            minor: Some("minor".to_string()),
            email: Some("email".to_string()),
            birth_date: Some("birth_date".to_string()),
        }
    }
}

/// Parsed academic record from CSV
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedRecord {
    pub row_number: usize,
    pub student_id: String,
    pub first_name: String,
    pub last_name: String,
    pub degree_name: String,
    pub major: String,
    pub conferral_date: String,
    pub gpa: Option<f32>,
    pub honors: Vec<String>,
    pub minor: Option<String>,
    pub email: Option<String>,
    pub birth_date: Option<String>,
    /// Raw CSV data for reference
    pub raw_data: HashMap<String, String>,
}

/// Parse error with row context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseError {
    pub row: usize,
    pub field: String,
    pub message: String,
    pub code: String,
}

/// Result of parsing a CSV file
#[derive(Debug)]
pub struct ParseResult {
    pub records: Vec<ParsedRecord>,
    pub errors: Vec<ParseError>,
    pub total_rows: usize,
}

/// Parse a CSV file into academic records
pub fn parse_csv(path: &Path, mapping: &FieldMapping) -> Result<ParseResult> {
    let file = File::open(path).context("Failed to open CSV file")?;
    let mut reader = csv::Reader::from_reader(file);

    let headers = reader.headers()?.clone();
    let header_map: HashMap<&str, usize> = headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h, i))
        .collect();

    // Validate required columns exist
    let required = vec![
        (&mapping.student_id, "student_id"),
        (&mapping.first_name, "first_name"),
        (&mapping.last_name, "last_name"),
        (&mapping.degree_name, "degree_name"),
        (&mapping.major, "major"),
        (&mapping.conferral_date, "conferral_date"),
    ];

    for (col, name) in &required {
        if !header_map.contains_key(col.as_str()) {
            anyhow::bail!("Required column '{}' (mapped to {}) not found in CSV", col, name);
        }
    }

    let mut records = Vec::new();
    let mut errors = Vec::new();
    let mut row_num = 1; // 1-indexed for user-friendly reporting

    for result in reader.records() {
        row_num += 1;

        let record = match result {
            Ok(r) => r,
            Err(e) => {
                errors.push(ParseError {
                    row: row_num,
                    field: "row".to_string(),
                    message: format!("Failed to parse row: {}", e),
                    code: "CSV_PARSE_ERROR".to_string(),
                });
                continue;
            }
        };

        // Build raw data map
        let raw_data: HashMap<String, String> = headers
            .iter()
            .zip(record.iter())
            .map(|(h, v)| (h.to_string(), v.to_string()))
            .collect();

        // Extract required fields
        let student_id = get_field(&record, &header_map, &mapping.student_id);
        let first_name = get_field(&record, &header_map, &mapping.first_name);
        let last_name = get_field(&record, &header_map, &mapping.last_name);
        let degree_name = get_field(&record, &header_map, &mapping.degree_name);
        let major = get_field(&record, &header_map, &mapping.major);
        let conferral_date = get_field(&record, &header_map, &mapping.conferral_date);

        // Validate required fields
        let mut row_valid = true;
        for (value, field_name) in [
            (&student_id, "student_id"),
            (&first_name, "first_name"),
            (&last_name, "last_name"),
            (&degree_name, "degree_name"),
            (&major, "major"),
            (&conferral_date, "conferral_date"),
        ] {
            if value.trim().is_empty() {
                errors.push(ParseError {
                    row: row_num,
                    field: field_name.to_string(),
                    message: format!("{} is required", field_name),
                    code: "MISSING_REQUIRED".to_string(),
                });
                row_valid = false;
            }
        }

        if !row_valid {
            continue;
        }

        // Parse optional fields
        let gpa = mapping
            .gpa
            .as_ref()
            .and_then(|col| get_optional_field(&record, &header_map, col))
            .and_then(|s| {
                s.parse::<f32>().ok().or_else(|| {
                    errors.push(ParseError {
                        row: row_num,
                        field: "gpa".to_string(),
                        message: format!("Invalid GPA value: {}", s),
                        code: "INVALID_GPA".to_string(),
                    });
                    None
                })
            });

        // Validate GPA range
        if let Some(g) = gpa {
            if !(0.0..=4.0).contains(&g) {
                errors.push(ParseError {
                    row: row_num,
                    field: "gpa".to_string(),
                    message: format!("GPA {} out of range (0.0-4.0)", g),
                    code: "GPA_OUT_OF_RANGE".to_string(),
                });
            }
        }

        let honors = mapping
            .honors
            .as_ref()
            .and_then(|col| get_optional_field(&record, &header_map, col))
            .map(|s| s.split(',').map(|h| h.trim().to_string()).filter(|h| !h.is_empty()).collect())
            .unwrap_or_default();

        let minor = mapping
            .minor
            .as_ref()
            .and_then(|col| get_optional_field(&record, &header_map, col));

        let email = mapping
            .email
            .as_ref()
            .and_then(|col| get_optional_field(&record, &header_map, col));

        let birth_date = mapping
            .birth_date
            .as_ref()
            .and_then(|col| get_optional_field(&record, &header_map, col));

        // Validate date format
        if let Err(e) = validate_date(&conferral_date) {
            errors.push(ParseError {
                row: row_num,
                field: "conferral_date".to_string(),
                message: e,
                code: "INVALID_DATE".to_string(),
            });
        }

        records.push(ParsedRecord {
            row_number: row_num,
            student_id,
            first_name,
            last_name,
            degree_name,
            major,
            conferral_date,
            gpa,
            honors,
            minor,
            email,
            birth_date,
            raw_data,
        });
    }

    Ok(ParseResult {
        total_rows: row_num - 1,
        records,
        errors,
    })
}

fn get_field(record: &csv::StringRecord, header_map: &HashMap<&str, usize>, col: &str) -> String {
    header_map
        .get(col)
        .and_then(|&i| record.get(i))
        .unwrap_or("")
        .to_string()
}

fn get_optional_field(
    record: &csv::StringRecord,
    header_map: &HashMap<&str, usize>,
    col: &str,
) -> Option<String> {
    header_map
        .get(col)
        .and_then(|&i| record.get(i))
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.to_string())
}

fn validate_date(date: &str) -> Result<(), String> {
    // Accept ISO 8601 (YYYY-MM-DD) or common US formats
    let formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%Y/%m/%d",
    ];

    for fmt in &formats {
        if chrono::NaiveDate::parse_from_str(date, fmt).is_ok() {
            return Ok(());
        }
    }

    Err(format!(
        "Invalid date format '{}'. Expected YYYY-MM-DD or MM/DD/YYYY",
        date
    ))
}

/// Generate a sample CSV template
pub fn generate_template(output: &Path) -> Result<()> {
    let mut file = File::create(output).context("Failed to create template file")?;

    writeln!(
        file,
        "student_id,first_name,last_name,degree_name,major,conferral_date,gpa,honors,minor,email,birth_date"
    )?;
    writeln!(
        file,
        "STU001,Jane,Doe,Bachelor of Science,Computer Science,2024-05-15,3.85,\"Magna Cum Laude,Dean's List\",Mathematics,jane.doe@university.edu,1999-03-20"
    )?;
    writeln!(
        file,
        "STU002,John,Smith,Master of Arts,Philosophy,2024-05-15,3.92,Summa Cum Laude,,john.smith@university.edu,1998-07-10"
    )?;
    writeln!(
        file,
        "STU003,Alice,Johnson,Doctor of Philosophy,Physics,2024-05-15,,,,,1995-01-05"
    )?;

    Ok(())
}

/// Load field mapping from JSON file
pub fn load_mapping(path: &Path) -> Result<FieldMapping> {
    let file = File::open(path).context("Failed to open mapping file")?;
    serde_json::from_reader(file).context("Failed to parse mapping file")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn test_parse_valid_csv() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "student_id,first_name,last_name,degree_name,major,conferral_date,gpa").unwrap();
        writeln!(file, "STU001,Jane,Doe,Bachelor of Science,CS,2024-05-15,3.85").unwrap();

        let result = parse_csv(file.path(), &FieldMapping::default()).unwrap();
        assert_eq!(result.records.len(), 1);
        assert_eq!(result.errors.len(), 0);
        assert_eq!(result.records[0].student_id, "STU001");
        assert_eq!(result.records[0].gpa, Some(3.85));
    }

    #[test]
    fn test_parse_missing_required() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "student_id,first_name,last_name,degree_name,major,conferral_date").unwrap();
        writeln!(file, "STU001,,Doe,Bachelor,CS,2024-05-15").unwrap();

        let result = parse_csv(file.path(), &FieldMapping::default()).unwrap();
        assert_eq!(result.records.len(), 0);
        assert!(result.errors.iter().any(|e| e.field == "first_name"));
    }
}
