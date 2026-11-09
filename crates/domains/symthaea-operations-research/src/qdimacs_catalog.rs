// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Strict QBFGallery benchmark catalog ingestion.
//!
//! Parses the Gallery's benchmark.csv metadata and GNU sha256sum -b checksum
//! list. This catalog is not a solver result receipt and does not itself hash
//! benchmark files. It binds declared catalog rows to declared SHA-256 values;
//! a caller must independently hash the source bytes before execution.

use std::collections::{BTreeMap, BTreeSet};

const REQUIRED_COLUMNS: [&str; 12] = [
    "Filename",
    "Authors",
    "ORCIDs",
    "Year",
    "Format",
    "Result",
    "Family",
    "Description",
    "Converters",
    "License",
    "DOI",
    "Notes",
];

/// Bounds for CSV metadata and checksum catalog ingestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QbfCatalogBudget {
    pub max_csv_bytes: usize,
    pub max_checksum_bytes: usize,
    pub max_rows: usize,
    pub max_columns: usize,
    pub max_field_bytes: usize,
    pub max_checksum_records: usize,
}

impl Default for QbfCatalogBudget {
    fn default() -> Self {
        Self {
            max_csv_bytes: 16 * 1024 * 1024,
            max_checksum_bytes: 16 * 1024 * 1024,
            max_rows: 1 << 18,
            max_columns: 32,
            max_field_bytes: 1 << 20,
            max_checksum_records: 1 << 20,
        }
    }
}

/// Formula encoding named by a QBFGallery catalog row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QbfBenchmarkFormat {
    Qdimacs,
    Qcir,
    Dqdimacs,
}

/// Expected result stated by the benchmark provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QbfBenchmarkExpectedResult {
    True,
    False,
    Unknown,
}

/// One catalog row joined to its declared checksum entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QbfGalleryBenchmarkRecord {
    pub filename: String,
    pub authors: Vec<String>,
    /// Each entry is an ORCID or the Gallery's literal MISSING marker.
    pub orcids: Vec<String>,
    pub year: u16,
    pub format: QbfBenchmarkFormat,
    pub expected_result: QbfBenchmarkExpectedResult,
    pub family: Option<String>,
    pub description: String,
    pub converters: Option<String>,
    pub license: String,
    /// DOI without a resolver URL, as requested by the Gallery.
    pub doi: Option<String>,
    pub notes: Option<String>,
    pub sha256: String,
}

/// Catalog-level metadata. This is a frozen source index, not a run receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QbfGalleryCatalog {
    pub catalog_version: u32,
    pub benchmark_csv_sha256: String,
    pub records: Vec<QbfGalleryBenchmarkRecord>,
    /// Adjacent artifact files that are checksummed but are not formula rows.
    pub extra_checksum_files: Vec<String>,
}

/// Typed, fail-closed catalog parser errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QbfCatalogError {
    CsvByteBudgetExceeded { required: usize, limit: usize },
    ChecksumByteBudgetExceeded { required: usize, limit: usize },
    RowBudgetExceeded { required: usize, limit: usize },
    ColumnBudgetExceeded { required: usize, limit: usize },
    FieldByteBudgetExceeded { required: usize, limit: usize },
    ChecksumRecordBudgetExceeded { required: usize, limit: usize },
    CsvUnclosedQuote { record: usize, field: usize },
    CsvQuoteInUnquotedField { record: usize, field: usize },
    CsvUnexpectedCharacterAfterQuote { record: usize, field: usize, character: char },
    EmptyCsv,
    MissingHeader,
    DuplicateHeader { name: String },
    MissingHeaderColumn { name: String },
    UnknownHeaderColumn { name: String },
    RowColumnCountMismatch { row: usize, expected: usize, actual: usize },
    EmptyRequiredField { row: usize, column: String },
    InvalidFilename { row: usize, filename: String },
    DuplicateFilename { filename: String },
    InvalidYear { row: usize, value: String },
    InvalidFormat { row: usize, value: String },
    InvalidExpectedResult { row: usize, value: String },
    InvalidOrcid { row: usize, value: String },
    OrcidAuthorCountMismatch { row: usize, authors: usize, orcids: usize },
    InvalidDoi { row: usize, value: String },
    InvalidSha256 { line: usize, value: String },
    MalformedChecksumLine { line: usize },
    UnsupportedEscapedChecksumFilename { line: usize },
    EmptyChecksumFilename { line: usize },
    InvalidChecksumFilename { line: usize, filename: String },
    DuplicateChecksumFilename { filename: String },
    MissingChecksumForFormula { filename: String },
    MissingBenchmarkCsvChecksum,
    ResourceAccountingOverflow,
}

/// Append one CSV character, checking the bounded field size before growth.
fn push_field_character(
    field: &mut String,
    character: char,
    record: usize,
    field_index: usize,
    budget: QbfCatalogBudget,
) -> Result<(), QbfCatalogError> {
    let next_bytes = field
        .len()
        .checked_add(character.len_utf8())
        .ok_or(QbfCatalogError::ResourceAccountingOverflow)?;
    if next_bytes > budget.max_field_bytes {
        return Err(QbfCatalogError::FieldByteBudgetExceeded {
            required: next_bytes,
            limit: budget.max_field_bytes,
        });
    }
    field.push(character);
    let _ = (record, field_index);
    Ok(())
}

/// Parse RFC 4180-style CSV with CRLF/LF records, quoted commas, doubled
/// quotes and quoted multiline fields. Bare CR and malformed quote placement
/// are rejected. A final record terminator does not create a phantom row.
fn parse_csv_records(
    source: &str,
    budget: QbfCatalogBudget,
) -> Result<Vec<Vec<String>>, QbfCatalogError> {
    if source.len() > budget.max_csv_bytes {
        return Err(QbfCatalogError::CsvByteBudgetExceeded {
            required: source.len(),
            limit: budget.max_csv_bytes,
        });
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum State {
        FieldStart,
        Unquoted,
        Quoted,
        AfterQuote,
    }

    let mut records = Vec::new();
    let mut record = Vec::new();
    let mut field = String::new();
    let mut state = State::FieldStart;
    let mut record_started = false;
    let mut chars = source.chars().peekable();

    while let Some(character) = chars.next() {
        match state {
            State::FieldStart => match character {
                '"' => {
                    state = State::Quoted;
                    record_started = true;
                }
                ',' => {
                    record.push(String::new());
                    record_started = true;
                    if record.len() >= budget.max_columns {
                        return Err(QbfCatalogError::ColumnBudgetExceeded {
                            required: record.len().saturating_add(1),
                            limit: budget.max_columns,
                        });
                    }
                }
                '\n' => {
                    record.push(String::new());
                    if !record.iter().all(String::is_empty) {
                        let next_rows = records
                            .len()
                            .checked_add(1)
                            .ok_or(QbfCatalogError::ResourceAccountingOverflow)?;
                        if next_rows > budget.max_rows {
                            return Err(QbfCatalogError::RowBudgetExceeded {
                                required: next_rows,
                                limit: budget.max_rows,
                            });
                        }
                        records.push(std::mem::take(&mut record));
                    }
                    state = State::FieldStart;
                    record_started = false;
                }
                '\r' => {
                    if chars.peek() == Some(&'\n') {
                        chars.next();
                        record.push(String::new());
                        if !record.iter().all(String::is_empty) {
                            let next_rows = records
                                .len()
                                .checked_add(1)
                                .ok_or(QbfCatalogError::ResourceAccountingOverflow)?;
                            if next_rows > budget.max_rows {
                                return Err(QbfCatalogError::RowBudgetExceeded {
                                    required: next_rows,
                                    limit: budget.max_rows,
                                });
                            }
                            records.push(std::mem::take(&mut record));
                        }
                        state = State::FieldStart;
                        record_started = false;
                    } else {
                        return Err(QbfCatalogError::CsvUnexpectedCharacterAfterQuote {
                            record: records.len().saturating_add(1),
                            field: record.len().saturating_add(1),
                            character,
                        });
                    }
                }
                other => {
                    push_field_character(
                        &mut field,
                        other,
                        records.len() + 1,
                        record.len() + 1,
                        budget,
                    )?;
                    state = State::Unquoted;
                    record_started = true;
                }
            },
            State::Unquoted => match character {
                ',' => {
                    record.push(std::mem::take(&mut field));
                    if record.len() >= budget.max_columns {
                        return Err(QbfCatalogError::ColumnBudgetExceeded {
                            required: record.len().saturating_add(1),
                            limit: budget.max_columns,
                        });
                    }
                    state = State::FieldStart;
                }
                '\n' => {
                    record.push(std::mem::take(&mut field));
                    if !record.iter().all(String::is_empty) {
                        let next_rows = records
                            .len()
                            .checked_add(1)
                            .ok_or(QbfCatalogError::ResourceAccountingOverflow)?;
                        if next_rows > budget.max_rows {
                            return Err(QbfCatalogError::RowBudgetExceeded {
                                required: next_rows,
                                limit: budget.max_rows,
                            });
                        }
                        records.push(std::mem::take(&mut record));
                    } else {
                        record.clear();
                    }
                    state = State::FieldStart;
                    record_started = false;
                }
                '\r' => {
                    if chars.peek() == Some(&'\n') {
                        chars.next();
                        record.push(std::mem::take(&mut field));
                        if !record.iter().all(String::is_empty) {
                            let next_rows = records
                                .len()
                                .checked_add(1)
                                .ok_or(QbfCatalogError::ResourceAccountingOverflow)?;
                            if next_rows > budget.max_rows {
                                return Err(QbfCatalogError::RowBudgetExceeded {
                                    required: next_rows,
                                    limit: budget.max_rows,
                                });
                            }
                            records.push(std::mem::take(&mut record));
                        } else {
                            record.clear();
                        }
                        state = State::FieldStart;
                        record_started = false;
                    } else {
                        return Err(QbfCatalogError::CsvUnexpectedCharacterAfterQuote {
                            record: records.len().saturating_add(1),
                            field: record.len().saturating_add(1),
                            character,
                        });
                    }
                }
                '"' => {
                    return Err(QbfCatalogError::CsvQuoteInUnquotedField {
                        record: records.len().saturating_add(1),
                        field: record.len().saturating_add(1),
                    });
                }
                other => push_field_character(
                    &mut field,
                    other,
                    records.len() + 1,
                    record.len() + 1,
                    budget,
                )?,
            },
            State::Quoted => match character {
                '"' => {
                    if chars.peek() == Some(&'"') {
                        chars.next();
                        push_field_character(
                            &mut field,
                            '"',
                            records.len() + 1,
                            record.len() + 1,
                            budget,
                        )?;
                    } else {
                        state = State::AfterQuote;
                    }
                }
                '\r' => {
                    if chars.peek() == Some(&'\n') {
                        chars.next();
                        push_field_character(
                            &mut field,
                            '\n',
                            records.len() + 1,
                            record.len() + 1,
                            budget,
                        )?;
                    } else {
                        push_field_character(
                            &mut field,
                            '\r',
                            records.len() + 1,
                            record.len() + 1,
                            budget,
                        )?;
                    }
                }
                other => push_field_character(
                    &mut field,
                    other,
                    records.len() + 1,
                    record.len() + 1,
                    budget,
                )?,
            },
            State::AfterQuote => match character {
                ',' => {
                    record.push(std::mem::take(&mut field));
                    if record.len() >= budget.max_columns {
                        return Err(QbfCatalogError::ColumnBudgetExceeded {
                            required: record.len().saturating_add(1),
                            limit: budget.max_columns,
                        });
                    }
                    state = State::FieldStart;
                }
                '\n' => {
                    record.push(std::mem::take(&mut field));
                    if !record.iter().all(String::is_empty) {
                        let next_rows = records
                            .len()
                            .checked_add(1)
                            .ok_or(QbfCatalogError::ResourceAccountingOverflow)?;
                        if next_rows > budget.max_rows {
                            return Err(QbfCatalogError::RowBudgetExceeded {
                                required: next_rows,
                                limit: budget.max_rows,
                            });
                        }
                        records.push(std::mem::take(&mut record));
                    } else {
                        record.clear();
                    }
                    state = State::FieldStart;
                    record_started = false;
                }
                '\r' => {
                    if chars.peek() == Some(&'\n') {
                        chars.next();
                        record.push(std::mem::take(&mut field));
                        if !record.iter().all(String::is_empty) {
                            let next_rows = records
                                .len()
                                .checked_add(1)
                                .ok_or(QbfCatalogError::ResourceAccountingOverflow)?;
                            if next_rows > budget.max_rows {
                                return Err(QbfCatalogError::RowBudgetExceeded {
                                    required: next_rows,
                                    limit: budget.max_rows,
                                });
                            }
                            records.push(std::mem::take(&mut record));
                        } else {
                            record.clear();
                        }
                        state = State::FieldStart;
                        record_started = false;
                    } else {
                        return Err(QbfCatalogError::CsvUnexpectedCharacterAfterQuote {
                            record: records.len().saturating_add(1),
                            field: record.len().saturating_add(1),
                            character,
                        });
                    }
                }
                other => {
                    return Err(QbfCatalogError::CsvUnexpectedCharacterAfterQuote {
                        record: records.len().saturating_add(1),
                        field: record.len().saturating_add(1),
                        character: other,
                    });
                }
            },
        }
    }

    if state == State::Quoted {
        return Err(QbfCatalogError::CsvUnclosedQuote {
            record: records.len().saturating_add(1),
            field: record.len().saturating_add(1),
        });
    }
    if record_started || !field.is_empty() || !record.is_empty() {
        record.push(field);
        if !record.iter().all(String::is_empty) {
            let next_rows = records
                .len()
                .checked_add(1)
                .ok_or(QbfCatalogError::ResourceAccountingOverflow)?;
            if next_rows > budget.max_rows {
                return Err(QbfCatalogError::RowBudgetExceeded {
                    required: next_rows,
                    limit: budget.max_rows,
                });
            }
            records.push(record);
        }
    }

    if records.is_empty() {
        return Err(QbfCatalogError::EmptyCsv);
    }
    Ok(records)
}

fn clean_optional(value: &str) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

fn validate_filename(row: usize, filename: &str) -> Result<(), QbfCatalogError> {
    let invalid = filename.is_empty()
        || filename == "."
        || filename == ".."
        || filename.contains('/')
        || filename.contains('\\')
        || filename.chars().any(|c| c.is_control());
    if invalid {
        return Err(QbfCatalogError::InvalidFilename {
            row,
            filename: filename.to_owned(),
        });
    }
    Ok(())
}

fn parse_sha256(line: usize, value: &str) -> Result<(), QbfCatalogError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(QbfCatalogError::InvalidSha256 {
            line,
            value: value.to_owned(),
        });
    }
    Ok(())
}

/// Verify the ORCID modulo 11-2 check digit for a canonical 16-character ID.
fn valid_orcid(value: &str) -> bool {
    if value == "MISSING" {
        return true;
    }
    let compact: String = value.chars().filter(|character| *character != '-').collect();
    if compact.len() != 16 {
        return false;
    }
    let mut total: u32 = 0;
    for byte in compact.bytes().take(15) {
        if !byte.is_ascii_digit() {
            return false;
        }
        total = (total + u32::from(byte - b'0')) * 2;
    }
    let remainder = total % 11;
    let result = (12 - remainder) % 11;
    let expected = if result == 10 {
        b'X'
    } else {
        b'0' + result as u8
    };
    compact.as_bytes()[15].to_ascii_uppercase() == expected
}

fn split_and_values(value: &str) -> Vec<String> {
    value
        .split(" and ")
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_owned)
        .collect()
}

fn parse_format(row: usize, value: &str) -> Result<QbfBenchmarkFormat, QbfCatalogError> {
    match value.trim().to_ascii_uppercase().as_str() {
        "QDIMACS" => Ok(QbfBenchmarkFormat::Qdimacs),
        "QCIR" => Ok(QbfBenchmarkFormat::Qcir),
        "DQDIMACS" => Ok(QbfBenchmarkFormat::Dqdimacs),
        _ => Err(QbfCatalogError::InvalidFormat {
            row,
            value: value.to_owned(),
        }),
    }
}

fn parse_result(
    row: usize,
    value: &str,
) -> Result<QbfBenchmarkExpectedResult, QbfCatalogError> {
    match value.trim().to_ascii_uppercase().as_str() {
        "TRUE" => Ok(QbfBenchmarkExpectedResult::True),
        "FALSE" => Ok(QbfBenchmarkExpectedResult::False),
        "UNKNOWN" => Ok(QbfBenchmarkExpectedResult::Unknown),
        _ => Err(QbfCatalogError::InvalidExpectedResult {
            row,
            value: value.to_owned(),
        }),
    }
}

fn parse_checksum_lines(
    source: &str,
    budget: QbfCatalogBudget,
) -> Result<BTreeMap<String, String>, QbfCatalogError> {
    if source.len() > budget.max_checksum_bytes {
        return Err(QbfCatalogError::ChecksumByteBudgetExceeded {
            required: source.len(),
            limit: budget.max_checksum_bytes,
        });
    }
    let mut checksums = BTreeMap::new();
    for (index, raw_line) in source.lines().enumerate() {
        let line_number = index + 1;
        if raw_line.is_empty() {
            continue;
        }
        if raw_line.starts_with('\\') {
            return Err(QbfCatalogError::UnsupportedEscapedChecksumFilename {
                line: line_number,
            });
        }
        if raw_line.len() < 67 {
            return Err(QbfCatalogError::MalformedChecksumLine { line: line_number });
        }
        let digest = &raw_line[..64];
        parse_sha256(line_number, digest)?;
        let separator = &raw_line[64..66];
        if separator != " *" && separator != "  " {
            return Err(QbfCatalogError::MalformedChecksumLine { line: line_number });
        }
        let filename = &raw_line[66..];
        if filename.is_empty() {
            return Err(QbfCatalogError::EmptyChecksumFilename { line: line_number });
        }
        validate_filename(line_number, filename).map_err(|_| {
            QbfCatalogError::InvalidChecksumFilename {
                line: line_number,
                filename: filename.to_owned(),
            }
        })?;
        let next = checksums
            .len()
            .checked_add(1)
            .ok_or(QbfCatalogError::ResourceAccountingOverflow)?;
        if next > budget.max_checksum_records {
            return Err(QbfCatalogError::ChecksumRecordBudgetExceeded {
                required: next,
                limit: budget.max_checksum_records,
            });
        }
        if checksums
            .insert(filename.to_owned(), digest.to_ascii_lowercase())
            .is_some()
        {
            return Err(QbfCatalogError::DuplicateChecksumFilename {
                filename: filename.to_owned(),
            });
        }
    }
    Ok(checksums)
}

/// Parse QBFGallery benchmark.csv and sha256sums, binding each row to its
/// declared SHA-256 entry. This does not hash formula file bytes.
pub fn parse_qbf_gallery_catalog(
    benchmark_csv: &str,
    sha256sums: &str,
    budget: QbfCatalogBudget,
) -> Result<QbfGalleryCatalog, QbfCatalogError> {
    let rows = parse_csv_records(benchmark_csv, budget)?;
    let header = rows.first().ok_or(QbfCatalogError::MissingHeader)?;
    if header.len() > budget.max_columns {
        return Err(QbfCatalogError::ColumnBudgetExceeded {
            required: header.len(),
            limit: budget.max_columns,
        });
    }

    let mut columns = BTreeMap::new();
    for (index, name) in header.iter().enumerate() {
        if !REQUIRED_COLUMNS.contains(&name.as_str()) {
            return Err(QbfCatalogError::UnknownHeaderColumn { name: name.clone() });
        }
        if columns.insert(name.as_str(), index).is_some() {
            return Err(QbfCatalogError::DuplicateHeader { name: name.clone() });
        }
    }
    for required in REQUIRED_COLUMNS {
        if !columns.contains_key(required) {
            return Err(QbfCatalogError::MissingHeaderColumn {
                name: required.to_owned(),
            });
        }
    }

    let checksums = parse_checksum_lines(sha256sums, budget)?;
    let benchmark_csv_sha256 = checksums
        .get("benchmark.csv")
        .cloned()
        .ok_or(QbfCatalogError::MissingBenchmarkCsvChecksum)?;

    let mut records = Vec::with_capacity(rows.len().saturating_sub(1));
    let mut filenames = BTreeSet::new();
    for (offset, row_values) in rows.iter().skip(1).enumerate() {
        let row_number = offset + 2;
        if row_values.len() != header.len() {
            return Err(QbfCatalogError::RowColumnCountMismatch {
                row: row_number,
                expected: header.len(),
                actual: row_values.len(),
            });
        }
        let value = |name: &'static str| -> &str {
            &row_values[*columns.get(name).expect("required column validated")]
        };

        for required in ["Filename", "Authors", "ORCIDs", "Year", "Format", "Result", "Description", "License"] {
            if value(required).trim().is_empty() {
                return Err(QbfCatalogError::EmptyRequiredField {
                    row: row_number,
                    column: required.to_owned(),
                });
            }
        }

        let filename = value("Filename").trim().to_owned();
        validate_filename(row_number, &filename)?;
        if !filenames.insert(filename.clone()) {
            return Err(QbfCatalogError::DuplicateFilename { filename });
        }

        let authors = split_and_values(value("Authors"));
        let orcids = split_and_values(value("ORCIDs"));
        if authors.is_empty() || orcids.is_empty() {
            return Err(QbfCatalogError::EmptyRequiredField {
                row: row_number,
                column: "Authors/ORCIDs".to_owned(),
            });
        }
        if authors.len() != orcids.len() {
            return Err(QbfCatalogError::OrcidAuthorCountMismatch {
                row: row_number,
                authors: authors.len(),
                orcids: orcids.len(),
            });
        }
        for orcid in &orcids {
            if !valid_orcid(orcid) {
                return Err(QbfCatalogError::InvalidOrcid {
                    row: row_number,
                    value: orcid.clone(),
                });
            }
        }

        let year_value = value("Year").trim();
        let year = year_value
            .parse::<u16>()
            .ok()
            .filter(|year| (1900..=2100).contains(year))
            .ok_or_else(|| QbfCatalogError::InvalidYear {
                row: row_number,
                value: year_value.to_owned(),
            })?;

        let format = parse_format(row_number, value("Format"))?;
        let expected_result = parse_result(row_number, value("Result"))?;
        let license = value("License").trim().to_owned();

        let doi = clean_optional(value("DOI"));
        if let Some(doi_value) = &doi {
            if !doi_value.starts_with("10.") || !doi_value.contains('/') || doi_value.contains("://") {
                return Err(QbfCatalogError::InvalidDoi {
                    row: row_number,
                    value: doi_value.clone(),
                });
            }
        }

        let sha256 = checksums
            .get(&filename)
            .cloned()
            .ok_or_else(|| QbfCatalogError::MissingChecksumForFormula {
                filename: filename.clone(),
            })?;

        records.push(QbfGalleryBenchmarkRecord {
            filename,
            authors,
            orcids,
            year,
            format,
            expected_result,
            family: clean_optional(value("Family")),
            description: value("Description").trim().to_owned(),
            converters: clean_optional(value("Converters")),
            license,
            doi,
            notes: clean_optional(value("Notes")),
            sha256,
        });
    }

    let formula_names: BTreeSet<&str> = records.iter().map(|record| record.filename.as_str()).collect();
    let mut extra_checksum_files: Vec<String> = checksums
        .keys()
        .filter(|name| name.as_str() != "benchmark.csv" && !formula_names.contains(name.as_str()))
        .cloned()
        .collect();
    extra_checksum_files.sort();

    Ok(QbfGalleryCatalog {
        catalog_version: 1,
        benchmark_csv_sha256,
        records,
        extra_checksum_files,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: &str = "Filename,Authors,ORCIDs,Year,Format,Result,Family,Description,Converters,License,DOI,Notes\n";
    const ORCID_A: &str = "0000-0002-1825-0097";

    fn budget() -> QbfCatalogBudget {
        QbfCatalogBudget {
            max_csv_bytes: 4096,
            max_checksum_bytes: 4096,
            max_rows: 32,
            max_columns: 16,
            max_field_bytes: 1024,
            max_checksum_records: 32,
        }
    }

    fn checksum(filename: &str, marker: &str, byte: char) -> String {
        format!("{}{}{}\n", byte.to_string().repeat(64), marker, filename)
    }

    fn fixture() -> (String, String) {
        let csv = format!(
            "{}toy.qdimacs,Alice,{ORCID_A},2026,QDIMACS,TRUE,parity,\"Simple, quoted description\",,CC-BY-4.0,10.1234/example,\"fixture note\"\n",
            HEADER,
            ORCID_A = ORCID_A
        );
        let sums = format!(
            "{}{}",
            checksum("benchmark.csv", " *", 'a'),
            checksum("toy.qdimacs", " *", 'b')
        );
        (csv, sums)
    }

    #[test]
    fn parses_gallery_catalog_and_joins_formula_checksums() {
        let (csv, sums) = fixture();
        let catalog = parse_qbf_gallery_catalog(&csv, &sums, budget()).unwrap();
        assert_eq!(catalog.catalog_version, 1);
        assert_eq!(catalog.records.len(), 1);
        assert_eq!(catalog.records[0].filename, "toy.qdimacs");
        assert_eq!(catalog.records[0].authors, vec!["Alice"]);
        assert_eq!(catalog.records[0].orcids, vec![ORCID_A]);
        assert_eq!(catalog.records[0].format, QbfBenchmarkFormat::Qdimacs);
        assert_eq!(catalog.records[0].expected_result, QbfBenchmarkExpectedResult::True);
        assert_eq!(catalog.records[0].description, "Simple, quoted description");
        assert_eq!(catalog.records[0].sha256, "b".repeat(64));
        assert_eq!(catalog.benchmark_csv_sha256, "a".repeat(64));
    }

    #[test]
    fn supports_reordered_header_columns() {
        let csv = "Result,Filename,Authors,ORCIDs,Year,Format,Family,Description,Converters,License,DOI,Notes\nTRUE,toy.qdimacs,Alice,0000-0002-1825-0097,2026,QDIMACS,,ok,,CC0,,\n";
        let sums = format!(
            "{}{}",
            checksum("benchmark.csv", "  ", 'a'),
            checksum("toy.qdimacs", "  ", 'b')
        );
        let catalog = parse_qbf_gallery_catalog(csv, &sums, budget()).unwrap();
        assert_eq!(catalog.records[0].expected_result, QbfBenchmarkExpectedResult::True);
    }

    #[test]
    fn handles_doubled_quotes_and_embedded_newlines() {
        let csv = format!(
            "{}toy.qdimacs,Alice,{ORCID_A},2026,QDIMACS,UNKNOWN,,\"A \"\"quoted\"\" line\\nand a second line\",,MIT,,\n",
            HEADER,
            ORCID_A = ORCID_A
        );
        let sums = format!(
            "{}{}",
            checksum("benchmark.csv", " *", 'a'),
            checksum("toy.qdimacs", " *", 'b')
        );
        let catalog = parse_qbf_gallery_catalog(&csv, &sums, budget()).unwrap();
        assert!(catalog.records[0].description.contains("\"quoted\""));
        assert!(catalog.records[0].description.contains('\n'));
        assert_eq!(catalog.records[0].expected_result, QbfBenchmarkExpectedResult::Unknown);
    }

    #[test]
    fn supports_dqdimacs_and_qcir_metadata_without_coercing_them_to_qdimacs() {
        let csv = format!(
            "{}a.dqdimacs,Alice,{ORCID_A},2026,DQDIMACS,FALSE,,dependency QBF,,MIT,,\\nb.qcir,Alice,{ORCID_A},2026,QCIR,UNKNOWN,,circuit QBF,,MIT,,\\n",
            HEADER,
            ORCID_A = ORCID_A
        );
        let sums = format!(
            "{}{}{}",
            checksum("benchmark.csv", " *", 'a'),
            checksum("a.dqdimacs", " *", 'b'),
            checksum("b.qcir", " *", 'c')
        );
        let catalog = parse_qbf_gallery_catalog(&csv, &sums, budget()).unwrap();
        assert_eq!(catalog.records[0].format, QbfBenchmarkFormat::Dqdimacs);
        assert_eq!(catalog.records[1].format, QbfBenchmarkFormat::Qcir);
    }

    #[test]
    fn supports_multiple_authors_and_missing_orcid_marker() {
        let csv = format!(
            "{}toy.qdimacs,Alice and Bob,{ORCID_A} and MISSING,2026,QDIMACS,TRUE,,ok,,MIT,,\\n",
            HEADER,
            ORCID_A = ORCID_A
        );
        let sums = format!(
            "{}{}",
            checksum("benchmark.csv", " *", 'a'),
            checksum("toy.qdimacs", " *", 'b')
        );
        let catalog = parse_qbf_gallery_catalog(&csv, &sums, budget()).unwrap();
        assert_eq!(catalog.records[0].authors.len(), 2);
        assert_eq!(catalog.records[0].orcids, vec![ORCID_A, "MISSING"]);
    }

    #[test]
    fn permits_additional_checksummed_artifact_files() {
        let (csv, sums) = fixture();
        let sums = format!("{}{}", sums, checksum("README.txt", " *", 'c'));
        let catalog = parse_qbf_gallery_catalog(&csv, &sums, budget()).unwrap();
        assert_eq!(catalog.extra_checksum_files, vec!["README.txt"]);
    }

    #[test]
    fn rejects_duplicate_rows_and_missing_file_checksums() {
        let duplicate = format!(
            "{}{}",
            HEADER,
            "toy.qdimacs,Alice,0000-0002-1825-0097,2026,QDIMACS,TRUE,,ok,,MIT,,\n".repeat(2)
        );
        let sums = format!(
            "{}{}",
            checksum("benchmark.csv", " *", 'a'),
            checksum("toy.qdimacs", " *", 'b')
        );
        assert!(matches!(
            parse_qbf_gallery_catalog(&duplicate, &sums, budget()),
            Err(QbfCatalogError::DuplicateFilename { .. })
        ));

        let missing = checksum("benchmark.csv", " *", 'a');
        assert!(matches!(
            parse_qbf_gallery_catalog(&duplicate[..HEADER.len() + duplicate[HEADER.len()..].find('\n').unwrap() + 1], &missing, budget()),
            Err(QbfCatalogError::MissingChecksumForFormula { .. })
        ));
    }

    #[test]
    fn rejects_duplicate_checksum_filenames_and_missing_csv_checksum() {
        let (csv, _) = fixture();
        let duplicate = format!(
            "{}{}{}",
            checksum("benchmark.csv", " *", 'a'),
            checksum("toy.qdimacs", " *", 'b'),
            checksum("toy.qdimacs", " *", 'c')
        );
        assert!(matches!(
            parse_qbf_gallery_catalog(&csv, &duplicate, budget()),
            Err(QbfCatalogError::DuplicateChecksumFilename { .. })
        ));
        assert!(matches!(
            parse_qbf_gallery_catalog(&csv, &checksum("toy.qdimacs", " *", 'b'), budget()),
            Err(QbfCatalogError::MissingBenchmarkCsvChecksum)
        ));
    }

    #[test]
    fn rejects_invalid_digest_and_checksum_line() {
        let (csv, _) = fixture();
        let bad = format!("{} toy.qdimacs\n", "g".repeat(64));
        assert!(matches!(
            parse_qbf_gallery_catalog(&csv, &bad, budget()),
            Err(QbfCatalogError::InvalidSha256 { .. })
        ));
        assert!(matches!(
            parse_qbf_gallery_catalog(&csv, "bad checksum line\n", budget()),
            Err(QbfCatalogError::MalformedChecksumLine { .. })
        ));
    }

    #[test]
    fn rejects_escaped_checksum_filenames_until_supported() {
        let (csv, _) = fixture();
        let sums = format!("\\{} *toy.qdimacs\n", "a".repeat(64));
        assert!(matches!(
            parse_qbf_gallery_catalog(&csv, &sums, budget()),
            Err(QbfCatalogError::UnsupportedEscapedChecksumFilename { .. })
        ));
    }

    #[test]
    fn rejects_path_traversal_and_duplicate_header_names() {
        let csv = format!(
            "{}../toy.qdimacs,Alice,{ORCID_A},2026,QDIMACS,TRUE,,ok,,MIT,,\n",
            HEADER,
            ORCID_A = ORCID_A
        );
        let sums = format!(
            "{}{}",
            checksum("benchmark.csv", " *", 'a'),
            checksum("../toy.qdimacs", " *", 'b')
        );
        assert!(matches!(
            parse_qbf_gallery_catalog(&csv, &sums, budget()),
            Err(QbfCatalogError::InvalidFilename { .. })
        ));

        let duplicate_header = HEADER.replacen("Filename", "Authors", 1);
        assert!(matches!(
            parse_qbf_gallery_catalog(&duplicate_header, &sums, budget()),
            Err(QbfCatalogError::DuplicateHeader { .. })
                | Err(QbfCatalogError::UnknownHeaderColumn { .. })
        ));
    }

    #[test]
    fn rejects_unknown_columns_and_missing_required_columns() {
        let unknown = format!("{},Extra\n", HEADER.trim_end());
        assert!(matches!(
            parse_qbf_gallery_catalog(&unknown, "", budget()),
            Err(QbfCatalogError::UnknownHeaderColumn { .. })
        ));

        let missing = "Filename,Authors\n";
        assert!(matches!(
            parse_qbf_gallery_catalog(missing, "", budget()),
            Err(QbfCatalogError::MissingHeaderColumn { .. })
        ));
    }

    #[test]
    fn rejects_invalid_result_format_year_license_orcid_and_doi() {
        let bad_cases = [
            ("QDIMACS", "MAYBE", "2026", "MIT", ORCID_A, ""),
            ("UNKNOWN_FORMAT", "TRUE", "2026", "MIT", ORCID_A, ""),
            ("QDIMACS", "TRUE", "1800", "MIT", ORCID_A, ""),
            ("QDIMACS", "TRUE", "2026", "", ORCID_A, ""),
            ("QDIMACS", "TRUE", "2026", "MIT", "0000-0002-1825-0098", ""),
            ("QDIMACS", "TRUE", "2026", "MIT", ORCID_A, "https://doi.org/10.1234/x"),
        ];
        for (format, result, year, license, orcid, doi) in bad_cases {
            let csv = format!(
                "{}toy.qdimacs,Alice,{},{},{},{},,ok,,{},{},\n",
                HEADER, orcid, year, format, result, license, doi
            );
            let sums = format!(
                "{}{}",
                checksum("benchmark.csv", " *", 'a'),
                checksum("toy.qdimacs", " *", 'b')
            );
            assert!(parse_qbf_gallery_catalog(&csv, &sums, budget()).is_err());
        }
    }

    #[test]
    fn rejects_malformed_csv_quotes_and_column_counts() {
        let malformed = format!(
            "{}toy.qdimacs,Alice,{ORCID_A},2026,QDIMACS,TRUE,,\"unterminated,,MIT,,\n",
            HEADER,
            ORCID_A = ORCID_A
        );
        assert!(matches!(
            parse_qbf_gallery_catalog(&malformed, "", budget()),
            Err(QbfCatalogError::CsvUnclosedQuote { .. })
        ));

        let wrong_columns = format!(
            "{}toy.qdimacs,Alice,{ORCID_A},2026,QDIMACS,TRUE\n",
            HEADER,
            ORCID_A = ORCID_A
        );
        let sums = format!(
            "{}{}",
            checksum("benchmark.csv", " *", 'a'),
            checksum("toy.qdimacs", " *", 'b')
        );
        assert!(matches!(
            parse_qbf_gallery_catalog(&wrong_columns, &sums, budget()),
            Err(QbfCatalogError::RowColumnCountMismatch { .. })
        ));
    }

    #[test]
    fn rejects_csv_byte_field_row_and_checksum_budgets() {
        let (csv, sums) = fixture();
        assert!(matches!(
            parse_qbf_gallery_catalog(
                &csv,
                &sums,
                QbfCatalogBudget {
                    max_csv_bytes: 8,
                    ..budget()
                }
            ),
            Err(QbfCatalogError::CsvByteBudgetExceeded { .. })
        ));
        assert!(matches!(
            parse_qbf_gallery_catalog(
                &csv,
                &sums,
                QbfCatalogBudget {
                    max_checksum_bytes: 8,
                    ..budget()
                }
            ),
            Err(QbfCatalogError::ChecksumByteBudgetExceeded { .. })
        ));
        assert!(matches!(
            parse_qbf_gallery_catalog(
                &csv,
                &sums,
                QbfCatalogBudget {
                    max_rows: 1,
                    ..budget()
                }
            ),
            Err(QbfCatalogError::RowBudgetExceeded { .. })
        ));
        assert!(matches!(
            parse_qbf_gallery_catalog(
                &csv,
                &sums,
                QbfCatalogBudget {
                    max_field_bytes: 3,
                    ..budget()
                }
            ),
            Err(QbfCatalogError::FieldByteBudgetExceeded { .. })
        ));
        assert!(matches!(
            parse_qbf_gallery_catalog(
                &csv,
                &sums,
                QbfCatalogBudget {
                    max_checksum_records: 1,
                    ..budget()
                }
            ),
            Err(QbfCatalogError::ChecksumRecordBudgetExceeded { .. })
        ));
    }
}
