// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dependency-light, fail-closed audit primitives for Cobaya text-chain files.
//!
//! This is a chain-summary auditor, not an MCMC sampler, GetDist replacement,
//! cosmological likelihood, or inference reproduction. It parses the explicit
//! column header and supports weighted scalar summaries for selected columns.
//! Its weighted empirical quantiles intentionally use the left-continuous
//! inverse empirical CDF; these are not guaranteed to equal GetDist's KDE-based
//! marginal limits. Autocorrelation-corrected MCMC ESS is not computed here.

use std::collections::HashSet;
use std::fmt;
use std::fs::File;
use std::io::{BufRead, Read};
use std::path::Path;

use sha2::{Digest, Sha256};

pub const CHAIN_AUDIT_PROTOCOL: &str = "symthaea-cobaya-chain-audit";
pub const CHAIN_AUDIT_VERSION: u16 = 1;

#[derive(Clone, Debug, PartialEq)]
pub struct ChainTable {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<f64>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WeightedColumnSummary {
    pub name: String,
    pub sample_rows: usize,
    pub positive_weight_rows: usize,
    pub total_weight: f64,
    pub weight_effective_sample_size: f64,
    pub weighted_mean: f64,
    /// Population variance under the normalized sample weights.
    pub weighted_variance: f64,
    /// Left-continuous weighted empirical quantiles at 0.16, 0.50, and 0.84.
    pub q16: f64,
    pub q50: f64,
    pub q84: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChainAuditError {
    Io(String),
    MissingHeader,
    InvalidHeader(String),
    DuplicateColumn(String),
    MissingSamples,
    MismatchedChainHeaders,
    WrongColumnCount {
        line: usize,
        expected: usize,
        actual: usize,
    },
    InvalidNumber {
        line: usize,
        column: String,
        value: String,
    },
    NonFiniteValue {
        line: usize,
        column: String,
    },
    NegativeWeight {
        line: usize,
    },
    ZeroTotalWeight,
    UnknownColumn(String),
    InvalidSha256(String),
    MissingManifestEntry(String),
    DuplicateManifestEntry(String),
    MalformedChecksumManifestLine {
        line: usize,
    },
    NumericalOverflow,
}

impl fmt::Display for ChainAuditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(message) => write!(f, "I/O error: {message}"),
            Self::MissingHeader => {
                f.write_str("missing Cobaya header beginning with '# weight minuslogpost'")
            }
            Self::InvalidHeader(message) => write!(f, "invalid chain header: {message}"),
            Self::DuplicateColumn(name) => write!(f, "duplicate chain column: {name}"),
            Self::MissingSamples => f.write_str("chain contains no sample rows"),
            Self::MismatchedChainHeaders => {
                f.write_str("cannot combine chain files with different column headers")
            }
            Self::WrongColumnCount {
                line,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "line {line}: expected {expected} columns, found {actual}"
                )
            }
            Self::InvalidNumber {
                line,
                column,
                value,
            } => {
                write!(f, "line {line}, column {column}: invalid number {value:?}")
            }
            Self::NonFiniteValue { line, column } => {
                write!(f, "line {line}, column {column}: non-finite value")
            }
            Self::NegativeWeight { line } => {
                write!(f, "line {line}: sample weight must be non-negative")
            }
            Self::ZeroTotalWeight => f.write_str("chain total weight must be finite and positive"),
            Self::UnknownColumn(name) => write!(f, "unknown chain column: {name}"),
            Self::InvalidSha256(value) => write!(f, "invalid SHA-256 digest: {value:?}"),
            Self::MissingManifestEntry(path) => {
                write!(f, "SHA-256 manifest has no entry for {path:?}")
            }
            Self::DuplicateManifestEntry(path) => {
                write!(f, "SHA-256 manifest has duplicate entries for {path:?}")
            }
            Self::MalformedChecksumManifestLine { line } => {
                write!(f, "malformed SHA-256 manifest line {line}")
            }
            Self::NumericalOverflow => {
                f.write_str("chain summary overflowed floating-point arithmetic")
            }
        }
    }
}

impl std::error::Error for ChainAuditError {}

/// Return the lowercase SHA-256 hex digest of the exact bytes supplied.
pub fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Stream a file and return its lowercase SHA-256 hex digest.
pub fn sha256_file(path: impl AsRef<Path>) -> Result<String, ChainAuditError> {
    let file = File::open(path.as_ref()).map_err(|error| ChainAuditError::Io(error.to_string()))?;
    let mut reader = std::io::BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let bytes_read = reader
            .read(&mut buffer)
            .map_err(|error| ChainAuditError::Io(error.to_string()))?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Compare actual and expected SHA-256 digests, accepting uppercase or
/// lowercase hex but rejecting prefixes, whitespace, or malformed lengths.
pub fn verify_sha256_digest(actual: &str, expected: &str) -> Result<bool, ChainAuditError> {
    validate_sha256_hex(actual)?;
    validate_sha256_hex(expected)?;
    Ok(actual.eq_ignore_ascii_case(expected))
}

/// Verify a local file against a SHA-256 digest obtained from an external
/// trusted manifest. A false result means bytes differ; malformed digests are
/// errors. The caller chooses the local path; URLs are never fetched here.
pub fn verify_sha256_file(path: impl AsRef<Path>, expected: &str) -> Result<bool, ChainAuditError> {
    let actual = sha256_file(path)?;
    verify_sha256_digest(&actual, expected)
}

fn validate_sha256_hex(value: &str) -> Result<(), ChainAuditError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(ChainAuditError::InvalidSha256(value.to_string()));
    }
    Ok(())
}

/// Resolve one exact relative-path entry from a standard SHA-256SUMS-style
/// manifest (64 hex digits, whitespace separator, then the path). It rejects
/// malformed non-empty lines, traversal-like target paths, missing entries,
/// and duplicate entries rather than guessing which digest was intended.
pub fn expected_sha256_from_manifest(
    manifest_text: &str,
    relative_path: &str,
) -> Result<String, ChainAuditError> {
    if relative_path.trim().is_empty()
        || relative_path.starts_with('/')
        || relative_path.contains('\\')
        || relative_path.chars().any(char::is_control)
        || relative_path
            .split('/')
            .any(|component| component == ".." || component.is_empty())
    {
        return Err(ChainAuditError::InvalidHeader(
            "manifest target must be a non-empty POSIX relative path without empty or '..' components".to_string(),
        ));
    }

    let mut match_digest: Option<String> = None;
    for (index, raw_line) in manifest_text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw_line.trim_end_matches('\r');
        if line.trim().is_empty() {
            continue;
        }
        if line.len() < 66
            || !line.as_bytes()[..64]
                .iter()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(ChainAuditError::MalformedChecksumManifestLine { line: line_number });
        }
        let digest = &line[..64];
        let remainder = &line[64..];
        let path = if let Some(path) = remainder.strip_prefix("  ") {
            path
        } else if let Some(path) = remainder.strip_prefix(" *") {
            path
        } else if let Some(path) = remainder.strip_prefix(' ') {
            path
        } else {
            return Err(ChainAuditError::MalformedChecksumManifestLine { line: line_number });
        };
        if path.is_empty() || path.trim() != path {
            return Err(ChainAuditError::MalformedChecksumManifestLine { line: line_number });
        }
        if path != relative_path {
            continue;
        }
        validate_sha256_hex(digest)?;
        if match_digest.is_some() {
            return Err(ChainAuditError::DuplicateManifestEntry(
                relative_path.to_string(),
            ));
        }
        match_digest = Some(digest.to_ascii_lowercase());
    }

    match_digest.ok_or_else(|| ChainAuditError::MissingManifestEntry(relative_path.to_string()))
}

/// Verify a local file using the expected digest resolved from a frozen
/// SHA-256SUMS-style manifest. The manifest itself must be acquired and
/// retained separately; this function does not fetch remote resources.
pub fn verify_sha256_file_against_manifest(
    file_path: impl AsRef<Path>,
    manifest_text: &str,
    relative_path: &str,
) -> Result<(String, String, bool), ChainAuditError> {
    let expected = expected_sha256_from_manifest(manifest_text, relative_path)?;
    let actual = sha256_file(file_path)?;
    let matches = verify_sha256_digest(&actual, &expected)?;
    Ok((expected, actual, matches))
}

impl ChainTable {
    /// Combine chain tables only when their full column order matches exactly.
    /// This prevents parameter columns from silently shifting between files.
    pub fn combine(mut tables: Vec<Self>) -> Result<Self, ChainAuditError> {
        if tables.is_empty() {
            return Err(ChainAuditError::MissingSamples);
        }
        for table in &tables {
            validate_table(table)?;
        }
        let mut combined = tables.remove(0);
        for table in tables {
            if table.columns != combined.columns {
                return Err(ChainAuditError::MismatchedChainHeaders);
            }
            combined.rows.extend(table.rows);
        }
        validate_table(&combined)?;
        Ok(combined)
    }

    /// Parse a Cobaya sample file. The first recognized header must list weight
    /// and minuslogpost in the first two columns. Malformed rows are rejected,
    /// never silently skipped or shifted into a wrong parameter column.
    pub fn parse<R: BufRead>(reader: R) -> Result<Self, ChainAuditError> {
        let mut columns: Option<Vec<String>> = None;
        let mut rows: Vec<Vec<f64>> = Vec::new();

        for (index, result) in reader.lines().enumerate() {
            let line_number = index + 1;
            let line = result.map_err(|error| ChainAuditError::Io(error.to_string()))?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if let Some(comment) = trimmed.strip_prefix('#') {
                let candidate = comment.split_whitespace().collect::<Vec<_>>();
                if columns.is_none()
                    && candidate.first() == Some(&"weight")
                    && candidate.get(1) == Some(&"minuslogpost")
                {
                    let parsed = candidate
                        .iter()
                        .map(|value| (*value).to_string())
                        .collect::<Vec<_>>();
                    validate_header(&parsed)?;
                    columns = Some(parsed);
                }
                continue;
            }

            let header = columns.as_ref().ok_or(ChainAuditError::MissingHeader)?;
            let tokens = trimmed.split_whitespace().collect::<Vec<_>>();
            if tokens.len() != header.len() {
                return Err(ChainAuditError::WrongColumnCount {
                    line: line_number,
                    expected: header.len(),
                    actual: tokens.len(),
                });
            }
            let mut values = Vec::with_capacity(tokens.len());
            for (column_index, token) in tokens.iter().enumerate() {
                let value = token
                    .parse::<f64>()
                    .map_err(|_| ChainAuditError::InvalidNumber {
                        line: line_number,
                        column: header[column_index].clone(),
                        value: (*token).to_string(),
                    })?;
                if !value.is_finite() {
                    return Err(ChainAuditError::NonFiniteValue {
                        line: line_number,
                        column: header[column_index].clone(),
                    });
                }
                if column_index == 0 && value < 0.0 {
                    return Err(ChainAuditError::NegativeWeight { line: line_number });
                }
                values.push(value);
            }
            rows.push(values);
        }

        let columns = columns.ok_or(ChainAuditError::MissingHeader)?;
        if rows.is_empty() {
            return Err(ChainAuditError::MissingSamples);
        }
        let weight_sum = rows.iter().map(|row| row[0]).sum::<f64>();
        if !weight_sum.is_finite() {
            return Err(ChainAuditError::NumericalOverflow);
        }
        if weight_sum <= 0.0 {
            return Err(ChainAuditError::ZeroTotalWeight);
        }
        Ok(Self { columns, rows })
    }

    /// Summarize one named column with the sample weight in column zero.
    ///
    /// The reported weight ESS is (sum weights)^2 / sum(weights^2). It accounts
    /// for unequal sample weights only and is not autocorrelation-corrected.
    pub fn summarize(&self, name: &str) -> Result<WeightedColumnSummary, ChainAuditError> {
        validate_table(self)?;
        let column_index = self
            .columns
            .iter()
            .position(|column| column == name)
            .ok_or_else(|| ChainAuditError::UnknownColumn(name.to_string()))?;
        if column_index == 0 {
            return Err(ChainAuditError::InvalidHeader(
                "weight is a sampling weight, not a parameter to summarize".to_string(),
            ));
        }

        let mut total_weight = 0.0;
        let mut weight_squared_sum = 0.0;
        let mut weighted_value_sum = 0.0;
        let mut positive_weight_rows = 0usize;
        let mut weighted_values = Vec::with_capacity(self.rows.len());

        for row in &self.rows {
            let weight = row[0];
            let value = row[column_index];
            if weight > 0.0 {
                positive_weight_rows += 1;
                weighted_values.push((value, weight));
            }
            total_weight += weight;
            weight_squared_sum += weight * weight;
            weighted_value_sum += weight * value;
        }

        if !total_weight.is_finite()
            || !weight_squared_sum.is_finite()
            || !weighted_value_sum.is_finite()
        {
            return Err(ChainAuditError::NumericalOverflow);
        }
        if total_weight <= 0.0 || weight_squared_sum <= 0.0 {
            return Err(ChainAuditError::ZeroTotalWeight);
        }

        let mean = weighted_value_sum / total_weight;
        if !mean.is_finite() {
            return Err(ChainAuditError::NumericalOverflow);
        }
        let mut weighted_squared_deviation = 0.0;
        for row in &self.rows {
            let weight = row[0];
            let difference = row[column_index] - mean;
            weighted_squared_deviation += weight * difference * difference;
        }
        let variance = weighted_squared_deviation / total_weight;
        let weight_ess_ratio = total_weight / weight_squared_sum.sqrt();
        let weight_ess = weight_ess_ratio * weight_ess_ratio;
        if !variance.is_finite() || !weight_ess.is_finite() {
            return Err(ChainAuditError::NumericalOverflow);
        }

        weighted_values.sort_by(|left, right| left.0.total_cmp(&right.0));
        let q16 = weighted_quantile(&weighted_values, total_weight, 0.16)?;
        let q50 = weighted_quantile(&weighted_values, total_weight, 0.50)?;
        let q84 = weighted_quantile(&weighted_values, total_weight, 0.84)?;

        Ok(WeightedColumnSummary {
            name: name.to_string(),
            sample_rows: self.rows.len(),
            positive_weight_rows,
            total_weight,
            weight_effective_sample_size: weight_ess,
            weighted_mean: mean,
            weighted_variance: variance,
            q16,
            q50,
            q84,
        })
    }
}

fn validate_header(columns: &[String]) -> Result<(), ChainAuditError> {
    if columns.len() < 3 {
        return Err(ChainAuditError::InvalidHeader(
            "expected weight, minuslogpost, and at least one parameter".to_string(),
        ));
    }
    if columns[0] != "weight" || columns[1] != "minuslogpost" {
        return Err(ChainAuditError::InvalidHeader(
            "first two columns must be weight and minuslogpost".to_string(),
        ));
    }
    let mut unique = HashSet::with_capacity(columns.len());
    for column in columns {
        if column.trim().is_empty() {
            return Err(ChainAuditError::InvalidHeader(
                "empty column name".to_string(),
            ));
        }
        if !unique.insert(column.as_str()) {
            return Err(ChainAuditError::DuplicateColumn(column.clone()));
        }
    }
    Ok(())
}

fn validate_table(table: &ChainTable) -> Result<(), ChainAuditError> {
    validate_header(&table.columns)?;
    if table.rows.is_empty() {
        return Err(ChainAuditError::MissingSamples);
    }
    let mut total_weight = 0.0;
    for (row_index, row) in table.rows.iter().enumerate() {
        if row.len() != table.columns.len() {
            return Err(ChainAuditError::WrongColumnCount {
                line: row_index + 1,
                expected: table.columns.len(),
                actual: row.len(),
            });
        }
        for (column_index, value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(ChainAuditError::NonFiniteValue {
                    line: row_index + 1,
                    column: table.columns[column_index].clone(),
                });
            }
            if column_index == 0 && *value < 0.0 {
                return Err(ChainAuditError::NegativeWeight {
                    line: row_index + 1,
                });
            }
        }
        total_weight += row[0];
    }
    if !total_weight.is_finite() {
        return Err(ChainAuditError::NumericalOverflow);
    }
    if total_weight <= 0.0 {
        return Err(ChainAuditError::ZeroTotalWeight);
    }
    Ok(())
}

fn weighted_quantile(
    sorted_values: &[(f64, f64)],
    total_weight: f64,
    quantile: f64,
) -> Result<f64, ChainAuditError> {
    if sorted_values.is_empty() || !total_weight.is_finite() || total_weight <= 0.0 {
        return Err(ChainAuditError::ZeroTotalWeight);
    }
    let target = quantile * total_weight;
    let mut cumulative = 0.0;
    for (value, weight) in sorted_values {
        cumulative += *weight;
        if !cumulative.is_finite() {
            return Err(ChainAuditError::NumericalOverflow);
        }
        if cumulative >= target {
            return Ok(*value);
        }
    }
    // Rounding can leave cumulative infinitesimally below total_weight.
    sorted_values
        .last()
        .map(|row| row.0)
        .ok_or(ChainAuditError::ZeroTotalWeight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const FIXTURE: &str = "# weight minuslogpost H0 omega_m\n\
1 -10 67 0.30\n\
1 -11 68 0.31\n\
2 -12 69 0.32\n";

    #[test]
    fn sha256_bytes_matches_the_published_known_vector() {
        assert_eq!(
            sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn sha256_verifier_matches_known_vector_and_rejects_mutation() {
        let path = std::env::temp_dir().join(format!(
            "symthaea-cosmology-bench-sha256-{}.bin",
            std::process::id()
        ));
        std::fs::write(&path, b"abc").unwrap();
        let actual = sha256_file(&path).unwrap();
        assert_eq!(
            actual,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert!(verify_sha256_file(&path, &actual).unwrap());
        assert!(
            verify_sha256_digest(
                &actual,
                "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD"
            )
            .unwrap()
        );
        assert!(
            !verify_sha256_digest(
                &actual,
                "ca7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
            )
            .unwrap()
        );
        assert!(verify_sha256_digest(&actual, "bad").is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn checksum_manifest_lookup_is_exact_and_fail_closed() {
        let digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        let manifest = format!("{digest}  data/abc.txt\n{digest}  data/def.txt\n");
        assert_eq!(
            expected_sha256_from_manifest(&manifest, "data/abc.txt").unwrap(),
            digest
        );
        assert_eq!(
            expected_sha256_from_manifest(&manifest, "missing.txt"),
            Err(ChainAuditError::MissingManifestEntry(
                "missing.txt".to_string()
            ))
        );
        let duplicate = format!("{digest}  data/abc.txt\n{digest}  data/abc.txt\n");
        assert_eq!(
            expected_sha256_from_manifest(&duplicate, "data/abc.txt"),
            Err(ChainAuditError::DuplicateManifestEntry(
                "data/abc.txt".to_string()
            ))
        );
        assert!(matches!(
            expected_sha256_from_manifest("not a checksum line\n", "data/abc.txt"),
            Err(ChainAuditError::MalformedChecksumManifestLine { line: 1 })
        ));
        assert!(expected_sha256_from_manifest(&manifest, "../abc.txt").is_err());
    }

    #[test]
    fn official_desi_manifest_entries_resolve_by_exact_relative_path() {
        let manifest = concat!(
            "fde589a17a4ca0fd58371b61290ada939dd672484cb3d523adacb8565a340ced  cobaya/base/desi-bao-all/chain.1.txt\n",
            "b6ce2202c5fd11049a0de5c277b017e91aa9fbc9c25d6fdce1d166e7f0ef72b7  cobaya/base/desi-bao-all/chain.2.txt\n",
            "7c869b6928ef91134674551165d8a6528af7e4f210f54e2632d5bf8c0bdebb89  cobaya/base/desi-bao-all/chain.margestats\n"
        );
        assert_eq!(
            expected_sha256_from_manifest(manifest, "cobaya/base/desi-bao-all/chain.1.txt")
                .unwrap(),
            "fde589a17a4ca0fd58371b61290ada939dd672484cb3d523adacb8565a340ced"
        );
        assert_eq!(
            expected_sha256_from_manifest(manifest, "cobaya/base/desi-bao-all/chain.margestats")
                .unwrap(),
            "7c869b6928ef91134674551165d8a6528af7e4f210f54e2632d5bf8c0bdebb89"
        );
    }

    #[test]
    fn manifest_file_verification_uses_the_named_entry() {
        let directory = std::env::temp_dir().join(format!(
            "symthaea-cosmology-manifest-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let local = directory.join("chain.1.txt");
        std::fs::write(&local, b"abc").unwrap();
        let digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        let manifest = format!("{digest}  cobaya/base/desi-bao-all/chain.1.txt\n");
        let result = verify_sha256_file_against_manifest(
            &local,
            &manifest,
            "cobaya/base/desi-bao-all/chain.1.txt",
        )
        .unwrap();
        assert_eq!(result, (digest.to_string(), digest.to_string(), true));
        std::fs::write(&local, b"abd").unwrap();
        let result = verify_sha256_file_against_manifest(
            &local,
            &manifest,
            "cobaya/base/desi-bao-all/chain.1.txt",
        )
        .unwrap();
        assert!(!result.2);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn weighted_summary_uses_named_columns_and_documented_quantiles() {
        let chain = ChainTable::parse(Cursor::new(FIXTURE)).unwrap();
        let summary = chain.summarize("H0").unwrap();
        assert_eq!(summary.sample_rows, 3);
        assert_eq!(summary.positive_weight_rows, 3);
        assert!((summary.total_weight - 4.0).abs() < 1e-12);
        assert!((summary.weighted_mean - 68.25).abs() < 1e-12);
        assert!((summary.weighted_variance - 0.6875).abs() < 1e-12);
        assert_eq!(summary.q16, 67.0);
        assert_eq!(summary.q50, 68.0);
        assert_eq!(summary.q84, 69.0);
        assert!((summary.weight_effective_sample_size - (16.0 / 6.0)).abs() < 1e-12);
    }

    #[test]
    fn combine_requires_identical_headers_and_preserves_weighted_samples() {
        let first = ChainTable::parse(Cursor::new(FIXTURE)).unwrap();
        let second = ChainTable::parse(Cursor::new(
            "# weight minuslogpost H0 omega_m\n1 -9 70 0.33\n",
        ))
        .unwrap();
        let combined = ChainTable::combine(vec![first.clone(), second]).unwrap();
        assert_eq!(combined.rows.len(), 4);
        let summary = combined.summarize("H0").unwrap();
        assert!((summary.total_weight - 5.0).abs() < 1e-12);
        assert!((summary.weighted_mean - 68.6).abs() < 1e-12);

        let wrong_header = ChainTable::parse(Cursor::new(
            "# weight minuslogpost omega_m H0\n1 -9 0.33 70\n",
        ))
        .unwrap();
        assert_eq!(
            ChainTable::combine(vec![first, wrong_header]),
            Err(ChainAuditError::MismatchedChainHeaders)
        );
    }

    #[test]
    fn constructed_tables_cannot_bypass_validation() {
        let valid = ChainTable::parse(Cursor::new(FIXTURE)).unwrap();
        let mut malformed = valid.clone();
        malformed.rows[0].pop();
        assert!(matches!(
            malformed.summarize("H0"),
            Err(ChainAuditError::WrongColumnCount { .. })
        ));
        assert!(matches!(
            ChainTable::combine(vec![malformed, valid.clone()]),
            Err(ChainAuditError::WrongColumnCount { .. })
        ));

        let mut non_finite = valid;
        non_finite.rows[0][2] = f64::INFINITY;
        assert!(matches!(
            non_finite.summarize("H0"),
            Err(ChainAuditError::NonFiniteValue { .. })
        ));
    }

    #[test]
    fn malformed_rows_are_rejected_instead_of_skipped() {
        let fixture = "# weight minuslogpost H0\n1 -10 67\n2 -11\n";
        assert!(matches!(
            ChainTable::parse(Cursor::new(fixture)),
            Err(ChainAuditError::WrongColumnCount { line: 3, .. })
        ));
    }

    #[test]
    fn duplicate_columns_and_non_finite_values_fail_closed() {
        let duplicate = "# weight minuslogpost H0 H0\n1 -10 67 68\n";
        assert!(matches!(
            ChainTable::parse(Cursor::new(duplicate)),
            Err(ChainAuditError::DuplicateColumn(_))
        ));

        let non_finite = "# weight minuslogpost H0\n1 -10 NaN\n";
        assert!(matches!(
            ChainTable::parse(Cursor::new(non_finite)),
            Err(ChainAuditError::NonFiniteValue { .. })
        ));
    }

    #[test]
    fn negative_and_all_zero_weights_fail_closed() {
        let negative = "# weight minuslogpost H0\n-1 -10 67\n";
        assert!(matches!(
            ChainTable::parse(Cursor::new(negative)),
            Err(ChainAuditError::NegativeWeight { line: 2 })
        ));

        let zero = "# weight minuslogpost H0\n0 -10 67\n0 -11 68\n";
        assert_eq!(
            ChainTable::parse(Cursor::new(zero)),
            Err(ChainAuditError::ZeroTotalWeight)
        );
    }

    #[test]
    fn absent_header_or_requested_parameter_is_explicit() {
        assert_eq!(
            ChainTable::parse(Cursor::new("1 -10 67\n")),
            Err(ChainAuditError::MissingHeader)
        );
        let chain = ChainTable::parse(Cursor::new(FIXTURE)).unwrap();
        assert_eq!(
            chain.summarize("not_a_parameter"),
            Err(ChainAuditError::UnknownColumn(
                "not_a_parameter".to_string()
            ))
        );
    }
}
