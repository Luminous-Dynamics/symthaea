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
use std::io::BufRead;

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
    NumericalOverflow,
}

impl fmt::Display for ChainAuditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(message) => write!(f, "I/O error: {message}"),
            Self::MissingHeader => f.write_str("missing Cobaya header beginning with '# weight minuslogpost'"),
            Self::InvalidHeader(message) => write!(f, "invalid chain header: {message}"),
            Self::DuplicateColumn(name) => write!(f, "duplicate chain column: {name}"),
            Self::MissingSamples => f.write_str("chain contains no sample rows"),
            Self::WrongColumnCount { line, expected, actual } => {
                write!(f, "line {line}: expected {expected} columns, found {actual}")
            }
            Self::InvalidNumber { line, column, value } => {
                write!(f, "line {line}, column {column}: invalid number {value:?}")
            }
            Self::NonFiniteValue { line, column } => {
                write!(f, "line {line}, column {column}: non-finite value")
            }
            Self::NegativeWeight { line } => write!(f, "line {line}: sample weight must be non-negative"),
            Self::ZeroTotalWeight => f.write_str("chain total weight must be finite and positive"),
            Self::UnknownColumn(name) => write!(f, "unknown chain column: {name}"),
            Self::NumericalOverflow => f.write_str("chain summary overflowed floating-point arithmetic"),
        }
    }
}

impl std::error::Error for ChainAuditError {}

impl ChainTable {
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
                    let parsed = candidate.iter().map(|value| (*value).to_string()).collect::<Vec<_>>();
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
                let value = token.parse::<f64>().map_err(|_| ChainAuditError::InvalidNumber {
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
            return Err(ChainAuditError::InvalidHeader("empty column name".to_string()));
        }
        if !unique.insert(column.as_str()) {
            return Err(ChainAuditError::DuplicateColumn(column.clone()));
        }
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
            Err(ChainAuditError::UnknownColumn("not_a_parameter".to_string()))
        );
    }
}
