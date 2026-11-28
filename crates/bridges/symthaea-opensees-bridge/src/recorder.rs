// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Strict parser for explicitly declared, whitespace-delimited OpenSees recorder tables.
//!
//! This module parses numeric samples only. It does not infer recorder identity,
//! units, model provenance, solver convergence, or engineering qualification.
//! Callers must bind those properties to the exact request and raw artifact.

use std::error::Error;
use std::fmt;

/// One scalar observation from a recorder table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecorderSample {
    /// Solver time from the declared time column.
    pub time: f64,
    /// Scalar response from the declared value column.
    pub value: f64,
}

/// A fail-closed error while parsing a declared recorder table.
#[derive(Debug, Clone, PartialEq)]
pub enum RecorderParseError {
    /// The table must contain at least a time column and a response column.
    InvalidExpectedColumnCount {
        expected_columns: usize,
    },
    /// The declared time/value columns are equal or outside the table width.
    InvalidColumnSelection {
        expected_columns: usize,
        time_column: usize,
        value_column: usize,
    },
    /// A data row does not have the exact declared number of columns.
    WrongColumnCount {
        line: usize,
        expected: usize,
        actual: usize,
    },
    /// A token in a declared numeric table could not be parsed as a number.
    InvalidNumber {
        line: usize,
        column: usize,
        token: String,
    },
    /// NaN and infinity are not admissible recorder values.
    NonFiniteNumber {
        line: usize,
        column: usize,
        token: String,
    },
    /// Recorder time must be strictly increasing; duplicates and reordering fail.
    NonIncreasingTime {
        line: usize,
        previous: f64,
        current: f64,
    },
    /// No numeric samples were present after blank/comment lines were removed.
    NoSamples,
}

impl fmt::Display for RecorderParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidExpectedColumnCount { expected_columns } => write!(
                formatter,
                "expected column count must be at least 2, got {expected_columns}"
            ),
            Self::InvalidColumnSelection {
                expected_columns,
                time_column,
                value_column,
            } => write!(
                formatter,
                "time/value columns ({time_column}, {value_column}) must be distinct and below declared width {expected_columns}"
            ),
            Self::WrongColumnCount {
                line,
                expected,
                actual,
            } => write!(
                formatter,
                "line {line}: expected exactly {expected} columns, got {actual}"
            ),
            Self::InvalidNumber {
                line,
                column,
                token,
            } => write!(
                formatter,
                "line {line}, column {column}: invalid numeric token {token:?}"
            ),
            Self::NonFiniteNumber {
                line,
                column,
                token,
            } => write!(
                formatter,
                "line {line}, column {column}: non-finite numeric token {token:?}"
            ),
            Self::NonIncreasingTime {
                line,
                previous,
                current,
            } => write!(
                formatter,
                "line {line}: recorder time {current} is not greater than previous time {previous}"
            ),
            Self::NoSamples => formatter.write_str("recorder table contains no numeric samples"),
        }
    }
}

impl Error for RecorderParseError {}

/// Parse one scalar column from a whitespace-delimited numeric recorder table.
///
/// The caller declares the exact table width and zero-based time/value columns.
/// Every token in each data row is parsed and required to be finite, not merely
/// the selected columns. Blank lines and full-line comments beginning with #
/// are ignored. Header rows, comma-separated tables, ragged rows, duplicate or
/// decreasing times, and non-finite values are rejected.
///
/// This deliberately small grammar matches an explicit recorder profile; it is
/// not an auto-detecting parser for arbitrary OpenSees output.
pub fn parse_scalar_time_series(
    input: &str,
    expected_columns: usize,
    time_column: usize,
    value_column: usize,
) -> Result<Vec<RecorderSample>, RecorderParseError> {
    if expected_columns < 2 {
        return Err(RecorderParseError::InvalidExpectedColumnCount {
            expected_columns,
        });
    }

    if time_column >= expected_columns
        || value_column >= expected_columns
        || time_column == value_column
    {
        return Err(RecorderParseError::InvalidColumnSelection {
            expected_columns,
            time_column,
            value_column,
        });
    }

    let mut samples = Vec::new();
    let mut previous_time: Option<f64> = None;

    for (zero_based_line, raw_line) in input.lines().enumerate() {
        let line_number = zero_based_line + 1;
        let line = raw_line.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() != expected_columns {
            return Err(RecorderParseError::WrongColumnCount {
                line: line_number,
                expected: expected_columns,
                actual: tokens.len(),
            });
        }

        let mut values = Vec::with_capacity(expected_columns);
        for (zero_based_column, token) in tokens.iter().enumerate() {
            let column_number = zero_based_column + 1;
            let value = token.parse::<f64>().map_err(|_| {
                RecorderParseError::InvalidNumber {
                    line: line_number,
                    column: column_number,
                    token: (*token).to_owned(),
                }
            })?;

            if !value.is_finite() {
                return Err(RecorderParseError::NonFiniteNumber {
                    line: line_number,
                    column: column_number,
                    token: (*token).to_owned(),
                });
            }
            values.push(value);
        }

        let time = values[time_column];
        if let Some(previous) = previous_time {
            if time <= previous {
                return Err(RecorderParseError::NonIncreasingTime {
                    line: line_number,
                    previous,
                    current: time,
                });
            }
        }

        samples.push(RecorderSample {
            time,
            value: values[value_column],
        });
        previous_time = Some(time);
    }

    if samples.is_empty() {
        return Err(RecorderParseError::NoSamples);
    }

    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_declared_two_column_time_series() {
        let input = "# time displacement\n0.0 0.000\n0.1 0.012\n0.2 -0.004\n";
        let samples = parse_scalar_time_series(input, 2, 0, 1).unwrap();

        assert_eq!(
            samples,
            vec![
                RecorderSample {
                    time: 0.0,
                    value: 0.0
                },
                RecorderSample {
                    time: 0.1,
                    value: 0.012
                },
                RecorderSample {
                    time: 0.2,
                    value: -0.004
                }
            ]
        );
    }

    #[test]
    fn selects_declared_value_column_from_multi_column_recorder() {
        let input = "0.0 10.0 20.0\n0.5 11.0 21.5\n";
        let samples = parse_scalar_time_series(input, 3, 0, 2).unwrap();

        assert_eq!(samples[1].time, 0.5);
        assert_eq!(samples[1].value, 21.5);
    }

    #[test]
    fn rejects_ragged_rows() {
        let error = parse_scalar_time_series("0.0 1.0\n0.1 2.0 3.0\n", 2, 0, 1)
            .unwrap_err();

        assert_eq!(
            error,
            RecorderParseError::WrongColumnCount {
                line: 2,
                expected: 2,
                actual: 3
            }
        );
    }

    #[test]
    fn rejects_non_numeric_tokens() {
        let error = parse_scalar_time_series("0.0 1.0\n0.1 nope\n", 2, 0, 1)
            .unwrap_err();

        assert_eq!(
            error,
            RecorderParseError::InvalidNumber {
                line: 2,
                column: 2,
                token: "nope".into()
            }
        );
    }

    #[test]
    fn rejects_non_finite_values_even_in_unselected_columns() {
        let error = parse_scalar_time_series("0.0 1.0 NaN\n", 3, 0, 1)
            .unwrap_err();

        assert_eq!(
            error,
            RecorderParseError::NonFiniteNumber {
                line: 1,
                column: 3,
                token: "NaN".into()
            }
        );
    }

    #[test]
    fn rejects_duplicate_and_decreasing_times() {
        let duplicate = parse_scalar_time_series("0.0 1.0\n0.0 2.0\n", 2, 0, 1)
            .unwrap_err();
        assert!(matches!(
            duplicate,
            RecorderParseError::NonIncreasingTime { .. }
        ));

        let decreasing = parse_scalar_time_series("0.2 1.0\n0.1 2.0\n", 2, 0, 1)
            .unwrap_err();
        assert!(matches!(
            decreasing,
            RecorderParseError::NonIncreasingTime { .. }
        ));
    }

    #[test]
    fn rejects_invalid_column_declarations() {
        assert_eq!(
            parse_scalar_time_series("0.0 1.0\n", 1, 0, 0).unwrap_err(),
            RecorderParseError::InvalidExpectedColumnCount {
                expected_columns: 1
            }
        );
        assert_eq!(
            parse_scalar_time_series("0.0 1.0\n", 2, 0, 2).unwrap_err(),
            RecorderParseError::InvalidColumnSelection {
                expected_columns: 2,
                time_column: 0,
                value_column: 2
            }
        );
        assert_eq!(
            parse_scalar_time_series("0.0 1.0\n", 2, 0, 0).unwrap_err(),
            RecorderParseError::InvalidColumnSelection {
                expected_columns: 2,
                time_column: 0,
                value_column: 0
            }
        );
    }

    #[test]
    fn rejects_empty_and_comment_only_tables() {
        assert_eq!(
            parse_scalar_time_series("", 2, 0, 1).unwrap_err(),
            RecorderParseError::NoSamples
        );
        assert_eq!(
            parse_scalar_time_series("# recorder output\n\n", 2, 0, 1).unwrap_err(),
            RecorderParseError::NoSamples
        );
    }
}
