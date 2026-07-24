// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Checked rectangular block interleaving.
//!
//! Equal-length frames are laid out row-major and transmitted column-major.
//! A contiguous channel burst of at most `rows` symbols is therefore dispersed
//! across distinct frames, allowing each component decoder to see at most one
//! symbol from that burst.

use std::fmt;

/// Invalid interleaver dimensions, buffers, or coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterleaverError {
    ZeroRows,
    ZeroColumns,
    SizeOverflow { rows: usize, columns: usize },
    InputLengthMismatch { expected: usize, actual: usize },
    OutputLengthMismatch { expected: usize, actual: usize },
    PositionOutOfRange { position: usize, symbols: usize },
    FrameLengthMismatch {
        frame: usize,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for InterleaverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroRows => write!(f, "block interleaver requires at least one row"),
            Self::ZeroColumns => write!(f, "block interleaver requires at least one column"),
            Self::SizeOverflow { rows, columns } => write!(
                f,
                "block interleaver dimensions {rows} x {columns} overflow usize"
            ),
            Self::InputLengthMismatch { expected, actual } => write!(
                f,
                "interleaver input length {actual} does not match required length {expected}"
            ),
            Self::OutputLengthMismatch { expected, actual } => write!(
                f,
                "interleaver output length {actual} does not match required length {expected}"
            ),
            Self::PositionOutOfRange { position, symbols } => write!(
                f,
                "interleaver position {position} is outside {symbols} symbols"
            ),
            Self::FrameLengthMismatch {
                frame,
                expected,
                actual,
            } => write!(
                f,
                "frame {frame} length {actual} does not match required length {expected}"
            ),
        }
    }
}

impl std::error::Error for InterleaverError {}

/// Rectangular row-to-column block permutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockInterleaver {
    rows: usize,
    columns: usize,
    symbols: usize,
}

impl BlockInterleaver {
    /// Construct a checked `rows x columns` permutation.
    pub fn new(rows: usize, columns: usize) -> Result<Self, InterleaverError> {
        if rows == 0 {
            return Err(InterleaverError::ZeroRows);
        }
        if columns == 0 {
            return Err(InterleaverError::ZeroColumns);
        }
        let symbols = rows
            .checked_mul(columns)
            .ok_or(InterleaverError::SizeOverflow { rows, columns })?;
        Ok(Self {
            rows,
            columns,
            symbols,
        })
    }

    #[must_use]
    pub const fn rows(self) -> usize {
        self.rows
    }

    #[must_use]
    pub const fn columns(self) -> usize {
        self.columns
    }

    #[must_use]
    pub const fn symbols(self) -> usize {
        self.symbols
    }

    /// Map a row-major source position to its column-major wire position.
    pub fn source_to_interleaved(self, position: usize) -> Result<usize, InterleaverError> {
        self.validate_position(position)?;
        let row = position / self.columns;
        let column = position % self.columns;
        Ok(column * self.rows + row)
    }

    /// Map a column-major wire position back to its row-major source position.
    pub fn interleaved_to_source(self, position: usize) -> Result<usize, InterleaverError> {
        self.validate_position(position)?;
        let column = position / self.rows;
        let row = position % self.rows;
        Ok(row * self.columns + column)
    }

    /// Allocate and return the column-major transmission order.
    pub fn interleave<T: Clone>(self, input: &[T]) -> Result<Vec<T>, InterleaverError> {
        self.validate_input(input.len())?;
        let mut output = Vec::with_capacity(self.symbols);
        for column in 0..self.columns {
            for row in 0..self.rows {
                output.push(input[row * self.columns + column].clone());
            }
        }
        Ok(output)
    }

    /// Allocate and restore row-major source order.
    pub fn deinterleave<T: Clone>(self, input: &[T]) -> Result<Vec<T>, InterleaverError> {
        self.validate_input(input.len())?;
        let mut output = Vec::with_capacity(self.symbols);
        for source_position in 0..self.symbols {
            let wire_position = self.source_to_interleaved(source_position)?;
            output.push(input[wire_position].clone());
        }
        Ok(output)
    }

    /// Interleave into a caller-owned exactly-sized buffer.
    pub fn interleave_into<T: Clone>(
        self,
        input: &[T],
        output: &mut [T],
    ) -> Result<(), InterleaverError> {
        self.validate_input(input.len())?;
        self.validate_output(output.len())?;
        for source_position in 0..self.symbols {
            let wire_position = self.source_to_interleaved(source_position)?;
            output[wire_position] = input[source_position].clone();
        }
        Ok(())
    }

    /// Deinterleave into a caller-owned exactly-sized buffer.
    pub fn deinterleave_into<T: Clone>(
        self,
        input: &[T],
        output: &mut [T],
    ) -> Result<(), InterleaverError> {
        self.validate_input(input.len())?;
        self.validate_output(output.len())?;
        for source_position in 0..self.symbols {
            let wire_position = self.source_to_interleaved(source_position)?;
            output[source_position] = input[wire_position].clone();
        }
        Ok(())
    }

    /// Interleave equal-length frames without exposing the flattened layout.
    pub fn interleave_frames<T: Clone>(frames: &[&[T]]) -> Result<Vec<T>, InterleaverError> {
        if frames.is_empty() {
            return Err(InterleaverError::ZeroRows);
        }
        let columns = frames[0].len();
        let interleaver = Self::new(frames.len(), columns)?;
        let mut flattened = Vec::with_capacity(interleaver.symbols);
        for (frame, symbols) in frames.iter().enumerate() {
            if symbols.len() != columns {
                return Err(InterleaverError::FrameLengthMismatch {
                    frame,
                    expected: columns,
                    actual: symbols.len(),
                });
            }
            flattened.extend_from_slice(symbols);
        }
        interleaver.interleave(&flattened)
    }

    /// Restore equal-length row frames from a column-major transmission.
    pub fn deinterleave_frames<T: Clone>(self, input: &[T]) -> Result<Vec<Vec<T>>, InterleaverError> {
        let flattened = self.deinterleave(input)?;
        Ok(flattened
            .chunks(self.columns)
            .map(|frame| frame.to_vec())
            .collect())
    }

    fn validate_input(self, actual: usize) -> Result<(), InterleaverError> {
        if actual != self.symbols {
            return Err(InterleaverError::InputLengthMismatch {
                expected: self.symbols,
                actual,
            });
        }
        Ok(())
    }

    fn validate_output(self, actual: usize) -> Result<(), InterleaverError> {
        if actual != self.symbols {
            return Err(InterleaverError::OutputLengthMismatch {
                expected: self.symbols,
                actual,
            });
        }
        Ok(())
    }

    fn validate_position(self, position: usize) -> Result<(), InterleaverError> {
        if position >= self.symbols {
            return Err(InterleaverError::PositionOutOfRange {
                position,
                symbols: self.symbols,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectangular_permutations_round_trip() {
        for rows in 1..=12 {
            for columns in 1..=31 {
                let interleaver = BlockInterleaver::new(rows, columns).unwrap();
                let source = (0..interleaver.symbols()).collect::<Vec<_>>();
                let wire = interleaver.interleave(&source).unwrap();
                assert_eq!(interleaver.deinterleave(&wire).unwrap(), source);
                for position in 0..interleaver.symbols() {
                    let wire_position = interleaver.source_to_interleaved(position).unwrap();
                    assert_eq!(
                        interleaver.interleaved_to_source(wire_position).unwrap(),
                        position
                    );
                }
            }
        }
    }

    #[test]
    fn one_row_sized_burst_hits_each_frame_once() {
        let interleaver = BlockInterleaver::new(8, 31).unwrap();
        let burst_start = 47;
        let mut hits_per_frame = [0usize; 8];
        for wire_position in burst_start..burst_start + interleaver.rows() {
            let source_position = interleaver.interleaved_to_source(wire_position).unwrap();
            hits_per_frame[source_position / interleaver.columns()] += 1;
        }
        assert_eq!(hits_per_frame, [1; 8]);
    }

    #[test]
    fn equal_frame_helpers_preserve_boundaries() {
        let a = [0u8, 1, 2, 3];
        let b = [10u8, 11, 12, 13];
        let c = [20u8, 21, 22, 23];
        let wire = BlockInterleaver::interleave_frames(&[&a, &b, &c]).unwrap();
        assert_eq!(wire, [0, 10, 20, 1, 11, 21, 2, 12, 22, 3, 13, 23]);
        let frames = BlockInterleaver::new(3, 4)
            .unwrap()
            .deinterleave_frames(&wire)
            .unwrap();
        assert_eq!(frames, vec![a.to_vec(), b.to_vec(), c.to_vec()]);
    }

    #[test]
    fn invalid_dimensions_lengths_and_frames_fail_closed() {
        assert_eq!(BlockInterleaver::new(0, 2), Err(InterleaverError::ZeroRows));
        assert_eq!(BlockInterleaver::new(2, 0), Err(InterleaverError::ZeroColumns));
        let interleaver = BlockInterleaver::new(2, 3).unwrap();
        assert_eq!(
            interleaver.interleave(&[0u8; 5]),
            Err(InterleaverError::InputLengthMismatch {
                expected: 6,
                actual: 5,
            })
        );
        assert_eq!(
            BlockInterleaver::interleave_frames(&[&[1u8, 2][..], &[3u8][..]]),
            Err(InterleaverError::FrameLengthMismatch {
                frame: 1,
                expected: 2,
                actual: 1,
            })
        );
    }
}
