//! Explicit binary-grid transforms and spatial traversal for encoding tests.
//!
//! This consumes an already constructed binary mask. It does not segment
//! aerial images or decide which crop regions should be represented as bits.

use crate::decoder::{decode, DecodeError, DecodedPayload, DecoderSpec};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridTransform {
    Identity,
    Rotate90Clockwise,
    Rotate180,
    Rotate270Clockwise,
    ReflectHorizontal,
    ReflectVertical,
    Transpose,
    AntiTranspose,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraversalOrder {
    RowMajor,
    ColumnMajor,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GridError {
    EmptyDimensions,
    DimensionOverflow,
    CellCountMismatch { expected: usize, provided: usize },
}

impl std::fmt::Display for GridError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyDimensions => write!(f, "grid dimensions must be non-zero"),
            Self::DimensionOverflow => write!(f, "grid dimensions overflow usize"),
            Self::CellCountMismatch { expected, provided } => write!(
                f,
                "grid requires {expected} cells but received {provided}"
            ),
        }
    }
}

impl std::error::Error for GridError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryGrid {
    rows: usize,
    columns: usize,
    cells: Vec<bool>,
}

impl BinaryGrid {
    pub fn new(rows: usize, columns: usize, cells: Vec<bool>) -> Result<Self, GridError> {
        if rows == 0 || columns == 0 {
            return Err(GridError::EmptyDimensions);
        }
        let expected = rows
            .checked_mul(columns)
            .ok_or(GridError::DimensionOverflow)?;
        if cells.len() != expected {
            return Err(GridError::CellCountMismatch {
                expected,
                provided: cells.len(),
            });
        }
        Ok(Self {
            rows,
            columns,
            cells,
        })
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Flatten the grid under one explicitly selected dihedral transform and
    /// traversal order. The function does not search for a favorable reading.
    pub fn flatten(
        &self,
        transform: GridTransform,
        traversal: TraversalOrder,
    ) -> Vec<bool> {
        let (output_rows, output_columns) = match transform {
            GridTransform::Rotate90Clockwise
            | GridTransform::Rotate270Clockwise
            | GridTransform::Transpose
            | GridTransform::AntiTranspose => (self.columns, self.rows),
            GridTransform::Identity
            | GridTransform::Rotate180
            | GridTransform::ReflectHorizontal
            | GridTransform::ReflectVertical => (self.rows, self.columns),
        };

        let source_coordinate = |row: usize, column: usize| -> (usize, usize) {
            match transform {
                GridTransform::Identity => (row, column),
                GridTransform::Rotate90Clockwise => {
                    (self.rows - 1 - column, row)
                }
                GridTransform::Rotate180 => {
                    (self.rows - 1 - row, self.columns - 1 - column)
                }
                GridTransform::Rotate270Clockwise => {
                    (column, self.columns - 1 - row)
                }
                GridTransform::ReflectHorizontal => {
                    (self.rows - 1 - row, column)
                }
                GridTransform::ReflectVertical => {
                    (row, self.columns - 1 - column)
                }
                GridTransform::Transpose => (column, row),
                GridTransform::AntiTranspose => (
                    self.rows - 1 - column,
                    self.columns - 1 - row,
                ),
            }
        };

        let mut output = Vec::with_capacity(self.cells.len());
        match traversal {
            TraversalOrder::RowMajor => {
                for row in 0..output_rows {
                    for column in 0..output_columns {
                        let (source_row, source_column) = source_coordinate(row, column);
                        output.push(self.cells[source_row * self.columns + source_column]);
                    }
                }
            }
            TraversalOrder::ColumnMajor => {
                for column in 0..output_columns {
                    for row in 0..output_rows {
                        let (source_row, source_column) = source_coordinate(row, column);
                        output.push(self.cells[source_row * self.columns + source_column]);
                    }
                }
            }
        }
        output
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GridDecoderSpec {
    /// Unique identifier for this transform + traversal + decoder attempt.
    pub id: String,
    pub transform: GridTransform,
    pub traversal: TraversalOrder,
    pub decoder: DecoderSpec,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GridDecodeOutcome {
    pub spec: GridDecoderSpec,
    pub result: Result<DecodedPayload, DecodeError>,
}

/// Execute every supplied grid interpretation and preserve all outcomes.
/// The caller must freeze this complete list before looking at a challenge case.
pub fn run_grid_tournament(
    grid: &BinaryGrid,
    specs: &[GridDecoderSpec],
) -> Vec<GridDecodeOutcome> {
    specs
        .iter()
        .cloned()
        .map(|spec| {
            let bits = grid.flatten(spec.transform, spec.traversal);
            GridDecodeOutcome {
                result: decode(&bits, &spec.decoder),
                spec,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::{encode_ascii7, BitOrder, DecoderSpec};

    #[test]
    fn identity_row_major_recovers_known_grid_payload() {
        let bits = encode_ascii7("HELLO", BitOrder::MostSignificantFirst)
            .expect("known ASCII positive control");
        let grid = BinaryGrid::new(5, 7, bits.clone()).expect("valid grid");
        let spec = GridDecoderSpec {
            id: "hello-identity-row-major".to_string(),
            transform: GridTransform::Identity,
            traversal: TraversalOrder::RowMajor,
            decoder: DecoderSpec::ascii7("ascii7-msb", BitOrder::MostSignificantFirst),
        };
        let outcomes = run_grid_tournament(&grid, &[spec]);

        assert_eq!(outcomes.len(), 1);
        assert_eq!(
            outcomes[0].result.as_ref().expect("decode passes").bytes,
            b"HELLO"
        );
        assert_eq!(grid.flatten(GridTransform::Identity, TraversalOrder::RowMajor), bits);
    }

    #[test]
    fn transpose_row_major_equals_original_column_major() {
        let cells = vec![true, false, true, false, true, false];
        let grid = BinaryGrid::new(2, 3, cells).expect("valid grid");

        assert_eq!(
            grid.flatten(GridTransform::Transpose, TraversalOrder::RowMajor),
            grid.flatten(GridTransform::Identity, TraversalOrder::ColumnMajor)
        );
    }

    #[test]
    fn quarter_turn_changes_output_dimensions_but_not_cell_count() {
        let grid = BinaryGrid::new(
            2,
            3,
            vec![true, false, true, false, true, false],
        )
        .expect("valid grid");
        let rotated = grid.flatten(
            GridTransform::Rotate90Clockwise,
            TraversalOrder::RowMajor,
        );

        assert_eq!(rotated.len(), 6);
        assert_eq!(grid.rows(), 2);
        assert_eq!(grid.columns(), 3);
    }

    #[test]
    fn invalid_grid_dimensions_and_cell_counts_fail_closed() {
        assert_eq!(
            BinaryGrid::new(0, 3, vec![]),
            Err(GridError::EmptyDimensions)
        );
        assert_eq!(
            BinaryGrid::new(2, 3, vec![true; 5]),
            Err(GridError::CellCountMismatch {
                expected: 6,
                provided: 5
            })
        );
        assert_eq!(
            BinaryGrid::new(usize::MAX, 2, vec![]),
            Err(GridError::DimensionOverflow)
        );
    }

    #[test]
    fn grid_tournament_preserves_every_declared_transform() {
        let bits = encode_ascii7("A", BitOrder::MostSignificantFirst)
            .expect("known ASCII positive control");
        let grid = BinaryGrid::new(1, 7, bits).expect("valid grid");
        let specs = [
            GridDecoderSpec {
                id: "identity".to_string(),
                transform: GridTransform::Identity,
                traversal: TraversalOrder::RowMajor,
                decoder: DecoderSpec::ascii7("ascii7-msb", BitOrder::MostSignificantFirst),
            },
            GridDecoderSpec {
                id: "vertical-reflection".to_string(),
                transform: GridTransform::ReflectHorizontal,
                traversal: TraversalOrder::RowMajor,
                decoder: DecoderSpec::ascii7("ascii7-msb-reflected", BitOrder::MostSignificantFirst),
            },
        ];

        let outcomes = run_grid_tournament(&grid, &specs);
        assert_eq!(outcomes.len(), 2);
        assert_eq!(outcomes[0].spec.id, "identity");
        assert_eq!(outcomes[1].spec.id, "vertical-reflection");
    }
}

    #[test]
    fn every_dihedral_transform_has_the_expected_inverse() {
        let original = BinaryGrid::new(
            2,
            3,
            vec![true, false, true, false, true, false],
        )
        .expect("valid non-square grid");

        let rotated_90 = BinaryGrid::new(
            3,
            2,
            original.flatten(
                GridTransform::Rotate90Clockwise,
                TraversalOrder::RowMajor,
            ),
        )
        .expect("valid rotated grid");
        assert_eq!(
            rotated_90.flatten(
                GridTransform::Rotate270Clockwise,
                TraversalOrder::RowMajor,
            ),
            original.flatten(GridTransform::Identity, TraversalOrder::RowMajor)
        );

        let rotated_180 = BinaryGrid::new(
            2,
            3,
            original.flatten(GridTransform::Rotate180, TraversalOrder::RowMajor),
        )
        .expect("valid rotated grid");
        assert_eq!(
            rotated_180.flatten(GridTransform::Rotate180, TraversalOrder::RowMajor),
            original.flatten(GridTransform::Identity, TraversalOrder::RowMajor)
        );

        for transform in [
            GridTransform::ReflectHorizontal,
            GridTransform::ReflectVertical,
            GridTransform::Transpose,
            GridTransform::AntiTranspose,
        ] {
            let (rows, columns) = match transform {
                GridTransform::Transpose | GridTransform::AntiTranspose => (3, 2),
                _ => (2, 3),
            };
            let once = BinaryGrid::new(
                rows,
                columns,
                original.flatten(transform, TraversalOrder::RowMajor),
            )
            .expect("valid transformed grid");
            assert_eq!(
                once.flatten(transform, TraversalOrder::RowMajor),
                original.flatten(GridTransform::Identity, TraversalOrder::RowMajor)
            );
        }
    }
