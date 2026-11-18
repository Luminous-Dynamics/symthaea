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

/// Result of one deterministic row/column-margin-preserving null shuffle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NullGridResult {
    pub grid: BinaryGrid,
    pub seed: u64,
    pub attempted_swaps: usize,
    pub accepted_swaps: usize,
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        value
    }

    fn index(&mut self, upper: usize) -> usize {
        ((self.next_u64() % upper as u64) as usize).min(upper - 1)
    }

    fn distinct_pair(&mut self, upper: usize) -> (usize, usize) {
        let first = self.index(upper);
        let mut second = self.index(upper - 1);
        if second >= first {
            second += 1;
        }
        (first, second)
    }
}

/// Generate a deterministic binary-grid null by applying 2x2 switches.
///
/// A switch is accepted only for a checkerboard block:
///
/// `1 0`       `0 1`
/// `0 1`  <->  `1 0`
///
/// Each accepted switch preserves every row sum, every column sum, and the
/// global occupied-cell count. It does **not** preserve connected components,
/// perimeter, local texture or higher-order spatial structure. It is one
/// matched-null family, not a universal null model.
///
/// XorShift64 is used for reproducible test/control generation, not security.
pub fn margin_preserving_shuffle(
    grid: &BinaryGrid,
    seed: u64,
    attempts: usize,
) -> NullGridResult {
    let mut cells = grid.cells.clone();
    let mut random = XorShift64::new(seed);
    let mut attempted_swaps = 0;
    let mut accepted_swaps = 0;

    if grid.rows >= 2 && grid.columns >= 2 {
        for _ in 0..attempts {
            attempted_swaps += 1;
            let (row_a, row_b) = random.distinct_pair(grid.rows);
            let (column_a, column_b) = random.distinct_pair(grid.columns);

            let top_left = row_a * grid.columns + column_a;
            let top_right = row_a * grid.columns + column_b;
            let bottom_left = row_b * grid.columns + column_a;
            let bottom_right = row_b * grid.columns + column_b;

            if cells[top_left] == cells[bottom_right]
                && cells[top_right] == cells[bottom_left]
                && cells[top_left] != cells[top_right]
            {
                cells[top_left] = !cells[top_left];
                cells[top_right] = !cells[top_right];
                cells[bottom_left] = !cells[bottom_left];
                cells[bottom_right] = !cells[bottom_right];
                accepted_swaps += 1;
            }
        }
    }

    NullGridResult {
        grid: BinaryGrid {
            rows: grid.rows,
            columns: grid.columns,
            cells,
        },
        seed,
        attempted_swaps,
        accepted_swaps,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestError {
    EmptyProtocolId,
    ZeroVersion,
    EmptySpecificationList,
    EmptySpecificationId,
    DuplicateSpecificationId(String),
    EmptyDecoderId { specification_id: String },
}

impl std::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyProtocolId => write!(f, "protocol id must not be empty"),
            Self::ZeroVersion => write!(f, "protocol version must be greater than zero"),
            Self::EmptySpecificationList => {
                write!(f, "decoder manifest must contain at least one specification")
            }
            Self::EmptySpecificationId => {
                write!(f, "decoder specification id must not be empty")
            }
            Self::DuplicateSpecificationId(id) => {
                write!(f, "duplicate decoder specification id: {id}")
            }
            Self::EmptyDecoderId { specification_id } => write!(
                f,
                "decoder id for specification {specification_id} must not be empty"
            ),
        }
    }
}

impl std::error::Error for ManifestError {}

/// Immutable-in-use declaration of the complete decoder search family.
///
/// The manifest does not prove when it was created. Persist its canonical
/// representation, timestamp and digest in the external evidence receipt
/// before inspecting a challenge case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GridDecoderManifest {
    protocol_id: String,
    version: u32,
    specifications: Vec<GridDecoderSpec>,
}

impl GridDecoderManifest {
    pub fn new(
        protocol_id: impl Into<String>,
        version: u32,
        specifications: Vec<GridDecoderSpec>,
    ) -> Result<Self, ManifestError> {
        let protocol_id = protocol_id.into();
        if protocol_id.trim().is_empty() {
            return Err(ManifestError::EmptyProtocolId);
        }
        if version == 0 {
            return Err(ManifestError::ZeroVersion);
        }
        if specifications.is_empty() {
            return Err(ManifestError::EmptySpecificationList);
        }

        let mut identifiers = std::collections::HashSet::with_capacity(specifications.len());
        for specification in &specifications {
            let id = specification.id.trim();
            if id.is_empty() {
                return Err(ManifestError::EmptySpecificationId);
            }
            if !identifiers.insert(id.to_owned()) {
                return Err(ManifestError::DuplicateSpecificationId(id.to_owned()));
            }
            if specification.decoder.id.trim().is_empty() {
                return Err(ManifestError::EmptyDecoderId {
                    specification_id: id.to_owned(),
                });
            }
        }

        Ok(Self {
            protocol_id,
            version,
            specifications,
        })
    }

    pub fn protocol_id(&self) -> &str {
        &self.protocol_id
    }

    pub const fn version(&self) -> u32 {
        self.version
    }

    pub fn specifications(&self) -> &[GridDecoderSpec] {
        &self.specifications
    }

    pub fn run(&self, grid: &BinaryGrid) -> Vec<GridDecodeOutcome> {
        run_grid_tournament(grid, &self.specifications)
    }
}

/// Build a standard 32-way D4 × traversal × ASCII-7 bit-order search family.
///
/// This is a convenient starting protocol, not a universal decoder. Dimensions,
/// segmentation, codebooks, and other transforms are intentionally not searched.
/// Freeze and record the returned manifest before examining challenge cases.
pub fn standard_d4_ascii7_manifest(
    protocol_id: impl Into<String>,
    version: u32,
) -> Result<GridDecoderManifest, ManifestError> {
    let transforms = [
        (GridTransform::Identity, "identity"),
        (GridTransform::Rotate90Clockwise, "rotate90_cw"),
        (GridTransform::Rotate180, "rotate180"),
        (GridTransform::Rotate270Clockwise, "rotate270_cw"),
        (GridTransform::ReflectHorizontal, "reflect_horizontal"),
        (GridTransform::ReflectVertical, "reflect_vertical"),
        (GridTransform::Transpose, "transpose"),
        (GridTransform::AntiTranspose, "anti_transpose"),
    ];
    let traversals = [
        (TraversalOrder::RowMajor, "row_major"),
        (TraversalOrder::ColumnMajor, "column_major"),
    ];
    let bit_orders = [
        (crate::decoder::BitOrder::MostSignificantFirst, "msb"),
        (crate::decoder::BitOrder::LeastSignificantFirst, "lsb"),
    ];

    let mut specifications = Vec::with_capacity(32);
    for (transform, transform_name) in transforms {
        for (traversal, traversal_name) in traversals {
            for (bit_order, bit_order_name) in bit_orders {
                let id = format!("{transform_name}-{traversal_name}-{bit_order_name}");
                specifications.push(GridDecoderSpec {
                    id: id.clone(),
                    transform,
                    traversal,
                    decoder: crate::decoder::DecoderSpec::ascii7(
                        format!("ascii7-{bit_order_name}"),
                        bit_order,
                    ),
                });
            }
        }
    }

    GridDecoderManifest::new(protocol_id, version, specifications)
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

    #[test]
    fn manifest_rejects_duplicate_and_incomplete_search_families() {
        let spec = GridDecoderSpec {
            id: "identity-row-major".to_string(),
            transform: GridTransform::Identity,
            traversal: TraversalOrder::RowMajor,
            decoder: DecoderSpec::ascii7("ascii7-msb", BitOrder::MostSignificantFirst),
        };
        assert_eq!(
            GridDecoderManifest::new("crop-intel-v1", 1, vec![]),
            Err(ManifestError::EmptySpecificationList)
        );
        assert_eq!(
            GridDecoderManifest::new("crop-intel-v1", 1, vec![spec.clone(), spec]),
            Err(ManifestError::DuplicateSpecificationId(
                "identity-row-major".to_string()
            ))
        );
        assert_eq!(
            GridDecoderManifest::new(" ", 1, vec![GridDecoderSpec {
                id: "id".to_string(),
                transform: GridTransform::Identity,
                traversal: TraversalOrder::RowMajor,
                decoder: DecoderSpec::ascii7("ascii7", BitOrder::MostSignificantFirst),
            }]),
            Err(ManifestError::EmptyProtocolId)
        );
    }

    #[test]
    fn manifest_runs_only_its_declared_specifications() {
        let bits = encode_ascii7("A", BitOrder::MostSignificantFirst)
            .expect("known ASCII positive control");
        let grid = BinaryGrid::new(1, 7, bits).expect("valid grid");
        let manifest = GridDecoderManifest::new(
            "crop-intel-positive-control",
            1,
            vec![GridDecoderSpec {
                id: "identity-row-major".to_string(),
                transform: GridTransform::Identity,
                traversal: TraversalOrder::RowMajor,
                decoder: DecoderSpec::ascii7("ascii7-msb", BitOrder::MostSignificantFirst),
            }],
        )
        .expect("valid manifest");

        assert_eq!(manifest.protocol_id(), "crop-intel-positive-control");
        assert_eq!(manifest.version(), 1);
        assert_eq!(manifest.run(&grid).len(), 1);
    }

    #[test]
    fn standard_d4_manifest_has_exactly_32_unique_attempts() {
        let manifest = standard_d4_ascii7_manifest("crop-intel-d4-ascii7-v1", 1)
            .expect("standard decoder manifest");

        assert_eq!(manifest.specifications().len(), 32);
        let unique_ids: std::collections::HashSet<&str> = manifest
            .specifications()
            .iter()
            .map(|specification| specification.id.as_str())
            .collect();
        assert_eq!(unique_ids.len(), 32);
        assert!(manifest
            .specifications()
            .iter()
            .any(|specification| specification.id == "identity-row_major-msb"));
        assert!(manifest
            .specifications()
            .iter()
            .any(|specification| specification.id == "anti_transpose-column_major-lsb"));
    }

    fn row_and_column_margins(grid: &BinaryGrid) -> (Vec<usize>, Vec<usize>) {
        let rows = (0..grid.rows())
            .map(|row| {
                (0..grid.columns())
                    .filter(|column| grid.cells[row * grid.columns() + *column])
                    .count()
            })
            .collect();
        let columns = (0..grid.columns())
            .map(|column| {
                (0..grid.rows())
                    .filter(|row| grid.cells[*row * grid.columns() + column])
                    .count()
            })
            .collect();
        (rows, columns)
    }

    #[test]
    fn margin_shuffle_preserves_each_row_and_column_sum() {
        let grid = BinaryGrid::new(
            2,
            2,
            vec![true, false, false, true],
        )
        .expect("valid checkerboard");
        let before = row_and_column_margins(&grid);
        let shuffled = margin_preserving_shuffle(&grid, 42, 11);
        let after = row_and_column_margins(&shuffled.grid);

        assert_eq!(before, after);
        assert_eq!(shuffled.attempted_swaps, 11);
        assert_eq!(shuffled.accepted_swaps, 11);
    }

    #[test]
    fn margin_shuffle_is_deterministic_for_a_frozen_seed() {
        let grid = BinaryGrid::new(
            3,
            3,
            vec![
                true, false, true,
                false, true, false,
                true, false, true,
            ],
        )
        .expect("valid grid");
        let a = margin_preserving_shuffle(&grid, 12345, 100);
        let b = margin_preserving_shuffle(&grid, 12345, 100);

        assert_eq!(a, b);
    }

    #[test]
    fn margin_shuffle_does_not_change_non_switchable_patterns() {
        let grid = BinaryGrid::new(
            2,
            2,
            vec![true, true, false, false],
        )
        .expect("valid grid");
        let shuffled = margin_preserving_shuffle(&grid, 7, 20);

        assert_eq!(shuffled.grid, grid);
        assert_eq!(shuffled.accepted_swaps, 0);
    }
}
