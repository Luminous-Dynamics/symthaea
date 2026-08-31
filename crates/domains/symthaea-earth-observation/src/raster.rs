//! Deterministic raster-grid and pixel-window semantics.
//!
//! This module deliberately stops before raster I/O, reprojection, resampling,
//! interpolation, cloud classification, or provider-specific metadata parsing.
//! Its job is narrower: make the exact source pixel support of a derived raster
//! artifact explicit, replayable, and difficult to reinterpret accidentally.
//!
//! Window coordinates are integer and half-open. Windowing never changes pixel
//! spacing, CRS identity, affine coefficients, or sample count. A later explicit
//! operation must own any reprojection or resampling.

use std::error::Error;
use std::fmt::{Display, Formatter};

pub type RasterResult<T> = std::result::Result<T, RasterError>;

#[derive(Debug, Clone, PartialEq)]
pub enum RasterError {
    EmptyCrsId,
    ZeroDimension(&'static str),
    EmptyWindow,
    ArithmeticOverflow(&'static str),
    NonFiniteTransform {
        field: &'static str,
        value: f64,
    },
    DegenerateTransform,
    WindowOutOfBounds {
        row_offset: u32,
        col_offset: u32,
        rows: u32,
        cols: u32,
        raster_rows: u32,
        raster_cols: u32,
    },
    PixelOutOfBounds {
        row: u32,
        col: u32,
        raster_rows: u32,
        raster_cols: u32,
    },
}

impl Display for RasterError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyCrsId => write!(f, "CRS identifier must not be empty"),
            Self::ZeroDimension(field) => write!(f, "{field} must be greater than zero"),
            Self::EmptyWindow => write!(f, "raster windows must contain at least one pixel"),
            Self::ArithmeticOverflow(field) => write!(f, "arithmetic overflow while computing {field}"),
            Self::NonFiniteTransform { field, value } => {
                write!(f, "affine transform field {field} must be finite, got {value}")
            }
            Self::DegenerateTransform => write!(f, "affine raster transform must be invertible"),
            Self::WindowOutOfBounds {
                row_offset,
                col_offset,
                rows,
                cols,
                raster_rows,
                raster_cols,
            } => write!(
                f,
                "window row={row_offset} col={col_offset} rows={rows} cols={cols} exceeds raster rows={raster_rows} cols={raster_cols}"
            ),
            Self::PixelOutOfBounds {
                row,
                col,
                raster_rows,
                raster_cols,
            } => write!(
                f,
                "pixel row={row} col={col} is outside raster rows={raster_rows} cols={raster_cols}"
            ),
        }
    }
}

impl Error for RasterError {}

/// Stable spatial-reference identifier.
///
/// This is an identity string, not a CRS parser. Examples include `EPSG:32635`
/// or a separately frozen WKT/proj definition digest. Reprojection belongs in a
/// later explicit transform layer.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CrsId(String);

impl CrsId {
    pub fn new(value: impl Into<String>) -> RasterResult<Self> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(RasterError::EmptyCrsId);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RasterShape {
    rows: u32,
    cols: u32,
}

impl RasterShape {
    pub fn new(rows: u32, cols: u32) -> RasterResult<Self> {
        if rows == 0 {
            return Err(RasterError::ZeroDimension("raster rows"));
        }
        if cols == 0 {
            return Err(RasterError::ZeroDimension("raster cols"));
        }
        Ok(Self { rows, cols })
    }

    pub const fn rows(self) -> u32 {
        self.rows
    }

    pub const fn cols(self) -> u32 {
        self.cols
    }

    pub const fn pixel_count(self) -> u64 {
        self.rows as u64 * self.cols as u64
    }
}

/// Integer, half-open source-pixel window.
///
/// The window covers rows `[row_offset, row_offset + rows)` and columns
/// `[col_offset, col_offset + cols)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PixelWindow {
    row_offset: u32,
    col_offset: u32,
    rows: u32,
    cols: u32,
}

impl PixelWindow {
    pub fn new(row_offset: u32, col_offset: u32, rows: u32, cols: u32) -> RasterResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(RasterError::EmptyWindow);
        }
        row_offset
            .checked_add(rows)
            .ok_or(RasterError::ArithmeticOverflow("window row end"))?;
        col_offset
            .checked_add(cols)
            .ok_or(RasterError::ArithmeticOverflow("window column end"))?;
        Ok(Self {
            row_offset,
            col_offset,
            rows,
            cols,
        })
    }

    pub fn full(shape: RasterShape) -> Self {
        Self {
            row_offset: 0,
            col_offset: 0,
            rows: shape.rows,
            cols: shape.cols,
        }
    }

    pub const fn row_offset(self) -> u32 {
        self.row_offset
    }

    pub const fn col_offset(self) -> u32 {
        self.col_offset
    }

    pub const fn rows(self) -> u32 {
        self.rows
    }

    pub const fn cols(self) -> u32 {
        self.cols
    }

    pub fn row_end_exclusive(self) -> u32 {
        // Construction proves this addition cannot overflow.
        self.row_offset + self.rows
    }

    pub fn col_end_exclusive(self) -> u32 {
        // Construction proves this addition cannot overflow.
        self.col_offset + self.cols
    }

    pub fn validate_for(self, shape: RasterShape) -> RasterResult<()> {
        if self.row_end_exclusive() > shape.rows || self.col_end_exclusive() > shape.cols {
            return Err(RasterError::WindowOutOfBounds {
                row_offset: self.row_offset,
                col_offset: self.col_offset,
                rows: self.rows,
                cols: self.cols,
                raster_rows: shape.rows,
                raster_cols: shape.cols,
            });
        }
        Ok(())
    }

    pub fn contains(self, row: u32, col: u32) -> bool {
        row >= self.row_offset
            && row < self.row_end_exclusive()
            && col >= self.col_offset
            && col < self.col_end_exclusive()
    }

    pub fn shape(self) -> RasterShape {
        RasterShape {
            rows: self.rows,
            cols: self.cols,
        }
    }
}

/// Meaning of the affine transform's `(row=0, col=0)` anchor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridAnchor {
    /// The outer corner at the start of row 0 / column 0.
    PixelCorner,
    /// The center of pixel row 0 / column 0.
    PixelCenter,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MapPoint {
    pub x: f64,
    pub y: f64,
}

/// Two-dimensional affine map from raster row/column coordinates into a
/// declared CRS.
///
/// ```text
/// x = origin_x + col * col_step_x + row * row_step_x
/// y = origin_y + col * col_step_y + row * row_step_y
/// ```
///
/// Rotated/skewed grids are supported. The 2x2 step matrix must be invertible.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineGridTransform {
    pub origin_x: f64,
    pub origin_y: f64,
    pub col_step_x: f64,
    pub col_step_y: f64,
    pub row_step_x: f64,
    pub row_step_y: f64,
    pub anchor: GridAnchor,
}

impl AffineGridTransform {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        origin_x: f64,
        origin_y: f64,
        col_step_x: f64,
        col_step_y: f64,
        row_step_x: f64,
        row_step_y: f64,
        anchor: GridAnchor,
    ) -> RasterResult<Self> {
        for (field, value) in [
            ("origin_x", origin_x),
            ("origin_y", origin_y),
            ("col_step_x", col_step_x),
            ("col_step_y", col_step_y),
            ("row_step_x", row_step_x),
            ("row_step_y", row_step_y),
        ] {
            if !value.is_finite() {
                return Err(RasterError::NonFiniteTransform { field, value });
            }
        }

        let determinant = col_step_x * row_step_y - row_step_x * col_step_y;
        if !determinant.is_finite() {
            return Err(RasterError::NonFiniteTransform {
                field: "determinant",
                value: determinant,
            });
        }
        if determinant == 0.0 {
            return Err(RasterError::DegenerateTransform);
        }

        Ok(Self {
            origin_x,
            origin_y,
            col_step_x,
            col_step_y,
            row_step_x,
            row_step_y,
            anchor,
        })
    }

    pub fn map_coordinate(self, row: f64, col: f64) -> RasterResult<MapPoint> {
        if !row.is_finite() {
            return Err(RasterError::NonFiniteTransform {
                field: "row coordinate",
                value: row,
            });
        }
        if !col.is_finite() {
            return Err(RasterError::NonFiniteTransform {
                field: "column coordinate",
                value: col,
            });
        }

        let x = self.origin_x + col * self.col_step_x + row * self.row_step_x;
        let y = self.origin_y + col * self.col_step_y + row * self.row_step_y;
        if !x.is_finite() {
            return Err(RasterError::NonFiniteTransform {
                field: "mapped x",
                value: x,
            });
        }
        if !y.is_finite() {
            return Err(RasterError::NonFiniteTransform {
                field: "mapped y",
                value: y,
            });
        }
        Ok(MapPoint { x, y })
    }

    fn translated(self, row_offset: u32, col_offset: u32) -> RasterResult<Self> {
        let origin = self.map_coordinate(row_offset as f64, col_offset as f64)?;
        Ok(Self {
            origin_x: origin.x,
            origin_y: origin.y,
            ..self
        })
    }
}

/// Immutable reference system shared by a root raster and all exact windows
/// extracted from it.
#[derive(Debug, Clone, PartialEq)]
pub struct RasterReference {
    pub crs: CrsId,
    pub transform: AffineGridTransform,
}

impl RasterReference {
    pub const fn new(crs: CrsId, transform: AffineGridTransform) -> Self {
        Self { crs, transform }
    }
}

/// A raster grid expressed as an exact integer window within a root reference.
///
/// Keeping root-relative integer offsets rather than repeatedly translating the
/// floating-point affine origin prevents nested windowing from accumulating
/// avoidable floating-point drift. An effective local affine transform can be
/// materialized deterministically when needed.
#[derive(Debug, Clone, PartialEq)]
pub struct RasterGrid {
    reference: RasterReference,
    root_row_offset: u32,
    root_col_offset: u32,
    shape: RasterShape,
}

impl RasterGrid {
    pub const fn new(shape: RasterShape, reference: RasterReference) -> Self {
        Self {
            reference,
            root_row_offset: 0,
            root_col_offset: 0,
            shape,
        }
    }

    pub const fn shape(&self) -> RasterShape {
        self.shape
    }

    pub const fn root_row_offset(&self) -> u32 {
        self.root_row_offset
    }

    pub const fn root_col_offset(&self) -> u32 {
        self.root_col_offset
    }

    pub fn reference(&self) -> &RasterReference {
        &self.reference
    }

    pub fn crs(&self) -> &CrsId {
        &self.reference.crs
    }

    /// Materialize this grid's local affine transform from the immutable root
    /// transform and exact accumulated integer offsets.
    pub fn effective_transform(&self) -> RasterResult<AffineGridTransform> {
        self.reference
            .transform
            .translated(self.root_row_offset, self.root_col_offset)
    }

    /// Map the center of one local pixel into the declared CRS.
    pub fn pixel_center(&self, row: u32, col: u32) -> RasterResult<MapPoint> {
        if row >= self.shape.rows || col >= self.shape.cols {
            return Err(RasterError::PixelOutOfBounds {
                row,
                col,
                raster_rows: self.shape.rows,
                raster_cols: self.shape.cols,
            });
        }

        let root_row = self
            .root_row_offset
            .checked_add(row)
            .ok_or(RasterError::ArithmeticOverflow("root pixel row"))?;
        let root_col = self
            .root_col_offset
            .checked_add(col)
            .ok_or(RasterError::ArithmeticOverflow("root pixel column"))?;

        let (row_coordinate, col_coordinate) = match self.reference.transform.anchor {
            GridAnchor::PixelCorner => (root_row as f64 + 0.5, root_col as f64 + 0.5),
            GridAnchor::PixelCenter => (root_row as f64, root_col as f64),
        };
        self.reference
            .transform
            .map_coordinate(row_coordinate, col_coordinate)
    }

    /// Create an exact subwindow. No interpolation, padding, reprojection, or
    /// resampling is performed or implied.
    pub fn window(&self, window: PixelWindow) -> RasterResult<RasterWindowPlan> {
        window.validate_for(self.shape)?;

        let root_row_offset = self
            .root_row_offset
            .checked_add(window.row_offset)
            .ok_or(RasterError::ArithmeticOverflow("root window row offset"))?;
        let root_col_offset = self
            .root_col_offset
            .checked_add(window.col_offset)
            .ok_or(RasterError::ArithmeticOverflow("root window column offset"))?;

        let output = Self {
            reference: self.reference.clone(),
            root_row_offset,
            root_col_offset,
            shape: window.shape(),
        };

        Ok(RasterWindowPlan {
            source_shape: self.shape,
            source_root_row_offset: self.root_row_offset,
            source_root_col_offset: self.root_col_offset,
            window,
            output,
        })
    }
}

/// Deterministic plan/receipt for exact pixel-window extraction.
///
/// It contains geometry only. Payload hashes and processing-lineage digests are
/// owned by the fixture/evidence layer that materializes the resulting bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct RasterWindowPlan {
    pub source_shape: RasterShape,
    pub source_root_row_offset: u32,
    pub source_root_col_offset: u32,
    pub window: PixelWindow,
    pub output: RasterGrid,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn north_up_reference(anchor: GridAnchor) -> RasterReference {
        RasterReference::new(
            CrsId::new("EPSG:32635").unwrap(),
            AffineGridTransform::new(
                500_000.0,
                7_200_000.0,
                10.0,
                0.0,
                0.0,
                -10.0,
                anchor,
            )
            .unwrap(),
        )
    }

    fn root_grid() -> RasterGrid {
        RasterGrid::new(
            RasterShape::new(100, 200).unwrap(),
            north_up_reference(GridAnchor::PixelCorner),
        )
    }

    #[test]
    fn shape_and_crs_fail_closed() {
        assert_eq!(
            RasterShape::new(0, 1),
            Err(RasterError::ZeroDimension("raster rows"))
        );
        assert_eq!(
            RasterShape::new(1, 0),
            Err(RasterError::ZeroDimension("raster cols"))
        );
        assert_eq!(CrsId::new("   "), Err(RasterError::EmptyCrsId));
    }

    #[test]
    fn windows_are_nonempty_and_overflow_checked() {
        assert_eq!(PixelWindow::new(0, 0, 0, 1), Err(RasterError::EmptyWindow));
        assert!(matches!(
            PixelWindow::new(u32::MAX, 0, 1, 1),
            Err(RasterError::ArithmeticOverflow("window row end"))
        ));
    }

    #[test]
    fn half_open_edge_window_is_exact() {
        let grid = root_grid();
        let window = PixelWindow::new(90, 180, 10, 20).unwrap();
        let plan = grid.window(window).unwrap();

        assert_eq!(plan.output.shape(), RasterShape::new(10, 20).unwrap());
        assert!(window.contains(90, 180));
        assert!(window.contains(99, 199));
        assert!(!window.contains(100, 199));
        assert!(!window.contains(99, 200));
    }

    #[test]
    fn one_pixel_past_edge_is_rejected() {
        let grid = root_grid();
        let window = PixelWindow::new(90, 181, 10, 20).unwrap();
        assert!(matches!(
            grid.window(window),
            Err(RasterError::WindowOutOfBounds { .. })
        ));
    }

    #[test]
    fn windowing_never_resamples_or_changes_reference() {
        let grid = root_grid();
        let plan = grid.window(PixelWindow::new(7, 11, 20, 30).unwrap()).unwrap();

        assert_eq!(plan.output.shape().pixel_count(), 600);
        assert_eq!(plan.output.reference(), grid.reference());
        assert_eq!(plan.output.root_row_offset(), 7);
        assert_eq!(plan.output.root_col_offset(), 11);
    }

    #[test]
    fn nested_windows_compose_with_exact_integer_origin() {
        let grid = root_grid();
        let first = grid.window(PixelWindow::new(10, 20, 50, 80).unwrap()).unwrap();
        let nested = first
            .output
            .window(PixelWindow::new(3, 4, 10, 12).unwrap())
            .unwrap();
        let direct = grid
            .window(PixelWindow::new(13, 24, 10, 12).unwrap())
            .unwrap();

        assert_eq!(nested.output.root_row_offset(), 13);
        assert_eq!(nested.output.root_col_offset(), 24);
        assert_eq!(nested.output, direct.output);
        assert_eq!(
            nested.output.effective_transform().unwrap(),
            direct.output.effective_transform().unwrap()
        );
    }

    #[test]
    fn effective_transform_translates_from_the_root_once() {
        let grid = root_grid();
        let output = grid
            .window(PixelWindow::new(2, 3, 10, 10).unwrap())
            .unwrap()
            .output;
        let transform = output.effective_transform().unwrap();

        assert_eq!(transform.origin_x, 500_030.0);
        assert_eq!(transform.origin_y, 7_199_980.0);
        assert_eq!(transform.col_step_x, 10.0);
        assert_eq!(transform.row_step_y, -10.0);
    }

    #[test]
    fn rotated_affine_translation_uses_both_axes() {
        let reference = RasterReference::new(
            CrsId::new("LOCAL:rotated").unwrap(),
            AffineGridTransform::new(
                100.0,
                200.0,
                2.0,
                0.5,
                -0.25,
                -3.0,
                GridAnchor::PixelCorner,
            )
            .unwrap(),
        );
        let grid = RasterGrid::new(RasterShape::new(20, 20).unwrap(), reference);
        let output = grid
            .window(PixelWindow::new(4, 5, 3, 3).unwrap())
            .unwrap()
            .output;
        let transform = output.effective_transform().unwrap();

        assert_eq!(transform.origin_x, 109.0);
        assert_eq!(transform.origin_y, 190.5);
    }

    #[test]
    fn pixel_center_is_anchor_explicit() {
        let corner_grid = RasterGrid::new(
            RasterShape::new(2, 2).unwrap(),
            north_up_reference(GridAnchor::PixelCorner),
        );
        let center_grid = RasterGrid::new(
            RasterShape::new(2, 2).unwrap(),
            north_up_reference(GridAnchor::PixelCenter),
        );

        let from_corner = corner_grid.pixel_center(0, 0).unwrap();
        let from_center = center_grid.pixel_center(0, 0).unwrap();
        assert_eq!(from_corner, MapPoint { x: 500_005.0, y: 7_199_995.0 });
        assert_eq!(from_center, MapPoint { x: 500_000.0, y: 7_200_000.0 });
    }

    #[test]
    fn out_of_bounds_pixel_center_is_rejected() {
        let grid = root_grid();
        assert!(matches!(
            grid.pixel_center(100, 0),
            Err(RasterError::PixelOutOfBounds { .. })
        ));
    }

    #[test]
    fn transform_rejects_nonfinite_and_degenerate_geometry() {
        assert!(matches!(
            AffineGridTransform::new(
                f64::NAN,
                0.0,
                1.0,
                0.0,
                0.0,
                -1.0,
                GridAnchor::PixelCorner,
            ),
            Err(RasterError::NonFiniteTransform {
                field: "origin_x",
                ..
            })
        ));
        assert_eq!(
            AffineGridTransform::new(
                0.0,
                0.0,
                1.0,
                2.0,
                2.0,
                4.0,
                GridAnchor::PixelCorner,
            ),
            Err(RasterError::DegenerateTransform)
        );
    }
}
