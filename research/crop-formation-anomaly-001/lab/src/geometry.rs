//! Descriptive geometry for calibrated 2D point sets.
//!
//! All coordinates must already be in a declared coordinate frame. This
//! module does not calibrate aerial images or convert pixels to metres.

use std::f64::consts::TAU;

const MAX_SYMMETRY_ORDER: usize = 24;
const MIN_SCALE: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2 {
    pub x: f64,
    pub y: f64,
}

impl Point2 {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn distance_squared(self, other: Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds2 {
    pub min: Point2,
    pub max: Point2,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymmetryScore {
    /// Number of repetitions tested (for example, 4 means quarter-turn symmetry).
    pub order: usize,
    /// Symmetric nearest-neighbour RMS residual divided by RMS radius.
    /// Lower is more geometrically symmetric; this is descriptive, not a p-value.
    pub normalized_rms_error: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeometryDescriptor {
    pub point_count: usize,
    pub centroid: Point2,
    pub bounds: Bounds2,
    pub mean_radius: f64,
    pub rms_radius: f64,
    pub radial_coefficient_of_variation: f64,
    /// Counts in equal-width shells between zero and the maximum observed radius.
    pub radial_histogram: Vec<usize>,
    /// Counts in equal-width angular bins over [0, 2π).
    pub angular_histogram: Vec<usize>,
    /// Approximate rotational symmetry residuals for orders 2..=max_order.
    pub rotational_symmetry: Vec<SymmetryScore>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeometryError {
    TooFewPoints { provided: usize },
    NonFiniteCoordinate { index: usize },
    EmptyHistogram,
    InvalidSymmetryOrder { requested: usize, maximum: usize },
    DegeneratePointSet,
}

impl std::fmt::Display for GeometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewPoints { provided } => {
                write!(f, "at least three points are required; received {provided}")
            }
            Self::NonFiniteCoordinate { index } => {
                write!(f, "point {index} has a non-finite coordinate")
            }
            Self::EmptyHistogram => write!(f, "histogram bin counts must be non-zero"),
            Self::InvalidSymmetryOrder { requested, maximum } => write!(
                f,
                "symmetry order must be between 2 and {maximum}; received {requested}"
            ),
            Self::DegeneratePointSet => {
                write!(f, "point set has no measurable radial extent")
            }
        }
    }
}

impl std::error::Error for GeometryError {}

/// Describe a point set in an already calibrated 2D coordinate frame.
///
/// The symmetry score uses symmetric nearest-neighbour RMS distance. It is a
/// geometric descriptor only: significance requires matched null patterns
/// and a predeclared search family outside this function.
pub fn describe(
    points: &[Point2],
    radial_bins: usize,
    angular_bins: usize,
    max_symmetry_order: usize,
) -> Result<GeometryDescriptor, GeometryError> {
    if points.len() < 3 {
        return Err(GeometryError::TooFewPoints {
            provided: points.len(),
        });
    }
    if radial_bins == 0 || angular_bins == 0 {
        return Err(GeometryError::EmptyHistogram);
    }
    if !(2..=MAX_SYMMETRY_ORDER).contains(&max_symmetry_order) {
        return Err(GeometryError::InvalidSymmetryOrder {
            requested: max_symmetry_order,
            maximum: MAX_SYMMETRY_ORDER,
        });
    }

    for (index, point) in points.iter().enumerate() {
        if !point.x.is_finite() || !point.y.is_finite() {
            return Err(GeometryError::NonFiniteCoordinate { index });
        }
    }

    let count = points.len() as f64;
    let centroid = Point2::new(
        points.iter().map(|point| point.x).sum::<f64>() / count,
        points.iter().map(|point| point.y).sum::<f64>() / count,
    );

    let mut min = points[0];
    let mut max = points[0];
    let mut radii = Vec::with_capacity(points.len());
    let mut max_radius = 0.0_f64;
    let mut radius_sum = 0.0;
    let mut radius_squared_sum = 0.0;

    for point in points {
        min.x = min.x.min(point.x);
        min.y = min.y.min(point.y);
        max.x = max.x.max(point.x);
        max.y = max.y.max(point.y);

        let radius = point.distance_squared(centroid).sqrt();
        max_radius = max_radius.max(radius);
        radius_sum += radius;
        radius_squared_sum += radius * radius;
        radii.push(radius);
    }

    if !max_radius.is_finite() || max_radius <= MIN_SCALE {
        return Err(GeometryError::DegeneratePointSet);
    }

    let mean_radius = radius_sum / count;
    let rms_radius = (radius_squared_sum / count).sqrt();
    let variance = (radius_squared_sum / count - mean_radius * mean_radius).max(0.0);
    let radial_coefficient_of_variation = if mean_radius > MIN_SCALE {
        variance.sqrt() / mean_radius
    } else {
        0.0
    };

    let mut radial_histogram = vec![0; radial_bins];
    let mut angular_histogram = vec![0; angular_bins];

    for (point, radius) in points.iter().zip(radii.iter().copied()) {
        let radial_position = (radius / max_radius * radial_bins as f64).floor();
        let radial_index = (radial_position as usize).min(radial_bins - 1);
        radial_histogram[radial_index] += 1;

        let angle = (point.y - centroid.y).atan2(point.x - centroid.x);
        let wrapped = (angle + TAU) % TAU;
        let angular_position = (wrapped / TAU * angular_bins as f64).floor();
        let angular_index = (angular_position as usize).min(angular_bins - 1);
        angular_histogram[angular_index] += 1;
    }

    let rotational_symmetry = (2..=max_symmetry_order)
        .map(|order| SymmetryScore {
            order,
            normalized_rms_error: rotational_residual(points, centroid, rms_radius, order),
        })
        .collect();

    Ok(GeometryDescriptor {
        point_count: points.len(),
        centroid,
        bounds: Bounds2 { min, max },
        mean_radius,
        rms_radius,
        radial_coefficient_of_variation,
        radial_histogram,
        angular_histogram,
        rotational_symmetry,
    })
}

fn rotational_residual(
    points: &[Point2],
    centroid: Point2,
    rms_radius: f64,
    order: usize,
) -> f64 {
    let angle = TAU / order as f64;
    let (sin_angle, cos_angle) = angle.sin_cos();

    let rotate = |point: Point2, sin: f64, cos: f64| {
        let x = point.x - centroid.x;
        let y = point.y - centroid.y;
        Point2::new(
            centroid.x + x * cos - y * sin,
            centroid.y + x * sin + y * cos,
        )
    };

    let directed_mean_squared = |sin: f64, cos: f64| {
        points
            .iter()
            .map(|point| {
                let transformed = rotate(*point, sin, cos);
                points
                    .iter()
                    .map(|candidate| transformed.distance_squared(*candidate))
                    .fold(f64::INFINITY, f64::min)
            })
            .sum::<f64>()
            / points.len() as f64
    };

    let reverse_sin = -sin_angle;
    let reverse_cos = cos_angle;
    let symmetric_mean_squared = 0.5
        * (directed_mean_squared(sin_angle, cos_angle)
            + directed_mean_squared(reverse_sin, reverse_cos));

    (symmetric_mean_squared.max(0.0).sqrt() / rms_radius.max(MIN_SCALE)).min(2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square() -> [Point2; 4] {
        [
            Point2::new(-1.0, -1.0),
            Point2::new(1.0, -1.0),
            Point2::new(1.0, 1.0),
            Point2::new(-1.0, 1.0),
        ]
    }

    #[test]
    fn square_has_low_four_fold_rotational_residual() {
        let result = describe(&square(), 4, 8, 8).expect("valid square");
        let score = result
            .rotational_symmetry
            .iter()
            .find(|score| score.order == 4)
            .expect("order 4 is reported");
        assert!(score.normalized_rms_error < 1.0e-10);
    }

    #[test]
    fn descriptor_is_translation_invariant_except_for_centroid_and_bounds() {
        let original = square();
        let translated = original.map(|point| Point2::new(point.x + 17.0, point.y - 9.0));
        let a = describe(&original, 4, 8, 8).expect("valid original");
        let b = describe(&translated, 4, 8, 8).expect("valid translated");

        assert!((a.mean_radius - b.mean_radius).abs() < 1.0e-10);
        assert_eq!(a.radial_histogram, b.radial_histogram);
        assert_eq!(a.angular_histogram, b.angular_histogram);
        assert!((b.centroid.x - a.centroid.x - 17.0).abs() < 1.0e-10);
        assert!((b.centroid.y - a.centroid.y + 9.0).abs() < 1.0e-10);
    }

    #[test]
    fn malformed_inputs_are_rejected() {
        assert_eq!(
            describe(&[Point2::new(0.0, 0.0)], 4, 8, 4),
            Err(GeometryError::TooFewPoints { provided: 1 })
        );
        assert_eq!(
            describe(
                &[
                    Point2::new(0.0, 0.0),
                    Point2::new(1.0, 0.0),
                    Point2::new(f64::NAN, 1.0)
                ],
                4,
                8,
                4
            ),
            Err(GeometryError::NonFiniteCoordinate { index: 2 })
        );
        assert_eq!(
            describe(
                &[
                    Point2::new(2.0, 3.0),
                    Point2::new(2.0, 3.0),
                    Point2::new(2.0, 3.0)
                ],
                4,
                8,
                4
            ),
            Err(GeometryError::DegeneratePointSet)
        );
    }

    #[test]
    fn histograms_preserve_point_count() {
        let result = describe(&square(), 5, 7, 4).expect("valid square");
        assert_eq!(result.radial_histogram.iter().sum::<usize>(), 4);
        assert_eq!(result.angular_histogram.iter().sum::<usize>(), 4);
    }
}
