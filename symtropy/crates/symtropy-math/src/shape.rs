use crate::point::Point;
use nalgebra::SVector;

/// Trait for D-dimensional convex shapes compatible with GJK collision detection.
///
/// The support function is the only requirement for GJK — given a direction,
/// return the point on the shape's boundary furthest in that direction.
/// This generalizes naturally to any dimension.
pub trait Shape<const D: usize>: Send + Sync {
    /// Support function: furthest point on the boundary in the given direction.
    fn support(&self, direction: &SVector<f64, D>) -> SVector<f64, D>;

    /// Bounding sphere: (center, radius).
    fn bounding_sphere(&self) -> (Point<D>, f64);
}
