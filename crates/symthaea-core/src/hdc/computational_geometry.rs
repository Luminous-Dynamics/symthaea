// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Computational Geometry Engine
//!
//! Basic computational geometry with HDC encoding: points, lines, polygons,
//! convex hull, line intersection, point-in-polygon, area/perimeter.

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::seed_from_name;
use serde::{Deserialize, Serialize};

// ─── Types ───────────────────────────────────────────────────────────────────

/// A 2D point with HDC encoding
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

impl Point2D {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn distance(&self, other: &Point2D) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    pub fn encode(&self) -> BinaryHV {
        let point_prim = BinaryHV::random(seed_from_name("POINT"));
        let x_hv = BinaryHV::random(seed_from_name(&format!("PX_{}", self.x.to_bits())));
        let y_hv = BinaryHV::random(seed_from_name(&format!("PY_{}", self.y.to_bits())));
        point_prim.bind(&x_hv).bind(&y_hv)
    }
}

/// A 2D line segment
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LineSegment {
    pub start: Point2D,
    pub end: Point2D,
}

impl LineSegment {
    pub fn new(start: Point2D, end: Point2D) -> Self {
        Self { start, end }
    }

    pub fn length(&self) -> f64 {
        self.start.distance(&self.end)
    }
}

/// A circle
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Circle {
    pub center: Point2D,
    pub radius: f64,
}

impl Circle {
    pub fn new(center: Point2D, radius: f64) -> Self {
        Self { center, radius }
    }

    pub fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }

    pub fn circumference(&self) -> f64 {
        2.0 * std::f64::consts::PI * self.radius
    }

    pub fn contains(&self, point: &Point2D) -> bool {
        self.center.distance(point) <= self.radius
    }
}

/// A polygon defined by its vertices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Polygon {
    pub vertices: Vec<Point2D>,
}

// ─── Geometry Engine ─────────────────────────────────────────────────────────

pub struct GeometryEngine;

impl GeometryEngine {
    /// Cross product of vectors OA and OB (2D, returns scalar z-component)
    pub fn cross(o: &Point2D, a: &Point2D, b: &Point2D) -> f64 {
        (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    }

    /// Convex hull via Graham scan. Returns vertices in counter-clockwise order.
    pub fn convex_hull(points: &[Point2D]) -> Vec<Point2D> {
        if points.len() < 3 {
            return points.to_vec();
        }

        let mut pts = points.to_vec();

        // Find the lowest point (leftmost if tie)
        let mut lowest = 0;
        for i in 1..pts.len() {
            if pts[i].y < pts[lowest].y || (pts[i].y == pts[lowest].y && pts[i].x < pts[lowest].x) {
                lowest = i;
            }
        }
        pts.swap(0, lowest);
        let pivot = pts[0];

        // Sort by polar angle with respect to pivot
        pts[1..].sort_by(|a, b| {
            let cross = Self::cross(&pivot, a, b);
            if cross.abs() < 1e-12 {
                // Collinear: closer point first
                pivot
                    .distance(a)
                    .partial_cmp(&pivot.distance(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else if cross > 0.0 {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });

        let mut hull: Vec<Point2D> = Vec::new();
        for &p in &pts {
            while hull.len() >= 2
                && Self::cross(&hull[hull.len() - 2], &hull[hull.len() - 1], &p) <= 0.0
            {
                hull.pop();
            }
            hull.push(p);
        }
        hull
    }

    /// Check if two line segments intersect
    pub fn segments_intersect(seg1: &LineSegment, seg2: &LineSegment) -> bool {
        let d1 = Self::cross(&seg2.start, &seg2.end, &seg1.start);
        let d2 = Self::cross(&seg2.start, &seg2.end, &seg1.end);
        let d3 = Self::cross(&seg1.start, &seg1.end, &seg2.start);
        let d4 = Self::cross(&seg1.start, &seg1.end, &seg2.end);

        if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
            && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
        {
            return true;
        }

        // Collinear cases
        if d1.abs() < 1e-12 && Self::on_segment(&seg2.start, &seg2.end, &seg1.start) {
            return true;
        }
        if d2.abs() < 1e-12 && Self::on_segment(&seg2.start, &seg2.end, &seg1.end) {
            return true;
        }
        if d3.abs() < 1e-12 && Self::on_segment(&seg1.start, &seg1.end, &seg2.start) {
            return true;
        }
        if d4.abs() < 1e-12 && Self::on_segment(&seg1.start, &seg1.end, &seg2.end) {
            return true;
        }

        false
    }

    fn on_segment(p: &Point2D, q: &Point2D, r: &Point2D) -> bool {
        r.x >= p.x.min(q.x) && r.x <= p.x.max(q.x) && r.y >= p.y.min(q.y) && r.y <= p.y.max(q.y)
    }

    /// Point-in-polygon test (ray casting)
    pub fn point_in_polygon(point: &Point2D, polygon: &Polygon) -> bool {
        let n = polygon.vertices.len();
        if n < 3 {
            return false;
        }

        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let vi = &polygon.vertices[i];
            let vj = &polygon.vertices[j];

            if (vi.y > point.y) != (vj.y > point.y)
                && point.x < (vj.x - vi.x) * (point.y - vi.y) / (vj.y - vi.y) + vi.x
            {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    /// Area of a polygon using the shoelace formula
    pub fn polygon_area(polygon: &Polygon) -> f64 {
        let n = polygon.vertices.len();
        if n < 3 {
            return 0.0;
        }

        let mut area = 0.0;
        let mut j = n - 1;
        for i in 0..n {
            area += polygon.vertices[j].x * polygon.vertices[i].y;
            area -= polygon.vertices[i].x * polygon.vertices[j].y;
            j = i;
        }
        area.abs() / 2.0
    }

    /// Perimeter of a polygon
    pub fn polygon_perimeter(polygon: &Polygon) -> f64 {
        let n = polygon.vertices.len();
        if n < 2 {
            return 0.0;
        }

        let mut perimeter = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            perimeter += polygon.vertices[i].distance(&polygon.vertices[j]);
        }
        perimeter
    }

    /// Centroid of a polygon
    pub fn polygon_centroid(polygon: &Polygon) -> Point2D {
        let n = polygon.vertices.len();
        if n == 0 {
            return Point2D::new(0.0, 0.0);
        }

        let cx: f64 = polygon.vertices.iter().map(|v| v.x).sum::<f64>() / n as f64;
        let cy: f64 = polygon.vertices.iter().map(|v| v.y).sum::<f64>() / n as f64;
        Point2D::new(cx, cy)
    }

    /// Triangle area from three points
    pub fn triangle_area(a: &Point2D, b: &Point2D, c: &Point2D) -> f64 {
        Self::cross(a, b, c).abs() / 2.0
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-8;

    #[test]
    fn test_distance() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(3.0, 4.0);
        assert!((p1.distance(&p2) - 5.0).abs() < TOL);
    }

    #[test]
    fn test_convex_hull_triangle() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.5, 1.0),
            Point2D::new(0.5, 0.3), // Interior point
        ];
        let hull = GeometryEngine::convex_hull(&points);
        assert_eq!(hull.len(), 3);
    }

    #[test]
    fn test_convex_hull_square() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
            Point2D::new(0.5, 0.5), // Interior
        ];
        let hull = GeometryEngine::convex_hull(&points);
        assert_eq!(hull.len(), 4);
    }

    #[test]
    fn test_segments_intersect() {
        let seg1 = LineSegment::new(Point2D::new(0.0, 0.0), Point2D::new(2.0, 2.0));
        let seg2 = LineSegment::new(Point2D::new(0.0, 2.0), Point2D::new(2.0, 0.0));
        assert!(GeometryEngine::segments_intersect(&seg1, &seg2));
    }

    #[test]
    fn test_segments_no_intersect() {
        let seg1 = LineSegment::new(Point2D::new(0.0, 0.0), Point2D::new(1.0, 0.0));
        let seg2 = LineSegment::new(Point2D::new(0.0, 1.0), Point2D::new(1.0, 1.0));
        assert!(!GeometryEngine::segments_intersect(&seg1, &seg2));
    }

    #[test]
    fn test_point_in_polygon() {
        let square = Polygon {
            vertices: vec![
                Point2D::new(0.0, 0.0),
                Point2D::new(2.0, 0.0),
                Point2D::new(2.0, 2.0),
                Point2D::new(0.0, 2.0),
            ],
        };
        assert!(GeometryEngine::point_in_polygon(
            &Point2D::new(1.0, 1.0),
            &square
        ));
        assert!(!GeometryEngine::point_in_polygon(
            &Point2D::new(3.0, 3.0),
            &square
        ));
    }

    #[test]
    fn test_polygon_area_square() {
        let square = Polygon {
            vertices: vec![
                Point2D::new(0.0, 0.0),
                Point2D::new(2.0, 0.0),
                Point2D::new(2.0, 2.0),
                Point2D::new(0.0, 2.0),
            ],
        };
        assert!((GeometryEngine::polygon_area(&square) - 4.0).abs() < TOL);
    }

    #[test]
    fn test_polygon_area_triangle() {
        let triangle = Polygon {
            vertices: vec![
                Point2D::new(0.0, 0.0),
                Point2D::new(4.0, 0.0),
                Point2D::new(0.0, 3.0),
            ],
        };
        assert!((GeometryEngine::polygon_area(&triangle) - 6.0).abs() < TOL);
    }

    #[test]
    fn test_polygon_perimeter() {
        let square = Polygon {
            vertices: vec![
                Point2D::new(0.0, 0.0),
                Point2D::new(1.0, 0.0),
                Point2D::new(1.0, 1.0),
                Point2D::new(0.0, 1.0),
            ],
        };
        assert!((GeometryEngine::polygon_perimeter(&square) - 4.0).abs() < TOL);
    }

    #[test]
    fn test_centroid() {
        let triangle = Polygon {
            vertices: vec![
                Point2D::new(0.0, 0.0),
                Point2D::new(3.0, 0.0),
                Point2D::new(0.0, 3.0),
            ],
        };
        let c = GeometryEngine::polygon_centroid(&triangle);
        assert!((c.x - 1.0).abs() < TOL);
        assert!((c.y - 1.0).abs() < TOL);
    }

    #[test]
    fn test_triangle_area() {
        let area = GeometryEngine::triangle_area(
            &Point2D::new(0.0, 0.0),
            &Point2D::new(4.0, 0.0),
            &Point2D::new(0.0, 3.0),
        );
        assert!((area - 6.0).abs() < TOL);
    }

    #[test]
    fn test_circle() {
        let c = Circle::new(Point2D::new(0.0, 0.0), 5.0);
        assert!((c.area() - 25.0 * std::f64::consts::PI).abs() < TOL);
        assert!((c.circumference() - 10.0 * std::f64::consts::PI).abs() < TOL);
        assert!(c.contains(&Point2D::new(3.0, 4.0)));
        assert!(!c.contains(&Point2D::new(4.0, 4.0)));
    }

    #[test]
    fn test_point_encoding() {
        let p1 = Point2D::new(1.0, 2.0);
        let p2 = Point2D::new(3.0, 4.0);
        let sim = p1.encode().similarity(&p2.encode());
        assert!(
            sim < 0.6,
            "Different points should have different encodings: {}",
            sim
        );
    }
}
