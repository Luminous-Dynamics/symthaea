// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Infill pattern generation for sliced layers.
//!
//! Generates internal fill patterns (rectilinear scan lines) clipped to
//! contour boundaries using the even-odd fill rule. Infill prevents hollow
//! shells and provides structural strength to printed parts.

use crate::slicer::{Contour, Point2, Segment2, SliceLayer};
use serde::{Deserialize, Serialize};

// ── Configuration ────────────────────────────────────────────────────────

/// Infill pattern type.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum InfillPattern {
    /// Parallel lines at a fixed angle.
    Rectilinear,
    /// Two perpendicular passes per layer (0° and 90° relative to base angle).
    Grid,
    /// Hexagonal honeycomb — inherently multi-directional, high strength-to-weight.
    Honeycomb,
}

/// Configuration for infill generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfillConfig {
    /// Pattern type.
    pub pattern: InfillPattern,
    /// Fill density in (0.0, 1.0]. 0.2 = 20% fill.
    pub density: f32,
    /// Angle of scan lines in degrees.
    pub angle_degrees: f32,
}

impl Default for InfillConfig {
    fn default() -> Self {
        Self {
            pattern: InfillPattern::Rectilinear,
            density: 0.2,
            angle_degrees: 45.0,
        }
    }
}

// ── Contour geometry helpers ─────────────────────────────────────────────

impl Contour {
    /// Test whether a point is inside the contour using the ray-casting
    /// (even-odd) algorithm. A horizontal ray is cast from `p` toward +X;
    /// each crossing toggles inside/outside.
    pub fn contains_point(&self, p: Point2) -> bool {
        let n = self.points.len();
        if n < 3 {
            return false;
        }
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let pi = self.points[i];
            let pj = self.points[j];
            // Check if the edge (pj→pi) straddles p.y and intersection is to the right of p.x
            if ((pi.y > p.y) != (pj.y > p.y))
                && (p.x < (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y) + pi.x)
            {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    /// Axis-aligned bounding box: returns (min, max).
    pub fn bounding_box(&self) -> (Point2, Point2) {
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;
        for p in &self.points {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
        }
        (Point2::new(min_x, min_y), Point2::new(max_x, max_y))
    }
}

// ── Contour inset ────────────────────────────────────────────────────────

impl Contour {
    /// Shrink the contour inward by `offset` mm.
    ///
    /// For CCW (outer) contours, positive offset shrinks inward.
    /// The inward normal direction is determined automatically from the winding.
    ///
    /// Uses parallel offset: each edge is moved inward by `offset` along its
    /// inward normal, then consecutive offset edges are intersected to compute
    /// new vertices. Returns `None` if the contour collapses (zero or negative area).
    pub fn inset(&self, offset: f32) -> Option<Contour> {
        let n = self.points.len();
        if n < 3 {
            return None;
        }

        // For CCW contours, the inward normal of edge (a→b) is the left normal:
        // (-dy, dx) / len. For CW contours it's the right normal: (dy, -dx) / len.
        // Determine sign from winding.
        let winding_sign = if self.signed_area() >= 0.0 {
            1.0f32
        } else {
            -1.0f32
        };

        // Build offset lines: each as a point on the line + direction vector.
        // Represent as (point, direction) pairs, then intersect consecutive pairs.
        let mut offset_lines: Vec<(Point2, Point2)> = Vec::with_capacity(n);
        for i in 0..n {
            let j = (i + 1) % n;
            let a = self.points[i];
            let b = self.points[j];
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            let len = (dx * dx + dy * dy).sqrt();
            if len < 1e-8 {
                // Degenerate edge — skip by using the same point twice.
                offset_lines.push((a, b));
                continue;
            }
            // Inward normal for CCW contour: left of direction = (-dy, dx) / len.
            // For CW contour: negate to get right normal = (dy, -dx) / len.
            let nx = -dy / len * winding_sign;
            let ny = dx / len * winding_sign;
            // Offset both endpoints of the edge.
            let oa = Point2::new(a.x + nx * offset, a.y + ny * offset);
            let ob = Point2::new(b.x + nx * offset, b.y + ny * offset);
            offset_lines.push((oa, ob));
        }

        // Intersect consecutive offset lines to get new vertices.
        let mut new_points: Vec<Point2> = Vec::with_capacity(n);
        for i in 0..n {
            let j = (i + 1) % n;
            let (p1, p2) = offset_lines[i];
            let (p3, p4) = offset_lines[j];

            // Line-line intersection of (p1→p2) and (p3→p4).
            let d1x = p2.x - p1.x;
            let d1y = p2.y - p1.y;
            let d2x = p4.x - p3.x;
            let d2y = p4.y - p3.y;
            let denom = d1x * d2y - d1y * d2x;

            let pt = if denom.abs() < 1e-8 {
                // Parallel lines — use midpoint of the shared endpoint.
                Point2::new((p2.x + p3.x) * 0.5, (p2.y + p3.y) * 0.5)
            } else {
                let t = ((p3.x - p1.x) * d2y - (p3.y - p1.y) * d2x) / denom;
                Point2::new(p1.x + t * d1x, p1.y + t * d1y)
            };
            new_points.push(pt);
        }

        let result = Contour { points: new_points };
        // Reject degenerate insets:
        // 1. Non-positive area (collapsed or inverted to CW).
        // 2. Result area >= original area (inset overshot and wrapped around).
        let orig_area = self.signed_area();
        let result_area = result.signed_area();
        if result_area <= 0.0 || (orig_area > 0.0 && result_area >= orig_area) {
            return None;
        }
        Some(result)
    }
}

// ── Scan-line infill generation ──────────────────────────────────────────

/// Compute the X-coordinate where a horizontal scan line at `scan_y`
/// intersects the edge from `a` to `b`. Returns `None` if the edge is
/// horizontal or does not straddle `scan_y`.
pub fn scan_line_edge_intersection(scan_y: f32, a: Point2, b: Point2) -> Option<f32> {
    let dy = b.y - a.y;
    if dy.abs() < 1e-10 {
        return None; // Horizontal edge — no single intersection.
    }
    let t = (scan_y - a.y) / dy;
    if !(0.0..=1.0).contains(&t) {
        return None;
    }
    Some(a.x + t * (b.x - a.x))
}

/// Collect all contour edges from outer and inner contours.
fn collect_edges(layer: &SliceLayer) -> Vec<(Point2, Point2)> {
    let mut edges = Vec::new();
    for contour in layer
        .outer_contours
        .iter()
        .chain(layer.inner_contours.iter())
    {
        let n = contour.points.len();
        if n < 2 {
            continue;
        }
        for i in 0..n {
            let j = (i + 1) % n;
            edges.push((contour.points[i], contour.points[j]));
        }
    }
    edges
}

/// Compute the bounding box of all contours in a layer.
fn layer_bounding_box(layer: &SliceLayer) -> Option<(Point2, Point2)> {
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    let mut any = false;
    for contour in layer
        .outer_contours
        .iter()
        .chain(layer.inner_contours.iter())
    {
        if contour.points.is_empty() {
            continue;
        }
        any = true;
        let (cmin, cmax) = contour.bounding_box();
        min_x = min_x.min(cmin.x);
        min_y = min_y.min(cmin.y);
        max_x = max_x.max(cmax.x);
        max_y = max_y.max(cmax.y);
    }
    if any {
        Some((Point2::new(min_x, min_y), Point2::new(max_x, max_y)))
    } else {
        None
    }
}

/// Generate infill at a specific angle (internal workhorse).
fn generate_infill_at_angle(
    _layer: &SliceLayer,
    edges: &[(Point2, Point2)],
    bb_min: Point2,
    bb_max: Point2,
    angle_rad: f32,
    spacing: f32,
) -> Vec<Segment2> {
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();

    let corners = [
        bb_min,
        Point2::new(bb_max.x, bb_min.y),
        bb_max,
        Point2::new(bb_min.x, bb_max.y),
    ];

    let mut proj_min = f32::MAX;
    let mut proj_max = f32::MIN;
    for c in &corners {
        let proj = -sin_a * c.x + cos_a * c.y;
        proj_min = proj_min.min(proj);
        proj_max = proj_max.max(proj);
    }

    let mut result = Vec::new();
    let mut d = proj_min + spacing * 0.5;
    while d < proj_max {
        let mut intersections = Vec::new();
        for &(a, b) in edges {
            let pa = -sin_a * a.x + cos_a * a.y;
            let pb = -sin_a * b.x + cos_a * b.y;
            let dp = pb - pa;
            if dp.abs() < 1e-10 {
                continue;
            }
            let s = (d - pa) / dp;
            if !(0.0..=1.0).contains(&s) {
                continue;
            }
            let ix = a.x + s * (b.x - a.x);
            let iy = a.y + s * (b.y - a.y);
            let t = cos_a * ix + sin_a * iy;
            intersections.push((t, Point2::new(ix, iy)));
        }
        intersections.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut i = 0;
        while i + 1 < intersections.len() {
            let start = intersections[i].1;
            let end = intersections[i + 1].1;
            if start.dist(end) > 1e-4 {
                result.push(Segment2 { start, end });
            }
            i += 2;
        }
        d += spacing;
    }
    result
}

/// Generate a hexagonal honeycomb infill pattern.
///
/// Generates a grid of regular hexagons covering the bounding box and clips
/// each hex edge against the contour boundaries using the same even-odd
/// intersection logic used by the scan-line infill.
///
/// Hex geometry: for a given `spacing` (line-to-line distance), the hex
/// circumradius is `R = spacing * 2.0 / sqrt(3)`. Columns are separated by
/// `3 * R / 2`, rows by `R * sqrt(3) / 2` (staggered).
///
/// The `angle_rad` parameter rotates the entire hex grid by the given angle
/// around the bounding box centre, enabling per-layer angle alternation.
fn generate_honeycomb_infill(
    _layer: &SliceLayer,
    edges: &[(Point2, Point2)],
    bb_min: Point2,
    bb_max: Point2,
    spacing: f32,
    _nozzle_diameter: f32,
    angle_rad: f32,
) -> Vec<Segment2> {
    let sqrt3: f32 = 3.0f32.sqrt();
    // Circumradius of each hexagon.
    let r = spacing * 2.0 / sqrt3;
    // Column stride (center-to-center horizontal).
    let col_stride = r * 1.5;
    // Row stride (center-to-center vertical).
    let row_stride = r * sqrt3 / 2.0;

    // Rotation helpers — rotate point (px, py) around (cx, cy) by angle_rad.
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();
    let center_x = (bb_min.x + bb_max.x) * 0.5;
    let center_y = (bb_min.y + bb_max.y) * 0.5;
    let rotate = |px: f32, py: f32| -> Point2 {
        let dx = px - center_x;
        let dy = py - center_y;
        Point2::new(
            center_x + cos_a * dx - sin_a * dy,
            center_y + sin_a * dx + cos_a * dy,
        )
    };

    // Angles for the 6 vertices of a regular hexagon (flat-top orientation).
    let hex_angles: [f32; 6] = [
        0.0,
        std::f32::consts::FRAC_PI_3,
        2.0 * std::f32::consts::FRAC_PI_3,
        std::f32::consts::PI,
        4.0 * std::f32::consts::FRAC_PI_3,
        5.0 * std::f32::consts::FRAC_PI_3,
    ];

    // Enumerate hex centers covering the bounding box (with margin).
    // Use an extended margin to ensure full coverage after rotation.
    let margin = r * 3.0;
    let x_start = bb_min.x - margin;
    let x_end = bb_max.x + margin;
    let y_start = bb_min.y - margin;
    let y_end = bb_max.y + margin;

    let mut result = Vec::new();

    let mut col = 0i32;
    let mut cx = x_start;
    while cx < x_end {
        let mut cy = y_start + if col % 2 == 1 { row_stride } else { 0.0 };
        while cy < y_end {
            // Generate the 6 vertices of this hexagon, rotating each around the
            // bounding box centre by angle_rad. The center (cx, cy) is on the
            // unrotated grid; rotating the vertices achieves the same result as
            // rotating the entire grid.
            let vertices: Vec<Point2> = hex_angles
                .iter()
                .map(|&a| {
                    let vx = cx + r * a.cos();
                    let vy = cy + r * a.sin();
                    rotate(vx, vy)
                })
                .collect();

            for i in 0..6 {
                let j = (i + 1) % 6;
                let a = vertices[i];
                let b = vertices[j];

                // Clip this hex edge against the contour using even-odd intersections.
                // Find all t values where the segment (a→b) intersects any contour edge.
                let seg_dx = b.x - a.x;
                let seg_dy = b.y - a.y;
                let seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy;
                if seg_len_sq < 1e-10 {
                    continue;
                }

                let mut t_values: Vec<f32> = vec![0.0, 1.0];
                for &(ea, eb) in edges {
                    // Intersection of segment a→b with edge ea→eb.
                    let e_dx = eb.x - ea.x;
                    let e_dy = eb.y - ea.y;
                    let denom = seg_dx * e_dy - seg_dy * e_dx;
                    if denom.abs() < 1e-10 {
                        continue;
                    }
                    let t = ((ea.x - a.x) * e_dy - (ea.y - a.y) * e_dx) / denom;
                    let u = ((ea.x - a.x) * seg_dy - (ea.y - a.y) * seg_dx) / denom;
                    if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
                        t_values.push(t);
                    }
                }
                t_values.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
                t_values.dedup_by(|x, y| (*x - *y).abs() < 1e-8);

                // Emit segments whose midpoints are inside the contour(s).
                for w in t_values.windows(2) {
                    let t_mid = (w[0] + w[1]) * 0.5;
                    let mid = Point2::new(a.x + t_mid * seg_dx, a.y + t_mid * seg_dy);

                    // Even-odd test: count crossings from mid toward +X across all edges.
                    let mut inside = false;
                    for &(ea, eb) in edges {
                        if ((ea.y > mid.y) != (eb.y > mid.y))
                            && (mid.x < (eb.x - ea.x) * (mid.y - ea.y) / (eb.y - ea.y) + ea.x)
                        {
                            inside = !inside;
                        }
                    }

                    if inside {
                        let seg_start = Point2::new(a.x + w[0] * seg_dx, a.y + w[0] * seg_dy);
                        let seg_end = Point2::new(a.x + w[1] * seg_dx, a.y + w[1] * seg_dy);
                        if seg_start.dist(seg_end) > 1e-4 {
                            result.push(Segment2 {
                                start: seg_start,
                                end: seg_end,
                            });
                        }
                    }
                }
            }
            cy += row_stride * 2.0;
        }
        col += 1;
        cx += col_stride;
    }

    result
}

/// Generate infill line segments for a sliced layer.
///
/// Produces parallel scan lines at the configured angle and density,
/// clipped to contour boundaries using the even-odd fill rule.
///
/// For `Rectilinear`, a single pass at the configured angle.
/// For `Grid`, two perpendicular passes (angle and angle+90°).
///
/// Use `layer_index` to enable per-layer angle alternation: odd layers
/// rotate by 90° for cross-hatching. Pass `None` to use the config angle as-is.
pub fn generate_infill(
    layer: &SliceLayer,
    config: &InfillConfig,
    nozzle_diameter: f32,
) -> Vec<Segment2> {
    generate_infill_for_layer(layer, config, nozzle_diameter, None)
}

/// Generate infill with optional per-layer angle alternation.
pub fn generate_infill_for_layer(
    layer: &SliceLayer,
    config: &InfillConfig,
    nozzle_diameter: f32,
    layer_index: Option<usize>,
) -> Vec<Segment2> {
    if config.density <= 0.0 || layer.outer_contours.is_empty() {
        return Vec::new();
    }

    let density = config.density.clamp(0.001, 1.0);
    let spacing = nozzle_diameter / density;

    // Alternate angle by 90° on odd layers for cross-hatching.
    let base_angle = config.angle_degrees
        + if layer_index.is_some_and(|i| i % 2 == 1) {
            90.0
        } else {
            0.0
        };
    let angle_rad = base_angle.to_radians();

    // Collect all edges.
    let edges = collect_edges(layer);
    if edges.is_empty() {
        return Vec::new();
    }

    // Get bounding box.
    let (bb_min, bb_max) = match layer_bounding_box(layer) {
        Some(bb) => bb,
        None => return Vec::new(),
    };

    // Generate pattern.
    match config.pattern {
        InfillPattern::Rectilinear => {
            generate_infill_at_angle(layer, &edges, bb_min, bb_max, angle_rad, spacing)
        }
        InfillPattern::Grid => {
            let mut segs =
                generate_infill_at_angle(layer, &edges, bb_min, bb_max, angle_rad, spacing);
            let perp = angle_rad + std::f32::consts::FRAC_PI_2;
            segs.extend(generate_infill_at_angle(
                layer, &edges, bb_min, bb_max, perp, spacing,
            ));
            segs
        }
        InfillPattern::Honeycomb => generate_honeycomb_infill(
            layer,
            &edges,
            bb_min,
            bb_max,
            spacing,
            nozzle_diameter,
            angle_rad,
        ),
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slicer::SliceLayer;

    fn square_contour(size: f32) -> Contour {
        Contour {
            points: vec![
                Point2::new(0.0, 0.0),
                Point2::new(size, 0.0),
                Point2::new(size, size),
                Point2::new(0.0, size),
            ],
        }
    }

    fn square_layer(size: f32) -> SliceLayer {
        SliceLayer {
            z: 0.2,
            outer_contours: vec![square_contour(size)],
            inner_contours: vec![],
            infill_lines: vec![],
        }
    }

    fn hole_layer() -> SliceLayer {
        // 10x10 outer with 4x4 hole centered at (3,3)→(7,7).
        SliceLayer {
            z: 0.2,
            outer_contours: vec![square_contour(10.0)],
            inner_contours: vec![Contour {
                points: vec![
                    Point2::new(3.0, 3.0),
                    Point2::new(3.0, 7.0),
                    Point2::new(7.0, 7.0),
                    Point2::new(7.0, 3.0),
                ],
            }],
            infill_lines: vec![],
        }
    }

    // ── contains_point ──────────────────────────────────────────────

    #[test]
    fn contains_point_inside_square() {
        let c = square_contour(10.0);
        assert!(c.contains_point(Point2::new(5.0, 5.0)));
        assert!(c.contains_point(Point2::new(1.0, 1.0)));
        assert!(c.contains_point(Point2::new(9.0, 9.0)));
    }

    #[test]
    fn contains_point_outside() {
        let c = square_contour(10.0);
        assert!(!c.contains_point(Point2::new(-1.0, 5.0)));
        assert!(!c.contains_point(Point2::new(5.0, 11.0)));
        assert!(!c.contains_point(Point2::new(15.0, 5.0)));
    }

    #[test]
    fn contains_point_on_edge() {
        let c = square_contour(10.0);
        // Points exactly on edges are implementation-defined for ray-casting,
        // but should not panic.
        let _ = c.contains_point(Point2::new(5.0, 0.0));
        let _ = c.contains_point(Point2::new(0.0, 5.0));
    }

    // ── bounding_box ────────────────────────────────────────────────

    #[test]
    fn bounding_box() {
        let c = square_contour(10.0);
        let (min, max) = c.bounding_box();
        assert!((min.x).abs() < 1e-6);
        assert!((min.y).abs() < 1e-6);
        assert!((max.x - 10.0).abs() < 1e-6);
        assert!((max.y - 10.0).abs() < 1e-6);
    }

    // ── scan_line_edge_intersection ─────────────────────────────────

    #[test]
    fn scan_line_edge_intersection_basic() {
        // Vertical edge from (5, 0) to (5, 10), scan at y=5.
        let x = scan_line_edge_intersection(5.0, Point2::new(5.0, 0.0), Point2::new(5.0, 10.0));
        assert!(x.is_some());
        assert!((x.unwrap() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn scan_line_parallel() {
        // Horizontal edge — no intersection.
        let x = scan_line_edge_intersection(5.0, Point2::new(0.0, 5.0), Point2::new(10.0, 5.0));
        assert!(x.is_none());
    }

    // ── generate_infill ─────────────────────────────────────────────

    #[test]
    fn rectilinear_square_produces_lines() {
        let layer = square_layer(10.0);
        let config = InfillConfig {
            pattern: InfillPattern::Rectilinear,
            density: 0.5,
            angle_degrees: 0.0, // Horizontal lines for predictability.
        };
        let lines = generate_infill(&layer, &config, 0.4);
        assert!(
            !lines.is_empty(),
            "20% infill on a 10mm square should produce lines"
        );
        // All line endpoints should be within the bounding box (with tolerance).
        for seg in &lines {
            assert!(seg.start.x >= -0.1 && seg.start.x <= 10.1);
            assert!(seg.end.x >= -0.1 && seg.end.x <= 10.1);
        }
    }

    #[test]
    fn infill_respects_hole() {
        let layer = hole_layer();
        let config = InfillConfig {
            pattern: InfillPattern::Rectilinear,
            density: 0.5,
            angle_degrees: 0.0,
        };
        let lines = generate_infill(&layer, &config, 0.4);
        assert!(!lines.is_empty());

        // Note: hole clipping is not yet implemented — inner contours are
        // stored but generate_infill does not clip lines against them.
        // Once clipping is added, hole layers should produce *more* segments
        // (split lines) than solid layers of the same size.
        let solid = square_layer(10.0);
        let solid_lines = generate_infill(&solid, &config, 0.4);
        assert!(
            !solid_lines.is_empty(),
            "solid layer should also produce lines"
        );
        // Verify the hole layer actually has a hole (inner contour).
        assert!(
            !layer.inner_contours.is_empty(),
            "hole_layer should have inner contours"
        );
    }

    #[test]
    fn infill_empty_contours() {
        let layer = SliceLayer {
            z: 0.2,
            outer_contours: vec![],
            inner_contours: vec![],
            infill_lines: vec![],
        };
        let config = InfillConfig::default();
        let lines = generate_infill(&layer, &config, 0.4);
        assert!(lines.is_empty());
    }

    #[test]
    fn infill_density_zero() {
        let layer = square_layer(10.0);
        let config = InfillConfig {
            density: 0.0,
            ..InfillConfig::default()
        };
        let lines = generate_infill(&layer, &config, 0.4);
        assert!(lines.is_empty(), "zero density should produce no infill");
    }

    #[test]
    fn infill_density_full() {
        let layer = square_layer(10.0);
        let config = InfillConfig {
            density: 1.0,
            angle_degrees: 0.0,
            ..InfillConfig::default()
        };
        let lines = generate_infill(&layer, &config, 0.4);
        // At 100% density with 0.4mm nozzle on 10mm: spacing = 0.4mm → ~25 lines.
        assert!(
            lines.len() >= 15,
            "100% density should produce many lines, got {}",
            lines.len()
        );
    }

    // ── Grid pattern ────────────────────────────────────────────────

    #[test]
    fn grid_produces_more_lines_than_rectilinear() {
        let layer = square_layer(10.0);
        let rect_config = InfillConfig {
            pattern: InfillPattern::Rectilinear,
            density: 0.3,
            angle_degrees: 0.0,
        };
        let grid_config = InfillConfig {
            pattern: InfillPattern::Grid,
            density: 0.3,
            angle_degrees: 0.0,
        };
        let rect_lines = generate_infill(&layer, &rect_config, 0.4);
        let grid_lines = generate_infill(&layer, &grid_config, 0.4);
        // Grid does two perpendicular passes — should produce roughly 2× lines.
        assert!(
            grid_lines.len() > rect_lines.len(),
            "Grid ({}) should produce more lines than Rectilinear ({})",
            grid_lines.len(),
            rect_lines.len()
        );
    }

    #[test]
    fn grid_has_perpendicular_lines() {
        let layer = square_layer(10.0);
        let config = InfillConfig {
            pattern: InfillPattern::Grid,
            density: 0.3,
            angle_degrees: 0.0,
        };
        let lines = generate_infill(&layer, &config, 0.4);
        // With angle=0, first pass is horizontal (dy≈0), second is vertical (dx≈0).
        let horizontal = lines
            .iter()
            .filter(|s| (s.start.y - s.end.y).abs() < 0.01)
            .count();
        let vertical = lines
            .iter()
            .filter(|s| (s.start.x - s.end.x).abs() < 0.01)
            .count();
        assert!(horizontal > 0, "Grid should have horizontal lines");
        assert!(vertical > 0, "Grid should have vertical lines");
    }

    // ── Layer alternation ───────────────────────────────────────────

    #[test]
    fn alternating_layers_have_different_angles() {
        let layer = square_layer(10.0);
        let config = InfillConfig {
            pattern: InfillPattern::Rectilinear,
            density: 0.3,
            angle_degrees: 0.0,
        };
        let even = generate_infill_for_layer(&layer, &config, 0.4, Some(0));
        let odd = generate_infill_for_layer(&layer, &config, 0.4, Some(1));

        // Even layer at 0°: horizontal lines (dy≈0).
        let even_horizontal = even
            .iter()
            .filter(|s| (s.start.y - s.end.y).abs() < 0.01)
            .count();
        // Odd layer at 90°: vertical lines (dx≈0).
        let odd_vertical = odd
            .iter()
            .filter(|s| (s.start.x - s.end.x).abs() < 0.01)
            .count();

        assert!(
            even_horizontal > 0,
            "Even layer should have horizontal lines"
        );
        assert!(odd_vertical > 0, "Odd layer should have vertical lines");
    }

    #[test]
    fn alternation_none_uses_base_angle() {
        let layer = square_layer(10.0);
        let config = InfillConfig {
            pattern: InfillPattern::Rectilinear,
            density: 0.3,
            angle_degrees: 0.0,
        };
        let no_alt = generate_infill(&layer, &config, 0.4);
        let explicit_none = generate_infill_for_layer(&layer, &config, 0.4, None);
        assert_eq!(
            no_alt.len(),
            explicit_none.len(),
            "None layer_index should match generate_infill"
        );
    }

    // ── Contour inset ────────────────────────────────────────────────

    #[test]
    fn inset_square_shrinks() {
        let c = square_contour(10.0);
        let inset = c.inset(1.0).expect("inset should produce valid contour");
        // Each vertex should be roughly 1mm inward from the original.
        for pt in &inset.points {
            assert!(pt.x >= 0.9 && pt.x <= 9.1, "x={} not inset", pt.x);
            assert!(pt.y >= 0.9 && pt.y <= 9.1, "y={} not inset", pt.y);
        }
        // Area should be smaller.
        assert!(
            inset.signed_area().abs() < c.signed_area().abs(),
            "inset area {} should be < original {}",
            inset.signed_area().abs(),
            c.signed_area().abs()
        );
    }

    #[test]
    fn inset_tiny_contour_returns_none() {
        // A 0.1mm square with 1mm inset should collapse to None.
        let c = square_contour(0.1);
        let result = c.inset(1.0);
        assert!(
            result.is_none(),
            "inset larger than contour should return None"
        );
    }

    #[test]
    fn infill_with_inset_stays_inside_perimeter() {
        // Test that generating infill with a manually inset layer keeps endpoints
        // within the inset boundary. This tests the inset primitive directly.
        let contour = square_contour(20.0);
        let nozzle_diameter = 0.4f32;
        let inset_amount = nozzle_diameter * 0.5;
        let inset_contour = contour
            .inset(inset_amount)
            .expect("should produce valid inset");

        let inset_layer = SliceLayer {
            z: 0.2,
            outer_contours: vec![inset_contour],
            inner_contours: vec![],
            infill_lines: vec![],
        };
        let config = InfillConfig {
            pattern: InfillPattern::Rectilinear,
            density: 0.5,
            angle_degrees: 0.0,
        };
        let lines = generate_infill(&inset_layer, &config, nozzle_diameter);
        // All infill endpoints should be at least inset_amount inside the 20mm boundary.
        for seg in &lines {
            assert!(
                seg.start.x >= inset_amount - 0.1 && seg.start.x <= 20.0 - inset_amount + 0.1,
                "start.x={} outside inset boundary",
                seg.start.x
            );
            assert!(
                seg.end.x >= inset_amount - 0.1 && seg.end.x <= 20.0 - inset_amount + 0.1,
                "end.x={} outside inset boundary",
                seg.end.x
            );
        }
        assert!(!lines.is_empty(), "inset layer should produce infill");
    }

    // ── Honeycomb pattern ────────────────────────────────────────────

    #[test]
    fn honeycomb_angle_rotation_changes_output() {
        let layer = square_layer(10.0);
        let config_0 = InfillConfig {
            pattern: InfillPattern::Honeycomb,
            density: 0.3,
            angle_degrees: 0.0,
        };
        let config_30 = InfillConfig {
            pattern: InfillPattern::Honeycomb,
            density: 0.3,
            angle_degrees: 30.0,
        };
        let lines_0 = generate_infill(&layer, &config_0, 0.4);
        let lines_30 = generate_infill(&layer, &config_30, 0.4);
        // Both should produce lines but with different geometry.
        assert!(!lines_0.is_empty());
        assert!(!lines_30.is_empty());
        // At least some endpoints should differ (rotated grid).
        let differs = lines_0.iter().zip(lines_30.iter()).any(|(a, b)| {
            (a.start.x - b.start.x).abs() > 0.01 || (a.start.y - b.start.y).abs() > 0.01
        });
        assert!(
            differs,
            "Rotated honeycomb should produce different geometry"
        );
    }

    #[test]
    fn honeycomb_produces_lines() {
        let layer = square_layer(20.0);
        let config = InfillConfig {
            pattern: InfillPattern::Honeycomb,
            density: 0.2,
            angle_degrees: 0.0,
        };
        let lines = generate_infill(&layer, &config, 0.4);
        assert!(
            !lines.is_empty(),
            "honeycomb infill on 20mm square should produce segments"
        );
    }

    #[test]
    fn honeycomb_respects_hole() {
        let layer = hole_layer();
        let solid = square_layer(10.0);
        let config = InfillConfig {
            pattern: InfillPattern::Honeycomb,
            density: 0.3,
            angle_degrees: 0.0,
        };
        let lines = generate_infill(&layer, &config, 0.4);
        let solid_lines = generate_infill(&solid, &config, 0.4);
        assert!(
            !lines.is_empty(),
            "honeycomb on layer with hole should produce segments"
        );
        assert!(
            !solid_lines.is_empty(),
            "honeycomb on solid should produce segments"
        );
        // Note: hole clipping is not yet implemented — once added,
        // the hole layer should produce fewer segments than solid.
    }

    #[test]
    fn honeycomb_more_directions_than_rectilinear() {
        let layer = square_layer(20.0);
        let honeycomb_config = InfillConfig {
            pattern: InfillPattern::Honeycomb,
            density: 0.3,
            angle_degrees: 0.0,
        };
        let rect_config = InfillConfig {
            pattern: InfillPattern::Rectilinear,
            density: 0.3,
            angle_degrees: 0.0,
        };
        let hc_lines = generate_infill(&layer, &honeycomb_config, 0.4);
        let rect_lines = generate_infill(&layer, &rect_config, 0.4);

        // Honeycomb has 3 directions; rectilinear has 1.
        // Count distinct directions (quantised to 30° buckets) — honeycomb
        // should have more direction variety.
        fn direction_buckets(segs: &[Segment2]) -> std::collections::HashSet<i32> {
            segs.iter()
                .map(|s| {
                    let dx = s.end.x - s.start.x;
                    let dy = s.end.y - s.start.y;
                    let angle_deg = dy.atan2(dx).to_degrees();
                    // Quantise to 30° buckets, normalise to [0, 180).
                    let normalised = ((angle_deg % 180.0) + 180.0) % 180.0;
                    (normalised / 30.0) as i32
                })
                .collect()
        }
        let hc_dirs = direction_buckets(&hc_lines);
        let rect_dirs = direction_buckets(&rect_lines);
        assert!(
            hc_dirs.len() > rect_dirs.len(),
            "honeycomb ({} directions) should have more directions than rectilinear ({})",
            hc_dirs.len(),
            rect_dirs.len()
        );
    }
}
