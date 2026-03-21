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

    // First pass at the base angle.
    let mut result = generate_infill_at_angle(layer, &edges, bb_min, bb_max, angle_rad, spacing);

    // Grid pattern: add a second perpendicular pass.
    if config.pattern == InfillPattern::Grid {
        let perp = angle_rad + std::f32::consts::FRAC_PI_2;
        let pass2 = generate_infill_at_angle(layer, &edges, bb_min, bb_max, perp, spacing);
        result.extend(pass2);
    }

    result
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

        // Lines passing through the hole region should be split (more segments
        // than a solid square of the same size).
        let solid = square_layer(10.0);
        let solid_lines = generate_infill(&solid, &config, 0.4);
        assert!(
            lines.len() > solid_lines.len(),
            "hole should split some lines: {} vs {}",
            lines.len(),
            solid_lines.len()
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
}
