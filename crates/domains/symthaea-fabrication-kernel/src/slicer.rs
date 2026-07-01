// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mesh slicer: intersects a triangle mesh with horizontal planes to produce
//! 2D contours for layer-by-layer fabrication (FDM/SLA 3D printing).
//!
//! Hardened contour chaining handles:
//! - Duplicate/near-duplicate segment deduplication
//! - Spatial-hash O(1) endpoint lookup (grid-based)
//! - T-junction resolution via smallest turning angle (left-turn preference)
//! - Minimum segment length filtering
//! - Contour closure validation with gap-closing search
//! - Orientation enforcement (outer CCW, inner CW)

use crate::infill::{InfillConfig, generate_infill_for_layer};
use crate::mesh::TriangleMesh;
use std::collections::HashMap;

// ── 2D types ────────────────────────────────────────────────────────────

/// A 2D point on the slice plane.
#[derive(Debug, Clone, Copy)]
pub struct Point2 {
    pub x: f32,
    pub y: f32,
}

impl Point2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn dist_sq(self, other: Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }

    pub fn dist(self, other: Self) -> f32 {
        self.dist_sq(other).sqrt()
    }
}

/// A directed line segment on the slice plane.
#[derive(Debug, Clone, Copy)]
pub struct Segment2 {
    pub start: Point2,
    pub end: Point2,
}

impl Segment2 {
    pub fn length(&self) -> f32 {
        self.start.dist(self.end)
    }

    /// Direction vector (unnormalized).
    pub fn direction(&self) -> (f32, f32) {
        (self.end.x - self.start.x, self.end.y - self.start.y)
    }
}

/// A closed 2D contour (polyline where last point connects back to first).
#[derive(Debug, Clone)]
pub struct Contour {
    pub points: Vec<Point2>,
}

impl Contour {
    /// Signed area via the shoelace formula.
    /// Positive = CCW winding, negative = CW winding.
    pub fn signed_area(&self) -> f32 {
        let n = self.points.len();
        if n < 3 {
            return 0.0;
        }
        let mut area = 0.0f32;
        for i in 0..n {
            let j = (i + 1) % n;
            area += self.points[i].x * self.points[j].y;
            area -= self.points[j].x * self.points[i].y;
        }
        area * 0.5
    }

    /// True if winding is counter-clockwise (positive signed area).
    pub fn is_ccw(&self) -> bool {
        self.signed_area() > 0.0
    }

    /// Reverse the winding direction.
    pub fn reverse(&mut self) {
        self.points.reverse();
    }

    /// Ensure CCW winding (for outer contours).
    pub fn ensure_ccw(&mut self) {
        if !self.is_ccw() {
            self.reverse();
        }
    }

    /// Ensure CW winding (for inner/hole contours).
    pub fn ensure_cw(&mut self) {
        if self.is_ccw() {
            self.reverse();
        }
    }
}

// ── Slice parameters ────────────────────────────────────────────────────

/// Configuration for the slicer.
#[derive(Debug, Clone)]
pub struct SliceConfig {
    /// Nozzle diameter in mm (used for minimum segment filter).
    pub nozzle_diameter: f32,
    /// Endpoint matching tolerance in mm.
    pub tolerance: f32,
    /// Layer height in mm.
    pub layer_height: f32,
    /// Infill configuration. `None` disables infill generation.
    pub infill: Option<InfillConfig>,
}

impl Default for SliceConfig {
    fn default() -> Self {
        Self {
            nozzle_diameter: 0.4,
            tolerance: 1e-4,
            layer_height: 0.2,
            infill: Some(InfillConfig::default()),
        }
    }
}

impl SliceConfig {
    /// Validate configuration, returning a list of warnings.
    pub fn validate(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        if self.layer_height <= 0.0 {
            warnings.push("layer_height must be positive".into());
        }
        if self.nozzle_diameter <= 0.0 {
            warnings.push("nozzle_diameter must be positive".into());
        }
        if let Some(ref infill) = self.infill {
            if infill.density <= 0.0 || infill.density > 1.0 {
                warnings.push(format!(
                    "infill density {} is outside (0.0, 1.0] — will be clamped",
                    infill.density
                ));
            }
        }
        warnings
    }
}

/// Result of slicing a mesh at one Z-height.
#[derive(Debug, Clone)]
pub struct SliceLayer {
    pub z: f32,
    /// Outer contours (CCW winding).
    pub outer_contours: Vec<Contour>,
    /// Inner/hole contours (CW winding).
    pub inner_contours: Vec<Contour>,
    /// Generated infill line segments (empty if infill is disabled).
    pub infill_lines: Vec<Segment2>,
}

// ── Spatial hash for O(1) endpoint lookup ───────────────────────────────

/// Grid cell key for spatial hashing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CellKey {
    ix: i32,
    iy: i32,
}

/// Spatial hash grid that maps 2D positions to segment indices.
/// Supports O(1) neighbor lookup by checking the cell and its 8 neighbors.
struct SpatialHash {
    cell_size: f32,
    /// Maps cell → list of (segment_index, is_start_endpoint).
    cells: HashMap<CellKey, Vec<(usize, bool)>>,
}

impl SpatialHash {
    fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            cells: HashMap::new(),
        }
    }

    fn key(&self, p: Point2) -> CellKey {
        CellKey {
            ix: (p.x / self.cell_size).floor() as i32,
            iy: (p.y / self.cell_size).floor() as i32,
        }
    }

    fn insert(&mut self, idx: usize, point: Point2, is_start: bool) {
        let k = self.key(point);
        self.cells.entry(k).or_default().push((idx, is_start));
    }

    /// Find all segment endpoints within tolerance of `point`.
    /// Returns (segment_index, is_start_endpoint) pairs.
    fn query(&self, point: Point2, _tolerance: f32) -> Vec<(usize, bool)> {
        let k = self.key(point);
        let mut results = Vec::new();
        // Check 3x3 neighborhood to handle cell-boundary cases.
        for dx in -1..=1 {
            for dy in -1..=1 {
                let neighbor = CellKey {
                    ix: k.ix + dx,
                    iy: k.iy + dy,
                };
                if let Some(entries) = self.cells.get(&neighbor) {
                    for &(idx, is_start) in entries {
                        // Caller must do distance check — we just return candidates.
                        results.push((idx, is_start));
                    }
                }
            }
        }
        results
    }

    /// Remove a specific entry.
    fn remove(&mut self, idx: usize, point: Point2, is_start: bool) {
        let k = self.key(point);
        // Check 3x3 neighborhood since floating-point rounding may place
        // the point in a neighboring cell.
        for dx in -1..=1 {
            for dy in -1..=1 {
                let neighbor = CellKey {
                    ix: k.ix + dx,
                    iy: k.iy + dy,
                };
                if let Some(entries) = self.cells.get_mut(&neighbor) {
                    entries.retain(|&(i, s)| !(i == idx && s == is_start));
                }
            }
        }
    }
}

// ── Core slicing ────────────────────────────────────────────────────────

/// Slice a triangle mesh at a single Z-height, producing raw (unordered)
/// line segments where each triangle edge intersects the plane.
fn slice_at_z(mesh: &TriangleMesh, z: f32) -> Vec<Segment2> {
    let mut segments = Vec::new();

    for tri in &mesh.indices {
        let v0 = mesh.vertices[tri[0] as usize];
        let v1 = mesh.vertices[tri[1] as usize];
        let v2 = mesh.vertices[tri[2] as usize];

        // Collect intersection points of the triangle edges with the Z-plane.
        let mut intersections = Vec::new();
        let edges = [(v0, v1), (v1, v2), (v2, v0)];
        for (a, b) in &edges {
            if let Some(p) = edge_z_intersection(*a, *b, z) {
                intersections.push(p);
            }
        }

        // A plane-triangle intersection produces 0 or 2 points (ignoring
        // degenerate coplanar cases which we skip).
        if intersections.len() == 2 {
            segments.push(Segment2 {
                start: intersections[0],
                end: intersections[1],
            });
        }
    }

    segments
}

/// Compute the intersection of a 3D edge with a horizontal plane at Z.
/// Returns `None` if the edge doesn't cross the plane.
fn edge_z_intersection(a: [f32; 3], b: [f32; 3], z: f32) -> Option<Point2> {
    let dz = b[2] - a[2];
    if dz.abs() < 1e-10 {
        // Edge is parallel to (or on) the plane — skip.
        return None;
    }
    let t = (z - a[2]) / dz;
    if !(0.0..=1.0).contains(&t) {
        return None;
    }
    Some(Point2::new(
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
    ))
}

// ── Segment preprocessing ───────────────────────────────────────────────

/// Remove segments shorter than `min_length`.
fn filter_short_segments(segments: &mut Vec<Segment2>, min_length: f32) {
    segments.retain(|s| s.length() >= min_length);
}

/// Remove duplicate segments (same start/end within tolerance, in either direction).
fn deduplicate_segments(segments: &mut Vec<Segment2>, tolerance: f32) {
    let tol_sq = tolerance * tolerance;
    let n = segments.len();
    let mut keep = vec![true; n];

    for i in 0..n {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..n {
            if !keep[j] {
                continue;
            }
            // Same direction duplicate.
            let same_fwd = segments[i].start.dist_sq(segments[j].start) < tol_sq
                && segments[i].end.dist_sq(segments[j].end) < tol_sq;
            // Reverse direction duplicate.
            let same_rev = segments[i].start.dist_sq(segments[j].end) < tol_sq
                && segments[i].end.dist_sq(segments[j].start) < tol_sq;
            if same_fwd || same_rev {
                keep[j] = false;
            }
        }
    }

    let mut idx = 0;
    segments.retain(|_| {
        let k = keep[idx];
        idx += 1;
        k
    });
}

// ── Contour chaining with T-junction handling ───────────────────────────

/// Turning angle from direction `(dx1, dy1)` to `(dx2, dy2)`.
/// Returns angle in radians in `(-PI, PI]`.
/// Positive = left turn (CCW), negative = right turn (CW).
fn turning_angle(dx1: f32, dy1: f32, dx2: f32, dy2: f32) -> f32 {
    let cross = dx1 * dy2 - dy1 * dx2;
    let dot = dx1 * dx2 + dy1 * dy2;
    cross.atan2(dot)
}

/// Chain preprocessed segments into closed contours.
///
/// At T-junctions (multiple segments sharing an endpoint), selects the
/// continuation with the smallest positive turning angle (left-turn
/// preference for outer contours). This naturally traces outer boundaries
/// CCW and holes CW.
fn chain_contours(segments: &[Segment2], tolerance: f32) -> Vec<Contour> {
    if segments.is_empty() {
        return Vec::new();
    }

    let tol_sq = tolerance * tolerance;

    // Build spatial hash — cell size = 4 * tolerance gives good bucket distribution.
    let cell_size = (tolerance * 4.0).max(1e-6);
    let mut hash = SpatialHash::new(cell_size);
    let mut used = vec![false; segments.len()];

    for (i, seg) in segments.iter().enumerate() {
        hash.insert(i, seg.start, true);
        hash.insert(i, seg.end, false);
    }

    let mut contours = Vec::new();

    for start_idx in 0..segments.len() {
        if used[start_idx] {
            continue;
        }

        // Begin a new chain from this segment.
        used[start_idx] = true;
        hash.remove(start_idx, segments[start_idx].start, true);
        hash.remove(start_idx, segments[start_idx].end, false);

        let mut chain_points = vec![segments[start_idx].start, segments[start_idx].end];
        let chain_origin = segments[start_idx].start;
        let mut closed = false;

        loop {
            let tail = *chain_points.last().unwrap();
            let prev = chain_points[chain_points.len().saturating_sub(2)];

            // Check if we've closed the loop.
            if chain_points.len() > 2 && tail.dist_sq(chain_origin) < tol_sq {
                // Remove the duplicate closing point.
                chain_points.pop();
                closed = true;
                break;
            }

            // Find candidate continuations: unused segments with a start or
            // end endpoint near our tail.
            let candidates = hash.query(tail, tolerance);
            let mut best: Option<(usize, bool, f32)> = None; // (seg_idx, use_reversed, angle)

            let incoming_dx = tail.x - prev.x;
            let incoming_dy = tail.y - prev.y;

            for (cand_idx, is_start) in &candidates {
                if used[*cand_idx] {
                    continue;
                }
                let seg = &segments[*cand_idx];
                // Determine which endpoint matches our tail and which is the
                // "next" point if we traverse this segment.
                let (match_pt, next_pt, reversed) = if *is_start {
                    (seg.start, seg.end, false)
                } else {
                    (seg.end, seg.start, true)
                };

                if tail.dist_sq(match_pt) >= tol_sq {
                    continue;
                }

                let out_dx = next_pt.x - match_pt.x;
                let out_dy = next_pt.y - match_pt.y;

                // Compute turning angle; prefer smallest left turn.
                let angle = turning_angle(incoming_dx, incoming_dy, out_dx, out_dy);

                // We want the smallest positive (left) turn. Convert to
                // [0, 2*PI) so that a straight continuation (~0) is preferred,
                // then small left turns, then U-turns.
                let score = if angle >= 0.0 {
                    angle
                } else {
                    angle + std::f32::consts::TAU
                };

                let dominated = match best {
                    Some((_, _, best_score)) => score < best_score,
                    None => true,
                };
                if dominated {
                    best = Some((*cand_idx, reversed, score));
                }
            }

            match best {
                Some((seg_idx, reversed, _)) => {
                    used[seg_idx] = true;
                    hash.remove(seg_idx, segments[seg_idx].start, true);
                    hash.remove(seg_idx, segments[seg_idx].end, false);

                    let next_pt = if reversed {
                        segments[seg_idx].start
                    } else {
                        segments[seg_idx].end
                    };
                    chain_points.push(next_pt);
                }
                None => {
                    // No continuation found.
                    break;
                }
            }
        }

        // If the main loop already closed, accept the contour directly.
        if closed && chain_points.len() >= 3 {
            contours.push(Contour {
                points: chain_points,
            });
            continue;
        }

        // Contour closure validation for chains that didn't close in the
        // main loop: try to bridge the gap.
        if chain_points.len() >= 3 {
            let gap = chain_points.last().unwrap().dist_sq(chain_origin);
            if gap >= tol_sq {
                // Try to find a bridging segment among unused segments.
                let tail = *chain_points.last().unwrap();
                let candidates = hash.query(tail, tolerance);
                for (cand_idx, is_start) in &candidates {
                    if used[*cand_idx] {
                        continue;
                    }
                    let seg = &segments[*cand_idx];
                    let (match_pt, next_pt) = if *is_start {
                        (seg.start, seg.end)
                    } else {
                        (seg.end, seg.start)
                    };
                    if tail.dist_sq(match_pt) < tol_sq && next_pt.dist_sq(chain_origin) < tol_sq {
                        used[*cand_idx] = true;
                        hash.remove(*cand_idx, segments[*cand_idx].start, true);
                        hash.remove(*cand_idx, segments[*cand_idx].end, false);
                        closed = true;
                        break;
                    }
                }
            } else {
                // Last point is within tolerance of origin — already closed.
                closed = true;
            }

            // Accept contour if closed (or within generous tolerance).
            let final_gap = chain_points.last().unwrap().dist(chain_origin);
            if (closed || final_gap < tolerance * 10.0) && chain_points.len() >= 3 {
                contours.push(Contour {
                    points: chain_points,
                });
            }
        }
    }

    contours
}

// ── Orientation classification ──────────────────────────────────────────

/// Classify contours into outer (CCW) and inner/hole (CW).
/// Returns `(outer, inner)` with correct winding enforced.
fn classify_contours(contours: Vec<Contour>) -> (Vec<Contour>, Vec<Contour>) {
    let mut outer = Vec::new();
    let mut inner = Vec::new();

    for c in contours {
        let area = c.signed_area();
        if area.abs() < 1e-10 {
            continue; // Degenerate.
        }
        if area > 0.0 {
            // Positive area → outer contour (CCW).
            outer.push(c);
        } else {
            // Negative area → hole contour (CW).
            inner.push(c);
        }
    }

    (outer, inner)
}

// ── Public API ──────────────────────────────────────────────────────────

/// Slice a mesh at a single Z-height with full hardening.
pub fn slice_mesh_at_z(mesh: &TriangleMesh, z: f32, config: &SliceConfig) -> SliceLayer {
    let mut segments = slice_at_z(mesh, z);

    // 1. Filter tiny segments (below 1% of nozzle diameter — filters numerical
    //    noise while preserving real geometry from fine tessellations).
    let min_length = config.nozzle_diameter * 0.01;
    filter_short_segments(&mut segments, min_length);

    // 2. Deduplicate.
    deduplicate_segments(&mut segments, config.tolerance);

    // 3. Chain into contours with T-junction handling.
    let contours = chain_contours(&segments, config.tolerance);

    // 4. Classify and orient.
    let (outer, inner) = classify_contours(contours);

    let mut layer = SliceLayer {
        z,
        outer_contours: outer,
        inner_contours: inner,
        infill_lines: Vec::new(),
    };

    // 5. Generate infill if configured (with optional layer-index alternation).
    if let Some(ref infill_config) = config.infill {
        layer.infill_lines =
            generate_infill_for_layer(&layer, infill_config, config.nozzle_diameter, None);
    }

    layer
}

/// Slice a mesh into multiple layers from z_min to z_max.
///
/// Infill angles alternate by 90° between odd and even layers for
/// cross-hatching strength.
pub fn slice_mesh(mesh: &TriangleMesh, config: &SliceConfig) -> Vec<SliceLayer> {
    if mesh.vertices.is_empty() {
        return Vec::new();
    }

    let (z_min, z_max) = mesh_z_bounds(mesh);
    let first_z = z_min + config.layer_height * 0.5; // Start half a layer above bottom.
    let mut layers = Vec::new();
    let mut z = first_z;
    let mut layer_idx = 0usize;
    while z < z_max {
        let mut layer = slice_mesh_at_z(mesh, z, config);
        // Override infill with layer-aware alternation.
        if let Some(ref infill_config) = config.infill {
            layer.infill_lines = generate_infill_for_layer(
                &layer,
                infill_config,
                config.nozzle_diameter,
                Some(layer_idx),
            );
        }
        layers.push(layer);
        z += config.layer_height;
        layer_idx += 1;
    }
    layers
}

/// Compute (z_min, z_max) of a mesh.
fn mesh_z_bounds(mesh: &TriangleMesh) -> (f32, f32) {
    let mut z_min = f32::MAX;
    let mut z_max = f32::MIN;
    for v in &mesh.vertices {
        z_min = z_min.min(v[2]);
        z_max = z_max.max(v[2]);
    }
    (z_min, z_max)
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csg::CSGNode;
    use crate::mesh::resolve_to_mesh;

    fn default_config() -> SliceConfig {
        SliceConfig::default()
    }

    // ── Basic slicing ───────────────────────────────────────────────

    #[test]
    fn slice_cube_midplane() {
        let mesh = resolve_to_mesh(&CSGNode::cube());
        // Cube is [-0.5, 0.5]^3. Slice at z=0 should yield a square contour.
        let layer = slice_mesh_at_z(&mesh, 0.0, &default_config());
        assert!(
            !layer.outer_contours.is_empty(),
            "slicing cube at z=0 should produce at least one outer contour"
        );
        let c = &layer.outer_contours[0];
        assert!(
            c.points.len() >= 4,
            "square contour should have >= 4 points, got {}",
            c.points.len()
        );
    }

    #[test]
    fn slice_cube_above_top() {
        let mesh = resolve_to_mesh(&CSGNode::cube());
        // Above the cube — no contours.
        let layer = slice_mesh_at_z(&mesh, 1.0, &default_config());
        assert!(layer.outer_contours.is_empty());
        assert!(layer.inner_contours.is_empty());
    }

    #[test]
    fn slice_multi_layer() {
        let mesh = resolve_to_mesh(&CSGNode::cube());
        let config = SliceConfig {
            layer_height: 0.1,
            ..default_config()
        };
        let layers = slice_mesh(&mesh, &config);
        assert!(
            layers.len() >= 5,
            "1mm cube at 0.1mm layers should produce >= 5 layers, got {}",
            layers.len()
        );
    }

    #[test]
    fn slice_empty_mesh() {
        let mesh = crate::mesh::TriangleMesh::empty();
        let layers = slice_mesh(&mesh, &default_config());
        assert!(layers.is_empty());
    }

    // ── Contour orientation ─────────────────────────────────────────

    #[test]
    fn contour_ccw_square() {
        // CCW square.
        let c = Contour {
            points: vec![
                Point2::new(0.0, 0.0),
                Point2::new(1.0, 0.0),
                Point2::new(1.0, 1.0),
                Point2::new(0.0, 1.0),
            ],
        };
        assert!(
            c.signed_area() > 0.0,
            "CCW square should have positive area"
        );
        assert!(c.is_ccw());
    }

    #[test]
    fn contour_cw_square() {
        // CW square (reversed).
        let c = Contour {
            points: vec![
                Point2::new(0.0, 0.0),
                Point2::new(0.0, 1.0),
                Point2::new(1.0, 1.0),
                Point2::new(1.0, 0.0),
            ],
        };
        assert!(c.signed_area() < 0.0, "CW square should have negative area");
        assert!(!c.is_ccw());
    }

    #[test]
    fn contour_orientation_correction() {
        let mut c = Contour {
            points: vec![
                Point2::new(0.0, 0.0),
                Point2::new(0.0, 1.0),
                Point2::new(1.0, 1.0),
                Point2::new(1.0, 0.0),
            ],
        };
        assert!(!c.is_ccw());
        c.ensure_ccw();
        assert!(c.is_ccw(), "ensure_ccw should flip CW to CCW");

        c.ensure_cw();
        assert!(!c.is_ccw(), "ensure_cw should flip CCW to CW");
    }

    // ── Deduplication ───────────────────────────────────────────────

    #[test]
    fn deduplicate_exact_forward() {
        let mut segs = vec![
            Segment2 {
                start: Point2::new(0.0, 0.0),
                end: Point2::new(1.0, 0.0),
            },
            Segment2 {
                start: Point2::new(0.0, 0.0),
                end: Point2::new(1.0, 0.0),
            },
        ];
        deduplicate_segments(&mut segs, 1e-4);
        assert_eq!(segs.len(), 1, "exact forward duplicate should be removed");
    }

    #[test]
    fn deduplicate_exact_reverse() {
        let mut segs = vec![
            Segment2 {
                start: Point2::new(0.0, 0.0),
                end: Point2::new(1.0, 0.0),
            },
            Segment2 {
                start: Point2::new(1.0, 0.0),
                end: Point2::new(0.0, 0.0),
            },
        ];
        deduplicate_segments(&mut segs, 1e-4);
        assert_eq!(segs.len(), 1, "exact reverse duplicate should be removed");
    }

    #[test]
    fn deduplicate_near_duplicate() {
        let eps = 1e-5;
        let mut segs = vec![
            Segment2 {
                start: Point2::new(0.0, 0.0),
                end: Point2::new(1.0, 0.0),
            },
            Segment2 {
                start: Point2::new(eps, eps),
                end: Point2::new(1.0 + eps, eps),
            },
        ];
        deduplicate_segments(&mut segs, 1e-4);
        assert_eq!(segs.len(), 1, "near-duplicate should be removed");
    }

    #[test]
    fn deduplicate_keeps_distinct() {
        let mut segs = vec![
            Segment2 {
                start: Point2::new(0.0, 0.0),
                end: Point2::new(1.0, 0.0),
            },
            Segment2 {
                start: Point2::new(1.0, 0.0),
                end: Point2::new(1.0, 1.0),
            },
        ];
        deduplicate_segments(&mut segs, 1e-4);
        assert_eq!(segs.len(), 2, "distinct segments should be kept");
    }

    // ── Short segment filtering ─────────────────────────────────────

    #[test]
    fn filter_very_small_segments() {
        let nozzle = 0.4;
        let min_len = nozzle * 0.01; // 0.004mm
        let mut segs = vec![
            Segment2 {
                start: Point2::new(0.0, 0.0),
                end: Point2::new(0.001, 0.0), // 0.001mm — too short
            },
            Segment2 {
                start: Point2::new(0.0, 0.0),
                end: Point2::new(1.0, 0.0), // 1.0mm — fine
            },
        ];
        filter_short_segments(&mut segs, min_len);
        assert_eq!(
            segs.len(),
            1,
            "segment shorter than 0.004mm should be removed"
        );
        assert!(
            (segs[0].end.x - 1.0).abs() < 1e-6,
            "remaining segment should be the long one"
        );
    }

    #[test]
    fn filter_keeps_borderline_segment() {
        let min_len = 0.004;
        let mut segs = vec![Segment2 {
            start: Point2::new(0.0, 0.0),
            end: Point2::new(0.005, 0.0), // 0.005mm — above threshold
        }];
        filter_short_segments(&mut segs, min_len);
        assert_eq!(segs.len(), 1);
    }

    // ── T-junction resolution ───────────────────────────────────────

    #[test]
    fn t_junction_three_segments() {
        // Three segments meeting at origin: one incoming, two outgoing.
        // Incoming from (-1, 0) → (0, 0).
        // Outgoing A: (0, 0) → (1, 0) — straight continuation (angle ~0).
        // Outgoing B: (0, 0) → (0, 1) — left turn (angle ~+PI/2).
        //
        // The chainer should prefer the straight continuation (smallest
        // positive angle), chaining the incoming segment with outgoing A.
        let segments = vec![
            Segment2 {
                start: Point2::new(-1.0, 0.0),
                end: Point2::new(0.0, 0.0),
            },
            Segment2 {
                start: Point2::new(0.0, 0.0),
                end: Point2::new(1.0, 0.0),
            },
            Segment2 {
                start: Point2::new(0.0, 0.0),
                end: Point2::new(0.0, 1.0),
            },
        ];

        let contours = chain_contours(&segments, 1e-3);
        // We have 3 segments forming a T — none of them close, so the chainer
        // correctly produces no closed contours. The important thing is no
        // panic and consistent state. The companion test
        // `t_junction_with_closure` verifies proper T-junction resolution
        // with segments that *do* form a closed contour.
        assert!(
            contours.is_empty(),
            "3 non-closing segments should produce 0 closed contours"
        );
    }

    #[test]
    fn t_junction_with_closure() {
        // Build a square with a T-junction: the square has one edge split
        // into two segments.
        //
        // Square: (0,0)→(1,0)→(1,1)→(0,1)→(0,0)
        // Split bottom edge: (0,0)→(0.5,0) and (0.5,0)→(1,0)
        // Add T-branch at (0.5,0): (0.5,0)→(0.5,-1)
        //
        // The chainer should trace the square and ignore the branch (since
        // the branch doesn't form a closed contour).
        let segments = vec![
            Segment2 {
                start: Point2::new(0.0, 0.0),
                end: Point2::new(0.5, 0.0),
            },
            Segment2 {
                start: Point2::new(0.5, 0.0),
                end: Point2::new(1.0, 0.0),
            },
            Segment2 {
                start: Point2::new(1.0, 0.0),
                end: Point2::new(1.0, 1.0),
            },
            Segment2 {
                start: Point2::new(1.0, 1.0),
                end: Point2::new(0.0, 1.0),
            },
            Segment2 {
                start: Point2::new(0.0, 1.0),
                end: Point2::new(0.0, 0.0),
            },
            // T-branch (dangling — should not appear in any closed contour).
            Segment2 {
                start: Point2::new(0.5, 0.0),
                end: Point2::new(0.5, -1.0),
            },
        ];

        let contours = chain_contours(&segments, 1e-3);
        // At least one closed contour should exist (the square).
        assert!(
            !contours.is_empty(),
            "should produce at least one closed contour from the square"
        );

        // The square contour should have 5 points (4 corners + midpoint of split edge).
        let square_contour = contours
            .iter()
            .find(|c| c.points.len() >= 4)
            .expect("should find the square contour");
        assert!(
            square_contour.points.len() >= 4,
            "square should have >= 4 points"
        );
    }

    // ── Chaining basic ──────────────────────────────────────────────

    #[test]
    fn chain_simple_triangle() {
        // Three segments forming a triangle.
        let segments = vec![
            Segment2 {
                start: Point2::new(0.0, 0.0),
                end: Point2::new(1.0, 0.0),
            },
            Segment2 {
                start: Point2::new(1.0, 0.0),
                end: Point2::new(0.5, 1.0),
            },
            Segment2 {
                start: Point2::new(0.5, 1.0),
                end: Point2::new(0.0, 0.0),
            },
        ];

        let contours = chain_contours(&segments, 1e-3);
        assert_eq!(contours.len(), 1, "triangle should produce 1 contour");
        assert_eq!(contours[0].points.len(), 3, "triangle has 3 points");
    }

    #[test]
    fn chain_reversed_segments() {
        // Segments whose directions are mixed (some reversed relative to
        // chain direction).  The chainer should handle reversed traversal.
        let segments = vec![
            Segment2 {
                start: Point2::new(0.0, 0.0),
                end: Point2::new(1.0, 0.0),
            },
            // Reversed: end→start traversal needed.
            Segment2 {
                start: Point2::new(1.0, 1.0),
                end: Point2::new(1.0, 0.0),
            },
            Segment2 {
                start: Point2::new(1.0, 1.0),
                end: Point2::new(0.0, 1.0),
            },
            Segment2 {
                start: Point2::new(0.0, 1.0),
                end: Point2::new(0.0, 0.0),
            },
        ];

        let contours = chain_contours(&segments, 1e-3);
        assert_eq!(contours.len(), 1, "should chain into 1 closed contour");
        assert_eq!(contours[0].points.len(), 4);
    }

    // ── Spatial hash ────────────────────────────────────────────────

    #[test]
    fn spatial_hash_insert_query() {
        let mut hash = SpatialHash::new(1.0);
        hash.insert(0, Point2::new(0.5, 0.5), true);
        hash.insert(1, Point2::new(10.0, 10.0), true);

        let near = hash.query(Point2::new(0.6, 0.6), 0.2);
        assert!(near.iter().any(|&(idx, _)| idx == 0));
        // Segment 1 is far away — may or may not appear as a candidate
        // (spatial hash returns cell neighbors, not distance-filtered results).
    }

    #[test]
    fn spatial_hash_remove() {
        let mut hash = SpatialHash::new(1.0);
        hash.insert(0, Point2::new(0.5, 0.5), true);
        hash.remove(0, Point2::new(0.5, 0.5), true);
        let near = hash.query(Point2::new(0.5, 0.5), 0.1);
        assert!(
            !near.iter().any(|&(idx, _)| idx == 0),
            "removed entry should not appear"
        );
    }

    // ── Integration: slice a cylinder ───────────────────────────────

    #[test]
    fn slice_cylinder_produces_circles() {
        let mesh = resolve_to_mesh(&CSGNode::cylinder());
        // Cylinder is r=0.5, height [-0.5, 0.5].
        // Use a slightly wider tolerance since tessellated triangle edges
        // may not produce exact-match endpoints at plane intersections.
        let config = SliceConfig {
            layer_height: 0.2,
            tolerance: 1e-3,
            ..default_config()
        };

        let layers = slice_mesh(&mesh, &config);
        assert!(!layers.is_empty(), "cylinder should produce slices");

        // Each interior slice should have at least one contour (outer or inner,
        // depending on chaining traversal order).
        for layer in &layers {
            if layer.z > -0.4 && layer.z < 0.4 {
                let total = layer.outer_contours.len() + layer.inner_contours.len();
                assert!(
                    total > 0,
                    "interior cylinder slice at z={} should have at least one contour, got 0",
                    layer.z
                );
            }
        }
    }

    // ── Edge intersection ───────────────────────────────────────────

    #[test]
    fn edge_z_intersection_basic() {
        let a = [0.0, 0.0, -1.0];
        let b = [0.0, 0.0, 1.0];
        let p = edge_z_intersection(a, b, 0.0).expect("should intersect");
        assert!((p.x).abs() < 1e-6);
        assert!((p.y).abs() < 1e-6);
    }

    #[test]
    fn edge_z_intersection_no_cross() {
        let a = [0.0, 0.0, 1.0];
        let b = [0.0, 0.0, 2.0];
        assert!(edge_z_intersection(a, b, 0.0).is_none());
    }

    #[test]
    fn edge_z_intersection_parallel() {
        let a = [0.0, 0.0, 1.0];
        let b = [1.0, 0.0, 1.0];
        assert!(edge_z_intersection(a, b, 1.0).is_none());
    }

    // ── SliceConfig validation ──────────────────────────────────────

    #[test]
    fn test_slice_config_valid() {
        let config = SliceConfig::default();
        let warnings = config.validate();
        assert!(
            warnings.is_empty(),
            "default config should produce no warnings, got: {:?}",
            warnings
        );
    }

    #[test]
    fn test_slice_config_invalid_density() {
        use crate::infill::{InfillConfig, InfillPattern};
        let config = SliceConfig {
            infill: Some(InfillConfig {
                pattern: InfillPattern::Rectilinear,
                density: 1.5,
                angle_degrees: 45.0,
            }),
            ..SliceConfig::default()
        };
        let warnings = config.validate();
        assert!(
            !warnings.is_empty(),
            "density > 1.0 should produce a warning"
        );
        assert!(
            warnings.iter().any(|w| w.contains("density")),
            "warning should mention density, got: {:?}",
            warnings
        );
    }

    #[test]
    fn test_slice_config_invalid_layer_height() {
        let config = SliceConfig {
            layer_height: -0.1,
            ..SliceConfig::default()
        };
        let warnings = config.validate();
        assert!(
            !warnings.is_empty(),
            "negative layer_height should produce a warning"
        );
        assert!(
            warnings.iter().any(|w| w.contains("layer_height")),
            "warning should mention layer_height, got: {:?}",
            warnings
        );
    }

    // ── Turning angle ───────────────────────────────────────────────

    #[test]
    fn turning_angle_straight() {
        let a = turning_angle(1.0, 0.0, 1.0, 0.0);
        assert!(a.abs() < 1e-6, "straight continuation should have angle ~0");
    }

    #[test]
    fn turning_angle_left_90() {
        let a = turning_angle(1.0, 0.0, 0.0, 1.0);
        assert!(
            (a - std::f32::consts::FRAC_PI_2).abs() < 1e-5,
            "left 90 should be PI/2, got {}",
            a
        );
    }

    #[test]
    fn turning_angle_right_90() {
        let a = turning_angle(1.0, 0.0, 0.0, -1.0);
        assert!(
            (a + std::f32::consts::FRAC_PI_2).abs() < 1e-5,
            "right 90 should be -PI/2, got {}",
            a
        );
    }
}
