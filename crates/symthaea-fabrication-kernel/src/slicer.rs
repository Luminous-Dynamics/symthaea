//! Mesh slicer for 3D printing
//!
//! Converts a [`TriangleMesh`] into a stack of horizontal layers suitable for
//! FDM 3D printing.  Each layer contains perimeter contours and infill line
//! segments.  The slicer computes triangle–plane intersections, chains the
//! resulting edge segments into closed contours, generates rectilinear (or
//! grid) infill, and produces filament-length and time estimates.

use crate::mesh::TriangleMesh;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Infill pattern used between perimeter walls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InfillPattern {
    /// Parallel lines alternating between X and Y each layer.
    Rectilinear,
    /// X + Y lines on every layer (denser than rectilinear at same density).
    Grid,
    /// Hexagonal honeycomb pattern (approximated as offset rectilinear).
    Honeycomb,
    /// Gyroid-inspired pattern (approximated as alternating ±45° lines).
    Gyroid,
}

impl fmt::Display for InfillPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InfillPattern::Rectilinear => write!(f, "rectilinear"),
            InfillPattern::Grid => write!(f, "grid"),
            InfillPattern::Honeycomb => write!(f, "honeycomb"),
            InfillPattern::Gyroid => write!(f, "gyroid"),
        }
    }
}

/// Configuration for the mesh slicer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlicerConfig {
    /// Layer height in mm.
    pub layer_height: f32,
    /// Infill density in \[0.0, 1.0\].  0 = hollow, 1 = solid.
    pub infill_density: f32,
    /// Which infill pattern to use.
    pub infill_pattern: InfillPattern,
    /// Number of perimeter walls.
    pub wall_count: u32,
    /// Number of solid top layers.
    pub top_layers: u32,
    /// Number of solid bottom layers.
    pub bottom_layers: u32,
    /// Nozzle diameter in mm (governs extrusion width).
    pub nozzle_diameter: f32,
}

impl Default for SlicerConfig {
    fn default() -> Self {
        Self {
            layer_height: 0.2,
            infill_density: 0.2,
            infill_pattern: InfillPattern::Rectilinear,
            wall_count: 2,
            top_layers: 3,
            bottom_layers: 3,
            nozzle_diameter: 0.4,
        }
    }
}

impl SlicerConfig {
    /// Validate the configuration, returning an error message if invalid.
    pub fn validate(&self) -> Result<(), String> {
        if self.layer_height <= 0.0 {
            return Err("layer_height must be positive".into());
        }
        if self.layer_height > 10.0 {
            return Err("layer_height exceeds 10 mm maximum".into());
        }
        if !(0.0..=1.0).contains(&self.infill_density) {
            return Err("infill_density must be in [0.0, 1.0]".into());
        }
        if self.nozzle_diameter <= 0.0 {
            return Err("nozzle_diameter must be positive".into());
        }
        if self.wall_count == 0 {
            return Err("wall_count must be at least 1".into());
        }
        Ok(())
    }

    /// Effective extrusion width (typically ~1.2× nozzle diameter).
    pub fn extrusion_width(&self) -> f32 {
        self.nozzle_diameter * 1.2
    }
}

// ---------------------------------------------------------------------------
// Geometry types
// ---------------------------------------------------------------------------

/// A 2D line segment within a single slice layer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LineSegment2D {
    pub start: [f32; 2],
    pub end: [f32; 2],
}

impl LineSegment2D {
    pub fn new(start: [f32; 2], end: [f32; 2]) -> Self {
        Self { start, end }
    }

    /// Euclidean length of the segment.
    pub fn length(&self) -> f32 {
        let dx = self.end[0] - self.start[0];
        let dy = self.end[1] - self.start[1];
        (dx * dx + dy * dy).sqrt()
    }
}

/// A closed contour (polygon) on a 2D slice plane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contour {
    /// Ordered ring of 2D points.  The last point implicitly connects back
    /// to the first.
    pub points: Vec<[f32; 2]>,
    /// `true` if this is the outermost perimeter (printed first for better
    /// surface finish), `false` for inner perimeters.
    pub is_outer: bool,
}

impl Contour {
    /// Signed area via the shoelace formula.
    /// Positive = counter-clockwise, negative = clockwise.
    pub fn signed_area(&self) -> f32 {
        let n = self.points.len();
        if n < 3 {
            return 0.0;
        }
        let mut area = 0.0f32;
        for i in 0..n {
            let j = (i + 1) % n;
            area += self.points[i][0] * self.points[j][1];
            area -= self.points[j][0] * self.points[i][1];
        }
        area * 0.5
    }

    /// Perimeter length.
    pub fn perimeter_length(&self) -> f32 {
        let n = self.points.len();
        if n < 2 {
            return 0.0;
        }
        let mut total = 0.0f32;
        for i in 0..n {
            let j = (i + 1) % n;
            let dx = self.points[j][0] - self.points[i][0];
            let dy = self.points[j][1] - self.points[i][1];
            total += (dx * dx + dy * dy).sqrt();
        }
        total
    }

    /// Axis-aligned bounding box: (min_x, min_y, max_x, max_y).
    pub fn bounding_box(&self) -> (f32, f32, f32, f32) {
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for p in &self.points {
            min_x = min_x.min(p[0]);
            min_y = min_y.min(p[1]);
            max_x = max_x.max(p[0]);
            max_y = max_y.max(p[1]);
        }
        (min_x, min_y, max_x, max_y)
    }

    /// Point-in-polygon test (ray casting).
    pub fn contains_point(&self, x: f32, y: f32) -> bool {
        let n = self.points.len();
        if n < 3 {
            return false;
        }
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let yi = self.points[i][1];
            let yj = self.points[j][1];
            let xi = self.points[i][0];
            let xj = self.points[j][0];
            if ((yi > y) != (yj > y))
                && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
            {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    /// Inset the contour inward by `offset` mm (simple offset, not Minkowski).
    /// Shrinks CCW contours, grows CW contours.
    pub fn offset(&self, offset: f32) -> Self {
        let n = self.points.len();
        if n < 3 {
            return self.clone();
        }
        let sign = if self.signed_area() >= 0.0 { -1.0 } else { 1.0 };
        let actual_offset = sign * offset;

        let mut new_points = Vec::with_capacity(n);
        for i in 0..n {
            let prev = if i == 0 { n - 1 } else { i - 1 };
            let next = (i + 1) % n;

            // Edge normals (pointing inward for CCW contour)
            let dx1 = self.points[i][0] - self.points[prev][0];
            let dy1 = self.points[i][1] - self.points[prev][1];
            let len1 = (dx1 * dx1 + dy1 * dy1).sqrt().max(1e-12);
            let nx1 = -dy1 / len1;
            let ny1 = dx1 / len1;

            let dx2 = self.points[next][0] - self.points[i][0];
            let dy2 = self.points[next][1] - self.points[i][1];
            let len2 = (dx2 * dx2 + dy2 * dy2).sqrt().max(1e-12);
            let nx2 = -dy2 / len2;
            let ny2 = dx2 / len2;

            // Average normal at vertex
            let nx = (nx1 + nx2) * 0.5;
            let ny = (ny1 + ny2) * 0.5;
            let nlen = (nx * nx + ny * ny).sqrt().max(1e-12);

            new_points.push([
                self.points[i][0] + actual_offset * nx / nlen,
                self.points[i][1] + actual_offset * ny / nlen,
            ]);
        }
        Contour {
            points: new_points,
            is_outer: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Sliced layer
// ---------------------------------------------------------------------------

/// One horizontal layer of the sliced model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlicedLayer {
    /// Z height of this layer in mm.
    pub z_height: f32,
    /// Perimeter contours (outermost first).
    pub perimeters: Vec<Contour>,
    /// Infill line segments within the contour region.
    pub infill_lines: Vec<LineSegment2D>,
    /// Whether this layer is printed solid (top/bottom skin).
    pub is_solid: bool,
}

impl SlicedLayer {
    /// Total toolpath length for this layer (perimeters + infill).
    pub fn total_path_length(&self) -> f32 {
        let perim: f32 = self.perimeters.iter().map(|c| c.perimeter_length()).sum();
        let infill: f32 = self.infill_lines.iter().map(|l| l.length()).sum();
        perim + infill
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// The complete result of slicing a mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliceResult {
    /// Ordered layers from bottom to top.
    pub layers: Vec<SlicedLayer>,
    /// Configuration used for slicing.
    pub config: SlicerConfig,
    /// Estimated total filament consumption in mm of 1.75 mm filament.
    pub total_filament_mm: f64,
    /// Estimated print time in seconds.
    pub estimated_time_seconds: f64,
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors that can occur during slicing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SlicerError {
    /// The mesh has no triangles.
    EmptyMesh,
    /// The slicer configuration is invalid.
    InvalidConfig(String),
    /// A geometric intersection computation failed.
    IntersectionError,
}

impl fmt::Display for SlicerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SlicerError::EmptyMesh => write!(f, "mesh contains no triangles"),
            SlicerError::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            SlicerError::IntersectionError => write!(f, "intersection computation failed"),
        }
    }
}

impl std::error::Error for SlicerError {}

// ---------------------------------------------------------------------------
// Bounding box helper
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box in 3D.
#[derive(Debug, Clone, Copy)]
struct BBox3 {
    min: [f32; 3],
    max: [f32; 3],
}

impl BBox3 {
    fn from_vertices(verts: &[[f32; 3]]) -> Option<Self> {
        if verts.is_empty() {
            return None;
        }
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for v in verts {
            for i in 0..3 {
                min[i] = min[i].min(v[i]);
                max[i] = max[i].max(v[i]);
            }
        }
        Some(BBox3 { min, max })
    }

    fn z_range(&self) -> (f32, f32) {
        (self.min[2], self.max[2])
    }
}

// ---------------------------------------------------------------------------
// Triangle–plane intersection
// ---------------------------------------------------------------------------

/// Intersect a triangle (given by three 3D vertices) with a horizontal plane
/// at z = `z_plane`.  Returns 0 or 2 intersection points (edge of cut).
fn intersect_triangle_z(
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
    z_plane: f32,
) -> Option<([f32; 2], [f32; 2])> {
    let edges: [([f32; 3], [f32; 3]); 3] = [(v0, v1), (v1, v2), (v2, v0)];
    let mut pts: Vec<[f32; 2]> = Vec::with_capacity(2);

    for (a, b) in &edges {
        let za = a[2];
        let zb = b[2];

        // Check if the plane z crosses this edge (one endpoint on each side,
        // or exactly on the plane at one end).
        if (za - z_plane) * (zb - z_plane) < 0.0 {
            // Edge crosses the plane
            let t = (z_plane - za) / (zb - za);
            let x = a[0] + t * (b[0] - a[0]);
            let y = a[1] + t * (b[1] - a[1]);
            pts.push([x, y]);
        } else if (za - z_plane).abs() < 1e-7 && (zb - z_plane).abs() >= 1e-7 {
            // Vertex `a` lies exactly on the plane
            pts.push([a[0], a[1]]);
        }
    }

    // De-duplicate very close points
    pts.dedup_by(|a, b| {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        (dx * dx + dy * dy) < 1e-12
    });

    if pts.len() >= 2 {
        Some((pts[0], pts[1]))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Contour chaining
// ---------------------------------------------------------------------------

/// Merge a set of unordered 2D line segments into closed contours by matching
/// endpoints.
fn chain_segments(segments: &[([f32; 2], [f32; 2])]) -> Vec<Contour> {
    if segments.is_empty() {
        return Vec::new();
    }

    // Build an adjacency list keyed by quantized endpoint.
    // We quantize to 0.001 mm resolution to merge nearly-coincident points.
    fn key(p: [f32; 2]) -> (i64, i64) {
        ((p[0] * 1000.0).round() as i64, (p[1] * 1000.0).round() as i64)
    }

    // Map from quantized point to list of (segment_index, which_end: 0=start, 1=end)
    let mut adjacency: HashMap<(i64, i64), Vec<(usize, u8)>> = HashMap::new();
    for (i, (s, e)) in segments.iter().enumerate() {
        adjacency.entry(key(*s)).or_default().push((i, 0));
        adjacency.entry(key(*e)).or_default().push((i, 1));
    }

    let mut used = vec![false; segments.len()];
    let mut contours = Vec::new();

    for start_idx in 0..segments.len() {
        if used[start_idx] {
            continue;
        }
        used[start_idx] = true;

        let mut chain = vec![segments[start_idx].0, segments[start_idx].1];
        let mut current_key = key(*chain.last().unwrap());

        // Walk forward
        loop {
            let neighbors = match adjacency.get(&current_key) {
                Some(n) => n,
                None => break,
            };

            let mut found = false;
            for &(seg_idx, end_flag) in neighbors {
                if used[seg_idx] {
                    continue;
                }
                used[seg_idx] = true;
                // If we matched the start of the segment, walk to its end
                let next_pt = if end_flag == 0 {
                    segments[seg_idx].1
                } else {
                    segments[seg_idx].0
                };
                chain.push(next_pt);
                current_key = key(next_pt);
                found = true;
                break;
            }
            if !found {
                break;
            }
        }

        if chain.len() >= 3 {
            contours.push(Contour {
                points: chain,
                is_outer: true,
            });
        }
    }

    // Mark inner contours (smaller area = inner)
    if contours.len() > 1 {
        let mut max_area = 0.0f32;
        let mut max_idx = 0;
        for (i, c) in contours.iter().enumerate() {
            let area = c.signed_area().abs();
            if area > max_area {
                max_area = area;
                max_idx = i;
            }
        }
        for (i, c) in contours.iter_mut().enumerate() {
            c.is_outer = i == max_idx;
        }
    }

    contours
}

// ---------------------------------------------------------------------------
// Infill generation
// ---------------------------------------------------------------------------

/// Generate rectilinear infill lines inside `contour` at given `density` and
/// `extrusion_width`.  `layer_index` determines line direction (alternating
/// X/Y for rectilinear, both for grid).
fn generate_infill(
    contour: &Contour,
    pattern: InfillPattern,
    density: f32,
    extrusion_width: f32,
    layer_index: usize,
) -> Vec<LineSegment2D> {
    if density <= 0.0 || contour.points.len() < 3 {
        return Vec::new();
    }

    let effective_density = density.min(1.0);
    let spacing = if effective_density >= 1.0 {
        extrusion_width
    } else {
        extrusion_width / effective_density
    };

    let (min_x, min_y, max_x, max_y) = contour.bounding_box();
    let mut lines = Vec::new();

    match pattern {
        InfillPattern::Rectilinear => {
            if layer_index % 2 == 0 {
                generate_x_lines(contour, min_y, max_y, min_x, max_x, spacing, &mut lines);
            } else {
                generate_y_lines(contour, min_x, max_x, min_y, max_y, spacing, &mut lines);
            }
        }
        InfillPattern::Grid => {
            generate_x_lines(contour, min_y, max_y, min_x, max_x, spacing, &mut lines);
            generate_y_lines(contour, min_x, max_x, min_y, max_y, spacing, &mut lines);
        }
        InfillPattern::Honeycomb => {
            // Approximate honeycomb as offset rectilinear rows
            let offset = if layer_index % 2 == 0 {
                0.0
            } else {
                spacing * 0.5
            };
            generate_x_lines_offset(
                contour, min_y, max_y, min_x, max_x, spacing, offset, &mut lines,
            );
        }
        InfillPattern::Gyroid => {
            // Approximate gyroid as alternating ±45° lines
            if layer_index % 2 == 0 {
                generate_diagonal_lines(contour, min_x, min_y, max_x, max_y, spacing, 1.0, &mut lines);
            } else {
                generate_diagonal_lines(contour, min_x, min_y, max_x, max_y, spacing, -1.0, &mut lines);
            }
        }
    }

    lines
}

/// Generate horizontal scan lines (constant Y, sweeping X).
fn generate_x_lines(
    contour: &Contour,
    min_y: f32,
    max_y: f32,
    min_x: f32,
    max_x: f32,
    spacing: f32,
    out: &mut Vec<LineSegment2D>,
) {
    generate_x_lines_offset(contour, min_y, max_y, min_x, max_x, spacing, 0.0, out);
}

/// Horizontal scan lines with a Y-offset (for honeycomb staggering).
fn generate_x_lines_offset(
    contour: &Contour,
    min_y: f32,
    max_y: f32,
    _min_x: f32,
    _max_x: f32,
    spacing: f32,
    y_offset: f32,
    out: &mut Vec<LineSegment2D>,
) {
    let mut y = min_y + spacing * 0.5 + y_offset;
    while y < max_y {
        // Find intersections of the scan line with the contour edges
        let mut intersections = scan_line_intersections(contour, y, true);
        intersections.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Pair up intersections
        let mut i = 0;
        while i + 1 < intersections.len() {
            let x0 = intersections[i];
            let x1 = intersections[i + 1];
            if (x1 - x0).abs() > 1e-6 {
                out.push(LineSegment2D::new([x0, y], [x1, y]));
            }
            i += 2;
        }

        y += spacing;
    }
}

/// Vertical scan lines (constant X, sweeping Y).
fn generate_y_lines(
    contour: &Contour,
    min_x: f32,
    max_x: f32,
    _min_y: f32,
    _max_y: f32,
    spacing: f32,
    out: &mut Vec<LineSegment2D>,
) {
    let mut x = min_x + spacing * 0.5;
    while x < max_x {
        let mut intersections = scan_line_intersections(contour, x, false);
        intersections.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut i = 0;
        while i + 1 < intersections.len() {
            let y0 = intersections[i];
            let y1 = intersections[i + 1];
            if (y1 - y0).abs() > 1e-6 {
                out.push(LineSegment2D::new([x, y0], [x, y1]));
            }
            i += 2;
        }

        x += spacing;
    }
}

/// Generate diagonal lines at ±45° for gyroid approximation.
fn generate_diagonal_lines(
    contour: &Contour,
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    spacing: f32,
    direction: f32, // +1.0 or -1.0
    out: &mut Vec<LineSegment2D>,
) {
    // Diagonal spacing needs sqrt(2) correction
    let diag_spacing = spacing * std::f32::consts::SQRT_2;
    let range = (max_x - min_x) + (max_y - min_y);
    let steps = (range / diag_spacing).ceil() as usize;

    for step in 0..=steps {
        let offset = min_x + min_y * direction + step as f32 * diag_spacing;

        // Line: y = direction * (x - offset) if direction > 0
        //        y = -direction * x + offset if direction < 0
        // Sample intersections with contour
        let mut pts_on_line = Vec::new();

        let n = contour.points.len();
        for i in 0..n {
            let j = (i + 1) % n;
            let p0 = contour.points[i];
            let p1 = contour.points[j];

            // Parameterize contour edge: P(t) = p0 + t*(p1-p0)
            // Intersect with: x + direction*y = offset
            let d0 = p0[0] + direction * p0[1] - offset;
            let d1 = p1[0] + direction * p1[1] - offset;

            if d0 * d1 < 0.0 {
                let t = d0 / (d0 - d1);
                let ix = p0[0] + t * (p1[0] - p0[0]);
                let iy = p0[1] + t * (p1[1] - p0[1]);
                pts_on_line.push([ix, iy]);
            }
        }

        // Sort along the line direction
        pts_on_line.sort_by(|a, b| {
            let ka = a[0] + direction * a[1];
            let kb = b[0] + direction * b[1];
            ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut k = 0;
        while k + 1 < pts_on_line.len() {
            let s = pts_on_line[k];
            let e = pts_on_line[k + 1];
            let dx = e[0] - s[0];
            let dy = e[1] - s[1];
            if (dx * dx + dy * dy) > 1e-10 {
                out.push(LineSegment2D::new(s, e));
            }
            k += 2;
        }
    }
}

/// Find where a horizontal (or vertical) scan line intersects a contour.
/// If `horizontal` is true, the scan line is at y = `coord` and we return
/// the X coordinates of intersections; otherwise the scan line is at
/// x = `coord` and we return Y coordinates.
fn scan_line_intersections(contour: &Contour, coord: f32, horizontal: bool) -> Vec<f32> {
    let n = contour.points.len();
    let mut result = Vec::new();

    for i in 0..n {
        let j = (i + 1) % n;
        let (a_par, a_perp, b_par, b_perp) = if horizontal {
            (
                contour.points[i][0],
                contour.points[i][1],
                contour.points[j][0],
                contour.points[j][1],
            )
        } else {
            (
                contour.points[i][1],
                contour.points[i][0],
                contour.points[j][1],
                contour.points[j][0],
            )
        };

        if (a_perp - coord) * (b_perp - coord) < 0.0 {
            let t = (coord - a_perp) / (b_perp - a_perp);
            result.push(a_par + t * (b_par - a_par));
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Slice a triangle mesh into horizontal layers for 3D printing.
///
/// # Errors
///
/// Returns [`SlicerError::EmptyMesh`] if the mesh has no triangles, or
/// [`SlicerError::InvalidConfig`] if the configuration is invalid.
pub fn slice_mesh(
    mesh: &TriangleMesh,
    config: &SlicerConfig,
) -> Result<SliceResult, SlicerError> {
    // Validate inputs
    if mesh.indices.is_empty() || mesh.vertices.is_empty() {
        return Err(SlicerError::EmptyMesh);
    }
    config
        .validate()
        .map_err(SlicerError::InvalidConfig)?;

    // Compute bounding box
    let bbox = BBox3::from_vertices(&mesh.vertices).ok_or(SlicerError::EmptyMesh)?;
    let (z_min, z_max) = bbox.z_range();

    // Compute layer z-heights (first layer at z_min + layer_height/2)
    let mut z_heights = Vec::new();
    let mut z = z_min + config.layer_height * 0.5;
    while z < z_max {
        z_heights.push(z);
        z += config.layer_height;
    }

    let total_layers = z_heights.len();

    // Slice each layer
    let mut layers = Vec::with_capacity(total_layers);

    for (layer_idx, &z_plane) in z_heights.iter().enumerate() {
        // Determine if this is a solid layer (top/bottom)
        let is_solid = layer_idx < config.bottom_layers as usize
            || layer_idx >= total_layers.saturating_sub(config.top_layers as usize);

        // Intersect all triangles with this z-plane
        let mut segments: Vec<([f32; 2], [f32; 2])> = Vec::new();
        for tri in &mesh.indices {
            let v0 = mesh.vertices[tri[0] as usize];
            let v1 = mesh.vertices[tri[1] as usize];
            let v2 = mesh.vertices[tri[2] as usize];

            if let Some((p0, p1)) = intersect_triangle_z(v0, v1, v2, z_plane) {
                segments.push((p0, p1));
            }
        }

        // Chain segments into contours
        let outer_contours = chain_segments(&segments);

        // Generate perimeters (wall_count insets of the outer contour)
        let mut perimeters = Vec::new();
        for contour in &outer_contours {
            // First perimeter is the original contour (outer wall)
            let mut outer = contour.clone();
            outer.is_outer = true;
            perimeters.push(outer);

            // Inner perimeters
            let mut current = contour.clone();
            for wall in 1..config.wall_count {
                let inset = current.offset(config.extrusion_width());
                let mut inner = inset;
                inner.is_outer = false;
                perimeters.push(inner.clone());
                current = inner;
                let _ = wall; // suppress unused warning
            }
        }

        // Generate infill
        let effective_density = if is_solid { 1.0 } else { config.infill_density };
        let effective_pattern = if is_solid {
            InfillPattern::Rectilinear
        } else {
            config.infill_pattern
        };

        let mut infill_lines = Vec::new();
        // Use the innermost perimeter (or outer contour) as the infill boundary
        let infill_boundary = if config.wall_count > 1 && !perimeters.is_empty() {
            // Find the last (innermost) contour for each outer contour
            let walls_per_contour = config.wall_count as usize;
            let mut boundaries = Vec::new();
            for chunk_start in (0..perimeters.len()).step_by(walls_per_contour) {
                let inner_idx = (chunk_start + walls_per_contour - 1).min(perimeters.len() - 1);
                boundaries.push(&perimeters[inner_idx]);
            }
            boundaries
        } else {
            outer_contours.iter().collect()
        };

        for boundary in &infill_boundary {
            let mut fill = generate_infill(
                boundary,
                effective_pattern,
                effective_density,
                config.extrusion_width(),
                layer_idx,
            );
            infill_lines.append(&mut fill);
        }

        layers.push(SlicedLayer {
            z_height: z_plane,
            perimeters,
            infill_lines,
            is_solid,
        });
    }

    // Estimate filament usage and time
    let extrusion_width = config.extrusion_width() as f64;
    let layer_height = config.layer_height as f64;
    let filament_diameter = 1.75_f64; // Standard FDM filament
    let filament_cross_section =
        std::f64::consts::PI * (filament_diameter / 2.0) * (filament_diameter / 2.0);
    let bead_cross_section = extrusion_width * layer_height;

    let mut total_path_mm = 0.0_f64;
    let mut total_travel_mm = 0.0_f64;

    for (i, layer) in layers.iter().enumerate() {
        total_path_mm += layer.total_path_length() as f64;
        // Estimate travel moves as ~20% of path length
        total_travel_mm += layer.total_path_length() as f64 * 0.2;
        // Add Z travel
        if i > 0 {
            total_travel_mm += config.layer_height as f64;
        }
    }

    let total_filament_mm = total_path_mm * bead_cross_section / filament_cross_section;

    // Time estimate: 30 mm/s print, 120 mm/s travel (rough defaults)
    let print_speed = 30.0_f64;
    let travel_speed = 120.0_f64;
    let estimated_time_seconds = total_path_mm / print_speed + total_travel_mm / travel_speed;

    Ok(SliceResult {
        layers,
        config: config.clone(),
        total_filament_mm,
        estimated_time_seconds,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::TriangleMesh;

    /// Build a unit cube mesh centered at origin, side length = `size`.
    fn make_cube_mesh(size: f32) -> TriangleMesh {
        let h = size / 2.0;
        let vertices = vec![
            // Front face
            [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
            // Back face
            [h, -h, -h], [-h, -h, -h], [-h, h, -h], [h, h, -h],
            // Top face
            [-h, h, h], [h, h, h], [h, h, -h], [-h, h, -h],
            // Bottom face
            [-h, -h, -h], [h, -h, -h], [h, -h, h], [-h, -h, h],
            // Right face
            [h, -h, h], [h, -h, -h], [h, h, -h], [h, h, h],
            // Left face
            [-h, -h, -h], [-h, -h, h], [-h, h, h], [-h, h, -h],
        ];
        let normals = vec![
            [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
        ];
        let indices = vec![
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [8, 9, 10], [8, 10, 11],
            [12, 13, 14], [12, 14, 15],
            [16, 17, 18], [16, 18, 19],
            [20, 21, 22], [20, 22, 23],
        ];
        TriangleMesh { vertices, normals, indices }
    }

    /// Build a single upward-facing triangle at z = 0..1.
    fn make_single_triangle() -> TriangleMesh {
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [5.0, 0.0, 10.0],
        ];
        let normals = vec![
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let indices = vec![[0, 1, 2]];
        TriangleMesh { vertices, normals, indices }
    }

    #[test]
    fn test_empty_mesh_returns_error() {
        let mesh = TriangleMesh::empty();
        let config = SlicerConfig::default();
        let result = slice_mesh(&mesh, &config);
        assert!(matches!(result, Err(SlicerError::EmptyMesh)));
    }

    #[test]
    fn test_invalid_config_layer_height() {
        let mesh = make_cube_mesh(10.0);
        let mut config = SlicerConfig::default();
        config.layer_height = -0.1;
        let result = slice_mesh(&mesh, &config);
        assert!(matches!(result, Err(SlicerError::InvalidConfig(_))));
    }

    #[test]
    fn test_invalid_config_infill_density() {
        let mesh = make_cube_mesh(10.0);
        let mut config = SlicerConfig::default();
        config.infill_density = 1.5;
        let result = slice_mesh(&mesh, &config);
        assert!(matches!(result, Err(SlicerError::InvalidConfig(_))));
    }

    #[test]
    fn test_invalid_config_zero_walls() {
        let mesh = make_cube_mesh(10.0);
        let mut config = SlicerConfig::default();
        config.wall_count = 0;
        let result = slice_mesh(&mesh, &config);
        assert!(matches!(result, Err(SlicerError::InvalidConfig(_))));
    }

    #[test]
    fn test_single_triangle_slices() {
        let mesh = make_single_triangle();
        let config = SlicerConfig {
            layer_height: 1.0,
            ..SlicerConfig::default()
        };
        let result = slice_mesh(&mesh, &config).unwrap();
        // Triangle spans z=0..10, so at 1mm layers we expect ~10 layers
        assert!(!result.layers.is_empty());
        assert!(result.layers.len() >= 8);
        assert!(result.layers.len() <= 10);
    }

    #[test]
    fn test_cube_layer_count() {
        let mesh = make_cube_mesh(10.0);
        let config = SlicerConfig {
            layer_height: 1.0,
            infill_density: 0.2,
            ..SlicerConfig::default()
        };
        let result = slice_mesh(&mesh, &config).unwrap();
        // Cube from z=-5 to z=5, 10mm total → ~10 layers at 1mm
        let expected_layers = 10;
        let actual = result.layers.len();
        assert!(
            (actual as i32 - expected_layers as i32).unsigned_abs() <= 1,
            "expected ~{expected_layers} layers, got {actual}"
        );
    }

    #[test]
    fn test_cube_has_contours() {
        let mesh = make_cube_mesh(10.0);
        let config = SlicerConfig {
            layer_height: 1.0,
            infill_density: 0.2,
            ..SlicerConfig::default()
        };
        let result = slice_mesh(&mesh, &config).unwrap();
        // Middle layers should have contours
        let mid = result.layers.len() / 2;
        assert!(
            !result.layers[mid].perimeters.is_empty(),
            "middle layer should have perimeters"
        );
    }

    #[test]
    fn test_infill_density_affects_line_count() {
        let mesh = make_cube_mesh(20.0);

        let config_low = SlicerConfig {
            layer_height: 2.0,
            infill_density: 0.1,
            wall_count: 1,
            top_layers: 0,
            bottom_layers: 0,
            ..SlicerConfig::default()
        };
        let config_high = SlicerConfig {
            layer_height: 2.0,
            infill_density: 0.8,
            wall_count: 1,
            top_layers: 0,
            bottom_layers: 0,
            ..SlicerConfig::default()
        };

        let result_low = slice_mesh(&mesh, &config_low).unwrap();
        let result_high = slice_mesh(&mesh, &config_high).unwrap();

        // Sum infill lines across all layers
        let low_infill: usize = result_low
            .layers
            .iter()
            .map(|l| l.infill_lines.len())
            .sum();
        let high_infill: usize = result_high
            .layers
            .iter()
            .map(|l| l.infill_lines.len())
            .sum();

        assert!(
            high_infill > low_infill,
            "higher density ({high_infill}) should produce more infill than lower ({low_infill})"
        );
    }

    #[test]
    fn test_solid_layers_bottom() {
        let mesh = make_cube_mesh(10.0);
        let config = SlicerConfig {
            layer_height: 1.0,
            infill_density: 0.2,
            bottom_layers: 3,
            top_layers: 0,
            ..SlicerConfig::default()
        };
        let result = slice_mesh(&mesh, &config).unwrap();
        // First 3 layers should be solid
        for i in 0..3.min(result.layers.len()) {
            assert!(result.layers[i].is_solid, "layer {i} should be solid (bottom)");
        }
        // A middle layer should NOT be solid (if enough layers exist)
        if result.layers.len() > 6 {
            let mid = result.layers.len() / 2;
            assert!(!result.layers[mid].is_solid, "middle layer should not be solid");
        }
    }

    #[test]
    fn test_solid_layers_top() {
        let mesh = make_cube_mesh(10.0);
        let config = SlicerConfig {
            layer_height: 1.0,
            infill_density: 0.2,
            bottom_layers: 0,
            top_layers: 3,
            ..SlicerConfig::default()
        };
        let result = slice_mesh(&mesh, &config).unwrap();
        let n = result.layers.len();
        // Last 3 layers should be solid
        for i in n.saturating_sub(3)..n {
            assert!(result.layers[i].is_solid, "layer {i} should be solid (top)");
        }
    }

    #[test]
    fn test_filament_and_time_positive() {
        let mesh = make_cube_mesh(10.0);
        let config = SlicerConfig::default();
        let result = slice_mesh(&mesh, &config).unwrap();
        assert!(
            result.total_filament_mm > 0.0,
            "filament estimate should be positive"
        );
        assert!(
            result.estimated_time_seconds > 0.0,
            "time estimate should be positive"
        );
    }

    #[test]
    fn test_z_heights_monotonically_increasing() {
        let mesh = make_cube_mesh(10.0);
        let config = SlicerConfig {
            layer_height: 0.5,
            ..SlicerConfig::default()
        };
        let result = slice_mesh(&mesh, &config).unwrap();
        for i in 1..result.layers.len() {
            assert!(
                result.layers[i].z_height > result.layers[i - 1].z_height,
                "layer z-heights must be monotonically increasing"
            );
        }
    }

    #[test]
    fn test_grid_pattern_more_lines_than_rectilinear() {
        let mesh = make_cube_mesh(20.0);
        let config_rect = SlicerConfig {
            layer_height: 2.0,
            infill_density: 0.3,
            infill_pattern: InfillPattern::Rectilinear,
            wall_count: 1,
            top_layers: 0,
            bottom_layers: 0,
            ..SlicerConfig::default()
        };
        let config_grid = SlicerConfig {
            layer_height: 2.0,
            infill_density: 0.3,
            infill_pattern: InfillPattern::Grid,
            wall_count: 1,
            top_layers: 0,
            bottom_layers: 0,
            ..SlicerConfig::default()
        };

        let result_rect = slice_mesh(&mesh, &config_rect).unwrap();
        let result_grid = slice_mesh(&mesh, &config_grid).unwrap();

        let rect_infill: usize = result_rect.layers.iter().map(|l| l.infill_lines.len()).sum();
        let grid_infill: usize = result_grid.layers.iter().map(|l| l.infill_lines.len()).sum();

        assert!(
            grid_infill > rect_infill,
            "grid ({grid_infill}) should produce more infill lines than rectilinear ({rect_infill})"
        );
    }

    #[test]
    fn test_line_segment_length() {
        let seg = LineSegment2D::new([0.0, 0.0], [3.0, 4.0]);
        assert!((seg.length() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_contour_signed_area_unit_square() {
        let contour = Contour {
            points: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            is_outer: true,
        };
        let area = contour.signed_area();
        assert!((area.abs() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_contour_contains_point() {
        let contour = Contour {
            points: vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            is_outer: true,
        };
        assert!(contour.contains_point(5.0, 5.0));
        assert!(!contour.contains_point(15.0, 5.0));
        assert!(!contour.contains_point(-1.0, -1.0));
    }

    #[test]
    fn test_triangle_z_intersection() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [10.0, 0.0, 0.0];
        let v2 = [5.0, 0.0, 10.0];

        // Plane at z=5 should intersect
        let result = intersect_triangle_z(v0, v1, v2, 5.0);
        assert!(result.is_some(), "should intersect at z=5");

        // Plane at z=11 should NOT intersect
        let result = intersect_triangle_z(v0, v1, v2, 11.0);
        assert!(result.is_none(), "should not intersect above triangle");
    }

    #[test]
    fn test_config_default_validates() {
        assert!(SlicerConfig::default().validate().is_ok());
    }

    #[test]
    fn test_slicer_error_display() {
        let e = SlicerError::EmptyMesh;
        assert_eq!(format!("{e}"), "mesh contains no triangles");

        let e = SlicerError::InvalidConfig("bad".into());
        assert!(format!("{e}").contains("bad"));
    }
}
