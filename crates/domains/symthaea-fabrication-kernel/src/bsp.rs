// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Binary-space-partitioning mesh booleans.
//!
//! This implementation follows the classic solid CSG sequence: polygons are
//! classified against oriented planes, back leaves represent solid space during
//! clipping, and union/subtraction/intersection are composed from clip, invert,
//! and rebuild operations. It is suitable for closed, consistently oriented
//! triangle solids; callers should validate inputs and outputs.

use crate::mesh::TriangleMesh;

const EPSILON: f64 = 1.0e-5;

// Triangle/plane math below is done entirely in f64, even though the public
// TriangleMesh (and the mesh's own coordinates) are f32. This is not a
// tolerance change: a single boolean op recurses a candidate triangle
// through every BSP node along its path, and each "spanning" classification
// re-interpolates an intersection point from vertices that may themselves
// already be the result of an earlier interpolation. For a smooth/convex
// operand (sphere, cylinder, cone) the BSP tree built from its own facets is
// inherently a near-linear chain (every facet's plane is close to a
// supporting plane of the whole convex surface, so almost every other facet
// falls on one side), so a coarser operand's larger triangles can be
// re-split dozens of levels deep. At f32 precision the compounding rounding
// error from that many chained interpolations was landing above the mesh
// validator's edge-matching quantization grid, so two fragments that were
// meant to meet at the exact same edge point ended up at measurably
// different positions -- producing real (not just quantization-artifact)
// boundary gaps and self-intersecting slivers, not merely "extra" triangles.
// f64 keeps the accumulated error many orders of magnitude below that grid.

#[derive(Debug, Clone)]
struct Triangle {
    v: [[f64; 3]; 3],
    normal: [f64; 3],
}

impl Triangle {
    fn flip(&mut self) {
        self.v.swap(0, 2);
        self.normal = [-self.normal[0], -self.normal[1], -self.normal[2]];
    }
}

#[derive(Debug, Clone)]
struct Plane {
    normal: [f64; 3],
    d: f64,
}

impl Plane {
    fn from_triangle(triangle: &Triangle) -> Self {
        let normal = triangle.normal;
        Self {
            normal,
            d: -(normal[0] * triangle.v[0][0]
                + normal[1] * triangle.v[0][1]
                + normal[2] * triangle.v[0][2]),
        }
    }

    fn distance(&self, point: &[f64; 3]) -> f64 {
        self.normal[0] * point[0] + self.normal[1] * point[1] + self.normal[2] * point[2] + self.d
    }

    fn flip(&mut self) {
        self.normal = [-self.normal[0], -self.normal[1], -self.normal[2]];
        self.d = -self.d;
    }

    fn split_triangle(
        &self,
        triangle: &Triangle,
        coplanar_front: &mut Vec<Triangle>,
        coplanar_back: &mut Vec<Triangle>,
        front: &mut Vec<Triangle>,
        back: &mut Vec<Triangle>,
    ) {
        const COPLANAR: u8 = 0;
        const FRONT: u8 = 1;
        const BACK: u8 = 2;
        const SPANNING: u8 = FRONT | BACK;

        let mut polygon_type = COPLANAR;
        let mut vertex_types = [COPLANAR; 3];
        for (index, vertex) in triangle.v.iter().enumerate() {
            let distance = self.distance(vertex);
            let vertex_type = if distance < -EPSILON {
                BACK
            } else if distance > EPSILON {
                FRONT
            } else {
                COPLANAR
            };
            polygon_type |= vertex_type;
            vertex_types[index] = vertex_type;
        }

        match polygon_type {
            COPLANAR => {
                if dot(self.normal, triangle.normal) >= 0.0 {
                    coplanar_front.push(triangle.clone());
                } else {
                    coplanar_back.push(triangle.clone());
                }
            }
            FRONT => front.push(triangle.clone()),
            BACK => back.push(triangle.clone()),
            SPANNING => {
                let mut front_vertices = Vec::with_capacity(4);
                let mut back_vertices = Vec::with_capacity(4);

                for index in 0..3 {
                    let next = (index + 1) % 3;
                    let current_type = vertex_types[index];
                    let next_type = vertex_types[next];
                    let current = triangle.v[index];
                    let next_vertex = triangle.v[next];

                    if current_type != BACK {
                        front_vertices.push(current);
                    }
                    if current_type != FRONT {
                        back_vertices.push(current);
                    }

                    if (current_type | next_type) == SPANNING {
                        let direction = [
                            next_vertex[0] - current[0],
                            next_vertex[1] - current[1],
                            next_vertex[2] - current[2],
                        ];
                        let denominator = dot(self.normal, direction);
                        if denominator.abs() > 1.0e-12 {
                            let t = -self.distance(&current) / denominator;
                            let intersection = [
                                current[0] + direction[0] * t,
                                current[1] + direction[1] * t,
                                current[2] + direction[2] * t,
                            ];
                            front_vertices.push(intersection);
                            back_vertices.push(intersection);
                        }
                    }
                }

                front.extend(fan_triangulate(&front_vertices));
                back.extend(fan_triangulate(&back_vertices));
            }
            _ => unreachable!("triangle classification uses only two side bits"),
        }
    }
}

#[derive(Debug, Clone, Default)]
struct BSPNode {
    plane: Option<Plane>,
    front: Option<Box<BSPNode>>,
    back: Option<Box<BSPNode>>,
    coplanar: Vec<Triangle>,
}

impl BSPNode {
    fn from_triangles(triangles: Vec<Triangle>) -> Self {
        let mut node = Self::default();
        node.build(triangles);
        node
    }

    fn build(&mut self, triangles: Vec<Triangle>) {
        if triangles.is_empty() {
            return;
        }

        if self.plane.is_none() {
            self.plane = Some(choose_split_plane(&triangles));
        }
        let plane = self.plane.as_ref().expect("plane initialized").clone();
        let mut coplanar_front = Vec::new();
        let mut coplanar_back = Vec::new();
        let mut front = Vec::new();
        let mut back = Vec::new();

        for triangle in triangles {
            plane.split_triangle(
                &triangle,
                &mut coplanar_front,
                &mut coplanar_back,
                &mut front,
                &mut back,
            );
        }
        self.coplanar.extend(coplanar_front);
        self.coplanar.extend(coplanar_back);

        if !front.is_empty() {
            self.front
                .get_or_insert_with(|| Box::new(Self::default()))
                .build(front);
        }
        if !back.is_empty() {
            self.back
                .get_or_insert_with(|| Box::new(Self::default()))
                .build(back);
        }
    }

    /// Axis-aligned bounding box of every triangle actually stored anywhere
    /// in this subtree (coplanar here, plus front's and back's subtrees).
    ///
    /// Used by [`Self::clip_triangles`] as a locality guard: a splitting
    /// plane is only meaningful in the vicinity of the real geometry it was
    /// derived from. Testing a candidate against it as an *infinite* plane
    /// is mathematically well-defined but produces a false "spanning"
    /// classification whenever the candidate happens to be far from any
    /// actual triangle in this subtree yet still straddles the plane's
    /// unbounded extension (see `clip_triangles` doc comment for the
    /// concrete failure mode this prevents).
    fn subtree_bbox(&self) -> Option<([f64; 3], [f64; 3])> {
        let mut bbox: Option<([f64; 3], [f64; 3])> = None;
        let mut extend = |point: [f64; 3]| {
            bbox = Some(match bbox {
                None => (point, point),
                Some((min, max)) => (
                    [
                        min[0].min(point[0]),
                        min[1].min(point[1]),
                        min[2].min(point[2]),
                    ],
                    [
                        max[0].max(point[0]),
                        max[1].max(point[1]),
                        max[2].max(point[2]),
                    ],
                ),
            });
        };
        for triangle in &self.coplanar {
            for vertex in &triangle.v {
                extend(*vertex);
            }
        }
        let child_bbox =
            |child: &Option<Box<BSPNode>>| child.as_ref().and_then(|n| n.subtree_bbox());
        if let Some((min, max)) = child_bbox(&self.front) {
            extend(min);
            extend(max);
        }
        if let Some((min, max)) = child_bbox(&self.back) {
            extend(min);
            extend(max);
        }
        bbox
    }

    /// Classic BSP polygon clipping, guarded by a bounding-box locality
    /// check to avoid a known pathological failure mode.
    ///
    /// Splitting planes here come from individual mesh facets; for a
    /// smooth/convex operand (a tessellated sphere, cylinder, cone) almost
    /// every facet's plane is close to a supporting plane of the whole
    /// surface, so the tree built from it is a near-linear chain, not a
    /// balanced tree. Clipping a coarser operand's much larger triangles
    /// through such a chain re-tests them against *every* facet's plane
    /// along the path, including facets nowhere near where the candidate
    /// actually is. Because a plane is unbounded, a distant/irrelevant facet
    /// can still classify a spatially-remote candidate as "spanning" (its
    /// infinite extension crosses the candidate) even though no real
    /// surface exists anywhere near that candidate. Each such false split
    /// computes a boundary point from that irrelevant facet's own equation;
    /// different irrelevant facets produce different, mutually
    /// inconsistent "boundaries" for what should be untouched geometry —
    /// this is what turned a 12-triangle cube face into tens of thousands
    /// of fragments with genuine self-intersections and non-manifold edges
    /// when unioned with a tessellated sphere, independent of floating
    /// point precision (confirmed: switching this module's internal math
    /// from f32 to f64 did not remove the defects).
    ///
    /// The fix: before classifying a candidate against this node's plane,
    /// check whether the candidate's own bounding box overlaps this
    /// subtree's aggregate bounding box at all. If it does not, no real
    /// geometry anywhere under this node can be near the candidate, so it
    /// is definitively outside whatever solid this subtree helps bound —
    /// route it through as a front (surviving) triangle without ever
    /// computing a spurious split against an irrelevant plane.
    fn clip_triangles(&self, triangles: Vec<Triangle>) -> Vec<Triangle> {
        let Some(plane) = &self.plane else {
            return triangles;
        };

        let bbox = self.subtree_bbox();

        let mut coplanar_front = Vec::new();
        let mut coplanar_back = Vec::new();
        let mut front = Vec::new();
        let mut back = Vec::new();
        for triangle in triangles {
            if let Some((bbox_min, bbox_max)) = bbox {
                if !triangle_bbox_overlaps(&triangle, bbox_min, bbox_max, EPSILON) {
                    // Route the whole triangle by majority vertex sign
                    // relative to this plane, instead of ever computing a
                    // split. Do NOT default unconditionally to "front" here:
                    // `invert()` swaps front/back polarity (it flips the
                    // plane and swaps the front/back subtrees), and a
                    // subtree representing an inverted (complemented, thus
                    // spatially unbounded) solid has its finite stored
                    // geometry's bounding box on the *opposite* side of
                    // "inside" from a non-inverted one. Using the plane's
                    // actual sign keeps this correct in both cases while
                    // still avoiding the spurious fragmentation a full
                    // `split_triangle` call would otherwise perform against
                    // an irrelevant, spatially-distant plane.
                    let mut front_votes = 0u8;
                    let mut back_votes = 0u8;
                    for vertex in &triangle.v {
                        let distance = plane.distance(vertex);
                        if distance > 0.0 {
                            front_votes += 1;
                        } else if distance < 0.0 {
                            back_votes += 1;
                        }
                    }
                    if back_votes > front_votes {
                        back.push(triangle);
                    } else {
                        front.push(triangle);
                    }
                    continue;
                }
            }
            plane.split_triangle(
                &triangle,
                &mut coplanar_front,
                &mut coplanar_back,
                &mut front,
                &mut back,
            );
        }
        front.extend(coplanar_front);
        back.extend(coplanar_back);

        if let Some(front_node) = &self.front {
            front = front_node.clip_triangles(front);
        }
        if let Some(back_node) = &self.back {
            back = back_node.clip_triangles(back);
        } else {
            // Back space is inside this closed solid and is removed by clipping.
            back.clear();
        }

        front.extend(back);
        front
    }

    fn clip_to(&mut self, other: &BSPNode) {
        self.coplanar = other.clip_triangles(std::mem::take(&mut self.coplanar));
        if let Some(front) = &mut self.front {
            front.clip_to(other);
        }
        if let Some(back) = &mut self.back {
            back.clip_to(other);
        }
    }

    fn invert(&mut self) {
        for triangle in &mut self.coplanar {
            triangle.flip();
        }
        if let Some(plane) = &mut self.plane {
            plane.flip();
        }
        if let Some(front) = &mut self.front {
            front.invert();
        }
        if let Some(back) = &mut self.back {
            back.invert();
        }
        std::mem::swap(&mut self.front, &mut self.back);
    }

    fn all_triangles(&self) -> Vec<Triangle> {
        let mut triangles = self.coplanar.clone();
        if let Some(front) = &self.front {
            triangles.extend(front.all_triangles());
        }
        if let Some(back) = &self.back {
            triangles.extend(back.all_triangles());
        }
        triangles
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Does `triangle`'s bounding box overlap `[bbox_min, bbox_max]`, padded by
/// `epsilon` on every axis so genuinely touching/coplanar geometry is never
/// spuriously excluded?
fn triangle_bbox_overlaps(
    triangle: &Triangle,
    bbox_min: [f64; 3],
    bbox_max: [f64; 3],
    epsilon: f64,
) -> bool {
    let mut tri_min = triangle.v[0];
    let mut tri_max = triangle.v[0];
    for vertex in &triangle.v[1..] {
        for axis in 0..3 {
            tri_min[axis] = tri_min[axis].min(vertex[axis]);
            tri_max[axis] = tri_max[axis].max(vertex[axis]);
        }
    }
    (0..3).all(|axis| {
        tri_min[axis] <= bbox_max[axis] + epsilon && bbox_min[axis] <= tri_max[axis] + epsilon
    })
}

/// Choose a well-balanced splitting plane from a candidate triangle set.
///
/// Unconditionally using the first remaining triangle's plane (the naive
/// choice) degenerates into a near-linear BSP chain for smooth, convex
/// surfaces such as tessellated spheres/cylinders/cones: an arbitrary local
/// facet's plane typically has almost every *other* facet strictly on one
/// side, so each `build()` level only ever peels off a handful of triangles
/// before recursing into an equally large remainder. For an n-triangle mesh
/// this produces O(n) tree depth instead of the O(log n) a balanced BSP
/// achieves. Clipping a second (coarser) operand's larger triangles through
/// such a chain re-splits them at nearly every level — each spanning split
/// re-interpolates already-once-split vertices against yet another plane —
/// causing both an exponential fragment-count blowup and compounding f32
/// precision loss that eventually yields genuinely self-intersecting /
/// non-manifold output, not just an inefficient-but-correct decomposition.
///
/// This samples a bounded number of evenly-spaced candidate triangles and
/// picks the one whose plane minimizes spanning-triangle count (primary,
/// classic BSP-compiler heuristic) with front/back imbalance as a
/// tiebreaker, keeping the tree close to balanced without requiring a full
/// O(n^2) scan for larger meshes.
///
/// For a strictly convex mesh (sphere, cylinder, cone) this facet-plane
/// heuristic cannot actually help: by definition of convexity, *every*
/// facet's plane is a near-supporting plane of the whole surface, so
/// virtually all other facets fall on the same (back) side no matter which
/// one is picked — every candidate ties. When that tie is detected (the
/// best candidate still sends the large majority of triangles to one side),
/// fall back to a synthetic spatial median-cut plane (a k-d-tree-style
/// bisection of the triangle centroids along their bounding box's longest
/// axis). This is not derived from any mesh facet, but it guarantees
/// O(log n) tree depth regardless of input convexity, which is what
/// actually fixes the pathological chain-tree depth (confirmed via direct
/// measurement: the facet-only heuristic left tree depth unchanged for a
/// tessellated sphere, exactly matching the naive "always pick the first
/// triangle" baseline).
fn choose_split_plane(triangles: &[Triangle]) -> Plane {
    const MAX_CANDIDATES: usize = 32;
    let step = (triangles.len() / MAX_CANDIDATES).max(1);

    let mut best_index = 0;
    let mut best_cost = f64::INFINITY;
    let mut best_counts = (0usize, 0usize, 0usize);

    for candidate_index in (0..triangles.len()).step_by(step) {
        let plane = Plane::from_triangle(&triangles[candidate_index]);
        let counts = classify_counts(&plane, triangles);
        let (front_count, back_count, spanning_count) = counts;

        let balance = (front_count as f64 - back_count as f64).abs();
        // Spanning triangles are weighted heavily: they are what actually
        // grows the tree/fragment count, balance only matters as a
        // tiebreaker between equally-splitting candidates.
        let cost = spanning_count as f64 * 8.0 + balance;
        if cost < best_cost {
            best_cost = cost;
            best_index = candidate_index;
            best_counts = counts;
        }
    }

    let (front_count, back_count, _spanning_count) = best_counts;
    let majority = front_count.max(back_count) as f64;
    let total = triangles.len() as f64;
    if triangles.len() > 8 && majority / total > 0.9 {
        return median_split_plane(triangles);
    }

    Plane::from_triangle(&triangles[best_index])
}

/// Count how many triangles fall strictly in front, strictly behind, and
/// spanning `plane` (per-vertex classification, matching
/// [`Plane::split_triangle`]'s own thresholding).
fn classify_counts(plane: &Plane, triangles: &[Triangle]) -> (usize, usize, usize) {
    let mut front_count = 0usize;
    let mut back_count = 0usize;
    let mut spanning_count = 0usize;

    for triangle in triangles {
        let mut has_front = false;
        let mut has_back = false;
        for vertex in &triangle.v {
            let distance = plane.distance(vertex);
            if distance > EPSILON {
                has_front = true;
            } else if distance < -EPSILON {
                has_back = true;
            }
        }
        match (has_front, has_back) {
            (true, true) => spanning_count += 1,
            (true, false) => front_count += 1,
            (false, true) => back_count += 1,
            (false, false) => {}
        }
    }

    (front_count, back_count, spanning_count)
}

/// Synthetic (non-facet) splitting plane: bisects the triangle centroids
/// through their median along the longest axis of their bounding box.
///
/// Standard k-d-tree "median cut" technique. Used only as a fallback (see
/// [`choose_split_plane`]) when no facet plane can meaningfully balance the
/// set, which is precisely the case for smooth/convex operands.
fn median_split_plane(triangles: &[Triangle]) -> Plane {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    let mut centroids = Vec::with_capacity(triangles.len());
    for triangle in triangles {
        let centroid = [
            (triangle.v[0][0] + triangle.v[1][0] + triangle.v[2][0]) / 3.0,
            (triangle.v[0][1] + triangle.v[1][1] + triangle.v[2][1]) / 3.0,
            (triangle.v[0][2] + triangle.v[1][2] + triangle.v[2][2]) / 3.0,
        ];
        for axis in 0..3 {
            min[axis] = min[axis].min(centroid[axis]);
            max[axis] = max[axis].max(centroid[axis]);
        }
        centroids.push(centroid);
    }

    let extent = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    let axis = if extent[0] >= extent[1] && extent[0] >= extent[2] {
        0
    } else if extent[1] >= extent[2] {
        1
    } else {
        2
    };

    let mut values: Vec<f64> = centroids.iter().map(|c| c[axis]).collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = values[values.len() / 2];

    let mut normal = [0.0; 3];
    normal[axis] = 1.0;
    Plane { normal, d: -median }
}

fn compute_normal(v0: &[f64; 3], v1: &[f64; 3], v2: &[f64; 3]) -> Option<[f64; 3]> {
    let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
    let normal = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    let length = dot(normal, normal).sqrt();
    if !length.is_finite() || length <= 1.0e-10 {
        None
    } else {
        Some([normal[0] / length, normal[1] / length, normal[2] / length])
    }
}

fn fan_triangulate(vertices: &[[f64; 3]]) -> Vec<Triangle> {
    if vertices.len() < 3 {
        return Vec::new();
    }
    (1..vertices.len() - 1)
        .filter_map(|index| {
            let v = [vertices[0], vertices[index], vertices[index + 1]];
            compute_normal(&v[0], &v[1], &v[2]).map(|normal| Triangle { v, normal })
        })
        .collect()
}

fn to_f64(v: [f32; 3]) -> [f64; 3] {
    [v[0] as f64, v[1] as f64, v[2] as f64]
}

fn to_f32(v: [f64; 3]) -> [f32; 3] {
    [v[0] as f32, v[1] as f32, v[2] as f32]
}

fn mesh_to_triangles(mesh: &TriangleMesh) -> Vec<Triangle> {
    mesh.indices
        .iter()
        .filter_map(|indices| {
            let v0 = to_f64(*mesh.vertices.get(indices[0] as usize)?);
            let v1 = to_f64(*mesh.vertices.get(indices[1] as usize)?);
            let v2 = to_f64(*mesh.vertices.get(indices[2] as usize)?);
            let normal = compute_normal(&v0, &v1, &v2)?;
            Some(Triangle {
                v: [v0, v1, v2],
                normal,
            })
        })
        .collect()
}

fn triangles_to_mesh(triangles: &[Triangle]) -> TriangleMesh {
    let mut vertices = Vec::with_capacity(triangles.len() * 3);
    let mut normals = Vec::with_capacity(triangles.len() * 3);
    let mut indices = Vec::with_capacity(triangles.len());
    for triangle in triangles {
        let base = vertices.len() as u32;
        vertices.push(to_f32(triangle.v[0]));
        vertices.push(to_f32(triangle.v[1]));
        vertices.push(to_f32(triangle.v[2]));
        let normal_f32 = to_f32(triangle.normal);
        normals.extend_from_slice(&[normal_f32; 3]);
        indices.push([base, base + 1, base + 2]);
    }
    TriangleMesh {
        vertices,
        normals,
        indices,
    }
}

/// Point-on-open-segment test (excludes the endpoints themselves): is `p`
/// collinear with, and strictly between, `a` and `b` within `epsilon`?
fn point_on_open_segment(p: [f64; 3], a: [f64; 3], b: [f64; 3], epsilon: f64) -> bool {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ab_len2 = dot(ab, ab);
    if ab_len2 <= epsilon * epsilon {
        return false;
    }
    let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
    let t = dot(ap, ab) / ab_len2;
    // Exclude points at (or past) the endpoints -- those are handled by the
    // ordinary "shares a vertex" case, not a T-junction split.
    let margin = epsilon / ab_len2.sqrt();
    if t <= margin || t >= 1.0 - margin {
        return false;
    }
    let closest = [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t];
    let dist2 =
        (p[0] - closest[0]).powi(2) + (p[1] - closest[1]).powi(2) + (p[2] - closest[2]).powi(2);
    dist2 <= epsilon * epsilon
}

fn points_coincide(a: [f64; 3], b: [f64; 3], epsilon: f64) -> bool {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2) <= epsilon * epsilon
}

/// Repair T-junctions produced by clipping a solid's faces independently
/// against the other operand's BSP tree.
///
/// Two triangles that were originally adjacent faces of the *same* solid
/// (sharing a full edge) can each get fragmented differently near a
/// boolean-op boundary, even where that shared edge itself never comes
/// anywhere near the other operand: each face independently encounters a
/// different subset of the other tree's planes, so each accumulates its own
/// subdivision pattern. The result is a classic non-conforming mesh
/// ("T-junction"): one side's edge is intact while the other side's
/// matching edge has been split partway along by a vertex the first side
/// doesn't have. Bounding-box locality pruning (see `clip_triangles`)
/// eliminates *unnecessary* splits, but a T-junction can still appear near
/// a genuine local boolean-op boundary because the two faces approach it
/// from different directions and do not coordinate their subdivision.
///
/// This performs the standard fix: for every mesh edge, find any other
/// vertex in the mesh lying exactly on that edge (not at an endpoint) and
/// insert it, splitting the owning triangle in two along the original
/// triangle's own plane (so its normal is unchanged and no new geometry is
/// introduced -- purely a re-triangulation). Runs to a fixed point (bounded
/// by `MAX_PASSES`) since inserting a point can reveal a further point on
/// one of the two new, shorter edges.
fn weld_t_junctions(mut triangles: Vec<Triangle>) -> Vec<Triangle> {
    const WELD_EPSILON: f64 = 1.0e-6;
    const MAX_PASSES: usize = 24;

    for _pass in 0..MAX_PASSES {
        let mut points: Vec<[f64; 3]> = Vec::new();
        for triangle in &triangles {
            for vertex in &triangle.v {
                if !points
                    .iter()
                    .any(|p| points_coincide(*p, *vertex, WELD_EPSILON))
                {
                    points.push(*vertex);
                }
            }
        }

        let mut result = Vec::with_capacity(triangles.len());
        let mut any_split = false;
        for triangle in triangles {
            let mut found: Option<(usize, usize, usize, [f64; 3])> = None;
            'edges: for &(i0, i1, i2) in &[(0usize, 1usize, 2usize), (1, 2, 0), (2, 0, 1)] {
                let a = triangle.v[i0];
                let b = triangle.v[i1];
                for &p in &points {
                    if points_coincide(p, a, WELD_EPSILON) || points_coincide(p, b, WELD_EPSILON) {
                        continue;
                    }
                    if point_on_open_segment(p, a, b, WELD_EPSILON) {
                        found = Some((i0, i1, i2, p));
                        break 'edges;
                    }
                }
            }
            if let Some((i0, i1, i2, p)) = found {
                let normal = triangle.normal;
                result.push(Triangle {
                    v: [triangle.v[i0], p, triangle.v[i2]],
                    normal,
                });
                result.push(Triangle {
                    v: [p, triangle.v[i1], triangle.v[i2]],
                    normal,
                });
                any_split = true;
            } else {
                result.push(triangle);
            }
        }
        triangles = result;
        if !any_split {
            break;
        }
    }

    triangles
}

/// CSG union of two closed triangle solids.
pub fn csg_union(a: &TriangleMesh, b: &TriangleMesh) -> TriangleMesh {
    let a_triangles = mesh_to_triangles(a);
    let b_triangles = mesh_to_triangles(b);
    if a_triangles.is_empty() {
        return b.clone();
    }
    if b_triangles.is_empty() {
        return a.clone();
    }

    let mut a_node = BSPNode::from_triangles(a_triangles);
    let mut b_node = BSPNode::from_triangles(b_triangles);
    a_node.clip_to(&b_node);
    b_node.clip_to(&a_node);
    b_node.invert();
    b_node.clip_to(&a_node);
    b_node.invert();
    a_node.build(b_node.all_triangles());
    triangles_to_mesh(&weld_t_junctions(a_node.all_triangles()))
}

/// CSG subtraction of two closed triangle solids: `a - b`.
pub fn csg_subtract(a: &TriangleMesh, b: &TriangleMesh) -> TriangleMesh {
    let a_triangles = mesh_to_triangles(a);
    let b_triangles = mesh_to_triangles(b);
    if a_triangles.is_empty() || b_triangles.is_empty() {
        return a.clone();
    }

    let mut a_node = BSPNode::from_triangles(a_triangles);
    let mut b_node = BSPNode::from_triangles(b_triangles);
    a_node.invert();
    a_node.clip_to(&b_node);
    b_node.clip_to(&a_node);
    b_node.invert();
    b_node.clip_to(&a_node);
    b_node.invert();
    a_node.build(b_node.all_triangles());
    a_node.invert();
    triangles_to_mesh(&weld_t_junctions(a_node.all_triangles()))
}

/// CSG intersection of two closed triangle solids.
pub fn csg_intersect(a: &TriangleMesh, b: &TriangleMesh) -> TriangleMesh {
    let a_triangles = mesh_to_triangles(a);
    let b_triangles = mesh_to_triangles(b);
    if a_triangles.is_empty() || b_triangles.is_empty() {
        return TriangleMesh::empty();
    }

    let mut a_node = BSPNode::from_triangles(a_triangles);
    let mut b_node = BSPNode::from_triangles(b_triangles);
    a_node.invert();
    b_node.clip_to(&a_node);
    b_node.invert();
    a_node.clip_to(&b_node);
    b_node.clip_to(&a_node);
    a_node.build(b_node.all_triangles());
    a_node.invert();
    triangles_to_mesh(&weld_t_junctions(a_node.all_triangles()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csg::{CSGNode, Transform3D};
    use crate::mesh::resolve_to_mesh;
    use crate::validate::compute_signed_volume;

    fn unit_cube() -> TriangleMesh {
        resolve_to_mesh(&CSGNode::cube())
    }

    fn offset_cube() -> TriangleMesh {
        resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
            translate: [0.25, 0.25, 0.25],
            ..Default::default()
        }))
    }

    fn small_cube() -> TriangleMesh {
        resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
            scale: [0.5; 3],
            ..Default::default()
        }))
    }

    fn volume(mesh: &TriangleMesh) -> f32 {
        compute_signed_volume(mesh).abs()
    }

    fn assert_volume(mesh: &TriangleMesh, expected: f32) {
        let actual = volume(mesh);
        assert!(
            (actual - expected).abs() < 1.0e-3,
            "expected volume {expected}, got {actual}"
        );
    }

    #[test]
    fn union_overlapping_has_oracle_volume() {
        // Two unit cubes offset by 0.25 overlap by 0.75^3.
        assert_volume(&csg_union(&unit_cube(), &offset_cube()), 1.578_125);
    }

    #[test]
    fn union_nonoverlapping_preserves_both_solids() {
        let distant = resolve_to_mesh(&CSGNode::cube().translate(10.0, 0.0, 0.0));
        assert_volume(&csg_union(&unit_cube(), &distant), 2.0);
    }

    #[test]
    fn subtract_nonoverlapping_preserves_a() {
        let distant = resolve_to_mesh(&CSGNode::cube().translate(10.0, 0.0, 0.0));
        assert_volume(&csg_subtract(&unit_cube(), &distant), 1.0);
    }

    #[test]
    fn subtract_overlapping_has_oracle_volume() {
        assert_volume(&csg_subtract(&unit_cube(), &offset_cube()), 0.578_125);
    }

    #[test]
    fn subtract_contained_has_oracle_volume() {
        assert_volume(&csg_subtract(&unit_cube(), &small_cube()), 0.875);
    }

    #[test]
    fn intersect_overlapping_has_oracle_volume() {
        assert_volume(&csg_intersect(&unit_cube(), &offset_cube()), 0.421_875);
    }

    #[test]
    fn intersect_nonoverlapping_is_empty() {
        let distant = resolve_to_mesh(&CSGNode::cube().translate(10.0, 0.0, 0.0));
        let result = csg_intersect(&unit_cube(), &distant);
        assert_eq!(result.triangle_count(), 0);
        assert_eq!(volume(&result), 0.0);
    }

    #[test]
    fn empty_operands_follow_set_identities() {
        let empty = TriangleMesh::empty();
        assert_volume(&csg_union(&unit_cube(), &empty), 1.0);
        assert_volume(&csg_subtract(&unit_cube(), &empty), 1.0);
        assert_eq!(csg_intersect(&unit_cube(), &empty).triangle_count(), 0);
    }
}
