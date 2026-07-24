// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Artifact-inspecting feature extraction from a scene graph.
//!
//! Everything here is computed from the **artwork itself** — element sizes,
//! actual fill/stroke colors, spatial layout — as opposed to the cognitive
//! snapshot the artwork was generated from. This is the input side of an
//! honest aesthetic scorer: without it, two scenes generated from the same
//! snapshot are indistinguishable to the evaluator no matter how differently
//! they look (which was exactly the state of `score_scene` before 2026-07-10:
//! its only artwork-dependent signal was node count).
//!
//! Path geometry is approximated by treating the numeric tokens of the `d`
//! string as absolute coordinate pairs. Every generator in this workspace
//! emits absolute `M`/`L`/`C` commands (see e.g. `atelier::curves`), so this
//! is accurate for in-tree art; paths using relative commands or arc flags
//! would degrade to a rough bounding box, never a panic.

use crate::scene_graph::{NodeKind, SceneNode, Transform};

/// Number of hue bins in the color histogram (30° each).
pub const HUE_BINS: usize = 12;

/// Number of element-kind categories tracked
/// (circle, ellipse, line, polygon, rect, path).
pub const KIND_BINS: usize = 6;

/// Features measured directly from a scene graph.
#[derive(Debug, Clone)]
pub struct SceneFeatures {
    /// Characteristic sizes of drawable elements (viewport units, transform-
    /// scaled), sorted descending. Consecutive ratios of this size hierarchy
    /// are what golden-proportion scoring consumes.
    pub element_sizes: Vec<f32>,
    /// Histogram of chromatic color samples over [`HUE_BINS`] hue bins
    /// (fills, strokes, and gradient stops; saturation ≥ 0.1, alpha > 0).
    pub hue_histogram: [f32; HUE_BINS],
    /// Number of chromatic color samples (saturation ≥ 0.1).
    pub chromatic_samples: f32,
    /// Number of achromatic color samples (gray/white/black).
    pub achromatic_samples: f32,
    /// Histogram of drawable element kinds over [`KIND_BINS`] categories.
    pub kind_histogram: [f32; KIND_BINS],
    /// Element centers per quadrant of the content bounding box.
    pub quadrant_histogram: [f32; 4],
    /// Distance of the mean element center from the content-bbox center,
    /// normalized by the bbox half-diagonal: 0.0 = perfectly centered mass,
    /// 1.0 = everything piled in a corner.
    pub centroid_offset: f32,
    /// Number of drawable (non-group, non-defs) elements.
    pub element_count: usize,
}

impl SceneFeatures {
    /// Color diversity in [0, 1]: normalized entropy of the hue histogram,
    /// weighted by the chromatic fraction. A monochrome or grayscale piece
    /// scores near 0; an evenly multi-hued piece scores near 1.
    pub fn color_diversity(&self) -> f32 {
        let total = self.chromatic_samples + self.achromatic_samples;
        if total <= 0.0 {
            return 0.0;
        }
        let chromatic_fraction = self.chromatic_samples / total;
        normalized_entropy(&self.hue_histogram) * chromatic_fraction
    }

    /// Element-kind diversity in [0, 1]: normalized entropy of the kind
    /// histogram. All-circles scores 0; a balanced mix of primitives scores
    /// high.
    pub fn kind_diversity(&self) -> f32 {
        normalized_entropy(&self.kind_histogram)
    }

    /// Spatial balance in [0, 1]: how evenly the composition occupies its own
    /// bounding box. Blend of quadrant-occupancy entropy and centered visual
    /// mass. Single-element or empty scenes return 0.
    pub fn spatial_balance(&self) -> f32 {
        if self.element_count < 2 {
            return 0.0;
        }
        let quadrant_entropy = normalized_entropy(&self.quadrant_histogram);
        0.5 * quadrant_entropy + 0.5 * (1.0 - self.centroid_offset)
    }
}

/// Shannon entropy of a non-negative histogram, normalized to [0, 1]
/// (1.0 = uniform over all bins). Local helper so this crate stays
/// dependency-free; mirrors `symthaea_aesthetic::information`.
fn normalized_entropy(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }
    let sum: f32 = values.iter().filter(|v| **v > 0.0).sum();
    if sum <= 0.0 {
        return 0.0;
    }
    let entropy: f32 = values
        .iter()
        .filter(|&&v| v > 0.0)
        .map(|&v| {
            let p = v / sum;
            -p * p.ln()
        })
        .sum();
    (entropy / (values.len() as f32).ln()).clamp(0.0, 1.0)
}

/// Accumulated similarity transform (uniform scale + rotation + translation).
/// SVG applies `translate(..) rotate(..) scale(..)` right-to-left to a point:
/// scale first, then rotate, then translate.
#[derive(Debug, Clone, Copy)]
struct Affine {
    tx: f32,
    ty: f32,
    rot: f32, // radians
    scale: f32,
}

impl Affine {
    fn identity() -> Self {
        Self {
            tx: 0.0,
            ty: 0.0,
            rot: 0.0,
            scale: 1.0,
        }
    }

    fn apply(&self, (x, y): (f32, f32)) -> (f32, f32) {
        let (sx, sy) = (x * self.scale, y * self.scale);
        let (sin, cos) = self.rot.sin_cos();
        (self.tx + sx * cos - sy * sin, self.ty + sx * sin + sy * cos)
    }

    /// Compose with a node's transform: the result applies `t` first, then
    /// `self` (parent-then-child ordering of the scene walk).
    fn then(&self, t: &Transform) -> Self {
        let scale = t.scale;
        let (tx, ty) = self.apply((t.translate_x, t.translate_y));
        Self {
            tx,
            ty,
            rot: self.rot + t.rotate_deg.to_radians(),
            scale: self.scale * scale,
        }
    }
}

/// Extract [`SceneFeatures`] from a scene graph.
///
/// Single recursive walk; all coordinates are mapped through the accumulated
/// node transforms (uniform-scale approximation matching [`Transform`]'s own
/// representation). Cheap relative to SVG rendering — safe to call once per
/// candidate inside an iteration budget.
pub fn extract_scene_features(scene: &SceneNode) -> SceneFeatures {
    let mut acc = Accumulator::default();
    walk(scene, Affine::identity(), &mut acc);
    acc.finish()
}

#[derive(Default)]
struct Accumulator {
    element_sizes: Vec<f32>,
    hue_histogram: [f32; HUE_BINS],
    chromatic_samples: f32,
    achromatic_samples: f32,
    kind_histogram: [f32; KIND_BINS],
    centers: Vec<(f32, f32)>,
}

impl Accumulator {
    fn add_color(&mut self, color: &crate::color::Color) {
        if color.a <= 0.0 {
            return;
        }
        let (h, s, _l) = color.to_hsl();
        if s < 0.1 {
            self.achromatic_samples += 1.0;
        } else {
            let bin = ((h.rem_euclid(360.0) / 30.0) as usize).min(HUE_BINS - 1);
            self.hue_histogram[bin] += 1.0;
            self.chromatic_samples += 1.0;
        }
    }

    fn add_element(&mut self, kind_bin: usize, center: (f32, f32), size: f32) {
        self.kind_histogram[kind_bin] += 1.0;
        self.centers.push(center);
        if size.is_finite() && size > 0.0 {
            self.element_sizes.push(size);
        }
    }

    fn finish(mut self) -> SceneFeatures {
        self.element_sizes
            .sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Content bounding box over element centers → quadrant occupancy
        // and centroid offset.
        let mut quadrants = [0.0f32; 4];
        let mut centroid_offset = 0.0;
        if self.centers.len() >= 2 {
            let (mut min_x, mut min_y) = (f32::INFINITY, f32::INFINITY);
            let (mut max_x, mut max_y) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
            for &(x, y) in &self.centers {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
            let cx = (min_x + max_x) / 2.0;
            let cy = (min_y + max_y) / 2.0;
            let half_diag = (((max_x - min_x).powi(2) + (max_y - min_y).powi(2)).sqrt() / 2.0)
                .max(f32::EPSILON);

            let (mut mean_x, mut mean_y) = (0.0f32, 0.0f32);
            for &(x, y) in &self.centers {
                let qx = usize::from(x > cx);
                let qy = usize::from(y > cy);
                quadrants[qy * 2 + qx] += 1.0;
                mean_x += x;
                mean_y += y;
            }
            mean_x /= self.centers.len() as f32;
            mean_y /= self.centers.len() as f32;
            centroid_offset = (((mean_x - cx).powi(2) + (mean_y - cy).powi(2)).sqrt() / half_diag)
                .clamp(0.0, 1.0);
        }

        SceneFeatures {
            element_sizes: self.element_sizes,
            hue_histogram: self.hue_histogram,
            chromatic_samples: self.chromatic_samples,
            achromatic_samples: self.achromatic_samples,
            kind_histogram: self.kind_histogram,
            quadrant_histogram: quadrants,
            centroid_offset,
            element_count: self.centers.len(),
        }
    }
}

fn walk(node: &SceneNode, parent: Affine, acc: &mut Accumulator) {
    let affine = parent.then(&node.transform);

    if let Some(fill) = &node.style.fill {
        acc.add_color(fill);
    }
    if let Some(stroke) = &node.style.stroke {
        acc.add_color(stroke);
    }

    match &node.kind {
        NodeKind::Circle { cx, cy, r } => {
            acc.add_element(0, affine.apply((*cx, *cy)), 2.0 * r * affine.scale);
        }
        NodeKind::Ellipse { cx, cy, rx, ry } => {
            acc.add_element(1, affine.apply((*cx, *cy)), (rx + ry) * affine.scale);
        }
        NodeKind::Line { x1, y1, x2, y2 } => {
            let mid = affine.apply(((x1 + x2) / 2.0, (y1 + y2) / 2.0));
            let len = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt() * affine.scale;
            acc.add_element(2, mid, len);
        }
        NodeKind::Polygon { points, .. } => {
            if let Some((center, diag)) = bbox_center_diag(points) {
                acc.add_element(3, affine.apply(center), diag * affine.scale);
            }
        }
        NodeKind::Rect { x, y, w, h, .. } => {
            let center = affine.apply((x + w / 2.0, y + h / 2.0));
            // A rect contributes both side lengths to the size hierarchy —
            // its own aspect ratio participates in golden scoring.
            acc.kind_histogram[4] += 1.0;
            acc.centers.push(center);
            for side in [*w, *h] {
                let s = side * affine.scale;
                if s.is_finite() && s > 0.0 {
                    acc.element_sizes.push(s);
                }
            }
        }
        NodeKind::Path { d } => {
            let points = path_numeric_points(d);
            if let Some((center, diag)) = bbox_center_diag(&points) {
                acc.add_element(5, affine.apply(center), diag * affine.scale);
            }
        }
        NodeKind::RadialGradient { stops, .. } => {
            for stop in stops {
                acc.add_color(&stop.color);
            }
        }
        NodeKind::Group { .. } | NodeKind::Filter { .. } | NodeKind::UseFilter { .. } => {}
    }

    for child in &node.children {
        walk(child, affine, acc);
    }
}

/// Per-element features for relational analysis (e.g. compositional
/// integration: building an element-similarity graph and measuring how
/// irreducible the composition is). One entry per drawable element,
/// transform-applied, document order.
#[derive(Debug, Clone)]
pub struct ElementFeature {
    /// Element center in scene coordinates.
    pub center: (f32, f32),
    /// Characteristic size (same convention as `SceneFeatures::element_sizes`;
    /// rects use their bbox diagonal here since one scalar is needed).
    pub size: f32,
    /// Paint as (hue°, saturation, lightness) from the fill (stroke as
    /// fallback); `None` for unstyled or fully transparent elements.
    pub color: Option<(f32, f32, f32)>,
}

/// Extract per-element features from a scene graph. Same walk semantics as
/// [`extract_scene_features`] (accumulated transforms, path bbox
/// approximation), but keeps elements individual instead of aggregating.
pub fn extract_element_features(scene: &SceneNode) -> Vec<ElementFeature> {
    fn paint_of(node: &SceneNode) -> Option<(f32, f32, f32)> {
        node.style
            .fill
            .as_ref()
            .or(node.style.stroke.as_ref())
            .filter(|c| c.a > 0.0)
            .map(|c| c.to_hsl())
    }

    fn walk_elements(node: &SceneNode, parent: Affine, out: &mut Vec<ElementFeature>) {
        let affine = parent.then(&node.transform);
        let color = paint_of(node);
        let geometry: Option<((f32, f32), f32)> = match &node.kind {
            NodeKind::Circle { cx, cy, r } => {
                Some((affine.apply((*cx, *cy)), 2.0 * r * affine.scale))
            }
            NodeKind::Ellipse { cx, cy, rx, ry } => {
                Some((affine.apply((*cx, *cy)), (rx + ry) * affine.scale))
            }
            NodeKind::Line { x1, y1, x2, y2 } => Some((
                affine.apply(((x1 + x2) / 2.0, (y1 + y2) / 2.0)),
                ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt() * affine.scale,
            )),
            NodeKind::Polygon { points, .. } => bbox_center_diag(points)
                .map(|(center, diag)| (affine.apply(center), diag * affine.scale)),
            NodeKind::Rect { x, y, w, h, .. } => Some((
                affine.apply((x + w / 2.0, y + h / 2.0)),
                (w * w + h * h).sqrt() * affine.scale,
            )),
            NodeKind::Path { d } => bbox_center_diag(&path_numeric_points(d))
                .map(|(center, diag)| (affine.apply(center), diag * affine.scale)),
            NodeKind::Group { .. }
            | NodeKind::RadialGradient { .. }
            | NodeKind::Filter { .. }
            | NodeKind::UseFilter { .. } => None,
        };
        if let Some((center, size)) = geometry {
            if size.is_finite() && size > 0.0 && center.0.is_finite() && center.1.is_finite() {
                out.push(ElementFeature {
                    center,
                    size,
                    color,
                });
            }
        }
        for child in &node.children {
            walk_elements(child, affine, out);
        }
    }

    let mut out = Vec::new();
    walk_elements(scene, Affine::identity(), &mut out);
    out
}

/// Bounding-box center and diagonal of a point set; `None` when empty.
fn bbox_center_diag(points: &[(f32, f32)]) -> Option<((f32, f32), f32)> {
    if points.is_empty() {
        return None;
    }
    let (mut min_x, mut min_y) = (f32::INFINITY, f32::INFINITY);
    let (mut max_x, mut max_y) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &(x, y) in points {
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    if !min_x.is_finite() {
        return None;
    }
    let center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0);
    let diag = ((max_x - min_x).powi(2) + (max_y - min_y).powi(2)).sqrt();
    Some((center, diag))
}

/// Extract numeric tokens from an SVG path `d` string as (x, y) pairs.
/// See the module docs for the absolute-coordinate approximation this makes.
fn path_numeric_points(d: &str) -> Vec<(f32, f32)> {
    let mut numbers: Vec<f32> = Vec::new();
    let mut current = String::new();
    let flush = |current: &mut String, numbers: &mut Vec<f32>| {
        if !current.is_empty() {
            if let Ok(v) = current.parse::<f32>() {
                numbers.push(v);
            }
            current.clear();
        }
    };
    for ch in d.chars() {
        match ch {
            '0'..='9' | '.' => current.push(ch),
            '-' => {
                // '-' both separates ("10-20") and signs ("-20") numbers.
                flush(&mut current, &mut numbers);
                current.push(ch);
            }
            _ => flush(&mut current, &mut numbers),
        }
    }
    flush(&mut current, &mut numbers);

    numbers.chunks_exact(2).map(|p| (p[0], p[1])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color;
    use crate::scene_graph::Style;

    fn filled(node: SceneNode, color: Color) -> SceneNode {
        node.with_style(Style {
            fill: Some(color),
            ..Style::default()
        })
    }

    #[test]
    fn empty_scene_is_safe() {
        let features = extract_scene_features(&SceneNode::group(None));
        assert_eq!(features.element_count, 0);
        assert_eq!(features.color_diversity(), 0.0);
        assert_eq!(features.kind_diversity(), 0.0);
        assert_eq!(features.spatial_balance(), 0.0);
        assert!(features.element_sizes.is_empty());
    }

    #[test]
    fn multi_hue_beats_monochrome() {
        let mut mono = SceneNode::group(None);
        let mut varied = SceneNode::group(None);
        for i in 0..6 {
            let x = i as f32 * 50.0;
            mono.children.push(filled(
                SceneNode::circle(x, 0.0, 10.0),
                Color::from_hsl(200.0, 0.8, 0.5),
            ));
            varied.children.push(filled(
                SceneNode::circle(x, 0.0, 10.0),
                Color::from_hsl(i as f32 * 60.0, 0.8, 0.5),
            ));
        }
        let d_mono = extract_scene_features(&mono).color_diversity();
        let d_varied = extract_scene_features(&varied).color_diversity();
        assert!(
            d_varied > d_mono,
            "varied hues {d_varied} should beat monochrome {d_mono}"
        );
    }

    #[test]
    fn element_features_carry_transform_and_paint() {
        let scene = SceneNode::group(None)
            .with_transform(Transform {
                translate_x: 100.0,
                translate_y: 0.0,
                rotate_deg: 0.0,
                scale: 2.0,
            })
            .with_child(filled(
                SceneNode::circle(10.0, 0.0, 5.0),
                Color::from_hsl(120.0, 0.9, 0.5),
            ))
            .with_child(SceneNode::rect(0.0, 0.0, 30.0, 40.0));
        let elements = extract_element_features(&scene);
        assert_eq!(elements.len(), 2);
        // Circle: center (100 + 2*10, 0), size 2r*scale = 20.
        assert!((elements[0].center.0 - 120.0).abs() < 1e-3);
        assert!((elements[0].size - 20.0).abs() < 1e-3);
        let (h, s, _l) = elements[0].color.expect("painted");
        assert!((h - 120.0).abs() < 1.0 && s > 0.5);
        // Unstyled rect: no paint, bbox-diagonal size 2*50 = 100.
        assert!(elements[1].color.is_none());
        assert!((elements[1].size - 100.0).abs() < 1e-3);
    }

    #[test]
    fn grayscale_scores_zero_color_diversity() {
        let mut scene = SceneNode::group(None);
        for i in 0..4 {
            scene.children.push(filled(
                SceneNode::circle(i as f32 * 30.0, 0.0, 8.0),
                Color::rgb(
                    0.3 + i as f32 * 0.15,
                    0.3 + i as f32 * 0.15,
                    0.3 + i as f32 * 0.15,
                ),
            ));
        }
        assert_eq!(extract_scene_features(&scene).color_diversity(), 0.0);
    }

    #[test]
    fn spread_layout_beats_piled_layout() {
        // Four circles in four quadrants vs. four in one corner plus one
        // far outlier (same element count, same bbox scale).
        let spread = SceneNode::group(None)
            .with_child(SceneNode::circle(0.0, 0.0, 5.0))
            .with_child(SceneNode::circle(100.0, 0.0, 5.0))
            .with_child(SceneNode::circle(0.0, 100.0, 5.0))
            .with_child(SceneNode::circle(100.0, 100.0, 5.0));
        let piled = SceneNode::group(None)
            .with_child(SceneNode::circle(0.0, 0.0, 5.0))
            .with_child(SceneNode::circle(1.0, 1.0, 5.0))
            .with_child(SceneNode::circle(2.0, 0.0, 5.0))
            .with_child(SceneNode::circle(100.0, 100.0, 5.0));
        let b_spread = extract_scene_features(&spread).spatial_balance();
        let b_piled = extract_scene_features(&piled).spatial_balance();
        assert!(
            b_spread > b_piled,
            "spread {b_spread} should beat piled {b_piled}"
        );
    }

    #[test]
    fn kind_mix_beats_all_circles() {
        let circles = SceneNode::group(None)
            .with_child(SceneNode::circle(0.0, 0.0, 5.0))
            .with_child(SceneNode::circle(10.0, 0.0, 5.0))
            .with_child(SceneNode::circle(20.0, 0.0, 5.0));
        let mixed = SceneNode::group(None)
            .with_child(SceneNode::circle(0.0, 0.0, 5.0))
            .with_child(SceneNode::rect(10.0, 0.0, 8.0, 5.0))
            .with_child(SceneNode::line(20.0, 0.0, 30.0, 10.0));
        let d_circles = extract_scene_features(&circles).kind_diversity();
        let d_mixed = extract_scene_features(&mixed).kind_diversity();
        assert!(d_mixed > d_circles, "mixed {d_mixed} > circles {d_circles}");
    }

    #[test]
    fn zero_scale_is_not_treated_as_identity() {
        let scene = SceneNode::group(None)
            .with_transform(Transform {
                scale: 0.0,
                ..Transform::identity()
            })
            .with_child(SceneNode::circle(10.0, 10.0, 5.0));
        let features = extract_scene_features(&scene);
        assert_eq!(features.element_count, 0);
    }

    #[test]
    fn transforms_affect_position_and_size() {
        // A circle inside a translated, scaled group must land where the
        // transform puts it, at the scaled size.
        let inner = SceneNode::circle(0.0, 0.0, 10.0);
        let group = SceneNode::group(None)
            .with_transform(Transform {
                translate_x: 200.0,
                translate_y: 200.0,
                rotate_deg: 0.0,
                scale: 2.0,
            })
            .with_child(inner)
            .with_child(SceneNode::circle(50.0, 50.0, 10.0));
        let features = extract_scene_features(&group);
        // Scaled size: 2r * scale = 40
        assert!(features.element_sizes.contains(&40.0));
        // Second circle center at 200 + 2*50 = 300 on both axes — verified
        // indirectly: both centers distinct so bbox is non-degenerate.
        assert_eq!(features.element_count, 2);
    }

    #[test]
    fn path_points_extracted() {
        let points = path_numeric_points("M 10.0 20.0 L 30.0 40.0 C 1 2, 3 4, 5 6");
        assert_eq!(points[0], (10.0, 20.0));
        assert_eq!(points[1], (30.0, 40.0));
        assert_eq!(points.len(), 5);
        // Negative coordinates split correctly
        let neg = path_numeric_points("M 10-20 L -5 -6");
        assert_eq!(neg[0], (10.0, -20.0));
        assert_eq!(neg[1], (-5.0, -6.0));
    }

    #[test]
    fn gradient_stops_count_as_colors() {
        use crate::scene_graph::GradientStop;
        let scene = SceneNode::group(None).with_child(SceneNode {
            kind: NodeKind::RadialGradient {
                id: "g".into(),
                stops: vec![
                    GradientStop {
                        offset: 0.0,
                        color: Color::from_hsl(0.0, 0.9, 0.5),
                    },
                    GradientStop {
                        offset: 1.0,
                        color: Color::from_hsl(180.0, 0.9, 0.5),
                    },
                ],
            },
            transform: Transform::identity(),
            style: Style::default(),
            children: vec![],
        });
        let features = extract_scene_features(&scene);
        assert_eq!(features.chromatic_samples, 2.0);
    }
}
