// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! WebGPU projection of the cognitive canvas scene.
//!
//! The scene graph remains the semantic source. This module compiles the small,
//! effect-free cognitive subset into a flat GPU vertex stream, then renders it
//! with wgpu. SVG remains available as a deterministic export/debug adapter.
//!
//! The compiler intentionally targets the primitives emitted by the cognitive
//! canvas geometry builder: groups, circles, ellipses, lines, polygons, and
//! rectangles. Raw SVG paths and effect nodes remain on the SVG path until a
//! dedicated GPU tessellator is introduced.

use std::collections::HashMap;

use crate::color::Color;
use crate::scene_graph::{NodeKind, SceneNode, Style, Transform};

const VIEWPORT_W: f32 = 512.0;
const VIEWPORT_H: f32 = 512.0;
const MAX_GPU_NODES: usize = 512;
const MAX_GPU_NESTING: usize = 24;
const MAX_GPU_VERTICES: usize = 200_000;
const CIRCLE_SEGMENTS: usize = 32;
const MAX_POLYGON_POINTS: usize = 128;
const MIN_LINE_WIDTH: f32 = 0.25;
const DEFAULT_STROKE_WIDTH: f32 = 1.0;

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct GpuVertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuScene {
    pub vertices: Vec<GpuVertex>,
    pub skipped_nodes: usize,
}

impl GpuScene {
    pub fn from_scene(root: &SceneNode) -> Self {
        let mut out = Self::default();
        let gradients = collect_first_gradient_colors(root);
        let mut nodes = 0usize;
        visit(
            root,
            Affine::identity(),
            1.0,
            &gradients,
            0,
            &mut nodes,
            &mut out,
        );
        out
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    pub fn byte_len(&self) -> usize {
        self.vertices.len() * (std::mem::size_of::<f32>() * 6)
    }

}

#[derive(Debug, Clone, Copy)]
struct Affine {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32,
    f: f32,
}

impl Affine {
    const fn identity() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            e: 0.0,
            f: 0.0,
        }
    }

    fn from_transform(t: Transform) -> Self {
        let tx = finite(t.translate_x, 0.0).clamp(-1_000_000.0, 1_000_000.0);
        let ty = finite(t.translate_y, 0.0).clamp(-1_000_000.0, 1_000_000.0);
        let rotation = finite(t.rotate_deg, 0.0)
            .clamp(-1_000_000.0, 1_000_000.0)
            .to_radians();
        let scale = finite(t.scale, 1.0).clamp(-8.0, 8.0);
        let (sin, cos) = rotation.sin_cos();
        Self {
            a: cos * scale,
            b: sin * scale,
            c: -sin * scale,
            d: cos * scale,
            e: tx,
            f: ty,
        }
    }

    fn then(self, next: Self) -> Self {
        Self {
            a: self.a * next.a + self.c * next.b,
            b: self.b * next.a + self.d * next.b,
            c: self.a * next.c + self.c * next.d,
            d: self.b * next.c + self.d * next.d,
            e: self.a * next.e + self.c * next.f + self.e,
            f: self.b * next.e + self.d * next.f + self.f,
        }
    }

    fn apply(self, x: f32, y: f32) -> [f32; 2] {
        [
            self.a * x + self.c * y + self.e,
            self.b * x + self.d * y + self.f,
        ]
    }

    fn scale_abs(self) -> f32 {
        (self.a * self.a + self.b * self.b).sqrt().max(1e-6)
    }
}

fn visit(
    node: &SceneNode,
    parent: Affine,
    parent_opacity: f32,
    gradients: &HashMap<&str, Color>,
    depth: usize,
    nodes: &mut usize,
    out: &mut GpuScene,
) {
    if *nodes >= MAX_GPU_NODES || depth > MAX_GPU_NESTING {
        out.skipped_nodes = out.skipped_nodes.saturating_add(1);
        return;
    }
    *nodes += 1;

    let transform = parent.then(Affine::from_transform(node.transform));
    let opacity = parent_opacity
        * node
            .style
            .opacity
            .map(|value| finite(value, 0.0).clamp(0.0, 1.0))
            .unwrap_or(1.0);

    match &node.kind {
        NodeKind::Group { .. } => {
            for child in &node.children {
                visit(
                    child,
                    transform,
                    opacity,
                    gradients,
                    depth + 1,
                    nodes,
                    out,
                );
                if *nodes >= MAX_GPU_NODES {
                    break;
                }
            }
        }
        NodeKind::Circle { cx, cy, r } => {
            emit_ellipse(
                out,
                transform,
                *cx,
                *cy,
                *r,
                *r,
                opacity,
                &node.style,
                gradients,
            );
        }
        NodeKind::Ellipse { cx, cy, rx, ry } => {
            emit_ellipse(
                out,
                transform,
                *cx,
                *cy,
                *rx,
                *ry,
                opacity,
                &node.style,
                gradients,
            );
        }
        NodeKind::Line { x1, y1, x2, y2 } => {
            let a = transform.apply(finite(*x1, 0.0), finite(*y1, 0.0));
            let b = transform.apply(finite(*x2, 0.0), finite(*y2, 0.0));
            if let Some(color) = effective_stroke(&node.style, opacity) {
                emit_thick_segment(out, a, b, stroke_width(&node.style, transform), color);
            }
        }
        NodeKind::Polygon { points, closed } => {
            if points.len() < 2 || points.len() > MAX_POLYGON_POINTS {
                out.skipped_nodes = out.skipped_nodes.saturating_add(1);
                return;
            }
            let transformed = points
                .iter()
                .map(|&(x, y)| transform.apply(finite(x, 0.0), finite(y, 0.0)))
                .collect::<Vec<_>>();

            if let Some(fill) = effective_fill(&node.style, gradients, opacity) {
                if *closed && !emit_polygon_fill(out, &transformed, fill) {
                    out.skipped_nodes = out.skipped_nodes.saturating_add(1);
                }
            }
            if let Some(stroke) = effective_stroke(&node.style, opacity) {
                let width = stroke_width(&node.style, transform);
                for segment in transformed.windows(2) {
                    emit_thick_segment(out, segment[0], segment[1], width, stroke);
                }
                if *closed {
                    emit_thick_segment(
                        out,
                        *transformed.last().unwrap(),
                        transformed[0],
                        width,
                        stroke,
                    );
                }
            }
        }
        NodeKind::Rect { x, y, w, h, rx } => {
            let x = finite(*x, 0.0);
            let y = finite(*y, 0.0);
            let w = finite(*w, 0.0).max(0.0);
            let h = finite(*h, 0.0).max(0.0);
            if w == 0.0 || h == 0.0 {
                return;
            }

            let rx = finite(*rx, 0.0).max(0.0).min(w * 0.5).min(h * 0.5);
            let points = rounded_rect_points(x, y, w, h, rx)
                .into_iter()
                .map(|(px, py)| transform.apply(px, py))
                .collect::<Vec<_>>();

            if let Some(fill) = effective_fill(&node.style, gradients, opacity) {
                if !emit_polygon_fill(out, &points, fill) {
                    out.skipped_nodes = out.skipped_nodes.saturating_add(1);
                }
            }
            if let Some(stroke) = effective_stroke(&node.style, opacity) {
                let width = stroke_width(&node.style, transform);
                for i in 0..points.len() {
                    emit_thick_segment(
                        out,
                        points[i],
                        points[(i + 1) % points.len()],
                        width,
                        stroke,
                    );
                }
            }
        }
        NodeKind::RadialGradient { .. }
        | NodeKind::Filter { .. }
        | NodeKind::UseFilter { .. } => {}
        NodeKind::Path { .. } => {
            out.skipped_nodes = out.skipped_nodes.saturating_add(1);
        }
    }
}

fn collect_first_gradient_colors(root: &SceneNode) -> HashMap<&str, Color> {
    let mut colors = HashMap::new();
    let mut stack = vec![(root, 0usize)];
    let mut visited = 0usize;

    while let Some((node, depth)) = stack.pop() {
        if depth > MAX_GPU_NESTING || visited >= MAX_GPU_NODES {
            break;
        }
        visited += 1;
        if let NodeKind::RadialGradient { id, stops } = &node.kind {
            if let Some(stop) = stops.first() {
                colors.entry(id.as_str()).or_insert(stop.color);
            }
        }
        for child in node.children.iter().rev() {
            if visited.saturating_add(stack.len()) >= MAX_GPU_NODES {
                break;
            }
            stack.push((child, depth + 1));
        }
    }
    colors
}
fn effective_fill(
    style: &Style,
    gradients: &HashMap<&str, Color>,
    opacity: f32,
) -> Option<[f32; 4]> {
    style
        .fill
        .or_else(|| style.fill_url.as_deref().and_then(|id| gradients.get(id).copied()))
        .map(|color| color.sanitized())
        .map(|color| [color.r, color.g, color.b, color.a * opacity])
        .filter(|rgba| rgba[3] > 0.0)
}

fn effective_stroke(style: &Style, opacity: f32) -> Option<[f32; 4]> {
    style
        .stroke
        .map(|color| color.sanitized())
        .map(|color| [color.r, color.g, color.b, color.a * opacity])
        .filter(|rgba| rgba[3] > 0.0)
}

fn stroke_width(style: &Style, transform: Affine) -> f32 {
    let base = style
        .stroke_width
        .map(|value| finite(value, DEFAULT_STROKE_WIDTH).max(MIN_LINE_WIDTH))
        .unwrap_or(DEFAULT_STROKE_WIDTH);
    (base * transform.scale_abs()).max(MIN_LINE_WIDTH)
}

fn emit_ellipse(
    out: &mut GpuScene,
    transform: Affine,
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    opacity: f32,
    style: &Style,
    gradients: &HashMap<&str, Color>,
) {
    let cx = finite(cx, 0.0);
    let cy = finite(cy, 0.0);
    let rx = finite(rx, 0.0).max(0.0);
    let ry = finite(ry, 0.0).max(0.0);
    if rx == 0.0 || ry == 0.0 {
        return;
    }

    let center = transform.apply(cx, cy);
    let mut points = Vec::with_capacity(CIRCLE_SEGMENTS);
    for i in 0..CIRCLE_SEGMENTS {
        let angle = i as f32 * std::f32::consts::TAU / CIRCLE_SEGMENTS as f32;
        points.push(transform.apply(
            cx + rx * angle.cos(),
            cy + ry * angle.sin(),
        ));
    }

    if let Some(fill) = effective_fill(style, gradients, opacity) {
        for i in 0..CIRCLE_SEGMENTS {
            push_triangle(out, center, points[i], points[(i + 1) % CIRCLE_SEGMENTS], fill);
        }
    }
    if let Some(stroke) = effective_stroke(style, opacity) {
        let width = stroke_width(style, transform);
        for i in 0..CIRCLE_SEGMENTS {
            emit_thick_segment(
                out,
                points[i],
                points[(i + 1) % CIRCLE_SEGMENTS],
                width,
                stroke,
            );
        }
    }
}

fn emit_polygon_fill(out: &mut GpuScene, points: &[[f32; 2]], color: [f32; 4]) -> bool {
    if points.len() < 3 {
        return true;
    }
    if points
        .iter()
        .any(|point| !point[0].is_finite() || !point[1].is_finite())
    {
        return false;
    }
    // Remove only exact adjacent duplicates. They are common at shape seams and
    // otherwise create zero-area candidate ears that can stall triangulation.
    let mut polygon = Vec::with_capacity(points.len());
    for (index, point) in points.iter().enumerate() {
        if polygon
            .last()
            .map_or(true, |&last| points[last] != *point)
        {
            polygon.push(index);
        }
    }
    if polygon.len() > 1
        && points[*polygon.first().unwrap()] == points[*polygon.last().unwrap()]
    {
        polygon.pop();
    }
    if polygon.len() < 3 {
        return true;
    }
    if !is_simple_polygon(points, &polygon) {
        return false;
    }

    let area2 = polygon
        .iter()
        .enumerate()
        .map(|(i, &a)| {
            let b = polygon[(i + 1) % polygon.len()];
            points[a][0] * points[b][1] - points[a][1] * points[b][0]
        })
        .sum::<f32>();
    if !area2.is_finite() || area2.abs() <= 1e-6 {
        return false;
    }

    // Ear clipping handles concave simple polygons without creating triangles
    // that cross the polygon boundary. The point count is already hard-bounded.
    let ccw = area2 > 0.0;
    let mut remaining = polygon;
    let mut triangles = Vec::with_capacity(remaining.len().saturating_sub(2));
    let max_iterations = remaining
        .len()
        .saturating_mul(remaining.len())
        .max(1);
    let mut iterations = 0usize;
    let mut cursor = 0usize;

    while remaining.len() > 3 {
        let len = remaining.len();
        let prev = remaining[(cursor + len - 1) % len];
        let curr = remaining[cursor];
        let next = remaining[(cursor + 1) % len];
        let turn = cross(points[prev], points[curr], points[next]);
        let convex = if ccw { turn > 1e-6 } else { turn < -1e-6 };

        if convex
            && !remaining.iter().any(|&candidate| {
                candidate != prev
                    && candidate != curr
                    && candidate != next
                    && point_in_triangle(
                        points[candidate],
                        points[prev],
                        points[curr],
                        points[next],
                        ccw,
                    )
            })
        {
            triangles.push((prev, curr, next));
            remaining.remove(cursor);
            cursor %= remaining.len();
            iterations = 0;
            continue;
        }

        cursor = (cursor + 1) % len;
        iterations += 1;
        if iterations >= max_iterations {
            return false;
        }
    }

    triangles.push((remaining[0], remaining[1], remaining[2]));

    let needed_vertices = triangles.len().saturating_mul(3);
    if out.vertices.len().saturating_add(needed_vertices) > MAX_GPU_VERTICES {
        return false;
    }
    for (a, b, c) in triangles {
        push_triangle(out, points[a], points[b], points[c], color);
    }
    true
}

fn cross(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Reject self-intersecting simple-polygon violations before ear clipping.
///
/// The wire scene is remote input, so a bow-tie or otherwise crossing polygon
/// must not be handed to the triangulator. The input is already bounded to
/// 128 points, making this O(n²) check small and deterministic.
fn is_simple_polygon(points: &[[f32; 2]], polygon: &[usize]) -> bool {
    let n = polygon.len();
    if n < 4 {
        return true;
    }

    for i in 0..n {
        let a = points[polygon[i]];
        let b = points[polygon[(i + 1) % n]];

        for j in (i + 1)..n {
            // Adjacent edges are allowed to meet at their shared endpoint.
            if j == i + 1 || (i == 0 && j == n - 1) {
                continue;
            }

            let c = points[polygon[j]];
            let d = points[polygon[(j + 1) % n]];
            if segments_intersect_or_touch(a, b, c, d) {
                return false;
            }
        }
    }

    true
}

fn segments_intersect_or_touch(
    a: [f32; 2],
    b: [f32; 2],
    c: [f32; 2],
    d: [f32; 2],
) -> bool {
    const EPSILON: f32 = 1e-6;

    let ab_c = cross(a, b, c);
    let ab_d = cross(a, b, d);
    let cd_a = cross(c, d, a);
    let cd_b = cross(c, d, b);

    if !ab_c.is_finite() || !ab_d.is_finite() || !cd_a.is_finite() || !cd_b.is_finite() {
        return false;
    }

    let proper = ((ab_c > EPSILON && ab_d < -EPSILON)
        || (ab_c < -EPSILON && ab_d > EPSILON))
        && ((cd_a > EPSILON && cd_b < -EPSILON)
            || (cd_a < -EPSILON && cd_b > EPSILON));
    if proper {
        return true;
    }

    (ab_c.abs() <= EPSILON && point_on_segment(a, b, c, EPSILON))
        || (ab_d.abs() <= EPSILON && point_on_segment(a, b, d, EPSILON))
        || (cd_a.abs() <= EPSILON && point_on_segment(c, d, a, EPSILON))
        || (cd_b.abs() <= EPSILON && point_on_segment(c, d, b, EPSILON))
}

fn point_on_segment(
    a: [f32; 2],
    b: [f32; 2],
    point: [f32; 2],
    epsilon: f32,
) -> bool {
    point[0] >= a[0].min(b[0]) - epsilon
        && point[0] <= a[0].max(b[0]) + epsilon
        && point[1] >= a[1].min(b[1]) - epsilon
        && point[1] <= a[1].max(b[1]) + epsilon
}

fn point_in_triangle(
    point: [f32; 2],
    a: [f32; 2],
    b: [f32; 2],
    c: [f32; 2],
    ccw: bool,
) -> bool {
    let ab = cross(a, b, point);
    let bc = cross(b, c, point);
    let ca = cross(c, a, point);
    let epsilon = 1e-6;
    if ccw {
        ab >= -epsilon && bc >= -epsilon && ca >= -epsilon
    } else {
        ab <= epsilon && bc <= epsilon && ca <= epsilon
    }
}

fn emit_thick_segment(
    out: &mut GpuScene,
    start: [f32; 2],
    end: [f32; 2],
    width: f32,
    color: [f32; 4],
) {
    let dx = end[0] - start[0];
    let dy = end[1] - start[1];
    let len = (dx * dx + dy * dy).sqrt();
    if !len.is_finite() || len <= 1e-6 {
        return;
    }
    let half = width.max(MIN_LINE_WIDTH) * 0.5;
    let nx = -dy / len * half;
    let ny = dx / len * half;
    let a = [start[0] + nx, start[1] + ny];
    let b = [start[0] - nx, start[1] - ny];
    let c = [end[0] - nx, end[1] - ny];
    let d = [end[0] + nx, end[1] + ny];
    push_triangle(out, a, b, c, color);
    push_triangle(out, a, c, d, color);
}

fn push_triangle(
    out: &mut GpuScene,
    a: [f32; 2],
    b: [f32; 2],
    c: [f32; 2],
    color: [f32; 4],
) {
    if out.vertices.len().saturating_add(3) > MAX_GPU_VERTICES {
        out.skipped_nodes = out.skipped_nodes.saturating_add(1);
        return;
    }
    let Some(a) = gpu_vertex(a, color) else { return };
    let Some(b) = gpu_vertex(b, color) else { return };
    let Some(c) = gpu_vertex(c, color) else { return };
    out.vertices.extend([a, b, c]);
}

fn gpu_vertex(point: [f32; 2], color: [f32; 4]) -> Option<GpuVertex> {
    if !point[0].is_finite()
        || !point[1].is_finite()
        || color.iter().any(|value| !value.is_finite())
    {
        return None;
    }
    Some(GpuVertex {
        position: [
            (point[0] / (VIEWPORT_W * 0.5)) - 1.0,
            1.0 - (point[1] / (VIEWPORT_H * 0.5)),
        ],
        color,
    })
}

fn rounded_rect_points(x: f32, y: f32, w: f32, h: f32, r: f32) -> Vec<(f32, f32)> {
    if r <= 0.0 {
        return vec![(x, y), (x + w, y), (x + w, y + h), (x, y + h)];
    }

    let mut points = Vec::with_capacity(16);
    let corners = [
        (x + r, y + r, std::f32::consts::PI, std::f32::consts::FRAC_PI_2),
        (x + w - r, y + r, -std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2),
        (x + w - r, y + h - r, 0.0, std::f32::consts::FRAC_PI_2),
        (x + r, y + h - r, std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2),
    ];
    for &(cx, cy, start, step) in &corners {
        for i in 0..4 {
            let angle = start + i as f32 * step / 3.0;
            points.push((cx + r * angle.cos(), cy + r * angle.sin()));
        }
    }
    points
}

fn finite(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

#[cfg(target_arch = "wasm32")]
use std::borrow::Cow;
#[cfg(target_arch = "wasm32")]
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[cfg(target_arch = "wasm32")]
use web_sys::HtmlCanvasElement;

/// Browser renderer. WebGPU is requested explicitly; callers should retain
/// SVG as a graceful fallback when initialization fails.
#[cfg(target_arch = "wasm32")]
pub struct WebGpuRenderer {
    instance: wgpu::Instance,
    canvas: HtmlCanvasElement,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertex_buffer_bytes: usize,
    config: wgpu::SurfaceConfiguration,
    /// Reusable CPU-side staging bytes for scene uploads; avoids a fresh
    /// allocation on every cognitive-frame render.
    upload_bytes: Vec<u8>,
    /// Latched when the underlying WebGPU device is lost. The next render
    /// returns an error so the caller can activate its compatibility path.
    device_lost: Arc<AtomicBool>,
}

#[cfg(target_arch = "wasm32")]
impl WebGpuRenderer {
    /// Initialize a movie renderer for the supplied canvas.
    pub async fn new(canvas: HtmlCanvasElement) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });

        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|error| format!("failed to create WebGPU canvas surface: {error}"))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .map_err(|error| format!("failed to request WebGPU adapter: {error}"))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Symthaea WebGPU Canvas"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|error| format!("failed to create WebGPU device: {error}"))?;

        let device_lost = Arc::new(AtomicBool::new(false));
        {
            let device_lost = Arc::clone(&device_lost);
            device.set_device_lost_callback(move |reason, message| {
                device_lost.store(true, Ordering::Release);
                web_sys::console::warn_2(
                    &format!("Symthaea WebGPU cognitive device lost ({reason:?})").into(),
                    &message.into(),
                );
            });
        }

        let width = canvas.width().max(1);
        let height = canvas.height().max(1);
        let capabilities = surface.get_capabilities(&adapter);
        let Some(&format) = capabilities.formats.first() else {
            return Err("WebGPU adapter exposed no surface formats".to_string());
        };
        let Some(&alpha_mode) = capabilities.alpha_modes.first() else {
            return Err("WebGPU adapter exposed no alpha modes".to_string());
        };
        let present_mode = if capabilities
            .present_modes
            .contains(&wgpu::PresentMode::Fifo)
        {
            wgpu::PresentMode::Fifo
        } else {
            *capabilities
                .present_modes
                .first()
                .ok_or_else(|| "WebGPU adapter exposed no present modes".to_string())?
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode,
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Symthaea WebGPU Canvas Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL_SHADER)),
        });

        let vertex_buffer_bytes = MAX_GPU_VERTICES * std::mem::size_of::<f32>() * 6;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Symthaea WebGPU Persistent Scene Vertex Buffer"),
            size: vertex_buffer_bytes as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Symthaea WebGPU Canvas Pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: (std::mem::size_of::<f32>() * 6) as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: (std::mem::size_of::<f32>() * 2) as u64,
                            shader_location: 1,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Ok(Self {
            instance,
            canvas,
            surface,
            device,
            queue,
            pipeline,
            vertex_buffer,
            vertex_buffer_bytes,
            config,
            upload_bytes: Vec::new(),
            device_lost,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if self.is_device_lost() {
            return;
        }
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.device, &self.config);
    }

    /// Returns whether the device-lost callback has fired.
    pub fn is_device_lost(&self) -> bool {
        self.device_lost.load(Ordering::Acquire)
    }

    pub fn render(&mut self, scene: &GpuScene) -> Result<(), String> {
        if self.device_lost.load(Ordering::Acquire) {
            return Err("WebGPU cognitive device was lost".to_string());
        }
        scene_to_bytes(&scene.vertices, &mut self.upload_bytes);
        if self.upload_bytes.len() > self.vertex_buffer_bytes {
            return Err(format!(
                "GPU scene upload exceeds {} byte bound",
                self.vertex_buffer_bytes
            ));
        }
        let frame = self.acquire_surface_frame()?;
        let bytes = &self.upload_bytes;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        if !bytes.is_empty() {
            self.queue.write_buffer(&self.vertex_buffer, 0, &bytes);
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Symthaea WebGPU Canvas Encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Symthaea WebGPU Canvas Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline);
            if !scene.vertices.is_empty() {
                pass.set_vertex_buffer(0, self.vertex_buffer.slice(..bytes.len() as u64));
                pass.draw(0..scene.vertices.len() as u32, 0..1);
            }
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }

    fn acquire_surface_frame(&mut self) -> Result<wgpu::SurfaceTexture, String> {
        match self.surface.get_current_texture() {
            Ok(frame) => Ok(frame),
            Err(wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.config);
                self.surface
                    .get_current_texture()
                    .map_err(|error| format!("failed to reacquire WebGPU surface texture: {error}"))
            }
            Err(wgpu::SurfaceError::Lost) => {
                if self.is_device_lost() {
                    return Err("WebGPU cognitive device was lost".to_string());
                }
                self.surface = self
                    .instance
                    .create_surface(wgpu::SurfaceTarget::Canvas(self.canvas.clone()))
                    .map_err(|error| {
                        format!("failed to recreate WebGPU canvas surface: {error}")
                    })?;
                self.surface.configure(&self.device, &self.config);
                self.surface
                    .get_current_texture()
                    .map_err(|error| format!("failed to reacquire recreated WebGPU surface texture: {error}"))
            }
            Err(wgpu::SurfaceError::Timeout) => {
                Err("WebGPU surface acquisition timed out".to_string())
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                Err("WebGPU surface acquisition ran out of memory".to_string())
            }
            Err(wgpu::SurfaceError::Other) => {
                Err("WebGPU surface acquisition failed".to_string())
            }
        }
    }

}


#[cfg(target_arch = "wasm32")]
const MAX_MOVIE_WIDTH: u32 = 2048;
#[cfg(target_arch = "wasm32")]
const MAX_MOVIE_HEIGHT: u32 = 2048;
#[cfg(target_arch = "wasm32")]
const MAX_MOVIE_RGBA_BYTES: usize = 32 * 1024 * 1024;

/// WebGPU renderer for decoded RGBA cognitive movie frames.
#[cfg(target_arch = "wasm32")]
pub struct WebGpuMovieRenderer {
    instance: wgpu::Instance,
    canvas: HtmlCanvasElement,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    texture: Option<wgpu::Texture>,
    bind_group: Option<wgpu::BindGroup>,
    config: wgpu::SurfaceConfiguration,
    frame_width: u32,
    frame_height: u32,
    /// Latched when the underlying WebGPU device is lost. The next render
    /// returns an error so the caller can activate its compatibility path.
    device_lost: Arc<AtomicBool>,
}

#[cfg(target_arch = "wasm32")]
impl WebGpuMovieRenderer {
    pub async fn new(canvas: HtmlCanvasElement) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|error| format!("failed to create WebGPU movie surface: {error}"))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .map_err(|error| format!("failed to request WebGPU movie adapter: {error}"))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Symthaea WebGPU Movie"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|error| format!("failed to create WebGPU movie device: {error}"))?;

        let device_lost = Arc::new(AtomicBool::new(false));
        {
            let device_lost = Arc::clone(&device_lost);
            device.set_device_lost_callback(move |reason, message| {
                device_lost.store(true, Ordering::Release);
                web_sys::console::warn_2(
                    &format!("Symthaea WebGPU movie device lost ({reason:?})").into(),
                    &message.into(),
                );
            });
        }

        let width = canvas.width().max(1);
        let height = canvas.height().max(1);
        let capabilities = surface.get_capabilities(&adapter);
        let Some(&format) = capabilities.formats.first() else {
            return Err("WebGPU movie adapter exposed no surface formats".to_string());
        };
        let Some(&alpha_mode) = capabilities.alpha_modes.first() else {
            return Err("WebGPU movie adapter exposed no alpha modes".to_string());
        };
        let present_mode = if capabilities.present_modes.contains(&wgpu::PresentMode::Fifo) {
            wgpu::PresentMode::Fifo
        } else {
            *capabilities
                .present_modes
                .first()
                .ok_or_else(|| "WebGPU movie adapter exposed no present modes".to_string())?
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode,
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Symthaea WebGPU Movie Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Symthaea WebGPU Movie Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Symthaea WebGPU Movie Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL_MOVIE_SHADER)),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Symthaea WebGPU Movie Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Symthaea WebGPU Movie Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Ok(Self {
            instance,
            canvas,
            surface,
            device,
            queue,
            pipeline,
            bind_group_layout,
            sampler,
            texture: None,
            bind_group: None,
            config,
            frame_width: 0,
            frame_height: 0,
            device_lost,
        })
    }

    /// Returns whether the device-lost callback has fired.
    pub fn is_device_lost(&self) -> bool {
        self.device_lost.load(Ordering::Acquire)
    }

    /// Upload one bounded RGBA frame and present it with nearest-neighbour sampling.
    pub fn render(&mut self, width: u32, height: u32, rgba: &[u8]) -> Result<(), String> {
        if self.device_lost.load(Ordering::Acquire) {
            return Err("WebGPU movie device was lost".to_string());
        }
        if width == 0
            || height == 0
            || width > MAX_MOVIE_WIDTH
            || height > MAX_MOVIE_HEIGHT
        {
            return Err("WebGPU movie frame dimensions exceed bounds".to_string());
        }
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| "WebGPU movie frame size overflow".to_string())?;
        if expected > MAX_MOVIE_RGBA_BYTES || rgba.len() != expected {
            return Err("WebGPU movie frame payload exceeds bounds".to_string());
        }

        if self.frame_width != width
            || self.frame_height != height
            || self.bind_group.is_none()
        {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Symthaea WebGPU Persistent Movie Texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Symthaea WebGPU Movie Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            self.texture = Some(texture);
            self.bind_group = Some(bind_group);
            self.frame_width = width;
            self.frame_height = height;
        }

        let texture = self
            .texture
            .as_ref()
            .ok_or_else(|| "WebGPU movie texture was not initialized".to_string())?;
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width.saturating_mul(4)),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.config);
                self.surface
                    .get_current_texture()
                    .map_err(|error| format!("failed to reacquire WebGPU movie surface: {error}"))?
            }
            Err(wgpu::SurfaceError::Lost) => {
                if self.is_device_lost() {
                    return Err("WebGPU movie device was lost".to_string());
                }
                self.surface = self
                    .instance
                    .create_surface(wgpu::SurfaceTarget::Canvas(self.canvas.clone()))
                    .map_err(|error| format!("failed to recreate WebGPU movie surface: {error}"))?;
                self.surface.configure(&self.device, &self.config);
                self.surface
                    .get_current_texture()
                    .map_err(|error| {
                        format!("failed to reacquire recreated WebGPU movie surface: {error}")
                    })?
            }
            Err(wgpu::SurfaceError::Timeout) => {
                return Err("WebGPU movie surface acquisition timed out".to_string());
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                return Err("WebGPU movie surface acquisition ran out of memory".to_string());
            }
            Err(wgpu::SurfaceError::Other) => {
                return Err("WebGPU movie surface acquisition failed".to_string());
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self
            .bind_group
            .as_ref()
            .ok_or_else(|| "WebGPU movie bind group was not initialized".to_string())?;
        let mut encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Symthaea WebGPU Movie Encoder"),
                });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Symthaea WebGPU Movie Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }
}

#[cfg(target_arch = "wasm32")]
fn scene_to_bytes(vertices: &[GpuVertex], bytes: &mut Vec<u8>) {
    let required = vertices.len() * std::mem::size_of::<f32>() * 6;
    bytes.clear();
    if bytes.capacity() < required {
        bytes.reserve(required - bytes.capacity());
    }
    for vertex in vertices {
        for value in vertex.position.iter().chain(vertex.color.iter()) {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
    }
}

#[cfg(target_arch = "wasm32")]
const WGSL_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(position, 0.0, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
    return color;
}
"#;


#[cfg(target_arch = "wasm32")]
const WGSL_MOVIE_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
    );
    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0),
    );
    var out: VertexOutput;
    out.position = vec4<f32>(positions[index], 0.0, 1.0);
    out.uv = uvs[index];
    return out;
}

@group(0) @binding(0)
var movie_texture: texture_2d<f32>;
@group(0) @binding(1)
var movie_sampler: sampler;

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return textureSampleLevel(movie_texture, movie_sampler, input.uv, 0.0);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn circle_scene() -> SceneNode {
        SceneNode::group(None)
            .with_child(
                SceneNode::circle(256.0, 256.0, 80.0).with_style(Style {
                    fill: Some(Color::rgba(0.8, 0.5, 0.1, 0.75)),
                    stroke: Some(Color::rgba(0.1, 0.2, 0.9, 0.5)),
                    stroke_width: Some(2.0),
                    ..Style::default()
                }),
            )
            .with_child(SceneNode::line(0.0, 0.0, 512.0, 512.0).with_style(Style {
                stroke: Some(Color::rgb(1.0, 1.0, 1.0)),
                stroke_width: Some(1.0),
                ..Style::default()
            }))
    }

    #[test]
    fn compiles_core_canvas_primitives_to_bounded_gpu_geometry() {
        let scene = GpuScene::from_scene(&circle_scene());
        assert!(scene.vertex_count() >= 32);
        assert_eq!(scene.skipped_nodes, 0);
        assert!(scene.vertices.iter().all(|vertex| {
            vertex.position.iter().all(|value| value.is_finite())
                && vertex.color.iter().all(|value| value.is_finite())
        }));
    }

    #[test]
    fn effect_nodes_and_paths_are_not_emitted_as_geometry() {
        let root = SceneNode::group(None)
            .with_child(SceneNode {
                kind: NodeKind::Filter {
                    id: "blur".to_string(),
                    filter_type: crate::scene_graph::FilterType::Blur { std_dev: 5.0 },
                },
                transform: Transform::identity(),
                style: Style::default(),
                children: vec![],
            })
            .with_child(SceneNode::path("M 0 0 L 1 1"));
        let scene = GpuScene::from_scene(&root);
        assert!(scene.vertices.is_empty());
        assert_eq!(scene.skipped_nodes, 1);
    }

    #[test]
    fn canonical_cognitive_scene_uses_gpu_supported_primitives() {
        let snapshot = crate::CognitiveSnapshot::dormant();
        let mut engine = crate::AestheticEngine::new();
        let state = engine.process_frame(
            &snapshot,
            crate::FrameContext::from_cycle_count(snapshot.cycle_count),
        );
        let scene = crate::build_scene(&state);
        let gpu = GpuScene::from_scene(&scene);
        assert!(
            gpu.skipped_nodes == 0,
            "canonical cognitive scene emitted unsupported GPU nodes: {}",
            gpu.skipped_nodes
        );
        assert!(
            gpu.vertex_count() > 0,
            "canonical cognitive scene must produce visible GPU geometry"
        );
    }

    #[test]
    fn pathological_scene_is_bounded() {
        let mut root = SceneNode::group(None);
        for _ in 0..600 {
            root.children.push(SceneNode::circle(1.0, 1.0, 1.0));
        }
        let scene = GpuScene::from_scene(&root);
        assert!(scene.vertex_count() <= MAX_GPU_VERTICES);
        assert!(scene.skipped_nodes > 0);
    }

    #[test]
    fn rounded_rect_has_stable_vertex_count() {
        let rect = SceneNode::rect(10.0, 20.0, 100.0, 60.0);
        let scene = GpuScene::from_scene(&rect);
        assert_eq!(scene.vertex_count(), 6);
    }

    #[test]
    fn concave_polygon_is_triangulated_without_centroid_fan_artifacts() {
        let polygon = SceneNode::polygon(
            vec![
                (20.0, 20.0),
                (140.0, 20.0),
                (140.0, 60.0),
                (80.0, 60.0),
                (80.0, 140.0),
                (20.0, 140.0),
            ],
            true,
        )
        .with_style(Style {
            fill: Some(Color::rgb(0.2, 0.4, 0.8)),
            ..Style::default()
        });
        let scene = GpuScene::from_scene(&polygon);
        assert_eq!(scene.vertex_count(), 12, "six-point simple polygon needs four triangles");
        assert_eq!(scene.skipped_nodes, 0);
    }

    #[test]
    fn self_intersecting_polygon_is_rejected() {
        let polygon = SceneNode::polygon(
            vec![
                (20.0, 20.0),
                (140.0, 140.0),
                (20.0, 140.0),
                (140.0, 20.0),
            ],
            true,
        )
        .with_style(Style {
            fill: Some(Color::rgb(0.8, 0.2, 0.2)),
            ..Style::default()
        });
        let scene = GpuScene::from_scene(&polygon);
        assert!(scene.vertices.is_empty());
        assert_eq!(scene.skipped_nodes, 1);
    }

    #[test]
    fn non_adjacent_touching_polygon_edges_are_rejected() {
        let polygon = SceneNode::polygon(
            vec![
                (20.0, 20.0),
                (140.0, 20.0),
                (140.0, 140.0),
                (20.0, 140.0),
                (80.0, 20.0),
            ],
            true,
        )
        .with_style(Style {
            fill: Some(Color::rgb(0.3, 0.5, 0.8)),
            ..Style::default()
        });
        let scene = GpuScene::from_scene(&polygon);
        assert!(scene.vertices.is_empty());
        assert_eq!(scene.skipped_nodes, 1);
    }

    #[test]
    fn non_adjacent_repeated_polygon_vertex_is_rejected() {
        let polygon = SceneNode::polygon(
            vec![
                (20.0, 20.0),
                (140.0, 20.0),
                (140.0, 140.0),
                (20.0, 20.0),
                (20.0, 140.0),
            ],
            true,
        )
        .with_style(Style {
            fill: Some(Color::rgb(0.3, 0.5, 0.8)),
            ..Style::default()
        });
        let scene = GpuScene::from_scene(&polygon);
        assert!(scene.vertices.is_empty());
        assert_eq!(scene.skipped_nodes, 1);
    }

    #[test]
    fn adjacent_duplicate_polygon_vertices_are_normalized() {
        let polygon = SceneNode::polygon(
            vec![
                (20.0, 20.0),
                (140.0, 20.0),
                (140.0, 20.0),
                (140.0, 140.0),
                (20.0, 140.0),
            ],
            true,
        )
        .with_style(Style {
            fill: Some(Color::rgb(0.3, 0.5, 0.8)),
            ..Style::default()
        });
        let scene = GpuScene::from_scene(&polygon);
        assert_eq!(scene.vertex_count(), 9);
        assert_eq!(scene.skipped_nodes, 0);
    }

    #[test]
    fn clockwise_concave_polygon_is_supported() {
        let polygon = SceneNode::polygon(
            vec![
                (20.0, 140.0),
                (80.0, 140.0),
                (80.0, 60.0),
                (140.0, 60.0),
                (140.0, 20.0),
                (20.0, 20.0),
            ],
            true,
        )
        .with_style(Style {
            fill: Some(Color::rgb(0.8, 0.4, 0.2)),
            ..Style::default()
        });
        let scene = GpuScene::from_scene(&polygon);
        assert_eq!(scene.vertex_count(), 12);
        assert_eq!(scene.skipped_nodes, 0);
    }

    #[test]
    fn nested_transforms_and_opacity_survive_gpu_compilation() {
        let scene = SceneNode::group(None)
            .with_style(Style {
                opacity: Some(0.5),
                ..Style::default()
            })
            .with_transform(Transform {
                translate_x: 10.0,
                translate_y: 20.0,
                scale: 2.0,
                ..Transform::identity()
            })
            .with_child(
                SceneNode::rect(0.0, 0.0, 10.0, 10.0).with_style(Style {
                    fill: Some(Color::rgba(0.25, 0.5, 0.75, 0.5)),
                    ..Style::default()
                }),
            );
        let gpu = GpuScene::from_scene(&scene);

        assert_eq!(gpu.vertex_count(), 6);
        assert_eq!(gpu.skipped_nodes, 0);
        let vertex = gpu.vertices[0];
        assert!((vertex.position[0] - (-0.9609375)).abs() < 1e-6);
        assert!((vertex.position[1] - 0.921875).abs() < 1e-6);
        assert!((vertex.color[3] - 0.25).abs() < 1e-6);
    }
}
