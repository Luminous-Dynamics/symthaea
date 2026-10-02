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
                if *closed {
                    emit_fan_fill(out, &transformed, fill);
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
                emit_fan_fill(out, &points, fill);
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
    fn visit<'a>(node: &'a SceneNode, colors: &mut HashMap<&'a str, Color>) {
        if let NodeKind::RadialGradient { id, stops } = &node.kind {
            if let Some(stop) = stops.first() {
                colors.entry(id.as_str()).or_insert(stop.color);
            }
        }
        for child in &node.children {
            visit(child, colors);
        }
    }

    let mut colors = HashMap::new();
    visit(root, &mut colors);
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

fn emit_fan_fill(out: &mut GpuScene, points: &[[f32; 2]], color: [f32; 4]) {
    if points.len() < 3 {
        return;
    }
    let center = polygon_centroid(points);
    for i in 0..points.len() {
        push_triangle(
            out,
            center,
            points[i],
            points[(i + 1) % points.len()],
            color,
        );
    }
}

fn polygon_centroid(points: &[[f32; 2]]) -> [f32; 2] {
    let mut x = 0.0;
    let mut y = 0.0;
    for point in points {
        x += point[0];
        y += point[1];
    }
    let inv = 1.0 / points.len() as f32;
    [x * inv, y * inv]
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
use web_sys::HtmlCanvasElement;

/// Browser renderer. WebGPU is requested explicitly; callers should retain
/// SVG as a graceful fallback when initialization fails.
#[cfg(target_arch = "wasm32")]
pub struct WebGpuRenderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    config: wgpu::SurfaceConfiguration,
}

#[cfg(target_arch = "wasm32")]
impl WebGpuRenderer {
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

        let width = canvas.width().max(1);
        let height = canvas.height().max(1);
        let capabilities = surface.get_capabilities(&adapter);
        let Some(&format) = capabilities.formats.first() else {
            return Err("WebGPU adapter exposed no surface formats".to_string());
        };
        let Some(&alpha_mode) = capabilities.alpha_modes.first() else {
            return Err("WebGPU adapter exposed no alpha modes".to_string());
        };
        let Some(&present_mode) = capabilities.present_modes.first() else {
            return Err("WebGPU adapter exposed no present modes".to_string());
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
            surface,
            device,
            queue,
            pipeline,
            config,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.device, &self.config);
    }

    pub fn render(&self, scene: &GpuScene) -> Result<(), String> {
        let frame = self
            .surface
            .get_current_texture()
            .map_err(|error| format!("failed to acquire WebGPU surface texture: {error}"))?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let bytes = scene_to_bytes(&scene.vertices);
        let buffer = self
            .device
            .create_buffer(&wgpu::util::BufferInitDescriptor {
                label: Some("Symthaea WebGPU Scene Vertex Buffer"),
                contents: &bytes,
                usage: wgpu::BufferUsages::VERTEX,
            });

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
                pass.set_vertex_buffer(0, buffer.slice(..));
                pass.draw(0..scene.vertices.len() as u32, 0..1);
            }
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }
}

#[cfg(target_arch = "wasm32")]
fn scene_to_bytes(vertices: &[GpuVertex]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vertices.len() * std::mem::size_of::<f32>() * 6);
    for vertex in vertices {
        for value in vertex.position.iter().chain(vertex.color.iter()) {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
    }
    bytes
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
}
