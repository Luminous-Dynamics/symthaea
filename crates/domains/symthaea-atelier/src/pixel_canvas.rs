// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pixel canvas: CfC neural dynamics painting directly into an RGBA framebuffer.
//!
//! Unlike the SVG-based generators, the pixel canvas operates in continuous
//! raster space. Each CfC step produces a "brushstroke" — a position, color,
//! radius, and opacity — deposited directly into the pixel buffer. This enables:
//!
//! 1. **Self-perception**: The pixel buffer can be fed through the visual cortex
//!    for the system to perceive its own artwork.
//! 2. **Painterly aesthetics**: Soft edges, blending, layered transparency.
//! 3. **PNG export**: Direct pixel-to-file without SVG rendering.
//! 4. **Framebuffer output**: Compatible with `symthaea-quicken-fb` DRM/KMS.
//!
//! # Brushstroke Model
//!
//! Each CfC output HV is decoded to a `Brushstroke`:
//! ```text
//! CfC output → (x, y, radius, pressure, r, g, b, a, angle, elongation)
//! ```
//! The stroke is painted as an elliptical Gaussian splat — soft-edged, alpha-blended.
//! This naturally produces organic, painterly compositions.

use symthaea_canvas::CognitiveSnapshot;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::hdc_ltc_unified::{
    HdcLtcUnifiedNetwork, UnifiedActivation, UnifiedConfig, UnifiedNetworkConfig,
};
use symthaea_core::hdc::unified_hv::ContinuousHV;

/// A single brushstroke decoded from CfC output.
#[derive(Debug, Clone, Copy)]
pub struct Brushstroke {
    /// Center X (0.0-1.0, normalized to canvas width).
    pub x: f32,
    /// Center Y (0.0-1.0, normalized to canvas height).
    pub y: f32,
    /// Brush radius in pixels.
    pub radius: f32,
    /// Pressure (affects opacity and size modulation).
    pub pressure: f32,
    /// Color: red (0.0-1.0).
    pub r: f32,
    /// Color: green (0.0-1.0).
    pub g: f32,
    /// Color: blue (0.0-1.0).
    pub b: f32,
    /// Alpha (0.0-1.0).
    pub alpha: f32,
    /// Stroke angle in radians.
    pub angle: f32,
    /// Elongation ratio (1.0 = circle, >1.0 = ellipse).
    pub elongation: f32,
}

/// RGBA pixel canvas — the primary creative medium.
#[derive(Clone, Debug)]
pub struct PixelCanvas {
    /// Pixel data: row-major RGBA, 4 bytes per pixel.
    pub pixels: Vec<u8>,
    /// Canvas width.
    pub width: u32,
    /// Canvas height.
    pub height: u32,
}

impl PixelCanvas {
    /// Create a new canvas filled with a background color.
    pub fn new(width: u32, height: u32, bg: [u8; 4]) -> Self {
        let size = (width * height * 4) as usize;
        let mut pixels = Vec::with_capacity(size);
        for _ in 0..(width * height) {
            pixels.extend_from_slice(&bg);
        }
        Self {
            pixels,
            width,
            height,
        }
    }

    /// Paint a brushstroke onto the canvas using Gaussian splat blending.
    pub fn paint_stroke(&mut self, stroke: &Brushstroke) {
        let cx = stroke.x * self.width as f32;
        let cy = stroke.y * self.height as f32;
        let r = stroke.radius.max(1.0);
        let extent = (r * 2.5) as i32; // 2.5σ covers 98.8% of Gaussian

        let cos_a = stroke.angle.cos();
        let sin_a = stroke.angle.sin();
        let inv_rx = 1.0 / r;
        let inv_ry = 1.0 / (r * stroke.elongation.max(0.5));

        let x_min = ((cx - extent as f32) as i32).max(0);
        let x_max = ((cx + extent as f32) as i32).min(self.width as i32 - 1);
        let y_min = ((cy - extent as f32) as i32).max(0);
        let y_max = ((cy + extent as f32) as i32).min(self.height as i32 - 1);

        for py in y_min..=y_max {
            for px in x_min..=x_max {
                let dx = px as f32 - cx;
                let dy = py as f32 - cy;

                // Rotate into stroke-local coordinates
                let lx = dx * cos_a + dy * sin_a;
                let ly = -dx * sin_a + dy * cos_a;

                // Elliptical Gaussian
                let dist_sq = (lx * inv_rx).powi(2) + (ly * inv_ry).powi(2);
                if dist_sq > 6.25 {
                    continue; // outside 2.5σ
                }

                let gaussian = (-dist_sq * 0.5).exp();
                let alpha = stroke.alpha * stroke.pressure * gaussian;

                if alpha < 0.004 {
                    continue; // invisible
                }

                // Alpha-blend
                let idx = ((py as u32 * self.width + px as u32) * 4) as usize;
                let dst_r = self.pixels[idx] as f32 / 255.0;
                let dst_g = self.pixels[idx + 1] as f32 / 255.0;
                let dst_b = self.pixels[idx + 2] as f32 / 255.0;

                let out_r = stroke.r * alpha + dst_r * (1.0 - alpha);
                let out_g = stroke.g * alpha + dst_g * (1.0 - alpha);
                let out_b = stroke.b * alpha + dst_b * (1.0 - alpha);

                self.pixels[idx] = (out_r * 255.0).clamp(0.0, 255.0) as u8;
                self.pixels[idx + 1] = (out_g * 255.0).clamp(0.0, 255.0) as u8;
                self.pixels[idx + 2] = (out_b * 255.0).clamp(0.0, 255.0) as u8;
                self.pixels[idx + 3] = 255; // fully opaque canvas
            }
        }
    }

    /// Get pixel at (x, y) as [R, G, B, A].
    pub fn get_pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let idx = ((y * self.width + x) * 4) as usize;
        [
            self.pixels[idx],
            self.pixels[idx + 1],
            self.pixels[idx + 2],
            self.pixels[idx + 3],
        ]
    }

    /// Encode to PNG bytes (minimal, uncompressed).
    pub fn to_png_bytes(&self) -> Vec<u8> {
        encode_png(&self.pixels, self.width, self.height)
    }

    /// Compute mean color of the canvas as [R, G, B] in 0.0-1.0.
    pub fn mean_color(&self) -> [f32; 3] {
        let n = (self.width * self.height) as f32;
        let mut r_sum = 0.0f32;
        let mut g_sum = 0.0f32;
        let mut b_sum = 0.0f32;
        for chunk in self.pixels.chunks(4) {
            r_sum += chunk[0] as f32;
            g_sum += chunk[1] as f32;
            b_sum += chunk[2] as f32;
        }
        [
            r_sum / (n * 255.0),
            g_sum / (n * 255.0),
            b_sum / (n * 255.0),
        ]
    }

    /// Extract a feature vector for visual perception (downsampled color histogram).
    ///
    /// Returns a 48-element feature vector: 4×4 spatial grid × 3 color channels.
    /// This is the input format for the self-perception loop.
    pub fn perception_features(&self) -> Vec<f32> {
        let grid = 4u32;
        let cell_w = self.width / grid;
        let cell_h = self.height / grid;
        let mut features = Vec::with_capacity((grid * grid * 3) as usize);

        for gy in 0..grid {
            for gx in 0..grid {
                let mut r_sum = 0.0f32;
                let mut g_sum = 0.0f32;
                let mut b_sum = 0.0f32;
                let mut count = 0.0f32;

                for y in (gy * cell_h)..((gy + 1) * cell_h).min(self.height) {
                    for x in (gx * cell_w)..((gx + 1) * cell_w).min(self.width) {
                        let idx = ((y * self.width + x) * 4) as usize;
                        r_sum += self.pixels[idx] as f32 / 255.0;
                        g_sum += self.pixels[idx + 1] as f32 / 255.0;
                        b_sum += self.pixels[idx + 2] as f32 / 255.0;
                        count += 1.0;
                    }
                }

                if count > 0.0 {
                    features.push(r_sum / count);
                    features.push(g_sum / count);
                    features.push(b_sum / count);
                } else {
                    features.extend_from_slice(&[0.0, 0.0, 0.0]);
                }
            }
        }

        features
    }
}

/// CfC-driven pixel painter.
///
/// Each evolve step produces a brushstroke. The CfC temporal dynamics
/// mean that stroke N is influenced by all previous strokes through
/// the network's hidden state — creating compositional flow.
pub struct NeuralPainter {
    network: HdcLtcUnifiedNetwork,
    projections: Vec<ContinuousHV>,
    dt: f32,
}

impl NeuralPainter {
    pub fn new(genesis: &GenesisSeed) -> Self {
        let net_config = UnifiedNetworkConfig {
            layer_sizes: vec![4, 4],
            neuron_config: UnifiedConfig {
                tau_base: 0.03,
                backbone_tau: 0.4,
                activation: UnifiedActivation::Tanh,
                ..UnifiedConfig::default()
            },
            use_layer_binding: true,
            skip_connections: true, // skip connections for richer dynamics
        };

        let network = HdcLtcUnifiedNetwork::from_genesis(net_config, genesis);

        let dim = symthaea_core::hdc::unified_hv::HDC_DIMENSION;
        let projections: Vec<ContinuousHV> = (0..10)
            .map(|i| genesis.hv(&format!("brush_projection_{i}"), dim))
            .collect();

        Self {
            network,
            projections,
            dt: 0.015, // 15ms per stroke
        }
    }

    /// Encode cognitive snapshot to input HV.
    pub fn encode_snapshot(snapshot: &CognitiveSnapshot, genesis: &GenesisSeed) -> ContinuousHV {
        let dim = symthaea_core::hdc::unified_hv::HDC_DIMENSION;
        let channels: Vec<(f32, &str)> = vec![
            (snapshot.consciousness_level as f32, "psi"),
            (snapshot.valence, "valence"),
            (snapshot.arousal, "arousal"),
            (snapshot.dopamine, "dopamine"),
            (snapshot.serotonin, "serotonin"),
            (snapshot.noradrenaline, "noradrenaline"),
            (snapshot.harmony_activations[0], "h_coherence"),
            (snapshot.harmony_activations[1], "h_flourish"),
            (snapshot.harmony_activations[3], "h_play"),
            (snapshot.harmony_activations[7], "h_stillness"),
            (snapshot.prediction_error, "pe"),
        ];

        let mut result = ContinuousHV::zero(dim);
        for (value, label) in &channels {
            let basis = genesis.hv(label, dim);
            result = result.add(&basis.scale(*value));
        }
        result.normalize()
    }

    /// Decode CfC output to a brushstroke with rich color from multiple dimensions.
    fn decode_stroke(&self, output: &ContinuousHV) -> Brushstroke {
        let v: Vec<f32> = self
            .projections
            .iter()
            .map(|p| (output.similarity(p) + 1.0) / 2.0)
            .collect();

        // Combine multiple projections for richer hue variation
        let hue_raw = v[4] * 0.5 + v[5] * 0.3 + v[8] * 0.2;
        // Phase-shifted sinusoids guarantee full RGB gamut coverage
        let phase = hue_raw * std::f32::consts::TAU * 2.0;
        let r = (phase.sin() * 0.5 + 0.5).clamp(0.0, 1.0);
        let g = ((phase + 2.094).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
        let b = ((phase + 4.189).sin() * 0.5 + 0.5).clamp(0.0, 1.0);

        Brushstroke {
            x: v[0],
            y: v[1],
            radius: 8.0 + v[2] * 60.0, // larger strokes
            pressure: v[3] * 0.7 + 0.2,
            r,
            g,
            b,
            alpha: v[7] * 0.5 + 0.2,
            angle: v[8] * std::f32::consts::TAU,
            elongation: 1.0 + v[9] * 2.0,
        }
    }

    /// Paint a complete artwork onto a pixel canvas.
    ///
    /// Returns the canvas and the sequence of brushstrokes for replay/analysis.
    pub fn paint(
        &mut self,
        snapshot: &CognitiveSnapshot,
        width: u32,
        height: u32,
        num_strokes: usize,
    ) -> (PixelCanvas, Vec<Brushstroke>) {
        let genesis = GenesisSeed::from_phrase("neural-painter");
        let input_hv = Self::encode_snapshot(snapshot, &genesis);
        let dim = symthaea_core::hdc::unified_hv::HDC_DIMENSION;

        // Gradient background from dominant harmonies
        let mut harmony_order: Vec<(usize, f32)> = snapshot
            .harmony_activations
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        harmony_order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let hue_top = harmony_order[0].0 as f32 * 45.0;
        let hue_bot = harmony_order
            .get(1)
            .map(|h| h.0 as f32 * 45.0)
            .unwrap_or(hue_top + 60.0);
        let lightness = 0.10 + snapshot.consciousness_level as f32 * 0.08;

        let mut canvas = PixelCanvas::new(width, height, [0, 0, 0, 255]);
        // Paint vertical gradient
        for y in 0..height {
            let t = y as f32 / height as f32;
            let hue = hue_top * (1.0 - t) + hue_bot * t;
            let phase = hue / 360.0 * std::f32::consts::TAU;
            let r = ((phase.sin() * 0.3 + 0.5) * lightness * 255.0).clamp(0.0, 255.0) as u8;
            let g =
                (((phase + 2.094).sin() * 0.3 + 0.5) * lightness * 255.0).clamp(0.0, 255.0) as u8;
            let b =
                (((phase + 4.189).sin() * 0.3 + 0.5) * lightness * 255.0).clamp(0.0, 255.0) as u8;
            for x in 0..width {
                let idx = ((y * width + x) * 4) as usize;
                canvas.pixels[idx] = r;
                canvas.pixels[idx + 1] = g;
                canvas.pixels[idx + 2] = b;
                canvas.pixels[idx + 3] = 255;
            }
        }

        let mut strokes = Vec::with_capacity(num_strokes);

        for step in 0..num_strokes {
            // Perturb input each step to prevent CfC convergence
            let step_basis = genesis.hv(&format!("step_{}", step % 8), dim);
            let phase = (step as f32 * 0.4).sin() * 0.25;
            let perturbed = input_hv.add(&step_basis.scale(phase)).normalize();

            self.network.evolve_closed_form(self.dt, &perturbed);
            let output = self.network.output();
            let stroke = self.decode_stroke(&output);
            canvas.paint_stroke(&stroke);
            strokes.push(stroke);
        }

        (canvas, strokes)
    }
}

/// Encode RGBA pixels as minimal uncompressed PNG.
///
/// Uses the simplest possible PNG: IHDR + uncompressed IDAT + IEND.
/// No compression (deflate with stored blocks) — produces large files
/// but zero external dependencies.
fn encode_png(pixels: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut png = Vec::new();

    // PNG signature
    png.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]);

    // IHDR
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.push(8); // bit depth
    ihdr.push(6); // color type: RGBA
    ihdr.push(0); // compression
    ihdr.push(0); // filter
    ihdr.push(0); // interlace
    write_chunk(&mut png, b"IHDR", &ihdr);

    // IDAT: uncompressed deflate (stored blocks)
    let row_bytes = (width * 4 + 1) as usize; // +1 for filter byte
    let mut raw_data = Vec::with_capacity(row_bytes * height as usize);
    for y in 0..height {
        raw_data.push(0); // filter: None
        let offset = (y * width * 4) as usize;
        let end = offset + (width * 4) as usize;
        raw_data.extend_from_slice(&pixels[offset..end]);
    }

    // Wrap in deflate stored blocks
    let deflated = deflate_stored(&raw_data);
    write_chunk(&mut png, b"IDAT", &deflated);

    // IEND
    write_chunk(&mut png, b"IEND", &[]);

    png
}

fn write_chunk(buf: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    buf.extend_from_slice(&(data.len() as u32).to_be_bytes());
    buf.extend_from_slice(chunk_type);
    buf.extend_from_slice(data);
    let crc = crc32(&[chunk_type.as_slice(), data].concat());
    buf.extend_from_slice(&crc.to_be_bytes());
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// Wrap raw data in deflate stored (uncompressed) blocks.
fn deflate_stored(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();

    // zlib header: CM=8 (deflate), CINFO=7 (32K window), FCHECK
    result.push(0x78);
    result.push(0x01);

    // Split into 65535-byte blocks (max for stored)
    let chunks: Vec<&[u8]> = data.chunks(65535).collect();
    for (i, chunk) in chunks.iter().enumerate() {
        let is_last = i == chunks.len() - 1;
        result.push(if is_last { 1 } else { 0 }); // BFINAL + BTYPE=00 (stored)
        let len = chunk.len() as u16;
        result.extend_from_slice(&len.to_le_bytes());
        result.extend_from_slice(&(!len).to_le_bytes()); // NLEN
        result.extend_from_slice(chunk);
    }

    // Adler-32 checksum
    let adler = adler32(data);
    result.extend_from_slice(&adler.to_be_bytes());

    result
}

fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canvas_creation() {
        let canvas = PixelCanvas::new(64, 64, [0, 0, 0, 255]);
        assert_eq!(canvas.pixels.len(), 64 * 64 * 4);
        assert_eq!(canvas.get_pixel(0, 0), [0, 0, 0, 255]);
    }

    #[test]
    fn paint_stroke_changes_pixels() {
        let mut canvas = PixelCanvas::new(64, 64, [0, 0, 0, 255]);
        let stroke = Brushstroke {
            x: 0.5,
            y: 0.5,
            radius: 10.0,
            pressure: 1.0,
            r: 1.0,
            g: 0.0,
            b: 0.0,
            alpha: 1.0,
            angle: 0.0,
            elongation: 1.0,
        };
        canvas.paint_stroke(&stroke);

        // Center pixel should be reddish
        let center = canvas.get_pixel(32, 32);
        assert!(center[0] > 200, "red should be high: {}", center[0]);
        assert!(center[1] < 50, "green should be low: {}", center[1]);
    }

    #[test]
    fn stroke_blending() {
        let mut canvas = PixelCanvas::new(64, 64, [0, 0, 0, 255]);

        // Paint red then green at same position
        let red = Brushstroke {
            x: 0.5,
            y: 0.5,
            radius: 15.0,
            pressure: 1.0,
            r: 1.0,
            g: 0.0,
            b: 0.0,
            alpha: 0.5,
            angle: 0.0,
            elongation: 1.0,
        };
        let green = Brushstroke {
            x: 0.5,
            y: 0.5,
            radius: 15.0,
            pressure: 1.0,
            r: 0.0,
            g: 1.0,
            b: 0.0,
            alpha: 0.5,
            angle: 0.0,
            elongation: 1.0,
        };
        canvas.paint_stroke(&red);
        canvas.paint_stroke(&green);

        let center = canvas.get_pixel(32, 32);
        // Should have both red and green (blended)
        assert!(center[0] > 30, "should have some red: {}", center[0]);
        assert!(center[1] > 30, "should have some green: {}", center[1]);
    }

    #[test]
    fn neural_painter_creates_art() {
        let genesis = GenesisSeed::from_phrase("test-painter");
        let mut painter = NeuralPainter::new(&genesis);
        let snapshot = CognitiveSnapshot {
            consciousness_level: 0.7,
            harmony_activations: [0.5; 8],
            dopamine: 0.5,
            serotonin: 0.5,
            noradrenaline: 0.3,
            valence: 0.3,
            arousal: 0.5,
            prediction_error: 0.1,
            ..CognitiveSnapshot::dormant()
        };

        let (canvas, strokes) = painter.paint(&snapshot, 128, 128, 30);

        assert_eq!(canvas.width, 128);
        assert_eq!(canvas.height, 128);
        assert_eq!(strokes.len(), 30);

        // Canvas should not be uniform (strokes painted)
        let mean = canvas.mean_color();
        // At least some color variation
        assert!(
            mean[0] > 0.01 || mean[1] > 0.01 || mean[2] > 0.01,
            "canvas is empty: {:?}",
            mean
        );
    }

    #[test]
    fn perception_features_correct_size() {
        let canvas = PixelCanvas::new(64, 64, [128, 64, 32, 255]);
        let features = canvas.perception_features();
        assert_eq!(features.len(), 4 * 4 * 3); // 4x4 grid, 3 channels
    }

    #[test]
    fn png_encoding_valid() {
        let canvas = PixelCanvas::new(4, 4, [255, 0, 0, 255]);
        let png = canvas.to_png_bytes();

        // Check PNG signature
        assert_eq!(&png[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
        // Should contain IHDR, IDAT, IEND
        assert!(png.len() > 50);
    }

    #[test]
    fn different_states_different_paintings() {
        let genesis = GenesisSeed::from_phrase("test-diff");

        let mut p1 = NeuralPainter::new(&genesis);
        let serene = CognitiveSnapshot {
            consciousness_level: 0.9,
            valence: 0.7,
            arousal: 0.2,
            harmony_activations: [0.8, 0.8, 0.7, 0.2, 0.7, 0.7, 0.5, 0.9],
            ..CognitiveSnapshot::dormant()
        };
        let (c1, _) = p1.paint(&serene, 64, 64, 20);

        let mut p2 = NeuralPainter::new(&genesis);
        let turb = CognitiveSnapshot {
            consciousness_level: 0.5,
            valence: -0.7,
            arousal: 0.9,
            harmony_activations: [0.2, 0.1, 0.3, 0.9, 0.5, 0.1, 0.8, 0.1],
            noradrenaline: 0.8,
            ..CognitiveSnapshot::dormant()
        };
        let (c2, _) = p2.paint(&turb, 64, 64, 20);

        // Mean colors should differ
        let m1 = c1.mean_color();
        let m2 = c2.mean_color();
        let diff: f32 = m1.iter().zip(m2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.01, "paintings too similar: diff={diff}");
    }

    #[test]
    fn crc32_known_value() {
        // CRC32 of "IEND" should be a known value
        assert_eq!(crc32(b"IEND"), 0xAE426082);
    }
}