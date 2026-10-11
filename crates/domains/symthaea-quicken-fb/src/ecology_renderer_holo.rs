// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Holographic presentation layer for the exact Spore ecology renderer.
//!
//! The underlying mycelial topology remains authoritative for visual identity.
//! This module adds a deterministic, bounded spatial field around it: projected
//! membranes, segmented orbital rings, parallax anchors, spectral echoes, and a
//! very subtle scanline sheen. The goal is depth and presence rather than a
//! generic rectangular sci-fi HUD.

use crate::color::{LEAF_GREEN, MYCELIAL_WHITE, Rgba, SOLAR_GOLD};
use crate::ecology_renderer_base::EcologyRenderer as BaseEcologyRenderer;
pub use crate::ecology_renderer_base::EcologyFrameState;
use symthaea_boot_ecology::{BootGenome, BootStageKind, MorphologyFamily};

const HOLO_CYAN: Rgba = Rgba(0x58, 0xd8, 0xd2, 0xff);
const MAX_RINGS: usize = 7;
const MAX_ANCHORS: usize = 18;

#[derive(Debug, Clone, Copy)]
struct Point {
    x: f32,
    y: f32,
}

#[derive(Debug, Clone, Copy)]
struct HoloRing {
    radius: f32,
    flattening: f32,
    rotation: f32,
    phase: f32,
    dash_period: usize,
}

#[derive(Debug, Clone, Copy)]
struct HoloAnchor {
    position: Point,
    phase: f32,
    scale: f32,
}

#[derive(Debug)]
struct HolographicField {
    width: usize,
    height: usize,
    center: Point,
    rings: Vec<HoloRing>,
    anchors: Vec<HoloAnchor>,
    family: MorphologyFamily,
    spectral_bias: f32,
    field_strength: f32,
}

/// Exact ecology renderer plus deterministic holographic depth effects.
pub struct EcologyRenderer {
    base: BaseEcologyRenderer,
    field: HolographicField,
}

impl EcologyRenderer {
    pub fn new(width: u32, height: u32, genome: BootGenome) -> Self {
        let field = HolographicField::new(width, height, &genome);
        let base = BaseEcologyRenderer::new(width, height, genome);
        Self { base, field }
    }

    pub fn genome(&self) -> &BootGenome {
        self.base.genome()
    }

    pub fn total_duration_ms(&self) -> u32 {
        self.base.total_duration_ms()
    }

    pub fn frame_state(&self, elapsed_ms: u32) -> EcologyFrameState {
        self.base.frame_state(elapsed_ms)
    }

    /// Render the organic ecology first, then project the holographic field.
    pub fn render_at(&self, elapsed_ms: u32, buffer: &mut [u32]) -> EcologyFrameState {
        let state = self.base.render_at(elapsed_ms, buffer);
        if state.stage != BootStageKind::Blackout {
            self.field.render(buffer, state);
        }
        state
    }
}

impl HolographicField {
    fn new(width: u32, height: u32, genome: &BootGenome) -> Self {
        let width = width as usize;
        let height = height as usize;
        let min_dim = width.min(height) as f32;
        let center = Point {
            x: width as f32 * 0.5,
            y: height as f32 * 0.48,
        };

        let ring_count = (4 + (genome.seed[18] as usize % 4)).min(MAX_RINGS);
        let mut rings = Vec::with_capacity(ring_count);
        for index in 0..ring_count {
            let selector = genome.seed[(19 + index) % 32] as f32 / 255.0;
            rings.push(HoloRing {
                radius: min_dim * (0.13 + index as f32 * 0.055 + selector * 0.018),
                flattening: 0.42 + selector * 0.30,
                rotation: (selector - 0.5) * 0.42,
                phase: selector * std::f32::consts::TAU,
                dash_period: 5 + genome.seed[(7 + index) % 32] as usize % 6,
            });
        }

        let anchor_count = (10 + (genome.seed[27] as usize % 9)).min(MAX_ANCHORS);
        let mut anchors = Vec::with_capacity(anchor_count);
        for index in 0..anchor_count {
            let a = genome.seed[index % 32] as f32 / 255.0;
            let b = genome.seed[(index + 11) % 32] as f32 / 255.0;
            let angle = std::f32::consts::TAU * index as f32 / anchor_count as f32
                + (a - 0.5) * 0.20;
            let radius = min_dim * (0.30 + b * 0.16);
            anchors.push(HoloAnchor {
                position: Point {
                    x: center.x + angle.cos() * radius,
                    y: center.y + angle.sin() * radius * 0.62,
                },
                phase: a * std::f32::consts::TAU,
                scale: 0.65 + b * 0.75,
            });
        }

        let family_strength = match genome.family {
            MorphologyFamily::HdcOrganic | MorphologyFamily::ConstellationHyphae => 1.0,
            MorphologyFamily::CrystalThaw | MorphologyFamily::MinimalRelight => 0.82,
            MorphologyFamily::KintsugiRepair => 0.72,
            _ => 0.88,
        };

        Self {
            width,
            height,
            center,
            rings,
            anchors,
            family: genome.family,
            spectral_bias: genome.seed[30] as f32 / 255.0,
            field_strength: family_strength,
        }
    }

    fn render(&self, buffer: &mut [u32], state: EcologyFrameState) {
        if self.width == 0 || self.height == 0 || buffer.len() < self.width * self.height {
            return;
        }

        let handoff = if state.stage == BootStageKind::Handoff {
            1.0 - smoothstep(state.stage_progress)
        } else {
            1.0
        };
        let arrival = smoothstep((state.sequence_progress * 4.5).clamp(0.0, 1.0));
        let gain = self.field_strength * handoff * arrival;
        if gain <= 0.001 {
            return;
        }

        let phase = state.sequence_progress * std::f32::consts::TAU * 1.8;
        let primary = stage_color(state.stage, self.spectral_bias);
        let cyan = HOLO_CYAN.with_opacity((0.16 * gain).clamp(0.0, 1.0));

        // Nested projected membranes give the mycelium a spatial volume without
        // introducing heavyweight GPU dependencies into early boot.
        for (index, ring) in self.rings.iter().enumerate() {
            let breathe = 1.0 + 0.018 * (phase + ring.phase).sin();
            let radius = ring.radius * breathe;
            let alpha = (0.10 + index as f32 * 0.012) * gain;
            let color = Rgba::lerp(primary, HOLO_CYAN, index as f32 / self.rings.len() as f32)
                .with_opacity(alpha.clamp(0.0, 0.22));
            draw_segmented_ellipse(
                buffer,
                self.width,
                self.height,
                self.center,
                radius,
                radius * ring.flattening,
                ring.rotation + phase * 0.035,
                ring.dash_period,
                color,
            );

            // One-pixel cyan spectral echo creates a restrained holographic
            // diffraction/parallax cue around bright structural rings.
            let echo_center = Point {
                x: self.center.x + 1.25,
                y: self.center.y - 0.75,
            };
            draw_segmented_ellipse(
                buffer,
                self.width,
                self.height,
                echo_center,
                radius * 1.004,
                radius * ring.flattening * 1.004,
                ring.rotation + phase * 0.035,
                ring.dash_period + 2,
                cyan,
            );
        }

        self.draw_anchor_field(buffer, state, gain, primary);
        self.draw_energy_sweep(buffer, state, gain, primary);
        self.draw_scanline_sheen(buffer, phase, gain);
    }

    fn draw_anchor_field(
        &self,
        buffer: &mut [u32],
        state: EcologyFrameState,
        gain: f32,
        primary: Rgba,
    ) {
        let pulse_phase = state.sequence_progress * 18.0;
        for (index, anchor) in self.anchors.iter().enumerate() {
            let pulse = 0.5 + 0.5 * (pulse_phase + anchor.phase).sin();
            let alpha = (0.07 + pulse * 0.12) * gain;
            let node_color = Rgba::lerp(HOLO_CYAN, primary, pulse)
                .with_opacity(alpha.clamp(0.0, 0.24));
            let radius = (1.2 + anchor.scale * 1.6 + pulse).clamp(1.0, 4.0);
            draw_glow_point(
                buffer,
                self.width,
                self.height,
                anchor.position,
                radius,
                node_color,
            );

            // Sparse, curved-feeling field chords connect the projected shell
            // to itself rather than drawing rectangular HUD panels.
            if index % 3 == 0 && self.anchors.len() > 3 {
                let peer = self.anchors[(index + 3) % self.anchors.len()].position;
                let chord = Rgba::lerp(HOLO_CYAN, LEAF_GREEN, 0.52)
                    .with_opacity((0.025 + pulse * 0.035) * gain);
                draw_line_alpha(
                    buffer,
                    self.width,
                    self.height,
                    anchor.position,
                    peer,
                    chord,
                );
            }
        }
    }

    fn draw_energy_sweep(
        &self,
        buffer: &mut [u32],
        state: EcologyFrameState,
        gain: f32,
        primary: Rgba,
    ) {
        let min_dim = self.width.min(self.height) as f32;
        let cycle = (state.sequence_progress * 2.25).fract();
        let radius = min_dim * (0.10 + smoothstep(cycle) * 0.38);
        let fade = (1.0 - cycle).powf(1.7);
        let stage_boost = match state.stage {
            BootStageKind::GrowthRing | BootStageKind::MeshLink | BootStageKind::Relight => 1.45,
            BootStageKind::Repair => 1.30,
            _ => 1.0,
        };
        let color = Rgba::lerp(primary, MYCELIAL_WHITE, 0.34)
            .with_opacity((0.11 * fade * gain * stage_boost).clamp(0.0, 0.28));
        draw_segmented_ellipse(
            buffer,
            self.width,
            self.height,
            self.center,
            radius,
            radius * 0.58,
            -0.10,
            4,
            color,
        );
    }

    fn draw_scanline_sheen(&self, buffer: &mut [u32], phase: f32, gain: f32) {
        // Extremely low-opacity and sparse: just enough to make the field feel
        // projected on high-DPI panels without becoming a retro CRT effect.
        let stride = 9usize;
        let offset = ((phase.sin() * 2.0 + 2.0) as usize) % stride;
        let color = match self.family {
            MorphologyFamily::CrystalThaw | MorphologyFamily::MinimalRelight => HOLO_CYAN,
            _ => Rgba::lerp(HOLO_CYAN, LEAF_GREEN, 0.42),
        }
        .with_opacity((0.016 * gain).clamp(0.0, 0.025));

        for y in (offset..self.height).step_by(stride) {
            let start = Point { x: 0.0, y: y as f32 };
            let end = Point {
                x: self.width.saturating_sub(1) as f32,
                y: y as f32,
            };
            draw_line_alpha(buffer, self.width, self.height, start, end, color);
        }
    }
}

fn stage_color(stage: BootStageKind, spectral_bias: f32) -> Rgba {
    match stage {
        BootStageKind::Repair | BootStageKind::GrowthRing | BootStageKind::RetractFailedGrowth => {
            Rgba::lerp(SOLAR_GOLD, MYCELIAL_WHITE, 0.18 + spectral_bias * 0.18)
        }
        BootStageKind::Relight | BootStageKind::MeshLink => {
            Rgba::lerp(HOLO_CYAN, MYCELIAL_WHITE, 0.20 + spectral_bias * 0.20)
        }
        _ => Rgba::lerp(LEAF_GREEN, HOLO_CYAN, 0.34 + spectral_bias * 0.22),
    }
}

fn draw_segmented_ellipse(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    center: Point,
    radius_x: f32,
    radius_y: f32,
    rotation: f32,
    dash_period: usize,
    color: Rgba,
) {
    let segments = ((radius_x.max(radius_y) * 1.7) as usize).clamp(72, 420);
    let cos_r = rotation.cos();
    let sin_r = rotation.sin();
    let point_at = |index: usize| {
        let angle = std::f32::consts::TAU * index as f32 / segments as f32;
        let ex = angle.cos() * radius_x;
        let ey = angle.sin() * radius_y;
        Point {
            x: center.x + ex * cos_r - ey * sin_r,
            y: center.y + ex * sin_r + ey * cos_r,
        }
    };

    let period = dash_period.max(3);
    let mut previous = point_at(0);
    for index in 1..=segments {
        let current = point_at(index);
        if (index / period) % 3 != 1 {
            draw_line_alpha(buffer, width, height, previous, current, color);
        }
        previous = current;
    }
}

fn draw_glow_point(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    center: Point,
    radius: f32,
    color: Rgba,
) {
    let outer = radius.ceil() as i32 + 2;
    let cx = center.x.round() as i32;
    let cy = center.y.round() as i32;
    for oy in -outer..=outer {
        for ox in -outer..=outer {
            let px = cx + ox;
            let py = cy + oy;
            if px < 0 || py < 0 || px as usize >= width || py as usize >= height {
                continue;
            }
            let distance = ((ox * ox + oy * oy) as f32).sqrt();
            if distance > outer as f32 {
                continue;
            }
            let falloff = (1.0 - distance / outer.max(1) as f32).powf(1.6);
            let src = color.with_opacity((color.3 as f32 / 255.0 * falloff).clamp(0.0, 1.0));
            blend_pixel(buffer, width, px as usize, py as usize, src);
        }
    }
}

fn draw_line_alpha(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    a: Point,
    b: Point,
    color: Rgba,
) {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let distance = (dx * dx + dy * dy).sqrt().max(1.0);
    let steps = distance.ceil() as usize;
    for index in 0..=steps {
        let t = index as f32 / steps.max(1) as f32;
        let x = (a.x + dx * t).round() as i32;
        let y = (a.y + dy * t).round() as i32;
        if x < 0 || y < 0 || x as usize >= width || y as usize >= height {
            continue;
        }
        blend_pixel(buffer, width, x as usize, y as usize, color);
    }
}

fn blend_pixel(buffer: &mut [u32], width: usize, x: usize, y: usize, src: Rgba) {
    let index = y * width + x;
    let value = buffer[index];
    let dst = Rgba(
        ((value >> 16) & 0xff) as u8,
        ((value >> 8) & 0xff) as u8,
        (value & 0xff) as u8,
        0xff,
    );
    buffer[index] = src.over(dst).to_xrgb8888();
}

fn smoothstep(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    value * value * (3.0 - 2.0 * value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_boot_ecology::{BootEcologyComposer, BootStateReceipt, MorphologyLineage};

    #[test]
    fn holographic_wrapper_is_deterministic() {
        let receipt = BootStateReceipt::first_boot([0x55; 32]);
        let genome = BootEcologyComposer::compose(&receipt, &MorphologyLineage::default());
        let a = EcologyRenderer::new(320, 180, genome.clone());
        let b = EcologyRenderer::new(320, 180, genome);
        let mut fa = vec![0u32; 320 * 180];
        let mut fb = vec![0u32; 320 * 180];
        a.render_at(2_100, &mut fa);
        b.render_at(2_100, &mut fb);
        assert_eq!(fa, fb);
    }

    #[test]
    fn holographic_layer_changes_non_blackout_frame() {
        let receipt = BootStateReceipt::first_boot([0x33; 32]);
        let genome = BootEcologyComposer::compose(&receipt, &MorphologyLineage::default());
        let base = BaseEcologyRenderer::new(320, 180, genome.clone());
        let holo = EcologyRenderer::new(320, 180, genome);
        let mut base_frame = vec![0u32; 320 * 180];
        let mut holo_frame = vec![0u32; 320 * 180];
        base.render_at(2_000, &mut base_frame);
        holo.render_at(2_000, &mut holo_frame);
        assert_ne!(base_frame, holo_frame);
    }
}
