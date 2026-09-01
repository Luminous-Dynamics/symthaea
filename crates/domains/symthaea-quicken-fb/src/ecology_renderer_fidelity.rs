// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Visual-fidelity wrapper for the exact Spore renderer.
//!
//! This layer deliberately stays CPU-only and deterministic. It adds two things
//! that static concept art gets almost for free but a bare framebuffer does not:
//!
//! - a low-resolution, thresholded bloom buffer for luminous depth;
//! - a subtle deterministic membrane / caustic field around focal spores.
//!
//! The pass is bounded in memory and work. The bloom buffer is quarter-resolution
//! in each dimension (1/16 the full pixel count), re-used between frames, and is
//! never authoritative for boot. If this module is removed, the holographic
//! ecology underneath remains a complete renderer.

use std::cell::RefCell;

use crate::color::{LEAF_GREEN, MYCELIAL_WHITE, Rgba, SOLAR_GOLD};
use crate::ecology_renderer_holo::EcologyRenderer as HolographicEcologyRenderer;
pub use crate::ecology_renderer_holo::EcologyFrameState;
use symthaea_boot_ecology::{BootCue, BootGenome, BootStageKind, MorphologyFamily};

const HOLO_CYAN: Rgba = Rgba(0x58, 0xd8, 0xd2, 0xff);
const BLOOM_DOWNSAMPLE: usize = 4;
const BLOOM_THRESHOLD: f32 = 118.0;

#[derive(Debug, Clone, Copy)]
struct Point {
    x: f32,
    y: f32,
}

#[derive(Debug)]
struct FidelityField {
    width: usize,
    height: usize,
    center: Point,
    family: MorphologyFamily,
    cue: BootCue,
    shell_phase: f32,
    shell_radius: f32,
    shell_gain: f32,
    bloom_gain: f32,
    cell_offsets: [f32; 16],
}

/// The public renderer used by live DRM and exact preview paths.
///
/// Rendering order is intentionally explicit:
/// organic topology -> holographic field -> membrane/caustics -> bloom.
pub struct EcologyRenderer {
    inner: HolographicEcologyRenderer,
    fidelity: FidelityField,
    bloom: RefCell<BloomWorkspace>,
}

impl EcologyRenderer {
    pub fn new(width: u32, height: u32, genome: BootGenome) -> Self {
        let fidelity = FidelityField::new(width, height, &genome);
        let bloom = RefCell::new(BloomWorkspace::new(width as usize, height as usize));
        let inner = HolographicEcologyRenderer::new(width, height, genome);
        Self {
            inner,
            fidelity,
            bloom,
        }
    }

    pub fn genome(&self) -> &BootGenome {
        self.inner.genome()
    }

    pub fn total_duration_ms(&self) -> u32 {
        self.inner.total_duration_ms()
    }

    pub fn frame_state(&self, elapsed_ms: u32) -> EcologyFrameState {
        self.inner.frame_state(elapsed_ms)
    }

    pub fn render_at(&self, elapsed_ms: u32, buffer: &mut [u32]) -> EcologyFrameState {
        let state = self.inner.render_at(elapsed_ms, buffer);
        if state.stage == BootStageKind::Blackout {
            return state;
        }

        self.fidelity.render(buffer, state);
        let bloom_strength = self.fidelity.bloom_strength(state);
        self.bloom.borrow_mut().apply(
            buffer,
            self.fidelity.width,
            self.fidelity.height,
            bloom_strength,
        );
        state
    }
}

impl FidelityField {
    fn new(width: u32, height: u32, genome: &BootGenome) -> Self {
        let width = width as usize;
        let height = height as usize;
        let min_dim = width.min(height) as f32;
        let selector = genome.seed[6] as f32 / 255.0;
        let mut cell_offsets = [0.0; 16];
        for (index, value) in cell_offsets.iter_mut().enumerate() {
            *value = genome.seed[(index + 8) % 32] as f32 / 255.0;
        }

        let family_gain = match genome.family {
            MorphologyFamily::CentralSpore | MorphologyFamily::HdcOrganic => 1.0,
            MorphologyFamily::KintsugiRepair | MorphologyFamily::AnastomoticWeb => 0.88,
            MorphologyFamily::CrystalThaw | MorphologyFamily::MinimalRelight => 0.76,
            MorphologyFamily::FairyRing => 0.58,
            _ => 0.48,
        };
        let cue_gain = match genome.cue {
            BootCue::FirstBoot => 1.0,
            BootCue::RestoringKnownGood | BootCue::RecoveringState => 0.84,
            BootCue::ApplyingGeneration => 0.78,
            BootCue::Resuming => 0.66,
            BootCue::Starting => 0.62,
        };

        Self {
            width,
            height,
            center: Point {
                x: width as f32 * 0.5,
                y: height as f32 * 0.48,
            },
            family: genome.family,
            cue: genome.cue,
            shell_phase: selector * std::f32::consts::TAU,
            shell_radius: min_dim * (0.052 + selector * 0.018),
            shell_gain: family_gain * cue_gain,
            bloom_gain: 0.72 + genome.seed[29] as f32 / 255.0 * 0.28,
            cell_offsets,
        }
    }

    fn bloom_strength(&self, state: EcologyFrameState) -> f32 {
        let stage = match state.stage {
            BootStageKind::Repair | BootStageKind::GrowthRing => 0.38,
            BootStageKind::Relight | BootStageKind::MeshLink => 0.34,
            BootStageKind::Germinate | BootStageKind::Grow => 0.31,
            BootStageKind::Settle => 0.28,
            BootStageKind::Handoff => 0.22 * (1.0 - smoothstep(state.stage_progress)),
            _ => 0.25,
        };
        (stage * self.bloom_gain).clamp(0.0, 0.42)
    }

    fn render(&self, buffer: &mut [u32], state: EcologyFrameState) {
        if self.width == 0 || self.height == 0 || buffer.len() < self.width * self.height {
            return;
        }
        let arrival = smoothstep((state.sequence_progress * 4.0).clamp(0.0, 1.0));
        let handoff = if state.stage == BootStageKind::Handoff {
            1.0 - smoothstep(state.stage_progress)
        } else {
            1.0
        };
        let gain = self.shell_gain * arrival * handoff;
        if gain <= 0.002 {
            return;
        }

        self.draw_membrane_shell(buffer, state, gain);
        self.draw_caustic_field(buffer, state, gain);
    }

    fn draw_membrane_shell(&self, buffer: &mut [u32], state: EcologyFrameState, gain: f32) {
        let time = state.sequence_progress * std::f32::consts::TAU * 1.35 + self.shell_phase;
        let breathe = 1.0 + 0.045 * time.sin();
        let radius = self.shell_radius * breathe;
        let stage_color = match state.stage {
            BootStageKind::Repair | BootStageKind::GrowthRing | BootStageKind::RetractFailedGrowth => {
                Rgba::lerp(SOLAR_GOLD, MYCELIAL_WHITE, 0.34)
            }
            BootStageKind::Relight | BootStageKind::MeshLink => {
                Rgba::lerp(HOLO_CYAN, MYCELIAL_WHITE, 0.30)
            }
            _ => Rgba::lerp(MYCELIAL_WHITE, LEAF_GREEN, 0.28),
        };

        // Outer membrane. Two displaced ellipses read as a translucent shell
        // instead of a flat ring, especially after the bloom pass.
        draw_ellipse(
            buffer,
            self.width,
            self.height,
            self.center,
            radius,
            radius * 0.86,
            time * 0.035,
            stage_color.with_opacity((0.15 * gain).clamp(0.0, 0.18)),
        );
        draw_ellipse(
            buffer,
            self.width,
            self.height,
            Point {
                x: self.center.x + 1.2,
                y: self.center.y - 0.8,
            },
            radius * 0.965,
            radius * 0.82,
            -time * 0.028,
            HOLO_CYAN.with_opacity((0.07 * gain).clamp(0.0, 0.09)),
        );

        // Seeded shell cells. The topology is a bounded geodesic-like mesh,
        // giving the central spore a recognisable membrane without a texture or
        // image asset. The same genome always reconstructs the same shell.
        let mut nodes = [Point { x: 0.0, y: 0.0 }; 16];
        for (index, node) in nodes.iter_mut().enumerate() {
            let base = std::f32::consts::TAU * index as f32 / nodes.len() as f32;
            let jitter = (self.cell_offsets[index] - 0.5) * 0.24;
            let angle = base + jitter + time * 0.012;
            let radial = radius * (0.72 + self.cell_offsets[(index + 5) % 16] * 0.20);
            *node = Point {
                x: self.center.x + angle.cos() * radial,
                y: self.center.y + angle.sin() * radial * 0.82,
            };
        }

        let mesh_color = stage_color.with_opacity((0.075 * gain).clamp(0.0, 0.10));
        for index in 0..nodes.len() {
            draw_line_alpha(
                buffer,
                self.width,
                self.height,
                nodes[index],
                nodes[(index + 1) % nodes.len()],
                mesh_color,
            );
            if index % 2 == 0 {
                draw_line_alpha(
                    buffer,
                    self.width,
                    self.height,
                    nodes[index],
                    nodes[(index + 5) % nodes.len()],
                    mesh_color.with_opacity((0.045 * gain).clamp(0.0, 0.07)),
                );
            }
        }

        // A small inner light suggests depth beneath the membrane rather than a
        // solid logo disc. It is strongest during germination and repair.
        let inner_gain = match state.stage {
            BootStageKind::Germinate | BootStageKind::Repair | BootStageKind::Relight => 1.0,
            _ => 0.72,
        };
        draw_glow_disc(
            buffer,
            self.width,
            self.height,
            self.center,
            radius * 0.30,
            Rgba::lerp(LEAF_GREEN, MYCELIAL_WHITE, 0.58)
                .with_opacity((0.12 * gain * inner_gain).clamp(0.0, 0.16)),
        );
    }

    fn draw_caustic_field(&self, buffer: &mut [u32], state: EcologyFrameState, gain: f32) {
        // Three slowly drifting interference arcs are enough to make the center
        // feel volumetric after bloom. They are intentionally not particles and
        // cannot grow without bound.
        let min_dim = self.width.min(self.height) as f32;
        let phase = state.sequence_progress * std::f32::consts::TAU + self.shell_phase;
        let base_alpha = match self.cue {
            BootCue::FirstBoot => 0.050,
            BootCue::Resuming => 0.032,
            _ => 0.040,
        };
        for index in 0..3 {
            let local = phase + index as f32 * 2.07;
            let center = Point {
                x: self.center.x + local.cos() * min_dim * 0.020,
                y: self.center.y + local.sin() * min_dim * 0.012,
            };
            let radius = min_dim * (0.105 + index as f32 * 0.041 + local.sin() * 0.006);
            let color = if index == 1 {
                HOLO_CYAN
            } else if matches!(state.stage, BootStageKind::Repair | BootStageKind::GrowthRing) {
                SOLAR_GOLD
            } else {
                MYCELIAL_WHITE
            }
            .with_opacity((base_alpha * gain).clamp(0.0, 0.065));
            draw_arc(
                buffer,
                self.width,
                self.height,
                center,
                radius,
                radius * (0.42 + index as f32 * 0.08),
                local * 0.11,
                0.18 + index as f32 * 0.22,
                2.7 + index as f32 * 0.35,
                color,
            );
        }
    }
}

#[derive(Debug)]
struct BloomWorkspace {
    low_width: usize,
    low_height: usize,
    source: Vec<[f32; 3]>,
    scratch: Vec<[f32; 3]>,
}

impl BloomWorkspace {
    fn new(width: usize, height: usize) -> Self {
        let low_width = width.div_ceil(BLOOM_DOWNSAMPLE).max(1);
        let low_height = height.div_ceil(BLOOM_DOWNSAMPLE).max(1);
        let len = low_width.saturating_mul(low_height);
        Self {
            low_width,
            low_height,
            source: vec![[0.0; 3]; len],
            scratch: vec![[0.0; 3]; len],
        }
    }

    fn apply(&mut self, buffer: &mut [u32], width: usize, height: usize, strength: f32) {
        if strength <= 0.001 || width == 0 || height == 0 || buffer.len() < width * height {
            return;
        }
        self.extract(buffer, width, height);
        self.blur_horizontal();
        self.blur_vertical();
        self.composite(buffer, width, height, strength);
    }

    fn extract(&mut self, buffer: &[u32], width: usize, height: usize) {
        self.source.fill([0.0; 3]);
        for low_y in 0..self.low_height {
            for low_x in 0..self.low_width {
                let start_x = low_x * BLOOM_DOWNSAMPLE;
                let start_y = low_y * BLOOM_DOWNSAMPLE;
                let end_x = (start_x + BLOOM_DOWNSAMPLE).min(width);
                let end_y = (start_y + BLOOM_DOWNSAMPLE).min(height);
                let mut accum = [0.0f32; 3];
                let mut samples = 0usize;
                for y in start_y..end_y {
                    for x in start_x..end_x {
                        let value = buffer[y * width + x];
                        let rgb = [
                            ((value >> 16) & 0xff) as f32,
                            ((value >> 8) & 0xff) as f32,
                            (value & 0xff) as f32,
                        ];
                        let peak = rgb[0].max(rgb[1]).max(rgb[2]);
                        if peak > BLOOM_THRESHOLD {
                            let knee = ((peak - BLOOM_THRESHOLD) / (255.0 - BLOOM_THRESHOLD))
                                .clamp(0.0, 1.0);
                            accum[0] += rgb[0] * knee;
                            accum[1] += rgb[1] * knee;
                            accum[2] += rgb[2] * knee;
                        }
                        samples += 1;
                    }
                }
                if samples > 0 {
                    let inv = 1.0 / samples as f32;
                    self.source[low_y * self.low_width + low_x] = [
                        accum[0] * inv,
                        accum[1] * inv,
                        accum[2] * inv,
                    ];
                }
            }
        }
    }

    fn blur_horizontal(&mut self) {
        const WEIGHTS: [f32; 5] = [1.0, 4.0, 6.0, 4.0, 1.0];
        const NORMALIZER: f32 = 16.0;
        self.scratch.fill([0.0; 3]);
        for y in 0..self.low_height {
            for x in 0..self.low_width {
                let mut out = [0.0f32; 3];
                for (tap, weight) in WEIGHTS.iter().copied().enumerate() {
                    let offset = tap as isize - 2;
                    let sx = (x as isize + offset).clamp(0, self.low_width as isize - 1) as usize;
                    let sample = self.source[y * self.low_width + sx];
                    out[0] += sample[0] * weight;
                    out[1] += sample[1] * weight;
                    out[2] += sample[2] * weight;
                }
                self.scratch[y * self.low_width + x] = [
                    out[0] / NORMALIZER,
                    out[1] / NORMALIZER,
                    out[2] / NORMALIZER,
                ];
            }
        }
    }

    fn blur_vertical(&mut self) {
        const WEIGHTS: [f32; 5] = [1.0, 4.0, 6.0, 4.0, 1.0];
        const NORMALIZER: f32 = 16.0;
        self.source.fill([0.0; 3]);
        for y in 0..self.low_height {
            for x in 0..self.low_width {
                let mut out = [0.0f32; 3];
                for (tap, weight) in WEIGHTS.iter().copied().enumerate() {
                    let offset = tap as isize - 2;
                    let sy = (y as isize + offset).clamp(0, self.low_height as isize - 1) as usize;
                    let sample = self.scratch[sy * self.low_width + x];
                    out[0] += sample[0] * weight;
                    out[1] += sample[1] * weight;
                    out[2] += sample[2] * weight;
                }
                self.source[y * self.low_width + x] = [
                    out[0] / NORMALIZER,
                    out[1] / NORMALIZER,
                    out[2] / NORMALIZER,
                ];
            }
        }
    }

    fn composite(&self, buffer: &mut [u32], width: usize, height: usize, strength: f32) {
        for y in 0..height {
            let low_y = (y / BLOOM_DOWNSAMPLE).min(self.low_height - 1);
            for x in 0..width {
                let low_x = (x / BLOOM_DOWNSAMPLE).min(self.low_width - 1);
                let glow = self.source[low_y * self.low_width + low_x];
                let index = y * width + x;
                let value = buffer[index];
                let dst = [
                    ((value >> 16) & 0xff) as f32,
                    ((value >> 8) & 0xff) as f32,
                    (value & 0xff) as f32,
                ];
                let r = screen_add(dst[0], glow[0] * strength);
                let g = screen_add(dst[1], glow[1] * strength);
                let b = screen_add(dst[2], glow[2] * strength);
                buffer[index] = ((r as u32) << 16) | ((g as u32) << 8) | b as u32;
            }
        }
    }
}

fn screen_add(dst: f32, glow: f32) -> u8 {
    let dst = dst.clamp(0.0, 255.0);
    let glow = glow.clamp(0.0, 255.0);
    let out = 255.0 - (255.0 - dst) * (1.0 - glow / 255.0);
    out.round().clamp(0.0, 255.0) as u8
}

fn draw_ellipse(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    center: Point,
    radius_x: f32,
    radius_y: f32,
    rotation: f32,
    color: Rgba,
) {
    let segments = ((radius_x.max(radius_y) * 2.1) as usize).clamp(48, 280);
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
    let mut previous = point_at(0);
    for index in 1..=segments {
        let current = point_at(index);
        draw_line_alpha(buffer, width, height, previous, current, color);
        previous = current;
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_arc(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    center: Point,
    radius_x: f32,
    radius_y: f32,
    rotation: f32,
    start_turn: f32,
    span: f32,
    color: Rgba,
) {
    let segments = 90usize;
    let cos_r = rotation.cos();
    let sin_r = rotation.sin();
    let point_at = |t: f32| {
        let angle = start_turn + span * t;
        let ex = angle.cos() * radius_x;
        let ey = angle.sin() * radius_y;
        Point {
            x: center.x + ex * cos_r - ey * sin_r,
            y: center.y + ex * sin_r + ey * cos_r,
        }
    };
    let mut previous = point_at(0.0);
    for index in 1..=segments {
        let current = point_at(index as f32 / segments as f32);
        draw_line_alpha(buffer, width, height, previous, current, color);
        previous = current;
    }
}

fn draw_glow_disc(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    center: Point,
    radius: f32,
    color: Rgba,
) {
    let outer = radius.ceil().clamp(1.0, 24.0) as i32;
    let cx = center.x.round() as i32;
    let cy = center.y.round() as i32;
    for oy in -outer..=outer {
        for ox in -outer..=outer {
            let distance = ((ox * ox + oy * oy) as f32).sqrt();
            if distance > outer as f32 {
                continue;
            }
            let x = cx + ox;
            let y = cy + oy;
            if x < 0 || y < 0 || x as usize >= width || y as usize >= height {
                continue;
            }
            let falloff = (1.0 - distance / outer.max(1) as f32).powf(1.8);
            blend_pixel(
                buffer,
                width,
                x as usize,
                y as usize,
                color.with_opacity((color.3 as f32 / 255.0 * falloff).clamp(0.0, 1.0)),
            );
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

    fn genome() -> BootGenome {
        BootEcologyComposer::compose(
            &BootStateReceipt::first_boot([0x74; 32]),
            &MorphologyLineage::default(),
        )
    }

    #[test]
    fn fidelity_renderer_is_deterministic() {
        let a = EcologyRenderer::new(320, 180, genome());
        let b = EcologyRenderer::new(320, 180, genome());
        let mut fa = vec![0u32; 320 * 180];
        let mut fb = vec![0u32; 320 * 180];
        a.render_at(2_100, &mut fa);
        b.render_at(2_100, &mut fb);
        assert_eq!(fa, fb);
    }

    #[test]
    fn bloom_is_bounded_and_brightens_without_darkening() {
        let mut workspace = BloomWorkspace::new(64, 36);
        let mut frame = vec![0x00101010u32; 64 * 36];
        frame[18 * 64 + 32] = 0x00f0f0f0;
        let before = frame.clone();
        workspace.apply(&mut frame, 64, 36, 0.35);
        assert!(frame.iter().zip(before.iter()).all(|(after, before)| {
            ((after >> 16) & 0xff) >= ((before >> 16) & 0xff)
                && ((after >> 8) & 0xff) >= ((before >> 8) & 0xff)
                && (after & 0xff) >= (before & 0xff)
        }));
        assert_ne!(frame, before);
    }

    #[test]
    fn workspace_is_sixteen_times_smaller_than_full_frame_in_pixel_count() {
        let workspace = BloomWorkspace::new(1920, 1080);
        let full = 1920usize * 1080usize;
        assert!(workspace.source.len() <= full.div_ceil(16));
    }

    #[test]
    fn membrane_changes_holographic_frame() {
        let genome = genome();
        let holo = HolographicEcologyRenderer::new(320, 180, genome.clone());
        let fidelity = EcologyRenderer::new(320, 180, genome);
        let mut a = vec![0u32; 320 * 180];
        let mut b = vec![0u32; 320 * 180];
        holo.render_at(2_000, &mut a);
        fidelity.render_at(2_000, &mut b);
        assert_ne!(a, b);
    }
}
