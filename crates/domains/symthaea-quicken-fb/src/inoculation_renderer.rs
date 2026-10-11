// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Distinct installation/inoculation visual grammar.
//!
//! Installation should feel related to Spore boot ecology without looking like
//! an ordinary boot with different text. The inoculation renderer treats the
//! system as a substrate being prepared inside a projected incubation chamber:
//! orbital seals, vertical field lines, module seeds, and a progress halo are
//! layered over the same deterministic organic/holographic fidelity stack used
//! by first boot — but without the boot-specific Germination identity overlay.

use crate::color::{LEAF_GREEN, MYCELIAL_WHITE, Rgba, SOLAR_GOLD};
use crate::ecology_renderer_fidelity_v2::{EcologyFrameState, EcologyRenderer};
use crate::microtype;
use symthaea_boot_ecology::BootGenome;

const HOLO_CYAN: Rgba = Rgba(0x58, 0xd8, 0xd2, 0xff);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InoculationPhase {
    Attestation,
    PreparingSubstrate,
    WeavingSystem,
    SeedingSecurity,
    OpeningChannels,
    Personalizing,
    Finalizing,
    Complete,
}

impl InoculationPhase {
    pub const ALL: [Self; 8] = [
        Self::Attestation,
        Self::PreparingSubstrate,
        Self::WeavingSystem,
        Self::SeedingSecurity,
        Self::OpeningChannels,
        Self::Personalizing,
        Self::Finalizing,
        Self::Complete,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Attestation => "attestation",
            Self::PreparingSubstrate => "preparing-substrate",
            Self::WeavingSystem => "weaving-system",
            Self::SeedingSecurity => "seeding-security",
            Self::OpeningChannels => "opening-channels",
            Self::Personalizing => "personalizing",
            Self::Finalizing => "finalizing",
            Self::Complete => "complete",
        }
    }

    pub fn display_label(self) -> &'static str {
        match self {
            Self::Attestation => "ATTESTATION",
            Self::PreparingSubstrate => "PREPARING SUBSTRATE",
            Self::WeavingSystem => "WEAVING SYSTEM",
            Self::SeedingSecurity => "SEEDING SECURITY",
            Self::OpeningChannels => "OPENING CHANNELS",
            Self::Personalizing => "PERSONALIZING",
            Self::Finalizing => "FINALIZING",
            Self::Complete => "INSTALLATION COMPLETE",
        }
    }
}

pub struct InoculationRenderer {
    width: usize,
    height: usize,
    ecology: EcologyRenderer,
    phase_offsets: [f32; 8],
}

impl InoculationRenderer {
    pub fn new(width: u32, height: u32, genome: BootGenome) -> Self {
        let mut phase_offsets = [0.0; 8];
        for (index, slot) in phase_offsets.iter_mut().enumerate() {
            *slot = genome.seed[(index + 9) % genome.seed.len()] as f32 / 255.0;
        }
        Self {
            width: width as usize,
            height: height as usize,
            ecology: EcologyRenderer::new(width, height, genome),
            phase_offsets,
        }
    }

    pub fn render(
        &self,
        phase: InoculationPhase,
        progress: f32,
        elapsed_ms: u32,
        buffer: &mut [u32],
    ) -> EcologyFrameState {
        let progress = progress.clamp(0.0, 1.0);
        let state = self.ecology.render_at(elapsed_ms, buffer);
        if self.width == 0 || self.height == 0 || buffer.len() < self.width * self.height {
            return state;
        }

        let center = Point {
            x: self.width as f32 * 0.5,
            y: self.height as f32 * 0.48,
        };
        let min_dim = self.width.min(self.height) as f32;
        let phase_index = phase_index(phase);
        let phase_seed = self.phase_offsets[phase_index];
        let pulse = 0.5
            + 0.5
                * (elapsed_ms as f32 * 0.0022
                    + phase_seed * std::f32::consts::TAU)
                    .sin();

        let primary = phase_color(phase, pulse);
        draw_incubation_chamber(
            buffer,
            self.width,
            self.height,
            center,
            min_dim,
            progress,
            elapsed_ms,
            primary,
        );
        draw_module_seeds(
            buffer,
            self.width,
            self.height,
            center,
            min_dim,
            phase_index,
            progress,
            elapsed_ms,
            primary,
        );
        draw_progress_halo(
            buffer,
            self.width,
            self.height,
            center,
            min_dim * 0.235,
            progress,
            primary,
        );
        draw_install_identity(
            buffer,
            self.width,
            self.height,
            phase,
            progress,
            primary,
        );
        state
    }
}

#[derive(Debug, Clone, Copy)]
struct Point {
    x: f32,
    y: f32,
}

fn phase_index(phase: InoculationPhase) -> usize {
    match phase {
        InoculationPhase::Attestation => 0,
        InoculationPhase::PreparingSubstrate => 1,
        InoculationPhase::WeavingSystem => 2,
        InoculationPhase::SeedingSecurity => 3,
        InoculationPhase::OpeningChannels => 4,
        InoculationPhase::Personalizing => 5,
        InoculationPhase::Finalizing => 6,
        InoculationPhase::Complete => 7,
    }
}

fn phase_color(phase: InoculationPhase, pulse: f32) -> Rgba {
    match phase {
        InoculationPhase::Attestation => Rgba::lerp(HOLO_CYAN, MYCELIAL_WHITE, 0.55),
        InoculationPhase::PreparingSubstrate => Rgba::lerp(LEAF_GREEN, HOLO_CYAN, 0.35),
        InoculationPhase::WeavingSystem => Rgba::lerp(LEAF_GREEN, MYCELIAL_WHITE, 0.35),
        InoculationPhase::SeedingSecurity => Rgba::lerp(SOLAR_GOLD, MYCELIAL_WHITE, 0.20),
        InoculationPhase::OpeningChannels => Rgba::lerp(HOLO_CYAN, LEAF_GREEN, 0.42),
        InoculationPhase::Personalizing => Rgba::lerp(LEAF_GREEN, SOLAR_GOLD, 0.18 + pulse * 0.15),
        InoculationPhase::Finalizing => Rgba::lerp(SOLAR_GOLD, MYCELIAL_WHITE, 0.48),
        InoculationPhase::Complete => MYCELIAL_WHITE,
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_incubation_chamber(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    center: Point,
    min_dim: f32,
    progress: f32,
    elapsed_ms: u32,
    color: Rgba,
) {
    let chamber_radius = min_dim * 0.245;
    let half_height = min_dim * 0.245;
    let vertical_gain = 0.32 + progress * 0.68;
    let top = Point {
        x: center.x,
        y: center.y - half_height,
    };
    let bottom = Point {
        x: center.x,
        y: center.y + half_height,
    };

    draw_ellipse(
        buffer,
        width,
        height,
        top,
        chamber_radius,
        chamber_radius * 0.22,
        color.with_opacity(0.18 + progress * 0.10),
        true,
    );
    draw_ellipse(
        buffer,
        width,
        height,
        bottom,
        chamber_radius,
        chamber_radius * 0.22,
        Rgba::lerp(color, SOLAR_GOLD, 0.25).with_opacity(0.20 + progress * 0.12),
        false,
    );

    let columns = 10usize;
    let motion = elapsed_ms as f32 * 0.0014;
    for index in 0..columns {
        let t = index as f32 / (columns - 1) as f32;
        let angle = std::f32::consts::PI * (t - 0.5);
        let x = center.x + angle.sin() * chamber_radius;
        let depth = angle.cos().abs();
        let shimmer = 0.5 + 0.5 * (motion + index as f32 * 0.81).sin();
        let alpha = (0.025 + depth * 0.045 + shimmer * 0.018) * vertical_gain;
        draw_line(
            buffer,
            width,
            height,
            Point { x, y: top.y },
            Point { x, y: bottom.y },
            Rgba::lerp(HOLO_CYAN, color, depth).with_opacity(alpha),
        );
    }

    let sweep = (elapsed_ms as f32 * 0.00055).sin() * 0.05;
    for ring in 0..3 {
        let scale = 1.0 - ring as f32 * 0.11;
        let y = top.y - min_dim * (0.04 + ring as f32 * 0.026);
        draw_ellipse(
            buffer,
            width,
            height,
            Point { x: center.x, y },
            chamber_radius * scale,
            chamber_radius * (0.12 + sweep.abs()),
            HOLO_CYAN.with_opacity(0.07 + ring as f32 * 0.018),
            ring % 2 == 0,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_module_seeds(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    center: Point,
    min_dim: f32,
    active_phase: usize,
    progress: f32,
    elapsed_ms: u32,
    color: Rgba,
) {
    let orbit_x = min_dim * 0.39;
    let orbit_y = min_dim * 0.29;
    for index in 0..8usize {
        let angle = std::f32::consts::TAU * index as f32 / 8.0 - std::f32::consts::FRAC_PI_2;
        let point = Point {
            x: center.x + angle.cos() * orbit_x,
            y: center.y + angle.sin() * orbit_y,
        };
        let completed = index < active_phase || (index == active_phase && progress > 0.82);
        let active = index == active_phase;
        let pulse = 0.5 + 0.5 * (elapsed_ms as f32 * 0.004 + index as f32).sin();
        let node_color = if completed {
            Rgba::lerp(LEAF_GREEN, MYCELIAL_WHITE, 0.28)
        } else if active {
            Rgba::lerp(color, MYCELIAL_WHITE, pulse * 0.28)
        } else {
            HOLO_CYAN
        };
        let alpha = if completed {
            0.38
        } else if active {
            0.28 + pulse * 0.22
        } else {
            0.08
        };
        draw_node(
            buffer,
            width,
            height,
            point,
            if active { 4.2 } else { 2.8 },
            node_color.with_opacity(alpha),
        );

        if completed || active {
            draw_line(
                buffer,
                width,
                height,
                point,
                center,
                node_color.with_opacity(if active { 0.065 } else { 0.035 }),
            );
        }
    }
}

fn draw_progress_halo(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    center: Point,
    radius: f32,
    progress: f32,
    color: Rgba,
) {
    let segments = 180usize;
    let complete = ((segments as f32 * progress).round() as usize).min(segments);
    let mut previous = point_on_ellipse(center, radius, radius * 0.25, 0.0);
    for index in 1..=segments {
        let angle = std::f32::consts::TAU * index as f32 / segments as f32;
        let current = point_on_ellipse(center, radius, radius * 0.25, angle);
        let active = index <= complete;
        let ring_color = if active {
            Rgba::lerp(color, SOLAR_GOLD, 0.24).with_opacity(0.34)
        } else {
            HOLO_CYAN.with_opacity(0.045)
        };
        if index % 4 != 1 {
            draw_line(buffer, width, height, previous, current, ring_color);
        }
        previous = current;
    }
}

fn draw_install_identity(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    phase: InoculationPhase,
    progress: f32,
    primary: Rgba,
) {
    if width < 120 || height < 70 {
        return;
    }
    let scale = (height / 270).clamp(1, 4);
    let margin_x = (width / 24).max(8);
    let margin_y = (height / 18).max(8);
    let title = "SPORE";
    let subtitle = "INOCULATION";
    let title_metrics = microtype::measure(title, scale, scale);
    let subtitle_metrics = microtype::measure(subtitle, scale, scale);
    let block_height = title_metrics.height + subtitle_metrics.height + scale * 12;
    let y = height.saturating_sub(margin_y + block_height);

    microtype::draw_text(
        buffer,
        width,
        height,
        margin_x + 1,
        y + 1,
        title,
        scale,
        scale,
        HOLO_CYAN.with_opacity(0.10),
    );
    microtype::draw_text(
        buffer,
        width,
        height,
        margin_x,
        y,
        title,
        scale,
        scale,
        MYCELIAL_WHITE.with_opacity(0.86),
    );
    microtype::draw_text(
        buffer,
        width,
        height,
        margin_x,
        y + title_metrics.height + scale * 4,
        subtitle,
        scale,
        scale,
        Rgba::lerp(LEAF_GREEN, primary, 0.42).with_opacity(0.68),
    );

    let phase_text = phase.display_label();
    let phase_metrics = microtype::measure(phase_text, scale, scale);
    if phase_metrics.width + margin_x < width {
        microtype::draw_text(
            buffer,
            width,
            height,
            width.saturating_sub(margin_x + phase_metrics.width),
            height.saturating_sub(margin_y + phase_metrics.height),
            phase_text,
            scale,
            scale,
            primary.with_opacity(0.46),
        );
    }

    // Progress is shown as geometry first. A small numeric readout is a factual
    // supplement, not authority over whether installation is complete.
    let percent = (progress * 100.0).round().clamp(0.0, 100.0) as u8;
    let percent_text = format!("{percent}");
    let percent_metrics = microtype::measure(&percent_text, scale, scale);
    let px = width.saturating_sub(margin_x + percent_metrics.width);
    let py = height.saturating_sub(margin_y + phase_metrics.height + scale * 12);
    microtype::draw_text(
        buffer,
        width,
        height,
        px,
        py,
        &percent_text,
        scale,
        scale,
        Rgba::lerp(primary, MYCELIAL_WHITE, 0.35).with_opacity(0.40),
    );
}

#[allow(clippy::too_many_arguments)]
fn draw_ellipse(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    center: Point,
    radius_x: f32,
    radius_y: f32,
    color: Rgba,
    alternate_dash: bool,
) {
    let segments = ((radius_x * 1.4) as usize).clamp(64, 360);
    let mut previous = point_on_ellipse(center, radius_x, radius_y, 0.0);
    for index in 1..=segments {
        let angle = std::f32::consts::TAU * index as f32 / segments as f32;
        let current = point_on_ellipse(center, radius_x, radius_y, angle);
        let dash = if alternate_dash { 6 } else { 9 };
        if (index / dash) % 3 != 1 {
            draw_line(buffer, width, height, previous, current, color);
        }
        previous = current;
    }
}

fn point_on_ellipse(center: Point, radius_x: f32, radius_y: f32, angle: f32) -> Point {
    Point {
        x: center.x + angle.cos() * radius_x,
        y: center.y + angle.sin() * radius_y,
    }
}

fn draw_node(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    center: Point,
    radius: f32,
    color: Rgba,
) {
    let outer = radius.ceil() as i32 + 2;
    for oy in -outer..=outer {
        for ox in -outer..=outer {
            let distance = ((ox * ox + oy * oy) as f32).sqrt();
            if distance > outer as f32 {
                continue;
            }
            let x = center.x.round() as i32 + ox;
            let y = center.y.round() as i32 + oy;
            if x < 0 || y < 0 || x as usize >= width || y as usize >= height {
                continue;
            }
            let falloff = (1.0 - distance / outer.max(1) as f32).powf(1.7);
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

fn draw_line(
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

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_boot_ecology::{BootEcologyComposer, BootStateReceipt, MorphologyLineage};

    #[test]
    fn inoculation_phases_have_stable_labels() {
        let labels: Vec<_> = InoculationPhase::ALL.iter().map(|phase| phase.label()).collect();
        assert_eq!(labels.len(), 8);
        assert_eq!(labels[0], "attestation");
        assert_eq!(labels[7], "complete");
    }

    #[test]
    fn inoculation_overlay_changes_fidelity_frame() {
        let receipt = BootStateReceipt::first_boot([0x71; 32]);
        let genome = BootEcologyComposer::compose(&receipt, &MorphologyLineage::default());
        let ecology = EcologyRenderer::new(320, 180, genome.clone());
        let inoculation = InoculationRenderer::new(320, 180, genome);
        let mut base = vec![0u32; 320 * 180];
        let mut install = vec![0u32; 320 * 180];
        ecology.render_at(1_800, &mut base);
        inoculation.render(
            InoculationPhase::WeavingSystem,
            0.55,
            1_800,
            &mut install,
        );
        assert_ne!(base, install);
    }

    #[test]
    fn install_labels_are_factual() {
        for phase in InoculationPhase::ALL {
            let label = phase.display_label().to_ascii_lowercase();
            assert!(!label.contains("conscious"));
            assert!(!label.contains("sentient"));
            assert!(!label.contains("aware"));
        }
    }
}
