// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Installation-path-specific Spore Inoculation ceremonies.
//!
//! The core installation phases stay factual and shared, but a Web portal, USB
//! forge, Windows pivot, Apple/Asahi handoff, LAN deployment, and local direct
//! installation do not have to look identical. This wrapper adds a small,
//! deterministic visual grammar for the *route* by which the system is being
//! installed without coupling the framebuffer crate to the larger Spore
//! orchestration crate.

use crate::color::{LEAF_GREEN, MYCELIAL_WHITE, Rgba, SOLAR_GOLD};
use crate::ecology_renderer::EcologyFrameState;
use crate::inoculation_renderer::{InoculationPhase, InoculationRenderer};
use symthaea_boot_ecology::BootGenome;

const HOLO_CYAN: Rgba = Rgba(0x58, 0xd8, 0xd2, 0xff);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InoculationPath {
    WebPortal,
    UsbForge,
    Wsl2Pivot,
    AsahiHandshake,
    LanInoculate,
    LocalDirect,
}

impl InoculationPath {
    pub const ALL: [Self; 6] = [
        Self::WebPortal,
        Self::UsbForge,
        Self::Wsl2Pivot,
        Self::AsahiHandshake,
        Self::LanInoculate,
        Self::LocalDirect,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::WebPortal => "web-portal",
            Self::UsbForge => "usb-forge",
            Self::Wsl2Pivot => "wsl2-pivot",
            Self::AsahiHandshake => "asahi-handshake",
            Self::LanInoculate => "lan-inoculate",
            Self::LocalDirect => "local-direct",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Point {
    x: f32,
    y: f32,
}

pub struct InoculationCeremonyRenderer {
    width: usize,
    height: usize,
    path: InoculationPath,
    seed_phase: f32,
    inner: InoculationRenderer,
}

impl InoculationCeremonyRenderer {
    pub fn new(width: u32, height: u32, genome: BootGenome, path: InoculationPath) -> Self {
        let seed_phase = genome.seed[23] as f32 / 255.0 * std::f32::consts::TAU;
        Self {
            width: width as usize,
            height: height as usize,
            path,
            seed_phase,
            inner: InoculationRenderer::new(width, height, genome),
        }
    }

    pub fn path(&self) -> InoculationPath {
        self.path
    }

    pub fn render(
        &self,
        phase: InoculationPhase,
        progress: f32,
        elapsed_ms: u32,
        buffer: &mut [u32],
    ) -> EcologyFrameState {
        let state = self.inner.render(phase, progress, elapsed_ms, buffer);
        if self.width == 0 || self.height == 0 || buffer.len() < self.width * self.height {
            return state;
        }
        let progress = progress.clamp(0.0, 1.0);
        let center = Point {
            x: self.width as f32 * 0.5,
            y: self.height as f32 * 0.48,
        };
        let min_dim = self.width.min(self.height) as f32;
        let time = elapsed_ms as f32 * 0.0014 + self.seed_phase;
        let arrival = smoothstep((progress * 1.25).clamp(0.0, 1.0));

        match self.path {
            InoculationPath::WebPortal => {
                draw_web_portal(buffer, self.width, self.height, center, min_dim, time, arrival)
            }
            InoculationPath::UsbForge => {
                draw_usb_forge(buffer, self.width, self.height, center, min_dim, time, arrival)
            }
            InoculationPath::Wsl2Pivot => {
                draw_bridge_bloom(buffer, self.width, self.height, center, min_dim, time, arrival)
            }
            InoculationPath::AsahiHandshake => {
                draw_orchard_orbits(buffer, self.width, self.height, center, min_dim, time, arrival)
            }
            InoculationPath::LanInoculate => {
                draw_mesh_seeding(buffer, self.width, self.height, center, min_dim, time, arrival)
            }
            InoculationPath::LocalDirect => {
                draw_substrate_contours(buffer, self.width, self.height, center, min_dim, time, arrival)
            }
        }

        if phase == InoculationPhase::Complete {
            draw_completion_convergence(
                buffer,
                self.width,
                self.height,
                center,
                min_dim,
                time,
                progress,
            );
        }
        state
    }
}

fn draw_web_portal(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    center: Point,
    min_dim: f32,
    time: f32,
    gain: f32,
) {
    let color = HOLO_CYAN.with_opacity((0.085 * gain).clamp(0.0, 0.11));
    for index in 0..3 {
        let offset = (index as f32 - 1.0) * min_dim * 0.020;
        draw_ellipse(
            buffer,
            width,
            height,
            Point {
                x: center.x + offset,
                y: center.y,
            },
            min_dim * (0.19 + index as f32 * 0.018),
            min_dim * 0.34,
            (time * 0.018 + index as f32 * 0.08).sin() * 0.06,
            color,
            5 + index,
        );
    }
    let sweep_y = center.y - min_dim * 0.28 + (time * 0.55).sin() * min_dim * 0.03;
    draw_line_alpha(
        buffer,
        width,
        Point {
            x: center.x - min_dim * 0.18,
            y: sweep_y,
        },
        Point {
            x: center.x + min_dim * 0.18,
            y: sweep_y,
        },
        Rgba::lerp(HOLO_CYAN, MYCELIAL_WHITE, 0.55)
            .with_opacity((0.07 * gain).clamp(0.0, 0.10)),
    );
}

fn draw_usb_forge(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    center: Point,
    min_dim: f32,
    time: f32,
    gain: f32,
) {
    let base = Point {
        x: center.x,
        y: center.y + min_dim * 0.245,
    };
    for index in 0..4 {
        let radius = min_dim * (0.10 + index as f32 * 0.045);
        draw_arc(
            buffer,
            width,
            height,
            base,
            radius,
            radius * 0.24,
            0.0,
            0.10,
            std::f32::consts::PI * 0.80,
            Rgba::lerp(SOLAR_GOLD, MYCELIAL_WHITE, index as f32 * 0.12)
                .with_opacity((0.085 * gain).clamp(0.0, 0.11)),
        );
    }
    for index in 0..7 {
        let t = index as f32 / 6.0;
        let x = center.x + (t - 0.5) * min_dim * 0.28;
        let rise = min_dim * (0.09 + 0.04 * (time + index as f32 * 0.7).sin().abs());
        draw_line_alpha(
            buffer,
            width,
            Point { x, y: base.y },
            Point { x, y: base.y - rise },
            SOLAR_GOLD.with_opacity((0.040 * gain).clamp(0.0, 0.06)),
        );
    }
}

fn draw_bridge_bloom(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    center: Point,
    min_dim: f32,
    time: f32,
    gain: f32,
) {
    let separation = min_dim * 0.105;
    let left = Point {
        x: center.x - separation,
        y: center.y,
    };
    let right = Point {
        x: center.x + separation,
        y: center.y,
    };
    draw_ellipse(
        buffer,
        width,
        height,
        left,
        min_dim * 0.17,
        min_dim * 0.24,
        time.sin() * 0.035,
        HOLO_CYAN.with_opacity((0.075 * gain).clamp(0.0, 0.10)),
        7,
    );
    draw_ellipse(
        buffer,
        width,
        height,
        right,
        min_dim * 0.17,
        min_dim * 0.24,
        -time.cos() * 0.035,
        LEAF_GREEN.with_opacity((0.075 * gain).clamp(0.0, 0.10)),
        7,
    );
    for index in 0..5 {
        let y = center.y + (index as f32 - 2.0) * min_dim * 0.035;
        draw_line_alpha(
            buffer,
            width,
            Point { x: left.x, y },
            Point { x: right.x, y: y + time.sin() * 2.0 },
            Rgba::lerp(HOLO_CYAN, LEAF_GREEN, index as f32 / 4.0)
                .with_opacity((0.06 * gain).clamp(0.0, 0.08)),
        );
    }
}

fn draw_orchard_orbits(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    center: Point,
    min_dim: f32,
    time: f32,
    gain: f32,
) {
    for index in 0..5 {
        let rotation = std::f32::consts::TAU * index as f32 / 5.0 + time * 0.012;
        draw_ellipse(
            buffer,
            width,
            height,
            center,
            min_dim * 0.105,
            min_dim * 0.31,
            rotation,
            Rgba::lerp(MYCELIAL_WHITE, SOLAR_GOLD, index as f32 * 0.08)
                .with_opacity((0.052 * gain).clamp(0.0, 0.07)),
            9,
        );
    }
}

fn draw_mesh_seeding(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    center: Point,
    min_dim: f32,
    time: f32,
    gain: f32,
) {
    let count = 7usize;
    let mut nodes = [Point { x: 0.0, y: 0.0 }; 7];
    for (index, node) in nodes.iter_mut().enumerate() {
        let angle = std::f32::consts::TAU * index as f32 / count as f32 + time * 0.018;
        let radius = min_dim * (0.31 + 0.018 * (time + index as f32).sin());
        *node = Point {
            x: center.x + angle.cos() * radius,
            y: center.y + angle.sin() * radius * 0.62,
        };
    }
    for (index, node) in nodes.iter().copied().enumerate() {
        draw_glow_point(
            buffer,
            width,
            height,
            node,
            2.0 + (time + index as f32).sin().abs(),
            Rgba::lerp(LEAF_GREEN, HOLO_CYAN, index as f32 / count as f32)
                .with_opacity((0.14 * gain).clamp(0.0, 0.18)),
        );
        draw_line_alpha(
            buffer,
            width,
            node,
            center,
            HOLO_CYAN.with_opacity((0.032 * gain).clamp(0.0, 0.05)),
        );
        if index % 2 == 0 {
            draw_line_alpha(
                buffer,
                width,
                node,
                nodes[(index + 2) % count],
                LEAF_GREEN.with_opacity((0.025 * gain).clamp(0.0, 0.04)),
            );
        }
    }
}

fn draw_substrate_contours(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    center: Point,
    min_dim: f32,
    time: f32,
    gain: f32,
) {
    for index in 0..6 {
        let y = center.y + min_dim * (0.14 + index as f32 * 0.035);
        let wobble = (time * 0.4 + index as f32 * 0.8).sin() * min_dim * 0.007;
        draw_arc(
            buffer,
            width,
            height,
            Point {
                x: center.x,
                y: y + wobble,
            },
            min_dim * (0.24 + index as f32 * 0.035),
            min_dim * 0.055,
            0.0,
            0.0,
            std::f32::consts::PI,
            Rgba::lerp(LEAF_GREEN, MYCELIAL_WHITE, index as f32 / 8.0)
                .with_opacity((0.045 * gain).clamp(0.0, 0.065)),
        );
    }
}

fn draw_completion_convergence(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    center: Point,
    min_dim: f32,
    time: f32,
    progress: f32,
) {
    let radius = min_dim * (0.34 - smoothstep(progress) * 0.22).max(0.08);
    draw_ellipse(
        buffer,
        width,
        height,
        center,
        radius,
        radius * 0.64,
        time * 0.025,
        MYCELIAL_WHITE.with_opacity((0.10 * progress).clamp(0.0, 0.12)),
        4,
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
    rotation: f32,
    color: Rgba,
    dash_period: usize,
) {
    let segments = ((radius_x.max(radius_y) * 1.7) as usize).clamp(64, 360);
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
    let period = dash_period.max(2);
    let mut previous = point_at(0);
    for index in 1..=segments {
        let current = point_at(index);
        if (index / period) % 3 != 1 {
            draw_line_alpha(buffer, width, previous, current, color);
        }
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
    start: f32,
    span: f32,
    color: Rgba,
) {
    let segments = 84usize;
    let cos_r = rotation.cos();
    let sin_r = rotation.sin();
    let point_at = |t: f32| {
        let angle = start + span * t;
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
        draw_line_alpha(buffer, width, previous, current, color);
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
    let radius = radius.ceil().clamp(1.0, 7.0) as i32;
    let cx = center.x.round() as i32;
    let cy = center.y.round() as i32;
    for oy in -radius..=radius {
        for ox in -radius..=radius {
            let d = ((ox * ox + oy * oy) as f32).sqrt();
            if d > radius as f32 {
                continue;
            }
            let x = cx + ox;
            let y = cy + oy;
            if x < 0 || y < 0 || x as usize >= width || y as usize >= height {
                continue;
            }
            let falloff = (1.0 - d / radius.max(1) as f32).powf(1.7);
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
    a: Point,
    b: Point,
    color: Rgba,
) {
    let height = buffer.len() / width.max(1);
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
            &BootStateReceipt::first_boot([0x61; 32]),
            &MorphologyLineage::default(),
        )
    }

    #[test]
    fn ceremonies_are_deterministic() {
        let a = InoculationCeremonyRenderer::new(320, 180, genome(), InoculationPath::LanInoculate);
        let b = InoculationCeremonyRenderer::new(320, 180, genome(), InoculationPath::LanInoculate);
        let mut fa = vec![0u32; 320 * 180];
        let mut fb = vec![0u32; 320 * 180];
        a.render(InoculationPhase::WeavingSystem, 0.55, 1_800, &mut fa);
        b.render(InoculationPhase::WeavingSystem, 0.55, 1_800, &mut fb);
        assert_eq!(fa, fb);
    }

    #[test]
    fn install_paths_have_distinct_visual_signatures() {
        let usb = InoculationCeremonyRenderer::new(320, 180, genome(), InoculationPath::UsbForge);
        let lan = InoculationCeremonyRenderer::new(320, 180, genome(), InoculationPath::LanInoculate);
        let mut a = vec![0u32; 320 * 180];
        let mut b = vec![0u32; 320 * 180];
        usb.render(InoculationPhase::WeavingSystem, 0.60, 1_900, &mut a);
        lan.render(InoculationPhase::WeavingSystem, 0.60, 1_900, &mut b);
        assert_ne!(a, b);
    }
}
