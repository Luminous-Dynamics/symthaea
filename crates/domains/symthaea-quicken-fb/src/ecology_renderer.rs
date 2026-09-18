// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic procedural renderer for `BootGenome`.
//!
//! The renderer pre-generates a bounded curve topology from the genome seed and
//! reveals/modulates it as semantic boot stages advance. It uses only a CPU pixel
//! buffer so the exact same renderer works for DRM/KMS and offline previews.

use crate::color::{BLACK, LEAF_GREEN, MOSS_DEEP, MYCELIAL_WHITE, Rgba, SOLAR_GOLD};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use symthaea_boot_ecology::{BootGenome, BootStageKind, MorphologyFamily};

const MAX_CURVES: usize = 2_400;
const MAX_DEPTH: u8 = 10;

#[derive(Debug, Clone, Copy)]
struct Point {
    x: f32,
    y: f32,
}

impl Point {
    fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
        }
    }

    fn distance(self, other: Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

#[derive(Debug, Clone)]
struct Curve {
    start: Point,
    control: Point,
    end: Point,
    thickness: f32,
    depth: u8,
    reveal_start: f32,
    reveal_end: f32,
    repair_mark: bool,
    candidate_growth: bool,
    node_phase: f32,
}

#[derive(Debug, Clone, Copy)]
struct Spore {
    center: Point,
    radius: f32,
    phase: f32,
}

/// Frame metadata useful to preview tools and tests.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EcologyFrameState {
    pub stage: BootStageKind,
    pub stage_progress: f32,
    pub sequence_progress: f32,
    pub visible_fraction: f32,
}

/// CPU renderer shared by live DRM output and offline preview generation.
pub struct EcologyRenderer {
    width: u32,
    height: u32,
    genome: BootGenome,
    curves: Vec<Curve>,
    spores: Vec<Spore>,
    mesh_links: Vec<(Point, Point, f32)>,
    total_duration_ms: u32,
}

impl EcologyRenderer {
    pub fn new(width: u32, height: u32, genome: BootGenome) -> Self {
        let total_duration_ms = genome.visual_budget_ms().max(1);
        let mut rng = StdRng::from_seed(genome.seed);
        let spores = build_spores(width, height, &genome, &mut rng);
        let curves = build_topology(width, height, &genome, &spores, &mut rng);
        let mesh_links = build_mesh_links(&genome, &spores, &curves, &mut rng);
        Self {
            width,
            height,
            genome,
            curves,
            spores,
            mesh_links,
            total_duration_ms,
        }
    }

    pub fn genome(&self) -> &BootGenome {
        &self.genome
    }

    pub fn total_duration_ms(&self) -> u32 {
        self.total_duration_ms
    }

    pub fn frame_state(&self, elapsed_ms: u32) -> EcologyFrameState {
        let elapsed_ms = elapsed_ms.min(self.total_duration_ms);
        let mut cursor = 0u32;
        for stage in &self.genome.stages {
            let end = cursor.saturating_add(stage.duration_ms);
            if elapsed_ms <= end || end >= self.total_duration_ms {
                let local = elapsed_ms.saturating_sub(cursor);
                let stage_progress = if stage.duration_ms == 0 {
                    1.0
                } else {
                    local as f32 / stage.duration_ms as f32
                }
                .clamp(0.0, 1.0);
                return EcologyFrameState {
                    stage: stage.kind,
                    stage_progress,
                    sequence_progress: elapsed_ms as f32 / self.total_duration_ms as f32,
                    visible_fraction: self.visible_fraction(elapsed_ms),
                };
            }
            cursor = end;
        }

        EcologyFrameState {
            stage: BootStageKind::Handoff,
            stage_progress: 1.0,
            sequence_progress: 1.0,
            visible_fraction: 1.0,
        }
    }

    fn visible_fraction(&self, elapsed_ms: u32) -> f32 {
        let mut cursor = 0u32;
        let mut growth_budget = 0u32;
        let mut growth_elapsed = 0u32;

        for stage in &self.genome.stages {
            let contributes = matches!(
                stage.kind,
                BootStageKind::Germinate
                    | BootStageKind::Grow
                    | BootStageKind::Anastomose
                    | BootStageKind::GrowthRing
                    | BootStageKind::HardwareBud
                    | BootStageKind::Repair
                    | BootStageKind::MeshLink
            );
            if contributes {
                growth_budget = growth_budget.saturating_add(stage.duration_ms);
                let local = elapsed_ms.saturating_sub(cursor).min(stage.duration_ms);
                growth_elapsed = growth_elapsed.saturating_add(local);
            }
            cursor = cursor.saturating_add(stage.duration_ms);
            if cursor >= elapsed_ms {
                break;
            }
        }

        if matches!(
            self.genome.family,
            MorphologyFamily::MinimalRelight | MorphologyFamily::CrystalThaw
        ) {
            // Resume visuals represent continuity: topology exists already and
            // illumination returns instead of regrowing from zero.
            return (0.55 + elapsed_ms as f32 / self.total_duration_ms as f32 * 0.45)
                .clamp(0.0, 1.0);
        }

        if growth_budget == 0 {
            return elapsed_ms as f32 / self.total_duration_ms as f32;
        }
        (growth_elapsed as f32 / growth_budget as f32).clamp(0.0, 1.0)
    }

    /// Render a frame at an absolute sequence time into XRGB8888 pixels.
    pub fn render_at(&self, elapsed_ms: u32, buffer: &mut [u32]) -> EcologyFrameState {
        let w = self.width as usize;
        let h = self.height as usize;
        assert!(buffer.len() >= w * h, "buffer too small");

        let state = self.frame_state(elapsed_ms);
        let bg = background_color(&self.genome, state);
        buffer[..w * h].fill(bg.to_xrgb8888());

        if state.stage == BootStageKind::Blackout {
            return state;
        }

        let handoff_opacity = if state.stage == BootStageKind::Handoff {
            1.0 - smoothstep(state.stage_progress)
        } else {
            1.0
        };

        self.draw_spores(buffer, state, handoff_opacity);
        self.draw_curves(buffer, state, handoff_opacity);

        if matches!(state.stage, BootStageKind::GrowthRing) {
            self.draw_growth_ring(buffer, state, handoff_opacity);
        }
        if matches!(state.stage, BootStageKind::MeshLink | BootStageKind::Settle | BootStageKind::Handoff)
        {
            self.draw_mesh_links(buffer, state, handoff_opacity);
        }

        state
    }

    fn draw_spores(&self, buffer: &mut [u32], state: EcologyFrameState, opacity: f32) {
        let pulse = 0.5 + 0.5 * (state.sequence_progress * std::f32::consts::TAU * 2.0).sin();
        for spore in &self.spores {
            let local = 0.5 + 0.5 * (state.sequence_progress * 8.0 + spore.phase).sin();
            let radius = spore.radius * (0.92 + 0.10 * local);
            let core = Rgba::lerp(LEAF_GREEN, MYCELIAL_WHITE, 0.35 + 0.35 * pulse)
                .with_opacity((0.75 * opacity).clamp(0.0, 1.0));
            draw_glow_circle(
                buffer,
                self.width as usize,
                self.height as usize,
                spore.center.x,
                spore.center.y,
                radius,
                core,
                self.genome.parameters.glow_radius,
            );
        }
    }

    fn draw_curves(&self, buffer: &mut [u32], state: EcologyFrameState, opacity: f32) {
        let visible = state.visible_fraction;
        let retract = if state.stage == BootStageKind::RetractFailedGrowth {
            1.0 - smoothstep(state.stage_progress)
        } else {
            1.0
        };
        let repair_stage = if state.stage == BootStageKind::Repair {
            smoothstep(state.stage_progress)
        } else {
            1.0
        };

        for curve in &self.curves {
            if visible <= curve.reveal_start {
                continue;
            }
            let mut local = ((visible - curve.reveal_start)
                / (curve.reveal_end - curve.reveal_start).max(0.001))
                .clamp(0.0, 1.0);
            if curve.candidate_growth {
                local *= retract;
            }
            if curve.repair_mark {
                local *= repair_stage;
            }
            if local <= 0.001 {
                continue;
            }

            let pulse = 0.5
                + 0.5
                    * (state.sequence_progress * self.genome.parameters.pulse_velocity * 18.0
                        + curve.node_phase)
                        .sin();
            let depth_fade = (1.0 - curve.depth as f32 * 0.055).max(0.32);
            let color = if curve.repair_mark {
                Rgba::lerp(SOLAR_GOLD, MYCELIAL_WHITE, pulse * 0.35)
            } else {
                let depth_mix = (curve.depth as f32 / MAX_DEPTH as f32).clamp(0.0, 1.0);
                let base = Rgba::lerp(LEAF_GREEN, MYCELIAL_WHITE, depth_mix * 0.72);
                Rgba::lerp(base, SOLAR_GOLD, self.genome.parameters.solar_gold_fraction * pulse * 0.45)
            }
            .with_opacity((depth_fade * opacity).clamp(0.0, 1.0));

            draw_quadratic_curve(
                buffer,
                self.width as usize,
                self.height as usize,
                curve,
                local,
                color,
                self.genome.parameters.glow_radius,
            );

            if local > 0.96 && curve.depth > 0 {
                let radius = (curve.thickness * 1.25 + 0.8).max(1.2);
                draw_glow_circle(
                    buffer,
                    self.width as usize,
                    self.height as usize,
                    curve.end.x,
                    curve.end.y,
                    radius,
                    color.brighten(1.0 + pulse * 0.55),
                    self.genome.parameters.glow_radius * 0.65,
                );
            }
        }
    }

    fn draw_growth_ring(&self, buffer: &mut [u32], state: EcologyFrameState, opacity: f32) {
        if self.spores.is_empty() {
            return;
        }
        let center = self.spores[0].center;
        let min_dim = self.width.min(self.height) as f32;
        let radius = min_dim * (0.08 + 0.34 * smoothstep(state.stage_progress));
        let color = SOLAR_GOLD.with_opacity((0.60 * (1.0 - state.stage_progress) * opacity).clamp(0.0, 1.0));
        draw_ring(
            buffer,
            self.width as usize,
            self.height as usize,
            center,
            radius,
            1.5 + self.genome.parameters.glow_radius * 2.5,
            color,
        );
    }

    fn draw_mesh_links(&self, buffer: &mut [u32], state: EcologyFrameState, opacity: f32) {
        if self.genome.parameters.mesh_opacity <= 0.0 {
            return;
        }
        let stage_gain = if state.stage == BootStageKind::MeshLink {
            smoothstep(state.stage_progress)
        } else {
            1.0
        };
        let alpha = (self.genome.parameters.mesh_opacity * 0.48 * stage_gain * opacity).clamp(0.0, 1.0);
        let color = Rgba::lerp(LEAF_GREEN, MYCELIAL_WHITE, 0.55).with_opacity(alpha);
        for (a, b, phase) in &self.mesh_links {
            let pulse = 0.55 + 0.45 * (state.sequence_progress * 16.0 + phase).sin();
            draw_soft_line(
                buffer,
                self.width as usize,
                self.height as usize,
                *a,
                *b,
                1.0,
                color.brighten(0.8 + pulse * 0.5),
                1,
            );
        }
    }
}

fn build_spores(
    width: u32,
    height: u32,
    genome: &BootGenome,
    rng: &mut StdRng,
) -> Vec<Spore> {
    let w = width as f32;
    let h = height as f32;
    let min_dim = w.min(h);
    let requested = genome.parameters.spore_count.max(1) as usize;
    let count = match genome.family {
        MorphologyFamily::CentralSpore | MorphologyFamily::KintsugiRepair => 1,
        MorphologyFamily::MinimalRelight => requested.min(2),
        MorphologyFamily::FairyRing => requested.max(3).min(6),
        _ => requested.min(5),
    };

    let mut spores = Vec::with_capacity(count);
    match genome.family {
        MorphologyFamily::MycelialFan | MorphologyFamily::MemoryGarden => {
            for i in 0..count {
                let t = (i + 1) as f32 / (count + 1) as f32;
                spores.push(Spore {
                    center: Point {
                        x: w * (0.18 + 0.64 * t),
                        y: h * (0.70 + rng.gen_range(-0.04..0.04)),
                    },
                    radius: min_dim * rng.gen_range(0.010..0.018),
                    phase: rng.gen_range(0.0..std::f32::consts::TAU),
                });
            }
        }
        MorphologyFamily::RiverDelta => spores.push(Spore {
            center: Point { x: w * 0.13, y: h * 0.50 },
            radius: min_dim * 0.015,
            phase: 0.0,
        }),
        MorphologyFamily::FairyRing => {
            let center = Point { x: w * 0.5, y: h * 0.48 };
            let ring = min_dim * 0.16;
            for i in 0..count {
                let angle = std::f32::consts::TAU * i as f32 / count as f32;
                spores.push(Spore {
                    center: Point {
                        x: center.x + angle.cos() * ring,
                        y: center.y + angle.sin() * ring,
                    },
                    radius: min_dim * 0.010,
                    phase: angle,
                });
            }
        }
        MorphologyFamily::LichenCells | MorphologyFamily::ConstellationHyphae => {
            for _ in 0..count {
                spores.push(Spore {
                    center: Point {
                        x: w * rng.gen_range(0.28..0.72),
                        y: h * rng.gen_range(0.28..0.68),
                    },
                    radius: min_dim * rng.gen_range(0.008..0.014),
                    phase: rng.gen_range(0.0..std::f32::consts::TAU),
                });
            }
        }
        _ => spores.push(Spore {
            center: Point { x: w * 0.5, y: h * 0.48 },
            radius: min_dim * 0.016,
            phase: 0.0,
        }),
    }
    spores
}

fn build_topology(
    width: u32,
    height: u32,
    genome: &BootGenome,
    spores: &[Spore],
    rng: &mut StdRng,
) -> Vec<Curve> {
    #[derive(Clone, Copy)]
    struct Tip {
        point: Point,
        angle: f32,
        thickness: f32,
        depth: u8,
        reveal: f32,
    }

    let w = width as f32;
    let h = height as f32;
    let min_dim = w.min(h);
    let params = &genome.parameters;
    let mut curves = Vec::with_capacity(1_024);
    let mut frontier = Vec::<Tip>::new();

    for (spore_index, spore) in spores.iter().enumerate() {
        let roots = root_count(genome.family, genome.seed[spore_index % 32]);
        for i in 0..roots {
            let angle = root_angle(genome.family, i, roots, rng);
            frontier.push(Tip {
                point: spore.center,
                angle,
                thickness: (min_dim / 420.0).clamp(1.3, 4.5),
                depth: 0,
                reveal: rng.gen_range(0.0..0.06),
            });
        }
    }

    let mut cursor = 0usize;
    while cursor < frontier.len() && curves.len() < MAX_CURVES {
        let tip = frontier[cursor];
        cursor += 1;
        if tip.depth > MAX_DEPTH {
            continue;
        }

        let depth_scale = 1.0 / (1.0 + tip.depth as f32 * 0.28);
        let family_length = match genome.family {
            MorphologyFamily::ConstellationHyphae => 1.35,
            MorphologyFamily::RiverDelta => 1.20,
            MorphologyFamily::LichenCells => 0.74,
            MorphologyFamily::MinimalRelight => 0.82,
            _ => 1.0,
        };
        let length = min_dim
            * 0.105
            * depth_scale
            * family_length
            * rng.gen_range(0.72..1.25)
            * params.camera_scale;
        let turbulence = (rng.gen_range(-1.0..1.0) * params.turbulence * 0.42)
            + family_angle_bias(genome.family, tip.point, width, height);
        let angle = tip.angle + turbulence;
        let end = Point {
            x: tip.point.x + angle.cos() * length,
            y: tip.point.y + angle.sin() * length,
        };
        if end.x < -min_dim * 0.04
            || end.x > w + min_dim * 0.04
            || end.y < -min_dim * 0.04
            || end.y > h + min_dim * 0.04
        {
            continue;
        }

        let normal = Point {
            x: -angle.sin(),
            y: angle.cos(),
        };
        let curve_offset = length
            * params.curvature
            * rng.gen_range(-0.42..0.42)
            * (0.55 + params.growth_anisotropy * 0.45);
        let midpoint = tip.point.lerp(end, 0.50);
        let control = Point {
            x: midpoint.x + normal.x * curve_offset,
            y: midpoint.y + normal.y * curve_offset,
        };

        let reveal_span = (0.050 + tip.depth as f32 * 0.010).min(0.13);
        let reveal_start = tip.reveal.clamp(0.0, 0.96);
        let reveal_end = (reveal_start + reveal_span).min(1.0);
        let repair_probability = params.repair_intensity * 0.24;
        let repair_mark = matches!(genome.family, MorphologyFamily::KintsugiRepair)
            && tip.depth >= 2
            && rng.gen_bool(repair_probability.clamp(0.0, 0.65) as f64);
        let candidate_growth = matches!(
            genome.cue,
            symthaea_boot_ecology::BootCue::RestoringKnownGood
        ) && tip.depth >= 4
            && rng.gen_bool(0.22);

        curves.push(Curve {
            start: tip.point,
            control,
            end,
            thickness: tip.thickness,
            depth: tip.depth,
            reveal_start,
            reveal_end,
            repair_mark,
            candidate_growth,
            node_phase: rng.gen_range(0.0..std::f32::consts::TAU),
        });

        if tip.depth == MAX_DEPTH || curves.len() >= MAX_CURVES {
            continue;
        }

        let base_children = if rng.gen_bool(params.branching_probability.clamp(0.0, 0.96) as f64) {
            2
        } else {
            1
        };
        let extra = if tip.depth < 4 && rng.gen_bool((params.branching_probability * 0.22) as f64) {
            1
        } else {
            0
        };
        let children = base_children + extra;
        let fork_base = match genome.family {
            MorphologyFamily::RiverDelta => 0.28,
            MorphologyFamily::CrystalThaw => 0.68,
            MorphologyFamily::LichenCells => 0.52,
            _ => 0.42,
        };
        for child in 0..children {
            if frontier.len() >= MAX_CURVES * 2 {
                break;
            }
            let centered = child as f32 - (children - 1) as f32 * 0.5;
            let fork = centered * fork_base + rng.gen_range(-0.18..0.18);
            frontier.push(Tip {
                point: end,
                angle: angle + fork,
                thickness: (tip.thickness * rng.gen_range(0.68..0.82)).max(0.65),
                depth: tip.depth + 1,
                reveal: (reveal_end - rng.gen_range(0.0..0.025)).clamp(0.0, 1.0),
            });
        }
    }

    // Anastomosis: add a bounded set of softly curved bridges between nearby
    // mature endpoints. We sample pairs rather than O(n^2) scanning.
    let attempts = ((curves.len() as f32 * params.anastomosis_probability * 0.28) as usize)
        .min(240);
    let base_len = curves.len();
    if base_len > 8 {
        for _ in 0..attempts {
            if curves.len() >= MAX_CURVES {
                break;
            }
            let a = rng.gen_range(0..base_len);
            let b = rng.gen_range(0..base_len);
            if a == b {
                continue;
            }
            let pa = curves[a].end;
            let pb = curves[b].end;
            let distance = pa.distance(pb);
            if distance < min_dim * 0.035 || distance > min_dim * 0.18 {
                continue;
            }
            let midpoint = pa.lerp(pb, 0.5);
            let dx = pb.x - pa.x;
            let dy = pb.y - pa.y;
            let len = distance.max(1.0);
            let bend = rng.gen_range(-0.12..0.12) * distance;
            let control = Point {
                x: midpoint.x - dy / len * bend,
                y: midpoint.y + dx / len * bend,
            };
            let reveal_start = curves[a].reveal_end.max(curves[b].reveal_end).min(0.96);
            curves.push(Curve {
                start: pa,
                control,
                end: pb,
                thickness: curves[a].thickness.min(curves[b].thickness) * 0.65,
                depth: curves[a].depth.max(curves[b].depth),
                reveal_start,
                reveal_end: (reveal_start + 0.06).min(1.0),
                repair_mark: false,
                candidate_growth: false,
                node_phase: rng.gen_range(0.0..std::f32::consts::TAU),
            });
        }
    }

    curves
}

fn build_mesh_links(
    genome: &BootGenome,
    spores: &[Spore],
    curves: &[Curve],
    rng: &mut StdRng,
) -> Vec<(Point, Point, f32)> {
    if !genome.parameters.mesh_opacity.is_sign_positive() || curves.len() < 8 {
        return Vec::new();
    }
    let count = (2 + (genome.parameters.mesh_opacity * 8.0) as usize).min(10);
    let mut links = Vec::with_capacity(count);
    for i in 0..count {
        let a = if i < spores.len() {
            spores[i].center
        } else {
            curves[rng.gen_range(0..curves.len())].end
        };
        let b = curves[rng.gen_range(0..curves.len())].end;
        if a.distance(b) > 24.0 {
            links.push((a, b, rng.gen_range(0.0..std::f32::consts::TAU)));
        }
    }
    links
}

fn root_count(family: MorphologyFamily, selector: u8) -> usize {
    match family {
        MorphologyFamily::CentralSpore => 7 + selector as usize % 5,
        MorphologyFamily::RiverDelta => 5 + selector as usize % 3,
        MorphologyFamily::LichenCells => 4 + selector as usize % 3,
        MorphologyFamily::ConstellationHyphae => 3 + selector as usize % 3,
        MorphologyFamily::FairyRing => 3,
        MorphologyFamily::MinimalRelight => 5,
        MorphologyFamily::CrystalThaw => 6,
        _ => 6 + selector as usize % 4,
    }
}

fn root_angle(family: MorphologyFamily, index: usize, count: usize, rng: &mut StdRng) -> f32 {
    match family {
        MorphologyFamily::MycelialFan | MorphologyFamily::MemoryGarden => {
            let spread = 2.25;
            -std::f32::consts::FRAC_PI_2 - spread * 0.5
                + spread * index as f32 / (count.saturating_sub(1).max(1)) as f32
                + rng.gen_range(-0.08..0.08)
        }
        MorphologyFamily::RiverDelta => rng.gen_range(-0.48..0.48),
        MorphologyFamily::CrystalThaw => {
            let cardinal = [0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI, -std::f32::consts::FRAC_PI_2];
            cardinal[index % cardinal.len()] + rng.gen_range(-0.08..0.08)
        }
        _ => std::f32::consts::TAU * index as f32 / count.max(1) as f32
            + rng.gen_range(-0.10..0.10),
    }
}

fn family_angle_bias(family: MorphologyFamily, point: Point, width: u32, height: u32) -> f32 {
    match family {
        MorphologyFamily::FairyRing => {
            let cx = width as f32 * 0.5;
            let cy = height as f32 * 0.48;
            let radial = (point.y - cy).atan2(point.x - cx);
            radial.sin() * 0.035
        }
        MorphologyFamily::RiverDelta => 0.0,
        MorphologyFamily::HdcOrganic => ((point.x * 0.013).sin() + (point.y * 0.011).cos()) * 0.025,
        _ => 0.0,
    }
}

fn background_color(genome: &BootGenome, state: EcologyFrameState) -> Rgba {
    let base = match genome.family {
        MorphologyFamily::CrystalThaw => Rgba::lerp(BLACK, MOSS_DEEP, 0.50),
        MorphologyFamily::KintsugiRepair => Rgba::lerp(BLACK, MOSS_DEEP, 0.72),
        _ => MOSS_DEEP,
    };
    let intro = smoothstep((state.sequence_progress * 5.0).clamp(0.0, 1.0));
    Rgba::lerp(BLACK, base, intro)
}

fn draw_quadratic_curve(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    curve: &Curve,
    reveal: f32,
    color: Rgba,
    glow_radius: f32,
) {
    let steps = ((curve.start.distance(curve.end) / 3.5).ceil() as usize).clamp(6, 64);
    let visible_steps = ((steps as f32 * reveal).ceil() as usize).clamp(1, steps);
    let mut previous = curve.start;
    for i in 1..=visible_steps {
        let t = i as f32 / steps as f32;
        let current = quadratic(curve.start, curve.control, curve.end, t);
        let thickness = (curve.thickness * (1.0 - t * 0.20)).max(0.75);
        let glow = (glow_radius * thickness * 1.7).round() as i32;
        if glow > 0 {
            let glow_color = color.with_opacity((color.3 as f32 / 255.0 * 0.16).clamp(0.0, 1.0));
            draw_soft_line(buffer, width, height, previous, current, thickness + glow as f32, glow_color, glow);
        }
        draw_soft_line(buffer, width, height, previous, current, thickness, color, 0);
        previous = current;
    }
}

fn quadratic(a: Point, c: Point, b: Point, t: f32) -> Point {
    let u = 1.0 - t;
    Point {
        x: u * u * a.x + 2.0 * u * t * c.x + t * t * b.x,
        y: u * u * a.y + 2.0 * u * t * c.y + t * t * b.y,
    }
}

fn draw_soft_line(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    a: Point,
    b: Point,
    thickness: f32,
    color: Rgba,
    extra_radius: i32,
) {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let distance = (dx * dx + dy * dy).sqrt().max(1.0);
    let steps = distance.ceil() as usize;
    let radius = ((thickness * 0.5).ceil() as i32 + extra_radius).clamp(1, 12);
    for i in 0..=steps {
        let t = i as f32 / steps.max(1) as f32;
        let x = a.x + dx * t;
        let y = a.y + dy * t;
        draw_disc_alpha(buffer, width, height, x, y, radius, color);
    }
}

fn draw_glow_circle(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    x: f32,
    y: f32,
    radius: f32,
    color: Rgba,
    glow_radius: f32,
) {
    let glow = (radius * (1.0 + glow_radius * 1.8)).ceil() as i32;
    let core = radius.ceil() as i32;
    if glow > core {
        let halo = color.with_opacity((color.3 as f32 / 255.0 * 0.12).clamp(0.0, 1.0));
        draw_disc_alpha(buffer, width, height, x, y, glow.min(18), halo);
    }
    draw_disc_alpha(buffer, width, height, x, y, core.max(1), color);
}

fn draw_ring(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    center: Point,
    radius: f32,
    thickness: f32,
    color: Rgba,
) {
    let segments = ((radius * 0.9) as usize).clamp(48, 240);
    let mut previous = Point { x: center.x + radius, y: center.y };
    for i in 1..=segments {
        let angle = std::f32::consts::TAU * i as f32 / segments as f32;
        let current = Point {
            x: center.x + angle.cos() * radius,
            y: center.y + angle.sin() * radius,
        };
        draw_soft_line(buffer, width, height, previous, current, thickness, color, 1);
        previous = current;
    }
}

fn draw_disc_alpha(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    x: f32,
    y: f32,
    radius: i32,
    color: Rgba,
) {
    let cx = x.round() as i32;
    let cy = y.round() as i32;
    let r2 = radius * radius;
    for oy in -radius..=radius {
        for ox in -radius..=radius {
            let d2 = ox * ox + oy * oy;
            if d2 > r2 {
                continue;
            }
            let px = cx + ox;
            let py = cy + oy;
            if px < 0 || py < 0 || px as usize >= width || py as usize >= height {
                continue;
            }
            let edge = 1.0 - (d2 as f32 / r2.max(1) as f32).sqrt();
            let src = color.with_opacity((color.3 as f32 / 255.0 * (0.35 + 0.65 * edge)).clamp(0.0, 1.0));
            let idx = py as usize * width + px as usize;
            let dst_val = buffer[idx];
            let dst = Rgba(
                ((dst_val >> 16) & 0xff) as u8,
                ((dst_val >> 8) & 0xff) as u8,
                (dst_val & 0xff) as u8,
                0xff,
            );
            buffer[idx] = src.over(dst).to_xrgb8888();
        }
    }
}

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_boot_ecology::{BootEcologyComposer, BootStateReceipt, MorphologyLineage};

    fn renderer() -> EcologyRenderer {
        let receipt = BootStateReceipt::first_boot([0x2a; 32]);
        let genome = BootEcologyComposer::compose(&receipt, &MorphologyLineage::default());
        EcologyRenderer::new(320, 180, genome)
    }

    #[test]
    fn renderer_is_deterministic() {
        let a = renderer();
        let b = renderer();
        let mut fa = vec![0u32; 320 * 180];
        let mut fb = vec![0u32; 320 * 180];
        a.render_at(2_000, &mut fa);
        b.render_at(2_000, &mut fb);
        assert_eq!(fa, fb);
    }

    #[test]
    fn sequence_changes_over_time() {
        let renderer = renderer();
        let mut early = vec![0u32; 320 * 180];
        let mut late = vec![0u32; 320 * 180];
        renderer.render_at(500, &mut early);
        renderer.render_at(2_500, &mut late);
        assert_ne!(early, late);
    }

    #[test]
    fn frame_state_is_bounded() {
        let renderer = renderer();
        let state = renderer.frame_state(u32::MAX);
        assert!((0.0..=1.0).contains(&state.stage_progress));
        assert!((0.0..=1.0).contains(&state.sequence_progress));
        assert!((0.0..=1.0).contains(&state.visible_fraction));
    }
}
