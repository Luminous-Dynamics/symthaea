// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/// L-system mycelial growth renderer.
///
/// Generates procedural mycelial network growth seeded by the genesis phrase.
/// Renders to a raw pixel buffer using Bresenham's line algorithm — no GPU required.
use crate::color::{LEAF_GREEN, LICHEN_GREY, MOSS_DEEP, MYCELIAL_WHITE, Rgba, SOLAR_GOLD};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Minimum branch length in pixels before a branch can spawn children.
const MIN_BRANCH_LEN: f32 = 4.0;

/// Maximum number of branches to prevent runaway growth.
const MAX_BRANCHES: usize = 8192;

/// Maximum depth of branching recursion.
const MAX_DEPTH: u32 = 12;

/// A single branch of the mycelial network.
#[derive(Debug, Clone)]
pub struct Branch {
    pub start: (f32, f32),
    pub end: (f32, f32),
    pub angle: f32,
    pub thickness: f32,
    pub opacity: f32,
    pub depth: u32,
    pub growing: bool,
    pub growth_progress: f32,
    /// Whether this branch has formed a node at its tip (intersection).
    pub has_node: bool,
    /// Node pulse brightness (0.0 = dormant, 1.0 = fully lit).
    pub node_brightness: f32,
}

impl Branch {
    fn new(start: (f32, f32), angle: f32, thickness: f32, depth: u32) -> Self {
        Self {
            start,
            end: start,
            angle,
            thickness,
            opacity: 1.0,
            depth,
            growing: true,
            growth_progress: 0.0,
            has_node: false,
            node_brightness: 0.0,
        }
    }

    fn length(&self) -> f32 {
        let dx = self.end.0 - self.start.0;
        let dy = self.end.1 - self.start.1;
        (dx * dx + dy * dy).sqrt()
    }

    /// Target length for this branch based on depth (deeper = shorter).
    fn target_length(&self) -> f32 {
        80.0 / (1.0 + self.depth as f32 * 0.5)
    }
}

/// The full mycelial network state.
pub struct MycelialNetwork {
    pub branches: Vec<Branch>,
    pub width: u32,
    pub height: u32,
    rng: StdRng,
    center: (f32, f32),
    /// Elapsed time in seconds (fractional).
    pub elapsed: f32,
    /// All nodes pulsing simultaneously (completion events).
    pub global_pulse: f32,
    /// Contraction progress (0.0 = normal, 1.0 = fully contracted to center).
    pub contraction: f32,
}

impl MycelialNetwork {
    /// Create a new network seeded by the genesis phrase.
    pub fn new(width: u32, height: u32, genesis_phrase: &str) -> Self {
        let hash = blake3::hash(genesis_phrase.as_bytes());
        let seed_bytes: [u8; 32] = *hash.as_bytes();
        let rng = StdRng::from_seed(seed_bytes);
        let center = (width as f32 / 2.0, height as f32 / 2.0);

        let mut net = Self {
            branches: Vec::with_capacity(512),
            width,
            height,
            rng,
            center,
            elapsed: 0.0,
            global_pulse: 0.0,
            contraction: 0.0,
        };

        // Seed initial branches radiating from center (4-8 based on phrase hash).
        let initial_count = 4 + (seed_bytes[0] % 5) as usize;
        let angle_step = std::f32::consts::TAU / initial_count as f32;
        for i in 0..initial_count {
            let angle = angle_step * i as f32 + (seed_bytes[1] as f32 / 255.0) * 0.5;
            net.branches.push(Branch::new(center, angle, 2.5, 0));
        }

        net
    }

    /// Advance the simulation by `dt` seconds.
    /// `io_rate` controls growth speed (0.0 = dormant, 1.0 = maximum growth).
    pub fn grow(&mut self, dt: f32, io_rate: f32) {
        self.elapsed += dt;

        // Decay global pulse
        if self.global_pulse > 0.0 {
            self.global_pulse = (self.global_pulse - dt * 2.0).max(0.0);
        }

        // Decay node brightness
        for branch in &mut self.branches {
            if branch.node_brightness > 0.0 {
                branch.node_brightness = (branch.node_brightness - dt * 1.5).max(0.0);
            }
        }

        let growth_speed = 30.0 * io_rate.max(0.05); // minimum crawl even with no I/O

        // Grow existing branches
        let mut new_branches: Vec<Branch> = Vec::new();
        for branch in &mut self.branches {
            if !branch.growing {
                continue;
            }

            let target = branch.target_length();
            branch.growth_progress += growth_speed * dt / target;
            branch.growth_progress = branch.growth_progress.min(1.0);

            let len = target * branch.growth_progress;
            branch.end = (
                branch.start.0 + branch.angle.cos() * len,
                branch.start.1 + branch.angle.sin() * len,
            );

            // Branch complete — try spawning children
            if branch.growth_progress >= 1.0 {
                branch.growing = false;
                branch.has_node = true;
            }
        }

        // Spawn children from completed branches
        let total_branches = self.branches.len();
        for i in 0..total_branches {
            let branch = &self.branches[i];
            if branch.growing || !branch.has_node || branch.depth >= MAX_DEPTH {
                continue;
            }
            if branch.length() < MIN_BRANCH_LEN {
                continue;
            }
            if total_branches + new_branches.len() >= MAX_BRANCHES {
                break;
            }

            // Already spawned children? Check if any branch starts at our end.
            let end = self.branches[i].end;
            let already_spawned = self.branches.iter().any(|b| {
                let dx = b.start.0 - end.0;
                let dy = b.start.1 - end.1;
                (dx * dx + dy * dy) < 1.0 && !std::ptr::eq(b, &self.branches[i])
            }) || new_branches.iter().any(|b: &Branch| {
                let dx = b.start.0 - end.0;
                let dy = b.start.1 - end.1;
                (dx * dx + dy * dy) < 1.0
            });

            if already_spawned {
                continue;
            }

            let depth = self.branches[i].depth;
            let angle = self.branches[i].angle;
            let thickness = self.branches[i].thickness;

            // 1-3 children per node
            let n_children = 1 + self.rng.gen_range(0u32..3);
            for _ in 0..n_children {
                if total_branches + new_branches.len() >= MAX_BRANCHES {
                    break;
                }
                let fork_angle = self.rng.gen_range(15.0f32..45.0).to_radians();
                let sign = if self.rng.gen_bool(0.5) { 1.0 } else { -1.0 };
                let child_angle = angle + fork_angle * sign;
                let child_thickness = (thickness * 0.75).max(0.5);
                new_branches.push(Branch::new(end, child_angle, child_thickness, depth + 1));
            }
        }

        self.branches.extend(new_branches);
    }

    /// Pulse all nodes simultaneously (call on derivation completion).
    pub fn pulse(&mut self) {
        self.global_pulse = 1.0;
        for branch in &mut self.branches {
            if branch.has_node {
                branch.node_brightness = 1.0;
            }
        }
    }

    /// Begin contraction toward center (for final kexec animation).
    /// Call each frame with increasing `progress` from 0.0 to 1.0.
    pub fn contract(&mut self, progress: f32) {
        self.contraction = progress.clamp(0.0, 1.0);
    }

    /// Render the network to a pixel buffer (XRGB8888 format, row-major).
    /// The buffer must be `width * height` u32 values.
    pub fn render(&self, buffer: &mut [u32]) {
        let w = self.width as usize;
        let h = self.height as usize;
        assert!(buffer.len() >= w * h, "buffer too small");

        // Clear to background (MOSS_DEEP or black depending on elapsed time)
        let bg = if self.elapsed < 1.0 {
            Rgba::lerp(crate::color::BLACK, MOSS_DEEP, self.elapsed)
        } else {
            MOSS_DEEP
        };

        // White flash during final contraction
        let bg = if self.contraction > 0.95 {
            let flash = ((self.contraction - 0.95) / 0.05).clamp(0.0, 1.0);
            Rgba::lerp(bg, MYCELIAL_WHITE, flash)
        } else {
            bg
        };

        // Fade to black after flash
        let bg_val = bg.to_xrgb8888();
        for pixel in buffer.iter_mut().take(w * h) {
            *pixel = bg_val;
        }

        // Don't draw during initial blackout or after contraction flash
        if self.elapsed < 1.0 || self.contraction > 0.98 {
            // During T=0-1s, draw single teal pixel at center after 1s
            if self.elapsed >= 0.9 && self.elapsed < 1.0 {
                let cx = self.center.0 as usize;
                let cy = self.center.1 as usize;
                if cx < w && cy < h {
                    buffer[cy * w + cx] = LEAF_GREEN.to_xrgb8888();
                }
            }
            return;
        }

        // Draw all branches
        for branch in &self.branches {
            if branch.growth_progress < 0.01 {
                continue;
            }

            let (start, end) = if self.contraction > 0.0 {
                // Contract toward center
                let c = self.contraction;
                let s = (
                    branch.start.0 + (self.center.0 - branch.start.0) * c,
                    branch.start.1 + (self.center.1 - branch.start.1) * c,
                );
                let e = (
                    branch.end.0 + (self.center.0 - branch.end.0) * c,
                    branch.end.1 + (self.center.1 - branch.end.1) * c,
                );
                (s, e)
            } else {
                (branch.start, branch.end)
            };

            // Color based on depth
            let color = match branch.depth {
                0 => LEAF_GREEN,
                1..=3 => Rgba::lerp(LEAF_GREEN, LICHEN_GREY, branch.depth as f32 * 0.15),
                _ => LICHEN_GREY,
            };

            // Brightness boost during global pulse
            let color = if self.global_pulse > 0.0 {
                color.brighten(1.0 + self.global_pulse * 0.8)
            } else {
                color
            };

            // Apply opacity based on depth
            let opacity = branch.opacity * (1.0 - branch.depth as f32 * 0.06).max(0.3);
            let color = color.with_opacity(opacity);

            // Draw the branch line
            draw_line_thick(
                buffer,
                w,
                h,
                start.0 as i32,
                start.1 as i32,
                end.0 as i32,
                end.1 as i32,
                branch.thickness.max(1.0) as u32,
                color,
            );

            // Draw node if present
            if branch.has_node && branch.length() > MIN_BRANCH_LEN {
                let node_color = if branch.node_brightness > 0.0 {
                    Rgba::lerp(LEAF_GREEN, SOLAR_GOLD, branch.node_brightness)
                        .brighten(1.0 + branch.node_brightness * 0.5)
                } else {
                    LEAF_GREEN.brighten(1.2)
                };
                let radius = (branch.thickness * 1.5 + 1.0) as i32;
                draw_filled_circle(buffer, w, h, end.0 as i32, end.1 as i32, radius, node_color);
            }
        }
    }
}

/// Bresenham's line algorithm.
fn draw_line(
    buffer: &mut [u32],
    buf_w: usize,
    buf_h: usize,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: Rgba,
) {
    let mut x0 = x0;
    let mut y0 = y0;
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    let packed = if color.3 == 0xff {
        color.to_xrgb8888()
    } else {
        // Will need compositing
        0
    };
    let opaque = color.3 == 0xff;

    loop {
        if x0 >= 0 && (x0 as usize) < buf_w && y0 >= 0 && (y0 as usize) < buf_h {
            let idx = y0 as usize * buf_w + x0 as usize;
            if opaque {
                buffer[idx] = packed;
            } else {
                // Alpha composite over existing pixel
                let dst_val = buffer[idx];
                let dst = Rgba(
                    ((dst_val >> 16) & 0xff) as u8,
                    ((dst_val >> 8) & 0xff) as u8,
                    (dst_val & 0xff) as u8,
                    0xff,
                );
                buffer[idx] = color.over(dst).to_xrgb8888();
            }
        }

        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            if x0 == x1 {
                break;
            }
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            if y0 == y1 {
                break;
            }
            err += dx;
            y0 += sy;
        }
    }
}

/// Draw a thick line by drawing multiple parallel Bresenham lines.
fn draw_line_thick(
    buffer: &mut [u32],
    buf_w: usize,
    buf_h: usize,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    thickness: u32,
    color: Rgba,
) {
    if thickness <= 1 {
        draw_line(buffer, buf_w, buf_h, x0, y0, x1, y1, color);
        return;
    }

    let half = thickness as i32 / 2;
    let dx = (x1 - x0) as f32;
    let dy = (y1 - y0) as f32;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 0.001 {
        return;
    }
    // Perpendicular direction
    let nx = -dy / len;
    let ny = dx / len;

    for i in -half..=half {
        let ox = (nx * i as f32).round() as i32;
        let oy = (ny * i as f32).round() as i32;
        draw_line(
            buffer,
            buf_w,
            buf_h,
            x0 + ox,
            y0 + oy,
            x1 + ox,
            y1 + oy,
            color,
        );
    }
}

/// Draw a filled circle using the midpoint circle algorithm.
fn draw_filled_circle(
    buffer: &mut [u32],
    buf_w: usize,
    buf_h: usize,
    cx: i32,
    cy: i32,
    radius: i32,
    color: Rgba,
) {
    let packed = color.to_xrgb8888();
    for dy in -radius..=radius {
        let y = cy + dy;
        if y < 0 || y as usize >= buf_h {
            continue;
        }
        let half_w = ((radius * radius - dy * dy) as f32).sqrt() as i32;
        let x_start = (cx - half_w).max(0) as usize;
        let x_end = ((cx + half_w) as usize).min(buf_w - 1);
        let row = y as usize * buf_w;
        for x in x_start..=x_end {
            buffer[row + x] = packed;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let net = MycelialNetwork::new(800, 600, "test genesis phrase");
        assert!(!net.branches.is_empty());
        assert!(net.branches.len() <= 9); // 4-8 initial branches
    }

    #[test]
    fn test_deterministic_seeding() {
        let a = MycelialNetwork::new(800, 600, "sovereign quickening");
        let b = MycelialNetwork::new(800, 600, "sovereign quickening");
        assert_eq!(a.branches.len(), b.branches.len());
        for (ba, bb) in a.branches.iter().zip(b.branches.iter()) {
            assert_eq!(ba.angle, bb.angle);
        }
    }

    #[test]
    fn test_different_phrases_differ() {
        let a = MycelialNetwork::new(800, 600, "phrase one");
        let b = MycelialNetwork::new(800, 600, "phrase two");
        // Angles should differ
        assert_ne!(a.branches[0].angle, b.branches[0].angle);
    }

    #[test]
    fn test_growth() {
        let mut net = MycelialNetwork::new(800, 600, "grow test");
        let initial = net.branches.len();
        for _ in 0..100 {
            net.grow(0.1, 1.0);
        }
        assert!(net.branches.len() > initial);
    }

    #[test]
    fn test_max_branches_cap() {
        let mut net = MycelialNetwork::new(200, 200, "cap test");
        for _ in 0..2000 {
            net.grow(0.05, 1.0);
        }
        assert!(net.branches.len() <= MAX_BRANCHES);
    }

    #[test]
    fn test_render_no_panic() {
        let mut net = MycelialNetwork::new(100, 80, "render test");
        net.grow(0.5, 1.0);
        net.elapsed = 2.0;
        let mut buf = vec![0u32; 100 * 80];
        net.render(&mut buf);
    }

    #[test]
    fn test_pulse_sets_brightness() {
        let mut net = MycelialNetwork::new(100, 100, "pulse test");
        for _ in 0..50 {
            net.grow(0.1, 1.0);
        }
        net.pulse();
        assert_eq!(net.global_pulse, 1.0);
        let any_bright = net.branches.iter().any(|b| b.node_brightness > 0.0);
        // Only branches with nodes should be bright
        let any_node = net.branches.iter().any(|b| b.has_node);
        assert_eq!(any_bright, any_node);
    }

    #[test]
    fn test_contraction() {
        let mut net = MycelialNetwork::new(100, 100, "contract test");
        net.contract(0.5);
        assert!((net.contraction - 0.5).abs() < f32::EPSILON);
        net.contract(1.5); // clamp
        assert!((net.contraction - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bresenham_horizontal() {
        let mut buf = vec![0u32; 10 * 10];
        draw_line(&mut buf, 10, 10, 2, 5, 7, 5, LEAF_GREEN);
        // Pixels from x=2 to x=7 on row 5 should be set
        for x in 2..=7 {
            assert_ne!(buf[5 * 10 + x], 0);
        }
    }

    #[test]
    fn test_bresenham_out_of_bounds() {
        let mut buf = vec![0u32; 10 * 10];
        // Should not panic even with coords outside the buffer
        draw_line(&mut buf, 10, 10, -5, -5, 15, 15, LEAF_GREEN);
    }

    #[test]
    fn test_filled_circle() {
        let mut buf = vec![0u32; 20 * 20];
        draw_filled_circle(&mut buf, 20, 20, 10, 10, 3, SOLAR_GOLD);
        // Center pixel should be set
        assert_ne!(buf[10 * 20 + 10], 0);
    }
}
