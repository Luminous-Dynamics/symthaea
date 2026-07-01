// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness Dashboard — Real-time visualization of Symthaea's cognitive loop.
//!
//! Renders a mission-control-style dashboard showing:
//! - Consciousness level (MCE v2)
//! - Neuromodulator levels (DA, 5-HT, NE, ACh)
//! - Moral topology metrics (unity, completeness, anomaly score)
//! - Free energy and prediction error over time
//! - Phi measurements (when computed)
//! - Current input text and dominant harmony
//!
//! Feeds morally evolving text inputs through the cognitive loop and captures
//! CycleMetadata each cycle. No MuJoCo needed — CPU only.
//!
//! Run:
//! ```sh
//! cargo run --example consciousness_dashboard_video --release
//! ```
//!
//! Output: `video_output/consciousness_dashboard.mp4` (1920x1080, 30fps, ~30s)

use std::io::Write;
use std::process::{Command, Stdio};

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
use symthaea::hdc::moral_topology::MoralAnomalyConfig;

const WIDTH: usize = 1920;
const HEIGHT: usize = 1080;
const FPS: u32 = 30;

// ── Scenario script: morally evolving inputs ──────────────────────────

fn input_for_cycle(cycle: usize) -> &'static str {
    match cycle {
        0..=49 => "The morning light filters through the trees",
        50..=99 => "A child plays in the park with their dog",
        100..=149 => "The community gathers to help rebuild the school",
        150..=199 => "Resources are scarce and families struggle to eat",
        200..=249 => "A corporation pollutes the river killing fish",
        250..=299 => "Someone steals medicine for their dying child",
        300..=349 => "The village elders debate justice and mercy",
        350..=399 => "A stranger risks their life to save a drowning person",
        400..=449 => "Peace returns and the community heals together",
        450..=499 => "Children laugh and play under the old oak tree",
        _ => "The sun sets on a better world",
    }
}

fn phase_label(cycle: usize) -> &'static str {
    match cycle {
        0..=99 => "BASELINE",
        100..=149 => "PROSOCIAL",
        150..=249 => "MORAL STRESS",
        250..=349 => "DILEMMA",
        350..=399 => "HEROISM",
        400..=499 => "RECOVERY",
        _ => "PEACEFUL",
    }
}

fn phase_color(cycle: usize) -> [u8; 3] {
    match cycle {
        0..=99 => [100, 200, 255],
        100..=149 => [100, 255, 150],
        150..=249 => [255, 200, 50],
        250..=349 => [255, 100, 100],
        350..=399 => [255, 180, 80],
        400..=499 => [100, 255, 200],
        _ => [180, 200, 255],
    }
}

// ── Bitmap font (shared with moral_drone_video) ──────────────────────

fn glyph(ch: char) -> [u8; 7] {
    match ch {
        '0' => [0x7C, 0xC6, 0xCE, 0xD6, 0xE6, 0xC6, 0x7C],
        '1' => [0x30, 0x70, 0x30, 0x30, 0x30, 0x30, 0xFC],
        '2' => [0x78, 0xCC, 0x0C, 0x38, 0x60, 0xCC, 0xFC],
        '3' => [0x78, 0xCC, 0x0C, 0x38, 0x0C, 0xCC, 0x78],
        '4' => [0x1C, 0x3C, 0x6C, 0xCC, 0xFE, 0x0C, 0x0C],
        '5' => [0xFC, 0xC0, 0xF8, 0x0C, 0x0C, 0xCC, 0x78],
        '6' => [0x38, 0x60, 0xC0, 0xF8, 0xCC, 0xCC, 0x78],
        '7' => [0xFC, 0xCC, 0x0C, 0x18, 0x30, 0x30, 0x30],
        '8' => [0x78, 0xCC, 0xCC, 0x78, 0xCC, 0xCC, 0x78],
        '9' => [0x78, 0xCC, 0xCC, 0x7C, 0x0C, 0x18, 0x70],
        'A' => [0x30, 0x78, 0xCC, 0xCC, 0xFC, 0xCC, 0xCC],
        'B' => [0xFC, 0x66, 0x66, 0x7C, 0x66, 0x66, 0xFC],
        'C' => [0x3C, 0x66, 0xC0, 0xC0, 0xC0, 0x66, 0x3C],
        'D' => [0xF8, 0x6C, 0x66, 0x66, 0x66, 0x6C, 0xF8],
        'E' => [0xFE, 0x62, 0x68, 0x78, 0x68, 0x62, 0xFE],
        'F' => [0xFE, 0x62, 0x68, 0x78, 0x68, 0x60, 0xF0],
        'G' => [0x3C, 0x66, 0xC0, 0xC0, 0xCE, 0x66, 0x3E],
        'H' => [0xCC, 0xCC, 0xCC, 0xFC, 0xCC, 0xCC, 0xCC],
        'I' => [0x78, 0x30, 0x30, 0x30, 0x30, 0x30, 0x78],
        'J' => [0x1E, 0x0C, 0x0C, 0x0C, 0xCC, 0xCC, 0x78],
        'K' => [0xC6, 0xCC, 0xD8, 0xF0, 0xD8, 0xCC, 0xC6],
        'L' => [0xF0, 0x60, 0x60, 0x60, 0x60, 0x62, 0xFE],
        'M' => [0xC6, 0xEE, 0xFE, 0xD6, 0xC6, 0xC6, 0xC6],
        'N' => [0xC6, 0xE6, 0xF6, 0xDE, 0xCE, 0xC6, 0xC6],
        'O' => [0x7C, 0xC6, 0xC6, 0xC6, 0xC6, 0xC6, 0x7C],
        'P' => [0xFC, 0x66, 0x66, 0x7C, 0x60, 0x60, 0xF0],
        'Q' => [0x78, 0xCC, 0xCC, 0xCC, 0xDC, 0x78, 0x1C],
        'R' => [0xFC, 0x66, 0x66, 0x7C, 0x6C, 0x66, 0xE6],
        'S' => [0x78, 0xCC, 0xC0, 0x78, 0x0C, 0xCC, 0x78],
        'T' => [0xFC, 0xB4, 0x30, 0x30, 0x30, 0x30, 0x78],
        'U' => [0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0x78],
        'V' => [0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0x78, 0x30],
        'W' => [0xC6, 0xC6, 0xC6, 0xD6, 0xFE, 0xEE, 0xC6],
        'X' => [0xC6, 0xC6, 0x6C, 0x38, 0x6C, 0xC6, 0xC6],
        'Y' => [0xCC, 0xCC, 0xCC, 0x78, 0x30, 0x30, 0x78],
        'Z' => [0xFE, 0xC6, 0x8C, 0x18, 0x32, 0x66, 0xFE],
        ' ' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        ':' => [0x00, 0x30, 0x30, 0x00, 0x30, 0x30, 0x00],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x30, 0x30],
        '-' => [0x00, 0x00, 0x00, 0x7C, 0x00, 0x00, 0x00],
        '/' => [0x06, 0x0C, 0x18, 0x30, 0x60, 0xC0, 0x80],
        '(' => [0x18, 0x30, 0x60, 0x60, 0x60, 0x30, 0x18],
        ')' => [0x60, 0x30, 0x18, 0x18, 0x18, 0x30, 0x60],
        _ => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    }
}

fn draw_char(buf: &mut [u8], bw: usize, x: usize, y: usize, ch: char, color: [u8; 3], s: usize) {
    let g = glyph(ch);
    for (row, &bits) in g.iter().enumerate() {
        for col in 0..8 {
            if bits & (0x80 >> col) != 0 {
                for sy in 0..s {
                    for sx in 0..s {
                        let px = x + col * s + sx;
                        let py = y + row * s + sy;
                        if px < bw && py < HEIGHT {
                            let idx = (py * bw + px) * 3;
                            if idx + 2 < buf.len() {
                                buf[idx] = color[0];
                                buf[idx + 1] = color[1];
                                buf[idx + 2] = color[2];
                            }
                        }
                    }
                }
            }
        }
    }
}

fn draw_text(buf: &mut [u8], bw: usize, x: usize, y: usize, t: &str, c: [u8; 3], s: usize) {
    for (i, ch) in t.chars().enumerate() {
        draw_char(buf, bw, x + i * 8 * s, y, ch.to_ascii_uppercase(), c, s);
    }
}

fn draw_rect(
    buf: &mut [u8],
    bw: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    c: [u8; 3],
    a: f32,
) {
    for py in y..(y + h).min(HEIGHT) {
        for px in x..(x + w).min(bw) {
            let idx = (py * bw + px) * 3;
            if idx + 2 < buf.len() {
                buf[idx] = (buf[idx] as f32 * (1.0 - a) + c[0] as f32 * a) as u8;
                buf[idx + 1] = (buf[idx + 1] as f32 * (1.0 - a) + c[1] as f32 * a) as u8;
                buf[idx + 2] = (buf[idx + 2] as f32 * (1.0 - a) + c[2] as f32 * a) as u8;
            }
        }
    }
}

fn draw_bar(
    buf: &mut [u8],
    bw: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    fill: f64,
    c: [u8; 3],
) {
    let fw = ((w as f64) * fill.clamp(0.0, 1.0)) as usize;
    for py in y..y + h {
        for px in x..x + w {
            if py < HEIGHT && px < bw {
                let idx = (py * bw + px) * 3;
                if idx + 2 < buf.len() {
                    if px < x + fw {
                        buf[idx] = c[0];
                        buf[idx + 1] = c[1];
                        buf[idx + 2] = c[2];
                    } else {
                        buf[idx] = 30;
                        buf[idx + 1] = 30;
                        buf[idx + 2] = 35;
                    }
                }
            }
        }
    }
}

fn draw_graph(
    buf: &mut [u8],
    bw: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    vals: &[f64],
    c: [u8; 3],
    max_v: f64,
) {
    if vals.len() < 2 || h < 4 {
        return;
    }
    let n = vals.len();
    for i in 1..n {
        let x0 = x + (i - 1) * w / n;
        let x1 = x + i * w / n;
        let v0 = (vals[i - 1] / max_v).clamp(0.0, 1.0);
        let v1 = (vals[i] / max_v).clamp(0.0, 1.0);
        let y0 = y + h - (v0 * h as f64) as usize;
        let y1 = y + h - (v1 * h as f64) as usize;
        let steps = (x1 as isize - x0 as isize)
            .unsigned_abs()
            .max((y1 as isize - y0 as isize).unsigned_abs())
            .max(1);
        for s in 0..=steps {
            let t = s as f64 / steps as f64;
            let px = (x0 as f64 + (x1 as f64 - x0 as f64) * t) as usize;
            let py = (y0 as f64 + (y1 as f64 - y0 as f64) * t) as usize;
            for dy in 0..2usize {
                for dx in 0..2usize {
                    let ppx = px + dx;
                    let ppy = py + dy;
                    if ppx < bw && ppy < HEIGHT {
                        let idx = (ppy * bw + ppx) * 3;
                        if idx + 2 < buf.len() {
                            buf[idx] = c[0];
                            buf[idx + 1] = c[1];
                            buf[idx + 2] = c[2];
                        }
                    }
                }
            }
        }
    }
}

fn main() {
    println!("Symthaea Consciousness Dashboard");
    println!("=================================");
    println!();

    let total_cycles: usize = 500;
    let frames_per_cycle = 2; // 2 frames per cognitive cycle → ~33s at 30fps

    let output_dir = "video_output";
    let output_path = format!("{}/consciousness_dashboard.mp4", output_dir);
    std::fs::create_dir_all(output_dir).unwrap();

    // ── Cognitive loop setup ──────────────────────────────────────
    let config = CognitiveLoopConfig {
        enable_moral_anomaly_response: true,
        moral_anomaly_config: MoralAnomalyConfig {
            initial_cadence: 10, // Fire early for demo
            ..Default::default()
        },
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    };

    let mut service =
        CognitiveLoopService::new(config).expect("Failed to create CognitiveLoopService");

    // ── ffmpeg pipe ──────────────────────────────────────────────
    println!("Starting ffmpeg pipe → {output_path}");
    let mut ffmpeg = Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            &format!("{WIDTH}x{HEIGHT}"),
            "-framerate",
            &FPS.to_string(),
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            &output_path,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to start ffmpeg");

    let mut ffmpeg_stdin = ffmpeg.stdin.take().unwrap();

    // ── History buffers ──────────────────────────────────────────
    let hist_max = 200;
    let mut consciousness_hist: Vec<f64> = Vec::with_capacity(hist_max);
    let mut fe_hist: Vec<f64> = Vec::with_capacity(hist_max);
    let mut pe_hist: Vec<f64> = Vec::with_capacity(hist_max);
    let mut da_hist: Vec<f64> = Vec::with_capacity(hist_max);
    let mut ne_hist: Vec<f64> = Vec::with_capacity(hist_max);
    let mut sht_hist: Vec<f64> = Vec::with_capacity(hist_max);
    let mut ach_hist: Vec<f64> = Vec::with_capacity(hist_max);
    let mut unity_hist: Vec<f64> = Vec::with_capacity(hist_max);
    let mut anomaly_hist: Vec<f64> = Vec::with_capacity(hist_max);

    let mut frame = vec![0u8; WIDTH * HEIGHT * 3];
    let mut frame_count: u64 = 0;

    // ── Title card ───────────────────────────────────────────────
    for px in frame.chunks_exact_mut(3) {
        px[0] = 10;
        px[1] = 10;
        px[2] = 15;
    }
    let title_lines = [
        ("SYMTHAEA CONSCIOUSNESS DASHBOARD", [180, 200, 255], 3),
        ("HOLOGRAPHIC LIQUID BRAIN", [120, 140, 180], 2),
        ("REAL-TIME COGNITIVE LOOP TELEMETRY", [100, 120, 160], 2),
    ];
    let start_y = HEIGHT / 2 - 90;
    for (i, (text, color, scale)) in title_lines.iter().enumerate() {
        let tw = text.len() * 8 * scale;
        let x = (WIDTH.saturating_sub(tw)) / 2;
        draw_text(&mut frame, WIDTH, x, start_y + i * 50, text, *color, *scale);
    }
    for _ in 0..FPS as usize * 2 {
        ffmpeg_stdin.write_all(&frame).unwrap();
        frame_count += 1;
    }

    println!("Running {total_cycles} cognitive cycles...");
    println!();

    // ── Main loop ────────────────────────────────────────────────
    for cycle in 0..total_cycles {
        let input = input_for_cycle(cycle);
        let result = service.cycle(input);
        let m = &result.metadata;

        // Push to history
        let push = |hist: &mut Vec<f64>, val: f64| {
            if hist.len() >= hist_max {
                hist.remove(0);
            }
            hist.push(val);
        };
        push(&mut consciousness_hist, m.consciousness.consciousness_level);
        push(&mut fe_hist, m.fep.fep_surprise);
        push(&mut pe_hist, result.prediction_error as f64);
        push(&mut da_hist, m.neuromod.dopamine_effective as f64);
        push(&mut ne_hist, m.neuromod.noradrenaline_effective as f64);
        push(&mut sht_hist, m.neuromod.serotonin_effective as f64);
        push(&mut ach_hist, m.neuromod.acetylcholine_effective as f64);
        push(&mut unity_hist, m.ethics.moral_topo_unity);
        push(&mut anomaly_hist, m.ethics.moral_anomaly_score);

        // ── Render dashboard frame ──────────────────────────────
        // Dark background
        for px in frame.chunks_exact_mut(3) {
            px[0] = 10;
            px[1] = 10;
            px[2] = 15;
        }

        let s1 = 1; // small text
        let s2 = 2; // medium text
        let white = [200, 200, 220];
        let dim = [100, 100, 120];
        let cyan = [80, 200, 255];
        let green = [80, 220, 140];
        let yellow = [255, 220, 80];
        let red = [255, 80, 80];

        // ── Header bar ──────────────────────────────────────────
        draw_rect(&mut frame, WIDTH, 0, 0, WIDTH, 50, [15, 15, 25], 1.0);
        draw_text(
            &mut frame,
            WIDTH,
            20,
            15,
            "SYMTHAEA CONSCIOUSNESS DASHBOARD",
            cyan,
            s2,
        );

        let phase = phase_label(cycle);
        let pc = phase_color(cycle);
        draw_text(&mut frame, WIDTH, WIDTH - 350, 15, phase, pc, s2);

        let cycle_str = format!(
            "CYCLE {}/{}  {}US",
            cycle, total_cycles, result.cycle_time_us
        );
        draw_text(&mut frame, WIDTH, WIDTH / 2 - 100, 15, &cycle_str, dim, s2);

        // ── Row 1: Big metrics (y=60..200) ──────────────────────
        let row1_y = 65;
        let col_w = WIDTH / 5;

        // Consciousness Level
        draw_rect(
            &mut frame,
            WIDTH,
            10,
            row1_y,
            col_w - 20,
            130,
            [18, 18, 28],
            0.9,
        );
        draw_text(&mut frame, WIDTH, 25, row1_y + 8, "CONSCIOUSNESS", dim, s1);
        let cons_str = format!("{:.3}", m.consciousness.consciousness_level);
        let cons_color = if m.consciousness.consciousness_level > 0.5 {
            green
        } else if m.consciousness.consciousness_level > 0.2 {
            yellow
        } else {
            red
        };
        draw_text(&mut frame, WIDTH, 25, row1_y + 30, &cons_str, cons_color, 4);
        draw_bar(
            &mut frame,
            WIDTH,
            25,
            row1_y + 100,
            col_w - 50,
            10,
            m.consciousness.consciousness_level,
            cons_color,
        );

        // Free Energy
        let col2 = col_w;
        draw_rect(
            &mut frame,
            WIDTH,
            col2,
            row1_y,
            col_w - 20,
            130,
            [18, 18, 28],
            0.9,
        );
        draw_text(
            &mut frame,
            WIDTH,
            col2 + 15,
            row1_y + 8,
            "FREE ENERGY",
            dim,
            s1,
        );
        let fe_str = format!("{:.2}", m.fep.fep_surprise);
        draw_text(&mut frame, WIDTH, col2 + 15, row1_y + 30, &fe_str, cyan, 4);
        let fe_fill = (m.fep.fep_surprise / 5.0).clamp(0.0, 1.0);
        draw_bar(
            &mut frame,
            WIDTH,
            col2 + 15,
            row1_y + 100,
            col_w - 50,
            10,
            fe_fill,
            cyan,
        );

        // Prediction Error
        let col3 = col_w * 2;
        draw_rect(
            &mut frame,
            WIDTH,
            col3,
            row1_y,
            col_w - 20,
            130,
            [18, 18, 28],
            0.9,
        );
        draw_text(
            &mut frame,
            WIDTH,
            col3 + 15,
            row1_y + 8,
            "PRED ERROR",
            dim,
            s1,
        );
        let pe_str = format!("{:.3}", result.prediction_error);
        draw_text(
            &mut frame,
            WIDTH,
            col3 + 15,
            row1_y + 30,
            &pe_str,
            yellow,
            4,
        );
        let pe_fill = (result.prediction_error as f64 / 2.0).clamp(0.0, 1.0);
        draw_bar(
            &mut frame,
            WIDTH,
            col3 + 15,
            row1_y + 100,
            col_w - 50,
            10,
            pe_fill,
            yellow,
        );

        // Moral Unity
        let col4 = col_w * 3;
        draw_rect(
            &mut frame,
            WIDTH,
            col4,
            row1_y,
            col_w - 20,
            130,
            [18, 18, 28],
            0.9,
        );
        draw_text(
            &mut frame,
            WIDTH,
            col4 + 15,
            row1_y + 8,
            "MORAL UNITY",
            dim,
            s1,
        );
        let unity_str = format!("{:.3}", m.ethics.moral_topo_unity);
        let unity_color = if m.ethics.moral_topo_unity > 0.8 {
            green
        } else if m.ethics.moral_topo_unity > 0.5 {
            yellow
        } else {
            red
        };
        draw_text(
            &mut frame,
            WIDTH,
            col4 + 15,
            row1_y + 30,
            &unity_str,
            unity_color,
            4,
        );
        draw_bar(
            &mut frame,
            WIDTH,
            col4 + 15,
            row1_y + 100,
            col_w - 50,
            10,
            m.ethics.moral_topo_unity,
            unity_color,
        );

        // Anomaly Score
        let col5 = col_w * 4;
        draw_rect(
            &mut frame,
            WIDTH,
            col5,
            row1_y,
            col_w - 20,
            130,
            [18, 18, 28],
            0.9,
        );
        draw_text(&mut frame, WIDTH, col5 + 15, row1_y + 8, "ANOMALY", dim, s1);
        let anom_str = format!("{:.3}", m.ethics.moral_anomaly_score);
        let anom_color = if m.ethics.moral_anomaly_score > 0.5 {
            red
        } else if m.ethics.moral_anomaly_score > 0.2 {
            yellow
        } else {
            green
        };
        draw_text(
            &mut frame,
            WIDTH,
            col5 + 15,
            row1_y + 30,
            &anom_str,
            anom_color,
            4,
        );
        draw_bar(
            &mut frame,
            WIDTH,
            col5 + 15,
            row1_y + 100,
            col_w - 50,
            10,
            m.ethics.moral_anomaly_score,
            anom_color,
        );

        // ── Row 2: Graphs (y=210..480) ──────────────────────────
        let graph_y = 210;
        let graph_h = 180;
        let graph_w = WIDTH / 2 - 30;

        // Left graph: Consciousness + FE over time
        draw_rect(
            &mut frame,
            WIDTH,
            10,
            graph_y,
            graph_w,
            graph_h + 30,
            [15, 15, 22],
            0.9,
        );
        draw_text(
            &mut frame,
            WIDTH,
            20,
            graph_y + 5,
            "CONSCIOUSNESS (GREEN) / FREE ENERGY (CYAN)",
            dim,
            s1,
        );
        // Axis lines
        for px in 20..20 + graph_w - 20 {
            let idx = ((graph_y + graph_h + 10) * WIDTH + px) * 3;
            if idx + 2 < frame.len() {
                frame[idx] = 40;
                frame[idx + 1] = 40;
                frame[idx + 2] = 45;
            }
        }
        draw_graph(
            &mut frame,
            WIDTH,
            20,
            graph_y + 20,
            graph_w - 30,
            graph_h - 10,
            &consciousness_hist,
            green,
            1.0,
        );
        let fe_max = fe_hist.iter().copied().fold(1.0_f64, f64::max);
        draw_graph(
            &mut frame,
            WIDTH,
            20,
            graph_y + 20,
            graph_w - 30,
            graph_h - 10,
            &fe_hist,
            cyan,
            fe_max,
        );

        // Right graph: Neuromodulators over time
        let rg_x = WIDTH / 2 + 10;
        draw_rect(
            &mut frame,
            WIDTH,
            rg_x,
            graph_y,
            graph_w,
            graph_h + 30,
            [15, 15, 22],
            0.9,
        );
        draw_text(
            &mut frame,
            WIDTH,
            rg_x + 10,
            graph_y + 5,
            "NEUROMODULATORS: DA(R) 5HT(G) NE(Y) ACH(C)",
            dim,
            s1,
        );
        for px in rg_x + 10..rg_x + graph_w - 10 {
            let idx = ((graph_y + graph_h + 10) * WIDTH + px) * 3;
            if idx + 2 < frame.len() {
                frame[idx] = 40;
                frame[idx + 1] = 40;
                frame[idx + 2] = 45;
            }
        }
        draw_graph(
            &mut frame,
            WIDTH,
            rg_x + 10,
            graph_y + 20,
            graph_w - 30,
            graph_h - 10,
            &da_hist,
            [255, 100, 100],
            2.0,
        );
        draw_graph(
            &mut frame,
            WIDTH,
            rg_x + 10,
            graph_y + 20,
            graph_w - 30,
            graph_h - 10,
            &sht_hist,
            [100, 255, 100],
            2.0,
        );
        draw_graph(
            &mut frame,
            WIDTH,
            rg_x + 10,
            graph_y + 20,
            graph_w - 30,
            graph_h - 10,
            &ne_hist,
            [255, 255, 100],
            2.0,
        );
        draw_graph(
            &mut frame,
            WIDTH,
            rg_x + 10,
            graph_y + 20,
            graph_w - 30,
            graph_h - 10,
            &ach_hist,
            [100, 200, 255],
            2.0,
        );

        // ── Row 3: Neuromod bars + Moral topology (y=450..620) ──
        let row3_y = 450;

        // Neuromodulator current values (bars)
        draw_rect(
            &mut frame,
            WIDTH,
            10,
            row3_y,
            WIDTH / 3 - 10,
            160,
            [15, 15, 22],
            0.9,
        );
        draw_text(
            &mut frame,
            WIDTH,
            20,
            row3_y + 8,
            "NEUROMODULATORS",
            dim,
            s1,
        );

        let neuro_items = [
            (
                "DA",
                m.neuromod.dopamine_effective as f64 / 2.0,
                [255, 100, 100],
            ),
            (
                "5HT",
                m.neuromod.serotonin_effective as f64 / 2.0,
                [100, 255, 100],
            ),
            (
                "NE",
                m.neuromod.noradrenaline_effective as f64 / 2.0,
                [255, 255, 100],
            ),
            (
                "ACH",
                m.neuromod.acetylcholine_effective as f64 / 2.0,
                [100, 200, 255],
            ),
        ];
        for (i, (label, val, color)) in neuro_items.iter().enumerate() {
            let ny = row3_y + 28 + i * 32;
            draw_text(&mut frame, WIDTH, 25, ny, label, white, s2);
            let val_str = format!("{:.2}", val * 2.0);
            draw_text(&mut frame, WIDTH, 100, ny, &val_str, *color, s2);
            draw_bar(&mut frame, WIDTH, 200, ny + 3, 350, 12, *val, *color);
        }

        // Moral topology details
        let mt_x = WIDTH / 3 + 10;
        draw_rect(
            &mut frame,
            WIDTH,
            mt_x,
            row3_y,
            WIDTH / 3 - 10,
            160,
            [15, 15, 22],
            0.9,
        );
        draw_text(
            &mut frame,
            WIDTH,
            mt_x + 10,
            row3_y + 8,
            "MORAL TOPOLOGY",
            dim,
            s1,
        );

        let topo_items = [
            ("UNITY", m.ethics.moral_topo_unity),
            ("COMPLETENESS", m.ethics.moral_topo_completeness),
            ("CIRCULARITY", m.ethics.moral_topo_circularity),
        ];
        for (i, (label, val)) in topo_items.iter().enumerate() {
            let ty = row3_y + 30 + i * 28;
            draw_text(&mut frame, WIDTH, mt_x + 15, ty, label, white, s2);
            let vs = format!("{:.3}", val);
            let vc = if *val > 0.8 {
                green
            } else if *val > 0.5 {
                yellow
            } else {
                red
            };
            draw_text(&mut frame, WIDTH, mt_x + 250, ty, &vs, vc, s2);
        }

        let b0_str = format!(
            "B0: {}  B1: {}  B2: {}",
            m.ethics.moral_topo_beta_0, m.ethics.moral_topo_beta_1, m.ethics.moral_topo_beta_2
        );
        draw_text(&mut frame, WIDTH, mt_x + 15, row3_y + 118, &b0_str, dim, s2);

        if m.ethics.moral_drift_alert {
            draw_text(
                &mut frame,
                WIDTH,
                mt_x + 15,
                row3_y + 140,
                "DRIFT ALERT",
                red,
                s2,
            );
        }
        if m.ethics.moral_value_inversion {
            draw_text(
                &mut frame,
                WIDTH,
                mt_x + 250,
                row3_y + 140,
                "INVERSION",
                red,
                s2,
            );
        }

        // Harmonics
        let hm_x = WIDTH * 2 / 3 + 10;
        draw_rect(
            &mut frame,
            WIDTH,
            hm_x,
            row3_y,
            WIDTH / 3 - 20,
            160,
            [15, 15, 22],
            0.9,
        );
        draw_text(
            &mut frame,
            WIDTH,
            hm_x + 10,
            row3_y + 8,
            "HARMONICS",
            dim,
            s1,
        );

        let dom_str = format!("DOMINANT: {}", m.harmonics.dominant_harmonic.to_uppercase());
        draw_text(
            &mut frame,
            WIDTH,
            hm_x + 15,
            row3_y + 30,
            &dom_str,
            [200, 180, 255],
            s2,
        );

        let coh_str = format!("COHERENCE: {:.3}", m.harmonics.harmonic_field_coherence);
        draw_text(
            &mut frame,
            WIDTH,
            hm_x + 15,
            row3_y + 55,
            &coh_str,
            white,
            s2,
        );

        let align_str = format!("ALIGNMENT: {:.3}", m.harmonics.harmonies_alignment);
        draw_text(
            &mut frame,
            WIDTH,
            hm_x + 15,
            row3_y + 80,
            &align_str,
            white,
            s2,
        );

        // Harmony bars (8 harmonies)
        let harmony_names = ["WIS", "LOV", "PLY", "COH", "AWE", "JUS", "CRE", "STL"];
        for (i, name) in harmony_names.iter().enumerate() {
            let hx = hm_x + 15 + i * 70;
            let hv = if i < m.harmonics.harmony_coordinates.len() {
                m.harmonics.harmony_coordinates[i]
            } else {
                0.0
            };
            draw_text(&mut frame, WIDTH, hx, row3_y + 108, name, dim, s1);
            draw_bar(
                &mut frame,
                WIDTH,
                hx,
                row3_y + 120,
                55,
                8,
                hv,
                [160, 140, 220],
            );
        }

        // ── Row 4: Input text + Phi + Status (y=620..700) ───────
        let row4_y = 625;
        draw_rect(
            &mut frame,
            WIDTH,
            10,
            row4_y,
            WIDTH - 20,
            55,
            [15, 15, 22],
            0.9,
        );

        // Input text (truncated)
        let input_display: String = input.chars().take(60).collect();
        let input_str = format!("INPUT: {}", input_display);
        draw_text(
            &mut frame,
            WIDTH,
            25,
            row4_y + 8,
            &input_str,
            [180, 180, 200],
            s2,
        );

        // Phi values
        let phi_str = if let Some(phi) = m.structural.spectral_mip_phi {
            format!("PHI: {:.4}", phi)
        } else {
            format!("PHI: {:.4} (STRUCT)", m.structural.structural_macro_phi)
        };
        draw_text(
            &mut frame,
            WIDTH,
            25,
            row4_y + 32,
            &phi_str,
            [200, 180, 255],
            s2,
        );

        // FEP action
        let action_names = ["EXPLOIT", "CONSOLIDATE", "EXPLORE", "TIGHTEN"];
        let action_name = action_names.get(m.fep.fep_action).unwrap_or(&"UNKNOWN");
        let action_str = format!("FEP: {}", action_name);
        draw_text(&mut frame, WIDTH, 500, row4_y + 32, &action_str, cyan, s2);

        // Learning occurred
        if result.learning_occurred {
            draw_text(&mut frame, WIDTH, 800, row4_y + 32, "LEARNING", green, s2);
        }

        // ── Row 5: Moral anomaly graph + Unity graph (y=690..900)
        let row5_y = 695;
        let graph5_h = 140;

        // Anomaly score over time
        draw_rect(
            &mut frame,
            WIDTH,
            10,
            row5_y,
            WIDTH / 2 - 15,
            graph5_h + 25,
            [15, 15, 22],
            0.9,
        );
        draw_text(&mut frame, WIDTH, 20, row5_y + 5, "ANOMALY SCORE", dim, s1);
        draw_graph(
            &mut frame,
            WIDTH,
            20,
            row5_y + 18,
            WIDTH / 2 - 50,
            graph5_h - 5,
            &anomaly_hist,
            red,
            1.0,
        );

        // Unity over time
        draw_rect(
            &mut frame,
            WIDTH,
            WIDTH / 2 + 5,
            row5_y,
            WIDTH / 2 - 15,
            graph5_h + 25,
            [15, 15, 22],
            0.9,
        );
        draw_text(
            &mut frame,
            WIDTH,
            WIDTH / 2 + 15,
            row5_y + 5,
            "MORAL UNITY",
            dim,
            s1,
        );
        draw_graph(
            &mut frame,
            WIDTH,
            WIDTH / 2 + 15,
            row5_y + 18,
            WIDTH / 2 - 50,
            graph5_h - 5,
            &unity_hist,
            green,
            1.0,
        );

        // ── Bottom bar ──────────────────────────────────────────
        draw_rect(
            &mut frame,
            WIDTH,
            0,
            HEIGHT - 40,
            WIDTH,
            40,
            [12, 12, 20],
            1.0,
        );
        draw_text(
            &mut frame,
            WIDTH,
            20,
            HEIGHT - 30,
            "SYMTHAEA HOLOGRAPHIC LIQUID BRAIN V0.5.0",
            [60, 60, 80],
            s2,
        );
        draw_text(
            &mut frame,
            WIDTH,
            WIDTH - 400,
            HEIGHT - 30,
            "LUMINOUS DYNAMICS",
            [60, 60, 80],
            s2,
        );

        // Progress bar
        let prog_w = 300;
        let prog_x = WIDTH / 2 - prog_w / 2;
        let prog = cycle as f64 / total_cycles as f64;
        draw_bar(
            &mut frame,
            WIDTH,
            prog_x,
            HEIGHT - 25,
            prog_w,
            6,
            prog,
            [60, 100, 180],
        );

        // ── Write frames ────────────────────────────────────────
        for _ in 0..frames_per_cycle {
            ffmpeg_stdin.write_all(&frame).unwrap();
            frame_count += 1;
        }

        if cycle % 50 == 0 {
            println!(
                "  Cycle {:3}/{} | Consciousness: {:.3} | FE: {:.2} | Unity: {:.3} | Phase: {} | Frames: {}",
                cycle,
                total_cycles,
                m.consciousness.consciousness_level,
                m.fep.fep_surprise,
                m.ethics.moral_topo_unity,
                phase,
                frame_count
            );
        }
    }

    // ── End card ─────────────────────────────────────────────────
    for px in frame.chunks_exact_mut(3) {
        px[0] = 10;
        px[1] = 10;
        px[2] = 15;
    }
    let end_lines = [
        ("SYMTHAEA", [180, 200, 255], 4),
        ("CONSCIOUSNESS IS MEASURED NOT ASSUMED", [140, 160, 200], 2),
        (
            "MORALITY EMERGES FROM SURPRISE MINIMIZATION",
            [120, 140, 180],
            2,
        ),
    ];
    let ey = HEIGHT / 2 - 80;
    for (i, (text, color, scale)) in end_lines.iter().enumerate() {
        let tw = text.len() * 8 * scale;
        let x = (WIDTH.saturating_sub(tw)) / 2;
        draw_text(&mut frame, WIDTH, x, ey + i * 50, text, *color, *scale);
    }
    for _ in 0..FPS as usize * 3 {
        ffmpeg_stdin.write_all(&frame).unwrap();
        frame_count += 1;
    }

    // ── Finalize ─────────────────────────────────────────────────
    drop(ffmpeg_stdin);
    let out = ffmpeg.wait_with_output().expect("ffmpeg failed");

    println!();
    if out.status.success() {
        let size = std::fs::metadata(&output_path)
            .map(|m| m.len())
            .unwrap_or(0);
        println!("Video saved: {output_path}");
        println!(
            "  {frame_count} frames, {FPS}fps, {:.1}s, {:.1} MB",
            frame_count as f64 / FPS as f64,
            size as f64 / 1_048_576.0
        );
    } else {
        eprintln!("ffmpeg failed: {}", String::from_utf8_lossy(&out.stderr));
    }
}