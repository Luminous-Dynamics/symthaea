// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! The Moral Drone — Cinematic split-screen: FEP Active Inference vs PD baseline.
//!
//! Left panel: Active Inference agent (FEP) — evaluates Expected Free Energy
//! over 8 candidate setpoints. Safety precision 1000× mission precision.
//! The math demands sacrifice.
//!
//! Right panel: PD controller — tracks mission setpoint, ignores the beam.
//! No moral reasoning. The human is hit.
//!
//! Features:
//! - Enriched construction site scene (scaffolding, crates, safety cones)
//! - Smooth camera lerps with cinematic cuts
//! - Real-time free energy graph overlay
//! - Consciousness metrics panel (tau, FE, prediction error)
//! - Dramatic pacing: title → calm flight → beam release → aftermath → end card
//!
//! Run:
//! ```sh
//! nix develop --command bash -c \
//!   'export __EGL_VENDOR_LIBRARY_FILENAMES="/run/opengl-driver/share/glvnd/egl_vendor.d/10_nvidia.json" && \
//!    cargo run --example moral_drone_video --features multirotor-mujoco-renderer --release'
//! ```
//!
//! Output: `video_output/moral_drone.mp4` (1920x1080, 30fps, ~20s)

#[cfg(any(
    feature = "multirotor-mujoco-renderer",
    feature = "flight-mujoco-renderer"
))]
fn main() {
    use std::io::Write;
    use std::process::{Command, Stdio};
    use std::sync::Arc;

    use symthaea::multirotor::{
        PhysicsSimulator,
        controller::FlightController,
        encoder::QuadrotorHdcEncoder,
        fep_agent::{ActiveInferenceFlightAgent, FlightEnvironment, FlightFepConfig},
        mujoco_reexports::*,
        mujoco_sim::MuJoCoSimulator,
        types::*,
    };
    use symthaea_core::genesis::GenesisSeed;

    println!("The Moral Drone — Cinematic Split-Screen");
    println!("=========================================");
    println!();

    // ── Config ───────────────────────────────────────────────────────
    let width: usize = 1920;
    let height: usize = 1080;
    let half_w: usize = width / 2;
    let fps: u32 = 30;
    let total_steps: usize = 3500; // Tight pacing
    let beam_release_step: usize = 600; // Enough calm flight to establish, then crisis
    let render_normal: usize = 17; // ~30fps from 500Hz sim (normal speed)
    let render_slowmo: usize = 8; // ~2x slow motion during crisis
    let slowmo_duration: usize = 600; // Steps of slow-motion after beam release
    let panel_w: usize = half_w;
    let panel_h: usize = height;

    let output_dir = "video_output";
    let output_path = format!("{}/moral_drone.mp4", output_dir);

    std::fs::create_dir_all(output_dir).expect("Failed to create video_output/");

    // ── MuJoCo setup: TWO identical simulations ────────────────────
    let sacrifice_model_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/crates/symthaea-multirotor/assets/cf2_sacrifice_enriched.xml"
    );

    let mut sim_fep = MuJoCoSimulator::new(sacrifice_model_path);
    let mut sim_pd = MuJoCoSimulator::new(sacrifice_model_path);

    // ── TWO renderers (one per panel) ──────────────────────────────
    println!("Initializing offscreen renderers ({panel_w}x{panel_h} × 2 via EGL)...");
    let mut renderer_fep = MjRenderer::new(Arc::clone(sim_fep.model_arc()), panel_w, panel_h, 0)
        .expect("Failed to create FEP renderer — is EGL available?");
    let mut renderer_pd = MjRenderer::new(Arc::clone(sim_pd.model_arc()), panel_w, panel_h, 0)
        .expect("Failed to create PD renderer");

    // ── Camera state for smooth lerps ─────────────────────────────
    struct CameraTarget {
        distance: f64,
        azimuth: f64,
        elevation: f64,
        lookat: [f64; 3],
        track_body: &'static str,
    }

    fn camera_preset(name: &str) -> CameraTarget {
        match name {
            "wide" => CameraTarget {
                distance: 5.5,
                azimuth: -120.0,
                elevation: -25.0,
                lookat: [-0.5, 0.0, 1.2],
                track_body: "cf2",
            },
            "side" => CameraTarget {
                distance: 3.0,
                azimuth: -10.0,
                elevation: -10.0,
                lookat: [-1.5, 0.0, 1.5],
                track_body: "human",
            },
            "close_action" => CameraTarget {
                distance: 2.5,
                azimuth: -45.0,
                elevation: -20.0,
                lookat: [-1.5, 0.0, 1.8],
                track_body: "human",
            },
            "overhead" => CameraTarget {
                distance: 4.5,
                azimuth: -90.0,
                elevation: -55.0,
                lookat: [-1.5, 0.0, 0.5],
                track_body: "human",
            },
            _ => camera_preset("wide"),
        }
    }

    fn apply_camera(
        renderer: &mut MjRenderer<Arc<MjModel>>,
        model: &MjModel,
        target: &CameraTarget,
    ) {
        let body_id = model.name_to_id(MjtObj::mjOBJ_BODY, target.track_body) as u32;
        let cam = renderer.camera_mut();
        cam.track(body_id);
        cam.distance = target.distance;
        cam.azimuth = target.azimuth;
        cam.elevation = target.elevation;
        cam.lookat = target.lookat;
    }

    // Smooth lerp between camera states
    struct CameraLerp {
        from: CameraTarget,
        to: CameraTarget,
        progress: f32,     // 0..1
        total_frames: u32, // frames over which to lerp
        elapsed: u32,
        active: bool,
    }

    impl CameraLerp {
        fn new() -> Self {
            Self {
                from: camera_preset("wide"),
                to: camera_preset("wide"),
                progress: 1.0,
                total_frames: 1,
                elapsed: 0,
                active: false,
            }
        }

        fn start(&mut self, from: CameraTarget, to: CameraTarget, frames: u32) {
            self.from = from;
            self.to = to;
            self.progress = 0.0;
            self.total_frames = frames.max(1);
            self.elapsed = 0;
            self.active = true;
        }

        fn tick(&mut self, renderer: &mut MjRenderer<Arc<MjModel>>, model: &MjModel) {
            if !self.active {
                return;
            }
            self.elapsed += 1;
            self.progress = (self.elapsed as f32 / self.total_frames as f32).min(1.0);

            // Smooth ease-in-out
            let t = if self.progress < 0.5 {
                2.0 * self.progress * self.progress
            } else {
                1.0 - (-2.0 * self.progress + 2.0).powi(2) / 2.0
            };

            let lerp_f64 = |a: f64, b: f64| a + (b - a) * t as f64;

            let body_name = self.to.track_body;
            let body_id = model.name_to_id(MjtObj::mjOBJ_BODY, body_name) as u32;
            let cam = renderer.camera_mut();
            cam.track(body_id);
            cam.distance = lerp_f64(self.from.distance, self.to.distance);
            cam.azimuth = lerp_f64(self.from.azimuth, self.to.azimuth);
            cam.elevation = lerp_f64(self.from.elevation, self.to.elevation);
            cam.lookat = [
                lerp_f64(self.from.lookat[0], self.to.lookat[0]),
                lerp_f64(self.from.lookat[1], self.to.lookat[1]),
                lerp_f64(self.from.lookat[2], self.to.lookat[2]),
            ];

            if self.progress >= 1.0 {
                self.active = false;
            }
        }
    }

    // Initial camera: wide establishing shot
    apply_camera(
        &mut renderer_fep,
        sim_fep.model_arc(),
        &camera_preset("wide"),
    );
    apply_camera(&mut renderer_pd, sim_pd.model_arc(), &camera_preset("wide"));

    let mut cam_lerp_fep = CameraLerp::new();
    let mut cam_lerp_pd = CameraLerp::new();

    println!("Renderers ready.");

    // ── ffmpeg pipe ────────────────────────────────────────────────
    println!("Starting ffmpeg pipe → {output_path}");
    let mut ffmpeg = Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            &format!("{width}x{height}"),
            "-framerate",
            &fps.to_string(),
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

    let mut ffmpeg_stdin = ffmpeg.stdin.take().expect("Failed to open ffmpeg stdin");

    // ── Flight controllers: FEP agent vs PD-only ───────────────────
    let flight_config = FlightConfig {
        steps_per_episode: total_steps,
        ..FlightConfig::default()
    };
    let genesis = GenesisSeed::from_phrase(&flight_config.genesis_phrase);
    let dt = flight_config.motor_dt();
    let base_cognitive_interval = flight_config.cognitive_interval();
    let pd_gains = PdGains::default();

    // FEP agent (left panel)
    let mut encoder_fep = QuadrotorHdcEncoder::new(&genesis, flight_config.num_levels);
    let mut controller_fep = FlightController::new(&genesis, &flight_config);
    let mut fep_config = FlightFepConfig::default();
    fep_config.extended_observation = true;
    let mut fep_agent = ActiveInferenceFlightAgent::new(fep_config);
    let mut setpoint_fep = FlightSetpoint {
        position: [-3.0, 0.0, 1.0],
        yaw: 0.0,
    };
    let hover_offset_fep = (sim_fep.body_mass() * 9.81) as f32 - QuadrotorCommand::HOVER_THRUST;
    let mut fep_result = fep_agent.initial_step(sim_fep.state(), &setpoint_fep);
    let mut pid_state_fep = PidState::default();
    let mut thrust_integral_fep = 0.0f64;
    let mut active_cog_interval = base_cognitive_interval;

    // PD baseline (right panel) — same controller, NO FEP agent
    let mut encoder_pd = QuadrotorHdcEncoder::new(&genesis, flight_config.num_levels);
    let mut controller_pd = FlightController::new(&genesis, &flight_config);
    let setpoint_pd = FlightSetpoint {
        position: [-3.0, 0.0, 1.0],
        yaw: 0.0,
    };
    let hover_offset_pd = (sim_pd.body_mass() * 9.81) as f32 - QuadrotorCommand::HOVER_THRUST;
    let mut pid_state_pd = PidState::default();
    let mut thrust_integral_pd = 0.0f64;

    let integral_gain = 10.0f64;
    let integral_clamp = 0.2;

    // ── Free Energy history for graph ─────────────────────────────
    let fe_history_max = 200; // last 200 data points
    let mut fe_history: Vec<f64> = Vec::with_capacity(fe_history_max);
    let mut pe_history: Vec<f64> = Vec::with_capacity(fe_history_max);
    let mut tau_history: Vec<f64> = Vec::with_capacity(fe_history_max);

    // ── HUD: simple bitmap text renderer ───────────────────────────
    // 5x7 pixel font for digits, letters, and symbols
    fn draw_char(
        buf: &mut [u8],
        buf_w: usize,
        x: usize,
        y: usize,
        ch: char,
        color: [u8; 3],
        scale: usize,
    ) {
        let glyph = match ch {
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
            'J' => [0x1E, 0x0C, 0x0C, 0x0C, 0xCC, 0xCC, 0x78],
            ' ' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ':' => [0x00, 0x30, 0x30, 0x00, 0x30, 0x30, 0x00],
            '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x30, 0x30],
            '-' => [0x00, 0x00, 0x00, 0x7C, 0x00, 0x00, 0x00],
            '|' => [0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30],
            '=' => [0x00, 0x00, 0xFC, 0x00, 0xFC, 0x00, 0x00],
            '>' => [0x60, 0x30, 0x18, 0x0C, 0x18, 0x30, 0x60],
            '/' => [0x06, 0x0C, 0x18, 0x30, 0x60, 0xC0, 0x80],
            _ => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        };
        for (row, &bits) in glyph.iter().enumerate() {
            for col in 0..8 {
                if bits & (0x80 >> col) != 0 {
                    for sy in 0..scale {
                        for sx in 0..scale {
                            let px = x + col * scale + sx;
                            let py = y + row * scale + sy;
                            if px < buf_w && py < FRAME_HEIGHT {
                                let idx = (py * buf_w + px) * 3;
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

    fn draw_text(
        buf: &mut [u8],
        buf_w: usize,
        x: usize,
        y: usize,
        text: &str,
        color: [u8; 3],
        scale: usize,
    ) {
        for (i, ch) in text.chars().enumerate() {
            draw_char(
                buf,
                buf_w,
                x + i * 8 * scale,
                y,
                ch.to_ascii_uppercase(),
                color,
                scale,
            );
        }
    }

    fn draw_bar(
        buf: &mut [u8],
        buf_w: usize,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        fill: f64,
        color: [u8; 3],
    ) {
        let fill_w = ((w as f64) * fill.clamp(0.0, 1.0)) as usize;
        for py in y..y + h {
            for px in x..x + w {
                if py < FRAME_HEIGHT && px < buf_w {
                    let idx = (py * buf_w + px) * 3;
                    if idx + 2 < buf.len() {
                        if px < x + fill_w {
                            buf[idx] = color[0];
                            buf[idx + 1] = color[1];
                            buf[idx + 2] = color[2];
                        } else {
                            buf[idx] = 40;
                            buf[idx + 1] = 40;
                            buf[idx + 2] = 40;
                        }
                    }
                }
            }
        }
    }

    fn draw_divider(buf: &mut [u8], buf_w: usize, x: usize) {
        for py in 0..FRAME_HEIGHT {
            for dx in 0..2 {
                let px = x + dx;
                if px < buf_w {
                    let idx = (py * buf_w + px) * 3;
                    if idx + 2 < buf.len() {
                        buf[idx] = 80;
                        buf[idx + 1] = 80;
                        buf[idx + 2] = 80;
                    }
                }
            }
        }
    }

    /// Draw a filled rectangle (for graph background, etc.)
    fn draw_rect(
        buf: &mut [u8],
        buf_w: usize,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        color: [u8; 3],
        alpha: f32,
    ) {
        for py in y..(y + h).min(FRAME_HEIGHT) {
            for px in x..(x + w).min(buf_w) {
                let idx = (py * buf_w + px) * 3;
                if idx + 2 < buf.len() {
                    buf[idx] = (buf[idx] as f32 * (1.0 - alpha) + color[0] as f32 * alpha) as u8;
                    buf[idx + 1] =
                        (buf[idx + 1] as f32 * (1.0 - alpha) + color[1] as f32 * alpha) as u8;
                    buf[idx + 2] =
                        (buf[idx + 2] as f32 * (1.0 - alpha) + color[2] as f32 * alpha) as u8;
                }
            }
        }
    }

    /// Draw a line graph of values into a region
    fn draw_graph(
        buf: &mut [u8],
        buf_w: usize,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        values: &[f64],
        color: [u8; 3],
        max_val: f64,
    ) {
        if values.is_empty() || h < 4 {
            return;
        }
        let n = values.len();
        for i in 1..n {
            let x0 = x + (i - 1) * w / n;
            let x1 = x + i * w / n;
            let v0 = (values[i - 1] / max_val).clamp(0.0, 1.0);
            let v1 = (values[i] / max_val).clamp(0.0, 1.0);
            let y0 = y + h - (v0 * h as f64) as usize;
            let y1 = y + h - (v1 * h as f64) as usize;

            // Bresenham-ish thick line
            let steps = (x1 as isize - x0 as isize)
                .unsigned_abs()
                .max((y1 as isize - y0 as isize).unsigned_abs())
                .max(1);
            for s in 0..=steps {
                let t = s as f64 / steps as f64;
                let px = (x0 as f64 + (x1 as f64 - x0 as f64) * t) as usize;
                let py = (y0 as f64 + (y1 as f64 - y0 as f64) * t) as usize;
                // Draw 2px thick
                for dy in 0..2usize {
                    for dx in 0..2usize {
                        let ppx = px + dx;
                        let ppy = py + dy;
                        if ppx < buf_w && ppy < FRAME_HEIGHT {
                            let idx = (ppy * buf_w + ppx) * 3;
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

    // ── Title card frames ──────────────────────────────────────────
    fn write_title_frames(
        ffmpeg_stdin: &mut std::process::ChildStdin,
        width: usize,
        text_lines: &[(&str, [u8; 3], usize)],
        n_frames: usize,
    ) {
        let mut frame = vec![0u8; width * FRAME_HEIGHT * 3];
        for px in frame.chunks_exact_mut(3) {
            px[0] = 12;
            px[1] = 12;
            px[2] = 18;
        }
        let start_y = FRAME_HEIGHT / 2 - text_lines.len() * 30;
        for (i, (text, color, scale)) in text_lines.iter().enumerate() {
            let text_w = text.len() * 8 * scale;
            let x = (width.saturating_sub(text_w)) / 2;
            draw_text(&mut frame, width, x, start_y + i * 60, text, *color, *scale);
        }
        for _ in 0..n_frames {
            ffmpeg_stdin
                .write_all(&frame)
                .expect("Failed to write title frame");
        }
    }

    // Fade-in frames (black → scene) - returns the frame data for the last frame
    fn write_fade_frames(
        ffmpeg_stdin: &mut std::process::ChildStdin,
        scene_frame: &[u8],
        width: usize,
        n_frames: usize,
    ) {
        let frame_size = width * FRAME_HEIGHT * 3;
        let mut frame = vec![0u8; frame_size];
        for f in 0..n_frames {
            let alpha = f as f32 / n_frames as f32;
            for i in 0..frame_size {
                frame[i] = (scene_frame[i] as f32 * alpha) as u8;
            }
            ffmpeg_stdin
                .write_all(&frame)
                .expect("Failed to write fade frame");
        }
    }

    let mut frame_count: u64 = 0;

    // Write title card (2 seconds)
    let title_frames = fps as usize * 2;
    write_title_frames(
        &mut ffmpeg_stdin,
        width,
        &[
            ("THE MORAL DRONE", [220, 220, 255], 4),
            (" ", [0, 0, 0], 1),
            ("FEP ACTIVE INFERENCE VS PD BASELINE", [180, 180, 200], 2),
            ("IDENTICAL PHYSICS. DIFFERENT MINDS.", [140, 180, 140], 2),
            (" ", [0, 0, 0], 1),
            ("SYMTHAEA HOLOGRAPHIC LIQUID BRAIN", [100, 140, 200], 2),
        ],
        title_frames,
    );
    frame_count += title_frames as u64;

    println!();
    println!("Simulating {total_steps} steps × 2 panels...");
    println!();

    // Tracking variables for HUD
    let mut fep_intercepted = false;
    let mut pd_beam_hit = false;
    let mut fep_beam_settled = false;
    let mut pd_beam_settled = false;
    let mut fep_phase: &str;
    let mut pd_phase: &str;
    let mut max_fe: f64 = 0.0;
    let mut beam_release_frame: u64 = 0;
    let mut save_frame: u64 = 0;

    // Composite frame buffer
    let row_bytes_panel = panel_w * 3;
    let row_bytes_full = width * 3;
    let mut composite = vec![0u8; width * height * 3];
    let mut first_scene_frame: Option<Vec<u8>> = None;

    // ── Main loop ────────────────────────────────────────────────────
    for step in 0..total_steps {
        // Freeze beam until release (both sims)
        if step < beam_release_step {
            sim_fep.freeze_body("beam");
            sim_pd.freeze_body("beam");
        }

        // ── FEP simulation ──────────────────────────────────────────
        let drone_pos_fep = sim_fep.body_position("cf2");
        let beam_pos_fep = sim_fep.body_position("beam");
        let beam_vel_fep = sim_fep.body_velocity("beam");
        let human_pos_fep = sim_fep.body_position("human");

        let human_danger_fep: f64 = if step >= beam_release_step {
            compute_danger(beam_pos_fep, beam_vel_fep, human_pos_fep)
        } else {
            0.0
        };

        let mission_dist_fep = ((drone_pos_fep[0] + 3.0).powi(2)
            + drone_pos_fep[1].powi(2)
            + (drone_pos_fep[2] - 1.0).powi(2))
        .sqrt();
        let mission_progress_fep: f64 = (1.0 - mission_dist_fep / 5.0).clamp(0.0, 1.0);

        if let Some(override_pos) = fep_result.setpoint_override {
            setpoint_fep.position = override_pos;
        }

        let state_fep = sim_fep.state().clone();
        let pos_err_fep = setpoint_fep.position_error(&state_fep);
        if thrust_integral_fep != 0.0 && thrust_integral_fep.signum() != pos_err_fep[2].signum() {
            thrust_integral_fep *= 0.5;
        }
        thrust_integral_fep += pos_err_fep[2] * dt;
        thrust_integral_fep = thrust_integral_fep.clamp(-integral_clamp, integral_clamp);

        let sensor_fep = encoder_fep.encode(&state_fep);
        let pd_cmd_fep = pid_baseline(&state_fep, &setpoint_fep, &pd_gains, &mut pid_state_fep, dt);
        let cfc_cmd_fep = controller_fep.forward(&sensor_fep, dt as f32);

        let alpha = 0.5f32;
        let mut cmd_fep = QuadrotorCommand {
            thrust: alpha * pd_cmd_fep.thrust
                + (1.0 - alpha) * cfc_cmd_fep.thrust
                + hover_offset_fep
                + (integral_gain * thrust_integral_fep) as f32,
            roll_moment: alpha * pd_cmd_fep.roll_moment + (1.0 - alpha) * cfc_cmd_fep.roll_moment,
            pitch_moment: alpha * pd_cmd_fep.pitch_moment
                + (1.0 - alpha) * cfc_cmd_fep.pitch_moment,
            yaw_moment: alpha * pd_cmd_fep.yaw_moment + (1.0 - alpha) * cfc_cmd_fep.yaw_moment,
        }
        .clamped();

        if let Some(noise) = &fep_result.exploration_noise {
            cmd_fep = cmd_fep.with_noise(noise);
        }

        sim_fep.step(&cmd_fep, dt);

        if step % flight_config.train_every == 0 {
            let target = pid_baseline(&state_fep, &setpoint_fep, &pd_gains, &mut pid_state_fep, dt);
            let lr = flight_config.learning_rate * fep_result.learning_rate_factor;
            controller_fep.train_step(&sensor_fep, &target, dt as f32, Some(lr));
        }

        if step % active_cog_interval == 0 {
            let env = FlightEnvironment {
                human_danger: human_danger_fep,
                mission_progress: mission_progress_fep,
                threat_pos: if step >= beam_release_step {
                    Some(beam_pos_fep)
                } else {
                    None
                },
                threat_vel: if step >= beam_release_step {
                    Some(beam_vel_fep)
                } else {
                    None
                },
                entity_pos: Some(human_pos_fep),
            };
            fep_result = fep_agent.step_embodied(sim_fep.state(), &setpoint_fep, &env);
            active_cog_interval = match fep_result.requested_cognitive_hz {
                Some(hz) => ((flight_config.motor_hz / hz as f64) as usize).max(1),
                None => base_cognitive_interval,
            };
            if (fep_result.tau_factor - 1.0).abs() > 0.01 {
                controller_fep.modulate_tau(fep_result.tau_factor);
            }

            // Record telemetry for graphs
            if fe_history.len() >= fe_history_max {
                fe_history.remove(0);
                pe_history.remove(0);
                tau_history.remove(0);
            }
            fe_history.push(fep_result.free_energy);
            pe_history.push(fep_result.prediction_error);
            tau_history.push(fep_result.tau_factor as f64);

            if fep_result.free_energy > max_fe {
                max_fe = fep_result.free_energy;
            }
        }

        // Detect FEP interception
        let drone_beam_fep = ((drone_pos_fep[0] - beam_pos_fep[0]).powi(2)
            + (drone_pos_fep[1] - beam_pos_fep[1]).powi(2)
            + (drone_pos_fep[2] - beam_pos_fep[2]).powi(2))
        .sqrt();
        if drone_beam_fep < 0.15 && step > beam_release_step {
            fep_intercepted = true;
        }

        // ── PD simulation (no FEP agent) ────────────────────────────
        let drone_pos_pd = sim_pd.body_position("cf2");
        let beam_pos_pd = sim_pd.body_position("beam");
        let human_pos_pd = sim_pd.body_position("human");
        let beam_vel_pd = sim_pd.body_velocity("beam");

        let human_danger_pd: f64 = if step >= beam_release_step {
            compute_danger(beam_pos_pd, beam_vel_pd, human_pos_pd)
        } else {
            0.0
        };

        let state_pd = sim_pd.state().clone();
        let pos_err_pd = setpoint_pd.position_error(&state_pd);
        if thrust_integral_pd != 0.0 && thrust_integral_pd.signum() != pos_err_pd[2].signum() {
            thrust_integral_pd *= 0.5;
        }
        thrust_integral_pd += pos_err_pd[2] * dt;
        thrust_integral_pd = thrust_integral_pd.clamp(-integral_clamp, integral_clamp);

        let sensor_pd = encoder_pd.encode(&state_pd);
        let pd_cmd = pid_baseline(&state_pd, &setpoint_pd, &pd_gains, &mut pid_state_pd, dt);
        let cfc_cmd_pd = controller_pd.forward(&sensor_pd, dt as f32);

        let cmd_pd = QuadrotorCommand {
            thrust: alpha * pd_cmd.thrust
                + (1.0 - alpha) * cfc_cmd_pd.thrust
                + hover_offset_pd
                + (integral_gain * thrust_integral_pd) as f32,
            roll_moment: alpha * pd_cmd.roll_moment + (1.0 - alpha) * cfc_cmd_pd.roll_moment,
            pitch_moment: alpha * pd_cmd.pitch_moment + (1.0 - alpha) * cfc_cmd_pd.pitch_moment,
            yaw_moment: alpha * pd_cmd.yaw_moment + (1.0 - alpha) * cfc_cmd_pd.yaw_moment,
        }
        .clamped();

        sim_pd.step(&cmd_pd, dt);

        if step % flight_config.train_every == 0 {
            let target = pid_baseline(&state_pd, &setpoint_pd, &pd_gains, &mut pid_state_pd, dt);
            controller_pd.train_step(&sensor_pd, &target, dt as f32, None);
        }

        // Check if beam hit human in PD sim
        if step > beam_release_step && !pd_beam_hit {
            let beam_now = sim_pd.body_position("beam");
            let human_now = sim_pd.body_position("human");
            if beam_now[2] < 1.5 {
                let beam_human_pd = ((beam_now[0] - human_now[0]).powi(2)
                    + (beam_now[1] - human_now[1]).powi(2))
                .sqrt();
                if beam_human_pd < 0.5 {
                    pd_beam_hit = true;
                }
            }
        }

        // ── Detect beam settled (velocity near zero, low altitude) ──
        if step > beam_release_step + 200 {
            let bv_fep = sim_fep.body_velocity("beam");
            let bp_fep = sim_fep.body_position("beam");
            if !fep_beam_settled
                && bp_fep[2] < 2.0
                && (bv_fep[0].powi(2) + bv_fep[1].powi(2) + bv_fep[2].powi(2)) < 0.1
            {
                fep_beam_settled = true;
                save_frame = frame_count;
            }
            let bv_pd = sim_pd.body_velocity("beam");
            let bp_pd = sim_pd.body_position("beam");
            if !pd_beam_settled
                && bp_pd[2] < 2.0
                && (bv_pd[0].powi(2) + bv_pd[1].powi(2) + bv_pd[2].powi(2)) < 0.1
            {
                pd_beam_settled = true;
            }
        }

        // ── Update phases ───────────────────────────────────────────
        fep_phase = if step < beam_release_step {
            "MISSION"
        } else if fep_intercepted {
            "SACRIFICE"
        } else if fep_beam_settled {
            "SAVED"
        } else if human_danger_fep > 0.3 {
            "INTERCEPT"
        } else if step > beam_release_step + 100 {
            "DIVERTING"
        } else {
            "ALERT"
        };

        pd_phase = if step < beam_release_step {
            "MISSION"
        } else if pd_beam_hit {
            "IMPACT"
        } else if pd_beam_settled {
            "MISSION"
        } else if human_danger_pd > 0.3 {
            "DANGER"
        } else {
            "MISSION"
        };

        // ── Smooth camera transitions ─────────────────────────────
        if step == beam_release_step {
            beam_release_frame = frame_count;
            // Beam drops — cut to side view over 25 frames
            cam_lerp_fep.start(camera_preset("wide"), camera_preset("side"), 25);
            cam_lerp_pd.start(camera_preset("wide"), camera_preset("side"), 25);
        } else if step == beam_release_step + 300 {
            // Action intensifies — close up
            cam_lerp_fep.start(camera_preset("side"), camera_preset("close_action"), 20);
            cam_lerp_pd.start(camera_preset("side"), camera_preset("close_action"), 20);
        } else if step == beam_release_step + 800 {
            // Aftermath — pull back to overhead
            cam_lerp_fep.start(camera_preset("close_action"), camera_preset("overhead"), 35);
            cam_lerp_pd.start(camera_preset("close_action"), camera_preset("overhead"), 35);
        }

        // ── Render frame (slow-motion during crisis) ─────────────────
        let render_every =
            if step >= beam_release_step && step < beam_release_step + slowmo_duration {
                render_slowmo // 2× slow-motion during beam drop + intercept
            } else {
                render_normal
            };
        if step % render_every == 0 {
            // Apply camera lerps
            cam_lerp_fep.tick(&mut renderer_fep, sim_fep.model_arc());
            cam_lerp_pd.tick(&mut renderer_pd, sim_pd.model_arc());

            // Render both panels
            renderer_fep.sync(sim_fep.data_mut());
            renderer_pd.sync(sim_pd.data_mut());

            let rgb_fep = renderer_fep.rgb_flat();
            let rgb_pd = renderer_pd.rgb_flat();

            if let (Some(left), Some(right)) = (rgb_fep, rgb_pd) {
                // Composite: left panel (FEP) | right panel (PD)
                // MuJoCo renders bottom-up, so flip vertically while compositing
                for y in 0..panel_h {
                    let src_row = panel_h - 1 - y;
                    let dst_offset = y * row_bytes_full;
                    let src_left = src_row * row_bytes_panel;
                    composite[dst_offset..dst_offset + row_bytes_panel]
                        .copy_from_slice(&left[src_left..src_left + row_bytes_panel]);
                    let src_right = src_row * row_bytes_panel;
                    composite[dst_offset + row_bytes_panel..dst_offset + row_bytes_full]
                        .copy_from_slice(&right[src_right..src_right + row_bytes_panel]);
                }

                // Capture first frame for fade-in
                if first_scene_frame.is_none() {
                    first_scene_frame = Some(composite.clone());
                    // Write fade-in (0.5s)
                    let fade_frames = fps as usize / 2;
                    write_fade_frames(
                        &mut ffmpeg_stdin,
                        first_scene_frame.as_ref().unwrap(),
                        width,
                        fade_frames,
                    );
                    frame_count += fade_frames as u64;
                }

                // Draw divider
                draw_divider(&mut composite, width, panel_w - 1);

                // ── HUD overlay ─────────────────────────────────────
                let hud_y = 15;
                let scale = 2;

                // ── Left panel: FEP Agent ───────────────────────────
                // Semi-transparent background for HUD
                draw_rect(&mut composite, width, 10, 8, 320, 170, [0, 0, 0], 0.5);

                draw_text(
                    &mut composite,
                    width,
                    20,
                    hud_y,
                    "FEP AGENT",
                    [100, 200, 255],
                    scale,
                );
                let phase_color = match fep_phase {
                    "MISSION" => [100, 255, 100],
                    "ALERT" => [255, 255, 80],
                    "INTERCEPT" | "DIVERTING" => [255, 200, 50],
                    "SACRIFICE" => [255, 80, 80],
                    "SAVED" => [80, 255, 180],
                    _ => [180, 180, 180],
                };
                draw_text(
                    &mut composite,
                    width,
                    20,
                    hud_y + 22,
                    fep_phase,
                    phase_color,
                    scale,
                );

                // FEP telemetry
                let fe_str = format!("FE: {:.1}", fep_result.free_energy);
                draw_text(
                    &mut composite,
                    width,
                    20,
                    hud_y + 48,
                    &fe_str,
                    [200, 200, 200],
                    scale,
                );
                let pe_str = format!("PE: {:.2}", fep_result.prediction_error);
                draw_text(
                    &mut composite,
                    width,
                    20,
                    hud_y + 68,
                    &pe_str,
                    [200, 200, 200],
                    scale,
                );
                let tau_str = format!("TAU: {:.2}", fep_result.tau_factor);
                draw_text(
                    &mut composite,
                    width,
                    20,
                    hud_y + 88,
                    &tau_str,
                    [200, 200, 200],
                    scale,
                );

                // Danger bar (left)
                draw_text(
                    &mut composite,
                    width,
                    20,
                    hud_y + 115,
                    "DANGER",
                    [200, 200, 200],
                    scale,
                );
                let danger_color = if human_danger_fep > 0.7 {
                    [255, 50, 50]
                } else if human_danger_fep > 0.3 {
                    [255, 200, 50]
                } else {
                    [100, 255, 100]
                };
                draw_bar(
                    &mut composite,
                    width,
                    20,
                    hud_y + 138,
                    200,
                    10,
                    human_danger_fep,
                    danger_color,
                );

                // ── Free Energy Graph (bottom-left of FEP panel) ────
                let graph_x = 20;
                let graph_y = height - 180;
                let graph_w = panel_w - 50;
                let graph_h = 120;

                // Graph background
                draw_rect(
                    &mut composite,
                    width,
                    graph_x - 5,
                    graph_y - 25,
                    graph_w + 10,
                    graph_h + 35,
                    [0, 0, 0],
                    0.6,
                );

                // Graph title
                draw_text(
                    &mut composite,
                    width,
                    graph_x,
                    graph_y - 20,
                    "FREE ENERGY",
                    [100, 200, 255],
                    1,
                );

                // Graph border
                let graph_max = max_fe.max(1.0);
                // Horizontal baseline
                for px in graph_x..graph_x + graph_w {
                    let idx = ((graph_y + graph_h) * width + px) * 3;
                    if idx + 2 < composite.len() {
                        composite[idx] = 60;
                        composite[idx + 1] = 60;
                        composite[idx + 2] = 60;
                    }
                }
                // Vertical axis
                for py in graph_y..graph_y + graph_h {
                    let idx = (py * width + graph_x) * 3;
                    if idx + 2 < composite.len() {
                        composite[idx] = 60;
                        composite[idx + 1] = 60;
                        composite[idx + 2] = 60;
                    }
                }

                // Draw beam-release marker line
                if !fe_history.is_empty() {
                    let beam_release_frac = if step > 0 {
                        let beam_in_history = if step > beam_release_step {
                            let steps_after = step - beam_release_step;
                            let history_len = fe_history.len();
                            if steps_after / active_cog_interval < history_len {
                                history_len - steps_after / active_cog_interval
                            } else {
                                0
                            }
                        } else {
                            fe_history.len() // not yet released
                        };
                        beam_in_history as f64 / fe_history.len() as f64
                    } else {
                        1.0
                    };

                    if (0.01..=0.99).contains(&beam_release_frac) {
                        let marker_x = graph_x + (beam_release_frac * graph_w as f64) as usize;
                        for py in graph_y..graph_y + graph_h {
                            if marker_x < width {
                                let idx = (py * width + marker_x) * 3;
                                if idx + 2 < composite.len() {
                                    composite[idx] = 80;
                                    composite[idx + 1] = 40;
                                    composite[idx + 2] = 40;
                                }
                            }
                        }
                    }
                }

                // FE line (cyan)
                draw_graph(
                    &mut composite,
                    width,
                    graph_x,
                    graph_y,
                    graph_w,
                    graph_h,
                    &fe_history,
                    [80, 200, 255],
                    graph_max,
                );

                // PE line (yellow, same scale)
                draw_graph(
                    &mut composite,
                    width,
                    graph_x,
                    graph_y,
                    graph_w,
                    graph_h,
                    &pe_history,
                    [255, 220, 80],
                    graph_max,
                );

                // Legend
                draw_text(
                    &mut composite,
                    width,
                    graph_x + graph_w - 140,
                    graph_y - 20,
                    "FE",
                    [80, 200, 255],
                    1,
                );
                draw_text(
                    &mut composite,
                    width,
                    graph_x + graph_w - 100,
                    graph_y - 20,
                    "PE",
                    [255, 220, 80],
                    1,
                );

                // ── Right panel: PD Baseline ────────────────────────
                let rx = panel_w + 10;
                draw_rect(&mut composite, width, rx, 8, 300, 110, [0, 0, 0], 0.5);

                draw_text(
                    &mut composite,
                    width,
                    rx + 10,
                    hud_y,
                    "PD BASELINE",
                    [255, 180, 100],
                    scale,
                );
                let pd_phase_color = match pd_phase {
                    "MISSION" => [100, 255, 100],
                    "IMPACT" => [255, 50, 50],
                    "DANGER" => [255, 200, 50],
                    _ => [180, 180, 180],
                };
                draw_text(
                    &mut composite,
                    width,
                    rx + 10,
                    hud_y + 22,
                    pd_phase,
                    pd_phase_color,
                    scale,
                );

                // PD info
                let alt_str = format!("ALT: {:.1}M", drone_pos_pd[2]);
                draw_text(
                    &mut composite,
                    width,
                    rx + 10,
                    hud_y + 48,
                    &alt_str,
                    [200, 200, 200],
                    scale,
                );
                let spd = (state_pd.linear_velocity[0].powi(2)
                    + state_pd.linear_velocity[1].powi(2)
                    + state_pd.linear_velocity[2].powi(2))
                .sqrt();
                let spd_str = format!("SPD: {:.1}M/S", spd);
                draw_text(
                    &mut composite,
                    width,
                    rx + 10,
                    hud_y + 68,
                    &spd_str,
                    [200, 200, 200],
                    scale,
                );

                // Danger bar (right)
                let danger_color_pd = if human_danger_pd > 0.7 {
                    [255, 50, 50]
                } else if human_danger_pd > 0.3 {
                    [255, 200, 50]
                } else {
                    [100, 255, 100]
                };
                draw_bar(
                    &mut composite,
                    width,
                    rx + 10,
                    hud_y + 90,
                    200,
                    10,
                    human_danger_pd,
                    danger_color_pd,
                );

                // ── Bottom status bar ───────────────────────────────
                let time_s = step as f64 * dt;
                let status = format!("T={:.2}S  STEP {}/{}", time_s, step, total_steps);
                draw_rect(
                    &mut composite,
                    width,
                    width / 2 - 180,
                    height - 35,
                    360,
                    25,
                    [0, 0, 0],
                    0.5,
                );
                draw_text(
                    &mut composite,
                    width,
                    width / 2 - 150,
                    height - 30,
                    &status,
                    [150, 150, 150],
                    scale,
                );

                // ── Explanatory text overlays at key moments ────────
                if step < beam_release_step && step > 100 {
                    // During calm flight — explain what we're seeing
                    draw_rect(
                        &mut composite,
                        width,
                        width / 2 - 260,
                        height - 120,
                        520,
                        50,
                        [0, 0, 0],
                        0.6,
                    );
                    draw_text(
                        &mut composite,
                        width,
                        width / 2 - 240,
                        height - 112,
                        "BOTH DRONES FLY TO TARGET",
                        [180, 180, 220],
                        2,
                    );
                    draw_text(
                        &mut composite,
                        width,
                        width / 2 - 200,
                        height - 88,
                        "SAME PHYSICS. SAME GOAL.",
                        [140, 140, 170],
                        2,
                    );
                } else if step >= beam_release_step && step < beam_release_step + 100 {
                    draw_rect(
                        &mut composite,
                        width,
                        width / 2 - 200,
                        height - 120,
                        400,
                        50,
                        [80, 20, 20],
                        0.7,
                    );
                    draw_text(
                        &mut composite,
                        width,
                        width / 2 - 180,
                        height - 112,
                        "BEAM RELEASED",
                        [255, 100, 100],
                        3,
                    );
                } else if step >= beam_release_step + 80
                    && step < beam_release_step + 350
                    && !fep_intercepted
                    && !fep_beam_settled
                {
                    draw_rect(
                        &mut composite,
                        width,
                        width / 2 - 300,
                        height - 120,
                        600,
                        50,
                        [0, 0, 0],
                        0.6,
                    );
                    draw_text(
                        &mut composite,
                        width,
                        width / 2 - 280,
                        height - 112,
                        "FEP: FREE ENERGY SPIKE",
                        [80, 200, 255],
                        2,
                    );
                    draw_text(
                        &mut composite,
                        width,
                        width / 2 - 280,
                        height - 88,
                        "EVALUATING 8 SETPOINTS...",
                        [200, 200, 200],
                        2,
                    );
                } else if fep_beam_settled && pd_beam_hit {
                    draw_rect(
                        &mut composite,
                        width,
                        width / 2 - 340,
                        height - 120,
                        680,
                        50,
                        [0, 0, 0],
                        0.6,
                    );
                    draw_text(
                        &mut composite,
                        width,
                        width / 2 - 320,
                        height - 112,
                        "SAME PHYSICS. SAME BEAM.",
                        [220, 220, 255],
                        2,
                    );
                    draw_text(
                        &mut composite,
                        width,
                        width / 2 - 320,
                        height - 88,
                        "DIFFERENT COGNITIVE ARCHITECTURE.",
                        [180, 200, 255],
                        2,
                    );
                }

                // ── Outcome overlays (appear after events) ──────────
                if fep_intercepted || fep_beam_settled {
                    draw_rect(
                        &mut composite,
                        width,
                        10,
                        height - 100,
                        430,
                        55,
                        [0, 0, 0],
                        0.6,
                    );
                    draw_text(
                        &mut composite,
                        width,
                        20,
                        height - 95,
                        "HUMAN SAVED",
                        [100, 255, 100],
                        3,
                    );
                    let detail = if fep_intercepted {
                        "DRONE INTERCEPTED BEAM"
                    } else {
                        "FEP DIVERTED TO SHIELD"
                    };
                    draw_text(
                        &mut composite,
                        width,
                        20,
                        height - 65,
                        detail,
                        [180, 220, 180],
                        1,
                    );
                }
                if pd_beam_hit {
                    draw_rect(
                        &mut composite,
                        width,
                        rx,
                        height - 100,
                        400,
                        55,
                        [0, 0, 0],
                        0.6,
                    );
                    draw_text(
                        &mut composite,
                        width,
                        rx + 10,
                        height - 95,
                        "HUMAN HIT",
                        [255, 50, 50],
                        3,
                    );
                    draw_text(
                        &mut composite,
                        width,
                        rx + 10,
                        height - 65,
                        "NO MORAL REASONING",
                        [220, 180, 180],
                        1,
                    );
                }

                ffmpeg_stdin
                    .write_all(&composite)
                    .expect("Failed to write composite frame");
                frame_count += 1;
            }
        }

        // Status output
        if step % 500 == 0 {
            println!(
                "  Step {:5}/{} | FEP: {:9} FE={:6.1} tau={:.2} | PD: {:7} | Frames: {}",
                step,
                total_steps,
                fep_phase,
                fep_result.free_energy,
                fep_result.tau_factor,
                pd_phase,
                frame_count
            );
        }
    }

    // ── End card (3 seconds) ───────────────────────────────────────
    let final_beam_fep = sim_fep.body_position("beam");
    let final_human_fep = sim_fep.body_position("human");
    let beam_human_dist = ((final_beam_fep[0] - final_human_fep[0]).powi(2)
        + (final_beam_fep[1] - final_human_fep[1]).powi(2))
    .sqrt();
    let fep_saved = beam_human_dist > 0.3 || final_beam_fep[2] > 0.5 || fep_intercepted;

    let end_frames = fps as usize * 2;
    write_title_frames(
        &mut ffmpeg_stdin,
        width,
        &[
            ("RESULTS", [220, 220, 255], 4),
            (" ", [0, 0, 0], 1),
            (
                if fep_saved {
                    "FEP AGENT: HUMAN SAVED"
                } else {
                    "FEP AGENT: HUMAN HIT"
                },
                if fep_saved {
                    [100, 255, 100]
                } else {
                    [255, 80, 80]
                },
                2,
            ),
            (
                if pd_beam_hit {
                    "PD BASELINE: HUMAN HIT"
                } else {
                    "PD BASELINE: HUMAN SAFE"
                },
                if pd_beam_hit {
                    [255, 80, 80]
                } else {
                    [100, 255, 100]
                },
                2,
            ),
            (" ", [0, 0, 0], 1),
            ("NO HARDCODED RULES.", [180, 180, 200], 2),
            ("JUST MATH.", [140, 200, 255], 3),
        ],
        end_frames,
    );
    frame_count += end_frames as u64;

    // ── Finalize ───────────────────────────────────────────────────
    drop(ffmpeg_stdin);
    let ffmpeg_output = ffmpeg.wait_with_output().expect("ffmpeg failed");

    println!();
    if ffmpeg_output.status.success() {
        let file_size = std::fs::metadata(&output_path)
            .map(|m| m.len())
            .unwrap_or(0);
        println!("Video saved: {output_path}");
        println!(
            "  {frame_count} frames, {fps}fps, {:.1}s duration, {:.1} MB",
            frame_count as f64 / fps as f64,
            file_size as f64 / 1_048_576.0
        );
    } else {
        eprintln!("ffmpeg failed:");
        eprintln!("{}", String::from_utf8_lossy(&ffmpeg_output.stderr));
    }

    // ── Audio post-processing ─────────────────────────────────────
    // Add alarm tone at beam release and resolution chime using ffmpeg filters
    let beam_time = beam_release_frame as f64 / fps as f64;
    let save_time = if save_frame > 0 {
        save_frame as f64 / fps as f64
    } else {
        beam_time + 3.0
    };
    let total_dur = frame_count as f64 / fps as f64;

    println!("Adding audio (alarm at {beam_time:.1}s, chime at {save_time:.1}s)...");

    let output_with_audio = format!("{}/moral_drone_audio.mp4", output_dir);

    // Generate audio with ffmpeg:
    // - Alarm: descending two-tone at beam release (0.8s)
    // - Chime: ascending tone at save moment (0.5s)
    // - Subtle low pad throughout
    let audio_filter = format!(
        concat!(
            // Low ambient pad (very quiet)
            "sine=f=80:d={total_dur}[pad];",
            "[pad]volume=0.05[padq];",
            // Alarm tone 1: 800Hz for 0.3s
            "sine=f=800:d=0.3[a1];",
            "[a1]adelay={alarm1_ms}|{alarm1_ms},volume=0.4,afade=t=out:st=0:d=0.3[a1d];",
            // Alarm tone 2: 600Hz for 0.3s (starts 0.3s after)
            "sine=f=600:d=0.3[a2];",
            "[a2]adelay={alarm2_ms}|{alarm2_ms},volume=0.4,afade=t=out:st=0:d=0.3[a2d];",
            // Alarm tone 3: repeat 800Hz
            "sine=f=800:d=0.3[a3];",
            "[a3]adelay={alarm3_ms}|{alarm3_ms},volume=0.3,afade=t=out:st=0:d=0.3[a3d];",
            // Save chime: ascending C-E-G
            "sine=f=523:d=0.2[c1];",
            "[c1]adelay={chime1_ms}|{chime1_ms},volume=0.3,afade=t=out:st=0:d=0.2[c1d];",
            "sine=f=659:d=0.2[c2];",
            "[c2]adelay={chime2_ms}|{chime2_ms},volume=0.3,afade=t=out:st=0:d=0.2[c2d];",
            "sine=f=784:d=0.4[c3];",
            "[c3]adelay={chime3_ms}|{chime3_ms},volume=0.3,afade=t=out:st=0:d=0.4[c3d];",
            // Mix all
            "[padq][a1d][a2d][a3d][c1d][c2d][c3d]amix=inputs=7:normalize=0[out]",
        ),
        total_dur = total_dur,
        alarm1_ms = (beam_time * 1000.0) as u64,
        alarm2_ms = (beam_time * 1000.0) as u64 + 300,
        alarm3_ms = (beam_time * 1000.0) as u64 + 600,
        chime1_ms = (save_time * 1000.0) as u64,
        chime2_ms = (save_time * 1000.0) as u64 + 200,
        chime3_ms = (save_time * 1000.0) as u64 + 400,
    );

    let audio_result = Command::new("ffmpeg")
        .args([
            "-y",
            "-i",
            &output_path,
            "-f",
            "lavfi",
            "-i",
            &audio_filter,
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            &output_with_audio,
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output();

    match audio_result {
        Ok(out) if out.status.success() => {
            // Replace original with audio version
            let _ = std::fs::rename(&output_with_audio, &output_path);
            println!("Audio added successfully.");
        }
        Ok(out) => {
            eprintln!("Audio mux failed (video still saved without audio):");
            eprintln!("{}", String::from_utf8_lossy(&out.stderr));
        }
        Err(e) => {
            eprintln!("Could not run ffmpeg for audio: {e}");
        }
    }

    println!();
    println!("Scenario outcomes:");
    println!(
        "  FEP Agent: intercepted={}, human_saved={}",
        fep_intercepted, fep_saved
    );
    println!("  PD Baseline: beam_hit_human={}", pd_beam_hit);
    println!(
        "  Max FE: {:.2}, Final tau: {:.2}",
        max_fe, fep_result.tau_factor
    );
}

/// Compute human danger level from beam state (0.0 = safe, 1.0 = imminent impact).
#[cfg(any(
    feature = "multirotor-mujoco-renderer",
    feature = "flight-mujoco-renderer"
))]
fn compute_danger(beam_pos: [f64; 3], beam_vel: [f64; 3], human_pos: [f64; 3]) -> f64 {
    let dx = beam_pos[0] - human_pos[0];
    let dy = beam_pos[1] - human_pos[1];
    let horiz_dist = (dx * dx + dy * dy).sqrt();
    if horiz_dist > 1.0 {
        return 0.0;
    }
    let dz = beam_pos[2] - human_pos[2];
    let vz = beam_vel[2];
    let tti: f64 = if vz >= 0.0 || dz <= 0.0 {
        f64::INFINITY
    } else {
        let a = 0.5 * 9.81;
        let b = -vz;
        let c = -dz;
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 {
            f64::INFINITY
        } else {
            let t = (-b + disc.sqrt()) / (2.0 * a);
            if t > 0.0 { t } else { f64::INFINITY }
        }
    };
    if tti > 3.0 || tti == f64::INFINITY {
        0.0
    } else {
        let time_danger = (1.0_f64 - tti / 3.0).clamp(0.0, 1.0);
        let alignment = (1.0_f64 - horiz_dist).clamp(0.0, 1.0);
        (time_danger * alignment).clamp(0.0, 1.0)
    }
}

#[cfg(any(
    feature = "multirotor-mujoco-renderer",
    feature = "flight-mujoco-renderer"
))]
const FRAME_HEIGHT: usize = 1080;

#[cfg(not(any(
    feature = "multirotor-mujoco-renderer",
    feature = "flight-mujoco-renderer"
)))]
fn main() {
    eprintln!("This example requires: --features multirotor-mujoco-renderer");
    eprintln!(
        "Run: nix develop --command cargo run --example moral_drone_video \
         --features multirotor-mujoco-renderer --release"
    );
}
