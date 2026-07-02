// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Headless screenshot scheduler shared across the symtropy demo crates.
//!
//! Usage inside a demo's `main.rs`:
//!
//! ```ignore
//! App::new()
//!     .add_plugins(DefaultPlugins.set(/* ... */))
//!     .add_plugins(MyDemoPlugin)
//!     // Add this line — stem is used in the PNG filenames:
//!     .add_plugins(symtropy_demo_capture::CapturePlugin::new("flight"))
//!     .run();
//! ```
//!
//! When `SYMTROPY_DEMO_CAPTURE_DIR` is unset the plugin is a no-op. When
//! set, it schedules PNG screenshots at `t = 1.5, 4.0, 7.0 s` of real
//! time, then writes `AppExit::Success` at `t = 8.5 s` so the process
//! terminates cleanly.
//!
//! Pattern cloned verbatim from
//! `symtropy/crates/symtropy-bevy/examples/pendulum_swarm.rs` (which is
//! the first demo where this approach was proven).

use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};

/// Per-capture-run scheduling state. Present as a resource only when the
/// env var was set at plugin-build time.
#[derive(Resource)]
struct CaptureSchedule {
    /// Output directory (read from `SYMTROPY_DEMO_CAPTURE_DIR`).
    dir: String,
    /// Filename stem (e.g. "flight" → `flight_t1.5.png`).
    stem: String,
    /// Screenshot timestamps (seconds since startup, ascending).
    schedule: Vec<f32>,
    /// How many screenshots have been queued so far.
    fired: usize,
    /// Wall-clock time (seconds since startup) at which to exit.
    exit_at: f32,
    /// Last time a screenshot request was queued. Exit waits briefly after this
    /// so Bevy can flush the screenshot observer to disk.
    last_capture_at: Option<f32>,
    /// Expected capture files. The plugin exits only after these exist.
    pending_paths: Vec<String>,
}

/// Plugin that wires up headless capture when
/// `SYMTROPY_DEMO_CAPTURE_DIR` is set in the environment. No-op otherwise.
pub struct CapturePlugin {
    stem: String,
}

impl CapturePlugin {
    /// `stem` is used as the filename prefix (e.g. "flight" →
    /// `flight_t1.5.png`). Typically pass the demo's short name.
    pub fn new(stem: impl Into<String>) -> Self {
        Self { stem: stem.into() }
    }
}

impl Plugin for CapturePlugin {
    fn build(&self, app: &mut App) {
        let Ok(dir) = std::env::var("SYMTROPY_DEMO_CAPTURE_DIR") else {
            // Default path: env not set — plugin is a pure no-op.
            return;
        };
        let schedule = capture_schedule_from_env().unwrap_or_else(|| vec![1.5, 4.0, 7.0]);
        let exit_at = std::env::var("SYMTROPY_DEMO_CAPTURE_EXIT_AT")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .unwrap_or_else(|| schedule.last().copied().unwrap_or(7.0) + 1.5);
        app.insert_resource(CaptureSchedule {
            dir,
            stem: self.stem.clone(),
            schedule,
            fired: 0,
            exit_at,
            last_capture_at: None,
            pending_paths: Vec::new(),
        });
        app.add_systems(Update, headless_capture);
    }
}

fn capture_schedule_from_env() -> Option<Vec<f32>> {
    let raw = std::env::var("SYMTROPY_DEMO_CAPTURE_TIMES").ok()?;
    let mut schedule = Vec::new();
    for part in raw.split(',') {
        let value = part.trim();
        if value.is_empty() {
            continue;
        }
        match value.parse::<f32>() {
            Ok(time) if time.is_finite() && time >= 0.0 => schedule.push(time),
            _ => {
                warn!("symtropy-demo-capture: ignoring invalid capture time `{value}`");
            }
        }
    }
    schedule.sort_by(f32::total_cmp);
    schedule.dedup_by(|a, b| (*a - *b).abs() <= f32::EPSILON);
    (!schedule.is_empty()).then_some(schedule)
}

fn headless_capture(
    mut commands: Commands,
    time: Res<Time<Real>>,
    mut sched: ResMut<CaptureSchedule>,
    mut exit: MessageWriter<AppExit>,
) {
    let now = time.elapsed_secs();
    if sched.fired < sched.schedule.len() && now >= sched.schedule[sched.fired] {
        let label = sched.schedule[sched.fired];
        let path = format!("{}/{}_t{:.1}.png", sched.dir, sched.stem, label);
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path.clone()));
        info!("symtropy-demo-capture: queued screenshot {}", path);
        sched.fired += 1;
        sched.last_capture_at = Some(now);
        sched.pending_paths.push(path);
    }
    let captures_flushed = sched
        .last_capture_at
        .map(|last_capture_at| now - last_capture_at >= 1.0)
        .unwrap_or(true);
    let captures_on_disk = sched.pending_paths.len() >= sched.schedule.len()
        && sched.pending_paths.iter().all(|path| {
            std::fs::metadata(path)
                .map(|metadata| metadata.len() > 0)
                .unwrap_or(false)
        });
    let timed_out = now >= sched.exit_at + 15.0;
    if timed_out && !captures_on_disk {
        warn!(
            "symtropy-demo-capture: timed out waiting for screenshots: {:?}",
            sched.pending_paths
        );
    }
    if sched.fired >= sched.schedule.len()
        && now >= sched.exit_at
        && captures_flushed
        && (captures_on_disk || timed_out)
    {
        info!("symtropy-demo-capture: done — exiting");
        exit.write(AppExit::Success);
    }
}
