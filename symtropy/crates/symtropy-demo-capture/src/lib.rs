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
use bevy::render::view::screenshot::{save_to_disk, Screenshot};

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
        app.insert_resource(CaptureSchedule {
            dir,
            stem: self.stem.clone(),
            schedule: vec![1.5, 4.0, 7.0],
            fired: 0,
            exit_at: 8.5,
        });
        app.add_systems(Update, headless_capture);
    }
}

fn headless_capture(
    mut commands: Commands,
    time: Res<Time<Real>>,
    mut sched: ResMut<CaptureSchedule>,
    mut exit: MessageWriter<AppExit>,
) {
    let now = time.elapsed_secs();
    while sched.fired < sched.schedule.len() && now >= sched.schedule[sched.fired] {
        let label = sched.schedule[sched.fired];
        let path = format!("{}/{}_t{:.1}.png", sched.dir, sched.stem, label);
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path.clone()));
        info!("symtropy-demo-capture: queued screenshot {}", path);
        sched.fired += 1;
    }
    if now >= sched.exit_at {
        info!("symtropy-demo-capture: done — exiting");
        exit.write(AppExit::Success);
    }
}
