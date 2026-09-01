// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/// Mycelial colonization boot animation for NixOS installation and system boot.
///
/// Runs on bare-metal DRM/KMS framebuffer — no display server, no GPU acceleration.
/// Renders procedural mycelial growth seeded by the installation's genesis phrase.
///
/// Usage:
///   quicken-fb --genesis-phrase "your sovereign phrase here"
///   quicken-fb --genesis-phrase "..." --progress-pipe /run/quicken-progress
///   quicken-fb --genesis-phrase "..." --boot-events-socket /run/symthaea/boot-events.sock \
///       --boot-state-path /run/symthaea-boot/state-v1.json
///   quicken-fb --genesis-phrase "..." --handoff-receipt /run/symthaea/boot-display-released-v1.json
///   quicken-fb --genesis-phrase "..." --device /dev/dri/card1
///
/// Signal handling:
///   SIGTERM — bounded fast display release
///   SIGINT  — same as SIGTERM
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use symthaea_quicken_fb::boot_protocol::{BootTelemetry, BootVisualState};
use symthaea_quicken_fb::framebuffer::DrmFramebuffer;
use symthaea_quicken_fb::handoff::{DisplayReleaseReceipt, ExitReason};
use symthaea_quicken_fb::mycelium::MycelialNetwork;
use symthaea_quicken_fb::progress::{ProgressEvent, ProgressMonitor};

/// Target frame rate for the animation.
const TARGET_FPS: u32 = 30;

/// Duration of the installation-completion contraction animation in seconds.
const CONTRACTION_DURATION: f32 = 3.0;

/// Duration to hold the white flash before fading to black.
const FLASH_HOLD: f32 = 0.5;

/// Duration of the final fade to black.
const FADE_DURATION: f32 = 1.5;

/// Global flag set by signal handler. Signal-triggered shutdown deliberately
/// bypasses the long installation ceremony so login/recovery is never delayed.
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

fn main() {
    let process_start = Instant::now();
    let args = parse_args();

    install_signal_handlers();

    let mut fb = match DrmFramebuffer::open(&args.device) {
        Ok(fb) => fb,
        Err(e) => {
            eprintln!("quicken-fb: failed to open framebuffer: {e}");
            std::process::exit(1);
        }
    };

    eprintln!(
        "quicken-fb: display {}x{} @ {}Hz on {}",
        fb.width,
        fb.height,
        fb.mode.vrefresh(),
        args.device,
    );

    let mut network = MycelialNetwork::new(fb.width, fb.height, &args.genesis_phrase);
    let mut progress = ProgressMonitor::new(args.progress_pipe.as_deref());
    let mut boot = BootTelemetry::new(
        args.boot_events_socket.as_deref().map(Path::new),
        args.boot_state_path.as_deref().map(Path::new),
    );
    let mut last_boot_visual = boot.visual_state();

    let buf_size = (fb.width * fb.height) as usize;
    let mut render_buf = vec![0u32; buf_size];

    let frame_duration = Duration::from_nanos(1_000_000_000 / TARGET_FPS as u64);
    let start_time = Instant::now();
    let mut last_frame = Instant::now();

    let mut completing = false;
    let mut contraction_start: Option<Instant> = None;
    let mut exit_reason = ExitReason::Natural;

    loop {
        let now = Instant::now();
        let _elapsed_total = now.duration_since(start_time).as_secs_f32();

        if SHUTDOWN.load(Ordering::Relaxed) {
            exit_reason = ExitReason::Signal;
            break;
        }

        let dt = now.duration_since(last_frame).as_secs_f32();
        if dt < frame_duration.as_secs_f32() {
            std::thread::sleep(Duration::from_micros(500));
            continue;
        }
        last_frame = now;

        let boot_report = boot.poll();
        let current_boot_visual = boot.visual_state();
        if boot_report.applied > 0 || boot_report.lineage_resets > 0 {
            if visual_changed(last_boot_visual, current_boot_visual) {
                network.pulse();
                if let Some(state) = current_boot_visual {
                    eprintln!(
                        "quicken-fb: boot state phase={:?} health={:?} sequence={}",
                        state.phase, state.health, state.sequence
                    );
                }
            }
            last_boot_visual = current_boot_visual;
        }

        let events = progress.poll();
        for event in &events {
            match event {
                ProgressEvent::DerivationComplete(name) => {
                    network.pulse();
                    eprintln!("quicken-fb: derivation complete: {name}");
                }
                ProgressEvent::PhaseChange(phase) => {
                    eprintln!("quicken-fb: phase: {phase}");
                }
                ProgressEvent::InstallComplete => {
                    if !completing {
                        completing = true;
                        contraction_start = Some(now);
                        network.pulse();
                        eprintln!("quicken-fb: installation complete — beginning final animation");
                    }
                }
                _ => {}
            }
        }

        if completing {
            if let Some(cs) = contraction_start {
                let t = now.duration_since(cs).as_secs_f32();
                if t < CONTRACTION_DURATION {
                    network.contract(t / CONTRACTION_DURATION);
                    if (t * 4.0) as u32 % 2 == 0 {
                        network.pulse();
                    }
                } else if t < CONTRACTION_DURATION + FLASH_HOLD {
                    network.contract(1.0);
                } else if t < CONTRACTION_DURATION + FLASH_HOLD + FADE_DURATION {
                    network.contract(1.0);
                } else {
                    exit_reason = ExitReason::InstallComplete;
                    break;
                }
            }
        }

        // Installer I/O remains authoritative for installer animation pacing.
        // Typed boot telemetry contributes only a minimum visual growth rate.
        let boot_growth_floor = current_boot_visual
            .map(|state| state.growth_floor)
            .unwrap_or(0.0);
        let growth_rate = progress.io_rate.max(boot_growth_floor);
        network.grow(dt, growth_rate);

        network.render(&mut render_buf);
        fb.blit_from(&render_buf);
    }

    // Present one deterministic black frame before releasing the KMS objects.
    // The authoritative release boundary is the subsequent Drop of `fb`, which
    // restores the saved CRTC before closing/destroying renderer resources.
    render_buf.fill(0);
    fb.blit_from(&render_buf);

    let release_start = Instant::now();
    drop(fb);
    let release_elapsed = release_start.elapsed();

    if let Some(path) = args.handoff_receipt.as_deref() {
        let receipt = DisplayReleaseReceipt::new(release_elapsed, process_start.elapsed(), exit_reason);
        if let Err(error) = receipt.write_atomic(Path::new(path)) {
            // Receipt failure is diagnostic-only and must never turn a successful
            // DRM release into a boot/session failure.
            eprintln!("quicken-fb: failed to write handoff receipt: {error}");
        }
    }

    eprintln!(
        "quicken-fb: display released in {}us; clean exit ({})",
        release_elapsed.as_micros(),
        exit_reason.as_str()
    );
}

fn visual_changed(previous: Option<BootVisualState>, current: Option<BootVisualState>) -> bool {
    match (previous, current) {
        (None, None) => false,
        (Some(a), Some(b)) => {
            a.sequence != b.sequence || a.phase != b.phase || a.health != b.health
        }
        _ => true,
    }
}

/// Parsed command-line arguments.
struct Args {
    genesis_phrase: String,
    progress_pipe: Option<String>,
    boot_events_socket: Option<String>,
    boot_state_path: Option<String>,
    handoff_receipt: Option<String>,
    device: String,
}

/// Minimal argument parser (no clap dependency to keep binary small).
fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut genesis_phrase = None;
    let mut progress_pipe = None;
    let mut boot_events_socket = None;
    let mut boot_state_path = None;
    let mut handoff_receipt = None;
    let mut device = "/dev/dri/card0".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--genesis-phrase" => {
                i += 1;
                if i < args.len() {
                    genesis_phrase = Some(args[i].clone());
                }
            }
            "--progress-pipe" => {
                i += 1;
                if i < args.len() {
                    progress_pipe = Some(args[i].clone());
                }
            }
            "--boot-events-socket" => {
                i += 1;
                if i < args.len() {
                    boot_events_socket = Some(args[i].clone());
                }
            }
            "--boot-state-path" => {
                i += 1;
                if i < args.len() {
                    boot_state_path = Some(args[i].clone());
                }
            }
            "--handoff-receipt" => {
                i += 1;
                if i < args.len() {
                    handoff_receipt = Some(args[i].clone());
                }
            }
            "--device" => {
                i += 1;
                if i < args.len() {
                    device = args[i].clone();
                }
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                eprintln!("quicken-fb: unknown argument: {other}");
                print_usage();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let genesis_phrase = match genesis_phrase {
        Some(p) => p,
        None => {
            eprintln!("quicken-fb: --genesis-phrase is required");
            print_usage();
            std::process::exit(1);
        }
    };

    Args {
        genesis_phrase,
        progress_pipe,
        boot_events_socket,
        boot_state_path,
        handoff_receipt,
        device,
    }
}

fn print_usage() {
    eprintln!(
        "Usage: quicken-fb --genesis-phrase <PHRASE> [OPTIONS]\n\
         \n\
         Options:\n\
         \x20 --genesis-phrase <PHRASE>   Genesis phrase for deterministic pattern seeding\n\
         \x20 --progress-pipe <PATH>      Named pipe for installer progress events\n\
         \x20 --boot-events-socket <PATH> Unix datagram socket for typed boot telemetry\n\
         \x20 --boot-state-path <PATH>    Lineage-bound boot snapshot side channel\n\
         \x20 --handoff-receipt <PATH>    Atomic acknowledgement written after DRM release\n\
         \x20 --device <PATH>             DRM device path (default: /dev/dri/card0)\n\
         \x20 --help                      Show this help"
    );
}

fn install_signal_handlers() {
    unsafe {
        nix::libc::signal(
            nix::libc::SIGTERM,
            signal_handler as nix::libc::sighandler_t,
        );
        nix::libc::signal(nix::libc::SIGINT, signal_handler as nix::libc::sighandler_t);
    }
}

extern "C" fn signal_handler(_sig: std::ffi::c_int) {
    SHUTDOWN.store(true, Ordering::Relaxed);
}
