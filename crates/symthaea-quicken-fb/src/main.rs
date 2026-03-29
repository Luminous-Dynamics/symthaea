// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/// Mycelial colonization boot animation for NixOS installation.
///
/// Runs on bare-metal DRM/KMS framebuffer — no display server, no GPU acceleration.
/// Renders procedural mycelial growth seeded by the installation's genesis phrase.
///
/// Usage:
///   quicken-fb --genesis-phrase "your sovereign phrase here"
///   quicken-fb --genesis-phrase "..." --progress-pipe /run/quicken-progress
///   quicken-fb --genesis-phrase "..." --device /dev/dri/card1
///
/// Signal handling:
///   SIGTERM — clean exit (restore CRTC, unmap, fade to black)
///   SIGINT  — same as SIGTERM
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use symthaea_quicken_fb::framebuffer::DrmFramebuffer;
use symthaea_quicken_fb::mycelium::MycelialNetwork;
use symthaea_quicken_fb::progress::{ProgressEvent, ProgressMonitor};

/// Target frame rate for the animation.
const TARGET_FPS: u32 = 30;

/// Duration of the final contraction animation in seconds.
const CONTRACTION_DURATION: f32 = 3.0;

/// Duration to hold the white flash before fading to black.
const FLASH_HOLD: f32 = 0.5;

/// Duration of the final fade to black.
const FADE_DURATION: f32 = 1.5;

/// Global flag set by signal handler.
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

fn main() {
    let args = parse_args();

    // Install signal handlers
    install_signal_handlers();

    // Open DRM framebuffer
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

    // Create mycelial network
    let mut network = MycelialNetwork::new(fb.width, fb.height, &args.genesis_phrase);

    // Create progress monitor
    let mut progress = ProgressMonitor::new(args.progress_pipe.as_deref());

    // Allocate render buffer (row-major, no stride padding)
    let buf_size = (fb.width * fb.height) as usize;
    let mut render_buf = vec![0u32; buf_size];

    let frame_duration = Duration::from_nanos(1_000_000_000 / TARGET_FPS as u64);
    let start_time = Instant::now();
    let mut last_frame = Instant::now();

    // Animation state
    let mut completing = false;
    let mut contraction_start: Option<Instant> = None;

    // Main animation loop
    loop {
        let now = Instant::now();
        let _elapsed_total = now.duration_since(start_time).as_secs_f32();

        // Check for shutdown signal
        if SHUTDOWN.load(Ordering::Relaxed) {
            break;
        }

        // Frame timing
        let dt = now.duration_since(last_frame).as_secs_f32();
        if dt < frame_duration.as_secs_f32() {
            // Busy-wait for precision (no display server vsync available)
            std::thread::sleep(Duration::from_micros(500));
            continue;
        }
        last_frame = now;

        // Poll progress events
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

        // Handle final contraction animation
        if completing {
            if let Some(cs) = contraction_start {
                let t = now.duration_since(cs).as_secs_f32();
                if t < CONTRACTION_DURATION {
                    // Contraction phase: all threads pull toward center
                    network.contract(t / CONTRACTION_DURATION);
                    // Keep pulsing during contraction
                    if (t * 4.0) as u32 % 2 == 0 {
                        network.pulse();
                    }
                } else if t < CONTRACTION_DURATION + FLASH_HOLD {
                    // White flash hold
                    network.contract(1.0);
                } else if t < CONTRACTION_DURATION + FLASH_HOLD + FADE_DURATION {
                    // Fade to black
                    network.contract(1.0);
                } else {
                    // Done — exit
                    break;
                }
            }
        }

        // Grow the network
        let io_rate = progress.io_rate;
        network.grow(dt, io_rate);

        // Render
        network.render(&mut render_buf);

        // Blit to framebuffer
        fb.blit_from(&render_buf);
    }

    // Clean exit — clear to black
    for pixel in render_buf.iter_mut() {
        *pixel = 0;
    }
    fb.blit_from(&render_buf);

    eprintln!("quicken-fb: clean exit");
}

/// Parsed command-line arguments.
struct Args {
    genesis_phrase: String,
    progress_pipe: Option<String>,
    device: String,
}

/// Minimal argument parser (no clap dependency to keep binary small).
fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut genesis_phrase = None;
    let mut progress_pipe = None;
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
         \x20 --device <PATH>             DRM device path (default: /dev/dri/card0)\n\
         \x20 --help                      Show this help"
    );
}

/// Install SIGTERM and SIGINT handlers for clean shutdown.
fn install_signal_handlers() {
    // Use a simple atomic flag approach — safe for signal handlers
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
