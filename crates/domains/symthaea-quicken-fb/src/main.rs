// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Spore state-aware boot renderer.
//!
//! Live rendering is fail-open: inability to read an ecology receipt or acquire
//! DRM never becomes authority to block system startup. Preview mode is strict
//! and returns a non-zero status on invalid input because it is an operator tool.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use symthaea_boot_ecology::{BootEcologyComposer, BootStateReceipt, MorphologyLineage};
use symthaea_quicken_fb::ecology_renderer::EcologyRenderer;
use symthaea_quicken_fb::framebuffer::DrmFramebuffer;
use symthaea_quicken_fb::mycelium::MycelialNetwork;
use symthaea_quicken_fb::preview::render_preview;
use symthaea_quicken_fb::progress::{ProgressEvent, ProgressMonitor};

const LEGACY_TARGET_FPS: u32 = 30;
const LEGACY_HARD_DEADLINE: Duration = Duration::from_secs(9);

static SHUTDOWN: AtomicBool = AtomicBool::new(false);

fn main() {
    let command = match parse_args() {
        Ok(command) => command,
        Err(error) => {
            eprintln!("quicken-fb: {error}");
            print_usage();
            std::process::exit(2);
        }
    };

    match command {
        Command::Preview(args) => {
            if let Err(error) = run_preview(args) {
                eprintln!("quicken-fb preview: {error}");
                std::process::exit(1);
            }
        }
        Command::Live(args) => {
            install_signal_handlers();
            if let Err(error) = run_live(args) {
                // Boot rendering is explicitly fail-open. The service may log a
                // renderer failure, but the boot path remains successful.
                eprintln!("quicken-fb: renderer skipped: {error}");
            }
        }
    }
}

#[derive(Debug)]
enum Command {
    Live(LiveArgs),
    Preview(PreviewArgs),
}

#[derive(Debug)]
struct LiveArgs {
    source: RenderSource,
    progress_pipe: Option<PathBuf>,
    device: String,
}

#[derive(Debug)]
enum RenderSource {
    Ecology {
        receipt: PathBuf,
        lineage: Option<PathBuf>,
    },
    LegacyGenesis(String),
}

#[derive(Debug)]
struct PreviewArgs {
    receipt: PathBuf,
    lineage: Option<PathBuf>,
    output_dir: PathBuf,
    width: u32,
    height: u32,
    fps: u16,
}

fn run_preview(args: PreviewArgs) -> Result<(), String> {
    let receipt = read_json::<BootStateReceipt>(&args.receipt)?;
    let lineage = match args.lineage {
        Some(path) => read_json::<MorphologyLineage>(&path)?,
        None => MorphologyLineage::default(),
    };
    let genome = BootEcologyComposer::compose(&receipt, &lineage);
    let summary = render_preview(
        genome,
        &args.output_dir,
        args.width,
        args.height,
        args.fps,
    )
    .map_err(|error| error.to_string())?;

    eprintln!(
        "quicken-fb preview: {} frames, {}x{} @ {}fps, {}ms -> {}",
        summary.frame_count,
        summary.width,
        summary.height,
        summary.fps,
        summary.duration_ms,
        summary.output_dir.display()
    );
    Ok(())
}

fn run_live(args: LiveArgs) -> Result<(), String> {
    let (mut fb, resolved_device) = open_framebuffer(&args.device)?;
    eprintln!(
        "quicken-fb: display {}x{} @ {}Hz on {}",
        fb.width,
        fb.height,
        fb.mode.vrefresh(),
        resolved_device,
    );

    let mut progress = ProgressMonitor::new(args.progress_pipe.as_deref().and_then(Path::to_str));
    match args.source {
        RenderSource::Ecology { receipt, lineage } => {
            let receipt = read_json::<BootStateReceipt>(&receipt)?;
            let lineage = match lineage {
                Some(path) => read_json::<MorphologyLineage>(&path)?,
                None => MorphologyLineage::default(),
            };
            let genome = BootEcologyComposer::compose(&receipt, &lineage);
            eprintln!(
                "quicken-fb: ecology {:?} seed={} cue={:?}",
                genome.family,
                genome.seed_hex(),
                genome.cue,
            );
            run_ecology_loop(&mut fb, genome, &mut progress);
        }
        RenderSource::LegacyGenesis(genesis_phrase) => {
            run_legacy_loop(&mut fb, &genesis_phrase, &mut progress);
        }
    }

    clear_to_black(&mut fb);
    eprintln!("quicken-fb: clean exit");
    Ok(())
}

fn run_ecology_loop(
    fb: &mut DrmFramebuffer,
    genome: symthaea_boot_ecology::BootGenome,
    progress: &mut ProgressMonitor,
) {
    let fps = genome.render_policy.target_fps.max(1) as u64;
    let hard_deadline = Duration::from_millis(genome.render_policy.hard_deadline_ms as u64);
    let renderer = EcologyRenderer::new(fb.width, fb.height, genome);
    let sequence_duration = Duration::from_millis(renderer.total_duration_ms() as u64);
    let frame_duration = Duration::from_nanos(1_000_000_000 / fps);
    let mut render_buf = vec![0u32; (fb.width * fb.height) as usize];
    let start = Instant::now();
    let mut next_frame = start;

    loop {
        if SHUTDOWN.load(Ordering::Relaxed) {
            break;
        }
        let now = Instant::now();
        let elapsed = now.duration_since(start);
        if elapsed >= sequence_duration || elapsed >= hard_deadline {
            break;
        }
        if now < next_frame {
            std::thread::sleep((next_frame - now).min(Duration::from_millis(2)));
            continue;
        }
        next_frame += frame_duration;

        // Progress is decorative and optional. No event source can block or
        // extend the renderer lifetime.
        for event in progress.poll() {
            match event {
                ProgressEvent::DerivationComplete(name) => {
                    eprintln!("quicken-fb: derivation complete: {name}");
                }
                ProgressEvent::PhaseChange(phase) => {
                    eprintln!("quicken-fb: phase: {phase}");
                }
                ProgressEvent::InstallComplete => {
                    eprintln!("quicken-fb: installation complete");
                }
                _ => {}
            }
        }

        let elapsed_ms = elapsed.as_millis().min(u32::MAX as u128) as u32;
        renderer.render_at(elapsed_ms, &mut render_buf);
        fb.blit_from(&render_buf);
    }
}

fn run_legacy_loop(
    fb: &mut DrmFramebuffer,
    genesis_phrase: &str,
    progress: &mut ProgressMonitor,
) {
    let mut network = MycelialNetwork::new(fb.width, fb.height, genesis_phrase);
    let mut render_buf = vec![0u32; (fb.width * fb.height) as usize];
    let frame_duration = Duration::from_nanos(1_000_000_000 / LEGACY_TARGET_FPS as u64);
    let start = Instant::now();
    let mut last_frame = Instant::now();

    while start.elapsed() < LEGACY_HARD_DEADLINE && !SHUTDOWN.load(Ordering::Relaxed) {
        let now = Instant::now();
        let dt = now.duration_since(last_frame).as_secs_f32();
        if dt < frame_duration.as_secs_f32() {
            std::thread::sleep(Duration::from_micros(500));
            continue;
        }
        last_frame = now;

        for event in progress.poll() {
            match event {
                ProgressEvent::DerivationComplete(name) => {
                    network.pulse();
                    eprintln!("quicken-fb: derivation complete: {name}");
                }
                ProgressEvent::PhaseChange(phase) => {
                    eprintln!("quicken-fb: phase: {phase}");
                }
                ProgressEvent::InstallComplete => {
                    network.pulse();
                    eprintln!("quicken-fb: installation complete");
                }
                _ => {}
            }
        }

        network.grow(dt, progress.io_rate);
        network.render(&mut render_buf);
        fb.blit_from(&render_buf);
    }
}

fn open_framebuffer(device: &str) -> Result<(DrmFramebuffer, String), String> {
    if device != "auto" {
        return DrmFramebuffer::open(device)
            .map(|fb| (fb, device.to_string()))
            .map_err(|error| error.to_string());
    }

    let mut errors = Vec::new();
    for index in 0..16 {
        let path = format!("/dev/dri/card{index}");
        if !Path::new(&path).exists() {
            continue;
        }
        match DrmFramebuffer::open(&path) {
            Ok(fb) => return Ok((fb, path)),
            Err(error) => errors.push(format!("{path}: {error}")),
        }
    }

    if errors.is_empty() {
        Err("no DRM/KMS card devices found".to_string())
    } else {
        Err(format!("no usable DRM/KMS card: {}", errors.join("; ")))
    }
}

fn clear_to_black(fb: &mut DrmFramebuffer) {
    let render_buf = vec![0u32; (fb.width * fb.height) as usize];
    fb.blit_from(&render_buf);
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("parse {}: {error}", path.display()))
}

fn parse_args() -> Result<Command, String> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).is_some_and(|arg| arg == "preview") {
        return parse_preview_args(&args[2..]).map(Command::Preview);
    }
    parse_live_args(&args[1..]).map(Command::Live)
}

fn parse_live_args(args: &[String]) -> Result<LiveArgs, String> {
    let mut genesis_phrase = None;
    let mut receipt = None;
    let mut lineage = None;
    let mut progress_pipe = None;
    let mut device = "auto".to_string();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--genesis-phrase" => genesis_phrase = Some(next_value(args, &mut i, "--genesis-phrase")?),
            "--receipt" => receipt = Some(PathBuf::from(next_value(args, &mut i, "--receipt")?)),
            "--lineage" => lineage = Some(PathBuf::from(next_value(args, &mut i, "--lineage")?)),
            "--progress-pipe" => {
                progress_pipe = Some(PathBuf::from(next_value(args, &mut i, "--progress-pipe")?))
            }
            "--device" => device = next_value(args, &mut i, "--device")?,
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }

    let source = match (receipt, genesis_phrase) {
        (Some(receipt), _) => RenderSource::Ecology { receipt, lineage },
        (None, Some(phrase)) => RenderSource::LegacyGenesis(phrase),
        (None, None) => {
            return Err("live mode requires --receipt or --genesis-phrase".to_string());
        }
    };

    Ok(LiveArgs {
        source,
        progress_pipe,
        device,
    })
}

fn parse_preview_args(args: &[String]) -> Result<PreviewArgs, String> {
    let mut receipt = None;
    let mut lineage = None;
    let mut output_dir = PathBuf::from("spore-boot-preview");
    let mut width = 640u32;
    let mut height = 360u32;
    let mut fps = 10u16;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--receipt" => receipt = Some(PathBuf::from(next_value(args, &mut i, "--receipt")?)),
            "--lineage" => lineage = Some(PathBuf::from(next_value(args, &mut i, "--lineage")?)),
            "--out" => output_dir = PathBuf::from(next_value(args, &mut i, "--out")?),
            "--width" => width = parse_number(&next_value(args, &mut i, "--width")?, "width")?,
            "--height" => height = parse_number(&next_value(args, &mut i, "--height")?, "height")?,
            "--fps" => fps = parse_number(&next_value(args, &mut i, "--fps")?, "fps")?,
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown preview argument: {other}")),
        }
        i += 1;
    }

    Ok(PreviewArgs {
        receipt: receipt.ok_or_else(|| "preview mode requires --receipt".to_string())?,
        lineage,
        output_dir,
        width,
        height,
        fps,
    })
}

fn next_value(args: &[String], index: &mut usize, flag: &str) -> Result<String, String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_number<T>(value: &str, label: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse::<T>()
        .map_err(|error| format!("invalid {label} {value:?}: {error}"))
}

fn print_usage() {
    eprintln!(
        "Spore boot renderer\n\
         \n\
         Live state-aware mode:\n\
           quicken-fb --receipt <BOOT_STATE.json> [--lineage <LINEAGE.json>] [OPTIONS]\n\
         \n\
         Exact offline preview:\n\
           quicken-fb preview --receipt <BOOT_STATE.json> [--lineage <LINEAGE.json>] \\\n             [--out DIR] [--width 640] [--height 360] [--fps 10]\n\
         \n\
         Legacy compatibility:\n\
           quicken-fb --genesis-phrase <PHRASE> [OPTIONS]\n\
         \n\
         Live options:\n\
           --device <PATH|auto>        DRM device selection (default: auto)\n\
           --progress-pipe <PATH>      Optional non-authoritative progress events\n\
         \n\
         The live renderer is fail-open and time-bounded."
    );
}

fn install_signal_handlers() {
    unsafe {
        nix::libc::signal(
            nix::libc::SIGTERM,
            signal_handler as nix::libc::sighandler_t,
        );
        nix::libc::signal(
            nix::libc::SIGINT,
            signal_handler as nix::libc::sighandler_t,
        );
    }
}

extern "C" fn signal_handler(_sig: std::ffi::c_int) {
    SHUTDOWN.store(true, Ordering::Relaxed);
}
