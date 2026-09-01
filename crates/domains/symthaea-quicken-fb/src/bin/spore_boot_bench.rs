// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic headless benchmark for the Spore boot renderer.
//!
//! This intentionally excludes DRM/KMS so CI can compare simulation and CPU
//! raster cost reproducibly. Hardware qualification measures DRM separately.

use std::time::{Duration, Instant};

use symthaea_quicken_fb::mycelium::MycelialNetwork;
use symthaea_quicken_fb::perf::TimingSeries;

const FIXED_DT: f32 = 1.0 / 30.0;

fn main() {
    let args = Args::parse();
    let result = if args.matrix {
        let cases = [
            (1280, 720),
            (1920, 1080),
            (2560, 1440),
            (3840, 2160),
            (5120, 1440),
        ];
        serde_json::Value::Array(
            cases
                .into_iter()
                .map(|(width, height)| run_case(width, height, &args))
                .collect(),
        )
    } else {
        run_case(args.width, args.height, &args)
    };

    println!("{}", serde_json::to_string_pretty(&result).unwrap());
}

fn run_case(width: u32, height: u32, args: &Args) -> serde_json::Value {
    let mut network = MycelialNetwork::new(width, height, &args.seed);
    let mut buffer = vec![0_u32; width as usize * height as usize];

    for _ in 0..args.warmup_frames {
        network.grow(FIXED_DT, args.growth_rate);
        network.render(&mut buffer);
    }

    let mut grow = TimingSeries::with_capacity(args.frames as usize);
    let mut render = TimingSeries::with_capacity(args.frames as usize);
    let mut total = TimingSeries::with_capacity(args.frames as usize);

    for _ in 0..args.frames {
        let frame_start = Instant::now();

        let grow_start = Instant::now();
        network.grow(FIXED_DT, args.growth_rate);
        grow.record(grow_start.elapsed());

        let render_start = Instant::now();
        network.render(&mut buffer);
        render.record(render_start.elapsed());

        total.record(frame_start.elapsed());
    }

    let bytes_per_surface = width as u64 * height as u64 * 4;
    let fixed_dt_us = u64::try_from(Duration::from_secs_f32(FIXED_DT).as_micros())
        .unwrap_or(u64::MAX);
    serde_json::json!({
        "schema": "spore-boot-headless-benchmark-v1",
        "width": width,
        "height": height,
        "frames": args.frames,
        "warmup_frames": args.warmup_frames,
        "fixed_dt_us": fixed_dt_us,
        "growth_rate": args.growth_rate,
        "seed": args.seed.as_str(),
        "branch_count": network.branches.len(),
        "bytes_per_xrgb8888_surface": bytes_per_surface,
        "grow": grow.summary().to_json(),
        "render": render.summary().to_json(),
        "total_cpu_frame": total.summary().to_json(),
    })
}

struct Args {
    width: u32,
    height: u32,
    frames: u32,
    warmup_frames: u32,
    growth_rate: f32,
    seed: String,
    matrix: bool,
}

impl Args {
    fn parse() -> Self {
        let mut result = Self {
            width: 1920,
            height: 1080,
            frames: 300,
            warmup_frames: 60,
            growth_rate: 0.65,
            seed: "spore-boot-benchmark-v1".to_string(),
            matrix: false,
        };

        let mut args = std::env::args().skip(1);
        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--width" => result.width = parse_next(&mut args, "--width"),
                "--height" => result.height = parse_next(&mut args, "--height"),
                "--frames" => result.frames = parse_next(&mut args, "--frames"),
                "--warmup-frames" => {
                    result.warmup_frames = parse_next(&mut args, "--warmup-frames")
                }
                "--growth-rate" => result.growth_rate = parse_next(&mut args, "--growth-rate"),
                "--seed" => {
                    result.seed = args.next().unwrap_or_else(|| missing_value("--seed"));
                }
                "--matrix" => result.matrix = true,
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => {
                    eprintln!("spore-boot-bench: unknown argument {other}");
                    print_help();
                    std::process::exit(2);
                }
            }
        }

        if result.width == 0 || result.height == 0 || result.frames == 0 {
            eprintln!("spore-boot-bench: width, height, and frames must be non-zero");
            std::process::exit(2);
        }
        if !result.growth_rate.is_finite() || result.growth_rate < 0.0 {
            eprintln!("spore-boot-bench: growth rate must be finite and non-negative");
            std::process::exit(2);
        }
        result
    }
}

fn parse_next<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    flag: &'static str,
) -> T {
    let value = args.next().unwrap_or_else(|| missing_value(flag));
    value.parse().unwrap_or_else(|_| {
        eprintln!("spore-boot-bench: invalid value for {flag}: {value}");
        std::process::exit(2);
    })
}

fn missing_value(flag: &'static str) -> ! {
    eprintln!("spore-boot-bench: {flag} requires a value");
    std::process::exit(2)
}

fn print_help() {
    eprintln!(
        "Usage: spore-boot-bench [OPTIONS]\n\
         \n\
         Options:\n\
           --matrix                 Run the standard resolution matrix\n\
           --width <PX>             Width for a single case (default 1920)\n\
           --height <PX>            Height for a single case (default 1080)\n\
           --frames <N>             Measured frames (default 300)\n\
           --warmup-frames <N>      Deterministic unmeasured warmup (default 60)\n\
           --growth-rate <F>        Fixed growth input (default 0.65)\n\
           --seed <TEXT>            Deterministic scene seed\n\
           --help                   Show this help"
    );
}
