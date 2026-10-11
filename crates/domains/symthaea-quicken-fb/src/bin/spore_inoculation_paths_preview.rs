// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Exact visual review matrix for install-path-specific Inoculation ceremonies.

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use symthaea_boot_ecology::{BootEcologyComposer, BootStateReceipt, MorphologyLineage};
use symthaea_quicken_fb::inoculation_ceremony::{
    InoculationCeremonyRenderer, InoculationPath,
};
use symthaea_quicken_fb::inoculation_renderer::InoculationPhase;

fn main() {
    if let Err(error) = run() {
        eprintln!("spore_inoculation_paths_preview: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let mut out = PathBuf::from("spore-inoculation-paths-preview");
    let mut width = 640u32;
    let mut height = 360u32;

    let mut index = 1usize;
    while index < args.len() {
        match args[index].as_str() {
            "--out" => out = PathBuf::from(next_value(&args, &mut index, "--out")?),
            "--width" => width = parse_number(&next_value(&args, &mut index, "--width")?, "width")?,
            "--height" => {
                height = parse_number(&next_value(&args, &mut index, "--height")?, "height")?
            }
            "--help" | "-h" => {
                eprintln!("spore_inoculation_paths_preview [--out DIR] [--width N] [--height N]");
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        index += 1;
    }
    if width == 0 || height == 0 {
        return Err("preview dimensions must be non-zero".to_string());
    }

    fs::create_dir_all(&out).map_err(|error| error.to_string())?;
    let receipt = BootStateReceipt::first_boot([0xa7; 32]);
    let genome = BootEcologyComposer::compose(&receipt, &MorphologyLineage::default());
    let representative = [
        (InoculationPhase::PreparingSubstrate, 0.32f32, 1_100u32),
        (InoculationPhase::WeavingSystem, 0.64f32, 2_100u32),
        (InoculationPhase::Finalizing, 0.92f32, 3_600u32),
    ];
    let mut cards = Vec::new();

    for path in InoculationPath::ALL {
        let renderer = InoculationCeremonyRenderer::new(width, height, genome.clone(), path);
        for (phase, progress, elapsed_ms) in representative {
            let label = format!("{}--{}", path.label(), phase.label());
            let case_dir = out.join(&label);
            fs::create_dir_all(&case_dir).map_err(|error| error.to_string())?;
            let mut pixels = vec![0u32; width as usize * height as usize];
            renderer.render(phase, progress, elapsed_ms, &mut pixels);
            let relative = format!("{label}/frame-00.ppm");
            write_ppm(&out.join(&relative), width, height, &pixels)
                .map_err(|error| error.to_string())?;
            cards.push(serde_json::json!({
                "phase": label,
                "path": path.label(),
                "lifecycle_phase": phase.label(),
                "progress": progress,
                "frames": [relative],
            }));
        }
    }

    let manifest = serde_json::json!({
        "schema": "spore-inoculation-paths-preview-v1",
        "renderer": "symthaea-quicken-fb/inoculation_ceremony",
        "width": width,
        "height": height,
        "samples_per_phase": 1,
        "paths": InoculationPath::ALL.len(),
        "representative_phases_per_path": representative.len(),
        "phases": cards,
    });
    fs::write(
        out.join("inoculation-manifest.json"),
        serde_json::to_vec_pretty(&manifest).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;

    eprintln!(
        "spore_inoculation_paths_preview: {} paths x {} phases -> {}",
        InoculationPath::ALL.len(),
        representative.len(),
        out.display()
    );
    Ok(())
}

fn write_ppm(path: &Path, width: u32, height: u32, pixels: &[u32]) -> io::Result<()> {
    let expected = width as usize * height as usize;
    if pixels.len() < expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "pixel buffer shorter than preview dimensions",
        ));
    }
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    write!(writer, "P6\n{width} {height}\n255\n")?;
    for pixel in pixels.iter().take(expected) {
        writer.write_all(&[
            ((pixel >> 16) & 0xff) as u8,
            ((pixel >> 8) & 0xff) as u8,
            (pixel & 0xff) as u8,
        ])?;
    }
    writer.flush()
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
