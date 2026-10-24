// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Offline capture for the exact CPU renderer used by the boot path.
//!
//! PPM is intentionally used as the zero-dependency capture format. The preview
//! directory also contains the serialized genome and a JSON manifest so frames
//! are reproducible and can be converted to PNG/video by external tooling.

use crate::ecology_renderer::EcologyRenderer;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use symthaea_boot_ecology::BootGenome;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreviewSummary {
    pub output_dir: PathBuf,
    pub frame_count: usize,
    pub width: u32,
    pub height: u32,
    pub fps: u16,
    pub duration_ms: u32,
}

/// Render every frame at `fps` using the exact ecology renderer.
pub fn render_preview(
    genome: BootGenome,
    output_dir: impl AsRef<Path>,
    width: u32,
    height: u32,
    fps: u16,
) -> io::Result<PreviewSummary> {
    if width == 0 || height == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "preview dimensions must be non-zero",
        ));
    }
    if fps == 0 || fps > 120 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "preview fps must be between 1 and 120",
        ));
    }

    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir)?;

    let renderer = EcologyRenderer::new(width, height, genome.clone());
    let duration_ms = renderer.total_duration_ms();
    let frame_count = ((duration_ms as u64 * fps as u64 + 999) / 1_000).max(1) as usize;
    let mut pixels = vec![0u32; width as usize * height as usize];

    for frame_index in 0..frame_count {
        let elapsed_ms = ((frame_index as u64 * 1_000) / fps as u64) as u32;
        renderer.render_at(elapsed_ms.min(duration_ms), &mut pixels);
        let path = output_dir.join(format!("frame-{frame_index:05}.ppm"));
        write_ppm(&path, width, height, &pixels)?;
    }

    let genome_json = serde_json::to_vec_pretty(&genome)
        .map_err(|error| io::Error::other(format!("serialize genome: {error}")))?;
    fs::write(output_dir.join("boot-genome.json"), genome_json)?;

    let manifest = serde_json::json!({
        "schema": "spore-boot-preview-v1",
        "genome_seed": genome.seed_hex(),
        "family": format!("{:?}", genome.family),
        "accent_family": genome.accent_family.map(|family| format!("{family:?}")),
        "cue": format!("{:?}", genome.cue),
        "width": width,
        "height": height,
        "fps": fps,
        "duration_ms": duration_ms,
        "frame_count": frame_count,
        "format": "PPM P6",
        "renderer": "symthaea-quicken-fb/ecology_renderer",
    });
    let manifest_json = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| io::Error::other(format!("serialize preview manifest: {error}")))?;
    fs::write(output_dir.join("preview-manifest.json"), manifest_json)?;

    Ok(PreviewSummary {
        output_dir: output_dir.to_path_buf(),
        frame_count,
        width,
        height,
        fps,
        duration_ms,
    })
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
        let rgb = [
            ((pixel >> 16) & 0xff) as u8,
            ((pixel >> 8) & 0xff) as u8,
            (pixel & 0xff) as u8,
        ];
        writer.write_all(&rgb)?;
    }
    writer.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_boot_ecology::{BootEcologyComposer, BootStateReceipt, MorphologyLineage};

    #[test]
    fn rejects_invalid_dimensions_and_fps() {
        let genome = BootEcologyComposer::compose(
            &BootStateReceipt::first_boot([1; 32]),
            &MorphologyLineage::default(),
        );
        let tmp = std::env::temp_dir().join("spore-preview-invalid");
        assert!(render_preview(genome.clone(), &tmp, 0, 100, 10).is_err());
        assert!(render_preview(genome, &tmp, 100, 100, 0).is_err());
    }
}
