// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea Visualization Tool
//!
//! Generate visualizations for spectrograms, discovered units, and HDC vectors.
//!
//! ```bash
//! # Visualize spectrogram
//! symthaea-viz spectrogram audio.wav -o spectrogram.svg
//!
//! # Visualize discovered units timeline
//! symthaea-viz timeline discovery_results.json -o timeline.html
//!
//! # Show HDC similarity matrix
//! symthaea-viz similarity models/prototypes.bin -o similarity.svg
//! ```

use clap::{Parser, Subcommand};
use console::style;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use symthaea_stt::{AudioConfig, AudioFrontend, HV16, TrainedPrototypes};

#[derive(Parser)]
#[command(name = "symthaea-viz")]
#[command(about = "Visualization tools for Symthaea STT")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Visualize audio spectrogram
    Spectrogram {
        /// Input audio file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file (svg, html, or txt for ASCII)
        #[arg(short, long, default_value = "spectrogram.svg")]
        output: PathBuf,

        /// Width in characters/pixels
        #[arg(long, default_value = "120")]
        width: usize,

        /// Height in characters/pixels
        #[arg(long, default_value = "40")]
        height: usize,
    },

    /// Visualize discovered units timeline
    Timeline {
        /// Discovery results JSON file
        #[arg(short, long)]
        input: PathBuf,

        /// Output HTML file
        #[arg(short, long, default_value = "timeline.html")]
        output: PathBuf,
    },

    /// Visualize HDC similarity matrix
    Similarity {
        /// Prototypes file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file (svg or txt)
        #[arg(short, long, default_value = "similarity.svg")]
        output: PathBuf,

        /// Maximum items to show
        #[arg(long, default_value = "20")]
        max_items: usize,
    },

    /// Visualize audio waveform
    Waveform {
        /// Input audio file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file
        #[arg(short, long, default_value = "waveform.svg")]
        output: PathBuf,

        /// Width
        #[arg(long, default_value = "800")]
        width: usize,

        /// Height
        #[arg(long, default_value = "200")]
        height: usize,
    },

    /// Generate evaluation report HTML
    Report {
        /// Evaluation results JSON
        #[arg(short, long)]
        input: PathBuf,

        /// Output HTML file
        #[arg(short, long, default_value = "report.html")]
        output: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Spectrogram {
            input,
            output,
            width,
            height,
        } => {
            if let Err(e) = visualize_spectrogram(&input, &output, width, height) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Timeline { input, output } => {
            if let Err(e) = visualize_timeline(&input, &output) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Similarity {
            input,
            output,
            max_items,
        } => {
            if let Err(e) = visualize_similarity(&input, &output, max_items) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Waveform {
            input,
            output,
            width,
            height,
        } => {
            if let Err(e) = visualize_waveform(&input, &output, width, height) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Report { input, output } => {
            if let Err(e) = generate_report(&input, &output) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
    }
}

// ============================================================================
// SPECTROGRAM VISUALIZATION
// ============================================================================

fn visualize_spectrogram(
    input: &Path,
    output: &Path,
    width: usize,
    height: usize,
) -> Result<(), String> {
    println!("{}Generating spectrogram...", style("").bold());

    // Load audio
    let (audio, _sample_rate) =
        AudioFrontend::load_audio(input).map_err(|e| format!("Failed to load audio: {}", e))?;

    // Extract features
    let mut frontend = AudioFrontend::new(AudioConfig::default());
    let features = frontend.extract_features(&audio);

    if features.is_empty() {
        return Err("No features extracted".to_string());
    }

    let ext = output.extension().and_then(|e| e.to_str()).unwrap_or("svg");

    match ext {
        "svg" => write_spectrogram_svg(&features, output, width, height),
        "txt" => write_spectrogram_ascii(&features, output, width, height),
        "html" => write_spectrogram_html(&features, output, width, height),
        _ => Err(format!("Unsupported output format: {}", ext)),
    }
}

fn write_spectrogram_svg(
    features: &[Vec<f32>],
    output: &Path,
    width: usize,
    height: usize,
) -> Result<(), String> {
    let num_frames = features.len();
    let num_mels = features.first().map(|f| f.len()).unwrap_or(0);

    // Normalize features
    let (min_val, max_val) = features
        .iter()
        .flat_map(|f| f.iter())
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &v| {
            (min.min(v), max.max(v))
        });
    let range = (max_val - min_val).max(1e-6);

    let cell_w = width as f32 / num_frames as f32;
    let cell_h = height as f32 / num_mels as f32;

    let mut svg = String::new();
    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        width, height, width, height
    ));
    svg.push_str(r##"<rect width="100%" height="100%" fill="#1a1a2e"/>"##);

    // Draw cells
    for (t, frame) in features.iter().enumerate() {
        for (f, &val) in frame.iter().enumerate() {
            let normalized = ((val - min_val) / range).clamp(0.0, 1.0);
            let color = viridis_color(normalized);

            let x = t as f32 * cell_w;
            let y = (num_mels - 1 - f) as f32 * cell_h; // Flip vertically

            svg.push_str(&format!(
                r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="{}"/>"#,
                x,
                y,
                cell_w + 0.5,
                cell_h + 0.5,
                color
            ));
        }
    }

    // Add axis labels
    svg.push_str(&format!(
        r#"<text x="5" y="{}" fill="white" font-size="12">Freq</text>"#,
        height / 2
    ));
    svg.push_str(&format!(
        r#"<text x="{}" y="{}" fill="white" font-size="12">Time</text>"#,
        width / 2,
        height - 5
    ));

    svg.push_str("</svg>");

    let mut file = File::create(output).map_err(|e| format!("Failed to create file: {}", e))?;
    file.write_all(svg.as_bytes())
        .map_err(|e| format!("Failed to write: {}", e))?;

    println!("Saved spectrogram to: {}", output.display());
    Ok(())
}

fn write_spectrogram_ascii(
    features: &[Vec<f32>],
    output: &Path,
    width: usize,
    height: usize,
) -> Result<(), String> {
    let num_frames = features.len();
    let num_mels = features.first().map(|f| f.len()).unwrap_or(0);

    // Resample to fit dimensions
    let step_t = (num_frames as f32 / width as f32).max(1.0);
    let step_f = (num_mels as f32 / height as f32).max(1.0);

    // Normalize
    let (min_val, max_val) = features
        .iter()
        .flat_map(|f| f.iter())
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &v| {
            (min.min(v), max.max(v))
        });
    let range = (max_val - min_val).max(1e-6);

    let chars = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];

    let mut output_str = String::new();

    for y in 0..height {
        let f = ((height - 1 - y) as f32 * step_f) as usize;
        for x in 0..width {
            let t = (x as f32 * step_t) as usize;

            if t < features.len() && f < features[t].len() {
                let val = features[t][f];
                let normalized = ((val - min_val) / range).clamp(0.0, 0.999);
                let idx = (normalized * chars.len() as f32) as usize;
                output_str.push(chars[idx]);
            } else {
                output_str.push(' ');
            }
        }
        output_str.push('\n');
    }

    let mut file = File::create(output).map_err(|e| format!("Failed to create file: {}", e))?;
    file.write_all(output_str.as_bytes())
        .map_err(|e| format!("Failed to write: {}", e))?;

    println!("Saved ASCII spectrogram to: {}", output.display());
    Ok(())
}

fn write_spectrogram_html(
    features: &[Vec<f32>],
    output: &Path,
    width: usize,
    height: usize,
) -> Result<(), String> {
    let mut svg = String::new();

    // Generate SVG content
    let num_frames = features.len();
    let num_mels = features.first().map(|f| f.len()).unwrap_or(0);

    let (min_val, max_val) = features
        .iter()
        .flat_map(|f| f.iter())
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &v| {
            (min.min(v), max.max(v))
        });
    let range = (max_val - min_val).max(1e-6);

    let cell_w = width as f32 / num_frames as f32;
    let cell_h = height as f32 / num_mels as f32;

    svg.push_str(&format!(
        r#"<svg width="{}" height="{}" viewBox="0 0 {} {}">"#,
        width, height, width, height
    ));
    svg.push_str(r##"<rect width="100%" height="100%" fill="#1a1a2e"/>"##);

    for (t, frame) in features.iter().enumerate() {
        for (f, &val) in frame.iter().enumerate() {
            let normalized = ((val - min_val) / range).clamp(0.0, 1.0);
            let color = viridis_color(normalized);
            let x = t as f32 * cell_w;
            let y = (num_mels - 1 - f) as f32 * cell_h;
            svg.push_str(&format!(
                r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="{}"/>"#,
                x,
                y,
                cell_w + 0.5,
                cell_h + 0.5,
                color
            ));
        }
    }
    svg.push_str("</svg>");

    let html = format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <title>Spectrogram - Symthaea</title>
    <style>
        body {{ background: #0d1117; color: #c9d1d9; font-family: system-ui; padding: 20px; }}
        h1 {{ color: #58a6ff; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .spectrogram {{ background: #161b22; padding: 20px; border-radius: 8px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Mel Spectrogram</h1>
        <div class="spectrogram">
            {}
        </div>
        <p>Frames: {} | Mel bins: {} | Duration: {:.2}s</p>
    </div>
</body>
</html>"#,
        svg,
        num_frames,
        num_mels,
        num_frames as f32 * 0.01
    );

    let mut file = File::create(output).map_err(|e| format!("Failed to create file: {}", e))?;
    file.write_all(html.as_bytes())
        .map_err(|e| format!("Failed to write: {}", e))?;

    println!("Saved HTML spectrogram to: {}", output.display());
    Ok(())
}

// ============================================================================
// TIMELINE VISUALIZATION
// ============================================================================

fn visualize_timeline(input: &Path, output: &Path) -> Result<(), String> {
    println!("{}Generating timeline...", style("").bold());

    let content =
        std::fs::read_to_string(input).map_err(|e| format!("Failed to read file: {}", e))?;

    let json: serde_json::Value =
        serde_json::from_str(&content).map_err(|e| format!("Failed to parse JSON: {}", e))?;

    let segments = json
        .get("segments")
        .and_then(|s| s.as_array())
        .ok_or("No segments found")?;

    let units = json
        .get("units")
        .and_then(|u| u.as_array())
        .ok_or("No units found")?;

    // Generate colors for units
    let unit_ids: Vec<String> = units
        .iter()
        .filter_map(|u| u.get("id").and_then(|v| v.as_str()))
        .map(|s| s.to_string())
        .collect();

    let mut unit_colors: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for (i, id) in unit_ids.iter().enumerate() {
        let hue = (i * 137) % 360; // Golden angle for distinct colors
        unit_colors.insert(id.clone(), format!("hsl({}, 70%, 50%)", hue));
    }

    // Find time range
    let max_time = segments
        .iter()
        .filter_map(|s| s.get("end").and_then(|v| v.as_f64()))
        .fold(0.0f64, |a, b| a.max(b));

    let width = 1000;
    let height = 150;
    let timeline_y = 50;
    let timeline_h = 60;

    let mut svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        width, height, width, height
    );
    svg.push_str(r##"<rect width="100%" height="100%" fill="#1a1a2e"/>"##);

    // Draw segments
    for seg in segments {
        let start = seg.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let end = seg.get("end").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let unit = seg.get("unit").and_then(|v| v.as_str()).unwrap_or("?");

        let x = (start / max_time * (width - 100) as f64) as f32 + 50.0;
        let w = ((end - start) / max_time * (width - 100) as f64) as f32;
        let color = unit_colors.get(unit).map(|s| s.as_str()).unwrap_or("#666");

        svg.push_str(&format!(
            r##"<rect x="{:.1}" y="{}" width="{:.1}" height="{}" fill="{}" stroke="#fff" stroke-width="0.5"/>"##,
            x, timeline_y, w.max(2.0), timeline_h, color
        ));

        // Label if wide enough
        if w > 30.0 {
            svg.push_str(&format!(
                r#"<text x="{:.1}" y="{}" fill="white" font-size="10" text-anchor="middle">{}</text>"#,
                x + w / 2.0, timeline_y + timeline_h / 2 + 3, unit
            ));
        }
    }

    // Time axis
    svg.push_str(&format!(
        r##"<line x1="50" y1="{}" x2="{}" y2="{}" stroke="#666"/>"##,
        timeline_y + timeline_h + 10,
        width - 50,
        timeline_y + timeline_h + 10
    ));

    // Time labels
    for i in 0..=5 {
        let t = max_time * i as f64 / 5.0;
        let x = 50.0 + (width - 100) as f64 * i as f64 / 5.0;
        svg.push_str(&format!(
            r##"<text x="{:.0}" y="{}" fill="#888" font-size="10" text-anchor="middle">{:.1}s</text>"##,
            x, timeline_y + timeline_h + 25, t
        ));
    }

    svg.push_str("</svg>");

    // Wrap in HTML
    let html = format!(r#"<!DOCTYPE html>
<html>
<head>
    <title>Discovery Timeline - Symthaea</title>
    <style>
        body {{ background: #0d1117; color: #c9d1d9; font-family: system-ui; padding: 20px; }}
        h1 {{ color: #58a6ff; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .timeline {{ background: #161b22; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .legend {{ display: flex; flex-wrap: wrap; gap: 10px; }}
        .legend-item {{ display: flex; align-items: center; gap: 5px; }}
        .color-box {{ width: 20px; height: 20px; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Discovered Units Timeline</h1>
        <div class="timeline">
            {}
        </div>
        <h2>Legend</h2>
        <div class="legend">
            {}
        </div>
        <p>Total segments: {} | Unique units: {}</p>
    </div>
</body>
</html>"#,
        svg,
        unit_colors.iter()
            .map(|(id, color)| format!(
                r#"<div class="legend-item"><div class="color-box" style="background: {}"></div>{}</div>"#,
                color, id
            ))
            .collect::<Vec<_>>()
            .join("\n"),
        segments.len(),
        unit_ids.len()
    );

    let mut file = File::create(output).map_err(|e| format!("Failed to create file: {}", e))?;
    file.write_all(html.as_bytes())
        .map_err(|e| format!("Failed to write: {}", e))?;

    println!("Saved timeline to: {}", output.display());
    Ok(())
}

// ============================================================================
// SIMILARITY MATRIX
// ============================================================================

fn visualize_similarity(input: &Path, output: &Path, max_items: usize) -> Result<(), String> {
    println!("{}Generating similarity matrix...", style("").bold());

    let prototypes =
        TrainedPrototypes::load(input).map_err(|e| format!("Failed to load prototypes: {}", e))?;

    let items: Vec<(String, HV16)> = prototypes.as_pairs().into_iter().take(max_items).collect();

    if items.is_empty() {
        return Err("No prototypes found".to_string());
    }

    let n = items.len();
    let cell_size = 30;
    let margin = 80;
    let width = n * cell_size + margin;
    let height = n * cell_size + margin;

    let mut svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        width, height, width, height
    );
    svg.push_str(r##"<rect width="100%" height="100%" fill="#1a1a2e"/>"##);

    // Draw matrix
    for (i, (name_i, hv_i)) in items.iter().enumerate() {
        // Row label
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" fill="white" font-size="10" text-anchor="end">{}</text>"#,
            margin - 5,
            margin + i * cell_size + cell_size / 2 + 3,
            name_i
        ));

        // Column label
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" fill="white" font-size="10" text-anchor="start" transform="rotate(-45 {} {})">{}</text>"#,
            margin + i * cell_size + cell_size / 2, margin - 5,
            margin + i * cell_size + cell_size / 2, margin - 5,
            name_i
        ));

        for (j, (_, hv_j)) in items.iter().enumerate() {
            let sim = hv_i.similarity(hv_j);
            let color = similarity_color(sim);

            let x = margin + j * cell_size;
            let y = margin + i * cell_size;

            svg.push_str(&format!(
                r##"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" stroke="#333"/>"##,
                x,
                y,
                cell_size - 1,
                cell_size - 1,
                color
            ));

            // Show value on diagonal
            if i == j {
                svg.push_str(&format!(
                    r#"<text x="{}" y="{}" fill="white" font-size="8" text-anchor="middle">{:.2}</text>"#,
                    x + cell_size / 2, y + cell_size / 2 + 3, sim
                ));
            }
        }
    }

    svg.push_str("</svg>");

    let ext = output.extension().and_then(|e| e.to_str()).unwrap_or("svg");

    if ext == "html" {
        let html = format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>Similarity Matrix - Symthaea</title>
    <style>
        body {{ background: #0d1117; color: #c9d1d9; font-family: system-ui; padding: 20px; }}
        h1 {{ color: #58a6ff; }}
    </style>
</head>
<body>
    <h1>HDC Similarity Matrix</h1>
    {}
    <p>Showing {} prototypes</p>
</body>
</html>"#,
            svg, n
        );

        let mut file = File::create(output).map_err(|e| format!("Failed to create file: {}", e))?;
        file.write_all(html.as_bytes())
            .map_err(|e| format!("Failed to write: {}", e))?;
    } else {
        let mut file = File::create(output).map_err(|e| format!("Failed to create file: {}", e))?;
        file.write_all(svg.as_bytes())
            .map_err(|e| format!("Failed to write: {}", e))?;
    }

    println!("Saved similarity matrix to: {}", output.display());
    Ok(())
}

// ============================================================================
// WAVEFORM VISUALIZATION
// ============================================================================

fn visualize_waveform(
    input: &Path,
    output: &Path,
    width: usize,
    height: usize,
) -> Result<(), String> {
    println!("{}Generating waveform...", style("").bold());

    let (audio, sample_rate) =
        AudioFrontend::load_audio(input).map_err(|e| format!("Failed to load audio: {}", e))?;

    let duration = audio.len() as f32 / sample_rate as f32;

    // Downsample for visualization
    let samples_per_pixel = audio.len() / width;
    let mut peaks: Vec<(f32, f32)> = Vec::with_capacity(width);

    for chunk in audio.chunks(samples_per_pixel.max(1)) {
        let min = chunk.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = chunk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        peaks.push((min, max));
    }

    let mid_y = height as f32 / 2.0;
    let scale = (height as f32 / 2.0) * 0.9;

    let mut svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        width, height, width, height
    );
    svg.push_str(r##"<rect width="100%" height="100%" fill="#1a1a2e"/>"##);

    // Center line
    svg.push_str(&format!(
        r##"<line x1="0" y1="{:.0}" x2="{}" y2="{:.0}" stroke="#333"/>"##,
        mid_y, width, mid_y
    ));

    // Draw waveform
    let mut path = format!("M 0 {:.1}", mid_y);
    for (i, &(min, max)) in peaks.iter().enumerate() {
        let x = i as f32;
        let y_min = mid_y - min * scale;
        let y_max = mid_y - max * scale;
        path.push_str(&format!(" L {:.1} {:.1} L {:.1} {:.1}", x, y_min, x, y_max));
    }

    svg.push_str(&format!(
        r##"<path d="{}" fill="none" stroke="#58a6ff" stroke-width="1"/>"##,
        path
    ));

    svg.push_str("</svg>");

    let html = format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <title>Waveform - Symthaea</title>
    <style>
        body {{ background: #0d1117; color: #c9d1d9; font-family: system-ui; padding: 20px; }}
        h1 {{ color: #58a6ff; }}
    </style>
</head>
<body>
    <h1>Audio Waveform</h1>
    {}
    <p>Duration: {:.2}s | Sample rate: {} Hz | Samples: {}</p>
</body>
</html>"#,
        svg,
        duration,
        sample_rate,
        audio.len()
    );

    let mut file = File::create(output).map_err(|e| format!("Failed to create file: {}", e))?;
    file.write_all(html.as_bytes())
        .map_err(|e| format!("Failed to write: {}", e))?;

    println!("Saved waveform to: {}", output.display());
    Ok(())
}

// ============================================================================
// REPORT GENERATION
// ============================================================================

fn generate_report(input: &Path, output: &Path) -> Result<(), String> {
    println!("{}Generating evaluation report...", style("").bold());

    let content =
        std::fs::read_to_string(input).map_err(|e| format!("Failed to read file: {}", e))?;

    let json: serde_json::Value =
        serde_json::from_str(&content).map_err(|e| format!("Failed to parse JSON: {}", e))?;

    let html = format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <title>Evaluation Report - Symthaea</title>
    <style>
        body {{ background: #0d1117; color: #c9d1d9; font-family: system-ui; padding: 20px; }}
        h1, h2 {{ color: #58a6ff; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        .card {{ background: #161b22; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .metric {{ font-size: 2em; color: #7ee787; }}
        .label {{ color: #8b949e; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #30363d; }}
        th {{ color: #58a6ff; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Symthaea STT Evaluation Report</h1>

        <div class="card">
            <h2>Summary</h2>
            <pre>{}</pre>
        </div>
    </div>
</body>
</html>"#,
        serde_json::to_string_pretty(&json).unwrap_or_default()
    );

    let mut file = File::create(output).map_err(|e| format!("Failed to create file: {}", e))?;
    file.write_all(html.as_bytes())
        .map_err(|e| format!("Failed to write: {}", e))?;

    println!("Saved report to: {}", output.display());
    Ok(())
}

// ============================================================================
// COLOR UTILITIES
// ============================================================================

fn viridis_color(t: f32) -> String {
    // Simplified viridis colormap
    let t = t.clamp(0.0, 1.0);

    let r = (0.267 + t * (0.993 - 0.267 + t * (-0.876))).clamp(0.0, 1.0);
    let g = (0.004 + t * (0.906 - 0.004 + t * (-0.349))).clamp(0.0, 1.0);
    let b = (0.329 + t * (0.143 - 0.329 + t * (0.658))).clamp(0.0, 1.0);

    format!(
        "#{:02x}{:02x}{:02x}",
        (r * 255.0) as u8,
        (g * 255.0) as u8,
        (b * 255.0) as u8
    )
}

fn similarity_color(sim: f32) -> String {
    // Blue (low) to green (high) gradient
    let t = sim.clamp(0.0, 1.0);

    let r = ((1.0 - t) * 0.2) as u8 * 255;
    let g = (t * 0.8 + 0.2) * 255.0;
    let b = ((1.0 - t) * 0.8) * 255.0;

    format!("#{:02x}{:02x}{:02x}", { r }, g as u8, b as u8)
}
