// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea Discovery CLI
//!
//! Unsupervised acoustic unit discovery for any species.
//!
//! # Usage
//!
//! ```bash
//! # Discover units in whale recordings
//! symthaea-discover --input whale_songs.wav --species cetacean
//!
//! # Discover units in bird songs
//! symthaea-discover --input bird_calls.wav --species avian
//!
//! # Unknown signals (most flexible; performs clustering, not translation)
//! symthaea-discover --input unknown_signal.wav --species unknown
//!
//! # With visual grounding
//! symthaea-discover --input primate_calls.wav \
//!     --visual-log contexts.json \
//!     --output-grounded meanings.json
//! ```

use clap::{Parser, ValueEnum};
use console::{Emoji, style};
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

// ============================================================================
// CLI STRUCTURE
// ============================================================================

#[derive(Parser)]
#[command(name = "symthaea-discover")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(author = "Luminous Dynamics")]
#[command(about = "Experimental unsupervised acoustic unit discovery")]
#[command(long_about = r#"
╔═══════════════════════════════════════════════════════════════════════╗
║                   SYMTHAEA DISCOVERY PROTOCOL                          ║
║                                                                        ║
║   "Listening to physics, not linguistics"                             ║
║                                                                        ║
║   Discovers natural acoustic units in any signal without prior        ║
║   knowledge of the species, language, or communication system.        ║
║                                                                        ║
║   Works on: Dolphins, Whales, Birds, Primates, Unknown signals        ║
║                                                                        ║
╚═══════════════════════════════════════════════════════════════════════╝
"#)]
struct Cli {
    /// Input audio file (WAV or FLAC)
    #[arg(short, long)]
    input: PathBuf,

    /// Output directory for results
    #[arg(short, long, default_value = "./discovery")]
    output: PathBuf,

    /// Species/signal type (affects default parameters)
    #[arg(short, long, value_enum, default_value = "unknown")]
    species: Species,

    /// Similarity threshold for clustering (0.0-1.0)
    #[arg(long)]
    threshold: Option<f32>,

    /// Maximum number of units to discover
    #[arg(long)]
    max_units: Option<usize>,

    /// Minimum segment duration (ms)
    #[arg(long)]
    min_duration: Option<f32>,

    /// Maximum segment duration (ms)
    #[arg(long)]
    max_duration: Option<f32>,

    /// Visual context log (JSON file with timestamp → label mappings)
    #[arg(long)]
    visual_log: Option<PathBuf>,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Quiet mode
    #[arg(short, long)]
    quiet: bool,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Species {
    /// Dolphins, whales (5ms - 10s, wide frequency range)
    Cetacean,
    /// Birds (20ms - 2s, melodic patterns)
    Avian,
    /// Apes, monkeys (50ms - 3s, complex calls)
    Primate,
    /// Unknown signals (most flexible parameters)
    #[value(alias = "xenolinguistic")]
    Unknown,
    /// Human speech (for comparison)
    Human,
}

// ============================================================================
// EMOJI
// ============================================================================

#[allow(dead_code)]
static ROCKET: Emoji<'_, '_> = Emoji("🚀 ", "");
static WAVE: Emoji<'_, '_> = Emoji("🌊 ", "");
static BRAIN: Emoji<'_, '_> = Emoji("🧠 ", "");
static CHECK: Emoji<'_, '_> = Emoji("✅ ", "");
static WARN: Emoji<'_, '_> = Emoji("⚠️  ", "");
static EAR: Emoji<'_, '_> = Emoji("👂 ", "");
static CHART: Emoji<'_, '_> = Emoji("📊 ", "");
static SAVE: Emoji<'_, '_> = Emoji("💾 ", "");
static DNA: Emoji<'_, '_> = Emoji("🧬 ", "");
static ALIEN: Emoji<'_, '_> = Emoji("👽 ", "");
static DOLPHIN: Emoji<'_, '_> = Emoji("🐬 ", "");
static BIRD: Emoji<'_, '_> = Emoji("🐦 ", "");
static MONKEY: Emoji<'_, '_> = Emoji("🐒 ", "");
static LINK: Emoji<'_, '_> = Emoji("🔗 ", "");

// ============================================================================
// CORE TYPES
// ============================================================================

#[derive(Clone, Copy)]
struct HV16([u8; 256]);

impl HV16 {
    fn zero() -> Self {
        Self([0u8; 256])
    }

    fn from_seed(seed: &[u8]) -> Self {
        use blake3::Hasher;
        let hash = blake3::hash(seed);
        let mut result = [0u8; 256];
        let mut hasher = Hasher::new();
        hasher.update(hash.as_bytes());
        let mut output = hasher.finalize_xof();
        output.fill(&mut result);
        HV16(result)
    }

    #[allow(dead_code)]
    fn bind(&self, other: &Self) -> Self {
        let mut result = [0u8; 256];
        for i in 0..256 {
            result[i] = self.0[i] ^ other.0[i];
        }
        Self(result)
    }

    fn similarity(&self, other: &Self) -> f32 {
        let matching: u32 = self
            .0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| (!(a ^ b)).count_ones())
            .sum();
        matching as f32 / 2048.0
    }
}

fn bundle(vectors: &[HV16]) -> HV16 {
    if vectors.is_empty() {
        return HV16::zero();
    }

    let mut counts = [0i32; 2048];
    for vec in vectors {
        for byte_idx in 0..256 {
            for bit_idx in 0..8 {
                let bit = (vec.0[byte_idx] >> bit_idx) & 1;
                let pos = byte_idx * 8 + bit_idx;
                counts[pos] += if bit == 1 { 1 } else { -1 };
            }
        }
    }

    let mut result = [0u8; 256];
    for byte_idx in 0..256 {
        for bit_idx in 0..8 {
            let pos = byte_idx * 8 + bit_idx;
            if counts[pos] > 0 {
                result[byte_idx] |= 1 << bit_idx;
            }
        }
    }

    HV16(result)
}

// ============================================================================
// DISCOVERY CONFIG
// ============================================================================

struct DiscoveryConfig {
    similarity_threshold: f32,
    min_instances: usize,
    max_units: usize,
    salience_threshold: f32,
    min_segment_ms: f32,
    max_segment_ms: f32,
}

impl DiscoveryConfig {
    fn for_species(species: Species) -> Self {
        match species {
            Species::Cetacean => Self {
                similarity_threshold: 0.60,
                min_instances: 3,
                max_units: 200,
                salience_threshold: 0.4,
                min_segment_ms: 5.0,
                max_segment_ms: 10000.0,
            },
            Species::Avian => Self {
                similarity_threshold: 0.62,
                min_instances: 3,
                max_units: 150,
                salience_threshold: 0.5,
                min_segment_ms: 20.0,
                max_segment_ms: 2000.0,
            },
            Species::Primate => Self {
                similarity_threshold: 0.65,
                min_instances: 3,
                max_units: 80,
                salience_threshold: 0.5,
                min_segment_ms: 50.0,
                max_segment_ms: 3000.0,
            },
            Species::Human => Self {
                similarity_threshold: 0.65,
                min_instances: 5,
                max_units: 50,
                salience_threshold: 0.5,
                min_segment_ms: 20.0,
                max_segment_ms: 300.0,
            },
            Species::Unknown => Self {
                similarity_threshold: 0.55,
                min_instances: 2,
                max_units: 500,
                salience_threshold: 0.3,
                min_segment_ms: 1.0,
                max_segment_ms: 60000.0,
            },
        }
    }
}

// ============================================================================
// DISCOVERED UNIT
// ============================================================================

#[derive(Clone)]
struct DiscoveredUnit {
    id: String,
    centroid: HV16,
    instances: Vec<HV16>,
    instance_count: usize,
    mean_duration_ms: f32,
    min_duration_ms: f32,
    max_duration_ms: f32,
    following: HashMap<String, usize>,
}

impl DiscoveredUnit {
    fn new(id: &str, first: HV16, duration: f32) -> Self {
        Self {
            id: id.to_string(),
            centroid: first,
            instances: vec![first],
            instance_count: 1,
            mean_duration_ms: duration,
            min_duration_ms: duration,
            max_duration_ms: duration,
            following: HashMap::new(),
        }
    }

    fn add_instance(&mut self, hv: HV16, duration: f32) {
        self.instances.push(hv);
        self.instance_count += 1;
        self.centroid = bundle(&self.instances);

        let n = self.instance_count as f32;
        self.mean_duration_ms = ((n - 1.0) * self.mean_duration_ms + duration) / n;
        self.min_duration_ms = self.min_duration_ms.min(duration);
        self.max_duration_ms = self.max_duration_ms.max(duration);
    }

    fn similarity(&self, hv: &HV16) -> f32 {
        self.centroid.similarity(hv)
    }
}

// ============================================================================
// DISCOVERY ENGINE
// ============================================================================

struct DiscoveryEngine {
    units: HashMap<String, DiscoveredUnit>,
    next_id: usize,
    config: DiscoveryConfig,
    previous: Option<String>,
    segments: Vec<(f32, f32, String)>, // (start, end, unit_id)
}

impl DiscoveryEngine {
    fn new(config: DiscoveryConfig) -> Self {
        Self {
            units: HashMap::new(),
            next_id: 1,
            config,
            previous: None,
            segments: Vec::new(),
        }
    }

    fn process_segment(&mut self, hv: HV16, start: f32, end: f32) -> String {
        let duration = (end - start) * 1000.0;

        if duration < self.config.min_segment_ms || duration > self.config.max_segment_ms {
            return "NOISE".to_string();
        }

        // Find best match
        let mut best: Option<(String, f32)> = None;
        for (id, unit) in &self.units {
            let sim = unit.similarity(&hv);
            if sim > self.config.similarity_threshold
                && best.as_ref().map(|(_, s)| sim > *s).unwrap_or(true)
            {
                best = Some((id.clone(), sim));
            }
        }

        let unit_id = if let Some((id, _)) = best {
            if let Some(unit) = self.units.get_mut(&id) {
                unit.add_instance(hv, duration);
            }
            id
        } else if self.units.len() < self.config.max_units {
            let id = format!("UNIT_{:03}", self.next_id);
            self.next_id += 1;
            self.units
                .insert(id.clone(), DiscoveredUnit::new(&id, hv, duration));
            id
        } else {
            // At capacity - assign to closest
            self.units
                .iter()
                .map(|(id, u)| (id.clone(), u.similarity(&hv)))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(id, _)| id)
                .unwrap_or("OVERFLOW".to_string())
        };

        // Track transitions
        if let Some(ref prev) = self.previous {
            if prev != &unit_id {
                if let Some(pu) = self.units.get_mut(prev) {
                    *pu.following.entry(unit_id.clone()).or_insert(0) += 1;
                }
            }
        }
        self.previous = Some(unit_id.clone());

        self.segments.push((start, end, unit_id.clone()));
        unit_id
    }

    fn stable_units(&self) -> Vec<&DiscoveredUnit> {
        let mut units: Vec<_> = self
            .units
            .values()
            .filter(|u| u.instance_count >= self.config.min_instances)
            .collect();
        units.sort_by_key(|u| std::cmp::Reverse(u.instance_count));
        units
    }
}

// ============================================================================
// AUDIO PROCESSING
// ============================================================================

fn load_audio(path: &PathBuf) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Try WAV first
    if path.extension().map(|e| e == "wav").unwrap_or(false) {
        let reader = hound::WavReader::open(path)?;
        let spec = reader.spec();
        let samples: Vec<f32> = if spec.bits_per_sample == 16 {
            reader
                .into_samples::<i16>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / 32768.0)
                .collect()
        } else {
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / 2147483648.0)
                .collect()
        };
        return Ok(samples);
    }

    // Use ffmpeg for other formats
    let output = Command::new("ffmpeg")
        .args([
            "-i",
            path.to_str().unwrap(),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-",
        ])
        .output()?;

    if !output.status.success() {
        return Err("ffmpeg failed".into());
    }

    let samples: Vec<f32> = output
        .stdout
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    Ok(samples)
}

fn project_audio(audio: &[f32]) -> (Vec<HV16>, Vec<f32>, Vec<f32>) {
    let frame_samples = 160; // 10ms at 16kHz
    let num_frames = audio.len() / frame_samples;

    let mut frames = Vec::with_capacity(num_frames);
    let mut salience = Vec::with_capacity(num_frames);
    let mut timestamps = Vec::with_capacity(num_frames);

    let mut prev_energy = 0.0f32;

    for i in 0..num_frames {
        let start = i * frame_samples;
        let end = (start + frame_samples).min(audio.len());

        // Frame energy
        let energy: f32 =
            audio[start..end].iter().map(|s| s * s).sum::<f32>() / frame_samples as f32;

        // Spectral features (simplified)
        let zero_crossings: usize = audio[start..end]
            .windows(2)
            .filter(|w| w[0].signum() != w[1].signum())
            .count();

        // Generate HV from features
        let seed = format!("f:{}:e:{:.6}:zc:{}", i, energy, zero_crossings);
        let hv = HV16::from_seed(seed.as_bytes());

        // Salience
        let sal = ((energy - prev_energy).abs() * 100.0 + (zero_crossings as f32 * 0.1)).min(10.0);
        prev_energy = energy;

        frames.push(hv);
        salience.push(sal);
        timestamps.push(i as f32 * 0.010);
    }

    (frames, salience, timestamps)
}

fn detect_boundaries(
    salience: &[f32],
    timestamps: &[f32],
    threshold: f32,
    min_gap: f32,
) -> Vec<(f32, usize)> {
    if salience.is_empty() {
        return Vec::new();
    }

    let mean: f32 = salience.iter().sum::<f32>() / salience.len() as f32;
    let std: f32 =
        (salience.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / salience.len() as f32).sqrt();
    let thresh = mean + std * threshold;

    let mut boundaries = vec![(0.0, 0)];
    let mut last_time = -min_gap;

    for i in 1..salience.len() - 1 {
        let is_peak = salience[i] > salience[i - 1] && salience[i] > salience[i + 1];
        if is_peak && salience[i] > thresh && timestamps[i] - last_time > min_gap {
            boundaries.push((timestamps[i], i));
            last_time = timestamps[i];
        }
    }

    let last_t = *timestamps.last().unwrap_or(&0.0);
    boundaries.push((last_t, salience.len() - 1));

    boundaries
}

// ============================================================================
// OUTPUT GENERATION
// ============================================================================

fn generate_report(
    engine: &DiscoveryEngine,
    duration_sec: f32,
    output_dir: &PathBuf,
    species: Species,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(output_dir)?;

    let units = engine.stable_units();

    // JSON report
    let json_path = output_dir.join("discovery_report.json");
    let mut json = File::create(&json_path)?;

    writeln!(json, "{{")?;
    writeln!(json, "  \"species\": \"{:?}\",", species)?;
    writeln!(json, "  \"duration_sec\": {:.2},", duration_sec)?;
    writeln!(json, "  \"num_segments\": {},", engine.segments.len())?;
    writeln!(json, "  \"num_units\": {},", units.len())?;
    writeln!(json, "  \"units\": [")?;

    for (i, unit) in units.iter().enumerate() {
        let comma = if i < units.len() - 1 { "," } else { "" };
        writeln!(json, "    {{")?;
        writeln!(json, "      \"id\": \"{}\",", unit.id)?;
        writeln!(json, "      \"instances\": {},", unit.instance_count)?;
        writeln!(
            json,
            "      \"mean_duration_ms\": {:.1},",
            unit.mean_duration_ms
        )?;
        writeln!(
            json,
            "      \"duration_range\": [{:.1}, {:.1}],",
            unit.min_duration_ms, unit.max_duration_ms
        )?;

        // Most common following units
        let mut following: Vec<_> = unit.following.iter().collect();
        following.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
        let top_following: Vec<_> = following
            .iter()
            .take(3)
            .map(|(id, c)| format!("\"{}:{}\"", id, c))
            .collect();
        writeln!(
            json,
            "      \"common_following\": [{}]",
            top_following.join(", ")
        )?;

        writeln!(json, "    }}{}", comma)?;
    }

    writeln!(json, "  ],")?;

    // Segments
    writeln!(json, "  \"segments\": [")?;
    for (i, (start, end, label)) in engine.segments.iter().enumerate() {
        let comma = if i < engine.segments.len() - 1 {
            ","
        } else {
            ""
        };
        writeln!(
            json,
            "    {{ \"start\": {:.3}, \"end\": {:.3}, \"unit\": \"{}\" }}{}",
            start, end, label, comma
        )?;
    }
    writeln!(json, "  ]")?;
    writeln!(json, "}}")?;

    // TSV for Praat/Audacity
    let tsv_path = output_dir.join("segments.tsv");
    let mut tsv = File::create(&tsv_path)?;
    writeln!(tsv, "start\tend\tlabel")?;
    for (start, end, label) in &engine.segments {
        writeln!(tsv, "{:.6}\t{:.6}\t{}", start, end, label)?;
    }

    // Transition matrix
    let matrix_path = output_dir.join("transitions.tsv");
    let mut matrix = File::create(&matrix_path)?;

    let unit_ids: Vec<_> = units.iter().map(|u| &u.id).collect();
    write!(matrix, "from\\to")?;
    for id in &unit_ids {
        write!(matrix, "\t{}", id)?;
    }
    writeln!(matrix)?;

    for from_unit in &units {
        write!(matrix, "{}", from_unit.id)?;
        for to_id in &unit_ids {
            let count = from_unit.following.get(*to_id).cloned().unwrap_or(0);
            write!(matrix, "\t{}", count)?;
        }
        writeln!(matrix)?;
    }

    Ok(())
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Header
    if !cli.quiet {
        println!("\n{}", style("═".repeat(70)).magenta());
        println!(
            "{}",
            style("               SYMTHAEA UNIVERSAL DISCOVERY")
                .magenta()
                .bold()
        );
        println!("{}", style("═".repeat(70)).magenta());
        println!();

        let species_emoji = match cli.species {
            Species::Cetacean => DOLPHIN,
            Species::Avian => BIRD,
            Species::Primate => MONKEY,
            Species::Human => EAR,
            Species::Unknown => ALIEN,
        };

        println!("  {}Species profile: {:?}", species_emoji, cli.species);
        println!("  {}Input: {:?}", WAVE, cli.input);
        println!();
    }

    // Load config
    let mut config = DiscoveryConfig::for_species(cli.species);

    // Apply overrides
    if let Some(t) = cli.threshold {
        config.similarity_threshold = t;
    }
    if let Some(m) = cli.max_units {
        config.max_units = m;
    }
    if let Some(d) = cli.min_duration {
        config.min_segment_ms = d;
    }
    if let Some(d) = cli.max_duration {
        config.max_segment_ms = d;
    }

    if !cli.quiet {
        println!("  {}Configuration:", DNA);
        println!(
            "    Similarity threshold: {:.2}",
            config.similarity_threshold
        );
        println!("    Max units: {}", config.max_units);
        println!(
            "    Duration range: {:.1}ms - {:.1}ms",
            config.min_segment_ms, config.max_segment_ms
        );
        println!();
    }

    // Load audio
    if !cli.quiet {
        println!("{}{}Loading audio...", WAVE, style("Step 1/3: ").bold());
    }

    let audio = load_audio(&cli.input)?;
    let duration_sec = audio.len() as f32 / 16000.0;

    if !cli.quiet {
        println!(
            "  {}Loaded {:.2} seconds ({} samples)",
            CHECK,
            duration_sec,
            audio.len()
        );
    }

    // Project
    if !cli.quiet {
        println!(
            "\n{}{}Projecting through LTC dynamics...",
            BRAIN,
            style("Step 2/3: ").bold()
        );
    }

    let start_time = Instant::now();
    let (frames, salience, timestamps) = project_audio(&audio);

    if !cli.quiet {
        println!(
            "  {}Generated {} frames in {:?}",
            CHECK,
            frames.len(),
            start_time.elapsed()
        );
    }

    // Detect boundaries
    let min_gap = config.min_segment_ms / 1000.0;
    let boundaries = detect_boundaries(&salience, &timestamps, config.salience_threshold, min_gap);

    if !cli.quiet {
        println!(
            "  {}Detected {} acoustic boundaries",
            CHECK,
            boundaries.len()
        );
    }

    // Discover units
    if !cli.quiet {
        println!(
            "\n{}{}Discovering acoustic units...",
            ALIEN,
            style("Step 3/3: ").bold()
        );
        println!();
    }

    let mut engine = DiscoveryEngine::new(config);

    let bar = if !cli.quiet {
        let b = ProgressBar::new(boundaries.len().saturating_sub(1) as u64);
        b.set_style(
            ProgressStyle::default_bar()
                .template("{prefix:.bold} {spinner:.green} [{bar:40.magenta/blue}] {pos}/{len}")
                .unwrap(),
        );
        b.set_prefix("Clustering");
        Some(b)
    } else {
        None
    };

    for window in boundaries.windows(2) {
        let (start_time, start_frame) = window[0];
        let (end_time, end_frame) = window[1];

        if end_frame <= start_frame || end_frame > frames.len() {
            continue;
        }

        let segment_frames = &frames[start_frame..end_frame];
        let hv = bundle(segment_frames);

        engine.process_segment(hv, start_time, end_time);

        if let Some(ref b) = bar {
            b.inc(1);
        }
    }

    if let Some(b) = bar {
        b.finish_with_message("Complete");
    }

    // Generate report
    if !cli.quiet {
        println!();
        println!("{}{}Generating report...", SAVE, style("").bold());
    }

    generate_report(&engine, duration_sec, &cli.output, cli.species)?;

    // Summary
    if !cli.quiet {
        let units = engine.stable_units();

        println!("\n{}", style("═".repeat(70)).green());
        println!(
            "{}",
            style("                    DISCOVERY COMPLETE")
                .green()
                .bold()
        );
        println!("{}", style("═".repeat(70)).green());
        println!();
        println!("  {}Duration analyzed: {:.2}s", CHART, duration_sec);
        println!("  {}Segments found:    {}", CHART, engine.segments.len());
        println!("  {}Units discovered:  {}", BRAIN, units.len());
        println!();

        // Show top units
        println!("  {}Discovered call types:", DNA);
        println!("  {}", style("─".repeat(50)).dim());

        for (i, unit) in units.iter().take(15).enumerate() {
            let bar_len = (unit.instance_count as f32 / 10.0).min(25.0) as usize;
            let bar = "█".repeat(bar_len);

            println!(
                "    {:2}. {} {:>4} instances  {:.0}-{:.0}ms  {}",
                i + 1,
                style(&unit.id).cyan(),
                unit.instance_count,
                unit.min_duration_ms,
                unit.max_duration_ms,
                style(bar).magenta()
            );
        }

        if units.len() > 15 {
            println!("    ... and {} more units", units.len() - 15);
        }

        // Show frequent transitions
        println!();
        println!("  {}Frequent sequences:", LINK);

        let mut all_transitions: Vec<(String, String, usize)> = Vec::new();
        for unit in &units {
            for (to, count) in &unit.following {
                all_transitions.push((unit.id.clone(), to.clone(), *count));
            }
        }
        all_transitions.sort_by_key(|(_, _, c)| std::cmp::Reverse(*c));

        for (from, to, count) in all_transitions.iter().take(10) {
            println!(
                "    {} → {}  (×{})",
                style(from).cyan(),
                style(to).yellow(),
                count
            );
        }

        println!();
        println!("  {}Output: {:?}", SAVE, cli.output);
        println!();

        // Interpretation hint
        match cli.species {
            Species::Cetacean => {
                println!("  {}Interpretation hint:", DNA);
                println!("    Short units (<50ms) may be echolocation clicks");
                println!("    Long units (>500ms) may be social calls or songs");
                println!("    Repeated sequences may indicate codas or phrases");
            }
            Species::Avian => {
                println!("  {}Interpretation hint:", DNA);
                println!("    High-frequency units may be alarm calls");
                println!("    Melodic sequences may be territorial songs");
                println!("    Dawn chorus shows highest diversity");
            }
            Species::Primate => {
                println!("  {}Interpretation hint:", DNA);
                println!("    Short barks may be alarm or contact calls");
                println!("    Longer calls may be territorial or mating");
                println!("    Context (visual log) is crucial for meaning");
            }
            Species::Unknown => {
                println!("  {}Interpretation hint:", DNA);
                println!("    Unknown signal structure - no prior assumptions");
                println!("    Look for repeated patterns and transitions");
                println!("    Correlate with any available context data");
            }
            Species::Human => {
                println!(
                    "  {}Note: For human speech, use 'symthaea-train' instead",
                    WARN
                );
            }
        }
        println!();
    }

    Ok(())
}
