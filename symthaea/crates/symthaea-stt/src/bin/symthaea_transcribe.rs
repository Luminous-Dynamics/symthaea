// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea Transcription CLI
//!
//! Speech-to-text inference using trained prototypes.
//!
//! # Usage
//!
//! ```bash
//! # Transcribe a single file
//! symthaea-transcribe audio.wav
//!
//! # Transcribe with custom model
//! symthaea-transcribe --model prototypes.json audio.wav
//!
//! # Stream from microphone (future)
//! symthaea-transcribe --stream
//! ```

use clap::{Parser, Subcommand};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};
use std::time::Instant;

use symthaea_stt::{
    AudioFrontend, AudioProjector, BootstrapConfig, CmuDictionary, HV16, PhonemeDecoder,
    PhonemeInventory, PhonemeResonator, TrainedPrototypes,
};

// ============================================================================
// CLI DEFINITION
// ============================================================================

#[derive(Parser)]
#[command(
    name = "symthaea-transcribe",
    about = "Symthaea Speech-to-Text Transcription",
    version,
    author
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Transcribe an audio file
    Transcribe {
        /// Audio file to transcribe (WAV format)
        #[arg(required = true)]
        audio_file: PathBuf,

        /// Path to trained prototypes
        #[arg(short, long, default_value = "models/prototypes.json")]
        model: PathBuf,

        /// Path to CMU dictionary
        #[arg(short, long, default_value = "data/cmudict.dict")]
        dictionary: PathBuf,

        /// Output format (text, json, phonemes)
        #[arg(short, long, default_value = "text")]
        format: String,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Download required resources (CMU dictionary)
    Download {
        /// Output directory
        #[arg(short, long, default_value = "data")]
        output: PathBuf,

        /// Force re-download even if exists
        #[arg(short, long)]
        force: bool,
    },

    /// Initialize a new model from prototypes
    Init {
        /// Output model path
        #[arg(short, long, default_value = "models/prototypes.json")]
        output: PathBuf,
    },

    /// Show information about a model
    Info {
        /// Path to model file
        #[arg(required = true)]
        model: PathBuf,
    },

    /// Benchmark transcription speed
    Benchmark {
        /// Audio file to benchmark
        #[arg(required = true)]
        audio_file: PathBuf,

        /// Number of iterations
        #[arg(short, long, default_value = "10")]
        iterations: usize,
    },
}

// ============================================================================
// TRANSCRIPTION ENGINE
// ============================================================================

struct TranscriptionEngine {
    projector: AudioProjector,
    decoder: PhonemeDecoder,
    dictionary: Option<CmuDictionary>,
    resonator: PhonemeResonator,
}

impl TranscriptionEngine {
    fn new() -> Self {
        Self {
            projector: AudioProjector::default_config(),
            decoder: PhonemeDecoder::new(),
            dictionary: None,
            resonator: PhonemeResonator::new(),
        }
    }

    fn load_model(&mut self, path: &Path) -> std::io::Result<()> {
        let prototypes = TrainedPrototypes::load(path)?;
        self.decoder.load_prototypes(&prototypes.as_pairs());

        // Also load into resonator
        for (label, hv) in prototypes.as_pairs() {
            self.resonator.store(&label, hv);
        }

        Ok(())
    }

    fn load_dictionary(&mut self, path: &Path) -> std::io::Result<()> {
        self.dictionary = Some(CmuDictionary::load(path)?);
        Ok(())
    }

    fn transcribe(&mut self, audio: &[f32]) -> TranscriptionResult {
        let start = Instant::now();

        // Project audio to HDC space
        self.projector.reset();
        let (hvs, _saliences) = self.projector.project_with_salience(audio);

        let projection_time = start.elapsed();

        // Decode phonemes
        let mut phonemes = Vec::new();
        let mut confidences = Vec::new();

        self.decoder.reset();
        for hv in &hvs {
            if let Some((phoneme, confidence)) = self.decoder.decode(hv) {
                // Collapse repeated phonemes
                if phonemes.last() != Some(&phoneme) {
                    phonemes.push(phoneme);
                    confidences.push(confidence);
                }
            }
        }

        let decode_time = start.elapsed() - projection_time;

        // Convert phonemes to words (if dictionary available)
        let words = self.phonemes_to_words(&phonemes);

        TranscriptionResult {
            text: words.join(" "),
            phonemes,
            confidences,
            num_frames: hvs.len(),
            projection_time_ms: projection_time.as_secs_f32() * 1000.0,
            decode_time_ms: decode_time.as_secs_f32() * 1000.0,
        }
    }

    fn phonemes_to_words(&self, phonemes: &[String]) -> Vec<String> {
        // Simple approach: just return phonemes as IPA for now
        // A full implementation would use the dictionary for reverse lookup
        if phonemes.is_empty() {
            return vec![];
        }

        // Filter out silence and noise tokens
        let filtered: Vec<String> = phonemes
            .iter()
            .filter(|p| *p != "sil" && *p != "SIL" && !p.starts_with('<'))
            .cloned()
            .collect();

        if filtered.is_empty() {
            vec!["[silence]".to_string()]
        } else {
            // Return as phoneme string for now
            vec![format!("/{}/", filtered.join(" "))]
        }
    }
}

struct TranscriptionResult {
    text: String,
    phonemes: Vec<String>,
    #[allow(dead_code)]
    confidences: Vec<f32>,
    num_frames: usize,
    projection_time_ms: f32,
    decode_time_ms: f32,
}

// ============================================================================
// COMMANDS
// ============================================================================

fn cmd_transcribe(
    audio_file: &Path,
    model: &Path,
    dictionary: &Path,
    format: &str,
    verbose: bool,
) -> std::io::Result<()> {
    println!("{}", style("Symthaea Transcription").bold().cyan());
    println!();

    // Load audio
    if verbose {
        println!("Loading audio from {:?}...", audio_file);
    }

    let (audio, sample_rate) = AudioFrontend::load_wav(audio_file)?;
    let duration_sec = audio.len() as f32 / sample_rate as f32;

    if verbose {
        println!(
            "  Duration: {:.2}s, Sample rate: {}Hz",
            duration_sec, sample_rate
        );
    }

    // Initialize engine
    let mut engine = TranscriptionEngine::new();

    // Load model if exists
    if model.exists() {
        if verbose {
            println!("Loading model from {:?}...", model);
        }
        engine.load_model(model)?;
    } else {
        println!(
            "{}",
            style("Warning: No model found, using random prototypes").yellow()
        );
        // Initialize with phoneme inventory defaults
        let inventory = PhonemeInventory::new();
        for phoneme in inventory.all() {
            let hv = HV16::random(&phoneme.ipa);
            engine.resonator.store(&phoneme.ipa, hv);
        }
    }

    // Load dictionary if exists
    if dictionary.exists() {
        if verbose {
            println!("Loading dictionary from {:?}...", dictionary);
        }
        engine.load_dictionary(dictionary)?;
    }

    // Transcribe
    if verbose {
        println!("Transcribing...");
    }

    let result = engine.transcribe(&audio);

    // Output based on format
    match format {
        "json" => {
            println!("{{");
            println!("  \"text\": \"{}\",", result.text);
            println!("  \"phonemes\": {:?},", result.phonemes);
            println!("  \"num_frames\": {},", result.num_frames);
            println!("  \"duration_sec\": {:.2},", duration_sec);
            println!(
                "  \"realtime_factor\": {:.2},",
                duration_sec * 1000.0 / (result.projection_time_ms + result.decode_time_ms)
            );
            println!("}}");
        }
        "phonemes" => {
            println!("{}", result.phonemes.join(" "));
        }
        _ => {
            // Default text output
            println!("{}", style("Transcription:").bold());
            println!("  {}", result.text);

            if verbose {
                println!();
                println!("{}", style("Details:").bold());
                println!("  Phonemes: {}", result.phonemes.join(" "));
                println!("  Frames: {}", result.num_frames);
                println!("  Projection: {:.1}ms", result.projection_time_ms);
                println!("  Decoding: {:.1}ms", result.decode_time_ms);
                println!(
                    "  Realtime factor: {:.1}x",
                    duration_sec * 1000.0 / (result.projection_time_ms + result.decode_time_ms)
                );
            }
        }
    }

    Ok(())
}

fn cmd_download(output: &Path, force: bool) -> std::io::Result<()> {
    println!("{}", style("Downloading CMU Dictionary").bold().cyan());
    println!();

    std::fs::create_dir_all(output)?;

    let dict_path = output.join("cmudict.dict");

    if dict_path.exists() && !force {
        println!(
            "{}",
            style("CMU Dictionary already exists. Use --force to re-download.").yellow()
        );
        return Ok(());
    }

    // CMU Dict is available from multiple sources
    // We'll embed a small subset for testing, with instructions to download full version
    println!("Note: Full CMU Dictionary download requires external fetch.");
    println!("Creating minimal embedded dictionary for testing...");
    println!();

    // Create a minimal dictionary with common words
    let _mini_dict = include_str!("../../data/mini_cmudict.txt");

    // If the embedded dict doesn't exist, create it
    let mini_dict_content = r#";;; Mini CMU Dictionary (subset for testing)
;;; Full dictionary: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
THE  DH AH0
A  AH0
TO  T UW1
OF  AH1 V
AND  AE1 N D
IN  IH1 N
IS  IH1 Z
IT  IH1 T
YOU  Y UW1
THAT  DH AE1 T
HE  HH IY1
WAS  W AA1 Z
FOR  F AO1 R
ON  AA1 N
ARE  AA1 R
WITH  W IH1 DH
AS  AE1 Z
I  AY1
HIS  HH IH1 Z
THEY  DH EY1
BE  B IY1
AT  AE1 T
ONE  W AH1 N
HAVE  HH AE1 V
THIS  DH IH1 S
FROM  F R AH1 M
OR  AO1 R
HAD  HH AE1 D
BY  B AY1
HOT  HH AA1 T
WORD  W ER1 D
BUT  B AH1 T
WHAT  W AH1 T
SOME  S AH1 M
WE  W IY1
CAN  K AE1 N
OUT  AW1 T
OTHER  AH1 DH ER0
WERE  W ER1
ALL  AO1 L
THERE  DH EH1 R
WHEN  W EH1 N
UP  AH1 P
USE  Y UW1 Z
YOUR  Y AO1 R
HOW  HH AW1
SAID  S EH1 D
AN  AE1 N
EACH  IY1 CH
SHE  SH IY1
WHICH  W IH1 CH
DO  D UW1
THEIR  DH EH1 R
TIME  T AY1 M
IF  IH1 F
WILL  W IH1 L
WAY  W EY1
ABOUT  AH0 B AW1 T
MANY  M EH1 N IY0
THEN  DH EH1 N
THEM  DH EH1 M
WRITE  R AY1 T
WOULD  W UH1 D
LIKE  L AY1 K
SO  S OW1
THESE  DH IY1 Z
HER  HH ER1
LONG  L AO1 NG
MAKE  M EY1 K
THING  TH IH1 NG
SEE  S IY1
HIM  HH IH1 M
TWO  T UW1
HAS  HH AE1 Z
LOOK  L UH1 K
MORE  M AO1 R
DAY  D EY1
COULD  K UH1 D
GO  G OW1
COME  K AH1 M
DID  D IH1 D
NUMBER  N AH1 M B ER0
SOUND  S AW1 N D
NO  N OW1
MOST  M OW1 S T
PEOPLE  P IY1 P AH0 L
MY  M AY1
OVER  OW1 V ER0
KNOW  N OW1
WATER  W AO1 T ER0
THAN  DH AE1 N
CALL  K AO1 L
FIRST  F ER1 S T
WHO  HH UW1
MAY  M EY1
DOWN  D AW1 N
SIDE  S AY1 D
BEEN  B IH1 N
NOW  N AW1
FIND  F AY1 N D
HEAD  HH EH1 D
STAND  S T AE1 N D
OWN  OW1 N
PAGE  P EY1 JH
SHOULD  SH UH1 D
COUNTRY  K AH1 N T R IY0
FOUND  F AW1 N D
ANSWER  AE1 N S ER0
SCHOOL  S K UW1 L
GROW  G R OW1
STUDY  S T AH1 D IY0
STILL  S T IH1 L
LEARN  L ER1 N
PLANT  P L AE1 N T
COVER  K AH1 V ER0
FOOD  F UW1 D
SUN  S AH1 N
FOUR  F AO1 R
BETWEEN  B IH0 T W IY1 N
STATE  S T EY1 T
KEEP  K IY1 P
EYE  AY1
NEVER  N EH1 V ER0
LAST  L AE1 S T
LET  L EH1 T
THOUGHT  TH AO1 T
CITY  S IH1 T IY0
TREE  T R IY1
CROSS  K R AO1 S
FARM  F AA1 R M
HARD  HH AA1 R D
START  S T AA1 R T
MIGHT  M AY1 T
STORY  S T AO1 R IY0
SAW  S AO1
FAR  F AA1 R
SEA  S IY1
DRAW  D R AO1
LEFT  L EH1 F T
LATE  L EY1 T
RUN  R AH1 N
WHILE  W AY1 L
PRESS  P R EH1 S
CLOSE  K L OW1 S
NIGHT  N AY1 T
REAL  R IY1 L
LIFE  L AY1 F
FEW  F Y UW1
NORTH  N AO1 R TH
HELLO  HH AH0 L OW1
HELLO(2)  HH EH0 L OW1
WORLD  W ER1 L D
SYMTHAEA  S IH0 M TH EY1 AH0
"#;

    std::fs::write(&dict_path, mini_dict_content)?;

    println!(
        "{} Created mini dictionary at {:?}",
        style("✓").green(),
        dict_path
    );
    println!();
    println!("For full dictionary, download from:");
    println!("  https://github.com/cmusphinx/cmudict/raw/master/cmudict.dict");
    println!();
    println!("And save to: {:?}", dict_path);

    Ok(())
}

fn cmd_init(output: &Path) -> std::io::Result<()> {
    println!("{}", style("Initializing Prototype Model").bold().cyan());
    println!();

    // Create output directory
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Create prototypes from phoneme inventory
    let config = BootstrapConfig::default();
    let mut prototypes = TrainedPrototypes::new(config);

    let inventory = PhonemeInventory::new();
    for phoneme in inventory.all() {
        // Create a random prototype for each phoneme
        let hv = HV16::random(&format!("phoneme:{}", phoneme.ipa));
        prototypes.prototypes.insert(phoneme.ipa.clone(), hv);
        prototypes.counts.insert(phoneme.ipa.clone(), 1);
    }

    // Save
    prototypes.save(output)?;

    println!(
        "{} Created initial prototypes at {:?}",
        style("✓").green(),
        output
    );
    println!("  Phonemes: {}", prototypes.len());
    println!();
    println!("Note: These are random prototypes. Train on real data for accuracy:");
    println!(
        "  symthaea-train bootstrap --data <librispeech-path> --output {:?}",
        output
    );

    Ok(())
}

fn cmd_info(model: &Path) -> std::io::Result<()> {
    println!("{}", style("Model Information").bold().cyan());
    println!();

    let prototypes = TrainedPrototypes::load(model)?;

    println!("Path: {:?}", model);
    println!("Version: {}", prototypes.version);
    println!("Phonemes: {}", prototypes.len());
    println!();

    println!("{}", style("Phoneme counts:").bold());
    let mut counts: Vec<_> = prototypes.counts.iter().collect();
    counts.sort_by_key(|(_, count)| std::cmp::Reverse(*count));

    for (phoneme, count) in counts.iter().take(20) {
        println!("  {:8} {:6} instances", phoneme, count);
    }

    if counts.len() > 20 {
        println!("  ... and {} more", counts.len() - 20);
    }

    Ok(())
}

fn cmd_benchmark(audio_file: &Path, iterations: usize) -> std::io::Result<()> {
    println!("{}", style("Benchmarking Transcription").bold().cyan());
    println!();

    // Load audio
    let (audio, sample_rate) = AudioFrontend::load_wav(audio_file)?;
    let duration_sec = audio.len() as f32 / sample_rate as f32;

    println!("Audio: {:?}", audio_file);
    println!("Duration: {:.2}s", duration_sec);
    println!("Iterations: {}", iterations);
    println!();

    // Initialize engine with random prototypes
    let mut engine = TranscriptionEngine::new();
    let inventory = PhonemeInventory::new();
    for phoneme in inventory.all() {
        let hv = HV16::random(&phoneme.ipa);
        engine.resonator.store(&phoneme.ipa, hv);
    }

    // Warmup
    let _ = engine.transcribe(&audio);

    // Benchmark
    let pb = ProgressBar::new(iterations as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap(),
    );

    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = engine.transcribe(&audio);
        times.push(start.elapsed().as_secs_f32() * 1000.0);
        pb.inc(1);
    }
    pb.finish_and_clear();

    // Statistics
    times.sort_by(|a, b| a.total_cmp(b));
    let mean: f32 = times.iter().sum::<f32>() / times.len() as f32;
    let median = times[times.len() / 2];
    let min = times[0];
    let max = times[times.len() - 1];
    let p95 = times[(times.len() as f32 * 0.95) as usize];

    println!("{}", style("Results:").bold());
    println!("  Mean:   {:.2}ms", mean);
    println!("  Median: {:.2}ms", median);
    println!("  Min:    {:.2}ms", min);
    println!("  Max:    {:.2}ms", max);
    println!("  p95:    {:.2}ms", p95);
    println!();
    println!("  Realtime factor: {:.1}x", duration_sec * 1000.0 / mean);

    Ok(())
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Transcribe {
            audio_file,
            model,
            dictionary,
            format,
            verbose,
        } => cmd_transcribe(&audio_file, &model, &dictionary, &format, verbose),
        Commands::Download { output, force } => cmd_download(&output, force),
        Commands::Init { output } => cmd_init(&output),
        Commands::Info { model } => cmd_info(&model),
        Commands::Benchmark {
            audio_file,
            iterations,
        } => cmd_benchmark(&audio_file, iterations),
    };

    if let Err(e) = result {
        eprintln!("{} {}", style("Error:").red().bold(), e);
        std::process::exit(1);
    }
}
