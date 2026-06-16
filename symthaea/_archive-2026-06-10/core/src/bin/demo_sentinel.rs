// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Semantic Sentinel Demo - Zero-Shot Temporal Rhythm Recognition
//!
//! This demo proves Symthaea's unique capability: learning temporal dynamics
//! instantly, using 5W of power, zero training data, and pure CPU dynamics.
//!
//! ## The Core Thesis
//!
//! "Transformers spatialize time. Symthaea lives in time."
//!
//! This demo proves it by distinguishing behaviors that look identical in a
//! single frame but differ in time:
//! - Pattern A: 60 BPM simple rhythm
//! - Pattern B: 120 BPM simple rhythm
//! - Pattern C: Heartbeat (irregular LUB-dub pattern)
//! - Pattern D: Polyrhythm (3:2 cross-rhythm)
//!
//! A standard vision model sees "brightness change" in all cases.
//! The HierarchicalLTC locks onto the temporal dynamics of each pattern,
//! treating different rhythms as distinct attractors in state space.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --bin demo_sentinel --release
//! cargo run --bin demo_sentinel --release -- --complex  # Include complex patterns
//! ```

use anyhow::Result;
use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use symthaea::perception::video::{
    LtcRhythmConfig, LtcRhythmDetector, MockPattern, MockVideoSource, TemporalFeatureExtractor,
    VideoSource, VideoSourceConfig,
};

#[cfg(feature = "webcam")]
use symthaea::perception::video::WebcamSource;

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Pattern definition for the demo
#[derive(Clone)]
struct PatternDef {
    key: char,
    name: &'static str,
    description: &'static str,
    pattern: MockPattern,
}

/// Demo configuration
struct DemoConfig {
    /// Use real webcam (vs mock)
    use_webcam: bool,
    /// Learning duration in seconds
    learn_duration: f32,
    /// Frame rate
    fps: u32,
    /// Include complex patterns (C, D)
    complex_mode: bool,
}

impl Default for DemoConfig {
    fn default() -> Self {
        Self {
            use_webcam: false,
            learn_duration: 3.0,
            fps: 30,
            complex_mode: false,
        }
    }
}

/// Get available patterns based on mode
fn get_patterns(complex: bool) -> Vec<PatternDef> {
    let mut patterns = vec![
        PatternDef {
            key: 'A',
            name: "Simple 60 BPM",
            description: "1 beat per second",
            pattern: MockPattern::Blink { bpm: 60 },
        },
        PatternDef {
            key: 'B',
            name: "Simple 120 BPM",
            description: "2 beats per second",
            pattern: MockPattern::Blink { bpm: 120 },
        },
    ];

    if complex {
        patterns.extend(vec![
            PatternDef {
                key: 'C',
                name: "Heartbeat",
                description: "LUB-dub cardiac rhythm",
                pattern: MockPattern::Heartbeat { bpm: 72 },
            },
            PatternDef {
                key: 'D',
                name: "Polyrhythm 3:2",
                description: "90 + 60 BPM interference",
                pattern: MockPattern::Polyrhythm {
                    bpm_a: 90,
                    bpm_b: 60,
                },
            },
            PatternDef {
                key: 'E',
                name: "Triplet Feel",
                description: "Swing/shuffle rhythm",
                pattern: MockPattern::Triplet { bpm: 80 },
            },
            PatternDef {
                key: 'F',
                name: "Accelerando",
                description: "60→120 BPM speedup",
                pattern: MockPattern::Accelerando {
                    start_bpm: 60,
                    end_bpm: 120,
                    cycle_frames: 90,
                },
            },
        ]);
    }

    patterns
}

// =============================================================================
// DEMO STATE
// =============================================================================

struct SentinelDemo {
    config: DemoConfig,
    detector: LtcRhythmDetector,
    feature_extractor: TemporalFeatureExtractor,
    patterns: Vec<PatternDef>,
    running: Arc<AtomicBool>,
}

impl SentinelDemo {
    fn new(config: DemoConfig) -> Self {
        let ltc_config = LtcRhythmConfig {
            dt_ms: 1000.0 / config.fps as f32,
            ..Default::default()
        };

        let patterns = get_patterns(config.complex_mode);

        Self {
            config,
            detector: LtcRhythmDetector::new(ltc_config),
            feature_extractor: TemporalFeatureExtractor::new(),
            patterns,
            running: Arc::new(AtomicBool::new(true)),
        }
    }

    fn print_header(&self) {
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║       SEMANTIC SENTINEL - LTC Temporal Rhythm Detection        ║");
        println!("╠════════════════════════════════════════════════════════════════╣");
        println!("║  \"Transformers spatialize time. Symthaea LIVES in time.\"       ║");
        println!("╠════════════════════════════════════════════════════════════════╣");
        println!("║  Learn Patterns:                                               ║");
        for p in &self.patterns {
            println!(
                "║    [{:}] - {:16} ({:24})   ║",
                p.key, p.name, p.description
            );
        }
        println!("╠════════════════════════════════════════════════════════════════╣");
        println!("║  Commands:                                                     ║");
        println!("║    [D] - Start Detection Mode                                  ║");
        println!("║    [S] - Show LTC Network Summary                              ║");
        println!("║    [R] - Reset All Patterns                                    ║");
        println!("║    [Q] - Quit                                                  ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");
    }

    fn get_pattern_def(&self, key: char) -> Option<&PatternDef> {
        self.patterns.iter().find(|p| p.key == key)
    }

    fn create_video_source(&self, pattern: MockPattern) -> Box<dyn VideoSource> {
        let config = VideoSourceConfig {
            width: 64,
            height: 64,
            fps: self.config.fps,
            camera_index: 0,
        };

        #[cfg(feature = "webcam")]
        if self.config.use_webcam {
            match WebcamSource::new(config.clone()) {
                Ok(source) => return Box::new(source),
                Err(e) => {
                    eprintln!("Failed to open webcam: {}. Falling back to mock.", e);
                }
            }
        }

        let mut mock = MockVideoSource::new(config);
        mock.pattern = pattern;
        Box::new(mock)
    }

    fn run_learning(&mut self, key: char) -> Result<()> {
        let pattern_def = match self.get_pattern_def(key) {
            Some(p) => p.clone(),
            None => {
                println!("\n⚠️  Unknown pattern: {}", key);
                return Ok(());
            }
        };

        let learn_duration = self.config.learn_duration;
        let fps = self.config.fps;

        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!(
            "║  🧠 Learning Pattern {} - {}                       ",
            key, pattern_def.name
        );
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  {} ", pattern_def.description);
        println!("║  The HierarchicalLTC will absorb the temporal dynamics...   ║");
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        // Start learning mode
        self.detector.start_learning(key);

        // Create video source with the pattern
        let mut source = self.create_video_source(pattern_def.pattern);
        source.start()?;

        let start = Instant::now();
        let duration = Duration::from_secs_f32(learn_duration);

        while start.elapsed() < duration && self.running.load(Ordering::Relaxed) {
            if let Some(frame) = source.next_frame()? {
                let features = self.feature_extractor.extract(&frame);
                let result = self.detector.process(&features);

                let progress = start.elapsed().as_secs_f32() / learn_duration;
                let bar_len = 30;
                let filled = (progress * bar_len as f32) as usize;
                print!(
                    "\r   [{}{}] {:.0}%  Φ = {:.4}",
                    "█".repeat(filled),
                    "░".repeat(bar_len - filled),
                    progress * 100.0,
                    result.phi
                );
                io::stdout().flush()?;
            }
        }

        source.stop()?;

        // Finalize learning
        if let Some((p, frames, _period)) = self.detector.stop_learning() {
            println!("\n\n   ✓ Pattern {} learned!", p);
            println!("   • Trajectory: {} frames", frames);

            // Show frequency signature
            if let Some(freq_sig) = self.detector.get_pattern_frequency_signature(p) {
                println!("   • Frequency signature:");
                println!(
                    "     0.5Hz: {:.3}  1Hz: {:.3}  2Hz: {:.3}  4Hz: {:.3}",
                    freq_sig[0], freq_sig[1], freq_sig[2], freq_sig[3]
                );
            }

            if let Some(detected_bpm) = self.detector.get_pattern_bpm(p, fps) {
                println!("   • Estimated period: {:.1} BPM", detected_bpm);
            }
        }

        Ok(())
    }

    fn run_detection(&mut self) -> Result<()> {
        let learned_count = self.detector.learned_pattern_count();
        if learned_count < 2 {
            println!(
                "\n⚠️  Please learn at least 2 patterns first! (Currently: {})",
                learned_count
            );
            return Ok(());
        }

        let learned = self.detector.learned_patterns();
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║  🔍 Detection Mode - LTC Trajectory Matching                 ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!(
            "║  Learned patterns: {:?}                                       ",
            learned
        );
        println!("║  Comparing input trajectory shape to learned attractors...   ║");
        println!("║  Press Ctrl+C to stop                                        ║");
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        // Build list of patterns to cycle through during demo
        let mut demo_patterns: Vec<(MockPattern, String)> = Vec::new();
        for key in &learned {
            if let Some(p) = self.get_pattern_def(*key) {
                demo_patterns.push((p.pattern, format!("{} ({})", p.name, key)));
            }
        }
        // Add a "mix" pattern if we have A and B
        if learned.contains(&'A') && learned.contains(&'B') {
            demo_patterns.push((MockPattern::Blink { bpm: 90 }, "Mix (90 BPM)".to_string()));
        }

        let mut pattern_idx = 0;
        let mut pattern_start = Instant::now();

        let config = VideoSourceConfig {
            width: 64,
            height: 64,
            fps: self.config.fps,
            camera_index: 0,
        };

        let (pattern, name) = &demo_patterns[pattern_idx];
        println!("   [Starting with {}]\n", name);

        let mut mock = MockVideoSource::new(config.clone());
        mock.pattern = *pattern;
        let mut source: Box<dyn VideoSource> = Box::new(mock);
        source.start()?;

        self.running.store(true, Ordering::Relaxed);

        while self.running.load(Ordering::Relaxed) {
            // Switch demo pattern every 6 seconds
            if pattern_start.elapsed() > Duration::from_secs(6) {
                pattern_idx = (pattern_idx + 1) % demo_patterns.len();
                pattern_start = Instant::now();

                let (pattern, name) = &demo_patterns[pattern_idx];
                println!("\n\n   [Switching to {}]\n", name);

                source.stop()?;
                let mut mock = MockVideoSource::new(config.clone());
                mock.pattern = *pattern;
                source = Box::new(mock);
                source.start()?;
            }

            if let Some(frame) = source.next_frame()? {
                let features = self.feature_extractor.extract(&frame);
                let result = self.detector.process(&features);

                if result.is_valid {
                    self.display_detection(&result);
                }
            }
        }

        source.stop()?;
        Ok(())
    }

    fn display_detection(&self, result: &symthaea::perception::video::RhythmDetection) {
        // Build similarity display for all learned patterns
        let mut sim_display = String::new();
        let bar_len = 10;

        let mut sorted_sims: Vec<_> = result.similarities.iter().collect();
        sorted_sims.sort_by_key(|(k, _)| *k);

        for (key, sim) in sorted_sims {
            let is_best = result.pattern == key.to_string();
            let color = if is_best { "\x1b[32m" } else { "\x1b[90m" };
            let bar = (sim * bar_len as f32) as usize;
            sim_display.push_str(&format!(
                "{}: {}[{}{}]\x1b[0m {:.2}  ",
                key,
                color,
                "█".repeat(bar.min(bar_len)),
                "░".repeat(bar_len - bar.min(bar_len)),
                sim
            ));
        }

        let detected_color = match result.pattern.as_str() {
            "Uncertain" => "\x1b[33m",
            "Unknown" | "None" => "\x1b[90m",
            _ => "\x1b[32m",
        };

        print!(
            "\r   {} │ {}Detected: {}\x1b[0m ({:.0}%)  Φ = {:.4}   ",
            sim_display,
            detected_color,
            result.pattern,
            result.confidence * 100.0,
            result.phi
        );
        io::stdout().flush().ok();
    }

    fn show_network_summary(&self) {
        println!("\n{}", self.detector.network_summary());

        // Also show learned patterns summary
        let learned = self.detector.learned_patterns();
        if !learned.is_empty() {
            println!("\n╔══════════════════════════════════════════════════════════════╗");
            println!("║  Learned Patterns                                            ║");
            println!("╠══════════════════════════════════════════════════════════════╣");
            for key in learned {
                if let Some(p) = self.get_pattern_def(key) {
                    if let Some(freq) = self.detector.get_pattern_frequency_signature(key) {
                        println!(
                            "║  {} - {:16}                                     ║",
                            key, p.name
                        );
                        println!(
                            "║      Freq: 0.5Hz={:.2} 1Hz={:.2} 2Hz={:.2} 4Hz={:.2}        ║",
                            freq[0], freq[1], freq[2], freq[3]
                        );
                    }
                }
            }
            println!("╚══════════════════════════════════════════════════════════════╝");
        }
    }

    fn reset_patterns(&mut self) {
        self.detector.reset();
        self.feature_extractor.reset();
        println!("\n🔄 All patterns and LTC state reset.");
    }

    fn run(&mut self) -> Result<()> {
        self.print_header();

        // Set up Ctrl+C handler
        let running = self.running.clone();
        ctrlc::set_handler(move || {
            running.store(false, Ordering::Relaxed);
        })
        .ok();

        loop {
            print!("\nCommand> ");
            io::stdout().flush()?;

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;

            let cmd = input.trim().to_uppercase();
            let first_char = cmd.chars().next();

            match first_char {
                Some('D') if cmd == "D" => {
                    self.running.store(true, Ordering::Relaxed);
                    self.run_detection()?;
                    println!("\n\n   Detection stopped.");
                }
                Some(c @ 'A'..='F') => {
                    // Learn pattern A-F
                    self.run_learning(c)?;
                }
                Some('S') => {
                    self.show_network_summary();
                }
                Some('R') => {
                    self.reset_patterns();
                }
                Some('Q') => {
                    println!("\nGoodbye!");
                    break;
                }
                Some('?') | Some('H') => {
                    self.print_header();
                }
                None => continue,
                _ => {
                    println!(
                        "Unknown command. Use A-F to learn, D to detect, S for summary, R to reset, Q to quit."
                    );
                }
            }
        }

        Ok(())
    }
}

// =============================================================================
// MAIN
// =============================================================================

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let mut config = DemoConfig::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--webcam" => config.use_webcam = true,
            "--complex" | "-c" => config.complex_mode = true,
            "--fps" => {
                i += 1;
                if i < args.len() {
                    config.fps = args[i].parse().unwrap_or(30);
                }
            }
            "--duration" | "-d" => {
                i += 1;
                if i < args.len() {
                    config.learn_duration = args[i].parse().unwrap_or(3.0);
                }
            }
            "--help" | "-h" => {
                println!("Semantic Sentinel Demo - LTC Temporal Rhythm Recognition");
                println!();
                println!("USAGE:");
                println!("    demo_sentinel [OPTIONS]");
                println!();
                println!("OPTIONS:");
                println!("    --webcam          Use real webcam instead of mock video");
                println!(
                    "    --complex, -c     Enable complex patterns (heartbeat, polyrhythm, etc.)"
                );
                println!("    --fps <FPS>       Frame rate (default: 30)");
                println!("    --duration <SEC>  Learning duration in seconds (default: 3.0)");
                println!("    -h, --help        Show this help message");
                println!();
                println!("PATTERNS:");
                println!("    A - Simple 60 BPM     (1 beat/second)");
                println!("    B - Simple 120 BPM    (2 beats/second)");
                println!();
                println!("COMPLEX PATTERNS (with --complex):");
                println!("    C - Heartbeat         (LUB-dub cardiac rhythm)");
                println!("    D - Polyrhythm 3:2    (90 + 60 BPM interference)");
                println!("    E - Triplet Feel      (Swing/shuffle rhythm)");
                println!("    F - Accelerando       (60→120 BPM speedup)");
                println!();
                println!("The demo uses a HierarchicalLTC network with 31 nodes across");
                println!("5 time scales (τ = 500ms → 6ms) to encode temporal dynamics.");
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    let mode_str = if config.complex_mode {
        "Complex (6 patterns)"
    } else {
        "Simple (2 patterns)"
    };

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  🧠 Symthaea Semantic Sentinel                                 ║");
    println!("║     Zero-Shot Temporal Rhythm Recognition via LTC Dynamics     ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║  Mode: {:^54} ║", mode_str);
    println!(
        "║  Video: {:^53} ║",
        if config.use_webcam { "Webcam" } else { "Mock" }
    );
    println!(
        "║  Frame Rate: {:>2} fps                                           ║",
        config.fps
    );
    println!(
        "║  Learn Duration: {:.1}s                                         ║",
        config.learn_duration
    );
    println!("╚════════════════════════════════════════════════════════════════╝");

    let mut demo = SentinelDemo::new(config);
    demo.run()
}
