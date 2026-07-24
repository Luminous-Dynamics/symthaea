// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Diagnostic: voiced-F1 distribution over whole WAV files.
//!
//! Distillation v2/v2.5 found extracted vowel F1 medians collapsed to
//! ~280-540 Hz (open vowels like AA should sit near 730). Two candidate
//! culprits: (a) the LPC extractor never produces high F1 at all, or
//! (b) the extractor is fine and duration-proportional alignment places
//! phoneme spans on the wrong frames, so per-phoneme medians regress to
//! the all-speech F1 population median. This probe separates them: scan
//! entire utterances (no alignment involved) and print voiced-frame F1
//! quantiles + the fraction of frames with F1 > 600 Hz. Sentences with
//! stressed open vowels MUST show a high-F1 tail if the extractor works.
//!
//! ```bash
//! cargo run -p symthaea-vocal-tract --example f1_probe --features hound -- corpus_dir/*.wav
//! ```

use symthaea_vocal_tract::formant_extraction::{ExtractionConfig, extract_formant_frames};
use symthaea_vocal_tract::types::SourceType;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: f1_probe <wav> [<wav>...]");
        std::process::exit(2);
    }
    let config = ExtractionConfig::default();
    let mut all_f1: Vec<f32> = Vec::new();

    for path in &paths {
        let mut reader = hound::WavReader::open(path)?;
        let spec = reader.spec();
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => reader
                .samples::<i16>()
                .map(|s| s.map(|v| v as f32 / 32768.0))
                .collect::<Result<_, _>>()?,
            hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        };
        // Fold to mono if needed.
        let mono: Vec<f32> = if spec.channels > 1 {
            samples
                .chunks(spec.channels as usize)
                .map(|c| c.iter().sum::<f32>() / c.len() as f32)
                .collect()
        } else {
            samples
        };

        let frames = extract_formant_frames(&mono, spec.sample_rate, &config);

        // Island-span diagnostic (v3 distillation follow-up): does the
        // "voiced == Vowel" run structure actually isolate one island per
        // vowel nucleus, or do voiced consonants (nasals/liquids/voiced
        // stops) merge into the same run? Print island count + length stats.
        let mut islands: Vec<(usize, usize)> = Vec::new();
        let mut start: Option<usize> = None;
        for (i, f) in frames.iter().enumerate() {
            let voiced = f.source_type == SourceType::Vowel && f.voicing > 0.5;
            match (voiced, start) {
                (true, None) => start = Some(i),
                (false, Some(s)) => {
                    if i - s >= 3 {
                        islands.push((s, i));
                    }
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(s) = start
            && frames.len() - s >= 3
        {
            islands.push((s, frames.len()));
        }
        let lens: Vec<usize> = islands.iter().map(|(s, e)| e - s).collect();
        let max_len = lens.iter().copied().max().unwrap_or(0);
        println!(
            "{path}: {} islands, lengths={:?} (max {}fr = {:.0}ms)",
            islands.len(),
            lens,
            max_len,
            max_len as f32 * 1000.0 / config.frame_rate
        );

        let mut f1: Vec<f32> = frames
            .iter()
            .filter(|f| f.source_type == SourceType::Vowel && f.energy > 0.1)
            .map(|f| f.f1)
            .collect();
        if f1.is_empty() {
            println!("{path}: no voiced frames");
            continue;
        }
        f1.sort_by(f32::total_cmp);
        let q = |p: f32| f1[((f1.len() - 1) as f32 * p) as usize];
        let hi = f1.iter().filter(|&&v| v > 600.0).count() as f32 / f1.len() as f32;
        println!(
            "{path}: n={:4}  p10={:4.0}  p50={:4.0}  p90={:4.0}  max={:4.0}  frac(F1>600)={:.1}%",
            f1.len(),
            q(0.10),
            q(0.50),
            q(0.90),
            q(1.0),
            hi * 100.0
        );
        all_f1.extend_from_slice(&f1);
    }

    if all_f1.len() > 1 {
        all_f1.sort_by(f32::total_cmp);
        let q = |p: f32| all_f1[((all_f1.len() - 1) as f32 * p) as usize];
        let hi = all_f1.iter().filter(|&&v| v > 600.0).count() as f32 / all_f1.len() as f32;
        println!(
            "\nALL: n={}  p10={:.0}  p50={:.0}  p90={:.0}  max={:.0}  frac(F1>600)={:.1}%",
            all_f1.len(),
            q(0.10),
            q(0.50),
            q(0.90),
            q(1.0),
            hi * 100.0
        );
    }
    Ok(())
}
