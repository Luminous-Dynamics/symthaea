// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Text-prompt steering demo: describe the music you want, get the best of N
//! composed candidates — ranked by real CLAP text↔audio similarity.
//!
//! Run (needs network for the one-time CLAP tower downloads and
//! ORT_DYLIB_PATH pointing at libonnxruntime.so):
//!     cargo run --release -p symthaea-muse \
//!         --features "theory clap-fad" --example prompt_steering \
//!         -- "a gentle nostalgic waltz" 6
//!
//! Output: audio_output/prompt_steering/best.wav + the full ranking on
//! stdout (the scores tell you honestly how close the match is — this is
//! generate-and-rank over the symbolic composer's space, not free-form
//! audio generation).

use symthaea_muse::clap_embed::{ClapEmbedder, ClapTextEmbedder};
use symthaea_muse::steering::{STEERING_SAMPLE_RATE, steer};
use symthaea_muse::{AudioData, MusicalState};
use symthaea_music_theory::{MusicalIntent, Style};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let prompt = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "a gentle nostalgic waltz".to_string());
    let n: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(6);

    // Style from the prompt's obvious keyword, else Classical — the point of
    // this demo is the RANKING; callers with real style-routing needs can
    // steer per style and compare score distributions.
    let lower = prompt.to_lowercase();
    let style = if lower.contains("waltz") {
        Style::Waltz
    } else if lower.contains("folk") {
        Style::Folk
    } else if lower.contains("cinematic") || lower.contains("epic") {
        Style::Cinematic
    } else if lower.contains("playful") || lower.contains("bouncy") {
        Style::Playful
    } else {
        Style::Classical
    };

    eprintln!("Loading CLAP towers (first run downloads ~620MB, cached after)…");
    let mut audio_tower = ClapEmbedder::new()?;
    let mut text_tower = ClapTextEmbedder::new()?;

    eprintln!("Composing and ranking {n} candidates for: \"{prompt}\" [{style:?}]");
    let intent = MusicalIntent::default();
    let state = MusicalState::default();
    let (best, scores) = steer(
        &intent,
        style,
        &state,
        &prompt,
        n,
        &mut audio_tower,
        &mut text_tower,
    )?;

    println!("\nRanking (cosine similarity to the prompt):");
    for (rank, s) in scores.iter().enumerate() {
        println!(
            "  #{:<2} seed {:<3} similarity {:.4}",
            rank + 1,
            s.seed,
            s.similarity
        );
    }

    let out_dir = std::path::Path::new("audio_output/prompt_steering");
    std::fs::create_dir_all(out_dir)?;
    let out = out_dir.join("best.wav");
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: STEERING_SAMPLE_RATE,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(&out, spec)?;
    if let AudioData::StereoF32(frames) = &best.audio {
        for [l, r] in frames {
            writer.write_sample(*l)?;
            writer.write_sample(*r)?;
        }
    }
    writer.finalize()?;
    println!(
        "\nBest candidate (seed {}) written to {}",
        scores[0].seed,
        out.display()
    );
    Ok(())
}
