// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Track A TRAINED smoke test: does training the controller before speaking
//! actually help? (Follow-up to `track_a_smoke_test.rs`, which drove the
//! pipeline through 3 short phrases with genesis-random weights only --
//! WER=100%, naturalness below even a known-bad baseline. That harness's
//! own adversarial-review agent flagged the result as not a fair capability
//! ceiling, since the controller had literally never been trained -- matches
//! the "untrained genesis-random weights" pattern documented for every
//! robotics/embodiment bridge in this monorepo.)
//!
//! This harness:
//! 1. Builds a FormantTarget for EVERY canonical ARPAbet phoneme this crate
//!    recognizes (derived from `phonetics::arpabet_articulation`'s own
//!    articulatory metadata, not hand-picked) -- 39 non-silence phonemes.
//! 2. Trains `VocalTractPipeline::controller` on that full table via the
//!    crate's own real, unit-tested `train_on_phoneme_targets` method,
//!    recording a first-epoch loss and a longer-training final loss (same
//!    before/after pattern as `controller.rs::test_phoneme_training_reduces_loss`)
//!    so the harness itself proves training genuinely reduced loss.
//! 3. Synthesizes a larger, phonetically-balanced phrase set (8 words/phrases,
//!    up from the original 3) covering vowels, stops, fricatives, nasals,
//!    liquids, glides, and an affricate, through the SAME audio backend as
//!    the untrained harness (`speech::vocoder::synthesize` -- the only
//!    audio-producing path this crate has).
//!
//! Output goes to a NEW directory (`/tmp/vocal-tract-track-a-trained-smoketest`),
//! deliberately separate from the untrained harness's
//! `/tmp/vocal-tract-track-a-smoketest`, so both exist side by side for a
//! before/after comparison.
//!
//! ```bash
//! cargo run -p symthaea-vocal-tract --example track_a_trained_smoke_test --features hound
//! ```

use std::fs;
use std::io::Write;

use symthaea_core::genesis::GenesisSeed;
use symthaea_vocal_tract::controller::VocalTractController;
use symthaea_vocal_tract::encoder::VoiceCognitiveState;
use symthaea_vocal_tract::phonetics::{PhonemeClass, arpabet_articulation};
use symthaea_vocal_tract::pipeline::VocalTractPipeline;
use symthaea_vocal_tract::speech;
use symthaea_vocal_tract::types::{FormantFrame, FormantTarget, SourceType};

const SAMPLE_RATE: u32 = 22_050; // same as track_a_smoke_test.rs, for comparable scoring
const DT: f32 = 0.005; // 200 Hz motor tick
const FRAMES_PER_PHONEME: usize = 10; // 10 * 5ms = 50ms per phoneme

/// First-epoch loss vs. this many epochs. `test_phoneme_training_reduces_loss`
/// uses 1-vs-5 and `test_adaptive_rate_limiting` uses 20 on a 2-phoneme table;
/// this table is ~20x bigger (39 phonemes covering every manner class), so more
/// epochs are budgeted to give the network a realistic chance to separate all
/// of them, while staying well under `test_bptt_modifies_hidden_weights`'s
/// 200-step precedent for "how much training this crate considers reasonable
/// to run in a single test/example."
const FINAL_EPOCHS: usize = 40;

/// 8 short phrases, deliberately larger and more phonetically balanced than
/// the original 3-word smoke test. Between them they exercise every manner
/// class the crate's vocoder distinguishes: vowels, stops, fricatives,
/// nasals, liquids, glides, and at least one affricate (JH, CH). Every
/// symbol below is drawn from `phonetics::canonical_arpabet_symbol`'s
/// accepted set (verified against source before writing this).
const PHRASES: &[(&str, &[&str])] = &[
    ("hello", &["HH", "AH", "L", "OW"]), // fricative, vowel, liquid, vowel
    ("cat", &["K", "AE", "T"]),          // stop, vowel, stop
    ("goodnight", &["G", "UH", "D", "N", "AY", "T"]), // stop, vowel, stop, nasal, vowel, stop
    ("fish", &["F", "IH", "SH"]),        // fricative, vowel, fricative
    ("jump", &["JH", "AH", "M", "P"]),   // affricate, vowel, nasal, stop
    ("yellow", &["Y", "EH", "L", "OW"]), // glide, vowel, liquid, vowel
    ("watch", &["W", "AA", "CH"]),       // glide, vowel, affricate
    ("singing", &["S", "IH", "NG", "IH", "NG"]), // fricative, vowel, nasal, vowel, nasal
];

/// Build a `FormantTarget` for one ARPAbet phoneme from the crate's own
/// articulatory metadata (`arpabet_articulation`), not hand-copied numbers.
/// Mirrors the SourceType mapping `pipeline.rs::source_type_for_class` uses
/// internally (not `pub`, so duplicated here rather than imported) --
/// `PhonemeClass::Glide` (W/Y) maps to `SourceType::Liquid` since `SourceType`
/// itself has no Glide variant.
fn target_for_phoneme(symbol: &str) -> FormantTarget {
    let meta = arpabet_articulation(symbol);
    // Typical duration per manner class (ms) -- vowels/nasals/liquids are
    // longer sonorants, stops/affricates are short and transient.
    let duration_ms = match meta.class {
        PhonemeClass::Vowel => 150.0,
        PhonemeClass::Nasal => 100.0,
        PhonemeClass::Liquid | PhonemeClass::Glide => 90.0,
        PhonemeClass::Fricative => 120.0,
        PhonemeClass::Affricate => 100.0,
        PhonemeClass::Stop => 80.0,
        PhonemeClass::Silence => 0.0,
    };

    match meta.class {
        PhonemeClass::Vowel => FormantTarget::vowel(meta.f1, meta.f2, meta.f3, duration_ms),
        PhonemeClass::Stop => {
            if meta.voiced {
                FormantTarget::voiced_consonant(meta.f1, meta.f2, meta.f3, duration_ms)
                    .with_manner(SourceType::Stop)
            } else {
                FormantTarget::unvoiced_consonant(meta.f1, meta.f2, meta.f3, duration_ms)
                    .with_manner(SourceType::Stop)
            }
        }
        PhonemeClass::Fricative => {
            if meta.voiced {
                FormantTarget::voiced_consonant(meta.f1, meta.f2, meta.f3, duration_ms)
                    .with_manner(SourceType::Fricative)
            } else {
                FormantTarget::unvoiced_consonant(meta.f1, meta.f2, meta.f3, duration_ms)
                    .with_manner(SourceType::Fricative)
            }
        }
        PhonemeClass::Nasal => {
            // Nasals are always voiced in this crate's articulation table.
            FormantTarget::voiced_consonant(meta.f1, meta.f2, meta.f3, duration_ms)
                .with_manner(SourceType::Nasal)
        }
        PhonemeClass::Liquid | PhonemeClass::Glide => {
            // L/R (Liquid) and W/Y (Glide) are always voiced; SourceType has
            // no Glide variant, so both classes map to SourceType::Liquid,
            // matching pipeline.rs's own internal mapping.
            FormantTarget::voiced_consonant(meta.f1, meta.f2, meta.f3, duration_ms)
                .with_manner(SourceType::Liquid)
        }
        PhonemeClass::Affricate => {
            if meta.voiced {
                FormantTarget::voiced_consonant(meta.f1, meta.f2, meta.f3, duration_ms)
                    .with_manner(SourceType::Affricate)
            } else {
                FormantTarget::unvoiced_consonant(meta.f1, meta.f2, meta.f3, duration_ms)
                    .with_manner(SourceType::Affricate)
            }
        }
        PhonemeClass::Silence => {
            // Not expected to be reached -- CANONICAL_PHONEMES below excludes
            // silence symbols -- but handled explicitly rather than panicking.
            FormantTarget::unvoiced_consonant(meta.f1, meta.f2, meta.f3, duration_ms)
                .with_manner(SourceType::Silent)
        }
    }
}

/// Every canonical non-silence ARPAbet symbol this crate's phonetics module
/// recognizes (verified against `phonetics.rs`'s own
/// `canonical_arpabet_symbol`/`arpabet_articulation` match arms and its
/// `CANONICAL_NONSILENCE_SYMBOLS` test fixture before writing this -- this
/// list is NOT a cherry-picked subset, it is the full accepted set).
const CANONICAL_PHONEMES: &[&str] = &[
    // Vowels (15)
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",
    // Stops (6)
    "P", "B", "T", "D", "K", "G", // Fricatives (9)
    "F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH", // Nasals (3)
    "M", "N", "NG", // Liquids (2)
    "L", "R", // Glides (2)
    "W", "Y", // Affricates (2)
    "CH", "JH",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = "/tmp/vocal-tract-track-a-trained-smoketest";
    fs::create_dir_all(out_dir)?;

    println!("=== symthaea-vocal-tract Track A TRAINED smoke test ===");
    println!(
        "Audio backend in use: speech::vocoder::synthesize (legacy formant \
         cascade synthesizer -- 3 StableResonators + glottal/noise excitation), \
         same as the untrained track_a_smoke_test.rs."
    );
    println!(
        "Sample rate: {SAMPLE_RATE} Hz, output dir: {out_dir}\n\
         Untrained comparison harness output remains at \
         /tmp/vocal-tract-track-a-smoketest (not touched by this run).\n"
    );

    // ─────────────────────────────────────────────────────────────────
    // Phase 1: build the full phoneme target table from real articulatory
    // metadata, and train the pipeline's own controller on it.
    // ─────────────────────────────────────────────────────────────────
    assert_eq!(
        CANONICAL_PHONEMES.len(),
        39,
        "expected 39 canonical non-silence ARPAbet symbols"
    );

    let phoneme_targets: Vec<(&str, FormantTarget)> = CANONICAL_PHONEMES
        .iter()
        .map(|&sym| (sym, target_for_phoneme(sym)))
        .collect();
    let target_refs: Vec<(&str, &FormantTarget)> =
        phoneme_targets.iter().map(|(name, t)| (*name, t)).collect();

    println!(
        "Training table: {} phonemes covering vowels, stops, fricatives, \
         nasals, liquids, glides, and affricates (full canonical set, \
         derived from arpabet_articulation -- not cherry-picked).\n",
        target_refs.len()
    );

    let training_genesis = GenesisSeed::from_phrase("track-a-trained-smoke::training");

    // First-epoch loss, on a throwaway baseline controller -- same
    // before/after pattern as controller.rs::test_phoneme_training_reduces_loss.
    let config = symthaea_vocal_tract::controller::VocalTractConfig::default();
    let mut baseline_ctrl = VocalTractController::new(&training_genesis, &config);
    let first_epoch_loss =
        baseline_ctrl.train_on_phoneme_targets(&training_genesis, &target_refs, 1);

    // Now build the pipeline we'll actually synthesize with, and train ITS
    // controller for FINAL_EPOCHS epochs.
    let mut pipeline = VocalTractPipeline::new(&training_genesis);
    let final_loss =
        pipeline
            .controller
            .train_on_phoneme_targets(&training_genesis, &target_refs, FINAL_EPOCHS);

    let improved = final_loss < first_epoch_loss;
    println!(
        "Training result: first_epoch_loss={first_epoch_loss:.4}, \
         final_loss(after {FINAL_EPOCHS} epochs)={final_loss:.4} -> \
         {}\n",
        if improved {
            "GENUINE CONVERGENCE (final < first)"
        } else {
            "NO IMPROVEMENT (final >= first) -- training did not converge"
        }
    );

    // ─────────────────────────────────────────────────────────────────
    // Phase 2: synthesize the phonetically-balanced phrase set using the
    // NOW-TRAINED pipeline controller.
    // ─────────────────────────────────────────────────────────────────
    let state = VoiceCognitiveState::default();

    for (name, phonemes) in PHRASES {
        let mut frames: Vec<FormantFrame> = Vec::new();
        for phoneme in *phonemes {
            for _ in 0..FRAMES_PER_PHONEME {
                let frame = pipeline.tick_phoneme(&state, None, DT, Some(phoneme));
                frames.push(frame);
            }
        }

        let duration_s = frames.len() as f32 * DT;
        println!(
            "[{name}] phonemes={:?} frames={} duration={:.3}s",
            phonemes,
            frames.len(),
            duration_s
        );

        // Render real PCM via the legacy formant vocoder (same backend as
        // the untrained harness).
        let samples = speech::vocoder::synthesize(&frames, SAMPLE_RATE);
        let non_silent = samples.iter().any(|s| s.abs() > 1e-4);
        let has_nan_or_inf = samples.iter().any(|s| !s.is_finite());
        let peak = samples.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        println!(
            "  -> synthesized {} PCM samples ({:.3}s at {SAMPLE_RATE}Hz), \
             non_silent={non_silent}, peak_amplitude={peak:.4}, \
             finite_only={}",
            samples.len(),
            samples.len() as f32 / SAMPLE_RATE as f32,
            !has_nan_or_inf
        );

        #[cfg(feature = "hound")]
        {
            let wav_path = format!("{out_dir}/{name}.wav");
            symthaea_vocal_tract::metrics::save_wav(&wav_path, &samples, SAMPLE_RATE)?;
            println!("  -> wrote {wav_path}");
        }
        #[cfg(not(feature = "hound"))]
        {
            println!(
                "  -> 'hound' feature not enabled: skipping .wav write. \
                 Re-run with --features hound to actually write audio to disk."
            );
        }

        // Always write the debug FormantFrame trajectory as CSV, independent
        // of the audio backend.
        let csv_path = format!("{out_dir}/{name}_frames.csv");
        let mut f = fs::File::create(&csv_path)?;
        writeln!(f, "time,f1,f2,f3,b1,b2,b3,f0,energy,voicing")?;
        for frame in &frames {
            writeln!(
                f,
                "{},{},{},{},{},{},{},{},{},{}",
                frame.time,
                frame.f1,
                frame.f2,
                frame.f3,
                frame.b1,
                frame.b2,
                frame.b3,
                frame.f0,
                frame.energy,
                frame.voicing
            )?;
        }
        println!("  -> wrote {csv_path}\n");
    }

    println!(
        "Done. Trained-controller comparison harness: {} phrases synthesized \
         after training on the full {}-phoneme canonical ARPAbet table \
         ({FINAL_EPOCHS} epochs, first_epoch_loss={first_epoch_loss:.4} -> \
         final_loss={final_loss:.4}).",
        PHRASES.len(),
        CANONICAL_PHONEMES.len()
    );
    Ok(())
}
