// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Voice gallery: after the formant-vocoder voice was rejected outright
//! ("not pleasant") and the user asked for something "sexy silk to the ears,"
//! this synthesizes the SAME phrase with a curated spread of Kokoro's other
//! built-in voices (the model ships ~50; only `af_heart` — the current
//! default — and `am_michael` were previously downloaded) so a human can
//! actually pick one, rather than anyone guessing which voice "sounds" a
//! particular way from its name alone.
//!
//! Each voice reloads the whole engine (the ONNX session itself is shared
//! logic, but `KokoroEngine::load` ties one style-embedding file to one
//! engine instance — see `src/voice/kokoro_engine.rs`), so this is slower
//! than synthesizing once, but still just a handful of seconds per voice.
//!
//! ```bash
//! nix develop -c cargo run --example kokoro_voice_gallery --features voice-tts
//! ```

use anyhow::Result;
use symthaea::voice::{KokoroConfig, KokoroEngine, save_wav};

const PHRASE: &str = "Hello, I'm Symthaea. It's lovely to speak with you.";

/// The FULL voice pack — 55 voices (not the 69 asked for; that's the real
/// count in `onnx-community/Kokoro-82M-v1.0-ONNX`'s `voices/` directory,
/// confirmed via the HF Hub API file listing, 2026-07-18). Deliberately not
/// claiming to know in advance which one is "silky" — that's a subjective,
/// listen-and-judge call.
///
/// About half of these are NOT English (`jf_`/`jm_` Japanese, `zf_`/`zm_`
/// Mandarin, `hf_`/`hm_` Hindi, `if_`/`im_` Italian, `pf_`/`pm_`
/// Portuguese, `ff_` French, `ef_`/`em_` Spanish) — Kokoro's G2P/pronunciation
/// model is voice-specific, so synthesizing the English PHRASE through a
/// non-English voice will likely mispronounce it, not just "sound accented."
/// Included anyway since all voices were asked for, but the fair
/// apples-to-apples "sexy silk" candidates are the `af_`/`am_`/`bf_`/`bm_`
/// (American/British English) voices — 29 of the 55.
const VOICES: &[&str] = &[
    // American female (12)
    "af",
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart", // current default
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    // American male (9)
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    // British female (4)
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    // British male (4)
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
    // Spanish (3)
    "ef_dora",
    "em_alex",
    "em_santa",
    // French (1)
    "ff_siwis",
    // Hindi (4)
    "hf_alpha",
    "hf_beta",
    "hm_omega",
    "hm_psi",
    // Italian (2)
    "if_sara",
    "im_nicola",
    // Japanese (5)
    "jf_alpha",
    "jf_gongitsune",
    "jf_nezumi",
    "jf_tebukuro",
    "jm_kumo",
    // Brazilian Portuguese (3)
    "pf_dora",
    "pm_alex",
    "pm_santa",
    // Mandarin Chinese (8)
    "zf_xiaobei",
    "zf_xiaoni",
    "zf_xiaoxiao",
    "zf_xiaoyi",
    "zm_yunjian",
    "zm_yunxi",
    "zm_yunxia",
    "zm_yunyang",
];

fn main() -> Result<()> {
    let out_dir = "audio_output/kokoro_voice_gallery_2026-07-18";
    std::fs::create_dir_all(out_dir)?;

    println!(
        "Kokoro voice gallery: synthesizing {PHRASE:?} with {} voices\n",
        VOICES.len()
    );

    for voice in VOICES {
        let path = format!("{out_dir}/{voice}.wav");
        if std::path::Path::new(&path).exists() {
            println!("  [SKIP] {voice:<12} already generated -> {path}");
            continue;
        }
        let config = KokoroConfig {
            voices_filename: format!("voices/{voice}.bin"),
            ..KokoroConfig::default()
        };
        match KokoroEngine::load(config) {
            Some(mut engine) => {
                let sample_rate = engine.sample_rate();
                match engine.synthesize(PHRASE, None) {
                    Some(samples) => {
                        let secs = samples.len() as f32 / sample_rate as f32;
                        save_wav(&samples, sample_rate, &path)?;
                        println!("  [OK]   {voice:<12} {secs:.2}s -> {path}");
                    }
                    None => println!("  [FAIL] {voice:<12} synthesis returned no audio"),
                }
            }
            None => println!("  [FAIL] {voice:<12} engine failed to load (download issue?)"),
        }
    }

    println!("\nDone. Listen to each in {out_dir}/ and pick one.");
    Ok(())
}
