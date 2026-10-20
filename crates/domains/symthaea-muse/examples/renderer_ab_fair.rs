// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Controlled renderer comparison plus same-MIDI profile ablation across all eight dialects.
//!
//! Three arms are emitted per style: native with the dialect profile, FluidSynth
//! with its default/no-dialect profile, and FluidSynth with the dialect profile.
//! The two FluidSynth arms consume the exact same performed MIDI and SoundFont;
//! their contrast isolates the combined dialect room/mastering settings. The
//! native-versus-FluidSynth comparison remains a backend comparison, not a pure
//! profile test. All arms share the loudness target and limiter ceiling.
//!
//! # What this measures
//!
//! It records objective per-arm loudness, sample peak, crest factor, and broad
//! spectral balance, plus hashes for source/build/assets/MIDI/audio and the
//! applied profile. These identify gross differences and catch failed or
//! imbalanced renders; none alone is a musical-quality metric. The first
//! profile A/B predates the eight-dialect mapping and is historical context only.
//! Re-run this harness for any new profile revision, then use the separate
//! blind listening protocol in STYLE_RENDER_PROFILE_BENCHMARK_V1.md before
//! accepting a profile as musically beneficial.
//!
//! # What it does NOT do
//!
//! It produces no blinded listening pack and asks for no listening. It emits the
//! WAV variants for a later randomized study, but audio metrics are not a verdict
//! on preference or authenticity.
//!
//! Renders land in `audio_output/renderer-ab-fair-{synth|vcsl}/` relative to the
//! CARGO WORKSPACE root, which `symthaea/.gitignore` already excludes.
//!
//! # Historical baseline (2026-07-30; NOT the current eight-profile run)
//!
//! Before the per-style profiles and eight-dialect cohort were added, both arms
//! landed within 1.1 dB of the −16 LUFS target. Crest factor differed by +6.52 dB
//! in FluidSynth's favour, with the same sign on 5/5 pieces; native renders had
//! 6.36–8.19 dB crest and peaks at −8.3 to −10.6 dB, while FluidSynth renders had
//! 11.77–15.03 dB crest and peaks near the −1.5 dB ceiling. FluidSynth was also
//! slightly brighter (+0.063 high band) with slightly less low weight (−0.029).
//!
//! Those measurements belong to the older five-piece/default-mastering cohort.
//! They do NOT validate the new per-dialect profiles and must not be presented as
//! evidence for this revision. Crest factor is a diagnostic, not a quality metric;
//! higher is not automatically better. Re-run the current harness under a pinned
//! SoundFont/VCSL environment and record the new output before interpreting it.
//!
//! ```sh
//! export SYMTHAEA_GIT_REVISION="$(git rev-parse HEAD)"
//! nix shell nixpkgs#fluidsynth nixpkgs#soundfont-fluid --command bash -c '
//!   export SYMTHAEA_GIT_REVISION="$SYMTHAEA_GIT_REVISION"
//!   export SYMTHAEA_FLUIDSYNTH=$(which fluidsynth)
//!   export SYMTHAEA_SOUNDFONT=$(ls /nix/store/*/share/soundfonts/FluidR3_GM2-2.sf2 | head -1)
//!   cargo run --release -p symthaea-muse --features theory --example renderer_ab_fair'
//! ```

use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use symthaea_muse::{AudioData, MusicalState, auto_master, fluid_render, midi_export};
use symthaea_music_theory::{MusicalIntent, Style, compose_styled};

/// Objective, renderer-agnostic measurements of one rendered arm.
struct Arm {
    name: &'static str,
    lufs: f32,
    peak_db: f32,
    crest_db: f32,
    low: f32,
    mid: f32,
    high: f32,
    frames: usize,
    /// SHA-256 over canonical little-endian f32 stereo samples (WAV header excluded).
    pcm_sha256: String,
}

/// Per-cell factorial contrasts. Metric order is LUFS, crest, low-band ratio,
/// high-band ratio. Interaction is full - room-only - mastering-only + default.
struct ProfileFactorial {
    room_effect_at_default_master: [f32; 4],
    master_effect_at_default_room: [f32; 4],
    room_master_interaction: [f32; 4],
}

fn subtract_metrics(left: &Arm, right: &Arm) -> [f32; 4] {
    [
        left.lufs - right.lufs,
        left.crest_db - right.crest_db,
        left.low - right.low,
        left.high - right.high,
    ]
}

fn factorial_contrasts(
    baseline: &Arm,
    room_only: &Arm,
    master_only: &Arm,
    combined: &Arm,
) -> ProfileFactorial {
    let room_effect_at_default_master = subtract_metrics(room_only, baseline);
    let master_effect_at_default_room = subtract_metrics(master_only, baseline);
    let room_master_interaction: [f32; 4] = std::array::from_fn(|i| {
        combined_metric(combined, i)
            - combined_metric(room_only, i)
            - combined_metric(master_only, i)
            + combined_metric(baseline, i)
    });
    ProfileFactorial {
        room_effect_at_default_master,
        master_effect_at_default_room,
        room_master_interaction,
    }
}

fn combined_metric(arm: &Arm, index: usize) -> f32 {
    match index {
        0 => arm.lufs,
        1 => arm.crest_db,
        2 => arm.low,
        3 => arm.high,
        _ => unreachable!("factorial metric index is bounded by the four-element array"),
    }
}

fn factorial_json(value: [f32; 4]) -> serde_json::Value {
    serde_json::json!({
        "integrated_lufs": value[0],
        "crest_factor_db": value[1],
        "spectral_balance_low": value[2],
        "spectral_balance_high": value[3]
    })
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Streaming hash so large SoundFonts and executables need not be read into memory.
fn sha256_file(path: &Path) -> std::io::Result<(String, u64)> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut total = 0u64;
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        total += read as u64;
    }
    Ok((format!("{:x}", hasher.finalize()), total))
}

fn sha256_frames(frames: &[[f32; 2]]) -> String {
    let mut hasher = Sha256::new();
    for frame in frames {
        for sample in frame {
            hasher.update(sample.to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn valid_git_revision(value: &str) -> bool {
    (value.len() == 40 || value.len() == 64) && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn fluidsynth_version(binary: &Path) -> Option<String> {
    let output = std::process::Command::new(binary)
        .arg("--version")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if stdout.is_empty() && stderr.is_empty() {
        return None;
    }
    Some(format!("stdout={stdout}; stderr={stderr}"))
}

fn arm_metrics(arm: &Arm) -> serde_json::Value {
    serde_json::json!({
        "integrated_lufs": arm.lufs,
        "sample_peak_dbfs": arm.peak_db,
        "crest_factor_db": arm.crest_db,
        "spectral_balance_low_mid_high": [arm.low, arm.mid, arm.high],
        "stereo_frames": arm.frames,
        "canonical_pcm_sha256": arm.pcm_sha256,
        "pcm_hash_encoding": "interleaved stereo f32 little-endian; WAV header excluded"
    })
}

fn measure(name: &'static str, frames: &[[f32; 2]], sr: u32) -> Arm {
    let l = auto_master::measure_lufs(frames, sr);
    let b = auto_master::analyze_balance(frames, sr);
    Arm {
        name,
        lufs: l.integrated,
        peak_db: l.peak_db,
        crest_db: l.crest_factor_db,
        low: b.low_ratio(),
        mid: b.mid_ratio(),
        high: b.high_ratio(),
        frames: frames.len(),
        pcm_sha256: sha256_frames(frames),
    }
}

fn wav_to_frames(bytes: &[u8]) -> Option<(Vec<[f32; 2]>, u32)> {
    let mut r = hound::WavReader::new(std::io::Cursor::new(bytes)).ok()?;
    let spec = r.spec();
    if spec.channels == 0 || spec.sample_rate == 0 {
        return None;
    }
    let ch = spec.channels as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 * scale))
                .collect::<Result<Vec<_>, _>>()
                .ok()?
        }
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<Vec<_>, _>>().ok()?,
    };
    if samples.is_empty()
        || samples.len() % ch != 0
        || samples.iter().any(|sample| !sample.is_finite())
    {
        return None;
    }
    let frames = if ch == 1 {
        samples.iter().map(|&s| [s, s]).collect()
    } else {
        samples.chunks_exact(ch).map(|c| [c[0], c[1]]).collect()
    };
    if frames.is_empty() {
        return None;
    }
    Some((frames, spec.sample_rate))
}

fn main() {
    const SR: u32 = 48_000;
    // One deterministic style per declared grammar performance dialect.
    // This is a renderer/profile smoke cohort, not a perceptual truth set.
    // A separate randomized blind study is required to accept/reject profile
    // hypotheses for style identity or listener preference.
    let cohort: &[(Style, u64)] = &[
        (Style::Classical, 7),          // ClassicalRubato
        (Style::Tango, 7),              // DanceLocked
        (Style::JazzBallad, 7),         // JazzLaidBack
        (Style::Folk, 7),               // FolkLift
        (Style::Minimalism, 7),         // ProcessExact
        (Style::HindustaniInspired, 7), // DroneElastic
        (Style::Cinematic, 7),          // DramaticTiming
        (Style::SacredChoral, 7),       // ChoralBlend
    ];

    let Some((bin, sf)) = fluid_render::available() else {
        eprintln!(
            "ABORT: FluidSynth unavailable. This harness compares TWO renderers; running it \
             with one would silently measure the native arm twice and report a meaningless \
             null. Set SYMTHAEA_FLUIDSYNTH and SYMTHAEA_SOUNDFONT (see this file's header)."
        );
        std::process::exit(2);
    };
    println!("FluidSynth: {}", bin.display());
    println!("SoundFont:  {}", sf.display());

    // Evidence identity is caller-bound to the checked-out source revision; this
    // report never claims that the caller-supplied revision authenticates a build.
    let source_revision = std::env::var("SYMTHAEA_GIT_REVISION").ok();
    let source_revision_valid = source_revision.as_deref().is_some_and(valid_git_revision);
    let executable_path = std::env::current_exe().unwrap_or_else(|error| {
        eprintln!("ABORT: cannot identify current executable: {error}");
        std::process::exit(2);
    });
    let (executable_sha256, executable_bytes) =
        sha256_file(&executable_path).unwrap_or_else(|error| {
            eprintln!("ABORT: cannot hash current executable: {error}");
            std::process::exit(2);
        });
    let (fluidsynth_sha256, fluidsynth_bytes) = sha256_file(&bin).unwrap_or_else(|error| {
        eprintln!("ABORT: cannot hash FluidSynth executable: {error}");
        std::process::exit(2);
    });
    let (soundfont_sha256, soundfont_bytes) = sha256_file(&sf).unwrap_or_else(|error| {
        eprintln!("ABORT: cannot hash SoundFont: {error}");
        std::process::exit(2);
    });
    let fluidsynth_version = fluidsynth_version(&bin);
    let vcsl_manifest_sha256 = std::env::var("SYMTHAEA_VCSL_MANIFEST_SHA256").ok();
    let vcsl_manifest_valid = vcsl_manifest_sha256.as_deref().is_some_and(valid_sha256);

    // WHICH native renderer is under test. `vcsl::library()` is a process-global
    // OnceLock, so this cannot be toggled mid-run — the three-way comparison is
    // two invocations of this binary with and without SYMTHAEA_VCSL_DIR, and
    // this banner is what keeps their outputs from being confused for each
    // other. The first run of this harness (2026-07-30) did NOT set the
    // variable, so its "native" arm was pure synthesis while 14.8 GB of real
    // orchestral samples sat unused — a limitation discovered only afterwards,
    // which is exactly why the mode is now printed rather than assumed.
    let vcsl_enabled = symthaea_muse::vcsl::library().is_some();
    let native_mode = if vcsl_enabled {
        "native+VCSL (real recorded samples where mapped, synthesis elsewhere)"
    } else {
        "native-synthesis (NO samples — SYMTHAEA_VCSL_DIR unset)"
    };
    println!("Native arm: {native_mode}");
    println!(
        "Both arms use the same dialect-aware mastering profile per piece; all profiles share \
         target {} LUFS and ceiling {} dB, while broad EQ targets vary as experimental hypotheses.\n",
        auto_master::MasteringConfig::default().target_lufs,
        auto_master::MasteringConfig::default().limiter_ceiling_db
    );

    let tag = if vcsl_enabled { "vcsl" } else { "synth" };
    let out_dir = PathBuf::from(format!("audio_output/renderer-ab-fair-{tag}"));
    std::fs::create_dir_all(&out_dir).expect("create output dir");
    let manifest_path = out_dir.join("renderer-ab-fair-manifest.json");

    let mut deltas: Vec<(String, f32, f32, f32)> = Vec::new();
    let mut profile_deltas: Vec<(String, f32, f32, f32, f32)> = Vec::new();
    let mut rows: Vec<serde_json::Value> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    for &(style, seed) in cohort {
        let intent = MusicalIntent {
            seed,
            ..Default::default()
        };
        let score = compose_styled(&intent, style);
        let spec = style.spec();
        let state = MusicalState::default();

        // ── Arm A: native additive/sample renderer, mastered in realize_core.
        let grammar_dialect = style.grammar_profile().performance;
        let renderer_dialect = symthaea_muse::theory_realize::map_grammar_dialect(grammar_dialect);
        let comp = symthaea_muse::theory_realize::realize_with_spec_and_dialect(
            &score,
            &spec,
            seed,
            &state,
            SR,
            grammar_dialect,
        );
        let AudioData::StereoF32(native_frames) = &comp.audio else {
            let reason = format!("{style:?}/{seed}: native render was not stereo f32");
            eprintln!("{reason}; skipping");
            failures.push(reason);
            rows.push(serde_json::json!({
                "style": format!("{style:?}"), "seed": seed, "status": "failed",
                "failure_code": "native_render_not_stereo_f32"
            }));
            continue;
        };
        let a = measure(
            if tag == "vcsl" {
                "native+vcsl"
            } else {
                "native-synth"
            },
            native_frames,
            SR,
        );

        // ── Arm B: the SAME performed MIDI through FluidSynth, now also mastered.
        let mid = out_dir.join(format!("{style:?}_{seed}.mid"));
        if midi_export::export_performance_midi_with_dialect(
            &score,
            &spec,
            seed,
            &state,
            grammar_dialect,
            &mid,
        )
        .is_err()
        {
            let reason = format!("{style:?}/{seed}: MIDI export failed");
            eprintln!("{reason}; skipping");
            failures.push(reason);
            rows.push(serde_json::json!({
                "style": format!("{style:?}"), "seed": seed, "status": "failed",
                "failure_code": "midi_export_failed"
            }));
            continue;
        }
        let (midi_sha256, midi_bytes) = match sha256_file(&mid) {
            Ok(identity) => identity,
            Err(error) => {
                let reason = format!("{style:?}/{seed}: cannot hash MIDI: {error}");
                eprintln!("{reason}; skipping");
                failures.push(reason);
                rows.push(serde_json::json!({
                    "style": format!("{style:?}"), "seed": seed, "status": "failed",
                    "failure_code": "midi_hash_failed"
                }));
                continue;
            }
        };
        let base_color = fluid_render::RenderColor::from_state(&state);
        let neutral_mastering = auto_master::MasteringConfig::default();

        // Same MIDI, same SoundFont, same FluidSynth binary/sample rate: this
        // arm is the no-dialect baseline for the profile ablation.
        let Some(neutral_wav) = fluid_render::render_midi_to_wav_with_mastering(
            &mid,
            SR,
            Some(base_color),
            &neutral_mastering,
        ) else {
            let reason = format!("{style:?}/{seed}: default-profile FluidSynth render failed");
            eprintln!("{reason}; skipping");
            failures.push(reason);
            rows.push(serde_json::json!({
                "style": format!("{style:?}"), "seed": seed, "status": "failed",
                "failure_code": "fluidsynth_default_profile_render_failed",
                "midi_sha256": midi_sha256
            }));
            continue;
        };
        let Some((neutral_frames, neutral_sr)) = wav_to_frames(&neutral_wav) else {
            let reason = format!("{style:?}/{seed}: cannot decode default-profile WAV");
            eprintln!("{reason}; skipping");
            failures.push(reason);
            rows.push(serde_json::json!({
                "style": format!("{style:?}"), "seed": seed, "status": "failed",
                "failure_code": "fluidsynth_default_profile_decode_failed",
                "midi_sha256": midi_sha256
            }));
            continue;
        };
        let neutral = measure("fluidsynth-default-profile", &neutral_frames, neutral_sr);
        let neutral_wav_sha256 = sha256_bytes(&neutral_wav);
        let neutral_wav_path = out_dir.join(format!("{style:?}_{seed}_fluidsynth-default.wav"));
        if let Err(error) = std::fs::write(&neutral_wav_path, &neutral_wav) {
            let reason = format!("{style:?}/{seed}: cannot preserve default-profile WAV: {error}");
            eprintln!("{reason}; skipping");
            failures.push(reason);
            rows.push(serde_json::json!({
                "style": format!("{style:?}"), "seed": seed, "status": "failed",
                "failure_code": "fluidsynth_default_profile_write_failed",
                "midi_sha256": midi_sha256, "wav_sha256": neutral_wav_sha256
            }));
            continue;
        }

        // Factor 1: change only room controls; keep default mastering.
        let color = fluid_render::room_for_dialect(base_color, renderer_dialect);
        let Some(room_wav) = fluid_render::render_midi_to_wav_with_mastering(
            &mid,
            SR,
            Some(color),
            &neutral_mastering,
        ) else {
            let reason = format!("{style:?}/{seed}: room-only FluidSynth render failed");
            eprintln!("{reason}; skipping");
            failures.push(reason);
            rows.push(serde_json::json!({
                "style": format!("{style:?}"), "seed": seed, "status": "failed",
                "failure_code": "fluidsynth_room_only_render_failed",
                "midi_sha256": midi_sha256
            }));
            continue;
        };
        let Some((room_frames, room_sr)) = wav_to_frames(&room_wav) else {
            let reason = format!("{style:?}/{seed}: cannot decode room-only WAV");
            eprintln!("{reason}; skipping");
            failures.push(reason);
            rows.push(serde_json::json!({
                "style": format!("{style:?}"), "seed": seed, "status": "failed",
                "failure_code": "fluidsynth_room_only_decode_failed",
                "midi_sha256": midi_sha256
            }));
            continue;
        };
        let room_only = measure("fluidsynth-room-only", &room_frames, room_sr);
        let room_wav_sha256 = sha256_bytes(&room_wav);
        let room_wav_path = out_dir.join(format!("{style:?}_{seed}_fluidsynth-room-only.wav"));
        if let Err(error) = std::fs::write(&room_wav_path, &room_wav) {
            let reason = format!("{style:?}/{seed}: cannot preserve room-only WAV: {error}");
            eprintln!("{reason}; skipping");
            failures.push(reason);
            rows.push(serde_json::json!({
                "style": format!("{style:?}"), "seed": seed, "status": "failed",
                "failure_code": "fluidsynth_room_only_write_failed",
                "midi_sha256": midi_sha256, "wav_sha256": room_wav_sha256
            }));
            continue;
        }

        // Factor 2: change only mastering; keep slider-driven base room.
        let mastering = auto_master::MasteringConfig::for_dialect(renderer_dialect);
        let Some(master_wav) =
            fluid_render::render_midi_to_wav_with_mastering(&mid, SR, Some(base_color), &mastering)
        else {
            let reason = format!("{style:?}/{seed}: mastering-only FluidSynth render failed");
            eprintln!("{reason}; skipping");
            failures.push(reason);
            rows.push(serde_json::json!({
                "style": format!("{style:?}"), "seed": seed, "status": "failed",
                "failure_code": "fluidsynth_mastering_only_render_failed",
                "midi_sha256": midi_sha256, "room_wav_sha256": room_wav_sha256
            }));
            continue;
        };
        let Some((master_frames, master_sr)) = wav_to_frames(&master_wav) else {
            let reason = format!("{style:?}/{seed}: cannot decode mastering-only WAV");
            eprintln!("{reason}; skipping");
            failures.push(reason);
            rows.push(serde_json::json!({
                "style": format!("{style:?}"), "seed": seed, "status": "failed",
                "failure_code": "fluidsynth_mastering_only_decode_failed",
                "midi_sha256": midi_sha256, "room_wav_sha256": room_wav_sha256
            }));
            continue;
        };
        let master_only = measure("fluidsynth-mastering-only", &master_frames, master_sr);
        let master_wav_sha256 = sha256_bytes(&master_wav);
        let master_wav_path = out_dir.join(format!("{style:?}_{seed}_fluidsynth-master-only.wav"));
        if let Err(error) = std::fs::write(&master_wav_path, &master_wav) {
            let reason = format!("{style:?}/{seed}: cannot preserve mastering-only WAV: {error}");
            eprintln!("{reason}; skipping");
            failures.push(reason);
            rows.push(serde_json::json!({
                "style": format!("{style:?}"), "seed": seed, "status": "failed",
                "failure_code": "fluidsynth_mastering_only_write_failed",
                "midi_sha256": midi_sha256, "wav_sha256": master_wav_sha256
            }));
            continue;
        }

        let Some(wav) =
            fluid_render::render_midi_to_wav_with_mastering(&mid, SR, Some(color), &mastering)
        else {
            let reason = format!("{style:?}/{seed}: dialect-profile FluidSynth render failed");
            eprintln!("{reason}; skipping");
            failures.push(reason);
            rows.push(serde_json::json!({
                "style": format!("{style:?}"), "seed": seed, "status": "failed",
                "failure_code": "fluidsynth_profile_render_failed",
                "midi_sha256": midi_sha256, "default_wav_sha256": neutral_wav_sha256
            }));
            continue;
        };
        let Some((fluid_frames, fsr)) = wav_to_frames(&wav) else {
            let reason = format!("{style:?}/{seed}: could not decode dialect-profile WAV");
            eprintln!("{reason}; skipping");
            failures.push(reason);
            rows.push(serde_json::json!({
                "style": format!("{style:?}"), "seed": seed, "status": "failed",
                "failure_code": "fluidsynth_profile_decode_failed",
                "midi_sha256": midi_sha256, "default_wav_sha256": neutral_wav_sha256
            }));
            continue;
        };
        let b = measure("fluidsynth-dialect-profile", &fluid_frames, fsr);
        let wav_sha256 = sha256_bytes(&wav);
        let wav_path = out_dir.join(format!("{style:?}_{seed}_fluidsynth-profile.wav"));
        if let Err(error) = std::fs::write(&wav_path, &wav) {
            let reason = format!("{style:?}/{seed}: cannot preserve dialect-profile WAV: {error}");
            eprintln!("{reason}; skipping");
            failures.push(reason);
            rows.push(serde_json::json!({
                "style": format!("{style:?}"), "seed": seed, "status": "failed",
                "failure_code": "fluidsynth_profile_write_failed",
                "midi_sha256": midi_sha256, "default_wav_sha256": neutral_wav_sha256,
                "wav_sha256": wav_sha256
            }));
            continue;
        }
        if [neutral_sr, room_sr, master_sr, fsr]
            .iter()
            .any(|&rate| rate != SR)
        {
            let reason =
                format!("{style:?}/{seed}: a FluidSynth arm returned an unexpected sample rate");
            eprintln!("{reason}; skipping");
            failures.push(reason);
            rows.push(serde_json::json!({
                "style": format!("{style:?}"), "seed": seed, "status": "failed",
                "failure_code": "unexpected_sample_rate",
                "expected_sample_rate_hz": SR,
                "observed_sample_rates_hz": [neutral_sr, room_sr, master_sr, fsr],
                "midi_sha256": midi_sha256
            }));
            continue;
        }

        if ![
            a.lufs,
            a.peak_db,
            a.crest_db,
            a.low,
            a.mid,
            a.high,
            neutral.lufs,
            neutral.peak_db,
            neutral.crest_db,
            neutral.low,
            neutral.mid,
            neutral.high,
            room_only.lufs,
            room_only.peak_db,
            room_only.crest_db,
            room_only.low,
            room_only.mid,
            room_only.high,
            master_only.lufs,
            master_only.peak_db,
            master_only.crest_db,
            master_only.low,
            master_only.mid,
            master_only.high,
            b.lufs,
            b.peak_db,
            b.crest_db,
            b.low,
            b.mid,
            b.high,
        ]
        .iter()
        .all(|value| value.is_finite())
        {
            let reason = format!("{style:?}/{seed}: non-finite metric; row invalid");
            eprintln!("{reason}; skipping");
            failures.push(reason);
            rows.push(serde_json::json!({
                "style": format!("{style:?}"), "seed": seed, "status": "failed",
                "failure_code": "non_finite_metric",
                "midi_sha256": midi_sha256, "wav_sha256": wav_sha256
            }));
            continue;
        }

        println!(
            "── {style:?} seed={seed} dialect={:?} ──",
            symthaea_music_theory::Style::grammar_profile(style).performance
        );
        for arm in [&a, &neutral, &room_only, &master_only, &b] {
            println!(
                "  {:<11} LUFS {:>7.2}  peak {:>6.2} dB  crest {:>5.2} dB  \
                 balance L/M/H {:.3}/{:.3}/{:.3}  frames {}",
                arm.name,
                arm.lufs,
                arm.peak_db,
                arm.crest_db,
                arm.low,
                arm.mid,
                arm.high,
                arm.frames
            );
        }
        let d_crest = b.crest_db - a.crest_db;
        let d_high = b.high - a.high;
        let d_low = b.low - a.low;
        println!(
            "  Δ(profile−native): crest {d_crest:+.2} dB   high-band {d_high:+.3}   \
             low-band {d_low:+.3}"
        );
        let profile_crest_delta = b.crest_db - neutral.crest_db;
        let profile_high_delta = b.high - neutral.high;
        let profile_low_delta = b.low - neutral.low;
        let profile_lufs_delta = b.lufs - neutral.lufs;
        let factorial = factorial_contrasts(&neutral, &room_only, &master_only, &b);
        println!(
            "  Δ(dialect−default FluidSynth): LUFS {profile_lufs_delta:+.2}  crest {profile_crest_delta:+.2} dB  \
             high-band {profile_high_delta:+.3}  low-band {profile_low_delta:+.3}"
        );

        let profile_material = format!(
            "melothaea-style-render-profile-v1|grammar={grammar_dialect:?}|renderer={renderer_dialect:?}|target_lufs={:.6}|limiter_ceiling_db={:.6}|low={:.6}|mid={:.6}|high={:.6}|eq_cap={:.6}|high_boost_cap={:.6}|section_leveling={:.6}|room_size={:.6}|room_level={:.6}|room_damp={:.6}|chorus={:.6}",
            mastering.target_lufs,
            mastering.limiter_ceiling_db,
            mastering.target_low_ratio,
            mastering.target_mid_ratio,
            mastering.target_high_ratio,
            mastering.eq_max_correction_db,
            mastering.eq_max_high_boost_db,
            mastering.section_leveling_db,
            color.reverb_room,
            color.reverb_level,
            color.reverb_damp,
            color.chorus_level
        );
        let profile_fingerprint_sha256 = sha256_bytes(profile_material.as_bytes());
        let profile_controls_differ_from_default = color.reverb_room != base_color.reverb_room
            || color.reverb_level != base_color.reverb_level
            || color.reverb_damp != base_color.reverb_damp
            || color.chorus_level != base_color.chorus_level
            || mastering.target_lufs != neutral_mastering.target_lufs
            || mastering.limiter_ceiling_db != neutral_mastering.limiter_ceiling_db
            || mastering.section_leveling_db != neutral_mastering.section_leveling_db
            || mastering.target_low_ratio != neutral_mastering.target_low_ratio
            || mastering.target_mid_ratio != neutral_mastering.target_mid_ratio
            || mastering.target_high_ratio != neutral_mastering.target_high_ratio
            || mastering.eq_max_correction_db != neutral_mastering.eq_max_correction_db
            || mastering.eq_max_high_boost_db != neutral_mastering.eq_max_high_boost_db;
        rows.push(serde_json::json!({
            "style": format!("{style:?}"),
            "seed": seed,
            "status": "complete",
            "grammar_performance_dialect": format!("{grammar_dialect:?}"),
            "renderer_dialect": format!("{renderer_dialect:?}"),
            "profile_version": "melothaea-style-render-profile-v1",
            "profile_fingerprint_sha256": profile_fingerprint_sha256,
            "profile_controls_differ_from_default": profile_controls_differ_from_default,
            "default_vs_profile_pcm_hash_equal": neutral.pcm_sha256 == b.pcm_sha256,
            "default_vs_profile_wav_hash_equal": neutral_wav_sha256 == wav_sha256,
            "profile_vs_default_signal_delta": {
                "integrated_lufs": profile_lufs_delta,
                "crest_factor_db": profile_crest_delta,
                "spectral_balance_low": profile_low_delta,
                "spectral_balance_high": profile_high_delta
            },
            "factorial_contrasts": {
                "metric_order": ["integrated_lufs", "crest_factor_db", "spectral_balance_low", "spectral_balance_high"],
                "room_effect_at_default_master": factorial_json(factorial.room_effect_at_default_master),
                "mastering_effect_at_default_room": factorial_json(factorial.master_effect_at_default_room),
                "room_master_interaction": factorial_json(factorial.room_master_interaction)
            },
            "input_midi": {
                "path": mid.display().to_string(),
                "sha256": midi_sha256,
                "bytes": midi_bytes
            },
            "fluidsynth_room": {
                "room_size": color.reverb_room,
                "level": color.reverb_level,
                "damp": color.reverb_damp,
                "chorus_level": color.chorus_level
            },
            "mastering": {
                "target_lufs": mastering.target_lufs,
                "limiter_ceiling_db": mastering.limiter_ceiling_db,
                "target_low_mid_high": [
                    mastering.target_low_ratio, mastering.target_mid_ratio,
                    mastering.target_high_ratio
                ],
                "eq_max_correction_db": mastering.eq_max_correction_db,
                "eq_max_high_boost_db": mastering.eq_max_high_boost_db,
                "section_leveling_db": mastering.section_leveling_db
            },
            "arms": {
                "native_dialect_profile": {
                    "renderer": a.name,
                    "sample_rate_hz": SR,
                    "metrics": arm_metrics(&a)
                },
                "fluidsynth_default_profile": {
                    "renderer": neutral.name,
                    "sample_rate_hz": neutral_sr,
                    "wav_path": neutral_wav_path.display().to_string(),
                    "wav_sha256": neutral_wav_sha256,
                    "wav_bytes": neutral_wav.len(),
                    "room": {
                        "room_size": base_color.reverb_room,
                        "level": base_color.reverb_level,
                        "damp": base_color.reverb_damp,
                        "chorus_level": base_color.chorus_level
                    },
                    "mastering": {
                        "target_lufs": neutral_mastering.target_lufs,
                        "limiter_ceiling_db": neutral_mastering.limiter_ceiling_db,
                        "target_low_mid_high": [
                            neutral_mastering.target_low_ratio, neutral_mastering.target_mid_ratio,
                            neutral_mastering.target_high_ratio
                        ],
                        "eq_max_correction_db": neutral_mastering.eq_max_correction_db,
                        "eq_max_high_boost_db": neutral_mastering.eq_max_high_boost_db,
                        "section_leveling_db": neutral_mastering.section_leveling_db
                    },
                    "metrics": arm_metrics(&neutral)
                },
                "fluidsynth_room_only": {
                    "renderer": room_only.name,
                    "sample_rate_hz": room_sr,
                    "wav_path": room_wav_path.display().to_string(),
                    "wav_sha256": room_wav_sha256,
                    "wav_bytes": room_wav.len(),
                    "metrics": arm_metrics(&room_only)
                },
                "fluidsynth_mastering_only": {
                    "renderer": master_only.name,
                    "sample_rate_hz": master_sr,
                    "wav_path": master_wav_path.display().to_string(),
                    "wav_sha256": master_wav_sha256,
                    "wav_bytes": master_wav.len(),
                    "metrics": arm_metrics(&master_only)
                },
                "fluidsynth_dialect_profile": {
                    "renderer": b.name,
                    "sample_rate_hz": fsr,
                    "wav_path": wav_path.display().to_string(),
                    "wav_sha256": wav_sha256,
                    "wav_bytes": wav.len(),
                    "metrics": arm_metrics(&b)
                }
            }
        }));
        deltas.push((format!("{style:?}/{seed}"), d_crest, d_high, d_low));
        profile_deltas.push((
            format!("{style:?}/{seed}"),
            profile_lufs_delta,
            profile_crest_delta,
            profile_high_delta,
            profile_low_delta,
        ));
        println!();
    }

    if deltas.is_empty() {
        eprintln!("No arms completed — nothing measured. Treat as a FAILED run, not a null.");
    } else {
        let n = deltas.len() as f32;
        let mc: f32 = deltas.iter().map(|d| d.1).sum::<f32>() / n;
        let mh: f32 = deltas.iter().map(|d| d.2).sum::<f32>() / n;
        let ml: f32 = deltas.iter().map(|d| d.3).sum::<f32>() / n;
        println!("═══ Summary over {} pieces ═══", deltas.len());
        println!("mean Δcrest      {mc:+.2} dB   (higher = more dynamic range preserved)");
        println!("mean Δhigh-band  {mh:+.3}      (higher = brighter)");
        println!("mean Δlow-band   {ml:+.3}      (higher = more low-end weight)");
        if !profile_deltas.is_empty() {
            let pn = profile_deltas.len() as f32;
            let mean_profile_lufs = profile_deltas.iter().map(|d| d.1).sum::<f32>() / pn;
            let mean_profile_crest = profile_deltas.iter().map(|d| d.2).sum::<f32>() / pn;
            let mean_profile_high = profile_deltas.iter().map(|d| d.3).sum::<f32>() / pn;
            let mean_profile_low = profile_deltas.iter().map(|d| d.4).sum::<f32>() / pn;
            println!(
                "mean profile−default FluidSynth: ΔLUFS {mean_profile_lufs:+.2}, Δcrest {mean_profile_crest:+.2} dB, Δhigh {mean_profile_high:+.3}, Δlow {mean_profile_low:+.3}"
            );
        }
        let consistent_crest = deltas.iter().all(|d| d.1 > 0.0) || deltas.iter().all(|d| d.1 < 0.0);
        println!(
            "Δcrest sign consistent across all pieces: {}",
            if consistent_crest { "YES" } else { "no" }
        );
    }
    println!(
        "\nInterpretation limits, stated: LUFS is expected to match closely BY \
         CONSTRUCTION now that both arms share one mastering target, so a small \
         ΔLUFS is a sanity check on the chain, NOT evidence about renderer quality. \
         Crest and spectral balance are the informative columns. None of these \
         measures perceived quality; they only say whether the arms differ enough \
         to be worth a listening escalation."
    );
    println!("Renders kept in {}", out_dir.display());

    // Complete means every planned style rendered, exact source/build/assets are
    // identified, and any VCSL sample root is accompanied by its caller-computed
    // complete-manifest digest. The SHA is identity evidence, not authenticity.
    let vcsl_provenance_complete = !vcsl_enabled || vcsl_manifest_valid;
    let run_complete = deltas.len() == cohort.len()
        && failures.is_empty()
        && profile_deltas.len() == cohort.len()
        && source_revision_valid
        && vcsl_provenance_complete
        && fluidsynth_version.is_some();
    let manifest = serde_json::json!({
        "schema": "symthaea-muse-renderer-ab-evidence-v1",
        "run_status": if run_complete { "complete" } else { "incomplete" },
        "source_revision": {
            "value": source_revision,
            "format_valid": source_revision_valid,
            "claim_type": "caller-supplied; does not authenticate the source tree or build"
        },
        "build_executable": {
            "path": executable_path.display().to_string(),
            "sha256": executable_sha256,
            "bytes": executable_bytes
        },
        "fluidsynth": {
            "path": bin.display().to_string(),
            "sha256": fluidsynth_sha256,
            "bytes": fluidsynth_bytes,
            "version_output": fluidsynth_version
        },
        "soundfont": {
            "path": sf.display().to_string(),
            "sha256": soundfont_sha256,
            "bytes": soundfont_bytes
        },
        "native_backend": {
            "mode": native_mode,
            "vcsl_enabled": vcsl_enabled,
            "vcsl_root": std::env::var("SYMTHAEA_VCSL_DIR").ok(),
            "declared_manifest_sha256": vcsl_manifest_sha256,
            "manifest_sha256_valid": vcsl_manifest_valid,
            "manifest_required_when_enabled": true
        },
        "contract": {
            "version": "melothaea-style-render-profile-v1",
            "sample_rate_hz_requested": SR,
            "target_lufs": auto_master::MasteringConfig::default().target_lufs,
            "limiter_ceiling_dbfs": auto_master::MasteringConfig::default().limiter_ceiling_db,
            "expected_dialect_count": 8,
            "profile_numbers_are_hypotheses_not_genre_ground_truth": true
        },
        "cohort": {
            "expected": cohort.len(),
            "completed": deltas.len(),
            "record_count_including_failures": rows.len()
        },
        "records": rows,
        "failures": failures,
        "interpretation_limits": [
            "Native PCM hashes canonical f32 samples; FluidSynth WAV hashes include container bytes.",
            "Sample peak is not true peak.",
            "Audio metrics diagnose signal and render differences; they do not establish style authenticity or listener preference.",
            "The four FluidSynth arms use the same MIDI, SoundFont, binary, and sample rate. The 2x2 default/room-only/mastering-only/combined design reports simple effects at the default setting and the room-by-mastering interaction; it does not establish perceptual benefit.",
            "A caller-supplied source revision plus file digests is not a signed build attestation."
        ]
    });
    let manifest_bytes = serde_json::to_vec_pretty(&manifest).expect("serialize evidence manifest");
    std::fs::write(&manifest_path, &manifest_bytes).expect("write evidence manifest");
    println!(
        "Evidence manifest: {} (status={})",
        manifest_path.display(),
        if run_complete {
            "complete"
        } else {
            "incomplete"
        }
    );
    if !run_complete {
        eprintln!(
            "INCOMPLETE evidence: require all {} cohort rows, a 40/64-hex SYMTHAEA_GIT_REVISION, \
             FluidSynth version output, and a valid SYMTHAEA_VCSL_MANIFEST_SHA256 when VCSL is enabled.",
            cohort.len()
        );
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arm(name: &'static str, metrics: [f32; 4]) -> Arm {
        Arm {
            name,
            lufs: metrics[0],
            peak_db: -3.0,
            crest_db: metrics[1],
            low: metrics[2],
            mid: 1.0 - metrics[2] - metrics[3],
            high: metrics[3],
            frames: 1,
            pcm_sha256: String::new(),
        }
    }

    fn assert_metrics_close(actual: [f32; 4], expected: [f32; 4]) {
        for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "metric {index}: got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn factorial_contrasts_recover_room_mastering_and_interaction() {
        let baseline_metrics = [-16.0, 8.0, 0.30, 0.25];
        let room_effect = [0.5, 2.0, 0.02, -0.01];
        let master_effect = [0.4, -0.5, -0.01, 0.03];
        let interaction = [0.2, 0.25, 0.005, -0.004];
        let add = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] { std::array::from_fn(|i| a[i] + b[i]) };
        let baseline = arm("default", baseline_metrics);
        let room = arm("room-only", add(baseline_metrics, room_effect));
        let master = arm("master-only", add(baseline_metrics, master_effect));
        let combined = arm(
            "combined",
            add(
                add(add(baseline_metrics, room_effect), master_effect),
                interaction,
            ),
        );

        let result = factorial_contrasts(&baseline, &room, &master, &combined);
        assert_metrics_close(result.room_effect_at_default_master, room_effect);
        assert_metrics_close(result.master_effect_at_default_room, master_effect);
        assert_metrics_close(result.room_master_interaction, interaction);
    }

    #[test]
    fn factorial_interaction_is_zero_for_additive_effects() {
        let baseline = arm("default", [-16.0, 8.0, 0.30, 0.25]);
        let room = arm("room-only", [-15.5, 9.0, 0.32, 0.24]);
        let master = arm("master-only", [-15.7, 7.5, 0.29, 0.28]);
        let combined = arm("combined", [-15.2, 8.5, 0.31, 0.27]);
        let result = factorial_contrasts(&baseline, &room, &master, &combined);
        assert_metrics_close(result.room_master_interaction, [0.0, 0.0, 0.0, 0.0]);
    }
}
