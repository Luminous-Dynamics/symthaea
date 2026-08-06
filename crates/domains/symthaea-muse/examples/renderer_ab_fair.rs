// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! The FluidSynth-vs-native A/B, run FAIRLY for the first time.
//!
//! # Why this exists
//!
//! `fluid_render.rs`'s module doc records FluidSynth + FluidR3 beating the
//! in-crate renderer "decisively" ("no longer fights the composition"). That
//! test was real but NOT a like-for-like comparison of renderers: the native arm
//! went through `auto_master` (real BS.1770-4 K-weighting, corrective 3-band EQ,
//! section leveling, brick-wall limiter — wired into `realize_core` on
//! 2026-07-08, `4c613f9d49`) while the FluidSynth arm got only a single peak
//! scan and one gain multiply. FluidSynth won *despite* strictly weaker
//! post-processing.
//!
//! A correction worth recording: an earlier reading of this had the confound
//! backwards, claiming the NATIVE arm was unmastered. History refutes that —
//! `fluid_render.rs` first appeared 2026-07-10 (`c0fda87d0b`), two days after
//! mastering was wired, and the `auto_master` call is verifiably present in
//! `theory_realize.rs` at that commit. The misreading came from taking
//! `theory_realize.rs`'s "never called from ANY generation path before this"
//! comment as describing the state at test time rather than before 2026-07-08.
//!
//! Both paths now share one chain and one `MasteringConfig::default()`, so the
//! comparison finally isolates renderer timbre. This harness runs it.
//!
//! # What it does NOT do
//!
//! It produces no listening pack and asks for no listening. Per the project's
//! standing rule, analytics gate first; clips are escalated only if the numbers
//! justify it. It prints objective per-arm measurements and a verdict on whether
//! the arms are even distinguishable.
//!
//! Renders land in `audio_output/renderer-ab-fair/` relative to the CARGO
//! WORKSPACE root (i.e. `symthaea/audio_output/...`), which `symthaea/.gitignore`
//! already excludes — the WAVs are ~97 MB for the default cohort.
//!
//! # Result of the first run (2026-07-30)
//!
//! Both arms land within 1.1 dB of the −16 LUFS target, confirming the shared
//! chain works on both paths. **Crest factor differs by +6.52 dB in FluidSynth's
//! favour, same sign on 5/5 pieces**: native renders 6.36–8.19 dB crest with
//! peaks at −8.3 to −10.6 dB, FluidSynth 11.77–15.03 dB with peaks near the
//! −1.5 dB ceiling. Equal loudness, far less dynamic range on the native arm —
//! a dense, transient-poor signal, which is the measurable shape of the
//! "instruments sound harsh / fights the composition" complaint. FluidSynth is
//! also slightly brighter (+0.063 high band) with a little less low weight
//! (−0.029).
//!
//! Caveat stated plainly: crest factor is not a quality metric, and higher is
//! not automatically better. Its value here is that an INDEPENDENT objective
//! measure points the same direction as the earlier listening verdict, under
//! equal post-processing. Two independent lines agreeing is the evidence; neither
//! alone would be.
//!
//! ```sh
//! nix shell nixpkgs#fluidsynth nixpkgs#soundfont-fluid --command bash -c '
//!   export SYMTHAEA_FLUIDSYNTH=$(which fluidsynth)
//!   export SYMTHAEA_SOUNDFONT=$(ls /nix/store/*/share/soundfonts/FluidR3_GM2-2.sf2 | head -1)
//!   cargo run --release -p symthaea-muse --features theory --example renderer_ab_fair'
//! ```

use std::path::PathBuf;
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
    }
}

fn wav_to_frames(bytes: &[u8]) -> Option<(Vec<[f32; 2]>, u32)> {
    let mut r = hound::WavReader::new(std::io::Cursor::new(bytes)).ok()?;
    let spec = r.spec();
    let ch = spec.channels.max(1) as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .filter_map(Result::ok)
                .map(|s| s as f32 * scale)
                .collect()
        }
        hound::SampleFormat::Float => r.samples::<f32>().filter_map(Result::ok).collect(),
    };
    let frames = if ch == 1 {
        samples.iter().map(|&s| [s, s]).collect()
    } else {
        samples
            .chunks_exact(ch)
            .map(|c| [c[0], c[1 % ch]])
            .collect()
    };
    Some((frames, spec.sample_rate))
}

fn main() {
    const SR: u32 = 48_000;
    // A fixed, deterministic cohort. Styles chosen to span the instrument
    // families that differ most between a soundfont and additive synthesis:
    // sustained bowed/blown, plucked/struck, and a groove kit.
    let cohort: &[(Style, u64)] = &[
        (Style::Classical, 7),
        (Style::Nocturne, 7),
        (Style::BaroqueSuite, 7),
        (Style::Folk, 3),
        (Style::JazzBallad, 3),
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

    // WHICH native renderer is under test. `vcsl::library()` is a process-global
    // OnceLock, so this cannot be toggled mid-run — the three-way comparison is
    // two invocations of this binary with and without SYMTHAEA_VCSL_DIR, and
    // this banner is what keeps their outputs from being confused for each
    // other. The first run of this harness (2026-07-30) did NOT set the
    // variable, so its "native" arm was pure synthesis while 14.8 GB of real
    // orchestral samples sat unused — a limitation discovered only afterwards,
    // which is exactly why the mode is now printed rather than assumed.
    let native_mode = if symthaea_muse::vcsl::library().is_some() {
        "native+VCSL (real recorded samples where mapped, synthesis elsewhere)"
    } else {
        "native-synthesis (NO samples — SYMTHAEA_VCSL_DIR unset)"
    };
    println!("Native arm: {native_mode}");
    println!(
        "Both arms run auto_master with MasteringConfig::default() \
         (target {} LUFS, ceiling {} dB).\n",
        auto_master::MasteringConfig::default().target_lufs,
        auto_master::MasteringConfig::default().limiter_ceiling_db
    );

    let tag = if symthaea_muse::vcsl::library().is_some() {
        "vcsl"
    } else {
        "synth"
    };
    let out_dir = PathBuf::from(format!("audio_output/renderer-ab-fair-{tag}"));
    std::fs::create_dir_all(&out_dir).expect("create output dir");

    let mut deltas: Vec<(String, f32, f32, f32)> = Vec::new();

    for &(style, seed) in cohort {
        let intent = MusicalIntent {
            seed,
            ..Default::default()
        };
        let score = compose_styled(&intent, style);
        let spec = style.spec();
        let state = MusicalState::default();

        // ── Arm A: native additive/sample renderer, mastered in realize_core.
        let comp =
            symthaea_muse::theory_realize::realize_with_spec(&score, &spec, seed, &state, SR);
        let AudioData::StereoF32(native_frames) = &comp.audio else {
            eprintln!("{style:?}/{seed}: native render was not stereo f32; skipping");
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
        if midi_export::export_performance_midi(&score, &spec, seed, &state, &mid).is_err() {
            eprintln!("{style:?}/{seed}: MIDI export failed; skipping");
            continue;
        }
        let color = fluid_render::room_for_dialect(
            fluid_render::RenderColor::from_state(&state),
            symthaea_muse::theory_realize::dialect_for_spec(&spec),
        );
        let Some(wav) = fluid_render::render_midi_to_wav(&mid, SR, Some(color)) else {
            eprintln!("{style:?}/{seed}: FluidSynth render failed; skipping");
            continue;
        };
        let Some((fluid_frames, fsr)) = wav_to_frames(&wav) else {
            eprintln!("{style:?}/{seed}: could not decode FluidSynth WAV; skipping");
            continue;
        };
        let b = measure("fluidsynth", &fluid_frames, fsr);

        // Keep both renders for a possible later listening escalation.
        std::fs::write(
            out_dir.join(format!("{style:?}_{seed}_fluidsynth.wav")),
            &wav,
        )
        .ok();

        println!("── {style:?} seed={seed} ──");
        for arm in [&a, &b] {
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
            "  Δ(fluid−native): crest {d_crest:+.2} dB   high-band {d_high:+.3}   \
             low-band {d_low:+.3}"
        );
        deltas.push((format!("{style:?}/{seed}"), d_crest, d_high, d_low));
        println!();
    }

    if deltas.is_empty() {
        eprintln!("No arms completed — nothing measured. Treat as a FAILED run, not a null.");
        std::process::exit(1);
    }

    let n = deltas.len() as f32;
    let mc: f32 = deltas.iter().map(|d| d.1).sum::<f32>() / n;
    let mh: f32 = deltas.iter().map(|d| d.2).sum::<f32>() / n;
    let ml: f32 = deltas.iter().map(|d| d.3).sum::<f32>() / n;
    println!("═══ Summary over {} pieces ═══", deltas.len());
    println!("mean Δcrest      {mc:+.2} dB   (higher = more dynamic range preserved)");
    println!("mean Δhigh-band  {mh:+.3}      (higher = brighter)");
    println!("mean Δlow-band   {ml:+.3}      (higher = more low-end weight)");
    let consistent_crest = deltas.iter().all(|d| d.1 > 0.0) || deltas.iter().all(|d| d.1 < 0.0);
    println!(
        "Δcrest sign consistent across all pieces: {}",
        if consistent_crest { "YES" } else { "no" }
    );
    println!(
        "\nInterpretation limits, stated: LUFS is expected to match closely BY \
         CONSTRUCTION now that both arms share one mastering target, so a small \
         ΔLUFS is a sanity check on the chain, NOT evidence about renderer quality. \
         Crest and spectral balance are the informative columns. None of these \
         measures perceived quality; they only say whether the arms differ enough \
         to be worth a listening escalation."
    );
    println!("Renders kept in {}", out_dir.display());
}
