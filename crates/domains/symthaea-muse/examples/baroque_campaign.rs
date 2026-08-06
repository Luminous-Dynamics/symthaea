// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic multi-seed Baroque harmonic-syntax campaign harness.
//!
//! Supersedes the single-pair `baroque_harmony_pilot_ab.rs` A/B render with
//! a frozen, non-cherry-picked cohort (default 16 seeds) comparing
//! BaroqueSuite's OLD fixed `[1,4,5,1]` compatibility progression against
//! its NEW functional-harmony generator (`ProgressionSpec::Grammar`).
//! Per the campaign design: seed, form, rhythmic material, motif
//! identities, instrumentation, tempo, and render config are held
//! IDENTICAL between the two variants for a given seed -- only the
//! progression system differs, with dependent voices re-realized
//! accordingly (see `symthaea-music-theory/HARMONIC_SYNTAX_REWORK_SCOPE_2026-07-26.md`).
//!
//! For each seed × variant, writes: the score (JSON), the MIDI export, a
//! FluidSynth WAV (best-effort -- degrades to a clear stderr warning, not
//! silent, when `SYMTHAEA_SOUNDFONT`/`SYMTHAEA_FLUIDSYNTH` aren't set), the
//! progression trace (degrees + realized chords), a per-voice note trace,
//! the narrow realized-harmony verifier's report
//! (`symthaea_music_theory::harmony_verifier`), and one shared provenance
//! record for the whole run. A top-level `campaign_summary.json`
//! aggregates the verifier metrics across the full cohort.
//!
//! Real FluidSynth only for the WAVs (per this project's standing rule):
//! run under `nix develop .#muse`, or export `SYMTHAEA_SOUNDFONT`/
//! `SYMTHAEA_FLUIDSYNTH` directly (see this session's notes on resolving
//! them via `nix build --no-link --print-out-paths nixpkgs#fluidsynth
//! nixpkgs#soundfont-fluid` when `nix develop` is unreliable under load).
//!
//! Usage: `cargo run --example baroque_campaign --features theory -- [seeds] [bars]`
//! `seeds`: comma-separated (default `1,2,3,...,16`). `bars`: default `8`.
//! No cherry-picking: every seed in the requested range is composed and
//! stored, whether its verifier report looks good or bad.

use serde::Serialize;
use symthaea_music_theory::harmony_verifier::{self, HarmonyReport};
use symthaea_music_theory::{
    CompositionSpec, Key, MusicalIntent, ProgressionSpec, Score, ScoreNote, Style, VoiceRole,
};

const CAMPAIGN_DIR: &str = "audio_output/baroque_campaign_2026-07-27";

#[derive(Serialize)]
struct Provenance {
    git_commit: String,
    git_dirty: bool,
    generated_at_unix_ms: u128,
    seeds: Vec<u64>,
    bars: usize,
    fluidsynth_path: Option<String>,
    soundfont_path: Option<String>,
    fluidsynth_available: bool,
}

#[derive(Serialize)]
struct VoiceTrace {
    voice: VoiceRole,
    notes: Vec<ScoreNote>,
}

/// The seed's ABSTRACT base progression as `spec.progression()` generates
/// it -- NOT necessarily the piece's full realized bar-by-bar sequence.
/// `Period::parallel_in` (see `harmony_verifier`'s doc comment) reuses this
/// base once per phrase-half with only trailing degrees cadence-forced, so
/// a multi-section `Form` plays it several times over, each copy
/// independently steered. For the piece's ACTUAL realized per-bar harmony
/// (inferred from the notes themselves), see `harmony_report.json`'s
/// `progression_diversity`/`cadences` fields instead.
#[derive(Serialize)]
struct ProgressionTrace {
    seed_base_degrees: Vec<i32>,
    key: Key,
    seed_base_realized_chord_roots_and_qualities: Vec<String>,
}

/// Whether comparing `old_fixed_i_iv_v_i` against `new_functional_walk` is
/// meaningful for a given seed. `Fugue`/`Passacaglia`/etc. bypass
/// `CompositionSpec.progression` entirely (see
/// `FormKind::uses_progression_pipeline`), so for those forms the two
/// variants compose byte-identical output -- naming them "old"/"new" would
/// misleadingly imply a harmonic-syntax intervention that never runs.
#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq)]
enum ProgressionIntervention {
    Applicable,
    NotApplicable { reason: &'static str },
}

#[derive(Serialize)]
struct SeedVariantSummary {
    seed: u64,
    variant: &'static str,
    progression_intervention: ProgressionIntervention,
    report: HarmonyReport,
}

fn git_commit() -> (String, bool) {
    let sha = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(true);
    (sha, dirty)
}

fn unix_time_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn write_json<T: Serialize>(path: &std::path::Path, value: &T) {
    let bytes = serde_json::to_vec_pretty(value).expect("serialize");
    std::fs::write(path, bytes).expect("write json");
}

fn compose_one(
    tag: &'static str,
    spec: &CompositionSpec,
    intent: &MusicalIntent,
    seed_dir: &std::path::Path,
) -> (Score, HarmonyReport) {
    use std::io::Write;
    let variant_dir = seed_dir.join(tag);
    std::fs::create_dir_all(&variant_dir).expect("create variant dir");

    let score = symthaea_music_theory::compose_with_spec(intent, spec);
    let report = harmony_verifier::verify(&score, score.key);

    write_json(&variant_dir.join("score.json"), &score);

    let seed_base_progression = spec.progression(intent.bars.max(1), intent.seed);
    let seed_base_chords = seed_base_progression.chords(score.key);
    let trace = ProgressionTrace {
        seed_base_degrees: seed_base_progression.degrees.clone(),
        key: score.key,
        seed_base_realized_chord_roots_and_qualities: seed_base_chords
            .iter()
            .map(|c| format!("{:?} {:?}", c.root, c.quality))
            .collect(),
    };
    write_json(&variant_dir.join("progression_trace.json"), &trace);

    let voice_traces: Vec<VoiceTrace> = [
        VoiceRole::Melody,
        VoiceRole::Harmony,
        VoiceRole::Bass,
        VoiceRole::CounterMelody,
    ]
    .into_iter()
    .map(|voice| VoiceTrace {
        voice,
        notes: score.voice(voice),
    })
    .collect();
    write_json(&variant_dir.join("voice_trace.json"), &voice_traces);

    write_json(&variant_dir.join("harmony_report.json"), &report);

    let state = symthaea_muse::MusicalState::default();
    let midi_path = variant_dir.join("piece.mid");
    if let Err(error) = symthaea_muse::midi_export::export_performance_midi(
        &score,
        spec,
        intent.seed,
        &state,
        &midi_path,
    ) {
        eprintln!("seed {} [{tag}]: MIDI export failed: {error}", intent.seed);
    } else if let Some(wav) =
        symthaea_muse::fluid_render::render_midi_to_wav(&midi_path, 44100, None)
    {
        std::fs::write(variant_dir.join("piece.wav"), &wav).expect("write wav");
    } else {
        eprintln!(
            "seed {} [{tag}]: FluidSynth unavailable (SYMTHAEA_SOUNDFONT/SYMTHAEA_FLUIDSYNTH not \
             set) -- no piece.wav written for this variant. Run under `nix develop .#muse` or \
             export both env vars for a real campaign run.",
            intent.seed
        );
    }
    std::io::stderr().flush().ok();

    (score, report)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let seeds: Vec<u64> = args
        .next()
        .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .filter(|v: &Vec<u64>| !v.is_empty())
        .unwrap_or_else(|| (1..=16).collect());
    let bars: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);

    // Overridable so a quick post-change "lighter loop" check (render a
    // few seeds, listen, keep/revise/revert) never writes into -- and
    // risks corrupting -- a previously frozen/closed formal campaign's
    // directory. Defaults to the original campaign's path for backward
    // compatibility with any existing invocation.
    let campaign_dir_owned =
        std::env::var("BAROQUE_CAMPAIGN_DIR").unwrap_or_else(|_| CAMPAIGN_DIR.to_string());
    let campaign_dir = std::path::Path::new(&campaign_dir_owned);
    std::fs::create_dir_all(campaign_dir).expect("create campaign dir");

    let new_spec = Style::BaroqueSuite.spec();
    let mut old_spec = new_spec.clone();
    old_spec.progression = ProgressionSpec::Archetype(
        symthaea_music_theory::style::BAROQUE_SUITE_COMPATIBILITY_PROGRESSION.to_vec(),
    );

    let (git_sha, git_dirty) = git_commit();
    let fluidsynth_path = std::env::var("SYMTHAEA_FLUIDSYNTH").ok();
    let soundfont_path = std::env::var("SYMTHAEA_SOUNDFONT").ok();
    let fluidsynth_available = symthaea_muse::fluid_render::available().is_some();
    let provenance = Provenance {
        git_commit: git_sha,
        git_dirty,
        generated_at_unix_ms: unix_time_ms(),
        seeds: seeds.clone(),
        bars,
        fluidsynth_path,
        soundfont_path,
        fluidsynth_available,
    };
    write_json(&campaign_dir.join("provenance.json"), &provenance);
    if !fluidsynth_available {
        eprintln!(
            "WARNING: FluidSynth not configured for this run -- every piece.wav will be \
             skipped. Symbolic artifacts (score/progression/voice-trace/verifier-report) are \
             still written for every seed; re-run with FluidSynth configured before rendering \
             the blinded listening pack."
        );
    }

    let mut summaries: Vec<SeedVariantSummary> = Vec::new();
    for &seed in &seeds {
        let seed_dir = campaign_dir.join(format!("seed_{seed}"));
        let intent = MusicalIntent {
            seed,
            bars,
            ..MusicalIntent::default()
        };
        eprintln!("composing seed {seed}...");
        let form = new_spec.form_kind(seed);
        let intervention = if form.uses_progression_pipeline() {
            ProgressionIntervention::Applicable
        } else {
            ProgressionIntervention::NotApplicable {
                reason: "this seed's form bypasses CompositionSpec.progression entirely \
                         (dedicated engine, e.g. Fugue/Passacaglia) -- old and new variants \
                         compose byte-identical output",
            }
        };
        // Only render the "old" compatibility-progression variant when the
        // intervention it's meant to demonstrate can actually fire for this
        // seed's form -- otherwise it's a redundant, identical render of
        // "new" under a misleading name.
        let variants: &[(&'static str, &CompositionSpec)] = if form.uses_progression_pipeline() {
            &[
                ("old_fixed_i_iv_v_i", &old_spec),
                ("new_functional_walk", &new_spec),
            ]
        } else {
            &[("new_functional_walk", &new_spec)]
        };
        for &(tag, spec) in variants {
            let (_score, report) = compose_one(tag, spec, &intent, &seed_dir);
            summaries.push(SeedVariantSummary {
                seed,
                variant: tag,
                progression_intervention: intervention,
                report,
            });
        }
    }

    write_json(&campaign_dir.join("campaign_summary.json"), &summaries);
    eprintln!(
        "Campaign complete: {} seeds x 2 variants written to {}",
        seeds.len(),
        campaign_dir.display()
    );
}
