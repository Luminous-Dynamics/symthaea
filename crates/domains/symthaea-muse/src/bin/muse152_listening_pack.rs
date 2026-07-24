// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reproducible blind-listening packs for Muse 152 grammar gates.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use symthaea_music_theory::{
    DrumPolicy, Emphasis, GrammarFamily, GrammarPlanEvidence, Motif, MusicalIntent, Score, Style,
    VoiceRole, compose_with_grammar_plan,
};

const SAMPLE_RATE: u32 = 48_000;
const DEVELOPMENT_SEEDS: [u64; 4] = [11, 28, 45, 62];
const VALIDATION_SEEDS: [u64; 12] = [79, 96, 113, 130, 147, 164, 181, 198, 215, 232, 249, 266];
const STUDY_CLIP_SECS: f32 = 30.0;
const FADE_OUT_SECS: f32 = 1.5;
const CONTROLLED_TARGET_DBFS: f32 = -27.0;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum StudySplit {
    Development,
    Validation,
    LockedHoldout,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum StudyArm {
    Ecological,
    StructuralChallenge,
    MinimalGrammarPair,
    MinimalGrammarPairNatural,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum StudyLength {
    IdentityExcerpt,
    CompleteForm,
}

#[derive(Debug, Clone, Copy)]
struct StudyConfig {
    split: StudySplit,
    arm: StudyArm,
    length: StudyLength,
}

impl Default for StudyConfig {
    fn default() -> Self {
        Self {
            split: StudySplit::Development,
            arm: StudyArm::Ecological,
            length: StudyLength::IdentityExcerpt,
        }
    }
}

impl StudyConfig {
    fn seeds(self) -> &'static [u64] {
        match self.split {
            StudySplit::Development => &DEVELOPMENT_SEEDS,
            StudySplit::Validation => &VALIDATION_SEEDS,
            StudySplit::LockedHoldout => &[],
        }
    }

    fn controlled(self) -> bool {
        matches!(
            self.arm,
            StudyArm::StructuralChallenge
                | StudyArm::MinimalGrammarPair
                | StudyArm::MinimalGrammarPairNatural
        )
    }

    fn shared_motif(self) -> bool {
        matches!(
            self.arm,
            StudyArm::MinimalGrammarPair | StudyArm::MinimalGrammarPairNatural
        )
    }

    fn neutral_performance(self) -> bool {
        matches!(
            self.arm,
            StudyArm::StructuralChallenge | StudyArm::MinimalGrammarPair
        )
    }
}

#[derive(Clone)]
struct Pack {
    id: &'static str,
    styles: &'static [Style],
    melody_only: bool,
    question: &'static str,
}

impl Pack {
    fn label(&self, style: Style) -> String {
        if self.id == "flagship_families" {
            match style {
                Style::Classical => "PeriodSentence".into(),
                Style::AfroCuban => "GrooveCycle".into(),
                Style::Minimalism => "ProcessAdditive".into(),
                // The engine is a generic modal-density arc, not a reviewed
                // implementation of a complete South Asian tradition.
                Style::HindustaniInspired => "ModalArcInformed".into(),
                _ => canonical_style_label(style),
            }
        } else {
            canonical_style_label(style)
        }
    }
}

const FLAGSHIP: [Style; 4] = [
    Style::Classical,
    Style::AfroCuban,
    Style::Minimalism,
    Style::HindustaniInspired,
];
const GROOVE: [Style; 4] = [
    Style::Tango,
    Style::AfroCuban,
    Style::Flamenco,
    Style::BossaNova,
];
const PERIOD_IDENTITY: [Style; 3] = [Style::Tango, Style::Nocturne, Style::March];

fn pack(id: &str) -> Option<Pack> {
    match id {
        "flagship_families" => Some(Pack {
            id: "flagship_families",
            styles: &FLAGSHIP,
            melody_only: false,
            question: "Which compositional grammar family owns this piece?",
        }),
        "groove_cycle" => Some(Pack {
            id: "groove_cycle",
            styles: &GROOVE,
            melody_only: false,
            question: "Which cycle language owns this piece?",
        }),
        "period_identity" => Some(Pack {
            id: "period_identity",
            styles: &PERIOD_IDENTITY,
            melody_only: true,
            question: "Which canonical style does this lead melody express?",
        }),
        _ => None,
    }
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    schema_version: u32,
    protocol_version: String,
    pack_id: String,
    generated_at_unix_ms: u64,
    theory_engine_version: String,
    question: String,
    labels: Vec<String>,
    seeds_per_label: usize,
    seeds: Vec<u64>,
    chance_probability: f64,
    melody_only: bool,
    drums_removed: bool,
    shared_instrument: String,
    shuffle_seed: u64,
    #[serde(default)]
    clip_duration_secs: Option<f32>,
    #[serde(default)]
    study_split: Option<StudySplit>,
    #[serde(default)]
    study_arm: Option<StudyArm>,
    #[serde(default)]
    study_length: Option<StudyLength>,
    #[serde(default)]
    controls: Vec<String>,
    #[serde(default)]
    generation_commit: Option<String>,
    #[serde(default)]
    generation_worktree_dirty: Option<bool>,
    #[serde(default)]
    renderer_version: Option<String>,
    #[serde(default)]
    soundfont_sha256: Option<String>,
    #[serde(default)]
    pairing_design: Option<String>,
    inclusion_policy: String,
}

#[derive(Serialize)]
struct ClipAudit {
    clip: String,
    seed: u64,
    accepted: bool,
    rejection_reason: Option<String>,
    source_bars: usize,
    symbolic_duration_secs: f64,
    note_count: usize,
    onset_density_per_sec: f64,
    median_midi: u8,
}

#[derive(Debug, Serialize)]
struct ArtifactRecord {
    stable_clip_id: String,
    filename: String,
    wav_sha256: String,
    score_sha256: String,
    normalized_score_sha256: String,
    recipe_sha256: String,
    seed: u64,
    exact_wav_duration_secs: f64,
    symbolic_duration_secs: f64,
    excerpt_start_secs: f64,
    grammar_profile: serde_json::Value,
    performance_dialect: symthaea_music_theory::PerformanceDialect,
    source_bars: usize,
    inclusion: &'static str,
    curated: bool,
    rejection_history: Vec<String>,
    block_id: Option<String>,
    premise_id: Option<String>,
    motif_id: Option<String>,
    pairing_intentional: bool,
    observed_controls: ObservedControls,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct ObservedControls {
    symbolic_tempo_bpm: f32,
    onset_density_per_sec: f64,
    register_min_midi: u8,
    register_median_midi: u8,
    register_max_midi: u8,
    velocity_mean: f64,
    velocity_std_dev: f64,
    wav_rms_dbfs: f64,
    low_activity_frame_proportion: f64,
}

struct SurfaceObservation {
    audio_sha256: String,
    family: String,
    group_id: String,
    features: [f64; 7],
}

#[derive(Deserialize)]
struct StoredArtifactForNuisance {
    wav_sha256: String,
    seed: u64,
    block_id: Option<String>,
    observed_controls: ObservedControls,
}

#[derive(Debug, Serialize)]
struct NuisancePrediction {
    audio_sha256: String,
    held_out_group: String,
    truth: String,
    predicted: String,
    correct: bool,
}

#[derive(Debug, Serialize)]
struct NuisanceBaseline {
    method: &'static str,
    evaluation_unit: &'static str,
    group_count: usize,
    features: [&'static str; 7],
    correct: usize,
    total: usize,
    accuracy: f64,
    cluster_bootstrap_accuracy_95: [f64; 2],
    exact_within_group_permutation_p_value: f64,
    feature_group_accuracy: BTreeMap<&'static str, f64>,
    unavailable_feature_groups: Vec<&'static str>,
    pairwise_accuracy: BTreeMap<String, f64>,
    predictions: Vec<NuisancePrediction>,
}

#[derive(Debug, Serialize)]
struct PairedTrial {
    block_id: String,
    premise_id: String,
    motif_id: String,
    clips: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct StructuralTruth {
    family: String,
    grammar_profile: serde_json::Value,
    structural_plan: serde_json::Value,
    phrase_start_positions: Vec<f64>,
    cadence_positions: Vec<f64>,
    climax_position: Option<f64>,
    recurrence_intervals: Vec<f64>,
    literal_or_transposed_motif_occurrences: Vec<f64>,
    density_arc: Vec<f64>,
    harmonic_pitch_class_trajectory: Vec<f64>,
    declared_development_operations: Vec<String>,
    #[serde(default)]
    composer_assertion_trace: serde_json::Value,
}

#[derive(Deserialize)]
struct StoredArtifactForAugment {
    filename: String,
    wav_sha256: String,
    seed: u64,
    source_bars: usize,
    motif_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct CloneWarning {
    family: String,
    clip_a: String,
    clip_b: String,
    onset_trajectory_correlation: f64,
    chroma_trajectory_similarity: f64,
    cadence_position_similarity: f64,
    climax_position_delta: Option<f64>,
    reasons: Vec<&'static str>,
}

struct AnalyzedClip {
    family: String,
    filename: String,
    onset_trajectory: Vec<f64>,
    chroma_trajectory: Vec<f64>,
    cadence_positions: Vec<f64>,
    climax_position: Option<f64>,
}

fn canonical_style_label(style: Style) -> String {
    serde_json::to_value(style)
        .ok()
        .and_then(|value| value.as_str().map(str::to_owned))
        .unwrap_or_else(|| format!("{style:?}"))
}

fn shuffle<T>(items: &mut [T], mut state: u64) {
    for i in (1..items.len()).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        items.swap(i, ((state >> 33) as usize) % (i + 1));
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("generate") => {
            let pack_id = args
                .get(2)
                .map(String::as_str)
                .unwrap_or("flagship_families");
            let out = args
                .get(3)
                .cloned()
                .unwrap_or_else(|| format!("audio_output/muse152_{pack_id}"));
            generate(
                pack(pack_id).unwrap_or_else(|| panic!("unknown pack {pack_id}")),
                Path::new(&out),
            );
        }
        Some("study") => {
            let split = match args.get(2).map(String::as_str).unwrap_or("development") {
                "development" => StudySplit::Development,
                "validation" => StudySplit::Validation,
                "holdout" => StudySplit::LockedHoldout,
                other => panic!("unknown study split {other}"),
            };
            if split == StudySplit::LockedHoldout {
                eprintln!(
                    "holdout generation is locked: commit an external seed-file hash and add an explicit unlock workflow before the final test"
                );
                std::process::exit(3);
            }
            let arm = match args.get(3).map(String::as_str).unwrap_or("ecological") {
                "ecological" => StudyArm::Ecological,
                "structural" => StudyArm::StructuralChallenge,
                "minimal-pairs" => StudyArm::MinimalGrammarPair,
                "minimal-pairs-natural" => StudyArm::MinimalGrammarPairNatural,
                other => panic!("unknown study arm {other}"),
            };
            let length = match args.get(4).map(String::as_str).unwrap_or("identity") {
                "identity" => StudyLength::IdentityExcerpt,
                "complete" => StudyLength::CompleteForm,
                other => panic!("unknown study length {other}"),
            };
            let out = args.get(5).cloned().unwrap_or_else(|| {
                format!("audio_output/muse152_{split:?}_{arm:?}_{length:?}").to_lowercase()
            });
            generate_configured(
                pack("flagship_families").unwrap(),
                Path::new(&out),
                StudyConfig { split, arm, length },
            );
        }
        Some("score") => {
            let dir = args.get(2).expect("score <dir> <guesses>");
            let guesses = args.get(3).expect("score <dir> <guesses>");
            score(Path::new(dir), guesses);
        }
        Some("reanalyze-nuisance") => {
            let dir = args.get(2).expect("reanalyze-nuisance <pack-dir>");
            reanalyze_nuisance(Path::new(dir));
        }
        Some("augment-evidence") => {
            let dir = args.get(2).expect("augment-evidence <pack-dir>");
            augment_evidence(Path::new(dir));
        }
        _ => {
            eprintln!(
                "usage: muse152_listening_pack generate <pack> [out_dir] | study <development|validation|holdout> <ecological|structural|minimal-pairs|minimal-pairs-natural> <identity|complete> [out_dir] | score <dir> <guesses> | reanalyze-nuisance <pack-dir> | augment-evidence <pack-dir>"
            );
            std::process::exit(2);
        }
    }
}

fn augment_evidence(out: &Path) {
    let sealed = out.join("sealed");
    let artifacts: Vec<StoredArtifactForAugment> = serde_json::from_slice(
        &std::fs::read(sealed.join("artifacts.json")).expect("read artifact manifest"),
    )
    .expect("parse artifact manifest");
    let answers: BTreeMap<String, String> = serde_json::from_slice(
        &std::fs::read(sealed.join("answer_key_by_sha256.json")).expect("read answer key"),
    )
    .expect("parse answer key");
    let mut truths: BTreeMap<String, StructuralTruth> = serde_json::from_slice(
        &std::fs::read(sealed.join("structural_truth_by_sha256.json"))
            .expect("read structural truth"),
    )
    .expect("parse structural truth");
    let mut integrity = BTreeMap::new();
    for artifact in artifacts {
        let family = answers
            .get(&artifact.wav_sha256)
            .unwrap_or_else(|| panic!("missing answer for {}", artifact.wav_sha256));
        let style = match family.as_str() {
            "PeriodSentence" => Style::Classical,
            "GrooveCycle" => Style::AfroCuban,
            "ProcessAdditive" => Style::Minimalism,
            "ModalArcInformed" => Style::HindustaniInspired,
            other => panic!("unsupported flagship family {other}"),
        };
        let mut spec = style.spec();
        spec.texture.drums = DrumPolicy::None;
        spec.ensemble_pool = vec![["piano".into(), "piano".into(), "piano".into()]];
        if artifact.motif_id.is_some() {
            install_shared_motif(&mut spec, &shared_motif(artifact.seed), artifact.seed);
        }
        let intent = MusicalIntent {
            seed: artifact.seed,
            bars: artifact.source_bars,
            valence: 0.0,
            arousal: 0.5,
            energy: 0.5,
            ..MusicalIntent::default()
        };
        let realized = compose_with_grammar_plan(style.grammar_profile(), &intent, &spec);
        truths
            .get_mut(&artifact.wav_sha256)
            .unwrap_or_else(|| panic!("missing truth for {}", artifact.wav_sha256))
            .composer_assertion_trace = serde_json::to_value(realized.trace).unwrap();

        let wav = std::fs::read(out.join(&artifact.filename)).expect("read study WAV");
        assert_eq!(sha256_bytes(&wav), artifact.wav_sha256);
        integrity.insert(
            artifact.wav_sha256.clone(),
            symthaea_muse::analyst::analyze_audio_integrity(&wav, &artifact.wav_sha256, 0),
        );
    }
    std::fs::write(
        sealed.join("structural_truth_with_composer_trace_v2.json"),
        serde_json::to_vec_pretty(&truths).unwrap(),
    )
    .expect("write augmented structural truth");
    std::fs::write(
        sealed.join("audio_integrity_by_sha256.json"),
        serde_json::to_vec_pretty(&integrity).unwrap(),
    )
    .expect("write audio integrity evidence");
    println!(
        "{}: augmented {} immutable artifacts",
        out.display(),
        integrity.len()
    );
}

fn reanalyze_nuisance(out: &Path) {
    let sealed = out.join("sealed");
    let artifacts: Vec<StoredArtifactForNuisance> = serde_json::from_slice(
        &std::fs::read(sealed.join("artifacts.json")).expect("read artifact manifest"),
    )
    .expect("parse artifact manifest");
    let answer_key: BTreeMap<String, String> = serde_json::from_slice(
        &std::fs::read(sealed.join("answer_key_by_sha256.json")).expect("read answer key"),
    )
    .expect("parse answer key");
    let observations: Vec<_> = artifacts
        .into_iter()
        .map(|artifact| SurfaceObservation {
            family: answer_key
                .get(&artifact.wav_sha256)
                .unwrap_or_else(|| panic!("missing answer for {}", artifact.wav_sha256))
                .clone(),
            group_id: artifact
                .block_id
                .unwrap_or_else(|| format!("seed-{:016x}", artifact.seed)),
            audio_sha256: artifact.wav_sha256,
            features: [
                artifact.observed_controls.symbolic_tempo_bpm as f64,
                artifact.observed_controls.onset_density_per_sec,
                artifact.observed_controls.register_median_midi as f64,
                artifact.observed_controls.velocity_mean,
                artifact.observed_controls.velocity_std_dev,
                artifact.observed_controls.wav_rms_dbfs,
                artifact.observed_controls.low_activity_frame_proportion,
            ],
        })
        .collect();
    let report = nuisance_baseline(&observations);
    let path = sealed.join("nuisance_baseline_block_safe_v2.json");
    std::fs::write(&path, serde_json::to_vec_pretty(&report).unwrap())
        .expect("write block-safe nuisance report");
    println!(
        "{}: {}/{} ({:.1}%), p={:.6}, block-bootstrap 95% [{:.1}%, {:.1}%]",
        path.display(),
        report.correct,
        report.total,
        report.accuracy * 100.0,
        report.exact_within_group_permutation_p_value,
        report.cluster_bootstrap_accuracy_95[0] * 100.0,
        report.cluster_bootstrap_accuracy_95[1] * 100.0,
    );
}

fn generate(pack: Pack, out: &Path) {
    generate_configured(pack, out, StudyConfig::default());
}

fn generate_configured(pack: Pack, out: &Path, config: StudyConfig) {
    std::fs::create_dir_all(out).expect("create pack directory");
    let sealed = out.join("sealed");
    std::fs::create_dir_all(&sealed).expect("create sealed evidence directory");
    let shuffle_seed = 0x4D55_5345_3135_3201;
    let mut clips = Vec::new();
    for &style in pack.styles {
        for &seed in config.seeds() {
            clips.push((style, seed));
        }
    }
    shuffle(&mut clips, shuffle_seed);
    let manifest = Manifest {
        schema_version: 3,
        protocol_version: "muse-grammar-evidence-1.1".into(),
        pack_id: pack.id.into(),
        generated_at_unix_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64,
        theory_engine_version: symthaea_music_theory::MUSIC_THEORY_ENGINE_VERSION.into(),
        question: pack.question.into(),
        labels: pack
            .styles
            .iter()
            .copied()
            .map(|style| pack.label(style))
            .collect(),
        seeds_per_label: config.seeds().len(),
        seeds: config.seeds().to_vec(),
        chance_probability: 1.0 / pack.styles.len() as f64,
        melody_only: pack.melody_only || config.controlled(),
        drums_removed: true,
        shared_instrument: "piano".into(),
        shuffle_seed,
        clip_duration_secs: if config.length == StudyLength::IdentityExcerpt {
            Some(STUDY_CLIP_SECS)
        } else {
            None
        },
        study_split: Some(config.split),
        study_arm: Some(config.arm),
        study_length: Some(config.length),
        controls: match config.arm {
            StudyArm::Ecological => vec![
                "common piano rendering path".into(),
                "drums removed".into(),
                "natural grammar tempo, density, register, dynamics and dialect retained".into(),
            ],
            StudyArm::StructuralChallenge => vec![
                "common piano rendering path".into(),
                "drums removed".into(),
                "melody-only projection".into(),
                "shared slow/moderate/fast tempo strata (72/96/120 BPM)".into(),
                "median register octave-normalized near MIDI 64".into(),
                "velocity range normalized while preserving accents".into(),
                "final study waveform RMS normalized to -27 dBFS".into(),
                "common process-exact performance dialect".into(),
                "note density measured and reported, not forced".into(),
            ],
            StudyArm::MinimalGrammarPair => vec![
                "same seed-indexed literal motif bank supplied to every grammar".into(),
                "style hook graft disabled".into(),
                "common piano rendering path".into(),
                "drums removed".into(),
                "melody-only projection".into(),
                "shared slow/moderate/fast tempo strata (72/96/120 BPM)".into(),
                "median register octave-normalized near MIDI 64".into(),
                "velocity and waveform RMS normalized".into(),
                "common process-exact performance dialect".into(),
            ],
            StudyArm::MinimalGrammarPairNatural => vec![
                "same seed-indexed literal motif bank supplied to every grammar".into(),
                "style hook graft disabled".into(),
                "common piano rendering path and no drums".into(),
                "shared slow/moderate/fast tempo strata (72/96/120 BPM)".into(),
                "median register, velocity range and waveform RMS normalized".into(),
                "family-native performance dialect retained".into(),
            ],
        },
        generation_commit: generation_commit(),
        generation_worktree_dirty: generation_worktree_dirty(),
        renderer_version: renderer_version(),
        soundfont_sha256: soundfont_sha256(),
        pairing_design: config
            .shared_motif()
            .then(|| "intentional_same_motif_four_grammar_block".into()),
        inclusion_policy:
            "all preregistered seeds included; mechanical failures recorded; no aesthetic curation"
                .into(),
    };
    std::fs::write(
        out.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .expect("write manifest");

    let mut answer_key = BTreeMap::new();
    let mut answer_key_by_hash = BTreeMap::new();
    let mut audit = Vec::new();
    let mut artifacts = Vec::new();
    let mut analyzed = Vec::new();
    let mut paired_clips: BTreeMap<u64, Vec<String>> = BTreeMap::new();
    let mut structural_truth_by_hash = BTreeMap::new();
    let mut audio_integrity_by_hash = BTreeMap::new();
    let mut surface_observations = Vec::new();
    for (index, (style, seed)) in clips.into_iter().enumerate() {
        let mut spec = style.spec();
        spec.texture.drums = DrumPolicy::None;
        spec.ensemble_pool = vec![["piano".into(), "piano".into(), "piano".into()]];
        let shared_motif = config.shared_motif().then(|| shared_motif(seed));
        if let Some(motif) = &shared_motif {
            install_shared_motif(&mut spec, motif, seed);
        }
        let initial_bars = if style == Style::Minimalism { 16 } else { 8 };
        let intent = MusicalIntent {
            seed,
            valence: 0.0,
            arousal: 0.5,
            energy: 0.5,
            // The additive engine's cycles are intentionally compact; give
            // it enough source material to fill the same study window as the
            // longer-form families before all clips are cut to equal length.
            bars: initial_bars,
            ..MusicalIntent::default()
        };
        let controlled_tempo = config.controlled().then(|| tempo_stratum(seed));
        let (mut score, source_bars, structural_plan, composer_trace) =
            compose_for_study(style, &intent, &spec, config.length, controlled_tempo);
        if pack.melody_only || config.controlled() {
            score.notes.retain(|note| note.role == VoiceRole::Melody);
        }
        if config.controlled() {
            normalize_structural_surface(&mut score, controlled_tempo.unwrap());
        }
        let symbolic_duration = score.seconds();
        let note_count = score.notes.len();
        let median_midi = median_midi(&score);
        let name = format!("clip_{:02}.wav", index + 1);
        let dialect = if config.neutral_performance() {
            symthaea_music_theory::PerformanceDialect::ProcessExact
        } else {
            style.grammar_profile().performance
        };
        let render_score = if config.length == StudyLength::IdentityExcerpt {
            render_excerpt_score(&score, STUDY_CLIP_SECS + FADE_OUT_SECS)
        } else {
            score.clone()
        };
        let wav = render(&render_score, &spec, seed, dialect)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let mut wav = if config.length == StudyLength::IdentityExcerpt {
            study_excerpt(wav, STUDY_CLIP_SECS, FADE_OUT_SECS)
                .unwrap_or_else(|error| panic!("{name}: {error}"))
        } else {
            wav
        };
        if config.controlled() {
            wav = normalize_wav_rms(wav, CONTROLLED_TARGET_DBFS)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
        }
        let family = pack.label(style);
        let wav_sha256 = sha256_bytes(&wav);
        audio_integrity_by_hash.insert(
            wav_sha256.clone(),
            symthaea_muse::analyst::analyze_audio_integrity(&wav, &wav_sha256, 0),
        );
        let score_sha256 = sha256_json(&score);
        let normalized_score_sha256 = normalized_score_hash(&score);
        let recipe_sha256 = sha256_json(&(
            &intent,
            &spec,
            style.grammar_profile(),
            config.arm,
            config.length,
        ));
        let exact_wav_duration_secs = wav_duration_secs(&wav).unwrap_or_default();
        let stable_clip_id = format!("muse-{}", &wav_sha256[..16]);
        let motif_id = shared_motif
            .as_ref()
            .map(|motif| format!("motif-{}", &sha256_json(motif)[..16]));
        let block_id = config
            .shared_motif()
            .then(|| format!("block-seed-{seed:016x}"));
        let premise_id = block_id.as_ref().map(|block| format!("premise-{block}"));
        let observed_controls = observed_controls(&score, &wav);
        surface_observations.push(SurfaceObservation {
            audio_sha256: wav_sha256.clone(),
            family: family.clone(),
            group_id: block_id
                .clone()
                .unwrap_or_else(|| format!("seed-{seed:016x}")),
            features: [
                observed_controls.symbolic_tempo_bpm as f64,
                observed_controls.onset_density_per_sec,
                observed_controls.register_median_midi as f64,
                observed_controls.velocity_mean,
                observed_controls.velocity_std_dev,
                observed_controls.wav_rms_dbfs,
                observed_controls.low_activity_frame_proportion,
            ],
        });
        std::fs::write(out.join(&name), &wav).expect("write clip");
        answer_key.insert(name.clone(), family.clone());
        answer_key_by_hash.insert(wav_sha256.clone(), family.clone());
        artifacts.push(ArtifactRecord {
            stable_clip_id,
            filename: name.clone(),
            wav_sha256: wav_sha256.clone(),
            score_sha256,
            normalized_score_sha256,
            recipe_sha256,
            seed,
            exact_wav_duration_secs,
            symbolic_duration_secs: symbolic_duration,
            excerpt_start_secs: 0.0,
            grammar_profile: serde_json::to_value(style.grammar_profile()).unwrap(),
            performance_dialect: dialect,
            source_bars,
            inclusion: "automatic_preregistered_seed",
            curated: false,
            rejection_history: Vec::new(),
            block_id: block_id.clone(),
            premise_id: premise_id.clone(),
            motif_id: motif_id.clone(),
            pairing_intentional: config.shared_motif(),
            observed_controls,
        });
        if config.shared_motif() {
            paired_clips.entry(seed).or_default().push(name.clone());
        }
        structural_truth_by_hash.insert(
            wav_sha256.clone(),
            structural_truth(
                &score,
                family.clone(),
                style,
                &structural_plan,
                shared_motif.as_ref(),
                &composer_trace,
            ),
        );
        analyzed.push(analyze_score(&score, family, name.clone()));
        audit.push(ClipAudit {
            clip: name,
            seed,
            accepted: true,
            rejection_reason: None,
            source_bars,
            symbolic_duration_secs: symbolic_duration,
            note_count,
            onset_density_per_sec: note_count as f64 / symbolic_duration.max(1e-9),
            median_midi,
        });
    }
    std::fs::write(
        sealed.join("answer_key.json"),
        serde_json::to_vec_pretty(&answer_key).unwrap(),
    )
    .expect("write answer key");
    std::fs::write(
        sealed.join("answer_key_by_sha256.json"),
        serde_json::to_vec_pretty(&answer_key_by_hash).unwrap(),
    )
    .expect("write hash-addressed answer key");
    std::fs::write(
        sealed.join("artifacts.json"),
        serde_json::to_vec_pretty(&artifacts).unwrap(),
    )
    .expect("write artifact manifest");
    std::fs::write(
        sealed.join("clone_warnings.json"),
        serde_json::to_vec_pretty(&clone_warnings(&analyzed)).unwrap(),
    )
    .expect("write within-family clone warnings");
    std::fs::write(
        sealed.join("structural_truth_by_sha256.json"),
        serde_json::to_vec_pretty(&structural_truth_by_hash).unwrap(),
    )
    .expect("write sealed symbolic truth");
    std::fs::write(
        sealed.join("audio_integrity_by_sha256.json"),
        serde_json::to_vec_pretty(&audio_integrity_by_hash).unwrap(),
    )
    .expect("write sealed audio integrity evidence");
    std::fs::write(
        sealed.join("nuisance_baseline.json"),
        serde_json::to_vec_pretty(&nuisance_baseline(&surface_observations)).unwrap(),
    )
    .expect("write nuisance-only baseline");
    if config.shared_motif() {
        let trials: Vec<_> = paired_clips
            .into_iter()
            .map(|(seed, clips)| {
                let motif_id = format!("motif-{}", &sha256_json(&shared_motif(seed))[..16]);
                let block_id = format!("block-seed-{seed:016x}");
                PairedTrial {
                    premise_id: format!("premise-{block_id}"),
                    block_id,
                    motif_id,
                    clips,
                }
            })
            .collect();
        std::fs::write(
            out.join("paired_trials.json"),
            serde_json::to_vec_pretty(&trials).unwrap(),
        )
        .expect("write paired trials");
        write_paired_participant_page(out).expect("write paired participant page");
    }
    std::fs::write(
        out.join("rejection_ledger.json"),
        serde_json::to_vec_pretty(&audit).unwrap(),
    )
    .expect("write rejection ledger");
    write_participant_page(out, answer_key.len()).expect("write participant study page");
    let mut readme = std::fs::File::create(out.join("README.md")).expect("write README");
    writeln!(
        readme,
        "# {}\n\n{}\n\nStudy split: `{:?}`. Arm: `{:?}`. Length: `{:?}`. Every pre-registered seed is included; see `rejection_ledger.json`. Chance is {:.0}%. Keep `sealed/` unavailable to participants until responses are frozen.\n\nScore with:\n\n```bash\ncargo run -p symthaea-muse --features studio --bin muse152_listening_pack -- score {} \"1=label,2=label,...\"\n```",
        manifest.pack_id,
        manifest.question,
        config.split,
        config.arm,
        config.length,
        manifest.chance_probability * 100.0,
        out.display()
    )
    .unwrap();
    println!(
        "wrote {} blinded clips to {}",
        answer_key.len(),
        out.display()
    );
}

fn shared_motif(seed: u64) -> Motif {
    Style::Classical.spec().motif(0.5, seed)
}

fn install_shared_motif(
    spec: &mut symthaea_music_theory::CompositionSpec,
    source: &Motif,
    seed: u64,
) {
    assert!(
        spec.install_primary_motif_for_seed(source, seed),
        "shared study motif must fit the target grammar's metric cycle"
    );
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_json<T: Serialize>(value: &T) -> String {
    sha256_bytes(&serde_json::to_vec(value).expect("hashable study value must serialize"))
}

fn sha256_file(path: &Path) -> Option<String> {
    use std::io::Read as _;
    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).ok()?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Some(format!("{:x}", hasher.finalize()))
}

fn generation_commit() -> Option<String> {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_owned())
}

fn generation_worktree_dirty() -> Option<bool> {
    std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| !output.stdout.is_empty())
}

fn renderer_version() -> Option<String> {
    let Some((binary, _)) = symthaea_muse::fluid_render::available() else {
        return Some(format!("symthaea-native-{}", env!("CARGO_PKG_VERSION")));
    };
    std::process::Command::new(binary)
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| {
            String::from_utf8(if output.stdout.is_empty() {
                output.stderr
            } else {
                output.stdout
            })
            .ok()
        })
        .and_then(|output| output.lines().next().map(str::to_owned))
}

fn soundfont_sha256() -> Option<String> {
    let (_, soundfont) = symthaea_muse::fluid_render::available()?;
    sha256_file(&soundfont)
}

fn wav_duration_secs(wav: &[u8]) -> Result<f64, String> {
    let reader = hound::WavReader::new(std::io::Cursor::new(wav))
        .map_err(|error| format!("read WAV duration: {error}"))?;
    Ok(reader.duration() as f64 / reader.spec().sample_rate as f64)
}

fn observed_controls(score: &Score, wav: &[u8]) -> ObservedControls {
    let mut pitches: Vec<u8> = score.notes.iter().map(|note| note.pitch.midi()).collect();
    pitches.sort_unstable();
    let velocities: Vec<f64> = score
        .notes
        .iter()
        .map(|note| note.velocity as f64)
        .collect();
    let velocity_mean = velocities.iter().sum::<f64>() / velocities.len().max(1) as f64;
    let velocity_std_dev = (velocities
        .iter()
        .map(|value| (value - velocity_mean).powi(2))
        .sum::<f64>()
        / velocities.len().max(1) as f64)
        .sqrt();

    let mut wav_rms_dbfs = f64::NEG_INFINITY;
    let mut low_activity_frame_proportion = 1.0;
    if let Ok(mut reader) = hound::WavReader::new(std::io::Cursor::new(wav)) {
        let spec = reader.spec();
        if spec.sample_format == hound::SampleFormat::Int && spec.bits_per_sample <= 16 {
            if let Ok(samples) = reader.samples::<i16>().collect::<Result<Vec<_>, _>>() {
                let normalized: Vec<f64> = samples
                    .iter()
                    .map(|sample| *sample as f64 / i16::MAX as f64)
                    .collect();
                let rms = (normalized.iter().map(|value| value * value).sum::<f64>()
                    / normalized.len().max(1) as f64)
                    .sqrt();
                wav_rms_dbfs = 20.0 * rms.max(1e-12).log10();
                let frame_samples = (spec.sample_rate as usize / 20)
                    .saturating_mul(spec.channels as usize)
                    .max(1);
                let frames: Vec<f64> = normalized
                    .chunks(frame_samples)
                    .map(|frame| {
                        (frame.iter().map(|value| value * value).sum::<f64>()
                            / frame.len().max(1) as f64)
                            .sqrt()
                    })
                    .collect();
                low_activity_frame_proportion = frames
                    .iter()
                    .filter(|rms| **rms < 10_f64.powf(-45.0 / 20.0))
                    .count() as f64
                    / frames.len().max(1) as f64;
            }
        }
    }

    ObservedControls {
        symbolic_tempo_bpm: score.tempo_bpm,
        onset_density_per_sec: score.notes.len() as f64 / score.seconds().max(1e-9),
        register_min_midi: pitches.first().copied().unwrap_or(0),
        register_median_midi: pitches.get(pitches.len() / 2).copied().unwrap_or(0),
        register_max_midi: pitches.last().copied().unwrap_or(0),
        velocity_mean,
        velocity_std_dev,
        wav_rms_dbfs,
        low_activity_frame_proportion,
    }
}

fn motif_occurrences(score: &Score, motif: &Motif) -> Vec<f64> {
    let melody = score.voice(VoiceRole::Melody);
    let motif_notes: Vec<_> = motif
        .notes
        .iter()
        .filter(|note| note.degree.is_some())
        .collect();
    if motif_notes.len() < 2 || melody.len() < motif_notes.len() {
        return Vec::new();
    }
    let motif_contour: Vec<i8> = motif_notes
        .windows(2)
        .map(|pair| (pair[1].degree.unwrap() - pair[0].degree.unwrap()).signum() as i8)
        .collect();
    let motif_total: f64 = motif_notes.iter().map(|note| note.duration.beats()).sum();
    let motif_rhythm: Vec<f64> = motif_notes
        .iter()
        .map(|note| note.duration.beats() / motif_total.max(1e-9))
        .collect();
    let total = score.total_beats.beats().max(1e-9);
    melody
        .windows(motif_notes.len())
        .filter_map(|window| {
            let contour: Vec<i8> = window
                .windows(2)
                .map(|pair| {
                    (pair[1].pitch.midi() as i32 - pair[0].pitch.midi() as i32).signum() as i8
                })
                .collect();
            let window_total: f64 = window.iter().map(|note| note.duration.beats()).sum();
            let rhythm_matches = window.iter().zip(&motif_rhythm).all(|(note, expected)| {
                (note.duration.beats() / window_total.max(1e-9) - expected).abs() <= 0.08
            });
            (contour == motif_contour && rhythm_matches).then(|| window[0].onset.beats() / total)
        })
        .collect()
}

fn structural_truth(
    score: &Score,
    family: String,
    style: Style,
    plan: &GrammarPlanEvidence,
    motif: Option<&Motif>,
    composer_trace: &symthaea_music_theory::GrammarStructuralTrace,
) -> StructuralTruth {
    let total = score.total_beats.beats().max(1e-9);
    let positions = |emphasis| {
        score
            .notes
            .iter()
            .filter(|note| note.role == VoiceRole::Melody && note.emphasis == emphasis)
            .map(|note| note.onset.beats() / total)
            .collect::<Vec<_>>()
    };
    let phrase_start_positions = positions(Emphasis::PhraseStart);
    let cadence_positions = positions(Emphasis::Cadential);
    let climax_position = positions(Emphasis::Climax).into_iter().next();
    let recurrence_intervals = phrase_start_positions
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect();
    let analysis = analyze_score(score, family.clone(), "sealed".into());
    let declared_development_operations = match style.grammar_profile().family {
        GrammarFamily::PeriodSentence => vec![
            "antecedent_consequent".into(),
            "cadential_answer".into(),
            "formal_return".into(),
        ],
        GrammarFamily::GrooveCycle => vec![
            "whole_cycle_repetition".into(),
            "cycle_section_layering".into(),
            "absolute_phase_return".into(),
        ],
        GrammarFamily::ProcessAdditive => vec![
            "prefix_addition".into(),
            "process_trajectory".into(),
            "liquidation_or_reset".into(),
        ],
        GrammarFamily::RagaModalArc => vec![
            "augmented_exposition".into(),
            "pulse_entry".into(),
            "diminished_intensification".into(),
        ],
        other => vec![format!("{other:?}")],
    };
    StructuralTruth {
        family,
        grammar_profile: serde_json::to_value(style.grammar_profile()).unwrap(),
        structural_plan: serde_json::to_value(plan).unwrap(),
        phrase_start_positions,
        cadence_positions,
        climax_position,
        recurrence_intervals,
        literal_or_transposed_motif_occurrences: motif
            .map(|motif| motif_occurrences(score, motif))
            .unwrap_or_default(),
        density_arc: analysis.onset_trajectory,
        harmonic_pitch_class_trajectory: analysis.chroma_trajectory,
        declared_development_operations,
        composer_assertion_trace: serde_json::to_value(composer_trace).unwrap(),
    }
}

fn nuisance_baseline(observations: &[SurfaceObservation]) -> NuisanceBaseline {
    const FEATURE_NAMES: [&str; 7] = [
        "symbolic_tempo_bpm",
        "onset_density_per_sec",
        "register_median_midi",
        "velocity_mean",
        "velocity_std_dev",
        "wav_rms_dbfs",
        "low_activity_frame_proportion",
    ];
    let families: std::collections::BTreeSet<_> = observations
        .iter()
        .map(|item| item.family.clone())
        .collect();
    let groups: std::collections::BTreeSet<_> = observations
        .iter()
        .map(|item| item.group_id.clone())
        .collect();
    let predict = |held_out: usize, candidates: &[&String], axes: &[usize]| {
        let held_out_group = &observations[held_out].group_id;
        let training: Vec<_> = observations
            .iter()
            .enumerate()
            .filter(|(_, item)| {
                &item.group_id != held_out_group && candidates.contains(&&item.family)
            })
            .collect();
        let means: [f64; 7] = std::array::from_fn(|axis| {
            training
                .iter()
                .map(|(_, item)| item.features[axis])
                .sum::<f64>()
                / training.len().max(1) as f64
        });
        let deviations: [f64; 7] = std::array::from_fn(|axis| {
            (training
                .iter()
                .map(|(_, item)| (item.features[axis] - means[axis]).powi(2))
                .sum::<f64>()
                / training.len().max(1) as f64)
                .sqrt()
                .max(1e-9)
        });
        let standardized = |features: &[f64; 7]| -> [f64; 7] {
            std::array::from_fn(|axis| (features[axis] - means[axis]) / deviations[axis])
        };
        let target = standardized(&observations[held_out].features);
        candidates
            .iter()
            .filter_map(|family| {
                let members: Vec<_> = training
                    .iter()
                    .filter(|(_, item)| &item.family == *family)
                    .map(|(_, item)| standardized(&item.features))
                    .collect();
                if members.is_empty() {
                    return None;
                }
                let centroid: [f64; 7] = std::array::from_fn(|axis| {
                    members.iter().map(|item| item[axis]).sum::<f64>() / members.len() as f64
                });
                let distance = axes
                    .iter()
                    .map(|axis| (target[*axis] - centroid[*axis]).powi(2))
                    .sum::<f64>();
                Some((distance, (*family).clone()))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, family)| family)
            .unwrap_or_default()
    };
    let family_refs: Vec<_> = families.iter().collect();
    const ALL_AXES: [usize; 7] = [0, 1, 2, 3, 4, 5, 6];
    let mut predictions = Vec::new();
    for (held_out, observation) in observations.iter().enumerate() {
        let predicted = predict(held_out, &family_refs, &ALL_AXES);
        predictions.push(NuisancePrediction {
            audio_sha256: observation.audio_sha256.clone(),
            held_out_group: observation.group_id.clone(),
            truth: observation.family.clone(),
            correct: predicted == observation.family,
            predicted,
        });
    }
    let correct = predictions.iter().filter(|item| item.correct).count();
    let family_list: Vec<_> = families.iter().cloned().collect();
    let mut pairwise_accuracy = BTreeMap::new();
    for left in 0..family_list.len() {
        for right in (left + 1)..family_list.len() {
            let pair = [&family_list[left], &family_list[right]];
            let held_out_indices: Vec<_> = observations
                .iter()
                .enumerate()
                .filter(|(_, item)| pair.contains(&&item.family))
                .map(|(index, _)| index)
                .collect();
            let candidate_refs = vec![pair[0], pair[1]];
            let pair_correct = held_out_indices
                .iter()
                .filter(|&&held_out| {
                    predict(held_out, &candidate_refs, &ALL_AXES) == observations[held_out].family
                })
                .count();
            pairwise_accuracy.insert(
                format!("{}__{}", pair[0], pair[1]),
                pair_correct as f64 / held_out_indices.len().max(1) as f64,
            );
        }
    }
    let group_accuracies: Vec<f64> = groups
        .iter()
        .map(|group| {
            let members: Vec<_> = predictions
                .iter()
                .filter(|item| &item.held_out_group == group)
                .collect();
            members.iter().filter(|item| item.correct).count() as f64 / members.len().max(1) as f64
        })
        .collect();
    let mut bootstrap = Vec::new();
    fn cluster_bootstrap(values: &[f64], depth: usize, sum: f64, output: &mut Vec<f64>) {
        if depth == values.len() {
            output.push(sum / values.len().max(1) as f64);
            return;
        }
        for value in values {
            cluster_bootstrap(values, depth + 1, sum + value, output);
        }
    }
    cluster_bootstrap(&group_accuracies, 0, 0.0, &mut bootstrap);
    bootstrap.sort_by(f64::total_cmp);
    let quantile = |probability: f64| {
        bootstrap[((bootstrap.len().saturating_sub(1)) as f64 * probability).round() as usize]
    };

    // Exact null for a four-way matched design: independently permute the
    // four true family labels inside each held-out block while predictions
    // remain fixed.
    fn permutations(items: &mut [String], start: usize, output: &mut Vec<Vec<String>>) {
        if start == items.len() {
            output.push(items.to_vec());
            return;
        }
        for index in start..items.len() {
            items.swap(start, index);
            permutations(items, start + 1, output);
            items.swap(start, index);
        }
    }
    let mut null_distribution = vec![(0usize, 1usize)];
    for group in &groups {
        let group_predictions: Vec<_> = predictions
            .iter()
            .filter(|item| &item.held_out_group == group)
            .map(|item| item.predicted.clone())
            .collect();
        let mut labels: Vec<_> = predictions
            .iter()
            .filter(|item| &item.held_out_group == group)
            .map(|item| item.truth.clone())
            .collect();
        let mut assignments = Vec::new();
        permutations(&mut labels, 0, &mut assignments);
        let mut match_counts = BTreeMap::<usize, usize>::new();
        for assignment in assignments {
            let matches = assignment
                .iter()
                .zip(&group_predictions)
                .filter(|(truth, predicted)| truth == predicted)
                .count();
            *match_counts.entry(matches).or_default() += 1;
        }
        let mut next = BTreeMap::<usize, usize>::new();
        for (prior_matches, prior_count) in null_distribution {
            for (&matches, &count) in &match_counts {
                *next.entry(prior_matches + matches).or_default() += prior_count * count;
            }
        }
        null_distribution = next.into_iter().collect();
    }
    let null_total: usize = null_distribution.iter().map(|(_, count)| count).sum();
    let null_extreme: usize = null_distribution
        .iter()
        .filter(|(matches, _)| *matches >= correct)
        .map(|(_, count)| count)
        .sum();
    let feature_groups: [(&str, &[usize]); 4] = [
        ("explicitly_controlled_nuisance", &[0, 2, 5]),
        ("legitimate_realized_grammar_consequence", &[1, 6]),
        ("symbolic_dynamic_consequence", &[3, 4]),
        ("all_surface_features", &ALL_AXES),
    ];
    let feature_group_accuracy = feature_groups
        .into_iter()
        .map(|(name, axes)| {
            let correct = observations
                .iter()
                .enumerate()
                .filter(|(held_out, item)| predict(*held_out, &family_refs, axes) == item.family)
                .count();
            (name, correct as f64 / observations.len().max(1) as f64)
        })
        .collect();
    NuisanceBaseline {
        method: "leave_one_premise_motif_block_out_nearest_family_centroid_training_zscore",
        evaluation_unit: "premise_id_and_motif_id_block",
        group_count: groups.len(),
        features: FEATURE_NAMES,
        correct,
        total: predictions.len(),
        accuracy: correct as f64 / predictions.len().max(1) as f64,
        cluster_bootstrap_accuracy_95: [quantile(0.025), quantile(0.975)],
        exact_within_group_permutation_p_value: null_extreme as f64 / null_total.max(1) as f64,
        feature_group_accuracy,
        unavailable_feature_groups: vec![
            "performance_dialect_consequence",
            "unknown_or_confounded",
        ],
        pairwise_accuracy,
        predictions,
    }
}

fn normalized_score_hash(score: &Score) -> String {
    let events = score.events();
    let anchor = events.first().map(|note| note.pitch.midi()).unwrap_or(60) as i32;
    let total = score.total_beats.beats().max(1e-9);
    let normalized: Vec<_> = events
        .iter()
        .map(|note| {
            (
                note.pitch.midi() as i32 - anchor,
                (note.onset.beats() / total * 1_000_000.0).round() as i64,
                (note.duration.beats() / total * 1_000_000.0).round() as i64,
                note.role,
                note.emphasis,
            )
        })
        .collect();
    sha256_json(&normalized)
}

fn analyze_score(score: &Score, family: String, filename: String) -> AnalyzedClip {
    const TIME_BINS: usize = 32;
    let total = score.total_beats.beats().max(1e-9);
    let mut onset_trajectory = vec![0.0; TIME_BINS];
    let mut chroma_trajectory = vec![0.0; TIME_BINS * 12];
    let mut cadence_positions = Vec::new();
    let mut climax_position = None;
    for note in &score.notes {
        let position = (note.onset.beats() / total).clamp(0.0, 0.999_999);
        let bin = (position * TIME_BINS as f64) as usize;
        onset_trajectory[bin] += 1.0;
        chroma_trajectory[bin * 12 + note.pitch.pitch_class().value() as usize] += 1.0;
        if note.emphasis == Emphasis::Cadential {
            cadence_positions.push(position);
        }
        if note.emphasis == Emphasis::Climax {
            climax_position = Some(position);
        }
    }
    let note_count = score.notes.len().max(1) as f64;
    onset_trajectory
        .iter_mut()
        .for_each(|value| *value /= note_count);
    for bin in chroma_trajectory.chunks_mut(12) {
        let sum: f64 = bin.iter().sum();
        if sum > 0.0 {
            bin.iter_mut().for_each(|value| *value /= sum);
        }
    }
    AnalyzedClip {
        family,
        filename,
        onset_trajectory,
        chroma_trajectory,
        cadence_positions,
        climax_position,
    }
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mean_a = a.iter().sum::<f64>() / a.len() as f64;
    let mean_b = b.iter().sum::<f64>() / b.len() as f64;
    let mut numerator = 0.0;
    let mut denominator_a = 0.0;
    let mut denominator_b = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        numerator += (x - mean_a) * (y - mean_b);
        denominator_a += (x - mean_a).powi(2);
        denominator_b += (y - mean_b).powi(2);
    }
    let denominator = (denominator_a * denominator_b).sqrt();
    if denominator > 1e-12 {
        numerator / denominator
    } else if a == b {
        1.0
    } else {
        0.0
    }
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a > 1e-12 && norm_b > 1e-12 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

fn cadence_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let directed = |left: &[f64], right: &[f64]| {
        left.iter()
            .map(|x| right.iter().map(|y| (x - y).abs()).fold(1.0, f64::min))
            .sum::<f64>()
            / left.len() as f64
    };
    (1.0 - 0.5 * (directed(a, b) + directed(b, a))).clamp(0.0, 1.0)
}

fn clone_warnings(clips: &[AnalyzedClip]) -> Vec<CloneWarning> {
    let mut warnings = Vec::new();
    for i in 0..clips.len() {
        for j in (i + 1)..clips.len() {
            let (a, b) = (&clips[i], &clips[j]);
            if a.family != b.family {
                continue;
            }
            let onset = pearson(&a.onset_trajectory, &b.onset_trajectory);
            let chroma = cosine(&a.chroma_trajectory, &b.chroma_trajectory);
            let cadence = cadence_similarity(&a.cadence_positions, &b.cadence_positions);
            let climax_delta = a
                .climax_position
                .zip(b.climax_position)
                .map(|(x, y)| (x - y).abs());
            let mut reasons = Vec::new();
            if onset >= 0.95 {
                reasons.push("onset_trajectory_correlation>=0.95");
            }
            if chroma >= 0.95 {
                reasons.push("chroma_trajectory_similarity>=0.95");
            }
            if !a.cadence_positions.is_empty() && !b.cadence_positions.is_empty() && cadence >= 0.98
            {
                reasons.push("cadence_position_similarity>=0.98");
            }
            if climax_delta.is_some_and(|delta| delta <= 0.02) {
                reasons.push("climax_position_delta<=0.02");
            }
            if !reasons.is_empty() {
                warnings.push(CloneWarning {
                    family: a.family.clone(),
                    clip_a: a.filename.clone(),
                    clip_b: b.filename.clone(),
                    onset_trajectory_correlation: onset,
                    chroma_trajectory_similarity: chroma,
                    cadence_position_similarity: cadence,
                    climax_position_delta: climax_delta,
                    reasons,
                });
            }
        }
    }
    warnings
}

fn write_participant_page(out: &Path, clip_count: usize) -> std::io::Result<()> {
    let page = format!(
        r#"<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Muse grammar study</title>
<style>body{{font:16px system-ui;max-width:860px;margin:2rem auto;padding:0 1rem;background:#111;color:#eee}}.card{{background:#1d1d1d;padding:1rem;margin:1rem 0;border-radius:10px}}label{{display:block;margin:.65rem 0}}select,input,textarea,button{{font:inherit}}textarea{{width:100%;min-height:4rem}}audio{{width:100%}}button{{padding:.7rem 1rem}}.short .complete{{display:none}}</style>
<h1>Muse grammar listening study</h1>
<div class="card"><label>Participant ID <input id="pid" required></label><label>Cohort <select id="cohort"><option value="untrained">Untrained listener</option><option value="trained">Trained musician</option></select></label><button id="start">Start / reshuffle</button></div>
<main id="trials"></main><button id="export" hidden>Export anonymized responses</button>
<script>
const N={clip_count};let manifest,order=[];
const hash=s=>[...s].reduce((h,c)=>Math.imul(h^c.charCodeAt(0),16777619)>>>0,2166136261);
const rng=s=>()=>((s=Math.imul(s,1664525)+1013904223>>>0)/4294967296);
const shuffled=(a,seed)=>{{const r=rng(seed);for(let i=a.length-1;i;i--){{const j=Math.floor(r()*(i+1));[a[i],a[j]]=[a[j],a[i]]}}return a}};
async function begin(){{manifest=await fetch('manifest.json').then(r=>r.json());document.body.classList.toggle('short',manifest.study_length!=='complete_form');const pid=document.querySelector('#pid').value.trim();if(!pid)return;order=shuffled(Array.from({{length:N}},(_,i)=>i+1),hash(pid+'|'+manifest.shuffle_seed));const root=document.querySelector('#trials');root.innerHTML='';order.forEach((n,pos)=>{{const el=document.createElement('section');el.className='card';el.dataset.clip=n;el.innerHTML=`<h2>Excerpt ${{pos+1}} of ${{N}}</h2><audio controls preload="none" src="clip_${{String(n).padStart(2,'0')}}.wav"></audio><label>Family <select class="family"><option value="">Choose…</option>${{manifest.labels.map(x=>`<option>${{x}}</option>`).join('')}}</select></label><label>Confidence (0–100) <input class="confidence" type="range" min="0" max="100" value="50"></label><label>What evidence did you hear? <textarea class="evidence" placeholder="Form, rhythm, harmony, texture, register, dynamics…"></textarea></label><label>Distinctiveness (1–7) <input class="distinct" type="number" min="1" max="7"></label><label>Coherence (1–7) <input class="coherence" type="number" min="1" max="7"></label><label>Musical quality (1–7) <input class="quality" type="number" min="1" max="7"></label><label>Desire to replay (1–7) <input class="replay" type="number" min="1" max="7"></label><label>Too similar to an earlier piece? <select class="similar"><option value="">Unsure</option><option value="no">No</option><option value="yes">Yes</option></select></label><label class="complete">Development feels intentional (1–7) <input class="development" type="number" min="1" max="7"></label><label class="complete">Repetition gains meaning (1–7) <input class="repetition" type="number" min="1" max="7"></label><label class="complete">Climax feels earned (1–7) <input class="climax" type="number" min="1" max="7"></label><label class="complete">Returns are recognizable (1–7) <input class="returns" type="number" min="1" max="7"></label><label class="complete">Ending concludes rather than stops (1–7) <input class="ending" type="number" min="1" max="7"></label><label class="complete">Sustains attention (1–7) <input class="attention" type="number" min="1" max="7"></label>`;root.appendChild(el)}});document.querySelector('#export').hidden=false}}
document.querySelector('#start').onclick=begin;
document.querySelector('#export').onclick=()=>{{const num=(el,q)=>Number(el.querySelector(q).value)||null;const response={{schema_version:2,participant_id:document.querySelector('#pid').value.trim(),cohort:document.querySelector('#cohort').value,pack_id:manifest.pack_id,study_split:manifest.study_split,study_arm:manifest.study_arm,study_length:manifest.study_length,presentation_order:order,responses:[...document.querySelectorAll('section')].map((el,index)=>({{presentation_index:index+1,clip:`clip_${{String(el.dataset.clip).padStart(2,'0')}}.wav`,family:el.querySelector('.family').value,confidence:Number(el.querySelector('.confidence').value),heard_evidence:el.querySelector('.evidence').value,distinctiveness:num(el,'.distinct'),coherence:num(el,'.coherence'),musical_quality:num(el,'.quality'),desire_to_replay:num(el,'.replay'),too_similar_to_earlier:el.querySelector('.similar').value||null,intentional_development:num(el,'.development'),meaningful_repetition:num(el,'.repetition'),earned_climax:num(el,'.climax'),recognizable_returns:num(el,'.returns'),conclusive_ending:num(el,'.ending'),sustained_attention:num(el,'.attention')}}))}};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(response,null,2)],{{type:'application/json'}}));a.download=`muse-study-${{hash(response.participant_id)}}.json`;a.click()}};
</script></html>"#
    );
    std::fs::write(out.join("study.html"), page)
}

fn write_paired_participant_page(out: &Path) -> std::io::Result<()> {
    let page = r#"<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Muse paired grammar study</title>
<style>body{font:16px system-ui;max-width:980px;margin:2rem auto;padding:0 1rem;background:#111;color:#eee}.card,.treatment{background:#1d1d1d;padding:1rem;margin:1rem 0;border-radius:10px}.treatment{background:#282828}label{display:block;margin:.6rem 0}audio{width:100%}select,input,button{font:inherit}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:.7rem}button{padding:.7rem 1rem}</style>
<h1>Matched four-way grammar study</h1><p>Each block uses the same source motif and controlled premise. Assign each treatment, then compare structure and musical value.</p>
<div class="card"><label>Participant ID <input id="pid"></label><label>Cohort <select id="cohort"><option value="untrained">Untrained listener</option><option value="trained">Trained musician</option></select></label><button id="start">Begin / reshuffle</button></div><main id="blocks"></main><button id="export" hidden>Export responses</button>
<script>
let manifest,trials,presented=[];const hash=s=>[...s].reduce((h,c)=>Math.imul(h^c.charCodeAt(0),16777619)>>>0,2166136261);const rng=s=>()=>((s=Math.imul(s,1664525)+1013904223>>>0)/4294967296);const shuffle=(a,s)=>{const r=rng(s);for(let i=a.length-1;i;i--){const j=Math.floor(r()*(i+1));[a[i],a[j]]=[a[j],a[i]]}return a};
const rating=(name,label)=>`<label>${label} (1–7) <input class="${name}" type="number" min="1" max="7"></label>`;
async function begin(){[manifest,trials]=await Promise.all([fetch('manifest.json').then(r=>r.json()),fetch('paired_trials.json').then(r=>r.json())]);const pid=document.querySelector('#pid').value.trim();if(!pid)return;presented=shuffle([...trials],hash(pid+'blocks')).map((block,index)=>({...block,clips:shuffle([...block.clips],hash(pid+block.block_id)),index}));const root=document.querySelector('#blocks');root.innerHTML='';presented.forEach((block,bi)=>{const section=document.createElement('section');section.className='card block';section.dataset.block=block.block_id;const treatments=block.clips.map((clip,i)=>`<div class="treatment" data-clip="${clip}"><h3>Treatment ${String.fromCharCode(65+i)}</h3><audio controls preload="none" src="${clip}"></audio><label>Grammar <select class="family"><option value="">Choose…</option>${manifest.labels.map(x=>`<option>${x}</option>`).join('')}</select></label><label>Confidence 0–100 <input class="confidence" type="range" min="0" max="100" value="50"></label>${rating('coherence','Coherence')}${rating('interest','Musical interest')}${rating('beauty','Beauty')}${rating('memorability','Memorability')}${rating('replay','Desire to replay')}</div>`).join('');const choices=block.clips.map((_,i)=>`<option value="${i}">Treatment ${String.fromCharCode(65+i)}</option>`).join('');section.innerHTML=`<h2>Matched block ${bi+1} of ${presented.length}</h2><div class="grid">${treatments}</div><fieldset><legend>Evidence used</legend>${['groove_or_pulse','phrase_endings','repetition_pattern','harmonic_movement','density','silence','melodic_development','dynamics','register','unsure'].map(x=>`<label><input type="checkbox" class="evidence" value="${x}"> ${x.replaceAll('_',' ')}</label>`).join('')}</fieldset><label>Most structurally similar pair <select class="similar_a"><option value="">A…</option>${choices}</select> <select class="similar_b"><option value="">B…</option>${choices}</select></label><label>Best preserves the source motif <select class="preserves"><option value="">Choose…</option>${choices}</select></label><label>Most musically convincing <select class="convincing"><option value="">Choose…</option>${choices}</select></label>`;root.appendChild(section)});document.querySelector('#export').hidden=false}
document.querySelector('#start').onclick=begin;document.querySelector('#export').onclick=()=>{const num=(el,q)=>Number(el.querySelector(q).value)||null;const blocks=[...document.querySelectorAll('.block')].map((block,bi)=>({block_id:block.dataset.block,presentation_index:bi+1,treatments:[...block.querySelectorAll('.treatment')].map((el,i)=>({presentation_label:String.fromCharCode(65+i),clip:el.dataset.clip,family:el.querySelector('.family').value,confidence:Number(el.querySelector('.confidence').value),coherence:num(el,'.coherence'),musical_interest:num(el,'.interest'),beauty:num(el,'.beauty'),memorability:num(el,'.memorability'),desire_to_replay:num(el,'.replay')})),evidence_used:[...block.querySelectorAll('.evidence:checked')].map(x=>x.value),most_similar_pair:[block.querySelector('.similar_a').value,block.querySelector('.similar_b').value],best_motif_preservation:block.querySelector('.preserves').value,most_musically_convincing:block.querySelector('.convincing').value}));if(blocks.some(block=>{const labels=block.treatments.map(x=>x.family);return labels.some(x=>!x)||new Set(labels).size!==manifest.labels.length})){alert('Assign each grammar exactly once in every block.');return}const data={schema_version:1,design:'matched_four_way',participant_id:document.querySelector('#pid').value.trim(),cohort:document.querySelector('#cohort').value,study_arm:manifest.study_arm,blocks};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download=`muse-paired-${hash(data.participant_id)}.json`;a.click()};
</script></html>"#;
    std::fs::write(out.join("paired_study.html"), page)
}

fn compose_for_study(
    style: Style,
    intent: &MusicalIntent,
    spec: &symthaea_music_theory::CompositionSpec,
    length: StudyLength,
    controlled_tempo: Option<f32>,
) -> (
    symthaea_music_theory::Score,
    usize,
    GrammarPlanEvidence,
    symthaea_music_theory::GrammarStructuralTrace,
) {
    if length == StudyLength::IdentityExcerpt {
        let realized = compose_with_grammar_plan(style.grammar_profile(), intent, spec);
        return (realized.score, intent.bars, realized.plan, realized.trace);
    }

    // Mechanical, pre-declared selection: use the complete form closest to
    // 120 seconds within [90, 180]. Listening never enters this choice.
    const BAR_CANDIDATES: [usize; 11] = [4, 6, 8, 12, 16, 24, 32, 40, 48, 56, 64];
    let mut best: Option<(
        f64,
        symthaea_music_theory::Score,
        usize,
        GrammarPlanEvidence,
        symthaea_music_theory::GrammarStructuralTrace,
    )> = None;
    let mut observed = Vec::new();
    for bars in BAR_CANDIDATES {
        let mut candidate_intent = intent.clone();
        candidate_intent.bars = bars;
        let realized = compose_with_grammar_plan(style.grammar_profile(), &candidate_intent, spec);
        let mut score = realized.score;
        if let Some(tempo) = controlled_tempo {
            score.tempo_bpm = tempo;
        }
        let seconds = score.seconds();
        observed.push((bars, seconds));
        if (90.0..=180.0).contains(&seconds) {
            let distance = (seconds - 120.0).abs();
            if best
                .as_ref()
                .is_none_or(|(prior, _, _, _, _)| distance < *prior)
            {
                best = Some((distance, score, bars, realized.plan, realized.trace));
            }
        }
    }
    best.map(|(_, score, bars, plan, trace)| (score, bars, plan, trace))
        .unwrap_or_else(|| {
            panic!(
                "{style:?} has no mechanically eligible 90-180 second complete form; observed {observed:?}"
            )
        })
}

fn median_midi(score: &symthaea_music_theory::Score) -> u8 {
    let mut pitches: Vec<u8> = score.notes.iter().map(|note| note.pitch.midi()).collect();
    pitches.sort_unstable();
    pitches.get(pitches.len() / 2).copied().unwrap_or(64)
}

fn render_excerpt_score(score: &Score, seconds: f32) -> Score {
    let limit_beats = f64::from(seconds) * f64::from(score.tempo_bpm) / 60.0;
    let limit = symthaea_music_theory::Duration::new((limit_beats * 960.0).ceil() as i64, 960);
    let mut excerpt = score.clone();
    excerpt
        .notes
        .retain(|note| note.onset.beats() < limit_beats);
    for note in &mut excerpt.notes {
        if (note.onset + note.duration).beats() > limit_beats {
            note.duration = limit.saturating_sub(note.onset);
        }
    }
    excerpt.total_beats = limit;
    excerpt
}

fn tempo_stratum(seed: u64) -> f32 {
    [72.0, 96.0, 120.0][seed as usize % 3]
}

fn normalize_structural_surface(score: &mut symthaea_music_theory::Score, tempo_bpm: f32) {
    score.tempo_bpm = tempo_bpm;
    let median = median_midi(score) as i32;
    let octave_shift = ((64 - median) as f32 / 12.0).round() as i32 * 12;
    for note in &mut score.notes {
        note.pitch = note.pitch.transpose(octave_shift);
    }

    let (min, max) = score
        .notes
        .iter()
        .fold((1.0_f32, 0.0_f32), |(lo, hi), note| {
            (lo.min(note.velocity), hi.max(note.velocity))
        });
    let span = (max - min).max(1e-6);
    for note in &mut score.notes {
        // Preserve accents but give every family the same dynamic range.
        note.velocity = 0.52 + 0.24 * (note.velocity - min) / span;
    }
}

fn normalize_wav_rms(wav: Vec<u8>, target_dbfs: f32) -> Result<Vec<u8>, String> {
    let mut reader = hound::WavReader::new(std::io::Cursor::new(wav))
        .map_err(|error| format!("read WAV for normalization: {error}"))?;
    let spec = reader.spec();
    if spec.sample_format == hound::SampleFormat::Float {
        let samples: Vec<f32> = reader
            .samples::<f32>()
            .collect::<Result<_, _>>()
            .map_err(|error| format!("decode float WAV for normalization: {error}"))?;
        let rms = (samples
            .iter()
            .map(|sample| f64::from(*sample).powi(2))
            .sum::<f64>()
            / samples.len().max(1) as f64)
            .sqrt();
        let peak = samples
            .iter()
            .map(|sample| f64::from(sample.abs()))
            .fold(0.0_f64, f64::max);
        let requested = if rms > 1e-9 {
            10_f64.powf(target_dbfs as f64 / 20.0) / rms
        } else {
            1.0
        };
        let gain = requested.min(if peak > 1e-9 { 0.98 / peak } else { 1.0 });
        let mut output = std::io::Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut output, spec)
                .map_err(|error| format!("create normalized float WAV: {error}"))?;
            for sample in samples {
                writer
                    .write_sample((f64::from(sample) * gain) as f32)
                    .map_err(|error| format!("write normalized float WAV: {error}"))?;
            }
            writer
                .finalize()
                .map_err(|error| format!("finalize normalized float WAV: {error}"))?;
        }
        return Ok(output.into_inner());
    }
    if spec.sample_format != hound::SampleFormat::Int || spec.bits_per_sample > 16 {
        return Err(format!("unsupported normalization WAV format: {spec:?}"));
    }
    let samples: Vec<i16> = reader
        .samples::<i16>()
        .collect::<Result<_, _>>()
        .map_err(|error| format!("decode WAV for normalization: {error}"))?;
    let rms = (samples
        .iter()
        .map(|sample| (*sample as f64 / i16::MAX as f64).powi(2))
        .sum::<f64>()
        / samples.len().max(1) as f64)
        .sqrt();
    let peak = samples
        .iter()
        .map(|sample| sample.unsigned_abs() as f64 / i16::MAX as f64)
        .fold(0.0_f64, f64::max);
    let requested = if rms > 1e-9 {
        10_f64.powf(target_dbfs as f64 / 20.0) / rms
    } else {
        1.0
    };
    let gain = requested.min(if peak > 1e-9 { 0.98 / peak } else { 1.0 });
    let mut output = std::io::Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut output, spec)
            .map_err(|error| format!("create normalized WAV: {error}"))?;
        for sample in samples {
            writer
                .write_sample((sample as f64 * gain).round() as i16)
                .map_err(|error| format!("write normalized WAV: {error}"))?;
        }
        writer
            .finalize()
            .map_err(|error| format!("finalize normalized WAV: {error}"))?;
    }
    Ok(output.into_inner())
}

fn study_excerpt(wav: Vec<u8>, seconds: f32, fade_seconds: f32) -> Result<Vec<u8>, String> {
    let mut reader = hound::WavReader::new(std::io::Cursor::new(wav))
        .map_err(|error| format!("read rendered WAV: {error}"))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let target_frames = (seconds * spec.sample_rate as f32).round() as usize;
    let target_samples = target_frames * channels;
    let fade_frames = (fade_seconds * spec.sample_rate as f32).round() as usize;
    let fade_start = target_frames.saturating_sub(fade_frames);
    let mut output = std::io::Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut output, spec)
            .map_err(|error| format!("create study WAV: {error}"))?;
        match spec.sample_format {
            hound::SampleFormat::Int if spec.bits_per_sample <= 16 => {
                let mut samples: Vec<i16> = reader
                    .samples::<i16>()
                    .collect::<Result<_, _>>()
                    .map_err(|error| format!("decode rendered WAV: {error}"))?;
                samples.resize(target_samples, 0);
                samples.truncate(target_samples);
                for (index, sample) in samples.into_iter().enumerate() {
                    let frame = index / channels;
                    let gain = if frame >= fade_start && fade_frames > 0 {
                        (target_frames - frame) as f32 / fade_frames as f32
                    } else {
                        1.0
                    };
                    writer
                        .write_sample((sample as f32 * gain) as i16)
                        .map_err(|error| format!("write study WAV: {error}"))?;
                }
            }
            hound::SampleFormat::Float => {
                let mut samples: Vec<f32> = reader
                    .samples::<f32>()
                    .collect::<Result<_, _>>()
                    .map_err(|error| format!("decode rendered WAV: {error}"))?;
                samples.resize(target_samples, 0.0);
                samples.truncate(target_samples);
                for (index, sample) in samples.into_iter().enumerate() {
                    let frame = index / channels;
                    let gain = if frame >= fade_start && fade_frames > 0 {
                        (target_frames - frame) as f32 / fade_frames as f32
                    } else {
                        1.0
                    };
                    writer
                        .write_sample(sample * gain)
                        .map_err(|error| format!("write study WAV: {error}"))?;
                }
            }
            _ => return Err(format!("unsupported WAV format: {spec:?}")),
        }
        writer
            .finalize()
            .map_err(|error| format!("finalize study WAV: {error}"))?;
    }
    Ok(output.into_inner())
}

fn render(
    score: &symthaea_music_theory::Score,
    spec: &symthaea_music_theory::CompositionSpec,
    seed: u64,
    dialect: symthaea_music_theory::PerformanceDialect,
) -> Result<Vec<u8>, String> {
    if symthaea_muse::fluid_render::available().is_none() {
        let composition = symthaea_muse::theory_realize::realize_with_spec_and_dialect(
            score,
            spec,
            seed,
            &symthaea_muse::MusicalState::default(),
            SAMPLE_RATE,
            dialect,
        );
        return symthaea_muse::export::wav_bytes(&composition);
    }
    let path = std::env::temp_dir().join(format!(
        "muse152_listening_{}_{}.mid",
        std::process::id(),
        seed
    ));
    symthaea_muse::midi_export::export_performance_midi_with_dialect(
        score,
        spec,
        seed,
        &symthaea_muse::MusicalState::default(),
        dialect,
        &path,
    )
    .map_err(|error| error.to_string())?;
    let wav = symthaea_muse::fluid_render::render_midi_to_wav(&path, SAMPLE_RATE, None)
        .ok_or_else(|| "FluidSynth render failed".to_owned());
    let _ = std::fs::remove_file(path);
    wav
}

fn score(dir: &Path, guesses: &str) {
    let manifest: Manifest = serde_json::from_str(
        &std::fs::read_to_string(dir.join("manifest.json")).expect("read manifest"),
    )
    .expect("parse manifest");
    let answer_path = if dir.join("sealed/answer_key.json").is_file() {
        dir.join("sealed/answer_key.json")
    } else {
        dir.join("answer_key.json")
    };
    let key: BTreeMap<String, String> =
        serde_json::from_str(&std::fs::read_to_string(answer_path).expect("read answer key"))
            .expect("parse answer key");
    let mut correct = 0usize;
    let mut total = 0usize;
    for item in guesses.split(',') {
        let Some((number, guess)) = item.split_once('=') else {
            continue;
        };
        let Ok(number) = number.trim().parse::<usize>() else {
            continue;
        };
        let name = format!("clip_{number:02}.wav");
        let Some(truth) = key.get(&name) else {
            continue;
        };
        total += 1;
        let hit = truth.eq_ignore_ascii_case(guess.trim());
        correct += usize::from(hit);
        println!(
            "{name}: {} (truth: {truth})",
            if hit { "correct" } else { "wrong" }
        );
    }
    let p = binomial_tail(correct, total, manifest.chance_probability);
    println!(
        "{correct}/{total} correct ({:.1}%); chance {:.1}%; one-sided p={p:.5}",
        100.0 * correct as f64 / total.max(1) as f64,
        100.0 * manifest.chance_probability
    );
}

fn binomial_tail(hits: usize, trials: usize, probability: f64) -> f64 {
    (hits..=trials)
        .map(|k| {
            let combinations =
                (0..k).fold(1.0, |value, i| value * (trials - i) as f64 / (i + 1) as f64);
            combinations
                * probability.powi(k as i32)
                * (1.0 - probability).powi((trials - k) as i32)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_are_balanced_and_statistic_has_known_edges() {
        for id in ["flagship_families", "groove_cycle", "period_identity"] {
            let pack = pack(id).unwrap();
            assert!(pack.styles.len() >= 3);
        }
        assert!((binomial_tail(0, 4, 0.25) - 1.0).abs() < 1e-9);
        assert!((binomial_tail(4, 4, 0.25) - 0.25f64.powi(4)).abs() < 1e-9);
    }

    #[test]
    fn study_excerpt_has_equal_duration_and_fades() {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut source = std::io::Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut source, spec).unwrap();
            for _ in 0..500 {
                writer.write_sample(10_000i16).unwrap();
            }
            writer.finalize().unwrap();
        }
        let excerpt = study_excerpt(source.into_inner(), 3.0, 1.0).unwrap();
        let mut reader = hound::WavReader::new(std::io::Cursor::new(excerpt)).unwrap();
        let samples: Vec<i16> = reader.samples::<i16>().map(Result::unwrap).collect();
        assert_eq!(samples.len(), 300);
        assert_eq!(samples[0], 10_000);
        assert!(samples[299].abs() < 200);
    }

    #[test]
    fn holdout_has_no_embedded_seed_list() {
        assert!(
            StudyConfig {
                split: StudySplit::LockedHoldout,
                arm: StudyArm::Ecological,
                length: StudyLength::IdentityExcerpt,
            }
            .seeds()
            .is_empty()
        );
        assert_ne!(DEVELOPMENT_SEEDS.as_slice(), &VALIDATION_SEEDS[..4]);
        assert_eq!(VALIDATION_SEEDS.len(), 12);
    }

    #[test]
    fn rms_normalization_hits_target_without_clipping() {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut source = std::io::Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut source, spec).unwrap();
            for i in 0..1000 {
                writer
                    .write_sample(if i % 2 == 0 { 1_000i16 } else { -1_000i16 })
                    .unwrap();
            }
            writer.finalize().unwrap();
        }
        let normalized = normalize_wav_rms(source.into_inner(), -20.0).unwrap();
        let mut reader = hound::WavReader::new(std::io::Cursor::new(normalized)).unwrap();
        let samples: Vec<i16> = reader.samples::<i16>().map(Result::unwrap).collect();
        let rms = (samples
            .iter()
            .map(|sample| (*sample as f64 / i16::MAX as f64).powi(2))
            .sum::<f64>()
            / samples.len() as f64)
            .sqrt();
        assert!((rms - 0.1).abs() < 0.002);
        assert!(
            samples
                .iter()
                .all(|sample| sample.unsigned_abs() < i16::MAX as u16)
        );
    }

    #[test]
    fn minimal_pair_specs_receive_the_same_literal_motif() {
        for seed in DEVELOPMENT_SEEDS {
            let mut period = Style::Classical.spec();
            let mut groove = Style::AfroCuban.spec();
            let mut process = Style::Minimalism.spec();
            let mut modal = Style::HindustaniInspired.spec();
            let motif = shared_motif(seed);
            for spec in [&mut period, &mut groove, &mut process, &mut modal] {
                install_shared_motif(spec, &motif, seed);
                assert!(!spec.texture.hook_cell);
            }
            let motifs = [period, groove, process, modal].map(|spec| spec.motif(0.5, seed));
            assert!(motifs.windows(2).all(|pair| pair[0] == pair[1]));
        }
    }

    #[test]
    fn clone_gate_flags_identical_symbolic_trajectories() {
        let intent = MusicalIntent {
            seed: 11,
            bars: 8,
            ..MusicalIntent::default()
        };
        let style = Style::Classical;
        let score =
            compose_with_grammar_plan(style.grammar_profile(), &intent, &style.spec()).score;
        let clips = vec![
            analyze_score(&score, "PeriodSentence".into(), "a.wav".into()),
            analyze_score(&score, "PeriodSentence".into(), "b.wav".into()),
        ];
        let warnings = clone_warnings(&clips);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].onset_trajectory_correlation > 0.999);
        assert!(warnings[0].chroma_trajectory_similarity > 0.999);
    }

    #[test]
    fn complete_form_manifest_duration_is_unambiguously_null() {
        let value = serde_json::to_value(Manifest {
            schema_version: 3,
            protocol_version: "test".into(),
            pack_id: "test".into(),
            generated_at_unix_ms: 0,
            theory_engine_version: "test".into(),
            question: "test".into(),
            labels: vec![],
            seeds_per_label: 0,
            seeds: vec![],
            chance_probability: 0.25,
            melody_only: false,
            drums_removed: true,
            shared_instrument: "piano".into(),
            shuffle_seed: 0,
            clip_duration_secs: None,
            study_split: Some(StudySplit::Development),
            study_arm: Some(StudyArm::Ecological),
            study_length: Some(StudyLength::CompleteForm),
            controls: vec![],
            generation_commit: None,
            generation_worktree_dirty: None,
            renderer_version: None,
            soundfont_sha256: None,
            pairing_design: None,
            inclusion_policy: "test".into(),
        })
        .unwrap();
        assert!(value["clip_duration_secs"].is_null());
    }

    #[test]
    fn nuisance_baseline_holds_out_complete_blocks_and_is_hash_addressed() {
        let observations = vec![
            SurfaceObservation {
                audio_sha256: "a1".into(),
                family: "A".into(),
                group_id: "block-1".into(),
                features: [0.0; 7],
            },
            SurfaceObservation {
                audio_sha256: "a2".into(),
                family: "A".into(),
                group_id: "block-2".into(),
                features: [0.1; 7],
            },
            SurfaceObservation {
                audio_sha256: "b1".into(),
                family: "B".into(),
                group_id: "block-1".into(),
                features: [9.9; 7],
            },
            SurfaceObservation {
                audio_sha256: "b2".into(),
                family: "B".into(),
                group_id: "block-2".into(),
                features: [10.0; 7],
            },
        ];
        let baseline = nuisance_baseline(&observations);
        assert_eq!(baseline.correct, 4);
        assert_eq!(baseline.total, 4);
        assert_eq!(baseline.group_count, 2);
        assert_eq!(baseline.predictions[0].held_out_group, "block-1");
        assert_eq!(baseline.predictions[0].audio_sha256, "a1");
    }
}
