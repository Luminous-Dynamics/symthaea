// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Muse Studio: the composer's sketch partner, as a local web app.
//!
//! Give it an intent (mood, energy, style, key) and it composes N genuinely
//! different candidates (seed drives motif, orientation, form, accompaniment
//! pattern, and ensemble). Listen, compare, and take away the one you like —
//! as a **MIDI file you own outright** (the symbolic score, every note
//! editable in any DAW) and/or the rendered WAV. Optionally describe what
//! you want in words: with the CLAP towers available, candidates are ranked
//! by real text↔audio similarity (`symthaea_muse::steering`).
//!
//! Everything runs locally; nothing leaves the machine. The compositions
//! carry zero scraped-training-data liability: the composer is symbolic
//! music theory, the instrument samples are CC0, and the expressive model
//! is fitted on the research-licensed MAESTRO corpus with full provenance
//! embedded.
//!
//! Run:
//! ```bash
//! # sampled instruments (recommended):
//! SYMTHAEA_VCSL_DIR=data/samples/vcsl \
//!   cargo run --release -p symthaea-muse --features studio --bin muse_studio
//! # then open http://localhost:8400
//! ```
//! Port 8400 is the monorepo's ad-hoc dev-server slot (PORTS.md).

use axum::extract::{Path as AxPath, State};
use axum::http::{StatusCode, header};
use axum::response::{Html, IntoResponse};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::{Arc, Mutex, atomic::AtomicU64, atomic::Ordering};
use symthaea_muse::theory_realize::compose_and_realize_styled;
use symthaea_muse::{AudioData, MusicalState};
use symthaea_music_theory::pitch::PitchClass;
use symthaea_music_theory::{CompositionSpec, MusicalIntent, Score, Style, compose_styled};

const SAMPLE_RATE: u32 = 48_000;
const MAX_CANDIDATES: u64 = 12;
const MAX_STORED: usize = 200;

struct Candidate {
    wav: Vec<u8>,
    score: Score,
    /// The spec this candidate was composed with (the style preset when the
    /// user didn't author one) — MIDI export re-derives the same performed
    /// voices from it, so the `.mid` matches the audio.
    spec: symthaea_music_theory::CompositionSpec,
    state: symthaea_muse::MusicalState,
    seed: u64,
    renderer: &'static str,
    phi: f32,
}

#[derive(Default)]
struct Studio {
    candidates: Mutex<HashMap<u64, Candidate>>,
    next_id: AtomicU64,
}

#[derive(Deserialize)]
struct ComposeRequest {
    /// -1..1 dark→bright.
    valence: f32,
    /// 0..1 calm→excited.
    arousal: f32,
    /// 0..1 soft→full.
    energy: f32,
    /// Semitone 0-11 (0 = C).
    tonic: i32,
    style: Style,
    bars: usize,
    base_seed: u64,
    n_candidates: u64,
    /// Optional natural-language prompt for CLAP ranking.
    #[serde(default)]
    prompt: String,
    /// Optional user-authored spec — when present it REPLACES the style's
    /// preset entirely (complete control). Validated; errors come back as
    /// HTTP 400 with every problem listed.
    #[serde(default)]
    spec: Option<CompositionSpec>,
    /// Seed step between candidates (default 1). "More like this" uses a
    /// stride of 6: with the built-in pools that holds the FORM (seed % 2),
    /// accompaniment (seed/2 % ≤3) and motif template (seed % 2..3) fixed
    /// while varying orientation and progression details — genuine
    /// neighborhood exploration instead of a full re-roll.
    #[serde(default = "default_stride")]
    seed_stride: u64,
    /// Consciousness-state dimensions driving the RENDERER (timbre, FM,
    /// filter movement, reverb, drum color, humanization tightness) — the
    /// dimension of muse no slider previously reached: the Studio rendered
    /// everything from MusicalState::default(). Defaults match it.
    #[serde(default = "default_half")]
    dopamine: f32,
    #[serde(default = "default_half")]
    serotonin: f32,
    #[serde(default = "default_noradrenaline")]
    noradrenaline: f32,
    #[serde(default = "default_half")]
    consciousness: f32,
}

fn default_half() -> f32 {
    0.5
}

fn default_noradrenaline() -> f32 {
    0.3
}

fn default_stride() -> u64 {
    1
}

#[derive(Serialize)]
struct CandidateMeta {
    id: u64,
    seed: u64,
    duration_secs: f32,
    /// Cosine similarity to the prompt in CLAP space, when ranking ran.
    similarity: Option<f32>,
    /// Which engine rendered the audio: "fluidsynth" (performed MIDI
    /// through a real soundfont — preferred when the environment provides
    /// it) or "native" (the in-crate synthesizer fallback).
    renderer: &'static str,
    /// Musical Φ: integration of the score-as-system (spectral MIP over
    /// the voice×segment dependency graph — see
    /// `symthaea_music_theory::integration`). Score analysis, not
    /// consciousness.
    phi: f32,
}

#[derive(Serialize)]
struct ComposeResponse {
    candidates: Vec<CandidateMeta>,
    /// Human-readable note about ranking (e.g. why it was skipped).
    ranking_note: String,
    sampled_instruments: bool,
}

#[tokio::main]
async fn main() {
    let studio = Arc::new(Studio::default());
    let app = Router::new()
        .route("/", get(index))
        .route("/api/compose", post(compose))
        .route("/api/spec/{style}", get(spec_preset))
        .route("/api/specs", get(list_specs).post(save_spec))
        .route("/api/specs/{name}", get(load_spec))
        .route("/api/audio/{id}", get(audio))
        .route("/api/midi/{id}", get(midi))
        .route("/api/notes/{id}", get(notes))
        .route("/api/keeper/{id}", post(keeper))
        .with_state(studio);

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], 8400));
    println!("Muse Studio → http://localhost:8400");
    if sampled_active() {
        println!("Instruments: VCSL/VSCO2 samples active");
    } else {
        println!(
            "Instruments: synthesis (set SYMTHAEA_VCSL_DIR=data/samples/vcsl for real samples)"
        );
    }
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("bind :8400");
    axum::serve(listener, app).await.expect("serve");
}

fn sampled_active() -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        symthaea_muse::vcsl::library().is_some()
    }
    #[cfg(target_arch = "wasm32")]
    {
        false
    }
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../../studio/index.html"))
}

/// The built-in preset spec for a style, as editable JSON — the "load, then
/// make it yours" starting point for the spec editor.
async fn spec_preset(AxPath(style): AxPath<String>) -> Result<impl IntoResponse, StatusCode> {
    // Parse through serde so new Style variants can never be forgotten
    // here again (the hardcoded match silently 404'd Nocturne/March/
    // Lullaby/ModalFolk while /api/compose accepted them — caught by the
    // export freshness gate).
    let style: Style = serde_json::from_value(serde_json::Value::String(style))
        .map_err(|_| StatusCode::NOT_FOUND)?;
    let json = serde_json::to_string_pretty(&style.spec())
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(([(header::CONTENT_TYPE, "application/json")], json))
}

const SPEC_DIR: &str = "data/specs";

fn spec_slug(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect()
}

#[derive(Deserialize)]
struct SaveSpecRequest {
    name: String,
    spec: CompositionSpec,
}

/// Save a named user spec to `data/specs/<slug>.json` — "make it yours"
/// persists across restarts. Validation errors come back as 400.
async fn save_spec(
    Json(req): Json<SaveSpecRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    if let Err(errors) = req.spec.validate() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("spec invalid:\n{}", errors.join("\n")),
        ));
    }
    let slug = spec_slug(&req.name);
    if slug.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "empty spec name".into()));
    }
    std::fs::create_dir_all(SPEC_DIR).map_err(internal)?;
    let json = serde_json::to_string_pretty(&req.spec).map_err(internal)?;
    std::fs::write(format!("{SPEC_DIR}/{slug}.json"), json).map_err(internal)?;
    Ok(Json(serde_json::json!({ "saved": slug })))
}

/// Names of all saved specs.
async fn list_specs() -> Json<Vec<String>> {
    let mut names: Vec<String> = std::fs::read_dir(SPEC_DIR)
        .map(|entries| {
            entries
                .flatten()
                .filter_map(|e| {
                    let p = e.path();
                    (p.extension()?.to_str()? == "json")
                        .then(|| p.file_stem()?.to_str().map(String::from))?
                })
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    Json(names)
}

async fn load_spec(AxPath(name): AxPath<String>) -> Result<impl IntoResponse, StatusCode> {
    let slug = spec_slug(&name);
    let body = std::fs::read_to_string(format!("{SPEC_DIR}/{slug}.json"))
        .map_err(|_| StatusCode::NOT_FOUND)?;
    Ok(([(header::CONTENT_TYPE, "application/json")], body))
}

async fn compose(
    State(studio): State<Arc<Studio>>,
    Json(req): Json<ComposeRequest>,
) -> Result<Json<ComposeResponse>, (StatusCode, String)> {
    let n = req.n_candidates.clamp(1, MAX_CANDIDATES);
    let intent_base = MusicalIntent {
        valence: req.valence.clamp(-1.0, 1.0),
        arousal: req.arousal.clamp(0.0, 1.0),
        energy: req.energy.clamp(0.0, 1.0),
        bars: req.bars.clamp(2, 16),
        seed: req.base_seed,
        tonic: PitchClass::new(req.tonic),
    };
    let style = req.style;
    let stride = req.seed_stride.clamp(1, 1_000);
    let prompt = req.prompt.trim().to_string();
    // A user-authored spec replaces the style preset entirely.
    let spec = req.spec;
    // What MIDI export will need later — the authored spec, or the style
    // preset the styled path is equivalent to.
    let spec_used = spec.clone().unwrap_or_else(|| style.spec());
    if let Some(spec) = &spec
        && let Err(errors) = spec.validate()
    {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("spec invalid:\n{}", errors.join("\n")),
        ));
    }

    // Rendering is CPU-bound — do it on the blocking pool.
    let state = MusicalState {
        dopamine: req.dopamine.clamp(0.0, 1.0),
        serotonin: req.serotonin.clamp(0.0, 1.0),
        noradrenaline: req.noradrenaline.clamp(0.0, 1.0),
        consciousness_level: req.consciousness.clamp(0.0, 1.0),
        arousal: req.arousal.clamp(0.0, 1.0),
        valence: req.valence.clamp(-1.0, 1.0),
        ..MusicalState::default()
    };
    let state_used = state.clone();
    let spec_for_render = spec_used.clone();
    let rendered = tokio::task::spawn_blocking(move || {
        let mut out = Vec::new();
        for i in 0..n {
            let seed = intent_base.seed.wrapping_add(i.wrapping_mul(stride));
            let intent = MusicalIntent {
                seed,
                ..intent_base
            };
            let (comp, score) = match &spec {
                Some(spec) => (
                    symthaea_muse::theory_realize::compose_and_realize_spec(
                        &intent,
                        spec,
                        &state,
                        SAMPLE_RATE,
                    ),
                    symthaea_music_theory::compose_with_spec(&intent, spec),
                ),
                None => (
                    compose_and_realize_styled(&intent, style, &state, SAMPLE_RATE),
                    compose_styled(&intent, style),
                ),
            };
            // Preferred render path: the performed MIDI through FluidSynth
            // (an A/B review settled it: "the sound no longer fights the
            // composition"). None → the native render above serves.
            let fluid_wav = fluidsynth_candidate_wav(&score, &spec_for_render, seed, &state);
            out.push((seed, comp, score, fluid_wav));
        }
        out
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Optional CLAP ranking. Degrades gracefully (feature off, no ORT, no
    // network) — the UI shows WHY instead of silently dropping the scores.
    let (similarities, ranking_note) = rank(&prompt, &rendered);

    let mut metas = Vec::with_capacity(rendered.len());
    {
        let mut store = studio.candidates.lock().unwrap();
        if store.len() > MAX_STORED {
            store.clear(); // simple session-scale memory bound
        }
        for (idx, (seed, comp, score, fluid_wav)) in rendered.into_iter().enumerate() {
            let id = studio.next_id.fetch_add(1, Ordering::Relaxed);
            let renderer = if fluid_wav.is_some() {
                "fluidsynth"
            } else {
                "native"
            };
            let wav = match fluid_wav {
                Some(w) => w,
                None => wav_bytes(&comp.audio).map_err(internal)?,
            };
            let phi = symthaea_music_theory::musical_phi(&score).phi;
            metas.push(CandidateMeta {
                id,
                seed,
                duration_secs: comp.duration_secs,
                similarity: similarities.as_ref().map(|s| s[idx]),
                renderer,
                phi,
            });
            store.insert(
                id,
                Candidate {
                    wav,
                    score,
                    spec: spec_used.clone(),
                    state: state_used.clone(),
                    seed,
                    renderer,
                    phi,
                },
            );
        }
    }
    // Best-first when ranked; stable by seed otherwise.
    if similarities.is_some() {
        metas.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    Ok(Json(ComposeResponse {
        candidates: metas,
        ranking_note,
        sampled_instruments: sampled_active(),
    }))
}

#[cfg(feature = "clap-fad")]
fn rank(
    prompt: &str,
    rendered: &[(u64, symthaea_muse::Composition, Score, Option<Vec<u8>>)],
) -> (Option<Vec<f32>>, String) {
    use symthaea_muse::clap_embed::{ClapEmbedder, ClapTextEmbedder, cosine_similarity};
    if prompt.is_empty() {
        return (None, "no prompt given — candidates unranked".into());
    }
    let towers =
        (|| -> anyhow::Result<_> { Ok((ClapEmbedder::new()?, ClapTextEmbedder::new()?)) })();
    let (mut audio_tower, mut text_tower) = match towers {
        Ok(t) => t,
        Err(e) => {
            return (
                None,
                format!("prompt ranking unavailable ({e}) — is ORT_DYLIB_PATH set?"),
            );
        }
    };
    let target = match text_tower.embed(prompt) {
        Ok(t) => t,
        Err(e) => return (None, format!("prompt embedding failed: {e}")),
    };
    let mut sims = Vec::with_capacity(rendered.len());
    for (_, comp, _) in rendered {
        let mono: Vec<f64> = match &comp.audio {
            AudioData::StereoF32(frames) => {
                frames.iter().map(|[l, r]| ((l + r) * 0.5) as f64).collect()
            }
            AudioData::F32(m) => m.iter().map(|&s| s as f64).collect(),
            AudioData::I16(m) => m.iter().map(|&s| s as f64 / 32768.0).collect(),
        };
        match audio_tower.embed(&mono) {
            Ok(emb) => sims.push(cosine_similarity(&emb, &target)),
            Err(e) => return (None, format!("audio embedding failed: {e}")),
        }
    }
    (
        Some(sims),
        format!("ranked by CLAP similarity to “{prompt}”"),
    )
}

#[cfg(not(feature = "clap-fad"))]
fn rank(
    prompt: &str,
    _rendered: &[(u64, symthaea_muse::Composition, Score, Option<Vec<u8>>)],
) -> (Option<Vec<f32>>, String) {
    if prompt.is_empty() {
        (None, "no prompt given — candidates unranked".into())
    } else {
        (
            None,
            "prompt ranking needs the `clap-fad` build feature".into(),
        )
    }
}

async fn audio(
    State(studio): State<Arc<Studio>>,
    AxPath(id): AxPath<u64>,
) -> Result<impl IntoResponse, StatusCode> {
    let store = studio.candidates.lock().unwrap();
    let c = store.get(&id).ok_or(StatusCode::NOT_FOUND)?;
    Ok(([(header::CONTENT_TYPE, "audio/wav")], c.wav.clone()))
}

async fn midi(
    State(studio): State<Arc<Studio>>,
    AxPath(id): AxPath<u64>,
) -> Result<impl IntoResponse, StatusCode> {
    let (score, spec, state, seed) = {
        let store = studio.candidates.lock().unwrap();
        let c = store.get(&id).ok_or(StatusCode::NOT_FOUND)?;
        (c.score.clone(), c.spec.clone(), c.state.clone(), c.seed)
    };
    // The PERFORMED export — swing, rubato, expression, and the contrast
    // counter-instrument baked in, matching the audio render. Writes to a
    // path; round-trip through a temp file.
    let path = std::env::temp_dir().join(format!("muse_studio_{id}.mid"));
    symthaea_muse::midi_export::export_performance_midi(&score, &spec, seed, &state, &path)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let bytes = std::fs::read(&path).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let _ = std::fs::remove_file(&path);
    Ok((
        [
            (header::CONTENT_TYPE, "audio/midi".to_string()),
            (
                header::CONTENT_DISPOSITION,
                format!("attachment; filename=\"muse_seed{seed}.mid\""),
            ),
        ],
        bytes,
    ))
}

/// Render one candidate's PERFORMED MIDI through FluidSynth, when the
/// environment provides it (see `symthaea_muse::fluid_render`). None →
/// the caller serves the native render instead.
fn fluidsynth_candidate_wav(
    score: &Score,
    spec: &symthaea_music_theory::CompositionSpec,
    seed: u64,
    state: &symthaea_muse::MusicalState,
) -> Option<Vec<u8>> {
    symthaea_muse::fluid_render::available()?;
    let path = std::env::temp_dir().join(format!(
        "muse_studio_fluid_{}_{seed}.mid",
        std::process::id()
    ));
    symthaea_muse::midi_export::export_performance_midi(score, spec, seed, state, &path).ok()?;
    let color = symthaea_muse::fluid_render::RenderColor::from_state(state);
    let wav = symthaea_muse::fluid_render::render_midi_to_wav(&path, SAMPLE_RATE, Some(color));
    let _ = std::fs::remove_file(&path);
    wav
}

/// Mark a candidate as a KEEPER — the start of the taste dataset. Every
/// keep appends one JSON line to `data/taste/keepers.jsonl`: seed, spec
/// name, mode, ensemble, renderer, and the hook cell the piece opened
/// with. This is the raw material for learned taste ("which motifs do
/// humans voluntarily keep?") and for evolving the hook skeleton banks
/// from what survives contact with ears.
async fn keeper(
    State(studio): State<Arc<Studio>>,
    AxPath(id): AxPath<u64>,
) -> Result<impl IntoResponse, StatusCode> {
    let (spec, seed, renderer, phi) = {
        let store = studio.candidates.lock().unwrap();
        let c = store.get(&id).ok_or(StatusCode::NOT_FOUND)?;
        (c.spec.clone(), c.seed, c.renderer, c.phi)
    };
    let hook = symthaea_music_theory::HookCell::generate(seed, spec.meter as f64);
    let entry = serde_json::json!({
        "ts": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        "seed": seed,
        "spec": spec.name,
        "mode": spec.mode.map(|m| format!("{m:?}")),
        "ensemble": spec.ensemble(seed),
        "renderer": renderer,
        "phi": phi,
        "hook": hook
            .notes
            .iter()
            .map(|(deg, dur)| (*deg, dur.beats()))
            .collect::<Vec<_>>(),
    });
    std::fs::create_dir_all("data/taste").map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("data/taste/keepers.jsonl")
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    writeln!(f, "{entry}").map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(StatusCode::NO_CONTENT)
}

/// The performed notes of a candidate, per voice — powers the piano-roll.
async fn notes(
    State(studio): State<Arc<Studio>>,
    AxPath(id): AxPath<u64>,
) -> Result<impl IntoResponse, StatusCode> {
    let (score, spec, state, seed) = {
        let store = studio.candidates.lock().unwrap();
        let c = store.get(&id).ok_or(StatusCode::NOT_FOUND)?;
        (c.score.clone(), c.spec.clone(), c.state.clone(), c.seed)
    };
    let voices = symthaea_muse::theory_realize::perform_with_spec(&score, &spec, seed, &state);
    Ok(axum::Json(voices))
}

fn wav_bytes(audio: &AudioData) -> anyhow::Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
        match audio {
            AudioData::StereoF32(frames) => {
                for [l, r] in frames {
                    writer.write_sample((l.clamp(-1.0, 1.0) * 32767.0) as i16)?;
                    writer.write_sample((r.clamp(-1.0, 1.0) * 32767.0) as i16)?;
                }
            }
            AudioData::F32(mono) => {
                for s in mono {
                    let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
                    writer.write_sample(v)?;
                    writer.write_sample(v)?;
                }
            }
            AudioData::I16(mono) => {
                for s in mono {
                    writer.write_sample(*s)?;
                    writer.write_sample(*s)?;
                }
            }
        }
        writer.finalize()?;
    }
    Ok(cursor.into_inner())
}

fn internal<E: std::fmt::Display>(e: E) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}
