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
//!
//! See also: `symthaea-ui` (`crates/bridges/symthaea-ui/`) for the general
//! Symthaea consciousness/chat UI over the `symthaea-service` HTTP gateway
//! — a separate app, kept deliberately separate per
//! `SYMTHAEA_UNIFIED_UI_PLAN_2026-07-10.md`'s non-goals (this tool stays a
//! composer-specific instrument, not folded into the unified UI).

use axum::extract::{Path as AxPath, State};
use axum::http::{StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, atomic::AtomicU64, atomic::Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use symthaea_muse::adaptive_prediction::{
    AdaptiveOutcomeModel, InterventionCalibrationEvidence, PredictionContext, TextureBand,
};
use symthaea_muse::closed_loop::record_symbolic_region;
use symthaea_muse::cognitive_bridge::{CognitiveGoal, CognitiveSection, propose_symbolic_action};
use symthaea_muse::cognitive_session::{CognitiveSessionConfig, run_sonata_cognitive_session};
use symthaea_muse::piece_recipe::{
    DecisionDisposition, PieceRecipe, RendererRecipe, ReproductionGap,
};
use symthaea_muse::sonata_intervention::{
    SonataReturnStrategy, generate_and_rank_sonata_return_with_model,
    learn_symbolic_outcomes_from_batch,
};
use symthaea_muse::{AudioData, MusicalState};
use symthaea_muse_protocol::{
    ANALYST_PIECE_BUNDLE_VERSION, BundleEnvelope, BundleWarning, CadenceEvent, EvidenceBasis,
    EvidenceStatus, GrammarProvenance, LISTEN_COMPOSITION_BUNDLE_VERSION,
    LISTEN_PERFORMANCE_BUNDLE_VERSION, ListenCompositionBundle, ListenPerformanceBundle,
    MeterPoint, MotifDefinition, MotifOccurrence, MusicalTime, OrchestrationRegion,
    PIECE_PROVENANCE_BUNDLE_VERSION, PerformanceVoiceSummary, PerformedNoteEvent, PhraseRegion,
    PieceProvenanceBundle, ProvenanceArtifact, ReproducibilityClaim, ResonanceCurve,
    ResonanceSample, SectionRegion, SonorityRegion, SymbolicNoteEvent, TempoPoint,
    TitleRecipeSummary, VoiceActivity,
};
use symthaea_music_theory::pitch::PitchClass;
use symthaea_music_theory::{
    CompositionSpec, Duration, Emphasis, FormKind, GrammarFamily, GrammarPlanEvidence,
    MusicalIntent, ObligationKind, ReturnTransformation, Score, ScoreNote, SonataRealization,
    SonataSectionKind, Style, VoiceRole, compare_melodic_sequences, compose_sonata_with_plan,
    profile_score_region, verify_sonata_obligations,
};
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::services::{ServeDir, ServeFile};

const SAMPLE_RATE: u32 = 48_000;
const TICKS_PER_BEAT: u32 = 960;
const MAX_CANDIDATES: u64 = 12;
const MAX_STORED: usize = 200;
/// `ComposeRequest::bars` must fall in this range or `/api/compose`
/// rejects the request outright (HTTP 400) rather than silently
/// rescaling it. Upper bound raised to 36 (was a silent `clamp(2, 16)`)
/// specifically to make Blues call-response's 3-chorus/36-bar case
/// reachable live -- 36 bars was already the render workload this
/// interactive endpoint was accidentally producing before the chorus-
/// count fix, so this isn't new cost, just an honest ceiling for it.
/// Interactive-render-budget concerns for engines that DON'T scale
/// linearly with `bars` (see `groove_cycle.rs`/`modal_arc.rs`) remain a
/// real, separate, not-yet-built per-engine cost estimate -- this range
/// only restores "the request means what it says," not a full budget
/// system.
const COMPOSE_BARS_RANGE: std::ops::RangeInclusive<usize> = 2..=36;
/// Pre-render novelty floor (`MUSE_DIVERSITY_TRUTH_PLAN_2026-07-18.md` Phase
/// 2): below this, a batch candidate is close enough — to its nearest
/// BATCH neighbor's `NoveltyBreakdown.overall`, or to a recent KEEPER's —
/// that it gets a real high-level structural variation via `DiversityPlan`,
/// not just its existing seed. A floor, never a fitness function — it only
/// fires below the threshold and never optimizes further once cleared.
/// Initial, adjustable value: small enough to leave legitimate
/// close-but-distinct premises alone (see `NoveltyBreakdown`'s doc comment
/// on harmonic distance being honestly near-zero for Archetype-sourced
/// styles), large enough to catch genuine near-twins.
const NOVELTY_FLOOR: f64 = 0.5;

/// Evict entries with the smallest keys down to `max_stored`, one at a
/// time. Generic over the value type so the eviction algorithm itself
/// can be unit-tested without constructing real `Candidate` fixtures
/// (`Candidate` carries a rendered WAV, full `Score`, `PieceRecipe`,
/// etc.).
fn evict_oldest_by_id<V>(store: &mut HashMap<u64, V>, max_stored: usize) {
    while store.len() > max_stored {
        let Some(&oldest_id) = store.keys().min() else {
            break;
        };
        store.remove(&oldest_id);
    }
}

/// Keeps the candidate store down to `MAX_STORED` by evicting the oldest
/// entries (smallest IDs — `next_id` only ever increases, so ID order is
/// insertion order) individually, instead of wiping every candidate in
/// the session. `store.clear()` used to invalidate every previous audio/
/// MIDI/notes/keeper-by-candidate link the moment any one session
/// composed past 200 candidates total — including whatever the user was
/// actively listening to. This doesn't pin currently-displayed/queued/
/// keeper-in-progress candidates (that needs real reference tracking the
/// server doesn't have yet); it only stops the wholesale-wipe behavior.
fn evict_oldest_candidates(store: &mut HashMap<u64, Candidate>) {
    evict_oldest_by_id(store, MAX_STORED);
}

/// Records `id` as having produced `fingerprint` and reports whether an
/// EARLIER id in the same batch already produced the identical
/// fingerprint. Returns `None` the first time a fingerprint is seen, and
/// `Some(first_id)` — always the very first id, never a later duplicate's
/// id — on every subsequent match, so a batch of 3+ identical candidates
/// all point back to the same original rather than chaining.
fn mark_duplicate(seen: &mut HashMap<u64, u64>, fingerprint: u64, id: u64) -> Option<u64> {
    let first_id = *seen.entry(fingerprint).or_insert(id);
    (first_id != id).then_some(first_id)
}

#[cfg(test)]
mod dedup_tests {
    use super::mark_duplicate;
    use std::collections::HashMap;

    #[test]
    fn first_occurrence_of_a_fingerprint_is_not_a_duplicate() {
        let mut seen = HashMap::new();
        assert_eq!(mark_duplicate(&mut seen, 0xABC, 1), None);
    }

    #[test]
    fn second_occurrence_points_back_to_the_first_id() {
        let mut seen = HashMap::new();
        assert_eq!(mark_duplicate(&mut seen, 0xABC, 1), None);
        assert_eq!(mark_duplicate(&mut seen, 0xABC, 2), Some(1));
    }

    #[test]
    fn three_or_more_duplicates_all_point_to_the_original_not_each_other() {
        let mut seen = HashMap::new();
        assert_eq!(mark_duplicate(&mut seen, 0xABC, 1), None);
        assert_eq!(mark_duplicate(&mut seen, 0xABC, 2), Some(1));
        assert_eq!(mark_duplicate(&mut seen, 0xABC, 3), Some(1));
    }

    #[test]
    fn distinct_fingerprints_never_collide() {
        let mut seen = HashMap::new();
        assert_eq!(mark_duplicate(&mut seen, 0x111, 1), None);
        assert_eq!(mark_duplicate(&mut seen, 0x222, 2), None);
        assert_eq!(mark_duplicate(&mut seen, 0x111, 3), Some(1));
    }
}

#[cfg(test)]
mod eviction_tests {
    use super::evict_oldest_by_id;
    use std::collections::HashMap;

    #[test]
    fn evicts_smallest_ids_first_down_to_the_limit() {
        let mut store: HashMap<u64, &str> = (0..10).map(|id| (id, "candidate")).collect();
        evict_oldest_by_id(&mut store, 6);
        assert_eq!(store.len(), 6);
        // The 6 highest IDs (4..=9) survive; the 4 oldest (0..=3) are gone.
        for id in 4..10 {
            assert!(
                store.contains_key(&id),
                "expected id {id} to survive eviction"
            );
        }
        for id in 0..4 {
            assert!(!store.contains_key(&id), "expected id {id} to be evicted");
        }
    }

    #[test]
    fn does_not_touch_the_store_when_already_under_the_limit() {
        let mut store: HashMap<u64, &str> = (0..5).map(|id| (id, "candidate")).collect();
        evict_oldest_by_id(&mut store, 200);
        assert_eq!(store.len(), 5);
    }
}

/// Proves `CandidateMeta`/`ComposeResponse`'s real JSON output — the
/// exact bytes a client actually receives — deserializes cleanly into
/// `symthaea-muse-protocol`'s shared client types. If a future field
/// rename/type change here ever breaks that, this test is what catches
/// it, not a client somewhere quietly failing to parse a response.
#[cfg(test)]
mod wire_compat_tests {
    use super::{CandidateMeta, ComposeResponse, Style, TitleRecipeSummary, snake_case_variant};

    fn sample_candidate_meta() -> CandidateMeta {
        CandidateMeta {
            id: 7,
            seed: 42,
            duration_secs: 12.5,
            similarity: Some(0.83),
            renderer: "native",
            phi: 0.013,
            local_coherence: 0.53,
            global_coherence: 0.05,
            ground: None,
            grammar: "memory",
            ending: None,
            card: None,
            title: "Copper Lantern".to_string(),
            title_recipe: TitleRecipeSummary {
                family: "image".to_string(),
                template_id: "adjective-object-v2".to_string(),
                source_traits: vec!["ternary form".to_string()],
                alternatives: vec!["Lantern at Low Water".to_string()],
            },
            why: vec!["Ideas return.".to_string()],
            meter: 4,
            novelty: None,
            style: "Classical".to_string(),
            duplicate_of: None,
            identity: symthaea_muse_protocol::ArtifactIdentity {
                score_content: symthaea_muse_protocol::ScoreContentArtifactId("scr".to_string()),
                composition: symthaea_muse_protocol::CompositionArtifactId("cmp".to_string()),
                rendition: symthaea_muse_protocol::RenditionArtifactId("rnd".to_string()),
            },
        }
    }

    #[test]
    fn candidate_meta_json_matches_the_shared_client_type() {
        let json = serde_json::to_string(&sample_candidate_meta()).unwrap();
        let client: symthaea_muse_protocol::Candidate = serde_json::from_str(&json).unwrap();
        assert_eq!(client.id, 7);
        assert_eq!(client.title, "Copper Lantern");
        assert_eq!(
            client
                .title_recipe
                .as_ref()
                .map(|recipe| recipe.family.as_str()),
            Some("image")
        );
        assert_eq!(client.style, "Classical");
        assert_eq!(client.renderer, "native");
        assert_eq!(client.similarity, Some(0.83));
        assert_eq!(client.duplicate_of, None);
        assert_eq!(
            client.identity.as_ref().map(|i| i.score_content.0.as_str()),
            Some("scr")
        );
    }

    #[test]
    fn duplicate_of_round_trips_through_the_shared_client_type() {
        let mut meta = sample_candidate_meta();
        meta.duplicate_of = Some(3);
        let json = serde_json::to_string(&meta).unwrap();
        let client: symthaea_muse_protocol::Candidate = serde_json::from_str(&json).unwrap();
        assert_eq!(client.duplicate_of, Some(3));
    }

    #[test]
    fn compose_response_json_matches_the_shared_client_type() {
        let response = ComposeResponse {
            candidates: vec![sample_candidate_meta()],
            ranking_note: "CLAP ranking unavailable".to_string(),
            sampled_instruments: false,
        };
        let json = serde_json::to_string(&response).unwrap();
        let client: symthaea_muse_protocol::ComposeResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(client.candidates.len(), 1);
        assert_eq!(client.ranking_note, "CLAP ranking unavailable");
    }

    #[test]
    fn listen_styles_reports_every_style_with_its_real_family() {
        let entries: Vec<symthaea_muse_protocol::StyleFamily> = Style::ALL
            .into_iter()
            .map(|style| symthaea_muse_protocol::StyleFamily {
                name: format!("{style:?}"),
                family: snake_case_variant(&style.grammar_family()),
            })
            .collect();
        assert_eq!(entries.len(), Style::ALL.len());
        let afrocuban = entries
            .iter()
            .find(|entry| entry.name == "AfroCuban")
            .expect("AfroCuban present");
        assert_eq!(afrocuban.family, "groove_cycle");
        let json = serde_json::to_string(&entries).unwrap();
        let round_tripped: Vec<symthaea_muse_protocol::StyleFamily> =
            serde_json::from_str(&json).unwrap();
        assert_eq!(round_tripped, entries);
    }

    #[test]
    fn keeper_entry_json_matches_the_shared_client_type() {
        // `keeper()`'s handler builds this via `serde_json::json!` rather
        // than a typed struct (see its doc comment) — this mirrors that
        // literal shape rather than constructing a `CandidateMeta`, since
        // there's no server-side struct to derive it from.
        let entry = serde_json::json!({
            "ts": 1_752_600_000u64,
            "seed": 42,
            "spec": "Classical",
            "mode": "Major",
            "ensemble": ["Piano"],
            "renderer": "native",
            "phi": 0.013,
            "local_coherence": 0.53,
            "global_coherence": 0.05,
            "ground_worthiness": null,
            "grammar": "memory",
            "ending": null,
            "title": "Copper Lantern",
            "novelty": null,
            "audio_key": "abc123",
            "artifact_layout": "keeper-directory-v1",
            "midi_available": true,
            "reproduction_gaps": [],
            "recipe": {},
            "hook": [],
        });
        let json = serde_json::to_string(&entry).unwrap();
        let client: symthaea_muse_protocol::KeeperEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(client.title.as_deref(), Some("Copper Lantern"));
        assert_eq!(client.audio_key, "abc123");
        assert!(client.midi_available);
    }
}

const ADAPTIVE_MODEL_PATH: &str = "data/muse-adaptive-outcomes-v2.json";
const LEGACY_ADAPTIVE_MODEL_PATH: &str = "data/muse-adaptive-outcomes-v1.json";

struct Candidate {
    wav: Vec<u8>,
    created_at_unix_ms: u64,
    score: Score,
    /// The spec this candidate was composed with (the style preset when the
    /// user didn't author one) — MIDI export re-derives the same performed
    /// voices from it, so the `.mid` matches the audio.
    spec: symthaea_music_theory::CompositionSpec,
    state: symthaea_muse::MusicalState,
    seed: u64,
    renderer: &'static str,
    phi: f32,
    /// Local coherence: the consonance-excess channel of musical Φ
    /// (vertical, within-segment). Kept separate from `global_coherence`
    /// per the listening review: "there are at least two different kinds
    /// of musical coherence... resist compressing both into one number."
    local_coherence: f32,
    /// Global coherence: the motif-trigram channel (long-range identity).
    global_coherence: f32,
    /// Ground-worthiness of the audition-winning subject, when this
    /// candidate is a ground form (passacaglia/erosion/lineage). The five
    /// scores + composite the composition acted on — logged into keeper
    /// entries so ♥ data can learn which weighting matches the ear.
    ground: Option<symthaea_music_theory::passacaglia::GroundWorthiness>,
    /// The internal section-return structure `compose_with_spec_and_form`
    /// produced alongside `score` — `None` for form kinds that never build
    /// one (Fugue/ProgSuite/Sonata/Renaissance/Opera/ground forms). Powers
    /// the Muse Atlas endpoint's structural fingerprint.
    form: Option<symthaea_music_theory::Form>,
    /// The REAL grammar-plan evidence this candidate was composed with
    /// (`compose_with_grammar_plan`'s own `GrammarRealization::plan`) --
    /// `None` only for a candidate composed from a user-authored custom
    /// spec, which has no `Style` to derive a `GrammarProfile` from.
    /// Consumed by the Analyst endpoint (`analyst_bundle`) so its
    /// verification reflects what actually produced this piece, not a
    /// synthesized guess.
    plan: Option<GrammarPlanEvidence>,
    /// Identity grammar + erosion ending (see CandidateMeta).
    grammar: &'static str,
    ending: Option<&'static str>,
    /// Identity card (see CandidateMeta).
    card: Option<symthaea_music_theory::describe::IdentityCard>,
    /// Novelty within the batch (see CandidateMeta).
    novelty: Option<symthaea_music_theory::explorer::NoveltyBreakdown>,
    /// Complete resolved input and provenance for exact symbolic reproduction.
    recipe: PieceRecipe,
}

struct Studio {
    candidates: Mutex<HashMap<u64, Candidate>>,
    next_id: AtomicU64,
    next_keeper_id: AtomicU64,
    /// Serializes atomic read-replace updates of the keeper index.
    keeper_log: Mutex<()>,
    /// Context-sensitive action-outcome calibration shared across Studio requests.
    adaptive_outcomes: Mutex<AdaptiveOutcomeModel>,
    /// Genealogy ledger for kept pieces. `None` if the sqlite file couldn't
    /// be opened (e.g. an unwritable `data/` dir) -- degrades to "no
    /// genealogy allocated," never to a crashed server, matching this
    /// file's existing graceful-degradation posture for the adaptive
    /// outcome model.
    genealogy: Option<symthaea_muse::genealogy::GenealogyStore>,
}

impl Default for Studio {
    fn default() -> Self {
        let genealogy = match symthaea_muse::genealogy::GenealogyStore::open(Path::new(
            "data/genealogy/ledger.sqlite3",
        )) {
            Ok(store) => Some(store),
            Err(error) => {
                eprintln!("[muse_studio] genealogy ledger unavailable: {error}");
                None
            }
        };
        Self {
            candidates: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(0),
            next_keeper_id: AtomicU64::new(0),
            keeper_log: Mutex::new(()),
            adaptive_outcomes: Mutex::new(load_adaptive_outcome_model()),
            genealogy,
        }
    }
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
    /// Must be in `COMPOSE_BARS_RANGE` (currently 2-36) -- validated, not
    /// silently clamped: a request outside that range comes back as
    /// HTTP 400 rather than being silently rewritten to a different
    /// value. Found necessary 2026-07-26 after a live listening/analysis
    /// session: two DIFFERENT `bars` requests (24 and 36) were silently
    /// collapsing to the same clamped value, so the returned evidence no
    /// longer matched what was actually asked for.
    bars: usize,
    base_seed: u64,
    n_candidates: u64,
    /// Optional natural-language prompt for CLAP ranking.
    #[serde(default)]
    prompt: String,
    /// Identity exploration (default ON): for a plain Compose (stride 1,
    /// several candidates), seeds are chosen by the Identity Explorer —
    /// maximally-different hooks/forms/textures from a wide window — so
    /// the batch offers genuinely different identities instead of
    /// neighborhood variants. "More like this" (stride 6) bypasses it by
    /// construction, keeping its deliberate-neighborhood meaning.
    #[serde(default = "default_true")]
    explore: bool,
    /// Identity grammar override — how the piece's ideas live over time:
    /// "Auto" (the style's pool + seed decide), "Memory" (ideas return),
    /// "Persistence" (the ground remains), "Lineage" (the ground becomes
    /// its descendants), "Erosion" (the ground loses itself). Rewrites the
    /// effective spec's form_pool, so it composes across ANY style — an
    /// Erosion Nocturne is one dropdown away.
    #[serde(default)]
    grammar: Option<String>,
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
    /// When `true`, a single-candidate compose (`n_candidates == 1`) still
    /// gets the premise layer's variation (tempo third, texture budget,
    /// phrase length, ensemble persona, mode) — the same mechanism
    /// multi-candidate explored batches already get, just applied to the
    /// one seed instead of picking among several. Without this, every
    /// Listen-radio piece in a style composes under that style's single
    /// fixed premise forever, which is a large, avoidable share of "the
    /// songs sound the same": see `MUSE_DIVERSITY_TRUTH_PLAN_2026-07-18.md`
    /// Phase 2. Ignored when `n_candidates > 1` (the explorer already
    /// premise-varies each candidate there). Defaults to `false` so an
    /// authored Create-mode compose keeps its exact premise unless it
    /// opts in.
    #[serde(default)]
    vary_premise: bool,
    /// Which render backend to use: `"fluidsynth"` (real soundfont,
    /// preferred — see `fluid_render.rs`'s A/B rationale), `"native"`
    /// (the in-crate synthesizer, VCSL/VSCO2-sampled where available), or
    /// omitted/anything else for the server's own default choice
    /// (FluidSynth when the environment provides it, native otherwise).
    /// `"fluidsynth"` degrades to native rather than erroring if the
    /// environment doesn't actually have FluidSynth available — the
    /// per-candidate `renderer` field in the response always reports
    /// which one actually rendered, so a client can't be silently misled
    /// either way.
    #[serde(default)]
    renderer: Option<String>,
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

fn default_true() -> bool {
    true
}

fn load_adaptive_outcome_model() -> AdaptiveOutcomeModel {
    for path in [ADAPTIVE_MODEL_PATH, LEGACY_ADAPTIVE_MODEL_PATH] {
        match std::fs::read_to_string(path) {
            Ok(body) => match serde_json::from_str::<AdaptiveOutcomeModel>(&body) {
                Ok(mut model) if model.is_compatible() => {
                    let upgraded = model.upgrade_legacy();
                    if upgraded {
                        eprintln!(
                            "[muse_studio] upgraded legacy adaptive outcome evidence from {path}"
                        );
                    }
                    return model;
                }
                Ok(_) => {
                    eprintln!(
                        "[muse_studio] ignoring incompatible adaptive outcome model at {path}"
                    );
                }
                Err(error) => {
                    eprintln!(
                        "[muse_studio] ignoring invalid adaptive outcome model at {path}: {error}"
                    );
                }
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                eprintln!("[muse_studio] could not read adaptive outcome model at {path}: {error}");
            }
        }
    }
    AdaptiveOutcomeModel::default()
}

fn persist_adaptive_outcome_model(model: &AdaptiveOutcomeModel) -> std::io::Result<()> {
    let path = Path::new(ADAPTIVE_MODEL_PATH);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("json.tmp.{}", std::process::id()));
    let body = serde_json::to_vec_pretty(model)
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    std::fs::write(&temporary, body)?;
    std::fs::rename(temporary, path)
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
    /// Local (vertical/consonance) coherence — Φ's dependency channel.
    local_coherence: f32,
    /// Global (long-range/motif) coherence — Φ's trigram channel.
    global_coherence: f32,
    /// Ground-worthiness scores when this candidate is a ground form.
    ground: Option<symthaea_music_theory::passacaglia::GroundWorthiness>,
    /// Which identity grammar this candidate used (memory / subject /
    /// persistence / erosion / lineage) — the taste log records which
    /// mechanism every ♥ endorsed.
    grammar: &'static str,
    /// The erosion ending (recovery / acceptance / elegy), when relevant.
    ending: Option<&'static str>,
    /// The candidate's identity card: a stable evocative name plus honest
    /// trait words derived from its premise and hook ("identities, not
    /// seeds"). Present only for EXPLORED batches — a single compose or
    /// "More like this" runs your authored spec with no premise, so there
    /// is no premise to describe.
    card: Option<symthaea_music_theory::describe::IdentityCard>,
    /// A name for this candidate, always present unlike `card`: the
    /// card's own title when one exists, otherwise `describe::title_recipe`
    /// (no premise needed) — a plain Listen-tab piece gets a real name
    /// too, not just "seed N".
    title: String,
    /// Deterministic naming provenance and stable alternatives for `title`.
    title_recipe: TitleRecipeSummary,
    /// "Why this piece": 2-5 honest sentences translating the actual
    /// grammar/development/accompaniment/texture mechanisms that composed
    /// THIS candidate into prose. Unlike `card`, always present — it needs
    /// no premise, so it works for a plain Listen-tab piece too, not just
    /// explored Discovery batches.
    why: Vec<String>,
    /// The candidate's meter (premises may pick from a style's pool).
    meter: u8,
    /// Novelty within THIS batch: channel distances to the candidate's
    /// nearest batch neighbor (explored batches only). An observable —
    /// "why is this candidate different" — never a fitness function.
    novelty: Option<symthaea_music_theory::explorer::NoveltyBreakdown>,
    /// The spec/style name this candidate composed under (drives the
    /// Listen tab's style-reactive palette).
    style: String,
    /// Set when this candidate's score hashes identically to an EARLIER
    /// candidate already produced in this same compose batch — the value
    /// is that earlier candidate's `id`. `None` for a genuinely distinct
    /// score. Detected via `exact_fingerprint` (canonicalized note-event
    /// hash), computed once per candidate and compared within this
    /// request only — cross-batch/keeper/history dedup is a further step,
    /// not done here. Each duplicate still gets its own `id` (so its
    /// audio/MIDI stay independently downloadable) rather than reusing the
    /// earlier one, to avoid a fingerprint collision silently aliasing two
    /// different pieces' stored audio.
    duplicate_of: Option<u64>,
    /// Real content-hash identity (score/composition/rendition), computed
    /// from the same `score`/`recipe`/`wav` this candidate was stored
    /// with — the same hashes `piece_provenance`/the genealogy ledger use,
    /// just computed at compose time instead of keep time.
    identity: symthaea_muse_protocol::ArtifactIdentity,
}

#[derive(Serialize)]
struct ComposeResponse {
    candidates: Vec<CandidateMeta>,
    /// Human-readable note about ranking (e.g. why it was skipped).
    ranking_note: String,
    sampled_instruments: bool,
}

/// Frozen narrow intervention request. `return_perturbation_semitones` is a
/// research-only negative control; zero audits the canonical Sonata unchanged.
#[derive(Deserialize)]
struct SonataCognitiveRequest {
    seed: u64,
    tonic: i32,
    #[serde(default)]
    valence: f32,
    #[serde(default = "default_half")]
    arousal: f32,
    #[serde(default = "default_half")]
    energy: f32,
    #[serde(default = "default_sonata_bars")]
    bars: usize,
    #[serde(default)]
    return_perturbation_semitones: i8,
    #[serde(default = "default_half")]
    dopamine: f32,
    #[serde(default = "default_half")]
    serotonin: f32,
    #[serde(default = "default_noradrenaline")]
    noradrenaline: f32,
    #[serde(default = "default_half")]
    consciousness: f32,
}

fn default_sonata_bars() -> usize {
    4
}

#[derive(Serialize)]
struct SonataAlternativeMeta {
    alternative_id: String,
    strategy: SonataReturnStrategy,
    selected: bool,
    motif_return_similarity: f32,
    driving_obligation_verified: Option<bool>,
    theory_valid: bool,
    theory_fatal_count: usize,
    theory_warning_count: usize,
    preserved_invariants: bool,
    policy_utility: f32,
    mean_prediction_error: Option<f32>,
}

#[derive(Serialize)]
struct SonataCognitiveResponse {
    candidate_id: u64,
    selected_alternative_id: String,
    driving_obligation_id: u64,
    research_perturbation_semitones: i8,
    alternatives: Vec<SonataAlternativeMeta>,
    selection_rationale: Vec<String>,
    calibration: InterventionCalibrationEvidence,
    observed_outcome: symthaea_muse::cognitive_bridge::ObservedMusicalOutcome,
    model_intervention_context_samples_after: u64,
    world_model_alternatives_learned: usize,
    artist_response_recorded: bool,
    renderer: &'static str,
    duration_secs: f32,
    reproduction_gaps: Vec<ReproductionGap>,
    cognitive_session_version: String,
    cognitive_backend: String,
    cognitive_session_fingerprint: String,
    cognitive_frame_count: usize,
    fep_rng_seed: u64,
    fep_goal_preferences: Vec<f64>,
    fep_goal_precision: f64,
    fep_actions_committed: u64,
    fep_td_updates: u64,
    fep_transition_history_size: usize,
    terminal_fep_action: String,
    terminal_free_energy: f64,
}

#[derive(Debug, Clone, Copy, Deserialize)]
enum SonataCommitDisposition {
    Accepted,
    Edited,
    Rejected,
}

#[derive(Deserialize)]
struct SonataCommitRequest {
    disposition: SonataCommitDisposition,
    #[serde(default)]
    alternative_id: Option<String>,
    #[serde(default)]
    artist_note: Option<String>,
}

#[derive(Serialize)]
struct SonataCommitResponse {
    candidate_id: u64,
    disposition: DecisionDisposition,
    selected_alternative_id: Option<String>,
    preference_model_updated: bool,
}

/// Exact-host match for the `Origin` header's `http://<host>[:port]` (or
/// bracketed-IPv6 `http://[<host>]:<port>`) form. Deliberately NOT a
/// `starts_with` check: `origin.starts_with(b"http://localhost")` also
/// matches `http://localhost.attacker.example`, which is exactly the
/// origin a malicious page would use to reach this locally-bound API and
/// spend CPU on renders, read results, or write keeper artifacts.
fn is_allowed_dev_origin(origin: &[u8]) -> bool {
    let Ok(s) = std::str::from_utf8(origin) else {
        return false;
    };
    let Some(rest) = s.strip_prefix("http://") else {
        return false;
    };
    let host = if let Some(after_bracket) = rest.strip_prefix('[') {
        after_bracket.split(']').next().unwrap_or("")
    } else {
        rest.split(['/', ':']).next().unwrap_or("")
    };
    matches!(host, "localhost" | "127.0.0.1" | "::1")
}

/// Localhost-only CORS so the `symthaea-muse-ui` Leptos dev server (a
/// separate origin/port under Trunk) can call this API. Mirrors the
/// pattern in `symthaea/src/api/mod.rs::build_cors_layer`, hardened to an
/// exact host match (see `is_allowed_dev_origin`).
fn localhost_cors_layer() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(|origin, _| {
            is_allowed_dev_origin(origin.as_bytes())
        }))
        .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
        .allow_headers(tower_http::cors::Any)
}

#[cfg(test)]
mod cors_tests {
    use super::is_allowed_dev_origin;

    #[test]
    fn allows_exact_localhost_variants() {
        assert!(is_allowed_dev_origin(b"http://localhost"));
        assert!(is_allowed_dev_origin(b"http://localhost:8402"));
        assert!(is_allowed_dev_origin(b"http://127.0.0.1:8402"));
        assert!(is_allowed_dev_origin(b"http://[::1]:8402"));
    }

    #[test]
    fn rejects_lookalike_and_other_origins() {
        // The exact bug this replaced: a `starts_with` check let any
        // subdomain of "localhost" through.
        assert!(!is_allowed_dev_origin(b"http://localhost.attacker.example"));
        assert!(!is_allowed_dev_origin(b"http://127.0.0.1.attacker.example"));
        assert!(!is_allowed_dev_origin(b"http://evil.example"));
        assert!(!is_allowed_dev_origin(b"https://localhost:8402"));
    }
}

#[tokio::main]
async fn main() {
    let studio = Arc::new(Studio::default());
    // Serves the Leptos `symthaea-muse-ui` build (produced by `trunk build`
    // into `crates/domains/symthaea-muse-ui/dist/`, relative to this
    // binary's cwd — the launcher runs it from `symthaea/`). Falls back to
    // `index.html` for any path ServeDir can't find on disk (Leptos's
    // client-side router paths — `/create`, `/research`, `/liked`,
    // `/atlas` — have no corresponding file, and a direct navigation or
    // refresh must still boot the SPA shell, not 404). The legacy
    // `studio/index.html`/`.css`/`.js` (still served at `/legacy` and
    // `/assets/muse-studio.{css,js}`, unchanged) remain reachable for
    // comparison and are still exercised by `ui_asset_tests` below, but
    // this is the surface a browser actually gets at `/`.
    let dist_dir = std::path::Path::new("crates/domains/symthaea-muse-ui/dist");
    let spa =
        ServeDir::new(dist_dir).not_found_service(ServeFile::new(dist_dir.join("index.html")));
    let app = Router::new()
        .route("/legacy", get(index))
        .route("/assets/muse-studio.css", get(studio_css))
        .route("/assets/muse-studio.js", get(studio_js))
        .route("/api/compose", post(compose))
        .route(
            "/api/cognitive/sonata-return",
            post(cognitive_sonata_return),
        )
        .route(
            "/api/cognitive/sonata-return/{id}/commit",
            post(commit_cognitive_sonata_return),
        )
        .route("/api/spec/{style}", get(spec_preset))
        .route("/api/specs", get(list_specs).post(save_spec))
        .route("/api/specs/{name}", get(load_spec))
        .route("/api/audio/{id}", get(audio))
        .route("/api/midi/{id}", get(midi))
        .route("/api/notes/{id}", get(notes))
        .route("/api/piece/{id}/listen-bundle", get(listen_bundle))
        .route("/api/motifs/{id}", get(motifs_summary))
        .route(
            "/api/piece/{id}/performance-bundle",
            get(performance_bundle),
        )
        .route("/api/piece/{id}/provenance", get(piece_provenance))
        .route("/api/piece/{id}/analyst", get(analyst_bundle))
        .route("/api/genealogy/{id}", get(genealogy_manifest))
        .route("/api/genealogy/{id}/children", get(genealogy_children))
        .route("/api/genealogy/{id}/ancestry", get(genealogy_ancestry))
        .route("/api/styles", get(listen_styles))
        .route("/api/atlas", get(atlas_summary))
        .route("/api/atlas/compare", get(atlas_compare))
        .route("/api/keeper/{id}", post(keeper))
        .route("/api/keepers", get(keepers))
        .route("/api/keeper-audio/{key}", get(keeper_audio))
        .route("/api/keeper-midi/{key}", get(keeper_midi))
        .route("/api/keeper-recipe/{key}", get(keeper_recipe))
        .route("/api/preview/{instrument}", get(instrument_preview))
        .fallback_service(spa)
        .layer(localhost_cors_layer())
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
    // A/B-tested decisively in favor of FluidSynth ("the sound no longer
    // fights the composition" — see fluid_render.rs); when this silently
    // falls back, every render uses the harsher in-crate synth with no
    // visible signal anywhere except the per-candidate `renderer` field.
    // Loud at startup so a broken SYMTHAEA_SOUNDFONT (empty, unset, wrong
    // path) is caught immediately instead of discovered by ear.
    match symthaea_muse::fluid_render::available() {
        Some((bin, sf)) => println!(
            "Renderer: FluidSynth ({}) + soundfont ({})",
            bin.display(),
            sf.display()
        ),
        None => println!(
            "Renderer: NATIVE FALLBACK — FluidSynth unavailable (set SYMTHAEA_SOUNDFONT to a \
             valid .sf2 path and ensure `fluidsynth` is on PATH or SYMTHAEA_FLUIDSYNTH). \
             Every render will use the harsher in-crate synth until this is fixed."
        ),
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

async fn studio_css() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
        include_str!("../../studio/muse-studio.css"),
    )
}

async fn studio_js() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
        include_str!("../../studio/muse-studio.js"),
    )
}

#[cfg(test)]
mod ui_asset_tests {
    const INDEX: &str = include_str!("../../studio/index.html");
    const CSS: &str = include_str!("../../studio/muse-studio.css");
    const JS: &str = include_str!("../../studio/muse-studio.js");

    #[test]
    fn shell_references_external_assets() {
        assert!(INDEX.contains("/assets/muse-studio.css"));
        assert!(INDEX.contains("/assets/muse-studio.js"));
        assert!(!INDEX.contains("<style>"));
        assert!(!INDEX.contains("<script>"));
    }

    #[test]
    fn extracted_assets_are_not_empty() {
        assert!(CSS.len() > 4_000);
        assert!(JS.len() > 20_000);
    }

    #[test]
    fn listen_shell_uses_real_intent_ranges_and_grounded_panels() {
        assert!(INDEX.contains("id=\"emotionArc\""));
        assert!(INDEX.contains("id=\"voiceBlend\""));
        assert!(INDEX.contains("id=\"researchView\""));
        assert!(JS.contains("function journeyIntent()"));
        assert!(!JS.contains("valence: 0.15, arousal: 0.45, energy: 0.5"));
    }

    #[test]
    fn research_overview_discloses_evidence_boundaries() {
        assert!(INDEX.contains("id=\"researchTimeline\""));
        assert!(INDEX.contains("id=\"analysisAvailability\""));
        assert!(INDEX.contains("remain score-side observations with explicit limitations"));
        assert!(JS.contains("function renderResearch()"));
        assert!(INDEX.contains("Musical integration is score analysis"));
    }

    #[test]
    fn product_language_and_motion_preferences_are_explicit() {
        assert!(INDEX.contains(">Guided</button>"));
        assert!(INDEX.contains(">Composer</button>"));
        assert!(INDEX.contains(">Advanced</button>"));
        assert!(JS.contains("Integration is score-structure analysis"));
        assert!(!JS.contains(" · Φ "));
        assert!(JS.contains("prefers-reduced-motion: reduce"));
        assert!(JS.contains("aria-current"));
    }
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

async fn cognitive_sonata_return(
    State(studio): State<Arc<Studio>>,
    Json(req): Json<SonataCognitiveRequest>,
) -> Result<Json<SonataCognitiveResponse>, (StatusCode, String)> {
    let intent = MusicalIntent {
        valence: req.valence.clamp(-1.0, 1.0),
        arousal: req.arousal.clamp(0.0, 1.0),
        energy: req.energy.clamp(0.0, 1.0),
        bars: req.bars.clamp(2, 16),
        seed: req.seed,
        tonic: PitchClass::new(req.tonic),
    };
    let spec = Style::Sonata.spec();
    let state = MusicalState {
        dopamine: req.dopamine.clamp(0.0, 1.0),
        serotonin: req.serotonin.clamp(0.0, 1.0),
        noradrenaline: req.noradrenaline.clamp(0.0, 1.0),
        consciousness_level: req.consciousness.clamp(0.0, 1.0),
        arousal: intent.arousal,
        valence: intent.valence,
        ..MusicalState::default()
    };

    let intent_for_compose = intent;
    let spec_for_compose = spec.clone();
    let mut realization = tokio::task::spawn_blocking(move || {
        compose_sonata_with_plan(&intent_for_compose, &spec_for_compose)
    })
    .await
    .map_err(internal)?
    .ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Sonata preset did not resolve to Sonata form".to_owned(),
        )
    })?;

    let perturbation = req.return_perturbation_semitones.clamp(-12, 12);
    if perturbation != 0 {
        perturb_primary_return(&mut realization, perturbation as i32)?;
    }

    let obligation = realization
        .plan
        .obligations
        .items()
        .iter()
        .find(|item| {
            matches!(
                &item.kind,
                ObligationKind::ReturnMotif { motif_id, .. }
                    if motif_id == "sonata.primary"
            )
        })
        .cloned()
        .ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Sonata primary-return obligation is missing".to_owned(),
            )
        })?;
    let target = realization
        .plan
        .sections
        .iter()
        .find(|section| section.kind == SonataSectionKind::RecapitulationPrimary)
        .cloned()
        .ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Sonata recapitulation-primary section is missing".to_owned(),
            )
        })?;
    let profile =
        profile_score_region(&realization.score, target.start, target.end).unwrap_or_default();
    let realization_for_session = realization.clone();
    let state_for_session = state.clone();
    let cognitive_session = tokio::task::spawn_blocking(move || {
        run_sonata_cognitive_session(
            &realization_for_session,
            &state_for_session,
            intent.seed,
            CognitiveSessionConfig::default(),
        )
    })
    .await
    .map_err(internal)?
    .map_err(|error| internal(format!("{error:?}")))?;
    let observation = cognitive_session.bridge_observation(
        &realization.plan.obligations,
        target.start,
        CognitiveSection::Recapitulation,
        Some(CognitiveGoal::Recapitulate),
        1.0,
    );
    let trace = propose_symbolic_action(cognitive_session.terminal_inference(), observation);
    let context = PredictionContext::new(
        trace.proposal.action,
        trace.observation.section,
        spec.name.clone(),
        "Sonata",
        spec.meter,
        TextureBand::from_active_voices(profile.active_voice_count),
    );
    let batch = {
        let model = studio.adaptive_outcomes.lock().unwrap();
        generate_and_rank_sonata_return_with_model(&realization, &trace, &model, context)
    }
    .map_err(|error| internal(format!("{error:?}")))?;
    let selected_alternative_id = batch.selection.recommended_id.clone().ok_or_else(|| {
        (
            StatusCode::UNPROCESSABLE_ENTITY,
            "no Sonata return candidate satisfied theory and Preserve constraints".to_owned(),
        )
    })?;
    let selected = batch
        .candidates
        .iter()
        .find(|candidate| candidate.alternative_id == selected_alternative_id)
        .cloned()
        .ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "selected Sonata candidate was not retained".to_owned(),
            )
        })?;

    let score_for_render = selected.score.clone();
    let spec_for_render = spec.clone();
    let state_for_render = state.clone();
    let seed = intent.seed;
    let (composition, fluid_wav) = tokio::task::spawn_blocking(move || {
        let composition = symthaea_muse::theory_realize::realize_with_spec(
            &score_for_render,
            &spec_for_render,
            seed,
            &state_for_render,
            SAMPLE_RATE,
        );
        let fluid_wav =
            fluidsynth_candidate_wav(&score_for_render, &spec_for_render, seed, &state_for_render);
        (composition, fluid_wav)
    })
    .await
    .map_err(internal)?;
    let renderer = if fluid_wav.is_some() {
        "fluidsynth"
    } else {
        "native"
    };
    let wav = match fluid_wav {
        Some(bytes) => bytes,
        None => wav_bytes(&composition.audio).map_err(internal)?,
    };

    let mut renderer_recipe = RendererRecipe::new(
        renderer,
        SAMPLE_RATE,
        env!("CARGO_PKG_VERSION"),
        symthaea_music_theory::MUSIC_THEORY_ENGINE_VERSION,
    );
    renderer_recipe.renderer_version = if renderer == "native" {
        Some(env!("CARGO_PKG_VERSION").to_owned())
    } else {
        option_env!("SYMTHAEA_RENDERER_VERSION").map(str::to_owned)
    };
    renderer_recipe.muse_source_revision = option_env!("SYMTHAEA_MUSE_GIT_REV").map(str::to_owned);
    renderer_recipe.theory_source_revision =
        option_env!("SYMTHAEA_MUSIC_THEORY_GIT_REV").map(str::to_owned);
    renderer_recipe.soundfont_sha256 = option_env!("SYMTHAEA_SOUNDFONT_SHA256").map(str::to_owned);
    renderer_recipe.renderer_binary_sha256 =
        option_env!("SYMTHAEA_RENDERER_BINARY_SHA256").map(str::to_owned);
    renderer_recipe.performance_model_sha256 =
        option_env!("SYMTHAEA_PERFORMANCE_MODEL_SHA256").map(str::to_owned);
    renderer_recipe.render_environment_sha256 =
        option_env!("SYMTHAEA_RENDER_ENVIRONMENT_SHA256").map(str::to_owned);

    let calibration = selected.prediction.clone();
    let mut selected_trace = trace.clone();
    selected_trace.predicted_outcome = calibration.calibrated;
    let mut recipe = PieceRecipe::new(intent, spec.clone(), renderer_recipe)
        .with_initial_musical_state(state.clone());
    let sequence = recipe.record_decision(selected_trace);
    {
        let decision = &mut recipe.cognitive_decisions[sequence as usize];
        decision
            .attach_cognitive_session(cognitive_session.clone())
            .map_err(|error| internal(format!("{error:?}")))?;
        decision.intervention_descriptor = Some(selected.descriptor.clone());
        decision.intervention_prediction = Some(calibration.clone());
        decision.policy_preference = Some(batch.selection.policy.clone());
    }
    recipe
        .record_preview(sequence, selected_alternative_id.clone())
        .map_err(|error| internal(format!("{error:?}")))?;
    let measurement = record_symbolic_region(
        &mut recipe,
        sequence,
        &realization.score,
        &selected.score,
        target.start,
        target.end,
    )
    .map_err(|error| internal(format!("{error:?}")))?;
    let observed_outcome = measurement.observed_outcome;
    let (model_intervention_context_samples_after, world_model_alternatives_learned) = {
        let mut model = studio.adaptive_outcomes.lock().unwrap();
        let mut updated = model.clone();
        let learned = learn_symbolic_outcomes_from_batch(&mut updated, &batch)
            .map_err(|error| internal(format!("{error:?}")))?;
        persist_adaptive_outcome_model(&updated).map_err(internal)?;
        let samples = updated
            .predict_intervention(&calibration.context)
            .intervention_context_samples;
        *model = updated;
        (samples, learned)
    };

    let alternatives = batch
        .candidates
        .iter()
        .map(|candidate| {
            let assessment = batch
                .selection
                .assessments
                .iter()
                .find(|assessment| assessment.alternative_id == candidate.alternative_id)
                .expect("every retained candidate has an assessment");
            let driving_obligation_verified = candidate
                .verification
                .iter()
                .find(|evidence| evidence.obligation_id == obligation.id)
                .map(|evidence| evidence.verified);
            SonataAlternativeMeta {
                alternative_id: candidate.alternative_id.clone(),
                strategy: candidate.strategy,
                selected: candidate.alternative_id == selected_alternative_id,
                motif_return_similarity: candidate.motif_return.overall_similarity,
                driving_obligation_verified,
                theory_valid: candidate.theory_validation.valid,
                theory_fatal_count: candidate.theory_validation.fatal_count(),
                theory_warning_count: candidate.theory_validation.warning_count(),
                preserved_invariants: candidate.preserved_invariants,
                policy_utility: assessment.outcome_utility,
                mean_prediction_error: assessment
                    .prediction_error
                    .map(|error| error.mean_absolute_error),
            }
        })
        .collect();

    let analysis = symthaea_music_theory::musical_phi(&selected.score);
    let ground = symthaea_music_theory::composer::ground_audition_for(&intent, &spec);
    let (grammar, ending) = symthaea_music_theory::composer::identity_grammar_for(&intent, &spec);
    let reproduction_gaps = recipe.reproduction_gaps();
    let candidate_id = studio.next_id.fetch_add(1, Ordering::Relaxed);
    {
        let mut store = studio.candidates.lock().unwrap();
        evict_oldest_candidates(&mut store);
        store.insert(
            candidate_id,
            Candidate {
                wav,
                created_at_unix_ms: unix_time_ms(),
                score: selected.score,
                spec,
                state,
                seed: intent.seed,
                renderer,
                phi: analysis.phi,
                local_coherence: analysis.mean_consonance_edge,
                global_coherence: analysis.mean_trigram_edge,
                ground,
                // The Sonata plan's score bypasses the period/Form pipeline
                // entirely (see compose_with_spec_and_form's doc comment),
                // so it never produces a Form.
                form: None,
                // This endpoint has its own dedicated, richer verification
                // (theory_validation/preserved_invariants/obligation
                // tracking, above) -- there's no matching
                // `GrammarPlanEvidence` variant for it, so the Analyst
                // endpoint's generic plan-based checks fall back to
                // synthesizing `Compatibility` for this candidate, same as
                // a user-authored custom spec.
                plan: None,
                grammar,
                ending,
                card: None,
                novelty: None,
                recipe,
            },
        );
    }

    Ok(Json(SonataCognitiveResponse {
        candidate_id,
        selected_alternative_id,
        driving_obligation_id: obligation.id,
        research_perturbation_semitones: perturbation,
        alternatives,
        selection_rationale: batch.selection.rationale,
        calibration,
        observed_outcome,
        model_intervention_context_samples_after,
        world_model_alternatives_learned,
        artist_response_recorded: false,
        renderer,
        duration_secs: composition.duration_secs,
        reproduction_gaps,
        cognitive_session_version: cognitive_session.session_version.clone(),
        cognitive_backend: cognitive_session.backend.clone(),
        cognitive_session_fingerprint: cognitive_session.session_fingerprint.clone(),
        cognitive_frame_count: cognitive_session.frames.len(),
        fep_rng_seed: cognitive_session.fep_rng_seed,
        fep_goal_preferences: cognitive_session.fep_goal_preferences.clone(),
        fep_goal_precision: cognitive_session.fep_goal_precision,
        fep_actions_committed: cognitive_session.fep_learning.committed_actions,
        fep_td_updates: cognitive_session.fep_learning.td_total_updates,
        fep_transition_history_size: cognitive_session.fep_learning.td_transition_history_size,
        terminal_fep_action: format!("{:?}", cognitive_session.terminal_inference.action),
        terminal_free_energy: cognitive_session.terminal_inference.free_energy,
    }))
}

async fn commit_cognitive_sonata_return(
    State(studio): State<Arc<Studio>>,
    AxPath(candidate_id): AxPath<u64>,
    Json(req): Json<SonataCommitRequest>,
) -> Result<Json<SonataCommitResponse>, (StatusCode, String)> {
    let mut store = studio.candidates.lock().unwrap();
    let candidate = store.get_mut(&candidate_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            "cognitive Sonata preview was not found".to_owned(),
        )
    })?;
    let sequence = candidate
        .recipe
        .cognitive_decisions
        .first()
        .map(|decision| decision.sequence)
        .ok_or_else(|| {
            (
                StatusCode::CONFLICT,
                "candidate has no cognitive decision to commit".to_owned(),
            )
        })?;
    let previewed = candidate.recipe.cognitive_decisions[sequence as usize]
        .selected_alternative_id
        .clone();
    let (disposition, selected_alternative_id) = match req.disposition {
        SonataCommitDisposition::Accepted => {
            let selected = req.alternative_id.or(previewed.clone()).ok_or_else(|| {
                (
                    StatusCode::BAD_REQUEST,
                    "accepted response must identify the previewed alternative".to_owned(),
                )
            })?;
            if previewed.as_deref() != Some(selected.as_str()) {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "this narrow commit endpoint can accept only the rendered preview".to_owned(),
                ));
            }
            (DecisionDisposition::Accepted, Some(selected))
        }
        SonataCommitDisposition::Edited => {
            let selected = req.alternative_id.or(previewed.clone()).ok_or_else(|| {
                (
                    StatusCode::BAD_REQUEST,
                    "edited response must identify the previewed alternative".to_owned(),
                )
            })?;
            if previewed.as_deref() != Some(selected.as_str()) {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "this narrow commit endpoint can edit only the rendered preview".to_owned(),
                ));
            }
            (DecisionDisposition::Edited, Some(selected))
        }
        SonataCommitDisposition::Rejected => (DecisionDisposition::Rejected, None),
    };
    candidate
        .recipe
        .record_artist_response(
            sequence,
            disposition,
            selected_alternative_id.clone(),
            req.artist_note,
        )
        .map_err(|error| internal(format!("{error:?}")))?;

    Ok(Json(SonataCommitResponse {
        candidate_id,
        disposition,
        selected_alternative_id,
        // V6 records explicit preference evidence but deliberately does not
        // train a preference model from a single interaction.
        preference_model_updated: false,
    }))
}

fn perturb_primary_return(
    realization: &mut SonataRealization,
    semitones: i32,
) -> Result<(), (StatusCode, String)> {
    let target = realization
        .plan
        .sections
        .iter()
        .find(|section| section.kind == SonataSectionKind::RecapitulationPrimary)
        .map(|section| (section.start, section.end))
        .ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Sonata recapitulation-primary section is missing".to_owned(),
            )
        })?;
    for note in &mut realization.score.notes {
        if note.role == VoiceRole::Melody
            && note.onset.beats() >= target.0.beats()
            && note.onset.beats() < target.1.beats()
        {
            note.pitch = note.pitch.transpose(semitones);
        }
    }
    let (resolution, verification) =
        verify_sonata_obligations(&realization.score, &realization.plan);
    realization.resolution = resolution;
    realization.verification = verification;
    Ok(())
}

async fn compose(
    State(studio): State<Arc<Studio>>,
    Json(req): Json<ComposeRequest>,
) -> Result<Json<ComposeResponse>, (StatusCode, String)> {
    if !COMPOSE_BARS_RANGE.contains(&req.bars) {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "bars must be between {} and {} (got {}) -- requests outside \
                 this range are rejected, never silently rescaled to a \
                 different value",
                COMPOSE_BARS_RANGE.start(),
                COMPOSE_BARS_RANGE.end(),
                req.bars
            ),
        ));
    }
    let n = req.n_candidates.clamp(1, MAX_CANDIDATES);
    let intent_base = MusicalIntent {
        valence: req.valence.clamp(-1.0, 1.0),
        arousal: req.arousal.clamp(0.0, 1.0),
        energy: req.energy.clamp(0.0, 1.0),
        bars: req.bars,
        seed: req.base_seed,
        tonic: PitchClass::new(req.tonic),
    };
    let style = req.style;
    let stride = req.seed_stride.clamp(1, 1_000);
    let prompt = req.prompt.trim().to_string();
    // Captured BEFORE `req.spec` is moved below: whether the CALLER
    // submitted their own spec, as distinct from the `spec` local variable
    // further down (which becomes `Some` unconditionally once the
    // identity-grammar override branch runs, even for a plain preset
    // request). Real grammar-family engines (GrooveCycle/ProcessAdditive/
    // RagaModalArc/CallResponse, and every other family's cadence/phrase-
    // grammar-aware routing) assume a `Style`-derived `GrammarProfile`
    // whose invariants (e.g. CallResponse's 12-bar-multiple progression)
    // a genuinely user-authored spec has no obligation to satisfy -- so
    // this stays the deciding signal for whether it's safe to route
    // through them below, not the post-override `spec` variable.
    let is_custom_spec = req.spec.is_some();
    // A user-authored spec replaces the style preset entirely.
    let spec = req.spec;
    // The identity-grammar override rewrites the effective form pool (and
    // therefore routes through the spec path even for preset styles).
    let spec = match req.grammar.as_deref().unwrap_or("Auto") {
        "Auto" => spec,
        g => {
            use symthaea_music_theory::FormKind;
            let mut s = spec.unwrap_or_else(|| style.spec());
            s.form_pool = match g {
                "Memory" => vec![FormKind::Ternary, FormKind::Rondo, FormKind::Variations],
                "Persistence" => vec![FormKind::Passacaglia],
                "Lineage" => vec![FormKind::Lineage],
                "Erosion" => vec![FormKind::Erosion],
                other => {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        format!("unknown identity grammar '{other}'"),
                    ));
                }
            };
            Some(s)
        }
    };
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
    // The Identity Explorer replaces the consecutive window for plain
    // Compose: candidates become maximally-different identities (the
    // artist's own seed always leads). Never offers less diversity than
    // the old window — the consecutive set is the explorer's fallback
    // baseline by construction.
    let exploring = req.explore && stride == 1 && n > 1;
    // See `ComposeRequest::vary_premise`'s doc comment: applies the same
    // premise mechanism to a single-candidate compose, which `exploring`
    // (an inherently multi-candidate concept — it picks AMONG seeds)
    // never covers on its own.
    let vary_single = req.vary_premise && n == 1;
    let seeds: Vec<u64> = if exploring {
        symthaea_music_theory::explorer::explore_identities(&spec_used, &intent_base, n as usize)
    } else {
        (0..n)
            .map(|i| intent_base.seed.wrapping_add(i.wrapping_mul(stride)))
            .collect()
    };
    // Per-seed novelty within the batch (explored batches only) — computed
    // on the BASE spec (same contract as identity cards) BEFORE the premise
    // layer below, so the pre-render novelty floor can react to it while
    // building `per_candidate`.
    let novelties: Option<Vec<symthaea_music_theory::explorer::NoveltyBreakdown>> = exploring
        .then(|| symthaea_music_theory::explorer::novelty_within(&spec_used, &intent_base, &seeds));
    // Cross-history novelty (MUSE_DIVERSITY_TRUTH_PLAN Phase 2): a batch can
    // look healthily diverse internally while every candidate quietly
    // clones a recent keeper. Featurize the last 20 keepers (each one's
    // stored recipe already carries the exact intent/resolved_spec that
    // composed it) into the same Identity space `novelty_within` uses, so
    // the floor check below reacts to history, not just batch peers.
    let recent_history = exploring
        .then(|| recent_keeper_identities(20))
        .unwrap_or_default();
    // THE PREMISE LAYER ("a seeded identity layer above note generation"):
    // when exploring (or a single compose opted into `vary_premise`), each
    // candidate composes under its OWN premise — tempo third, texture
    // budget, phrase length, ensemble persona, mode (where the style
    // opted in). Plain single composes and "More like this" keep the
    // authored spec untouched: the premise diversifies an OFFER, it never
    // rewrites a deliberate choice.
    let per_candidate: Vec<(u64, symthaea_music_theory::CompositionSpec, usize)> = seeds
        .iter()
        .enumerate()
        .map(|(i, &seed)| {
            let (seed, mut spec, bars_multiplier) = if exploring || vary_single {
                let p = symthaea_music_theory::premise::premise_for(&spec_used, seed);
                (seed, p.spec, p.bars_multiplier)
            } else {
                (seed, spec_used.clone(), 1)
            };
            // Pre-render novelty floor: a candidate below the floor (either
            // against its nearest BATCH neighbor, or against a recent
            // KEEPER) gets a real high-level structural variation (form/
            // motif-development/harmony/rhythm/orchestration/climax/
            // ending), deterministically keyed on its own seed via
            // `DiversityPlan` — not just a different seed feeding the same
            // premise mechanism.
            let below_batch_floor = novelties
                .as_ref()
                .is_some_and(|nv| nv[i].overall < NOVELTY_FLOOR);
            let below_history_floor = recent_history.iter().any(|(hist_spec, hist_intent)| {
                symthaea_music_theory::explorer::identity_distance(
                    &spec_used,
                    &intent_base,
                    seed,
                    hist_spec,
                    hist_intent,
                    hist_intent.seed,
                )
                .overall
                    < NOVELTY_FLOOR
            });
            if below_batch_floor || below_history_floor {
                symthaea_music_theory::diversity_plan::DiversityPlan::for_seed(&spec, seed)
                    .apply(&mut spec);
            }
            (seed, spec, bars_multiplier)
        })
        .collect();
    let per_candidate_render = per_candidate.clone();
    // Each candidate is an independent, CPU-bound compose+FluidSynth-render
    // (~4-6s apiece) — rendering them one after another in a single
    // blocking task made a 4-candidate batch take 15-25s wall-clock, long
    // enough to read as a hang under load. Spawned as N independent
    // blocking tasks instead: tokio's blocking pool runs them across
    // cores, so wall-clock tracks the SLOWEST candidate, not their sum.
    // `MusicalState` is small and Clone; nothing here shares mutable
    // state (each candidate's temp MIDI path is already seed-unique).
    // "native" forces the in-crate synthesizer even when FluidSynth is
    // available; anything else (including no preference at all) leaves
    // the server's existing auto-preference — FluidSynth when the
    // environment provides it — in place. See `ComposeRequest::renderer`.
    let force_native = req.renderer.as_deref() == Some("native");
    let mut handles = Vec::with_capacity(n as usize);
    for i in 0..n {
        let (seed, cand_spec, bars_mul) = per_candidate_render[i as usize].clone();
        let intent = MusicalIntent {
            seed,
            // A SEPARATE cap from `COMPOSE_BARS_RANGE`'s input validation:
            // `bars_mul` (from the premise layer) can INFLATE the
            // validated `intent_base.bars` well past what the caller
            // asked for, so this still needs its own ceiling -- just kept
            // in sync with the same range's upper bound rather than a
            // second, independent magic number (found out of sync
            // 2026-07-26: this site was still hardcoded to the OLD 16-bar
            // limit after `COMPOSE_BARS_RANGE` raised it to 36, silently
            // re-clamping a validated 36-bar request back down to 16).
            bars: (intent_base.bars * bars_mul).min(*COMPOSE_BARS_RANGE.end()),
            ..intent_base
        };
        let state = state.clone();
        handles.push(tokio::task::spawn_blocking(move || {
            // Compose once, not twice: `compose_and_realize_spec` already
            // calls `compose_with_spec` internally and discards the score
            // it produced — a second, separate `compose_with_spec` call
            // used to run here purely to get a `score` value back for
            // this handler's own use (phi/coherence analysis, /api/notes,
            // MIDI export). Composing is deterministic given the same
            // (intent, spec), so the second call was pure waste — every
            // candidate was being composed twice. Compose once and pass
            // the result into `realize_with_spec` directly.
            //
            // Route through `compose_with_grammar_plan` (the SAME entry
            // point `compose_styled`/`compose` use) whenever the request
            // is still preset-derived (`!is_custom_spec`) — this is what
            // actually reaches GrooveCycle/ProcessAdditive/RagaModalArc/
            // CallResponse's dedicated engines, and gives every OTHER
            // style the Jul22-24 grammar-family cadence/phrase-grammar
            // routing too. Found 2026-07-25: this handler previously
            // called `compose_with_spec_and_form` unconditionally, which
            // NEVER reaches `compose_with_grammar_plan`'s family dispatch
            // at all — so every dedicated grammar engine in this crate
            // was real, tested, and completely unreachable from the live
            // product. For a genuinely user-authored spec (`is_custom_
            // spec`), there is no `Style` to derive a `GrammarProfile`
            // from, so the plain spec path remains correct — a dedicated
            // engine's invariants (e.g. CallResponse's 12-bar-multiple
            // progression) aren't something arbitrary user data is
            // obligated to satisfy.
            let (score, form, plan) = if is_custom_spec {
                let (score, form) =
                    symthaea_music_theory::compose_with_spec_and_form(&intent, &cand_spec);
                (score, form, None)
            } else {
                let realized = symthaea_music_theory::compose_with_grammar_plan(
                    style.grammar_profile(),
                    &intent,
                    &cand_spec,
                );
                (realized.score, realized.form, Some(realized.plan))
            };
            let comp = symthaea_muse::theory_realize::realize_with_spec(
                &score,
                &cand_spec,
                intent.seed,
                &state,
                SAMPLE_RATE,
            );
            // Preferred render path: the performed MIDI through FluidSynth
            // (an A/B review settled it: "the sound no longer fights the
            // composition"). None → the native render above serves. Also
            // `None` outright when the request explicitly asked for the
            // native renderer (`ComposeRequest::renderer == "native"`).
            let fluid_wav = if force_native {
                None
            } else {
                fluidsynth_candidate_wav(&score, &cand_spec, seed, &state)
            };
            (seed, comp, score, fluid_wav, form, plan)
        }));
    }
    let _ = (&spec, style); // kept alive for the request's lifetime; unused past dispatch
    let mut rendered = Vec::with_capacity(n as usize);
    for h in handles {
        rendered.push(
            h.await
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?,
        );
    }

    // Optional CLAP ranking. Degrades gracefully (feature off, no ORT, no
    // network) — the UI shows WHY instead of silently dropping the scores.
    let (similarities, ranking_note) = rank(&prompt, &rendered);

    let mut metas = Vec::with_capacity(rendered.len());
    // Exact-duplicate detection within this batch: different seeds/premises
    // can still land on byte-identical music (archetype styles whose
    // harmonic channel is structurally near-zero are the known case —
    // `explorer.rs`'s own tests document this). Tracks the FIRST id seen
    // for each score fingerprint so later duplicates in the batch all point
    // back to the same original, not to each other.
    let mut seen_fingerprints: HashMap<u64, u64> = HashMap::new();
    {
        let mut store = studio.candidates.lock().unwrap();
        evict_oldest_candidates(&mut store); // session-scale memory bound
        for (idx, (seed, comp, score, fluid_wav, form, plan)) in rendered.into_iter().enumerate() {
            let id = studio.next_id.fetch_add(1, Ordering::Relaxed);
            let fingerprint = symthaea_music_theory::fingerprint::exact_fingerprint(&score);
            let duplicate_of = mark_duplicate(&mut seen_fingerprints, fingerprint, id);
            let renderer = if fluid_wav.is_some() {
                "fluidsynth"
            } else {
                "native"
            };
            let wav = match fluid_wav {
                Some(w) => w,
                None => wav_bytes(&comp.audio).map_err(internal)?,
            };
            let analysis = symthaea_music_theory::musical_phi(&score);
            let (phi, local_coherence, global_coherence) = (
                analysis.phi,
                analysis.mean_consonance_edge,
                analysis.mean_trigram_edge,
            );
            let cand_spec = per_candidate[idx].1.clone();
            let candidate_intent = MusicalIntent {
                seed,
                // Must match the spawn_blocking closure's own bars
                // computation above exactly -- this reconstructs the same
                // per-candidate intent for the metadata/title/novelty
                // pipeline below, not a second independent decision.
                bars: (intent_base.bars * per_candidate[idx].2).min(*COMPOSE_BARS_RANGE.end()),
                ..intent_base
            };
            let ground =
                symthaea_music_theory::composer::ground_audition_for(&candidate_intent, &cand_spec);
            let (grammar, ending) = symthaea_music_theory::composer::identity_grammar_for(
                &candidate_intent,
                &cand_spec,
            );
            // From the BASE spec (identity_card derives the premise itself
            // — see its contract), and only when a premise was applied.
            let card = exploring.then(|| {
                symthaea_music_theory::describe::identity_card(&spec_used, &candidate_intent, seed)
            });
            // Deterministic title grammar v2. The title is a poetic label,
            // while `title_recipe` records the real traits that constrained
            // its vocabulary and offers stable alternatives without
            // recomposing the piece.
            let generated_title = symthaea_music_theory::describe::title_recipe(
                &cand_spec,
                &candidate_intent,
                seed,
                grammar,
                ending,
            );
            let title = card
                .as_ref()
                .map(|c| c.title.clone())
                .unwrap_or_else(|| generated_title.generated_title.clone());
            let title_recipe = title_recipe_summary(&generated_title);
            // Always present, unlike `card`: reads only the resolved spec
            // (grammar/development/accompaniment/texture), which is well-
            // defined for every compose — premised or not.
            let why = symthaea_music_theory::describe::why_lines(&cand_spec, grammar, ending, seed);
            let novelty = novelties.as_ref().map(|n| n[idx]);
            let mut renderer_recipe = RendererRecipe::new(
                renderer,
                SAMPLE_RATE,
                env!("CARGO_PKG_VERSION"),
                symthaea_music_theory::MUSIC_THEORY_ENGINE_VERSION,
            );
            renderer_recipe.renderer_version = if renderer == "native" {
                Some(env!("CARGO_PKG_VERSION").to_owned())
            } else {
                option_env!("SYMTHAEA_RENDERER_VERSION").map(str::to_owned)
            };
            renderer_recipe.muse_source_revision =
                option_env!("SYMTHAEA_MUSE_GIT_REV").map(str::to_owned);
            renderer_recipe.theory_source_revision =
                option_env!("SYMTHAEA_MUSIC_THEORY_GIT_REV").map(str::to_owned);
            renderer_recipe.soundfont_sha256 =
                option_env!("SYMTHAEA_SOUNDFONT_SHA256").map(str::to_owned);
            renderer_recipe.renderer_binary_sha256 =
                option_env!("SYMTHAEA_RENDERER_BINARY_SHA256").map(str::to_owned);
            renderer_recipe.performance_model_sha256 =
                option_env!("SYMTHAEA_PERFORMANCE_MODEL_SHA256").map(str::to_owned);
            renderer_recipe.render_environment_sha256 =
                option_env!("SYMTHAEA_RENDER_ENVIRONMENT_SHA256").map(str::to_owned);
            let recipe = PieceRecipe::new(candidate_intent, cand_spec.clone(), renderer_recipe)
                .with_initial_musical_state(state_used.clone());
            let identity = symthaea_muse_protocol::ArtifactIdentity {
                score_content: symthaea_muse_protocol::ScoreContentArtifactId(
                    serialized_sha256(&score).unwrap_or_default(),
                ),
                composition: symthaea_muse_protocol::CompositionArtifactId(
                    serialized_sha256(&recipe).unwrap_or_default(),
                ),
                rendition: symthaea_muse_protocol::RenditionArtifactId(sha256_hex(&wav)),
            };
            metas.push(CandidateMeta {
                id,
                seed,
                duration_secs: comp.duration_secs,
                similarity: similarities.as_ref().map(|s| s[idx]),
                renderer,
                phi,
                local_coherence,
                global_coherence,
                ground,
                grammar,
                ending,
                card: card.clone(),
                title,
                title_recipe,
                why,
                meter: cand_spec.meter,
                novelty,
                style: cand_spec.name.clone(),
                duplicate_of,
                identity,
            });
            store.insert(
                id,
                Candidate {
                    wav,
                    created_at_unix_ms: unix_time_ms(),
                    score,
                    spec: cand_spec,
                    state: state_used.clone(),
                    seed,
                    renderer,
                    phi,
                    local_coherence,
                    global_coherence,
                    ground,
                    form,
                    plan,
                    grammar,
                    ending,
                    card,
                    novelty,
                    recipe,
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
    rendered: &[(
        u64,
        symthaea_muse::Composition,
        Score,
        Option<Vec<u8>>,
        Option<symthaea_music_theory::Form>,
        Option<GrammarPlanEvidence>,
    )],
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
    for (_, comp, _, _, _, _) in rendered {
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
    _rendered: &[(
        u64,
        symthaea_muse::Composition,
        Score,
        Option<Vec<u8>>,
        Option<symthaea_music_theory::Form>,
        Option<GrammarPlanEvidence>,
    )],
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
    headers: axum::http::HeaderMap,
) -> Result<axum::response::Response, StatusCode> {
    use axum::response::IntoResponse as _;
    let wav = {
        let store = studio.candidates.lock().unwrap();
        store.get(&id).ok_or(StatusCode::NOT_FOUND)?.wav.clone()
    };
    let total = wav.len() as u64;
    // HTTP Range support: without it browsers treat the WAV as an
    // unseekable stream — the transport bar wouldn't scrub and the
    // piano-roll's click-to-seek silently failed. One satisfiable range
    // per request (the only shape media elements send).
    if let Some(range) = headers
        .get(header::RANGE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("bytes="))
    {
        let mut parts = range.splitn(2, '-');
        let start: u64 = parts.next().unwrap_or("").parse().unwrap_or(0);
        let end: u64 = parts
            .next()
            .filter(|e| !e.is_empty())
            .and_then(|e| e.parse().ok())
            .map(|e: u64| e.min(total.saturating_sub(1)))
            .unwrap_or(total.saturating_sub(1));
        if start > end || start >= total {
            return Ok((
                StatusCode::RANGE_NOT_SATISFIABLE,
                [(header::CONTENT_RANGE, format!("bytes */{total}"))],
            )
                .into_response());
        }
        let body = wav[start as usize..=end as usize].to_vec();
        return Ok((
            StatusCode::PARTIAL_CONTENT,
            [
                (header::CONTENT_TYPE, "audio/wav".to_string()),
                (header::ACCEPT_RANGES, "bytes".to_string()),
                (
                    header::CONTENT_RANGE,
                    format!("bytes {start}-{end}/{total}"),
                ),
            ],
            body,
        )
            .into_response());
    }
    Ok((
        [
            (header::CONTENT_TYPE, "audio/wav".to_string()),
            (header::ACCEPT_RANGES, "bytes".to_string()),
        ],
        wav,
    )
        .into_response())
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

/// Construct a collision-resistant, path-safe keeper version identifier.
fn keeper_artifact_key(
    unix_nanos: u128,
    process_id: u32,
    candidate_id: u64,
    sequence: u64,
) -> String {
    format!("{unix_nanos:x}_{process_id:x}_{candidate_id:x}_{sequence:x}")
}

fn write_synced(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

fn atomic_replace_file(path: &Path, bytes: &[u8], sequence: u64) -> std::io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("keeper-index");
    let temporary = parent.join(format!(
        ".{file_name}.tmp-{}-{sequence}",
        std::process::id()
    ));
    let result = (|| {
        write_synced(&temporary, bytes)?;
        std::fs::rename(&temporary, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

fn persist_keeper_bundle(
    audio_key: &str,
    wav: &[u8],
    recipe_bytes: &[u8],
    score: &Score,
    spec: &CompositionSpec,
    seed: u64,
    state: &MusicalState,
    sequence: u64,
) -> std::io::Result<(PathBuf, bool)> {
    let root = Path::new("data/taste/audio");
    std::fs::create_dir_all(root)?;
    let staging = root.join(format!(
        ".tmp-{audio_key}-{}-{sequence}",
        std::process::id()
    ));
    let destination = root.join(audio_key);
    std::fs::create_dir(&staging)?;

    let result = (|| {
        write_synced(&staging.join("audio.wav"), wav)?;
        write_synced(&staging.join("recipe.json"), recipe_bytes)?;

        let midi_path = staging.join("performance.mid");
        let midi_available = symthaea_muse::midi_export::export_performance_midi(
            score, spec, seed, state, &midi_path,
        )
        .is_ok();
        if midi_available {
            std::fs::File::open(&midi_path)?.sync_all()?;
        } else {
            let _ = std::fs::remove_file(&midi_path);
        }

        // The directory rename publishes the complete canonical keeper bundle
        // in one filesystem operation. Existing flat-file keepers remain
        // readable through the compatibility paths below.
        std::fs::rename(&staging, &destination)?;
        Ok((destination.clone(), midi_available))
    })();

    if result.is_err() {
        let _ = std::fs::remove_dir_all(&staging);
    }
    result
}

/// Whether `data/taste/keepers.jsonl` already has an entry whose
/// `score_sha256` matches `hash` — the exact-duplicate dedup key, since
/// different recipes can provably produce identical music (see
/// `explorer.rs`'s `same_score_different_recipe` test). Malformed lines are
/// skipped, matching `keepers()`'s existing fault-tolerant read.
fn keeper_log_has_score_hash(hash: &str) -> bool {
    keeper_log_has_score_hash_at(Path::new("data/taste/keepers.jsonl"), hash)
}

fn keeper_log_has_score_hash_at(path: &Path, hash: &str) -> bool {
    let Ok(text) = std::fs::read_to_string(path) else {
        return false;
    };
    text.lines().any(|line| {
        serde_json::from_str::<serde_json::Value>(line)
            .ok()
            .and_then(|value| value.get("score_sha256")?.as_str().map(str::to_owned))
            .is_some_and(|existing| existing == hash)
    })
}

/// The most recent `limit` keepers' `(resolved_spec, intent)`, most-recent-
/// first — the featurized-history side of the cross-history novelty floor
/// (`MUSE_DIVERSITY_TRUTH_PLAN_2026-07-18.md` Phase 2: "featurize recent
/// keepers... into the same Identity space"). Each keeper's stored
/// `recipe` already carries the exact `intent`/`resolved_spec` that
/// composed it (`PieceRecipe`), so this needs no recomposition. Entries
/// whose `recipe` field is missing or fails to parse (older schema
/// generations) are skipped rather than failing the whole read, matching
/// `keepers()`'s existing fault tolerance.
fn recent_keeper_identities(
    limit: usize,
) -> Vec<(symthaea_music_theory::CompositionSpec, MusicalIntent)> {
    recent_keeper_identities_at(Path::new("data/taste/keepers.jsonl"), limit)
}

fn recent_keeper_identities_at(
    path: &Path,
    limit: usize,
) -> Vec<(symthaea_music_theory::CompositionSpec, MusicalIntent)> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut recipes: Vec<PieceRecipe> = text
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter_map(|entry| entry.get("recipe").cloned())
        .filter_map(|recipe| serde_json::from_value::<PieceRecipe>(recipe).ok())
        .collect();
    recipes.reverse();
    recipes.truncate(limit);
    recipes
        .into_iter()
        .map(|r| (r.resolved_spec, r.intent))
        .collect()
}

#[cfg(test)]
mod keeper_dedup_tests {
    use super::*;

    #[test]
    fn absent_file_reports_no_match() {
        let dir = std::env::temp_dir().join(format!(
            "muse_studio_dedup_test_missing_{}",
            std::process::id()
        ));
        assert!(!keeper_log_has_score_hash_at(
            &dir.join("keepers.jsonl"),
            "abc"
        ));
    }

    #[test]
    fn finds_a_matching_hash_and_ignores_non_matching_and_malformed_lines() {
        let dir = std::env::temp_dir().join(format!(
            "muse_studio_dedup_test_{}_{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("keepers.jsonl");
        std::fs::write(
            &path,
            "not json at all\n\
             {\"score_sha256\": \"other-hash\"}\n\
             {\"score_sha256\": \"target-hash\"}\n",
        )
        .unwrap();
        assert!(keeper_log_has_score_hash_at(&path, "target-hash"));
        assert!(!keeper_log_has_score_hash_at(&path, "absent-hash"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn recent_keeper_identities_reads_recipes_most_recent_first_and_respects_limit() {
        let dir = std::env::temp_dir().join(format!(
            "muse_studio_recent_identities_test_{}_{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("keepers.jsonl");
        let renderer = RendererRecipe::new("test", 48_000, "0.0.0", "0.0.0");
        let make_line = |seed: u64| {
            let recipe = PieceRecipe::new(
                MusicalIntent {
                    seed,
                    ..Default::default()
                },
                symthaea_music_theory::Style::Classical.spec(),
                renderer.clone(),
            );
            serde_json::json!({ "seed": seed, "recipe": recipe }).to_string()
        };
        std::fs::write(
            &path,
            format!(
                "not json at all\n{}\n{{\"no_recipe_field\": true}}\n{}\n{}\n",
                make_line(1),
                make_line(2),
                make_line(3)
            ),
        )
        .unwrap();
        let all = recent_keeper_identities_at(&path, 10);
        assert_eq!(all.len(), 3, "malformed/missing-recipe lines are skipped");
        assert_eq!(
            all.iter().map(|(_, i)| i.seed).collect::<Vec<_>>(),
            vec![3, 2, 1],
            "most-recent-first"
        );
        assert_eq!(
            recent_keeper_identities_at(&path, 2).len(),
            2,
            "limit is respected"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn recent_keeper_identities_missing_file_is_empty() {
        let dir = std::env::temp_dir().join(format!(
            "muse_studio_recent_identities_missing_{}",
            std::process::id()
        ));
        assert!(recent_keeper_identities_at(&dir.join("keepers.jsonl"), 10).is_empty());
    }
}

fn append_keeper_entry_atomically(entry: &serde_json::Value, sequence: u64) -> std::io::Result<()> {
    let path = Path::new("data/taste/keepers.jsonl");
    let mut contents = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => return Err(error),
    };
    if !contents.is_empty() && !contents.ends_with(b"\n") {
        contents.push(b'\n');
    }
    serde_json::to_writer(&mut contents, entry)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    contents.push(b'\n');
    atomic_replace_file(path, &contents, sequence)
}

/// Mark a candidate as a KEEPER — the start of the taste dataset. Every
/// keep atomically publishes a versioned artifact directory and then replaces
/// `data/taste/keepers.jsonl` with the prior entries plus one complete line.
/// The index is never made visible before its WAV and recipe exist.
/// What the caller actually needs back from a keep: enough to look the
/// piece up again without guessing (`genealogy` carries its own `id`, the
/// stable identity — never the ephemeral candidate `id` this request came
/// in on). `already_kept` distinguishes "kept just now" from "was already
/// in the log" without changing the response shape between the two.
#[derive(Serialize)]
struct KeeperResponse {
    already_kept: bool,
    audio_key: Option<String>,
    recipe_sha256: Option<String>,
    score_sha256: Option<String>,
    genealogy: Option<symthaea_muse_protocol::GenealogyManifest>,
}

async fn keeper(
    State(studio): State<Arc<Studio>>,
    AxPath(id): AxPath<u64>,
) -> Result<Response, StatusCode> {
    tokio::task::spawn_blocking(move || keeper_sync(&studio, id))
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
}

/// The actual keep operation: sqlite writes, filesystem writes, and a
/// whole-file JSONL rewrite, none of which are `async` -- this used to run
/// directly in `keeper`'s `async fn` body (no `.await` anywhere in it),
/// which meant every keep blocked whatever tokio worker thread drew it for
/// its entire duration. Run via `spawn_blocking` instead (`keeper`, above),
/// matching this file's own existing convention for CPU/IO-bound work
/// (`compose`'s per-candidate render already uses `spawn_blocking`).
fn keeper_sync(studio: &Studio, id: u64) -> Result<Response, StatusCode> {
    let (
        spec,
        seed,
        renderer,
        phi,
        local_coherence,
        global_coherence,
        ground,
        grammar,
        ending,
        title,
        novelty,
        wav,
        score,
        form,
        state,
        recipe,
    ) = {
        let store = studio.candidates.lock().unwrap();
        let c = store.get(&id).ok_or(StatusCode::NOT_FOUND)?;
        (
            c.spec.clone(),
            c.seed,
            c.renderer,
            c.phi,
            c.local_coherence,
            c.global_coherence,
            c.ground,
            c.grammar,
            c.ending,
            c.card.as_ref().map(|k| k.title.clone()),
            c.novelty,
            c.wav.clone(),
            c.score.clone(),
            c.form.clone(),
            c.state.clone(),
            c.recipe.clone(),
        )
    };
    if !recipe.is_valid() {
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    // Exact-duplicate dedup: a candidate already kept under a different id
    // (or a double-click on the same candidate) must not append a second
    // keeper entry for the identical score. `serialized_sha256` failing is
    // not itself a reason to refuse the keep — dedup degrades to "assume
    // not a duplicate" rather than blocking the user's action.
    if let Ok(score_hash) = serialized_sha256(&score) {
        if keeper_log_has_score_hash(&score_hash) {
            return Ok(Json(KeeperResponse {
                already_kept: true,
                audio_key: None,
                recipe_sha256: None,
                score_sha256: Some(score_hash),
                genealogy: None,
            })
            .into_response());
        }
    }

    let hook =
        symthaea_music_theory::HookCell::generate_with(&spec.melody, seed, spec.meter as f64);
    let duration = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let ts = duration.as_secs();
    let keeper_sequence = studio.next_keeper_id.fetch_add(1, Ordering::Relaxed);
    let audio_key =
        keeper_artifact_key(duration.as_nanos(), std::process::id(), id, keeper_sequence);
    debug_assert!(valid_artifact_key(&audio_key));

    let recipe_bytes =
        serde_json::to_vec_pretty(&recipe).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let (bundle_path, midi_available) = persist_keeper_bundle(
        &audio_key,
        &wav,
        &recipe_bytes,
        &score,
        &spec,
        seed,
        &state,
        keeper_sequence,
    )
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Genealogy is an enrichment, not a requirement -- a ledger failure
    // (missing store, sqlite error) must not block the keep the user asked
    // for. Every manifest allocated here is a `Root`: nothing upstream yet
    // tracks "this candidate was derived from kept piece X" (see
    // `symthaea_muse::genealogy`'s module docs).
    let recipe_sha256 = serialized_sha256(&recipe).ok();
    let score_sha256 = serialized_sha256(&score).ok();
    let genealogy = studio.genealogy.as_ref().and_then(|store| {
        let Some(recipe_sha256) = recipe_sha256.as_deref() else {
            eprintln!("[muse_studio] genealogy: could not hash recipe");
            return None;
        };
        let origin = symthaea_muse_protocol::GenealogyOrigin::MuseGenerated {
            seed,
            style_name: spec.name.clone(),
        };
        match store.allocate_root(
            &origin,
            &audio_key,
            recipe_sha256,
            score_sha256.as_deref(),
            &sha256_hex(&wav),
            unix_time_ms(),
        ) {
            Ok(manifest) => Some(manifest),
            Err(error) => {
                eprintln!("[muse_studio] genealogy: allocate_root failed: {error}");
                None
            }
        }
    });

    let entry = serde_json::json!({
        "ts": ts,
        "seed": seed,
        "spec": spec.name,
        "mode": spec.mode.map(|m| format!("{m:?}")),
        "ensemble": spec.ensemble(seed),
        "renderer": renderer,
        "phi": phi,
        "local_coherence": local_coherence,
        "global_coherence": global_coherence,
        "ground_worthiness": ground,
        "grammar": grammar,
        "ending": ending,
        "title": title,
        "novelty": novelty,
        "audio_key": audio_key.clone(),
        "artifact_layout": "keeper-directory-v1",
        "midi_available": midi_available,
        "genealogy_id": genealogy.as_ref().map(|m| m.id),
        "score_sha256": score_sha256.clone(),
        // Persisted so future Atlas reads don't depend on recompose
        // determinism across engine versions (MUSE_DIVERSITY_TRUTH_PLAN
        // Phase 3's "persist fingerprints at keep time"). Dimension count
        // travels alongside so a later schema change can be detected
        // rather than silently mixed with older entries.
        "structural_fingerprint":
            symthaea_music_theory::fingerprint::structural_fingerprint(&score, &form).to_vec(),
        "structural_fingerprint_dims": symthaea_music_theory::fingerprint::STRUCT_DIMS,
        "reproduction_gaps": recipe.reproduction_gaps(),
        "recipe": recipe,
        "hook": hook
            .notes
            .iter()
            .map(|(deg, dur)| (*deg, dur.beats()))
            .collect::<Vec<_>>(),
    });

    let index_result = {
        let _guard = studio
            .keeper_log
            .lock()
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        append_keeper_entry_atomically(&entry, keeper_sequence)
    };
    if index_result.is_err() {
        let _ = std::fs::remove_dir_all(bundle_path);
        // The keep itself failed downstream of genealogy allocation --
        // don't leave a manifest referencing an audio_key whose bundle
        // directory was just deleted and whose JSONL entry never landed.
        if let (Some(store), Some(manifest)) = (studio.genealogy.as_ref(), genealogy.as_ref()) {
            if let Err(error) = store.delete(manifest.id) {
                eprintln!(
                    "[muse_studio] genealogy: failed to roll back manifest {} after keep failure: {error}",
                    manifest.id
                );
            }
        }
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }
    Ok(Json(KeeperResponse {
        already_kept: false,
        audio_key: Some(audio_key),
        recipe_sha256,
        score_sha256,
        genealogy,
    })
    .into_response())
}

/// Every kept piece, most-recent-first — the "Liked Songs" view's data
/// source. Reads the same jsonl the keep endpoint appends to; malformed
/// lines (there shouldn't be any, but the file is hand-append-only) are
/// skipped rather than failing the whole list.
async fn keepers() -> Result<impl IntoResponse, StatusCode> {
    let text = match std::fs::read_to_string("data/taste/keepers.jsonl") {
        Ok(t) => t,
        Err(_) => return Ok(axum::Json(Vec::<serde_json::Value>::new())),
    };
    let mut entries: Vec<serde_json::Value> = text
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    entries.reverse();
    entries.truncate(200);
    Ok(axum::Json(entries))
}

fn valid_artifact_key(key: &str) -> bool {
    !key.is_empty()
        && key.len() <= 80
        && key
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
}

fn kept_artifact_path(
    key: &str,
    nested_name: &str,
    legacy_suffix: &str,
) -> Result<PathBuf, StatusCode> {
    if !valid_artifact_key(key) {
        return Err(StatusCode::BAD_REQUEST);
    }
    let nested = Path::new("data/taste/audio").join(key).join(nested_name);
    if nested.is_file() {
        return Ok(nested);
    }
    Ok(Path::new("data/taste/audio").join(format!("{key}{legacy_suffix}")))
}

/// Serve a kept piece's actual saved audio (not a recomposed approximation).
async fn keeper_audio(AxPath(key): AxPath<String>) -> Result<impl IntoResponse, StatusCode> {
    let path = kept_artifact_path(&key, "audio.wav", ".wav")?;
    let bytes = std::fs::read(path).map_err(|_| StatusCode::NOT_FOUND)?;
    Ok((
        [
            (header::CONTENT_TYPE, "audio/wav".to_string()),
            (header::ACCEPT_RANGES, "bytes".to_string()),
        ],
        bytes,
    ))
}

/// Serve a kept piece's saved performed MIDI.
async fn keeper_midi(AxPath(key): AxPath<String>) -> Result<impl IntoResponse, StatusCode> {
    let path = kept_artifact_path(&key, "performance.mid", ".mid")?;
    let bytes = std::fs::read(path).map_err(|_| StatusCode::NOT_FOUND)?;
    Ok((
        [
            (header::CONTENT_TYPE, "audio/midi".to_string()),
            (
                header::CONTENT_DISPOSITION,
                format!("attachment; filename=\"muse_{key}.mid\""),
            ),
        ],
        bytes,
    ))
}

/// Serve the canonical recipe sidecar for a kept piece.
async fn keeper_recipe(AxPath(key): AxPath<String>) -> Result<impl IntoResponse, StatusCode> {
    let path = kept_artifact_path(&key, "recipe.json", ".recipe.json")?;
    let bytes = std::fs::read(path).map_err(|_| StatusCode::NOT_FOUND)?;
    Ok((
        [
            (header::CONTENT_TYPE, "application/json".to_string()),
            (
                header::CONTENT_DISPOSITION,
                format!("attachment; filename=\"muse_{key}.recipe.json\""),
            ),
        ],
        bytes,
    ))
}

/// A short (~3.5s) audio preview of a single instrument — a one-octave
/// scale up to a held tonic, entirely decoupled from composition. Lets
/// the Studio's per-voice pickers be auditioned before committing to a
/// choice. Rendered once per instrument name and cached to disk
/// (`data/previews/{name}.wav`), so repeat previews cost nothing beyond
/// the file read; the render itself never touches Score/CompositionSpec.
async fn instrument_preview(AxPath(name): AxPath<String>) -> Result<impl IntoResponse, StatusCode> {
    let instrument =
        symthaea_muse::instruments::Instrument::from_name(&name).ok_or(StatusCode::NOT_FOUND)?;
    let cache_path = format!("data/previews/{name}.wav");
    if let Ok(bytes) = std::fs::read(&cache_path) {
        return Ok((
            [
                (header::CONTENT_TYPE, "audio/wav".to_string()),
                (header::ACCEPT_RANGES, "bytes".to_string()),
            ],
            bytes,
        ));
    }
    symthaea_muse::fluid_render::available().ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let midi_path =
        std::env::temp_dir().join(format!("muse_preview_{name}_{}.mid", std::process::id()));
    symthaea_muse::midi_export::export_preview_midi(instrument, &midi_path)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let wav = symthaea_muse::fluid_render::render_midi_to_wav(&midi_path, SAMPLE_RATE, None);
    let _ = std::fs::remove_file(&midi_path);
    let wav = wav.ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;
    if let Some(parent) = std::path::Path::new(&cache_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&cache_path, &wav);
    Ok((
        [
            (header::CONTENT_TYPE, "audio/wav".to_string()),
            (header::ACCEPT_RANGES, "bytes".to_string()),
        ],
        wav,
    ))
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    format!("{:x}", sha2::Sha256::digest(bytes))
}

fn serialized_sha256<T: Serialize>(value: &T) -> Result<String, StatusCode> {
    serde_json::to_vec(value)
        .map(|bytes| sha256_hex(&bytes))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

fn render_id(id: u64, wav: &[u8]) -> String {
    let digest = sha256_hex(wav);
    format!("candidate-{id}-{}", &digest[..12])
}

fn musical_time(beats: f64, tempo_bpm: f32) -> MusicalTime {
    MusicalTime {
        tick: (beats.max(0.0) * TICKS_PER_BEAT as f64).round() as u64,
        beats,
        seconds: beats * 60.0 / tempo_bpm.max(1.0) as f64,
    }
}

fn role_slug(role: VoiceRole) -> &'static str {
    match role {
        VoiceRole::Melody => "melody",
        VoiceRole::Harmony => "harmony",
        VoiceRole::Bass => "bass",
        VoiceRole::CounterMelody => "counter",
    }
}

fn role_label(role: VoiceRole) -> &'static str {
    match role {
        VoiceRole::Melody => "Melody",
        VoiceRole::Harmony => "Harmony",
        VoiceRole::Bass => "Bass",
        VoiceRole::CounterMelody => "Counter melody",
    }
}

fn emphasis_label(emphasis: Emphasis) -> &'static str {
    match emphasis {
        Emphasis::Normal => "normal",
        Emphasis::Climax => "climax",
        Emphasis::Cadential => "cadential",
        Emphasis::PhraseStart => "phrase-start",
    }
}

fn symbolic_note_id(role: VoiceRole, role_index: usize) -> String {
    format!("score-{}-{role_index}", role_slug(role))
}

fn score_note_event(
    role: VoiceRole,
    role_index: usize,
    note: ScoreNote,
    tempo: f32,
) -> SymbolicNoteEvent {
    let beats = note.onset.beats();
    let duration_beats = note.duration.beats();
    let pitch = note.pitch;
    SymbolicNoteEvent {
        id: symbolic_note_id(role, role_index),
        midi: pitch.midi(),
        pitch_name: format!("{}{}", pitch.pitch_class().name(), pitch.octave()),
        onset: musical_time(beats, tempo),
        duration_ticks: (duration_beats * TICKS_PER_BEAT as f64).round() as u64,
        duration_beats,
        duration_seconds: duration_beats * 60.0 / tempo.max(1.0) as f64,
        velocity: note.velocity,
        voice_role: role_label(role).to_string(),
        emphasis: emphasis_label(note.emphasis).to_string(),
        section_intensity: note.section_intensity,
    }
}

fn symbolic_notes(score: &Score) -> Vec<SymbolicNoteEvent> {
    let mut events = Vec::with_capacity(score.notes.len());
    for role in [
        VoiceRole::Bass,
        VoiceRole::Harmony,
        VoiceRole::CounterMelody,
        VoiceRole::Melody,
    ] {
        events.extend(
            score
                .voice(role)
                .into_iter()
                .enumerate()
                .map(|(index, note)| score_note_event(role, index, note, score.tempo_bpm)),
        );
    }
    events.sort_by(|a, b| {
        a.onset
            .beats
            .total_cmp(&b.onset.beats)
            .then_with(|| a.voice_role.cmp(&b.voice_role))
            .then_with(|| a.midi.cmp(&b.midi))
    });
    events
}

fn expected_section_labels(
    form: FormKind,
    count: usize,
) -> Option<Vec<(&'static str, &'static str)>> {
    let labels: &[(&str, &str)] = match form {
        FormKind::Ternary if count == 3 => &[
            ("A · Opening", "opening"),
            ("B · Departure", "departure"),
            ("A′ · Return", "return"),
        ],
        FormKind::Rondo if count == 5 => &[
            ("A · Refrain", "refrain"),
            ("B · Episode I", "episode"),
            ("A · Return", "return"),
            ("C · Episode II", "episode"),
            ("A′ · Final return", "return"),
        ],
        FormKind::Variations if count == 4 => &[
            ("Theme", "theme"),
            ("Minore", "variation"),
            ("Figuration", "variation"),
            ("Theme return", "return"),
        ],
        FormKind::Fugue if count == 5 => &[
            ("Exposition", "exposition"),
            ("Episodes", "episode"),
            ("Middle entry", "entry"),
            ("Stretto", "climax"),
            ("Final entry", "resolution"),
        ],
        FormKind::Passacaglia | FormKind::Erosion | FormKind::Lineage if count == 7 => &[
            ("Cycle I", "cycle"),
            ("Cycle II", "cycle"),
            ("Cycle III", "cycle"),
            ("Cycle IV", "cycle"),
            ("Cycle V · Peak", "climax"),
            ("Cycle VI", "cycle"),
            ("Cycle VII", "resolution"),
        ],
        FormKind::ProgSuite if count == 4 => &[
            ("Opening", "opening"),
            ("First contrast", "contrast"),
            ("Second contrast", "contrast"),
            ("Return", "return"),
        ],
        FormKind::Sonata if count == 5 => &[
            ("Exposition · Primary", "exposition"),
            ("Exposition · Secondary", "exposition"),
            ("Development", "development"),
            ("Recapitulation · Primary", "recapitulation"),
            ("Recapitulation · Secondary", "recapitulation"),
        ],
        _ => return None,
    };
    Some(labels.to_vec())
}

fn section_regions(score: &Score, form: FormKind) -> Vec<SectionRegion> {
    let mut source = score.voice(VoiceRole::Melody);
    if source.is_empty() {
        source = score.events();
    }
    if source.is_empty() {
        return Vec::new();
    }
    let mut runs: Vec<(f64, f32)> = Vec::new();
    for note in source {
        let onset = note.onset.beats();
        match runs.last_mut() {
            Some((_, intensity)) if (*intensity - note.section_intensity).abs() <= 1e-4 => {}
            _ => runs.push((onset, note.section_intensity)),
        }
    }
    let labels = expected_section_labels(form, runs.len());
    runs.iter()
        .enumerate()
        .map(|(index, (start, intensity))| {
            let end = runs
                .get(index + 1)
                .map(|next| next.0)
                .unwrap_or_else(|| score.total_beats.beats());
            let (label, role) = labels
                .as_ref()
                .and_then(|items| items.get(index).copied())
                .unwrap_or(("Structural region", "region"));
            SectionRegion {
                id: format!("section-{index}"),
                label: if labels.is_some() {
                    label.to_string()
                } else {
                    format!("{label} {}", index + 1)
                },
                role: role.to_string(),
                start: musical_time(*start, score.tempo_bpm),
                end: musical_time(end, score.tempo_bpm),
                intensity: *intensity,
                source_method: "score-section-intensity-runs-v1".to_string(),
            }
        })
        .collect()
}

fn phrase_regions(score: &Score) -> Vec<PhraseRegion> {
    let mut source = score.voice(VoiceRole::Melody);
    if source.is_empty() {
        return Vec::new();
    }
    source.sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));
    let mut starts = vec![source[0].onset.beats()];
    for note in &source {
        if note.emphasis == Emphasis::PhraseStart
            && starts
                .last()
                .is_none_or(|last| (note.onset.beats() - *last).abs() > 1e-6)
        {
            starts.push(note.onset.beats());
        }
    }
    starts.sort_by(f64::total_cmp);
    starts.dedup_by(|a, b| (*a - *b).abs() <= 1e-6);
    starts
        .iter()
        .enumerate()
        .map(|(index, start)| {
            let next_start = starts
                .get(index + 1)
                .copied()
                .unwrap_or_else(|| score.total_beats.beats());
            let cadence = source.iter().find(|note| {
                let onset = note.onset.beats();
                onset + 1e-6 >= *start
                    && onset < next_start - 1e-6
                    && note.emphasis == Emphasis::Cadential
            });
            let end = cadence
                .map(|note| (note.onset + note.duration).beats())
                .unwrap_or(next_start)
                .max(*start);
            PhraseRegion {
                id: format!("phrase-{index}"),
                label: format!("Phrase {}", index + 1),
                start: musical_time(*start, score.tempo_bpm),
                end: musical_time(end, score.tempo_bpm),
                closes_with_cadential_marker: cadence.is_some(),
                source_method: "score-emphasis-annotations-v1".to_string(),
            }
        })
        .collect()
}

fn recipe_motif_definition(recipe: &PieceRecipe) -> MotifDefinition {
    let motif = recipe
        .resolved_spec
        .motif(recipe.intent.arousal, recipe.intent.seed);
    MotifDefinition {
        id: "motif-primary".to_string(),
        label: "Primary recipe motif".to_string(),
        degrees: motif.degrees(),
        durations_beats: motif
            .notes
            .iter()
            .filter(|note| note.degree.is_some())
            .map(|note| note.duration.beats())
            .collect(),
        basis: EvidenceBasis {
            status: EvidenceStatus::Observed,
            source_method: "resolved-composition-recipe-motif-v1".to_string(),
            confidence: None,
            limitations: vec![
                "The definition records the deterministic recipe motif; later score occurrences may transform it.".to_string(),
            ],
        },
    }
}

fn recipe_motif_score_notes(score: &Score, recipe: &PieceRecipe) -> Vec<ScoreNote> {
    let motif = recipe
        .resolved_spec
        .motif(recipe.intent.arousal, recipe.intent.seed);
    let mut onset = Duration::zero();
    let mut notes = Vec::new();
    for (pitch, duration) in motif.render(score.key.scale(), 4) {
        if let Some(pitch) = pitch {
            notes.push(ScoreNote {
                pitch,
                onset,
                duration,
                velocity: 0.7,
                role: VoiceRole::Melody,
                emphasis: Emphasis::Normal,
                section_intensity: 1.0,
            });
        }
        onset = onset + duration;
    }
    notes
}

fn best_motif_match(source: &[ScoreNote], target: &[ScoreNote]) -> (ReturnTransformation, f32) {
    let transformations = [
        ReturnTransformation::Literal,
        ReturnTransformation::Transposed,
        ReturnTransformation::Inverted,
        ReturnTransformation::Augmented,
        ReturnTransformation::Diminished,
        ReturnTransformation::Fragmented,
        ReturnTransformation::Restored,
    ];
    transformations
        .into_iter()
        .map(|transformation| {
            let evidence = compare_melodic_sequences(source, target, transformation);
            (transformation, evidence.overall_similarity)
        })
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap_or((ReturnTransformation::Literal, 0.0))
}

fn motif_occurrences(score: &Score, recipe: &PieceRecipe) -> Vec<MotifOccurrence> {
    let source = recipe_motif_score_notes(score, recipe);
    let melody = score.voice(VoiceRole::Melody);
    let width = source.len();
    if width < 2 || melody.len() < width {
        return Vec::new();
    }

    let mut measured = Vec::with_capacity(melody.len() - width + 1);
    for index in 0..=melody.len() - width {
        let (transformation, similarity) = best_motif_match(&source, &melody[index..index + width]);
        measured.push((index, transformation, similarity));
    }

    let threshold = 0.68_f32;
    let mut occurrences = Vec::new();
    for position in 0..measured.len() {
        let (start_index, transformation, similarity) = measured[position];
        if similarity < threshold {
            continue;
        }
        let previous = position
            .checked_sub(1)
            .and_then(|index| measured.get(index))
            .map_or(-1.0, |entry| entry.2);
        let next = measured.get(position + 1).map_or(-1.0, |entry| entry.2);
        if similarity < previous || similarity < next {
            continue;
        }
        let notes = &melody[start_index..start_index + width];
        let start = notes.first().map_or(0.0, |note| note.onset.beats());
        let end = notes
            .last()
            .map_or(start, |note| (note.onset + note.duration).beats());
        let occurrence_index = occurrences.len();
        occurrences.push(MotifOccurrence {
            id: format!("motif-primary-occurrence-{occurrence_index}"),
            motif_id: "motif-primary".to_string(),
            start: musical_time(start, score.tempo_bpm),
            end: musical_time(end, score.tempo_bpm),
            transformation: format!("{transformation:?}").to_ascii_lowercase(),
            similarity,
            source_note_ids: (start_index..start_index + width)
                .map(|index| symbolic_note_id(VoiceRole::Melody, index))
                .collect(),
            basis: EvidenceBasis {
                status: EvidenceStatus::Inferred,
                source_method: "recipe-motif-score-window-scan-v1".to_string(),
                confidence: Some(similarity),
                limitations: vec![
                    "Symbolic transformation similarity is a score-side proxy, not a listener-recognition claim.".to_string(),
                    "Overlapping windows are reduced to local maxima.".to_string(),
                ],
            },
        });
    }
    occurrences
}

fn cadence_events(score: &Score) -> Vec<CadenceEvent> {
    let mut events = Vec::new();
    for role in [
        VoiceRole::Melody,
        VoiceRole::CounterMelody,
        VoiceRole::Harmony,
        VoiceRole::Bass,
    ] {
        for (index, note) in score.voice(role).into_iter().enumerate() {
            if note.emphasis != Emphasis::Cadential {
                continue;
            }
            events.push(CadenceEvent {
                id: format!("cadence-{}-{index}", role_slug(role)),
                at: musical_time(note.onset.beats(), score.tempo_bpm),
                end: musical_time((note.onset + note.duration).beats(), score.tempo_bpm),
                arrival_pitch_name: format!(
                    "{}{}",
                    note.pitch.pitch_class().name(),
                    note.pitch.octave()
                ),
                voice_role: role_label(role).to_string(),
                source_note_ids: vec![symbolic_note_id(role, index)],
                basis: EvidenceBasis {
                    status: EvidenceStatus::Observed,
                    source_method: "score-cadential-emphasis-v1".to_string(),
                    confidence: None,
                    limitations: vec![
                        "The score marks a cadential arrival but does not emit a cadence classification here.".to_string(),
                    ],
                },
            });
        }
    }
    events.sort_by(|left, right| left.at.beats.total_cmp(&right.at.beats));
    events
}

fn exact_home_key_degree(score: &Score, pitch_classes: &BTreeSet<u8>) -> Option<u8> {
    (1_u8..=7).find(|degree| {
        let triad: BTreeSet<u8> = score
            .key
            .diatonic_triad(i32::from(*degree))
            .pitch_classes()
            .into_iter()
            .map(|pitch_class| pitch_class.value())
            .collect();
        &triad == pitch_classes
    })
}

fn sonority_regions(score: &Score) -> Vec<SonorityRegion> {
    let duration = score.total_beats.beats();
    if duration <= 0.0 {
        return Vec::new();
    }
    let mut regions: Vec<SonorityRegion> = Vec::new();
    for beat_index in 0..duration.ceil() as usize {
        let start = beat_index as f64;
        let end = (start + 1.0).min(duration);
        let active: Vec<_> = score
            .notes
            .iter()
            .filter(|note| {
                note.onset.beats() <= start + 1e-6
                    && (note.onset + note.duration).beats() > start + 1e-6
            })
            .collect();
        if active.is_empty() {
            continue;
        }
        let pitch_class_values: BTreeSet<u8> = active
            .iter()
            .map(|note| note.pitch.pitch_class().value())
            .collect();
        let pitch_classes: Vec<String> = pitch_class_values
            .iter()
            .map(|value| PitchClass::new(i32::from(*value)).name().to_string())
            .collect();
        let bass_pitch_class = active
            .iter()
            .min_by_key(|note| note.pitch.midi())
            .map(|note| note.pitch.pitch_class().name().to_string());
        let degree = exact_home_key_degree(score, &pitch_class_values);
        let function = degree.map(|degree| {
            format!("{:?}", score.key.function(i32::from(degree))).to_ascii_lowercase()
        });
        if let Some(previous) = regions.last_mut() {
            if previous.pitch_classes == pitch_classes
                && previous.bass_pitch_class == bass_pitch_class
                && previous.home_key_degree == degree
            {
                previous.end = musical_time(end, score.tempo_bpm);
                continue;
            }
        }
        regions.push(SonorityRegion {
            id: format!("sonority-{}", regions.len()),
            start: musical_time(start, score.tempo_bpm),
            end: musical_time(end, score.tempo_bpm),
            pitch_classes,
            bass_pitch_class,
            home_key_degree: degree,
            home_key_function: function,
            basis: EvidenceBasis {
                status: EvidenceStatus::Reconstructed,
                source_method: "active-score-pitch-classes-per-beat-v1".to_string(),
                confidence: degree.map(|_| 1.0),
                limitations: vec![
                    "Pitch classes are observed exactly at beat boundaries.".to_string(),
                    "A degree label appears only for an exact triad in the declared home key; this is not modulation analysis.".to_string(),
                ],
            },
        });
    }
    regions
}

fn orchestration_regions(score: &Score, sections: &[SectionRegion]) -> Vec<OrchestrationRegion> {
    let ranges: Vec<(String, f64, f64)> = if sections.is_empty() {
        vec![(
            "orchestration-whole-piece".to_string(),
            0.0,
            score.total_beats.beats(),
        )]
    } else {
        sections
            .iter()
            .map(|section| {
                (
                    format!("orchestration-{}", section.id),
                    section.start.beats,
                    section.end.beats,
                )
            })
            .collect()
    };
    ranges
        .into_iter()
        .filter_map(|(id, start, end)| {
            let notes: Vec<_> = score
                .notes
                .iter()
                .filter(|note| note.onset.beats() >= start && note.onset.beats() < end)
                .collect();
            if notes.is_empty() {
                return None;
            }
            let active_voices = [
                VoiceRole::Melody,
                VoiceRole::CounterMelody,
                VoiceRole::Harmony,
                VoiceRole::Bass,
            ]
            .into_iter()
            .filter_map(|role| {
                let note_count = notes.iter().filter(|note| note.role == role).count();
                (note_count > 0).then(|| VoiceActivity {
                    voice_role: role_label(role).to_string(),
                    note_count,
                })
            })
            .collect();
            let register_min_midi = notes.iter().map(|note| note.pitch.midi()).min();
            let register_max_midi = notes.iter().map(|note| note.pitch.midi()).max();
            let mean_velocity = notes.iter().map(|note| note.velocity).sum::<f32>() / notes.len() as f32;
            Some(OrchestrationRegion {
                id,
                start: musical_time(start, score.tempo_bpm),
                end: musical_time(end, score.tempo_bpm),
                active_voices,
                register_min_midi,
                register_max_midi,
                mean_velocity,
                basis: EvidenceBasis {
                    status: EvidenceStatus::Observed,
                    source_method: "symbolic-voice-assignment-by-structural-region-v1".to_string(),
                    confidence: None,
                    limitations: vec![
                        "This describes assigned score voices, not rendered prominence or perceived orchestral balance.".to_string(),
                    ],
                },
            })
        })
        .collect()
}

fn resonance_curve(score: &Score) -> Option<ResonanceCurve> {
    let duration = score.total_beats.beats();
    if duration <= 0.0 || score.notes.is_empty() {
        return None;
    }
    let sample_count = ((duration.ceil() as usize).saturating_mul(4)).clamp(16, 96);
    let melody = score.voice(VoiceRole::Melody);
    let mut raw = Vec::with_capacity(sample_count);
    for index in 0..sample_count {
        let beat = if sample_count == 1 {
            0.0
        } else {
            duration * index as f64 / (sample_count - 1) as f64
        };
        let active: Vec<_> = score
            .notes
            .iter()
            .filter(|note| {
                note.onset.beats() <= beat + 1e-6
                    && (note.onset + note.duration).beats() > beat + 1e-6
            })
            .collect();
        let mean_velocity = if active.is_empty() {
            0.0
        } else {
            active.iter().map(|note| note.velocity).sum::<f32>() / active.len() as f32
        };
        let mean_intensity = if active.is_empty() {
            0.0
        } else {
            active
                .iter()
                .map(|note| note.section_intensity)
                .sum::<f32>()
                / active.len() as f32
        };
        let local_melody: Vec<_> = melody
            .iter()
            .copied()
            .filter(|note| (note.onset.beats() - beat).abs() <= 1.0)
            .collect();
        let motion = if local_melody.len() < 2 {
            0.0
        } else {
            let total = local_melody
                .windows(2)
                .map(|pair| pair[0].pitch.semitones_to(pair[1].pitch).unsigned_abs() as f32)
                .sum::<f32>();
            (total / (local_melody.len() - 1) as f32 / 12.0).clamp(0.0, 1.0)
        };
        raw.push((beat, active.len(), mean_velocity, mean_intensity, motion));
    }
    let max_density = raw.iter().map(|entry| entry.1).max().unwrap_or(1).max(1) as f32;
    let samples = raw
        .into_iter()
        .map(
            |(beat, active_count, mean_velocity, mean_intensity, motion)| {
                let density = active_count as f32 / max_density;
                let normalized_intensity = (mean_intensity / 1.5).clamp(0.0, 1.0);
                ResonanceSample {
                    at: musical_time(beat, score.tempo_bpm),
                    energy: (0.5 * mean_velocity + 0.3 * density + 0.2 * normalized_intensity)
                        .clamp(0.0, 1.0),
                    density,
                    motion,
                }
            },
        )
        .collect();
    Some(ResonanceCurve {
        basis: EvidenceBasis {
            status: EvidenceStatus::Inferred,
            source_method: "score-activity-resonance-proxy-v1".to_string(),
            confidence: None,
            limitations: vec![
                "Energy, density, and motion are normalized symbolic proxies derived from active notes, velocity, section intensity, and melodic intervals.".to_string(),
                "They are not objective emotion measurements and do not include spectral audio analysis.".to_string(),
            ],
        },
        samples,
    })
}

fn composition_bundle(
    score: &Score,
    recipe: &PieceRecipe,
) -> (ListenCompositionBundle, Vec<BundleWarning>) {
    let form = recipe.resolved_spec.form_kind(recipe.intent.seed);
    let sections = section_regions(score, form);
    let phrases = phrase_regions(score);
    let motif_definitions = vec![recipe_motif_definition(recipe)];
    let motif_occurrences = motif_occurrences(score, recipe);
    let cadences = cadence_events(score);
    let sonorities = sonority_regions(score);
    let orchestration = orchestration_regions(score, &sections);
    let resonance = resonance_curve(score);
    let mut warnings = vec![
        BundleWarning {
            code: "sections-derived-from-score-annotations".to_string(),
            message: "Section boundaries are reconstructed from emitted section-intensity annotations; explicit composer section plans are not retained for every form yet.".to_string(),
        },
        BundleWarning {
            code: "motif-occurrences-inferred".to_string(),
            message: "The recipe motif is exact, but occurrence relationships are a transformation-aware score-window analysis rather than composer-emitted occurrence IDs.".to_string(),
        },
        BundleWarning {
            code: "harmony-analysis-bounded".to_string(),
            message: "Harmony evidence exposes exact sounding pitch classes and exact home-key triad matches only; modulation and ambiguous chord interpretation are not claimed.".to_string(),
        },
        BundleWarning {
            code: "cadence-types-unavailable".to_string(),
            message: "Cadential score markers are exposed, but cadence types are not invented when the composer did not emit them.".to_string(),
        },
        BundleWarning {
            code: "resonance-is-a-symbolic-proxy".to_string(),
            message: "The structural-activity curve is a normalized score-derived proxy, not objective emotional truth.".to_string(),
        },
    ];
    if phrases.is_empty() {
        warnings.push(BundleWarning {
            code: "phrase-annotations-unavailable".to_string(),
            message: "The score did not emit melody phrase markers for this form.".to_string(),
        });
    }
    if motif_occurrences.is_empty() {
        warnings.push(BundleWarning {
            code: "motif-occurrences-below-threshold".to_string(),
            message: "No score window met the conservative motif-occurrence threshold; the recipe motif remains available without fabricated returns.".to_string(),
        });
    }
    let zero = musical_time(0.0, score.tempo_bpm);
    (
        ListenCompositionBundle {
            ticks_per_beat: TICKS_PER_BEAT,
            duration_ticks: (score.total_beats.beats() * TICKS_PER_BEAT as f64).round() as u64,
            duration_beats: score.total_beats.beats(),
            duration_seconds: score.seconds(),
            form_kind: format!("{form:?}"),
            tempo_map: vec![TempoPoint {
                at: zero,
                bpm: score.tempo_bpm,
            }],
            meter_map: vec![MeterPoint {
                at: zero,
                numerator: score.meter,
                denominator: 4,
            }],
            sections,
            phrases,
            notes: symbolic_notes(score),
            motif_definitions,
            motif_occurrences,
            cadences,
            sonorities,
            orchestration,
            resonance,
        },
        warnings,
    )
}

fn performance_bundle_payload(
    score: &Score,
    spec: &CompositionSpec,
    seed: u64,
    state: &MusicalState,
) -> (ListenPerformanceBundle, Vec<BundleWarning>) {
    let voices = symthaea_muse::theory_realize::perform_with_spec(score, spec, seed, state);
    let symbolic = symbolic_notes(score);
    let symbolic_times: HashMap<&str, (f64, f64)> = symbolic
        .iter()
        .map(|note| {
            (
                note.id.as_str(),
                (note.onset.seconds, note.duration_seconds),
            )
        })
        .collect();
    let mut warnings = Vec::new();
    let mut events = Vec::new();
    let mut summaries = Vec::new();
    let mut duration = 0.0_f64;
    for (voice_index, voice) in voices.into_iter().enumerate() {
        let voice_id = voice.name.to_ascii_lowercase().replace(' ', "-");
        summaries.push(PerformanceVoiceSummary {
            id: voice_id.clone(),
            name: voice.name.clone(),
            instrument: voice.instrument.clone(),
            note_count: voice.notes.len(),
        });
        let source_role = match voice.name.as_str() {
            "Bass" => Some(VoiceRole::Bass),
            "Harmony" => Some(VoiceRole::Harmony),
            "Counter" => Some(VoiceRole::CounterMelody),
            "Melody" => Some(VoiceRole::Melody),
            _ => None,
        };
        if let Some(role) = source_role {
            let expected = score.voice(role).len();
            if expected != voice.notes.len() {
                warnings.push(BundleWarning {
                    code: "performance-source-count-mismatch".to_string(),
                    message: format!(
                        "{} emitted {} performed events for {expected} symbolic events; source IDs after the shared prefix were withheld.",
                        voice.name,
                        voice.notes.len()
                    ),
                });
            }
        }
        for (note_index, note) in voice.notes.into_iter().enumerate() {
            let source_note_id = source_role.and_then(|role| {
                (score.voice(role).len() == summaries.last().map_or(0, |v| v.note_count))
                    .then(|| symbolic_note_id(role, note_index))
            });
            let nominal = source_note_id
                .as_deref()
                .and_then(|id| symbolic_times.get(id).copied());
            duration = duration.max((note.start_time + note.duration) as f64);
            events.push(PerformedNoteEvent {
                id: format!("performance-{voice_index}-{note_index}"),
                voice_id: voice_id.clone(),
                source_note_id,
                start_seconds: note.start_time as f64,
                duration_seconds: note.duration as f64,
                frequency_hz: note.frequency as f64,
                velocity: note.velocity as f64,
                onset_deviation_seconds: nominal.map(|(start, _)| note.start_time as f64 - start),
                duration_deviation_seconds: nominal
                    .map(|(_, duration)| note.duration as f64 - duration),
            });
        }
    }
    events.sort_by(|a, b| a.start_seconds.total_cmp(&b.start_seconds));
    (
        ListenPerformanceBundle {
            duration_seconds: duration,
            mapping_method: "ordered-primary-score-voice-v1; renderer-added-voices-unmapped"
                .to_string(),
            voices: summaries,
            notes: events,
        },
        warnings,
    )
}

/// Converts a music-theory `TitleRecipe` into the wire-shaped summary
/// clients receive alongside `title`.
fn title_recipe_summary(
    recipe: &symthaea_music_theory::describe::TitleRecipe,
) -> TitleRecipeSummary {
    TitleRecipeSummary {
        family: format!("{:?}", recipe.family).to_ascii_lowercase(),
        template_id: recipe.template_id.clone(),
        source_traits: recipe.source_traits.clone(),
        alternatives: recipe.alternatives.clone(),
    }
}

#[cfg(test)]
mod piece_bundle_tests {
    use super::{
        cadence_events, orchestration_regions, section_regions, sonority_regions, symbolic_notes,
    };
    use symthaea_music_theory::{
        Duration, Emphasis, FormKind, Key, Pitch, PitchClass, Score, ScoreNote, VoiceRole,
    };

    fn note(onset: i64, intensity: f32, emphasis: Emphasis) -> ScoreNote {
        ScoreNote {
            pitch: Pitch::new(PitchClass::C, 4),
            onset: Duration::new(onset, 1),
            duration: Duration::quarter(),
            velocity: 0.7,
            role: VoiceRole::Melody,
            emphasis,
            section_intensity: intensity,
        }
    }

    #[test]
    fn ternary_regions_follow_real_score_intensity_boundaries() {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        score.push(note(0, 0.85, Emphasis::PhraseStart));
        score.push(note(4, 1.0, Emphasis::PhraseStart));
        score.push(note(8, 0.95, Emphasis::PhraseStart));
        score.push(note(11, 0.95, Emphasis::Cadential));

        let regions = section_regions(&score, FormKind::Ternary);
        assert_eq!(regions.len(), 3);
        assert_eq!(regions[0].label, "A · Opening");
        assert_eq!(regions[1].start.beats, 4.0);
        assert_eq!(regions[2].label, "A′ · Return");
    }

    #[test]
    fn cadence_events_preserve_exact_score_note_ids() {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        score.push(note(0, 0.85, Emphasis::PhraseStart));
        score.push(note(3, 0.85, Emphasis::Cadential));
        let events = cadence_events(&score);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].source_note_ids, vec!["score-melody-1"]);
        assert_eq!(events[0].arrival_pitch_name, "C4");
    }

    #[test]
    fn exact_home_key_triad_is_labelled_without_guessing_modulation() {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        for (midi, role) in [
            (48, VoiceRole::Bass),
            (60, VoiceRole::Melody),
            (64, VoiceRole::Harmony),
            (67, VoiceRole::Harmony),
        ] {
            score.push(ScoreNote {
                pitch: Pitch::from_midi(midi),
                onset: Duration::zero(),
                duration: Duration::quarter(),
                velocity: 0.7,
                role,
                emphasis: Emphasis::Normal,
                section_intensity: 1.0,
            });
        }
        let regions = sonority_regions(&score);
        assert_eq!(regions[0].home_key_degree, Some(1));
        assert_eq!(regions[0].home_key_function.as_deref(), Some("tonic"));
    }

    #[test]
    fn orchestration_reports_assigned_voices_not_perceptual_prominence() {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        score.push(note(0, 0.85, Emphasis::Normal));
        score.push(ScoreNote {
            pitch: Pitch::from_midi(48),
            onset: Duration::zero(),
            duration: Duration::quarter(),
            velocity: 0.5,
            role: VoiceRole::Bass,
            emphasis: Emphasis::Normal,
            section_intensity: 0.85,
        });
        let regions = orchestration_regions(&score, &[]);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].active_voices.len(), 2);
        assert_eq!(regions[0].register_min_midi, Some(48));
        assert_eq!(regions[0].register_max_midi, Some(60));
    }

    #[test]
    fn symbolic_ids_are_stable_within_each_voice() {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        score.push(note(0, 0.85, Emphasis::Normal));
        score.push(note(1, 0.85, Emphasis::Normal));
        let notes = symbolic_notes(&score);
        assert_eq!(notes[0].id, "score-melody-0");
        assert_eq!(notes[1].id, "score-melody-1");
        assert_eq!(notes[0].midi, 60);
    }
}

async fn listen_bundle(
    State(studio): State<Arc<Studio>>,
    AxPath(id): AxPath<u64>,
) -> Result<Json<BundleEnvelope<ListenCompositionBundle>>, StatusCode> {
    let (score, recipe, created_at_unix_ms, wav) = {
        let store = studio.candidates.lock().unwrap();
        let candidate = store.get(&id).ok_or(StatusCode::NOT_FOUND)?;
        (
            candidate.score.clone(),
            candidate.recipe.clone(),
            candidate.created_at_unix_ms,
            candidate.wav.clone(),
        )
    };
    let (payload, warnings) = composition_bundle(&score, &recipe);
    Ok(Json(BundleEnvelope {
        piece_id: id,
        render_id: Some(render_id(id, &wav)),
        bundle_version: LISTEN_COMPOSITION_BUNDLE_VERSION,
        created_at_unix_ms,
        warnings,
        payload,
    }))
}

async fn performance_bundle(
    State(studio): State<Arc<Studio>>,
    AxPath(id): AxPath<u64>,
) -> Result<Json<BundleEnvelope<ListenPerformanceBundle>>, StatusCode> {
    let (score, spec, seed, state, created_at_unix_ms, wav) = {
        let store = studio.candidates.lock().unwrap();
        let candidate = store.get(&id).ok_or(StatusCode::NOT_FOUND)?;
        (
            candidate.score.clone(),
            candidate.spec.clone(),
            candidate.seed,
            candidate.state.clone(),
            candidate.created_at_unix_ms,
            candidate.wav.clone(),
        )
    };
    let (payload, warnings) = performance_bundle_payload(&score, &spec, seed, &state);
    Ok(Json(BundleEnvelope {
        piece_id: id,
        render_id: Some(render_id(id, &wav)),
        bundle_version: LISTEN_PERFORMANCE_BUNDLE_VERSION,
        created_at_unix_ms,
        warnings,
        payload,
    }))
}

async fn piece_provenance(
    State(studio): State<Arc<Studio>>,
    AxPath(id): AxPath<u64>,
) -> Result<Json<BundleEnvelope<PieceProvenanceBundle>>, StatusCode> {
    let (score, recipe, seed, style_name, created_at_unix_ms, wav) = {
        let store = studio.candidates.lock().unwrap();
        let candidate = store.get(&id).ok_or(StatusCode::NOT_FOUND)?;
        (
            candidate.score.clone(),
            candidate.recipe.clone(),
            candidate.seed,
            candidate.spec.name.clone(),
            candidate.created_at_unix_ms,
            candidate.wav.clone(),
        )
    };
    let recipe_sha256 = serialized_sha256(&recipe)?;
    let score_sha256 = serialized_sha256(&score)?;
    let audio_sha256 = sha256_hex(&wav);
    let gaps = recipe.reproduction_gaps();
    let limitations: Vec<String> = gaps.iter().map(|gap| format!("{gap:?}")).collect();
    let symbolic_score_exact = recipe.renderer.theory_source_revision.is_some();
    let midi_exact = symbolic_score_exact
        && recipe.renderer.muse_source_revision.is_some()
        && recipe.renderer.performance_model_sha256.is_some();
    let rendered_audio_exact = gaps.is_empty();
    let warnings = (!limitations.is_empty())
        .then(|| BundleWarning {
            code: "bounded-reproducibility".to_string(),
            message: format!(
                "Exact independent reproduction is not claimed while these fields are absent: {}.",
                limitations.join(", ")
            ),
        })
        .into_iter()
        .collect();
    let render_id = render_id(id, &wav);
    let payload = PieceProvenanceBundle {
        recipe_schema_version: recipe.schema_version,
        recipe_sha256,
        score_sha256,
        audio_sha256: audio_sha256.clone(),
        seed,
        style_name,
        renderer_name: recipe.renderer.renderer_name.clone(),
        renderer_version: recipe.renderer.renderer_version.clone(),
        muse_engine_version: recipe.renderer.muse_engine_version.clone(),
        theory_engine_version: recipe.renderer.theory_engine_version.clone(),
        muse_source_revision: recipe.renderer.muse_source_revision.clone(),
        theory_source_revision: recipe.renderer.theory_source_revision.clone(),
        soundfont_sha256: recipe.renderer.soundfont_sha256.clone(),
        renderer_binary_sha256: recipe.renderer.renderer_binary_sha256.clone(),
        performance_model_sha256: recipe.renderer.performance_model_sha256.clone(),
        render_environment_sha256: recipe.renderer.render_environment_sha256.clone(),
        reproduction: ReproducibilityClaim {
            symbolic_score_exact,
            midi_exact,
            rendered_audio_exact,
            limitations,
        },
        artifacts: vec![
            ProvenanceArtifact {
                kind: "rendered-audio".to_string(),
                media_type: "audio/wav".to_string(),
                uri: format!("/api/audio/{id}"),
                sha256: Some(audio_sha256),
            },
            ProvenanceArtifact {
                kind: "performed-midi".to_string(),
                media_type: "audio/midi".to_string(),
                uri: format!("/api/midi/{id}"),
                sha256: None,
            },
            ProvenanceArtifact {
                kind: "composition-bundle".to_string(),
                media_type: "application/vnd.symthaea.muse.listen+json".to_string(),
                uri: format!("/api/piece/{id}/listen-bundle"),
                sha256: None,
            },
            ProvenanceArtifact {
                kind: "performance-bundle".to_string(),
                media_type: "application/vnd.symthaea.muse.performance+json".to_string(),
                uri: format!("/api/piece/{id}/performance-bundle"),
                sha256: None,
            },
        ],
    };
    Ok(Json(BundleEnvelope {
        piece_id: id,
        render_id: Some(render_id),
        bundle_version: PIECE_PROVENANCE_BUNDLE_VERSION,
        created_at_unix_ms,
        warnings,
        payload,
    }))
}

/// Serialize a unit-only enum via its own `#[serde(rename_all = "snake_case")]`
/// representation rather than hand-duplicating the mapping here.
fn snake_case_variant<T: Serialize>(value: &T) -> String {
    match serde_json::to_value(value) {
        Ok(serde_json::Value::String(name)) => name,
        _ => "unknown".to_string(),
    }
}

/// `GrammarPlanEvidence`'s variants carry data, so they don't serialize to
/// a bare string the way `snake_case_variant` expects (they'd become a
/// JSON object like `{"call_response": {...}}`) — this names each variant
/// directly instead.
fn plan_kind_str(plan: &GrammarPlanEvidence) -> &'static str {
    match plan {
        GrammarPlanEvidence::PeriodSentence(_) => "period_sentence",
        GrammarPlanEvidence::Contrapuntal(_) => "contrapuntal",
        GrammarPlanEvidence::GrooveCycle(_) => "groove_cycle",
        GrammarPlanEvidence::AdditiveProcess(_) => "additive_process",
        GrammarPlanEvidence::ModalArc(_) => "modal_arc",
        GrammarPlanEvidence::CallResponse(_) => "call_response",
        GrammarPlanEvidence::JazzChorus(_) => "jazz_chorus",
        GrammarPlanEvidence::Compatibility { .. } => "compatibility",
    }
}

/// Every style Muse can compose in, each with its real
/// `Style::grammar_family()` — lets `symthaea-muse-ui` make policy-aware
/// style choices (`JourneyPolicy::Resonance`/`Contrast`) without
/// depending on this crate's native `symthaea-music-theory`. Static for
/// the process lifetime; the client fetches this once and caches it.
async fn listen_styles() -> Json<Vec<symthaea_muse_protocol::StyleFamily>> {
    Json(
        Style::ALL
            .into_iter()
            .map(|style| symthaea_muse_protocol::StyleFamily {
                name: format!("{style:?}"),
                family: snake_case_variant(&style.grammar_family()),
            })
            .collect(),
    )
}

/// The Analyst's independent, deterministic verification of a composed piece
/// (`symthaea_muse::analyst::analyze_piece`) — turns the compiled-but-inert
/// verification pipeline into something actually reachable.
///
/// Updated 2026-07-25: the live `/api/compose` handler now routes
/// preset-derived candidates through `compose_with_grammar_plan` for real
/// (previously it always called `compose_with_spec_and_form` directly,
/// which never reached any dedicated grammar engine — see git history for
/// that finding). The REAL `GrammarPlanEvidence` produced during
/// composition is now stored on the `Candidate` and used here directly
/// (`candidate.plan`), rather than always synthesizing a
/// `GrammarPlanEvidence::Compatibility` fallback. That fallback still
/// applies, correctly, for the one case where there genuinely is no
/// dedicated-engine plan to report: a candidate composed from a
/// user-authored custom spec (`ComposeRequest::spec`), which has no
/// `Style` to derive a `GrammarProfile` from in the first place — see
/// `symthaea_muse::analyst::plan_checks`.
async fn analyst_bundle(
    State(studio): State<Arc<Studio>>,
    AxPath(id): AxPath<u64>,
) -> Result<Json<BundleEnvelope<symthaea_muse_protocol::AnalystPieceBundle>>, StatusCode> {
    let (score, recipe, spec, seed, state, form_available, created_at_unix_ms, wav, real_plan) = {
        let store = studio.candidates.lock().unwrap();
        let candidate = store.get(&id).ok_or(StatusCode::NOT_FOUND)?;
        (
            candidate.score.clone(),
            candidate.recipe.clone(),
            candidate.spec.clone(),
            candidate.seed,
            candidate.state.clone(),
            candidate.form.is_some(),
            candidate.created_at_unix_ms,
            candidate.wav.clone(),
            candidate.plan.clone(),
        )
    };
    // `spec.name` matches a `Style` variant exactly for every preset-style
    // candidate; a candidate composed from a user-authored custom spec
    // (`ComposeRequest::spec`) carries an arbitrary name instead, which
    // can't be attributed to a grammar family. Fall back to the same
    // family the composer itself falls back to (`compose_with_spec_and_form`'s
    // `grammar: None` path is a `PeriodSentence`-shaped heuristic) rather
    // than failing the request, and disclose it.
    let (style_fallback, style): (bool, Style) =
        match serde_json::from_value(serde_json::Value::String(spec.name.clone())) {
            Ok(style) => (false, style),
            Err(_) => (true, Style::Classical),
        };
    let mut style_warnings = Vec::new();
    if style_fallback {
        style_warnings.push(BundleWarning {
            code: "custom-spec-grammar-family-unknown".to_string(),
            message: format!(
                "Spec name '{}' does not match a preset Style; grammar-family attribution defaulted to Classical/PeriodSentence.",
                spec.name
            ),
        });
    }
    let profile = style.grammar_profile();
    let family = style.grammar_family();
    let culturally_qualified = matches!(
        family,
        GrammarFamily::GrooveCycle | GrammarFamily::RagaModalArc
    );
    // The REAL plan the candidate was actually composed with, when one was
    // stored (every preset-derived candidate since the /api/compose fix
    // above) -- falls back to synthesizing `Compatibility` only for
    // candidates with no stored plan (a user-authored custom spec, which
    // has no `Style` to derive a `GrammarProfile` from at all).
    let plan = real_plan.unwrap_or(GrammarPlanEvidence::Compatibility {
        family,
        form_available,
    });
    let provenance = GrammarProvenance {
        family: snake_case_variant(&family),
        phrase_grammar: snake_case_variant(&profile.phrase),
        harmonic_syntax: snake_case_variant(&profile.harmony),
        performance_dialect: snake_case_variant(&profile.performance),
        plan_kind: plan_kind_str(&plan).to_string(),
        culturally_qualified,
        obligations: Vec::new(),
        supported_intent_axes: profile
            .supported_intent_axes
            .iter()
            .map(snake_case_variant)
            .collect(),
        performance_features: None,
    };
    let (composition, mut warnings) = composition_bundle(&score, &recipe);
    let (performance, perf_warnings) = performance_bundle_payload(&score, &spec, seed, &state);
    warnings.extend(perf_warnings);
    warnings.extend(style_warnings);
    let mut payload = symthaea_muse::analyst::analyze_piece(
        &composition,
        Some(&performance),
        &provenance,
        Some(&plan),
        &recipe.intent,
    );
    let audio_sha256 = sha256_hex(&wav);
    payload.audio_integrity = Some(symthaea_muse::analyst::analyze_audio_integrity(
        &wav,
        &audio_sha256,
        created_at_unix_ms,
    ));
    Ok(Json(BundleEnvelope {
        piece_id: id,
        render_id: Some(render_id(id, &wav)),
        bundle_version: ANALYST_PIECE_BUNDLE_VERSION,
        created_at_unix_ms,
        warnings,
        payload,
    }))
}

/// A genealogy manifest is keyed by its own ledger id, not a `piece_id` —
/// it outlives the in-memory `Candidate` it was allocated from (which is
/// evicted under memory pressure and renumbered every server restart).
fn genealogy_store(
    studio: &Studio,
) -> Result<&symthaea_muse::genealogy::GenealogyStore, StatusCode> {
    studio
        .genealogy
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)
}

async fn genealogy_manifest(
    State(studio): State<Arc<Studio>>,
    AxPath(id): AxPath<i64>,
) -> Result<Json<symthaea_muse_protocol::GenealogyManifest>, StatusCode> {
    let store = genealogy_store(&studio)?;
    let manifest = store
        .manifest(id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;
    Ok(Json(manifest))
}

async fn genealogy_children(
    State(studio): State<Arc<Studio>>,
    AxPath(id): AxPath<i64>,
) -> Result<Json<Vec<symthaea_muse_protocol::GenealogyManifest>>, StatusCode> {
    let store = genealogy_store(&studio)?;
    let children = store
        .children(id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(children))
}

async fn genealogy_ancestry(
    State(studio): State<Arc<Studio>>,
    AxPath(id): AxPath<i64>,
) -> Result<Json<Vec<symthaea_muse_protocol::GenealogyManifest>>, StatusCode> {
    let store = genealogy_store(&studio)?;
    let ancestry = store
        .ancestry(id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    if ancestry.is_empty() {
        return Err(StatusCode::NOT_FOUND);
    }
    Ok(Json(ancestry))
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

/// One piece's fingerprint-ready `(Score, Option<Form>)` plus the display
/// metadata the Atlas needs, before 2D projection.
struct AtlasSource {
    id: String,
    title: String,
    style: String,
    score: symthaea_music_theory::Score,
    form: Option<symthaea_music_theory::Form>,
    kept: bool,
}

fn duration_secs(score: &symthaea_music_theory::Score) -> f32 {
    (score.total_beats.beats() / score.tempo_bpm as f64 * 60.0) as f32
}

/// Gathers every in-session candidate plus persisted kept/liked piece into
/// fingerprint-ready `AtlasSource`s — shared by `/api/atlas` and
/// `/api/atlas/compare` so both endpoints see the same set of pieces.
///
/// Kept/liked pieces don't carry their `Score`/`Form` directly in
/// `keepers.jsonl` — only a fully deterministic `PieceRecipe` (`intent` +
/// `resolved_spec`). Composition is deterministic given `(intent, spec)`, so
/// this recomposes each keeper via `compose_with_spec_and_form` rather than
/// standing up new storage for the raw score.
fn gather_atlas_sources(studio: &Studio) -> Vec<AtlasSource> {
    let mut sources: Vec<AtlasSource> = Vec::new();

    {
        let store = studio.candidates.lock().unwrap();
        for (id, c) in store.iter() {
            let title =
                symthaea_music_theory::describe::title_for(&c.spec, &c.recipe.intent, c.seed);
            sources.push(AtlasSource {
                id: format!("candidate:{id}"),
                title,
                style: c.spec.name.clone(),
                score: c.score.clone(),
                form: c.form.clone(),
                kept: false,
            });
        }
    }

    if let Ok(text) = std::fs::read_to_string("data/taste/keepers.jsonl") {
        for line in text.lines() {
            let Ok(entry) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            let Some(recipe) = entry.get("recipe") else {
                continue;
            };
            let (Some(intent), Some(spec)) = (
                recipe.get("intent").and_then(|v| {
                    serde_json::from_value::<symthaea_music_theory::MusicalIntent>(v.clone()).ok()
                }),
                recipe.get("resolved_spec").and_then(|v| {
                    serde_json::from_value::<symthaea_music_theory::CompositionSpec>(v.clone()).ok()
                }),
            ) else {
                continue;
            };
            let audio_key = entry
                .get("audio_key")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            let title = entry
                .get("title")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| spec.name.clone());
            let style = entry
                .get("spec")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| spec.name.clone());
            let (score, form) =
                symthaea_music_theory::composer::compose_with_spec_and_form(&intent, &spec);
            sources.push(AtlasSource {
                id: format!("keeper:{audio_key}"),
                title,
                style,
                score,
                form,
                kept: true,
            });
        }
    }

    sources
}

/// Resolves an Atlas lens name to per-layer weights, in
/// `symthaea_music_theory::fingerprint::LAYERS` order (form, harmony,
/// orchestration, rhythm, contour, tempo_meter). A lens is just a
/// reweighting of the SAME structural fingerprint before
/// [`symthaea_music_theory::fingerprint::project_2d`] — not a new embedding
/// or analysis. Unrecognized/absent names resolve to `"combined"` (even
/// weighting, Phase 1's original behavior). `tempo_meter` rides with
/// `rhythm` (both metric/temporal attributes) rather than owning its own
/// lens — boosted under `"rhythm"`, kept low elsewhere, same as every other
/// non-owning layer.
///
/// Returns `(resolved_name, weights)`.
// Layer order (symthaea_music_theory::fingerprint::LAYERS, 11 wide as of
// the Atlas fingerprint v2 pass): [0]form [1]harmony [2]orchestration
// [3]rhythm [4]contour [5]tempo_meter [6]pitch_class [7]interval
// [8]chord_quality [9]cadence [10]motif.
fn resolve_atlas_lens(
    lens: Option<&str>,
) -> (
    &'static str,
    [f64; symthaea_music_theory::fingerprint::LAYERS.len()],
) {
    match lens {
        // Melodic identity: form/contour/motif carry it directly;
        // pitch_class/interval get a modest boost (melodic content, but
        // not what this lens is centrally about).
        Some("motif_form") => (
            "motif_form",
            [2.5, 0.4, 0.4, 0.4, 2.5, 0.4, 1.2, 1.2, 0.4, 0.4, 3.0],
        ),
        // chord_quality + cadence are the new harmonic-content layers this
        // pass added -- boosted alongside the existing harmony layer.
        Some("harmony") => (
            "harmony",
            [0.4, 3.0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 3.0, 3.0, 0.4],
        ),
        // tempo_meter is metric/temporal like rhythm; interval (melodic
        // step SIZE, not timing) deliberately stays at baseline here.
        Some("rhythm") => (
            "rhythm",
            [0.4, 0.4, 0.4, 3.0, 0.4, 3.0, 0.4, 0.4, 0.4, 0.4, 0.4],
        ),
        Some("orchestration") => (
            "orchestration",
            [0.4, 0.4, 3.0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
        ),
        _ => (
            "combined",
            [1.0; symthaea_music_theory::fingerprint::LAYERS.len()],
        ),
    }
}

#[derive(serde::Deserialize)]
struct AtlasQuery {
    lens: Option<String>,
}

/// Exact-duplicate collapse: the Atlas plots one point per DISTINCT score,
/// not one per candidate/keeper (`AtlasPoint::multiplicity`'s contract) —
/// several generations landing on byte-identical music otherwise stacks
/// indistinguishable points on top of each other and pads every
/// duplicate's fingerprint into the NN/projection math as if it were
/// independent evidence. Groups `sources` by
/// `symthaea_music_theory::fingerprint::exact_fingerprint`, returns one
/// `(representative_index, multiplicity)` per group, sorted by
/// `representative_index` (so callers get a stable, deterministic order).
/// The representative is a kept piece over a transient candidate when the
/// group has both — the more meaningful thing to show.
fn collapse_exact_duplicates(sources: &[AtlasSource]) -> Vec<(usize, u32)> {
    let mut groups: std::collections::HashMap<u64, Vec<usize>> = std::collections::HashMap::new();
    for (i, s) in sources.iter().enumerate() {
        groups
            .entry(symthaea_music_theory::fingerprint::exact_fingerprint(
                &s.score,
            ))
            .or_default()
            .push(i);
    }
    let mut representatives: Vec<(usize, u32)> = groups
        .into_values()
        .map(|members| {
            let rep = members
                .iter()
                .copied()
                .find(|&i| sources[i].kept)
                .unwrap_or(members[0]);
            (rep, members.len() as u32)
        })
        .collect();
    representatives.sort_by_key(|&(i, _)| i);
    representatives
}

#[cfg(test)]
mod atlas_dedup_tests {
    use super::*;

    fn source(id: &str, kept: bool, note: bool) -> AtlasSource {
        use symthaea_music_theory::{
            Duration, Emphasis, Pitch, PitchClass, Score, ScoreNote, VoiceRole,
        };
        let mut score = Score::new(
            symthaea_music_theory::Key::major(PitchClass::new(0)),
            120.0,
            4,
        );
        if note {
            score.push(ScoreNote {
                pitch: Pitch::new(PitchClass::new(0), 4),
                onset: Duration::zero(),
                duration: Duration::whole(),
                velocity: 0.7,
                role: VoiceRole::Melody,
                emphasis: Emphasis::Normal,
                section_intensity: 1.0,
            });
        }
        AtlasSource {
            id: id.to_string(),
            title: id.to_string(),
            style: "Classical".to_string(),
            score,
            form: None,
            kept,
        }
    }

    #[test]
    fn distinct_scores_each_get_their_own_representative_with_multiplicity_one() {
        let sources = vec![source("a", false, false), source("b", false, true)];
        let reps = collapse_exact_duplicates(&sources);
        assert_eq!(reps, vec![(0, 1), (1, 1)]);
    }

    #[test]
    fn identical_scores_collapse_to_one_representative_with_the_real_count() {
        // All three have empty notes -> the same exact_fingerprint.
        let sources = vec![
            source("a", false, false),
            source("b", false, false),
            source("c", false, false),
        ];
        let reps = collapse_exact_duplicates(&sources);
        assert_eq!(reps.len(), 1);
        assert_eq!(reps[0].1, 3, "multiplicity must count every duplicate");
    }

    #[test]
    fn a_kept_duplicate_is_preferred_as_the_representative() {
        let sources = vec![
            source("candidate", false, false),
            source("keeper", true, false),
        ];
        let reps = collapse_exact_duplicates(&sources);
        assert_eq!(reps, vec![(1, 2)], "the kept source (index 1) must win");
    }
}

/// Muse Atlas Phase 1 (+lenses) — a diagnostic 2D map of Muse's OWN
/// generated output (in-session candidates + persisted kept/liked pieces),
/// reusing the same structural fingerprints the diversity census measures
/// (`symthaea_music_theory::fingerprint`). Deliberately NOT the full "Muse
/// Atlas" design (no rights/provenance manifests, no external/human
/// registered works, no pgvector/tile-server, no public registration, no
/// lineage graph) — this is the minimal slice that answers "where does Muse
/// repeatedly generate, and where are the keepers" using storage/data that
/// already exists, plus a small preset menu of reweighted views over that
/// same data (`?lens=motif_form|harmony|rhythm|orchestration`, default
/// `combined`).
async fn atlas_summary(
    State(studio): State<Arc<Studio>>,
    axum::extract::Query(q): axum::extract::Query<AtlasQuery>,
) -> Result<impl IntoResponse, StatusCode> {
    let sources = gather_atlas_sources(&studio);
    let representatives = collapse_exact_duplicates(&sources);

    let raw_vectors: Vec<[f64; symthaea_music_theory::fingerprint::STRUCT_DIMS]> = representatives
        .iter()
        .map(|&(i, _)| {
            symthaea_music_theory::fingerprint::structural_fingerprint(
                &sources[i].score,
                &sources[i].form,
            )
        })
        .collect();

    // Nearest neighbor by RAW (unweighted) fingerprint — independent of the
    // active lens, so switching lenses doesn't change which piece is
    // treated as "nearby" for the "why nearby" evidence panel (`atlas_compare`
    // uses the same raw convention).
    let nearest_ids: Vec<Option<String>> = raw_vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            raw_vectors
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(j, w)| (j, symthaea_music_theory::fingerprint::dist(v, w)))
                .min_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(j, _)| sources[representatives[j].0].id.clone())
        })
        .collect();

    let (lens_name, lens_weights) = resolve_atlas_lens(q.lens.as_deref());
    let weighted_vectors: Vec<_> = raw_vectors
        .iter()
        .map(|v| symthaea_music_theory::fingerprint::weighted(v, &lens_weights))
        .collect();
    let coords = symthaea_music_theory::fingerprint::project_2d(&weighted_vectors);

    // Nearest neighbor UNDER THE ACTIVE LENS — answers "closest under the
    // lens I'm looking through" (e.g. "closest rhythmically") rather than
    // `nearest_id`'s lens-independent "overall closest". Equal to
    // `nearest_id` for the "combined" lens (uniform weights) by
    // construction, not a special case.
    let nearest_for_lens: Vec<Option<String>> = weighted_vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            weighted_vectors
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(j, w)| (j, symthaea_music_theory::fingerprint::dist(v, w)))
                .min_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(j, _)| sources[representatives[j].0].id.clone())
        })
        .collect();

    let points: Vec<serde_json::Value> = (0..representatives.len())
        .map(|r| {
            let (i, multiplicity) = representatives[r];
            let s = &sources[i];
            let (x, y) = coords[r];
            serde_json::json!({
                "id": s.id,
                "title": s.title,
                "style": s.style,
                "duration_secs": duration_secs(&s.score),
                "x": x,
                "y": y,
                "kept": s.kept,
                "nearest_id": &nearest_ids[r],
                "nearest_for_lens": &nearest_for_lens[r],
                "multiplicity": multiplicity,
            })
        })
        .collect();

    Ok(axum::Json(serde_json::json!({
        "points": points,
        "lens": lens_name,
    })))
}

#[derive(serde::Deserialize)]
struct AtlasCompareQuery {
    a: String,
    b: String,
}

/// "Why nearby" evidence panel: a real, computed per-layer distance
/// breakdown between two specific pieces (identified by their `AtlasPoint`
/// ids), from the SAME structural fingerprint the map itself uses — not a
/// fabricated or hand-wavy explanation. Always uses the RAW (unweighted)
/// fingerprint, matching `nearest_id`'s convention, so the explanation is
/// stable across lens switches.
async fn atlas_compare(
    State(studio): State<Arc<Studio>>,
    axum::extract::Query(q): axum::extract::Query<AtlasCompareQuery>,
) -> Result<impl IntoResponse, StatusCode> {
    let sources = gather_atlas_sources(&studio);
    let find = |id: &str| sources.iter().find(|s| s.id == id);
    let (Some(a), Some(b)) = (find(&q.a), find(&q.b)) else {
        return Err(StatusCode::NOT_FOUND);
    };

    let fp_a = symthaea_music_theory::fingerprint::structural_fingerprint(&a.score, &a.form);
    let fp_b = symthaea_music_theory::fingerprint::structural_fingerprint(&b.score, &b.form);
    let total = symthaea_music_theory::fingerprint::dist(&fp_a, &fp_b);
    let layer_dists = symthaea_music_theory::fingerprint::layer_dists(&fp_a, &fp_b);

    let layers: Vec<serde_json::Value> = symthaea_music_theory::fingerprint::LAYERS
        .iter()
        .zip(layer_dists.iter())
        .map(|((name, _, _), d)| serde_json::json!({ "name": name, "distance": d }))
        .collect();

    Ok(axum::Json(serde_json::json!({
        "a": a.id,
        "b": b.id,
        "a_title": a.title,
        "b_title": b.title,
        "total_distance": total,
        "layers": layers,
    })))
}

/// A section role as the frontend's label: a discrete, human-readable
/// string rather than the server's own enum spelling — kept separate so
/// changing `SectionRole`'s Rust names never becomes a wire-format change.
fn section_role_label(role: symthaea_music_theory::SectionRole) -> &'static str {
    match role {
        symthaea_music_theory::SectionRole::A => "A",
        symthaea_music_theory::SectionRole::B => "B",
        symthaea_music_theory::SectionRole::ReturnA => "A (return)",
        symthaea_music_theory::SectionRole::C => "C",
    }
}

/// The candidate's discrete motif-return structure — WHEN the engine's
/// `Form` actually applies. `has_structure: false` (empty `sections`) for
/// the 6+ form kinds (Fugue, ProgSuite, Sonata, Renaissance, Opera, the 3
/// ground forms) that bypass the period/Form pipeline entirely and so
/// never produce one — see `compose_with_spec_and_form`'s doc comment.
/// The frontend must present that honestly ("this piece's form doesn't
/// have discrete motif-return structure"), not as an empty/broken view.
/// `form` is already stored on every `Candidate` (used today for Atlas's
/// structural fingerprint) — this just exposes it directly for the first
/// time, reusing `composer::section_bar_map`'s existing per-section
/// geometry rather than recomputing it.
async fn motifs_summary(
    State(studio): State<Arc<Studio>>,
    AxPath(id): AxPath<u64>,
) -> Result<impl IntoResponse, StatusCode> {
    let (form, tempo_bpm, meter) = {
        let store = studio.candidates.lock().unwrap();
        let c = store.get(&id).ok_or(StatusCode::NOT_FOUND)?;
        (c.form.clone(), c.score.tempo_bpm, c.score.meter)
    };
    let Some(form) = form else {
        return Ok(axum::Json(serde_json::json!({
            "has_structure": false,
            "sections": [],
        })));
    };
    let bars = symthaea_music_theory::composer::section_bar_map(&form);
    let bar_to_seconds = |bar: i64| -> f64 { bar as f64 * meter as f64 / tempo_bpm as f64 * 60.0 };
    let sections: Vec<serde_json::Value> = bars
        .iter()
        .zip(form.sections.iter())
        .map(|(sb, section)| {
            serde_json::json!({
                "role": section_role_label(sb.role),
                "start_bar": sb.start_bar,
                "end_bar": sb.end_bar(),
                "start_seconds": bar_to_seconds(sb.start_bar),
                "end_seconds": bar_to_seconds(sb.end_bar()),
                "key_tonic": section.key.tonic.name(),
                "key_tonality": format!("{:?}", section.key.tonality),
            })
        })
        .collect();
    Ok(axum::Json(serde_json::json!({
        "has_structure": true,
        "sections": sections,
    })))
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

#[cfg(test)]
mod artifact_key_tests {
    use super::{keeper_artifact_key, valid_artifact_key};

    #[test]
    fn keeper_artifact_keys_reject_path_traversal() {
        assert!(valid_artifact_key("1720900000_42"));
        assert!(valid_artifact_key("version-A_1"));
        assert!(!valid_artifact_key("../keepers"));
        assert!(!valid_artifact_key("nested/path"));
        assert!(!valid_artifact_key(""));
    }

    #[test]
    fn generated_keeper_keys_are_unique_and_bounded() {
        let first = keeper_artifact_key(u128::MAX, u32::MAX, u64::MAX, 0);
        let second = keeper_artifact_key(u128::MAX, u32::MAX, u64::MAX, 1);
        assert_ne!(first, second);
        assert!(valid_artifact_key(&first));
        assert!(first.len() <= 80);
    }
}

/// End-to-end tests for the `bars` request contract fixed 2026-07-26: the
/// live `/api/compose` handler must neither silently rescale an
/// out-of-range request nor lose track of what was actually asked for.
/// Calls the real `compose` handler directly (not a mocked stand-in),
/// forcing the native renderer so the test doesn't depend on FluidSynth
/// being available in whatever environment runs `cargo test`.
#[cfg(test)]
mod compose_bars_contract_tests {
    use super::{Candidate, ComposeRequest, Studio, compose};
    use axum::Json;
    use axum::extract::State;
    use std::sync::Arc;
    use symthaea_music_theory::{GrammarPlanEvidence, Style};

    fn base_request(bars: usize) -> ComposeRequest {
        ComposeRequest {
            valence: 0.0,
            arousal: 0.5,
            energy: 0.6,
            tonic: 0,
            style: Style::Blues,
            bars,
            base_seed: 7,
            n_candidates: 1,
            prompt: String::new(),
            explore: false,
            grammar: None,
            spec: None,
            seed_stride: 1,
            dopamine: 0.5,
            serotonin: 0.5,
            noradrenaline: 0.3,
            consciousness: 0.5,
            vary_premise: false,
            renderer: Some("native".to_string()),
        }
    }

    #[tokio::test]
    async fn a_36_bar_request_reaches_the_engine_with_intent_preserved() {
        let studio = Arc::new(Studio::default());
        let response = compose(State(studio.clone()), Json(base_request(36)))
            .await
            .expect("36 bars is within COMPOSE_BARS_RANGE");
        let id = response.0.candidates[0].id;
        let store = studio.candidates.lock().unwrap();
        let candidate: &Candidate = store.get(&id).expect("candidate was stored");
        let plan = candidate
            .plan
            .as_ref()
            .expect("Blues is preset-derived, so a real plan must be stored");
        match plan {
            GrammarPlanEvidence::CallResponse(call_response_plan) => {
                assert_eq!(
                    call_response_plan.requested_bars, 36,
                    "the server must not silently rewrite the request before \
                     it reaches the engine"
                );
                assert_eq!(call_response_plan.realized_bars, 36);
                assert_eq!(call_response_plan.choruses, 3);
            }
            other => panic!("expected GrammarPlanEvidence::CallResponse, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_request_above_the_maximum_is_rejected_not_rescaled() {
        // Not `.expect_err(...)` -- `ComposeResponse` (the Ok side) has no
        // Debug impl, which `.expect_err` needs to format the failure
        // message if the call unexpectedly succeeds. Match explicitly
        // instead.
        let studio = Arc::new(Studio::default());
        match compose(State(studio), Json(base_request(37))).await {
            Err((status, message)) => {
                assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
                assert!(
                    message.contains("37"),
                    "the rejection message should name the offending value: {message}"
                );
            }
            Ok(_) => panic!("37 bars exceeds COMPOSE_BARS_RANGE and must be rejected"),
        }
    }

    #[tokio::test]
    async fn a_request_below_the_minimum_is_rejected_not_rescaled() {
        let studio = Arc::new(Studio::default());
        match compose(State(studio), Json(base_request(1))).await {
            Err((status, _)) => assert_eq!(status, axum::http::StatusCode::BAD_REQUEST),
            Ok(_) => panic!("1 bar is below COMPOSE_BARS_RANGE and must be rejected"),
        }
    }
}
