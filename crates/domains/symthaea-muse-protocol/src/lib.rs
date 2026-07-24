// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared wire types for `muse_studio`'s HTTP API
//! (`symthaea-muse/src/bin/muse_studio.rs`), consumed by both the server
//! (native, `symthaea-muse`) and the client (wasm32, `symthaea-muse-ui`).
//!
//! Before this crate existed, `symthaea-muse-ui/src/api.rs` hand-mirrored
//! these types field-for-field, because the real ones live in
//! `symthaea-muse` itself, which pulls in native-only audio deps (cpal,
//! hound, ort) that don't target wasm32. That's still true — this crate
//! is the fix: it holds nothing but the wire shapes (serde-only, no audio/
//! ONNX/native dependencies), so it compiles for both targets and the two
//! sides can no longer drift apart silently.
//!
//! Deliberately not a full mirror of every server-side type
//! (`CandidateMeta` also carries `ground: Option<GroundWorthiness>` and
//! `novelty: Option<NoveltyBreakdown>`, both richer nested types from
//! `symthaea-music-theory`) — only the fields an actual view currently
//! binds to. Add fields when a view wants them, not speculatively; that
//! was the working rule before this crate existed and it doesn't stop
//! being a good one just because the types moved.
//!
//! `#[serde(deny_unknown_fields)]` is deliberately NOT set anywhere here:
//! the server's real structs carry additional fields this crate omits by
//! design (see above), and a client deserializing a subset of a larger
//! JSON object is exactly the intended, supported use.

use serde::{Deserialize, Serialize};

/// Identifies one *rendered* audio artifact. Populated by `muse_studio`'s
/// `/api/compose` handler with the real sha256 of the rendered WAV bytes
/// (`sha256_hex(&wav)`, the same hash `piece_provenance`/the genealogy
/// ledger use) — real content-addressing, not an opaque wrapper around a
/// caller-supplied id.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RenditionArtifactId(pub String);

/// Identifies the symbolic score underlying a composition. Populated with
/// the real `serialized_sha256(&score)` of the composed `Score` — the same
/// hash `piece_provenance`/`GenealogyManifest::score_sha256` use.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScoreContentArtifactId(pub String);

/// Identifies one composition (a score realized under a specific recipe/
/// intent) — distinct from [`ScoreContentArtifactId`] (the notes alone)
/// and [`RenditionArtifactId`] (one rendered audio artifact of it).
/// Populated with the real `serialized_sha256(&recipe)` of the
/// `PieceRecipe` (intent + resolved spec) that produced this composition.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompositionArtifactId(pub String);

/// The three identities `journey.rs`'s reducer needs to name one piece
/// unambiguously: the notes, the composition they were realized as, and
/// the specific audio rendition currently playing. All three are real
/// content hashes, computed once at compose time and carried through
/// unchanged — not recomputed or reinterpreted downstream.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactIdentity {
    pub score_content: ScoreContentArtifactId,
    pub composition: CompositionArtifactId,
    pub rendition: RenditionArtifactId,
}

/// Mirrors `IdentityCard` in `symthaea-music-theory/src/describe.rs`.
/// Only `traits` is pulled in — `Candidate::title` already incorporates
/// the card's own title when one exists (see `CandidateMeta::title`'s doc
/// comment on the server side), so there's no separate use for it yet.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IdentityCard {
    pub traits: Vec<String>,
}

/// Reproducible naming metadata for a candidate.
///
/// The title itself remains `Candidate::title`; this summary explains which
/// deterministic grammar family produced it and offers stable alternatives
/// without implying that the title is a musical measurement.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TitleRecipeSummary {
    pub family: String,
    pub template_id: String,
    #[serde(default)]
    pub source_traits: Vec<String>,
    #[serde(default)]
    pub alternatives: Vec<String>,
}

/// The fields of `CandidateMeta` (`muse_studio.rs`) that a client
/// currently binds to.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Candidate {
    pub id: u64,
    pub seed: u64,
    pub duration_secs: f32,
    pub similarity: Option<f32>,
    pub renderer: String,
    /// Musical Φ: structural integration (spectral MIP over the
    /// voice×segment dependency graph). Score analysis, not consciousness.
    pub phi: f32,
    pub local_coherence: f32,
    pub global_coherence: f32,
    pub grammar: String,
    pub ending: Option<String>,
    pub card: Option<IdentityCard>,
    pub title: String,
    #[serde(default)]
    pub title_recipe: Option<TitleRecipeSummary>,
    pub why: Vec<String>,
    pub meter: u8,
    pub style: String,
    /// Set when this candidate's score hash exactly matched an EARLIER
    /// candidate already in the same compose batch — the value is that
    /// earlier candidate's `id` (also this candidate's own `id`: a
    /// duplicate reuses rather than mints one). `None` for a genuinely new
    /// score. Added for the Muse diversity-truth dedup pass; `#[serde(default)]`
    /// so an older server response with no such key still deserializes.
    #[serde(default)]
    pub duplicate_of: Option<u64>,
    /// Real content-hash identity (score/composition/rendition), computed
    /// at compose time. `#[serde(default)]` so an older server response
    /// missing this field still deserializes.
    #[serde(default)]
    pub identity: Option<ArtifactIdentity>,
}

/// `POST /api/compose`'s response envelope.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComposeResponse {
    pub candidates: Vec<Candidate>,
    /// Human-readable note about CLAP ranking (e.g. why it was skipped —
    /// feature off, no ORT, no network). Not yet surfaced by any client
    /// view.
    #[serde(default)]
    pub ranking_note: String,
    #[serde(default)]
    pub sampled_instruments: bool,
}

/// The subset of the server's `ComposeRequest` (`muse_studio.rs`) that a
/// client currently controls. The server defaults everything else
/// (grammar, spec, seed_stride, neuromodulator dims) via
/// `#[serde(default)]`, so omitting them here is a valid, complete
/// request, not a partial one.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComposeRequest {
    pub valence: f32,
    pub arousal: f32,
    pub energy: f32,
    pub tonic: i32,
    pub style: String,
    pub bars: usize,
    pub base_seed: u64,
    pub n_candidates: u64,
    pub prompt: String,
    /// Optional user-authored style spec (motifs, progression, forms,
    /// textures, ensembles) — when present it REPLACES the style's preset
    /// entirely on the server (`ComposeRequest::spec` in `muse_studio.rs`).
    /// Kept as an opaque `serde_json::Value` rather than a typed
    /// `CompositionSpec` deliberately: the client only ever round-trips
    /// this as raw JSON text in an editable textarea (load a preset/saved
    /// spec, edit, save, compose) — it never constructs or inspects
    /// individual fields — so modeling `symthaea_music_theory`'s full,
    /// native-leaning `CompositionSpec` type here would pull unnecessary
    /// weight into a wasm32 client for no behavioral benefit. The server
    /// validates and deserializes it into the real type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec: Option<serde_json::Value>,
    /// When `true`, the server premise-varies a single-candidate compose
    /// (`n_candidates == 1`): tempo third, texture budget, phrase length,
    /// ensemble persona, and mode get seed-driven variation instead of the
    /// style preset's single fixed premise (see `premise_varied` in
    /// `muse_studio.rs`). The Listen radio sets this so successive pieces
    /// stop sharing one premise; Create-mode authored composes leave it
    /// `false` so an exact spec/style stays exactly what the user asked
    /// for. Defaults to `false` on both sides — an older client omitting
    /// the key gets the old behavior.
    #[serde(default)]
    pub vary_premise: bool,
    /// Which render backend to use: `"fluidsynth"` (real soundfont),
    /// `"native"` (the in-crate synthesizer, VCSL/VSCO2-sampled where
    /// available), or `None` for the server's own default (FluidSynth
    /// when the environment provides it, native otherwise). Requesting
    /// `"fluidsynth"` degrades to native rather than erroring if it isn't
    /// actually available — the response's per-candidate `renderer` field
    /// always reports which one rendered, regardless of preference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub renderer: Option<String>,
}

/// One performed note — the wire shape `/api/notes/{id}` returns per
/// voice (`theory_realize::perform_with_spec`'s output, mirrored by
/// `pianoRoll()` in the legacy `studio/index.html`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerformedNote {
    pub start_time: f64,
    pub duration: f64,
    pub frequency: f64,
    pub velocity: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerformedVoice {
    pub name: String,
    pub instrument: String,
    pub notes: Vec<PerformedNote>,
}

/// Current version of the composition-side Listen bundle.
pub const LISTEN_COMPOSITION_BUNDLE_VERSION: u32 = 2;
/// Current version of the rendered-performance bundle.
pub const LISTEN_PERFORMANCE_BUNDLE_VERSION: u32 = 1;
/// Current version of the provenance bundle.
pub const PIECE_PROVENANCE_BUNDLE_VERSION: u32 = 1;
/// Current version of the Analyst verification bundle.
pub const ANALYST_PIECE_BUNDLE_VERSION: u32 = 1;
/// Current version of the genealogy manifest.
pub const GENEALOGY_MANIFEST_VERSION: u32 = 1;

/// How a genealogy manifest's underlying artifact came to exist. Always
/// asserted by whichever caller allocated the manifest -- never inferred
/// from content (a hash can prove what an artifact is, not why it should
/// be considered a descendant of another work).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GenealogyOrigin {
    MuseGenerated { seed: u64, style_name: String },
    ImportedScore { source_score_sha256: String },
    ImportedRecording { source_audio_sha256: String },
    ManuallyAuthored { initial_recipe_sha256: String },
}

/// Why a manifest is a descendant of its parent (or the root of its own
/// family). Only `Root` is populated by any live code path today: genealogy
/// manifests are allocated at *keep* time
/// (`muse_studio`'s `/api/keeper/{id}` handler), and nothing in the
/// compose/keep flow yet records "this kept piece was derived from that
/// kept piece" -- there is no real parent to report yet. The remaining
/// variants exist so a future derivation feature (e.g. "more like this"
/// or an explicit remix/reharmonize action recording what it varied from)
/// doesn't need a wire-format break.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GenealogyRelation {
    Root,
    Reharmonization { preserved_melody: bool },
    MotifVariation { motif_ids: Vec<String> },
    Reorchestration,
    NewPerformance,
    ManualRevision,
    ImportedArrangement,
}

/// One allocated row in the genealogy ledger. `family_id` equals `id` for a
/// root manifest and equals the root's `id` for every descendant --
/// "every piece in this family" is `WHERE family_id = ?`, "the ancestry
/// chain" is a recursive walk up `parent_id`. `manifest_sha256` is a
/// content commitment over every other field except `id`/`family_id`
/// (both are ledger-assigned sequence numbers, not content) -- it does not
/// cover the audio/recipe/score bytes themselves, only their sha256
/// pointers, which are what this manifest actually asserts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenealogyManifest {
    pub id: i64,
    pub family_id: i64,
    pub parent_id: Option<i64>,
    /// `"C"` (compositional lineage) for every V1 manifest. `"P"`/`"R"`
    /// (performance/rendition lineage) are reserved for when a rendition
    /// gets its own manifest independent of its composition's.
    pub namespace: String,
    pub relation: GenealogyRelation,
    pub origin: GenealogyOrigin,
    /// Ties this manifest to the existing keeper content-addressed layout
    /// (`data/taste/audio/<audio_key>/...`, served by `/api/keeper-audio/{key}`).
    pub audio_key: String,
    pub recipe_sha256: String,
    pub score_sha256: Option<String>,
    pub audio_sha256: String,
    pub manifest_sha256: String,
    pub created_at_unix_ms: u64,
}

/// A non-fatal limitation attached to a versioned bundle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleWarning {
    pub code: String,
    pub message: String,
}

/// Shared envelope used by all inspectable piece bundles.
///
/// `created_at_unix_ms` records when the candidate itself was created, not
/// when this HTTP response happened to be requested.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BundleEnvelope<T> {
    pub piece_id: u64,
    pub render_id: Option<String>,
    pub bundle_version: u32,
    pub created_at_unix_ms: u64,
    #[serde(default)]
    pub warnings: Vec<BundleWarning>,
    pub payload: T,
}

/// One location expressed in every coordinate system the current score can
/// ground exactly. Ticks use a fixed bundle-local `ticks_per_beat` value.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MusicalTime {
    pub tick: u64,
    pub beats: f64,
    pub seconds: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TempoPoint {
    pub at: MusicalTime,
    pub bpm: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeterPoint {
    pub at: MusicalTime,
    pub numerator: u8,
    pub denominator: u8,
}

/// A score-grounded structural region. The `source_method` makes clear
/// whether the region came from explicit composer output or a conservative
/// score annotation pass.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SectionRegion {
    pub id: String,
    pub label: String,
    pub role: String,
    pub start: MusicalTime,
    pub end: MusicalTime,
    pub intensity: f32,
    pub source_method: String,
}

/// A phrase-like region reconstructed from explicit PhraseStart and
/// Cadential score annotations. No cadence type is invented when the score
/// does not expose one.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhraseRegion {
    pub id: String,
    pub label: String,
    pub start: MusicalTime,
    pub end: MusicalTime,
    pub closes_with_cadential_marker: bool,
    pub source_method: String,
}

/// One exact symbolic score event, independent of any renderer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SymbolicNoteEvent {
    pub id: String,
    pub midi: u8,
    pub pitch_name: String,
    pub onset: MusicalTime,
    pub duration_ticks: u64,
    pub duration_beats: f64,
    pub duration_seconds: f64,
    pub velocity: f32,
    pub voice_role: String,
    pub emphasis: String,
    pub section_intensity: f32,
}

/// Epistemic status of one musical claim.
///
/// `Observed` is copied directly from a score or recipe field;
/// `Reconstructed` deterministically rebuilds a relationship from those
/// fields; `Inferred` is a bounded analysis result and must carry a method
/// and, where meaningful, a confidence value.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EvidenceStatus {
    Observed,
    Reconstructed,
    Inferred,
}

/// Method and limitations attached to inspectable musical evidence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvidenceBasis {
    pub status: EvidenceStatus,
    pub source_method: String,
    pub confidence: Option<f32>,
    #[serde(default)]
    pub limitations: Vec<String>,
}

/// The deterministic motif selected by the resolved composition recipe.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MotifDefinition {
    pub id: String,
    pub label: String,
    pub degrees: Vec<i32>,
    pub durations_beats: Vec<f64>,
    pub basis: EvidenceBasis,
}

/// One score-side occurrence related to a motif definition.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MotifOccurrence {
    pub id: String,
    pub motif_id: String,
    pub start: MusicalTime,
    pub end: MusicalTime,
    pub transformation: String,
    pub similarity: f32,
    pub source_note_ids: Vec<String>,
    pub basis: EvidenceBasis,
}

/// An exact cadence marker emitted on one or more score notes. This contract
/// deliberately does not invent a cadence type when only the arrival marker
/// is available.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CadenceEvent {
    pub id: String,
    pub at: MusicalTime,
    pub end: MusicalTime,
    pub arrival_pitch_name: String,
    pub voice_role: String,
    pub source_note_ids: Vec<String>,
    pub basis: EvidenceBasis,
}

/// A sounding pitch-class region. `home_key_degree` is populated only when
/// the region contains an exact diatonic triad in the score's declared home
/// key; it is not a modulation analysis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SonorityRegion {
    pub id: String,
    pub start: MusicalTime,
    pub end: MusicalTime,
    pub pitch_classes: Vec<String>,
    pub bass_pitch_class: Option<String>,
    pub home_key_degree: Option<u8>,
    pub home_key_function: Option<String>,
    pub basis: EvidenceBasis,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VoiceActivity {
    pub voice_role: String,
    pub note_count: usize,
}

/// Composition-side voice assignment and register evidence for a structural
/// region. This is not a claim about rendered loudness or perceptual prominence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrchestrationRegion {
    pub id: String,
    pub start: MusicalTime,
    pub end: MusicalTime,
    pub active_voices: Vec<VoiceActivity>,
    pub register_min_midi: Option<u8>,
    pub register_max_midi: Option<u8>,
    pub mean_velocity: f32,
    pub basis: EvidenceBasis,
}

/// One sample in a score-derived structural-activity curve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResonanceSample {
    pub at: MusicalTime,
    pub energy: f32,
    pub density: f32,
    pub motion: f32,
}

/// Derived perceptual proxy with explicit method and limitations. It is not
/// an objective emotional measurement.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResonanceCurve {
    pub basis: EvidenceBasis,
    pub samples: Vec<ResonanceSample>,
}

/// Composition truth used by Listen and Research. Every non-observed layer
/// carries its own evidence basis so the client can distinguish recipe truth,
/// deterministic reconstruction, and bounded inference.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ListenCompositionBundle {
    pub ticks_per_beat: u32,
    pub duration_ticks: u64,
    pub duration_beats: f64,
    pub duration_seconds: f64,
    pub form_kind: String,
    pub tempo_map: Vec<TempoPoint>,
    pub meter_map: Vec<MeterPoint>,
    pub sections: Vec<SectionRegion>,
    pub phrases: Vec<PhraseRegion>,
    pub notes: Vec<SymbolicNoteEvent>,
    #[serde(default)]
    pub motif_definitions: Vec<MotifDefinition>,
    #[serde(default)]
    pub motif_occurrences: Vec<MotifOccurrence>,
    #[serde(default)]
    pub cadences: Vec<CadenceEvent>,
    #[serde(default)]
    pub sonorities: Vec<SonorityRegion>,
    #[serde(default)]
    pub orchestration: Vec<OrchestrationRegion>,
    #[serde(default)]
    pub resonance: Option<ResonanceCurve>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PerformanceVoiceSummary {
    pub id: String,
    pub name: String,
    pub instrument: String,
    pub note_count: usize,
}

/// One realized note. Primary score voices carry an exact `source_note_id`
/// because the renderer preserves one output event per source event in order;
/// renderer-added doubling/color voices deliberately leave it `None`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PerformedNoteEvent {
    pub id: String,
    pub voice_id: String,
    pub source_note_id: Option<String>,
    pub start_seconds: f64,
    pub duration_seconds: f64,
    pub frequency_hz: f64,
    pub velocity: f64,
    pub onset_deviation_seconds: Option<f64>,
    pub duration_deviation_seconds: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ListenPerformanceBundle {
    pub duration_seconds: f64,
    pub mapping_method: String,
    pub voices: Vec<PerformanceVoiceSummary>,
    pub notes: Vec<PerformedNoteEvent>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceArtifact {
    pub kind: String,
    pub media_type: String,
    pub uri: String,
    pub sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReproducibilityClaim {
    pub symbolic_score_exact: bool,
    pub midi_exact: bool,
    pub rendered_audio_exact: bool,
    #[serde(default)]
    pub limitations: Vec<String>,
}

/// Traceability facts already present in `PieceRecipe`, plus hashes of the
/// concrete candidate artifacts held by the Studio server.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PieceProvenanceBundle {
    pub recipe_schema_version: u32,
    pub recipe_sha256: String,
    pub score_sha256: String,
    pub audio_sha256: String,
    pub seed: u64,
    pub style_name: String,
    pub renderer_name: String,
    pub renderer_version: Option<String>,
    pub muse_engine_version: String,
    pub theory_engine_version: String,
    pub muse_source_revision: Option<String>,
    pub theory_source_revision: Option<String>,
    pub soundfont_sha256: Option<String>,
    pub renderer_binary_sha256: Option<String>,
    pub performance_model_sha256: Option<String>,
    pub render_environment_sha256: Option<String>,
    pub reproduction: ReproducibilityClaim,
    pub artifacts: Vec<ProvenanceArtifact>,
}

/// One line of `data/taste/keepers.jsonl`, as returned by `GET
/// /api/keepers` — the "Liked Songs" view's data source. Mirrors the
/// `serde_json::json!` object built in `muse_studio.rs`'s `keeper()`
/// handler; only the fields `loadLiked()` in the legacy page actually
/// renders (title/spec/mode/seed/grammar/ending/phi/ts/audio_key) — the
/// entry also carries `recipe`/`hook`/`reproduction_gaps`/`novelty`/
/// `ground_worthiness`, omitted here for the same reason `Candidate`
/// omits `ground`/`novelty` above. All fields are `#[serde(default)]`:
/// the file is hand-append-only JSONL, not a versioned schema, so a
/// slightly older or newer line must still deserialize.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct KeeperEntry {
    #[serde(default)]
    pub ts: u64,
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub spec: String,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub renderer: String,
    #[serde(default)]
    pub phi: f32,
    #[serde(default)]
    pub local_coherence: f32,
    #[serde(default)]
    pub global_coherence: f32,
    #[serde(default)]
    pub grammar: String,
    #[serde(default)]
    pub ending: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub audio_key: String,
    #[serde(default)]
    pub midi_available: bool,
}

/// One diatonic triad in `GET /api/harmony/{id}`'s response — the
/// harmonic vocabulary of the candidate's key (`Key::diatonic_triad` for
/// scale degrees 1..=7), NOT a chord-by-time analysis of the piece
/// itself. See `harmony_summary()` in `muse_studio.rs` for why: the
/// engine's real per-piece `Progression` is computed mid-compose and
/// discarded before reaching `Candidate` — this is the smaller, honest
/// step of exposing what chords the key offers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiatonicChord {
    /// Scale degree, 1..=7.
    pub degree: u8,
    /// Roman-numeral harmonic-analysis label (e.g. "ii", "V", "vii\u{b0}").
    pub roman: String,
    /// Chord symbol (root name + quality suffix, e.g. "Dm", "G").
    pub symbol: String,
    /// Functional role: "Tonic", "Predominant", or "Dominant".
    pub function: String,
}

/// `GET /api/harmony/{id}`'s response.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HarmonySummary {
    /// Tonic pitch-class name (sharp spelling, e.g. "C", "D#").
    pub tonic: String,
    /// Tonality name: "Major", "Minor", or a mode debug name (e.g. "Dorian").
    pub tonality: String,
    pub chords: Vec<DiatonicChord>,
}

/// One section of `GET /api/motifs/{id}`'s response — a single entry in the
/// engine's internal `Form` (section role + bar range + key), converted to
/// a client-friendly timeline entry. See `motifs_summary()`'s doc comment
/// in `muse_studio.rs` for the full provenance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SectionInfo {
    /// Human-readable role label: "A", "B", "A (return)", or "C" — never
    /// the server's own `SectionRole` enum spelling (kept as a display
    /// string so renaming that enum is never a wire-format change).
    pub role: String,
    pub start_bar: i64,
    pub end_bar: i64,
    pub start_seconds: f64,
    pub end_seconds: f64,
    /// This section's tonic pitch-class name (sharp spelling, e.g. "C").
    pub key_tonic: String,
    /// This section's tonality, `Debug`-formatted (e.g. "Major", "Dorian").
    pub key_tonality: String,
}

/// `GET /api/motifs/{id}`'s response.
///
/// `has_structure: false` (with an empty `sections`) is the honest, expected
/// answer for the 6+ form kinds that bypass the period/`Form` pipeline
/// entirely (Fugue, ProgSuite, Sonata, Renaissance, Opera, the 3 ground
/// forms) — NOT an error state. A client MUST render that case as "this
/// piece's form doesn't have discrete motif-return structure", never as an
/// empty/broken view.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MotifsSummary {
    pub has_structure: bool,
    pub sections: Vec<SectionInfo>,
}

/// One point on the Muse Atlas — a real piece (an in-session candidate or a
/// kept/liked piece) placed at 2D coordinates by
/// `symthaea_music_theory::fingerprint::project_2d`'s deterministic top-2
/// principal components of its 40-dim structural fingerprint. This is a
/// diagnostic map of Muse's OWN generated output (see `GET /api/atlas`'s doc
/// comment in `muse_studio.rs`) — deliberately NOT the full "Muse Atlas"
/// essay's architecture (no rights/provenance model, no external/human
/// registered works, no lineage graph, no multi-lens weighting): those are
/// out of scope for this Phase 1 slice.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AtlasPoint {
    /// Either `"candidate:{id}"` (an in-memory session candidate) or
    /// `"keeper:{audio_key}"` (a persisted kept/liked piece) — the two
    /// origins share no numeric ID space, so this disambiguates them.
    pub id: String,
    pub title: String,
    pub style: String,
    pub duration_secs: f32,
    pub x: f64,
    pub y: f64,
    /// `true` for a kept/liked piece, `false` for an in-session candidate
    /// that hasn't been kept.
    pub kept: bool,
    /// This point's nearest neighbor by RAW (unweighted) structural
    /// fingerprint distance — deliberately independent of the active lens,
    /// so switching lenses doesn't change which piece is treated as
    /// "nearby" for the "why nearby" evidence panel. `None` when there's
    /// only one point on the map.
    #[serde(default)]
    pub nearest_id: Option<String>,
    /// This point's nearest neighbor by LENS-WEIGHTED structural
    /// fingerprint distance — computed against the currently active lens'
    /// weights, so it answers "closest under the lens I'm looking through"
    /// (e.g. "closest rhythmically") rather than `nearest_id`'s
    /// lens-independent "overall closest". Equal to `nearest_id` when the
    /// lens is `"combined"` (uniform weights) — that's the expected
    /// degenerate case, not a bug. `None` when there's only one point on
    /// the map. Added for the Muse diversity-truth per-lens-NN pass;
    /// `#[serde(default)]` so an older server response with no such key
    /// still deserializes.
    #[serde(default)]
    pub nearest_for_lens: Option<String>,
    /// How many exact-duplicate sources (candidates and/or keepers whose
    /// scores hash to the same
    /// `symthaea_music_theory::fingerprint::exact_fingerprint`) this point
    /// stands in for — the Atlas plots one point per distinct score, not
    /// one per candidate/keeper, so a value above 1 means several
    /// generations landed on byte-identical music. Added for the Muse
    /// diversity-truth exact-dedup pass; `#[serde(default = "one")]` so an
    /// older server response with no such key defaults to "just itself",
    /// not zero.
    #[serde(default = "one")]
    pub multiplicity: u32,
}

fn default_lens_name() -> String {
    "combined".to_string()
}

fn one() -> u32 {
    1
}

/// `GET /api/atlas`'s response.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AtlasSummary {
    #[serde(default)]
    pub points: Vec<AtlasPoint>,
    /// The lens actually used to compute `points`' `(x, y)` — echoes back
    /// the resolved value (unrecognized/absent `?lens=` query params
    /// resolve to `"combined"`), so the client can keep its lens selector
    /// in sync after a reload.
    #[serde(default = "default_lens_name")]
    pub lens: String,
}

impl Default for AtlasSummary {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            lens: default_lens_name(),
        }
    }
}

/// One fingerprint layer's L2 distance between two compared pieces — see
/// `symthaea_music_theory::fingerprint::LAYERS` for the canonical layer
/// names/order this mirrors.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AtlasLayerDistance {
    pub name: String,
    pub distance: f64,
}

/// `GET /api/atlas/compare`'s response — a real, computed per-layer
/// distance breakdown between two specific pieces on the map (the "why
/// nearby" evidence panel), not a fabricated or hand-wavy explanation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AtlasCompareResponse {
    pub a: String,
    pub b: String,
    pub a_title: String,
    pub b_title: String,
    pub total_distance: f64,
    pub layers: Vec<AtlasLayerDistance>,
    /// The lens the server actually compared under: `"combined"` (the
    /// pre-existing raw/unweighted behavior) unless the request's optional
    /// `?lens=` was recognized. Added for the Muse diversity-truth
    /// per-lens-NN pass; `#[serde(default = "default_lens_name")]` so an
    /// older server response with no such key still deserializes as the
    /// unweighted default.
    #[serde(default = "default_lens_name")]
    pub lens: String,
}

#[cfg(test)]
mod atlas_tests {
    use super::*;

    #[test]
    fn atlas_summary_round_trips() {
        let summary = AtlasSummary {
            points: vec![AtlasPoint {
                id: "candidate:3".to_string(),
                title: "Copper Meridian".to_string(),
                style: "Folk".to_string(),
                duration_secs: 42.5,
                x: 0.125,
                y: -0.5,
                kept: false,
                nearest_id: Some("keeper:abc".to_string()),
                nearest_for_lens: Some("candidate:9".to_string()),
                multiplicity: 3,
            }],
            lens: "harmony".to_string(),
        };
        let json = serde_json::to_string(&summary).unwrap();
        let back: AtlasSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(back.points.len(), 1);
        assert_eq!(back.points[0].id, "candidate:3");
        assert!((back.points[0].x - 0.125).abs() < 1e-9);
        assert_eq!(back.points[0].nearest_id.as_deref(), Some("keeper:abc"));
        assert_eq!(
            back.points[0].nearest_for_lens.as_deref(),
            Some("candidate:9")
        );
        assert_eq!(back.points[0].multiplicity, 3);
        assert_eq!(back.lens, "harmony");
    }

    #[test]
    fn atlas_point_nearest_id_defaults_to_none_when_absent() {
        let json = r#"{"id":"candidate:1","title":"t","style":"Folk","duration_secs":1.0,"x":0.0,"y":0.0,"kept":false}"#;
        let p: AtlasPoint = serde_json::from_str(json).unwrap();
        assert!(p.nearest_id.is_none());
    }

    /// An older response body with neither `nearest_for_lens` nor
    /// `multiplicity` still deserializes: the former defaults to `None`,
    /// the latter to `1` (a lone point stands only for itself), never `0`.
    #[test]
    fn atlas_point_lens_nn_and_multiplicity_default_when_absent() {
        let json = r#"{"id":"candidate:1","title":"t","style":"Folk","duration_secs":1.0,"x":0.0,"y":0.0,"kept":false}"#;
        let p: AtlasPoint = serde_json::from_str(json).unwrap();
        assert!(p.nearest_for_lens.is_none());
        assert_eq!(p.multiplicity, 1);
    }

    #[test]
    fn atlas_summary_lens_defaults_to_combined_when_absent() {
        let json = r#"{"points":[]}"#;
        let s: AtlasSummary = serde_json::from_str(json).unwrap();
        assert_eq!(s.lens, "combined");
        assert_eq!(AtlasSummary::default().lens, "combined");
    }

    #[test]
    fn atlas_compare_response_round_trips() {
        let resp = AtlasCompareResponse {
            a: "candidate:3".to_string(),
            b: "keeper:abc".to_string(),
            a_title: "Copper Meridian".to_string(),
            b_title: "Daylight Thread".to_string(),
            total_distance: 0.42,
            layers: vec![
                AtlasLayerDistance {
                    name: "form".to_string(),
                    distance: 0.04,
                },
                AtlasLayerDistance {
                    name: "harmony".to_string(),
                    distance: 0.09,
                },
            ],
            lens: "rhythm".to_string(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: AtlasCompareResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.a, "candidate:3");
        assert_eq!(back.lens, "rhythm");
        assert_eq!(back.layers.len(), 2);
        assert_eq!(back.layers[0].name, "form");
        assert!((back.total_distance - 0.42).abs() < 1e-9);
    }

    #[test]
    fn atlas_summary_defaults_to_empty_when_missing() {
        let empty: AtlasSummary = serde_json::from_str("{}").unwrap();
        assert!(empty.points.is_empty());
    }

    /// An older `/api/atlas/compare` response body with no `lens` key at
    /// all still deserializes, defaulting to `"combined"` — the unweighted
    /// behavior every response had before the per-lens-NN pass.
    #[test]
    fn atlas_compare_response_lens_defaults_to_combined_when_absent() {
        let json = r#"{"a":"candidate:1","b":"candidate:2","a_title":"A","b_title":"B",
            "total_distance":0.1,"layers":[]}"#;
        let resp: AtlasCompareResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.lens, "combined");
    }
}

// ---------------------------------------------------------------------
// Analyst / evidence-trace types
//
// `symthaea-muse`'s `analyst.rs` (the deterministic Muse Analyst v1 --
// verifies composer-asserted structure/motif/cadence/obligation claims
// against independently-recovered evidence) needs these. They were
// authored by reading `analyst.rs`'s actual construction and
// pattern-match sites, not guessed. A handful of fields
// (`GrammarProvenance::{obligations,supported_intent_axes,
// performance_features}`, `AnalystPieceBundle::external_measurements`,
// every field of `CulturalReviewSummary`) are only ever constructed as
// empty/default there and never read -- their exact shape is this
// crate's free choice, not something `analyst.rs` constrains; picked to
// match this file's existing conventions rather than left unspecified.
// ---------------------------------------------------------------------

/// Pass/fail/insufficient-evidence outcome of one [`AnalystCheck`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AnalystCheckStatus {
    Pass,
    Fail,
    InsufficientEvidence,
}

/// One deterministic structural/motif/cadence check the Analyst ran
/// against a piece, with its expected vs. observed value and the
/// [`EvidenceBasis`] backing the observation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnalystCheck {
    pub code: String,
    pub label: String,
    pub status: AnalystCheckStatus,
    pub expected: String,
    pub observed: String,
    pub basis: EvidenceBasis,
}

/// Whether a piece's evidence is complete enough to resolve routinely, or
/// needs a human reviewer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AnalystDisposition {
    RoutineEvidenceComplete,
    HumanReview,
}

/// One reason a piece was escalated to [`AnalystDisposition::HumanReview`],
/// naming who needs to look at it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnalystEscalation {
    pub code: String,
    pub reason: String,
    pub required_reviewer: String,
}

/// Coarse structural landmarks (as fractional piece positions, 0.0-1.0)
/// the Analyst located independently of any composer assertion.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnalystStructuralSummary {
    pub phrase_start_positions: Vec<f64>,
    pub cadence_positions: Vec<f64>,
    pub climax_positions: Vec<f64>,
    pub motif_positions: Vec<f64>,
    pub phrase_recurrence_intervals: Vec<f64>,
    pub ending_has_cadential_marker: bool,
    pub ending_has_motif_return: bool,
}

/// What was asked for ([`crate`]-external `MusicalIntent`-shaped request
/// fields) alongside what the realized score/performance actually came
/// out as -- lets the Analyst check realization against intent without
/// depending on the request type directly.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RequestedRealizedIntent {
    pub requested_valence: f32,
    pub requested_arousal: f32,
    pub requested_energy: f32,
    pub requested_bars: usize,
    pub requested_tonic: String,
    pub realized_tempo_bpm: f32,
    pub realized_duration_beats: f64,
    pub realized_duration_seconds: f64,
    pub realized_note_count: usize,
    pub realized_onset_density_per_second: f64,
    pub realized_register_min_midi: Option<u8>,
    pub realized_register_max_midi: Option<u8>,
}

/// Which kind of evidence backs an [`EvidenceSourceEnvelope`] -- a
/// composer's own assertion, an audio measurement, an independent
/// symbolic re-verification, or a model's prediction. Distinct from
/// [`EvidenceStatus`] (which describes how directly a *value* was
/// obtained); this describes *what produced the record at all*.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EvidenceSource {
    ComposerAssertion,
    AudioMeasured,
    SymbolicallyVerified,
    ModelPrediction,
}

/// Whether an [`EvidenceSourceEnvelope`]-wrapped claim has actually been
/// checked yet, and if so, whether the check found the claim upheld or in
/// conflict with independently-recovered evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum VerificationStatus {
    Unchecked,
    InsufficientEvidence,
    Verified,
    Discrepancy,
}

/// The shared provenance/verification envelope every individual trace
/// record (structural span, motif occurrence, cadence, obligation
/// transition, audio-integrity report) carries -- who/what produced it,
/// whether it's been independently checked yet, and an honest
/// uncertainty/limitations disclosure. Mirrors this crate's
/// [`EvidenceBasis`] pattern one level up, at the level of a whole record
/// rather than one field.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvidenceSourceEnvelope {
    pub record_id: String,
    pub schema_version: u32,
    pub source: EvidenceSource,
    pub verification_status: VerificationStatus,
    pub producer: String,
    pub producer_version: String,
    pub artifact_sha256: String,
    pub created_at_unix_ms: u64,
    #[serde(default)]
    pub dependency_record_ids: Vec<String>,
    pub uncertainty: Option<f32>,
    #[serde(default)]
    pub limitations: Vec<String>,
}

/// One composer-asserted structural span (a phrase, section, or other
/// named region) linking its start/end score events back to a parent
/// span when nested.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StructuralSpanTrace {
    pub id: String,
    pub kind: String,
    pub parent_id: Option<String>,
    pub start_event_id: String,
    pub end_event_id: String,
    pub assertion: EvidenceSourceEnvelope,
}

/// One composer-asserted occurrence of a motif family, recording which
/// invariants it claims to preserve, which dimensions changed under
/// transformation, and both a literal and a structural distance from the
/// motif's canonical form. Field names/types mirror
/// `symthaea_music_theory::grammar_trace::MotifAssertion` exactly, since
/// this is that same claim carried over the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MotifOccurrenceTrace {
    pub motif_family_id: String,
    pub motif_family_version: u32,
    pub occurrence_id: String,
    pub score_event_ids: Vec<String>,
    pub voice_or_layer: String,
    pub formal_region_id: String,
    #[serde(default)]
    pub transformation_chain: Vec<String>,
    #[serde(default)]
    pub claimed_preserved_invariants: Vec<String>,
    #[serde(default)]
    pub changed_dimensions: Vec<String>,
    pub literal_distance: f32,
    pub structural_distance: f32,
    pub role_binding: Option<String>,
    pub originating_decision_id: String,
    pub assertion: EvidenceSourceEnvelope,
}

/// One composer-asserted cadence: its proposed type, which grammar owns
/// it, the score events that prepare and arrive at it, and whether a
/// later pass altered the material downstream of it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CadenceTrace {
    pub cadence_id: String,
    pub proposed_type: String,
    pub grammar_owner: String,
    #[serde(default)]
    pub preparation_event_ids: Vec<String>,
    pub arrival_event_id: String,
    #[serde(default)]
    pub harmonic_evidence_event_ids: Vec<String>,
    #[serde(default)]
    pub melodic_evidence_event_ids: Vec<String>,
    pub altered_downstream: bool,
    pub fulfils_obligation_id: Option<String>,
    pub assertion: EvidenceSourceEnvelope,
}

/// A compositional obligation's lifecycle state -- mirrors
/// `symthaea_music_theory::grammar_trace::AssertedObligationState`
/// exactly (same 7 states, 1:1 converted when carried over the wire).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ObligationState {
    Created,
    Reinforced,
    Deferred,
    Transformed,
    Fulfilled,
    Abandoned,
    Unresolved,
}

/// One composer-asserted transition of a compositional obligation from
/// one lifecycle state to another (`from: None` for its creation).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObligationTransitionTrace {
    pub obligation_id: String,
    pub from: Option<ObligationState>,
    pub to: ObligationState,
    #[serde(default)]
    pub score_event_ids: Vec<String>,
    pub responsible_pass: String,
    pub transformation: Option<String>,
    pub assertion: EvidenceSourceEnvelope,
}

/// The composer's own full structural trace for one piece -- every
/// asserted structural span, motif occurrence, cadence, and obligation
/// transition, at a specific trace schema version. What the Analyst
/// checks independently-recovered evidence *against*.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ComposerStructuralTrace {
    pub trace_schema_version: u32,
    #[serde(default)]
    pub structures: Vec<StructuralSpanTrace>,
    #[serde(default)]
    pub motif_occurrences: Vec<MotifOccurrenceTrace>,
    #[serde(default)]
    pub cadences: Vec<CadenceTrace>,
    #[serde(default)]
    pub obligation_transitions: Vec<ObligationTransitionTrace>,
}

/// One integrity issue the audio-analysis pass detected (clipping, a
/// suspicious impulse, an unexpected silence run, etc.) -- always
/// disclosed alongside [`AudioIntegrityEvidence`], never silently
/// dropped.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceVerificationIssue {
    pub code: String,
    pub record_id: String,
    pub message: String,
}

/// The result of independently re-verifying a [`ComposerStructuralTrace`]
/// against the actual rendered/symbolic evidence -- how many records were
/// checked, how many verified clean, and every issue found along the way.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceVerificationReport {
    pub verifier_version: String,
    pub source_trace_schema_version: u32,
    pub checked_records: usize,
    pub verified_records: usize,
    pub evidence: EvidenceSourceEnvelope,
    #[serde(default)]
    pub issues: Vec<TraceVerificationIssue>,
}

/// Real measured properties of one rendered audio artifact -- loudness/
/// peak/DC/clipping/impulse statistics -- used to gate a rendition out of
/// artistic comparison before it's ever heard, not just to describe it
/// after the fact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AudioIntegrityEvidence {
    pub analyzer_version: String,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub frame_count: usize,
    pub true_peak: f32,
    pub dc_offset: f32,
    pub clipping_sample_count: usize,
    pub near_silence_fraction: f32,
    pub first_difference_rms: f32,
    pub second_difference_rms: f32,
    pub impulse_outlier_count: usize,
    pub high_frequency_proxy_ratio: f32,
    #[serde(default)]
    pub issues: Vec<String>,
    pub evidence: EvidenceSourceEnvelope,
}

/// How one composer-asserted [`MotifOccurrenceTrace`] relates to what the
/// Analyst independently recovered -- the reconciliation vocabulary that
/// keeps "the trace is internally valid" from being read as "every
/// claimed occurrence was independently rediscovered."
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MotifEvidenceRelationship {
    AssertedAndMatched,
    AssertedNotRecovered,
    InferredNotAsserted,
    Ambiguous,
    Rejected,
}

/// One reconciled pairing (or non-pairing) between a composer-asserted
/// motif occurrence and an independently-inferred one.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MotifEvidenceReconciliationEntry {
    pub relationship: MotifEvidenceRelationship,
    pub asserted_occurrence_id: Option<String>,
    pub inferred_occurrence_id: Option<String>,
    pub shared_score_event_count: usize,
    pub reason: String,
}

/// The full asserted-vs-inferred motif reconciliation for one piece, with
/// per-relationship counts kept alongside the entries themselves so a
/// caller never has to recompute (or silently misreport) the breakdown.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MotifEvidenceReconciliation {
    pub reconciler_version: String,
    pub asserted_and_matched: usize,
    pub asserted_not_recovered: usize,
    pub inferred_not_asserted: usize,
    pub ambiguous: usize,
    pub rejected: usize,
    #[serde(default)]
    pub entries: Vec<MotifEvidenceReconciliationEntry>,
}

/// Independent analysis of how a single motif was actually realized in
/// the final score -- occurrence count, transformations seen, symbolic
/// similarity to the canonical form, and separately, whether the
/// canonical material survives as an exact ingress installation versus a
/// literal or contract-valid occurrence in the finished piece (two
/// different, non-implying claims -- see this type's own
/// `canonical_installed_exactly_at_ingress` field).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MotifRealizationAnalysis {
    pub motif_id: String,
    pub occurrence_count: usize,
    #[serde(default)]
    pub transformations: Vec<String>,
    pub best_symbolic_similarity: Option<f32>,
    pub mean_symbolic_similarity: Option<f32>,
    pub appears_in_final_fifth: bool,
    pub omitted_or_below_threshold: bool,
    /// Tri-state by design: `None` means this specific ingress-vs-final
    /// distinction hasn't been checked for this motif yet, not "checked
    /// and unknown."
    pub canonical_installed_exactly_at_ingress: Option<bool>,
    pub final_literal_occurrence_count: usize,
    pub final_contract_valid_occurrence_count: usize,
    #[serde(default)]
    pub occurrence_positions: Vec<f64>,
    pub basis: EvidenceBasis,
}

/// Cultural-authenticity review status for one piece, kept independent of
/// a motif's own (possibly `not_applicable`) cultural provenance -- a
/// procedurally-generated groove motif can be culturally not-applicable
/// at the motif level while the finished piece's full style realization
/// still needs its own review.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CulturalReviewSummary {
    #[serde(default)]
    pub reviewed: bool,
    #[serde(default)]
    pub reviewer: Option<String>,
    #[serde(default)]
    pub notes: Vec<String>,
}

/// What a style/grammar realization actually used -- the grammar family
/// and its specific profile components, plus whether the realization is
/// culturally qualified. `obligations`/`supported_intent_axes`/
/// `performance_features` are honest placeholders (always empty/`None`
/// today): nothing in the Analyst pipeline populates them yet, but the
/// fields exist so a future pass can without a wire-format break.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GrammarProvenance {
    pub family: String,
    pub phrase_grammar: String,
    pub harmonic_syntax: String,
    pub performance_dialect: String,
    pub plan_kind: String,
    pub culturally_qualified: bool,
    #[serde(default)]
    pub obligations: Vec<String>,
    #[serde(default)]
    pub supported_intent_axes: Vec<String>,
    #[serde(default)]
    pub performance_features: Option<Vec<String>>,
}

/// The Analyst's full, deterministic verdict on one piece -- every check
/// it ran, the independently-recovered structural/motif evidence, the
/// composer's own trace and how it reconciles against that evidence,
/// audio-integrity gating, and the final routine-vs-human-review
/// disposition with its escalation reasons. `external_measurements` is an
/// honest placeholder (`Vec::new()` today, no consumer reads it yet) for
/// a future non-audio measurement source.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnalystPieceBundle {
    pub analyzer_version: String,
    pub grammar_family: String,
    pub phrase_grammar: String,
    pub harmonic_syntax: String,
    pub performance_dialect: String,
    pub plan_kind: String,
    pub culturally_qualified: bool,
    pub cultural_review: CulturalReviewSummary,
    pub requested_realized: RequestedRealizedIntent,
    pub structural: AnalystStructuralSummary,
    pub motif: Option<MotifRealizationAnalysis>,
    #[serde(default)]
    pub checks: Vec<AnalystCheck>,
    pub uncertainty: f32,
    pub disposition: AnalystDisposition,
    #[serde(default)]
    pub escalations: Vec<AnalystEscalation>,
    pub composer_trace: Option<ComposerStructuralTrace>,
    pub trace_verification: Option<TraceVerificationReport>,
    pub motif_reconciliation: Option<MotifEvidenceReconciliation>,
    pub audio_integrity: Option<AudioIntegrityEvidence>,
    #[serde(default)]
    pub external_measurements: Vec<String>,
    #[serde(default)]
    pub limitations: Vec<String>,
}

/// One style's real, server-computed
/// `symthaea_music_theory::Style::grammar_family()` — `family` is that
/// enum's own `#[serde(rename_all = "snake_case")]` string form (e.g.
/// `"period_sentence"`, `"raga_modal_arc"`), not a client-invented
/// grouping. Exists so `symthaea-muse-ui` can make policy-aware style
/// choices (`JourneyPolicy::Resonance`/`Contrast` picking a same-/
/// different-family style) without depending on the native
/// `symthaea-music-theory` crate, mirroring the reason this whole
/// protocol crate exists (see the module doc comment).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StyleFamily {
    pub name: String,
    pub family: String,
}

/// `GET /api/styles`'s response: every style Muse can compose in, each
/// with its real grammar family. Fetched once and cached — this is
/// static for the lifetime of a server process (styles/families are
/// compile-time data, not per-request state).
pub const STYLE_FAMILIES_VERSION: u32 = 1;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compose_request_round_trips() {
        let req = ComposeRequest {
            valence: 0.15,
            arousal: 0.45,
            energy: 0.5,
            tonic: 0,
            style: "Classical".to_string(),
            bars: 4,
            base_seed: 42,
            n_candidates: 1,
            prompt: String::new(),
            spec: None,
            vary_premise: false,
            renderer: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        // `spec: None` must not appear in the wire payload at all — the
        // server's own field is `#[serde(default)]`, so an omitted key and
        // an explicit `null` are both valid, but omitting matches what a
        // client with no spec editor open (or one who left it blank) sends.
        assert!(!json.contains("\"spec\""));
        let back: ComposeRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.style, "Classical");
        assert_eq!(back.base_seed, 42);
        assert!(back.spec.is_none());
        assert!(!back.vary_premise);
    }

    #[test]
    fn compose_request_vary_premise_defaults_off_and_round_trips_when_true() {
        // Mirrors the server-side test in `muse_studio.rs`
        // (`vary_premise_defaults_off_and_deserializes_when_present`): an
        // older client's payload without the key must deserialize to
        // `false` (authored composes keep their exact premise), and a
        // Listen-radio payload with `true` must carry it through.
        let without = r#"{"valence":0.0,"arousal":0.5,"energy":0.5,"tonic":0,
            "style":"Classical","bars":4,"base_seed":1,"n_candidates":1,
            "prompt":""}"#;
        let req: ComposeRequest = serde_json::from_str(without).unwrap();
        assert!(!req.vary_premise);

        let req = ComposeRequest {
            valence: 0.15,
            arousal: 0.45,
            energy: 0.5,
            tonic: 3,
            style: "Nocturne".to_string(),
            bars: 4,
            base_seed: 7,
            n_candidates: 1,
            prompt: String::new(),
            spec: None,
            vary_premise: true,
            renderer: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"vary_premise\":true"));
        let back: ComposeRequest = serde_json::from_str(&json).unwrap();
        assert!(back.vary_premise);
    }

    #[test]
    fn compose_request_carries_a_raw_spec_object_through() {
        // Mirrors how the legacy page's specBox and the new spec-editor
        // panel actually use this field: parse whatever JSON the user has
        // in the textarea and attach it verbatim, without this crate
        // knowing anything about CompositionSpec's shape.
        let req = ComposeRequest {
            valence: 0.0,
            arousal: 0.5,
            energy: 0.5,
            tonic: 0,
            style: "Classical".to_string(),
            bars: 4,
            base_seed: 1,
            n_candidates: 1,
            prompt: String::new(),
            spec: Some(serde_json::json!({
                "name": "Custom",
                "motif_pool": ["a", "b"],
                "ensemble_pool": [["piano", "cello", "upright_bass"]],
            })),
            vary_premise: false,
            renderer: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: ComposeRequest = serde_json::from_str(&json).unwrap();
        let spec = back.spec.expect("spec should round-trip");
        assert_eq!(spec["name"], "Custom");
        assert_eq!(spec["motif_pool"][1], "b");
    }

    #[test]
    fn compose_response_round_trips_and_defaults_missing_fields() {
        // A server that hasn't been updated to send ranking_note/
        // sampled_instruments yet (or a hand-written test fixture) should
        // still deserialize — both fields are #[serde(default)].
        let json = r#"{"candidates":[]}"#;
        let resp: ComposeResponse = serde_json::from_str(json).unwrap();
        assert!(resp.candidates.is_empty());
        assert_eq!(resp.ranking_note, "");
        assert!(!resp.sampled_instruments);
    }

    #[test]
    fn candidate_deserializes_from_a_server_response_with_extra_fields() {
        // Simulates the real server sending MORE fields than this crate's
        // Candidate models (ground/novelty) — must not fail to parse.
        let json = r#"{
            "id": 7, "seed": 42, "duration_secs": 12.5, "similarity": null,
            "renderer": "native", "phi": 0.01, "local_coherence": 0.5,
            "global_coherence": 0.1, "ground": null, "grammar": "memory",
            "ending": null, "card": null, "title": "Test Piece",
            "why": ["because"], "meter": 4, "novelty": null, "style": "Folk"
        }"#;
        let c: Candidate = serde_json::from_str(json).unwrap();
        assert_eq!(c.title, "Test Piece");
        assert_eq!(c.id, 7);
    }

    #[test]
    fn keeper_entry_deserializes_from_a_real_server_line_with_extra_fields() {
        // Simulates a real keepers.jsonl line, which also carries
        // ensemble/ground_worthiness/novelty/artifact_layout/
        // reproduction_gaps/recipe/hook — must not fail to parse.
        let json = r#"{
            "ts": 1752600000, "seed": 42, "spec": "Classical", "mode": "Major",
            "ensemble": ["Piano"], "renderer": "native", "phi": 0.12,
            "local_coherence": 0.5, "global_coherence": 0.3,
            "ground_worthiness": null, "grammar": "memory", "ending": "authentic",
            "title": "Test Piece", "novelty": null, "audio_key": "abc123",
            "artifact_layout": "keeper-directory-v1", "midi_available": true,
            "reproduction_gaps": [], "recipe": {}, "hook": []
        }"#;
        let entry: KeeperEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.title.as_deref(), Some("Test Piece"));
        assert_eq!(entry.audio_key, "abc123");
        assert!(entry.midi_available);
    }

    #[test]
    fn keeper_entry_defaults_every_field_from_an_empty_object() {
        // An older/malformed line shouldn't take down the whole list —
        // `keepers()` on the server already skips unparseable lines, but a
        // line that's valid-but-sparse JSON should still parse to defaults.
        let entry: KeeperEntry = serde_json::from_str("{}").unwrap();
        assert_eq!(entry.audio_key, "");
        assert!(entry.title.is_none());
    }

    #[test]
    fn harmony_summary_round_trips() {
        let summary = HarmonySummary {
            tonic: "D".to_string(),
            tonality: "Dorian".to_string(),
            chords: vec![DiatonicChord {
                degree: 5,
                roman: "v".to_string(),
                symbol: "Am".to_string(),
                function: "Dominant".to_string(),
            }],
        };
        let json = serde_json::to_string(&summary).unwrap();
        let back: HarmonySummary = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tonic, "D");
        assert_eq!(back.chords.len(), 1);
        assert_eq!(back.chords[0].roman, "v");
    }

    #[test]
    fn motifs_summary_with_structure_round_trips() {
        let summary = MotifsSummary {
            has_structure: true,
            sections: vec![SectionInfo {
                role: "A (return)".to_string(),
                start_bar: 8,
                end_bar: 12,
                start_seconds: 16.0,
                end_seconds: 24.0,
                key_tonic: "C".to_string(),
                key_tonality: "Major".to_string(),
            }],
        };
        let json = serde_json::to_string(&summary).unwrap();
        let back: MotifsSummary = serde_json::from_str(&json).unwrap();
        assert!(back.has_structure);
        assert_eq!(back.sections.len(), 1);
        assert_eq!(back.sections[0].role, "A (return)");
        assert_eq!(back.sections[0].end_bar, 12);
    }

    #[test]
    fn motifs_summary_without_structure_deserializes_from_server_shape() {
        // The exact shape `motifs_summary()` sends for Fugue/Sonata/etc. —
        // must parse cleanly as the honest "no structure" case, not an error.
        let json = r#"{"has_structure": false, "sections": []}"#;
        let summary: MotifsSummary = serde_json::from_str(json).unwrap();
        assert!(!summary.has_structure);
        assert!(summary.sections.is_empty());
    }

    #[test]
    fn performed_voice_round_trips() {
        let voice = PerformedVoice {
            name: "Melody".to_string(),
            instrument: "Piano".to_string(),
            notes: vec![PerformedNote {
                start_time: 0.0,
                duration: 0.5,
                frequency: 440.0,
                velocity: 0.8,
            }],
        };
        let json = serde_json::to_string(&voice).unwrap();
        let back: PerformedVoice = serde_json::from_str(&json).unwrap();
        assert_eq!(back.notes.len(), 1);
        assert_eq!(back.notes[0].frequency, 440.0);
    }

    #[test]
    fn listen_bundle_envelope_round_trips() {
        let zero = MusicalTime {
            tick: 0,
            beats: 0.0,
            seconds: 0.0,
        };
        let bundle = BundleEnvelope {
            piece_id: 9,
            render_id: Some("candidate-9".to_string()),
            bundle_version: LISTEN_COMPOSITION_BUNDLE_VERSION,
            created_at_unix_ms: 1_752_600_000_000,
            warnings: vec![BundleWarning {
                code: "motifs-unavailable".to_string(),
                message: "Motif analysis is not emitted yet.".to_string(),
            }],
            payload: ListenCompositionBundle {
                ticks_per_beat: 960,
                duration_ticks: 3840,
                duration_beats: 4.0,
                duration_seconds: 2.0,
                form_kind: "Ternary".to_string(),
                tempo_map: vec![TempoPoint {
                    at: zero,
                    bpm: 120.0,
                }],
                meter_map: vec![MeterPoint {
                    at: zero,
                    numerator: 4,
                    denominator: 4,
                }],
                sections: Vec::new(),
                phrases: Vec::new(),
                notes: Vec::new(),
                motif_definitions: Vec::new(),
                motif_occurrences: Vec::new(),
                cadences: Vec::new(),
                sonorities: Vec::new(),
                orchestration: Vec::new(),
                resonance: None,
            },
        };
        let json = serde_json::to_string(&bundle).unwrap();
        let back: BundleEnvelope<ListenCompositionBundle> = serde_json::from_str(&json).unwrap();
        assert_eq!(back.piece_id, 9);
        assert_eq!(back.payload.ticks_per_beat, 960);
        assert_eq!(back.warnings[0].code, "motifs-unavailable");
    }

    #[test]
    fn v1_composition_payload_defaults_new_evidence_layers() {
        let json = r#"{
            "ticks_per_beat":960,
            "duration_ticks":3840,
            "duration_beats":4.0,
            "duration_seconds":2.0,
            "form_kind":"Ternary",
            "tempo_map":[],
            "meter_map":[],
            "sections":[],
            "phrases":[],
            "notes":[]
        }"#;
        let bundle: ListenCompositionBundle = serde_json::from_str(json).unwrap();
        assert!(bundle.motif_definitions.is_empty());
        assert!(bundle.motif_occurrences.is_empty());
        assert!(bundle.cadences.is_empty());
        assert!(bundle.sonorities.is_empty());
        assert!(bundle.orchestration.is_empty());
        assert!(bundle.resonance.is_none());
    }

    #[test]
    fn evidence_status_serializes_as_product_language() {
        let basis = EvidenceBasis {
            status: EvidenceStatus::Reconstructed,
            source_method: "motif-occurrence-scan-v1".to_string(),
            confidence: Some(0.82),
            limitations: vec!["Symbolic proxy.".to_string()],
        };
        let json = serde_json::to_string(&basis).unwrap();
        assert!(json.contains("reconstructed"));
        assert!(json.contains("motif-occurrence-scan-v1"));
    }

    #[test]
    fn performance_mapping_allows_renderer_added_notes() {
        let event = PerformedNoteEvent {
            id: "performance-doubling-0".to_string(),
            voice_id: "doubling".to_string(),
            source_note_id: None,
            start_seconds: 1.0,
            duration_seconds: 0.5,
            frequency_hz: 440.0,
            velocity: 0.4,
            onset_deviation_seconds: None,
            duration_deviation_seconds: None,
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: PerformedNoteEvent = serde_json::from_str(&json).unwrap();
        assert!(back.source_note_id.is_none());
    }
}
