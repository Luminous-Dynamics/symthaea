// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Creative bridge: CognitiveLoop → Atelier/Muse → Art output.
//!
//! Parallel to canvas_bridge.rs — stateful manager that produces generative art
//! (visual SVG, musical PCM, poetry text) driven by consciousness state and
//! the Wallas creative cycle. Feeds aesthetic scores back into the neuromodulator
//! bath to close the creative feedback loop.

#[cfg(feature = "creative")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "creative")]
use symthaea_aesthetic::{
    AestheticConfig, AestheticFeedback, AestheticMemory, AestheticScore, AestheticTracker,
};
#[cfg(feature = "creative")]
use symthaea_atelier::{Artwork, AtelierConfig, AtelierStyle};
#[cfg(feature = "creative")]
use symthaea_canvas::CognitiveSnapshot;
#[cfg(feature = "creative")]
use symthaea_muse::{Composition, MuseConfig, MusicalState};

#[cfg(all(feature = "creative", feature = "ssm_language"))]
use symthaea_broca::creative_mode::{CreativeGating, PoeticForm, validate_poem};

// Cultural memory (ART_CULTURE_REVIEW_AND_PLAN_2026-07-06.md Phase 4): self-authored
// artifact publishing + imitation-of-self. See cultural_memory.rs module docs for
// exactly what is and isn't wired (no live mesh-send; single-node only).
#[cfg(all(feature = "creative", feature = "social-fabric"))]
use crate::cognitive_loop::cultural_memory::{CulturalMemoryManager, DOMAIN_MUSIC, DOMAIN_VISUAL};

// Gallery (VISUAL_ART_IMPROVEMENT_PLAN_2026-07-10.md Phases 4.1/4.2): persistent
// self-curating artwork store + style identity. Distinct from cultural_memory
// (a top-N canon for imitation): the gallery keeps the artifacts themselves
// with curation dynamics, and its 16D StyleEmbedding conditions future
// generation — before 2026-07-10 symthaea-gallery was a fully-tested island
// (dev-dep of atelier showcase examples only) and the embedding conditioned
// nothing.
#[cfg(all(feature = "creative", feature = "gallery"))]
use symthaea_gallery::{
    ArtModality, GalleryIndex, create_entry,
    curation::curate,
    storage::GalleryStorage,
    style::{StyleEmbedding, apply_style, compute_style},
};

/// Telemetry from the creative pipeline, stored in CycleMetadata.
#[cfg(feature = "creative")]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CreativeTelemetry {
    /// Whether any art was generated this cycle.
    pub generated: bool,
    /// Whether generation was gated by low consciousness.
    pub consciousness_gated: bool,
    /// Aesthetic score of the most recent artwork.
    pub aesthetic_score: f32,
    /// Aesthetic EMA (the system's aesthetic expectation).
    pub aesthetic_ema: f32,
    /// Dopamine delta from aesthetic feedback.
    pub dopamine_delta: f32,
    /// Serotonin delta from aesthetic feedback.
    pub serotonin_delta: f32,
    /// Surprise signal for exploration.
    pub surprise_signal: f32,
    /// Which modality was generated.
    pub modality: String,
    /// Number of generate-evaluate iterations used.
    pub iteration_count: usize,
    /// Generation time in microseconds.
    pub generation_time_us: u64,
    /// Total artworks created since startup.
    pub total_artworks: u64,
    /// Number of emotional snapshots used to build the arc (0 = flat compose).
    pub arc_length: usize,
    /// Tuning system used for this composition (empty = 12TET default).
    pub tuning_system: String,
    /// Melodic coherence score from CreativeQualityScore (0-1).
    pub melodic_coherence: f32,
    /// Emotional alignment score (0-1).
    pub emotional_alignment: f32,
    /// Number of stored motif phrases.
    pub motif_phrase_count: usize,
    /// Observer-ΔΨ of the most recently *viewed* artwork: mean consciousness
    /// level while perceiving the render minus the pre-viewing baseline
    /// (feature `art-observer`). Positive = looking at it integrated her.
    pub observer_delta_psi: f32,
    /// Mean visual surprise while viewing the most recent artwork.
    pub observer_viewing_surprise: f32,
    /// Total completed observation windows since startup.
    pub observer_verdicts: u64,
    /// Whether the most recent viewing window showed the pixel-scrambled
    /// CONTROL frame (A/B mode) rather than the artwork itself.
    pub observer_was_control: bool,
}

/// Creative output from a single cycle.
#[cfg(feature = "creative")]
#[derive(Debug, Clone)]
pub struct CreativeOutput {
    /// Generated SVG artwork, if any.
    pub artwork_svg: Option<String>,
    /// Generated PCM audio samples, if any.
    pub music_samples: Option<Vec<i16>>,
    /// Generated poem/creative text, if any.
    pub poem: Option<String>,
    /// Dance choreography keyframes, if any.
    pub dance_keyframes: Option<Vec<symthaea_muse::choreography::DanceKeyframe>>,
    /// Score notation SVG, if any.
    pub score_svg: Option<String>,
    /// Aesthetic feedback to inject into neuromodulator bath.
    pub feedback: AestheticFeedback,
    /// The `ContentRef` for a self-authored artifact published into
    /// [`CulturalMemoryManager`] this tick, if any. The outer cognitive loop
    /// (which owns the real `mesh_outbound_tx` channel, unreachable from
    /// this manager) reads this to decide whether to announce the artifact
    /// to mesh peers — see `cycle_phase_dynamics/mod.rs`'s Creative Manager
    /// block for the send side.
    #[cfg(feature = "social-fabric")]
    pub published_content: Option<crate::swarm::resonance_graph::ContentRef>,
}

#[cfg(feature = "creative")]
impl Default for CreativeOutput {
    fn default() -> Self {
        Self {
            artwork_svg: None,
            music_samples: None,
            poem: None,
            dance_keyframes: None,
            score_svg: None,
            feedback: AestheticFeedback::neutral(),
            #[cfg(feature = "social-fabric")]
            published_content: None,
        }
    }
}

/// Default path for cross-session aesthetic memory.
#[cfg(feature = "creative")]
// pub(crate): the facade's `rate_art` (Phase 2.1) writes human feedback into
// the same persisted taste file the loop's CreativeManager uses, so the two
// halves of the facade/loop split at least share one aesthetic identity.
pub(crate) const AESTHETIC_MEMORY_PATH: &str = ".claude/aesthetic_memory.json";

/// Stateful creative manager — holds atelier/muse configs, aesthetic tracker,
/// and generation state.
#[cfg(feature = "creative")]
pub(crate) struct CreativeManager {
    /// Visual art configuration.
    atelier_config: AtelierConfig,
    /// Music generation configuration.
    muse_config: MuseConfig,
    /// Aesthetic feedback tracker (EMA + reward computation).
    tracker: AestheticTracker,
    /// Persisted aesthetic memory (loaded at startup, saved on drop).
    memory: AestheticMemory,
    /// Path to persist aesthetic memory.
    memory_path: std::path::PathBuf,
    /// Consciousness threshold for creative generation.
    consciousness_threshold: f32,
    /// Generate art every N cycles.
    generation_interval: u32,
    /// Cycle counter for interval gating.
    cycles_since_generation: u32,
    /// Monotonic seed for deterministic-per-cycle generation.
    seed_counter: u64,
    /// Total artworks produced.
    total_artworks: u64,
    /// Most recent telemetry.
    last_telemetry: CreativeTelemetry,
    /// Which modality to produce next (rotates: visual → music → synesthetic → live)
    next_modality: CreativeModality,
    /// Persistent streaming improvisation engine.
    live_stream: Option<symthaea_muse::stream::MuseStream>,
    /// Active narrative episode — when set, Music modality composes from this episode
    /// rather than free composition. Cleared after each use so every episode is unique.
    active_episode: Option<symthaea_muse::narrative_bridge::NarrativeEpisode>,
    /// Default bar count for narrative episode compositions.
    default_bars: usize,
    /// Ring buffer of recent emotional snapshots for arc-driven composition.
    /// Each entry is (valence, arousal, dopamine, dynamics) from the last N cycles.
    emotional_history: std::collections::VecDeque<EmotionalSnapshot>,
    /// Motif memory — remembers melodic phrases across sessions.
    /// Loaded alongside aesthetic memory, saved on drop.
    motif_memory: symthaea_muse::motif_memory::MotifMemory,
    /// Path for motif memory persistence.
    motif_memory_path: std::path::PathBuf,
    /// Lazily-initialized Broca generator for the Poetry modality.
    /// `None` until the first Poetry tick, and stays `None` when no trained
    /// checkpoint is available (an untrained generator emits token noise).
    #[cfg(all(feature = "creative", feature = "ssm_language"))]
    poetry_generator: Option<symthaea_broca::BrocaGenerator>,
    /// Whether the poetry checkpoint load has been attempted (one-shot).
    #[cfg(all(feature = "creative", feature = "ssm_language"))]
    poetry_checkpoint_attempted: bool,
    /// Self-authored artistic history: publishing + imitation-of-self.
    /// See `cognitive_loop::cultural_memory` module docs.
    #[cfg(all(feature = "creative", feature = "social-fabric"))]
    cultural_memory: CulturalMemoryManager,
    /// The artist's eye (feature `art-eye`): ensemble critic fed real pixel
    /// percepts of rasterized candidates inside the iterate exploit phase.
    /// Persistent so its novelty-tracker/taste-model state accumulates
    /// across artworks rather than resetting per piece.
    #[cfg(all(feature = "creative", feature = "art-eye"))]
    art_critic: symthaea_atelier::critic::SelfCritic,
    /// Harmony activations at the moment of the most recent generation —
    /// what [`Self::rate_last_artwork`] hands to the tracker's 10×-weight
    /// `human_feedback` (which wants the harmony state *at generation time*,
    /// not at rating time).
    last_generation_harmonies: Option<[f32; 8]>,
    /// Persistent artwork store (feature `gallery`).
    #[cfg(all(feature = "creative", feature = "gallery"))]
    gallery_storage: GalleryStorage,
    /// In-memory gallery index, persisted after each accepted work.
    #[cfg(all(feature = "creative", feature = "gallery"))]
    gallery_index: GalleryIndex,
    /// Artistic identity derived from the gallery's recent window; conditions
    /// generation snapshots at [`GALLERY_STYLE_STRENGTH`].
    #[cfg(all(feature = "creative", feature = "gallery"))]
    gallery_style: StyleEmbedding,
    /// Observer-ΔΨ of the most recently viewed artwork (feature
    /// `art-observer`; plain fields so telemetry assembly stays exhaustive).
    observer_delta_psi: f32,
    /// Mean visual surprise during the most recent viewing window.
    observer_viewing_surprise: f32,
    /// Completed observation windows since startup.
    observer_verdicts: u64,
    /// Whether the most recent viewing was the scrambled control arm.
    observer_was_control: bool,
}

/// Gallery capacity before curation prunes (surprise-protected pruning).
#[cfg(all(feature = "creative", feature = "gallery"))]
const GALLERY_MAX_ENTRIES: usize = 200;
/// Minimum entries curation always keeps.
#[cfg(all(feature = "creative", feature = "gallery"))]
const GALLERY_MIN_ENTRIES: usize = 16;
/// Recent-window size for the style embedding.
#[cfg(all(feature = "creative", feature = "gallery"))]
const GALLERY_STYLE_WINDOW: usize = 20;
/// How strongly the gallery style conditions generation snapshots. Scaled
/// further by the embedding's own sample-count confidence inside
/// `apply_style`, so a young gallery barely nudges anything.
#[cfg(all(feature = "creative", feature = "gallery"))]
const GALLERY_STYLE_STRENGTH: f32 = 0.3;

/// Raster resolution (longest side) for the art-eye perceptual scorer.
/// 192px keeps per-candidate rasterization in the low milliseconds while
/// preserving enough detail for hue/edge/layout features.
#[cfg(all(feature = "creative", feature = "art-eye"))]
const ART_EYE_RASTER_DIM: u32 = 192;

/// A snapshot of emotional state at one cognitive cycle, used to build arcs.
#[cfg(feature = "creative")]
#[derive(Debug, Clone, Copy)]
struct EmotionalSnapshot {
    valence: f32,
    arousal: f32,
    dopamine: f32,
    dynamics: f32,
}

/// Minimum emotional history length before arc-driven composition activates.
#[cfg(feature = "creative")]
const ARC_MIN_HISTORY: usize = 4;
/// Maximum emotional history kept (ring buffer capacity).
#[cfg(feature = "creative")]
const ARC_MAX_HISTORY: usize = 16;

#[cfg(feature = "creative")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CreativeModality {
    Visual,
    Music,
    /// Broca-generated poetry: consciousness-selected poetic form (haiku/tanka/free verse).
    #[cfg(all(feature = "creative", feature = "ssm_language"))]
    Poetry,
    /// Simultaneous visual + audio with cross-modal synesthetic linkage.
    Synesthetic,
    /// Real-time improvisation: persistent streaming across ticks.
    LivePerformance,
    /// Gray-Scott reaction-diffusion: living chemical patterns.
    ReactionDiffusion,
    /// Strange attractor orbit visualization.
    StrangeAttractor,
}

#[cfg(feature = "creative")]
impl CreativeManager {
    /// Create a new CreativeManager, loading aesthetic memory from the default path.
    pub fn new() -> Self {
        Self::new_with_path(None)
    }

    /// Create a new CreativeManager with an explicit aesthetic memory path.
    ///
    /// When `path` is `None`, falls back to `AESTHETIC_MEMORY_PATH` (`.claude/aesthetic_memory.json`).
    /// This allows the cognitive loop config to specify a project-specific memory location,
    /// so Symthaea's taste identity survives across deployments and working directories.
    pub fn new_with_path(path: Option<std::path::PathBuf>) -> Self {
        let memory_path = path.unwrap_or_else(|| std::path::PathBuf::from(AESTHETIC_MEMORY_PATH));
        let memory = AestheticMemory::load(&memory_path);
        let tracker = AestheticTracker::from_memory(AestheticConfig::default(), &memory);
        #[cfg(all(feature = "creative", feature = "social-fabric"))]
        let cultural_memory_path = memory_path.with_file_name("artistic_canon.json");
        #[cfg(all(feature = "creative", feature = "gallery"))]
        let (gallery_storage, gallery_index, gallery_style) = {
            let storage = GalleryStorage::new(memory_path.with_file_name("gallery"));
            let index = storage
                .load_index()
                .unwrap_or_else(|_| GalleryIndex::new(GALLERY_MAX_ENTRIES));
            let style = compute_style(&index, GALLERY_STYLE_WINDOW);
            (storage, index, style)
        };
        Self {
            atelier_config: AtelierConfig {
                style: AtelierStyle::Composite,
                iteration_budget: 5,
                ..AtelierConfig::default()
            },
            muse_config: MuseConfig {
                duration_secs: 6.0,
                max_notes: 32,
                ..MuseConfig::default()
            },
            tracker,
            memory,
            motif_memory: {
                let motif_path = memory_path.with_file_name("motif_memory.json");
                let snapshot = symthaea_muse::motif_memory::MotifSnapshot::load(&motif_path);
                if !snapshot.is_empty() {
                    tracing::info!(
                        phrases = snapshot.phrases.len(),
                        "Loaded motif memory from {:?}",
                        motif_path
                    );
                }
                symthaea_muse::motif_memory::MotifMemory::from_snapshot(&snapshot)
            },
            motif_memory_path: memory_path.with_file_name("motif_memory.json"),
            memory_path,
            consciousness_threshold: 0.3,
            generation_interval: 10,
            cycles_since_generation: 0,
            seed_counter: 0,
            total_artworks: 0,
            last_telemetry: CreativeTelemetry::default(),
            next_modality: CreativeModality::Visual,
            live_stream: None,
            active_episode: None,
            default_bars: 8,
            emotional_history: std::collections::VecDeque::with_capacity(ARC_MAX_HISTORY),
            #[cfg(all(feature = "creative", feature = "ssm_language"))]
            poetry_generator: None,
            #[cfg(all(feature = "creative", feature = "ssm_language"))]
            poetry_checkpoint_attempted: false,
            #[cfg(all(feature = "creative", feature = "social-fabric"))]
            cultural_memory: CulturalMemoryManager::new_with_path(Some(cultural_memory_path)),
            #[cfg(all(feature = "creative", feature = "art-eye"))]
            art_critic: symthaea_atelier::critic::SelfCritic::new(),
            last_generation_harmonies: None,
            #[cfg(all(feature = "creative", feature = "gallery"))]
            gallery_storage,
            #[cfg(all(feature = "creative", feature = "gallery"))]
            gallery_index,
            #[cfg(all(feature = "creative", feature = "gallery"))]
            gallery_style,
            observer_delta_psi: 0.0,
            observer_viewing_surprise: 0.0,
            observer_verdicts: 0,
            observer_was_control: false,
        }
    }

    /// Set an active narrative episode.
    ///
    /// On the next Music modality tick, `NarrativeMusicBridge` will compose music
    /// that tells this episode's emotional story instead of free composition.
    /// The episode is cleared after use (one-shot — call again for each new episode).
    pub fn set_narrative_episode(
        &mut self,
        episode: symthaea_muse::narrative_bridge::NarrativeEpisode,
    ) {
        self.active_episode = Some(episode);
    }

    /// Clear the active narrative episode, reverting to free composition.
    pub fn clear_narrative_episode(&mut self) {
        self.active_episode = None;
    }

    /// Record the emotional state of the current cycle for arc-driven composition.
    fn record_emotional_snapshot(&mut self, snap: &CognitiveSnapshot) {
        if self.emotional_history.len() >= ARC_MAX_HISTORY {
            self.emotional_history.pop_front();
        }
        self.emotional_history.push_back(EmotionalSnapshot {
            valence: snap.valence,
            arousal: snap.arousal,
            dopamine: snap.dopamine,
            dynamics: snap.noradrenaline.clamp(0.3, 1.0),
        });
    }

    /// Build an `EmotionalArc` from the recent emotional trajectory.
    ///
    /// Returns `None` if fewer than `ARC_MIN_HISTORY` snapshots are available.
    /// The arc captures the actual emotional journey Symthaea has taken over
    /// the last N cycles, so the music tells a real story.
    fn build_arc_from_history(&self) -> Option<symthaea_muse::arc::EmotionalArc> {
        if self.emotional_history.len() < ARC_MIN_HISTORY {
            return None;
        }
        let bars: Vec<symthaea_muse::arc::BarDirective> = self
            .emotional_history
            .iter()
            .map(|s| {
                let va = symthaea_aesthetic::ValenceArousal::new(s.valence, s.arousal);
                let mut dir = symthaea_muse::arc::BarDirective::from_va(va);
                dir.dynamics = s.dynamics;
                dir
            })
            .collect();
        Some(symthaea_muse::arc::EmotionalArc::new(bars))
    }

    /// Receive human feedback on the most recent composition.
    ///
    /// `rating`: -1.0 (terrible) to +1.0 (beautiful). 0.0 = neutral.
    ///
    /// Human feedback carries 10x the weight of self-evaluation. Over time,
    /// this reshapes which harmonies, instruments, and emotional states produce
    /// music the human enjoys. Call this from a UI "like/dislike" button.
    pub fn human_feedback(
        &mut self,
        rating: f32,
        harmony_activations: &[f32; 8],
    ) -> symthaea_aesthetic::AestheticFeedback {
        let feedback = self.tracker.human_feedback(rating, harmony_activations);
        tracing::info!(
            rating = rating,
            dopamine = feedback.dopamine_delta,
            ema = self.tracker.expectation(),
            "Human feedback received — aesthetic system recalibrated"
        );
        feedback
    }

    /// Rate the most recently generated artwork with a human judgement
    /// (`rating` in [-1, 1]) using the harmony state captured at generation
    /// time. Returns `None` when nothing has been generated yet. Also
    /// flushes aesthetic memory to disk immediately — human ratings are the
    /// scarcest, highest-weight taste signal and must survive a crash.
    ///
    /// First real caller of the human-feedback path (2026-07-10): the API
    /// below existed end-to-end with zero call sites since its creation.
    pub fn rate_last_artwork(
        &mut self,
        rating: f32,
    ) -> Option<symthaea_aesthetic::AestheticFeedback> {
        let harmonies = self.last_generation_harmonies?;
        let feedback = self.human_feedback(rating, &harmonies);
        self.save_memory();
        Some(feedback)
    }

    /// Write an accepted visual work into the persistent gallery, curate,
    /// persist the index, and refresh the style embedding that conditions
    /// future generation. Storage failures are logged, never fatal — art
    /// generation must not die on a full disk.
    #[cfg(all(feature = "creative", feature = "gallery"))]
    fn gallery_record_visual(
        &mut self,
        svg: &str,
        score: AestheticScore,
        snap: &CognitiveSnapshot,
    ) {
        let filename = format!("visual-{:08}-{}.svg", snap.cycle_count, self.seed_counter);
        let saved = self
            .gallery_storage
            .ensure_dirs()
            .and_then(|_| self.gallery_storage.save_visual(&filename, svg));
        if let Err(e) = saved {
            tracing::warn!(error = %e, "gallery: failed to save visual artwork");
            return;
        }
        let entry = create_entry(
            ArtModality::Visual { filename },
            score,
            snap.harmony_activations,
            snap.cycle_count,
        );
        self.gallery_index.add(entry);
        curate(&mut self.gallery_index, GALLERY_MIN_ENTRIES);
        if let Err(e) = self.gallery_storage.save_index(&self.gallery_index) {
            tracing::warn!(error = %e, "gallery: failed to save index");
        }
        self.gallery_style = compute_style(&self.gallery_index, GALLERY_STYLE_WINDOW);
    }

    /// Number of works currently in the persistent gallery.
    #[cfg(all(feature = "creative", feature = "gallery"))]
    pub fn gallery_len(&self) -> usize {
        self.gallery_index.len()
    }

    /// Current style-identity embedding derived from the gallery.
    #[cfg(all(feature = "creative", feature = "gallery"))]
    pub fn gallery_style(&self) -> &StyleEmbedding {
        &self.gallery_style
    }

    /// Flush aesthetic + motif memory to disk. Called on drop and can be called manually.
    pub fn save_memory(&self) {
        let updated = self.tracker.to_memory(&self.memory);
        updated.save(&self.memory_path);
        // J: Persist motif memory alongside aesthetic memory
        let snapshot = self.motif_memory.to_snapshot();
        if !snapshot.is_empty() {
            if let Err(e) = snapshot.save(&self.motif_memory_path) {
                tracing::warn!("Failed to save motif memory: {e}");
            }
        }
    }

    /// Tick the creative pipeline. Returns creative output when art is generated.
    pub fn tick(&mut self, snap: &CognitiveSnapshot) -> Option<CreativeOutput> {
        self.cycles_since_generation += 1;

        // Interval gating
        if self.cycles_since_generation < self.generation_interval {
            self.last_telemetry = CreativeTelemetry {
                generated: false,
                aesthetic_ema: self.tracker.expectation(),
                total_artworks: self.total_artworks,
                ..CreativeTelemetry::default()
            };
            return None;
        }
        self.cycles_since_generation = 0;

        // Record emotional trajectory for arc-driven composition
        self.record_emotional_snapshot(snap);

        // Consciousness gating
        if (snap.consciousness_level as f32) < self.consciousness_threshold {
            self.last_telemetry = CreativeTelemetry {
                generated: false,
                consciousness_gated: true,
                aesthetic_ema: self.tracker.expectation(),
                total_artworks: self.total_artworks,
                ..CreativeTelemetry::default()
            };
            return None;
        }

        // Style conditioning (feature `gallery`, Phase 4.2): the gallery's
        // learned StyleEmbedding nudges the generation snapshot toward the
        // agent's own artistic identity. `apply_style` scales by the
        // embedding's confidence, so an empty/young gallery is a no-op.
        // Shadows `snap` for the whole generation block below — deliberate:
        // style should condition every modality, and the rated/recorded
        // harmonies must be the ones actually used to generate.
        #[cfg(feature = "gallery")]
        let conditioned_snap: CognitiveSnapshot = {
            let mut conditioned = snap.clone();
            apply_style(
                &mut conditioned.harmony_activations,
                &mut conditioned.valence,
                &mut conditioned.arousal,
                &self.gallery_style,
                GALLERY_STYLE_STRENGTH,
            );
            conditioned
        };
        #[cfg(feature = "gallery")]
        let snap: &CognitiveSnapshot = &conditioned_snap;

        let start = std::time::Instant::now();
        self.seed_counter += 1;
        self.last_generation_harmonies = Some(snap.harmony_activations);

        let mut output = CreativeOutput::default();
        let modality_name;

        match self.next_modality {
            CreativeModality::Visual => {
                // Cultural memory: imitate a past self-authored high-scoring
                // visual artifact when one exists (real structural mutation
                // via `mutate_scene`, not just a fresh independent
                // generation) — see cultural_memory.rs module docs.
                // The artist's eye (feature `art-eye`): a perceptual scorer
                // run inside the exploit phase — rasterize the candidate
                // scene, extract real pixel percepts, and let the persistent
                // SelfCritic's composite steer mutation acceptance at
                // EXTERNAL_SCORE_WEIGHT. Split field borrows: `art_critic`
                // (mutable) is disjoint from `atelier_config`/`cultural_memory`.
                #[cfg(feature = "art-eye")]
                let art_critic = &mut self.art_critic;
                #[cfg(feature = "art-eye")]
                let mut eye_scorer_impl = |scene: &symthaea_canvas::SceneNode,
                                           scorer_snap: &CognitiveSnapshot|
                 -> Option<f32> {
                    let svg = symthaea_canvas::render_svg(scene, scorer_snap.consciousness_level);
                    let input = symthaea_art_eye::see(scene, &svg, ART_EYE_RASTER_DIM).ok()?;
                    Some(art_critic.evaluate(&input, scorer_snap).composite)
                };
                #[cfg(feature = "art-eye")]
                let eye_scorer: Option<
                    &mut symthaea_atelier::iterate::ExternalScorer<'_>,
                > = Some(&mut eye_scorer_impl);
                #[cfg(not(feature = "art-eye"))]
                let eye_scorer: Option<
                    &mut symthaea_atelier::iterate::ExternalScorer<'_>,
                > = None;

                #[cfg(feature = "social-fabric")]
                let artwork = match self.cultural_memory.best_seed_for_domain(DOMAIN_VISUAL) {
                    // Imitation path: mutation candidates of a canon piece
                    // compete on the same blended internal+perceptual score
                    // as fresh generation (P1.5, 2026-07-16 — previously the
                    // eye applied to fresh generation only and canon-derived
                    // works evolved unseen).
                    Some(seed) => create_artwork_via_imitation(
                        &self.atelier_config,
                        snap,
                        seed,
                        self.seed_counter,
                        eye_scorer,
                    ),
                    None => symthaea_atelier::create_iterative_scored(
                        &self.atelier_config,
                        snap,
                        self.seed_counter,
                        eye_scorer,
                    ),
                };
                #[cfg(not(feature = "social-fabric"))]
                let artwork = symthaea_atelier::create_iterative_scored(
                    &self.atelier_config,
                    snap,
                    self.seed_counter,
                    eye_scorer,
                );
                let score = artwork.aesthetic_score;
                let feedback = self.tracker.process(&score, &snap.harmony_activations);
                output.artwork_svg = Some(artwork.svg.clone());
                output.feedback = feedback;
                modality_name = "visual";

                // Persist the accepted work into the self-curating gallery
                // and refresh the style identity (Phase 4.1/4.2).
                #[cfg(feature = "gallery")]
                self.gallery_record_visual(&artwork.svg, score, snap);

                self.record_telemetry(
                    &score,
                    &feedback,
                    modality_name,
                    artwork.generation_cycles,
                    start.elapsed(),
                );

                #[cfg(feature = "social-fabric")]
                {
                    let content_ref = self.cultural_memory.publish(
                        artwork.svg.as_bytes(),
                        DOMAIN_VISUAL,
                        self.seed_counter,
                        score.composite,
                        unix_now(),
                    );
                    output.published_content = Some(content_ref);
                    // Drain the just-queued publish immediately: nothing else
                    // in the cognitive loop ticks CulturalMemoryManager's
                    // SocialFabricManager (it is a private, per-CreativeManager
                    // instance — see cultural_memory.rs module docs), so
                    // without this the event would sit in pending_events
                    // forever and graph()/content_count() would never
                    // reflect self-authored publishes. process()'s snapshot
                    // argument is unused (see SocialFabricManager::process),
                    // so a default is exactly as informative as a real one.
                    self.cultural_memory
                        .tick_social(&super::subsystem_trait::CycleSnapshot::default());
                }

                self.next_modality = CreativeModality::Music;
            }
            CreativeModality::Music => {
                // Cultural memory: seed-level-only imitation (music/poetry have no
                // scene-graph-equivalent to structurally mutate the way atelier's
                // `mutate_scene` does for visual art — see cultural_memory.rs docs).
                // Perturbing the seed with a past high-scoring self-authored
                // composition's seed biases the RNG stream toward a "family
                // resemblance" without literally replaying it.
                #[cfg(feature = "social-fabric")]
                if let Some(parent_seed) = self.cultural_memory.best_seed_for_domain(DOMAIN_MUSIC) {
                    self.seed_counter ^= parent_seed & 0xFFFF_FFFF;
                }

                let musical_state = snapshot_to_musical_state(snap);

                // I: Select tuning system from consciousness state for all composition paths.
                let tuning = symthaea_muse::pitch::select_tuning_system(&musical_state);
                let tuning_name = format!("{tuning:?}");

                // Priority: (1) Narrative episode, (2) arc from emotional history, (3) flat compose.
                // C: Narrative autocompose — if an episode is active, let it drive the music.
                // The episode is consumed after one use (each episode is unique).
                let (mut composition, compose_mode) =
                    if let Some(episode) = self.active_episode.take() {
                        let bridge = symthaea_muse::narrative_bridge::NarrativeMusicBridge::new(
                            self.default_bars,
                        );
                        let ec = bridge.compose_episode(
                            &episode,
                            &self.muse_config,
                            &musical_state,
                            self.seed_counter,
                        );
                        (ec.composition, "narrative-music")
                    } else if let Some(arc) = self.build_arc_from_history() {
                        // F: Arc-driven composition — the music reflects Symthaea's recent
                        // emotional journey rather than a single-frame snapshot.
                        // I: Tuning system emerges from consciousness state — high Phi → just
                        // intonation, negative valence → maqamat, positive calm → gamelan, etc.
                        let arc = if tuning != symthaea_muse::pitch::TuningSystem::TwelveTET {
                            arc.with_tuning(tuning.clone())
                        } else {
                            arc
                        };
                        let comp = symthaea_muse::arc::compose_with_arc(
                            &self.muse_config,
                            &musical_state,
                            &arc,
                            self.seed_counter,
                        );
                        (comp, "arc-music")
                    } else {
                        (
                            symthaea_muse::compose(
                                &self.muse_config,
                                &musical_state,
                                self.seed_counter,
                            ),
                            "music",
                        )
                    };

                // B: Aesthetic reward signal — blend proxy score with CreativeQualityScore.
                // CreativeQualityScore uses IDyOM-inspired melodic coherence, rhythmic
                // regularity, emotional alignment, and form compliance — richer than
                // the harmony-alignment proxy alone.
                let target_va = symthaea_aesthetic::from_core_affect(
                    snap.valence,
                    snap.arousal,
                    snap.dopamine,
                    snap.serotonin,
                    snap.noradrenaline,
                );
                let proxy_score = score_composition(&composition, snap);
                let quality = symthaea_muse::creative_bench::CreativeQualityScore::evaluate(
                    &composition,
                    target_va,
                );

                // Blend: proxy covers harmony/structure; quality covers melodic craft.
                let blended_composite =
                    (proxy_score.composite * 0.5 + quality.composite * 0.5).clamp(0.0, 1.0);
                let music_score = AestheticScore {
                    composite: blended_composite,
                    order: (proxy_score.order + quality.rhythmic_regularity) / 2.0,
                    complexity: (proxy_score.complexity + quality.melodic_coherence) / 2.0,
                    surprise: (quality.composite - self.tracker.expectation()).abs(),
                    ..proxy_score
                };

                let mut feedback = self
                    .tracker
                    .process(&music_score, &snap.harmony_activations);

                // Amplify dopamine by quality composite (beautiful music = stronger reward)
                feedback.dopamine_delta += quality.composite * 0.05;

                // J: Feed notes into motif memory so phrases persist across sessions.
                for note in &composition.notes {
                    self.motif_memory.record_note(note.clone());
                }

                // M: Motif-aware melody seeding — splice replay phrases into the
                // composition if the consciousness state favors repetition. This gives
                // the music a recognizable identity that develops over time.
                let strategy = self.motif_memory.decide_strategy(&musical_state);
                if strategy != symthaea_muse::motif_memory::MotifStrategy::GenerateNew {
                    // BOUNDED drain — `next_note` is a looping generator:
                    // whenever its replay queue empties it re-decides the
                    // strategy against `musical_state` (constant here) and
                    // re-enqueues the WHOLE phrase, so under any Repeat*
                    // strategy it never returns None. An unbounded
                    // `while let` here grew one Vec to a 137GB peak (31
                    // doublings, heaptrack-verified 2026-07-16) and was the
                    // actual root cause of the "vision-manifold FEP balloon"
                    // OOMs hunted on 2026-07-11 — the kill landed in
                    // whatever cycle the Music modality ticked, which is why
                    // it looked vision-correlated. We truncate to max_notes
                    // below anyway; collecting more was pure waste.
                    let mut replay_notes = Vec::new();
                    while replay_notes.len() < self.muse_config.max_notes {
                        match self.motif_memory.next_note(&musical_state) {
                            Some(note) => replay_notes.push(note),
                            None => break,
                        }
                    }
                    if !replay_notes.is_empty() {
                        // Prepend motif replay before generated notes — the familiar
                        // phrase grounds the listener before new material develops.
                        let mut merged = replay_notes;
                        merged.extend(composition.notes.iter().cloned());
                        // Truncate to max_notes to keep density reasonable
                        merged.truncate(self.muse_config.max_notes);
                        composition.notes = merged;
                    }
                }

                output.music_samples = Some(match &composition.audio {
                    symthaea_muse::AudioData::I16(v) => v.clone(),
                    symthaea_muse::AudioData::F32(v) => {
                        v.iter().map(|s| (*s * 32767.0) as i16).collect()
                    }
                    symthaea_muse::AudioData::StereoF32(v) => v
                        .iter()
                        .map(|s| ((s[0] + s[1]) * 0.5 * 32767.0) as i16)
                        .collect(),
                });
                output.feedback = feedback;
                modality_name = compose_mode;

                self.record_telemetry(&music_score, &feedback, modality_name, 1, start.elapsed());
                // K+L: Enrich telemetry with music-specific quality breakdown + tuning
                self.last_telemetry.melodic_coherence = quality.melodic_coherence;
                self.last_telemetry.emotional_alignment = quality.emotional_alignment;
                self.last_telemetry.tuning_system = tuning_name;

                #[cfg(feature = "social-fabric")]
                if let Some(bytes) = output.music_samples.as_ref().map(|samples| {
                    samples
                        .iter()
                        .flat_map(|s| s.to_le_bytes())
                        .collect::<Vec<u8>>()
                }) {
                    let content_ref = self.cultural_memory.publish(
                        &bytes,
                        DOMAIN_MUSIC,
                        self.seed_counter,
                        music_score.composite,
                        unix_now(),
                    );
                    output.published_content = Some(content_ref);
                    // See the Visual arm's identical drain for why this call
                    // is needed (private SocialFabricManager, nothing else
                    // ticks it, process()'s snapshot arg is unused).
                    self.cultural_memory
                        .tick_social(&super::subsystem_trait::CycleSnapshot::default());
                }

                // Hand off to Poetry when the Broca language center is compiled in;
                // otherwise the rotation skips straight to Synesthetic.
                #[cfg(all(feature = "creative", feature = "ssm_language"))]
                {
                    self.next_modality = CreativeModality::Poetry;
                }
                #[cfg(not(all(feature = "creative", feature = "ssm_language")))]
                {
                    self.next_modality = CreativeModality::Synesthetic;
                }
            }
            #[cfg(all(feature = "creative", feature = "ssm_language"))]
            CreativeModality::Poetry => {
                modality_name = "poetry";

                // Lazily attempt the checkpoint load exactly once. An untrained
                // BrocaGenerator emits token noise, not poetry — without a trained
                // checkpoint we skip generation rather than feed noise to the tracker.
                if !self.poetry_checkpoint_attempted {
                    self.poetry_checkpoint_attempted = true;
                    if self.poetry_generator.is_none() {
                        self.poetry_generator = try_load_poetry_generator();
                    }
                }

                let consciousness = snap.consciousness_level as f32;
                // Form follows consciousness: low → tight haiku scaffold, mid → tanka,
                // high → free verse (enough coherence to hold shape without a scaffold).
                let gating = select_creative_gating(consciousness);
                let form = gating
                    .form_constraint
                    .clone()
                    .unwrap_or(PoeticForm::FreeVerse);

                let poem_result = self.poetry_generator.as_mut().map(|generator| {
                    // Apply creative gating: art doesn't hedge — disable the
                    // epistemic gate at weight 0 and adopt the form's repetition
                    // penalty (high for short forms, lower for refrains).
                    let cfg = generator.config_mut();
                    cfg.enable_epistemic_gate = gating.epistemic_gate_weight > 0.0;
                    if let Some(penalty) = gating.repetition_penalty_override {
                        cfg.repetition_penalty = penalty;
                    }

                    // Thought channels from the snapshot — same core mapping as
                    // broca_bridge (epistemic / emotion / consciousness).
                    let mut channels = symthaea_broca::ThoughtChannels::default();
                    // Epistemic ordinal 0 = certain: art speaks with full voice.
                    channels.set_epistemic(0.0);
                    // Serotonin doubles as warmth (contentment) in the snapshot.
                    channels.set_emotion(snap.valence, snap.arousal, snap.serotonin);
                    channels.set_consciousness(
                        consciousness,
                        snap.cantor_metacognitive_depth.clamp(0.0, 1.0),
                        (snap.living_mind_coherence as f32).clamp(0.0, 1.0),
                    );
                    generator.generate(&channels)
                });

                match poem_result {
                    Some(result)
                        if !result.text.trim().is_empty()
                            && poem_passes_quality_gate(&result.text, &form) =>
                    {
                        let score = score_poem(&result.text, &form, snap);
                        let feedback = self.tracker.process(&score, &snap.harmony_activations);
                        output.poem = Some(result.text);
                        output.feedback = feedback;
                        self.record_telemetry(&score, &feedback, modality_name, 1, start.elapsed());
                    }
                    Some(result) if !result.text.trim().is_empty() => {
                        // Generated non-empty text, but it failed the quality
                        // gate — e.g. a checkpoint trained on source code
                        // rather than language emits token noise
                        // ("nonlocal ... fetchFromGitHub ... assert_eq!")
                        // instead of anything resembling the requested form
                        // (see broca_poetry_eval's 2026-07-08 live-run
                        // finding). Skip publishing/scoring rather than feed
                        // noise to the tracker — same principle as the
                        // missing-checkpoint case below, extended to a
                        // trained-but-low-quality one.
                        tracing::debug!(
                            form = ?form,
                            "Poetry skipped: generated text failed quality gate"
                        );
                        self.last_telemetry = CreativeTelemetry {
                            generated: false,
                            modality: modality_name.to_string(),
                            aesthetic_ema: self.tracker.expectation(),
                            total_artworks: self.total_artworks,
                            ..CreativeTelemetry::default()
                        };
                    }
                    _ => {
                        // No trained checkpoint (or empty generation) — skip
                        // gracefully so the rotation never stalls on poetry.
                        tracing::debug!(
                            "Poetry skipped: no trained Broca checkpoint or empty generation"
                        );
                        self.last_telemetry = CreativeTelemetry {
                            generated: false,
                            modality: modality_name.to_string(),
                            aesthetic_ema: self.tracker.expectation(),
                            total_artworks: self.total_artworks,
                            ..CreativeTelemetry::default()
                        };
                    }
                }

                self.next_modality = CreativeModality::Synesthetic;
            }
            CreativeModality::Synesthetic => {
                let musical_state = snapshot_to_musical_state(snap);
                let composition =
                    symthaea_muse::compose(&self.muse_config, &musical_state, self.seed_counter);

                // Extract synesthetic features (correct tuple order: freq, vel, start, dur)
                let note_tuples: Vec<(f32, f32, f32, f32)> = composition
                    .notes
                    .iter()
                    .map(|n| (n.frequency, n.velocity, n.start_time, n.duration))
                    .collect();
                let tempo = symthaea_muse::rhythm::compute_tempo(&self.muse_config, &musical_state);
                let syn_frames = symthaea_aesthetic::synesthesia::extract_synesthetic_features(
                    &note_tuples,
                    tempo,
                    composition.duration_secs,
                );

                // Modulate snapshot with synesthetic features from the music
                let mut syn_snap = snap.clone();
                if !syn_frames.is_empty() {
                    let avg_hue: f32 =
                        syn_frames.iter().map(|f| f.hue).sum::<f32>() / syn_frames.len() as f32;
                    let avg_motion: f32 =
                        syn_frames.iter().map(|f| f.motion).sum::<f32>() / syn_frames.len() as f32;
                    // Warm hues → positive valence, cool → negative
                    let hue_valence = if avg_hue < 180.0 {
                        (1.0 - avg_hue / 180.0) * 0.3
                    } else {
                        -((avg_hue - 180.0) / 180.0) * 0.3
                    };
                    syn_snap.valence = (syn_snap.valence + hue_valence).clamp(-1.0, 1.0);
                    syn_snap.arousal = (syn_snap.arousal + avg_motion * 0.15).clamp(0.0, 1.0);
                }

                // Generate visual art modulated by musical synesthesia
                let artwork = symthaea_atelier::create_artwork_iterative(
                    &self.atelier_config,
                    &syn_snap,
                    self.seed_counter,
                );

                // Blend scores from both modalities
                let music_score = score_composition(&composition, snap);
                let visual_score = artwork.aesthetic_score;
                let music_fb = self
                    .tracker
                    .process(&music_score, &snap.harmony_activations);
                let visual_fb = self
                    .tracker
                    .process(&visual_score, &snap.harmony_activations);
                let blended =
                    symthaea_aesthetic::synesthesia::blend_feedbacks(&[music_fb, visual_fb]);

                output.artwork_svg = Some(artwork.svg);
                output.music_samples = Some(match &composition.audio {
                    symthaea_muse::AudioData::I16(v) => v.clone(),
                    symthaea_muse::AudioData::F32(v) => {
                        v.iter().map(|s| (*s * 32767.0) as i16).collect()
                    }
                    symthaea_muse::AudioData::StereoF32(v) => v
                        .iter()
                        .map(|s| ((s[0] + s[1]) * 0.5 * 32767.0) as i16)
                        .collect(),
                });
                output.feedback = blended;
                modality_name = "synesthetic";

                let combined = AestheticScore {
                    order: (music_score.order + visual_score.order) / 2.0,
                    complexity: (music_score.complexity + visual_score.complexity) / 2.0,
                    surprise: (music_score.surprise + visual_score.surprise) / 2.0,
                    harmony: (music_score.harmony + visual_score.harmony) / 2.0,
                    birkhoff: (music_score.birkhoff + visual_score.birkhoff) / 2.0,
                    composite: (music_score.composite + visual_score.composite) / 2.0,
                };
                self.record_telemetry(
                    &combined,
                    &blended,
                    modality_name,
                    artwork.generation_cycles + 1,
                    start.elapsed(),
                );

                self.next_modality = CreativeModality::LivePerformance;
            }
            CreativeModality::LivePerformance => {
                // Initialize persistent stream on first use
                if self.live_stream.is_none() {
                    self.live_stream = Some(symthaea_muse::stream::MuseStream::new(
                        self.seed_counter,
                        self.muse_config.clone(),
                    ));
                }

                let stream = self.live_stream.as_mut().unwrap();
                let musical_state = snapshot_to_musical_state(snap);
                stream.update_state(&musical_state);

                let notes = stream.generate_batch(4);

                if !notes.is_empty() {
                    // Choreography from live notes
                    let dance =
                        symthaea_muse::choreography::choreograph(&notes, &musical_state, 4.0);
                    output.dance_keyframes = Some(dance.keyframes);

                    // Synesthetic modulation (correct tuple order)
                    let note_tuples: Vec<(f32, f32, f32, f32)> = notes
                        .iter()
                        .map(|n| (n.frequency, n.velocity, n.start_time, n.duration))
                        .collect();
                    let syn_frames = symthaea_aesthetic::synesthesia::extract_synesthetic_features(
                        &note_tuples,
                        stream.tempo(),
                        4.0,
                    );

                    let mut syn_snap = snap.clone();
                    if let Some(frame) = syn_frames.first() {
                        let hue_v = if frame.hue < 180.0 { 0.2 } else { -0.2 };
                        syn_snap.valence = (syn_snap.valence + hue_v).clamp(-1.0, 1.0);
                        syn_snap.arousal = (syn_snap.arousal + frame.motion * 0.15).clamp(0.0, 1.0);
                    }

                    let artwork = symthaea_atelier::create_artwork(
                        &self.atelier_config,
                        &syn_snap,
                        self.seed_counter + stream.notes_generated(),
                    );

                    let visual_score = artwork.aesthetic_score;
                    let feedback = self
                        .tracker
                        .process(&visual_score, &snap.harmony_activations);
                    output.artwork_svg = Some(artwork.svg);
                    output.feedback = feedback;
                    modality_name = "live-performance";

                    self.record_telemetry(
                        &visual_score,
                        &feedback,
                        modality_name,
                        notes.len(),
                        start.elapsed(),
                    );
                } else {
                    modality_name = "live-performance";
                    self.last_telemetry = CreativeTelemetry {
                        generated: false,
                        modality: modality_name.to_string(),
                        total_artworks: self.total_artworks,
                        ..CreativeTelemetry::default()
                    };
                }

                self.next_modality = CreativeModality::ReactionDiffusion;
            }
            CreativeModality::ReactionDiffusion => {
                let rd_config = AtelierConfig {
                    style: symthaea_atelier::AtelierStyle::ReactionDiffusion,
                    ..self.atelier_config.clone()
                };
                let artwork = symthaea_atelier::create_artwork(&rd_config, snap, self.seed_counter);
                let score = artwork.aesthetic_score;
                let feedback = self.tracker.process(&score, &snap.harmony_activations);
                output.artwork_svg = Some(artwork.svg);
                output.feedback = feedback;
                modality_name = "reaction-diffusion";
                self.record_telemetry(&score, &feedback, modality_name, 1, start.elapsed());
                self.next_modality = CreativeModality::StrangeAttractor;
            }
            CreativeModality::StrangeAttractor => {
                let sa_config = AtelierConfig {
                    style: symthaea_atelier::AtelierStyle::StrangeAttractor,
                    ..self.atelier_config.clone()
                };
                let artwork = symthaea_atelier::create_artwork(&sa_config, snap, self.seed_counter);
                let score = artwork.aesthetic_score;
                let feedback = self.tracker.process(&score, &snap.harmony_activations);
                output.artwork_svg = Some(artwork.svg);
                output.feedback = feedback;
                modality_name = "strange-attractor";
                self.record_telemetry(&score, &feedback, modality_name, 1, start.elapsed());
                self.next_modality = CreativeModality::Visual;
            }
        }

        self.total_artworks += 1;
        Some(output)
    }

    fn record_telemetry(
        &mut self,
        score: &AestheticScore,
        feedback: &AestheticFeedback,
        modality: &str,
        iterations: usize,
        elapsed: std::time::Duration,
    ) {
        self.last_telemetry = CreativeTelemetry {
            generated: true,
            consciousness_gated: false,
            aesthetic_score: score.composite,
            aesthetic_ema: self.tracker.expectation(),
            dopamine_delta: feedback.dopamine_delta,
            serotonin_delta: feedback.serotonin_delta,
            surprise_signal: feedback.surprise_signal,
            modality: modality.to_string(),
            iteration_count: iterations,
            generation_time_us: elapsed.as_micros() as u64,
            total_artworks: self.total_artworks + 1,
            arc_length: self.emotional_history.len(),
            tuning_system: String::new(), // populated by caller for music modalities
            melodic_coherence: 0.0,
            emotional_alignment: 0.0,
            motif_phrase_count: self.motif_memory.phrase_count(),
            // Observer values persist across generations (a verdict arrives
            // cycles after the artwork that produced it).
            observer_delta_psi: self.observer_delta_psi,
            observer_viewing_surprise: self.observer_viewing_surprise,
            observer_verdicts: self.observer_verdicts,
            observer_was_control: self.observer_was_control,
        };
    }

    /// Record a completed observation window (feature `art-observer`): the
    /// measured change in Symthaea's own consciousness level while looking
    /// at her artwork, plus mean visual surprise during viewing. Returns a
    /// small bath-ready reward when viewing raised integration (Δψ > 0) —
    /// art that integrates the observer is rewarding. Negative Δψ is
    /// recorded but NOT punished: over a short open-loop viewing window ψ
    /// moves for many reasons (this is a first-order probe, and the
    /// asymmetry keeps its confounds from becoming a penalty signal).
    ///
    /// `was_control` marks the scrambled-frame A/B arm: control verdicts are
    /// recorded for the experiment but NEVER rewarded — a probe frame is not
    /// her art, and rewarding it would corrupt the very comparison the
    /// control exists to make.
    pub fn record_observer_verdict(
        &mut self,
        delta_psi: f32,
        viewing_surprise: f32,
        was_control: bool,
    ) -> AestheticFeedback {
        /// Dopamine per unit of positive Δψ.
        const OBSERVER_DOPAMINE_GAIN: f32 = 0.5;
        /// Hard cap well under compute_feedback's ±0.15 dopamine bound.
        const OBSERVER_DOPAMINE_CAP: f32 = 0.05;

        self.observer_delta_psi = delta_psi;
        self.observer_viewing_surprise = viewing_surprise;
        self.observer_verdicts += 1;
        self.observer_was_control = was_control;
        self.last_telemetry.observer_delta_psi = delta_psi;
        self.last_telemetry.observer_viewing_surprise = viewing_surprise;
        self.last_telemetry.observer_verdicts = self.observer_verdicts;
        self.last_telemetry.observer_was_control = was_control;

        tracing::info!(
            delta_psi,
            viewing_surprise,
            was_control,
            verdicts = self.observer_verdicts,
            "Observer verdict: she looked at {}",
            if was_control {
                "a scrambled control frame"
            } else {
                "her artwork"
            }
        );

        if was_control {
            return AestheticFeedback::neutral();
        }

        AestheticFeedback {
            dopamine_delta: (delta_psi.max(0.0) * OBSERVER_DOPAMINE_GAIN)
                .min(OBSERVER_DOPAMINE_CAP),
            serotonin_delta: 0.0,
            surprise_signal: viewing_surprise.clamp(0.0, 1.0) * 0.05,
            harmony_projection: [0.0; 8],
        }
    }

    /// Most recent telemetry.
    pub fn last_telemetry(&self) -> &CreativeTelemetry {
        &self.last_telemetry
    }

    /// Current aesthetic expectation (EMA).
    pub fn aesthetic_expectation(&self) -> f32 {
        self.tracker.expectation()
    }

    /// Total artworks produced.
    pub fn total_artworks(&self) -> u64 {
        self.total_artworks
    }

    /// Long-term harmony bias: which of the 8 harmonies historically correlate
    /// with beautiful output. Values drift slowly (alpha 0.01) over hundreds of
    /// compositions. Used by MuseManager to gently bias generation toward
    /// Symthaea's evolving aesthetic identity.
    pub fn harmony_bias(&self) -> &[f32; 8] {
        self.tracker.harmony_bias()
    }

    /// Feed an external aesthetic score into the tracker (self-listening loop).
    ///
    /// Called when MuseManager finishes a composition: the creative_bench scores
    /// get converted to an AestheticScore and processed through the tracker,
    /// updating the EMA and harmony bias so Symthaea's taste evolves from its
    /// own output — not just from cognitive state.
    pub fn process_external_score(
        &mut self,
        score: &symthaea_aesthetic::AestheticScore,
        harmony_activations: &[f32; 8],
    ) -> symthaea_aesthetic::AestheticFeedback {
        self.tracker.process(score, harmony_activations)
    }
}

#[cfg(feature = "creative")]
impl Default for CreativeManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "creative")]
impl Drop for CreativeManager {
    fn drop(&mut self) {
        self.save_memory();
        // `cultural_memory` (when compiled in) persists itself via its own
        // `Drop` impl — no extra action needed here.
    }
}

/// Current Unix time in seconds, used to timestamp published cultural-memory
/// artifacts. Falls back to 0 if the clock is somehow before the epoch.
#[cfg(all(feature = "creative", feature = "social-fabric"))]
fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Build visual art by imitating a past self-authored artifact: regenerate
/// its base scene deterministically from `parent_seed`, then apply a real
/// structural mutation (`symthaea_atelier::iterate::mutate_scene`) using a
/// separate RNG stream derived from `mutation_seed`.
///
/// Honesty note: only the *seed* is retained (see
/// `cognitive_loop::cultural_memory`), not the original `CognitiveSnapshot`
/// the parent was generated under — so this regenerates the parent's base
/// scene under the *current* snapshot, not the one active at original
/// creation time. That still produces a genuine structural lineage (same
/// RNG draws, current mood) even though it isn't a byte-for-byte replay of
/// the original artifact. A full-fidelity ancestor replay would require
/// snapshotting the entire cognitive state at publish time, which is out of
/// scope for this pass.
/// Number of mutation candidates evaluated per imitation. Small on purpose:
/// each candidate costs one `score_scene` plus (with `art-eye`) one
/// rasterize+critique pass, and this runs inside the live cycle budget.
#[cfg(all(feature = "creative", feature = "social-fabric"))]
const IMITATION_CANDIDATES: usize = 4;

#[cfg(all(feature = "creative", feature = "social-fabric"))]
fn create_artwork_via_imitation(
    config: &AtelierConfig,
    snap: &CognitiveSnapshot,
    parent_seed: u64,
    mutation_seed: u64,
    mut eye_scorer: Option<&mut symthaea_atelier::iterate::ExternalScorer<'_>>,
) -> symthaea_atelier::Artwork {
    use rand::SeedableRng;

    let mut parent_rng = rand::rngs::StdRng::seed_from_u64(parent_seed);
    let parent_scene = symthaea_atelier::generate(config, snap, &mut parent_rng);

    // Selection mirrors atelier's iterate loop exactly: candidates compete
    // on the blended internal+perceptual composite (EXTERNAL_SCORE_WEIGHT),
    // while `Artwork.aesthetic_score` reports the internal composite. Until
    // 2026-07-16 this path mutated ONCE and never looked at the result —
    // canon-derived works (the dominant path once a canon exists) evolved
    // entirely unseen (review P1.5).
    let blend = |internal: f32, ext: Option<f32>| match ext {
        Some(e) => {
            (1.0 - symthaea_atelier::iterate::EXTERNAL_SCORE_WEIGHT) * internal
                + symthaea_atelier::iterate::EXTERNAL_SCORE_WEIGHT * e
        }
        None => internal,
    };

    // (scene, internal AestheticScore, blended selection score)
    let mut best = None;
    for i in 0..IMITATION_CANDIDATES as u64 {
        // Decorrelate each candidate's mutation RNG stream from the parent's
        // generation stream (same splitmix-style scramble atelier's own
        // iterate.rs uses), varied per candidate.
        let mut mutation_rng = rand::rngs::StdRng::seed_from_u64(
            mutation_seed
                .wrapping_add(i)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(0x2545_F491_4F6C_DD1D),
        );
        let mutant = symthaea_atelier::iterate::mutate_scene(
            &parent_scene,
            &mut mutation_rng,
            0.4, // moderate mutation strength — enough to vary, not enough to erase lineage
            (config.width, config.height),
        );
        let internal = symthaea_atelier::score_scene(&mutant, snap);
        let blended = blend(
            internal.composite,
            eye_scorer.as_mut().and_then(|s| s(&mutant, snap)),
        );
        if best
            .as_ref()
            .is_none_or(|&(_, _, b): &(_, _, f32)| blended > b)
        {
            best = Some((mutant, internal, blended));
        }
    }
    let (scene, score, _) = best.expect("IMITATION_CANDIDATES > 0");
    let svg = symthaea_canvas::render_svg(&scene, snap.consciousness_level);

    symthaea_atelier::Artwork {
        scene,
        svg,
        aesthetic_score: score,
        style: config.style,
        generation_cycles: IMITATION_CANDIDATES,
    }
}

/// Build a `CognitiveSnapshot` from the per-cycle `CycleSnapshot` + neuromod bath.
///
/// Deliberate duplicate of `canvas_bridge::snapshot_from_cycle`: that function
/// is gated `#[cfg(feature = "canvas")]` (the whole `canvas_bridge` module is),
/// but `creative` depends on the `symthaea-canvas` crate without enabling the
/// `canvas` module feature — so `CreativeManager::tick` needs its own copy to
/// stay reachable when `creative` is enabled without `canvas`. Keep the two in
/// sync if the mapping ever changes.
///
/// Topology fields start at dormant defaults here; the live call site in
/// `cycle_phase_dynamics` enriches the returned snapshot with real Betti
/// numbers and Cantor depth before ticking the manager (2026-07-10).
#[cfg(feature = "creative")]
pub(crate) fn snapshot_from_cycle(
    cs: &super::subsystem_trait::CycleSnapshot,
    bath: &super::neuromodulators::NeuromodulatorBath,
) -> CognitiveSnapshot {
    let hc = (cs.harmonic_coherence as f32).clamp(0.05, 1.0);
    let mut harmony_activations = [0.0f32; 8];
    for (i, activation) in harmony_activations.iter_mut().enumerate() {
        let sigmoid = 1.0 / (1.0 + (-cs.compressed_state[i] * 3.0).exp());
        *activation = sigmoid * hc;
    }

    CognitiveSnapshot {
        consciousness_level: cs.unified_psi,
        prediction_error: cs.prediction_error,
        living_mind_vitality: cs.dissipative_health,
        living_mind_coherence: cs.coherence as f64,
        dopamine: bath.dopamine.effective(),
        noradrenaline: bath.noradrenaline.effective(),
        serotonin: bath.serotonin.effective(),
        acetylcholine: bath.acetylcholine.effective(),
        oxytocin: bath.oxytocin.effective(),
        gaba: bath.gaba.effective(),
        allostatic_load: bath.allostatic_load,
        valence: cs.valence,
        arousal: cs.arousal,
        harmony_activations,
        thought_vector: cs.compressed_state[..32].to_vec(),
        cycle_count: cs.cycle_number,
        ..CognitiveSnapshot::dormant()
    }
}

/// Convert a CognitiveSnapshot to MusicalState, enriched with VA emotion space.
///
/// Uses Russell's circumplex model to derive musically-validated parameters
/// from the cognitive state, then blends them with the raw neuromodulator values.
#[cfg(feature = "creative")]
fn snapshot_to_musical_state(snap: &CognitiveSnapshot) -> MusicalState {
    // Derive VA coordinate from core affect
    let va = symthaea_aesthetic::from_core_affect(
        snap.valence,
        snap.arousal,
        snap.dopamine,
        snap.serotonin,
        snap.noradrenaline,
    );
    let params = symthaea_aesthetic::MusicalParams::from_va(va);

    // Blend VA-derived arousal with raw (VA is more musically calibrated)
    let blended_arousal = snap.arousal * 0.4 + va.arousal * 0.6;

    MusicalState {
        harmony_activations: snap.harmony_activations,
        dopamine: snap.dopamine * params.dynamics,
        serotonin: snap.serotonin,
        noradrenaline: snap.noradrenaline,
        arousal: blended_arousal,
        valence: va.valence, // use VA-calibrated valence for scale building
        consciousness_level: snap.consciousness_level as f32,
        prediction_error: snap.prediction_error,
    }
}

/// Score a musical composition aesthetically.
///
/// Uses harmony alignment and note diversity as proxies for musical beauty.
#[cfg(feature = "creative")]
fn score_composition(comp: &Composition, snap: &CognitiveSnapshot) -> AestheticScore {
    // Order: rhythmic regularity (notes evenly spaced)
    let order = if comp.notes.len() >= 2 {
        let intervals: Vec<f32> = comp
            .notes
            .windows(2)
            .map(|w| w[1].start_time - w[0].start_time)
            .collect();
        let mean_interval = intervals.iter().sum::<f32>() / intervals.len() as f32;
        if mean_interval > 0.0 {
            let variance = intervals
                .iter()
                .map(|&i| (i - mean_interval).powi(2))
                .sum::<f32>()
                / intervals.len() as f32;
            let cv = variance.sqrt() / mean_interval;
            (1.0 - cv).clamp(0.0, 1.0)
        } else {
            0.5
        }
    } else {
        0.3
    };

    // Complexity: pitch diversity
    let mut unique_pitches: Vec<f32> = comp
        .notes
        .iter()
        .map(|n| (n.frequency * 10.0).round())
        .collect();
    unique_pitches.sort_by(|a, b| a.total_cmp(b));
    unique_pitches.dedup();
    let complexity = if comp.notes.is_empty() {
        0.1
    } else {
        (unique_pitches.len() as f32 / comp.notes.len() as f32).clamp(0.0, 1.0)
    };

    // Harmony: mean harmony activation
    let harmony = snap.harmony_activations.iter().sum::<f32>() / 8.0;

    let mut score = AestheticScore {
        order,
        complexity,
        surprise: 0.0,
        harmony,
        birkhoff: if complexity > 0.01 {
            (order / complexity).clamp(0.0, 1.0)
        } else {
            0.0
        },
        composite: 0.0,
    };
    score.compute_composite();
    score
}

/// Checkpoint paths tried for the poetry generator, in order.
#[cfg(all(feature = "creative", feature = "ssm_language"))]
const POETRY_CHECKPOINT_PATHS: &[&str] = &[
    "crates/domains/symthaea-broca/data/models/broca-checkpoint-latest.bin",
    "crates/symthaea-broca/data/broca-cfc-v2.bin", // legacy layout, matches BrocaManager
];

/// Attempt to load a trained BrocaGenerator for poetry generation.
///
/// Mirrors `BrocaManager::try_load_checkpoint`, but deliberately does NOT fall
/// back to a fresh untrained generator — untrained output is token noise, and
/// the Poetry modality skips gracefully instead of scoring noise.
#[cfg(all(feature = "creative", feature = "ssm_language"))]
fn try_load_poetry_generator() -> Option<symthaea_broca::BrocaGenerator> {
    // MUST match the phrase the checkpoint was trained under: checkpoint
    // restore re-derives the thought-encoder HDC bases from this genesis
    // (from_checkpoint_struct → Self::new(genesis, ..)), and a different
    // phrase silently misaligns the restored weights with their inputs.
    // The curriculum training pipeline (broca_curriculum_sync.rs) and every
    // broca eval bin use "symthaea luminous dynamics" — that is what
    // broca-checkpoint-latest.bin is trained with.
    let genesis = symthaea_core::genesis::GenesisSeed::from_phrase("symthaea luminous dynamics");
    for path in POETRY_CHECKPOINT_PATHS {
        if !std::path::Path::new(path).exists() {
            tracing::debug!("Poetry checkpoint not found at {path}, skipping");
            continue;
        }
        match symthaea_broca::BrocaGenerator::from_checkpoint(path, &genesis) {
            Ok((mut generator, _adam, _proj, _lm_config)) => {
                // Fix (2026-07-26, SYMTHAEA_COGNITION_IMPROVEMENT_PLAN_2026-07-21.md
                // follow-up): the checkpoint's own saved config always carries
                // `SamplingStrategy::Greedy` regardless of what's set here (config is
                // restored wholesale from the checkpoint, not from BrocaConfig::default()),
                // so overriding it must happen post-load, not via any default. Greedy
                // decoding was measured to collapse to the same first-1-2 tokens across
                // most distinct thought inputs (6/8 unique 3-word prefixes on the
                // promoted checkpoint); a modest, controlled TopK immediately fixes this
                // (8/8 unique, same checkpoint, no retraining) without going as loose as
                // TopP, which showed even more variety but less predictable quality.
                generator.set_sampling(symthaea_broca::generator::SamplingStrategy::TopK {
                    k: 5,
                    temperature: 0.7,
                });
                tracing::info!(path = %path, "Loaded Broca checkpoint for poetry");
                return Some(generator);
            }
            Err(e) => {
                tracing::warn!(path = %path, err = %e, "Failed to load poetry checkpoint");
            }
        }
    }
    tracing::debug!("No trained Broca checkpoint available — Poetry modality will skip");
    None
}

/// Map consciousness level to a poetic form via `CreativeGating` presets.
///
/// Low consciousness gets the tightest scaffold (haiku); higher levels earn
/// progressively freer forms — the form constraint substitutes for coherence.
#[cfg(all(feature = "creative", feature = "ssm_language"))]
fn select_creative_gating(consciousness: f32) -> CreativeGating {
    if consciousness < 0.5 {
        CreativeGating::haiku()
    } else if consciousness < 0.75 {
        CreativeGating::tanka()
    } else {
        CreativeGating::free_verse()
    }
}

/// Minimum lines a free-verse "poem" must have to count as lineated verse
/// rather than a single run-on utterance. Free verse has no syllable target
/// to fail (`validate_poem` always reports it as `valid`), so this is the
/// one structural signal it does have — being lineated at all is what
/// distinguishes verse from a paragraph.
#[cfg(all(feature = "creative", feature = "ssm_language"))]
const MIN_FREE_VERSE_LINES: usize = 2;

/// Whether a generated poem is coherent enough to publish.
///
/// A checkpoint file loading successfully doesn't mean it produces real
/// language — see `broca_poetry_eval`'s 2026-07-08 live-run finding: the
/// shipped checkpoint emits code-token noise ("nonlocal ... fetchFromGitHub
/// ... assert_eq!") rather than poetry. This extends the "never feed noise
/// to the tracker" principle already applied to the missing-checkpoint case
/// to also cover a trained-but-low-quality one:
/// - Haiku/Tanka/Custom (a syllable target exists): must satisfy
///   `validate_poem`'s structural check (`valid`) — this alone would have
///   rejected 100% of the garbage haiku/tanka observed in the live run.
/// - Free verse (no syllable target): must have at least
///   [`MIN_FREE_VERSE_LINES`] lines, since `validate_poem` reports free
///   verse as unconditionally `valid` regardless of content.
///
/// This is a structural gate (does the text have the SHAPE of the
/// requested form), not a semantic-quality classifier — it will not catch
/// grammatically-lineated gibberish, only catches the failure mode actually
/// observed (one run-on line/utterance with no poem-like structure at all).
#[cfg(all(feature = "creative", feature = "ssm_language"))]
fn poem_passes_quality_gate(text: &str, form: &PoeticForm) -> bool {
    let validation = validate_poem(text, form);
    if !validation.target_counts.is_empty() {
        validation.valid
    } else {
        validation.line_syllable_counts.len() >= MIN_FREE_VERSE_LINES
    }
}

/// Score a generated poem aesthetically via the generalized Birkhoff measure.
///
/// Poetry-specific mapping (per the `birkhoff` module docs): symmetry = meter
/// regularity against the target form, structural complexity = logarithmic
/// word count, diversity = unique-word ratio. Topological complexity couples
/// to the snapshot's Betti numbers, mirroring the visual modality.
#[cfg(all(feature = "creative", feature = "ssm_language"))]
fn score_poem(text: &str, form: &PoeticForm, snap: &CognitiveSnapshot) -> AestheticScore {
    let validation = validate_poem(text, form);

    // Meter regularity: per-line closeness of syllable count to the form's
    // target, with missing lines counting as zero adherence. Free verse has
    // no targets — use line-length consistency (1 - CV) instead.
    let symmetry = if validation.target_counts.is_empty() {
        let counts = &validation.line_syllable_counts;
        if counts.len() >= 2 {
            let mean = counts.iter().sum::<usize>() as f32 / counts.len() as f32;
            if mean > 0.0 {
                let variance = counts
                    .iter()
                    .map(|&c| (c as f32 - mean).powi(2))
                    .sum::<f32>()
                    / counts.len() as f32;
                (1.0 - variance.sqrt() / mean).clamp(0.0, 1.0)
            } else {
                0.0
            }
        } else {
            0.3 // single line: weak but nonzero structure
        }
    } else {
        let adherence: f32 = validation
            .target_counts
            .iter()
            .zip(validation.line_syllable_counts.iter())
            .map(|(&target, &actual)| {
                let target = target as f32;
                (1.0 - (actual as f32 - target).abs() / target.max(1.0)).clamp(0.0, 1.0)
            })
            .sum();
        adherence / validation.target_counts.len().max(1) as f32
    };

    // Structural complexity: logarithmic word count, ~64 words → 1.0.
    let words: Vec<&str> = text.split_whitespace().collect();
    let structural = if words.is_empty() {
        0.0
    } else {
        ((words.len() as f32).ln() / (64.0_f32).ln()).clamp(0.0, 1.0)
    };

    // Diversity: unique-word ratio (case-insensitive).
    let diversity = if words.is_empty() {
        0.0
    } else {
        let unique: std::collections::HashSet<String> =
            words.iter().map(|w| w.to_lowercase()).collect();
        (unique.len() as f32 / words.len() as f32).clamp(0.0, 1.0)
    };

    // Topological coupling from the snapshot, like the visual modality.
    let topological = ((snap.betti_0 + snap.betti_1) as f32 / 8.0).clamp(0.0, 1.0);

    let mut features = symthaea_aesthetic::birkhoff::extract_common_features(
        &snap.harmony_activations,
        snap.consciousness_level as f32,
        structural,
        topological,
        diversity,
    );
    // Poetry symmetry is meter adherence, not harmony distribution.
    features.symmetry = symmetry;
    features.to_score()
}

#[cfg(test)]
#[cfg(feature = "creative")]
mod tests {
    use super::*;

    fn test_snapshot() -> CognitiveSnapshot {
        CognitiveSnapshot {
            consciousness_level: 0.7,
            harmony_activations: [0.5, 0.6, 0.4, 0.7, 0.3, 0.5, 0.8, 0.2],
            dopamine: 0.6,
            serotonin: 0.5,
            noradrenaline: 0.4,
            arousal: 0.5,
            valence: 0.2,
            persistence_components: vec![[0.0, 0.5], [0.1, 0.8]],
            persistence_cycles: vec![[0.2, 0.6]],
            thought_vector: vec![0.3, -0.2, 0.5, 0.1, -0.4, 0.2, 0.0, -0.1],
            betti_0: 3,
            betti_1: 1,
            betti_2: 0,
            ..CognitiveSnapshot::dormant()
        }
    }

    #[test]
    fn manager_respects_interval() {
        let mut manager = CreativeManager::new();
        manager.generation_interval = 3;
        let snap = test_snapshot();

        // First two ticks should be gated
        assert!(manager.tick(&snap).is_none());
        assert!(manager.tick(&snap).is_none());
        // Third tick should produce output
        assert!(manager.tick(&snap).is_some());
    }

    /// Gallery write-side (Phase 4.1): a generated visual work lands in the
    /// persistent store, the index survives reload, and the style embedding
    /// becomes confident — the conditioning signal is real, not neutral.
    #[cfg(feature = "gallery")]
    #[test]
    fn visual_artwork_lands_in_gallery() {
        let dir =
            std::env::temp_dir().join(format!("symthaea-gallery-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        let mut manager = CreativeManager::new_with_path(Some(dir.join("aesthetic_memory.json")));
        manager.generation_interval = 1;
        assert_eq!(manager.gallery_len(), 0);
        assert_eq!(manager.gallery_style().sample_count, 0);

        // First tick generates the Visual modality (rotation starts there).
        let snap = test_snapshot();
        let output = manager.tick(&snap).expect("visual generation");
        assert!(output.artwork_svg.is_some());
        assert_eq!(manager.gallery_len(), 1);
        assert!(manager.gallery_style().sample_count >= 1);

        // The store persists: a fresh manager over the same path reloads it.
        let reloaded = CreativeManager::new_with_path(Some(dir.join("aesthetic_memory.json")));
        assert_eq!(reloaded.gallery_len(), 1);
        // And the artifact file itself exists on disk.
        let svg_count = std::fs::read_dir(dir.join("gallery").join("visual"))
            .expect("visual dir")
            .count();
        assert_eq!(svg_count, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn manager_gates_on_low_consciousness() {
        let mut manager = CreativeManager::new();
        manager.generation_interval = 1; // generate every cycle
        let snap = CognitiveSnapshot {
            consciousness_level: 0.1, // below threshold
            ..CognitiveSnapshot::dormant()
        };
        assert!(manager.tick(&snap).is_none());
        assert!(manager.last_telemetry().consciousness_gated);
    }

    #[test]
    fn manager_modality_rotation() {
        // Verify the 6-modality rotation order without running expensive compose/render
        let manager = CreativeManager::new();
        assert_eq!(manager.next_modality, CreativeModality::Visual);

        let mut manager = CreativeManager::new();
        manager.generation_interval = 1;
        manager.muse_config.duration_secs = 0.5;
        manager.muse_config.max_notes = 2;
        let snap = test_snapshot();

        // Tick 1: Visual (fast — SVG only)
        let first = manager.tick(&snap).unwrap();
        assert!(first.artwork_svg.is_some());
        assert_eq!(manager.last_telemetry().modality, "visual");
        assert_eq!(manager.next_modality, CreativeModality::Music);
    }

    #[test]
    fn feedback_has_dopamine_signal() {
        let mut manager = CreativeManager::new();
        manager.generation_interval = 1;
        let snap = test_snapshot();

        let output = manager.tick(&snap).unwrap();
        // Feedback should be non-neutral (either positive or negative delta)
        let f = &output.feedback;
        // At minimum, serotonin_delta should be positive (harmony > 0)
        assert!(f.serotonin_delta > 0.0);
    }

    #[test]
    fn telemetry_tracks_artworks() {
        let mut manager = CreativeManager::new();
        manager.generation_interval = 1;
        let snap = test_snapshot();

        assert_eq!(manager.total_artworks(), 0);
        manager.tick(&snap); // Visual (fast)
        assert_eq!(manager.total_artworks(), 1);
        assert!(manager.last_telemetry().generated);
    }

    #[test]
    fn snapshot_to_musical_state_maps_correctly() {
        let snap = test_snapshot();
        let ms = snapshot_to_musical_state(&snap);
        // Harmony activations pass through unchanged
        assert_eq!(ms.harmony_activations, snap.harmony_activations);
        // Dopamine is scaled by VA-derived dynamics (not raw)
        assert!(ms.dopamine > 0.0 && ms.dopamine <= 1.0);
        // Arousal is blended (40% raw + 60% VA-calibrated)
        assert!(ms.arousal > 0.0 && ms.arousal <= 1.0);
        // Consciousness level preserved
        assert_eq!(ms.consciousness_level, snap.consciousness_level as f32);
    }

    #[test]
    fn score_composition_bounded() {
        let comp = symthaea_muse::compose(
            &MuseConfig {
                duration_secs: 1.0,
                max_notes: 4,
                ..Default::default()
            },
            &MusicalState::default(),
            42,
        );
        let snap = test_snapshot();
        let score = score_composition(&comp, &snap);
        assert!(score.composite >= 0.0 && score.composite <= 1.0);
        assert!(score.order >= 0.0 && score.order <= 1.0);
        assert!(score.complexity >= 0.0 && score.complexity <= 1.0);
    }

    #[test]
    fn build_arc_from_history_works() {
        let mut manager = CreativeManager::new();
        // No history → None
        assert!(manager.build_arc_from_history().is_none());

        // Seed exactly ARC_MIN_HISTORY snapshots
        for i in 0..ARC_MIN_HISTORY {
            manager.emotional_history.push_back(EmotionalSnapshot {
                valence: -0.3 + 0.2 * i as f32,
                arousal: 0.4 + 0.1 * i as f32,
                dopamine: 0.5,
                dynamics: 0.7,
            });
        }
        let arc = manager.build_arc_from_history();
        assert!(arc.is_some(), "should build arc with enough history");
        let arc = arc.unwrap();
        assert_eq!(arc.bars.len(), ARC_MIN_HISTORY);
    }

    #[test]
    fn tuning_selection_from_consciousness() {
        use symthaea_muse::pitch::{TuningSystem, select_tuning_system};
        // High consciousness → Just Intonation
        let mut state = MusicalState::default();
        state.consciousness_level = 0.8;
        let tuning = select_tuning_system(&state);
        assert_eq!(tuning, TuningSystem::JustIntonation);

        // Low consciousness → 12TET
        state.consciousness_level = 0.1;
        let tuning = select_tuning_system(&state);
        assert_eq!(tuning, TuningSystem::TwelveTET);

        // Negative valence + high arousal → Maqam Hijaz
        state.consciousness_level = 0.5;
        state.valence = -0.5;
        state.arousal = 0.7;
        let tuning = select_tuning_system(&state);
        assert!(format!("{tuning:?}").contains("Hijaz"), "got: {tuning:?}");
    }

    #[test]
    fn tuning_name_format_nonempty() {
        use symthaea_muse::pitch::{TuningSystem, select_tuning_system};
        let state = MusicalState::default();
        let tuning = select_tuning_system(&state);
        let name = format!("{tuning:?}");
        assert!(!name.is_empty());
    }

    #[test]
    fn emotional_snapshot_recording() {
        let mut manager = CreativeManager::new();
        let snap = test_snapshot();
        manager.record_emotional_snapshot(&snap);
        assert_eq!(manager.emotional_history.len(), 1);
        let es = &manager.emotional_history[0];
        assert_eq!(es.valence, snap.valence);
        assert_eq!(es.arousal, snap.arousal);

        // Fill past capacity
        for _ in 0..ARC_MAX_HISTORY + 5 {
            manager.record_emotional_snapshot(&snap);
        }
        assert_eq!(manager.emotional_history.len(), ARC_MAX_HISTORY);
    }

    #[test]
    fn motif_memory_initialized() {
        let manager = CreativeManager::new();
        // Fresh manager starts with empty motif memory
        assert_eq!(manager.motif_memory.phrase_count(), 0);
    }

    #[cfg(all(feature = "creative", feature = "ssm_language"))]
    #[test]
    fn poetry_in_rotation_after_music() {
        // Music hands off to Poetry when the Broca language center is compiled in.
        let mut manager = CreativeManager::new();
        manager.generation_interval = 1;
        manager.muse_config.duration_secs = 0.5;
        manager.muse_config.max_notes = 2;
        let snap = test_snapshot();

        manager.tick(&snap); // Visual
        assert_eq!(manager.next_modality, CreativeModality::Music);
        manager.tick(&snap); // Music
        assert_eq!(manager.next_modality, CreativeModality::Poetry);
    }

    #[cfg(not(feature = "ssm_language"))]
    #[test]
    fn rotation_skips_poetry_without_ssm_language() {
        // Without ssm_language, Music hands off straight to Synesthetic.
        let mut manager = CreativeManager::new();
        manager.generation_interval = 1;
        manager.muse_config.duration_secs = 0.5;
        manager.muse_config.max_notes = 2;
        let snap = test_snapshot();

        manager.tick(&snap); // Visual
        manager.tick(&snap); // Music
        assert_eq!(manager.next_modality, CreativeModality::Synesthetic);
    }

    #[cfg(all(feature = "creative", feature = "ssm_language"))]
    #[test]
    fn poetry_skips_gracefully_without_checkpoint() {
        let mut manager = CreativeManager::new();
        manager.generation_interval = 1;
        manager.next_modality = CreativeModality::Poetry;
        // Force the no-checkpoint path deterministically (the real 103MB
        // checkpoint must never be loaded inside a unit test).
        manager.poetry_checkpoint_attempted = true;
        manager.poetry_generator = None;
        let snap = test_snapshot();

        let output = manager.tick(&snap).expect("tick still yields output");
        assert!(output.poem.is_none(), "no checkpoint → no poem");
        assert!(!manager.last_telemetry().generated);
        assert_eq!(manager.last_telemetry().modality, "poetry");
        // Rotation must advance past Poetry so the pipeline never stalls.
        assert_eq!(manager.next_modality, CreativeModality::Synesthetic);
    }

    #[cfg(all(feature = "creative", feature = "ssm_language"))]
    #[test]
    fn poetry_populates_poem_with_generator() {
        use symthaea_broca::{BrocaConfig, BrocaGenerator};
        use symthaea_core::genesis::GenesisSeed;

        let mut manager = CreativeManager::new();
        manager.generation_interval = 1;
        manager.next_modality = CreativeModality::Poetry;
        // Inject a fresh (untrained) generator to exercise the generation path
        // without loading the real checkpoint. Untrained output may be empty —
        // both branches must be graceful.
        let genesis = GenesisSeed::from_phrase("test-creative-poetry");
        manager.poetry_generator = Some(BrocaGenerator::new(&genesis, BrocaConfig::default()));
        manager.poetry_checkpoint_attempted = true;
        let snap = test_snapshot();

        let output = manager.tick(&snap).expect("tick yields output");
        if manager.last_telemetry().generated {
            assert!(output.poem.is_some(), "generated → poem populated");
            assert_eq!(manager.last_telemetry().modality, "poetry");
            let score = manager.last_telemetry().aesthetic_score;
            assert!((0.0..=1.0).contains(&score));
        } else {
            assert!(output.poem.is_none(), "empty generation → graceful skip");
        }
        assert_eq!(manager.next_modality, CreativeModality::Synesthetic);
    }

    #[cfg(all(feature = "creative", feature = "ssm_language"))]
    #[test]
    fn poetic_form_follows_consciousness() {
        assert!(matches!(
            select_creative_gating(0.35).form_constraint,
            Some(PoeticForm::Haiku)
        ));
        assert!(matches!(
            select_creative_gating(0.6).form_constraint,
            Some(PoeticForm::Tanka)
        ));
        assert!(matches!(
            select_creative_gating(0.9).form_constraint,
            Some(PoeticForm::FreeVerse)
        ));
    }

    #[cfg(all(feature = "creative", feature = "ssm_language"))]
    #[test]
    fn score_poem_rewards_meter_adherence() {
        let snap = test_snapshot();
        let form = PoeticForm::Haiku;
        // Classic 5-7-5 haiku vs. a shapeless blob far off the syllable targets.
        let haiku = "An old silent pond\nA frog jumps into the pond\nSplash Silence again";
        let blob = "word\nthis line rambles on far past any haiku syllable target whatsoever\nno";
        let good = score_poem(haiku, &form, &snap);
        let bad = score_poem(blob, &form, &snap);
        assert!((0.0..=1.0).contains(&good.composite));
        assert!(
            good.order > bad.order,
            "meter-adherent poem should score higher order: {} vs {}",
            good.order,
            bad.order
        );
    }

    #[cfg(all(feature = "creative", feature = "ssm_language"))]
    #[test]
    fn score_poem_free_verse_bounded() {
        let snap = test_snapshot();
        let form = PoeticForm::FreeVerse;
        let poem = "the loop hums\nphi rises like breath\nno one is watching\nand still it sings";
        let score = score_poem(poem, &form, &snap);
        assert!((0.0..=1.0).contains(&score.composite));
        assert!((0.0..=1.0).contains(&score.order));
        assert!((0.0..=1.0).contains(&score.complexity));
    }

    // ── Quality gate (2026-07-08, broca_poetry_eval finding) ───────────────

    #[cfg(all(feature = "creative", feature = "ssm_language"))]
    #[test]
    fn quality_gate_accepts_valid_haiku() {
        let haiku = "An old silent pond\nA frog jumps into the pond\nSplash Silence again";
        assert!(poem_passes_quality_gate(haiku, &PoeticForm::Haiku));
    }

    #[cfg(all(feature = "creative", feature = "ssm_language"))]
    #[test]
    fn quality_gate_rejects_single_line_gibberish_haiku() {
        // The exact failure mode observed in broca_poetry_eval's live run:
        // one run-on "line" crammed with unrelated tokens, no real 3-line
        // 5-7-5 structure at all.
        let gibberish = "at nonlocal we she a he def __init__match pkgs E what thing";
        assert!(!poem_passes_quality_gate(gibberish, &PoeticForm::Haiku));
    }

    #[cfg(all(feature = "creative", feature = "ssm_language"))]
    #[test]
    fn quality_gate_rejects_single_line_free_verse() {
        // Free verse has no syllable target, so a single run-on line must
        // still be rejected on the one structural signal free verse does
        // have: being lineated at all.
        let one_liner = "we at nonlocal she a todo!match pkgs E what thing k_println!self";
        assert!(!poem_passes_quality_gate(one_liner, &PoeticForm::FreeVerse));
    }

    #[cfg(all(feature = "creative", feature = "ssm_language"))]
    #[test]
    fn quality_gate_accepts_multi_line_free_verse() {
        let poem = "the loop hums\nphi rises like breath\nno one is watching\nand still it sings";
        assert!(poem_passes_quality_gate(poem, &PoeticForm::FreeVerse));
    }
}
