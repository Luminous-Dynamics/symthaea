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
        }
    }
}

/// Default path for cross-session aesthetic memory.
#[cfg(feature = "creative")]
const AESTHETIC_MEMORY_PATH: &str = ".claude/aesthetic_memory.json";

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
}

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

        let start = std::time::Instant::now();
        self.seed_counter += 1;

        let mut output = CreativeOutput::default();
        let modality_name;

        match self.next_modality {
            CreativeModality::Visual => {
                let artwork = symthaea_atelier::create_artwork_iterative(
                    &self.atelier_config,
                    snap,
                    self.seed_counter,
                );
                let score = artwork.aesthetic_score;
                let feedback = self.tracker.process(&score, &snap.harmony_activations);
                output.artwork_svg = Some(artwork.svg);
                output.feedback = feedback;
                modality_name = "visual";

                self.record_telemetry(
                    &score,
                    &feedback,
                    modality_name,
                    artwork.generation_cycles,
                    start.elapsed(),
                );

                self.next_modality = CreativeModality::Music;
            }
            CreativeModality::Music => {
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
                    let mut replay_notes = Vec::new();
                    while let Some(note) = self.motif_memory.next_note(&musical_state) {
                        replay_notes.push(note);
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
        };
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
    unique_pitches.sort_by(|a, b| a.partial_cmp(b).unwrap());
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
        use symthaea_muse::pitch::{select_tuning_system, TuningSystem};
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
        use symthaea_muse::pitch::{select_tuning_system, TuningSystem};
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
}
