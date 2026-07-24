// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! App-wide shared state: the current piece and its playback, provided at
//! the `App` root via `provide_context` and read by every mode.
//!
//! All three design specs list this as separate shared stores
//! (`PlaybackStore` / `CurrentPieceStore`) with the same note attached
//! every time: *"Playback state remains independent from
//! Listen/Research/Studio Mode state."* This is that store — one shared
//! `MuseState` rather than two, since in practice every field here
//! changes together (a new piece means a new playback target).
//!
//! The `<audio>` element itself lives in `App`, outside `<Routes>`, so
//! navigating between Listen/Create/Research does not unmount it —
//! satisfying Research Mode's acceptance criterion #1 ("Playback persists
//! across Research routes and mode switches") and Listen Mode's P0 item
//! "global persistent playback".
//!
//! Playback itself is NOT a grab bag of ad hoc `is_playing`/`current_time`
//! signals here — it's `playback::PlaybackState`, a pure, browser-
//! independent reducer with load-epoch discipline against stale/late
//! browser events (see `playback.rs`'s module doc and its own test suite).
//! This module is the adapter: `dispatch()` reduces an event and executes
//! whatever `PlaybackEffect`s come back against the real `<audio>`
//! element; `app.rs`'s element event handlers dispatch the browser's own
//! events back in. Views read `muse.playback.get()` reactively.

use std::collections::HashMap;

use leptos::prelude::*;
use leptos::task::spawn_local;
use wasm_bindgen_futures::JsFuture;
use web_sys::HtmlAudioElement;

use crate::api::{self, Candidate};
use crate::audio_reactivity;
use crate::journey::{JourneyArtifact, JourneyCommand, JourneyEffect, JourneyPolicy, JourneyState};
use crate::palette;
use crate::playback::{
    PlaybackEffect, PlaybackEvent, PlaybackPhase, PlaybackSource, PlaybackState,
};

/// The Listen mode: how the *hero* visualizer (the big piece-map canvas)
/// renders. Shared with the player bar's mini preview — one choice,
/// reflected everywhere. Deliberately does NOT include `Bars`/`Waves`
/// anymore — those are timeline concepts (see [`TimelineMode`]), not
/// whole-piece map experiences; a hero-mode toggle sitting in the player
/// bar next to the timeline used to read as if it controlled the
/// timeline's own display, when it actually always controlled this.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum VizMode {
    /// The whole-piece structural map: section arcs, phrase ticks,
    /// cadence/motif markers, audio-reactive spokes and glow.
    Radial,
    /// A frozen, non-animated scene — for anyone who finds continuous
    /// motion distracting rather than immersive. Draws exactly once.
    Still,
}

impl VizMode {
    pub const ALL: [Self; 2] = [Self::Radial, Self::Still];

    pub fn label(self) -> &'static str {
        match self {
            VizMode::Radial => "Journey",
            VizMode::Still => "Still",
        }
    }
}

/// How the player bar's own timeline/scrub strip renders — independent of
/// [`VizMode`]. `Bar` and `Wave` are both genuinely audio-reactive (real
/// samples, not decoration); neither is a "map" of the whole piece, which
/// is what distinguishes this from the hero's `VizMode`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TimelineMode {
    /// A compact amplitude-envelope bar strip.
    Bar,
    /// The continuous waveform line (this app's original/default timeline
    /// look).
    Wave,
}

impl TimelineMode {
    pub const ALL: [Self; 2] = [Self::Bar, Self::Wave];

    pub fn label(self) -> &'static str {
        match self {
            TimelineMode::Bar => "Bar",
            TimelineMode::Wave => "Wave",
        }
    }
}

/// A per-page-load identifier for `JourneyState::journey_id` — only needs
/// to be unique enough within one session's own `composition_request_id`
/// strings, not globally unique across users/devices.
fn journey_id() -> String {
    format!("j{:x}", (js_sys::Math::random() * u32::MAX as f64) as u32)
}

/// The journey's traversal seed — deterministic reproduction of one
/// journey's compose sequence (given the same policy/history) isn't relied
/// on anywhere yet, so a fresh random seed per page load is fine.
fn traversal_seed() -> u64 {
    (js_sys::Math::random() * 1_000_000_000.0) as u64
}

/// `JourneyArtifact::relation_from_previous` — presentation text, not an
/// asserted fact (matches this app's existing "why" text conventions).
/// Client-side heuristic per MUSE_JOURNEY_WIRING_PLAN_2026-07-24.md §2
/// step 2, rather than a new backend endpoint.
fn relation_from_previous(previous_style: Option<&str>, new_style: &str) -> String {
    match previous_style {
        None => "the opening piece".to_string(),
        Some(style) if style == new_style => "same mood, new voice".to_string(),
        Some(_) => format!("a shift into {new_style}"),
    }
}

/// The real grammar family `GET /api/styles` reported for `style`, or
/// `None` if `families` hasn't loaded yet (or doesn't recognize the name --
/// shouldn't happen once loaded, since the server reports every style it
/// can compose).
fn family_of<'a>(
    families: &'a [symthaea_muse_protocol::StyleFamily],
    style: &str,
) -> Option<&'a str> {
    families
        .iter()
        .find(|entry| entry.name == style)
        .map(|entry| entry.family.as_str())
}

/// Pick the next compose style for `policy`. Degrades to a plain uniform
/// random style (the pre-policy-wiring behavior) whenever the real
/// family data isn't available yet, or a policy's own preferred set comes
/// up empty (e.g. every style was recently played) -- never returns
/// nothing.
fn pick_style_for_policy(
    policy: JourneyPolicy,
    previous_style: Option<&str>,
    recent_styles: &[String],
    families: &[symthaea_muse_protocol::StyleFamily],
) -> String {
    if families.is_empty() {
        return palette::random_style().to_string();
    }
    let pick_from = |candidates: Vec<&str>| -> Option<String> {
        if candidates.is_empty() {
            return None;
        }
        let idx = (js_sys::Math::random() * candidates.len() as f64) as usize;
        candidates.get(idx).map(|s| s.to_string())
    };
    let result = match policy {
        JourneyPolicy::Resonance => previous_style
            .and_then(|prev| family_of(families, prev))
            .and_then(|family| {
                pick_from(
                    families
                        .iter()
                        .filter(|entry| entry.family == family)
                        .map(|entry| entry.name.as_str())
                        .collect(),
                )
            }),
        JourneyPolicy::Contrast => previous_style
            .and_then(|prev| family_of(families, prev))
            .and_then(|family| {
                pick_from(
                    families
                        .iter()
                        .filter(|entry| entry.family != family)
                        .map(|entry| entry.name.as_str())
                        .collect(),
                )
            }),
        JourneyPolicy::Discovery => pick_from(
            families
                .iter()
                .map(|entry| entry.name.as_str())
                .filter(|name| !recent_styles.iter().any(|recent| recent == name))
                .collect(),
        ),
    };
    result.unwrap_or_else(|| palette::random_style().to_string())
}

#[derive(Clone, Copy)]
pub struct MuseState {
    pub current: RwSignal<Option<Candidate>>,
    pub current_style: RwSignal<String>,
    /// The active Listen visualization mode — shared so the hero canvas
    /// and the player bar's mini preview always agree.
    pub viz_mode: RwSignal<VizMode>,
    /// The player bar's own timeline display — independent of `viz_mode`
    /// (see [`TimelineMode`]'s doc comment for why these are two separate
    /// choices, not one).
    pub timeline_mode: RwSignal<TimelineMode>,
    /// The deterministic Listen-journey reducer (`journey.rs`) — replaces
    /// the previous ad hoc `queue`/`request_generation` pair with a single
    /// tested state machine (`previous`/`current`/`next`/`pending`, epoch-
    /// discriminated staleness). It only tracks lightweight identity
    /// (`JourneyArtifact`), not playable audio data — see `candidate_cache`.
    journey: RwSignal<JourneyState>,
    /// Full `Candidate` data for every `JourneyArtifact` `journey` currently
    /// references (by `candidate_id`) -- `journey.rs`'s reducer is
    /// deliberately audio-agnostic, so this is where the actual audio
    /// URL/duration/style/title data a `JourneyArtifact` names actually
    /// lives. Never pruned against `journey.previous`/`next` truncation --
    /// `Candidate` carries no audio bytes (just metadata + a URL), so
    /// letting it grow for a session's lifetime is cheap.
    candidate_cache: RwSignal<HashMap<u64, Candidate>>,
    /// Whether the *next* `CurrentChanged` journey effect (however it
    /// arrives -- synchronously from `Advance`, or asynchronously once an
    /// in-flight compose resolves via `PrefetchCompleted`) should autoplay.
    /// `journey.rs`'s effects don't carry this bit themselves (it's a
    /// player concern, not a journey-identity concern), so it's tracked
    /// here, set immediately before every journey dispatch that might
    /// produce a `CurrentChanged`.
    pending_autoplay: RwSignal<bool>,
    /// Real server-computed style -> grammar-family map (`GET
    /// /api/styles`), fetched once at app start and cached -- lets
    /// `JourneyPolicy::Resonance`/`Contrast` pick a same-/different-family
    /// style without duplicating `Style::grammar_family()`'s mapping
    /// client-side (see MUSE_JOURNEY_WIRING_PLAN_2026-07-24.md). Empty
    /// until the fetch resolves; policy selection degrades to a plain
    /// random style in the meantime rather than blocking on it.
    style_families: RwSignal<Vec<symthaea_muse_protocol::StyleFamily>>,
    /// The last few styles played, most-recent-last, capped at
    /// `RECENT_STYLES_CAP` -- `JourneyPolicy::Discovery`'s avoid-repeats
    /// signal. Deliberately separate from `journey.rs`'s own
    /// `recent_compositions` (composition-hash exact-repeat detection, not
    /// style variety).
    recent_styles: RwSignal<Vec<String>>,
    pub status: RwSignal<String>,
    pub composing: RwSignal<bool>,
    pub kept: RwSignal<bool>,
    pub audio_ref: NodeRef<leptos::html::Audio>,
    /// The playback reducer's state — phase, position, duration, load
    /// epoch. See this module's doc comment and `playback.rs`.
    pub playback: RwSignal<PlaybackState>,
    /// 0.0-1.0, bound bidirectionally to `audio.volume` — read on mount and
    /// written on every slider change (see `player_bar.rs`). Not part of
    /// `PlaybackState` — volume is a user preference independent of what
    /// phase playback is in.
    pub volume: RwSignal<f64>,
    /// User's persistent render-backend preference: `Some("native")`/
    /// `Some("fluidsynth")` to force one, `None` for the server's own
    /// default (FluidSynth when the environment provides it). Sent on
    /// every compose (`ComposeRequest::renderer`); the server always
    /// reports which one actually rendered in `Candidate::renderer`
    /// regardless, so a forced-but-unavailable FluidSynth preference
    /// degrades visibly rather than silently.
    pub renderer_preference: RwSignal<Option<&'static str>>,
}

impl MuseState {
    pub fn new() -> Self {
        Self {
            current: RwSignal::new(None),
            current_style: RwSignal::new("Classical".to_string()),
            viz_mode: RwSignal::new(VizMode::Radial),
            timeline_mode: RwSignal::new(TimelineMode::Wave),
            journey: RwSignal::new(JourneyState::new(
                journey_id(),
                JourneyPolicy::Resonance,
                traversal_seed(),
            )),
            candidate_cache: RwSignal::new(HashMap::new()),
            pending_autoplay: RwSignal::new(false),
            style_families: RwSignal::new(Vec::new()),
            recent_styles: RwSignal::new(Vec::new()),
            status: RwSignal::new(String::new()),
            composing: RwSignal::new(false),
            kept: RwSignal::new(false),
            audio_ref: NodeRef::new(),
            playback: RwSignal::new(PlaybackState::default()),
            volume: RwSignal::new(1.0),
            renderer_preference: RwSignal::new(None),
        }
    }

    /// Reduce a playback event, then execute whatever effects come back
    /// against the real `<audio>` element. The ONE place browser side
    /// effects happen — the Listen hero canvas, the persistent player
    /// bar, and `app.rs`'s own element event handlers all funnel through
    /// this instead of touching `HtmlAudioElement` directly, so they can
    /// never disagree with the reducer about what state playback is in.
    pub fn dispatch(self, event: PlaybackEvent) {
        let effects = self
            .playback
            .try_update(|s| s.reduce(event))
            .unwrap_or_default();
        for effect in effects {
            self.execute(effect);
        }
    }

    fn execute(self, effect: PlaybackEffect) {
        let Some(audio) = self.audio_ref.get_untracked() else {
            return;
        };
        let audio: HtmlAudioElement = audio.into();
        match effect {
            PlaybackEffect::Load { audio_url, .. } => {
                audio.set_src(&audio_url);
                audio.set_volume(self.volume.get_untracked());
            }
            PlaybackEffect::Play { load_epoch } => {
                // Ensure the Web Audio analysis tap exists before playing
                // — this only ever runs in response to a real event chain
                // starting from a user gesture (a click, or a
                // `MetadataLoaded` reaction to a load that same gesture
                // started), which is what browsers require before an
                // `AudioContext` will actually produce sound.
                if audio_reactivity::ensure_connected(&audio).is_err() {
                    leptos::logging::warn!("audio_reactivity::ensure_connected failed");
                }
                let Ok(promise) = audio.play() else {
                    self.dispatch(PlaybackEvent::AutoplayRejected { load_epoch });
                    return;
                };
                spawn_local(async move {
                    // `play()`'s returned Promise rejecting is the only
                    // reliable cross-browser signal that autoplay was
                    // blocked — a `pause` event does not reliably follow
                    // a blocked autoplay attempt in every browser.
                    if JsFuture::from(promise).await.is_err() {
                        self.dispatch(PlaybackEvent::AutoplayRejected { load_epoch });
                    }
                });
            }
            PlaybackEffect::Pause { .. } => {
                let _ = audio.pause();
            }
            PlaybackEffect::Seek { seconds, .. } => {
                audio.set_current_time(seconds);
            }
            PlaybackEffect::AdvanceJourney => {
                self.next_piece(true);
            }
        }
    }

    /// Toggle play/pause — dispatches through the reducer rather than
    /// checking `audio.paused()` directly, so this and every other caller
    /// agree on what "playing" means.
    pub fn toggle_play(self) {
        match self.playback.get_untracked().phase {
            PlaybackPhase::Playing => self.dispatch(PlaybackEvent::PauseRequested),
            _ => self.dispatch(PlaybackEvent::PlayRequested),
        }
    }

    /// Restart the current piece from 0:00 — the honest "previous" action
    /// given there's no navigable play history (see `icons::RestartIcon`).
    pub fn restart(self) {
        self.dispatch(PlaybackEvent::SeekRequested { seconds: 0.0 });
        self.dispatch(PlaybackEvent::PlayRequested);
    }

    /// Seek to an absolute position — the player bar's range input calls
    /// this on every drag/keyboard change.
    pub fn seek(self, seconds: f64) {
        self.dispatch(PlaybackEvent::SeekRequested { seconds });
    }

    fn show_piece(self, c: Candidate, autoplay: bool) {
        self.kept.set(false);
        self.current_style.set(c.style.clone());
        let source = PlaybackSource {
            // Real content-hash identity now that /api/compose populates
            // it (see MUSE_JOURNEY_WIRING_PLAN_2026-07-24.md step 1) --
            // previously a synthetic `c.id.to_string()`.
            rendition_id: c
                .identity
                .as_ref()
                .map(|identity| identity.rendition.clone()),
            audio_url: api::audio_url(api::DEFAULT_BACKEND, c.id),
            // A provisional hint so the seek bar isn't zero-max before
            // real `<audio>` metadata arrives — the reducer overwrites
            // this with the real duration on `MetadataLoaded`, per its
            // own doc comment on why: the actual rendered file can differ
            // slightly from the composed estimate.
            duration_hint_seconds: Some(c.duration_secs.max(0.0) as f64),
            advance_on_end: true,
        };
        self.current.set(Some(c));
        // `LoadRequested` bumps the load epoch and returns a `Load`
        // effect; if `autoplay` is set, the reducer's own `MetadataLoaded`
        // handling (once `app.rs` dispatches it in response to the real
        // `loadedmetadata` event) returns the `Play` effect — this
        // function doesn't need to (and must not) trigger play itself.
        self.dispatch(PlaybackEvent::LoadRequested { source, autoplay });
    }

    /// Show whatever `journey.current` now names, using the cached
    /// `Candidate` its `candidate_id` points at. A cache miss shouldn't
    /// happen in practice (every artifact is built from a freshly-cached
    /// candidate right before being handed to the reducer) but degrades to
    /// a status message rather than panicking if it ever does.
    fn show_current_from_journey(self, autoplay: bool) {
        let Some(artifact) = self.journey.get_untracked().current else {
            return;
        };
        let Some(candidate) = self
            .candidate_cache
            .get_untracked()
            .get(&artifact.candidate_id)
            .cloned()
        else {
            self.status
                .set("lost track of the next piece — try Next again".to_string());
            return;
        };
        self.status.set(String::new());
        self.show_piece(candidate, autoplay);
    }

    /// Apply every effect a journey dispatch produced. `CurrentChanged`
    /// shows whatever `journey.current` now is (autoplay per
    /// `pending_autoplay`, set by the caller before dispatching);
    /// `ComposeNext` starts the async compose that will eventually feed a
    /// `PrefetchCompleted`/`CompositionFailed` back into the reducer,
    /// recursing into this same function for whatever effects THAT
    /// produces.
    fn apply_journey_effects(self, effects: Vec<JourneyEffect>) {
        for effect in effects {
            match effect {
                JourneyEffect::CurrentChanged => {
                    self.show_current_from_journey(self.pending_autoplay.get_untracked());
                }
                JourneyEffect::ComposeNext { request, .. } => {
                    self.status.set("composing the next piece…".to_string());
                    let previous_style = self
                        .journey
                        .get_untracked()
                        .current
                        .map(|artifact| artifact.style);
                    let style = pick_style_for_policy(
                        self.journey.get_untracked().policy,
                        previous_style.as_deref(),
                        &self.recent_styles.get_untracked(),
                        &self.style_families.get_untracked(),
                    );
                    let renderer = self.renderer_preference.get_untracked();
                    spawn_local(async move {
                        let reduced =
                            match api::compose_listen_piece(api::DEFAULT_BACKEND, &style, renderer)
                                .await
                            {
                                Ok(c) => {
                                    let selected = JourneyArtifact {
                                        identity: c.identity.clone().unwrap_or_default(),
                                        candidate_id: c.id,
                                        title: c.title.clone(),
                                        style: c.style.clone(),
                                        relation_from_previous: relation_from_previous(
                                            previous_style.as_deref(),
                                            &c.style,
                                        ),
                                    };
                                    self.recent_styles.update(|recent| {
                                        recent.push(c.style.clone());
                                        const RECENT_STYLES_CAP: usize = 5;
                                        if recent.len() > RECENT_STYLES_CAP {
                                            let excess = recent.len() - RECENT_STYLES_CAP;
                                            recent.drain(..excess);
                                        }
                                    });
                                    self.candidate_cache.update(|cache| {
                                        cache.insert(c.id, c);
                                    });
                                    self.status.set(String::new());
                                    self.journey.try_update(|j| {
                                        j.reduce(JourneyCommand::PrefetchCompleted {
                                            composition_request_id: request
                                                .composition_request_id
                                                .clone(),
                                            prefetch_epoch: request.prefetch_epoch,
                                            selected,
                                        })
                                    })
                                }
                                Err(e) => {
                                    self.status
                                        .set(format!("couldn't reach the composer — {e}"));
                                    self.journey.try_update(|j| {
                                        j.reduce(JourneyCommand::CompositionFailed {
                                            composition_request_id: request
                                                .composition_request_id
                                                .clone(),
                                            prefetch_epoch: request.prefetch_epoch,
                                            message: e,
                                        })
                                    })
                                }
                            };
                        self.apply_journey_effects(reduced.unwrap_or_default());
                    });
                }
            }
        }
    }

    /// Advance to a new piece: prefer whatever the journey already
    /// prefetched, else compose fresh. `autoplay` starts playback
    /// immediately (used by "Next Piece" and auto-advance-on-end); the
    /// very first piece on load stays paused until the user acts.
    pub fn next_piece(self, autoplay: bool) {
        self.pending_autoplay.set(autoplay);
        let effects = self
            .journey
            .try_update(|j| j.reduce(JourneyCommand::Advance))
            .unwrap_or_default();
        self.apply_journey_effects(effects);
        // `Advance` only replenishes the prefetch slot when it actually
        // consumed a `next` -- the very first piece (cold start, nothing
        // prefetched yet) leaves nothing scheduled for the SECOND piece.
        // `RequestNext` is a safe no-op if something's already
        // pending/queued, so dispatching it unconditionally here restores
        // the old behavior of always trying to stay one piece ahead.
        let effects = self
            .journey
            .try_update(|j| j.reduce(JourneyCommand::RequestNext))
            .unwrap_or_default();
        self.apply_journey_effects(effects);
    }

    /// Change which policy future prefetches use. UI-only for now — all
    /// three `JourneyPolicy` variants compose identically until a real
    /// selection heuristic is wired to them (open question, see
    /// MUSE_JOURNEY_WIRING_PLAN_2026-07-24.md §3).
    pub fn set_journey_policy(self, policy: JourneyPolicy) {
        let effects = self
            .journey
            .try_update(|j| j.reduce(JourneyCommand::ChangePolicy(policy)))
            .unwrap_or_default();
        self.apply_journey_effects(effects);
    }

    pub fn journey_policy(self) -> JourneyPolicy {
        self.journey.get().policy
    }

    /// Fetch `GET /api/styles` once and cache it (see `style_families`'s
    /// doc comment). Called once from `App`'s root mount. Failure just
    /// leaves `style_families` empty -- `pick_style_for_policy` already
    /// degrades to plain random selection in that case, so there's
    /// nothing else to do here but log it.
    pub fn load_style_families(self) {
        spawn_local(async move {
            match api::fetch_style_families(api::DEFAULT_BACKEND).await {
                Ok(families) => self.style_families.set(families),
                Err(e) => leptos::logging::warn!("fetch_style_families failed: {e}"),
            }
        });
    }

    pub fn keep(self) {
        let Some(c) = self.current.get_untracked() else {
            return;
        };
        spawn_local(async move {
            if api::keep_piece(api::DEFAULT_BACKEND, c.id).await.is_ok() {
                self.kept.set(true);
            }
        });
    }
}

impl Default for MuseState {
    fn default() -> Self {
        Self::new()
    }
}
