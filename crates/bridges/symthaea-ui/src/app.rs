// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Root application: Converse (chat with the facade) + Observe (live
//! telemetry from the experience bridge, when enabled on the daemon).
//!
//! Design note (standing project rule): this is meant to read as a living
//! organism's vitals, not an admin dashboard — the vital orb's pulse rate
//! and hue are driven directly by consciousness_level/valence/thermodynamic
//! load rather than laid out as a grid of numeric gauges. v0 keeps this
//! modest (one orb + a few readouts); it should grow toward that goal
//! rather than toward more grid cells.

use base64::Engine as _;
use leptos::prelude::*;
use leptos::task::spawn_local_scoped_with_cancellation;
use serde_json::Value;
use std::cell::Cell;
use std::rc::Rc;
use wasm_bindgen::JsCast;

use crate::api::{self};
use crate::presentation::{
    cognitive_spans, event_between, next_event_sequence, next_telemetry_session,
    push_cognitive_event, telemetry_session_is_current, CognitiveEvent, CognitiveEventKind,
    CognitiveState,
};

const DEFAULT_GATEWAY: &str = "http://127.0.0.1:8090";

/// Allocate an authority generation without ever reusing an identity.
///
/// Request/status generations gate asynchronous responses. Saturating at
/// u64::MAX would make the terminal generation repeat forever, which could
/// allow an arbitrarily old response to become authoritative again. Exhaustion
/// is therefore a deliberate fail-closed state.
fn next_authority_generation(current: &mut u64) -> Option<u64> {
    let next = current.checked_add(1)?;
    *current = next;
    Some(next)
}

#[derive(Clone, Debug)]
struct Turn {
    role: &'static str,
    text: String,
}

/// Latest telemetry snapshot extracted from a `CycleMetadata` payload.
/// Fields default to a neutral resting state until the first message
/// arrives — see `Vitals::from_json` for the exact wire paths, which
/// mirror `src/cognitive_loop/types/telemetry.rs`.
#[derive(Clone, Copy, Debug, Default)]
struct Vitals {
    consciousness_level: f64,
    valence: f32,
    arousal: f32,
    mood_temperature: f32,
    thermodynamic_load: f32,
    moral_score: f64,
    coherence: f64,
    gwt_broadcast: bool,
    dream_insights: u64,
    surprise_triggered: bool,
    reasoning_confidence: f32,
    prediction_error: f32,
}

impl Vitals {
    /// `CycleMetadata`'s sub-structs (`consciousness`, `embodied`, `temporal`,
    /// `attention`, `memory`, `harmonics`, `ethics`, `neuromod`, ...) are ALL
    /// `#[serde(flatten)]`, so despite the nested Rust field access used
    /// elsewhere in this codebase (e.g. `m.consciousness.consciousness_level`),
    /// the actual wire JSON is flat — every field is a top-level key. Confirmed
    /// live against a real daemon 2026-07-12 (an earlier nested-path version of
    /// this function silently read `None` for everything).
    fn from_json(v: &Value) -> Option<Self> {
        // These four fields are the minimum evidence required to construct a
        // semantic cognitive state. Missing telemetry must not silently become
        // zeros: doing so would turn malformed gateway data into apparently
        // valid observations and could generate false timeline events.
        let required_f64 = |key: &str| -> Option<f64> {
            let value = v.get(key)?.as_f64()?;
            value.is_finite().then_some(value)
        };
        let required_f32 = |key: &str| -> Option<f32> {
            let value = required_f64(key)? as f32;
            value.is_finite().then_some(value)
        };
        let consciousness_level = required_f64("consciousness_level")?;
        let valence = required_f32("affective_valence")?;
        let arousal = required_f32("affective_arousal")?;
        let mood_temperature = required_f32("mood_temperature")?;
        let thermodynamic_load = required_f32("thermodynamic_load")?;
        let moral_score = required_f64("value_evaluator_score")?;
        let coherence = required_f64("harmonic_field_coherence")?;
        let reasoning_confidence = required_f32("reasoning_confidence")?;
        let prediction_error = required_f32("prediction_error")?;
        Some(Self {
            consciousness_level,
            valence,
            arousal,
            mood_temperature,
            thermodynamic_load,
            moral_score,
            coherence,
            gwt_broadcast: v["gwt_broadcast"].as_bool().unwrap_or(false),
            dream_insights: v["dream_insights"].as_u64().unwrap_or(0),
            surprise_triggered: v["surprise_triggered"].as_bool().unwrap_or(false),
            reasoning_confidence,
            prediction_error,
        })
    }
}

/// A decoded mental movie: RGBA frames ready for canvas `putImageData`.
/// Wire form (`mental_movie` key on the ws-live payload): base64 raw frames
/// of `width*height*channels` bytes, channels 1 or 3 — expanded to RGBA here
/// once, at parse time.
#[derive(Clone)]
struct Movie {
    frames_rgba: Vec<Vec<u8>>,
    width: u32,
    height: u32,
    semantic_coherence: f32,
}

/// Generous upper bound on a single "mental movie" frame's pixel count
/// (far more than a telemetry visualization frame plausibly needs). The
/// gateway URL is a user-editable text field, so a malicious or
/// compromised gateway must not be able to drive an unbounded (or, on
/// 32-bit wasm, integer-overflowing) allocation via `width`/`height`.
const MAX_MOVIE_PIXELS: usize = 4096 * 4096;
/// Bound the total RGBA allocation for one remote projection, not merely each
/// individual frame. A 4096² frame is already far larger than the UI canvas;
/// without a total budget, a bounded 120-frame sequence could still request
/// multi-gigabyte browser memory before the canvas ever displays it.
const MAX_MOVIE_RGBA_BYTES: usize = 64 * 1024 * 1024;

impl Movie {
    fn from_json(v: &Value) -> Option<Movie> {
        use base64::Engine as _;
        let m = v.get("mental_movie")?;
        // Decode dimensions without truncating hostile u64 JSON values into
        // the wasm32 u32/usize domain. Truncation here could turn an enormous
        // remote dimension into a small allocation and then leave indexing
        // arithmetic inconsistent with the declared wire shape.
        let width = u32::try_from(m["width"].as_u64()?).ok()?;
        let height = u32::try_from(m["height"].as_u64()?).ok()?;
        let channels = usize::try_from(m["channels"].as_u64()?).ok()?;
        if channels != 1 && channels != 3 {
            return None;
        }
        let engine = base64::engine::general_purpose::STANDARD;
        let px = (width as usize).checked_mul(height as usize)?;
        if px == 0 || px > MAX_MOVIE_PIXELS {
            return None;
        }
        let bytes_per_frame = px.checked_mul(channels)?;
        let rgba_capacity = px.checked_mul(4)?;
        const MAX_MOVIE_FRAMES: usize = 120;
        let frames = m["frames_b64"].as_array()?;
        if frames.len() > MAX_MOVIE_FRAMES {
            return None;
        }
        let total_rgba_bytes = rgba_capacity.checked_mul(frames.len())?;
        if total_rgba_bytes > MAX_MOVIE_RGBA_BYTES {
            return None;
        }
        let frames_rgba: Vec<Vec<u8>> = frames
            .iter()
            .filter_map(|f| engine.decode(f.as_str()?).ok())
            .filter(|raw| raw.len() == bytes_per_frame)
            .map(|raw| {
                let mut rgba = Vec::with_capacity(rgba_capacity);
                for i in 0..px {
                    let (r, g, b) = if channels >= 3 {
                        (
                            raw[i * channels],
                            raw[i * channels + 1],
                            raw[i * channels + 2],
                        )
                    } else {
                        (raw[i], raw[i], raw[i])
                    };
                    rgba.extend_from_slice(&[r, g, b, 255]);
                }
                rgba
            })
            .collect();
        if frames_rgba.is_empty() {
            return None;
        }
        Some(Movie {
            frames_rgba,
            width,
            height,
            semantic_coherence: m["semantic_coherence"]
                .as_f64()
                .filter(|value| value.is_finite())
                .and_then(|value| {
                    let value = value as f32;
                    value.is_finite().then_some(value)
                })
                .unwrap_or(0.0),
        })
    }
}

/// Extract the live cognitive self-portrait SVG as an image data URL.
///
/// The gateway payload is remote data. Rendering it through an `<img>` keeps
/// SVG markup out of the application DOM and prevents script/event attributes
/// from executing with page privileges.
fn portrait_from_json(v: &Value) -> Option<String> {
    let svg = v.get("canvas_svg")?.as_str()?;
    let start = svg.find("<svg")?;
    let svg = &svg[start..];
    if svg.len() > 512 * 1024 || !svg.trim_end().ends_with("</svg>") {
        return None;
    }
    // Keep the image-only rendering contract explicit at the trust boundary.
    // The browser already disables scripting and external resource loading for
    // SVG used through <img>, but rejecting active/resource-bearing SVG here
    // makes the invariant independent of browser-specific behavior and avoids
    // turning a future embedding change into a security regression.
    let lower = svg.to_ascii_lowercase();
    for marker in ["<script", "<foreignobject", "<iframe", "<object", "<embed", "<use", "href="http", "href='http", "xlink:href="http", "xlink:href='http"] {
        if lower.contains(marker) {
            return None;
        }
    }
    let encoded = base64::engine::general_purpose::STANDARD.encode(svg.as_bytes());
    Some(format!("data:image/svg+xml;base64,{encoded}"))
}

#[component]
pub fn App() -> impl IntoView {
    let gateway = RwSignal::new(DEFAULT_GATEWAY.to_string());
    let ws_connected = RwSignal::new(false);
    let vitals = RwSignal::new(Vitals::default());
    let telemetry_count = RwSignal::new(0_u64);
    let last_cycle = RwSignal::new(0_u64);

    let history = RwSignal::new(Vec::<Turn>::new());
    let draft = RwSignal::new(String::new());
    let sending = RwSignal::new(false);
    let last_error = RwSignal::new(Option::<String>::None);
    // Request generations prevent a response from an obsolete gateway/session
    // from mutating the current conversation or clearing a newer send state.
    let request_generation = Rc::new(Cell::new(0_u64));
    // Status responses cross an async boundary too. Keep a generation so a
    // response from the previous gateway cannot overwrite the new gateway's
    // liveness state if it resolves during a session transition.
    let status_generation = Rc::new(Cell::new(0_u64));
    let daemon_status = RwSignal::new(Option::<Value>::None);
    let events = RwSignal::new(Vec::<CognitiveEvent>::new());
    // Pauses the presentation of the event stream, not the daemon itself.
    // Incoming telemetry continues so resuming returns to current truth.
    let timeline_paused = RwSignal::new(false);

    // This is a derived semantic value, not another mutable state machine.
    // The memo only changes when presence/mode changes, so assistive
    // technology is not fed the continuous telemetry stream.
    let semantic_key = Memo::new(move |_| {
        let v = vitals.get();
        let state = CognitiveState::from_observation(
            ws_connected.get(),
            sending.get(),
            v.coherence,
            v.thermodynamic_load,
            v.reasoning_confidence as f64,
            v.prediction_error as f64,
        );
        format!("{}:{}", state.presence.label(), state.mode.label())
    });
    let semantic_announcement = Memo::new(move |_| {
        let key = semantic_key.get();
        let (presence, mode) = key.split_once(':').unwrap_or(("unknown", "unknown"));
        format!("Symthaea is {presence} and {mode}.")
    });

    // Projection exits (VISION_PROJECTION_REVIEW_2026-07-15.md P1.2): the
    // live cognitive self-portrait and the imagination decode, previously
    // produced every applicable cycle and displayed nowhere.
    let portrait = RwSignal::new(Option::<String>::None);
    let movie = RwSignal::new(Option::<Movie>::None);
    let movie_frame = RwSignal::new(0_usize);
    let movie_canvas = NodeRef::<leptos::html::Canvas>::new();

    // The gateway is a session boundary: changing it starts a new telemetry
    // generation. Owner-scoped cancellation handles the normal teardown;
    // the generation guard additionally makes stale callbacks harmless if
    // they race with a reconnect or gateway change.
    let telemetry_generation = Rc::new(Cell::new(0_u64));
    // Event identity spans telemetry sessions because the retained timeline
    // survives reconnects and gateway changes. Resetting this counter inside
    // each WebSocket task would allow old and new events to share a key.
    let event_sequence = Rc::new(Cell::new(0_u64));

    Effect::new(move |_| {
        let gw = gateway.get();
        // Gateway changes are authority changes, not merely reconnects.
        // Clear gateway-derived presentation state immediately so the UI
        // cannot display the previous service's cognition while the new
        // session is still negotiating or has not produced a fresh sample.
        ws_connected.set(false);
        vitals.set(Vitals::default());
        telemetry_count.set(0);
        last_cycle.set(0);
        daemon_status.set(None);
        portrait.set(None);
        movie.set(None);
        movie_frame.set(0);
        // Cycle identifiers are daemon-local; do not carry the previous
        // session's cycle into a new session that has not emitted telemetry.
        last_cycle.set(0);
        let session = {
            let mut generation = telemetry_generation.get();
            let Some(session) = next_telemetry_session(&mut generation) else {
                leptos::logging::error!("telemetry session generation exhausted; refusing to reuse identity");
                return;
            };
            telemetry_generation.set(generation);
            session
        };
        let telemetry_callback_generation = Rc::clone(&telemetry_generation);
        let connected_callback_generation = Rc::clone(&telemetry_generation);
        let disconnect_generation = Rc::clone(&telemetry_generation);
        let telemetry_gateway = gateway;
        let message_gateway = gw.clone();
        let connected_gateway = gw.clone();
        let disconnect_gateway = gw.clone();
        let telemetry_sequence = Rc::clone(&event_sequence);
        let connected_sequence = Rc::clone(&event_sequence);
        let disconnect_sequence = Rc::clone(&event_sequence);
        spawn_local_scoped_with_cancellation(async move {
            let mut previous_state: Option<CognitiveState> = None;
            let connected = api::stream_telemetry(
                &gw,
                move |payload| {
                    if !telemetry_session_is_current(
                        telemetry_callback_generation.get(),
                        session,
                    ) || telemetry_gateway.get_untracked() != message_gateway
                    {
                        return;
                    }
                    let Some(v) = Vitals::from_json(&payload) else {
                        leptos::logging::warn!(
                            "telemetry payload missing finite required cognitive measurements"
                        );
                        return;
                    };
                    let Some(cycle) = payload["cycle"].as_u64() else {
                        leptos::logging::warn!("telemetry payload missing cycle identifier");
                        return;
                    };
                    let current_state = CognitiveState::from_observation(
                        true,
                        sending.get_untracked(),
                        v.coherence,
                        v.thermodynamic_load,
                        v.reasoning_confidence as f64,
                        v.prediction_error as f64,
                    );
                    // Allocate identity only when a semantic event is actually
                    // emitted. A quiet telemetry sample must not consume event IDs:
                    // sequence numbers identify retained presentation events, not
                    // observation count.
                    let mut sequence_counter = telemetry_sequence.get();
                    if let Some(sequence) = next_event_sequence(&mut sequence_counter) {
                        if let Some(event) = event_between(
                            previous_state,
                            current_state,
                            sequence,
                            cycle,
                            v.surprise_triggered,
                            v.gwt_broadcast,
                        ) {
                            telemetry_sequence.set(sequence_counter);
                            events.update(|items| push_cognitive_event(items, event));
                        }
                    } else {
                        leptos::logging::error!("cognitive event sequence exhausted; suppressing event identity reuse");
                    }
                    previous_state = Some(current_state);
                    last_cycle.set(cycle);
                    vitals.set(v);
                    telemetry_count.update(|n| *n += 1);
                    if let Some(svg) = portrait_from_json(&payload) {
                        portrait.set(Some(svg));
                    }
                    if let Some(m) = Movie::from_json(&payload) {
                        movie.set(Some(m));
                        movie_frame.set(0);
                    }
                },
                move || {
                    if !telemetry_session_is_current(
                        connected_callback_generation.get(),
                        session,
                    ) || telemetry_gateway.get_untracked() != connected_gateway
                    {
                        return;
                    }
                    ws_connected.set(true);
                    let mut sequence_counter = connected_sequence.get();
                    if let Some(sequence) = next_event_sequence(&mut sequence_counter) {
                        connected_sequence.set(sequence_counter);
                        events.update(|items| {
                            push_cognitive_event(
                                items,
                                CognitiveEvent::lifecycle(sequence, CognitiveEventKind::Connected, 0),
                            );
                        });
                    } else {
                        leptos::logging::error!("cognitive event sequence exhausted; suppressing connected marker");
                    }
                },
            )
            .await;
            if connected
                && telemetry_session_is_current(
                    disconnect_generation.get(),
                    session,
                )
                && telemetry_gateway.get_untracked() == disconnect_gateway
            {
                ws_connected.set(false);
                let mut sequence_counter = disconnect_sequence.get();
                if let Some(sequence) = next_event_sequence(&mut sequence_counter) {
                    disconnect_sequence.set(sequence_counter);
                    let cycle = last_cycle.get_untracked();
                    events.update(|items| {
                        push_cognitive_event(
                            items,
                            CognitiveEvent::lifecycle(
                                sequence,
                                CognitiveEventKind::Disconnected,
                                cycle,
                            ),
                        );
                    });
                } else {
                    leptos::logging::error!("cognitive event sequence exhausted; suppressing disconnected marker");
                }
            }
        });
    });

    // Gateway changes also invalidate in-flight conversation requests.
    // The old future may still resolve, but its response is no longer
    // authoritative for the current gateway generation.
    {
        let request_generation = Rc::clone(&request_generation);
        Effect::new(move |_| {
            let _gateway_generation = gateway.get();
            let mut generation = request_generation.get();
            if next_authority_generation(&mut generation).is_some() {
                request_generation.set(generation);
                sending.set(false);
            } else {
                leptos::logging::error!(
                    "conversation request generation exhausted; refusing to reuse authority"
                );
                sending.set(false);
                last_error.set(Some(
                    "conversation request authority exhausted; reload required".to_string(),
                ));
            }
        });
    }

    // Poll GET-equivalent /v1/service status every 5s. This is baseline
    // liveness feedback independent of the telemetry WS above, which stays
    // silent whenever the daemon's experience bridge is off (the common
    // case in production) — without this, an idle daemon would look
    // indistinguishable from an unreachable one.
    Effect::new(move |_| {
        // Track the gateway so changing it starts a fresh status loop; the
        // previous owner-scoped loop is cancelled on effect cleanup.
        let _gateway_generation = gateway.get();
        let status_session = {
            let mut generation = status_generation.get();
            let Some(next) = next_authority_generation(&mut generation) else {
                leptos::logging::error!(
                    "status generation exhausted; refusing to reuse authority"
                );
                return;
            };
            status_generation.set(generation);
            next
        };
        let status_authority = Rc::clone(&status_generation);
        spawn_local_scoped_with_cancellation(async move {
            loop {
                let gw = gateway.get_untracked();
                let response = api::send_simple(&gw, "status").await;
                // Owner cancellation handles normal teardown; the generation
                // and direct gateway check together cover a response that
                // races the gateway change before the dependent effect reruns.
                if status_authority.get() != status_session
                    || gateway.get_untracked() != gw
                {
                    return;
                }
                match response {
                    Ok(resp) if resp["type"] != "error" => daemon_status.set(Some(resp)),
                    _ => daemon_status.set(None),
                }
                gloo_timers::future::TimeoutFuture::new(5_000).await;
            }
        });
    });

    // Advance the imagination loop at ~3fps whenever a movie is present.
    Effect::new(move |_| {
        spawn_local_scoped_with_cancellation(async move {
            loop {
                gloo_timers::future::TimeoutFuture::new(300).await;
                if movie.with_untracked(|m| m.as_ref().is_some_and(|m| m.frames_rgba.len() > 1)) {
                    movie_frame.update(|i| *i = i.wrapping_add(1));
                }
            }
        });
    });

    // Draw the current imagination frame whenever the movie or frame index
    // changes. putImageData wants RGBA at native size; CSS scales it up with
    // image-rendering: pixelated.
    Effect::new(move |_| {
        let idx = movie_frame.get();
        let Some(canvas) = movie_canvas.get() else {
            return;
        };
        movie.with(|m| {
            let Some(m) = m.as_ref() else { return };
            canvas.set_width(m.width);
            canvas.set_height(m.height);
            let Some(ctx) = canvas
                .get_context("2d")
                .ok()
                .flatten()
                .and_then(|c| c.dyn_into::<web_sys::CanvasRenderingContext2d>().ok())
            else {
                return;
            };
            let frame = &m.frames_rgba[idx % m.frames_rgba.len()];
            if let Ok(img) =
                web_sys::ImageData::new_with_u8_clamped_array(wasm_bindgen::Clamped(frame), m.width)
            {
                let _ = ctx.put_image_data(&img, 0.0, 0.0);
            }
        });
    });

    let send = move || {
        let text = draft.get_untracked();
        if text.trim().is_empty() || sending.get_untracked() {
            return;
        }
        let gw = gateway.get_untracked();
        let request_id = {
            let mut generation = request_generation.get();
            let Some(next) = next_authority_generation(&mut generation) else {
                leptos::logging::error!(
                    "conversation request generation exhausted; refusing to reuse authority"
                );
                last_error.set(Some(
                    "conversation request authority exhausted; reload required".to_string(),
                ));
                return;
            };
            request_generation.set(generation);
            next
        };
        draft.set(String::new());
        sending.set(true);
        last_error.set(None);
        history.update(|h| {
            h.push(Turn {
                role: "you",
                text: text.clone(),
            })
        });
        let request_generation = Rc::clone(&request_generation);
        spawn_local_scoped_with_cancellation(async move {
            let response = api::send_query(&gw, &text).await;
            if request_generation.get() != request_id
                || gateway.get_untracked() != gw
            {
                return;
            }
            match response {
                Ok(resp) => {
                    if resp["type"] == "error" {
                        let msg = resp["message"]
                            .as_str()
                            .unwrap_or("unknown error")
                            .to_string();
                        last_error.set(Some(msg));
                    } else {
                        let content = resp["content"].as_str().unwrap_or("").to_string();
                        history.update(|h| {
                            h.push(Turn {
                                role: "symthaea",
                                text: content,
                            })
                        });
                    }
                }
                Err(e) => last_error.set(Some(e)),
            }
            sending.set(false);
        });
    };

    let visible_spans = move || {
        let snapshot = if timeline_paused.get() {
            events.get_untracked()
        } else {
            events.get()
        };
        cognitive_spans(&snapshot)
            .into_iter()
            .rev()
            .take(4)
            .collect::<Vec<_>>()
    };

    view! {
        <div class="shell">
            <header class="shell-header">
                <h1>"Symthaea"</h1>
                <div class="gateway-row">
                    <label for="gateway">"gateway"</label>
                    <input id="gateway" type="text"
                        prop:value=move || gateway.get()
                        on:change=move |ev| gateway.set(event_target_value(&ev))
                    />
                    <span class=move || if ws_connected.get() { "dot dot-live" } else { "dot dot-dark" }
                        title=move || if ws_connected.get() { "telemetry connected" } else { "telemetry disconnected" }
                    ></span>
                </div>
                <div class="daemon-status">
                    {move || match daemon_status.get() {
                        Some(s) => format!(
                            "up {}s · {} requests · {} sleep cycles",
                            s["uptime_seconds"].as_u64().unwrap_or(0),
                            s["requests_processed"].as_u64().unwrap_or(0),
                            s["sleep_cycles"].as_u64().unwrap_or(0),
                        ),
                        None => "daemon unreachable".to_string(),
                    }}
                </div>
            </header>

            <section class="cognitive-state" aria-label="Cognitive state">
                {move || {
                    let v = vitals.get();
                    let state = CognitiveState::from_observation(
                        ws_connected.get(),
                        sending.get(),
                        v.coherence,
                        v.thermodynamic_load,
                        v.reasoning_confidence as f64,
                        v.prediction_error as f64,
                    );
                    view! {
                        <span class="sr-only" aria-live="polite">{move || semantic_announcement.get()}</span>
                        <div class="state-presence">
                            <span class="state-mode">{state.mode.label()}</span>
                            <span class="state-separator">" · "</span>
                            <span class="state-presence-label">{state.presence.label()}</span>
                        </div>
                        <p class="state-description">
                            {format!(
                                "coherence {:.2} · load {:.2} · confidence {:.2} · prediction error {:.2}",
                                state.coherence, state.thermodynamic_load, state.confidence, state.prediction_error
                            )}
                        </p>
                    }
                }}
            </section>

            <section class="vitals">
                <div class="orb-wrap">
                    <div class="orb"
                        style:animation-duration=move || {
                            let load = vitals.get().thermodynamic_load.clamp(0.0, 1.0);
                            format!("{:.2}s", 2.6 - (load as f64 * 1.8))
                        }
                        style:background=move || {
                            let v = vitals.get();
                            let warm = ((v.valence + 1.0) / 2.0).clamp(0.0, 1.0) as f64;
                            let hue = 260.0 - warm * 220.0;
                            format!("radial-gradient(circle at 35% 30%, hsl({hue:.0} 90% 70%), hsl({hue:.0} 70% 35%))")
                        }
                    ></div>
                </div>
                <dl class="readouts">
                    <div><dt>"integration measure"</dt><dd>{move || format!("{:.2}", vitals.get().consciousness_level)}</dd></div>
                    <div><dt>"valence"</dt><dd>{move || format!("{:+.2}", vitals.get().valence)}</dd></div>
                    <div><dt>"arousal"</dt><dd>{move || format!("{:.2}", vitals.get().arousal)}</dd></div>
                    <div><dt>"mood temp"</dt><dd>{move || format!("{:.2}", vitals.get().mood_temperature)}</dd></div>
                    <div><dt>"thermodynamic load"</dt><dd>{move || format!("{:.2}", vitals.get().thermodynamic_load)}</dd></div>
                    <div><dt>"moral score"</dt><dd>{move || format!("{:.2}", vitals.get().moral_score)}</dd></div>
                    <div><dt>"coherence"</dt><dd>{move || format!("{:.2}", vitals.get().coherence)}</dd></div>
                    <div><dt>"reasoning confidence"</dt><dd>{move || format!("{:.2}", vitals.get().reasoning_confidence)}</dd></div>
                    <div><dt>"prediction error"</dt><dd>{move || format!("{:.2}", vitals.get().prediction_error)}</dd></div>
                    <div><dt>"gwt broadcast"</dt><dd>{move || if vitals.get().gwt_broadcast { "yes" } else { "no" }}</dd></div>
                    <div><dt>"dream insights"</dt><dd>{move || vitals.get().dream_insights.to_string()}</dd></div>
                    <div><dt>"surprise"</dt><dd>{move || if vitals.get().surprise_triggered { "triggered" } else { "—" }}</dd></div>
                    <div><dt>"cycles observed"</dt><dd>{move || telemetry_count.get().to_string()}</dd></div>
                </dl>
                <p class="vitals-note">
                    "Telemetry only flows once the daemon's experience bridge is enabled "
                    "(--experience-bridge) and a query has been sent — it is turn-synchronous, "
                    "not an idle clock."
                </p>
            </section>

            <section class="cognitive-timeline" aria-label="Cognitive timeline">
                <div class="timeline-header">
                    <div>
                        <h2>"recent events"</h2>
                        <span class="timeline-state">
                            {move || if timeline_paused.get() { "paused view" } else { "live view" }}
                        </span>
                    </div>
                    <div class="timeline-controls">
                        <span class="timeline-count">{
                            move || {
                                if timeline_paused.get() {
                                    events.get_untracked().len().to_string()
                                } else {
                                    events.get().len().to_string()
                                }
                            }
                        }</span>
                        <button
                            type="button"
                            class="timeline-pause"
                            aria-pressed=move || timeline_paused.get().to_string()
                            on:click=move |_| timeline_paused.update(|paused| *paused = !*paused)
                        >
                            {move || if timeline_paused.get() { "resume" } else { "pause" }}
                        </button>
                    </div>
                </div>
                {move || {
                    let spans = visible_spans();
                    if spans.is_empty() {
                        None
                    } else {
                        Some(view! {
                            <div class="timeline-spans" aria-label="Observed cognitive spans">
                                <span class="timeline-spans-label">"cycle spans"</span>
                                <For
                                    each=move || spans.clone().into_iter()
                                    key=|span| span.start_sequence
                                    children=move |span| {
                                        let range = match span.end_cycle {
                                            Some(end) => format!("cycle {}–{}", span.start_cycle, end),
                                            None => format!("cycle {}–open", span.start_cycle),
                                        };
                                        view! {
                                            <span class="timeline-span">
                                                <span class="timeline-span-kind">{span.kind.label()}</span>
                                                <span class="timeline-span-range">{range}</span>
                                            </span>
                                        }
                                    }
                                />
                                <span class="timeline-spans-note">"open means open within the retained event window"</span>
                            </div>
                        })
                    }
                }}
                <div class="timeline-list" role="log" aria-live="off">
                    <For
                        each=move || if timeline_paused.get() {
                            events.get_untracked().into_iter().rev().enumerate().collect::<Vec<_>>()
                        } else {
                            events.get().into_iter().rev().enumerate().collect::<Vec<_>>()
                        }
                        key=|(_, event)| event.sequence
                        children=move |(_, event)| {
                            view! {
                                <div class="timeline-event">
                                    <span class="timeline-kind">{event.kind.label()}</span>
                                    <span class="timeline-cycle">{format!("cycle {}", event.cycle)}</span>
                                    <span class="timeline-basis">{event.evidence_basis.label()}</span>
                                    {move || match event.kind {
                                        CognitiveEventKind::SurpriseDetected => Some(view! {
                                            <span class="timeline-evidence">
                                                {format!("prediction error {:.2}", event.prediction_error)}
                                            </span>
                                        }),
                                        CognitiveEventKind::EnteredRest | CognitiveEventKind::ExitedRest => Some(view! {
                                            <span class="timeline-evidence">
                                                {format!("load {:.2}", event.thermodynamic_load)}
                                            </span>
                                        }),
                                        _ => None,
                                    }}
                                </div>
                            }
                        }
                    />
                    {move || if events.with(|items| items.is_empty()) {
                        Some(view! {
                            <p class="timeline-empty">"No semantic events yet."</p>
                        })
                    } else {
                        None
                    }}
                </div>
            </section>

            // Projections: what she renders of herself. Panes appear only
            // once the corresponding stream has actually delivered content.
            <section class="projections" style="display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start;">
                {move || portrait.get().map(|portrait_src| view! {
                    <div class="projection-pane">
                        <h2 style="font-size:0.9em;opacity:0.7;">"self-portrait"</h2>
                        <img class="portrait" style="max-width:220px;" src=portrait_src
                            alt="Live cognitive self-portrait" />
                    </div>
                })}
                <div class="projection-pane"
                    style:display=move || if movie.with(|m| m.is_some()) { "block" } else { "none" }
                >
                    <h2 style="font-size:0.9em;opacity:0.7;">"imagination"</h2>
                    <canvas node_ref=movie_canvas
                        style="width:192px;height:192px;image-rendering:pixelated;border-radius:8px;"
                    ></canvas>
                    {move || movie.with(|m| m.as_ref().map(|m| view! {
                        <p class="movie-note" style="font-size:0.8em;opacity:0.6;">
                            {format!("{} frames · coherence {:.2}", m.frames_rgba.len(), m.semantic_coherence)}
                        </p>
                    }))}
                </div>
            </section>

            <section class="converse">
                <div class="transcript">
                    <For
                        each=move || history.get().into_iter().enumerate()
                        key=|(i, _)| *i
                        children=move |(_, turn): (usize, Turn)| {
                            view! {
                                <div class=format!("turn turn-{}", turn.role)>
                                    <span class="turn-role">{turn.role}</span>
                                    <span class="turn-text">{turn.text}</span>
                                </div>
                            }
                        }
                    />
                </div>
                {move || last_error.get().map(|e| view! { <div class="error">{e}</div> })}
                <form class="composer" on:submit=move |ev| { ev.prevent_default(); send(); }>
                    <input type="text" placeholder="talk to symthaea..."
                        prop:value=move || draft.get()
                        on:input=move |ev| draft.set(event_target_value(&ev))
                    />
                    <button type="submit" disabled=move || sending.get()>
                        {move || if sending.get() { "…" } else { "send" }}
                    </button>
                </form>
            </section>
        </div>
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn valid_payload() -> Value {
        serde_json::json!({
            "cycle": 42,
            "consciousness_level": 0.7,
            "affective_valence": 0.2,
            "affective_arousal": 0.3,
            "mood_temperature": 0.4,
            "thermodynamic_load": 0.5,
            "value_evaluator_score": 0.6,
            "harmonic_field_coherence": 0.8,
            "reasoning_confidence": 0.9,
            "prediction_error": 0.1,
            "gwt_broadcast": false,
            "dream_insights": 0,
            "surprise_triggered": false
        })
    }

    #[test]
    fn telemetry_requires_finite_cognitive_measurements() {
        assert!(Vitals::from_json(&valid_payload()).is_some());
    }

    #[test]
    fn portrait_rejects_active_or_external_svg_content() {
        for svg in [
            r#"<svg><script>alert(1)</script></svg>"#,
            r#"<svg><foreignObject><div>x</div></foreignObject></svg>"#,
            r#"<svg><use href="https://attacker.example/icon.svg#x"/></svg>"#,
            r#"<svg><image href="https://attacker.example/pixel.png"/></svg>"#,
            r#"<svg><rect style="fill:url(https://attacker.example/pixel.svg#x)"/></svg>"#,
        ] {
            let payload = serde_json::json!({ "canvas_svg": svg });
            assert!(portrait_from_json(&payload).is_none(), "accepted: {svg}");
        }
    }

    #[test]
    fn movie_dimensions_and_channels_fail_closed_without_integer_truncation() {
        let mut payload = serde_json::json!({
            "mental_movie": {
                "width": (u32::MAX as u64) + 1,
                "height": 1,
                "channels": 1,
                "frames_b64": ["AA=="]
            }
        });
        assert!(Movie::from_json(&payload).is_none());

        payload["mental_movie"]["width"] = 1;
        payload["mental_movie"]["height"] = 1;
        payload["mental_movie"]["channels"] = 2;
        assert!(Movie::from_json(&payload).is_none());
    }

    #[test]
    fn movie_semantic_coherence_normalizes_non_finite_or_f32_overflow() {
        let mut payload = serde_json::json!({
            "mental_movie": {
                "width": 1,
                "height": 1,
                "channels": 1,
                "frames_b64": ["AA=="],
                "semantic_coherence": f64::MAX
            }
        });
        assert_eq!(Movie::from_json(&payload).unwrap().semantic_coherence, 0.0);

        payload["mental_movie"]["semantic_coherence"] = serde_json::json!(f64::NAN);
        assert_eq!(Movie::from_json(&payload).unwrap().semantic_coherence, 0.0);
    }

    #[test]
    fn movie_frame_byte_count_is_checked_before_filtering() {
        let payload = serde_json::json!({
            "mental_movie": {
                "width": 2,
                "height": 2,
                "channels": u64::MAX,
                "frames_b64": ["AA=="]
            }
        });
        assert!(Movie::from_json(&payload).is_none());
    }

    #[test]
    fn movie_rejects_unsupported_four_channel_frames() {
        let payload = serde_json::json!({
            "mental_movie": {
                "width": 1,
                "height": 1,
                "channels": 4,
                "frames_b64": ["AAAA"]
            }
        });
        assert!(Movie::from_json(&payload).is_none());
    }

    #[test]
    fn movie_rejects_trailing_frame_bytes_instead_of_silently_ignoring_them() {
        let payload = serde_json::json!({
            "mental_movie": {
                "width": 1,
                "height": 1,
                "channels": 1,
                "frames_b64": ["AQID"]
            }
        });
        assert!(Movie::from_json(&payload).is_none());
    }

    #[test]
    fn movie_rejects_more_than_the_bounded_frame_count() {
        let payload = serde_json::json!({
            "mental_movie": {
                "width": 1,
                "height": 1,
                "channels": 1,
                "frames_b64": vec!["AQ=="; 121]
            }
        });
        assert!(Movie::from_json(&payload).is_none());
    }

    #[test]
    fn movie_rejects_excessive_total_rgba_allocation() {
        let payload = serde_json::json!({
            "mental_movie": {
                "width": 4096,
                "height": 4096,
                "channels": 1,
                "frames_b64": vec!["AA=="; 120]
            }
        });
        assert!(Movie::from_json(&payload).is_none());
    }

    #[test]
    fn movie_accepts_a_bounded_total_rgba_allocation() {
        let payload = serde_json::json!({
            "mental_movie": {
                "width": 64,
                "height": 64,
                "channels": 1,
                "frames_b64": vec!["AA=="; 120]
            }
        });
        assert!(Movie::from_json(&payload).is_some());
    }

    #[test]
    fn authority_generation_is_monotonic() {
        let mut generation = 0_u64;
        let first = next_authority_generation(&mut generation).unwrap();
        let second = next_authority_generation(&mut generation).unwrap();
        assert_eq!(generation, 2);
        assert_eq!(first, 1);
        assert_eq!(second, 2);
        assert_ne!(first, second);
    }

    #[test]
    fn authority_generation_fails_closed_at_exhaustion() {
        let mut generation = u64::MAX - 1;
        assert_eq!(next_authority_generation(&mut generation), Some(u64::MAX));
        assert_eq!(generation, u64::MAX);
        assert_eq!(next_authority_generation(&mut generation), None);
        assert_eq!(generation, u64::MAX);
        assert_eq!(next_authority_generation(&mut generation), None);
        assert_eq!(generation, u64::MAX);
    }

    #[test]
    fn missing_required_telemetry_is_rejected_instead_of_zero_filled() {
        let mut payload = valid_payload();
        payload.as_object_mut().unwrap().remove("harmonic_field_coherence");
        assert!(Vitals::from_json(&payload).is_none());
    }

    #[test]
    fn f32_overflow_after_decode_is_rejected() {
        let mut payload = valid_payload();
        payload["thermodynamic_load"] = serde_json::json!(f64::MAX);
        assert!(Vitals::from_json(&payload).is_none());
    }

    #[test]
    fn f32_representable_boundary_remains_accepted() {
        let mut payload = valid_payload();
        payload["thermodynamic_load"] = serde_json::json!(f32::MAX as f64);
        assert!(Vitals::from_json(&payload).is_some());
    }
}
