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
use leptos::task::spawn_local;
use serde_json::Value;
use wasm_bindgen::JsCast;

use crate::api::{self};

const DEFAULT_GATEWAY: &str = "http://127.0.0.1:8090";

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
    dream_insights: usize,
    surprise_triggered: bool,
    reasoning_confidence: f32,
}

impl Vitals {
    /// `CycleMetadata`'s sub-structs (`consciousness`, `embodied`, `temporal`,
    /// `attention`, `memory`, `harmonics`, `ethics`, `neuromod`, ...) are ALL
    /// `#[serde(flatten)]`, so despite the nested Rust field access used
    /// elsewhere in this codebase (e.g. `m.consciousness.consciousness_level`),
    /// the actual wire JSON is flat — every field is a top-level key. Confirmed
    /// live against a real daemon 2026-07-12 (an earlier nested-path version of
    /// this function silently read `None` for everything).
    fn from_json(v: &Value) -> Self {
        let f64_at = |key: &str| -> f64 { v[key].as_f64().unwrap_or(0.0) };
        Self {
            consciousness_level: f64_at("consciousness_level"),
            valence: f64_at("affective_valence") as f32,
            arousal: f64_at("affective_arousal") as f32,
            mood_temperature: f64_at("mood_temperature") as f32,
            thermodynamic_load: f64_at("thermodynamic_load") as f32,
            moral_score: f64_at("value_evaluator_score"),
            coherence: f64_at("harmonic_field_coherence"),
            gwt_broadcast: v["gwt_broadcast"].as_bool().unwrap_or(false),
            dream_insights: v["dream_insights"].as_u64().unwrap_or(0) as usize,
            surprise_triggered: v["surprise_triggered"].as_bool().unwrap_or(false),
            reasoning_confidence: f64_at("reasoning_confidence") as f32,
        }
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

impl Movie {
    fn from_json(v: &Value) -> Option<Movie> {
        use base64::Engine as _;
        let m = v.get("mental_movie")?;
        let width = m["width"].as_u64()? as u32;
        let height = m["height"].as_u64()? as u32;
        let channels = m["channels"].as_u64()? as usize;
        let engine = base64::engine::general_purpose::STANDARD;
        let px = (width as usize) * (height as usize);
        let frames_rgba: Vec<Vec<u8>> = m["frames_b64"]
            .as_array()?
            .iter()
            .filter_map(|f| engine.decode(f.as_str()?).ok())
            .filter(|raw| raw.len() >= px * channels.max(1))
            .map(|raw| {
                let mut rgba = Vec::with_capacity(px * 4);
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
            semantic_coherence: m["semantic_coherence"].as_f64().unwrap_or(0.0) as f32,
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
    let encoded = base64::engine::general_purpose::STANDARD.encode(svg.as_bytes());
    Some(format!("data:image/svg+xml;base64,{encoded}"))
}

#[component]
pub fn App() -> impl IntoView {
    let gateway = RwSignal::new(DEFAULT_GATEWAY.to_string());
    let ws_connected = RwSignal::new(false);
    let vitals = RwSignal::new(Vitals::default());
    let telemetry_count = RwSignal::new(0_u64);

    let history = RwSignal::new(Vec::<Turn>::new());
    let draft = RwSignal::new(String::new());
    let sending = RwSignal::new(false);
    let last_error = RwSignal::new(Option::<String>::None);
    let daemon_status = RwSignal::new(Option::<Value>::None);

    // Projection exits (VISION_PROJECTION_REVIEW_2026-07-15.md P1.2): the
    // live cognitive self-portrait and the imagination decode, previously
    // produced every applicable cycle and displayed nowhere.
    let portrait = RwSignal::new(Option::<String>::None);
    let movie = RwSignal::new(Option::<Movie>::None);
    let movie_frame = RwSignal::new(0_usize);
    let movie_canvas = NodeRef::<leptos::html::Canvas>::new();

    // Open the telemetry stream once, on mount, against whatever gateway
    // URL is set at that moment. Reconnecting on URL change is a v1 nicety
    // — not required for the wiring to be real and useful today.
    Effect::new(move |_| {
        let gw = gateway.get_untracked();
        ws_connected.set(false);
        spawn_local(async move {
            ws_connected.set(true);
            api::stream_telemetry(&gw, move |payload| {
                vitals.set(Vitals::from_json(&payload));
                telemetry_count.update(|n| *n += 1);
                if let Some(svg) = portrait_from_json(&payload) {
                    portrait.set(Some(svg));
                }
                if let Some(m) = Movie::from_json(&payload) {
                    movie.set(Some(m));
                    movie_frame.set(0);
                }
            })
            .await;
            ws_connected.set(false);
        });
    });

    // Poll GET-equivalent /v1/service status every 5s. This is baseline
    // liveness feedback independent of the telemetry WS above, which stays
    // silent whenever the daemon's experience bridge is off (the common
    // case in production) — without this, an idle daemon would look
    // indistinguishable from an unreachable one.
    Effect::new(move |_| {
        spawn_local(async move {
            loop {
                let gw = gateway.get_untracked();
                match api::send_simple(&gw, "status").await {
                    Ok(resp) if resp["type"] != "error" => daemon_status.set(Some(resp)),
                    _ => daemon_status.set(None),
                }
                gloo_timers::future::TimeoutFuture::new(5_000).await;
            }
        });
    });

    // Advance the imagination loop at ~3fps whenever a movie is present.
    Effect::new(move |_| {
        spawn_local(async move {
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
        draft.set(String::new());
        sending.set(true);
        last_error.set(None);
        history.update(|h| {
            h.push(Turn {
                role: "you",
                text: text.clone(),
            })
        });
        let gw = gateway.get_untracked();
        spawn_local(async move {
            match api::send_query(&gw, &text).await {
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
                    <div><dt>"consciousness"</dt><dd>{move || format!("{:.1}%", vitals.get().consciousness_level * 100.0)}</dd></div>
                    <div><dt>"valence"</dt><dd>{move || format!("{:+.2}", vitals.get().valence)}</dd></div>
                    <div><dt>"arousal"</dt><dd>{move || format!("{:.2}", vitals.get().arousal)}</dd></div>
                    <div><dt>"mood temp"</dt><dd>{move || format!("{:.2}", vitals.get().mood_temperature)}</dd></div>
                    <div><dt>"thermodynamic load"</dt><dd>{move || format!("{:.2}", vitals.get().thermodynamic_load)}</dd></div>
                    <div><dt>"moral score"</dt><dd>{move || format!("{:.2}", vitals.get().moral_score)}</dd></div>
                    <div><dt>"coherence"</dt><dd>{move || format!("{:.2}", vitals.get().coherence)}</dd></div>
                    <div><dt>"reasoning confidence"</dt><dd>{move || format!("{:.2}", vitals.get().reasoning_confidence)}</dd></div>
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
