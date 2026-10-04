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
use std::cell::RefCell;
use std::rc::Rc;
use symthaea_canvas::{GpuScene, RemoteScene, WebGpuMovieRenderer, WebGpuRenderer};
#[cfg(feature = "browser-qualification")]
use symthaea_canvas::{Color, SceneNode};
#[cfg(feature = "browser-qualification")]
use symthaea_canvas::scene_graph::{Style, Transform};
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
    dream_insights: u64,
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
            dream_insights: v["dream_insights"].as_u64().unwrap_or(0),
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

/// Bound decoded mental-movie dimensions, frame count, and expanded RGBA
/// storage together. The gateway is remote input and must not be able to
/// drive an unbounded allocation through width/height/frame multiplicity.
const MAX_MOVIE_PIXELS: usize = 2048 * 2048;
const MAX_MOVIE_FRAMES: usize = 24;
const MAX_MOVIE_RGBA_BYTES: usize = 32 * 1024 * 1024;

#[cfg(feature = "browser-qualification")]
fn browser_qualification_scene() -> RemoteScene {
    let background = SceneNode::rect(0.0, 0.0, 512.0, 512.0).with_style(Style {
        fill: Some(Color::rgb(0.04, 0.06, 0.10)),
        ..Style::default()
    });
    let concave = SceneNode::polygon(
        vec![
            (52.0, 72.0),
            (420.0, 72.0),
            (420.0, 180.0),
            (246.0, 180.0),
            (246.0, 430.0),
            (52.0, 430.0),
        ],
        true,
    )
    .with_style(Style {
        fill: Some(Color::rgba(0.92, 0.48, 0.16, 0.9)),
        stroke: Some(Color::rgb(0.95, 0.9, 0.75)),
        stroke_width: Some(3.0),
        ..Style::default()
    });
    let circle = SceneNode::circle(360.0, 350.0, 70.0).with_style(Style {
        fill: Some(Color::rgba(0.18, 0.72, 0.92, 0.75)),
        ..Style::default()
    });
    let transform_group = SceneNode::group(Some("transformed"))
        .with_transform(Transform {
            translate_x: 18.0,
            translate_y: -14.0,
            rotate_deg: 7.0,
            scale: 0.82,
        })
        .with_style(Style {
            opacity: Some(0.65),
            ..Style::default()
        })
        .with_child(
            SceneNode::line(80.0, 470.0, 450.0, 120.0).with_style(Style {
                stroke: Some(Color::rgb(0.9, 0.96, 1.0)),
                stroke_width: Some(4.0),
                ..Style::default()
            }),
        );
    RemoteScene::from_scene(
        &SceneNode::group(Some("browser-qualification"))
            .with_child(background)
            .with_child(concave)
            .with_child(circle)
            .with_child(transform_group),
    )
}

#[cfg(feature = "browser-qualification")]
fn browser_qualification_portrait() -> String {
    use base64::Engine as _;
    const SVG: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512"><rect width="512" height="512" fill="#0a0f1a"/><circle cx="256" cy="256" r="150" fill="#2eb8ea" fill-opacity=".22" stroke="#f2ead0" stroke-width="4"/><path d="M156 290 C196 210 232 210 256 270 C280 210 316 210 356 290" fill="none" stroke="#f2ead0" stroke-width="6"/><circle cx="210" cy="255" r="12" fill="#f2ead0"/><circle cx="302" cy="255" r="12" fill="#f2ead0"/></svg>"##;
    format!(
        "data:image/svg+xml;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(SVG.as_bytes())
    )
}

#[cfg(feature = "browser-qualification")]
fn browser_qualification_movie() -> Movie {
    let width = 32u32;
    let height = 24u32;
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 255) / (width - 1)) as u8;
            let g = ((y * 255) / (height - 1)) as u8;
            let b = (((x + y) * 255) / (width + height - 2)) as u8;
            rgba.extend_from_slice(&[r, g, b, 255]);
        }
    }
    Movie {
        frames_rgba: vec![rgba],
        width,
        height,
        semantic_coherence: 1.0,
    }
}

impl Movie {
    fn from_json(v: &Value) -> Option<Movie> {
        use base64::Engine as _;
        let m = v.get("mental_movie")?;
        let width_raw = m["width"].as_u64()?;
        let height_raw = m["height"].as_u64()?;
        if width_raw == 0
            || height_raw == 0
            || width_raw > 2048
            || height_raw > 2048
        {
            return None;
        }
        let width = width_raw as u32;
        let height = height_raw as u32;
        let channels_raw = m["channels"].as_u64()?;
        if channels_raw != 1 && channels_raw != 3 {
            return None;
        }
        let channels = channels_raw as usize;
        let engine = base64::engine::general_purpose::STANDARD;
        let px = (width as usize).checked_mul(height as usize)?;
        if px == 0 || px > MAX_MOVIE_PIXELS {
            return None;
        }
        let frame_bytes = px.checked_mul(4)?;
        let expected_raw_bytes = px.checked_mul(channels)?;
        let frames = m["frames_b64"].as_array()?;
        if frames.is_empty() || frames.len() > MAX_MOVIE_FRAMES {
            return None;
        }
        let total_rgba_bytes = frame_bytes.checked_mul(frames.len())?;
        if total_rgba_bytes > MAX_MOVIE_RGBA_BYTES {
            return None;
        }

        let mut frames_rgba = Vec::with_capacity(frames.len());
        for encoded in frames {
            let encoded = encoded.as_str()?;
            let max_encoded_len = expected_raw_bytes
                .checked_add(2)?
                .checked_div(3)?
                .checked_mul(4)?;
            if encoded.len() > max_encoded_len {
                return None;
            }
            let raw = engine.decode(encoded).ok()?;
            if raw.len() != expected_raw_bytes {
                return None;
            }

            let mut rgba = Vec::with_capacity(frame_bytes);
            for i in 0..px {
                let (r, g, b) = if channels == 3 {
                    (
                        raw[i * channels],
                        raw[i * channels + 1],
                        raw[i * channels + 2],
                    )
                } else {
                    let value = raw[i];
                    (value, value, value)
                };
                rgba.extend_from_slice(&[r, g, b, 255]);
            }
            frames_rgba.push(rgba);
        }

        let semantic_coherence = m["semantic_coherence"]
            .as_f64()
            .map(|value| value as f32)
            .filter(|value| value.is_finite())
            .unwrap_or(0.0)
            .clamp(-1.0, 1.0);

        Some(Movie {
            frames_rgba,
            width,
            height,
            semantic_coherence,
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
    let movie_webgpu_canvas = NodeRef::<leptos::html::Canvas>::new();
    let webgpu_canvas = NodeRef::<leptos::html::Canvas>::new();
    let gpu_scene = RwSignal::new(Option::<RemoteScene>::None);

    #[cfg(feature = "browser-qualification")]
    {
        let enabled = web_sys::window()
            .and_then(|window| window.location().search().ok())
            .is_some_and(|search| search.contains("symthaea_webgpu_fixture=1"));
        if enabled {
            let scene = browser_qualification_scene();
            let fixture_movie = browser_qualification_movie();
            let fixture_portrait = browser_qualification_portrait();
            Effect::new(move |_| {
                gpu_scene.set(Some(scene.clone()));
                movie.set(Some(fixture_movie.clone()));
                portrait.set(Some(fixture_portrait.clone()));
            });
        }
    }
    let webgpu_ready = RwSignal::new(false);
    let movie_webgpu_ready = RwSignal::new(false);
    let webgpu_renderer: Rc<RefCell<Option<WebGpuRenderer>>> = Rc::new(RefCell::new(None));
    let movie_webgpu_renderer: Rc<RefCell<Option<WebGpuMovieRenderer>>> =
        Rc::new(RefCell::new(None));
    let webgpu_init_started = Rc::new(RefCell::new(false));
    let movie_webgpu_init_started = Rc::new(RefCell::new(false));
    #[cfg(feature = "browser-qualification")]
    let qualification_cognitive_init_count = Rc::new(RefCell::new(0_u32));
    #[cfg(feature = "browser-qualification")]
    let qualification_movie_init_count = Rc::new(RefCell::new(0_u32));
    #[cfg(feature = "browser-qualification")]
    let qualification_cognitive_loss_requested = Rc::new(RefCell::new(false));
    #[cfg(feature = "browser-qualification")]
    let qualification_movie_loss_requested = Rc::new(RefCell::new(false));

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
                if let Some(scene_value) = payload.get("canvas_scene") {
                    if let Ok(scene) = serde_json::from_value::<RemoteScene>(scene_value.clone()) {
                        if scene.is_supported() {
                            gpu_scene.set(Some(scene));
                        }
                    }
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

    // Initialize WebGPU once after the browser canvas is mounted. Failure is
    // non-fatal: the existing SVG projection remains the compatibility path.
    {
        let renderer = Rc::clone(&webgpu_renderer);
        let started = Rc::clone(&webgpu_init_started);
        #[cfg(feature = "browser-qualification")]
        let init_count = Rc::clone(&qualification_cognitive_init_count);
        #[cfg(feature = "browser-qualification")]
        let loss_requested = Rc::clone(&qualification_cognitive_loss_requested);
        Effect::new(move |_| {
            let ready = webgpu_ready.get();
            if ready || *started.borrow() {
                return;
            }
            let Some(canvas) = webgpu_canvas.get() else {
                return;
            };
            *started.borrow_mut() = true;
            let renderer = Rc::clone(&renderer);
            #[cfg(feature = "browser-qualification")]
            let init_count = Rc::clone(&init_count);
            #[cfg(feature = "browser-qualification")]
            let loss_requested = Rc::clone(&loss_requested);
            spawn_local(async move {
                match WebGpuRenderer::new(canvas).await {
                    Ok(gpu) => {
                        *renderer.borrow_mut() = Some(gpu);
                        webgpu_ready.set(true);
                        #[cfg(feature = "browser-qualification")]
                        {
                            let count = {
                                let mut count = init_count.borrow_mut();
                                *count += 1;
                                *count
                            };
                            if let Some(canvas) = webgpu_canvas.get_untracked() {
                                let _ = canvas.set_attribute(
                                    "data-qualification-init-count",
                                    &count.to_string(),
                                );
                            }
                            if count >= 2 {
                                if let Some(canvas) = webgpu_canvas.get_untracked() {
                                    let _ = canvas.set_attribute("data-qualification-recovered", "true");
                                }
                            } else if web_sys::window()
                                .and_then(|window| window.location().search().ok())
                                .is_some_and(|search| search.contains("symthaea_webgpu_recovery=1"))
                            {
                                let renderer_for_loss = Rc::clone(&renderer);
                                let requested_for_loss = Rc::clone(&loss_requested);
                                spawn_local(async move {
                                    gloo_timers::future::TimeoutFuture::new(100).await;
                                    if *requested_for_loss.borrow() {
                                        return;
                                    }
                                    if let Some(renderer) = renderer_for_loss.borrow().as_ref() {
                                        *requested_for_loss.borrow_mut() = true;
                                        if let Some(canvas) = webgpu_canvas.get_untracked() {
                                            let _ = canvas.set_attribute(
                                                "data-qualification-recovery-requested",
                                                "true",
                                            );
                                        }
                                        renderer.qualification_force_device_loss();
                                    }
                                });
                            }
                        }
                    }
                    Err(error) => {
                        leptos::logging::warn!("WebGPU unavailable: {error}");
                    }
                }
            });
        });
    }

    // Initialize the WebGPU movie renderer independently from the cognitive
    // scene renderer. Either projection can degrade to its legacy path alone.
    {
        let renderer = Rc::clone(&movie_webgpu_renderer);
        let started = Rc::clone(&movie_webgpu_init_started);
        #[cfg(feature = "browser-qualification")]
        let init_count = Rc::clone(&qualification_movie_init_count);
        #[cfg(feature = "browser-qualification")]
        let loss_requested = Rc::clone(&qualification_movie_loss_requested);
        Effect::new(move |_| {
            let ready = movie_webgpu_ready.get();
            if ready || *started.borrow() {
                return;
            }
            let Some(canvas) = movie_webgpu_canvas.get() else {
                return;
            };
            *started.borrow_mut() = true;
            let renderer = Rc::clone(&renderer);
            #[cfg(feature = "browser-qualification")]
            let init_count = Rc::clone(&init_count);
            #[cfg(feature = "browser-qualification")]
            let loss_requested = Rc::clone(&loss_requested);
            spawn_local(async move {
                match WebGpuMovieRenderer::new(canvas).await {
                    Ok(gpu) => {
                        *renderer.borrow_mut() = Some(gpu);
                        movie_webgpu_ready.set(true);
                        #[cfg(feature = "browser-qualification")]
                        {
                            let count = {
                                let mut count = init_count.borrow_mut();
                                *count += 1;
                                *count
                            };
                            if let Some(canvas) = movie_webgpu_canvas.get_untracked() {
                                let _ = canvas.set_attribute(
                                    "data-qualification-init-count",
                                    &count.to_string(),
                                );
                            }
                            if count >= 2 {
                                if let Some(canvas) = movie_webgpu_canvas.get_untracked() {
                                    let _ = canvas.set_attribute("data-qualification-recovered", "true");
                                }
                            } else if web_sys::window()
                                .and_then(|window| window.location().search().ok())
                                .is_some_and(|search| search.contains("symthaea_webgpu_recovery=1"))
                            {
                                let renderer_for_loss = Rc::clone(&renderer);
                                let requested_for_loss = Rc::clone(&loss_requested);
                                spawn_local(async move {
                                    gloo_timers::future::TimeoutFuture::new(100).await;
                                    if *requested_for_loss.borrow() {
                                        return;
                                    }
                                    if let Some(renderer) = renderer_for_loss.borrow().as_ref() {
                                        *requested_for_loss.borrow_mut() = true;
                                        if let Some(canvas) = movie_webgpu_canvas.get_untracked() {
                                            let _ = canvas.set_attribute(
                                                "data-qualification-recovery-requested",
                                                "true",
                                            );
                                        }
                                        renderer.qualification_force_device_loss();
                                    }
                                });
                            }
                        }
                    }
                    Err(error) => {
                        leptos::logging::warn!("WebGPU movie renderer unavailable: {error}");
                    }
                }
            });
        });
    }

    // Render each typed cognitive scene through WebGPU. The renderer-neutral
    // scene is reconstructed into native scene nodes only at the backend edge.
    {
        let renderer = Rc::clone(&webgpu_renderer);
        let init_started = Rc::clone(&webgpu_init_started);
        Effect::new(move |_| {
            if !webgpu_ready.get() {
                return;
            }
            let Some(scene) = gpu_scene.get() else {
                return;
            };
            let native = scene.to_scene_node();
            let gpu = GpuScene::from_scene(&native);
            #[cfg(feature = "browser-qualification")]
            if let Some(canvas) = webgpu_canvas.get_untracked() {
                let _ = canvas.set_attribute("data-qualification-ready", "false");
            }
            let render_result = {
                let mut renderer_ref = renderer.borrow_mut();
                let Some(renderer) = renderer_ref.as_mut() else {
                    return;
                };
                renderer.render(&gpu)
            };
            if let Err(error) = render_result {
                leptos::logging::warn!("WebGPU cognitive canvas render failed: {error}");
                *renderer.borrow_mut() = None;
                *init_started.borrow_mut() = false;
                webgpu_ready.set(false);
            } else {
                #[cfg(feature = "browser-qualification")]
                if let Some(canvas) = webgpu_canvas.get_untracked() {
                    let _ = canvas.set_attribute("data-qualification-ready", "true");
                }
            }
        });
    }

    // Poll WebGPU device health so a static cognitive projection can
    // demote to its compatibility renderer even when no new scene arrives.
    {
        let health_init_started = Rc::clone(&webgpu_init_started);
        Effect::new(move |_| {
            if !webgpu_ready.get() {
                return;
            }
            let renderer = Rc::clone(&webgpu_renderer);
            let init_started = Rc::clone(&health_init_started);
            spawn_local(async move {
            loop {
                if !webgpu_ready.get_untracked() {
                    break;
                }
                gloo_timers::future::TimeoutFuture::new(500).await;
                if !webgpu_ready.get_untracked() {
                    break;
                }
                let lost = renderer
                    .borrow()
                    .as_ref()
                    .is_some_and(WebGpuRenderer::is_device_lost);
                if lost {
                    #[cfg(feature = "browser-qualification")]
                    if let Some(canvas) = webgpu_canvas.get_untracked() {
                        let _ = canvas.set_attribute("data-qualification-loss-observed", "true");
                    }
                    *renderer.borrow_mut() = None;
                    *init_started.borrow_mut() = false;
                    webgpu_ready.set(false);
                    break;
                }
            }
        });
        });
    }

    // Poll the movie renderer independently for the same device-loss handoff.
    {
        let renderer = Rc::clone(&movie_webgpu_renderer);
        let health_init_started = Rc::clone(&movie_webgpu_init_started);
        Effect::new(move |_| {
            if !movie_webgpu_ready.get() {
                return;
            }
            let renderer = Rc::clone(&renderer);
            let init_started = Rc::clone(&health_init_started);
            spawn_local(async move {
                loop {
                    if !movie_webgpu_ready.get_untracked() {
                        break;
                    }
                    gloo_timers::future::TimeoutFuture::new(500).await;
                    if !movie_webgpu_ready.get_untracked() {
                        break;
                    }
                    let lost = renderer
                        .borrow()
                        .as_ref()
                        .is_some_and(WebGpuMovieRenderer::is_device_lost);
                    if lost {
                        #[cfg(feature = "browser-qualification")]
                        if let Some(canvas) = movie_webgpu_canvas.get_untracked() {
                            let _ = canvas.set_attribute("data-qualification-loss-observed", "true");
                        }
                        *renderer.borrow_mut() = None;
                        *init_started.borrow_mut() = false;
                        movie_webgpu_ready.set(false);
                        break;
                    }
                }
            });
        });
        });
    }

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

    // Render the current imagination frame through WebGPU when available.
    // The texture is persistent across frames; only the RGBA payload changes.
    {
        let init_started = Rc::clone(&movie_webgpu_init_started);
        Effect::new(move |_| {
            if !movie_webgpu_ready.get() {
                return;
            }
            let idx = movie_frame.get();
        let Some(movie) = movie.get() else {
            return;
        };
        #[cfg(feature = "browser-qualification")]
        if let Some(canvas) = movie_webgpu_canvas.get_untracked() {
            let _ = canvas.set_attribute("data-qualification-ready", "false");
        }
        let frame = &movie.frames_rgba[idx % movie.frames_rgba.len()];
        let render_result = {
            let mut renderer_ref = movie_webgpu_renderer.borrow_mut();
            let Some(renderer) = renderer_ref.as_mut() else {
                return;
            };
            renderer.render(movie.width, movie.height, frame)
        };
        if let Err(error) = render_result {
            leptos::logging::warn!("WebGPU movie render failed: {error}");
            *movie_webgpu_renderer.borrow_mut() = None;
            *init_started.borrow_mut() = false;
            movie_webgpu_ready.set(false);
        } else {
            #[cfg(feature = "browser-qualification")]
            if let Some(canvas) = movie_webgpu_canvas.get_untracked() {
                let _ = canvas.set_attribute("data-qualification-ready", "true");
            }
        }
        });
    }

    // Draw the current imagination frame through Canvas2D as a graceful
    // fallback. putImageData wants RGBA at native size; CSS scales it up with
    // image-rendering: pixelated.
    Effect::new(move |_| {
        if movie_webgpu_ready.get() {
            return;
        }
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
                <div class="projection-pane"
                    style:display=move || {
                        let visible = webgpu_ready.get() || portrait.with(|p| p.is_some());
                        if visible { "block" } else { "none" }
                    }
                >
                    <h2 style="font-size:0.9em;opacity:0.7;">"self-portrait"</h2>
                    <canvas id="webgpu-cognitive-canvas" node_ref=webgpu_canvas
                        aria-label="WebGPU cognitive projection"
                        style="width:220px;height:220px;border-radius:8px;"
                        style:display=move || if webgpu_ready.get() { "block" } else { "none" }
                        width="512" height="512"
                    ></canvas>
                    {move || (!webgpu_ready.get()).then(|| portrait.get()).flatten().map(|portrait_src| view! {
                        <img class="portrait" style="max-width:220px;" src=portrait_src
                            alt="Live cognitive self-portrait (SVG fallback)" />
                    })}
                </div>
                <div class="projection-pane"
                    style:display=move || if movie.with(|m| m.is_some()) { "block" } else { "none" }
                >
                    <h2 style="font-size:0.9em;opacity:0.7;">"imagination"</h2>
                    <canvas id="webgpu-movie-canvas" node_ref=movie_webgpu_canvas
                        aria-label="WebGPU mental movie"
                        style:display=move || if movie_webgpu_ready.get() { "block" } else { "none" }
                        style="width:192px;height:192px;image-rendering:pixelated;border-radius:8px;"
                        width="192" height="192"
                    ></canvas>
                    <canvas id="canvas2d-movie-fallback" node_ref=movie_canvas
                        aria-label="Canvas 2D mental movie fallback"
                        style:display=move || if movie_webgpu_ready.get() { "none" } else { "block" }
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
    use base64::Engine as _;

    fn movie_json(width: u64, height: u64, channels: u64, frames: usize) -> Value {
        // Keep malformed-dimension fixtures tiny: the parser must reject them
        // before any payload-sized allocation becomes possible.
        let raw_len = if width <= 64 && height <= 64 {
            width
                .checked_mul(height)
                .and_then(|px| px.checked_mul(channels))
                .unwrap_or(0) as usize
        } else {
            0
        };
        let raw = vec![7_u8; raw_len];
        let encoded = base64::engine::general_purpose::STANDARD.encode(raw);
        serde_json::json!({
            "mental_movie": {
                "width": width,
                "height": height,
                "channels": channels,
                "frames_b64": vec![encoded; frames],
                "semantic_coherence": 1.7
            }
        })
    }

    #[test]
    fn movie_parser_accepts_bounded_grayscale_frame() {
        let movie = Movie::from_json(&movie_json(2, 2, 1, 1)).expect("valid movie");
        assert_eq!(movie.frames_rgba.len(), 1);
        assert_eq!(movie.frames_rgba[0], vec![7, 7, 7, 255, 7, 7, 7, 255, 7, 7, 7, 255, 7, 7, 7, 255]);
        assert_eq!(movie.semantic_coherence, 1.0);
    }

    #[test]
    fn movie_parser_rejects_unsupported_channels() {
        assert!(Movie::from_json(&movie_json(2, 2, 2, 1)).is_none());
    }

    #[test]
    fn movie_parser_rejects_channels_that_would_truncate_on_wasm() {
        assert!(Movie::from_json(&movie_json(2, 2, (u32::MAX as u64) + 1, 1)).is_none());
    }


    #[test]
    fn movie_parser_rejects_dimension_overflow_before_cast() {
        assert!(Movie::from_json(&movie_json((u32::MAX as u64) + 1, 1, 1, 1)).is_none());
    }

    #[test]
    fn movie_parser_rejects_excessive_frame_count() {
        assert!(Movie::from_json(&movie_json(2, 2, 1, MAX_MOVIE_FRAMES + 1)).is_none());
    }

    #[test]
    fn movie_parser_rejects_total_rgba_budget_before_decode() {
        let oversized = serde_json::json!({
            "mental_movie": {
                "width": 2048,
                "height": 2048,
                "channels": 1,
                "frames_b64": ["", "", ""],
            }
        });
        assert!(Movie::from_json(&oversized).is_none());
    }

    #[test]
    fn movie_parser_rejects_oversized_base64_before_decode() {
        let width = 64_u32;
        let height = 64_u32;
        let expected_raw_bytes = (width * height) as usize;
        let max_encoded_len = expected_raw_bytes.div_ceil(3) * 4;
        let oversized = serde_json::json!({
            "mental_movie": {
                "width": width,
                "height": height,
                "channels": 1,
                "frames_b64": ["A".repeat(max_encoded_len + 1)],
            }
        });
        assert!(Movie::from_json(&oversized).is_none());
    }
}
