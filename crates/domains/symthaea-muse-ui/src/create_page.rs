// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Create Mode: the compose form, ported from the legacy `studio/index.html`
//! Studio tab's core loop (describe a piece, get N genuinely different
//! candidates, pick one) — not the full Studio Mode precision-editing
//! surface from `UI_mocks/MUSE_STUDIO_MODE_DESIGN_SPEC.md` (that needs
//! constrained alternative-generation and a version graph the backend
//! doesn't have yet; see that spec's own note on this).
//!
//! Picking a candidate here writes it into the shared `MuseState` and
//! hands off to Listen Mode, the same way the legacy page's "Listen"
//! prefetch queue could pull from `latestBatch`.

use leptos::prelude::*;
use leptos::task::spawn_local;
use leptos_router::hooks::use_navigate;

use symthaea_muse_protocol::catalog::{self, CanonicalStyle, Constellation};

use crate::api::{self, Candidate, ComposeRequest};
use crate::icons::HeartIcon;
use crate::palette;
use crate::state::MuseState;

const NOTE_NAMES: [&str; 12] = [
    "C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B",
];

#[component]
pub fn CreatePage() -> impl IntoView {
    let muse = use_context::<MuseState>().expect("MuseState provided by App");
    let navigate = use_navigate();

    let style = RwSignal::new("Classical".to_string());

    // The catalog entry for whatever engine style is currently selected, and
    // the constellation it belongs to. Both are DERIVED from `style` rather
    // than held as separate state: the compose request only ever carries an
    // engine style name, so making that the single source of truth means the
    // constellation picker cannot drift out of sync with it — including when
    // Today's Discoveries or Surprise Me set `style` directly below.
    //
    // Reads the compile-time `CATALOG` const straight from the protocol crate
    // this app already links; there is no /api/catalog round-trip (see the
    // note in api.rs).
    let selected_entry = Memo::new(move |_| -> Option<CanonicalStyle> {
        let s = style.get();
        catalog::CATALOG
            .iter()
            .find(|e| e.composer_style == Some(s.as_str()))
            .copied()
    });
    let current_constellation = Memo::new(move |_| {
        selected_entry
            .get()
            .map(|e| e.constellation)
            // Defensive only: every engine Style has a catalog entry, and
            // muse_studio's catalog_entries_all_resolve_to_real_engine_styles
            // test enforces that 1:1 mapping.
            .unwrap_or(Constellation::ClassicalLyricCharacter)
    });

    let valence = RwSignal::new(0.0_f32);
    let arousal = RwSignal::new(0.5_f32);
    let energy = RwSignal::new(0.5_f32);
    let tonic = RwSignal::new(0_i32);
    let bars = RwSignal::new(4_usize);
    let n_candidates = RwSignal::new(3_u64);
    let prompt = RwSignal::new(String::new());
    let use_motif_foundry = RwSignal::new(false);
    let composition_lesson_id = RwSignal::new(String::new());

    let composing = RwSignal::new(false);
    let error = RwSignal::new(None::<String>);
    let results = RwSignal::new(Vec::<Candidate>::new());
    // Heading over the Candidates panel — "Today's Discoveries — {style}" on
    // first load, "Discoveries — {style}" after Surprise, cleared once the
    // user composes from the form themselves.
    let discovery_title = RwSignal::new(String::new());

    // Advanced spec editor state — the raw-JSON escape hatch ported from
    // the legacy page's `specBox`/"Advanced" details. Empty `spec_text`
    // means "let the style preset decide", exactly like the legacy page's
    // `if (specBox.value.trim()) { body.spec = ... }` guard.
    let spec_text = RwSignal::new(String::new());
    let spec_name = RwSignal::new(String::new());
    let saved_specs = RwSignal::new(Vec::<String>::new());
    let spec_note = RwSignal::new(String::new());

    let compose_with = move |req: ComposeRequest| {
        if composing.get_untracked() {
            return;
        }
        composing.set(true);
        error.set(None);
        spawn_local(async move {
            match api::compose(api::DEFAULT_BACKEND, &req).await {
                Ok(cs) => results.set(cs),
                Err(e) => error.set(Some(e)),
            }
            composing.set(false);
        });
    };

    // Shared request builder for all three compose entry points (plain
    // Compose, discovery-first landing, Surprise Me) — mirrors the legacy
    // page funneling every trigger through one `form.requestSubmit()` and
    // its single submit handler's spec-attachment logic. Returns `None`
    // (and reports the error) when `spec_text` holds unparseable JSON,
    // matching the legacy page's `JSON.parse` try/catch before it will
    // send anything.
    let build_request = move |style: String, seed: u64, n: u64| -> Option<ComposeRequest> {
        let raw_spec = spec_text.get_untracked();
        let spec = if raw_spec.trim().is_empty() {
            None
        } else {
            match serde_json::from_str::<serde_json::Value>(&raw_spec) {
                Ok(v) => Some(v),
                Err(e) => {
                    error.set(Some(format!("spec is not valid JSON: {e}")));
                    return None;
                }
            }
        };
        Some(ComposeRequest {
            valence: valence.get_untracked(),
            arousal: arousal.get_untracked(),
            energy: energy.get_untracked(),
            tonic: tonic.get_untracked(),
            style,
            bars: bars.get_untracked(),
            base_seed: seed,
            n_candidates: n,
            prompt: prompt.get_untracked(),
            spec,
            // Authored composes keep their exact spec/style premise — only
            // the Listen radio opts into server-side premise variation
            // (`api::compose_listen_piece`).
            vary_premise: false,
            renderer: muse.renderer_preference.get_untracked().map(str::to_string),
            use_motif_foundry: use_motif_foundry.get_untracked(),
            composition_lesson_id: {
                let lesson = composition_lesson_id.get_untracked();
                (!lesson.is_empty()).then_some(lesson)
            },
        })
    };

    let submit = move |_| {
        discovery_title.set(String::new());
        let seed = (js_sys::Math::random() * 900_000.0) as u64 + 1;
        if let Some(req) = build_request(style.get_untracked(), seed, n_candidates.get_untracked())
        {
            compose_with(req);
        }
    };

    // Discovery-first landing: auto-compose "Today's Discoveries" the same
    // way the legacy page's `composeToday()` did — same four identities all
    // day, different tomorrow, reproducible from the seed the form now
    // shows. Called directly in the component body (not `Effect::new`) —
    // Leptos component functions run exactly once per mount, and this has
    // no NodeRef/DOM dependency, so there's no ordering reason to defer it
    // into a reactive effect.
    {
        let today = palette::today_style();
        let seed = palette::today_seed();
        style.set(today.to_string());
        n_candidates.set(4);
        discovery_title.set(format!("Today's Discoveries — {today}"));
        if let Some(req) = build_request(today.to_string(), seed, 4) {
            compose_with(req);
        }
    }

    let surprise = move |_| {
        let today = palette::random_style();
        let seed = (js_sys::Math::random() * 900_000.0) as u64 + 1;
        style.set(today.to_string());
        n_candidates.set(4);
        discovery_title.set(format!("Discoveries — {today}"));
        if let Some(req) = build_request(today.to_string(), seed, 4) {
            compose_with(req);
        }
    };

    let refresh_saved_specs = move || {
        spawn_local(async move {
            if let Ok(names) = api::list_specs(api::DEFAULT_BACKEND).await {
                saved_specs.set(names);
            }
        });
    };
    refresh_saved_specs();

    let load_preset_spec = move |_| {
        let s = style.get_untracked();
        spawn_local(async move {
            match api::spec_preset(api::DEFAULT_BACKEND, &s).await {
                Ok(text) => {
                    spec_text.set(text);
                    spec_note.set(String::new());
                }
                Err(e) => spec_note.set(e),
            }
        });
    };

    let load_saved_spec = move |ev: leptos::ev::Event| {
        let name = event_target_value(&ev);
        if name.is_empty() {
            return;
        }
        spec_name.set(name.clone());
        spawn_local(async move {
            match api::load_named_spec(api::DEFAULT_BACKEND, &name).await {
                Ok(text) => {
                    spec_text.set(text);
                    spec_note.set(String::new());
                }
                Err(e) => spec_note.set(e),
            }
        });
    };

    let save_named_spec = move |_| {
        let name = spec_name.get_untracked();
        let text = spec_text.get_untracked();
        if name.trim().is_empty() || text.trim().is_empty() {
            spec_note.set("name + spec JSON required to save".to_string());
            return;
        }
        spawn_local(async move {
            match api::save_spec(api::DEFAULT_BACKEND, &name, &text).await {
                Ok(()) => {
                    spec_note.set(format!("saved \u{201c}{name}\u{201d}"));
                    refresh_saved_specs();
                }
                Err(e) => spec_note.set(e),
            }
        });
    };

    let listen_to = move |c: Candidate| {
        muse.current_style.set(c.style.clone());
        muse.kept.set(false);
        let source = crate::playback::PlaybackSource {
            rendition_id: Some(symthaea_muse_protocol::RenditionArtifactId(
                c.id.to_string(),
            )),
            audio_url: api::audio_url(api::DEFAULT_BACKEND, c.id),
            duration_hint_seconds: Some(c.duration_secs.max(0.0) as f64),
            advance_on_end: true,
        };
        muse.current.set(Some(c));
        // Through the reducer, not a direct `audio.set_src`/`.play()` —
        // otherwise the player bar's `load_epoch`-gated state (position,
        // duration, playing/paused) would silently stop updating: the
        // reducer would never learn a new source loaded, so every
        // subsequent browser event (`app.rs`) fails its `accepts()` check.
        muse.dispatch(crate::playback::PlaybackEvent::LoadRequested {
            source,
            autoplay: true,
        });
        navigate("/", Default::default());
    };

    view! {
        <div class="panel">
            <h2>"Create"</h2>
            <p class="muted">
                "Describe a piece; Muse composes several genuinely different candidates. \
                 Pick one to send it to Listen, or download its WAV/MIDI directly."
            </p>

            <form class="compose-form" on:submit=move |ev| { ev.prevent_default(); submit(()); }>
                <label>
                    "Constellation"
                    <select
                        prop:value=move || current_constellation.get().name().to_string()
                        on:change=move |ev| {
                            let picked = event_target_value(&ev);
                            // Jump to the chosen constellation's first
                            // routable style. `style` stays the single source
                            // of truth — the constellation shown above is
                            // derived from it, so Surprise Me and Today's
                            // Discoveries can move `style` freely without the
                            // two selects ever disagreeing.
                            if let Some(c) = Constellation::ALL
                                .into_iter()
                                .find(|c| c.name() == picked)
                                && let Some(first) = catalog::catalog_for_constellation(c)
                                    .find(|e| e.is_composable())
                                    .and_then(|e| e.composer_style)
                            {
                                style.set(first.to_string());
                            }
                        }
                    >
                        {Constellation::ALL.into_iter().map(|c| {
                            let routable = catalog::catalog_for_constellation(c)
                                .filter(|e| e.is_composable())
                                .count();
                            let name = c.name();
                            view! {
                                <option value=name disabled=routable == 0>
                                    {format!("{name} ({routable} of 8 playable)")}
                                </option>
                            }
                        }).collect_view()}
                    </select>
                </label>
                <label>
                    "Style"
                    <select
                        prop:value=move || style.get()
                        on:change=move |ev| style.set(event_target_value(&ev))
                    >
                        {move || catalog::catalog_for_constellation(current_constellation.get())
                            .map(|entry| {
                                // Research entries stay VISIBLE but unselectable.
                                // That is the catalog's own design rule: a style
                                // the engine cannot route should be honest about
                                // existing rather than hidden, and honest about
                                // not being playable rather than offered.
                                let composable = entry.is_composable();
                                let value = entry.composer_style.unwrap_or("").to_string();
                                let label = if composable {
                                    entry.display_name.to_string()
                                } else {
                                    format!("{} — {}", entry.display_name, entry.status.label())
                                };
                                view! {
                                    <option value=value disabled=!composable>{label}</option>
                                }
                            })
                            .collect_view()}
                    </select>
                </label>
                {move || selected_entry.get().map(|entry| {
                    let a = entry.anatomy();
                    view! {
                        <dl class="style-anatomy">
                            <dt>"Grammar"</dt><dd>{a.grammar}</dd>
                            <dt>"Phrase behavior"</dt><dd>{a.phrase_behavior}</dd>
                            <dt>"Harmonic system"</dt><dd>{a.harmonic_system}</dd>
                            <dt>"Rhythm"</dt><dd>{a.rhythm}</dd>
                            <dt>"Melodic language"</dt><dd>{a.melodic_language}</dd>
                            <dt>"Ensemble"</dt><dd>{a.ensemble}</dd>
                            <dt>"Performance dialect"</dt><dd>{a.performance_dialect}</dd>
                            <dt>"Production environment"</dt><dd>{a.production_environment}</dd>
                        </dl>
                    }
                })}
                <label>
                    "Key"
                    <select on:change=move |ev| {
                        if let Ok(v) = event_target_value(&ev).parse::<i32>() {
                            tonic.set(v);
                        }
                    }>
                        {NOTE_NAMES.iter().enumerate().map(|(i, name)| view! {
                            <option value=i.to_string() selected=i == 0>{*name}</option>
                        }).collect_view()}
                    </select>
                </label>
                <label>
                    {move || format!("Mood (valence: {:.2})", valence.get())}
                    <input type="range" min="-1" max="1" step="0.05"
                        prop:value=move || valence.get().to_string()
                        on:input=move |ev| {
                            if let Ok(v) = event_target_value(&ev).parse() { valence.set(v); }
                        }
                    />
                </label>
                <label>
                    {move || format!("Arousal ({:.2})", arousal.get())}
                    <input type="range" min="0" max="1" step="0.05"
                        prop:value=move || arousal.get().to_string()
                        on:input=move |ev| {
                            if let Ok(v) = event_target_value(&ev).parse() { arousal.set(v); }
                        }
                    />
                </label>
                <label>
                    {move || format!("Energy ({:.2})", energy.get())}
                    <input type="range" min="0" max="1" step="0.05"
                        prop:value=move || energy.get().to_string()
                        on:input=move |ev| {
                            if let Ok(v) = event_target_value(&ev).parse() { energy.set(v); }
                        }
                    />
                </label>
                <label>
                    {move || format!("Bars ({})", bars.get())}
                    <input type="range" min="2" max="16" step="1"
                        prop:value=move || bars.get().to_string()
                        on:input=move |ev| {
                            if let Ok(v) = event_target_value(&ev).parse() { bars.set(v); }
                        }
                    />
                </label>
                <label>
                    {move || format!("Candidates ({})", n_candidates.get())}
                    <input type="range" min="1" max="12" step="1"
                        prop:value=move || n_candidates.get().to_string()
                        on:input=move |ev| {
                            if let Ok(v) = event_target_value(&ev).parse() { n_candidates.set(v); }
                        }
                    />
                </label>
                <label class="prompt-label">
                    "Prompt (optional — ranks candidates by text\u{2194}audio similarity)"
                    <input type="text" placeholder="e.g. \"a long road home\""
                        prop:value=move || prompt.get()
                        on:input=move |ev| prompt.set(event_target_value(&ev))
                    />
                </label>
                <fieldset class="create-research-options">
                    <legend>"Musical research (optional)"</legend>
                    <label class="authorship-confirmation">
                        <input type="checkbox"
                            prop:checked=move || use_motif_foundry.get()
                            on:change=move |event| use_motif_foundry.set(event_target_checked(&event)) />
                        <span>"Use Motif Foundry for the primary identity"</span>
                    </label>
                    <label>
                        <span>"Composition lesson"</span>
                        <select
                            prop:value=move || composition_lesson_id.get()
                            on:change=move |event| composition_lesson_id.set(event_target_value(&event))
                        >
                            <option value="">"No authored lesson"</option>
                            <option value="etude:the-door-remembers">"The Door Remembers · altered return"</option>
                            <option value="etude:the-missing-thread">"The Missing Thread · withholding and restoration"</option>
                            <option value="etude:breath-between-stones">"Breath Between Stones · structural silence"</option>
                            <option value="etude:held-ground">"Held Ground · harmonic stasis"</option>
                        </select>
                    </label>
                    <small class="muted">
                        "Lessons apply provenance-clean strategies only. Muse never copies the etude's notes."
                    </small>
                </fieldset>

                <details class="spec-editor prompt-label">
                    <summary>
                        "Advanced: edit the style spec (motifs, progression, forms, textures, ensembles — complete control)"
                    </summary>
                    <div class="spec-editor-row">
                        <button type="button" on:click=load_preset_spec>"Load current style preset"</button>
                        <select on:change=load_saved_spec>
                            <option value="">"saved specs…"</option>
                            {move || saved_specs.get().into_iter().map(|n| view! {
                                <option value=n.clone()>{n.clone()}</option>
                            }).collect_view()}
                        </select>
                    </div>
                    <textarea
                        class="spec-editor-textarea"
                        placeholder="leave empty to use the style's built-in preset"
                        prop:value=move || spec_text.get()
                        on:input=move |ev| spec_text.set(event_target_value(&ev))
                    ></textarea>
                    <div class="spec-editor-row">
                        <input type="text" placeholder="spec name"
                            prop:value=move || spec_name.get()
                            on:input=move |ev| spec_name.set(event_target_value(&ev))
                        />
                        <button type="button" on:click=save_named_spec>"Save"</button>
                    </div>
                    {move || (!spec_note.get().is_empty()).then(|| view! {
                        <p class="muted small">{spec_note.get()}</p>
                    })}
                </details>

                <div class="compose-actions">
                    <button type="submit" disabled=move || composing.get()>
                        {move || if composing.get() { "Composing…" } else { "Compose" }}
                    </button>
                    <button type="button" disabled=move || composing.get() on:click=surprise>
                        "Surprise Me"
                    </button>
                </div>
            </form>

            {move || error.get().map(|e| view! {
                <p class="status-line error">{e}</p>
            })}
        </div>

        <div class="panel">
            <h2>
                {move || {
                    let title = discovery_title.get();
                    if title.is_empty() { "Candidates".to_string() } else { title }
                }}
            </h2>
            {move || {
                let rs = results.get();
                if rs.is_empty() {
                    view! { <p class="muted">"No candidates yet — compose above."</p> }.into_any()
                } else {
                    view! {
                        <div class="candidate-grid">
                            {rs.into_iter().map(|c| {
                                let c_listen = c.clone();
                                let c_keep = c.clone();
                                let diversity_plan = c.diversity_plan.clone();
                                let kept = RwSignal::new(false);
                                // `listen_to` is Clone (it only closes over
                                // Copy/Clone state), so each candidate card
                                // needs its own clone rather than moving the
                                // single outer binding — this whole `.map`
                                // body reruns on every `results` change, so
                                // a bare `move` here would only work once.
                                let listen_to = listen_to.clone();
                                view! {
                                    <div class="candidate-card">
                                        <h3>{c.title.clone()}</h3>
                                        <p class="muted">
                                            {c.card.map(|card| card.traits.join(" · ")).unwrap_or_default()}
                                        </p>
                                        <p class="why-this-piece">{c.why.join(" ")}</p>
                                        {diversity_plan.map(|plan| view! {
                                            <div class="candidate-plan" aria-label="Composition difference plan">
                                                <span>{plan.formal_topology}</span>
                                                <span>{plan.motif_development}</span>
                                                <span>{plan.orchestration}</span>
                                                <span>{plan.ending}</span>
                                            </div>
                                        })}
                                        <div class="candidate-actions">
                                            <button type="button" on:click=move |_| listen_to(c_listen.clone())>
                                                "Listen"
                                            </button>
                                            <button
                                                type="button"
                                                class="heart-btn"
                                                class:kept=move || kept.get()
                                                on:click=move |_| {
                                                    let id = c_keep.id;
                                                    spawn_local(async move {
                                                        if api::keep_piece(api::DEFAULT_BACKEND, id).await.is_ok() {
                                                            kept.set(true);
                                                        }
                                                    });
                                                }
                                            >
                                                {move || view! { <HeartIcon filled=kept.get() /> }}
                                                {move || if kept.get() { " kept" } else { " keep" }}
                                            </button>
                                            <a class="link-btn" href=api::midi_url(api::DEFAULT_BACKEND, c.id)
                                                download=format!("muse_seed{}.mid", c.seed)>
                                                "MIDI"
                                            </a>
                                        </div>
                                    </div>
                                }
                            }).collect_view()}
                        </div>
                    }.into_any()
                }
            }}
        </div>
    }
}
