// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Root application shell: global header + Listen/Create/Research routing.
//!
//! Matches the three design specs' top-level nav exactly (`Listen · Discover`,
//! `Create · Compose`, `Research · Understand` in the mockups) so the routes
//! map 1:1 onto `UI_mocks/MUSE_{LISTEN,STUDIO,RESEARCH}_MODE_DESIGN_SPEC.md`.
//! "Studio Mode" in those specs is the precision-editing surface reached
//! *from* a piece, not a fourth top-level tab — it isn't routed here yet
//! since it depends on backend capability (constrained alternative
//! generation, version graph) that doesn't exist yet.

use leptos::prelude::*;
use leptos_router::components::{A, Route, Router, Routes};
use leptos_router::hooks::use_location;
use leptos_router::path;
use web_sys::HtmlAudioElement;

use crate::atlas_page::AtlasPage;
use crate::liked_page::LikedPage;
use crate::pages::{CreatePage, ListenPage, ResearchPage};
use crate::playback::PlaybackEvent;
use crate::player_bar::PlayerBar;
use crate::state::MuseState;

#[component]
pub fn App() -> impl IntoView {
    let muse = MuseState::new();
    provide_context(muse);
    muse.load_style_families();

    view! {
        <Router>
            <div class="app-shell">
                <GlobalHeader muse=muse />
                // Outside <Routes> so navigating between modes never
                // unmounts it — see state.rs's module doc. Every handler
                // here just relays the browser's own event into the
                // playback reducer (`muse.dispatch`) tagged with the load
                // epoch it currently believes it's playing — the reducer
                // itself decides whether a late event from an already-
                // superseded piece should be ignored (see `playback.rs`).
                // Nothing here mutates transport state directly.
                <audio
                    node_ref=muse.audio_ref
                    // Required for the Web Audio analyser tap
                    // (`audio_reactivity::ensure_connected`, used by the
                    // player-bar wave and the Listen visualizer) to read
                    // real data — the backend serves audio from a
                    // different origin than a dev-server frontend would
                    // use, and without this attribute the browser
                    // silently taints the decoded media for Web Audio as
                    // a security measure: playback sounds completely
                    // normal, but `AnalyserNode.getByteFrequencyData`/
                    // `getByteTimeDomainData` read all-zero forever. The
                    // backend's CORS layer already allows any localhost
                    // origin (`muse_studio.rs::localhost_cors_layer`), so
                    // this just opts the client into using it.
                    crossorigin="anonymous"
                    on:loadedmetadata=move |_| {
                        if let Some(audio) = muse.audio_ref.get_untracked() {
                            let audio: HtmlAudioElement = audio.into();
                            let load_epoch = muse.playback.get_untracked().load_epoch;
                            muse.dispatch(PlaybackEvent::MetadataLoaded {
                                load_epoch,
                                duration_seconds: audio.duration(),
                            });
                        }
                    }
                    on:play=move |_| {
                        let load_epoch = muse.playback.get_untracked().load_epoch;
                        muse.dispatch(PlaybackEvent::PlaybackStarted { load_epoch });
                    }
                    on:pause=move |_| {
                        let load_epoch = muse.playback.get_untracked().load_epoch;
                        muse.dispatch(PlaybackEvent::PlaybackPaused { load_epoch });
                    }
                    on:timeupdate=move |_| {
                        if let Some(audio) = muse.audio_ref.get_untracked() {
                            let audio: HtmlAudioElement = audio.into();
                            let load_epoch = muse.playback.get_untracked().load_epoch;
                            muse.dispatch(PlaybackEvent::TimeAdvanced {
                                load_epoch,
                                seconds: audio.current_time(),
                            });
                        }
                    }
                    on:seeked=move |_| {
                        if let Some(audio) = muse.audio_ref.get_untracked() {
                            let audio: HtmlAudioElement = audio.into();
                            let load_epoch = muse.playback.get_untracked().load_epoch;
                            muse.dispatch(PlaybackEvent::SeekCompleted {
                                load_epoch,
                                seconds: audio.current_time(),
                            });
                        }
                    }
                    on:ended=move |_| {
                        let load_epoch = muse.playback.get_untracked().load_epoch;
                        muse.dispatch(PlaybackEvent::Ended { load_epoch });
                    }
                    on:error=move |_| {
                        let load_epoch = muse.playback.get_untracked().load_epoch;
                        muse.dispatch(PlaybackEvent::PlaybackFailed {
                            load_epoch,
                            message: "the browser could not play this audio".to_string(),
                        });
                    }
                ></audio>
                <div class="page-body">
                    <Routes fallback=|| view! { <p>"Page not found"</p> }>
                        <Route path=path!("/") view=ListenPage />
                        <Route path=path!("/create") view=CreatePage />
                        <Route path=path!("/research") view=ResearchPage />
                        <Route path=path!("/liked") view=LikedPage />
                        <Route path=path!("/atlas") view=AtlasPage />
                    </Routes>
                </div>
                <PlayerBar muse=muse />
            </div>
        </Router>
    }
}

#[component]
fn GlobalHeader(muse: MuseState) -> impl IntoView {
    let location = use_location();
    let is_active = move |path: &'static str| {
        let current = location.pathname.get();
        if path == "/" {
            current == "/"
        } else {
            current.starts_with(path)
        }
    };

    view! {
        <header class="global-header">
            <div class="brand-lockup">
                <span class="brand-mark" aria-hidden="true">"Φ"</span>
                <div>
                    <h1>"Muse"</h1>
                    <p>"by Symthaea"</p>
                </div>
            </div>
            <div class="header-now-playing">
                <span>"Now playing"</span>
                <strong>{move || muse.current.get().map(|piece| piece.title).unwrap_or_else(|| "Awaiting a piece".to_string())}</strong>
            </div>
            <nav class="mode-nav" aria-label="Muse modes">
                <A href="/" attr:class=move || if is_active("/") { "active" } else { "" }>
                    <span>"Listen"</span><small>"Immerse"</small>
                </A>
                <A
                    href="/create"
                    attr:class=move || if is_active("/create") { "active" } else { "" }
                >
                    <span>"Create"</span><small>"Compose"</small>
                </A>
                <A
                    href="/research"
                    attr:class=move || if is_active("/research") { "active" } else { "" }
                >
                    <span>"Research"</span><small>"Understand"</small>
                </A>
                <A
                    href="/liked"
                    attr:class=move || if is_active("/liked") { "active" } else { "" }
                >
                    <span>"Library"</span><small>"Keep"</small>
                </A>
                <A
                    href="/atlas"
                    attr:class=move || if is_active("/atlas") { "active" } else { "" }
                >
                    <span>"Atlas"</span><small>"Map"</small>
                </A>
            </nav>
        </header>
    }
}
