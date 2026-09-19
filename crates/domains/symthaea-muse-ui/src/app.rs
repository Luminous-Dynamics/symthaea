// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Root application shell: global header + Listen/Create/Research routing.
//!
//! Matches the three design specs' top-level nav exactly (`Listen · Discover`,
//! `Create · Compose`, `Research · Understand` in the mockups) while the
//! Library and Atlas expose persisted/private workspace material around those
//! core modes. "Studio Mode" in those specs is the precision-editing surface
//! reached *from* a piece, not a fourth top-level tab — it isn't routed here
//! yet because its semantic authority/evidence program remains separate.

use leptos::prelude::*;
use leptos_router::components::{A, Route, Router, Routes};
use leptos_router::hooks::use_location;
use leptos_router::path;
use web_sys::HtmlAudioElement;

use crate::add_music_page::AddMusicPage;
use crate::atlas_page::AtlasPage;
use crate::browser_playback::dispatch_browser_event;
use crate::liked_page::LikedPage;
use crate::pages::{CreatePage, ListenPage, ResearchPage};
use crate::playback::{PlaybackEvent, PlaybackSubjectKind};
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
                // unmounts it — see state.rs's module doc. Browser media
                // callbacks first pass through `browser_playback`'s exact
                // epoch + phase admission boundary, then accepted events enter
                // the pure reducer through `MuseState::dispatch`.
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
                            dispatch_browser_event(
                                muse,
                                PlaybackEvent::MetadataLoaded {
                                    load_epoch,
                                    duration_seconds: audio.duration(),
                                },
                            );
                        }
                    }
                    on:play=move |_| {
                        let load_epoch = muse.playback.get_untracked().load_epoch;
                        dispatch_browser_event(
                            muse,
                            PlaybackEvent::PlaybackStarted { load_epoch },
                        );
                    }
                    on:pause=move |_| {
                        let load_epoch = muse.playback.get_untracked().load_epoch;
                        dispatch_browser_event(
                            muse,
                            PlaybackEvent::PlaybackPaused { load_epoch },
                        );
                    }
                    on:timeupdate=move |_| {
                        if let Some(audio) = muse.audio_ref.get_untracked() {
                            let audio: HtmlAudioElement = audio.into();
                            let load_epoch = muse.playback.get_untracked().load_epoch;
                            dispatch_browser_event(
                                muse,
                                PlaybackEvent::TimeAdvanced {
                                    load_epoch,
                                    seconds: audio.current_time(),
                                },
                            );
                        }
                    }
                    on:seeked=move |_| {
                        if let Some(audio) = muse.audio_ref.get_untracked() {
                            let audio: HtmlAudioElement = audio.into();
                            let load_epoch = muse.playback.get_untracked().load_epoch;
                            dispatch_browser_event(
                                muse,
                                PlaybackEvent::SeekCompleted {
                                    load_epoch,
                                    seconds: audio.current_time(),
                                },
                            );
                        }
                    }
                    on:ended=move |_| {
                        let load_epoch = muse.playback.get_untracked().load_epoch;
                        dispatch_browser_event(muse, PlaybackEvent::Ended { load_epoch });
                    }
                    on:error=move |_| {
                        let load_epoch = muse.playback.get_untracked().load_epoch;
                        dispatch_browser_event(
                            muse,
                            PlaybackEvent::PlaybackFailed {
                                load_epoch,
                                message: "the browser could not play this audio".to_string(),
                            },
                        );
                    }
                ></audio>
                <div class="page-body" style="position: relative;">
                    <ListenReviewInterlock muse=muse />
                    <Routes fallback=|| view! { <p>"Page not found"</p> }>
                        <Route path=path!("/") view=ListenPage />
                        <Route path=path!("/create") view=CreatePage />
                        <Route path=path!("/research") view=ResearchPage />
                        <Route path=path!("/library/import") view=AddMusicPage />
                        <Route path=path!("/library") view=LikedPage />
                        // Compatibility alias for existing bookmarks and the
                        // still-unmigrated links inside older UI surfaces.
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
fn ListenReviewInterlock(muse: MuseState) -> impl IntoView {
    let location = use_location();

    move || {
        if location.pathname.get() != "/" {
            return None;
        }
        let playback = muse.playback.get();
        let source = playback.source.as_ref()?;
        if source.presentation.kind != PlaybackSubjectKind::Review {
            return None;
        }

        let review_title = source.presentation.title.clone();
        let action_label = playback
            .return_bookmark
            .as_ref()
            .map(|bookmark| format!("Return to {}", bookmark.source.presentation.title))
            .unwrap_or_else(|| "End audition".to_string());

        Some(view! {
            <div
                role="status"
                aria-live="polite"
                style="position:absolute; inset:0; z-index:40; display:flex; align-items:flex-start; justify-content:center; padding:clamp(1rem,4vw,3rem); background:rgba(14,11,9,.88); backdrop-filter:blur(8px);"
            >
                <div class="panel" style="max-width:44rem; margin-top:clamp(1rem,8vh,5rem);">
                    <p class="muted small">"Temporary audition"</p>
                    <h2>{review_title}</h2>
                    <p>
                        "Listen is temporarily interlocked because its score, structure, section badges, downloads, and journey controls belong to the canonical candidate—not to the review audio currently playing."
                    </p>
                    <p class="muted">
                        "This prevents the review clock from being presented as evidence about a different piece."
                    </p>
                    <button
                        type="button"
                        on:click=move |_| {
                            muse.exit_review_audition();
                            if muse.playback.get_untracked().source.is_none()
                                && muse.current.get_untracked().is_none()
                            {
                                muse.next_piece(false);
                            }
                        }
                    >
                        {action_label}
                    </button>
                </div>
            </div>
        })
    }
}

#[component]
fn GlobalHeader(muse: MuseState) -> impl IntoView {
    let location = use_location();
    let is_active = move |path: &'static str| {
        let current = location.pathname.get();
        if path == "/" {
            current == "/"
        } else if path == "/library" {
            current.starts_with("/library") || current.starts_with("/liked")
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
                <strong>{move || {
                    muse.playback
                        .get()
                        .source
                        .map(|source| source.presentation.title)
                        .unwrap_or_else(|| "Awaiting a piece".to_string())
                }}</strong>
                {move || {
                    let playback = muse.playback.get();
                    let is_review = playback
                        .source
                        .as_ref()
                        .is_some_and(|source| source.presentation.kind == PlaybackSubjectKind::Review);
                    if !is_review {
                        return None;
                    }
                    let action_label = playback
                        .return_bookmark
                        .as_ref()
                        .map(|bookmark| format!("Return to {}", bookmark.source.presentation.title))
                        .unwrap_or_else(|| "End audition".to_string());
                    let aria_label = action_label.clone();
                    Some(view! {
                        <button
                            type="button"
                            class="link-btn"
                            title=aria_label.clone()
                            aria-label=aria_label
                            on:click=move |_| muse.exit_review_audition()
                        >
                            {action_label}
                        </button>
                    })
                }}
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
                    href="/library"
                    attr:class=move || if is_active("/library") { "active" } else { "" }
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
