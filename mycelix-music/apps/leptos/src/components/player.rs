// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;
use crate::app::PlayerState;

/// Persistent audio player bar at the bottom of the screen.
/// Plays audio from IPFS gateway URLs and records plays via zome calls.
#[component]
pub fn Player() -> impl IntoView {
    let player = expect_context::<PlayerState>();
    let current = player.current_song;
    let is_playing = player.is_playing;
    let volume = player.volume;

    let toggle_play = move |_| {
        is_playing.update(|p| *p = !*p);
    };

    view! {
        <div class="player-bar">
            {move || {
                if let Some(song) = current.get() {
                    view! {
                        <div class="player-info">
                            <span class="player-title">{song.title.clone()}</span>
                            <span class="player-duration">{song.duration_display()}</span>
                        </div>
                        <div class="player-controls">
                            <button class="btn-player" on:click=toggle_play>
                                {move || if is_playing.get() { "⏸" } else { "▶" }}
                            </button>
                        </div>
                        <audio
                            prop:src=song.audio_url()
                            prop:volume=move || volume.get()
                            prop:autoplay=move || is_playing.get()
                        />
                    }.into_any()
                } else {
                    view! {
                        <div class="player-empty">
                            <span>"No song selected — browse "
                                <a href="/discover">"Discover"</a>
                            </span>
                        </div>
                    }.into_any()
                }
            }}
        </div>
    }
}
