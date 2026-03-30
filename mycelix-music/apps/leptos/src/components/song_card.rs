// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;
use crate::app::PlayerState;
use crate::types::Song;

/// A card displaying a song with play button, title, artist, genres, and duration.
#[component]
pub fn SongCard(song: Song) -> impl IntoView {
    let player = expect_context::<PlayerState>();
    let song_clone = song.clone();

    let play = move |_| {
        player.current_song.set(Some(song_clone.clone()));
        player.is_playing.set(true);
    };

    let genre_tags = song
        .genres
        .iter()
        .map(|g| {
            let g = g.clone();
            view! { <span class="genre-tag">{g}</span> }
        })
        .collect_view();

    let strategy_label = match song.strategy_id.as_str() {
        "pay_per_stream" => "Pay Per Stream",
        "gift" => "Gift Economy",
        "patronage" => "Patronage",
        "premium" => "Premium",
        "freemium" => "Freemium",
        _ => "Standard",
    };

    view! {
        <div class="song-card">
            <div class="song-card-header">
                <button class="btn-play" on:click=play>"▶"</button>
                <div class="song-info">
                    <h3 class="song-title">{song.title.clone()}</h3>
                    <span class="song-duration">{song.duration_display()}</span>
                </div>
            </div>
            <div class="song-genres">{genre_tags}</div>
            <div class="song-strategy">
                <span class="strategy-badge">{strategy_label}</span>
            </div>
        </div>
    }
}
