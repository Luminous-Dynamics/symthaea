// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;
use crate::components::SongCard;
use crate::types::{mock_songs, Song};

const GENRES: &[&str] = &[
    "All", "Electronic", "Rock", "Indie", "Pop", "Ambient",
    "Hip-Hop", "Jazz", "Classical", "Folk",
];

#[component]
pub fn DiscoverPage() -> impl IntoView {
    let selected_genre = RwSignal::new("All".to_string());
    let songs = RwSignal::new(mock_songs());

    // TODO: Replace with use_zome_call for get_all_songs / get_songs_by_genre
    // when conductor is available. For now, filter mock data.
    let filtered_songs = move || {
        let genre = selected_genre.get();
        let all = songs.get();
        if genre == "All" {
            all
        } else {
            all.into_iter()
                .filter(|s| s.genres.iter().any(|g| g == &genre))
                .collect::<Vec<Song>>()
        }
    };

    let genre_chips = GENRES
        .iter()
        .map(|&g| {
            let g_str = g.to_string();
            let g_click = g_str.clone();
            view! {
                <button
                    class=move || {
                        if selected_genre.get() == g_str { "genre-chip active" } else { "genre-chip" }
                    }
                    on:click=move |_| selected_genre.set(g_click.clone())
                >
                    {g}
                </button>
            }
        })
        .collect_view();

    view! {
        <div class="page discover-page">
            <h1>"Discover"</h1>
            <div class="genre-filters">{genre_chips}</div>
            <div class="song-grid">
                {move || {
                    filtered_songs()
                        .into_iter()
                        .map(|song| view! { <SongCard song=song /> })
                        .collect_view()
                }}
            </div>
        </div>
    }
}
