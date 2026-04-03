// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;
use crate::app::PlayerState;

/// Toggleable queue panel showing the current playlist.
#[component]
pub fn QueuePanel() -> impl IntoView {
    let player = expect_context::<PlayerState>();

    let show = player.show_queue;

    view! {
        <Show when=move || show.get()>
            <QueuePanelInner />
        </Show>
    }
}

#[component]
fn QueuePanelInner() -> impl IntoView {
    let player = expect_context::<PlayerState>();

    let p1 = player.clone();
    let clear_queue = move |_| {
        p1.queue.set(Vec::new());
        p1.queue_index.set(None);
        p1.current_song.set(None);
        p1.is_playing.set(false);
        p1.show_queue.set(false);
    };

    let show_q = player.show_queue;
    let close = move |_: web_sys::MouseEvent| {
        show_q.set(false);
    };

    let close2 = move |_: web_sys::MouseEvent| {
        show_q.set(false);
    };

    // Escape key closes queue
    let on_keydown = move |ev: web_sys::KeyboardEvent| {
        if ev.key() == "Escape" {
            show_q.set(false);
        }
    };

    view! {
        <div class="queue-overlay" on:click=close on:keydown=on_keydown tabindex="-1"></div>
        <div class="queue-panel">
            <div class="queue-header">
                <h3>"Queue"</h3>
                <div class="queue-actions">
                    <button class="btn-sm" on:click=clear_queue>"Clear"</button>
                    <button class="btn-sm" on:click=close2>"\u{2715}"</button>
                </div>
            </div>
            <div class="queue-list">
                {move || {
                    let q = player.queue.get();
                    let current_idx = player.queue_index.get();
                    if q.is_empty() {
                        view! {
                            <div class="queue-empty">"Queue is empty"</div>
                        }.into_any()
                    } else {
                        q.into_iter()
                            .enumerate()
                            .map(|(i, song)| {
                                let is_current = current_idx == Some(i);
                                let class = if is_current {
                                    "queue-item current"
                                } else {
                                    "queue-item"
                                };
                                let player_a = player.clone();
                                let song_clone = song.clone();
                                let play_this = move |_| {
                                    player_a.play_song(song_clone.clone());
                                };
                                let player_b = player.clone();
                                let song_hash = song.song_hash.clone();
                                let remove = move |_| {
                                    player_b.queue.update(|q| {
                                        q.retain(|s| s.song_hash != song_hash);
                                    });
                                };
                                view! {
                                    <div class=class>
                                        <button class="queue-play" on:click=play_this>
                                            {if is_current { "\u{1f50a}" } else { "\u{25b6}" }}
                                        </button>
                                        <div class="queue-song-info">
                                            <span class="queue-title">{song.title.clone()}</span>
                                            <span class="queue-duration">{song.duration_display()}</span>
                                        </div>
                                        <button class="queue-remove" on:click=remove>"\u{2715}"</button>
                                    </div>
                                }
                            })
                            .collect_view()
                            .into_any()
                    }
                }}
            </div>
        </div>
    }
}
