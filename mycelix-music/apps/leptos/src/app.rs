// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;
use leptos_router::{
    components::{Route, Router, Routes, A},
    path,
};

use mycelix_leptos_core::{ConnectionStatusIndicator, HolochainProvider};

use crate::components::Nav;
use crate::components::Player;
use crate::pages::*;

/// Player state shared across the app via context.
#[derive(Clone, Debug)]
pub struct PlayerState {
    pub current_song: RwSignal<Option<crate::types::Song>>,
    pub is_playing: RwSignal<bool>,
    pub volume: RwSignal<f64>,
}

impl PlayerState {
    pub fn new() -> Self {
        Self {
            current_song: RwSignal::new(None),
            is_playing: RwSignal::new(false),
            volume: RwSignal::new(0.8),
        }
    }
}

#[component]
pub fn App() -> impl IntoView {
    let player = PlayerState::new();
    provide_context(player.clone());

    view! {
        <HolochainProvider>
            <Router>
                <Nav />
                <main class="main-content">
                    <Routes fallback=|| view! { <div class="page"><h1>"404 — Page not found"</h1></div> }>
                        <Route path=path!("/") view=HomePage />
                        <Route path=path!("/discover") view=DiscoverPage />
                        <Route path=path!("/artist") view=ArtistPage />
                        <Route path=path!("/dashboard") view=DashboardPage />
                        <Route path=path!("/upload") view=UploadPage />
                        <Route path=path!("/gallery") view=GalleryPage />
                    </Routes>
                </main>
                <Player />
                <footer class="footer">
                    <ConnectionStatusIndicator />
                    <span class="footer-text">"Mycelix Music — Zero-cost streaming on Holochain"</span>
                </footer>
            </Router>
        </HolochainProvider>
    }
}
