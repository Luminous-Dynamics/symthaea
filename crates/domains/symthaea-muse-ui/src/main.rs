// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Muse web UI — Listen / Create / Research modes over the `muse_studio`
//! HTTP API. Routing scaffold per MUSE_STUDIO/LISTEN/RESEARCH_MODE_DESIGN_SPEC.md
//! (`symthaea-muse/UI_mocks/`); each mode starts as a stub page and grows
//! toward its spec's P0 acceptance criteria one slice at a time — Listen
//! Mode first (see MASTER_ROADMAP.md's Muse UI row for the agreed order).

mod add_music_page;
mod api;
mod app;
mod atlas_page;
mod audio_reactivity;
mod create_page;
mod evidence_view;
mod harmony_view;
mod icons;
mod journey;
mod liked_page;
mod motifs_view;
mod orchestration_view;
mod pages;
mod palette;
mod playback;
mod player_bar;
mod score_view;
mod state;

fn main() {
    leptos::mount::mount_to_body(app::App);
}
