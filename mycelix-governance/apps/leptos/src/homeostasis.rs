// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Civic Homeostasis detection.
//!
//! When all proposals are resolved, no votes are pending, and the treasury
//! is above its reserve threshold, the UI enters homeostasis. The chrome
//! fades. "The commons is at rest. Your stillness regenerates phi."

use leptos::prelude::*;

#[derive(Clone)]
#[allow(dead_code)]
pub struct HomeostasisState {
    pub pending_proposals: ReadSignal<u32>,
    pub pending_votes: ReadSignal<u32>,
    pub in_homeostasis: ReadSignal<bool>,
}

pub fn provide_homeostasis_context() -> HomeostasisState {
    let (pending_proposals, set_pending_proposals) = signal(0u32);
    let (pending_votes, set_pending_votes) = signal(0u32);
    let (in_homeostasis, set_in_homeostasis) = signal(false);

    Effect::new(move |_| {
        let proposals = pending_proposals.get();
        let votes = pending_votes.get();
        let is_home = proposals == 0 && votes == 0;
        set_in_homeostasis.set(is_home);
        set_css_var("--homeostasis", if is_home { "1" } else { "0" });
    });

    let state = HomeostasisState {
        pending_proposals,
        pending_votes,
        in_homeostasis,
    };

    provide_context(set_pending_proposals);
    provide_context(set_pending_votes);
    provide_context(state.clone());

    state
}

#[allow(dead_code)]
pub fn use_homeostasis() -> HomeostasisState {
    expect_context::<HomeostasisState>()
}

fn set_css_var(name: &str, value: &str) {
    use wasm_bindgen::JsCast;
    if let Some(window) = web_sys::window() {
        if let Some(document) = window.document() {
            if let Some(root) = document.document_element() {
                if let Some(el) = root.dyn_ref::<web_sys::HtmlElement>() {
                    let _ = el.style().set_property(name, value);
                }
            }
        }
    }
}
