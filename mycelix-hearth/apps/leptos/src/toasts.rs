// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Toast notification system.
//!
//! Small messages slide in from the bottom when family members do things.
//! Creates the feeling of a shared space where life happens around you.

use leptos::prelude::*;
use std::collections::VecDeque;

/// A single toast message.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct Toast {
    pub id: u64,
    pub message: String,
    /// "gratitude", "presence", "bond", "care", "decision"
    pub kind: String,
    /// Seconds remaining before auto-dismiss.
    pub ttl: f64,
}

/// Shared toast state.
#[derive(Clone)]
pub struct ToastState {
    pub toasts: RwSignal<VecDeque<Toast>>,
    next_id: RwSignal<u64>,
}

impl ToastState {
    /// Push a new toast. Auto-dismissed after `ttl` seconds.
    pub fn push(&self, message: impl Into<String>, kind: impl Into<String>) {
        let id = self.next_id.get_untracked();
        self.next_id.set(id + 1);
        self.toasts.update(|t| {
            t.push_back(Toast {
                id,
                message: message.into(),
                kind: kind.into(),
                ttl: 4.0,
            });
            // Cap at 5 visible toasts
            while t.len() > 5 {
                t.pop_front();
            }
        });

        // Auto-dismiss after 4 seconds
        let toasts = self.toasts;
        wasm_bindgen_futures::spawn_local(async move {
            gloo_timers::future::TimeoutFuture::new(4000).await;
            toasts.update(|t| {
                t.retain(|toast| toast.id != id);
            });
        });
    }
}

pub fn provide_toast_context() -> ToastState {
    let state = ToastState {
        toasts: RwSignal::new(VecDeque::new()),
        next_id: RwSignal::new(1),
    };
    provide_context(state.clone());
    state
}

pub fn use_toasts() -> ToastState {
    expect_context::<ToastState>()
}

/// The toast container — renders at the bottom of the screen.
#[component]
pub fn ToastContainer() -> impl IntoView {
    let state = use_toasts();

    view! {
        <div class="toast-container" role="status" aria-live="polite" aria-label="notifications">
            {move || {
                state.toasts.get().iter().map(|t| {
                    let msg = t.message.clone();
                    let kind = t.kind.clone();
                    view! {
                        <div class=format!("toast toast-{kind}")>
                            {msg}
                        </div>
                    }
                }).collect_view()
            }}
        </div>
    }
}
