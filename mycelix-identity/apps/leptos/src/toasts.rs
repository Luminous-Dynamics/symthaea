// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Toast notification system.

use leptos::prelude::*;
use std::collections::VecDeque;
use wasm_bindgen_futures::spawn_local;

#[derive(Clone, Debug)]
pub struct Toast {
    pub message: String,
    pub category: String,
    pub id: u32,
}

#[derive(Clone)]
pub struct ToastCtx {
    toasts: RwSignal<VecDeque<Toast>>,
    next_id: RwSignal<u32>,
}

impl ToastCtx {
    pub fn push(&self, message: impl Into<String>, category: impl Into<String>) {
        let id = self.next_id.get_untracked();
        self.next_id.update(|n| *n += 1);
        self.toasts.update(|t| {
            t.push_back(Toast {
                message: message.into(),
                category: category.into(),
                id,
            });
            while t.len() > 5 {
                t.pop_front();
            }
        });

        // Auto-dismiss after 4 seconds
        let toasts = self.toasts;
        spawn_local(async move {
            gloo_timers::future::sleep(std::time::Duration::from_secs(4)).await;
            toasts.update(|t| t.retain(|toast| toast.id != id));
        });
    }
}

pub fn provide_toast_context() {
    provide_context(ToastCtx {
        toasts: RwSignal::new(VecDeque::new()),
        next_id: RwSignal::new(0),
    });
}

pub fn use_toasts() -> ToastCtx {
    expect_context::<ToastCtx>()
}

#[component]
pub fn ToastContainer() -> impl IntoView {
    let ctx = use_toasts();
    view! {
        <div class="toast-container" role="status" aria-live="polite">
            {move || ctx.toasts.get().iter().map(|t| {
                let cat = t.category.clone();
                view! {
                    <div class=format!("toast toast-{cat}")>
                        {t.message.clone()}
                    </div>
                }
            }).collect::<Vec<_>>()}
        </div>
    }
}
