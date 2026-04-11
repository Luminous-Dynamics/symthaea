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
    undo_actions: RwSignal<Vec<UndoAction>>,
}

/// Pending undo action — stored until toast expires.
#[derive(Clone)]
pub struct UndoAction {
    pub id: u32,
    pub callback: std::sync::Arc<dyn Fn() + Send + Sync>,
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

        let toasts = self.toasts;
        spawn_local(async move {
            gloo_timers::future::sleep(std::time::Duration::from_secs(4)).await;
            toasts.update(|t| t.retain(|toast| toast.id != id));
        });
    }

    /// Push a toast with an undo callback. The undo button is shown for 5 seconds.
    pub fn push_with_undo(&self, message: impl Into<String>, undo_fn: impl Fn() + Send + Sync + 'static) {
        let id = self.next_id.get_untracked();
        self.next_id.update(|n| *n += 1);
        let undo = std::sync::Arc::new(undo_fn);
        self.toasts.update(|t| {
            t.push_back(Toast {
                message: message.into(),
                category: "undo".into(),
                id,
            });
            while t.len() > 5 { t.pop_front(); }
        });

        // Store undo action
        self.undo_actions.update(|actions| {
            actions.push(UndoAction { id, callback: undo });
        });

        let toasts = self.toasts;
        let undo_actions = self.undo_actions;
        spawn_local(async move {
            gloo_timers::future::sleep(std::time::Duration::from_secs(5)).await;
            toasts.update(|t| t.retain(|toast| toast.id != id));
            undo_actions.update(|a| a.retain(|u| u.id != id));
        });
    }

    /// Execute undo for a given toast ID.
    pub fn execute_undo(&self, id: u32) {
        let action = self.undo_actions.get_untracked().iter().find(|a| a.id == id).cloned();
        if let Some(a) = action {
            (a.callback)();
            self.undo_actions.update(|actions| actions.retain(|u| u.id != id));
            self.toasts.update(|t| t.retain(|toast| toast.id != id));
            self.push("Undone", "success");
        }
    }
}

pub fn provide_toast_context() {
    provide_context(ToastCtx {
        toasts: RwSignal::new(VecDeque::new()),
        next_id: RwSignal::new(0),
        undo_actions: RwSignal::new(Vec::new()),
    });
}

pub fn use_toasts() -> ToastCtx {
    expect_context::<ToastCtx>()
}

#[component]
pub fn ToastContainer() -> impl IntoView {
    let ctx = use_toasts();
    let ctx_undo = ctx.clone();
    view! {
        <div class="toast-container" role="status" aria-live="polite">
            {move || ctx.toasts.get().iter().map(|t| {
                let cat = t.category.clone();
                let is_undo = cat == "undo";
                let toast_id = t.id;
                let ctx2 = ctx_undo.clone();
                view! {
                    <div class=format!("toast toast-{cat}")>
                        <span class="toast-message">{t.message.clone()}</span>
                        {is_undo.then(|| view! {
                            <button class="toast-undo-btn" on:click=move |_| ctx2.execute_undo(toast_id)>"Undo"</button>
                        })}
                    </div>
                }
            }).collect::<Vec<_>>()}
        </div>
    }
}
