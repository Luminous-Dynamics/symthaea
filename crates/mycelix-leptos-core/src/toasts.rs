// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Toast notification system.
//!
//! Quiet civic signals — a vote has been heard, a proposal submitted,
//! a delegation renewed. The commons speaks in whispers.

use leptos::prelude::*;
use std::collections::VecDeque;

/// Toast notification kind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToastKind {
    Success,
    Error,
    Warning,
    Info,
    /// Domain-specific kind (e.g. "governance", "finance").
    Custom(String),
}

impl ToastKind {
    /// CSS class suffix for styling.
    pub fn css_class(&self) -> String {
        match self {
            ToastKind::Success => "success".into(),
            ToastKind::Error => "error".into(),
            ToastKind::Warning => "warning".into(),
            ToastKind::Info => "info".into(),
            ToastKind::Custom(s) => s.clone(),
        }
    }
}

/// A single toast notification.
#[derive(Clone, Debug)]
pub struct Toast {
    pub id: u64,
    pub message: String,
    pub kind: ToastKind,
}

/// Reactive toast state shared via Leptos context.
#[derive(Clone)]
pub struct ToastState {
    pub toasts: RwSignal<VecDeque<Toast>>,
    next_id: RwSignal<u64>,
}

impl ToastState {
    /// Push a toast. Auto-removes after 4 seconds. Max 5 visible.
    pub fn push(&self, message: impl Into<String>, kind: ToastKind) {
        let id = self.next_id.get_untracked();
        self.next_id.set(id + 1);
        self.toasts.update(|t| {
            t.push_back(Toast {
                id,
                message: message.into(),
                kind,
            });
            while t.len() > 5 {
                t.pop_front();
            }
        });

        let toasts = self.toasts;
        wasm_bindgen_futures::spawn_local(async move {
            gloo_timers::future::TimeoutFuture::new(4000).await;
            toasts.update(|t| {
                t.retain(|toast| toast.id != id);
            });
        });
    }

    /// Convenience: push a success toast.
    pub fn success(&self, message: impl Into<String>) {
        self.push(message, ToastKind::Success);
    }

    /// Convenience: push an error toast.
    pub fn error(&self, message: impl Into<String>) {
        self.push(message, ToastKind::Error);
    }
}

/// Initialize toast context. Call once at app root.
pub fn provide_toast_context() -> ToastState {
    let state = ToastState {
        toasts: RwSignal::new(VecDeque::new()),
        next_id: RwSignal::new(1),
    };
    provide_context(state.clone());
    state
}

/// Retrieve toast state from context.
pub fn use_toasts() -> ToastState {
    expect_context::<ToastState>()
}

/// Toast container component. Place once in your app (typically after Router).
#[component]
pub fn ToastContainer() -> impl IntoView {
    let state = use_toasts();

    view! {
        <div class="toast-container" role="status" aria-live="polite" aria-label="notifications">
            {move || {
                state.toasts.get().iter().map(|t| {
                    let msg = t.message.clone();
                    let css = t.kind.css_class();
                    view! {
                        <div class=format!("toast toast-{css}")>
                            {msg}
                        </div>
                    }
                }).collect_view()
            }}
        </div>
    }
}
