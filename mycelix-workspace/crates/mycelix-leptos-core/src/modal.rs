// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Modal and confirmation dialog components.

use leptos::prelude::*;

/// Modal size variant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ModalSize {
    Small,
    #[default]
    Medium,
    Large,
}

impl ModalSize {
    fn css_class(&self) -> &'static str {
        match self {
            ModalSize::Small => "modal-sm",
            ModalSize::Medium => "modal-md",
            ModalSize::Large => "modal-lg",
        }
    }
}

/// Modal dialog with backdrop.
///
/// Closes on backdrop click or Escape key. Uses display toggling
/// instead of `<Show>` to avoid FnOnce/Fn issues with children.
#[component]
pub fn Modal(
    open: ReadSignal<bool>,
    on_close: Callback<()>,
    #[prop(optional, into)] title: Option<String>,
    #[prop(optional)] size: ModalSize,
    children: Children,
) -> impl IntoView {
    let size_class = size.css_class();
    let rendered = children();

    view! {
        <div
            class="modal-backdrop"
            style=move || if open.get() { "display: flex" } else { "display: none" }
            on:click=move |_| on_close.run(())
            on:keydown=move |ev| {
                if ev.key() == "Escape" {
                    on_close.run(());
                }
            }
            tabindex="-1"
            role="dialog"
            aria-modal="true"
        >
            <div
                class=format!("modal-content {size_class}")
                on:click=move |ev| ev.stop_propagation()
            >
                {title.map(|t| view! {
                    <div class="modal-header">
                        <h2 class="modal-title">{t}</h2>
                    </div>
                })}
                <div class="modal-body">
                    {rendered}
                </div>
            </div>
        </div>
    }
}

/// Confirmation dialog.
#[component]
pub fn ConfirmDialog(
    open: ReadSignal<bool>,
    #[prop(into)] title: String,
    #[prop(into)] message: String,
    on_confirm: Callback<()>,
    on_cancel: Callback<()>,
    #[prop(optional, default = "Confirm")] confirm_label: &'static str,
    #[prop(optional)] danger: bool,
) -> impl IntoView {
    view! {
        <Modal open on_close=on_cancel title=title size=ModalSize::Small>
            <p class="confirm-message">{message}</p>
            <div class="modal-footer">
                <button class="btn btn-ghost" on:click=move |_| on_cancel.run(())>
                    "Cancel"
                </button>
                <button
                    class=if danger { "btn btn-danger" } else { "btn btn-primary" }
                    on:click=move |_| on_confirm.run(())
                >
                    {confirm_label}
                </button>
            </div>
        </Modal>
    }
}
