// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Zome call button component.
//!
//! A button that executes an action with visual loading, success, and error
//! states. Designed for triggering Holochain zome calls but works with any
//! async action.

use leptos::prelude::*;

/// A button that shows loading/disabled states for zome call interactions.
///
/// When `loading` is true, the button displays a spinner and is disabled.
/// The `disabled` prop provides additional control for form validation, etc.
///
/// # Props
///
/// * `label` — Button text.
/// * `on_click` — Callback invoked when the button is clicked.
/// * `loading` — Optional signal indicating an operation is in progress.
/// * `disabled` — Optional signal to disable the button independently.
///
/// # Example
///
/// ```rust,no_run
/// use mycelix_leptos_core::ZomeCallButton;
/// // let (loading, _) = signal(false);
/// // view! {
/// //     <ZomeCallButton
/// //         label="Submit Proposal"
/// //         on_click=move || { /* dispatch zome call */ }
/// //         loading=loading
/// //     />
/// // }
/// ```
#[component]
pub fn ZomeCallButton(
    label: &'static str,
    on_click: impl Fn() + 'static,
    #[prop(optional)] loading: Option<ReadSignal<bool>>,
    #[prop(optional)] disabled: Option<ReadSignal<bool>>,
) -> impl IntoView {
    let is_loading = move || loading.map_or(false, |s| s.get());
    let is_disabled = move || disabled.map_or(false, |s| s.get()) || is_loading();

    let button_style = move || {
        let base = "padding: 8px 20px; border-radius: 6px; font-size: 0.875rem; \
                    font-weight: 500; font-family: system-ui, sans-serif; \
                    border: 1px solid transparent; cursor: pointer; \
                    display: inline-flex; align-items: center; gap: 8px; \
                    transition: opacity 0.15s ease;";
        if is_disabled() {
            format!(
                "{base} background-color: #9ca3af; color: white; \
                 cursor: not-allowed; opacity: 0.7;"
            )
        } else {
            format!(
                "{base} background-color: #2563eb; color: white; \
                 border-color: #1d4ed8;"
            )
        }
    };

    let on_click = std::rc::Rc::new(on_click);
    let on_click_clone = on_click.clone();

    view! {
        <button
            disabled=is_disabled
            style=button_style
            on:click=move |_| {
                if !is_disabled() {
                    on_click_clone();
                }
            }
        >
            {move || if is_loading() {
                view! {
                    <span
                        style="display: inline-block; width: 14px; height: 14px; \
                               border: 2px solid rgba(255,255,255,0.3); \
                               border-top-color: white; border-radius: 50%; \
                               animation: mycelix-spin 0.6s linear infinite;"
                    ></span>
                }.into_any()
            } else {
                view! { <span></span> }.into_any()
            }}
            <span>{label}</span>
        </button>
    }
}
