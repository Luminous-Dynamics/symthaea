// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Application error boundary component.
//!
//! Wraps child components and catches rendering errors, displaying a styled
//! fallback UI with the error message and a retry button.

use leptos::prelude::*;

/// A styled error boundary that catches child component errors.
///
/// When an error occurs in a child component, this displays a card with
/// the error message and a retry button that clears the error state.
///
/// # Props
///
/// * `children` — The child components to wrap.
///
/// # Example
///
/// ```rust,no_run
/// use mycelix_leptos_core::AppErrorBoundary;
/// // view! {
/// //     <AppErrorBoundary>
/// //         <MyComponent />
/// //     </AppErrorBoundary>
/// // }
/// ```
#[component]
pub fn AppErrorBoundary(children: Children) -> impl IntoView {
    let fallback = move |errors: ArcRwSignal<Errors>| {
        let error_messages: Vec<String> =
            errors.read().iter().map(|(_, e)| e.to_string()).collect();

        let errors_clone = errors.clone();

        view! {
            <div
                style="padding: 16px; margin: 8px 0; border-radius: 8px; \
                       background-color: #fef2f2; border: 1px solid #fecaca; \
                       font-family: system-ui, sans-serif;"
            >
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    <span style="font-size: 1.25rem; color: #dc2626;">!</span>
                    <span style="font-weight: 600; color: #991b1b; font-size: 0.9rem;">
                        "Something went wrong"
                    </span>
                </div>
                <ul style="margin: 0 0 12px 0; padding-left: 20px; color: #7f1d1d; font-size: 0.85rem;">
                    {error_messages.into_iter().map(|msg| {
                        view! { <li>{msg}</li> }
                    }).collect::<Vec<_>>()}
                </ul>
                <button
                    on:click=move |_| {
                        // Collect error keys first, then remove them (Errors has no clear())
                        let keys: Vec<_> = errors_clone
                            .read()
                            .iter()
                            .map(|(k, _)| k.clone())
                            .collect();
                        errors_clone.update(|errs| {
                            for key in &keys {
                                errs.remove(key);
                            }
                        });
                    }
                    style="padding: 6px 16px; border-radius: 6px; border: 1px solid #fca5a5; \
                           background-color: white; color: #dc2626; font-size: 0.85rem; \
                           cursor: pointer; font-weight: 500; font-family: system-ui, sans-serif;"
                >
                    "Retry"
                </button>
            </div>
        }
    };

    view! {
        <ErrorBoundary fallback=fallback>
            {children()}
        </ErrorBoundary>
    }
}
