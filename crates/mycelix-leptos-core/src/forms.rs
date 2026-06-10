// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Shared form components for Mycelix cluster frontends.
//!
//! Uses the `.form-input`, `.form-textarea`, `.form-select`, `.form-label`
//! classes defined in base.css.

use leptos::prelude::*;

/// A labeled form field with optional error message.
#[component]
pub fn FormField(
    #[prop(into)] label: String,
    #[prop(optional, into)] error: Option<String>,
    #[prop(optional)] required: bool,
    children: Children,
) -> impl IntoView {
    view! {
        <div class="form-field">
            <label class="form-label">
                {label}
                {required.then(|| view! { <span class="form-required">" *"</span> })}
            </label>
            {children()}
            {error.map(|e| view! { <span class="form-error">{e}</span> })}
        </div>
    }
}

/// Option for a Select component.
#[derive(Clone, Debug)]
pub struct SelectOption {
    pub value: String,
    pub label: String,
}

/// Text input.
#[component]
pub fn TextInput(
    value: ReadSignal<String>,
    on_change: Callback<String>,
    #[prop(optional)] placeholder: &'static str,
    #[prop(optional, default = "text")] input_type: &'static str,
    #[prop(optional)] disabled: bool,
    #[prop(optional)] required: bool,
    #[prop(optional)] invalid: bool,
) -> impl IntoView {
    view! {
        <input
            class="form-input"
            type=input_type
            placeholder=placeholder
            disabled=disabled
            aria-required=required.to_string()
            aria-invalid=invalid.to_string()
            prop:value=move || value.get()
            on:input=move |ev| {
                on_change.run(event_target_value(&ev));
            }
        />
    }
}

/// Textarea.
#[component]
pub fn TextArea(
    value: ReadSignal<String>,
    on_change: Callback<String>,
    #[prop(optional, default = 4)] rows: u32,
    #[prop(optional)] placeholder: &'static str,
    #[prop(optional)] required: bool,
) -> impl IntoView {
    view! {
        <textarea
            class="form-textarea"
            rows=rows
            placeholder=placeholder
            aria-required=required.to_string()
            prop:value=move || value.get()
            on:input=move |ev| {
                on_change.run(event_target_value(&ev));
            }
        ></textarea>
    }
}

/// Select dropdown.
#[component]
pub fn Select(
    value: ReadSignal<String>,
    on_change: Callback<String>,
    #[prop(into)] options: Vec<SelectOption>,
) -> impl IntoView {
    view! {
        <select
            class="form-select"
            prop:value=move || value.get()
            on:change=move |ev| {
                on_change.run(event_target_value(&ev));
            }
        >
            {options.into_iter().map(|opt| {
                let val = opt.value.clone();
                view! {
                    <option value=val>{opt.label}</option>
                }
            }).collect_view()}
        </select>
    }
}

/// Checkbox with label.
#[component]
pub fn Checkbox(
    checked: ReadSignal<bool>,
    on_toggle: Callback<bool>,
    #[prop(into)] label: String,
) -> impl IntoView {
    view! {
        <label class="form-checkbox">
            <input
                type="checkbox"
                prop:checked=move || checked.get()
                on:change=move |ev| {
                    on_toggle.run(event_target_checked(&ev));
                }
            />
            <span>{label}</span>
        </label>
    }
}

fn event_target_checked(ev: &leptos::ev::Event) -> bool {
    use wasm_bindgen::JsCast;
    ev.target()
        .and_then(|t| t.dyn_into::<web_sys::HtmlInputElement>().ok())
        .map(|el| el.checked())
        .unwrap_or(false)
}
