// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Keyboard shortcuts help overlay (#1).

use leptos::prelude::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

#[component]
pub fn KeyboardHelp() -> impl IntoView {
    let show = RwSignal::new(false);

    // Listen for ? key
    let closure = Closure::<dyn Fn(web_sys::KeyboardEvent)>::new(move |ev: web_sys::KeyboardEvent| {
        if let Some(target) = ev.target() {
            if let Ok(el) = target.dyn_into::<web_sys::HtmlElement>() {
                let tag = el.tag_name().to_lowercase();
                if tag == "input" || tag == "textarea" { return; }
            }
        }
        if ev.key() == "?" {
            show.update(|v| *v = !*v);
        }
        if ev.key() == "Escape" && show.get_untracked() {
            show.set(false);
        }
    });
    let window = web_sys::window().unwrap();
    window.add_event_listener_with_callback(
        "keydown", closure.as_ref().unchecked_ref()
    ).unwrap();
    closure.forget();

    view! {
        {move || show.get().then(|| view! {
            <div class="keyboard-overlay" on:click=move |_| show.set(false)>
                <div class="keyboard-modal" role="dialog" aria-label="Keyboard shortcuts" on:click=move |ev: leptos::ev::MouseEvent| ev.stop_propagation()>
                    <h2>"Keyboard Shortcuts"</h2>
                    <div class="shortcut-grid">
                        <div class="shortcut-section">
                            <h3>"Navigation"</h3>
                            <div class="shortcut-row"><kbd>"j"</kbd><span>"Next email"</span></div>
                            <div class="shortcut-row"><kbd>"k"</kbd><span>"Previous email"</span></div>
                            <div class="shortcut-row"><kbd>"n/p"</kbd><span>"Next/prev in thread"</span></div>
                            <div class="shortcut-row"><kbd>"Enter"</kbd><span>"Open email"</span></div>
                            <div class="shortcut-row"><kbd>"u"</kbd><span>"Return to inbox"</span></div>
                            <div class="shortcut-row"><kbd>"Esc"</kbd><span>"Back / deselect"</span></div>
                            <div class="shortcut-row"><kbd>"/"</kbd><span>"Search"</span></div>
                            <div class="shortcut-row"><kbd>"g"</kbd><span>"Go to inbox"</span></div>
                        </div>
                        <div class="shortcut-section">
                            <h3>"Actions"</h3>
                            <div class="shortcut-row"><kbd>"c"</kbd><span>"Compose new"</span></div>
                            <div class="shortcut-row"><kbd>"r"</kbd><span>"Reply"</span></div>
                            <div class="shortcut-row"><kbd>"f"</kbd><span>"Forward"</span></div>
                            <div class="shortcut-row"><kbd>"s"</kbd><span>"Cycle star type"</span></div>
                            <div class="shortcut-row"><kbd>"x"</kbd><span>"Select/deselect"</span></div>
                            <div class="shortcut-row"><kbd>"e"</kbd><span>"Archive"</span></div>
                            <div class="shortcut-row"><kbd>"#"</kbd><span>"Delete"</span></div>
                            <div class="shortcut-row"><kbd>"m"</kbd><span>"Mute thread"</span></div>
                            <div class="shortcut-row"><kbd>"l"</kbd><span>"Labels"</span></div>
                            <div class="shortcut-row"><kbd>"?"</kbd><span>"This help"</span></div>
                        </div>
                    </div>
                    <button class="btn btn-secondary" on:click=move |_| show.set(false)>"Close"</button>
                </div>
            </div>
        })}
    }
}
