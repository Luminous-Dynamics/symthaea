// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pull-to-refresh — native-feeling mobile gesture.

use leptos::prelude::*;

const THRESHOLD: f64 = 60.0;
const MAX_PULL: f64 = 120.0;

#[component]
pub fn PullToRefresh(
    on_refresh: impl Fn() + 'static + Clone + Send,
    children: Children,
) -> impl IntoView {
    let pull_y = RwSignal::new(0.0f64);
    let start_y = RwSignal::new(0.0f64);
    let pulling = RwSignal::new(false);
    let refreshing = RwSignal::new(false);

    let on_refresh2 = on_refresh.clone();

    let on_touch_start = move |ev: web_sys::TouchEvent| {
        if refreshing.get() { return; }
        if let Some(t) = ev.touches().get(0) {
            start_y.set(t.client_y() as f64);
            pulling.set(true);
        }
    };

    let on_touch_move = move |ev: web_sys::TouchEvent| {
        if !pulling.get() || refreshing.get() { return; }
        if let Some(t) = ev.touches().get(0) {
            let dy = (t.client_y() as f64 - start_y.get()).max(0.0).min(MAX_PULL);
            // Only activate if page is scrolled to top
            let at_top = js_sys::eval("document.querySelector('.main-content')?.scrollTop <= 0")
                .ok()
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            if at_top && dy > 10.0 {
                ev.prevent_default();
                pull_y.set(dy);
            }
        }
    };

    let on_touch_end = move |_: web_sys::TouchEvent| {
        if !pulling.get() { return; }
        pulling.set(false);
        if pull_y.get() >= THRESHOLD {
            refreshing.set(true);
            on_refresh2();
            // Reset after 1.5s
            let refreshing_reset = refreshing;
            let pull_reset = pull_y;
            wasm_bindgen_futures::spawn_local(async move {
                gloo_timers::future::sleep(std::time::Duration::from_millis(1500)).await;
                refreshing_reset.set(false);
                pull_reset.set(0.0);
            });
        } else {
            pull_y.set(0.0);
        }
    };

    view! {
        <div class="pull-refresh-container"
             on:touchstart=on_touch_start
             on:touchmove=on_touch_move
             on:touchend=on_touch_end
             style="overscroll-behavior-y: contain;">
            // Pull indicator
            <div class="pull-indicator"
                 style=move || {
                     let y = pull_y.get();
                     if y > 0.0 || refreshing.get() {
                         let h = if refreshing.get() { 40.0 } else { y.min(60.0) };
                         format!("height:{h}px;opacity:{};", (y / THRESHOLD).min(1.0))
                     } else {
                         "height:0;opacity:0;".into()
                     }
                 }>
                <span class="pull-icon" style=move || {
                    if refreshing.get() {
                        "animation: spin 0.8s linear infinite;".into()
                    } else {
                        let rotation = (pull_y.get() / THRESHOLD * 180.0).min(180.0);
                        format!("transform:rotate({rotation}deg);")
                    }
                }>
                    {move || if refreshing.get() { "\u{21BB}" } else if pull_y.get() >= THRESHOLD { "\u{2191}" } else { "\u{2193}" }}
                </span>
                <span class="pull-text">
                    {move || if refreshing.get() { "Refreshing..." } else if pull_y.get() >= THRESHOLD { "Release to refresh" } else { "Pull to refresh" }}
                </span>
            </div>
            {children()}
        </div>
    }
}
