// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Onboarding tour — guided tooltip walkthrough for new users.

use leptos::prelude::*;

const TOUR_KEY: &str = "mycelix_pulse_tour_done";

struct TourStep {
    title: &'static str,
    body: &'static str,
    selector: &'static str,
    position: &'static str, // "bottom", "top", "right", "left"
}

const STEPS: &[TourStep] = &[
    TourStep {
        title: "Your Inbox",
        body: "All your encrypted messages appear here. Unread messages are bold with a cyan accent.",
        selector: ".page-inbox .email-list, .page-inbox .skeleton-list",
        position: "bottom",
    },
    TourStep {
        title: "Compose",
        body: "Send PQC-encrypted emails. Tab to autocomplete contacts, Ctrl+Enter to send.",
        selector: ".compose-btn, .bottom-nav-item:nth-child(2)",
        position: "right",
    },
    TourStep {
        title: "Semantic Search",
        body: "Search by meaning, not just keywords. Try operators like from: and is:unread.",
        selector: ".palette-trigger, .bottom-nav-item:nth-child(4)",
        position: "bottom",
    },
    TourStep {
        title: "Encryption",
        body: "Every message is end-to-end encrypted. The shield badge shows the crypto suite used.",
        selector: ".encryption-badge, .email-indicators",
        position: "left",
    },
    TourStep {
        title: "Trust Network",
        body: "Karma scores show sender reputation. Green = trusted, amber = new, red = flagged.",
        selector: ".karma-badge, .trust-stats",
        position: "bottom",
    },
];

#[component]
pub fn OnboardingTour() -> impl IntoView {
    let done = web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item(TOUR_KEY).ok().flatten())
        .is_some();

    let active = RwSignal::new(false);
    let step = RwSignal::new(0usize);

    // Expose a global function to start the tour
    provide_context(TourCtx { active, step });

    let total = STEPS.len();

    let current_step = move || STEPS.get(step.get());

    let on_next = move |_| {
        if step.get() + 1 >= total {
            // Tour complete
            active.set(false);
            if let Some(s) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
                let _ = s.set_item(TOUR_KEY, "1");
            }
        } else {
            step.update(|s| *s += 1);
        }
    };

    let on_skip = move |_| {
        active.set(false);
        if let Some(s) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
            let _ = s.set_item(TOUR_KEY, "1");
        }
    };

    // Get position of target element
    let tooltip_style = move || {
        let s = current_step()?;
        // Try to find the target element and position the tooltip near it
        // Use JS eval to get bounding rect (avoids needing DomRect web-sys feature)
        let js = format!("JSON.stringify(document.querySelector('{}')?.getBoundingClientRect())", s.selector.replace('\'', "\\'"));
        let rect: Option<serde_json::Value> = js_sys::eval(&js).ok()
            .and_then(|v| v.as_string())
            .and_then(|json| serde_json::from_str(&json).ok());

        if let Some(r) = rect {
            let get_f = |key: &str| r.get(key).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let (top, left) = match s.position {
                "bottom" => (get_f("bottom") + 12.0, get_f("left")),
                "top" => (get_f("top") - 160.0, get_f("left")),
                "right" => (get_f("top"), get_f("right") + 12.0),
                "left" => (get_f("top"), get_f("left") - 280.0),
                _ => (get_f("bottom") + 12.0, get_f("left")),
            };
            Some(format!("top:{top}px;left:{}px;", left.max(8.0)))
        } else {
            // Fallback: center of screen
            Some("top:40%;left:50%;transform:translate(-50%,-50%);".into())
        }
    };

    view! {
        {move || active.get().then(|| {
            let s = current_step();
            let style = tooltip_style().unwrap_or_default();
            let step_num = step.get() + 1;
            let title = s.map(|s| s.title).unwrap_or("Welcome");
            let body = s.map(|s| s.body).unwrap_or("");
            let is_last = step.get() + 1 >= total;

            view! {
                <div class="tour-overlay" on:click=on_skip>
                    <div class="tour-tooltip" style=style on:click=|ev: leptos::ev::MouseEvent| ev.stop_propagation()>
                        <div class="tour-header">
                            <span class="tour-step-num">{format!("{step_num}/{total}")}</span>
                            <button class="tour-skip" on:click=on_skip>"Skip"</button>
                        </div>
                        <h4 class="tour-title">{title}</h4>
                        <p class="tour-body">{body}</p>
                        <div class="tour-progress">
                            {(0..total).map(|i| {
                                view! { <span class=if i <= step.get() { "tour-dot active" } else { "tour-dot" } /> }
                            }).collect::<Vec<_>>()}
                        </div>
                        <button class="btn btn-primary btn-sm tour-next" on:click=on_next>
                            {if is_last { "Done" } else { "Next" }}
                        </button>
                    </div>
                </div>
            }
        })}
    }
}

#[derive(Clone, Copy)]
pub struct TourCtx {
    pub active: RwSignal<bool>,
    pub step: RwSignal<usize>,
}

pub fn start_tour() {
    if let Some(ctx) = use_context::<TourCtx>() {
        ctx.step.set(0);
        ctx.active.set(true);
    }
}
