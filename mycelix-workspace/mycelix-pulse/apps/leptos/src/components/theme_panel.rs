// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Theme customization panel — named themes, accent color picker, density slider, custom CSS.
//! Reuses Hearth's 10 named themes adapted for the Pulse workspace.

use leptos::prelude::*;
use wasm_bindgen::JsCast;
use crate::theme::use_theme;
use crate::preferences::{clamp_font_scale, use_preferences, MAX_FONT_SCALE, MIN_FONT_SCALE};

const ACCENT_COLORS: &[(&str, &str)] = &[
    ("Cyan", "#06D6C8"),
    ("Purple", "#8b7ec8"),
    ("Amber", "#f59e0b"),
    ("Green", "#4ade80"),
    ("Pink", "#ec4899"),
    ("Blue", "#3b82f6"),
    ("Red", "#ef4444"),
    ("Orange", "#f97316"),
    ("Teal", "#14b8a6"),
    ("Indigo", "#6366f1"),
];

const NAMED_THEMES: &[(&str, &str, &str, &str)] = &[
    // (name, bg, surface, accent)
    ("Ember", "#1a0a0a", "#2a1010", "#ef4444"),
    ("Tide", "#0a1420", "#0f1e30", "#3b82f6"),
    ("Canopy", "#0a1a0f", "#102818", "#4ade80"),
    ("Dusk", "#120a1a", "#1e1030", "#a78bfa"),
    ("Stone", "#141414", "#1e1e1e", "#9ca3af"),
    ("Meadow", "#f0f4e8", "#fafdf2", "#65a30d"),
    ("Cosmos", "#050510", "#0a0a1a", "#818cf8"),
    ("Clay", "#1a1410", "#2a2018", "#d4a574"),
    ("Frost", "#f0f4f8", "#ffffff", "#06b6d4"),
    ("Circuit", "#0a0c10", "#101418", "#06D6C8"),
];

const CUSTOM_CSS_KEY: &str = "mycelix_pulse_custom_css";
const ACCENT_KEY: &str = "mycelix_pulse_accent";
const THEME_NAME_KEY: &str = "mycelix_pulse_theme_name";

#[component]
pub fn ThemePanel(show: RwSignal<bool>) -> impl IntoView {
    let theme = use_theme();
    let prefs = use_preferences();
    let custom_css = RwSignal::new(load_custom_css());
    let current_accent = RwSignal::new(load_accent());
    let current_theme_name = RwSignal::new(load_theme_name());

    // Apply accent color
    let apply_accent = move |color: &str| {
        let _ = js_sys::eval(&format!(
            "document.documentElement.style.setProperty('--primary','{}');document.documentElement.style.setProperty('--primary-hover','{}cc')",
            color, color
        ));
        save_string(ACCENT_KEY, color);
        current_accent.set(color.to_string());
    };

    // Apply named theme
    let apply_theme_name = move |name: &str, bg: &str, surface: &str, accent: &str| {
        let _ = js_sys::eval(&format!(
            "const s=document.documentElement.style;s.setProperty('--bg','{}');s.setProperty('--surface','{}');s.setProperty('--primary','{}');s.setProperty('--primary-hover','{}cc')",
            bg, surface, accent, accent
        ));
        save_string(THEME_NAME_KEY, name);
        current_theme_name.set(name.to_string());
    };

    // Apply custom CSS
    let apply_custom_css = move || {
        let css = custom_css.get_untracked();
        save_string(CUSTOM_CSS_KEY, &css);
        let _ = js_sys::eval(&format!(
            "let el=document.getElementById('custom-user-css');if(!el){{el=document.createElement('style');el.id='custom-user-css';document.head.appendChild(el)}};el.textContent=`{}`",
            css.replace('`', "\\`")
        ));
    };

    view! {
        <div class="theme-panel-overlay" style=move || if show.get() { "" } else { "display:none" }
             on:click=move |_| show.set(false)>
            <div class="theme-panel" on:click=move |ev: leptos::ev::MouseEvent| ev.stop_propagation()>
                <div class="theme-panel-header">
                    <h2>"Appearance"</h2>
                    <button class="btn btn-icon" on:click=move |_| show.set(false)>"\u{2715}"</button>
                </div>

                // Dark/Light toggle
                <section class="theme-section">
                    <h3>"Mode"</h3>
                    <div class="theme-mode-toggle">
                        <button class=move || if theme.current.get() == mail_leptos_types::Theme::Dark { "mode-btn active" } else { "mode-btn" }
                                on:click=move |_| theme.current.set(mail_leptos_types::Theme::Dark)>
                            "\u{1F319} Dark"
                        </button>
                        <button class=move || if theme.current.get() == mail_leptos_types::Theme::Light { "mode-btn active" } else { "mode-btn" }
                                on:click=move |_| theme.current.set(mail_leptos_types::Theme::Light)>
                            "\u{2600} Light"
                        </button>
                    </div>
                </section>

                // Named themes
                <section class="theme-section">
                    <h3>"Theme"</h3>
                    <div class="theme-grid">
                        {NAMED_THEMES.iter().map(|(name, bg, surface, accent)| {
                            let n = *name; let b = *bg; let s = *surface; let a = *accent;
                            let is_active = move || current_theme_name.get() == n;
                            view! {
                                <button
                                    class=move || if is_active() { "theme-swatch active" } else { "theme-swatch" }
                                    style=format!("background: {}; border-color: {}", bg, accent)
                                    on:click=move |_| apply_theme_name(n, b, s, a)
                                    title=n
                                >
                                    <span class="swatch-dot" style=format!("background: {}", accent) />
                                    <span class="swatch-name">{n}</span>
                                </button>
                            }
                        }).collect::<Vec<_>>()}
                    </div>
                </section>

                // Accent color picker
                <section class="theme-section">
                    <h3>"Accent Color"</h3>
                    <div class="accent-grid">
                        {ACCENT_COLORS.iter().map(|(name, color)| {
                            let c = *color; let n = *name;
                            let is_active = move || current_accent.get() == c;
                            view! {
                                <button
                                    class=move || if is_active() { "accent-swatch active" } else { "accent-swatch" }
                                    style=format!("background: {}", color)
                                    on:click=move |_| apply_accent(c)
                                    title=n
                                />
                            }
                        }).collect::<Vec<_>>()}
                    </div>
                </section>

                // Density
                <section class="theme-section">
                    <h3>"Density"</h3>
                    <div class="density-buttons">
                        <button class=move || if theme.density.get() == mail_leptos_types::Density::Compact { "density-btn active" } else { "density-btn" }
                                on:click=move |_| theme.density.set(mail_leptos_types::Density::Compact)>"Compact"</button>
                        <button class=move || if theme.density.get() == mail_leptos_types::Density::Default { "density-btn active" } else { "density-btn" }
                                on:click=move |_| theme.density.set(mail_leptos_types::Density::Default)>"Default"</button>
                        <button class=move || if theme.density.get() == mail_leptos_types::Density::Comfortable { "density-btn active" } else { "density-btn" }
                                on:click=move |_| theme.density.set(mail_leptos_types::Density::Comfortable)>"Comfortable"</button>
                    </div>
                </section>

                // Font scale slider
                <section class="theme-section">
                    <h3>"Font Size: " {move || format!("{}%", prefs.get().font_scale)}</h3>
                    <input type="range" min=MIN_FONT_SCALE.to_string() max=MAX_FONT_SCALE.to_string() step="5"
                        class="font-slider"
                        prop:value=move || prefs.get().font_scale.to_string()
                        on:input=move |ev| {
                            if let Ok(v) = ev.target().and_then(|t| t.dyn_into::<web_sys::HtmlInputElement>().ok()).map(|el| el.value()).unwrap_or_default().parse::<u32>() {
                                prefs.update(|p| p.font_scale = clamp_font_scale(v));
                            }
                        }
                    />
                </section>

                // Custom CSS
                <section class="theme-section">
                    <h3>"Custom CSS"</h3>
                    <textarea
                        class="custom-css-input"
                        rows="6"
                        placeholder="/* Your custom CSS here */\n.email-card { border-radius: 0; }"
                        prop:value=move || custom_css.get()
                        on:input=move |ev| {
                            let val = ev.target().and_then(|t| t.dyn_into::<web_sys::HtmlTextAreaElement>().ok()).map(|el| el.value()).unwrap_or_default();
                            custom_css.set(val);
                        }
                    />
                    <button class="btn btn-sm btn-primary" on:click=move |_| apply_custom_css()>"Apply CSS"</button>
                </section>
            </div>
        </div>
    }
}

fn load_custom_css() -> String {
    web_sys::window().and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item(CUSTOM_CSS_KEY).ok().flatten())
        .unwrap_or_default()
}

fn load_accent() -> String {
    web_sys::window().and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item(ACCENT_KEY).ok().flatten())
        .unwrap_or_else(|| "#06D6C8".into())
}

fn load_theme_name() -> String {
    web_sys::window().and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item(THEME_NAME_KEY).ok().flatten())
        .unwrap_or_else(|| "Circuit".into())
}

fn save_string(key: &str, value: &str) {
    if let Some(s) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
        let _ = s.set_item(key, value);
    }
}
