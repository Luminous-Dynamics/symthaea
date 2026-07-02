// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Settings page for Prism.

use leptos::prelude::*;
use prism_search::SearchEngine;
use wasm_bindgen::JsCast;

const THEMES: &[(&str, &str, &str, &str)] = &[
    (
        "abyss",
        "Abyssal Bioluminescence",
        "#4aecd8",
        "Deep ocean, life in darkness",
    ),
    (
        "parchment",
        "Mycelial Parchment",
        "#8fbc8f",
        "Living manuscript, organic warmth",
    ),
    (
        "prism",
        "Hyperdimensional Prism",
        "#a78bfa",
        "Crystalline, scientific precision",
    ),
    (
        "solar",
        "Solar Resonance",
        "#f59e0b",
        "Warm ember, sunset glow",
    ),
    (
        "verdant",
        "Verdant Canopy",
        "#22c55e",
        "Forest canopy, growth",
    ),
    (
        "arctic",
        "Arctic Aurora",
        "#38bdf8",
        "Ice blue, aurora shimmer",
    ),
    (
        "obsidian",
        "Obsidian Forge",
        "#ef4444",
        "Volcanic, industrial power",
    ),
    (
        "lavender",
        "Lavender Twilight",
        "#c084fc",
        "Dusk, contemplative softness",
    ),
];

#[component]
pub fn SettingsPage() -> impl IntoView {
    let engine = expect_context::<StoredValue<Option<SearchEngine>>>();
    let claim_count = engine.with_value(|e| e.as_ref().map(|s| s.claim_count()).unwrap_or(0));

    // Current theme
    let (current_theme, set_current_theme) =
        signal(crate::persistence::load::<String>("theme").unwrap_or_else(|| "abyss".to_string()));

    // Font size
    let (font_size, set_font_size) =
        signal(crate::persistence::load::<f64>("font-size").unwrap_or(17.0));

    // Line height
    let (line_height, set_line_height) =
        signal(crate::persistence::load::<f64>("line-height").unwrap_or(1.7));

    // Content width
    let (content_width, set_content_width) =
        signal(crate::persistence::load::<f64>("content-width").unwrap_or(680.0));

    // Toggle settings
    let (show_vector_viz, set_show_vector_viz) =
        signal(crate::persistence::load::<bool>("vector-viz").unwrap_or(true));
    let (show_overlay, set_show_overlay) =
        signal(crate::persistence::load::<bool>("overlay").unwrap_or(true));
    let (show_scores, set_show_scores) =
        signal(crate::persistence::load::<bool>("show-scores").unwrap_or(true));

    // Apply font size to CSS
    Effect::new(move |_| {
        let fs = font_size.get();
        let lh = line_height.get();
        let cw = content_width.get();
        if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
            if let Some(el) = doc.document_element() {
                if let Ok(html_el) = el.dyn_into::<web_sys::HtmlElement>() {
                    let style = html_el.style();
                    let _ = style.set_property("--font-size", &format!("{}px", fs));
                    let _ = style.set_property("--line-height", &format!("{}", lh));
                    let _ = style.set_property("--content-width", &format!("{}px", cw));
                }
            }
        }
        crate::persistence::save("font-size", &fs);
        crate::persistence::save("line-height", &lh);
        crate::persistence::save("content-width", &cw);
    });

    // Theme switch handler
    let apply_theme = move |theme: &str| {
        let theme = theme.to_string();
        set_current_theme.set(theme.clone());
        if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
            if let Some(html) = doc.document_element() {
                let _ = html.set_attribute("data-theme", &theme);
            }
        }
        crate::persistence::save("theme", &theme);
    };

    view! {
        <div class="reader-content settings-page">
            <h1>"Settings"</h1>

            // ── Appearance ──
            <section class="settings-section">
                <h2>"Appearance"</h2>

                <div class="setting-group">
                    <label class="setting-label">"Theme"</label>
                    <div class="theme-grid">
                        {THEMES.iter().map(|(id, name, color, desc)| {
                            let id = id.to_string();
                            let name = name.to_string();
                            let color = color.to_string();
                            let desc = desc.to_string();
                            let id_click = id.clone();
                            let id_check = id.clone();
                            view! {
                                <div
                                    class=move || if current_theme.get() == id_check { "theme-card active" } else { "theme-card" }
                                    on:click=move |_| apply_theme(&id_click)
                                >
                                    <div class="theme-swatch" style:background=color.clone()></div>
                                    <div class="theme-name">{name.clone()}</div>
                                    <div class="theme-desc">{desc.clone()}</div>
                                </div>
                            }
                        }).collect_view()}
                    </div>
                </div>

                <div class="setting-group">
                    <label class="setting-label">
                        "Font Size: " <span class="setting-value">{move || format!("{}px", font_size.get() as u32)}</span>
                    </label>
                    <input type="range" min="14" max="22" step="1"
                        class="setting-slider"
                        prop:value=move || font_size.get().to_string()
                        on:input:target=move |ev| {
                            if let Ok(v) = ev.target().value().parse::<f64>() { set_font_size.set(v); }
                        }
                    />
                </div>

                <div class="setting-group">
                    <label class="setting-label">
                        "Line Height: " <span class="setting-value">{move || format!("{:.1}", line_height.get())}</span>
                    </label>
                    <input type="range" min="1.4" max="2.0" step="0.1"
                        class="setting-slider"
                        prop:value=move || line_height.get().to_string()
                        on:input:target=move |ev| {
                            if let Ok(v) = ev.target().value().parse::<f64>() { set_line_height.set(v); }
                        }
                    />
                </div>

                <div class="setting-group">
                    <label class="setting-label">
                        "Content Width: " <span class="setting-value">{move || format!("{}px", content_width.get() as u32)}</span>
                    </label>
                    <input type="range" min="480" max="900" step="20"
                        class="setting-slider"
                        prop:value=move || content_width.get().to_string()
                        on:input:target=move |ev| {
                            if let Ok(v) = ev.target().value().parse::<f64>() { set_content_width.set(v); }
                        }
                    />
                </div>
            </section>

            // ── Search ──
            <section class="settings-section">
                <h2>"Search"</h2>

                <div class="setting-group">
                    <label class="setting-toggle">
                        <input type="checkbox"
                            prop:checked=show_scores
                            on:change:target=move |ev| {
                                let v = ev.target().checked();
                                set_show_scores.set(v);
                                crate::persistence::save("show-scores", &v);
                            }
                        />
                        " Show similarity scores"
                    </label>
                </div>

                <div class="setting-group">
                    <label class="setting-toggle">
                        <input type="checkbox"
                            prop:checked=show_vector_viz
                            on:change:target=move |ev| {
                                let v = ev.target().checked();
                                set_show_vector_viz.set(v);
                                crate::persistence::save("vector-viz", &v);
                            }
                        />
                        " Vector spectrum visualizer"
                    </label>
                </div>

                <div class="setting-group">
                    <label class="setting-toggle">
                        <input type="checkbox"
                            prop:checked=show_overlay
                            on:change:target=move |ev| {
                                let v = ev.target().checked();
                                set_show_overlay.set(v);
                                crate::persistence::save("overlay", &v);
                            }
                        />
                        " Sentient Overlay (epistemic annotations)"
                    </label>
                </div>
            </section>

            // ── API Keys ──
            <section class="settings-section">
                <h2>"API Keys"</h2>
                <p style="font-size: 13px; color: var(--content-text-secondary); margin-bottom: 12px;">
                    "Optional keys for the Prism comparison view. Stored in your browser only."
                </p>

                <div class="setting-group">
                    <label class="setting-label">"Brave Search API Key"</label>
                    <input type="password"
                        class="setting-input"
                        placeholder="BSA..."
                        prop:value={
                            crate::persistence::load::<String>("brave-api-key").unwrap_or_default()
                        }
                        on:change:target=move |ev| {
                            let v = ev.target().value();
                            crate::persistence::save("brave-api-key", &v);
                        }
                    />
                    <p style="font-size: 11px; color: var(--content-text-secondary); margin-top: 4px;">
                        "Get a key at "
                        <a href="https://api.search.brave.com" target="_blank">"api.search.brave.com"</a>
                    </p>
                </div>

                <div class="setting-group">
                    <label class="setting-label">"Perplexity API Key"</label>
                    <input type="password"
                        class="setting-input"
                        placeholder="pplx-..."
                        prop:value={
                            crate::persistence::load::<String>("perplexity-api-key").unwrap_or_default()
                        }
                        on:change:target=move |ev| {
                            let v = ev.target().value();
                            crate::persistence::save("perplexity-api-key", &v);
                        }
                    />
                    <p style="font-size: 11px; color: var(--content-text-secondary); margin-top: 4px;">
                        "Get a key at "
                        <a href="https://docs.perplexity.ai" target="_blank">"docs.perplexity.ai"</a>
                    </p>
                </div>
            </section>

            // ── About ──
            <section class="settings-section">
                <h2>"About"</h2>
                <div class="about-grid">
                    <div class="about-item"><strong>"Version"</strong><span>"0.2.0"</span></div>
                    <div class="about-item"><strong>"Engine"</strong><span>"BinaryHV 16,384-bit"</span></div>
                    <div class="about-item"><strong>"Claims Indexed"</strong><span>{claim_count.to_string()}</span></div>
                    <div class="about-item"><strong>"Crates"</strong><span>"13"</span></div>
                    <div class="about-item"><strong>"Tests"</strong><span>"176 passing"</span></div>
                    <div class="about-item"><strong>"Language"</strong><span>"Pure Rust"</span></div>
                    <div class="about-item"><strong>"C++ Dependencies"</strong><span>"Zero"</span></div>
                    <div class="about-item"><strong>"JavaScript"</strong><span>"None"</span></div>
                </div>
                <p style="margin-top: 16px; font-size: 13px; color: var(--content-text-secondary);">
                    "Prism is part of the Mycelix ecosystem. "
                    <a href="https://mycelix.net">"mycelix.net"</a>
                </p>
            </section>
        </div>
    }
}
