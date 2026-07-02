// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! KaTeX equation rendering component.
//!
//! Renders LaTeX math notation via the KaTeX JavaScript library (loaded via CDN).
//! Falls back to raw LaTeX string if KaTeX is not yet loaded.

use leptos::prelude::*;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = katex, js_name = renderToString, catch)]
    fn katex_render(latex: &str, options: &JsValue) -> Result<String, JsValue>;
}

/// Render a LaTeX equation using KaTeX.
///
/// If KaTeX is not loaded yet (CDN hasn't finished), falls back to the raw string.
fn render_latex(latex: &str) -> String {
    let options = js_sys::Object::new();
    js_sys::Reflect::set(&options, &"throwOnError".into(), &false.into()).ok();
    js_sys::Reflect::set(&options, &"displayMode".into(), &false.into()).ok();

    match katex_render(latex, &options.into()) {
        Ok(html) => html,
        Err(_) => format!("<code>{}</code>", latex),
    }
}

/// Inline equation component (renders KaTeX).
#[component]
pub fn Equation(
    /// LaTeX source (e.g., r"E = mc^2")
    #[prop(into)]
    latex: String,
) -> impl IntoView {
    let html = render_latex(&latex);
    view! {
        <span class="equation" inner_html=html></span>
    }
}

/// Display-mode equation (centered, larger).
#[component]
pub fn DisplayEquation(
    #[prop(into)]
    latex: String,
) -> impl IntoView {
    let options = js_sys::Object::new();
    js_sys::Reflect::set(&options, &"throwOnError".into(), &false.into()).ok();
    js_sys::Reflect::set(&options, &"displayMode".into(), &true.into()).ok();

    let html = match katex_render(&latex, &options.into()) {
        Ok(h) => h,
        Err(_) => format!("<pre>{}</pre>", latex),
    };

    view! {
        <div class="display-equation" style="text-align: center; margin: 1rem 0;" inner_html=html></div>
    }
}
