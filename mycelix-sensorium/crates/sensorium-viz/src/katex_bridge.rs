// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! KaTeX bridge — renders LaTeX math via wasm-bindgen call to KaTeX.js.
//!
//! KaTeX.js (28KB) must be loaded via `<script>` in index.html:
//! ```html
//! <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.28/dist/katex.min.css">
//! <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.28/dist/katex.min.js"></script>
//! ```
//!
//! This bridge calls `katex.renderToString()` from Rust and injects the
//! resulting HTML into a Leptos component.

use leptos::prelude::*;
use wasm_bindgen::prelude::*;

/// Call `katex.renderToString(expr, opts)` from JavaScript.
fn render_katex(expr: &str, display_mode: bool) -> Result<String, String> {
    let window = web_sys::window().ok_or("no window")?;
    let katex_obj = js_sys::Reflect::get(&window, &JsValue::from_str("katex"))
        .map_err(|_| "KaTeX not loaded — add <script> tag to index.html")?;

    if katex_obj.is_undefined() {
        return Err("KaTeX not loaded — add <script> tag to index.html".into());
    }

    let render_fn = js_sys::Reflect::get(&katex_obj, &JsValue::from_str("renderToString"))
        .map_err(|_| "katex.renderToString not found")?;

    let render_fn: js_sys::Function = render_fn.dyn_into()
        .map_err(|_| "katex.renderToString is not a function")?;

    let opts = js_sys::Object::new();
    js_sys::Reflect::set(&opts, &"throwOnError".into(), &JsValue::FALSE).ok();
    js_sys::Reflect::set(&opts, &"displayMode".into(), &JsValue::from_bool(display_mode)).ok();

    let result = render_fn.call2(&katex_obj, &JsValue::from_str(expr), &opts.into())
        .map_err(|e| format!("KaTeX render error: {:?}", e))?;

    result.as_string().ok_or_else(|| "KaTeX returned non-string".into())
}

/// Leptos component for rendering LaTeX math expressions.
///
/// Calls KaTeX.js via wasm-bindgen to render the expression to HTML,
/// then injects it via `inner_html`. Falls back to raw expression on error.
///
/// # Props
/// - `expr` — LaTeX expression (e.g., `"\\frac{a}{b}"`)
/// - `display` — If true, renders in display mode (centered block). Default: false (inline).
///
/// # Example
/// ```rust,no_run
/// use sensorium_viz::Math;
/// // view! { <Math expr="E = mc^2" /> }
/// // view! { <Math expr="\\int_0^\\infty e^{-x} dx = 1" display=true /> }
/// ```
#[component]
pub fn Math(
    expr: String,
    #[prop(default = false)] display: bool,
) -> impl IntoView {
    let html = render_katex(&expr, display)
        .unwrap_or_else(|_| format!("<code>{}</code>", expr));

    if display {
        view! { <div class="katex-display" inner_html=html /> }.into_any()
    } else {
        view! { <span class="katex-inline" inner_html=html /> }.into_any()
    }
}
