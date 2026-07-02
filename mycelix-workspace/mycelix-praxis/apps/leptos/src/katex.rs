// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! KaTeX math rendering — splits `$...$` delimited strings into text and math
//! segments, renders math via the KaTeX JS library, and returns combined HTML.

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    /// Access the global `katex` object loaded from the CDN.
    #[wasm_bindgen(js_namespace = katex, js_name = renderToString, catch)]
    fn katex_render_to_string(expression: &str, options: &JsValue) -> Result<JsValue, JsValue>;
}

/// Returns `true` if the KaTeX library is loaded in the browser.
fn katex_available() -> bool {
    js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("katex"))
        .map(|v| !v.is_undefined() && !v.is_null())
        .unwrap_or(false)
}

/// Build KaTeX options: `{ throwOnError: false }`.
fn katex_options() -> JsValue {
    let opts = js_sys::Object::new();
    let _ = js_sys::Reflect::set(
        &opts,
        &JsValue::from_str("throwOnError"),
        &JsValue::from_bool(false),
    );
    opts.into()
}

/// Render a string that may contain `$...$` delimited math into HTML.
///
/// Text segments are HTML-escaped. Math segments are rendered via KaTeX.
/// If KaTeX is not loaded, returns the input with basic HTML escaping.
pub fn render_math_html(input: &str) -> String {
    if !katex_available() {
        return html_escape(input);
    }

    let opts = katex_options();
    let mut result = String::new();
    let mut in_math = false;
    let mut segments = input.split('$');

    // The first segment is always text (before any $).
    if let Some(first) = segments.next() {
        result.push_str(&html_escape(first));
    }

    for segment in segments {
        if in_math {
            // Attempt to render math; fall back to escaped text on error.
            match katex_render_to_string(segment, &opts) {
                Ok(html) => {
                    if let Some(s) = html.as_string() {
                        result.push_str(&s);
                    } else {
                        result.push('$');
                        result.push_str(&html_escape(segment));
                        result.push('$');
                    }
                }
                Err(_) => {
                    result.push('$');
                    result.push_str(&html_escape(segment));
                    result.push('$');
                }
            }
        } else {
            result.push_str(&html_escape(segment));
        }
        in_math = !in_math;
    }

    result
}

/// Minimal HTML escaping for text segments.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
