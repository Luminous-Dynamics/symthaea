// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Shared DOM utilities.

use wasm_bindgen::JsCast;

/// Set a CSS custom property on the document root element.
pub fn set_css_var(name: &str, value: &str) {
    if let Some(window) = web_sys::window() {
        if let Some(document) = window.document() {
            if let Some(root) = document.document_element() {
                if let Some(el) = root.dyn_ref::<web_sys::HtmlElement>() {
                    let _ = el.style().set_property(name, value);
                }
            }
        }
    }
}

/// Set an attribute on the document root element.
pub fn set_root_attribute(name: &str, value: &str) {
    if let Some(window) = web_sys::window() {
        if let Some(document) = window.document() {
            if let Some(root) = document.document_element() {
                let _ = root.set_attribute(name, value);
            }
        }
    }
}
