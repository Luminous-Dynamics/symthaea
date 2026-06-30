// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! WASM entry points for browser-side physics discovery.
//!
//! ```javascript
//! import init, { search_catalog, classify_physics_claim, catalog_size } from './pkg/symthaea_physics_catalog.js';
//! await init();
//! console.log("Equations:", catalog_size());
//! const results = search_catalog("Yukawa potential", 5);
//! const parsed = JSON.parse(results);
//! ```

use crate::catalog::PhysicsCatalog;
use crate::search;
use std::sync::OnceLock;
use wasm_bindgen::prelude::*;

/// Global catalog instance (lazily initialized, thread-safe).
static CATALOG: OnceLock<PhysicsCatalog> = OnceLock::new();

fn get_catalog() -> &'static PhysicsCatalog {
    CATALOG.get_or_init(PhysicsCatalog::new)
}

/// Initialize the catalog (call once on page load).
#[wasm_bindgen]
pub fn init_catalog() -> usize {
    let catalog = get_catalog();
    catalog.len()
}

/// Number of equations in the catalog.
#[wasm_bindgen]
pub fn catalog_size() -> usize {
    get_catalog().len()
}

/// Search the catalog by query text. Returns JSON array of results.
#[wasm_bindgen]
pub fn search_catalog(query: &str, max_results: usize) -> String {
    let catalog = get_catalog();
    let results = search::search_by_text(catalog, query, max_results);
    serde_json::to_string(&results).unwrap_or_else(|_| "[]".to_string())
}

/// List all domains with equation counts. Returns JSON.
#[wasm_bindgen]
pub fn list_domains() -> String {
    use std::collections::HashMap;
    let catalog = get_catalog();
    let mut counts: HashMap<String, usize> = HashMap::new();
    for entry in catalog.entries() {
        *counts.entry(entry.domain.label().to_string()).or_insert(0) += 1;
    }
    serde_json::to_string(&counts).unwrap_or_else(|_| "{}".to_string())
}

/// Get all equations as JSON (for catalog browser).
#[wasm_bindgen]
pub fn all_equations() -> String {
    let catalog = get_catalog();
    let eqs: Vec<_> = catalog
        .entries()
        .iter()
        .map(|e| {
            serde_json::json!({
                "name": e.name,
                "domain": e.domain.label(),
                "latex": e.latex,
            })
        })
        .collect();
    serde_json::to_string(&eqs).unwrap_or_else(|_| "[]".to_string())
}

/// Classify a physics claim. Returns JSON LEM classification.
#[wasm_bindgen]
pub fn classify_physics_claim(description: &str, confidence: f64, is_open_source: bool) -> String {
    let catalog = get_catalog();
    let results = search::search_by_text(catalog, description, 5);
    let discovery = crate::discovery::classify_claim(
        description,
        crate::catalog::Domain::ModifiedGravity, // Default for unknown claims
        confidence,
        &results,
        is_open_source,
    );
    serde_json::to_string(&discovery).unwrap_or_else(|_| "{}".to_string())
}
