// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! AI accessibility layer for the portal.
//!
//! Makes the portal interactable by AI agents (Claude, GPT, Playwright bots)
//! through three mechanisms:
//!
//! 1. **State JSON** — A `<script type="application/json" id="portal-state">`
//!    element containing the current domain state as machine-readable JSON.
//!    AI agents can read this to understand what's on screen.
//!
//! 2. **Data attributes** — `data-action`, `data-domain`, `data-id` on interactive
//!    elements so AI agents can find and click them programmatically.
//!
//! 3. **ARIA labels** — Screen reader AND AI agent accessibility.
//!
//! # Usage for AI Agents
//!
//! ```javascript
//! // Read current state
//! const state = JSON.parse(document.getElementById('portal-state').textContent);
//! console.log(state.domain, state.proposals?.length);
//!
//! // Click a specific proposal
//! document.querySelector('[data-action="select-proposal"][data-id="MIP-042"]').click();
//!
//! // Cast a vote
//! document.querySelector('[data-action="vote-for"]').click();
//!
//! // Create a proposal
//! document.querySelector('[data-action="create-proposal"]').click();
//! document.querySelector('[data-field="title"]').value = "My proposal";
//! document.querySelector('[data-field="description"]').value = "Description";
//! document.querySelector('[data-action="submit"]').click();
//! ```

use leptos::prelude::*;

/// Renders a machine-readable JSON state block for AI agents.
///
/// Place this inside each domain page. The JSON is updated reactively
/// whenever the domain state changes.
#[component]
pub fn AiState(
    /// JSON string representing the current domain state.
    json: String,
) -> impl IntoView {
    view! {
        <script type="application/json" id="portal-state" inner_html=json />
    }
}

/// Returns a formatted JSON string for the AI state block.
pub fn ai_state_json(domain: &str, data: &serde_json::Value) -> String {
    serde_json::to_string(&serde_json::json!({
        "domain": domain,
        "timestamp": js_sys::Date::now() as u64,
        "data": data,
    }))
    .unwrap_or_else(|_| "{}".into())
}
