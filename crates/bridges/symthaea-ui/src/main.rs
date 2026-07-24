// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Unified Symthaea web UI — SYMTHAEA_UNIFIED_UI_PLAN_2026-07-10.md Phase 3.
//!
//! Talks to a `symthaea-service` HTTP gateway (Phases 1+2: POST
//! /v1/service, GET /v1/ws/live) — never to the `symthaea` crate directly,
//! so this stays a small, fast WASM build independent of the daemon's own
//! compile graph.

mod api;
mod app;

fn main() {
    leptos::mount::mount_to_body(app::App);
}
