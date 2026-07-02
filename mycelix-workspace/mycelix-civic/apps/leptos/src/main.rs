// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Mycelix Civic — Leptos CSR frontend.
//!
//! Greenfield scaffold (2026-04-18). Civic owns justice, emergency coordination,
//! and media commons across 20 zomes + 2,276 tests, but has had zero UI until
//! now. This scaffold lands the minimum-viable greenfield using the full
//! mycelix-leptos-core primitive set from day 1 — so this cluster becomes
//! the reference adoption for the Type 1 civ substrate design language.
//!
//! Current scope: renders `<Showcase />` as the landing page, demonstrating
//! that the Trunk build + semantic-color tokens + primitives work end-to-end.
//! Subsequent commits split this into real civic pages (justice docket,
//! emergency coordination, media verification) each using the primitives
//! to surface live data.

use leptos::prelude::*;
use mycelix_leptos_core::Showcase;

fn main() {
    console_error_panic_hook::set_once();
    mount_to_body(App);
}

#[component]
fn App() -> impl IntoView {
    view! {
        // Root-level landing: the aesthetic showcase. This proves:
        //   - mycelix-leptos-core wires up cleanly
        //   - civic's semantic-color tokens re-skin the primitives
        //   - the Type 1 civ substrate aesthetic renders end-to-end
        //
        // Replace with a real civic router + pages once the shell is validated.
        <Showcase />
    }
}
