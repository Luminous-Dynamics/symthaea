// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use leptos::prelude::*;

use crate::pages::install::InstallPage;

#[component]
pub fn App() -> impl IntoView {
    view! {
        <main style="display: block;">
            <InstallPage />
        </main>
    }
}
