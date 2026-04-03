// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;
use crate::identity_context::use_identity;

#[component]
pub fn RecoveryPage() -> impl IntoView {
    let _ctx = use_identity();
    view! {
        <div class="page page-recovery">
            <h1>"Social Recovery"</h1>
            <p class="page-subtitle">"Coming soon"</p>
        </div>
    }
}
