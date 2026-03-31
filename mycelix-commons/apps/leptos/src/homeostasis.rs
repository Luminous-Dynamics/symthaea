// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;

#[derive(Clone)]
pub struct HomeostasisState { pub pending_needs: ReadSignal<u32>, pub in_homeostasis: ReadSignal<bool> }

pub fn provide_homeostasis_context() -> HomeostasisState {
    let (pending, set_pending) = signal(0u32);
    let (in_home, set_in_home) = signal(false);
    Effect::new(move |_| { set_in_home.set(pending.get() == 0); });
    provide_context(set_pending);
    let state = HomeostasisState { pending_needs: pending, in_homeostasis: in_home };
    provide_context(state.clone());
    state
}
