// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;
use serde::{Deserialize, Serialize};

use crate::persistence;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UserRole {
    Teacher,
    Student,
    Parent,
}

const ROLE_KEY: &str = "praxis_role";

/// Provide the selected role as a context signal.
/// Returns (read, write) signals for the optional role.
///
/// Initializes from localStorage if a previously selected role exists.
/// Wraps the write signal to persist on every change.
pub fn provide_role_context() -> (ReadSignal<Option<UserRole>>, WriteSignal<Option<UserRole>>) {
    // Restore persisted role (if any)
    let initial: Option<UserRole> = persistence::load(ROLE_KEY);
    let (role, set_role) = signal(initial);

    // Persist whenever role changes
    Effect::new(move |_| {
        let current = role.get();
        if let Some(r) = current {
            persistence::save(ROLE_KEY, &r);
        } else {
            persistence::remove(ROLE_KEY);
        }
    });

    provide_context(role);
    provide_context(set_role);
    (role, set_role)
}

/// Read the current role from context (panics if not provided).
pub fn use_role() -> ReadSignal<Option<UserRole>> {
    expect_context::<ReadSignal<Option<UserRole>>>()
}

/// Get the setter for the current role from context.
pub fn use_set_role() -> WriteSignal<Option<UserRole>> {
    expect_context::<WriteSignal<Option<UserRole>>>()
}
