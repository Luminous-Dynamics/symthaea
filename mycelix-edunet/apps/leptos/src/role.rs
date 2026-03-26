// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UserRole {
    Teacher,
    Student,
    Parent,
}

/// Provide the selected role as a context signal.
/// Returns (read, write) signals for the optional role.
pub fn provide_role_context() -> (ReadSignal<Option<UserRole>>, WriteSignal<Option<UserRole>>) {
    let (role, set_role) = signal(None::<UserRole>);
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
