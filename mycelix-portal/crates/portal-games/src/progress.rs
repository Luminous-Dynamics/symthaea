// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Progress tracking interface for educational games.
//!
//! Games call `use_set_progress()` when a student completes a challenge.
//! The portal (or EduNet) provides the actual implementation via context.

use leptos::prelude::*;

/// Progress status for a curriculum node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProgressStatus {
    NotStarted,
    Studying,
    Mastered,
}

/// Set progress for a curriculum node. Retrieved from Leptos context.
///
/// In the portal, this updates the domain-edunet state store.
/// In standalone EduNet, this updates the curriculum signals.
/// If no provider is in context, this is a no-op.
pub fn use_set_progress() -> impl Fn(&str, u32, ProgressStatus) {
    // Try to get a progress setter from context; no-op if absent
    let setter = use_context::<WriteSignal<Vec<(String, u32, ProgressStatus)>>>();
    move |node_id: &str, mastery: u32, status: ProgressStatus| {
        if let Some(set) = setter {
            set.update(|v| v.push((node_id.to_string(), mastery, status)));
        }
    }
}
