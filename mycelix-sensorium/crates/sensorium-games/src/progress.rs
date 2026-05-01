// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Progress tracking for educational games.
//! Compatible with Praxis curriculum system.

use leptos::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum ProgressStatus {
    #[default]
    NotStarted,
    Studying,
    Mastered,
}

#[derive(Clone, Debug, Default)]
pub struct NodeProgress {
    pub mastery_permille: u32,
    pub status: ProgressStatus,
    pub attempts: u32,
    pub correct: u32,
}

#[derive(Clone, Debug, Default)]
pub struct ProgressStore {
    pub nodes: HashMap<String, NodeProgress>,
}

/// Get the writable progress store. Games call `set_progress.update(|p| ...)`.
pub fn use_set_progress() -> WriteSignal<ProgressStore> {
    use_context::<WriteSignal<ProgressStore>>().unwrap_or_else(|| {
        let (_, setter) = signal(ProgressStore::default());
        setter
    })
}

/// Provide the progress store in context. Call from a parent component.
pub fn provide_progress_store() -> (ReadSignal<ProgressStore>, WriteSignal<ProgressStore>) {
    let (read, write) = signal(ProgressStore::default());
    provide_context(write);
    (read, write)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_store_default_empty() {
        let store = ProgressStore::default();
        assert!(store.nodes.is_empty());
    }

    #[test]
    fn node_progress_default_values() {
        let np = NodeProgress::default();
        assert_eq!(np.status, ProgressStatus::NotStarted);
        assert_eq!(np.mastery_permille, 0);
        assert_eq!(np.attempts, 0);
    }
}
