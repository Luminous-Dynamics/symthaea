// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Observable fail-stop health for mutable store handles.

/// Whether a mutable store handle can safely issue more writes.
///
/// A handle becomes poisoned when an operation has already changed mapped
/// bytes and then fails before the corresponding durable publication boundary
/// completes. Reads remain available for diagnostics, but the handle must be
/// dropped and the store reopened (normally through recovery) before writing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreHealth {
    /// No uncertain partial mutation has been observed by this handle.
    Healthy,
    /// A mutation crossed its first irreversible byte write and then failed.
    Poisoned {
        /// Operation that entered the uncertain state.
        operation: &'static str,
        /// Original failure rendered for operator diagnostics.
        cause: String,
    },
}

impl StoreHealth {
    /// Whether this handle can safely issue another mutation.
    pub const fn is_healthy(&self) -> bool {
        matches!(self, Self::Healthy)
    }

    /// Operation that poisoned the handle, when applicable.
    pub const fn poisoned_operation(&self) -> Option<&'static str> {
        match self {
            Self::Healthy => None,
            Self::Poisoned { operation, .. } => Some(*operation),
        }
    }

    /// Human-readable cause retained when the handle was poisoned.
    pub fn poisoned_cause(&self) -> Option<&str> {
        match self {
            Self::Healthy => None,
            Self::Poisoned { cause, .. } => Some(cause),
        }
    }
}

impl Default for StoreHealth {
    fn default() -> Self {
        Self::Healthy
    }
}
