// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Homeostasis detection framework.
//!
//! When all pending work is resolved, the UI enters homeostasis. The chrome
//! fades and the organism rewards stillness. Being idle regenerates Phi.
//!
//! Apps configure the number of pending-count signals they need:
//! - Governance: pending_proposals + pending_votes
//! - Hearth: pending_care_tasks + pending_decisions

use crate::util::set_css_var;
use leptos::prelude::*;

/// Reactive state for homeostasis detection.
#[derive(Clone)]
pub struct HomeostasisState {
    /// Whether the UI is in homeostatic void mode.
    pub in_homeostasis: ReadSignal<bool>,
    /// Read signals for each pending counter.
    pub pending_counts: Vec<ReadSignal<u32>>,
}

/// A write handle for a specific pending counter.
/// Exposed as context so pages can update their counters.
#[derive(Clone)]
pub struct PendingCountWriter {
    pub index: usize,
    pub writer: WriteSignal<u32>,
}

/// Initialize homeostasis detection with `n` pending-count signals.
///
/// Returns the HomeostasisState. Each pending counter's WriteSignal is
/// provided as a `PendingCountWriter` context (indexed 0..n) so pages
/// can update counts.
///
/// `css_var_name` is the CSS custom property to set (e.g. "--homeostasis").
pub fn provide_homeostasis_context(
    num_counters: usize,
    css_var_name: &'static str,
) -> HomeostasisState {
    let mut read_signals = Vec::with_capacity(num_counters);
    let mut write_signals = Vec::with_capacity(num_counters);

    for _ in 0..num_counters {
        let (r, w) = signal(0u32);
        read_signals.push(r);
        write_signals.push(w);
    }

    let (in_homeostasis, set_in_homeostasis) = signal(false);

    // Derive homeostasis from all counters
    let reads = read_signals.clone();
    Effect::new(move |_| {
        let all_zero = reads.iter().all(|r| r.get() == 0);
        set_in_homeostasis.set(all_zero);
        set_css_var(css_var_name, if all_zero { "1" } else { "0" });
    });

    // Provide write signals as context
    for (i, w) in write_signals.into_iter().enumerate() {
        provide_context(PendingCountWriter {
            index: i,
            writer: w,
        });
    }

    let state = HomeostasisState {
        in_homeostasis,
        pending_counts: read_signals,
    };

    provide_context(state.clone());
    state
}

pub fn use_homeostasis() -> HomeostasisState {
    expect_context::<HomeostasisState>()
}
