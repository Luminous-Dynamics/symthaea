// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Diagnostic probe for the empty-input hang (E2 finding, 2026-07-08):
//! a 500-cycle empty-string regime ran >24h at 97.7% CPU while all other
//! regimes finished in minutes. A single empty cycle terminates fine
//! (`empty_input_does_not_panic`), so the pathology is state-dependent.
//!
//! This probe mimics E2's shape (varied warmup, then empty cycles) and prints
//! per-cycle wall time so the failure profile is observable:
//! - progressively slower cycles → unbounded growth + superlinear scan
//! - one cycle that never returns → genuine infinite loop (attach gdb)
//!
//! Run: cargo run --release --example probe_empty_hang [-- <empty_cycles> [warmup_cycles]]
//!
//! FINDING (2026-07-16, 40-cycle warmup): 600 empty cycles complete at ~50ms
//! each (two spikes: 9.2s at cycle 6, 632ms at 100) — no hang at small scale.
//! Yesterday's full E2 runs (1500 varied preamble) still stalled in the empty
//! regime post-state-fix, so the pathology needs accumulated state from a
//! long varied preamble: use warmup_cycles=1500 to reproduce at true scale.

use std::time::Instant;
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

fn main() {
    let empty_cycles: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(600);
    let warmup_cycles: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);

    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).expect("service");

    let warmup = [
        "the quick brown fox jumps over the lazy dog",
        "what is the capital of France?",
        "URGENT: fire detected in the server room!",
        "the system hums quietly in the background",
        "energy prices rose sharply across the region today",
        "she planted tomatoes and basil along the south fence",
        "the committee voted to postpone the decision by a week",
        "rainfall exceeded seasonal averages in three provinces",
        "the melody resolved unexpectedly to a minor chord",
        "consider the trolley problem from a deontological view",
        "2 + 2 = 4 and the square root of 81 is 9",
        "the reactor coolant loop operates at 15 megapascals",
    ];
    println!("warmup: {warmup_cycles} varied cycles");
    for i in 0..warmup_cycles {
        let t = Instant::now();
        let _ = svc.cycle(warmup[i % warmup.len()]);
        let ms = t.elapsed().as_millis();
        if i % 100 == 0 || ms > 2000 {
            println!("  warmup {i}: {ms} ms");
        }
    }

    println!("empty regime: {empty_cycles} cycles of \"\"");
    let mut slowest: u128 = 0;
    for i in 0..empty_cycles {
        let t = Instant::now();
        let _ = svc.cycle("");
        let ms = t.elapsed().as_millis();
        slowest = slowest.max(ms);
        // Print every cycle's timing for the first 20, then every 20th, plus
        // any cycle that is anomalously slow. Line-buffered stdout means the
        // last printed line before a hang identifies the hanging cycle index.
        if i < 20 || i % 20 == 0 || ms > 4 * slowest.max(1) / 3 {
            println!("  empty {i}: {ms} ms");
        }
    }
    println!("COMPLETED: all {empty_cycles} empty cycles terminated (slowest {slowest} ms)");
}
