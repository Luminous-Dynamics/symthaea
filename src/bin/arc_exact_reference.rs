// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Runnable target for the unchanged RQ-003 exact-output ARC reference lane.
//!
//! The source theorem remains in `examples/benchmark_arc_exact_output.rs`; this wrapper exists
//! because the root manifest uses `autoexamples = false` while automatic `src/bin` discovery is
//! enabled. Keeping the original file unchanged preserves its RQ-003 evidence identity.

include!("../../examples/benchmark_arc_exact_output.rs");
