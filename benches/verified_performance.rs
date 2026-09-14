// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Historical `verified_performance` target retirement sentinel.
//!
//! This target previously registered an empty Criterion group, so a successful
//! invocation proved only that a no-op benchmark executable could run. Keep the
//! historical target name resolvable, but fail closed until a real qualified
//! performance suite with explicit measurements and evidence semantics replaces
//! it under a deliberately reviewed theorem.

fn main() {
    eprintln!("RETIRED_EVIDENCE_SURFACE");
    eprintln!("reason=no_registered_performance_measurements");
    eprintln!("replacement=qualified_target_specific_performance_suite_required");
    std::process::exit(2);
}
